use crate::compaction::{
    compact_session_context, compacted_history_message, precheck_session_context_compaction,
};
use crate::context::{
    build_agents_instructions_message, build_environment_context, build_skills_system_message,
    push_turn_records_as_context_messages,
};
use crate::events::spawn_session_agent_event_bridge;
use crate::title::generate_session_title;
use crate::{
    AppKernelConfig, AppServices, CreateSessionRequest, OpenWorkspaceRequest, SessionTitleChanged,
    StartedAgentTask, WorkspaceWithSessions,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use seekcode_agent_core::{
    Agent, AgentContextCompactionOutcome, AgentContextPrecheck, AgentContextPreparer,
    AgentHistoryMessage, AgentTaskContext, AgentToolContext, PreparedAgentContext,
    StartTaskRequest,
};
use seekcode_common::{
    init_tracing, ChatMessage, ChatRole, MessageId, SeekCodeError, SeekCodeResult, SessionId,
    TaskId, ToolCallId, WorkspaceId,
};
use seekcode_deepseek_client::{DeepSeekClient, DeepSeekConfig, ModelProvider};
use seekcode_policy::{AutonomousPolicy, PolicyEngine};
use seekcode_secrets::{InMemorySecretStore, SecretStore};
use seekcode_shell_sandbox::CommandRunner;
use seekcode_storage::{
    local_now_text, NewModelCallLog, NewSession, NewSessionMessage, NewWorkspace,
    SessionMessageRecord, SessionModelCallStats, SessionRecord, Storage, WorkspaceRecord,
};
use seekcode_tool_system::{system_tool_registry, SystemToolConfig, ToolRegistry};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Session service owns persisted conversation state and session-scoped agent events.
pub struct SessionService {
    storage: Option<Arc<dyn Storage>>,
    running_sessions: RwLock<HashSet<SessionId>>,
    title_sessions: RwLock<HashSet<SessionId>>,
}

/// Application kernel exposed to thin adapters.
pub struct AppKernel {
    config: RwLock<AppKernelConfig>,
    services: AppServices,
}

impl SessionService {
    fn new(storage: Option<Arc<dyn Storage>>) -> Self {
        Self {
            storage,
            running_sessions: RwLock::new(HashSet::new()),
            title_sessions: RwLock::new(HashSet::new()),
        }
    }

    fn start_running_task(&self, session_id: SessionId) -> SeekCodeResult<()> {
        let mut running_sessions = self.running_sessions.write();
        if !running_sessions.insert(session_id) {
            return Err(SeekCodeError::Validation(
                "session already has a running agent task".to_string(),
            ));
        }

        Ok(())
    }

    pub(crate) fn finish_running_task(&self, session_id: SessionId) {
        self.running_sessions.write().remove(&session_id);
    }

    fn start_title_task(&self, session_id: SessionId) -> bool {
        self.title_sessions.write().insert(session_id)
    }

    fn finish_title_task(&self, session_id: SessionId) {
        self.title_sessions.write().remove(&session_id);
    }

    async fn append_user_prompt(&self, session_id: SessionId, prompt: &str) -> SeekCodeResult<i64> {
        let turn_sequence = self
            .storage()?
            .next_session_turn_sequence(session_id)
            .await?;

        self.storage()?
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence,
                role: ChatRole::User,
                content: prompt.to_string(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                created_at: local_now_text(),
            })
            .await?;

        Ok(turn_sequence)
    }

    pub(crate) async fn persist_assistant_choice(
        &self,
        session_id: SessionId,
        turn_sequence: i64,
        choice: &seekcode_deepseek_client::ChatChoiceChunk,
    ) {
        let content = choice.delta.content.clone().unwrap_or_default();
        let reasoning_content = choice.delta.reasoning_content.clone();
        let tool_calls = choice_tool_calls_json(choice);
        tracing::debug!(
            target: "seekcode_app_kernel::session_messages",
            %session_id,
            turn_sequence,
            finish_reason = ?choice.finish_reason,
            content_len = content.len(),
            reasoning_len = reasoning_content.as_deref().map(str::len).unwrap_or(0),
            tool_call_count = tool_calls.len(),
            tool_calls = %serde_json::Value::Array(tool_calls.clone()),
            "persisting assistant choice delta"
        );

        if content.is_empty() && reasoning_content.is_none() && tool_calls.is_empty() {
            return;
        }

        let storage = match self.storage() {
            Ok(storage) => storage,
            Err(error) => {
                tracing::warn!(%error, "failed to persist assistant delta");
                return;
            }
        };

        if let Err(error) = storage
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence,
                role: ChatRole::Assistant,
                content,
                reasoning_content,
                tool_calls,
                tool_call_id: None,
                created_at: local_now_text(),
            })
            .await
        {
            tracing::warn!(%error, "failed to persist assistant delta");
        }
    }

    pub(crate) async fn persist_tool_result(
        &self,
        session_id: SessionId,
        turn_sequence: i64,
        tool_call_id: ToolCallId,
        output: Option<serde_json::Value>,
        error: Option<String>,
    ) {
        let content = match (output, error) {
            (Some(output), _) => serde_json::to_string(&output).unwrap_or_default(),
            (None, Some(error)) => error,
            (None, None) => String::new(),
        };

        let storage = match self.storage() {
            Ok(storage) => storage,
            Err(error) => {
                tracing::warn!(%error, "failed to persist tool result");
                return;
            }
        };

        if let Err(error) = storage
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence,
                role: ChatRole::Tool,
                content,
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: Some(tool_call_id),
                created_at: local_now_text(),
            })
            .await
        {
            tracing::warn!(%error, "failed to persist tool result");
        }
    }

    #[cfg(test)]
    async fn assemble_context(
        &self,
        session_id: SessionId,
        prompt: &str,
    ) -> SeekCodeResult<Vec<ChatMessage>> {
        self.assemble_task_context_before_turn(session_id, prompt, None)
            .await
            .map(|context| context.messages())
    }

    async fn assemble_task_context_excluding_turn_from(
        &self,
        session_id: SessionId,
        prompt: &str,
        excluded_turn_sequence: i64,
    ) -> SeekCodeResult<AgentTaskContext> {
        self.assemble_task_context_before_turn(session_id, prompt, Some(excluded_turn_sequence))
            .await
    }

    async fn assemble_task_context_before_turn(
        &self,
        session_id: SessionId,
        prompt: &str,
        before_turn_sequence: Option<i64>,
    ) -> SeekCodeResult<AgentTaskContext> {
        let storage = self.storage()?;
        let session = storage.get_session(session_id).await?;
        let workspace = storage.get_workspace(session.workspace_id).await?;
        let context_state = storage.get_session_context_state(session_id).await?;
        // Turns at or below this boundary are represented by the summary instead
        // of their original messages.
        let compacted_through_turn = context_state
            .as_ref()
            .map(|state| state.compacted_through_turn)
            .unwrap_or(0);
        let records = storage
            .list_session_messages_in_turn_range(
                session_id,
                compacted_through_turn,
                before_turn_sequence,
            )
            .await?;
        let last_input_tokens = session.last_input_tokens;
        let mut system_prompt = Vec::new();
        if let Some(skills_message) = build_skills_system_message() {
            system_prompt.push(ChatMessage::new(ChatRole::System, skills_message));
        }
        system_prompt.push(ChatMessage::new(
            ChatRole::System,
                "You are SeekCode, a coding agent based on DeepSeek. You and the user share the same workspace and collaborate to achieve the user's goals.\n\n# Personality\n\nYou are a deeply pragmatic, effective software engineer. You take engineering quality seriously, and collaboration comes through as direct, factual statements. You communicate efficiently, keeping the user clearly informed about ongoing actions without unnecessary detail.\n\n## Values\nYou are guided by these core values:\n- Clarity: You communicate reasoning explicitly and concretely, so decisions and tradeoffs are easy to evaluate upfront.\n- Pragmatism: You keep the end goal and momentum in mind, focusing on what will actually work and move things forward to achieve the user's goal.\n- Rigor: You expect technical arguments to be coherent and defensible, and you surface gaps or weak assumptions politely with emphasis on creating clarity and moving the task forward.\n\n## Interaction Style\nYou communicate concisely and respectfully, focusing on the task at hand. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.\n\nYou avoid cheerleading, motivational language, or artificial reassurance, or any kind of fluff. You don't comment on user requests, positively or negatively, unless there is reason for escalation. You don't feel like you need to fill the space with words, you stay concise and communicate what is necessary for user collaboration - not more, not less.\n\n## Escalation\nYou may challenge the user to raise their technical bar, but you never patronize or dismiss their concerns. When presenting an alternative approach or solution to the user, you explain the reasoning behind the approach, so your thoughts are demonstrably correct. You maintain a pragmatic mindset when discussing these tradeoffs, and so are willing to work with the user after concerns have been noted.\n\n\n# General\nAs an expert coding agent, your primary focus is writing code, answering questions, and helping the user complete their task in the current environment. You build context by examining the codebase first without making assumptions or jumping to conclusions. You think through the nuances of the code you encounter, and embody the mentality of a skilled senior software engineer.\n\n- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)\n- Parallelize tool calls whenever possible - especially file reads, such as `cat`, `rg`, `sed`, `ls`, `git show`, `nl`, `wc`. Use `multi_tool_use.parallel` to parallelize tool calls and only this. Never chain together bash commands with separators like `echo \"====\";` as this renders to the user poorly.\n\n## Editing constraints\n\n- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.\n- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like \"Assigns the value to the variable\", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.\n- Use the write_file tool to create or edit files. Do not use cat or shell redirection to create or edit files.\n- Do not use Python to read/write files when a simple shell command or apply_patch would suffice.\n- You may be in a dirty git worktree.\n  * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.\n  * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.\n  * If the changes are in files you've touched recently, you should read carefully and understand how you can work with the changes rather than reverting them.\n  * If the changes are in unrelated files, just ignore them and don't revert them.\n- Do not amend a commit unless explicitly requested to do so.\n- While you are working, you might notice unexpected changes that you didn't make. It's likely the user made them, or were autogenerated. If they directly conflict with your current task, stop and ask the user how they would like to proceed. Otherwise, focus on the task at hand.\n- **NEVER** use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.\n- You struggle using the git interactive console. **ALWAYS** prefer using non-interactive git commands.\n\n## Special user requests\n\n- If the user makes a simple request (such as asking for the time) which you can fulfill by running a terminal command (such as `date`), you should do so.\n- If the user asks for a \"review\", default to a code review mindset: prioritise identifying bugs, risks, behavioural regressions, and missing tests. Findings must be the primary focus of the response - keep summaries or overviews brief and only after enumerating the issues. Present findings first (ordered by severity with file/line references), follow with open questions or assumptions, and offer a change-summary only as a secondary detail. If no findings are discovered, state that explicitly and mention any residual risks or testing gaps.\n\n## Autonomy and persistence\nPersist until the task is fully handled end-to-end within the current turn whenever feasible: do not stop at analysis or partial fixes; carry changes through implementation, verification, and a clear explanation of outcomes unless the user explicitly pauses or redirects you.\n\nUnless the user explicitly asks for a plan, asks a question about the code, is brainstorming potential solutions, or some other intent that makes it clear that code should not be written, assume the user wants you to make code changes or run tools to solve the user's problem. In these cases, it's bad to output your proposed solution in a message, you should go ahead and actually implement the change. If you encounter challenges or blockers, you should attempt to resolve them yourself.\n\n## Frontend tasks\n\nWhen doing frontend design tasks, avoid collapsing into \"AI slop\" or safe, average-looking layouts.\nAim for interfaces that feel intentional, bold, and a bit surprising.\n- Typography: Use expressive, purposeful fonts and avoid default stacks (Inter, Roboto, Arial, system).\n- Color & Look: Choose a clear visual direction; define CSS variables; avoid purple-on-white defaults. No purple bias or dark mode bias.\n- Motion: Use a few meaningful animations (page-load, staggered reveals) instead of generic micro-motions.\n- Background: Don't rely on flat, single-color backgrounds; use gradients, shapes, or subtle patterns to build atmosphere.\n- Ensure the page loads properly on both desktop and mobile\n- For React code, prefer modern patterns including useEffectEvent, startTransition, and useDeferredValue when appropriate if used by the team. Do not add useMemo/useCallback by default unless already used; follow the repo's React Compiler guidance.\n- Overall: Avoid boilerplate layouts and interchangeable UI patterns. Vary themes, type families, and visual languages across outputs.\n\nException: If working within an existing website or design system, preserve the established patterns, structure, and visual language.\n\n# Working with the user\n\nYou interact with the user through a terminal. You have 2 ways of communicating with the users:\n- Share intermediary updates in `commentary` channel. \n- After you have completed all your work, send a message to the `final` channel.\nYou are producing plain text that will later be styled by the program you run in. Formatting should make results easy to scan, but not feel mechanical. Use judgment to decide how much structure adds value. Follow the formatting rules exactly.\n\n## Formatting rules\n\n- You may format with GitHub-flavored Markdown.\n- Structure your answer if necessary, the complexity of the answer should match the task. If the task is simple, your answer should be a one-liner. Order sections from general to specific to supporting.\n- Never use nested bullets. Keep lists flat (single level). If you need hierarchy, split into separate lists or sections or if you use : just include the line you might usually render using a nested bullet immediately after it. For numbered lists, only use the `1. 2. 3.` style markers (with a period), never `1)`.\n- Headers are optional, only use them when you think they are necessary. If you do use them, use short Title Case (1-3 words) wrapped in **…**. Don't add a blank line.\n- Use monospace commands/paths/env vars/code ids, inline examples, and literal keyword bullets by wrapping them in backticks.\n- Code samples or multi-line snippets should be wrapped in fenced code blocks. Include an info string as often as possible.\n- File References: When referencing files in your response follow the below rules:\n  * Use markdown links (not inline code) for clickable file paths.\n  * Each reference should have a stand alone path. Even if it's the same file.\n  * For clickable/openable file references, the path target must be an absolute filesystem path. Labels may be short (for example, `[app.ts](/abs/path/app.ts)`).\n  * Optionally include line/column (1‑based): :line[:column] or #Lline[Ccolumn] (column defaults to 1).\n  * Do not use URIs like file://, vscode://, or https://.\n  * Do not provide range of lines\n- Don’t use emojis or em dashes unless explicitly instructed.\n\n## Final answer instructions\n\n- Balance conciseness to not overwhelm the user with appropriate detail for the request. Do not narrate abstractly; explain what you are doing and why.\n- Do not begin responses with conversational interjections or meta commentary. Avoid openers such as acknowledgements (“Done —”, “Got it”, “Great question, ”) or framing phrases.\n- The user does not see command execution outputs. When asked to show the output of a command (e.g. `git show`), relay the important details in your answer or summarize the key lines so the user understands the result.\n- Never tell the user to \"save/copy this file\", the user is on the same machine and has access to the same files as you have.\n- If the user asks for a code explanation, structure your answer with code references.\n- When given a simple task, just provide the outcome in a short answer without strong formatting.\n- When you make big or complex changes, state the solution first, then walk the user through what you did and why.\n- For casual chit-chat, just chat.\n- If you weren't able to do something, for example run tests, tell the user.\n- If there are natural next steps the user may want to take, suggest them at the end of your response. Do not make suggestions if there are no natural next steps. When suggesting multiple options, use numeric lists for the suggestions so the user can quickly respond with a single number.\n\n## Intermediary updates \n\n- Intermediary updates go to the `commentary` channel.\n- User updates are short updates while you are working, they are NOT final answers.\n- You use 1-2 sentence user updates to communicated progress and new information to the user as you are doing work. \n- Do not begin responses with conversational interjections or meta commentary. Avoid openers such as acknowledgements (“Done —”, “Got it”, “Great question, ”) or framing phrases.\n- Before exploring or doing substantial work, you start with a user update acknowledging the request and explaining your first step. You should include your understanding of the user request and explain what you will do. Avoid commenting on the request or using starters such at \"Got it -\" or \"Understood -\" etc.\n- You provide user updates frequently, every 30s.\n- When exploring, e.g. searching, reading files you provide user updates as you go, explaining what context you are gathering and what you've learned. Vary your sentence structure when providing these updates to avoid sounding repetitive - in particular, don't start each sentence the same way.\n- When working for a while, keep updates informative and varied, but stay concise.\n- After you have sufficient context, and the work is substantial you provide a longer plan (this is the only user update that may be longer than 2 sentences and can contain formatting).\n- Before performing file edits of any kind, you provide updates explaining what edits you are making.\n- As you are thinking, you very frequently provide updates even if not taking any actions, informing the user of your progress. You interrupt your thinking and send multiple updates in a row if thinking for more than 100 words.\n- Tone of your updates MUST match your personality.\n",
        ));

        let mut general_prompt = vec![ChatMessage::new(
            ChatRole::User,
            build_environment_context(&workspace.absolute_path),
        )];
        if let Some(agents_message) = build_agents_instructions_message(&workspace.absolute_path) {
            general_prompt.push(ChatMessage::new(ChatRole::User, agents_message));
        }

        // Compressed history summary replaces the compacted rounds (requirement 3).
        let compacted_context = context_state
            .as_ref()
            .filter(|state| !state.summary.trim().is_empty())
            .map(|state| compacted_history_message(&state.summary))
            .into_iter()
            .collect::<Vec<_>>();

        let mut history_messages = Vec::new();
        let mut current_turn = None;
        let mut turn_records = Vec::new();
        for record in records {
            if current_turn.is_some_and(|turn| turn != record.turn_sequence) {
                push_turn_records_as_history_messages(
                    session_id,
                    turn_records,
                    &mut history_messages,
                );
                turn_records = Vec::new();
            }

            current_turn = Some(record.turn_sequence);
            turn_records.push(record);
        }
        if !turn_records.is_empty() {
            push_turn_records_as_history_messages(session_id, turn_records, &mut history_messages);
        }

        Ok(AgentTaskContext {
            last_input_tokens,
            system_prompt,
            general_prompt,
            compacted_context,
            history_messages,
            latest_user_messages: vec![ChatMessage::new(ChatRole::User, prompt)],
        })
    }

    async fn system_tools_for_session(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<(ToolRegistry, AgentToolContext)> {
        let storage = self.storage()?;
        let session = storage.get_session(session_id).await?;
        let workspace = storage.get_workspace(session.workspace_id).await?;
        let tool_context = AgentToolContext::workspace(workspace.id, workspace.absolute_path);
        let registry = system_tool_registry(SystemToolConfig::new())?;
        Ok((registry, tool_context))
    }

    async fn get_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>> {
        self.storage()?.list_sessions().await
    }

    pub(crate) async fn get_session(&self, session_id: SessionId) -> SeekCodeResult<SessionRecord> {
        self.storage()?.get_session(session_id).await
    }

    pub(crate) async fn rename_session(
        &self,
        session_id: SessionId,
        name: String,
    ) -> SeekCodeResult<SessionRecord> {
        self.storage()?.rename_session(session_id, name).await
    }

    async fn update_session_model(
        &self,
        session_id: SessionId,
        model_provider: String,
        model: String,
        thinking_enabled: bool,
        reasoning_effort: Option<String>,
    ) -> SeekCodeResult<SessionRecord> {
        let model_provider = model_provider.trim().to_string();
        if model_provider.is_empty() {
            return Err(SeekCodeError::Validation(
                "session model provider cannot be empty".to_string(),
            ));
        }
        let model = model.trim().to_string();
        if model.is_empty() {
            return Err(SeekCodeError::Validation(
                "session model cannot be empty".to_string(),
            ));
        }

        self.storage()?
            .update_session_model(
                session_id,
                model_provider,
                model,
                thinking_enabled,
                normalize_reasoning_effort(reasoning_effort),
            )
            .await
    }

    pub(crate) async fn save_compaction(
        &self,
        session_id: SessionId,
        summary: String,
        compacted_through_turn: i64,
    ) -> SeekCodeResult<()> {
        self.storage()?
            .save_session_compaction(session_id, summary, compacted_through_turn)
            .await
    }

    /// Records the latest model input token count, logging any storage failure.
    pub(crate) async fn update_last_input_tokens(&self, session_id: SessionId, tokens: i64) {
        let result = match self.storage() {
            Ok(storage) => {
                storage
                    .update_session_last_input_tokens(session_id, tokens)
                    .await
            }
            Err(error) => Err(error),
        };

        if let Err(error) = result {
            tracing::warn!(
                target: "seekcode_app_kernel::context_compaction",
                %error,
                "failed to persist last input tokens"
            );
        }
    }

    pub(crate) async fn append_model_call_log(&self, log: NewModelCallLog) {
        let result = match self.storage() {
            Ok(storage) => storage.append_model_call_log(log).await,
            Err(error) => Err(error),
        };

        if let Err(error) = result {
            tracing::warn!(
                target: "seekcode_app_kernel::model_call_log",
                %error,
                "failed to persist model call log"
            );
        }
    }

    async fn list_workspace_sessions(
        &self,
        workspace_id: WorkspaceId,
    ) -> SeekCodeResult<Vec<SessionRecord>> {
        self.storage()?.list_workspace_sessions(workspace_id).await
    }

    async fn create_session(
        &self,
        request: CreateSessionRequest,
        config: &AppKernelConfig,
    ) -> SeekCodeResult<SessionRecord> {
        let model = request
            .model
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(&config.agent.default_model)
            .to_string();
        let name = request
            .name
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .to_string();

        self.storage()?
            .create_session(NewSession {
                id: SessionId::new(),
                workspace_id: request.workspace_id,
                name,
                model_provider: request
                    .model_provider
                    .unwrap_or_else(|| "deepseek".to_string()),
                model,
                thinking_enabled: request.thinking_enabled.unwrap_or(config.agent.thinking),
                reasoning_effort: normalize_reasoning_effort(request.reasoning_effort),
            })
            .await
    }

    async fn delete_session(&self, session_id: SessionId) -> SeekCodeResult<()> {
        self.storage()?.delete_session(session_id).await
    }

    async fn delete_workspace_sessions(&self, workspace_id: WorkspaceId) -> SeekCodeResult<()> {
        self.storage()?
            .delete_workspace_sessions(workspace_id)
            .await
    }

    fn storage(&self) -> SeekCodeResult<&Arc<dyn Storage>> {
        self.storage
            .as_ref()
            .ok_or(SeekCodeError::NotImplemented("storage is not wired yet"))
    }
}

struct SessionTaskContextPreparer {
    sessions: Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
}

fn push_turn_records_as_history_messages(
    session_id: SessionId,
    records: Vec<SessionMessageRecord>,
    history_messages: &mut Vec<AgentHistoryMessage>,
) {
    let Some(turn_sequence) = records.first().map(|record| record.turn_sequence) else {
        return;
    };
    let mut messages = Vec::new();
    push_turn_records_as_context_messages(session_id, records, &mut messages);
    history_messages.extend(messages.into_iter().map(|message| AgentHistoryMessage {
        turn_sequence,
        message,
    }));
}

#[async_trait]
impl AgentContextPreparer for SessionTaskContextPreparer {
    async fn precheck_context(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        model: &str,
        _prompt: &str,
        current_context: &AgentTaskContext,
    ) -> SeekCodeResult<AgentContextPrecheck> {
        let should_compact_context = match precheck_session_context_compaction(
            self.provider.clone(),
            model.to_string(),
            current_context,
        )
        .await
        {
            Ok(should_compact) => should_compact,
            Err(error) => {
                tracing::warn!(
                    target: "seekcode_app_kernel::context_compaction",
                    %session_id,
                    %task_id,
                    %error,
                    "context compaction precheck failed; continuing without compaction"
                );
                false
            }
        };

        Ok(AgentContextPrecheck {
            should_compact_context,
        })
    }

    async fn prepare_context(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        model: &str,
        _prompt: &str,
        current_context: &AgentTaskContext,
        precheck: AgentContextPrecheck,
    ) -> SeekCodeResult<PreparedAgentContext> {
        if !precheck.should_compact_context {
            return Ok(PreparedAgentContext {
                context: current_context.clone(),
                compaction: None,
            });
        }

        let outcome = compact_session_context(
            self.sessions.clone(),
            self.provider.clone(),
            session_id,
            model.to_string(),
            current_context,
        )
        .await?;

        let mut context = current_context.clone();
        let compaction = if let Some(outcome) = outcome {
            context.replace_compacted_context(
                vec![compacted_history_message(&outcome.summary)],
                outcome.compacted_through_turn,
            );
            Some(AgentContextCompactionOutcome {
                compacted_rounds: outcome.compacted_rounds,
                compacted_through_turn: outcome.compacted_through_turn,
                summary_chars: outcome.summary_chars,
            })
        } else {
            tracing::warn!(
                target: "seekcode_app_kernel::context_compaction",
                %session_id,
                %task_id,
                "context compaction precheck succeeded but compaction did not produce a summary"
            );
            None
        };

        Ok(PreparedAgentContext {
            context,
            compaction,
        })
    }
}

impl AppKernel {
    /// Builds the application service graph.
    pub fn new(config: AppKernelConfig) -> anyhow::Result<Self> {
        Self::new_with_optional_storage(config, None)
    }

    /// Builds the application service graph with durable storage.
    pub fn with_storage(
        config: AppKernelConfig,
        storage: Arc<dyn Storage>,
    ) -> anyhow::Result<Self> {
        Self::new_with_optional_storage(config, Some(storage))
    }

    fn new_with_optional_storage(
        config: AppKernelConfig,
        storage: Option<Arc<dyn Storage>>,
    ) -> anyhow::Result<Self> {
        init_tracing(&config.telemetry)?;

        let provider: Arc<dyn ModelProvider> =
            Arc::new(DeepSeekClient::new(config.deepseek.clone())?);
        let provider_slot = Arc::new(RwLock::new(provider.clone()));
        let tools = Arc::new(ToolRegistry::new());
        let agent = Arc::new(Agent::new(
            config.agent.clone(),
            provider.clone(),
            tools.clone(),
        ));
        let policy: Arc<dyn PolicyEngine> = Arc::new(AutonomousPolicy);
        let secrets: Arc<dyn SecretStore> = Arc::new(InMemorySecretStore::new());
        let shell = Arc::new(CommandRunner::new(config.shell.clone()));
        let sessions = Arc::new(SessionService::new(storage.clone()));

        Ok(Self {
            config: RwLock::new(config),
            services: AppServices {
                provider: provider_slot,
                agent,
                tools,
                policy,
                storage,
                sessions,
                secrets,
                shell,
            },
        })
    }

    /// Returns the kernel configuration.
    pub fn config(&self) -> AppKernelConfig {
        self.config.read().clone()
    }

    /// Returns assembled services.
    pub fn services(&self) -> &AppServices {
        &self.services
    }

    /// Updates DeepSeek provider configuration for newly-started tasks.
    pub async fn update_deepseek_config(&self, deepseek: DeepSeekConfig) -> anyhow::Result<()> {
        let provider: Arc<dyn ModelProvider> = Arc::new(DeepSeekClient::new(deepseek.clone())?);
        self.services.agent.set_provider(provider.clone());
        *self.services.provider.write() = provider;

        self.config.write().deepseek = deepseek;

        Ok(())
    }

    /// Updates the model used for background session-title generation.
    pub fn update_title_model(&self, title_model: String) {
        self.config.write().title_model = title_model;
    }

    /// Starts background title generation for an empty-title session.
    pub fn spawn_session_title_generation(
        &self,
        session_id: SessionId,
        prompt: String,
        title_events: mpsc::UnboundedSender<SessionTitleChanged>,
    ) {
        if !self.services.sessions.start_title_task(session_id) {
            return;
        }

        let sessions = self.services.sessions.clone();
        let provider = self.services.provider.read().clone();
        let title_model = self.config.read().title_model.clone();

        tokio::spawn(async move {
            match generate_session_title(
                sessions.clone(),
                provider,
                session_id,
                title_model,
                prompt,
            )
            .await
            {
                Ok(Some(event)) => {
                    let _ = title_events.send(event);
                }
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(
                        target: "seekcode_app_kernel::session_title",
                        %session_id,
                        %error,
                        "failed to generate session title"
                    );
                }
            }
            sessions.finish_title_task(session_id);
        });
    }

    /// Starts an agent task.
    pub async fn start_agent_task(
        &self,
        request: StartTaskRequest,
    ) -> SeekCodeResult<StartedAgentTask> {
        let prompt = request.prompt.trim();
        if prompt.is_empty() {
            return Err(SeekCodeError::Validation(
                "agent task prompt cannot be empty".to_string(),
            ));
        }

        let session_id = request.session_id;
        self.services
            .sessions
            .start_running_task(session_id)
            .map_err(|error| {
                tracing::warn!(
                    target: "seekcode_app_kernel::start_agent_task",
                    %session_id,
                    %error,
                    "failed to mark session as running"
                );
                error
            })?;
        let (tools, tool_context) = match self
            .services
            .sessions
            .system_tools_for_session(session_id)
            .await
        {
            Ok((tools, tool_context)) => (Arc::new(tools), tool_context),
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
                tracing::error!(
                    target: "seekcode_app_kernel::start_agent_task",
                    %session_id,
                    %error,
                    "failed to build system tools for session"
                );
                return Err(error);
            }
        };
        let turn_sequence = match self
            .services
            .sessions
            .append_user_prompt(session_id, prompt)
            .await
        {
            Ok(turn_sequence) => turn_sequence,
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
                tracing::error!(
                    target: "seekcode_app_kernel::start_agent_task",
                    %session_id,
                    %error,
                    "failed to append user prompt before starting agent task"
                );
                return Err(error);
            }
        };
        let (agent_events, agent_event_receiver) = mpsc::unbounded_channel();
        let (ui_events, ui_event_receiver) = mpsc::unbounded_channel();
        let context_preparer: Arc<dyn AgentContextPreparer> =
            Arc::new(SessionTaskContextPreparer {
                sessions: self.services.sessions.clone(),
                provider: self.services.provider.read().clone(),
            });
        let initial_context = match self
            .services
            .sessions
            .assemble_task_context_excluding_turn_from(session_id, prompt, turn_sequence)
            .await
        {
            Ok(context) => context,
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
                tracing::error!(
                    target: "seekcode_app_kernel::start_agent_task",
                    %session_id,
                    turn_sequence,
                    %error,
                    "failed to assemble agent task context"
                );
                return Err(error);
            }
        };
        let task = match self
            .services
            .agent
            .start_task_with_messages_tools_tool_context_and_context_preparer(
                request,
                initial_context,
                tools,
                tool_context,
                agent_events,
                Some(context_preparer),
            )
            .await
        {
            Ok(task) => task,
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
                tracing::error!(
                    target: "seekcode_app_kernel::start_agent_task",
                    %session_id,
                    turn_sequence,
                    %error,
                    "failed to start agent runtime task"
                );
                return Err(error);
            }
        };
        spawn_session_agent_event_bridge(
            self.services.sessions.clone(),
            agent_event_receiver,
            ui_events,
            task.id,
            session_id,
            turn_sequence,
        );

        Ok(StartedAgentTask {
            task,
            events: ui_event_receiver,
        })
    }

    /// Cancels an agent task.
    pub async fn cancel_agent_task(&self, task_id: TaskId) -> SeekCodeResult<()> {
        self.services.agent.cancel_task(task_id).await
    }

    /// Lists persisted sessions.
    pub async fn get_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>> {
        self.services.sessions.get_sessions().await
    }

    /// Opens an existing workspace by path or creates a new visible workspace.
    pub async fn open_workspace(
        &self,
        request: OpenWorkspaceRequest,
    ) -> SeekCodeResult<WorkspaceWithSessions> {
        let storage = self.storage()?;
        let absolute_path = request.absolute_path.trim();
        if absolute_path.is_empty() {
            return Err(SeekCodeError::Validation(
                "workspace absolute_path cannot be empty".to_string(),
            ));
        }

        let workspace = match storage.find_workspace_by_path(absolute_path).await? {
            Some(workspace) => {
                if !workspace.is_visible {
                    storage.set_workspace_visibility(workspace.id, true).await?;
                }
                storage
                    .find_workspace_by_path(absolute_path)
                    .await?
                    .ok_or_else(|| {
                        SeekCodeError::NotFound(format!("workspace path: {absolute_path}"))
                    })?
            }
            None => {
                let name = request
                    .name
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| workspace_name_from_path(absolute_path));
                storage
                    .create_workspace(NewWorkspace {
                        id: WorkspaceId::new(),
                        name,
                        absolute_path: absolute_path.to_string(),
                        is_visible: true,
                    })
                    .await?
            }
        };

        self.workspace_with_sessions(workspace).await
    }

    /// Lists visible workspaces and their sessions.
    pub async fn list_visible_workspaces(&self) -> SeekCodeResult<Vec<WorkspaceWithSessions>> {
        let storage = self.storage()?;
        let workspaces = storage.list_visible_workspaces().await?;
        let mut items = Vec::with_capacity(workspaces.len());

        for workspace in workspaces {
            items.push(self.workspace_with_sessions(workspace).await?);
        }

        Ok(items)
    }

    /// Hides a workspace from the sidebar.
    pub async fn hide_workspace(&self, workspace_id: WorkspaceId) -> SeekCodeResult<()> {
        self.storage()?
            .set_workspace_visibility(workspace_id, false)
            .await
    }

    /// Creates a persisted session.
    pub async fn create_session(
        &self,
        request: CreateSessionRequest,
    ) -> SeekCodeResult<SessionRecord> {
        let config = self.config();
        self.services
            .sessions
            .create_session(request, &config)
            .await
    }

    /// Deletes a session and its messages.
    pub async fn delete_session(&self, session_id: SessionId) -> SeekCodeResult<()> {
        self.services.sessions.delete_session(session_id).await
    }

    /// Updates the model selected for one session.
    pub async fn update_session_model(
        &self,
        session_id: SessionId,
        model_provider: String,
        model: String,
        thinking_enabled: bool,
        reasoning_effort: Option<String>,
    ) -> SeekCodeResult<SessionRecord> {
        self.services
            .sessions
            .update_session_model(
                session_id,
                model_provider,
                model,
                thinking_enabled,
                reasoning_effort,
            )
            .await
    }

    /// Deletes all sessions under a workspace.
    pub async fn delete_workspace_sessions(&self, workspace_id: WorkspaceId) -> SeekCodeResult<()> {
        self.services
            .sessions
            .delete_workspace_sessions(workspace_id)
            .await
    }

    /// Lists persisted messages for one session.
    pub async fn list_session_messages(
        &self,
        session_id: SessionId,
        before_turn_sequence: Option<i64>,
        turn_limit: Option<i64>,
    ) -> SeekCodeResult<Vec<SessionMessageRecord>> {
        let limit = turn_limit.unwrap_or(20).clamp(1, 100);
        self.storage()?
            .list_session_messages_page(session_id, before_turn_sequence, limit)
            .await
    }

    /// Returns the most recent model input token count recorded for a session.
    pub async fn session_context_usage(&self, session_id: SessionId) -> SeekCodeResult<i64> {
        let session = self.services.sessions.get_session(session_id).await?;
        Ok(session.last_input_tokens)
    }

    /// Returns aggregated model call telemetry for a session.
    pub async fn session_model_call_stats(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<SessionModelCallStats> {
        self.storage()?.session_model_call_stats(session_id).await
    }

    fn storage(&self) -> SeekCodeResult<&Arc<dyn Storage>> {
        self.services
            .storage
            .as_ref()
            .ok_or(SeekCodeError::NotImplemented("storage is not wired yet"))
    }

    async fn workspace_with_sessions(
        &self,
        workspace: WorkspaceRecord,
    ) -> SeekCodeResult<WorkspaceWithSessions> {
        let sessions = self
            .services
            .sessions
            .list_workspace_sessions(workspace.id)
            .await?;
        Ok(WorkspaceWithSessions {
            workspace,
            sessions,
        })
    }
}

fn workspace_name_from_path(path: &str) -> String {
    PathBuf::from(path)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("Untitled Workspace")
        .to_string()
}

#[cfg(test)]
fn normalize_generated_session_title(value: &str) -> String {
    let mut title = value
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .trim_start_matches(|ch: char| {
            ch.is_ascii_digit()
                || matches!(ch, '-' | '*' | '•' | '.' | ')' | ']' | '、' | ' ' | '\t')
        })
        .trim()
        .trim_matches(|ch| {
            matches!(
                ch,
                '"' | '\'' | '`' | '“' | '”' | '‘' | '’' | '「' | '」' | '《' | '》'
            )
        })
        .trim()
        .to_string();

    if title.chars().count() > 48 {
        title = title.chars().take(48).collect();
    }

    title
}

fn normalize_reasoning_effort(value: Option<String>) -> Option<String> {
    let value = value?.trim().to_lowercase();
    matches!(value.as_str(), "high" | "max").then_some(value)
}

fn choice_tool_calls_json(
    choice: &seekcode_deepseek_client::ChatChoiceChunk,
) -> Vec<serde_json::Value> {
    choice
        .delta
        .tool_calls
        .iter()
        .map(|tool_call| {
            let mut call = serde_json::Map::new();
            if let Some(id) = tool_call.id {
                call.insert("id".to_string(), serde_json::Value::String(id.to_string()));
            }
            call.insert(
                "type".to_string(),
                serde_json::Value::String(
                    tool_call.kind.as_deref().unwrap_or("function").to_string(),
                ),
            );

            let mut function = serde_json::Map::new();
            if let Some(name) = &tool_call.name {
                function.insert("name".to_string(), serde_json::Value::String(name.clone()));
            }
            if let Some(arguments) = &tool_call.arguments {
                function.insert(
                    "arguments".to_string(),
                    serde_json::Value::String(arguments.clone()),
                );
            }
            call.insert("function".to_string(), serde_json::Value::Object(function));
            serde_json::Value::Object(call)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{
        build_skills_system_message_for_dir, current_timezone_name, SKILLS_SYSTEM_PREFIX,
    };
    use async_trait::async_trait;
    use chrono::Local;
    use seekcode_agent_core::AgentEvent;
    use seekcode_deepseek_client::{
        ChatChunk, ChatRequest, ChatResponse, ChatStream, ModelProfile,
    };
    use seekcode_storage::{SessionContextStore, SessionStore, SqliteStorage, WorkspaceStore};
    use std::sync::atomic::{AtomicI64, Ordering};

    static TEST_MESSAGE_CLOCK: AtomicI64 = AtomicI64::new(0);

    fn next_test_created_at() -> String {
        let offset = TEST_MESSAGE_CLOCK.fetch_add(1, Ordering::Relaxed);
        format!("2026-01-01 00:{:02}:{:02}", (offset / 60) % 60, offset % 60)
    }

    struct CapturingProvider {
        context_window: u32,
        summary: String,
        requests: Arc<std::sync::Mutex<Vec<ChatRequest>>>,
    }

    #[async_trait]
    impl ModelProvider for CapturingProvider {
        fn stream_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatStream> {
            self.requests.lock().expect("requests lock").push(request);
            Ok(Box::pin(futures_util::stream::iter([Ok(
                ChatChunk::Finished,
            )])))
        }

        async fn complete_chat(&self, _request: ChatRequest) -> SeekCodeResult<ChatResponse> {
            Ok(ChatResponse {
                content: self.summary.clone(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                usage: None,
            })
        }

        async fn model_profile(&self, model: &str) -> SeekCodeResult<ModelProfile> {
            Ok(ModelProfile {
                id: model.to_string(),
                context_window: self.context_window,
                supports_tools: true,
                supports_thinking: true,
            })
        }
    }

    async fn seed_session(storage: &SqliteStorage) -> SessionId {
        let workspace_id = WorkspaceId::new();
        let session_id = SessionId::new();
        let workspace_path = std::env::temp_dir()
            .join(format!("seekcode-workspace-test-{workspace_id}"))
            .to_string_lossy()
            .to_string();
        std::fs::create_dir_all(&workspace_path).expect("workspace dir creates");
        storage
            .create_workspace(NewWorkspace {
                id: workspace_id,
                name: "SeekCode".to_string(),
                absolute_path: workspace_path,
                is_visible: true,
            })
            .await
            .expect("workspace creates");
        storage
            .create_session(NewSession {
                id: session_id,
                workspace_id,
                name: "Chat".to_string(),
                model_provider: "deepseek".to_string(),
                model: "deepseek-v4-pro".to_string(),
                thinking_enabled: true,
                reasoning_effort: None,
            })
            .await
            .expect("session creates");
        session_id
    }

    async fn seed_user_turn(storage: &SqliteStorage, session_id: SessionId, turn: i64, text: &str) {
        storage
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence: turn,
                role: ChatRole::User,
                content: text.to_string(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                created_at: next_test_created_at(),
            })
            .await
            .expect("message appends");
    }

    async fn seed_assistant_turn(
        storage: &SqliteStorage,
        session_id: SessionId,
        turn: i64,
        content: &str,
        reasoning_content: Option<&str>,
        tool_calls: Vec<serde_json::Value>,
    ) {
        storage
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence: turn,
                role: ChatRole::Assistant,
                content: content.to_string(),
                reasoning_content: reasoning_content.map(ToOwned::to_owned),
                tool_calls,
                tool_call_id: None,
                created_at: next_test_created_at(),
            })
            .await
            .expect("message appends");
    }

    async fn seed_tool_result(
        storage: &SqliteStorage,
        session_id: SessionId,
        turn: i64,
        tool_call_id: ToolCallId,
        content: &str,
    ) {
        storage
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence: turn,
                role: ChatRole::Tool,
                content: content.to_string(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: Some(tool_call_id),
                created_at: next_test_created_at(),
            })
            .await
            .expect("message appends");
    }

    fn tool_call_json(tool_call_id: ToolCallId) -> serde_json::Value {
        serde_json::json!({
            "id": tool_call_id.to_string(),
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\":\"src/lib.rs\"}"
            }
        })
    }

    #[test]
    fn app_kernel_can_be_constructed() {
        let kernel = AppKernel::new(AppKernelConfig::default()).expect("kernel builds");

        assert_eq!(kernel.config().agent.default_model, "deepseek-v4-pro");
        assert_eq!(kernel.config().title_model, "deepseek-v4-flash");
    }

    #[tokio::test]
    async fn app_kernel_updates_deepseek_config() {
        let kernel = AppKernel::new(AppKernelConfig::default()).expect("kernel builds");
        let mut deepseek = DeepSeekConfig::default();
        deepseek.base_url = "https://example.test".to_string();
        deepseek.api_key = Some("sk-test".to_string());

        kernel
            .update_deepseek_config(deepseek)
            .await
            .expect("deepseek config updates");

        let config = kernel.config();
        assert_eq!(config.deepseek.base_url, "https://example.test");
        assert_eq!(config.deepseek.api_key.as_deref(), Some("sk-test"));
    }

    #[tokio::test]
    async fn assemble_context_appends_environment_context_before_prompt() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let workspace_id = WorkspaceId::new();
        let session_id = SessionId::new();
        let root =
            std::env::temp_dir().join(format!("seekcode-env-context-test-{}", MessageId::new()));
        std::fs::create_dir_all(&root).expect("workspace dir creates");
        let workspace_path = root.to_string_lossy().to_string();

        storage
            .create_workspace(NewWorkspace {
                id: workspace_id,
                name: "SeekCode".to_string(),
                absolute_path: workspace_path.clone(),
                is_visible: true,
            })
            .await
            .expect("workspace creates");
        storage
            .create_session(NewSession {
                id: session_id,
                workspace_id,
                name: "Initial chat".to_string(),
                model_provider: "deepseek".to_string(),
                model: "deepseek-v4-pro".to_string(),
                thinking_enabled: true,
                reasoning_effort: None,
            })
            .await
            .expect("session creates");
        storage
            .append_session_message(NewSessionMessage {
                id: MessageId::new(),
                session_id,
                turn_sequence: 1,
                role: ChatRole::User,
                content: "previous prompt".to_string(),
                reasoning_content: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
                created_at: local_now_text(),
            })
            .await
            .expect("message appends");

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "new prompt")
            .await
            .expect("context assembles");

        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, ChatRole::System);
        assert_eq!(messages[1].role, ChatRole::User);
        assert!(messages[1]
            .content
            .contains(&format!("<cwd>{workspace_path}</cwd>")));
        assert!(messages[1].content.contains("<shell>powershell</shell>"));
        assert!(messages[1].content.contains(&format!(
            "<timezone>{}</timezone>",
            current_timezone_name(&Local::now())
        )));
        assert!(messages[1].content.contains(&format!(
            "<workspace_roots><root>{workspace_path}</root></workspace_roots>"
        )));
        assert_eq!(messages[2].content, "previous prompt");
        assert_eq!(messages[3].role, ChatRole::User);
        assert_eq!(messages[3].content, "new prompt");

        std::fs::remove_dir_all(root).expect("workspace dir cleans up");
    }

    #[tokio::test]
    async fn assemble_context_appends_agents_instructions_after_environment_context() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let root = std::env::temp_dir().join(format!("seekcode-agents-test-{}", MessageId::new()));
        std::fs::create_dir_all(&root).expect("workspace dir creates");
        std::fs::write(root.join("AGENTS.md"), "Use workspace conventions.")
            .expect("agents file writes");
        let workspace_path = root.to_string_lossy().to_string();
        let workspace_id = WorkspaceId::new();
        let session_id = SessionId::new();

        storage
            .create_workspace(NewWorkspace {
                id: workspace_id,
                name: "SeekCode".to_string(),
                absolute_path: workspace_path.clone(),
                is_visible: true,
            })
            .await
            .expect("workspace creates");
        storage
            .create_session(NewSession {
                id: session_id,
                workspace_id,
                name: "Initial chat".to_string(),
                model_provider: "deepseek".to_string(),
                model: "deepseek-v4-pro".to_string(),
                thinking_enabled: true,
                reasoning_effort: None,
            })
            .await
            .expect("session creates");
        seed_user_turn(&storage, session_id, 1, "previous prompt").await;

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "new prompt")
            .await
            .expect("context assembles");

        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, ChatRole::System);
        assert!(messages[1].content.contains("<environment_context>"));
        assert_eq!(messages[2].role, ChatRole::User);
        assert!(messages[2]
            .content
            .contains(&format!("# AGENTS.md instructions for {workspace_path}")));
        assert!(messages[2].content.contains("<INSTRUCTIONS>"));
        assert!(messages[2].content.contains("Use workspace conventions."));
        assert!(messages[2].content.contains("</INSTRUCTIONS>"));
        assert_eq!(messages[3].content, "previous prompt");
        assert_eq!(messages[4].content, "new prompt");

        std::fs::remove_dir_all(root).expect("workspace dir cleans up");
    }

    #[tokio::test]
    async fn assemble_context_uses_summary_and_skips_compacted_turns() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let session_id = seed_session(&storage).await;
        seed_user_turn(&storage, session_id, 1, "oldest question").await;
        seed_user_turn(&storage, session_id, 2, "middle question").await;
        seed_user_turn(&storage, session_id, 3, "recent question").await;
        storage
            .save_session_compaction(session_id, "COMPACTED SUMMARY".to_string(), 2)
            .await
            .expect("save compaction");

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "new prompt")
            .await
            .expect("context assembles");

        // system(personality) + env context + system(summary) + turn 3 + new prompt.
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, ChatRole::System);
        assert!(messages[1].content.contains("<environment_context>"));
        assert_eq!(messages[2].role, ChatRole::System);
        assert!(messages[2].content.contains("COMPACTED SUMMARY"));
        assert!(messages[2].content.contains("<compacted_history>"));
        // Compacted turns must not appear verbatim; the recent turn must remain.
        assert!(!messages.iter().any(|m| m.content == "oldest question"));
        assert!(!messages.iter().any(|m| m.content == "middle question"));
        assert!(messages.iter().any(|m| m.content == "recent question"));
        assert_eq!(messages[4].content, "new prompt");
    }

    #[tokio::test]
    async fn assemble_context_drops_reasoning_for_turn_without_tool_calls() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let session_id = seed_session(&storage).await;
        seed_user_turn(&storage, session_id, 1, "question 1").await;
        seed_assistant_turn(
            &storage,
            session_id,
            1,
            "answer 1",
            Some("reasoning 1"),
            Vec::new(),
        )
        .await;

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "question 2")
            .await
            .expect("context assembles");

        let assistant = messages
            .iter()
            .find(|message| message.role == ChatRole::Assistant && message.content == "answer 1")
            .expect("assistant message is present");
        assert_eq!(assistant.reasoning_content, None);
        assert!(assistant.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn assemble_context_preserves_reasoning_for_turn_with_tool_call() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let session_id = seed_session(&storage).await;
        let tool_call_id = ToolCallId::new();
        seed_user_turn(&storage, session_id, 1, "question 1").await;
        seed_assistant_turn(
            &storage,
            session_id,
            1,
            "",
            Some("reasoning before tool"),
            vec![tool_call_json(tool_call_id)],
        )
        .await;
        seed_tool_result(&storage, session_id, 1, tool_call_id, "{\"ok\":true}").await;

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "question 2")
            .await
            .expect("context assembles");

        let assistant = messages
            .iter()
            .find(|message| message.role == ChatRole::Assistant)
            .expect("assistant message is present");
        assert_eq!(
            assistant.reasoning_content.as_deref(),
            Some("reasoning before tool")
        );
        assert_eq!(assistant.tool_calls.len(), 1);
        assert!(messages.iter().any(|message| {
            message.role == ChatRole::Tool
                && message.tool_call_id == Some(tool_call_id)
                && message.content == "{\"ok\":true}"
        }));
    }

    #[tokio::test]
    async fn assemble_context_drops_unanswered_tool_call_assistant() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let session_id = seed_session(&storage).await;
        let tool_call_id = ToolCallId::new();
        seed_user_turn(&storage, session_id, 1, "question 1").await;
        seed_assistant_turn(
            &storage,
            session_id,
            1,
            "I will inspect a file.",
            Some("reasoning before canceled tool"),
            vec![tool_call_json(tool_call_id)],
        )
        .await;

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "question 2")
            .await
            .expect("context assembles");

        assert!(!messages.iter().any(|message| {
            message.role == ChatRole::Assistant
                && (!message.tool_calls.is_empty()
                    || message.content == "I will inspect a file."
                    || message.reasoning_content.as_deref()
                        == Some("reasoning before canceled tool"))
        }));
        assert!(!messages
            .iter()
            .any(|message| message.role == ChatRole::Tool));
    }

    #[tokio::test]
    async fn assemble_context_preserves_only_tool_turn_reasoning_in_mixed_history() {
        let storage = SqliteStorage::connect("sqlite::memory:")
            .await
            .expect("storage connects");
        let session_id = seed_session(&storage).await;
        let tool_call_id = ToolCallId::new();
        seed_user_turn(&storage, session_id, 1, "question 1").await;
        seed_assistant_turn(
            &storage,
            session_id,
            1,
            "",
            Some("reasoning 1"),
            vec![tool_call_json(tool_call_id)],
        )
        .await;
        seed_tool_result(&storage, session_id, 1, tool_call_id, "tool result").await;
        seed_assistant_turn(
            &storage,
            session_id,
            1,
            "answer 1",
            Some("reasoning after tool"),
            Vec::new(),
        )
        .await;
        seed_user_turn(&storage, session_id, 2, "question 2").await;
        seed_assistant_turn(
            &storage,
            session_id,
            2,
            "answer 2",
            Some("reasoning 2"),
            Vec::new(),
        )
        .await;

        let service = SessionService::new(Some(Arc::new(storage)));
        let messages = service
            .assemble_context(session_id, "question 3")
            .await
            .expect("context assembles");

        let tool_call_assistant = messages
            .iter()
            .find(|message| message.role == ChatRole::Assistant && message.tool_calls.len() == 1)
            .expect("tool-call assistant message is present");
        assert_eq!(
            tool_call_assistant.reasoning_content.as_deref(),
            Some("reasoning 1")
        );

        let tool_turn_answer = messages
            .iter()
            .find(|message| message.role == ChatRole::Assistant && message.content == "answer 1")
            .expect("tool turn final assistant message is present");
        assert_eq!(
            tool_turn_answer.reasoning_content.as_deref(),
            Some("reasoning after tool")
        );

        let plain_turn_assistant = messages
            .iter()
            .find(|message| message.role == ChatRole::Assistant && message.content == "answer 2")
            .expect("plain turn assistant message is present");
        assert_eq!(plain_turn_assistant.reasoning_content, None);
        assert!(plain_turn_assistant.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn start_agent_task_compacts_inside_runner_and_reassembles_context() {
        let storage = Arc::new(
            SqliteStorage::connect("sqlite::memory:")
                .await
                .expect("storage connects"),
        );
        let session_id = seed_session(&storage).await;
        for turn in 1..=5 {
            seed_user_turn(&storage, session_id, turn, &format!("turn {turn}")).await;
        }
        storage
            .update_session_last_input_tokens(session_id, 900)
            .await
            .expect("seed token watermark");

        let kernel =
            AppKernel::with_storage(AppKernelConfig::default(), storage).expect("kernel builds");
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            context_window: 1_000,
            summary: "STUB SUMMARY".to_string(),
            requests: requests.clone(),
        });
        kernel.services.agent.set_provider(provider.clone());
        *kernel.services.provider.write() = provider;

        let mut started = kernel
            .start_agent_task(StartTaskRequest {
                session_id,
                prompt: "current question".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            })
            .await
            .expect("task starts");

        let mut event_order = Vec::new();
        while let Some(event) = started.events.recv().await {
            match event {
                AgentEvent::ContextCompactionStarted { task_id, .. }
                    if task_id == started.task.id =>
                {
                    event_order.push("compaction_started");
                }
                AgentEvent::ContextCompactionFinished { task_id, .. }
                    if task_id == started.task.id =>
                {
                    event_order.push("compaction_finished");
                }
                AgentEvent::ModelRequestStarted { task_id, .. } if task_id == started.task.id => {
                    event_order.push("model_request_started");
                }
                AgentEvent::Finished { task_id, .. } if task_id == started.task.id => break,
                _ => {}
            }
        }

        let started_index = event_order
            .iter()
            .position(|event| *event == "compaction_started")
            .expect("compaction started event is published");
        let finished_index = event_order
            .iter()
            .position(|event| *event == "compaction_finished")
            .expect("compaction finished event is published");
        let model_request_index = event_order
            .iter()
            .position(|event| *event == "model_request_started")
            .expect("model request started event is published");
        assert!(started_index < finished_index);
        assert!(finished_index < model_request_index);
        let requests = requests.lock().expect("requests lock");
        assert_eq!(requests.len(), 1);
        let request_text = requests[0]
            .messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(request_text.contains("STUB SUMMARY"));
        assert!(request_text.contains("turn 3"));
        assert!(request_text.contains("turn 4"));
        assert!(request_text.contains("turn 5"));
        assert!(request_text.contains("current question"));
        assert!(!request_text.contains("turn 1"));
        assert!(!request_text.contains("turn 2"));
        assert_eq!(request_text.matches("current question").count(), 1);
    }

    #[tokio::test]
    async fn start_agent_task_does_not_emit_compaction_events_when_not_needed() {
        let storage = Arc::new(
            SqliteStorage::connect("sqlite::memory:")
                .await
                .expect("storage connects"),
        );
        let session_id = seed_session(&storage).await;
        for turn in 1..=5 {
            seed_user_turn(&storage, session_id, turn, &format!("turn {turn}")).await;
        }
        storage
            .update_session_last_input_tokens(session_id, 100)
            .await
            .expect("seed token watermark");

        let kernel =
            AppKernel::with_storage(AppKernelConfig::default(), storage).expect("kernel builds");
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            context_window: 1_000,
            summary: "STUB SUMMARY".to_string(),
            requests: requests.clone(),
        });
        kernel.services.agent.set_provider(provider.clone());
        *kernel.services.provider.write() = provider;

        let mut started = kernel
            .start_agent_task(StartTaskRequest {
                session_id,
                prompt: "current question".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            })
            .await
            .expect("task starts");

        let mut saw_compaction_event = false;
        while let Some(event) = started.events.recv().await {
            match event {
                AgentEvent::ContextCompactionStarted { task_id, .. }
                | AgentEvent::ContextCompactionFinished { task_id, .. }
                    if task_id == started.task.id =>
                {
                    saw_compaction_event = true;
                }
                AgentEvent::Finished { task_id, .. } if task_id == started.task.id => break,
                _ => {}
            }
        }

        assert!(!saw_compaction_event);
        assert_eq!(requests.lock().expect("requests lock").len(), 1);
    }

    #[test]
    fn skills_system_message_lists_skill_md_files() {
        let root = std::env::temp_dir().join(format!("seekcode-skills-test-{}", MessageId::new()));
        let skill_dir = root.join("skills").join("sample-skill");
        std::fs::create_dir_all(&skill_dir).expect("skill dir creates");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: sample-skill\ndescription: Use when testing skill discovery.\n---\n# Body\n",
        )
        .expect("skill writes");

        let message =
            build_skills_system_message_for_dir(&root.join("skills")).expect("skills message");
        assert!(message.starts_with(SKILLS_SYSTEM_PREFIX));
        assert!(message.contains("- sample-skill: Use when testing skill discovery. (file: "));
        assert!(message.contains("sample-skill"));
        assert!(message.contains("SKILL.md"));

        std::fs::remove_dir_all(root).expect("test dir cleans up");
    }

    #[test]
    fn skills_system_message_is_absent_without_skill_files() {
        let root =
            std::env::temp_dir().join(format!("seekcode-empty-skills-test-{}", MessageId::new()));
        std::fs::create_dir_all(root.join("skills")).expect("skills dir creates");

        let message = build_skills_system_message_for_dir(&root.join("skills"));
        assert!(message.is_none());

        std::fs::remove_dir_all(root).expect("test dir cleans up");
    }

    #[test]
    fn generated_session_title_is_normalized() {
        assert_eq!(
            normalize_generated_session_title("1. “修复工具调用显示”\n说明"),
            "修复工具调用显示"
        );
    }
}
