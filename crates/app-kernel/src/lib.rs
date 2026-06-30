//! Application-level service composition for the Tauri adapter and future CLI.

use chrono::Local;
use parking_lot::RwLock;
use seekcode_agent_core::{Agent, AgentConfig, AgentEvent, AgentTask, StartTaskRequest};
use seekcode_common::{
    ChatMessage, ChatRole, MessageId, ModelCallLogId, SeekCodeError, SeekCodeResult, SessionId,
    TaskId, TokenUsage, ToolCallId, WorkspaceId,
};
use seekcode_deepseek_client::{DeepSeekClient, DeepSeekConfig};
use seekcode_model_provider::{ChatRequest, ModelProvider};
use seekcode_policy::{AutonomousPolicy, PolicyEngine};
use seekcode_secrets::{InMemorySecretStore, SecretStore};
use seekcode_shell_sandbox::{CommandRunner, SandboxPolicy};
use seekcode_storage::{
    local_now_text, NewModelCallLog, NewSession, NewSessionMessage, NewWorkspace,
    SessionMessageRecord, SessionRecord, Storage, WorkspaceRecord,
};
use seekcode_telemetry::{init_tracing, TelemetryConfig};
use seekcode_tool_system::{system_tool_registry, SystemToolConfig, ToolRegistry};
use seekcode_workspace::{FileEntry, FileSnapshot, ListOptions, WorkspaceRoot, WorkspaceService};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

const CONTEXT_SHELL: &str = "powershell";
const SKILLS_SYSTEM_PREFIX: &str = "Skills\nA skill is a set of instructions provided through a `SKILL.md` source. Below is the list of skills that can be used. Each entry includes a name, description, and source locator. `file` locators are on the host filesystem, `environment resource` locators are owned by an execution environment, `orchestrator resource` locators are opaque non-filesystem resources, and `custom resource` locators use their provider's access mechanism.\n### Available skills";

pub use seekcode_storage::SessionRecord as AppSessionRecord;

/// Workspace plus nested sessions for the sidebar.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceWithSessions {
    /// Workspace metadata.
    pub workspace: WorkspaceRecord,
    /// Sessions that belong to the workspace.
    pub sessions: Vec<SessionRecord>,
}

/// Request to open or create a workspace from the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenWorkspaceRequest {
    /// Human-readable workspace name.
    pub name: Option<String>,
    /// Absolute workspace path.
    pub absolute_path: String,
}

/// Request to create a new persisted session.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    /// Parent workspace identifier.
    pub workspace_id: WorkspaceId,
    /// Optional session name.
    pub name: Option<String>,
    /// Model provider name.
    pub model_provider: Option<String>,
    /// Model identifier.
    pub model: Option<String>,
    /// Whether thinking is enabled.
    pub thinking_enabled: Option<bool>,
    /// Optional provider-specific reasoning intensity.
    pub reasoning_effort: Option<String>,
}

/// App kernel configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppKernelConfig {
    /// DeepSeek provider configuration.
    #[serde(skip)]
    pub deepseek: DeepSeekConfig,
    /// Agent runtime configuration.
    pub agent: AgentConfig,
    /// Telemetry setup.
    pub telemetry: TelemetryConfig,
    /// Shell sandbox settings.
    pub shell: SandboxPolicy,
    /// Fast model used to generate empty session titles.
    pub title_model: String,
}

impl Default for AppKernelConfig {
    fn default() -> Self {
        Self {
            deepseek: DeepSeekConfig::default(),
            agent: AgentConfig::default(),
            telemetry: TelemetryConfig::default(),
            shell: SandboxPolicy::default(),
            title_model: "deepseek-v4-flash".to_string(),
        }
    }
}

/// Concrete services assembled for the application.
pub struct AppServices {
    /// Model provider used by the agent.
    pub provider: Arc<RwLock<Arc<dyn ModelProvider>>>,
    /// Agent runtime.
    pub agent: Arc<Agent>,
    /// Tool registry.
    pub tools: Arc<ToolRegistry>,
    /// Workspace service.
    pub workspace: Arc<WorkspaceService>,
    /// Policy engine.
    pub policy: Arc<dyn PolicyEngine>,
    /// Optional durable storage.
    pub storage: Option<Arc<dyn Storage>>,
    /// Session-level application service.
    pub sessions: Arc<SessionService>,
    /// Secret storage.
    pub secrets: Arc<dyn SecretStore>,
    /// Command runner.
    pub shell: Arc<CommandRunner>,
}

/// Session service owns persisted conversation state and session-scoped agent events.
pub struct SessionService {
    storage: Option<Arc<dyn Storage>>,
    running_sessions: RwLock<HashSet<SessionId>>,
    title_sessions: RwLock<HashSet<SessionId>>,
}

/// Started agent task plus the UI event stream for that task.
pub struct StartedAgentTask {
    /// Started task snapshot.
    pub task: AgentTask,
    /// Session-processed events ready for UI forwarding.
    pub events: mpsc::UnboundedReceiver<AgentEvent>,
}

/// Notification emitted when a background title task updates one session.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionTitleChanged {
    /// Session whose title changed.
    pub session_id: SessionId,
    /// Generated display title.
    pub title: String,
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

    fn finish_running_task(&self, session_id: SessionId) {
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

    async fn persist_assistant_choice(
        &self,
        session_id: SessionId,
        turn_sequence: i64,
        choice: &seekcode_model_provider::ChatChoiceChunk,
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

    async fn persist_tool_result(
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

    async fn assemble_context(
        &self,
        session_id: SessionId,
        prompt: &str,
    ) -> SeekCodeResult<Vec<ChatMessage>> {
        let storage = self.storage()?;
        let session = storage.get_session(session_id).await?;
        let workspace = storage.get_workspace(session.workspace_id).await?;
        let records = storage.list_session_messages(session_id).await?;
        let mut messages = vec![ChatMessage::new(
            ChatRole::System,
            build_skills_system_message(),
        )];
        messages.push(ChatMessage::new(
            ChatRole::System,
                "You are SeekCode, a coding agent based on DeepSeek. You and the user share the same workspace and collaborate to achieve the user's goals.\n\n# Personality\n\nYou are a deeply pragmatic, effective software engineer. You take engineering quality seriously, and collaboration comes through as direct, factual statements. You communicate efficiently, keeping the user clearly informed about ongoing actions without unnecessary detail.\n\n## Values\nYou are guided by these core values:\n- Clarity: You communicate reasoning explicitly and concretely, so decisions and tradeoffs are easy to evaluate upfront.\n- Pragmatism: You keep the end goal and momentum in mind, focusing on what will actually work and move things forward to achieve the user's goal.\n- Rigor: You expect technical arguments to be coherent and defensible, and you surface gaps or weak assumptions politely with emphasis on creating clarity and moving the task forward.\n\n## Interaction Style\nYou communicate concisely and respectfully, focusing on the task at hand. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.\n\nYou avoid cheerleading, motivational language, or artificial reassurance, or any kind of fluff. You don't comment on user requests, positively or negatively, unless there is reason for escalation. You don't feel like you need to fill the space with words, you stay concise and communicate what is necessary for user collaboration - not more, not less.\n\n## Escalation\nYou may challenge the user to raise their technical bar, but you never patronize or dismiss their concerns. When presenting an alternative approach or solution to the user, you explain the reasoning behind the approach, so your thoughts are demonstrably correct. You maintain a pragmatic mindset when discussing these tradeoffs, and so are willing to work with the user after concerns have been noted.\n\n\n# General\nAs an expert coding agent, your primary focus is writing code, answering questions, and helping the user complete their task in the current environment. You build context by examining the codebase first without making assumptions or jumping to conclusions. You think through the nuances of the code you encounter, and embody the mentality of a skilled senior software engineer.\n\n- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. (If the `rg` command is not found, then use alternatives.)\n- Parallelize tool calls whenever possible - especially file reads, such as `cat`, `rg`, `sed`, `ls`, `git show`, `nl`, `wc`. Use `multi_tool_use.parallel` to parallelize tool calls and only this. Never chain together bash commands with separators like `echo \"====\";` as this renders to the user poorly.\n\n## Editing constraints\n\n- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.\n- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like \"Assigns the value to the variable\", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.\n- Always use apply_patch for manual code edits. Do not use cat or any other commands when creating or editing files. Formatting commands or bulk edits don't need to be done with apply_patch.\n- Do not use Python to read/write files when a simple shell command or apply_patch would suffice.\n- You may be in a dirty git worktree.\n  * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.\n  * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.\n  * If the changes are in files you've touched recently, you should read carefully and understand how you can work with the changes rather than reverting them.\n  * If the changes are in unrelated files, just ignore them and don't revert them.\n- Do not amend a commit unless explicitly requested to do so.\n- While you are working, you might notice unexpected changes that you didn't make. It's likely the user made them, or were autogenerated. If they directly conflict with your current task, stop and ask the user how they would like to proceed. Otherwise, focus on the task at hand.\n- **NEVER** use destructive commands like `git reset --hard` or `git checkout --` unless specifically requested or approved by the user.\n- You struggle using the git interactive console. **ALWAYS** prefer using non-interactive git commands.\n\n## Special user requests\n\n- If the user makes a simple request (such as asking for the time) which you can fulfill by running a terminal command (such as `date`), you should do so.\n- If the user asks for a \"review\", default to a code review mindset: prioritise identifying bugs, risks, behavioural regressions, and missing tests. Findings must be the primary focus of the response - keep summaries or overviews brief and only after enumerating the issues. Present findings first (ordered by severity with file/line references), follow with open questions or assumptions, and offer a change-summary only as a secondary detail. If no findings are discovered, state that explicitly and mention any residual risks or testing gaps.\n\n## Autonomy and persistence\nPersist until the task is fully handled end-to-end within the current turn whenever feasible: do not stop at analysis or partial fixes; carry changes through implementation, verification, and a clear explanation of outcomes unless the user explicitly pauses or redirects you.\n\nUnless the user explicitly asks for a plan, asks a question about the code, is brainstorming potential solutions, or some other intent that makes it clear that code should not be written, assume the user wants you to make code changes or run tools to solve the user's problem. In these cases, it's bad to output your proposed solution in a message, you should go ahead and actually implement the change. If you encounter challenges or blockers, you should attempt to resolve them yourself.\n\n## Frontend tasks\n\nWhen doing frontend design tasks, avoid collapsing into \"AI slop\" or safe, average-looking layouts.\nAim for interfaces that feel intentional, bold, and a bit surprising.\n- Typography: Use expressive, purposeful fonts and avoid default stacks (Inter, Roboto, Arial, system).\n- Color & Look: Choose a clear visual direction; define CSS variables; avoid purple-on-white defaults. No purple bias or dark mode bias.\n- Motion: Use a few meaningful animations (page-load, staggered reveals) instead of generic micro-motions.\n- Background: Don't rely on flat, single-color backgrounds; use gradients, shapes, or subtle patterns to build atmosphere.\n- Ensure the page loads properly on both desktop and mobile\n- For React code, prefer modern patterns including useEffectEvent, startTransition, and useDeferredValue when appropriate if used by the team. Do not add useMemo/useCallback by default unless already used; follow the repo's React Compiler guidance.\n- Overall: Avoid boilerplate layouts and interchangeable UI patterns. Vary themes, type families, and visual languages across outputs.\n\nException: If working within an existing website or design system, preserve the established patterns, structure, and visual language.\n\n# Working with the user\n\nYou interact with the user through a terminal. You have 2 ways of communicating with the users:\n- Share intermediary updates in `commentary` channel. \n- After you have completed all your work, send a message to the `final` channel.\nYou are producing plain text that will later be styled by the program you run in. Formatting should make results easy to scan, but not feel mechanical. Use judgment to decide how much structure adds value. Follow the formatting rules exactly.\n\n## Formatting rules\n\n- You may format with GitHub-flavored Markdown.\n- Structure your answer if necessary, the complexity of the answer should match the task. If the task is simple, your answer should be a one-liner. Order sections from general to specific to supporting.\n- Never use nested bullets. Keep lists flat (single level). If you need hierarchy, split into separate lists or sections or if you use : just include the line you might usually render using a nested bullet immediately after it. For numbered lists, only use the `1. 2. 3.` style markers (with a period), never `1)`.\n- Headers are optional, only use them when you think they are necessary. If you do use them, use short Title Case (1-3 words) wrapped in **…**. Don't add a blank line.\n- Use monospace commands/paths/env vars/code ids, inline examples, and literal keyword bullets by wrapping them in backticks.\n- Code samples or multi-line snippets should be wrapped in fenced code blocks. Include an info string as often as possible.\n- File References: When referencing files in your response follow the below rules:\n  * Use markdown links (not inline code) for clickable file paths.\n  * Each reference should have a stand alone path. Even if it's the same file.\n  * For clickable/openable file references, the path target must be an absolute filesystem path. Labels may be short (for example, `[app.ts](/abs/path/app.ts)`).\n  * Optionally include line/column (1‑based): :line[:column] or #Lline[Ccolumn] (column defaults to 1).\n  * Do not use URIs like file://, vscode://, or https://.\n  * Do not provide range of lines\n- Don’t use emojis or em dashes unless explicitly instructed.\n\n## Final answer instructions\n\n- Balance conciseness to not overwhelm the user with appropriate detail for the request. Do not narrate abstractly; explain what you are doing and why.\n- Do not begin responses with conversational interjections or meta commentary. Avoid openers such as acknowledgements (“Done —”, “Got it”, “Great question, ”) or framing phrases.\n- The user does not see command execution outputs. When asked to show the output of a command (e.g. `git show`), relay the important details in your answer or summarize the key lines so the user understands the result.\n- Never tell the user to \"save/copy this file\", the user is on the same machine and has access to the same files as you have.\n- If the user asks for a code explanation, structure your answer with code references.\n- When given a simple task, just provide the outcome in a short answer without strong formatting.\n- When you make big or complex changes, state the solution first, then walk the user through what you did and why.\n- For casual chit-chat, just chat.\n- If you weren't able to do something, for example run tests, tell the user.\n- If there are natural next steps the user may want to take, suggest them at the end of your response. Do not make suggestions if there are no natural next steps. When suggesting multiple options, use numeric lists for the suggestions so the user can quickly respond with a single number.\n\n## Intermediary updates \n\n- Intermediary updates go to the `commentary` channel.\n- User updates are short updates while you are working, they are NOT final answers.\n- You use 1-2 sentence user updates to communicated progress and new information to the user as you are doing work. \n- Do not begin responses with conversational interjections or meta commentary. Avoid openers such as acknowledgements (“Done —”, “Got it”, “Great question, ”) or framing phrases.\n- Before exploring or doing substantial work, you start with a user update acknowledging the request and explaining your first step. You should include your understanding of the user request and explain what you will do. Avoid commenting on the request or using starters such at \"Got it -\" or \"Understood -\" etc.\n- You provide user updates frequently, every 30s.\n- When exploring, e.g. searching, reading files you provide user updates as you go, explaining what context you are gathering and what you've learned. Vary your sentence structure when providing these updates to avoid sounding repetitive - in particular, don't start each sentence the same way.\n- When working for a while, keep updates informative and varied, but stay concise.\n- After you have sufficient context, and the work is substantial you provide a longer plan (this is the only user update that may be longer than 2 sentences and can contain formatting).\n- Before performing file edits of any kind, you provide updates explaining what edits you are making.\n- As you are thinking, you very frequently provide updates even if not taking any actions, informing the user of your progress. You interrupt your thinking and send multiple updates in a row if thinking for more than 100 words.\n- Tone of your updates MUST match your personality.\n",
        ));

        let mut pending_assistant = PendingAssistantContext::default();
        for record in records {
            tracing::debug!(
                target: "seekcode_app_kernel::context",
                %session_id,
                message_id = %record.id,
                turn_sequence = record.turn_sequence,
                role = ?record.role,
                content_len = record.content.len(),
                reasoning_len = record.reasoning_content.as_deref().map(str::len).unwrap_or(0),
                tool_call_count = record.tool_calls.len(),
                tool_call_id = ?record.tool_call_id,
                tool_calls = %serde_json::Value::Array(record.tool_calls.clone()),
                "adding persisted message to agent context"
            );

            if record.role == ChatRole::Assistant {
                pending_assistant.apply_record(session_id, record);
                continue;
            }

            pending_assistant.flush(session_id, &mut messages);
            push_record_as_context_message(record, &mut messages);
        }
        pending_assistant.flush(session_id, &mut messages);

        messages.push(ChatMessage::new(
            ChatRole::User,
            build_environment_context(&workspace.absolute_path),
        ));
        messages.push(ChatMessage::new(ChatRole::User, prompt));

        Ok(messages)
    }

    async fn system_tool_registry_for_session(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<ToolRegistry> {
        let storage = self.storage()?;
        let session = storage.get_session(session_id).await?;
        let workspace = storage.get_workspace(session.workspace_id).await?;
        system_tool_registry(SystemToolConfig::new(workspace.absolute_path)?)
    }

    async fn get_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>> {
        self.storage()?.list_sessions().await
    }

    async fn get_session(&self, session_id: SessionId) -> SeekCodeResult<SessionRecord> {
        self.storage()?.get_session(session_id).await
    }

    async fn rename_session(
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
            .update_session_model(session_id, model_provider, model)
            .await
    }

    async fn append_model_call_log(&self, log: NewModelCallLog) {
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
                reasoning_effort: request.reasoning_effort,
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

    async fn list_session_messages(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<Vec<SessionMessageRecord>> {
        self.storage()?.list_session_messages(session_id).await
    }

    fn storage(&self) -> SeekCodeResult<&Arc<dyn Storage>> {
        self.storage
            .as_ref()
            .ok_or(SeekCodeError::NotImplemented("storage is not wired yet"))
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
        let workspace = Arc::new(WorkspaceService::new());
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
                workspace,
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
        self.services.sessions.start_running_task(session_id)?;
        let tools = match self
            .services
            .sessions
            .system_tool_registry_for_session(session_id)
            .await
        {
            Ok(tools) => Arc::new(tools),
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
                return Err(error);
            }
        };

        let context = match self
            .services
            .sessions
            .assemble_context(session_id, prompt)
            .await
        {
            Ok(context) => context,
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
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
                return Err(error);
            }
        };
        let (agent_events, agent_event_receiver) = mpsc::unbounded_channel();
        let (ui_events, ui_event_receiver) = mpsc::unbounded_channel();
        let task = match self
            .services
            .agent
            .start_task_with_messages_and_tools(request, context, tools, agent_events)
            .await
        {
            Ok(task) => task,
            Err(error) => {
                self.services.sessions.finish_running_task(session_id);
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

    /// Reads one workspace file.
    pub async fn read_file(
        &self,
        root: WorkspaceRoot,
        path: PathBuf,
    ) -> SeekCodeResult<FileSnapshot> {
        self.services.workspace.read_file(&root, path).await
    }

    /// Lists workspace files.
    pub async fn list_files(
        &self,
        root: WorkspaceRoot,
        options: ListOptions,
    ) -> SeekCodeResult<Vec<FileEntry>> {
        self.services.workspace.list_tree(&root, options).await
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
    ) -> SeekCodeResult<SessionRecord> {
        self.services
            .sessions
            .update_session_model(session_id, model_provider, model)
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
    ) -> SeekCodeResult<Vec<SessionMessageRecord>> {
        self.services
            .sessions
            .list_session_messages(session_id)
            .await
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

async fn generate_session_title(
    sessions: Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
    session_id: SessionId,
    title_model: String,
    prompt: String,
) -> SeekCodeResult<Option<SessionTitleChanged>> {
    let session = sessions.get_session(session_id).await?;
    if !session.name.trim().is_empty() {
        return Ok(None);
    }

    let called_at = local_now_text();
    let started_at = Instant::now();
    let model = title_model.clone();
    let response = provider
        .complete_chat(ChatRequest {
            model: title_model.clone(),
            messages: vec![
                ChatMessage::new(
                    ChatRole::System,
                    "Generate a concise chat title from the user's first message. Return only the title, with no quotes, punctuation wrapper, or explanation. Use the same language as the user. Keep it under 18 Chinese characters or 8 English words.",
                ),
                ChatMessage::new(ChatRole::User, prompt),
            ],
            tools: Vec::new(),
            thinking: false,
            strict_tools: false,
        })
        .await;
    let elapsed_ms = started_at.elapsed().as_millis().min(i64::MAX as u128) as i64;
    match &response {
        Ok(response) => {
            sessions
                .append_model_call_log(new_model_call_log(
                    session.model_provider.clone(),
                    model,
                    session_id,
                    response.usage.clone(),
                    elapsed_ms,
                    true,
                    called_at,
                ))
                .await;
        }
        Err(_) => {
            sessions
                .append_model_call_log(new_model_call_log(
                    session.model_provider,
                    model,
                    session_id,
                    None,
                    elapsed_ms,
                    false,
                    called_at,
                ))
                .await;
        }
    }

    let response = response?;

    let title = normalize_generated_session_title(&response.content);
    if title.is_empty() {
        return Ok(None);
    }

    let session = sessions.get_session(session_id).await?;
    if session.name.trim().is_empty() {
        let session = sessions.rename_session(session_id, title).await?;
        return Ok(Some(SessionTitleChanged {
            session_id,
            title: session.name,
        }));
    }

    Ok(None)
}

fn new_model_call_log(
    model_provider: String,
    model: String,
    session_id: SessionId,
    usage: Option<TokenUsage>,
    elapsed_ms: i64,
    success: bool,
    called_at: String,
) -> NewModelCallLog {
    let usage = usage.unwrap_or_default();
    NewModelCallLog {
        id: ModelCallLogId::new(),
        model_provider,
        model,
        session_id,
        input_tokens: i64::from(usage.prompt_tokens),
        output_tokens: i64::from(usage.completion_tokens),
        cache_hit_tokens: i64::from(usage.cached_tokens),
        elapsed_ms,
        success,
        called_at,
    }
}

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

fn build_skills_system_message() -> String {
    match seekcode_skills_dir() {
        Some(skills_dir) => build_skills_system_message_for_dir(&skills_dir),
        None => SKILLS_SYSTEM_PREFIX.to_string(),
    }
}

fn seekcode_skills_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .map(|home| home.join(".seekcode").join("skills"))
}

fn build_skills_system_message_for_dir(skills_dir: &Path) -> String {
    let mut skill_paths = Vec::new();
    collect_skill_paths(skills_dir, &mut skill_paths);
    skill_paths.sort();

    let mut message = SKILLS_SYSTEM_PREFIX.to_string();
    for path in skill_paths {
        message.push('\n');
        message.push_str(&skill_entry_from_path(&path));
    }

    message
}

fn collect_skill_paths(path: &Path, skill_paths: &mut Vec<PathBuf>) {
    let Ok(metadata) = std::fs::metadata(path) else {
        return;
    };

    if metadata.is_file() {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("skill.md"))
        {
            skill_paths.push(path.to_path_buf());
        }
        return;
    }

    if !metadata.is_dir() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        collect_skill_paths(&entry.path(), skill_paths);
    }
}

fn skill_entry_from_path(path: &Path) -> String {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let name = extract_frontmatter_value(&content, "name")
        .or_else(|| {
            path.parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_string());
    let description = extract_frontmatter_value(&content, "description")
        .unwrap_or_else(|| "No description provided.".to_string());
    let source = path.to_string_lossy();

    format!("- {name}: {description} (file: {source})")
}

fn extract_frontmatter_value(content: &str, key: &str) -> Option<String> {
    let mut lines = content.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }

    let prefix = format!("{key}:");
    for line in lines {
        let line = line.trim();
        if line == "---" {
            break;
        }
        if let Some(value) = line.strip_prefix(&prefix) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(unquote_frontmatter_value(value));
            }
        }
    }

    None
}

fn unquote_frontmatter_value(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 {
        let first = value.as_bytes()[0] as char;
        let last = value.as_bytes()[value.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return value[1..value.len() - 1].to_string();
        }
    }

    value.to_string()
}

fn build_environment_context(cwd: &str) -> String {
    let cwd = escape_environment_context_text(cwd);
    let shell = escape_environment_context_text(CONTEXT_SHELL);
    let now = Local::now();
    let current_date = now.format("%Y-%m-%d").to_string();
    let timezone = escape_environment_context_text(&current_timezone_name(&now));

    format!(
        "<environment_context>\n  <cwd>{cwd}</cwd>\n  <shell>{shell}</shell>\n  <current_date>{current_date}</current_date>\n  <timezone>{timezone}</timezone>\n  <filesystem><workspace_roots><root>{cwd}</root></workspace_roots></filesystem>\n</environment_context>"
    )
}

fn current_timezone_name(now: &chrono::DateTime<Local>) -> String {
    iana_time_zone::get_timezone().unwrap_or_else(|_| now.format("%:z").to_string())
}

fn escape_environment_context_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn push_record_as_context_message(record: SessionMessageRecord, messages: &mut Vec<ChatMessage>) {
    let mut message = ChatMessage::new(record.role, record.content);
    message.id = record.id;
    message.reasoning_content = record.reasoning_content;
    message.tool_calls = record.tool_calls;
    message.tool_call_id = record.tool_call_id;
    messages.push(message);
}

#[derive(Default)]
struct PendingAssistantContext {
    content: String,
    reasoning_content: String,
    tool_calls: PersistedToolCallAccumulator,
}

impl PendingAssistantContext {
    fn apply_record(&mut self, session_id: SessionId, record: SessionMessageRecord) {
        self.content.push_str(&record.content);
        if let Some(reasoning_content) = record.reasoning_content {
            self.reasoning_content.push_str(&reasoning_content);
        }
        if !record.tool_calls.is_empty() {
            self.tool_calls
                .apply(session_id, record.id, record.tool_calls);
        }
    }

    fn flush(&mut self, session_id: SessionId, messages: &mut Vec<ChatMessage>) {
        let tool_calls = self.tool_calls.take_completed(session_id);
        if self.content.is_empty() && self.reasoning_content.is_empty() && tool_calls.is_empty() {
            return;
        }

        tracing::debug!(
            target: "seekcode_app_kernel::context",
            %session_id,
            content_len = self.content.len(),
            reasoning_len = self.reasoning_content.len(),
            tool_call_count = tool_calls.len(),
            tool_calls = %serde_json::Value::Array(tool_calls.clone()),
            "flushing persisted assistant deltas into context message"
        );

        let mut message = ChatMessage::new(ChatRole::Assistant, std::mem::take(&mut self.content));
        if !self.reasoning_content.is_empty() {
            message.reasoning_content = Some(std::mem::take(&mut self.reasoning_content));
        }
        message.tool_calls = tool_calls;
        messages.push(message);
    }
}

#[derive(Default)]
struct PersistedToolCallAccumulator {
    partials: BTreeMap<String, PersistedToolCallPartial>,
}

impl PersistedToolCallAccumulator {
    fn apply(
        &mut self,
        session_id: SessionId,
        message_id: MessageId,
        tool_calls: Vec<serde_json::Value>,
    ) {
        for tool_call in tool_calls {
            let Some(id) = tool_call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
            else {
                tracing::warn!(
                    target: "seekcode_app_kernel::context",
                    %session_id,
                    %message_id,
                    tool_call = %tool_call,
                    "skipping persisted tool call delta without id"
                );
                continue;
            };

            let partial =
                self.partials
                    .entry(id.clone())
                    .or_insert_with(|| PersistedToolCallPartial {
                        id,
                        kind: "function".to_string(),
                        name: None,
                        arguments: String::new(),
                    });

            if let Some(kind) = tool_call
                .get("type")
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
            {
                partial.kind = kind.to_string();
            }

            let function = tool_call
                .get("function")
                .and_then(serde_json::Value::as_object);
            if let Some(name) = function
                .and_then(|value| value.get("name"))
                .and_then(serde_json::Value::as_str)
                .filter(|value| !value.is_empty())
            {
                partial.name = Some(name.to_string());
            }
            if let Some(arguments) = function
                .and_then(|value| value.get("arguments"))
                .and_then(serde_json::Value::as_str)
            {
                partial.arguments.push_str(arguments);
            }
        }
    }

    fn take_completed(&mut self, session_id: SessionId) -> Vec<serde_json::Value> {
        if self.partials.is_empty() {
            return Vec::new();
        }

        let mut tool_calls = Vec::new();
        for partial in std::mem::take(&mut self.partials).into_values() {
            let Some(name) = partial.name else {
                tracing::warn!(
                    target: "seekcode_app_kernel::context",
                    %session_id,
                    tool_call_id = %partial.id,
                    arguments_len = partial.arguments.len(),
                    "dropping incomplete persisted tool call without function name"
                );
                continue;
            };

            tool_calls.push(serde_json::json!({
                "id": partial.id,
                "type": partial.kind,
                "function": {
                    "name": name,
                    "arguments": partial.arguments
                }
            }));
        }

        tool_calls
    }
}

struct PersistedToolCallPartial {
    id: String,
    kind: String,
    name: Option<String>,
    arguments: String,
}

fn choice_tool_calls_json(
    choice: &seekcode_model_provider::ChatChoiceChunk,
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

fn spawn_session_agent_event_bridge(
    sessions: Arc<SessionService>,
    mut events: mpsc::UnboundedReceiver<AgentEvent>,
    ui_events: mpsc::UnboundedSender<AgentEvent>,
    task_id: TaskId,
    session_id: SessionId,
    turn_sequence: i64,
) {
    tokio::spawn(async move {
        let mut content = String::new();
        let mut reasoning = String::new();
        let model_provider = sessions
            .get_session(session_id)
            .await
            .map(|session| session.model_provider)
            .unwrap_or_else(|_| "deepseek".to_string());
        let mut active_model_call: Option<PendingModelCallLog> = None;

        loop {
            match events.recv().await {
                Some(AgentEvent::ModelChoice {
                    session_id: event_session_id,
                    task_id: event_task_id,
                    round_id,
                    choice,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    if let Some(text) = &choice.delta.content {
                        content.push_str(text);
                    }
                    if let Some(text) = &choice.delta.reasoning_content {
                        reasoning.push_str(text);
                    }
                    sessions
                        .persist_assistant_choice(session_id, turn_sequence, &choice)
                        .await;
                    let _ = ui_events.send(AgentEvent::ModelChoice {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        choice,
                    });
                }
                Some(AgentEvent::AssistantToken {
                    task_id: event_task_id,
                    session_id: event_session_id,
                    round_id,
                    text,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    content.push_str(&text);
                    let _ = ui_events.send(AgentEvent::AssistantToken {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        text,
                    });
                }
                Some(AgentEvent::AssistantReasoning {
                    task_id: event_task_id,
                    session_id: event_session_id,
                    round_id,
                    text,
                    ..
                }) if event_task_id == task_id && event_session_id == session_id => {
                    reasoning.push_str(&text);
                    let _ = ui_events.send(AgentEvent::AssistantReasoning {
                        session_id: event_session_id,
                        task_id: event_task_id,
                        round_id,
                        text,
                    });
                }
                Some(event)
                    if agent_event_task_id(&event) == Some(task_id)
                        && agent_event_session_id(&event) == Some(session_id)
                        && is_terminal_agent_event(&event) =>
                {
                    if let Some(call) = active_model_call.take() {
                        let success = matches!(event, AgentEvent::Finished { .. });
                        sessions
                            .append_model_call_log(call.into_new_log(
                                model_provider.clone(),
                                session_id,
                                success,
                                None,
                            ))
                            .await;
                    }
                    let _ = ui_events.send(event);
                    break;
                }
                Some(event)
                    if agent_event_task_id(&event) == Some(task_id)
                        && agent_event_session_id(&event) == Some(session_id) =>
                {
                    match &event {
                        AgentEvent::ModelRequestStarted {
                            round_id, model, ..
                        } => {
                            active_model_call = Some(PendingModelCallLog {
                                round_id: *round_id,
                                model: model.clone(),
                                started_at: Instant::now(),
                                called_at: local_now_text(),
                            });
                        }
                        AgentEvent::ModelRoundFinished {
                            round_id, usage, ..
                        } => {
                            if active_model_call
                                .as_ref()
                                .is_some_and(|call| call.round_id == *round_id)
                            {
                                let call = active_model_call.take().expect("active call exists");
                                sessions
                                    .append_model_call_log(call.into_new_log(
                                        model_provider.clone(),
                                        session_id,
                                        true,
                                        usage.clone(),
                                    ))
                                    .await;
                            }
                        }
                        AgentEvent::ToolCallFinished {
                            tool_call_id,
                            output,
                            error,
                            ..
                        } => {
                            sessions
                                .persist_tool_result(
                                    session_id,
                                    turn_sequence,
                                    *tool_call_id,
                                    output.clone(),
                                    error.clone(),
                                )
                                .await;
                        }
                        _ => {}
                    }
                    let _ = ui_events.send(event);
                }
                Some(_) => {}
                None => break,
            }
        }

        sessions.finish_running_task(session_id);
    });
}

struct PendingModelCallLog {
    round_id: u32,
    model: String,
    started_at: Instant,
    called_at: String,
}

impl PendingModelCallLog {
    fn into_new_log(
        self,
        model_provider: String,
        session_id: SessionId,
        success: bool,
        usage: Option<TokenUsage>,
    ) -> NewModelCallLog {
        let usage = usage.unwrap_or_default();
        NewModelCallLog {
            id: ModelCallLogId::new(),
            model_provider,
            model: self.model,
            session_id,
            input_tokens: i64::from(usage.prompt_tokens),
            output_tokens: i64::from(usage.completion_tokens),
            cache_hit_tokens: i64::from(usage.cached_tokens),
            elapsed_ms: self.started_at.elapsed().as_millis().min(i64::MAX as u128) as i64,
            success,
            called_at: self.called_at,
        }
    }
}

fn agent_event_task_id(event: &AgentEvent) -> Option<TaskId> {
    match event {
        AgentEvent::TaskStarted { task_id, .. }
        | AgentEvent::StateChanged { task_id, .. }
        | AgentEvent::ModelRequestStarted { task_id, .. }
        | AgentEvent::ModelChoice { task_id, .. }
        | AgentEvent::AssistantToken { task_id, .. }
        | AgentEvent::AssistantReasoning { task_id, .. }
        | AgentEvent::ToolCallStarted { task_id, .. }
        | AgentEvent::ToolCallFinished { task_id, .. }
        | AgentEvent::ModelRoundFinished { task_id, .. }
        | AgentEvent::Finished { task_id, .. }
        | AgentEvent::Failed { task_id, .. }
        | AgentEvent::Canceled { task_id, .. } => Some(*task_id),
    }
}

fn agent_event_session_id(event: &AgentEvent) -> Option<SessionId> {
    match event {
        AgentEvent::TaskStarted { session_id, .. }
        | AgentEvent::StateChanged { session_id, .. }
        | AgentEvent::ModelRequestStarted { session_id, .. }
        | AgentEvent::ModelChoice { session_id, .. }
        | AgentEvent::AssistantToken { session_id, .. }
        | AgentEvent::AssistantReasoning { session_id, .. }
        | AgentEvent::ToolCallStarted { session_id, .. }
        | AgentEvent::ToolCallFinished { session_id, .. }
        | AgentEvent::ModelRoundFinished { session_id, .. }
        | AgentEvent::Finished { session_id, .. }
        | AgentEvent::Failed { session_id, .. }
        | AgentEvent::Canceled { session_id, .. } => Some(*session_id),
    }
}

fn is_terminal_agent_event(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::Finished { .. } | AgentEvent::Failed { .. } | AgentEvent::Canceled { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use seekcode_storage::{SessionStore, SqliteStorage, WorkspaceStore};

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

        storage
            .create_workspace(NewWorkspace {
                id: workspace_id,
                name: "SeekCode".to_string(),
                absolute_path: "D:\\rust\\tmp\\seekcode".to_string(),
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

        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, ChatRole::System);
        assert!(messages[0].content.starts_with("Skills\n"));
        assert_eq!(messages[1].role, ChatRole::System);
        assert_eq!(messages[2].content, "previous prompt");
        assert_eq!(messages[3].role, ChatRole::User);
        assert!(messages[3]
            .content
            .contains("<cwd>D:\\rust\\tmp\\seekcode</cwd>"));
        assert!(messages[3].content.contains("<shell>powershell</shell>"));
        assert!(messages[3].content.contains(&format!(
            "<timezone>{}</timezone>",
            current_timezone_name(&Local::now())
        )));
        assert!(messages[3]
            .content
            .contains("<workspace_roots><root>D:\\rust\\tmp\\seekcode</root></workspace_roots>"));
        assert_eq!(messages[4].role, ChatRole::User);
        assert_eq!(messages[4].content, "new prompt");
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

        let message = build_skills_system_message_for_dir(&root.join("skills"));
        assert!(message.starts_with(SKILLS_SYSTEM_PREFIX));
        assert!(message.contains("- sample-skill: Use when testing skill discovery. (file: "));
        assert!(message.contains("sample-skill"));
        assert!(message.contains("SKILL.md"));

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
