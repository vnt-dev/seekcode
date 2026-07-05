//! Application kernel: the public API surface exposed to thin adapters (Tauri,
//! future CLI). Composes services and orchestrates agent tasks, workspaces, and
//! sessions on top of `SessionService`.

use crate::events::spawn_session_agent_event_bridge;
use crate::session_service::{SessionService, SessionTaskContextPreparer};
use crate::title::generate_session_title;
use crate::tool_call_display::hydrate_session_message_tool_call_displays;
use crate::{
    AppKernelConfig, AppServices, CreateSessionRequest, OpenWorkspaceRequest, SessionTitleChanged,
    StartedAgentTask, WorkspaceWithSessions,
};
use parking_lot::RwLock;
use seekcode_agent_core::{Agent, AgentContextPreparer, StartTaskRequest};
use seekcode_common::{SeekCodeError, SeekCodeResult, SessionId, TaskId, WorkspaceId};
use seekcode_deepseek_client::{DeepSeekClient, DeepSeekConfig, ModelProvider};
use seekcode_storage::{
    NewWorkspace, SessionMessageRecord, SessionModelCallStats, SessionRecord, Storage,
    WorkspaceRecord,
};
use seekcode_tool_system::ToolRegistry;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Application kernel exposed to thin adapters.
pub struct AppKernel {
    config: RwLock<AppKernelConfig>,
    services: AppServices,
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
        let provider: Arc<dyn ModelProvider> =
            Arc::new(DeepSeekClient::new(config.deepseek.clone())?);
        let provider_slot = Arc::new(RwLock::new(provider.clone()));
        let tools = Arc::new(ToolRegistry::new());
        let agent = Arc::new(Agent::new(
            config.agent.clone(),
            provider.clone(),
            tools.clone(),
        ));
        let sessions = Arc::new(SessionService::new(storage.clone()));

        Ok(Self {
            config: RwLock::new(config),
            services: AppServices {
                provider: provider_slot,
                agent,
                tools,
                storage,
                sessions,
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
            Arc::new(SessionTaskContextPreparer::new(
                self.services.sessions.clone(),
                self.services.provider.read().clone(),
            ));
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
        let mut records = self
            .storage()?
            .list_session_messages_page(session_id, before_turn_sequence, limit)
            .await?;
        hydrate_session_message_tool_call_displays(&mut records);
        Ok(records)
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
mod tests {
    use super::*;
    use crate::context::{build_skills_system_message_for_dir, SKILLS_SYSTEM_PREFIX};
    use crate::test_support::{seed_session, seed_user_turn, CapturingProvider, StreamingProvider};
    use seekcode_agent_core::AgentEvent;
    use seekcode_common::{ChatRole, TokenUsage};
    use seekcode_deepseek_client::ChatChunk;
    use seekcode_storage::{NewSession, NewWorkspace, SessionStore, SqliteStorage, WorkspaceStore};

    #[test]
    fn app_kernel_can_be_constructed() {
        let kernel = AppKernel::new(AppKernelConfig::default()).expect("kernel builds");

        assert_eq!(kernel.config().agent.default_model, "deepseek-v4-pro");
        assert_eq!(kernel.config().title_model, "deepseek-v4-flash");
    }

    #[tokio::test]
    async fn app_kernel_updates_deepseek_config() {
        let kernel = AppKernel::new(AppKernelConfig::default()).expect("kernel builds");
        let deepseek = DeepSeekConfig {
            base_url: "https://example.test".to_string(),
            api_key: Some("sk-test".to_string()),
            ..Default::default()
        };

        kernel
            .update_deepseek_config(deepseek)
            .await
            .expect("deepseek config updates");

        let config = kernel.config();
        assert_eq!(config.deepseek.base_url, "https://example.test");
        assert_eq!(config.deepseek.api_key.as_deref(), Some("sk-test"));
    }

    #[tokio::test]
    async fn start_agent_task_persists_completed_model_round_messages_and_usage() {
        let storage = Arc::new(
            SqliteStorage::connect("sqlite::memory:")
                .await
                .expect("storage connects"),
        );
        let workspace_id = WorkspaceId::new();
        let session_id = SessionId::new();
        let workspace_root =
            std::env::temp_dir().join(format!("seekcode-round-persist-test-{workspace_id}"));
        std::fs::create_dir_all(&workspace_root).expect("workspace dir creates");
        std::fs::write(workspace_root.join("fixture.txt"), "needle\n").expect("fixture writes");
        storage
            .create_workspace(NewWorkspace {
                id: workspace_id,
                name: "SeekCode".to_string(),
                absolute_path: workspace_root.to_string_lossy().to_string(),
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

        let mut rounds = std::collections::VecDeque::new();
        rounds.push_back(vec![
            ChatChunk::Choice(seekcode_deepseek_client::ChatChoiceChunk {
                delta: seekcode_deepseek_client::ChatDelta {
                    content: None,
                    reasoning_content: Some("checking files".to_string()),
                    tool_calls: vec![seekcode_deepseek_client::ToolCallDelta {
                        index: 0,
                        id: Some("call_search".to_string()),
                        kind: Some("function".to_string()),
                        name: Some(seekcode_tool_system::RUN_COMMAND_TOOL.to_string()),
                        arguments: Some(r#"{"command":"echo fixture.txt"}"#.to_string()),
                    }],
                },
                finish_reason: Some("tool_calls".to_string()),
            }),
            ChatChunk::Usage(TokenUsage {
                prompt_tokens: 21,
                completion_tokens: 9,
                total_tokens: 30,
                cached_tokens: 4,
            }),
            ChatChunk::Finished,
        ]);
        rounds.push_back(vec![
            ChatChunk::Content("done".to_string()),
            ChatChunk::Usage(TokenUsage {
                prompt_tokens: 31,
                completion_tokens: 5,
                total_tokens: 36,
                cached_tokens: 2,
            }),
            ChatChunk::Finished,
        ]);

        let kernel = AppKernel::with_storage(AppKernelConfig::default(), storage.clone())
            .expect("kernel builds");
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(StreamingProvider {
            rounds: std::sync::Mutex::new(rounds),
            requests,
        });
        kernel.services.agent.set_provider(provider.clone());
        *kernel.services.provider.write() = provider;

        let mut started = kernel
            .start_agent_task(StartTaskRequest {
                session_id,
                prompt: "Find needle".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            })
            .await
            .expect("task starts");

        while let Some(event) = started.events.recv().await {
            if matches!(event, AgentEvent::Finished { task_id, .. } if task_id == started.task.id) {
                break;
            }
        }

        let messages = storage
            .list_session_messages(session_id)
            .await
            .expect("messages list");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, ChatRole::User);
        assert_eq!(messages[1].role, ChatRole::Assistant);
        assert_eq!(
            messages[1].reasoning_content.as_deref(),
            Some("checking files")
        );
        assert_eq!(messages[1].tool_calls.len(), 1);
        assert_eq!(messages[2].role, ChatRole::Tool);
        assert!(messages[2].tool_call_id.is_some());
        assert!(messages[2].content.contains("fixture.txt"));
        assert_eq!(messages[3].role, ChatRole::Assistant);
        assert_eq!(messages[3].content, "done");

        let stats = kernel
            .session_model_call_stats(session_id)
            .await
            .expect("stats load");
        assert_eq!(stats.call_count, 2);
        assert_eq!(stats.input_tokens, 52);
        assert_eq!(stats.output_tokens, 14);
        assert_eq!(stats.cache_hit_tokens, 6);
        assert_eq!(
            kernel
                .session_context_usage(session_id)
                .await
                .expect("context usage loads"),
            31
        );

        std::fs::remove_dir_all(workspace_root).expect("workspace dir cleans up");
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
        let root =
            std::env::temp_dir().join(format!("seekcode-skills-test-{}", WorkspaceId::new()));
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
            std::env::temp_dir().join(format!("seekcode-empty-skills-test-{}", WorkspaceId::new()));
        std::fs::create_dir_all(root.join("skills")).expect("skills dir creates");

        let message = build_skills_system_message_for_dir(&root.join("skills"));
        assert!(message.is_none());

        std::fs::remove_dir_all(root).expect("test dir cleans up");
    }
}
