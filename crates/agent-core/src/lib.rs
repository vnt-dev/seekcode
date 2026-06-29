//! Agent state machine, task context, and execution loop scaffolding.

use futures_util::StreamExt;
use parking_lot::RwLock;
use seekcode_common::{
    ChatMessage, ChatRole, SeekCodeError, SeekCodeResult, TaskId, TokenUsage, ToolCallId,
    WorkspaceId,
};
use seekcode_model_provider::{ChatChunk, ChatRequest, ModelProvider, ToolCall};
use seekcode_tool_system::{ToolContext, ToolOutput, ToolRegistry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Agent runtime configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Default model used for coding tasks.
    pub default_model: String,
    /// Whether thinking mode is enabled by default.
    pub thinking: bool,
    /// Whether tool schemas should be strict.
    pub strict_tools: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            default_model: "deepseek-v4-pro".to_string(),
            thinking: true,
            strict_tools: true,
        }
    }
}

/// Request to start an agent task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StartTaskRequest {
    /// User prompt.
    pub prompt: String,
    /// Optional workspace bound to the task.
    pub workspace_id: Option<WorkspaceId>,
    /// Optional model override.
    pub model: Option<String>,
}

/// Agent task snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTask {
    /// Task identifier.
    pub id: TaskId,
    /// Current task state.
    pub state: AgentState,
}

/// Agent task lifecycle state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    /// Task is queued.
    Queued,
    /// Task is calling a model provider.
    Thinking,
    /// Task is executing a tool.
    RunningTool,
    /// Task has completed.
    Completed,
    /// Task was canceled.
    Canceled,
    /// Task failed.
    Failed,
}

/// Event produced by the agent loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Task was accepted and scheduled.
    TaskStarted {
        /// Task identifier.
        task_id: TaskId,
        /// Model selected for the task.
        model: String,
    },
    /// Task state changed.
    StateChanged { task_id: TaskId, state: AgentState },
    /// One model request round has started.
    ModelRequestStarted {
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Model selected for this request.
        model: String,
        /// Number of messages sent to the provider.
        message_count: usize,
        /// Number of tools exposed to the provider.
        tool_count: usize,
    },
    /// Assistant emitted text.
    AssistantToken {
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant content delta.
        text: String,
    },
    /// Assistant emitted reasoning text.
    AssistantReasoning {
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant reasoning delta.
        text: String,
    },
    /// Tool call execution is about to start.
    ToolCallStarted {
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Tool call identifier.
        tool_call_id: ToolCallId,
        /// Tool name.
        name: String,
        /// Raw JSON arguments.
        arguments: serde_json::Value,
    },
    /// Tool call execution has completed.
    ToolCallFinished {
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Tool call identifier.
        tool_call_id: ToolCallId,
        /// Tool name.
        name: String,
        /// Whether execution succeeded.
        ok: bool,
        /// Short result summary.
        summary: Option<String>,
        /// Machine-readable output for detail panels.
        output: Option<serde_json::Value>,
        /// Error text if execution failed.
        error: Option<String>,
    },
    /// One model request round has finished.
    ModelRoundFinished {
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Final usage accounting if returned by the provider.
        usage: Option<TokenUsage>,
    },
    /// Task finished.
    Finished { task_id: TaskId },
    /// Task failed.
    Failed {
        /// Task identifier.
        task_id: TaskId,
        /// Error text.
        error: String,
    },
    /// Task was canceled.
    Canceled {
        /// Task identifier.
        task_id: TaskId,
    },
}

/// Context assembled for an agent task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentContext {
    /// Task identifier.
    pub task_id: TaskId,
    /// Optional workspace identifier.
    pub workspace_id: Option<WorkspaceId>,
    /// Conversation messages used for the next provider request.
    pub messages: Vec<ChatMessage>,
}

/// Provider-backed agent runtime.
pub struct Agent {
    config: AgentConfig,
    provider: Arc<RwLock<Arc<dyn ModelProvider>>>,
    tools: Arc<ToolRegistry>,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    events: broadcast::Sender<AgentEvent>,
}

impl Agent {
    /// Creates a new agent runtime.
    pub fn new(
        config: AgentConfig,
        provider: Arc<dyn ModelProvider>,
        tools: Arc<ToolRegistry>,
    ) -> Self {
        let (events, _) = broadcast::channel(1_024);

        Self {
            config,
            provider: Arc::new(RwLock::new(provider)),
            tools,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            events,
        }
    }

    /// Returns the active agent configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Subscribes to the agent event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Replaces the model provider used by newly-started tasks.
    pub fn set_provider(&self, provider: Arc<dyn ModelProvider>) {
        *self.provider.write() = provider;
    }

    /// Starts a new task.
    pub async fn start_task(&self, request: StartTaskRequest) -> SeekCodeResult<AgentTask> {
        let prompt = request.prompt.trim();
        if prompt.is_empty() {
            return Err(SeekCodeError::Validation(
                "agent task prompt cannot be empty".to_string(),
            ));
        }

        let task_id = TaskId::new();
        self.insert_task(AgentTask {
            id: task_id,
            state: AgentState::Queued,
        })
        .await;
        self.set_task_state(task_id, AgentState::Thinking).await?;

        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone());
        let context = AgentContext::new(task_id, request.workspace_id, prompt.to_string());

        publish(
            &self.events,
            AgentEvent::TaskStarted {
                task_id,
                model: model.clone(),
            },
        );
        publish_state(&self.events, task_id, AgentState::Thinking);

        let runner = AgentTaskRunner {
            config: self.config.clone(),
            provider: self.provider.read().clone(),
            tools: self.tools.clone(),
            tasks: self.tasks.clone(),
            events: self.events.clone(),
        };
        tokio::spawn(async move {
            runner.run(task_id, model, context).await;
        });

        let task = AgentTask {
            id: task_id,
            state: AgentState::Thinking,
        };
        Ok(task)
    }

    /// Cancels a running task.
    pub async fn cancel_task(&self, task_id: TaskId) -> SeekCodeResult<()> {
        let mut tasks = self.tasks.write();
        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;

        task.state = AgentState::Canceled;
        Ok(())
    }

    /// Resumes a paused or persisted task.
    pub async fn resume_task(&self, task_id: TaskId) -> SeekCodeResult<AgentTask> {
        let mut tasks = self.tasks.write();
        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;

        if matches!(task.state, AgentState::Canceled | AgentState::Failed) {
            task.state = AgentState::Queued;
        }

        Ok(task.clone())
    }

    /// Handles one provider streaming chunk.
    pub async fn handle_model_chunk(&self, chunk: ChatChunk) -> SeekCodeResult<Vec<AgentEvent>> {
        self.handle_model_chunk_for_task(TaskId::new(), None, 1, chunk)
            .await
    }

    /// Dispatches a provider-requested tool call.
    pub async fn dispatch_tool_call(&self, tool_call: ToolCall) -> SeekCodeResult<AgentEvent> {
        self.dispatch_tool_call_for_task(TaskId::new(), None, 1, tool_call)
            .await
    }

    async fn dispatch_tool_call_for_task(
        &self,
        task_id: TaskId,
        workspace_id: Option<WorkspaceId>,
        round_id: u32,
        tool_call: ToolCall,
    ) -> SeekCodeResult<AgentEvent> {
        let name = tool_call.name.clone();
        let tool_call_id = tool_call.id;
        let output = self
            .tools
            .execute(
                &tool_call.name,
                ToolContext {
                    task_id,
                    workspace_id,
                    workspace_root: None,
                },
                tool_call.arguments,
            )
            .await?;

        Ok(tool_finished_event(
            task_id,
            round_id,
            tool_call_id,
            name,
            Ok(output),
        ))
    }

    async fn handle_model_chunk_for_task(
        &self,
        task_id: TaskId,
        workspace_id: Option<WorkspaceId>,
        round_id: u32,
        chunk: ChatChunk,
    ) -> SeekCodeResult<Vec<AgentEvent>> {
        match chunk {
            ChatChunk::Content(text) => Ok(vec![AgentEvent::AssistantToken {
                task_id,
                round_id,
                text,
            }]),
            ChatChunk::Reasoning(text) => Ok(vec![AgentEvent::AssistantReasoning {
                task_id,
                round_id,
                text,
            }]),
            ChatChunk::ToolCall(tool_call) => {
                let tracked = self.task_state(task_id).await.is_ok();
                if tracked {
                    self.set_task_state(task_id, AgentState::RunningTool)
                        .await?;
                }
                let event = self
                    .dispatch_tool_call_for_task(task_id, workspace_id, round_id, tool_call)
                    .await?;
                if tracked {
                    self.set_task_state(task_id, AgentState::Thinking).await?;
                }
                Ok(vec![event])
            }
            ChatChunk::Usage(usage) => Ok(vec![AgentEvent::ModelRoundFinished {
                task_id,
                round_id,
                usage: Some(usage),
            }]),
            ChatChunk::Finished => Ok(vec![AgentEvent::ModelRoundFinished {
                task_id,
                round_id,
                usage: None,
            }]),
        }
    }

    async fn insert_task(&self, task: AgentTask) {
        self.tasks.write().insert(task.id, task);
    }

    async fn set_task_state(&self, task_id: TaskId, state: AgentState) -> SeekCodeResult<()> {
        let mut tasks = self.tasks.write();
        let task = tasks
            .get_mut(&task_id)
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;
        task.state = state;
        Ok(())
    }

    async fn task_state(&self, task_id: TaskId) -> SeekCodeResult<AgentState> {
        self.tasks
            .read()
            .get(&task_id)
            .map(|task| task.state.clone())
            .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))
    }
}

impl AgentContext {
    fn new(task_id: TaskId, workspace_id: Option<WorkspaceId>, prompt: String) -> Self {
        Self {
            task_id,
            workspace_id,
            messages: vec![
                ChatMessage::new(
                    ChatRole::System,
                    "You are SeekCode, an autonomous coding agent powered by DeepSeek.",
                ),
                ChatMessage::new(ChatRole::User, prompt),
            ],
        }
    }
}

struct AgentTaskRunner {
    config: AgentConfig,
    provider: Arc<dyn ModelProvider>,
    tools: Arc<ToolRegistry>,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    events: broadcast::Sender<AgentEvent>,
}

impl AgentTaskRunner {
    async fn run(self, task_id: TaskId, model: String, context: AgentContext) {
        let result = self.run_inner(task_id, model, context).await;
        if let Err(error) = result {
            let _ = set_task_state(&self.tasks, task_id, AgentState::Failed).await;
            publish_state(&self.events, task_id, AgentState::Failed);
            publish(
                &self.events,
                AgentEvent::Failed {
                    task_id,
                    error: error.to_string(),
                },
            );
        }
    }

    async fn run_inner(
        &self,
        task_id: TaskId,
        model: String,
        context: AgentContext,
    ) -> SeekCodeResult<()> {
        let round_id = 1;
        let tool_specs = self.tools.tool_specs(self.config.strict_tools);
        let chat_request = ChatRequest {
            model: model.clone(),
            messages: context.messages.clone(),
            tools: tool_specs.clone(),
            thinking: self.config.thinking,
            strict_tools: self.config.strict_tools,
        };

        publish(
            &self.events,
            AgentEvent::ModelRequestStarted {
                task_id,
                round_id,
                model,
                message_count: chat_request.messages.len(),
                tool_count: tool_specs.len(),
            },
        );

        let mut stream = self.provider.stream_chat(chat_request)?;
        let mut final_usage = None;
        let mut sent_round_finished = false;

        while let Some(chunk) = stream.next().await {
            if task_state(&self.tasks, task_id).await? == AgentState::Canceled {
                publish_state(&self.events, task_id, AgentState::Canceled);
                publish(&self.events, AgentEvent::Canceled { task_id });
                return Ok(());
            }

            match chunk? {
                ChatChunk::Content(text) => {
                    publish(
                        &self.events,
                        AgentEvent::AssistantToken {
                            task_id,
                            round_id,
                            text,
                        },
                    );
                }
                ChatChunk::Reasoning(text) => {
                    publish(
                        &self.events,
                        AgentEvent::AssistantReasoning {
                            task_id,
                            round_id,
                            text,
                        },
                    );
                }
                ChatChunk::ToolCall(tool_call) => {
                    self.run_tool_call(task_id, context.workspace_id, round_id, tool_call)
                        .await?;
                }
                ChatChunk::Usage(usage) => {
                    final_usage = Some(usage.clone());
                    publish(
                        &self.events,
                        AgentEvent::ModelRoundFinished {
                            task_id,
                            round_id,
                            usage: Some(usage),
                        },
                    );
                    sent_round_finished = true;
                }
                ChatChunk::Finished => {
                    if !sent_round_finished {
                        publish(
                            &self.events,
                            AgentEvent::ModelRoundFinished {
                                task_id,
                                round_id,
                                usage: final_usage,
                            },
                        );
                    }
                    break;
                }
            }
        }

        if task_state(&self.tasks, task_id).await? == AgentState::Canceled {
            publish_state(&self.events, task_id, AgentState::Canceled);
            publish(&self.events, AgentEvent::Canceled { task_id });
            return Ok(());
        }

        set_task_state(&self.tasks, task_id, AgentState::Completed).await?;
        publish_state(&self.events, task_id, AgentState::Completed);
        publish(&self.events, AgentEvent::Finished { task_id });
        Ok(())
    }

    async fn run_tool_call(
        &self,
        task_id: TaskId,
        workspace_id: Option<WorkspaceId>,
        round_id: u32,
        tool_call: ToolCall,
    ) -> SeekCodeResult<()> {
        set_task_state(&self.tasks, task_id, AgentState::RunningTool).await?;
        publish_state(&self.events, task_id, AgentState::RunningTool);

        let tool_call_id = tool_call.id;
        let name = tool_call.name.clone();
        let arguments = tool_call.arguments.clone();
        publish(
            &self.events,
            AgentEvent::ToolCallStarted {
                task_id,
                round_id,
                tool_call_id,
                name: name.clone(),
                arguments,
            },
        );

        let result = self
            .tools
            .execute(
                &tool_call.name,
                ToolContext {
                    task_id,
                    workspace_id,
                    workspace_root: None,
                },
                tool_call.arguments,
            )
            .await;
        let event = tool_finished_event(task_id, round_id, tool_call_id, name, result);
        let failed = matches!(event, AgentEvent::ToolCallFinished { ok: false, .. });
        publish(&self.events, event);

        set_task_state(&self.tasks, task_id, AgentState::Thinking).await?;
        publish_state(&self.events, task_id, AgentState::Thinking);

        if failed {
            return Err(SeekCodeError::ToolExecution(
                "tool execution failed".to_string(),
            ));
        }

        Ok(())
    }
}

fn tool_finished_event(
    task_id: TaskId,
    round_id: u32,
    tool_call_id: ToolCallId,
    name: String,
    result: SeekCodeResult<ToolOutput>,
) -> AgentEvent {
    match result {
        Ok(output) => AgentEvent::ToolCallFinished {
            task_id,
            round_id,
            tool_call_id,
            name,
            ok: true,
            summary: Some(output.summary),
            output: Some(output.content),
            error: None,
        },
        Err(error) => AgentEvent::ToolCallFinished {
            task_id,
            round_id,
            tool_call_id,
            name,
            ok: false,
            summary: None,
            output: None,
            error: Some(error.to_string()),
        },
    }
}

fn publish(events: &broadcast::Sender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
}

fn publish_state(events: &broadcast::Sender<AgentEvent>, task_id: TaskId, state: AgentState) {
    publish(events, AgentEvent::StateChanged { task_id, state });
}

async fn set_task_state(
    tasks: &RwLock<HashMap<TaskId, AgentTask>>,
    task_id: TaskId,
    state: AgentState,
) -> SeekCodeResult<()> {
    let mut tasks = tasks.write();
    let task = tasks
        .get_mut(&task_id)
        .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;
    task.state = state;
    Ok(())
}

async fn task_state(
    tasks: &RwLock<HashMap<TaskId, AgentTask>>,
    task_id: TaskId,
) -> SeekCodeResult<AgentState> {
    tasks
        .read()
        .get(&task_id)
        .map(|task| task.state.clone())
        .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::stream;
    use seekcode_model_provider::{ChatResponse, ChatStream, ModelProfile};

    #[derive(Default)]
    struct MockProvider {
        chunks: Vec<ChatChunk>,
    }

    #[async_trait]
    impl ModelProvider for MockProvider {
        fn stream_chat(&self, _request: ChatRequest) -> SeekCodeResult<ChatStream> {
            let chunks = self.chunks.clone().into_iter().map(Ok);
            Ok(Box::pin(stream::iter(chunks)))
        }

        async fn complete_chat(&self, _request: ChatRequest) -> SeekCodeResult<ChatResponse> {
            todo!("mock complete_chat is not used by agent start_task tests")
        }

        async fn model_profile(&self, model: &str) -> SeekCodeResult<ModelProfile> {
            Ok(ModelProfile {
                id: model.to_string(),
                context_window: 1_000,
                supports_tools: true,
                supports_thinking: true,
            })
        }
    }

    #[tokio::test]
    async fn start_task_completes_when_provider_stream_finishes() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider {
                chunks: vec![ChatChunk::Content("hello".to_string()), ChatChunk::Finished],
            }),
            Arc::new(ToolRegistry::new()),
        );

        let task = agent
            .start_task(StartTaskRequest {
                prompt: "Explain this project".to_string(),
                workspace_id: None,
                model: None,
            })
            .await
            .expect("task starts and completes");

        assert_eq!(task.state, AgentState::Thinking);
        wait_for_state(&agent, task.id, AgentState::Completed).await;
    }

    #[tokio::test]
    async fn start_task_rejects_empty_prompt() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider::default()),
            Arc::new(ToolRegistry::new()),
        );

        let error = agent
            .start_task(StartTaskRequest {
                prompt: "   ".to_string(),
                workspace_id: None,
                model: None,
            })
            .await
            .expect_err("empty prompt is rejected");

        assert!(matches!(error, SeekCodeError::Validation(_)));
    }

    #[tokio::test]
    async fn start_task_publishes_stream_events() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider {
                chunks: vec![ChatChunk::Content("hello".to_string()), ChatChunk::Finished],
            }),
            Arc::new(ToolRegistry::new()),
        );
        let mut events = agent.subscribe();
        let task = agent
            .start_task(StartTaskRequest {
                prompt: "Say hello".to_string(),
                workspace_id: None,
                model: None,
            })
            .await
            .expect("task starts");

        let mut saw_started = false;
        let mut saw_token = false;
        let mut saw_finished = false;
        for _ in 0..20 {
            let event = events.recv().await.expect("event is published");
            match event {
                AgentEvent::TaskStarted { task_id, .. } if task_id == task.id => {
                    saw_started = true;
                }
                AgentEvent::AssistantToken { task_id, text, .. }
                    if task_id == task.id && text == "hello" =>
                {
                    saw_token = true;
                }
                AgentEvent::Finished { task_id } if task_id == task.id => {
                    saw_finished = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_started);
        assert!(saw_token);
        assert!(saw_finished);
    }

    #[tokio::test]
    async fn cancel_task_marks_known_task_canceled() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider::default()),
            Arc::new(ToolRegistry::new()),
        );
        let task_id = TaskId::new();
        agent
            .insert_task(AgentTask {
                id: task_id,
                state: AgentState::Thinking,
            })
            .await;

        agent.cancel_task(task_id).await.expect("task cancels");

        assert_eq!(
            agent.task_state(task_id).await.unwrap(),
            AgentState::Canceled
        );
    }

    #[tokio::test]
    async fn resume_task_requeues_canceled_task() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider::default()),
            Arc::new(ToolRegistry::new()),
        );
        let task_id = TaskId::new();
        agent
            .insert_task(AgentTask {
                id: task_id,
                state: AgentState::Canceled,
            })
            .await;

        let task = agent.resume_task(task_id).await.expect("task resumes");

        assert_eq!(task.state, AgentState::Queued);
    }

    #[tokio::test]
    async fn handle_model_chunk_maps_content_to_token_event() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider::default()),
            Arc::new(ToolRegistry::new()),
        );

        let events = agent
            .handle_model_chunk(ChatChunk::Content("hello".to_string()))
            .await
            .expect("chunk maps");

        assert!(matches!(
            &events[0],
            AgentEvent::AssistantToken { text, .. } if text == "hello"
        ));
    }

    #[tokio::test]
    async fn cancel_unknown_task_returns_not_found() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider::default()),
            Arc::new(ToolRegistry::new()),
        );

        let error = agent
            .cancel_task(TaskId::new())
            .await
            .expect_err("unknown task cannot be canceled");

        assert!(matches!(error, SeekCodeError::NotFound(_)));
    }

    async fn wait_for_state(agent: &Agent, task_id: TaskId, expected: AgentState) {
        for _ in 0..50 {
            if agent.task_state(task_id).await.expect("task exists") == expected {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        panic!("task did not reach expected state");
    }
}
