//! Agent state machine, task context, and execution loop scaffolding.

use futures_util::StreamExt;
use parking_lot::RwLock;
use seekcode_common::{
    ChatMessage, ChatRole, SeekCodeError, SeekCodeResult, SessionId, TaskId, TokenUsage, ToolCallId,
};
use seekcode_model_provider::{ChatChunk, ChatRequest, ModelProvider, ToolCall};
use seekcode_tool_system::{ToolContext, ToolOutput, ToolRegistry};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

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
    /// Persisted session identifier bound to the task.
    pub session_id: SessionId,
    /// User prompt.
    pub prompt: String,
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
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// Model selected for the task.
        model: String,
    },
    /// Task state changed.
    StateChanged {
        session_id: SessionId,
        task_id: TaskId,
        state: AgentState,
    },
    /// One model request round has started.
    ModelRequestStarted {
        /// Persisted session identifier.
        session_id: SessionId,
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
    /// One provider choice chunk was emitted.
    ModelChoice {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Complete provider choice chunk.
        choice: seekcode_model_provider::ChatChoiceChunk,
    },
    /// Assistant emitted text.
    AssistantToken {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant content delta.
        text: String,
    },
    /// Assistant emitted reasoning text.
    AssistantReasoning {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant reasoning delta.
        text: String,
    },
    /// Tool call execution is about to start.
    ToolCallStarted {
        /// Persisted session identifier.
        session_id: SessionId,
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
        /// Persisted session identifier.
        session_id: SessionId,
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
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Final usage accounting if returned by the provider.
        usage: Option<TokenUsage>,
    },
    /// Task finished.
    Finished {
        session_id: SessionId,
        task_id: TaskId,
    },
    /// Task failed.
    Failed {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// Error text.
        error: String,
    },
    /// Task was canceled.
    Canceled {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
    },
}

/// Context assembled for an agent task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentContext {
    /// Persisted session identifier bound to the task.
    pub session_id: SessionId,
    /// Task identifier.
    pub task_id: TaskId,
    /// Conversation messages used for the next provider request.
    pub messages: Vec<ChatMessage>,
}

/// Provider-backed agent runtime.
pub struct Agent {
    config: AgentConfig,
    provider: Arc<RwLock<Arc<dyn ModelProvider>>>,
    tools: Arc<ToolRegistry>,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    task_controls: Arc<RwLock<HashMap<TaskId, TaskControl>>>,
}

struct TaskControl {
    abort_handle: AbortHandle,
}

struct RunCompletionGuard {
    task_id: TaskId,
    session_id: SessionId,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    task_controls: Arc<RwLock<HashMap<TaskId, TaskControl>>>,
    events: mpsc::UnboundedSender<AgentEvent>,
}

impl Drop for RunCompletionGuard {
    fn drop(&mut self) {
        self.task_controls.write().remove(&self.task_id);

        let is_canceled = self
            .tasks
            .read()
            .get(&self.task_id)
            .map(|task| task.state == AgentState::Canceled)
            .unwrap_or(false);
        if !is_canceled {
            return;
        }

        publish_state(
            &self.events,
            self.session_id,
            self.task_id,
            AgentState::Canceled,
        );
        publish(
            &self.events,
            AgentEvent::Canceled {
                session_id: self.session_id,
                task_id: self.task_id,
            },
        );
    }
}

impl Agent {
    /// Creates a new agent runtime.
    pub fn new(
        config: AgentConfig,
        provider: Arc<dyn ModelProvider>,
        tools: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            config,
            provider: Arc::new(RwLock::new(provider)),
            tools,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            task_controls: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the active agent configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Replaces the model provider used by newly-started tasks.
    pub fn set_provider(&self, provider: Arc<dyn ModelProvider>) {
        *self.provider.write() = provider;
    }

    /// Starts a new task.
    pub async fn start_task(
        &self,
        request: StartTaskRequest,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> SeekCodeResult<AgentTask> {
        let prompt = request.prompt.trim();
        let messages = AgentContext::default_messages(prompt)?;
        self.start_task_with_messages(request, messages, events)
            .await
    }

    /// Starts a new task with an application-assembled conversation context.
    pub async fn start_task_with_messages(
        &self,
        request: StartTaskRequest,
        messages: Vec<ChatMessage>,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> SeekCodeResult<AgentTask> {
        self.start_task_with_messages_and_tools(request, messages, self.tools.clone(), events)
            .await
    }

    /// Starts a new task with an application-assembled context and task-scoped tools.
    pub async fn start_task_with_messages_and_tools(
        &self,
        request: StartTaskRequest,
        messages: Vec<ChatMessage>,
        tools: Arc<ToolRegistry>,
        events: mpsc::UnboundedSender<AgentEvent>,
    ) -> SeekCodeResult<AgentTask> {
        let prompt = request.prompt.trim();
        if prompt.is_empty() {
            return Err(SeekCodeError::Validation(
                "agent task prompt cannot be empty".to_string(),
            ));
        }
        if messages.is_empty() {
            return Err(SeekCodeError::Validation(
                "agent task context cannot be empty".to_string(),
            ));
        }

        let task_id = TaskId::new();
        let session_id = request.session_id;
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
        let context = AgentContext::new(task_id, session_id, messages);

        publish(
            &events,
            AgentEvent::TaskStarted {
                session_id,
                task_id,
                model: model.clone(),
            },
        );
        publish_state(&events, session_id, task_id, AgentState::Thinking);

        let runner = AgentTaskRunner {
            config: self.config.clone(),
            provider: self.provider.read().clone(),
            tools,
            tasks: self.tasks.clone(),
            events: events.clone(),
        };
        let task_controls = self.task_controls.clone();
        let completion_guard = RunCompletionGuard {
            task_id,
            session_id,
            tasks: self.tasks.clone(),
            task_controls: task_controls.clone(),
            events: events.clone(),
        };
        let handle = tokio::spawn(async move {
            let _completion_guard = completion_guard;
            runner.run(task_id, session_id, model, context).await;
        });
        self.task_controls.write().insert(
            task_id,
            TaskControl {
                abort_handle: handle.abort_handle(),
            },
        );
        if matches!(
            self.task_state(task_id).await?,
            AgentState::Completed | AgentState::Canceled | AgentState::Failed
        ) {
            self.task_controls.write().remove(&task_id);
        }

        let task = AgentTask {
            id: task_id,
            state: AgentState::Thinking,
        };
        Ok(task)
    }

    /// Cancels a running task.
    pub async fn cancel_task(&self, task_id: TaskId) -> SeekCodeResult<()> {
        {
            let mut tasks = self.tasks.write();
            let task = tasks
                .get_mut(&task_id)
                .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;

            if matches!(
                task.state,
                AgentState::Completed | AgentState::Canceled | AgentState::Failed
            ) {
                return Ok(());
            }

            task.state = AgentState::Canceled;
        }

        if let Some(control) = self.task_controls.write().remove(&task_id) {
            control.abort_handle.abort();
        }

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
        self.handle_model_chunk_for_task(TaskId::new(), SessionId::new(), 1, chunk)
            .await
    }

    /// Dispatches a provider-requested tool call.
    pub async fn dispatch_tool_call(&self, tool_call: ToolCall) -> SeekCodeResult<AgentEvent> {
        self.dispatch_tool_call_for_task(TaskId::new(), SessionId::new(), 1, tool_call)
            .await
    }

    async fn dispatch_tool_call_for_task(
        &self,
        task_id: TaskId,
        session_id: SessionId,
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
                    workspace_id: None,
                    workspace_root: None,
                },
                tool_call.arguments,
            )
            .await?;

        Ok(tool_finished_event(
            task_id,
            session_id,
            round_id,
            tool_call_id,
            name,
            Ok(output),
        ))
    }

    async fn handle_model_chunk_for_task(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        round_id: u32,
        chunk: ChatChunk,
    ) -> SeekCodeResult<Vec<AgentEvent>> {
        match chunk {
            ChatChunk::Content(text) => Ok(vec![AgentEvent::AssistantToken {
                session_id,
                task_id,
                round_id,
                text,
            }]),
            ChatChunk::Choice(choice) => Ok(vec![AgentEvent::ModelChoice {
                session_id,
                task_id,
                round_id,
                choice,
            }]),
            ChatChunk::Reasoning(text) => Ok(vec![AgentEvent::AssistantReasoning {
                session_id,
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
                    .dispatch_tool_call_for_task(task_id, session_id, round_id, tool_call)
                    .await?;
                if tracked {
                    self.set_task_state(task_id, AgentState::Thinking).await?;
                }
                Ok(vec![event])
            }
            ChatChunk::Usage(usage) => Ok(vec![AgentEvent::ModelRoundFinished {
                session_id,
                task_id,
                round_id,
                usage: Some(usage),
            }]),
            ChatChunk::Finished => Ok(vec![AgentEvent::ModelRoundFinished {
                session_id,
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
    fn new(task_id: TaskId, session_id: SessionId, messages: Vec<ChatMessage>) -> Self {
        Self {
            session_id,
            task_id,
            messages,
        }
    }

    fn default_messages(prompt: &str) -> SeekCodeResult<Vec<ChatMessage>> {
        if prompt.trim().is_empty() {
            return Err(SeekCodeError::Validation(
                "agent task prompt cannot be empty".to_string(),
            ));
        }

        Ok(vec![
            ChatMessage::new(
                ChatRole::System,
                "You are SeekCode, an autonomous coding agent powered by DeepSeek.",
            ),
            ChatMessage::new(ChatRole::User, prompt),
        ])
    }
}

struct AgentTaskRunner {
    config: AgentConfig,
    provider: Arc<dyn ModelProvider>,
    tools: Arc<ToolRegistry>,
    tasks: Arc<RwLock<HashMap<TaskId, AgentTask>>>,
    events: mpsc::UnboundedSender<AgentEvent>,
}

struct ToolCallRunResult {
    tool_call: ToolCall,
    result_content: String,
}

impl AgentTaskRunner {
    async fn run(
        self,
        task_id: TaskId,
        session_id: SessionId,
        model: String,
        context: AgentContext,
    ) {
        let result = self.run_inner(task_id, session_id, model, context).await;
        if let Err(error) = result {
            let _ = set_task_state(&self.tasks, task_id, AgentState::Failed).await;
            publish_state(&self.events, session_id, task_id, AgentState::Failed);
            publish(
                &self.events,
                AgentEvent::Failed {
                    session_id,
                    task_id,
                    error: error.to_string(),
                },
            );
        }
    }

    async fn run_inner(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        model: String,
        context: AgentContext,
    ) -> SeekCodeResult<()> {
        let tool_specs = self.tools.tool_specs(self.config.strict_tools);
        let mut messages = context.messages.clone();

        for round_id in 1..=8 {
            let chat_request = ChatRequest {
                model: model.clone(),
                messages: messages.clone(),
                tools: tool_specs.clone(),
                thinking: self.config.thinking,
                strict_tools: self.config.strict_tools,
            };

            publish(
                &self.events,
                AgentEvent::ModelRequestStarted {
                    session_id,
                    task_id,
                    round_id,
                    model: model.clone(),
                    message_count: chat_request.messages.len(),
                    tool_count: tool_specs.len(),
                },
            );

            let mut stream = self.provider.stream_chat(chat_request)?;
            let mut final_usage = None;
            let mut sent_round_finished = false;
            let mut tool_calls = ToolCallAccumulator::default();
            let mut tool_results = Vec::new();
            let mut round_content = String::new();
            let mut round_reasoning = String::new();

            while let Some(chunk) = stream.next().await {
                if task_state(&self.tasks, task_id).await? == AgentState::Canceled {
                    return Ok(());
                }

                match chunk? {
                    ChatChunk::Choice(mut choice) => {
                        if let Some(text) = &choice.delta.content {
                            round_content.push_str(text);
                        }
                        if let Some(text) = &choice.delta.reasoning_content {
                            round_reasoning.push_str(text);
                        }
                        tool_calls.apply_choice_delta(&mut choice);
                        let should_run_tools =
                            choice.finish_reason.as_deref() == Some("tool_calls");
                        publish(
                            &self.events,
                            AgentEvent::ModelChoice {
                                session_id,
                                task_id,
                                round_id,
                                choice,
                            },
                        );

                        if should_run_tools {
                            for tool_call in tool_calls.take_completed()? {
                                tool_results.push(
                                    self.run_tool_call(task_id, session_id, round_id, tool_call)
                                        .await?,
                                );
                            }
                        }
                    }
                    ChatChunk::Content(text) => {
                        round_content.push_str(&text);
                        publish(
                            &self.events,
                            AgentEvent::AssistantToken {
                                session_id,
                                task_id,
                                round_id,
                                text,
                            },
                        );
                    }
                    ChatChunk::Reasoning(text) => {
                        round_reasoning.push_str(&text);
                        publish(
                            &self.events,
                            AgentEvent::AssistantReasoning {
                                session_id,
                                task_id,
                                round_id,
                                text,
                            },
                        );
                    }
                    ChatChunk::ToolCall(tool_call) => {
                        tool_results.push(
                            self.run_tool_call(task_id, session_id, round_id, tool_call)
                                .await?,
                        );
                    }
                    ChatChunk::Usage(usage) => {
                        final_usage = Some(usage.clone());
                        publish(
                            &self.events,
                            AgentEvent::ModelRoundFinished {
                                session_id,
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
                                    session_id,
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
                return Ok(());
            }

            if tool_results.is_empty() {
                set_task_state(&self.tasks, task_id, AgentState::Completed).await?;
                publish_state(&self.events, session_id, task_id, AgentState::Completed);
                publish(
                    &self.events,
                    AgentEvent::Finished {
                        session_id,
                        task_id,
                    },
                );
                return Ok(());
            }

            append_tool_results_to_context(
                &mut messages,
                round_content,
                round_reasoning,
                tool_results,
            );
        }

        Err(SeekCodeError::ModelProvider(
            "model requested too many tool rounds".to_string(),
        ))
    }

    async fn run_tool_call(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        round_id: u32,
        tool_call: ToolCall,
    ) -> SeekCodeResult<ToolCallRunResult> {
        set_task_state(&self.tasks, task_id, AgentState::RunningTool).await?;
        publish_state(&self.events, session_id, task_id, AgentState::RunningTool);

        let tool_call_id = tool_call.id;
        let name = tool_call.name.clone();
        let arguments = tool_call.arguments.clone();
        let started_at = Instant::now();
        tracing::debug!(
            target: "seekcode_agent_core::tools",
            %session_id,
            %task_id,
            round_id,
            %tool_call_id,
            tool_name = %name,
            arguments = %arguments,
            "tool call started"
        );
        publish(
            &self.events,
            AgentEvent::ToolCallStarted {
                session_id,
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
                    workspace_id: None,
                    workspace_root: None,
                },
                tool_call.arguments.clone(),
            )
            .await;
        match &result {
            Ok(output) => tracing::debug!(
                target: "seekcode_agent_core::tools",
                %session_id,
                %task_id,
                round_id,
                %tool_call_id,
                tool_name = %name,
                elapsed_ms = started_at.elapsed().as_millis(),
                output_summary = %output.summary,
                output_content = %output.content,
                "tool call finished successfully"
            ),
            Err(error) => tracing::warn!(
                target: "seekcode_agent_core::tools",
                %session_id,
                %task_id,
                round_id,
                %tool_call_id,
                tool_name = %name,
                elapsed_ms = started_at.elapsed().as_millis(),
                %error,
                "tool call failed"
            ),
        }
        let result_content = match result {
            Ok(output) => {
                let content = serde_json::to_string(&output.content).unwrap_or_default();
                publish(
                    &self.events,
                    tool_finished_event(
                        task_id,
                        session_id,
                        round_id,
                        tool_call_id,
                        name,
                        Ok(output),
                    ),
                );
                content
            }
            Err(error) => {
                publish(
                    &self.events,
                    tool_finished_event(
                        task_id,
                        session_id,
                        round_id,
                        tool_call_id,
                        name,
                        Err(error),
                    ),
                );
                set_task_state(&self.tasks, task_id, AgentState::Thinking).await?;
                publish_state(&self.events, session_id, task_id, AgentState::Thinking);
                return Err(SeekCodeError::ToolExecution(
                    "tool execution failed".to_string(),
                ));
            }
        };

        set_task_state(&self.tasks, task_id, AgentState::Thinking).await?;
        publish_state(&self.events, session_id, task_id, AgentState::Thinking);

        Ok(ToolCallRunResult {
            tool_call,
            result_content,
        })
    }
}

fn tool_finished_event(
    task_id: TaskId,
    session_id: SessionId,
    round_id: u32,
    tool_call_id: ToolCallId,
    name: String,
    result: SeekCodeResult<ToolOutput>,
) -> AgentEvent {
    match result {
        Ok(output) => AgentEvent::ToolCallFinished {
            session_id,
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
            session_id,
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

fn append_tool_results_to_context(
    messages: &mut Vec<ChatMessage>,
    content: String,
    reasoning_content: String,
    tool_results: Vec<ToolCallRunResult>,
) {
    let mut assistant = ChatMessage::new(ChatRole::Assistant, content);
    if !reasoning_content.is_empty() {
        assistant.reasoning_content = Some(reasoning_content);
    }
    assistant.tool_calls = tool_results
        .iter()
        .map(|result| {
            serde_json::json!({
                "id": result.tool_call.id.to_string(),
                "type": "function",
                "function": {
                    "name": result.tool_call.name,
                    "arguments": serde_json::to_string(&result.tool_call.arguments).unwrap_or_default()
                }
            })
        })
        .collect();
    messages.push(assistant);

    for result in tool_results {
        let mut message = ChatMessage::new(ChatRole::Tool, result.result_content);
        message.tool_call_id = Some(result.tool_call.id);
        messages.push(message);
    }
}

fn publish(events: &mpsc::UnboundedSender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
}

fn publish_state(
    events: &mpsc::UnboundedSender<AgentEvent>,
    session_id: SessionId,
    task_id: TaskId,
    state: AgentState,
) {
    publish(
        events,
        AgentEvent::StateChanged {
            session_id,
            task_id,
            state,
        },
    );
}

#[derive(Default)]
struct ToolCallAccumulator {
    partials: BTreeMap<u32, PartialToolCall>,
}

impl ToolCallAccumulator {
    fn apply_choice_delta(&mut self, choice: &mut seekcode_model_provider::ChatChoiceChunk) {
        for delta in &mut choice.delta.tool_calls {
            let partial = self
                .partials
                .entry(delta.index)
                .or_insert_with(|| PartialToolCall {
                    id: ToolCallId::new(),
                    name: None,
                    arguments: String::new(),
                });
            delta.id = Some(partial.id);

            if let Some(name) = &delta.name {
                partial.name = Some(name.clone());
            }
            if let Some(arguments) = &delta.arguments {
                partial.arguments.push_str(arguments);
            }
        }
    }

    fn take_completed(&mut self) -> SeekCodeResult<Vec<ToolCall>> {
        let partials = std::mem::take(&mut self.partials);
        partials
            .into_values()
            .map(|partial| {
                let name = partial.name.ok_or_else(|| {
                    SeekCodeError::ModelProvider("missing streamed tool call name".to_string())
                })?;
                let arguments = if partial.arguments.trim().is_empty() {
                    serde_json::Value::Object(Default::default())
                } else {
                    serde_json::from_str(&partial.arguments).map_err(|error| {
                        SeekCodeError::ModelProvider(format!(
                            "invalid streamed tool arguments: {error}"
                        ))
                    })?
                };

                Ok(ToolCall {
                    id: partial.id,
                    name,
                    arguments,
                })
            })
            .collect()
    }
}

struct PartialToolCall {
    id: ToolCallId,
    name: Option<String>,
    arguments: String,
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
        pending: bool,
    }

    struct RoundProvider {
        rounds: std::sync::Mutex<std::collections::VecDeque<Vec<ChatChunk>>>,
    }

    #[async_trait]
    impl ModelProvider for MockProvider {
        fn stream_chat(&self, _request: ChatRequest) -> SeekCodeResult<ChatStream> {
            if self.pending {
                return Ok(Box::pin(stream::pending()));
            }

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

    #[async_trait]
    impl ModelProvider for RoundProvider {
        fn stream_chat(&self, _request: ChatRequest) -> SeekCodeResult<ChatStream> {
            let chunks = self
                .rounds
                .lock()
                .expect("rounds lock")
                .pop_front()
                .unwrap_or_default()
                .into_iter()
                .map(Ok);
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
                pending: false,
            }),
            Arc::new(ToolRegistry::new()),
        );
        let (events, _) = mpsc::unbounded_channel();

        let task = agent
            .start_task(
                StartTaskRequest {
                    session_id: SessionId::new(),
                    prompt: "Explain this workspace".to_string(),
                    model: None,
                },
                events,
            )
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
        let (events, _) = mpsc::unbounded_channel();

        let error = agent
            .start_task(
                StartTaskRequest {
                    session_id: SessionId::new(),
                    prompt: "   ".to_string(),
                    model: None,
                },
                events,
            )
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
                pending: false,
            }),
            Arc::new(ToolRegistry::new()),
        );
        let (event_sender, mut events) = mpsc::unbounded_channel();
        let task = agent
            .start_task(
                StartTaskRequest {
                    session_id: SessionId::new(),
                    prompt: "Say hello".to_string(),
                    model: None,
                },
                event_sender,
            )
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
                AgentEvent::Finished { task_id, .. } if task_id == task.id => {
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
    async fn start_task_with_messages_and_tools_publishes_tool_count() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider {
                chunks: vec![ChatChunk::Finished],
                pending: false,
            }),
            Arc::new(ToolRegistry::new()),
        );
        let config = seekcode_tool_system::SystemToolConfig::new(std::env::temp_dir())
            .expect("system tool config builds");
        let tools = Arc::new(
            seekcode_tool_system::system_tool_registry(config).expect("system tools register"),
        );
        let (event_sender, mut events) = mpsc::unbounded_channel();

        let task = agent
            .start_task_with_messages_and_tools(
                StartTaskRequest {
                    session_id: SessionId::new(),
                    prompt: "Use tools".to_string(),
                    model: None,
                },
                AgentContext::default_messages("Use tools").expect("messages build"),
                tools,
                event_sender,
            )
            .await
            .expect("task starts");

        let mut saw_tool_count = false;
        for _ in 0..20 {
            let event = events.recv().await.expect("event is published");
            if matches!(
                event,
                AgentEvent::ModelRequestStarted {
                    task_id,
                    tool_count: 8,
                    ..
                } if task_id == task.id
            ) {
                saw_tool_count = true;
                break;
            }
        }

        assert!(saw_tool_count);
    }

    #[tokio::test]
    async fn start_task_continues_after_tool_result() {
        let mut rounds = std::collections::VecDeque::new();
        rounds.push_back(vec![
            ChatChunk::Choice(seekcode_model_provider::ChatChoiceChunk {
                delta: seekcode_model_provider::ChatDelta {
                    content: None,
                    reasoning_content: None,
                    tool_calls: vec![seekcode_model_provider::ToolCallDelta {
                        index: 0,
                        id: None,
                        kind: Some("function".to_string()),
                        name: Some(seekcode_tool_system::LIST_FILES_TOOL.to_string()),
                        arguments: Some("{}".to_string()),
                    }],
                },
                finish_reason: Some("tool_calls".to_string()),
            }),
            ChatChunk::Finished,
        ]);
        rounds.push_back(vec![
            ChatChunk::Content("done".to_string()),
            ChatChunk::Finished,
        ]);
        let config = seekcode_tool_system::SystemToolConfig::new(std::env::temp_dir())
            .expect("system tool config builds");
        let tools = Arc::new(
            seekcode_tool_system::system_tool_registry(config).expect("system tools register"),
        );
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(RoundProvider {
                rounds: std::sync::Mutex::new(rounds),
            }),
            Arc::new(ToolRegistry::new()),
        );
        let (event_sender, mut events) = mpsc::unbounded_channel();

        let task = agent
            .start_task_with_messages_and_tools(
                StartTaskRequest {
                    session_id: SessionId::new(),
                    prompt: "List files".to_string(),
                    model: None,
                },
                AgentContext::default_messages("List files").expect("messages build"),
                tools,
                event_sender,
            )
            .await
            .expect("task starts");

        let mut saw_tool_finished = false;
        let mut saw_second_round = false;
        let mut saw_final_text = false;
        for _ in 0..30 {
            let event = events.recv().await.expect("event is published");
            match event {
                AgentEvent::ToolCallFinished { task_id, ok, .. } if task_id == task.id && ok => {
                    saw_tool_finished = true;
                }
                AgentEvent::ModelRequestStarted {
                    task_id,
                    round_id: 2,
                    ..
                } if task_id == task.id => {
                    saw_second_round = true;
                }
                AgentEvent::AssistantToken { task_id, text, .. }
                    if task_id == task.id && text == "done" =>
                {
                    saw_final_text = true;
                }
                AgentEvent::Finished { task_id, .. } if task_id == task.id => break,
                _ => {}
            }
        }

        assert!(saw_tool_finished);
        assert!(saw_second_round);
        assert!(saw_final_text);
    }

    #[test]
    fn append_tool_results_preserves_assistant_reasoning_for_next_round() {
        let tool_call_id = ToolCallId::new();
        let mut messages = Vec::new();

        append_tool_results_to_context(
            &mut messages,
            String::new(),
            "I need to inspect the workspace.".to_string(),
            vec![ToolCallRunResult {
                tool_call: seekcode_model_provider::ToolCall {
                    id: tool_call_id,
                    name: seekcode_tool_system::LIST_FILES_TOOL.to_string(),
                    arguments: serde_json::json!({ "path": "." }),
                },
                result_content: "[]".to_string(),
            }],
        );

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, ChatRole::Assistant);
        assert_eq!(
            messages[0].reasoning_content.as_deref(),
            Some("I need to inspect the workspace.")
        );
        assert_eq!(messages[0].tool_calls.len(), 1);
        assert_eq!(messages[1].role, ChatRole::Tool);
        assert_eq!(messages[1].tool_call_id, Some(tool_call_id));
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
    async fn cancel_task_aborts_running_task_and_publishes_canceled() {
        let agent = Agent::new(
            AgentConfig::default(),
            Arc::new(MockProvider {
                chunks: Vec::new(),
                pending: true,
            }),
            Arc::new(ToolRegistry::new()),
        );
        let (event_sender, mut events) = mpsc::unbounded_channel();
        let session_id = SessionId::new();
        let task = agent
            .start_task(
                StartTaskRequest {
                    session_id,
                    prompt: "wait".to_string(),
                    model: None,
                },
                event_sender,
            )
            .await
            .expect("task starts");

        agent.cancel_task(task.id).await.expect("task cancels");

        let mut saw_canceled = false;
        for _ in 0..10 {
            let event = events.recv().await.expect("event is published");
            if matches!(
                event,
                AgentEvent::Canceled {
                    task_id,
                    session_id: event_session_id,
                } if task_id == task.id && event_session_id == session_id
            ) {
                saw_canceled = true;
                break;
            }
        }

        assert!(saw_canceled);
        assert_eq!(
            agent.task_state(task.id).await.unwrap(),
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
