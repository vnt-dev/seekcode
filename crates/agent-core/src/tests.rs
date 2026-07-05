//! Integration-style tests for the agent runtime and task loop.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream;
use seekcode_common::{
    ChatMessage, ChatRole, SeekCodeError, SeekCodeResult, SessionId, TaskId, TokenUsage,
    WorkspaceId,
};
use seekcode_deepseek_client::{
    ChatChunk, ChatRequest, ChatResponse, ChatStream, ModelProfile, ModelProvider,
};
use seekcode_tool_system::ToolRegistry;
use tokio::sync::mpsc;

use crate::runner::{append_tool_results_to_context, ToolCallRunResult};
use crate::{
    Agent, AgentConfig, AgentContextCompactionOutcome, AgentContextPrecheck, AgentContextPreparer,
    AgentEvent, AgentHistoryMessage, AgentState, AgentTask, AgentTaskContext, AgentToolContext,
    PreparedAgentContext, RunningContextCompaction, StartTaskRequest,
};

#[derive(Default)]
struct MockProvider {
    chunks: Vec<ChatChunk>,
    pending: bool,
}

struct RoundProvider {
    rounds: std::sync::Mutex<std::collections::VecDeque<Vec<ChatChunk>>>,
}

struct RecordingRoundProvider {
    rounds: std::sync::Mutex<std::collections::VecDeque<Vec<ChatChunk>>>,
    requests: Arc<std::sync::Mutex<Vec<ChatRequest>>>,
}

struct RetryProvider {
    attempts: std::sync::Mutex<std::collections::VecDeque<ProviderAttempt>>,
    requests: Arc<std::sync::Mutex<Vec<ChatRequest>>>,
}

enum ProviderAttempt {
    OpenError(String),
    Stream(Vec<StreamItem>),
}

enum StreamItem {
    Chunk(ChatChunk),
    Error(String),
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

#[async_trait]
impl ModelProvider for RecordingRoundProvider {
    fn stream_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatStream> {
        self.requests.lock().expect("requests lock").push(request);
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

#[async_trait]
impl ModelProvider for RetryProvider {
    fn stream_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatStream> {
        self.requests.lock().expect("requests lock").push(request);
        let attempt = self
            .attempts
            .lock()
            .expect("attempts lock")
            .pop_front()
            .unwrap_or_else(|| ProviderAttempt::Stream(Vec::new()));
        match attempt {
            ProviderAttempt::OpenError(error) => Err(SeekCodeError::ModelProvider(error)),
            ProviderAttempt::Stream(items) => {
                let chunks = items.into_iter().map(|item| match item {
                    StreamItem::Chunk(chunk) => Ok(chunk),
                    StreamItem::Error(error) => Err(SeekCodeError::ModelProvider(error)),
                });
                Ok(Box::pin(stream::iter(chunks)))
            }
        }
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

fn task_context(prompt: &str) -> AgentTaskContext {
    AgentTaskContext {
        last_input_tokens: 0,
        system_prompt: Vec::new(),
        general_prompt: Vec::new(),
        compacted_context: Vec::new(),
        history_messages: Vec::new(),
        latest_user_messages: vec![seekcode_common::ChatMessage::new(ChatRole::User, prompt)],
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
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Explain this workspace".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Explain this workspace"),
            Arc::new(ToolRegistry::new()),
            events,
            None,
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
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "   ".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("ignored"),
            Arc::new(ToolRegistry::new()),
            events,
            None,
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
            chunks: vec![
                ChatChunk::Content("hello".to_string()),
                ChatChunk::Usage(TokenUsage {
                    prompt_tokens: 11,
                    completion_tokens: 7,
                    total_tokens: 18,
                    cached_tokens: 3,
                }),
                ChatChunk::Finished,
            ],
            pending: false,
        }),
        Arc::new(ToolRegistry::new()),
    );
    let (event_sender, mut events) = mpsc::unbounded_channel();
    let task = agent
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Say hello".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Say hello"),
            Arc::new(ToolRegistry::new()),
            event_sender,
            None,
        )
        .await
        .expect("task starts");

    let mut saw_started = false;
    let mut saw_message_delta = false;
    let mut saw_round_finished = false;
    let mut saw_finished = false;
    for _ in 0..20 {
        let event = events.recv().await.expect("event is published");
        match event {
            AgentEvent::TaskStarted { task_id, .. } if task_id == task.id => {
                saw_started = true;
            }
            AgentEvent::AssistantMessageDelta {
                task_id,
                content: Some(text),
                ..
            } if task_id == task.id && text == "hello" => {
                saw_message_delta = true;
            }
            AgentEvent::ModelRoundFinished {
                task_id,
                assistant_message,
                tool_messages,
                usage,
                ..
            } if task_id == task.id => {
                saw_round_finished = true;
                assert_eq!(assistant_message.content, "hello");
                assert!(tool_messages.is_empty());
                assert_eq!(usage.expect("usage is present").prompt_tokens, 11);
            }
            AgentEvent::Finished { task_id, .. } if task_id == task.id => {
                saw_finished = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_started);
    assert!(saw_message_delta);
    assert!(saw_round_finished);
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
    let config = seekcode_tool_system::SystemToolConfig::new();
    let tools = Arc::new(
        seekcode_tool_system::system_tool_registry(config).expect("system tools register"),
    );
    let expected_tool_count = tools.tool_specs(false).len();
    let (event_sender, mut events) = mpsc::unbounded_channel();

    let task = agent
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Use tools".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Use tools"),
            tools,
            event_sender,
            None,
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
                tool_count,
                ..
            } if task_id == task.id && tool_count == expected_tool_count
        ) {
            saw_tool_count = true;
            break;
        }
    }

    assert!(saw_tool_count);
}

#[tokio::test]
async fn start_task_retries_when_model_stream_open_fails() {
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut attempts = std::collections::VecDeque::new();
    attempts.push_back(ProviderAttempt::OpenError("temporary outage".to_string()));
    attempts.push_back(ProviderAttempt::Stream(vec![
        StreamItem::Chunk(ChatChunk::Content("done".to_string())),
        StreamItem::Chunk(ChatChunk::Finished),
    ]));
    let agent = Agent::new(
        AgentConfig::default(),
        Arc::new(RetryProvider {
            attempts: std::sync::Mutex::new(attempts),
            requests: Arc::clone(&requests),
        }),
        Arc::new(ToolRegistry::new()),
    );
    let (event_sender, mut events) = mpsc::unbounded_channel();

    let task = agent
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Retry model".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Retry model"),
            Arc::new(ToolRegistry::new()),
            event_sender,
            None,
        )
        .await
        .expect("task starts");

    let mut saw_retry = false;
    let mut saw_final_delta = false;
    for _ in 0..20 {
        let event = events.recv().await.expect("event is published");
        match event {
            AgentEvent::ModelRequestRetrying {
                task_id,
                round_id,
                retry_count,
                max_retries,
                error,
                ..
            } if task_id == task.id => {
                assert_eq!(round_id, 1);
                assert_eq!(retry_count, 1);
                assert_eq!(max_retries, 5);
                assert!(error.contains("temporary outage"));
                saw_retry = true;
            }
            AgentEvent::AssistantMessageDelta {
                task_id,
                content: Some(text),
                ..
            } if task_id == task.id && text == "done" => {
                saw_final_delta = true;
            }
            AgentEvent::Failed { task_id, error, .. } if task_id == task.id => {
                panic!("model request should retry instead of failing immediately: {error}");
            }
            AgentEvent::Finished { task_id, .. } if task_id == task.id => break,
            _ => {}
        }
    }

    assert!(saw_retry);
    assert!(saw_final_delta);
    assert_eq!(requests.lock().expect("requests lock").len(), 2);
}

#[tokio::test]
async fn start_task_retries_stream_chunk_failure_and_discards_partial_round_data() {
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut attempts = std::collections::VecDeque::new();
    attempts.push_back(ProviderAttempt::Stream(vec![
        StreamItem::Chunk(ChatChunk::Content("partial".to_string())),
        StreamItem::Error("stream disconnected".to_string()),
    ]));
    attempts.push_back(ProviderAttempt::Stream(vec![
        StreamItem::Chunk(ChatChunk::Content("final".to_string())),
        StreamItem::Chunk(ChatChunk::Finished),
    ]));
    let agent = Agent::new(
        AgentConfig::default(),
        Arc::new(RetryProvider {
            attempts: std::sync::Mutex::new(attempts),
            requests: Arc::clone(&requests),
        }),
        Arc::new(ToolRegistry::new()),
    );
    let (event_sender, mut events) = mpsc::unbounded_channel();

    let task = agent
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Retry stream".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Retry stream"),
            Arc::new(ToolRegistry::new()),
            event_sender,
            None,
        )
        .await
        .expect("task starts");

    let mut saw_partial_delta = false;
    let mut saw_retry = false;
    let mut finished_content = None;
    for _ in 0..30 {
        let event = events.recv().await.expect("event is published");
        match event {
            AgentEvent::AssistantMessageDelta {
                task_id,
                content: Some(text),
                ..
            } if task_id == task.id && text == "partial" => {
                saw_partial_delta = true;
            }
            AgentEvent::ModelRequestRetrying { task_id, error, .. } if task_id == task.id => {
                assert!(error.contains("stream disconnected"));
                saw_retry = true;
            }
            AgentEvent::ModelRoundFinished {
                task_id,
                assistant_message,
                ..
            } if task_id == task.id => {
                finished_content = Some(assistant_message.content);
            }
            AgentEvent::Failed { task_id, error, .. } if task_id == task.id => {
                panic!("stream failure should retry instead of failing immediately: {error}");
            }
            AgentEvent::Finished { task_id, .. } if task_id == task.id => break,
            _ => {}
        }
    }

    assert!(saw_partial_delta);
    assert!(saw_retry);
    assert_eq!(finished_content.as_deref(), Some("final"));
    assert_eq!(requests.lock().expect("requests lock").len(), 2);
}

#[tokio::test]
async fn start_task_continues_after_tool_result() {
    let workspace_root =
        std::env::temp_dir().join(format!("seekcode-agent-core-search-{}", WorkspaceId::new()));
    std::fs::create_dir_all(&workspace_root).expect("create temp workspace");
    std::fs::write(workspace_root.join("fixture.txt"), "needle\n").expect("write fixture");

    let mut rounds = std::collections::VecDeque::new();
    rounds.push_back(vec![
        ChatChunk::Choice(seekcode_deepseek_client::ChatChoiceChunk {
            delta: seekcode_deepseek_client::ChatDelta {
                content: None,
                reasoning_content: None,
                tool_calls: vec![seekcode_deepseek_client::ToolCallDelta {
                    index: 0,
                    id: Some("call_".to_string()),
                    kind: Some("function".to_string()),
                    name: Some("run_c".to_string()),
                    arguments: Some(r#"{"command":"echo"#.to_string()),
                }],
            },
            finish_reason: None,
        }),
        ChatChunk::Choice(seekcode_deepseek_client::ChatChoiceChunk {
            delta: seekcode_deepseek_client::ChatDelta {
                content: None,
                reasoning_content: None,
                tool_calls: vec![seekcode_deepseek_client::ToolCallDelta {
                    index: 0,
                    id: Some("cmd_".to_string()),
                    kind: None,
                    name: Some("ommand".to_string()),
                    arguments: Some(r#" hello"}"#.to_string()),
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
        ChatChunk::Finished,
    ]);
    let config = seekcode_tool_system::SystemToolConfig::new();
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
        .start_task_with_messages_tools_tool_context_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "List files".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Search files"),
            tools,
            AgentToolContext::workspace(WorkspaceId::new(), workspace_root.clone()),
            event_sender,
            None,
        )
        .await
        .expect("task starts");

    let mut saw_tool_finished = false;
    let mut saw_tool_started_with_display = false;
    let mut saw_tool_round_finished = false;
    let mut saw_second_round = false;
    let mut saw_final_delta = false;
    for _ in 0..30 {
        let event = events.recv().await.expect("event is published");
        match event {
            AgentEvent::ToolCallStarted {
                task_id,
                tool_call_id,
                name,
                display,
                ..
            } if task_id == task.id => {
                assert_eq!(name, seekcode_tool_system::RUN_COMMAND_TOOL);
                let display = display.expect("tool display is available when tool starts");
                assert_eq!(tool_call_id, "call_cmd_");
                assert_eq!(display.title, "Shell");
                assert!(display.preview.contains("echo hello"));
                saw_tool_started_with_display = true;
            }
            AgentEvent::ToolCallFinished { task_id, ok, .. } if task_id == task.id && ok => {
                assert!(
                    saw_tool_started_with_display,
                    "tool display should be published before the tool finishes"
                );
                saw_tool_finished = true;
            }
            AgentEvent::ModelRoundFinished {
                task_id,
                round_id: 1,
                assistant_message,
                tool_messages,
                usage,
                ..
            } if task_id == task.id => {
                saw_tool_round_finished = true;
                assert_eq!(assistant_message.tool_calls.len(), 1);
                assert_eq!(tool_messages.len(), 1);
                assert_eq!(usage.expect("usage is present").prompt_tokens, 21);
            }
            AgentEvent::ModelRequestStarted {
                task_id,
                round_id: 2,
                ..
            } if task_id == task.id => {
                saw_second_round = true;
            }
            AgentEvent::AssistantMessageDelta {
                task_id,
                content: Some(text),
                ..
            } if task_id == task.id && text == "done" => {
                saw_final_delta = true;
            }
            AgentEvent::Finished { task_id, .. } if task_id == task.id => break,
            _ => {}
        }
    }

    assert!(saw_tool_finished);
    assert!(saw_tool_started_with_display);
    assert!(saw_tool_round_finished);
    assert!(saw_second_round);
    assert!(saw_final_delta);
    let _ = std::fs::remove_dir_all(workspace_root);
}

#[tokio::test]
async fn start_task_returns_failed_tool_result_to_model() {
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut rounds = std::collections::VecDeque::new();
    rounds.push_back(vec![
        ChatChunk::Choice(seekcode_deepseek_client::ChatChoiceChunk {
            delta: seekcode_deepseek_client::ChatDelta {
                content: None,
                reasoning_content: None,
                tool_calls: vec![seekcode_deepseek_client::ToolCallDelta {
                    index: 0,
                    id: Some("call_missing".to_string()),
                    kind: Some("function".to_string()),
                    name: Some("missing_tool".to_string()),
                    arguments: Some(r#"{"value":true}"#.to_string()),
                }],
            },
            finish_reason: Some("tool_calls".to_string()),
        }),
        ChatChunk::Finished,
    ]);
    rounds.push_back(vec![
        ChatChunk::Content("recovered".to_string()),
        ChatChunk::Finished,
    ]);
    let agent = Agent::new(
        AgentConfig::default(),
        Arc::new(RecordingRoundProvider {
            rounds: std::sync::Mutex::new(rounds),
            requests: Arc::clone(&requests),
        }),
        Arc::new(ToolRegistry::new()),
    );
    let (event_sender, mut events) = mpsc::unbounded_channel();

    let task = agent
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Use a missing tool".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("Use a missing tool"),
            Arc::new(ToolRegistry::new()),
            event_sender,
            None,
        )
        .await
        .expect("task starts");

    let mut saw_failed_tool_event = false;
    let mut saw_second_round = false;
    let mut saw_final_delta = false;
    for _ in 0..30 {
        let event = events.recv().await.expect("event is published");
        match event {
            AgentEvent::ToolCallFinished {
                task_id, ok, error, ..
            } if task_id == task.id && !ok => {
                let error = error.expect("failed tool event includes error");
                assert!(error.contains("missing_tool"));
                saw_failed_tool_event = true;
            }
            AgentEvent::ModelRequestStarted {
                task_id,
                round_id: 2,
                ..
            } if task_id == task.id => {
                saw_second_round = true;
            }
            AgentEvent::AssistantMessageDelta {
                task_id,
                content: Some(text),
                ..
            } if task_id == task.id && text == "recovered" => {
                saw_final_delta = true;
            }
            AgentEvent::Failed { task_id, error, .. } if task_id == task.id => {
                panic!("tool failure should be returned to the model, not fail the task: {error}");
            }
            AgentEvent::Finished { task_id, .. } if task_id == task.id => break,
            _ => {}
        }
    }

    let requests = requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 2);
    let tool_message = requests[1]
        .messages
        .iter()
        .find(|message| message.role == ChatRole::Tool)
        .expect("second request includes failed tool result");
    assert_eq!(tool_message.tool_call_id.as_deref(), Some("call_missing"));
    let content: serde_json::Value =
        serde_json::from_str(&tool_message.content).expect("tool failure content is json");
    assert_eq!(content["ok"], false);
    assert!(content["error"]
        .as_str()
        .expect("error is a string")
        .contains("missing_tool"));
    assert!(saw_failed_tool_event);
    assert!(saw_second_round);
    assert!(saw_final_delta);
}

/// Context preparer that records in-loop compaction calls and returns a summary.
struct RecordingRunningPreparer {
    compact_calls: Arc<std::sync::Mutex<usize>>,
}

#[async_trait]
impl AgentContextPreparer for RecordingRunningPreparer {
    async fn prepare_context(
        &self,
        _task_id: TaskId,
        _session_id: SessionId,
        _model: &str,
        _prompt: &str,
        current_context: &AgentTaskContext,
        _precheck: AgentContextPrecheck,
    ) -> SeekCodeResult<PreparedAgentContext> {
        Ok(PreparedAgentContext {
            context: current_context.clone(),
            compaction: None,
        })
    }

    async fn compact_running_context(
        &self,
        _task_id: TaskId,
        _session_id: SessionId,
        _model: &str,
        _messages_to_compact: &[ChatMessage],
        compacted_through_turn: i64,
    ) -> SeekCodeResult<Option<RunningContextCompaction>> {
        *self.compact_calls.lock().expect("compact calls lock") += 1;
        Ok(Some(RunningContextCompaction {
            summary_message: ChatMessage::new(ChatRole::System, "SUMMARY"),
            outcome: AgentContextCompactionOutcome {
                compacted_rounds: 1,
                compacted_through_turn,
                summary_chars: "SUMMARY".chars().count(),
            },
        }))
    }
}

/// Builds one streamed round that requests a missing tool and reports usage.
fn tool_round_with_usage(prompt_tokens: u32) -> Vec<ChatChunk> {
    vec![
        ChatChunk::Choice(seekcode_deepseek_client::ChatChoiceChunk {
            delta: seekcode_deepseek_client::ChatDelta {
                content: None,
                reasoning_content: None,
                tool_calls: vec![seekcode_deepseek_client::ToolCallDelta {
                    index: 0,
                    id: Some("call_x".to_string()),
                    kind: Some("function".to_string()),
                    name: Some("missing_tool".to_string()),
                    arguments: Some("{}".to_string()),
                }],
            },
            finish_reason: Some("tool_calls".to_string()),
        }),
        ChatChunk::Usage(TokenUsage {
            prompt_tokens,
            completion_tokens: 1,
            total_tokens: prompt_tokens + 1,
            cached_tokens: 0,
        }),
        ChatChunk::Finished,
    ]
}

#[tokio::test]
async fn in_loop_compaction_folds_context_once_past_threshold() {
    // context_window is 1_000, so the 95% in-loop threshold is 950 tokens.
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut rounds = std::collections::VecDeque::new();
    // Rounds 1 and 2 both exceed the threshold and request another round.
    rounds.push_back(tool_round_with_usage(960));
    rounds.push_back(tool_round_with_usage(960));
    // Round 3 stops requesting tools, ending the task.
    rounds.push_back(vec![
        ChatChunk::Content("done".to_string()),
        ChatChunk::Finished,
    ]);

    let agent = Agent::new(
        AgentConfig::default(),
        Arc::new(RecordingRoundProvider {
            rounds: std::sync::Mutex::new(rounds),
            requests: Arc::clone(&requests),
        }),
        Arc::new(ToolRegistry::new()),
    );
    let compact_calls = Arc::new(std::sync::Mutex::new(0usize));
    let (event_sender, mut events) = mpsc::unbounded_channel();

    // Seed expanded history so the compactable portion is non-empty.
    let mut context = task_context("Keep calling tools");
    context.history_messages = vec![
        AgentHistoryMessage {
            turn_sequence: 1,
            message: ChatMessage::new(ChatRole::User, "old-user"),
        },
        AgentHistoryMessage {
            turn_sequence: 1,
            message: ChatMessage::new(ChatRole::Assistant, "old-assistant"),
        },
    ];

    let task = agent
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id: SessionId::new(),
                prompt: "Keep calling tools".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            context,
            Arc::new(ToolRegistry::new()),
            event_sender,
            Some(Arc::new(RecordingRunningPreparer {
                compact_calls: Arc::clone(&compact_calls),
            })),
        )
        .await
        .expect("task starts");

    let mut saw_compaction_started = false;
    let mut saw_compaction_finished = false;
    for _ in 0..40 {
        let event = events.recv().await.expect("event is published");
        match event {
            AgentEvent::ContextCompactionStarted { task_id, .. } if task_id == task.id => {
                saw_compaction_started = true;
            }
            AgentEvent::ContextCompactionFinished { task_id, .. } if task_id == task.id => {
                saw_compaction_finished = true;
            }
            AgentEvent::Failed { task_id, error, .. } if task_id == task.id => {
                panic!("task should complete, not fail: {error}");
            }
            AgentEvent::Finished { task_id, .. } if task_id == task.id => break,
            _ => {}
        }
    }

    assert!(saw_compaction_started);
    assert!(saw_compaction_finished);
    // Compaction is attempted exactly once even though rounds 1 and 2 both crossed
    // the threshold.
    assert_eq!(*compact_calls.lock().expect("compact calls lock"), 1);

    let requests = requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 3);

    let has_content = |request: &ChatRequest, content: &str| {
        request
            .messages
            .iter()
            .any(|message| message.content == content)
    };
    // Round 1 (before the fold) still carries the expanded history.
    assert!(has_content(&requests[0], "old-assistant"));
    assert!(!has_content(&requests[0], "SUMMARY"));
    // Round 2 (after the fold) replaces the history with the summary but keeps the
    // latest user message and round 1's appended turn.
    assert!(has_content(&requests[1], "SUMMARY"));
    assert!(!has_content(&requests[1], "old-assistant"));
    assert!(has_content(&requests[1], "Keep calling tools"));
    // Round 3 was not folded again: the summary persists, no second compaction.
    assert!(has_content(&requests[2], "SUMMARY"));
}

#[test]
fn append_tool_results_preserves_assistant_reasoning_for_next_round() {
    let tool_call_id = "call_reasoning".to_string();
    let mut messages = Vec::new();

    append_tool_results_to_context(
        &mut messages,
        String::new(),
        "I need to inspect the workspace.".to_string(),
        vec![ToolCallRunResult {
            tool_call: seekcode_deepseek_client::ToolCall {
                id: tool_call_id.clone(),
                name: seekcode_tool_system::RUN_COMMAND_TOOL.to_string(),
                arguments: serde_json::json!({ "path": "fixture.txt" }),
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
        .start_task_with_messages_tools_and_context_preparer(
            StartTaskRequest {
                session_id,
                prompt: "wait".to_string(),
                model: None,
                thinking: None,
                reasoning_effort: None,
            },
            task_context("wait"),
            Arc::new(ToolRegistry::new()),
            event_sender,
            None,
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
