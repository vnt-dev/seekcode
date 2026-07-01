//! Integration-style tests for the agent runtime and task loop.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream;
use seekcode_common::{
    ChatRole, SeekCodeError, SeekCodeResult, SessionId, TaskId, ToolCallId, WorkspaceId,
};
use seekcode_deepseek_client::{
    ChatChunk, ChatRequest, ChatResponse, ChatStream, ModelProfile, ModelProvider,
};
use seekcode_tool_system::ToolRegistry;
use tokio::sync::mpsc;

use crate::runner::{append_tool_results_to_context, ToolCallRunResult};
use crate::{
    Agent, AgentConfig, AgentEvent, AgentState, AgentTask, AgentTaskContext, AgentToolContext,
    StartTaskRequest,
};

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
            chunks: vec![ChatChunk::Content("hello".to_string()), ChatChunk::Finished],
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
                    id: None,
                    kind: Some("function".to_string()),
                    name: Some(seekcode_tool_system::SEARCH_TEXT_TOOL.to_string()),
                    arguments: Some(r#"{"pattern":"needle"}"#.to_string()),
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
    let _ = std::fs::remove_dir_all(workspace_root);
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
            tool_call: seekcode_deepseek_client::ToolCall {
                id: tool_call_id,
                name: seekcode_tool_system::SEARCH_TEXT_TOOL.to_string(),
                arguments: serde_json::json!({ "pattern": "needle", "path": "." }),
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
