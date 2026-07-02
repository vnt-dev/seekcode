//! Shared test fixtures for the app-kernel crate: deterministic timestamps,
//! session/message seeding helpers, and stub model providers.

use async_trait::async_trait;
use seekcode_common::{ChatRole, SeekCodeResult, SessionId, WorkspaceId};
use seekcode_deepseek_client::{
    ChatChunk, ChatRequest, ChatResponse, ChatStream, ModelProfile, ModelProvider,
};
use seekcode_storage::{
    NewSession, NewSessionMessage, NewWorkspace, SessionMessageRecord, SessionStore, SqliteStorage,
    WorkspaceStore,
};
use serde_json::Value;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

static TEST_MESSAGE_CLOCK: AtomicI64 = AtomicI64::new(0);

pub(crate) fn next_test_created_at() -> String {
    let offset = TEST_MESSAGE_CLOCK.fetch_add(1, Ordering::Relaxed);
    format!("2026-01-01 00:{:02}:{:02}", (offset / 60) % 60, offset % 60)
}

pub(crate) fn test_message_record(
    session_id: SessionId,
    turn_sequence: i64,
    role: ChatRole,
    tool_calls: Vec<Value>,
) -> SessionMessageRecord {
    SessionMessageRecord {
        id: 0,
        session_id,
        turn_sequence,
        role,
        content: String::new(),
        reasoning_content: None,
        tool_calls,
        tool_call_id: None,
        created_at: next_test_created_at(),
    }
}

pub(crate) struct CapturingProvider {
    pub(crate) context_window: u32,
    pub(crate) summary: String,
    pub(crate) requests: Arc<std::sync::Mutex<Vec<ChatRequest>>>,
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

pub(crate) struct StreamingProvider {
    pub(crate) rounds: std::sync::Mutex<std::collections::VecDeque<Vec<ChatChunk>>>,
    pub(crate) requests: Arc<std::sync::Mutex<Vec<ChatRequest>>>,
}

#[async_trait]
impl ModelProvider for StreamingProvider {
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
        Ok(Box::pin(futures_util::stream::iter(chunks)))
    }

    async fn complete_chat(&self, _request: ChatRequest) -> SeekCodeResult<ChatResponse> {
        Ok(ChatResponse {
            content: "summary".to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            usage: None,
        })
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

pub(crate) async fn seed_session(storage: &SqliteStorage) -> SessionId {
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

pub(crate) async fn seed_user_turn(
    storage: &SqliteStorage,
    session_id: SessionId,
    turn: i64,
    text: &str,
) {
    storage
        .append_session_message(NewSessionMessage {
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

pub(crate) async fn seed_assistant_turn(
    storage: &SqliteStorage,
    session_id: SessionId,
    turn: i64,
    content: &str,
    reasoning_content: Option<&str>,
    tool_calls: Vec<serde_json::Value>,
) {
    storage
        .append_session_message(NewSessionMessage {
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

pub(crate) async fn seed_tool_result(
    storage: &SqliteStorage,
    session_id: SessionId,
    turn: i64,
    tool_call_id: String,
    content: &str,
) {
    storage
        .append_session_message(NewSessionMessage {
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

pub(crate) fn tool_call_json(tool_call_id: String) -> serde_json::Value {
    serde_json::json!({
        "id": tool_call_id.to_string(),
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": "{\"path\":\"src/lib.rs\"}"
        }
    })
}
