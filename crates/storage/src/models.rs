use seekcode_common::{ChatRole, ModelCallLogId, SessionId, TaskId, ToolCallId, WorkspaceId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Workspace metadata persisted locally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceRecord {
    /// Workspace identifier.
    pub id: WorkspaceId,
    /// Human-readable workspace name.
    pub name: String,
    /// Canonical absolute workspace path.
    pub absolute_path: String,
    /// Whether the workspace should be shown in the UI.
    pub is_visible: bool,
    /// Local creation timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub created_at: String,
    /// Local update timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub updated_at: String,
}

/// New workspace data accepted by storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewWorkspace {
    /// Workspace identifier.
    pub id: WorkspaceId,
    /// Human-readable workspace name.
    pub name: String,
    /// Canonical absolute workspace path.
    pub absolute_path: String,
    /// Whether the workspace should be shown in the UI.
    pub is_visible: bool,
}

/// Session metadata persisted locally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Session identifier.
    pub id: SessionId,
    /// Parent workspace identifier.
    pub workspace_id: WorkspaceId,
    /// Human-readable session name.
    pub name: String,
    /// Model provider name.
    pub model_provider: String,
    /// Model identifier.
    pub model: String,
    /// Whether thinking mode is enabled.
    pub thinking_enabled: bool,
    /// Optional provider-specific reasoning intensity.
    pub reasoning_effort: Option<String>,
    /// Most recent model input token count observed for the session.
    pub last_input_tokens: i64,
    /// Local creation timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub created_at: String,
    /// Local update timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub updated_at: String,
}

/// New session data accepted by storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewSession {
    /// Session identifier.
    pub id: SessionId,
    /// Parent workspace identifier.
    pub workspace_id: WorkspaceId,
    /// Human-readable session name.
    pub name: String,
    /// Model provider name.
    pub model_provider: String,
    /// Model identifier.
    pub model: String,
    /// Whether thinking mode is enabled.
    pub thinking_enabled: bool,
    /// Optional provider-specific reasoning intensity.
    pub reasoning_effort: Option<String>,
}

/// Session message persisted locally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionMessageRecord {
    /// Auto-incremented message row identifier.
    pub id: i64,
    /// Parent session identifier.
    pub session_id: SessionId,
    /// Conversation turn sequence number within the session.
    pub turn_sequence: i64,
    /// Message role.
    pub role: ChatRole,
    /// Message content.
    pub content: String,
    /// Optional provider-specific reasoning content.
    pub reasoning_content: Option<String>,
    /// Assistant tool calls as provider-compatible JSON values.
    pub tool_calls: Vec<Value>,
    /// Tool call identifier for tool result messages.
    pub tool_call_id: Option<ToolCallId>,
    /// Local creation timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub created_at: String,
}

/// New session message data accepted by storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewSessionMessage {
    /// Parent session identifier.
    pub session_id: SessionId,
    /// Conversation turn sequence number within the session.
    pub turn_sequence: i64,
    /// Message role.
    pub role: ChatRole,
    /// Message content.
    pub content: String,
    /// Optional provider-specific reasoning content.
    pub reasoning_content: Option<String>,
    /// Assistant tool calls as provider-compatible JSON values.
    pub tool_calls: Vec<Value>,
    /// Tool call identifier for tool result messages.
    pub tool_call_id: Option<ToolCallId>,
    /// Local creation timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub created_at: String,
}

/// Per-session context compression state persisted locally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionContextStateRecord {
    /// Parent session identifier.
    pub session_id: SessionId,
    /// Compressed summary text covering the compacted history.
    pub summary: String,
    /// Highest turn sequence already folded into the summary.
    pub compacted_through_turn: i64,
    /// Local update timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub updated_at: String,
}

/// Audit log entry for model, tool, file, and shell actions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditLogRecord {
    /// Associated task identifier.
    pub task_id: Option<TaskId>,
    /// Associated tool call identifier.
    pub tool_call_id: Option<ToolCallId>,
    /// Event category.
    pub category: String,
    /// JSON event payload.
    pub payload: Value,
    /// Local creation timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub created_at: String,
}

/// Persisted model provider call telemetry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCallLogRecord {
    /// Log row identifier.
    pub id: ModelCallLogId,
    /// Model provider name.
    pub model_provider: String,
    /// Model identifier.
    pub model: String,
    /// Associated session identifier retained even if the session is deleted.
    pub session_id: SessionId,
    /// Prompt/input token count.
    pub input_tokens: i64,
    /// Completion/output token count.
    pub output_tokens: i64,
    /// Provider cache-hit token count.
    pub cache_hit_tokens: i64,
    /// Request elapsed time in milliseconds.
    pub elapsed_ms: i64,
    /// Whether the provider call succeeded.
    pub success: bool,
    /// Local call start timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub called_at: String,
}

/// Aggregated model call telemetry for one session.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SessionModelCallStats {
    /// Total number of model calls recorded for the session.
    pub call_count: i64,
    /// Sum of prompt/input tokens across all calls.
    pub input_tokens: i64,
    /// Sum of completion/output tokens across all calls.
    pub output_tokens: i64,
    /// Sum of provider cache-hit tokens across all calls.
    pub cache_hit_tokens: i64,
}

/// New model provider call telemetry row.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NewModelCallLog {
    /// Log row identifier.
    pub id: ModelCallLogId,
    /// Model provider name.
    pub model_provider: String,
    /// Model identifier.
    pub model: String,
    /// Associated session identifier retained even if the session is deleted.
    pub session_id: SessionId,
    /// Prompt/input token count.
    pub input_tokens: i64,
    /// Completion/output token count.
    pub output_tokens: i64,
    /// Provider cache-hit token count.
    pub cache_hit_tokens: i64,
    /// Request elapsed time in milliseconds.
    pub elapsed_ms: i64,
    /// Whether the provider call succeeded.
    pub success: bool,
    /// Local call start timestamp formatted as yyyy-MM-dd HH:mm:ss.
    pub called_at: String,
}
