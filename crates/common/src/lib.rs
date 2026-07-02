//! Shared DTOs, IDs, error types, timestamps, and redaction helpers.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::{self, Debug, Display};
use std::path::Path;
use thiserror::Error;
use tracing_subscriber::{fmt as tracing_fmt, EnvFilter};
use ulid::Ulid;

/// Result type used across SeekCode backend crates.
pub type SeekCodeResult<T> = Result<T, SeekCodeError>;

/// UTC timestamp type used in persisted and streamed DTOs.
pub type UtcDateTime = DateTime<Utc>;

/// Telemetry configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Env-filter compatible tracing directive.
    pub filter: String,
    /// Whether ANSI colors should be emitted.
    pub ansi: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            filter: "info,seekcode=debug".to_string(),
            ansi: true,
        }
    }
}

/// Initializes tracing for the desktop application.
pub fn init_tracing(config: &TelemetryConfig) -> SeekCodeResult<()> {
    let filter = EnvFilter::try_new(&config.filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = tracing_fmt()
        .with_env_filter(filter)
        .with_ansi(config.ansi)
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}

macro_rules! id_type {
    ($name:ident) => {
        #[doc = concat!("Stable ULID-backed identifier for ", stringify!($name), ".")]
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
        )]
        pub struct $name(pub Ulid);

        impl $name {
            /// Creates a new time-sortable identifier.
            pub fn new() -> Self {
                Self(Ulid::new())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                Display::fmt(&self.0, f)
            }
        }

        impl std::str::FromStr for $name {
            type Err = ulid::DecodeError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                value.parse::<Ulid>().map(Self)
            }
        }
    };
}

id_type!(SessionId);
id_type!(TaskId);
id_type!(WorkspaceId);
id_type!(ModelCallLogId);

/// Unified error type for scaffolded backend services.
#[derive(Debug, Error)]
pub enum SeekCodeError {
    /// A requested entity does not exist.
    #[error("not found: {0}")]
    NotFound(String),
    /// A request failed validation before execution.
    #[error("validation failed: {0}")]
    Validation(String),
    /// A local policy rejected the action.
    #[error("policy denied action: {0}")]
    PolicyDenied(String),
    /// A model provider returned an error.
    #[error("model provider error: {0}")]
    ModelProvider(String),
    /// A tool failed before producing a valid result.
    #[error("tool execution failed: {0}")]
    ToolExecution(String),
    /// Workspace operations failed.
    #[error("workspace error: {0}")]
    Workspace(String),
    /// Patch operations failed.
    #[error("patch error: {0}")]
    Patch(String),
    /// Shell operations failed.
    #[error("shell error: {0}")]
    Shell(String),
    /// Storage operations failed.
    #[error("storage error: {0}")]
    Storage(String),
    /// Secret operations failed.
    #[error("secret error: {0}")]
    Secret(String),
    /// A dependency has not been implemented yet.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// An unexpected internal error occurred.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Role for a chat message exchanged with a model provider.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    /// System-level instruction.
    System,
    /// End-user message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool result message.
    Tool,
}

/// Basic chat message DTO shared by providers and storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role.
    pub role: ChatRole,
    /// Plain text content.
    pub content: String,
    /// Optional provider-specific reasoning content.
    pub reasoning_content: Option<String>,
    /// Assistant tool calls attached to this message.
    pub tool_calls: Vec<Value>,
    /// Tool call identifier for tool result messages.
    pub tool_call_id: Option<String>,
    /// Creation timestamp.
    pub created_at: UtcDateTime,
}

impl ChatMessage {
    /// Creates a new chat message with the current UTC timestamp.
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            created_at: Utc::now(),
        }
    }
}

/// Token accounting returned by a model provider.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of prompt tokens.
    pub prompt_tokens: u32,
    /// Number of completion tokens.
    pub completion_tokens: u32,
    /// Total number of billed tokens.
    pub total_tokens: u32,
    /// Number of prompt tokens served from provider cache.
    pub cached_tokens: u32,
}

/// Event emitted by backend streams to the Tauri adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Assistant text token.
    Token { task_id: TaskId, text: String },
    /// Assistant reasoning text token.
    Reasoning { task_id: TaskId, text: String },
    /// Tool call has been requested.
    ToolCall {
        task_id: TaskId,
        tool_call_id: String,
        name: String,
    },
    /// Tool execution produced a result.
    ToolResult {
        task_id: TaskId,
        tool_call_id: String,
        ok: bool,
    },
    /// Task completed successfully.
    TaskFinished { task_id: TaskId },
    /// Task failed.
    Error {
        task_id: Option<TaskId>,
        message: String,
    },
}

/// Redacts a secret-like value while keeping a small prefix for diagnostics.
pub fn redact_secret(value: &str) -> String {
    if value.len() <= 8 {
        return "********".to_string();
    }

    format!("{}…{}", &value[..4], &value[value.len() - 4..])
}

/// Redacts a filesystem path to a stable display string.
pub fn redact_path(path: impl AsRef<Path>) -> String {
    path.as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("…/{name}"))
        .unwrap_or_else(|| "…".to_string())
}

/// Wrapper that prevents accidental debug logging of sensitive values.
#[derive(Clone, Serialize, Deserialize)]
pub struct Redacted<T>(pub T);

impl<T> Debug for Redacted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl<T> Redacted<T> {
    /// Returns a shared reference to the wrapped value.
    pub fn expose(&self) -> &T {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_serializes_as_ulid() {
        let id = SessionId::new();
        let json = serde_json::to_string(&id).expect("serialize session id");
        let restored: SessionId = serde_json::from_str(&json).expect("deserialize session id");

        assert_eq!(id, restored);
        assert_eq!(id.to_string().len(), 26);
    }
}
