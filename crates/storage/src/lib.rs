//! SQLite-backed storage abstractions for sessions and audit logs.

use async_trait::async_trait;
use seekcode_common::{ChatMessage, SeekCodeResult, SessionId, TaskId, ToolCallId, UtcDateTime};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Session metadata persisted locally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Session identifier.
    pub id: SessionId,
    /// Human-readable title.
    pub title: String,
    /// Creation timestamp.
    pub created_at: UtcDateTime,
    /// Last update timestamp.
    pub updated_at: UtcDateTime,
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
    /// Creation timestamp.
    pub created_at: UtcDateTime,
}

/// Root storage service marker.
pub trait Storage: SessionStore + AuditStore + Send + Sync {}

impl<T> Storage for T where T: SessionStore + AuditStore + Send + Sync {}

/// Session persistence API.
#[async_trait]
pub trait SessionStore {
    /// Lists known sessions.
    async fn list_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>>;

    /// Appends a chat message to a session.
    async fn append_message(
        &self,
        session_id: SessionId,
        message: ChatMessage,
    ) -> SeekCodeResult<()>;
}

/// Audit persistence API.
#[async_trait]
pub trait AuditStore {
    /// Writes one audit log record.
    async fn write_audit_log(&self, record: AuditLogRecord) -> SeekCodeResult<()>;
}

/// Database migration runner.
pub struct MigrationRunner;

impl MigrationRunner {
    /// Runs storage migrations.
    pub async fn run(_pool: &sqlx::SqlitePool) -> SeekCodeResult<()> {
        todo!("run SQLite migrations")
    }
}

/// SQLite storage placeholder.
pub struct SqliteStorage {
    pool: sqlx::SqlitePool,
}

impl SqliteStorage {
    /// Creates a storage wrapper from a SQLite pool.
    pub fn new(pool: sqlx::SqlitePool) -> Self {
        Self { pool }
    }

    /// Returns the underlying SQLite pool.
    pub fn pool(&self) -> &sqlx::SqlitePool {
        &self.pool
    }
}

#[async_trait]
impl SessionStore for SqliteStorage {
    async fn list_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>> {
        todo!("list sessions from SQLite")
    }

    async fn append_message(
        &self,
        _session_id: SessionId,
        _message: ChatMessage,
    ) -> SeekCodeResult<()> {
        todo!("append chat message to SQLite")
    }
}

#[async_trait]
impl AuditStore for SqliteStorage {
    async fn write_audit_log(&self, _record: AuditLogRecord) -> SeekCodeResult<()> {
        todo!("write audit log to SQLite")
    }
}
