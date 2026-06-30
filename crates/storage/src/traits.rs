use crate::models::{
    AuditLogRecord, ModelCallLogRecord, NewModelCallLog, NewSession, NewSessionMessage,
    NewWorkspace, SessionMessageRecord, SessionRecord, WorkspaceRecord,
};
use async_trait::async_trait;
use seekcode_common::{ChatMessage, SeekCodeResult, SessionId, WorkspaceId};

/// Root storage service marker.
pub trait Storage:
    WorkspaceStore + SessionStore + ModelCallLogStore + AuditStore + Send + Sync
{
}

impl<T> Storage for T where
    T: WorkspaceStore + SessionStore + ModelCallLogStore + AuditStore + Send + Sync
{
}

/// Workspace persistence API.
#[async_trait]
pub trait WorkspaceStore {
    /// Inserts a workspace.
    async fn create_workspace(&self, workspace: NewWorkspace) -> SeekCodeResult<WorkspaceRecord>;

    /// Reads a workspace by id.
    async fn get_workspace(&self, workspace_id: WorkspaceId) -> SeekCodeResult<WorkspaceRecord>;

    /// Finds a workspace by its absolute path.
    async fn find_workspace_by_path(
        &self,
        absolute_path: &str,
    ) -> SeekCodeResult<Option<WorkspaceRecord>>;

    /// Lists visible and hidden workspaces.
    async fn list_workspaces(&self) -> SeekCodeResult<Vec<WorkspaceRecord>>;

    /// Lists workspaces that should be shown in the UI.
    async fn list_visible_workspaces(&self) -> SeekCodeResult<Vec<WorkspaceRecord>>;

    /// Sets whether a workspace should be shown in the UI.
    async fn set_workspace_visibility(
        &self,
        workspace_id: WorkspaceId,
        is_visible: bool,
    ) -> SeekCodeResult<()>;
}

/// Session persistence API.
#[async_trait]
pub trait SessionStore {
    /// Inserts a session.
    async fn create_session(&self, session: NewSession) -> SeekCodeResult<SessionRecord>;

    /// Reads a session by id.
    async fn get_session(&self, session_id: SessionId) -> SeekCodeResult<SessionRecord>;

    /// Updates a session title.
    async fn rename_session(
        &self,
        session_id: SessionId,
        name: String,
    ) -> SeekCodeResult<SessionRecord>;

    /// Updates the model selected for a session.
    async fn update_session_model(
        &self,
        session_id: SessionId,
        model_provider: String,
        model: String,
    ) -> SeekCodeResult<SessionRecord>;

    /// Lists known sessions.
    async fn list_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>>;

    /// Lists sessions for one workspace.
    async fn list_workspace_sessions(
        &self,
        workspace_id: WorkspaceId,
    ) -> SeekCodeResult<Vec<SessionRecord>>;

    /// Deletes a session and cascades its messages.
    async fn delete_session(&self, session_id: SessionId) -> SeekCodeResult<()>;

    /// Deletes all sessions under one workspace and cascades their messages.
    async fn delete_workspace_sessions(&self, workspace_id: WorkspaceId) -> SeekCodeResult<()>;

    /// Inserts a session message with explicit sequence and message type.
    async fn append_session_message(
        &self,
        message: NewSessionMessage,
    ) -> SeekCodeResult<SessionMessageRecord>;

    /// Lists messages for one session.
    async fn list_session_messages(
        &self,
        session_id: SessionId,
    ) -> SeekCodeResult<Vec<SessionMessageRecord>>;

    /// Returns the next conversation turn sequence number.
    async fn next_session_turn_sequence(&self, session_id: SessionId) -> SeekCodeResult<i64>;

    /// Appends a plain text chat message to a session.
    async fn append_message(
        &self,
        session_id: SessionId,
        message: ChatMessage,
    ) -> SeekCodeResult<()>;
}

/// Model call telemetry persistence API.
#[async_trait]
pub trait ModelCallLogStore {
    /// Inserts one model call log row.
    async fn append_model_call_log(
        &self,
        log: NewModelCallLog,
    ) -> SeekCodeResult<ModelCallLogRecord>;
}

/// Audit persistence API.
#[async_trait]
pub trait AuditStore {
    /// Writes one audit log record.
    async fn write_audit_log(&self, record: AuditLogRecord) -> SeekCodeResult<()>;
}
