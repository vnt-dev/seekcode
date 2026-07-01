use crate::kernel::SessionService;
use parking_lot::RwLock;
use seekcode_agent_core::{Agent, AgentEvent, AgentTask};
use seekcode_common::{SessionId, WorkspaceId};
use seekcode_deepseek_client::ModelProvider;
use seekcode_policy::PolicyEngine;
use seekcode_secrets::SecretStore;
use seekcode_shell_sandbox::CommandRunner;
use seekcode_storage::{SessionRecord, Storage, WorkspaceRecord};
use seekcode_tool_system::ToolRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Workspace plus nested sessions for the sidebar.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceWithSessions {
    /// Workspace metadata.
    pub workspace: WorkspaceRecord,
    /// Sessions that belong to the workspace.
    pub sessions: Vec<SessionRecord>,
}

/// Request to open or create a workspace from the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenWorkspaceRequest {
    /// Human-readable workspace name.
    pub name: Option<String>,
    /// Absolute workspace path.
    pub absolute_path: String,
}

/// Request to create a new persisted session.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    /// Parent workspace identifier.
    pub workspace_id: WorkspaceId,
    /// Optional session name.
    pub name: Option<String>,
    /// Model provider name.
    pub model_provider: Option<String>,
    /// Model identifier.
    pub model: Option<String>,
    /// Whether thinking is enabled.
    pub thinking_enabled: Option<bool>,
    /// Optional provider-specific reasoning intensity.
    pub reasoning_effort: Option<String>,
}

/// Concrete services assembled for the application.
pub struct AppServices {
    /// Model provider used by the agent.
    pub provider: Arc<RwLock<Arc<dyn ModelProvider>>>,
    /// Agent runtime.
    pub agent: Arc<Agent>,
    /// Tool registry.
    pub tools: Arc<ToolRegistry>,
    /// Policy engine.
    pub policy: Arc<dyn PolicyEngine>,
    /// Optional durable storage.
    pub storage: Option<Arc<dyn Storage>>,
    /// Session-level application service.
    pub sessions: Arc<SessionService>,
    /// Secret storage.
    pub secrets: Arc<dyn SecretStore>,
    /// Command runner.
    pub shell: Arc<CommandRunner>,
}

/// Started agent task plus the UI event stream for that task.
pub struct StartedAgentTask {
    /// Started task snapshot.
    pub task: AgentTask,
    /// Session-processed events ready for UI forwarding.
    pub events: mpsc::UnboundedReceiver<AgentEvent>,
}

/// Notification emitted when a background title task updates one session.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionTitleChanged {
    /// Session whose title changed.
    pub session_id: SessionId,
    /// Generated display title.
    pub title: String,
}
