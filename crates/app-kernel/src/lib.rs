//! Application-level service composition for the Tauri adapter and future CLI.

mod compaction;
mod config;
mod context;
mod events;
mod kernel;
mod title;
mod types;

pub use config::AppKernelConfig;
pub use kernel::{AppKernel, SessionService};
pub use seekcode_deepseek_client::DEFAULT_CONTEXT_WINDOW;
pub use seekcode_storage::SessionRecord as AppSessionRecord;
pub use types::{
    AppServices, CreateSessionRequest, OpenWorkspaceRequest, SessionTitleChanged, StartedAgentTask,
    WorkspaceWithSessions,
};
