//! Application-level service composition for the Tauri adapter and future CLI.

mod compaction;
mod config;
mod context;
mod events;
mod kernel;
mod session_service;
mod title;
mod tool_call_display;
mod types;

#[cfg(test)]
mod test_support;

pub use config::AppKernelConfig;
pub use kernel::AppKernel;
pub use seekcode_deepseek_client::DEFAULT_CONTEXT_WINDOW;
pub use seekcode_storage::SessionRecord as AppSessionRecord;
pub use session_service::SessionService;
pub use types::{
    AppServices, CreateSessionRequest, OpenWorkspaceRequest, SessionTitleChanged, StartedAgentTask,
    WorkspaceWithSessions,
};
