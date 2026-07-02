//! Agent state machine, task context, and execution loop scaffolding.

mod agent;
mod config;
mod context;
mod event;
mod runner;
mod task;

pub use agent::{Agent, AgentToolContext};
pub use config::AgentConfig;
pub use context::{
    AgentContext, AgentContextCompactionOutcome, AgentContextPrecheck, AgentContextPreparer,
    AgentHistoryMessage, AgentTaskContext, PreparedAgentContext,
};
pub use event::{tool_call_display, AgentEvent, ToolCallDisplay};
pub use task::{AgentState, AgentTask, StartTaskRequest};

#[cfg(test)]
mod tests;
