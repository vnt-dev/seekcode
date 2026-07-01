//! Task lifecycle types and shared task-state helpers.

use parking_lot::RwLock;
use seekcode_common::{SeekCodeError, SeekCodeResult, SessionId, TaskId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request to start an agent task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StartTaskRequest {
    /// Persisted session identifier bound to the task.
    pub session_id: SessionId,
    /// User prompt.
    pub prompt: String,
    /// Optional model override.
    pub model: Option<String>,
    /// Optional thinking mode override.
    #[serde(default)]
    pub thinking: Option<bool>,
    /// Optional provider-specific reasoning intensity.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Agent task snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTask {
    /// Task identifier.
    pub id: TaskId,
    /// Current task state.
    pub state: AgentState,
}

/// Agent task lifecycle state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentState {
    /// Task is queued.
    Queued,
    /// Task is calling a model provider.
    Thinking,
    /// Task is executing a tool.
    RunningTool,
    /// Task has completed.
    Completed,
    /// Task was canceled.
    Canceled,
    /// Task failed.
    Failed,
}

/// Updates the stored state for a task, returning an error when it is unknown.
pub(crate) async fn set_task_state(
    tasks: &RwLock<HashMap<TaskId, AgentTask>>,
    task_id: TaskId,
    state: AgentState,
) -> SeekCodeResult<()> {
    let mut tasks = tasks.write();
    let task = tasks
        .get_mut(&task_id)
        .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))?;
    task.state = state;
    Ok(())
}

/// Reads the current state for a task, returning an error when it is unknown.
pub(crate) async fn task_state(
    tasks: &RwLock<HashMap<TaskId, AgentTask>>,
    task_id: TaskId,
) -> SeekCodeResult<AgentState> {
    tasks
        .read()
        .get(&task_id)
        .map(|task| task.state.clone())
        .ok_or_else(|| SeekCodeError::NotFound(format!("agent task '{task_id}'")))
}
