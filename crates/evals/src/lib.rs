//! Agent evaluation task scaffolding.

use seekcode_common::{SeekCodeResult, TaskId};
use serde::{Deserialize, Serialize};

/// Evaluation case definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalCase {
    /// Evaluation name.
    pub name: String,
    /// User prompt used for the eval.
    pub prompt: String,
}

/// Evaluation result placeholder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalResult {
    /// Task associated with the eval run.
    pub task_id: TaskId,
    /// Whether the eval passed.
    pub passed: bool,
    /// Human-readable notes.
    pub notes: String,
}

/// Runs one evaluation case.
pub async fn run_eval(_case: EvalCase) -> SeekCodeResult<EvalResult> {
    todo!("run agent evaluation case")
}
