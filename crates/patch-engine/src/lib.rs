//! Diff generation, patch application, rollback, and conflict boundaries.

use seekcode_common::{SeekCodeResult, TaskId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A multi-file patch associated with an agent task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Patch {
    /// Task that produced the patch.
    pub task_id: TaskId,
    /// Files changed by the patch.
    pub files: Vec<PatchFile>,
}

/// Patch data for one file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchFile {
    /// Path relative to the workspace root.
    pub relative_path: PathBuf,
    /// Hunks included in the patch.
    pub hunks: Vec<PatchHunk>,
}

/// One hunk in a file patch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchHunk {
    /// One-based starting line in the original file.
    pub old_start: usize,
    /// Number of original lines.
    pub old_lines: usize,
    /// One-based starting line in the new file.
    pub new_start: usize,
    /// Number of new lines.
    pub new_lines: usize,
    /// Unified diff body.
    pub body: String,
}

/// Result of applying a patch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchApplyResult {
    /// Whether the patch applied cleanly.
    pub applied: bool,
    /// Files touched by the operation.
    pub changed_files: Vec<PathBuf>,
    /// Rollback plan for undoing the operation.
    pub rollback: Option<RollbackPlan>,
}

/// Data required to roll back a patch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackPlan {
    /// Task that created the rollback plan.
    pub task_id: TaskId,
    /// Reverse patch or serialized snapshot metadata.
    pub payload: String,
}

/// Creates a unified diff between two text buffers.
pub fn create_diff(_before: &str, _after: &str) -> SeekCodeResult<String> {
    todo!("create unified diff from text buffers")
}
