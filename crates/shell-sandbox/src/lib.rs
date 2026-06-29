//! Command execution, timeout, output truncation, and environment isolation.

use futures_util::stream::BoxStream;
use seekcode_common::{SeekCodeResult, TaskId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

/// Request to run a command in a workspace-scoped sandbox.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommandRequest {
    /// Task that requested the command.
    pub task_id: Option<TaskId>,
    /// Program to execute.
    pub program: String,
    /// Program arguments.
    pub args: Vec<String>,
    /// Working directory.
    pub cwd: PathBuf,
    /// Extra environment variables.
    pub env: BTreeMap<String, String>,
    /// Timeout for the command.
    #[serde(with = "duration_millis")]
    pub timeout: Duration,
}

/// Completed command output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommandOutput {
    /// Process exit code, if available.
    pub exit_code: Option<i32>,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
    /// Whether output was truncated.
    pub truncated: bool,
}

/// Streaming command event.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum CommandEvent {
    /// Process has started.
    Started { process_id: u64 },
    /// Stdout chunk.
    Stdout { chunk: String },
    /// Stderr chunk.
    Stderr { chunk: String },
    /// Process exited.
    Exited { output: CommandOutput },
}

/// Shell sandbox policy knobs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Maximum output bytes retained in memory.
    pub max_output_bytes: usize,
    /// Whether shell builtins are allowed.
    pub allow_shell: bool,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            max_output_bytes: 512 * 1024,
            allow_shell: false,
        }
    }
}

/// Command runner boundary.
pub struct CommandRunner {
    policy: SandboxPolicy,
}

impl CommandRunner {
    /// Creates a command runner.
    pub fn new(policy: SandboxPolicy) -> Self {
        Self { policy }
    }

    /// Returns the active sandbox policy.
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Runs a command and collects output.
    pub async fn run(&self, _request: CommandRequest) -> SeekCodeResult<CommandOutput> {
        todo!("run command in sandbox")
    }

    /// Streams command output events.
    pub fn stream(
        &self,
        _request: CommandRequest,
    ) -> SeekCodeResult<BoxStream<'static, SeekCodeResult<CommandEvent>>> {
        todo!("stream command output from sandbox")
    }

    /// Kills a running process.
    pub async fn kill(&self, _process_id: u64) -> SeekCodeResult<()> {
        todo!("kill sandboxed process")
    }
}

mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(value.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}
