//! Agent loop events and the helpers used to publish them.

use seekcode_common::{ChatMessage, SeekCodeResult, SessionId, TaskId, TokenUsage};
use seekcode_tool_system::{ToolOutput, RUN_COMMAND_TOOL};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::task::AgentState;

/// Event produced by the agent loop.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Task was accepted and scheduled.
    TaskStarted {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// Model selected for the task.
        model: String,
    },
    /// Task state changed.
    StateChanged {
        session_id: SessionId,
        task_id: TaskId,
        state: AgentState,
    },
    /// One model request round has started.
    ModelRequestStarted {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Model selected for this request.
        model: String,
        /// Number of messages sent to the provider.
        message_count: usize,
        /// Number of tools exposed to the provider.
        tool_count: usize,
    },
    /// A model request attempt failed and the same round will be retried.
    ModelRequestRetrying {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// One-based retry count for this round.
        retry_count: u32,
        /// Maximum retry count for this round.
        max_retries: u32,
        /// Error text from the failed model request attempt.
        error: String,
    },
    /// Assistant message text deltas were emitted.
    AssistantMessageDelta {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant content delta.
        content: Option<String>,
        /// Assistant reasoning delta.
        reasoning_content: Option<String>,
    },
    /// Tool call execution is about to start.
    ToolCallStarted {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Tool call identifier.
        tool_call_id: String,
        /// Tool name.
        name: String,
        /// Raw JSON arguments.
        arguments: Value,
        /// Preformatted display information for the UI.
        display: Option<ToolCallDisplay>,
    },
    /// Tool call execution has completed.
    ToolCallFinished {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Tool call identifier.
        tool_call_id: String,
        /// Tool name.
        name: String,
        /// Whether execution succeeded.
        ok: bool,
        /// Short result summary.
        summary: Option<String>,
        /// Machine-readable output for detail panels.
        output: Option<Value>,
        /// Error text if execution failed.
        error: Option<String>,
    },
    /// One model request round has finished.
    ModelRoundFinished {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Full assistant message assembled from streamed model output.
        assistant_message: ChatMessage,
        /// Tool result messages produced while handling this model round.
        tool_messages: Vec<ChatMessage>,
        /// Final usage accounting if returned by the provider.
        usage: Option<TokenUsage>,
    },
    /// Task finished.
    Finished {
        session_id: SessionId,
        task_id: TaskId,
    },
    /// Task failed.
    Failed {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// Error text.
        error: String,
    },
    /// Task was canceled.
    Canceled {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
    },
    /// History compression has started for the session context.
    ContextCompactionStarted {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier the compaction runs ahead of.
        task_id: TaskId,
    },
    /// History compression was skipped after it had started.
    ContextCompactionCanceled {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier the compaction ran ahead of.
        task_id: TaskId,
    },
    /// History compression has finished for the session context.
    ContextCompactionFinished {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier the compaction runs ahead of.
        task_id: TaskId,
        /// Number of conversation rounds folded into the summary.
        compacted_rounds: usize,
        /// Highest turn sequence now covered by the summary.
        compacted_through_turn: i64,
        /// Character length of the produced summary.
        summary_chars: usize,
    },
}

impl AgentEvent {
    /// Returns the task identifier attached to this agent event.
    pub fn task_id(&self) -> TaskId {
        match self {
            AgentEvent::TaskStarted { task_id, .. }
            | AgentEvent::StateChanged { task_id, .. }
            | AgentEvent::ModelRequestStarted { task_id, .. }
            | AgentEvent::ModelRequestRetrying { task_id, .. }
            | AgentEvent::AssistantMessageDelta { task_id, .. }
            | AgentEvent::ToolCallStarted { task_id, .. }
            | AgentEvent::ToolCallFinished { task_id, .. }
            | AgentEvent::ModelRoundFinished { task_id, .. }
            | AgentEvent::Finished { task_id, .. }
            | AgentEvent::Failed { task_id, .. }
            | AgentEvent::Canceled { task_id, .. }
            | AgentEvent::ContextCompactionStarted { task_id, .. }
            | AgentEvent::ContextCompactionCanceled { task_id, .. }
            | AgentEvent::ContextCompactionFinished { task_id, .. } => *task_id,
        }
    }

    /// Returns the session identifier attached to this agent event.
    pub fn session_id(&self) -> SessionId {
        match self {
            AgentEvent::TaskStarted { session_id, .. }
            | AgentEvent::StateChanged { session_id, .. }
            | AgentEvent::ModelRequestStarted { session_id, .. }
            | AgentEvent::ModelRequestRetrying { session_id, .. }
            | AgentEvent::AssistantMessageDelta { session_id, .. }
            | AgentEvent::ToolCallStarted { session_id, .. }
            | AgentEvent::ToolCallFinished { session_id, .. }
            | AgentEvent::ModelRoundFinished { session_id, .. }
            | AgentEvent::Finished { session_id, .. }
            | AgentEvent::Failed { session_id, .. }
            | AgentEvent::Canceled { session_id, .. }
            | AgentEvent::ContextCompactionStarted { session_id, .. }
            | AgentEvent::ContextCompactionCanceled { session_id, .. }
            | AgentEvent::ContextCompactionFinished { session_id, .. } => *session_id,
        }
    }

    /// Returns whether this event ends the task stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentEvent::Finished { .. } | AgentEvent::Failed { .. } | AgentEvent::Canceled { .. }
        )
    }
}

/// UI-ready summary for a tool call.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallDisplay {
    /// Label shown above the expanded detail block.
    pub title: String,
    /// Single-line preview shown in the collapsed tool call row.
    pub preview: String,
    /// Full detail shown when the tool call row is expanded.
    pub detail: String,
}

/// Sends an event to the subscriber, ignoring a closed channel.
pub(crate) fn publish(events: &mpsc::UnboundedSender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
}

/// Publishes a `StateChanged` event for the given task.
pub(crate) fn publish_state(
    events: &mpsc::UnboundedSender<AgentEvent>,
    session_id: SessionId,
    task_id: TaskId,
    state: AgentState,
) {
    publish(
        events,
        AgentEvent::StateChanged {
            session_id,
            task_id,
            state,
        },
    );
}

/// Builds a `ToolCallFinished` event from a tool execution result.
pub(crate) fn tool_finished_event(
    task_id: TaskId,
    session_id: SessionId,
    round_id: u32,
    tool_call_id: String,
    name: String,
    result: SeekCodeResult<ToolOutput>,
) -> AgentEvent {
    match result {
        Ok(output) => AgentEvent::ToolCallFinished {
            session_id,
            task_id,
            round_id,
            tool_call_id,
            name,
            ok: true,
            summary: Some(output.summary),
            output: Some(output.content),
            error: None,
        },
        Err(error) => AgentEvent::ToolCallFinished {
            session_id,
            task_id,
            round_id,
            tool_call_id,
            name,
            ok: false,
            summary: None,
            output: None,
            error: Some(error.to_string()),
        },
    }
}

/// Formats tool arguments once in the backend so the UI does not parse tool-specific JSON.
pub fn tool_call_display(name: &str, arguments: &Value) -> Option<ToolCallDisplay> {
    match name {
        RUN_COMMAND_TOOL => {
            let command = string_arg(arguments, "command")?;
            let preview = first_line(&command)?;
            Some(ToolCallDisplay {
                title: "Shell".to_string(),
                preview,
                detail: command,
            })
        }
        _ => None,
    }
}

fn string_arg(arguments: &Value, key: &str) -> Option<String> {
    arguments.get(key)?.as_str().map(ToString::to_string)
}

fn first_line(value: &str) -> Option<String> {
    value
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn formats_run_command_display_from_first_line() {
        let display = tool_call_display(
            RUN_COMMAND_TOOL,
            &json!({
                "command": "cargo test -p seekcode-agent-core\ncargo check --workspace"
            }),
        )
        .expect("run command display");

        assert_eq!(display.title, "Shell");
        assert_eq!(display.preview, "cargo test -p seekcode-agent-core");
        assert!(display.detail.contains("cargo check --workspace"));
    }
}
