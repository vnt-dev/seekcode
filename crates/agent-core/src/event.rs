//! Agent loop events and the helpers used to publish them.

use seekcode_common::{SeekCodeResult, SessionId, TaskId, TokenUsage, ToolCallId};
use seekcode_tool_system::{
    ToolOutput, INSERT_LINES_TOOL, READ_FILE_TOOL, RUN_COMMAND_TOOL, SEARCH_TEXT_TOOL,
    WRITE_FILE_TOOL,
};
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
    /// One provider choice chunk was emitted.
    ModelChoice {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Complete provider choice chunk.
        choice: seekcode_deepseek_client::ChatChoiceChunk,
    },
    /// Assistant emitted text.
    AssistantToken {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant content delta.
        text: String,
    },
    /// Assistant emitted reasoning text.
    AssistantReasoning {
        /// Persisted session identifier.
        session_id: SessionId,
        /// Task identifier.
        task_id: TaskId,
        /// One-based model round identifier within this task.
        round_id: u32,
        /// Assistant reasoning delta.
        text: String,
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
        tool_call_id: ToolCallId,
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
        tool_call_id: ToolCallId,
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
    tool_call_id: ToolCallId,
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
pub(crate) fn tool_call_display(name: &str, arguments: &Value) -> Option<ToolCallDisplay> {
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
        READ_FILE_TOOL => {
            let path = string_arg(arguments, "path")?;
            let start_line = integer_arg(arguments, "start_line").unwrap_or(1);
            let preview = format!("path={path} start_line={start_line}");
            Some(argument_display(
                preview,
                vec![("path", path), ("start_line", start_line.to_string())],
            ))
        }
        WRITE_FILE_TOOL => {
            let path = string_arg(arguments, "path")?;
            let content = string_arg(arguments, "content").unwrap_or_default();
            let create_dirs = bool_arg(arguments, "create_dirs").unwrap_or(false);
            let preview = format!(
                "path={path} lines={} create_dirs={create_dirs}",
                text_line_count(&content)
            );
            Some(argument_display(
                preview,
                vec![
                    ("path", path),
                    ("create_dirs", create_dirs.to_string()),
                    ("content", content),
                ],
            ))
        }
        INSERT_LINES_TOOL => {
            let path = string_arg(arguments, "path")?;
            let line = integer_arg(arguments, "line")?;
            let content = string_arg(arguments, "content").unwrap_or_default();
            let preview = format!(
                "path={path} after_line={line} lines={}",
                text_line_count(&content)
            );
            Some(argument_display(
                preview,
                vec![
                    ("path", path),
                    ("after_line", line.to_string()),
                    ("content", content),
                ],
            ))
        }
        SEARCH_TEXT_TOOL => {
            let pattern = string_arg(arguments, "pattern")?;
            let path = string_arg(arguments, "path").unwrap_or_else(|| ".".to_string());
            let limit = integer_arg(arguments, "limit")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "*".to_string());
            let preview = format!("pattern={pattern:?} path={path} limit={limit}");
            Some(argument_display(
                preview,
                vec![("pattern", pattern), ("path", path), ("limit", limit)],
            ))
        }
        _ => None,
    }
}

fn argument_display(preview: String, entries: Vec<(&'static str, String)>) -> ToolCallDisplay {
    let mut detail = String::new();
    for (index, (key, value)) in entries.into_iter().enumerate() {
        if index > 0 {
            detail.push('\n');
        }
        if value.contains('\n') {
            detail.push_str(key);
            detail.push_str(":\n");
            detail.push_str(&value);
        } else {
            detail.push_str(key);
            detail.push_str(": ");
            detail.push_str(&value);
        }
    }

    ToolCallDisplay {
        title: "Arguments".to_string(),
        preview,
        detail,
    }
}

fn string_arg(arguments: &Value, key: &str) -> Option<String> {
    arguments.get(key)?.as_str().map(ToString::to_string)
}

fn integer_arg(arguments: &Value, key: &str) -> Option<i64> {
    arguments.get(key)?.as_i64()
}

fn bool_arg(arguments: &Value, key: &str) -> Option<bool> {
    arguments.get(key)?.as_bool()
}

fn first_line(value: &str) -> Option<String> {
    value
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToString::to_string)
}

fn text_line_count(value: &str) -> usize {
    value.lines().count().max(1)
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

    #[test]
    fn formats_file_tool_display_without_frontend_parsing() {
        let display = tool_call_display(
            INSERT_LINES_TOOL,
            &json!({
                "path": "src/main.jsx",
                "line": 12,
                "content": "first\nsecond"
            }),
        )
        .expect("insert lines display");

        assert_eq!(display.title, "Arguments");
        assert_eq!(display.preview, "path=src/main.jsx after_line=12 lines=2");
        assert!(display.detail.contains("after_line: 12"));
        assert!(display.detail.contains("content:\nfirst\nsecond"));
    }
}
