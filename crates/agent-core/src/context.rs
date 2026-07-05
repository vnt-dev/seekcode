//! Conversation context assembled for an agent task.

use async_trait::async_trait;
use seekcode_common::{ChatMessage, SeekCodeResult, SessionId, TaskId};
use serde::{Deserialize, Serialize};

/// Context assembled for an agent task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentContext {
    /// Persisted session identifier bound to the task.
    pub session_id: SessionId,
    /// Task identifier.
    pub task_id: TaskId,
    /// Segmented conversation context used for the next provider request.
    pub task_context: AgentTaskContext,
}

impl AgentContext {
    /// Wraps an application-assembled context for a task.
    pub(crate) fn new(
        task_id: TaskId,
        session_id: SessionId,
        task_context: AgentTaskContext,
    ) -> Self {
        Self {
            session_id,
            task_id,
            task_context,
        }
    }
}

/// Segmented context supplied to a task.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentTaskContext {
    /// Previous request input token count recorded for the session.
    pub last_input_tokens: i64,
    /// System prompt messages.
    pub system_prompt: Vec<ChatMessage>,
    /// General user-role context placed immediately after system messages.
    pub general_prompt: Vec<ChatMessage>,
    /// Previously compacted context messages.
    pub compacted_context: Vec<ChatMessage>,
    /// Persisted history messages that still remain expanded.
    pub history_messages: Vec<AgentHistoryMessage>,
    /// Latest user messages for the task.
    pub latest_user_messages: Vec<ChatMessage>,
}

impl AgentTaskContext {
    /// Flattens the segmented context into provider request messages.
    pub fn messages(&self) -> Vec<ChatMessage> {
        let mut messages = Vec::with_capacity(
            self.system_prompt.len()
                + self.general_prompt.len()
                + self.compacted_context.len()
                + self.history_messages.len()
                + self.latest_user_messages.len(),
        );
        messages.extend(self.system_prompt.iter().cloned());
        messages.extend(self.general_prompt.iter().cloned());
        messages.extend(self.compacted_context.iter().cloned());
        messages.extend(
            self.history_messages
                .iter()
                .map(|history| history.message.clone()),
        );
        messages.extend(self.latest_user_messages.iter().cloned());
        messages
    }

    /// Replaces compacted and history portions after a successful compaction.
    pub fn replace_compacted_context(
        &mut self,
        compacted_context: Vec<ChatMessage>,
        compacted_through_turn: i64,
    ) {
        self.compacted_context = compacted_context;
        self.history_messages
            .retain(|message| message.turn_sequence > compacted_through_turn);
    }

    /// Returns the compacted summary plus expanded history — the portion of the
    /// context eligible for in-loop compaction.
    pub fn compactable_messages(&self) -> Vec<ChatMessage> {
        let mut messages =
            Vec::with_capacity(self.compacted_context.len() + self.history_messages.len());
        messages.extend(self.compacted_context.iter().cloned());
        messages.extend(
            self.history_messages
                .iter()
                .map(|history| history.message.clone()),
        );
        messages
    }

    /// Folds the compactable portion into a fresh summary: replaces the compacted
    /// context and clears all expanded history messages.
    pub fn replace_compactable_with_summary(&mut self, compacted_context: Vec<ChatMessage>) {
        self.compacted_context = compacted_context;
        self.history_messages.clear();
    }
}

/// A history message with the persisted turn it belongs to.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentHistoryMessage {
    /// Persisted conversation turn.
    pub turn_sequence: i64,
    /// Message sent to the model.
    pub message: ChatMessage,
}

/// Prepared context returned by an application-level preflight hook.
pub struct PreparedAgentContext {
    /// Conversation context to use for the next provider request.
    pub context: AgentTaskContext,
    /// Context compaction result, present only when compaction was performed.
    pub compaction: Option<AgentContextCompactionOutcome>,
}

/// Result of the lightweight context preparation precheck.
#[derive(Clone, Debug, Default)]
pub struct AgentContextPrecheck {
    /// Whether context compaction should be attempted.
    pub should_compact_context: bool,
}

/// Metadata known after context compaction finishes.
#[derive(Clone, Debug)]
pub struct AgentContextCompactionOutcome {
    /// Number of conversation rounds folded into the summary.
    pub compacted_rounds: usize,
    /// Highest turn sequence now covered by the summary.
    pub compacted_through_turn: i64,
    /// Character length of the produced summary.
    pub summary_chars: usize,
}

/// Result of compacting the live message list during the running round loop.
pub struct RunningContextCompaction {
    /// Summary message that replaces the folded portion of the running context.
    pub summary_message: ChatMessage,
    /// Statistics describing the compaction that was performed.
    pub outcome: AgentContextCompactionOutcome,
}

/// Optional application hook that can refresh context after a task has started.
#[async_trait]
pub trait AgentContextPreparer: Send + Sync {
    /// Performs a lightweight check before preparing the context.
    async fn precheck_context(
        &self,
        _task_id: TaskId,
        _session_id: SessionId,
        _model: &str,
        _prompt: &str,
        _current_context: &AgentTaskContext,
    ) -> SeekCodeResult<AgentContextPrecheck> {
        Ok(AgentContextPrecheck::default())
    }

    /// Prepares the message list used by the runner before the first model call.
    async fn prepare_context(
        &self,
        task_id: TaskId,
        session_id: SessionId,
        model: &str,
        prompt: &str,
        current_context: &AgentTaskContext,
        precheck: AgentContextPrecheck,
    ) -> SeekCodeResult<PreparedAgentContext>;

    /// Folds the given running-loop messages into a single summary message.
    ///
    /// Called mid-run when the live context grows past the in-loop trigger; the
    /// returned summary replaces the folded portion of the running message list.
    /// Returns `Ok(None)` when no summary could be produced.
    async fn compact_running_context(
        &self,
        _task_id: TaskId,
        _session_id: SessionId,
        _model: &str,
        _messages_to_compact: &[ChatMessage],
        _compacted_through_turn: i64,
    ) -> SeekCodeResult<Option<RunningContextCompaction>> {
        Ok(None)
    }
}
