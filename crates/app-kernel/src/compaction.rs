//! Context compression: fold older conversation rounds into a summary when the
//! previous request's input tokens approach the model context window.

use crate::kernel::SessionService;
use seekcode_agent_core::{AgentHistoryMessage, AgentTaskContext};
use seekcode_common::{
    ChatMessage, ChatRole, ModelCallLogId, SeekCodeResult, SessionId, TokenUsage,
};
use seekcode_deepseek_client::{ChatRequest, ModelProvider};
use seekcode_storage::{local_now_text, NewModelCallLog};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

/// Percentage of the context window at which compression is triggered.
const COMPACTION_TRIGGER_PERCENT: i64 = 60;
/// Number of most-recent conversation rounds kept fully expanded.
pub(crate) const KEEP_RECENT_ROUNDS: usize = 3;

/// System prompt instructing the model to compress the conversation history.
const COMPACTION_SYSTEM_PROMPT: &str =
    "You compress a coding assistant's conversation history into a dense summary. \
Preserve concrete facts needed to continue the task: user goals and constraints, decisions made, \
files and paths touched, code changes, command results, and any open or pending work. \
Drop pleasantries and redundancy. Do not invent information. Output only the summary text.";

/// Statistics describing a compaction that was performed.
pub(crate) struct CompactionOutcome {
    pub(crate) compacted_rounds: usize,
    pub(crate) compacted_through_turn: i64,
    pub(crate) summary_chars: usize,
    pub(crate) summary: String,
}

/// A concrete plan for which rounds to compress.
struct CompactionPlan {
    /// Highest turn sequence that will be covered by the summary.
    compacted_through_turn: i64,
    /// Turn sequences (ascending) that are folded into the summary.
    turns_to_compress: Vec<i64>,
}

/// Returns the input token count above which compaction should run.
fn compaction_threshold(context_window: u32) -> i64 {
    i64::from(context_window) * COMPACTION_TRIGGER_PERCENT / 100
}

/// Reports whether the previous input token count crossed the trigger threshold.
fn should_compact(last_input_tokens: i64, context_window: u32) -> bool {
    last_input_tokens >= compaction_threshold(context_window)
}

/// Plans which expanded history rounds to compress, keeping the most recent rounds.
fn plan_compaction(sorted_turns: &[i64], keep_recent: usize) -> Option<CompactionPlan> {
    if sorted_turns.len() <= keep_recent {
        return None;
    }

    // Highest turn that is NOT among the kept recent rounds.
    let boundary_index = sorted_turns.len() - keep_recent - 1;
    let compacted_through_turn = sorted_turns[boundary_index];

    let turns_to_compress: Vec<i64> = sorted_turns
        .iter()
        .copied()
        .filter(|turn| *turn <= compacted_through_turn)
        .collect();
    if turns_to_compress.is_empty() {
        return None;
    }

    Some(CompactionPlan {
        compacted_through_turn,
        turns_to_compress,
    })
}

/// Builds a readable transcript of the prior summary plus the rounds to compress.
fn build_history_transcript(
    compacted_context: &[ChatMessage],
    history_messages: &[AgentHistoryMessage],
    turns_to_compress: &[i64],
) -> String {
    let selected: BTreeSet<i64> = turns_to_compress.iter().copied().collect();
    let mut transcript = String::new();

    let previous_summary = compacted_context
        .iter()
        .map(|message| message.content.trim())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    if !previous_summary.trim().is_empty() {
        transcript.push_str("Earlier summary of the conversation so far:\n");
        transcript.push_str(previous_summary.trim());
        transcript.push_str("\n\nAdditional conversation to fold into the summary:\n");
    }

    for history in history_messages {
        if !selected.contains(&history.turn_sequence) {
            continue;
        }
        let message = &history.message;
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        let content = message.content.trim();
        if !content.is_empty() {
            transcript.push_str(&format!("[{role}] {content}\n"));
        }
        if !message.tool_calls.is_empty() {
            transcript.push_str(&format!(
                "[{role}] (requested {} tool call(s))\n",
                message.tool_calls.len()
            ));
        }
    }

    transcript
}

/// Checks whether older rounds before the in-flight user turn should be compacted.
pub(crate) async fn precheck_session_context_compaction(
    provider: Arc<dyn ModelProvider>,
    model: String,
    context: &AgentTaskContext,
) -> SeekCodeResult<bool> {
    let context_window = provider.model_profile(&model).await?.context_window;
    if !should_compact(context.last_input_tokens, context_window) {
        return Ok(false);
    }

    Ok(distinct_sorted_turns(&context.history_messages).len() > KEEP_RECENT_ROUNDS)
}

/// Compacts older rounds before the in-flight user turn.
pub(crate) async fn compact_session_context(
    sessions: Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
    session_id: SessionId,
    model: String,
    context: &AgentTaskContext,
) -> SeekCodeResult<Option<CompactionOutcome>> {
    compact_session_context_inner(sessions, provider, session_id, model, context).await
}

async fn compact_session_context_inner(
    sessions: Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
    session_id: SessionId,
    model: String,
    context: &AgentTaskContext,
) -> SeekCodeResult<Option<CompactionOutcome>> {
    let Some(pending) =
        prepare_session_context_compaction(provider.clone(), model.clone(), context).await?
    else {
        return Ok(None);
    };

    compact_pending_session_context(sessions, provider, session_id, model, pending).await
}

async fn compact_pending_session_context(
    sessions: Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
    session_id: SessionId,
    model: String,
    pending: PendingCompaction,
) -> SeekCodeResult<Option<CompactionOutcome>> {
    let summary =
        summarize_history(&sessions, provider, session_id, model, pending.transcript).await?;
    let summary = summary.trim().to_string();
    if summary.is_empty() {
        return Ok(None);
    }

    let summary_chars = summary.chars().count();
    sessions
        .save_compaction(session_id, summary.clone(), pending.compacted_through_turn)
        .await?;

    Ok(Some(CompactionOutcome {
        compacted_rounds: pending.compacted_rounds,
        compacted_through_turn: pending.compacted_through_turn,
        summary_chars,
        summary,
    }))
}

struct PendingCompaction {
    compacted_rounds: usize,
    compacted_through_turn: i64,
    transcript: String,
}

async fn prepare_session_context_compaction(
    provider: Arc<dyn ModelProvider>,
    model: String,
    context: &AgentTaskContext,
) -> SeekCodeResult<Option<PendingCompaction>> {
    let last_input_tokens = context.last_input_tokens;
    let context_window = provider.model_profile(&model).await?.context_window;
    if !should_compact(last_input_tokens, context_window) {
        return Ok(None);
    }

    let sorted_turns = distinct_sorted_turns(&context.history_messages);
    let Some(plan) = plan_compaction(&sorted_turns, KEEP_RECENT_ROUNDS) else {
        return Ok(None);
    };
    let compacted_rounds = plan.turns_to_compress.len();

    let transcript = build_history_transcript(
        &context.compacted_context,
        &context.history_messages,
        &plan.turns_to_compress,
    );
    Ok(Some(PendingCompaction {
        compacted_rounds,
        compacted_through_turn: plan.compacted_through_turn,
        transcript,
    }))
}

/// Returns the distinct turn sequences present in the records, ascending.
fn distinct_sorted_turns(history_messages: &[AgentHistoryMessage]) -> Vec<i64> {
    history_messages
        .iter()
        .map(|message| message.turn_sequence)
        .collect::<BTreeSet<i64>>()
        .into_iter()
        .collect()
}

pub(crate) fn compacted_history_message(summary: &str) -> ChatMessage {
    ChatMessage::new(
        ChatRole::System,
        format!(
            "<compacted_history>\n{}\n</compacted_history>",
            summary.trim()
        ),
    )
}

/// Runs a one-off model call to produce the summary and logs the call.
async fn summarize_history(
    sessions: &Arc<SessionService>,
    provider: Arc<dyn ModelProvider>,
    session_id: SessionId,
    model: String,
    transcript: String,
) -> SeekCodeResult<String> {
    let model_provider = sessions
        .get_session(session_id)
        .await
        .map(|session| session.model_provider)
        .unwrap_or_else(|_| "deepseek".to_string());

    let called_at = local_now_text();
    let started_at = Instant::now();
    let response = provider
        .complete_chat(ChatRequest {
            model: model.clone(),
            messages: vec![
                ChatMessage::new(ChatRole::System, COMPACTION_SYSTEM_PROMPT),
                ChatMessage::new(ChatRole::User, transcript),
            ],
            tools: Vec::new(),
            thinking: false,
            reasoning_effort: None,
            strict_tools: false,
        })
        .await;
    let elapsed_ms = started_at.elapsed().as_millis().min(i64::MAX as u128) as i64;

    match &response {
        Ok(response) => {
            sessions
                .append_model_call_log(new_model_call_log(
                    model_provider,
                    model,
                    session_id,
                    response.usage.clone(),
                    elapsed_ms,
                    true,
                    called_at,
                ))
                .await;
        }
        Err(_) => {
            sessions
                .append_model_call_log(new_model_call_log(
                    model_provider,
                    model,
                    session_id,
                    None,
                    elapsed_ms,
                    false,
                    called_at,
                ))
                .await;
        }
    }

    Ok(response?.content)
}

fn new_model_call_log(
    model_provider: String,
    model: String,
    session_id: SessionId,
    usage: Option<TokenUsage>,
    elapsed_ms: i64,
    success: bool,
    called_at: String,
) -> NewModelCallLog {
    let usage = usage.unwrap_or_default();
    NewModelCallLog {
        id: ModelCallLogId::new(),
        model_provider,
        model,
        session_id,
        input_tokens: i64::from(usage.prompt_tokens),
        output_tokens: i64::from(usage.completion_tokens),
        cache_hit_tokens: i64::from(usage.cached_tokens),
        elapsed_ms,
        success,
        called_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seekcode_common::SessionId;
    use seekcode_storage::SessionMessageRecord;

    fn record(turn: i64, role: ChatRole, content: &str) -> SessionMessageRecord {
        SessionMessageRecord {
            id: 1,
            session_id: SessionId::new(),
            turn_sequence: turn,
            role,
            content: content.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            created_at: local_now_text(),
        }
    }

    fn record_as_message(record: SessionMessageRecord) -> ChatMessage {
        let mut message = ChatMessage::new(record.role, record.content);
        message.reasoning_content = record.reasoning_content;
        message.tool_calls = record.tool_calls;
        message.tool_call_id = record.tool_call_id;
        message
    }

    #[test]
    fn should_compact_respects_60_percent_threshold() {
        // 64_000 * 60 / 100 = 38_400.
        assert!(!should_compact(38_399, 64_000));
        assert!(should_compact(38_400, 64_000));
        assert!(should_compact(38_401, 64_000));
    }

    #[test]
    fn plan_compaction_returns_none_when_within_kept_rounds() {
        assert!(plan_compaction(&[1, 2, 3], KEEP_RECENT_ROUNDS).is_none());
    }

    #[test]
    fn plan_compaction_keeps_recent_and_compresses_the_rest() {
        let plan =
            plan_compaction(&[1, 2, 3, 4, 5], KEEP_RECENT_ROUNDS).expect("a plan is produced");
        // Keep 3, 4, 5; compress 1 and 2; boundary turn is 2.
        assert_eq!(plan.compacted_through_turn, 2);
        assert_eq!(plan.turns_to_compress, vec![1, 2]);
    }

    #[test]
    fn plan_compaction_compresses_expanded_history_before_kept_rounds() {
        let plan = plan_compaction(&[3, 4, 5, 6], KEEP_RECENT_ROUNDS).expect("a plan is produced");
        assert_eq!(plan.compacted_through_turn, 3);
        assert_eq!(plan.turns_to_compress, vec![3]);
    }

    #[test]
    fn plan_compaction_returns_none_without_extra_expanded_rounds() {
        assert!(plan_compaction(&[4, 5, 6], KEEP_RECENT_ROUNDS).is_none());
    }

    #[test]
    fn transcript_includes_previous_summary_and_selected_turns() {
        let history_messages = vec![
            record(1, ChatRole::User, "first question"),
            record(2, ChatRole::Assistant, "second answer"),
            record(3, ChatRole::User, "recent question"),
        ]
        .into_iter()
        .map(|record| AgentHistoryMessage {
            turn_sequence: record.turn_sequence,
            message: record_as_message(record),
        })
        .collect::<Vec<_>>();
        let compacted_context = vec![compacted_history_message("prior summary")];
        let transcript = build_history_transcript(&compacted_context, &history_messages, &[1, 2]);
        assert!(transcript.contains("prior summary"));
        assert!(transcript.contains("first question"));
        assert!(transcript.contains("second answer"));
        // Turn 3 is kept, not compressed, so it must be excluded.
        assert!(!transcript.contains("recent question"));
    }
}
