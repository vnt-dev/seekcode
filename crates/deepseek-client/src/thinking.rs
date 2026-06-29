//! Thinking mode helpers for preserving DeepSeek reasoning content.

use seekcode_common::{ChatMessage, SeekCodeResult};

/// Applies DeepSeek thinking-mode message rules before a follow-up tool request.
pub fn preserve_reasoning_for_tool_round(
    _messages: &mut Vec<ChatMessage>,
    _assistant_reasoning: String,
) -> SeekCodeResult<()> {
    todo!("preserve DeepSeek reasoning_content across tool call rounds")
}
