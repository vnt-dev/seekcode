//! SSE parsing boundary for DeepSeek streaming responses.

use crate::dto::{DeepSeekChatChunk, DeepSeekFunctionCall};
use crate::tool_calls::decode_usage;
use crate::{ChatChoiceChunk, ChatChunk, ChatDelta, ToolCallDelta};
use seekcode_common::{SeekCodeError, SeekCodeResult};

/// Parses one DeepSeek SSE data frame into provider-neutral choice chunks.
pub fn parse_sse_frame_choices(frame: &str) -> SeekCodeResult<Vec<ChatChunk>> {
    // tracing::debug!("parse_sse_frame_choices {:?}", frame);
    let data = frame.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(Vec::new());
    }

    let chunk: DeepSeekChatChunk = serde_json::from_str(data)
        .map_err(|error| SeekCodeError::ModelProvider(error.to_string()))?;
    let mut chunks = Vec::new();

    if let Some(usage) = chunk.usage {
        chunks.push(ChatChunk::Usage(decode_usage(usage)));
    }

    for choice in chunk.choices {
        chunks.push(ChatChunk::Choice(ChatChoiceChunk {
            delta: ChatDelta {
                content: choice.delta.content,
                reasoning_content: choice.delta.reasoning_content,
                tool_calls: choice
                    .delta
                    .tool_calls
                    .into_iter()
                    .map(|tool_delta| {
                        let (name, arguments) = tool_delta
                            .function
                            .map(|function| (function.name, function.arguments))
                            .unwrap_or((None, None));
                        ToolCallDelta {
                            index: tool_delta.index,
                            id: None,
                            kind: tool_delta.kind,
                            name,
                            arguments,
                        }
                    })
                    .collect(),
            },
            finish_reason: choice.finish_reason,
        }));
    }

    Ok(chunks)
}

#[allow(dead_code)]
fn _typed_function_call(_call: DeepSeekFunctionCall) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_chunk() {
        let chunks = parse_sse_frame_choices(
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
        )
        .expect("parse chunk");

        assert!(matches!(
            &chunks[0],
            ChatChunk::Choice(choice) if choice.delta.content.as_deref() == Some("hello")
        ));
    }

    #[test]
    fn parses_streaming_tool_call_delta() {
        let chunks = parse_sse_frame_choices(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/lib.rs\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        )
        .expect("parse final tool chunk");

        assert!(matches!(
            &chunks[0],
            ChatChunk::Choice(choice)
                if choice.delta.tool_calls[0].name.as_deref() == Some("read_file")
                    && choice.finish_reason.as_deref() == Some("tool_calls")
        ));
    }
}
