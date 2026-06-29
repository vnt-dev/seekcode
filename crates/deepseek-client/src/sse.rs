//! SSE parsing boundary for DeepSeek streaming responses.

use crate::dto::{DeepSeekChatChunk, DeepSeekFunctionCall};
use crate::tool_calls::decode_usage;
use seekcode_common::{SeekCodeError, SeekCodeResult, ToolCallId};
use seekcode_model_provider::{ChatChunk, ToolCall};
use serde_json::Value;
use std::collections::BTreeMap;

/// Parses one DeepSeek SSE data frame into provider-neutral chunks.
pub fn parse_sse_frame(frame: &str) -> SeekCodeResult<Option<ChatChunk>> {
    Ok(
        parse_sse_frame_with_accumulator(frame, &mut ToolCallAccumulator::default())?
            .into_iter()
            .next(),
    )
}

/// Parses one DeepSeek SSE data frame and updates tool-call accumulation state.
pub fn parse_sse_frame_with_accumulator(
    frame: &str,
    accumulator: &mut ToolCallAccumulator,
) -> SeekCodeResult<Vec<ChatChunk>> {
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
        if let Some(content) = choice.delta.content {
            if !content.is_empty() {
                chunks.push(ChatChunk::Content(content));
            }
        }

        if let Some(reasoning) = choice.delta.reasoning_content {
            if !reasoning.is_empty() {
                chunks.push(ChatChunk::Reasoning(reasoning));
            }
        }

        for tool_delta in choice.delta.tool_calls {
            let (name, arguments) = if let Some(function) = tool_delta.function {
                (function.name, function.arguments)
            } else {
                (None, None)
            };
            accumulator.push_delta(tool_delta.index, tool_delta.id, name, arguments);
        }

        if choice.finish_reason.as_deref() == Some("tool_calls") {
            for call in accumulator.take_completed()? {
                chunks.push(ChatChunk::ToolCall(call));
            }
        }
    }

    Ok(chunks)
}

/// Accumulates streaming tool call deltas until DeepSeek marks tool calls complete.
#[derive(Default)]
pub struct ToolCallAccumulator {
    partials: BTreeMap<u32, PartialToolCall>,
}

impl ToolCallAccumulator {
    /// Appends one tool call delta.
    pub fn push_delta(
        &mut self,
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) {
        let partial = self.partials.entry(index).or_default();
        if let Some(id) = id {
            partial.id = Some(id);
        }
        if let Some(name) = name {
            partial.name = Some(name);
        }
        if let Some(arguments) = arguments {
            partial.arguments.push_str(&arguments);
        }
    }

    /// Drains completed tool calls in index order.
    pub fn take_completed(&mut self) -> SeekCodeResult<Vec<ToolCall>> {
        let partials = std::mem::take(&mut self.partials);
        partials
            .into_values()
            .map(|partial| {
                let name = partial.name.ok_or_else(|| {
                    SeekCodeError::ModelProvider("missing streamed tool call name".to_string())
                })?;
                let arguments = if partial.arguments.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&partial.arguments).map_err(|error| {
                        SeekCodeError::ModelProvider(format!(
                            "invalid streamed tool arguments: {error}"
                        ))
                    })?
                };
                let _provider_id = partial.id;

                Ok(ToolCall {
                    id: ToolCallId::new(),
                    name,
                    arguments,
                })
            })
            .collect()
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[allow(dead_code)]
fn _typed_function_call(_call: DeepSeekFunctionCall) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_chunk() {
        let chunks = parse_sse_frame_with_accumulator(
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
            &mut ToolCallAccumulator::default(),
        )
        .expect("parse chunk");

        assert!(matches!(&chunks[0], ChatChunk::Content(text) if text == "hello"));
    }

    #[test]
    fn accumulates_streaming_tool_call_arguments() {
        let mut accumulator = ToolCallAccumulator::default();
        parse_sse_frame_with_accumulator(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
            &mut accumulator,
        )
        .expect("parse first tool chunk");
        let chunks = parse_sse_frame_with_accumulator(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"src/lib.rs\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            &mut accumulator,
        )
        .expect("parse final tool chunk");

        assert!(matches!(
            &chunks[0],
            ChatChunk::ToolCall(call)
                if call.name == "read_file" && call.arguments["path"] == "src/lib.rs"
        ));
    }
}
