//! Tool call conversion helpers for DeepSeek payloads.

use crate::dto::{DeepSeekFunctionCall, DeepSeekToolCall, DeepSeekUsage};
use crate::{ToolCall, ToolSpec};
use seekcode_common::{SeekCodeError, SeekCodeResult, TokenUsage};
use serde_json::{json, Value};

/// Converts provider-neutral tool specs into DeepSeek tool definitions.
pub fn encode_tool_specs(tools: &[ToolSpec]) -> SeekCodeResult<Vec<Value>> {
    Ok(tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": tool.strict
                }
            })
        })
        .collect())
}

/// Converts a DeepSeek tool call payload into a provider-neutral call.
pub fn decode_tool_call(value: &Value) -> SeekCodeResult<ToolCall> {
    let call: DeepSeekToolCall = serde_json::from_value(value.clone())
        .map_err(|error| SeekCodeError::ModelProvider(error.to_string()))?;

    decode_deepseek_tool_call(&call)
}

/// Converts a typed DeepSeek tool call into a provider-neutral call.
pub fn decode_deepseek_tool_call(call: &DeepSeekToolCall) -> SeekCodeResult<ToolCall> {
    decode_function_call(call.id.as_deref(), &call.function)
}

/// Converts a DeepSeek function call into a provider-neutral call.
pub fn decode_function_call(
    id: Option<&str>,
    function: &DeepSeekFunctionCall,
) -> SeekCodeResult<ToolCall> {
    let name = function.name.clone().ok_or_else(|| {
        SeekCodeError::ModelProvider("missing tool call function name".to_string())
    })?;
    let id = id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| SeekCodeError::ModelProvider("missing tool call id".to_string()))?;
    let arguments = function.arguments.as_deref().unwrap_or("{}");
    let arguments = serde_json::from_str(arguments).map_err(|error| {
        SeekCodeError::ModelProvider(format!("invalid tool arguments: {error}"))
    })?;

    Ok(ToolCall {
        id: id.to_string(),
        name,
        arguments,
    })
}

/// Converts DeepSeek token usage into provider-neutral usage.
pub fn decode_usage(usage: DeepSeekUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: usage.prompt_tokens.unwrap_or_default(),
        completion_tokens: usage.completion_tokens.unwrap_or_default(),
        total_tokens: usage.total_tokens.unwrap_or_default(),
        cached_tokens: usage.prompt_cache_hit_tokens.unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_specs_encode_as_function_tools() {
        let specs = encode_tool_specs(&[ToolSpec {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: json!({"type": "object"}),
            strict: true,
        }])
        .expect("encode tools");

        assert_eq!(specs[0]["type"], "function");
        assert_eq!(specs[0]["function"]["name"], "read_file");
        assert_eq!(specs[0]["function"]["strict"], true);
    }

    #[test]
    fn function_call_arguments_decode_from_json_string() {
        let call = decode_function_call(
            Some("call_1"),
            &DeepSeekFunctionCall {
                name: Some("read_file".to_string()),
                arguments: Some(r#"{"path":"src/lib.rs"}"#.to_string()),
            },
        )
        .expect("decode tool call");

        assert_eq!(call.name, "read_file");
        assert_eq!(call.arguments["path"], "src/lib.rs");
    }
}
