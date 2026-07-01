//! DeepSeek request and response DTOs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAI-compatible chat completion request for DeepSeek.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekChatRequest {
    /// DeepSeek model name.
    pub model: String,
    /// Provider-native message list.
    pub messages: Vec<DeepSeekMessage>,
    /// Whether the response should stream as SSE.
    pub stream: bool,
    /// Provider-native tool definitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// Streaming options used to request final usage frames.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<DeepSeekStreamOptions>,
    /// Thinking mode toggle.
    pub thinking: DeepSeekThinking,
    /// Provider-specific reasoning intensity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// DeepSeek thinking mode request option.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekThinking {
    /// Thinking mode state.
    #[serde(rename = "type")]
    pub kind: String,
}

/// DeepSeek chat message DTO.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekMessage {
    /// Provider role name.
    pub role: String,
    /// Message content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Thinking mode reasoning content that must be preserved across tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Assistant tool calls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<Value>,
    /// Tool call identifier for tool result messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// DeepSeek streaming options.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekStreamOptions {
    /// Whether DeepSeek should send an extra usage chunk before [DONE].
    pub include_usage: bool,
}

/// DeepSeek chat completion response placeholder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekChatResponse {
    /// Provider response identifier.
    pub id: String,
    /// Provider choices.
    pub choices: Vec<DeepSeekChoice>,
    /// Optional usage payload.
    pub usage: Option<DeepSeekUsage>,
}

/// Non-streaming response choice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekChoice {
    /// Assistant message payload.
    pub message: DeepSeekAssistantMessage,
    /// Provider finish reason.
    pub finish_reason: Option<String>,
}

/// Assistant message returned by DeepSeek.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekAssistantMessage {
    /// Assistant content.
    pub content: Option<String>,
    /// Thinking mode reasoning content.
    pub reasoning_content: Option<String>,
    /// Tool calls requested by the assistant.
    #[serde(default)]
    pub tool_calls: Vec<DeepSeekToolCall>,
}

/// DeepSeek tool call payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekToolCall {
    /// Provider tool call identifier.
    pub id: Option<String>,
    /// Tool call type, normally "function".
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Function call payload.
    pub function: DeepSeekFunctionCall,
}

/// DeepSeek function call payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekFunctionCall {
    /// Function name.
    pub name: Option<String>,
    /// JSON arguments encoded as a string.
    pub arguments: Option<String>,
}

/// DeepSeek token usage payload.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeepSeekUsage {
    /// Number of prompt tokens.
    pub prompt_tokens: Option<u32>,
    /// Number of completion tokens.
    pub completion_tokens: Option<u32>,
    /// Total number of tokens.
    pub total_tokens: Option<u32>,
    /// Number of prompt tokens served from prompt cache.
    pub prompt_cache_hit_tokens: Option<u32>,
}

/// Streaming chat completion chunk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekChatChunk {
    /// Provider choices.
    #[serde(default)]
    pub choices: Vec<DeepSeekChunkChoice>,
    /// Optional final usage chunk.
    pub usage: Option<DeepSeekUsage>,
}

/// One streaming choice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekChunkChoice {
    /// Delta payload.
    pub delta: DeepSeekDelta,
    /// Provider finish reason.
    pub finish_reason: Option<String>,
}

/// Streaming delta payload.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeepSeekDelta {
    /// Assistant content delta.
    pub content: Option<String>,
    /// Assistant reasoning delta.
    pub reasoning_content: Option<String>,
    /// Tool call deltas.
    #[serde(default)]
    pub tool_calls: Vec<DeepSeekToolCallDelta>,
}

/// Streaming tool call delta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeepSeekToolCallDelta {
    /// Tool call index in the streaming response.
    pub index: u32,
    /// Provider tool call identifier.
    pub id: Option<String>,
    /// Tool call type.
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Function call delta.
    pub function: Option<DeepSeekFunctionCall>,
}
