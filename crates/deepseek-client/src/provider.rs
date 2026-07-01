//! Provider-neutral model API used by the agent runtime.

use async_trait::async_trait;
use futures_util::stream::BoxStream;
use seekcode_common::{ChatMessage, SeekCodeResult, TokenUsage, ToolCallId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Streaming response type returned by model providers.
pub type ChatStream = BoxStream<'static, SeekCodeResult<ChatChunk>>;

/// Request sent to a chat model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Model identifier understood by the provider.
    pub model: String,
    /// Messages included in the conversation.
    pub messages: Vec<ChatMessage>,
    /// Tools exposed to the provider.
    pub tools: Vec<ToolSpec>,
    /// Whether provider reasoning output should be requested.
    pub thinking: bool,
    /// Optional provider-specific reasoning intensity.
    pub reasoning_effort: Option<String>,
    /// Whether strict JSON schema validation is requested for tools.
    pub strict_tools: bool,
}

/// Streaming chunk emitted by a model provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum ChatChunk {
    /// One provider choice chunk containing one complete streaming delta.
    Choice(ChatChoiceChunk),
    /// Assistant content delta.
    Content(String),
    /// Assistant reasoning delta.
    Reasoning(String),
    /// Final usage summary.
    Usage(TokenUsage),
    /// Provider-specific completion marker.
    Finished,
}

/// One streaming choice emitted by a chat provider.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatChoiceChunk {
    /// Delta payload.
    pub delta: ChatDelta,
    /// Provider finish reason for this choice, if any.
    pub finish_reason: Option<String>,
}

/// One streaming delta emitted by a chat provider.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Assistant content delta.
    pub content: Option<String>,
    /// Assistant reasoning delta.
    pub reasoning_content: Option<String>,
    /// Tool call deltas.
    pub tool_calls: Vec<ToolCallDelta>,
}

/// One streaming tool-call delta.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Tool call index in the assistant message.
    pub index: u32,
    /// Provider or local tool call identifier.
    pub id: Option<ToolCallId>,
    /// Tool call kind, normally "function".
    pub kind: Option<String>,
    /// Function name delta.
    pub name: Option<String>,
    /// JSON arguments string delta.
    pub arguments: Option<String>,
}

/// Complete non-streaming model response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Assistant content.
    pub content: String,
    /// Optional provider reasoning content.
    pub reasoning_content: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage, if returned by the provider.
    pub usage: Option<TokenUsage>,
}

/// Tool definition exposed to a model provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Stable tool name.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// JSON schema describing tool arguments.
    pub input_schema: Value,
    /// Whether the provider should enforce strict tool arguments.
    pub strict: bool,
}

/// Tool call requested by a model provider.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCall {
    /// Backend tool call identifier.
    pub id: ToolCallId,
    /// Name of the requested tool.
    pub name: String,
    /// Raw JSON arguments.
    pub arguments: Value,
}

/// Capability metadata for a model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Model identifier.
    pub id: String,
    /// Maximum context window in tokens.
    pub context_window: u32,
    /// Whether the model can request tools.
    pub supports_tools: bool,
    /// Whether the model can produce reasoning output.
    pub supports_thinking: bool,
}

/// Provider-neutral chat model interface.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    /// Streams a chat completion.
    fn stream_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatStream>;

    /// Runs a non-streaming chat completion.
    async fn complete_chat(&self, request: ChatRequest) -> SeekCodeResult<ChatResponse>;

    /// Returns capability metadata for a model.
    async fn model_profile(&self, model: &str) -> SeekCodeResult<ModelProfile>;
}
