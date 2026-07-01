//! DeepSeek API adapter for the provider-neutral model interface.

pub mod client;
pub mod dto;
pub mod provider;
pub mod sse;
pub mod tool_calls;

pub use client::{DeepSeekClient, DeepSeekConfig, DEFAULT_CONTEXT_WINDOW};
pub use provider::{
    ChatChoiceChunk, ChatChunk, ChatDelta, ChatRequest, ChatResponse, ChatStream, ModelProfile,
    ModelProvider, ToolCall, ToolCallDelta, ToolSpec,
};
