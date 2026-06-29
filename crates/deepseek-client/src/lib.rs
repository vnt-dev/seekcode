//! DeepSeek API adapter for the provider-neutral model interface.

pub mod client;
pub mod dto;
pub mod sse;
pub mod thinking;
pub mod tool_calls;

pub use client::{DeepSeekClient, DeepSeekConfig};
