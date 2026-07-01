//! Agent runtime configuration.

use serde::{Deserialize, Serialize};

/// Agent runtime configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Default model used for coding tasks.
    pub default_model: String,
    /// Whether thinking mode is enabled by default.
    pub thinking: bool,
    /// Whether tool schemas should be strict.
    pub strict_tools: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            default_model: "deepseek-v4-pro".to_string(),
            thinking: true,
            strict_tools: true,
        }
    }
}
