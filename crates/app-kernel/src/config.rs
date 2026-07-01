use seekcode_agent_core::AgentConfig;
use seekcode_common::TelemetryConfig;
use seekcode_deepseek_client::DeepSeekConfig;
use seekcode_shell_sandbox::SandboxPolicy;
use serde::{Deserialize, Serialize};

/// App kernel configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppKernelConfig {
    /// DeepSeek provider configuration.
    #[serde(skip)]
    pub deepseek: DeepSeekConfig,
    /// Agent runtime configuration.
    pub agent: AgentConfig,
    /// Telemetry setup.
    pub telemetry: TelemetryConfig,
    /// Shell sandbox settings.
    pub shell: SandboxPolicy,
    /// Fast model used to generate empty session titles.
    pub title_model: String,
}

impl Default for AppKernelConfig {
    fn default() -> Self {
        Self {
            deepseek: DeepSeekConfig::default(),
            agent: AgentConfig::default(),
            telemetry: TelemetryConfig::default(),
            shell: SandboxPolicy::default(),
            title_model: "deepseek-v4-flash".to_string(),
        }
    }
}
