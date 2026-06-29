//! Application-level service composition for the Tauri adapter and future CLI.

use parking_lot::RwLock;
use seekcode_agent_core::{Agent, AgentConfig, AgentEvent, AgentTask, StartTaskRequest};
use seekcode_common::{SeekCodeResult, TaskId};
use seekcode_deepseek_client::{DeepSeekClient, DeepSeekConfig};
use seekcode_model_provider::ModelProvider;
use seekcode_policy::{AutonomousPolicy, PolicyEngine};
use seekcode_secrets::{InMemorySecretStore, SecretStore};
use seekcode_shell_sandbox::{CommandRunner, SandboxPolicy};
use seekcode_storage::{SessionRecord, Storage};
use seekcode_telemetry::{init_tracing, TelemetryConfig};
use seekcode_tool_system::ToolRegistry;
use seekcode_workspace::{FileEntry, FileSnapshot, ListOptions, WorkspaceRoot, WorkspaceService};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;

pub use seekcode_storage::SessionRecord as AppSessionRecord;

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
}

impl Default for AppKernelConfig {
    fn default() -> Self {
        Self {
            deepseek: DeepSeekConfig::default(),
            agent: AgentConfig::default(),
            telemetry: TelemetryConfig::default(),
            shell: SandboxPolicy::default(),
        }
    }
}

/// Concrete services assembled for the application.
pub struct AppServices {
    /// Model provider used by the agent.
    pub provider: Arc<RwLock<Arc<dyn ModelProvider>>>,
    /// Agent runtime.
    pub agent: Arc<Agent>,
    /// Tool registry.
    pub tools: Arc<ToolRegistry>,
    /// Workspace service.
    pub workspace: Arc<WorkspaceService>,
    /// Policy engine.
    pub policy: Arc<dyn PolicyEngine>,
    /// Optional durable storage.
    pub storage: Option<Arc<dyn Storage>>,
    /// Secret storage.
    pub secrets: Arc<dyn SecretStore>,
    /// Command runner.
    pub shell: Arc<CommandRunner>,
}

/// Application kernel exposed to thin adapters.
pub struct AppKernel {
    config: RwLock<AppKernelConfig>,
    services: AppServices,
}

impl AppKernel {
    /// Builds the application service graph.
    pub fn new(config: AppKernelConfig) -> anyhow::Result<Self> {
        init_tracing(&config.telemetry)?;

        let provider: Arc<dyn ModelProvider> =
            Arc::new(DeepSeekClient::new(config.deepseek.clone())?);
        let provider_slot = Arc::new(RwLock::new(provider.clone()));
        let tools = Arc::new(ToolRegistry::new());
        let agent = Arc::new(Agent::new(
            config.agent.clone(),
            provider.clone(),
            tools.clone(),
        ));
        let workspace = Arc::new(WorkspaceService::new());
        let policy: Arc<dyn PolicyEngine> = Arc::new(AutonomousPolicy);
        let secrets: Arc<dyn SecretStore> = Arc::new(InMemorySecretStore::new());
        let shell = Arc::new(CommandRunner::new(config.shell.clone()));

        Ok(Self {
            config: RwLock::new(config),
            services: AppServices {
                provider: provider_slot,
                agent,
                tools,
                workspace,
                policy,
                storage: None,
                secrets,
                shell,
            },
        })
    }

    /// Returns the kernel configuration.
    pub fn config(&self) -> AppKernelConfig {
        self.config.read().clone()
    }

    /// Returns assembled services.
    pub fn services(&self) -> &AppServices {
        &self.services
    }

    /// Updates DeepSeek provider configuration for newly-started tasks.
    pub async fn update_deepseek_config(&self, deepseek: DeepSeekConfig) -> anyhow::Result<()> {
        let provider: Arc<dyn ModelProvider> = Arc::new(DeepSeekClient::new(deepseek.clone())?);
        self.services.agent.set_provider(provider.clone());
        *self.services.provider.write() = provider;

        self.config.write().deepseek = deepseek;

        Ok(())
    }

    /// Opens a workspace.
    pub async fn open_workspace(&self, path: PathBuf) -> SeekCodeResult<WorkspaceRoot> {
        self.services.workspace.open(path).await
    }

    /// Starts an agent task.
    pub async fn start_agent_task(&self, request: StartTaskRequest) -> SeekCodeResult<AgentTask> {
        self.services.agent.start_task(request).await
    }

    /// Subscribes to agent runtime events.
    pub fn subscribe_agent_events(&self) -> broadcast::Receiver<AgentEvent> {
        self.services.agent.subscribe()
    }

    /// Cancels an agent task.
    pub async fn cancel_agent_task(&self, task_id: TaskId) -> SeekCodeResult<()> {
        self.services.agent.cancel_task(task_id).await
    }

    /// Reads one workspace file.
    pub async fn read_file(
        &self,
        root: WorkspaceRoot,
        path: PathBuf,
    ) -> SeekCodeResult<FileSnapshot> {
        self.services.workspace.read_file(&root, path).await
    }

    /// Lists workspace files.
    pub async fn list_files(
        &self,
        root: WorkspaceRoot,
        options: ListOptions,
    ) -> SeekCodeResult<Vec<FileEntry>> {
        self.services.workspace.list_tree(&root, options).await
    }

    /// Lists persisted sessions.
    pub async fn get_sessions(&self) -> SeekCodeResult<Vec<SessionRecord>> {
        let storage = self.services.storage.as_ref().ok_or(
            seekcode_common::SeekCodeError::NotImplemented("storage is not wired yet"),
        )?;

        storage.list_sessions().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_kernel_can_be_constructed() {
        let kernel = AppKernel::new(AppKernelConfig::default()).expect("kernel builds");

        assert_eq!(kernel.config().agent.default_model, "deepseek-v4-pro");
    }

    #[tokio::test]
    async fn app_kernel_updates_deepseek_config() {
        let kernel = AppKernel::new(AppKernelConfig::default()).expect("kernel builds");
        let mut deepseek = DeepSeekConfig::default();
        deepseek.base_url = "https://example.test".to_string();
        deepseek.api_key = Some("sk-test".to_string());

        kernel
            .update_deepseek_config(deepseek)
            .await
            .expect("deepseek config updates");

        let config = kernel.config();
        assert_eq!(config.deepseek.base_url, "https://example.test");
        assert_eq!(config.deepseek.api_key.as_deref(), Some("sk-test"));
    }
}
