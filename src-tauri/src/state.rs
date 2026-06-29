//! Tauri application state wrapper.

use crate::config::load_app_settings_sync;
use seekcode_app_kernel::{AppKernel, AppKernelConfig};
use std::sync::Arc;

/// Shared Tauri state for backend commands.
pub struct AppState {
    /// Application service kernel.
    pub kernel: Arc<AppKernel>,
}

impl AppState {
    /// Initializes application state.
    pub fn new() -> anyhow::Result<Self> {
        let settings = load_app_settings_sync()?;
        let mut config = AppKernelConfig::default();
        config.deepseek.base_url = settings.base_url;
        config.deepseek.api_key = (!settings.api_key.is_empty()).then_some(settings.api_key);

        Ok(Self {
            kernel: Arc::new(AppKernel::new(config)?),
        })
    }
}
