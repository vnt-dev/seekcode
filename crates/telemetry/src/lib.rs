//! Tracing, metrics, and redacted logging setup.

use seekcode_common::SeekCodeResult;
use serde::{Deserialize, Serialize};
use tracing_subscriber::{fmt, EnvFilter};

/// Telemetry configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Env-filter compatible tracing directive.
    pub filter: String,
    /// Whether ANSI colors should be emitted.
    pub ansi: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            filter: "info,seekcode=debug".to_string(),
            ansi: true,
        }
    }
}

/// Initializes tracing for the desktop application.
pub fn init_tracing(config: &TelemetryConfig) -> SeekCodeResult<()> {
    let filter = EnvFilter::try_new(&config.filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let subscriber = fmt()
        .with_env_filter(filter)
        .with_ansi(config.ansi)
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}
