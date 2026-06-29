//! User configuration persisted under ~/.seekcode.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const CONFIG_DIR_NAME: &str = ".seekcode";
const CONFIG_FILE_NAME: &str = "config.toml";

/// User-editable application settings persisted to disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// DeepSeek-compatible API base URL.
    pub base_url: String,
    /// DeepSeek API key.
    pub api_key: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            base_url: "https://api.deepseek.com".to_string(),
            api_key: String::new(),
        }
    }
}

/// Returns the path to ~/.seekcode/config.toml.
pub fn config_path() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("failed to resolve home directory"))?;

    Ok(home.join(CONFIG_DIR_NAME).join(CONFIG_FILE_NAME))
}

/// Loads application settings from ~/.seekcode/config.toml.
pub fn load_app_settings_sync() -> anyhow::Result<AppSettings> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(AppSettings::default());
    }

    let content = std::fs::read_to_string(&path)?;
    let mut settings: AppSettings = toml::from_str(&content)?;
    normalize_settings(&mut settings);
    Ok(settings)
}

/// Loads application settings from ~/.seekcode/config.toml asynchronously.
pub async fn load_app_settings() -> anyhow::Result<AppSettings> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(AppSettings::default());
    }

    let content = tokio::fs::read_to_string(&path).await?;
    let mut settings: AppSettings = toml::from_str(&content)?;
    normalize_settings(&mut settings);
    Ok(settings)
}

/// Saves application settings to ~/.seekcode/config.toml.
pub async fn save_app_settings(mut settings: AppSettings) -> anyhow::Result<AppSettings> {
    normalize_settings(&mut settings);
    validate_settings(&settings)?;

    let path = config_path()?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let content = toml::to_string_pretty(&settings)?;
    tokio::fs::write(&path, content).await?;
    Ok(settings)
}

fn normalize_settings(settings: &mut AppSettings) {
    settings.base_url = settings.base_url.trim().to_string();
    settings.api_key = settings.api_key.trim().to_string();
}

fn validate_settings(settings: &AppSettings) -> anyhow::Result<()> {
    let base_url = settings.base_url.trim();
    if base_url.is_empty() {
        anyhow::bail!("base_url cannot be empty");
    }
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        anyhow::bail!("base_url must start with http:// or https://");
    }

    Ok(())
}
