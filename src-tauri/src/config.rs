//! User configuration persisted under ~/.seekcode.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const CONFIG_DIR_NAME: &str = ".seekcode";
const CONFIG_FILE_NAME: &str = "config.toml";
const DATABASE_FILE_NAME: &str = "seekcode.sqlite";
pub const DEFAULT_PROVIDER_ID: &str = "default";

/// User-editable application settings persisted to disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    /// DeepSeek-compatible API base URL.
    pub base_url: String,
    /// DeepSeek API key.
    pub api_key: String,
    /// Fast model used to generate empty chat titles.
    pub title_model: String,
    /// Models shown in the chat model selector.
    pub models: Vec<ModelSetting>,
    /// Additional DeepSeek/OpenAI-compatible model providers.
    pub providers: Vec<ModelProviderSetting>,
}

/// User-editable model provider entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelProviderSetting {
    /// Stable provider identifier persisted on sessions.
    pub id: String,
    /// Human-readable provider name shown in the UI.
    pub name: String,
    /// API base URL.
    pub base_url: String,
    /// API key.
    pub api_key: String,
    /// Models available from this provider.
    pub models: Vec<ModelSetting>,
}

/// User-editable model entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSetting {
    /// Model identifier sent to the provider.
    pub id: String,
    /// Human-readable model label shown in the UI.
    pub label: String,
}

impl Default for ModelProviderSetting {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            models: Vec::new(),
        }
    }
}

impl Default for ModelSetting {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
        }
    }
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            base_url: "https://api.deepseek.com".to_string(),
            api_key: String::new(),
            title_model: "deepseek-v4-flash".to_string(),
            models: default_models(),
            providers: Vec::new(),
        }
    }
}

/// Returns the path to ~/.seekcode/config.toml.
pub fn config_path() -> anyhow::Result<PathBuf> {
    Ok(config_dir()?.join(CONFIG_FILE_NAME))
}

/// Returns the path to ~/.seekcode/seekcode.sqlite.
pub fn database_path() -> anyhow::Result<PathBuf> {
    Ok(config_dir()?.join(DATABASE_FILE_NAME))
}

fn config_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("failed to resolve home directory"))?;

    Ok(home.join(CONFIG_DIR_NAME))
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
    settings.title_model = settings.title_model.trim().to_string();
    if settings.title_model.is_empty() {
        settings.title_model = AppSettings::default().title_model;
    }

    let mut models = Vec::new();
    for model in std::mem::take(&mut settings.models) {
        let id = model.id.trim().to_string();
        if id.is_empty() || models.iter().any(|item: &ModelSetting| item.id == id) {
            continue;
        }
        let label = model.label.trim().to_string();
        models.push(ModelSetting {
            label: if label.is_empty() { id.clone() } else { label },
            id,
        });
    }
    if models.is_empty() {
        models = default_models();
    }
    settings.models = models;

    let mut provider_ids = vec![DEFAULT_PROVIDER_ID.to_string()];
    let mut providers = Vec::new();
    for provider in std::mem::take(&mut settings.providers) {
        let id = provider.id.trim().to_string();
        if id.is_empty() || provider_ids.iter().any(|item| item == &id) {
            continue;
        }
        let name = provider.name.trim().to_string();
        let base_url = provider.base_url.trim().trim_end_matches('/').to_string();
        let api_key = provider.api_key.trim().to_string();
        let models = normalize_models(provider.models, Vec::new());
        provider_ids.push(id.clone());
        providers.push(ModelProviderSetting {
            name: if name.is_empty() { id.clone() } else { name },
            id,
            base_url,
            api_key,
            models,
        });
    }
    settings.providers = providers;
}

fn validate_settings(settings: &AppSettings) -> anyhow::Result<()> {
    let base_url = settings.base_url.trim();
    if base_url.is_empty() {
        anyhow::bail!("base_url cannot be empty");
    }
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        anyhow::bail!("base_url must start with http:// or https://");
    }
    if settings.title_model.trim().is_empty() {
        anyhow::bail!("title_model cannot be empty");
    }
    if settings.models.is_empty() {
        anyhow::bail!("models cannot be empty");
    }
    if settings
        .models
        .iter()
        .any(|model| model.id.trim().is_empty())
    {
        anyhow::bail!("model id cannot be empty");
    }
    for provider in &settings.providers {
        if provider.id.trim().is_empty() {
            anyhow::bail!("provider id cannot be empty");
        }
        let base_url = provider.base_url.trim();
        if base_url.is_empty() {
            anyhow::bail!("provider base_url cannot be empty");
        }
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            anyhow::bail!("provider base_url must start with http:// or https://");
        }
        if provider.models.is_empty() {
            anyhow::bail!("provider models cannot be empty");
        }
        if provider
            .models
            .iter()
            .any(|model| model.id.trim().is_empty())
        {
            anyhow::bail!("provider model id cannot be empty");
        }
    }

    Ok(())
}

pub fn provider_connection(settings: &AppSettings, provider_id: &str) -> Option<(String, String)> {
    let provider_id = provider_id.trim();
    if provider_id.is_empty() || provider_id == DEFAULT_PROVIDER_ID {
        return Some((settings.base_url.clone(), settings.api_key.clone()));
    }

    settings
        .providers
        .iter()
        .find(|provider| provider.id == provider_id)
        .map(|provider| (provider.base_url.clone(), provider.api_key.clone()))
}

fn default_models() -> Vec<ModelSetting> {
    vec![
        ModelSetting {
            id: "deepseek-v4-pro".to_string(),
            label: "DeepSeek V4 Pro".to_string(),
        },
        ModelSetting {
            id: "deepseek-v4-flash".to_string(),
            label: "DeepSeek V4 Flash".to_string(),
        },
    ]
}

fn normalize_models(models: Vec<ModelSetting>, fallback: Vec<ModelSetting>) -> Vec<ModelSetting> {
    let mut seen = Vec::new();
    let mut normalized = Vec::new();
    for model in models {
        let id = model.id.trim().to_string();
        if id.is_empty() || seen.iter().any(|item| item == &id) {
            continue;
        }
        let label = model.label.trim().to_string();
        seen.push(id.clone());
        normalized.push(ModelSetting {
            label: if label.is_empty() { id.clone() } else { label },
            id,
        });
    }
    if normalized.is_empty() {
        fallback
    } else {
        normalized
    }
}
