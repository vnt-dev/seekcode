//! Thin Tauri command adapter around AppKernel.

use crate::config::{self, provider_connection, AppSettings, ModelSetting};
use crate::state::AppState;
use seekcode_agent_core::{AgentTask, StartTaskRequest};
use seekcode_app_kernel::{
    CreateSessionRequest, OpenWorkspaceRequest, WorkspaceWithSessions, DEFAULT_CONTEXT_WINDOW,
};
use seekcode_common::{SessionId, TaskId, WorkspaceId};
use seekcode_storage::{SessionMessageRecord, SessionModelCallStats, SessionRecord};
use tauri::{AppHandle, Emitter, State};

/// Starts an agent task.
#[tauri::command]
pub async fn start_agent_task(
    app: AppHandle,
    state: State<'_, AppState>,
    mut request: StartTaskRequest,
) -> Result<AgentTask, String> {
    let session_id = request.session_id;
    let prompt = request.prompt.clone();
    let session = state
        .kernel
        .get_sessions()
        .await
        .map_err(|error| {
            tracing::warn!(
                target: "seekcode_tauri::commands",
                %session_id,
                %error,
                "failed to load sessions before starting agent task"
            );
            error.to_string()
        })?
        .into_iter()
        .find(|session| session.id == session_id)
        .ok_or_else(|| {
            tracing::warn!(
                target: "seekcode_tauri::commands",
                %session_id,
                "session was not found before starting agent task"
            );
            format!("session {session_id} not found")
        })?;
    let settings = config::load_app_settings().await.map_err(|error| {
        tracing::warn!(
            target: "seekcode_tauri::commands",
            %session_id,
            %error,
            "failed to load app settings before starting agent task"
        );
        error.to_string()
    })?;
    let (base_url, api_key) =
        provider_connection(&settings, &session.model_provider).ok_or_else(|| {
            tracing::warn!(
                target: "seekcode_tauri::commands",
                %session_id,
                model_provider = %session.model_provider,
                "model provider is not configured before starting agent task"
            );
            format!(
                "model provider {} is not configured",
                session.model_provider
            )
        })?;
    let mut deepseek = state.kernel.config().deepseek;
    deepseek.base_url = base_url;
    deepseek.api_key = (!api_key.is_empty()).then_some(api_key);
    state
        .kernel
        .update_deepseek_config(deepseek)
        .await
        .map_err(|error| {
            tracing::warn!(
                target: "seekcode_tauri::commands",
                %session_id,
                model_provider = %session.model_provider,
                %error,
                "failed to update model provider config before starting agent task"
            );
            error.to_string()
        })?;
    request.thinking = Some(session.thinking_enabled);
    request.reasoning_effort = normalize_reasoning_effort(session.reasoning_effort.clone());
    let started = state
        .kernel
        .start_agent_task(request)
        .await
        .map_err(|error| {
            tracing::error!(
                target: "seekcode_tauri::commands",
                %session_id,
                model_provider = %session.model_provider,
                %error,
                "failed to start agent task"
            );
            error.to_string()
        })?;
    let task = started.task;
    let mut events = started.events;
    let task_id = task.id;
    let (title_sender, mut title_events) = tokio::sync::mpsc::unbounded_channel();
    state
        .kernel
        .spawn_session_title_generation(session_id, prompt, title_sender);

    let agent_app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            match events.recv().await {
                Some(event) if event.task_id() == task_id && event.session_id() == session_id => {
                    let terminal = event.is_terminal();
                    if let Err(error) = agent_app.emit("agent:event", event) {
                        tracing::warn!(
                            target: "seekcode_tauri::commands",
                            %session_id,
                            %task_id,
                            %error,
                            "failed to emit agent event to frontend"
                        );
                    }
                    if terminal {
                        break;
                    }
                }
                Some(_) => {}
                None => {
                    tracing::warn!(
                        target: "seekcode_tauri::commands",
                        %session_id,
                        %task_id,
                        "agent event stream closed before a terminal event was emitted"
                    );
                    break;
                }
            }
        }
    });

    tauri::async_runtime::spawn(async move {
        while let Some(event) = title_events.recv().await {
            if let Err(error) = app.emit("session:title_changed", event) {
                tracing::warn!(
                    target: "seekcode_tauri::commands",
                    %session_id,
                    %error,
                    "failed to emit session title change to frontend"
                );
            }
        }
    });

    Ok(task)
}

/// Cancels an agent task.
#[tauri::command]
pub async fn cancel_agent_task(state: State<'_, AppState>, task_id: TaskId) -> Result<(), String> {
    state
        .kernel
        .cancel_agent_task(task_id)
        .await
        .map_err(|error| error.to_string())
}

/// Lists persisted sessions.
#[tauri::command]
pub async fn get_sessions(state: State<'_, AppState>) -> Result<Vec<SessionRecord>, String> {
    state
        .kernel
        .get_sessions()
        .await
        .map_err(|error| error.to_string())
}

/// Opens an existing workspace by path or creates a visible one.
#[tauri::command]
pub async fn open_workspace(
    state: State<'_, AppState>,
    request: OpenWorkspaceRequest,
) -> Result<WorkspaceWithSessions, String> {
    state
        .kernel
        .open_workspace(request)
        .await
        .map_err(|error| error.to_string())
}

/// Lists visible workspaces and their sessions.
#[tauri::command]
pub async fn list_visible_workspaces(
    state: State<'_, AppState>,
) -> Result<Vec<WorkspaceWithSessions>, String> {
    state
        .kernel
        .list_visible_workspaces()
        .await
        .map_err(|error| error.to_string())
}

/// Hides one workspace from the sidebar.
#[tauri::command]
pub async fn hide_workspace(
    state: State<'_, AppState>,
    workspace_id: WorkspaceId,
) -> Result<(), String> {
    state
        .kernel
        .hide_workspace(workspace_id)
        .await
        .map_err(|error| error.to_string())
}

/// Creates a persisted session.
#[tauri::command]
pub async fn create_session(
    state: State<'_, AppState>,
    request: CreateSessionRequest,
) -> Result<SessionRecord, String> {
    state
        .kernel
        .create_session(request)
        .await
        .map_err(|error| error.to_string())
}

/// Deletes one session and cascades its messages.
#[tauri::command]
pub async fn delete_session(
    state: State<'_, AppState>,
    session_id: SessionId,
) -> Result<(), String> {
    state
        .kernel
        .delete_session(session_id)
        .await
        .map_err(|error| error.to_string())
}

/// Updates the model selected for one session.
#[tauri::command]
pub async fn update_session_model(
    state: State<'_, AppState>,
    session_id: SessionId,
    model_provider: String,
    model: String,
    thinking_enabled: bool,
    reasoning_effort: Option<String>,
) -> Result<SessionRecord, String> {
    state
        .kernel
        .update_session_model(
            session_id,
            model_provider,
            model,
            thinking_enabled,
            reasoning_effort,
        )
        .await
        .map_err(|error| error.to_string())
}

/// Deletes all sessions under one workspace.
#[tauri::command]
pub async fn delete_workspace_sessions(
    state: State<'_, AppState>,
    workspace_id: WorkspaceId,
) -> Result<(), String> {
    state
        .kernel
        .delete_workspace_sessions(workspace_id)
        .await
        .map_err(|error| error.to_string())
}

/// Lists persisted messages for one session.
#[tauri::command]
pub async fn list_session_messages(
    state: State<'_, AppState>,
    session_id: SessionId,
    before_turn_sequence: Option<i64>,
    turn_limit: Option<i64>,
) -> Result<Vec<SessionMessageRecord>, String> {
    state
        .kernel
        .list_session_messages(session_id, before_turn_sequence, turn_limit)
        .await
        .map_err(|error| error.to_string())
}

/// Returns the most recent input token count recorded for one session.
#[tauri::command]
pub async fn session_context_usage(
    state: State<'_, AppState>,
    session_id: SessionId,
) -> Result<i64, String> {
    state
        .kernel
        .session_context_usage(session_id)
        .await
        .map_err(|error| error.to_string())
}

/// Returns aggregated model call telemetry for one session.
#[tauri::command]
pub async fn session_model_call_stats(
    state: State<'_, AppState>,
    session_id: SessionId,
) -> Result<SessionModelCallStats, String> {
    state
        .kernel
        .session_model_call_stats(session_id)
        .await
        .map_err(|error| error.to_string())
}

/// Loads user-editable application settings from the config file.
#[tauri::command]
pub async fn load_app_settings() -> Result<AppSettings, String> {
    config::load_app_settings()
        .await
        .map_err(|error| error.to_string())
}

/// Saves user-editable application settings to the config file.
#[tauri::command]
pub async fn save_app_settings(
    state: State<'_, AppState>,
    settings: AppSettings,
) -> Result<AppSettings, String> {
    let settings = config::save_app_settings(settings)
        .await
        .map_err(|error| error.to_string())?;

    crate::tray::update_close_behavior_cache(settings.minimize_to_tray);

    let mut kernel_config = state.kernel.config();
    kernel_config.deepseek.context_window =
        config::parse_context_window(&settings.context_window).unwrap_or(DEFAULT_CONTEXT_WINDOW);
    kernel_config.deepseek.base_url = settings.base_url.clone();
    kernel_config.deepseek.api_key =
        (!settings.api_key.is_empty()).then_some(settings.api_key.clone());
    kernel_config.title_model = settings.title_model.clone();
    state
        .kernel
        .update_deepseek_config(kernel_config.deepseek)
        .await
        .map_err(|error| error.to_string())?;
    state.kernel.update_title_model(kernel_config.title_model);

    Ok(settings)
}

/// Fetches models from a DeepSeek/OpenAI-compatible endpoint.
#[tauri::command]
pub async fn fetch_provider_models(
    base_url: String,
    api_key: String,
) -> Result<Vec<ModelSetting>, String> {
    let base_url = base_url.trim().trim_end_matches('/').to_string();
    let api_key = api_key.trim().to_string();
    if base_url.is_empty() {
        return Err("base_url cannot be empty".to_string());
    }
    if api_key.is_empty() {
        return Err("api_key cannot be empty".to_string());
    }

    let url = format!("{base_url}/models");
    let response = reqwest::Client::new()
        .get(&url)
        .header(reqwest::header::ACCEPT, "application/json")
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(|error| format!("failed to fetch models: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("failed to read model response: {error}"))?;
    if !status.is_success() {
        return Err(format!("fetch models failed with status {status}: {body}"));
    }

    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|error| format!("invalid model response: {error}"))?;
    let items = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .or_else(|| value.as_array())
        .ok_or_else(|| "model response does not contain a data array".to_string())?;

    let mut models = Vec::new();
    for item in items {
        let id = item
            .get("id")
            .or_else(|| item.get("model"))
            .or_else(|| item.get("name"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(id) = id else {
            continue;
        };
        if models.iter().any(|model: &ModelSetting| model.id == id) {
            continue;
        }
        models.push(ModelSetting {
            id: id.to_string(),
            label: id.to_string(),
        });
    }

    if models.is_empty() {
        return Err("model response did not include any model ids".to_string());
    }

    Ok(models)
}
fn normalize_reasoning_effort(value: Option<String>) -> Option<String> {
    let value = value?.trim().to_lowercase();
    matches!(value.as_str(), "high" | "max").then_some(value)
}
