//! Thin Tauri command adapter around AppKernel.

use crate::config::{self, provider_connection, AppSettings, ModelSetting};
use crate::state::AppState;
use seekcode_agent_core::{AgentEvent, AgentTask, StartTaskRequest};
use seekcode_app_kernel::{CreateSessionRequest, OpenWorkspaceRequest, WorkspaceWithSessions};
use seekcode_common::{SessionId, TaskId, WorkspaceId};
use seekcode_storage::{SessionMessageRecord, SessionRecord};
use seekcode_workspace::{FileEntry, FileSnapshot, ListOptions, WorkspaceRoot};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};

/// Starts an agent task.
#[tauri::command]
pub async fn start_agent_task(
    app: AppHandle,
    state: State<'_, AppState>,
    request: StartTaskRequest,
) -> Result<AgentTask, String> {
    let session_id = request.session_id;
    let prompt = request.prompt.clone();
    let session = state
        .kernel
        .get_sessions()
        .await
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|session| session.id == session_id)
        .ok_or_else(|| format!("session {session_id} not found"))?;
    let settings = config::load_app_settings()
        .await
        .map_err(|error| error.to_string())?;
    let (base_url, api_key) =
        provider_connection(&settings, &session.model_provider).ok_or_else(|| {
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
        .map_err(|error| error.to_string())?;
    let started = state
        .kernel
        .start_agent_task(request)
        .await
        .map_err(|error| error.to_string())?;
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
                Some(event)
                    if agent_event_task_id(&event) == Some(task_id)
                        && agent_event_session_id(&event) == Some(session_id) =>
                {
                    let terminal = is_terminal_agent_event(&event);
                    let _ = agent_app.emit("agent:event", event);
                    if terminal {
                        break;
                    }
                }
                Some(_) => {}
                None => break,
            }
        }
    });

    tauri::async_runtime::spawn(async move {
        while let Some(event) = title_events.recv().await {
            let _ = app.emit("session:title_changed", event);
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

/// Lists workspace files.
#[tauri::command]
pub async fn list_workspace(
    state: State<'_, AppState>,
    root: WorkspaceRoot,
    options: Option<ListOptions>,
) -> Result<Vec<FileEntry>, String> {
    state
        .kernel
        .list_files(root, options.unwrap_or_default())
        .await
        .map_err(|error| error.to_string())
}

/// Reads a workspace file.
#[tauri::command]
pub async fn read_file(
    state: State<'_, AppState>,
    root: WorkspaceRoot,
    path: String,
) -> Result<FileSnapshot, String> {
    state
        .kernel
        .read_file(root, PathBuf::from(path))
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
) -> Result<SessionRecord, String> {
    state
        .kernel
        .update_session_model(session_id, model_provider, model)
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
) -> Result<Vec<SessionMessageRecord>, String> {
    state
        .kernel
        .list_session_messages(session_id)
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

    let mut kernel_config = state.kernel.config();
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

fn agent_event_task_id(event: &AgentEvent) -> Option<TaskId> {
    match event {
        AgentEvent::TaskStarted { task_id, .. }
        | AgentEvent::StateChanged { task_id, .. }
        | AgentEvent::ModelRequestStarted { task_id, .. }
        | AgentEvent::ModelChoice { task_id, .. }
        | AgentEvent::AssistantToken { task_id, .. }
        | AgentEvent::AssistantReasoning { task_id, .. }
        | AgentEvent::ToolCallStarted { task_id, .. }
        | AgentEvent::ToolCallFinished { task_id, .. }
        | AgentEvent::ModelRoundFinished { task_id, .. }
        | AgentEvent::Finished { task_id, .. }
        | AgentEvent::Failed { task_id, .. }
        | AgentEvent::Canceled { task_id, .. } => Some(*task_id),
    }
}

fn agent_event_session_id(event: &AgentEvent) -> Option<SessionId> {
    match event {
        AgentEvent::TaskStarted { session_id, .. }
        | AgentEvent::StateChanged { session_id, .. }
        | AgentEvent::ModelRequestStarted { session_id, .. }
        | AgentEvent::ModelChoice { session_id, .. }
        | AgentEvent::AssistantToken { session_id, .. }
        | AgentEvent::AssistantReasoning { session_id, .. }
        | AgentEvent::ToolCallStarted { session_id, .. }
        | AgentEvent::ToolCallFinished { session_id, .. }
        | AgentEvent::ModelRoundFinished { session_id, .. }
        | AgentEvent::Finished { session_id, .. }
        | AgentEvent::Failed { session_id, .. }
        | AgentEvent::Canceled { session_id, .. } => Some(*session_id),
    }
}

fn is_terminal_agent_event(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::Finished { .. } | AgentEvent::Failed { .. } | AgentEvent::Canceled { .. }
    )
}
