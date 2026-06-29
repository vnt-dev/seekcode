//! Thin Tauri command adapter around AppKernel.

use crate::config::{self, AppSettings};
use crate::state::AppState;
use seekcode_agent_core::{AgentEvent, AgentTask, StartTaskRequest};
use seekcode_common::TaskId;
use seekcode_storage::SessionRecord;
use seekcode_workspace::{FileEntry, FileSnapshot, ListOptions, WorkspaceRoot};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter, State};

/// Opens a workspace root.
#[tauri::command]
pub async fn open_workspace(
    state: State<'_, AppState>,
    path: String,
) -> Result<WorkspaceRoot, String> {
    state
        .kernel
        .open_workspace(PathBuf::from(path))
        .await
        .map_err(|error| error.to_string())
}

/// Starts an agent task.
#[tauri::command]
pub async fn start_agent_task(
    app: AppHandle,
    state: State<'_, AppState>,
    request: StartTaskRequest,
) -> Result<AgentTask, String> {
    let mut events = state.kernel.subscribe_agent_events();
    let task = state
        .kernel
        .start_agent_task(request)
        .await
        .map_err(|error| error.to_string())?;
    let task_id = task.id;

    tauri::async_runtime::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) if agent_event_task_id(&event) == Some(task_id) => {
                    let terminal = is_terminal_agent_event(&event);
                    let _ = app.emit("agent:event", event);
                    if terminal {
                        break;
                    }
                }
                Ok(_) => {}
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
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
    state
        .kernel
        .update_deepseek_config(kernel_config.deepseek)
        .await
        .map_err(|error| error.to_string())?;

    Ok(settings)
}

fn agent_event_task_id(event: &AgentEvent) -> Option<TaskId> {
    match event {
        AgentEvent::TaskStarted { task_id, .. }
        | AgentEvent::StateChanged { task_id, .. }
        | AgentEvent::ModelRequestStarted { task_id, .. }
        | AgentEvent::AssistantToken { task_id, .. }
        | AgentEvent::AssistantReasoning { task_id, .. }
        | AgentEvent::ToolCallStarted { task_id, .. }
        | AgentEvent::ToolCallFinished { task_id, .. }
        | AgentEvent::ModelRoundFinished { task_id, .. }
        | AgentEvent::Finished { task_id }
        | AgentEvent::Failed { task_id, .. }
        | AgentEvent::Canceled { task_id } => Some(*task_id),
    }
}

fn is_terminal_agent_event(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::Finished { .. } | AgentEvent::Failed { .. } | AgentEvent::Canceled { .. }
    )
}
