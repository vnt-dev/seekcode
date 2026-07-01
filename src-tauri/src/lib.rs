mod commands;
mod config;
mod state;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app_state = state::AppState::new().expect("failed to initialize app kernel");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::start_agent_task,
            commands::cancel_agent_task,
            commands::get_sessions,
            commands::open_workspace,
            commands::list_visible_workspaces,
            commands::hide_workspace,
            commands::create_session,
            commands::delete_session,
            commands::update_session_model,
            commands::delete_workspace_sessions,
            commands::list_session_messages,
            commands::session_context_usage,
            commands::session_model_call_stats,
            commands::load_app_settings,
            commands::fetch_provider_models,
            commands::save_app_settings
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
