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
            commands::open_workspace,
            commands::start_agent_task,
            commands::cancel_agent_task,
            commands::list_workspace,
            commands::read_file,
            commands::get_sessions,
            commands::load_app_settings,
            commands::save_app_settings
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
