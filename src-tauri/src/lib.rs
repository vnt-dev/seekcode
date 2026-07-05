mod commands;
mod config;
mod state;

use anyhow::{anyhow, Context, Result};
use rolling_file::{BasicRollingFileAppender, RollingConditionBasic};
use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    if let Err(error) = try_run() {
        eprintln!("failed to run SeekCode: {error:#}");
        std::process::exit(1);
    }
}

fn try_run() -> Result<()> {
    let _log_guard = init_file_logging()?;
    let app_state = state::AppState::new().context("failed to initialize app kernel")?;

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
        .context("error while running tauri application")?;

    Ok(())
}

fn init_file_logging() -> Result<WorkerGuard> {
    let logs_dir = seekcode_home_dir()?.join("logs");
    if !logs_dir.exists() {
        std::fs::create_dir_all(&logs_dir).context("failed to create logs directory")?;
    }

    let file_appender = BasicRollingFileAppender::new(
        logs_dir.join("seekcode.log"),
        RollingConditionBasic::new()
            .daily()
            .max_size(5 * 1024 * 1024),
        10,
    )
    .context("failed to build RollingFileAppender")?;
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "info,seekcode=debug,seekcode_agent_core=debug,seekcode_app_kernel=debug,seekcode_deepseek_client=debug,seekcode_tool_system=debug,tower_http=debug".into()
            }),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .try_init()
        .map_err(|error| anyhow!("failed to initialize tracing subscriber: {error}"))?;

    tracing::info!(
        target: "seekcode::logging",
        path = %logs_dir.display(),
        "file logging initialized"
    );

    Ok(guard)
}

fn seekcode_home_dir() -> Result<PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .context("failed to resolve home directory")?;
    Ok(home.join(".seekcode"))
}
