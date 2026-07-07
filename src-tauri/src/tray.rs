//! System tray and window close behavior management.

use crate::config;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{
    image::Image,
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
    App, Emitter, Manager, Window, WindowEvent,
};

/// Cached close behavior: whether the user has configured the close action.
static CLOSE_BEHAVIOR_CONFIGURED: AtomicBool = AtomicBool::new(false);
/// Cached close behavior: whether to minimize to tray on close.
static MINIMIZE_TO_TRAY: AtomicBool = AtomicBool::new(true);

/// Loads the tray icon from the embedded 128x128 PNG.
fn load_tray_icon() -> Image<'static> {
    let png_bytes = include_bytes!("../icons/128x128.png");
    let img = image::load_from_memory(png_bytes).expect("failed to load tray icon");
    let rgba = img.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    Image::new_owned(rgba.into_raw(), width, height)
}

/// Initializes cached close behavior from disk settings.
/// Called once during app setup.
pub fn init_close_behavior_cache() {
    match config::load_app_settings_sync() {
        Ok(settings) => {
            CLOSE_BEHAVIOR_CONFIGURED.store(settings.close_behavior_configured, Ordering::Relaxed);
            MINIMIZE_TO_TRAY.store(settings.minimize_to_tray, Ordering::Relaxed);
        }
        Err(error) => {
            tracing::warn!(
                target: "seekcode::tray",
                %error,
                "failed to load close behavior settings on startup"
            );
        }
    }
}

/// Updates the cached close behavior values (called after settings save).
pub fn update_close_behavior_cache(minimize_to_tray: bool) {
    CLOSE_BEHAVIOR_CONFIGURED.store(true, Ordering::Relaxed);
    MINIMIZE_TO_TRAY.store(minimize_to_tray, Ordering::Relaxed);
}

/// Tray menu item IDs.
const TRAY_SHOW: &str = "show";
const TRAY_QUIT: &str = "quit";

/// Creates the system tray icon with context menu.
pub fn create_system_tray(app: &App) -> anyhow::Result<()> {
    let show_item = MenuItemBuilder::new("Show Window")
        .id(TRAY_SHOW)
        .build(app)?;
    let quit_item = MenuItemBuilder::new("Quit").id(TRAY_QUIT).build(app)?;

    let menu = MenuBuilder::new(app)
        .item(&show_item)
        .item(&quit_item)
        .build()?;

    let _tray = TrayIconBuilder::new()
        .icon(load_tray_icon())
        .menu(&menu)
        .tooltip("SeekCode")
        .on_menu_event(move |app, event| match event.id().as_ref() {
            TRAY_SHOW => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            TRAY_QUIT => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Double-click on tray icon shows the window
            if let tauri::tray::TrayIconEvent::DoubleClick { .. } = event {
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    Ok(())
}

/// Handles window close events based on cached user preference.
///
/// If close behavior hasn't been configured yet, prevents the close and emits
/// an event to the frontend to show the configuration dialog.
pub fn handle_window_event(window: &Window, event: &WindowEvent) {
    if !matches!(event, WindowEvent::CloseRequested { .. }) {
        return;
    }

    if window.label() != "main" {
        return;
    }

    if !CLOSE_BEHAVIOR_CONFIGURED.load(Ordering::Relaxed) {
        // First time: prevent close and ask frontend to show dialog
        let api_ref = match event {
            WindowEvent::CloseRequested { api, .. } => api,
            _ => unreachable!(),
        };
        api_ref.prevent_close();
        let _ = window.emit("app:show-close-behavior-dialog", ());
    } else if MINIMIZE_TO_TRAY.load(Ordering::Relaxed) {
        let api_ref = match event {
            WindowEvent::CloseRequested { api, .. } => api,
            _ => unreachable!(),
        };
        api_ref.prevent_close();
        let _ = window.hide();
    }
}
