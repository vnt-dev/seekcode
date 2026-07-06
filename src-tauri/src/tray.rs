//! System tray and window close behavior management.

use crate::config;
use tauri::{
    image::Image,
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
    App, Emitter, Manager, Window, WindowEvent,
};

/// Loads the tray icon from the embedded 128x128 PNG.
fn load_tray_icon() -> Image<'static> {
    let png_bytes = include_bytes!("../icons/128x128.png");
    let img = image::load_from_memory(png_bytes).expect("failed to load tray icon");
    let rgba = img.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    Image::new_owned(rgba.into_raw(), width, height)
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

/// Handles window close events based on user preference.
///
/// If close behavior hasn't been configured yet, prevents the close and emits
/// an event to the frontend to show the configuration dialog.
pub fn handle_window_event(window: &Window, event: &WindowEvent) {
    if !matches!(event, WindowEvent::CloseRequested { .. }) {
        return;
    }

    // Only handle the main window
    if window.label() != "main" {
        return;
    }

    match config::load_app_settings_sync() {
        Ok(settings) => {
            if !settings.close_behavior_configured {
                // First time: prevent close and ask frontend to show dialog
                let api_ref = match event {
                    WindowEvent::CloseRequested { api, .. } => api,
                    _ => unreachable!(),
                };
                api_ref.prevent_close();
                let _ = window.emit("app:show-close-behavior-dialog", ());
            } else if settings.minimize_to_tray {
                // User chose to minimize to tray: hide instead of closing
                let api_ref = match event {
                    WindowEvent::CloseRequested { api, .. } => api,
                    _ => unreachable!(),
                };
                api_ref.prevent_close();
                let _ = window.hide();
            }
            // If minimize_to_tray is false, let the default close behavior proceed
        }
        Err(error) => {
            tracing::warn!(
                target: "seekcode::tray",
                %error,
                "failed to load settings for close behavior"
            );
        }
    }
}
