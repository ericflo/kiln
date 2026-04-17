use std::sync::Arc;
use std::time::Duration;

use tauri::menu::{MenuBuilder, MenuItem, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};

use crate::supervisor::{ServerState, Supervisor};

const TRAY_ID: &str = "kiln-main";
const ITEM_DASHBOARD: &str = "open_dashboard";
const ITEM_SETTINGS: &str = "open_settings";
const ITEM_START: &str = "start_server";
const ITEM_STOP: &str = "stop_server";
const ITEM_LOGS: &str = "view_logs";
const ITEM_QUIT: &str = "quit";

const ICON_STOPPED: &[u8] = include_bytes!("../icons/tray/stopped.png");
const ICON_STARTING: &[u8] = include_bytes!("../icons/tray/starting.png");
const ICON_RUNNING: &[u8] = include_bytes!("../icons/tray/running.png");
const ICON_TRAINING: &[u8] = include_bytes!("../icons/tray/training.png");
const ICON_ERROR: &[u8] = include_bytes!("../icons/tray/error.png");

fn state_icon_bytes(state: &ServerState) -> &'static [u8] {
    match state {
        ServerState::Stopped => ICON_STOPPED,
        ServerState::Starting => ICON_STARTING,
        ServerState::Running => ICON_RUNNING,
        ServerState::TrainingActive => ICON_TRAINING,
        ServerState::Error(_) => ICON_ERROR,
    }
}

pub fn build_tray(app: &AppHandle, supervisor: Arc<Supervisor>) -> tauri::Result<()> {
    let dashboard = MenuItemBuilder::with_id(ITEM_DASHBOARD, "Open Dashboard").build(app)?;
    let settings = MenuItemBuilder::with_id(ITEM_SETTINGS, "Settings…").build(app)?;
    let start = MenuItemBuilder::with_id(ITEM_START, "Start Server").build(app)?;
    let stop = MenuItemBuilder::with_id(ITEM_STOP, "Stop Server")
        .enabled(false)
        .build(app)?;
    let logs = MenuItemBuilder::with_id(ITEM_LOGS, "View Logs").build(app)?;
    let quit = MenuItemBuilder::with_id(ITEM_QUIT, "Quit").build(app)?;

    let menu = MenuBuilder::new(app)
        .items(&[&dashboard, &settings])
        .separator()
        .items(&[&start, &stop, &logs])
        .separator()
        .item(&quit)
        .build()?;

    let icon = tauri::image::Image::from_bytes(state_icon_bytes(&ServerState::Stopped))?;

    let supervisor_for_events = Arc::clone(&supervisor);
    let _tray = TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .tooltip("kiln: Stopped")
        .menu(&menu)
        .on_menu_event(move |app, event| {
            let app_handle = app.clone();
            let supervisor = Arc::clone(&supervisor_for_events);
            match event.id().as_ref() {
                ITEM_START => {
                    tauri::async_runtime::spawn(async move {
                        if let Err(e) = supervisor.start().await {
                            eprintln!("[tray] start_server failed: {}", e);
                        }
                    });
                }
                ITEM_STOP => {
                    tauri::async_runtime::spawn(async move {
                        if let Err(e) = supervisor.stop().await {
                            eprintln!("[tray] stop_server failed: {}", e);
                        }
                    });
                }
                ITEM_QUIT => {
                    app_handle.exit(0);
                }
                ITEM_DASHBOARD => {
                    if let Err(e) = open_dashboard_window(&app_handle) {
                        eprintln!("[tray] open_dashboard_window failed: {}", e);
                    }
                    let _ = app_handle.emit("menu://open-dashboard", ());
                }
                ITEM_SETTINGS => {
                    if let Err(e) = open_settings_window(&app_handle) {
                        eprintln!("[tray] open_settings_window failed: {}", e);
                    }
                    let _ = app_handle.emit("menu://open-settings", ());
                }
                ITEM_LOGS => {
                    if let Err(e) = open_logs_window(&app_handle) {
                        eprintln!("[tray] open_logs_window failed: {}", e);
                    }
                    let _ = app_handle.emit("menu://view-logs", ());
                }
                _ => {}
            }
        })
        .build(app)?;

    spawn_state_watcher(app.clone(), supervisor, start, stop);
    Ok(())
}

fn open_settings_window(app: &AppHandle) -> tauri::Result<()> {
    if let Some(win) = app.get_webview_window("settings") {
        win.show()?;
        win.set_focus()?;
        return Ok(());
    }
    let win = WebviewWindowBuilder::new(app, "settings", WebviewUrl::App("settings.html".into()))
        .title("Kiln Settings")
        .inner_size(520.0, 640.0)
        .resizable(true)
        .visible(true)
        .build()?;
    win.set_focus()?;
    Ok(())
}

fn open_dashboard_window(app: &AppHandle) -> tauri::Result<()> {
    if let Some(win) = app.get_webview_window("dashboard") {
        win.show()?;
        win.set_focus()?;
        return Ok(());
    }
    let win = WebviewWindowBuilder::new(app, "dashboard", WebviewUrl::App("dashboard.html".into()))
        .title("Kiln Dashboard")
        .inner_size(1024.0, 768.0)
        .resizable(true)
        .visible(true)
        .build()?;
    win.set_focus()?;
    Ok(())
}

fn open_logs_window(app: &AppHandle) -> tauri::Result<()> {
    if let Some(win) = app.get_webview_window("logs") {
        win.show()?;
        win.set_focus()?;
        return Ok(());
    }
    let win = WebviewWindowBuilder::new(app, "logs", WebviewUrl::App("logs.html".into()))
        .title("Kiln Logs")
        .inner_size(900.0, 600.0)
        .resizable(true)
        .visible(true)
        .build()?;
    win.set_focus()?;
    Ok(())
}

fn spawn_state_watcher(
    app: AppHandle,
    supervisor: Arc<Supervisor>,
    start_item: MenuItem<tauri::Wry>,
    stop_item: MenuItem<tauri::Wry>,
) {
    tauri::async_runtime::spawn(async move {
        let mut last_kind: Option<&'static str> = None;
        loop {
            let state = supervisor.state().await;
            let kind = state_kind(&state);
            if last_kind != Some(kind) {
                last_kind = Some(kind);
                let tooltip = format!("kiln: {}", state_label(&state));
                if let Some(tray) = app.tray_by_id(TRAY_ID) {
                    let _ = tray.set_tooltip(Some(tooltip.as_str()));
                    if let Ok(img) = tauri::image::Image::from_bytes(state_icon_bytes(&state)) {
                        let _ = tray.set_icon(Some(img));
                    }
                }
                let _ = start_item.set_enabled(start_enabled(&state));
                let _ = stop_item.set_enabled(stop_enabled(&state));
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });
}

fn state_kind(state: &ServerState) -> &'static str {
    match state {
        ServerState::Stopped => "stopped",
        ServerState::Starting => "starting",
        ServerState::Running => "running",
        ServerState::TrainingActive => "training",
        ServerState::Error(_) => "error",
    }
}

pub fn state_label(state: &ServerState) -> String {
    match state {
        ServerState::Stopped => "Stopped".to_string(),
        ServerState::Starting => "Starting…".to_string(),
        ServerState::Running => "Running".to_string(),
        ServerState::TrainingActive => "Training".to_string(),
        ServerState::Error(msg) => format!("Error: {}", msg),
    }
}

pub fn start_enabled(state: &ServerState) -> bool {
    matches!(state, ServerState::Stopped | ServerState::Error(_))
}

pub fn stop_enabled(state: &ServerState) -> bool {
    matches!(
        state,
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_cover_all_variants() {
        assert_eq!(state_label(&ServerState::Stopped), "Stopped");
        assert_eq!(state_label(&ServerState::Starting), "Starting…");
        assert_eq!(state_label(&ServerState::Running), "Running");
        assert_eq!(state_label(&ServerState::TrainingActive), "Training");
        assert_eq!(
            state_label(&ServerState::Error("boom".into())),
            "Error: boom"
        );
    }

    #[test]
    fn kinds_are_distinct() {
        let kinds = [
            state_kind(&ServerState::Stopped),
            state_kind(&ServerState::Starting),
            state_kind(&ServerState::Running),
            state_kind(&ServerState::TrainingActive),
            state_kind(&ServerState::Error("x".into())),
        ];
        let mut sorted = kinds.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), kinds.len(), "kinds must be unique");
    }

    #[test]
    fn start_button_enabled_when_idle() {
        assert!(start_enabled(&ServerState::Stopped));
        assert!(start_enabled(&ServerState::Error("x".into())));
        assert!(!start_enabled(&ServerState::Starting));
        assert!(!start_enabled(&ServerState::Running));
        assert!(!start_enabled(&ServerState::TrainingActive));
    }

    #[test]
    fn stop_button_enabled_when_active() {
        assert!(stop_enabled(&ServerState::Starting));
        assert!(stop_enabled(&ServerState::Running));
        assert!(stop_enabled(&ServerState::TrainingActive));
        assert!(!stop_enabled(&ServerState::Stopped));
        assert!(!stop_enabled(&ServerState::Error("x".into())));
    }

    #[test]
    fn icon_bytes_distinct_per_state() {
        let states = [
            ServerState::Stopped,
            ServerState::Starting,
            ServerState::Running,
            ServerState::TrainingActive,
            ServerState::Error("boom".into()),
        ];
        let mut ptrs: Vec<*const u8> = Vec::with_capacity(states.len());
        for s in &states {
            let bytes = state_icon_bytes(s);
            assert!(!bytes.is_empty(), "icon bytes for {:?} must be non-empty", s);
            ptrs.push(bytes.as_ptr());
        }
        let mut sorted = ptrs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            ptrs.len(),
            "each ServerState must map to a distinct icon slice"
        );
    }
}
