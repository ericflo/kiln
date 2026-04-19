use std::sync::Arc;
use std::time::Duration;

use tauri::menu::{MenuBuilder, MenuItem, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_clipboard_manager::ClipboardExt;
use tauri_plugin_notification::NotificationExt;
use tauri_plugin_shell::ShellExt;
use tokio::sync::RwLock;

use crate::settings::Settings;
use crate::supervisor::{ServerState, Supervisor};

const TRAY_ID: &str = "kiln-main";
const ITEM_STATUS: &str = "status";
const ITEM_DASHBOARD: &str = "open_dashboard";
const ITEM_SETTINGS: &str = "open_settings";
const ITEM_START: &str = "start_server";
const ITEM_STOP: &str = "stop_server";
const ITEM_RESTART: &str = "restart_server";
const ITEM_LOGS: &str = "view_logs";
const ITEM_COPY_URL: &str = "copy_openai_url";
const ITEM_OPEN_IN_BROWSER: &str = "open_in_browser";
const ITEM_CHECK_UPDATES: &str = "check_updates";
const ITEM_ABOUT: &str = "about";
const ITEM_QUIT: &str = "quit";

fn status_menu_text(state: &ServerState) -> String {
    format!("Status: {}", state_label(state))
}

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
    let status = MenuItemBuilder::with_id(ITEM_STATUS, status_menu_text(&ServerState::Stopped))
        .enabled(false)
        .build(app)?;
    let dashboard = MenuItemBuilder::with_id(ITEM_DASHBOARD, "Open Dashboard").build(app)?;
    let settings = MenuItemBuilder::with_id(ITEM_SETTINGS, "Settings…").build(app)?;
    let copy_url =
        MenuItemBuilder::with_id(ITEM_COPY_URL, "Copy OpenAI Base URL").build(app)?;
    let open_in_browser =
        MenuItemBuilder::with_id(ITEM_OPEN_IN_BROWSER, "Open Kiln UI in Browser").build(app)?;
    let start = MenuItemBuilder::with_id(ITEM_START, "Start Server").build(app)?;
    let stop = MenuItemBuilder::with_id(ITEM_STOP, "Stop Server")
        .enabled(false)
        .build(app)?;
    let restart = MenuItemBuilder::with_id(ITEM_RESTART, "Restart Server")
        .enabled(false)
        .build(app)?;
    let logs = MenuItemBuilder::with_id(ITEM_LOGS, "View Logs").build(app)?;
    let check_updates =
        MenuItemBuilder::with_id(ITEM_CHECK_UPDATES, "Check for Updates…").build(app)?;
    let about = MenuItemBuilder::with_id(ITEM_ABOUT, "About Kiln Desktop").build(app)?;
    let quit = MenuItemBuilder::with_id(ITEM_QUIT, "Quit").build(app)?;

    let menu = MenuBuilder::new(app)
        .item(&status)
        .separator()
        .items(&[&dashboard, &settings])
        .separator()
        .items(&[&copy_url, &open_in_browser])
        .separator()
        .items(&[&start, &stop, &restart, &logs])
        .separator()
        .item(&check_updates)
        .separator()
        .item(&about)
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
                ITEM_RESTART => {
                    tauri::async_runtime::spawn(async move {
                        if let Err(e) = supervisor.restart().await {
                            eprintln!("[tray] restart_server failed: {}", e);
                        }
                    });
                }
                ITEM_QUIT => {
                    tauri::async_runtime::spawn(async move {
                        if let Err(e) = supervisor.stop().await {
                            eprintln!("[tray] quit: supervisor.stop failed (continuing): {}", e);
                        }
                        app_handle.exit(0);
                    });
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
                ITEM_COPY_URL => {
                    let supervisor = Arc::clone(&supervisor);
                    tauri::async_runtime::spawn(async move {
                        copy_openai_base_url_to_clipboard(&app_handle, supervisor).await;
                    });
                }
                ITEM_OPEN_IN_BROWSER => {
                    let supervisor = Arc::clone(&supervisor);
                    let app_for_emit = app_handle.clone();
                    tauri::async_runtime::spawn(async move {
                        open_kiln_ui_in_default_browser(&app_handle, supervisor).await;
                    });
                    let _ = app_for_emit.emit("menu://open-in-browser", ());
                }
                ITEM_CHECK_UPDATES => {
                    if let Err(e) = open_dashboard_window(&app_handle) {
                        eprintln!("[tray] open_dashboard_window failed: {}", e);
                    }
                    // Give a freshly-opened dashboard window a moment to mount
                    // its event listener before we emit, so the request isn't
                    // dropped on cold open.
                    tauri::async_runtime::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        if let Err(e) = app_handle.emit("menu://check-updates", ()) {
                            eprintln!("[tray] emit menu://check-updates failed: {}", e);
                        }
                    });
                }
                ITEM_ABOUT => {
                    if let Err(e) = open_about_window(&app_handle) {
                        eprintln!("[tray] open_about_window failed: {}", e);
                    }
                }
                ITEM_STATUS => {}
                _ => {}
            }
        })
        .build(app)?;

    spawn_state_watcher(app.clone(), supervisor, status, start, stop, restart);
    Ok(())
}

pub(crate) fn open_settings_window(app: &AppHandle) -> tauri::Result<()> {
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

/// Return the kiln `/ui` URL for the current server state, or `None` when
/// the server is Stopped or in Error. Pure function kept out of the Tauri
/// runtime so it can be unit-tested directly.
pub(crate) fn kiln_ui_url(state: &ServerState, host: &str, port: u16) -> Option<String> {
    match state {
        ServerState::Stopped | ServerState::Error(_) => None,
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            Some(format!("http://{}:{}/ui", host, port))
        }
    }
}

async fn open_kiln_ui_in_default_browser(app: &AppHandle, supervisor: Arc<Supervisor>) {
    let state = supervisor.state().await;
    let (host, port) = {
        let settings_state = app.state::<Arc<RwLock<Settings>>>();
        let s = settings_state.read().await;
        (s.host.clone(), s.port)
    };
    match kiln_ui_url(&state, &host, port) {
        Some(url) => {
            if let Err(e) = app.shell().open(url, None) {
                eprintln!("[tray] open_kiln_ui_in_default_browser shell.open failed: {}", e);
            }
        }
        None => {
            // Surface the "server not running" case to the user instead of
            // silently no-opping: without this the tray item looks broken.
            eprintln!(
                "[tray] open Kiln UI skipped — server not running (state: {})",
                state_label(&state)
            );
            let body = match &state {
                ServerState::Error(msg) => format!("Server not running ({}).", msg),
                _ => format!("Server is {}. Start it first.", state_label(&state)),
            };
            let _ = app
                .notification()
                .builder()
                .title("Kiln UI unavailable")
                .body(body)
                .show();
        }
    }
}

async fn copy_openai_base_url_to_clipboard(app: &AppHandle, supervisor: Arc<Supervisor>) {
    let state = supervisor.state().await;
    if !matches!(
        state,
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive
    ) {
        eprintln!(
            "[tray] copy OpenAI base URL skipped — server not running (state: {})",
            state_label(&state)
        );
        return;
    }
    let (host, port) = {
        let settings_state = app.state::<Arc<RwLock<Settings>>>();
        let s = settings_state.read().await;
        (s.host.clone(), s.port)
    };
    let url = crate::openai_base_url(&host, port);
    if let Err(e) = app.clipboard().write_text(url) {
        eprintln!("[tray] clipboard write_text failed: {}", e);
    }
}

pub(crate) fn open_logs_window(app: &AppHandle) -> tauri::Result<()> {
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

pub(crate) fn open_about_window(app: &AppHandle) -> tauri::Result<()> {
    if let Some(win) = app.get_webview_window("about") {
        win.show()?;
        win.set_focus()?;
        return Ok(());
    }
    let win = WebviewWindowBuilder::new(app, "about", WebviewUrl::App("about.html".into()))
        .title("About Kiln Desktop")
        .inner_size(420.0, 380.0)
        .resizable(false)
        .visible(true)
        .build()?;
    win.set_focus()?;
    Ok(())
}

fn spawn_state_watcher(
    app: AppHandle,
    supervisor: Arc<Supervisor>,
    status_item: MenuItem<tauri::Wry>,
    start_item: MenuItem<tauri::Wry>,
    stop_item: MenuItem<tauri::Wry>,
    restart_item: MenuItem<tauri::Wry>,
) {
    tauri::async_runtime::spawn(async move {
        let mut last_kind: Option<&'static str> = None;
        loop {
            let state = supervisor.state().await;
            let kind = state_kind(&state);
            if last_kind != Some(kind) {
                let prev_kind = last_kind;
                last_kind = Some(kind);
                let port = {
                    let settings_state = app.state::<Arc<RwLock<Settings>>>();
                    let s = settings_state.read().await;
                    s.port
                };
                let tooltip = format_tray_tooltip(&state, port);
                if let Some(tray) = app.tray_by_id(TRAY_ID) {
                    let _ = tray.set_tooltip(Some(tooltip.as_str()));
                    if let Ok(img) = tauri::image::Image::from_bytes(state_icon_bytes(&state)) {
                        let _ = tray.set_icon(Some(img));
                    }
                }
                let _ = status_item.set_text(status_menu_text(&state));
                let _ = start_item.set_enabled(start_enabled(&state));
                let _ = stop_item.set_enabled(stop_enabled(&state));
                let _ = restart_item.set_enabled(restart_enabled(&state));

                // Fire a single native OS notification on entry into Error.
                // We only notify on the transition into the error kind, not
                // on every poll while still errored, and we ignore failures
                // so a broken notification backend can never panic the
                // watcher loop.
                if kind == "error" && prev_kind != Some("error") {
                    if let ServerState::Error(msg) = &state {
                        let _ = app
                            .notification()
                            .builder()
                            .title("Kiln server crashed")
                            .body(msg.as_str())
                            .show();
                    }
                }
                if kind == "running" && prev_kind != Some("running") {
                    let _ = app
                        .notification()
                        .builder()
                        .title("Kiln server ready")
                        .body(ready_notification_body(port))
                        .show();
                }
                if kind == "stopped" && should_notify_stopped(prev_kind) {
                    let _ = app
                        .notification()
                        .builder()
                        .title("Kiln server stopped")
                        .body("The local kiln server is no longer running.")
                        .show();
                }
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

pub fn format_tray_tooltip(state: &ServerState, port: u16) -> String {
    match state {
        ServerState::Running => format!("kiln: Running on :{}", port),
        ServerState::TrainingActive => format!("kiln: Training on :{}", port),
        _ => format!("kiln: {}", state_label(state)),
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

pub fn restart_enabled(state: &ServerState) -> bool {
    matches!(
        state,
        ServerState::Running | ServerState::TrainingActive | ServerState::Error(_)
    )
}

pub fn ready_notification_body(port: u16) -> String {
    format!("OpenAI base URL: http://localhost:{}/v1", port)
}

pub fn should_notify_stopped(prev_kind: Option<&str>) -> bool {
    matches!(prev_kind, Some("running") | Some("training"))
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
    fn state_kind_for_error_is_error() {
        assert_eq!(state_kind(&ServerState::Error("boom".into())), "error");
        assert_eq!(state_kind(&ServerState::Error(String::new())), "error");
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
    fn restart_button_enabled_when_running_or_errored() {
        assert!(restart_enabled(&ServerState::Running));
        assert!(restart_enabled(&ServerState::TrainingActive));
        assert!(restart_enabled(&ServerState::Error("x".into())));
        assert!(!restart_enabled(&ServerState::Stopped));
        assert!(!restart_enabled(&ServerState::Starting));
    }

    #[test]
    fn status_text_matches_label() {
        assert_eq!(
            format!("Status: {}", state_label(&ServerState::Stopped)),
            "Status: Stopped"
        );
        assert_eq!(
            format!("Status: {}", state_label(&ServerState::Starting)),
            "Status: Starting…"
        );
        assert_eq!(
            format!("Status: {}", state_label(&ServerState::Running)),
            "Status: Running"
        );
        assert_eq!(
            format!("Status: {}", state_label(&ServerState::TrainingActive)),
            "Status: Training"
        );
        assert_eq!(
            format!("Status: {}", state_label(&ServerState::Error("boom".into()))),
            "Status: Error: boom"
        );
        // Helper must match the inline format used by the watcher.
        assert_eq!(
            status_menu_text(&ServerState::Running),
            format!("Status: {}", state_label(&ServerState::Running))
        );
    }

    #[test]
    fn tooltip_includes_port_when_running() {
        assert_eq!(
            format_tray_tooltip(&ServerState::Running, 9000),
            "kiln: Running on :9000"
        );
        assert_eq!(
            format_tray_tooltip(&ServerState::TrainingActive, 9000),
            "kiln: Training on :9000"
        );
    }

    #[test]
    fn tooltip_omits_port_for_non_active_states() {
        assert_eq!(
            format_tray_tooltip(&ServerState::Stopped, 9000),
            "kiln: Stopped"
        );
        assert_eq!(
            format_tray_tooltip(&ServerState::Starting, 9000),
            "kiln: Starting…"
        );
        assert_eq!(
            format_tray_tooltip(&ServerState::Error("boom".into()), 9000),
            "kiln: Error: boom"
        );
    }

    #[test]
    fn tooltip_uses_configured_port() {
        assert_eq!(
            format_tray_tooltip(&ServerState::Running, 8000),
            "kiln: Running on :8000"
        );
        assert_eq!(
            format_tray_tooltip(&ServerState::Running, 12345),
            "kiln: Running on :12345"
        );
    }

    #[test]
    fn kiln_ui_url_returns_none_for_stopped() {
        assert_eq!(kiln_ui_url(&ServerState::Stopped, "127.0.0.1", 9000), None);
    }

    #[test]
    fn kiln_ui_url_returns_none_for_error() {
        assert_eq!(
            kiln_ui_url(&ServerState::Error("boom".into()), "127.0.0.1", 9000),
            None
        );
        assert_eq!(
            kiln_ui_url(&ServerState::Error(String::new()), "127.0.0.1", 9000),
            None
        );
    }

    #[test]
    fn kiln_ui_url_returns_url_for_running_and_training() {
        assert_eq!(
            kiln_ui_url(&ServerState::Running, "127.0.0.1", 9000),
            Some("http://127.0.0.1:9000/ui".to_string())
        );
        assert_eq!(
            kiln_ui_url(&ServerState::TrainingActive, "127.0.0.1", 9000),
            Some("http://127.0.0.1:9000/ui".to_string())
        );
        assert_eq!(
            kiln_ui_url(&ServerState::Starting, "127.0.0.1", 8080),
            Some("http://127.0.0.1:8080/ui".to_string())
        );
    }

    #[test]
    fn ready_notification_body_contains_openai_base_url() {
        let body = ready_notification_body(9000);
        assert!(body.contains("http://localhost:9000/v1"), "body was {}", body);
        assert!(body.contains("OpenAI base URL"), "body was {}", body);
        assert_eq!(
            ready_notification_body(8080),
            "OpenAI base URL: http://localhost:8080/v1"
        );
    }

    #[test]
    fn should_notify_stopped_only_when_previously_up() {
        assert!(should_notify_stopped(Some("running")));
        assert!(should_notify_stopped(Some("training")));
        assert!(!should_notify_stopped(None));
        assert!(!should_notify_stopped(Some("starting")));
        assert!(!should_notify_stopped(Some("error")));
        assert!(!should_notify_stopped(Some("stopped")));
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
