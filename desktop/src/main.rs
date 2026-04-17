#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod poller;
mod settings;
mod supervisor;
mod tray;

use std::sync::Arc;

use settings::{apply_to_supervisor_config, Settings};
use supervisor::{ServerState, Supervisor, SupervisorConfig};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_autostart::{ManagerExt, MacosLauncher};
use tokio::sync::RwLock;

/// Reconcile the OS autostart registration with the desired `launch_at_login`
/// setting. Errors are logged but never propagated — a broken autostart
/// registration must not crash the app or block settings persistence.
fn reconcile_autolaunch(app: &AppHandle, desired: bool) {
    let manager = app.autolaunch();
    let enabled = match manager.is_enabled() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[main] autolaunch is_enabled failed: {}", e);
            return;
        }
    };
    if desired && !enabled {
        if let Err(e) = manager.enable() {
            eprintln!("[main] autolaunch enable failed: {}", e);
        }
    } else if !desired && enabled {
        if let Err(e) = manager.disable() {
            eprintln!("[main] autolaunch disable failed: {}", e);
        }
    }
}

type SettingsState = Arc<RwLock<Settings>>;

#[tauri::command]
async fn start_server(sup: State<'_, Arc<Supervisor>>) -> Result<(), String> {
    sup.start().await
}

#[tauri::command]
async fn stop_server(sup: State<'_, Arc<Supervisor>>) -> Result<(), String> {
    sup.stop().await
}

#[tauri::command]
async fn server_state(sup: State<'_, Arc<Supervisor>>) -> Result<ServerState, String> {
    Ok(sup.state().await)
}

#[tauri::command]
async fn server_logs(sup: State<'_, Arc<Supervisor>>) -> Result<Vec<String>, String> {
    Ok(sup.logs().await)
}

#[tauri::command]
async fn get_settings(state: State<'_, SettingsState>) -> Result<Settings, String> {
    Ok(state.read().await.clone())
}

/// Build the OpenAI-compatible base URL for a given host/port. Kept as a pure
/// helper so it can be unit-tested without a Tauri runtime.
pub fn openai_base_url(host: &str, port: u16) -> String {
    format!("http://{}:{}/v1", host, port)
}

/// Return the kiln server's /ui URL based on the current settings, but only
/// when the supervisor reports a state that should have the HTTP server up.
/// Returns an empty string when the server is Stopped or in Error, so the
/// dashboard UI can distinguish "server not running" from a real URL.
#[tauri::command]
async fn get_kiln_url(
    state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<String, String> {
    let server_state = sup.state().await;
    match server_state {
        ServerState::Stopped | ServerState::Error(_) => Ok(String::new()),
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            let s = state.read().await;
            Ok(format!("http://{}:{}/ui", s.host, s.port))
        }
    }
}

/// Return the OpenAI-compatible base URL (`http://<host>:<port>/v1`) based on
/// current settings, but only when the supervisor reports a state that should
/// have the HTTP server up. Returns an empty string when the server is
/// Stopped or in Error, so the UI can distinguish "server not running" from a
/// real URL.
#[tauri::command]
async fn get_openai_base_url(
    state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<String, String> {
    let server_state = sup.state().await;
    match server_state {
        ServerState::Stopped | ServerState::Error(_) => Ok(String::new()),
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            let s = state.read().await;
            Ok(openai_base_url(&s.host, s.port))
        }
    }
}

/// Persist the supplied settings to disk and rebuild the supervisor's
/// `SupervisorConfig`. A currently-running server is NOT restarted; the new
/// args take effect on the next `start_server` call.
#[tauri::command]
async fn set_settings(
    new: Settings,
    app: tauri::AppHandle,
    state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<(), String> {
    new.save(&app)?;
    {
        let mut guard = state.write().await;
        *guard = new.clone();
    }
    let mut cfg = SupervisorConfig::default();
    apply_to_supervisor_config(&new, &mut cfg);
    sup.update_config(cfg).await;
    reconcile_autolaunch(&app, new.launch_at_login);
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let handle = app.handle().clone();
            let settings = Settings::load(&handle);

            let mut cfg = SupervisorConfig::default();
            apply_to_supervisor_config(&settings, &mut cfg);
            let supervisor = Arc::new(Supervisor::new(cfg));
            let settings_state: SettingsState = Arc::new(RwLock::new(settings.clone()));

            app.manage(Arc::clone(&supervisor));
            app.manage(Arc::clone(&settings_state));

            tray::build_tray(app.handle(), Arc::clone(&supervisor))?;

            reconcile_autolaunch(&handle, settings.launch_at_login);

            if settings.auto_start {
                let sup = Arc::clone(&supervisor);
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = sup.start().await {
                        eprintln!("[main] auto_start failed: {}", e);
                    }
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            server_state,
            server_logs,
            get_settings,
            set_settings,
            get_kiln_url,
            get_openai_base_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running kiln-desktop");
}

#[cfg(test)]
mod tests {
    use super::openai_base_url;

    #[test]
    fn default_loopback() {
        assert_eq!(openai_base_url("127.0.0.1", 8000), "http://127.0.0.1:8000/v1");
    }

    #[test]
    fn alternate_host_and_port() {
        assert_eq!(openai_base_url("0.0.0.0", 9001), "http://0.0.0.0:9001/v1");
        assert_eq!(openai_base_url("localhost", 80), "http://localhost:80/v1");
    }
}
