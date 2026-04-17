#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod settings;
mod supervisor;
mod tray;

use std::sync::Arc;

use settings::{apply_to_supervisor_config, Settings};
use supervisor::{ServerState, Supervisor, SupervisorConfig};
use tauri::{Manager, State};
use tokio::sync::RwLock;

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
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
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
            set_settings
        ])
        .run(tauri::generate_context!())
        .expect("error while running kiln-desktop");
}
