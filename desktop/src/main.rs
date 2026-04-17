#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod supervisor;

use std::sync::Arc;

use supervisor::{ServerState, Supervisor, SupervisorConfig};
use tauri::State;

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

fn main() {
    let supervisor = Arc::new(Supervisor::new(SupervisorConfig::default()));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(supervisor)
        .invoke_handler(tauri::generate_handler![
            start_server,
            stop_server,
            server_state,
            server_logs
        ])
        .run(tauri::generate_context!())
        .expect("error while running kiln-desktop");
}
