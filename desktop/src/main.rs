#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod installer;
mod poller;
mod settings;
mod supervisor;
mod tray;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use settings::{apply_to_supervisor_config, Settings};
use supervisor::{ServerState, Supervisor, SupervisorConfig};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_autostart::{ManagerExt, MacosLauncher};
use tauri_plugin_updater::UpdaterExt;
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

/// Shared cancel/busy flags for the install pipeline.
///
/// `in_progress` guards against parallel downloads from impatient
/// clicking; `cancel` is flipped by `cancel_kiln_download` and polled
/// inside the download loop.
#[derive(Default)]
struct InstallerState {
    in_progress: AtomicBool,
    cancel: Arc<AtomicBool>,
}

type InstallerHandle = Arc<InstallerState>;

#[tauri::command]
async fn start_server(sup: State<'_, Arc<Supervisor>>) -> Result<(), String> {
    sup.start().await
}

#[tauri::command]
async fn stop_server(sup: State<'_, Arc<Supervisor>>) -> Result<(), String> {
    sup.stop().await
}

#[tauri::command]
async fn restart_server(sup: State<'_, Arc<Supervisor>>) -> Result<(), String> {
    sup.restart().await
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
async fn copy_logs(
    app: tauri::AppHandle,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<usize, String> {
    use tauri_plugin_clipboard_manager::ClipboardExt;
    let lines = sup.logs().await;
    let text = lines.join("\n");
    let count = lines.len();
    app.clipboard()
        .write_text(text)
        .map_err(|e| format!("clipboard write failed: {}", e))?;
    Ok(count)
}

#[tauri::command]
async fn save_logs_to_file(
    path: String,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<usize, String> {
    let lines = sup.logs().await;
    let count = lines.len();
    let text = lines.join("\n");
    std::fs::write(&path, text).map_err(|e| format!("write failed: {}", e))?;
    Ok(count)
}

#[tauri::command]
async fn get_settings(state: State<'_, SettingsState>) -> Result<Settings, String> {
    Ok(state.read().await.clone())
}

#[tauri::command]
async fn default_settings() -> Result<Settings, String> {
    Ok(Settings::default())
}

#[derive(serde::Serialize)]
struct BinaryStatus {
    installed: bool,
    resolved_path: Option<String>,
    configured_path: String,
    platform_supported: bool,
    install_target: Option<String>,
    install_dir: Option<String>,
}

/// Report whether the `kiln` binary is reachable, whether this platform
/// has a prebuilt release the desktop can download, and where an
/// auto-install would place the binary. Used by the dashboard onboarding
/// screen to decide between "Download" and "Build from source" paths.
#[tauri::command]
async fn get_binary_status(
    app: AppHandle,
    state: State<'_, SettingsState>,
) -> Result<BinaryStatus, String> {
    let configured = {
        let s = state.read().await;
        s.kiln_binary
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("kiln"))
    };
    let resolved = installer::resolve_binary(&configured);
    let install_dir = installer::install_dir(&app).ok().map(|p| p.display().to_string());
    Ok(BinaryStatus {
        installed: resolved.is_some(),
        resolved_path: resolved.map(|p| p.display().to_string()),
        configured_path: configured.display().to_string(),
        platform_supported: installer::supports_auto_install(),
        install_target: installer::current_target().map(|s| s.to_string()),
        install_dir,
    })
}

/// Download, verify, and install the latest prebuilt `kiln` server binary
/// for this platform. Emits progress events at `kiln-install-progress` and
/// a terminal `kiln-install-done` / `kiln-install-failed` event when the
/// pipeline finishes.
#[tauri::command]
async fn download_kiln_server(
    app: AppHandle,
    settings_state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
    inst: State<'_, InstallerHandle>,
) -> Result<(), String> {
    // Reject overlapping installs rather than racing two downloads into
    // the same file. The frontend's "Download" button is disabled while
    // `in_progress` is true, but belt-and-suspenders: we also guard here.
    if inst.in_progress.swap(true, Ordering::SeqCst) {
        return Err("an install is already in progress".into());
    }
    inst.cancel.store(false, Ordering::SeqCst);

    let app_for_task = app.clone();
    let settings_state = (*settings_state).clone();
    let supervisor = (*sup).clone();
    let inst_for_task = (*inst).clone();

    tauri::async_runtime::spawn(async move {
        let result = installer::install_latest_server(
            app_for_task.clone(),
            Arc::clone(&inst_for_task.cancel),
        )
        .await;
        match result {
            Ok((path, version)) => {
                let cfg_update = {
                    let mut s = settings_state.write().await;
                    s.kiln_binary = Some(path.clone());
                    if let Err(e) = s.save(&app_for_task) {
                        eprintln!("[install] settings.save failed: {}", e);
                    }
                    let mut cfg = SupervisorConfig::default();
                    apply_to_supervisor_config(&s, &mut cfg);
                    cfg
                };
                supervisor.update_config(cfg_update).await;
                let _ = app_for_task.emit(
                    installer::INSTALL_DONE_EVENT,
                    installer::InstallDone {
                        path: path.display().to_string(),
                        version,
                    },
                );
            }
            Err(err) => {
                let cancelled = inst_for_task.cancel.load(Ordering::SeqCst);
                let _ = app_for_task.emit(
                    installer::INSTALL_FAILED_EVENT,
                    installer::InstallFailed {
                        error: err,
                        cancelled,
                    },
                );
            }
        }
        inst_for_task.in_progress.store(false, Ordering::SeqCst);
    });

    Ok(())
}

/// Signal the currently-running install to abort. The download loop polls
/// the cancel flag after every chunk; cancellation surfaces as an error
/// event with `cancelled: true`.
#[tauri::command]
async fn cancel_kiln_download(inst: State<'_, InstallerHandle>) -> Result<(), String> {
    inst.cancel.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
async fn open_settings(app: AppHandle) -> Result<(), String> {
    tray::open_settings_window(&app).map_err(|e| e.to_string())
}

#[tauri::command]
async fn open_logs(app: AppHandle) -> Result<(), String> {
    tray::open_logs_window(&app).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[derive(serde::Serialize)]
struct DiagnosticInfo {
    version: String,
    os: String,
    arch: String,
}

#[tauri::command]
fn get_diagnostic_info() -> DiagnosticInfo {
    DiagnosticInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

#[tauri::command]
fn path_info(path: String) -> serde_json::Value {
    if path.trim().is_empty() {
        return serde_json::json!({
            "exists": false,
            "is_file": false,
            "is_dir": false,
        });
    }
    let p = std::path::Path::new(&path);
    serde_json::json!({
        "exists": p.exists(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    })
}

#[derive(serde::Serialize)]
struct UpdateCheckResult {
    available: bool,
    version: Option<String>,
    current_version: Option<String>,
    notes: Option<String>,
}

/// Ask the configured updater endpoint whether a new release is available.
/// The frontend uses the result to prompt the user to install via
/// `install_update`.
#[tauri::command]
async fn check_for_updates(app: AppHandle) -> Result<UpdateCheckResult, String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    let update = updater.check().await.map_err(|e| e.to_string())?;
    match update {
        Some(u) => Ok(UpdateCheckResult {
            available: true,
            version: Some(u.version.clone()),
            current_version: Some(u.current_version.clone()),
            notes: u.body.clone(),
        }),
        None => Ok(UpdateCheckResult {
            available: false,
            version: None,
            current_version: None,
            notes: None,
        }),
    }
}

/// Re-check, then download and install the available update, then restart the
/// app. We re-check rather than caching an `Update` handle because Tauri
/// commands are stateless and the small race window is acceptable.
#[tauri::command]
async fn install_update(app: AppHandle) -> Result<(), String> {
    let updater = app.updater().map_err(|e| e.to_string())?;
    let update = updater.check().await.map_err(|e| e.to_string())?;
    let Some(update) = update else {
        return Err("No update available".to_string());
    };
    update
        .download_and_install(|_chunk, _total| {}, || {})
        .await
        .map_err(|e| e.to_string())?;
    app.restart();
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
        ServerState::Stopped | ServerState::Error(_) | ServerState::NoBinary(_) => Ok(String::new()),
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            let s = state.read().await;
            Ok(format!("http://{}:{}/ui", s.host, s.port))
        }
    }
}

/// Open the kiln /ui page in the user's default external browser. This gives
/// users devtools/bookmarking/URL-sharing access without crowding the in-app
/// dashboard toolbar. Returns Err when the server is not running so the UI
/// can surface an appropriate hint.
#[tauri::command]
async fn open_kiln_ui_in_browser(
    app: AppHandle,
    state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<(), String> {
    use tauri_plugin_shell::ShellExt;
    let server_state = sup.state().await;
    let (host, port) = {
        let s = state.read().await;
        (s.host.clone(), s.port)
    };
    match tray::kiln_ui_url(&server_state, &host, port) {
        Some(url) => app.shell().open(url, None).map_err(|e| e.to_string()),
        None => Err("Kiln server is not running".into()),
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
        ServerState::Stopped | ServerState::Error(_) | ServerState::NoBinary(_) => Ok(String::new()),
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            let s = state.read().await;
            Ok(openai_base_url(&s.host, s.port))
        }
    }
}

#[derive(serde::Serialize)]
struct ActiveAdapterInfo {
    active: Option<String>,
    available_count: usize,
}

/// Report the currently active LoRA adapter (if any) plus the count of
/// adapters available on disk by polling GET /v1/adapters on the kiln
/// server. Returns an empty result on any HTTP/parse error so the UI can
/// degrade gracefully rather than throwing.
#[tauri::command]
async fn get_active_adapter(
    state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<ActiveAdapterInfo, String> {
    let server_state = sup.state().await;
    let url = match server_state {
        ServerState::Stopped | ServerState::Error(_) | ServerState::NoBinary(_) => {
            return Ok(ActiveAdapterInfo {
                active: None,
                available_count: 0,
            });
        }
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            let s = state.read().await;
            format!("http://{}:{}/v1/adapters", s.host, s.port)
        }
    };

    #[derive(serde::Deserialize)]
    struct Response {
        active: Option<String>,
        available: Vec<serde_json::Value>,
    }

    let empty = ActiveAdapterInfo {
        active: None,
        available_count: 0,
    };

    let resp = match reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        _ => return Ok(empty),
    };
    match resp.json::<Response>().await {
        Ok(r) => Ok(ActiveAdapterInfo {
            active: r.active,
            available_count: r.available.len(),
        }),
        Err(_) => Ok(empty),
    }
}

#[derive(serde::Serialize)]
struct ActiveTrainingJob {
    job_id: String,
    state: String,
    progress: Option<f64>,
    current_loss: Option<f64>,
    adapter_name: Option<String>,
}

#[derive(serde::Serialize, Default)]
struct TrainingStatusInfo {
    active: Option<ActiveTrainingJob>,
    total_jobs: usize,
}

/// Report the currently-running training job (if any) plus the total number
/// of jobs known to the kiln server by polling GET /v1/train/status. Returns
/// an empty result on any HTTP/parse error or when the server is
/// Stopped/Error so the UI can degrade gracefully.
#[tauri::command]
async fn get_training_status(
    state: State<'_, SettingsState>,
    sup: State<'_, Arc<Supervisor>>,
) -> Result<TrainingStatusInfo, String> {
    let server_state = sup.state().await;
    let url = match server_state {
        ServerState::Stopped | ServerState::Error(_) | ServerState::NoBinary(_) => {
            return Ok(TrainingStatusInfo::default());
        }
        ServerState::Starting | ServerState::Running | ServerState::TrainingActive => {
            let s = state.read().await;
            format!("http://{}:{}/v1/train/status", s.host, s.port)
        }
    };

    #[derive(serde::Deserialize)]
    struct Job {
        job_id: String,
        state: String,
        #[serde(default)]
        progress: Option<f64>,
        #[serde(default)]
        current_loss: Option<f64>,
        #[serde(default)]
        adapter_name: Option<String>,
    }

    let resp = match reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        _ => return Ok(TrainingStatusInfo::default()),
    };
    let jobs: Vec<Job> = match resp.json().await {
        Ok(j) => j,
        Err(_) => return Ok(TrainingStatusInfo::default()),
    };

    let active = jobs
        .iter()
        .find(|j| j.state.eq_ignore_ascii_case("running"))
        .or_else(|| jobs.first())
        .map(|j| ActiveTrainingJob {
            job_id: j.job_id.clone(),
            state: j.state.clone(),
            progress: j.progress,
            current_loss: j.current_loss,
            adapter_name: j.adapter_name.clone(),
        });

    Ok(TrainingStatusInfo {
        active,
        total_jobs: jobs.len(),
    })
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
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
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
            app.manage(Arc::new(InstallerState::default()) as InstallerHandle);

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
            restart_server,
            server_state,
            server_logs,
            copy_logs,
            save_logs_to_file,
            get_settings,
            default_settings,
            set_settings,
            get_kiln_url,
            get_openai_base_url,
            open_kiln_ui_in_browser,
            get_active_adapter,
            get_training_status,
            open_settings,
            open_logs,
            get_app_version,
            get_diagnostic_info,
            check_for_updates,
            install_update,
            get_binary_status,
            download_kiln_server,
            cancel_kiln_download,
            path_info
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
