#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod hf_download;
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

/// Updater slice 2: download a newer kiln release tarball into the staging
/// directory under `app_data_dir/kiln-updates/`. **Does not** extract, swap
/// the live binary, or restart the supervisor — those steps land in slice 3
/// (see `desktop/docs/binary-update.md`).
///
/// Progress is streamed as `download_update_progress`; terminal events are
/// `download_update_done` (success) and `download_update_failed` (error or
/// cancelled). Reuses [`InstallerState`] so a fresh-install and an update
/// download can't race the same on-disk file or cancel flag.
#[tauri::command]
async fn download_kiln_update(
    app: AppHandle,
    version: String,
    inst: State<'_, InstallerHandle>,
) -> Result<(), String> {
    if inst.in_progress.swap(true, Ordering::SeqCst) {
        return Err("an install or update download is already in progress".into());
    }
    inst.cancel.store(false, Ordering::SeqCst);

    let app_for_task = app.clone();
    let inst_for_task = (*inst).clone();
    let version_for_task = version.clone();

    tauri::async_runtime::spawn(async move {
        let result = installer::download_release_binary(
            &app_for_task,
            &version_for_task,
            Arc::clone(&inst_for_task.cancel),
        )
        .await;
        match result {
            Ok((path, bytes)) => {
                let _ = app_for_task.emit(
                    installer::UPDATE_DOWNLOAD_DONE_EVENT,
                    installer::UpdateDownloadDone {
                        version: version_for_task,
                        path: path.display().to_string(),
                        bytes,
                    },
                );
            }
            Err(err) => {
                let cancelled = inst_for_task.cancel.load(Ordering::SeqCst);
                let _ = app_for_task.emit(
                    installer::UPDATE_DOWNLOAD_FAILED_EVENT,
                    installer::UpdateDownloadFailed {
                        version: version_for_task,
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

/// Fetch the sha256 digest for the tarball published alongside `version`
/// on the current target. The settings UI calls this right before
/// [`extract_update`] so it can hand the expected digest to the
/// orchestrator. Kept as a separate command (rather than baked into
/// `extract_update`) so slice 3a stays self-contained without forcing
/// slice 2's download step to persist the digest to disk.
#[tauri::command]
async fn fetch_update_sha256(version: String) -> Result<String, String> {
    installer::fetch_release_sha256(&version).await
}

/// Updater slice 3a: verify the staged tarball and extract the kiln
/// binary to `<app_data_dir>/bin/kiln.new`. **Does not** swap
/// `bin/kiln`, stop the supervisor, or restart — atomic swap + restart
/// land in slice 3b (see `desktop/docs/binary-update.md`).
///
/// Progress is streamed as `extract_update_progress`; terminal events are
/// `extract_update_done` (success) and `extract_update_failed` (error or
/// cancelled). Reuses [`InstallerState`] so a fresh-install, slice-2
/// download, and slice-3a extract can't race the same on-disk file or
/// cancel flag.
#[tauri::command]
async fn extract_update(
    app: AppHandle,
    version: String,
    expected_sha256: String,
    inst: State<'_, InstallerHandle>,
) -> Result<(), String> {
    if inst.in_progress.swap(true, Ordering::SeqCst) {
        return Err(
            "an install, update download, or extract is already in progress".into(),
        );
    }
    inst.cancel.store(false, Ordering::SeqCst);

    let app_for_task = app.clone();
    let inst_for_task = (*inst).clone();
    let version_for_task = version.clone();

    tauri::async_runtime::spawn(async move {
        let result = installer::extract_and_verify_update(
            &app_for_task,
            &version_for_task,
            &expected_sha256,
            Arc::clone(&inst_for_task.cancel),
        )
        .await;
        match result {
            Ok((path, bytes)) => {
                let _ = app_for_task.emit(
                    installer::EXTRACT_UPDATE_DONE_EVENT,
                    installer::ExtractUpdateDone {
                        version: version_for_task,
                        path: path.display().to_string(),
                        bytes,
                        sha256_ok: true,
                    },
                );
            }
            Err(err) => {
                let cancelled = inst_for_task.cancel.load(Ordering::SeqCst);
                let _ = app_for_task.emit(
                    installer::EXTRACT_UPDATE_FAILED_EVENT,
                    installer::ExtractUpdateFailed {
                        version: version_for_task,
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

/// Kick off a HuggingFace model download into
/// `app_data_dir()/models/<sanitized_repo>/`. Progress is streamed via
/// the `hf-download-progress` event; terminal success / failure arrive as
/// `hf-download-done` / `hf-download-error`. The returned `Result` only
/// indicates whether the background task was spawned; the caller should
/// wait on events for actual completion.
#[tauri::command]
async fn download_hf_model(
    app: AppHandle,
    repo_id: String,
    revision: Option<String>,
    token: Option<String>,
) -> Result<(), String> {
    let req = hf_download::HfDownloadRequest {
        repo_id,
        revision,
        token,
    };
    let app_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        match hf_download::download_hf_model(app_for_task.clone(), req).await {
            Ok(path) => {
                let _ = app_for_task.emit(
                    hf_download::HF_DOWNLOAD_DONE_EVENT,
                    hf_download::HfDownloadDone {
                        path: path.display().to_string(),
                    },
                );
            }
            Err(err) => {
                let _ = app_for_task.emit(
                    hf_download::HF_DOWNLOAD_ERROR_EVENT,
                    hf_download::HfDownloadError { error: err },
                );
            }
        }
    });
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

#[derive(serde::Serialize)]
struct KilnUpdateCheckResult {
    current: Option<String>,
    latest: Option<String>,
    update_available: bool,
    platform_supported: bool,
    /// `Some(sm)` when the local GPU's SM arch is NOT in
    /// `installer::SUPPORTED_SM_ARCHS` — the UI should surface a warn
    /// pill and disable the Download button. `None` when supported,
    /// unknown (no nvidia-smi, detection failed), or irrelevant (macOS).
    /// See `desktop/docs/binary-update.md` (CUDA / GPU compat).
    gpu_unsupported: Option<u32>,
}

/// Detection-only check for a newer kiln binary release. Looks up the
/// currently-installed version by invoking `kiln --version`, queries the
/// latest `kiln-v*` GitHub release for this platform, and compares via
/// semver. No download or swap here — that lands in slice 2 of
/// `desktop/docs/binary-update.md`.
#[tauri::command]
async fn check_for_kiln_update(
    state: State<'_, SettingsState>,
) -> Result<KilnUpdateCheckResult, String> {
    let configured = {
        let s = state.read().await;
        s.kiln_binary
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("kiln"))
    };
    let current = match installer::resolve_binary(&configured) {
        Some(p) => installer::current_kiln_version(&p).await,
        None => None,
    };

    let platform_supported = installer::supports_auto_install();
    let latest = if platform_supported {
        let client = reqwest::Client::builder()
            .user_agent(concat!("kiln-desktop/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| format!("build reqwest client: {}", e))?;
        installer::discover_latest_version(&client).await
    } else {
        None
    };

    let mut update_available = match &latest {
        Some(l) => installer::is_update_available(current.as_deref(), l),
        None => false,
    };

    // GPU arch compat gate (slice 6): if nvidia-smi reports a compute
    // capability outside `installer::SUPPORTED_SM_ARCHS`, refuse to offer
    // the update. Unknown detection (no nvidia-smi, timeout, etc.) is
    // NOT a block — preserve existing behavior on systems without
    // nvidia-smi. See `desktop/docs/binary-update.md` (CUDA / GPU compat).
    let gpu_unsupported = match installer::gpu_compat().await {
        installer::GpuCompat::Unsupported(sm) => {
            update_available = false;
            Some(sm)
        }
        installer::GpuCompat::Supported(_) | installer::GpuCompat::Unknown => None,
    };

    Ok(KilnUpdateCheckResult {
        current,
        latest,
        update_available,
        platform_supported,
        gpu_unsupported,
    })
}

/// Event emitted when the launch-time check discovers a newer kiln binary.
/// Payload matches `KilnUpdateAvailablePayload` (current + latest version).
const KILN_UPDATE_AVAILABLE_EVENT: &str = "kiln-update-available";

#[derive(serde::Serialize, Clone)]
struct KilnUpdateAvailablePayload {
    current: Option<String>,
    latest: String,
}

/// Launch-time auto-check for a newer kiln binary. Non-blocking, fails
/// silently on any error, never panics. Runs once per launch: no periodic
/// polling (per `desktop/docs/binary-update.md` open question #4 — stay
/// within the 60/hr anonymous GitHub rate limit). Emits
/// `kiln-update-available` so the dashboard can surface a banner and
/// settings can pre-populate the status pill.
async fn check_kiln_update_on_launch(app: AppHandle, settings: SettingsState) {
    // Give the supervisor ~10s to stabilize before we make a network call.
    // Keeps the launch path fast and avoids racing with auto_start.
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    if !installer::supports_auto_install() {
        return;
    }

    let configured = {
        let s = settings.read().await;
        s.kiln_binary
            .clone()
            .unwrap_or_else(|| std::path::PathBuf::from("kiln"))
    };
    let current = match installer::resolve_binary(&configured) {
        Some(p) => installer::current_kiln_version(&p).await,
        None => None,
    };

    let client = match reqwest::Client::builder()
        .user_agent(concat!("kiln-desktop/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[main] auto_update check: build reqwest client failed: {}", e);
            return;
        }
    };

    let Some(latest) = installer::discover_latest_version(&client).await else {
        return;
    };

    if !installer::is_update_available(current.as_deref(), &latest) {
        return;
    }

    // GPU arch compat gate (slice 6): silently skip the banner when the
    // local GPU's SM arch is outside `installer::SUPPORTED_SM_ARCHS`.
    // Unknown detection is treated as supported — the check_for_kiln_update
    // command path still surfaces the compat state to the settings UI when
    // the user clicks "Check for Updates".
    if let installer::GpuCompat::Unsupported(sm) = installer::gpu_compat().await {
        eprintln!(
            "[main] auto_update: skipping — GPU SM {} not supported",
            sm
        );
        return;
    }

    if let Err(e) = app.emit(
        KILN_UPDATE_AVAILABLE_EVENT,
        KilnUpdateAvailablePayload {
            current: current.clone(),
            latest: latest.clone(),
        },
    ) {
        eprintln!("[main] emit kiln-update-available failed: {}", e);
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

            // Non-blocking auto-check for a newer kiln binary. Runs once per
            // launch after a short delay so the supervisor can stabilize.
            // Emits `kiln-update-available` on success; failures are logged
            // and swallowed so a flaky GitHub API call never blocks startup.
            // Per `desktop/docs/binary-update.md` this is launch-only — no
            // periodic polling, to stay under the anonymous GitHub rate limit.
            let app_for_update = handle.clone();
            let settings_for_update: SettingsState = Arc::clone(&settings_state);
            tauri::async_runtime::spawn(async move {
                check_kiln_update_on_launch(app_for_update, settings_for_update).await;
            });

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
            check_for_kiln_update,
            get_binary_status,
            download_kiln_server,
            cancel_kiln_download,
            download_kiln_update,
            fetch_update_sha256,
            extract_update,
            installer::install_staged_update,
            download_hf_model,
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
