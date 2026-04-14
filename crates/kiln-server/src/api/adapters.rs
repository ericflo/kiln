use std::path::Path;

use axum::extract::{State, Path as AxumPath};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use kiln_model::lora_loader::LoraWeights;

use crate::error::ApiError;
use crate::state::{AppState, ModelBackend};

/// Response for GET /v1/adapters.
#[derive(Serialize)]
struct AdaptersResponse {
    /// Name of the currently active adapter, if any.
    active: Option<String>,
    /// Adapters available on disk in the adapter directory.
    available: Vec<AdapterDiskEntry>,
}

/// An adapter directory found on disk.
#[derive(Serialize)]
struct AdapterDiskEntry {
    /// Directory name (used as the adapter name).
    name: String,
    /// Whether this adapter has adapter_config.json.
    has_config: bool,
    /// Whether this adapter has adapter_model.safetensors.
    has_weights: bool,
    /// Total size of all files in the adapter directory in bytes.
    size_bytes: u64,
    /// ISO 8601 timestamp of the most recently modified file.
    modified_at: Option<String>,
    /// List of filenames in the adapter directory.
    files: Vec<String>,
}

/// Request body for POST /v1/adapters/load.
#[derive(Deserialize)]
struct LoadAdapterRequest {
    /// Adapter name (subdirectory under adapter_dir) or absolute path.
    name: String,
}

/// Response for POST /v1/adapters/load.
#[derive(Serialize)]
struct LoadAdapterResponse {
    status: &'static str,
    name: String,
}

/// Response for POST /v1/adapters/unload.
#[derive(Serialize)]
struct UnloadAdapterResponse {
    status: &'static str,
}

/// Response for DELETE /v1/adapters/:name.
#[derive(Serialize)]
struct DeleteAdapterResponse {
    status: &'static str,
    name: String,
}

/// List loaded and available adapters.
async fn list_adapters(State(state): State<AppState>) -> Json<AdaptersResponse> {
    // Read the active adapter name from shared state.
    let active = state.active_adapter_name.read().unwrap().clone();

    // Scan adapter directory for available adapters
    let available = scan_adapter_dir(&state.adapter_dir);

    Json(AdaptersResponse { active, available })
}

/// Load a LoRA adapter from disk.
async fn load_adapter(
    State(state): State<AppState>,
    Json(req): Json<LoadAdapterRequest>,
) -> Result<Json<LoadAdapterResponse>, ApiError> {
    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner,
        ModelBackend::Mock { .. } => {
            return Err(ApiError::mock_mode_no_adapters());
        }
    };

    // Resolve adapter path: if name is absolute, use it directly; otherwise look in adapter_dir.
    let adapter_path = if Path::new(&req.name).is_absolute() {
        std::path::PathBuf::from(&req.name)
    } else {
        state.adapter_dir.join(&req.name)
    };

    if !adapter_path.exists() {
        return Err(ApiError::adapter_not_found(&req.name));
    }

    // Two-phase load: read device/num_layers under a brief read lock, then load
    // weights outside any lock so inference is not blocked during I/O.
    let (device, num_layers) = {
        let guard = runner.read().unwrap();
        (guard.weights.embed_tokens.device().clone(), guard.config.num_layers)
    };

    let runner = runner.clone();
    let path = adapter_path.clone();
    let name = req.name.clone();

    tokio::task::spawn_blocking(move || {
        // Load weights outside any lock (I/O + tensor allocation).
        let lora = LoraWeights::load(&path, num_layers, &device)
            .map_err(|e| format!("{e}"))?;
        // Brief write lock to swap the adapter in.
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?
    .map_err(ApiError::adapter_load_failed)?;

    // Update the shared active adapter name.
    *state.active_adapter_name.write().unwrap() = Some(req.name.clone());

    tracing::info!(adapter = %req.name, path = %adapter_path.display(), operation = "load", "loaded LoRA adapter");

    Ok(Json(LoadAdapterResponse {
        status: "loaded",
        name,
    }))
}

/// Unload the active LoRA adapter, reverting to base model.
async fn unload_adapter(
    State(state): State<AppState>,
) -> Result<Json<UnloadAdapterResponse>, ApiError> {
    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner,
        ModelBackend::Mock { .. } => {
            return Err(ApiError::mock_mode_no_adapters());
        }
    };

    let runner = runner.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = runner.write().unwrap();
        guard.unload_adapter();
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?;

    // Clear the shared active adapter name.
    *state.active_adapter_name.write().unwrap() = None;

    tracing::info!(operation = "unload", "unloaded LoRA adapter — reverted to base model");

    Ok(Json(UnloadAdapterResponse {
        status: "unloaded",
    }))
}

/// Delete an adapter from disk.
async fn delete_adapter(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<DeleteAdapterResponse>, ApiError> {
    let adapter_path = state.adapter_dir.join(&name);

    if !adapter_path.exists() || !adapter_path.is_dir() {
        return Err(ApiError::adapter_not_found(&name));
    }

    // Check if this adapter is currently active.
    let active = state.active_adapter_name.read().unwrap().clone();
    if active.as_deref() == Some(&name) {
        return Err(ApiError::adapter_active(&name));
    }

    std::fs::remove_dir_all(&adapter_path).map_err(|e| ApiError::adapter_delete_failed(e))?;

    // Clean up any checkpoint directories (e.g. "my-adapter-checkpoint-50")
    let checkpoint_prefix = format!("{name}-checkpoint-");
    if let Ok(entries) = std::fs::read_dir(&state.adapter_dir) {
        for entry in entries.flatten() {
            let entry_name = entry.file_name().to_string_lossy().to_string();
            if entry_name.starts_with(&checkpoint_prefix) && entry.path().is_dir() {
                if let Err(e) = std::fs::remove_dir_all(entry.path()) {
                    tracing::warn!(checkpoint = %entry_name, error = %e, "failed to delete checkpoint directory");
                } else {
                    tracing::info!(checkpoint = %entry_name, "deleted checkpoint directory");
                }
            }
        }
    }

    tracing::info!(adapter = %name, operation = "delete", "deleted adapter from disk");

    Ok(Json(DeleteAdapterResponse {
        status: "deleted",
        name,
    }))
}

/// Scan a directory for adapter subdirectories.
fn scan_adapter_dir(dir: &Path) -> Vec<AdapterDiskEntry> {
    let mut entries = Vec::new();
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return entries,
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            let has_config = path.join("adapter_config.json").exists();
            let has_weights = path.join("adapter_model.safetensors").exists();
            if has_config || has_weights {
                let mut size_bytes: u64 = 0;
                let mut latest_modified: Option<std::time::SystemTime> = None;
                let mut files = Vec::new();

                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub in sub_entries.flatten() {
                        if let Ok(meta) = sub.metadata() {
                            if meta.is_file() {
                                files.push(sub.file_name().to_string_lossy().to_string());
                                size_bytes += meta.len();
                                if let Ok(modified) = meta.modified() {
                                    latest_modified = Some(match latest_modified {
                                        Some(prev) if prev >= modified => prev,
                                        _ => modified,
                                    });
                                }
                            }
                        }
                    }
                }
                files.sort();

                let modified_at = latest_modified.map(|t| {
                    let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
                    let secs = d.as_secs();
                    // Format as ISO 8601 UTC manually (no chrono dependency needed).
                    let days_since_epoch = secs / 86400;
                    let time_of_day = secs % 86400;
                    let hours = time_of_day / 3600;
                    let minutes = (time_of_day % 3600) / 60;
                    let seconds = time_of_day % 60;
                    // Compute year/month/day from days since 1970-01-01.
                    let (year, month, day) = days_to_ymd(days_since_epoch);
                    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
                });

                entries.push(AdapterDiskEntry {
                    name,
                    has_config,
                    has_weights,
                    size_bytes,
                    modified_at,
                    files,
                });
            }
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm from Howard Hinnant.
    days += 719_468;
    let era = days / 146_097;
    let doe = days % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/adapters", get(list_adapters))
        .route("/v1/adapters/load", post(load_adapter))
        .route("/v1/adapters/unload", post(unload_adapter))
        .route("/v1/adapters/{name}", delete(delete_adapter))
}
