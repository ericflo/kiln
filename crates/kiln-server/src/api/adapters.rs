use std::path::Path;

use axum::extract::{State, Path as AxumPath};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use kiln_model::adapter_merge::{merge_linear, PeftLora};
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

    // Update the shared active adapter name and drop stale real-backend prefix state.
    *state.active_adapter_name.write().unwrap() = Some(req.name.clone());
    state.clear_real_prefix_cache();

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

    // Clear the shared active adapter name and drop stale real-backend prefix state.
    *state.active_adapter_name.write().unwrap() = None;
    state.clear_real_prefix_cache();

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

/// Source adapter to include in a merge.
#[derive(Deserialize)]
struct MergeSource {
    /// Adapter name (subdirectory under adapter_dir).
    name: String,
    /// Weight applied to this adapter's tensors in the linear interpolation.
    weight: f32,
}

/// Request body for POST /v1/adapters/merge.
#[derive(Deserialize)]
struct MergeAdapterRequest {
    /// Source adapters and their interpolation weights. Must contain at least
    /// two entries to be a true merge (a single source is allowed and behaves
    /// as a copy with optional rescaling).
    sources: Vec<MergeSource>,
    /// Name of the output adapter (subdirectory created under adapter_dir).
    output_name: String,
    /// Merge mode. Currently only "weighted_average" (the default) is
    /// supported. TIES and concatenation will arrive in follow-up PRs.
    #[serde(default)]
    mode: Option<String>,
}

/// Source summary echoed back in the merge response.
#[derive(Serialize)]
struct MergeSourceInfo {
    name: String,
    weight: f32,
}

/// Response body for POST /v1/adapters/merge.
#[derive(Serialize)]
struct MergeAdapterResponse {
    status: &'static str,
    output_name: String,
    mode: &'static str,
    sources: Vec<MergeSourceInfo>,
    /// Number of tensors in the merged adapter.
    num_tensors: usize,
}

/// Merge multiple LoRA adapters via linear interpolation.
///
/// For each tensor key shared across the input adapters, computes the
/// weighted sum `Σᵢ wᵢ · tensor_i` and writes the result to a new adapter
/// directory in `adapter_dir`. Source adapters must have identical rank,
/// target_modules, base model, and tensor shapes.
async fn merge_adapters(
    State(state): State<AppState>,
    Json(req): Json<MergeAdapterRequest>,
) -> Result<Json<MergeAdapterResponse>, ApiError> {
    // Validate mode (default = weighted_average).
    let mode = req.mode.as_deref().unwrap_or("weighted_average");
    if mode != "weighted_average" {
        return Err(ApiError::adapter_merge_invalid(format!(
            "unsupported merge mode '{mode}' — only 'weighted_average' is supported in v1"
        )));
    }

    // Need at least one source.
    if req.sources.is_empty() {
        return Err(ApiError::adapter_merge_invalid(
            "sources must contain at least one entry",
        ));
    }

    // Validate output_name: no path separators, not "." or "..", non-empty.
    let output_name = req.output_name.trim().to_string();
    if output_name.is_empty()
        || output_name == "."
        || output_name == ".."
        || output_name.contains('/')
        || output_name.contains('\\')
    {
        return Err(ApiError::adapter_merge_bad_name(&output_name));
    }

    // Resolve and confirm all source adapter directories exist before
    // doing any I/O work.
    let mut source_paths: Vec<(String, f32, std::path::PathBuf)> =
        Vec::with_capacity(req.sources.len());
    for src in &req.sources {
        let path = state.adapter_dir.join(&src.name);
        if !path.exists() || !path.is_dir() {
            return Err(ApiError::adapter_not_found(&src.name));
        }
        source_paths.push((src.name.clone(), src.weight, path));
    }

    // Refuse to overwrite an existing output adapter.
    let output_path = state.adapter_dir.join(&output_name);
    if output_path.exists() {
        return Err(ApiError::adapter_merge_output_exists(&output_name));
    }

    let sources_info: Vec<MergeSourceInfo> = req
        .sources
        .iter()
        .map(|s| MergeSourceInfo {
            name: s.name.clone(),
            weight: s.weight,
        })
        .collect();

    // Run the (potentially slow, CPU-bound) merge work on a blocking thread.
    let output_name_for_task = output_name.clone();
    let output_path_for_task = output_path.clone();
    let merge_result = tokio::task::spawn_blocking(move || -> Result<usize, MergeError> {
        // Load each source adapter from disk.
        let mut loaded: Vec<(PeftLora, f32)> = Vec::with_capacity(source_paths.len());
        for (name, weight, path) in source_paths {
            let adapter = PeftLora::load(&path).map_err(|e| MergeError::Failed(format!(
                "loading source adapter '{name}' from {}: {e}",
                path.display()
            )))?;
            loaded.push((adapter, weight));
        }

        let refs: Vec<(&PeftLora, f32)> =
            loaded.iter().map(|(a, w)| (a, *w)).collect();

        let merged = merge_linear(&refs).map_err(|e| MergeError::Invalid(format!("{e}")))?;
        let num_tensors = merged.tensors.len();
        merged
            .save(&output_path_for_task)
            .map_err(|e| MergeError::Failed(format!(
                "saving merged adapter to {}: {e}",
                output_path_for_task.display()
            )))?;
        let _ = output_name_for_task;
        Ok(num_tensors)
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?;

    let num_tensors = match merge_result {
        Ok(n) => n,
        Err(MergeError::Invalid(msg)) => {
            // Best-effort cleanup if we partially wrote anything.
            let _ = std::fs::remove_dir_all(&output_path);
            return Err(ApiError::adapter_merge_invalid(msg));
        }
        Err(MergeError::Failed(msg)) => {
            let _ = std::fs::remove_dir_all(&output_path);
            return Err(ApiError::adapter_merge_failed(msg));
        }
    };

    tracing::info!(
        output = %output_name,
        num_sources = req.sources.len(),
        num_tensors,
        mode,
        operation = "merge",
        "merged LoRA adapters"
    );

    Ok(Json(MergeAdapterResponse {
        status: "merged",
        output_name,
        mode: "weighted_average",
        sources: sources_info,
        num_tensors,
    }))
}

/// Internal error type for the blocking merge task — distinguishes user
/// validation failures (400) from internal I/O failures (500).
enum MergeError {
    Invalid(String),
    Failed(String),
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
        .route("/v1/adapters/merge", post(merge_adapters))
        .route("/v1/adapters/{name}", delete(delete_adapter))
}
