use std::io::Write;
use std::path::{Component, Path, PathBuf};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use kiln_model::adapter_merge::{PeftLora, merge_concat, merge_linear, merge_ties};
use kiln_model::lora_loader::LoraWeights;

use crate::error::ApiError;
use crate::state::{AppState, ModelBackend};

/// Response for GET /v1/adapters.
#[derive(Serialize)]
struct AdaptersResponse {
    /// Name of the current server default adapter, if any.
    active_adapter: Option<String>,
    /// Name of the currently active adapter, if any.
    active: Option<String>,
    /// Adapter currently loaded into the model runner, if any.
    loaded_adapter: Option<String>,
    /// All adapters currently resident in server state. Kiln has one runner,
    /// so this is at most the default and runtime-loaded adapter names.
    loaded_adapters: Vec<String>,
    /// Absolute adapter registry directory path.
    adapter_dir: String,
    /// Adapters available on disk in the adapter directory.
    available: Vec<AdapterDiskEntry>,
    /// Expanded registry entries, including invalid directories.
    available_adapters: Vec<AdapterDiskEntry>,
}

/// An adapter directory found on disk.
#[derive(Clone, Serialize)]
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
    /// Resolved absolute directory path when available.
    path: Option<String>,
    /// Registry status: `loaded`, `available`, `quarantined`, or `invalid`.
    status: String,
    /// SHA-256 of adapter_model.safetensors when present.
    adapter_model_sha256: Option<String>,
    /// Size of adapter_model.safetensors when present.
    adapter_model_size_bytes: Option<u64>,
    /// LoRA rank from adapter_config.json.
    rank: Option<u64>,
    /// LoRA alpha from adapter_config.json.
    alpha: Option<f64>,
    /// LoRA alpha/rank from adapter_config.json.
    alpha_over_rank: Option<f64>,
    /// Target modules from adapter_config.json.
    target_modules: Vec<String>,
    /// PEFT base model metadata when present.
    base_model_name_or_path: Option<String>,
    /// Parent/base adapter lineage metadata when present.
    parent_adapter_metadata: Option<serde_json::Value>,
    /// Canary status read from adapter_canary_status.json or train_receipt.json.
    canary_status: String,
    /// Human-readable failure reason when the canary quarantined this adapter.
    canary_failure_reason: Option<String>,
    /// Canary warnings copied from the training receipt.
    canary_warnings: Vec<String>,
    /// Individual canary checks and their pass/fail state.
    canary_checks: Vec<kiln_train::AdapterCanaryCheckReceipt>,
    /// Resolved path for the canary status source when available.
    canary_status_path: Option<String>,
    /// Parsed adapter_manifest.json when present.
    adapter_manifest: Option<kiln_train::AdapterManifest>,
    /// Resolved path for adapter_manifest.json when present.
    adapter_manifest_path: Option<String>,
    /// Manifest parse/read error when adapter_manifest.json is present but invalid.
    adapter_manifest_error: Option<String>,
    /// Last load error recorded for this adapter name.
    last_load_error: Option<String>,
    /// Validation error for invalid adapter directories.
    error: Option<String>,
}

/// Request body for POST /v1/adapters/load.
#[derive(Deserialize)]
struct LoadAdapterRequest {
    /// Adapter name (subdirectory under adapter_dir) or absolute path.
    name: String,
    /// Allow loading an adapter whose canary status is quarantined.
    #[serde(default)]
    allow_quarantined: bool,
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

/// List saved/available adapters and identify the active adapter.
async fn list_adapters(State(state): State<AppState>) -> Json<AdaptersResponse> {
    // Read the active adapter name from shared state.
    let active = state.active_adapter_name.read().unwrap().clone();
    let loaded_adapter = state.loaded_adapter_name.read().unwrap().clone();
    let load_errors = state.adapter_load_errors.read().unwrap().clone();

    let mut loaded_adapters = Vec::new();
    if let Some(name) = active.clone() {
        loaded_adapters.push(name);
    }
    if let Some(name) = loaded_adapter.clone()
        && !loaded_adapters.iter().any(|existing| existing == &name)
    {
        loaded_adapters.push(name);
    }
    // Scan adapter directory for available adapters
    let available = scan_adapter_dir(&state.adapter_dir, &loaded_adapters, &load_errors);
    let adapter_dir = state
        .adapter_dir
        .canonicalize()
        .unwrap_or_else(|_| state.adapter_dir.clone())
        .display()
        .to_string();

    Json(AdaptersResponse {
        active_adapter: active.clone(),
        active,
        loaded_adapter,
        loaded_adapters,
        adapter_dir,
        available: available.clone(),
        available_adapters: available,
    })
}

/// Load a LoRA adapter from disk.
async fn load_adapter(
    State(state): State<AppState>,
    Json(req): Json<LoadAdapterRequest>,
) -> Result<Json<LoadAdapterResponse>, ApiError> {
    validate_adapter_name(&req.name)?;

    let adapter_path = state.adapter_dir.join(&req.name);
    let resolved_adapter_path = validate_loadable_adapter_dir(&adapter_path).map_err(|err| {
        state
            .adapter_load_errors
            .write()
            .unwrap()
            .insert(req.name.clone(), err.message.clone());
        tracing::warn!(
            adapter = %req.name,
            path = %adapter_path.display(),
            reason = %err.message,
            operation = "load",
            "adapter load rejected"
        );
        err
    })?;

    let canary = read_adapter_canary_metadata(&resolved_adapter_path);
    if canary.is_quarantined() && !req.allow_quarantined {
        let reason = canary
            .failure_reason
            .clone()
            .unwrap_or_else(|| "adapter canary status is quarantined".to_string());
        state
            .adapter_load_errors
            .write()
            .unwrap()
            .insert(req.name.clone(), reason.clone());
        tracing::warn!(
            adapter = %req.name,
            path = %resolved_adapter_path.display(),
            reason = %reason,
            operation = "load",
            "adapter load rejected because canary status is quarantined"
        );
        return Err(ApiError::adapter_quarantined(&req.name, reason));
    }

    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner,
        ModelBackend::Mock { .. } => {
            #[cfg(test)]
            {
                record_adapter_loaded(&state, &req.name, &resolved_adapter_path);
                return Ok(Json(LoadAdapterResponse {
                    status: "loaded",
                    name: req.name,
                }));
            }
            #[cfg(not(test))]
            {
                return Err(ApiError::mock_mode_no_adapters());
            }
        }
    };

    // Two-phase load: read device/num_layers under a brief read lock, then load
    // weights outside any lock so inference is not blocked during I/O.
    let (device, num_layers) = {
        let guard = runner.read().unwrap();
        (
            guard.weights.embed_tokens.device().clone(),
            guard.config.num_layers,
        )
    };

    let runner = runner.clone();
    let path = resolved_adapter_path.clone();
    let name = req.name.clone();

    let load_result = tokio::task::spawn_blocking(move || {
        // Load weights outside any lock (I/O + tensor allocation).
        let lora = LoraWeights::load(&path, num_layers, &device).map_err(|e| format!("{e}"))?;
        // Brief write lock to swap the adapter in.
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?;
    if let Err(err) = load_result {
        state
            .adapter_load_errors
            .write()
            .unwrap()
            .insert(req.name.clone(), err.clone());
        tracing::warn!(
            adapter = %req.name,
            path = %resolved_adapter_path.display(),
            reason = %err,
            operation = "load",
            "adapter load rejected"
        );
        return Err(ApiError::adapter_load_failed(err));
    }

    record_adapter_loaded(&state, &req.name, &resolved_adapter_path);

    Ok(Json(LoadAdapterResponse {
        status: "loaded",
        name,
    }))
}

fn record_adapter_loaded(state: &AppState, name: &str, path: &Path) {
    let old = state.active_adapter_name.read().unwrap().clone();
    *state.active_adapter_name.write().unwrap() = Some(name.to_string());
    *state.loaded_adapter_name.write().unwrap() = Some(name.to_string());
    state.adapter_load_errors.write().unwrap().remove(name);
    state.clear_real_prefix_cache();

    tracing::info!(
        request_id = %Uuid::new_v4(),
        old_adapter = ?old,
        new_adapter = ?Some(name.to_string()),
        reason = "adapter_load_endpoint",
        path = %path.display(),
        operation = "load",
        "adapter transition"
    );
}

fn validate_loadable_adapter_dir(adapter_path: &Path) -> Result<PathBuf, ApiError> {
    if !adapter_path.exists() || !adapter_path.is_dir() {
        return Err(ApiError::adapter_not_found(adapter_path.display()));
    }

    let resolved = adapter_path.canonicalize().map_err(|e| {
        ApiError::adapter_layout_invalid(format!(
            "failed to resolve absolute path {}: {e}",
            adapter_path.display()
        ))
    })?;

    let required = ["adapter_config.json", "adapter_model.safetensors"];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|file| !resolved.join(file).is_file())
        .collect();

    if missing.is_empty() {
        return Ok(resolved);
    }

    let mut detail = format!(
        "path {} is missing required file(s): {}",
        resolved.display(),
        missing.join(", ")
    );

    if let Some(child) = find_single_nested_adapter_dir(&resolved) {
        detail.push_str(&format!(
            "; found nested adapter directory at {}",
            child.display()
        ));
    }

    Err(ApiError::adapter_layout_invalid(detail))
}

fn find_single_nested_adapter_dir(parent: &Path) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(parent).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir()
            && path.join("adapter_config.json").is_file()
            && path.join("adapter_model.safetensors").is_file()
        {
            matches.push(path);
        }
    }
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

/// Unload the active LoRA adapter, reverting to base model.
async fn unload_adapter(
    State(state): State<AppState>,
) -> Result<Json<UnloadAdapterResponse>, ApiError> {
    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner,
        ModelBackend::Mock { .. } => {
            #[cfg(test)]
            {
                record_adapter_unloaded(&state);
                return Ok(Json(UnloadAdapterResponse { status: "unloaded" }));
            }
            #[cfg(not(test))]
            {
                return Err(ApiError::mock_mode_no_adapters());
            }
        }
    };

    let runner = runner.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = runner.write().unwrap();
        guard.unload_adapter();
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?;

    record_adapter_unloaded(&state);

    Ok(Json(UnloadAdapterResponse { status: "unloaded" }))
}

fn record_adapter_unloaded(state: &AppState) {
    let old = state.active_adapter_name.read().unwrap().clone();
    *state.active_adapter_name.write().unwrap() = None;
    *state.loaded_adapter_name.write().unwrap() = None;
    state.clear_real_prefix_cache();

    tracing::info!(
        request_id = %Uuid::new_v4(),
        old_adapter = ?old,
        new_adapter = ?Option::<String>::None,
        reason = "adapter_unload_endpoint",
        operation = "unload",
        "adapter transition"
    );
}

/// Delete an adapter from disk.
async fn delete_adapter(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<DeleteAdapterResponse>, ApiError> {
    validate_adapter_name(&name)?;
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
fn scan_adapter_dir(
    dir: &Path,
    loaded_adapters: &[String],
    load_errors: &std::collections::HashMap<String, String>,
) -> Vec<AdapterDiskEntry> {
    let mut entries = Vec::new();
    let read_dir = match std::fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return entries,
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".composed" || name.starts_with(".upload-tmp-") {
                continue;
            }
            let has_config = path.join("adapter_config.json").exists();
            let has_weights = path.join("adapter_model.safetensors").exists();

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

            let path = path.canonicalize().unwrap_or(path);
            let config = read_adapter_config_metadata(&path);
            let canary = read_adapter_canary_metadata(&path);
            let manifest = read_adapter_manifest_metadata(&path);
            let weights_path = path.join("adapter_model.safetensors");
            let adapter_model_size_bytes = std::fs::metadata(&weights_path).ok().map(|m| m.len());
            let adapter_model_sha256 = if has_weights {
                sha256_file_hex(&weights_path).ok()
            } else {
                None
            };
            let error = validate_loadable_adapter_dir(&path)
                .err()
                .map(|err| err.message);
            let status = if error.is_some() {
                "invalid"
            } else if canary.is_quarantined() {
                "quarantined"
            } else if loaded_adapters.iter().any(|loaded| loaded == &name) {
                "loaded"
            } else {
                "available"
            }
            .to_string();

            entries.push(AdapterDiskEntry {
                name: name.clone(),
                has_config,
                has_weights,
                size_bytes,
                modified_at,
                files,
                path: Some(path.display().to_string()),
                status,
                adapter_model_sha256,
                adapter_model_size_bytes,
                rank: config.rank,
                alpha: config.alpha,
                alpha_over_rank: config.alpha_over_rank,
                target_modules: config.target_modules,
                base_model_name_or_path: config.base_model_name_or_path,
                parent_adapter_metadata: read_parent_adapter_metadata(&path),
                canary_status: canary.status,
                canary_failure_reason: canary.failure_reason,
                canary_warnings: canary.warnings,
                canary_checks: canary.checks,
                canary_status_path: canary.status_path,
                adapter_manifest: manifest.manifest,
                adapter_manifest_path: manifest.path,
                adapter_manifest_error: manifest.error,
                last_load_error: load_errors.get(&name).cloned(),
                error,
            });
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

#[derive(Default)]
struct AdapterConfigMetadata {
    rank: Option<u64>,
    alpha: Option<f64>,
    alpha_over_rank: Option<f64>,
    target_modules: Vec<String>,
    base_model_name_or_path: Option<String>,
}

struct AdapterCanaryMetadata {
    status: String,
    failure_reason: Option<String>,
    warnings: Vec<String>,
    checks: Vec<kiln_train::AdapterCanaryCheckReceipt>,
    status_path: Option<String>,
}

struct AdapterManifestMetadata {
    manifest: Option<kiln_train::AdapterManifest>,
    path: Option<String>,
    error: Option<String>,
}

impl AdapterCanaryMetadata {
    fn unknown() -> Self {
        Self {
            status: "unknown".to_string(),
            failure_reason: None,
            warnings: Vec::new(),
            checks: Vec::new(),
            status_path: None,
        }
    }

    fn is_quarantined(&self) -> bool {
        self.status == kiln_train::AdapterCanaryState::Quarantined.as_str()
    }
}

fn read_adapter_canary_metadata(path: &Path) -> AdapterCanaryMetadata {
    let status_path = adapter_canary_status_source_path(path);
    match kiln_train::read_adapter_canary_status_from_adapter_dir(path) {
        Ok(Some(status)) => AdapterCanaryMetadata {
            status: status.status.as_str().to_string(),
            failure_reason: status.failure_reason,
            warnings: status.warnings,
            checks: status.checks,
            status_path,
        },
        Ok(None) => AdapterCanaryMetadata::unknown(),
        Err(err) => AdapterCanaryMetadata {
            status: "unknown".to_string(),
            failure_reason: Some(format!("failed to read adapter canary status: {err:#}")),
            warnings: Vec::new(),
            checks: Vec::new(),
            status_path,
        },
    }
}

fn adapter_canary_status_source_path(path: &Path) -> Option<String> {
    let sidecar = path.join(kiln_train::ADAPTER_CANARY_STATUS_FILENAME);
    if sidecar.is_file() {
        return Some(sidecar.display().to_string());
    }
    let receipt = path.join(kiln_train::TRAIN_RECEIPT_FILENAME);
    if receipt.is_file() {
        return Some(receipt.display().to_string());
    }
    None
}

fn read_adapter_manifest_metadata(path: &Path) -> AdapterManifestMetadata {
    let manifest_path = path.join(kiln_train::ADAPTER_MANIFEST_FILENAME);
    if !manifest_path.is_file() {
        return AdapterManifestMetadata {
            manifest: None,
            path: None,
            error: None,
        };
    }
    match kiln_train::read_adapter_manifest(&manifest_path) {
        Ok(manifest) => AdapterManifestMetadata {
            manifest: Some(manifest),
            path: Some(manifest_path.display().to_string()),
            error: None,
        },
        Err(err) => AdapterManifestMetadata {
            manifest: None,
            path: Some(manifest_path.display().to_string()),
            error: Some(format!("{err:#}")),
        },
    }
}

fn read_adapter_config_metadata(path: &Path) -> AdapterConfigMetadata {
    let Ok(config_bytes) = std::fs::read(path.join("adapter_config.json")) else {
        return AdapterConfigMetadata::default();
    };
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(&config_bytes) else {
        return AdapterConfigMetadata::default();
    };
    let rank = config.get("r").and_then(|v| v.as_u64());
    let alpha = config.get("lora_alpha").and_then(|v| v.as_f64());
    let alpha_over_rank = match (alpha, rank) {
        (Some(alpha), Some(rank)) if rank != 0 => Some(alpha / rank as f64),
        _ => None,
    };
    let target_modules = config
        .get("target_modules")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let base_model_name_or_path = config
        .get("base_model_name_or_path")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    AdapterConfigMetadata {
        rank,
        alpha,
        alpha_over_rank,
        target_modules,
        base_model_name_or_path,
    }
}

fn read_parent_adapter_metadata(path: &Path) -> Option<serde_json::Value> {
    for filename in ["lineage.json", "adapter_receipt.json", "receipt.json"] {
        let value = std::fs::read(path.join(filename))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        if value.is_some() {
            return value;
        }
    }
    None
}

fn sha256_file_hex(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
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
    /// Merge mode. Supported values: `"weighted_average"` (the default),
    /// `"ties"` (Yadav et al. 2023), and `"concat"` (rank concatenation —
    /// produces a higher-rank adapter where `r_total = Σᵢ rᵢ`).
    #[serde(default)]
    mode: Option<String>,
    /// Density used by the TIES merge: keep the top fraction of values per
    /// adapter per tensor. Required range `(0.0, 1.0]`. Only valid with
    /// `mode == "ties"`. Defaults to 0.2 when `mode == "ties"`, following
    /// the TIES paper's recommendation.
    #[serde(default)]
    density: Option<f32>,
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

/// Merge multiple LoRA adapters via one of three modes.
///
/// - `weighted_average` (default) — elementwise weighted sum across
///   tensors. Source adapters must have identical rank, target_modules,
///   base model, and tensor shapes.
/// - `ties` — TIES merge (Yadav et al. 2023). Same shape requirements
///   as weighted_average; additionally accepts a `density` parameter
///   in `(0.0, 1.0]` that controls per-adapter trimming (default 0.2).
/// - `concat` — concatenate ranks. Produces a merged adapter with
///   `r_total = Σᵢ rᵢ`; ranks may differ across sources. `lora_A` is
///   row-stacked, `lora_B` is column-stacked with each block scaled by
///   its source weight, and `B_concat @ A_concat = Σᵢ wᵢ · (Bᵢ @ Aᵢ)`.
///
/// In every mode the result is written to a new adapter directory in
/// `adapter_dir`.
async fn merge_adapters(
    State(state): State<AppState>,
    Json(req): Json<MergeAdapterRequest>,
) -> Result<Json<MergeAdapterResponse>, ApiError> {
    // Validate mode (default = weighted_average).
    let mode_str = req.mode.as_deref().unwrap_or("weighted_average");
    let resolved_mode: &'static str = match mode_str {
        "weighted_average" => "weighted_average",
        "ties" => "ties",
        "concat" => "concat",
        other => {
            return Err(ApiError::adapter_merge_invalid(format!(
                "unsupported merge mode '{other}' — supported modes are 'weighted_average', 'ties', and 'concat'"
            )));
        }
    };

    // Validate TIES density up-front so callers get a clean 400 before
    // we touch any I/O.
    let ties_density: f32 = match resolved_mode {
        "ties" => {
            let d = req.density.unwrap_or(0.2);
            if !(d.is_finite() && d > 0.0 && d <= 1.0) {
                return Err(ApiError::adapter_merge_invalid(format!(
                    "density must be in (0.0, 1.0]; got {d}"
                )));
            }
            d
        }
        _ => {
            if req.density.is_some() {
                return Err(ApiError::adapter_merge_invalid(
                    "density is only supported when mode == \"ties\"",
                ));
            }
            // Unused for weighted_average.
            0.0
        }
    };

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
        validate_adapter_name(&src.name)?;
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
    let mode_for_task = resolved_mode;
    let merge_result = tokio::task::spawn_blocking(move || -> Result<usize, MergeError> {
        // Load each source adapter from disk.
        let mut loaded: Vec<(PeftLora, f32)> = Vec::with_capacity(source_paths.len());
        for (name, weight, path) in source_paths {
            let adapter = PeftLora::load(&path).map_err(|e| {
                MergeError::Failed(format!(
                    "loading source adapter '{name}' from {}: {e}",
                    path.display()
                ))
            })?;
            loaded.push((adapter, weight));
        }

        let refs: Vec<(&PeftLora, f32)> = loaded.iter().map(|(a, w)| (a, *w)).collect();

        let merged = match mode_for_task {
            "ties" => {
                merge_ties(&refs, ties_density).map_err(|e| MergeError::Invalid(format!("{e}")))?
            }
            "concat" => merge_concat(&refs).map_err(|e| MergeError::Invalid(format!("{e}")))?,
            _ => merge_linear(&refs).map_err(|e| MergeError::Invalid(format!("{e}")))?,
        };
        let num_tensors = merged.tensors.len();
        merged.save(&output_path_for_task).map_err(|e| {
            MergeError::Failed(format!(
                "saving merged adapter to {}: {e}",
                output_path_for_task.display()
            ))
        })?;
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
        mode = resolved_mode,
        operation = "merge",
        "merged LoRA adapters"
    );

    Ok(Json(MergeAdapterResponse {
        status: "merged",
        output_name,
        mode: resolved_mode,
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

/// Reject names that aren't safe single-segment identifiers. Read-only export
/// of managed adapters only — never absolute paths, never path traversal.
fn validate_adapter_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || Path::new(name).is_absolute()
    {
        return Err(ApiError::invalid_adapter_name(name));
    }
    Ok(())
}

/// Recursively append all regular files in `dir` to a tar builder. The first
/// path component in each entry name is the adapter's directory name, so
/// extracting the archive recreates the on-disk layout. Non-file entries
/// (symlinks, sockets, devices) are silently skipped to keep export robust.
fn append_dir_to_tar<W: std::io::Write>(
    tar: &mut tar::Builder<W>,
    dir: &Path,
    name_prefix: &Path,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let entry_path = entry.path();
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let entry_name = name_prefix.join(entry.file_name());
        if metadata.is_file() {
            let mut file = std::fs::File::open(&entry_path)?;
            tar.append_file(&entry_name, &mut file)?;
        } else if metadata.is_dir() {
            append_dir_to_tar(tar, &entry_path, &entry_name)?;
        }
        // Skip symlinks, devices, sockets, etc.
    }
    Ok(())
}

/// Stream a tar.gz of the adapter directory. The body is built on a blocking
/// thread and pushed through a bounded mpsc channel so the response streams to
/// the client without buffering the whole archive in memory.
async fn download_adapter(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Response, ApiError> {
    validate_adapter_name(&name)?;

    let adapter_path = state.adapter_dir.join(&name);
    if !adapter_path.exists() || !adapter_path.is_dir() {
        return Err(ApiError::adapter_not_found(&name));
    }

    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, std::io::Error>>(8);

    let adapter_path_for_task = adapter_path.clone();
    let name_for_task = name.clone();
    let tx_for_task = tx.clone();
    tokio::task::spawn_blocking(move || {
        struct ChannelWriter {
            tx: mpsc::Sender<Result<Vec<u8>, std::io::Error>>,
        }
        impl std::io::Write for ChannelWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let chunk = buf.to_vec();
                let len = chunk.len();
                self.tx.blocking_send(Ok(chunk)).map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "client disconnected")
                })?;
                Ok(len)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let writer = ChannelWriter {
            tx: tx_for_task.clone(),
        };
        let gz = GzEncoder::new(writer, Compression::default());
        let mut tar = tar::Builder::new(gz);

        let prefix = Path::new(&name_for_task);
        if let Err(e) = append_dir_to_tar(&mut tar, &adapter_path_for_task, prefix) {
            let _ = tx_for_task.blocking_send(Err(e));
            return;
        }
        match tar.into_inner() {
            Ok(gz) => {
                if let Err(e) = gz.finish() {
                    let _ = tx_for_task.blocking_send(Err(e));
                }
            }
            Err(e) => {
                let _ = tx_for_task.blocking_send(Err(e));
            }
        }
    });
    drop(tx);

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    let disposition = format!("attachment; filename=\"{name}.tar.gz\"");
    let disposition_value = HeaderValue::from_str(&disposition)
        .map_err(|e| ApiError::adapter_export_failed(format!("invalid filename header: {e}")))?;

    tracing::info!(adapter = %name, operation = "download", "streaming adapter tar.gz");

    Ok((
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/gzip"),
            ),
            (header::CONTENT_DISPOSITION, disposition_value),
        ],
        body,
    )
        .into_response())
}

/// Response for POST /v1/adapters/upload.
#[derive(Serialize)]
struct UploadAdapterResponse {
    /// Adapter name as installed on disk.
    name: String,
    /// Total bytes written across all extracted files.
    size_bytes: u64,
    /// Number of regular files extracted.
    files: u32,
}

/// Upper bound on uploaded archive size. Real Qwen3.5-4B rank-8 LoRA adapters
/// land at ~30-60 MB compressed; 2 GB leaves room for high-rank or multi-LoRA
/// archives without letting the body extractor become a DoS vector.
const ADAPTER_UPLOAD_BODY_LIMIT: usize = 2 * 1024 * 1024 * 1024;

/// Maximum total size of extracted bytes per upload. Mirrors the body limit so
/// a small archive cannot expand into a multi-terabyte tarbomb.
const ADAPTER_EXTRACT_BYTES_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

/// Maximum number of entries we'll extract from a single archive — guards
/// against a tar with millions of zero-byte files.
const ADAPTER_EXTRACT_ENTRIES_LIMIT: u32 = 100_000;

/// Sum the on-disk byte size of all finalized adapter directories under
/// `adapter_dir`. Skips the in-flight `.upload-tmp-*/` staging dirs and the
/// `.composed/<hash>/` cache (which is bounded separately by the §6/§8 LRU
/// eviction work — see roadmap item 8). A directory entry that fails to
/// stat is ignored rather than aborting the whole walk; this keeps the
/// guard available on a partially-corrupt adapter_dir.
fn measure_finalized_adapter_dir_bytes(adapter_dir: &Path) -> u64 {
    let read_dir = match std::fs::read_dir(adapter_dir) {
        Ok(rd) => rd,
        Err(_) => return 0,
    };
    let mut total: u64 = 0;
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        if name_lossy.starts_with(".upload-tmp-") || name_lossy == ".composed" {
            continue;
        }
        total = total.saturating_add(dir_size_recursive(&entry.path()));
    }
    total
}

/// Recursively sum the regular-file byte sizes under `root`. Symlinks and
/// errors are silently treated as zero (matching the conservative
/// best-effort spirit of `measure_finalized_adapter_dir_bytes`).
fn dir_size_recursive(root: &Path) -> u64 {
    let meta = match std::fs::symlink_metadata(root) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    if meta.file_type().is_file() {
        return meta.len();
    }
    if !meta.file_type().is_dir() {
        return 0;
    }
    let read_dir = match std::fs::read_dir(root) {
        Ok(rd) => rd,
        Err(_) => return 0,
    };
    let mut total: u64 = 0;
    for entry in read_dir.flatten() {
        total = total.saturating_add(dir_size_recursive(&entry.path()));
    }
    total
}

/// Receive a multipart/form-data archive and install it as a new adapter
/// directory under `adapter_dir`. Symmetric to `download_adapter`.
async fn upload_adapter(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<UploadAdapterResponse>, ApiError> {
    let mut name: Option<String> = None;
    let mut archive_path: Option<PathBuf> = None;
    let mut tmp_root: Option<PathBuf> = None;

    // Cleanup helper used on every error path to remove the temp dir we
    // allocated to buffer the upload bytes.
    let cleanup_tmp = |tmp_root: &Option<PathBuf>| {
        if let Some(root) = tmp_root.as_ref() {
            let _ = std::fs::remove_dir_all(root);
        }
    };

    while let Some(mut field) = multipart.next_field().await.map_err(|e| {
        cleanup_tmp(&tmp_root);
        ApiError::adapter_import_invalid(format!("multipart parse error: {e}"))
    })? {
        let field_name = match field.name() {
            Some(n) => n.to_string(),
            None => continue,
        };
        match field_name.as_str() {
            "name" => {
                let bytes = field.bytes().await.map_err(|e| {
                    cleanup_tmp(&tmp_root);
                    ApiError::adapter_import_invalid(format!("reading 'name' field: {e}"))
                })?;
                let s = std::str::from_utf8(&bytes)
                    .map_err(|_| {
                        cleanup_tmp(&tmp_root);
                        ApiError::adapter_import_invalid("'name' field must be UTF-8 text")
                    })?
                    .trim()
                    .to_string();
                if let Err(e) = validate_adapter_name(&s) {
                    cleanup_tmp(&tmp_root);
                    return Err(e);
                }
                name = Some(s);
            }
            "archive" => {
                // Lazily create a tmp dir under adapter_dir on the first
                // archive byte. Putting the temp file on the same filesystem
                // lets the eventual rename into place be atomic.
                if tmp_root.is_none() {
                    let suffix: u128 = rand_suffix();
                    let root = state.adapter_dir.join(format!(".upload-tmp-{suffix:032x}"));
                    std::fs::create_dir_all(&root).map_err(|e| {
                        ApiError::adapter_import_failed(format!(
                            "creating temp dir {}: {e}",
                            root.display()
                        ))
                    })?;
                    tmp_root = Some(root);
                }
                let tmp_dir = tmp_root.as_ref().unwrap();
                let archive_target = tmp_dir.join("archive.tar.gz");
                let mut file = std::fs::File::create(&archive_target).map_err(|e| {
                    cleanup_tmp(&tmp_root);
                    ApiError::adapter_import_failed(format!(
                        "creating temp archive {}: {e}",
                        archive_target.display()
                    ))
                })?;
                let mut total: u64 = 0;
                loop {
                    match field.chunk().await {
                        Ok(Some(chunk)) => {
                            total = total.saturating_add(chunk.len() as u64);
                            if total > ADAPTER_EXTRACT_BYTES_LIMIT {
                                cleanup_tmp(&tmp_root);
                                return Err(ApiError::adapter_import_invalid(format!(
                                    "archive exceeds {} GB byte limit",
                                    ADAPTER_EXTRACT_BYTES_LIMIT / (1024 * 1024 * 1024)
                                )));
                            }
                            if let Err(e) = file.write_all(&chunk) {
                                cleanup_tmp(&tmp_root);
                                return Err(ApiError::adapter_import_failed(format!(
                                    "writing temp archive: {e}"
                                )));
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            cleanup_tmp(&tmp_root);
                            return Err(ApiError::adapter_import_invalid(format!(
                                "reading 'archive' field: {e}"
                            )));
                        }
                    }
                }
                if let Err(e) = file.sync_all() {
                    cleanup_tmp(&tmp_root);
                    return Err(ApiError::adapter_import_failed(format!(
                        "flushing temp archive: {e}"
                    )));
                }
                archive_path = Some(archive_target);
            }
            _ => {
                // Unknown field — drain its body and ignore.
                let _ = field.bytes().await;
            }
        }
    }

    let name = match name {
        Some(n) => n,
        None => {
            cleanup_tmp(&tmp_root);
            return Err(ApiError::adapter_import_invalid("missing 'name' field"));
        }
    };
    let archive_path = match archive_path {
        Some(p) => p,
        None => {
            cleanup_tmp(&tmp_root);
            return Err(ApiError::adapter_import_invalid("missing 'archive' field"));
        }
    };

    let target_dir = state.adapter_dir.join(&name);
    if target_dir.exists() {
        cleanup_tmp(&tmp_root);
        return Err(ApiError::adapter_already_exists(&name));
    }

    // Extract on a blocking thread — tar/flate2 are sync, and decompression of
    // a real adapter pegs a CPU for ~hundreds of ms.
    let tmp_root_owned = tmp_root
        .clone()
        .expect("tmp_root set when archive_path is set");
    let archive_path_owned = archive_path.clone();
    let target_dir_owned = target_dir.clone();
    let adapter_dir_owned = state.adapter_dir.clone();
    let max_disk_bytes = state.adapter_max_disk_bytes;
    let extract_result = tokio::task::spawn_blocking(move || -> Result<(u64, u32), ImportError> {
        let staging = tmp_root_owned.join("staging");
        std::fs::create_dir_all(&staging)
            .map_err(|e| ImportError::Failed(format!("creating staging dir: {e}")))?;

        let file = std::fs::File::open(&archive_path_owned)
            .map_err(|e| ImportError::Failed(format!("opening archive: {e}")))?;
        let gz = GzDecoder::new(file);
        let mut tar = tar::Archive::new(gz);

        let entries = tar
            .entries()
            .map_err(|e| ImportError::Invalid(format!("reading tar entries: {e}")))?;

        let mut bytes_written: u64 = 0;
        let mut files_written: u32 = 0;
        let mut strip_prefix: Option<Option<String>> = None;

        let staging_canon = staging
            .canonicalize()
            .map_err(|e| ImportError::Failed(format!("canonicalizing staging: {e}")))?;

        for entry in entries {
            let mut entry =
                entry.map_err(|e| ImportError::Invalid(format!("reading tar entry: {e}")))?;

            files_written = files_written.saturating_add(1);
            if files_written > ADAPTER_EXTRACT_ENTRIES_LIMIT {
                return Err(ImportError::Invalid(format!(
                    "archive contains more than {ADAPTER_EXTRACT_ENTRIES_LIMIT} entries"
                )));
            }

            let entry_type = entry.header().entry_type();
            if entry_type.is_dir() {
                files_written = files_written.saturating_sub(1);
                continue;
            }
            // Reject non-regular files (symlinks, hard links, devices, fifos).
            if !entry_type.is_file() {
                return Err(ImportError::Invalid(format!(
                    "archive contains unsupported entry type {entry_type:?}"
                )));
            }

            let entry_path = entry
                .path()
                .map_err(|e| ImportError::Invalid(format!("decoding entry path: {e}")))?
                .into_owned();

            // Strip a single leading top-level directory if every entry shares
            // the same prefix (matches download's `name/<files>` layout).
            let stripped: PathBuf = {
                let mut comps = entry_path.components();
                let first = comps.next();
                match first {
                    Some(Component::Normal(first_seg)) => {
                        let first_str = first_seg.to_string_lossy().to_string();
                        let observed = strip_prefix.get_or_insert_with(|| Some(first_str.clone()));
                        match observed {
                            Some(prefix) if *prefix == first_str => {
                                let rest: PathBuf = comps.collect();
                                if rest.as_os_str().is_empty() {
                                    files_written = files_written.saturating_sub(1);
                                    continue;
                                }
                                rest
                            }
                            _ => {
                                // Mixed prefixes — install entries flat, no
                                // strip after the first inconsistency.
                                *observed = None;
                                entry_path.clone()
                            }
                        }
                    }
                    _ => entry_path.clone(),
                }
            };

            // Reject any path that is absolute, contains `..`, or resolves
            // outside the staging directory.
            for comp in stripped.components() {
                match comp {
                    Component::Normal(_) => {}
                    Component::CurDir => {}
                    _ => {
                        return Err(ImportError::Invalid(format!(
                            "archive entry has unsafe path: {}",
                            stripped.display()
                        )));
                    }
                }
            }
            if stripped.as_os_str().is_empty() {
                files_written = files_written.saturating_sub(1);
                continue;
            }

            let dest = staging.join(&stripped);
            // Ensure dest stays under the staging dir even after canonicalization.
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    ImportError::Failed(format!("creating dir {}: {e}", parent.display()))
                })?;
                let parent_canon = parent.canonicalize().map_err(|e| {
                    ImportError::Failed(format!("canonicalizing {}: {e}", parent.display()))
                })?;
                if !parent_canon.starts_with(&staging_canon) {
                    return Err(ImportError::Invalid(format!(
                        "archive entry escapes staging dir: {}",
                        stripped.display()
                    )));
                }
            }

            let size = entry.size();
            bytes_written = bytes_written.saturating_add(size);
            if bytes_written > ADAPTER_EXTRACT_BYTES_LIMIT {
                return Err(ImportError::Invalid(format!(
                    "decompressed archive exceeds {} GB limit",
                    ADAPTER_EXTRACT_BYTES_LIMIT / (1024 * 1024 * 1024)
                )));
            }

            let mut out = std::fs::File::create(&dest)
                .map_err(|e| ImportError::Failed(format!("creating {}: {e}", dest.display())))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| ImportError::Failed(format!("writing {}: {e}", dest.display())))?;
        }

        if files_written == 0 {
            return Err(ImportError::Invalid(
                "archive contained no regular files".to_string(),
            ));
        }

        // Disk-quota guard. We have the real extracted byte count now, and
        // the staging dir is still under `.upload-tmp-*/` so it is excluded
        // from `measure_finalized_adapter_dir_bytes`. Reject before rename so
        // a quota failure leaves no partial adapter behind.
        if let Some(cap) = max_disk_bytes {
            let current = measure_finalized_adapter_dir_bytes(&adapter_dir_owned);
            if current.saturating_add(bytes_written) > cap {
                return Err(ImportError::DiskQuota(format!(
                    "{} GiB used + {} GiB upload > {} GiB cap",
                    current as f64 / 1024.0 / 1024.0 / 1024.0,
                    bytes_written as f64 / 1024.0 / 1024.0 / 1024.0,
                    cap as f64 / 1024.0 / 1024.0 / 1024.0
                )));
            }
        }

        // Atomic rename of staging → target_dir. On the same filesystem this is
        // a single inode move; if it fails (e.g. EXDEV) fall back to a copy.
        std::fs::rename(&staging, &target_dir_owned).map_err(|e| {
            ImportError::Failed(format!(
                "renaming staging to {}: {e}",
                target_dir_owned.display()
            ))
        })?;

        Ok((bytes_written, files_written))
    })
    .await
    .map_err(|e| {
        cleanup_tmp(&tmp_root);
        ApiError::adapter_import_failed(format!("join error: {e}"))
    })?;

    // Always remove the upload temp dir; the staging child has been renamed
    // out by the success path or cleaned up on failure.
    cleanup_tmp(&tmp_root);

    let (size_bytes, files) = match extract_result {
        Ok(v) => v,
        Err(ImportError::Invalid(msg)) => {
            let _ = std::fs::remove_dir_all(&target_dir);
            return Err(ApiError::adapter_import_invalid(msg));
        }
        Err(ImportError::Failed(msg)) => {
            let _ = std::fs::remove_dir_all(&target_dir);
            return Err(ApiError::adapter_import_failed(msg));
        }
        Err(ImportError::DiskQuota(msg)) => {
            // The rejection happened before rename, so target_dir was never
            // created. The remove_dir_all is defensive in case a future change
            // ever creates target_dir earlier in the flow.
            let _ = std::fs::remove_dir_all(&target_dir);
            tracing::warn!(
                adapter = %name,
                detail = %msg,
                "rejected adapter upload: adapter_dir disk quota would be exceeded"
            );
            return Err(ApiError::adapter_disk_quota_exceeded(msg));
        }
    };

    tracing::info!(
        adapter = %name,
        size_bytes,
        files,
        operation = "upload",
        "imported adapter from upload"
    );

    Ok(Json(UploadAdapterResponse {
        name,
        size_bytes,
        files,
    }))
}

/// Internal error type for the blocking extract task — separates user input
/// problems (400) from server-side I/O failures (500) and quota rejections
/// (507).
enum ImportError {
    Invalid(String),
    Failed(String),
    DiskQuota(String),
}

/// Tiny stdlib-only random suffix so concurrent uploads don't collide.
fn rand_suffix() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id() as u128;
    let addr = &nanos as *const _ as u128;
    nanos ^ (pid << 64) ^ addr.rotate_left(33)
}

/// Provenance + audit summary for a single adapter, surfaced at
/// `GET /v1/adapters/:name/detail`. Used by the adapter drill-in modal
/// to show "what built this LoRA, what evals has it run against."
#[derive(Serialize)]
struct AdapterDetail {
    name: String,
    is_active: bool,
    has_config: bool,
    has_weights: bool,
    size_bytes: u64,
    files: Vec<AdapterFileEntry>,
    /// Training jobs from the in-memory tracking map whose `adapter_name`
    /// matches. May not include older jobs that have TTL'd out.
    training_jobs: Vec<AdapterLinkedJob>,
    /// Eval jobs whose `adapters` includes this adapter. Sorted newest first.
    eval_jobs: Vec<AdapterLinkedEval>,
    /// Provenance loaded from `<adapter>/lineage.json` if present. Lets
    /// the UI surface which Kiln build / base model / training input
    /// produced this adapter without forcing the user to open the file
    /// on disk. `None` when no lineage file exists (e.g. uploaded
    /// adapters or pre-lineage adapters).
    #[serde(skip_serializing_if = "Option::is_none")]
    lineage: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct AdapterFileEntry {
    name: String,
    size_bytes: u64,
}

#[derive(Serialize)]
struct AdapterLinkedJob {
    job_id: String,
    job_type: crate::state::TrainingJobType,
    state: kiln_train::TrainingState,
    elapsed_secs: f64,
    final_loss: Option<f64>,
}

#[derive(Serialize)]
struct AdapterLinkedEval {
    job_id: String,
    suite_name: String,
    accuracy: Option<f32>,
    state: kiln_eval::EvalJobState,
}

async fn adapter_detail(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<AdapterDetail>, ApiError> {
    validate_adapter_name(&name)?;
    let path = state.adapter_dir.join(&name);
    let mut files: Vec<AdapterFileEntry> = Vec::new();
    let mut size_bytes = 0u64;
    let rd = match std::fs::read_dir(&path) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::adapter_not_found(&name));
        }
        Err(e) => return Err(ApiError::internal(format!("read_dir: {e}"))),
    };
    {
        for entry in rd.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    let n = entry.file_name().to_string_lossy().to_string();
                    files.push(AdapterFileEntry {
                        name: n,
                        size_bytes: meta.len(),
                    });
                    size_bytes += meta.len();
                }
            }
        }
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));

    let active = state.active_adapter_name.read().unwrap().clone();
    let is_active = active.as_deref() == Some(name.as_str());

    // Both lookups follow the same pattern: hold the read lock just long
    // enough to collect a `(submitted_at, projection)` Vec, drop the
    // lock, then sort outside it. Sorting under the lock would block the
    // training/eval workers' writers for the duration on a busy server.
    let mut training_pairs: Vec<(std::time::Instant, AdapterLinkedJob)> = {
        let jobs = state.training_jobs.read().unwrap();
        jobs.values()
            .filter(|j| j.adapter_name == name)
            .map(|j| {
                (
                    j.submitted_at,
                    AdapterLinkedJob {
                        job_id: j.job_id.clone(),
                        job_type: j.job_type,
                        state: j.state,
                        elapsed_secs: j.submitted_at.elapsed().as_secs_f64(),
                        final_loss: j.loss,
                    },
                )
            })
            .collect()
    };
    // Newest-first: more-recent submitted_at (larger Instant) first.
    training_pairs.sort_by(|a, b| b.0.cmp(&a.0));
    let training_jobs: Vec<AdapterLinkedJob> = training_pairs.into_iter().map(|(_, v)| v).collect();

    let mut eval_pairs: Vec<(String, AdapterLinkedEval)> = {
        let jobs = state.eval_jobs.read().unwrap();
        jobs.values()
            .filter(|j| j.adapters.iter().any(|a| a.as_deref() == Some(name.as_str())))
            .map(|j| {
                (
                    j.submitted_at_iso.clone(),
                    AdapterLinkedEval {
                        job_id: j.job_id.clone(),
                        suite_name: j.suite_name.clone(),
                        accuracy: j.headline_accuracy,
                        state: j.state,
                    },
                )
            })
            .collect()
    };
    eval_pairs.sort_by(|a, b| b.0.cmp(&a.0));
    let eval_jobs: Vec<AdapterLinkedEval> = eval_pairs.into_iter().map(|(_, v)| v).collect();

    // Best-effort: load `<adapter>/lineage.json` if present. Schema is
    // intentionally opaque (any `serde_json::Value`) so future lineage
    // schema bumps don't require touching this handler — the UI knows
    // how to render whatever fields are there.
    let lineage = std::fs::read(path.join("lineage.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());

    Ok(Json(AdapterDetail {
        name: name.clone(),
        is_active,
        has_config: path.join("adapter_config.json").exists(),
        has_weights: path.join("adapter_model.safetensors").exists(),
        size_bytes,
        files,
        training_jobs,
        eval_jobs,
        lineage,
    }))
}

/// `GET /v1/adapters/{name}/receipt` — return the §8.11 reproducibility
/// receipt for the adapter, or 404 when no receipt is on disk.
async fn adapter_receipt(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<kiln_train::AdapterReceipt>, ApiError> {
    let adapter_dir = state.adapter_dir.join(&name);
    if !adapter_dir.exists() {
        return Err(ApiError::adapter_not_found(&name));
    }
    match kiln_train::AdapterReceipt::read_from_adapter_dir(&adapter_dir) {
        Ok(Some(r)) => Ok(Json(r)),
        Ok(None) => Err(ApiError::adapter_not_found(&format!(
            "{name}/receipt.json"
        ))),
        Err(e) => Err(ApiError::internal(format!(
            "failed to read adapter receipt for {name}: {e:#}"
        ))),
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/adapters", get(list_adapters))
        .route("/v1/adapters/load", post(load_adapter))
        .route("/v1/adapters/unload", post(unload_adapter))
        .route("/v1/adapters/merge", post(merge_adapters))
        .route(
            "/v1/adapters/upload",
            post(upload_adapter).layer(DefaultBodyLimit::max(ADAPTER_UPLOAD_BODY_LIMIT)),
        )
        .route("/v1/adapters/{name}", delete(delete_adapter))
        .route("/v1/adapters/{name}/detail", get(adapter_detail))
        .route("/v1/adapters/{name}/download", get(download_adapter))
        .route("/v1/adapters/{name}/receipt", get(adapter_receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{HeaderMap, Method, Request, StatusCode};
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use serde_json::json;
    use std::sync::Arc;
    use std::time::Duration;
    use tower::ServiceExt;

    const STRESS_CYCLES: usize = 100;

    fn make_state(adapter_dir: PathBuf) -> AppState {
        let config = ModelConfig::qwen3_5_4b();
        let scheduler = Scheduler::new(
            SchedulerConfig {
                max_batch_tokens: 8192,
                max_batch_size: 64,
                block_size: 16,
                prefix_cache_enabled: false,
                ..Default::default()
            },
            256,
        );
        let engine = MockEngine::new(config.clone());
        let mut state = AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            crate::api::test_tokenizer(),
            300,
            "kiln-test".to_string(),
        );
        state.adapter_dir = adapter_dir;
        state.eval_mode = true;
        state
    }

    fn write_test_adapter(adapter_dir: &Path, name: &str) {
        let path = adapter_dir.join(name);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("adapter_config.json"),
            br#"{
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": "Qwen/Qwen3.5-4B"
            }"#,
        )
        .unwrap();
        std::fs::write(
            path.join("adapter_model.safetensors"),
            b"\x00\x00\x00\x00tiny-test-adapter",
        )
        .unwrap();
    }

    fn write_quarantined_canary(adapter_dir: &Path, name: &str, reason: &str) {
        let path = adapter_dir.join(name);
        std::fs::write(
            path.join(kiln_train::ADAPTER_CANARY_STATUS_FILENAME),
            json!({
                "schema_version": 1,
                "receipt_type": "kiln_adapter_canary_status",
                "adapter_name": name,
                "produced_at": "2026-05-21T00:00:00Z",
                "source": "adapter_smoke_test",
                "status": "quarantined",
                "passed": false,
                "failure_reason": reason,
                "warnings": [reason],
                "notes": [],
                "checks": [
                    {
                        "name": "nonzero_logit_delta_from_base",
                        "passed": false,
                        "failure_reason": reason
                    }
                ],
                "prompt_diagnostics": []
            })
            .to_string(),
        )
        .unwrap();
    }

    async fn request_json(
        app: &axum::Router,
        method: Method,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, HeaderMap, serde_json::Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = if let Some(body) = body {
            builder = builder.header("content-type", "application/json");
            Body::from(body.to_string())
        } else {
            Body::empty()
        };
        let resp = app.clone().oneshot(builder.body(body).unwrap()).await.unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body = if bytes.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, headers, body)
    }

    fn assert_adapter_state(body: &serde_json::Value, expected: Option<&str>) {
        match expected {
            Some(name) => {
                assert_eq!(body["active"].as_str(), Some(name));
                assert_eq!(body["active_adapter"].as_str(), Some(name));
                assert_eq!(body["loaded_adapter"].as_str(), Some(name));
                assert!(
                    body["loaded_adapters"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|value| value.as_str() == Some(name)),
                    "loaded_adapters should include {name}: {body:?}"
                );
            }
            None => {
                assert!(body["active"].is_null());
                assert!(body["active_adapter"].is_null());
                assert!(body["loaded_adapter"].is_null());
                assert!(body["loaded_adapters"].as_array().unwrap().is_empty());
            }
        }
    }

    fn memory_growth_counter_bytes(state: &AppState) -> Option<u64> {
        let budget = state.memory_budget.as_ref();
        if budget.total_vram_bytes == 0 {
            return None;
        }
        Some(
            budget
                .model_memory_bytes
                .saturating_add(budget.kv_cache_bytes)
                .saturating_add(budget.peak_prefill_used_vram_bytes()),
        )
    }

    fn percentile_micros(samples: &[Duration], percentile: f64) -> u128 {
        let mut values: Vec<u128> = samples.iter().map(|sample| sample.as_micros()).collect();
        values.sort_unstable();
        let idx = ((values.len() as f64 * percentile).ceil() as usize)
            .saturating_sub(1)
            .min(values.len().saturating_sub(1));
        values[idx]
    }

    #[tokio::test]
    async fn adapter_registry_reports_quarantined_canary_status() {
        let tmp = tempfile::tempdir().unwrap();
        write_test_adapter(tmp.path(), "bad-canary");
        write_quarantined_canary(
            tmp.path(),
            "bad-canary",
            "adapter canary observed no nonzero logit delta from base",
        );

        let state = make_state(tmp.path().to_path_buf());
        let app = crate::api::router(state);
        let (status, _, body) = request_json(&app, Method::GET, "/v1/adapters", None).await;
        assert_eq!(status, StatusCode::OK);
        let entry = body["available_adapters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["name"] == "bad-canary")
            .unwrap();

        assert_eq!(entry["status"], "quarantined");
        assert_eq!(entry["canary_status"], "quarantined");
        assert!(
            entry["canary_failure_reason"]
                .as_str()
                .unwrap()
                .contains("no nonzero logit delta")
        );
        assert_eq!(
            entry["canary_checks"][0]["name"],
            "nonzero_logit_delta_from_base"
        );
    }

    #[tokio::test]
    async fn adapter_load_rejects_quarantined_adapter_without_override() {
        let tmp = tempfile::tempdir().unwrap();
        write_test_adapter(tmp.path(), "bad-canary");
        write_quarantined_canary(
            tmp.path(),
            "bad-canary",
            "adapter canary observed no nonzero logit delta from base",
        );

        let state = make_state(tmp.path().to_path_buf());
        let state_for_assert = state.clone();
        let app = crate::api::router(state);
        let (status, _, body) = request_json(
            &app,
            Method::POST,
            "/v1/adapters/load",
            Some(json!({ "name": "bad-canary" })),
        )
        .await;

        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "adapter_quarantined");
        assert!(
            state_for_assert
                .active_adapter_name
                .read()
                .unwrap()
                .is_none()
        );
        assert!(
            state_for_assert
                .loaded_adapter_name
                .read()
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn adapter_load_override_allows_quarantined_adapter() {
        let tmp = tempfile::tempdir().unwrap();
        write_test_adapter(tmp.path(), "bad-canary");
        write_quarantined_canary(
            tmp.path(),
            "bad-canary",
            "adapter canary observed no nonzero logit delta from base",
        );

        let state = make_state(tmp.path().to_path_buf());
        let state_for_assert = state.clone();
        let app = crate::api::router(state);
        let (status, _, body) = request_json(
            &app,
            Method::POST,
            "/v1/adapters/load",
            Some(json!({ "name": "bad-canary", "allow_quarantined": true })),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "loaded");
        assert_eq!(body["name"], "bad-canary");
        assert_eq!(
            state_for_assert
                .active_adapter_name
                .read()
                .unwrap()
                .as_deref(),
            Some("bad-canary")
        );
    }

    #[tokio::test]
    async fn adapter_load_eval_unload_stress_tracks_state_latency_and_memory() {
        let tmp = tempfile::tempdir().unwrap();
        write_test_adapter(tmp.path(), "stress-a");
        write_test_adapter(tmp.path(), "stress-b");

        let state = make_state(tmp.path().to_path_buf());
        let state_for_assert = state.clone();
        let memory_before = memory_growth_counter_bytes(&state_for_assert);
        let app = crate::api::router(state);
        let mut cycle_latencies = Vec::with_capacity(STRESS_CYCLES);

        for cycle in 0..STRESS_CYCLES {
            let adapter = if cycle % 2 == 0 {
                "stress-a"
            } else {
                "stress-b"
            };
            let started = std::time::Instant::now();

            let (status, _, body) = request_json(
                &app,
                Method::POST,
                "/v1/adapters/load",
                Some(json!({ "name": adapter })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["status"], "loaded");
            assert_eq!(body["name"], adapter);

            let (status, _, body) =
                request_json(&app, Method::GET, "/v1/adapters", None).await;
            assert_eq!(status, StatusCode::OK);
            assert_adapter_state(&body, Some(adapter));
            assert_eq!(
                state_for_assert
                    .active_adapter_name
                    .read()
                    .unwrap()
                    .as_deref(),
                Some(adapter)
            );
            assert_eq!(
                state_for_assert
                    .loaded_adapter_name
                    .read()
                    .unwrap()
                    .as_deref(),
                Some(adapter)
            );

            let (status, headers, body) = request_json(
                &app,
                Method::POST,
                "/v1/chat/completions",
                Some(json!({
                    "messages": [{"role": "user", "content": format!("stress cycle {cycle}")}],
                    "max_tokens": 1
                })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(headers.get("x-kiln-eval-mode").unwrap(), "true");
            assert_eq!(headers.get("x-kiln-active-adapter").unwrap(), adapter);
            assert_eq!(headers.get("x-kiln-loaded-adapter").unwrap(), adapter);
            assert_eq!(body["object"], "chat.completion");

            let (status, _, body) =
                request_json(&app, Method::POST, "/v1/adapters/unload", Some(json!({}))).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["status"], "unloaded");

            let (status, _, body) =
                request_json(&app, Method::GET, "/v1/adapters", None).await;
            assert_eq!(status, StatusCode::OK);
            assert_adapter_state(&body, None);
            assert!(state_for_assert.active_adapter_name.read().unwrap().is_none());
            assert!(state_for_assert.loaded_adapter_name.read().unwrap().is_none());

            cycle_latencies.push(started.elapsed());
        }

        let first_p95 = percentile_micros(&cycle_latencies[..10], 0.95);
        let last_p95 = percentile_micros(&cycle_latencies[STRESS_CYCLES - 10..], 0.95);
        let severe_drift_limit = first_p95.saturating_mul(20).max(first_p95 + 100_000);
        assert!(
            last_p95 <= severe_drift_limit,
            "adapter stress latency drifted severely: first_p95={first_p95}us \
             last_p95={last_p95}us limit={severe_drift_limit}us"
        );

        if let (Some(before), Some(after)) =
            (memory_before, memory_growth_counter_bytes(&state_for_assert))
        {
            assert!(
                after <= before.saturating_add(16 * 1024 * 1024),
                "adapter stress memory grew unexpectedly: before={before} after={after}"
            );
        }
    }
}
