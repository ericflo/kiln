use std::path::Path;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

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

/// List loaded and available adapters.
async fn list_adapters(State(state): State<AppState>) -> Json<AdaptersResponse> {
    // Check active adapter from ModelRunner
    let active = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => {
            let guard = runner.lock().unwrap();
            guard.active_lora().map(|lora| {
                format!("rank={}, alpha={}", lora.rank, lora.alpha)
            })
        }
        ModelBackend::Mock { .. } => None,
    };

    // Scan adapter directory for available adapters
    let available = scan_adapter_dir(&state.adapter_dir);

    Json(AdaptersResponse { active, available })
}

/// Load a LoRA adapter from disk.
async fn load_adapter(
    State(state): State<AppState>,
    Json(req): Json<LoadAdapterRequest>,
) -> Result<Json<LoadAdapterResponse>, (StatusCode, String)> {
    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner,
        ModelBackend::Mock { .. } => {
            return Err((
                StatusCode::BAD_REQUEST,
                "adapter loading not supported in mock mode".to_string(),
            ));
        }
    };

    // Resolve adapter path: if name is absolute, use it directly; otherwise look in adapter_dir.
    let adapter_path = if Path::new(&req.name).is_absolute() {
        std::path::PathBuf::from(&req.name)
    } else {
        state.adapter_dir.join(&req.name)
    };

    if !adapter_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("adapter directory not found: {}", adapter_path.display()),
        ));
    }

    // Load adapter (this does I/O + tensor allocation, so run on blocking thread)
    let runner = runner.clone();
    let path = adapter_path.clone();
    let name = req.name.clone();

    tokio::task::spawn_blocking(move || {
        let mut guard = runner.lock().unwrap();
        guard.load_adapter(&path)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")))?
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to load adapter: {e}"),
        )
    })?;

    tracing::info!(adapter = %req.name, path = %adapter_path.display(), "loaded LoRA adapter");

    Ok(Json(LoadAdapterResponse {
        status: "loaded",
        name,
    }))
}

/// Unload the active LoRA adapter, reverting to base model.
async fn unload_adapter(
    State(state): State<AppState>,
) -> Result<Json<UnloadAdapterResponse>, (StatusCode, String)> {
    let runner = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner,
        ModelBackend::Mock { .. } => {
            return Err((
                StatusCode::BAD_REQUEST,
                "adapter management not supported in mock mode".to_string(),
            ));
        }
    };

    let runner = runner.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = runner.lock().unwrap();
        guard.unload_adapter();
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")))?;

    tracing::info!("unloaded LoRA adapter — reverted to base model");

    Ok(Json(UnloadAdapterResponse {
        status: "unloaded",
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
                entries.push(AdapterDiskEntry {
                    name,
                    has_config,
                    has_weights,
                });
            }
        }
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/adapters", get(list_adapters))
        .route("/v1/adapters/load", post(load_adapter))
        .route("/v1/adapters/unload", post(unload_adapter))
}
