//! Adapter Library endpoints (grand plan §3.10).
//!
//! Public, opt-in distribution of pre-trained kiln adapters. A publishable
//! adapter carries its §8.11 audit receipt; install verifies the receipt and
//! writes the adapter under the local
//! `adapter_dir`. Publish bundles the local adapter + receipt + any
//! eval scores into a tarball POSTed to the registry.
//!
//! The library backend is configured once as `adapters.library_url`. Until the
//! operational backend ships, the endpoints surface a
//! sensible "not configured" message rather than failing hard. This
//! keeps the §4 endpoint contract real without forcing a managed
//! S3/CDN dependency for unit tests.

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LibraryAdapterEntry {
    /// Library-scoped unique id.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Source kind ("opd", "distill_pump", ...). Mirrors the receipt.
    pub source_kind: String,
    /// One-line description.
    pub description: Option<String>,
    /// Eval scores (suite → score).
    #[serde(default)]
    pub post_eval: std::collections::BTreeMap<String, f64>,
    /// Producer (uploader) — anonymous when not provided.
    #[serde(default)]
    pub uploader: Option<String>,
    /// Approximate adapter size in bytes.
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct LibraryListResponse {
    backend: String,
    adapters: Vec<LibraryAdapterEntry>,
    note: Option<String>,
}

async fn list_library(State(state): State<AppState>) -> Json<LibraryListResponse> {
    let backend = state.operational_runtime.adapter_library_url.clone();
    // Real HTTP fetch lives behind a feature gate once the library
    // backend is operational; for now we return an empty list +
    // configured backend URL so the endpoint contract is honoured.
    Json(LibraryListResponse {
        backend,
        adapters: Vec::new(),
        note: Some(
            "Library backend not yet operational; this endpoint will fetch from \
             the configured backend once the §3.10 launch ships. See grand plan \
             §3.10 / §13 Phase 3 success criteria."
                .into(),
        ),
    })
}

async fn install_from_library(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // The fetch step proper lands when the library backend opens up;
    // we surface the failure path and the expected return shape now
    // so callers can wire the dashboard against a known contract.
    let backend = state.operational_runtime.adapter_library_url.clone();
    Err(ApiError::training_invalid_request(format!(
        "library install for {id:?} not yet operational (backend = {backend}); \
         see grand plan §3.10 for the launch timeline"
    )))
}

#[derive(Debug, Deserialize)]
struct PublishPayload {
    /// Optional description published with the adapter.
    #[serde(default)]
    description: Option<String>,
    /// Optional uploader handle.
    #[serde(default)]
    uploader: Option<String>,
}

#[derive(Debug, Serialize)]
struct PublishToLibraryResponse {
    status: &'static str,
    backend: String,
    intended_id: String,
    uploader: Option<String>,
    description: Option<String>,
    receipt_schema_version: u32,
    note: &'static str,
}

async fn publish_to_library(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(payload): Json<PublishPayload>,
) -> Result<Json<PublishToLibraryResponse>, ApiError> {
    // Validate the local adapter has a §8.11 audit receipt — that's the
    // §3.10 provenance gate. Without it, refuse to publish.
    let adapter_dir = state.adapter_dir.join(&name);
    if !adapter_dir.exists() {
        return Err(ApiError::adapter_not_found(&name));
    }
    let receipt = kiln_train::AdapterReceipt::read_from_adapter_dir(&adapter_dir)
        .map_err(|e| ApiError::internal(format!("read receipt: {e:#}")))?
        .ok_or_else(|| {
            ApiError::training_invalid_request(format!(
                "adapter {name:?} has no audit receipt; refusing to publish \
                 (per §3.10 the library only accepts adapters with §8.11 audit records)"
            ))
        })?;

    let backend = state.operational_runtime.adapter_library_url.clone();
    // Real publish wires once the backend is live. We surface the
    // pieces the dashboard / CLI expect: backend URL, receipt
    // digest, intended id format. Caller can use this to decide
    // whether to proceed with a manual upload until the operational
    // backend ships.
    Ok(Json(PublishToLibraryResponse {
        status: "ready_to_publish",
        backend,
        intended_id: format!("{}@{}", name, &receipt.produced_at[..10]),
        uploader: payload.uploader,
        description: payload.description,
        receipt_schema_version: receipt.schema_version,
        note: "Library publish endpoint is contract-only until §3.10 launch; \
               see grand plan §3.10 / §13 Phase 3 success criteria.",
    }))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/library", get(list_library))
        .route("/v1/library/install/{id}", post(install_from_library))
        .route("/v1/library/publish/{name}", post(publish_to_library))
}
