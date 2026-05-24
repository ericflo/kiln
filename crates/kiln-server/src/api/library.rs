//! Adapter Library endpoints (grand plan §3.10).
//!
//! Public, opt-in distribution of pre-trained kiln adapters. Each
//! adapter ships with its §8.11 reproducibility receipt; install
//! verifies the receipt and writes the adapter under the local
//! `adapter_dir`. Publish bundles the local adapter + receipt + any
//! eval scores into a tarball POSTed to the registry.
//!
//! For milestone-10 the library backend is **configurable but
//! optional**: set `KILN_ADAPTER_LIBRARY_URL` (default
//! `https://library.kiln.run`) for the real backend. When the var is
//! unset or points at a local-only URL, the endpoints surface a
//! sensible "not configured" message rather than failing hard. This
//! keeps the §4 endpoint contract real without forcing a managed
//! S3/CDN dependency for unit tests.

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

const DEFAULT_LIBRARY_URL: &str = "https://library.kiln.run";

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

fn library_url() -> String {
    std::env::var("KILN_ADAPTER_LIBRARY_URL").unwrap_or_else(|_| DEFAULT_LIBRARY_URL.to_string())
}

async fn list_library(State(_state): State<AppState>) -> Json<LibraryListResponse> {
    let backend = library_url();
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
    let backend = library_url();
    let _ = state;
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

async fn publish_to_library(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
    Json(payload): Json<PublishPayload>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Validate the local adapter has a §8.11 receipt — that's the
    // §3.10 trust gate. Without it, refuse to publish.
    let adapter_dir = state.adapter_dir.join(&name);
    if !adapter_dir.exists() {
        return Err(ApiError::adapter_not_found(&name));
    }
    let receipt = kiln_train::AdapterReceipt::read_from_adapter_dir(&adapter_dir)
        .map_err(|e| ApiError::internal(format!("read receipt: {e:#}")))?
        .ok_or_else(|| {
            ApiError::training_invalid_request(format!(
                "adapter {name:?} has no reproducibility receipt; refusing to publish \
                 (per §3.10 the library only accepts adapters with §8.11 receipts)"
            ))
        })?;

    let backend = library_url();
    // Real publish wires once the backend is live. We surface the
    // pieces the dashboard / CLI expect: backend URL, receipt
    // digest, intended id format. Caller can use this to decide
    // whether to proceed with a manual upload until the operational
    // backend ships.
    Ok(Json(serde_json::json!({
        "status": "ready_to_publish",
        "backend": backend,
        "intended_id": format!("{}@{}", name, &receipt.produced_at[..10]),
        "uploader": payload.uploader,
        "description": payload.description,
        "receipt_schema_version": receipt.schema_version,
        "note": "Library publish endpoint is contract-only until §3.10 launch; \
                 see grand plan §3.10 / §13 Phase 3 success criteria."
    })))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/library", get(list_library))
        .route(
            "/v1/library/install/{id}",
            post(install_from_library),
        )
        .route(
            "/v1/library/publish/{name}",
            post(publish_to_library),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize tests in this module that mutate the
    /// `KILN_ADAPTER_LIBRARY_URL` process-env so they don't race:
    /// cargo's test harness runs `#[test]`s in parallel by default,
    /// and a `set_var` from one test will leak into another's
    /// `library_url()` read otherwise. CI run 26347552033 failed on
    /// 39555704 with `default_library_url_when_env_unset` seeing
    /// `https://custom.example/` (the override test's value).
    fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn default_library_url_when_env_unset() {
        let _g = env_test_lock();
        unsafe { std::env::remove_var("KILN_ADAPTER_LIBRARY_URL"); }
        assert_eq!(library_url(), DEFAULT_LIBRARY_URL);
    }

    #[test]
    fn env_var_override_takes_precedence() {
        let _g = env_test_lock();
        unsafe { std::env::set_var("KILN_ADAPTER_LIBRARY_URL", "https://custom.example/"); }
        assert_eq!(library_url(), "https://custom.example/");
        unsafe { std::env::remove_var("KILN_ADAPTER_LIBRARY_URL"); }
    }
}
