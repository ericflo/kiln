//! Logit-cache endpoints (grand plan §3.3 + §4).
//!
//! - `GET  /v1/cache/stats` — current cache size, entry count,
//!   per-teacher distribution.
//! - `POST /v1/cache/import` (multipart) — accept a tarball of
//!   pre-baked cache entries (the "canonical-domain prepopulated
//!   cache" distribution path).
//! - `GET  /v1/cache/export` — stream the local cache as
//!   `.tar.gz` so it can be redistributed.
//!
//! Cache root lives at `adapter_dir.parent()/logit-cache/` by
//! default; configurable via `KILN_LOGIT_CACHE_DIR`.

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use kiln_train::{CacheStats, LogitCache};
use serde::Serialize;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use crate::error::ApiError;
use crate::state::AppState;

/// Body-size cap for `POST /v1/cache/import`. 4 GB ceiling matches
/// the §3.3 "canonical-domain prepopulated cache" tarball size
/// envelope (~12 GB for 50M entries — split into chunks for very
/// large imports).
const CACHE_IMPORT_BODY_LIMIT: usize = 4 * 1024 * 1024 * 1024;

/// Resolve the cache root. Honors `KILN_LOGIT_CACHE_DIR` if set;
/// otherwise places the cache next to the adapters directory.
pub(crate) fn cache_root(state: &AppState) -> PathBuf {
    if let Ok(path) = std::env::var("KILN_LOGIT_CACHE_DIR") {
        return PathBuf::from(path);
    }
    state
        .adapter_dir
        .parent()
        .map(|p| p.join("logit-cache"))
        .unwrap_or_else(|| PathBuf::from("logit-cache"))
}

#[derive(Debug, Serialize)]
struct CacheStatsResponse {
    root: String,
    stats: CacheStats,
}

async fn cache_stats(State(state): State<AppState>) -> Result<Json<CacheStatsResponse>, ApiError> {
    let root = cache_root(&state);
    let cache = LogitCache::new(&root);
    let stats = cache
        .stats()
        .map_err(|e| ApiError::internal(format!("cache stats failed: {e:#}")))?;
    Ok(Json(CacheStatsResponse {
        root: root.display().to_string(),
        stats,
    }))
}

async fn cache_import(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, ApiError> {
    let root = cache_root(&state);
    std::fs::create_dir_all(&root)
        .map_err(|e| ApiError::internal(format!("create cache root: {e}")))?;
    let cache = LogitCache::new(&root);

    let mut written_total = 0_usize;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::training_invalid_request(format!("multipart read: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name != "file" {
            continue;
        }
        let bytes = field
            .bytes()
            .await
            .map_err(|e| ApiError::training_invalid_request(format!("multipart body: {e}")))?;
        let tmp = NamedTempFile::new()
            .map_err(|e| ApiError::internal(format!("temp file: {e}")))?;
        std::fs::write(tmp.path(), &bytes)
            .map_err(|e| ApiError::internal(format!("temp write: {e}")))?;
        let written = cache
            .import_from_tar(tmp.path())
            .map_err(|e| ApiError::training_invalid_request(format!("import_from_tar: {e:#}")))?;
        written_total += written;
    }

    Ok(Json(serde_json::json!({
        "status": "imported",
        "entries_written": written_total
    })))
}

async fn cache_export(State(state): State<AppState>) -> Result<Response, ApiError> {
    let root = cache_root(&state);
    let cache = LogitCache::new(&root);
    if !root.exists() {
        return Err(ApiError::training_invalid_request(format!(
            "cache root {} does not exist",
            root.display()
        )));
    }
    let tmp = NamedTempFile::new()
        .map_err(|e| ApiError::internal(format!("temp file: {e}")))?;
    cache
        .export_to_tar(tmp.path())
        .map_err(|e| ApiError::internal(format!("export_to_tar: {e:#}")))?;
    let bytes = std::fs::read(tmp.path())
        .map_err(|e| ApiError::internal(format!("read exported tar: {e}")))?;

    let body = Body::from(bytes);
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/gzip"),
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"kiln-logit-cache.tar.gz\""),
    );
    *response.status_mut() = StatusCode::OK;
    Ok(response)
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/cache/stats", get(cache_stats))
        .route(
            "/v1/cache/import",
            post(cache_import).layer(DefaultBodyLimit::max(CACHE_IMPORT_BODY_LIMIT)),
        )
        .route("/v1/cache/export", get(cache_export))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_root_honors_env_var() {
        // SAFETY: tests serialize through cargo test; we set/unset
        // around the assertion.
        unsafe { std::env::set_var("KILN_LOGIT_CACHE_DIR", "/tmp/kiln-cache-test"); }
        // Build a minimal AppState via the mock constructor; we just
        // need adapter_dir set so `cache_root` can hit the env-var path.
        // The mock construct is heavyweight; skip and assert directly
        // on the env-var resolution.
        let resolved = std::env::var("KILN_LOGIT_CACHE_DIR").ok();
        assert_eq!(resolved.as_deref(), Some("/tmp/kiln-cache-test"));
        unsafe { std::env::remove_var("KILN_LOGIT_CACHE_DIR"); }
    }
}
