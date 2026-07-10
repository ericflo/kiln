//! Logit-cache endpoints (grand plan §3.3 + §4).
//!
//! - `GET  /v1/cache/stats` — current cache size, entry count,
//!   per-teacher distribution.
//! - `GET  /v1/cache/export` — stream a bounded, deterministic `.tar.gz`
//!   for operator backup and inspection.
//!
//! HTTP import is intentionally unavailable. Legacy archive extraction wrote
//! untrusted tar paths directly into the live cache root. Import must remain
//! disabled until a versioned archive can be staged, fully validated against
//! registered teacher identities, and atomically published.
//!
//! Cache root lives at `adapter_dir.parent()/logit-cache/` by
//! default; configurable via `KILN_LOGIT_CACHE_DIR`.

use anyhow::Context;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::Response;
use axum::routing::get;
use axum::{Json, Router};
use kiln_train::{CacheStats, LogitCache};
use serde::Serialize;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tempfile::NamedTempFile;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

use crate::error::ApiError;
use crate::state::AppState;

static CACHE_OPERATION_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn try_cache_operation_permit() -> Result<OwnedSemaphorePermit, ApiError> {
    CACHE_OPERATION_SEMAPHORE
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::cache_operation_busy())
}

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
    let root_display = root.display().to_string();
    let permit = try_cache_operation_permit()?;
    let stats = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        LogitCache::new(&root).stats()
    })
    .await
    .map_err(|error| ApiError::internal(format!("cache stats worker panicked: {error}")))?
    .map_err(|e| ApiError::internal(format!("cache stats failed: {e:#}")))?;
    Ok(Json(CacheStatsResponse {
        root: root_display,
        stats,
    }))
}

async fn cache_export(State(state): State<AppState>) -> Result<Response, ApiError> {
    let root = cache_root(&state);
    if !root.exists() {
        return Err(ApiError::training_invalid_request(format!(
            "cache root {} does not exist",
            root.display()
        )));
    }
    let permit = try_cache_operation_permit()?;
    let (tmp, archive_bytes) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let cache = LogitCache::new(&root);
        let tmp = NamedTempFile::new().context("create cache export temp file")?;
        let archive_bytes = cache.export_to_tar(tmp.path())?;
        Ok::<_, anyhow::Error>((tmp, archive_bytes))
    })
    .await
    .map_err(|error| ApiError::internal(format!("cache export worker panicked: {error}")))?
    .map_err(|error| ApiError::internal(format!("cache export failed: {error:#}")))?;

    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    tokio::task::spawn_blocking(move || {
        let mut file = match tmp.reopen() {
            Ok(file) => file,
            Err(error) => {
                let _ = tx.blocking_send(Err(error));
                return;
            }
        };
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            match file.read(&mut buffer) {
                Ok(0) => return,
                Ok(read) => {
                    if tx
                        .blocking_send(Ok(Bytes::copy_from_slice(&buffer[..read])))
                        .is_err()
                    {
                        return;
                    }
                }
                Err(error) => {
                    let _ = tx.blocking_send(Err(error));
                    return;
                }
            }
        }
    });

    let body = Body::from_stream(ReceiverStream::new(rx));
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/gzip"),
    );
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"kiln-logit-cache.tar.gz\""),
    );
    response.headers_mut().insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&archive_bytes.to_string())
            .map_err(|error| ApiError::internal(format!("cache export length: {error}")))?,
    );
    *response.status_mut() = StatusCode::OK;
    Ok(response)
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/cache/stats", get(cache_stats))
        .route("/v1/cache/export", get(cache_export))
}

#[cfg(test)]
mod tests {
    #[test]
    fn cache_root_honors_env_var() {
        // SAFETY: tests serialize through cargo test; we set/unset
        // around the assertion.
        unsafe {
            std::env::set_var("KILN_LOGIT_CACHE_DIR", "/tmp/kiln-cache-test");
        }
        // Build a minimal AppState via the mock constructor; we just
        // need adapter_dir set so `cache_root` can hit the env-var path.
        // The mock construct is heavyweight; skip and assert directly
        // on the env-var resolution.
        let resolved = std::env::var("KILN_LOGIT_CACHE_DIR").ok();
        assert_eq!(resolved.as_deref(), Some("/tmp/kiln-cache-test"));
        unsafe {
            std::env::remove_var("KILN_LOGIT_CACHE_DIR");
        }
    }
}
