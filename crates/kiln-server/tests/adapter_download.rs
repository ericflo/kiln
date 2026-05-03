//! Integration test: GET /v1/adapters/{name}/download streams the adapter
//! directory as a tar.gz archive.

use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use flate2::read::GzDecoder;
use serde_json::json;
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;

/// Minimal tokenizer for tests — the adapter download path doesn't tokenize
/// anything, but AppState::new_mock requires a real tokenizer.
fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("a".to_string(), 0);
    vocab.insert("b".to_string(), 1);
    let json = json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] }
    });
    KilnTokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn make_state(adapter_dir: std::path::PathBuf) -> AppState {
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
        test_tokenizer(),
        300,
        "qwen3.5-4b-kiln".to_string(),
    );
    state.adapter_dir = adapter_dir;
    state
}

fn write_adapter(adapter_dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let path = adapter_dir.join(name);
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(
        path.join("adapter_config.json"),
        br#"{"r": 8, "lora_alpha": 16}"#,
    )
    .unwrap();
    std::fs::write(
        path.join("adapter_model.safetensors"),
        b"\x00\x01\x02\x03fake-safetensors-bytes",
    )
    .unwrap();
    path
}

#[tokio::test]
async fn test_download_adapter_returns_valid_tar_gz() {
    let tmp = tempfile::tempdir().unwrap();
    write_adapter(tmp.path(), "test-adapter");
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let request = Request::builder()
        .method("GET")
        .uri("/v1/adapters/test-adapter/download")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/gzip"
    );
    assert_eq!(
        response
            .headers()
            .get("content-disposition")
            .unwrap()
            .to_str()
            .unwrap(),
        "attachment; filename=\"test-adapter.tar.gz\""
    );

    // Decompress and verify the archive contains both files with original
    // bytes under the adapter-name prefix.
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let gz = GzDecoder::new(&body_bytes[..]);
    let mut tar = tar::Archive::new(gz);
    let mut found: HashMap<String, Vec<u8>> = HashMap::new();
    for entry in tar.entries().unwrap() {
        let mut entry = entry.unwrap();
        let path = entry.path().unwrap().to_string_lossy().to_string();
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).unwrap();
        found.insert(path, buf);
    }

    let cfg = found
        .get("test-adapter/adapter_config.json")
        .expect("config entry missing");
    assert_eq!(cfg.as_slice(), br#"{"r": 8, "lora_alpha": 16}"#);

    let weights = found
        .get("test-adapter/adapter_model.safetensors")
        .expect("weights entry missing");
    assert_eq!(
        weights.as_slice(),
        b"\x00\x01\x02\x03fake-safetensors-bytes"
    );
}

#[tokio::test]
async fn test_download_adapter_not_found_returns_404() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let request = Request::builder()
        .method("GET")
        .uri("/v1/adapters/does-not-exist/download")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "adapter_not_found");
}

#[tokio::test]
async fn test_download_adapter_rejects_path_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // ".." as a single segment should be rejected — escapes the adapter dir.
    let request = Request::builder()
        .method("GET")
        .uri("/v1/adapters/..%2E/download")
        .body(Body::empty())
        .unwrap();

    // Some axum routers reject ".." segments at the routing layer entirely;
    // accept either a 400 from our validator or a 404/405 from the router as
    // long as the request is NOT served. The point is no traversal succeeds.
    let response = app.oneshot(request).await.unwrap();
    assert_ne!(response.status(), StatusCode::OK);

    // Try a name that definitely reaches our handler: contains a backslash.
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);
    let request = Request::builder()
        .method("GET")
        .uri("/v1/adapters/foo%5Cbar/download")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["code"], "invalid_adapter_name");
}
