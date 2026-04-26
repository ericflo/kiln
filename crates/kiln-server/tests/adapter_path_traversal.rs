//! Integration tests for path-traversal protections in the adapter
//! HTTP handlers. Covers Phase 9 audit findings §2b (`DELETE /v1/adapters/:name`),
//! §2c (`POST /v1/adapters/load`), and §2d (`POST /v1/adapters/merge`).
//!
//! Each handler must reject names that escape `adapter_dir` — `..`, segments
//! containing `/` or `\`, and absolute paths — before touching the
//! filesystem.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;

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

async fn read_error_code(resp: axum::http::Response<Body>) -> (StatusCode, String) {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value =
        serde_json::from_slice(&body).expect("error response must be JSON");
    let code = json["error"]["code"]
        .as_str()
        .expect("error.code must be a string")
        .to_string();
    (status, code)
}

/// `DELETE /v1/adapters/..` (and a few other escape variants) must return a
/// 4xx with `invalid_adapter_name`, must not call `remove_dir_all` on the
/// adapter directory, and must not delete sibling directories.
#[tokio::test]
async fn test_delete_rejects_path_traversal() {
    let tmp = tempfile::tempdir().unwrap();

    // Sentinel adapter directory the handler must NOT touch.
    write_adapter(tmp.path(), "real-adapter");

    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    // `..` is the canonical path-traversal name. Test it plus a couple of
    // other traversal-shaped values that the validator should reject. URL-
    // encode `/` and `\` so axum's path extractor matches the `{name}`
    // segment instead of falling through to a different route.
    let bad_names = ["..", ".", "%2E%2E", "foo%2Fbar", "%2Ftmp%2Ffoo"];

    for bad in bad_names {
        let uri = format!("/v1/adapters/{bad}");
        let req = Request::builder()
            .method("DELETE")
            .uri(&uri)
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let (status, code) = read_error_code(resp).await;
        assert!(
            status.is_client_error(),
            "DELETE {uri} returned {status}, expected 4xx (code={code})"
        );
        assert_eq!(
            code, "invalid_adapter_name",
            "DELETE {uri} returned code {code}, expected invalid_adapter_name"
        );
    }

    // The sentinel adapter and its files must still exist.
    let sentinel = tmp.path().join("real-adapter");
    assert!(sentinel.exists() && sentinel.is_dir());
    assert!(sentinel.join("adapter_config.json").exists());
    assert!(sentinel.join("adapter_model.safetensors").exists());
}

/// `POST /v1/adapters/load` with `name: ".."` must be rejected with
/// `invalid_adapter_name` before touching any filesystem path. (Mock backend
/// would otherwise return `mock_mode_no_adapters`; with validation in front
/// the path-traversal name fails first.)
#[tokio::test]
async fn test_load_rejects_relative_path_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = serde_json::to_vec(&json!({ "name": ".." })).unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/load")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let (status, code) = read_error_code(resp).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "load with name='..' should return 400, got {status} (code={code})"
    );
    assert_eq!(code, "invalid_adapter_name");
}

/// `POST /v1/adapters/load` with an absolute path (e.g. `/tmp/foo`) must be
/// rejected. The legacy "absolute paths bypass adapter_dir" branch was
/// removed in this PR.
#[tokio::test]
async fn test_load_rejects_absolute_path() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let body = serde_json::to_vec(&json!({ "name": "/tmp/foo" })).unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/adapters/load")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let (status, code) = read_error_code(resp).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "load with absolute path should return 400, got {status} (code={code})"
    );
    assert_eq!(code, "invalid_adapter_name");
}

/// `POST /v1/adapters/merge` with a source adapter name of `".."` (or other
/// traversal-shaped names) must be rejected with `invalid_adapter_name`
/// before any filesystem access. The existing `output_name` check handles
/// the destination side; this test pins the source side.
#[tokio::test]
async fn test_merge_rejects_path_traversal_in_source_name() {
    let tmp = tempfile::tempdir().unwrap();
    let state = make_state(tmp.path().to_path_buf());
    let app = api::router(state);

    let bad_names = ["..", "/tmp/foo", "foo/bar", "foo\\bar"];
    for bad in bad_names {
        let body = serde_json::to_vec(&json!({
            "sources": [{ "name": bad, "weight": 1.0 }],
            "output_name": "merged-output",
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/adapters/merge")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let (status, code) = read_error_code(resp).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "merge with source name {bad:?} should return 400, got {status} (code={code})"
        );
        assert_eq!(
            code, "invalid_adapter_name",
            "merge with source name {bad:?} returned unexpected code {code}"
        );
    }

    // Nothing should have been written to adapter_dir as a side effect.
    let entries: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().flatten().collect();
    assert!(
        entries.is_empty(),
        "merge rejection must not create any files in adapter_dir"
    );
}
