//! Integration tests: the durable corrections store endpoints.
//!
//! The corrections basket is the literal mechanism of "your model gets
//! better every time you use it" — these pin that it survives the server
//! (JSONL on disk), that pi can file rows programmatically, and that
//! trained rows are kept as marked history rather than deleted.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use tower::ServiceExt;

use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use kiln_server::api;
use kiln_server::state::AppState;

fn test_tokenizer() -> KilnTokenizer {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0u32..32 {
        vocab.insert(format!("t{i}"), i);
    }
    let json = json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": [
            {
                "id": 0, "content": "<|endoftext|>",
                "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            },
        ]
    });
    KilnTokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
}

fn make_state() -> (AppState, tempfile::TempDir) {
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
        "Qwen3.5-4B".to_string(),
    );
    let dir = tempfile::tempdir().unwrap();
    state.adapter_dir = dir.path().to_path_buf();
    (state, dir)
}

async fn req(app: &axum::Router, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let builder = Request::builder().method(method).uri(path);
    let request = match body {
        Some(v) => builder
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&v).unwrap()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

#[tokio::test]
async fn corrections_crud_round_trip_persists_on_disk() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());

    // pi files a correction programmatically.
    let (status, row) = req(
        &app,
        "POST",
        "/v1/corrections",
        Some(json!({
            "request_id": "chatcmpl-1",
            "agent": "pi",
            "user": "rename the struct",
            "original": "renamed the wrong struct",
            "ideal": ""
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{row}");
    assert!(row["created_at"].as_str().is_some_and(|s| !s.is_empty()));

    // The operator writes the ideal answer — an upsert by request_id.
    let (status, _) = req(
        &app,
        "POST",
        "/v1/corrections",
        Some(json!({
            "request_id": "chatcmpl-1",
            "agent": "pi",
            "user": "rename the struct",
            "original": "renamed the wrong struct",
            "ideal": "renamed TokenStore and updated 3 call sites"
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, list) = req(&app, "GET", "/v1/corrections", None).await;
    assert_eq!(status, StatusCode::OK);
    let rows = list["corrections"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "upsert must not duplicate: {list}");
    assert_eq!(rows[0]["ideal"], "renamed TokenStore and updated 3 call sites");

    // The store is a real file under the adapter dir.
    assert!(
        state
            .adapter_dir
            .join(".eval/corrections/data.jsonl")
            .is_file(),
        "corrections must live on disk, not in a browser"
    );

    let (status, _) = req(&app, "DELETE", "/v1/corrections/chatcmpl-1", None).await;
    assert_eq!(status, StatusCode::OK);
    let (_, list) = req(&app, "GET", "/v1/corrections", None).await;
    assert_eq!(list["corrections"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn trained_rows_become_marked_history_not_garbage() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());

    for id in ["a", "b"] {
        let (status, _) = req(
            &app,
            "POST",
            "/v1/corrections",
            Some(json!({"request_id": id, "user": "q", "original": "bad", "ideal": "good"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    let (status, marked) = req(
        &app,
        "POST",
        "/v1/corrections/mark_trained",
        Some(json!({"request_ids": ["a"], "adapter": "fixes-v1"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{marked}");
    assert_eq!(marked["marked"], 1);

    // Active view hides the trained row; history view keeps it with the
    // adapter it trained into.
    let (_, active) = req(&app, "GET", "/v1/corrections", None).await;
    assert_eq!(active["corrections"].as_array().unwrap().len(), 1);
    let (_, all) = req(&app, "GET", "/v1/corrections?include_trained=1", None).await;
    let rows = all["corrections"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    let trained = rows.iter().find(|r| r["request_id"] == "a").unwrap();
    assert_eq!(trained["trained_into"], "fixes-v1");
    assert!(trained["trained_at"].as_str().is_some());

    // Clearing the basket removes only active rows — the trained ideal
    // answer survives.
    let (status, cleared) = req(&app, "DELETE", "/v1/corrections", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cleared["removed"], 1);
    let (_, all) = req(&app, "GET", "/v1/corrections?include_trained=1", None).await;
    assert_eq!(all["corrections"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn rejects_rows_without_identity_or_prompt() {
    let (state, _dir) = make_state();
    let app = api::router(state);
    let (status, _) = req(
        &app,
        "POST",
        "/v1/corrections",
        Some(json!({"request_id": "", "user": "q"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = req(
        &app,
        "POST",
        "/v1/corrections",
        Some(json!({"request_id": "x", "user": "  "})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
