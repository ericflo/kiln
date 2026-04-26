//! Integration test: training queue cap (audit MEDIUM §4 part 1).
//!
//! When the in-memory training queue is at its configured cap, submissions
//! to `/v1/train/sft` and `/v1/train/grpo` must return HTTP 503 with the
//! `training_queue_full` code and a `Retry-After: 30` header instead of
//! growing the queue without bound.

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
use kiln_server::training_queue::{QueueEntry, QueuedJob};
use kiln_train::{ChatMessage, GrpoRequest, ScoredCompletion, SftExample, SftRequest};

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

fn make_state(max_queued: usize) -> AppState {
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
    state.max_queued_training_jobs = max_queued;
    state
}

/// Pre-populate the training queue with `n` placeholder SFT entries so the
/// cap-fill condition is reached without going through `submit_sft`.
fn fill_queue(state: &AppState, n: usize) {
    let mut q = state.training_queue.lock().unwrap();
    for i in 0..n {
        q.push(QueueEntry {
            job_id: format!("placeholder-{i}"),
            job: QueuedJob::Sft(SftRequest {
                examples: Vec::new(),
                config: Default::default(),
            }),
        });
    }
}

fn sft_body() -> Value {
    json!({
        "examples": [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}
                ]
            }
        ]
    })
}

fn grpo_body() -> Value {
    json!({
        "groups": [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "completions": [
                    {"text": "a", "reward": 1.0},
                    {"text": "b", "reward": 0.0}
                ]
            }
        ]
    })
}

async fn read_body(resp: axum::response::Response) -> (StatusCode, Option<String>, Value) {
    let status = resp.status();
    let retry_after = resp
        .headers()
        .get(axum::http::header::RETRY_AFTER)
        .map(|v| v.to_str().unwrap().to_string());
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    (status, retry_after, body)
}

#[tokio::test]
async fn submit_sft_returns_503_when_queue_full() {
    let state = make_state(1);
    fill_queue(&state, 1);

    let app = api::router(state);
    let req = Request::post("/v1/train/sft")
        .header("content-type", "application/json")
        .body(Body::from(sft_body().to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let (status, retry_after, body) = read_body(resp).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(retry_after.as_deref(), Some("30"));
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("training_queue_full"),
        "body was: {body}"
    );
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("Training queue is at capacity"),
        "unexpected message: {msg}"
    );
}

#[tokio::test]
async fn submit_grpo_returns_503_when_queue_full() {
    let state = make_state(1);
    fill_queue(&state, 1);

    let app = api::router(state);
    let req = Request::post("/v1/train/grpo")
        .header("content-type", "application/json")
        .body(Body::from(grpo_body().to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let (status, retry_after, body) = read_body(resp).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(retry_after.as_deref(), Some("30"));
    assert_eq!(
        body["error"]["code"].as_str(),
        Some("training_queue_full"),
        "body was: {body}"
    );
}

/// Below the cap, the request must NOT be rejected with `training_queue_full`.
/// In mock mode it's rejected with `mock_mode` instead — the important thing
/// is that the cap branch hasn't fired.
#[tokio::test]
async fn submit_sft_below_cap_does_not_emit_queue_full() {
    let state = make_state(2);
    fill_queue(&state, 1); // 1 < cap of 2

    let app = api::router(state);
    let req = Request::post("/v1/train/sft")
        .header("content-type", "application/json")
        .body(Body::from(sft_body().to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let (status, _retry_after, body) = read_body(resp).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE); // mock-mode fallback
    let code = body["error"]["code"].as_str().unwrap_or("");
    assert_ne!(
        code, "training_queue_full",
        "cap should not have fired; body was: {body}"
    );
}

/// Sanity unused-import check — confirms `ScoredCompletion`/`ChatMessage`/
/// `GrpoRequest` are reachable so the JSON we construct above keeps shape
/// parity with the wire types.
#[allow(dead_code)]
fn _imports_keep_alive() -> (
    Option<ScoredCompletion>,
    Option<ChatMessage>,
    Option<GrpoRequest>,
    Option<SftExample>,
) {
    (None, None, None, None)
}
