//! Integration test: training tracked-jobs cap and TTL GC (audit MEDIUM §4 part 2).
//!
//! When the `training_jobs` tracking map is at its configured cap,
//! submissions to `/v1/train/sft` and `/v1/train/grpo` must return HTTP 503
//! with the `training_tracked_full` code and a `Retry-After: 30` header
//! instead of growing the map without bound. Terminal entries
//! (`Completed` / `Failed`) older than `tracked_job_ttl` must be evicted by
//! the GC pass.
//!
//! Cross-link: PR #607 added the queue cap (part 1). This test exercises
//! the tracking-map cap and TTL GC (part 2).

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
use kiln_server::state::{AppState, TrainingJobInfo, TrainingJobType};
use kiln_server::training_queue::gc_tracked_jobs;
use kiln_train::{
    ChatMessage, GrpoRequest, ScoredCompletion, SftExample, SftRequest, TrainingState,
};

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

fn make_state(max_tracked: usize, ttl: std::time::Duration) -> AppState {
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
    // Leave the queue cap permissive so we exercise the tracked-map cap
    // independently of the queue cap from PR #607.
    state.max_queued_training_jobs = 1_000_000;
    state.max_tracked_jobs = max_tracked;
    state.tracked_job_ttl = ttl;
    state
}

/// Pre-populate the tracking map with `n` synthetic entries at a given state.
/// Used to drive the cap and the GC pass without going through `submit_*`.
fn fill_tracked(
    state: &AppState,
    n: usize,
    job_state: TrainingState,
    finished_at: Option<std::time::Instant>,
) {
    let mut jobs = state.training_jobs.write().unwrap();
    for i in 0..n {
        let job_id = format!("synthetic-{i}");
        jobs.insert(
            job_id.clone(),
            TrainingJobInfo {
                job_id,
                adapter_name: format!("syn-adapter-{i}"),
                job_type: TrainingJobType::Sft,
                state: job_state,
                progress: if matches!(job_state, TrainingState::Completed) {
                    1.0
                } else {
                    0.0
                },
                loss: None,
                epoch: None,
                adapter_path: None,
                submitted_at: std::time::Instant::now(),
                auto_load: false,
                finished_at,
            },
        );
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
async fn submit_sft_returns_503_when_tracked_full() {
    let state = make_state(2, std::time::Duration::from_secs(3600));
    // Fill the tracking map with two terminal entries (no `finished_at`
    // so the GC won't drop them in this synchronous path).
    fill_tracked(&state, 2, TrainingState::Completed, None);

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
        Some("training_tracked_full"),
        "body was: {body}"
    );
    let msg = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        msg.contains("Training tracking map is at capacity"),
        "unexpected message: {msg}"
    );
}

#[tokio::test]
async fn submit_grpo_returns_503_when_tracked_full() {
    let state = make_state(2, std::time::Duration::from_secs(3600));
    fill_tracked(&state, 2, TrainingState::Failed, None);

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
        Some("training_tracked_full"),
        "body was: {body}"
    );
}

/// Below the cap, the request must NOT be rejected with `training_tracked_full`.
/// In mock mode it's rejected with `mock_mode` instead — the important thing
/// is that the cap branch hasn't fired.
#[tokio::test]
async fn submit_sft_below_cap_does_not_emit_tracked_full() {
    let state = make_state(4, std::time::Duration::from_secs(3600));
    fill_tracked(&state, 2, TrainingState::Completed, None); // 2 < cap 4

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
        code, "training_tracked_full",
        "tracked cap should not have fired; body was: {body}"
    );
}

/// `gc_tracked_jobs` evicts terminal entries past TTL.
#[test]
fn gc_evicts_terminal_entries_past_ttl() {
    // Tiny TTL so we can drive eviction without sleeping for real time.
    let state = make_state(64, std::time::Duration::from_millis(10));

    // Two completed entries with a stale `finished_at` (well past the
    // 10ms TTL by the time we call gc), one queued without a timestamp,
    // and one running.
    let stale = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(60))
        .expect("subtract 60s from now");
    fill_tracked(&state, 2, TrainingState::Completed, Some(stale));
    {
        let mut jobs = state.training_jobs.write().unwrap();
        jobs.insert(
            "live-queued".to_string(),
            TrainingJobInfo {
                job_id: "live-queued".to_string(),
                adapter_name: "live".to_string(),
                job_type: TrainingJobType::Sft,
                state: TrainingState::Queued,
                progress: 0.0,
                loss: None,
                epoch: None,
                adapter_path: None,
                submitted_at: std::time::Instant::now(),
                auto_load: false,
                finished_at: None,
            },
        );
        jobs.insert(
            "live-running".to_string(),
            TrainingJobInfo {
                job_id: "live-running".to_string(),
                adapter_name: "running".to_string(),
                job_type: TrainingJobType::Sft,
                state: TrainingState::Running,
                progress: 0.5,
                loss: None,
                epoch: None,
                adapter_path: None,
                submitted_at: std::time::Instant::now(),
                auto_load: false,
                finished_at: None,
            },
        );
    }

    assert_eq!(state.training_jobs.read().unwrap().len(), 4);
    let removed = gc_tracked_jobs(&state);
    assert_eq!(
        removed, 2,
        "should have evicted both stale terminal entries"
    );
    let after = state.training_jobs.read().unwrap();
    assert_eq!(after.len(), 2, "active entries must survive GC");
    assert!(
        after.contains_key("live-queued"),
        "queued entry must survive GC"
    );
    assert!(
        after.contains_key("live-running"),
        "running entry must survive GC"
    );
}

/// Terminal entries WITHIN the TTL window are not evicted, even when a GC
/// pass runs.
#[test]
fn gc_keeps_recent_terminal_entries() {
    let state = make_state(64, std::time::Duration::from_secs(3600));
    let recent = std::time::Instant::now();
    fill_tracked(&state, 3, TrainingState::Completed, Some(recent));

    let removed = gc_tracked_jobs(&state);
    assert_eq!(removed, 0, "recent terminal entries must not be evicted");
    assert_eq!(state.training_jobs.read().unwrap().len(), 3);
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
    Option<SftRequest>,
) {
    (None, None, None, None, None)
}
