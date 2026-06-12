//! Integration test: a FAILED training job leaves the corrections basket
//! intact end-to-end.
//!
//! This is the contract the dashboard's Train button rides since it moved
//! to dataset "corrections:active": consumed row ids travel on the job and
//! flip to `trained_into` only when the worker reports Completed. The
//! 0.4.1 dead-end — submit succeeds, job fails, hand-written ideals gone —
//! must stay impossible. The worker's deterministic mock-mode failure arm
//! plays the failing job here.

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
use kiln_server::training_queue::{QueueEntry, QueuedJob, new_shutdown_flag, spawn_training_worker};
use kiln_train::{SftRequest, TrainingState};

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

async fn request(app: &axum::Router, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
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
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// Register a queued SFT job carrying consumed correction ids directly on
/// the state (mock mode rejects the HTTP submit; the worker's mock arm is
/// exactly the deterministic failure under test).
fn enqueue_corrections_job(state: &AppState, job_id: &str, correction_ids: Vec<String>) {
    let req: SftRequest = serde_json::from_value(json!({
        "examples": [{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]}],
    }))
    .unwrap();
    let info = TrainingJobInfo {
        job_id: job_id.to_string(),
        adapter_name: "codebase-corrections".to_string(),
        job_type: TrainingJobType::Sft,
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: 1,
        auto_load: false,
        consumed_correction_ids: correction_ids,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        loss_history: Vec::new(),
        cancel_requested: Default::default(),
    };
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job_id.to_string(), info);
    state.training_queue.lock().unwrap().push(QueueEntry {
        job_id: job_id.to_string(),
        reserved_bytes: 0,
        job: QueuedJob::Sft(req),
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_job_leaves_corrections_basket_intact() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());

    // A hand-written correction in the durable store.
    let (status, _) = request(
        &app,
        "POST",
        "/v1/corrections",
        Some(json!({
            "request_id": "chatcmpl-r1",
            "agent": "pi",
            "user": "what's this repo all about?",
            "original": "wrong answer",
            "ideal": "the hand-written ideal answer",
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // A training job that consumed it, headed for the worker's
    // deterministic mock-mode failure.
    enqueue_corrections_job(&state, "job-1", vec!["chatcmpl-r1".to_string()]);
    spawn_training_worker(state.clone(), new_shutdown_flag());

    // Wait for the terminal state.
    let mut last = Value::Null;
    for _ in 0..200 {
        let (_, body) = request(&app, "GET", "/v1/train/status/job-1", None).await;
        last = body;
        if last["state"] == "failed" || last["state"] == "completed" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(last["state"], "failed", "mock worker must fail the job: {last}");

    // The row is STILL active — visible in the default list, not marked.
    let (status, body) = request(&app, "GET", "/v1/corrections", None).await;
    assert_eq!(status, StatusCode::OK);
    let rows = body["corrections"].as_array().expect("corrections array");
    assert_eq!(
        rows.len(),
        1,
        "failed job must leave the correction active: {body}"
    );
    assert_eq!(rows[0]["request_id"], "chatcmpl-r1");
    assert!(
        rows[0].get("trained_into").is_none() || rows[0]["trained_into"].is_null(),
        "failed job must not mark rows trained: {body}"
    );
    assert_eq!(
        rows[0]["ideal"], "the hand-written ideal answer",
        "the hand-written ideal must survive the failure"
    );
}
