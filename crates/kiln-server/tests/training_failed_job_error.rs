//! Integration tests: failed training jobs carry their failure detail
//! through the status API instead of a bare `"state": "failed"`.
//!
//! Two failure paths are exercised end-to-end against the mock backend:
//! the worker's mock-mode arm (deterministic failure with a "real model
//! weights" message) and the queued-job cancel path ("cancelled while
//! queued"). Both must surface `error` on `GET /v1/train/status/{id}`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

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
use kiln_server::training_queue::{QueueEntry, QueuedJob, spawn_training_worker};
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
    // Keep the terminal-job archive write inside a tempdir.
    let dir = tempfile::tempdir().unwrap();
    state.adapter_dir = dir.path().to_path_buf();
    (state, dir)
}

/// Register a queued SFT job directly on the state (mock mode rejects
/// `/v1/train/sft` at submission, but the worker's mock arm is exactly
/// the deterministic failure under test).
fn enqueue_sft_job(state: &AppState, job_id: &str) {
    let req: SftRequest = serde_json::from_value(json!({
        "examples": [{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]}],
    }))
    .unwrap();
    let info = TrainingJobInfo {
        job_id: job_id.to_string(),
        adapter_name: "test-adapter".to_string(),
        job_type: TrainingJobType::Sft,
        effective_seed: Some(17),
        state: TrainingState::Queued,
        progress: 0.0,
        loss: None,
        epoch: None,
        adapter_path: None,
        submitted_at: std::time::Instant::now(),
        submitted_unix_ms: 1,
        auto_load: false,
        consumed_correction_ids: Vec::new(),
        training_data: None,
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
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
        external_promotion_gate_pending: false,
        reserved_bytes: 0,
        teacher_bindings: Vec::new(),
        admitted_resume_checkpoint: None,
        prepared_data: Default::default(),
        prepared_data_permit: Default::default(),
        job: QueuedJob::Sft(req),
    });
}

async fn get_status(app: &axum::Router, job_id: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/train/status/{job_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_failed_job_surfaces_error_in_status_api() {
    let (state, _dir) = make_state();
    enqueue_sft_job(&state, "job-mock-fail");
    let app = api::router(state.clone());

    let shutdown = kiln_server::training_queue::new_shutdown_flag();
    spawn_training_worker(state.clone(), shutdown.clone());

    // The worker polls every 500ms; bound the wait at ~10s.
    let mut body = Value::Null;
    for _ in 0..200 {
        let (status, value) = get_status(&app, "job-mock-fail").await;
        assert_eq!(status, StatusCode::OK, "{value}");
        if value["state"] == "failed" {
            body = value;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    shutdown.store(true, Ordering::Relaxed);

    assert_eq!(body["state"], "failed", "job never failed: {body}");
    let error = body["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("real model weights"),
        "failed job must carry the mock-arm failure detail: {body}"
    );
    assert!(
        body["finished_unix_ms"].is_u64(),
        "terminal job must stamp finished_unix_ms: {body}"
    );
}

#[tokio::test]
async fn cancelled_queued_job_stamps_error() {
    let (state, _dir) = make_state();
    enqueue_sft_job(&state, "job-cancel");
    let app = api::router(state.clone());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/train/queue/job-cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let (status, body) = get_status(&app, "job-cancel").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["state"], "failed", "{body}");
    assert_eq!(body["error"], "cancelled while queued", "{body}");
    assert!(
        body["finished_unix_ms"].is_u64(),
        "cancelled job must stamp finished_unix_ms: {body}"
    );
}

/// Active (non-failed) jobs keep `error` off the wire entirely.
#[tokio::test]
async fn queued_job_status_omits_error_key() {
    let (state, _dir) = make_state();
    enqueue_sft_job(&state, "job-queued");
    let app = api::router(state.clone());

    let (status, body) = get_status(&app, "job-queued").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["state"], "queued", "{body}");
    assert!(
        body.get("error").is_none(),
        "queued job must not serialize an error key: {body}"
    );
}

/// DELETE on a RUNNING job sets the cooperative cancel flag (the trainer's
/// per-step progress callback observes it and aborts at the next step
/// boundary) and answers "cancelling" — it must NOT remove the job or mark
/// it terminal; the worker does that when the trainer actually stops.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_running_job_sets_flag_without_terminal_transition() {
    let (state, _dir) = make_state();
    enqueue_sft_job(&state, "job-running");
    {
        let mut jobs = state.training_jobs.write().unwrap();
        jobs.get_mut("job-running").unwrap().state = TrainingState::Running;
    }
    let app = api::router(state.clone());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/train/queue/job-running")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["status"], "cancelling", "{body}");

    let jobs = state.training_jobs.read().unwrap();
    let job = jobs.get("job-running").unwrap();
    assert_eq!(
        job.state,
        TrainingState::Running,
        "stays Running until the trainer stops"
    );
    assert!(
        job.cancel_requested
            .load(std::sync::atomic::Ordering::Relaxed),
        "the cooperative flag must be set"
    );
}

/// Terminal jobs are not cancellable — the existing contract holds.
#[tokio::test]
async fn cancelling_terminal_job_is_rejected() {
    let (state, _dir) = make_state();
    enqueue_sft_job(&state, "job-done");
    {
        let mut jobs = state.training_jobs.write().unwrap();
        jobs.get_mut("job-done").unwrap().state = TrainingState::Completed;
    }
    let app = api::router(state.clone());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/train/queue/job-done")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
}
