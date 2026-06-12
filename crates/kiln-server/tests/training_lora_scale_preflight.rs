//! Integration test: LoRA alpha/rank scaling is validated AT SUBMIT.
//!
//! The trainer has always enforced `alpha/rank <= 2.0` (the
//! `unsafe_lora_scale` gate) — but it fired only after the job was
//! accepted, queued, and run, so the caller saw a 200 and a doomed job.
//! The dashboard's corrections train shipped exactly that: rank 8 with
//! the server-default alpha 32 (ratio 4.0) failed every job *after* the
//! UI had already marked the basket trained. Every submission endpoint
//! must reject an unsafe pair with 400 `training_invalid_request` before
//! any queue/registration work happens, and let `allow_high_lora_scale`
//! opt out exactly like the trainer does.

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

fn make_state() -> AppState {
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
    AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        test_tokenizer(),
        300,
        "Qwen3.5-4B".to_string(),
    )
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

fn assert_no_jobs(state: &AppState, context: &str) {
    assert_eq!(
        state.training_queue.lock().unwrap().len(),
        0,
        "{context}: queue must stay empty"
    );
    assert!(
        state.training_jobs.read().unwrap().is_empty(),
        "{context}: tracking map must stay empty"
    );
}

fn sft_body(config: Value) -> Value {
    json!({
        "examples": [{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]}],
        "config": config,
    })
}

fn grpo_body(config: Value) -> Value {
    json!({
        "groups": [{
            "messages": [{"role": "user", "content": "hi"}],
            "completions": [
                {"text": "a", "reward": 1.0},
                {"text": "b", "reward": 0.0}
            ],
        }],
        "config": config,
    })
}

async fn assert_unsafe_scale(app: &axum::Router, path: &str, body: Value) {
    let (status, response) = post(app, path, body).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{path} must reject an unsafe alpha/rank pair, got {status}: {response}"
    );
    assert_eq!(
        response["error"]["code"], "training_invalid_request",
        "{path}: {response}"
    );
    let message = response["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("unsafe LoRA scaling"),
        "{path}: message must carry the trainer's wording, got: {message}"
    );
}

/// The exact pair the dashboard used to send: rank 8 with the server
/// default alpha (32) — ratio 4.0, rejected.
#[tokio::test]
async fn sft_rejects_rank_8_with_default_alpha_at_submit() {
    let state = make_state();
    let app = api::router(state.clone());
    assert_unsafe_scale(
        &app,
        "/v1/train/sft",
        sft_body(json!({"output_name": "fixes", "lora_rank": 8})),
    )
    .await;
    assert_no_jobs(&state, "unsafe sft scale");
}

#[tokio::test]
async fn grpo_and_opd_reject_unsafe_scale_at_submit() {
    let state = make_state();
    let app = api::router(state.clone());
    assert_unsafe_scale(
        &app,
        "/v1/train/grpo",
        grpo_body(json!({"output_name": "fixes", "lora_rank": 8})),
    )
    .await;
    assert_unsafe_scale(
        &app,
        "/v1/train/opd",
        json!({
            "prompts": [{"messages": [{"role": "user", "content": "hi"}]}],
            "teacher": "qwen3.6-27b@local",
            "config": {"output_name": "fixes", "lora_rank": 4, "lora_alpha": 32.0},
        }),
    )
    .await;
    assert_no_jobs(&state, "unsafe grpo/opd scale");
}

/// rank 8 + alpha 16 is ratio 2.0 — exactly the limit, allowed. Mock mode
/// can't actually train; reaching the mock-mode rejection proves the pair
/// passed the scale gate (same proof shape as the name-validation tests).
#[tokio::test]
async fn paired_alpha_passes_the_gate() {
    let app = api::router(make_state());
    let (status, response) = post(
        &app,
        "/v1/train/sft",
        sft_body(json!({"output_name": "fixes", "lora_rank": 8, "lora_alpha": 16.0})),
    )
    .await;
    assert_ne!(
        response["error"]["code"], "training_invalid_request",
        "rank 8 / alpha 16 must pass the scale gate: {response}"
    );
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "mock mode rejects training AFTER the scale gate: {response}"
    );
}

/// `allow_high_lora_scale` is the deliberate-experiment escape hatch — the
/// submit-time gate must honor it exactly like the trainer does.
#[tokio::test]
async fn allow_high_lora_scale_opts_out() {
    let app = api::router(make_state());
    let (status, response) = post(
        &app,
        "/v1/train/sft",
        sft_body(json!({
            "output_name": "fixes",
            "lora_rank": 8,
            "lora_alpha": 32.0,
            "allow_high_lora_scale": true,
        })),
    )
    .await;
    assert_ne!(
        response["error"]["code"], "training_invalid_request",
        "allow_high_lora_scale must bypass the gate: {response}"
    );
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
}

#[tokio::test]
async fn distill_endpoints_reject_unsafe_scale() {
    let state = make_state();
    let app = api::router(state.clone());
    assert_unsafe_scale(
        &app,
        "/v1/adapters/distill_merge",
        json!({
            "name": "merged",
            "sources": [
                {"adapter": "a", "weight": 1.0},
                {"adapter": "b", "weight": 1.0}
            ],
            "config": {"lora_rank": 4, "lora_alpha": 32.0},
        }),
    )
    .await;
    assert_unsafe_scale(
        &app,
        "/v1/distill/self",
        json!({
            "name": "selfy",
            "mode": "conciseness",
            "config": {"lora_rank": 4, "lora_alpha": 32.0},
        }),
    )
    .await;
    assert_no_jobs(&state, "unsafe distill scale");
}
