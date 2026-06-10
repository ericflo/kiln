//! Integration test: training-submission adapter names are validated.
//!
//! `output_name` (SFT/GRPO/OPD) and `name` (distill endpoints) become
//! directory names under `adapter_dir`. Unvalidated, `../evil` escapes the
//! adapter root and `.composed` collides with kiln's reserved internals —
//! and the resulting adapters can't be managed by the (validated) adapter
//! API. Every submission endpoint must reject unsafe names with 400
//! `invalid_adapter_name` before any queue/registration work happens.

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

const BAD_NAMES: &[&str] = &["../evil", "a/b", "/abs", ".composed", "..", "a\\b"];

fn sft_body(name: &str) -> Value {
    json!({
        "examples": [{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]}],
        "config": {"output_name": name},
    })
}

fn grpo_body(name: &str) -> Value {
    json!({
        "groups": [{
            "messages": [{"role": "user", "content": "hi"}],
            "completions": [
                {"text": "a", "reward": 1.0},
                {"text": "b", "reward": 0.0}
            ],
        }],
        "config": {"output_name": name},
    })
}

fn opd_body(name: &str) -> Value {
    json!({
        "prompts": [{"messages": [{"role": "user", "content": "hi"}]}],
        "teacher": "qwen3.6-27b@local",
        "config": {"output_name": name},
    })
}

async fn assert_invalid_name(app: &axum::Router, path: &str, body: Value, name: &str) {
    let (status, response) = post(app, path, body).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "{path} must reject name {name:?}, got {status}: {response}"
    );
    assert_eq!(
        response["error"]["code"], "invalid_adapter_name",
        "{path} name {name:?}: {response}"
    );
}

#[tokio::test]
async fn sft_grpo_opd_reject_unsafe_output_names() {
    let app = api::router(make_state());
    for name in BAD_NAMES {
        assert_invalid_name(&app, "/v1/train/sft", sft_body(name), name).await;
        assert_invalid_name(&app, "/v1/train/grpo", grpo_body(name), name).await;
        assert_invalid_name(&app, "/v1/train/opd", opd_body(name), name).await;
    }
}

#[tokio::test]
async fn distill_endpoints_reject_unsafe_names() {
    let app = api::router(make_state());
    for name in BAD_NAMES {
        assert_invalid_name(
            &app,
            "/v1/adapters/distill_merge",
            json!({
                "name": name,
                "sources": [
                    {"adapter": "a", "weight": 1.0},
                    {"adapter": "b", "weight": 1.0}
                ],
            }),
            name,
        )
        .await;
        assert_invalid_name(
            &app,
            "/v1/distill/self",
            json!({ "name": name, "mode": "conciseness" }),
            name,
        )
        .await;
    }
}

#[tokio::test]
async fn valid_output_name_proceeds_past_validation() {
    // Mock mode can't actually train; reaching the mock-mode rejection (or
    // any non-validation error) proves the name itself was accepted.
    let app = api::router(make_state());
    let (status, response) = post(&app, "/v1/train/sft", sft_body("my-adapter_v2")).await;
    assert_ne!(
        response["error"]["code"], "invalid_adapter_name",
        "safe name must pass validation: {response}"
    );
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "mock mode rejects training AFTER validation: {response}"
    );
}
