//! Integration tests: §10.6 agent endpoints fail fast on unresolvable
//! teachers/judges instead of enqueueing jobs that are guaranteed to die
//! at resolution time, and `judge_drift_check` validates its inputs and
//! returns an honest 501 instead of fake success (#31).

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

/// Mock state with adapter_dir on a tempdir (drift-check probes it for the
/// judge LoRA; teacher registration persists `teachers.json` into it).
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

async fn register_teacher(app: &axum::Router, alias: &str) {
    let (status, response) = post(
        app,
        "/v1/teachers",
        json!({"alias": alias, "kind": "fixture", "model_id": "test-model"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "teacher registration: {response}");
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

// ── judge_distill / self_improve teacher resolvability ──────────────

#[tokio::test]
async fn judge_distill_unregistered_teacher_fails_fast() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    let (status, response) = post(
        &app,
        "/v1/agent/judge_distill",
        json!({"name": "judge-x", "teacher": "missing@alias"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(
        response["error"]["code"], "teacher_not_registered",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("missing@alias"),
        "must name the alias: {message}"
    );
    let hint = response["error"]["hint"].as_str().unwrap();
    assert!(
        hint.contains("/v1/teachers"),
        "must show remediation: {hint}"
    );
    assert_no_jobs(&state, "judge_distill unregistered teacher");
}

#[tokio::test]
async fn judge_distill_error_lists_registered_aliases() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_teacher(&app, "other@t").await;
    let (status, response) = post(
        &app,
        "/v1/agent/judge_distill",
        json!({"name": "judge-x", "teacher": "missing@alias"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("other@t"),
        "must list registered aliases: {message}"
    );
    assert_no_jobs(&state, "judge_distill alias listing");
}

/// With the teacher registered, the request gets past resolvability and
/// hits the next gate — in mock mode that's the mock-training rejection,
/// which proves the teacher check no longer blocks valid submissions.
#[tokio::test]
async fn judge_distill_registered_teacher_proceeds_past_resolution() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_teacher(&app, "fixture@t").await;
    // Isolate from the developer's real ~/.pi sessions: auto-discovery
    // (ensure_agent_trace_index) would otherwise build a populated index
    // and sail past the corpus-resolution 400 this test asserts.
    unsafe { std::env::set_var("KILN_PI_SESSIONS_DIR", "/nonexistent/pi/sessions") };
    let (status, response) = post(
        &app,
        "/v1/agent/judge_distill",
        json!({"name": "judge-x", "teacher": "fixture@t"}),
    )
    .await;
    assert_ne!(
        response["error"]["code"], "teacher_not_registered",
        "registered teacher must pass resolution: {response}"
    );
    // Past the teacher gate, the next stop is §10.6.1 corpus resolution —
    // with no trace index on this fresh state that's the actionable 400.
    // (agent_traces_bridge.rs covers the populated-index path.)
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("agent-trace index"), "{message}");
}

#[tokio::test]
async fn self_improve_unregistered_judge_fails_fast() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    let (status, response) = post(
        &app,
        "/v1/agent/self_improve",
        json!({"agent": "pi-coder", "judge": "judge-nope"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(
        response["error"]["code"], "teacher_not_registered",
        "{response}"
    );
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("judge-nope"),
        "must name the judge: {message}"
    );
    assert!(
        message.contains("must be registered as a teacher alias"),
        "must explain the judge-as-teacher requirement: {message}"
    );
    assert_no_jobs(&state, "self_improve unregistered judge");
}

#[tokio::test]
async fn self_improve_registered_judge_proceeds_past_resolution() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_teacher(&app, "judge-pi-v1").await;
    unsafe { std::env::set_var("KILN_PI_SESSIONS_DIR", "/nonexistent/pi/sessions") };
    let (status, response) = post(
        &app,
        "/v1/agent/self_improve",
        json!({"agent": "pi-coder", "judge": "judge-pi-v1"}),
    )
    .await;
    assert_ne!(
        response["error"]["code"], "teacher_not_registered",
        "registered judge must pass resolution: {response}"
    );
    // Past the judge gate, the next stop is §10.6.2 task resolution —
    // with no trace index on this fresh state that's the actionable 400.
    // (agent_traces_bridge.rs covers the populated-index path.)
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("agent-trace index"), "{message}");
}

// ── judge_drift_check honesty ────────────────────────────────────────

#[tokio::test]
async fn drift_check_missing_judge_dir_is_404() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    let (status, response) = post(
        &app,
        "/v1/agent/judge_drift_check",
        json!({"judge": "no-such-judge", "teacher": "t@x"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{response}");
    assert_eq!(response["error"]["code"], "adapter_not_found", "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("no-such-judge"),
        "must name the missing judge path: {message}"
    );
}

#[tokio::test]
async fn drift_check_unregistered_teacher_is_400_with_remediation() {
    let (state, _dir) = make_state();
    std::fs::create_dir(state.adapter_dir.join("judge-pi-v1")).unwrap();
    let app = api::router(state.clone());
    let (status, response) = post(
        &app,
        "/v1/agent/judge_drift_check",
        json!({"judge": "judge-pi-v1", "teacher": "missing@alias"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(
        response["error"]["code"], "teacher_not_registered",
        "{response}"
    );
    let hint = response["error"]["hint"].as_str().unwrap();
    assert!(
        hint.contains("/v1/teachers"),
        "must show remediation: {hint}"
    );
}

#[tokio::test]
async fn drift_check_validates_bounds() {
    let (state, _dir) = make_state();
    std::fs::create_dir(state.adapter_dir.join("judge-pi-v1")).unwrap();
    let app = api::router(state.clone());
    register_teacher(&app, "fixture@t").await;

    for body in [
        json!({"judge": "judge-pi-v1", "teacher": "fixture@t", "sample_size": 0}),
        json!({"judge": "judge-pi-v1", "teacher": "fixture@t", "agreement_threshold": 0.0}),
        json!({"judge": "judge-pi-v1", "teacher": "fixture@t", "agreement_threshold": 1.5}),
    ] {
        let (status, response) = post(&app, "/v1/agent/judge_drift_check", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {response}");
        assert_eq!(
            response["error"]["code"], "training_invalid_request",
            "{body}: {response}"
        );
    }
}

#[tokio::test]
async fn drift_check_valid_inputs_return_honest_501() {
    let (state, _dir) = make_state();
    std::fs::create_dir(state.adapter_dir.join("judge-pi-v1")).unwrap();
    let app = api::router(state.clone());
    register_teacher(&app, "fixture@t").await;
    let (status, response) = post(
        &app,
        "/v1/agent/judge_drift_check",
        json!({"judge": "judge-pi-v1", "teacher": "fixture@t"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{response}");
    assert_eq!(response["error"]["code"], "not_implemented", "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("#31"),
        "must reference issue #31: {message}"
    );
}

/// Hostile judge names must be rejected before the filesystem probe —
/// `adapter_dir.join("../..")` would otherwise escape the adapter root.
#[tokio::test]
async fn drift_check_rejects_path_traversal_judge_names() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    let (status, response) = post(
        &app,
        "/v1/agent/judge_drift_check",
        json!({"judge": "../..", "teacher": "t@x"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(
        response["error"]["code"], "invalid_adapter_name",
        "{response}"
    );
}

// ── teacher registration honesty (adapter-wearing + provider gates) ──

/// `adapter` only means something on Local teachers — and it must exist.
/// Both rejections happen at REGISTRATION, not at job dequeue hours later.
#[tokio::test]
async fn teacher_registration_validates_adapter_field() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());

    // adapter on a fixture teacher: rejected.
    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({"alias": "t@f", "kind": "fixture", "model_id": "m", "adapter": "judge-v1"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("kind=local"),
        "{response}"
    );

    // Local teacher wearing a missing adapter: rejected with the path.
    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({"alias": "self@local", "kind": "local", "model_id": "m", "adapter": "ghost"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("ghost"),
        "{response}"
    );

    // Local teacher wearing an adapter that exists: accepted, echoed back.
    std::fs::create_dir(state.adapter_dir.join("prior-self")).unwrap();
    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({"alias": "self@local", "kind": "local", "model_id": "m", "adapter": "prior-self"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["spec"]["adapter"], "prior-self", "{response}");
}

/// Remote teachers must declare the protocol explicitly. Unsupported and
/// legacy missing-provider registrations fail before they can queue work.
#[tokio::test]
async fn teacher_registration_rejects_unwired_remote_providers() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());

    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({"alias": "t@tgi", "kind": "remote", "provider": "tgi", "model_id": "m", "url": "https://tgi.example.com"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("vLLM"),
        "names the supported providers: {message}"
    );

    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({"alias": "t@legacy", "kind": "remote", "model_id": "m", "url": "https://vllm.example.com"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("explicit provider"),
        "{response}"
    );

    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({
            "alias": "t@missing-key",
            "kind": "remote",
            "provider": "vllm",
            "model_id": "m",
            "url": "https://vllm.example.com",
            "api_key_env": "KILN_TEST_REMOTE_KEY_INTENTIONALLY_UNSET"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not set"),
        "{response}"
    );

    // An explicit vLLM registration is accepted on any valid HTTP(S) URL.
    let (status, response) = post(
        &app,
        "/v1/teachers",
        json!({"alias": "t@vllm", "kind": "remote", "provider": "vllm", "model_id": "m", "url": "https://vllm.example.com"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
}

// ── OPD / distill submission fail-fast ───────────────────────────────

/// A typo'd teacher alias on /v1/train/opd fails at submission with the
/// registered list, before the mock/queue gates.
#[tokio::test]
async fn opd_submission_rejects_unregistered_teacher() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_teacher(&app, "real@t").await;
    let (status, response) = post(
        &app,
        "/v1/train/opd",
        json!({
            "prompts": [{"messages": [{"role": "user", "content": "hi"}]}],
            "teacher": "typo@t",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(
        response["error"]["code"], "teacher_not_registered",
        "{response}"
    );
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("typo@t"),
        "{response}"
    );
    assert_no_jobs(&state, "opd unregistered teacher");
}

/// dataset_path (pre-scored off-policy teacher JSONL) with the default
/// on-policy mode is a guaranteed worker failure — reject at submission
/// and point at the agent_traces alternative.
#[tokio::test]
async fn opd_submission_rejects_dataset_path_mode_mismatch() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_teacher(&app, "real@t").await;
    let (status, response) = post(
        &app,
        "/v1/train/opd",
        json!({
            "dataset_path": "/tmp/teacher.jsonl",
            "teacher": "real@t",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    let message = response["error"]["message"].as_str().unwrap();
    assert!(message.contains("off_policy"), "{message}");
    assert!(
        message.contains("agent_traces"),
        "offers the on-policy path: {message}"
    );
    assert_no_jobs(&state, "opd mode mismatch");
}

/// distill/pump with an unregistered teacher fails at submission.
#[tokio::test]
async fn distill_pump_rejects_unregistered_teacher() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    let (status, response) = post(
        &app,
        "/v1/distill/pump",
        json!({"name": "pumped", "teacher": "nope@t", "mode": {"domain": "math"}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(
        response["error"]["code"], "teacher_not_registered",
        "{response}"
    );
    assert_no_jobs(&state, "pump unregistered teacher");
}

/// distill_merge with a source adapter that doesn't exist on disk is a
/// typo — fail at submission instead of silently distilling that
/// source's prompts from the base model after dequeue.
#[tokio::test]
async fn distill_merge_rejects_missing_source_adapters() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    std::fs::create_dir(state.adapter_dir.join("exists")).unwrap();
    let (status, response) = post(
        &app,
        "/v1/adapters/distill_merge",
        json!({
            "name": "merged",
            "sources": [
                {"adapter": "exists", "weight": 1.0},
                {"adapter": "missing", "weight": 1.0}
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("missing"),
        "{response}"
    );
    assert_no_jobs(&state, "merge missing source");
}

#[tokio::test]
async fn front_door_rejects_structurally_empty_opd_before_enqueue() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_teacher(&app, "real@t").await;

    let (status, response) = post(
        &app,
        "/v1/train",
        json!({"kind": "opd", "prompts": [], "teacher": "real@t"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("exactly one"),
        "{response}"
    );
    assert_no_jobs(&state, "front-door empty OPD");
}

#[tokio::test]
async fn front_door_rejects_missing_merge_source_before_enqueue() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());

    let (status, response) = post(
        &app,
        "/v1/train",
        json!({
            "kind": "distill_merge",
            "name": "merged",
            "sources": [{"adapter": "missing"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("missing"),
        "{response}"
    );
    assert_no_jobs(&state, "front-door missing merge source");
}
