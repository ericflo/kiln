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

/// State whose adapter_dir points into a tempdir so teacher-registry
/// persistence (POST /v1/teachers) never writes into the repo checkout.
fn make_state_with_dir() -> (AppState, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut state = make_state();
    state.adapter_dir = dir.path().to_path_buf();
    (state, dir)
}

/// Seed a minimal agent-trace index so the §10.6 agent endpoints get past
/// corpus resolution and exercise the gate under test (queue cap / mock
/// mode) instead of the missing-index 400.
fn seed_trace_index(state: &AppState) {
    let now = chrono::Utc::now().to_rfc3339();
    let trace = kiln_server::api::agent_traces::AgentTrace {
        id: "session-a".into(),
        working_dir: "/home/user/proj".into(),
        num_turns: 2,
        num_tool_calls: 0,
        outcome: kiln_server::api::agent_traces::TraceOutcome {
            ended_with_exit_0: Some(true),
            user_edited_agent_files: Vec::new(),
            has_followup_attempt: Some(false),
        },
        first_event_at: Some(now.clone()),
        last_event_at: Some(now),
        forked: false,
        parent_id: None,
        tool_manifest_sha: None,
        prompt_messages: vec![kiln_train::ChatMessage {
            role: "user".into(),
            content: "Fix the failing test".into(),
        }],
        trajectory: vec![kiln_train::trajectory::TurnSegment {
            role: "assistant".into(),
            content: "On it.".into(),
            kind: kiln_train::trajectory::TurnKind::Action,
            tool_call_id: None,
            warning_prefix_len: None,
        }],
    };
    let mut map = std::collections::BTreeMap::new();
    map.insert(trace.id.clone(), trace);
    std::fs::write(
        state.adapter_dir.join("agent_traces.json"),
        serde_json::to_vec_pretty(&map).unwrap(),
    )
    .unwrap();
}

fn seed_merge_sources(state: &AppState) {
    for name in ["a", "b"] {
        std::fs::create_dir_all(state.adapter_dir.join(name)).unwrap();
    }
}

/// Register a fixture teacher alias through the API so endpoints with
/// teacher-resolvability validation get past that gate.
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

// ── The four endpoints #1484 missed ─────────────────────────────────

/// All six `FrontDoorRequest` variants for `POST /v1/train`, each
/// carrying `name` as the (eventual) output adapter name.
fn front_door_bodies(name: &str) -> Vec<(&'static str, Value)> {
    vec![
        (
            "sft",
            json!({"kind": "sft", "examples": sft_body(name)["examples"], "config": {"output_name": name}}),
        ),
        (
            "grpo",
            json!({"kind": "grpo", "groups": grpo_body(name)["groups"], "config": {"output_name": name}}),
        ),
        (
            "opd",
            json!({"kind": "opd", "teacher": "fixture@t", "prompts": opd_body(name)["prompts"], "config": {"output_name": name}}),
        ),
        (
            "distill_refresh",
            json!({"kind": "distill_refresh", "name": name, "new_data": {"dataset": "q4"}, "behavioural_teacher": "fixture@t"}),
        ),
        (
            "distill_merge",
            json!({"kind": "distill_merge", "name": name, "sources": [{"adapter": "a"}, {"adapter": "b"}]}),
        ),
        (
            "distill_pump",
            json!({"kind": "distill_pump", "name": name, "teacher": "fixture@t", "mode": {"domain": "math_reasoning"}}),
        ),
    ]
}

fn inline_recipe_body(step_name: &str) -> Value {
    json!({
        "body": {
            "name": "test-recipe",
            "steps": [{
                "kind": "sft",
                "name": step_name,
                "examples_from": {"examples": sft_body("x")["examples"]},
            }],
        },
    })
}

#[tokio::test]
async fn front_door_rejects_unsafe_names_across_all_variants() {
    let state = make_state();
    let app = api::router(state.clone());
    for name in BAD_NAMES {
        for (variant, body) in front_door_bodies(name) {
            assert_invalid_name(&app, "/v1/train", body, &format!("{variant}:{name}")).await;
        }
    }
    assert_no_jobs(&state, "/v1/train hostile names");
}

#[tokio::test]
async fn recipe_run_rejects_unsafe_step_names() {
    let state = make_state();
    let app = api::router(state.clone());
    for name in BAD_NAMES {
        assert_invalid_name(&app, "/v1/recipes/run", inline_recipe_body(name), name).await;
    }
    assert_no_jobs(&state, "/v1/recipes/run hostile names");
}

/// A hostile name on a LATER step must reject the whole recipe before
/// the earlier (valid) steps are enqueued.
#[tokio::test]
async fn recipe_run_rejects_whole_recipe_when_any_step_name_is_unsafe() {
    let state = make_state();
    let app = api::router(state.clone());
    let body = json!({
        "body": {
            "name": "two-step",
            "steps": [
                {"kind": "sft", "name": "good-step",
                 "examples_from": {"examples": sft_body("x")["examples"]}},
                {"kind": "sft", "name": "../evil",
                 "examples_from": {"examples": sft_body("x")["examples"]}},
            ],
        },
    });
    assert_invalid_name(&app, "/v1/recipes/run", body, "../evil (step 2)").await;
    assert_no_jobs(&state, "/v1/recipes/run partial recipe");
}

#[tokio::test]
async fn agent_endpoints_reject_unsafe_names() {
    let state = make_state();
    let app = api::router(state.clone());
    for name in BAD_NAMES {
        assert_invalid_name(&app, "/v1/agent/judge_distill", json!({"name": name}), name).await;
        assert_invalid_name(&app, "/v1/agent/self_improve", json!({"agent": name}), name).await;
    }
    assert_no_jobs(&state, "agent endpoints hostile names");
}

// ── Queue caps + mock-mode on the same four endpoints ───────────────

async fn assert_error_code(
    app: &axum::Router,
    path: &str,
    body: Value,
    status: StatusCode,
    code: &str,
) {
    let (got_status, response) = post(app, path, body).await;
    assert_eq!(got_status, status, "{path}: {response}");
    assert_eq!(response["error"]["code"], code, "{path}: {response}");
}

#[tokio::test]
async fn missed_endpoints_enforce_queue_cap() {
    let (mut state, _dir) = make_state_with_dir();
    state.max_queued_training_jobs = 0; // queue is "at cap" immediately
    let app = api::router(state.clone());
    register_teacher(&app, "fixture@t").await;
    seed_trace_index(&state);
    seed_merge_sources(&state);

    let full = StatusCode::SERVICE_UNAVAILABLE;
    let fd_sft = front_door_bodies("ok-name").remove(0).1;
    assert_error_code(&app, "/v1/train", fd_sft, full, "training_queue_full").await;
    assert_error_code(
        &app,
        "/v1/recipes/run",
        inline_recipe_body("ok-step"),
        full,
        "training_queue_full",
    )
    .await;
    assert_error_code(
        &app,
        "/v1/agent/judge_distill",
        json!({"name": "judge-ok", "teacher": "fixture@t"}),
        full,
        "training_queue_full",
    )
    .await;
    assert_error_code(
        &app,
        "/v1/agent/self_improve",
        json!({"agent": "agent-ok", "judge": "fixture@t"}),
        full,
        "training_queue_full",
    )
    .await;
    assert_no_jobs(&state, "queue at cap");
}

#[tokio::test]
async fn missed_endpoints_reject_mock_mode() {
    let (state, _dir) = make_state_with_dir();
    let app = api::router(state.clone());
    register_teacher(&app, "fixture@t").await;
    seed_trace_index(&state);
    seed_merge_sources(&state);

    let unavailable = StatusCode::SERVICE_UNAVAILABLE;
    for (variant, body) in front_door_bodies("ok-name") {
        let (status, response) = post(&app, "/v1/train", body).await;
        assert_eq!(status, unavailable, "{variant}: {response}");
        assert_eq!(
            response["error"]["code"], "mock_mode",
            "{variant}: {response}"
        );
    }
    assert_error_code(
        &app,
        "/v1/recipes/run",
        inline_recipe_body("ok-step"),
        unavailable,
        "mock_mode",
    )
    .await;
    assert_error_code(
        &app,
        "/v1/agent/judge_distill",
        json!({"name": "judge-ok", "teacher": "fixture@t"}),
        unavailable,
        "mock_mode",
    )
    .await;
    assert_error_code(
        &app,
        "/v1/agent/self_improve",
        json!({"agent": "agent-ok", "judge": "fixture@t"}),
        unavailable,
        "mock_mode",
    )
    .await;
    assert_no_jobs(&state, "mock mode");
}
