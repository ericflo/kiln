//! Integration tests: the §10 pi-session→training bridge.
//!
//! `/v1/agent/self_improve` and `/v1/agent/judge_distill` resolve their
//! corpora from the §10.3 agent-trace index only after static backend
//! admission succeeds. These integration tests pin that cheap admission
//! ordering in mock mode; the corpus builders' unit tests pin trace
//! resolution and the discover remediation used with a real backend.

use std::collections::{BTreeMap, HashMap};
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
use kiln_server::api::agent_traces::{AgentTrace, TraceOutcome};
use kiln_server::state::AppState;
use kiln_train::ChatMessage;
use kiln_train::trajectory::{TurnKind, TurnSegment};

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

async fn register_fixture_teacher(app: &axum::Router, alias: &str) {
    let (status, response) = post(
        app,
        "/v1/teachers",
        json!({"alias": alias, "kind": "fixture", "model_id": "test-model"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "teacher registration: {response}");
}

/// A fresh, successful trace with a real prompt scaffold and two actions.
fn recent_successful_trace(id: &str) -> AgentTrace {
    let now = chrono::Utc::now();
    AgentTrace {
        id: id.to_string(),
        working_dir: "/home/user/proj".to_string(),
        num_turns: 4,
        num_tool_calls: 1,
        outcome: TraceOutcome {
            ended_with_exit_0: Some(true),
            user_edited_agent_files: Vec::new(),
            has_followup_attempt: Some(false),
        },
        first_event_at: Some(now.to_rfc3339()),
        last_event_at: Some(now.to_rfc3339()),
        forked: false,
        parent_id: None,
        tool_manifest_sha: None,
        prompt_messages: vec![
            ChatMessage::new("system", "You are pi."),
            ChatMessage::new("user", format!("Fix the flaky test in {id}")),
        ],
        trajectory: vec![
            TurnSegment {
                role: "assistant".into(),
                content: "Reading the test.".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "tool".into(),
                content: "FAILED tests/flaky.rs".into(),
                kind: TurnKind::Observation,
                tool_call_id: None,
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "assistant".into(),
                content: "Pinning the seed.".into(),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ],
    }
}

fn write_trace_index(state: &AppState, traces: &[AgentTrace]) {
    let map: BTreeMap<String, AgentTrace> =
        traces.iter().map(|t| (t.id.clone(), t.clone())).collect();
    std::fs::write(
        state.adapter_dir.join("agent_traces.json"),
        serde_json::to_vec_pretty(&map).unwrap(),
    )
    .unwrap();
}

fn assert_no_jobs(state: &AppState, context: &str) {
    assert_eq!(
        state.training_queue.lock().unwrap().len(),
        0,
        "{context}: queue must stay empty"
    );
}

// ── self_improve ────────────────────────────────────────────────────

#[tokio::test]
async fn self_improve_without_trace_index_stops_before_discovery_in_mock_mode() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_fixture_teacher(&app, "judge-pi-v1").await;

    let (status, response) = post(&app, "/v1/agent/self_improve", json!({})).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
    assert_eq!(response["error"]["code"], "mock_mode", "{response}");
    assert!(
        !state.adapter_dir.join("agent_traces.json").exists(),
        "backend admission must precede trace discovery"
    );
    assert_no_jobs(&state, "self_improve without index");
}

#[tokio::test]
async fn self_improve_with_traces_stops_at_backend_admission() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_fixture_teacher(&app, "judge-pi-v1").await;
    write_trace_index(
        &state,
        &[
            recent_successful_trace("session-a"),
            recent_successful_trace("session-b"),
        ],
    );

    let (status, response) = post(&app, "/v1/agent/self_improve", json!({})).await;

    // The cheap backend gate runs before corpus resolution. The queued-job
    // contents are pinned by build_self_improve_jobs unit tests.
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
    assert_eq!(response["error"]["code"], "mock_mode", "{response}");
    assert_no_jobs(&state, "self_improve in mock mode");
}

// ── judge_distill ───────────────────────────────────────────────────

#[tokio::test]
async fn judge_distill_without_trace_index_stops_before_discovery_in_mock_mode() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_fixture_teacher(&app, "qwen3.6-27b@local").await;

    let (status, response) = post(&app, "/v1/agent/judge_distill", json!({})).await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
    assert_eq!(response["error"]["code"], "mock_mode", "{response}");
    assert!(
        !state.adapter_dir.join("agent_traces.json").exists(),
        "backend admission must precede trace discovery"
    );
    assert_no_jobs(&state, "judge_distill without index");
}

#[tokio::test]
async fn judge_distill_with_traces_stops_at_backend_admission() {
    let (state, _dir) = make_state();
    let app = api::router(state.clone());
    register_fixture_teacher(&app, "qwen3.6-27b@local").await;
    write_trace_index(&state, &[recent_successful_trace("session-a")]);

    let (status, response) = post(&app, "/v1/agent/judge_distill", json!({})).await;

    // Corpus contents and missing-index remediation are pinned by the
    // build_judge_pump_request unit tests in api/self_improve.rs.
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{response}");
    assert_eq!(response["error"]["code"], "mock_mode", "{response}");
    assert_no_jobs(&state, "judge_distill in mock mode");
}
