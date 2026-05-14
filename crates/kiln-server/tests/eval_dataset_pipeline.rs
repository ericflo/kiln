//! End-to-end test for the dataset → synthesis → judgment flywheel,
//! exercised through the live axum router. Uses the mock backend so it
//! runs on any host (no GPU required), and validates that the on-disk
//! state machine actually transitions the way the UI expects.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use kiln_core::config::ModelConfig;
use kiln_server::api;
use kiln_server::eval::{DatasetRegistry, JudgmentStore, SuiteRegistry};
use kiln_server::state::AppState;
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt;

fn build_state() -> (AppState, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = ModelConfig::qwen3_5_4b();
    let sched_config = kiln_scheduler::SchedulerConfig {
        max_batch_tokens: 8192,
        max_batch_size: 64,
        block_size: 16,
        prefix_cache_enabled: false,
        ..Default::default()
    };
    let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
    let engine = kiln_model::engine::MockEngine::new(config.clone());
    let tokenizer = {
        let json = br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}"#;
        kiln_core::tokenizer::KilnTokenizer::from_bytes(json).unwrap()
    };
    let mut state = AppState::new_mock(
        config,
        scheduler,
        Arc::new(engine),
        tokenizer,
        60,
        "kiln-test".into(),
    );
    state.dataset_registry = Some(Arc::new(DatasetRegistry::new(dir.path().join("datasets"))));
    state.suite_registry = Some(Arc::new(SuiteRegistry::new(dir.path().join("suites"))));
    state.judgment_store = Some(Arc::new(JudgmentStore::new(dir.path().join("judgments"))));
    std::fs::create_dir_all(dir.path().join("datasets")).unwrap();
    std::fs::create_dir_all(dir.path().join("suites")).unwrap();
    std::fs::create_dir_all(dir.path().join("judgments")).unwrap();
    (state, dir)
}

async fn json_call(router: &axum::Router, method: &str, path: &str, body: &serde_json::Value) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    let res = router.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 16 * 1024 * 1024).await.unwrap();
    let json: serde_json::Value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, json)
}

async fn get_json(router: &axum::Router, path: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder().uri(path).body(Body::empty()).unwrap();
    let res = router.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 16 * 1024 * 1024).await.unwrap();
    let json: serde_json::Value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, json)
}

/// Upload an SFT dataset by going around the multipart layer (which is hard
/// to mock cleanly) and writing directly to the registry. The HTTP
/// pipeline downstream (preview / synthesize / judgments) is what we want
/// to exercise.
fn seed_sft_dataset(state: &AppState, name: &str) {
    let reg = state.dataset_registry.as_ref().unwrap();
    let body = [
        serde_json::json!({"messages":[
            {"role":"user","content":"What's the capital of France?"},
            {"role":"assistant","content":"Paris"},
        ]}),
        serde_json::json!({"messages":[
            {"role":"user","content":"2+2?"},
            {"role":"assistant","content":"4"},
        ]}),
        serde_json::json!({"messages":[
            {"role":"system","content":"You are an agent. Use tools."},
            {"role":"user","content":"open /tmp/x.txt"},
            {"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x.txt\"}"}}]},
            {"role":"tool","content":"hello"},
            {"role":"assistant","content":"The file contains: hello"},
        ]}),
        serde_json::json!({"messages":[
            {"role":"user","content":"add 3 to 5 with python"},
            {"role":"assistant","content":"","tool_calls":[{"function":{"name":"bash","arguments":"{\"command\":\"python3 -c 'print(3+5)'\"}"}}]},
            {"role":"tool","content":"8"},
            {"role":"assistant","content":"3 + 5 = 8"},
        ]}),
    ]
    .iter()
    .map(|v| serde_json::to_string(v).unwrap())
    .collect::<Vec<_>>()
    .join("\n");
    reg.create(
        name,
        kiln_server::eval::DatasetFormat::SftChat,
        Some("seed dataset".into()),
        body.as_bytes(),
    )
    .unwrap();
}

#[tokio::test]
async fn dataset_listing_after_seed() {
    let (state, _dir) = build_state();
    seed_sft_dataset(&state, "smoke");
    let router = api::router(state);
    let (status, body) = get_json(&router, "/v1/eval/datasets").await;
    assert_eq!(status, StatusCode::OK);
    let names: Vec<String> = body["datasets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"smoke".to_string()));
}

#[tokio::test]
async fn synthesis_preview_round_trip() {
    let (state, _dir) = build_state();
    seed_sft_dataset(&state, "smoke");
    let router = api::router(state);
    let body = serde_json::json!({
        "suite_name": "smoke-preview",
        "strategy": "final_assistant",
        "scorer": {"kind": "auto_detect"},
        "sampling": {"max_examples": 10, "max_prompt_chars": 1000000, "max_target_chars": 100000, "seed": 7, "dedupe": true},
        "head_n": 2,
    });
    let (status, json) = json_call(&router, "POST", "/v1/eval/datasets/smoke/preview", &body).await;
    assert_eq!(status, StatusCode::OK, "body={json}");
    let examples = json["examples"].as_array().unwrap();
    assert!(!examples.is_empty(), "expected preview examples, got {examples:?}");
    let kind = examples[0]["scorer"]["kind"].as_str().unwrap();
    // First conversation has target "Paris" → exact_match auto-detect.
    assert!(matches!(kind, "exact_match" | "contains"), "kind was {kind}");
}

#[tokio::test]
async fn synthesize_persists_suite() {
    let (state, _dir) = build_state();
    seed_sft_dataset(&state, "smoke");
    let router = api::router(state);
    let body = serde_json::json!({
        "suite_name": "smoke-final",
        "strategy": "final_assistant",
        "scorer": {"kind": "auto_detect"},
        "sampling": {"max_examples": 10, "max_prompt_chars": 1000000, "max_target_chars": 100000, "seed": 7, "dedupe": true},
        "force": false,
    });
    let (status, json) = json_call(&router, "POST", "/v1/eval/datasets/smoke/synthesize", &body).await;
    assert_eq!(status, StatusCode::OK, "body={json}");
    let suite_name = json["suite"]["name"].as_str().unwrap();
    assert_eq!(suite_name, "smoke-final");
    // List the suites — the new one must show up.
    let (s2, list) = get_json(&router, "/v1/eval/suites").await;
    assert_eq!(s2, StatusCode::OK);
    let suites = list["suites"].as_array().unwrap();
    assert!(suites.iter().any(|s| s["name"].as_str() == Some("smoke-final")));
}

#[tokio::test]
async fn tool_call_strategy_produces_tool_call_scorer() {
    let (state, _dir) = build_state();
    seed_sft_dataset(&state, "agent");
    let router = api::router(state);
    let body = serde_json::json!({
        "suite_name": "agent-tools",
        "strategy": "tool_call_predict",
        "scorer": {"kind": "auto_detect"},
        "sampling": {"max_examples": 10, "max_prompt_chars": 1000000, "max_target_chars": 100000, "seed": 11, "dedupe": false},
        "head_n": 10,
    });
    let (status, preview) = json_call(&router, "POST", "/v1/eval/datasets/agent/preview", &body).await;
    assert_eq!(status, StatusCode::OK, "{preview}");
    // We seeded 2 assistant tool-call turns; the strategy must produce ≥1.
    let examples = preview["examples"].as_array().unwrap();
    assert!(!examples.is_empty(), "expected tool-call examples, got {examples:?}");
    // Auto-detect should classify them as tool_call.
    let has_tool_call_scorer = examples.iter().any(|ex| ex["scorer"]["kind"].as_str() == Some("tool_call"));
    assert!(has_tool_call_scorer, "expected at least one tool_call scorer, got {examples:?}");
}

#[tokio::test]
async fn judgment_flywheel_create_append_compile() {
    let (state, _dir) = build_state();
    let router = api::router(state.clone());
    // 1. Create a judgment dataset.
    let (status, body) = json_call(
        &router,
        "POST",
        "/v1/judgments",
        &serde_json::json!({"name": "prose-judge"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body={body}");
    assert_eq!(body["num_rows"].as_u64().unwrap(), 0);

    // 2. Append three judgments.
    for (i, winner) in [("a", "a"), ("b", "b"), ("tie", "tie")].iter().enumerate() {
        let body = serde_json::json!({
            "id": format!("j{i}"),
            "prompt": [{"role":"user", "content":"What's 2+2?"}],
            "adapter_a": null,
            "adapter_b": "v1",
            "response_a": "answer A",
            "response_b": "answer B",
            "winner": winner.1,
            "note": "test",
            "tags": ["prose"],
        });
        let (status, _) = json_call(&router, "POST", "/v1/judgments/prose-judge/rows", &body).await;
        assert_eq!(status, StatusCode::OK, "winner={} {}", winner.0, body);
    }

    // 3. List judgments — should see prose-judge with 3 rows.
    let (status, body) = get_json(&router, "/v1/judgments").await;
    assert_eq!(status, StatusCode::OK);
    let manifests = body["judgments"].as_array().unwrap();
    let prose = manifests
        .iter()
        .find(|m| m["name"] == "prose-judge")
        .expect("prose-judge exists");
    assert_eq!(prose["num_rows"].as_u64().unwrap(), 3);
    assert_eq!(prose["winner_histogram"]["a"].as_u64().unwrap(), 1);
    assert_eq!(prose["winner_histogram"]["b"].as_u64().unwrap(), 1);
    assert_eq!(prose["winner_histogram"]["tie"].as_u64().unwrap(), 1);

    // 4. Compile to SFT — produces a new dataset.
    let (status, compiled) = json_call(
        &router,
        "POST",
        "/v1/judgments/prose-judge/compile",
        &serde_json::json!({"output_dataset": "prose-judge-sft", "include_skips": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body={compiled}");
    assert_eq!(compiled["status"], "compiled");
    assert_eq!(compiled["rows"].as_u64().unwrap(), 3);
    assert_eq!(compiled["dataset"]["name"], "prose-judge-sft");
    assert_eq!(compiled["dataset"]["num_rows"].as_u64().unwrap(), 3);

    // 5. Remove one judgment — manifest reflects the deletion.
    let (status, after) = json_call(
        &router,
        "DELETE",
        "/v1/judgments/prose-judge/rows/j1",
        &serde_json::json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body={after}");
    assert_eq!(after["num_rows"].as_u64().unwrap(), 2);
}

#[tokio::test]
async fn render_prompt_endpoint_emits_winner_a_b_tie_template() {
    let (state, _dir) = build_state();
    let router = api::router(state);
    let body = serde_json::json!({
        "prompt": [{"role":"user", "content":"What's 2+2?"}],
        "response_a": "the answer is 4",
        "response_b": "I think it's 5",
        "winner": "a",
    });
    let (status, json) = json_call(&router, "POST", "/v1/judgments/render_prompt", &body).await;
    assert_eq!(status, StatusCode::OK);
    let prompt = json["prompt"].as_str().unwrap();
    assert!(prompt.contains("Reply A"));
    assert!(prompt.contains("Reply B"));
    assert!(prompt.contains("Winner: A"));
}
