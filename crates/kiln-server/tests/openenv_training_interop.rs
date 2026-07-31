use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use kiln_server::openenv_cli::{OpenEnvRolloutOptions, OpenEnvTrainOptions, run_openenv_train};
use kiln_server::openenv_replay::{replay_openenv, verify_openenv_artifacts};
use kiln_train::{GrpoConfig, OpenEnvEpisodeTerminationV1};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct FakeKilnState {
    training_preflight: Arc<Mutex<Option<Value>>>,
    training_submission: Arc<Mutex<Option<Value>>>,
    chat_requests: Arc<AtomicUsize>,
}

/// End-to-end proof that the native collector can learn from recoverable
/// OpenEnv feedback, queue through remote capacity, emit a verifiable bundle,
/// submit canonical GRPO, and replay the environment transcript exactly.
#[tokio::test]
#[ignore = "requires a live max-sessions=1 miniopenenv bandit server"]
async fn collects_submits_verifies_and_replays_a_real_arcade_batch() {
    let environment_url = std::env::var("KILN_OPENENV_INTEROP_BANDIT_URL")
        .expect("KILN_OPENENV_INTEROP_BANDIT_URL must identify the live bandit");
    let state = FakeKilnState::default();
    let app = Router::new()
        .route("/v1/openenv/training/preflight", post(fake_preflight))
        .route("/v1/chat/completions", post(fake_chat))
        .route("/v1/train/grpo", post(fake_train))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let directory = tempfile::tempdir().unwrap();
    let dataset = directory.path().join("rollouts.jsonl");
    let replay = directory.path().join("replay.json");
    let summary_path = directory.path().join("summary.json");
    let summary = run_openenv_train(OpenEnvTrainOptions {
        rollout: OpenEnvRolloutOptions {
            kiln_url: format!("http://{address}"),
            environment_urls: vec![environment_url.clone(), environment_url],
            credential_envs: Vec::new(),
            adapter: "base".to_string(),
            groups: 2,
            group_size: 2,
            seed_start: 71,
            reset_options: None,
            reset_options_value: None,
            environment_reset_options: Vec::new(),
            environment_reset_options_values: vec![
                serde_json::json!({"difficulty": "hard"}),
                serde_json::json!({"split": "train"}),
            ],
            max_steps: 2,
            concurrency: 2,
            max_action_tokens: 32,
            temperature: 0.0,
            thinking: false,
            protocol_error_reward: -1.0,
            max_recoverable_errors: 1,
            capacity_wait_seconds: 10,
            output: dataset.clone(),
            replay_output: replay.clone(),
            summary_output: summary_path.clone(),
        },
        output_adapter: "bandit-e2e".to_string(),
        lora_rank: Some(8),
        auto_load: true,
    })
    .await
    .unwrap();

    assert_eq!(summary.schema, "kiln.openenv-rollout-summary.v5");
    assert_eq!(
        serde_json::to_value(summary.behavior_policy.as_ref().unwrap()).unwrap(),
        fake_behavior_policy()
    );
    assert_eq!(summary.rollout_count, 4);
    assert_eq!(summary.stats.recoverable_protocol_error_count, 4);
    assert_eq!(summary.stats.protocol_error_count, 0);
    assert!(state.chat_requests.load(Ordering::Relaxed) > 0);
    assert!(
        summary.stats.capacity_retry_count >= 1,
        "max-sessions=1 must exercise capacity-aware acquisition"
    );
    assert!(summary.rollouts.iter().all(|record| {
        record.termination == OpenEnvEpisodeTerminationV1::MaxSteps
            && record.steps == 2
            && record.recoverable_protocol_errors == 1
    }));
    assert_eq!(
        state.training_preflight.lock().unwrap().as_ref().unwrap()["training_config"]["lora_rank"],
        8
    );
    assert_eq!(
        summary
            .training_submission
            .as_ref()
            .and_then(|value| value["job_id"].as_str()),
        Some("openenv-e2e-job")
    );

    let submitted = state
        .training_submission
        .lock()
        .unwrap()
        .clone()
        .expect("GRPO submission was captured");
    let preflight_request = state
        .training_preflight
        .lock()
        .unwrap()
        .clone()
        .expect("training preflight was captured");
    let submitted_config: GrpoConfig = serde_json::from_value(submitted["config"].clone()).unwrap();
    let preflight_config: GrpoConfig =
        serde_json::from_value(fake_effective_config(&preflight_request)).unwrap();
    assert_eq!(
        serde_json::to_value(submitted_config).unwrap(),
        serde_json::to_value(preflight_config).unwrap(),
        "the CLI must submit the complete server-owned effective config unchanged"
    );
    assert_eq!(
        submitted["config"]["behavior_policy"],
        "no_importance_correction"
    );
    assert_eq!(submitted["config"]["output_name"], "bandit-e2e");
    assert_eq!(
        serde_json::to_value(
            &summary
                .training_contract
                .as_ref()
                .expect("direct train summary must retain its admitted contract")
                .effective_config
        )
        .unwrap(),
        submitted["config"],
        "the artifact contract must match the exact submitted native config"
    );
    assert_eq!(submitted["groups"].as_array().unwrap().len(), 2);
    assert!(
        submitted["groups"][0]["completions"]
            .as_array()
            .unwrap()
            .iter()
            .all(|rollout| rollout["openenv"]["environment_name"] == "BanditEnvironment")
    );

    let verified = verify_openenv_artifacts(&summary_path, None, None).unwrap();
    assert_eq!(
        verified.replay.groups[0].reset_payload,
        serde_json::json!({"difficulty": "hard", "seed": 71})
    );
    assert_eq!(
        verified.replay.groups[1].reset_payload,
        serde_json::json!({"seed": 72, "split": "train"})
    );
    assert_eq!(verified.report.rollouts, 4);
    assert_eq!(verified.report.environment_exchanges, 8);

    let legacy_summary_path = directory.path().join("legacy-summary.json");
    let mut legacy_summary = summary.clone();
    legacy_summary.schema = "kiln.openenv-rollout-summary.v2".to_string();
    legacy_summary.reset_options_sha256 = Some(format!("sha256:{}", "0".repeat(64)));
    legacy_summary.reset_plan_sha256 = None;
    legacy_summary.training_contract = None;
    legacy_summary.training_submission = None;
    legacy_summary.behavior_policy = None;
    std::fs::write(
        &legacy_summary_path,
        serde_json::to_vec_pretty(&legacy_summary).unwrap(),
    )
    .unwrap();
    assert_eq!(
        verify_openenv_artifacts(&legacy_summary_path, None, None)
            .unwrap()
            .report
            .rollouts,
        4,
        "summary v2 bundles remain offline-verifiable after the v3 migration"
    );

    let replay_report = replay_openenv(
        &verified.replay,
        verified.report.replay_sha256,
        2,
        Duration::from_secs(10),
        &[],
    )
    .await
    .unwrap();
    assert_eq!(replay_report.rollouts, 4);
    assert_eq!(replay_report.environment_exchanges, 8);
    assert!(replay_report.capacity_retries >= 1);
    assert_eq!(replay_report.environment_prefix_only_rollouts, 0);

    server.abort();
}

#[tokio::test]
async fn direct_train_rejection_happens_before_environment_contact_or_artifacts() {
    let app = Router::new().route(
        "/v1/openenv/training/preflight",
        post(|| async {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "code": "training_invalid_request",
                        "message": "deliberate pre-collection rejection"
                    }
                })),
            )
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let directory = tempfile::tempdir().unwrap();
    let dataset = directory.path().join("must-not-exist.jsonl");
    let replay = directory.path().join("must-not-exist.replay.json");
    let summary = directory.path().join("must-not-exist.summary.json");
    let error = run_openenv_train(OpenEnvTrainOptions {
        rollout: OpenEnvRolloutOptions {
            kiln_url: format!("http://{address}"),
            environment_urls: vec!["http://127.0.0.1:9".to_string()],
            credential_envs: Vec::new(),
            adapter: "base".to_string(),
            groups: 1,
            group_size: 1,
            seed_start: 0,
            reset_options: None,
            reset_options_value: None,
            environment_reset_options: Vec::new(),
            environment_reset_options_values: Vec::new(),
            max_steps: 1,
            concurrency: 1,
            max_action_tokens: 16,
            temperature: 0.0,
            thinking: false,
            protocol_error_reward: -1.0,
            max_recoverable_errors: 0,
            capacity_wait_seconds: 1,
            output: dataset.clone(),
            replay_output: replay.clone(),
            summary_output: summary.clone(),
        },
        output_adapter: "rejected-agent".to_string(),
        lora_rank: Some(8),
        auto_load: true,
    })
    .await
    .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("before collection"), "{message}");
    assert!(
        message.contains("deliberate pre-collection rejection"),
        "{message}"
    );
    assert!(!dataset.exists());
    assert!(!replay.exists());
    assert!(!summary.exists());
    server.abort();
}

async fn fake_preflight(
    State(state): State<FakeKilnState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    *state.training_preflight.lock().unwrap() = Some(body.clone());
    let config = fake_effective_config(&body);
    let mut receipt = json!({
        "schema": "kiln.openenv-training-preflight.v1",
        "effective_config": config,
        "behavior_policy": fake_behavior_policy(),
        "capacity": {
            "checked_unix_ms": 1,
            "queued_jobs": 0,
            "max_queued_jobs": 8,
            "tracked_jobs": 0,
            "max_tracked_jobs": 32
        },
        "capacity_reserved": false
    });
    if let Some(post_eval) = body.get("post_eval") {
        receipt["post_eval"] = post_eval.clone();
    }
    Json(receipt)
}

fn fake_effective_config(body: &Value) -> Value {
    let mut config = body["training_config"].clone();
    config["output_name"] = body["output_adapter"].clone();
    config["auto_load"] = body["auto_load"].clone();
    config["behavior_policy"] = Value::from("no_importance_correction");
    config["base_adapter"] = Value::Null;
    config
}

fn fake_behavior_policy() -> Value {
    json!({
        "served_model_id": "fake-kiln-model",
        "base_model_sha256": format!("sha256:{}", "a".repeat(64)),
        "inference_config_sha256": format!("sha256:{}", "b".repeat(64)),
        "implementation": "kiln/fake-openenv-test"
    })
}

async fn fake_chat(State(state): State<FakeKilnState>, Json(body): Json<Value>) -> Json<Value> {
    assert!(
        state.training_preflight.lock().unwrap().is_some(),
        "OpenEnv training must preflight before its first policy request"
    );
    state.chat_requests.fetch_add(1, Ordering::Relaxed);
    assert_eq!(body["rollout_provenance"], true);
    assert!(
        body["messages"].as_array().is_some_and(|messages| {
            messages.iter().any(|message| {
                message["content"].as_str().is_some_and(|content| {
                    content.contains("OpenEnv input_text")
                        && content.contains("Reply with one digit, the arm to pull")
                        && content.contains("\"pulls\":0")
                })
            })
        }),
        "the generic policy prompt must foreground input_text and retain the complete wire observation"
    );
    let recovering = body["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|message| {
            message["content"]
                .as_str()
                .is_some_and(|content| content.contains("\"openenv_error\""))
        })
    });
    let action = if recovering {
        r#"{"arm":0}"#
    } else {
        r#"{"arm":99}"#
    };
    Json(json!({
        "choices": [{
            "message": {"role": "assistant", "content": action},
            "rollout_provenance": {"behavior_policy": fake_behavior_policy()}
        }],
        "usage": {"total_tokens": 4}
    }))
}

async fn fake_train(State(state): State<FakeKilnState>, Json(body): Json<Value>) -> Json<Value> {
    assert!(state.training_preflight.lock().unwrap().is_some());
    *state.training_submission.lock().unwrap() = Some(body);
    Json(json!({
        "job_id": "openenv-e2e-job",
        "status": "queued"
    }))
}
