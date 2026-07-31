use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{Json, Router, extract::State, routing::post};
use kiln_server::openenv_cli::{OpenEnvRolloutOptions, OpenEnvTrainOptions, run_openenv_train};
use kiln_server::openenv_replay::{replay_openenv, verify_openenv_artifacts};
use kiln_train::OpenEnvEpisodeTerminationV1;
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct FakeKilnState {
    training_submission: Arc<Mutex<Option<Value>>>,
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
            environment_urls: vec![environment_url],
            adapter: "base".to_string(),
            groups: 1,
            group_size: 2,
            seed_start: 71,
            reset_options: None,
            reset_options_value: None,
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

    assert_eq!(summary.schema, "kiln.openenv-rollout-summary.v2");
    assert_eq!(summary.rollout_count, 2);
    assert_eq!(summary.stats.recoverable_protocol_error_count, 2);
    assert_eq!(summary.stats.protocol_error_count, 0);
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
    assert_eq!(
        submitted["config"]["behavior_policy"],
        "no_importance_correction"
    );
    assert_eq!(submitted["config"]["output_name"], "bandit-e2e");
    assert_eq!(submitted["groups"].as_array().unwrap().len(), 1);
    assert!(
        submitted["groups"][0]["completions"]
            .as_array()
            .unwrap()
            .iter()
            .all(|rollout| rollout["openenv"]["environment_name"] == "BanditEnvironment")
    );

    let verified = verify_openenv_artifacts(&summary_path, None, None).unwrap();
    assert_eq!(verified.report.rollouts, 2);
    assert_eq!(verified.report.environment_exchanges, 4);
    let replay_report = replay_openenv(
        &verified.replay,
        verified.report.replay_sha256,
        2,
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert_eq!(replay_report.rollouts, 2);
    assert_eq!(replay_report.environment_exchanges, 4);
    assert!(replay_report.capacity_retries >= 1);
    assert_eq!(replay_report.environment_prefix_only_rollouts, 0);

    server.abort();
}

async fn fake_chat(Json(body): Json<Value>) -> Json<Value> {
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
        "choices": [{"message": {"role": "assistant", "content": action}}],
        "usage": {"total_tokens": 4}
    }))
}

async fn fake_train(State(state): State<FakeKilnState>, Json(body): Json<Value>) -> Json<Value> {
    *state.training_submission.lock().unwrap() = Some(body);
    Json(json!({
        "job_id": "openenv-e2e-job",
        "status": "queued"
    }))
}
