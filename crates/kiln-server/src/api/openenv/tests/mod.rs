use super::*;
use crate::eval::queue::{EvalJobInfo, EvalSubmissionKind};
use crate::state::{TrainingJobInfo, TrainingJobType};
use axum::body::Body;
use axum::extract::Path as AxumPath;
use axum::http::Request;
use axum::response::Response as AxumResponse;
use axum::routing::{get as axum_get, post as axum_post};
use kiln_core::config::ModelConfig;
use kiln_model::engine::MockEngine;
use kiln_scheduler::{Scheduler, SchedulerConfig};
use serde_json::json;
use tower::ServiceExt;

fn test_state(temp: &tempfile::TempDir, policy: OpenEnvConfig) -> AppState {
    let model_config = ModelConfig::qwen3_5_4b();
    let scheduler = Scheduler::new(SchedulerConfig::default(), 256);
    let mut state = AppState::new_mock(
        model_config.clone(),
        scheduler,
        Arc::new(MockEngine::new(model_config)),
        crate::api::test_tokenizer(),
        300,
        "Qwen3.5-4B".to_string(),
    );
    state.openenv_runs =
        Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap());
    state
}

fn request(kind: OpenEnvRunKind) -> OpenEnvRunRequest {
    OpenEnvRunRequest {
        kind,
        idempotency_key: None,
        environment_urls: vec!["http://127.0.0.1:8000".into()],
        credential_ids: Vec::new(),
        adapter: "base".into(),
        groups: 2,
        group_size: 3,
        seed_start: 0,
        reset_options: default_reset_options(),
        environment_reset_options: Vec::new(),
        max_steps: 8,
        concurrency: 2,
        max_action_tokens: 128,
        thinking_budget_tokens: None,
        temperature: 1.0,
        thinking: false,
        protocol_error_reward: -1.0,
        max_recoverable_errors: 3,
        capacity_wait_seconds: 30,
        output_adapter: (kind == OpenEnvRunKind::Train).then(|| "agent".into()),
        training_config: None,
        auto_load: true,
        post_eval: None,
        environment_eval: None,
    }
}

#[test]
fn openenv_run_request_defaults_to_thinking_trajectories() {
    let request: OpenEnvRunRequest = serde_json::from_value(json!({
        "environment_urls": ["http://127.0.0.1:8000"]
    }))
    .unwrap();
    assert!(request.thinking);
    assert_eq!(request.thinking_budget_tokens, None);
}

fn insert_created(
    registry: &OpenEnvRunRegistry,
    request: OpenEnvRunRequest,
) -> (OpenEnvRunStatus, OpenEnvRunControl) {
    match insert_test_run(registry, request).unwrap() {
        OpenEnvRunInsertOutcome::Created { status, control } => (status, control),
        OpenEnvRunInsertOutcome::Replayed(_) => {
            panic!("test expected a newly created OpenEnv run")
        }
    }
}

fn insert_test_run(
    registry: &OpenEnvRunRegistry,
    request: OpenEnvRunRequest,
) -> Result<OpenEnvRunInsertOutcome> {
    let training_contract = materialized_openenv_training_contract(&request)?;
    registry.insert(request, training_contract)
}

fn test_training_data() -> TrainingDataProvenance {
    let groups = (0..2)
        .map(|group_index| {
            let provenance = kiln_train::OpenEnvRolloutProvenanceV1::new(
                "CounterEnvironment",
                "http://127.0.0.1:8000",
                Some("1.0".into()),
                format!("sha256:{}", "d".repeat(64)),
                format!("sha256:{}", "a".repeat(64)),
                format!("sha256:{}", "b".repeat(64)),
                format!("sha256:{}", "c".repeat(64)),
                17 + group_index,
                1,
                1.0,
                true,
                kiln_train::OpenEnvEpisodeTerminationV1::Done,
                None,
            )
            .unwrap();
            kiln_train::AgenticGroup {
                messages: Vec::new(),
                completions: (0..3)
                    .map(|_| {
                        kiln_train::ScoredRollout::legacy("{\"amount\":1}".into(), 1.0)
                            .with_openenv(provenance.clone())
                    })
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let admitted_corpus_sha256 = kiln_eval::sha256_json(&serde_json::to_value(&groups).unwrap());
    TrainingDataProvenance {
        source: "inline".into(),
        dataset: None,
        split: None,
        dataset_corpus_sha256: None,
        split_manifest_sha256: None,
        admitted_corpus_sha256,
        rows: groups.len() as u64,
        openenv: kiln_train::openenv_training_data_provenance(&groups).unwrap(),
    }
}

fn test_adapter_path(temp: &tempfile::TempDir, job_id: &str) -> PathBuf {
    temp.path().join(format!("adapter-{job_id}"))
}

fn write_test_training_evidence(
    temp: &tempfile::TempDir,
    job_id: &str,
    training_data: &TrainingDataProvenance,
) -> PathBuf {
    let adapter_path = test_adapter_path(temp, job_id);
    std::fs::create_dir_all(&adapter_path).unwrap();
    std::fs::write(
        adapter_path.join("adapter_config.json"),
        br#"{"r":8,"lora_alpha":16.0}"#,
    )
    .unwrap();
    std::fs::write(
        adapter_path.join("adapter_model.safetensors"),
        b"test adapter weights",
    )
    .unwrap();
    let mut receipt = kiln_train::TrainReceipt::new(
        "agent",
        "grpo",
        &ModelConfig::qwen3_5_4b(),
        &crate::api::test_tokenizer(),
        kiln_train::train_receipt::HyperparameterReceipt {
            mode: "grpo".into(),
            rank: 8,
            alpha: 16.0,
            alpha_over_rank: Some(2.0),
            learning_rate: 1e-4,
            epochs: 1,
            seed: Some(17),
            shuffle: false,
        },
        json!({"output_name": "agent"}),
    );
    receipt.training_data = kiln_train::train_receipt::TrainingDataReceipt {
        source: "inline_grpo_groups".into(),
        path: None,
        sha256: Some(training_data.admitted_corpus_sha256.clone()),
        openenv: training_data.openenv.clone(),
    };
    receipt.write_to_adapter_dir(&adapter_path).unwrap();
    adapter_path
}

fn training_job(temp: &tempfile::TempDir, job_id: &str, state: TrainingState) -> TrainingJobInfo {
    let training_data = test_training_data();
    let adapter_path = write_test_training_evidence(temp, job_id, &training_data);
    TrainingJobInfo {
        job_id: job_id.into(),
        adapter_name: "agent".into(),
        job_type: TrainingJobType::Grpo,
        effective_seed: Some(17),
        state,
        progress: if state == TrainingState::Completed {
            1.0
        } else {
            0.0
        },
        loss: None,
        epoch: None,
        adapter_path: (state == TrainingState::Completed)
            .then(|| adapter_path.display().to_string()),
        submitted_at: Instant::now(),
        submitted_unix_ms: now_unix_ms(),
        auto_load: false,
        consumed_correction_ids: Vec::new(),
        training_data: Some(training_data),
        finished_at: None,
        finished_unix_ms: None,
        error: None,
        linked_eval_job_ids: Vec::new(),
        post_eval_verdict: None,
        gate_outcome: None,
        post_eval_gate_evidence: Vec::new(),
        cancel_requested: Default::default(),
        loss_history: Vec::new(),
    }
}

async fn task_fixture_post(
    AxumPath((_environment, operation)): AxumPath<(String, String)>,
    Json(body): Json<Value>,
) -> AxumResponse {
    match operation.as_str() {
        "num_tasks" => {
            assert_eq!(body, json!({"split": "train"}));
            Json(json!({"num_tasks": 3})).into_response()
        }
        "task_range" => {
            assert_eq!(body, json!({"split": "train", "start": 1, "stop": 2}));
            Json(json!({
                "tasks": [{"id": 1, "prompt": "2 + 2", "answer": "4"}]
            }))
            .into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn task_fixture() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = Router::new()
        .route(
            "/list_environments",
            axum_get(|| async { Json(json!(["task_env"])) }),
        )
        .route(
            "/{environment}/splits",
            axum_get(|| async {
                Json(json!([
                    {"name": "train", "type": "train"},
                    {"name": "holdout", "type": "validation"}
                ]))
            }),
        )
        .route("/{environment}/{operation}", axum_post(task_fixture_post));
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), server)
}

#[tokio::test]
async fn run_registry_admits_fifo_and_remains_bounded_persisted_and_restored() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig {
        max_active_runs: 1,
        max_tracked_runs: 3,
        ..Default::default()
    };
    let registry =
        Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap());
    let (first, first_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    let (second, second_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    let (third, third_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    assert!(insert_test_run(&registry, request(OpenEnvRunKind::Rollout)).is_err());
    assert_eq!(
        registry
            .get(&first.run_id)
            .unwrap()
            .admission
            .unwrap()
            .queue_position,
        Some(1)
    );
    assert_eq!(
        registry
            .get(&third.run_id)
            .unwrap()
            .admission
            .unwrap()
            .queue_position,
        Some(3)
    );

    let first_permit = registry
        .acquire(&first.run_id, &first_control)
        .await
        .unwrap();
    assert_eq!(registry.counts(), (1, 2, 3));
    let third_registry = registry.clone();
    let third_run_id = third.run_id.clone();
    let third_wait =
        tokio::spawn(async move { third_registry.acquire(&third_run_id, &third_control).await });
    let second_registry = registry.clone();
    let second_run_id = second.run_id.clone();
    let second_wait = tokio::spawn(async move {
        second_registry
            .acquire(&second_run_id, &second_control)
            .await
    });
    tokio::task::yield_now().await;
    assert!(!second_wait.is_finished());
    assert!(!third_wait.is_finished());

    registry
        .update(&first.run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.submitted_unix_ms = 1;
            status.finished_unix_ms = Some(now_unix_ms());
        })
        .unwrap();
    drop(first_permit);
    let second_permit = tokio::time::timeout(Duration::from_secs(1), second_wait)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        !third_wait.is_finished(),
        "the third run must not bypass FIFO order"
    );
    registry
        .update(&second.run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.submitted_unix_ms = 2;
            status.finished_unix_ms = Some(now_unix_ms());
        })
        .unwrap();
    drop(second_permit);
    let third_permit = tokio::time::timeout(Duration::from_secs(1), third_wait)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    registry
        .update(&third.run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.submitted_unix_ms = 3;
            status.finished_unix_ms = Some(now_unix_ms());
        })
        .unwrap();
    drop(third_permit);

    let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
    assert_eq!(
        restored.get(&first.run_id).unwrap().state,
        OpenEnvRunState::RolloutReady
    );
    insert_test_run(&restored, request(OpenEnvRunKind::Rollout))
        .expect("the oldest terminal status should be evicted to admit new work");
    assert!(restored.get(&first.run_id).is_none());
}

#[test]
fn run_registry_idempotency_is_atomic_conflict_safe_and_restart_durable() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig::default();
    let registry =
        Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap());
    let mut submitted = request(OpenEnvRunKind::Rollout);
    submitted.idempotency_key = Some("experiment:counter:17".into());
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let mut handles = Vec::new();
    for _ in 0..2 {
        let registry = registry.clone();
        let barrier = barrier.clone();
        let submitted = submitted.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            match insert_test_run(&registry, submitted).unwrap() {
                OpenEnvRunInsertOutcome::Created { status, .. } => (true, status.run_id),
                OpenEnvRunInsertOutcome::Replayed(status) => (false, status.run_id),
            }
        }));
    }
    barrier.wait();
    let first = handles.remove(0).join().unwrap();
    let second = handles.remove(0).join().unwrap();
    assert_ne!(first.0, second.0, "exactly one caller must create the run");
    assert_eq!(first.1, second.1);
    assert_eq!(registry.counts(), (0, 1, 1));
    assert_eq!(
        registry
            .replay_idempotent(&submitted)
            .unwrap()
            .unwrap()
            .run_id,
        first.1
    );

    let mut conflict = submitted.clone();
    conflict.groups += 1;
    let error = registry.replay_idempotent(&conflict).unwrap_err();
    assert!(error.downcast_ref::<OpenEnvIdempotencyConflict>().is_some());
    drop(registry);

    let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
    assert_eq!(
        restored
            .replay_idempotent(&submitted)
            .unwrap()
            .unwrap()
            .run_id,
        first.1
    );
    let mut duplicate = restored.get(&first.1).unwrap();
    duplicate.run_id = uuid::Uuid::new_v4().to_string();
    duplicate.admission.as_mut().unwrap().sequence += 1;
    std::fs::create_dir(restored.run_dir(&duplicate.run_id)).unwrap();
    persist_status_to(&restored.status_path(&duplicate.run_id), &duplicate).unwrap();
    drop(restored);
    let error =
        OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap_err();
    assert!(error.to_string().contains("share idempotency key"));
}

#[test]
fn admitted_training_contract_is_persisted_and_migrates_pristine_v4_runs() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig::default();
    let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
    let mut submitted = request(OpenEnvRunKind::Train);
    submitted.auto_load = false;
    submitted.training_config = Some(GrpoConfig {
        learning_rate: Some(3e-5),
        lora_rank: 16,
        output_name: Some("ignored-request-output".into()),
        behavior_policy: BehaviorPolicy::Recorded,
        ..GrpoConfig::default()
    });
    let expected = materialized_openenv_training_contract(&submitted)
        .unwrap()
        .unwrap();
    let mut mismatched = expected.clone();
    mismatched.effective_config.lora_rank += 1;
    assert!(
        registry
            .insert(submitted.clone(), Some(mismatched))
            .is_err(),
        "persistence must reject a contract that disagrees with its owned request fields"
    );
    let mut malformed_policy = expected.clone();
    malformed_policy.behavior_policy = Some(kiln_train::RolloutBehaviorPolicyIdentityV1 {
        served_model_id: "test-model".into(),
        base_model_sha256: "not-a-digest".into(),
        adapter: None,
        inference_config_sha256: format!("sha256:{}", "a".repeat(64)),
        implementation: "kiln/test".into(),
    });
    assert!(
        registry
            .insert(submitted.clone(), Some(malformed_policy))
            .is_err(),
        "persistence must reject a malformed behavior-policy identity"
    );
    let (created, _) = insert_created(&registry, submitted);
    assert_eq!(created.schema, OPENENV_RUN_SCHEMA_V5);
    assert_eq!(
        serde_json::to_value(created.training_contract.as_ref().unwrap()).unwrap(),
        serde_json::to_value(&expected).unwrap()
    );
    let persisted: OpenEnvRunStatus =
        serde_json::from_slice(&std::fs::read(registry.status_path(&created.run_id)).unwrap())
            .unwrap();
    assert_eq!(
        serde_json::to_value(persisted.training_contract.as_ref().unwrap()).unwrap(),
        serde_json::to_value(&expected).unwrap(),
        "run.json must retain the exact admitted config before collection"
    );

    registry
        .update(&created.run_id, |status| {
            status.schema = OPENENV_RUN_SCHEMA_V4.into();
            status.training_contract = None;
        })
        .unwrap();
    drop(registry);

    let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
    let migrated = restored.get(&created.run_id).unwrap();
    assert_eq!(migrated.schema, OPENENV_RUN_SCHEMA_V5);
    assert!(migrated.safely_restartable_queued());
    assert_eq!(
        serde_json::to_value(migrated.training_contract.unwrap()).unwrap(),
        serde_json::to_value(expected).unwrap(),
        "a pristine v4 queue entry must be sealed exactly once before resume"
    );
}

#[test]
fn capacity_eviction_releases_idempotency_key_across_restart() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig {
        max_tracked_runs: 1,
        ..Default::default()
    };
    let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
    let mut first_request = request(OpenEnvRunKind::Rollout);
    first_request.idempotency_key = Some("reusable:attempt".into());
    let (first, _) = insert_created(&registry, first_request.clone());
    registry.cancel(&first.run_id).unwrap();

    let mut displacement = request(OpenEnvRunKind::Rollout);
    displacement.idempotency_key = Some("displacement".into());
    let (second, _) = insert_created(&registry, displacement);
    registry.cancel(&second.run_id).unwrap();

    let (replacement, _) = insert_created(&registry, first_request.clone());
    assert_ne!(replacement.run_id, first.run_id);
    drop(registry);

    let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
    assert_eq!(
        restored
            .replay_idempotent(&first_request)
            .unwrap()
            .unwrap()
            .run_id,
        replacement.run_id
    );
}

#[tokio::test]
async fn queued_run_cancels_immediately_without_consuming_execution_capacity() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig {
        max_active_runs: 1,
        max_tracked_runs: 3,
        ..Default::default()
    };
    let registry = Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap());
    let (first, first_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    let (second, second_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    let first_permit = registry
        .acquire(&first.run_id, &first_control)
        .await
        .unwrap();

    let (cancelled, settled_queued) = registry.cancel(&second.run_id).unwrap();
    assert!(settled_queued);
    assert_eq!(cancelled.state, OpenEnvRunState::Cancelled);
    assert!(cancelled.finished_unix_ms.is_some());
    assert_eq!(cancelled.admission.unwrap().queue_position, None);
    assert_eq!(registry.counts(), (1, 0, 2));
    assert!(
        tokio::time::timeout(
            Duration::from_secs(1),
            registry.acquire(&second.run_id, &second_control)
        )
        .await
        .unwrap()
        .is_err()
    );
    drop(first_permit);
}

#[tokio::test]
async fn restart_resumes_only_fifo_entries_that_never_acquired_capacity() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig {
        max_active_runs: 1,
        max_tracked_runs: 3,
        ..Default::default()
    };
    let registry =
        Arc::new(OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap());
    let (active, active_control) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    let (queued_first, _) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    let (queued_second, _) = insert_created(&registry, request(OpenEnvRunKind::Rollout));
    assert!(
        queued_first.admission.as_ref().unwrap().sequence
            < queued_second.admission.as_ref().unwrap().sequence
    );
    let permit = registry
        .acquire(&active.run_id, &active_control)
        .await
        .unwrap();
    drop(permit);
    drop(registry);

    let restored = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).unwrap();
    let interrupted = restored.get(&active.run_id).unwrap();
    assert_eq!(interrupted.state, OpenEnvRunState::Failed);
    assert!(interrupted.error.as_deref().unwrap().contains("restarted"));
    let failure = interrupted.failure.unwrap();
    assert_eq!(failure.schema, OPENENV_RUN_FAILURE_SCHEMA_V1);
    assert_eq!(failure.code, OpenEnvRunFailureCode::RunInterrupted);
    assert_eq!(failure.stage, OpenEnvRunFailureStage::Restoration);
    assert!(failure.retryable);
    let safe = restored.get(&queued_first.run_id).unwrap();
    assert_eq!(safe.state, OpenEnvRunState::Queued);
    assert_eq!(safe.admission.unwrap().queue_position, Some(1));
    assert_eq!(
        restored
            .get(&queued_second.run_id)
            .unwrap()
            .admission
            .unwrap()
            .queue_position,
        Some(2)
    );
    let queued_controls = restored.queued_controls();
    assert_eq!(queued_controls.len(), 2);
    assert_eq!(queued_controls[0].0, queued_first.run_id);
    assert_eq!(queued_controls[1].0, queued_second.run_id);
}

#[test]
fn remote_environment_policy_is_fail_closed() {
    assert!(validate_environment_urls(&["http://127.0.0.1:1".into()], false).is_ok());
    assert!(validate_environment_urls(&["http://[::1]:1".into()], false).is_ok());
    assert!(validate_environment_urls(&["https://example.com".into()], false).is_err());
    assert!(validate_environment_urls(&["https://example.com".into()], true).is_ok());
    assert!(validate_environment_urls(&["http://user:secret@127.0.0.1:1".into()], true).is_err());
}

#[test]
fn idempotency_keys_are_bounded_non_secret_opaque_tokens() {
    for key in ["experiment-17", "client.retry_2", "capability:math:003"] {
        assert!(validate_openenv_idempotency_key(key).is_ok());
    }
    for key in ["", "contains space", "secret/token", "line\nbreak"] {
        assert!(validate_openenv_idempotency_key(key).is_err(), "{key:?}");
    }
    assert!(validate_openenv_idempotency_key(&"a".repeat(128)).is_ok());
    assert!(validate_openenv_idempotency_key(&"a".repeat(129)).is_err());

    let mut invalid = request(OpenEnvRunKind::Rollout);
    invalid.idempotency_key = Some("bad key".into());
    assert!(validate_run_request(&invalid, &OpenEnvConfig::default()).is_err());
}

#[test]
fn credential_handles_are_aligned_and_resolved_before_admission() {
    let policy = OpenEnvConfig::default();
    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout.credential_ids = vec![Some("missing".into())];
    let error = validate_run_request(&rollout, &policy).unwrap_err();
    assert_eq!(error.code, "openenv_invalid_credential");
    assert!(!error.message.contains("bearer"));

    rollout.credential_ids = vec![None, None];
    let error = validate_run_request(&rollout, &policy).unwrap_err();
    assert_eq!(error.code, "openenv_invalid_credential");
    assert!(error.message.contains("exactly one"));

    rollout.credential_ids = vec![None];
    assert!(validate_run_request(&rollout, &policy).is_ok());
}

#[test]
fn train_requires_a_valid_output_adapter() {
    let policy = OpenEnvConfig::default();
    let mut train = request(OpenEnvRunKind::Train);
    train.output_adapter = None;
    assert!(validate_run_request(&train, &policy).is_err());
    train.output_adapter = Some("../escape".into());
    assert!(validate_run_request(&train, &policy).is_err());
}

#[test]
fn rollout_rejects_every_training_only_field() {
    let policy = OpenEnvConfig::default();

    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout.output_adapter = Some("agent".into());
    assert!(validate_run_request(&rollout, &policy).is_err());

    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout.training_config = Some(GrpoConfig::default());
    assert!(validate_run_request(&rollout, &policy).is_err());

    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout.post_eval = Some(kiln_eval::PostEvalConfig {
        suite: "held-out".into(),
        data_scope: Default::default(),
        generation: None,
        min_accuracy: None,
        include_baseline: true,
    });
    assert!(validate_run_request(&rollout, &policy).is_err());

    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
        groups: 1,
        group_size: 1,
        seed_start: None,
        gate: None,
    });
    assert!(validate_run_request(&rollout, &policy).is_err());
}

#[test]
fn effective_grpo_config_is_owned_by_the_live_rollout_contract() {
    let mut train = request(OpenEnvRunKind::Train);
    train.adapter = "behavior-agent".into();
    train.auto_load = true;
    train.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
        groups: 1,
        group_size: 1,
        seed_start: None,
        gate: Some(crate::openenv_evaluation::OpenEnvEnvironmentEvalGate {
            min_mean_return: None,
            min_mean_improvement: 0.0,
        }),
    });
    let supplied = GrpoConfig {
        output_name: Some("ignored".into()),
        auto_load: true,
        base_adapter: Some("ignored".into()),
        behavior_policy: BehaviorPolicy::Recorded,
        ..Default::default()
    };
    train.training_config = Some(supplied);

    let effective = effective_openenv_grpo_config(&train, "trained-agent");
    assert_eq!(effective.output_name.as_deref(), Some("trained-agent"));
    assert_eq!(effective.base_adapter.as_deref(), Some("behavior-agent"));
    assert_eq!(
        effective.behavior_policy,
        BehaviorPolicy::NoImportanceCorrection
    );
    assert!(!effective.auto_load, "environment gate owns promotion");
}

#[test]
fn train_preflight_rejects_static_failures_before_mock_backend_admission() {
    let temp = tempfile::tempdir().unwrap();
    let mut state = test_state(&temp, OpenEnvConfig::default());

    let mut invalid_config = request(OpenEnvRunKind::Train);
    let config = GrpoConfig {
        checkpoint_interval: Some(0),
        ..Default::default()
    };
    invalid_config.training_config = Some(config);
    let error = validate_openenv_training_preflight(&state, &invalid_config).unwrap_err();
    assert_eq!(error.code, "training_invalid_request");
    assert!(error.message.contains("checkpoint_interval"));

    state.suite_registry = Some(Arc::new(crate::eval::SuiteRegistry::new(
        temp.path().join("suites"),
    )));
    let mut missing_suite = request(OpenEnvRunKind::Train);
    missing_suite.post_eval = Some(kiln_eval::PostEvalConfig {
        suite: "not-installed".into(),
        data_scope: Default::default(),
        generation: None,
        min_accuracy: None,
        include_baseline: true,
    });
    let error = validate_openenv_training_preflight(&state, &missing_suite).unwrap_err();
    assert_eq!(error.code, "training_invalid_request");
    assert!(error.message.contains("not an installed eval suite"));

    let mut missing_policy = request(OpenEnvRunKind::Train);
    missing_policy.adapter = "missing-policy".into();
    let error = validate_openenv_training_preflight(&state, &missing_policy).unwrap_err();
    assert_eq!(error.code, "adapter_not_found");

    let error =
        validate_openenv_training_preflight(&state, &request(OpenEnvRunKind::Train)).unwrap_err();
    assert_eq!(error.code, "mock_mode");
}

#[test]
fn collection_bounds_are_rejected_before_run_admission() {
    let policy = OpenEnvConfig::default();
    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout.groups = 0;
    assert!(validate_run_request(&rollout, &policy).is_err());
    rollout.groups = 2;
    rollout.temperature = f32::NAN;
    assert!(validate_run_request(&rollout, &policy).is_err());
    rollout.temperature = 1.0;
    rollout.adapter = "../escape".into();
    assert!(validate_run_request(&rollout, &policy).is_err());
}

#[test]
fn heterogeneous_reset_plan_is_aligned_exclusive_and_preserved() {
    let policy = OpenEnvConfig::default();
    let mut rollout = request(OpenEnvRunKind::Rollout);
    rollout
        .environment_urls
        .push("http://127.0.0.1:8001".into());
    rollout.environment_reset_options =
        vec![json!({"difficulty": "hard"}), json!({"split": "train"})];
    assert!(validate_run_request(&rollout, &policy).is_ok());

    let options = rollout_options_for(&rollout, Path::new("."), vec![None, None]);
    assert_eq!(
        options.environment_reset_options_values,
        rollout.environment_reset_options
    );
    assert!(options.reset_options_value.is_none());

    rollout.groups = 1;
    assert!(validate_run_request(&rollout, &policy).is_err());
    rollout.groups = 2;
    rollout.environment_reset_options.pop();
    assert!(validate_run_request(&rollout, &policy).is_err());
    rollout
        .environment_reset_options
        .push(json!(["not", "an", "object"]));
    assert!(validate_run_request(&rollout, &policy).is_err());
    rollout.environment_reset_options[1] = json!({});
    rollout.reset_options = json!({"shared": true});
    assert!(validate_run_request(&rollout, &policy).is_err());
}

#[test]
fn environment_eval_requires_disjoint_seeds_and_one_gate_owner() {
    let policy = OpenEnvConfig::default();
    let mut train = request(OpenEnvRunKind::Train);
    train.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
        groups: 20,
        group_size: 1,
        seed_start: None,
        gate: Some(crate::openenv_evaluation::OpenEnvEnvironmentEvalGate {
            min_mean_return: None,
            min_mean_improvement: 0.0,
        }),
    });
    assert!(validate_run_request(&train, &policy).is_ok());
    assert_eq!(
        resolved_environment_eval_seed_start(&train),
        Some(train.seed_start + train.groups as u64)
    );

    train.environment_eval.as_mut().unwrap().groups = 1;
    train.environment_eval.as_mut().unwrap().group_size = 20;
    assert!(validate_run_request(&train, &policy).is_err());
    train.environment_eval.as_mut().unwrap().groups = 20;
    train.environment_eval.as_mut().unwrap().group_size = 1;
    train.environment_eval.as_mut().unwrap().seed_start = Some(1);
    assert!(validate_run_request(&train, &policy).is_err());
    train.environment_eval.as_mut().unwrap().seed_start = Some(100);
    train.post_eval = Some(kiln_eval::PostEvalConfig {
        suite: "held-out".into(),
        data_scope: Default::default(),
        generation: None,
        min_accuracy: Some(0.8),
        include_baseline: false,
    });
    assert!(validate_run_request(&train, &policy).is_err());
}

#[test]
fn environment_eval_preserves_a_distinct_baseline_revision() {
    let policy = OpenEnvConfig::default();
    let mut train = request(OpenEnvRunKind::Train);
    train.adapter = "agent".into();
    train.output_adapter = Some("agent".into());
    train.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
        groups: 1,
        group_size: 1,
        seed_start: None,
        gate: None,
    });
    assert!(validate_run_request(&train, &policy).is_err());
}

#[test]
fn cancellation_remains_available_after_training_handoff() {
    let temp = tempfile::tempdir().unwrap();
    let registry =
        OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap();
    let (status, _) = insert_created(&registry, request(OpenEnvRunKind::Train));
    registry
        .update(&status.run_id, |status| {
            status.state = OpenEnvRunState::Submitting;
        })
        .unwrap();
    assert!(registry.cancel(&status.run_id).is_ok());
    registry
        .update(&status.run_id, |status| {
            status.state = OpenEnvRunState::Completed;
            status.finished_unix_ms = Some(now_unix_ms());
        })
        .unwrap();
    assert!(registry.cancel(&status.run_id).is_err());
}

#[test]
fn v1_training_handoffs_remain_terminal_after_upgrade() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig {
        max_active_runs: 1,
        ..Default::default()
    };
    let registry = OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy.clone()).unwrap();
    let (legacy, _) = insert_created(&registry, request(OpenEnvRunKind::Train));
    registry
        .update(&legacy.run_id, |status| {
            status.schema = OPENENV_RUN_SCHEMA_V1.into();
            status.state = OpenEnvRunState::TrainingQueued;
            status.finished_unix_ms = Some(now_unix_ms());
        })
        .unwrap();
    assert_eq!(registry.counts().0, 0);
    assert!(
        insert_test_run(&registry, request(OpenEnvRunKind::Train)).is_ok(),
        "a historical v1 handoff must not consume v2 active-run capacity"
    );
    let restored =
        OpenEnvRunRegistry::open(temp.path().to_path_buf(), policy).expect("restore registry");
    assert_eq!(
        restored.get(&legacy.run_id).unwrap().state,
        OpenEnvRunState::TrainingQueued
    );
}

#[tokio::test]
async fn openenv_run_follows_trainer_to_actual_completion() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let (run, cancel) = insert_created(&state.openenv_runs, request(OpenEnvRunKind::Train));
    state.training_jobs.write().unwrap().insert(
        "train-1".into(),
        training_job(&temp, "train-1", TrainingState::Queued),
    );

    let followed_state = state.clone();
    let followed_run_id = run.run_id.clone();
    let follow = tokio::spawn(async move {
        follow_openenv_training(
            &followed_state,
            &followed_run_id,
            &request(OpenEnvRunKind::Train),
            None,
            "train-1",
            cancel.cancel,
        )
        .await
    });
    tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
    assert_eq!(
        state.openenv_runs.get(&run.run_id).unwrap().state,
        OpenEnvRunState::TrainingQueued
    );
    {
        let mut jobs = state.training_jobs.write().unwrap();
        let job = jobs.get_mut("train-1").unwrap();
        job.state = TrainingState::Running;
        job.progress = 0.5;
        job.loss = Some(0.25);
        job.epoch = Some(1);
    }
    tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
    let running = state.openenv_runs.get(&run.run_id).unwrap();
    assert_eq!(running.state, OpenEnvRunState::TrainingRunning);
    assert_eq!(running.training.unwrap().current_loss, Some(0.25));
    {
        let mut jobs = state.training_jobs.write().unwrap();
        let job = jobs.get_mut("train-1").unwrap();
        job.state = TrainingState::Completed;
        job.progress = 1.0;
        job.adapter_path = Some(test_adapter_path(&temp, "train-1").display().to_string());
    }
    follow.await.unwrap().unwrap();
    let completed = state.openenv_runs.get(&run.run_id).unwrap();
    assert_eq!(completed.state, OpenEnvRunState::Completed);
    assert!(completed.finished_unix_ms.is_some());
    let training = completed.training.as_ref().unwrap();
    let lineage = training
        .training_data
        .as_ref()
        .and_then(|data| data.openenv.as_ref())
        .unwrap();
    assert_eq!(lineage.groups, 2);
    assert_eq!(lineage.rollouts, 6);
    assert_eq!(lineage.seed_min, 17);
    assert_eq!(lineage.seed_max, 18);
    assert_eq!(
        completed
            .artifacts
            .iter()
            .map(|artifact| artifact.kind.as_str())
            .collect::<Vec<_>>(),
        ["train_receipt", "adapter_manifest"]
    );
    for artifact in &completed.artifacts {
        let (path, _, manifest) = state
            .openenv_runs
            .artifact_path(&run.run_id, &artifact.kind)
            .unwrap();
        let (sha256, bytes) = crate::openenv_replay::bounded_artifact_metadata(&path).unwrap();
        assert_eq!(manifest, *artifact);
        assert_eq!(sha256, artifact.sha256);
        assert_eq!(bytes, artifact.bytes);
    }
    assert!(completed.terminal());
}

#[test]
fn openenv_training_evidence_rejects_manifest_drift_before_publication() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let (run, _) = insert_created(&state.openenv_runs, request(OpenEnvRunKind::Train));
    let job = training_job(&temp, "train-tampered", TrainingState::Completed);
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(job.job_id.clone(), job);
    let training = training_status_for(&state, "train-tampered").unwrap();
    let manifest_path =
        test_adapter_path(&temp, "train-tampered").join(kiln_train::ADAPTER_MANIFEST_FILENAME);
    let mut manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["receipt_hash"] = json!(format!("sha256:{}", "0".repeat(64)));
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let error = publish_openenv_training_evidence(
        &state,
        &run.run_id,
        &request(OpenEnvRunKind::Train),
        &training,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("receipt hash differs"),
        "{error:#}"
    );
    assert!(
        state
            .openenv_runs
            .get(&run.run_id)
            .unwrap()
            .artifacts
            .is_empty()
    );
    assert!(
        !state
            .openenv_runs
            .run_dir(&run.run_id)
            .join(kiln_train::TRAIN_RECEIPT_FILENAME)
            .exists()
    );
}

#[tokio::test]
async fn completed_training_hands_off_to_native_environment_evaluation() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let mut run_request = request(OpenEnvRunKind::Train);
    run_request.environment_eval = Some(OpenEnvEnvironmentEvalConfig {
        groups: 20,
        group_size: 1,
        seed_start: None,
        gate: None,
    });
    let (run, cancel) = insert_created(&state.openenv_runs, run_request.clone());
    state.training_jobs.write().unwrap().insert(
        "train-environment-eval".into(),
        training_job(&temp, "train-environment-eval", TrainingState::Completed),
    );
    follow_openenv_training(
        &state,
        &run.run_id,
        &run_request,
        None,
        "train-environment-eval",
        cancel.cancel,
    )
    .await
    .unwrap();
    let handed_off = state.openenv_runs.get(&run.run_id).unwrap();
    assert_eq!(handed_off.state, OpenEnvRunState::EnvironmentEvaluating);
    assert!(handed_off.finished_unix_ms.is_none());
    assert!(!handed_off.terminal());
    let evaluation = handed_off.environment_evaluation.unwrap();
    assert_eq!(evaluation.seed_start, 2);
    assert_eq!(evaluation.state, OpenEnvEnvironmentEvalState::Pending);
}

#[tokio::test]
async fn openenv_run_waits_for_requested_post_evaluation() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let mut run_request = request(OpenEnvRunKind::Train);
    run_request.post_eval = Some(kiln_eval::PostEvalConfig {
        suite: "held-out".into(),
        data_scope: Default::default(),
        generation: None,
        min_accuracy: None,
        include_baseline: false,
    });
    let (run, cancel) = insert_created(&state.openenv_runs, run_request.clone());
    let mut training = training_job(&temp, "train-eval", TrainingState::Completed);
    training.linked_eval_job_ids.push("eval-1".into());
    state
        .training_jobs
        .write()
        .unwrap()
        .insert(training.job_id.clone(), training);
    state.eval_jobs.write().unwrap().insert(
        "eval-1".into(),
        EvalJobInfo::queued(
            "eval-1".into(),
            "held-out".into(),
            vec![Some("agent".into())],
            EvalSubmissionKind::PostTraining,
            Some("train-eval".into()),
            19,
        ),
    );

    let followed_state = state.clone();
    let followed_run_id = run.run_id.clone();
    let post_eval = run_request.post_eval.clone();
    let follow = tokio::spawn(async move {
        follow_openenv_training(
            &followed_state,
            &followed_run_id,
            &run_request,
            post_eval.as_ref(),
            "train-eval",
            cancel.cancel,
        )
        .await
    });
    tokio::time::sleep(LIFECYCLE_POLL_INTERVAL + Duration::from_millis(50)).await;
    let evaluating = state.openenv_runs.get(&run.run_id).unwrap();
    assert_eq!(evaluating.state, OpenEnvRunState::PostEvaluating);
    assert!(
        ["train_receipt", "adapter_manifest"]
            .into_iter()
            .all(|kind| {
                evaluating
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.kind == kind)
            })
    );
    {
        let mut evals = state.eval_jobs.write().unwrap();
        let eval = evals.get_mut("eval-1").unwrap();
        eval.state = EvalJobState::Completed;
        eval.progress.examples_completed = 20;
        eval.progress.examples_total = 20;
        eval.headline_accuracy = Some(0.9);
    }
    follow.await.unwrap().unwrap();
    let completed = state.openenv_runs.get(&run.run_id).unwrap();
    assert_eq!(completed.state, OpenEnvRunState::Completed);
    assert_eq!(completed.post_evaluations[0].headline_accuracy, Some(0.9));
}

#[tokio::test]
async fn http_surface_accepts_fifo_work_when_execution_capacity_is_occupied() {
    let temp = tempfile::tempdir().unwrap();
    let policy = OpenEnvConfig {
        max_active_runs: 1,
        max_tracked_runs: 3,
        ..Default::default()
    };
    let state = test_state(&temp, policy);
    let (active, active_control) =
        insert_created(&state.openenv_runs, request(OpenEnvRunKind::Rollout));
    let active_permit = state
        .openenv_runs
        .acquire(&active.run_id, &active_control)
        .await
        .unwrap();
    let app = routes().with_state(state.clone());
    let mut submitted = request(OpenEnvRunKind::Rollout);
    submitted.idempotency_key = Some("ui:retry:17".into());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&submitted).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let queued: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();
    assert_eq!(queued.schema, OPENENV_RUN_SCHEMA_V5);
    assert_eq!(queued.state, OpenEnvRunState::Queued);
    assert_eq!(queued.admission.as_ref().unwrap().queue_position, Some(1));
    assert_eq!(state.openenv_runs.counts(), (1, 1, 2));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&submitted).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let replayed: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();
    assert_eq!(replayed.run_id, queued.run_id);
    assert_eq!(state.openenv_runs.counts(), (1, 1, 2));
    assert_eq!(
        state
            .metrics
            .openenv_run_idempotent_replays
            .load(Ordering::Relaxed),
        1
    );

    let mut conflict = submitted.clone();
    conflict.groups += 1;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&conflict).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
        "openenv_run_idempotency_conflict"
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/openenv/runs/{}", queued.run_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let cancelled: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();
    assert_eq!(cancelled.state, OpenEnvRunState::Cancelled);
    assert_eq!(state.openenv_runs.counts(), (1, 0, 2));
    drop(active_permit);
}

#[tokio::test]
async fn failed_discovery_persists_typed_retryable_diagnosis() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let unavailable = listener.local_addr().unwrap();
    drop(listener);
    let mut submitted = request(OpenEnvRunKind::Rollout);
    submitted.environment_urls = vec![format!("http://{unavailable}")];
    let app = routes().with_state(state.clone());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&submitted).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let accepted: OpenEnvRunStatus = serde_json::from_slice(&body).unwrap();

    let failed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let status = state.openenv_runs.get(&accepted.run_id).unwrap();
            if status.state == OpenEnvRunState::Failed {
                break status;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let failure = failed.failure.unwrap();
    assert_eq!(failure.code, OpenEnvRunFailureCode::EnvironmentUnavailable);
    assert_eq!(failure.stage, OpenEnvRunFailureStage::Discovery);
    assert!(failure.retryable);
    assert_eq!(failed.error.as_deref(), Some(failure.message.as_str()));
    assert_eq!(state.metrics.openenv_runs_failed.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn http_surface_lists_runs_and_rejects_invalid_work_before_spawning() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let app = routes().with_state(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/openenv/runs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["schema"],
        OPENENV_RUN_LIST_SCHEMA_V5
    );

    let mut invalid = request(OpenEnvRunKind::Rollout);
    invalid.groups = 0;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&invalid).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        temp.path()
            .join(".openenv")
            .join("runs")
            .read_dir()
            .unwrap()
            .next()
            .is_none()
    );
}

#[tokio::test]
async fn http_train_preflight_rejection_is_observable_and_persists_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let metrics = state.metrics.clone();
    let app = routes().with_state(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&request(OpenEnvRunKind::Train)).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
        "mock_mode"
    );
    assert_eq!(
        metrics
            .openenv_training_preflight_rejected
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        metrics
            .openenv_training_preflights_rejected
            .load(Ordering::Relaxed),
        1
    );
    assert!(
        temp.path()
            .join(".openenv")
            .join("runs")
            .read_dir()
            .unwrap()
            .next()
            .is_none()
    );
}

#[tokio::test]
async fn direct_training_preflight_uses_the_same_fail_closed_backend_contract() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let metrics = state.metrics.clone();
    let app = routes().with_state(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/training/preflight")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&OpenEnvTrainingPreflightRequest {
                        adapter: "base".into(),
                        output_adapter: "direct-agent".into(),
                        training_config: GrpoConfig::default(),
                        auto_load: true,
                        post_eval: None,
                    })
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
        "mock_mode"
    );
    assert_eq!(
        metrics
            .openenv_training_preflights_rejected
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        metrics
            .openenv_training_preflights_accepted
            .load(Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn artifact_downloads_require_publication_and_reverify_the_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let (run, _) = insert_created(&state.openenv_runs, request(OpenEnvRunKind::Rollout));
    let artifact_bytes = b"{\"group\":1}\n";
    let artifact_path = state
        .openenv_runs
        .run_dir(&run.run_id)
        .join("rollouts.jsonl");
    std::fs::write(&artifact_path, artifact_bytes).unwrap();
    let artifact_url = format!("/v1/openenv/runs/{}/artifacts/dataset", run.run_id);
    let app = routes().with_state(state.clone());

    let unpublished = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&artifact_url)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unpublished.status(), StatusCode::NOT_FOUND);

    let sha256 = crate::openenv_replay::sha256_bytes(artifact_bytes);
    state
        .openenv_runs
        .update(&run.run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.artifacts = vec![OpenEnvArtifact {
                kind: "dataset".into(),
                url: artifact_url.clone(),
                sha256: sha256.clone(),
                bytes: artifact_bytes.len(),
            }];
        })
        .unwrap();

    let published = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&artifact_url)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(published.status(), StatusCode::OK);
    assert_eq!(
        published.headers()[header::CONTENT_LENGTH],
        artifact_bytes.len().to_string()
    );
    assert_eq!(published.headers()[header::ETAG], format!("\"{sha256}\""));
    assert_eq!(
        published.headers()[header::CACHE_CONTROL],
        "private, no-store"
    );
    let body = axum::body::to_bytes(published.into_body(), artifact_bytes.len())
        .await
        .unwrap();
    assert_eq!(body.as_ref(), artifact_bytes);

    std::fs::write(&artifact_path, b"{\"group\":2}\n").unwrap();
    let drifted = app
        .oneshot(
            Request::builder()
                .uri(&artifact_url)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(drifted.status(), StatusCode::CONFLICT);
    let body = axum::body::to_bytes(drifted.into_body(), 16 * 1024)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
        "openenv_artifact_integrity_failed"
    );
}

#[tokio::test]
async fn task_catalog_surface_pages_reference_shaped_tasks_and_tracks_outcomes() {
    let (environment_url, server) = task_fixture().await;
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(&temp, OpenEnvConfig::default());
    let metrics = state.metrics.clone();
    let app = routes().with_state(state);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/tasks")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "environment_urls": [environment_url],
                        "split": "TRAIN",
                        "start": 1,
                        "limit": 1
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let catalog: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(catalog["schema"], OPENENV_TASK_CATALOG_SCHEMA_V1);
    assert_eq!(catalog["catalogs"][0]["catalog"]["task_api"], "available");
    assert_eq!(catalog["catalogs"][0]["catalog"]["selected_split"], "train");
    assert_eq!(catalog["catalogs"][0]["catalog"]["num_tasks"], 3);
    assert_eq!(
        catalog["catalogs"][0]["catalog"]["tasks"][0],
        json!({"id": 1, "prompt": "2 + 2", "answer": "4"})
    );
    assert_eq!(
        metrics
            .openenv_task_catalog_inspections_started
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        metrics
            .openenv_task_catalog_inspections_completed
            .load(Ordering::Relaxed),
        1
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/tasks")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "environment_urls": ["http://127.0.0.1:1"],
                        "limit": MAX_OPENENV_TASK_PAGE_SIZE + 1
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        metrics
            .openenv_task_catalog_inspections_failed
            .load(Ordering::Relaxed),
        1
    );
    server.abort();
}

#[tokio::test]
async fn disabled_http_surface_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    let state = test_state(
        &temp,
        OpenEnvConfig {
            enabled: false,
            ..Default::default()
        },
    );
    let response = routes()
        .with_state(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/openenv/runs")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&request(OpenEnvRunKind::Rollout)).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
