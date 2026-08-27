use super::*;
use std::sync::Mutex;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode, header};
use axum::routing::{get, post};
use clap::Parser;

use crate::cli::{Cli, Commands};

#[test]
fn cli_parses_local_and_server_owned_openenv_commands() {
    let inspect = Cli::try_parse_from([
        "kiln",
        "openenv",
        "inspect",
        "--environment",
        "127.0.0.1:8990",
        "--credential-env",
        "OPENENV_TEST_TOKEN",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        inspect.command,
        Some(Commands::Openenv(OpenEnvCommands::Inspect {
            environment,
            credential_env,
            json: true,
            ..
        })) if environment == "127.0.0.1:8990"
            && credential_env.as_deref() == Some("OPENENV_TEST_TOKEN")
    ));

    let tasks = Cli::try_parse_from([
        "kiln",
        "openenv",
        "tasks",
        "--environment",
        "127.0.0.1:8990",
        "--environment-name",
        "math_env",
        "--split",
        "train",
        "--start",
        "20",
        "--limit",
        "10",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        tasks.command,
        Some(Commands::Openenv(OpenEnvCommands::Tasks {
            environment,
            environment_name,
            split,
            start: 20,
            limit: 10,
            json: true,
            ..
        })) if environment == "127.0.0.1:8990"
            && environment_name.as_deref() == Some("math_env")
            && split.as_deref() == Some("train")
    ));

    let rollout = Cli::try_parse_from([
        "kiln",
        "openenv",
        "rollout",
        "--environment",
        "http://127.0.0.1:8000",
        "--environment",
        "http://127.0.0.1:8001",
        "--credential-env",
        "-",
        "--credential-env",
        "ARCADE_TOKEN",
        "--environment-reset-options",
        "arcade.json",
        "--environment-reset-options",
        "-",
        "--groups",
        "12",
        "--group-size",
        "6",
        "--thinking",
        "true",
    ])
    .unwrap();
    let Some(Commands::Openenv(OpenEnvCommands::Rollout { rollout })) = rollout.command else {
        panic!("expected openenv rollout command");
    };
    assert_eq!(rollout.environment_urls.len(), 2);
    assert_eq!(rollout.credential_envs, ["-", "ARCADE_TOKEN"]);
    let options = openenv_rollout_options(&rollout);
    assert_eq!(
        options.credential_envs,
        [None, Some("ARCADE_TOKEN".to_string())]
    );
    assert_eq!(
        options.environment_reset_options,
        [Some(PathBuf::from("arcade.json")), None]
    );
    assert!(!format!("{options:?}").contains("ARCADE_TOKEN"));
    assert_eq!(rollout.groups, 12);
    assert_eq!(rollout.group_size, 6);
    assert!(rollout.thinking);
    assert_eq!(rollout.protocol_error_reward, -1.0);
    assert_eq!(rollout.max_recoverable_errors, 3);
    assert_eq!(rollout.capacity_wait_seconds, 300);
    assert_eq!(rollout.replay_output, PathBuf::from("openenv.replay.json"));

    let train = Cli::try_parse_from([
        "kiln",
        "openenv",
        "train",
        "--environment",
        "http://127.0.0.1:8000",
        "--adapter",
        "agent-v1",
        "--output-adapter",
        "agent-v2",
        "--auto-load",
        "false",
    ])
    .unwrap();
    let Some(Commands::Openenv(OpenEnvCommands::Train {
        rollout,
        output_adapter,
        auto_load: false,
        ..
    })) = train.command
    else {
        panic!("expected openenv train command");
    };
    assert_eq!(output_adapter, "agent-v2");
    assert!(
        rollout.thinking,
        "OpenEnv training must default to reasoning trajectories"
    );

    let start = Cli::try_parse_from([
        "kiln",
        "openenv",
        "start",
        "--request",
        "run.json",
        "--idempotency-key",
        "experiment:counter:17",
        "--follow",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        start.command,
        Some(Commands::Openenv(OpenEnvCommands::Start {
            request,
            idempotency_key: Some(idempotency_key),
            follow: true,
            json: true,
            ..
        })) if request == Path::new("run.json") && idempotency_key == "experiment:counter:17"
    ));

    let status = Cli::try_parse_from([
        "kiln",
        "openenv",
        "status",
        "80a26e21-8451-4a64-8666-890c06fd80bd",
        "--follow",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        status.command,
        Some(Commands::Openenv(OpenEnvCommands::Status {
            follow: true,
            json: true,
            ..
        }))
    ));
    let cancel = Cli::try_parse_from([
        "kiln",
        "openenv",
        "cancel",
        "80a26e21-8451-4a64-8666-890c06fd80bd",
    ])
    .unwrap();
    assert!(matches!(
        cancel.command,
        Some(Commands::Openenv(OpenEnvCommands::Cancel { .. }))
    ));

    let artifact = Cli::try_parse_from([
        "kiln",
        "openenv",
        "artifact",
        "80a26e21-8451-4a64-8666-890c06fd80bd",
        "environment_eval_receipt",
        "--output",
        "receipt.json",
        "--force",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        artifact.command,
        Some(Commands::Openenv(OpenEnvCommands::Artifact {
            kind,
            output,
            force: true,
            json: true,
            ..
        })) if kind == "environment_eval_receipt"
            && output == Path::new("receipt.json")
    ));

    let verify = Cli::try_parse_from([
        "kiln",
        "openenv",
        "verify",
        "--summary",
        "batch.summary.json",
        "--json",
    ])
    .unwrap();
    assert!(matches!(
        verify.command,
        Some(Commands::Openenv(OpenEnvCommands::Verify {
            summary,
            json: true,
            ..
        })) if summary == Path::new("batch.summary.json")
    ));

    let replay = Cli::try_parse_from([
        "kiln",
        "openenv",
        "replay",
        "--summary",
        "batch.summary.json",
        "--concurrency",
        "2",
        "--credential-env",
        "REPLAY_TOKEN",
    ])
    .unwrap();
    assert!(matches!(
        replay.command,
        Some(Commands::Openenv(OpenEnvCommands::Replay {
            concurrency: 2,
            capacity_wait_seconds: 300,
            credential_envs,
            ..
        })) if credential_envs == ["REPLAY_TOKEN"]
    ));

    assert!(
        Cli::try_parse_from(["kiln", "openenv", "rollout", "--groups", "2"]).is_err(),
        "an OpenEnv command without an environment must fail during parsing"
    );
}

#[test]
fn server_run_terminal_detection_preserves_v1_handoffs() {
    assert!(openenv_server_run_terminal(
        &json!({"schema":"kiln.openenv-run.v2","state":"completed"})
    ));
    assert!(!openenv_server_run_terminal(
        &json!({"schema":"kiln.openenv-run.v2","state":"training_queued"})
    ));
    assert!(openenv_server_run_terminal(
        &json!({"schema":"kiln.openenv-run.v1","state":"training_queued"})
    ));
}

#[test]
fn direct_training_preflight_receipt_binds_rollout_owned_fields_and_capacity() {
    let request = OpenEnvTrainingPreflightRequest {
        adapter: " behavior-agent ".into(),
        output_adapter: "candidate-agent".into(),
        training_config: GrpoConfig {
            lora_rank: 16,
            ..GrpoConfig::default()
        },
        auto_load: false,
        post_eval: None,
    };
    let mut effective_config = request.training_config.clone();
    effective_config.output_name = Some(request.output_adapter.clone());
    effective_config.base_adapter = Some("behavior-agent".into());
    effective_config.auto_load = false;
    effective_config.behavior_policy = BehaviorPolicy::NoImportanceCorrection;
    let mut receipt = OpenEnvTrainingPreflightReceipt {
        schema: OPENENV_TRAINING_PREFLIGHT_SCHEMA_V1.into(),
        effective_config,
        post_eval: None,
        behavior_policy: Some(RolloutBehaviorPolicyIdentityV1 {
            served_model_id: "test-model".into(),
            base_model_sha256: format!("sha256:{}", "a".repeat(64)),
            adapter: Some(kiln_train::RolloutAdapterIdentityV1 {
                name: "behavior-agent".into(),
                content_sha256: format!("sha256:{}", "b".repeat(64)),
            }),
            inference_config_sha256: format!("sha256:{}", "c".repeat(64)),
            implementation: "kiln/test".into(),
        }),
        capacity: OpenEnvTrainingCapacitySnapshot {
            checked_unix_ms: 17,
            queued_jobs: 1,
            max_queued_jobs: 8,
            tracked_jobs: 2,
            max_tracked_jobs: 32,
        },
        capacity_reserved: false,
    };
    validate_openenv_training_preflight_receipt(&request, &receipt).unwrap();
    let contract = receipt.training_contract();
    assert_eq!(contract.schema, OPENENV_TRAINING_CONTRACT_SCHEMA_V1);
    assert_eq!(
        serde_json::to_value(&contract.effective_config).unwrap(),
        serde_json::to_value(&receipt.effective_config).unwrap()
    );

    receipt.effective_config.output_name = Some("wrong-agent".into());
    assert!(
        validate_openenv_training_preflight_receipt(&request, &receipt)
            .unwrap_err()
            .to_string()
            .contains("rollout-owned")
    );
    receipt.effective_config.output_name = Some(request.output_adapter.clone());
    receipt.post_eval = Some(serde_json::from_value(json!({"suite": "unexpected-suite"})).unwrap());
    assert!(
        validate_openenv_training_preflight_receipt(&request, &receipt)
            .unwrap_err()
            .to_string()
            .contains("post-evaluation")
    );
    receipt.post_eval = None;
    receipt.capacity.queued_jobs = receipt.capacity.max_queued_jobs;
    assert!(
        validate_openenv_training_preflight_receipt(&request, &receipt)
            .unwrap_err()
            .to_string()
            .contains("no native queue capacity")
    );
}

#[test]
fn server_run_follow_fingerprint_tracks_fifo_position_changes() {
    let queued_second = json!({
        "schema": "kiln.openenv-run.v5",
        "state": "queued",
        "admission": {"max_active_runs": 1, "sequence": 2, "queue_position": 2}
    });
    let queued_first = json!({
        "schema": "kiln.openenv-run.v5",
        "state": "queued",
        "admission": {"max_active_runs": 1, "sequence": 2, "queue_position": 1}
    });
    assert!(!openenv_server_run_terminal(&queued_second));
    assert_ne!(
        openenv_server_run_fingerprint(&queued_second),
        openenv_server_run_fingerprint(&queued_first)
    );
}

#[test]
fn server_run_follow_fingerprint_tracks_corpus_and_evidence_publication() {
    let before = json!({
        "schema": "kiln.openenv-run.v5",
        "state": "training_queued",
        "training": {
            "state": "running",
            "training_data": {"openenv": {"group_plan_sha256": "sha256:before"}}
        },
        "artifacts": [{"kind": "dataset"}]
    });
    let after = json!({
        "schema": "kiln.openenv-run.v5",
        "state": "post_evaluating",
        "training": {
            "state": "completed",
            "training_data": {"openenv": {"group_plan_sha256": "sha256:after"}}
        },
        "artifacts": [
            {"kind": "dataset"},
            {"kind": "train_receipt"},
            {"kind": "adapter_manifest"}
        ]
    });
    assert_ne!(
        openenv_server_run_fingerprint(&before),
        openenv_server_run_fingerprint(&after)
    );
}

#[test]
fn server_run_follow_fingerprint_tracks_typed_failure_diagnosis() {
    let before = json!({"schema": "kiln.openenv-run.v5", "state": "collecting"});
    let after = json!({
        "schema": "kiln.openenv-run.v5",
        "state": "failed",
        "failure": {
            "code": "environment_capacity_exhausted",
            "stage": "collection"
        }
    });
    assert_ne!(
        openenv_server_run_fingerprint(&before),
        openenv_server_run_fingerprint(&after)
    );
}

#[test]
fn persisted_run_requests_are_bounded_regular_json_objects() {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        file.path(),
        br#"{"kind":"train","environment_urls":["http://127.0.0.1:8990"]}"#,
    )
    .unwrap();
    let request = read_openenv_run_request(file.path()).unwrap();
    assert_eq!(request["kind"], "train");

    std::fs::write(file.path(), b"[]").unwrap();
    assert!(
        read_openenv_run_request(file.path())
            .unwrap_err()
            .to_string()
            .contains("one JSON object")
    );

    file.as_file()
        .set_len((MAX_OPENENV_RUN_REQUEST_BYTES as u64) + 1)
        .unwrap();
    assert!(
        read_openenv_run_request(file.path())
            .unwrap_err()
            .to_string()
            .contains("limit")
    );
}

#[test]
fn persisted_start_injects_one_matching_retry_key() {
    let mut request = json!({
        "kind": "rollout",
        "environment_urls": ["http://127.0.0.1:8990"]
    });
    apply_openenv_idempotency_key(&mut request, Some("experiment:counter:17")).unwrap();
    assert_eq!(request["idempotency_key"], "experiment:counter:17");
    apply_openenv_idempotency_key(&mut request, Some("experiment:counter:17")).unwrap();
    assert!(apply_openenv_idempotency_key(&mut request, Some("different-key")).is_err());
    assert!(apply_openenv_idempotency_key(&mut request, Some("not secret")).is_err());
}

#[cfg(unix)]
#[test]
fn persisted_run_request_rejects_symlinks() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("request.json");
    let link = directory.path().join("request-link.json");
    std::fs::write(&target, b"{}").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert!(
        read_openenv_run_request(&link)
            .unwrap_err()
            .to_string()
            .contains("regular non-symlink file")
    );
}

#[test]
fn artifact_selection_is_manifest_bound_and_same_origin() {
    let run_id = "80a26e21-8451-4a64-8666-890c06fd80bd";
    let sha256 = format!("sha256:{}", "a".repeat(64));
    let run = json!({
        "run_id": run_id,
        "artifacts": [{
            "kind": "dataset",
            "url": format!("/v1/openenv/runs/{run_id}/artifacts/dataset"),
            "sha256": sha256,
            "bytes": 12
        }]
    });
    assert_eq!(
        manifest_artifact(&run, run_id, "dataset").unwrap(),
        OpenEnvManifestArtifact {
            url: format!("/v1/openenv/runs/{run_id}/artifacts/dataset"),
            sha256: format!("sha256:{}", "a".repeat(64)),
            bytes: 12,
        }
    );
    assert!(manifest_artifact(&run, run_id, "summary").is_err());

    let mut external = run.clone();
    external["artifacts"][0]["url"] = json!("https://example.com/secret");
    assert!(
        manifest_artifact(&external, run_id, "dataset")
            .unwrap_err()
            .to_string()
            .contains("does not match")
    );

    let mut malformed_digest = run.clone();
    malformed_digest["artifacts"][0]["sha256"] = json!(format!("sha256:{}", "A".repeat(64)));
    assert!(manifest_artifact(&malformed_digest, run_id, "dataset").is_err());

    let mut oversized = run;
    oversized["artifacts"][0]["bytes"] = json!((MAX_OPENENV_ARTIFACT_BYTES as u64) + 1);
    assert!(manifest_artifact(&oversized, run_id, "dataset").is_err());
}

#[tokio::test]
async fn persisted_start_and_artifact_download_follow_and_reverify_the_manifest() {
    let run_id = "80a26e21-8451-4a64-8666-890c06fd80bd";
    let original = b"{\"group\":1}\n".to_vec();
    let drifted = b"{\"group\":2}\n".to_vec();
    assert_eq!(original.len(), drifted.len());
    let sha256 = replay_sha256(&original);
    let artifact_url = format!("/v1/openenv/runs/{run_id}/artifacts/dataset");
    let status = json!({
        "schema": "kiln.openenv-run.v3",
        "run_id": run_id,
        "kind": "train",
        "state": "completed",
        "progress": {
            "groups_completed": 1,
            "groups_total": 1,
            "rollouts_completed": 1,
            "rollouts_total": 1
        },
        "artifacts": [{
            "kind": "dataset",
            "url": artifact_url,
            "sha256": sha256,
            "bytes": original.len()
        }]
    });
    let post_status = status.clone();
    let get_status = status.clone();
    let serve_drift = Arc::new(AtomicBool::new(false));
    let artifact_drift = serve_drift.clone();
    let response_sha256 = replay_sha256(&original);
    let response_len = original.len();
    let app = Router::new()
        .route(
            "/v1/openenv/runs",
            post(move |Json(request): Json<Value>| {
                let status = post_status.clone();
                async move {
                    if request.pointer("/environment_eval/groups") == Some(&json!(20)) {
                        (StatusCode::CREATED, Json(status))
                    } else {
                        (
                            StatusCode::BAD_REQUEST,
                            Json(json!({"error": "missing eval"})),
                        )
                    }
                }
            }),
        )
        .route(
            "/v1/openenv/runs/{run_id}",
            get(move || {
                let status = get_status.clone();
                async move { Json(status) }
            }),
        )
        .route(
            "/v1/openenv/runs/{run_id}/artifacts/dataset",
            get(move |headers: HeaderMap| {
                assert_eq!(
                    headers.get(header::ACCEPT_ENCODING),
                    Some(&HeaderValue::from_static("identity"))
                );
                let body = if artifact_drift.load(Ordering::Relaxed) {
                    drifted.clone()
                } else {
                    original.clone()
                };
                let etag = format!("\"{response_sha256}\"");
                async move {
                    let mut response = Response::new(Body::from(body));
                    response.headers_mut().insert(
                        header::CONTENT_LENGTH,
                        HeaderValue::from_str(&response_len.to_string()).unwrap(),
                    );
                    response
                        .headers_mut()
                        .insert(header::ETAG, HeaderValue::from_str(&etag).unwrap());
                    response.headers_mut().insert(
                        header::CACHE_CONTROL,
                        HeaderValue::from_static("private, no-store"),
                    );
                    response.headers_mut().insert(
                        header::X_CONTENT_TYPE_OPTIONS,
                        HeaderValue::from_static("nosniff"),
                    );
                    response
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let kiln_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let request = json!({
        "kind": "train",
        "environment_urls": ["http://127.0.0.1:8990"],
        "environment_eval": {"groups": 20, "group_size": 1}
    });
    let started = start_openenv_control_plane_run(&kiln_url, &request)
        .await
        .unwrap();
    assert_eq!(
        validated_openenv_server_run_id(&started, None).unwrap(),
        run_id
    );
    let terminal = watch_openenv_server_run(&kiln_url, run_id, true, false, Some(started))
        .await
        .unwrap();
    assert_eq!(terminal["state"], "completed");

    let output_dir = tempfile::tempdir().unwrap();
    let output = output_dir.path().join("rollouts.jsonl");
    let receipt = download_openenv_server_artifact(&kiln_url, run_id, "dataset", &output, false)
        .await
        .unwrap();
    assert_eq!(receipt.schema, OPENENV_ARTIFACT_DOWNLOAD_SCHEMA_V1);
    assert_eq!(
        receipt.sha256,
        replay_sha256(&std::fs::read(&output).unwrap())
    );
    assert_eq!(receipt.bytes, response_len);
    assert_eq!(receipt.source_url, artifact_url);

    let no_clobber = download_openenv_server_artifact(&kiln_url, run_id, "dataset", &output, false)
        .await
        .unwrap_err();
    assert!(no_clobber.to_string().contains("without replacement"));
    assert_eq!(std::fs::read(&output).unwrap(), b"{\"group\":1}\n");

    serve_drift.store(true, Ordering::Relaxed);
    let rejected = output_dir.path().join("drifted.jsonl");
    let error = download_openenv_server_artifact(&kiln_url, run_id, "dataset", &rejected, false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("digest"), "{error:#}");
    assert!(!rejected.exists(), "drifted bytes must never publish");

    server.abort();
}

#[test]
fn action_parser_accepts_objects_and_rejects_other_shapes() {
    assert_eq!(
        parse_model_action(r#"{"answer":"B"}"#).unwrap(),
        json!({"answer": "B"})
    );
    assert_eq!(
        parse_model_action("```json\n{\"answer\":\"B\"}\n```").unwrap(),
        json!({"answer": "B"})
    );
    assert!(parse_model_action("answer B").is_err());
    assert!(parse_model_action("[1,2]").is_err());
    assert!(parse_model_action("{} trailing").is_err());
    assert_eq!(
        parse_trajectory_model_action("reason carefully\n</think>\n\n{\"answer\":\"B\"}").unwrap(),
        json!({"answer": "B"})
    );
    assert_eq!(
        parse_trajectory_model_action(
            "reason carefully</think>{\"answer\":\"literal </think> text\"}"
        )
        .unwrap(),
        json!({"answer": "literal </think> text"})
    );
}

#[test]
fn model_actions_must_satisfy_the_advertised_openenv_schema() {
    let validator = OpenEnvActionValidator::compile(&json!({
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": false
    }))
    .unwrap();
    assert_eq!(
        parse_and_validate_model_action(r#"{"answer": 4}"#, &validator).unwrap(),
        json!({"answer": 4})
    );
    let secret = "MODEL_VALUE_MUST_NOT_LEAK";
    let (code, message) = parse_and_validate_model_action(
        &json!({"answer": secret, "extra": secret}).to_string(),
        &validator,
    )
    .unwrap_err();
    assert_eq!(code, "ACTION_SCHEMA_VALIDATION_FAILED");
    assert!(message.contains("type"));
    assert!(!message.contains(secret));
}

#[test]
fn rollout_float_aggregates_survive_json_roundtrip_exactly() {
    let exact_receipt_value = 51.24817837550540_4_f64;
    let decoded_receipt_value: f64 =
        serde_json::from_str(&serde_json::to_string(&exact_receipt_value).unwrap()).unwrap();
    assert_eq!(
        decoded_receipt_value, exact_receipt_value,
        "OpenEnv receipt floats must retain their exact IEEE-754 value"
    );

    let latencies = vec![1.468847_f64, 1.302474_f64];
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let encoded = serde_json::to_vec(&(mean, &latencies)).unwrap();
    let (decoded_mean, decoded_latencies): (f64, Vec<f64>) =
        serde_json::from_slice(&encoded).unwrap();

    assert_eq!(
        decoded_latencies.iter().sum::<f64>() / decoded_latencies.len() as f64,
        decoded_mean,
        "published OpenEnv records must reproduce their exact aggregate receipt"
    );
}

#[test]
fn reset_seed_overrides_caller_value_and_plan_normalization_removes_it() {
    let base = json!({"difficulty": 3, "seed": 999});
    let reset = reset_payload(&base, 7).unwrap();
    assert_eq!(reset, json!({"difficulty": 3, "seed": 7}));
    assert_eq!(sha256_json(&reset).unwrap().len(), "sha256:".len() + 64);
    assert_eq!(
        normalize_reset_options(base).unwrap(),
        json!({"difficulty": 3})
    );
}

#[test]
fn aligned_reset_plan_is_ordered_bounded_and_exclusive() {
    let parsed = Cli::try_parse_from([
        "kiln",
        "openenv",
        "rollout",
        "--environment",
        "http://127.0.0.1:8000",
        "--environment",
        "http://127.0.0.1:8001",
        "--groups",
        "2",
    ])
    .unwrap();
    let Some(Commands::Openenv(OpenEnvCommands::Rollout { rollout })) = parsed.command else {
        panic!("expected openenv rollout command");
    };
    let mut options = openenv_rollout_options(&rollout);
    options.environment_reset_options_values = vec![
        json!({"difficulty": "hard", "seed": 999}),
        json!({"split": "train"}),
    ];
    validate_options(&options).unwrap();
    assert_eq!(
        read_reset_plan(&options).unwrap(),
        [json!({"difficulty": "hard"}), json!({"split": "train"})]
    );

    options.reset_options_value = Some(json!({}));
    assert!(
        validate_options(&options)
            .unwrap_err()
            .to_string()
            .contains("mutually exclusive")
    );
    options.reset_options_value = None;
    options.environment_reset_options_values.pop();
    assert!(
        validate_options(&options)
            .unwrap_err()
            .to_string()
            .contains("exactly one object per environment")
    );
}

#[test]
fn optional_input_text_is_the_exact_prompt_with_generic_json_fallback() {
    let observation = OpenEnvObservation {
        observation: json!({
            "input_text": "Board here. Reply with one digit.",
            "legal_actions": [0, 2]
        }),
        reward: kiln_openenv::OpenEnvReward::Integer(1),
        done: false,
        metadata: None,
    };
    let content = model_observation_content("step result", &observation).unwrap();
    assert_eq!(content, "Board here. Reply with one digit.");

    let ordinary = OpenEnvObservation {
        observation: json!({"position": 7}),
        reward: kiln_openenv::OpenEnvReward::Null,
        done: false,
        metadata: None,
    };
    let content = model_observation_content("reset result", &ordinary).unwrap();
    assert!(content.starts_with("OpenEnv reset result"));
    assert!(content.contains(r#""position":7"#));
}

#[test]
fn reasoning_is_retained_as_generated_action_without_duplicating_prompt_think_open() {
    assert_eq!(
        model_action_trajectory_content(
            r#"{"answer":"42"}"#,
            Some(" Work through the equation. "),
            true,
        ),
        " Work through the equation. </think>{\"answer\":\"42\"}"
    );
    assert_eq!(
        model_action_trajectory_content(r#"{"answer":"42"}"#, None, false),
        r#"{"answer":"42"}"#
    );
    assert_eq!(
        model_action_trajectory_content("\n\nanswer", None, true),
        "</think>\n\nanswer",
        "an immediate thinking close is generated output even when its interior is empty"
    );
}

#[tokio::test]
async fn omitted_openenv_thinking_budget_is_explicitly_unlimited_and_no_output_stays_open() {
    let observed = Arc::new(Mutex::new(None));
    let observed_request = observed.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let observed_request = observed_request.clone();
            async move {
                *observed_request.lock().unwrap() = Some(body);
                Json(json!({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "unfinished reasoning exactly as generated",
                            "content": ""
                        },
                        "finish_reason": "length",
                        "rollout_provenance": {
                            "behavior_policy": {
                                "served_model_id": "test-policy",
                                "base_model_sha256": format!("sha256:{}", "a".repeat(64)),
                                "inference_config_sha256": format!("sha256:{}", "b".repeat(64)),
                                "implementation": "kiln/test"
                            }
                        }
                    }],
                    "usage": {"total_tokens": 32}
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let policy = OpenEnvPolicyTransport::Http {
        client: reqwest::Client::new(),
        kiln_url: format!("http://{address}"),
    };
    let validator = OpenEnvActionValidator::compile(&json!({
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
        "additionalProperties": false
    }))
    .unwrap();
    let adapter = Value::Null;

    let failure = generate_model_action(
        &policy,
        &validator,
        &[ChatMessage::new("user", "environment prompt verbatim")],
        &[],
        &adapter,
        "base",
        7,
        32,
        None,
        1.0,
        true,
    )
    .await
    .unwrap_err();

    let request = observed.lock().unwrap().clone().unwrap();
    assert_eq!(request["thinking_budget_tokens"], Value::Null);
    assert_eq!(request["thinking_budget_ms"], Value::Null);
    assert_eq!(request["chat_template_kwargs"]["enable_thinking"], true);
    match failure {
        ModelActionFailure::NoOutput { reasoning, .. } => assert_eq!(
            reasoning.as_deref(),
            Some("unfinished reasoning exactly as generated")
        ),
        other => panic!("expected a no-output diagnostic, got {other:?}"),
    }
    server.abort();
}

#[test]
fn training_discards_no_output_and_skips_zero_gradient_groups() {
    let no_output = || {
        ScoredRollout::from_trajectory(
            vec![
                action_segment("unfinished reasoning".to_string()),
                harness_error_segment(&json!({
                    "openenv_harness_error": {
                        "code": "MODEL_ACTION_NO_OUTPUT",
                        "message": "no final action"
                    },
                    "done": true
                }))
                .unwrap(),
            ],
            -1.0,
        )
    };
    let prompt = vec![ChatMessage::new("user", "exact environment prompt")];
    let groups = vec![
        AgenticGroup {
            messages: prompt.clone(),
            completions: vec![no_output(), no_output()],
        },
        AgenticGroup {
            messages: prompt.clone(),
            completions: vec![
                ScoredRollout::legacy("a".to_string(), 0.0),
                ScoredRollout::legacy("b".to_string(), 0.0),
            ],
        },
        AgenticGroup {
            messages: prompt,
            completions: vec![
                no_output(),
                ScoredRollout::legacy("a".to_string(), 0.0),
                ScoredRollout::legacy("b".to_string(), 1.0),
            ],
        },
    ];

    let (selected, selection) = openenv_training_group_selection(&groups);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].completions.len(), 2);
    assert_eq!(selection.groups_submitted, 1);
    assert_eq!(selection.rollouts_submitted, 2);
    assert_eq!(selection.no_output_rollouts_discarded, 3);
    assert_eq!(selection.no_usable_output_groups_skipped, 1);
    assert_eq!(selection.no_reward_gradient_groups_skipped, 1);
    assert_eq!(selection.warnings.len(), 3);
    assert!(
        selection
            .warnings
            .iter()
            .any(|warning| { warning.contains("group 2 discarded 1 completion(s)") })
    );
}

#[test]
fn adapter_base_aliases_are_explicit_null() {
    for alias in ["base", "none", "null", "BASE"] {
        let adapter = parse_adapter_selection(alias);
        assert_eq!(adapter.request_value, Value::Null);
        assert_eq!(adapter.label, "base");
    }
    let named = parse_adapter_selection("agent-v2");
    assert_eq!(named.request_value, Value::String("agent-v2".into()));
}

#[test]
fn generation_seeds_separate_candidates_and_steps_deterministically() {
    assert_eq!(generation_seed(42, 0, 0), 42);
    assert_eq!(generation_seed(42, 2, 3), generation_seed(42, 2, 3));
    assert_ne!(generation_seed(42, 1, 0), generation_seed(42, 0, 1));
}

#[test]
fn serialized_counter_matches_json_without_allocating_an_output_buffer() {
    let value = json!({
        "escaped": "line one\nline two\t\"quoted\"",
        "nested": [1, 2, 3, {"ok": true}]
    });
    assert_eq!(
        serialized_len(&value, "test value").unwrap(),
        serde_json::to_vec(&value).unwrap().len()
    );
    assert_eq!(
        pretty_serialized_len(&value, "test value").unwrap(),
        serde_json::to_vec_pretty(&value).unwrap().len()
    );

    let mut output = Vec::new();
    let mut writer = BoundedWriter::new(&mut output, 4, "test output");
    writer.write_all(b"1234").unwrap();
    assert!(writer.write_all(b"5").is_err());
    drop(writer);
    assert_eq!(output, b"1234");
}

#[test]
fn retained_byte_budget_rejects_incrementally_and_releases_on_compaction() {
    let budget = OpenEnvRetainedByteBudget::new(32);
    budget.charge(20, "first candidate").unwrap();
    let error = budget.charge(13, "second candidate").unwrap_err();
    assert!(error.to_string().contains("32 byte collection budget"));
    assert_eq!(budget.used(), 20);

    budget.replace(20, 8, "compacted group").unwrap();
    assert_eq!(budget.used(), 8);
    budget.charge(24, "remaining groups").unwrap();
    assert!(budget.charge(1, "one byte too many").is_err());
}

#[test]
fn reset_option_files_are_rejected_from_metadata_before_large_reads() {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file()
        .set_len((MAX_OPENENV_RESET_OPTIONS_BYTES as u64) + 1)
        .unwrap();
    let error = read_reset_options(Some(file.path()), None).unwrap_err();
    assert!(error.to_string().contains("exceed"), "{error:#}");
    assert!(error.to_string().contains("input limit"), "{error:#}");
}
