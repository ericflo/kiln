use super::*;
use kiln_train::{ChatMessage, GrpoConfig, GrpoGroup, ScoredCompletion};

fn grpo_group() -> GrpoGroup {
    GrpoGroup {
        messages: vec![ChatMessage::new("user", "a")],
        completions: vec![ScoredCompletion {
            text: "b".to_string(),
            reward: 1.0,
            ..Default::default()
        }],
    }
}

fn openenv_grpo_group(seed: u64) -> GrpoGroup {
    let behavior_policy = kiln_train::RolloutBehaviorPolicyIdentityV1 {
        served_model_id: "test-model".to_string(),
        base_model_sha256: format!("sha256:{}", "d".repeat(64)),
        adapter: None,
        inference_config_sha256: format!("sha256:{}", "e".repeat(64)),
        implementation: "kiln/test".to_string(),
    };
    let episode = kiln_train::OpenEnvRolloutProvenanceV1::new(
        "math-env",
        "https://env.test",
        Some("3.1.0".to_string()),
        format!("sha256:{}", "d".repeat(64)),
        format!("sha256:{}", "a".repeat(64)),
        format!("sha256:{}", "b".repeat(64)),
        format!("sha256:{}", "c".repeat(64)),
        seed,
        1,
        1.0,
        true,
        kiln_train::OpenEnvEpisodeTerminationV1::Done,
        None,
    )
    .unwrap()
    .with_behavior_policy(behavior_policy)
    .unwrap();
    GrpoGroup {
        messages: vec![ChatMessage::new("user", "a")],
        completions: vec![ScoredCompletion::legacy("b".to_string(), 1.0).with_openenv(episode)],
    }
}

#[test]
fn grpo_dataset_path_submission_rejects_an_invalid_tail_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grpo.jsonl");
    let first = serde_json::to_string(&grpo_group()).unwrap();
    std::fs::write(&path, format!("{first}\nthis is not json\n")).unwrap();
    let tokenizer = crate::api::test_tokenizer().with_chat_template(
        "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
    );

    let mut permit = crate::training_queue::PreparedTrainingDataPermit::default();
    let error = validate_grpo_jsonl_submission(
        path.to_str().unwrap(),
        dir.path(),
        &mut permit,
        &tokenizer,
        &GrpoConfig::default(),
        2,
        None,
    )
    .unwrap_err();
    assert!(error.message.contains("line 2"), "{}", error.message);
    assert!(
        std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .all(|entry| !entry.file_name().to_string_lossy().starts_with("grpo-")),
        "invalid admission must remove its incomplete private snapshot"
    );
}

#[test]
fn grpo_dataset_path_submission_scans_every_row_for_the_maximum_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grpo.jsonl");
    let short = grpo_group();
    let mut long = grpo_group();
    long.completions[0].text = "ab".repeat(64);
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&short).unwrap(),
            serde_json::to_string(&long).unwrap()
        ),
    )
    .unwrap();
    let tokenizer = crate::api::test_tokenizer().with_chat_template(
        "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
    );

    let mut permit = crate::training_queue::PreparedTrainingDataPermit::default();
    let stats = validate_grpo_jsonl_submission(
        path.to_str().unwrap(),
        dir.path(),
        &mut permit,
        &tokenizer,
        &GrpoConfig::default(),
        2,
        None,
    )
    .unwrap();
    assert!(stats.streaming_dataset);
    assert_eq!(stats.num_groups, Some(2));
    assert_eq!(stats.total_completions, Some(2));
    assert!(stats.max_seq_len > 0);
    let receipt = stats.source_receipt.unwrap();
    assert_eq!(receipt.groups, 2);
    assert_eq!(receipt.completions, 2);
    assert_eq!(receipt.max_seq_len, stats.max_seq_len);
    assert!(receipt.source_sha256.starts_with("sha256:"));
    let original = std::fs::canonicalize(&path).unwrap();
    assert_ne!(receipt.path, original);
    assert!(receipt.server_owned);
    assert!(
        std::fs::metadata(&receipt.path)
            .unwrap()
            .permissions()
            .readonly()
    );
    let snapshot_path = receipt.path.clone();
    let snapshot_sha256 = kiln_train::train_receipt::sha256_file(&snapshot_path).unwrap();
    std::fs::write(&path, b"caller replaced the original after admission\n").unwrap();
    assert_eq!(
        kiln_train::train_receipt::sha256_file(&snapshot_path).unwrap(),
        snapshot_sha256,
        "the trainer source must be independent of the caller path"
    );
    assert_eq!(
        permit.bytes(),
        receipt.size_bytes + receipt.preflight_host_bytes
    );
    drop(receipt);
    assert!(!snapshot_path.exists());
}

#[test]
fn grpo_dataset_path_admission_preserves_openenv_corpus_identity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("openenv.jsonl");
    std::fs::write(
        &path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&openenv_grpo_group(7)).unwrap(),
            serde_json::to_string(&openenv_grpo_group(8)).unwrap()
        ),
    )
    .unwrap();
    let tokenizer = crate::api::test_tokenizer().with_chat_template(
        "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
    );

    let mut permit = crate::training_queue::PreparedTrainingDataPermit::default();
    let stats = validate_grpo_jsonl_submission(
        path.to_str().unwrap(),
        dir.path(),
        &mut permit,
        &tokenizer,
        &GrpoConfig::default(),
        2,
        None,
    )
    .unwrap();
    let receipt = stats.source_receipt.unwrap();
    let openenv = receipt.openenv.as_ref().expect("OpenEnv corpus identity");
    assert_eq!(openenv.schema(), "kiln.openenv-training-data.v1");
    assert_eq!(openenv.groups, 2);
    assert_eq!(openenv.rollouts, 2);
    assert_eq!(openenv.unique_seeds, 2);
    assert_eq!(openenv.total_steps, 2);
    assert_eq!(openenv.environments[0].environment_name, "math-env");
    assert_eq!(
        openenv
            .behavior_policy
            .as_ref()
            .expect("OpenEnv behavior-policy identity")
            .implementation,
        "kiln/test"
    );
}
