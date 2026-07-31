use super::*;

fn rollout_request() -> OpenEnvRunRequest {
    serde_json::from_value(serde_json::json!({
        "kind": "rollout",
        "environment_urls": ["http://127.0.0.1:8990"]
    }))
    .unwrap()
}

fn published_stats() -> OpenEnvRolloutStats {
    OpenEnvRolloutStats {
        mean_episode_return: 0.5,
        min_episode_return: Some(0.0),
        max_episode_return: Some(1.0),
        done_count: 3,
        max_steps_count: 1,
        invalid_model_action_count: 0,
        protocol_error_count: 0,
        recoverable_protocol_error_count: 2,
        capacity_retry_count: 1,
        total_environment_steps: 7,
        total_model_tokens: 19,
        mean_model_latency_ms: 12.5,
    }
}

#[test]
fn rollout_stats_are_durable_and_legacy_status_without_them_still_loads() {
    let temp = tempfile::tempdir().unwrap();
    let registry =
        OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap();
    let status = match registry.insert(rollout_request(), None).unwrap() {
        OpenEnvRunInsertOutcome::Created { status, .. } => status,
        OpenEnvRunInsertOutcome::Replayed(_) => panic!("fresh request unexpectedly replayed"),
    };
    registry
        .update(&status.run_id, |status| {
            status.state = OpenEnvRunState::RolloutReady;
            status.finished_unix_ms = Some(now_unix_ms());
            status.progress.groups_completed = status.progress.groups_total;
            status.progress.rollouts_completed = status.progress.rollouts_total;
            status.rollout_stats = Some(published_stats());
        })
        .unwrap();
    let run_path = registry.run_dir(&status.run_id).join("run.json");
    drop(registry);

    let restored =
        OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap();
    assert_eq!(
        restored.get(&status.run_id).unwrap().rollout_stats.unwrap(),
        published_stats()
    );
    drop(restored);

    let mut legacy: Value = serde_json::from_slice(&std::fs::read(&run_path).unwrap()).unwrap();
    legacy.as_object_mut().unwrap().remove("rollout_stats");
    std::fs::write(&run_path, serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();
    let restored =
        OpenEnvRunRegistry::open(temp.path().to_path_buf(), OpenEnvConfig::default()).unwrap();
    assert!(
        restored
            .get(&status.run_id)
            .unwrap()
            .rollout_stats
            .is_none()
    );
}
