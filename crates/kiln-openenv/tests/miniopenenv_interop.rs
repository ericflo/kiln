use kiln_openenv::{
    OPENENV_CLIENT_PROFILE, OpenEnvClient, OpenEnvClientError, OpenEnvErrorCode, OpenEnvReward,
};
use serde_json::json;

/// Byte-real interoperability check for the sibling miniopenenv counter.
///
/// `scripts/check_miniopenenv_interop.sh` owns the server lifecycle and opts
/// this ignored test in with a generic OpenEnv interoperability URL.
#[tokio::test]
#[ignore = "requires a live miniopenenv counter server"]
async fn drives_a_stateful_miniopenenv_counter_episode() {
    let url = std::env::var("KILN_OPENENV_INTEROP_COUNTER_URL")
        .expect("KILN_OPENENV_INTEROP_COUNTER_URL must identify the live counter server");
    let client = OpenEnvClient::new(url).unwrap();
    let inspection = client.inspect().await.unwrap();

    assert_eq!(inspection.identity.client_profile, OPENENV_CLIENT_PROFILE);
    assert_eq!(inspection.identity.metadata.name, "CounterEnvironment");
    assert_eq!(inspection.identity.environments, ["counter_env"]);
    assert_eq!(
        inspection.schema.action.pointer("/properties/amount/type"),
        Some(&json!("integer"))
    );

    let mut episode = client.connect().await.unwrap();
    let reset = episode
        .reset(&json!({"seed": 17, "episode_id": "kiln-interop"}))
        .await
        .unwrap();
    assert_eq!(reset.observation["total"], 0);
    assert_eq!(reset.reward, OpenEnvReward::Null);
    assert!(!reset.done);

    let first = episode
        .step(&json!({"amount": 2, "note": "first"}))
        .await
        .unwrap();
    assert_eq!(first.observation["total"], 2);
    assert_eq!(first.reward, OpenEnvReward::Float(2.0));
    assert!(!first.done);

    let second = episode
        .step(&json!({"amount": 2, "note": "second"}))
        .await
        .unwrap();
    assert_eq!(second.observation["total"], 4);
    assert_eq!(second.reward, OpenEnvReward::Float(4.0));
    assert!(second.done);

    let state = episode.state().await.unwrap();
    assert_eq!(state["episode_id"], "kiln-interop");
    assert_eq!(state["step_count"], 2);
    assert_eq!(state["total"], 4);
    episode.close().await.unwrap();
}

/// The pinned downstream matrix currently publishes an optional, generic
/// text-observation profile. Kiln does not require this field from OpenEnv,
/// but it must preserve and expose it when an environment offers one.
#[tokio::test]
#[ignore = "requires the live OpenEnv arcade matrix"]
async fn discovers_and_resets_every_text_profiled_arcade_environment() {
    let urls = std::env::var("KILN_OPENENV_INTEROP_ARCADE_URLS")
        .expect("KILN_OPENENV_INTEROP_ARCADE_URLS must identify the live arcade matrix");
    let urls = urls
        .split(',')
        .filter(|url| !url.trim().is_empty())
        .collect::<Vec<_>>();
    assert_eq!(urls.len(), 14, "the pinned arcade matrix must have 14 URLs");

    for (index, url) in urls.into_iter().enumerate() {
        let client = OpenEnvClient::new(url).unwrap();
        let inspection = client.inspect().await.unwrap();
        assert_eq!(
            inspection
                .schema
                .observation
                .pointer("/properties/input_text/type"),
            Some(&json!("string")),
            "{} did not advertise the optional text observation profile",
            inspection.identity.metadata.name
        );
        let actionable = inspection
            .schema
            .action
            .pointer("/properties")
            .and_then(serde_json::Value::as_object)
            .unwrap()
            .iter()
            .filter(|(name, _)| name.as_str() != "metadata")
            .collect::<Vec<_>>();
        assert_eq!(
            actionable.len(),
            1,
            "{} must expose one schema-discoverable action field",
            inspection.identity.metadata.name
        );
        assert!(
            matches!(
                actionable[0]
                    .1
                    .get("type")
                    .and_then(serde_json::Value::as_str),
                Some("integer" | "string")
            ),
            "{} action must be text-coercible in the pinned matrix",
            inspection.identity.metadata.name
        );

        let mut episode = client.connect().await.unwrap();
        let reset = episode
            .reset(&json!({
                "seed": 10_000 + index,
                "difficulty": 2,
                "episode_id": format!("kiln-text-profile-{index}")
            }))
            .await
            .unwrap();
        assert!(
            reset.observation["input_text"]
                .as_str()
                .is_some_and(|text| !text.trim().is_empty()),
            "{} returned an empty input_text",
            inspection.identity.metadata.name
        );
        assert!(!reset.done);
        episode.close().await.unwrap();
    }
}

/// Representative downstream environments cover changing action spaces,
/// semantic execution errors, procedural state, string actions, and all the
/// stateful reset/step/state behavior used by training.
#[tokio::test]
#[ignore = "requires live miniopenenv arcade servers"]
async fn drives_representative_arcade_environments_and_recovers_in_session() {
    let bandit = OpenEnvClient::new(
        std::env::var("KILN_OPENENV_INTEROP_BANDIT_URL")
            .expect("KILN_OPENENV_INTEROP_BANDIT_URL must identify a live OpenEnv server"),
    )
    .unwrap();
    let bandit_inspection = bandit.inspect().await.unwrap();
    assert_eq!(
        bandit_inspection
            .schema
            .action
            .pointer("/properties/arm/type"),
        Some(&json!("integer"))
    );
    let mut episode = bandit.connect().await.unwrap();
    let reset = episode.reset(&json!({"seed": 41})).await.unwrap();
    assert_eq!(reset.observation["pulls"], 0);
    let error = episode.step(&json!({"arm": 99})).await.unwrap_err();
    let OpenEnvClientError::Protocol(error) = error else {
        panic!("expected an OpenEnv execution error");
    };
    assert_eq!(error.code, OpenEnvErrorCode::ExecutionError);
    assert!(!error.code.is_terminal());
    let recovered = episode.step(&json!({"arm": 0})).await.unwrap();
    assert_eq!(recovered.observation["pulls"], 1);
    assert_eq!(recovered.observation["last_arm"], 0);
    assert!(matches!(recovered.reward, OpenEnvReward::Integer(0 | 1)));
    episode.close().await.unwrap();

    let connect4 = OpenEnvClient::new(
        std::env::var("KILN_OPENENV_INTEROP_CONNECT4_URL")
            .expect("KILN_OPENENV_INTEROP_CONNECT4_URL must identify a live OpenEnv server"),
    )
    .unwrap();
    let inspection = connect4.inspect().await.unwrap();
    assert_eq!(
        inspection.schema.action.pointer("/properties/col/type"),
        Some(&json!("integer"))
    );
    let mut episode = connect4.connect().await.unwrap();
    let reset = episode.reset(&json!({"seed": 42})).await.unwrap();
    assert_eq!(
        reset.observation["legal_actions"],
        json!([0, 1, 2, 3, 4, 5, 6])
    );
    let turn = episode.step(&json!({"col": 3})).await.unwrap();
    assert!(turn.observation["board"].as_str().unwrap().contains('X'));
    assert!(turn.observation["legal_actions"].is_array());
    assert_eq!(episode.state().await.unwrap()["step_count"], 1);
    episode.close().await.unwrap();

    let maze = OpenEnvClient::new(
        std::env::var("KILN_OPENENV_INTEROP_MAZE_URL")
            .expect("KILN_OPENENV_INTEROP_MAZE_URL must identify a live OpenEnv server"),
    )
    .unwrap();
    let inspection = maze.inspect().await.unwrap();
    assert_eq!(
        inspection.schema.action.pointer("/properties/move/type"),
        Some(&json!("integer"))
    );
    let mut episode = maze.connect().await.unwrap();
    let reset = episode.reset(&json!({"seed": 43})).await.unwrap();
    let grid = reset.observation["grid"].as_str().unwrap();
    assert!(grid.contains('A') && grid.contains('G'));
    let turn = episode.step(&json!({"move": 0})).await.unwrap();
    assert_eq!(turn.observation["goal_x"], 13);
    assert_eq!(episode.state().await.unwrap()["step_count"], 1);
    episode.close().await.unwrap();

    let wordle = OpenEnvClient::new(
        std::env::var("KILN_OPENENV_INTEROP_WORDLE_URL")
            .expect("KILN_OPENENV_INTEROP_WORDLE_URL must identify a live OpenEnv server"),
    )
    .unwrap();
    let inspection = wordle.inspect().await.unwrap();
    assert_eq!(
        inspection.schema.action.pointer("/properties/guess/type"),
        Some(&json!("string"))
    );
    let mut episode = wordle.connect().await.unwrap();
    let reset = episode.reset(&json!({"seed": 44})).await.unwrap();
    assert_eq!(reset.observation["guesses"], 0);
    let turn = episode.step(&json!({"guess": "crane"})).await.unwrap();
    assert_eq!(turn.observation["valid"], true);
    assert_eq!(turn.observation["guesses"], 1);
    assert_eq!(episode.state().await.unwrap()["step_count"], 1);
    episode.close().await.unwrap();
}

/// Capacity is an application frame after a successful WebSocket upgrade, not
/// an HTTP handshake failure. Kiln must classify it as terminal and allow a
/// fresh session after capacity is released.
#[tokio::test]
#[ignore = "requires a live max-sessions=1 miniopenenv bandit server"]
async fn observes_capacity_as_a_terminal_first_frame_and_reacquires() {
    let client = OpenEnvClient::new(
        std::env::var("KILN_OPENENV_INTEROP_BANDIT_URL")
            .expect("KILN_OPENENV_INTEROP_BANDIT_URL must identify a live OpenEnv server"),
    )
    .unwrap();
    let mut occupied = client.connect().await.unwrap();
    occupied.reset(&json!({"seed": 1})).await.unwrap();

    let mut refused = client.connect().await.unwrap();
    let error = refused.reset(&json!({"seed": 2})).await.unwrap_err();
    let OpenEnvClientError::Protocol(error) = error else {
        panic!("expected CAPACITY_REACHED");
    };
    assert_eq!(error.code, OpenEnvErrorCode::CapacityReached);
    assert!(error.code.is_terminal());
    assert_eq!(error.active_sessions, Some(1));
    assert_eq!(error.max_sessions, Some(1));

    occupied.close().await.unwrap();
    let mut reacquired = client.connect().await.unwrap();
    reacquired.reset(&json!({"seed": 3})).await.unwrap();
    reacquired.close().await.unwrap();
}
