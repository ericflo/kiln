use kiln_openenv::{OPENENV_CLIENT_PROFILE, OpenEnvClient, OpenEnvReward};
use serde_json::json;

/// Byte-real interoperability check for the sibling miniopenenv counter.
///
/// `scripts/check_miniopenenv_interop.sh` owns the server lifecycle and opts
/// this ignored test in with `KILN_MINIOPENENV_URL`.
#[tokio::test]
#[ignore = "requires a live miniopenenv counter server"]
async fn drives_a_stateful_miniopenenv_counter_episode() {
    let url = std::env::var("KILN_MINIOPENENV_URL")
        .expect("KILN_MINIOPENENV_URL must identify the live counter server");
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
