use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{watch, Mutex};
use tokio::task::JoinHandle;

use crate::supervisor::ServerState;

const POLL_INTERVAL_SECS: u64 = 2;
const REQUEST_TIMEOUT_SECS: u64 = 1;
const MAX_CONSECUTIVE_FAILURES: u32 = 3;

/// Spawn a background task that polls `/v1/health` and `/v1/train/status` on
/// the managed kiln server every `POLL_INTERVAL_SECS` seconds and drives
/// `state` transitions accordingly. Exits when `shutdown` changes or when the
/// observed state is `Stopped`.
pub fn spawn_health_poller(
    state: Arc<Mutex<ServerState>>,
    host: String,
    port: u16,
    mut shutdown: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
        {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut interval = tokio::time::interval(Duration::from_secs(POLL_INTERVAL_SECS));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut consecutive_failures: u32 = 0;

        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    return;
                }
                _ = interval.tick() => {
                    {
                        let s = state.lock().await;
                        if matches!(*s, ServerState::Stopped) {
                            return;
                        }
                    }

                    let health_url = format!("http://{}:{}/v1/health", host, port);
                    let (health_reachable, inference_ready) =
                        match client.get(&health_url).send().await {
                            Ok(resp) => {
                                let status_success = resp.status().is_success();
                                let body = if status_success {
                                    resp.json::<serde_json::Value>().await.ok()
                                } else {
                                    None
                                };
                                (
                                    status_success,
                                    health_response_ready(status_success, body.as_ref()),
                                )
                            }
                            Err(_) => (false, false),
                        };

                    if health_reachable {
                        consecutive_failures = 0;
                    } else {
                        consecutive_failures = consecutive_failures.saturating_add(1);
                    }

                    let train_status = if inference_ready {
                        let train_url =
                            format!("http://{}:{}/v1/train/status", host, port);
                        match client.get(&train_url).send().await {
                            Ok(resp) if resp.status().is_success() => {
                                resp.json::<serde_json::Value>().await.ok()
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };

                    let mut s = state.lock().await;
                    let new_state = derive_state(
                        inference_ready,
                        consecutive_failures,
                        train_status.as_ref(),
                        &s,
                    );
                    *s = new_state;
                }
            }
        }
    })
}

/// Pure, testable decision function that derives the next `ServerState`
/// from the latest poll outcome.
///
/// Rules:
/// - `Stopped` is terminal from the poller's perspective — never overwrite it.
/// - `Starting` is only exited by an inference-ready health response; socket
///   readiness without prewarm keeps us in `Starting`.
/// - Otherwise, after `MAX_CONSECUTIVE_FAILURES` consecutive health failures,
///   transition to `Error`. Before that, preserve the prior state.
/// - On health success, transition to `TrainingActive` if any entry in
///   `train_status` reports `state == "Running"` (case-sensitive, matching
///   the `TrainingState` enum serialized by the kiln server). Otherwise,
///   transition to `Running`.
pub fn derive_state(
    health_ok: bool,
    consecutive_failures: u32,
    train_status: Option<&serde_json::Value>,
    prior: &ServerState,
) -> ServerState {
    if matches!(prior, ServerState::Stopped) {
        return ServerState::Stopped;
    }

    if matches!(prior, ServerState::Starting) && !health_ok {
        return ServerState::Starting;
    }

    if !health_ok {
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            return ServerState::Error("health check failed".into());
        }
        return prior.clone();
    }

    if let Some(status) = train_status {
        if has_running_job(status) {
            return ServerState::TrainingActive;
        }
    }
    ServerState::Running
}

fn health_response_ready(status_success: bool, body: Option<&serde_json::Value>) -> bool {
    if !status_success {
        return false;
    }

    let Some(body) = body else {
        return false;
    };

    inference_prewarm_complete(body).unwrap_or(true)
}

fn inference_prewarm_complete(body: &serde_json::Value) -> Option<bool> {
    body.get("checks")?
        .as_array()?
        .iter()
        .find(|check| {
            check.get("name").and_then(|name| name.as_str()) == Some("inference_prewarm_complete")
        })?
        .get("pass")?
        .as_bool()
}

fn has_running_job(status: &serde_json::Value) -> bool {
    let Some(arr) = status.as_array() else {
        return false;
    };
    arr.iter()
        .any(|item| item.get("state").and_then(|v| v.as_str()) == Some("Running"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn starting_plus_health_ok_transitions_to_running() {
        let got = derive_state(true, 0, None, &ServerState::Starting);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn starting_plus_health_fail_stays_starting_even_after_many_failures() {
        // Starting must never be overwritten by health failures — only a
        // successful health check can transition Starting into Running.
        let got = derive_state(false, 10, None, &ServerState::Starting);
        assert!(matches!(got, ServerState::Starting), "got {:?}", got);
    }

    #[test]
    fn health_ready_waits_for_inference_prewarm() {
        let health = json!({
            "status": "ok",
            "checks": [
                {"name": "model_loaded", "pass": true},
                {"name": "scheduler_responsive", "pass": true},
                {"name": "inference_prewarm_complete", "pass": false}
            ]
        });
        assert!(!health_response_ready(true, Some(&health)));
    }

    #[test]
    fn health_ready_accepts_completed_inference_prewarm() {
        let health = json!({
            "status": "ok",
            "checks": [
                {"name": "model_loaded", "pass": true},
                {"name": "scheduler_responsive", "pass": true},
                {"name": "inference_prewarm_complete", "pass": true}
            ]
        });
        assert!(health_response_ready(true, Some(&health)));
    }

    #[test]
    fn health_ready_accepts_older_health_without_prewarm_check() {
        let health = json!({
            "status": "ok",
            "checks": [
                {"name": "model_loaded", "pass": true},
                {"name": "scheduler_responsive", "pass": true}
            ]
        });
        assert!(health_response_ready(true, Some(&health)));
    }

    #[test]
    fn health_ready_rejects_failed_status_or_missing_body() {
        let health = json!({"status": "degraded", "checks": []});
        assert!(!health_response_ready(false, Some(&health)));
        assert!(!health_response_ready(true, None));
    }

    #[test]
    fn running_plus_two_failures_stays_running() {
        let got = derive_state(false, 2, None, &ServerState::Running);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn running_plus_three_failures_transitions_to_error() {
        let got = derive_state(false, 3, None, &ServerState::Running);
        assert!(matches!(got, ServerState::Error(_)), "got {:?}", got);
    }

    #[test]
    fn running_plus_training_job_transitions_to_training_active() {
        let train = json!([{"state": "Running", "loss": 0.42}]);
        let got = derive_state(true, 0, Some(&train), &ServerState::Running);
        assert!(matches!(got, ServerState::TrainingActive), "got {:?}", got);
    }

    #[test]
    fn training_active_plus_no_running_job_returns_to_running() {
        let train = json!([{"state": "Completed"}, {"state": "Queued"}]);
        let got = derive_state(true, 0, Some(&train), &ServerState::TrainingActive);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn training_active_plus_empty_array_returns_to_running() {
        let train = json!([]);
        let got = derive_state(true, 0, Some(&train), &ServerState::TrainingActive);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn training_active_plus_no_train_status_returns_to_running() {
        // If the train endpoint didn't return a parseable response but health
        // is OK, we can't confirm an active job — go back to Running.
        let got = derive_state(true, 0, None, &ServerState::TrainingActive);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn stopped_stays_stopped_regardless_of_inputs() {
        let train = json!([{"state": "Running"}]);
        let got = derive_state(true, 0, Some(&train), &ServerState::Stopped);
        assert!(matches!(got, ServerState::Stopped), "got {:?}", got);
        let got = derive_state(false, 10, None, &ServerState::Stopped);
        assert!(matches!(got, ServerState::Stopped), "got {:?}", got);
    }

    #[test]
    fn error_plus_health_ok_recovers_to_running() {
        let prior = ServerState::Error("health check failed".into());
        let got = derive_state(true, 0, None, &prior);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn error_plus_health_ok_with_training_job_goes_to_training_active() {
        let prior = ServerState::Error("health check failed".into());
        let train = json!([{"state": "Running"}]);
        let got = derive_state(true, 0, Some(&train), &prior);
        assert!(matches!(got, ServerState::TrainingActive), "got {:?}", got);
    }

    #[test]
    fn running_job_match_is_case_sensitive() {
        // Lowercase "running" must not match — the kiln server serializes the
        // `TrainingState::Running` variant as "Running" via default serde.
        let train = json!([{"state": "running"}]);
        let got = derive_state(true, 0, Some(&train), &ServerState::Running);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
        let train = json!([{"state": "RUNNING"}]);
        let got = derive_state(true, 0, Some(&train), &ServerState::Running);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn training_array_with_one_running_among_many_triggers_training_active() {
        let train = json!([
            {"state": "Completed"},
            {"state": "Running", "loss": 0.1},
            {"state": "Queued"},
        ]);
        let got = derive_state(true, 0, Some(&train), &ServerState::Running);
        assert!(matches!(got, ServerState::TrainingActive), "got {:?}", got);
    }

    #[test]
    fn non_array_train_status_is_ignored() {
        // If the kiln server ever returns an object instead of an array, we
        // should not crash — just treat it as "no active job".
        let train = json!({"state": "Running"});
        let got = derive_state(true, 0, Some(&train), &ServerState::Running);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }

    #[test]
    fn healthy_poll_resets_error_state_via_recovery_path() {
        // Simulate: Error -> one healthy poll -> Running.
        let prior = ServerState::Error("health check failed".into());
        let got = derive_state(true, 0, None, &prior);
        assert!(matches!(got, ServerState::Running), "got {:?}", got);
    }
}
