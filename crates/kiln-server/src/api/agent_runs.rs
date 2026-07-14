//! Embedded agent runs API — submit tasks for the server-driven pi.
//!
//! | Method | Path                       | Purpose                                |
//! |--------|----------------------------|----------------------------------------|
//! | GET    | /v1/agent/runs/status      | Gate state, pi availability, capacity  |
//! | POST   | /v1/agent/runs             | Start a run `{task, cwd?, label?, ...}`|
//! | GET    | /v1/agent/runs             | List runs (newest first)               |
//! | GET    | /v1/agent/runs/{id}        | One run record                         |
//! | GET    | /v1/agent/runs/{id}/events | Event feed: returns events with seq >= `after`; pass the response's `next_after` back to poll incrementally |
//! | POST   | /v1/agent/runs/{id}/steer  | Queue a steering message               |
//! | POST   | /v1/agent/runs/{id}/follow_up | Queue a follow-up task              |
//! | POST   | /v1/agent/runs/{id}/abort  | Abort the run                          |
//!
//! Security gate: an embedded run is arbitrary-code-execution grade —
//! same posture as the dashboard terminal. Enabled when the server is
//! bound to loopback (the default); `[agent].runs_access = "enabled"` opts in
//! on network binds, while `"disabled"` force-disables.

use std::path::PathBuf;

use axum::extract::{Path as AxumPath, Query, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::agent_runs::{AgentRunRecord, ControlError, NewRunParams, StartRunError};
use crate::error::ApiError;
use crate::state::AppState;

const THINKING_LEVELS: &[&str] = &["off", "minimal", "low", "medium", "high", "xhigh"];

/// (enabled, human-readable reason when disabled)
fn runs_gate(state: &AppState) -> (bool, Option<String>) {
    use crate::config::LocalCapabilityAccess;

    let runtime = state.operational_runtime.as_ref();
    if runtime.agent_runs_enabled {
        return (true, None);
    }
    match runtime.agent_runs_access {
        LocalCapabilityAccess::Disabled => (
            false,
            Some("disabled by agent.runs_access on the server".into()),
        ),
        LocalCapabilityAccess::LoopbackOnly => (
            false,
            Some(format!(
                "the server is bound to {} (not loopback) — embedded runs execute arbitrary \
                 code. Set agent.runs_access=\"enabled\" to opt in.",
                runtime.bind_host
            )),
        ),
        LocalCapabilityAccess::Enabled => (false, Some("agent run gate is inconsistent".into())),
    }
}

/// The gate guards reads too: run records and event feeds carry task
/// prompts, server paths, and raw tool output — closing the gate must
/// take the data side off the network, not just creation. /status
/// stays open (it only reports the gate itself).
fn require_runs_enabled(state: &AppState) -> Result<(), ApiError> {
    let (enabled, reason) = runs_gate(state);
    if !enabled {
        return Err(ApiError::agent_runs_disabled(
            reason.unwrap_or_else(|| "gate closed".into()),
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct AgentRunsStatusResponse {
    enabled: bool,
    disabled_reason: Option<String>,
    pi_available: bool,
    pi_path: Option<String>,
    max_concurrent_runs: usize,
    active_runs: usize,
    sessions_dir: String,
}

async fn runs_status(State(state): State<AppState>) -> Json<AgentRunsStatusResponse> {
    let (enabled, reason) = runs_gate(&state);
    let pi = state.operational_runtime.pi_bin.clone();
    Json(AgentRunsStatusResponse {
        enabled,
        disabled_reason: reason,
        pi_available: pi.is_some(),
        pi_path: pi.map(|path| path.display().to_string()),
        max_concurrent_runs: state.agent_runs.max_concurrent(),
        active_runs: state.agent_runs.active_count(),
        sessions_dir: state.agent_runs.sessions_dir().display().to_string(),
    })
}

#[derive(Debug, Deserialize)]
struct CreateRunRequest {
    /// The task prompt handed to pi.
    task: String,
    /// Working directory for the agent. Defaults to the server's cwd.
    #[serde(default)]
    cwd: Option<String>,
    /// Free-form grouping label (e.g. a rollout batch name).
    #[serde(default)]
    label: Option<String>,
    /// Tool allowlist (pi `--tools`), e.g. `["read", "bash"]`.
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// pi thinking level: off|minimal|low|medium|high|xhigh.
    #[serde(default)]
    thinking_level: Option<String>,
    /// Per-run wall-clock cap; defaults to `[agent].run_timeout_secs`.
    #[serde(default)]
    timeout_secs: Option<u64>,
}

async fn create_run(
    State(state): State<AppState>,
    Json(req): Json<CreateRunRequest>,
) -> Result<Json<AgentRunRecord>, ApiError> {
    if state.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(ApiError::shutting_down());
    }
    state
        .ensure_inference_admission_allowed()
        .map_err(|_| ApiError::inference_disabled_by_profile(state.serving_profile.profile()))?;
    require_runs_enabled(&state)?;
    let task = req.task.trim();
    if task.is_empty() {
        return Err(ApiError::agent_run_invalid_request(
            "`task` must be non-empty",
        ));
    }
    if let Some(level) = &req.thinking_level {
        if !THINKING_LEVELS.contains(&level.as_str()) {
            return Err(ApiError::agent_run_invalid_request(format!(
                "`thinking_level` must be one of {THINKING_LEVELS:?}, got '{level}'"
            )));
        }
    }
    if let Some(secs) = req.timeout_secs {
        if secs < 10 {
            return Err(ApiError::agent_run_invalid_request(
                "`timeout_secs` must be >= 10",
            ));
        }
    }
    let cwd = match &req.cwd {
        Some(dir) => {
            let p = PathBuf::from(dir);
            if !p.is_dir() {
                return Err(ApiError::agent_run_invalid_request(format!(
                    "`cwd` {} is not a directory on the server",
                    p.display()
                )));
            }
            p
        }
        None => std::env::current_dir().map_err(|e| {
            ApiError::agent_run_invalid_request(format!("server cwd unavailable: {e}"))
        })?,
    };
    let Some(pi_bin) = state.operational_runtime.pi_bin.clone() else {
        return Err(ApiError::agent_runs_unavailable(
            "`pi` was not resolved from agent.pi_bin or the startup PATH",
        ));
    };
    let record = state.agent_runs.start_run(NewRunParams {
        task: task.to_string(),
        cwd,
        label: req.label.clone(),
        tools: req.tools.clone(),
        thinking_level: req.thinking_level.clone(),
        timeout_secs: req.timeout_secs,
        pi_bin,
        model: state.served_model_id.clone(),
        kiln_url: Some(crate::agent_runs::self_url()),
    });
    match record {
        Ok(record) => Ok(Json(record)),
        Err(StartRunError::AtCapacity(max)) => Err(ApiError::agent_runs_at_capacity(max)),
    }
}

#[derive(Debug, Deserialize)]
struct ListRunsQuery {
    /// Filter runs by exact label.
    #[serde(default)]
    label: Option<String>,
}

async fn list_runs(
    State(state): State<AppState>,
    Query(q): Query<ListRunsQuery>,
) -> Result<Json<AgentRunListResponse>, ApiError> {
    require_runs_enabled(&state)?;
    let mut runs = state.agent_runs.list();
    if let Some(label) = &q.label {
        runs.retain(|r| r.label.as_deref() == Some(label.as_str()));
    }
    Ok(Json(AgentRunListResponse { runs }))
}

#[derive(Debug, Serialize)]
struct AgentRunListResponse {
    runs: Vec<AgentRunRecord>,
}

async fn get_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<AgentRunRecord>, ApiError> {
    require_runs_enabled(&state)?;
    state
        .agent_runs
        .get(&id)
        .map(Json)
        .ok_or_else(|| ApiError::agent_run_not_found(&id))
}

#[derive(Debug, Deserialize)]
struct EventsQuery {
    /// Return events with seq >= after (cursor for incremental polls).
    #[serde(default)]
    after: u64,
}

async fn run_events(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<EventsQuery>,
) -> Result<Json<AgentRunEventsResponse>, ApiError> {
    require_runs_enabled(&state)?;
    let Some(page) = state.agent_runs.events_after(&id, q.after) else {
        return Err(ApiError::agent_run_not_found(&id));
    };
    let next_after = page
        .events
        .last()
        .map(|(seq, _)| seq + 1)
        .unwrap_or(q.after);
    let events: Vec<AgentRunEvent> = page
        .events
        .into_iter()
        .map(|(seq, event)| AgentRunEvent { seq, event })
        .collect();
    Ok(Json(AgentRunEventsResponse {
        events,
        next_after,
        status: page.status,
        // Replay-gap detection: events before the cursor are gone when
        // truncated (ring prune on a long run, or a server restart).
        first_available_seq: page.first_available_seq,
        truncated: page.truncated,
    }))
}

#[derive(Debug, Serialize)]
struct AgentRunEvent {
    seq: u64,
    event: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AgentRunEventsResponse {
    events: Vec<AgentRunEvent>,
    next_after: u64,
    status: crate::agent_runs::RunStatus,
    first_available_seq: Option<u64>,
    truncated: bool,
}

#[derive(Debug, Deserialize)]
struct MessageRequest {
    message: String,
}

fn map_control_error(id: &str, err: ControlError) -> ApiError {
    match err {
        ControlError::NotFound => ApiError::agent_run_not_found(id),
        ControlError::NotActive(status) => ApiError::agent_run_not_active(id, status),
    }
}

async fn steer_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(req): Json<MessageRequest>,
) -> Result<Json<AgentRunQueuedResponse>, ApiError> {
    require_runs_enabled(&state)?;
    if req.message.trim().is_empty() {
        return Err(ApiError::agent_run_invalid_request(
            "`message` must be non-empty",
        ));
    }
    state
        .agent_runs
        .steer(&id, req.message)
        .map_err(|e| map_control_error(&id, e))?;
    Ok(Json(AgentRunQueuedResponse { queued: true }))
}

async fn follow_up_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(req): Json<MessageRequest>,
) -> Result<Json<AgentRunQueuedResponse>, ApiError> {
    require_runs_enabled(&state)?;
    if req.message.trim().is_empty() {
        return Err(ApiError::agent_run_invalid_request(
            "`message` must be non-empty",
        ));
    }
    state
        .agent_runs
        .follow_up(&id, req.message)
        .map_err(|e| map_control_error(&id, e))?;
    Ok(Json(AgentRunQueuedResponse { queued: true }))
}

#[derive(Debug, Serialize)]
struct AgentRunQueuedResponse {
    queued: bool,
}

async fn abort_run(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<AgentRunAbortResponse>, ApiError> {
    require_runs_enabled(&state)?;
    state
        .agent_runs
        .abort(&id)
        .map_err(|e| map_control_error(&id, e))?;
    Ok(Json(AgentRunAbortResponse { aborting: true }))
}

#[derive(Debug, Serialize)]
struct AgentRunAbortResponse {
    aborting: bool,
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/agent/runs/status", get(runs_status))
        .route("/v1/agent/runs", post(create_run).get(list_runs))
        .route("/v1/agent/runs/{id}", get(get_run))
        .route("/v1/agent/runs/{id}/events", get(run_events))
        .route("/v1/agent/runs/{id}/steer", post(steer_run))
        .route("/v1/agent/runs/{id}/follow_up", post(follow_up_run))
        .route("/v1/agent/runs/{id}/abort", post(abort_run))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_run_request_minimal_body_parses() {
        let req: CreateRunRequest = serde_json::from_str(r#"{"task": "fix the test"}"#).unwrap();
        assert_eq!(req.task, "fix the test");
        assert!(req.cwd.is_none());
        assert!(req.tools.is_none());
        assert!(req.timeout_secs.is_none());
    }
}
