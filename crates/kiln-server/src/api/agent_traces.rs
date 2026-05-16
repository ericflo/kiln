//! Agent Trace Layer (grand plan §10.3).
//!
//! Consumes pi-format session JSONL as a first-class data source for
//! the agentic OPD path. Endpoints:
//!
//! - `POST /v1/agent/traces/discover` — index a directory of pi
//!   sessions (default `~/.pi/agent/sessions/`).
//! - `GET  /v1/agent/traces` — list indexed sessions with outcome
//!   heuristics.
//! - `GET  /v1/agent/traces/{id}` — fetch one session.
//!
//! Outcome heuristics (per §10.3):
//! - Did the session end with a `bash` exit-0 sequence on the user's
//!   task command?
//! - Did the user run `/tree` to fork (likely indicates the original
//!   branch went wrong)?
//! - Did the user manually edit files the agent had written?
//! - Is there a follow-up session in the same directory with similar
//!   intent (indicating repeat-attempts)?
//!
//! Privacy defaults (§10.3): all processing local by default. Sharing
//! requires explicit opt-in per session (pi-share-hf upstream). The
//! redaction layer is owned by pi itself; kiln consumes redacted
//! traces as-is.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

/// One pi session, normalised to kiln's trajectory schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrace {
    /// pi session id (UUID-like).
    pub id: String,
    /// Working directory the session ran in.
    pub working_dir: String,
    /// Number of turns (user + assistant).
    pub num_turns: usize,
    /// Number of tool calls observed in the trace.
    pub num_tool_calls: usize,
    /// Outcome heuristics. All None = no signal extracted.
    pub outcome: TraceOutcome,
    /// Wall-clock timestamps (RFC3339).
    pub first_event_at: Option<String>,
    pub last_event_at: Option<String>,
    /// Whether the user invoked `/tree` to fork (indicates a wrong
    /// branch).
    pub forked: bool,
    /// Parent session id when this trace is a fork of another.
    pub parent_id: Option<String>,
    /// pi tool manifest fingerprint at the time the session ran
    /// (§10.11 tool-schema versioning).
    pub tool_manifest_sha: Option<String>,
}

/// Outcome heuristics inferred from the session JSONL.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceOutcome {
    /// True when the last bash invocation exited 0.
    pub ended_with_exit_0: Option<bool>,
    /// User-edited file paths the agent had written. Empty when no
    /// post-agent edits detected.
    #[serde(default)]
    pub user_edited_agent_files: Vec<String>,
    /// True when a sibling session with a similar `intent` (heuristic
    /// match on the first user message) exists in the same directory.
    pub has_followup_attempt: Option<bool>,
}

/// In-memory index of discovered traces. Persisted to
/// `<adapter_dir>/agent_traces.json` on every change.
#[derive(Debug, Default)]
pub struct AgentTraceIndex {
    pub traces: BTreeMap<String, AgentTrace>,
}

impl AgentTraceIndex {
    pub fn load_from_path(path: &Path) -> Self {
        if !path.exists() {
            return Self::default();
        }
        match std::fs::read(path) {
            Ok(bytes) => match serde_json::from_slice::<BTreeMap<String, AgentTrace>>(&bytes) {
                Ok(traces) => Self { traces },
                Err(e) => {
                    tracing::warn!(error = %e, "agent_traces.json parse failed");
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "agent_traces.json read failed");
                Self::default()
            }
        }
    }

    pub fn save_to_path(&self, path: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.traces)?;
        std::fs::write(path, bytes)
    }
}

#[derive(Debug, Deserialize)]
struct DiscoverRequest {
    /// Path to the directory holding pi session JSONL files.
    /// Defaults to `$HOME/.pi/agent/sessions/`.
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Serialize)]
struct DiscoverResponse {
    indexed: usize,
    path: String,
}

fn default_pi_sessions_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".pi").join("agent").join("sessions")
    } else {
        PathBuf::from("/tmp/pi/agent/sessions")
    }
}

async fn discover_traces(
    State(state): State<AppState>,
    Json(req): Json<DiscoverRequest>,
) -> Result<Json<DiscoverResponse>, ApiError> {
    let path = req
        .path
        .map(PathBuf::from)
        .unwrap_or_else(default_pi_sessions_dir);
    if !path.exists() {
        return Err(ApiError::training_invalid_request(format!(
            "pi sessions dir {} does not exist (pass `path` to override)",
            path.display()
        )));
    }
    let mut index = AgentTraceIndex::default();
    let mut count = 0;
    for entry in std::fs::read_dir(&path)
        .map_err(|e| ApiError::internal(format!("read_dir {}: {e}", path.display())))?
    {
        let entry = entry.map_err(|e| ApiError::internal(format!("dir entry: {e}")))?;
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|s| s == "jsonl") {
            if let Some(trace) = parse_pi_session(&p) {
                index.traces.insert(trace.id.clone(), trace);
                count += 1;
            }
        }
    }
    let out_path = state.adapter_dir.join("agent_traces.json");
    if let Err(e) = index.save_to_path(&out_path) {
        tracing::warn!(error = %e, "failed to persist agent_traces.json");
    }
    Ok(Json(DiscoverResponse {
        indexed: count,
        path: path.display().to_string(),
    }))
}

#[derive(Debug, Serialize)]
struct AgentTracesListResponse {
    traces: Vec<AgentTrace>,
}

async fn list_traces(State(state): State<AppState>) -> Json<AgentTracesListResponse> {
    let path = state.adapter_dir.join("agent_traces.json");
    let index = AgentTraceIndex::load_from_path(&path);
    Json(AgentTracesListResponse {
        traces: index.traces.into_values().collect(),
    })
}

async fn get_trace(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<AgentTrace>, ApiError> {
    let path = state.adapter_dir.join("agent_traces.json");
    let index = AgentTraceIndex::load_from_path(&path);
    index
        .traces
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| ApiError::training_invalid_request(format!("trace {id} not indexed")))
}

/// Parse one pi session JSONL into an `AgentTrace`. Best-effort: pi's
/// schema is documented as `{messages, tool_calls, tool_results, id,
/// parentId}` per line; we accept any line that has `id`.
fn parse_pi_session(path: &Path) -> Option<AgentTrace> {
    let bytes = std::fs::read(path).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;
    let mut id: Option<String> = None;
    let mut parent_id: Option<String> = None;
    let mut num_turns = 0;
    let mut num_tool_calls = 0;
    let mut last_exit_code: Option<i64> = None;
    let mut first_event_at: Option<String> = None;
    let mut last_event_at: Option<String> = None;
    let mut forked = false;
    let mut tool_manifest_sha: Option<String> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if id.is_none() {
            if let Some(s) = v.get("id").and_then(|x| x.as_str()) {
                id = Some(s.to_string());
            }
        }
        if parent_id.is_none() {
            if let Some(s) = v.get("parentId").and_then(|x| x.as_str()) {
                parent_id = Some(s.to_string());
            }
        }
        if let Some(s) = v.get("tool_manifest_sha").and_then(|x| x.as_str()) {
            tool_manifest_sha = Some(s.to_string());
        }
        if v.get("messages").is_some() {
            num_turns += 1;
        }
        if let Some(arr) = v.get("tool_calls").and_then(|x| x.as_array()) {
            num_tool_calls += arr.len();
        }
        if v.get("event").and_then(|x| x.as_str()) == Some("/tree") {
            forked = true;
        }
        if let Some(rc) = v.get("exit_code").and_then(|x| x.as_i64()) {
            last_exit_code = Some(rc);
        }
        if let Some(ts) = v.get("at").and_then(|x| x.as_str()) {
            let ts = ts.to_string();
            if first_event_at.is_none() {
                first_event_at = Some(ts.clone());
            }
            last_event_at = Some(ts);
        }
    }

    let id = id?;
    let working_dir = path
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    Some(AgentTrace {
        id,
        working_dir,
        num_turns,
        num_tool_calls,
        outcome: TraceOutcome {
            ended_with_exit_0: last_exit_code.map(|c| c == 0),
            user_edited_agent_files: Vec::new(),
            has_followup_attempt: None,
        },
        first_event_at,
        last_event_at,
        forked,
        parent_id,
        tool_manifest_sha,
    })
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/agent/traces/discover", post(discover_traces))
        .route("/v1/agent/traces", get(list_traces))
        .route("/v1/agent/traces/{id}", get(get_trace))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parse_minimal_pi_session_extracts_id_and_counts_turns() {
        let dir = tempdir().unwrap();
        let session_path = dir.path().join("abc.jsonl");
        std::fs::write(
            &session_path,
            "\
{\"id\":\"session-abc\",\"at\":\"2026-05-15T10:00:00Z\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n\
{\"id\":\"session-abc\",\"at\":\"2026-05-15T10:00:01Z\",\"messages\":[{\"role\":\"assistant\",\"content\":\"hi back\"}],\"tool_calls\":[{\"name\":\"read\"}]}\n\
{\"id\":\"session-abc\",\"at\":\"2026-05-15T10:00:02Z\",\"exit_code\":0}\n\
",
        )
        .unwrap();
        let trace = parse_pi_session(&session_path).unwrap();
        assert_eq!(trace.id, "session-abc");
        assert_eq!(trace.num_turns, 2);
        assert_eq!(trace.num_tool_calls, 1);
        assert_eq!(trace.outcome.ended_with_exit_0, Some(true));
        assert!(!trace.forked);
        assert_eq!(trace.first_event_at.as_deref(), Some("2026-05-15T10:00:00Z"));
    }

    #[test]
    fn parse_session_detects_fork_event() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("fork.jsonl");
        std::fs::write(
            &p,
            r#"{"id":"s","event":"/tree","at":"2026-05-15T10:00:00Z"}
"#,
        )
        .unwrap();
        let t = parse_pi_session(&p).unwrap();
        assert!(t.forked);
    }

    #[test]
    fn parse_handles_garbage_lines_gracefully() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("g.jsonl");
        std::fs::write(
            &p,
            "this is not json\n{\"id\":\"x\",\"messages\":[]}\nanother junk line\n",
        )
        .unwrap();
        let t = parse_pi_session(&p).unwrap();
        assert_eq!(t.id, "x");
        assert_eq!(t.num_turns, 1);
    }

    #[test]
    fn parse_returns_none_when_no_id_in_session() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("noid.jsonl");
        std::fs::write(&p, "{\"messages\":[]}\n").unwrap();
        assert!(parse_pi_session(&p).is_none());
    }

    #[test]
    fn index_round_trips_through_disk() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_traces.json");
        let mut idx = AgentTraceIndex::default();
        idx.traces.insert(
            "abc".into(),
            AgentTrace {
                id: "abc".into(),
                working_dir: "/x".into(),
                num_turns: 5,
                num_tool_calls: 2,
                outcome: TraceOutcome::default(),
                first_event_at: Some("2026-05-15T10:00:00Z".into()),
                last_event_at: Some("2026-05-15T10:01:00Z".into()),
                forked: false,
                parent_id: None,
                tool_manifest_sha: Some("sha256:abc".into()),
            },
        );
        idx.save_to_path(&path).unwrap();
        let loaded = AgentTraceIndex::load_from_path(&path);
        assert_eq!(loaded.traces.len(), 1);
        assert_eq!(loaded.traces.get("abc").unwrap().num_turns, 5);
    }
}
