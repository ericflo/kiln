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
use kiln_train::pi_trajectory::{is_pi_message_event, parse_pi_session_str};
use kiln_train::trajectory::TurnSegment;
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
    /// Leading system/user context the session started from — the task
    /// scaffold the training bridge re-rolls (`agent_traces:` selectors).
    /// Empty on indices written before prompt capture; re-run discover.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_messages: Vec<kiln_train::ChatMessage>,
    /// Canonical trajectory parsed from Pi message events: Action and
    /// Observation segments, with any mid-session user/system turns
    /// preserved as Context segments (the masking layer skips Context).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trajectory: Vec<TurnSegment>,
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
        kiln_resource::locked_atomic_write(path, &bytes)
    }
}

#[derive(Debug, Deserialize)]
struct DiscoverRequest {
    /// Path to the directory holding pi session JSONL files.
    /// Defaults to the operating-system account's `.pi/agent/sessions/`.
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Serialize)]
struct DiscoverResponse {
    indexed: usize,
    path: String,
}

/// How deep below the sessions root to look for `*.jsonl` files. pi nests
/// sessions one level down under a per-project slug directory
/// (`sessions/<project-slug>/<session>.jsonl`); depth 3 leaves headroom for
/// a future extra level without risking a runaway walk of `$HOME`.
const SESSIONS_SCAN_MAX_DEPTH: usize = 3;

/// Recursively collect pi session JSONL files under `dir` (depth-capped) and
/// parse them into `index`. Returns how many sessions were indexed. Unreadable
/// subdirectories are skipped with a warning rather than failing the scan.
fn scan_sessions_dir(dir: &Path, depth_left: usize, index: &mut AgentTraceIndex) -> usize {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(dir = %dir.display(), error = %e, "agent trace scan: read_dir failed");
            return 0;
        }
    };
    let mut count = 0;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if depth_left > 0 {
                count += scan_sessions_dir(&p, depth_left - 1, index);
            }
        } else if p.is_file()
            && p.extension().is_some_and(|s| s == "jsonl")
            && let Some(trace) = parse_pi_session(&p)
        {
            index.traces.insert(trace.id.clone(), trace);
            count += 1;
        }
    }
    count
}

/// Build and persist `agent_traces.json` from a sessions directory —
/// shared by the explicit POST /v1/agent/traces/discover route and the
/// self_improve/judge_distill auto-discovery below. Returns the number
/// of indexed sessions.
///
/// Embedded-run sessions (`<adapter_dir>/agent_runs/sessions/`) are
/// always swept in too: a rebuild from the user's `~/.pi` dir must not
/// drop the rollouts kiln generated itself.
pub(crate) fn discover_traces_into(
    sessions_dir: &Path,
    adapter_dir: &Path,
) -> std::io::Result<usize> {
    let mut index = AgentTraceIndex::default();
    let mut count = scan_sessions_dir(sessions_dir, SESSIONS_SCAN_MAX_DEPTH, &mut index);
    let embedded = adapter_dir.join("agent_runs").join("sessions");
    if embedded.is_dir() && embedded != sessions_dir {
        count += scan_sessions_dir(&embedded, SESSIONS_SCAN_MAX_DEPTH, &mut index);
    }
    index.save_to_path(&adapter_dir.join("agent_traces.json"))?;
    Ok(count)
}

/// Merge one session JSONL into the persisted index — the embedded-run
/// finalizer calls this so every run kiln drives lands in the trace
/// layer immediately, without clobbering previously discovered traces.
///
/// The whole read-modify-write happens inside `locked_update`: two run
/// finalizers landing at once must both survive into the index (an
/// unlocked load→insert→save here lost one of them most of the time).
/// Returns `None` when the merge did not persist, so callers don't
/// report a trace as indexed that isn't.
pub(crate) fn index_session_file(adapter_dir: &Path, session_path: &Path) -> Option<AgentTrace> {
    let trace = parse_pi_session(session_path)?;
    let index_path = adapter_dir.join("agent_traces.json");
    let merged = kiln_resource::locked_update(&index_path, |existing| {
        let mut traces: BTreeMap<String, AgentTrace> = existing
            .and_then(|bytes| serde_json::from_slice(bytes).ok())
            .unwrap_or_default();
        traces.insert(trace.id.clone(), trace.clone());
        serde_json::to_vec_pretty(&traces).map_err(std::io::Error::other)
    });
    match merged {
        Ok(()) => Some(trace),
        Err(e) => {
            tracing::warn!(error = %e, "agent_traces.json merge write failed");
            None
        }
    }
}

/// Auto-discovery for the §10.6 endpoints: when `agent_traces.json` is
/// missing, scan the default pi sessions dir and build it — the
/// canonical onboarding (`kiln serve` → use pi → `kiln self-improve`)
/// must not hard-fail asking for a manual POST
/// /v1/agent/traces/discover first. An existing index is left alone
/// (re-discovery stays explicit), and a missing sessions dir falls
/// through to the resolver's actionable error.
pub(crate) fn ensure_agent_trace_index(
    adapter_dir: &Path,
    pi_sessions_dir: &Path,
) -> Option<usize> {
    if adapter_dir.join("agent_traces.json").exists() {
        return None;
    }
    if !pi_sessions_dir.exists() {
        return None;
    }
    match discover_traces_into(pi_sessions_dir, adapter_dir) {
        Ok(count) => {
            tracing::info!(
                indexed = count,
                sessions_dir = %pi_sessions_dir.display(),
                "auto-discovered pi agent traces for self_improve"
            );
            Some(count)
        }
        Err(e) => {
            tracing::warn!(error = %e, "agent-trace auto-discovery failed");
            None
        }
    }
}

async fn discover_traces(
    State(state): State<AppState>,
    Json(req): Json<DiscoverRequest>,
) -> Result<Json<DiscoverResponse>, ApiError> {
    let path = req
        .path
        .map(PathBuf::from)
        .unwrap_or_else(|| state.operational_runtime.pi_sessions_dir.clone());
    if !path.exists() {
        return Err(ApiError::training_invalid_request(format!(
            "pi sessions dir {} does not exist (pass `path` to override)",
            path.display()
        )));
    }
    let count = match discover_traces_into(&path, &state.adapter_dir) {
        Ok(count) => count,
        Err(e) => {
            tracing::warn!(error = %e, "failed to persist agent_traces.json");
            0
        }
    };
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

/// Parse one pi session JSONL into an `AgentTrace`. Best-effort: accepts both
/// the legacy summary rows with `{messages, tool_calls, id, parentId}` and the
/// observed Pi 0.75.x event stream with `{type:"message", message:{...}}`.
fn parse_pi_session(path: &Path) -> Option<AgentTrace> {
    let bytes = std::fs::read(path).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;
    // include_context=true keeps user/system turns ordered within the
    // trajectory; the leading Context run is split off below as the prompt
    // scaffold so the training bridge can re-roll the session's task.
    let parsed_pi = parse_pi_session_str(text, true);
    let mut id: Option<String> = None;
    let mut cwd: Option<String> = None;
    let mut parent_id: Option<String> = None;
    let mut num_turns = 0;
    let mut num_tool_calls = 0;
    let mut last_exit_code: Option<i64> = None;
    let mut first_event_at: Option<String> = None;
    let mut last_event_at: Option<String> = None;
    let mut forked = false;
    let mut tool_manifest_sha: Option<String> = None;
    let mut saw_pi_message_event = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if id.is_none()
            && let Some(s) = v.get("id").and_then(|x| x.as_str())
        {
            id = Some(s.to_string());
        }
        if cwd.is_none() {
            // Pi's session header row ({"type":"session", ..., "cwd": "..."})
            // names the real working directory — the parent path is only the
            // slug-encoded (and ambiguous) project directory.
            if let Some(s) = v.get("cwd").and_then(|x| x.as_str()) {
                cwd = Some(s.to_string());
            }
        }
        if parent_id.is_none()
            && let Some(s) = v.get("parentId").and_then(|x| x.as_str())
        {
            parent_id = Some(s.to_string());
        }
        if let Some(s) = v.get("tool_manifest_sha").and_then(|x| x.as_str()) {
            tool_manifest_sha = Some(s.to_string());
        }
        if is_pi_message_event(&v) {
            saw_pi_message_event = true;
            if let Some(message) = v.get("message").and_then(|x| x.as_object())
                && let Some(role) = message.get("role").and_then(|x| x.as_str())
            {
                if matches!(role, "user" | "assistant") {
                    num_turns += 1;
                }
                if role == "assistant"
                    && let Some(blocks) = message.get("content").and_then(|x| x.as_array())
                {
                    num_tool_calls += blocks
                        .iter()
                        .filter(|block| {
                            block.get("type").and_then(|x| x.as_str()) == Some("toolCall")
                        })
                        .count();
                }
            }
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
        // Legacy summary rows stamp "at"; the Pi 0.75.x event stream stamps
        // "timestamp". Accept both so real sessions get first/last times.
        if let Some(ts) = v
            .get("at")
            .or_else(|| v.get("timestamp"))
            .and_then(|x| x.as_str())
        {
            let ts = ts.to_string();
            if first_event_at.is_none() {
                first_event_at = Some(ts.clone());
            }
            last_event_at = Some(ts);
        }
    }

    let id = id.or_else(|| {
        saw_pi_message_event
            .then(|| {
                path.file_stem()
                    .map(|stem| stem.to_string_lossy().to_string())
            })
            .flatten()
    })?;
    let working_dir = cwd.unwrap_or_else(|| {
        path.parent()
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    });
    // Split the leading Context run (system/user turns before the first
    // action) into the prompt scaffold; mid-session Context turns stay in
    // the trajectory, ordered, where the masking layer skips them.
    let (prompt_messages, trajectory) = if saw_pi_message_event {
        let mut trajectory = parsed_pi.trajectory;
        let leading = trajectory
            .iter()
            .take_while(|seg| seg.kind == kiln_train::trajectory::TurnKind::Context)
            .count();
        let prompt_messages = trajectory
            .drain(..leading)
            .map(|seg| kiln_train::ChatMessage {
                role: seg.role,
                content: seg.content,
                tool_call_id: seg.tool_call_id,
                ..Default::default()
            })
            .collect();
        (prompt_messages, trajectory)
    } else {
        (Vec::new(), Vec::new())
    };
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
        prompt_messages,
        trajectory,
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
    use serde_json::Value;
    use serde_json::json;
    use tempfile::tempdir;

    fn write_jsonl(path: &Path, rows: &[Value]) {
        let mut body = String::new();
        for row in rows {
            body.push_str(&serde_json::to_string(row).unwrap());
            body.push('\n');
        }
        std::fs::write(path, body).unwrap();
    }

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
        assert_eq!(
            trace.first_event_at.as_deref(),
            Some("2026-05-15T10:00:00Z")
        );
        assert!(trace.trajectory.is_empty());
    }

    #[test]
    fn parse_pi_0751_message_events_normalizes_trajectory() {
        let dir = tempdir().unwrap();
        let session_path = dir.path().join("pi0751.jsonl");
        write_jsonl(
            &session_path,
            &[
                json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"Print 42"}]}}),
                json!({"type":"message","message":{"role":"assistant","content":[{"type":"thinking","thinking":"use bash"},{"type":"toolCall","name":"bash","input":{"cmd":"python3 -c 'print(42)'"},"id":"c1"}]}}),
                json!({"type":"message","message":{"role":"tool","content":[{"type":"toolResult","content":"42\n","toolCallId":"c1"}]}}),
                json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Done"}]}}),
            ],
        );

        let trace = parse_pi_session(&session_path).unwrap();

        assert_eq!(trace.id, "pi0751");
        assert_eq!(trace.num_turns, 3);
        assert_eq!(trace.num_tool_calls, 1);
        // The leading user turn is the task scaffold, not a trajectory
        // segment — the training bridge re-rolls from it.
        assert_eq!(trace.prompt_messages.len(), 1);
        assert_eq!(trace.prompt_messages[0].role, "user");
        assert_eq!(trace.prompt_messages[0].content, "Print 42");
        assert_eq!(trace.trajectory.len(), 3);
        assert_eq!(trace.trajectory[0].role, "assistant");
        assert_eq!(
            trace.trajectory[0].kind,
            kiln_train::trajectory::TurnKind::Action
        );
        assert!(
            trace.trajectory[0]
                .content
                .contains("<think>use bash</think>")
        );
        assert!(trace.trajectory[0].content.contains("\"arguments\""));
        assert_eq!(trace.trajectory[1].role, "tool");
        assert_eq!(
            trace.trajectory[1].kind,
            kiln_train::trajectory::TurnKind::Observation
        );
        assert_eq!(trace.trajectory[1].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(trace.trajectory[1].content, "42\n");
        assert_eq!(trace.trajectory[2].content, "Done");
    }

    #[test]
    fn parse_keeps_mid_session_user_turns_as_context_segments() {
        let dir = tempdir().unwrap();
        let session_path = dir.path().join("midctx.jsonl");
        write_jsonl(
            &session_path,
            &[
                json!({"type":"message","message":{"role":"system","content":[{"type":"text","text":"You are pi."}]}}),
                json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"Run the tests"}]}}),
                json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Running."}]}}),
                json!({"type":"message","message":{"role":"user","content":[{"type":"text","text":"Now fix the failure"}]}}),
                json!({"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"Fixed."}]}}),
            ],
        );

        let trace = parse_pi_session(&session_path).unwrap();

        // Leading system+user run = scaffold.
        assert_eq!(trace.prompt_messages.len(), 2);
        assert_eq!(trace.prompt_messages[0].role, "system");
        assert_eq!(trace.prompt_messages[1].content, "Run the tests");
        // Mid-session user turn stays ordered inside the trajectory as
        // Context, between the two actions.
        assert_eq!(trace.trajectory.len(), 3);
        assert_eq!(trace.trajectory[0].content, "Running.");
        assert_eq!(
            trace.trajectory[1].kind,
            kiln_train::trajectory::TurnKind::Context
        );
        assert_eq!(trace.trajectory[1].content, "Now fix the failure");
        assert_eq!(trace.trajectory[2].content, "Fixed.");
    }

    #[test]
    fn parse_pi_0753_tool_result_role_normalizes_to_tool() {
        let dir = tempdir().unwrap();
        let session_path = dir.path().join("pi0753.jsonl");
        write_jsonl(
            &session_path,
            &[
                json!({"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","name":"bash","input":{"cmd":"echo ok"},"id":"c1"}]}}),
                json!({"type":"message","message":{"role":"toolResult","content":[{"type":"text","text":"ok\n"}]}}),
            ],
        );

        let trace = parse_pi_session(&session_path).unwrap();

        assert_eq!(trace.id, "pi0753");
        assert_eq!(trace.num_turns, 1);
        assert_eq!(trace.num_tool_calls, 1);
        assert_eq!(trace.trajectory.len(), 2);
        assert_eq!(trace.trajectory[1].role, "tool");
        assert_eq!(
            trace.trajectory[1].kind,
            kiln_train::trajectory::TurnKind::Observation
        );
        assert_eq!(trace.trajectory[1].content, "ok\n");
    }

    #[test]
    fn scan_finds_sessions_nested_under_project_slug_dirs() {
        // Real pi layout: sessions/<project-slug>/<session>.jsonl — the
        // pre-recursive scan indexed zero of these.
        let dir = tempdir().unwrap();
        let slug_dir = dir.path().join("--home-user-Development-proj--");
        std::fs::create_dir(&slug_dir).unwrap();
        write_jsonl(
            &slug_dir.join("nested.jsonl"),
            &[
                json!({"type":"session","id":"nested-session","timestamp":"2026-06-01T10:00:00Z","cwd":"/home/user/Development/proj"}),
                json!({"type":"message","timestamp":"2026-06-01T10:00:05Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}),
            ],
        );
        // A flat file at the root must still be picked up.
        write_jsonl(
            &dir.path().join("flat.jsonl"),
            &[
                json!({"id":"flat-session","at":"2026-06-01T11:00:00Z","messages":[{"role":"user","content":"hi"}]}),
            ],
        );
        // Non-jsonl files and too-deep nesting are ignored.
        std::fs::write(slug_dir.join("notes.txt"), "ignore me").unwrap();

        let mut index = AgentTraceIndex::default();
        let count = scan_sessions_dir(dir.path(), SESSIONS_SCAN_MAX_DEPTH, &mut index);

        assert_eq!(count, 2);
        assert!(index.traces.contains_key("nested-session"));
        assert!(index.traces.contains_key("flat-session"));
    }

    #[test]
    fn scan_depth_cap_stops_runaway_walks() {
        let dir = tempdir().unwrap();
        let mut deep = dir.path().to_path_buf();
        for level in 0..5 {
            deep = deep.join(format!("level{level}"));
        }
        std::fs::create_dir_all(&deep).unwrap();
        write_jsonl(
            &deep.join("too-deep.jsonl"),
            &[json!({"id":"too-deep","messages":[{"role":"user","content":"hi"}]})],
        );

        let mut index = AgentTraceIndex::default();
        let count = scan_sessions_dir(dir.path(), SESSIONS_SCAN_MAX_DEPTH, &mut index);

        assert_eq!(count, 0);
        assert!(index.traces.is_empty());
    }

    #[test]
    fn parse_session_header_supplies_working_dir_and_timestamps() {
        // The session header's cwd is the real working directory; the parent
        // path is only pi's slug-encoded (ambiguous) project name. Event rows
        // stamp "timestamp" in Pi 0.75.x, not the legacy "at".
        let dir = tempdir().unwrap();
        let p = dir.path().join("019e7acf.jsonl");
        write_jsonl(
            &p,
            &[
                json!({"type":"session","version":3,"id":"019e7acf","timestamp":"2026-05-30T21:35:00.144Z","cwd":"/tmp/rlm-agentic-test"}),
                json!({"type":"message","timestamp":"2026-05-30T21:35:02.000Z","message":{"role":"user","content":[{"type":"text","text":"go"}]}}),
                json!({"type":"message","timestamp":"2026-05-30T21:36:00.000Z","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}),
            ],
        );
        let t = parse_pi_session(&p).unwrap();
        assert_eq!(t.working_dir, "/tmp/rlm-agentic-test");
        assert_eq!(
            t.first_event_at.as_deref(),
            Some("2026-05-30T21:35:00.144Z")
        );
        assert_eq!(t.last_event_at.as_deref(), Some("2026-05-30T21:36:00.000Z"));
    }

    #[test]
    fn parse_without_session_header_falls_back_to_parent_dir() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("legacy.jsonl");
        write_jsonl(
            &p,
            &[
                json!({"id":"legacy","at":"2026-05-15T10:00:00Z","messages":[{"role":"user","content":"hi"}]}),
            ],
        );
        let t = parse_pi_session(&p).unwrap();
        assert_eq!(t.working_dir, dir.path().display().to_string());
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
                prompt_messages: Vec::new(),
                trajectory: Vec::new(),
            },
        );
        idx.save_to_path(&path).unwrap();
        let loaded = AgentTraceIndex::load_from_path(&path);
        assert_eq!(loaded.traces.len(), 1);
        assert_eq!(loaded.traces.get("abc").unwrap().num_turns, 5);
    }
    /// The shared discovery routine persists the index, and
    /// ensure_agent_trace_index leaves an existing index alone (explicit
    /// re-discovery only).
    #[test]
    fn discover_traces_into_persists_and_ensure_respects_existing() {
        let sessions = tempfile::tempdir().unwrap();
        let adapters = tempfile::tempdir().unwrap();
        write_jsonl(
            &sessions.path().join("s.jsonl"),
            &[
                serde_json::json!({"type":"session","id":"s1","timestamp":"2026-06-01T10:00:00Z","cwd":"/p"}),
                serde_json::json!({"type":"message","timestamp":"2026-06-01T10:00:05Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}),
            ],
        );
        let count = discover_traces_into(sessions.path(), adapters.path()).unwrap();
        assert_eq!(count, 1);
        let index_path = adapters.path().join("agent_traces.json");
        assert!(index_path.exists());

        // An existing index is never clobbered by the auto path.
        std::fs::write(&index_path, "{\"sessions\":[]}").unwrap();
        assert!(ensure_agent_trace_index(adapters.path(), sessions.path()).is_none());
        assert_eq!(
            std::fs::read_to_string(&index_path).unwrap(),
            "{\"sessions\":[]}"
        );
    }
}
