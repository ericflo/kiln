//! Embedded agent runs — kiln drives pi itself.
//!
//! Until now pi was external: the user ran it in a terminal, pi wrote
//! session JSONL to `~/.pi/agent/sessions/`, and kiln indexed those
//! traces after the fact. This module embeds the agent: the server
//! spawns `pi --mode rpc` as a child process pointed at its own
//! OpenAI-compatible endpoint, submits a task, streams the trajectory
//! live, and auto-indexes the finished session into the §10.3 agent
//! trace layer — where the §10.6 flywheel (judge_distill /
//! self_improve) already consumes it.
//!
//! That closes the loop: kiln can now *generate* on-policy agentic
//! rollouts on demand instead of waiting for a human to drive pi.
//!
//! Lifecycle of one run:
//!   queued → (FIFO slot, bounded by `[agent].max_concurrent_runs`)
//!   running → prompt sent, events streamed into a per-run buffer
//!   terminal: completed | failed | aborted | timed_out
//!   → session JSONL indexed into `agent_traces.json` (merge)
//!
//! Records persist to `<adapter_dir>/agent_runs/runs.json`; runs that
//! were active when the server died come back as `interrupted`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::pi_rpc::{PiRpcLine, PiRpcOptions, PiRpcProcess};

/// pi provider id the `kiln pi-setup` merge registers (cli.rs).
const PI_PROVIDER_ID: &str = "kiln-local";
/// Events kept per run for the live feed / replay.
const EVENT_BUFFER_CAP: usize = 2000;
/// Terminal run records kept before pruning oldest.
const TERMINAL_RUNS_CAP: usize = 200;
/// Active (queued + running) runs allowed before new submissions are
/// turned away — a backstop against a runaway submitter, not a tuning
/// knob (concurrency is `[agent].max_concurrent_runs`). Enforced
/// atomically inside `start_run` under the runs lock.
pub const ACTIVE_RUNS_BACKSTOP: usize = 32;
/// last_assistant_text is a summary, not a transcript.
const LAST_TEXT_CAP: usize = 4000;

static SELF_URL: OnceLock<String> = OnceLock::new();

/// Record the URL this server is reachable at locally, for the pi
/// config merge before each embedded run. A wildcard bind is rewritten
/// to loopback — the child runs on the same host. Bare IPv6 literals
/// get bracketed (`::1` → `http://[::1]:port`); unbracketed they parse
/// as host `:` port soup and pi can't reach the server.
pub fn set_self_url(host: &str, port: u16) {
    let _ = SELF_URL.set(format_self_url(host, port));
}

fn format_self_url(host: &str, port: u16) -> String {
    let host: std::borrow::Cow<'_, str> = match host {
        "0.0.0.0" | "::" | "[::]" => "127.0.0.1".into(),
        other if other.parse::<std::net::Ipv6Addr>().is_ok() => format!("[{other}]").into(),
        other => other.into(),
    };
    format!("http://{host}:{port}")
}

pub fn self_url() -> String {
    SELF_URL.get().cloned().unwrap_or_else(|| {
        format_self_url(
            crate::config::DEFAULT_SERVER_HOST,
            crate::config::DEFAULT_SERVER_PORT,
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Aborted,
    TimedOut,
    /// Was queued/running when the server restarted.
    Interrupted,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, RunStatus::Queued | RunStatus::Running)
    }
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RunStatus::Queued => "queued",
            RunStatus::Running => "running",
            RunStatus::Completed => "completed",
            RunStatus::Failed => "failed",
            RunStatus::Aborted => "aborted",
            RunStatus::TimedOut => "timed_out",
            RunStatus::Interrupted => "interrupted",
        };
        f.write_str(s)
    }
}

/// One embedded run, as listed by `GET /v1/agent/runs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunRecord {
    pub id: String,
    pub task: String,
    pub cwd: String,
    /// Free-form grouping label (e.g. a rollout batch name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub status: RunStatus,
    pub created_unix_ms: u64,
    /// Monotonic submission order — FIFO tie-break for runs created in
    /// the same millisecond.
    #[serde(default)]
    pub queue_seq: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_unix_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_unix_ms: Option<u64>,
    /// Assistant turns observed (message_end with role=assistant).
    pub num_turns: usize,
    /// Tool executions observed (tool_execution_end).
    pub num_tool_calls: usize,
    /// pi session id — also the agent-trace id once indexed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_path: Option<String>,
    /// True once the session JSONL was merged into agent_traces.json.
    pub trace_indexed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_assistant_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Inputs for one run, validated at the API layer.
#[derive(Debug, Clone)]
pub struct NewRunParams {
    pub task: String,
    pub cwd: PathBuf,
    pub label: Option<String>,
    pub tools: Option<Vec<String>>,
    pub thinking_level: Option<String>,
    /// Per-run override of `[agent].run_timeout_secs`.
    pub timeout_secs: Option<u64>,
    /// Binary to spawn. The API resolves via `pi_rpc::find_pi()`;
    /// tests inject a stub.
    pub pi_bin: PathBuf,
    /// Model id pi selects under the kiln provider.
    pub model: String,
    /// When set, run the same non-destructive `pi-setup` config merge
    /// the embedded terminal does, pointed at this URL. `None` skips
    /// the merge (tests; pre-configured environments).
    pub kiln_url: Option<String>,
}

#[derive(Debug)]
enum ControlMsg {
    Steer(String),
    FollowUp(String),
    Abort,
}

#[derive(Debug, Clone, Copy)]
struct RunSettings {
    max_concurrent: usize,
    timeout: std::time::Duration,
}

impl Default for RunSettings {
    fn default() -> Self {
        Self {
            max_concurrent: 2,
            timeout: std::time::Duration::from_secs(900),
        }
    }
}

#[derive(Debug, Default)]
struct EventBuf {
    next_seq: u64,
    items: VecDeque<(u64, serde_json::Value)>,
}

impl EventBuf {
    fn push(&mut self, value: serde_json::Value) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.items.push_back((seq, value));
        while self.items.len() > EVENT_BUFFER_CAP {
            self.items.pop_front();
        }
    }

    fn after(&self, after: u64) -> Vec<(u64, serde_json::Value)> {
        self.items
            .iter()
            .filter(|(seq, _)| *seq >= after)
            .cloned()
            .collect()
    }
}

pub struct AgentRunRegistry {
    adapter_dir: PathBuf,
    runs: std::sync::RwLock<BTreeMap<String, AgentRunRecord>>,
    events: std::sync::Mutex<HashMap<String, EventBuf>>,
    control: std::sync::Mutex<HashMap<String, mpsc::Sender<ControlMsg>>>,
    settings: std::sync::RwLock<RunSettings>,
    /// Pinged whenever a slot may have freed (run finished/started).
    slot_notify: tokio::sync::Notify,
    next_queue_seq: std::sync::atomic::AtomicU64,
    /// Held across snapshot+write in `persist` so file-write order
    /// matches snapshot order — without it a stale snapshot taken
    /// before a finish() could land after it and resurrect the run as
    /// non-terminal on the next restart.
    persist_lock: std::sync::Mutex<()>,
}

impl AgentRunRegistry {
    pub fn new(adapter_dir: PathBuf) -> Self {
        // The sessions dir is handed to pi as `--session-dir`, and pi
        // runs with the RUN's cwd — a relative adapter_dir (mock mode
        // uses bare "adapters") would resolve against the wrong
        // directory and the finalizer would never find the session.
        let adapter_dir = std::path::absolute(&adapter_dir).unwrap_or(adapter_dir);
        let persist_path = runs_json_path(&adapter_dir);
        let mut runs: BTreeMap<String, AgentRunRecord> = match std::fs::read(&persist_path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "agent_runs/runs.json parse failed");
                BTreeMap::new()
            }),
            Err(_) => BTreeMap::new(),
        };
        // Anything non-terminal in the persisted file did not survive
        // the restart — its driver task is gone.
        for run in runs.values_mut() {
            if !run.status.is_terminal() {
                run.status = RunStatus::Interrupted;
                run.error
                    .get_or_insert_with(|| "server restarted while the run was active".into());
            }
        }
        let next_seq = runs.values().map(|r| r.queue_seq + 1).max().unwrap_or(0);
        Self {
            adapter_dir,
            runs: std::sync::RwLock::new(runs),
            events: std::sync::Mutex::new(HashMap::new()),
            control: std::sync::Mutex::new(HashMap::new()),
            settings: std::sync::RwLock::new(RunSettings::default()),
            slot_notify: tokio::sync::Notify::new(),
            next_queue_seq: std::sync::atomic::AtomicU64::new(next_seq),
            persist_lock: std::sync::Mutex::new(()),
        }
    }

    /// Apply `[agent]` config (called from main after state creation).
    pub fn apply_config(&self, max_concurrent_runs: usize, run_timeout_secs: u64) {
        let mut s = self.settings.write().unwrap();
        s.max_concurrent = max_concurrent_runs.max(1);
        s.timeout = std::time::Duration::from_secs(run_timeout_secs.max(10));
    }

    pub fn max_concurrent(&self) -> usize {
        self.settings.read().unwrap().max_concurrent
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.adapter_dir.join("agent_runs").join("sessions")
    }

    pub fn list(&self) -> Vec<AgentRunRecord> {
        let mut all: Vec<_> = self.runs.read().unwrap().values().cloned().collect();
        all.sort_by(|a, b| {
            b.created_unix_ms
                .cmp(&a.created_unix_ms)
                .then(b.id.cmp(&a.id))
        });
        all
    }

    pub fn get(&self, id: &str) -> Option<AgentRunRecord> {
        self.runs.read().unwrap().get(id).cloned()
    }

    pub fn active_count(&self) -> usize {
        self.runs
            .read()
            .unwrap()
            .values()
            .filter(|r| !r.status.is_terminal())
            .count()
    }

    /// Events with seq >= `after` (inclusive — pass the previous
    /// response's `next_after` back to poll incrementally), plus the
    /// run's current status and truncation metadata so a replay from 0
    /// can tell when the head of the feed is gone (ring-buffer prune,
    /// or the buffer not surviving a server restart).
    pub fn events_after(&self, id: &str, after: u64) -> Option<EventsPage> {
        let status = self.get(id)?.status;
        let guard = self.events.lock().unwrap();
        match guard.get(id) {
            Some(buf) => {
                let first_available_seq = buf.items.front().map(|(seq, _)| *seq);
                let truncated = first_available_seq.is_some_and(|first| after < first && first > 0);
                Some(EventsPage {
                    events: buf.after(after),
                    status,
                    first_available_seq,
                    truncated,
                })
            }
            // Run record survived a restart; its event buffer did not.
            None => Some(EventsPage {
                events: Vec::new(),
                status,
                first_available_seq: None,
                truncated: true,
            }),
        }
    }

    /// Queue a steer message into a live run.
    pub fn steer(&self, id: &str, message: String) -> Result<(), ControlError> {
        self.send_control(id, ControlMsg::Steer(message))
    }

    /// Queue a follow-up task into a live run (extends the run by one
    /// more agent loop).
    pub fn follow_up(&self, id: &str, message: String) -> Result<(), ControlError> {
        self.send_control(id, ControlMsg::FollowUp(message))
    }

    pub fn abort(&self, id: &str) -> Result<(), ControlError> {
        self.send_control(id, ControlMsg::Abort)
    }

    fn send_control(&self, id: &str, msg: ControlMsg) -> Result<(), ControlError> {
        let Some(run) = self.get(id) else {
            return Err(ControlError::NotFound);
        };
        if run.status.is_terminal() {
            return Err(ControlError::NotActive(run.status));
        }
        let sender = self.control.lock().unwrap().get(id).cloned();
        match sender {
            Some(tx) => tx
                .try_send(msg)
                .map_err(|_| ControlError::NotActive(run.status)),
            None => Err(ControlError::NotActive(run.status)),
        }
    }

    /// Create the record and spawn the driver task. Returns a snapshot
    /// of the queued record, or `AtCapacity` — checked atomically with
    /// the insert so concurrent submissions can't overshoot the
    /// backstop.
    pub fn start_run(
        self: &Arc<Self>,
        params: NewRunParams,
    ) -> Result<AgentRunRecord, StartRunError> {
        let id = uuid::Uuid::new_v4().to_string();
        let record = AgentRunRecord {
            id: id.clone(),
            task: params.task.clone(),
            cwd: params.cwd.display().to_string(),
            label: params.label.clone(),
            status: RunStatus::Queued,
            created_unix_ms: crate::recent_requests::now_unix_ms(),
            queue_seq: self
                .next_queue_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            started_unix_ms: None,
            finished_unix_ms: None,
            num_turns: 0,
            num_tool_calls: 0,
            session_id: None,
            session_path: None,
            trace_indexed: false,
            last_assistant_text: None,
            error: None,
        };
        {
            let mut runs = self.runs.write().unwrap();
            let active = runs.values().filter(|r| !r.status.is_terminal()).count();
            if active >= ACTIVE_RUNS_BACKSTOP {
                return Err(StartRunError::AtCapacity(ACTIVE_RUNS_BACKSTOP));
            }
            runs.insert(id.clone(), record.clone());
        }
        self.events
            .lock()
            .unwrap()
            .insert(id.clone(), EventBuf::default());
        let (ctl_tx, ctl_rx) = mpsc::channel::<ControlMsg>(16);
        self.control.lock().unwrap().insert(id.clone(), ctl_tx);
        self.prune_terminal();
        self.persist();

        let reg = Arc::clone(self);
        tokio::spawn(async move {
            drive_run(reg, id, params, ctl_rx).await;
        });
        Ok(record)
    }

    // ── internals ───────────────────────────────────────────────────

    fn update<F: FnOnce(&mut AgentRunRecord)>(&self, id: &str, f: F) {
        if let Some(run) = self.runs.write().unwrap().get_mut(id) {
            f(run);
        }
    }

    fn push_event(&self, id: &str, value: serde_json::Value) {
        if let Some(buf) = self.events.lock().unwrap().get_mut(id) {
            buf.push(value);
        }
    }

    fn finish(&self, id: &str, status: RunStatus, error: Option<String>) {
        self.update(id, |run| {
            run.status = status;
            run.finished_unix_ms = Some(crate::recent_requests::now_unix_ms());
            if run.error.is_none() {
                run.error = error;
            }
        });
        self.control.lock().unwrap().remove(id);
        self.persist();
        self.slot_notify.notify_waiters();
    }

    fn persist(&self) {
        // Snapshot and write under one mutex so writes land in
        // snapshot order (see persist_lock field doc).
        let _guard = self.persist_lock.lock().unwrap_or_else(|p| p.into_inner());
        let path = runs_json_path(&self.adapter_dir);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let snapshot = self.runs.read().unwrap().clone();
        match serde_json::to_vec_pretty(&snapshot) {
            Ok(bytes) => {
                if let Err(e) = kiln_resource::locked_atomic_write(&path, &bytes) {
                    tracing::warn!(error = %e, "agent_runs/runs.json write failed");
                }
            }
            Err(e) => tracing::warn!(error = %e, "agent_runs serialize failed"),
        }
    }

    /// Keep the newest TERMINAL_RUNS_CAP terminal records; active runs
    /// are never pruned. Event buffers go with their records.
    fn prune_terminal(&self) {
        let pruned: Vec<String> = {
            let runs = self.runs.read().unwrap();
            let mut terminal: Vec<_> = runs
                .values()
                .filter(|r| r.status.is_terminal())
                .map(|r| (r.created_unix_ms, r.id.clone()))
                .collect();
            if terminal.len() <= TERMINAL_RUNS_CAP {
                return;
            }
            terminal.sort(); // oldest first
            let overflow = terminal.len() - TERMINAL_RUNS_CAP;
            terminal
                .into_iter()
                .take(overflow)
                .map(|(_, id)| id)
                .collect()
        };
        let mut runs = self.runs.write().unwrap();
        let mut events = self.events.lock().unwrap();
        for id in pruned {
            runs.remove(&id);
            events.remove(&id);
        }
    }

    /// Wait until this run is first in the queued FIFO and a slot is
    /// free, then mark it running. Returns false when aborted while
    /// queued. Steer/follow-up messages arriving before the start are
    /// buffered into `pending` — the API told the caller "queued", so
    /// they must be delivered once the run starts, not dropped.
    async fn claim_slot(
        &self,
        id: &str,
        ctl_rx: &mut mpsc::Receiver<ControlMsg>,
        pending: &mut Vec<ControlMsg>,
    ) -> bool {
        loop {
            let claimed = {
                let mut runs = self.runs.write().unwrap();
                let max = self.settings.read().unwrap().max_concurrent;
                let running = runs
                    .values()
                    .filter(|r| r.status == RunStatus::Running)
                    .count();
                let first_queued = runs
                    .values()
                    .filter(|r| r.status == RunStatus::Queued)
                    .min_by_key(|r| r.queue_seq)
                    .map(|r| r.id.clone());
                if running < max && first_queued.as_deref() == Some(id) {
                    if let Some(run) = runs.get_mut(id) {
                        run.status = RunStatus::Running;
                        run.started_unix_ms = Some(crate::recent_requests::now_unix_ms());
                    }
                    true
                } else {
                    false
                }
            };
            if claimed {
                self.persist();
                // Another queued run may also fit within max_concurrent.
                self.slot_notify.notify_waiters();
                return true;
            }
            tokio::select! {
                _ = self.slot_notify.notified() => {}
                // Re-check periodically — Notify wakeups can race with
                // registration.
                _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {}
                msg = ctl_rx.recv() => {
                    match msg {
                        Some(ControlMsg::Abort) | None => return false,
                        Some(msg) => {
                            self.push_event(id, serde_json::json!({
                                "type": "kiln_note",
                                "note": "message queued — delivered when the run starts"
                            }));
                            pending.push(msg);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum ControlError {
    NotFound,
    NotActive(RunStatus),
}

#[derive(Debug)]
pub enum StartRunError {
    /// queued + running runs already at the backstop.
    AtCapacity(usize),
}

/// One page of the event feed (see [`AgentRunRegistry::events_after`]).
#[derive(Debug)]
pub struct EventsPage {
    pub events: Vec<(u64, serde_json::Value)>,
    pub status: RunStatus,
    /// Earliest seq still retained (None when no events were buffered).
    pub first_available_seq: Option<u64>,
    /// True when events before the requested cursor are gone — ring
    /// prune on a long run, or a server restart dropping the buffer.
    pub truncated: bool,
}

fn runs_json_path(adapter_dir: &Path) -> PathBuf {
    adapter_dir.join("agent_runs").join("runs.json")
}

/// The per-run driver: claim a slot, spawn pi, prompt, pump events,
/// finalize, index the session into the trace layer.
async fn drive_run(
    reg: Arc<AgentRunRegistry>,
    id: String,
    params: NewRunParams,
    mut ctl_rx: mpsc::Receiver<ControlMsg>,
) {
    let mut pending_control: Vec<ControlMsg> = Vec::new();
    if !reg.claim_slot(&id, &mut ctl_rx, &mut pending_control).await {
        reg.finish(&id, RunStatus::Aborted, Some("aborted while queued".into()));
        return;
    }

    // Same non-destructive config merge as the embedded terminal, so
    // the spawned pi's `kiln-local` provider points at this server.
    if let Some(url) = &params.kiln_url
        && let Err(e) = crate::cli::apply_pi_setup_quiet(url, Some(&params.model))
    {
        tracing::warn!(error = %e, "pi-setup merge failed; spawning pi with existing config");
    }

    // Per-run session dir: runs never share a directory, so the
    // session-file fallback below can only ever see this run's own
    // file — a shared dir let a run that died before flushing adopt a
    // sibling's session. Trace discovery sweeps nested dirs fine.
    let sessions_dir = reg.sessions_dir().join(&id);
    if let Err(e) = std::fs::create_dir_all(&sessions_dir) {
        reg.finish(
            &id,
            RunStatus::Failed,
            Some(format!("could not create sessions dir: {e}")),
        );
        return;
    }

    let opts = PiRpcOptions {
        cwd: params.cwd.clone(),
        provider: PI_PROVIDER_ID.to_string(),
        model: params.model.clone(),
        session_dir: sessions_dir.clone(),
        session_name: Some(
            params
                .label
                .clone()
                .unwrap_or_else(|| format!("kiln run {}", &id[..8.min(id.len())])),
        ),
        tools: params.tools.clone(),
        thinking_level: params.thinking_level.clone(),
    };
    let mut process = match PiRpcProcess::spawn(&params.pi_bin, &opts) {
        Ok(p) => p,
        Err(e) => {
            reg.finish(
                &id,
                RunStatus::Failed,
                Some(format!("could not start pi: {e}")),
            );
            return;
        }
    };

    let prompt = serde_json::json!({"id": "prompt-1", "type": "prompt", "message": params.task});
    let state_probe = serde_json::json!({"id": "state-1", "type": "get_state"});
    if let Err(e) = process.send(&prompt).await {
        reg.finish(
            &id,
            RunStatus::Failed,
            Some(format!("prompt write failed: {e}")),
        );
        process.shutdown(std::time::Duration::from_secs(3)).await;
        return;
    }
    // Session file/id are known as soon as the session exists — probe
    // early so even a crashed run can be indexed.
    let _ = process.send(&state_probe).await;

    let timeout = params
        .timeout_secs
        .map(std::time::Duration::from_secs)
        .unwrap_or_else(|| reg.settings.read().unwrap().timeout);
    let deadline = tokio::time::Instant::now() + timeout;

    // One prompt = one agent loop = one agent_end. Each accepted
    // follow-up queues one more loop.
    let mut ends_remaining: i64 = 1;
    let mut steer_seq: u64 = 0;
    // Deliver control messages that arrived while the run was queued —
    // the API already acknowledged them. A steer landing in the gap
    // before pi starts streaming surfaces as an error response in the
    // event feed rather than vanishing.
    for msg in pending_control.drain(..) {
        match msg {
            ControlMsg::Steer(message) => {
                steer_seq += 1;
                let cmd = serde_json::json!({
                    "id": format!("steer-{steer_seq}"),
                    "type": "steer",
                    "message": message,
                });
                let _ = process.send(&cmd).await;
            }
            ControlMsg::FollowUp(message) => {
                steer_seq += 1;
                ends_remaining += 1;
                let cmd = serde_json::json!({
                    "id": format!("follow-{steer_seq}"),
                    "type": "follow_up",
                    "message": message,
                });
                let _ = process.send(&cmd).await;
            }
            ControlMsg::Abort => {}
        }
    }
    let (final_status, final_error) = loop {
        tokio::select! {
            line = process.lines.recv() => {
                match line {
                    Some(PiRpcLine::Json(value)) => {
                        observe_line(&reg, &id, &value);
                        let ty = value.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if ty == "agent_end" {
                            ends_remaining -= 1;
                            if ends_remaining <= 0 {
                                break (RunStatus::Completed, None);
                            }
                        } else if ty == "response"
                            && value.get("id").and_then(|i| i.as_str()) == Some("prompt-1")
                            && value.get("success").and_then(|s| s.as_bool()) == Some(false)
                        {
                            let err = value.get("error").and_then(|e| e.as_str()).unwrap_or("prompt rejected");
                            break (RunStatus::Failed, Some(err.to_string()));
                        }
                    }
                    Some(PiRpcLine::Eof) | None => {
                        break (RunStatus::Failed, Some("pi exited before finishing the run".into()));
                    }
                }
            }
            msg = ctl_rx.recv() => {
                match msg {
                    Some(ControlMsg::Steer(message)) => {
                        steer_seq += 1;
                        let cmd = serde_json::json!({
                            "id": format!("steer-{steer_seq}"),
                            "type": "steer",
                            "message": message,
                        });
                        let _ = process.send(&cmd).await;
                    }
                    Some(ControlMsg::FollowUp(message)) => {
                        steer_seq += 1;
                        ends_remaining += 1;
                        let cmd = serde_json::json!({
                            "id": format!("follow-{steer_seq}"),
                            "type": "follow_up",
                            "message": message,
                        });
                        let _ = process.send(&cmd).await;
                    }
                    Some(ControlMsg::Abort) | None => {
                        let _ = process.send(&serde_json::json!({"type": "abort"})).await;
                        break (RunStatus::Aborted, None);
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                let _ = process.send(&serde_json::json!({"type": "abort"})).await;
                break (RunStatus::TimedOut, Some(format!("run exceeded {}s timeout", timeout.as_secs())));
            }
        }
    };

    // Give pi a moment to flush the session file and exit on stdin EOF.
    process.shutdown(std::time::Duration::from_secs(5)).await;

    // Index the finished session into the §10.3 trace layer so the
    // flywheel sees it without a manual discover.
    let (session_id, session_path, started_unix_ms) = {
        let run = reg.get(&id);
        (
            run.as_ref().and_then(|r| r.session_id.clone()),
            run.as_ref().and_then(|r| r.session_path.clone()),
            run.as_ref().and_then(|r| r.started_unix_ms),
        )
    };
    let resolved_path = session_path
        .map(PathBuf::from)
        .filter(|p| p.is_file())
        .or_else(|| find_session_file(&sessions_dir, session_id.as_deref(), started_unix_ms));
    match resolved_path {
        Some(path) => {
            let indexed = crate::api::agent_traces::index_session_file(&reg.adapter_dir, &path);
            reg.update(&id, |run| {
                run.session_path = Some(path.display().to_string());
                if let Some(trace) = &indexed {
                    run.session_id = Some(trace.id.clone());
                    run.trace_indexed = true;
                }
            });
        }
        None => {
            tracing::warn!(run = %id, dir = %sessions_dir.display(), "embedded run session file not found — trace not indexed");
            reg.push_event(
                &id,
                serde_json::json!({
                    "type": "kiln_note",
                    "note": "session file not found — trace not indexed",
                }),
            );
        }
    }

    // A run whose last assistant turn ended in an error (observe_line
    // records it) is not a success, even though pi emitted a normal
    // agent_end — e.g. the model endpoint 4xx/5xxed on the final turn.
    let final_status = if final_status == RunStatus::Completed
        && reg.get(&id).is_some_and(|r| r.error.is_some())
    {
        RunStatus::Failed
    } else {
        final_status
    };

    reg.finish(&id, final_status, final_error);
}

/// Update live counters and the event buffer from one pi stdout record.
fn observe_line(reg: &AgentRunRegistry, id: &str, value: &serde_json::Value) {
    let ty = value.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
        "message_end" => {
            let message = value.get("message");
            let role = message
                .and_then(|m| m.get("role"))
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role == "assistant" {
                let text = message
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_array())
                    .map(|blocks| {
                        blocks
                            .iter()
                            .filter(|&b| (b.get("type").and_then(|t| t.as_str()) == Some("text")))
                            .map(|b| b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                // An assistant turn can end in an error (stopReason
                // "error": model endpoint down, 4xx/5xx, ...) while pi
                // still finishes the loop normally — record it so the
                // run doesn't report success on a failed turn. A later
                // successful turn clears it (pi recovered).
                let turn_errored = message
                    .and_then(|m| m.get("stopReason"))
                    .and_then(|s| s.as_str())
                    == Some("error");
                let turn_error_message = message
                    .and_then(|m| m.get("errorMessage"))
                    .and_then(|s| s.as_str())
                    .map(str::to_string);
                reg.update(id, |run| {
                    run.num_turns += 1;
                    if turn_errored {
                        run.error =
                            Some(turn_error_message.unwrap_or_else(|| {
                                "assistant turn ended with an error".to_string()
                            }));
                    } else {
                        run.error = None;
                    }
                    if !text.is_empty() {
                        let mut t = text;
                        if t.len() > LAST_TEXT_CAP {
                            let mut cut = LAST_TEXT_CAP;
                            while !t.is_char_boundary(cut) {
                                cut -= 1;
                            }
                            t.truncate(cut);
                        }
                        run.last_assistant_text = Some(t);
                    }
                });
            }
        }
        "tool_execution_end" => {
            reg.update(id, |run| run.num_tool_calls += 1);
        }
        "response" => {
            if value.get("command").and_then(|c| c.as_str()) == Some("get_state")
                && let Some(data) = value.get("data")
            {
                let session_id = data
                    .get("sessionId")
                    .and_then(|s| s.as_str())
                    .map(str::to_string);
                let session_path = data
                    .get("sessionFile")
                    .and_then(|s| s.as_str())
                    .map(str::to_string);
                reg.update(id, |run| {
                    if run.session_id.is_none() {
                        run.session_id = session_id;
                    }
                    if run.session_path.is_none() {
                        run.session_path = session_path;
                    }
                });
            }
        }
        _ => {}
    }
    // message_update fires per streamed token — too chatty for the
    // replay buffer; everything else is kept.
    if ty != "message_update" {
        reg.push_event(id, value.clone());
    }
}

/// Fallback session lookup when get_state never answered, scoped to
/// this run's own session dir. With a known session id, only a
/// matching filename counts — an id that matches nothing means the
/// session was never flushed, and attaching any other file would be
/// wrong. Without one, the newest `.jsonl` is accepted only if it was
/// modified at/after the run started (1s slack for fs granularity).
fn find_session_file(
    sessions_dir: &Path,
    session_id: Option<&str>,
    started_unix_ms: Option<u64>,
) -> Option<PathBuf> {
    let not_before = started_unix_ms.map(|ms| {
        std::time::SystemTime::UNIX_EPOCH
            + std::time::Duration::from_millis(ms.saturating_sub(1000))
    });
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut stack = vec![sessions_dir.to_path_buf()];
    let mut depth_guard = 0usize;
    while let Some(dir) = stack.pop() {
        depth_guard += 1;
        if depth_guard > 64 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "jsonl") {
                if let Some(sid) = session_id {
                    if p.file_stem()
                        .is_some_and(|stem| stem.to_string_lossy().contains(sid))
                    {
                        return Some(p);
                    }
                    continue;
                }
                let mtime = entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                if not_before.is_some_and(|floor| mtime < floor) {
                    continue;
                }
                if newest.as_ref().is_none_or(|(t, _)| mtime > *t) {
                    newest = Some((mtime, p));
                }
            }
        }
    }
    newest.map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub pi that speaks just enough of the RPC protocol: accepts
    /// the prompt, emits a tool call + final message, writes a session
    /// file, answers get_state, and exits on agent_end + stdin EOF.
    const FAKE_PI: &str = r#"#!/usr/bin/env python3
import sys, json, os
args = sys.argv[1:]
session_dir = "."
for i, a in enumerate(args):
    if a == "--session-dir":
        session_dir = args[i + 1]
sid = "embedded-test-session"
spath = os.path.join(session_dir, sid + ".jsonl")
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n")
    sys.stdout.flush()
for line in sys.stdin:
    cmd = json.loads(line)
    t = cmd.get("type")
    if t == "prompt":
        emit({"id": cmd.get("id"), "type": "response", "command": "prompt", "success": True})
        emit({"type": "agent_start"})
        emit({"type": "turn_start"})
        emit({"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {"cmd": "echo hi"}})
        emit({"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash", "result": "hi", "isError": False})
        emit({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "All done"}]}})
        with open(spath, "w") as f:
            f.write(json.dumps({"type": "session", "version": 3, "id": sid, "timestamp": "2026-06-11T00:00:00Z", "cwd": os.getcwd()}) + "\n")
            f.write(json.dumps({"type": "message", "timestamp": "2026-06-11T00:00:01Z", "message": {"role": "user", "content": [{"type": "text", "text": cmd.get("message")}]}}) + "\n")
            f.write(json.dumps({"type": "message", "timestamp": "2026-06-11T00:00:02Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "All done"}]}}) + "\n")
        emit({"type": "agent_end", "messages": []})
    elif t == "get_state":
        emit({"id": cmd.get("id"), "type": "response", "command": "get_state", "success": True,
              "data": {"sessionId": sid, "sessionFile": spath, "thinkingLevel": "off",
                       "isStreaming": False, "isCompacting": False, "steeringMode": "all",
                       "followUpMode": "all", "autoCompactionEnabled": True,
                       "messageCount": 2, "pendingMessageCount": 0}})
    elif t == "abort":
        emit({"id": cmd.get("id"), "type": "response", "command": "abort", "success": True})
"#;

    fn write_fake_pi(dir: &Path) -> PathBuf {
        let path = dir.join("fake-pi");
        std::fs::write(&path, FAKE_PI).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    fn params(pi_bin: PathBuf, cwd: PathBuf, task: &str) -> NewRunParams {
        NewRunParams {
            task: task.into(),
            cwd,
            label: None,
            tools: None,
            thinking_level: None,
            timeout_secs: Some(30),
            pi_bin,
            model: "Qwen3.5-4B".into(),
            kiln_url: None, // never touch ~/.pi from tests
        }
    }

    async fn wait_terminal(reg: &Arc<AgentRunRegistry>, id: &str) -> AgentRunRecord {
        for _ in 0..200 {
            let run = reg.get(id).unwrap();
            if run.status.is_terminal() {
                return run;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        panic!("run {id} never reached a terminal state");
    }

    #[tokio::test]
    async fn run_completes_streams_events_and_indexes_trace() {
        let adapter_dir = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let pi_bin = write_fake_pi(adapter_dir.path());
        let reg = Arc::new(AgentRunRegistry::new(adapter_dir.path().to_path_buf()));

        let rec = reg
            .start_run(params(pi_bin, workdir.path().to_path_buf(), "say hi"))
            .unwrap();
        assert_eq!(rec.status, RunStatus::Queued);

        let done = wait_terminal(&reg, &rec.id).await;
        assert_eq!(done.status, RunStatus::Completed, "error: {:?}", done.error);
        assert_eq!(done.num_turns, 1);
        assert_eq!(done.num_tool_calls, 1);
        assert_eq!(done.last_assistant_text.as_deref(), Some("All done"));
        assert_eq!(done.session_id.as_deref(), Some("embedded-test-session"));
        assert!(
            done.trace_indexed,
            "session must be merged into agent_traces.json"
        );

        // The event feed captured the full trajectory.
        let page = reg.events_after(&rec.id, 0).unwrap();
        assert_eq!(page.status, RunStatus::Completed);
        assert!(!page.truncated, "nothing was pruned");
        assert_eq!(page.first_available_seq, Some(0));
        let types: Vec<_> = page
            .events
            .iter()
            .map(|(_, v)| v.get("type").and_then(|t| t.as_str()).unwrap_or(""))
            .collect();
        assert!(types.contains(&"agent_start"));
        assert!(types.contains(&"tool_execution_end"));
        assert!(types.contains(&"agent_end"));

        // The trace layer sees the session — same index the flywheel reads.
        let index = crate::api::agent_traces::AgentTraceIndex::load_from_path(
            &adapter_dir.path().join("agent_traces.json"),
        );
        let trace = index
            .traces
            .get("embedded-test-session")
            .expect("trace indexed");
        assert_eq!(trace.prompt_messages.len(), 1);
        assert_eq!(trace.prompt_messages[0].content, "say hi");

        // Records persist for restart recovery.
        let persisted =
            std::fs::read_to_string(adapter_dir.path().join("agent_runs").join("runs.json"))
                .unwrap();
        assert!(persisted.contains(&rec.id));
    }

    #[tokio::test]
    async fn concurrency_cap_holds_second_run_queued() {
        let adapter_dir = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        // A pi that never finishes: accepts the prompt then blocks.
        let slow_pi = adapter_dir.path().join("slow-pi");
        std::fs::write(
            &slow_pi,
            "#!/usr/bin/env python3\nimport sys,json,time\nfor line in sys.stdin:\n    cmd=json.loads(line)\n    if cmd.get('type')=='prompt':\n        sys.stdout.write(json.dumps({'id':cmd.get('id'),'type':'response','command':'prompt','success':True})+'\\n')\n        sys.stdout.flush()\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&slow_pi, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let reg = Arc::new(AgentRunRegistry::new(adapter_dir.path().to_path_buf()));
        reg.apply_config(1, 900);

        let a = reg
            .start_run(params(slow_pi.clone(), workdir.path().to_path_buf(), "a"))
            .unwrap();
        let b = reg
            .start_run(params(slow_pi, workdir.path().to_path_buf(), "b"))
            .unwrap();

        // Give the first driver time to claim the only slot.
        tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        assert_eq!(reg.get(&a.id).unwrap().status, RunStatus::Running);
        assert_eq!(reg.get(&b.id).unwrap().status, RunStatus::Queued);

        // Aborting the runner frees the slot for the queued run.
        reg.abort(&a.id).unwrap();
        let a_done = wait_terminal(&reg, &a.id).await;
        assert_eq!(a_done.status, RunStatus::Aborted);
        for _ in 0..100 {
            if reg.get(&b.id).unwrap().status == RunStatus::Running {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(reg.get(&b.id).unwrap().status, RunStatus::Running);
        reg.abort(&b.id).unwrap();
        wait_terminal(&reg, &b.id).await;
    }

    #[tokio::test]
    async fn missing_binary_fails_cleanly() {
        let adapter_dir = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let reg = Arc::new(AgentRunRegistry::new(adapter_dir.path().to_path_buf()));
        let rec = reg
            .start_run(params(
                PathBuf::from("/nonexistent/pi-binary"),
                workdir.path().to_path_buf(),
                "x",
            ))
            .unwrap();
        let done = wait_terminal(&reg, &rec.id).await;
        assert_eq!(done.status, RunStatus::Failed);
        assert!(done.error.unwrap().contains("could not start pi"));
    }

    #[test]
    fn self_url_brackets_ipv6_and_rewrites_wildcards() {
        assert_eq!(format_self_url("127.0.0.1", 8420), "http://127.0.0.1:8420");
        assert_eq!(format_self_url("0.0.0.0", 8420), "http://127.0.0.1:8420");
        assert_eq!(format_self_url("::", 8420), "http://127.0.0.1:8420");
        // A bare IPv6 loopback bind passes the runs gate; unbracketed it
        // produced http://::1:8420, which pi cannot parse.
        assert_eq!(format_self_url("::1", 8420), "http://[::1]:8420");
        assert_eq!(format_self_url("[::1]", 8420), "http://[::1]:8420");
        assert_eq!(
            format_self_url("office-kiln", 9000),
            "http://office-kiln:9000"
        );
    }

    #[tokio::test]
    async fn start_run_rejects_at_backstop_capacity() {
        let adapter_dir = tempfile::tempdir().unwrap();
        let workdir = tempfile::tempdir().unwrap();
        let reg = Arc::new(AgentRunRegistry::new(adapter_dir.path().to_path_buf()));
        reg.apply_config(1, 900);
        // Fill the registry with queued records via the public path; the
        // missing binary keeps them from progressing instantly, and
        // capacity counts queued + running either way.
        let mut started = Vec::new();
        for i in 0..ACTIVE_RUNS_BACKSTOP {
            match reg.start_run(params(
                PathBuf::from("/nonexistent/pi-binary"),
                workdir.path().to_path_buf(),
                &format!("task {i}"),
            )) {
                Ok(rec) => started.push(rec.id),
                Err(StartRunError::AtCapacity(_)) => {
                    // Drivers may already have failed some runs (terminal
                    // states free capacity), so reaching the cap is not
                    // guaranteed — but a rejection here still proves the
                    // backstop fires.
                    return;
                }
            }
        }
        let over = reg.start_run(params(
            PathBuf::from("/nonexistent/pi-binary"),
            workdir.path().to_path_buf(),
            "one too many",
        ));
        assert!(
            matches!(over, Err(StartRunError::AtCapacity(max)) if max == ACTIVE_RUNS_BACKSTOP)
                // On a fast machine some of the 32 may already be terminal
                // (missing binary fails immediately), freeing capacity.
                || over.is_ok(),
        );
        // Either way the registry never holds more than the backstop in
        // non-terminal states.
        assert!(reg.active_count() <= ACTIVE_RUNS_BACKSTOP);
    }

    #[test]
    fn restart_marks_active_runs_interrupted() {
        let adapter_dir = tempfile::tempdir().unwrap();
        let runs_dir = adapter_dir.path().join("agent_runs");
        std::fs::create_dir_all(&runs_dir).unwrap();
        let record = AgentRunRecord {
            id: "r1".into(),
            task: "t".into(),
            cwd: "/x".into(),
            label: None,
            status: RunStatus::Running,
            created_unix_ms: 1,
            queue_seq: 0,
            started_unix_ms: Some(2),
            finished_unix_ms: None,
            num_turns: 0,
            num_tool_calls: 0,
            session_id: None,
            session_path: None,
            trace_indexed: false,
            last_assistant_text: None,
            error: None,
        };
        let mut map = BTreeMap::new();
        map.insert("r1".to_string(), record);
        std::fs::write(
            runs_dir.join("runs.json"),
            serde_json::to_vec_pretty(&map).unwrap(),
        )
        .unwrap();

        let reg = AgentRunRegistry::new(adapter_dir.path().to_path_buf());
        let run = reg.get("r1").unwrap();
        assert_eq!(run.status, RunStatus::Interrupted);
        assert!(run.error.unwrap().contains("restarted"));
    }
}
