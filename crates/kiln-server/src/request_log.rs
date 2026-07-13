//! Durable request/response log for the inference endpoints.
//!
//! Every request to `/v1/chat/completions`, `/v1/completions`, and
//! `/v1/completions/batch` — and the exact response the client received,
//! including reassembled SSE streams and error responses — is appended as one
//! JSON line. The point is the flywheel: production traffic becomes a corpus
//! you can mine, filter, and train on.
//!
//! Each row carries the wire-format `request` and `response` JSON, so the log
//! feeds existing tooling without a custom exporter:
//!
//! ```bash
//! # SFT dataset from successful chats (request messages + assistant reply):
//! zcat requests-*.jsonl.gz | jq -c 'select(.status == 200 and .route == "/v1/chat/completions")
//!   | {messages: (.request.messages + [.response.choices[0].message])}' > sft.jsonl
//!
//! # Tool-call eval suite via the production-trace importer:
//! zcat requests-*.jsonl.gz | jq -c 'select(.status == 200)
//!   | {messages: (.request.messages + [.response.choices[0].message]), tools: .request.tools}' \
//!   | kiln-eval trace-suite --input /dev/stdin --format openai_jsonl ...
//! ```
//!
//! Files live under the log directory as `requests-current.jsonl` (active)
//! and `requests-<unix_ms>.jsonl.gz` (rotated + compressed). Rotation is
//! size-based; total disk use is retention-capped by deleting the oldest
//! rotated files. Writing happens on a dedicated thread behind a bounded
//! channel — the request path never blocks on disk, and overload drops log
//! rows (counted) rather than stalling traffic.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::task::Poll;
use std::time::Instant;

use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::state::AppState;

/// `[request_log]` section of kiln.toml. Canonical `KILN_REQUEST_LOG_<FIELD>`
/// startup overrides are resolved centrally by `KilnConfig`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RequestLogConfig {
    /// Master switch. Default true — the log is the raw material for the
    /// mine→filter→train flywheel, and rotation + retention keep it bounded.
    pub enabled: bool,
    /// Directory for log files. `None` → `<adapter_dir>/.requests`.
    pub dir: Option<PathBuf>,
    /// Rotate the active file once it exceeds this many bytes.
    pub max_file_bytes: u64,
    /// Delete the oldest rotated files once the directory exceeds this.
    pub max_total_bytes: u64,
    /// Gzip rotated files (`requests-<ts>.jsonl.gz`).
    pub compress: bool,
    /// Per-body cap on what gets *stored* in a row. Bodies larger than this
    /// are truncated in the log only — never on the wire.
    pub max_capture_bytes: usize,
}

impl Default for RequestLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
            max_file_bytes: 64 * 1024 * 1024,
            max_total_bytes: 2 * 1024 * 1024 * 1024,
            compress: true,
            max_capture_bytes: 4 * 1024 * 1024,
        }
    }
}

impl RequestLogConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.dir.as_ref().is_some_and(|path| {
            path.as_os_str().is_empty() || path.to_str().is_some_and(|path| path.trim().is_empty())
        }) {
            anyhow::bail!("request_log.dir must be non-empty when set");
        }
        if self.max_file_bytes < 4096 {
            anyhow::bail!(
                "request_log.max_file_bytes must be at least 4096, got {}",
                self.max_file_bytes
            );
        }
        if self.max_total_bytes == 0 {
            anyhow::bail!(
                "request_log.max_total_bytes must be > 0, got {}",
                self.max_total_bytes
            );
        }
        if self.max_capture_bytes == 0 {
            anyhow::bail!(
                "request_log.max_capture_bytes must be > 0, got {}",
                self.max_capture_bytes
            );
        }
        Ok(())
    }
}

/// One JSONL row. `request` / `response` hold the wire JSON when the body
/// parsed as JSON, otherwise a `{"_raw": "..."}` wrapper; oversized bodies
/// are stored truncated with the matching `*_truncated` flag set.
#[derive(Debug, Serialize)]
pub struct RequestLogEntry {
    /// RFC3339 completion timestamp (when the response finished).
    pub ts: String,
    pub route: String,
    pub status: u16,
    pub duration_ms: u64,
    /// True when the response went out as an SSE stream; `response` is then
    /// the reassembled final-message shape, not the raw event text.
    pub streamed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// First-party self-identification from the `X-Kiln-Client` header (the
    /// /ui dashboard sends `dashboard` on its own traffic) so the mining
    /// pipeline can exclude dashboard-originated rows. Absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    /// LoRA adapter that actually served this response (from the
    /// `x-kiln-loaded-adapter` response header); `None` = base model.
    /// Lets the mining pipeline split the corpus per adapter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    pub request: serde_json::Value,
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub request_truncated: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub response_truncated: bool,
    /// Set when an SSE stream ended before `[DONE]` (client disconnect or
    /// server abort) — the reassembled response covers what was sent.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream_interrupted: bool,
}

enum Msg {
    Entry(Box<RequestLogEntry>),
    Flush(SyncSender<()>),
}

/// Handle shared via `AppState`. Cheap to clone (Arc'd by the caller).
pub struct RequestLogger {
    tx: SyncSender<Msg>,
    dropped: AtomicU64,
}

impl std::fmt::Debug for RequestLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestLogger")
            .field("dropped", &self.dropped.load(Ordering::Relaxed))
            .finish()
    }
}

impl RequestLogger {
    /// Create the log directory, open (or continue) the active file, and
    /// spawn the writer thread.
    pub fn spawn(dir: PathBuf, config: RequestLogConfig) -> std::io::Result<Arc<Self>> {
        fs::create_dir_all(&dir)?;
        let writer = LogWriter::open(dir, config)?;
        // Bounded so a disk stall can never back up into request handling;
        // 1024 in-flight rows is minutes of headroom at realistic rates.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Msg>(1024);
        std::thread::Builder::new()
            .name("kiln-request-log".into())
            .spawn(move || writer.run(rx))?;
        Ok(Arc::new(Self {
            tx,
            dropped: AtomicU64::new(0),
        }))
    }

    /// Queue one row. Never blocks: on overload the row is dropped and
    /// counted (a full channel means the disk can't keep up — stalling
    /// inference to wait for it would be the wrong trade).
    pub fn log(&self, entry: RequestLogEntry) {
        match self.tx.try_send(Msg::Entry(Box::new(entry))) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                let dropped = self.dropped.fetch_add(1, Ordering::Relaxed) + 1;
                if dropped == 1 || dropped % 1000 == 0 {
                    tracing::warn!(dropped, "request log overloaded; dropping rows");
                }
            }
        }
    }

    /// Total rows dropped due to writer overload.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Block until the writer has persisted everything queued so far.
    /// Used by tests and graceful shutdown.
    pub fn flush(&self) {
        let (ack_tx, ack_rx) = std::sync::mpsc::sync_channel(1);
        if self.tx.send(Msg::Flush(ack_tx)).is_ok() {
            let _ = ack_rx.recv();
        }
    }
}

const ACTIVE_FILE: &str = "requests-current.jsonl";

struct LogWriter {
    dir: PathBuf,
    config: RequestLogConfig,
    file: BufWriter<fs::File>,
    current_bytes: u64,
}

impl LogWriter {
    fn open(dir: PathBuf, config: RequestLogConfig) -> std::io::Result<Self> {
        let path = dir.join(ACTIVE_FILE);
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let current_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            dir,
            config,
            file: BufWriter::new(file),
            current_bytes,
        })
    }

    fn run(mut self, rx: Receiver<Msg>) {
        while let Ok(msg) = rx.recv() {
            match msg {
                Msg::Entry(entry) => {
                    if let Err(e) = self.append(&entry) {
                        tracing::warn!(error = %e, "request log write failed");
                    }
                }
                Msg::Flush(ack) => {
                    let _ = self.file.flush();
                    let _ = ack.try_send(());
                }
            }
        }
        let _ = self.file.flush();
    }

    fn append(&mut self, entry: &RequestLogEntry) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(entry)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        // Flush per row: rows must survive an abrupt server exit, and this
        // thread is off the request path — durability wins over batching.
        self.file.flush()?;
        self.current_bytes += line.len() as u64;
        if self.current_bytes >= self.config.max_file_bytes {
            self.rotate()?;
        }
        Ok(())
    }

    fn rotate(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        let active = self.dir.join(ACTIVE_FILE);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let rotated = self.dir.join(format!("requests-{stamp:020}.jsonl"));
        fs::rename(&active, &rotated)?;
        let fresh = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&active)?;
        self.file = BufWriter::new(fresh);
        self.current_bytes = 0;
        if self.config.compress {
            if let Err(e) = gzip_file(&rotated) {
                tracing::warn!(error = %e, file = %rotated.display(), "request log compression failed; keeping plain file");
            }
        }
        self.enforce_retention();
        Ok(())
    }

    /// Delete the oldest rotated files until the directory fits the cap.
    /// The timestamped names sort chronologically, so lexicographic order is
    /// age order; the active file is never deleted.
    fn enforce_retention(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut rotated: Vec<(PathBuf, u64)> = entries
            .flatten()
            .filter_map(|e| {
                let path = e.path();
                let name = path.file_name()?.to_str()?;
                if !name.starts_with("requests-") || name == ACTIVE_FILE {
                    return None;
                }
                let len = e.metadata().ok()?.len();
                Some((path, len))
            })
            .collect();
        rotated.sort();
        let mut total: u64 = rotated.iter().map(|(_, len)| len).sum::<u64>() + self.current_bytes;
        for (path, len) in rotated {
            if total <= self.config.max_total_bytes {
                break;
            }
            match fs::remove_file(&path) {
                Ok(()) => total = total.saturating_sub(len),
                Err(e) => {
                    tracing::warn!(error = %e, file = %path.display(), "request log retention delete failed")
                }
            }
        }
    }
}

fn gzip_file(path: &Path) -> std::io::Result<()> {
    let gz_path = path.with_extension("jsonl.gz");
    let input = fs::File::open(path)?;
    let output = fs::File::create(&gz_path)?;
    let mut encoder =
        flate2::write::GzEncoder::new(BufWriter::new(output), flate2::Compression::default());
    let mut reader = std::io::BufReader::new(input);
    std::io::copy(&mut reader, &mut encoder)?;
    encoder.finish()?.flush()?;
    fs::remove_file(path)
}

// ── Capture middleware ──────────────────────────────────────────────────

/// Tap middleware applied to the inference routes. Buffers the request body
/// (the JSON handlers buffer it anyway, under the same 8 MiB route limit),
/// runs the handler, then captures the response — whole-body for JSON,
/// reassembled-final-message for SSE — and queues one log row.
pub async fn tap(State(state): State<AppState>, req: Request<Body>, next: Next) -> Response {
    let Some(logger) = state.request_log.clone() else {
        return next.run(req).await;
    };
    let started = Instant::now();
    let route = req.uri().path().to_string();
    let user_agent = req
        .headers()
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.chars().take(256).collect::<String>());
    // `X-Kiln-Client` follows the same path as User-Agent: the /ui dashboard
    // self-identifies (`dashboard`) so its own traffic is distinguishable.
    let client = req
        .headers()
        .get("x-kiln-client")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|v| v.chars().take(64).collect::<String>());
    let max_capture = state.request_log_max_capture_bytes;

    let (parts, body) = req.into_parts();
    // Same ceiling as the routes' DefaultBodyLimit (8 MiB); anything larger
    // would be rejected by the handler's extractor anyway.
    let body_bytes = match axum::body::to_bytes(body, 8 * 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            // Over-limit or aborted body: forward an empty body so the
            // handler produces its normal error; nothing useful to log.
            return next.run(Request::from_parts(parts, Body::empty())).await;
        }
    };
    let (request_value, request_truncated) = capture_json(&body_bytes, max_capture);
    let req = Request::from_parts(parts, Body::from(body_bytes));

    let response = next.run(req).await;
    let status = response.status().as_u16();
    let adapter = adapter_from_headers(response.headers());
    let is_sse = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));

    if !is_sse {
        let (parts, body) = response.into_parts();
        // Non-stream inference responses are in-memory JSON; usize::MAX is
        // safe because the handler already materialized the body.
        let bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(_) => Bytes::new(),
        };
        let (response_value, response_truncated) = capture_json(&bytes, max_capture);
        logger.log(RequestLogEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            route,
            status,
            duration_ms: started.elapsed().as_millis() as u64,
            streamed: false,
            user_agent,
            client,
            adapter,
            request: request_value,
            response: response_value,
            request_truncated,
            response_truncated,
            stream_interrupted: false,
        });
        return Response::from_parts(parts, Body::from(bytes));
    }

    // SSE: pass chunks through while accumulating (capped); log on stream
    // end — or on drop, so client disconnects still leave a row.
    let (parts, body) = response.into_parts();
    let ctx = SseTapCtx {
        logger,
        ts_route: route,
        status,
        started,
        user_agent,
        client,
        adapter,
        request_value,
        request_truncated,
        max_capture,
        buf: Vec::new(),
        truncated: false,
        finished: false,
    };
    let tapped = SseTap {
        inner: body.into_data_stream(),
        ctx: Some(ctx),
    };
    Response::from_parts(parts, Body::from_stream(tapped))
}

/// Adapter that served the response, read from the runtime headers the
/// chat handlers attach. "base"/missing → None (base model).
fn adapter_from_headers(headers: &axum::http::HeaderMap) -> Option<String> {
    headers
        .get("x-kiln-loaded-adapter")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && *v != "base" && *v != "invalid")
        .map(|v| v.to_string())
}

fn capture_json(bytes: &[u8], max_capture: usize) -> (serde_json::Value, bool) {
    if bytes.len() <= max_capture {
        match serde_json::from_slice::<serde_json::Value>(bytes) {
            Ok(v) => (v, false),
            Err(_) => (
                serde_json::json!({ "_raw": String::from_utf8_lossy(bytes) }),
                false,
            ),
        }
    } else {
        let head = String::from_utf8_lossy(&bytes[..max_capture]).into_owned();
        (serde_json::json!({ "_raw": head }), true)
    }
}

struct SseTapCtx {
    logger: Arc<RequestLogger>,
    ts_route: String,
    status: u16,
    started: Instant,
    user_agent: Option<String>,
    client: Option<String>,
    adapter: Option<String>,
    request_value: serde_json::Value,
    request_truncated: bool,
    max_capture: usize,
    buf: Vec<u8>,
    truncated: bool,
    finished: bool,
}

impl SseTapCtx {
    fn observe(&mut self, chunk: &[u8]) {
        if self.buf.len() < self.max_capture {
            let room = self.max_capture - self.buf.len();
            if chunk.len() > room {
                self.buf.extend_from_slice(&chunk[..room]);
                self.truncated = true;
            } else {
                self.buf.extend_from_slice(chunk);
            }
        } else {
            self.truncated = true;
        }
        if !self.finished && sse_saw_done(&self.buf) {
            self.finished = true;
        }
    }

    fn finalize(mut self) {
        let finished = self.finished || sse_saw_done(&self.buf);
        let response = reassemble_sse(&self.buf);
        self.logger.log(RequestLogEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            route: std::mem::take(&mut self.ts_route),
            status: self.status,
            duration_ms: self.started.elapsed().as_millis() as u64,
            streamed: true,
            user_agent: self.user_agent.take(),
            client: self.client.take(),
            adapter: self.adapter.take(),
            request: std::mem::take(&mut self.request_value),
            response,
            request_truncated: self.request_truncated,
            response_truncated: self.truncated,
            stream_interrupted: !finished,
        });
    }
}

struct SseTap<S> {
    inner: S,
    ctx: Option<SseTapCtx>,
}

impl<S, E> Stream for SseTap<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match std::pin::Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Some(ctx) = this.ctx.as_mut() {
                    ctx.observe(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                if let Some(ctx) = this.ctx.take() {
                    ctx.finalize();
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for SseTap<S> {
    fn drop(&mut self) {
        // Client disconnect / handler abort: still log what was streamed.
        if let Some(ctx) = self.ctx.take() {
            ctx.finalize();
        }
    }
}

fn sse_saw_done(buf: &[u8]) -> bool {
    // The DONE sentinel is within the final few bytes; a tail window check
    // avoids rescanning a large buffer per chunk.
    let tail_start = buf.len().saturating_sub(64);
    String::from_utf8_lossy(&buf[tail_start..]).contains("data: [DONE]")
}

/// Reassemble an accumulated SSE byte buffer into the non-streaming response
/// shape (`{id, model, choices: [{message, finish_reason}], usage, metadata}`),
/// so streamed and non-streamed rows are uniformly minable. Final
/// thinking-budget metadata is retained and its per-choice outcome is restored
/// to the same location used by non-streaming responses. Unparseable buffers
/// degrade to `{"_raw": ...}`.
fn reassemble_sse(buf: &[u8]) -> serde_json::Value {
    let text = String::from_utf8_lossy(buf);
    let mut id = None;
    let mut model = None;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut finish_reason = None;
    let mut usage = None;
    let mut metadata = None;
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut parsed_any = false;

    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else {
            continue;
        };
        parsed_any = true;
        if id.is_none() {
            id = chunk.get("id").cloned();
        }
        if model.is_none() {
            model = chunk.get("model").cloned();
        }
        if let Some(u) = chunk.get("usage") {
            if !u.is_null() {
                usage = Some(u.clone());
            }
        }
        if let Some(value) = chunk.get("metadata") {
            if !value.is_null() {
                metadata = Some(value.clone());
            }
        }
        let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
            continue;
        };
        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            finish_reason = Some(fr.to_string());
        }
        let Some(delta) = choice.get("delta") else {
            continue;
        };
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
            reasoning.push_str(r);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            merge_tool_call_deltas(&mut tool_calls, calls);
        }
    }

    if !parsed_any {
        return serde_json::json!({ "_raw": text });
    }

    let mut message = serde_json::json!({ "role": "assistant", "content": content });
    if !reasoning.is_empty() {
        message["reasoning_content"] = serde_json::Value::String(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    let mut choice = serde_json::json!({
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
    });
    if let Some(outcome) = metadata
        .as_ref()
        .and_then(thinking_budget_outcome_from_metadata)
    {
        choice["thinking_budget"] = outcome;
    }
    let mut out = serde_json::json!({ "choices": [choice] });
    if let Some(id) = id {
        out["id"] = id;
    }
    if let Some(model) = model {
        out["model"] = model;
    }
    if let Some(usage) = usage {
        out["usage"] = usage;
    }
    if let Some(metadata) = metadata {
        out["metadata"] = metadata;
    }
    out
}

fn thinking_budget_outcome_from_metadata(
    metadata: &serde_json::Value,
) -> Option<serde_json::Value> {
    let budget = metadata.get("thinking_budget")?;
    let triggered = budget.get("triggered")?.as_bool()?;
    let closed = budget.get("closed")?.as_bool()?;
    let thinking_tokens = budget.get("thinking_tokens")?.as_u64()?;
    let thinking_time_ms = budget.get("thinking_time_ms")?.as_u64()?;
    let mut outcome = serde_json::json!({
        "triggered": triggered,
        "closed": closed,
        "thinking_tokens": thinking_tokens,
        "thinking_time_ms": thinking_time_ms,
    });
    if let Some(trigger) = budget.get("trigger").and_then(|value| value.as_str()) {
        outcome["trigger"] = serde_json::Value::String(trigger.to_string());
    }
    Some(outcome)
}

/// Merge one chunk's `delta.tool_calls` into the accumulated list. Streaming
/// tool calls arrive as an `index`-keyed series where `function.arguments`
/// fragments concatenate.
fn merge_tool_call_deltas(acc: &mut Vec<serde_json::Value>, deltas: &[serde_json::Value]) {
    for delta in deltas {
        let index = delta.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        while acc.len() <= index {
            acc.push(serde_json::json!({
                "type": "function",
                "function": { "name": "", "arguments": "" },
            }));
        }
        let slot = &mut acc[index];
        if let Some(id) = delta.get("id").and_then(|v| v.as_str()) {
            slot["id"] = serde_json::Value::String(id.to_string());
        }
        if let Some(func) = delta.get("function") {
            if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                slot["function"]["name"] = serde_json::Value::String(name.to_string());
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                let existing = slot["function"]["arguments"].as_str().unwrap_or("");
                slot["function"]["arguments"] =
                    serde_json::Value::String(format!("{existing}{args}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn entry(route: &str, payload_size: usize) -> RequestLogEntry {
        RequestLogEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            route: route.into(),
            status: 200,
            duration_ms: 5,
            streamed: false,
            user_agent: Some("test".into()),
            client: None,
            adapter: None,
            request: serde_json::json!({ "messages": [{"role": "user", "content": "x".repeat(payload_size)}] }),
            response: serde_json::json!({ "choices": [{"message": {"role": "assistant", "content": "y"}}] }),
            request_truncated: false,
            response_truncated: false,
            stream_interrupted: false,
        }
    }

    fn read_first_line(path: &Path) -> serde_json::Value {
        let text = fs::read_to_string(path).unwrap();
        serde_json::from_str(text.lines().next().unwrap()).unwrap()
    }

    #[test]
    fn config_rejects_whitespace_only_directory() {
        let config = RequestLogConfig {
            dir: Some(PathBuf::from("   \t")),
            ..Default::default()
        };
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("request_log.dir"), "{error}");
        assert!(error.contains("non-empty"), "{error}");
    }

    #[test]
    fn logs_one_json_line_per_entry() {
        let dir = tempdir().unwrap();
        let logger =
            RequestLogger::spawn(dir.path().to_path_buf(), RequestLogConfig::default()).unwrap();
        logger.log(entry("/v1/chat/completions", 8));
        logger.flush();
        let row = read_first_line(&dir.path().join(ACTIVE_FILE));
        assert_eq!(row["route"], "/v1/chat/completions");
        assert_eq!(row["status"], 200);
        assert_eq!(row["request"]["messages"][0]["role"], "user");
        assert_eq!(row["response"]["choices"][0]["message"]["content"], "y");
    }

    #[test]
    fn rotates_compresses_and_enforces_retention() {
        let dir = tempdir().unwrap();
        let config = RequestLogConfig {
            max_file_bytes: 2_000,
            max_total_bytes: 6_000,
            compress: true,
            ..Default::default()
        };
        let logger = RequestLogger::spawn(dir.path().to_path_buf(), config).unwrap();
        for _ in 0..40 {
            logger.log(entry("/v1/chat/completions", 512));
        }
        logger.flush();

        let names: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().any(|n| n == ACTIVE_FILE));
        let rotated: Vec<&String> = names.iter().filter(|n| n.ends_with(".jsonl.gz")).collect();
        assert!(
            !rotated.is_empty(),
            "expected compressed rotated files, got {names:?}"
        );
        // No uncompressed rotated leftovers.
        assert!(
            !names
                .iter()
                .any(|n| n.starts_with("requests-0") && n.ends_with(".jsonl")),
            "rotated plain files should be gzipped+removed: {names:?}"
        );
        // Retention: total on-disk stays near the cap (compressed files are
        // tiny, so just assert the cap held with slack for the active file).
        let total: u64 = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.metadata().unwrap().len())
            .sum();
        assert!(
            total <= 6_000 + 2_000,
            "retention should bound the directory, got {total}"
        );

        // Rotated content survives the gzip round-trip as valid JSONL.
        let gz_name = rotated.iter().min().unwrap().to_string();
        let gz = fs::File::open(dir.path().join(&gz_name)).unwrap();
        let mut text = String::new();
        use std::io::Read;
        flate2::read::GzDecoder::new(gz)
            .read_to_string(&mut text)
            .unwrap();
        let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first["route"], "/v1/chat/completions");
    }

    #[test]
    fn adapter_from_headers_filters_base_and_invalid() {
        let mut headers = axum::http::HeaderMap::new();
        assert_eq!(adapter_from_headers(&headers), None);
        headers.insert("x-kiln-loaded-adapter", "base".parse().unwrap());
        assert_eq!(adapter_from_headers(&headers), None);
        headers.insert("x-kiln-loaded-adapter", "invalid".parse().unwrap());
        assert_eq!(adapter_from_headers(&headers), None);
        headers.insert("x-kiln-loaded-adapter", "my-adapter".parse().unwrap());
        assert_eq!(
            adapter_from_headers(&headers),
            Some("my-adapter".to_string())
        );
    }

    #[test]
    fn capture_json_truncates_oversized_bodies() {
        let big = format!("{{\"k\":\"{}\"}}", "v".repeat(100));
        let (value, truncated) = capture_json(big.as_bytes(), 16);
        assert!(truncated);
        assert!(value["_raw"].as_str().unwrap().len() <= 16);
        let (value, truncated) = capture_json(b"{\"a\":1}", 1024);
        assert!(!truncated);
        assert_eq!(value["a"], 1);
        let (value, truncated) = capture_json(b"not json", 1024);
        assert!(!truncated);
        assert_eq!(value["_raw"], "not json");
    }

    #[test]
    fn reassembles_sse_stream_into_final_message() {
        let sse = concat!(
            "data: {\"id\":\"chatcmpl-1\",\"model\":\"m\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"cmd\\\":\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":7}}\n\n",
            "data: [DONE]\n\n",
        );
        let out = reassemble_sse(sse.as_bytes());
        assert_eq!(out["id"], "chatcmpl-1");
        assert_eq!(out["model"], "m");
        let message = &out["choices"][0]["message"];
        assert_eq!(message["content"], "Hello");
        assert_eq!(message["reasoning_content"], "think");
        assert_eq!(message["tool_calls"][0]["id"], "call_1");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "bash");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"cmd\":\"ls\"}"
        );
        assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(out["usage"]["completion_tokens"], 7);
        assert!(sse_saw_done(sse.as_bytes()));
    }

    fn reassemble_budget_stream(metadata: &serde_json::Value) -> serde_json::Value {
        let first = serde_json::json!({
            "id": "chatcmpl-budget",
            "model": "m",
            "choices": [{"delta": {
                "role": "assistant",
                "reasoning_content": "reason",
                "content": "answer"
            }}]
        });
        let finish = serde_json::json!({
            "id": "chatcmpl-budget",
            "model": "m",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "metadata": metadata,
        });
        let usage = serde_json::json!({
            "id": "chatcmpl-budget",
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        });
        let sse = format!("data: {first}\n\ndata: {finish}\n\ndata: {usage}\n\ndata: [DONE]\n\n");
        reassemble_sse(sse.as_bytes())
    }

    #[test]
    fn durable_budget_fields_match_non_streaming_shape() {
        let cases = [
            (
                "token trigger",
                serde_json::json!({"thinking_budget": {
                    "configured": true,
                    "applied": true,
                    "max_tokens": 8,
                    "tokens_source": "request",
                    "time_source": "unlimited",
                    "triggered": true,
                    "trigger": "tokens",
                    "closed": true,
                    "thinking_tokens": 8,
                    "thinking_time_ms": 17
                }}),
                Some(serde_json::json!({
                    "triggered": true,
                    "trigger": "tokens",
                    "closed": true,
                    "thinking_tokens": 8,
                    "thinking_time_ms": 17
                })),
            ),
            (
                "natural close",
                serde_json::json!({"thinking_budget": {
                    "configured": true,
                    "applied": true,
                    "max_time_ms": 50,
                    "tokens_source": "unlimited",
                    "time_source": "server_default",
                    "triggered": false,
                    "closed": true,
                    "thinking_tokens": 5,
                    "thinking_time_ms": 9
                }}),
                Some(serde_json::json!({
                    "triggered": false,
                    "closed": true,
                    "thinking_tokens": 5,
                    "thinking_time_ms": 9
                })),
            ),
            (
                "partial forced close",
                serde_json::json!({"thinking_budget": {
                    "configured": true,
                    "applied": true,
                    "max_tokens": 0,
                    "tokens_source": "request",
                    "time_source": "unlimited",
                    "triggered": true,
                    "trigger": "tokens",
                    "closed": false,
                    "thinking_tokens": 0,
                    "thinking_time_ms": 2
                }}),
                Some(serde_json::json!({
                    "triggered": true,
                    "trigger": "tokens",
                    "closed": false,
                    "thinking_tokens": 0,
                    "thinking_time_ms": 2
                })),
            ),
            (
                "configured but inert",
                serde_json::json!({"thinking_budget": {
                    "configured": true,
                    "applied": false,
                    "max_tokens": 8,
                    "tokens_source": "request",
                    "time_source": "unlimited",
                    "triggered": false
                }}),
                None,
            ),
        ];

        for (name, metadata, expected_outcome) in cases {
            let streamed = reassemble_budget_stream(&metadata);
            let mut non_streaming_choice = serde_json::json!({
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reason"
                },
                "finish_reason": "stop"
            });
            if let Some(outcome) = expected_outcome {
                non_streaming_choice["thinking_budget"] = outcome;
            }
            let non_streaming = serde_json::json!({
                "id": "chatcmpl-budget",
                "model": "m",
                "choices": [non_streaming_choice],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
                "metadata": metadata,
            });

            for pointer in [
                "/choices/0/message",
                "/choices/0/finish_reason",
                "/choices/0/thinking_budget",
                "/usage",
                "/metadata/thinking_budget",
            ] {
                assert_eq!(
                    streamed.pointer(pointer),
                    non_streaming.pointer(pointer),
                    "{name}: durable response mismatch at {pointer}"
                );
            }
        }
    }

    #[test]
    fn reassemble_degrades_to_raw_on_garbage() {
        let out = reassemble_sse(b"plain text body");
        assert_eq!(out["_raw"], "plain text body");
    }
}
