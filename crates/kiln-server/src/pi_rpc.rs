//! pi RPC subprocess driver — spawns `pi --mode rpc` and speaks its
//! JSON-lines protocol over stdin/stdout.
//!
//! pi's RPC mode is the headless embedding surface: commands go in as
//! one JSON object per line (`{"type":"prompt","message":...}`), and
//! the process streams back command responses (`{"type":"response",..}`)
//! interleaved with agent events (`agent_start`, `message_end`,
//! `tool_execution_end`, `agent_end`, ...).
//!
//! Framing is strict JSONL with LF as the only record delimiter; a
//! trailing CR is stripped (the protocol allows CRLF input). Unicode
//! line separators (U+2028/U+2029) are valid inside JSON strings and
//! must NOT split records — `read_until(b'\n')` gets this right where
//! generic line readers would not.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::mpsc;

/// Resolve the `pi` binary from typed configuration or a startup-owned PATH
/// snapshot. Runtime request handling retains the result and never rereads the
/// process environment.
pub fn find_pi(configured: Option<&Path>, search_path: Option<&OsStr>) -> Option<PathBuf> {
    if let Some(path) = configured {
        return path.is_file().then(|| path.to_path_buf());
    }
    for dir in std::env::split_paths(search_path?) {
        let candidate = dir.join("pi");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Options for one `pi --mode rpc` child.
#[derive(Debug, Clone)]
pub struct PiRpcOptions {
    /// Working directory the agent operates in.
    pub cwd: PathBuf,
    /// pi provider name to select (the `kiln pi-setup` merge registers
    /// `kiln-local`).
    pub provider: String,
    /// Model id under that provider — the id this server announces.
    pub model: String,
    /// Directory pi stores the session JSONL in. Keeping embedded runs
    /// under a kiln-owned dir separates them from the user's own
    /// `~/.pi/agent/sessions` and lets the trace layer index them.
    pub session_dir: PathBuf,
    /// Session display name (shows up in pi's session list / traces).
    pub session_name: Option<String>,
    /// Optional tool allowlist (`--tools read,bash,edit,write`).
    pub tools: Option<Vec<String>>,
    /// Optional thinking level (`--thinking low|medium|high|...`).
    pub thinking_level: Option<String>,
}

/// A line pi emitted on stdout, parsed.
#[derive(Debug)]
pub enum PiRpcLine {
    /// A well-formed JSON record (response or agent event).
    Json(serde_json::Value),
    /// stdout closed — the child exited or killed its pipe.
    Eof,
}

/// Handle to a running `pi --mode rpc` child. Dropping the handle does
/// not kill the child — call [`PiRpcProcess::shutdown`].
pub struct PiRpcProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    pub lines: mpsc::Receiver<PiRpcLine>,
}

impl PiRpcProcess {
    /// Spawn `pi --mode rpc` with piped stdio. stdout is pumped into
    /// `lines` by a background task; stderr is drained into tracing.
    pub fn spawn(pi_bin: &std::path::Path, opts: &PiRpcOptions) -> std::io::Result<Self> {
        let mut cmd = Command::new(pi_bin);
        cmd.arg("--mode")
            .arg("rpc")
            .arg("--provider")
            .arg(&opts.provider)
            .arg("--model")
            .arg(&opts.model)
            .arg("--session-dir")
            .arg(&opts.session_dir);
        if let Some(name) = &opts.session_name {
            cmd.arg("--name").arg(name);
        }
        if let Some(tools) = &opts.tools {
            if !tools.is_empty() {
                cmd.arg("--tools").arg(tools.join(","));
            }
        }
        if let Some(level) = &opts.thinking_level {
            cmd.arg("--thinking").arg(level);
        }
        cmd.current_dir(&opts.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn()?;
        let stdout = child.stdout.take().expect("stdout piped above");
        let stderr = child.stderr.take().expect("stderr piped above");
        let stdin = child.stdin.take().expect("stdin piped above");

        let (tx, rx) = mpsc::channel::<PiRpcLine>(256);
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            let mut buf = Vec::with_capacity(8 * 1024);
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if let Some(value) = parse_rpc_line(&buf) {
                            if tx.send(PiRpcLine::Json(value)).await.is_err() {
                                return; // receiver gone — stop pumping
                            }
                        }
                    }
                }
            }
            let _ = tx.send(PiRpcLine::Eof).await;
        });
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        if !trimmed.is_empty() {
                            tracing::debug!(target: "pi_rpc", stderr = %trimmed, "pi stderr");
                        }
                    }
                }
            }
        });

        Ok(Self {
            child,
            stdin: Some(stdin),
            lines: rx,
        })
    }

    /// Send one RPC command (a JSON object) as a single LF-terminated line.
    pub async fn send(&mut self, command: &serde_json::Value) -> std::io::Result<()> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "pi stdin already closed",
            ));
        };
        let mut line = serde_json::to_vec(command)?;
        line.push(b'\n');
        stdin.write_all(&line).await?;
        stdin.flush().await
    }

    /// Graceful shutdown: close stdin (pi exits on EOF), wait up to
    /// `grace`, then kill. Always reaps the child.
    pub async fn shutdown(mut self, grace: std::time::Duration) {
        drop(self.stdin.take());
        match tokio::time::timeout(grace, self.child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                let _ = self.child.start_kill();
                let _ = self.child.wait().await;
            }
        }
    }
}

/// Parse one raw record: strip the LF delimiter and an optional
/// trailing CR, then parse JSON. Non-JSON lines (startup chatter,
/// partial writes at kill time) are dropped with a debug log.
fn parse_rpc_line(raw: &[u8]) -> Option<serde_json::Value> {
    let mut slice = raw;
    if slice.last() == Some(&b'\n') {
        slice = &slice[..slice.len() - 1];
    }
    if slice.last() == Some(&b'\r') {
        slice = &slice[..slice.len() - 1];
    }
    if slice.is_empty() {
        return None;
    }
    match serde_json::from_slice::<serde_json::Value>(slice) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::debug!(error = %e, line = %String::from_utf8_lossy(slice), "pi rpc: non-JSON stdout line dropped");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strips_lf_and_crlf() {
        let v = parse_rpc_line(b"{\"type\":\"agent_start\"}\n").unwrap();
        assert_eq!(v["type"], "agent_start");
        let v = parse_rpc_line(b"{\"type\":\"agent_end\"}\r\n").unwrap();
        assert_eq!(v["type"], "agent_end");
    }

    #[test]
    fn parse_keeps_unicode_line_separators_inside_strings() {
        // U+2028 inside a JSON string is data, not a record delimiter.
        let raw = "{\"type\":\"message_end\",\"text\":\"a\u{2028}b\"}\n";
        let v = parse_rpc_line(raw.as_bytes()).unwrap();
        assert_eq!(v["text"], "a\u{2028}b");
    }

    #[test]
    fn parse_drops_empty_and_garbage_lines() {
        assert!(parse_rpc_line(b"\n").is_none());
        assert!(parse_rpc_line(b"\r\n").is_none());
        assert!(parse_rpc_line(b"pi starting up...\n").is_none());
    }

    #[test]
    fn find_pi_prefers_typed_config() {
        let dir = tempfile::tempdir().unwrap();
        let fake = dir.path().join("pi");
        std::fs::write(&fake, "#!/bin/sh\n").unwrap();
        let found = find_pi(Some(&fake), None);
        assert_eq!(found, Some(fake));
    }
}
