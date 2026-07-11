//! Embedded pi terminal — a PTY-spawned `pi` session streamed to the
//! dashboard over a WebSocket, pre-configured to talk to THIS Kiln.
//!
//! | Method | Path                | Purpose                                   |
//! |--------|---------------------|-------------------------------------------|
//! | GET    | /v1/terminal/status | Availability: gate state, pi binary, cwd  |
//! | GET    | /v1/terminal/ws     | WebSocket: PTY in/out + resize control    |
//!
//! Protocol (after upgrade):
//! - server → client: Binary frames are raw PTY output; one initial Text JSON
//!   `{"type":"ready","cwd":...,"pi":...}`; a final Text JSON
//!   `{"type":"exit"}` when the child terminates.
//! - client → server: Binary frames are raw keystrokes; Text JSON
//!   `{"type":"resize","cols":N,"rows":N}` resizes the PTY.
//!
//! Security gate: an interactive agent terminal is arbitrary-code-execution
//! grade, so it is enabled only when the server is bound to a loopback
//! address (the default), or when the operator explicitly opts in with
//! `KILN_TERMINAL=1`. `KILN_TERMINAL=0` force-disables it everywhere.
//! Before each session the server runs the same non-destructive config merge
//! as `kiln pi-setup`, pointed at this server's own URL, so the embedded pi
//! is the user's pi — already connected.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{
    Json, Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    http::HeaderMap,
    response::IntoResponse,
    routing::get,
};
use portable_pty::{CommandBuilder, PtySize, native_pty_system};

use crate::state::AppState;

static BIND_HOST: OnceLock<String> = OnceLock::new();
/// One interactive session at a time — a second tab gets a clean refusal
/// instead of two keyboards fighting over one PTY.
static SESSION_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Record the host the server bound to (from main) so the gate can tell
/// loopback from network exposure.
pub fn set_bind_host(host: &str) {
    let _ = BIND_HOST.set(host.to_string());
}

pub(crate) fn bind_host_is_loopback() -> bool {
    let host = BIND_HOST.get().map(String::as_str).unwrap_or("127.0.0.1");
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
}

/// (enabled, human-readable reason when disabled)
fn terminal_gate() -> (bool, Option<String>) {
    match std::env::var("KILN_TERMINAL").as_deref() {
        Ok("0") => {
            return (
                false,
                Some("disabled by KILN_TERMINAL=0 on the server".into()),
            );
        }
        Ok("1") => return (true, None),
        _ => {}
    }
    if bind_host_is_loopback() {
        (true, None)
    } else {
        (
            false,
            Some(format!(
                "the server is bound to {} (not loopback) — an interactive terminal would be \
                 network-exposed. Set KILN_TERMINAL=1 to enable it anyway.",
                BIND_HOST.get().map(String::as_str).unwrap_or("?")
            )),
        )
    }
}

/// Locate `pi` — shared with the embedded-run engine (honors KILN_PI_BIN).
fn find_pi() -> Option<PathBuf> {
    crate::pi_rpc::find_pi()
}

async fn terminal_status() -> Json<serde_json::Value> {
    let (enabled, reason) = terminal_gate();
    let pi = find_pi();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "?".into());
    Json(serde_json::json!({
        "enabled": enabled,
        "disabled_reason": reason,
        "pi_available": pi.is_some(),
        "pi_path": pi.map(|p| p.display().to_string()),
        "cwd": cwd,
        "session_active": SESSION_ACTIVE.load(Ordering::SeqCst),
    }))
}

/// Best-effort server URL for pi's provider config, derived from the Host
/// header the browser actually used — correct across port-forwards.
fn kiln_url_from_headers(headers: &HeaderMap) -> String {
    let default_host = format!(
        "{}:{}",
        crate::config::DEFAULT_SERVER_CLIENT_HOST,
        crate::config::DEFAULT_SERVER_PORT
    );
    let host = headers
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&default_host);
    format!("http://{host}")
}

async fn terminal_ws(
    ws: WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let (enabled, reason) = terminal_gate();
    if !enabled {
        return (
            axum::http::StatusCode::FORBIDDEN,
            reason.unwrap_or_else(|| "terminal disabled".into()),
        )
            .into_response();
    }
    let kiln_url = kiln_url_from_headers(&headers);
    let served_model_id = state.served_model_id.clone();
    ws.on_upgrade(move |socket| handle_session(socket, kiln_url, served_model_id))
        .into_response()
}

async fn handle_session(mut socket: WebSocket, kiln_url: String, served_model_id: String) {
    // Single-session guard.
    if SESSION_ACTIVE.swap(true, Ordering::SeqCst) {
        let _ = socket
            .send(Message::Text(
                serde_json::json!({"type":"error","message":"another terminal session is already open — close it first"})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = socket.send(Message::Close(None)).await;
        return;
    }
    // RAII-ish release on every exit path below.
    struct Release;
    impl Drop for Release {
        fn drop(&mut self) {
            SESSION_ACTIVE.store(false, Ordering::SeqCst);
        }
    }
    let _release = Release;

    let Some(pi_path) = find_pi() else {
        let _ = socket
            .send(Message::Text(
                serde_json::json!({"type":"error","message":"`pi` is not installed on the server's PATH"})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = socket.send(Message::Close(None)).await;
        return;
    };

    // Same non-destructive merge as `kiln pi-setup`, pointed at this server —
    // with the model id this server actually announces.
    if let Err(err) = crate::cli::apply_pi_setup_quiet(&kiln_url, Some(&served_model_id)) {
        tracing::warn!(error = %err, "pi-setup merge failed; launching pi with existing config");
    }

    let pty = native_pty_system();
    let pair = match pty.openpty(PtySize {
        rows: 30,
        cols: 100,
        pixel_width: 0,
        pixel_height: 0,
    }) {
        Ok(p) => p,
        Err(err) => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"type":"error","message":format!("could not allocate a PTY: {err}")})
                        .to_string()
                        .into(),
                ))
                .await;
            let _ = socket.send(Message::Close(None)).await;
            return;
        }
    };

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut cmd = CommandBuilder::new(&pi_path);
    cmd.cwd(&cwd);
    // Make sure pi sees a sane terminal.
    cmd.env("TERM", "xterm-256color");
    let mut child = match pair.slave.spawn_command(cmd) {
        Ok(c) => c,
        Err(err) => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"type":"error","message":format!("could not start pi: {err}")})
                        .to_string()
                        .into(),
                ))
                .await;
            let _ = socket.send(Message::Close(None)).await;
            return;
        }
    };
    drop(pair.slave);

    let mut reader = match pair.master.try_clone_reader() {
        Ok(r) => r,
        Err(err) => {
            tracing::warn!(error = %err, "pty reader clone failed");
            let _ = child.kill();
            return;
        }
    };
    let mut writer = match pair.master.take_writer() {
        Ok(w) => w,
        Err(err) => {
            tracing::warn!(error = %err, "pty writer take failed");
            let _ = child.kill();
            return;
        }
    };
    let master = pair.master;

    let _ = socket
        .send(Message::Text(
            serde_json::json!({
                "type": "ready",
                "cwd": cwd.display().to_string(),
                "pi": pi_path.display().to_string(),
                "kiln_url": kiln_url,
            })
            .to_string()
            .into(),
        ))
        .await;

    // PTY output → channel (blocking reader thread).
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break, // child exited / PTY closed
                Ok(n) => {
                    if out_tx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Keystrokes → PTY (blocking writer thread).
    let (in_tx, in_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = in_rx.recv() {
            if writer.write_all(&bytes).is_err() {
                break;
            }
            let _ = writer.flush();
        }
    });

    loop {
        tokio::select! {
            out = out_rx.recv() => {
                match out {
                    Some(bytes) => {
                        if socket.send(Message::Binary(bytes.into())).await.is_err() {
                            break;
                        }
                    }
                    None => {
                        // Child exited — tell the client, then close.
                        let _ = socket
                            .send(Message::Text(serde_json::json!({"type":"exit"}).to_string().into()))
                            .await;
                        break;
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Binary(bytes))) => {
                        if in_tx.send(bytes.to_vec()).is_err() {
                            break;
                        }
                    }
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text)
                            && v.get("type").and_then(|t| t.as_str()) == Some("resize")
                        {
                            let cols = v.get("cols").and_then(|c| c.as_u64()).unwrap_or(100) as u16;
                            let rows = v.get("rows").and_then(|r| r.as_u64()).unwrap_or(30) as u16;
                            let _ = master.resize(PtySize { rows, cols, pixel_width: 0, pixel_height: 0 });
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
        }
    }

    let _ = child.kill();
    let _ = socket.send(Message::Close(None)).await;
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/terminal/status", get(terminal_status))
        .route("/v1/terminal/ws", get(terminal_ws))
}
