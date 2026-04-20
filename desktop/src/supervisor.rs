use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{watch, Mutex};
use tokio::task::JoinHandle;
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", content = "message")]
pub enum ServerState {
    Stopped,
    Starting,
    Running,
    TrainingActive,
    Error(String),
    /// The configured `kiln` binary is not installed at the expected
    /// location (and was not found on `PATH`). Distinct from `Error` so
    /// the desktop can show an onboarding flow instead of a crash
    /// notification on first run.
    NoBinary(String),
}

#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    pub binary_path: PathBuf,
    pub args: Vec<String>,
    /// Environment variables passed to the spawned kiln server. The desktop
    /// uses these (rather than CLI flags) because kiln's CLI is structured
    /// around `--config <toml>` and exposes overrides via `KILN_*` env vars,
    /// not per-setting flags. See `settings::apply_to_supervisor_config`.
    pub envs: Vec<(String, String)>,
    pub auto_restart: bool,
    pub max_restarts: u32,
    pub restart_backoff_ms: u64,
    pub log_buffer_bytes: usize,
    /// Host the kiln server binds to — used by the health poller to build
    /// `/v1/health` and `/v1/train/status` URLs.
    pub host: String,
    /// Port the kiln server binds to — used by the health poller.
    pub port: u16,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            binary_path: PathBuf::from("kiln"),
            args: Vec::new(),
            envs: Vec::new(),
            auto_restart: true,
            max_restarts: 5,
            restart_backoff_ms: 500,
            log_buffer_bytes: 4 * 1024 * 1024,
            host: "127.0.0.1".to_string(),
            port: 8000,
        }
    }
}

pub struct RingBuffer {
    lines: VecDeque<String>,
    bytes: usize,
    cap_bytes: usize,
}

impl RingBuffer {
    fn new(cap_bytes: usize) -> Self {
        Self {
            lines: VecDeque::new(),
            bytes: 0,
            cap_bytes,
        }
    }

    fn push(&mut self, line: String) {
        self.bytes = self.bytes.saturating_add(line.len());
        self.lines.push_back(line);
        while self.bytes > self.cap_bytes {
            match self.lines.pop_front() {
                Some(dropped) => {
                    self.bytes = self.bytes.saturating_sub(dropped.len());
                }
                None => break,
            }
        }
    }

    fn snapshot(&self) -> Vec<String> {
        self.lines.iter().cloned().collect()
    }
}

pub struct Supervisor {
    config: Arc<Mutex<SupervisorConfig>>,
    state: Arc<Mutex<ServerState>>,
    logs: Arc<Mutex<RingBuffer>>,
    task: Arc<Mutex<Option<JoinHandle<()>>>>,
    shutdown_tx: Arc<Mutex<Option<watch::Sender<bool>>>>,
}

impl Supervisor {
    pub fn new(config: SupervisorConfig) -> Self {
        let cap = config.log_buffer_bytes;
        Self {
            config: Arc::new(Mutex::new(config)),
            state: Arc::new(Mutex::new(ServerState::Stopped)),
            logs: Arc::new(Mutex::new(RingBuffer::new(cap))),
            task: Arc::new(Mutex::new(None)),
            shutdown_tx: Arc::new(Mutex::new(None)),
        }
    }

    /// Replace the supervisor config. Does NOT restart a running child.
    /// The new config takes effect on the next successful `start()`.
    pub async fn update_config(&self, new_cfg: SupervisorConfig) {
        *self.config.lock().await = new_cfg;
    }

    /// Clone of the current supervisor config. Used by the updater to
    /// build the `/v1/health` URL without taking a long-lived lock on
    /// the inner mutex.
    pub async fn config_snapshot(&self) -> SupervisorConfig {
        self.config.lock().await.clone()
    }

    pub async fn start(&self) -> Result<(), String> {
        let mut task_guard = self.task.lock().await;
        if let Some(handle) = task_guard.as_ref() {
            if !handle.is_finished() {
                return Err("supervisor already running".into());
            }
        }
        *task_guard = None;

        let (tx, rx) = watch::channel(false);
        let poller_shutdown = rx.clone();
        *self.shutdown_tx.lock().await = Some(tx);
        *self.state.lock().await = ServerState::Starting;

        let config = self.config.lock().await.clone();
        let state = Arc::clone(&self.state);
        let logs = Arc::clone(&self.logs);

        // Spawn the HTTP health poller alongside the run loop. It shares the
        // same shutdown receiver so `stop()` terminates both. The handle is
        // detached — the poller also exits on its own when state flips to
        // `Stopped`.
        let _ = crate::poller::spawn_health_poller(
            Arc::clone(&self.state),
            config.host.clone(),
            config.port,
            poller_shutdown,
        );

        let handle = tokio::spawn(async move {
            run_loop(config, state, logs, rx).await;
        });
        *task_guard = Some(handle);
        Ok(())
    }

    pub async fn stop(&self) -> Result<(), String> {
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(true);
        }
        let handle = self.task.lock().await.take();
        if let Some(handle) = handle {
            let _ = handle.await;
        }
        *self.state.lock().await = ServerState::Stopped;
        Ok(())
    }

    /// Stop the server if it's running, then start it again.
    ///
    /// Used by the tray's "Restart Server" menu item after the user changes
    /// settings that only take effect on the next start (port, host, model
    /// path, flags). Safe to call from any state: if the server is already
    /// stopped, this is equivalent to `start`.
    pub async fn restart(&self) -> Result<(), String> {
        let state = self.state().await;
        if matches!(
            state,
            ServerState::Starting | ServerState::Running | ServerState::TrainingActive
        ) {
            if let Err(e) = self.stop().await {
                eprintln!("[supervisor] restart: stop failed, attempting start anyway: {}", e);
            }
        }
        self.start().await
    }

    pub async fn state(&self) -> ServerState {
        self.state.lock().await.clone()
    }

    pub async fn logs(&self) -> Vec<String> {
        self.logs.lock().await.snapshot()
    }
}

async fn run_loop(
    config: SupervisorConfig,
    state: Arc<Mutex<ServerState>>,
    logs: Arc<Mutex<RingBuffer>>,
    mut shutdown: watch::Receiver<bool>,
) {
    let mut restarts: u32 = 0;
    loop {
        // Pre-flight: treat a missing binary as the `NoBinary` onboarding
        // state, not an Error. This both suppresses the "server crashed"
        // OS notification on fresh installs and lets the dashboard show a
        // download prompt instead of a raw spawn error.
        if crate::installer::resolve_binary(&config.binary_path).is_none() {
            let msg = config.binary_path.display().to_string();
            push_log(
                &logs,
                format!("[supervisor] kiln binary not found: {}", msg),
            )
            .await;
            *state.lock().await = ServerState::NoBinary(msg);
            return;
        }

        let spawn_result = Command::new(&config.binary_path)
            .args(&config.args)
            .envs(config.envs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match spawn_result {
            Ok(c) => c,
            Err(e) => {
                // Treat the specific "NotFound" kind as NoBinary too, in
                // case the binary disappeared between our pre-flight and
                // the spawn (race), or the configured path isn't resolvable
                // by our pre-flight heuristic.
                if matches!(e.kind(), std::io::ErrorKind::NotFound) {
                    let msg = config.binary_path.display().to_string();
                    push_log(
                        &logs,
                        format!("[supervisor] kiln binary not found at spawn: {}", msg),
                    )
                    .await;
                    *state.lock().await = ServerState::NoBinary(msg);
                    return;
                }
                let msg = format!(
                    "failed to spawn {}: {}",
                    config.binary_path.display(),
                    e
                );
                push_log(&logs, format!("[supervisor] {}", msg)).await;
                *state.lock().await = ServerState::Error(msg);
                return;
            }
        };

        *state.lock().await = ServerState::Running;

        if let Some(out) = child.stdout.take() {
            let logs_c = Arc::clone(&logs);
            tokio::spawn(async move {
                let mut reader = BufReader::new(out).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    push_log(&logs_c, format!("[stdout] {}", line)).await;
                }
            });
        }
        if let Some(err) = child.stderr.take() {
            let logs_c = Arc::clone(&logs);
            tokio::spawn(async move {
                let mut reader = BufReader::new(err).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    push_log(&logs_c, format!("[stderr] {}", line)).await;
                }
            });
        }

        let shutdown_fired;
        tokio::select! {
            exit = child.wait() => {
                shutdown_fired = false;
                push_log(
                    &logs,
                    format!("[supervisor] child exited: {:?}", exit),
                )
                .await;
            }
            _ = shutdown.changed() => {
                shutdown_fired = true;
            }
        }

        if shutdown_fired {
            let _ = child.kill().await;
            *state.lock().await = ServerState::Stopped;
            return;
        }

        if !config.auto_restart {
            *state.lock().await = ServerState::Stopped;
            return;
        }

        restarts = restarts.saturating_add(1);
        if restarts > config.max_restarts {
            let msg = format!("max restarts ({}) exceeded", config.max_restarts);
            push_log(&logs, format!("[supervisor] {}", msg)).await;
            *state.lock().await = ServerState::Error(msg);
            return;
        }

        // Exponential backoff capped at 30s: base * 2^min(restarts, 6)
        let shift = restarts.min(6) as u32;
        let backoff_ms = config
            .restart_backoff_ms
            .saturating_mul(1u64 << shift)
            .min(30_000);
        push_log(
            &logs,
            format!("[supervisor] restart {} in {}ms", restarts, backoff_ms),
        )
        .await;
        *state.lock().await = ServerState::Starting;
        tokio::select! {
            _ = sleep(Duration::from_millis(backoff_ms)) => {}
            _ = shutdown.changed() => {
                *state.lock().await = ServerState::Stopped;
                return;
            }
        }
    }
}

async fn push_log(logs: &Arc<Mutex<RingBuffer>>, line: String) {
    logs.lock().await.push(line);
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn starts_and_transitions_to_stopped_when_child_exits() {
        let config = SupervisorConfig {
            binary_path: PathBuf::from("/bin/sleep"),
            args: vec!["0.1".to_string()],
            envs: Vec::new(),
            auto_restart: false,
            max_restarts: 0,
            restart_backoff_ms: 100,
            log_buffer_bytes: 1024,
            host: "127.0.0.1".to_string(),
            port: 8000,
        };
        let sup = Supervisor::new(config);
        sup.start().await.expect("start");

        tokio::time::sleep(Duration::from_millis(50)).await;
        let s = sup.state().await;
        assert!(
            matches!(s, ServerState::Starting | ServerState::Running),
            "expected Starting or Running, got {:?}",
            s
        );

        tokio::time::sleep(Duration::from_millis(400)).await;
        let s = sup.state().await;
        assert!(
            matches!(s, ServerState::Stopped),
            "expected Stopped, got {:?}",
            s
        );
    }

    #[tokio::test]
    async fn stop_kills_running_process() {
        let config = SupervisorConfig {
            binary_path: PathBuf::from("/bin/sleep"),
            args: vec!["10".to_string()],
            envs: Vec::new(),
            auto_restart: false,
            max_restarts: 0,
            restart_backoff_ms: 100,
            log_buffer_bytes: 1024,
            host: "127.0.0.1".to_string(),
            port: 8000,
        };
        let sup = Supervisor::new(config);
        sup.start().await.expect("start");

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            matches!(sup.state().await, ServerState::Running),
            "expected Running"
        );

        sup.stop().await.expect("stop");
        assert!(
            matches!(sup.state().await, ServerState::Stopped),
            "expected Stopped"
        );
    }

    #[tokio::test]
    async fn no_binary_state_when_binary_missing() {
        let config = SupervisorConfig {
            binary_path: PathBuf::from("/definitely/not/a/binary/kiln-xyz"),
            args: vec![],
            envs: Vec::new(),
            auto_restart: false,
            max_restarts: 0,
            restart_backoff_ms: 10,
            log_buffer_bytes: 1024,
            host: "127.0.0.1".to_string(),
            port: 8000,
        };
        let sup = Supervisor::new(config);
        sup.start().await.expect("start");

        tokio::time::sleep(Duration::from_millis(200)).await;
        let s = sup.state().await;
        assert!(
            matches!(s, ServerState::NoBinary(_)),
            "expected NoBinary, got {:?}",
            s
        );
    }

    #[test]
    fn ring_buffer_evicts_oldest_over_cap() {
        let mut rb = RingBuffer::new(20);
        rb.push("aaaa".to_string()); // 4
        rb.push("bbbb".to_string()); // 8
        rb.push("cccccccccccc".to_string()); // 20
        // Total bytes = 20, at cap. Push one more pushes out first.
        rb.push("d".to_string());
        let snap = rb.snapshot();
        assert!(!snap.contains(&"aaaa".to_string()));
        assert!(snap.contains(&"d".to_string()));
    }
}
