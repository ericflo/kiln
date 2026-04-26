//! FIFO training job queue — accepts SFT and GRPO jobs, runs them sequentially.
//!
//! The queue ensures only one training job runs at a time, preventing GPU memory
//! conflicts between concurrent training jobs. Jobs are executed in submission order.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use kiln_model::lora_loader::LoraWeights;
use kiln_train::{GrpoRequest, SftRequest, TrainingState};
use kiln_train::trainer;
use serde::Serialize;

use crate::metrics::{TrainingMetricStatus, TrainingMetricType};
use crate::state::{AppState, ModelBackend, TrainingJobType};

/// JSON payload POSTed to the training-completion webhook.
///
/// The frontend contract documented in `TrainingConfig::webhook_url`
/// promises these field names — keep them stable for downstream
/// consumers.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TrainingCompletionEvent {
    pub job_id: String,
    pub job_type: &'static str,
    pub status: &'static str,
    pub adapter_name: String,
    pub adapter_path: Option<String>,
    pub error: Option<String>,
    pub timestamp: String,
}

impl TrainingCompletionEvent {
    pub fn job_type_str(job_type: TrainingJobType) -> &'static str {
        match job_type {
            TrainingJobType::Sft => "sft",
            TrainingJobType::Grpo => "grpo",
        }
    }
}

/// Fire-and-forget POST of `event` to `url`. Spawns a tokio task so the
/// caller (the training worker's blocking thread) is never blocked by
/// network I/O. Webhook failures are logged at WARN but never propagate
/// — a successful training job stays "completed" even if the
/// notification POST fails.
pub fn fire_completion_webhook(url: String, event: TrainingCompletionEvent) {
    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(error = %err, "failed to build webhook HTTP client");
                return;
            }
        };
        match client.post(&url).json(&event).send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    tracing::info!(
                        url = %url,
                        job_id = %event.job_id,
                        status = %status,
                        "training completion webhook delivered"
                    );
                } else {
                    tracing::warn!(
                        url = %url,
                        job_id = %event.job_id,
                        status = %status,
                        "training completion webhook returned non-2xx"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    url = %url,
                    job_id = %event.job_id,
                    error = %err,
                    "training completion webhook POST failed"
                );
            }
        }
    });
}

/// A pending training job in the queue.
pub enum QueuedJob {
    Sft(SftRequest),
    Grpo(GrpoRequest),
}

/// Entry in the training queue.
pub struct QueueEntry {
    pub job_id: String,
    pub job: QueuedJob,
}

/// Thread-safe training queue.
pub struct TrainingQueue {
    pub(crate) queue: VecDeque<QueueEntry>,
}

impl TrainingQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Add a job to the back of the queue.
    pub fn push(&mut self, entry: QueueEntry) {
        self.queue.push_back(entry);
    }

    /// Take the next job from the front of the queue.
    pub fn pop(&mut self) -> Option<QueueEntry> {
        self.queue.pop_front()
    }

    /// Number of jobs waiting in the queue (not including the currently running job).
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Remove a queued job by ID. Returns true if found and removed.
    pub fn remove(&mut self, job_id: &str) -> bool {
        let before = self.queue.len();
        self.queue.retain(|e| e.job_id != job_id);
        self.queue.len() < before
    }
}

pub type SharedTrainingQueue = Arc<std::sync::Mutex<TrainingQueue>>;

/// Shared shutdown flag — set to true when the server is shutting down.
/// Training queue rejects new jobs and the worker exits after the current job.
pub type ShutdownFlag = Arc<AtomicBool>;

/// Create a new shutdown flag (initially false).
pub fn new_shutdown_flag() -> ShutdownFlag {
    Arc::new(AtomicBool::new(false))
}

/// Create a new shared training queue.
pub fn new_shared_queue() -> SharedTrainingQueue {
    Arc::new(std::sync::Mutex::new(TrainingQueue::new()))
}

/// Spawn the background training worker that pulls jobs from the queue.
///
/// This runs as a tokio task that polls the queue every 500ms. When a job is
/// found, it executes it on a blocking thread (training is CPU/GPU-bound).
/// The worker exits cleanly when the shutdown flag is set, after finishing
/// any currently running job.
///
/// On every iteration the worker also runs a GC pass on `state.training_jobs`,
/// evicting terminal (`Completed` / `Failed`) entries whose `finished_at`
/// timestamp is older than `state.tracked_job_ttl`. This bounds the steady-
/// state size of the tracking map and works in concert with the
/// `max_tracked_jobs` cap to prevent memory growth from a flood of terminal
/// entries. See `gc_tracked_jobs` for the eviction predicate.
pub fn spawn_training_worker(state: AppState, shutdown: ShutdownFlag) {
    tokio::spawn(async move {
        loop {
            // Check shutdown flag before pulling the next job
            if shutdown.load(Ordering::Relaxed) {
                tracing::info!("training worker shutting down");
                break;
            }

            // GC stale terminal entries from the tracking map. Cheap when
            // the map is small; runs on every iteration so terminal
            // entries can never persist past TTL even on a quiescent
            // server.
            gc_tracked_jobs(&state);

            // Check for next job
            let entry = {
                let mut q = state.training_queue.lock().unwrap();
                q.pop()
            };

            if let Some(entry) = entry {
                // Execute the job on a blocking thread
                let state_clone = state.clone();
                let handle = tokio::task::spawn_blocking(move || {
                    execute_job(state_clone, entry);
                });
                // Wait for completion before pulling the next job
                if let Err(e) = handle.await {
                    tracing::error!("training worker task panicked: {e}");
                }
            } else {
                // No jobs — sleep briefly before checking again
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }
    });
}

/// Evict `Completed` / `Failed` entries from `state.training_jobs` whose
/// `finished_at` timestamp is older than `state.tracked_job_ttl`. Active
/// entries (`Queued` / `Running`) are never removed regardless of age.
///
/// Returns the number of entries removed.
///
/// Safe to call from any thread; takes a short write lock on
/// `training_jobs`. Called from the training worker loop on every
/// iteration and from tests directly.
pub fn gc_tracked_jobs(state: &AppState) -> usize {
    let ttl = state.tracked_job_ttl;
    let now = std::time::Instant::now();
    let mut jobs = state.training_jobs.write().unwrap();
    let before = jobs.len();
    jobs.retain(|_id, job| match job.state {
        TrainingState::Completed | TrainingState::Failed => match job.finished_at {
            // No timestamp recorded (legacy or in-flight transition) —
            // keep until the next pass observes a timestamp.
            None => true,
            Some(t) => now.saturating_duration_since(t) < ttl,
        },
        // Active jobs are never GC'd.
        TrainingState::Queued | TrainingState::Running => true,
    });
    let removed = before - jobs.len();
    if removed > 0 {
        tracing::debug!(removed, remaining = jobs.len(), "GC'd terminal training jobs past TTL");
    }
    removed
}

/// Execute a single training job (runs on a blocking thread).
fn execute_job(state: AppState, entry: QueueEntry) {
    let job_id = entry.job_id.clone();

    // Mark as running
    {
        let mut jobs = state.training_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            // Check if it was cancelled while queued
            if job.state == TrainingState::Failed {
                tracing::info!(job_id = %job_id, "skipping cancelled job");
                return;
            }
            job.state = TrainingState::Running;
        } else {
            tracing::warn!(job_id = %job_id, "job not found in tracking map, skipping");
            return;
        }
    }

    // Extract model weights reference
    let runner_arc = match state.backend.as_ref() {
        ModelBackend::Real { runner, .. } => runner.clone(),
        ModelBackend::Mock { .. } => {
            let mut jobs = state.training_jobs.write().unwrap();
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = TrainingState::Failed;
                job.finished_at = Some(std::time::Instant::now());
            }
            tracing::error!(job_id = %job_id, "training requires real model weights");
            return;
        }
    };

    let (weights_ref, num_layers) = {
        let guard = runner_arc.read().unwrap();
        (runner_arc.clone(), guard.config.num_layers)
    };

    // Get auto_load, adapter_name, and job_type from the job info
    let (auto_load, adapter_name, job_type) = {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs.get(&job_id).unwrap();
        (job.auto_load, job.adapter_name.clone(), job.job_type)
    };

    let metric_type = match job_type {
        TrainingJobType::Sft => TrainingMetricType::Sft,
        TrainingJobType::Grpo => TrainingMetricType::Grpo,
    };

    // Set up progress callback
    let training_jobs_cb = state.training_jobs.clone();
    let job_id_cb = job_id.clone();
    let progress_cb = Box::new(move |progress: trainer::TrainingProgress| {
        let mut jobs = training_jobs_cb.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id_cb) {
            job.progress = progress.progress;
            job.loss = Some(progress.loss);
            job.epoch = Some(progress.epoch as u32);
        }
    });

    // Apply server-level checkpoint_interval default if not set per-job
    let server_checkpoint_interval = state.checkpoint_interval;

    // Run the actual training under GPU write lock
    let result: Result<PathBuf, String> = match entry.job {
        QueuedJob::Sft(mut req) => {
            if req.config.checkpoint_interval.is_none() {
                req.config.checkpoint_interval = server_checkpoint_interval;
            }
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            trainer::sft_train(
                &req.examples,
                &req.config,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"))
        }
        QueuedJob::Grpo(mut req) => {
            if req.config.checkpoint_interval.is_none() {
                req.config.checkpoint_interval = server_checkpoint_interval;
            }
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            trainer::grpo_train(
                &req.groups,
                &req.config,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"))
        }
    };

    match result {
        Ok(adapter_path) => {
            let path_str = adapter_path.display().to_string();
            tracing::info!(job_id = %job_id, job_type = ?job_type, adapter = %adapter_name, path = %path_str, "training completed");

            {
                let mut jobs = state.training_jobs.write().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.state = TrainingState::Completed;
                    job.progress = 1.0;
                    job.adapter_path = Some(path_str.clone());
                    job.finished_at = Some(std::time::Instant::now());
                }
            }
            state.metrics.inc_training(metric_type, TrainingMetricStatus::Completed);

            if let Some(ref url) = state.training_webhook_url {
                let event = TrainingCompletionEvent {
                    job_id: job_id.clone(),
                    job_type: TrainingCompletionEvent::job_type_str(job_type),
                    status: "completed",
                    adapter_name: adapter_name.clone(),
                    adapter_path: Some(path_str.clone()),
                    error: None,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };
                fire_completion_webhook(url.clone(), event);
            }

            if auto_load {
                if let Err(e) = auto_load_adapter(
                    &weights_ref,
                    &state.active_adapter_name,
                    &adapter_path,
                    &adapter_name,
                    num_layers,
                ) {
                    tracing::error!(job_id = %job_id, "auto-load failed: {e}");
                } else {
                    tracing::info!(job_id = %job_id, "auto-loaded trained adapter");
                }
            }
        }
        Err(e) => {
            tracing::error!(job_id = %job_id, job_type = ?job_type, "training failed: {e}");
            let error_msg = e.clone();
            {
                let mut jobs = state.training_jobs.write().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.state = TrainingState::Failed;
                    job.finished_at = Some(std::time::Instant::now());
                }
            }
            state.metrics.inc_training(metric_type, TrainingMetricStatus::Failed);

            if let Some(ref url) = state.training_webhook_url {
                let event = TrainingCompletionEvent {
                    job_id: job_id.clone(),
                    job_type: TrainingCompletionEvent::job_type_str(job_type),
                    status: "failed",
                    adapter_name: adapter_name.clone(),
                    adapter_path: None,
                    error: Some(error_msg),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };
                fire_completion_webhook(url.clone(), event);
            }
        }
    }
}

/// Load a LoRA adapter using the two-phase RwLock pattern.
fn auto_load_adapter(
    runner: &Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    active_adapter_name: &Arc<std::sync::RwLock<Option<String>>>,
    adapter_path: &std::path::Path,
    adapter_name: &str,
    num_layers: usize,
) -> Result<(), String> {
    let device = {
        let guard = runner.read().unwrap();
        guard.weights.embed_tokens.device().clone()
    };

    let lora = LoraWeights::load(adapter_path, num_layers, &device)
        .map_err(|e| format!("failed to load adapter: {e}"))?;

    {
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
    }
    *active_adapter_name.write().unwrap() = Some(adapter_name.to_string());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_fifo_order() {
        let mut q = TrainingQueue::new();
        q.push(QueueEntry {
            job_id: "job-1".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });

        assert_eq!(q.len(), 3);
        assert_eq!(q.pop().unwrap().job_id, "job-1");
        assert_eq!(q.pop().unwrap().job_id, "job-2");
        assert_eq!(q.pop().unwrap().job_id, "job-3");
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_queue_remove() {
        let mut q = TrainingQueue::new();
        q.push(QueueEntry {
            job_id: "job-1".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
            }),
        });

        // Remove middle job
        assert!(q.remove("job-2"));
        assert_eq!(q.len(), 2);
        assert_eq!(q.pop().unwrap().job_id, "job-1");
        assert_eq!(q.pop().unwrap().job_id, "job-3");

        // Remove non-existent
        assert!(!q.remove("job-99"));
    }

    #[test]
    fn test_queue_empty() {
        let mut q = TrainingQueue::new();
        assert_eq!(q.len(), 0);
        assert!(q.pop().is_none());
        assert!(!q.remove("nonexistent"));
    }

    #[test]
    fn test_event_job_type_str() {
        assert_eq!(
            TrainingCompletionEvent::job_type_str(TrainingJobType::Sft),
            "sft"
        );
        assert_eq!(
            TrainingCompletionEvent::job_type_str(TrainingJobType::Grpo),
            "grpo"
        );
    }

    #[test]
    fn test_event_serializes_with_expected_field_names() {
        let event = TrainingCompletionEvent {
            job_id: "abc-123".into(),
            job_type: "sft",
            status: "completed",
            adapter_name: "my-adapter".into(),
            adapter_path: Some("/data/adapters/my-adapter".into()),
            error: None,
            timestamp: "2026-04-26T00:00:00+00:00".into(),
        };
        let v: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(v["job_id"], "abc-123");
        assert_eq!(v["job_type"], "sft");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["adapter_name"], "my-adapter");
        assert_eq!(v["adapter_path"], "/data/adapters/my-adapter");
        assert!(v["error"].is_null());
        assert_eq!(v["timestamp"], "2026-04-26T00:00:00+00:00");
    }

    /// End-to-end test: spin up a tiny axum mock server, fire a webhook
    /// at it, and assert that the captured POST body matches the
    /// documented payload shape.
    #[tokio::test]
    async fn test_fire_completion_webhook_posts_expected_payload() {
        use axum::extract::State;
        use axum::routing::post;
        use axum::Json;
        use std::sync::Arc as StdArc;
        use std::sync::Mutex as StdMutex;

        // Capture buffer shared between the handler and the assertions.
        let captured: StdArc<StdMutex<Vec<serde_json::Value>>> =
            StdArc::new(StdMutex::new(Vec::new()));

        async fn handler(
            State(captured): State<StdArc<StdMutex<Vec<serde_json::Value>>>>,
            Json(body): Json<serde_json::Value>,
        ) -> &'static str {
            captured.lock().unwrap().push(body);
            "ok"
        }

        let app = axum::Router::new()
            .route("/hook", post(handler))
            .with_state(captured.clone());

        // Bind to an ephemeral port so concurrent test runs don't collide.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let event = TrainingCompletionEvent {
            job_id: "test-job-001".into(),
            job_type: "grpo",
            status: "completed",
            adapter_name: "test-adapter".into(),
            adapter_path: Some("/tmp/adapters/test-adapter".into()),
            error: None,
            timestamp: "2026-04-26T01:23:45+00:00".into(),
        };

        let url = format!("http://{addr}/hook");
        fire_completion_webhook(url, event);

        // Poll the capture buffer for up to ~2s.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while captured.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let bodies = captured.lock().unwrap().clone();
        assert_eq!(bodies.len(), 1, "expected exactly one webhook POST");
        let body = &bodies[0];
        assert_eq!(body["job_id"], "test-job-001");
        assert_eq!(body["job_type"], "grpo");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["adapter_name"], "test-adapter");
        assert_eq!(body["adapter_path"], "/tmp/adapters/test-adapter");
        assert!(body["error"].is_null());
        assert_eq!(body["timestamp"], "2026-04-26T01:23:45+00:00");

        server.abort();
    }

    /// Failure event test: error string is propagated, adapter_path is null.
    #[tokio::test]
    async fn test_fire_completion_webhook_failure_event_shape() {
        use axum::extract::State;
        use axum::routing::post;
        use axum::Json;
        use std::sync::Arc as StdArc;
        use std::sync::Mutex as StdMutex;

        let captured: StdArc<StdMutex<Vec<serde_json::Value>>> =
            StdArc::new(StdMutex::new(Vec::new()));

        async fn handler(
            State(captured): State<StdArc<StdMutex<Vec<serde_json::Value>>>>,
            Json(body): Json<serde_json::Value>,
        ) -> &'static str {
            captured.lock().unwrap().push(body);
            "ok"
        }

        let app = axum::Router::new()
            .route("/hook", post(handler))
            .with_state(captured.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let event = TrainingCompletionEvent {
            job_id: "fail-job-001".into(),
            job_type: "sft",
            status: "failed",
            adapter_name: "broken-adapter".into(),
            adapter_path: None,
            error: Some("CUDA out of memory".into()),
            timestamp: "2026-04-26T01:23:45+00:00".into(),
        };
        let url = format!("http://{addr}/hook");
        fire_completion_webhook(url, event);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while captured.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let bodies = captured.lock().unwrap().clone();
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0];
        assert_eq!(body["status"], "failed");
        assert!(body["adapter_path"].is_null());
        assert_eq!(body["error"], "CUDA out of memory");

        server.abort();
    }

    /// Webhook errors must NOT panic or propagate — verified by firing
    /// at an unreachable address and ensuring the spawned task completes
    /// without taking the test process down.
    #[tokio::test]
    async fn test_fire_completion_webhook_swallows_errors() {
        let event = TrainingCompletionEvent {
            job_id: "x".into(),
            job_type: "sft",
            status: "completed",
            adapter_name: "x".into(),
            adapter_path: None,
            error: None,
            timestamp: "2026-04-26T00:00:00+00:00".into(),
        };
        // 127.0.0.1:1 is reliably not listening — connection should fail
        // fast within the 5s client timeout, and the failure must be
        // swallowed (logged, not propagated).
        fire_completion_webhook("http://127.0.0.1:1/never".into(), event);
        // Give the spawned task a moment so we're confident it ran and
        // completed without panicking.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
