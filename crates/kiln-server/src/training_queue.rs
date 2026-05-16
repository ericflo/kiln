//! FIFO training job queue — accepts SFT and GRPO jobs, runs them sequentially.
//!
//! The queue ensures only one training job runs at a time, preventing GPU memory
//! conflicts between concurrent training jobs. Jobs are executed in submission order.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use kiln_core::env_flag::env_tristate;
use kiln_model::lora_loader::LoraWeights;
use kiln_train::trainer;
use kiln_train::{self, GrpoRequest, OpdRequest, SftRequest, TrainingState};
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
            TrainingJobType::Opd => "opd",
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
        // Defensive: an operator-set webhook URL that 302s into internal infra
        // (e.g. 169.254.169.254 IMDS) must NOT be auto-followed. See
        // docs/audits/security-audit-v0.1.md §7.
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
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
    /// On-Policy Distillation request. The runtime currently exercises
    /// the full HTTP → tokenize → kernel-loss path against a fixture
    /// teacher to validate the wiring; sampling + optimizer step + hot-
    /// swap land in the next commit (the §3.1 pseudocode body shares
    /// most of its mechanics with the GRPO loop and gets refactored
    /// alongside).
    Opd(OpdRequest),
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
        tracing::debug!(
            removed,
            remaining = jobs.len(),
            "GC'd terminal training jobs past TTL"
        );
    }
    removed
}

/// Dispatch one SFT job to either the candle trainer, the CUDA-native trainer,
/// or the vk-native trainer.
///
/// The candle path takes a `replay_ctx` (request_body + lineage
/// tracking); native paths don't yet plumb replay so they drop the context.
/// When the binary is built without an explicitly requested backend feature,
/// the native flag falls through to the candle path with a warning. Vulkan SFT
/// also auto-engages on the Vulkan backend when the flag is unset.
#[allow(clippy::too_many_arguments)]
fn run_sft(
    cuda_native: bool,
    vk_native: bool,
    req: &SftRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    replay_ctx: trainer::ReplayContext,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    #[cfg(not(feature = "vulkan"))]
    let _ = job_id;

    if cuda_native {
        #[cfg(feature = "cuda")]
        {
            tracing::info!(
                job_id = %job_id,
                "KILN_CUDA_NATIVE_TRAINING=1 - routing to cuda_native_sft_train"
            );
            return kiln_train::cuda_train::cuda_native_sft_train(
                &req.examples,
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"));
        }
        #[cfg(not(feature = "cuda"))]
        {
            tracing::warn!(
                job_id = %job_id,
                "KILN_CUDA_NATIVE_TRAINING=1 set but kiln-server was built without \
                 --features cuda - falling back to candle SFT trainer"
            );
        }
    }
    if vk_native {
        #[cfg(feature = "vulkan")]
        {
            tracing::info!(
                job_id = %job_id,
                "KILN_VK_NATIVE_TRAINING=1 — routing to vk_native_sft_train"
            );
            return kiln_train::vk_train::vk_native_sft_train(
                &req.examples,
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"));
        }
        #[cfg(not(feature = "vulkan"))]
        {
            tracing::warn!(
                job_id = %job_id,
                "KILN_VK_NATIVE_TRAINING=1 set but kiln-server was built without \
                 --features vulkan — falling back to candle SFT trainer"
            );
        }
    }
    trainer::sft_train(
        &req.examples,
        &req.config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(replay_ctx),
    )
    .map_err(|e| format!("{e:#}"))
}

fn native_training_env_enabled(name: &str) -> bool {
    env_tristate(name).unwrap_or(false)
}

fn vk_native_sft_enabled(backend_name: &str) -> bool {
    match env_tristate("KILN_VK_NATIVE_TRAINING") {
        Some(enabled) => enabled,
        None => {
            #[cfg(feature = "vulkan")]
            {
                backend_name == "vulkan"
            }
            #[cfg(not(feature = "vulkan"))]
            {
                let _ = backend_name;
                false
            }
        }
    }
}

fn vk_native_grpo_enabled(backend_name: &str) -> bool {
    match env_tristate("KILN_VK_NATIVE_GRPO") {
        Some(enabled) => enabled,
        None => vk_native_sft_enabled(backend_name),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_grpo(
    cuda_native: bool,
    vk_native: bool,
    req: &GrpoRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    replay_ctx: trainer::ReplayContext,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    #[cfg(not(feature = "vulkan"))]
    let _ = job_id;

    if cuda_native {
        return Err(
            "KILN_CUDA_NATIVE_TRAINING=1 does not yet support GRPO - unset it for GRPO jobs"
                .to_string(),
        );
    }
    if let Some(dataset_path) = req.dataset_path.as_deref() {
        if dataset_path.trim().is_empty() {
            return Err("GRPO dataset_path streaming requires a non-empty path".to_string());
        }
        if !req.groups.is_empty() {
            return Err(
                "GRPO request must use either groups or dataset_path, not both".to_string(),
            );
        }
        if vk_native {
            #[cfg(feature = "vulkan")]
            {
                tracing::info!(
                    job_id = %job_id,
                    dataset_path,
                    "routing streamed GRPO dataset to vk_native_grpo_train_jsonl"
                );
                return kiln_train::vk_train::vk_native_grpo_train_jsonl(
                    std::path::Path::new(dataset_path),
                    &req.config,
                    model_config,
                    weights,
                    tokenizer,
                    adapter_dir,
                    adapter_name,
                    Some(progress_cb),
                )
                .map_err(|e| format!("{e:#}"));
            }
            #[cfg(not(feature = "vulkan"))]
            {
                return Err(
                    "GRPO dataset_path streaming requested but kiln-server was built without \
                     --features vulkan"
                        .to_string(),
                );
            }
        }
        {
            tracing::info!(
                job_id = %job_id,
                dataset_path,
                "routing streamed GRPO dataset to generic trainer"
            );
            return trainer::grpo_train_jsonl(
                std::path::Path::new(dataset_path),
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
                Some(replay_ctx),
            )
            .map_err(|e| format!("{e:#}"));
        }
    }
    if vk_native {
        #[cfg(feature = "vulkan")]
        {
            tracing::info!(
                job_id = %job_id,
                "routing GRPO to vk_native_grpo_train"
            );
            return kiln_train::vk_train::vk_native_grpo_train(
                &req.groups,
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
            )
            .map_err(|e| format!("{e:#}"));
        }
        #[cfg(not(feature = "vulkan"))]
        {
            return Err(
                "Vulkan-native GRPO requested but kiln-server was built without \
                 --features vulkan"
                    .to_string(),
            );
        }
    }
    trainer::grpo_train(
        &req.groups,
        &req.config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(replay_ctx),
    )
    .map_err(|e| format!("{e:#}"))
}

/// Run one OPD training request.
///
/// **Milestone 4 (this commit) scope.** The plumbing — HTTP → queue →
/// blocking worker → job tracking → metrics → webhook → auto-load → post-
/// eval — is wired identically to SFT/GRPO. The runtime body itself runs
/// the §3.1 pseudocode against a fixture teacher (no real teacher
/// resolution yet) so the entire host code path is exercised
/// end-to-end before the GPU model integration lands. Outcome:
///
/// 1. Validate the prompt set is non-empty.
/// 2. Verify the `OpdRequest`'s loss / top_k / Stable-OPD knobs
///    deserialise cleanly (already enforced via serde at the endpoint
///    boundary).
/// 3. Persist a stub PEFT adapter directory so `auto_load` and
///    `post_eval` callers see a real path. The adapter weights match
///    the base model exactly (no update yet) — the §3.1 loss + IS
///    advantage path is the next commit.
///
/// Returns an explicit error when the GPU runtime path *would* run but
/// the model is mocked, matching SFT/GRPO. The reason for the explicit
/// runtime stub: the §3.1 trainer body shares almost all of its
/// machinery (segment-checkpointed forward, LoRA Vars, AdamW step,
/// hot-swap) with `grpo_train`, and the refactor that factors those
/// pieces out so OPD can call them is its own diff; doing it here would
/// produce a much larger PR than this milestone wants.
#[allow(clippy::too_many_arguments)]
fn run_opd(
    req: &OpdRequest,
    _model_config: &kiln_core::config::ModelConfig,
    _weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    if req.prompts.is_empty() && req.dataset_path.is_none() {
        return Err("OPD request must include at least one prompt or a dataset_path".into());
    }

    tracing::info!(
        job_id = %job_id,
        teacher = %req.teacher,
        loss = ?req.config.loss,
        top_k = req.config.top_k,
        samples_per_prompt = req.config.samples_per_prompt,
        num_prompts = req.prompts.len(),
        "OPD training started (milestone-4 stub: validates plumbing, no optimizer step yet)"
    );

    // Validate prompts tokenize cleanly — same shape as
    // `tokenize_grpo_group` so the request is rejected early if the
    // tokenizer can't handle it. We don't run an actual GPU forward
    // here; the kernel + per-position KL path was already proven in
    // `kiln-train::opd::tests`.
    let _ = tokenizer; // tokenizer used by next-commit runtime body.
    for (i, prompt) in req.prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(format!("OPD prompt {i} has no messages"));
        }
    }

    // Stub-write a PEFT adapter directory matching SFT/GRPO output
    // layout. This lets `auto_load`, `post_eval`, the dashboard, and
    // reproducibility-receipt code (§8.11) all see a real path. The
    // adapter is a no-op LoRA (zeros) — the trainer body that produces
    // real weight deltas lands in the next commit.
    let output_dir = adapter_dir.join(adapter_name);
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create OPD adapter dir {}: {e}", output_dir.display()))?;
    let stub_marker = output_dir.join("KILN_OPD_STUB.txt");
    std::fs::write(
        &stub_marker,
        format!(
            "kiln OPD stub adapter for job {job_id}.\n\
             This adapter exists so the HTTP / queue / hot-swap / auto-load / post-eval\n\
             plumbing has a path to point at. The §3.1 OPD trainer body lands in the\n\
             next commit on this branch (see docs/plans/grand-plan-for-extraordinarily-\n\
             great-on-policy-distillation-for-everyone.md §3.1).\n\
             Teacher: {teacher}\n\
             Loss: {loss:?}\n\
             top_k: {top_k}\n",
            teacher = req.teacher,
            loss = req.config.loss,
            top_k = req.config.top_k,
        ),
    )
    .map_err(|e| format!("failed to write OPD stub marker: {e}"))?;

    // §8.11 reproducibility receipt — every adapter ships with one.
    // Even the milestone-4 stub gets a receipt so the verify tooling
    // and dashboard receipt-display path work end-to-end now.
    let seed = req.config.seed.unwrap_or(0);
    let hyperparameters = serde_json::to_value(&req.config)
        .unwrap_or_else(|_| serde_json::json!({"error": "failed to serialize OpdConfig"}));
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "opd", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: req.teacher.clone(),
            model_id: req.teacher.clone(),
            model_version_hash: None,
            snapshot_url: None,
        })
        .with_hyperparameters(hyperparameters);
    if let Err(e) = receipt.write_to_adapter_dir(&output_dir) {
        tracing::warn!(job_id = %job_id, "failed to write OPD receipt: {e}");
    }

    // Emit progress callbacks so the dashboard / progress bar sees the
    // job as "completed" rather than "stuck at 0%".
    progress_cb(trainer::TrainingProgress {
        epoch: 1,
        total_epochs: 1,
        step: 1,
        total_steps: 1,
        loss: 0.0,
        progress: 1.0,
    });

    tracing::warn!(
        job_id = %job_id,
        path = %output_dir.display(),
        "OPD adapter is a milestone-4 stub (no parameter update); see KILN_OPD_STUB.txt for details"
    );

    Ok(output_dir)
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
        TrainingJobType::Opd => TrainingMetricType::Opd,
    };

    // Set up progress callback. Records the latest scalar progress AND
    // appends a sample to `loss_history` so the live UI chart has a
    // bounded series to draw. Sampling is downsampled in-place at
    // `TRAINING_LOSS_HISTORY_CAP` to bound memory on long runs.
    let training_jobs_cb = state.training_jobs.clone();
    let job_id_cb = job_id.clone();
    let started_instant = std::time::Instant::now();
    let progress_cb = Box::new(move |progress: trainer::TrainingProgress| {
        let mut jobs = training_jobs_cb.write().unwrap();
        if let Some(job) = jobs.get_mut(&job_id_cb) {
            job.progress = progress.progress;
            job.loss = Some(progress.loss);
            job.epoch = Some(progress.epoch as u32);
            crate::state::push_loss_sample(
                &mut job.loss_history,
                crate::state::TrainingLossSample {
                    epoch: progress.epoch as u32,
                    progress: progress.progress,
                    loss: progress.loss,
                    elapsed_secs: started_instant.elapsed().as_secs_f64(),
                },
            );
        }
    });

    // Apply server-level checkpoint_interval default if not set per-job
    let server_checkpoint_interval = state.checkpoint_interval;

    let base_model = trainer::default_base_model(&state.model_config);

    // Run the actual training under GPU write lock
    // Capture the post-eval hook before the job request is consumed by the
    // trainer.
    let post_eval: Option<kiln_eval::PostEvalConfig> = match &entry.job {
        QueuedJob::Sft(req) => req.post_eval.clone(),
        QueuedJob::Grpo(req) => req.post_eval.clone(),
        QueuedJob::Opd(req) => req.post_eval.clone(),
    };

    let result: std::result::Result<PathBuf, String> = match entry.job {
        QueuedJob::Sft(mut req) => {
            if req.config.checkpoint_interval.is_none() {
                req.config.checkpoint_interval = server_checkpoint_interval;
            }
            let request_body = serde_json::to_value(&req)
                .unwrap_or_else(|_| serde_json::json!({"error": "failed to serialize SftRequest"}));
            let _replay_ctx = trainer::ReplayContext {
                request_id: job_id.clone(),
                kind: kiln_train::ReplayKind::Sft,
                request_body,
                base_model: base_model.clone(),
            };
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            let backend_name = guard.backend_name();
            // Native CUDA/Vulkan training keeps forward intermediates and
            // grads in backend memory. Replay context is candle-trainer-specific,
            // so native paths drop it until that integration is added.
            let cuda_native = native_training_env_enabled("KILN_CUDA_NATIVE_TRAINING");
            let vk_native = vk_native_sft_enabled(backend_name);
            run_sft(
                cuda_native,
                vk_native,
                &req,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                progress_cb,
                _replay_ctx,
                &job_id,
            )
        }
        QueuedJob::Grpo(mut req) => {
            if req.config.checkpoint_interval.is_none() {
                req.config.checkpoint_interval = server_checkpoint_interval;
            }
            let request_body = serde_json::to_value(&req).unwrap_or_else(
                |_| serde_json::json!({"error": "failed to serialize GrpoRequest"}),
            );
            let replay_ctx = trainer::ReplayContext {
                request_id: job_id.clone(),
                kind: kiln_train::ReplayKind::Grpo,
                request_body,
                base_model: base_model.clone(),
            };
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            let backend_name = guard.backend_name();
            let cuda_native = native_training_env_enabled("KILN_CUDA_NATIVE_TRAINING");
            let vk_native = vk_native_grpo_enabled(backend_name);
            run_grpo(
                cuda_native,
                vk_native,
                &req,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                progress_cb,
                replay_ctx,
                &job_id,
            )
        }
        QueuedJob::Opd(mut req) => {
            if req.config.checkpoint_interval.is_none() {
                req.config.checkpoint_interval = server_checkpoint_interval;
            }
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            run_opd(
                &req,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                progress_cb,
                &job_id,
            )
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
            state
                .metrics
                .inc_training(metric_type, TrainingMetricStatus::Completed);

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

            // Post-training auto-eval: enqueue an eval job against the
            // produced adapter so dashboards land directly on the eval
            // result. Failures here are warnings — we still consider the
            // training itself successful.
            if let Some(cfg) = post_eval.as_ref() {
                if let Err(e) = enqueue_post_training_eval(&state, &job_id, &adapter_name, cfg) {
                    tracing::warn!(job_id = %job_id, error = %e, "post-training eval enqueue failed");
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
            state
                .metrics
                .inc_training(metric_type, TrainingMetricStatus::Failed);

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

/// Enqueue a post-training eval against `adapter_name`. When
/// `cfg.include_baseline` is set, also enqueues a baseline run against the
/// base model so a side-by-side delta is computable. Returns Err only when
/// the eval queue is at capacity or no suite registry is configured;
/// individual eval failures are reported via the eval-job tracking map.
pub fn enqueue_post_training_eval(
    state: &AppState,
    training_job_id: &str,
    adapter_name: &str,
    cfg: &kiln_eval::PostEvalConfig,
) -> Result<(), String> {
    if state.suite_registry.is_none() {
        return Err("server has no eval suite registry".to_string());
    }
    let qlen = state.eval_queue.lock().unwrap().len();
    if qlen >= state.max_queued_eval_jobs {
        return Err(format!(
            "eval queue at capacity ({})",
            state.max_queued_eval_jobs
        ));
    }
    let push = |adapter: Option<String>| -> String {
        state.enqueue_eval(
            cfg.suite.clone(),
            vec![adapter.clone()],
            crate::eval::queue::EvalSubmissionKind::PostTraining,
            Some(training_job_id.to_string()),
            crate::eval::queue::QueuedEvalJob::Registered {
                suite_name: cfg.suite.clone(),
                adapter,
                generation_override: cfg.generation.clone(),
            },
        )
    };

    let mut linked_ids: Vec<String> = Vec::new();
    if cfg.include_baseline {
        linked_ids.push(push(None));
    }
    linked_ids.push(push(Some(adapter_name.to_string())));
    // Back-link the eval job IDs onto the training job so dashboards can
    // find them quickly.
    {
        let mut jobs = state.training_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(training_job_id) {
            job.linked_eval_job_ids = linked_ids;
        }
    }
    Ok(())
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
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn native_training_env_enabled_treats_empty_and_zero_as_disabled() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const VAR: &str = "KILN_TEST_NATIVE_TRAINING_FLAG";

        unsafe {
            std::env::remove_var(VAR);
        }
        assert!(!native_training_env_enabled(VAR));

        unsafe {
            std::env::set_var(VAR, "");
        }
        assert!(!native_training_env_enabled(VAR));

        unsafe {
            std::env::set_var(VAR, "0");
        }
        assert!(!native_training_env_enabled(VAR));

        unsafe {
            std::env::set_var(VAR, "1");
        }
        assert!(native_training_env_enabled(VAR));

        unsafe {
            std::env::remove_var(VAR);
        }
    }

    #[test]
    fn vk_native_grpo_defaults_to_vulkan_backend_and_honors_override() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        unsafe {
            std::env::remove_var("KILN_VK_NATIVE_TRAINING");
            std::env::remove_var("KILN_VK_NATIVE_GRPO");
        }

        #[cfg(feature = "vulkan")]
        assert!(vk_native_grpo_enabled("vulkan"));
        assert!(!vk_native_grpo_enabled("cpu"));

        unsafe {
            std::env::set_var("KILN_VK_NATIVE_GRPO", "0");
        }
        assert!(!vk_native_grpo_enabled("vulkan"));

        unsafe {
            std::env::set_var("KILN_VK_NATIVE_GRPO", "1");
        }
        assert!(vk_native_grpo_enabled("cpu"));

        unsafe {
            std::env::remove_var("KILN_VK_NATIVE_GRPO");
        }
    }

    #[test]
    fn test_queue_fifo_order() {
        let mut q = TrainingQueue::new();
        q.push(QueueEntry {
            job_id: "job-1".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
                post_eval: None,
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
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            job: QueuedJob::Sft(SftRequest {
                examples: vec![],
                config: Default::default(),
                post_eval: None,
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

    fn mk_post_eval_state() -> AppState {
        let config = kiln_core::config::ModelConfig::qwen3_5_4b();
        let sched_config = kiln_scheduler::SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        };
        let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
        let engine = kiln_model::engine::MockEngine::new(config.clone());
        let tokenizer = {
            let json = br#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "b": 1},
                    "merges": []
                }
            }"#;
            kiln_core::tokenizer::KilnTokenizer::from_bytes(json).unwrap()
        };
        let mut state = AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            60,
            "kiln-test".to_string(),
        );
        let dir = tempfile::tempdir().unwrap();
        // Stash the registry on the state with a suite saved upfront so the
        // post-eval enqueue succeeds.
        let reg = crate::eval::SuiteRegistry::new(dir.path().to_path_buf());
        let suite = kiln_eval::EvalSuite {
            name: "smoke".into(),
            description: None,
            default_scorer: kiln_eval::scorers::Scorer::ExactMatch {
                case_sensitive: false,
                strip_whitespace: true,
            },
            generation: kiln_eval::EvalGenerationParams::default(),
            system_prompt: None,
            examples: vec![kiln_eval::EvalExample {
                id: Some("e1".into()),
                messages: vec![kiln_eval::EvalChatMessage::new("user", "x")],
                target: Some("x".into()),
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        };
        reg.save(&suite, false).unwrap();
        state.suite_registry = Some(Arc::new(reg));
        // The tempdir would otherwise drop and remove the suite — leak it.
        std::mem::forget(dir);
        state
    }

    #[test]
    fn enqueue_post_training_eval_adds_one_job_when_no_baseline() {
        let state = mk_post_eval_state();
        let cfg = kiln_eval::PostEvalConfig {
            suite: "smoke".into(),
            generation: None,
            min_accuracy: None,
            include_baseline: false,
        };
        enqueue_post_training_eval(&state, "train-job-1", "trained-adapter", &cfg).unwrap();
        assert_eq!(state.eval_queue.lock().unwrap().len(), 1);
        assert_eq!(state.eval_jobs.read().unwrap().len(), 1);
    }

    #[test]
    fn enqueue_post_training_eval_adds_two_jobs_with_baseline() {
        let state = mk_post_eval_state();
        let cfg = kiln_eval::PostEvalConfig {
            suite: "smoke".into(),
            generation: None,
            min_accuracy: None,
            include_baseline: true,
        };
        enqueue_post_training_eval(&state, "train-job-2", "trained-adapter", &cfg).unwrap();
        assert_eq!(state.eval_queue.lock().unwrap().len(), 2);
        assert_eq!(state.eval_jobs.read().unwrap().len(), 2);
    }

    #[test]
    fn enqueue_post_training_eval_errors_when_no_registry() {
        let mut state = mk_post_eval_state();
        state.suite_registry = None;
        let cfg = kiln_eval::PostEvalConfig {
            suite: "smoke".into(),
            generation: None,
            min_accuracy: None,
            include_baseline: false,
        };
        let err = enqueue_post_training_eval(&state, "j", "a", &cfg).unwrap_err();
        assert!(err.contains("no eval suite registry"));
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
        use axum::Json;
        use axum::extract::State;
        use axum::routing::post;
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
        use axum::Json;
        use axum::extract::State;
        use axum::routing::post;
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

    /// Redirects must NOT be followed by the webhook client. An
    /// operator-set webhook URL that 302s to a secondary endpoint
    /// (e.g. internal IMDS at `http://169.254.169.254/`) is a
    /// belt-and-suspenders SSRF concern from `security-audit-v0.1.md` §7.
    /// We verify the redirect-target endpoint is never POSTed to.
    #[tokio::test]
    async fn test_fire_completion_webhook_does_not_follow_redirects() {
        use axum::Json;
        use axum::extract::State;
        use axum::http::{StatusCode, header};
        use axum::response::IntoResponse;
        use axum::routing::post;
        use std::sync::Arc as StdArc;
        use std::sync::Mutex as StdMutex;

        // Capture buffer for the *redirect target* — must remain empty.
        let captured_final: StdArc<StdMutex<Vec<serde_json::Value>>> =
            StdArc::new(StdMutex::new(Vec::new()));

        async fn redirect_handler() -> impl IntoResponse {
            (StatusCode::FOUND, [(header::LOCATION, "/hook-final")], "")
        }

        async fn final_handler(
            State(captured): State<StdArc<StdMutex<Vec<serde_json::Value>>>>,
            Json(body): Json<serde_json::Value>,
        ) -> &'static str {
            captured.lock().unwrap().push(body);
            "ok"
        }

        let app = axum::Router::new()
            .route("/hook-redirect", post(redirect_handler))
            .route("/hook-final", post(final_handler))
            .with_state(captured_final.clone());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let event = TrainingCompletionEvent {
            job_id: "redirect-test-001".into(),
            job_type: "sft",
            status: "completed",
            adapter_name: "redirect-test-adapter".into(),
            adapter_path: Some("/tmp/adapters/redirect-test-adapter".into()),
            error: None,
            timestamp: "2026-04-26T01:23:45+00:00".into(),
        };

        let url = format!("http://{addr}/hook-redirect");
        fire_completion_webhook(url, event);

        // Give the spawned task plenty of time to (incorrectly) follow
        // the redirect if our `redirect::Policy::none()` were missing.
        // The original POST resolves immediately to a 302; if redirects
        // were on, the secondary POST would land within ~tens of ms.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let bodies = captured_final.lock().unwrap().clone();
        assert!(
            bodies.is_empty(),
            "/hook-final must NOT be POSTed to — redirects must be disabled on the webhook client. Got bodies: {:?}",
            bodies
        );

        server.abort();
    }
}
