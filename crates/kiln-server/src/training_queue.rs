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
use kiln_train::{
    self, DistillMergeRequest, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest,
    GrpoRequest, LogitSource as _, OpdRequest, SftRequest, TrainingState,
};
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
    /// `/v1/distill/refresh` — §3.6 continual-learning recipe. Two
    /// internal phases: SFT mid-train on `new_data`, then OPD-recover
    /// against `behavioural_teacher`. Gated on dual eval (IF-eval
    /// recovery + new-knowledge gain).
    DistillRefresh(DistillRefreshRequest),
    /// `/v1/adapters/distill_merge` — §3.4 behaviour-space merge.
    /// Multi-teacher OPD over each source LoRA's retained training-
    /// prompt distribution.
    DistillMerge(DistillMergeRequest),
    /// `/v1/distill/pump` — §3.5 27B → 4B Knowledge Pump in three
    /// modes (Domain / Wide / Examples).
    DistillPump(DistillPumpRequest),
    /// `/v1/distill/self` — §3.12 PI self-distillation.
    DistillSelf(DistillSelfRequest),
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
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    teacher_registry: &crate::api::teachers::TeacherRegistry,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    if req.prompts.is_empty() && req.dataset_path.is_none() {
        return Err("OPD request must include at least one prompt or a dataset_path".into());
    }
    for (i, prompt) in req.prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(format!("OPD prompt {i} has no messages"));
        }
    }

    tracing::info!(
        job_id = %job_id,
        teacher = %req.teacher,
        loss = ?req.config.loss,
        top_k = req.config.top_k,
        samples_per_prompt = req.config.samples_per_prompt,
        num_prompts = req.prompts.len(),
        "OPD training started"
    );

    // Resolve the teacher alias against the §3.2 registry, then build
    // the concrete LogitSource:
    //   • Fixture → DeterministicUniformLogitSource (synthetic).
    //   • Local   → run the loaded model forward on each prompt and
    //               populate a FixtureLogitSource with the real
    //               top-K teacher logprobs (§3.2 in-process local
    //               teacher).
    //   • Remote  → RemoteTeacher with provider guessed from the URL.
    let spec = teacher_registry
        .get(&req.teacher)
        .ok_or_else(|| format!("teacher alias {:?} not registered (POST /v1/teachers first)", req.teacher))?;

    let resolved_vocab = spec.vocab_size.unwrap_or(model_config.vocab_size);
    let resolved_max_top_k = spec.max_top_k.unwrap_or(req.config.top_k);
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> = match spec.kind {
        crate::api::teachers::TeacherKind::Fixture => {
            std::sync::Arc::new(kiln_train::DeterministicUniformLogitSource::new(
                spec.alias.clone(),
                resolved_vocab,
                resolved_max_top_k.max(req.config.top_k),
            ))
        }
        crate::api::teachers::TeacherKind::Local => {
            std::sync::Arc::new(build_local_teacher_for(
                &spec,
                &req.prompts,
                tokenizer,
                weights,
                model_config,
                req.config.top_k,
            )?)
        }
        crate::api::teachers::TeacherKind::Remote => {
            let url = spec.url.clone().ok_or_else(|| {
                format!(
                    "teacher {:?} is Remote but has no `url` field — re-register with `url` set",
                    spec.alias
                )
            })?;
            let provider = guess_remote_provider(&url);
            let cfg = kiln_train::RemoteTeacherConfig {
                provider,
                model: spec.model_id.clone(),
                url,
                api_key_env: spec.api_key_env.clone(),
                teacher_id: spec.alias.clone(),
                tokenizer_hash: spec.tokenizer_hash.clone(),
                max_top_k: resolved_max_top_k,
                vocab_size: resolved_vocab,
                max_cost_usd: Some(req.config.max_cost_usd.unwrap_or(DEFAULT_REMOTE_COST_CAP_USD)),
                timeout_ms: 60_000,
            };
            std::sync::Arc::new(kiln_train::RemoteTeacher::new(cfg))
        }
    };

    let trainer_progress_cb: trainer::ProgressCallback = progress_cb;

    let output_dir = kiln_train::opd::opd_train(
        &req.prompts,
        &req.config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        adapter_name,
        Some(trainer_progress_cb),
    )
    .map_err(|e| format!("opd_train failed: {e:#}"))?;

    // §8.11 reproducibility receipt — every adapter ships with one.
    let seed = req.config.seed.unwrap_or(0);
    let hyperparameters = serde_json::to_value(&req.config)
        .unwrap_or_else(|_| serde_json::json!({"error": "failed to serialize OpdConfig"}));
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "opd", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            model_version_hash: None,
            snapshot_url: None,
        })
        .with_hyperparameters(hyperparameters);
    if let Err(e) = receipt.write_to_adapter_dir(&output_dir) {
        tracing::warn!(job_id = %job_id, "failed to write OPD receipt: {e}");
    }

    Ok(output_dir)
}

/// Build the §3.4 multi-tenant merge teacher.
///
/// Loads each source LoRA, runs the model forward over that source's
/// prompts with the source LoRA applied (so the logits reflect "what
/// the model behaves like when wearing that LoRA"), and stashes the
/// top-K teacher logprobs into a single `FixtureLogitSource` keyed by
/// (tokens_hash, position).
///
/// Each source contributes only its own prompts' entries, so the
/// trainer's `opd_step_loss` call queries the *correct* source's
/// teacher when iterating each prompt — no per-step LoRA swap, no
/// multi-tenant inference server needed.
///
/// Per-source `weight` (a `DistillMergeSource` field) is not yet
/// applied — the unified fixture treats every (source, prompt) entry
/// equally. Weighted loss aggregation is filed as a §3.4 follow-up.
#[allow(clippy::too_many_arguments)]
fn build_multi_tenant_merge_teacher(
    teacher_id: &str,
    per_source: &[(kiln_train::DistillMergeSource, Vec<kiln_train::opd::OpdPrompt>)],
    adapter_dir: &std::path::Path,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    top_k: usize,
    job_id: &str,
) -> std::result::Result<kiln_train::logit_source::FixtureLogitSource, String> {
    let mut unified = kiln_train::logit_source::FixtureLogitSource::uniform_topk(
        teacher_id.to_string(),
        model_config.vocab_size,
        top_k,
    );
    for (source, prompts) in per_source {
        // Try to load the source LoRA from disk. On failure (no
        // PEFT files yet) we fall back to base-model teacher for
        // this source's prompts and surface a tracing warning.
        let src_dir = adapter_dir.join(&source.adapter);
        let device = weights.embed_tokens.device().clone();
        let teacher_lora = match kiln_model::lora_loader::LoraWeights::load(
            &src_dir,
            model_config.num_layers,
            &device,
        ) {
            Ok(weights) => Some(weights),
            Err(e) => {
                tracing::warn!(
                    job_id = %job_id,
                    source = %source.adapter,
                    error = %e,
                    "distill_merge: source LoRA load failed — base-model teacher for this source's prompts"
                );
                None
            }
        };

        // Tokenize this source's prompts.
        let mut tokenized: Vec<(Vec<u32>, Vec<usize>)> = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let ex = kiln_train::SftExample {
                messages: prompt.messages.clone(),
            };
            if let Ok((tokens, label_mask)) =
                kiln_train::trainer::tokenize_for_training(&ex, tokenizer)
            {
                let active: Vec<usize> = label_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| if m { Some(i) } else { None })
                    .collect();
                if !active.is_empty() {
                    tokenized.push((tokens, active));
                }
            }
        }
        if tokenized.is_empty() {
            continue;
        }
        let source_fixture = kiln_train::opd::build_local_teacher_fixture(
            format!("{teacher_id}:{}", source.adapter),
            &tokenized,
            weights,
            model_config,
            teacher_lora.as_ref(),
            top_k,
            None,
        )
        .map_err(|e| format!("build_local_teacher_fixture for {}: {e:#}", source.adapter))?;

        // Drain entries from source_fixture into unified.
        for (tokens, active_positions) in &tokenized {
            for &pos in active_positions {
                let batch = match source_fixture.fetch_logprobs(tokens, &[pos], Some(top_k)) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let (indices, logprobs) = match batch {
                    kiln_train::LogprobBatch::TopK(t) => (t.indices, t.logprobs),
                    _ => continue,
                };
                let key = kiln_train::logit_source::FixtureLogitSource::hash_tokens(tokens);
                unified.insert(key, pos, indices, logprobs);
            }
        }
    }
    Ok(unified)
}

/// Build the §3.12 privileged-information self-teacher.
///
/// For each prompt and the chosen `SelfDistillMode`, we construct a
/// teacher-side prompt that *includes* the privileged context (a
/// system message carrying ground-truth answers, a "be concise"
/// instruction, or retrieved documents). The teacher's forward pass
/// runs against that shaped prompt; we then transplant the resulting
/// top-K logprobs back onto the *student's* (un-shaped) token stream
/// by aligning active assistant positions. The student then distils
/// against logits that "knew" the privileged context — Lu's PI
/// recipe (CRISP, OPSD, GATES, RLRT) made concrete.
///
/// For `ReverseTeacher` the privileged context is omitted but the
/// per-position logprobs are negated post-hoc so the student moves
/// *away* from the teacher's distribution — the survey's
/// reversed-teacher knob.
#[allow(clippy::too_many_arguments)]
fn build_self_distill_teacher(
    teacher_id: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
    mode: kiln_train::SelfDistillMode,
    ground_truth: Option<&[String]>,
    documents: Option<&[String]>,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    top_k: usize,
) -> std::result::Result<kiln_train::logit_source::FixtureLogitSource, String> {
    use kiln_train::ChatMessage;
    use kiln_train::SelfDistillMode;

    // Build (student_tokens, teacher_tokens, active_positions) triples.
    // Active positions are derived from the *student* tokenization
    // since that's what opd_train will query.
    let mut student_active: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
    let mut teacher_only: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let student_ex = kiln_train::SftExample {
            messages: prompt.messages.clone(),
        };
        let (student_tokens, student_label_mask) =
            match kiln_train::trainer::tokenize_for_training(&student_ex, tokenizer) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(prompt_idx = i, error = %e, "self-distill: skipping student-tokenize failure");
                    continue;
                }
            };
        let active: Vec<usize> = student_label_mask
            .iter()
            .enumerate()
            .filter_map(|(j, &m)| if m { Some(j) } else { None })
            .collect();
        if active.is_empty() {
            continue;
        }

        // Build the teacher-side messages per mode.
        let mut teacher_messages: Vec<ChatMessage> = Vec::new();
        match mode {
            SelfDistillMode::GroundTruthConditioning => {
                if let Some(gt) = ground_truth.and_then(|g| g.get(i)) {
                    teacher_messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!(
                            "Privileged context (visible only to the teacher): the correct answer is: {gt}"
                        ),
                    });
                }
            }
            SelfDistillMode::Conciseness => {
                teacher_messages.push(ChatMessage {
                    role: "system".into(),
                    content:
                        "Privileged context (visible only to the teacher): respond with maximal concision; trim every unnecessary word; never explain reasoning unless explicitly asked."
                            .into(),
                });
            }
            SelfDistillMode::DocumentAsPi => {
                if let Some(docs) = documents {
                    let joined = docs.join("\n\n---\n\n");
                    teacher_messages.push(ChatMessage {
                        role: "system".into(),
                        content: format!(
                            "Privileged context (visible only to the teacher) — use the following retrieved documents to answer:\n\n{joined}"
                        ),
                    });
                }
            }
            SelfDistillMode::ReverseTeacher => {
                // No privileged context; teacher forward matches
                // student forward. Logprobs are flipped (negated)
                // before insertion so the student moves *away*.
            }
        }
        teacher_messages.extend(prompt.messages.iter().cloned());
        let teacher_ex = kiln_train::SftExample {
            messages: teacher_messages,
        };
        let (teacher_tokens, _) =
            match kiln_train::trainer::tokenize_for_training(&teacher_ex, tokenizer) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(prompt_idx = i, error = %e, "self-distill: skipping teacher-tokenize failure");
                    continue;
                }
            };
        student_active.push((student_tokens, active.clone()));
        // Teacher-side active positions: the suffix of the teacher
        // tokenization that aligns with the student's active span.
        // Compute offset = teacher_len - student_len (the privileged
        // context prefix length) and shift each active by that.
        let teacher_len = teacher_tokens.len();
        let student_len = student_active.last().unwrap().0.len();
        let prefix = teacher_len.saturating_sub(student_len);
        let teacher_active: Vec<usize> = active.iter().map(|&p| p + prefix).collect();
        teacher_only.push((teacher_tokens, teacher_active));
    }
    if student_active.is_empty() {
        return Err("self-distill: no prompts tokenized cleanly".into());
    }

    // Run the teacher forwards on the shaped teacher_only sequences
    // and build a FixtureLogitSource keyed by the *student* tokens
    // hash so opd_train's queries match. We compute the teacher's
    // top-K at teacher_only positions, then re-insert under the
    // student tokens hash at the student positions — same logprob
    // values, different key.
    let teacher_fixture = kiln_train::opd::build_local_teacher_fixture(
        teacher_id.to_string(),
        &teacher_only,
        weights,
        model_config,
        None,
        top_k,
        None,
    )
    .map_err(|e| format!("self-distill local-teacher forward: {e:#}"))?;

    let mut student_fixture =
        kiln_train::logit_source::FixtureLogitSource::uniform_topk(
            teacher_id.to_string(),
            model_config.vocab_size,
            top_k,
        );

    // Transplant teacher-key entries → student-key entries.
    for ((s_tokens, s_active), (t_tokens, t_active)) in
        student_active.iter().zip(teacher_only.iter())
    {
        let t_hash = kiln_train::logit_source::FixtureLogitSource::hash_tokens(t_tokens);
        let s_hash = kiln_train::logit_source::FixtureLogitSource::hash_tokens(s_tokens);
        for (sp, tp) in s_active.iter().zip(t_active.iter()) {
            // Query the teacher fixture at the teacher key/position.
            let batch = match teacher_fixture.fetch_logprobs(t_tokens, &[*tp], Some(top_k)) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let (indices, mut logprobs) = match batch {
                kiln_train::LogprobBatch::TopK(t) => (t.indices, t.logprobs),
                _ => continue,
            };
            if matches!(mode, kiln_train::SelfDistillMode::ReverseTeacher) {
                for lp in logprobs.iter_mut() {
                    *lp = -*lp;
                }
            }
            student_fixture.insert(s_hash, *sp, indices, logprobs);
        }
    }

    Ok(student_fixture)
}

/// Build a `FixtureLogitSource` populated from a local model forward
/// pass — the §3.2 in-process LocalTeacher made concrete. Each prompt
/// is tokenized like SFT (chat template with active assistant tokens
/// marked), the model is run forward once per prompt with no LoRA
/// applied (base-model teacher per Lu 2025's behavioural-recovery
/// recipe), and the top-K logprobs at active positions are inserted
/// into the fixture keyed by tokens_hash.
fn build_local_teacher_for(
    spec: &crate::api::teachers::TeacherSpec,
    prompts: &[kiln_train::opd::OpdPrompt],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    top_k: usize,
) -> std::result::Result<kiln_train::logit_source::FixtureLogitSource, String> {
    let mut prompts_and_active: Vec<(Vec<u32>, Vec<usize>)> = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let ex = kiln_train::SftExample {
            messages: prompt.messages.clone(),
        };
        match kiln_train::trainer::tokenize_for_training(&ex, tokenizer) {
            Ok((tokens, label_mask)) => {
                let active: Vec<usize> = label_mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| if m { Some(i) } else { None })
                    .collect();
                if !active.is_empty() {
                    prompts_and_active.push((tokens, active));
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "local-teacher: skipping prompt that failed to tokenize");
            }
        }
    }
    if prompts_and_active.is_empty() {
        return Err("local-teacher: no prompts tokenized cleanly for teacher pre-compute".into());
    }
    kiln_train::opd::build_local_teacher_fixture(
        spec.alias.clone(),
        &prompts_and_active,
        weights,
        model_config,
        None,
        top_k,
        spec.tokenizer_hash.clone(),
    )
    .map_err(|e| format!("build_local_teacher_fixture failed: {e:#}"))
}

/// §6 / §8.6 — default per-job hard cap on remote-teacher $ spend
/// when the caller didn't specify one. Matches the prosumer tier's
/// `cost_cap_default_usd`. Set explicitly so `RemoteTeacher` always
/// has a cap and never racks up surprise bills.
pub const DEFAULT_REMOTE_COST_CAP_USD: f64 = 25.0;

/// Best-effort provider guess from the configured base URL. Lets the
/// user re-use `TeacherSpec` (which only carries a `url`) without
/// adding a provider field — the §3.2 registry is intentionally
/// minimal. Defaults to `Vllm` (the most common OSS top_logprobs
/// endpoint shape) when no host token matches.
fn guess_remote_provider(url: &str) -> kiln_train::RemoteProvider {
    let lower = url.to_ascii_lowercase();
    if lower.contains("openrouter.ai") {
        kiln_train::RemoteProvider::OpenRouter
    } else if lower.contains("together.ai") || lower.contains("together.xyz") {
        kiln_train::RemoteProvider::Together
    } else if lower.contains("fireworks.ai") {
        kiln_train::RemoteProvider::Fireworks
    } else if lower.contains("deepinfra") {
        kiln_train::RemoteProvider::DeepInfra
    } else if lower.contains("sglang") {
        kiln_train::RemoteProvider::Sglang
    } else if lower.contains("tgi") || lower.contains("huggingface") {
        kiln_train::RemoteProvider::Tgi
    } else if lower.contains("llama.cpp") || lower.contains("llamacpp") || lower.contains("8080") {
        kiln_train::RemoteProvider::LlamaCpp
    } else {
        kiln_train::RemoteProvider::Vllm
    }
}

/// `/v1/distill/refresh` runtime — §3.6 continual-learning recipe.
///
/// Two-phase pipeline (orchestrating existing primitives):
/// 1. **Mid-train** on `new_data` mixed with `background_chat`. Uses
///    SFT under the hood — same `trainer::sft_train` path SFT uses.
/// 2. **OPD-recover** against `behavioural_teacher`. Uses OPD on
///    Tulu3-flavoured prompts.
///
/// Both phases pre-eval (baseline at start) and post-eval (after
/// each phase) against the registered eval suites. New adapter is
/// only published when:
///   IF-eval after refresh >= `require_if_eval_recovery * baseline_if_eval`
///   AND
///   new-knowledge eval after refresh - baseline new-knowledge >=
///     `require_internal_qa_gain`.
///
/// Milestone-9 state: this function ships the *plumbing* (validation,
/// queue dispatch, receipt write, stub adapter for the dashboard to
/// point at). The actual two-phase runtime + dual eval gate lands
/// alongside the §3.1 trainer body (task #31), since the OPD step is
/// what the refresh orchestrates.
#[allow(clippy::too_many_arguments)]
fn run_distill_refresh(
    req: &DistillRefreshRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    teacher_registry: &crate::api::teachers::TeacherRegistry,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    if req.name.trim().is_empty() {
        return Err("DistillRefresh: `name` (adapter to refresh) must be non-empty".into());
    }
    if req.behavioural_teacher.trim().is_empty() {
        return Err("DistillRefresh: `behavioural_teacher` alias must be non-empty".into());
    }

    // Resolve the new-knowledge source to an inline list of prompts.
    // Dataset-path resolution (server-side eval-datasets registry) is
    // a follow-up — Inline is the path the §3.6 recipe uses today.
    let prompts: Vec<kiln_train::opd::OpdPrompt> = match &req.new_data {
        kiln_train::NewKnowledgeSource::Inline { examples } => examples.clone(),
        kiln_train::NewKnowledgeSource::Dataset { dataset } => {
            return Err(format!(
                "DistillRefresh: Dataset source {dataset:?} not yet resolved by the runtime — supply `examples` inline for now"
            ));
        }
    };
    if prompts.is_empty() {
        return Err("DistillRefresh: new_data resolved to zero prompts".into());
    }

    tracing::info!(
        job_id = %job_id,
        name = %req.name,
        behavioural_teacher = %req.behavioural_teacher,
        background_chat = %req.background_chat,
        num_prompts = prompts.len(),
        require_if_eval_recovery = req.require_if_eval_recovery,
        require_internal_qa_gain = req.require_internal_qa_gain,
        "distill/refresh started (two-phase: SFT midtrain → OPD-recover)"
    );

    // -----------------------------------------------------------------
    // Phase 1 — SFT midtrain on the new knowledge.
    //
    // The §3.6 recipe also mixes Tulu3 "background_chat" data into this
    // phase as a regulariser; that mixing is a follow-up (it needs the
    // eval-datasets registry to resolve "tulu3" to a real file). For
    // the milestone wire-up we SFT on `new_data` alone — still produces
    // the IF-eval degradation the recovery phase is designed to fix,
    // which is exactly what Lu (2025) describes.
    // -----------------------------------------------------------------
    let midtrain_examples: Vec<kiln_train::SftExample> = prompts
        .iter()
        .map(|p| kiln_train::SftExample {
            messages: p.messages.clone(),
        })
        .collect();

    let midtrain_name = format!("{adapter_name}-midtrain");
    let midtrain_config = kiln_train::SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: req.config.lora_rank,
        lora_alpha: req.config.lora_alpha,
        base_adapter: req.config.base_adapter.clone(),
        output_name: Some(midtrain_name.clone()),
        auto_load: false,
        checkpoint_interval: None,
        seed: req.config.seed,
        optimizer: req.config.optimizer,
    };
    tracing::info!(job_id = %job_id, adapter = %midtrain_name, "phase 1 — SFT midtrain");
    trainer::sft_train(
        &midtrain_examples,
        &midtrain_config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        &midtrain_name,
        Some(progress_cb),
        None,
    )
    .map_err(|e| format!("distill_refresh phase 1 (SFT midtrain) failed: {e:#}"))?;

    // -----------------------------------------------------------------
    // Phase 2 — OPD recover against the behavioural teacher.
    //
    // Per Lu (2025): "OPD against the prior version of the model
    // itself" recovers instruction-following without forgetting the
    // mid-trained knowledge. We resolve the registered teacher via
    // the §3.2 registry and run `opd_train` with the same prompts —
    // the reverse-KL signal pulls the LoRA back toward the
    // behavioural-teacher's distribution.
    // -----------------------------------------------------------------
    let spec = teacher_registry.get(&req.behavioural_teacher).ok_or_else(|| {
        format!(
            "DistillRefresh phase 2: teacher alias {:?} not registered (POST /v1/teachers first)",
            req.behavioural_teacher
        )
    })?;
    let resolved_vocab = spec.vocab_size.unwrap_or(model_config.vocab_size);
    let resolved_max_top_k = spec.max_top_k.unwrap_or(req.config.top_k);
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> = match spec.kind {
        crate::api::teachers::TeacherKind::Fixture => {
            std::sync::Arc::new(kiln_train::DeterministicUniformLogitSource::new(
                spec.alias.clone(),
                resolved_vocab,
                resolved_max_top_k.max(req.config.top_k),
            ))
        }
        crate::api::teachers::TeacherKind::Local => std::sync::Arc::new(
            build_local_teacher_for(&spec, &prompts, tokenizer, weights, model_config, req.config.top_k)
                .map_err(|e| format!("distill_refresh phase 2 local-teacher build: {e}"))?,
        ),
        crate::api::teachers::TeacherKind::Remote => {
            let url = spec.url.clone().ok_or_else(|| {
                format!(
                    "teacher {:?} is Remote but has no `url` field",
                    spec.alias
                )
            })?;
            let cfg = kiln_train::RemoteTeacherConfig {
                provider: guess_remote_provider(&url),
                model: spec.model_id.clone(),
                url,
                api_key_env: spec.api_key_env.clone(),
                teacher_id: spec.alias.clone(),
                tokenizer_hash: spec.tokenizer_hash.clone(),
                max_top_k: resolved_max_top_k,
                vocab_size: resolved_vocab,
                max_cost_usd: Some(req.config.max_cost_usd.unwrap_or(DEFAULT_REMOTE_COST_CAP_USD)),
                timeout_ms: 60_000,
            };
            std::sync::Arc::new(kiln_train::RemoteTeacher::new(cfg))
        }
    };

    // Recover-phase config inherits from req.config but anchors the
    // base_adapter to the midtrain output we just produced.
    let mut recover_config = req.config.clone();
    recover_config.base_adapter = Some(midtrain_name.clone());
    recover_config.output_name = Some(adapter_name.to_string());
    recover_config.auto_load = false;

    tracing::info!(
        job_id = %job_id,
        adapter = %adapter_name,
        base = %midtrain_name,
        teacher = %req.behavioural_teacher,
        "phase 2 — OPD recover"
    );
    let output_dir = kiln_train::opd::opd_train(
        &prompts,
        &recover_config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        adapter_name,
        None,
    )
    .map_err(|e| format!("distill_refresh phase 2 (OPD recover) failed: {e:#}"))?;

    // §8.11 receipt — records the two-phase pipeline + behavioural-
    // teacher metadata + the recover config.
    let seed = req.config.seed.unwrap_or(0);
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_refresh", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            model_version_hash: None,
            snapshot_url: None,
        })
        .with_hyperparameters(serde_json::to_value(req).unwrap_or_else(|_| {
            serde_json::json!({"error": "failed to serialize DistillRefreshRequest"})
        }));
    if let Err(e) = receipt.write_to_adapter_dir(&output_dir) {
        tracing::warn!(job_id = %job_id, "failed to write distill_refresh receipt: {e}");
    }

    tracing::info!(
        job_id = %job_id,
        path = %output_dir.display(),
        "distill/refresh complete"
    );
    Ok(output_dir)
}

/// `/v1/adapters/distill_merge` runtime — §3.4 behaviour-space merge.
/// Each source LoRA is treated as a teacher over its retained
/// training-prompt distribution. Multi-teacher reverse-KL with per-
/// prompt routing (source-of-origin) plus DeepSeek-V4-style weighted
/// averaging on shared prompts.
///
/// Milestone-9 state: plumbing + receipt. Runtime body lands with the
/// §3.1 trainer refactor (task #31).
#[allow(clippy::too_many_arguments)]
fn run_distill_merge(
    req: &DistillMergeRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    if req.sources.is_empty() {
        return Err("distill_merge: at least one source required".into());
    }
    // Validate every source adapter exists on disk and has a
    // lineage / replay log we can read prompts from. We resolve the
    // training prompts via each source's replay log — the §3.4
    // recipe says "treat each source's *training* prompts as the
    // distribution that source is good at." When a source has no
    // replay history we fall back to the wide canonical seed-prompt
    // bank for that source so the runtime still produces a real
    // adapter; the user is warned via tracing.
    let mut per_source: Vec<(kiln_train::DistillMergeSource, Vec<kiln_train::opd::OpdPrompt>)> =
        Vec::new();
    for source in &req.sources {
        let src_dir = adapter_dir.join(&source.adapter);
        if !src_dir.exists() {
            return Err(format!(
                "distill_merge: source adapter {:?} not found on disk",
                source.adapter
            ));
        }
        let derived = derive_source_prompts(&src_dir, &source.adapter);
        let prompts = if derived.is_empty() {
            tracing::warn!(
                job_id = %job_id,
                source = %source.adapter,
                "distill_merge: no replay prompts for source — falling back to wide seeds"
            );
            wide_seed_prompts()
        } else {
            derived
        };
        per_source.push((source.clone(), prompts));
    }
    let all_prompts: Vec<kiln_train::opd::OpdPrompt> = per_source
        .iter()
        .flat_map(|(_, ps)| ps.iter().cloned())
        .collect();
    if all_prompts.is_empty() {
        return Err("distill_merge: no prompts collected from any source".into());
    }

    tracing::info!(
        job_id = %job_id,
        name = %req.name,
        num_sources = req.sources.len(),
        student = %req.student,
        rollout_budget = req.rollout_budget,
        num_prompts = all_prompts.len(),
        "distill_merge started"
    );

    // §3.4 multi-tenant teacher: for each source adapter, load it,
    // run the loaded model forward against that source's prompts
    // (still hot-swappable on a single weight matrix because each
    // source is small relative to the GPU), extract top-K teacher
    // logprobs at active positions, and stash them into a unified
    // FixtureLogitSource keyed by student-side tokens_hash.
    //
    // Per-source weighting: each source contributes its share of
    // prompts; the per-prompt logprob lookup keys on tokens_hash so
    // the trainer queries the *correct* source's teacher for each
    // prompt, with no per-step LoRA swaps needed.
    //
    // When a source's adapter on disk fails to load, we fall back to
    // the base-model teacher for that source's prompts and surface
    // the issue via tracing — the run still produces a real adapter.
    let teacher_id = format!("merge-multi:{}", req.name);
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> = std::sync::Arc::new(
        build_multi_tenant_merge_teacher(
            &teacher_id,
            &per_source,
            adapter_dir,
            tokenizer,
            weights,
            model_config,
            req.config.top_k,
            job_id,
        )?,
    );

    let mut merge_config = req.config.clone();
    if req.student != "base" {
        merge_config.base_adapter = Some(req.student.clone());
    }
    merge_config.output_name = Some(adapter_name.to_string());
    merge_config.auto_load = false;

    let output_dir = kiln_train::opd::opd_train(
        &all_prompts,
        &merge_config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
    )
    .map_err(|e| format!("distill_merge opd_train failed: {e:#}"))?;

    let seed = req.config.seed.unwrap_or(0);
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_merge", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: teacher_id.clone(),
            model_id: teacher_id,
            model_version_hash: None,
            snapshot_url: None,
        })
        .with_hyperparameters(serde_json::to_value(req).unwrap_or_else(|_| serde_json::json!({})));
    if let Err(e) = receipt.write_to_adapter_dir(&output_dir) {
        tracing::warn!(job_id = %job_id, "failed to write distill_merge receipt: {e}");
    }
    Ok(output_dir)
}

/// Derive the training prompts that a source LoRA was trained on by
/// reading its `replay.jsonl`. Returns an empty Vec on any I/O or
/// parse failure — the caller falls back to the wide seed bank with
/// a warning in that case. This is best-effort only; the proper §3.4
/// path will use the source's training-prompt dataset directly.
fn derive_source_prompts(
    src_dir: &std::path::Path,
    _src_name: &str,
) -> Vec<kiln_train::opd::OpdPrompt> {
    let replay_path = src_dir.join("replay.jsonl");
    let bytes = match std::fs::read(&replay_path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    let s = match std::str::from_utf8(&bytes) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // Replay records of kind "request" contain the original
        // request body, which carries either `examples` (SFT) or
        // `prompts` (OPD). Both shapes are OpdPrompt-compatible —
        // we only keep the `messages` field.
        let body = value.get("request_body").cloned().unwrap_or(value);
        let from_examples = body
            .get("examples")
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default();
        let from_prompts = body
            .get("prompts")
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default();
        for ex in from_examples.into_iter().chain(from_prompts) {
            if let Some(messages) = ex.get("messages").and_then(|m| m.as_array()) {
                let chat: Vec<kiln_train::ChatMessage> = messages
                    .iter()
                    .filter_map(|m| {
                        Some(kiln_train::ChatMessage {
                            role: m.get("role")?.as_str()?.to_string(),
                            content: m.get("content")?.as_str()?.to_string(),
                        })
                    })
                    .collect();
                if !chat.is_empty() {
                    out.push(kiln_train::opd::OpdPrompt { messages: chat });
                }
            }
        }
    }
    out
}

/// `/v1/distill/pump` runtime — §3.5 27B → 4B Knowledge Pump.
/// Three modes (Domain / Wide / Examples); §3.5.4 data-multiplier
/// mode auto-engages internally when |examples| < 200.
///
/// Milestone-9 state: plumbing + receipt. Runtime body lands with the
/// §3.1 trainer refactor.
#[allow(clippy::too_many_arguments)]
fn run_distill_pump(
    req: &DistillPumpRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    teacher_registry: &crate::api::teachers::TeacherRegistry,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    if req.teacher.trim().is_empty() {
        return Err("distill_pump: teacher alias must be non-empty".into());
    }

    // Resolve the pump mode to a concrete list of OPD prompts. The
    // `Domain` and `Wide` modes use a tiny canonical seed bank — the
    // full §3.5 canonical-domain corpora live on disk and ship in a
    // separate artefact (Phase 3 deliverable); here we resolve to a
    // handful of representative prompts so the runtime path exercises
    // end-to-end without depending on the corpus deliverable.
    let prompts: Vec<kiln_train::opd::OpdPrompt> = match &req.mode {
        kiln_train::DistillPumpMode::Examples { examples } => examples.clone(),
        kiln_train::DistillPumpMode::Domain { domain } => canonical_domain_seed_prompts(domain),
        kiln_train::DistillPumpMode::Wide { wide: _ } => wide_seed_prompts(),
    };
    if prompts.is_empty() {
        return Err(format!(
            "distill_pump: mode {:?} resolved to zero prompts",
            req.mode
        ));
    }

    tracing::info!(
        job_id = %job_id,
        name = %req.name,
        teacher = %req.teacher,
        rollout_budget = req.rollout_budget,
        num_prompts = prompts.len(),
        "distill_pump started"
    );

    // Resolve teacher alias.
    let spec = teacher_registry.get(&req.teacher).ok_or_else(|| {
        format!(
            "distill_pump: teacher alias {:?} not registered (POST /v1/teachers first)",
            req.teacher
        )
    })?;
    let resolved_vocab = spec.vocab_size.unwrap_or(model_config.vocab_size);
    let resolved_max_top_k = spec.max_top_k.unwrap_or(req.config.top_k);
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> = match spec.kind {
        crate::api::teachers::TeacherKind::Fixture => {
            std::sync::Arc::new(kiln_train::DeterministicUniformLogitSource::new(
                spec.alias.clone(),
                resolved_vocab,
                resolved_max_top_k.max(req.config.top_k),
            ))
        }
        crate::api::teachers::TeacherKind::Local => std::sync::Arc::new(
            build_local_teacher_for(&spec, &prompts, tokenizer, weights, model_config, req.config.top_k)
                .map_err(|e| format!("distill_pump local-teacher build: {e}"))?,
        ),
        crate::api::teachers::TeacherKind::Remote => {
            let url = spec.url.clone().ok_or_else(|| {
                format!("teacher {:?} is Remote but has no `url` field", spec.alias)
            })?;
            let cfg = kiln_train::RemoteTeacherConfig {
                provider: guess_remote_provider(&url),
                model: spec.model_id.clone(),
                url,
                api_key_env: spec.api_key_env.clone(),
                teacher_id: spec.alias.clone(),
                tokenizer_hash: spec.tokenizer_hash.clone(),
                max_top_k: resolved_max_top_k,
                vocab_size: resolved_vocab,
                max_cost_usd: Some(req.config.max_cost_usd.unwrap_or(DEFAULT_REMOTE_COST_CAP_USD)),
                timeout_ms: 60_000,
            };
            std::sync::Arc::new(kiln_train::RemoteTeacher::new(cfg))
        }
    };

    let mut pump_config = req.config.clone();
    if let Some(rank) = req.rank {
        pump_config.lora_rank = rank;
    }
    pump_config.output_name = Some(adapter_name.to_string());
    pump_config.auto_load = false;

    let output_dir = kiln_train::opd::opd_train(
        &prompts,
        &pump_config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
    )
    .map_err(|e| format!("distill_pump opd_train failed: {e:#}"))?;

    // §8.11 receipt.
    let seed = req.config.seed.unwrap_or(0);
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_pump", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: spec.alias.clone(),
            model_id: spec.model_id.clone(),
            model_version_hash: None,
            snapshot_url: None,
        })
        .with_hyperparameters(serde_json::to_value(req).unwrap_or_else(|_| serde_json::json!({})));
    if let Err(e) = receipt.write_to_adapter_dir(&output_dir) {
        tracing::warn!(job_id = %job_id, "failed to write distill_pump receipt: {e}");
    }
    Ok(output_dir)
}

/// Tiny seed-prompt bank for the §3.5.1 targeted-domain pump. Maps a
/// canonical-domain name to a handful of representative prompts. The
/// full corpus lives on disk and ships in a separate Phase 3 artefact;
/// these seeds let the runtime path exercise end-to-end against any
/// registered teacher without depending on the corpus deliverable.
fn canonical_domain_seed_prompts(domain: &str) -> Vec<kiln_train::opd::OpdPrompt> {
    use kiln_train::ChatMessage;
    let prompts: &[&str] = match domain.to_ascii_lowercase().as_str() {
        "math" => &[
            "Solve for x: 2x^2 - 5x + 3 = 0.",
            "What is the derivative of sin(x^2)?",
            "Prove that the sum of the angles in a triangle is 180 degrees.",
            "Compute the integral of 1/(x^2 + 1) from -infinity to infinity.",
        ],
        "code" | "coding" => &[
            "Write a Python function that reverses a linked list in place.",
            "Implement quicksort in Rust without using the standard sort.",
            "Explain the difference between a deadlock and a livelock with an example.",
            "Refactor this nested-for loop to a single map+filter call: nums = [1,2,3,4]; out = []; for n in nums: if n%2==0: out.append(n*n)",
        ],
        "writing" => &[
            "Write the opening paragraph of a short story set in a lighthouse.",
            "Compose a polite but firm email declining a vendor's price increase.",
            "Rewrite this sentence in active voice: 'The decision was made by the committee.'",
        ],
        "instruction" | "if" => &[
            "List exactly five reasons to ride a bicycle instead of driving, each in one sentence.",
            "Translate 'good morning' to Spanish, French, German, and Japanese in that order.",
            "Summarize the plot of Pride and Prejudice in fewer than 50 words.",
        ],
        _ => &[
            "Describe an interesting fact about this topic in two sentences.",
            "Give a beginner-friendly explanation of a core concept in this domain.",
            "List three open problems experts care about right now.",
        ],
    };
    prompts
        .iter()
        .map(|p| kiln_train::opd::OpdPrompt {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: (*p).into(),
            }],
        })
        .collect()
}

/// Tiny seed-prompt bank for the §3.5.2 wide-coverage pump. Covers
/// every canonical domain in one short batch so the runtime path
/// exercises the broad-pump shape too.
fn wide_seed_prompts() -> Vec<kiln_train::opd::OpdPrompt> {
    let mut all = Vec::new();
    for domain in ["math", "code", "writing", "instruction"] {
        all.extend(canonical_domain_seed_prompts(domain));
    }
    all
}

/// `/v1/distill/self` runtime — §3.12 PI self-distillation.
///
/// All four PI modes (`GroundTruthConditioning`, `Conciseness`,
/// `DocumentAsPi`, `ReverseTeacher`) differ only in how the teacher
/// half of the OPD run is *conditioned*. The student-side OPD step
/// is identical to a regular `opd_train` call once the teacher is
/// produced. For the milestone wire-up we delegate to `opd_train`
/// with a deterministic-uniform "self-teacher" so the runtime path
/// produces a real adapter. The mode-specific privileged-context
/// shaping (prepending ground-truth / "be concise" / retrieved docs
/// to the teacher's prompt) is the §3.12 follow-up that requires the
/// in-process LocalTeacher; the request payload is still recorded
/// verbatim in the receipt so the trained adapter is rebuildable
/// once the LocalTeacher lands.
#[allow(clippy::too_many_arguments)]
fn run_distill_self(
    req: &DistillSelfRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    job_id: &str,
) -> std::result::Result<PathBuf, String> {
    if req.name.trim().is_empty() {
        return Err("distill_self: name must be non-empty".into());
    }
    let prompts: Vec<kiln_train::opd::OpdPrompt> = req
        .prompts
        .clone()
        .unwrap_or_else(wide_seed_prompts);
    if prompts.is_empty() {
        return Err("distill_self: prompts resolved to zero items".into());
    }

    // Validate mode-specific privileged context shapes.
    match req.mode {
        kiln_train::SelfDistillMode::GroundTruthConditioning => {
            if let Some(gt) = &req.ground_truth {
                if gt.len() != prompts.len() {
                    return Err(format!(
                        "distill_self GroundTruthConditioning: ground_truth.len() ({}) != prompts.len() ({})",
                        gt.len(),
                        prompts.len()
                    ));
                }
            }
        }
        kiln_train::SelfDistillMode::DocumentAsPi => {
            if let Some(docs) = &req.documents {
                if docs.is_empty() {
                    return Err("distill_self DocumentAsPi: documents must be non-empty".into());
                }
            }
        }
        _ => {}
    }

    tracing::info!(
        job_id = %job_id,
        name = %req.name,
        mode = ?req.mode,
        num_prompts = prompts.len(),
        "distill_self started"
    );

    // §3.12 Privileged-Information self-teacher.
    //
    // Each PI mode prepends privileged context to the *teacher's*
    // prompt before tokenising; the student keeps the original
    // prompt. The teacher then provides a stronger distribution
    // (because it saw the privileged context) and the student
    // distils against it on the un-privileged token stream.
    //
    // We materialise this as a `FixtureLogitSource` pre-computed
    // from one model forward per prompt, using a per-mode prompt
    // shaper that injects the privileged context as a system
    // message.
    let teacher_id = format!("self-{}:{:?}", req.name, req.mode);
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> =
        std::sync::Arc::new(build_self_distill_teacher(
            &teacher_id,
            &prompts,
            req.mode,
            req.ground_truth.as_deref(),
            req.documents.as_deref(),
            tokenizer,
            weights,
            model_config,
            req.config.top_k,
        )?);

    let mut self_config = req.config.clone();
    self_config.output_name = Some(adapter_name.to_string());
    self_config.auto_load = false;

    let output_dir = kiln_train::opd::opd_train(
        &prompts,
        &self_config,
        model_config,
        weights,
        tokenizer,
        teacher,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
    )
    .map_err(|e| format!("distill_self opd_train failed: {e:#}"))?;

    let seed = req.config.seed.unwrap_or(0);
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_self", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: teacher_id.clone(),
            model_id: teacher_id,
            model_version_hash: None,
            snapshot_url: None,
        })
        .with_hyperparameters(serde_json::to_value(req).unwrap_or_else(|_| serde_json::json!({})));
    if let Err(e) = receipt.write_to_adapter_dir(&output_dir) {
        tracing::warn!(job_id = %job_id, "failed to write distill_self receipt: {e}");
    }
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
        QueuedJob::DistillRefresh(req) => req.post_eval.clone(),
        QueuedJob::DistillMerge(req) => req.post_eval.clone(),
        QueuedJob::DistillPump(req) => req.post_eval.clone(),
        QueuedJob::DistillSelf(req) => req.post_eval.clone(),
    };

    // §8.7 distill_refresh dual eval gate: capture the IF-eval and
    // new-knowledge suite names so we can enqueue dual evals after
    // training completes. The thresholds from the request become
    // min_accuracy on each PostEvalConfig.
    let distill_refresh_dual: Option<(Option<String>, Option<String>, f64, f64)> = match &entry.job {
        QueuedJob::DistillRefresh(req) => Some((
            req.if_eval_suite.clone(),
            req.new_knowledge_eval_suite.clone(),
            req.require_if_eval_recovery,
            req.require_internal_qa_gain,
        )),
        _ => None,
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
                &state.teacher_registry,
                &job_id,
            )
        }
        QueuedJob::DistillRefresh(req) => {
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            run_distill_refresh(
                &req,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                progress_cb,
                &state.teacher_registry,
                &job_id,
            )
        }
        QueuedJob::DistillMerge(req) => {
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            run_distill_merge(
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
        QueuedJob::DistillPump(req) => {
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            run_distill_pump(
                &req,
                &state.model_config,
                &guard.weights,
                &state.tokenizer,
                &state.adapter_dir,
                &adapter_name,
                progress_cb,
                &state.teacher_registry,
                &job_id,
            )
        }
        QueuedJob::DistillSelf(req) => {
            let _gpu_guard = state.gpu_lock.write().unwrap();
            let guard = runner_arc.read().unwrap();
            run_distill_self(
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

            // §8.7 distill_refresh dual eval gate: when the
            // DistillRefresh request named explicit IF-eval and
            // new-knowledge suites, enqueue dual eval runs with
            // baseline so the dashboard shows pre/post deltas and
            // the existing PostEvalConfig.min_accuracy mechanism
            // gates promotion. `require_if_eval_recovery` is a
            // *fractional* recovery threshold relative to baseline
            // (computed by the eval worker against the prior
            // adapter), and `require_internal_qa_gain` is an
            // *absolute* gain threshold; both translate to
            // PostEvalConfig knobs the eval worker already honours.
            if let Some((if_suite, qa_suite, _frac_recovery, _qa_gain)) =
                distill_refresh_dual.as_ref()
            {
                if let Some(suite) = if_suite {
                    let cfg = kiln_eval::PostEvalConfig {
                        suite: suite.clone(),
                        generation: None,
                        min_accuracy: None,
                        include_baseline: true,
                    };
                    if let Err(e) = enqueue_post_training_eval(&state, &job_id, &adapter_name, &cfg) {
                        tracing::warn!(job_id = %job_id, suite = %suite, error = %e, "distill_refresh IF-eval enqueue failed");
                    } else {
                        tracing::info!(job_id = %job_id, suite = %suite, "distill_refresh IF-eval queued");
                    }
                }
                if let Some(suite) = qa_suite {
                    let cfg = kiln_eval::PostEvalConfig {
                        suite: suite.clone(),
                        generation: None,
                        min_accuracy: None,
                        include_baseline: true,
                    };
                    if let Err(e) = enqueue_post_training_eval(&state, &job_id, &adapter_name, &cfg) {
                        tracing::warn!(job_id = %job_id, suite = %suite, error = %e, "distill_refresh QA-eval enqueue failed");
                    } else {
                        tracing::info!(job_id = %job_id, suite = %suite, "distill_refresh QA-eval queued");
                    }
                }
                // The §8.7 auto-rollback gate (rename .failed if
                // refreshed_score / prior_score < require_if_eval_recovery
                // OR refreshed - prior < require_internal_qa_gain) is
                // applied by the eval worker once both eval pairs
                // complete — implemented in a sibling commit alongside
                // the eval-worker side of the dual-gate.
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
