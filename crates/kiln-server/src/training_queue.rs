//! FIFO training job queue — accepts SFT and GRPO jobs, runs them sequentially.
//!
//! The queue ensures only one training job runs at a time, preventing GPU memory
//! conflicts between concurrent training jobs. Jobs are executed in submission order.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use kiln_memory::vram::VramSource;
use kiln_train::trainer;
use kiln_train::{
    self, DistillMergeRequest, DistillPumpRequest, DistillRefreshRequest, DistillSelfRequest,
    GrpoRequest, LogitSource as _, OpdRequest, SftRequest, TrainingState,
};
use serde::Serialize;

use crate::batching_engine::KvResizeReason;
use crate::metrics::{TrainingMetricStatus, TrainingMetricType};
use crate::recent_requests::now_unix_ms;
use crate::state::{AppState, ModelBackend, TrainingJobType};
use crate::training_history;

/// Mark the tracked job terminal (Completed / Failed), stamp `finished_at`
/// + `finished_unix_ms` + the failure detail, and persist a clone to the
/// on-disk archive. Archive write failures are logged, never propagated —
/// disk wedged or quota-exceeded must not derail the worker's reporting path.
fn finalize_job(state: &AppState, job_id: &str, new_state: TrainingState, error: Option<String>) {
    let snapshot = {
        let mut jobs = state.training_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(job_id) {
            job.state = new_state;
            job.error = error;
            job.finished_at = Some(std::time::Instant::now());
            job.finished_unix_ms = Some(now_unix_ms());
            Some(job.clone())
        } else {
            None
        }
    };
    let Some(job) = snapshot else { return };
    // The corrections feed contract: rows consumed via the
    // `corrections:active` dataset flip to trained_into only when the
    // job actually COMPLETED — a failed/cancelled job leaves the basket
    // intact so the hand-written ideals stay re-trainable.
    if new_state == TrainingState::Completed && !job.consumed_correction_ids.is_empty() {
        let store = crate::api::corrections::CorrectionsStore::for_state(state);
        let marked = store.mark_trained_into(&job.consumed_correction_ids, &job.adapter_name);
        tracing::info!(
            job_id = %job_id,
            adapter = %job.adapter_name,
            marked,
            "corrections rows marked trained on job completion"
        );
    }
    if let Err(e) = training_history::save(&state.adapter_dir, &job) {
        tracing::warn!(error = %e, job_id = %job_id, "failed to archive terminal training job");
    }
    training_history::prune_to_max(&state.adapter_dir, training_history::MAX_ARCHIVED_JOBS);
}

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
/// Fire-and-forget POST of an arbitrary JSON event. Same contract as
/// [`fire_completion_webhook`]: failures are logged at WARN and never
/// affect the job that emitted the event.
pub fn fire_webhook_json(url: String, event: serde_json::Value) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        match client.post(&url).json(&event).send().await {
            Ok(resp) if !resp.status().is_success() => {
                tracing::warn!(url = %url, status = %resp.status(), "webhook POST returned non-success");
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(url = %url, error = %e, "webhook POST failed");
            }
        }
    });
}

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

impl QueuedJob {
    /// The registered teacher alias whose exact spec must be pinned when this
    /// job is admitted. Other job kinds either do not use a teacher or build
    /// their source directly from local adapter state.
    pub(crate) fn registered_teacher_alias(&self) -> Option<&str> {
        match self {
            Self::Opd(req) => Some(&req.teacher),
            Self::DistillRefresh(req) => Some(&req.behavioural_teacher),
            Self::DistillPump(req) => Some(&req.teacher),
            Self::Sft(_) | Self::Grpo(_) | Self::DistillMerge(_) | Self::DistillSelf(_) => None,
        }
    }
}

/// Entry in the training queue.
pub struct QueueEntry {
    pub job_id: String,
    /// Estimated per-step working-set bytes from the submit-time preflight
    /// (#24). `execute_job` holds a governor reservation of this size across the
    /// job so the KV autoscaler proactively shrinks inference KV before training
    /// allocates. `0` when no estimate was available (skips the reservation).
    pub reserved_bytes: u64,
    /// Submit-time snapshots of registered teachers used by this job. The
    /// worker requires exactly one matching snapshot for teacher-backed jobs
    /// and refuses to run if the alias was deleted or replaced while queued.
    pub teacher_bindings: Vec<crate::api::teachers::TeacherSpec>,
    pub job: QueuedJob,
}

fn resolve_pinned_teacher_for_job(
    job: &QueuedJob,
    bindings: &[crate::api::teachers::TeacherSpec],
    registry: &crate::api::teachers::TeacherRegistry,
) -> std::result::Result<Option<crate::api::teachers::TeacherSpec>, String> {
    let Some(alias) = job.registered_teacher_alias() else {
        if bindings.is_empty() {
            return Ok(None);
        }
        return Err(format!(
            "queued job does not use a registered teacher but carries {} pinned teacher binding(s)",
            bindings.len()
        ));
    };

    let matching: Vec<_> = bindings.iter().filter(|spec| spec.alias == alias).collect();
    if matching.is_empty() {
        return Err(format!(
            "queued teacher alias {alias:?} has no submit-time pinned binding; refuse to resolve mutable registry state"
        ));
    }
    if matching.len() > 1 {
        return Err(format!(
            "queued teacher alias {alias:?} has {} duplicate submit-time bindings; expected exactly one",
            matching.len()
        ));
    }
    if bindings.len() != 1 {
        return Err(format!(
            "queued teacher alias {alias:?} has {} total submit-time bindings; expected exactly one with no extras",
            bindings.len()
        ));
    }

    let pinned = matching[0];
    let current = registry.get(alias).ok_or_else(|| {
        format!(
            "queued teacher alias {alias:?} was deleted after submission; re-register it and submit a new job"
        )
    })?;
    if current != *pinned {
        return Err(format!(
            "queued teacher alias {alias:?} was replaced after submission; refusing to switch teacher identity (submit a new job)"
        ));
    }
    Ok(Some(pinned.clone()))
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

/// Evict `Completed` / `Failed` entries from `state.training_jobs` when the
/// tracking map grows past `state.max_tracked_jobs`. Oldest-by-finish-time
/// entries go first. Active entries (`Queued` / `Running`) are never
/// removed regardless of age. The TTL field is honored as a soft floor —
/// nothing inside the TTL window gets evicted, even when over the cap (in
/// that case a submission will be rejected with `training_tracked_full`
/// per the existing contract). Terminal entries that pass the TTL window
/// are eligible for cap-based eviction.
///
/// Returns the number of entries removed.
///
/// Safe to call from any thread; takes a short write lock on
/// `training_jobs`. Called from the training worker loop on every
/// iteration and from tests directly.
pub fn gc_tracked_jobs(state: &AppState) -> usize {
    let cap = state.max_tracked_jobs;
    let ttl = state.tracked_job_ttl;
    let now = std::time::Instant::now();
    let mut jobs = state.training_jobs.write().unwrap();
    if jobs.len() <= cap {
        return 0;
    }
    // Build a candidate list of terminal jobs past the TTL window, ordered
    // oldest-first by `finished_at`. We evict from the front of this list
    // until we're back under the cap or the candidate list is exhausted.
    let mut candidates: Vec<(String, std::time::Instant)> = jobs
        .iter()
        .filter_map(|(id, j)| match (j.state, j.finished_at) {
            (TrainingState::Completed | TrainingState::Failed, Some(t))
                if now.saturating_duration_since(t) >= ttl =>
            {
                Some((id.clone(), t))
            }
            _ => None,
        })
        .collect();
    candidates.sort_by_key(|(_, t)| *t);
    let want_to_remove = jobs.len().saturating_sub(cap);
    let mut removed = 0;
    for (id, _) in candidates.into_iter().take(want_to_remove) {
        jobs.remove(&id);
        removed += 1;
    }
    if removed > 0 {
        tracing::debug!(
            removed,
            remaining = jobs.len(),
            cap,
            "evicted oldest terminal training jobs past TTL to honor max_tracked_jobs cap"
        );
    }
    removed
}

/// Dispatch one SFT job to either the default in-process kt-tape trainer or a
/// backend-selected native trainer.
///
/// The shared kt-tape path takes a `replay_ctx` (request_body + lineage
/// tracking); the legacy CUDA-native path doesn't yet plumb replay so it drops
/// the context. When the binary is built without `--features cuda`, the native
/// route flag falls through to the shared kt-tape path with a warning.
fn normalize_sft_resume_checkpoint(
    config: &mut kiln_train::SftConfig,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<(), String> {
    normalize_training_resume_checkpoint(
        &mut config.resume_checkpoint,
        kiln_train::checkpoint::TrainingKind::Sft,
        "SFT",
        adapter_dir,
        adapter_name,
    )
}

fn normalize_grpo_resume_checkpoint(
    config: &mut kiln_train::GrpoConfig,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<(), String> {
    normalize_training_resume_checkpoint(
        &mut config.resume_checkpoint,
        kiln_train::checkpoint::TrainingKind::Grpo,
        "GRPO",
        adapter_dir,
        adapter_name,
    )
}

fn normalize_opd_resume_checkpoint(
    config: &mut kiln_train::OpdConfig,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<(), String> {
    normalize_training_resume_checkpoint(
        &mut config.resume_checkpoint,
        kiln_train::checkpoint::TrainingKind::Opd,
        "OPD",
        adapter_dir,
        adapter_name,
    )
}

pub(crate) fn materialize_sft_effective_seed(
    config: &mut kiln_train::SftConfig,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<u64, String> {
    normalize_sft_resume_checkpoint(config, adapter_dir, adapter_name)?;
    materialize_training_effective_seed(
        &mut config.seed,
        config.resume_checkpoint.as_deref(),
        "SFT",
    )
}

pub(crate) fn materialize_grpo_effective_seed(
    config: &mut kiln_train::GrpoConfig,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<u64, String> {
    normalize_grpo_resume_checkpoint(config, adapter_dir, adapter_name)?;
    materialize_training_effective_seed(
        &mut config.seed,
        config.resume_checkpoint.as_deref(),
        "GRPO",
    )
}

pub(crate) fn materialize_opd_effective_seed(
    config: &mut kiln_train::OpdConfig,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<u64, String> {
    normalize_opd_resume_checkpoint(config, adapter_dir, adapter_name)?;
    materialize_training_effective_seed(
        &mut config.seed,
        config.resume_checkpoint.as_deref(),
        "OPD",
    )
}

fn materialize_training_effective_seed(
    requested_seed: &mut Option<u64>,
    resume_checkpoint: Option<&str>,
    training_label: &str,
) -> Result<u64, String> {
    let effective_seed = if let Some(path) = resume_checkpoint {
        let checkpoint = kiln_train::checkpoint::load_training_checkpoint(std::path::Path::new(
            path,
        ))
        .map_err(|error| format!("read {training_label} resume seed from checkpoint: {error:#}"))?;
        let checkpoint_seed = checkpoint
            .manifest
            .rng_states
            .get("lora-init")
            .ok_or_else(|| {
                format!(
                    "{training_label} resume checkpoint is missing the authoritative lora-init RNG state"
                )
            })?
            .seed;
        if let Some(requested_seed) = *requested_seed
            && requested_seed != checkpoint_seed
        {
            return Err(format!(
                "{training_label} seed {requested_seed} does not match resume checkpoint seed {checkpoint_seed}; omit seed or use the checkpoint value"
            ));
        }
        checkpoint_seed
    } else {
        requested_seed.unwrap_or_else(rand::random)
    };
    *requested_seed = Some(effective_seed);
    Ok(effective_seed)
}

pub(crate) fn materialize_queued_job_effective_seed(
    job: &mut QueuedJob,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<u64, String> {
    match job {
        QueuedJob::Sft(request) => {
            materialize_sft_effective_seed(&mut request.config, adapter_dir, adapter_name)
        }
        QueuedJob::Grpo(request) => {
            materialize_grpo_effective_seed(&mut request.config, adapter_dir, adapter_name)
        }
        QueuedJob::Opd(kiln_train::OpdRequest {
            config: request, ..
        })
        | QueuedJob::DistillRefresh(kiln_train::DistillRefreshRequest {
            config: request, ..
        })
        | QueuedJob::DistillMerge(kiln_train::DistillMergeRequest {
            config: request, ..
        })
        | QueuedJob::DistillPump(kiln_train::DistillPumpRequest {
            config: request, ..
        })
        | QueuedJob::DistillSelf(kiln_train::DistillSelfRequest {
            config: request, ..
        }) => materialize_opd_effective_seed(request, adapter_dir, adapter_name),
    }
}

fn normalize_training_resume_checkpoint(
    resume_checkpoint: &mut Option<String>,
    expected_kind: kiln_train::checkpoint::TrainingKind,
    training_label: &str,
    adapter_dir: &std::path::Path,
    adapter_name: &str,
) -> Result<(), String> {
    let Some(raw) = resume_checkpoint.as_deref() else {
        return Ok(());
    };
    if raw.trim().is_empty() {
        return Err(format!(
            "{training_label} resume_checkpoint must not be empty"
        ));
    }
    let supplied = std::path::Path::new(raw);
    let candidate = if supplied.is_absolute() || supplied.parent() == Some(adapter_dir) {
        supplied.to_path_buf()
    } else {
        let mut components = supplied.components();
        let basename = match (components.next(), components.next()) {
            (Some(std::path::Component::Normal(name)), None) => name,
            _ => {
                return Err(format!(
                    "{training_label} resume_checkpoint must be one checkpoint basename, without traversal or nested directories"
                ));
            }
        };
        adapter_dir.join(basename)
    };
    if candidate.parent() != Some(adapter_dir) {
        return Err(format!(
            "{training_label} resume_checkpoint must be an immutable checkpoint directly beneath {}",
            adapter_dir.display()
        ));
    }
    if !candidate
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.ends_with(kiln_train::checkpoint::TRAINING_CHECKPOINT_DIRECTORY_SUFFIX)
        })
    {
        return Err(format!(
            "{training_label} resume_checkpoint must end with {}",
            kiln_train::checkpoint::TRAINING_CHECKPOINT_DIRECTORY_SUFFIX
        ));
    }
    let checkpoint = kiln_train::checkpoint::load_training_checkpoint(&candidate)
        .map_err(|error| format!("validate {training_label} resume_checkpoint: {error:#}"))?;
    if checkpoint.manifest.training_kind != expected_kind {
        return Err(format!(
            "{training_label} resume_checkpoint contains {:?} state",
            checkpoint.manifest.training_kind
        ));
    }
    if checkpoint.manifest.adapter_name != adapter_name {
        return Err(format!(
            "{training_label} resume_checkpoint adapter {:?} does not match output adapter {:?}",
            checkpoint.manifest.adapter_name, adapter_name
        ));
    }
    *resume_checkpoint = Some(candidate.display().to_string());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_sft(
    native_route_enabled: bool,
    native_route_env: Option<&'static str>,
    req: &SftRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    replay_ctx: trainer::ReplayContext,
    job_id: &str,
    gpu_step_coordination: Option<trainer::GpuStepCoordination>,
) -> std::result::Result<PathBuf, String> {
    let prepared = if let Some(dataset_path) = req.dataset_path.as_deref() {
        if dataset_path.trim().is_empty() {
            return Err("SFT dataset_path training requires a non-empty path".to_string());
        }
        if !req.examples.is_empty() {
            return Err(
                "SFT request must use either examples or dataset_path, not both".to_string(),
            );
        }
        let loaded = crate::sft_dataset::prepare_sft_jsonl(
            std::path::Path::new(dataset_path),
            tokenizer,
            req.config.invalid_row_policy,
            "dataset_path",
            Some(dataset_path.to_string()),
        )
        .map_err(|e| format!("load SFT dataset_path {dataset_path:?}: {e:#}"))?;
        let expected = req.ingestion.as_ref().ok_or_else(|| {
            "queued SFT dataset_path job has no submit-time ingestion receipt".to_string()
        })?;
        if loaded.ingestion != *expected {
            return Err(format!(
                "SFT dataset_path {dataset_path:?} changed after admission: submit kept/rejected={}/{}, worker kept/rejected={}/{}",
                expected.rows_kept,
                expected.rows_rejected,
                loaded.ingestion.rows_kept,
                loaded.ingestion.rows_rejected
            ));
        }
        tracing::info!(
            job_id = %job_id,
            dataset_path,
            examples = loaded.examples.len(),
            kept_corpus_sha256 = %loaded.ingestion.kept_corpus_sha256,
            "revalidated SFT dataset_path against submit-time row manifest"
        );
        loaded
    } else if let Some(ingestion) = req.ingestion.as_ref() {
        let (max_seq_len, max_supervised_tokens) =
            kiln_train::verify_prepared_sft_examples(&req.examples, tokenizer, ingestion)
                .map_err(|error| format!("verify queued SFT examples: {error:#}"))?;
        kiln_train::SftPreparedDataset {
            examples: req.examples.clone(),
            ingestion: ingestion.clone(),
            max_seq_len,
            max_supervised_tokens,
        }
    } else {
        kiln_train::prepare_sft_examples(
            req.examples.iter().cloned(),
            tokenizer,
            req.config.invalid_row_policy,
            "inline",
            None,
        )
        .map_err(|error| format!("ingest queued SFT examples: {error:#}"))?
    };
    let examples = prepared.examples.as_slice();
    let ingestion = &prepared.ingestion;

    if native_route_enabled {
        #[cfg(feature = "cuda")]
        {
            let native_route_env = native_route_env.unwrap_or("backend_native_training_policy");
            tracing::info!(
                job_id = %job_id,
                native_route_env,
                "backend native training route enabled - routing to cuda_native_sft_train"
            );
            return kiln_train::cuda_train::cuda_native_sft_train_to_with_checkpoint_root_and_ingestion(
                examples,
                ingestion,
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                output_adapter_dir,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
                gpu_step_coordination,
            )
            .map_err(|e| format!("{e:#}"));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let native_route_env = native_route_env.unwrap_or("backend_native_training_policy");
            tracing::warn!(
                job_id = %job_id,
                native_route_env,
                "backend native training route enabled but kiln-server was built without \
                 --features cuda - falling back to the default in-process SFT trainer (kt-tape)"
            );
        }
    }
    trainer::sft_train_to_with_checkpoint_root_and_ingestion(
        examples,
        ingestion,
        &req.config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(replay_ctx),
        gpu_step_coordination,
    )
    .map_err(|e| format!("{e:#}"))
}

#[allow(clippy::too_many_arguments)]
fn run_grpo(
    native_route_enabled: bool,
    native_route_env: Option<&'static str>,
    req: &GrpoRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    replay_ctx: trainer::ReplayContext,
    job_id: &str,
    gpu_step_coordination: Option<trainer::GpuStepCoordination>,
) -> std::result::Result<PathBuf, String> {
    if let Some(dataset_path) = req.dataset_path.as_deref() {
        if dataset_path.trim().is_empty() {
            return Err("GRPO dataset_path streaming requires a non-empty path".to_string());
        }
        if !req.groups.is_empty() {
            return Err(
                "GRPO request must use either groups or dataset_path, not both".to_string(),
            );
        }
        if native_route_enabled {
            #[cfg(feature = "cuda")]
            {
                let native_route_env = native_route_env.unwrap_or("backend_native_training_policy");
                tracing::info!(
                    job_id = %job_id,
                    dataset_path,
                    native_route_env,
                    "backend native training route enabled - routing streamed GRPO dataset to \
                     cuda_native_grpo_train_jsonl"
                );
                return kiln_train::cuda_train::cuda_native_grpo_train_jsonl_to_with_checkpoint_root(
                    std::path::Path::new(dataset_path),
                    &req.config,
                    model_config,
                    weights,
                    tokenizer,
                    adapter_dir,
                    output_adapter_dir,
                    adapter_dir,
                    adapter_name,
                    Some(progress_cb),
                    gpu_step_coordination.clone(),
                )
                .map_err(|e| format!("{e:#}"));
            }
            #[cfg(not(feature = "cuda"))]
            {
                let native_route_env = native_route_env.unwrap_or("backend_native_training_policy");
                tracing::warn!(
                    job_id = %job_id,
                    native_route_env,
                    "backend native training route enabled but kiln-server was built without \
                     --features cuda - falling back to the default in-process GRPO trainer (kt-tape)"
                );
            }
        }
        {
            tracing::info!(
                job_id = %job_id,
                dataset_path,
                "routing streamed GRPO dataset to generic trainer"
            );
            return trainer::grpo_train_jsonl_to_with_checkpoint_root(
                std::path::Path::new(dataset_path),
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                output_adapter_dir,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
                Some(replay_ctx),
                gpu_step_coordination,
            )
            .map_err(|e| format!("{e:#}"));
        }
    }
    if native_route_enabled {
        #[cfg(feature = "cuda")]
        {
            let native_route_env = native_route_env.unwrap_or("backend_native_training_policy");
            tracing::info!(
                job_id = %job_id,
                native_route_env,
                "backend native training route enabled - routing GRPO to cuda_native_grpo_train"
            );
            return kiln_train::cuda_train::cuda_native_grpo_train_to_with_checkpoint_root(
                &req.groups,
                &req.config,
                model_config,
                weights,
                tokenizer,
                adapter_dir,
                output_adapter_dir,
                adapter_dir,
                adapter_name,
                Some(progress_cb),
                gpu_step_coordination.clone(),
            )
            .map_err(|e| format!("{e:#}"));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let native_route_env = native_route_env.unwrap_or("backend_native_training_policy");
            tracing::warn!(
                job_id = %job_id,
                native_route_env,
                "backend native training route enabled but kiln-server was built without \
                 --features cuda - falling back to the default in-process GRPO trainer (kt-tape)"
            );
        }
    }
    trainer::grpo_train_to_with_checkpoint_root(
        &req.groups,
        &req.config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(replay_ctx),
        gpu_step_coordination,
    )
    .map_err(|e| format!("{e:#}"))
}

/// Build the receipt-facing descriptor for an admitted teacher binding.
fn registered_teacher_descriptor(
    spec: &crate::api::teachers::TeacherSpec,
) -> kiln_train::TeacherDescriptor {
    let identity = spec.identity.clone();
    let model_version_hash = identity
        .as_ref()
        .map(|identity| format!("sha256:{}", identity.content_revision()));
    kiln_train::TeacherDescriptor {
        alias: spec.alias.clone(),
        model_id: spec.model_id.clone(),
        model_version_hash,
        identity,
        snapshot_url: None,
    }
}

fn build_remote_teacher_for(
    spec: &crate::api::teachers::TeacherSpec,
    credentials: &crate::config::TeachersConfig,
    cache_root: Option<&std::path::Path>,
) -> std::result::Result<std::sync::Arc<dyn kiln_train::LogitSource>, String> {
    let config = crate::api::teachers::remote_teacher_config(spec, credentials)
        .map_err(|error| format!("resolve remote teacher credential: {error}"))?;
    let remote = kiln_train::RemoteTeacher::connect_pinned(config)
        .map_err(|error| format!("job-start remote teacher identity handshake: {error}"))?;
    tracing::info!(
        teacher = %spec.alias,
        identity_revision = %spec
            .identity
            .as_ref()
            .map(kiln_train::TeacherIdentityV1::content_revision)
            .unwrap_or_default(),
        cache_enabled = cache_root.is_some(),
        "verified pinned remote teacher at job start"
    );
    if let Some(cache_root) = cache_root {
        let cached =
            kiln_train::CachedLogitSource::new(kiln_train::LogitCache::new(cache_root), remote)
                .map_err(|error| format!("bind remote teacher cache to identity: {error}"))?;
        Ok(std::sync::Arc::new(cached))
    } else {
        Ok(std::sync::Arc::new(remote))
    }
}

fn materialize_remote_teacher_for_off_policy(
    operation: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
    config: &kiln_train::opd::OpdConfig,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    prepared_remote_teacher: Option<std::sync::Arc<dyn kiln_train::LogitSource>>,
) -> std::result::Result<std::sync::Arc<dyn kiln_train::LogitSource>, String> {
    let remote = prepared_remote_teacher
        .ok_or_else(|| format!("{operation}: remote teacher was not verified at job start"))?;
    let fixture = kiln_train::opd::materialize_verified_off_policy_teacher(
        prompts, config, tokenizer, remote,
    )
    .map_err(|error| format!("{operation}: prefetch remote teacher logits: {error:#}"))?;
    tracing::info!(
        operation,
        prompts = prompts.len(),
        identity_revision = %fixture
            .authoritative_teacher_identity()
            .map(kiln_train::TeacherIdentityV1::content_revision)
            .unwrap_or_default(),
        "materialized remote teacher before bounded GPU phases"
    );
    Ok(std::sync::Arc::new(fixture))
}

fn resolved_opd_seed(
    output_dir: &std::path::Path,
    adapter_name: &str,
) -> std::result::Result<u64, String> {
    let receipt = kiln_train::TrainReceipt::read_from_adapter_dir(output_dir)
        .map_err(|error| format!("read OPD train receipt seed: {error:#}"))?
        .ok_or_else(|| "OPD trainer completed without train_receipt.json".to_string())?;
    if receipt.status != kiln_train::TrainReceiptStatus::Success {
        return Err("OPD trainer returned a non-success train receipt".to_string());
    }
    if receipt.adapter_name != adapter_name || receipt.hyperparameters.mode != "opd" {
        return Err(format!(
            "OPD train receipt identity mismatch: expected adapter {adapter_name:?} in mode \"opd\", got adapter {:?} in mode {:?}",
            receipt.adapter_name, receipt.hyperparameters.mode
        ));
    }
    receipt.hyperparameters.seed.ok_or_else(|| {
        "OPD train receipt is missing the resolved effective seed used by the optimizer".to_string()
    })
}

fn combine_operation_and_cleanup<T>(
    operation: std::result::Result<T, String>,
    cleanup: std::result::Result<(), String>,
    cleanup_label: &str,
) -> std::result::Result<T, String> {
    match (operation, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
        (Err(error), Err(cleanup_error)) => Err(format!(
            "{error}; {cleanup_label} also failed: {cleanup_error}"
        )),
    }
}

fn release_teacher_lora(
    teacher_lora: Option<kiln_model::lora_loader::LoraWeights>,
    weights: &kiln_model::forward::GpuWeights,
    gpu_step_coordination: &trainer::GpuStepCoordination,
    phase: &'static str,
) -> std::result::Result<(), String> {
    let Some(teacher_lora) = teacher_lora else {
        return Ok(());
    };
    let device = weights.embed_tokens.device();
    let backend = kiln_model::backend::for_device_kt(&device);
    gpu_step_coordination
        .run_gpu_phase(&*backend, "OPD", phase, || {
            drop(teacher_lora);
            Ok(())
        })
        .map_err(|error| format!("{phase}: {error:#}"))
}

fn release_opd_teacher(
    teacher: std::sync::Arc<dyn kiln_train::LogitSource>,
    weights: &kiln_model::forward::GpuWeights,
    gpu_step_coordination: &trainer::GpuStepCoordination,
    phase: &'static str,
) -> std::result::Result<(), String> {
    let device = weights.embed_tokens.device();
    let backend = kiln_model::backend::for_device_kt(&device);
    gpu_step_coordination
        .run_gpu_phase(&*backend, "OPD", phase, || {
            drop(teacher);
            Ok(())
        })
        .map_err(|error| format!("{phase}: {error:#}"))
}

#[allow(clippy::too_many_arguments)]
fn run_opd(
    req: &OpdRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    teacher_spec: &crate::api::teachers::TeacherSpec,
    prepared_remote_teacher: Option<std::sync::Arc<dyn kiln_train::LogitSource>>,
    job_id: &str,
    gpu_step_coordination: trainer::GpuStepCoordination,
) -> std::result::Result<PathBuf, String> {
    req.config
        .validate_runtime_contract()
        .map_err(|error| format!("OPD request has unsupported configuration: {error:#}"))?;
    if req.prompts.is_empty() && req.dataset_path.is_none() {
        return Err("OPD request must include at least one prompt or a dataset_path".into());
    }
    if req.dataset_path.is_some() && !req.prompts.is_empty() {
        return Err("OPD request must use either prompts or dataset_path, not both".into());
    }

    let mut dataset_teacher: Option<std::sync::Arc<dyn kiln_train::LogitSource>> = None;
    let mut dataset_summary: Option<kiln_train::OffPolicyDistillationSummary> = None;
    let mut dataset_source_sha256: Option<String> = None;
    let mut dataset_identity: Option<kiln_train::TeacherIdentityV1> = None;
    let mut owned_prompts: Option<Vec<kiln_train::opd::OpdPrompt>> = None;
    if let Some(path) = req.dataset_path.as_deref() {
        // `agent_traces:<filter>` selectors resolve to live prompt
        // scaffolds from the §10.3 trace index — the student re-rolls
        // them on-policy against the registered teacher (this is the
        // `/v1/agent/self_improve` data path). Plain file paths remain
        // pre-scored off-policy teacher JSONL.
        if crate::dataset_resolve::is_agent_traces_selector(path) {
            let resolved = crate::dataset_resolve::resolve_agent_trace_prompts(
                adapter_dir,
                path,
                crate::recent_requests::now_unix_ms() as i64,
            )
            .map_err(|e| format!("resolve OPD dataset_path {path:?}: {e}"))?;
            tracing::info!(
                job_id = %job_id,
                dataset_path = %path,
                prompts = resolved.len(),
                "resolved agent-trace selector into OPD prompts"
            );
            owned_prompts = Some(resolved);
        } else {
            if !matches!(
                req.config.training_mode,
                kiln_train::opd::OpdTrainingMode::OffPolicy
            ) {
                return Err(
                "OPD dataset_path is only supported with config.training_mode = \"off_policy\" \
                 (or use an `agent_traces:` selector for on-policy training on pi sessions)"
                    .into(),
            );
            }
            let loaded = kiln_train::load_off_policy_distillation_dataset(path)
                .map_err(|e| format!("load off-policy OPD dataset_path {path:?}: {e:#}"))?;
            let contains_numeric_teacher_logits = loaded.examples.iter().any(|example| {
                example
                    .teacher_tokens
                    .iter()
                    .any(|token| token.logprob.is_some() || !token.top_logprobs.is_empty())
            });
            let manifest_identity = loaded
                .manifest
                .as_ref()
                .map(|manifest| manifest.teacher_identity().clone());
            if contains_numeric_teacher_logits && manifest_identity.is_none() {
                return Err(format!(
                    "off-policy OPD dataset_path {path:?} contains numeric teacher logits but has no canonical {} first record",
                    kiln_train::OFF_POLICY_DISTILLATION_MANIFEST_SCHEMA_V1
                ));
            }
            if let Some(identity) = manifest_identity.as_ref() {
                let expected = teacher_spec.identity.as_ref().ok_or_else(|| {
                    format!(
                        "off-policy OPD dataset_path {path:?} declares teacher revision sha256:{}, but registered teacher {:?} has no authoritative identity",
                        identity.content_revision(),
                        teacher_spec.alias
                    )
                })?;
                if identity != expected {
                    return Err(format!(
                        "off-policy OPD dataset_path {path:?} teacher revision sha256:{} does not match pinned registered teacher revision sha256:{}",
                        identity.content_revision(),
                        expected.content_revision()
                    ));
                }
            }
            let prepared = kiln_train::prepare_off_policy_distillation_dataset_with_identity(
                &loaded.examples,
                tokenizer,
                req.teacher.clone(),
                manifest_identity.clone(),
                model_config.vocab_size,
                req.config.top_k,
                req.config.objective,
                req.config.echo.as_ref(),
            )
            .map_err(|e| format!("prepare off-policy OPD dataset_path {path:?}: {e:#}"))?;
            tracing::info!(
                job_id = %job_id,
                dataset_path = %path,
                examples = prepared.summary.examples,
                action_tokens = prepared.summary.action_tokens,
                env_tokens = prepared.summary.env_tokens,
                objective = ?prepared.summary.objective,
                echo_combined = prepared.summary.echo_combined,
                "loaded off-policy OPD teacher JSONL dataset"
            );
            dataset_teacher = Some(std::sync::Arc::new(prepared.teacher));
            dataset_summary = Some(prepared.summary);
            dataset_source_sha256 = Some(loaded.source_sha256);
            dataset_identity = manifest_identity;
            owned_prompts = Some(prepared.prompts);
        }
    }
    let prompts: &[kiln_train::opd::OpdPrompt] =
        owned_prompts.as_deref().unwrap_or(req.prompts.as_slice());

    for (i, prompt) in prompts.iter().enumerate() {
        if prompt.messages.is_empty() {
            return Err(format!("OPD prompt {i} has no messages"));
        }
    }

    // A live remote source must never cross the GPU coordination boundary.
    // Fixed off-policy rows are fetched eagerly while inference still holds
    // read access; the trainer receives only an in-memory, identity-bound
    // fixture. Pre-scored datasets already supply their own fixture.
    let materialized_remote_teacher = if dataset_teacher.is_none()
        && matches!(teacher_spec.kind, crate::api::teachers::TeacherKind::Remote)
    {
        Some(materialize_remote_teacher_for_off_policy(
            "OPD",
            prompts,
            &req.config,
            tokenizer,
            prepared_remote_teacher,
        )?)
    } else {
        None
    };

    tracing::info!(
        job_id = %job_id,
        teacher = %req.teacher,
        loss = ?req.config.loss,
        top_k = req.config.top_k,
        samples_per_prompt = req.config.samples_per_prompt,
        num_prompts = prompts.len(),
        dataset_path = req.dataset_path.as_deref().unwrap_or(""),
        "OPD training started"
    );

    // Resolve the teacher alias against the §3.2 registry, then build
    // the concrete LogitSource:
    //   • Fixture → DeterministicUniformLogitSource (synthetic).
    //   • Local   → run the loaded model forward on each prompt and
    //               populate a FixtureLogitSource with the real
    //               top-K teacher logprobs (§3.2 in-process local
    //               teacher).
    //   • Remote  → job-start reverified RemoteTeacher, optionally wrapped
    //               by the identity-bound v3 cache.
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> =
        if let Some(teacher) = dataset_teacher {
            teacher
        } else {
            let spec = teacher_spec;
            let resolved_vocab = spec.vocab_size.unwrap_or(model_config.vocab_size);
            let resolved_max_top_k = spec.max_top_k.unwrap_or(req.config.top_k);
            match spec.kind {
                crate::api::teachers::TeacherKind::Fixture => {
                    std::sync::Arc::new(kiln_train::DeterministicUniformLogitSource::new(
                        spec.alias.clone(),
                        resolved_vocab,
                        resolved_max_top_k.max(req.config.top_k),
                    ))
                }
                crate::api::teachers::TeacherKind::Local => build_local_teacher_for(
                    &spec,
                    prompts,
                    tokenizer,
                    weights,
                    model_config,
                    adapter_dir,
                    req.config.top_k,
                    req.config.training_mode,
                    &gpu_step_coordination,
                )?,
                crate::api::teachers::TeacherKind::Remote => materialized_remote_teacher
                    .clone()
                    .ok_or_else(|| "OPD remote teacher was not materialized".to_string())?,
            }
        };

    let trainer_progress_cb: trainer::ProgressCallback = progress_cb;

    let train_result = kiln_train::opd::opd_train_to_with_checkpoint_root(
        prompts,
        &req.config,
        model_config,
        weights,
        tokenizer,
        teacher.clone(),
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        Some(trainer_progress_cb),
        Some(gpu_step_coordination.clone()),
    )
    .map_err(|e| format!("opd_train failed: {e:#}"));
    let teacher_release = release_opd_teacher(
        teacher,
        weights,
        &gpu_step_coordination,
        "direct teacher release",
    );
    let output_dir =
        combine_operation_and_cleanup(train_result, teacher_release, "direct OPD teacher release")?;

    if let (Some(path), Some(source_sha256)) =
        (req.dataset_path.as_deref(), dataset_source_sha256.as_ref())
    {
        let mut receipt = kiln_train::TrainReceipt::read_from_adapter_dir(&output_dir)
            .map_err(|error| format!("read OPD dataset provenance receipt: {error:#}"))?
            .ok_or_else(|| {
                "OPD trainer completed without its required train_receipt.json".to_string()
            })?;
        receipt.training_data = kiln_train::train_receipt::TrainingDataReceipt {
            source: "jsonl_off_policy_opd_teacher".to_string(),
            path: Some(path.to_string()),
            sha256: Some(source_sha256.clone()),
        };
        let opd = receipt.opd.as_mut().ok_or_else(|| {
            "OPD trainer receipt is missing its required OPD provenance section".to_string()
        })?;
        opd.teacher_id = Some(req.teacher.clone());
        receipt
            .write_to_adapter_dir(&output_dir)
            .map_err(|error| format!("persist OPD dataset provenance receipt: {error:#}"))?;
    }

    // §8.11 legacy audit receipt for this OPD adapter.
    let seed = resolved_opd_seed(&output_dir, adapter_name)?;
    let hyperparameters = serde_json::to_value(&req.config)
        .map_err(|error| format!("serialize OPD receipt hyperparameters: {error}"))?;
    let teacher_descriptor = if dataset_summary.is_some() {
        let model_version_hash = dataset_identity
            .as_ref()
            .map(|identity| format!("sha256:{}", identity.content_revision()));
        kiln_train::TeacherDescriptor {
            alias: teacher_spec.alias.clone(),
            model_id: teacher_spec.model_id.clone(),
            model_version_hash,
            identity: dataset_identity,
            snapshot_url: None,
        }
    } else {
        registered_teacher_descriptor(teacher_spec)
    };
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "opd", seed)
        .with_teacher(teacher_descriptor)
        .with_hyperparameters(hyperparameters);
    receipt
        .write_to_adapter_dir(&output_dir)
        .map_err(|error| format!("persist OPD provenance receipt: {error:#}"))?;

    if let Some(summary) = dataset_summary {
        tracing::info!(
            job_id = %job_id,
            examples = summary.examples,
            action_tokens = summary.action_tokens,
            env_tokens = summary.env_tokens,
            objective = ?summary.objective,
            "off-policy OPD dataset training summary"
        );
    }

    Ok(output_dir)
}

/// Load a declared merge source without permitting a base-model fallback.
fn load_declared_merge_source_lora(
    source_adapter: &str,
    src_dir: &std::path::Path,
    num_layers: usize,
    device: kiln_tensor::Device,
) -> std::result::Result<kiln_model::lora_loader::LoraWeights, String> {
    kiln_model::lora_loader::LoraWeights::load(src_dir, num_layers, device).map_err(|e| {
        format!(
            "distill_merge: declared source adapter '{source_adapter}' failed to load from {}: {e}",
            src_dir.display()
        )
    })
}

fn tokenize_teacher_prompts(
    operation: &str,
    source_id: &str,
    prompts: &[kiln_train::opd::OpdPrompt],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> std::result::Result<Vec<(Vec<u32>, Vec<usize>)>, String> {
    if prompts.is_empty() {
        return Err(format!(
            "{operation}: source '{source_id}' has no prompts to score"
        ));
    }

    let mut tokenized = Vec::with_capacity(prompts.len());
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let ex = kiln_train::SftExample {
            messages: prompt.messages.clone(),
        };
        let (tokens, label_mask) = kiln_train::trainer::tokenize_for_training(&ex, tokenizer)
            .map_err(|e| {
                format!(
                    "{operation}: source '{source_id}' prompt {prompt_index} failed to tokenize: {e:#}"
                )
            })?;
        let active: Vec<usize> = label_mask
            .iter()
            .enumerate()
            .filter_map(|(position, &is_active)| is_active.then_some(position))
            .collect();
        if active.is_empty() {
            return Err(format!(
                "{operation}: source '{source_id}' prompt {prompt_index} produced no active assistant tokens"
            ));
        }
        tokenized.push((tokens, active));
    }
    Ok(tokenized)
}

/// Build the §3.4 multi-tenant merge teacher.
///
/// Loads each source LoRA, runs the model forward over that source's
/// prompts with the source LoRA applied (so the logits reflect "what
/// the model behaves like when wearing that LoRA"), and stashes the
/// top-K teacher logprobs into a single `FixtureLogitSource` keyed by
/// (exact token sequence, position).
///
/// Each source contributes only its own prompts' entries, so the
/// trainer's `opd_step_loss` call queries the *correct* source's
/// teacher when iterating each prompt — no per-step LoRA swap, no
/// multi-tenant inference server needed.
///
/// Per-source `weight` (a `DistillMergeSource` field) is not yet
/// applied — the unified fixture treats every (source, prompt) entry
/// equally. Weighted loss aggregation is filed as a §3.4 follow-up.
/// A declared source adapter and every one of its prompts are required:
/// preparation fails closed rather than substituting the base model or
/// silently scoring only a subset of the source dataset.
#[allow(clippy::too_many_arguments)]
fn build_multi_tenant_merge_teacher(
    teacher_id: &str,
    per_source: &[(
        kiln_train::DistillMergeSource,
        Vec<kiln_train::opd::OpdPrompt>,
    )],
    adapter_dir: &std::path::Path,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    top_k: usize,
    gpu_step_coordination: &trainer::GpuStepCoordination,
) -> std::result::Result<kiln_train::logit_source::FixtureLogitSource, String> {
    let mut unified = kiln_train::logit_source::FixtureLogitSource::uniform_topk(
        teacher_id.to_string(),
        model_config.vocab_size,
        top_k,
    );
    let backend_device = weights.embed_tokens.device();
    let backend = kiln_model::backend::for_device_kt(&backend_device);
    for (source, prompts) in per_source {
        let tokenized =
            tokenize_teacher_prompts("distill_merge", &source.adapter, prompts, tokenizer)?;
        // The source identity is the declared LoRA. Loading it is part of the
        // teacher contract, so failure cannot degrade to base-model scoring.
        let src_dir = adapter_dir.join(&source.adapter);
        let device = weights.embed_tokens.device().clone();
        let teacher_lora = gpu_step_coordination
            .run_gpu_phase(&*backend, "OPD", "merge teacher adapter load", || {
                load_declared_merge_source_lora(
                    &source.adapter,
                    &src_dir,
                    model_config.num_layers,
                    device,
                )
                .map_err(anyhow::Error::msg)
            })
            .map_err(|error| format!("distill_merge teacher adapter load: {error:#}"))?;

        let source_fixture = kiln_train::opd::build_local_teacher_fixture_with_coordination(
            format!("{teacher_id}:{}", source.adapter),
            &tokenized,
            weights,
            model_config,
            Some(&teacher_lora),
            top_k,
            None,
            Some(gpu_step_coordination),
        )
        .map_err(|e| format!("build_local_teacher_fixture for {}: {e:#}", source.adapter));
        let teacher_release = release_teacher_lora(
            Some(teacher_lora),
            weights,
            gpu_step_coordination,
            "merge teacher adapter release",
        );
        let source_fixture = combine_operation_and_cleanup(
            source_fixture,
            teacher_release,
            "distill_merge teacher adapter release",
        )?;

        // Drain entries from source_fixture into unified.
        for (tokens, active_positions) in &tokenized {
            let logits_rows = kiln_train::logit_source::target_token_positions_to_logits_rows(
                teacher_id,
                tokens.len(),
                active_positions,
            )
            .map_err(|e| format!("distill_merge invalid action-token positions: {e}"))?;
            for &logits_row in &logits_rows {
                let batch = source_fixture
                    .fetch_logprobs(tokens, &[logits_row], Some(top_k))
                    .map_err(|e| format!("distill_merge fixture read failed: {e}"))?;
                let kiln_train::LogprobBatch::TopK(topk) = batch else {
                    return Err("distill_merge top-K fixture returned full-vocab logits".into());
                };
                unified
                    .insert(tokens, logits_row, topk.indices, topk.logprobs)
                    .map_err(|e| {
                        format!(
                            "distill_merge conflicting exact-sequence fixture row at {logits_row}: {e}"
                        )
                    })?;
            }
        }
    }
    Ok(unified)
}

fn validate_self_distill_target_alignment(
    prompt_index: usize,
    student_tokens: &[u32],
    student_targets: &[usize],
    teacher_tokens: &[u32],
    teacher_targets: &[usize],
) -> std::result::Result<(), String> {
    if student_targets.len() != teacher_targets.len() {
        return Err(format!(
            "self-distill prompt {prompt_index} has {} student action tokens but {} teacher action tokens",
            student_targets.len(),
            teacher_targets.len()
        ));
    }
    kiln_train::logit_source::target_token_positions_to_logits_rows(
        "self-distill-student",
        student_tokens.len(),
        student_targets,
    )
    .map_err(|e| format!("self-distill prompt {prompt_index} student targets: {e}"))?;
    kiln_train::logit_source::target_token_positions_to_logits_rows(
        "self-distill-teacher",
        teacher_tokens.len(),
        teacher_targets,
    )
    .map_err(|e| format!("self-distill prompt {prompt_index} teacher targets: {e}"))?;
    for (pair_index, (&student_target, &teacher_target)) in student_targets
        .iter()
        .zip(teacher_targets.iter())
        .enumerate()
    {
        let student_token = student_tokens[student_target];
        let teacher_token = teacher_tokens[teacher_target];
        if student_token != teacher_token {
            return Err(format!(
                "self-distill prompt {prompt_index} action pair {pair_index} differs: student token {student_token} at {student_target}, teacher token {teacher_token} at {teacher_target}"
            ));
        }
    }
    Ok(())
}

type TokenizedSelfDistillPrompts = Vec<(Vec<u32>, Vec<usize>)>;

fn validate_self_distill_conditioning(
    mode: kiln_train::SelfDistillMode,
    prompt_count: usize,
    ground_truth: Option<&[String]>,
    documents: Option<&[String]>,
) -> std::result::Result<(), String> {
    use kiln_train::SelfDistillMode;

    if prompt_count == 0 {
        return Err("distill_self: prompts resolved to zero items".into());
    }

    match mode {
        SelfDistillMode::GroundTruthConditioning => {
            let answers = ground_truth.ok_or_else(|| {
                "distill_self GroundTruthConditioning: ground_truth is required (one non-empty entry per prompt)"
                    .to_string()
            })?;
            if answers.len() != prompt_count {
                return Err(format!(
                    "distill_self GroundTruthConditioning: ground_truth.len() ({}) != prompts.len() ({prompt_count})",
                    answers.len()
                ));
            }
            if let Some((prompt_index, _)) = answers
                .iter()
                .enumerate()
                .find(|(_, answer)| answer.trim().is_empty())
            {
                return Err(format!(
                    "distill_self GroundTruthConditioning: ground_truth[{prompt_index}] must be non-empty"
                ));
            }
        }
        SelfDistillMode::DocumentAsPi => {
            let per_prompt_documents = documents.ok_or_else(|| {
                "distill_self DocumentAsPi: documents is required (one non-empty entry per prompt)"
                    .to_string()
            })?;
            if per_prompt_documents.len() != prompt_count {
                return Err(format!(
                    "distill_self DocumentAsPi: documents.len() ({}) != prompts.len() ({prompt_count})",
                    per_prompt_documents.len()
                ));
            }
            if let Some((prompt_index, _)) = per_prompt_documents
                .iter()
                .enumerate()
                .find(|(_, document)| document.trim().is_empty())
            {
                return Err(format!(
                    "distill_self DocumentAsPi: documents[{prompt_index}] must be non-empty"
                ));
            }
        }
        SelfDistillMode::Conciseness => {}
        SelfDistillMode::ReverseTeacher => {
            return Err(
                "self-distill reverse_teacher is unsupported: negated logprobs are not a probability distribution"
                    .into(),
            );
        }
    }
    Ok(())
}

fn prepare_self_distill_prompts(
    prompts: &[kiln_train::opd::OpdPrompt],
    mode: kiln_train::SelfDistillMode,
    ground_truth: Option<&[String]>,
    documents: Option<&[String]>,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
) -> std::result::Result<(TokenizedSelfDistillPrompts, TokenizedSelfDistillPrompts), String> {
    use kiln_train::{ChatMessage, SelfDistillMode};

    validate_self_distill_conditioning(mode, prompts.len(), ground_truth, documents)?;

    let mut student_active = Vec::with_capacity(prompts.len());
    let mut teacher_only = Vec::with_capacity(prompts.len());
    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let student_ex = kiln_train::SftExample {
            messages: prompt.messages.clone(),
        };
        let (student_tokens, student_label_mask) =
            kiln_train::trainer::tokenize_for_training(&student_ex, tokenizer).map_err(|e| {
                format!("self-distill prompt {prompt_index} student tokenization failed: {e:#}")
            })?;
        let student_targets: Vec<usize> = student_label_mask
            .iter()
            .enumerate()
            .filter_map(|(position, &is_active)| is_active.then_some(position))
            .collect();
        if student_targets.is_empty() {
            return Err(format!(
                "self-distill prompt {prompt_index} student tokenization produced no active assistant tokens"
            ));
        }

        let mut teacher_messages = Vec::new();
        match mode {
            SelfDistillMode::GroundTruthConditioning => {
                let answer = ground_truth.and_then(|values| values.get(prompt_index)).ok_or_else(
                    || {
                        format!(
                            "distill_self GroundTruthConditioning: missing ground_truth[{prompt_index}]"
                        )
                    },
                )?;
                teacher_messages.push(ChatMessage::new(
                    "system",
                    format!(
                        "Privileged context (visible only to the teacher): the correct answer is: {answer}"
                    ),
                ));
            }
            SelfDistillMode::Conciseness => {
                teacher_messages.push(ChatMessage::new(
                    "system",
                    "Privileged context (visible only to the teacher): respond with maximal concision; trim every unnecessary word; never explain reasoning unless explicitly asked.",
                ));
            }
            SelfDistillMode::DocumentAsPi => {
                let document = documents
                    .and_then(|values| values.get(prompt_index))
                    .ok_or_else(|| {
                        format!("distill_self DocumentAsPi: missing documents[{prompt_index}]")
                    })?;
                teacher_messages.push(ChatMessage::new(
                    "system",
                    format!(
                        "Privileged context (visible only to the teacher) — use the following retrieved document to answer:\n\n{document}"
                    ),
                ));
            }
            SelfDistillMode::ReverseTeacher => {
                return Err(
                    "self-distill reverse_teacher is unsupported: negated logprobs are not a probability distribution"
                        .into(),
                );
            }
        }
        teacher_messages.extend(prompt.messages.iter().cloned());
        let teacher_ex = kiln_train::SftExample {
            messages: teacher_messages,
        };
        let (teacher_tokens, teacher_label_mask) =
            kiln_train::trainer::tokenize_for_training(&teacher_ex, tokenizer).map_err(|e| {
                format!("self-distill prompt {prompt_index} teacher tokenization failed: {e:#}")
            })?;
        let teacher_targets: Vec<usize> = teacher_label_mask
            .iter()
            .enumerate()
            .filter_map(|(position, &is_active)| is_active.then_some(position))
            .collect();
        if teacher_targets.is_empty() {
            return Err(format!(
                "self-distill prompt {prompt_index} teacher tokenization produced no active assistant tokens"
            ));
        }
        validate_self_distill_target_alignment(
            prompt_index,
            &student_tokens,
            &student_targets,
            &teacher_tokens,
            &teacher_targets,
        )?;
        student_active.push((student_tokens, student_targets));
        teacher_only.push((teacher_tokens, teacher_targets));
    }
    Ok((student_active, teacher_only))
}

/// Build the §3.12 privileged-information self-teacher.
///
/// For each prompt and the chosen `SelfDistillMode`, we construct a
/// teacher-side prompt that *includes* the privileged context (a
/// system message carrying ground-truth answers, a "be concise"
/// instruction, or that prompt's retrieved document). The teacher's forward pass
/// runs against that shaped prompt; we then transplant the resulting
/// top-K logprobs back onto the *student's* (un-shaped) token stream
/// by aligning active assistant positions. The student then distils
/// against logits that "knew" the privileged context — Lu's PI
/// recipe (CRISP, OPSD, GATES, RLRT) made concrete.
///
/// `ReverseTeacher` is rejected: negating logprobs does not produce a valid
/// probability distribution, so that mode needs a distinct loss objective.
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
    gpu_step_coordination: &trainer::GpuStepCoordination,
) -> std::result::Result<kiln_train::logit_source::FixtureLogitSource, String> {
    // Tokenization and target alignment complete before any forward so a bad
    // prompt cannot create a partial fixture or silently change the dataset.
    let (student_active, teacher_only) =
        prepare_self_distill_prompts(prompts, mode, ground_truth, documents, tokenizer)?;

    // Run the teacher forwards on the shaped teacher_only sequences
    // and build a FixtureLogitSource keyed by the exact *student* token
    // sequence so opd_train's queries match. We compute the teacher's
    // top-K at teacher_only positions, then re-insert under the
    // exact student sequence at the student positions — same logprob
    // values, different key.
    let teacher_fixture = kiln_train::opd::build_local_teacher_fixture_with_coordination(
        teacher_id.to_string(),
        &teacher_only,
        weights,
        model_config,
        None,
        top_k,
        None,
        Some(gpu_step_coordination),
    )
    .map_err(|e| format!("self-distill local-teacher forward: {e:#}"))?;

    let mut student_fixture = kiln_train::logit_source::FixtureLogitSource::uniform_topk(
        teacher_id.to_string(),
        model_config.vocab_size,
        top_k,
    );

    // Transplant teacher-key entries to student-key entries.
    for ((s_tokens, s_active), (t_tokens, t_active)) in
        student_active.iter().zip(teacher_only.iter())
    {
        let student_rows = kiln_train::logit_source::target_token_positions_to_logits_rows(
            teacher_id,
            s_tokens.len(),
            s_active,
        )
        .map_err(|e| format!("self-distill invalid student action positions: {e}"))?;
        let teacher_rows = kiln_train::logit_source::target_token_positions_to_logits_rows(
            teacher_id,
            t_tokens.len(),
            t_active,
        )
        .map_err(|e| format!("self-distill invalid teacher action positions: {e}"))?;
        for (&student_row, &teacher_row) in student_rows.iter().zip(teacher_rows.iter()) {
            let batch = teacher_fixture
                .fetch_logprobs(t_tokens, &[teacher_row], Some(top_k))
                .map_err(|e| format!("self-distill teacher fixture read failed: {e}"))?;
            let kiln_train::LogprobBatch::TopK(topk) = batch else {
                return Err("self-distill top-K fixture returned full-vocab logits".into());
            };
            student_fixture
                .insert(s_tokens, student_row, topk.indices, topk.logprobs)
                .map_err(|e| {
                    format!(
                        "self-distill conflicting exact-sequence fixture row at {student_row}: {e}"
                    )
                })?;
        }
    }

    Ok(student_fixture)
}

/// Build a `FixtureLogitSource` populated from a local model forward
/// pass — the §3.2 in-process LocalTeacher made concrete. Each prompt
/// is tokenized like SFT (chat template with active assistant tokens
/// marked), the model is run forward once per prompt — wearing the
/// spec's `adapter` LoRA when one is registered, base model otherwise —
/// and the top-K logprobs at active positions are inserted into the
/// fixture keyed by the exact token sequence. Off-policy precomputation requires every
/// prompt to tokenize with active assistant targets; it never changes the
/// requested dataset by skipping an invalid prompt.
#[allow(clippy::too_many_arguments)]
fn build_local_teacher_for(
    spec: &crate::api::teachers::TeacherSpec,
    prompts: &[kiln_train::opd::OpdPrompt],
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    weights: &kiln_model::forward::GpuWeights,
    model_config: &kiln_core::config::ModelConfig,
    adapter_dir: &std::path::Path,
    top_k: usize,
    training_mode: kiln_train::opd::OpdTrainingMode,
    gpu_step_coordination: &trainer::GpuStepCoordination,
) -> std::result::Result<std::sync::Arc<dyn kiln_train::logit_source::LogitSource>, String> {
    let pinned_identity = spec.identity.as_ref().ok_or_else(|| {
        format!(
            "local teacher '{}' has no authoritative pinned identity; delete and re-register it",
            spec.alias
        )
    })?;
    let on_policy = matches!(training_mode, kiln_train::opd::OpdTrainingMode::OnPolicy);
    let prompts_and_active = if on_policy {
        None
    } else {
        Some(tokenize_teacher_prompts(
            "local-teacher",
            &spec.alias,
            prompts,
            tokenizer,
        )?)
    };
    // The teacher's LoRA. A spec that names an adapter MUST wear it —
    // the silent fall-through to `None` here is what made
    // `distill_refresh` toward a prior self and `self_improve`'s
    // judge-as-teacher distil toward the bare base model while claiming
    // otherwise. Registration validates existence, so a load failure
    // here is a hard error, not a quiet downgrade.
    let teacher_lora = match spec.adapter.as_deref() {
        Some(name) => {
            let dir = adapter_dir.join(name);
            let device = weights.embed_tokens.device().clone();
            let pinned_adapter = pinned_identity.adapter().ok_or_else(|| {
                format!(
                    "local teacher '{}' names adapter '{name}' but its pinned identity is the bare base model",
                    spec.alias
                )
            })?;
            if pinned_adapter.name() != name {
                return Err(format!(
                    "local teacher '{}' names adapter '{name}' but its pinned identity names '{}'",
                    spec.alias,
                    pinned_adapter.name()
                ));
            }
            let expected_source = kiln_model::lora_loader::LoraSourceIdentity::new(
                pinned_adapter.weights_sha256(),
                pinned_adapter.config_sha256(),
            )
            .map_err(|e| {
                format!(
                    "local teacher '{}' has invalid adapter identity: {e:#}",
                    spec.alias
                )
            })?;
            let backend = kiln_model::backend::for_device_kt(&device);
            let load = || {
                kiln_model::lora_loader::LoraWeights::load_pinned(
                    &dir,
                    model_config.num_layers,
                    device,
                    &expected_source,
                )
            };
            let loaded = gpu_step_coordination
                .run_gpu_phase(&*backend, "OPD", "local teacher adapter load", load)
            .map_err(|e| {
                format!(
                    "teacher '{}' is pinned to adapter '{name}', but the exact registered content \
                     could not be loaded from {}: {e:#}; re-register after any adapter rewrite",
                    spec.alias,
                    dir.display()
                )
            })?;
            Some(loaded)
        }
        None => {
            if let Some(adapter) = pinned_identity.adapter() {
                return Err(format!(
                    "local teacher '{}' is configured for the bare base model but its pinned identity names adapter '{}'",
                    spec.alias,
                    adapter.name()
                ));
            }
            None
        }
    };

    // ON-POLICY self-distillation (#31): the student generates fresh rollouts, so
    // the teacher must score ARBITRARY token sequences live — a fixed-sequence
    // fixture would miss every rollout. Return a LiveLocalTeacher that holds a
    // cheap (Arc-backed) clone of the loaded model and runs a detached forward on
    // demand.
    if on_policy {
        let construct = || {
            kiln_train::opd::LiveLocalTeacher::new(
                spec.alias.clone(),
                weights.clone(),
                model_config.clone(),
                teacher_lora,
                pinned_identity.clone(),
                top_k,
            )
        };
        let device = weights.embed_tokens.device();
        let backend = kiln_model::backend::for_device_kt(&device);
        let teacher = gpu_step_coordination
            .run_gpu_phase(&*backend, "OPD", "live local teacher ownership", construct)
            .map_err(|e| format!("construct pinned live local teacher: {e:#}"))?;
        return Ok(std::sync::Arc::new(teacher));
    }

    // OFF-POLICY: the assistant turns are fixed, so pre-compute the fixture keyed
    // by exact sequence (cheaper — one forward per prompt up front).
    let prompts_and_active = prompts_and_active.expect("off-policy prompts were tokenized");
    let fixture = kiln_train::opd::build_local_teacher_fixture_with_coordination(
        spec.alias.clone(),
        &prompts_and_active,
        weights,
        model_config,
        teacher_lora.as_ref(),
        top_k,
        spec.tokenizer_hash.clone(),
        Some(gpu_step_coordination),
    )
    .map_err(|e| format!("build_local_teacher_fixture failed: {e:#}"));
    let teacher_release = release_teacher_lora(
        teacher_lora,
        weights,
        gpu_step_coordination,
        "local teacher adapter release",
    );
    let fixture =
        combine_operation_and_cleanup(fixture, teacher_release, "local teacher adapter release")?;
    let fixture = fixture
        .with_authoritative_identity(pinned_identity.clone())
        .map_err(|e| format!("bind local teacher fixture to pinned identity: {e}"))?;
    Ok(std::sync::Arc::new(fixture))
}

/// `/v1/distill/refresh` runtime — §3.6 continual-learning recipe.
///
/// Two-phase pipeline (orchestrating existing primitives):
/// 1. **Mid-train** on `new_data` mixed with `background_chat`. Uses
///    SFT under the hood — same `trainer::sft_train` path SFT uses.
/// 2. **OPD-recover** against `behavioural_teacher`. Uses OPD on
///    Tulu3-flavoured prompts.
#[allow(clippy::too_many_arguments)]
fn run_distill_refresh(
    req: &DistillRefreshRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    teacher_spec: &crate::api::teachers::TeacherSpec,
    prepared_remote_teacher: Option<std::sync::Arc<dyn kiln_train::LogitSource>>,
    dataset_registry: Option<&crate::eval::DatasetRegistry>,
    job_id: &str,
    gpu_step_coordination: trainer::GpuStepCoordination,
) -> std::result::Result<PathBuf, String> {
    req.config
        .validate_runtime_contract()
        .map_err(|error| format!("DistillRefresh has unsupported OPD config: {error:#}"))?;
    if req.name.trim().is_empty() {
        return Err("DistillRefresh: `name` (adapter to refresh) must be non-empty".into());
    }
    if req.behavioural_teacher.trim().is_empty() {
        return Err("DistillRefresh: `behavioural_teacher` alias must be non-empty".into());
    }

    // Resolve the new-knowledge source to an inline list of prompts.
    // Dataset sources go through the shared resolver: `agent_traces:`
    // selectors hit the §10.3 trace index, bare names the uploaded
    // dataset registry.
    let prompts: Vec<kiln_train::opd::OpdPrompt> = match &req.new_data {
        kiln_train::NewKnowledgeSource::Inline { examples } => examples.clone(),
        kiln_train::NewKnowledgeSource::Dataset { dataset } => {
            crate::dataset_resolve::resolve_opd_dataset_selector(
                dataset,
                adapter_dir,
                dataset_registry,
                crate::recent_requests::now_unix_ms() as i64,
            )
            .map_err(|e| format!("DistillRefresh: resolve new_data dataset {dataset:?}: {e}"))?
        }
    };
    if prompts.is_empty() {
        return Err("DistillRefresh: new_data resolved to zero prompts".into());
    }

    let materialized_remote_teacher =
        if matches!(teacher_spec.kind, crate::api::teachers::TeacherKind::Remote) {
            Some(materialize_remote_teacher_for_off_policy(
                "DistillRefresh",
                &prompts,
                &req.config,
                tokenizer,
                prepared_remote_teacher,
            )?)
        } else {
            None
        };

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
        training_profile: kiln_train::SftTrainingProfile::NativeOnlineLoraV1,
        invalid_row_policy: kiln_train::SftInvalidRowPolicy::Fail,
        epochs: 1,
        // Resolved per optimizer at run start (the request's optimizer is
        // forwarded below) — a pinned AdamW-era 1e-4 would train Muon cold.
        learning_rate: None,
        lora_rank: req.config.lora_rank,
        lora_alpha: req.config.lora_alpha,
        // Midtrain is a scaffolding pass — the FINAL adapter's SFT run
        // trains the draft head; aligning it twice wastes a phase.
        train_mtp: Some(false),
        base_adapter: req.config.base_adapter.clone(),
        allow_adapter_shape_conversion: false,
        allow_high_lora_scale: req.config.allow_high_lora_scale,
        output_name: Some(midtrain_name.clone()),
        auto_load: false,
        checkpoint_interval: None,
        resume_checkpoint: None,
        grad_checkpoint_segments: None,
        seed: req.config.seed,
        optimizer: req.config.optimizer,
        adapter_smoke_test: false,
    };
    tracing::info!(job_id = %job_id, adapter = %midtrain_name, "phase 1 — SFT midtrain");
    trainer::sft_train_to(
        &midtrain_examples,
        &midtrain_config,
        model_config,
        weights,
        tokenizer,
        adapter_dir,
        output_adapter_dir,
        &midtrain_name,
        Some(progress_cb),
        None,
        Some(gpu_step_coordination.clone()),
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
    let spec = teacher_spec;
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
        crate::api::teachers::TeacherKind::Local => build_local_teacher_for(
            &spec,
            &prompts,
            tokenizer,
            weights,
            model_config,
            adapter_dir,
            req.config.top_k,
            req.config.training_mode,
            &gpu_step_coordination,
        )
        .map_err(|e| format!("distill_refresh phase 2 local-teacher build: {e}"))?,
        crate::api::teachers::TeacherKind::Remote => materialized_remote_teacher
            .clone()
            .ok_or_else(|| "DistillRefresh remote teacher was not materialized".to_string())?,
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
    let train_result = kiln_train::opd::opd_train_to_with_checkpoint_root(
        &prompts,
        &recover_config,
        model_config,
        weights,
        tokenizer,
        teacher.clone(),
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        None,
        Some(gpu_step_coordination.clone()),
    )
    .map_err(|e| format!("distill_refresh phase 2 (OPD recover) failed: {e:#}"));
    let teacher_release = release_opd_teacher(
        teacher,
        weights,
        &gpu_step_coordination,
        "refresh teacher release",
    );
    let output_dir = combine_operation_and_cleanup(
        train_result,
        teacher_release,
        "distill_refresh teacher release",
    )?;

    // §8.11 receipt — records the two-phase pipeline + behavioural-
    // teacher metadata + the recover config.
    let seed = resolved_opd_seed(&output_dir, adapter_name)?;
    let receipt_hyperparameters = serde_json::to_value(req)
        .map_err(|error| format!("serialize distill_refresh receipt: {error}"))?;
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_refresh", seed)
        .with_teacher(registered_teacher_descriptor(spec))
        .with_hyperparameters(receipt_hyperparameters);
    receipt
        .write_to_adapter_dir(&output_dir)
        .map_err(|error| format!("persist distill_refresh provenance receipt: {error:#}"))?;

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
#[allow(clippy::too_many_arguments)]
fn run_distill_merge(
    req: &DistillMergeRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    job_id: &str,
    gpu_step_coordination: trainer::GpuStepCoordination,
) -> std::result::Result<PathBuf, String> {
    req.config
        .validate_runtime_contract()
        .map_err(|error| format!("distill_merge has unsupported OPD config: {error:#}"))?;
    if req.sources.is_empty() {
        return Err("distill_merge: at least one source required".into());
    }
    if !matches!(
        req.config.training_mode,
        kiln_train::opd::OpdTrainingMode::OffPolicy
    ) {
        return Err(
            "distill_merge: fixed per-source teacher fixtures require config.training_mode=\"off_policy\""
                .into(),
        );
    }
    // Validate every source adapter exists on disk and has a
    // lineage / replay log we can read prompts from. We resolve the
    // training prompts via each source's replay log — the §3.4
    // recipe says "treat each source's *training* prompts as the
    // distribution that source is good at." Missing replay history is
    // fatal because substituting unrelated prompts changes the declared
    // merge dataset.
    let mut per_source: Vec<(
        kiln_train::DistillMergeSource,
        Vec<kiln_train::opd::OpdPrompt>,
    )> = Vec::new();
    for source in &req.sources {
        let src_dir = adapter_dir.join(&source.adapter);
        if !src_dir.exists() {
            return Err(format!(
                "distill_merge: source adapter {:?} not found on disk",
                source.adapter
            ));
        }
        let derived = derive_source_prompts(&src_dir, &source.adapter);
        if derived.is_empty() {
            return Err(format!(
                "distill_merge: source adapter {:?} has no usable replay prompts; refusing to substitute unrelated seed data",
                source.adapter
            ));
        }
        let prompts = derived;
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
    // FixtureLogitSource keyed by the exact student-side sequence.
    //
    // Per-source weighting: each source contributes its share of
    // prompts; the per-prompt logprob lookup keys on exact tokens so
    // the trainer queries the *correct* source's teacher for each
    // prompt, with no per-step LoRA swaps needed.
    //
    // Each source must load and every prompt must yield active targets. The
    // merge aborts if either contract fails, preserving the declared teacher
    // identities and exact prompt dataset.
    let teacher_id = format!("merge-multi:{}", req.name);
    let teacher: std::sync::Arc<dyn kiln_train::LogitSource> =
        std::sync::Arc::new(build_multi_tenant_merge_teacher(
            &teacher_id,
            &per_source,
            adapter_dir,
            tokenizer,
            weights,
            model_config,
            req.config.top_k,
            &gpu_step_coordination,
        )?);

    let mut merge_config = req.config.clone();
    if req.student != "base" {
        merge_config.base_adapter = Some(req.student.clone());
    }
    merge_config.output_name = Some(adapter_name.to_string());
    merge_config.auto_load = false;

    let train_result = kiln_train::opd::opd_train_to_with_checkpoint_root(
        &all_prompts,
        &merge_config,
        model_config,
        weights,
        tokenizer,
        teacher.clone(),
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(gpu_step_coordination.clone()),
    )
    .map_err(|e| format!("distill_merge opd_train failed: {e:#}"));
    let teacher_release = release_opd_teacher(
        teacher,
        weights,
        &gpu_step_coordination,
        "merge fixture release",
    );
    let output_dir = combine_operation_and_cleanup(
        train_result,
        teacher_release,
        "distill_merge teacher release",
    )?;

    let seed = resolved_opd_seed(&output_dir, adapter_name)?;
    let receipt_hyperparameters = serde_json::to_value(req)
        .map_err(|error| format!("serialize distill_merge receipt: {error}"))?;
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_merge", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: teacher_id.clone(),
            model_id: teacher_id,
            model_version_hash: None,
            identity: None,
            snapshot_url: None,
        })
        .with_hyperparameters(receipt_hyperparameters);
    receipt
        .write_to_adapter_dir(&output_dir)
        .map_err(|error| format!("persist distill_merge provenance receipt: {error:#}"))?;
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
                    .filter_map(|message| serde_json::from_value(message.clone()).ok())
                    .collect();
                if !chat.is_empty() {
                    out.push(kiln_train::opd::OpdPrompt {
                        messages: chat,
                        teacher_extra_messages: vec![],
                        trajectory: vec![],
                    });
                }
            }
        }
    }
    out
}

/// `/v1/distill/pump` runtime — §3.5 27B → 4B Knowledge Pump.
/// Three modes (Domain / Wide / Examples); §3.5.4 data-multiplier
/// mode auto-engages internally when |examples| < 200.
#[allow(clippy::too_many_arguments)]
fn run_distill_pump(
    req: &DistillPumpRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    teacher_spec: &crate::api::teachers::TeacherSpec,
    prepared_remote_teacher: Option<std::sync::Arc<dyn kiln_train::LogitSource>>,
    job_id: &str,
    gpu_step_coordination: trainer::GpuStepCoordination,
) -> std::result::Result<PathBuf, String> {
    req.config
        .validate_runtime_contract()
        .map_err(|error| format!("distill_pump has unsupported OPD config: {error:#}"))?;
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
        kiln_train::DistillPumpMode::Domain { domain } => {
            canonical_domain_seed_prompts(domain).map_err(|e| format!("distill_pump: {e}"))?
        }
        kiln_train::DistillPumpMode::Wide { wide: _ } => wide_seed_prompts(),
    };
    if prompts.is_empty() {
        return Err(format!(
            "distill_pump: mode {:?} resolved to zero prompts",
            req.mode
        ));
    }

    let materialized_remote_teacher =
        if matches!(teacher_spec.kind, crate::api::teachers::TeacherKind::Remote) {
            Some(materialize_remote_teacher_for_off_policy(
                "distill_pump",
                &prompts,
                &req.config,
                tokenizer,
                prepared_remote_teacher,
            )?)
        } else {
            None
        };

    tracing::info!(
        job_id = %job_id,
        name = %req.name,
        teacher = %req.teacher,
        rollout_budget = req.rollout_budget,
        num_prompts = prompts.len(),
        "distill_pump started"
    );

    // Resolve teacher alias.
    let spec = teacher_spec;
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
        crate::api::teachers::TeacherKind::Local => build_local_teacher_for(
            &spec,
            &prompts,
            tokenizer,
            weights,
            model_config,
            adapter_dir,
            req.config.top_k,
            req.config.training_mode,
            &gpu_step_coordination,
        )
        .map_err(|e| format!("distill_pump local-teacher build: {e}"))?,
        crate::api::teachers::TeacherKind::Remote => materialized_remote_teacher
            .clone()
            .ok_or_else(|| "distill_pump remote teacher was not materialized".to_string())?,
    };

    let mut pump_config = req.config.clone();
    if let Some(rank) = req.rank {
        pump_config.lora_rank = rank;
    }
    pump_config.output_name = Some(adapter_name.to_string());
    pump_config.auto_load = false;

    let train_result = kiln_train::opd::opd_train_to_with_checkpoint_root(
        &prompts,
        &pump_config,
        model_config,
        weights,
        tokenizer,
        teacher.clone(),
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(gpu_step_coordination.clone()),
    )
    .map_err(|e| format!("distill_pump opd_train failed: {e:#}"));
    let teacher_release = release_opd_teacher(
        teacher,
        weights,
        &gpu_step_coordination,
        "pump teacher release",
    );
    let output_dir = combine_operation_and_cleanup(
        train_result,
        teacher_release,
        "distill_pump teacher release",
    )?;

    // §8.11 receipt.
    let seed = resolved_opd_seed(&output_dir, adapter_name)?;
    let receipt_hyperparameters = serde_json::to_value(req)
        .map_err(|error| format!("serialize distill_pump receipt: {error}"))?;
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_pump", seed)
        .with_teacher(registered_teacher_descriptor(spec))
        .with_hyperparameters(receipt_hyperparameters);
    receipt
        .write_to_adapter_dir(&output_dir)
        .map_err(|error| format!("persist distill_pump provenance receipt: {error:#}"))?;
    Ok(output_dir)
}

/// Canonical pump domains, aligned with the §8.4 compatibility table in
/// `api/pit_of_success.rs` and the shipped recipes. Unknown domains are a
/// hard error — before this, anything (including typos and the agentic
/// `judge_traces` corpus) silently fell through to three generic filler
/// prompts and reported SUCCESS, producing meaningless adapters.
const CANONICAL_PUMP_DOMAINS: &[&str] = &[
    "math",
    "math_reasoning",
    "code",
    "coding",
    "python_codegen",
    "rust_codegen",
    "writing",
    "chinese_writing",
    "scientific_writing",
    "legal_drafting",
    "clinical_notes",
    "instruction",
    "if",
    "instruction_following",
    "tool_calling",
    "long_context_summarization",
];

/// Tiny seed-prompt bank for the §3.5.1 targeted-domain pump. Maps a
/// canonical-domain name to a handful of representative prompts. The
/// full corpus lives on disk and ships in a separate Phase 3 artefact;
/// these seeds let the runtime path exercise end-to-end against any
/// registered teacher without depending on the corpus deliverable.
fn canonical_domain_seed_prompts(
    domain: &str,
) -> std::result::Result<Vec<kiln_train::opd::OpdPrompt>, String> {
    use kiln_train::ChatMessage;
    let prompts: &[&str] = match domain.to_ascii_lowercase().as_str() {
        "math" | "math_reasoning" => &[
            "Solve for x: 2x^2 - 5x + 3 = 0.",
            "What is the derivative of sin(x^2)?",
            "Prove that the sum of the angles in a triangle is 180 degrees.",
            "Compute the integral of 1/(x^2 + 1) from -infinity to infinity.",
        ],
        "code" | "coding" | "python_codegen" => &[
            "Write a Python function that reverses a linked list in place.",
            "Implement quicksort in Rust without using the standard sort.",
            "Explain the difference between a deadlock and a livelock with an example.",
            "Refactor this nested-for loop to a single map+filter call: nums = [1,2,3,4]; out = []; for n in nums: if n%2==0: out.append(n*n)",
        ],
        "rust_codegen" => &[
            "Implement a thread-safe LRU cache in Rust with a fixed capacity.",
            "Write a Rust iterator adapter that yields overlapping windows of size N over a slice.",
            "Explain why this borrow fails and fix it: fn longest<'a>(a: &str, b: &'a str) -> &'a str { if a.len() > b.len() { a } else { b } }",
            "Convert this blocking std::net TCP echo server sketch to tokio.",
        ],
        "writing" => &[
            "Write the opening paragraph of a short story set in a lighthouse.",
            "Compose a polite but firm email declining a vendor's price increase.",
            "Rewrite this sentence in active voice: 'The decision was made by the committee.'",
        ],
        "chinese_writing" => &[
            "用三句话向新同事介绍团队的代码评审流程。",
            "把这句话改写得更正式：\"这个方案不行，得重做。\"",
            "为一款本地运行的笔记应用写一段 50 字以内的产品简介。",
        ],
        "scientific_writing" => &[
            "Rewrite this sentence for a journal abstract: 'We tried a bunch of learning rates and picked the one that worked best.'",
            "Summarize the difference between ablation and sensitivity analysis in two sentences.",
            "Draft a one-sentence limitations statement for a study with n=12 participants.",
        ],
        "legal_drafting" => &[
            "Draft a one-paragraph mutual confidentiality clause for a consulting agreement.",
            "Rewrite this clause in plain English: 'Notwithstanding anything to the contrary herein, Licensor shall not be liable for indirect damages.'",
            "List the essential elements of a valid termination-for-convenience clause.",
        ],
        "clinical_notes" => &[
            "Convert to a SOAP note: 54yo male, 2 days of chest tightness on exertion, resolves with rest, no radiation, vitals stable.",
            "Summarize this discharge plan for the patient in plain language: metoprolol 25mg BID, follow-up echo in 6 weeks, low-sodium diet.",
            "List three red-flag symptoms that should trigger immediate escalation for a post-operative knee replacement patient.",
        ],
        "instruction" | "if" | "instruction_following" => &[
            "List exactly five reasons to ride a bicycle instead of driving, each in one sentence.",
            "Translate 'good morning' to Spanish, French, German, and Japanese in that order.",
            "Summarize the plot of Pride and Prejudice in fewer than 50 words.",
        ],
        "tool_calling" => &[
            "You have tools read_file(path), write_file(path, content), and bash(cmd). Find every TODO comment under src/ and list the files that contain one.",
            "You have tools grep(pattern, glob) and read_file(path). Determine which test file covers the RateLimiter struct.",
            "You have a bash(cmd) tool. Check whether port 8420 is already in use and report the owning process.",
            "You have tools ls(path) and read_file(path). Summarize what this repository does from its top-level files.",
        ],
        "long_context_summarization" => &[
            "Summarize the key decisions from this meeting transcript in five bullets, citing the speaker for each.",
            "Given a long changelog, produce a one-paragraph 'what changed for users' summary, ignoring internal refactors.",
            "Condense this multi-chapter design document into a half-page executive brief preserving every numbered requirement.",
        ],
        _ => {
            return Err(format!(
                "unknown pump domain {domain:?} — valid domains: {}. For agentic corpora \
                 use the /v1/agent endpoints (they build prompts from your indexed pi \
                 sessions instead of a seed bank)",
                CANONICAL_PUMP_DOMAINS.join(", ")
            ));
        }
    };
    Ok(prompts
        .iter()
        .map(|p| kiln_train::opd::OpdPrompt {
            messages: vec![ChatMessage::new("user", *p)],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        })
        .collect())
}

/// Tiny seed-prompt bank for the §3.5.2 wide-coverage pump. Covers
/// every canonical domain in one short batch so the runtime path
/// exercises the broad-pump shape too.
fn wide_seed_prompts() -> Vec<kiln_train::opd::OpdPrompt> {
    let mut all = Vec::new();
    for domain in ["math", "code", "writing", "instruction"] {
        all.extend(
            canonical_domain_seed_prompts(domain)
                .expect("wide pump iterates known-canonical domains"),
        );
    }
    all
}

/// `/v1/distill/self` runtime — §3.12 PI self-distillation.
///
/// The three supported PI modes (`GroundTruthConditioning`, `Conciseness`,
/// and `DocumentAsPi`) shape the teacher context before a local-model forward,
/// then run the normal OPD student loss. `ReverseTeacher` is rejected because
/// moving away from a distribution requires a distinct objective; negating
/// logprobs would violate the teacher-source contract.
#[allow(clippy::too_many_arguments)]
fn run_distill_self(
    req: &DistillSelfRequest,
    model_config: &kiln_core::config::ModelConfig,
    weights: &kiln_model::forward::GpuWeights,
    tokenizer: &kiln_core::tokenizer::KilnTokenizer,
    adapter_dir: &std::path::Path,
    output_adapter_dir: &std::path::Path,
    adapter_name: &str,
    progress_cb: trainer::ProgressCallback,
    job_id: &str,
    gpu_step_coordination: trainer::GpuStepCoordination,
) -> std::result::Result<PathBuf, String> {
    req.config
        .validate_runtime_contract()
        .map_err(|error| format!("distill_self has unsupported OPD config: {error:#}"))?;
    if req.name.trim().is_empty() {
        return Err("distill_self: name must be non-empty".into());
    }
    if matches!(req.mode, kiln_train::SelfDistillMode::ReverseTeacher) {
        return Err(
            "distill_self reverse_teacher is unsupported: it requires a distinct reverse objective and cannot be represented by negated logprobs"
                .into(),
        );
    }
    if !matches!(
        req.config.training_mode,
        kiln_train::opd::OpdTrainingMode::OffPolicy
    ) {
        return Err(
            "distill_self: fixed privileged teacher fixtures require config.training_mode=\"off_policy\""
                .into(),
        );
    }
    let prompts: Vec<kiln_train::opd::OpdPrompt> = req.prompts.clone().ok_or_else(|| {
        "distill_self: explicit prompts with assistant actions are required".to_string()
    })?;
    validate_self_distill_conditioning(
        req.mode,
        prompts.len(),
        req.ground_truth.as_deref(),
        req.documents.as_deref(),
    )?;

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
            &gpu_step_coordination,
        )?);

    let mut self_config = req.config.clone();
    self_config.output_name = Some(adapter_name.to_string());
    self_config.auto_load = false;

    let train_result = kiln_train::opd::opd_train_to_with_checkpoint_root(
        &prompts,
        &self_config,
        model_config,
        weights,
        tokenizer,
        teacher.clone(),
        adapter_dir,
        output_adapter_dir,
        adapter_dir,
        adapter_name,
        Some(progress_cb),
        Some(gpu_step_coordination.clone()),
    )
    .map_err(|e| format!("distill_self opd_train failed: {e:#}"));
    let teacher_release = release_opd_teacher(
        teacher,
        weights,
        &gpu_step_coordination,
        "self-distill fixture release",
    );
    let output_dir = combine_operation_and_cleanup(
        train_result,
        teacher_release,
        "distill_self teacher release",
    )?;

    let seed = resolved_opd_seed(&output_dir, adapter_name)?;
    let receipt_hyperparameters = serde_json::to_value(req)
        .map_err(|error| format!("serialize distill_self receipt: {error}"))?;
    let receipt = kiln_train::AdapterReceipt::new(adapter_name, "distill_self", seed)
        .with_teacher(kiln_train::TeacherDescriptor {
            alias: teacher_id.clone(),
            model_id: teacher_id,
            model_version_hash: None,
            identity: None,
            snapshot_url: None,
        })
        .with_hyperparameters(receipt_hyperparameters);
    receipt
        .write_to_adapter_dir(&output_dir)
        .map_err(|error| format!("persist distill_self provenance receipt: {error:#}"))?;
    Ok(output_dir)
}

struct TrainingMemoryRuntime {
    batching_engine: Option<crate::batching_engine::BatchingEngineHandle>,
    paged_cache: Arc<kiln_model::PagedKvCacheKt>,
    allocator_policy: kiln_model::GpuAllocatorMemoryProbePolicy,
    device: kiln_tensor::Device,
    kv_cache_reclaimable: bool,
    vram_source: VramSource,
}

fn training_memory_runtime(state: &AppState) -> Option<TrainingMemoryRuntime> {
    let ModelBackend::Real {
        runner,
        paged_cache,
        batching_engine,
        ..
    } = state.backend.as_ref()
    else {
        return None;
    };
    let runner = runner.read().ok()?;
    let capabilities = runner.backend_capabilities();
    let device = runner.weights.embed_tokens.device();
    let cache_device_matches_model = paged_cache
        .device()
        .is_some_and(|cache_device| cache_device == device);
    Some(TrainingMemoryRuntime {
        batching_engine: batching_engine.clone(),
        paged_cache: paged_cache.clone(),
        allocator_policy: capabilities.storage.gpu_allocator_memory_probe_policy,
        device,
        kv_cache_reclaimable: capabilities.storage.kv_cache_device_memory_pressure
            && cache_device_matches_model,
        vram_source: kiln_memory::vram::detect_vram().source,
    })
}

fn allocator_can_expand_training_budget(vram_source: VramSource) -> bool {
    !matches!(
        vram_source,
        VramSource::LinuxDrmSysfsUnified | VramSource::AppleSilicon
    )
}

fn current_training_safe_bytes(
    runtime: Option<&TrainingMemoryRuntime>,
    current_reservation_bytes: u64,
) -> u64 {
    let governor = kiln_memory::MemoryGovernor::global();
    let other_reserved = governor
        .soft_reserved_bytes()
        .saturating_sub(current_reservation_bytes);
    let live = governor
        .refresh()
        .free_bytes
        .saturating_sub(governor.config().floor_bytes)
        .saturating_sub(other_reserved);
    let allocator = runtime
        .filter(|runtime| allocator_can_expand_training_budget(runtime.vram_source))
        .and_then(|runtime| {
            crate::device_memory::allocator_safe_available_bytes_with_soft_reserved(
                runtime.allocator_policy,
                &runtime.device,
                governor.config().floor_bytes,
                other_reserved,
            )
        })
        .unwrap_or(0);
    live.max(allocator)
}

fn current_kv_staging_available_bytes(
    runtime: &TrainingMemoryRuntime,
    current_reservation_bytes: u64,
) -> u64 {
    let governor = kiln_memory::MemoryGovernor::global();
    let other_reserved = governor
        .soft_reserved_bytes()
        .saturating_sub(current_reservation_bytes);
    let governor_available = governor
        .refresh()
        .free_bytes
        .saturating_sub(governor.config().floor_bytes)
        .saturating_sub(other_reserved);
    crate::device_memory::allocator_safe_available_bytes_with_soft_reserved(
        runtime.allocator_policy,
        &runtime.device,
        governor.config().floor_bytes,
        other_reserved,
    )
    .unwrap_or(governor_available)
}

fn kv_shrink_target_for_training(
    current_blocks: usize,
    bytes_per_block: u64,
    required_bytes: u64,
    available_bytes: u64,
) -> Option<usize> {
    if current_blocks <= 1 || bytes_per_block == 0 || required_bytes <= available_bytes {
        return None;
    }
    let deficit = required_bytes.saturating_sub(available_bytes);
    let blocks_to_free = deficit
        .div_ceil(bytes_per_block)
        .min(current_blocks as u64 - 1) as usize;
    let target = current_blocks.saturating_sub(blocks_to_free).max(1);
    (target < current_blocks).then_some(target)
}

fn prepare_training_memory_for_job(
    state: &AppState,
    required_bytes: u64,
    current_reservation_bytes: u64,
) -> Result<(), String> {
    state
        .ensure_backend_healthy()
        .map_err(|error| format!("{error:#}"))?;
    if required_bytes == 0 {
        return Ok(());
    }
    let runtime = training_memory_runtime(state);
    let before = current_training_safe_bytes(runtime.as_ref(), current_reservation_bytes);
    if before < required_bytes {
        if let Some(runtime) = runtime.as_ref() {
            let current_blocks = runtime.paged_cache.num_blocks();
            let bytes_per_block = runtime.paged_cache.bytes_per_block() as u64;
            if runtime.kv_cache_reclaimable
                && let Some(requested_target_blocks) = kv_shrink_target_for_training(
                    current_blocks,
                    bytes_per_block,
                    required_bytes,
                    before,
                )
                && let Some(engine) = runtime.batching_engine.as_ref()
            {
                let staging_available =
                    current_kv_staging_available_bytes(runtime, current_reservation_bytes);
                if let Some(plan) = crate::kv_autoscaler::plan_resize_with_staging_headroom(
                    current_blocks,
                    requested_target_blocks,
                    1,
                    staging_available,
                    bytes_per_block,
                ) {
                    state.clear_real_prefix_cache();
                    let _staging_reservation =
                        kiln_memory::MemoryGovernor::global().reserve(plan.replacement_bytes);
                    match engine.resize_kv_blocking(
                        plan.target_blocks,
                        KvResizeReason::TrainingMemoryPreparation,
                    ) {
                        Ok(achieved) => tracing::info!(
                            from_blocks = current_blocks,
                            requested_target_blocks,
                            planned_target_blocks = plan.target_blocks,
                            achieved_blocks = achieved,
                            replacement_mb = plan.replacement_bytes / (1024 * 1024),
                            staging_available_mb = staging_available / (1024 * 1024),
                            required_gb = required_bytes as f64 / 1e9,
                            reserved_gb = current_reservation_bytes as f64 / 1e9,
                            available_before_gb = before as f64 / 1e9,
                            "training worker shrank KV cache before allocation"
                        ),
                        Err(err) => tracing::warn!(
                            error = %format!("{err:#}"),
                            requested_target_blocks,
                            planned_target_blocks = plan.target_blocks,
                            replacement_mb = plan.replacement_bytes / (1024 * 1024),
                            "training worker failed to shrink KV cache before allocation"
                        ),
                    }
                } else {
                    tracing::warn!(
                        from_blocks = current_blocks,
                        requested_target_blocks,
                        staging_available_mb = staging_available / (1024 * 1024),
                        bytes_per_block,
                        "training worker skipped KV shrink: full replacement pool lacks staging headroom"
                    );
                }
            }
        }
        let reclaim_started = std::time::Instant::now();
        let reclaimed = kiln_memory::MemoryGovernor::global().reclaim(u64::MAX);
        tracing::info!(
            event = "gpu_memory_operation",
            operation = "reclaim",
            reason = "training_memory_preparation",
            outcome = if reclaimed > 0 {
                "reclaimed"
            } else {
                "zero_yield"
            },
            target_bytes = u64::MAX,
            actual_bytes = reclaimed,
            wait_ms = 0.0,
            duration_ms = reclaim_started.elapsed().as_secs_f64() * 1000.0,
            reclaimed_mb = reclaimed / (1024 * 1024),
            "training worker reclaimed pooled memory before allocation"
        );
    }

    let after = current_training_safe_bytes(runtime.as_ref(), current_reservation_bytes);
    if after < required_bytes {
        return Err(format!(
            "training memory could not be dynamically reclaimed: estimated step needs {:.2} GB but only {:.2} GB is available after cache/allocator reclamation",
            required_bytes as f64 / 1e9,
            after as f64 / 1e9,
        ));
    }
    state
        .ensure_backend_healthy()
        .map_err(|error| format!("{error:#}"))
}

fn reject_queued_training_job(
    state: &AppState,
    job_id: &str,
    detail: String,
    reason: &'static str,
) {
    let metadata = state
        .training_jobs
        .read()
        .unwrap()
        .get(job_id)
        .map(|job| (job.job_type, job.adapter_name.clone()));
    finalize_job(state, job_id, TrainingState::Failed, Some(detail.clone()));

    if let Some((job_type, adapter_name)) = metadata {
        let metric_type = match job_type {
            TrainingJobType::Sft => TrainingMetricType::Sft,
            TrainingJobType::Grpo => TrainingMetricType::Grpo,
            TrainingJobType::Opd => TrainingMetricType::Opd,
        };
        state
            .metrics
            .inc_training(metric_type, TrainingMetricStatus::Failed);
        if let Some(url) = state.training_webhook_url.as_ref() {
            fire_completion_webhook(
                url.clone(),
                TrainingCompletionEvent {
                    job_id: job_id.to_string(),
                    job_type: TrainingCompletionEvent::job_type_str(job_type),
                    status: "failed",
                    adapter_name,
                    adapter_path: None,
                    error: Some(detail.clone()),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                },
            );
        }
    }
    tracing::error!(job_id, error = %detail, reason, "queued training rejected before execution");
}

fn reject_queued_job_for_backend_quarantine(state: &AppState, job_id: &str, error: anyhow::Error) {
    reject_queued_training_job(
        state,
        job_id,
        format!("training rejected because {error:#}"),
        "backend_quarantine",
    );
}

fn reject_queued_job_for_serving_profile(state: &AppState, job_id: &str, error: anyhow::Error) {
    reject_queued_training_job(
        state,
        job_id,
        format!("training rejected because {error:#}"),
        "serving_profile",
    );
}

struct PreparedTrainingPublication {
    staging_root: tempfile::TempDir,
    final_path: PathBuf,
    expected_revision: crate::adapter_swap::AdapterDiskRevision,
}

impl PreparedTrainingPublication {
    fn output_root(&self) -> &std::path::Path {
        self.staging_root.path()
    }
}

struct PublishedTrainingOutput {
    path: PathBuf,
    reloaded: bool,
}

fn prepare_training_publication(
    state: &AppState,
    adapter_name: &str,
    allow_loaded_reload: bool,
) -> Result<PreparedTrainingPublication, String> {
    let staging_root = tempfile::Builder::new()
        .prefix(".training-tmp-")
        .tempdir_in(&state.adapter_dir)
        .map_err(|error| format!("create training staging root: {error}"))?;
    let final_path = state.adapter_dir.join(adapter_name);
    let serial = crate::adapter_swap::adapter_mutation_guard_blocking(state)?;
    if !allow_loaded_reload && state.loaded_adapter_name().as_deref() == Some(adapter_name) {
        return Err(format!(
            "adapter_revision_conflict: gated training cannot rewrite physically loaded adapter `{adapter_name}` before its post-eval gate runs; unload it or choose a different config.output_name, then resubmit (no weights were changed)"
        ));
    }
    let expected_revision =
        crate::adapter_swap::capture_adapter_disk_revision_locked(&final_path, &serial)?;
    snapshot_starting_adapter_locked(
        &final_path,
        staging_root.path(),
        &expected_revision,
        &serial,
    )?;
    Ok(PreparedTrainingPublication {
        staging_root,
        final_path,
        expected_revision,
    })
}

fn snapshot_starting_adapter_locked(
    source: &std::path::Path,
    staging_root: &std::path::Path,
    expected_revision: &crate::adapter_swap::AdapterDiskRevision,
    _serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) -> Result<(), String> {
    let crate::adapter_swap::AdapterDiskRevision::Content(expected_content) = expected_revision
    else {
        return Ok(());
    };
    let snapshot = staging_root.join(kiln_train::trainer::STARTING_ADAPTER_SNAPSHOT_DIR);
    snapshot_adapter_tree(source, &snapshot)?;
    let actual = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&snapshot)
        .map_err(|error| {
            format!(
                "validate prepared starting-adapter snapshot at {}: {error:#}",
                snapshot.display()
            )
        })?
        .content_revision();
    if &actual != expected_content {
        return Err(format!(
            "starting-adapter snapshot revision mismatch: expected {expected_content}, found {actual}"
        ));
    }
    Ok(())
}

fn snapshot_adapter_tree(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> Result<(), String> {
    std::fs::create_dir(destination).map_err(|error| {
        format!(
            "create starting-adapter snapshot directory {}: {error}",
            destination.display()
        )
    })?;
    let entries = std::fs::read_dir(source)
        .map_err(|error| format!("read adapter snapshot source {}: {error}", source.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "read adapter snapshot entry in {}: {error}",
                source.display()
            )
        })?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type().map_err(|error| {
            format!(
                "inspect adapter snapshot source {}: {error}",
                source_path.display()
            )
        })?;
        if file_type.is_dir() {
            snapshot_adapter_tree(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            if std::fs::hard_link(&source_path, &destination_path).is_err() {
                std::fs::copy(&source_path, &destination_path).map_err(|error| {
                    format!(
                        "copy adapter snapshot file {} to {}: {error}",
                        source_path.display(),
                        destination_path.display()
                    )
                })?;
            }
        } else {
            return Err(format!(
                "adapter snapshot source {} is not a regular file or directory",
                source_path.display()
            ));
        }
    }
    Ok(())
}

fn publish_training_output(
    state: &AppState,
    adapter_name: &str,
    staged_path: PathBuf,
    publication: &PreparedTrainingPublication,
) -> Result<PublishedTrainingOutput, String> {
    let expected_path = publication.output_root().join(adapter_name);
    if staged_path != expected_path {
        return Err(format!(
            "trainer returned unexpected staged path {}; expected {}",
            staged_path.display(),
            expected_path.display()
        ));
    }
    let serial = crate::adapter_swap::adapter_mutation_guard_blocking(state)?;
    let published = crate::adapter_swap::publish_staged_adapter_blocking_locked(
        state,
        adapter_name,
        &staged_path,
        &publication.final_path,
        &publication.output_root().join("previous-adapter"),
        &publication.expected_revision,
        &serial,
    )?;
    publish_training_checkpoints_locked(state, adapter_name, publication.output_root(), &serial);
    tracing::info!(
        adapter = adapter_name,
        content_revision = %published.content_revision,
        reloaded = published.reloaded,
        "published staged training output at adapter revision barrier"
    );
    Ok(PublishedTrainingOutput {
        path: publication.final_path.clone(),
        reloaded: published.reloaded,
    })
}

fn publish_training_checkpoints_locked(
    state: &AppState,
    adapter_name: &str,
    staging_root: &std::path::Path,
    _serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) {
    let prefix = format!("{adapter_name}-checkpoint-");
    let resumable_suffix = kiln_train::checkpoint::TRAINING_CHECKPOINT_DIRECTORY_SUFFIX;
    let staged_legacy_checkpoints: Vec<_> = std::fs::read_dir(staging_root)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with(&prefix) && !name.ends_with(resumable_suffix)
        })
        .collect();
    if staged_legacy_checkpoints.is_empty() {
        return;
    }

    if let Ok(entries) = std::fs::read_dir(&state.adapter_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix)
                && !name.ends_with(resumable_suffix)
                && let Err(error) = std::fs::remove_dir_all(entry.path())
            {
                tracing::warn!(checkpoint = %name, %error, "failed to remove stale training checkpoint");
            }
        }
    }
    for entry in staged_legacy_checkpoints {
        let name = entry.file_name().to_string_lossy().to_string();
        let destination = state.adapter_dir.join(&name);
        if let Err(error) = std::fs::rename(entry.path(), &destination) {
            tracing::warn!(checkpoint = %name, %error, "failed to publish training checkpoint");
        }
    }
}

/// Execute a single training job (runs on a blocking thread).
fn execute_job(state: AppState, mut entry: QueueEntry) {
    let job_id = entry.job_id.clone();

    {
        let jobs = state.training_jobs.read().unwrap();
        match jobs.get(&job_id) {
            Some(job) if job.state == TrainingState::Failed => {
                tracing::info!(job_id = %job_id, "skipping cancelled job");
                return;
            }
            Some(_) => {}
            None => {
                tracing::warn!(job_id = %job_id, "job not found in tracking map, skipping");
                return;
            }
        }
    }

    if let Err(error) = state.ensure_backend_healthy() {
        reject_queued_job_for_backend_quarantine(&state, &job_id, error);
        return;
    }
    if let Err(error) = state.ensure_training_gpu_ownership_allowed() {
        reject_queued_job_for_serving_profile(&state, &job_id, error);
        return;
    }

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

    if let Err(error) = state.ensure_backend_healthy() {
        reject_queued_job_for_backend_quarantine(&state, &job_id, error);
        return;
    }
    if let Err(error) = state.ensure_training_gpu_ownership_allowed() {
        reject_queued_job_for_serving_profile(&state, &job_id, error);
        return;
    }

    // Extract model weights reference
    let (runner_arc, backend_health) = match state.backend.as_ref() {
        ModelBackend::Real {
            runner,
            backend_health,
            ..
        } => (runner.clone(), backend_health.clone()),
        ModelBackend::Mock { .. } => {
            finalize_job(
                &state,
                &job_id,
                TrainingState::Failed,
                Some(
                    "training requires real model weights (not available in mock mode)".to_string(),
                ),
            );
            tracing::error!(job_id = %job_id, "training requires real model weights");
            return;
        }
    };

    // Get auto_load, adapter_name, and job_type from the job info
    let (auto_load, adapter_name, job_type) = {
        let jobs = state.training_jobs.read().unwrap();
        let job = jobs.get(&job_id).unwrap();
        (job.auto_load, job.adapter_name.clone(), job.job_type)
    };
    let resume_admission =
        materialize_queued_job_effective_seed(&mut entry.job, &state.adapter_dir, &adapter_name)
            .map(|_| ());
    if let Err(error) = resume_admission {
        reject_queued_training_job(&state, &job_id, error, "invalid_resume_checkpoint");
        return;
    }
    // Capture the post-eval hook before the job request is consumed by the
    // trainer. A gated same-name rewrite of physically loaded bytes must be
    // rejected before GPU work: reloading would violate the gate, while
    // replacing disk behind the old loaded identity would violate the
    // revision barrier.
    let post_eval: Option<kiln_eval::PostEvalConfig> = match &entry.job {
        QueuedJob::Sft(req) => req.post_eval.clone(),
        QueuedJob::Grpo(req) => req.post_eval.clone(),
        QueuedJob::Opd(req) => req.post_eval.clone(),
        QueuedJob::DistillRefresh(req) => req.post_eval.clone(),
        QueuedJob::DistillMerge(req) => req.post_eval.clone(),
        QueuedJob::DistillPump(req) => req.post_eval.clone(),
        QueuedJob::DistillSelf(req) => req.post_eval.clone(),
    };
    let promotion_gate_pending = post_eval
        .as_ref()
        .is_some_and(|cfg| cfg.min_accuracy.is_some());
    let publication = prepare_training_publication(&state, &adapter_name, !promotion_gate_pending);

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
        let mut control = trainer::TrainControl::Continue;
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
            // The per-step progress call doubles as the cancellation
            // point: DELETE /v1/train/queue/{id} on a RUNNING job sets
            // this flag and the trainer aborts at the next boundary.
            if job
                .cancel_requested
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                control = trainer::TrainControl::Stop;
            }
        }
        control
    });

    // Apply server-level checkpoint_interval default if not set per-job
    let server_checkpoint_interval = state.checkpoint_interval;

    let base_model = trainer::default_base_model(&state.model_config);

    // §8.7 distill_refresh dual eval gate: capture the IF-eval and
    // new-knowledge suite names so we can enqueue dual evals after
    // training completes. The thresholds from the request become
    // min_accuracy on each PostEvalConfig.
    let distill_refresh_dual: Option<(Option<String>, Option<String>, f64, f64)> = match &entry.job
    {
        QueuedJob::DistillRefresh(req) => Some((
            req.if_eval_suite.clone(),
            req.new_knowledge_eval_suite.clone(),
            req.require_if_eval_recovery,
            req.require_internal_qa_gain,
        )),
        _ => None,
    };

    // Resolve only the immutable submit-time binding. Registry deletion or
    // replacement while this job waited in the FIFO is a terminal failure,
    // never permission to silently train against a different teacher.
    let pinned_teacher = resolve_pinned_teacher_for_job(
        &entry.job,
        &entry.teacher_bindings,
        &state.teacher_registry,
    );

    // A registration-time probe is stale by definition once a job has waited
    // in the queue. Revalidate a pinned remote deployment before memory
    // reclamation, GPU coordination, or any cache lookup. The pump's existing
    // cache flag wraps only this freshly verified source.
    let remote_cache_root = matches!(
        &entry.job,
        QueuedJob::DistillPump(request) if request.use_cache
    )
    .then(|| crate::api::cache::cache_root(&state));
    let prepared_remote_teacher: std::result::Result<
        Option<std::sync::Arc<dyn kiln_train::LogitSource>>,
        String,
    > = match pinned_teacher.as_ref() {
        Ok(Some(spec)) if matches!(spec.kind, crate::api::teachers::TeacherKind::Remote) => {
            build_remote_teacher_for(
                spec,
                &state.teacher_credentials,
                remote_cache_root.as_deref(),
            )
            .map(Some)
        }
        Ok(_) => Ok(None),
        Err(error) => Err(error.clone()),
    };

    // #24: hold a governor soft-reservation for this job's estimated working set
    // across its entire execution. This lowers `MemoryGovernor::available_bytes()`
    // so the KV autoscaler proactively shrinks inference KV BEFORE the trainer
    // allocates — the training/inference VRAM arbiter. Capped at total VRAM so a
    // bad over-estimate degrades gracefully (it can never starve inference below
    // the autoscaler's floor). RAII: drops at the end of this function scope —
    // after the match AND finalize — releasing the budget back to inference.
    // Read `reserved_bytes` (Copy) here, before `match entry.job` moves the job.
    let binding_is_valid =
        pinned_teacher.is_ok() && prepared_remote_teacher.is_ok() && publication.is_ok();
    let _mem_reservation = (binding_is_valid && entry.reserved_bytes > 0).then(|| {
        let total = kiln_memory::vram::detect_vram().total_bytes;
        let bytes = if total > 0 {
            entry.reserved_bytes.min(total)
        } else {
            entry.reserved_bytes
        };
        tracing::info!(
            reserved_mb = bytes / (1024 * 1024),
            "training job holding governor memory reservation (inference KV will shrink to fit)"
        );
        kiln_memory::MemoryGovernor::global().reserve(bytes)
    });

    let held_reservation_bytes = _mem_reservation.as_ref().map_or(0, |guard| guard.bytes());
    let memory_ready = if binding_is_valid {
        state
            .ensure_backend_healthy()
            .map_err(|error| format!("{error:#}"))
            .and_then(|()| {
                prepare_training_memory_for_job(
                    &state,
                    entry.reserved_bytes,
                    held_reservation_bytes,
                )
            })
    } else {
        Ok(())
    };
    let staged_result: std::result::Result<PathBuf, String> = if let Err(err) = publication.as_ref()
    {
        Err(err.clone())
    } else if let Err(err) = pinned_teacher.as_ref() {
        Err(err.clone())
    } else if let Err(err) = prepared_remote_teacher.as_ref() {
        Err(err.clone())
    } else if let Err(err) = memory_ready {
        Err(err)
    } else {
        let pinned_teacher = pinned_teacher.expect("pinned teacher checked above");
        let prepared_remote_teacher =
            prepared_remote_teacher.expect("remote teacher handshake checked above");
        let output_adapter_dir = publication
            .as_ref()
            .expect("training publication checked above")
            .output_root();
        match entry.job {
            QueuedJob::Sft(mut req) => {
                if req.config.checkpoint_interval.is_none() {
                    req.config.checkpoint_interval = server_checkpoint_interval;
                }
                let request_body = serde_json::to_value(&req).unwrap_or_else(
                    |_| serde_json::json!({"error": "failed to serialize SftRequest"}),
                );
                let replay_ctx = trainer::ReplayContext {
                    request_id: job_id.clone(),
                    kind: kiln_train::ReplayKind::Sft,
                    request_body,
                    base_model: base_model.clone(),
                };
                // SFT acquires the write lock per optimizer step so healthy
                // inference can run between step boundaries.
                let guard = runner_arc.read().unwrap();
                let training_dispatch = guard.backend_capabilities().training.server_dispatch;
                let native_route_enabled = training_dispatch.native_route_enabled();
                run_sft(
                    native_route_enabled,
                    training_dispatch.native_training_env,
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    replay_ctx,
                    &job_id,
                    Some(trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    )),
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
                // GRPO coordinates setup, each optimizer group, snapshots, and
                // cleanup independently. Dataset reads, tokenization, and disk
                // publication therefore cannot strand healthy inference behind
                // one job-long writer.
                let guard = runner_arc.read().unwrap();
                let training_dispatch = guard.backend_capabilities().training.server_dispatch;
                let native_route_enabled = training_dispatch.native_route_enabled();
                run_grpo(
                    native_route_enabled,
                    training_dispatch.native_training_env,
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    replay_ctx,
                    &job_id,
                    Some(trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    )),
                )
            }
            QueuedJob::Opd(mut req) => {
                if req.config.checkpoint_interval.is_none() {
                    req.config.checkpoint_interval = server_checkpoint_interval;
                }
                let guard = runner_arc.read().unwrap();
                let teacher_spec = pinned_teacher
                    .as_ref()
                    .expect("OPD admission requires a pinned teacher");
                run_opd(
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    teacher_spec,
                    prepared_remote_teacher.clone(),
                    &job_id,
                    trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    ),
                )
            }
            QueuedJob::DistillRefresh(req) => {
                let guard = runner_arc.read().unwrap();
                let teacher_spec = pinned_teacher
                    .as_ref()
                    .expect("DistillRefresh admission requires a pinned teacher");
                run_distill_refresh(
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    teacher_spec,
                    prepared_remote_teacher.clone(),
                    state.dataset_registry.as_deref(),
                    &job_id,
                    trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    ),
                )
            }
            QueuedJob::DistillMerge(req) => {
                let guard = runner_arc.read().unwrap();
                run_distill_merge(
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    &job_id,
                    trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    ),
                )
            }
            QueuedJob::DistillPump(req) => {
                let guard = runner_arc.read().unwrap();
                let teacher_spec = pinned_teacher
                    .as_ref()
                    .expect("DistillPump admission requires a pinned teacher");
                run_distill_pump(
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    teacher_spec,
                    prepared_remote_teacher.clone(),
                    &job_id,
                    trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    ),
                )
            }
            QueuedJob::DistillSelf(req) => {
                let guard = runner_arc.read().unwrap();
                run_distill_self(
                    &req,
                    &state.model_config,
                    &guard.weights,
                    &state.tokenizer,
                    &state.adapter_dir,
                    output_adapter_dir,
                    &adapter_name,
                    progress_cb,
                    &job_id,
                    trainer::GpuStepCoordination::new(
                        state.gpu_lock.clone(),
                        backend_health.clone(),
                    ),
                )
            }
        }
    };

    let result: std::result::Result<PublishedTrainingOutput, String> =
        staged_result.and_then(|staged_path| {
            let publication = publication.as_ref().map_err(|error| error.clone())?;
            publish_training_output(&state, &adapter_name, staged_path, publication)
        });

    match result {
        Ok(published_output) => {
            let adapter_path = published_output.path;
            let reloaded_by_publication = published_output.reloaded;
            let path_str = adapter_path.display().to_string();
            tracing::info!(job_id = %job_id, job_type = ?job_type, adapter = %adapter_name, path = %path_str, "training completed");

            {
                let mut jobs = state.training_jobs.write().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.progress = 1.0;
                    job.adapter_path = Some(path_str.clone());
                }
            }
            finalize_job(&state, &job_id, TrainingState::Completed, None);
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

            // §8.7: when the request carries a promotion gate
            // (post_eval.min_accuracy), auto-load is DEFERRED until the
            // gate passes — the prior adapter stays active while the
            // verdict is pending, so a worse model is never serving while
            // the eval that would catch it is still in the queue. (The old
            // order hot-swapped the fresh adapter BEFORE the eval was even
            // enqueued.)
            let canary_ok = adapter_canary_allows_auto_load(&adapter_path, &adapter_name, &job_id);
            if auto_load && canary_ok && !promotion_gate_pending {
                if let Err(e) = auto_load_adapter(
                    &state,
                    &adapter_path,
                    &adapter_name,
                    !reloaded_by_publication,
                ) {
                    tracing::error!(job_id = %job_id, "auto-load failed: {e}");
                } else {
                    tracing::info!(job_id = %job_id, "auto-loaded trained adapter");
                }
            } else {
                if promotion_gate_pending {
                    tracing::info!(
                        job_id = %job_id,
                        adapter = %adapter_name,
                        "auto-load deferred until the post-eval gate passes (§8.7)"
                    );
                }
                // Publication already purged this name at the revision
                // barrier. A physically loaded target was reloaded there;
                // otherwise it remains an idle on-disk revision until chosen.
            }

            // Post-training auto-eval: enqueue an eval job against the
            // produced adapter so dashboards land directly on the eval
            // result. Failures here are warnings — we still consider the
            // training itself successful.
            if let Some(cfg) = post_eval.as_ref() {
                if let Err(e) = enqueue_post_training_eval(
                    &state,
                    &job_id,
                    &adapter_name,
                    cfg,
                    auto_load && canary_ok,
                ) {
                    tracing::warn!(job_id = %job_id, error = %e, "post-training eval enqueue failed");
                    // The gate could not be installed — an auto_load that
                    // was deferred to it would otherwise be lost silently.
                    if promotion_gate_pending {
                        let mut jobs = state.training_jobs.write().unwrap();
                        if let Some(job) = jobs.get_mut(&job_id) {
                            job.post_eval_verdict = Some(format!(
                                "post-eval gate could not be enqueued ({e}) — adapter `{adapter_name}` left on disk, NOT promoted"
                            ));
                            // Machine-readable twin (see GateOutcome): the
                            // gate never ran, so this is an error, not a
                            // measured pass/fail.
                            job.gate_outcome =
                                Some(crate::state::GateOutcome::Error.as_str().to_string());
                        }
                    }
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
            if let Some((if_suite, qa_suite, frac_recovery, qa_gain)) =
                distill_refresh_dual.as_ref()
            {
                // §6.4 dual gates, ENFORCED (round-5 discovery: these
                // thresholds were validated then silently discarded — the
                // recipe advertised a 0.95/0.05 safety net that gated
                // nothing). Each suite enqueues as a GATED comparison:
                // min_accuracy 0.0 makes the run a Compare job
                // [previous-active/base, refreshed] and the gate's new
                // relative_recovery / absolute_gain thresholds apply
                // against the baseline run in apply_post_eval_gate.
                let mut enqueue_gated = |suite: &String,
                                         relative_recovery: Option<f32>,
                                         absolute_gain: Option<f32>,
                                         label: &str| {
                    let cfg = kiln_eval::PostEvalConfig {
                        suite: suite.clone(),
                        generation: None,
                        min_accuracy: Some(0.0),
                        include_baseline: false,
                    };
                    match enqueue_post_training_eval(&state, &job_id, &adapter_name, &cfg, false) {
                        Err(e) => {
                            tracing::warn!(job_id = %job_id, suite = %suite, error = %e, "distill_refresh {label} enqueue failed")
                        }
                        Ok(()) => {
                            // Stamp the dual thresholds onto the gate the
                            // standard enqueue just installed.
                            let mut jobs = state.eval_jobs.write().unwrap();
                            if let Some((_, job)) = jobs.iter_mut().find(|(_, j)| {
                                j.post_eval_gate
                                    .as_ref()
                                    .is_some_and(|g| g.training_job_id == job_id)
                                    && j.suite_name == *suite
                            }) {
                                if let Some(gate) = job.post_eval_gate.as_mut() {
                                    gate.relative_recovery = relative_recovery;
                                    gate.absolute_gain = absolute_gain;
                                }
                            }
                            tracing::info!(job_id = %job_id, suite = %suite, "distill_refresh {label} queued (gated)");
                        }
                    }
                };
                if let Some(suite) = if_suite {
                    enqueue_gated(suite, Some(*frac_recovery as f32), None, "IF-eval");
                }
                if let Some(suite) = qa_suite {
                    enqueue_gated(suite, None, Some(*qa_gain as f32), "QA-eval");
                }
            }
        }
        Err(e) => {
            // Operator cancellation surfaces as a distinguishable error
            // from the trainer's step-boundary check; record it under the
            // Cancelled metric (not Failed) so dashboards don't count a
            // deliberate stop as a training failure.
            let cancelled = e.contains("cancelled by user");
            if cancelled {
                tracing::info!(job_id = %job_id, job_type = ?job_type, "training cancelled by user");
            } else {
                tracing::error!(job_id = %job_id, job_type = ?job_type, "training failed: {e}");
            }
            let error_msg = e.clone();
            finalize_job(&state, &job_id, TrainingState::Failed, Some(e));
            state.metrics.inc_training(
                metric_type,
                if cancelled {
                    TrainingMetricStatus::Cancelled
                } else {
                    TrainingMetricStatus::Failed
                },
            );

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
    auto_load_on_pass: bool,
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
    let paired_seed = std::cell::Cell::new(None::<u64>);
    let push = |adapter: Option<String>| -> Result<String, String> {
        let job = crate::eval::queue::QueuedEvalJob::Registered {
            suite_name: cfg.suite.clone(),
            adapter: adapter.clone(),
            generation_override: cfg.generation.clone(),
        };
        let admitted = match paired_seed.get() {
            Some(seed) => state.enqueue_eval_with_effective_seed(
                cfg.suite.clone(),
                vec![adapter],
                crate::eval::queue::EvalSubmissionKind::PostTraining,
                Some(training_job_id.to_string()),
                job,
                seed,
            ),
            None => state.enqueue_eval(
                cfg.suite.clone(),
                vec![adapter],
                crate::eval::queue::EvalSubmissionKind::PostTraining,
                Some(training_job_id.to_string()),
                job,
            ),
        }
        .map_err(|error| format!("post-training eval admission failed: {error:#}"))?;
        paired_seed.set(Some(admitted.effective_seed));
        Ok(admitted.job_id)
    };

    let mut linked_ids: Vec<String> = Vec::new();
    if cfg.include_baseline {
        linked_ids.push(push(None)?);
    }
    // Regression detection (round-4 discovery): a gated run compares the
    // new adapter against the CURRENT ACTIVE adapter (the previous
    // generation; base model when none) in ONE Compare job, so the gate
    // can reject a significant regression via the paired sign test even
    // when the static min_accuracy floor passes. Ungated runs keep the
    // single-adapter shape.
    let baseline_for_gate: Option<String> = if cfg.min_accuracy.is_some() {
        state
            .active_adapter_name
            .read()
            .unwrap()
            .clone()
            .filter(|name| name != adapter_name)
    } else {
        None
    };
    let adapter_eval_id = if cfg.min_accuracy.is_some() {
        let baseline_slot = baseline_for_gate.clone().unwrap_or_default();
        let job = crate::eval::queue::QueuedEvalJob::Compare(kiln_eval::EvalCompareSpec {
            suite: cfg.suite.clone(),
            adapters: vec![baseline_slot.clone(), adapter_name.to_string()],
            seed: paired_seed.get(),
            generation: cfg.generation.clone(),
        });
        let admitted = match paired_seed.get() {
            Some(seed) => state.enqueue_eval_with_effective_seed(
                cfg.suite.clone(),
                vec![
                    Some(baseline_slot.clone()).filter(|s| !s.is_empty()),
                    Some(adapter_name.to_string()),
                ],
                crate::eval::queue::EvalSubmissionKind::PostTraining,
                Some(training_job_id.to_string()),
                job,
                seed,
            ),
            None => state.enqueue_eval(
                cfg.suite.clone(),
                vec![
                    Some(baseline_slot).filter(|s| !s.is_empty()),
                    Some(adapter_name.to_string()),
                ],
                crate::eval::queue::EvalSubmissionKind::PostTraining,
                Some(training_job_id.to_string()),
                job,
            ),
        }
        .map_err(|error| format!("post-training eval admission failed: {error:#}"))?;
        paired_seed.set(Some(admitted.effective_seed));
        admitted.job_id
    } else {
        push(Some(adapter_name.to_string()))?
    };
    linked_ids.push(adapter_eval_id.clone());

    // §8.7 promotion gate: when the request set `min_accuracy`, the
    // adapter's run (never the baseline's) carries the gate. The eval
    // worker applies the verdict at terminal time — promote on pass,
    // rename to `<name>.failed` on fail.
    if let Some(min_accuracy) = cfg.min_accuracy {
        let mut jobs = state.eval_jobs.write().unwrap();
        if let Some(job) = jobs.get_mut(&adapter_eval_id) {
            job.post_eval_gate = Some(crate::eval::queue::PostEvalGate {
                min_accuracy,
                relative_recovery: None,
                absolute_gain: None,
                adapter_name: adapter_name.to_string(),
                training_job_id: training_job_id.to_string(),
                auto_load_on_pass,
            });
        }
    }

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

fn adapter_canary_allows_auto_load(
    adapter_path: &std::path::Path,
    adapter_name: &str,
    job_id: &str,
) -> bool {
    match kiln_train::read_adapter_canary_status_from_adapter_dir(adapter_path) {
        Ok(Some(status)) if status.is_quarantined() => {
            tracing::warn!(
                job_id = %job_id,
                adapter = %adapter_name,
                reason = ?status.failure_reason,
                "skipping post-training auto-load because adapter canary status is quarantined"
            );
            false
        }
        Ok(_) => true,
        Err(err) => {
            tracing::warn!(
                job_id = %job_id,
                adapter = %adapter_name,
                error = %err,
                "skipping post-training auto-load because adapter canary status could not be read"
            );
            false
        }
    }
}

/// Load a LoRA adapter using the two-phase RwLock pattern.
fn auto_load_adapter(
    state: &AppState,
    adapter_path: &std::path::Path,
    adapter_name: &str,
    content_changed: bool,
) -> Result<(), String> {
    // Barrier swap (see `adapter_swap`): in-flight requests finish on the
    // weights they started with, THEN the fresh adapter activates and its
    // stale name-keyed cache entries purge — `content_changed` because the
    // directory was just rewritten by training.
    crate::adapter_swap::swap_runtime_adapter_blocking(
        state,
        crate::adapter_swap::SwapRequest {
            target: crate::adapter_swap::SwapTarget::Resolved {
                active_name: adapter_name.to_string(),
                dir: adapter_path.to_path_buf(),
            },
            content_changed,
            default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Replace(Some(
                adapter_name.to_string(),
            )),
            reason: "training_auto_load",
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TrainingJobInfo;
    use kiln_model::{ServerTrainingDispatchPolicy, ServerTrainingNativeRoute};
    use kiln_train::checkpoint::{
        CheckpointArtifact, CheckpointFileRole, TrainingCheckpointData, TrainingCheckpointManifest,
        TrainingCheckpointOptimizer, TrainingCheckpointPrecision, TrainingCheckpointProgress,
        TrainingCheckpointRngState, TrainingCheckpointScheduler, TrainingCheckpointStateFiles,
        TrainingKind, write_training_checkpoint_atomic,
    };
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Mutex;

    use crate::TEST_ENV_LOCK as ENV_LOCK;

    fn write_resume_checkpoint_fixture(
        root: &std::path::Path,
        directory_name: &str,
        manifest_adapter_name: &str,
        training_kind: TrainingKind,
    ) {
        let manifest = TrainingCheckpointManifest::new(
            "step-1",
            training_kind,
            manifest_adapter_name,
            serde_json::json!({"epochs": 2}),
            TrainingCheckpointPrecision {
                parameter_dtype: "f32".into(),
                optimizer_state_dtype: "none".into(),
                activation_dtype: "f32".into(),
                gradient_dtype: "f32".into(),
                stochastic_rounding: serde_json::json!({"mode": "round_to_nearest"}),
            },
            TrainingCheckpointProgress {
                global_step: 1,
                total_steps: 2,
                epoch_index: 1,
                cursor_in_epoch: 0,
                data_order: vec![0],
            },
            TrainingCheckpointData {
                source_kind: "test".into(),
                content_sha256: "22".repeat(32),
                item_count: 1,
            },
            std::collections::BTreeMap::from([(
                "lora-init".to_string(),
                TrainingCheckpointRngState {
                    algorithm: "seeded-initialization".into(),
                    seed: 73,
                    position: 0,
                    state_file: None,
                },
            )]),
            TrainingCheckpointOptimizer {
                kind: "sgd".into(),
                step: 1,
                hyperparameters: serde_json::json!({"learning_rate": 0.1}),
                state_file: None,
            },
            TrainingCheckpointScheduler {
                kind: "constant".into(),
                step: 1,
                state: serde_json::json!({"learning_rate": 0.1}),
            },
            TrainingCheckpointStateFiles {
                adapter_parameters: "adapter.safetensors".into(),
                optimizer_state: None,
                reference_state: None,
                ema_state: None,
                reward_normalization_state: None,
                loss_history: None,
            },
            serde_json::json!({}),
        );
        write_training_checkpoint_atomic(
            &root.join(directory_name),
            manifest,
            &[CheckpointArtifact {
                relative_path: "adapter.safetensors".into(),
                role: CheckpointFileRole::AdapterParameters,
            }],
            |staging| {
                std::fs::write(staging.join("adapter.safetensors"), b"fixture")?;
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn sft_resume_admission_normalizes_a_valid_stable_basename() {
        let temp = tempfile::tempdir().unwrap();
        let name = "target-checkpoint-step-00000001.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), name, "target", TrainingKind::Sft);
        let mut config = kiln_train::SftConfig {
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };

        normalize_sft_resume_checkpoint(&mut config, temp.path(), "target").unwrap();

        assert_eq!(
            config.resume_checkpoint.as_deref(),
            temp.path().join(name).to_str()
        );

        materialize_sft_effective_seed(&mut config, temp.path(), "target").unwrap();
        assert_eq!(config.seed, Some(73));
        assert_eq!(
            config.resume_checkpoint.as_deref(),
            temp.path().join(name).to_str(),
            "normalization must be idempotent when admission and the worker both validate"
        );
    }

    #[test]
    fn training_seed_materialization_preserves_explicit_values_and_resolves_omissions() {
        let temp = tempfile::tempdir().unwrap();

        let mut implicit_sft = kiln_train::SftConfig::default();
        let sft_seed =
            materialize_sft_effective_seed(&mut implicit_sft, temp.path(), "sft").unwrap();
        assert_eq!(implicit_sft.seed, Some(sft_seed));

        let mut explicit_grpo = kiln_train::GrpoConfig {
            seed: Some(u64::MAX),
            ..Default::default()
        };
        assert_eq!(
            materialize_grpo_effective_seed(&mut explicit_grpo, temp.path(), "grpo").unwrap(),
            u64::MAX
        );
        assert_eq!(explicit_grpo.seed, Some(u64::MAX));

        let mut implicit_opd = kiln_train::OpdConfig::default();
        let opd_seed =
            materialize_opd_effective_seed(&mut implicit_opd, temp.path(), "opd").unwrap();
        assert_eq!(implicit_opd.seed, Some(opd_seed));
    }

    #[test]
    fn resume_seed_is_authoritative_and_mismatches_fail_before_publication() {
        let temp = tempfile::tempdir().unwrap();
        let name = "target-checkpoint-step-00000001.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), name, "target", TrainingKind::Grpo);

        let mut inherited = kiln_train::GrpoConfig {
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };
        assert_eq!(
            materialize_grpo_effective_seed(&mut inherited, temp.path(), "target").unwrap(),
            73
        );
        assert_eq!(inherited.seed, Some(73));

        let mut mismatched = kiln_train::GrpoConfig {
            seed: Some(72),
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };
        let error =
            materialize_grpo_effective_seed(&mut mismatched, temp.path(), "target").unwrap_err();
        assert!(
            error.contains("does not match resume checkpoint seed 73"),
            "{error}"
        );
    }

    #[test]
    fn resume_checkpoint_without_authoritative_init_seed_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let name = "target-checkpoint-step-00000001.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), name, "target", TrainingKind::Sft);
        let manifest_path = temp
            .path()
            .join(name)
            .join(kiln_train::checkpoint::TRAINING_CHECKPOINT_MANIFEST_FILENAME);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["rng_states"] = serde_json::json!({});
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let mut config = kiln_train::SftConfig {
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };

        let error = materialize_sft_effective_seed(&mut config, temp.path(), "target").unwrap_err();
        assert!(
            error.contains("missing the authoritative lora-init RNG state"),
            "{error}"
        );
    }

    #[test]
    fn already_normalized_relative_resume_path_remains_valid() {
        let current = std::env::current_dir().unwrap();
        let temp = tempfile::tempdir_in(&current).unwrap();
        let relative_root = temp.path().strip_prefix(&current).unwrap();
        let name = "target-checkpoint-step-00000001.kiln-checkpoint";
        write_resume_checkpoint_fixture(relative_root, name, "target", TrainingKind::Opd);
        let mut config = kiln_train::OpdConfig {
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };

        materialize_opd_effective_seed(&mut config, relative_root, "target").unwrap();
        materialize_opd_effective_seed(&mut config, relative_root, "target").unwrap();
        assert_eq!(config.seed, Some(73));
    }

    #[test]
    fn grpo_resume_admission_normalizes_only_matching_exact_state() {
        let temp = tempfile::tempdir().unwrap();
        let name = "target-checkpoint-step-00000001.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), name, "target", TrainingKind::Grpo);
        let mut config = kiln_train::GrpoConfig {
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };

        normalize_grpo_resume_checkpoint(&mut config, temp.path(), "target").unwrap();
        assert_eq!(
            config.resume_checkpoint.as_deref(),
            temp.path().join(name).to_str()
        );

        let sft_name = "target-checkpoint-sft.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), sft_name, "target", TrainingKind::Sft);
        let mut wrong_kind = kiln_train::GrpoConfig {
            resume_checkpoint: Some(sft_name.into()),
            ..Default::default()
        };
        let error =
            normalize_grpo_resume_checkpoint(&mut wrong_kind, temp.path(), "target").unwrap_err();
        assert!(error.contains("GRPO resume_checkpoint contains Sft state"));
    }

    #[test]
    fn opd_resume_admission_normalizes_only_matching_exact_state() {
        let temp = tempfile::tempdir().unwrap();
        let name = "target-checkpoint-step-00000001.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), name, "target", TrainingKind::Opd);
        let mut config = kiln_train::OpdConfig {
            resume_checkpoint: Some(name.into()),
            ..Default::default()
        };

        normalize_opd_resume_checkpoint(&mut config, temp.path(), "target").unwrap();
        assert_eq!(
            config.resume_checkpoint.as_deref(),
            temp.path().join(name).to_str()
        );

        let sft_name = "target-checkpoint-sft.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), sft_name, "target", TrainingKind::Sft);
        let mut wrong_kind = kiln_train::OpdConfig {
            resume_checkpoint: Some(sft_name.into()),
            ..Default::default()
        };
        let error =
            normalize_opd_resume_checkpoint(&mut wrong_kind, temp.path(), "target").unwrap_err();
        assert!(error.contains("OPD resume_checkpoint contains Sft state"));

        let mut traversal = kiln_train::OpdConfig {
            resume_checkpoint: Some("../escape.kiln-checkpoint".into()),
            ..Default::default()
        };
        assert!(normalize_opd_resume_checkpoint(&mut traversal, temp.path(), "target").is_err());
    }

    fn write_opd_seed_receipt_fixture(
        root: &std::path::Path,
        adapter_name: &str,
        mode: &str,
        seed: Option<u64>,
    ) {
        let receipt = kiln_train::TrainReceipt::new(
            adapter_name,
            "opd-test",
            &kiln_core::config::ModelConfig::qwen3_5_4b(),
            &merge_teacher_test_tokenizer(),
            kiln_train::train_receipt::HyperparameterReceipt {
                mode: mode.to_string(),
                rank: 8,
                alpha: 16.0,
                alpha_over_rank: Some(2.0),
                learning_rate: 1e-4,
                epochs: 1,
                seed,
                shuffle: false,
            },
            serde_json::json!({"seed": seed}),
        );
        std::fs::write(
            root.join(kiln_train::TRAIN_RECEIPT_FILENAME),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn resolved_opd_seed_comes_from_the_success_receipt_and_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        write_opd_seed_receipt_fixture(temp.path(), "target", "opd", Some(73));
        assert_eq!(resolved_opd_seed(temp.path(), "target").unwrap(), 73);

        write_opd_seed_receipt_fixture(temp.path(), "target", "opd", None);
        let missing = resolved_opd_seed(temp.path(), "target").unwrap_err();
        assert!(missing.contains("missing the resolved effective seed"));

        write_opd_seed_receipt_fixture(temp.path(), "other", "opd", Some(73));
        let mismatched = resolved_opd_seed(temp.path(), "target").unwrap_err();
        assert!(mismatched.contains("receipt identity mismatch"));

        write_opd_seed_receipt_fixture(temp.path(), "target", "sft", Some(73));
        let wrong_mode = resolved_opd_seed(temp.path(), "target").unwrap_err();
        assert!(wrong_mode.contains("receipt identity mismatch"));
    }

    #[test]
    fn operation_cleanup_combiner_preserves_both_failures() {
        assert_eq!(
            combine_operation_and_cleanup(Ok(7), Ok(()), "cleanup").unwrap(),
            7
        );
        assert_eq!(
            combine_operation_and_cleanup::<()>(Err("operation".into()), Ok(()), "cleanup")
                .unwrap_err(),
            "operation"
        );
        assert_eq!(
            combine_operation_and_cleanup(Ok(()), Err("release".into()), "cleanup").unwrap_err(),
            "release"
        );
        assert_eq!(
            combine_operation_and_cleanup::<()>(
                Err("operation".into()),
                Err("release".into()),
                "teacher cleanup",
            )
            .unwrap_err(),
            "operation; teacher cleanup also failed: release"
        );
    }

    #[test]
    fn sft_resume_admission_rejects_ambiguous_or_incompatible_inputs() {
        let temp = tempfile::tempdir().unwrap();
        for raw in ["", "../escape.kiln-checkpoint", "nested/x.kiln-checkpoint"] {
            let mut config = kiln_train::SftConfig {
                resume_checkpoint: Some(raw.into()),
                ..Default::default()
            };
            assert!(
                normalize_sft_resume_checkpoint(&mut config, temp.path(), "target").is_err(),
                "resume input {raw:?} must fail closed"
            );
        }

        let peft_name = "target-checkpoint-peft.kiln-checkpoint";
        let peft = temp.path().join(peft_name);
        std::fs::create_dir(&peft).unwrap();
        std::fs::write(peft.join("adapter_config.json"), b"{}").unwrap();
        let mut peft_config = kiln_train::SftConfig {
            resume_checkpoint: Some(peft_name.into()),
            ..Default::default()
        };
        let error =
            normalize_sft_resume_checkpoint(&mut peft_config, temp.path(), "target").unwrap_err();
        assert!(error.contains("not resumable"), "{error}");

        let grpo_name = "target-checkpoint-grpo.kiln-checkpoint";
        write_resume_checkpoint_fixture(temp.path(), grpo_name, "target", TrainingKind::Grpo);
        let mut grpo_config = kiln_train::SftConfig {
            resume_checkpoint: Some(grpo_name.into()),
            ..Default::default()
        };
        let error =
            normalize_sft_resume_checkpoint(&mut grpo_config, temp.path(), "target").unwrap_err();
        assert!(error.contains("Grpo state"), "{error}");

        let wrong_adapter_name = "target-checkpoint-other.kiln-checkpoint";
        write_resume_checkpoint_fixture(
            temp.path(),
            wrong_adapter_name,
            "other",
            TrainingKind::Sft,
        );
        let mut wrong_adapter_config = kiln_train::SftConfig {
            resume_checkpoint: Some(wrong_adapter_name.into()),
            ..Default::default()
        };
        let error =
            normalize_sft_resume_checkpoint(&mut wrong_adapter_config, temp.path(), "target")
                .unwrap_err();
        assert!(error.contains("does not match output adapter"), "{error}");
    }

    struct BlockingVerifiedTeacher {
        identity: kiln_train::TeacherIdentityV1,
        entered: std::sync::mpsc::SyncSender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl std::fmt::Debug for BlockingVerifiedTeacher {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("BlockingVerifiedTeacher")
                .field("identity", &self.identity)
                .finish_non_exhaustive()
        }
    }

    impl kiln_train::LogitSource for BlockingVerifiedTeacher {
        fn capabilities(&self) -> kiln_train::LogitSourceCaps {
            kiln_train::LogitSourceCaps {
                teacher_id: "blocking-verified-remote".into(),
                vocab_size: 32,
                max_top_k: 16,
                supports_full_vocab: false,
                supports_batched: true,
                tokenizer_hash: Some(self.identity.tokenizer_vocab_sha256().to_string()),
            }
        }

        fn authoritative_teacher_identity(&self) -> Option<&kiln_train::TeacherIdentityV1> {
            Some(&self.identity)
        }

        fn fetch_logprobs(
            &self,
            tokens: &[u32],
            positions: &[usize],
            top_k: Option<usize>,
        ) -> Result<kiln_train::LogprobBatch, kiln_train::LogitSourceError> {
            let caps = self.capabilities();
            kiln_train::logit_source::validate_logit_request(&caps, tokens, positions, top_k)?;
            let top_k = top_k.expect("materialization always requests top-K");
            self.entered.send(()).map_err(|error| {
                kiln_train::LogitSourceError::invalid(
                    &caps.teacher_id,
                    format!("signal blocked teacher entry: {error}"),
                )
            })?;
            self.release.lock().unwrap().recv().map_err(|error| {
                kiln_train::LogitSourceError::invalid(
                    &caps.teacher_id,
                    format!("wait for blocked teacher release: {error}"),
                )
            })?;

            let mut indices = Vec::with_capacity(positions.len() * top_k);
            let mut logprobs = Vec::with_capacity(positions.len() * top_k);
            for _ in positions {
                indices.extend(0..top_k as u32);
                logprobs.extend(std::iter::repeat(-(top_k as f32).ln()).take(top_k));
            }
            Ok(kiln_train::LogprobBatch::TopK(kiln_train::TopKLogprobs {
                indices,
                logprobs,
                top_k,
            }))
        }
    }

    fn mock_state_in(dir: &std::path::Path) -> AppState {
        let config = kiln_core::config::ModelConfig::qwen3_5_4b();
        let scheduler = kiln_scheduler::Scheduler::new(
            kiln_scheduler::SchedulerConfig {
                max_batch_tokens: 8192,
                max_batch_size: 64,
                block_size: 16,
                prefix_cache_enabled: false,
                ..Default::default()
            },
            256,
        );
        let engine = kiln_model::engine::MockEngine::new(config.clone());
        let mut vocab = std::collections::HashMap::new();
        for i in 0u32..32 {
            vocab.insert(format!("t{i}"), i);
        }
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": { "type": "BPE", "vocab": vocab, "merges": [] },
            "added_tokens": [{
                "id": 0, "content": "<|endoftext|>",
                "single_word": false, "lstrip": false, "rstrip": false,
                "normalized": false, "special": true,
            }]
        });
        let tokenizer = kiln_core::tokenizer::KilnTokenizer::from_bytes(
            &serde_json::to_vec(&tokenizer_json).unwrap(),
        )
        .unwrap();
        let mut state = AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            300,
            "Qwen3.5-4B".to_string(),
        );
        state.adapter_dir = dir.to_path_buf();
        state
    }

    fn write_revisioned_adapter(root: &std::path::Path, name: &str, value: f32) {
        let path = root.join(name);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("adapter_config.json"),
            br#"{"r":1,"lora_alpha":1,"target_modules":["q_proj"]}"#,
        )
        .unwrap();
        let bytes = value.to_le_bytes();
        let tensor =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &bytes).unwrap();
        let encoded = safetensors::tensor::serialize([("ignored.weight", tensor)], None).unwrap();
        std::fs::write(path.join("adapter_model.safetensors"), encoded).unwrap();
    }

    #[test]
    fn staged_training_publication_rejects_an_intervening_target_revision() {
        let tmp = tempfile::tempdir().unwrap();
        let state = mock_state_in(tmp.path());
        write_revisioned_adapter(tmp.path(), "target", 1.0);
        let starting_revision = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
            &tmp.path().join("target"),
        )
        .unwrap()
        .content_revision();
        let publication = prepare_training_publication(&state, "target", true).unwrap();
        write_revisioned_adapter(publication.output_root(), "target", 2.0);

        // Simulate a delete/upload or another serialized publisher winning
        // while the long GPU job was preparing its output.
        std::fs::remove_dir_all(tmp.path().join("target")).unwrap();
        write_revisioned_adapter(tmp.path(), "target", 3.0);
        let winning_revision = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
            &tmp.path().join("target"),
        )
        .unwrap()
        .content_revision();

        let error = publish_training_output(
            &state,
            "target",
            publication.output_root().join("target"),
            &publication,
        )
        .err()
        .expect("stale training publication must fail");
        assert!(error.contains("changed while training"), "{error}");
        assert_eq!(
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
                &tmp.path().join("target")
            )
            .unwrap()
            .content_revision(),
            winning_revision,
            "stale publisher must not overwrite the intervening winner"
        );
        assert_eq!(
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
                &publication
                    .output_root()
                    .join(kiln_train::trainer::STARTING_ADAPTER_SNAPSHOT_DIR)
            )
            .unwrap()
            .content_revision(),
            starting_revision,
            "the trainer input remains pinned after the durable target changes"
        );
    }

    #[test]
    fn staged_training_publication_replaces_idle_revision_and_checkpoints() {
        let tmp = tempfile::tempdir().unwrap();
        let state = mock_state_in(tmp.path());
        write_revisioned_adapter(tmp.path(), "target", 1.0);
        std::fs::create_dir_all(tmp.path().join("target-checkpoint-1")).unwrap();
        std::fs::write(tmp.path().join("target-checkpoint-1/marker"), b"old").unwrap();
        let resumable = tmp
            .path()
            .join("target-checkpoint-step-00000001.kiln-checkpoint");
        std::fs::create_dir_all(&resumable).unwrap();
        std::fs::write(resumable.join("marker"), b"immutable").unwrap();
        let publication = prepare_training_publication(&state, "target", true).unwrap();
        write_revisioned_adapter(publication.output_root(), "target", 2.0);
        std::fs::create_dir_all(publication.output_root().join("target-checkpoint-2")).unwrap();
        std::fs::write(
            publication.output_root().join("target-checkpoint-2/marker"),
            b"new",
        )
        .unwrap();
        let staged_revision = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
            &publication.output_root().join("target"),
        )
        .unwrap()
        .content_revision();

        let published = publish_training_output(
            &state,
            "target",
            publication.output_root().join("target"),
            &publication,
        )
        .unwrap();
        assert!(!published.reloaded);
        assert_eq!(published.path, tmp.path().join("target"));
        assert_eq!(
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&published.path)
                .unwrap()
                .content_revision(),
            staged_revision
        );
        assert!(!tmp.path().join("target-checkpoint-1").exists());
        assert_eq!(
            std::fs::read(tmp.path().join("target-checkpoint-2/marker")).unwrap(),
            b"new"
        );
        assert_eq!(
            std::fs::read(resumable.join("marker")).unwrap(),
            b"immutable",
            "publishing final adapter weights must preserve immutable resumable checkpoints"
        );
    }

    #[test]
    fn loaded_training_rewrite_fails_closed_without_a_weight_barrier() {
        let tmp = tempfile::tempdir().unwrap();
        let state = mock_state_in(tmp.path());
        write_revisioned_adapter(tmp.path(), "target", 1.0);
        let old_source = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
            &tmp.path().join("target"),
        )
        .unwrap();
        let old_revision = old_source.content_revision();
        *state.loaded_adapter.write().unwrap() = Some(
            crate::state::LoadedAdapterIdentity::from_source("target", &old_source),
        );
        let publication = prepare_training_publication(&state, "target", true).unwrap();
        write_revisioned_adapter(publication.output_root(), "target", 2.0);

        let error = publish_training_output(
            &state,
            "target",
            publication.output_root().join("target"),
            &publication,
        )
        .err()
        .expect("loaded content must never be replaced without a live runner barrier");
        assert!(error.contains("real model backend"), "{error}");
        assert_eq!(
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
                &tmp.path().join("target")
            )
            .unwrap()
            .content_revision(),
            old_revision
        );
        assert!(publication.output_root().join("target").is_dir());
    }

    #[test]
    fn gated_training_rejects_a_loaded_same_name_before_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let state = mock_state_in(tmp.path());
        write_revisioned_adapter(tmp.path(), "target", 1.0);
        let old_source = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
            &tmp.path().join("target"),
        )
        .unwrap();
        let old_revision = old_source.content_revision();
        *state.loaded_adapter.write().unwrap() = Some(
            crate::state::LoadedAdapterIdentity::from_source("target", &old_source),
        );

        let error = prepare_training_publication(&state, "target", false)
            .err()
            .expect("a gated rewrite cannot reload unapproved bytes");
        assert!(error.contains("adapter_revision_conflict"), "{error}");
        assert!(error.contains("different config.output_name"), "{error}");
        assert_eq!(
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(
                &tmp.path().join("target")
            )
            .unwrap()
            .content_revision(),
            old_revision
        );
        assert!(
            std::fs::read_dir(tmp.path())
                .unwrap()
                .flatten()
                .all(|entry| !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".training-tmp-")),
            "the rejected job must clean its empty staging root"
        );
    }

    fn pinned_teacher_spec(alias: &str, model_id: &str) -> crate::api::teachers::TeacherSpec {
        crate::api::teachers::TeacherSpec {
            alias: alias.into(),
            kind: crate::api::teachers::TeacherKind::Fixture,
            provider: None,
            model_id: model_id.into(),
            max_top_k: Some(32),
            vocab_size: Some(1024),
            supports_full_vocab: Some(false),
            tokenizer_hash: None,
            identity: None,
            url: None,
            credential_id: None,
            notes: None,
            adapter: None,
        }
    }

    fn read_teacher_probe(stream: &TcpStream) -> serde_json::Value {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        assert_eq!(request_line.trim_end(), "POST /v1/completions HTTP/1.1");
        let mut content_length = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some(value) = line
                .strip_prefix("content-length:")
                .or_else(|| line.strip_prefix("Content-Length:"))
            {
                content_length = Some(value.trim().parse::<usize>().unwrap());
            }
        }
        let mut body = vec![0; content_length.unwrap()];
        reader.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn spawn_pinned_teacher(
        identity: kiln_train::TeacherIdentityV1,
        request_count: usize,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let fingerprint = identity.fingerprint();
        let handle = std::thread::spawn(move || {
            for _ in 0..request_count {
                let (mut stream, _) = listener.accept().unwrap();
                let request = read_teacher_probe(&stream);
                let top_k = request["prompt_logprobs"].as_u64().unwrap() as usize;
                assert_eq!(request["prompt"], serde_json::json!([0, 0]));
                let mut row = serde_json::Map::new();
                row.insert("0".into(), serde_json::json!({"logprob": -1.0, "rank": 1}));
                if top_k == 2 {
                    row.insert("1".into(), serde_json::json!({"logprob": -2.0, "rank": 2}));
                }
                let response = serde_json::json!({
                    "object": "text_completion",
                    "model": "teacher-model",
                    "system_fingerprint": fingerprint,
                    "choices": [{
                        "index": 0,
                        "prompt_logprobs": [null, serde_json::Value::Object(row)]
                    }],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}
                });
                let body = serde_json::to_vec(&response).unwrap();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        (format!("http://{address}"), handle)
    }

    #[test]
    fn every_cached_remote_job_rehandshakes_before_accepting_a_hit() {
        const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let identity = kiln_train::TeacherIdentityV1::new(
            "teacher-model",
            A,
            B,
            C,
            None,
            3,
            2,
            64,
            3,
            "vllm-test",
            A,
        )
        .unwrap();
        // First job: two identity probes plus one scoring miss. Second job:
        // two identity probes, followed by a cache hit with no sixth request.
        let (url, server) = spawn_pinned_teacher(identity.clone(), 5);
        let spec = crate::api::teachers::TeacherSpec {
            alias: "remote@test".into(),
            kind: crate::api::teachers::TeacherKind::Remote,
            provider: Some(kiln_train::RemoteProvider::Vllm),
            model_id: "teacher-model".into(),
            max_top_k: Some(2),
            vocab_size: Some(3),
            supports_full_vocab: Some(false),
            tokenizer_hash: Some(B.into()),
            identity: Some(identity),
            url: Some(url),
            credential_id: None,
            notes: None,
            adapter: None,
        };
        let cache_dir = tempfile::tempdir().unwrap();
        let credentials = crate::config::TeachersConfig::default();

        let first = build_remote_teacher_for(&spec, &credentials, Some(cache_dir.path())).unwrap();
        first.fetch_logprobs(&[0, 0], &[0], Some(2)).unwrap();
        let second = build_remote_teacher_for(&spec, &credentials, Some(cache_dir.path())).unwrap();
        let hit = second.fetch_logprobs(&[0, 0], &[0], Some(2)).unwrap();
        assert_eq!(hit.flat_len(), 2);
        server.join().unwrap();

        let stats = kiln_train::LogitCache::new(cache_dir.path())
            .stats()
            .unwrap();
        assert_eq!(stats.total_entries, 1);
    }

    #[test]
    fn slow_remote_materialization_does_not_block_inference_gpu_ownership() {
        const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let identity = kiln_train::TeacherIdentityV1::new(
            "blocking-verified-remote",
            A,
            B,
            C,
            None,
            32,
            16,
            4096,
            65_536,
            "test-runtime",
            A,
        )
        .unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let source: Arc<dyn kiln_train::LogitSource> = Arc::new(BlockingVerifiedTeacher {
            identity,
            entered: entered_tx,
            release: Mutex::new(release_rx),
        });
        let gpu_lock: crate::state::GpuCoordinationLock = Arc::new(tokio::sync::RwLock::new(()));
        let worker_gpu_lock = gpu_lock.clone();
        let (write_acquired_tx, write_acquired_rx) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            let config = kiln_train::OpdConfig {
                training_mode: kiln_train::opd::OpdTrainingMode::OffPolicy,
                top_k: 16,
                ..kiln_train::OpdConfig::default()
            };
            let fixture = materialize_remote_teacher_for_off_policy(
                "scheduling-contract-test",
                &[self_distill_test_prompt(true)],
                &config,
                &merge_teacher_test_tokenizer(),
                Some(source),
            )
            .unwrap();
            let _gpu_guard = crate::state::gpu_coordination_write_guard(&worker_gpu_lock);
            write_acquired_tx.send(()).unwrap();
            fixture
        });

        entered_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("slow teacher reached its blocking fetch");
        let inference_guard = crate::state::gpu_coordination_read_guard(&gpu_lock);
        release_tx.send(()).unwrap();
        assert!(
            write_acquired_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "training must acquire GPU ownership only after remote materialization and current inference"
        );
        drop(inference_guard);
        write_acquired_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("training acquired GPU ownership after materialization completed");
        let fixture = worker.join().unwrap();
        assert!(fixture.authoritative_teacher_identity().is_some());
    }

    fn pinned_opd_job(alias: &str) -> QueuedJob {
        QueuedJob::Opd(OpdRequest {
            prompts: Vec::new(),
            dataset_path: None,
            teacher: alias.into(),
            config: Default::default(),
            post_eval: None,
        })
    }

    fn unrelated_sft_job() -> QueuedJob {
        QueuedJob::Sft(SftRequest {
            examples: Vec::new(),
            dataset_path: None,
            dataset: None,
            config: Default::default(),
            ingestion: None,
            post_eval: None,
        })
    }

    #[test]
    fn queued_teacher_binding_rejects_alias_replacement_and_deletion() {
        let registry = crate::api::teachers::TeacherRegistry::new();
        let pinned = pinned_teacher_spec("teacher", "model-v1");
        registry.insert(pinned.clone());
        let job = pinned_opd_job("teacher");

        assert_eq!(
            resolve_pinned_teacher_for_job(&job, std::slice::from_ref(&pinned), &registry).unwrap(),
            Some(pinned.clone())
        );

        registry.insert(pinned_teacher_spec("teacher", "model-v2"));
        let replaced =
            resolve_pinned_teacher_for_job(&job, std::slice::from_ref(&pinned), &registry)
                .unwrap_err();
        assert!(replaced.contains("replaced after submission"), "{replaced}");

        registry.remove("teacher");
        let deleted =
            resolve_pinned_teacher_for_job(&job, std::slice::from_ref(&pinned), &registry)
                .unwrap_err();
        assert!(deleted.contains("deleted after submission"), "{deleted}");
    }

    #[test]
    fn queued_teacher_binding_requires_one_exact_binding_and_no_extras() {
        let registry = crate::api::teachers::TeacherRegistry::new();
        let pinned = pinned_teacher_spec("teacher", "model-v1");
        registry.insert(pinned.clone());
        let job = pinned_opd_job("teacher");

        let missing = resolve_pinned_teacher_for_job(&job, &[], &registry).unwrap_err();
        assert!(
            missing.contains("no submit-time pinned binding"),
            "{missing}"
        );

        let duplicate =
            resolve_pinned_teacher_for_job(&job, &[pinned.clone(), pinned.clone()], &registry)
                .unwrap_err();
        assert!(
            duplicate.contains("duplicate submit-time bindings"),
            "{duplicate}"
        );

        let extra = resolve_pinned_teacher_for_job(
            &job,
            &[pinned.clone(), pinned_teacher_spec("other", "other-model")],
            &registry,
        )
        .unwrap_err();
        assert!(extra.contains("no extras"), "{extra}");

        assert_eq!(
            resolve_pinned_teacher_for_job(&unrelated_sft_job(), &[], &registry).unwrap(),
            None
        );
        let unrelated = resolve_pinned_teacher_for_job(
            &unrelated_sft_job(),
            std::slice::from_ref(&pinned),
            &registry,
        )
        .unwrap_err();
        assert!(
            unrelated.contains("does not use a registered teacher"),
            "{unrelated}"
        );
    }

    fn merge_teacher_test_tokenizer() -> kiln_core::tokenizer::KilnTokenizer {
        let json = br#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
            }
        }"#;
        kiln_core::tokenizer::KilnTokenizer::from_bytes(json)
            .unwrap()
            .with_chat_template(
                "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
            )
    }

    fn self_distill_test_prompt(with_assistant: bool) -> kiln_train::opd::OpdPrompt {
        let mut messages = vec![kiln_train::ChatMessage::new(
            "user",
            if with_assistant { "a" } else { "aa" },
        )];
        if with_assistant {
            messages.push(kiln_train::ChatMessage::new("assistant", "bb"));
        }
        kiln_train::opd::OpdPrompt {
            messages,
            teacher_extra_messages: vec![],
            trajectory: vec![],
        }
    }

    #[test]
    fn declared_merge_source_lora_load_failure_is_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let err = match load_declared_merge_source_lora(
            "missing-source",
            dir.path(),
            1,
            kiln_tensor::Device::Cpu,
        ) {
            Ok(_) => panic!("an absent declared LoRA must not become a base-model teacher"),
            Err(err) => err,
        };

        assert!(
            err.contains("declared source adapter 'missing-source'"),
            "{err}"
        );
        assert!(err.contains(&dir.path().display().to_string()), "{err}");
        assert!(err.contains("adapter_config.json"), "{err}");
    }

    #[test]
    fn teacher_prompt_tokenization_rejects_later_prompt_without_active_tokens() {
        let prompts = vec![
            kiln_train::opd::OpdPrompt {
                messages: vec![
                    kiln_train::ChatMessage::new("user", "a"),
                    kiln_train::ChatMessage::new("assistant", "bb"),
                ],
                teacher_extra_messages: vec![],
                trajectory: vec![],
            },
            kiln_train::opd::OpdPrompt {
                messages: vec![kiln_train::ChatMessage::new("user", "aa")],
                teacher_extra_messages: vec![],
                trajectory: vec![],
            },
        ];

        let err = tokenize_teacher_prompts(
            "local-teacher",
            "registered-teacher",
            &prompts,
            &merge_teacher_test_tokenizer(),
        )
        .unwrap_err();

        assert!(
            err.contains("local-teacher: source 'registered-teacher' prompt 1 failed to tokenize"),
            "{err}"
        );
        assert!(err.contains("no supervised assistant tokens"), "{err}");
    }

    #[test]
    fn teacher_prompt_tokenization_propagates_chat_template_failure() {
        let tokenizer = merge_teacher_test_tokenizer().with_chat_template("{% if".to_string());
        let prompts = vec![kiln_train::opd::OpdPrompt {
            messages: vec![kiln_train::ChatMessage::new("assistant", "bb")],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        }];

        let err = tokenize_teacher_prompts("distill_merge", "source-a", &prompts, &tokenizer)
            .unwrap_err();

        assert!(
            err.contains("distill_merge: source 'source-a' prompt 0 failed to tokenize"),
            "{err}"
        );
    }

    #[test]
    fn self_distill_alignment_uses_masked_targets_not_length_offsets() {
        validate_self_distill_target_alignment(
            7,
            &[9, 3, 4, 5],
            &[1, 3],
            &[8, 8, 3, 4, 6, 5, 7],
            &[2, 5],
        )
        .expect("pairwise target IDs align despite non-constant position offsets");
    }

    #[test]
    fn self_distill_alignment_rejects_target_token_mismatch() {
        let err = validate_self_distill_target_alignment(
            2,
            &[9, 3, 4, 5],
            &[1, 3],
            &[8, 3, 4, 6],
            &[1, 3],
        )
        .unwrap_err();
        assert!(err.contains("action pair 1 differs"), "{err}");
    }

    #[test]
    fn self_distill_ground_truth_requires_one_nonempty_answer_per_prompt() {
        use kiln_train::SelfDistillMode;

        let missing = validate_self_distill_conditioning(
            SelfDistillMode::GroundTruthConditioning,
            2,
            None,
            None,
        )
        .unwrap_err();
        assert!(missing.contains("ground_truth is required"), "{missing}");

        let too_short = vec!["answer".to_string()];
        let length = validate_self_distill_conditioning(
            SelfDistillMode::GroundTruthConditioning,
            2,
            Some(&too_short),
            None,
        )
        .unwrap_err();
        assert!(length.contains("ground_truth.len() (1)"), "{length}");

        let blank = vec!["answer".to_string(), "  ".to_string()];
        let empty = validate_self_distill_conditioning(
            SelfDistillMode::GroundTruthConditioning,
            2,
            Some(&blank),
            None,
        )
        .unwrap_err();
        assert!(
            empty.contains("ground_truth[1] must be non-empty"),
            "{empty}"
        );
    }

    #[test]
    fn self_distill_documents_require_nonempty_context_per_prompt() {
        use kiln_train::SelfDistillMode;

        let missing =
            validate_self_distill_conditioning(SelfDistillMode::DocumentAsPi, 2, None, None)
                .unwrap_err();
        assert!(missing.contains("documents is required"), "{missing}");

        let too_short = vec!["context".to_string()];
        let length = validate_self_distill_conditioning(
            SelfDistillMode::DocumentAsPi,
            2,
            None,
            Some(&too_short),
        )
        .unwrap_err();
        assert!(length.contains("documents.len() (1)"), "{length}");

        let blank = vec!["context".to_string(), "\n".to_string()];
        let empty = validate_self_distill_conditioning(
            SelfDistillMode::DocumentAsPi,
            2,
            None,
            Some(&blank),
        )
        .unwrap_err();
        assert!(empty.contains("documents[1] must be non-empty"), "{empty}");
    }

    #[test]
    fn self_distill_prompt_preparation_rejects_later_student_failure() {
        let prompts = vec![
            self_distill_test_prompt(true),
            self_distill_test_prompt(false),
        ];
        let tokenizer = merge_teacher_test_tokenizer().with_chat_template(
            concat!(
                "{% for message in messages %}",
                "{% if message.role != 'system' %}{{ message.content }}{% endif %}",
                "{% endfor %}"
            )
            .to_string(),
        );
        let err = prepare_self_distill_prompts(
            &prompts,
            kiln_train::SelfDistillMode::Conciseness,
            None,
            None,
            &tokenizer,
        )
        .unwrap_err();

        assert!(
            err.contains("self-distill prompt 1 student tokenization failed"),
            "{err}"
        );
        assert!(err.contains("no supervised assistant tokens"), "{err}");
    }

    #[test]
    fn self_distill_prompt_preparation_propagates_teacher_failure() {
        let tokenizer = merge_teacher_test_tokenizer().with_chat_template(
            concat!(
                "{% if messages[0].role == 'system' %}",
                "{{ raise_exception('teacher-only template failure') }}",
                "{% endif %}",
                "{% for message in messages %}{{ message.content }}{% endfor %}"
            )
            .to_string(),
        );
        let err = prepare_self_distill_prompts(
            &[self_distill_test_prompt(true)],
            kiln_train::SelfDistillMode::Conciseness,
            None,
            None,
            &tokenizer,
        )
        .unwrap_err();

        assert!(
            err.contains("self-distill prompt 0 teacher tokenization failed"),
            "{err}"
        );
        assert!(err.contains("teacher-only template failure"), "{err}");
    }

    #[test]
    fn kv_shrink_target_frees_only_needed_blocks() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            kv_shrink_target_for_training(16, gb, 12 * gb, 8 * gb),
            Some(12)
        );
    }

    #[test]
    fn training_kv_shrink_deepens_only_as_needed_to_stage_atomic_replacement() {
        let gb = 1024 * 1024 * 1024;
        let requested = kv_shrink_target_for_training(16, gb, 12 * gb, 8 * gb)
            .expect("training needs KV reclamation");
        assert_eq!(requested, 12);
        assert_eq!(
            crate::kv_autoscaler::plan_resize_with_staging_headroom(16, requested, 1, 8 * gb, gb,),
            Some(crate::kv_autoscaler::KvResizeStagingPlan {
                target_blocks: 8,
                replacement_bytes: 8 * gb,
            })
        );
    }

    #[test]
    fn kv_shrink_target_preserves_one_block_floor() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(
            kv_shrink_target_for_training(4, gb, 20 * gb, 1 * gb),
            Some(1)
        );
    }

    #[test]
    fn unified_memory_does_not_let_allocator_expand_training_budget() {
        assert!(!allocator_can_expand_training_budget(
            kiln_memory::vram::VramSource::LinuxDrmSysfsUnified
        ));
        assert!(!allocator_can_expand_training_budget(
            kiln_memory::vram::VramSource::AppleSilicon
        ));
        assert!(allocator_can_expand_training_budget(
            kiln_memory::vram::VramSource::NvidiaSmi
        ));
    }

    fn tracked_job(job_id: &str, adapter: &str, correction_ids: Vec<String>) -> TrainingJobInfo {
        TrainingJobInfo {
            job_id: job_id.to_string(),
            adapter_name: adapter.to_string(),
            job_type: TrainingJobType::Sft,
            effective_seed: Some(17),
            state: TrainingState::Running,
            progress: 0.5,
            loss: None,
            epoch: None,
            adapter_path: None,
            submitted_at: std::time::Instant::now(),
            submitted_unix_ms: 1,
            auto_load: false,
            consumed_correction_ids: correction_ids,
            finished_at: None,
            finished_unix_ms: None,
            error: None,
            linked_eval_job_ids: Vec::new(),
            post_eval_verdict: None,
            gate_outcome: None,
            loss_history: Vec::new(),
            cancel_requested: Default::default(),
        }
    }

    /// The completion-time corrections contract the dashboard now depends
    /// on (it submits dataset "corrections:active" and never marks rows
    /// itself): finalize_job flips consumed rows to trained_into ONLY on
    /// Completed. A Failed job — the rank-8/alpha-32 corrections train
    /// that shipped in 0.4.1 was one — must leave every row active and
    /// re-trainable.
    #[test]
    fn finalize_job_marks_corrections_on_completion_not_failure() {
        let dir = tempfile::tempdir().unwrap();
        let state = mock_state_in(dir.path());

        let store = crate::api::corrections::CorrectionsStore::for_state(&state);
        store
            .upsert(crate::api::corrections::CorrectionRow {
                request_id: "r1".to_string(),
                agent: "pi".to_string(),
                adapter: None,
                user: "what is 2+2".to_string(),
                original: "5".to_string(),
                ideal: "4".to_string(),
                truncated: false,
                created_at: String::new(),
                trained_into: None,
                trained_at: None,
            })
            .unwrap();

        let ids = vec!["r1".to_string()];
        {
            let mut jobs = state.training_jobs.write().unwrap();
            jobs.insert(
                "job-fail".into(),
                tracked_job("job-fail", "fixes-v1", ids.clone()),
            );
            jobs.insert(
                "job-done".into(),
                tracked_job("job-done", "fixes-v1", ids.clone()),
            );
        }

        finalize_job(
            &state,
            "job-fail",
            TrainingState::Failed,
            Some("unsafe LoRA scaling".to_string()),
        );
        let rows = store.list();
        assert_eq!(rows.len(), 1);
        assert!(
            rows[0].trained_into.is_none(),
            "a FAILED job must leave the basket intact and re-trainable"
        );

        finalize_job(&state, "job-done", TrainingState::Completed, None);
        let rows = store.list();
        assert_eq!(
            rows[0].trained_into.as_deref(),
            Some("fixes-v1"),
            "a COMPLETED job marks exactly the consumed rows"
        );
        assert!(rows[0].trained_at.is_some());
    }

    #[test]
    fn server_training_dispatch_policy_treats_empty_and_zero_as_disabled() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const VAR: &str = "KILN_TEST_NATIVE_TRAINING_FLAG";
        let policy = ServerTrainingDispatchPolicy {
            native_route: ServerTrainingNativeRoute::LegacyCudaNative,
            native_training_env: Some(VAR),
            native_training_default_enabled: false,
        };

        unsafe {
            std::env::remove_var(VAR);
        }
        assert!(!policy.native_route_enabled());

        unsafe {
            std::env::set_var(VAR, "");
        }
        assert!(!policy.native_route_enabled());

        unsafe {
            std::env::set_var(VAR, "0");
        }
        assert!(!policy.native_route_enabled());

        unsafe {
            std::env::set_var(VAR, "1");
        }
        assert!(policy.native_route_enabled());

        let default_enabled_policy = ServerTrainingDispatchPolicy {
            native_route: ServerTrainingNativeRoute::LegacyCudaNative,
            native_training_env: Some(VAR),
            native_training_default_enabled: true,
        };
        unsafe {
            std::env::remove_var(VAR);
        }
        assert!(default_enabled_policy.native_route_enabled());
    }

    #[test]
    fn server_training_dispatch_policy_follows_backend_policy() {
        let _env_guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const VAR: &str = "KILN_TEST_SERVER_NATIVE_TRAINING_POLICY";
        let legacy_cuda_policy = ServerTrainingDispatchPolicy {
            native_route: ServerTrainingNativeRoute::LegacyCudaNative,
            native_training_env: Some(VAR),
            native_training_default_enabled: false,
        };
        let shared_policy = ServerTrainingDispatchPolicy {
            native_route: ServerTrainingNativeRoute::SharedKtTape,
            native_training_env: Some(VAR),
            native_training_default_enabled: true,
        };

        unsafe {
            std::env::remove_var(VAR);
        }
        assert!(!legacy_cuda_policy.native_route_enabled());

        unsafe {
            std::env::set_var(VAR, "1");
        }
        assert!(legacy_cuda_policy.native_route_enabled());
        assert!(!shared_policy.native_route_enabled());

        unsafe {
            std::env::set_var(VAR, "0");
        }
        assert!(!legacy_cuda_policy.native_route_enabled());

        unsafe {
            std::env::remove_var(VAR);
        }
    }

    #[test]
    fn test_queue_fifo_order() {
        let mut q = TrainingQueue::new();
        q.push(QueueEntry {
            job_id: "job-1".into(),
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            job: QueuedJob::Sft(SftRequest {
                dataset_path: None,
                dataset: None,
                examples: vec![],
                config: Default::default(),
                ingestion: None,
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            job: QueuedJob::Sft(SftRequest {
                dataset_path: None,
                dataset: None,
                examples: vec![],
                config: Default::default(),
                ingestion: None,
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            job: QueuedJob::Sft(SftRequest {
                dataset_path: None,
                dataset: None,
                examples: vec![],
                config: Default::default(),
                ingestion: None,
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
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            job: QueuedJob::Sft(SftRequest {
                dataset_path: None,
                dataset: None,
                examples: vec![],
                config: Default::default(),
                ingestion: None,
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-2".into(),
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            job: QueuedJob::Sft(SftRequest {
                dataset_path: None,
                dataset: None,
                examples: vec![],
                config: Default::default(),
                ingestion: None,
                post_eval: None,
            }),
        });
        q.push(QueueEntry {
            job_id: "job-3".into(),
            reserved_bytes: 0,
            teacher_bindings: Vec::new(),
            job: QueuedJob::Sft(SftRequest {
                dataset_path: None,
                dataset: None,
                examples: vec![],
                config: Default::default(),
                ingestion: None,
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
        enqueue_post_training_eval(&state, "train-job-1", "trained-adapter", &cfg, false).unwrap();
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
        enqueue_post_training_eval(&state, "train-job-2", "trained-adapter", &cfg, false).unwrap();
        assert_eq!(state.eval_queue.lock().unwrap().len(), 2);
        let jobs = state.eval_jobs.read().unwrap();
        assert_eq!(jobs.len(), 2);
        let seeds = jobs
            .values()
            .map(|job| job.effective_seed.expect("new eval job must have a seed"))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            seeds.len(),
            1,
            "post-training baseline and candidate jobs must share one paired seed"
        );
    }

    #[test]
    fn enqueue_post_training_eval_installs_gate_on_adapter_job_only() {
        let state = mk_post_eval_state();
        let cfg = kiln_eval::PostEvalConfig {
            suite: "smoke".into(),
            generation: None,
            min_accuracy: Some(0.8),
            include_baseline: true,
        };
        enqueue_post_training_eval(&state, "train-gated", "trained-adapter", &cfg, true).unwrap();

        let jobs = state.eval_jobs.read().unwrap();
        assert_eq!(jobs.len(), 2);
        let mut gated = 0;
        for job in jobs.values() {
            if job
                .adapters
                .iter()
                .any(|a| a.as_deref() == Some("trained-adapter"))
            {
                // The gated job is a COMPARE over [previous-active (base
                // here — no active adapter in this fixture), new adapter]
                // so the verdict can run the paired sign test for
                // regression detection.
                let gate = job
                    .post_eval_gate
                    .as_ref()
                    .expect("adapter compare job carries the §8.7 gate");
                assert_eq!(gate.min_accuracy, 0.8);
                assert_eq!(gate.training_job_id, "train-gated");
                assert!(gate.auto_load_on_pass);
                assert_eq!(
                    job.adapters.len(),
                    2,
                    "gated runs compare [baseline, new] for the sign test"
                );
                gated += 1;
            } else {
                assert!(
                    job.post_eval_gate.is_none(),
                    "the include_baseline job never carries the gate"
                );
            }
        }
        assert_eq!(gated, 1);
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
        let err = enqueue_post_training_eval(&state, "j", "a", &cfg, false).unwrap_err();
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
    /// The eval-signals webhook helper posts arbitrary JSON with the
    /// same fire-and-forget contract as the typed training event.
    #[tokio::test]
    async fn test_fire_webhook_json_posts_payload() {
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

        fire_webhook_json(
            format!("http://{addr}/hook"),
            serde_json::json!({
                "event": "eval_completed",
                "job_id": "eval-1",
                "suite": "math-sentinel",
                "status": "completed",
                "headline_accuracy": 0.85,
                "gate_verdict": "promoted (accuracy 0.85 >= 0.8)",
            }),
        );

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !captured.lock().unwrap().is_empty() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "webhook never arrived"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let got = captured.lock().unwrap();
        assert_eq!(got[0]["event"], "eval_completed");
        assert_eq!(got[0]["suite"], "math-sentinel");
        assert_eq!(got[0]["headline_accuracy"], 0.85);
        server.abort();
    }

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
