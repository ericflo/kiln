use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::{Mutex, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, watch};

use kiln_core::block::BlockManager;
use kiln_core::config::ModelConfig;
use kiln_core::config_hashes::ConfigHashes;
use kiln_core::model_provenance::BaseWeightShardManifest;
use kiln_core::prefix_cache::default_prefix_cache_max_blocks;
use kiln_core::sampling::{SamplingParams, ThinkingBudgetStatus};
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::engine::Engine;
use kiln_model::lora_loader::LoraSourceIdentity;
use kiln_model::{
    BackendHealthHandle, DecodeBatcher, DecodeBatcherConfig, GpuAllocatorMemoryProbePolicy,
    GpuMemoryBudgetPolicy, GpuMemoryReclaimPolicy, GpuMemoryReclaimer,
    InferenceRecurrentStatePolicy, KvCacheAutoBlockPolicy, LinearAttentionState, ModelRunner,
    PagedKvCacheKt, PagedPrefixNextToken, PagedPrefixRegistration, Support,
    TrainingAccelerationProfileLogMessage, TrainingAccelerationProfilePolicy,
};
use kiln_scheduler::{PrefixCacheStats, Scheduler};
use kiln_tensor::DType;
use kiln_train::TrainingState;
use serde::Serialize;

use crate::decode_stats::DecodeStatsRing;
use crate::metrics::Metrics;
use crate::recent_requests::{DEFAULT_CAPACITY as RECENT_REQUESTS_CAPACITY, RecentRequestsRing};
use crate::training_queue::{SharedTrainingQueue, ShutdownFlag};

// #1082: 64 (was 16) so each FA2 split-KV decode tile (kBlockN=64 for the
// hdim256 GQA full-attn — flash_fwd_launch_template.h:170) maps to exactly ONE
// physical page. That makes the kernel's per-tile block_table lookup
// per-page-correct and removes the intra-tile physical-contiguity requirement
// entirely — so concurrent decode no longer fragments into the slow per-row
// loop (the n=64 cliff). 64 >= max kBlockN and divides cleanly everywhere.
const DEFAULT_BLOCK_SIZE: usize = 64;
const MIN_AUTO_KV_BLOCKS: usize = 64;
const DETERMINISTIC_COMPLETION_CACHE_CAPACITY: usize = 128;
const DETERMINISTIC_CHAT_REQUEST_CACHE_CAPACITY: usize = 128;
const DETERMINISTIC_CHAT_CHOICES_CACHE_CAPACITY: usize = 64;
const DETERMINISTIC_BATCH_CACHE_CAPACITY: usize = 64;
const RENDERED_PROMPT_CACHE_CAPACITY: usize = 256;
const PROMPT_TOKEN_CACHE_CAPACITY: usize = 256;

/// Exact identity of the LoRA weights currently published by the live runner.
///
/// The name remains an operator-facing selector. `content_revision` is derived
/// by the loader from the exact config and safetensor bytes it consumed, so a
/// same-name rewrite is a distinct inference and cache identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct LoadedAdapterIdentity {
    pub name: String,
    pub content_revision: String,
}

impl LoadedAdapterIdentity {
    pub fn from_source(name: impl Into<String>, source: &LoraSourceIdentity) -> Self {
        Self {
            name: name.into(),
            content_revision: source.content_revision(),
        }
    }
}

/// Fallback inference_memory_fraction values the KV cache auto-sizer tries in
/// order if the configured fraction OOMs at startup. Each entry is only tried
/// when it is strictly less than the configured value, so a user who pins
/// `inference_memory_fraction=0.5` gets retries at 0.45, not at 0.85→0.75→...
///
/// The descending shape (large initial step, smaller subsequent steps) matches
/// what we see in practice on A40/A6000-class 48 GiB cards with Qwen3.5-4B BF16:
/// the top of the fraction curve is the danger zone for activation peaks plus
/// driver overhead, and once you drop ~10 percentage points you have plenty of
/// room. See `phase11-685-autosizer-oom-default` and issue #685 for context.
const AUTO_SIZER_FALLBACK_FRACTIONS: &[f64] = &[0.75, 0.65, 0.55, 0.45];

/// GPU memory budget tracking for coordinating inference and training.
///
/// On startup, we compute how much VRAM is available and partition it:
/// - Model weights (fixed)
/// - KV cache for inference (controlled by KILN_INFERENCE_MEMORY_FRACTION)
/// - Remaining budget available for training
#[derive(Debug, Serialize)]
pub struct GpuMemoryBudget {
    /// Total GPU memory in bytes (0 if CPU-only).
    pub total_vram_bytes: u64,
    /// Post-load CUDA residency in bytes, or the static model estimate when
    /// runtime residency is unavailable.
    pub model_memory_bytes: u64,
    /// Static model parameter estimate in bytes.
    pub estimated_model_memory_bytes: u64,
    /// Post-load CUDA residency snapshot in bytes (0 when unavailable).
    pub post_load_used_vram_bytes: u64,
    /// Peak post-prefill CUDA residency observed at request boundaries.
    #[serde(skip)]
    pub peak_prefill_used_vram_bytes: std::sync::atomic::AtomicU64,
    /// KV cache allocation in bytes.
    pub kv_cache_bytes: u64,
    /// Memory available for training in bytes.
    pub training_budget_bytes: u64,
    /// Fraction of VRAM reserved for inference (KV cache). Default 0.7.
    pub inference_memory_fraction: f64,
}

impl GpuMemoryBudget {
    /// Compute the memory budget given model config and allocation parameters.
    ///
    /// `total_vram_bytes`: Total GPU VRAM (0 for CPU).
    /// `model_memory_bytes`: post-load residency or static model estimate.
    /// `kv_cache_bytes`: Actual KV cache allocation size.
    /// `inference_fraction`: Fraction of VRAM for inference.
    /// `training_memory_gb`: Optional cap on the remaining training budget in GiB.
    pub fn compute(
        total_vram_bytes: u64,
        model_memory_bytes: u64,
        estimated_model_memory_bytes: u64,
        post_load_used_vram_bytes: u64,
        kv_cache_bytes: u64,
        inference_fraction: f64,
        training_memory_gb: Option<f64>,
    ) -> Self {
        let training_budget_bytes = if total_vram_bytes == 0 {
            // CPU mode — no GPU memory budget applies
            0
        } else {
            let remaining = total_vram_bytes
                .saturating_sub(model_memory_bytes)
                .saturating_sub(kv_cache_bytes);
            training_memory_gb.map_or(remaining, |gib| {
                let requested = (gib * 1024.0 * 1024.0 * 1024.0) as u64;
                requested.min(remaining)
            })
        };

        Self {
            total_vram_bytes,
            model_memory_bytes,
            estimated_model_memory_bytes,
            post_load_used_vram_bytes,
            peak_prefill_used_vram_bytes: std::sync::atomic::AtomicU64::new(0),
            kv_cache_bytes,
            training_budget_bytes,
            inference_memory_fraction: inference_fraction,
        }
    }

    pub fn peak_prefill_used_vram_bytes(&self) -> u64 {
        self.peak_prefill_used_vram_bytes
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn observe_prefill_used_vram_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut current = self.peak_prefill_used_vram_bytes();
        while bytes > current {
            match self.peak_prefill_used_vram_bytes.compare_exchange_weak(
                current,
                bytes,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }

    /// Check if there is enough memory for training. Returns an error message if not.
    pub fn check_training_feasible(&self, estimated_training_bytes: u64) -> Result<(), String> {
        if self.total_vram_bytes == 0 {
            // CPU mode — no GPU budget enforcement
            return Ok(());
        }
        if estimated_training_bytes > self.training_budget_bytes {
            return Err(format!(
                "insufficient GPU memory for training: need ~{:.1}GB but only {:.1}GB available \
                 (total {:.1}GB - model {:.1}GB - KV cache {:.1}GB). \
                 Try reducing KILN_NUM_BLOCKS or setting KILN_INFERENCE_MEMORY_FRACTION lower",
                estimated_training_bytes as f64 / 1e9,
                self.training_budget_bytes as f64 / 1e9,
                self.total_vram_bytes as f64 / 1e9,
                self.model_memory_bytes as f64 / 1e9,
                self.kv_cache_bytes as f64 / 1e9,
            ));
        }
        Ok(())
    }
}

/// Coordination lock for GPU memory sharing between inference and training.
///
/// Inference acquires a read lock (multiple concurrent inference requests OK).
/// Training acquires a write lock (blocks inference during gradient computation).
/// This prevents combined peak VRAM from exceeding GPU capacity.
///
/// Training should acquire this per-segment (for gradient-checkpointed training),
/// not for the entire job, to minimize inference latency impact.
pub type GpuCoordinationLock = Arc<RwLock<()>>;

pub(crate) fn gpu_coordination_read_guard(
    gpu_lock: &GpuCoordinationLock,
) -> OwnedRwLockReadGuard<()> {
    futures::executor::block_on(gpu_lock.clone().read_owned())
}

#[cfg(test)]
pub(crate) fn gpu_coordination_write_guard(
    gpu_lock: &GpuCoordinationLock,
) -> OwnedRwLockWriteGuard<()> {
    futures::executor::block_on(gpu_lock.clone().write_owned())
}

const GPU_COORDINATION_HEALTH_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// Wait for exclusive GPU ownership without entering an uninterruptible wait
/// behind an inference owner whose completion state has been quarantined.
///
/// Quarantine deliberately leaks unknown GPU ownership. Polling `try_write`
/// keeps writers responsive to that process-lifetime latch, and the second
/// health check closes the acquisition race before the caller can mutate GPU
/// state.
pub(crate) fn gpu_coordination_write_guard_while_healthy(
    gpu_lock: &GpuCoordinationLock,
    backend_health: &BackendHealthHandle,
) -> anyhow::Result<OwnedRwLockWriteGuard<()>> {
    loop {
        backend_health.ensure_healthy()?;
        if let Ok(guard) = gpu_lock.clone().try_write_owned() {
            backend_health.ensure_healthy()?;
            return Ok(guard);
        }
        std::thread::sleep(GPU_COORDINATION_HEALTH_POLL);
    }
}

/// Async counterpart of [`gpu_coordination_write_guard_while_healthy`].
pub(crate) async fn gpu_coordination_write_guard_while_healthy_async(
    gpu_lock: &GpuCoordinationLock,
    backend_health: &BackendHealthHandle,
) -> anyhow::Result<OwnedRwLockWriteGuard<()>> {
    loop {
        backend_health.ensure_healthy()?;
        if let Ok(guard) = gpu_lock.clone().try_write_owned() {
            backend_health.ensure_healthy()?;
            return Ok(guard);
        }
        tokio::time::sleep(GPU_COORDINATION_HEALTH_POLL).await;
    }
}

/// Type of training job.
#[derive(Debug, Clone, Copy, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingJobType {
    Sft,
    Grpo,
    /// On-Policy Distillation (§3.1 of the grand plan). Sampling +
    /// teacher reverse-KL + importance-sampling loss. Same hot-swap
    /// semantics as SFT/GRPO.
    Opd,
}

/// Native training workload whose immutable server substrate is being queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainingWorkload {
    Sft,
    Grpo,
    Opd,
    DistillRefresh,
}

pub(crate) const DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE: &str = "distill_refresh is unavailable until admission pins separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves the maximum sequential working set";

impl TrainingWorkload {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Sft => "sft",
            Self::Grpo => "grpo",
            Self::Opd => "opd",
            Self::DistillRefresh => "distill_refresh",
        }
    }
}

fn training_workload_route_unavailable_reason(
    workload: TrainingWorkload,
    capabilities: kiln_model::backend::TrainingCapabilities,
    checkpoint: kiln_train::CheckpointConfig,
) -> Option<String> {
    use kiln_model::backend::{
        OpdLossRoute, OpdPhaseBBackwardRoute, SftFlceLossRoute, TrainingTapeRoute,
    };

    if workload == TrainingWorkload::DistillRefresh {
        return Some(DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE.to_string());
    }

    if capabilities.tape_forward_backward_route != TrainingTapeRoute::KtTapeAuthoritative {
        return Some(format!(
            "{} training requires tape route `kt_tape_authoritative`, but backend route is `{}`",
            workload.label(),
            capabilities.tape_forward_backward_route.as_str(),
        ));
    }

    match workload {
        TrainingWorkload::Sft
            if checkpoint.enabled
                && checkpoint.num_segments > 1
                && capabilities.sft_flce_loss_route == SftFlceLossRoute::FullLogits =>
        {
            Some(format!(
                "sft training cannot combine effective checkpointing ({} segments) with backend loss route `full_logits`",
                checkpoint.num_segments,
            ))
        }
        TrainingWorkload::Opd if capabilities.opd_loss_route == OpdLossRoute::Unsupported => {
            Some("opd training backend loss route is `unsupported`".to_string())
        }
        TrainingWorkload::Opd
            if capabilities.opd_phase_b_backward_route == OpdPhaseBBackwardRoute::Unsupported =>
        {
            Some("opd training backend phase-B backward route is `unsupported`".to_string())
        }
        TrainingWorkload::Sft | TrainingWorkload::Grpo | TrainingWorkload::Opd => None,
        TrainingWorkload::DistillRefresh => unreachable!("handled above"),
    }
}

fn now_instant_default() -> std::time::Instant {
    std::time::Instant::now()
}

/// Machine-readable §8.7 promotion-gate outcome, stamped on the training
/// job alongside the prose `post_eval_verdict`. The prose is for humans;
/// this is for the dashboard pill and API consumers — before it existed
/// the UI classified verdicts by substring matching and a PASSED gate
/// could render as a warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOutcome {
    /// Gate passed and the adapter was swapped into serving.
    Promoted,
    /// Gate passed; adapter kept on disk but no promotion was requested
    /// (`auto_load` off). A success, not a warning.
    Kept,
    /// Rejected relative to the previous generation: paired sign-test
    /// regression, or the distill_refresh recovery/gain thresholds.
    Regression,
    /// Failed the accuracy floor: adapter demoted to `<name>.failed`.
    Demoted,
    /// The gate could not measure or apply: eval errored/cancelled,
    /// produced no run, or the promotion swap itself failed.
    Error,
}

impl GateOutcome {
    /// Wire/persistence representation. Kept lowercase and stable — the
    /// dashboard pill colors key off these exact strings.
    pub fn as_str(self) -> &'static str {
        match self {
            GateOutcome::Promoted => "promoted",
            GateOutcome::Kept => "kept",
            GateOutcome::Regression => "regression",
            GateOutcome::Demoted => "demoted",
            GateOutcome::Error => "error",
        }
    }
}

/// Tracked training job info stored in AppState.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct TrainingJobInfo {
    pub job_id: String,
    pub adapter_name: String,
    pub job_type: TrainingJobType,
    /// Immutable effective seed materialized before the job is published.
    /// `None` is reserved for legacy archived jobs.
    #[serde(
        default,
        with = "kiln_eval::result::optional_u64_decimal",
        skip_serializing_if = "Option::is_none"
    )]
    pub effective_seed: Option<u64>,
    pub state: TrainingState,
    pub progress: f32,
    pub loss: Option<f64>,
    pub epoch: Option<u32>,
    pub adapter_path: Option<String>,
    #[serde(skip, default = "now_instant_default")]
    pub submitted_at: std::time::Instant,
    /// Wall-clock submit time as Unix milliseconds. Set at enqueue. Survives
    /// process restarts (the `Instant` above does not). Defaults to the load
    /// timestamp for legacy archived entries without this field.
    #[serde(default = "crate::recent_requests::now_unix_ms")]
    pub submitted_unix_ms: u64,
    pub auto_load: bool,
    /// Correction-row request_ids this job consumed (the
    /// `corrections:active` dataset). On COMPLETION — not submission —
    /// the queue marks these rows trained_into the produced adapter, so
    /// a failed job leaves the basket intact and re-trainable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub consumed_correction_ids: Vec<String>,
    /// Immutable corpus and partition identity recorded by authoritative
    /// admission. Missing only on legacy archived jobs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_data: Option<kiln_train::TrainingDataProvenance>,
    /// Wall-clock instant at which the job entered a terminal state
    /// (`Completed` or `Failed`). `None` while the job is still
    /// `Queued` or `Running`. Used by the training-queue worker's GC
    /// pass to TTL-evict stale terminal entries from the tracking map.
    /// See `AppState::tracked_job_ttl`.
    #[serde(skip, default)]
    pub finished_at: Option<std::time::Instant>,
    /// Wall-clock terminal-transition time as Unix milliseconds. `None`
    /// while the job is still active. Populated when the job hits
    /// Completed / Failed; persisted to disk so the dashboard can show
    /// "finished 3h ago" even after a restart.
    #[serde(default)]
    pub finished_unix_ms: Option<u64>,
    /// Failure detail for `Failed` jobs (trainer error message, mock-mode
    /// rejection, or "cancelled while queued"). `None` for active and
    /// `Completed` jobs — and for legacy archived entries that predate
    /// the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Eval job IDs that ran against this training run via the
    /// `post_eval` auto-hook. Populated at *enqueue* time by
    /// `enqueue_post_training_eval` (so the training-side dashboard can
    /// link to the eval the moment it lands in the queue, not after it
    /// finishes). Empty when no post-training eval was requested.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linked_eval_job_ids: Vec<String>,
    /// §8.7 gate verdict stamped by the eval worker when the request
    /// carried `post_eval.min_accuracy`: what the measured accuracy was,
    /// whether the adapter was promoted, demoted to `<name>.failed`, or
    /// left unpromoted because the eval itself errored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval_verdict: Option<String>,
    /// Machine-readable classification of `post_eval_verdict`:
    /// `promoted | kept | regression | demoted | error` (see
    /// [`GateOutcome`]). Stamped together with the prose verdict so API
    /// consumers and the dashboard pill never have to classify prose by
    /// substring. `None` for ungated jobs and for archives stamped
    /// before the field existed. Stored as a plain string (not the enum)
    /// so a future outcome value never fails deserialization of an
    /// archived job file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_outcome: Option<String>,
    /// Cooperative cancellation flag for a RUNNING job: set by
    /// `DELETE /v1/train/queue/{id}`, read by the training worker's
    /// per-step progress callback, which then returns
    /// `TrainControl::Stop` and the trainer aborts at the next step
    /// boundary. Shared (`Arc`) so the in-flight worker observes flips
    /// on the tracked entry. Never serialized.
    #[serde(skip, default)]
    pub cancel_requested: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Loss history for live charts. Each sample is
    /// `{epoch, progress, loss, elapsed_secs}`. Capped at
    /// `TRAINING_LOSS_HISTORY_CAP` to bound memory; once full, every
    /// second sample is dropped (downsampled in-place) so the curve
    /// retains shape without unbounded growth.
    #[serde(default)]
    pub loss_history: Vec<TrainingLossSample>,
}

/// Single point on the live loss curve. Lightweight on purpose — the
/// callback fires on every step.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct TrainingLossSample {
    pub epoch: u32,
    pub progress: f32,
    pub loss: f64,
    pub elapsed_secs: f64,
}

/// Maximum points retained per job. Past this size the in-memory list is
/// downsampled by 2× so a long run still shows a smooth curve.
pub const TRAINING_LOSS_HISTORY_CAP: usize = 512;

/// Append a loss sample to the job's history, downsampling in-place
/// (every-other-sample, last preserved) when the series would exceed
/// `2 * TRAINING_LOSS_HISTORY_CAP`. Downsampling at 2× the cap keeps the
/// amortized cost O(1) per push — the trainer's progress callback runs
/// at every step and previously triggered an O(n) rebuild every step
/// past the cap.
pub fn push_loss_sample(history: &mut Vec<TrainingLossSample>, sample: TrainingLossSample) {
    history.push(sample);
    if history.len() <= 2 * TRAINING_LOSS_HISTORY_CAP {
        return;
    }
    let last = history.last().cloned();
    let mut keep = Vec::with_capacity(TRAINING_LOSS_HISTORY_CAP + 1);
    for (i, s) in history.iter().enumerate() {
        if i % 2 == 0 {
            keep.push(s.clone());
        }
    }
    if let Some(last) = last {
        if keep.last().map(|s| s.elapsed_secs) != Some(last.elapsed_secs) {
            keep.push(last);
        }
    }
    *history = keep;
}

/// Thread-safe map of tracked training jobs.
pub type TrainingJobs = Arc<std::sync::RwLock<HashMap<String, TrainingJobInfo>>>;

pub const MIN_PREFIX_CACHE_MAX_ENTRIES: usize = 1;
const MIN_PREFIX_CACHE_STATE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_PREFIX_CACHE_STATE_BYTES: u64 = 1024 * 1024 * 1024;
const PREFIX_CACHE_STATE_FRACTION_DIVISOR: u64 = 40;
const REAL_PREFIX_CACHE_MIN_REGISTER_TOKENS: usize = 64;

fn effective_prefix_cache_enabled(requested: bool, device: &kiln_tensor::Device) -> bool {
    requested && !matches!(device, kiln_tensor::Device::Vulkan(_))
}

pub struct RealPrefixCache {
    enabled: bool,
    max_blocks: usize,
    max_entries: usize,
    state_bytes_per_entry: u64,
    max_state_bytes: u64,
    min_register_tokens: usize,
    block_size: usize,
    next_entry_id: u64,
    global_generation: u64,
    adapter_generations: HashMap<Option<String>, u64>,
    entries: Vec<RealPrefixCacheEntry>,
    block_refcounts: HashMap<u32, usize>,
    stats: PrefixCacheStats,
}

struct RealPrefixCacheEntry {
    id: u64,
    adapter: Option<LoadedAdapterIdentity>,
    prompt_tokens: Vec<TokenId>,
    block_ids: Vec<u32>,
    linear_state: LinearAttentionState,
    next_token: Option<PagedPrefixNextToken>,
    last_used: u64,
    active_uses: usize,
    retired: bool,
}

pub struct RealPrefixCacheHit {
    pub entry_id: u64,
    pub cached_tokens: usize,
    pub block_ids: Vec<u32>,
    pub linear_state: LinearAttentionState,
    pub next_token: Option<PagedPrefixNextToken>,
}

pub struct RealPrefixCacheRegisterOutcome {
    pub retained_blocks: Vec<u32>,
    pub evicted_blocks: Vec<u32>,
}

pub struct RealPrefixCacheRequestLookup {
    pub request: RealPrefixCacheRequest,
    pub hit: Option<RealPrefixCacheHit>,
    pub should_register: bool,
}

/// A prefix lookup whose successful hit snapshot may still own asynchronous
/// device-to-device copies.
///
/// The hit and its request lease remain inaccessible until [`Self::settle`]
/// proves those copies complete. Dropping an unsettled hit intentionally
/// retains both the snapshot and lease so their storage cannot be recycled.
#[must_use = "prefix-cache lookups must be settled before their hit state can be used"]
pub struct RealPrefixCachePendingLookup {
    request: Option<RealPrefixCacheRequest>,
    hit: Option<RealPrefixCacheHit>,
    should_register: bool,
}

impl RealPrefixCachePendingLookup {
    pub fn settle(mut self, runner: &ModelRunner) -> anyhow::Result<RealPrefixCacheRequestLookup> {
        self.settle_borrowed(runner)
    }

    pub(crate) fn settle_borrowed(
        &mut self,
        runner: &ModelRunner,
    ) -> anyhow::Result<RealPrefixCacheRequestLookup> {
        self.settle_borrowed_with(|| runner.synchronize_external_yield("prefix-cache hit snapshot"))
    }

    #[cfg(test)]
    fn settle_with(
        mut self,
        synchronize: impl FnOnce() -> anyhow::Result<()>,
    ) -> anyhow::Result<RealPrefixCacheRequestLookup> {
        self.settle_borrowed_with(synchronize)
    }

    fn settle_borrowed_with(
        &mut self,
        synchronize: impl FnOnce() -> anyhow::Result<()>,
    ) -> anyhow::Result<RealPrefixCacheRequestLookup> {
        if self.hit.is_some() {
            synchronize()?;
        }
        Ok(RealPrefixCacheRequestLookup {
            request: self.request.take().expect("pending prefix request present"),
            hit: self.hit.take(),
            should_register: self.should_register,
        })
    }

    #[cfg(test)]
    fn settle_synchronous_for_test(self) -> anyhow::Result<RealPrefixCacheRequestLookup> {
        self.settle_with(|| Ok(()))
    }
}

impl Drop for RealPrefixCachePendingLookup {
    fn drop(&mut self) {
        if self.hit.is_none() {
            return;
        }
        tracing::error!(
            "unsettled prefix-cache hit dropped; retaining its snapshot and source lease"
        );
        if let Some(request) = self.request.take() {
            std::mem::forget(request);
        }
        if let Some(hit) = self.hit.take() {
            std::mem::forget(hit);
        }
    }
}

/// Owns a provisional cache lease when hit-state snapshotting fails.
///
/// Call [`Self::settle`] while the request still owns its GPU coordination
/// permit. Dropping this value without settlement intentionally leaks the
/// lease, which keeps the source entry and its pages quarantined.
#[must_use = "prefix-cache begin failures must be settled before GPU ownership is released"]
pub struct RealPrefixCacheBeginFailure {
    error: Option<anyhow::Error>,
    request: Option<RealPrefixCacheRequest>,
}

impl RealPrefixCacheBeginFailure {
    fn without_request(error: anyhow::Error) -> Self {
        Self {
            error: Some(error),
            request: None,
        }
    }

    fn with_request(error: anyhow::Error, request: RealPrefixCacheRequest) -> Self {
        Self {
            error: Some(error),
            request: Some(request),
        }
    }

    pub fn settle(mut self, runner: &ModelRunner) -> anyhow::Error {
        self.settle_borrowed(runner)
    }

    pub(crate) fn settle_borrowed(&mut self, runner: &ModelRunner) -> anyhow::Error {
        let error = self.error.take().expect("prefix begin error present");
        if self.request.is_none() {
            return error;
        }
        match runner.synchronize_external_yield("prefix-cache hit snapshot failure") {
            Ok(()) => {
                let request = self
                    .request
                    .take()
                    .expect("provisional prefix request present after synchronization");
                drop(request);
                runner.backend_health_handle().quarantine(format!(
                    "prefix-cache hit snapshot failed after partial device-copy submission: {error:#}"
                ));
                error
            }
            Err(sync_error) => {
                let request = self
                    .request
                    .take()
                    .expect("provisional prefix request present after synchronization failure");
                std::mem::forget(request);
                sync_error.context(format!(
                    "prefix-cache hit snapshot also failed before synchronization: {error:#}"
                ))
            }
        }
    }
}

impl Drop for RealPrefixCacheBeginFailure {
    fn drop(&mut self) {
        let Some(request) = self.request.take() else {
            return;
        };
        tracing::error!("unsettled prefix-cache begin failure dropped; retaining its source lease");
        std::mem::forget(request);
    }
}

impl std::fmt::Debug for RealPrefixCacheBeginFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealPrefixCacheBeginFailure")
            .field("error", &self.error.as_ref().map(ToString::to_string))
            .field("has_provisional_lease", &self.request.is_some())
            .finish()
    }
}

impl std::fmt::Display for RealPrefixCacheBeginFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.error.as_ref() {
            Some(error) => std::fmt::Display::fmt(error, formatter),
            None => formatter.write_str("prefix-cache begin failure already consumed"),
        }
    }
}

impl std::error::Error for RealPrefixCacheBeginFailure {}

struct RealPrefixCacheLookupAttempt {
    hit: anyhow::Result<Option<RealPrefixCacheHit>>,
    leased_entry_id: Option<u64>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct RealPrefixCacheFinishOutcome {
    pub retained_blocks: Vec<u32>,
    pub released_blocks: Vec<u32>,
    pub registrations_accepted: bool,
}

/// Move-only ownership for one cache-enabled generation request.
///
/// A request without a hit still carries generation fences so an adapter
/// purge or global clear cannot be undone by stale prefill completing later.
/// Dropping a hit request releases its lease and reclaims a retired entry's
/// blocks, which makes error and cancellation paths fail closed.
#[must_use = "dropping a prefix-cache request abandons its registration and releases its hit lease"]
pub struct RealPrefixCacheRequest {
    cache: Arc<std::sync::Mutex<RealPrefixCache>>,
    block_manager: Arc<std::sync::Mutex<BlockManager>>,
    adapter: Option<LoadedAdapterIdentity>,
    global_generation: u64,
    adapter_generation: u64,
    hit_entry_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeterministicCompletionCacheKey {
    pub adapter: Option<LoadedAdapterIdentity>,
    pub global_generation: u64,
    pub adapter_generation: u64,
    pub prompt_tokens: Vec<TokenId>,
    pub temperature_bits: u32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
    pub thinking_budget_tokens: Option<usize>,
    pub stop: Vec<String>,
    pub top_p_bits: u32,
    pub top_k: u32,
    pub min_p_bits: u32,
    pub presence_penalty_bits: u32,
    pub frequency_penalty_bits: u32,
    pub repetition_penalty_bits: u32,
    pub seed: Option<u64>,
    pub fold_reasoning_into_content: bool,
}

#[derive(Debug, Clone)]
pub struct DeterministicCompletionCacheValue {
    pub text: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub finish_reason: String,
    pub completion_tokens: usize,
    pub thinking_budget_status: Option<ThinkingBudgetStatus>,
}

#[derive(Debug, Clone)]
pub enum DeterministicCompletionInFlightState {
    Pending,
    Ready(Option<DeterministicCompletionCacheValue>),
}

pub enum DeterministicCompletionCacheClaim {
    Hit(DeterministicCompletionCacheValue),
    Wait(watch::Receiver<DeterministicCompletionInFlightState>),
    Owner(DeterministicCacheClaimId),
}

pub enum DeterministicCompletionCacheProbe {
    Hit(DeterministicCompletionCacheValue),
    Wait(watch::Receiver<DeterministicCompletionInFlightState>),
    Miss,
}

pub struct DeterministicCompletionCache {
    capacity: usize,
    next_claim_id: u64,
    entries: HashMap<DeterministicCompletionCacheKey, DeterministicCompletionCacheValue>,
    lru: VecDeque<DeterministicCompletionCacheKey>,
    in_flight: HashMap<
        DeterministicCompletionCacheKey,
        (
            DeterministicCacheClaimId,
            watch::Sender<DeterministicCompletionInFlightState>,
        ),
    >,
}

#[derive(Debug, Clone)]
pub struct DeterministicChatRequestCacheValue {
    pub prompt_tokens: usize,
    pub completion: DeterministicCompletionCacheValue,
}

#[derive(Debug, Clone)]
pub enum DeterministicChatRequestInFlightState {
    Pending,
    Ready(Option<DeterministicChatRequestCacheValue>),
}

pub enum DeterministicChatRequestCacheClaim {
    Hit(DeterministicChatRequestCacheValue),
    Wait(watch::Receiver<DeterministicChatRequestInFlightState>),
    Owner(DeterministicCacheClaimId),
}

pub enum DeterministicChatRequestCacheProbe {
    Hit(DeterministicChatRequestCacheValue),
    Wait(watch::Receiver<DeterministicChatRequestInFlightState>),
    Miss,
}

pub struct DeterministicChatRequestCache {
    capacity: usize,
    next_claim_id: u64,
    entries: HashMap<DeterministicCacheKey, DeterministicChatRequestCacheValue>,
    lru: VecDeque<DeterministicCacheKey>,
    in_flight: HashMap<
        DeterministicCacheKey,
        (
            DeterministicCacheClaimId,
            watch::Sender<DeterministicChatRequestInFlightState>,
        ),
    >,
}

#[derive(Debug, Clone)]
pub struct DeterministicChatChoicesCacheValue {
    pub prompt_tokens: usize,
    pub completions: Vec<DeterministicCompletionCacheValue>,
}

#[derive(Debug, Clone)]
pub enum DeterministicChatChoicesInFlightState {
    Pending,
    Ready(Option<DeterministicChatChoicesCacheValue>),
}

pub enum DeterministicChatChoicesCacheClaim {
    Hit(DeterministicChatChoicesCacheValue),
    Wait(watch::Receiver<DeterministicChatChoicesInFlightState>),
    Owner(DeterministicCacheClaimId),
}

pub enum DeterministicChatChoicesCacheProbe {
    Hit(DeterministicChatChoicesCacheValue),
    Wait(watch::Receiver<DeterministicChatChoicesInFlightState>),
    Miss,
}

pub struct DeterministicChatChoicesCache {
    capacity: usize,
    next_claim_id: u64,
    entries: HashMap<DeterministicCacheKey, DeterministicChatChoicesCacheValue>,
    lru: VecDeque<DeterministicCacheKey>,
    in_flight: HashMap<
        DeterministicCacheKey,
        (
            DeterministicCacheClaimId,
            watch::Sender<DeterministicChatChoicesInFlightState>,
        ),
    >,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicCacheClaimId(u64);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeterministicCacheKey {
    pub adapter: Option<LoadedAdapterIdentity>,
    pub global_generation: u64,
    pub adapter_generation: u64,
    pub request: String,
}

impl DeterministicCacheKey {
    pub fn new(adapter: Option<LoadedAdapterIdentity>, request: String) -> Self {
        Self {
            adapter,
            global_generation: 0,
            adapter_generation: 0,
            request,
        }
    }
}

#[derive(Default)]
struct DeterministicCacheGenerations {
    global: u64,
    adapters: HashMap<Option<String>, u64>,
}

impl DeterministicCacheGenerations {
    fn snapshot(&self, adapter: &Option<LoadedAdapterIdentity>) -> (u64, u64) {
        let name = adapter.as_ref().map(|identity| identity.name.clone());
        (self.global, self.adapters.get(&name).copied().unwrap_or(0))
    }

    fn purge_adapter(&mut self, adapter: &Option<String>) {
        let generation = self.adapters.entry(adapter.clone()).or_default();
        *generation = generation
            .checked_add(1)
            .expect("deterministic cache adapter generation overflow");
    }

    fn clear(&mut self) {
        self.global = self
            .global
            .checked_add(1)
            .expect("deterministic cache global generation overflow");
        self.adapters.clear();
    }
}

pub type DeterministicBatchCacheKey = DeterministicCacheKey;

#[derive(Debug, Clone)]
pub struct DeterministicBatchCacheItem {
    pub prompt_index: usize,
    pub completion_index: usize,
    pub text: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub thinking_budget_status: Option<ThinkingBudgetStatus>,
}

#[derive(Debug, Clone)]
pub struct DeterministicBatchCacheValue {
    pub completions: Vec<DeterministicBatchCacheItem>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone)]
pub enum DeterministicBatchInFlightState {
    Pending,
    Ready(Option<DeterministicBatchCacheValue>),
}

pub enum DeterministicBatchCacheClaim {
    Hit(DeterministicBatchCacheValue),
    Wait(watch::Receiver<DeterministicBatchInFlightState>),
    Owner(DeterministicCacheClaimId),
}

pub struct DeterministicBatchCache {
    capacity: usize,
    next_claim_id: u64,
    entries: HashMap<DeterministicBatchCacheKey, DeterministicBatchCacheValue>,
    lru: VecDeque<DeterministicBatchCacheKey>,
    in_flight: HashMap<
        DeterministicBatchCacheKey,
        (
            DeterministicCacheClaimId,
            watch::Sender<DeterministicBatchInFlightState>,
        ),
    >,
}

fn allocate_cache_claim_id(next_claim_id: &mut u64) -> DeterministicCacheClaimId {
    let claim_id = DeterministicCacheClaimId(*next_claim_id);
    *next_claim_id = next_claim_id
        .checked_add(1)
        .expect("deterministic cache claim id overflow");
    claim_id
}

impl DeterministicBatchCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_claim_id: 1,
            entries: HashMap::new(),
            lru: VecDeque::with_capacity(capacity),
            in_flight: HashMap::new(),
        }
    }

    pub fn claim(&mut self, key: &DeterministicBatchCacheKey) -> DeterministicBatchCacheClaim {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicBatchCacheClaim::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicBatchCacheClaim::Wait(sender.subscribe());
        }

        let (sender, _receiver) = watch::channel(DeterministicBatchInFlightState::Pending);
        let claim_id = allocate_cache_claim_id(&mut self.next_claim_id);
        self.in_flight.insert(key.clone(), (claim_id, sender));
        DeterministicBatchCacheClaim::Owner(claim_id)
    }

    pub fn complete(
        &mut self,
        key: DeterministicBatchCacheKey,
        claim_id: DeterministicCacheClaimId,
        value: DeterministicBatchCacheValue,
    ) -> bool {
        if self
            .in_flight
            .get(&key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        self.insert_complete_value(key.clone(), value.clone());
        if let Some((_, sender)) = self.in_flight.remove(&key) {
            let _ = sender.send(DeterministicBatchInFlightState::Ready(Some(value)));
        }
        true
    }

    pub fn fail(
        &mut self,
        key: &DeterministicBatchCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> bool {
        if self
            .in_flight
            .get(key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        if let Some((_, sender)) = self.in_flight.remove(key) {
            let _ = sender.send(DeterministicBatchInFlightState::Ready(None));
        }
        true
    }

    pub fn insert(&mut self, key: DeterministicBatchCacheKey, value: DeterministicBatchCacheValue) {
        self.insert_complete_value(key, value);
    }

    pub fn clear_completed(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    pub fn clear(&mut self) {
        self.clear_completed();
        for (_, (_, sender)) in self.in_flight.drain() {
            let _ = sender.send(DeterministicBatchInFlightState::Ready(None));
        }
    }

    pub fn purge_adapter(&mut self, adapter: &Option<String>) {
        self.entries.retain(|key, _| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        self.lru.retain(|key| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        let stale: Vec<_> = self
            .in_flight
            .keys()
            .filter(|key| {
                key.adapter.as_ref().map(|identity| identity.name.as_str()) == adapter.as_deref()
            })
            .cloned()
            .collect();
        for key in stale {
            if let Some((_, sender)) = self.in_flight.remove(&key) {
                let _ = sender.send(DeterministicBatchInFlightState::Ready(None));
            }
        }
    }

    fn insert_complete_value(
        &mut self,
        key: DeterministicBatchCacheKey,
        value: DeterministicBatchCacheValue,
    ) {
        if self.capacity == 0 {
            return;
        }

        if self.entries.insert(key.clone(), value).is_some() {
            self.lru.retain(|existing| existing != &key);
            self.lru.push_back(key);
            return;
        }

        while self.entries.len() > self.capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.lru.push_back(key);
    }

    pub fn stats(&self) -> usize {
        self.entries.len()
    }
}

impl DeterministicChatRequestCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_claim_id: 1,
            entries: HashMap::new(),
            lru: VecDeque::with_capacity(capacity),
            in_flight: HashMap::new(),
        }
    }

    pub fn claim(&mut self, key: &DeterministicCacheKey) -> DeterministicChatRequestCacheClaim {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicChatRequestCacheClaim::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicChatRequestCacheClaim::Wait(sender.subscribe());
        }

        let (sender, _receiver) = watch::channel(DeterministicChatRequestInFlightState::Pending);
        let claim_id = allocate_cache_claim_id(&mut self.next_claim_id);
        self.in_flight.insert(key.clone(), (claim_id, sender));
        DeterministicChatRequestCacheClaim::Owner(claim_id)
    }

    pub fn probe(&mut self, key: &DeterministicCacheKey) -> DeterministicChatRequestCacheProbe {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicChatRequestCacheProbe::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicChatRequestCacheProbe::Wait(sender.subscribe());
        }

        DeterministicChatRequestCacheProbe::Miss
    }

    pub fn complete(
        &mut self,
        key: DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
        value: DeterministicChatRequestCacheValue,
    ) -> bool {
        if self
            .in_flight
            .get(&key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        self.insert_complete_value(key.clone(), value.clone());
        if let Some((_, sender)) = self.in_flight.remove(&key) {
            let _ = sender.send(DeterministicChatRequestInFlightState::Ready(Some(value)));
        }
        true
    }

    pub fn fail(
        &mut self,
        key: &DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> bool {
        if self
            .in_flight
            .get(key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        if let Some((_, sender)) = self.in_flight.remove(key) {
            let _ = sender.send(DeterministicChatRequestInFlightState::Ready(None));
        }
        true
    }

    pub fn insert(
        &mut self,
        key: DeterministicCacheKey,
        value: DeterministicChatRequestCacheValue,
    ) {
        self.insert_complete_value(key, value);
    }

    pub fn clear_completed(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    pub fn clear(&mut self) {
        self.clear_completed();
        for (_, (_, sender)) in self.in_flight.drain() {
            let _ = sender.send(DeterministicChatRequestInFlightState::Ready(None));
        }
    }

    fn insert_complete_value(
        &mut self,
        key: DeterministicCacheKey,
        value: DeterministicChatRequestCacheValue,
    ) {
        if self.capacity == 0 {
            return;
        }

        if self.entries.insert(key.clone(), value).is_some() {
            self.lru.retain(|existing| existing != &key);
            self.lru.push_back(key);
            return;
        }

        while self.entries.len() > self.capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.lru.push_back(key);
    }

    pub fn stats(&self) -> usize {
        self.entries.len()
    }

    pub fn purge_adapter(&mut self, adapter: &Option<String>) {
        self.entries.retain(|key, _| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        self.lru.retain(|key| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        let stale: Vec<_> = self
            .in_flight
            .keys()
            .filter(|key| {
                key.adapter.as_ref().map(|identity| identity.name.as_str()) == adapter.as_deref()
            })
            .cloned()
            .collect();
        for key in stale {
            if let Some((_, sender)) = self.in_flight.remove(&key) {
                let _ = sender.send(DeterministicChatRequestInFlightState::Ready(None));
            }
        }
    }
}

impl DeterministicChatChoicesCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_claim_id: 1,
            entries: HashMap::new(),
            lru: VecDeque::with_capacity(capacity),
            in_flight: HashMap::new(),
        }
    }

    pub fn claim(&mut self, key: &DeterministicCacheKey) -> DeterministicChatChoicesCacheClaim {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicChatChoicesCacheClaim::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicChatChoicesCacheClaim::Wait(sender.subscribe());
        }

        let (sender, _receiver) = watch::channel(DeterministicChatChoicesInFlightState::Pending);
        let claim_id = allocate_cache_claim_id(&mut self.next_claim_id);
        self.in_flight.insert(key.clone(), (claim_id, sender));
        DeterministicChatChoicesCacheClaim::Owner(claim_id)
    }

    pub fn probe(&mut self, key: &DeterministicCacheKey) -> DeterministicChatChoicesCacheProbe {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicChatChoicesCacheProbe::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicChatChoicesCacheProbe::Wait(sender.subscribe());
        }

        DeterministicChatChoicesCacheProbe::Miss
    }

    pub fn complete(
        &mut self,
        key: DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
        value: DeterministicChatChoicesCacheValue,
    ) -> bool {
        if self
            .in_flight
            .get(&key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        self.insert_complete_value(key.clone(), value.clone());
        if let Some((_, sender)) = self.in_flight.remove(&key) {
            let _ = sender.send(DeterministicChatChoicesInFlightState::Ready(Some(value)));
        }
        true
    }

    pub fn fail(
        &mut self,
        key: &DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> bool {
        if self
            .in_flight
            .get(key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        if let Some((_, sender)) = self.in_flight.remove(key) {
            let _ = sender.send(DeterministicChatChoicesInFlightState::Ready(None));
        }
        true
    }

    pub fn insert(
        &mut self,
        key: DeterministicCacheKey,
        value: DeterministicChatChoicesCacheValue,
    ) {
        self.insert_complete_value(key, value);
    }

    pub fn clear_completed(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    pub fn clear(&mut self) {
        self.clear_completed();
        for (_, (_, sender)) in self.in_flight.drain() {
            let _ = sender.send(DeterministicChatChoicesInFlightState::Ready(None));
        }
    }

    fn insert_complete_value(
        &mut self,
        key: DeterministicCacheKey,
        value: DeterministicChatChoicesCacheValue,
    ) {
        if self.capacity == 0 {
            return;
        }

        if self.entries.insert(key.clone(), value).is_some() {
            self.lru.retain(|existing| existing != &key);
            self.lru.push_back(key);
            return;
        }

        while self.entries.len() > self.capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.lru.push_back(key);
    }

    pub fn stats(&self) -> usize {
        self.entries.len()
    }

    pub fn purge_adapter(&mut self, adapter: &Option<String>) {
        self.entries.retain(|key, _| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        self.lru.retain(|key| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        let stale: Vec<_> = self
            .in_flight
            .keys()
            .filter(|key| {
                key.adapter.as_ref().map(|identity| identity.name.as_str()) == adapter.as_deref()
            })
            .cloned()
            .collect();
        for key in stale {
            if let Some((_, sender)) = self.in_flight.remove(&key) {
                let _ = sender.send(DeterministicChatChoicesInFlightState::Ready(None));
            }
        }
    }
}

impl DeterministicCompletionCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_claim_id: 1,
            entries: HashMap::new(),
            lru: VecDeque::with_capacity(capacity),
            in_flight: HashMap::new(),
        }
    }

    pub fn claim(
        &mut self,
        key: &DeterministicCompletionCacheKey,
    ) -> DeterministicCompletionCacheClaim {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicCompletionCacheClaim::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicCompletionCacheClaim::Wait(sender.subscribe());
        }

        let (sender, _receiver) = watch::channel(DeterministicCompletionInFlightState::Pending);
        let claim_id = allocate_cache_claim_id(&mut self.next_claim_id);
        self.in_flight.insert(key.clone(), (claim_id, sender));
        DeterministicCompletionCacheClaim::Owner(claim_id)
    }

    pub fn probe(
        &mut self,
        key: &DeterministicCompletionCacheKey,
    ) -> DeterministicCompletionCacheProbe {
        if let Some(value) = self.entries.get(key).cloned() {
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.clone());
            return DeterministicCompletionCacheProbe::Hit(value);
        }

        if let Some((_, sender)) = self.in_flight.get(key) {
            return DeterministicCompletionCacheProbe::Wait(sender.subscribe());
        }

        DeterministicCompletionCacheProbe::Miss
    }

    pub fn complete(
        &mut self,
        key: DeterministicCompletionCacheKey,
        claim_id: DeterministicCacheClaimId,
        value: DeterministicCompletionCacheValue,
    ) -> bool {
        if self
            .in_flight
            .get(&key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        self.insert_complete_value(key.clone(), value.clone());
        if let Some((_, sender)) = self.in_flight.remove(&key) {
            let _ = sender.send(DeterministicCompletionInFlightState::Ready(Some(value)));
        }
        true
    }

    pub fn fail(
        &mut self,
        key: &DeterministicCompletionCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> bool {
        if self
            .in_flight
            .get(key)
            .map(|(active_claim, _)| *active_claim)
            != Some(claim_id)
        {
            return false;
        }
        if let Some((_, sender)) = self.in_flight.remove(key) {
            let _ = sender.send(DeterministicCompletionInFlightState::Ready(None));
        }
        true
    }

    pub fn insert_complete_value(
        &mut self,
        key: DeterministicCompletionCacheKey,
        value: DeterministicCompletionCacheValue,
    ) {
        if self.capacity == 0 {
            return;
        }

        if self.entries.insert(key.clone(), value).is_some() {
            self.lru.retain(|existing| existing != &key);
            self.lru.push_back(key);
            return;
        }

        while self.entries.len() > self.capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.lru.push_back(key);
    }

    /// Store a value produced by a probe-only caller without disturbing a
    /// request that claimed the same key after that probe returned.
    pub fn insert_unowned_complete_value(
        &mut self,
        key: DeterministicCompletionCacheKey,
        value: DeterministicCompletionCacheValue,
    ) -> bool {
        if self.in_flight.contains_key(&key) {
            return false;
        }
        self.insert_complete_value(key, value);
        true
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        for (_, (_, sender)) in self.in_flight.drain() {
            let _ = sender.send(DeterministicCompletionInFlightState::Ready(None));
        }
    }

    pub fn clear_completed(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    /// Drop every entry keyed to `adapter` (completed and in-flight),
    /// leaving other adapters' entries intact. In-flight waiters get
    /// `Ready(None)` — the recompute path — since the value they're
    /// waiting on was produced under weights that no longer exist.
    pub fn purge_adapter(&mut self, adapter: &Option<String>) {
        self.entries.retain(|key, _| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        self.lru.retain(|key| {
            key.adapter.as_ref().map(|identity| identity.name.as_str()) != adapter.as_deref()
        });
        let stale: Vec<DeterministicCompletionCacheKey> = self
            .in_flight
            .keys()
            .filter(|key| {
                key.adapter.as_ref().map(|identity| identity.name.as_str()) == adapter.as_deref()
            })
            .cloned()
            .collect();
        for key in stale {
            if let Some((_, sender)) = self.in_flight.remove(&key) {
                let _ = sender.send(DeterministicCompletionInFlightState::Ready(None));
            }
        }
    }

    pub fn stats(&self) -> usize {
        self.entries.len()
    }
}

impl RealPrefixCacheRequest {
    pub fn begin(
        cache: &Arc<std::sync::Mutex<RealPrefixCache>>,
        block_manager: &Arc<std::sync::Mutex<BlockManager>>,
        adapter: Option<LoadedAdapterIdentity>,
        prompt_tokens: &[TokenId],
        sampling: &SamplingParams,
    ) -> Result<RealPrefixCachePendingLookup, RealPrefixCacheBeginFailure> {
        let mut cache_guard = cache.lock().map_err(|err| {
            RealPrefixCacheBeginFailure::without_request(anyhow::anyhow!(
                "prefix cache lock poisoned: {err}"
            ))
        })?;
        if !cache_guard.is_enabled() {
            return Err(RealPrefixCacheBeginFailure::without_request(
                anyhow::anyhow!("cannot begin a request against a disabled prefix cache"),
            ));
        }

        let global_generation = cache_guard.global_generation;
        let adapter_name = adapter.as_ref().map(|identity| identity.name.clone());
        let adapter_generation = cache_guard.adapter_generation(&adapter_name);
        let should_lookup = cache_guard.should_lookup_prompt(prompt_tokens);
        let should_register = cache_guard.should_register_prompt(prompt_tokens);
        let lookup = if should_lookup {
            cache_guard.lookup(&adapter, prompt_tokens, sampling)
        } else {
            RealPrefixCacheLookupAttempt {
                hit: Ok(None),
                leased_entry_id: None,
            }
        };
        let hit_entry_id = lookup.leased_entry_id;
        drop(cache_guard);

        let request = Self {
            cache: Arc::clone(cache),
            block_manager: Arc::clone(block_manager),
            adapter,
            global_generation,
            adapter_generation,
            hit_entry_id,
        };
        match lookup.hit {
            Ok(hit) => Ok(RealPrefixCachePendingLookup {
                request: Some(request),
                hit,
                should_register,
            }),
            Err(error) => Err(RealPrefixCacheBeginFailure::with_request(error, request)),
        }
    }

    pub fn finish(
        mut self,
        registrations: Vec<PagedPrefixRegistration>,
        allocated_blocks: Vec<u32>,
    ) -> RealPrefixCacheFinishOutcome {
        let mut released_blocks = Vec::new();
        let mut final_cache_blocks = HashSet::new();
        let registrations_accepted;

        {
            let Ok(mut cache) = self.cache.lock().map_err(|err| {
                tracing::error!(
                    error = %err,
                    "prefix cache lock poisoned while finishing a request; quarantining its lease"
                );
            }) else {
                // The cached hit remains quarantined because its refcount can
                // no longer be updated safely. Fresh suffix allocations were
                // never published into this poisoned cache, so they remain
                // uniquely owned and can be returned after backend quiescence.
                self.hit_entry_id = None;
                let mut known_private = allocated_blocks;
                known_private.sort_unstable();
                known_private.dedup();
                self.free_blocks(&known_private);
                return RealPrefixCacheFinishOutcome {
                    retained_blocks: Vec::new(),
                    released_blocks: known_private,
                    registrations_accepted: false,
                };
            };
            registrations_accepted = cache.global_generation == self.global_generation
                && cache.adapter_generation(
                    &self.adapter.as_ref().map(|identity| identity.name.clone()),
                ) == self.adapter_generation;
            if registrations_accepted {
                for registration in registrations {
                    let outcome = cache.register(self.adapter.clone(), registration);
                    released_blocks.extend(outcome.evicted_blocks);
                }
            }
            if let Some(entry_id) = self.hit_entry_id.take() {
                released_blocks.extend(cache.release_hit(entry_id));
            }
            final_cache_blocks.extend(cache.block_refcounts.keys().copied());
        }

        let mut retained_blocks = Vec::new();
        let mut blocks_to_free = Vec::new();
        let mut seen = HashSet::new();
        for block_id in allocated_blocks {
            if final_cache_blocks.contains(&block_id) {
                if seen.insert(block_id) {
                    retained_blocks.push(block_id);
                }
            } else if seen.insert(block_id) {
                blocks_to_free.push(block_id);
            }
        }
        for block_id in released_blocks {
            if !final_cache_blocks.contains(&block_id) && seen.insert(block_id) {
                blocks_to_free.push(block_id);
            }
        }
        self.free_blocks(&blocks_to_free);

        RealPrefixCacheFinishOutcome {
            retained_blocks,
            released_blocks: blocks_to_free,
            registrations_accepted,
        }
    }

    fn release_lease(&mut self) -> Vec<u32> {
        let Some(entry_id) = self.hit_entry_id.take() else {
            return Vec::new();
        };
        let Ok(mut cache) = self.cache.lock().map_err(|err| {
            tracing::error!(
                error = %err,
                "prefix cache lock poisoned while abandoning a request; quarantining its lease"
            );
        }) else {
            return Vec::new();
        };
        cache.release_hit(entry_id)
    }

    fn free_blocks(&self, block_ids: &[u32]) {
        if block_ids.is_empty() {
            return;
        }
        let Ok(mut block_manager) = self.block_manager.lock().map_err(|err| {
            tracing::error!(
                error = %err,
                "block manager lock poisoned while releasing prefix-cache request blocks; quarantining them"
            );
        }) else {
            return;
        };
        block_manager.free_all(block_ids);
    }
}

impl Drop for RealPrefixCacheRequest {
    fn drop(&mut self) {
        let released_blocks = self.release_lease();
        self.free_blocks(&released_blocks);
    }
}

impl RealPrefixCache {
    pub fn new(
        enabled: bool,
        block_size: usize,
        max_blocks: usize,
        max_entries: usize,
        state_bytes_per_entry: u64,
    ) -> Self {
        Self::new_with_min_register_tokens(
            enabled,
            block_size,
            max_blocks,
            max_entries,
            state_bytes_per_entry,
            0,
        )
    }

    pub fn new_with_min_register_tokens(
        enabled: bool,
        block_size: usize,
        max_blocks: usize,
        max_entries: usize,
        state_bytes_per_entry: u64,
        min_register_tokens: usize,
    ) -> Self {
        let max_entries = max_entries.max(MIN_PREFIX_CACHE_MAX_ENTRIES);
        let max_state_bytes = state_bytes_per_entry.saturating_mul(max_entries as u64);
        Self {
            enabled,
            max_blocks,
            max_entries,
            state_bytes_per_entry,
            max_state_bytes,
            min_register_tokens,
            block_size,
            next_entry_id: 1,
            global_generation: 0,
            adapter_generations: HashMap::new(),
            entries: Vec::new(),
            block_refcounts: HashMap::new(),
            stats: PrefixCacheStats {
                max_blocks,
                ..PrefixCacheStats::default()
            },
        }
    }

    pub fn disabled(block_size: usize) -> Self {
        let mut cache = Self::new(false, block_size, 0, MIN_PREFIX_CACHE_MAX_ENTRIES, 0);
        cache.max_entries = 0;
        cache
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.max_blocks > 0
    }

    pub fn should_register_prompt(&self, prompt_tokens: &[TokenId]) -> bool {
        self.is_enabled()
            && !prompt_tokens.is_empty()
            && prompt_tokens.len() >= self.min_register_tokens
    }

    pub fn can_register_strict_prefix_len(&self, prompt_len: usize) -> bool {
        self.is_enabled()
            && prompt_len >= self.min_register_tokens
            && self.block_size > 0
            && prompt_len % self.block_size == 0
    }

    pub fn should_lookup_prompt(&self, prompt_tokens: &[TokenId]) -> bool {
        self.should_register_prompt(prompt_tokens)
    }

    fn lookup(
        &mut self,
        adapter: &Option<LoadedAdapterIdentity>,
        prompt_tokens: &[TokenId],
        sampling: &SamplingParams,
    ) -> RealPrefixCacheLookupAttempt {
        if !self.is_enabled() {
            return RealPrefixCacheLookupAttempt {
                hit: Ok(None),
                leased_entry_id: None,
            };
        }

        let best_idx = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                if entry.retired {
                    return false;
                }
                let block_aligned =
                    self.block_size > 0 && entry.prompt_tokens.len() % self.block_size == 0;
                let block_shape_valid = block_aligned
                    && entry.block_ids.len() == entry.prompt_tokens.len() / self.block_size;
                let exact_source_compatible =
                    entry
                        .next_token
                        .as_ref()
                        .is_some_and(|source| match source {
                            PagedPrefixNextToken::Logits(_) => true,
                            PagedPrefixNextToken::GreedyToken(_) => {
                                sampling.is_effectively_greedy()
                            }
                        });
                let exact_hit = prompt_tokens.len() == entry.prompt_tokens.len()
                    && exact_source_compatible
                    && block_shape_valid;
                let strict_prefix_hit =
                    prompt_tokens.len() > entry.prompt_tokens.len() && block_shape_valid;
                &entry.adapter == adapter
                    && (exact_hit || strict_prefix_hit)
                    && prompt_tokens.starts_with(&entry.prompt_tokens)
            })
            .max_by_key(|(_, entry)| entry.prompt_tokens.len())
            .map(|(idx, _)| idx);

        let Some(idx) = best_idx else {
            self.stats.lookup_misses = self.stats.lookup_misses.saturating_add(1);
            return RealPrefixCacheLookupAttempt {
                hit: Ok(None),
                leased_entry_id: None,
            };
        };

        let Some(next_active_uses) = self.entries[idx].active_uses.checked_add(1) else {
            return RealPrefixCacheLookupAttempt {
                hit: Err(anyhow::anyhow!(
                    "prefix-cache active lease counter overflow"
                )),
                leased_entry_id: None,
            };
        };
        self.entries[idx].active_uses = next_active_uses;
        let entry_id = self.entries[idx].id;

        // Pin before the first device copy. A partial snapshot failure returns
        // the provisional lease to `RealPrefixCacheBeginFailure`, which may be
        // released only after backend completion is proved.
        let hit = (|| -> anyhow::Result<RealPrefixCacheHit> {
            let entry = &self.entries[idx];
            Ok(RealPrefixCacheHit {
                entry_id: entry.id,
                cached_tokens: entry.prompt_tokens.len(),
                block_ids: entry.block_ids.clone(),
                linear_state: entry.linear_state.snapshot()?,
                next_token: entry.next_token.clone(),
            })
        })();
        let hit = match hit {
            Ok(hit) => hit,
            Err(error) => {
                return RealPrefixCacheLookupAttempt {
                    hit: Err(error),
                    leased_entry_id: Some(entry_id),
                };
            }
        };

        self.stats.lookup_hits = self.stats.lookup_hits.saturating_add(1);
        self.stats.hit_tokens = self
            .stats
            .hit_tokens
            .saturating_add(hit.cached_tokens as u64);
        self.stats.hit_blocks = self
            .stats
            .hit_blocks
            .saturating_add(hit.block_ids.len() as u64);
        self.entries[idx].last_used = self
            .stats
            .lookup_hits
            .saturating_add(self.stats.lookup_misses);
        RealPrefixCacheLookupAttempt {
            hit: Ok(Some(hit)),
            leased_entry_id: Some(entry_id),
        }
    }

    fn release_hit(&mut self, entry_id: u64) -> Vec<u32> {
        let Some(idx) = self.entries.iter().position(|entry| entry.id == entry_id) else {
            tracing::error!(entry_id, "released prefix-cache hit entry does not exist");
            return Vec::new();
        };
        let entry = &mut self.entries[idx];
        let Some(active_uses) = entry.active_uses.checked_sub(1) else {
            tracing::error!(
                entry_id,
                "released prefix-cache hit entry has no active lease"
            );
            return Vec::new();
        };
        entry.active_uses = active_uses;
        if active_uses == 0 && entry.retired {
            let entry = self.entries.remove(idx);
            self.release_entry_blocks(&entry.block_ids)
        } else {
            Vec::new()
        }
    }

    fn register(
        &mut self,
        adapter: Option<LoadedAdapterIdentity>,
        registration: PagedPrefixRegistration,
    ) -> RealPrefixCacheRegisterOutcome {
        let block_aligned =
            self.block_size > 0 && registration.prompt_tokens.len() % self.block_size == 0;
        let block_shape_valid = block_aligned
            && registration.block_ids.len() == registration.prompt_tokens.len() / self.block_size;
        let exact_reusable = registration.next_token.is_some() && block_aligned;
        let strict_prefix_reusable = block_aligned;
        if !self.is_enabled()
            || !self.should_register_prompt(&registration.prompt_tokens)
            || !block_shape_valid
            || (!exact_reusable && !strict_prefix_reusable)
            || registration.block_ids.is_empty()
        {
            return RealPrefixCacheRegisterOutcome {
                retained_blocks: Vec::new(),
                evicted_blocks: Vec::new(),
            };
        }

        if self.entries.iter().any(|entry| {
            !entry.retired
                && entry.adapter == adapter
                && entry.prompt_tokens == registration.prompt_tokens
        }) {
            return RealPrefixCacheRegisterOutcome {
                retained_blocks: Vec::new(),
                evicted_blocks: Vec::new(),
            };
        }

        let registration_blocks: HashSet<u32> = registration.block_ids.iter().copied().collect();
        if registration_blocks.len() != registration.block_ids.len()
            || registration_blocks.len() > self.max_blocks
        {
            // A single entry must fit after every resident entry is gone, and
            // a sequence block table may not alias the same physical page at
            // multiple logical positions.
            return RealPrefixCacheRegisterOutcome {
                retained_blocks: Vec::new(),
                evicted_blocks: Vec::new(),
            };
        }

        // Plan the complete mutation against a private refcount copy. The old
        // loop evicted entries incrementally and could then discover a pinned
        // entry made the registration impossible, returning a partially
        // destroyed cache. It also kept `needed_new_blocks` fixed while
        // eviction removed blocks shared with the incoming entry, allowing the
        // committed cache to exceed `max_blocks`.
        let mut projected_refcounts = self.block_refcounts.clone();
        let mut eviction_candidates: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.active_uses == 0 && !entry.retired)
            .map(|(idx, _)| idx)
            .collect();
        eviction_candidates.sort_by_key(|&idx| {
            let entry = &self.entries[idx];
            (entry.last_used, entry.id)
        });
        let mut planned_evictions = Vec::new();
        let mut candidate_cursor = 0;
        loop {
            let projected_blocks = registration_blocks.len()
                + projected_refcounts
                    .keys()
                    .filter(|block_id| !registration_blocks.contains(block_id))
                    .count();
            let projected_entries = self.entries.len() - planned_evictions.len() + 1;
            if projected_blocks <= self.max_blocks && projected_entries <= self.max_entries {
                break;
            }

            let Some(&evict_idx) = eviction_candidates.get(candidate_cursor) else {
                return RealPrefixCacheRegisterOutcome {
                    retained_blocks: Vec::new(),
                    evicted_blocks: Vec::new(),
                };
            };
            candidate_cursor += 1;
            planned_evictions.push(evict_idx);
            for block_id in &self.entries[evict_idx].block_ids {
                let remove = projected_refcounts
                    .get_mut(block_id)
                    .is_some_and(|refcount| {
                        *refcount = refcount.saturating_sub(1);
                        *refcount == 0
                    });
                if remove {
                    projected_refcounts.remove(block_id);
                }
            }
        }

        let mut evicted_blocks = Vec::new();
        planned_evictions.sort_unstable_by(|a, b| b.cmp(a));
        for evict_idx in planned_evictions {
            let evicted = self.entries.remove(evict_idx);
            evicted_blocks.extend(self.release_entry_blocks(&evicted.block_ids));
        }

        let retained_blocks: Vec<u32> = registration
            .block_ids
            .iter()
            .copied()
            .filter(|block_id| !self.block_refcounts.contains_key(block_id))
            .collect();
        for &block_id in &registration.block_ids {
            *self.block_refcounts.entry(block_id).or_insert(0) += 1;
        }
        // #673: An entry evicted above may have shared block IDs with the
        // incoming registration. `release_entry_blocks` would have pushed those
        // IDs into `evicted_blocks`, but the refcount increments above have now
        // re-claimed them. Returning them in `evicted_blocks` would cause the
        // API layer to free live cached blocks back to the BlockManager, where
        // a concurrent request can re-allocate and overwrite them.
        evicted_blocks.retain(|block_id| !registration_blocks.contains(block_id));
        debug_assert!(
            evicted_blocks
                .iter()
                .all(|id| !self.block_refcounts.contains_key(id)),
            "RealPrefixCache::register: evicted_blocks must not contain any block currently in block_refcounts"
        );
        let id = self.next_entry_id;
        self.next_entry_id += 1;
        let last_used = self.stats.lookup_hits + self.stats.lookup_misses;
        self.entries.push(RealPrefixCacheEntry {
            id,
            adapter,
            prompt_tokens: registration.prompt_tokens,
            block_ids: registration.block_ids,
            linear_state: registration.linear_state,
            next_token: registration.next_token,
            last_used,
            active_uses: 0,
            retired: false,
        });
        debug_assert!(self.cached_blocks() <= self.max_blocks);
        debug_assert!(self.entries.len() <= self.max_entries);
        debug_assert_eq!(
            self.block_refcounts,
            self.entries
                .iter()
                .fold(HashMap::new(), |mut counts, entry| {
                    for block_id in &entry.block_ids {
                        *counts.entry(*block_id).or_insert(0) += 1;
                    }
                    counts
                })
        );
        RealPrefixCacheRegisterOutcome {
            retained_blocks,
            evicted_blocks,
        }
    }

    pub fn clear(&mut self) -> Vec<u32> {
        self.global_generation = self
            .global_generation
            .checked_add(1)
            .expect("prefix-cache global generation overflow");
        self.adapter_generations.clear();
        self.retire_matching_entries(|_| true)
    }

    /// Remove every entry cached for `adapter`, releasing its blocks.
    /// Returns the block ids whose refcount dropped to zero (the caller
    /// frees those in the BlockManager). Used when an adapter's directory
    /// content changes (retrain/import): name-keyed entries would
    /// otherwise replay KV computed under the OLD weights. Entries for
    /// other adapters are untouched — that's the point (a background eval
    /// swap must not destroy the serving agent's accumulated prefix
    /// cache).
    pub fn purge_adapter(&mut self, adapter: &Option<String>) -> Vec<u32> {
        let generation = self.adapter_generations.entry(adapter.clone()).or_default();
        *generation = generation
            .checked_add(1)
            .expect("prefix-cache adapter generation overflow");
        self.retire_matching_entries(|entry| {
            entry
                .adapter
                .as_ref()
                .map(|identity| identity.name.as_str())
                == adapter.as_deref()
        })
    }

    fn adapter_generation(&self, adapter: &Option<String>) -> u64 {
        self.adapter_generations.get(adapter).copied().unwrap_or(0)
    }

    fn retire_matching_entries(
        &mut self,
        mut matches: impl FnMut(&RealPrefixCacheEntry) -> bool,
    ) -> Vec<u32> {
        let mut released = Vec::new();
        let mut idx = 0;
        while idx < self.entries.len() {
            if !matches(&self.entries[idx]) {
                idx += 1;
                continue;
            }
            if self.entries[idx].active_uses > 0 {
                self.entries[idx].retired = true;
                idx += 1;
            } else {
                let entry = self.entries.remove(idx);
                released.extend(self.release_entry_blocks(&entry.block_ids));
            }
        }
        released
    }

    fn release_entry_blocks(&mut self, block_ids: &[u32]) -> Vec<u32> {
        let mut freed = Vec::new();
        for &block_id in block_ids {
            if let Some(refcount) = self.block_refcounts.get_mut(&block_id) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    self.block_refcounts.remove(&block_id);
                    freed.push(block_id);
                }
            }
        }
        freed
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            cached_blocks: self.cached_blocks(),
            max_blocks: self.max_blocks,
            cached_entries: self.entries.len(),
            max_entries: self.max_entries,
            cached_state_bytes: self
                .state_bytes_per_entry
                .saturating_mul(self.entries.len() as u64),
            max_state_bytes: self.max_state_bytes,
            active_leases: self.entries.iter().map(|entry| entry.active_uses).sum(),
            pending_release_entries: self.entries.iter().filter(|entry| entry.retired).count(),
            ..self.stats
        }
    }

    fn cached_blocks(&self) -> usize {
        self.block_refcounts.len()
    }
}

pub struct PromptTokenCache {
    capacity: usize,
    entries: HashMap<String, Vec<TokenId>>,
    lru: VecDeque<String>,
    hits: u64,
    misses: u64,
}

impl PromptTokenCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            lru: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, prompt_text: &str) -> Option<Vec<TokenId>> {
        if let Some(tokens) = self.entries.get(prompt_text).cloned() {
            self.hits = self.hits.saturating_add(1);
            self.lru.retain(|existing| existing != prompt_text);
            self.lru.push_back(prompt_text.to_string());
            return Some(tokens);
        }
        self.misses = self.misses.saturating_add(1);
        None
    }

    pub fn insert(&mut self, prompt_text: String, tokens: Vec<TokenId>) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.contains_key(&prompt_text) {
            self.lru.retain(|existing| existing != &prompt_text);
        }
        self.entries.insert(prompt_text.clone(), tokens);
        self.lru.push_back(prompt_text);
        while self.entries.len() > self.capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.entries.len())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

pub struct RenderedPromptCache {
    capacity: usize,
    entries: HashMap<String, String>,
    lru: VecDeque<String>,
    hits: u64,
    misses: u64,
}

impl RenderedPromptCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            lru: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<String> {
        if let Some(prompt) = self.entries.get(key).cloned() {
            self.hits = self.hits.saturating_add(1);
            self.lru.retain(|existing| existing != key);
            self.lru.push_back(key.to_string());
            return Some(prompt);
        }
        self.misses = self.misses.saturating_add(1);
        None
    }

    pub fn insert(&mut self, key: String, prompt: String) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.contains_key(&key) {
            self.lru.retain(|existing| existing != &key);
        }
        self.entries.insert(key.clone(), prompt);
        self.lru.push_back(key);
        while self.entries.len() > self.capacity {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    pub fn stats(&self) -> (u64, u64, usize) {
        (self.hits, self.misses, self.entries.len())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }
}

/// Which inference backend the server is using.
pub enum ModelBackend {
    /// Mock engine + scheduler for testing without real weights.
    Mock {
        scheduler: Arc<Mutex<Scheduler>>,
        engine: Arc<dyn Engine>,
    },
    /// Real model weights loaded via ModelRunner with paged KV cache.
    Real {
        runner: Arc<std::sync::RwLock<ModelRunner>>,
        rocm_graph_telemetry: kiln_model::RocmGraphTelemetryHandle,
        backend_health: BackendHealthHandle,
        block_manager: Arc<std::sync::Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCacheKt>,
        prefix_cache: Arc<std::sync::Mutex<RealPrefixCache>>,
        batching_engine: Option<crate::batching_engine::BatchingEngineHandle>,
        decode_batcher: Option<Arc<DecodeBatcher>>,
    },
}

/// Runtime availability of the fallback direct-streaming greedy rendezvous.
///
/// The worker is intentionally constructed independently of the batching
/// actor. It is routable only when the worker is live and the actor is absent;
/// publishing both facts prevents an idle compatibility worker from being
/// mistaken for the active production route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct DirectDecodeRendezvousRuntimeState {
    pub scope: &'static str,
    pub backend_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_unavailable_reason: Option<&'static str>,
    pub actor_active: bool,
    pub worker_active: bool,
    pub route_available: bool,
}

impl DirectDecodeRendezvousRuntimeState {
    pub const SCOPE: &'static str = "direct_streaming_greedy_only";

    const fn resolve(backend_available: bool, actor_active: bool, worker_active: bool) -> Self {
        Self {
            scope: Self::SCOPE,
            backend_available,
            backend_unavailable_reason: if backend_available {
                None
            } else {
                Some("mock_backend")
            },
            actor_active,
            worker_active,
            route_available: backend_available && !actor_active && worker_active,
        }
    }
}

/// Backend speculative capability facts captured once for diagnostics.
/// Serving remains fail-closed, so this snapshot is not a routing authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeRuntimePolicy {
    pub mtp_support: Support,
}

impl SpeculativeRuntimePolicy {
    pub const fn new(mtp_support: Support) -> Self {
        Self { mtp_support }
    }
}

impl Default for SpeculativeRuntimePolicy {
    fn default() -> Self {
        Self::new(Support::Unsupported)
    }
}

/// Shared application state passed to all handlers.
/// Durable status for the [agent] self_improve scheduler.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SelfImproveSchedulerStatus {
    pub interval_hours: u64,
    /// Unix ms of the last attempted run (success or failure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run_unix_ms: Option<u64>,
    /// "queued N jobs" or the error string from the last attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_result: Option<String>,
    /// Job ids the last successful round enqueued.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub last_job_ids: Vec<String>,
    /// Unix ms when the next round fires (the restart-surviving anchor).
    pub next_run_unix_ms: u64,
}

#[derive(Clone)]
pub struct AppState {
    /// Immutable process-lifetime serving policy and its startup provenance.
    pub serving_profile: crate::config::ServingProfileSetting,
    /// Immutable deterministic and concurrent-decode policy, including the
    /// configured, backend-selected, and final effective values.
    pub decode_runtime_config: crate::config::DecodeRuntimeConfig,
    /// Immutable batching-actor policy resolved against the selected backend
    /// and effective decode width during startup.
    pub batching_runtime_config: crate::config::BatchingRuntimeConfig,
    /// Immutable streaming-prefill dispatch and tiling policy resolved against
    /// the selected backend during startup.
    pub streaming_prefill_runtime_config: crate::config::StreamingPrefillRuntimeConfig,
    /// Requested cross-request prefix-cache policy. The live cache publishes
    /// the backend-qualified effective capability separately.
    pub prefix_cache_config: crate::config::PrefixCacheConfig,
    /// Versioned accelerator execution policy resolved before device context
    /// and model-runner construction.
    pub accelerator_runtime_policy: crate::config::ResolvedAcceleratorRuntimePolicy,
    /// Immutable speculative-decoding policy resolved once during startup.
    pub speculative_config: crate::config::SpeculativeDecodingConfig,
    /// Immutable backend capability snapshot used by diagnostics.
    pub speculative_runtime_policy: SpeculativeRuntimePolicy,
    /// Immutable operational policy consumed by request handlers without
    /// rereading process environment.
    pub operational_runtime: Arc<crate::config::OperationalRuntimeConfig>,
    /// Typed startup-only checkpoint read setting retained for resolved
    /// configuration diagnostics.
    pub checkpoint_read_mib_per_second: Option<u64>,
    /// Whether a real-model load applied the checkpoint read policy.
    pub checkpoint_read_applicable: bool,
    /// Stable public explanation when checkpoint read pacing is inapplicable.
    pub checkpoint_read_not_applicable_reason: Option<&'static str>,
    /// Exact snapshot/verification phase accounting for a real model.
    pub checkpoint_read_report: Option<kiln_model::CheckpointReadReport>,
    /// Typed startup-only accelerator-weight upload setting retained for
    /// resolved configuration diagnostics.
    pub accelerator_weight_upload_mib_per_second: Option<u64>,
    /// Whether the selected real-model device can apply accelerator upload
    /// pacing. False for mock mode and CPU-only real-model execution.
    pub accelerator_weight_upload_applicable: bool,
    /// Stable public explanation when accelerator upload pacing is inapplicable.
    pub accelerator_weight_upload_not_applicable_reason: Option<&'static str>,
    /// Exact completed upload accounting for a real model. Mock mode has no
    /// accelerator upload and leaves this absent.
    pub accelerator_weight_upload_report: Option<kiln_model::AcceleratorWeightUploadReport>,
    pub model_config: ModelConfig,
    /// Configured model directory path for real inference mode. `None` in mock mode.
    pub model_path: Option<PathBuf>,
    /// Immutable content identity for this process's base prompt-logprob
    /// teacher. Real production startup sets this exactly once after the
    /// loader-owned model source survives its post-upload verification. Mock
    /// and synthetic test states intentionally leave it absent.
    pub base_teacher_identity: Option<Arc<kiln_train::TeacherIdentityV1>>,
    /// Loader-verified identity of every safetensors shard behind the resident
    /// base model. Production startup sets this once; mock and synthetic test
    /// states may leave it absent.
    pub base_weight_shard_manifest: Option<Arc<BaseWeightShardManifest>>,
    /// Immutable process/backend/build/configuration identity. Production
    /// startup sets this once; mock and synthetic test states may omit it.
    pub execution_provenance: Option<Arc<kiln_core::execution_provenance::ExecutionProvenanceV1>>,
    pub backend: Arc<ModelBackend>,
    pub tokenizer: Arc<KilnTokenizer>,
    /// Directory where LoRA adapter weights are stored on disk.
    pub adapter_dir: PathBuf,
    /// Server default LoRA adapter selected by adapter load/unload endpoints.
    pub active_adapter_name: Arc<std::sync::RwLock<Option<String>>>,
    /// LoRA adapter currently loaded into the model runner for inference,
    /// including the exact content revision published with its weight flip.
    ///
    /// This can differ from `active_adapter_name` during explicit per-request
    /// chat adapter overrides; missing `adapter` requests reload the default.
    pub loaded_adapter: Arc<std::sync::RwLock<Option<LoadedAdapterIdentity>>>,
    /// [agent] self_improve scheduler status — None when the scheduler
    /// isn't armed. Persisted to `<adapter_dir>/.self_improve_scheduler.json`
    /// so the cadence survives restarts; surfaced via /health.
    pub self_improve_scheduler: Arc<std::sync::RwLock<Option<SelfImproveSchedulerStatus>>>,
    /// Embedded pi agent runs — the server-driven rollout engine
    /// (`/v1/agent/runs`). Records persist under `<adapter_dir>/agent_runs/`.
    pub agent_runs: Arc<crate::agent_runs::AgentRunRegistry>,
    /// Last adapter load failure by adapter name. Used by the registry so
    /// automation can distinguish "not loaded" from "failed to load".
    pub adapter_load_errors: Arc<std::sync::RwLock<HashMap<String, String>>>,
    /// Serializes every adapter filesystem publication and loaded-weight
    /// transition (see `adapter_swap`). The lock is the revision barrier that
    /// keeps mutable adapter directories, the server default, and the exact
    /// weights published by the runner from racing one another.
    pub adapter_mutation_lock: Arc<tokio::sync::Mutex<()>>,
    /// Serializes HF/TRL export publication, download snapshots, and deletion
    /// without holding the unrelated adapter mutation barrier for every
    /// archive transfer.
    pub hf_trl_export_lock: Arc<tokio::sync::Mutex<()>>,
    /// Tracked training jobs (job_id → info).
    pub training_jobs: TrainingJobs,
    /// GPU memory budget for coordinating inference and training.
    pub memory_budget: Arc<GpuMemoryBudget>,
    /// Whether the dynamic device KV control loop was requested and started.
    pub kv_autoscaler: crate::kv_autoscaler::KvAutoscalerState,
    /// Coordination lock: inference takes read lock, training takes write lock.
    /// This prevents simultaneous GPU-heavy operations from OOMing.
    pub gpu_lock: GpuCoordinationLock,
    /// FIFO training queue — jobs are enqueued here and executed sequentially
    /// by a background worker.
    pub training_queue: SharedTrainingQueue,
    /// Serializes bounded training-data validation and materialization for this
    /// server instance without coupling independent `AppState` instances.
    pub(crate) training_data_admission_lock: Arc<std::sync::Mutex<()>>,
    /// §3.2 / §4 teacher registry — alias → `TeacherSpec`. Persists
    /// to `adapter_dir/teachers.json` so registrations survive restart.
    /// Consulted by every OPD / distill_* handler that takes a
    /// `teacher: alias` request field.
    pub teacher_registry: crate::api::teachers::SharedTeacherRegistry,
    /// Immutable server-owned credential handles for remote teachers. API
    /// requests and persisted teacher specs can name only an id from this map;
    /// they never control or observe the backing environment-variable name.
    pub teacher_credentials: Arc<crate::config::TeachersConfig>,
    /// Detected VRAM info for config/debug reporting.
    pub vram_info: kiln_memory::vram::GpuVramInfo,
    /// Physical, requested, and cap-only effective memory capacity resolved at
    /// startup. This remains immutable for process-lifetime diagnostics.
    pub vram_capacity_resolution: kiln_memory::vram::VramCapacityResolution,
    /// Active accelerator selected for every live memory probe.
    pub vram_probe_selector: kiln_memory::vram::VramProbeSelector,
    /// Typed memory policy retained for diagnostics without re-reading process
    /// environment after startup.
    pub memory_config: crate::config::MemoryConfig,
    /// Immutable native-training planning inputs shared by every queued job.
    pub training_runtime: kiln_train::TrainingRuntimeContext,
    /// Immutable logical device selected for inference execution. Vulkan's
    /// hybrid path can keep weights on CPU while dispatching kernels through
    /// Vulkan, so this identity must remain distinct from weight placement.
    pub inference_device: kiln_tensor::Device,
    /// Immutable device identity of the frozen model-weight representation.
    /// This differs from the execution device on Vulkan's hybrid serving path
    /// and lets admission/reporting reject unqualified residency without
    /// locking the model runner.
    pub model_weight_device: kiln_tensor::Device,
    /// Shutdown flag — set to true when the server is shutting down.
    pub shutdown: ShutdownFlag,
    /// Per-request timeout duration. Configurable via KILN_REQUEST_TIMEOUT_SECS (default 600).
    pub request_timeout: std::time::Duration,
    /// Requested `SO_SNDBUF` for every accepted HTTP connection. `None` leaves
    /// the platform default untouched.
    pub http_send_buffer_bytes: Option<usize>,
    /// Raw listener `getsockopt(SO_SNDBUF)` result captured before readiness.
    pub http_send_buffer_preflight_actual_bytes: Option<usize>,
    /// Listener send-buffer bytes after normalizing platform accounting.
    pub http_send_buffer_preflight_effective_bytes: Option<usize>,
    /// Eval-serving mode: deterministic defaults, no-think defaults, headers,
    /// adapter-switch warnings, and per-request transient cache cleanup.
    pub eval_mode: bool,
    /// Typed access policy for the trusted model-state diagnostics endpoint.
    pub debug_model_state: bool,
    /// Server-level default for chat-template thinking mode. `None` preserves
    /// the template's own default.
    pub default_thinking_enabled: Option<bool>,
    /// Server defaults for forced reasoning closure. Requests may inherit,
    /// override, or explicitly disable each dimension independently.
    pub default_thinking_budget_tokens: Option<usize>,
    pub default_thinking_budget_ms: Option<u64>,
    /// Active model-specific runtime defaults profile.
    pub model_defaults_profile: crate::config::ModelDefaultsProfile,
    /// Compatibility mode: duplicate separated reasoning into `content`.
    pub fold_reasoning_into_content: bool,
    /// Include per-request performance counters in chat response metadata when
    /// a request does not explicitly opt in or out.
    pub chat_performance_metadata: bool,
    /// Include config hashes in chat response metadata when a request does not
    /// explicitly opt in or out.
    pub chat_config_hash_metadata: bool,
    /// Stable hashes of the model, tokenizer/template, and effective Kiln
    /// runtime config.
    pub config_hashes: ConfigHashes,
    /// Slow chat-completion warning threshold. None disables slow-request logs.
    pub slow_request_warn_threshold: Option<std::time::Duration>,
    /// Prometheus metrics counters.
    pub metrics: Arc<Metrics>,
    /// Server startup time — used to compute uptime in health checks.
    pub started_at: std::time::Instant,
    /// True once startup inference prewarm has finished or was not needed.
    pub inference_prewarm_complete: Arc<AtomicBool>,
    /// Server-level default checkpoint interval during training. SFT and GRPO
    /// write exact resumable checkpoints; legacy modes may still emit
    /// PEFT-only snapshots. Per-job config overrides this. `None` disables
    /// periodic checkpoints.
    pub checkpoint_interval: Option<usize>,
    /// Optional URL to POST a JSON notification to whenever a training
    /// job completes or fails. `None` disables webhook firing entirely.
    /// See `TrainingConfig::webhook_url` for the payload contract.
    pub training_webhook_url: Option<String>,
    /// Maximum number of training jobs that may sit in `training_queue`
    /// at once. Submissions while the queue is at this cap are rejected
    /// with HTTP 503 + `Retry-After: 30`. Mirrors
    /// `TrainingConfig::max_queued_jobs` (default 32).
    pub max_queued_training_jobs: usize,
    /// Maximum number of tracked training jobs that may live in
    /// `training_jobs` (the in-memory tracking map) at once. Submissions
    /// while the map is at this cap are rejected with HTTP 503 +
    /// `Retry-After: 30` and the `training_tracked_full` error code.
    /// Mirrors `TrainingConfig::max_tracked_jobs` (default 1024).
    pub max_tracked_jobs: usize,
    /// TTL for terminal (`Completed` / `Failed`) entries in
    /// `training_jobs`. The training worker periodically removes terminal
    /// entries whose `finished_at` timestamp is older than this duration.
    /// Active entries (`Queued` / `Running`) are never GC'd.
    /// Mirrors `TrainingConfig::tracked_job_ttl_secs` (default 3600s).
    pub tracked_job_ttl: std::time::Duration,
    /// Maximum total bytes that finalized adapters may occupy in
    /// `adapter_dir/`. Uploads to `POST /v1/adapters/upload` that would
    /// push the total over this cap are rejected before the rename-into-
    /// place step. `None` disables the cap entirely (operator opt-out).
    /// `.upload-tmp-*/` staging dirs and the `.composed/<hash>/` cache
    /// are excluded from the count — they are bounded separately. Mirrors
    /// `AdaptersConfig::max_disk_bytes` (default 100 GiB).
    pub adapter_max_disk_bytes: Option<u64>,
    /// Byte cap for the on-disk composed-adapter cache at
    /// `adapter_dir/.composed/<hash>/`. Enforced via LRU eviction
    /// (oldest mtime first) after a successful synthesize. `None`
    /// disables the byte cap. Mirrors
    /// `AdaptersConfig::composed_cache_max_bytes` (default 10 GiB).
    pub composed_cache_max_bytes: Option<u64>,
    /// Entry-count cap for the on-disk composed-adapter cache at
    /// `adapter_dir/.composed/`. Enforced via LRU eviction (oldest
    /// mtime first). `None` disables the entry cap. Mirrors
    /// `AdaptersConfig::composed_cache_max_entries` (default 64).
    pub composed_cache_max_entries: Option<u64>,
    /// Identifier exposed at `/v1/models` and echoed in chat completion responses.
    pub served_model_id: String,
    /// Rolling timestamp ring for live decode tok/s + ITL on the /ui dashboard.
    pub decode_stats: Arc<std::sync::Mutex<DecodeStatsRing>>,
    /// Bounded history of recent chat-completion requests for the /ui dashboard.
    pub recent_requests: Arc<std::sync::Mutex<RecentRequestsRing>>,
    /// Durable request/response JSONL log for the inference endpoints.
    /// `None` when `[request_log] enabled = false` (or in tests that don't
    /// set one up). See [`crate::request_log`].
    pub request_log: Option<Arc<crate::request_log::RequestLogger>>,
    /// Per-body cap on what the request log stores (never affects the wire).
    pub request_log_max_capture_bytes: usize,
    deterministic_cache_generations: Arc<std::sync::Mutex<DeterministicCacheGenerations>>,
    /// Full-response cache for replayable completions.
    pub completion_cache: Arc<std::sync::Mutex<DeterministicCompletionCache>>,
    /// Full-response cache keyed before chat-template rendering/tokenization.
    pub chat_request_cache: Arc<std::sync::Mutex<DeterministicChatRequestCache>>,
    /// Full-response cache for replayable non-streaming chat n>1 requests.
    pub chat_choices_cache: Arc<std::sync::Mutex<DeterministicChatChoicesCache>>,
    /// Full-response cache for replayable multi-output batch requests.
    pub batch_cache: Arc<std::sync::Mutex<DeterministicBatchCache>>,
    /// Rendered chat-template prompt cache used before tokenization.
    pub rendered_prompt_cache: Arc<std::sync::Mutex<RenderedPromptCache>>,
    /// Rendered-prompt token cache used before completion/prefix cache lookup.
    pub prompt_token_cache: Arc<std::sync::Mutex<PromptTokenCache>>,
    /// FIFO queue of pending eval jobs. Drained by the background eval
    /// worker (see `crate::eval::worker::spawn_eval_worker`).
    pub eval_queue: crate::eval::SharedEvalQueue,
    /// Tracked eval-job state (job_id → info). Mirrors `training_jobs`.
    pub eval_jobs: crate::eval::EvalJobs,
    /// On-disk registry of named eval suites. Set only when the server was
    /// configured with an `eval_dir`; mock-mode and tests can leave it None
    /// and use inline suites.
    pub suite_registry: Option<Arc<crate::eval::SuiteRegistry>>,
    /// On-disk dataset registry: SFT/GRPO JSONL files users upload to
    /// power synthesis. Co-located with the suite registry.
    pub dataset_registry: Option<Arc<crate::eval::DatasetRegistry>>,
    /// On-disk judgment store: append-only A/B preference rows. The
    /// flywheel that turns user picks into local judge LoRAs.
    pub judgment_store: Option<Arc<crate::eval::JudgmentStore>>,
    /// Maximum eval jobs allowed in `eval_queue` at once. Mirrors
    /// `max_queued_training_jobs`; over-cap submissions are rejected with
    /// 503 + Retry-After.
    pub max_queued_eval_jobs: usize,
    /// Maximum tracked eval entries in `eval_jobs`. Mirrors
    /// `max_tracked_jobs`.
    pub max_tracked_eval_jobs: usize,
    /// Fire-and-forget webhook for terminal eval jobs (`eval.webhook_url`
    /// in the TOML). Mirrors `training_webhook_url`.
    pub eval_webhook_url: Option<String>,
}

/// Map the active tensor backend to the one accelerator whose memory counters
/// are authoritative for this server process.
pub fn vram_probe_selector_for_device(
    device: kiln_tensor::Device,
) -> kiln_memory::vram::VramProbeSelector {
    device.memory_probe_selector()
}

/// Establish the conservative physical-device identity contract shared by the
/// selected tensor backend and its OS/driver memory probe.
///
/// The production startup path must call this immediately after backend device
/// selection and before loading/uploading model weights. Real-state
/// construction repeats the validation as defense in depth before any memory
/// detection or KV-cache allocation owned by `AppState`.
pub fn ensure_accelerator_memory_probe_identity(
    device: kiln_tensor::Device,
) -> anyhow::Result<kiln_memory::vram::VramProbeSelector> {
    let selector = vram_probe_selector_for_device(device);
    kiln_memory::vram::validate_vram_probe_identity(selector).map_err(|error| {
        anyhow::anyhow!(
            "cannot start selected backend device {}: {error}",
            device.short_name()
        )
    })?;
    Ok(selector)
}

/// Refuse accelerator startup unless the selected, device-scoped probe
/// established a non-zero safe capacity. A configured memory value is a cap,
/// not permission to invent capacity when hardware detection failed.
pub fn ensure_accelerator_memory_capacity(
    device: kiln_tensor::Device,
    selector: kiln_memory::vram::VramProbeSelector,
    capacity: kiln_memory::vram::GpuVramInfo,
) -> anyhow::Result<()> {
    if device.is_cpu() || capacity.total_bytes > 0 {
        return Ok(());
    }

    anyhow::bail!(
        "cannot start on selected accelerator {}: device-scoped memory probe {:?} established 0 bytes of safe effective capacity (source: {}). Refusing to load model weights or allocate the paged KV cache because that could trigger a process- or host-fatal allocator failure. Verify that the selected device and its driver memory counters are visible, or select CPU/a different accelerator. memory.gpu_memory_gb is cap-only and cannot replace a failed hardware probe.",
        device.short_name(),
        selector,
        capacity.source,
    )
}

/// Refuse a governor floor that consumes the selected accelerator's entire
/// effective capacity. This runs before model upload so an impossible policy
/// cannot reach allocator setup and fail later under load.
pub fn ensure_accelerator_memory_floor(
    device: kiln_tensor::Device,
    capacity: kiln_memory::vram::GpuVramInfo,
    memory: &crate::config::MemoryConfig,
) -> anyhow::Result<()> {
    if device.is_cpu() {
        return Ok(());
    }
    let floor_bytes = memory.floor_bytes();
    if floor_bytes < capacity.total_bytes {
        return Ok(());
    }

    anyhow::bail!(
        "cannot start on selected accelerator {}: memory.floor_gb={} GiB resolves to {} bytes, but effective accelerator capacity is {} bytes ({:.6} GiB, source: {}). memory.floor_gb must be strictly smaller than effective accelerator capacity before model upload",
        device.short_name(),
        memory.floor_gb,
        floor_bytes,
        capacity.total_bytes,
        capacity.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        capacity.source,
    )
}

fn latch_rocm_runtime_health(
    backend_health: &BackendHealthHandle,
    synchronization: &crate::accelerator_runtime::RocmSynchronizationRuntimeStats,
) {
    if backend_health.snapshot().quarantined {
        return;
    }
    if let Some(reason) = synchronization.fail_closed_reason() {
        backend_health.quarantine(reason);
    }
}

impl AppState {
    /// Bind the exact accelerator policy already used to construct the device
    /// context and model runner. This builder avoids adding another argument to
    /// the compatibility constructors while still rejecting profile drift.
    pub fn with_accelerator_runtime_policy(
        mut self,
        policy: crate::config::ResolvedAcceleratorRuntimePolicy,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            policy.serving_profile == self.serving_profile.profile(),
            "accelerator runtime policy serving profile {} does not match application state profile {}",
            policy.serving_profile,
            self.serving_profile.profile(),
        );
        anyhow::ensure!(
            policy.serving_profile_source == self.serving_profile.source(),
            "accelerator runtime policy profile source {} does not match application state source {}",
            policy.serving_profile_source,
            self.serving_profile.source(),
        );
        self.accelerator_runtime_policy = policy;
        Ok(self)
    }

    /// Read ROCm synchronization atomics without synchronizing or probing the
    /// device. Other backends return an inactive snapshot.
    pub fn rocm_synchronization_runtime_stats(
        &self,
    ) -> crate::accelerator_runtime::RocmSynchronizationRuntimeStats {
        crate::accelerator_runtime::rocm_synchronization_runtime_stats(self.model_weight_device)
    }

    /// Observe live ROCm runtime health and make any unsafe state sticky in the
    /// process-lifetime backend-health latch. Inactive non-ROCm backends are
    /// unchanged.
    pub(crate) fn observe_rocm_runtime_health(
        &self,
    ) -> crate::accelerator_runtime::RocmSynchronizationRuntimeStats {
        let synchronization = self.rocm_synchronization_runtime_stats();
        if let Some(backend_health) = self.backend_health_handle() {
            latch_rocm_runtime_health(&backend_health, &synchronization);
        }
        synchronization
    }

    /// Return why this process cannot execute `workload` on its immutable
    /// native-training substrate, or `None` when static admission may proceed.
    ///
    /// This deliberately excludes request shape and live-memory admission.
    /// Every fact inspected here is fixed at startup except runner-lock health,
    /// which fails closed rather than recovering a potentially inconsistent
    /// model/backend view.
    pub(crate) fn training_workload_unavailable_reason(
        &self,
        workload: TrainingWorkload,
    ) -> Option<String> {
        if workload == TrainingWorkload::DistillRefresh {
            return Some(DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE.to_string());
        }

        let ModelBackend::Real { runner, .. } = self.backend.as_ref() else {
            return Some(format!(
                "mock backend does not execute {} training",
                workload.label()
            ));
        };

        if !self.serving_profile.runtime_policy().training_gpu_ownership {
            return Some(format!(
                "serving profile `{}` prohibits training GPU ownership",
                self.serving_profile.profile()
            ));
        }

        let runner = match runner.read() {
            Ok(runner) => runner,
            Err(_) => {
                return Some(format!(
                    "model runner lock poisoned while resolving {} workload support",
                    workload.label()
                ));
            }
        };
        let resident_weight_device = runner.weights.device_kt();
        if resident_weight_device != self.model_weight_device {
            return Some(format!(
                "configured model weight device {} does not match runner weight device {}",
                self.model_weight_device.short_name(),
                resident_weight_device.short_name(),
            ));
        }
        let runtime_device = match self
            .training_runtime
            .resolve_device_for_weights(resident_weight_device)
        {
            Ok(device) => device,
            Err(error) => {
                return Some(format!(
                    "{} training runtime cannot execute resident weights: {error:#}",
                    workload.label()
                ));
            }
        };

        let backend_capabilities = runner.backend_capabilities();
        if !kiln_model::backend::native_backend_identity_matches(
            runtime_device,
            backend_capabilities.backend,
            backend_capabilities.device,
        ) {
            return Some(format!(
                "{} training requires an exact native backend for {}, but runner reports `{}` on {}",
                workload.label(),
                runtime_device.short_name(),
                backend_capabilities.backend,
                backend_capabilities.device.short_name(),
            ));
        }

        if runner.weights.has_any_marlin_packed_projection() {
            return Some(format!(
                "{} training is unavailable while any model projection is Marlin-packed",
                workload.label()
            ));
        }

        let checkpoint = kiln_train::CheckpointConfig::from_runtime(
            self.model_config.num_layers,
            &self.training_runtime,
        );
        training_workload_route_unavailable_reason(
            workload,
            backend_capabilities.training.hooks,
            checkpoint,
        )
    }

    /// Report actual direct-rendezvous process state, distinct from configured
    /// intent stored in [`Self::batching_runtime_config`].
    pub fn direct_decode_rendezvous_runtime_state(&self) -> DirectDecodeRendezvousRuntimeState {
        match self.backend.as_ref() {
            ModelBackend::Mock { .. } => {
                DirectDecodeRendezvousRuntimeState::resolve(false, false, false)
            }
            ModelBackend::Real {
                batching_engine,
                decode_batcher,
                ..
            } => DirectDecodeRendezvousRuntimeState::resolve(
                true,
                batching_engine.is_some(),
                decode_batcher.is_some(),
            ),
        }
    }

    pub fn loaded_adapter_identity(&self) -> Option<LoadedAdapterIdentity> {
        self.loaded_adapter.read().unwrap().clone()
    }

    pub fn loaded_adapter_name(&self) -> Option<String> {
        self.loaded_adapter_identity().map(|identity| identity.name)
    }

    pub(crate) fn deterministic_cache_key(
        &self,
        adapter: Option<LoadedAdapterIdentity>,
        request: String,
    ) -> DeterministicCacheKey {
        let (global_generation, adapter_generation) = self
            .deterministic_cache_generations
            .lock()
            .unwrap()
            .snapshot(&adapter);
        DeterministicCacheKey {
            adapter,
            global_generation,
            adapter_generation,
            request,
        }
    }

    pub(crate) fn deterministic_cache_fence(
        &self,
        adapter: &Option<LoadedAdapterIdentity>,
    ) -> (u64, u64) {
        self.deterministic_cache_generations
            .lock()
            .unwrap()
            .snapshot(adapter)
    }

    /// Return the process-lifetime health latch for a real backend. Mock mode
    /// has no accelerator state to quarantine.
    pub(crate) fn backend_health_handle(&self) -> Option<BackendHealthHandle> {
        match self.backend.as_ref() {
            ModelBackend::Real { backend_health, .. } => Some(backend_health.clone()),
            ModelBackend::Mock { .. } => None,
        }
    }

    /// Central admission gate shared by every server surface that can use or
    /// mutate real backend state.
    pub(crate) fn ensure_backend_healthy(&self) -> anyhow::Result<()> {
        if let Some(backend_health) = self.backend_health_handle() {
            backend_health.ensure_healthy()?;
            self.observe_rocm_runtime_health();
            backend_health.ensure_healthy()?;
        }
        Ok(())
    }

    /// Process-lifetime inference admission. Maintenance mode is entered only
    /// by restart, so rejecting every new owner makes that restart the explicit
    /// drain boundary for exclusive GPU work.
    pub fn ensure_inference_admission_allowed(&self) -> anyhow::Result<()> {
        if !self.serving_profile.runtime_policy().inference_admission {
            anyhow::bail!(
                "serving profile `{}` disables inference admission",
                self.serving_profile.profile()
            );
        }
        Ok(())
    }

    /// Refuse any real-backend training path that could take exclusive GPU
    /// ownership when the process was started for stable serving.
    pub(crate) fn ensure_training_gpu_ownership_allowed(&self) -> anyhow::Result<()> {
        if matches!(self.backend.as_ref(), ModelBackend::Real { .. })
            && !self.serving_profile.runtime_policy().training_gpu_ownership
        {
            anyhow::bail!(
                "serving profile `{}` prohibits training GPU ownership",
                self.serving_profile.profile()
            );
        }
        Ok(())
    }

    /// Refuse a live LoRA weight flip before loading weights or entering the
    /// batching actor's quiescence barrier. Mock mode has no GPU weights.
    pub(crate) fn ensure_adapter_weight_transition_allowed(&self) -> anyhow::Result<()> {
        if matches!(self.backend.as_ref(), ModelBackend::Real { .. })
            && !self
                .serving_profile
                .runtime_policy()
                .adapter_weight_transitions
        {
            anyhow::bail!(
                "serving profile `{}` prohibits live adapter weight transitions",
                self.serving_profile.profile()
            );
        }
        Ok(())
    }

    /// Register a new eval job: insert the `EvalJobInfo::queued` record
    /// into `eval_jobs` and push the corresponding `EvalQueueEntry` onto
    /// the worker queue. Returns the generated `job_id`. The two-write
    /// pattern was previously open-coded at four submission sites; keeping
    /// it here makes the cap checks easy to enforce and prevents the
    /// tracking map and the queue from drifting out of sync.
    pub fn enqueue_eval(
        &self,
        suite_name: String,
        adapters: Vec<Option<String>>,
        kind: crate::eval::queue::EvalSubmissionKind,
        source_training_job_id: Option<String>,
        job: crate::eval::queue::QueuedEvalJob,
    ) -> anyhow::Result<crate::eval::queue::EvalEnqueueReceipt> {
        self.enqueue_eval_inner(
            suite_name,
            adapters,
            kind,
            source_training_job_id,
            job,
            None,
        )
    }

    /// Enqueue another member of a paired/grouped eval with an already
    /// materialized seed. This is intentionally separate from ordinary
    /// admission so independently submitted jobs cannot accidentally reuse a
    /// seed, while post-training baseline/candidate jobs can stay paired.
    pub fn enqueue_eval_with_effective_seed(
        &self,
        suite_name: String,
        adapters: Vec<Option<String>>,
        kind: crate::eval::queue::EvalSubmissionKind,
        source_training_job_id: Option<String>,
        job: crate::eval::queue::QueuedEvalJob,
        effective_seed: u64,
    ) -> anyhow::Result<crate::eval::queue::EvalEnqueueReceipt> {
        self.enqueue_eval_inner(
            suite_name,
            adapters,
            kind,
            source_training_job_id,
            job,
            Some(effective_seed),
        )
    }

    fn enqueue_eval_inner(
        &self,
        suite_name: String,
        adapters: Vec<Option<String>>,
        kind: crate::eval::queue::EvalSubmissionKind,
        source_training_job_id: Option<String>,
        job: crate::eval::queue::QueuedEvalJob,
        forced_effective_seed: Option<u64>,
    ) -> anyhow::Result<crate::eval::queue::EvalEnqueueReceipt> {
        self.ensure_inference_admission_allowed()?;
        let real_backend = matches!(self.backend.as_ref(), ModelBackend::Real { .. });
        if real_backend && self.base_weight_shard_manifest.is_none() {
            anyhow::bail!("eval admission requires the resident base-weight shard manifest");
        }
        if real_backend && self.execution_provenance.is_none() {
            anyhow::bail!("eval admission requires startup-owned execution provenance");
        }
        if let Some(manifest) = self.base_weight_shard_manifest.as_deref() {
            manifest
                .validate()
                .map_err(|error| anyhow::anyhow!("invalid eval base-weight provenance: {error}"))?;
        }
        if let Some(provenance) = self.execution_provenance.as_deref() {
            provenance
                .validate()
                .map_err(|error| anyhow::anyhow!("invalid eval execution provenance: {error}"))?;
        }
        let job_id = uuid::Uuid::new_v4().to_string();
        let registered_suite_seed = |name: &str| {
            self.suite_registry
                .as_ref()
                .and_then(|registry| registry.load(name).ok())
                .and_then(|suite| suite.generation.seed)
        };
        let requested_seed = match &job {
            crate::eval::queue::QueuedEvalJob::Registered {
                suite_name,
                generation_override,
                ..
            } => generation_override
                .as_ref()
                .and_then(|params| params.seed)
                .or_else(|| registered_suite_seed(suite_name)),
            crate::eval::queue::QueuedEvalJob::Inline {
                suite,
                generation_override,
                ..
            } => generation_override
                .as_ref()
                .and_then(|params| params.seed)
                .or(suite.generation.seed),
            crate::eval::queue::QueuedEvalJob::Compare(spec) => spec.seed.or_else(|| {
                spec.generation
                    .as_ref()
                    .and_then(|params| params.seed)
                    .or_else(|| registered_suite_seed(&spec.suite))
            }),
        };
        let effective_seed = forced_effective_seed
            .or(requested_seed)
            .unwrap_or_else(rand::random);
        let mut info = crate::eval::queue::EvalJobInfo::queued(
            job_id.clone(),
            suite_name,
            adapters,
            kind,
            source_training_job_id,
            effective_seed,
        );
        info.base_weight_shard_manifest = self.base_weight_shard_manifest.as_deref().cloned();
        info.execution_provenance = self.execution_provenance.as_deref().cloned();
        self.eval_jobs.write().unwrap().insert(job_id.clone(), info);
        self.eval_queue
            .lock()
            .unwrap()
            .push(crate::eval::queue::EvalQueueEntry {
                job_id: job_id.clone(),
                effective_seed,
                job,
            });
        Ok(crate::eval::queue::EvalEnqueueReceipt {
            job_id,
            effective_seed,
        })
    }

    /// Logically invalidate every real prefix entry immediately. Unpinned
    /// entries are reclaimed before return; active entries remain physically
    /// retained and undiscoverable until their move-only request owners exit.
    pub fn clear_real_prefix_cache(&self) {
        let ModelBackend::Real {
            block_manager,
            prefix_cache,
            ..
        } = self.backend.as_ref()
        else {
            return;
        };
        let blocks = {
            let mut cache = prefix_cache.lock().unwrap();
            cache.clear()
        };
        if !blocks.is_empty() {
            let mut bm = block_manager.lock().unwrap();
            bm.free_all(&blocks);
        }
    }

    /// Invalidate everything cached under `adapter` — prefix KV entries plus
    /// the deterministic completion caches. Prefix entries with active request
    /// leases become undiscoverable tombstones and release their blocks only
    /// after the final owner exits; this method never waits for decode.
    /// Call this whenever the adapter's on-disk content changes (retrain
    /// auto-load, upload/import, delete): the name now refers to different
    /// weights, so name-keyed cache entries would replay the old model.
    /// Entries for OTHER adapters are deliberately left intact — prefix
    /// lookups are adapter-filtered, so they're still correct, and
    /// clearing them is what used to force minutes of re-prefill on the
    /// serving agent every time a background eval or training job swapped.
    ///
    /// Every deterministic key carries the exact loaded identity plus an
    /// adapter-name generation. Selective purge revokes matching in-flight
    /// claims and advances that generation before removing completed values,
    /// so a late unowned insertion remains undiscoverable too.
    pub fn purge_adapter_caches(&self, adapter: &Option<String>) {
        self.deterministic_cache_generations
            .lock()
            .unwrap()
            .purge_adapter(adapter);
        self.completion_cache.lock().unwrap().purge_adapter(adapter);
        self.chat_request_cache
            .lock()
            .unwrap()
            .purge_adapter(adapter);
        self.chat_choices_cache
            .lock()
            .unwrap()
            .purge_adapter(adapter);
        self.batch_cache.lock().unwrap().purge_adapter(adapter);
        if let ModelBackend::Real {
            block_manager,
            prefix_cache,
            ..
        } = self.backend.as_ref()
        {
            let blocks = prefix_cache.lock().unwrap().purge_adapter(adapter);
            if !blocks.is_empty() {
                block_manager.lock().unwrap().free_all(&blocks);
            }
        }
    }

    pub fn clear_eval_mode_transient_state(&self) {
        self.deterministic_cache_generations.lock().unwrap().clear();
        self.completion_cache.lock().unwrap().clear();
        self.chat_request_cache.lock().unwrap().clear();
        self.chat_choices_cache.lock().unwrap().clear();
        self.batch_cache.lock().unwrap().clear();
        self.rendered_prompt_cache.lock().unwrap().clear();
        self.prompt_token_cache.lock().unwrap().clear();
        self.clear_real_prefix_cache();
    }

    pub fn new_mock(
        model_config: ModelConfig,
        scheduler: Scheduler,
        engine: Arc<dyn Engine>,
        tokenizer: KilnTokenizer,
        request_timeout_secs: u64,
        served_model_id: String,
    ) -> Self {
        let config_hashes = ConfigHashes::from_model_tokenizer(&model_config, &tokenizer, None);
        let decode_runtime_config = crate::batching_engine::resolve_decode_runtime_config(
            crate::config::DeterministicInference::default(),
            crate::config::MaxDecodeBatch::default(),
            None,
            crate::config::BatchTokenBudget::default(),
        );
        let batching_runtime_config = crate::config::BatchingConfig::default().resolve(
            crate::config::BatchingBackendPolicy {
                batching_engine_default_enabled: false,
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                actor_prefill_tile_alignment_required: false,
                direct_decode_rendezvous: crate::config::DirectDecodeRendezvousBackendPolicy {
                    enabled: false,
                    max_batch: 1,
                    wait_us: 0,
                    mixed_seq_lens: false,
                },
            },
            decode_runtime_config.max_decode_batch.effective,
        );
        let streaming_prefill_runtime_config = crate::config::StreamingPrefillConfig::default()
            .resolve(kiln_model::StreamingPrefillBackendPolicy::for_backend(
                "cpu",
                kiln_tensor::Device::Cpu,
            ));
        let speculative_config = crate::config::SpeculativeDecodingConfig::default();
        let memory_config = crate::config::MemoryConfig::default();
        let kv_autoscaler_config = crate::kv_autoscaler::KvAutoscalerConfig {
            requested: memory_config.kv_autoscale.enabled(),
            requested_source: memory_config.kv_autoscale.source(),
            force_blocks: memory_config.kv_force_blocks.target(),
            force_blocks_source: memory_config.kv_force_blocks.source(),
        };
        Self {
            serving_profile: crate::config::ServingProfileSetting::default(),
            decode_runtime_config,
            batching_runtime_config,
            streaming_prefill_runtime_config,
            prefix_cache_config: crate::config::PrefixCacheConfig::default(),
            accelerator_runtime_policy: crate::config::AcceleratorRuntimeConfig::default()
                .resolved_policy(crate::config::ServingProfileSetting::default()),
            speculative_config,
            speculative_runtime_policy: SpeculativeRuntimePolicy::default(),
            operational_runtime: Arc::new(crate::config::OperationalRuntimeConfig::default()),
            checkpoint_read_mib_per_second: None,
            checkpoint_read_applicable: false,
            checkpoint_read_not_applicable_reason: Some("mock_mode"),
            checkpoint_read_report: None,
            accelerator_weight_upload_mib_per_second: None,
            accelerator_weight_upload_applicable: false,
            accelerator_weight_upload_not_applicable_reason: Some("mock_mode"),
            accelerator_weight_upload_report: None,
            model_config,
            model_path: None,
            base_teacher_identity: None,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            backend: Arc::new(ModelBackend::Mock {
                scheduler: Arc::new(Mutex::new(scheduler)),
                engine,
            }),
            tokenizer: Arc::new(tokenizer),
            adapter_dir: PathBuf::from("adapters"),
            active_adapter_name: Arc::new(std::sync::RwLock::new(None)),
            loaded_adapter: Arc::new(std::sync::RwLock::new(None)),
            self_improve_scheduler: Arc::new(std::sync::RwLock::new(None)),
            agent_runs: Arc::new(crate::agent_runs::AgentRunRegistry::new(PathBuf::from(
                "adapters",
            ))),
            adapter_load_errors: Arc::new(std::sync::RwLock::new(HashMap::new())),
            adapter_mutation_lock: Arc::new(tokio::sync::Mutex::new(())),
            hf_trl_export_lock: Arc::new(tokio::sync::Mutex::new(())),
            training_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            memory_budget: Arc::new(GpuMemoryBudget::compute(0, 0, 0, 0, 0, 1.0, None)),
            kv_autoscaler: crate::kv_autoscaler::KvAutoscalerState::unavailable(
                kv_autoscaler_config,
                "mock_backend",
            ),
            gpu_lock: Arc::new(RwLock::new(())),
            training_queue: crate::training_queue::new_shared_queue(),
            training_data_admission_lock: Arc::new(std::sync::Mutex::new(())),
            teacher_registry: Arc::new(crate::api::teachers::TeacherRegistry::new()),
            teacher_credentials: Arc::new(crate::config::TeachersConfig::default()),
            vram_info: kiln_memory::vram::GpuVramInfo {
                total_bytes: 0,
                source: kiln_memory::vram::VramSource::None,
                unified: false,
            },
            vram_capacity_resolution: kiln_memory::vram::VramCapacityResolution {
                physical: kiln_memory::vram::GpuVramInfo {
                    total_bytes: 0,
                    source: kiln_memory::vram::VramSource::None,
                    unified: false,
                },
                requested_bytes: None,
                effective: kiln_memory::vram::GpuVramInfo {
                    total_bytes: 0,
                    source: kiln_memory::vram::VramSource::None,
                    unified: false,
                },
                clamped: false,
            },
            vram_probe_selector: kiln_memory::vram::VramProbeSelector::None,
            memory_config,
            training_runtime: kiln_train::TrainingRuntimeContext::new_for_device(
                kiln_tensor::Device::Cpu,
                kiln_memory::vram::GpuVramInfo {
                    total_bytes: 0,
                    source: kiln_memory::vram::VramSource::None,
                    unified: false,
                },
                kiln_train::GradientCheckpointPolicy::Auto,
            )
            .with_streaming_prefill_policy(streaming_prefill_runtime_config.execution_policy()),
            inference_device: kiln_tensor::Device::Cpu,
            model_weight_device: kiln_tensor::Device::Cpu,
            shutdown: crate::training_queue::new_shutdown_flag(),
            request_timeout: std::time::Duration::from_secs(request_timeout_secs),
            http_send_buffer_bytes: None,
            http_send_buffer_preflight_actual_bytes: None,
            http_send_buffer_preflight_effective_bytes: None,
            eval_mode: false,
            debug_model_state: false,
            default_thinking_enabled: None,
            default_thinking_budget_tokens: None,
            default_thinking_budget_ms: None,
            model_defaults_profile: crate::config::ModelDefaultsProfile::qwen3_5_4b(),
            fold_reasoning_into_content: false,
            chat_performance_metadata: false,
            chat_config_hash_metadata: false,
            config_hashes,
            slow_request_warn_threshold: None,
            metrics: Arc::new(Metrics::new()),
            started_at: std::time::Instant::now(),
            inference_prewarm_complete: Arc::new(AtomicBool::new(true)),
            checkpoint_interval: None,
            training_webhook_url: None,
            max_queued_training_jobs: 32,
            max_tracked_jobs: 1024,
            tracked_job_ttl: std::time::Duration::from_secs(604_800),
            adapter_max_disk_bytes: Some(100 * 1024u64.pow(3)),
            composed_cache_max_bytes: Some(10 * 1024u64.pow(3)),
            composed_cache_max_entries: Some(64),
            served_model_id,
            decode_stats: Arc::new(std::sync::Mutex::new(DecodeStatsRing::new(4096))),
            recent_requests: Arc::new(std::sync::Mutex::new(RecentRequestsRing::new(
                RECENT_REQUESTS_CAPACITY,
            ))),
            request_log: None,
            request_log_max_capture_bytes: crate::request_log::RequestLogConfig::default()
                .max_capture_bytes,
            deterministic_cache_generations: Arc::new(std::sync::Mutex::new(
                DeterministicCacheGenerations::default(),
            )),
            completion_cache: Arc::new(std::sync::Mutex::new(DeterministicCompletionCache::new(
                DETERMINISTIC_COMPLETION_CACHE_CAPACITY,
            ))),
            chat_request_cache: Arc::new(std::sync::Mutex::new(
                DeterministicChatRequestCache::new(DETERMINISTIC_CHAT_REQUEST_CACHE_CAPACITY),
            )),
            chat_choices_cache: Arc::new(std::sync::Mutex::new(
                DeterministicChatChoicesCache::new(DETERMINISTIC_CHAT_CHOICES_CACHE_CAPACITY),
            )),
            batch_cache: Arc::new(std::sync::Mutex::new(DeterministicBatchCache::new(
                DETERMINISTIC_BATCH_CACHE_CAPACITY,
            ))),
            rendered_prompt_cache: Arc::new(std::sync::Mutex::new(RenderedPromptCache::new(
                RENDERED_PROMPT_CACHE_CAPACITY,
            ))),
            prompt_token_cache: Arc::new(std::sync::Mutex::new(PromptTokenCache::new(
                PROMPT_TOKEN_CACHE_CAPACITY,
            ))),
            eval_queue: crate::eval::new_shared_eval_queue(),
            eval_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            suite_registry: None,
            dataset_registry: None,
            judgment_store: None,
            max_queued_eval_jobs: 32,
            max_tracked_eval_jobs: 1024,
            eval_webhook_url: None,
        }
    }

    /// Create an AppState with a real ModelRunner backend and paged KV cache.
    ///
    /// Uses `block_size=16` by default. The number of blocks can be overridden
    /// via `memory_cfg.num_blocks`. Otherwise derived from available VRAM or
    /// `max_position_embeddings / block_size`.
    ///
    /// GPU memory sharing: `memory_cfg.inference_memory_fraction` (default 0.7)
    /// controls what fraction of remaining VRAM (after model weights) is allocated
    /// to KV cache, reserving the rest for training. Set to 1.0 for inference-only.
    pub fn new_real(
        model_config: ModelConfig,
        runner: ModelRunner,
        tokenizer: KilnTokenizer,
        device_kt: kiln_tensor::Device,
        adapter_dir: PathBuf,
        memory_cfg: &crate::config::MemoryConfig,
        response_delivery_policy: crate::batching_engine::ResponseDeliveryPolicy,
        max_batch_tokens: crate::config::BatchTokenBudget,
        request_timeout_secs: u64,
        served_model_id: String,
        prefix_cache_cfg: &crate::config::PrefixCacheConfig,
        base_teacher_identity: Option<Arc<kiln_train::TeacherIdentityV1>>,
    ) -> anyhow::Result<Self> {
        let backend_capabilities = runner.backend_capabilities();
        let decode_runtime_config = crate::batching_engine::resolve_decode_runtime_config(
            crate::config::DeterministicInference::default(),
            crate::config::MaxDecodeBatch::default(),
            Some(backend_capabilities.decode_batcher),
            max_batch_tokens,
        );
        let speculative_runtime_policy =
            SpeculativeRuntimePolicy::new(backend_capabilities.decode.mtp_speculative_generation);
        let streaming_prefill_runtime_config = crate::config::StreamingPrefillConfig::default()
            .resolve(backend_capabilities.streaming_prefill);
        Self::new_real_with_serving_profile(
            model_config,
            runner,
            tokenizer,
            device_kt,
            adapter_dir,
            memory_cfg,
            response_delivery_policy,
            decode_runtime_config,
            crate::config::BatchingConfig::default(),
            streaming_prefill_runtime_config,
            crate::config::SpeculativeDecodingConfig::default(),
            speculative_runtime_policy,
            max_batch_tokens,
            crate::config::PrefillTokenBudget::default(),
            crate::config::PrefillLayerBudget::default(),
            request_timeout_secs,
            served_model_id,
            prefix_cache_cfg,
            base_teacher_identity,
            crate::config::ServingProfileSetting::default(),
            kiln_train::GradientCheckpointPolicy::Auto,
            kiln_train::CheckpointBoundaryPolicy::default(),
        )
    }

    /// Production constructor with an explicit process-lifetime serving
    /// profile. The compatibility constructor above resolves to stable.
    #[allow(clippy::too_many_arguments)]
    pub fn new_real_with_serving_profile(
        model_config: ModelConfig,
        runner: ModelRunner,
        tokenizer: KilnTokenizer,
        device_kt: kiln_tensor::Device,
        adapter_dir: PathBuf,
        memory_cfg: &crate::config::MemoryConfig,
        response_delivery_policy: crate::batching_engine::ResponseDeliveryPolicy,
        decode_runtime_config: crate::config::DecodeRuntimeConfig,
        batching_config: crate::config::BatchingConfig,
        streaming_prefill_runtime_config: crate::config::StreamingPrefillRuntimeConfig,
        speculative_config: crate::config::SpeculativeDecodingConfig,
        speculative_runtime_policy: SpeculativeRuntimePolicy,
        max_batch_tokens: crate::config::BatchTokenBudget,
        max_prefill_tokens_per_cycle: crate::config::PrefillTokenBudget,
        max_prefill_layers_per_cycle: crate::config::PrefillLayerBudget,
        request_timeout_secs: u64,
        served_model_id: String,
        prefix_cache_cfg: &crate::config::PrefixCacheConfig,
        base_teacher_identity: Option<Arc<kiln_train::TeacherIdentityV1>>,
        serving_profile: crate::config::ServingProfileSetting,
        gradient_checkpoint_policy: kiln_train::GradientCheckpointPolicy,
        checkpoint_boundary_policy: kiln_train::CheckpointBoundaryPolicy,
    ) -> anyhow::Result<Self> {
        speculative_config.validate_for_model(&model_config)?;
        speculative_config.validate_for_serving()?;
        anyhow::ensure!(
            runner.streaming_prefill_policy()
                == streaming_prefill_runtime_config.execution_policy(),
            "model runner streaming-prefill policy does not match the server runtime configuration"
        );
        let vram_probe_selector = ensure_accelerator_memory_probe_identity(device_kt)?;
        let physical_vram = kiln_memory::vram::detect_vram_for(vram_probe_selector);
        let vram_capacity_resolution =
            kiln_memory::vram::resolve_vram_capacity(physical_vram, memory_cfg.gpu_memory_gb);
        ensure_accelerator_memory_capacity(
            device_kt,
            vram_probe_selector,
            vram_capacity_resolution.effective,
        )?;
        ensure_accelerator_memory_floor(device_kt, vram_capacity_resolution.effective, memory_cfg)?;
        kiln_memory::MemoryGovernor::configure_global(
            vram_probe_selector,
            memory_cfg.governor_config_for_capacity(vram_capacity_resolution.effective.total_bytes),
        )?;
        let serving_policy = serving_profile.runtime_policy();
        let kv_autoscaler_config = crate::kv_autoscaler::KvAutoscalerConfig {
            requested: memory_cfg.kv_autoscale.enabled(),
            requested_source: memory_cfg.kv_autoscale.source(),
            force_blocks: memory_cfg.kv_force_blocks.target(),
            force_blocks_source: memory_cfg.kv_force_blocks.source(),
        };
        if kv_autoscaler_config.force_blocks.is_some() {
            anyhow::ensure!(
                kv_autoscaler_config.requested,
                "memory.kv_force_blocks requires memory.kv_autoscale=true"
            );
            anyhow::ensure!(
                serving_profile.profile() == crate::config::ServingProfile::Maintenance,
                "memory.kv_force_blocks requires server.serving_profile=maintenance"
            );
        }
        let base_weight_shard_manifest = runner
            .weights
            .base_weight_shard_manifest
            .clone()
            .map(Arc::new);
        let execution_provenance = runner.weights.execution_provenance.clone().map(Arc::new);
        let block_size = DEFAULT_BLOCK_SIZE;
        // §3.2 teacher registry — loaded from `adapter_dir/teachers.json`
        // if present. Clone-able Arc so the AppState field below can
        // own its handle.
        let teacher_registry_for_real = {
            let teachers_path = adapter_dir.join("teachers.json");
            std::sync::Arc::new(crate::api::teachers::TeacherRegistry::load_from_path(
                &teachers_path,
            ))
        };

        // KV cache dtype must match the model's activation dtype, otherwise
        // `paged_cache.write` hits a slice-set dtype mismatch on the first
        // full-attention layer. The previous `cuda_is_available()` check was
        // a compile-time cfg(feature = "cuda"), so Metal builds ran the F32
        // branch even though the Qwen3.5-4B model loads in BF16 — prefill
        // failed on every request. Key the choice off `model_config.dtype`
        // instead so tests with F32 tiny configs keep working and any real
        // BF16 model gets a matching BF16 cache regardless of backend.
        let kv_dtype = match model_config.dtype {
            kiln_core::config::DType::BF16 => DType::BF16,
            kiln_core::config::DType::FP16 => DType::F16,
            kiln_core::config::DType::FP32 => DType::F32,
        };

        let kv_dtype_bytes: usize = if memory_cfg.kv_cache_fp8 {
            1 // FP8: 1 byte per element
        } else {
            match kv_dtype {
                DType::BF16 | DType::F16 => 2,
                _ => 4,
            }
        };

        let configured_inference_fraction = memory_cfg.inference_memory_fraction.clamp(0.1, 1.0);

        // Estimate model weight memory (approximate: params * dtype_bytes)
        // Qwen3.5-4B ≈ 4B params * 2 bytes (bf16) ≈ 8GB
        let estimated_model_bytes: u64 = estimate_model_memory_bytes(&model_config);

        // Compute KV cache bytes per block:
        // num_full_attention_layers * 2 (K+V) * num_kv_heads * head_dim * block_size * dtype_bytes
        let bytes_per_block: u64 = (model_config.num_full_attention_layers
            * 2
            * model_config.num_kv_heads
            * model_config.head_dim
            * block_size
            * kv_dtype_bytes) as u64;

        // Detect the selected device once, then apply the typed configured cap.
        // A failed physical probe stays at zero; startup never manufactures an
        // optimistic fallback capacity that could turn into a fatal allocation.
        let vram_info = vram_capacity_resolution.effective;
        let storage_capabilities = runner.backend_capabilities().storage;
        let inference_recurrent_state_policy =
            runner.backend_capabilities().gdn.inference_recurrent_state;
        let prefix_cache_state_bytes_per_entry =
            linear_attention_state_bytes(&model_config, inference_recurrent_state_policy);
        let kv_auto_block_policy = storage_capabilities.kv_auto_block_policy;
        let gpu_memory_budget_policy = storage_capabilities.gpu_memory_budget_policy;
        let gpu_allocator_memory_probe_policy =
            storage_capabilities.gpu_allocator_memory_probe_policy;
        let gpu_memory_reclaim_policy = storage_capabilities.gpu_memory_reclaim_policy;

        // Physical total and used normally come from the same all-process driver
        // snapshot. A configured total is different by design: it is the static
        // sizing ceiling, while used/free telemetry remains physical so the
        // governor and allocator caps still see coexisting GPU workloads.
        let snap = if gpu_memory_budget_policy.use_live_memory_snapshot {
            let s = kiln_memory::vram::current_memory_snapshot_for(vram_probe_selector);
            (s.total_bytes > 0).then_some(s)
        } else {
            None
        };
        if vram_probe_selector != kiln_memory::vram::VramProbeSelector::None {
            // Supervise the cached admission source before any large KV
            // allocation or retry. A slow zero-fill must not leave a later
            // retry depending on an initial sample that aged out while the
            // sampler had not yet been started. This publisher never reclaims
            // or mutates accelerator state; reclaim hooks remain wired only
            // after actor/GPU coordination exists below.
            let governor = kiln_memory::MemoryGovernor::global();
            let published = governor.refresh();
            anyhow::ensure!(
                published.total_bytes > 0 && !published.observations.probe_failed,
                "selected-device memory probe lost its safe capacity before KV allocation; refusing to allocate"
            );
            anyhow::ensure!(
                governor.start_sampler(),
                "failed to start the selected-device memory sampler before KV allocation"
            );
        }
        let total_vram = vram_info.total_bytes;
        let prefix_cache_enabled =
            effective_prefix_cache_enabled(prefix_cache_cfg.enabled, &device_kt);
        if prefix_cache_cfg.enabled && !prefix_cache_enabled {
            tracing::warn!(
                backend = "vulkan",
                "cross-request prefix reuse is correctness-quarantined; using fresh prefill for every request"
            );
        }
        let host_prefix_cache_reserve_bytes = prefix_cache_host_reserve_bytes(
            host_backed_free_bytes_for_device(device_kt, snap),
            prefix_cache_state_bytes_per_entry,
            prefix_cache_enabled,
            prefix_cache_cfg.max_entries,
        )?;
        if matches!(device_kt, kiln_tensor::Device::Vulkan(_)) {
            let host_backed = snap.and_then(|snapshot| snapshot.observations.host_backed);
            tracing::info!(
                host_backed_total_gb = host_backed.map(|tier| tier.total_bytes as f64 / 1e9),
                host_backed_used_gb = host_backed.map(|tier| tier.used_bytes as f64 / 1e9),
                host_backed_free_gb = host_backed.map(|tier| tier.free_bytes as f64 / 1e9),
                prefix_cache_host_reserve_gb = host_prefix_cache_reserve_bytes as f64 / 1e9,
                "Vulkan host-backed serving budget"
            );
        }
        let allocator_memory_snapshot = crate::device_memory::allocator_memory_snapshot(
            gpu_allocator_memory_probe_policy,
            &device_kt,
        );
        if let Some(allocator) = allocator_memory_snapshot {
            tracing::info!(
                allocator_free_gb = allocator.free_bytes as f64 / 1e9,
                allocator_total_gb = allocator.total_bytes as f64 / 1e9,
                allocator_pool_reserved_gb = allocator
                    .pool_reserved_bytes
                    .map(|bytes| bytes as f64 / 1e9),
                allocator_pool_used_gb = allocator.pool_used_bytes.map(|bytes| bytes as f64 / 1e9),
                allocator_pool_spare_gb = allocator
                    .pool_reserved_bytes
                    .zip(allocator.pool_used_bytes)
                    .map(|(reserved, used)| reserved.saturating_sub(used) as f64 / 1e9),
                source = allocator.source,
                "backend allocator memory snapshot for KV sizing"
            );
        }

        let post_load_used_vram_info =
            runtime_used_vram_for_policy(gpu_memory_budget_policy, vram_probe_selector);
        let post_load_used_vram = snap
            .map(|s| s.used_bytes)
            .or_else(|| post_load_used_vram_info.map(|info| info.used_bytes))
            .unwrap_or(0);
        let mut sizing_residency_bytes = post_load_used_vram.max(estimated_model_bytes);
        // Vulkan keeps BOTH the paged KV pool AND the resident-decode weight
        // prewarm caches (f32 decode weights + bf16-packed, empirically ~1.85x
        // the model) in VRAM. The post-load snapshot is taken BEFORE the prewarm
        // allocates, so without reserving for it the KV auto-sizer over-budgets
        // and the prewarm OOMs (KV pool + model + prewarm > VRAM). Reserve ~2x
        // the model here so the KV sizer leaves headroom for the prewarm.
        let reserve_multiplier = storage_capabilities.kv_sizing_residency_model_multiplier;
        if reserve_multiplier > 0 {
            sizing_residency_bytes = sizing_residency_bytes
                .saturating_add(estimated_model_bytes.saturating_mul(reserve_multiplier));
        }
        if post_load_used_vram > 0 {
            let used_source = snap
                .map(|s| s.source)
                .or_else(|| post_load_used_vram_info.map(|i| i.source))
                .unwrap_or(kiln_memory::vram::VramSource::None);
            tracing::info!(
                post_load_used_vram_gb = post_load_used_vram as f64 / 1e9,
                total_vram_gb = total_vram as f64 / 1e9,
                estimated_model_gb = estimated_model_bytes as f64 / 1e9,
                source = %used_source,
                "post-load device residency snapshot for KV sizing"
            );
        } else {
            tracing::warn!(
                estimated_model_gb = estimated_model_bytes as f64 / 1e9,
                "post-load device residency unavailable; falling back to static model memory estimate for KV sizing"
            );
        }

        // Compute num_blocks for a given fraction. Used both for the explicit
        // `memory_cfg.num_blocks` path and the auto-sizer retry loop below.
        let compute_blocks_for_fraction = |fraction: f64| -> usize {
            let n = auto_num_blocks_for_fraction(
                total_vram,
                sizing_residency_bytes,
                bytes_per_block,
                fraction,
                model_config.max_position_embeddings,
                block_size,
                kv_auto_block_policy,
            );
            // Additionally clamp so the KV pool fits within the live budget.
            // The governor provides the OS/driver-wide pressure view; CUDA/ROCm
            // also expose the allocator heap that the actual KV tensors will
            // allocate from, while Vulkan's current KV implementation requires
            // a separately bounded host-backed tier. Use the strictest known
            // ceiling so a large DRM aperture cannot authorize a fatal host
            // allocation.
            if gpu_memory_budget_policy.cap_kv_blocks_by_live_budget && bytes_per_block > 0 {
                let governor = kiln_memory::MemoryGovernor::global();
                let governor_observation = governor.cached_observation();
                let governor_avail = governor_observation.available_bytes;
                let allocator_budget = crate::device_memory::allocator_kv_budget_bytes_for_fraction(
                    gpu_allocator_memory_probe_policy,
                    governor,
                    &device_kt,
                    fraction,
                );
                let host_backed_budget = host_backed_kv_budget_for_fraction(
                    device_kt,
                    Some(governor_observation.snapshot),
                    host_prefix_cache_reserve_bytes,
                    fraction,
                );
                let residency_budget =
                    minimum_optional_budget(allocator_budget, host_backed_budget);
                let (capped, max_blocks) = cap_kv_blocks_to_live_budget(
                    n,
                    bytes_per_block,
                    governor_avail,
                    residency_budget,
                );
                if max_blocks < MIN_AUTO_KV_BLOCKS {
                    tracing::warn!(
                        fraction,
                        proposed_blocks = n,
                        max_live_budget_blocks = max_blocks,
                        min_auto_blocks = MIN_AUTO_KV_BLOCKS,
                        backend_policy_requested_minimum =
                            kv_auto_block_policy.allow_min_blocks_below_live_budget,
                        governor_available_gb = governor_avail as f64 / 1e9,
                        allocator_budget_gb = allocator_budget.map(|bytes| bytes as f64 / 1e9),
                        host_backed_budget_gb = host_backed_budget.map(|bytes| bytes as f64 / 1e9),
                        capped_blocks = capped,
                        "KV cache auto-sizer live budget is below the preferred minimum; refusing to allocate above the live budget"
                    );
                } else if capped < n {
                    tracing::warn!(
                        fraction,
                        proposed_blocks = n,
                        capped_blocks = capped,
                        max_live_budget_blocks = max_blocks,
                        governor_available_gb = governor_avail as f64 / 1e9,
                        allocator_budget_gb = allocator_budget.map(|bytes| bytes as f64 / 1e9),
                        host_backed_budget_gb = host_backed_budget.map(|bytes| bytes as f64 / 1e9),
                        "KV cache auto-sizer capped by live residency memory"
                    );
                }
                capped
            } else {
                n
            }
        };

        // Backend storage policy owns whether requested FP8 KV cache storage is
        // allowed.
        let fp8_enabled = {
            let requested = memory_cfg.kv_cache_fp8;
            let policy = storage_capabilities.kv_cache_fp8_policy;
            let enabled = policy.enabled(requested);
            if requested && !enabled {
                tracing::warn!(
                    reason = policy.disabled_reason,
                    "FP8 KV cache disabled by backend storage policy"
                );
            }
            enabled
        };
        // Allocation closure: try to build the paged KV cache for `n` blocks.
        // Used by the auto-sizer retry loop below. CUDA OOM bubbles up here as
        // an `Err` from `Tensor::empty`, which we catch and retry with a smaller
        // budget instead of panicking on the first failure.
        //
        // #1082 candle-drop: candle `PagedKvCache::new_uninit_with_fp8_kt(..,
        // &device_kt, ..)` -> kt `PagedKvCacheKt::new_with_fp8(.., device, fp8)`.
        // The kt cache now allocates its pools on the model's *runtime* device,
        // so pass `device_kt` through (a Metal model gets Metal pools, a CPU
        // model gets CPU pools) — this is the device-routing fix. NOTE: the kt
        // constructor zero-fills the pools; the old `new_uninit_*` left them
        // uninitialized — a one-time startup memset, not a correctness change
        // (paged writes overwrite slots before they are read).
        let allocation_attempt = std::cell::Cell::new(0u64);
        let allocate_cache = |n: usize| -> anyhow::Result<PagedKvCacheKt> {
            validate_kv_allocation_against_live_allocator(
                &device_kt,
                n,
                bytes_per_block,
                gpu_memory_budget_policy,
                gpu_allocator_memory_probe_policy,
                kv_auto_block_policy,
                kiln_memory::MemoryGovernor::global(),
                host_prefix_cache_reserve_bytes,
            )?;
            let attempt = || {
                let attempt_number = allocation_attempt.get().saturating_add(1);
                allocation_attempt.set(attempt_number);
                let requested_bytes = (n as u64).saturating_mul(bytes_per_block);
                let started = std::time::Instant::now();
                let result = PagedKvCacheKt::new_with_fp8(
                    model_config.num_full_attention_layers,
                    n,
                    block_size,
                    model_config.num_kv_heads,
                    model_config.head_dim,
                    kv_dtype,
                    device_kt,
                    fp8_enabled,
                );
                let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
                match &result {
                    Ok(_) => tracing::info!(
                        event = "gpu_memory_operation",
                        operation = "allocation",
                        reason = "initial_kv_cache",
                        outcome = "completed",
                        ?device_kt,
                        attempt = attempt_number,
                        num_blocks = n,
                        block_size,
                        requested_bytes,
                        actual_bytes = requested_bytes,
                        wait_ms = 0.0,
                        duration_ms,
                        ?kv_dtype,
                        fp8_enabled,
                        "paged KV cache allocation completed"
                    ),
                    Err(error) => tracing::warn!(
                        event = "gpu_memory_operation",
                        operation = "allocation",
                        reason = "initial_kv_cache",
                        outcome = "failed",
                        error = %format!("{error:#}"),
                        ?device_kt,
                        attempt = attempt_number,
                        num_blocks = n,
                        block_size,
                        requested_bytes,
                        actual_bytes = 0,
                        wait_ms = 0.0,
                        duration_ms,
                        ?kv_dtype,
                        fp8_enabled,
                        "paged KV cache allocation failed"
                    ),
                }
                result
            };
            match attempt() {
                Ok(c) => Ok(c),
                // OOM recovery (the "never OOM" path): before falling back to a
                // smaller fraction, ask the governor to return pooled-but-unused
                // VRAM to the OS and retry once at the SAME size — the alloc may
                // have failed only because freed blocks were still pooled (or a
                // coexisting job briefly spiked). Cheaper than shrinking the KV
                // cache if the memory is genuinely reclaimable.
                Err(first)
                    if serving_policy.allocator_reclaim
                        && gpu_memory_budget_policy.retry_kv_allocation_after_reclaim =>
                {
                    let reclaim_started = std::time::Instant::now();
                    let freed = kiln_memory::MemoryGovernor::global().reclaim(u64::MAX);
                    let reclaim_duration_ms = reclaim_started.elapsed().as_secs_f64() * 1000.0;
                    tracing::warn!(
                        event = "gpu_memory_operation",
                        operation = "reclaim",
                        reason = "initial_kv_allocation_retry",
                        outcome = if freed > 0 { "reclaimed" } else { "zero_yield" },
                        num_blocks = n,
                        target_bytes = u64::MAX,
                        actual_bytes = freed,
                        wait_ms = 0.0,
                        duration_ms = reclaim_duration_ms,
                        reclaimed_mb = freed / (1024 * 1024),
                        "KV cache allocation failed; reclaimed pooled VRAM and retrying at same size"
                    );
                    attempt().map_err(|_| first)
                }
                Err(e) => Err(e),
            }
        };

        // Determine num_blocks + paged cache:
        //   - If `memory_cfg.num_blocks` is set, honor it exactly when it fits
        //     the live safety budget (no retry or silent shrink).
        //   - Otherwise, run the auto-sizer retry loop, starting at the
        //     configured `inference_memory_fraction` and shrinking on OOM.
        let (paged_cache, num_blocks, inference_fraction) = if let Some(explicit) =
            memory_cfg.num_blocks
        {
            tracing::info!(
                num_blocks = explicit,
                block_size,
                ?kv_dtype,
                fp8_enabled,
                "allocating paged KV cache (explicit num_blocks)"
            );
            let cache = allocate_cache(explicit).map_err(|error| {
                anyhow::anyhow!(
                    "failed to allocate explicitly configured paged KV cache with memory.num_blocks={explicit}: {error:#}"
                )
            })?;
            (cache, explicit, configured_inference_fraction)
        } else {
            let initial_auto_blocks = compute_blocks_for_fraction(configured_inference_fraction);
            anyhow::ensure!(
                initial_auto_blocks > 0,
                "paged KV cache auto-sizing cannot fit even one block within the current live accelerator/host residency budget after model residency, prefix-cache state, memory.floor_gb, and allocator reservations. Refusing to attempt an allocation. Free accelerator or host memory, reduce memory.floor_gb, lower model residency, or choose a device with more available memory"
            );
            tracing::info!(
                total_vram_gb = total_vram as f64 / 1e9,
                model_gb = estimated_model_bytes as f64 / 1e9,
                post_load_used_vram_gb = post_load_used_vram as f64 / 1e9,
                sizing_residency_gb = sizing_residency_bytes as f64 / 1e9,
                inference_fraction = configured_inference_fraction,
                "memory-aware KV cache sizing"
            );
            match auto_size_with_retry(
                configured_inference_fraction,
                AUTO_SIZER_FALLBACK_FRACTIONS,
                &compute_blocks_for_fraction,
                |n| {
                    tracing::info!(
                        num_blocks = n,
                        block_size,
                        ?kv_dtype,
                        fp8_enabled,
                        "allocating paged KV cache"
                    );
                    allocate_cache(n).map_err(|e| format!("{e:#}"))
                },
            ) {
                Ok(success) => {
                    if success.fraction < configured_inference_fraction {
                        tracing::warn!(
                            configured_fraction = configured_inference_fraction,
                            actual_fraction = success.fraction,
                            num_blocks = success.num_blocks,
                            attempts = success.attempted_failures.len() + 1,
                            "KV cache auto-sizer fell back to a smaller inference_memory_fraction \
                             because the configured value OOM'd; set memory.inference_memory_fraction \
                             (or KILN_INFERENCE_MEMORY_FRACTION) to this value to silence the warning"
                        );
                    } else {
                        tracing::info!(
                            inference_fraction = success.fraction,
                            num_blocks = success.num_blocks,
                            "KV cache auto-sizer succeeded on first attempt"
                        );
                    }
                    (success.cache, success.num_blocks, success.fraction)
                }
                Err(failure) => {
                    let suggested_blocks = suggested_emergency_num_blocks(
                        total_vram,
                        sizing_residency_bytes,
                        bytes_per_block,
                        block_size,
                        model_config.max_position_embeddings,
                        kv_auto_block_policy,
                    );
                    let msg = format_oom_remediation_message(
                        &failure,
                        total_vram,
                        sizing_residency_bytes,
                        bytes_per_block,
                        suggested_blocks,
                        configured_inference_fraction,
                        vram_info.source,
                    );
                    tracing::error!("{msg}");
                    return Err(anyhow::anyhow!(msg));
                }
            }
        };

        let block_manager = BlockManager::new(num_blocks, block_size);
        let kv_cache_bytes = num_blocks as u64 * bytes_per_block;
        let memory_budget = GpuMemoryBudget::compute(
            total_vram,
            sizing_residency_bytes,
            estimated_model_bytes,
            post_load_used_vram,
            kv_cache_bytes,
            inference_fraction,
            memory_cfg.training_memory_gb,
        );
        let backend_name = runner.backend_name();
        let backend_capabilities = runner.backend_capabilities();

        tracing::info!(
            total_vram_gb = memory_budget.total_vram_bytes as f64 / 1e9,
            vram_source = %vram_info.source,
            model_gb = memory_budget.model_memory_bytes as f64 / 1e9,
            estimated_model_gb = memory_budget.estimated_model_memory_bytes as f64 / 1e9,
            post_load_used_vram_gb = memory_budget.post_load_used_vram_bytes as f64 / 1e9,
            kv_cache_gb = memory_budget.kv_cache_bytes as f64 / 1e9,
            training_budget_gb = memory_budget.training_budget_bytes as f64 / 1e9,
            inference_fraction = memory_budget.inference_memory_fraction,
            "GPU memory budget"
        );

        log_backend_training_acceleration_profile(
            backend_capabilities.training.acceleration_profile,
        );
        let prefix_cache_max_blocks = if prefix_cache_enabled {
            prefix_cache_cfg
                .max_blocks
                .unwrap_or_else(|| default_prefix_cache_max_blocks(num_blocks))
        } else {
            0
        };
        let prefix_cache_max_entries = if prefix_cache_enabled {
            prefix_cache_cfg.max_entries.unwrap_or_else(|| {
                if matches!(device_kt, kiln_tensor::Device::Vulkan(_)) {
                    prefix_cache_entries_for_state_budget(
                        host_prefix_cache_reserve_bytes,
                        prefix_cache_state_bytes_per_entry,
                    )
                } else {
                    default_prefix_cache_max_entries(total_vram, prefix_cache_state_bytes_per_entry)
                }
            })
        } else {
            MIN_PREFIX_CACHE_MAX_ENTRIES
        };
        tracing::info!(
            max_blocks = prefix_cache_max_blocks,
            max_entries = prefix_cache_max_entries,
            state_bytes_per_entry = prefix_cache_state_bytes_per_entry,
            max_state_bytes =
                prefix_cache_state_bytes_per_entry.saturating_mul(prefix_cache_max_entries as u64),
            min_register_tokens = REAL_PREFIX_CACHE_MIN_REGISTER_TOKENS,
            "prefix cache budget"
        );
        let prefix_cache = if prefix_cache_enabled {
            RealPrefixCache::new_with_min_register_tokens(
                true,
                block_size,
                prefix_cache_max_blocks,
                prefix_cache_max_entries,
                prefix_cache_state_bytes_per_entry,
                REAL_PREFIX_CACHE_MIN_REGISTER_TOKENS,
            )
        } else {
            RealPrefixCache::disabled(block_size)
        };

        let model_weight_device = runner.weights.embed_tokens.device();
        let rocm_graph_telemetry = runner.rocm_graph_telemetry_handle();
        let runner = Arc::new(std::sync::RwLock::new(runner));
        let backend_health = runner.read().unwrap().backend_health_handle();
        let block_manager = Arc::new(std::sync::Mutex::new(block_manager));
        let paged_cache = Arc::new(paged_cache);
        let prefix_cache = Arc::new(std::sync::Mutex::new(prefix_cache));
        let gpu_lock = Arc::new(RwLock::new(()));
        let loaded_adapter = Arc::new(std::sync::RwLock::new(None));
        let decode_batcher_policy = backend_capabilities.decode_batcher;
        debug_assert_eq!(
            decode_runtime_config.max_decode_batch.backend_policy,
            decode_batcher_policy
                .engine_max_decode_batch
                .unwrap_or(decode_batcher_policy.max_batch)
        );
        let max_decode_batch = decode_runtime_config.max_decode_batch.effective;
        let batching_runtime_config = batching_config.resolve(
            crate::config::BatchingBackendPolicy::from_decode_batcher_policy(decode_batcher_policy),
            max_decode_batch,
        );
        crate::config::validate_actor_prefill_tile_contract(
            batching_runtime_config,
            streaming_prefill_runtime_config,
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_decode_batch,
        )?;
        tracing::info!(
            backend = backend_name,
            mode_configured = %batching_runtime_config.mode.configured,
            mode_configured_source = %batching_runtime_config.mode.configured_source,
            mode_backend_policy_enabled = batching_runtime_config.mode.backend_policy_enabled,
            mode_effective_enabled = batching_runtime_config.mode.effective_enabled,
            mode_effective_source = %batching_runtime_config.mode.effective_source,
            actor_prefill_tile_alignment_required = batching_runtime_config
                .actor_prefill_tile_alignment_required,
            rowwise_decode = batching_runtime_config.rowwise_decode.enabled,
            rowwise_decode_source = %batching_runtime_config.rowwise_decode.source,
            prefix_aware_admission = batching_runtime_config.prefix_aware_admission.enabled,
            prefix_aware_admission_source = %batching_runtime_config.prefix_aware_admission.source,
            prefill_admission_quantum_configured = ?batching_runtime_config
                .prefill_admission_quantum
                .configured,
            prefill_admission_quantum_configured_source = %batching_runtime_config
                .prefill_admission_quantum
                .configured_source,
            prefill_admission_quantum_backend_policy = batching_runtime_config
                .prefill_admission_quantum
                .backend_policy,
            prefill_admission_quantum_effective = batching_runtime_config
                .prefill_admission_quantum
                .effective,
            prefill_admission_quantum_effective_source = %batching_runtime_config
                .prefill_admission_quantum
                .effective_source,
            actor_cycle_idle_ms = batching_runtime_config.actor_cycle_idle.milliseconds,
            actor_cycle_idle_source = %batching_runtime_config.actor_cycle_idle.source,
            actor_cycle_idle_enabled = batching_runtime_config.actor_cycle_idle.enabled,
            actor_cycle_idle_command_poll_ms = batching_runtime_config
                .actor_cycle_idle
                .command_poll_milliseconds,
            direct_decode_rendezvous_scope = DirectDecodeRendezvousRuntimeState::SCOPE,
            direct_decode_rendezvous_mode_configured = %batching_runtime_config
                .direct_decode_rendezvous
                .mode
                .configured,
            direct_decode_rendezvous_mode_configured_source = %batching_runtime_config
                .direct_decode_rendezvous
                .mode
                .configured_source,
            direct_decode_rendezvous_mode_backend_policy_enabled = batching_runtime_config
                .direct_decode_rendezvous
                .mode
                .backend_policy_enabled,
            direct_decode_rendezvous_mode_effective_enabled = batching_runtime_config
                .direct_decode_rendezvous
                .mode
                .effective_enabled,
            direct_decode_rendezvous_mode_effective_source = %batching_runtime_config
                .direct_decode_rendezvous
                .mode
                .effective_source,
            direct_decode_rendezvous_max_batch_configured = ?batching_runtime_config
                .direct_decode_rendezvous
                .max_batch
                .configured,
            direct_decode_rendezvous_max_batch_configured_source = %batching_runtime_config
                .direct_decode_rendezvous
                .max_batch
                .configured_source,
            direct_decode_rendezvous_max_batch_backend_policy = batching_runtime_config
                .direct_decode_rendezvous
                .max_batch
                .backend_policy,
            direct_decode_rendezvous_max_batch_effective = batching_runtime_config
                .direct_decode_rendezvous
                .max_batch
                .effective,
            direct_decode_rendezvous_max_batch_effective_source = %batching_runtime_config
                .direct_decode_rendezvous
                .max_batch
                .effective_source,
            direct_decode_rendezvous_wait_us_configured = ?batching_runtime_config
                .direct_decode_rendezvous
                .wait_us
                .configured,
            direct_decode_rendezvous_wait_us_configured_source = %batching_runtime_config
                .direct_decode_rendezvous
                .wait_us
                .configured_source,
            direct_decode_rendezvous_wait_us_backend_policy = batching_runtime_config
                .direct_decode_rendezvous
                .wait_us
                .backend_policy,
            direct_decode_rendezvous_wait_us_effective = batching_runtime_config
                .direct_decode_rendezvous
                .wait_us
                .effective,
            direct_decode_rendezvous_wait_us_effective_source = %batching_runtime_config
                .direct_decode_rendezvous
                .wait_us
                .effective_source,
            direct_decode_rendezvous_mixed_seq_lens_configured = ?batching_runtime_config
                .direct_decode_rendezvous
                .mixed_seq_lens
                .configured,
            direct_decode_rendezvous_mixed_seq_lens_configured_source = %batching_runtime_config
                .direct_decode_rendezvous
                .mixed_seq_lens
                .configured_source,
            direct_decode_rendezvous_mixed_seq_lens_backend_policy = batching_runtime_config
                .direct_decode_rendezvous
                .mixed_seq_lens
                .backend_policy,
            direct_decode_rendezvous_mixed_seq_lens_effective = batching_runtime_config
                .direct_decode_rendezvous
                .mixed_seq_lens
                .effective,
            direct_decode_rendezvous_mixed_seq_lens_effective_source = %batching_runtime_config
                .direct_decode_rendezvous
                .mixed_seq_lens
                .effective_source,
            burst_prefill_admission = batching_runtime_config.burst_prefill_admission,
            "batching runtime configuration resolved"
        );
        let direct_decode_rendezvous = batching_runtime_config.direct_decode_rendezvous;
        let decode_batcher_config =
            direct_decode_rendezvous
                .mode
                .effective_enabled
                .then_some(DecodeBatcherConfig {
                    max_batch: direct_decode_rendezvous.max_batch.effective,
                    wait: std::time::Duration::from_micros(
                        direct_decode_rendezvous.wait_us.effective,
                    ),
                    allow_mixed_seq_lens: direct_decode_rendezvous.mixed_seq_lens.effective,
                });
        if decode_batcher_policy.warm_resident_decode_pool_on_startup
            && backend_capabilities.decode.resident_decode.is_native()
        {
            let resident_max_batch = max_decode_batch.max(
                decode_batcher_config
                    .map(|config| config.max_batch)
                    .unwrap_or(1),
            );
            let ready = runner
                .read()
                .unwrap()
                .warm_resident_decode_pool(resident_max_batch)
                .unwrap_or_else(|error| {
                    tracing::warn!(
                        error = %format!("{error:#}"),
                        "resident decode pool startup allocation rejected"
                    );
                    false
                });
            tracing::info!(
                max_batch = resident_max_batch,
                ready,
                "resident decode pool startup allocation"
            );
        }
        if !batching_runtime_config.mode.effective_enabled {
            tracing::info!(
                effective_source = %batching_runtime_config.mode.effective_source,
                "batching engine disabled by resolved startup configuration"
            );
        }
        let batching_engine = batching_runtime_config.mode.effective_enabled.then(|| {
            tracing::info!(
                backend = backend_name,
                max_decode_batch,
                max_decode_batch_configured = ?decode_runtime_config.max_decode_batch.configured,
                max_decode_batch_configured_source = %decode_runtime_config.max_decode_batch.configured_source,
                max_decode_batch_backend_policy = decode_runtime_config.max_decode_batch.backend_policy,
                max_decode_batch_effective_source = %decode_runtime_config.max_decode_batch.effective_source,
                deterministic = decode_runtime_config.deterministic.enabled,
                deterministic_source = %decode_runtime_config.deterministic.source,
                max_batch_tokens = max_batch_tokens.tokens(),
                max_batch_tokens_source = %max_batch_tokens.source(),
                max_prefill_tokens_per_cycle = max_prefill_tokens_per_cycle.tokens(),
                max_prefill_tokens_per_cycle_source = %max_prefill_tokens_per_cycle.source(),
                max_prefill_layers_per_cycle = max_prefill_layers_per_cycle.layers(),
                max_prefill_layers_per_cycle_source = %max_prefill_layers_per_cycle.source(),
                stream_stall_grace_ms = response_delivery_policy
                    .stream_stall_grace_ms(),
                stream_stall_grace_source = %response_delivery_policy
                    .stream_stall_grace_source(),
                resident_prefill_enabled = backend_name == "vulkan"
                    && serving_policy.vulkan_resident_prefill,
                "batching engine enabled; routing streaming and non-streaming real completions through the batching actor"
            );
            let forward = crate::batching_engine::RealDecodeForward::new(
                    runner.clone(),
                    block_manager.clone(),
                    paged_cache.clone(),
                    prefix_cache.clone(),
                    gpu_lock.clone(),
                    loaded_adapter.clone(),
                    serving_policy.dynamic_kv_resize,
                )
                .with_resident_prefill_enabled(
                    backend_name == "vulkan" && serving_policy.vulkan_resident_prefill,
                )
                .with_rowwise_decode(batching_runtime_config.rowwise_decode.enabled);
            crate::batching_engine::BatchingEngineHandle::start_with_actor_runtime_config(
                Arc::new(forward),
                max_decode_batch,
                batching_runtime_config.actor_admission_config(),
                batching_runtime_config.actor_cycle_idle,
                max_batch_tokens,
                max_prefill_tokens_per_cycle,
                max_prefill_layers_per_cycle,
                response_delivery_policy,
            )
        });
        if vram_probe_selector != kiln_memory::vram::VramProbeSelector::None {
            let published = kiln_memory::MemoryGovernor::global().refresh();
            anyhow::ensure!(
                published.total_bytes > 0 && !published.observations.probe_failed,
                "selected-device memory probe lost its safe capacity after startup allocations; refusing to become ready"
            );
        }
        // Wire allocator reclaim only after the actor and GPU coordination lock
        // exist. A periodic monitor must never synchronize or trim underneath a
        // live request; the reclaimer below checks actor activity and takes the
        // exclusive guard without queueing behind inference.
        if serving_policy.allocator_reclaim {
            static GOVERNOR_WIRED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            GOVERNOR_WIRED.get_or_init(|| {
                register_backend_memory_reclaimer(
                    gpu_memory_reclaim_policy,
                    device_kt,
                    gpu_lock.clone(),
                    backend_health.clone(),
                    batching_engine.clone(),
                );
                kiln_memory::MemoryGovernor::global().start_monitor();
            });
        } else {
            tracing::info!(
                serving_profile = %serving_profile.profile(),
                "backend allocator reclaim hooks disabled by serving profile"
            );
        }
        // #24/#26: drive dynamic KV resize from live memory pressure on GPU
        // backends whose KV pools are device-resident (CUDA/ROCm) — that's where
        // inference and a coexisting training run / process actually contend for
        // VRAM. Host-resident pools (CPU/Vulkan) don't, so there's nothing to
        // arbitrate. The autoscaler shrinks KV when VRAM gets tight and grows it
        // back when headroom returns; the resize itself runs on the engine actor
        // at its barrier under exclusive GPU access.
        let kv_autoscaler = if !kv_autoscaler_config.requested {
            crate::kv_autoscaler::KvAutoscalerState::disabled(kv_autoscaler_config)
        } else if !serving_policy.dynamic_kv_resize {
            crate::kv_autoscaler::KvAutoscalerState::unavailable(
                kv_autoscaler_config,
                "serving_profile_stable",
            )
        } else if let Some(engine) = batching_engine.clone() {
            if backend_capabilities.storage.kv_cache_device_memory_pressure {
                crate::kv_autoscaler::spawn(
                    engine,
                    paged_cache.clone(),
                    gpu_allocator_memory_probe_policy,
                    kv_autoscaler_config,
                )
            } else {
                crate::kv_autoscaler::KvAutoscalerState::unavailable(
                    kv_autoscaler_config,
                    "backend_pool_not_device_resident",
                )
            }
        } else {
            crate::kv_autoscaler::KvAutoscalerState::unavailable(
                kv_autoscaler_config,
                "batching_engine_disabled",
            )
        };
        let decode_batcher = if let Some(config) = decode_batcher_config {
            tracing::info!(
                backend = backend_name,
                scope = DirectDecodeRendezvousRuntimeState::SCOPE,
                max_batch = config.max_batch,
                wait_us = config.wait.as_micros() as u64,
                mixed_seq_lens = config.allow_mixed_seq_lens,
                actor_active = batching_engine.is_some(),
                "starting direct streaming greedy decode rendezvous worker"
            );
            match DecodeBatcher::spawn(runner.clone(), paged_cache.clone(), config) {
                Ok(batcher) => {
                    tracing::info!(
                        scope = DirectDecodeRendezvousRuntimeState::SCOPE,
                        worker_active = true,
                        route_available = batching_engine.is_none(),
                        "direct streaming greedy decode rendezvous worker active"
                    );
                    Some(batcher)
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        scope = DirectDecodeRendezvousRuntimeState::SCOPE,
                        worker_active = false,
                        route_available = false,
                        "failed to spawn direct streaming greedy decode rendezvous worker; continuing without that compatibility route"
                    );
                    None
                }
            }
        } else {
            tracing::info!(
                scope = DirectDecodeRendezvousRuntimeState::SCOPE,
                worker_active = false,
                route_available = false,
                "direct streaming greedy decode rendezvous worker disabled"
            );
            None
        };

        let config_hashes = ConfigHashes::from_model_tokenizer(&model_config, &tokenizer, None);
        Ok(Self {
            serving_profile,
            decode_runtime_config,
            batching_runtime_config,
            streaming_prefill_runtime_config,
            prefix_cache_config: prefix_cache_cfg.clone(),
            accelerator_runtime_policy: crate::config::AcceleratorRuntimeConfig::default()
                .resolved_policy(serving_profile),
            speculative_config,
            speculative_runtime_policy,
            operational_runtime: Arc::new(crate::config::OperationalRuntimeConfig::default()),
            checkpoint_read_mib_per_second: None,
            checkpoint_read_applicable: true,
            checkpoint_read_not_applicable_reason: None,
            checkpoint_read_report: None,
            accelerator_weight_upload_mib_per_second: None,
            accelerator_weight_upload_applicable: device_kt.is_gpu(),
            accelerator_weight_upload_not_applicable_reason: device_kt
                .is_cpu()
                .then_some("cpu_device"),
            accelerator_weight_upload_report: None,
            model_config,
            model_path: None,
            base_teacher_identity,
            base_weight_shard_manifest,
            execution_provenance,
            backend: Arc::new(ModelBackend::Real {
                runner,
                rocm_graph_telemetry,
                backend_health,
                block_manager,
                paged_cache,
                prefix_cache,
                batching_engine,
                decode_batcher,
            }),
            tokenizer: Arc::new(tokenizer),
            agent_runs: Arc::new(crate::agent_runs::AgentRunRegistry::new(
                adapter_dir.clone(),
            )),
            adapter_dir,
            active_adapter_name: Arc::new(std::sync::RwLock::new(None)),
            loaded_adapter,
            self_improve_scheduler: Arc::new(std::sync::RwLock::new(None)),
            adapter_load_errors: Arc::new(std::sync::RwLock::new(HashMap::new())),
            adapter_mutation_lock: Arc::new(tokio::sync::Mutex::new(())),
            hf_trl_export_lock: Arc::new(tokio::sync::Mutex::new(())),
            training_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            memory_budget: Arc::new(memory_budget),
            kv_autoscaler,
            gpu_lock,
            training_queue: crate::training_queue::new_shared_queue(),
            training_data_admission_lock: Arc::new(std::sync::Mutex::new(())),
            teacher_registry: teacher_registry_for_real.clone(),
            teacher_credentials: Arc::new(crate::config::TeachersConfig::default()),
            vram_info,
            vram_capacity_resolution,
            vram_probe_selector,
            memory_config: memory_cfg.clone(),
            training_runtime: kiln_train::TrainingRuntimeContext::new_for_device(
                device_kt,
                vram_info,
                gradient_checkpoint_policy,
            )
            .with_checkpoint_boundary_policy(checkpoint_boundary_policy)
            .with_streaming_prefill_policy(streaming_prefill_runtime_config.execution_policy()),
            inference_device: device_kt,
            model_weight_device,
            shutdown: crate::training_queue::new_shutdown_flag(),
            request_timeout: std::time::Duration::from_secs(request_timeout_secs),
            http_send_buffer_bytes: None,
            http_send_buffer_preflight_actual_bytes: None,
            http_send_buffer_preflight_effective_bytes: None,
            eval_mode: false,
            debug_model_state: false,
            default_thinking_enabled: None,
            default_thinking_budget_tokens: None,
            default_thinking_budget_ms: None,
            model_defaults_profile: crate::config::ModelDefaultsProfile::qwen3_5_4b(),
            fold_reasoning_into_content: false,
            chat_performance_metadata: false,
            chat_config_hash_metadata: false,
            config_hashes,
            slow_request_warn_threshold: None,
            metrics: Arc::new(Metrics::new()),
            started_at: std::time::Instant::now(),
            inference_prewarm_complete: Arc::new(AtomicBool::new(
                !backend_capabilities
                    .startup
                    .require_inference_prewarm_for_health,
            )),
            checkpoint_interval: None,
            training_webhook_url: None,
            max_queued_training_jobs: 32,
            max_tracked_jobs: 1024,
            tracked_job_ttl: std::time::Duration::from_secs(604_800),
            adapter_max_disk_bytes: Some(100 * 1024u64.pow(3)),
            composed_cache_max_bytes: Some(10 * 1024u64.pow(3)),
            composed_cache_max_entries: Some(64),
            served_model_id,
            decode_stats: Arc::new(std::sync::Mutex::new(DecodeStatsRing::new(4096))),
            recent_requests: Arc::new(std::sync::Mutex::new(RecentRequestsRing::new(
                RECENT_REQUESTS_CAPACITY,
            ))),
            request_log: None,
            request_log_max_capture_bytes: crate::request_log::RequestLogConfig::default()
                .max_capture_bytes,
            deterministic_cache_generations: Arc::new(std::sync::Mutex::new(
                DeterministicCacheGenerations::default(),
            )),
            completion_cache: Arc::new(std::sync::Mutex::new(DeterministicCompletionCache::new(
                DETERMINISTIC_COMPLETION_CACHE_CAPACITY,
            ))),
            chat_request_cache: Arc::new(std::sync::Mutex::new(
                DeterministicChatRequestCache::new(DETERMINISTIC_CHAT_REQUEST_CACHE_CAPACITY),
            )),
            chat_choices_cache: Arc::new(std::sync::Mutex::new(
                DeterministicChatChoicesCache::new(DETERMINISTIC_CHAT_CHOICES_CACHE_CAPACITY),
            )),
            batch_cache: Arc::new(std::sync::Mutex::new(DeterministicBatchCache::new(
                DETERMINISTIC_BATCH_CACHE_CAPACITY,
            ))),
            rendered_prompt_cache: Arc::new(std::sync::Mutex::new(RenderedPromptCache::new(
                RENDERED_PROMPT_CACHE_CAPACITY,
            ))),
            prompt_token_cache: Arc::new(std::sync::Mutex::new(PromptTokenCache::new(
                PROMPT_TOKEN_CACHE_CAPACITY,
            ))),
            eval_queue: crate::eval::new_shared_eval_queue(),
            eval_jobs: Arc::new(std::sync::RwLock::new(HashMap::new())),
            suite_registry: None,
            dataset_registry: None,
            judgment_store: None,
            max_queued_eval_jobs: 32,
            max_tracked_eval_jobs: 1024,
            eval_webhook_url: None,
        })
    }
}

fn log_backend_training_acceleration_profile(policy: TrainingAccelerationProfilePolicy) {
    match policy.log_message {
        TrainingAccelerationProfileLogMessage::None => {}
        TrainingAccelerationProfileLogMessage::Vulkan => {
            tracing::info!(
                linear = policy.linear,
                sdpa = policy.sdpa,
                rmsnorm_inference = policy.rmsnorm_inference,
                rmsnorm_training = policy.rmsnorm_training,
                flce_provider = policy.flce_provider,
                resident_activation = policy.resident_activation,
                sgd_step_on_device = policy.sgd_step_on_device,
                "Vulkan training acceleration profile"
            );
        }
    }
}

fn linear_attention_state_bytes(
    config: &ModelConfig,
    policy: InferenceRecurrentStatePolicy,
) -> u64 {
    let num_linear_layers = config
        .num_layers
        .saturating_sub(config.num_full_attention_layers) as u64;
    // Mirrors `LinearAttentionState` inference recurrent-state allocation
    // through the backend-owned capability policy, so prefix-cache sizing does
    // not keep a separate server-side backend/env table.
    let recurrent_dtype = match config.dtype {
        kiln_core::config::DType::BF16 if policy.supports_dtype(DType::BF16) => DType::BF16,
        kiln_core::config::DType::FP16 if policy.supports_dtype(DType::F16) => DType::F16,
        _ => DType::F32,
    };
    let recurrent_dtype_bytes = match recurrent_dtype {
        DType::BF16 | DType::F16 => 2,
        _ => 4,
    };
    let recurrent_elems = (config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim) as u64;
    let conv_elems =
        (config.linear_qkv_dim() * config.linear_conv_kernel_dim.saturating_sub(1)) as u64;
    num_linear_layers.saturating_mul(
        recurrent_elems
            .saturating_mul(recurrent_dtype_bytes)
            .saturating_add(conv_elems.saturating_mul(4)),
    )
}

fn default_prefix_cache_max_entries(total_vram_bytes: u64, state_bytes_per_entry: u64) -> usize {
    if state_bytes_per_entry == 0 {
        return MIN_PREFIX_CACHE_MAX_ENTRIES;
    }
    prefix_cache_entries_for_state_budget(
        default_prefix_cache_state_budget(total_vram_bytes),
        state_bytes_per_entry,
    )
}

fn default_prefix_cache_state_budget(total_bytes: u64) -> u64 {
    if total_bytes == 0 {
        MIN_PREFIX_CACHE_STATE_BYTES
    } else {
        (total_bytes / PREFIX_CACHE_STATE_FRACTION_DIVISOR)
            .clamp(MIN_PREFIX_CACHE_STATE_BYTES, MAX_PREFIX_CACHE_STATE_BYTES)
    }
}

fn prefix_cache_entries_for_state_budget(
    state_budget_bytes: u64,
    state_bytes_per_entry: u64,
) -> usize {
    if state_bytes_per_entry == 0 {
        return MIN_PREFIX_CACHE_MAX_ENTRIES;
    }
    usize::try_from(state_budget_bytes / state_bytes_per_entry)
        .unwrap_or(usize::MAX)
        .max(MIN_PREFIX_CACHE_MAX_ENTRIES)
}

/// Return the independently governed host-backed allocation tier used by
/// Vulkan's current CPU-resident paged KV and recurrent-state storage.
///
/// A missing Vulkan tier is a failed safety proof, not permission to size from
/// the primary DRM VRAM aperture. Other backends do not use this second ceiling.
fn host_backed_free_bytes_for_device(
    device: kiln_tensor::Device,
    snapshot: Option<kiln_memory::MemorySnapshot>,
) -> Option<u64> {
    match device {
        kiln_tensor::Device::Vulkan(_) => Some(
            snapshot
                .and_then(|snapshot| snapshot.observations.host_backed)
                .map_or(0, |tier| tier.free_bytes),
        ),
        _ => None,
    }
}

fn prefix_cache_host_reserve_bytes(
    host_backed_free_bytes: Option<u64>,
    state_bytes_per_entry: u64,
    enabled: bool,
    configured_max_entries: Option<usize>,
) -> anyhow::Result<u64> {
    let Some(host_backed_free_bytes) = host_backed_free_bytes else {
        return Ok(0);
    };
    if !enabled || state_bytes_per_entry == 0 {
        return Ok(0);
    }

    let state_budget = if let Some(max_entries) = configured_max_entries {
        let max_entries = max_entries.max(MIN_PREFIX_CACHE_MAX_ENTRIES);
        let max_entries = u64::try_from(max_entries).map_err(|_| {
            anyhow::anyhow!(
                "prefix_cache.max_entries={max_entries} cannot be represented as a 64-bit memory budget"
            )
        })?;
        state_bytes_per_entry
            .checked_mul(max_entries)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "prefix cache state budget overflow: state_bytes_per_entry={state_bytes_per_entry}, prefix_cache.max_entries={max_entries}"
                )
            })?
    } else {
        default_prefix_cache_state_budget(host_backed_free_bytes)
            .min(host_backed_free_bytes)
            .max(state_bytes_per_entry)
    };

    anyhow::ensure!(
        state_budget <= host_backed_free_bytes,
        "Vulkan prefix cache state requires {state_budget} host-backed bytes, but the safe GTT/host tier has only {host_backed_free_bytes} bytes free. Lower prefix_cache.max_entries, disable prefix_cache.enabled, free host memory, or increase the host/cgroup memory limit"
    );
    Ok(state_budget)
}

fn host_backed_kv_budget_for_fraction(
    device: kiln_tensor::Device,
    snapshot: Option<kiln_memory::MemorySnapshot>,
    prefix_cache_reserve_bytes: u64,
    fraction: f64,
) -> Option<u64> {
    host_backed_free_bytes_for_device(device, snapshot).map(|free_bytes| {
        let after_prefix_cache = free_bytes.saturating_sub(prefix_cache_reserve_bytes);
        ((after_prefix_cache as f64) * fraction.clamp(0.0, 1.0)) as u64
    })
}

fn minimum_optional_budget(lhs: Option<u64>, rhs: Option<u64>) -> Option<u64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Estimate model weight memory in bytes from config.
///
/// Uses a rough formula: total parameters * dtype_bytes.
/// For Qwen3.5-4B in BF16: ~4B params * 2 bytes ≈ 8GB.
fn estimate_model_memory_bytes(config: &ModelConfig) -> u64 {
    let dtype_bytes: u64 = match config.dtype {
        kiln_core::config::DType::BF16 | kiln_core::config::DType::FP16 => 2,
        kiln_core::config::DType::FP32 => 4,
    };

    // Rough parameter count estimate for a transformer:
    // Embedding: vocab_size * hidden_size
    // Per layer: ~8 * hidden_size^2 + 3 * hidden_size * intermediate_size (approximate)
    // LM head: vocab_size * hidden_size (often tied with embedding)
    let embedding_params = config.vocab_size as u64 * config.hidden_size as u64;
    let per_layer_params = 8 * (config.hidden_size as u64 * config.hidden_size as u64)
        + 3 * (config.hidden_size as u64 * config.intermediate_size as u64);
    let total_params = embedding_params + per_layer_params * config.num_layers as u64;

    total_params * dtype_bytes
}

fn auto_num_blocks_for_fraction(
    total_vram: u64,
    sizing_residency_bytes: u64,
    bytes_per_block: u64,
    fraction: f64,
    max_position_embeddings: usize,
    block_size: usize,
    kv_auto_block_policy: KvCacheAutoBlockPolicy,
) -> usize {
    if total_vram > 0 && bytes_per_block > 0 {
        let available_for_kv =
            ((total_vram.saturating_sub(sizing_residency_bytes)) as f64 * fraction) as u64;
        let raw_auto_blocks = (available_for_kv / bytes_per_block) as usize;
        cap_auto_num_blocks(
            raw_auto_blocks,
            max_position_embeddings,
            block_size,
            kv_auto_block_policy,
            total_vram,
        )
    } else {
        let raw_auto_blocks = max_position_embeddings.div_ceil(block_size).max(256);
        cap_auto_num_blocks(
            raw_auto_blocks,
            max_position_embeddings,
            block_size,
            kv_auto_block_policy,
            total_vram,
        )
    }
}

fn cap_auto_num_blocks(
    raw_blocks: usize,
    max_position_embeddings: usize,
    block_size: usize,
    kv_auto_block_policy: KvCacheAutoBlockPolicy,
    total_vram_bytes: u64,
) -> usize {
    // On UMA backends, an eagerly-zeroed KV cache larger than the
    // model context can dominate memory pressure on the rest of the system,
    // so the backend-owned policy keeps the historical "≤ one full context,
    // further capped by detected memory tier" behavior where applicable.
    //
    // ROCm's HIP allocator can abort the process on later long-prefill scratch
    // OOMs instead of returning a catchable allocation error. Its capability
    // policy keeps the default KV pool to one Qwen3.5-class full-context pool
    // so long prefill has workspace headroom; explicit `memory.num_blocks` can
    // request a larger pool, but the live-budget validator remains authoritative.
    //
    // On CUDA / CPU, memory-aware sizing already drove `raw_blocks` from the
    // available VRAM × `inference_memory_fraction` budget. Capping again at
    // one model-context-worth of blocks (≈16K for Qwen3.5-4B's 256K window)
    // bottlenecks concurrent serving: 4 in-flight 25K-token prompts +
    // generation already exhaust 6.5K blocks each, leaving the auto cap
    // routinely OOM-borderline under realistic load even on a 48 GiB A40.
    // Trust the memory-aware ceiling here; users who want a stricter cap can
    // still set `KILN_NUM_BLOCKS` or `memory.num_blocks` explicitly.
    let runtime_cap_blocks = kv_auto_block_policy.runtime_cap_blocks(
        max_position_embeddings,
        block_size,
        MIN_AUTO_KV_BLOCKS,
        total_vram_bytes,
    );

    // `raw_blocks` is a capacity result, not a hint. Raising it to the
    // historical preferred minimum can turn a measured sub-minimum budget into
    // an over-budget allocation, which is especially dangerous on ROCm/UMA.
    raw_blocks.min(runtime_cap_blocks)
}

fn effective_live_kv_budget_bytes(
    governor_available_bytes: u64,
    allocator_available_bytes: Option<u64>,
) -> u64 {
    allocator_available_bytes.map_or(governor_available_bytes, |allocator| {
        governor_available_bytes.min(allocator)
    })
}

fn cap_kv_blocks_to_live_budget(
    proposed_blocks: usize,
    bytes_per_block: u64,
    governor_available_bytes: u64,
    allocator_available_bytes: Option<u64>,
) -> (usize, usize) {
    if bytes_per_block == 0 {
        return (proposed_blocks, usize::MAX);
    }
    let live_budget =
        effective_live_kv_budget_bytes(governor_available_bytes, allocator_available_bytes);
    let max_blocks = (live_budget / bytes_per_block) as usize;
    (proposed_blocks.min(max_blocks), max_blocks)
}

fn validate_kv_allocation_against_live_allocator(
    device: &kiln_tensor::Device,
    num_blocks: usize,
    bytes_per_block: u64,
    gpu_memory_budget_policy: GpuMemoryBudgetPolicy,
    gpu_allocator_memory_probe_policy: GpuAllocatorMemoryProbePolicy,
    kv_auto_block_policy: KvCacheAutoBlockPolicy,
    governor: &kiln_memory::MemoryGovernor,
    host_prefix_cache_reserve_bytes: u64,
) -> anyhow::Result<()> {
    if !gpu_memory_budget_policy.cap_kv_blocks_by_live_budget || bytes_per_block == 0 {
        return Ok(());
    }
    let governor_observation = governor.cached_observation();
    let governor_budget = governor_observation.available_bytes;
    let allocator_budget = crate::device_memory::allocator_safe_available_bytes(
        gpu_allocator_memory_probe_policy,
        governor,
        device,
    );
    let host_backed_budget = host_backed_kv_budget_for_fraction(
        *device,
        Some(governor_observation.snapshot),
        host_prefix_cache_reserve_bytes,
        1.0,
    );
    let residency_budget = minimum_optional_budget(allocator_budget, host_backed_budget);
    let live_budget = effective_live_kv_budget_bytes(governor_budget, residency_budget);
    validate_kv_allocation_against_live_budget(
        num_blocks,
        bytes_per_block,
        live_budget,
        allocator_budget.is_some(),
        host_backed_budget.is_some(),
        kv_auto_block_policy,
    )
}

fn validate_kv_allocation_against_live_budget(
    num_blocks: usize,
    bytes_per_block: u64,
    live_budget: u64,
    allocator_probe_available: bool,
    host_backed_budget_available: bool,
    kv_auto_block_policy: KvCacheAutoBlockPolicy,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        num_blocks > 0,
        "paged KV cache live memory budget cannot fit even one block; reduce model residency or memory.floor_gb, free accelerator and host memory, or choose a device with more available memory"
    );
    let requested = (num_blocks as u64).saturating_mul(bytes_per_block);
    if requested <= live_budget {
        return Ok(());
    }
    let max_blocks = (live_budget / bytes_per_block) as usize;
    let budget_source = match (allocator_probe_available, host_backed_budget_available) {
        (true, true) => "the stricter governor/allocator/host-backed budget",
        (true, false) => "the stricter governor/allocator budget",
        (false, true) => "the stricter governor/host-backed budget",
        (false, false) => "the memory governor budget",
    };
    let policy_floor_note = if kv_auto_block_policy.allow_min_blocks_below_live_budget
        && max_blocks < MIN_AUTO_KV_BLOCKS
    {
        " The backend's preferred minimum block policy is intentionally not applied above live memory."
    } else {
        ""
    };
    anyhow::bail!(
        "paged KV cache allocation request exceeds live accelerator/host residency memory: \
         requested num_blocks={num_blocks} (~{requested_gb:.2} GiB), but {budget_source} \
         fits at most num_blocks={max_blocks} (~{budget_gb:.2} GiB).{policy_floor_note} \
         Lower memory.num_blocks or memory.inference_memory_fraction, reduce \
         memory.floor_gb, or free accelerator and host memory.",
        requested_gb = requested as f64 / (1024.0 * 1024.0 * 1024.0),
        budget_gb = live_budget as f64 / (1024.0 * 1024.0 * 1024.0),
    )
}

fn runtime_used_vram_for_policy(
    policy: GpuMemoryBudgetPolicy,
    selector: kiln_memory::vram::VramProbeSelector,
) -> Option<kiln_memory::vram::GpuMemoryUsedInfo> {
    // The live used-memory probe is OS-level (nvidia-smi / AMD+Intel DRM sysfs /
    // unified-APU MemAvailable, see `kiln_memory::vram::current_memory_snapshot`),
    // so it is BACKEND-AGNOSTIC — it works for CUDA, ROCm, Vulkan, and Metal,
    // not just CUDA. Previously this was `#[cfg(feature = "cuda")]`-gated, so on
    // every other backend the KV-cache sizer fell back to a STATIC model-size
    // estimate that is blind to live residency and to coexisting GPU workloads.
    // Wiring it for all GPU backends makes the sizer mindful of the actual VRAM
    // on whatever device the user is running. CPU has no device memory to probe.
    if !policy.use_live_memory_snapshot {
        return None;
    }
    let snap = kiln_memory::vram::current_memory_snapshot_for(selector);
    (snap.used_bytes > 0).then_some(kiln_memory::vram::GpuMemoryUsedInfo {
        used_bytes: snap.used_bytes,
        source: snap.source,
    })
}

/// Successful auto-sizer outcome. Carries the live cache plus the metadata
/// needed to log the final decision and update the GPU memory budget.
struct AutoSizeSuccess {
    cache: PagedKvCacheKt,
    num_blocks: usize,
    fraction: f64,
    /// `(fraction, num_blocks, error)` for each attempt that failed before
    /// the eventual success. Empty when the configured fraction worked.
    attempted_failures: Vec<(f64, usize, String)>,
}

/// Failure outcome — every fraction in the retry sequence OOMed.
struct AutoSizeFailure {
    /// `(fraction, num_blocks, error)` for every attempt in order.
    attempts: Vec<(f64, usize, String)>,
}

fn register_backend_memory_reclaimer(
    policy: GpuMemoryReclaimPolicy,
    device: kiln_tensor::Device,
    gpu_lock: GpuCoordinationLock,
    backend_health: BackendHealthHandle,
    batching_engine: Option<crate::batching_engine::BatchingEngineHandle>,
) {
    let _ = (&device, &gpu_lock, &backend_health, &batching_engine);
    match policy.reclaimer {
        GpuMemoryReclaimer::None => {}
        GpuMemoryReclaimer::RocmTrimPool => {
            #[cfg(feature = "rocm")]
            {
                if let kiln_tensor::Device::Rocm(idx) = device {
                    kiln_memory::MemoryGovernor::global().register_reclaimer(move |target| {
                        let (observed_reserved, observed_used) =
                            match kiln_tensor::rocm_pool_stats(idx) {
                                Ok(stats) => stats,
                                Err(error) => {
                                    tracing::warn!(
                                        %error,
                                        reason = "pool_statistics_unavailable",
                                        target,
                                        "ROCm pool reclaim skipped"
                                    );
                                    return 0;
                                }
                            };
                        let (active_decode, active_prefill) = batching_engine
                            .as_ref()
                            .map(|engine| {
                                let snapshot = engine.cached_snapshot();
                                (snapshot.active_decode, snapshot.active_prefill)
                            })
                            .unwrap_or_default();
                        if pool_trim_min_keep_if_idle(
                            observed_reserved,
                            observed_used,
                            target,
                            active_decode,
                            active_prefill,
                        )
                        .is_none()
                        {
                            if active_decode > 0 || active_prefill > 0 {
                                tracing::debug!(
                                    reason = "active_requests",
                                    target,
                                    active_decode,
                                    active_prefill,
                                    reserved = observed_reserved,
                                    used = observed_used,
                                    "ROCm pool reclaim deferred"
                                );
                            } else {
                                tracing::debug!(
                                    reason = "no_releasable_bytes",
                                    target,
                                    reserved = observed_reserved,
                                    used = observed_used,
                                    spare = observed_reserved.saturating_sub(observed_used),
                                    "ROCm pool reclaim skipped"
                                );
                            }
                            return 0;
                        }

                        if let Err(error) = backend_health.ensure_healthy() {
                            tracing::warn!(
                                %error,
                                reason = "backend_unhealthy",
                                target,
                                "ROCm pool reclaim skipped"
                            );
                            return 0;
                        }
                        let coordination_started = std::time::Instant::now();
                        let Ok(_gpu_guard) = gpu_lock.clone().try_write_owned() else {
                            tracing::debug!(
                                reason = "gpu_coordination_busy",
                                target,
                                reserved = observed_reserved,
                                used = observed_used,
                                coordination_acquire_ms =
                                    coordination_started.elapsed().as_secs_f64() * 1000.0,
                                "ROCm pool reclaim deferred"
                            );
                            return 0;
                        };
                        if let Err(error) = backend_health.ensure_healthy() {
                            tracing::warn!(
                                %error,
                                reason = "backend_became_unhealthy",
                                target,
                                "ROCm pool reclaim skipped after coordination"
                            );
                            return 0;
                        }
                        let coordination_acquire_ms =
                            coordination_started.elapsed().as_secs_f64() * 1000.0;

                        // Activity can change between the preflight and the
                        // lock acquisition. Refuse the maintenance operation
                        // if the actor admitted work in that window.
                        let (active_decode, active_prefill) = batching_engine
                            .as_ref()
                            .map(|engine| {
                                let snapshot = engine.cached_snapshot();
                                (snapshot.active_decode, snapshot.active_prefill)
                            })
                            .unwrap_or_default();

                        // Re-read after exclusive acquisition. Allocations may
                        // have changed between the cheap preflight and this
                        // mutation boundary.
                        let (reserved, used) = match kiln_tensor::rocm_pool_stats(idx) {
                            Ok(stats) => stats,
                            Err(error) => {
                                tracing::warn!(
                                    %error,
                                    reason = "locked_pool_statistics_unavailable",
                                    target,
                                    coordination_acquire_ms,
                                    "ROCm pool reclaim skipped"
                                );
                                return 0;
                            }
                        };
                        let Some(min_keep) = pool_trim_min_keep_if_idle(
                            reserved,
                            used,
                            target,
                            active_decode,
                            active_prefill,
                        ) else {
                            if active_decode > 0 || active_prefill > 0 {
                                tracing::debug!(
                                    reason = "active_requests_after_coordination",
                                    target,
                                    active_decode,
                                    active_prefill,
                                    coordination_acquire_ms,
                                    "ROCm pool reclaim deferred"
                                );
                            } else {
                                tracing::debug!(
                                    reason = "no_releasable_bytes_after_coordination",
                                    target,
                                    reserved,
                                    used,
                                    spare = reserved.saturating_sub(used),
                                    coordination_acquire_ms,
                                    "ROCm pool reclaim skipped"
                                );
                            }
                            return 0;
                        };
                        let started = std::time::Instant::now();
                        match kiln_tensor::rocm_trim_pool(idx, min_keep) {
                            Ok(reclaimed) => {
                                tracing::info!(
                                    event = "gpu_memory_operation",
                                    operation = "trim",
                                    reason = "memory_governor",
                                    outcome = if reclaimed > 0 {
                                        "reclaimed"
                                    } else {
                                        "zero_yield"
                                    },
                                    target,
                                    target_bytes = target,
                                    actual_bytes = reclaimed,
                                    reserved_before = reserved,
                                    used_before = used,
                                    requested_min_keep = min_keep,
                                    reclaimed,
                                    coordination_acquire_ms,
                                    wait_ms = coordination_acquire_ms,
                                    duration_ms = started.elapsed().as_secs_f64() * 1000.0,
                                    "ROCm pool reclaim completed"
                                );
                                reclaimed
                            }
                            Err(error) => {
                                tracing::warn!(
                                    event = "gpu_memory_operation",
                                    operation = "trim",
                                    reason = "memory_governor",
                                    outcome = "failed",
                                    %error,
                                    target,
                                    target_bytes = target,
                                    actual_bytes = 0,
                                    reserved_before = reserved,
                                    used_before = used,
                                    requested_min_keep = min_keep,
                                    coordination_acquire_ms,
                                    wait_ms = coordination_acquire_ms,
                                    duration_ms = started.elapsed().as_secs_f64() * 1000.0,
                                    "ROCm pool reclaim failed"
                                );
                                0
                            }
                        }
                    });
                }
            }
        }
        GpuMemoryReclaimer::VulkanTrimPool =>
        {
            #[cfg(feature = "vulkan")]
            if matches!(device, kiln_tensor::Device::Vulkan(_)) {
                kiln_memory::MemoryGovernor::global().register_reclaimer(move |target| {
                    let stats = kiln_model::vulkan_buffer_pool_stats().unwrap_or_default();
                    let (active_decode, active_prefill) = batching_engine
                        .as_ref()
                        .map(|engine| {
                            let snapshot = engine.cached_snapshot();
                            (snapshot.active_decode, snapshot.active_prefill)
                        })
                        .unwrap_or_default();
                    if stats.free_bytes == 0 || active_decode > 0 || active_prefill > 0 {
                        tracing::debug!(
                            reason = if stats.free_bytes == 0 {
                                "no_releasable_bytes"
                            } else {
                                "active_requests"
                            },
                            target,
                            active_decode,
                            active_prefill,
                            retained_bytes = stats.total_bytes,
                            free_bytes = stats.free_bytes,
                            "Vulkan buffer-pool reclaim deferred"
                        );
                        return 0;
                    }
                    if let Err(error) = backend_health.ensure_healthy() {
                        tracing::warn!(
                            %error,
                            reason = "backend_unhealthy",
                            target,
                            "Vulkan buffer-pool reclaim skipped"
                        );
                        return 0;
                    }
                    let coordination_started = std::time::Instant::now();
                    let Ok(_gpu_guard) = gpu_lock.clone().try_write_owned() else {
                        tracing::debug!(
                            reason = "gpu_coordination_busy",
                            target,
                            "Vulkan buffer-pool reclaim deferred"
                        );
                        return 0;
                    };
                    if let Err(error) = backend_health.ensure_healthy() {
                        tracing::warn!(
                            %error,
                            reason = "backend_became_unhealthy",
                            target,
                            "Vulkan buffer-pool reclaim skipped after coordination"
                        );
                        return 0;
                    }
                    let (active_decode, active_prefill) = batching_engine
                        .as_ref()
                        .map(|engine| {
                            let snapshot = engine.cached_snapshot();
                            (snapshot.active_decode, snapshot.active_prefill)
                        })
                        .unwrap_or_default();
                    let before = kiln_model::vulkan_buffer_pool_stats().unwrap_or_default();
                    if before.free_bytes == 0 || active_decode > 0 || active_prefill > 0 {
                        return 0;
                    }

                    let requested = target.min(before.free_bytes);
                    let started = std::time::Instant::now();
                    let reclaimed = kiln_model::trim_vulkan_buffer_pool(requested);
                    let coordination_acquire_ms =
                        coordination_started.elapsed().as_secs_f64() * 1000.0;
                    tracing::info!(
                        event = "gpu_memory_operation",
                        operation = "trim",
                        reason = "memory_governor",
                        outcome = if reclaimed > 0 {
                            "reclaimed"
                        } else {
                            "zero_yield"
                        },
                        target,
                        target_bytes = requested,
                        actual_bytes = reclaimed,
                        retained_before = before.total_bytes,
                        free_before = before.free_bytes,
                        coordination_acquire_ms,
                        wait_ms = coordination_acquire_ms,
                        duration_ms = started.elapsed().as_secs_f64() * 1000.0,
                        "Vulkan buffer-pool reclaim completed"
                    );
                    reclaimed
                });
            }
        }
        GpuMemoryReclaimer::CudaTrimPool => {
            #[cfg(feature = "cuda")]
            if let kiln_tensor::Device::Cuda(idx) = device {
                // (#32) cudarc allocates from the stream-ordered mempool
                // (cuMemAllocAsync). Its RELEASE_THRESHOLD defaults to 0, so
                // the pool returns every freed page to the OS at each sync:
                // perf-churny and it leaves cuda_trim_pool nothing to reclaim.
                // Raise the threshold so the pool hoards freed pages for fast
                // reuse, turning this reclaimer into the pressure release valve.
                let _ = kiln_tensor::cuda_set_pool_release_threshold(idx, u64::MAX);
                kiln_memory::MemoryGovernor::global().register_reclaimer(move |_target| {
                    // Measure bytes actually returned to the OS via the live
                    // free-VRAM delta (the driver doesn't report trim yield).
                    let before = kiln_tensor::cuda_mem_get_info(idx)
                        .map(|(f, _)| f)
                        .unwrap_or(0);
                    let _ = kiln_tensor::cuda_trim_pool(idx, 0);
                    let after = kiln_tensor::cuda_mem_get_info(idx)
                        .map(|(f, _)| f)
                        .unwrap_or(0);
                    after.saturating_sub(before) as u64
                });
            }
        }
        GpuMemoryReclaimer::LoggedNoop { log_message } => {
            kiln_memory::MemoryGovernor::global().register_reclaimer(move |_target| {
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| tracing::info!("{}", log_message));
                0
            });
        }
    }
}

#[cfg(any(feature = "rocm", test))]
fn pool_trim_min_keep(reserved: u64, used: u64, target: u64) -> Option<usize> {
    let releasable = reserved.saturating_sub(used).min(target);
    if releasable == 0 {
        return None;
    }
    usize::try_from(reserved.saturating_sub(releasable)).ok()
}

#[cfg(any(feature = "rocm", test))]
fn pool_trim_min_keep_if_idle(
    reserved: u64,
    used: u64,
    target: u64,
    active_decode: usize,
    active_prefill: usize,
) -> Option<usize> {
    if active_decode > 0 || active_prefill > 0 {
        return None;
    }
    pool_trim_min_keep(reserved, used, target)
}

/// Auto-size the KV cache by trying `configured_fraction` first and then each
/// entry of `fallback_fractions` that is strictly less than the configured
/// value, in order. Returns the first attempt that allocates successfully, or
/// the full attempt history on failure.
///
/// `compute_blocks` maps a fraction to the number of blocks the auto-sizer
/// would request for that budget. `try_allocate` actually attempts the
/// allocation; returning `Err` means OOM (or any other failure) and the loop
/// will move to the next smaller fraction.
///
/// Pure logic — no GPU, no tensors, no logging. Tested directly with mock
/// allocators that return OOM until the fraction drops below a threshold.
fn auto_size_with_retry<C, A>(
    configured_fraction: f64,
    fallback_fractions: &[f64],
    compute_blocks: &C,
    mut try_allocate: A,
) -> Result<AutoSizeSuccess, AutoSizeFailure>
where
    C: Fn(f64) -> usize,
    A: FnMut(usize) -> Result<PagedKvCacheKt, String>,
{
    // Build the ordered fraction sequence: configured first, then each
    // fallback that is strictly smaller (avoids retrying the same value or
    // accidentally retrying *higher* than what the user asked for).
    let mut fractions: Vec<f64> = Vec::with_capacity(1 + fallback_fractions.len());
    fractions.push(configured_fraction);
    for &f in fallback_fractions {
        if f < configured_fraction - 1e-9 {
            fractions.push(f);
        }
    }

    let mut attempts: Vec<(f64, usize, String)> = Vec::with_capacity(fractions.len());
    for fraction in fractions {
        let num_blocks = compute_blocks(fraction);
        match try_allocate(num_blocks) {
            Ok(cache) => {
                return Ok(AutoSizeSuccess {
                    cache,
                    num_blocks,
                    fraction,
                    attempted_failures: attempts,
                });
            }
            Err(err) => {
                attempts.push((fraction, num_blocks, err));
            }
        }
    }

    Err(AutoSizeFailure { attempts })
}

/// Compute a conservative `KILN_NUM_BLOCKS=N` suggestion the user can paste
/// directly. We aim for ~30% of remaining VRAM after model weights — well
/// below the smallest fallback fraction we just tried — so the suggestion has
/// enough headroom to start cleanly even on the GPU/driver combo that just
/// OOM'd at our retry floor.
fn suggested_emergency_num_blocks(
    total_vram: u64,
    estimated_model_bytes: u64,
    bytes_per_block: u64,
    block_size: usize,
    max_position_embeddings: usize,
    kv_auto_block_policy: KvCacheAutoBlockPolicy,
) -> usize {
    if total_vram == 0 || bytes_per_block == 0 {
        // No VRAM signal — fall back to one model context worth of blocks.
        return max_position_embeddings
            .div_ceil(block_size)
            .max(MIN_AUTO_KV_BLOCKS);
    }
    let conservative_fraction = 0.30_f64;
    let available_for_kv =
        ((total_vram.saturating_sub(estimated_model_bytes)) as f64 * conservative_fraction) as u64;
    let raw = (available_for_kv / bytes_per_block) as usize;
    cap_auto_num_blocks(
        raw,
        max_position_embeddings,
        block_size,
        kv_auto_block_policy,
        total_vram,
    )
}

/// Render a multi-line error message that names the exact remediation flags
/// to set, instead of dumping the raw CUDA OOM. We include:
///   - what we tried (fractions + blocks counts)
///   - the underlying error from the deepest attempt
///   - a concrete `KILN_NUM_BLOCKS=N` value
///   - a concrete `inference_memory_fraction=X` value (the lowest we tried,
///     halved further to stay safely below the OOM floor)
///   - the effective GPU memory total + source so users can sanity-check
fn format_oom_remediation_message(
    failure: &AutoSizeFailure,
    total_vram: u64,
    estimated_model_bytes: u64,
    bytes_per_block: u64,
    suggested_blocks: usize,
    configured_fraction: f64,
    vram_source: kiln_memory::vram::VramSource,
) -> String {
    let mut buf = String::new();
    buf.push_str(
        "Auto-sizer could not fit any KV cache budget on this GPU. \
         Every inference_memory_fraction we tried OOM'd during paged KV cache allocation.\n",
    );
    buf.push_str("\nAttempts (in order, all failed):\n");
    for (fraction, num_blocks, err) in &failure.attempts {
        buf.push_str(&format!(
            "  - inference_memory_fraction={:.2} -> num_blocks={}: {}\n",
            fraction,
            num_blocks,
            // Show the error compactly — most CUDA OOMs are one or two lines.
            err.lines().next().unwrap_or("<no error message>")
        ));
    }
    let vram_gb = total_vram as f64 / (1024.0 * 1024.0 * 1024.0);
    let model_gb = estimated_model_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let kv_gb = (suggested_blocks as u64 * bytes_per_block) as f64 / (1024.0 * 1024.0 * 1024.0);
    buf.push_str(&format!(
        "\nGPU memory budget: {:.1} GiB effective total (source: {}), \
         estimated model weights: {:.1} GiB.\n",
        vram_gb, vram_source, model_gb
    ));
    let suggested_fraction = (failure
        .attempts
        .last()
        .map(|(f, _, _)| *f)
        .unwrap_or(configured_fraction)
        / 2.0)
        .max(0.10);
    buf.push_str(&format!(
        "\nRecommended remediation — set ONE of the following and restart:\n  \
         (a) KILN_NUM_BLOCKS={}        # ~{:.1} GiB KV cache, conservative; or in kiln.toml: [memory] num_blocks = {}\n  \
         (b) KILN_INFERENCE_MEMORY_FRACTION={:.2}   # equivalent fraction-based knob; or in kiln.toml: [memory] inference_memory_fraction = {:.2}\n",
        suggested_blocks, kv_gb, suggested_blocks,
        suggested_fraction, suggested_fraction,
    ));
    buf.push_str(&format!(
        "\nFor reference, the configured inference_memory_fraction was {:.2}. \
         Option (a) is preferred — it bypasses the auto-sizer entirely and is what \
         #685 documented as the working workaround on A40/A6000 + Qwen3.5-4B BF16.\n",
        configured_fraction
    ));
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_prefix_cache_is_correctness_quarantined() {
        assert!(effective_prefix_cache_enabled(
            true,
            &kiln_tensor::Device::Cpu
        ));
        assert!(!effective_prefix_cache_enabled(
            true,
            &kiln_tensor::Device::Vulkan(0)
        ));
        assert!(!effective_prefix_cache_enabled(
            false,
            &kiln_tensor::Device::Cpu
        ));

        let stats = RealPrefixCache::disabled(16).stats();
        assert_eq!(stats.max_blocks, 0);
        assert_eq!(stats.max_entries, 0);
        assert_eq!(stats.max_state_bytes, 0);
    }

    #[test]
    fn rocm_runtime_health_latch_is_inert_off_rocm_and_sticky_on_fault() {
        let backend_health = BackendHealthHandle::default();
        let inactive = crate::accelerator_runtime::RocmSynchronizationRuntimeStats::default();
        latch_rocm_runtime_health(&backend_health, &inactive);
        backend_health.ensure_healthy().unwrap();

        let mut unavailable = inactive;
        unavailable.active = true;
        unavailable.telemetry_error = Some("injected telemetry failure".to_string());
        latch_rocm_runtime_health(&backend_health, &unavailable);

        let error = backend_health.ensure_healthy().unwrap_err().to_string();
        assert!(error.contains("requires restart"));
        assert!(error.contains("telemetry is unavailable"));
        assert!(error.contains("injected telemetry failure"));

        let first_reason = backend_health.snapshot().reason;
        let mut cleanup_quarantined = unavailable;
        cleanup_quarantined.telemetry_available = true;
        cleanup_quarantined.telemetry_error = None;
        cleanup_quarantined.cleanup_quarantined = true;
        latch_rocm_runtime_health(&backend_health, &cleanup_quarantined);
        assert_eq!(backend_health.snapshot().reason, first_reason);
    }

    #[test]
    fn training_workload_labels_are_wire_stable() {
        assert_eq!(TrainingWorkload::Sft.label(), "sft");
        assert_eq!(TrainingWorkload::Grpo.label(), "grpo");
        assert_eq!(TrainingWorkload::Opd.label(), "opd");
        assert_eq!(TrainingWorkload::DistillRefresh.label(), "distill_refresh");
    }

    #[test]
    fn training_workload_routes_fail_closed_by_workload() {
        use kiln_model::backend::{
            OpdLossRoute, OpdPhaseBBackwardRoute, TrainingCapabilities, TrainingTapeRoute,
        };

        let checkpointed = kiln_train::CheckpointConfig {
            num_segments: 4,
            enabled: true,
            auto_configured: false,
        };
        let uncheckpointed = kiln_train::CheckpointConfig {
            num_segments: 1,
            enabled: false,
            auto_configured: false,
        };
        let mut capabilities = TrainingCapabilities::portable();

        assert_eq!(
            training_workload_route_unavailable_reason(
                TrainingWorkload::DistillRefresh,
                capabilities,
                uncheckpointed,
            )
            .as_deref(),
            Some(DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE),
            "the composite refresh worker must remain fail-closed until both phase plans are admitted"
        );

        let reason = training_workload_route_unavailable_reason(
            TrainingWorkload::Grpo,
            capabilities,
            checkpointed,
        )
        .expect("non-authoritative tape route must reject GRPO");
        assert!(reason.contains("backend route is `unsupported`"));

        capabilities.tape_forward_backward_route = TrainingTapeRoute::KtTapeAuthoritative;
        assert!(
            training_workload_route_unavailable_reason(
                TrainingWorkload::Grpo,
                capabilities,
                checkpointed,
            )
            .is_none(),
            "authoritative tape is the complete static GRPO route contract"
        );

        let reason = training_workload_route_unavailable_reason(
            TrainingWorkload::Sft,
            capabilities,
            checkpointed,
        )
        .expect("checkpointed full-logits SFT must fail closed");
        assert!(reason.contains("loss route `full_logits`"));
        assert!(
            training_workload_route_unavailable_reason(
                TrainingWorkload::Sft,
                capabilities,
                uncheckpointed,
            )
            .is_none(),
            "full-logits SFT remains valid when checkpointing is ineffective"
        );

        let reason = training_workload_route_unavailable_reason(
            TrainingWorkload::Opd,
            capabilities,
            uncheckpointed,
        )
        .expect("unsupported OPD loss route must fail closed");
        assert!(reason.contains("loss route is `unsupported`"));
        capabilities.opd_loss_route = OpdLossRoute::KtTapePhaseB;
        let reason = training_workload_route_unavailable_reason(
            TrainingWorkload::Opd,
            capabilities,
            uncheckpointed,
        )
        .expect("unsupported OPD backward route must fail closed");
        assert!(reason.contains("phase-B backward route is `unsupported`"));
        capabilities.opd_phase_b_backward_route = OpdPhaseBBackwardRoute::KtComposite;
        assert!(
            training_workload_route_unavailable_reason(
                TrainingWorkload::Opd,
                capabilities,
                uncheckpointed,
            )
            .is_none()
        );
    }

    #[test]
    fn mock_state_shares_one_cpu_streaming_prefill_policy() {
        let model_config = ModelConfig::qwen3_5_4b();
        let scheduler = Scheduler::new(kiln_scheduler::SchedulerConfig::default(), 256);
        let state = AppState::new_mock(
            model_config.clone(),
            scheduler,
            Arc::new(kiln_model::engine::MockEngine::new(model_config)),
            crate::api::test_tokenizer(),
            300,
            "Qwen3.5-4B".to_string(),
        );

        assert_eq!(
            state.training_runtime.configured_streaming_prefill_policy(),
            Some(state.streaming_prefill_runtime_config.execution_policy())
        );
        assert_eq!(
            state
                .streaming_prefill_runtime_config
                .dispatch
                .backend_policy
                .policy,
            crate::config::StreamingPrefillDispatchPolicy::Never
        );
        assert_eq!(
            state.training_workload_unavailable_reason(TrainingWorkload::Sft),
            Some("mock backend does not execute sft training".to_string())
        );
        assert_eq!(
            state.training_workload_unavailable_reason(TrainingWorkload::DistillRefresh),
            Some(DISTILL_REFRESH_COMPOSITE_ADMISSION_UNAVAILABLE.to_string()),
            "the composite refresh reason must remain stable across shared substrate failures"
        );
    }

    #[test]
    fn direct_decode_rendezvous_status_distinguishes_worker_from_route() {
        let unavailable = DirectDecodeRendezvousRuntimeState::resolve(false, false, false);
        assert_eq!(unavailable.scope, "direct_streaming_greedy_only");
        assert_eq!(unavailable.backend_unavailable_reason, Some("mock_backend"));
        assert!(!unavailable.backend_available);
        assert!(!unavailable.actor_active);
        assert!(!unavailable.worker_active);
        assert!(!unavailable.route_available);

        let impossible_mock_worker =
            DirectDecodeRendezvousRuntimeState::resolve(false, false, true);
        assert!(impossible_mock_worker.worker_active);
        assert!(!impossible_mock_worker.route_available);

        let disabled = DirectDecodeRendezvousRuntimeState::resolve(true, false, false);
        assert!(disabled.backend_available);
        assert_eq!(disabled.backend_unavailable_reason, None);
        assert!(!disabled.actor_active);
        assert!(!disabled.worker_active);
        assert!(!disabled.route_available);

        let routed = DirectDecodeRendezvousRuntimeState::resolve(true, false, true);
        assert!(routed.worker_active);
        assert!(routed.route_available);

        let shadowed = DirectDecodeRendezvousRuntimeState::resolve(true, true, true);
        assert!(shadowed.actor_active);
        assert!(shadowed.worker_active);
        assert!(!shadowed.route_available);
    }

    fn test_adapter(name: &str) -> Option<LoadedAdapterIdentity> {
        test_adapter_revision(name, &format!("revision:{name}"))
    }

    fn test_adapter_revision(name: &str, revision: &str) -> Option<LoadedAdapterIdentity> {
        Some(LoadedAdapterIdentity {
            name: name.to_string(),
            content_revision: revision.to_string(),
        })
    }

    fn test_cache_key(request: &str) -> DeterministicCacheKey {
        DeterministicCacheKey::new(None, request.to_string())
    }

    #[derive(Debug)]
    struct SnapshotFailureStorage;

    impl kiln_tensor::StorageBackend for SnapshotFailureStorage {
        fn device(&self) -> kiln_tensor::Device {
            kiln_tensor::Device::Vulkan(0)
        }

        fn dtype(&self) -> kiln_tensor::DType {
            kiln_tensor::DType::F32
        }

        fn byte_len(&self) -> usize {
            std::mem::size_of::<f32>()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn snapshot_failing_linear_state() -> anyhow::Result<LinearAttentionState> {
        let storage: kiln_tensor::Storage = std::sync::Arc::new(SnapshotFailureStorage);
        let recurrent_state = kiln_tensor::Tensor::from_parts(
            storage,
            kiln_tensor::Layout::contiguous([1]),
            kiln_tensor::TensorId::next(),
        )?;
        Ok(LinearAttentionState {
            recurrent_states: vec![recurrent_state],
            conv_states: Vec::new(),
        })
    }

    #[test]
    fn pool_trim_target_skips_empty_spare_and_honors_requested_bytes() {
        assert_eq!(pool_trim_min_keep(8_000, 8_000, u64::MAX), None);
        assert_eq!(pool_trim_min_keep(8_000, 4_000, 0), None);
        assert_eq!(pool_trim_min_keep(8_000, 4_000, 1_000), Some(7_000));
        assert_eq!(pool_trim_min_keep(8_000, 4_000, u64::MAX), Some(4_000));
        assert_eq!(pool_trim_min_keep(4_000, 8_000, u64::MAX), None);
    }

    #[test]
    fn pool_trim_target_requires_an_idle_batching_actor() {
        assert_eq!(
            pool_trim_min_keep_if_idle(8_000, 4_000, u64::MAX, 1, 0),
            None
        );
        assert_eq!(
            pool_trim_min_keep_if_idle(8_000, 4_000, u64::MAX, 0, 1),
            None
        );
        assert_eq!(
            pool_trim_min_keep_if_idle(8_000, 4_000, 1_000, 0, 0),
            Some(7_000)
        );
    }

    /// Shared CPU device for tests. #1082: now emits a kt `Device::Cpu`
    /// directly — `AppState::new_real` and `LinearAttentionState::new` both take
    /// `kt::Device` after the forward flip, so the previous kt→candle bridge
    /// (needed while those APIs still wanted `candle_core::Device`) is gone.
    /// Kept as a macro for call-site compatibility (`cpu_device!()` /
    /// `&cpu_device!()`); kt `Device` is `Copy`.
    macro_rules! cpu_device {
        () => {
            ::kiln_tensor::Device::Cpu
        };
    }

    #[test]
    fn gpu_coordination_read_owner_moves_and_excludes_writer_until_drop() {
        let gpu_lock: GpuCoordinationLock = std::sync::Arc::new(RwLock::new(()));
        let read_owner = gpu_coordination_read_guard(&gpu_lock);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(read_owner);
        });
        ready_rx.recv().unwrap();
        assert!(
            gpu_lock.try_write().is_err(),
            "training writer acquired while moved inference owner was live"
        );
        release_tx.send(()).unwrap();
        reader.join().unwrap();
        assert!(
            gpu_lock.try_write().is_ok(),
            "training writer must acquire after the moved read owner drops"
        );
    }

    #[test]
    fn health_checked_gpu_writer_rejects_without_waiting_for_retained_reader() {
        let gpu_lock: GpuCoordinationLock = std::sync::Arc::new(RwLock::new(()));
        let retained_reader = gpu_coordination_read_guard(&gpu_lock);
        let backend_health = BackendHealthHandle::default();
        let worker_lock = gpu_lock.clone();
        let worker_health = backend_health.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let result =
                gpu_coordination_write_guard_while_healthy(&worker_lock, &worker_health).map(drop);
            result_tx.send(result).unwrap();
        });

        assert!(
            result_rx
                .recv_timeout(std::time::Duration::from_millis(25))
                .is_err(),
            "healthy writer should still be waiting behind inference"
        );
        backend_health.quarantine("injected unknown inference completion");
        let error = result_rx
            .recv_timeout(std::time::Duration::from_millis(250))
            .expect("quarantine must interrupt the writer wait")
            .expect_err("quarantined writer must reject");
        assert!(error.to_string().contains("requires restart"));
        assert!(
            gpu_lock.try_write().is_err(),
            "the test must retain the unknown inference owner"
        );
        std::mem::forget(retained_reader);
    }

    #[tokio::test]
    async fn async_health_checked_gpu_writer_rejects_retained_reader() {
        let gpu_lock: GpuCoordinationLock = std::sync::Arc::new(RwLock::new(()));
        let retained_reader = gpu_lock.clone().read_owned().await;
        let backend_health = BackendHealthHandle::default();
        let worker_lock = gpu_lock.clone();
        let worker_health = backend_health.clone();
        let writer = tokio::spawn(async move {
            gpu_coordination_write_guard_while_healthy_async(&worker_lock, &worker_health)
                .await
                .map(drop)
        });

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(!writer.is_finished());
        backend_health.quarantine("injected async unknown inference completion");
        let error = tokio::time::timeout(std::time::Duration::from_millis(250), writer)
            .await
            .expect("quarantine must interrupt the async writer wait")
            .unwrap()
            .expect_err("quarantined async writer must reject");
        assert!(error.to_string().contains("requires restart"));
        assert!(gpu_lock.try_write().is_err());
        std::mem::forget(retained_reader);
    }

    fn tiny_linear_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            num_layers: 2,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 2,
            attn_output_gate: false,
            linear_num_key_heads: 1,
            linear_key_head_dim: 2,
            linear_num_value_heads: 1,
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 2,
            partial_rotary_factor: 0.5,
        }
    }

    #[tokio::test]
    async fn deterministic_completion_cache_coalesces_in_flight_request() {
        let mut cache = DeterministicCompletionCache::new(8);
        let key = DeterministicCompletionCacheKey {
            adapter: None,
            global_generation: 0,
            adapter_generation: 0,
            prompt_tokens: vec![1, 2, 3],
            temperature_bits: 0.0f32.to_bits(),
            max_tokens: 4,
            ignore_eos: false,
            thinking_budget_tokens: None,
            stop: Vec::new(),
            top_p_bits: 1.0f32.to_bits(),
            top_k: 0,
            min_p_bits: 0.0f32.to_bits(),
            presence_penalty_bits: 0.0f32.to_bits(),
            frequency_penalty_bits: 0.0f32.to_bits(),
            repetition_penalty_bits: 1.0f32.to_bits(),
            seed: None,
            fold_reasoning_into_content: false,
        };
        let value = DeterministicCompletionCacheValue {
            text: "cached".to_string(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "length".to_string(),
            completion_tokens: 4,
            thinking_budget_status: None,
        };

        let claim_id = match cache.claim(&key) {
            DeterministicCompletionCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first request should own the cache claim"),
        };

        let mut receiver = match cache.claim(&key) {
            DeterministicCompletionCacheClaim::Wait(receiver) => receiver,
            _ => panic!("second identical request should wait on in-flight owner"),
        };
        cache.complete(key.clone(), claim_id, value.clone());

        receiver
            .changed()
            .await
            .expect("owner should publish result");
        let ready = receiver.borrow().clone();
        let DeterministicCompletionInFlightState::Ready(Some(published)) = ready else {
            panic!("waiter should receive cached completion");
        };
        assert_eq!(published.text, value.text);
        assert_eq!(published.completion_tokens, value.completion_tokens);

        let hit = match cache.claim(&key) {
            DeterministicCompletionCacheClaim::Hit(hit) => hit,
            _ => panic!("completed result should be cached"),
        };
        assert_eq!(hit.text, "cached");
    }

    #[test]
    fn unowned_completion_store_does_not_replace_concurrent_owner() {
        let mut cache = DeterministicCompletionCache::new(8);
        let key = DeterministicCompletionCacheKey {
            adapter: None,
            global_generation: 0,
            adapter_generation: 0,
            prompt_tokens: vec![1, 2, 3],
            temperature_bits: 0.0f32.to_bits(),
            max_tokens: 4,
            ignore_eos: false,
            thinking_budget_tokens: None,
            stop: Vec::new(),
            top_p_bits: 1.0f32.to_bits(),
            top_k: 0,
            min_p_bits: 0.0f32.to_bits(),
            presence_penalty_bits: 0.0f32.to_bits(),
            frequency_penalty_bits: 0.0f32.to_bits(),
            repetition_penalty_bits: 1.0f32.to_bits(),
            seed: None,
            fold_reasoning_into_content: false,
        };
        let value = DeterministicCompletionCacheValue {
            text: "probe result".to_string(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "length".to_string(),
            completion_tokens: 4,
            thinking_budget_status: None,
        };

        let claim_id = match cache.claim(&key) {
            DeterministicCompletionCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first request should own the cache claim"),
        };
        assert!(!cache.insert_unowned_complete_value(key.clone(), value));
        assert!(matches!(
            cache.probe(&key),
            DeterministicCompletionCacheProbe::Wait(_)
        ));

        let owner_value = DeterministicCompletionCacheValue {
            text: "owner result".to_string(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "stop".to_string(),
            completion_tokens: 2,
            thinking_budget_status: None,
        };
        cache.complete(key.clone(), claim_id, owner_value);
        let DeterministicCompletionCacheProbe::Hit(hit) = cache.probe(&key) else {
            panic!("the concurrent owner must remain authoritative");
        };
        assert_eq!(hit.text, "owner result");
        assert_eq!(hit.completion_tokens, 2);
    }

    #[tokio::test]
    async fn deterministic_batch_cache_coalesces_in_flight_request() {
        let mut cache = DeterministicBatchCache::new(8);
        let key = test_cache_key("batch-key");
        let value = DeterministicBatchCacheValue {
            completions: vec![DeterministicBatchCacheItem {
                prompt_index: 0,
                completion_index: 0,
                text: "cached".to_string(),
                reasoning_content: None,
                tool_calls: None,
                finish_reason: "length".to_string(),
                prompt_tokens: 3,
                completion_tokens: 4,
                thinking_budget_status: None,
            }],
            prompt_tokens: 3,
            completion_tokens: 4,
        };

        let claim_id = match cache.claim(&key) {
            DeterministicBatchCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first batch should own the cache claim"),
        };

        let mut receiver = match cache.claim(&key) {
            DeterministicBatchCacheClaim::Wait(receiver) => receiver,
            _ => panic!("second identical batch should wait on in-flight owner"),
        };
        cache.complete(key.clone(), claim_id, value.clone());

        receiver
            .changed()
            .await
            .expect("owner should publish batch result");
        let ready = receiver.borrow().clone();
        let DeterministicBatchInFlightState::Ready(Some(published)) = ready else {
            panic!("waiter should receive cached batch");
        };
        assert_eq!(published.completions[0].text, value.completions[0].text);
        assert_eq!(published.completion_tokens, value.completion_tokens);

        let hit = match cache.claim(&key) {
            DeterministicBatchCacheClaim::Hit(hit) => hit,
            _ => panic!("completed batch should be cached"),
        };
        assert_eq!(hit.completions[0].text, "cached");
    }

    #[tokio::test]
    async fn deterministic_chat_request_cache_coalesces_in_flight_request() {
        let mut cache = DeterministicChatRequestCache::new(8);
        let key = test_cache_key("chat-key");
        let value = DeterministicChatRequestCacheValue {
            prompt_tokens: 3,
            completion: DeterministicCompletionCacheValue {
                text: "cached".to_string(),
                reasoning_content: None,
                tool_calls: None,
                finish_reason: "length".to_string(),
                completion_tokens: 4,
                thinking_budget_status: None,
            },
        };

        let claim_id = match cache.claim(&key) {
            DeterministicChatRequestCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first chat request should own the cache claim"),
        };

        let mut receiver = match cache.claim(&key) {
            DeterministicChatRequestCacheClaim::Wait(receiver) => receiver,
            _ => panic!("second identical chat request should wait on in-flight owner"),
        };
        cache.complete(key.clone(), claim_id, value.clone());

        receiver
            .changed()
            .await
            .expect("owner should publish chat result");
        let ready = receiver.borrow().clone();
        let DeterministicChatRequestInFlightState::Ready(Some(published)) = ready else {
            panic!("waiter should receive cached chat request");
        };
        assert_eq!(published.prompt_tokens, value.prompt_tokens);
        assert_eq!(published.completion.text, value.completion.text);

        let hit = match cache.claim(&key) {
            DeterministicChatRequestCacheClaim::Hit(hit) => hit,
            _ => panic!("completed chat request should be cached"),
        };
        assert_eq!(hit.completion.text, "cached");
    }

    #[tokio::test]
    async fn deterministic_chat_choices_cache_coalesces_in_flight_request() {
        let mut cache = DeterministicChatChoicesCache::new(8);
        let key = test_cache_key("chat-choices-key");
        let value = DeterministicChatChoicesCacheValue {
            prompt_tokens: 3,
            completions: vec![
                DeterministicCompletionCacheValue {
                    text: "first".to_string(),
                    reasoning_content: Some("think first".to_string()),
                    tool_calls: None,
                    finish_reason: "length".to_string(),
                    completion_tokens: 4,
                    thinking_budget_status: None,
                },
                DeterministicCompletionCacheValue {
                    text: "second".to_string(),
                    reasoning_content: None,
                    tool_calls: None,
                    finish_reason: "stop".to_string(),
                    completion_tokens: 2,
                    thinking_budget_status: None,
                },
            ],
        };

        let claim_id = match cache.claim(&key) {
            DeterministicChatChoicesCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first chat choices request should own the cache claim"),
        };

        let mut receiver = match cache.claim(&key) {
            DeterministicChatChoicesCacheClaim::Wait(receiver) => receiver,
            _ => panic!("second identical chat choices request should wait on in-flight owner"),
        };
        cache.complete(key.clone(), claim_id, value.clone());

        receiver
            .changed()
            .await
            .expect("owner should publish chat choices result");
        let ready = receiver.borrow().clone();
        let DeterministicChatChoicesInFlightState::Ready(Some(published)) = ready else {
            panic!("waiter should receive cached chat choices");
        };
        assert_eq!(published.prompt_tokens, value.prompt_tokens);
        assert_eq!(published.completions[0].text, value.completions[0].text);
        assert_eq!(
            published.completions[0].reasoning_content,
            value.completions[0].reasoning_content
        );
        assert_eq!(
            published.completions[1].completion_tokens,
            value.completions[1].completion_tokens
        );

        let hit = match cache.claim(&key) {
            DeterministicChatChoicesCacheClaim::Hit(hit) => hit,
            _ => panic!("completed chat choices should be cached"),
        };
        assert_eq!(hit.completions[0].text, "first");
    }

    #[test]
    fn purged_deterministic_cache_owners_cannot_resurrect_results() {
        let adapter = test_adapter_revision("rewritten", "revision-one");
        let adapter_name = Some("rewritten".to_string());
        let completion_value = DeterministicCompletionCacheValue {
            text: "stale".to_string(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "stop".to_string(),
            completion_tokens: 1,
            thinking_budget_status: None,
        };
        let completion_key = DeterministicCompletionCacheKey {
            adapter: adapter.clone(),
            global_generation: 0,
            adapter_generation: 0,
            prompt_tokens: vec![1, 2, 3],
            temperature_bits: 0,
            max_tokens: 1,
            ignore_eos: false,
            thinking_budget_tokens: None,
            stop: Vec::new(),
            top_p_bits: 0,
            top_k: 0,
            min_p_bits: 0,
            presence_penalty_bits: 0,
            frequency_penalty_bits: 0,
            repetition_penalty_bits: 0,
            seed: None,
            fold_reasoning_into_content: false,
        };
        let mut completion_cache = DeterministicCompletionCache::new(8);
        let stale_completion_claim = match completion_cache.claim(&completion_key) {
            DeterministicCompletionCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first completion should own its claim"),
        };
        completion_cache.purge_adapter(&adapter_name);
        assert!(!completion_cache.complete(
            completion_key.clone(),
            stale_completion_claim,
            completion_value.clone(),
        ));
        assert!(matches!(
            completion_cache.probe(&completion_key),
            DeterministicCompletionCacheProbe::Miss
        ));

        let fresh_completion_claim = match completion_cache.claim(&completion_key) {
            DeterministicCompletionCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("post-purge completion should get a fresh claim"),
        };
        assert!(!completion_cache.fail(&completion_key, stale_completion_claim));
        assert!(completion_cache.complete(
            completion_key.clone(),
            fresh_completion_claim,
            completion_value.clone(),
        ));

        let request_key =
            DeterministicCacheKey::new(adapter.clone(), "same deterministic request".to_string());
        let chat_value = DeterministicChatRequestCacheValue {
            prompt_tokens: 3,
            completion: completion_value.clone(),
        };
        let mut chat_cache = DeterministicChatRequestCache::new(8);
        let chat_claim = match chat_cache.claim(&request_key) {
            DeterministicChatRequestCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first chat request should own its claim"),
        };
        chat_cache.purge_adapter(&adapter_name);
        assert!(!chat_cache.complete(request_key.clone(), chat_claim, chat_value));
        assert!(matches!(
            chat_cache.probe(&request_key),
            DeterministicChatRequestCacheProbe::Miss
        ));

        let choices_value = DeterministicChatChoicesCacheValue {
            prompt_tokens: 3,
            completions: vec![completion_value.clone()],
        };
        let mut choices_cache = DeterministicChatChoicesCache::new(8);
        let choices_claim = match choices_cache.claim(&request_key) {
            DeterministicChatChoicesCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first choices request should own its claim"),
        };
        choices_cache.purge_adapter(&adapter_name);
        assert!(!choices_cache.complete(request_key.clone(), choices_claim, choices_value));
        assert!(matches!(
            choices_cache.probe(&request_key),
            DeterministicChatChoicesCacheProbe::Miss
        ));

        let batch_value = DeterministicBatchCacheValue {
            completions: Vec::new(),
            prompt_tokens: 3,
            completion_tokens: 1,
        };
        let mut batch_cache = DeterministicBatchCache::new(8);
        let batch_claim = match batch_cache.claim(&request_key) {
            DeterministicBatchCacheClaim::Owner(claim_id) => claim_id,
            _ => panic!("first batch should own its claim"),
        };
        batch_cache.purge_adapter(&adapter_name);
        assert!(!batch_cache.complete(request_key.clone(), batch_claim, batch_value));
        assert!(matches!(
            batch_cache.claim(&request_key),
            DeterministicBatchCacheClaim::Owner(_)
        ));
    }

    #[test]
    fn deterministic_keys_distinguish_same_name_content_revisions() {
        let old = DeterministicCacheKey::new(
            test_adapter_revision("same-name", "old-revision"),
            "request".to_string(),
        );
        let new = DeterministicCacheKey::new(
            test_adapter_revision("same-name", "new-revision"),
            "request".to_string(),
        );
        assert_ne!(old, new);

        let mut cache = DeterministicChatRequestCache::new(8);
        cache.insert(
            old,
            DeterministicChatRequestCacheValue {
                prompt_tokens: 1,
                completion: DeterministicCompletionCacheValue {
                    text: "old".to_string(),
                    reasoning_content: None,
                    tool_calls: None,
                    finish_reason: "stop".to_string(),
                    completion_tokens: 1,
                    thinking_budget_status: None,
                },
            },
        );
        assert!(matches!(
            cache.probe(&new),
            DeterministicChatRequestCacheProbe::Miss
        ));
    }

    #[test]
    fn adapter_purge_generation_hides_late_unowned_insert() {
        let adapter = test_adapter_revision("same-name", "same-revision");
        let mut generations = DeterministicCacheGenerations::default();
        let (old_global, old_adapter_generation) = generations.snapshot(&adapter);
        let old_key = DeterministicCacheKey {
            adapter: adapter.clone(),
            global_generation: old_global,
            adapter_generation: old_adapter_generation,
            request: "request".to_string(),
        };
        generations.purge_adapter(&Some("same-name".to_string()));
        let (new_global, new_adapter_generation) = generations.snapshot(&adapter);
        let new_key = DeterministicCacheKey {
            adapter,
            global_generation: new_global,
            adapter_generation: new_adapter_generation,
            request: "request".to_string(),
        };
        assert_ne!(old_key, new_key);

        let mut cache = DeterministicChatRequestCache::new(8);
        cache.insert(
            old_key,
            DeterministicChatRequestCacheValue {
                prompt_tokens: 1,
                completion: DeterministicCompletionCacheValue {
                    text: "late old result".to_string(),
                    reasoning_content: None,
                    tool_calls: None,
                    finish_reason: "stop".to_string(),
                    completion_tokens: 1,
                    thinking_budget_status: None,
                },
            },
        );
        assert!(matches!(
            cache.probe(&new_key),
            DeterministicChatRequestCacheProbe::Miss
        ));
    }

    #[test]
    fn prompt_token_cache_tracks_hits_misses_and_eviction() {
        let mut cache = PromptTokenCache::new(2);
        assert_eq!(cache.stats(), (0, 0, 0));

        assert!(cache.get("a").is_none());
        cache.insert("a".to_string(), vec![1, 2]);
        assert_eq!(cache.get("a"), Some(vec![1, 2]));
        assert_eq!(cache.stats(), (1, 1, 1));

        cache.insert("b".to_string(), vec![3]);
        cache.insert("c".to_string(), vec![4]);
        assert!(
            cache.get("a").is_none(),
            "oldest entry should be evicted after capacity is exceeded"
        );
        assert_eq!(cache.get("b"), Some(vec![3]));
        assert_eq!(cache.get("c"), Some(vec![4]));
    }

    #[test]
    fn rendered_prompt_cache_tracks_hits_misses_and_eviction() {
        let mut cache = RenderedPromptCache::new(2);
        assert_eq!(cache.stats(), (0, 0, 0));

        assert!(cache.get("key-a").is_none());
        cache.insert("key-a".to_string(), "prompt a".to_string());
        assert_eq!(cache.get("key-a"), Some("prompt a".to_string()));
        assert_eq!(cache.stats(), (1, 1, 1));

        cache.insert("key-b".to_string(), "prompt b".to_string());
        cache.insert("key-c".to_string(), "prompt c".to_string());
        assert!(
            cache.get("key-a").is_none(),
            "oldest rendered prompt should be evicted after capacity is exceeded"
        );
        assert_eq!(cache.get("key-b"), Some("prompt b".to_string()));
        assert_eq!(cache.get("key-c"), Some("prompt c".to_string()));
    }

    #[test]
    fn real_prefix_cache_records_hits_misses_and_cached_blocks() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let state = LinearAttentionState::new(&config, &device)?;
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);

        let registration = PagedPrefixRegistration {
            prompt_tokens: vec![1, 2, 3, 4],
            block_ids: vec![9],
            linear_state: state,
            next_token: None,
        };
        let outcome = cache.register(None, registration);
        assert_eq!(outcome.retained_blocks, vec![9]);
        assert!(outcome.evicted_blocks.is_empty());

        assert!(
            cache
                .lookup(&None, &[7, 8, 9, 10, 11], &SamplingParams::greedy())
                .hit
                .is_ok_and(|hit| hit.is_none())
        );
        let hit = cache
            .lookup(&None, &[1, 2, 3, 4, 5], &SamplingParams::greedy())
            .hit?
            .expect("prefix hit");
        assert_eq!(hit.cached_tokens, 4);
        assert_eq!(hit.block_ids, vec![9]);
        cache.release_hit(hit.entry_id);

        let stats = cache.stats();
        assert_eq!(stats.lookup_hits, 1);
        assert_eq!(stats.lookup_misses, 1);
        assert_eq!(stats.hit_tokens, 4);
        assert_eq!(stats.hit_blocks, 1);
        assert_eq!(stats.cached_blocks, 1);
        assert_eq!(stats.max_blocks, 4);
        assert_eq!(stats.cached_entries, 1);
        assert_eq!(stats.max_entries, 1024);
        assert_eq!(stats.cached_state_bytes, 49);
        assert_eq!(stats.max_state_bytes, 1024 * 49);
        Ok(())
    }

    #[test]
    fn real_prefix_cache_extended_entry_wins_over_prompt_only_on_multi_turn() -> anyhow::Result<()>
    {
        // Models the agentic (pi-style) workflow: turn 1 registers two
        // entries — the prompt at its true length (with a sampled next-token
        // for exact-hit reuse) and an "extended" block-aligned entry covering
        // prompt + emitted assistant tokens. Turn 2's prompt extends the
        // previous transcript with new user input. The extended entry must
        // win over the prompt-only one because it caches strictly more
        // tokens — otherwise every multi-turn call re-prefills the entire
        // growing conversation from scratch.
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 16, 1024, 49);

        // Turn 1 prompt is 5 tokens — non-block-aligned at block_size 4,
        // matches the common case where the chat template renders to
        // arbitrary lengths.
        let turn1_prompt = vec![10u32, 11, 12, 13, 14];
        let prompt_only = PagedPrefixRegistration {
            prompt_tokens: turn1_prompt.clone(),
            block_ids: vec![100, 101],
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: Some(PagedPrefixNextToken::GreedyToken(42)),
        };
        let _ = cache.register(None, prompt_only);

        // Extended entry: prompt + 3 generated tokens = 8 tokens (block-aligned).
        let extended_tokens = vec![10u32, 11, 12, 13, 14, 200, 201, 202];
        let extended = PagedPrefixRegistration {
            prompt_tokens: extended_tokens,
            block_ids: vec![100, 101],
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: None,
        };
        let _ = cache.register(None, extended);

        // Turn 2 prompt: turn-1 transcript + new user input.
        let turn2_prompt: Vec<u32> = vec![10, 11, 12, 13, 14, 200, 201, 202, 50, 51];
        let hit = cache
            .lookup(&None, &turn2_prompt, &SamplingParams::greedy())
            .hit?
            .expect("turn 2 must hit the cache on the extended entry");
        assert_eq!(
            hit.cached_tokens, 8,
            "extended entry covers 8 tokens (prompt + decoded); prompt-only would only cover 5 \
             and additionally fails strict-prefix because 5 % 4 != 0"
        );
        Ok(())
    }

    #[test]
    fn real_prefix_cache_unfittable_registration_does_not_evict_resident_entries()
    -> anyhow::Result<()> {
        // A registration that cannot fit even into an EMPTY cache (more new
        // blocks than max_blocks) must bail before the eviction loop. The
        // old behavior evicted every resident entry chasing the impossible
        // target and still registered nothing — one oversized conversation
        // wiped the cache for every other session.
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 4, 8, 1024);

        let resident_tokens = vec![10u32, 11, 12, 13, 14, 15, 16, 17];
        let resident = PagedPrefixRegistration {
            prompt_tokens: resident_tokens.clone(),
            block_ids: vec![100, 101],
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: None,
        };
        let outcome = cache.register(None, resident);
        assert_eq!(outcome.retained_blocks, vec![100, 101]);

        // 8 new blocks needed > max_blocks of 4: can never fit.
        let oversized = PagedPrefixRegistration {
            prompt_tokens: (0..32u32).collect(),
            block_ids: (200..208u32).collect(),
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: None,
        };
        let outcome = cache.register(None, oversized);
        assert!(
            outcome.retained_blocks.is_empty() && outcome.evicted_blocks.is_empty(),
            "unfittable registration must be a no-op"
        );

        let stats = cache.stats();
        assert_eq!(
            stats.cached_entries, 1,
            "resident entry must survive an unfittable registration attempt"
        );
        assert!(
            cache
                .lookup(
                    &None,
                    &[10u32, 11, 12, 13, 14, 15, 16, 17, 50, 51],
                    &SamplingParams::greedy(),
                )
                .hit?
                .is_some(),
            "resident entry must still be hittable after the unfittable attempt"
        );
        Ok(())
    }

    #[test]
    fn real_prefix_cache_min_register_tokens_skips_short_prompts() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let state = LinearAttentionState::new(&config, &device)?;
        let mut cache = RealPrefixCache::new_with_min_register_tokens(true, 4, 4, 1024, 49, 9);
        assert!(
            !cache.should_lookup_prompt(&[1, 2, 3, 4, 5, 6, 7, 8]),
            "short prompts cannot hit entries that are never registered"
        );
        assert!(cache.should_lookup_prompt(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));

        let outcome = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![9, 10],
                linear_state: state,
                next_token: Some(PagedPrefixNextToken::GreedyToken(123)),
            },
        );
        assert!(outcome.retained_blocks.is_empty());
        assert!(outcome.evicted_blocks.is_empty());
        assert!(
            cache
                .lookup(&None, &[1, 2, 3, 4, 5, 6, 7, 8], &SamplingParams::greedy(),)
                .hit?
                .is_none()
        );

        let stats = cache.stats();
        assert_eq!(stats.lookup_hits, 0);
        assert_eq!(stats.lookup_misses, 1);
        assert_eq!(stats.cached_blocks, 0);
        assert_eq!(stats.cached_entries, 0);
        assert_eq!(stats.cached_state_bytes, 0);
        Ok(())
    }

    #[test]
    fn real_prefix_cache_exact_prompt_hit_requires_next_token_source() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();

        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        let outcome = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![9],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        assert!(
            cache
                .lookup(&None, &[1, 2, 3, 4], &SamplingParams::greedy())
                .hit?
                .is_none(),
            "an exact prompt hit without a saved next-token source cannot skip prefill"
        );

        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![9],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: Some(PagedPrefixNextToken::GreedyToken(123)),
            },
        );
        assert_eq!(outcome.retained_blocks, vec![9]);
        assert!(outcome.evicted_blocks.is_empty());
        let hit = cache
            .lookup(&None, &[1, 2, 3, 4], &SamplingParams::greedy())
            .hit?
            .expect("exact hit");
        assert_eq!(hit.cached_tokens, 4);
        assert_eq!(hit.block_ids, vec![9]);
        assert!(matches!(
            hit.next_token,
            Some(PagedPrefixNextToken::GreedyToken(123))
        ));
        cache.release_hit(hit.entry_id);

        Ok(())
    }

    #[test]
    fn real_prefix_cache_ranks_exact_hits_by_sampling_compatibility() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 8, 1024, 49);

        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![10],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![10, 11],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: Some(PagedPrefixNextToken::GreedyToken(123)),
            },
        );

        let sampled = SamplingParams::default();
        assert!(!sampled.is_effectively_greedy());
        let sampled_hit = cache
            .lookup(&None, &[1, 2, 3, 4, 5, 6, 7, 8], &sampled)
            .hit?
            .expect("sampled lookup must fall back to the strict prefix");
        assert_eq!(sampled_hit.cached_tokens, 4);
        assert!(sampled_hit.next_token.is_none());
        assert_eq!(cache.entries[0].active_uses, 1);
        assert_eq!(cache.entries[1].active_uses, 0);
        cache.release_hit(sampled_hit.entry_id);

        let greedy_hit = cache
            .lookup(&None, &[1, 2, 3, 4, 5, 6, 7, 8], &SamplingParams::greedy())
            .hit?
            .expect("greedy lookup must use the longer exact entry");
        assert_eq!(greedy_hit.cached_tokens, 8);
        assert!(matches!(
            greedy_hit.next_token,
            Some(PagedPrefixNextToken::GreedyToken(123))
        ));
        assert_eq!(cache.entries[0].active_uses, 0);
        assert_eq!(cache.entries[1].active_uses, 1);
        cache.release_hit(greedy_hit.entry_id);

        let logits_adapter = test_adapter("logits");
        cache.register(
            logits_adapter.clone(),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![20, 21],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: Some(PagedPrefixNextToken::Logits(kiln_tensor::Tensor::from_vec(
                    vec![0.0f32; 4],
                    (1, 4),
                )?)),
            },
        );
        let logits_hit = cache
            .lookup(&logits_adapter, &[1, 2, 3, 4, 5, 6, 7, 8], &sampled)
            .hit?
            .expect("full logits must support sampled exact reuse");
        assert_eq!(logits_hit.cached_tokens, 8);
        assert!(matches!(
            logits_hit.next_token,
            Some(PagedPrefixNextToken::Logits(_))
        ));
        cache.release_hit(logits_hit.entry_id);

        let stats = cache.stats();
        assert_eq!(stats.lookup_hits, 3);
        assert_eq!(stats.lookup_misses, 0);
        assert_eq!(stats.hit_tokens, 20);
        assert_eq!(stats.hit_blocks, 5);
        assert!(cache.entries.iter().all(|entry| entry.active_uses == 0));
        Ok(())
    }

    #[test]
    fn real_prefix_cache_snapshot_failure_does_not_commit_hit_state() -> anyhow::Result<()> {
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![10],
                linear_state: snapshot_failing_linear_state()?,
                next_token: Some(PagedPrefixNextToken::GreedyToken(123)),
            },
        );

        let stats_before = cache.stats();
        let last_used_before = cache.entries[0].last_used;
        let active_uses_before = cache.entries[0].active_uses;
        let attempt = cache.lookup(&None, &[1, 2, 3, 4], &SamplingParams::greedy());
        let error = match attempt.hit {
            Err(error) => error,
            Ok(_) => anyhow::bail!("unsupported Vulkan deep copy unexpectedly succeeded"),
        };
        assert!(error.to_string().contains("snapshot recurrent state"));
        assert_eq!(cache.entries[0].last_used, last_used_before);
        assert_eq!(cache.entries[0].active_uses, active_uses_before + 1);
        cache.release_hit(attempt.leased_entry_id.expect("provisional lease"));
        assert_eq!(cache.entries[0].active_uses, active_uses_before);
        assert_eq!(cache.stats(), stats_before);
        Ok(())
    }

    #[test]
    fn real_prefix_cache_rejects_partial_block_exact_entry_and_uses_safe_prefix()
    -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();

        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        let safe = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![10],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        assert_eq!(safe.retained_blocks, vec![10]);
        assert!(safe.evicted_blocks.is_empty());

        let unsafe_exact = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5],
                block_ids: vec![10, 11],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: Some(PagedPrefixNextToken::GreedyToken(124)),
            },
        );
        assert!(unsafe_exact.retained_blocks.is_empty());
        assert!(unsafe_exact.evicted_blocks.is_empty());
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.cached_blocks(), 1);
        assert_eq!(cache.block_refcounts.get(&10), Some(&1));
        assert!(!cache.block_refcounts.contains_key(&11));
        assert_eq!(cache.entries[0].active_uses, 0);

        let malformed_aligned = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![20, 21, 22, 23, 24, 25, 26, 27],
                block_ids: vec![12],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        assert!(malformed_aligned.retained_blocks.is_empty());
        assert!(malformed_aligned.evicted_blocks.is_empty());
        assert_eq!(cache.entries.len(), 1);
        assert!(!cache.block_refcounts.contains_key(&12));

        let hit = cache
            .lookup(&None, &[1, 2, 3, 4, 5], &SamplingParams::greedy())
            .hit?
            .expect("safe strict-prefix fallback");
        assert_eq!(hit.cached_tokens, 4);
        assert_eq!(hit.block_ids, vec![10]);
        assert!(hit.next_token.is_none());
        assert_eq!(cache.entries[0].active_uses, 1);
        cache.release_hit(hit.entry_id);
        assert_eq!(cache.entries[0].active_uses, 0);
        assert_eq!(cache.cached_blocks(), 1);
        assert_eq!(cache.block_refcounts.get(&10), Some(&1));
        Ok(())
    }

    #[test]
    fn real_prefix_cache_lookup_skips_legacy_partial_block_entry() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 8, 1024, 49);

        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![10],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        // Simulate an entry created before partial-block exact reuse was
        // rejected. Candidate filtering must skip it before ranking and pin
        // only the shorter, complete-block prefix.
        let legacy_id = cache.next_entry_id;
        cache.next_entry_id += 1;
        *cache.block_refcounts.entry(10).or_insert(0) += 1;
        *cache.block_refcounts.entry(11).or_insert(0) += 1;
        cache.entries.push(RealPrefixCacheEntry {
            id: legacy_id,
            adapter: None,
            prompt_tokens: vec![1, 2, 3, 4, 5],
            block_ids: vec![10, 11],
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: Some(PagedPrefixNextToken::GreedyToken(124)),
            last_used: 0,
            active_uses: 0,
            retired: false,
        });

        let hit = cache
            .lookup(&None, &[1, 2, 3, 4, 5], &SamplingParams::greedy())
            .hit?
            .expect("safe strict-prefix fallback");
        assert_ne!(hit.entry_id, legacy_id);
        assert_eq!(hit.cached_tokens, 4);
        assert_eq!(hit.block_ids, vec![10]);
        assert_eq!(cache.entries[0].active_uses, 1);
        assert_eq!(cache.entries[1].active_uses, 0);
        let stats = cache.stats();
        assert_eq!(stats.lookup_hits, 1);
        assert_eq!(stats.hit_tokens, 4);
        assert_eq!(stats.hit_blocks, 1);

        cache.release_hit(hit.entry_id);
        assert_eq!(cache.entries[0].active_uses, 0);
        assert_eq!(cache.entries[1].active_uses, 0);
        Ok(())
    }

    #[test]
    fn real_prefix_cache_caps_entries_and_state_bytes() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 100, 2, 49);

        for i in 0..3u32 {
            let start = 1 + i * 4;
            cache.register(
                None,
                PagedPrefixRegistration {
                    prompt_tokens: vec![start, start + 1, start + 2, start + 3],
                    block_ids: vec![10 + i],
                    linear_state: LinearAttentionState::new(&config, &device)?,
                    next_token: None,
                },
            );
        }

        let stats = cache.stats();
        assert_eq!(stats.cached_entries, 2);
        assert_eq!(stats.max_entries, 2);
        assert_eq!(stats.cached_state_bytes, 98);
        assert_eq!(stats.max_state_bytes, 98);
        assert_eq!(stats.cached_blocks, 2);
        assert!(
            cache
                .lookup(&None, &[1, 2, 3, 4, 99], &SamplingParams::greedy())
                .hit?
                .is_none()
        );
        assert!(
            cache
                .lookup(&None, &[5, 6, 7, 8, 99], &SamplingParams::greedy())
                .hit?
                .is_some()
        );
        assert!(
            cache
                .lookup(&None, &[9, 10, 11, 12, 99], &SamplingParams::greedy())
                .hit?
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn default_prefix_cache_entries_reserves_state_memory_budget() {
        let entry = 49 * 1024 * 1024;
        assert_eq!(
            default_prefix_cache_max_entries(48 * 1024 * 1024 * 1024, entry),
            20
        );
        assert_eq!(
            default_prefix_cache_max_entries(24 * 1024 * 1024 * 1024, entry),
            12
        );
        assert_eq!(default_prefix_cache_max_entries(0, entry), 5);
    }

    #[test]
    fn vulkan_prefix_cache_reserves_only_proven_host_backed_memory() {
        const MIB: u64 = 1024 * 1024;
        const GIB: u64 = 1024 * MIB;

        assert_eq!(
            prefix_cache_host_reserve_bytes(Some(4 * GIB), 49 * MIB, true, None).unwrap(),
            256 * MIB
        );
        assert_eq!(
            prefix_cache_host_reserve_bytes(Some(4 * GIB), 49 * MIB, true, Some(3)).unwrap(),
            147 * MIB
        );
        assert_eq!(
            prefix_cache_host_reserve_bytes(Some(4 * GIB), 49 * MIB, false, Some(3)).unwrap(),
            0
        );

        let error = prefix_cache_host_reserve_bytes(Some(128 * MIB), 49 * MIB, true, Some(3))
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe GTT/host tier"));
        assert!(
            prefix_cache_host_reserve_bytes(Some(u64::MAX), u64::MAX, true, Some(2))
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
    }

    #[test]
    fn vulkan_host_backed_budget_is_independent_from_large_primary_vram() {
        const MIB: u64 = 1024 * 1024;
        const GIB: u64 = 1024 * MIB;
        let snapshot = kiln_memory::MemorySnapshot {
            total_bytes: 96 * GIB,
            used_bytes: 6 * GIB,
            free_bytes: 90 * GIB,
            source: kiln_memory::vram::VramSource::LinuxDrmSysfs,
            unified: false,
            observations: kiln_memory::MemorySnapshotObservations {
                host_backed: Some(kiln_memory::MemoryTierSnapshot {
                    total_bytes: 16 * GIB,
                    used_bytes: 12 * GIB,
                    free_bytes: 4 * GIB,
                }),
                ..Default::default()
            },
        };

        assert_eq!(
            host_backed_free_bytes_for_device(kiln_tensor::Device::Vulkan(0), None),
            Some(0),
            "a missing host tier must fail closed"
        );
        assert_eq!(
            host_backed_free_bytes_for_device(kiln_tensor::Device::Rocm(0), Some(snapshot)),
            None,
            "device-resident ROCm KV has no host-backed ceiling"
        );

        let host_budget = host_backed_kv_budget_for_fraction(
            kiln_tensor::Device::Vulkan(0),
            Some(snapshot),
            256 * MIB,
            0.7,
        )
        .unwrap();
        assert_eq!(host_budget, (((4 * GIB - 256 * MIB) as f64) * 0.7) as u64);
        let (capped, max_blocks) =
            cap_kv_blocks_to_live_budget(96 * 1024, MIB, 90 * GIB, Some(host_budget));
        assert_eq!(capped, max_blocks);
        assert_eq!(max_blocks, (host_budget / MIB) as usize);
        assert!(max_blocks < 4 * 1024);
    }

    #[test]
    fn real_prefix_cache_keys_by_adapter() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let state = LinearAttentionState::new(&config, &device)?;
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        cache.register(
            test_adapter("adapter-a"),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![9],
                linear_state: state,
                next_token: None,
            },
        );

        assert!(
            cache
                .lookup(&None, &[1, 2, 3, 4, 5], &SamplingParams::greedy())
                .hit?
                .is_none()
        );
        assert!(
            cache
                .lookup(
                    &test_adapter("adapter-b"),
                    &[1, 2, 3, 4, 5],
                    &SamplingParams::greedy(),
                )
                .hit?
                .is_none()
        );
        assert!(
            cache
                .lookup(
                    &test_adapter("adapter-a"),
                    &[1, 2, 3, 4, 5],
                    &SamplingParams::greedy(),
                )
                .hit?
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn real_prefix_cache_keys_same_name_by_content_revision() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        let old = test_adapter_revision("same-name", "old-revision");
        let new = test_adapter_revision("same-name", "new-revision");
        cache.register(
            old.clone(),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![9],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        assert!(
            cache
                .lookup(&new, &[1, 2, 3, 4, 5], &SamplingParams::greedy())
                .hit?
                .is_none(),
            "a same-name rewrite must not reuse the previous revision's KV"
        );
        let old_hit = cache
            .lookup(&old, &[1, 2, 3, 4, 5], &SamplingParams::greedy())
            .hit?
            .expect("the exact old revision remains addressable before purge");
        cache.release_hit(old_hit.entry_id);
        assert_eq!(cache.purge_adapter(&Some("same-name".to_string())), vec![9]);
        Ok(())
    }

    #[test]
    fn prefix_cache_purge_adapter_is_selective_and_frees_blocks() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 16, 1024, 49);
        cache.register(
            test_adapter("retrained"),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![10],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            test_adapter("retrained"),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                // Shares block 10 with the first entry, plus its own 11.
                block_ids: vec![10, 11],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            test_adapter("untouched"),
            PagedPrefixRegistration {
                prompt_tokens: vec![9, 9, 9, 9],
                block_ids: vec![20],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        let mut freed = cache.purge_adapter(&Some("retrained".to_string()));
        freed.sort_unstable();
        // Both of the retrained adapter's blocks come back exactly once,
        // shared refcounts unwound correctly; the other adapter's block 20
        // stays held.
        assert_eq!(freed, vec![10, 11]);

        assert!(
            cache
                .lookup(
                    &test_adapter("retrained"),
                    &[1, 2, 3, 4, 5],
                    &SamplingParams::greedy(),
                )
                .hit?
                .is_none(),
            "retrained adapter's entries are gone"
        );
        assert!(
            cache
                .lookup(
                    &test_adapter("untouched"),
                    &[9, 9, 9, 9, 1],
                    &SamplingParams::greedy(),
                )
                .hit?
                .is_some(),
            "other adapters' entries survive — a background eval/training \
             swap must not cost the serving agent its prefix cache"
        );
        Ok(())
    }

    #[test]
    fn prefix_cache_clear_defers_reclamation_until_final_request_lease() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let block_manager = Arc::new(std::sync::Mutex::new(BlockManager::new(8, 4)));
        let blocks = block_manager.lock().unwrap().allocate(2)?;
        let cache = Arc::new(std::sync::Mutex::new(RealPrefixCache::new(
            true, 4, 8, 1024, 49,
        )));
        cache.lock().unwrap().register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: blocks.clone(),
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        let first = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            None,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        let second = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            None,
            &[1, 2, 3, 4, 5, 6, 7, 8, 10],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        assert!(first.hit.is_some());
        assert!(second.hit.is_some());
        assert_eq!(cache.lock().unwrap().stats().active_leases, 2);

        assert!(cache.lock().unwrap().clear().is_empty());
        assert!(
            cache.lock().unwrap().clear().is_empty(),
            "repeated invalidation must not release an active tombstone"
        );
        let stats = cache.lock().unwrap().stats();
        assert_eq!(stats.cached_entries, 1);
        assert_eq!(stats.cached_blocks, 2);
        assert_eq!(stats.active_leases, 2);
        assert_eq!(stats.pending_release_entries, 1);

        let after_clear = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            None,
            &[1, 2, 3, 4, 5, 6, 7, 8, 11],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        assert!(
            after_clear.hit.is_none(),
            "retired entries must become undiscoverable immediately"
        );
        drop(after_clear.request);

        drop(first.request);
        assert_eq!(block_manager.lock().unwrap().num_used(), 2);
        assert_eq!(cache.lock().unwrap().stats().active_leases, 1);
        drop(second.request);
        assert_eq!(block_manager.lock().unwrap().num_used(), 0);
        let stats = cache.lock().unwrap().stats();
        assert_eq!(stats.cached_entries, 0);
        assert_eq!(stats.cached_blocks, 0);
        assert_eq!(stats.active_leases, 0);
        assert_eq!(stats.pending_release_entries, 0);
        Ok(())
    }

    #[test]
    fn unsettled_prefix_hit_retains_snapshot_and_source_lease() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let block_manager = Arc::new(std::sync::Mutex::new(BlockManager::new(4, 4)));
        let blocks = block_manager.lock().unwrap().allocate(1)?;
        let cache = Arc::new(std::sync::Mutex::new(RealPrefixCache::new(
            true, 4, 4, 1024, 49,
        )));
        cache.lock().unwrap().register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: blocks,
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        let pending = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            None,
            &[1, 2, 3, 4, 5],
            &SamplingParams::greedy(),
        )?;
        let error = match pending.settle_with(|| anyhow::bail!("injected sync failure")) {
            Ok(_) => anyhow::bail!("injected synchronization unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("injected sync failure"));
        assert_eq!(cache.lock().unwrap().stats().active_leases, 1);
        assert!(cache.lock().unwrap().clear().is_empty());
        assert_eq!(block_manager.lock().unwrap().num_used(), 1);
        Ok(())
    }

    #[test]
    fn prefix_cache_stale_hit_finish_cannot_resurrect_purged_blocks() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let block_manager = Arc::new(std::sync::Mutex::new(BlockManager::new(8, 4)));
        let initial = block_manager.lock().unwrap().allocate(1)?;
        let cache = Arc::new(std::sync::Mutex::new(RealPrefixCache::new(
            true, 4, 8, 1024, 49,
        )));
        let adapter = test_adapter("retrained");
        cache.lock().unwrap().register(
            adapter.clone(),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: initial.clone(),
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        let lookup = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            adapter.clone(),
            &[1, 2, 3, 4, 5],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        assert!(lookup.hit.is_some());
        assert!(
            cache
                .lock()
                .unwrap()
                .purge_adapter(&Some("retrained".to_string()))
                .is_empty()
        );

        let suffix = block_manager.lock().unwrap().allocate(1)?;
        let outcome = lookup.request.finish(
            vec![PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![initial[0], suffix[0]],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            }],
            suffix,
        );
        assert!(!outcome.registrations_accepted);
        assert_eq!(outcome.released_blocks.len(), 2);
        assert_eq!(block_manager.lock().unwrap().num_used(), 0);
        assert!(cache.lock().unwrap().entries.is_empty());
        Ok(())
    }

    #[test]
    fn prefix_cache_stale_miss_is_fenced_but_other_adapter_can_finish() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let block_manager = Arc::new(std::sync::Mutex::new(BlockManager::new(8, 4)));
        let cache = Arc::new(std::sync::Mutex::new(RealPrefixCache::new(
            true, 4, 8, 1024, 49,
        )));
        let adapter_a = test_adapter("adapter-a");
        let adapter_b = test_adapter("adapter-b");
        let request_a = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            adapter_a.clone(),
            &[1, 2, 3, 4],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        let request_b = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            adapter_b.clone(),
            &[5, 6, 7, 8],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        assert!(request_a.hit.is_none());
        assert!(request_b.hit.is_none());
        cache
            .lock()
            .unwrap()
            .purge_adapter(&Some("adapter-a".to_string()));
        let blocks = block_manager.lock().unwrap().allocate(2)?;

        let outcome_a = request_a.request.finish(
            vec![PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![blocks[0]],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            }],
            vec![blocks[0]],
        );
        assert!(!outcome_a.registrations_accepted);
        let outcome_b = request_b.request.finish(
            vec![PagedPrefixRegistration {
                prompt_tokens: vec![5, 6, 7, 8],
                block_ids: vec![blocks[1]],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            }],
            vec![blocks[1]],
        );
        assert!(outcome_b.registrations_accepted);
        assert_eq!(outcome_b.retained_blocks, vec![blocks[1]]);
        assert_eq!(block_manager.lock().unwrap().num_used(), 1);

        let released = cache.lock().unwrap().clear();
        assert_eq!(released, vec![blocks[1]]);
        block_manager.lock().unwrap().free_all(&released);
        assert_eq!(block_manager.lock().unwrap().num_used(), 0);
        Ok(())
    }

    #[test]
    fn prefix_cache_global_clear_fences_in_flight_miss() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let block_manager = Arc::new(std::sync::Mutex::new(BlockManager::new(4, 4)));
        let cache = Arc::new(std::sync::Mutex::new(RealPrefixCache::new(
            true, 4, 4, 1024, 49,
        )));
        let lookup = RealPrefixCacheRequest::begin(
            &cache,
            &block_manager,
            test_adapter("adapter"),
            &[1, 2, 3, 4],
            &SamplingParams::greedy(),
        )?
        .settle_synchronous_for_test()?;
        assert!(lookup.hit.is_none());
        assert!(cache.lock().unwrap().clear().is_empty());
        let block = block_manager.lock().unwrap().allocate(1)?;
        let outcome = lookup.request.finish(
            vec![PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: block.clone(),
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            }],
            block,
        );
        assert!(!outcome.registrations_accepted);
        assert_eq!(block_manager.lock().unwrap().num_used(), 0);
        assert!(cache.lock().unwrap().entries.is_empty());
        Ok(())
    }

    #[test]
    fn completion_cache_purge_adapter_is_selective() {
        let mut cache = DeterministicCompletionCache::new(8);
        let key = |adapter: Option<&str>| DeterministicCompletionCacheKey {
            adapter: adapter.and_then(test_adapter),
            global_generation: 0,
            adapter_generation: 0,
            prompt_tokens: vec![1, 2, 3],
            temperature_bits: 0,
            max_tokens: 16,
            ignore_eos: false,
            thinking_budget_tokens: None,
            stop: Vec::new(),
            top_p_bits: 0,
            top_k: 0,
            min_p_bits: 0,
            presence_penalty_bits: 0,
            frequency_penalty_bits: 0,
            repetition_penalty_bits: 0,
            seed: Some(7),
            fold_reasoning_into_content: false,
        };
        let value = DeterministicCompletionCacheValue {
            text: "old-weights answer".to_string(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "stop".to_string(),
            completion_tokens: 3,
            thinking_budget_status: None,
        };
        cache.insert_complete_value(key(Some("retrained")), value.clone());
        cache.insert_complete_value(key(Some("other")), value.clone());
        cache.insert_complete_value(key(None), value);

        cache.purge_adapter(&Some("retrained".to_string()));

        assert!(
            matches!(
                cache.probe(&key(Some("retrained"))),
                DeterministicCompletionCacheProbe::Miss
            ),
            "retrained adapter must recompute"
        );
        assert!(matches!(
            cache.probe(&key(Some("other"))),
            DeterministicCompletionCacheProbe::Hit(_)
        ));
        assert!(matches!(
            cache.probe(&key(None)),
            DeterministicCompletionCacheProbe::Hit(_)
        ));
    }

    // Prefix-cache registration is a capacity transaction: it either plans a
    // complete fit and commits it, or leaves every resident entry untouched.

    #[test]
    fn register_rejects_incoming_union_larger_than_capacity() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 2, 1024, 49);

        let outcome_a = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![10, 11],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        assert_eq!(outcome_a.retained_blocks, vec![10, 11]);
        assert!(outcome_a.evicted_blocks.is_empty());

        // The fixed `needed_new_blocks` calculation used to count only block 12,
        // evict A, and then commit all three incoming blocks under a two-block
        // limit. Incoming unique ownership alone makes this entry impossible.
        let outcome_b = cache.register(
            test_adapter("larger"),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                block_ids: vec![10, 11, 12],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        assert!(outcome_b.retained_blocks.is_empty());
        assert!(outcome_b.evicted_blocks.is_empty());
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.cached_blocks(), 2);
        assert_eq!(cache.block_refcounts, HashMap::from([(10, 1), (11, 1)]));
        assert_eq!(cache.stats().max_blocks, 2);
        Ok(())
    }

    #[test]
    fn register_failure_does_not_partially_evict_before_pinned_entry() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 2, 1024, 49);
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![10],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            test_adapter("evictable"),
            PagedPrefixRegistration {
                prompt_tokens: vec![5, 6, 7, 8],
                block_ids: vec![11],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        let hit = cache
            .lookup(&None, &[1, 2, 3, 4, 9], &SamplingParams::greedy())
            .hit?
            .expect("first entry must be pinned");
        let entries_before: Vec<u64> = cache.entries.iter().map(|entry| entry.id).collect();
        let refcounts_before = cache.block_refcounts.clone();

        // Fitting [20, 21] requires both residents to leave. The unpinned
        // entry is a candidate, but the first entry cannot be evicted until
        // this hit releases its lease.
        let outcome = cache.register(
            test_adapter("incoming"),
            PagedPrefixRegistration {
                prompt_tokens: vec![9, 10, 11, 12, 13, 14, 15, 16],
                block_ids: vec![20, 21],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        assert!(outcome.retained_blocks.is_empty());
        assert!(outcome.evicted_blocks.is_empty());
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|entry| entry.id)
                .collect::<Vec<_>>(),
            entries_before
        );
        assert_eq!(cache.block_refcounts, refcounts_before);
        cache.release_hit(hit.entry_id);
        Ok(())
    }

    #[test]
    fn register_rejects_duplicate_physical_blocks() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
        let outcome = cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![10, 10],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        assert!(outcome.retained_blocks.is_empty());
        assert!(outcome.evicted_blocks.is_empty());
        assert!(cache.entries.is_empty());
        assert!(cache.block_refcounts.is_empty());
        Ok(())
    }

    #[test]
    fn register_outcome_no_duplicate_or_overlap() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 3, 1024, 49);

        // Three small entries that together fill capacity, two of which share
        // block 20 with the eventual incoming registration.
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4],
                block_ids: vec![20],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            test_adapter("a"),
            PagedPrefixRegistration {
                prompt_tokens: vec![5, 6, 7, 8, 9, 10, 11, 12],
                block_ids: vec![20, 21],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            test_adapter("b"),
            PagedPrefixRegistration {
                prompt_tokens: vec![13, 14, 15, 16],
                block_ids: vec![22],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        // Incoming registration shares block 20 with two existing entries and
        // brings a fresh block 99. Eviction will likely run multiple times.
        let outcome = cache.register(
            test_adapter("c"),
            PagedPrefixRegistration {
                prompt_tokens: vec![17, 18, 19, 20, 21, 22, 23, 24],
                block_ids: vec![20, 99],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        let retained_set: HashSet<u32> = outcome.retained_blocks.iter().copied().collect();
        assert_eq!(
            retained_set.len(),
            outcome.retained_blocks.len(),
            "retained_blocks must not contain duplicates: {:?}",
            outcome.retained_blocks,
        );

        let evicted_set: HashSet<u32> = outcome.evicted_blocks.iter().copied().collect();
        assert_eq!(
            evicted_set.len(),
            outcome.evicted_blocks.len(),
            "evicted_blocks must not contain duplicates: {:?}",
            outcome.evicted_blocks,
        );

        assert!(
            retained_set.is_disjoint(&evicted_set),
            "retained {retained_set:?} and evicted {evicted_set:?} must be disjoint",
        );
        Ok(())
    }

    #[test]
    fn register_evicted_blocks_not_in_refcounts_after() -> anyhow::Result<()> {
        let config = tiny_linear_config();
        let device = cpu_device!();
        let mut cache = RealPrefixCache::new(true, 4, 3, 1024, 49);

        // Make the unrelated entry oldest so one eviction is sufficient. The
        // shared prefix entry must remain resident and gain a second refcount.
        cache.register(
            test_adapter("old"),
            PagedPrefixRegistration {
                prompt_tokens: vec![40, 41, 42, 43],
                block_ids: vec![40],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );
        cache.register(
            None,
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                block_ids: vec![30, 31],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        // Register a longer prompt that reuses [30, 31] and adds 32. Both old
        // entries leave, but only block 40 becomes unowned after the incoming
        // entry is committed. The unrelated oldest entry is the only removal.
        let outcome = cache.register(
            test_adapter("ad"),
            PagedPrefixRegistration {
                prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                block_ids: vec![30, 31, 32],
                linear_state: LinearAttentionState::new(&config, &device)?,
                next_token: None,
            },
        );

        assert_eq!(outcome.retained_blocks, vec![32]);
        assert_eq!(outcome.evicted_blocks, vec![40]);
        assert_eq!(
            cache.block_refcounts,
            HashMap::from([(30, 2), (31, 2), (32, 1)])
        );
        for block_id in &outcome.evicted_blocks {
            assert!(
                !cache.block_refcounts.contains_key(block_id),
                "evicted block {block_id} must not be tracked in block_refcounts after register(); refcounts={:?}",
                cache.block_refcounts,
            );
        }
        Ok(())
    }

    #[test]
    fn test_memory_budget_cpu_mode() {
        let budget = GpuMemoryBudget::compute(0, 0, 0, 0, 0, 0.7, None);
        assert_eq!(budget.total_vram_bytes, 0);
        assert_eq!(budget.training_budget_bytes, 0);
        // CPU mode: training feasibility check always passes
        assert!(budget.check_training_feasible(1_000_000_000).is_ok());
    }

    #[test]
    fn test_memory_budget_24gb_gpu() {
        let total: u64 = 24 * 1024 * 1024 * 1024; // 24 GB
        let model: u64 = 8 * 1024 * 1024 * 1024; // 8 GB model
        let kv: u64 = 2 * 1024 * 1024 * 1024; // 2 GB KV cache
        let budget = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.7, None);
        assert_eq!(budget.total_vram_bytes, total);
        assert_eq!(budget.model_memory_bytes, model);
        assert_eq!(budget.kv_cache_bytes, kv);
        // training_budget = 24 - 8 - 2 = 14 GB
        assert_eq!(budget.training_budget_bytes, 14 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_budget_insufficient() {
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 8 * 1024 * 1024 * 1024;
        let kv: u64 = 12 * 1024 * 1024 * 1024;
        let budget = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.7, None);
        // Only 4GB available for training
        assert_eq!(budget.training_budget_bytes, 4 * 1024 * 1024 * 1024);
        // Requesting 8GB should fail
        assert!(
            budget
                .check_training_feasible(8 * 1024 * 1024 * 1024)
                .is_err()
        );
        // Requesting 3GB should succeed
        assert!(
            budget
                .check_training_feasible(3 * 1024 * 1024 * 1024)
                .is_ok()
        );
    }

    #[test]
    fn test_memory_budget_saturating_sub() {
        // Edge case: model + KV > total VRAM
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 20 * 1024 * 1024 * 1024;
        let kv: u64 = 10 * 1024 * 1024 * 1024;
        let budget = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.7, None);
        // Should not underflow — saturating_sub handles it
        assert_eq!(budget.training_budget_bytes, 0);
    }

    #[test]
    fn configured_training_budget_is_cap_only() {
        let gib = 1024 * 1024 * 1024;
        let capped =
            GpuMemoryBudget::compute(24 * gib, 8 * gib, 8 * gib, 0, 4 * gib, 0.7, Some(6.0));
        assert_eq!(capped.training_budget_bytes, 6 * gib);

        let optimistic =
            GpuMemoryBudget::compute(24 * gib, 8 * gib, 8 * gib, 0, 4 * gib, 0.7, Some(40.0));
        assert_eq!(optimistic.training_budget_bytes, 12 * gib);
    }

    #[test]
    fn test_estimate_model_memory() {
        let config = ModelConfig::qwen3_5_4b();
        let bytes = estimate_model_memory_bytes(&config);
        let gb = bytes as f64 / 1e9;
        // Should be in the ballpark of 8GB for Qwen3.5-4B bf16
        assert!(
            gb > 4.0 && gb < 20.0,
            "model estimate {gb:.1}GB seems wrong"
        );
    }

    #[test]
    fn test_post_load_residency_lowers_cuda_auto_blocks() {
        let total = 51_527_024_640;
        let estimated_model = 9_156_689_920;
        let post_load_residency = 13_000_000_000;
        let bytes_per_block = 524_288;

        let old_estimate_blocks = auto_num_blocks_for_fraction(
            total,
            estimated_model,
            bytes_per_block,
            0.7,
            262_144,
            DEFAULT_BLOCK_SIZE,
            KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
        );
        let post_load_blocks = auto_num_blocks_for_fraction(
            total,
            post_load_residency,
            bytes_per_block,
            0.7,
            262_144,
            DEFAULT_BLOCK_SIZE,
            KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
        );

        assert_eq!(old_estimate_blocks, 56570);
        assert!(
            post_load_blocks < old_estimate_blocks,
            "post-load residency must reduce the default A6000 KV budget: old={old_estimate_blocks} post_load={post_load_blocks}"
        );
    }

    #[test]
    fn active_backend_maps_to_one_device_scoped_memory_probe() {
        use kiln_memory::vram::{LinuxDrmVendor, VramProbeSelector};

        assert_eq!(
            vram_probe_selector_for_device(kiln_tensor::Device::Cuda(2)),
            VramProbeSelector::Nvidia(2)
        );
        assert_eq!(
            vram_probe_selector_for_device(kiln_tensor::Device::Rocm(1)),
            VramProbeSelector::LinuxDrm {
                index: 1,
                vendor: Some(LinuxDrmVendor::Amd),
            }
        );
        assert_eq!(
            vram_probe_selector_for_device(kiln_tensor::Device::Vulkan(3)),
            VramProbeSelector::LinuxDrm {
                index: 3,
                vendor: None,
            }
        );
        assert_eq!(
            vram_probe_selector_for_device(kiln_tensor::Device::Metal(0)),
            VramProbeSelector::AppleUnified
        );
        assert_eq!(
            vram_probe_selector_for_device(kiln_tensor::Device::Cpu),
            VramProbeSelector::None
        );
        assert_eq!(
            ensure_accelerator_memory_probe_identity(kiln_tensor::Device::Cpu).unwrap(),
            VramProbeSelector::None
        );
    }

    #[test]
    fn accelerator_capacity_guard_fails_closed_without_a_safe_probe() {
        use kiln_memory::vram::{GpuVramInfo, VramProbeSelector, VramSource};

        let missing = GpuVramInfo {
            total_bytes: 0,
            source: VramSource::None,
            unified: false,
        };
        let error = ensure_accelerator_memory_capacity(
            kiln_tensor::Device::Rocm(0),
            VramProbeSelector::LinuxDrm {
                index: 0,
                vendor: Some(kiln_memory::vram::LinuxDrmVendor::Amd),
            },
            missing,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("0 bytes of safe effective capacity"));
        assert!(error.contains("cap-only"));

        assert!(
            ensure_accelerator_memory_capacity(
                kiln_tensor::Device::Cpu,
                VramProbeSelector::None,
                missing,
            )
            .is_ok()
        );
        assert!(
            ensure_accelerator_memory_capacity(
                kiln_tensor::Device::Vulkan(0),
                VramProbeSelector::LinuxDrm {
                    index: 0,
                    vendor: None,
                },
                GpuVramInfo {
                    total_bytes: 8 * 1024 * 1024 * 1024,
                    source: VramSource::LinuxDrmSysfs,
                    unified: false,
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn accelerator_floor_guard_rejects_equal_or_larger_effective_capacity() {
        use kiln_memory::vram::{GpuVramInfo, VramSource};

        let capacity = GpuVramInfo {
            total_bytes: 8 * 1024 * 1024 * 1024,
            source: VramSource::LinuxDrmSysfs,
            unified: false,
        };
        let mut memory = crate::config::MemoryConfig::default();
        memory.floor_gb = 8.0;
        let error =
            ensure_accelerator_memory_floor(kiln_tensor::Device::Vulkan(0), capacity, &memory)
                .unwrap_err()
                .to_string();
        assert!(error.contains("memory.floor_gb=8"));
        assert!(error.contains("8589934592 bytes"));
        assert!(error.contains("strictly smaller"));
        assert!(error.contains("before model upload"));

        memory.floor_gb = 8.5;
        assert!(
            ensure_accelerator_memory_floor(kiln_tensor::Device::Rocm(0), capacity, &memory,)
                .is_err()
        );
        memory.floor_gb = 7.5;
        assert!(
            ensure_accelerator_memory_floor(kiln_tensor::Device::Rocm(0), capacity, &memory,)
                .is_ok()
        );
        assert!(
            ensure_accelerator_memory_floor(
                kiln_tensor::Device::Cpu,
                GpuVramInfo {
                    total_bytes: 0,
                    source: VramSource::None,
                    unified: false,
                },
                &memory,
            )
            .is_ok()
        );
    }

    #[test]
    fn live_budget_caps_auto_blocks_below_the_preferred_minimum() {
        const MIB: u64 = 1024 * 1024;

        let (capped, max_blocks) =
            cap_kv_blocks_to_live_budget(MIN_AUTO_KV_BLOCKS, MIB, 63 * MIB, None);
        assert_eq!(max_blocks, 63);
        assert_eq!(capped, 63);

        let (capped, max_blocks) = cap_kv_blocks_to_live_budget(128, MIB, 96 * MIB, Some(32 * MIB));
        assert_eq!(max_blocks, 32);
        assert_eq!(capped, 32);
    }

    #[test]
    fn explicit_portable_kv_allocation_obeys_governor_without_allocator_probe() {
        const MIB: u64 = 1024 * 1024;

        for policy in [
            KvCacheAutoBlockPolicy::for_backend("vulkan", kiln_tensor::Device::Vulkan(0)),
            KvCacheAutoBlockPolicy::for_backend("metal", kiln_tensor::Device::Metal(0)),
        ] {
            let error = validate_kv_allocation_against_live_budget(
                MIN_AUTO_KV_BLOCKS,
                MIB,
                63 * MIB,
                false,
                false,
                policy,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("memory governor budget"));
            assert!(error.contains("at most num_blocks=63"));
        }
    }

    #[test]
    fn rocm_minimum_policy_never_overrides_live_allocator_budget() {
        const MIB: u64 = 1024 * 1024;
        let policy = KvCacheAutoBlockPolicy::for_backend("rocm", kiln_tensor::Device::Rocm(0));
        assert!(policy.allow_min_blocks_below_live_budget);

        let error = validate_kv_allocation_against_live_budget(
            MIN_AUTO_KV_BLOCKS,
            MIB,
            63 * MIB,
            true,
            false,
            policy,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("stricter governor/allocator budget"));
        assert!(error.contains("preferred minimum block policy"));
    }

    #[test]
    fn explicit_vulkan_kv_allocation_reports_host_backed_ceiling() {
        const MIB: u64 = 1024 * 1024;
        let policy = KvCacheAutoBlockPolicy::for_backend("vulkan", kiln_tensor::Device::Vulkan(0));
        let error =
            validate_kv_allocation_against_live_budget(128, MIB, 32 * MIB, false, true, policy)
                .unwrap_err()
                .to_string();
        assert!(error.contains("governor/host-backed budget"));
        assert!(error.contains("at most num_blocks=32"));
        assert!(error.contains("free accelerator and host memory"));
    }

    #[test]
    fn test_auto_num_blocks_no_model_context_cap_on_cuda() {
        // On CUDA, raw_blocks comes from the memory-aware sizing path
        // (available VRAM × inference fraction ÷ bytes-per-block), which
        // already accounts for what the GPU can hold. Clipping it to a single
        // model-context-worth of blocks (≈16K for Qwen3.5-4B's 256K window)
        // would bottleneck concurrent serving — multiple in-flight long
        // prompts collectively address more than one window's worth of KV
        // cache. cap_auto_num_blocks must trust the memory-aware ceiling
        // here.
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
                48 * 1024 * 1024 * 1024,
            ),
            50_000
        );
        // raw_blocks well under the model-context size still passes through —
        // this is the small-VRAM CUDA path (e.g. A10 / consumer card).
        assert_eq!(
            cap_auto_num_blocks(
                4_096,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
                10 * 1024 * 1024 * 1024,
            ),
            4_096
        );
        // raw_blocks above the model-context size on a large-VRAM CUDA host
        // is preserved (multi-tenant headroom). Pre-fix this returned 16_384.
        assert_eq!(
            cap_auto_num_blocks(
                65_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
                80 * 1024 * 1024 * 1024,
            ),
            65_000
        );
    }

    #[test]
    fn test_auto_num_blocks_caps_rocm_defaults_to_context_pool() {
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::for_backend("rocm", kiln_tensor::Device::Rocm(0)),
                120 * 1024 * 1024 * 1024,
            ),
            4096
        );
    }

    #[test]
    fn test_auto_num_blocks_caps_metal_desktop_defaults_by_memory_tier() {
        // On unified-memory Macs, pure memory-aware sizing can request a large
        // eagerly-zeroed KV cache. Default Metal auto-sizing is tier-capped by
        // detected memory; explicit KILN_NUM_BLOCKS still bypasses this helper
        // entirely in AppState::new_real.
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::for_backend("metal", kiln_tensor::Device::Metal(0)),
                10 * 1024 * 1024 * 1024,
            ),
            512
        );
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::for_backend("metal", kiln_tensor::Device::Metal(0)),
                16 * 1024 * 1024 * 1024,
            ),
            1024
        );
        assert_eq!(
            cap_auto_num_blocks(
                50_000,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::for_backend("metal", kiln_tensor::Device::Metal(0)),
                32 * 1024 * 1024 * 1024,
            ),
            2048
        );
    }

    #[test]
    fn test_auto_num_blocks_preserves_measured_sub_minimum_budget() {
        assert_eq!(
            cap_auto_num_blocks(
                512,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::for_backend("metal", kiln_tensor::Device::Metal(0)),
                10 * 1024 * 1024 * 1024,
            ),
            512
        );
        assert_eq!(
            cap_auto_num_blocks(
                1,
                262_144,
                DEFAULT_BLOCK_SIZE,
                KvCacheAutoBlockPolicy::for_backend("metal", kiln_tensor::Device::Metal(0)),
                10 * 1024 * 1024 * 1024,
            ),
            1
        );
    }

    #[test]
    fn test_inference_fraction_clamping() {
        // Verify the clamping logic works (tested indirectly through budget)
        let total: u64 = 24 * 1024 * 1024 * 1024;
        let model: u64 = 8 * 1024 * 1024 * 1024;
        let kv: u64 = 2 * 1024 * 1024 * 1024;

        // fraction = 1.0 means all VRAM for inference, but training budget is still calculated
        let budget_full = GpuMemoryBudget::compute(total, model, model, 0, kv, 1.0, None);
        assert_eq!(budget_full.training_budget_bytes, 14 * 1024 * 1024 * 1024);

        // fraction = 0.5
        let budget_half = GpuMemoryBudget::compute(total, model, model, 0, kv, 0.5, None);
        assert_eq!(budget_half.training_budget_bytes, 14 * 1024 * 1024 * 1024);
    }

    /// Build a tiny PagedKvCacheKt for use as the "successful allocation"
    /// return value in the auto-sizer retry tests below. The values are
    /// dummies — only the act of returning Ok(...) matters for the loop
    /// logic.
    ///
    /// #1082 candle-drop: `PagedKvCache::new_uninit_with_fp8_kt(&Device, ..)`
    /// -> `PagedKvCacheKt::new_with_fp8(.., device, fp8)`. Now that the kt
    /// cache allocates on the runtime `Device`, this dummy passes
    /// `Device::Cpu` so it builds host-resident pools and runs on any test
    /// host (no CUDA/Metal device required).
    fn dummy_cpu_cache() -> PagedKvCacheKt {
        PagedKvCacheKt::new_with_fp8(
            1,  // num_full_attn_layers
            8,  // num_blocks
            16, // block_size
            1,  // num_kv_heads
            4,  // head_dim
            DType::F32,
            kiln_tensor::Device::Cpu,
            false,
        )
        .expect("PagedKvCacheKt allocation never fails for tiny shape")
    }

    #[test]
    fn auto_sizer_succeeds_on_first_attempt_when_configured_fits() {
        let compute = |fraction: f64| -> usize {
            // Map fraction directly to a block count for inspection
            (fraction * 1000.0) as usize
        };
        let calls = std::cell::Cell::new(0u32);
        let result = auto_size_with_retry(0.85, AUTO_SIZER_FALLBACK_FRACTIONS, &compute, |_n| {
            calls.set(calls.get() + 1);
            Ok(dummy_cpu_cache())
        });
        let success = result.unwrap_or_else(|_| panic!("expected success"));
        assert_eq!(success.fraction, 0.85);
        assert_eq!(success.num_blocks, 850);
        assert!(success.attempted_failures.is_empty());
        assert_eq!(calls.get(), 1, "should have allocated exactly once");
    }

    #[test]
    fn auto_sizer_retries_until_fraction_drops_below_oom_threshold() {
        // Simulate an A40+BF16-like OOM zone: anything ≥ 0.70 OOMs, anything
        // strictly below succeeds. Configured fraction is 0.85 (issue #685
        // shape); the loop should fall through 0.85 → 0.75 (both OOM) and
        // succeed at 0.65.
        let oom_at_or_above = 0.70_f64;
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let calls = std::cell::Cell::new(0u32);
        let attempted_fractions = std::cell::RefCell::new(Vec::<f64>::new());
        let result = auto_size_with_retry(0.85, AUTO_SIZER_FALLBACK_FRACTIONS, &compute, |n| {
            calls.set(calls.get() + 1);
            let frac = (n as f64) / 1000.0;
            attempted_fractions.borrow_mut().push(frac);
            if frac >= oom_at_or_above - 1e-9 {
                Err(format!(
                    "CUDA OOM: out of memory while allocating k_pool for layer 0 (n={n})"
                ))
            } else {
                Ok(dummy_cpu_cache())
            }
        });
        let success = result.unwrap_or_else(|_| panic!("expected success after fallback"));
        assert_eq!(success.fraction, 0.65, "should land on the 0.65 fallback");
        assert_eq!(success.num_blocks, 650);
        assert_eq!(
            success.attempted_failures.len(),
            2,
            "should have failed twice (0.85 then 0.75) before succeeding at 0.65"
        );
        assert_eq!(calls.get(), 3);
        let attempts = attempted_fractions.borrow().clone();
        assert_eq!(attempts, vec![0.85, 0.75, 0.65]);
    }

    #[test]
    fn auto_sizer_skips_fallbacks_above_configured_fraction() {
        // If the user pinned a low inference_memory_fraction (say 0.50), the
        // retry loop must never try the higher-default fallbacks (0.75, 0.65)
        // — that would silently allocate MORE than the user asked for.
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let calls = std::cell::Cell::new(0u32);
        let attempted = std::cell::RefCell::new(Vec::<f64>::new());
        let result = auto_size_with_retry(0.50, AUTO_SIZER_FALLBACK_FRACTIONS, &compute, |n| {
            calls.set(calls.get() + 1);
            let frac = (n as f64) / 1000.0;
            attempted.borrow_mut().push(frac);
            Ok(dummy_cpu_cache())
        });
        let success = result.unwrap_or_else(|_| panic!("expected success"));
        assert_eq!(success.fraction, 0.50);
        assert_eq!(success.num_blocks, 500);
        assert!(success.attempted_failures.is_empty());
        let attempts = attempted.borrow().clone();
        assert_eq!(
            attempts,
            vec![0.50],
            "should have tried only the configured fraction"
        );
    }

    #[test]
    fn auto_sizer_returns_failure_when_every_fraction_ooms() {
        // Pathological case: every fraction OOMs (e.g. unreasonably small GPU
        // for the model). The loop must not loop forever; it must return
        // Failure with the full attempt history so the caller can build a
        // useful error message.
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let calls = std::cell::Cell::new(0u32);
        let result = auto_size_with_retry(
            0.85,
            AUTO_SIZER_FALLBACK_FRACTIONS,
            &compute,
            |n| -> Result<PagedKvCacheKt, String> {
                calls.set(calls.get() + 1);
                Err(format!("simulated OOM at n={n}"))
            },
        );
        let failure = result.err().unwrap_or_else(|| panic!("expected failure"));
        // 1 configured + 4 fallbacks (all strictly below 0.85) = 5 attempts
        assert_eq!(failure.attempts.len(), 5);
        assert_eq!(calls.get(), 5);
        let fractions: Vec<f64> = failure.attempts.iter().map(|(f, _, _)| *f).collect();
        assert_eq!(fractions, vec![0.85, 0.75, 0.65, 0.55, 0.45]);
    }

    #[test]
    fn auto_sizer_does_not_retry_with_duplicate_or_higher_values() {
        // If configured fraction equals one of the fallback values (e.g. 0.75),
        // the retry loop must not try it twice. Subsequent attempts must be
        // strictly lower.
        let compute = |fraction: f64| -> usize { (fraction * 1000.0) as usize };
        let attempted = std::cell::RefCell::new(Vec::<f64>::new());
        let result = auto_size_with_retry(
            0.75,
            AUTO_SIZER_FALLBACK_FRACTIONS,
            &compute,
            |n| -> Result<PagedKvCacheKt, String> {
                let frac = (n as f64) / 1000.0;
                attempted.borrow_mut().push(frac);
                Err(format!("OOM at n={n}"))
            },
        );
        let failure = result.err().unwrap_or_else(|| panic!("expected failure"));
        let fractions: Vec<f64> = failure.attempts.iter().map(|(f, _, _)| *f).collect();
        assert_eq!(
            fractions,
            vec![0.75, 0.65, 0.55, 0.45],
            "configured 0.75 must appear once and only fractions strictly below should follow"
        );
    }

    #[test]
    fn suggested_emergency_blocks_uses_30pct_of_remaining_vram() {
        // 48 GiB GPU, 8 GiB model -> 40 GiB remaining * 0.30 = 12 GiB for KV
        let total = 48u64 * 1024 * 1024 * 1024;
        let model = 8u64 * 1024 * 1024 * 1024;
        let bytes_per_block = 256u64 * 1024; // 256 KiB per block
        let suggested = suggested_emergency_num_blocks(
            total,
            model,
            bytes_per_block,
            DEFAULT_BLOCK_SIZE,
            262_144,
            KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
        );
        let expected_kv_bytes = ((total - model) as f64 * 0.30) as u64;
        let expected_blocks = (expected_kv_bytes / bytes_per_block) as usize;
        assert_eq!(suggested, expected_blocks);
        // Sanity: at least the floor
        assert!(suggested >= MIN_AUTO_KV_BLOCKS);
    }

    #[test]
    fn suggested_emergency_blocks_falls_back_when_no_vram_signal() {
        // total_vram = 0 (CPU or detection failed) — must still return a
        // sensible block count derived from max_position_embeddings.
        let suggested = suggested_emergency_num_blocks(
            0, // total_vram unknown
            0,
            0,
            DEFAULT_BLOCK_SIZE,
            262_144,
            KvCacheAutoBlockPolicy::MEMORY_BUDGET_ONLY,
        );
        let expected = (262_144_usize).div_ceil(DEFAULT_BLOCK_SIZE);
        assert_eq!(suggested, expected.max(MIN_AUTO_KV_BLOCKS));
    }

    #[test]
    fn oom_message_names_concrete_remediation_flags() {
        // The whole point of the new error message: it must give the user a
        // concrete `KILN_NUM_BLOCKS=N` and `KILN_INFERENCE_MEMORY_FRACTION=X`
        // value to set. Verify both appear in the rendered text.
        let failure = AutoSizeFailure {
            attempts: vec![
                (0.85, 24576, "CUDA OOM during k_pool layer 0".to_string()),
                (0.75, 21504, "CUDA OOM during k_pool layer 0".to_string()),
                (0.65, 18432, "CUDA OOM during k_pool layer 0".to_string()),
                (0.55, 15360, "CUDA OOM during k_pool layer 0".to_string()),
                (0.45, 12288, "CUDA OOM during k_pool layer 0".to_string()),
            ],
        };
        let total_vram = 48u64 * 1024 * 1024 * 1024;
        let model_bytes = 8u64 * 1024 * 1024 * 1024;
        let bytes_per_block = 1u64 * 1024 * 1024; // 1 MiB
        let suggested = 8192;
        let msg = format_oom_remediation_message(
            &failure,
            total_vram,
            model_bytes,
            bytes_per_block,
            suggested,
            0.85,
            kiln_memory::vram::VramSource::NvidiaSmi,
        );
        assert!(
            msg.contains("KILN_NUM_BLOCKS=8192"),
            "message must include the concrete num_blocks suggestion: {msg}"
        );
        assert!(
            msg.contains("KILN_INFERENCE_MEMORY_FRACTION="),
            "message must include the concrete fraction suggestion: {msg}"
        );
        // Suggested fraction = last attempt (0.45) / 2 = 0.225, max(0.10) = 0.225
        assert!(
            msg.contains("0.23") || msg.contains("0.22"),
            "message should suggest a fraction roughly half of the last failure: {msg}"
        );
        assert!(
            msg.contains("48.0 GiB") || msg.contains("48 GiB"),
            "message should mention detected VRAM: {msg}"
        );
        assert!(
            msg.contains("nvidia-smi"),
            "message should mention VRAM source for sanity-check: {msg}"
        );
        // All 5 attempted fractions should be enumerated
        for fraction_str in &["0.85", "0.75", "0.65", "0.55", "0.45"] {
            assert!(
                msg.contains(fraction_str),
                "message should enumerate attempt at fraction {fraction_str}: {msg}"
            );
        }
        // The recommendation banner must reference the working workaround
        assert!(
            msg.contains("#685"),
            "message should reference issue #685 for context: {msg}"
        );
    }

    #[test]
    fn oom_message_handles_unknown_vram() {
        // total_vram = 0 path (e.g. detection failed) — the message must
        // still render something sensible without panicking on the GiB
        // formatting.
        let failure = AutoSizeFailure {
            attempts: vec![(0.85, 100, "CUDA OOM".to_string())],
        };
        let msg = format_oom_remediation_message(
            &failure,
            0,
            0,
            1024,
            64,
            0.85,
            kiln_memory::vram::VramSource::None,
        );
        assert!(msg.contains("KILN_NUM_BLOCKS=64"), "message: {msg}");
        assert!(
            msg.contains("0.0 GiB"),
            "should print 0.0 GiB when unknown: {msg}"
        );
    }
}
