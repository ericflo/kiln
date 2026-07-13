//! HIP (ROCm) graph capture and replay for decode forward passes — the ROCm
//! twin of [`crate::cuda_graph`] (R.9).
//!
//! During decode each step processes exactly one token with identical tensor
//! shapes, so the kernel sequence is captured once (`hipStreamBeginCapture`)
//! and replayed (`hipGraphLaunch`) to eliminate per-step host launch overhead.
//! The machinery mirrors `CudaGraphRunner` and rests on the R.9 foundations:
//!   * `kiln_tensor::rocm_write_host_in_place` — refresh per-step inputs through
//!     a graph-stable device pointer (Phase 1).
//!   * `kiln_tensor::RocmCaptureArena` — freeze every activation pointer the
//!     captured forward touches across capture→replay (Phase 2).
//!   * the `any(cuda,rocm)`-gated graph-inputs forward in `forward.rs`, whose kt
//!     consumption arms already dispatch to the ROCm SDPA / paged-KV kernels
//!     (Phase 3B.1).
//!
//! Like the CUDA path, the captured forward stops at the PRE-final-norm hidden
//! (`LmHeadMode::HiddenOnly`); `final_norm` + lm_head run EAGERLY off the graph
//! (`lm_head_from_hidden_eager`) because the lm_head GEMV is not replay-safe.
//!
//! Enabled by default on Rocm devices; set `KILN_ROCM_GRAPHS=0` to force eager
//! decode. `KILN_ROCM_GRAPH_CAPTURE=0` is a narrower capture opt-out. Capture
//! or replay failures trip a runner-local circuit breaker and fall back to eager.

use anyhow::{Context, Result};
use serde::Serialize;
use tracing;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;

use crate::PagedKvCacheKt;
use crate::backend::BackendRuntime;
use crate::forward::{
    GpuWeights, LinearAttentionState, model_forward_paged,
    model_forward_paged_batched_decode_hidden, model_forward_paged_next_token_greedy,
};
use crate::lora_loader::LoraWeights;

#[cfg(feature = "rocm")]
use crate::forward::{PagedDecodeGraphInputs, model_forward_paged_hidden_with_graph_inputs};
#[cfg(feature = "rocm")]
use kiln_graph::{
    CaptureError, InvalidateReason, ReplayInputs, ReplayKey, ReplayOutputs, ReplayPlan,
    ReplayResourceStability, ReplayState, ResidentResourceRef,
};
#[cfg(feature = "rocm")]
use std::collections::HashMap;

#[cfg(feature = "rocm")]
use kiln_tensor::Backend;
use kiln_tensor::{Device, Tensor};

#[cfg(all(test, feature = "rocm"))]
static ROCM_TEST_NATIVE_REPLAY_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "rocm")]
fn attributed_rocm_graph_synchronize(
    phase: &'static str,
    reason: &'static str,
    affected_bytes: u64,
    byte_scope: &'static str,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let started = std::time::Instant::now();
    let result = synchronize();
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    match &result {
        Ok(()) => tracing::info!(
            event = "gpu_memory_operation",
            operation = "synchronize",
            reason,
            outcome = "completed",
            phase,
            affected_bytes,
            byte_scope,
            wait_ms = duration_ms,
            duration_ms,
            "ROCm graph host synchronization completed"
        ),
        Err(error) => tracing::warn!(
            event = "gpu_memory_operation",
            operation = "synchronize",
            reason,
            outcome = "failed",
            phase,
            affected_bytes,
            byte_scope,
            error = %format!("{error:#}"),
            wait_ms = duration_ms,
            duration_ms,
            "ROCm graph host synchronization failed"
        ),
    }
    result
}

#[cfg(feature = "rocm")]
fn graph_tensor_bytes<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    tensors.into_iter().fold(0u64, |total, tensor| {
        total.saturating_add(
            (tensor.element_count() as u64).saturating_mul(tensor.dtype().size_in_bytes() as u64),
        )
    })
}

#[cfg(feature = "rocm")]
fn synchronize_after_rocm_graph_capture_failure(device: &Device) {
    let Some(device_idx) = device.index() else {
        return;
    };
    if let Err(error) = attributed_rocm_graph_synchronize(
        "failure_recovery_default_stream",
        "rocm_graph_capture_failure_recovery",
        0,
        "unknown_in_flight_device_work",
        || {
            kiln_tensor::rocm_synchronize_default_stream(device_idx)
                .map_err(|error| anyhow::anyhow!("{error}"))
        },
    ) {
        tracing::warn!("post-capfail default-stream sync failed: {error:#}");
    }
}

/// Whether ROCm HIP-graph decode is requested via `KILN_ROCM_GRAPHS` (default
/// ON). This is the primary runtime gate; `KILN_ROCM_GRAPH_CAPTURE` is the
/// narrower capture opt-out described below.
fn rocm_graphs_env_on() -> bool {
    std::env::var("KILN_ROCM_GRAPHS")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "off" | "OFF"))
        .unwrap_or(true)
}

/// Whether to ATTEMPT capture/replay (vs. eager past warmup). Default ON
/// whenever the runner is enabled — mirrors CUDA, where the single
/// `KILN_CUDA_GRAPHS` flag turns on capture directly (no separate sub-flag).
///
/// This is only ever consulted AFTER the runner is confirmed enabled. Capture
/// is fully working: the paged-KV slot write is on-device
/// (`index_copy.cu` scatter with a DEVICE slot index), the freeze-pointer arena
/// keeps every activation pointer stable across capture→replay, and the warm
/// pass's host-round-trip detector condemns only geometries with a PERSISTENT
/// host copy (cold shape-cache fills are retried, see `CAPTURE_RETRY_LIMIT`).
/// With stable paged metadata (default on) one graph is captured per
/// `max_seqlen_k` bucket and replayed for every step in it; captured-graph
/// greedy decode is byte-identical to eager and never slower (gfx1151,
/// validated across bucket-crossing long gen + concurrency, zero failures).
/// Set `KILN_ROCM_GRAPH_CAPTURE=0` to opt OUT (eager past warmup) — e.g. to
/// A/B the launch-overhead win or isolate a capture regression.
#[cfg(feature = "rocm")]
fn rocm_graph_capture_supported() -> bool {
    std::env::var("KILN_ROCM_GRAPH_CAPTURE")
        .map(|v| !matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "off" | "OFF"))
        .unwrap_or(true)
}

/// Cache key for a captured bs=1 decode graph. Mirrors `CudaGraphKey`: with
/// stable paged metadata (default on) the block table / seq_len are refreshed
/// in place each replay, so they don't enter the key — only the FA2-bucketed
/// K/V geometry does.
#[cfg(feature = "rocm")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RocmGraphKey {
    stable_metadata: bool,
    seq_len: usize,
    block_table: Vec<u32>,
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
}

#[cfg(feature = "rocm")]
impl RocmGraphKey {
    fn exact_max_seqlen_k(attention_len: usize) -> usize {
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        attention_len.div_ceil(kblock_n) * kblock_n
    }

    fn new(block_table: &BlockTable, paged_cache: &PagedKvCacheKt, seq_len: usize) -> Self {
        let stable_metadata = Self::stable_paged_metadata_enabled();
        let attention_len = seq_len + 1;
        // Match eager's exact FA2 split geometry. A coarser graph bucket changes
        // the number of split-K partials and can perturb BF16 reductions even
        // when `seqused_k` masks the same logical K/V length.
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let max_seqlen_k = Self::exact_max_seqlen_k(attention_len);
        let pages_per_chunk = kblock_n / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        Self {
            stable_metadata,
            seq_len: if stable_metadata { 0 } else { seq_len },
            block_table: if stable_metadata {
                Vec::new()
            } else {
                block_table.blocks.clone()
            },
            max_seqlen_k,
            max_blocks_per_seq,
        }
    }

    /// Default ON. The graph-stable block-table buffer is refreshed in place per
    /// replay, so it reads the CURRENT table and never races block recycling.
    /// Opt out with `KILN_ROCM_GRAPH_STABLE_PAGED_METADATA=0`.
    fn stable_paged_metadata_enabled() -> bool {
        std::env::var("KILN_ROCM_GRAPH_STABLE_PAGED_METADATA")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on" | "ON"))
            .unwrap_or(true)
    }
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum RocmGraphOwner {
    Slot(u64),
}

#[cfg(feature = "rocm")]
impl RocmGraphOwner {
    fn slot_id(self) -> u64 {
        match self {
            Self::Slot(slot_id) => slot_id,
        }
    }
}

#[cfg(feature = "rocm")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RocmGraphCacheKey {
    owner: RocmGraphOwner,
    graph: RocmGraphKey,
}

#[cfg(feature = "rocm")]
impl RocmGraphCacheKey {
    fn new(owner: RocmGraphOwner, graph: RocmGraphKey) -> Self {
        Self { owner, graph }
    }
}

#[cfg(feature = "rocm")]
#[derive(Default)]
struct RocmGraphOwnerTimeline {
    last_decode_seq_len: Option<usize>,
    last_decode_block0: Option<u32>,
}

#[cfg(feature = "rocm")]
struct RocmGraphSlotState {
    assigned_row: Option<u64>,
    linear_state: LinearAttentionState,
}

/// A captured HIP graph ready for replay, plus every graph-stable buffer whose
/// device pointer the graph baked in. Mirrors `CapturedDecodeGraph`.
#[cfg(feature = "rocm")]
struct CapturedDecodeGraphRocm {
    /// The source graph — retained because dropping it `hipGraphDestroy`s the
    /// handle; the exec is launched, the graph is kept alive alongside it.
    _graph: kiln_hip::RocmGraph,
    /// The instantiated, launchable graph. ROCm uses plain instantiation
    /// (`flags = 0`); auto-free was rejected on gfx1151 / ROCm 7.2.4.
    exec: kiln_hip::RocmGraphExec,
    /// Graph-stable PRE-final-norm hidden `[1, 1, hidden]`; refreshed in place by
    /// the captured forward, read eagerly by lm_head after launch.
    output_hidden: Tensor,
    /// The non-default capture stream the graph launches on. Replay completion
    /// is ordered into `default_stream` without blocking the host.
    capture_stream: std::sync::Arc<kiln_hip::RocmStream>,
    /// The kt default stream that receives refreshed inputs and consumes the
    /// graph-stable hidden output.
    default_stream: std::sync::Arc<kiln_hip::RocmStream>,
    /// Reusable default-to-capture dependency for refreshed replay inputs.
    replay_inputs_ready_event: std::sync::Arc<kiln_hip::RocmEvent>,
    /// Reusable capture-to-default dependency for replayed hidden output.
    replay_complete_event: std::sync::Arc<kiln_hip::RocmEvent>,
    adapter_gen: u64,
    /// Exact paged-KV allocation/generation whose pool pointers are embedded in
    /// the captured kernels.
    kv_pool_identity: crate::KvPoolIdentity,
    token_buffer: Tensor,
    position_buffer: Tensor,
    block_table_buffer: Option<Tensor>,
    seqused_k_buffer: Option<Tensor>,
    kv_slot_buffer: Option<Tensor>,
    rotary_cos_buffer: Tensor,
    rotary_sin_buffer: Tensor,
    _paged_decode_outputs: Vec<Tensor>,
    _paged_decode_lse: Vec<Tensor>,
    max_seqlen_k: usize,
    _gdn_decode_outputs: Vec<Tensor>,
    /// The freeze-pointer arena buffers (Q/K/V/activations); retained so their
    /// device pointers stay mapped for every replay.
    _capture_arena_buffers: Vec<std::sync::Arc<kiln_tensor::RocmStorage>>,
    /// Shared graph-layer replay contract state captured alongside the native
    /// HIP graph. The production replay path validates this before launching.
    replay_state: ReplayState,
}

#[cfg(feature = "rocm")]
struct RocmDecodeReplayPlan<'a> {
    captured: &'a CapturedDecodeGraphRocm,
}

#[cfg(feature = "rocm")]
impl<'a> RocmDecodeReplayPlan<'a> {
    fn new(captured: &'a CapturedDecodeGraphRocm) -> Self {
        Self { captured }
    }
}

#[cfg(feature = "rocm")]
impl std::fmt::Debug for RocmDecodeReplayPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmDecodeReplayPlan")
            .field("key", &self.captured.replay_state.key)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "rocm")]
impl ReplayPlan for RocmDecodeReplayPlan<'_> {
    fn backend(&self) -> Backend {
        Backend::Rocm
    }

    fn key(&self) -> ReplayKey {
        self.captured.replay_state.key.clone()
    }

    fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError> {
        self.captured
            .replay_state
            .validate(inputs.key, inputs.resources)
    }

    fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError> {
        self.validate_inputs(inputs)?;
        #[cfg(test)]
        ROCM_TEST_NATIVE_REPLAY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.captured
            .exec
            .launch(&self.captured.capture_stream)
            .map_err(|e| CaptureError::Backend(format!("ROCm graph launch: {e}")))?;
        self.captured
            .capture_stream
            .record_event(&self.captured.replay_complete_event)
            .map_err(|e| {
                CaptureError::Backend(format!("record ROCm graph replay completion: {e}"))
            })?;
        self.captured
            .default_stream
            .wait_event(&self.captured.replay_complete_event)
            .map_err(|e| {
                CaptureError::Backend(format!("order ROCm graph replay output handoff: {e}"))
            })?;
        Ok(ReplayOutputs::new(inputs.resources.to_vec(), 1))
    }

    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
        self.captured
            .replay_state
            .invalidate_reason(&state.key, &state.inputs)
    }
}

#[cfg(feature = "rocm")]
enum RocmCaptureStep {
    CapturedHidden(Tensor),
    FallbackEager(RocmGraphFallbackReason),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RocmGraphCounters {
    capture_attempts: u64,
    capture_successes: u64,
    capture_deferrals: u64,
    capture_failures: u64,
    replay_attempts: u64,
    replay_successes: u64,
    replay_failures: u64,
    decode_owner_release_count: u64,
    decode_owner_graph_release_count: u64,
    graph_slot_create_count: u64,
    graph_slot_reuse_count: u64,
    fallbacks: RocmGraphFallbackStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphCaptureOutcome {
    Succeeded,
    Deferred,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphFallbackReason {
    WarmupForwardFailure,
    ColdCacheHostRoundTrip,
    PersistentHostRoundTrip,
    ShapeDependentAttention,
    GraphCacheCapacity,
    CriticalMemoryPressure,
    CaptureFailure,
    ReplayFailure,
}

impl RocmGraphFallbackReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::WarmupForwardFailure => "warmup_forward_failure",
            Self::ColdCacheHostRoundTrip => "cold_cache_host_round_trip",
            Self::PersistentHostRoundTrip => "persistent_host_round_trip",
            Self::ShapeDependentAttention => "shape_dependent_attention",
            Self::GraphCacheCapacity => "graph_cache_capacity",
            Self::CriticalMemoryPressure => "critical_memory_pressure",
            Self::CaptureFailure => "capture_failure",
            Self::ReplayFailure => "replay_failure",
        }
    }
}

/// Bounded ROCm graph fallback counts and end-to-end eager-fallback latency.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RocmGraphFallbackStats {
    pub total: u64,
    pub warmup_forward_failure: u64,
    pub cold_cache_host_round_trip: u64,
    pub persistent_host_round_trip: u64,
    pub shape_dependent_attention: u64,
    pub graph_cache_capacity: u64,
    pub critical_memory_pressure: u64,
    pub capture_failure: u64,
    pub replay_failure: u64,
    pub slow: u64,
    pub total_duration_micros: u64,
    pub max_duration_micros: u64,
}

impl RocmGraphFallbackStats {
    const SLOW_DURATION: std::time::Duration = std::time::Duration::from_millis(100);

    fn record(&mut self, reason: RocmGraphFallbackReason, duration: std::time::Duration) -> u64 {
        self.total = self.total.saturating_add(1);
        let reason_count = match reason {
            RocmGraphFallbackReason::WarmupForwardFailure => &mut self.warmup_forward_failure,
            RocmGraphFallbackReason::ColdCacheHostRoundTrip => &mut self.cold_cache_host_round_trip,
            RocmGraphFallbackReason::PersistentHostRoundTrip => {
                &mut self.persistent_host_round_trip
            }
            RocmGraphFallbackReason::ShapeDependentAttention => &mut self.shape_dependent_attention,
            RocmGraphFallbackReason::GraphCacheCapacity => &mut self.graph_cache_capacity,
            RocmGraphFallbackReason::CriticalMemoryPressure => &mut self.critical_memory_pressure,
            RocmGraphFallbackReason::CaptureFailure => &mut self.capture_failure,
            RocmGraphFallbackReason::ReplayFailure => &mut self.replay_failure,
        };
        *reason_count = reason_count.saturating_add(1);
        let occurrence = *reason_count;
        let duration_micros = duration.as_micros().min(u64::MAX as u128) as u64;
        self.total_duration_micros = self.total_duration_micros.saturating_add(duration_micros);
        self.max_duration_micros = self.max_duration_micros.max(duration_micros);
        if duration >= Self::SLOW_DURATION {
            self.slow = self.slow.saturating_add(1);
        }
        occurrence
    }
}

impl RocmGraphCounters {
    fn record_capture_outcome(&mut self, outcome: RocmGraphCaptureOutcome) {
        self.capture_attempts = self.capture_attempts.saturating_add(1);
        match outcome {
            RocmGraphCaptureOutcome::Succeeded => {
                self.capture_successes = self.capture_successes.saturating_add(1);
            }
            RocmGraphCaptureOutcome::Deferred => {
                self.capture_deferrals = self.capture_deferrals.saturating_add(1);
            }
            RocmGraphCaptureOutcome::Failed => {
                self.capture_failures = self.capture_failures.saturating_add(1);
            }
        }
    }

    fn record_replay_outcome(&mut self, succeeded: bool) {
        self.replay_attempts = self.replay_attempts.saturating_add(1);
        if succeeded {
            self.replay_successes = self.replay_successes.saturating_add(1);
        } else {
            self.replay_failures = self.replay_failures.saturating_add(1);
        }
    }

    fn record_decode_owner_release(&mut self, released_graphs: usize) {
        self.decode_owner_release_count = self.decode_owner_release_count.saturating_add(1);
        self.decode_owner_graph_release_count = self
            .decode_owner_graph_release_count
            .saturating_add(released_graphs as u64);
    }

    fn record_graph_slot_create(&mut self) {
        self.graph_slot_create_count = self.graph_slot_create_count.saturating_add(1);
    }

    fn record_graph_slot_reuse(&mut self) {
        self.graph_slot_reuse_count = self.graph_slot_reuse_count.saturating_add(1);
    }

    fn record_fallback(
        &mut self,
        reason: RocmGraphFallbackReason,
        duration: std::time::Duration,
    ) -> u64 {
        self.fallbacks.record(reason, duration)
    }
}

/// Point-in-time ROCm HIP-graph execution state.
///
/// The attempt/success/failure fields are monotonic for the lifetime of the
/// runner, including across adapter invalidations. The graph and owner counts
/// are live gauges and may decrease when state is released or invalidated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RocmGraphStats {
    /// The ROCm device and primary `KILN_ROCM_GRAPHS` gate requested the graph
    /// runner when it was constructed. This remains true after a circuit break.
    pub requested: bool,
    /// The narrower `KILN_ROCM_GRAPH_CAPTURE` gate requested native capture.
    pub capture_requested: bool,
    /// The runner is still armed. Capture or replay failures set this false.
    pub enabled: bool,
    /// Native capture is both requested and currently armed.
    pub capture_enabled: bool,
    /// Calls that reached the native capture state machine.
    pub capture_attempts: u64,
    /// Graphs successfully instantiated, launched once, and retained.
    pub capture_successes: u64,
    /// Attempts deferred before native capture, such as cold-cache warm passes.
    pub capture_deferrals: u64,
    /// Capture attempts that returned an error and tripped the circuit breaker.
    pub capture_failures: u64,
    /// Native replay launches attempted from any decode API.
    pub replay_attempts: u64,
    /// Native replay launches whose input/output stream dependencies and launch
    /// were queued successfully. External-yield settlement confirms device
    /// completion before progress is published.
    pub replay_successes: u64,
    /// Native replay launches that failed and tripped the circuit breaker.
    pub replay_failures: u64,
    /// Saturating sum of capture and replay failures.
    pub failures: u64,
    /// Finished decode owners whose tracked timeline or graph state was removed.
    pub decode_owner_release_count: u64,
    /// Captured graphs evicted by finished-owner cleanup.
    pub decode_owner_graph_release_count: u64,
    /// Persistent graph slots created over the runner lifetime.
    pub graph_slot_create_count: u64,
    /// Finished slots rebound to a later logical decode row.
    pub graph_slot_reuse_count: u64,
    /// Graphs currently retained in the live cache.
    pub captured_graph_count: usize,
    /// Persistent recurrent-state slots currently retained.
    pub graph_slot_count: usize,
    /// Retained slots currently assigned to live logical decode rows.
    pub active_graph_slot_count: usize,
    /// Retained slots available for a later logical decode row.
    pub idle_graph_slot_count: usize,
    /// Decode owners whose continuity timeline is currently retained.
    pub tracked_decode_owner_count: usize,
    /// Closed-reason eager fallback counts and latency.
    pub fallbacks: RocmGraphFallbackStats,
}

/// Runs decode steps through captured HIP graphs when enabled, falling back to
/// eager execution otherwise. ROCm analog of `CudaGraphRunner`.
pub struct RocmGraphRunner {
    requested: bool,
    capture_requested: bool,
    enabled: bool,
    counters: RocmGraphCounters,
    adapter_generation: u64,
    warmup_done: bool,
    #[cfg(feature = "rocm")]
    captured: HashMap<RocmGraphCacheKey, CapturedDecodeGraphRocm>,
    /// Stable recurrent/conv buffers captured by each graph slot. This field is
    /// declared after `captured` so native graphs are destroyed before their
    /// state buffers when the runner is dropped or invalidated.
    #[cfg(feature = "rocm")]
    graph_slots: HashMap<RocmGraphOwner, RocmGraphSlotState>,
    #[cfg(feature = "rocm")]
    decode_row_slots: HashMap<u64, RocmGraphOwner>,
    #[cfg(feature = "rocm")]
    next_graph_slot_id: u64,
    /// Geometries whose decode forward is not capture-safe and the typed eager
    /// fallback reason to reuse on subsequent steps. This includes persistent
    /// host round-trips and attention paths whose tensor shapes depend on the
    /// current sequence length.
    #[cfg(feature = "rocm")]
    non_capture_safe: std::collections::HashMap<RocmGraphKey, RocmGraphFallbackReason>,
    /// Per-geometry count of consecutive capture attempts whose warm pass did a
    /// host round-trip. The first attempt in a `max_seqlen_k` bucket fills the
    /// shape-keyed global caches (broadcast gather indices, gqa-expand indices)
    /// with a one-time host upload; the NEXT attempt in the same (bucket-stable)
    /// geometry finds them warm and captures cleanly. So we don't condemn a
    /// geometry to `non_capture_safe` on the first htod bump — we retry up to
    /// `CAPTURE_RETRY_LIMIT` times and only give up if the round-trip persists
    /// (a genuine per-step fallback, not a cold cache).
    #[cfg(feature = "rocm")]
    capture_retry: std::collections::HashMap<RocmGraphKey, u32>,
    #[cfg(feature = "rocm")]
    cache_full_warned: bool,
    /// Per-owner request-boundary detection. Captured bs=1 graphs carry GDN
    /// recurrent/conv state in their own buffers, evolved in place across that
    /// owner's replays; sharing one timeline across interleaved streaming rows
    /// lets unrelated requests evict or reuse each other's graph state.
    #[cfg(feature = "rocm")]
    decode_timelines: HashMap<RocmGraphOwner, RocmGraphOwnerTimeline>,
}

impl RocmGraphRunner {
    /// Construct a runner for `device`. Enabled only when `enabled`, the device
    /// is `Device::Rocm`, and `KILN_ROCM_GRAPHS` is not set to an off value.
    pub fn new(device: &Device, enabled: bool) -> Self {
        let is_rocm = matches!(device, Device::Rocm(_));
        let requested = enabled && is_rocm && rocm_graphs_env_on();
        #[cfg(feature = "rocm")]
        let capture_requested = requested && rocm_graph_capture_supported();
        #[cfg(not(feature = "rocm"))]
        let capture_requested = false;
        if requested && capture_requested {
            tracing::info!("ROCm HIP graphs enabled for decode");
        } else if requested {
            tracing::info!(
                "ROCm graph runner requested but native capture disabled by KILN_ROCM_GRAPH_CAPTURE; using eager decode after warmup"
            );
        } else if enabled && is_rocm {
            tracing::debug!("ROCm device present but ROCm graphs disabled — using eager decode");
        }
        Self {
            requested,
            capture_requested,
            enabled: requested,
            counters: RocmGraphCounters::default(),
            adapter_generation: 0,
            warmup_done: false,
            #[cfg(feature = "rocm")]
            captured: HashMap::new(),
            #[cfg(feature = "rocm")]
            graph_slots: HashMap::new(),
            #[cfg(feature = "rocm")]
            decode_row_slots: HashMap::new(),
            #[cfg(feature = "rocm")]
            next_graph_slot_id: 1,
            #[cfg(feature = "rocm")]
            non_capture_safe: std::collections::HashMap::new(),
            #[cfg(feature = "rocm")]
            capture_retry: std::collections::HashMap::new(),
            #[cfg(feature = "rocm")]
            cache_full_warned: false,
            #[cfg(feature = "rocm")]
            decode_timelines: HashMap::new(),
        }
    }

    /// Max consecutive capture attempts (per bucket-stable geometry) whose warm
    /// pass may do a host round-trip before the geometry is condemned to
    /// `non_capture_safe`. A cold shape-cache fill clears in one pass, so the
    /// 2nd attempt normally captures; the budget gives margin for multiple
    /// distinct caches warming across a step or two.
    #[cfg(feature = "rocm")]
    const CAPTURE_RETRY_LIMIT: u32 = 3;

    /// Whether captured-graph decode is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Return configuration, circuit-breaker state, and lifetime execution
    /// counters without resetting them.
    pub fn stats(&self) -> RocmGraphStats {
        #[cfg(feature = "rocm")]
        let captured_graph_count = self.captured.len();
        #[cfg(not(feature = "rocm"))]
        let captured_graph_count = 0;
        #[cfg(feature = "rocm")]
        let tracked_decode_owner_count = self.decode_timelines.len();
        #[cfg(not(feature = "rocm"))]
        let tracked_decode_owner_count = 0;
        #[cfg(feature = "rocm")]
        let graph_slot_count = self.graph_slots.len();
        #[cfg(not(feature = "rocm"))]
        let graph_slot_count = 0;
        #[cfg(feature = "rocm")]
        let active_graph_slot_count = self
            .graph_slots
            .values()
            .filter(|slot| slot.assigned_row.is_some())
            .count();
        #[cfg(not(feature = "rocm"))]
        let active_graph_slot_count = 0;

        RocmGraphStats {
            requested: self.requested,
            capture_requested: self.capture_requested,
            enabled: self.enabled,
            capture_enabled: self.capture_requested && self.enabled,
            capture_attempts: self.counters.capture_attempts,
            capture_successes: self.counters.capture_successes,
            capture_deferrals: self.counters.capture_deferrals,
            capture_failures: self.counters.capture_failures,
            replay_attempts: self.counters.replay_attempts,
            replay_successes: self.counters.replay_successes,
            replay_failures: self.counters.replay_failures,
            failures: self
                .counters
                .capture_failures
                .saturating_add(self.counters.replay_failures),
            decode_owner_release_count: self.counters.decode_owner_release_count,
            decode_owner_graph_release_count: self.counters.decode_owner_graph_release_count,
            graph_slot_create_count: self.counters.graph_slot_create_count,
            graph_slot_reuse_count: self.counters.graph_slot_reuse_count,
            captured_graph_count,
            graph_slot_count,
            active_graph_slot_count,
            idle_graph_slot_count: graph_slot_count.saturating_sub(active_graph_slot_count),
            tracked_decode_owner_count,
            fallbacks: self.counters.fallbacks,
        }
    }

    /// Invalidate any captured graphs (LoRA swap changes weight pointers) and
    /// force a fresh warmup.
    pub fn invalidate(&mut self) {
        self.adapter_generation += 1;
        self.warmup_done = false;
        #[cfg(feature = "rocm")]
        {
            self.captured.clear();
            self.graph_slots.clear();
            self.decode_row_slots.clear();
            self.non_capture_safe.clear();
            self.capture_retry.clear();
            self.cache_full_warned = false;
            self.decode_timelines.clear();
        }
    }

    /// Return a finished logical decode row's graph slot to the bounded reuse
    /// pool. Native graphs and the exact recurrent-state buffers they captured
    /// remain resident until explicit runner invalidation; a graphless slot is
    /// discarded immediately.
    pub fn release_decode_row(&mut self, row_id: u64) {
        #[cfg(feature = "rocm")]
        {
            let Some(owner) = self.decode_row_slots.remove(&row_id) else {
                return;
            };
            let removed_timeline = self.decode_timelines.remove(&owner).is_some();
            let retained_graphs = self
                .captured
                .keys()
                .filter(|key| key.owner == owner)
                .count();
            if retained_graphs == 0 {
                self.graph_slots.remove(&owner);
            } else if let Some(slot) = self.graph_slots.get_mut(&owner) {
                slot.assigned_row = None;
            }
            self.counters.record_decode_owner_release(0);
            tracing::debug!(
                event = "rocm_graph_decode_owner_released",
                row_id,
                graph_slot_id = owner.slot_id(),
                retained_graphs,
                removed_timeline,
                graph_slot_count = self.graph_slots.len(),
                active_graph_slot_count = self.decode_row_slots.len(),
                "rocm_graph_decode_owner_released"
            );
        }
        #[cfg(not(feature = "rocm"))]
        let _ = row_id;
    }

    #[cfg(feature = "rocm")]
    fn prepare_owner_decode(
        &mut self,
        owner: RocmGraphOwner,
        row_id: u64,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> bool {
        let block0 = block_table.blocks.first().copied();
        let owner_started = !self.decode_timelines.contains_key(&owner);
        if owner_started {
            tracing::debug!(
                event = "rocm_graph_decode_owner_started",
                row_id,
                graph_slot_id = owner.slot_id(),
                seq_len,
                block0 = block0.unwrap_or_default(),
                block0_present = block0.is_some(),
                "rocm_graph_decode_owner_started"
            );
        }
        let timeline = self.decode_timelines.entry(owner).or_default();
        let continues = block0.is_some()
            && timeline.last_decode_seq_len == Some(seq_len.wrapping_sub(1))
            && timeline.last_decode_block0 == block0;
        timeline.last_decode_seq_len = Some(seq_len);
        timeline.last_decode_block0 = block0;
        continues
    }

    #[cfg(feature = "rocm")]
    fn clone_linear_state_handles(state: &LinearAttentionState) -> LinearAttentionState {
        LinearAttentionState {
            recurrent_states: state.recurrent_states.clone(),
            conv_states: state.conv_states.clone(),
        }
    }

    #[cfg(feature = "rocm")]
    fn linear_state_handles_match(
        left: &LinearAttentionState,
        right: &LinearAttentionState,
    ) -> bool {
        left.recurrent_states.len() == right.recurrent_states.len()
            && left.conv_states.len() == right.conv_states.len()
            && left
                .recurrent_states
                .iter()
                .zip(&right.recurrent_states)
                .all(|(left, right)| left.id() == right.id())
            && left
                .conv_states
                .iter()
                .zip(&right.conv_states)
                .all(|(left, right)| left.id() == right.id())
    }

    #[cfg(any(feature = "rocm", test))]
    fn restore_linear_state_in_place(
        state: &mut LinearAttentionState,
        snapshot: &LinearAttentionState,
        context: &'static str,
    ) -> Result<()> {
        state
            .refresh_batched_state_from_rows_in_place(&[snapshot])
            .with_context(|| context)
    }

    #[cfg(feature = "rocm")]
    fn bind_decode_row_to_slot(
        &mut self,
        row_id: u64,
        requested_key: &RocmGraphKey,
        linear_state: &mut LinearAttentionState,
    ) -> Result<RocmGraphOwner> {
        let existing = self.decode_row_slots.get(&row_id).copied();
        let owner = if let Some(owner) = existing {
            owner
        } else {
            let preferred = self
                .graph_slots
                .iter()
                .filter(|(_, slot)| slot.assigned_row.is_none())
                .map(|(owner, _)| *owner)
                .filter(|owner| {
                    self.captured
                        .contains_key(&RocmGraphCacheKey::new(*owner, requested_key.clone()))
                })
                .min_by_key(|owner| owner.slot_id());
            let idle = preferred.or_else(|| {
                self.graph_slots
                    .iter()
                    .filter(|(_, slot)| slot.assigned_row.is_none())
                    .map(|(owner, _)| *owner)
                    .min_by_key(|owner| owner.slot_id())
            });
            let owner = if let Some(owner) = idle {
                self.counters.record_graph_slot_reuse();
                owner
            } else {
                let owner = RocmGraphOwner::Slot(self.next_graph_slot_id);
                self.next_graph_slot_id = self.next_graph_slot_id.saturating_add(1);
                self.graph_slots.insert(
                    owner,
                    RocmGraphSlotState {
                        assigned_row: None,
                        linear_state: Self::clone_linear_state_handles(linear_state),
                    },
                );
                self.counters.record_graph_slot_create();
                owner
            };
            self.decode_row_slots.insert(row_id, owner);
            self.decode_timelines.remove(&owner);
            self.graph_slots
                .get_mut(&owner)
                .expect("new or idle ROCm graph slot must exist")
                .assigned_row = Some(row_id);
            owner
        };

        let slot = self
            .graph_slots
            .get_mut(&owner)
            .context("ROCm graph row points to a missing persistent slot")?;
        anyhow::ensure!(
            slot.assigned_row == Some(row_id),
            "ROCm graph slot {} is assigned to {:?}, not row {row_id}",
            owner.slot_id(),
            slot.assigned_row
        );
        if !Self::linear_state_handles_match(&slot.linear_state, linear_state) {
            slot.linear_state
                .refresh_batched_state_from_rows_in_place(&[linear_state])
                .context("refresh reusable ROCm graph slot state")?;
            linear_state
                .recurrent_states
                .clone_from(&slot.linear_state.recurrent_states);
            linear_state
                .conv_states
                .clone_from(&slot.linear_state.conv_states);
        }
        Ok(owner)
    }

    #[cfg(feature = "rocm")]
    fn max_cached_graphs() -> usize {
        std::env::var("KILN_ROCM_GRAPH_CACHE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(8)
    }

    #[cfg(feature = "rocm")]
    fn run_eager_fallback<T>(
        &mut self,
        reason: RocmGraphFallbackReason,
        seq_len: usize,
        attempt_duration: std::time::Duration,
        eager: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let eager_started = std::time::Instant::now();
        let result = eager();
        let eager_duration = eager_started.elapsed();
        let duration = attempt_duration.saturating_add(eager_duration);
        let occurrence = self.counters.record_fallback(reason, duration);
        let attempt_duration_ms = attempt_duration.as_secs_f64() * 1000.0;
        let eager_duration_ms = eager_duration.as_secs_f64() * 1000.0;
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let slow = duration >= RocmGraphFallbackStats::SLOW_DURATION;
        match &result {
            Err(error) => tracing::warn!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_failed",
                %error,
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "ROCm graph eager fallback failed"
            ),
            Ok(_) if slow => tracing::warn!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "slow ROCm graph eager fallback completed"
            ),
            Ok(_) if occurrence == 1 => tracing::info!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "ROCm graph eager fallback activated"
            ),
            Ok(_) => tracing::debug!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "ROCm graph eager fallback completed"
            ),
        }
        result
    }

    /// Run one bs=1 paged decode step, returning kt logits `[1, 1, vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        graph_row_id: u64,
    ) -> Result<Tensor> {
        if !self.enabled || std::env::var("KILN_FORCE_EAGER_DECODE").ok().as_deref() == Some("1") {
            return Self::eager_forward(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            // NOTE: FP8 paged KV caches ARE capturable on ROCm (unlike CUDA): the
            // FP8 graph-slot write consumes the slot ON-DEVICE and scatters the
            // quantized U8 rows via `rocm_index_copy_dim0` (device_ptr_raw, never
            // `slice()`), so it records into the captured graph and is safe on the
            // Borrowed freeze-pointer arena buffers. See
            // `PagedKvCacheKt::write_token_major_native_graph_slot`. Any residual
            // host round-trip is still caught by the warm-pass htod check below
            // (graceful eager), so no explicit FP8 guard is needed here.

            // Warmup: first decode step runs eagerly (graph-shaped position
            // buffer) to prime the allocator pools before the first capture.
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
                let warmup_started = std::time::Instant::now();
                match Self::eager_forward_with_position_buffer(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                ) {
                    Ok(logits) => return Ok(logits),
                    Err(e) => {
                        tracing::warn!(
                            "ROCm graph-shaped warmup failed: {e:#}, plain eager decode"
                        );
                        return self.run_eager_fallback(
                            RocmGraphFallbackReason::WarmupForwardFailure,
                            seq_len,
                            warmup_started.elapsed(),
                            || {
                                Self::eager_forward(
                                    backend,
                                    token_id,
                                    weights,
                                    config,
                                    paged_cache,
                                    block_table,
                                    seq_len,
                                    linear_state,
                                    lora,
                                )
                            },
                        );
                    }
                }
            }

            // Native capture defaults on with the primary graph runner. The
            // narrower KILN_ROCM_GRAPH_CAPTURE=0 opt-out keeps the graph-shaped
            // warmup but uses eager steady-state decode.
            if !self.capture_requested {
                return Self::eager_forward(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                );
            }

            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);
            let owner = self.bind_decode_row_to_slot(graph_row_id, &requested_key, linear_state)?;
            self.prepare_owner_decode(owner, graph_row_id, block_table, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());

            // Geometry previously found non-capture-safe: skip the warm pass +
            // capture attempt and reuse its typed eager-fallback reason.
            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            // Replay if we have a valid captured graph for this geometry.
            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay(
                        &cache_key,
                        token_id,
                        backend,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                    ) {
                        Ok(logits) => {
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(logits);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ROCm graph replay failed: {e:#}; disabling ROCm HIP graphs for this runner and falling back to eager"
                            );
                            self.enabled = false;
                            self.captured.clear();
                            return self.run_eager_fallback(
                                RocmGraphFallbackReason::ReplayFailure,
                                seq_len,
                                replay_started.elapsed(),
                                || {
                                    Self::eager_forward(
                                        backend,
                                        token_id,
                                        weights,
                                        config,
                                        paged_cache,
                                        block_table,
                                        seq_len,
                                        linear_state,
                                        lora,
                                    )
                                },
                            );
                        }
                    }
                } else {
                    self.captured.clear();
                }
            }

            if self.captured.len() >= Self::max_cached_graphs() {
                if !self.cache_full_warned {
                    self.cache_full_warned = true;
                    tracing::warn!(
                        cached = self.captured.len(),
                        "ROCm graph capture skipped: paged metadata shape cache full"
                    );
                }
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::GraphCacheCapacity,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            // Memory-pressure guard: capturing a new graph mints freeze-pointer
            // arena + per-layer output buffers (a few MB). Under Critical memory
            // pressure (a coexisting job / training run has the VRAM), skip the
            // capture and run eager rather than risk the allocation tipping the
            // box into OOM — the governor sees all-process usage, so this respects
            // whatever else is on the GPU. Decode stays correct either way.
            if kiln_memory::MemoryGovernor::global().pressure()
                == kiln_memory::MemoryPressure::Critical
            {
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::CriticalMemoryPressure,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            // Capture.
            let capture_started = std::time::Instant::now();
            match self.try_capture(
                backend,
                owner,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!("ROCm graph capture failed: {e:#}, disabling graphs (eager)");
                    self.enabled = false;
                    // A failed capture can leave pending default-stream work.
                    // Drain it before eager fallback rather than cascading into
                    // a second failure. Capture-stream failures trip the runner
                    // circuit breaker and are settled at external yield.
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device());
                    return self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = graph_row_id;
            let _ = linear_state;
            Self::eager_forward(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )
        }
    }

    /// Run one bs=1 paged decode step, returning the pre-final-norm hidden
    /// `[1, 1, hidden]` so callers can keep sampling outside the graph.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_hidden(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        graph_row_id: u64,
    ) -> Result<Tensor> {
        if !self.enabled || std::env::var("KILN_FORCE_EAGER_DECODE").ok().as_deref() == Some("1") {
            return Self::eager_forward_hidden(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
                let warmup_started = std::time::Instant::now();
                match Self::eager_forward_hidden_with_position_buffer(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                ) {
                    Ok(hidden) => return Ok(hidden),
                    Err(e) => {
                        tracing::warn!(
                            "ROCm graph-shaped warmup failed: {e:#}, plain eager decode"
                        );
                        return self.run_eager_fallback(
                            RocmGraphFallbackReason::WarmupForwardFailure,
                            seq_len,
                            warmup_started.elapsed(),
                            || {
                                Self::eager_forward_hidden(
                                    backend,
                                    token_id,
                                    weights,
                                    config,
                                    paged_cache,
                                    block_table,
                                    seq_len,
                                    linear_state,
                                    lora,
                                )
                            },
                        );
                    }
                }
            }

            if !self.capture_requested {
                return Self::eager_forward_hidden(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                );
            }

            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);
            let owner = self.bind_decode_row_to_slot(graph_row_id, &requested_key, linear_state)?;
            self.prepare_owner_decode(owner, graph_row_id, block_table, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());

            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_hidden(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay_hidden(
                        &cache_key,
                        token_id,
                        weights,
                        paged_cache,
                        block_table,
                        seq_len,
                    ) {
                        Ok(hidden) => {
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(hidden);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ROCm graph replay failed: {e:#}; disabling ROCm HIP graphs for this runner and falling back to eager"
                            );
                            self.enabled = false;
                            self.captured.clear();
                            return self.run_eager_fallback(
                                RocmGraphFallbackReason::ReplayFailure,
                                seq_len,
                                replay_started.elapsed(),
                                || {
                                    Self::eager_forward_hidden(
                                        backend,
                                        token_id,
                                        weights,
                                        config,
                                        paged_cache,
                                        block_table,
                                        seq_len,
                                        linear_state,
                                        lora,
                                    )
                                },
                            );
                        }
                    }
                } else {
                    self.captured.clear();
                }
            }

            if self.captured.len() >= Self::max_cached_graphs() {
                if !self.cache_full_warned {
                    self.cache_full_warned = true;
                    tracing::warn!(
                        cached = self.captured.len(),
                        "ROCm graph capture skipped: paged metadata shape cache full"
                    );
                }
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::GraphCacheCapacity,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward_hidden(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            if kiln_memory::MemoryGovernor::global().pressure()
                == kiln_memory::MemoryPressure::Critical
            {
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::CriticalMemoryPressure,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward_hidden(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            let capture_started = std::time::Instant::now();
            match self.try_capture_hidden(
                backend,
                owner,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ) {
                Ok(RocmCaptureStep::CapturedHidden(hidden)) => return Ok(hidden),
                Ok(RocmCaptureStep::FallbackEager(reason)) => {
                    return self.run_eager_fallback(
                        reason,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward_hidden(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!("ROCm graph capture failed: {e:#}, disabling graphs (eager)");
                    self.enabled = false;
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device());
                    return self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward_hidden(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = graph_row_id;
            let _ = linear_state;
            Self::eager_forward_hidden(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )
        }
    }

    /// Run one bs=1 paged decode step, returning the greedy token directly.
    ///
    /// This follows the same graph warmup/capture/replay state machine as
    /// [`Self::decode_step_paged`] but keeps the eager tail on the fast
    /// `final_norm + lm_head argmax` path, avoiding materializing
    /// `[1, 1, vocab]` logits only to reduce them immediately.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_greedy(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        graph_row_id: u64,
    ) -> Result<u32> {
        if !self.enabled || std::env::var("KILN_FORCE_EAGER_DECODE").ok().as_deref() == Some("1") {
            return Self::eager_forward_greedy(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
                let warmup_started = std::time::Instant::now();
                match Self::eager_forward_greedy_with_position_buffer(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                ) {
                    Ok(token) => return Ok(token),
                    Err(e) => {
                        tracing::warn!(
                            "ROCm graph-shaped warmup failed: {e:#}, plain eager decode"
                        );
                        return self.run_eager_fallback(
                            RocmGraphFallbackReason::WarmupForwardFailure,
                            seq_len,
                            warmup_started.elapsed(),
                            || {
                                Self::eager_forward_greedy(
                                    backend,
                                    token_id,
                                    weights,
                                    config,
                                    paged_cache,
                                    block_table,
                                    seq_len,
                                    linear_state,
                                    lora,
                                )
                            },
                        );
                    }
                }
            }

            if !self.capture_requested {
                return Self::eager_forward_greedy(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                );
            }

            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);
            let owner = self.bind_decode_row_to_slot(graph_row_id, &requested_key, linear_state)?;
            self.prepare_owner_decode(owner, graph_row_id, block_table, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());

            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay_greedy(
                        &cache_key,
                        token_id,
                        backend,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                    ) {
                        Ok(token) => {
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(token);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ROCm graph replay failed: {e:#}; disabling ROCm HIP graphs for this runner and falling back to eager"
                            );
                            self.enabled = false;
                            self.captured.clear();
                            return self.run_eager_fallback(
                                RocmGraphFallbackReason::ReplayFailure,
                                seq_len,
                                replay_started.elapsed(),
                                || {
                                    Self::eager_forward_greedy(
                                        backend,
                                        token_id,
                                        weights,
                                        config,
                                        paged_cache,
                                        block_table,
                                        seq_len,
                                        linear_state,
                                        lora,
                                    )
                                },
                            );
                        }
                    }
                } else {
                    self.captured.clear();
                }
            }

            if self.captured.len() >= Self::max_cached_graphs() {
                if !self.cache_full_warned {
                    self.cache_full_warned = true;
                    tracing::warn!(
                        cached = self.captured.len(),
                        "ROCm graph capture skipped: paged metadata shape cache full"
                    );
                }
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::GraphCacheCapacity,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward_greedy(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            if kiln_memory::MemoryGovernor::global().pressure()
                == kiln_memory::MemoryPressure::Critical
            {
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::CriticalMemoryPressure,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward_greedy(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            let capture_started = std::time::Instant::now();
            match self.try_capture_greedy(
                backend,
                owner,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ) {
                Ok(token) => return Ok(token),
                Err(e) => {
                    tracing::warn!("ROCm graph capture failed: {e:#}, disabling graphs (eager)");
                    self.enabled = false;
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device());
                    return self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward_greedy(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = graph_row_id;
            let _ = linear_state;
            Self::eager_forward_greedy(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )
        }
    }

    /// Plain eager decode forward — `model_forward_paged` over a single token.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        model_forward_paged(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            None,
        )
        .context("eager decode forward pass failed (rocm)")
    }

    /// Plain eager greedy decode forward.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_greedy(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<u32> {
        model_forward_paged_next_token_greedy(
            backend,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            None,
        )
        .context("eager greedy decode forward pass failed (rocm)")
    }

    /// Plain eager hidden-only decode.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_hidden(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let sequence_lengths = [seq_len];
        let mut linear_states: [&mut LinearAttentionState; 1] = [linear_state];
        model_forward_paged_batched_decode_hidden(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            std::slice::from_ref(block_table),
            &sequence_lengths,
            &mut linear_states,
            lora,
        )
        .context("eager hidden-only decode forward pass failed (rocm)")
    }

    /// Eager decode with a graph-shaped position buffer (warms the allocator
    /// with the capture-shaped allocation sequence).
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_with_position_buffer(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let device = weights.embed_tokens.device();
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        model_forward_paged(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            Some(&position_buffer),
        )
        .context("graph-shaped eager decode forward pass failed (rocm)")
    }

    /// Eager greedy decode with a graph-shaped position buffer.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_greedy_with_position_buffer(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<u32> {
        let device = weights.embed_tokens.device();
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        model_forward_paged_next_token_greedy(
            backend,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            Some(&position_buffer),
        )
        .context("graph-shaped eager greedy decode forward pass failed (rocm)")
    }

    /// Eager hidden-only decode with a graph-shaped position buffer.
    #[cfg(feature = "rocm")]
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_hidden_with_position_buffer(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let device = weights.embed_tokens.device();
        let token_buffer = Self::new_token_buffer(device, token_id)?;
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        model_forward_paged_hidden_with_graph_inputs(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            &token_buffer,
            &position_buffer,
            None,
        )
        .context("graph-shaped eager hidden-only decode forward pass failed (rocm)")
    }

    fn new_position_buffer(device: Device, position: usize) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![position as f32], vec![1])
            .context("create ROCm graph position buffer")
    }
}

// ======================================================================
// Capture / replay machinery (ROCm only).
// ======================================================================
#[cfg(feature = "rocm")]
impl RocmGraphRunner {
    /// Replay a cached graph: refresh the per-step input buffers, sync the
    /// default stream, launch, sync the capture stream, then run lm_head eagerly
    /// on the replayed hidden.
    #[allow(clippy::too_many_arguments)]
    fn replay(
        &mut self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        backend: &dyn BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<Tensor> {
        let hidden =
            self.replay_hidden(key, token_id, weights, paged_cache, block_table, seq_len)?;
        crate::forward::lm_head_from_hidden_eager(backend, &hidden, weights, config)
            .context("eager lm_head on replayed hidden")
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_greedy(
        &mut self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        backend: &dyn BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<u32> {
        let hidden =
            self.replay_hidden(key, token_id, weights, paged_cache, block_table, seq_len)?;
        crate::forward::lm_head_argmax_from_hidden_eager(backend, &hidden, weights, config)
            .context("eager lm_head argmax on replayed hidden")
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_hidden(
        &mut self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        weights: &GpuWeights,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<Tensor> {
        let result =
            self.replay_hidden_inner(key, token_id, weights, paged_cache, block_table, seq_len);
        self.counters.record_replay_outcome(result.is_ok());
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_hidden_inner(
        &self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        weights: &GpuWeights,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<Tensor> {
        let captured = self
            .captured
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("replay: key vanished"))?;

        paged_cache
            .ensure_pool_identity(captured.kv_pool_identity)
            .context("refuse ROCm graph replay after paged-KV pool replacement")?;

        Self::update_token_buffer(&captured.token_buffer, token_id)?;
        Self::update_position_buffer(&captured.position_buffer, seq_len)?;
        Self::update_rotary_buffers(
            &captured.rotary_cos_buffer,
            &captured.rotary_sin_buffer,
            &weights.rotary_inv_freq,
            &captured.position_buffer,
        )?;
        if let (Some(bt), Some(sk), Some(slot)) = (
            captured.block_table_buffer.as_ref(),
            captured.seqused_k_buffer.as_ref(),
            captured.kv_slot_buffer.as_ref(),
        ) {
            Self::update_paged_metadata_buffers(
                bt,
                sk,
                slot,
                block_table,
                paged_cache,
                seq_len,
                captured.max_seqlen_k,
            )?;
        }

        // The writes above land on the kt default stream while the graph
        // launches on its capture stream. The event dependency prevents stale
        // replay inputs without forcing the host to wait for either stream.
        captured
            .default_stream
            .record_event(&captured.replay_inputs_ready_event)
            .context("record per-replay ROCm graph inputs")?;
        captured
            .capture_stream
            .wait_event(&captured.replay_inputs_ready_event)
            .context("order per-replay ROCm graph input handoff")?;

        let mut plan = RocmDecodeReplayPlan::new(captured);
        let replay_key = kiln_graph::ReplayPlan::key(&plan);
        let replay_inputs = ReplayInputs::new(&replay_key, &captured.replay_state.inputs);
        kiln_graph::ReplayPlan::replay(&mut plan, replay_inputs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(captured.output_hidden.clone())
    }

    fn replay_state_for_capture(
        key: &RocmGraphKey,
        output_hidden: &Tensor,
        token_buffer: &Tensor,
        position_buffer: &Tensor,
        block_table_buffer: Option<&Tensor>,
        seqused_k_buffer: Option<&Tensor>,
        kv_slot_buffer: Option<&Tensor>,
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        paged_decode_outputs: &[Tensor],
        paged_decode_lse: &[Tensor],
        gdn_decode_outputs: &[Tensor],
    ) -> ReplayState {
        let replay_key = ReplayKey::new(
            Backend::Rocm,
            "paged_decode_graph_outputs",
            vec![
                key.seq_len,
                key.block_table.len(),
                key.max_seqlen_k,
                key.max_blocks_per_seq,
                usize::from(key.stable_metadata),
            ],
            Some(output_hidden.dtype()),
            1,
            true,
        );
        let mut resources = vec![
            Self::stable_replay_resource(output_hidden),
            Self::stable_replay_resource(token_buffer),
            Self::stable_replay_resource(position_buffer),
            Self::stable_replay_resource(rotary_cos_buffer),
            Self::stable_replay_resource(rotary_sin_buffer),
        ];
        for tensor in [block_table_buffer, seqused_k_buffer, kv_slot_buffer]
            .into_iter()
            .flatten()
        {
            resources.push(Self::stable_replay_resource(tensor));
        }
        resources.extend(
            paged_decode_outputs
                .iter()
                .map(Self::stable_replay_resource),
        );
        resources.extend(paged_decode_lse.iter().map(Self::stable_replay_resource));
        resources.extend(gdn_decode_outputs.iter().map(Self::stable_replay_resource));
        ReplayState::new(replay_key, resources)
    }

    fn stable_replay_resource(tensor: &Tensor) -> ResidentResourceRef {
        ResidentResourceRef::from_tensor(
            tensor,
            Backend::Rocm,
            ReplayResourceStability::StableAcrossReplay,
        )
    }

    // --- per-replay in-place buffer refresh (frozen device pointers) ---

    fn update_token_buffer(token_buffer: &Tensor, token_id: u32) -> Result<()> {
        kiln_tensor::rocm_write_host_in_place(token_buffer, &[token_id])
            .context("update ROCm graph token buffer")
    }

    fn update_position_buffer(position_buffer: &Tensor, position: usize) -> Result<()> {
        kiln_tensor::rocm_write_host_in_place(position_buffer, &[position as f32])
            .context("update ROCm graph position buffer")
    }

    fn update_rotary_buffers(
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        rotary_inv_freq: &Tensor,
        position_buffer: &Tensor,
    ) -> Result<()> {
        // #34 BUG2 FIX: compute the rotary tables on the GPU via eager's exact
        // path (`forward::rotary_tables_from_tensor` -> device `cos`/`sin`), not
        // host CPU cos/sin. CPU cos != GPU cos (range reduction) perturbs only the
        // RoPE full-attention layers on replay -> divergence from eager. Same root
        // cause + fix as the CUDA path.
        let (cos, sin) =
            crate::forward::rotary_tables_from_tensor(position_buffer, rotary_inv_freq)?;
        let cos = cos
            .to_dtype(rotary_cos_buffer.dtype())?
            .reshape(rotary_cos_buffer.dims().to_vec())?;
        let sin = sin
            .to_dtype(rotary_sin_buffer.dtype())?
            .reshape(rotary_sin_buffer.dims().to_vec())?;
        rotary_cos_buffer
            .slice_set(&cos, 0, 0)
            .context("update ROCm graph rotary cos buffer (gpu)")?;
        rotary_sin_buffer
            .slice_set(&sin, 0, 0)
            .context("update ROCm graph rotary sin buffer (gpu)")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn update_paged_metadata_buffers(
        block_table_buffer: &Tensor,
        seqused_k_buffer: &Tensor,
        kv_slot_buffer: &Tensor,
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        seq_len: usize,
        max_seqlen_k: usize,
    ) -> Result<()> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        kiln_tensor::rocm_write_host_in_place(block_table_buffer, padded.as_slice())
            .context("update ROCm graph block table buffer")?;
        let attention_len = [(seq_len + 1) as u32];
        kiln_tensor::rocm_write_host_in_place(seqused_k_buffer, &attention_len)
            .context("update ROCm graph seqused_k buffer")?;
        let slot = [block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32];
        kiln_tensor::rocm_write_host_in_place(kv_slot_buffer, &slot)
            .context("update ROCm graph KV slot buffer")?;
        Ok(())
    }

    fn padded_block_table(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
    ) -> Result<Vec<u32>> {
        let block_size = paged_cache.block_size();
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let pages_per_chunk = kblock_n / block_size;
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        let take = max_blocks_per_seq.min(block_table.blocks.len());
        let mut padded = Vec::with_capacity(max_blocks_per_seq);
        padded.extend_from_slice(&block_table.blocks[..take]);
        if padded.is_empty() {
            anyhow::bail!("paged decode graph block table is empty");
        }
        // Pad with the LAST REAL block (repeated), never an incrementing one —
        // an out-of-pool page baked into the graph would fault on replay; a
        // repeated valid block stays in-bounds and is masked by seqused_k.
        let pad_block = *padded.last().expect("non-empty");
        while padded.len() < max_blocks_per_seq {
            padded.push(pad_block);
        }
        Ok(padded)
    }

    // --- graph-stable buffer constructors ---

    fn new_block_table_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
        device: Device,
    ) -> Result<Tensor> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        let len = padded.len();
        Tensor::from_vec_on(device, padded, vec![1, len])
            .context("create ROCm graph block table buffer")
    }

    fn new_seqused_k_buffer(device: Device, attention_len: usize) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![attention_len as u32], vec![1])
            .context("create ROCm graph seqused_k buffer")
    }

    fn new_kv_slot_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        seq_len: usize,
        device: Device,
    ) -> Result<Tensor> {
        let slot = block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32;
        Tensor::from_vec_on(device, vec![slot], vec![1]).context("create ROCm graph KV slot buffer")
    }

    fn new_rotary_cos_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        // #34 BUG2 FIX: GPU rotary (matches eager + update_rotary_buffers), not host CPU cos.
        let inv_freq = crate::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            &device,
        )?;
        let pos = Tensor::from_vec_on(device, vec![position as f32], vec![1])?;
        let (cos, _) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        cos.to_dtype(kiln_tensor::DType::F32)?
            .contiguous()
            .context("create ROCm graph rotary cos buffer (gpu)")
    }

    fn new_rotary_sin_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        // #34 BUG2 FIX: GPU rotary (see new_rotary_cos_buffer).
        let inv_freq = crate::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            &device,
        )?;
        let pos = Tensor::from_vec_on(device, vec![position as f32], vec![1])?;
        let (_, sin) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        sin.to_dtype(kiln_tensor::DType::F32)?
            .contiguous()
            .context("create ROCm graph rotary sin buffer (gpu)")
    }

    fn new_output_hidden(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
    ) -> Result<Tensor> {
        Tensor::zeros_on(device, vec![1, 1, config.hidden_size], dtype)
            .context("create ROCm graph output hidden")
    }

    fn new_paged_decode_outputs(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut outputs = Vec::with_capacity(config.num_full_attention_layers);
        let mut lse = Vec::with_capacity(config.num_full_attention_layers);
        for _ in 0..config.num_full_attention_layers {
            outputs.push(
                Tensor::zeros_on(
                    device,
                    vec![1, 1, config.num_attention_heads, config.head_dim],
                    dtype,
                )
                .context("create ROCm graph paged decode output")?,
            );
            lse.push(
                Tensor::zeros_on(
                    device,
                    vec![1, config.num_attention_heads, 1],
                    kiln_tensor::DType::F32,
                )
                .context("create ROCm graph paged decode LSE")?,
            );
        }
        Ok((outputs, lse))
    }

    fn prepare_gdn_recurrent_state_for_capture(
        linear_state: &mut LinearAttentionState,
    ) -> Result<()> {
        for state in &mut linear_state.recurrent_states {
            if state.dtype() != kiln_tensor::DType::BF16 {
                *state = state
                    .to_dtype(kiln_tensor::DType::BF16)
                    .context("prepare ROCm graph GDN recurrent state")?;
            }
        }
        Ok(())
    }

    fn new_gdn_decode_outputs(config: &ModelConfig, device: Device) -> Result<Vec<Tensor>> {
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let mut outputs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            outputs.push(
                Tensor::zeros_on(
                    device,
                    vec![
                        1,
                        1,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                    ],
                    kiln_tensor::DType::BF16,
                )
                .context("create ROCm graph GDN decode output")?,
            );
        }
        Ok(outputs)
    }

    /// Capture a HIP graph for this decode step (bs=1), launch it once to
    /// compute + advance state, and return this step's logits. Mirrors
    /// `CudaGraphRunner::try_capture`.
    #[allow(clippy::too_many_arguments)]
    fn try_capture(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let capture_started = std::time::Instant::now();
        match self.try_capture_hidden(
            backend,
            owner,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        )? {
            RocmCaptureStep::CapturedHidden(hidden) => {
                crate::forward::lm_head_from_hidden_eager(backend, &hidden, weights, config)
                    .context("eager lm_head on captured hidden (first launch)")
            }
            RocmCaptureStep::FallbackEager(reason) => {
                self.run_eager_fallback(reason, seq_len, capture_started.elapsed(), || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                })
            }
        }
    }

    /// Capture a HIP graph for this decode step (bs=1), launch it once to
    /// compute + advance state, and return this step's greedy token.
    #[allow(clippy::too_many_arguments)]
    fn try_capture_greedy(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<u32> {
        let capture_started = std::time::Instant::now();
        match self.try_capture_hidden(
            backend,
            owner,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        )? {
            RocmCaptureStep::CapturedHidden(hidden) => {
                crate::forward::lm_head_argmax_from_hidden_eager(backend, &hidden, weights, config)
                    .context("eager lm_head argmax on captured hidden (first launch)")
            }
            RocmCaptureStep::FallbackEager(reason) => {
                self.run_eager_fallback(reason, seq_len, capture_started.elapsed(), || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                })
            }
        }
    }

    /// Capture a HIP graph for this decode step (bs=1), launch it once to
    /// compute + advance state, and return the graph-stable hidden.
    #[allow(clippy::too_many_arguments)]
    fn try_capture_hidden(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<RocmCaptureStep> {
        let result = self.try_capture_hidden_inner(
            backend,
            owner,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        );
        let outcome = match &result {
            Ok(RocmCaptureStep::CapturedHidden(_)) => RocmGraphCaptureOutcome::Succeeded,
            Ok(RocmCaptureStep::FallbackEager(_)) => RocmGraphCaptureOutcome::Deferred,
            Err(_) => RocmGraphCaptureOutcome::Failed,
        };
        self.counters.record_capture_outcome(outcome);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn try_capture_hidden_inner(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<RocmCaptureStep> {
        let device = weights.embed_tokens.device();
        let dtype = weights.embed_tokens.dtype();
        let device_idx = match device {
            Device::Rocm(i) => i,
            _ => anyhow::bail!("ROCm graphs require a Rocm device"),
        };

        // Capture on a FRESH non-default stream (mirror the CUDA discipline; the
        // with_active_rocm_stream scope routes every kt op onto it).
        let context = kiln_tensor::primary_rocm_context(device_idx)
            .context("ROCm graph capture: primary_rocm_context for capture stream")?;
        let stream = context
            .new_stream()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create capture stream: {e}"))?;
        let default_stream = context.default_stream();
        let replay_inputs_ready_event = context
            .new_event()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create input event: {e}"))?;
        let replay_complete_event = context
            .new_event()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create completion event: {e}"))?;

        // Pre-allocate graph-stable buffers before capture.
        let token_buffer = Self::new_token_buffer(device, token_id)?;
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        let output_hidden = Self::new_output_hidden(config, device, dtype)?;
        let rotary_cos_buffer = Self::new_rotary_cos_buffer(config, device, seq_len)?;
        let rotary_sin_buffer = Self::new_rotary_sin_buffer(config, device, seq_len)?;
        let key = RocmGraphKey::new(block_table, paged_cache, seq_len);
        let (block_table_buffer, seqused_k_buffer, kv_slot_buffer) = if key.stable_metadata {
            (
                Some(Self::new_block_table_buffer(
                    block_table,
                    paged_cache,
                    key.max_seqlen_k,
                    device,
                )?),
                Some(Self::new_seqused_k_buffer(device, seq_len + 1)?),
                Some(Self::new_kv_slot_buffer(
                    block_table,
                    paged_cache,
                    seq_len,
                    device,
                )?),
            )
        } else {
            (None, None, None)
        };
        let (paged_decode_outputs, paged_decode_lse) = if key.stable_metadata {
            Self::new_paged_decode_outputs(config, device, dtype)?
        } else {
            (Vec::new(), Vec::new())
        };
        let graph_inputs = match (
            block_table_buffer.as_ref(),
            seqused_k_buffer.as_ref(),
            kv_slot_buffer.as_ref(),
        ) {
            (Some(bt), Some(sk), Some(slot)) => Some(PagedDecodeGraphInputs {
                block_table: bt,
                seqused_k: sk,
                kv_slot: slot,
                max_seqlen_k: key.max_seqlen_k,
                rotary_cos: &rotary_cos_buffer,
                rotary_sin: &rotary_sin_buffer,
                attn_out: &paged_decode_outputs[..],
                softmax_lse: &paged_decode_lse[..],
            }),
            _ => None,
        };
        let gdn_decode_outputs = Self::new_gdn_decode_outputs(config, device)?;
        Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
        let stable_graph_io_bytes = graph_tensor_bytes(
            [
                &token_buffer,
                &position_buffer,
                &output_hidden,
                &rotary_cos_buffer,
                &rotary_sin_buffer,
            ]
            .into_iter()
            .chain(
                [
                    block_table_buffer.as_ref(),
                    seqused_k_buffer.as_ref(),
                    kv_slot_buffer.as_ref(),
                ]
                .into_iter()
                .flatten(),
            )
            .chain(paged_decode_outputs.iter())
            .chain(paged_decode_lse.iter())
            .chain(gdn_decode_outputs.iter()),
        );

        // === freeze-pointers Pass 1 (Record / warm) ===
        let arena_ctx = kiln_tensor::primary_rocm_context(device_idx)
            .context("freeze-pointers: primary_rocm_context for capture arena")?;
        let arena = std::rc::Rc::new(std::cell::RefCell::new(
            kiln_tensor::RocmCaptureArena::new_record(arena_ctx, device_idx),
        ));
        let gdn_snapshot = linear_state
            .snapshot()
            .context("freeze-pointers: snapshot GDN recurrent state before warm pass")?;
        // The graph-stable input buffers and GDN snapshot above were filled on
        // the kt default stream. The warm pass deliberately runs on the SAME
        // non-default stream that will be captured so hipBLASLt's per-stream
        // workspace is allocated before `hipStreamBeginCapture`; sync first so
        // that stream sees initialized inputs/state.
        attributed_rocm_graph_synchronize(
            "default_inputs_before_warmup",
            "rocm_graph_capture_warmup",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                kiln_tensor::rocm_synchronize_default_stream(device_idx)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        // Snapshot the host→device copy count across the warm forward. Any
        // host_to_rocm_copy issued by the forward (e.g. a GDN-gates/softplus or
        // KV-write FALLBACK path that allocates a device tensor from host) does a
        // hipStreamSynchronize, which is ILLEGAL inside begin_capture and ABORTS
        // the capture — poisoning the device so even the eager fallback then
        // fails (an empty response under load). The warm pass runs the SAME
        // forward OUTSIDE capture, so if it did a host round-trip the captured
        // pass would too: skip capture for this geometry and fall back to eager
        // BEFORE begin_capture, leaving the device clean.
        let htod_before = kiln_tensor::rocm_htod_count();
        let warm_result = kiln_tensor::with_rocm_capture_arena(arena.clone(), || {
            kiln_tensor::with_active_rocm_stream(stream.clone(), || {
                let hidden = model_forward_paged_hidden_with_graph_inputs(
                    backend,
                    &[token_id],
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    Some(linear_state),
                    lora,
                    &token_buffer,
                    &position_buffer,
                    graph_inputs.as_ref(),
                )?;
                kiln_tensor::rocm_slice_set_dim0(&output_hidden, &hidden, 0)
                    .context("freeze-pointers warm pass: copy hidden into stable output")?;
                Ok::<(), anyhow::Error>(())
            })
        });
        let warm_sync_result = attributed_rocm_graph_synchronize(
            "capture_stream_warmup_completion",
            "rocm_graph_capture_warmup",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                stream
                    .synchronize()
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        );
        if let Err(err) = warm_result {
            if let Err(sync_err) = warm_sync_result {
                tracing::warn!("post-warm-failure capture stream sync failed: {sync_err:#}");
                return Err(sync_err)
                    .context("capture stream synchronization failed after warm-forward failure");
            }
            Self::restore_linear_state_in_place(
                linear_state,
                &gdn_snapshot,
                "restore graph-slot GDN state after warm-forward failure",
            )?;
            if crate::forward::is_rocm_graph_shape_dependent_attention(&err) {
                self.non_capture_safe
                    .insert(key, RocmGraphFallbackReason::ShapeDependentAttention);
                return Ok(RocmCaptureStep::FallbackEager(
                    RocmGraphFallbackReason::ShapeDependentAttention,
                ));
            }
            return Err(err).context("freeze-pointers warm (Record) pass failed");
        }
        if let Err(sync_err) = warm_sync_result {
            return Err(sync_err);
        }
        // Restore values without replacing the graph slot's tensor handles.
        // Captured graphs retain these exact addresses across request reuse.
        Self::restore_linear_state_in_place(
            linear_state,
            &gdn_snapshot,
            "restore graph-slot GDN state after warm pass",
        )?;
        let htod_after = kiln_tensor::rocm_htod_count();
        if htod_after > htod_before {
            // The warm forward did a host round-trip. This is EITHER a one-time
            // cold-cache fill (shape-keyed broadcast/gqa-expand gather indices
            // upload once per `max_seqlen_k` bucket, then every step + replay in
            // that bucket reuses the device buffer) OR a genuine per-step fallback
            // (e.g. a GDN-gates/softplus path that rebuilds host data every step).
            // The two are indistinguishable from a single warm pass, so retry: a
            // cold fill is absorbed by THIS pass and the next attempt for the same
            // (bucket-stable) geometry sees htod==0 and captures; a real per-step
            // round-trip keeps bumping htod and exhausts the retry budget. The warm
            // pass advanced then restored the recurrent state, and begin_capture was
            // never called, so the device stays clean either way.
            let attempts = self.capture_retry.entry(key.clone()).or_insert(0);
            *attempts += 1;
            let fallback_reason = if *attempts >= Self::CAPTURE_RETRY_LIMIT {
                tracing::debug!(
                    htod = htod_after - htod_before,
                    attempts = *attempts,
                    "ROCm graph: geometry not capture-safe (persistent host round-trip); \
                     caching skip + running eager"
                );
                self.non_capture_safe.insert(
                    key.clone(),
                    RocmGraphFallbackReason::PersistentHostRoundTrip,
                );
                self.capture_retry.remove(&key);
                RocmGraphFallbackReason::PersistentHostRoundTrip
            } else {
                tracing::debug!(
                    htod = htod_after - htod_before,
                    attempts = *attempts,
                    "ROCm graph: warm pass did a host round-trip (likely cold cache fill); \
                     running eager, will retry capture next step"
                );
                RocmGraphFallbackReason::ColdCacheHostRoundTrip
            };
            return Ok(RocmCaptureStep::FallbackEager(fallback_reason));
        }
        // Capture-safe: clear any retry bookkeeping for this geometry.
        self.capture_retry.remove(&key);
        arena.borrow_mut().begin_replay();
        let capture_snapshot = linear_state
            .snapshot()
            .context("snapshot GDN recurrent state before capture pass")?;

        // Capture establishment makes a host-side success/rollback decision,
        // so these one-time waits remain explicit and attributed.
        attributed_rocm_graph_synchronize(
            "default_inputs_before_capture",
            "rocm_graph_capture_begin",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                kiln_tensor::rocm_synchronize_default_stream(device_idx)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        attributed_rocm_graph_synchronize(
            "capture_stream_before_begin",
            "rocm_graph_capture_begin",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                stream
                    .synchronize()
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;

        // === capture Pass 2 (Replay arena views, on the capture stream) ===
        let _ = &gdn_decode_outputs;
        stream
            .begin_capture()
            .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;
        let capture_result = kiln_tensor::with_rocm_capture_arena(arena.clone(), || {
            kiln_tensor::with_active_rocm_stream(stream.clone(), || {
                let hidden = model_forward_paged_hidden_with_graph_inputs(
                    backend,
                    &[token_id],
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    Some(linear_state),
                    lora,
                    &token_buffer,
                    &position_buffer,
                    graph_inputs.as_ref(),
                )?;
                kiln_tensor::rocm_slice_set_dim0(&output_hidden, &hidden, 0)
                    .context("ROCm graph: copy kt hidden into stable output_hidden")?;
                Ok::<(), anyhow::Error>(())
            })
        });
        let graph_result = stream.end_capture();
        if let Err(err) = capture_result {
            Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore graph-slot GDN state after capture-forward failure",
            )?;
            return Err(err).context("forward pass failed during graph capture");
        }
        drop(graph_inputs);

        let graph = match graph_result {
            Ok(graph) => graph,
            Err(err) => {
                Self::restore_linear_state_in_place(
                    linear_state,
                    &capture_snapshot,
                    "restore graph-slot GDN state after end-capture failure",
                )?;
                return Err(anyhow::anyhow!("end_capture failed: {err}"));
            }
        };
        let exec = match graph.instantiate() {
            Ok(exec) => exec,
            Err(err) => {
                Self::restore_linear_state_in_place(
                    linear_state,
                    &capture_snapshot,
                    "restore graph-slot GDN state after graph-instantiation failure",
                )?;
                return Err(anyhow::anyhow!("instantiate captured graph: {err}"));
            }
        };
        tracing::info!(
            "ROCm HIP graph captured for decode ({} layers)",
            config.num_layers
        );

        // Stream capture only RECORDED the forward; launch once now to actually
        // compute this step + advance state, then sync so output_hidden is valid.
        if let Err(err) = exec.launch(&stream) {
            Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore graph-slot GDN state after first-launch failure",
            )?;
            return Err(anyhow::anyhow!(
                "execute captured decode graph (first run): {err}"
            ));
        }
        if let Err(err) = attributed_rocm_graph_synchronize(
            "first_launch_completion",
            "rocm_graph_first_launch",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                stream
                    .synchronize()
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        ) {
            return Err(anyhow::anyhow!(
                "sync after first captured-graph launch: {err}"
            ));
        }

        let captured_hidden = output_hidden.clone();
        let max_seqlen_k = key.max_seqlen_k;
        let arena_buffers = arena.borrow_mut().take_retained();
        let replay_state = Self::replay_state_for_capture(
            &key,
            &output_hidden,
            &token_buffer,
            &position_buffer,
            block_table_buffer.as_ref(),
            seqused_k_buffer.as_ref(),
            kv_slot_buffer.as_ref(),
            &rotary_cos_buffer,
            &rotary_sin_buffer,
            &paged_decode_outputs,
            &paged_decode_lse,
            &gdn_decode_outputs,
        );
        self.captured.insert(
            RocmGraphCacheKey::new(owner, key),
            CapturedDecodeGraphRocm {
                _graph: graph,
                exec,
                output_hidden,
                capture_stream: stream,
                default_stream,
                replay_inputs_ready_event,
                replay_complete_event,
                adapter_gen: self.adapter_generation,
                kv_pool_identity: paged_cache.pool_identity(),
                token_buffer,
                position_buffer,
                block_table_buffer,
                seqused_k_buffer,
                kv_slot_buffer,
                rotary_cos_buffer,
                rotary_sin_buffer,
                _paged_decode_outputs: paged_decode_outputs,
                _paged_decode_lse: paged_decode_lse,
                max_seqlen_k,
                _gdn_decode_outputs: gdn_decode_outputs,
                _capture_arena_buffers: arena_buffers,
                replay_state,
            },
        );
        Ok(RocmCaptureStep::CapturedHidden(captured_hidden))
    }

    fn new_token_buffer(device: Device, token_id: u32) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![token_id], vec![1])
            .context("create ROCm graph token buffer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rocm")]
    #[test]
    fn graph_attention_geometry_matches_exact_fa2_splits() {
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        for attention_len in [1, kblock_n, kblock_n + 1, 2395, 2560, 2561] {
            let expected = attention_len.div_ceil(kblock_n) * kblock_n;
            assert_eq!(RocmGraphKey::exact_max_seqlen_k(attention_len), expected);
        }
        assert_eq!(RocmGraphKey::exact_max_seqlen_k(2395), 2432);
    }

    #[test]
    fn graph_slot_restore_preserves_tensor_handles() {
        let mut state = LinearAttentionState {
            recurrent_states: vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 2]).unwrap()],
            conv_states: vec![Tensor::from_vec(vec![3.0f32, 4.0], vec![1, 2]).unwrap()],
        };
        let snapshot = LinearAttentionState {
            recurrent_states: vec![Tensor::from_vec(vec![5.0f32, 6.0], vec![1, 2]).unwrap()],
            conv_states: vec![Tensor::from_vec(vec![7.0f32, 8.0], vec![1, 2]).unwrap()],
        };
        let recurrent_id = state.recurrent_states[0].id();
        let conv_id = state.conv_states[0].id();

        RocmGraphRunner::restore_linear_state_in_place(
            &mut state,
            &snapshot,
            "test graph-slot restore",
        )
        .unwrap();

        assert_eq!(state.recurrent_states[0].id(), recurrent_id);
        assert_eq!(state.conv_states[0].id(), conv_id);
        assert_eq!(
            state.recurrent_states[0].to_vec2::<f32>().unwrap(),
            vec![vec![5.0, 6.0]]
        );
        assert_eq!(
            state.conv_states[0].to_vec2::<f32>().unwrap(),
            vec![vec![7.0, 8.0]]
        );
    }

    #[cfg(feature = "rocm")]
    fn rocm_graph_test_fixture(
        device: &Device,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> (ModelConfig, GpuWeights) {
        use crate::forward::{
            GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights,
        };

        let config = ModelConfig {
            hidden_size: num_attention_heads * head_dim,
            num_layers: 1,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: num_attention_heads * head_dim * 2,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: 0,
            linear_key_head_dim: 0,
            linear_num_value_heads: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            partial_rotary_factor: 1.0,
        };
        let random_bf16 = |shape: Vec<usize>| {
            Tensor::randn(0.0_f32, 0.02, shape, device)
                .expect("random ROCm graph fixture tensor")
                .to_dtype(kiln_tensor::DType::BF16)
                .expect("cast ROCm graph fixture tensor")
        };
        let transposed = |tensor: &Tensor| {
            tensor
                .t()
                .expect("transpose ROCm graph fixture tensor")
                .contiguous()
                .expect("materialize ROCm graph fixture transpose")
        };

        let embed_tokens = random_bf16(vec![config.vocab_size, config.hidden_size]);
        let q_proj = random_bf16(vec![
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        ]);
        let k_proj = random_bf16(vec![
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
        ]);
        let v_proj = random_bf16(vec![
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
        ]);
        let o_proj = random_bf16(vec![
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        ]);
        let gate_proj = random_bf16(vec![config.intermediate_size, config.hidden_size]);
        let up_proj = random_bf16(vec![config.intermediate_size, config.hidden_size]);
        let down_proj = random_bf16(vec![config.hidden_size, config.intermediate_size]);
        let layer = GpuLayerWeights {
            input_layernorm: Tensor::ones(config.hidden_size, kiln_tensor::DType::BF16, device)
                .expect("input norm"),
            post_attention_layernorm: Tensor::ones(
                config.hidden_size,
                kiln_tensor::DType::BF16,
                device,
            )
            .expect("post-attention norm"),
            attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj_t: transposed(&q_proj),
                k_proj_t: transposed(&k_proj),
                v_proj_t: transposed(&v_proj),
                o_proj_t: transposed(&o_proj),
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: Tensor::ones(config.head_dim, kiln_tensor::DType::BF16, device)
                    .expect("q norm"),
                k_norm: Tensor::ones(config.head_dim, kiln_tensor::DType::BF16, device)
                    .expect("k norm"),
                qkv_proj_t: None,
                qkv_proj_w8: None,
                o_proj_w8: None,
                q_proj_marlin: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj_t: transposed(&gate_proj),
                up_proj_t: transposed(&up_proj),
                down_proj_t: transposed(&down_proj),
                gate_proj,
                up_proj,
                down_proj,
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
            },
        };
        let rotary_inv_freq =
            crate::forward::compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)
                .expect("rotary frequencies");
        let weights = GpuWeights {
            source_content_sha256: None,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            embed_tokens_t: transposed(&embed_tokens),
            embed_tokens,
            layers: vec![layer],
            final_norm: Tensor::ones(config.hidden_size, kiln_tensor::DType::BF16, device)
                .expect("final norm"),
            rotary_inv_freq,
            lm_head_w8: None,
            mtp: None,
        };
        (config, weights)
    }

    #[cfg(feature = "rocm")]
    fn stale_generation_test_fixture(device: &Device) -> (ModelConfig, GpuWeights) {
        rocm_graph_test_fixture(device, 2, 1, 128)
    }

    #[cfg(feature = "rocm")]
    fn hidden_f32(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_device(Device::Cpu)
            .expect("copy hidden to CPU")
            .to_dtype(kiln_tensor::DType::F32)
            .expect("cast hidden to f32")
            .to_vec()
            .expect("read hidden")
    }

    #[test]
    fn disabled_off_device() {
        let mut r = RocmGraphRunner::new(&Device::Cpu, true);
        r.release_decode_row(7);
        assert!(!r.is_enabled());
        assert_eq!(
            r.stats(),
            RocmGraphStats {
                requested: false,
                capture_requested: false,
                enabled: false,
                capture_enabled: false,
                ..RocmGraphStats::default()
            }
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn release_decode_row_discards_only_the_finished_graphless_slot() {
        let target = RocmGraphOwner::Slot(7);
        let survivor = RocmGraphOwner::Slot(8);
        let second_survivor = RocmGraphOwner::Slot(9);
        let mut runner = RocmGraphRunner::new(&Device::Cpu, true);
        for (owner, row_id) in [(target, 7), (survivor, 8), (second_survivor, 9)] {
            runner.graph_slots.insert(
                owner,
                RocmGraphSlotState {
                    assigned_row: Some(row_id),
                    linear_state: LinearAttentionState {
                        recurrent_states: Vec::new(),
                        conv_states: Vec::new(),
                    },
                },
            );
            runner.decode_row_slots.insert(row_id, owner);
        }
        runner.decode_timelines.insert(target, Default::default());
        runner.decode_timelines.insert(survivor, Default::default());
        runner
            .decode_timelines
            .insert(second_survivor, Default::default());
        assert_eq!(runner.stats().tracked_decode_owner_count, 3);
        runner.release_decode_row(7);

        assert!(!runner.graph_slots.contains_key(&target));
        assert!(runner.graph_slots.contains_key(&survivor));
        assert!(runner.graph_slots.contains_key(&second_survivor));
        assert!(!runner.decode_timelines.contains_key(&target));
        assert!(runner.decode_timelines.contains_key(&survivor));
        assert!(runner.decode_timelines.contains_key(&second_survivor));
        let stats = runner.stats();
        assert_eq!(stats.tracked_decode_owner_count, 2);
        assert_eq!(stats.graph_slot_count, 2);
        assert_eq!(stats.active_graph_slot_count, 2);
        assert_eq!(stats.decode_owner_release_count, 1);
        assert_eq!(stats.decode_owner_graph_release_count, 0);

        runner.release_decode_row(7);
        let stats = runner.stats();
        assert_eq!(stats.tracked_decode_owner_count, 2);
        assert_eq!(stats.decode_owner_release_count, 1);
        assert_eq!(stats.decode_owner_graph_release_count, 0);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn recycled_block_continuity_never_crosses_graph_slots() {
        let mut runner = RocmGraphRunner::new(&Device::Cpu, true);
        let recycled_table = BlockTable { blocks: vec![11] };
        let first = RocmGraphOwner::Slot(41);
        let second = RocmGraphOwner::Slot(42);

        assert!(!runner.prepare_owner_decode(first, 401, &recycled_table, 63));
        assert!(runner.prepare_owner_decode(first, 401, &recycled_table, 64));

        // Even though both block zero and sequence continuity match the prior
        // call, a different generation must start a fresh recurrent timeline.
        assert!(!runner.prepare_owner_decode(second, 402, &recycled_table, 65));
        assert!(runner.prepare_owner_decode(second, 402, &recycled_table, 66));

        runner.decode_timelines.remove(&second);
        assert!(!runner.prepare_owner_decode(second, 403, &recycled_table, 67));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn idle_graph_slot_refreshes_and_adopts_a_new_rows_state() {
        fn state(recurrent: [f32; 2], conv: [f32; 2]) -> LinearAttentionState {
            LinearAttentionState {
                recurrent_states: vec![
                    Tensor::from_vec(recurrent.to_vec(), (1, 2)).expect("recurrent state"),
                ],
                conv_states: vec![Tensor::from_vec(conv.to_vec(), (1, 2)).expect("conv state")],
            }
        }

        let key = RocmGraphKey {
            stable_metadata: true,
            seq_len: 0,
            block_table: Vec::new(),
            max_seqlen_k: 512,
            max_blocks_per_seq: 8,
        };
        let mut runner = RocmGraphRunner::new(&Device::Cpu, true);
        let mut first = state([1.0, 2.0], [3.0, 4.0]);
        let owner = runner
            .bind_decode_row_to_slot(1001, &key, &mut first)
            .expect("bind first row");
        assert!(RocmGraphRunner::linear_state_handles_match(
            &runner.graph_slots[&owner].linear_state,
            &first
        ));

        runner.decode_row_slots.remove(&1001);
        runner.graph_slots.get_mut(&owner).unwrap().assigned_row = None;
        let mut second = state([11.0, 12.0], [13.0, 14.0]);
        let rebound = runner
            .bind_decode_row_to_slot(1002, &key, &mut second)
            .expect("reuse slot for second row");
        assert_eq!(rebound, owner);
        assert!(RocmGraphRunner::linear_state_handles_match(
            &runner.graph_slots[&owner].linear_state,
            &second
        ));
        assert_eq!(
            second.recurrent_states[0].to_vec::<f32>().unwrap(),
            vec![11.0, 12.0]
        );
        assert_eq!(
            second.conv_states[0].to_vec::<f32>().unwrap(),
            vec![13.0, 14.0]
        );
        assert_eq!(runner.stats().graph_slot_create_count, 1);
        assert_eq!(runner.stats().graph_slot_reuse_count, 1);
    }

    /// A decode geometry that cannot use graph-stable native attention must
    /// remain eager. Capturing its sequence-length-shaped fallback would appear
    /// to succeed but would silently reuse the captured K/V length on replay.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn shape_dependent_attention_is_cached_as_typed_eager_fallback() {
        assert_eq!(
            std::env::var("KILN_QUALIFICATION").ok().as_deref(),
            Some("1"),
            "set KILN_QUALIFICATION=1 for the explicit hardware run"
        );
        assert!(kiln_tensor::rocm_is_available());

        let device = Device::Rocm(0);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = rocm_graph_test_fixture(&device, 1, 1, 64);
        let graph_cache = PagedKvCacheKt::new(
            1,
            4,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("shape-dependent graph cache");
        let eager_cache = PagedKvCacheKt::new(
            1,
            4,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("shape-dependent eager cache");
        let table = BlockTable { blocks: vec![0] };
        let mut graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("graph state");
        let mut eager_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("eager state");
        let mut graph_runner = RocmGraphRunner::new(&device, true);

        for seq_len in 1..=4 {
            let token_id = (seq_len % config.vocab_size) as u32;
            let guarded = graph_runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &table,
                    seq_len,
                    &mut graph_state,
                    None,
                    811,
                )
                .expect("guarded graph decode");
            let eager = RocmGraphRunner::eager_forward_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &eager_cache,
                &table,
                seq_len,
                &mut eager_state,
                None,
            )
            .expect("shape-dependent eager oracle");
            assert_eq!(hidden_f32(&guarded), hidden_f32(&eager));
        }

        let stats = graph_runner.stats();
        assert!(stats.enabled);
        assert_eq!(stats.capture_attempts, 1);
        assert_eq!(stats.capture_deferrals, 1);
        assert_eq!(stats.capture_successes, 0);
        assert_eq!(stats.replay_attempts, 0);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.captured_graph_count, 0);
        assert_eq!(stats.fallbacks.total, 3);
        assert_eq!(stats.fallbacks.shape_dependent_attention, 3);
    }

    /// Real gfx1151 parity over the graph lifecycle that serving exercises:
    /// bucket transitions, growing metadata, retained-prefix reuse, cancelled
    /// owner cleanup, and adapter-generation invalidation.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn graph_parity_across_buckets_prefix_cancellation_and_adapter_boundary() {
        assert_eq!(
            std::env::var("KILN_QUALIFICATION").ok().as_deref(),
            Some("1"),
            "set KILN_QUALIFICATION=1 for the explicit hardware run"
        );
        assert!(kiln_tensor::rocm_is_available());

        let device = Device::Rocm(0);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = stale_generation_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            1,
            16,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("graph parity paged cache");
        let eager_cache = PagedKvCacheKt::new(
            1,
            16,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("eager parity paged cache");
        let mut graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("graph state");
        let mut eager_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("eager state");
        let mut graph_runner = RocmGraphRunner::new(&device, true);
        assert!(graph_runner.stats().capture_enabled);

        let mut assert_hidden_step = |runner: &mut RocmGraphRunner, row_id: u64, seq_len: usize| {
            let block_count = (seq_len + 1).div_ceil(graph_cache.block_size());
            let table = BlockTable {
                blocks: (0..block_count as u32).collect(),
            };
            let token_id = (seq_len % config.vocab_size) as u32;
            let graph_hidden = runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &table,
                    seq_len,
                    &mut graph_state,
                    None,
                    row_id,
                )
                .expect("graph hidden");
            let eager_hidden = RocmGraphRunner::eager_forward_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &eager_cache,
                &table,
                seq_len,
                &mut eager_state,
                None,
            )
            .expect("eager hidden");
            let graph_values = hidden_f32(&graph_hidden);
            let eager_values = hidden_f32(&eager_hidden);
            assert_eq!(
                graph_values, eager_values,
                "hidden mismatch at row {row_id}, sequence {seq_len}"
            );
        };

        const FIRST_OWNER: u64 = 901;
        for seq_len in 1..=80 {
            assert_hidden_step(&mut graph_runner, FIRST_OWNER, seq_len);
        }
        let before_cancel = graph_runner.stats();
        assert!(before_cancel.capture_successes >= 2);
        assert!(before_cancel.replay_successes > 0);
        assert!(before_cancel.captured_graph_count >= 2);

        // Cancellation returns the first owner's graph slot to the bounded
        // reuse pool. A new row then starts from already-populated KV pages,
        // matching prefix-cache reuse without native graph churn.
        graph_runner.release_decode_row(FIRST_OWNER);
        let after_cancel = graph_runner.stats();
        assert_eq!(
            after_cancel.captured_graph_count,
            before_cancel.captured_graph_count
        );
        assert_eq!(after_cancel.active_graph_slot_count, 0);
        assert_eq!(after_cancel.idle_graph_slot_count, 1);
        assert_eq!(after_cancel.decode_owner_release_count, 1);
        assert_eq!(after_cancel.decode_owner_graph_release_count, 0);

        // Rebind the idle native graph to an unrelated request with fresh GDN
        // state and disjoint KV pages. This exercises the real ROCm in-place
        // state refresh before replay rather than only continuing a prefix.
        const FRESH_OWNER: u64 = 903;
        let mut fresh_graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("fresh graph state");
        let mut fresh_eager_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("fresh eager state");
        let fresh_table = BlockTable { blocks: vec![8] };
        for seq_len in 1..=4 {
            let token_id = ((seq_len + 17) % config.vocab_size) as u32;
            let graph_hidden = graph_runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &fresh_table,
                    seq_len,
                    &mut fresh_graph_state,
                    None,
                    FRESH_OWNER,
                )
                .expect("fresh graph hidden");
            let eager_hidden = RocmGraphRunner::eager_forward_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &eager_cache,
                &fresh_table,
                seq_len,
                &mut fresh_eager_state,
                None,
            )
            .expect("fresh eager hidden");
            assert_eq!(hidden_f32(&graph_hidden), hidden_f32(&eager_hidden));
        }
        graph_runner.release_decode_row(FRESH_OWNER);

        const PREFIX_OWNER: u64 = 902;
        for seq_len in 64..=72 {
            assert_hidden_step(&mut graph_runner, PREFIX_OWNER, seq_len);
        }
        let before_adapter_boundary = graph_runner.stats();
        assert!(before_adapter_boundary.captured_graph_count > 0);
        assert!(before_adapter_boundary.graph_slot_reuse_count >= 2);
        let captures_before_adapter_boundary = before_adapter_boundary.capture_successes;

        // Adapter swaps invalidate pointer-bearing graph state before the next
        // request quantum. The fixture keeps weights numerically unchanged so
        // eager equality remains an exact oracle for lifecycle behavior.
        graph_runner.invalidate();
        assert_eq!(graph_runner.stats().captured_graph_count, 0);
        for seq_len in 73..=78 {
            assert_hidden_step(&mut graph_runner, PREFIX_OWNER, seq_len);
        }
        drop(assert_hidden_step);

        let seq_len = 79usize;
        let table = BlockTable {
            blocks: (0..(seq_len + 1).div_ceil(graph_cache.block_size()) as u32).collect(),
        };
        let token_id = (seq_len % config.vocab_size) as u32;
        let graph_logits = graph_runner
            .decode_step_paged(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &graph_cache,
                &table,
                seq_len,
                &mut graph_state,
                None,
                PREFIX_OWNER,
            )
            .expect("graph logits");
        let eager_logits = RocmGraphRunner::eager_forward(
            backend.as_ref(),
            token_id,
            &weights,
            &config,
            &eager_cache,
            &table,
            seq_len,
            &mut eager_state,
            None,
        )
        .expect("eager logits");
        assert_eq!(hidden_f32(&graph_logits), hidden_f32(&eager_logits));

        let seq_len = 80usize;
        let table = BlockTable {
            blocks: (0..(seq_len + 1).div_ceil(graph_cache.block_size()) as u32).collect(),
        };
        let token_id = (seq_len % config.vocab_size) as u32;
        let graph_token = graph_runner
            .decode_step_paged_greedy(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &graph_cache,
                &table,
                seq_len,
                &mut graph_state,
                None,
                PREFIX_OWNER,
            )
            .expect("graph greedy token");
        let eager_token = RocmGraphRunner::eager_forward_greedy(
            backend.as_ref(),
            token_id,
            &weights,
            &config,
            &eager_cache,
            &table,
            seq_len,
            &mut eager_state,
            None,
        )
        .expect("eager greedy token");
        assert_eq!(graph_token, eager_token);

        let final_stats = graph_runner.stats();
        assert!(final_stats.capture_successes > captures_before_adapter_boundary);
        assert_eq!(final_stats.failures, 0);
        assert_eq!(final_stats.fallbacks.total, 0);
        eprintln!(
            "[rocm-graph-parity] captures={}; replays={}; owner_releases={}; graph_releases={}; fallbacks={}",
            final_stats.capture_successes,
            final_stats.replay_successes,
            final_stats.decode_owner_release_count,
            final_stats.decode_owner_graph_release_count,
            final_stats.fallbacks.total,
        );
    }

    /// Deliberately corrupt only the generation retained by a live graph. The
    /// physical allocation remains valid, so a missing guard would safely but
    /// observably reach `hipGraphLaunch` via the unit-build launch probe.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn stale_pool_generation_refuses_native_replay_and_falls_back_eager() {
        assert_eq!(
            std::env::var("KILN_QUALIFICATION").ok().as_deref(),
            Some("1"),
            "set KILN_QUALIFICATION=1 for the explicit hardware run"
        );
        assert!(
            kiln_tensor::rocm_is_available(),
            "ROCm qualification requested but no ROCm device is available"
        );

        let device = Device::Rocm(0);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = stale_generation_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            1,
            64,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("graph paged cache");
        let table = BlockTable {
            blocks: vec![0, 1, 2, 3],
        };
        let mut graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("graph linear state");
        let mut graph_runner = RocmGraphRunner::new(&device, true);
        assert!(graph_runner.stats().capture_enabled);

        let row_id = 77;
        let mut next_seq_len = 1usize;
        while graph_runner.stats().replay_successes == 0 && next_seq_len < 24 {
            let token_id = (next_seq_len % config.vocab_size) as u32;
            graph_runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &table,
                    next_seq_len,
                    &mut graph_state,
                    None,
                    row_id,
                )
                .expect("graph warmup/capture/replay hidden");
            next_seq_len += 1;
        }

        let before = graph_runner.stats();
        assert!(
            before.capture_successes > 0,
            "native capture never succeeded"
        );
        assert!(before.replay_successes > 0, "native replay never succeeded");
        assert_eq!(before.failures, 0);
        assert_eq!(before.captured_graph_count, 1);

        // Compute the fallback oracle on this same cache before corrupting only
        // the graph's retained identity. Repeating a full-attention step is
        // idempotent: it overwrites the same slot with the same token-derived
        // KV and reads the same prefix.
        let token_id = (next_seq_len % config.vocab_size) as u32;
        let mut reference_state = LinearAttentionState::new_for_inference(&config, &device)
            .expect("reference linear state");
        let eager_hidden = RocmGraphRunner::eager_forward_hidden(
            backend.as_ref(),
            token_id,
            &weights,
            &config,
            &graph_cache,
            &table,
            next_seq_len,
            &mut reference_state,
            None,
        )
        .expect("same-cache eager mismatch-step control");

        let live_identity = graph_cache.pool_identity();
        for captured in graph_runner.captured.values_mut() {
            assert_eq!(captured.kv_pool_identity, live_identity);
            captured.kv_pool_identity.generation = live_identity
                .generation
                .checked_add(1)
                .expect("test generation increment");
        }

        let native_launches_before =
            ROCM_TEST_NATIVE_REPLAY_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed);
        let guarded_hidden = graph_runner
            .decode_step_paged_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &graph_cache,
                &table,
                next_seq_len,
                &mut graph_state,
                None,
                row_id,
            )
            .expect("identity mismatch must take eager fallback");
        assert_eq!(hidden_f32(&guarded_hidden), hidden_f32(&eager_hidden));

        let after = graph_runner.stats();
        let native_launches_after =
            ROCM_TEST_NATIVE_REPLAY_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after.replay_attempts, before.replay_attempts + 1);
        assert_eq!(after.replay_successes, before.replay_successes);
        assert_eq!(after.replay_failures, before.replay_failures + 1);
        assert_eq!(after.fallbacks.total, before.fallbacks.total + 1);
        assert_eq!(
            after.fallbacks.replay_failure,
            before.fallbacks.replay_failure + 1
        );
        assert_eq!(after.failures, before.failures + 1);
        assert_eq!(native_launches_after, native_launches_before);
        assert!(
            !after.enabled,
            "mismatch must trip the graph circuit breaker"
        );
        assert_eq!(after.captured_graph_count, 0);
        assert_eq!(graph_cache.pool_identity(), live_identity);
        eprintln!(
            "[rocm-stale-generation] replay_attempts={} -> {}; replay_failures={} -> {}; native_launches={} -> {}",
            before.replay_attempts,
            after.replay_attempts,
            before.replay_failures,
            after.replay_failures,
            native_launches_before,
            native_launches_after,
        );
    }

    #[test]
    fn invalidate_bumps_generation_and_resets_warmup() {
        let mut r = RocmGraphRunner::new(&Device::Cpu, true);
        r.warmup_done = true;
        r.counters
            .record_capture_outcome(RocmGraphCaptureOutcome::Succeeded);
        r.counters.record_replay_outcome(false);
        let gen0 = r.adapter_generation;
        r.invalidate();
        assert_eq!(r.adapter_generation, gen0 + 1);
        assert!(!r.warmup_done);
        let stats = r.stats();
        assert_eq!(stats.capture_attempts, 1);
        assert_eq!(stats.capture_successes, 1);
        assert_eq!(stats.capture_deferrals, 0);
        assert_eq!(stats.capture_failures, 0);
        assert_eq!(stats.replay_attempts, 1);
        assert_eq!(stats.replay_successes, 0);
        assert_eq!(stats.replay_failures, 1);
        assert_eq!(stats.failures, 1);
        assert_eq!(stats.captured_graph_count, 0);
    }

    #[test]
    fn counters_record_monotonic_capture_and_replay_outcomes() {
        let mut counters = RocmGraphCounters::default();
        counters.record_capture_outcome(RocmGraphCaptureOutcome::Failed);
        counters.record_capture_outcome(RocmGraphCaptureOutcome::Deferred);
        counters.record_capture_outcome(RocmGraphCaptureOutcome::Succeeded);
        counters.record_replay_outcome(true);
        counters.record_replay_outcome(false);
        counters.record_decode_owner_release(2);
        counters.record_decode_owner_release(1);
        counters.record_fallback(
            RocmGraphFallbackReason::CaptureFailure,
            std::time::Duration::from_millis(50),
        );
        counters.record_fallback(
            RocmGraphFallbackReason::ReplayFailure,
            std::time::Duration::from_millis(120),
        );
        counters.record_fallback(
            RocmGraphFallbackReason::ReplayFailure,
            std::time::Duration::from_millis(10),
        );

        assert_eq!(counters.capture_attempts, 3);
        assert_eq!(counters.capture_successes, 1);
        assert_eq!(counters.capture_deferrals, 1);
        assert_eq!(counters.capture_failures, 1);
        assert_eq!(counters.replay_attempts, 2);
        assert_eq!(counters.replay_successes, 1);
        assert_eq!(counters.replay_failures, 1);
        assert_eq!(counters.decode_owner_release_count, 2);
        assert_eq!(counters.decode_owner_graph_release_count, 3);
        assert_eq!(counters.fallbacks.total, 3);
        assert_eq!(counters.fallbacks.capture_failure, 1);
        assert_eq!(counters.fallbacks.replay_failure, 2);
        assert_eq!(counters.fallbacks.slow, 1);
        assert_eq!(counters.fallbacks.total_duration_micros, 180_000);
        assert_eq!(counters.fallbacks.max_duration_micros, 120_000);
    }

    #[test]
    fn fallback_reason_labels_are_closed_and_distinct() {
        let labels = [
            RocmGraphFallbackReason::WarmupForwardFailure,
            RocmGraphFallbackReason::ColdCacheHostRoundTrip,
            RocmGraphFallbackReason::PersistentHostRoundTrip,
            RocmGraphFallbackReason::ShapeDependentAttention,
            RocmGraphFallbackReason::GraphCacheCapacity,
            RocmGraphFallbackReason::CriticalMemoryPressure,
            RocmGraphFallbackReason::CaptureFailure,
            RocmGraphFallbackReason::ReplayFailure,
        ]
        .map(RocmGraphFallbackReason::as_str);
        let mut distinct = labels.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), labels.len());
    }
}
