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
    fn new(block_table: &BlockTable, paged_cache: &PagedKvCacheKt, seq_len: usize) -> Self {
        let stable_metadata = Self::stable_paged_metadata_enabled();
        let attention_len = seq_len + 1;
        // Bucket + size by a multiple of FA2_KBLOCK_N (=64 for hdim256). A
        // coarser default avoids recapturing on every 64-token boundary while
        // preserving the padded block-table layout that the captured forward
        // expects. `seqused_k` still bounds the actual usable K/V length.
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let bucket_tokens = std::env::var("KILN_ROCM_GRAPH_KBLOCK_BUCKET_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= kblock_n && v % kblock_n == 0)
            .unwrap_or(512);
        let max_seqlen_k = attention_len.div_ceil(bucket_tokens) * bucket_tokens;
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
    DecodeRow(u64),
}

#[cfg(feature = "rocm")]
impl RocmGraphOwner {
    fn row_id(self) -> u64 {
        match self {
            Self::DecodeRow(row_id) => row_id,
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
fn remove_graphs_owned_by<T>(
    captured: &mut HashMap<RocmGraphCacheKey, T>,
    owner: RocmGraphOwner,
) -> usize {
    let before = captured.len();
    captured.retain(|key, _| key.owner != owner);
    before - captured.len()
}

#[cfg(feature = "rocm")]
#[derive(Default)]
struct RocmGraphOwnerTimeline {
    last_decode_seq_len: Option<usize>,
    last_decode_block0: Option<u32>,
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
    /// The non-default capture stream the graph launches on; synchronized after
    /// each launch so `output_hidden` is visible before the eager lm_head.
    capture_stream: std::sync::Arc<kiln_hip::RocmStream>,
    adapter_gen: u64,
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
        self.captured
            .exec
            .launch(&self.captured.capture_stream)
            .map_err(|e| CaptureError::Backend(format!("ROCm graph launch: {e}")))?;
        self.captured.capture_stream.synchronize().map_err(|e| {
            CaptureError::Backend(format!("sync capture stream after replay launch: {e}"))
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
    FallbackEager,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphCaptureOutcome {
    Succeeded,
    Deferred,
    Failed,
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
    /// Native replay launches that completed successfully.
    pub replay_successes: u64,
    /// Native replay launches that failed and tripped the circuit breaker.
    pub replay_failures: u64,
    /// Saturating sum of capture and replay failures.
    pub failures: u64,
    /// Finished decode owners whose tracked timeline or graph state was removed.
    pub decode_owner_release_count: u64,
    /// Captured graphs evicted by finished-owner cleanup.
    pub decode_owner_graph_release_count: u64,
    /// Graphs currently retained in the live cache.
    pub captured_graph_count: usize,
    /// Decode owners whose continuity timeline is currently retained.
    pub tracked_decode_owner_count: usize,
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
    /// Geometries whose decode forward does a host round-trip (a non-capture-safe
    /// fallback, e.g. GDN gates softplus) and so cannot be captured. Cached so we
    /// skip the warm pass + capture attempt for them on every subsequent step and
    /// go straight to eager — without disabling capture for OTHER (capture-safe)
    /// geometries.
    #[cfg(feature = "rocm")]
    non_capture_safe: std::collections::HashSet<RocmGraphKey>,
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
            non_capture_safe: std::collections::HashSet::new(),
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
            captured_graph_count,
            tracked_decode_owner_count,
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
            self.non_capture_safe.clear();
            self.capture_retry.clear();
            self.cache_full_warned = false;
            self.decode_timelines.clear();
        }
    }

    /// Release graph state owned by a finished logical decode row.
    ///
    /// Batching-engine rows and direct generations share one process-wide id
    /// namespace. Retaining their graphs or continuity timelines after request
    /// completion would permanently fill the bounded graph cache and grow the
    /// timeline map.
    pub fn release_decode_row(&mut self, row_id: u64) {
        #[cfg(feature = "rocm")]
        {
            let owner = RocmGraphOwner::DecodeRow(row_id);
            let evicted_graphs = remove_graphs_owned_by(&mut self.captured, owner);
            let removed_timeline = self.decode_timelines.remove(&owner).is_some();
            if evicted_graphs > 0 {
                self.cache_full_warned = false;
            }
            if evicted_graphs > 0 || removed_timeline {
                self.counters.record_decode_owner_release(evicted_graphs);
                tracing::debug!(
                    event = "rocm_graph_decode_owner_released",
                    row_id,
                    evicted_graphs,
                    removed_timeline,
                    tracked_decode_owner_count = self.decode_timelines.len(),
                    "rocm_graph_decode_owner_released"
                );
            }
        }
        #[cfg(not(feature = "rocm"))]
        let _ = row_id;
    }

    #[cfg(feature = "rocm")]
    fn prepare_owner_decode(
        &mut self,
        owner: RocmGraphOwner,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> bool {
        let block0 = block_table.blocks.first().copied();
        let owner_started = !self.decode_timelines.contains_key(&owner);
        if owner_started {
            tracing::debug!(
                event = "rocm_graph_decode_owner_started",
                row_id = owner.row_id(),
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
        if !continues {
            let evicted_graphs = remove_graphs_owned_by(&mut self.captured, owner);
            if evicted_graphs > 0 {
                tracing::debug!(
                    seq_len,
                    ?owner,
                    "ROCm graph: owner boundary — evicting captured bs=1 graph"
                );
            }
        }
        timeline.last_decode_seq_len = Some(seq_len);
        timeline.last_decode_block0 = block0;
        continues
    }

    #[cfg(feature = "rocm")]
    fn max_cached_graphs() -> usize {
        std::env::var("KILN_ROCM_GRAPH_CACHE_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(8)
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

            let owner = RocmGraphOwner::DecodeRow(graph_row_id);
            self.prepare_owner_decode(owner, block_table, seq_len);

            // Warmup: first decode step runs eagerly (graph-shaped position
            // buffer) to prime the allocator pools before the first capture.
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
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
                    }
                }
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

            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());

            // Geometry previously found non-capture-safe (host round-trip in its
            // forward) — skip the warm pass + capture attempt and run eager.
            if self.non_capture_safe.contains(&requested_key) {
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

            // Replay if we have a valid captured graph for this geometry.
            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
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

            // Memory-pressure guard: capturing a new graph mints freeze-pointer
            // arena + per-layer output buffers (a few MB). Under Critical memory
            // pressure (a coexisting job / training run has the VRAM), skip the
            // capture and run eager rather than risk the allocation tipping the
            // box into OOM — the governor sees all-process usage, so this respects
            // whatever else is on the GPU. Decode stays correct either way.
            if kiln_memory::MemoryGovernor::global().pressure()
                == kiln_memory::MemoryPressure::Critical
            {
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

            // Capture.
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
                    // A failed capture can leave pending/poisoned device work (an
                    // aborted capture stream, a half-issued kernel). Synchronize
                    // the device before the eager fallback so it runs from a clean
                    // state rather than cascading into a second failure.
                    if let Some(idx) = weights.embed_tokens.device().index() {
                        if let Err(sync_err) = kiln_tensor::rocm_synchronize_default_stream(idx) {
                            tracing::warn!("post-capfail device sync failed: {sync_err:#}");
                        }
                    }
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
            let owner = RocmGraphOwner::DecodeRow(graph_row_id);
            self.prepare_owner_decode(owner, block_table, seq_len);

            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
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
                    }
                }
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

            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());

            if self.non_capture_safe.contains(&requested_key) {
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

            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
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

            if kiln_memory::MemoryGovernor::global().pressure()
                == kiln_memory::MemoryPressure::Critical
            {
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
                Ok(RocmCaptureStep::FallbackEager) => {
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
                Err(e) => {
                    tracing::warn!("ROCm graph capture failed: {e:#}, disabling graphs (eager)");
                    self.enabled = false;
                    if let Some(idx) = weights.embed_tokens.device().index() {
                        if let Err(sync_err) = kiln_tensor::rocm_synchronize_default_stream(idx) {
                            tracing::warn!("post-capfail device sync failed: {sync_err:#}");
                        }
                    }
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
            let owner = RocmGraphOwner::DecodeRow(graph_row_id);
            self.prepare_owner_decode(owner, block_table, seq_len);

            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
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
                    }
                }
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

            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());

            if self.non_capture_safe.contains(&requested_key) {
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

            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
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

            if kiln_memory::MemoryGovernor::global().pressure()
                == kiln_memory::MemoryPressure::Critical
            {
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
                    if let Some(idx) = weights.embed_tokens.device().index() {
                        if let Err(sync_err) = kiln_tensor::rocm_synchronize_default_stream(idx) {
                            tracing::warn!("post-capfail device sync failed: {sync_err:#}");
                        }
                    }
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

        // The per-replay writes above land on the kt DEFAULT stream; the graph
        // launches on its non-default capture stream. Sync the default stream so
        // the writes are visible before launch (else replay reads a stale token).
        if let Some(idx) = captured.token_buffer.device().index() {
            kiln_tensor::rocm_synchronize_default_stream(idx)
                .context("sync per-replay input writes before ROCm graph launch")?;
        }

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
            RocmCaptureStep::FallbackEager => Self::eager_forward(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ),
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
            RocmCaptureStep::FallbackEager => Self::eager_forward_greedy(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ),
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
            Ok(RocmCaptureStep::FallbackEager) => RocmGraphCaptureOutcome::Deferred,
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
        let stream = kiln_tensor::primary_rocm_context(device_idx)
            .context("ROCm graph capture: primary_rocm_context for capture stream")?
            .new_stream()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create capture stream: {e}"))?;

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
        kiln_tensor::rocm_synchronize_default_stream(device_idx)
            .context("ROCm graph capture: sync kt default stream before warm pass")?;
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
        let warm_sync_result = stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync capture stream after ROCm graph warm pass: {e}"));
        if let Err(err) = warm_result {
            if let Err(sync_err) = warm_sync_result {
                tracing::warn!("post-warm-failure capture stream sync failed: {sync_err:#}");
            }
            *linear_state = gdn_snapshot;
            return Err(err).context("freeze-pointers warm (Record) pass failed");
        }
        if let Err(sync_err) = warm_sync_result {
            *linear_state = gdn_snapshot;
            return Err(sync_err);
        }
        // Restore the GDN recurrent state so the captured pass advances it once.
        *linear_state = gdn_snapshot;
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
            if *attempts >= Self::CAPTURE_RETRY_LIMIT {
                tracing::debug!(
                    htod = htod_after - htod_before,
                    attempts = *attempts,
                    "ROCm graph: geometry not capture-safe (persistent host round-trip); \
                     caching skip + running eager"
                );
                self.non_capture_safe.insert(key.clone());
                self.capture_retry.remove(&key);
            } else {
                tracing::debug!(
                    htod = htod_after - htod_before,
                    attempts = *attempts,
                    "ROCm graph: warm pass did a host round-trip (likely cold cache fill); \
                     running eager, will retry capture next step"
                );
            }
            return Ok(RocmCaptureStep::FallbackEager);
        }
        // Capture-safe: clear any retry bookkeeping for this geometry.
        self.capture_retry.remove(&key);
        arena.borrow_mut().begin_replay();
        let capture_snapshot = linear_state
            .snapshot()
            .context("snapshot GDN recurrent state before capture pass")?;

        // The buffer allocs filled their contents via H2D on the kt DEFAULT
        // stream; sync it so those fills are visible to the captured forward.
        kiln_tensor::rocm_synchronize_default_stream(device_idx)
            .context("ROCm graph capture: sync kt default stream before capture")?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync capture stream before begin_capture: {e}"))?;

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
            *linear_state = capture_snapshot;
            return Err(err).context("forward pass failed during graph capture");
        }
        drop(graph_inputs);

        let graph = match graph_result {
            Ok(graph) => graph,
            Err(err) => {
                *linear_state = capture_snapshot;
                return Err(anyhow::anyhow!("end_capture failed: {err}"));
            }
        };
        let exec = match graph.instantiate() {
            Ok(exec) => exec,
            Err(err) => {
                *linear_state = capture_snapshot;
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
            *linear_state = capture_snapshot;
            return Err(anyhow::anyhow!(
                "execute captured decode graph (first run): {err}"
            ));
        }
        if let Err(err) = stream.synchronize() {
            *linear_state = capture_snapshot;
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
                adapter_gen: self.adapter_generation,
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
    fn release_decode_row_removes_only_the_finished_owner() {
        fn graph_key(seq_len: usize) -> RocmGraphKey {
            RocmGraphKey {
                stable_metadata: false,
                seq_len,
                block_table: vec![seq_len as u32],
                max_seqlen_k: 512,
                max_blocks_per_seq: 8,
            }
        }

        let target = RocmGraphOwner::DecodeRow(7);
        let survivor = RocmGraphOwner::DecodeRow(8);
        let second_survivor = RocmGraphOwner::DecodeRow(9);
        let mut captured = HashMap::from([
            (RocmGraphCacheKey::new(target, graph_key(1)), "target-a"),
            (RocmGraphCacheKey::new(target, graph_key(2)), "target-b"),
            (RocmGraphCacheKey::new(survivor, graph_key(1)), "survivor"),
            (
                RocmGraphCacheKey::new(second_survivor, graph_key(1)),
                "second-survivor",
            ),
        ]);
        assert_eq!(remove_graphs_owned_by(&mut captured, target), 2);
        assert_eq!(captured.len(), 2);
        assert!(captured.keys().all(|key| key.owner != target));
        assert!(captured.keys().any(|key| key.owner == survivor));
        assert!(captured.keys().any(|key| key.owner == second_survivor));

        let mut runner = RocmGraphRunner::new(&Device::Cpu, true);
        runner.decode_timelines.insert(target, Default::default());
        runner.decode_timelines.insert(survivor, Default::default());
        runner
            .decode_timelines
            .insert(second_survivor, Default::default());
        assert_eq!(runner.stats().tracked_decode_owner_count, 3);
        runner.release_decode_row(7);

        assert!(!runner.decode_timelines.contains_key(&target));
        assert!(runner.decode_timelines.contains_key(&survivor));
        assert!(runner.decode_timelines.contains_key(&second_survivor));
        let stats = runner.stats();
        assert_eq!(stats.tracked_decode_owner_count, 2);
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
    fn recycled_block_continuity_never_crosses_decode_owners() {
        let mut runner = RocmGraphRunner::new(&Device::Cpu, true);
        let recycled_table = BlockTable { blocks: vec![11] };
        let first = RocmGraphOwner::DecodeRow(41);
        let second = RocmGraphOwner::DecodeRow(42);

        assert!(!runner.prepare_owner_decode(first, &recycled_table, 63));
        assert!(runner.prepare_owner_decode(first, &recycled_table, 64));

        // Even though both block zero and sequence continuity match the prior
        // call, a different generation must start a fresh recurrent timeline.
        assert!(!runner.prepare_owner_decode(second, &recycled_table, 65));
        assert!(runner.prepare_owner_decode(second, &recycled_table, 66));

        runner.release_decode_row(42);
        assert!(!runner.prepare_owner_decode(second, &recycled_table, 67));
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

        assert_eq!(counters.capture_attempts, 3);
        assert_eq!(counters.capture_successes, 1);
        assert_eq!(counters.capture_deferrals, 1);
        assert_eq!(counters.capture_failures, 1);
        assert_eq!(counters.replay_attempts, 2);
        assert_eq!(counters.replay_successes, 1);
        assert_eq!(counters.replay_failures, 1);
        assert_eq!(counters.decode_owner_release_count, 2);
        assert_eq!(counters.decode_owner_graph_release_count, 3);
    }
}
