//! CUDA graph capture and replay for decode forward passes.
//!
//! During decode, each step processes exactly one token with identical tensor
//! shapes, making it a candidate for CUDA graph capture. Recording the kernel
//! sequence once and replaying it eliminates per-step CPU-side kernel launch
//! overhead for a 10-15% decode throughput improvement.
//!
//! ## How it works
//!
//! 1. **Warmup**: First decode step runs eagerly to prime GPU allocator pools.
//! 2. **Capture**: Second decode step is run under CUDA stream capture. All GPU
//!    operations (kernel launches, allocations via `cuMemAllocAsync`, memcpies)
//!    are recorded into a graph.
//! 3. **Replay**: Subsequent steps replay the captured graph. On Ampere+ GPUs,
//!    `cuMemAllocAsync` nodes allocate at the same device addresses, so all
//!    kernel arguments remain valid.
//!
//! ## Position buffer for RoPE
//!
//! RoPE requires position indices that change every decode step. A pre-allocated
//! GPU tensor holds the position value; its contents are updated via
//! `cudaMemcpyHtoDAsync` (outside the graph) before each replay. The captured
//! graph reads from the same device pointer but sees the updated position,
//! producing correct rotary embeddings at every step.
//!
//! ## Limitations
//!
//! - Only applies to single-token decode steps, not variable-length prefill.
//! - Requires `cuMemAllocAsync` support (Ampere+ / compute capability ≥ 8.0).
//! - Graph is invalidated on LoRA adapter swap (different weight pointers).
//! - Falls back gracefully to eager execution if capture fails.
//!
//! ## Multi-batch (`bs > 1`) capture is unavailable
//!
//! A batched implementation remains in-tree for continued engineering, but
//! real concurrent serving poisoned the CUDA context during capture/replay.
//! `is_batched_enabled()` therefore returns false unconditionally. There is no
//! process-environment opt-in; re-entry requires a source change plus NVIDIA
//! sanitizer, parity, resilience, and throughput evidence. The healthy eager
//! batched path remains authoritative meanwhile.
//!
//! Every graph route requires stable paged metadata. Block-table,
//! sequence-length, KV-slot, rotary, and attention-output buffers are retained
//! and refreshed in place; the transient-metadata mode that caused stale
//! pointer faults has been removed rather than exposed as configuration.

use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use tracing;

use kiln_core::config::ModelConfig;

use crate::PagedKvCacheKt;
use crate::backend::BackendRuntime;
#[cfg(feature = "cuda")]
use crate::execution_phase::{GraphPhase, GraphPhaseTimer};
#[cfg(feature = "cuda")]
use crate::forward::PagedDecodeGraphInputs;
#[cfg(feature = "cuda")]
use crate::forward::model_forward_paged_hidden_with_graph_inputs;
use crate::forward::{GpuWeights, LinearAttentionState, model_forward_paged};
use crate::lora_loader::LoraWeights;
#[cfg(feature = "cuda")]
use kiln_graph::{
    CaptureError, InvalidateReason, ReplayInputs, ReplayKey, ReplayOutputs, ReplayPlan,
    ReplayResourceStability, ReplayState, ResidentResourceRef,
};
#[cfg(feature = "cuda")]
use kiln_tensor::Backend;

// #1082: the CUDA-graph stable device buffers are now kt-native
// `kiln_tensor::Tensor`s (post-flip convention: bare `Tensor` = kt).
// Allocated once with stable device pointers (`Tensor::{zeros_on,
// from_vec_on}`) and refreshed in place before each replay via
// `kiln_tensor::cuda_write_host_in_place` — both honor the captured
// graph's baked device pointer.
#[cfg(feature = "cuda")]
use kiln_tensor::Device;
use kiln_tensor::Tensor;

use kiln_core::block::BlockTable;

/// Immutable CUDA decode-graph policy installed with a model runner.
///
/// Product owners resolve configuration before device selection and inject the
/// result here. Decode never re-reads process environment, so graph behavior
/// cannot change underneath an in-flight request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaGraphExecutionPolicy {
    enabled: bool,
    max_cached_graphs: usize,
}

impl CudaGraphExecutionPolicy {
    pub const DEFAULT_MAX_CACHED_GRAPHS: usize = 8;
    pub const MAX_CACHED_GRAPHS: usize = 64;

    /// Eager CUDA decode with the default dormant cache bound.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            max_cached_graphs: Self::DEFAULT_MAX_CACHED_GRAPHS,
        }
    }

    /// Build a policy from validated product configuration.
    pub fn try_new(enabled: bool, max_cached_graphs: usize) -> Result<Self> {
        anyhow::ensure!(
            max_cached_graphs > 0,
            "CUDA graph cache capacity must be greater than zero"
        );
        anyhow::ensure!(
            max_cached_graphs <= Self::MAX_CACHED_GRAPHS,
            "CUDA graph cache capacity must not exceed {}",
            Self::MAX_CACHED_GRAPHS
        );
        Ok(Self {
            enabled,
            max_cached_graphs,
        })
    }

    pub const fn enabled(self) -> bool {
        self.enabled
    }

    pub const fn max_cached_graphs(self) -> usize {
        self.max_cached_graphs
    }
}

impl Default for CudaGraphExecutionPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Holds a captured CUDA graph ready for replay.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaGraphKey {
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
}

#[cfg(feature = "cuda")]
impl CudaGraphKey {
    fn new(paged_cache: &PagedKvCacheKt, seq_len: usize) -> Self {
        let attention_len = seq_len + 1;
        // #1082: bucket + size by FA2_KBLOCK_N (=64 for hdim256), NOT a hardcoded
        // 128. Must match `forward.rs::try_flash_attn_paged_decode`'s K_BLOCK_N
        // exactly — otherwise the captured graph's block-table buffer is sized
        // differently from the table the forward actually builds, and replay
        // reads OOB → CUDA_ERROR_ILLEGAL_ADDRESS (the block_size=64 graph crash).
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let max_seqlen_k = attention_len.div_ceil(kblock_n) * kblock_n;
        let pages_per_chunk = kblock_n / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        Self {
            max_seqlen_k,
            max_blocks_per_seq,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CudaGraphOwner {
    Anonymous,
    DecodeRow(u64),
}

#[cfg(feature = "cuda")]
impl CudaGraphOwner {
    fn from_row_id(row_id: Option<u64>) -> Self {
        row_id.map_or(Self::Anonymous, Self::DecodeRow)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaGraphCacheKey {
    owner: CudaGraphOwner,
    graph: CudaGraphKey,
}

#[cfg(feature = "cuda")]
impl CudaGraphCacheKey {
    fn new(owner: CudaGraphOwner, graph: CudaGraphKey) -> Self {
        Self { owner, graph }
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct CudaGraphOwnerTimeline {
    last_decode_seq_len: Option<usize>,
    last_decode_block0: Option<u32>,
}

/// Cache key for the (planned, not-yet-wired) batched (`bs > 1`) decode
/// graph cache. Mirrors [`CudaGraphKey`] but with an explicit
/// `batch_size` bucket. See the multi-batch design note at the top of
/// this file for the surrounding plan.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaBatchedGraphKey {
    /// Number of rows the captured graph was specialized for.
    batch_size: usize,
    /// K/V geometry bucket shared by rows in the captured graph.
    max_seqlen_k: usize,
    /// Padded block-table width, in physical pages.
    max_blocks_per_seq: usize,
}

#[cfg(feature = "cuda")]
impl CudaBatchedGraphKey {
    /// Build a batched key from the same primitives used by
    /// [`CudaGraphKey::new`], applied to the largest seq_len in the
    /// batch (rounded up to the 128 K/V chunk). Bucketing all rows to
    /// the same `max_seqlen_k` lets one captured graph serve every
    /// row at that decode step.
    fn new(batch_size: usize, max_seq_len: usize, paged_cache: &PagedKvCacheKt) -> Self {
        let attention_len = max_seq_len + 1;
        // #1082: bucket + size by FA2_KBLOCK_N (=64 for hdim256), NOT a hardcoded
        // 128. Must match `forward.rs::try_flash_attn_paged_decode`'s K_BLOCK_N
        // exactly — otherwise the captured graph's block-table buffer is sized
        // differently from the table the forward actually builds, and replay
        // reads OOB → CUDA_ERROR_ILLEGAL_ADDRESS (the block_size=64 graph crash).
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let max_seqlen_k = attention_len.div_ceil(kblock_n) * kblock_n;
        let pages_per_chunk = kblock_n / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        Self {
            batch_size,
            max_seqlen_k,
            max_blocks_per_seq,
        }
    }
}

#[cfg(feature = "cuda")]
struct CapturedDecodeGraph {
    /// The instantiated CUDA graph.
    graph: cudarc::driver::CudaGraph,
    /// #1082 box-102 FIX: graph-stable PRE-final-norm hidden buffer, shape
    /// `[1, 1, hidden_size]`. The captured `HiddenOnly` forward writes here via
    /// `cuda_slice_set_dim0`; its storage is refreshed in place on every replay.
    /// `final_norm` + lm_head then run EAGERLY on this buffer (off the graph,
    /// via `crate::forward::lm_head_from_hidden_eager`) because the lm_head
    /// cublasLt GEMV is not CUDA-graph-replay-safe (the BUG2 doubling).
    output_hidden: Tensor,
    /// The non-default capture stream the graph launches on. Retained so the
    /// replay path can `synchronize()` it after `graph.launch()` — making the
    /// freshly-written `output_hidden` visible before the eager lm_head reads
    /// it on the default stream. (#1082 box-102 FIX)
    capture_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// Adapter generation when captured (invalidate on mismatch).
    adapter_gen: u64,
    /// Exact paged-KV allocation/generation whose pool pointers are embedded in
    /// the captured kernels.
    kv_pool_identity: crate::KvPoolIdentity,
    /// Pre-allocated token-id buffer on GPU (u32, shape [1]).
    /// Updated before each replay so embedding lookup reads the current token
    /// from a graph-stable device pointer.
    token_buffer: Tensor,
    /// Pre-allocated position buffer on GPU (f32, shape [1]).
    /// Updated via `cuda_write_host_in_place` before each replay so RoPE sees
    /// the correct position while reading from the same device pointer.
    position_buffer: Tensor,
    /// Pre-allocated padded block table buffer on GPU (u32, shape [1, max_blocks_per_seq]).
    /// Updated before replay so paged attention reads current page metadata from
    /// a graph-stable pointer.
    block_table_buffer: Tensor,
    /// Pre-allocated actual K/V attention length buffer on GPU.
    /// #1082: U32 (the kt flash-attn path requires `seqused_k` U32 — same
    /// 4-byte layout the candle i32 buffer carried, which the candle->kt
    /// borrow reinterpreted as U32 anyway).
    seqused_k_buffer: Tensor,
    /// Pre-allocated current KV write slot buffer on GPU (u32, shape [1]).
    kv_slot_buffer: Tensor,
    /// Pre-allocated RoPE cosine table on GPU (f32, shape [1, rotary_dim / 2]).
    /// Updated before replay so RoPE consumes graph-stable table pointers.
    rotary_cos_buffer: Tensor,
    /// Pre-allocated RoPE sine table on GPU (f32, shape [1, rotary_dim / 2]).
    rotary_sin_buffer: Tensor,
    /// Pre-allocated paged FlashAttention outputs, one per full-attention layer.
    /// The CUDA graph captures these destination pointers; replay must not
    /// write into capture-time temporary allocations that can be freed.
    _paged_decode_outputs: Vec<Tensor>,
    /// Pre-allocated paged FlashAttention LSE scratch tensors, one per
    /// full-attention layer, for the same graph-stable destination reason.
    _paged_decode_lse: Vec<Tensor>,
    /// Max K/V length baked into the captured kernel launch shape.
    max_seqlen_k: usize,
    /// Pre-allocated fused GDN decode recurrent outputs, one per linear layer.
    /// Their device pointers are captured by the graph and must stay alive for
    /// replay.
    _gdn_decode_outputs: Vec<Tensor>,
    /// #1082 freeze-pointers (Phase 5): every forward intermediate the captured
    /// graph touches — Q/K/V projections, per-layer activations, the
    /// `Flash_fwd_params` backing pointers — is handed to the forward as a
    /// `Borrowed` view into a buffer the capture arena owns. Those owned buffers
    /// are retained here so their device pointers stay mapped for every replay.
    /// Generalizes the per-buffer pinning above to ALL intermediates — the
    /// structural fix for the `flash_fwd_splitkv_kernel` ILLEGAL_ADDRESS that
    /// compute-sanitizer pinned (freed Q/activation read on replay).
    _capture_arena_buffers: Vec<std::sync::Arc<kiln_tensor::CudaStorage>>,
    /// Shared graph-layer replay contract state captured alongside the native
    /// CUDA graph. The production replay path validates this before launching.
    replay_state: ReplayState,
}

#[cfg(feature = "cuda")]
struct CudaDecodeReplayPlan<'a> {
    captured: &'a CapturedDecodeGraph,
}

#[cfg(feature = "cuda")]
impl<'a> CudaDecodeReplayPlan<'a> {
    fn new(captured: &'a CapturedDecodeGraph) -> Self {
        Self { captured }
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaDecodeReplayPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDecodeReplayPlan")
            .field("key", &self.captured.replay_state.key)
            .finish_non_exhaustive()
    }
}

// SAFETY: `CudaDecodeReplayPlan` is a short-lived adapter created inside
// `CudaGraphRunner::decode_step_paged`, whose runner is protected by the
// `ModelRunner` mutex. It only borrows an already-captured graph and stable
// device buffers. The raw CUDA graph handles are opaque driver objects, are not
// dereferenced on the CPU, and replay requires `&mut self`, so launches through
// this adapter remain serialized by the runner path.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaDecodeReplayPlan<'_> {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaDecodeReplayPlan<'_> {}

#[cfg(feature = "cuda")]
impl ReplayPlan for CudaDecodeReplayPlan<'_> {
    fn backend(&self) -> Backend {
        Backend::Cuda
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
            .graph
            .launch()
            .map_err(|e| CaptureError::Backend(format!("CUDA graph launch: {e}")))?;
        let device_index = self
            .captured
            .output_hidden
            .device()
            .index()
            .ok_or_else(|| CaptureError::Backend("CUDA graph output has no device index".into()))?;
        kiln_tensor::cuda_synchronize_stream_for(
            device_index,
            &self.captured.capture_stream,
            kiln_tensor::CudaSyncReason::GraphBoundary,
        )
        .map_err(|e| {
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

/// Captured graph + stable buffers for a batched (`bs > 1`) decode step.
/// The fields mirror [`CapturedDecodeGraph`] but every per-row tensor is
/// shaped for `[batch, ...]` so one graph replay services the whole batch.
///
/// #1082 boxes 432/433 (STEPS 2-3): the captured graph now records the
/// `HiddenOnly` batched transformer (no in-graph final_norm / lm_head);
/// `final_norm` + the large-N (`vocab = 151936`) cublasLt lm_head GEMV run
/// EAGERLY on [`Self::output_hidden`] after each launch. This is the batched
/// port of the bs=1 box-102 BUG2 fix — the lm_head GEMV is not
/// CUDA-graph-replay-deterministic at large N. The former in-graph
/// `output_logits` + `_lm_head_output_buffer` fields are gone.
#[cfg(feature = "cuda")]
struct CapturedBatchedDecodeGraph {
    /// The instantiated CUDA graph.
    graph: cudarc::driver::CudaGraph,
    /// #1082 box-102 BUG2 fix port (STEPS 2-3): graph-stable PRE-final-norm
    /// hidden buffer, shape `[batch, 1, hidden_size]`. The captured
    /// `HiddenOnly` batched forward
    /// ([`crate::forward::model_forward_paged_batched_hidden_with_graph_inputs`])
    /// writes here via `cuda_slice_set_dim0`; `final_norm` + lm_head then run
    /// EAGERLY on this buffer (off the graph, via
    /// [`crate::forward::lm_head_from_batched_hidden_eager`]) because the
    /// large-N lm_head cublasLt GEMV is not CUDA-graph-replay-safe (the
    /// BUG2 doubling). Mirrors [`CapturedDecodeGraph::output_hidden`].
    output_hidden: Tensor,
    /// The non-default capture stream the batched graph launches on.
    /// Retained so the replay path can `synchronize()` it after
    /// `graph.launch()` — making the freshly-written `output_hidden`
    /// visible before the eager batched lm_head reads it on the default
    /// stream. Mirrors [`CapturedDecodeGraph::capture_stream`] (#1082
    /// box-102 fix port).
    capture_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// Adapter generation when captured (invalidate on mismatch).
    adapter_gen: u64,
    /// Exact paged-KV allocation/generation whose pool pointers are embedded in
    /// the captured kernels.
    kv_pool_identity: crate::KvPoolIdentity,
    /// `[batch]` u32 token-id buffer; updated before replay.
    token_buffer: Tensor,
    /// `[batch]` f32 per-row decode position; updated before replay.
    position_buffer: Tensor,
    /// `[batch, max_blocks_per_seq]` u32 padded block table.
    block_table_buffer: Tensor,
    /// `[batch]` per-row K/V length. #1082: U32 (kt flash-attn contract).
    seqused_k_buffer: Tensor,
    /// `[batch]` u32 per-row current KV-write slot.
    kv_slot_buffer: Tensor,
    /// `[batch, rotary_dim / 2]` RoPE cos table; updated before replay.
    rotary_cos_buffer: Tensor,
    /// `[batch, rotary_dim / 2]` RoPE sin table; updated before replay.
    rotary_sin_buffer: Tensor,
    /// Per-full-attn-layer paged decode outputs, shape `[batch, 1, n_heads, head_dim]`.
    _paged_decode_outputs: Vec<Tensor>,
    /// Per-full-attn-layer LSE scratch, shape `[batch, n_heads, 1]`.
    _paged_decode_lse: Vec<Tensor>,
    /// Per-GDN-layer fused recurrent outputs, shape `[batch, ...]`.
    _gdn_decode_outputs: Vec<Tensor>,
    /// #1082 freeze-pointers (Phase 5, batched port): every forward
    /// intermediate the captured graph touches — Q/K/V projections, per-layer
    /// activations, the `Flash_fwd_params` backing pointers — is handed to the
    /// batched forward as a `Borrowed` view into a buffer the capture arena
    /// owns. Those owned buffers are retained here so their device pointers
    /// stay mapped for every replay. Mirrors
    /// [`CapturedDecodeGraph::_capture_arena_buffers`] — the structural fix
    /// for the `flash_fwd_splitkv_kernel` ILLEGAL_ADDRESS (freed
    /// Q/activation read on replay).
    _capture_arena_buffers: Vec<std::sync::Arc<kiln_tensor::CudaStorage>>,
    /// Max K/V length baked into the captured kernel launch shape.
    max_seqlen_k: usize,
    // NOTE: the captured graph reads GDN recurrent/conv state via the
    // device pointers carried by the runner's `batched_state_pool` slot
    // for this `batch_size`. The pool entry stays alive for the
    // runner's lifetime, which always outlives the captured graph (we
    // drop the captured map first on invalidate). No extra field
    // needed here.
}

/// Manages CUDA graph lifecycle for decode forward passes.
pub struct CudaGraphRunner {
    /// Immutable startup policy supplied by the owning product.
    // Read by the cuda-lane `decode_step_paged` cache-bound check; never
    // read in the default build, so the allow is required.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    policy: CudaGraphExecutionPolicy,
    /// Whether CUDA graphs are enabled.
    /// Runtime capture failure may permanently lower this from the policy.
    enabled: bool,
    /// Captured bs=1 graphs keyed by both decode-row owner and graph geometry.
    /// The graph carries the row's recurrent state through graph-owned buffers,
    /// so geometry alone is not a valid sharing key under interleaved serving.
    #[cfg(feature = "cuda")]
    captured: HashMap<CudaGraphCacheKey, CapturedDecodeGraph>,
    /// Captured batched graphs keyed on `(batch_size, max_seqlen_k, …)`.
    /// Empty today; populated by the planned multi-batch capture path.
    #[cfg(feature = "cuda")]
    captured_batched: HashMap<CudaBatchedGraphKey, CapturedBatchedDecodeGraph>,
    /// Per-batch-size warmup tracker. Each new bucket needs one eager
    /// call to prime the allocator before its first capture attempt;
    /// without per-bucket warmup the global `warmup_done` flag set by
    /// an earlier bs=1 capture caused new batched buckets to capture
    /// against a cold allocator and hit `CUDA_ERROR_ILLEGAL_ADDRESS`.
    #[cfg(feature = "cuda")]
    batched_bucket_warmup_done: std::collections::HashSet<usize>,
    /// Persistent batched [`LinearAttentionState`] pool, one slot per
    /// `batch_size` bucket. The captured-batched forward reads the GDN
    /// recurrent / conv state from the slot's device pointers, which
    /// must remain stable across replays. Callers refresh the slot's
    /// contents in-place via the existing
    /// `assemble_gdn_recurrent_resident_batch_rows` primitive (and the
    /// conv-state equivalent landed alongside the forward wrapper)
    /// before each replay.
    #[cfg(feature = "cuda")]
    batched_state_pool: HashMap<usize, crate::forward::LinearAttentionState>,
    /// Adapter generation counter; incremented on LoRA swap.
    adapter_generation: u64,
    /// Whether warmup is complete.
    warmup_done: bool,
    /// Whether we already warned that the paged metadata graph cache is full.
    #[cfg(feature = "cuda")]
    cache_full_warned: bool,
    /// Per-owner request-boundary detection for captured bs=1 decode graphs.
    /// A graph is only replayable for the decode row whose recurrent/conv state
    /// it captured; interleaved serving rows must not share one global timeline.
    #[cfg(feature = "cuda")]
    decode_timelines: HashMap<CudaGraphOwner, CudaGraphOwnerTimeline>,
}

impl CudaGraphRunner {
    /// Create a new graph runner. Enabled only on CUDA devices with the `cuda` feature.
    pub fn new(device: &kiln_tensor::Device, policy: CudaGraphExecutionPolicy) -> Self {
        let is_cuda = matches!(device, kiln_tensor::Device::Cuda(_));
        let actually_enabled = policy.enabled() && is_cuda;
        if actually_enabled {
            tracing::info!(
                max_cached_graphs = policy.max_cached_graphs(),
                "CUDA graphs enabled for decode"
            );
        } else if policy.enabled() && !is_cuda {
            tracing::debug!("CUDA graphs requested but no CUDA device, using eager decode");
        }
        Self {
            policy,
            enabled: actually_enabled,
            #[cfg(feature = "cuda")]
            captured: HashMap::new(),
            #[cfg(feature = "cuda")]
            captured_batched: HashMap::new(),
            #[cfg(feature = "cuda")]
            batched_state_pool: HashMap::new(),
            #[cfg(feature = "cuda")]
            batched_bucket_warmup_done: std::collections::HashSet::new(),
            adapter_generation: 0,
            warmup_done: false,
            #[cfg(feature = "cuda")]
            cache_full_warned: false,
            #[cfg(feature = "cuda")]
            decode_timelines: HashMap::new(),
        }
    }

    /// Invalidate the captured graph (call on LoRA adapter swap).
    pub fn invalidate(&mut self) {
        self.adapter_generation += 1;
        self.warmup_done = false;
        #[cfg(feature = "cuda")]
        {
            if !self.captured.is_empty() || !self.captured_batched.is_empty() {
                tracing::debug!(
                    "CUDA graph invalidated (adapter gen={})",
                    self.adapter_generation
                );
            }
            self.captured.clear();
            self.captured_batched.clear();
            // Persistent batched state survives LoRA swap — the GDN
            // recurrent/conv tensors are weights-independent and the
            // next replay refreshes their contents in-place. Drop only
            // the captured graph that baked stale weight pointers.
            // Per-bucket warmup also resets so the next capture for
            // a bucket primes the allocator under the new weights.
            self.batched_bucket_warmup_done.clear();
            self.cache_full_warned = false;
            self.decode_timelines.clear();
        }
    }

    /// Get or lazily allocate the persistent batched [`LinearAttentionState`]
    /// for `batch_size`. Allocation uses
    /// [`LinearAttentionState::new_with_batch_for_inference_backend`] so
    /// the recurrent/conv dtypes match the inference hot path.
    ///
    /// The returned reference is the canonical slot for that bucket —
    /// callers (the batched capture/replay path) must NOT replace the
    /// inner `recurrent_states[i]` / `conv_states[i]` tensors, only
    /// refresh their contents in-place. Replacement breaks the
    /// stable-device-pointer invariant the captured graph relies on.
    ///
    /// Returns `None` when the runner is disabled or the device is not
    /// CUDA — the batched graph path is CUDA-only.
    #[cfg(feature = "cuda")]
    pub(crate) fn persistent_batched_state(
        &mut self,
        batch_size: usize,
        config: &ModelConfig,
        device: &kiln_tensor::Device,
    ) -> Result<Option<&mut crate::forward::LinearAttentionState>> {
        if !self.enabled {
            return Ok(None);
        }
        if !matches!(device, kiln_tensor::Device::Cuda(_)) {
            return Ok(None);
        }
        anyhow::ensure!(
            batch_size > 0,
            "persistent batched state requires batch_size > 0"
        );
        if let std::collections::hash_map::Entry::Vacant(e) =
            self.batched_state_pool.entry(batch_size)
        {
            // (#1082) kt-native — the device is already kt.
            let state = crate::forward::LinearAttentionState::new_with_batch_for_inference_backend(
                config,
                batch_size,
                device,
                Some("cuda"),
            )
            .with_context(|| {
                format!("allocate persistent batched LinearAttentionState for bucket {batch_size}")
            })?;
            self.batched_state_pool.insert(batch_size, state);
        }
        Ok(self.batched_state_pool.get_mut(&batch_size))
    }

    #[cfg(feature = "cuda")]
    fn prepare_owner_decode(
        &mut self,
        owner: CudaGraphOwner,
        block_table: &BlockTable,
        seq_len: usize,
    ) {
        let block0 = block_table.blocks.first().copied();
        let timeline = self.decode_timelines.entry(owner).or_default();
        let continues = block0.is_some()
            && timeline.last_decode_seq_len == Some(seq_len.wrapping_sub(1))
            && timeline.last_decode_block0 == block0;
        if !continues {
            let before = self.captured.len();
            self.captured.retain(|key, _| key.owner != owner);
            if before != self.captured.len() {
                tracing::debug!(
                    seq_len,
                    ?owner,
                    "CUDA graph: owner boundary - evicting captured bs=1 graph"
                );
            }
        }
        timeline.last_decode_seq_len = Some(seq_len);
        timeline.last_decode_block0 = block0;
    }

    /// Whether graphs are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Number of bs=1 decode graphs currently captured (one per distinct
    /// paged-metadata shape).
    ///
    /// `pub(crate)` test-only accessor (#1082): exists so the bs=1
    /// CUDA-graph-vs-eager decode-parity test
    /// (`forward.rs::test_cuda_graph_bs1_decode_matches_eager`) can assert
    /// that a graph was actually *captured* — i.e. the parity check
    /// exercised the real capture+replay path and did not silently fall
    /// back to eager (which would make the comparison vacuously pass).
    /// Reads the private `captured` map count without exposing the map
    /// itself; touches no graph logic.
    #[cfg(feature = "cuda")]
    pub(crate) fn captured_graph_count(&self) -> usize {
        self.captured.len()
    }

    /// Whether multi-batch CUDA graph capture/replay is available.
    ///
    /// The in-tree implementation remains unqualified after poisoning the CUDA
    /// context during real concurrent serving. It is deliberately unavailable
    /// rather than reachable through a hidden process switch. Re-enabling it
    /// requires a source change plus NVIDIA correctness and resilience evidence.
    #[cfg(feature = "cuda")]
    pub fn is_batched_enabled(&self) -> bool {
        false
    }

    #[cfg(not(feature = "cuda"))]
    pub fn is_batched_enabled(&self) -> bool {
        false
    }

    /// Run a batched paged decode step, attempting CUDA graph capture/replay
    /// for the `(batch_size, max_seqlen_k, …)` bucket. Today this always
    /// returns `Ok(None)`, signalling the caller to take the eager batched
    /// path — the capture/replay glue is still pending (see top-of-file
    /// design note, sequencing step 5 onward).
    ///
    /// Callers (specifically `ModelRunner::paged_batched_decode_step`)
    /// can wire this in eagerly: on `Ok(Some(tokens))`, the captured
    /// graph just ran and produced the next per-row token ids; on
    /// `Ok(None)`, fall back to the existing eager
    /// `decode_next_tokens_paged_contiguous_batch_greedy_with_ids`
    /// path. The runner takes the args by reference because future
    /// implementations will both read the per-row metadata and write
    /// the next-step state back through these slices.
    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_batched(
        &mut self,
        _backend: &dyn BackendRuntime,
        _token_ids: &[u32],
        _weights: &GpuWeights,
        _config: &ModelConfig,
        _paged_cache: &PagedKvCacheKt,
        _block_tables: &[&BlockTable],
        _sequence_lengths: &[usize],
        _linear_states: &mut [&mut LinearAttentionState],
        _lora: Option<&LoraWeights>,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_batched(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        lora: Option<&LoraWeights>,
    ) -> Result<Option<Vec<u32>>> {
        // Two-stage gate: the batched-graph opt-in must be on, and the
        // runner-wide graph enable must also hold. Either being off
        // sends the caller down the eager batched path.
        if !self.is_batched_enabled() {
            return Ok(None);
        }
        let batch_size = token_ids.len();
        if batch_size <= 1 {
            // bs=1 has its own dedicated capture path; don't bucket it here.
            return Ok(None);
        }
        anyhow::ensure!(
            block_tables.len() == batch_size
                && sequence_lengths.len() == batch_size
                && linear_states.len() == batch_size,
            "decode_step_paged_batched: row count mismatch"
        );
        let max_seq_len = *sequence_lengths
            .iter()
            .max()
            .context("decode_step_paged_batched requires a non-empty batch")?;
        let key = CudaBatchedGraphKey::new(batch_size, max_seq_len, paged_cache);

        // Phase 1: per-bucket warmup. Each new `batch_size` bucket
        // runs eager once so cudarc primes any lazy
        // allocator state before we record the graph for that
        // bucket. The global `warmup_done` flag covers the bs=1
        // path; batched needs its own per-bucket tracker because a
        // capture at a cold allocator state for bucket N can
        // produce stale pointers even after bucket M's capture
        // succeeded.
        if !self.batched_bucket_warmup_done.contains(&batch_size) {
            self.batched_bucket_warmup_done.insert(batch_size);
            tracing::debug!(
                batch_size,
                max_seqlen_k = key.max_seqlen_k,
                "batched CUDA graph: per-bucket warmup iteration (eager)"
            );
            return Ok(None);
        }

        // Phase 2: replay path on cache hit + adapter-gen match.
        //
        // (1) Adapter-gen check (cheap, scalar). Drop on mismatch.
        // (2) Refresh GDN persistent state from per-row inputs
        //     in place (no tensor replacement; device pointers stay
        //     stable for the captured kernels).
        // (3) Refresh every stable input buffer in place via the
        //     batched updater family.
        // (4) Launch the captured graph.
        // (5) #1082 boxes 432/433 (STEPS 2-3): sync the capture stream, run
        //     `final_norm` + lm_head EAGERLY on the replayed `output_hidden`
        //     (off the graph — box-102 BUG2 fix), then argmax per row.
        // (6) Scatter the post-step persistent state back into each
        //     per-row `LinearAttentionState` so callers see the
        //     updated GDN history.
        //
        // Borrow plumbing: capture-time grabs `&self.captured_batched`,
        // while state refresh needs `&mut self.batched_state_pool`.
        // We use disjoint field borrows by going through the
        // HashMaps directly instead of via `self.persistent_batched_state(...)`.
        let adapter_gen_now = self.adapter_generation;
        let live_kv_pool_identity = paged_cache.pool_identity();
        if let Some(captured) = self.captured_batched.get(&key) {
            if captured.adapter_gen != adapter_gen_now {
                // Adapter changed since capture; drop the cached graph.
                self.captured_batched.remove(&key);
            } else if captured.kv_pool_identity != live_kv_pool_identity {
                tracing::warn!(
                    expected = ?captured.kv_pool_identity,
                    actual = ?live_kv_pool_identity,
                    "batched CUDA graph replay refused after paged-KV pool replacement"
                );
                self.captured_batched.remove(&key);
            }
        }
        let captured_exists_with_match = self.captured_batched.contains_key(&key);
        if captured_exists_with_match {
            // Step (2): refresh GDN persistent state via direct
            // HashMap access. Either-or with the captured map borrow
            // because both live on `self`; we touch the pool first
            // and let the borrow end before re-grabbing captured.
            let refresh_result = {
                let persistent = self
                    .batched_state_pool
                    .get_mut(&batch_size)
                    .context("missing persistent batched state slot at replay time")?;
                let row_refs: Vec<&crate::forward::LinearAttentionState> =
                    linear_states.iter().map(|s| &**s).collect();
                persistent.refresh_batched_state_from_rows_in_place(&row_refs)
            };
            if let Err(e) = refresh_result {
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = %e,
                    "batched graph replay: GDN state refresh failed, falling back to eager"
                );
                return Ok(None);
            }
            // Re-borrow captured for buffer refresh + launch.
            let captured = self
                .captured_batched
                .get(&key)
                .context("captured graph vanished between adapter-gen check and replay")?;
            // Step (3): refresh stable input buffers.
            if let Err(e) = Self::update_batched_token_buffer(&captured.token_buffer, token_ids)
                .and_then(|()| {
                    Self::update_batched_position_buffer(
                        &captured.position_buffer,
                        sequence_lengths,
                    )
                })
                .and_then(|()| {
                    Self::update_batched_rotary_buffers(
                        &captured.rotary_cos_buffer,
                        &captured.rotary_sin_buffer,
                        config,
                        sequence_lengths,
                    )
                })
                .and_then(|()| {
                    Self::update_batched_paged_metadata_buffers(
                        &captured.block_table_buffer,
                        &captured.seqused_k_buffer,
                        &captured.kv_slot_buffer,
                        block_tables,
                        paged_cache,
                        sequence_lengths,
                        captured.max_seqlen_k,
                    )
                })
            {
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = %e,
                    "batched graph replay: buffer refresh failed, falling back to eager"
                );
                return Ok(None);
            }
            // (#1082 Phase 5) The Step-(3) refreshes above run on the kt DEFAULT
            // stream, but the captured graph launches on its non-default capture
            // stream. Sync the default stream so the refreshed token/position/
            // metadata are visible before replay (else stale reads → garbage).
            if let Some(idx) = captured.token_buffer.device().index() {
                if let Err(e) = kiln_tensor::cuda_synchronize_default_stream_for(
                    idx,
                    kiln_tensor::CudaSyncReason::GraphBoundary,
                ) {
                    tracing::warn!(batch_size, error = %e, "batched: sync before graph launch failed, falling back to eager");
                    return Ok(None);
                }
            }
            // Step (4): launch. The timer spans only the native replay launch
            // and its existing completion boundary; eager LM-head/sampling work
            // below remains outside graph replay attribution.
            let graph_replay_phase = GraphPhaseTimer::start(GraphPhase::Replay);
            if let Err(e) = captured.graph.launch() {
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = %e,
                    "batched CUDA graph replay launch failed, dropping cached graph"
                );
                self.captured_batched.remove(&key);
                return Ok(None);
            }
            // Step (5): #1082 boxes 432/433 (STEPS 2-3) — the captured graph
            // replayed the batched transformer and wrote the PRE-final-norm
            // hidden into the graph-stable `output_hidden` on its capture
            // stream. Sync that stream so the write is visible, then run
            // `final_norm` + lm_head EAGERLY (off the graph) to produce this
            // step's logits, then argmax per row. The captured lm_head
            // cublasLt GEMV was the BUG2 source (wrong logits on replay despite
            // a bit-identical input hidden); the captured transformer win is
            // preserved. Mirrors the bs=1 `decode_step_paged` replay tail.
            let device_index = captured
                .output_hidden
                .device()
                .index()
                .expect("CUDA graph output has a device index");
            if let Err(e) = kiln_tensor::cuda_synchronize_stream_for(
                device_index,
                &captured.capture_stream,
                kiln_tensor::CudaSyncReason::GraphBoundary,
            ) {
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = %e,
                    "batched graph replay: capture-stream sync after launch failed, falling back to eager"
                );
                return Ok(None);
            }
            drop(graph_replay_phase);
            let replay_logits = match crate::forward::lm_head_from_batched_hidden_eager(
                backend,
                &captured.output_hidden,
                weights,
                config,
            ) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(
                        batch_size,
                        max_seqlen_k = key.max_seqlen_k,
                        error = %e,
                        "batched graph replay: eager lm_head on replayed hidden failed, falling back to eager"
                    );
                    return Ok(None);
                }
            };
            let tokens = match crate::sampling::greedy_sample_rows(&replay_logits) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        batch_size,
                        max_seqlen_k = key.max_seqlen_k,
                        error = %e,
                        "batched graph replay: argmax failed, falling back to eager"
                    );
                    return Ok(None);
                }
            };
            // Step (6): scatter persistent → per-row so callers see
            // the post-step GDN state.
            let scatter_result = {
                let persistent = self
                    .batched_state_pool
                    .get_mut(&batch_size)
                    .context("missing persistent batched state slot at scatter time")?;
                persistent.scatter_batch_rows_replace_with_backend(backend, linear_states)
            };
            if let Err(e) = scatter_result {
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = %e,
                    "batched graph replay: state scatter failed but tokens already produced"
                );
                // Tokens already consumed by caller above is fine —
                // surface the scatter error so the next decode step
                // sees an inconsistency rather than silent corruption.
                return Err(e);
            }
            tracing::debug!(
                batch_size,
                max_seqlen_k = key.max_seqlen_k,
                "batched CUDA graph replay produced tokens"
            );
            return Ok(Some(tokens));
        }

        // Phase 3 (capture): no graph cached → record one. The
        // attempt runs the forward pass once and returns its tokens;
        // a captured graph is then stored under `key` for the future
        // replay path to consume.
        //
        // #1082 bs>1 greedy-coherence fix: thread the caller's per-row
        // `linear_states` into the capture path. The captured transformer
        // is recorded on the PERSISTENT pool slot (zeros for a fresh
        // bucket, stale for a reused one), so without seeding the slot from
        // these rows first the captured first token is computed on the
        // wrong GDN recurrent/conv history. `try_capture_batched` now seeds
        // (and scatters back) exactly like the replay path above
        // (refresh_batched_state_from_rows_in_place / scatter back).
        let capture_result = {
            let _phase = GraphPhaseTimer::start(GraphPhase::Capture);
            self.try_capture_batched(
                backend,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                sequence_lengths,
                linear_states,
                lora,
            )
        };
        match capture_result {
            Ok(tokens) => {
                tracing::debug!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    "batched CUDA graph: capture succeeded, returning captured tokens"
                );
                Ok(Some(tokens))
            }
            Err(e) => {
                // `{:#}` prints the full anyhow error chain (e.g.
                // "batched forward failed during graph capture:
                // <inner cause>"). Without the `#` formatter the
                // inner error is silently dropped — that hid the
                // actual root cause from the 2026-05-26 Phase 5
                // batched-decode regression investigation.
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = format!("{e:#}"),
                    "batched CUDA graph capture failed, falling back to eager"
                );
                Ok(None)
            }
        }
    }

    /// Run a paged decode step, using graph capture/replay when possible.
    ///
    /// The lifecycle is:
    /// 1. First call → eager warmup (primes GPU allocator pools).
    /// 2. Second call → attempt CUDA graph capture; fall back to eager on failure.
    /// 3. Subsequent calls → replay captured graph; fall back to eager on failure.
    // #1082: returns a kt `Tensor`. The captured/replayed `output_logits`
    // buffer and the eager fallback are now all kt-native — callers
    // (`generate.rs` decode loops, the bs=1 graph-parity test) consume kt
    // directly, dropping the per-caller candle->kt logits bridge.
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
        graph_row_id: Option<u64>,
    ) -> Result<Tensor> {
        if !self.enabled {
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

        #[cfg(feature = "cuda")]
        {
            let owner = CudaGraphOwner::from_row_id(graph_row_id);
            self.prepare_owner_decode(owner, block_table, seq_len);
        }

        // Phase 1: warmup — run eagerly to prime GPU memory pools
        if !self.warmup_done {
            self.warmup_done = true;
            tracing::debug!("CUDA graph: warmup decode step with graph-shaped inputs");
            #[cfg(feature = "cuda")]
            {
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
                            "CUDA graph-shaped warmup failed: {e:#}, using plain eager decode"
                        );
                    }
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

        #[cfg(feature = "cuda")]
        {
            let owner = CudaGraphOwner::from_row_id(graph_row_id);
            let requested_key = CudaGraphKey::new(paged_cache, seq_len);
            let cache_key = CudaGraphCacheKey::new(owner, requested_key.clone());

            // Phase 3: replay if we have a valid captured graph
            if let Some(captured) = self.captured.get(&cache_key) {
                if captured.adapter_gen == self.adapter_generation {
                    if let Err(error) = paged_cache.ensure_pool_identity(captured.kv_pool_identity)
                    {
                        tracing::warn!(
                            error = %error,
                            "CUDA graph replay refused after paged-KV pool replacement"
                        );
                        self.captured.remove(&cache_key);
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
                    // Update position buffer BEFORE graph replay.
                    // The graph's RoPE kernels read from the same GPU pointer,
                    // so updating the data here gives them the correct position.
                    if let Err(e) = Self::update_token_buffer(&captured.token_buffer, token_id) {
                        tracing::warn!("Failed to update token buffer: {e}, falling back to eager");
                        self.captured.remove(&cache_key);
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
                    if let Err(e) = Self::update_position_buffer(&captured.position_buffer, seq_len)
                    {
                        tracing::warn!(
                            "Failed to update position buffer: {e}, falling back to eager"
                        );
                        self.captured.remove(&cache_key);
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
                    if let Err(e) = Self::update_rotary_buffers(
                        &captured.rotary_cos_buffer,
                        &captured.rotary_sin_buffer,
                        config,
                        seq_len,
                    ) {
                        tracing::warn!(
                            "Failed to update rotary graph buffers: {e}, falling back to eager"
                        );
                        self.captured.remove(&cache_key);
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
                    if let Err(e) = Self::update_paged_metadata_buffers(
                        &captured.block_table_buffer,
                        &captured.seqused_k_buffer,
                        &captured.kv_slot_buffer,
                        block_table,
                        paged_cache,
                        seq_len,
                        captured.max_seqlen_k,
                    ) {
                        tracing::warn!(
                            "Failed to update paged graph metadata buffers: {e}, falling back to eager"
                        );
                        self.captured.remove(&cache_key);
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

                    // (#1082 Phase 5) The per-replay input writes above
                    // (update_token/position/rotary/paged_metadata via
                    // cuda_write_host_in_place) land on the kt DEFAULT stream,
                    // but the captured graph launches on its non-default capture
                    // stream — without ordering, replay reads a stale token_id
                    // and the decode diverges into garbage. Sync the default
                    // stream so those writes are visible before launch.
                    if let Some(idx) = captured.token_buffer.device().index() {
                        kiln_tensor::cuda_synchronize_default_stream_for(
                            idx,
                            kiln_tensor::CudaSyncReason::GraphBoundary,
                        )
                        .context("sync per-replay input writes before CUDA graph launch")?;
                    }

                    let mut plan = CudaDecodeReplayPlan::new(captured);
                    let replay_key = kiln_graph::ReplayPlan::key(&plan);
                    let replay_inputs =
                        ReplayInputs::new(&replay_key, &captured.replay_state.inputs);
                    let replay_result = {
                        let _phase = GraphPhaseTimer::start(GraphPhase::Replay);
                        kiln_graph::ReplayPlan::replay(&mut plan, replay_inputs)
                    };
                    match replay_result {
                        Ok(_) => {
                            tracing::debug!(
                                max_seqlen_k = requested_key.max_seqlen_k,
                                max_blocks_per_seq = requested_key.max_blocks_per_seq,
                                "CUDA graph ReplayPlan replay succeeded"
                            );
                            // #1082 box-102 FIX: the ReplayPlan has launched the
                            // captured transformer and synchronized the capture
                            // stream, making graph-stable `output_hidden` visible.
                            // Now run final_norm + lm_head EAGERLY (off the graph)
                            // to produce this step's logits. The captured lm_head
                            // cublasLt GEMV was the BUG2 source (wrong logits on
                            // replay despite a bit-identical input hidden); the
                            // captured transformer win is preserved.
                            let replay_logits = crate::forward::lm_head_from_hidden_eager(
                                backend,
                                &captured.output_hidden,
                                weights,
                                config,
                            )
                            .context("box-102 fix: eager lm_head on replayed hidden")?;
                            return Ok(replay_logits);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "CUDA graph ReplayPlan replay failed: {e}, falling back to eager"
                            );
                            self.captured.remove(&cache_key);
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
                    // Adapter changed — drop stale graph
                    self.captured.clear();
                }
            } else if !self.captured.is_empty() {
                tracing::debug!(
                    requested_max_seqlen_k = requested_key.max_seqlen_k,
                    requested_max_blocks_per_seq = requested_key.max_blocks_per_seq,
                    cached_graphs = self.captured.len(),
                    "CUDA graph replay miss: paged decode metadata shape differs from captured graphs"
                );
            }

            if self.captured.len() >= self.policy.max_cached_graphs() {
                if self.cache_full_warned {
                    tracing::debug!(
                        cached_graphs = self.captured.len(),
                        requested_max_seqlen_k = requested_key.max_seqlen_k,
                        requested_max_blocks_per_seq = requested_key.max_blocks_per_seq,
                        "CUDA graph capture skipped: paged metadata shape cache is full"
                    );
                } else {
                    self.cache_full_warned = true;
                    tracing::warn!(
                        cached_graphs = self.captured.len(),
                        requested_max_seqlen_k = requested_key.max_seqlen_k,
                        requested_max_blocks_per_seq = requested_key.max_blocks_per_seq,
                        "CUDA graph capture skipped: paged metadata shape cache is full"
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

            // Phase 2: capture
            let capture_result = {
                let _phase = GraphPhaseTimer::start(GraphPhase::Capture);
                self.try_capture(
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
                )
            };
            match capture_result {
                Ok(logits) => Ok(logits),
                Err(e) => {
                    tracing::warn!("CUDA graph capture failed: {e:#}, using eager decode");
                    self.enabled = false;
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
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = graph_row_id;
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

    /// Update the position buffer tensor with a new position value.
    ///
    /// Copies a single f32 value into the existing GPU allocation without
    /// changing the device pointer. This is done outside the CUDA graph so
    /// replayed RoPE kernels read the correct position.
    #[cfg(feature = "cuda")]
    fn update_token_buffer(token_buffer: &Tensor, token_id: u32) -> Result<()> {
        kiln_tensor::cuda_write_host_in_place(token_buffer, &[token_id])
            .context("update CUDA graph token buffer")
    }

    #[cfg(feature = "cuda")]
    fn update_position_buffer(position_buffer: &Tensor, position: usize) -> Result<()> {
        kiln_tensor::cuda_write_host_in_place(position_buffer, &[position as f32])
            .context("update CUDA graph position buffer")
    }

    #[cfg(feature = "cuda")]
    fn update_rotary_buffers(
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        config: &ModelConfig,
        position: usize,
    ) -> Result<()> {
        // #34 BUG2 FIX: compute the rotary tables on the GPU via eager's exact
        // path (`forward::rotary_tables_from_tensor` -> `kt_cos`/`kt_sin`) rather
        // than host CPU `f32` cos/sin. Host CPU `cos` and GPU `cos` disagree
        // (range reduction) by ~0.1% at large position*freq; that perturbed ONLY
        // the RoPE full-attention layers on every replay (GDN has no RoPE and was
        // bit-identical), propagating to ~1% logit drift that flipped close-call
        // tokens. Computing on-device makes graph replay bit-identical to eager.
        let dev = rotary_cos_buffer.device();
        let inv_freq =
            crate::forward::compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, &dev)?;
        let pos = Tensor::from_vec_on(dev, vec![position as f32], vec![1])?;
        let (cos, sin) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        let cos = cos
            .to_dtype(rotary_cos_buffer.dtype())?
            .reshape(rotary_cos_buffer.dims().to_vec())?;
        let sin = sin
            .to_dtype(rotary_sin_buffer.dtype())?
            .reshape(rotary_sin_buffer.dims().to_vec())?;
        rotary_cos_buffer
            .slice_set(&cos, 0, 0)
            .context("update CUDA graph rotary cos buffer (gpu)")?;
        rotary_sin_buffer
            .slice_set(&sin, 0, 0)
            .context("update CUDA graph rotary sin buffer (gpu)")?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
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
        kiln_tensor::cuda_write_host_in_place(block_table_buffer, padded.as_slice())
            .context("update CUDA graph block table buffer")?;
        // #1082: seqused_k is U32 in the kt path (was i32 in candle; same bytes).
        let attention_len = [(seq_len + 1) as u32];
        kiln_tensor::cuda_write_host_in_place(seqused_k_buffer, &attention_len)
            .context("update CUDA graph seqused_k buffer")?;
        let slot = [block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32];
        kiln_tensor::cuda_write_host_in_place(kv_slot_buffer, &slot)
            .context("update CUDA graph KV slot buffer")?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn replay_state_for_capture(
        key: &CudaGraphKey,
        output_hidden: &Tensor,
        token_buffer: &Tensor,
        position_buffer: &Tensor,
        block_table_buffer: &Tensor,
        seqused_k_buffer: &Tensor,
        kv_slot_buffer: &Tensor,
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        paged_decode_outputs: &[Tensor],
        paged_decode_lse: &[Tensor],
        gdn_decode_outputs: &[Tensor],
    ) -> ReplayState {
        let replay_key = ReplayKey::new(
            Backend::Cuda,
            "paged_decode_graph_outputs",
            vec![key.max_seqlen_k, key.max_blocks_per_seq],
            Some(output_hidden.dtype()),
            1,
            true,
        );
        let mut resources = vec![
            Self::stable_replay_resource(output_hidden),
            Self::stable_replay_resource(token_buffer),
            Self::stable_replay_resource(position_buffer),
            Self::stable_replay_resource(block_table_buffer),
            Self::stable_replay_resource(seqused_k_buffer),
            Self::stable_replay_resource(kv_slot_buffer),
            Self::stable_replay_resource(rotary_cos_buffer),
            Self::stable_replay_resource(rotary_sin_buffer),
        ];
        resources.extend(
            paged_decode_outputs
                .iter()
                .map(Self::stable_replay_resource),
        );
        resources.extend(paged_decode_lse.iter().map(Self::stable_replay_resource));
        resources.extend(gdn_decode_outputs.iter().map(Self::stable_replay_resource));
        ReplayState::new(replay_key, resources)
    }

    #[cfg(feature = "cuda")]
    fn stable_replay_resource(tensor: &Tensor) -> ResidentResourceRef {
        ResidentResourceRef::from_tensor(
            tensor,
            Backend::Cuda,
            ReplayResourceStability::StableAcrossReplay,
        )
    }

    /// Rewrite the contents of the batched token buffer in place so the
    /// captured graph picks up the new per-row input tokens on the next
    /// replay.
    #[cfg(feature = "cuda")]
    fn update_batched_token_buffer(token_buffer: &Tensor, token_ids: &[u32]) -> Result<()> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "update_batched_token_buffer requires a non-empty batch"
        );
        kiln_tensor::cuda_write_host_in_place(token_buffer, token_ids)
            .context("update CUDA graph batched token buffer")
    }

    /// Rewrite the contents of the batched position buffer in place
    /// from per-row `start_positions`.
    #[cfg(feature = "cuda")]
    fn update_batched_position_buffer(
        position_buffer: &Tensor,
        start_positions: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            !start_positions.is_empty(),
            "update_batched_position_buffer requires a non-empty batch"
        );
        let pos_f32: Vec<f32> = start_positions.iter().map(|&p| p as f32).collect();
        kiln_tensor::cuda_write_host_in_place(position_buffer, pos_f32.as_slice())
            .context("update CUDA graph batched position buffer")
    }

    /// Rewrite the three paged-metadata buffers for the batched graph
    /// in place. Same contract as the bs=1
    /// `update_paged_metadata_buffers` but every buffer is `[batch, …]`.
    #[cfg(feature = "cuda")]
    fn update_batched_paged_metadata_buffers(
        block_table_buffer: &Tensor,
        seqused_k_buffer: &Tensor,
        kv_slot_buffer: &Tensor,
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        start_positions: &[usize],
        max_seqlen_k: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            !block_tables.is_empty(),
            "update_batched_paged_metadata_buffers requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == start_positions.len(),
            "update_batched_paged_metadata_buffers: row count mismatch ({} vs {})",
            block_tables.len(),
            start_positions.len()
        );
        // Block table: stack each row's padded view, same width as
        // capture time.
        let mut block_flat: Vec<u32> = Vec::new();
        for bt in block_tables {
            let padded = Self::padded_block_table(bt, paged_cache, max_seqlen_k)?;
            block_flat.extend_from_slice(&padded);
        }
        kiln_tensor::cuda_write_host_in_place(block_table_buffer, block_flat.as_slice())
            .context("update CUDA graph batched block table buffer")?;
        // seqused_k: per-row (start_pos + 1). #1082: U32 in the kt path.
        let seqused: Vec<u32> = start_positions
            .iter()
            .map(|&p| {
                u32::try_from(p + 1).context("batched seqused_k buffer: value exceeds u32 range")
            })
            .collect::<Result<Vec<_>>>()?;
        kiln_tensor::cuda_write_host_in_place(seqused_k_buffer, seqused.as_slice())
            .context("update CUDA graph batched seqused_k buffer")?;
        // KV slots: per-row current write slot.
        let mut slots: Vec<u32> = Vec::with_capacity(block_tables.len());
        for (bt, &pos) in block_tables.iter().zip(start_positions.iter()) {
            slots.push(
                bt.slot_for(pos, paged_cache.block_size())
                    .with_context(|| format!("no slot for decode position {pos}"))?
                    as u32,
            );
        }
        kiln_tensor::cuda_write_host_in_place(kv_slot_buffer, slots.as_slice())
            .context("update CUDA graph batched KV slot buffer")?;
        Ok(())
    }

    /// Rewrite the batched rotary cos/sin tables for the current batch
    /// of per-row decode positions. Each row computes its own table,
    /// then the rows are stacked into the `[batch, half]` buffer.
    #[cfg(feature = "cuda")]
    fn update_batched_rotary_buffers(
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        config: &ModelConfig,
        start_positions: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            !start_positions.is_empty(),
            "update_batched_rotary_buffers requires a non-empty batch"
        );
        // #34 BUG2 FIX: compute the batched rotary tables on the GPU via eager's
        // exact path (one position per batch row), not host CPU cos/sin. Same root
        // cause + fix as the bs=1 path (`update_rotary_buffers`): CPU cos != GPU
        // cos perturbs only the RoPE full-attention layers on replay.
        let dev = rotary_cos_buffer.device();
        let inv_freq =
            crate::forward::compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, &dev)?;
        let pos_f32: Vec<f32> = start_positions.iter().map(|&p| p as f32).collect();
        let n = pos_f32.len();
        let pos = Tensor::from_vec_on(dev, pos_f32, vec![n])?;
        let (cos, sin) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        let cos = cos
            .to_dtype(rotary_cos_buffer.dtype())?
            .reshape(rotary_cos_buffer.dims().to_vec())?;
        let sin = sin
            .to_dtype(rotary_sin_buffer.dtype())?
            .reshape(rotary_sin_buffer.dims().to_vec())?;
        rotary_cos_buffer
            .slice_set(&cos, 0, 0)
            .context("update CUDA graph batched rotary cos buffer (gpu)")?;
        rotary_sin_buffer
            .slice_set(&sin, 0, 0)
            .context("update CUDA graph batched rotary sin buffer (gpu)")?;
        Ok(())
    }

    // #1082: the candle `update_cuda_scalar` helper (raw
    // `memcpy_htod_async` into a candle CUDA storage) is gone — every
    // refresh now goes through the kt-native
    // `kiln_tensor::cuda_write_host_in_place`, which writes through the
    // kt buffer's stable device pointer on the kt active stream
    // (capture/replay-stream aware).

    /// Attempt to capture a CUDA graph during a decode forward pass.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_capture(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: CudaGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;

        // #1082: the graph-stable buffers are kt-native and allocated
        // directly on the kt device (`weights.embed_tokens.device()`),
        // so NO candle alloc + per-buffer candle->kt bridge any more.
        // The capture-control FFI (`begin_capture` / `end_capture` /
        // `capture_status` / `synchronize`) now targets a kt context
        // stream (used to live on a candle `CudaStream` handle).
        let device = weights.embed_tokens.device();
        let dtype = weights.embed_tokens.dtype();
        let device_idx = match device {
            kiln_tensor::Device::Cuda(i) => i,
            _ => anyhow::bail!("CUDA graphs require a CUDA device"),
        };
        // (#1082 Phase 5) Capture on a FRESH non-default CUstream, NOT
        // `default_stream()`: the kt default stream is the legacy NULL stream
        // (0x0), and `cuStreamBeginCapture` on the NULL stream returns
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED (silent eager fallback — the
        // capture stream printed `0x0` in the trace). The
        // `with_active_cuda_stream` scope below routes every kt op in the
        // captured forward onto this stream so its launches are recorded.
        let stream = kiln_tensor::primary_cuda_context(device_idx)
            .context("CUDA graph capture: kt primary_cuda_context for capture stream")?
            .new_stream()
            .context("CUDA graph capture: create non-default capture stream")?;

        // Pre-allocate graph-stable decode tensors BEFORE capture (kt
        // buffers own a persistent device pointer that gets baked into
        // the captured graph). Alloc happens on the capture stream via
        // the `with_active_cuda_stream` scope below, so the buffers'
        // backing allocations are recorded coherently. Allocating before
        // `begin_capture` is also fine — only the contents are refreshed
        // inside the capture/replay window.
        let token_buffer = Self::new_token_buffer(device, token_id)?;
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        let output_hidden = Self::new_output_hidden(config, device, dtype)?;
        let rotary_cos_buffer = Self::new_rotary_cos_buffer(config, device, seq_len)?;
        let rotary_sin_buffer = Self::new_rotary_sin_buffer(config, device, seq_len)?;
        let key = CudaGraphKey::new(paged_cache, seq_len);
        // Stable paged metadata is a correctness invariant: captured kernels
        // must read current block-table, sequence-length, and KV-slot contents
        // from retained device pointers on every replay.
        let block_table_buffer =
            Self::new_block_table_buffer(block_table, paged_cache, key.max_seqlen_k, device)?;
        let seqused_k_buffer = Self::new_seqused_k_buffer(device, seq_len + 1)?;
        let kv_slot_buffer = Self::new_kv_slot_buffer(block_table, paged_cache, seq_len, device)?;
        let (paged_decode_outputs, paged_decode_lse) =
            Self::new_paged_decode_outputs(config, device, dtype)?;
        // #1082: the kt graph-stable buffers feed the kt-typed
        // `PagedDecodeGraphInputs` directly — no bridges. Build the
        // struct from references into the owned buffers above.
        let graph_inputs = PagedDecodeGraphInputs {
            block_table: &block_table_buffer,
            seqused_k: &seqused_k_buffer,
            kv_slot: &kv_slot_buffer,
            max_seqlen_k: key.max_seqlen_k,
            rotary_cos: &rotary_cos_buffer,
            rotary_sin: &rotary_sin_buffer,
            attn_out: &paged_decode_outputs[..],
            softmax_lse: &paged_decode_lse[..],
        };
        let gdn_decode_outputs = Self::new_gdn_decode_outputs(config, device)?;
        // #1082 box-102 FIX: no lm-head output buffer for the bs=1 path — the
        // captured forward stops at the pre-final-norm hidden (`HiddenOnly`),
        // and final_norm + lm_head run EAGERLY on the replayed hidden, off the
        // captured graph (the lm_head cublasLt GEMV is not replay-safe).
        Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;

        // === #1082 freeze-pointers Pass 1 (Record / warm) ===
        // The captured forward allocates Q/K/V projections + per-layer
        // activations via `zeros_ctx` / `alloc_uninit_ctx`; on `main` those go
        // straight to `cudaMallocAsync` and are FREED when this fn returns, so
        // the captured `flash_fwd_splitkv_kernel` dereferences a dangling
        // pointer on replay (compute-sanitizer-confirmed ILLEGAL_ADDRESS — a
        // garbage `Flash_fwd_params` device pointer). Fix: route every such
        // alloc through a thread-local capture arena. Pass 1 runs the SAME
        // forward once BEFORE `begin_capture` to allocate + retain those buffers;
        // Pass 2 (the captured run below) hands out `Borrowed` views of them so
        // every recorded device pointer stays mapped for the graph's lifetime.
        //
        // Running the forward twice double-mutates decode state: the paged-KV
        // write is idempotent (same token -> same slot -> same K/V), but the GDN
        // recurrent `linear_state` is not, so snapshot it before Pass 1 and
        // restore before the captured Pass 2.
        let arena_device_index = device.index().ok_or_else(|| {
            anyhow::anyhow!("freeze-pointers: CUDA-graph capture requires a CUDA device index")
        })?;
        let arena_ctx = kiln_tensor::primary_cuda_context(arena_device_index)
            .context("freeze-pointers: primary_cuda_context for capture arena")?;
        let arena = std::rc::Rc::new(std::cell::RefCell::new(
            kiln_tensor::CaptureArena::new_record(arena_ctx, arena_device_index),
        ));
        let gdn_snapshot = linear_state
            .snapshot()
            .context("freeze-pointers: snapshot GDN recurrent state before warm pass")?;
        let warm_result = kiln_tensor::with_capture_arena(arena.clone(), || {
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
                Some(&graph_inputs),
            )?;
            kiln_tensor::cuda_slice_set_dim0(&output_hidden, &hidden, 0)
                .context("freeze-pointers warm pass: copy hidden into stable output")?;
            Ok::<(), anyhow::Error>(())
        });
        warm_result.context("freeze-pointers warm (Record) pass failed")?;
        // Restore the GDN recurrent state so the captured pass advances it
        // exactly once (KV writes are idempotent and need no restore).
        *linear_state = gdn_snapshot;
        // Flip to Replay: Pass 2 hands out Borrowed views of the recorded
        // buffers instead of allocating fresh ones.
        arena.borrow_mut().begin_replay();

        // #1082: the kt buffer allocs above filled their contents via an
        // H2D on the kt DEFAULT stream (not the capture stream below), so
        // sync the default stream before capture to guarantee those fills
        // are visible to the captured forward.
        if let Some(idx) = device.index() {
            kiln_tensor::cuda_synchronize_default_stream_for(
                idx,
                kiln_tensor::CudaSyncReason::GraphBoundary,
            )
            .context("CUDA graph capture: sync kt default stream before capture")?;
        }

        // Synchronize all pending work before capture
        let device_index = device
            .index()
            .context("CUDA graph capture device has no index")?;
        kiln_tensor::cuda_synchronize_stream_for(
            device_index,
            &stream,
            kiln_tensor::CudaSyncReason::GraphBoundary,
        )
        .map_err(|e| anyhow::anyhow!("sync before graph capture: {e}"))?;

        let capture_status = stream
            .capture_status()
            .map_err(|e| anyhow::anyhow!("capture_status before begin_capture: {e}"))?;
        tracing::debug!(?capture_status, stream = ?stream.cu_stream(), "CUDA graph stream status before begin_capture");

        // Begin stream capture — all subsequent GPU operations are recorded
        stream
            .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;

        // Run the forward pass with the pre-allocated position buffer.
        // All kernels are captured, including RoPE which reads from
        // position_buffer's stable GPU address.
        // Phase 7 closeout (#1082): the captured kt-path GDN decode
        // kernels allocate their own outputs (per gdn_decode_*_kt
        // entries in kiln-gdn-kernel/src/kt_api.rs) and never read
        // the legacy `with_decode_gates_recurrent_outputs`
        // thread-local. `gdn_decode_outputs` stays as a struct field
        // (`_gdn_decode_outputs`) so the pre-allocated graph-stable
        // buffers remain alive for the lifetime of the captured
        // graph — same buffer-ownership rationale as
        // `_paged_decode_outputs`. The wrapper is gone; the inner
        // closure is invoked directly.
        let _ = &gdn_decode_outputs;
        // #1082 CUDA-graph fix (Part C): engage the thread-local active-stream
        // override for the whole capture window so every kt CUDA op
        // (kernel launch / alloc / memcpy, resolved via
        // `active_cuda_stream`) lands on THIS capture stream instead of the
        // legacy NULL default stream. Issuing work on the NULL stream while
        // `stream` is mid-capture is the `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`
        // root cause. Outside this scope `active_cuda_stream` returns
        // `ctx.default_stream()` — zero behavior change for all other paths.
        let capture_result = kiln_tensor::with_capture_arena(arena.clone(), || {
            kiln_tensor::with_active_cuda_stream(stream.clone(), || {
                // #1082 box-102 FIX: capture the PRE-final-norm HIDDEN
                // (`HiddenOnly`), NOT the logits — the lm_head cublasLt GEMV is
                // excluded from the graph and run eagerly on replay.
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
                    Some(&graph_inputs),
                )?;
                // `hidden` is kt; copy it into the graph-stable kt
                // `output_hidden` via a KT-NATIVE copy that runs on the capture
                // stream (the override above makes `cuda_slice_set_dim0` resolve
                // to `stream`). The copy is recorded into the graph, so every
                // replay refreshes `output_hidden` in place at the same device
                // pointer.
                kiln_tensor::cuda_slice_set_dim0(&output_hidden, &hidden, 0)
                    .context("CUDA graph: copy kt hidden into stable output_hidden")?;
                Ok::<(), anyhow::Error>(())
            })
        });
        let capture_arena_result = arena
            .borrow()
            .ensure_replay_complete()
            .context("CUDA graph capture arena allocation sequence mismatch");

        // End capture — instantiates the graph with AUTO_FREE_ON_LAUNCH.
        //
        // NOTE (BUG2): an earlier change (reverted here) flipped this to flags=0
        // on the theory that AUTO_FREE re-allocated the GDN state buffers each
        // replay. That theory is WRONG: `bench-results/cuda-graph-box102-findings.md`
        // (runtime-instrumented on A6000) explicitly tested flags=0 here and saw
        // IDENTICAL token-doubling (finding #3) — flags=0 is orthogonal to BUG2.
        // The real divergence is a layer-0/input replay staleness still under
        // investigation. AUTO_FREE_ON_LAUNCH is restored because it correctly
        // reclaims the graph's per-launch async scratch (ROCm works for a
        // different reason — its forward emits no in-graph device-malloc nodes,
        // so AUTO_FREE has nothing to free there).
        let graph_result = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );

        // Check forward pass success first
        capture_result.context("forward pass failed during graph capture")?;
        capture_arena_result?;

        // Check graph capture success
        match graph_result {
            Ok(Some(graph)) => {
                tracing::info!(
                    "CUDA graph captured for decode ({} layers)",
                    config.num_layers,
                );
                // (#1082 Phase 5) Stream capture only RECORDS the forward — its
                // kernels did NOT execute, so `output_hidden` is still the
                // uninitialized capture-time buffer and the in-place recurrent/
                // KV state was not advanced. Launch the instantiated graph once
                // now (the token/position/rotary/metadata buffers already hold
                // THIS step's inputs) to actually compute this step + advance
                // state, then sync so `output_hidden` is valid before we read it.
                graph
                    .launch()
                    .context("execute captured decode graph (first run)")?;
                kiln_tensor::cuda_synchronize_stream_for(
                    device_index,
                    &stream,
                    kiln_tensor::CudaSyncReason::GraphBoundary,
                )
                .map_err(|e| anyhow::anyhow!("sync after first captured-graph launch: {e}"))?;
                // #1082 box-102 FIX: run final_norm + lm_head EAGERLY on the
                // capture-step hidden to produce this step's logits — the lm_head
                // cublasLt GEMV is OUT of the captured graph. `output_hidden` now
                // holds the first-launch hidden (synced above).
                let logits = crate::forward::lm_head_from_hidden_eager(
                    backend,
                    &output_hidden,
                    weights,
                    config,
                )
                .context("box-102 fix: eager lm_head on captured hidden (first launch)")?;
                let max_seqlen_k = key.max_seqlen_k;
                let replay_state = Self::replay_state_for_capture(
                    &key,
                    &output_hidden,
                    &token_buffer,
                    &position_buffer,
                    &block_table_buffer,
                    &seqused_k_buffer,
                    &kv_slot_buffer,
                    &rotary_cos_buffer,
                    &rotary_sin_buffer,
                    &paged_decode_outputs,
                    &paged_decode_lse,
                    &gdn_decode_outputs,
                );
                self.captured.insert(
                    CudaGraphCacheKey::new(owner, key),
                    CapturedDecodeGraph {
                        graph,
                        output_hidden,
                        capture_stream: stream.clone(),
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
                        _capture_arena_buffers: arena.borrow_mut().take_retained(),
                        replay_state,
                    },
                );
                Ok(logits)
            }
            Ok(None) => {
                anyhow::bail!("graph capture produced no operations");
            }
            Err(e) => {
                anyhow::bail!("end_capture failed: {e}");
            }
        }
    }

    /// Capture a batched (`bs > 1`) decode graph for the
    /// `(batch_size, max_seqlen_k, …)` bucket and run it once, returning
    /// the per-row next-token IDs. The captured graph is stored in
    /// `self.captured_batched`; subsequent calls with a matching key
    /// can replay it.
    ///
    /// #1082 boxes 432/433 (STEPS 2-3): rewritten to mirror the bs=1
    /// [`Self::try_capture`] discipline exactly — the box-102 BUG2 fix port:
    ///
    /// 1. Allocate every `new_batched_*` device buffer + the step-1
    ///    `output_hidden` BEFORE capture.
    /// 2. Build [`crate::forward::BatchedPagedDecodeGraphInputs`] over those
    ///    buffers + the persistent batched-state slot.
    /// 3. Freeze-pointers Pass-1 WARM RECORD: run the HiddenOnly batched
    ///    forward ONCE through a recording [`kiln_tensor::CaptureArena`] to
    ///    allocate+retain every transient Q/K/V/activation buffer. The GDN
    ///    recurrent/conv state in the persistent slot is non-idempotent, so
    ///    snapshot it before the warm pass and restore it IN PLACE (preserving
    ///    the slot's stable device pointers) after. The paged-KV write is
    ///    idempotent (same token→slot→same K/V) so it needs no restore.
    /// 4. `arena.begin_replay()`.
    /// 5. `begin_capture(...RELAXED)` on the FRESH non-default capture stream.
    /// 6. Inside `with_capture_arena` + `with_active_cuda_stream`, run the
    ///    HiddenOnly batched forward (`Borrowed` arena views) then
    ///    `cuda_slice_set_dim0` the hidden into `output_hidden`.
    /// 7. `end_capture(AUTO_FREE_ON_LAUNCH)` (the bs=1 flag — the in-graph
    ///    lm_head that forced the bs>1 `NO_FLAGS` workaround is gone, so the
    ///    pinned arena buffers + AUTO_FREE behave exactly as bs=1).
    /// 8. First `graph.launch()` + sync, then eager
    ///    [`crate::forward::lm_head_from_batched_hidden_eager`] on the
    ///    first-launch hidden to produce this step's logits.
    /// 9. Store `output_hidden` + `capture_stream` +
    ///    `_capture_arena_buffers` on [`CapturedBatchedDecodeGraph`].
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_capture_batched(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        // #1082 bs>1 greedy-coherence fix: caller's per-row post-prefill GDN
        // states. The capture seeds the persistent pool slot from these
        // (refresh_batched_state_from_rows_in_place) BEFORE the warm/capture
        // passes, then scatters the post-step slot back into them — exactly
        // mirroring the replay path so the captured first token runs on the
        // correct recurrent/conv history (not the slot's zeros/stale state).
        linear_states: &mut [&mut LinearAttentionState],
        lora: Option<&LoraWeights>,
    ) -> Result<Vec<u32>> {
        use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;

        let batch_size = token_ids.len();
        anyhow::ensure!(
            batch_size > 0,
            "try_capture_batched requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == batch_size
                && sequence_lengths.len() == batch_size
                && linear_states.len() == batch_size,
            "try_capture_batched: row count mismatch"
        );
        let max_seq_len = *sequence_lengths.iter().max().expect("non-empty batch");
        let key = CudaBatchedGraphKey::new(batch_size, max_seq_len, paged_cache);

        // #1082: the batched graph-stable buffers are kt-native and
        // allocated directly on the kt device — no candle alloc, no
        // per-buffer candle->kt bridge. The capture runs on a FRESH non-default
        // stream (see below) so `with_active_cuda_stream` can route the captured
        // kt forward onto it; capture-control FFI (begin/end_capture,
        // synchronize) drives that stream directly.
        let device = weights.embed_tokens.device();
        let dtype = weights.embed_tokens.dtype();
        let device_idx = match device {
            kiln_tensor::Device::Cuda(i) => i,
            _ => anyhow::bail!("CUDA graphs require a CUDA device"),
        };
        // (#1082 Phase 5) Fresh non-default capture stream — see the bs=1
        // try_capture note. `default_stream()` is the legacy NULL stream (0x0)
        // which cannot be captured; `with_active_cuda_stream` routes the
        // captured batched forward onto this stream.
        let stream = kiln_tensor::primary_cuda_context(device_idx)
            .context("batched CUDA graph capture: kt primary_cuda_context for capture stream")?
            .new_stream()
            .context("batched CUDA graph capture: create non-default capture stream")?;
        let adapter_gen = self.adapter_generation;

        // Pre-allocate every device buffer the captured graph will read
        // from or write to. Each pointer is baked into the recorded
        // kernel launches; the runner refreshes their contents in
        // place before each replay.
        let token_buffer = Self::new_batched_token_buffer(device, token_ids)?;
        let position_buffer = Self::new_batched_position_buffer(device, sequence_lengths)?;
        let rotary_cos_buffer = Self::new_batched_rotary_cos_buffer(config, device, batch_size)?;
        let rotary_sin_buffer = Self::new_batched_rotary_sin_buffer(config, device, batch_size)?;
        Self::update_batched_rotary_buffers(
            &rotary_cos_buffer,
            &rotary_sin_buffer,
            config,
            sequence_lengths,
        )?;
        let block_table_buffer = Self::new_batched_block_table_buffer(
            block_tables,
            paged_cache,
            key.max_seqlen_k,
            device,
        )?;
        let seqused_k_buffer = Self::new_batched_seqused_k_buffer(device, sequence_lengths)?;
        let kv_slot_buffer =
            Self::new_batched_kv_slot_buffer(block_tables, paged_cache, sequence_lengths, device)?;
        let (paged_decode_outputs, paged_decode_lse) =
            Self::new_batched_paged_decode_outputs(config, device, dtype, batch_size)?;
        let gdn_decode_outputs = Self::new_batched_gdn_decode_outputs(config, device, batch_size)?;
        // #1082 boxes 432/433 (STEPS 2-3) — graph-stable PRE-final-norm hidden
        // buffer (`[batch, 1, hidden]`). The captured HiddenOnly batched forward
        // (`model_forward_paged_batched_hidden_with_graph_inputs`) writes the
        // transformer-stack output here via `cuda_slice_set_dim0`; `final_norm`
        // + lm_head run EAGERLY on it after launch via
        // `lm_head_from_batched_hidden_eager` (the large-N lm_head cublasLt GEMV
        // is not graph-replay-safe — the box-102 BUG2 doubling). The former
        // in-graph `output_logits` + `lm_head_output_buffer` allocations are
        // gone; with the lm_head out of the captured region the bs>1 path no
        // longer needs the `NO_FLAGS` AUTO_FREE workaround.
        // `mut` because `graph_inputs.output_hidden` takes `&mut output_hidden`.
        let mut output_hidden = Self::new_batched_output_hidden(config, device, dtype, batch_size)?;

        // #1082 freeze-pointers (batched port of the bs=1 capture arena).
        // Allocate the recording arena + take a GDN snapshot of the persistent
        // batched slot BEFORE we move the `&mut` slot borrow into
        // `graph_inputs`. The warm pass advances the GDN recurrent/conv state
        // by exactly one step (non-idempotent), so we restore the slot's
        // CONTENTS in place afterward (preserving its stable device pointers
        // for the captured Pass-2). The paged-KV write is idempotent (same
        // token→slot→same K/V), so it needs no restore — mirrors bs=1.
        let arena_device_index = device.index().ok_or_else(|| {
            anyhow::anyhow!(
                "freeze-pointers (batched): CUDA-graph capture requires a CUDA device index"
            )
        })?;
        let arena_ctx = kiln_tensor::primary_cuda_context(arena_device_index)
            .context("freeze-pointers (batched): primary_cuda_context for capture arena")?;
        let arena = std::rc::Rc::new(std::cell::RefCell::new(
            kiln_tensor::CaptureArena::new_record(arena_ctx, arena_device_index),
        ));

        // Capture + forward inside a scope so the `&mut` borrow on
        // `self.batched_state_pool` (taken by `persistent_batched_state`)
        // ends before we mutate `self.captured_batched` below. The kt
        // buffers feed `BatchedPagedDecodeGraphInputs` directly (the
        // struct is already kt-typed) — no bridges.
        let captured: CapturedBatchedDecodeGraph = {
            let persistent_state = self
                .persistent_batched_state(batch_size, config, &device)?
                .context("persistent batched state required for capture")?;
            // #1082 bs>1 greedy-coherence fix: SEED the persistent pool slot
            // from the caller's per-row post-prefill states IN PLACE before
            // anything reads it. The slot is zeros for a fresh bucket or stale
            // for a reused one; the captured transformer below records on this
            // slot, so without this seed the captured first token is computed
            // on wrong GDN recurrent/conv history (full-attn KV + MLP are
            // unaffected). This mirrors the replay path's Step-(2) seed at the
            // top of `decode_step_paged_batched` EXACTLY (same `row_refs`
            // construction, same `refresh_batched_state_from_rows_in_place`
            // call). It must run BEFORE the snapshot below so the snapshot
            // captures the SEEDED contents — the warm pass then perturbs the
            // seeded slot and the in-place restore returns it to the seeded
            // state before capture (same as bs=1 which snapshots the caller's
            // live state).
            {
                let row_refs: Vec<&crate::forward::LinearAttentionState> =
                    linear_states.iter().map(|s| &**s).collect();
                persistent_state
                    .refresh_batched_state_from_rows_in_place(&row_refs)
                    .context("bs>1 capture: seed persistent GDN slot from caller per-row states")?;
            }
            Self::prepare_gdn_recurrent_state_for_capture(persistent_state)?;
            // Snapshot the persistent GDN slot before the warm pass advances it.
            let gdn_snapshot = persistent_state.snapshot().context(
                "freeze-pointers (batched): snapshot persistent GDN state before warm pass",
            )?;
            let mut graph_inputs = crate::forward::BatchedPagedDecodeGraphInputs {
                token_ids: &token_buffer,
                positions: &position_buffer,
                block_table: &block_table_buffer,
                seqused_k: &seqused_k_buffer,
                kv_slot: &kv_slot_buffer,
                max_seqlen_k: key.max_seqlen_k,
                rotary_cos: &rotary_cos_buffer,
                rotary_sin: &rotary_sin_buffer,
                attn_out: &paged_decode_outputs[..],
                softmax_lse: &paged_decode_lse[..],
                // #1082 boxes 432/433 (STEPS 2-3): the captured HiddenOnly
                // batched forward writes the transformer-stack output here.
                output_hidden: &mut output_hidden,
                linear_state: persistent_state,
            };

            // === #1082 freeze-pointers Pass 1 (Record / warm) — batched ===
            // Run the SAME HiddenOnly batched forward ONCE through the recording
            // arena BEFORE `begin_capture` so every transient Q/K/V/activation
            // + `Flash_fwd_params` backing buffer is allocated and retained;
            // Pass 2 (the captured run below) hands out `Borrowed` views of
            // them so every recorded device pointer stays mapped for the
            // graph's lifetime. Without this the captured
            // `flash_fwd_splitkv_kernel` dereferences a freed Q/activation
            // pointer on replay (compute-sanitizer ILLEGAL_ADDRESS). Mirrors
            // the bs=1 `try_capture` warm pass exactly.
            let warm_result = kiln_tensor::with_capture_arena(arena.clone(), || {
                crate::forward::model_forward_paged_batched_hidden_with_graph_inputs(
                    backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_tables,
                    sequence_lengths,
                    lora,
                    &mut graph_inputs,
                )
            });
            warm_result.context("freeze-pointers (batched) warm (Record) pass failed")?;
            // Restore the persistent GDN slot's CONTENTS in place so the
            // captured Pass-2 advances it exactly once (KV writes are
            // idempotent and need no restore). In-place `slice_set` preserves
            // the slot's canonical device pointers — `*slot = snapshot` would
            // swap in NEW pointers and break the stable-pointer invariant the
            // captured graph (and `refresh_batched_state_from_rows_in_place`)
            // relies on.
            {
                // `slice_set` takes `&self` (in-place write through the kt
                // buffer's device pointer), so a shared borrow of the slot
                // suffices and keeps the slot's tensors' pointers intact.
                let ls = &*graph_inputs.linear_state;
                anyhow::ensure!(
                    ls.recurrent_states.len() == gdn_snapshot.recurrent_states.len()
                        && ls.conv_states.len() == gdn_snapshot.conv_states.len(),
                    "freeze-pointers (batched): GDN snapshot layer-count mismatch on restore"
                );
                for (dst, src) in ls
                    .recurrent_states
                    .iter()
                    .zip(gdn_snapshot.recurrent_states.iter())
                {
                    dst.slice_set(src, 0, 0)
                        .context("freeze-pointers (batched): restore recurrent state in place")?;
                }
                for (dst, src) in ls.conv_states.iter().zip(gdn_snapshot.conv_states.iter()) {
                    dst.slice_set(src, 0, 0)
                        .context("freeze-pointers (batched): restore conv state in place")?;
                }
            }
            // Flip the arena to Replay: Pass 2 hands out Borrowed views of the
            // recorded buffers instead of allocating fresh ones.
            arena.borrow_mut().begin_replay();

            // #1082: the kt buffer allocs filled their contents via an H2D
            // on the kt DEFAULT stream; sync it before capture so those
            // fills (and the in-place GDN restore above) are visible to the
            // captured forward.
            if let Some(idx) = device.index() {
                kiln_tensor::cuda_synchronize_default_stream_for(
                    idx,
                    kiln_tensor::CudaSyncReason::GraphBoundary,
                )
                .context("batched CUDA graph capture: sync kt default stream before capture")?;
            }

            // Synchronize before capture — the capture window must not
            // race with any in-flight launches from the prior step.
            let device_index = device
                .index()
                .context("batched CUDA graph capture device has no index")?;
            kiln_tensor::cuda_synchronize_stream_for(
                device_index,
                &stream,
                kiln_tensor::CudaSyncReason::GraphBoundary,
            )
            .map_err(|e| anyhow::anyhow!("sync before batched graph capture: {e}"))?;

            stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| anyhow::anyhow!("begin_capture (batched): {e}"))?;

            // Phase 7 closeout (#1082): the kt-path GDN decode
            // kernels allocate their own outputs and never read the
            // legacy thread-local; the wrapper has been removed.
            // `gdn_decode_outputs` stays as a struct field
            // (`_gdn_decode_outputs`) so the pre-allocated
            // graph-stable buffers remain alive for the lifetime of
            // the captured graph — same buffer-ownership rationale
            // as `_paged_decode_outputs`. (Historical context: with
            // the candle-typed path, the GDN decode kernel would
            // `candle_core::Tensor::zeros(...)` outputs INSIDE capture, those got
            // freed by `AUTO_FREE_ON_LAUNCH`, and recorded pointers
            // went stale on replay — observed as
            // `CUDA_ERROR_ILLEGAL_ADDRESS` at bs>=16. The kt path
            // sidesteps that by owning its allocations end-to-end.)
            let _ = &gdn_decode_outputs;
            // #1082 boxes 432/433 (STEPS 2-3): engage the capture arena AND the
            // capture stream for the whole batched capture window — identical
            // to the bs=1 `try_capture` Pass-2 scope. The arena hands out
            // `Borrowed` views of the warm-pass buffers; `with_active_cuda_stream`
            // routes every kt op (alloc + the captured HiddenOnly forward + the
            // `cuda_slice_set_dim0` of the hidden into `output_hidden`) onto
            // THIS capture stream. The lm_head is OUT of the captured region
            // now (eager after launch), so there is no `with_lm_head_output_buffer`.
            let forward_result = kiln_tensor::with_capture_arena(arena.clone(), || {
                kiln_tensor::with_active_cuda_stream(stream.clone(), || {
                    crate::forward::model_forward_paged_batched_hidden_with_graph_inputs(
                        backend,
                        token_ids,
                        weights,
                        config,
                        paged_cache,
                        block_tables,
                        sequence_lengths,
                        lora,
                        &mut graph_inputs,
                    )
                })
            });
            let capture_arena_result = arena
                .borrow()
                .ensure_replay_complete()
                .context("batched CUDA graph capture arena allocation sequence mismatch");

            // Instantiate the bs>1 graph with AUTO_FREE_ON_LAUNCH (restored).
            // An earlier change flipped this to flags=0 as a purported BUG2 fix;
            // box102 disproved that (flags=0 → identical doubling), and on this
            // batched path AUTO_FREE is deliberate: it reclaims the per-replay
            // scratch while the freeze-pointers arena pins the persistent buffers.
            // Leaving it at flags=0 would leak that scratch every replay.
            let graph_result = stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );

            forward_result.context("batched forward failed during graph capture")?;
            capture_arena_result?;
            let graph = match graph_result {
                Ok(Some(g)) => g,
                Ok(None) => anyhow::bail!("batched graph capture produced no operations"),
                Err(e) => anyhow::bail!("batched end_capture failed: {e}"),
            };

            tracing::info!(
                batch_size,
                max_seqlen_k = key.max_seqlen_k,
                "CUDA graph captured for batched decode (HiddenOnly + eager lm_head)"
            );

            CapturedBatchedDecodeGraph {
                graph,
                output_hidden,
                capture_stream: stream.clone(),
                adapter_gen,
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
                _gdn_decode_outputs: gdn_decode_outputs,
                _capture_arena_buffers: arena.borrow_mut().take_retained(),
                max_seqlen_k: key.max_seqlen_k,
            }
        };
        // (#1082 Phase 5) Stream capture only RECORDED the batched forward — it
        // did NOT execute, so `output_hidden` is still the uninitialized
        // capture-time buffer and the in-place recurrent/KV state was not
        // advanced. Launch the instantiated graph once now (the token /
        // position / rotary / metadata buffers already hold THIS step's inputs)
        // to actually compute this step + advance state, then sync so
        // `output_hidden` is valid before we read it. Mirrors bs=1 try_capture.
        captured
            .graph
            .launch()
            .context("execute captured batched decode graph (first run)")?;
        kiln_tensor::cuda_synchronize_stream_for(
            arena_device_index,
            &stream,
            kiln_tensor::CudaSyncReason::GraphBoundary,
        )
        .map_err(|e| anyhow::anyhow!("sync after first batched captured-graph launch: {e}"))?;
        // #1082 bs>1 greedy-coherence fix: scatter the post-step persistent GDN
        // slot back into the caller's per-row `linear_states` so this capture
        // step advances them by exactly one token — same as the replay path's
        // Step-(6) scatter (cuda_graph.rs scatter tail). The first launch above
        // advanced the seeded slot in place; mirror the replay re-borrow of
        // `self.batched_state_pool` (the capture-scope `&mut` borrow already
        // ended when `graph_inputs` was dropped) and call
        // `scatter_batch_rows_replace_with_backend(backend, linear_states)`
        // EXACTLY as the replay tail does (same arg-borrow forms). Without this,
        // the caller's per-row states stay at their pre-capture (post-prefill)
        // values and the NEXT decode step seeds the slot from stale history.
        {
            let persistent = self
                .batched_state_pool
                .get_mut(&batch_size)
                .context("missing persistent batched state slot at capture scatter time")?;
            persistent
                .scatter_batch_rows_replace_with_backend(backend, linear_states)
                .context(
                    "bs>1 capture: scatter post-step GDN slot back into caller per-row states",
                )?;
        }
        // #1082 boxes 432/433 (STEPS 2-3): run `final_norm` + lm_head EAGERLY on
        // the capture-step hidden to produce this step's logits — the large-N
        // lm_head cublasLt GEMV is OUT of the captured graph (box-102 BUG2 fix).
        // `output_hidden` now holds the first-launch hidden (synced above).
        // Argmax + DtoH then run OUTSIDE the captured region. Mirrors the bs=1
        // try_capture tail (eager `lm_head_from_hidden_eager` → logits).
        let logits = crate::forward::lm_head_from_batched_hidden_eager(
            backend,
            &captured.output_hidden,
            weights,
            config,
        )
        .context("box-102 fix (batched): eager lm_head on captured hidden (first launch)")?;
        let tokens = crate::sampling::greedy_sample_rows(&logits)
            .context("argmax over batched eager-lm_head logits failed")?;
        self.captured_batched.insert(key, captured);
        Ok(tokens)
    }

    /// Eager (non-graph) paged decode.
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
        // #1082: `model_forward_paged` is kt-typed (returns kt logits) and
        // `decode_step_paged` now returns kt too — return the kt logits
        // directly, no candle bridge.
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
            None, // no pre-allocated position buffer — creates one internally
        )
        .context("eager decode forward pass failed")
    }

    // #1082: every `new_*_buffer` constructor now allocates a kt-native
    // `Tensor` directly on the kt `Device` via `Tensor::{from_vec_on,
    // zeros_on}` (device-correct, NOT a CPU-default `from_vec`). The
    // resulting buffers own a persistent device pointer that gets baked
    // into the captured graph; the `update_*` family refreshes their
    // contents in place via `cuda_write_host_in_place`.
    #[cfg(feature = "cuda")]
    fn new_token_buffer(device: Device, token_id: u32) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![token_id], vec![1])
            .context("create CUDA graph token buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_position_buffer(device: Device, position: usize) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![position as f32], vec![1])
            .context("create CUDA graph position buffer")
    }

    /// Allocate a `[batch] u32` token-id buffer with the row 0…batch-1
    /// contents pre-filled from `token_ids`. The captured batched graph
    /// reads its embedding-lookup indices from this device pointer; the
    /// runner refreshes the contents in place via `cuda_write_host_in_place`
    /// before each replay.
    #[cfg(feature = "cuda")]
    fn new_batched_token_buffer(device: Device, token_ids: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "new_batched_token_buffer requires a non-empty batch"
        );
        Tensor::from_vec_on(device, token_ids.to_vec(), vec![token_ids.len()])
            .context("create CUDA graph batched token buffer")
    }

    /// Allocate a `[batch] f32` per-row decode-position buffer pre-filled
    /// from `start_positions`. Used by the batched RoPE path under
    /// capture — the captured kernels see the same device pointer and
    /// read whatever the runner writes before each replay.
    #[cfg(feature = "cuda")]
    fn new_batched_position_buffer(device: Device, start_positions: &[usize]) -> Result<Tensor> {
        anyhow::ensure!(
            !start_positions.is_empty(),
            "new_batched_position_buffer requires a non-empty batch"
        );
        let positions_f32: Vec<f32> = start_positions.iter().map(|&p| p as f32).collect();
        let n = positions_f32.len();
        Tensor::from_vec_on(device, positions_f32, vec![n])
            .context("create CUDA graph batched position buffer")
    }

    #[cfg(feature = "cuda")]
    fn padded_block_table(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
    ) -> Result<Vec<u32>> {
        let block_size = paged_cache.block_size();
        // #1082: FA2_KBLOCK_N (=64 hdim256), matching the CudaGraphKey sizing
        // above + forward.rs's K_BLOCK_N. `max_seqlen_k` arrives already bucketed
        // to a multiple of FA2_KBLOCK_N by the key, so this divides cleanly.
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let pages_per_chunk = kblock_n / block_size;
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        let take = max_blocks_per_seq.min(block_table.blocks.len());
        let mut padded = Vec::with_capacity(max_blocks_per_seq);
        padded.extend_from_slice(&block_table.blocks[..take]);
        if padded.is_empty() {
            anyhow::bail!("paged decode graph block table is empty");
        }
        // #1082 box-102: pad with the LAST REAL block index (repeated), NOT an
        // incrementing one. Incrementing past the last real block can name a
        // page BEYOND the allocated KV cache; if the captured flash kernel
        // touches a padded entry (its `n_block` range derives from the baked
        // `max_seqlen_k`, not the live length) it computes a K base pointer
        // `k_pool + padded_block*stride` past the pool → the
        // CUDA_ERROR_ILLEGAL_ADDRESS at flash_fwd_kernel.h:827 (K-tile load,
        // sanitizer-confirmed). A repeated valid block index is always
        // in-bounds; the `seqused_k` predicate masks its redundant data.
        let pad_block = *padded.last().expect("padded is non-empty (checked above)");
        while padded.len() < max_blocks_per_seq {
            padded.push(pad_block);
        }
        Ok(padded)
    }

    #[cfg(feature = "cuda")]
    fn new_block_table_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
        device: Device,
    ) -> Result<Tensor> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        let len = padded.len();
        Tensor::from_vec_on(device, padded, vec![1, len])
            .context("create CUDA graph block table buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_seqused_k_buffer(device: Device, attention_len: usize) -> Result<Tensor> {
        // #1082: U32 in the kt path (kt flash-attn requires U32 seqused_k;
        // the value is bit-identical to the old i32 for valid lengths).
        Tensor::from_vec_on(device, vec![attention_len as u32], vec![1])
            .context("create CUDA graph seqused_k buffer")
    }

    #[cfg(feature = "cuda")]
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
        Tensor::from_vec_on(device, vec![slot], vec![1]).context("create CUDA graph KV slot buffer")
    }

    /// Allocate a `[batch, max_blocks_per_seq]` u32 padded block-table
    /// buffer covering every row. Reuses the per-row `padded_block_table`
    /// helper and stacks the rows. Used by the batched capture path so
    /// flash-attention reads page metadata from a graph-stable pointer.
    #[cfg(feature = "cuda")]
    fn new_batched_block_table_buffer(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
        device: Device,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            !block_tables.is_empty(),
            "new_batched_block_table_buffer requires a non-empty batch"
        );
        let mut flat: Vec<u32> = Vec::new();
        let mut width: Option<usize> = None;
        for bt in block_tables {
            let padded = Self::padded_block_table(bt, paged_cache, max_seqlen_k)?;
            if width.is_none() {
                width = Some(padded.len());
            } else if width != Some(padded.len()) {
                anyhow::bail!(
                    "new_batched_block_table_buffer: inconsistent padded widths ({} vs {})",
                    width.unwrap(),
                    padded.len()
                );
            }
            flat.extend_from_slice(&padded);
        }
        let width = width.unwrap();
        Tensor::from_vec_on(device, flat, vec![block_tables.len(), width])
            .context("create CUDA graph batched block table buffer")
    }

    /// Allocate a `[batch]` per-row seqused_k buffer pre-filled from
    /// each row's `start_pos + 1`. #1082: U32 (kt flash-attn contract).
    #[cfg(feature = "cuda")]
    fn new_batched_seqused_k_buffer(device: Device, start_positions: &[usize]) -> Result<Tensor> {
        anyhow::ensure!(
            !start_positions.is_empty(),
            "new_batched_seqused_k_buffer requires a non-empty batch"
        );
        let seqused: Vec<u32> = start_positions
            .iter()
            .map(|&p| u32::try_from(p + 1).context("seqused_k exceeds u32 range"))
            .collect::<Result<Vec<_>>>()?;
        let n = seqused.len();
        Tensor::from_vec_on(device, seqused, vec![n])
            .context("create CUDA graph batched seqused_k buffer")
    }

    /// Allocate a `[batch] u32` per-row KV-write-slot buffer for the
    /// current decode step.
    #[cfg(feature = "cuda")]
    fn new_batched_kv_slot_buffer(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        start_positions: &[usize],
        device: Device,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            !block_tables.is_empty(),
            "new_batched_kv_slot_buffer requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == start_positions.len(),
            "new_batched_kv_slot_buffer: block_tables.len() != start_positions.len()"
        );
        let mut slots: Vec<u32> = Vec::with_capacity(block_tables.len());
        for (bt, &pos) in block_tables.iter().zip(start_positions.iter()) {
            let slot = bt
                .slot_for(pos, paged_cache.block_size())
                .with_context(|| format!("no slot for decode position {pos}"))?
                as u32;
            slots.push(slot);
        }
        let n = slots.len();
        Tensor::from_vec_on(device, slots, vec![n])
            .context("create CUDA graph batched KV slot buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_rotary_cos_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        // #34 BUG2 FIX: GPU rotary (matches eager + `update_rotary_buffers`), not
        // host CPU cos. This is the capture-time fill; the per-replay refresh is
        // in `update_rotary_buffers`. Both must be on-device or replay diverges.
        let inv_freq = crate::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            &device,
        )?;
        let pos = Tensor::from_vec_on(device, vec![position as f32], vec![1])?;
        let (cos, _) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        cos.to_dtype(kiln_tensor::DType::F32)?
            .contiguous()
            .context("create CUDA graph rotary cos buffer (gpu)")
    }

    #[cfg(feature = "cuda")]
    fn new_rotary_sin_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        // #34 BUG2 FIX: GPU rotary (see `new_rotary_cos_buffer`).
        let inv_freq = crate::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            &device,
        )?;
        let pos = Tensor::from_vec_on(device, vec![position as f32], vec![1])?;
        let (_, sin) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        sin.to_dtype(kiln_tensor::DType::F32)?
            .contiguous()
            .context("create CUDA graph rotary sin buffer (gpu)")
    }

    /// #1082 box-102 FIX: graph-stable `[1, 1, hidden_size]` PRE-final-norm
    /// hidden buffer. The captured `HiddenOnly` forward writes here;
    /// final_norm + lm_head run eagerly on it after replay (the lm_head
    /// GEMV is not graph-replay-safe). Dtype matches the model's hidden
    /// dtype (bf16).
    #[cfg(feature = "cuda")]
    fn new_output_hidden(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
    ) -> Result<Tensor> {
        Tensor::zeros_on(device, vec![1, 1, config.hidden_size], dtype)
            .context("create CUDA graph output hidden")
    }

    // #1082 boxes 432/433 (STEPS 2-3): `new_lm_head_output_buffer` (the
    // `[batch, 1, vocab]` in-graph lm-head matmul destination) was deleted —
    // the batched capture path no longer records the lm_head; `final_norm` +
    // lm_head run eagerly on `output_hidden` after launch (box-102 BUG2 fix),
    // so there is no in-graph matmul output to pin.

    #[cfg(feature = "cuda")]
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
                .context("create CUDA graph paged decode output")?,
            );
            lse.push(
                Tensor::zeros_on(
                    device,
                    vec![1, config.num_attention_heads, 1],
                    kiln_tensor::DType::F32,
                )
                .context("create CUDA graph paged decode LSE")?,
            );
        }
        Ok((outputs, lse))
    }

    #[cfg(feature = "cuda")]
    fn prepare_gdn_recurrent_state_for_capture(
        linear_state: &mut LinearAttentionState,
    ) -> Result<()> {
        for state in &mut linear_state.recurrent_states {
            // #1082: recurrent_states are kt tensors; use kt DType.
            if state.dtype() != kiln_tensor::DType::BF16 {
                *state = state
                    .to_dtype(kiln_tensor::DType::BF16)
                    .context("prepare CUDA graph GDN recurrent state")?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
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
                .context("create CUDA graph GDN decode output")?,
            );
        }
        Ok(outputs)
    }

    /// `[batch, rotary_dim/2]` rotary cosine buffer initialized at
    /// `position 0`. The runner refreshes the per-row contents before
    /// each replay so the captured RoPE kernels see updated angles
    /// while reading from a graph-stable pointer.
    #[cfg(feature = "cuda")]
    fn new_batched_rotary_cos_buffer(
        config: &ModelConfig,
        device: Device,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "rotary cos buffer requires batch > 0");
        let half = config.rotary_dim() / 2;
        Tensor::zeros_on(device, vec![batch, half], kiln_tensor::DType::F32)
            .context("create CUDA graph batched rotary cos buffer")
    }

    /// `[batch, rotary_dim/2]` rotary sine buffer initialized at
    /// `position 0`. See `new_batched_rotary_cos_buffer` for the
    /// replay-time refresh semantics.
    #[cfg(feature = "cuda")]
    fn new_batched_rotary_sin_buffer(
        config: &ModelConfig,
        device: Device,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "rotary sin buffer requires batch > 0");
        let half = config.rotary_dim() / 2;
        Tensor::zeros_on(device, vec![batch, half], kiln_tensor::DType::F32)
            .context("create CUDA graph batched rotary sin buffer")
    }

    // #1082 boxes 432/433 (STEPS 2-3): `new_batched_output_logits` (the
    // `[batch, 1, vocab]` in-graph output-logits buffer) was deleted — the
    // batched capture path is now HiddenOnly; the runner samples per-row tokens
    // from the eager-lm_head logits (`lm_head_from_batched_hidden_eager`) off
    // the captured `output_hidden`, so there is no in-graph logits buffer.

    /// `[batch, 1, hidden_size]` PRE-final-norm hidden buffer for batched
    /// capture (#1082 boxes 432/433, STEPS 2-3). Batched twin of
    /// [`new_output_hidden`] (which is `[1, 1, hidden_size]`). The captured
    /// `HiddenOnly` batched forward writes the transformer-stack output here
    /// via `cuda_slice_set_dim0`; `final_norm` + lm_head run EAGERLY on it off
    /// the graph (see [`CapturedBatchedDecodeGraph::output_hidden`]).
    #[cfg(feature = "cuda")]
    fn new_batched_output_hidden(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "batched output hidden requires batch > 0");
        Tensor::zeros_on(device, vec![batch, 1, config.hidden_size], dtype)
            .context("create CUDA graph batched output hidden")
    }

    /// Per-full-attention-layer paged decode outputs and LSE scratch,
    /// shaped for `[batch, 1, n_heads, head_dim]` and `[batch, n_heads, 1]`.
    /// One element per full-attention layer in the model.
    #[cfg(feature = "cuda")]
    fn new_batched_paged_decode_outputs(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
        batch: usize,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        anyhow::ensure!(batch > 0, "batched paged decode outputs require batch > 0");
        let mut outputs = Vec::with_capacity(config.num_full_attention_layers);
        let mut lse = Vec::with_capacity(config.num_full_attention_layers);
        for _ in 0..config.num_full_attention_layers {
            outputs.push(
                Tensor::zeros_on(
                    device,
                    vec![batch, 1, config.num_attention_heads, config.head_dim],
                    dtype,
                )
                .context("create CUDA graph batched paged decode output")?,
            );
            lse.push(
                Tensor::zeros_on(
                    device,
                    vec![batch, config.num_attention_heads, 1],
                    kiln_tensor::DType::F32,
                )
                .context("create CUDA graph batched paged decode LSE")?,
            );
        }
        Ok((outputs, lse))
    }

    /// Per-linear-attention-layer fused GDN decode outputs, shaped for
    /// `[batch, 1, linear_num_value_heads, linear_value_head_dim]`.
    /// One element per linear-attention (GDN) layer in the model.
    #[cfg(feature = "cuda")]
    fn new_batched_gdn_decode_outputs(
        config: &ModelConfig,
        device: Device,
        batch: usize,
    ) -> Result<Vec<Tensor>> {
        anyhow::ensure!(batch > 0, "batched GDN decode outputs require batch > 0");
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let mut outputs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            outputs.push(
                Tensor::zeros_on(
                    device,
                    vec![
                        batch,
                        1,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                    ],
                    kiln_tensor::DType::BF16,
                )
                .context("create CUDA graph batched GDN decode output")?,
            );
        }
        Ok(outputs)
    }

    /// Eager decode that uses the same pre-allocated position tensor path as
    /// graph capture. This primes kernels/modules that the plain eager path
    /// skips, keeping unsupported lazy work out of the later capture window.
    #[cfg(feature = "cuda")]
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
        // #1082: allocate the kt position buffer directly on the kt device
        // (no candle alloc, no bridges) and return the kt logits directly.
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
        .context("graph-shaped eager decode forward pass failed")
    }
}

// SAFETY: CudaGraphRunner is protected by a Mutex in ModelRunner. The inner
// CudaGraph/CudaGraphExec are GPU-side recorded command sequences. Launching a
// graph is thread-safe — the CUDA driver serialises access on the stream.
// The raw pointers (*mut CUgraph_st, *mut CUgraphExec_st) are opaque handles
// to driver-managed objects and are not dereferenced on the CPU side.
unsafe impl Send for CudaGraphRunner {}
unsafe impl Sync for CudaGraphRunner {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_policy_defaults_eager_and_validates_cache_bounds() {
        let default_policy = CudaGraphExecutionPolicy::default();
        assert_eq!(default_policy, CudaGraphExecutionPolicy::disabled());
        assert!(!default_policy.enabled());
        assert_eq!(
            default_policy.max_cached_graphs(),
            CudaGraphExecutionPolicy::DEFAULT_MAX_CACHED_GRAPHS
        );

        let enabled = CudaGraphExecutionPolicy::try_new(true, 16).unwrap();
        assert!(enabled.enabled());
        assert_eq!(enabled.max_cached_graphs(), 16);
        assert!(CudaGraphExecutionPolicy::try_new(true, 0).is_err());
        assert!(
            CudaGraphExecutionPolicy::try_new(
                true,
                CudaGraphExecutionPolicy::MAX_CACHED_GRAPHS + 1,
            )
            .is_err()
        );
    }

    #[test]
    fn test_new_cpu_disables_graphs() {
        let policy = CudaGraphExecutionPolicy::try_new(true, 16).unwrap();
        let runner = CudaGraphRunner::new(&kiln_tensor::Device::Cpu, policy);
        assert!(!runner.is_enabled());
        assert_eq!(runner.policy.max_cached_graphs(), 16);
    }

    #[test]
    fn test_new_disabled() {
        let runner = CudaGraphRunner::new(
            &kiln_tensor::Device::Cpu,
            CudaGraphExecutionPolicy::disabled(),
        );
        assert!(!runner.is_enabled());
    }

    #[test]
    fn test_invalidate_resets_state() {
        let mut runner = CudaGraphRunner::new(
            &kiln_tensor::Device::Cpu,
            CudaGraphExecutionPolicy::disabled(),
        );
        runner.warmup_done = true;
        #[cfg(feature = "cuda")]
        {
            runner.cache_full_warned = true;
        }
        runner.invalidate();
        assert!(!runner.warmup_done);
        assert_eq!(runner.adapter_generation, 1);
        #[cfg(feature = "cuda")]
        assert!(!runner.cache_full_warned);
    }

    #[test]
    fn test_multiple_invalidations_increment_generation() {
        let mut runner = CudaGraphRunner::new(
            &kiln_tensor::Device::Cpu,
            CudaGraphExecutionPolicy::disabled(),
        );
        runner.invalidate();
        runner.invalidate();
        runner.invalidate();
        assert_eq!(runner.adapter_generation, 3);
    }
}
