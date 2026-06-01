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
//! ## Multi-batch (`bs > 1`) capture — not yet implemented
//!
//! Today the runner only captures the `bs = 1` decode shape. The
//! `paged_batched_decode_step` path for `row_count > 1` runs eager, which is
//! the single biggest remaining throughput gap vs. vLLM at high concurrency
//! on Qwen3.5-4B / L40S (kiln 1181 tok/s @ bs=64 vs. vLLM 1907 tok/s — see
//! `BENCHMARKS.md` "Direct head-to-head" section, commit `5fddb497`).
//!
//! ### Proposed structure
//!
//! - **`CudaBatchedGraphKey { batch_size, max_seqlen_k, max_blocks_per_seq,
//!   stable_metadata }`** — same `stable_metadata` cliff as today; batch-size
//!   bucketed cache.
//! - **`CapturedBatchedDecodeGraph`** — same fields as `CapturedDecodeGraph`
//!   but every per-row tensor (`token_buffer`, `position_buffer`,
//!   `block_table_buffer`, `seqused_k_buffer`, `kv_slot_buffer`,
//!   `output_logits`, per-full-attn-layer outputs/LSE) sized for `[batch,
//!   ...]` instead of `[1, ...]`. GDN decode outputs gain a `batch` dim too.
//! - **Persistent batched `LinearAttentionState` slot per bucket** — the
//!   current per-row `recurrent_states[layer_idx]` / `conv_states[layer_idx]`
//!   tensors have new pointers on every `from_batch_rows` call. Capture
//!   needs stable device addresses, so the runner should own one batched
//!   state pool keyed on `batch_size` and copy-in the per-row contents
//!   (via the existing scatter primitives) before each replay.
//! - **Per-bucket cap** — vLLM captures 51 sizes (1, 2, 4, …, 512); for
//!   kiln a bounded set like `[1, 2, 4, 8, 16, 32, 64]` covers the
//!   `KILN_MAX_DECODE_BATCH` default range. Bucket the request batch to
//!   the next-larger captured size (or fall back to eager).
//! - **Entry point** — `CudaGraphRunner::decode_step_paged_batched(...)`
//!   mirroring `decode_step_paged` but taking `&[u32]` / `&[&BlockTable]`
//!   / `&[usize]` / `&mut [&mut LinearAttentionState]`. Wire into
//!   `ModelRunner::paged_batched_decode_step` in `generate.rs` at the
//!   `try_contiguous_batched` branch — before invoking
//!   `model_forward_paged_batched_decode_hidden`, route through the
//!   batched graph runner when the bucket is captured.
//! - **Forward wrapper** — `model_forward_paged_batched_with_graph_inputs`
//!   in `forward.rs` mirroring the existing single-row
//!   `model_forward_paged_with_graph_inputs`, consuming
//!   `BatchedPagedDecodeGraphInputs` for the stable per-row tensors.
//!
//! ### Sequencing
//!
//! 1. Land the key + struct + runner fields behind `#[allow(dead_code)]`
//!    to verify the type design compiles cleanly.
//! 2. Add the persistent batched-state pool (separate commit) — reuses
//!    `LinearAttentionState::from_batch_rows` shape but with in-place
//!    storage via the existing `assemble_gdn_recurrent_resident_batch_rows`
//!    backend primitive.
//! 3. Add the batched forward wrapper consuming stable inputs.
//! 4. Add capture + replay logic.
//! 5. Wire into `paged_batched_decode_step`.
//! 6. Bench: target ≥ 1.5× kiln throughput at bs=32-64 to close most of
//!    the vLLM gap.
//!
//! ### Why this hasn't landed yet
//!
//! The graph capture has to pin **every** device pointer touched in
//! the forward, including the GDN recurrent state, the conv state,
//! per-layer attention scratch, paged-KV pool slots, RoPE tables, and
//! the LM head output. Any allocation Candle frees between capture and
//! replay turns the graph into a use-after-free. The bs=1 path
//! enumerates and pre-allocates these carefully; extending the same
//! discipline to a batched shape is several commits worth of work and
//! is currently the top-priority entry on this file's TODO list.

use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use tracing;

use kiln_core::config::ModelConfig;

use crate::backend::BackendRuntime;
#[cfg(feature = "cuda")]
use crate::forward::PagedDecodeGraphInputs;
#[cfg(feature = "cuda")]
use crate::forward::model_forward_paged_hidden_with_graph_inputs;
use crate::forward::{GpuWeights, LinearAttentionState, model_forward_paged};
use crate::lora_loader::LoraWeights;
use crate::PagedKvCacheKt;

// #1082: the CUDA-graph stable device buffers are now kt-native
// `kiln_tensor::Tensor`s (post-flip convention: bare `Tensor` = kt).
// Allocated once with stable device pointers (`Tensor::{zeros_on,
// from_vec_on}`) and refreshed in place before each replay via
// `kiln_tensor::cuda_write_host_in_place` — both honor the captured
// graph's baked device pointer.
use kiln_tensor::{Device, Tensor};

use kiln_core::block::BlockTable;

/// Holds a captured CUDA graph ready for replay.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaGraphKey {
    stable_metadata: bool,
    seq_len: usize,
    block_table: Vec<u32>,
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
}

#[cfg(feature = "cuda")]
impl CudaGraphKey {
    fn new(block_table: &BlockTable, paged_cache: &PagedKvCacheKt, seq_len: usize) -> Self {
        let stable_metadata = Self::stable_paged_metadata_enabled();
        let attention_len = seq_len + 1;
        let max_seqlen_k = attention_len.div_ceil(128) * 128;
        let pages_per_chunk = 128 / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / 128) * pages_per_chunk;
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

    fn stable_paged_metadata_enabled() -> bool {
        std::env::var("KILN_CUDA_GRAPH_STABLE_PAGED_METADATA")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
            .unwrap_or(false)
    }
}

/// Read the `KILN_CUDA_GRAPHS_BATCHED` env var.
///
/// Two-stage gating: `KILN_CUDA_GRAPHS=true` enables the (existing,
/// stable) bs=1 capture/replay path; `KILN_CUDA_GRAPHS_BATCHED`
/// additionally engages the bs>1 capture/replay path. Both must
/// hold for batched graphs to engage; either being off sends the
/// bs>1 caller down the eager batched path.
///
/// **Default REVERTED to OFF (2026-05-26)** — the earlier flip to ON
/// at `6d564b9a` was based on `kiln-bench` sequential-decode runs
/// that never exercised the actual batched concurrent decode path.
/// Concurrent bench against `kiln serve` showed every bs≥2 request
/// returning HTTP 500 with a swallowed inner error at
/// `cuda_graph.rs:1595` (`"batched forward failed during graph
/// capture"`) → bad CUDA context → `CUDA_ERROR_ILLEGAL_ADDRESS` on
/// the subsequent replay → eager-batched fallback also fails on the
/// same poisoned context. The eager-batched path (with this flag
/// `0`) is healthy: bs=64 → 498 tok/s on A6000 at HEAD `2d9d4fc4`
/// after the GDN-decode contiguity fix in the same commit.
///
/// ⚠️ CORRECTION (2026-06-01, server-path repro): the earlier
/// "Phase 5 sanitizer sweep" claim that the **bs=1** capture+replay
/// path was validated is FALSE. Driving the real graph path
/// (`kiln serve` → `decode_step_paged`, NOT `kiln-bench` which bypasses
/// the runner) surfaces TWO bugs even at bs=1. See
/// `bench-results/cuda-graph-box102-findings.md`:
///   BUG 1 (OOB, root-caused + fix confirmed): the graph-stable
///   metadata buffers are gated behind `KILN_CUDA_GRAPH_STABLE_PAGED_METADATA`
///   (default OFF, `cuda_graph.rs:156`). With it off the captured forward
///   builds a TRANSIENT block_table that is freed after capture, so the
///   captured `flash_fwd_splitkv_kernel` reads a dangling pointer →
///   `CUDA_ERROR_ILLEGAL_ADDRESS` (compute-sanitizer: wild/wrapped read
///   address). Setting the env to `1` eliminates the OOB.
///   BUG 2 (replay correctness, OPEN): with the env on, replay no longer
///   crashes but emits token-doubling garbage ("a a thinking thinking …"
///   vs eager "a thinking …"); not a stream race (persists under
///   `CUDA_LAUNCH_BLOCKING=1`). This matches the KV-slot-under-replay
///   suspect below. The keystone is NOT mergeable until BUG 2 is fixed.
///
/// Set `KILN_CUDA_GRAPHS_BATCHED=1` to opt in once the underlying
/// capture bug is fixed and re-validated end-to-end against
/// `bench-concurrent-batch.py`.
///
/// _Historical_: Phase 5 sanitizer sweep on A6000 at HEAD `a2cb9edb`:
/// decode 74.4 tok/s, mean ITL 13.45ms, peak VRAM 11.2 GB on
/// Qwen3.5-4B paged decode at
/// batches 1/4/8/16) AND sanitizer reports `========= ERROR
/// SUMMARY: 0 errors` under the full live-driver path with
/// `KILN_CUDA_GRAPHS_BATCHED=1 KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`.
/// See `bench-results/cuda-graph-status.md` "Phase 5 sanitizer
/// sweep" section for the validation trail.
///
/// Set `KILN_CUDA_GRAPHS_BATCHED=1` (or `true`, `yes`, `on`) to opt
/// in once the batched-capture bug is fixed.
#[cfg(feature = "cuda")]
fn batched_graph_enabled() -> bool {
    std::env::var("KILN_CUDA_GRAPHS_BATCHED")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Cache key for the (planned, not-yet-wired) batched (`bs > 1`) decode
/// graph cache. Mirrors [`CudaGraphKey`] but with an explicit
/// `batch_size` bucket. See the multi-batch design note at the top of
/// this file for the surrounding plan.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)]
struct CudaBatchedGraphKey {
    /// Same stable-paged-metadata cliff as `CudaGraphKey`.
    stable_metadata: bool,
    /// Number of rows the captured graph was specialized for.
    batch_size: usize,
    /// When `stable_metadata=false`, encodes the per-row seq_len so a
    /// changed K/V length triggers a re-capture; zero otherwise.
    max_seqlen_k: usize,
    /// Padded block-table width, in physical pages.
    max_blocks_per_seq: usize,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
impl CudaBatchedGraphKey {
    /// Build a batched key from the same primitives used by
    /// [`CudaGraphKey::new`], applied to the largest seq_len in the
    /// batch (rounded up to the 128 K/V chunk). Bucketing all rows to
    /// the same `max_seqlen_k` lets one captured graph serve every
    /// row at that decode step.
    fn new(
        batch_size: usize,
        max_seq_len: usize,
        paged_cache: &PagedKvCacheKt,
    ) -> Self {
        let stable_metadata = CudaGraphKey::stable_paged_metadata_enabled();
        let attention_len = max_seq_len + 1;
        let max_seqlen_k = attention_len.div_ceil(128) * 128;
        let pages_per_chunk = 128 / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / 128) * pages_per_chunk;
        Self {
            stable_metadata,
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
    block_table_buffer: Option<Tensor>,
    /// Pre-allocated actual K/V attention length buffer on GPU.
    /// #1082: U32 (the kt flash-attn path requires `seqused_k` U32 — same
    /// 4-byte layout the candle i32 buffer carried, which the candle->kt
    /// borrow reinterpreted as U32 anyway).
    seqused_k_buffer: Option<Tensor>,
    /// Pre-allocated current KV write slot buffer on GPU (u32, shape [1]).
    kv_slot_buffer: Option<Tensor>,
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
}

/// Captured graph + stable buffers for a batched (`bs > 1`) decode step.
/// Reserved for the multi-batch capture path documented at the top of
/// this file; not yet populated. The fields mirror
/// [`CapturedDecodeGraph`] but every per-row tensor is shaped for
/// `[batch, ...]` so one graph replay services the whole batch.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
struct CapturedBatchedDecodeGraph {
    /// The instantiated CUDA graph.
    graph: cudarc::driver::CudaGraph,
    /// `[batch, 1, vocab]` logits — replay writes into this storage.
    /// #1082: kt-native graph-stable buffer.
    output_logits: Tensor,
    /// Adapter generation when captured (invalidate on mismatch).
    adapter_gen: u64,
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
    /// Pre-allocated lm-head matmul output buffer, shape `[batch, 1, vocab]`.
    /// See [`CapturedDecodeGraph::_lm_head_output_buffer`] for the
    /// rationale. This is the structural Phase 5 #1082 fix that
    /// unblocks `KILN_CUDA_GRAPHS_BATCHED=1` end-to-end.
    _lm_head_output_buffer: Tensor,
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
    /// Whether CUDA graphs are enabled.
    enabled: bool,
    /// Captured graphs keyed by graph-unsafe paged metadata.
    #[cfg(feature = "cuda")]
    captured: HashMap<CudaGraphKey, CapturedDecodeGraph>,
    /// Captured batched graphs keyed on `(batch_size, max_seqlen_k, …)`.
    /// Empty today; populated by the planned multi-batch capture path.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    captured_batched: HashMap<CudaBatchedGraphKey, CapturedBatchedDecodeGraph>,
    /// Per-batch-size warmup tracker. Each new bucket needs one eager
    /// call to prime the allocator before its first capture attempt;
    /// without per-bucket warmup the global `warmup_done` flag set by
    /// an earlier bs=1 capture caused new batched buckets to capture
    /// against a cold allocator and hit `CUDA_ERROR_ILLEGAL_ADDRESS`.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    batched_state_pool: HashMap<usize, crate::forward::LinearAttentionState>,
    /// Adapter generation counter; incremented on LoRA swap.
    adapter_generation: u64,
    /// Whether warmup is complete.
    warmup_done: bool,
    /// Whether we already warned that the paged metadata graph cache is full.
    #[cfg(feature = "cuda")]
    cache_full_warned: bool,
    /// #1082 box-102 BUG-B fix: identity (recurrent-state `TensorId`) of the
    /// request whose GDN recurrent/conv state the captured bs=1 graph currently
    /// holds. A fresh `LinearAttentionState` per request (generate.rs
    /// `new_linear_state`) yields a new id; it is stable within a request (the
    /// graph replay path does not reassign `linear_state`). On change we evict
    /// the captured graph so the new request RE-CAPTURES with its own
    /// post-prefill state instead of replaying on the PRIOR request's leftover
    /// recurrent state (which produced deterministic garbage on every request
    /// after the first).
    #[cfg(feature = "cuda")]
    last_request_state_id: Option<kiln_tensor::TensorId>,
}

impl CudaGraphRunner {
    /// Create a new graph runner. Enabled only on CUDA devices with the `cuda` feature.
    pub fn new(device: &kiln_tensor::Device, enabled: bool) -> Self {
        let is_cuda = matches!(device, kiln_tensor::Device::Cuda(_));
        let actually_enabled = enabled && is_cuda;
        if actually_enabled {
            tracing::info!("CUDA graphs enabled for decode");
        } else if enabled && !is_cuda {
            tracing::debug!("CUDA graphs requested but no CUDA device, using eager decode");
        }
        Self {
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
            last_request_state_id: None,
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
    #[allow(dead_code)]
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
        if !self.batched_state_pool.contains_key(&batch_size) {
            // (#1082) kt-native — the device is already kt.
            let state =
                crate::forward::LinearAttentionState::new_with_batch_for_inference_backend(
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

    /// Whether multi-batch CUDA graph capture/replay is enabled.
    ///
    /// Even when `is_enabled()` is true, the batched path is gated on
    /// a separate opt-in (`KILN_CUDA_GRAPHS_BATCHED=1`) until the
    /// implementation is fully validated. The graph runner's bs=1
    /// path is the production default; the batched path lands
    /// behind this flag so it can be benched in isolation and rolled
    /// back without re-flipping `KILN_CUDA_GRAPHS`.
    #[cfg(feature = "cuda")]
    pub fn is_batched_enabled(&self) -> bool {
        self.enabled && batched_graph_enabled()
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
        // runs eager once so Candle / cudarc prime any lazy
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

        // Diagnostic: when `KILN_CUDA_GRAPHS_BATCHED_NO_REPLAY=1`,
        // always evict the cached graph before checking — this
        // forces every step to re-capture, never replay. If the
        // bench succeeds under this mode but fails without it,
        // the fault is isolated to the replay path; if it still
        // fails, capture itself is broken.
        if std::env::var("KILN_CUDA_GRAPHS_BATCHED_NO_REPLAY")
            .map(|v| matches!(v.as_str(), "1" | "true"))
            .unwrap_or(false)
        {
            self.captured_batched.remove(&key);
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
        // (5) Argmax `output_logits` outside the captured region to
        //     produce per-row tokens.
        // (6) Scatter the post-step persistent state back into each
        //     per-row `LinearAttentionState` so callers see the
        //     updated GDN history.
        //
        // Borrow plumbing: capture-time grabs `&self.captured_batched`,
        // while state refresh needs `&mut self.batched_state_pool`.
        // We use disjoint field borrows by going through the
        // HashMaps directly instead of via `self.persistent_batched_state(...)`.
        let adapter_gen_now = self.adapter_generation;
        let captured_exists_with_match = self
            .captured_batched
            .get(&key)
            .map(|c| c.adapter_gen == adapter_gen_now)
            .unwrap_or(false);
        if let Some(captured) = self.captured_batched.get(&key) {
            if captured.adapter_gen != adapter_gen_now {
                // Adapter changed since capture; drop the cached graph.
                self.captured_batched.remove(&key);
            }
        }
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
                if let Err(e) = kiln_tensor::cuda_synchronize_default_stream(idx) {
                    tracing::warn!(batch_size, error = %e, "batched: sync before graph launch failed, falling back to eager");
                    return Ok(None);
                }
            }
            // Step (4): launch.
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
            // Step (5): argmax over `output_logits` → per-row tokens.
            // `output_logits` is already the post-LM-head tensor
            // (`[batch, 1, vocab]`) — the wrapper does final_norm +
            // lm_head + slice_set inside the captured region.
            // #1082: `output_logits` is now a kt-native graph-stable buffer
            // (its device pointer is baked into the captured graph); feed it
            // directly to `greedy_sample_rows` — no candle->kt bridge.
            let tokens = match crate::sampling::greedy_sample_rows(&captured.output_logits) {
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
        let _ = linear_states; // refresh path lands with replay
        match self.try_capture_batched(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            sequence_lengths,
            lora,
        ) {
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
    ) -> Result<Tensor> {
        // #1082 box-102 diagnostic: KILN_FORCE_EAGER_DECODE=1 forces the bs=1
        // EAGER decode forward every step (never captures/replays the graph).
        // If output STILL doubles under this, the doubling is in the bs=1
        // `model_forward_paged` forward itself, NOT the cuda-graph machinery
        // (the graph would just be faithfully replaying a buggy forward). The
        // coherent baseline (graphs off, server) uses a DIFFERENT batched
        // forward, so it would not reveal a bs=1-forward bug.
        if !self.enabled
            || std::env::var("KILN_FORCE_EAGER_DECODE").ok().as_deref() == Some("1")
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

        // #1082 box-102 BUG-B fix: evict the captured bs=1 decode graph when
        // the request changes. The captured graph carries GDN recurrent/conv
        // state in its own buffers (evolved in-place across a request's
        // replays); a NEW request reuses the same graph (the stable-metadata
        // key zeros block_table/seq_len) but the replay path never re-injects
        // the new request's post-prefill state, so request #2+ ran on request
        // #1's leftover state -> deterministic garbage. A fresh
        // `LinearAttentionState` per request gives a new recurrent-state
        // TensorId (stable within a request); on change, evict so the new
        // request re-captures with its own state. Re-capture cost is ~one
        // warm+capture per request; the within-request replays are preserved.
        #[cfg(feature = "cuda")]
        {
            let cur_state_id = linear_state.recurrent_states.first().map(|t| t.id());
            if cur_state_id.is_some() && cur_state_id != self.last_request_state_id {
                if !self.captured.is_empty() {
                    tracing::debug!(
                        "CUDA graph: new request (GDN recurrent-state id changed) — evicting \
                         captured bs=1 graph so it re-captures with the new request's state"
                    );
                    self.captured.clear();
                }
                self.last_request_state_id = cur_state_id;
            }
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
            let requested_key = CudaGraphKey::new(block_table, paged_cache, seq_len);

            // Phase 3: replay if we have a valid captured graph
            if let Some(captured) = self.captured.get(&requested_key) {
                if captured.adapter_gen == self.adapter_generation {
                    // Update position buffer BEFORE graph replay.
                    // The graph's RoPE kernels read from the same GPU pointer,
                    // so updating the data here gives them the correct position.
                    if let Err(e) = Self::update_token_buffer(&captured.token_buffer, token_id) {
                        tracing::warn!("Failed to update token buffer: {e}, falling back to eager");
                        self.captured.remove(&requested_key);
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
                        self.captured.remove(&requested_key);
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
                        self.captured.remove(&requested_key);
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
                    if let (
                        Some(block_table_buffer),
                        Some(seqused_k_buffer),
                        Some(kv_slot_buffer),
                    ) = (
                        captured.block_table_buffer.as_ref(),
                        captured.seqused_k_buffer.as_ref(),
                        captured.kv_slot_buffer.as_ref(),
                    ) {
                        if let Err(e) = Self::update_paged_metadata_buffers(
                            block_table_buffer,
                            seqused_k_buffer,
                            kv_slot_buffer,
                            block_table,
                            paged_cache,
                            seq_len,
                            captured.max_seqlen_k,
                        ) {
                            tracing::warn!(
                                "Failed to update paged graph metadata buffers: {e}, falling back to eager"
                            );
                            self.captured.remove(&requested_key);
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

                    // (#1082 Phase 5) The per-replay input writes above
                    // (update_token/position/rotary/paged_metadata via
                    // cuda_write_host_in_place) land on the kt DEFAULT stream,
                    // but the captured graph launches on its non-default capture
                    // stream — without ordering, replay reads a stale token_id
                    // and the decode diverges into garbage. Sync the default
                    // stream so those writes are visible before launch.
                    if let Some(idx) = captured.token_buffer.device().index() {
                        kiln_tensor::cuda_synchronize_default_stream(idx).context(
                            "sync per-replay input writes before CUDA graph launch",
                        )?;
                    }

                    // #1082 box-102 TOKPROBE (KILN_BOX102_TOKPROBE=1): after the
                    // per-replay update_token_buffer + default-stream sync, read
                    // token_buffer device->host and compare to the intended
                    // token_id. This isolates the BUG2 doubling root cause:
                    //   buffer_holds == [token_id]  => update+sync are correct;
                    //     the captured embedding kernel must be reading a STALE
                    //     copy (baked capture-time token / an arena-copied index
                    //     tensor) rather than this persistent buffer's device ptr.
                    //   buffer_holds != [token_id]  => update_token_buffer /
                    //     cuda_synchronize_default_stream is not landing before
                    //     the readback (ordering/allocation bug in the write path).
                    // Graph replay runs captured KERNELS only, so this Rust-side
                    // readback is the only way to observe the buffer the kernels see.
                    if std::env::var("KILN_BOX102_TOKPROBE").ok().as_deref() == Some("1") {
                        let holds = captured.token_buffer.to_vec::<u32>().ok();
                        eprintln!(
                            "[BOX102-TOKPROBE] replay seq_len={seq_len} intended_token={token_id} buffer_holds={holds:?}"
                        );
                    }

                    // #1082 box-102 SAME-STEP differential (KILN_DEBUG_LAYER_NORMS):
                    // run the EAGER forward on THIS replay step's identical inputs
                    // (snapshot + restore linear_state; the KV write is idempotent
                    // for the same token→slot), dump its per-layer norms, then let
                    // the replay below dump its own (via debug_dump_gdn_state). The
                    // FIRST layer where EAGER ≠ replay is the box-102 root cause —
                    // eager is the correct reference (PASS1==FIRSTLAUNCH proved the
                    // captured compute matches eager on the capture step).
                    if crate::forward::read_layer_norm_debug().is_some() {
                        if let Ok(snap) = linear_state.snapshot() {
                            let elog = Self::eager_forward(
                                backend, token_id, weights, config, paged_cache,
                                block_table, seq_len, linear_state, lora,
                            );
                            if let Some(n) = crate::forward::read_layer_norm_debug() {
                                eprintln!("SAMESTEP EAGER step={seq_len} {n:?}");
                            }
                            // #1082 iter-2: sign-sensitive element-sum at every
                            // recorded slot (0-31 blocks + slot 40 final_norm/
                            // lm_head input). A ROTATED hidden has matching sumsq
                            // (above) but a DIVERGING sum (here) — the first slot
                            // where the sum differs localizes the divergence.
                            if let Some(s) = crate::forward::read_layer_sum_debug() {
                                eprintln!("SAMESTEP EAGER_SUM step={seq_len} {s:?}");
                            }
                            // Dump the eager LOGITS sumsq AND sum — the per-layer
                            // probe stops at the blocks, so this catches a stale
                            // final_norm/lm_head/output_logits on the replay side.
                            if let Ok(el) = &elog {
                                let ess = el.sqr().and_then(|s| s.sum_all()).and_then(|s| s.to_dtype(kiln_tensor::DType::F32)).and_then(|s| s.to_vec::<f32>()).ok();
                                let esum = el.to_dtype(kiln_tensor::DType::F32).and_then(|s| s.sum_all()).and_then(|s| s.to_vec::<f32>()).ok();
                                eprintln!("SAMESTEP EAGER_LOGITS step={seq_len} sumsq={ess:?} sum={esum:?}");
                            }
                            *linear_state = snap;
                        }
                    }

                    match captured.graph.launch() {
                        Ok(()) => {
                            tracing::debug!(
                                max_seqlen_k = requested_key.max_seqlen_k,
                                max_blocks_per_seq = requested_key.max_blocks_per_seq,
                                "CUDA graph replay succeeded"
                            );
                            // #1082 box-102 FIX: the captured graph replayed the
                            // transformer and wrote the PRE-final-norm hidden into
                            // the graph-stable `output_hidden` on its capture
                            // stream. Sync that stream so the write is visible,
                            // then run final_norm + lm_head EAGERLY (off the graph)
                            // to produce this step's logits. The captured lm_head
                            // cublasLt GEMV was the BUG2 source (wrong logits on
                            // replay despite a bit-identical input hidden); the
                            // captured transformer win is preserved.
                            captured.capture_stream.synchronize().map_err(|e| {
                                anyhow::anyhow!(
                                    "box-102 fix: sync capture stream after replay launch: {e}"
                                )
                            })?;
                            let replay_logits = crate::forward::lm_head_from_hidden_eager(
                                backend,
                                &captured.output_hidden,
                                weights,
                                config,
                            )
                            .context("box-102 fix: eager lm_head on replayed hidden")?;
                            if let Some(n) = crate::forward::read_layer_norm_debug() {
                                eprintln!("SAMESTEP REPLAY step={seq_len} {n:?}");
                                // #1082 iter-2: sign-sensitive element-sum on the
                                // REPLAY path. Blocks 0-31 come from the captured
                                // graph; slot 40 from the eager `final_norm` in
                                // `lm_head_from_hidden_eager` above.
                                if let Some(s) = crate::forward::read_layer_sum_debug() {
                                    eprintln!("SAMESTEP REPLAY_SUM step={seq_len} {s:?}");
                                }
                                // Replay LOGITS sumsq AND sum — now computed by the
                                // EAGER lm_head on the replayed hidden, so they
                                // should MATCH SAMESTEP EAGER_LOGITS (the fix).
                                let rss = replay_logits
                                    .sqr()
                                    .and_then(|s| s.sum_all())
                                    .and_then(|s| s.to_dtype(kiln_tensor::DType::F32))
                                    .and_then(|s| s.to_vec::<f32>())
                                    .ok();
                                let rsum = replay_logits
                                    .to_dtype(kiln_tensor::DType::F32)
                                    .and_then(|s| s.sum_all())
                                    .and_then(|s| s.to_vec::<f32>())
                                    .ok();
                                eprintln!("SAMESTEP REPLAY_LOGITS step={seq_len} sumsq={rss:?} sum={rsum:?}");
                            }
                            Self::debug_dump_gdn_state("replay", seq_len, linear_state);
                            return Ok(replay_logits);
                        }
                        Err(e) => {
                            tracing::warn!("CUDA graph replay failed: {e}, falling back to eager");
                            self.captured.remove(&requested_key);
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

            if self.captured.len() >= Self::max_cached_graphs() {
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
            match self.try_capture(
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
                    tracing::warn!("CUDA graph capture failed: {e:#}, using eager decode");
                    self.enabled = false;
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

        #[cfg(not(feature = "cuda"))]
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
    fn rotary_table_values(config: &ModelConfig, position: usize) -> (Vec<f32>, Vec<f32>) {
        let half_rotary = config.rotary_dim() / 2;
        let mut cos = Vec::with_capacity(half_rotary);
        let mut sin = Vec::with_capacity(half_rotary);
        for i in 0..half_rotary {
            let inv_freq = 1.0f32
                / (config
                    .rope_theta
                    .powf(2.0 * i as f64 / config.rotary_dim() as f64) as f32);
            let freq = position as f32 * inv_freq;
            cos.push(freq.cos());
            sin.push(freq.sin());
        }
        (cos, sin)
    }

    #[cfg(feature = "cuda")]
    fn update_rotary_buffers(
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        config: &ModelConfig,
        position: usize,
    ) -> Result<()> {
        let (cos, sin) = Self::rotary_table_values(config, position);
        kiln_tensor::cuda_write_host_in_place(rotary_cos_buffer, cos.as_slice())
            .context("update CUDA graph rotary cos buffer")?;
        kiln_tensor::cuda_write_host_in_place(rotary_sin_buffer, sin.as_slice())
            .context("update CUDA graph rotary sin buffer")?;
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

    /// Rewrite the contents of the batched token buffer in place so the
    /// captured graph picks up the new per-row input tokens on the next
    /// replay.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn update_batched_token_buffer(
        token_buffer: &Tensor,
        token_ids: &[u32],
    ) -> Result<()> {
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
                u32::try_from(p + 1)
                    .context("batched seqused_k buffer: value exceeds u32 range")
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
    #[allow(dead_code)]
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
        let half = config.rotary_dim() / 2;
        let mut cos_flat: Vec<f32> = Vec::with_capacity(start_positions.len() * half);
        let mut sin_flat: Vec<f32> = Vec::with_capacity(start_positions.len() * half);
        for &pos in start_positions {
            let (cos, sin) = Self::rotary_table_values(config, pos);
            cos_flat.extend_from_slice(&cos);
            sin_flat.extend_from_slice(&sin);
        }
        kiln_tensor::cuda_write_host_in_place(rotary_cos_buffer, cos_flat.as_slice())
            .context("update CUDA graph batched rotary cos buffer")?;
        kiln_tensor::cuda_write_host_in_place(rotary_sin_buffer, sin_flat.as_slice())
            .context("update CUDA graph batched rotary sin buffer")?;
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
        // We still need a candle `CudaStream` for the capture-control
        // FFI (`begin_capture` / `end_capture` / `capture_status` /
        // `synchronize`), which only exists on the candle device handle
        // — so bridge the kt device to candle ONCE for that, and pass
        // the resulting `stream` into `with_active_cuda_stream` so every
        // kt op (alloc + the captured forward) lands on it.
        let device = weights.embed_tokens.device();
        let dtype = weights.embed_tokens.dtype();
        // (#1082) kt-native capture stream: the model + captured forward run kt
        // ops on the kt context's default stream, so capture on THAT stream
        // directly (was a kt->candle device bridge just to reach `cuda_stream()`).
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
        let key = CudaGraphKey::new(block_table, paged_cache, seq_len);
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
        // #1082: the kt graph-stable buffers feed the kt-typed
        // `PagedDecodeGraphInputs` directly — no bridges. Build the
        // struct from references into the owned buffers above.
        let graph_inputs = match (
            block_table_buffer.as_ref(),
            seqused_k_buffer.as_ref(),
            kv_slot_buffer.as_ref(),
        ) {
            (Some(block_table), Some(seqused_k), Some(kv_slot)) => Some(PagedDecodeGraphInputs {
                block_table,
                seqused_k,
                kv_slot,
                max_seqlen_k: key.max_seqlen_k,
                rotary_cos: &rotary_cos_buffer,
                rotary_sin: &rotary_sin_buffer,
                attn_out: &paged_decode_outputs[..],
                softmax_lse: &paged_decode_lse[..],
            }),
            _ => None,
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
                graph_inputs.as_ref(),
            )?;
            kiln_tensor::cuda_slice_set_dim0(&output_hidden, &hidden, 0)
                .context("freeze-pointers warm pass: copy hidden into stable output")?;
            Ok::<(), anyhow::Error>(())
        });
        warm_result.context("freeze-pointers warm (Record) pass failed")?;
        // #1082 box-102 differential: Pass-1 (warm/EAGER) per-layer norms for
        // THIS capture-step input. Compared below against the first captured
        // launch's norms (same input) → first diverging layer = the broken
        // captured op. If these PASS1 norms look sane (monotonic, distinct per
        // layer) the per-layer probe is NOT an arena-aliasing artifact.
        if let Some(n) = crate::forward::read_layer_norm_debug() {
            eprintln!("BOX102DIFF PASS1 {n:?}");
        }
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
            kiln_tensor::cuda_synchronize_default_stream(idx)
                .context("CUDA graph capture: sync kt default stream before capture")?;
        }

        // Synchronize all pending work before capture
        stream
            .synchronize()
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
                    graph_inputs.as_ref(),
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

        // End capture — instantiates the graph
        let graph_result = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );

        // Check forward pass success first
        capture_result.context("forward pass failed during graph capture")?;

        // #1082: `graph_inputs` borrows the owned kt buffers
        // (`block_table_buffer`, `rotary_*`, `paged_decode_*`, …); drop it
        // so those buffers can be moved into `CapturedDecodeGraph` below.
        drop(graph_inputs);

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
                stream
                    .synchronize()
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
                // #1082 box-102 differential: first CAPTURED-launch per-layer
                // norms for the SAME capture-step input as PASS1 above (blocks
                // 0-31 from the captured graph; slot 40 from the eager lm_head).
                if let Some(n) = crate::forward::read_layer_norm_debug() {
                    eprintln!("BOX102DIFF FIRSTLAUNCH {n:?}");
                }
                let max_seqlen_k = key.max_seqlen_k;
                self.captured.insert(
                    key,
                    CapturedDecodeGraph {
                        graph,
                        output_hidden,
                        capture_stream: stream.clone(),
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
                        _capture_arena_buffers: arena.borrow_mut().take_retained(),
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
    /// This is the heart of step 6 in the multi-batch sequencing plan
    /// at the top of this file. Buffer allocation uses the
    /// `new_batched_*_buffer` helpers; the forward is invoked via
    /// `model_forward_paged_batched_with_graph_inputs` which threads
    /// the stable device pointers through the bs>1 hidden path.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)]
    fn try_capture_batched(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        lora: Option<&LoraWeights>,
    ) -> Result<Vec<u32>> {
        use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;

        let batch_size = token_ids.len();
        anyhow::ensure!(
            batch_size > 0,
            "try_capture_batched requires a non-empty batch"
        );
        anyhow::ensure!(
            block_tables.len() == batch_size && sequence_lengths.len() == batch_size,
            "try_capture_batched: row count mismatch"
        );
        let max_seq_len = *sequence_lengths.iter().max().expect("non-empty batch");
        let key = CudaBatchedGraphKey::new(batch_size, max_seq_len, paged_cache);

        // #1082: the batched graph-stable buffers are kt-native and
        // allocated directly on the kt device — no candle alloc, no
        // per-buffer candle->kt bridge. (#1082) The capture stream is the kt
        // context's default stream (the one the captured kt forward runs on);
        // capture-control FFI (begin/end_capture, synchronize) drives it directly.
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
        let output_logits = Self::new_batched_output_logits(config, device, dtype, batch_size)?;
        let (paged_decode_outputs, paged_decode_lse) =
            Self::new_batched_paged_decode_outputs(config, device, dtype, batch_size)?;
        let gdn_decode_outputs = Self::new_batched_gdn_decode_outputs(config, device, batch_size)?;
        // Phase 5 #1082 — pre-allocate the lm-head matmul output buffer
        // OUTSIDE the capture window. Installed via
        // `crate::forward::with_lm_head_output_buffer` below so the
        // kt-typed lm_head matmul writes directly into a graph-stable kt
        // `Tensor` (the buffer here). Without this, the captured
        // `slice_set(&logits, …)` would record a memcpy whose source is
        // a transient tensor freed at end-of-capture and dangling on
        // replay, triggering `CUDA_ERROR_ILLEGAL_ADDRESS` at
        // `greedy_sample_rows(captured.output_logits)`. The bs=1 path
        // works in production by luck (allocator determinism at
        // small shapes); the bs>1 path doubles the lm-head output
        // size which churns the pool. This is the structural fix
        // documented in bench-results/cuda-graph-status.md
        // (2026-05-26 entry recommending the
        // `with_lm_head_output_buffer` thread-local approach).
        let lm_head_output_buffer =
            Self::new_lm_head_output_buffer(config, device, dtype, batch_size)?;

        // Capture + forward inside a scope so the `&mut` borrow on
        // `self.batched_state_pool` (taken by `persistent_batched_state`)
        // ends before we mutate `self.captured_batched` below. The kt
        // buffers feed `BatchedPagedDecodeGraphInputs` directly (the
        // struct is already kt-typed) — no bridges.
        let captured: CapturedBatchedDecodeGraph = {
            let persistent_state = self
                .persistent_batched_state(batch_size, config, &device)?
                .context("persistent batched state required for capture")?;
            Self::prepare_gdn_recurrent_state_for_capture(persistent_state)?;
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
                output_logits: &output_logits,
                linear_state: persistent_state,
            };

            // #1082: the kt buffer allocs filled their contents via an H2D
            // on the kt DEFAULT stream; sync it before capture so those
            // fills are visible to the captured forward.
            if let Some(idx) = device.index() {
                kiln_tensor::cuda_synchronize_default_stream(idx).context(
                    "batched CUDA graph capture: sync kt default stream before capture",
                )?;
            }

            // Synchronize before capture — the capture window must not
            // race with any in-flight launches from the prior step.
            stream
                .synchronize()
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
            // #1082: engage the capture stream for the whole batched
            // capture window so every kt op (alloc + the captured
            // forward + the lm-head `slice_set` into `output_logits`)
            // resolves to THIS stream via `active_cuda_stream` — same
            // contract as the bs=1 path.
            let forward_result =
                kiln_tensor::with_active_cuda_stream(stream.clone(), || {
                    crate::forward::with_lm_head_output_buffer(
                        lm_head_output_buffer.clone(),
                        || {
                            crate::forward::model_forward_paged_batched_with_graph_inputs(
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
                        },
                    )
                });

            // NOTE: We deliberately do NOT pass
            // `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` for the
            // bs>1 batched-capture path. Instead we pass the CUDA
            // "no flags" value (0). cudarc's
            // `CUgraphInstantiate_flags_enum` does not expose a `_NONE`
            // variant, so we transmute `0u32` — `cuGraphInstantiate`
            // accepts 0 as "no flags" (this is the official
            // `CUDA_GRAPH_INSTANTIATE_FLAG_NONE` value in the CUDA C
            // headers), and the safe wrapper passes the enum through
            // as `flags as u32 as u64` to the FFI call.
            //
            // Root cause this avoids (see bench-results/cuda-graph-status.md
            // 2026-05-26 section): the captured region contains
            // intermediate tensor allocations (notably the lm-head matmul
            // output) that Candle's matmul pool returns from a host-side
            // allocator. AUTO_FREE_ON_LAUNCH frees device allocations
            // recorded as `cudaMemAllocNode` between replays, and the
            // captured `slice_set` source pointer ends up dangling on the
            // next replay → `CUDA_ERROR_ILLEGAL_ADDRESS` at
            // `greedy_sample_rows(captured.output_logits)`.
            //
            // Trading memory for correctness: each batched-graph bucket
            // keeps its intermediate buffers alive for the lifetime of
            // the captured graph. For a fixed workload with a small
            // number of `(batch_size, max_seqlen_k)` buckets, this cost
            // is acceptable. If memory growth becomes a concern, the
            // structural fix is to pre-allocate the lm-head output
            // buffer outside the capture window via a `matmul_into(dst)`
            // variant (see status doc for design notes).
            //
            // The bs=1 path at line ~1438 still uses
            // AUTO_FREE_ON_LAUNCH unchanged — it has shipped in
            // production for over a year and the matmul output for
            // shape `[1, 1, vocab]` lands at a deterministic pool
            // address in practice.
            let no_flags: cudarc::driver::sys::CUgraphInstantiate_flags =
                unsafe { std::mem::transmute::<u32, _>(0u32) };
            let graph_result = stream.end_capture(no_flags);

            forward_result.context("batched forward failed during graph capture")?;
            let graph = match graph_result {
                Ok(Some(g)) => g,
                Ok(None) => anyhow::bail!("batched graph capture produced no operations"),
                Err(e) => anyhow::bail!("batched end_capture failed: {e}"),
            };

            tracing::info!(
                batch_size,
                max_seqlen_k = key.max_seqlen_k,
                "CUDA graph captured for batched decode"
            );

            // #1082: `graph_inputs` borrows the owned kt buffers
            // (`token_buffer`, `output_logits`, …) and `persistent_state`;
            // drop it so those buffers can be moved into
            // `CapturedBatchedDecodeGraph` below.
            drop(graph_inputs);

            let captured = CapturedBatchedDecodeGraph {
                graph,
                output_logits,
                adapter_gen,
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
                _lm_head_output_buffer: lm_head_output_buffer,
                max_seqlen_k: key.max_seqlen_k,
            };
            captured
        };
        // (#1082 Phase 5) Stream capture only RECORDED the batched forward — it
        // did NOT execute. Launch the instantiated graph once now so
        // output_logits holds real results (and the recurrent/KV state is
        // advanced) before we sample, then sync.
        captured
            .graph
            .launch()
            .context("execute captured batched decode graph (first run)")?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after first batched captured-graph launch: {e}"))?;
        // Argmax + DtoH happens OUTSIDE the capture window — `output_logits`
        // holds the captured forward's final-norm + LM-head result.
        // #1082: `output_logits` is now a kt-native graph-stable buffer;
        // feed it directly to `greedy_sample_rows` — no candle->kt bridge.
        let tokens = crate::sampling::greedy_sample_rows(&captured.output_logits)
            .context("argmax over captured output_logits failed")?;
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
        let out = model_forward_paged(
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
        .context("eager decode forward pass failed");
        if out.is_ok() {
            Self::debug_dump_gdn_state("eager", seq_len, linear_state);
        }
        out
    }

    /// #1082 box-102 BUG2 probe (gated by `KILN_DEBUG_GDN_STATE=1`): dump the
    /// sum-of-squares of layer-0 GDN recurrent + conv state after a decode
    /// step. Compares the eager path (state advances every step) against the
    /// captured-graph replay path: if the replay state norm is FROZEN across
    /// steps, the recurrent state is not surviving replay (the captured graph
    /// updates an arena buffer but the Rust-side `linear_state` swap never runs
    /// on replay) — the leading hypothesis for the token-doubling correctness
    /// bug. Off by default; zero cost on the production path.
    #[cfg(feature = "cuda")]
    fn debug_dump_gdn_state(tag: &str, seq_len: usize, linear_state: &LinearAttentionState) {
        if std::env::var("KILN_DEBUG_GDN_STATE").ok().as_deref() == Some("1") {
            fn sumsq(t: &Tensor) -> f64 {
                match t
                    .to_dtype(kiln_tensor::DType::F32)
                    .and_then(|f| f.to_vec::<f32>())
                {
                    Ok(v) => v.iter().map(|x| (*x as f64) * (*x as f64)).sum(),
                    Err(_) => -1.0,
                }
            }
            let r = linear_state.recurrent_states.first().map(sumsq).unwrap_or(-1.0);
            let c = linear_state.conv_states.first().map(sumsq).unwrap_or(-1.0);
            eprintln!("GDNSTATE [{tag}] step={seq_len} rs0_sumsq={r:.6} conv0_sumsq={c:.6}");
        }
        // #1082 box-102 BUG2 localization: per-layer hidden-state norms
        // (gated by KILN_DEBUG_LAYER_NORMS inside read_layer_norm_debug).
        if let Some(norms) = crate::forward::read_layer_norm_debug() {
            let shown: Vec<String> = norms.iter().take(32).map(|x| format!("{x:.3}")).collect();
            eprintln!("LAYERNORM [{tag}] step={seq_len} [{}]", shown.join(","));
        }
    }

    #[cfg(feature = "cuda")]
    fn max_cached_graphs() -> usize {
        std::env::var("KILN_CUDA_GRAPH_CACHE_MAX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(8)
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
        let pages_per_chunk = 128 / block_size;
        let max_blocks_per_seq = (max_seqlen_k / 128) * pages_per_chunk;
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
        Tensor::from_vec_on(device, vec![slot], vec![1])
            .context("create CUDA graph KV slot buffer")
    }

    /// Allocate a `[batch, max_blocks_per_seq]` u32 padded block-table
    /// buffer covering every row. Reuses the per-row `padded_block_table`
    /// helper and stacks the rows. Used by the batched capture path so
    /// flash-attention reads page metadata from a graph-stable pointer.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    fn new_batched_seqused_k_buffer(
        device: Device,
        start_positions: &[usize],
    ) -> Result<Tensor> {
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
    #[allow(dead_code)]
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
        let (cos, _) = Self::rotary_table_values(config, position);
        let len = cos.len();
        Tensor::from_vec_on(device, cos, vec![1, len])
            .context("create CUDA graph rotary cos buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_rotary_sin_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        let (_, sin) = Self::rotary_table_values(config, position);
        let len = sin.len();
        Tensor::from_vec_on(device, sin, vec![1, len])
            .context("create CUDA graph rotary sin buffer")
    }

    /// #1082 box-102 FIX: graph-stable `[1, 1, hidden_size]` PRE-final-norm
    /// hidden buffer. The captured `HiddenOnly` forward writes here; final_norm
    /// + lm_head run eagerly on it after replay (the lm_head GEMV is not
    /// graph-replay-safe). Dtype matches the model's hidden dtype (bf16).
    #[cfg(feature = "cuda")]
    fn new_output_hidden(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
    ) -> Result<Tensor> {
        Tensor::zeros_on(device, vec![1, 1, config.hidden_size], dtype)
            .context("create CUDA graph output hidden")
    }

    /// `[batch, 1, vocab_size]` lm-head matmul output buffer. Used as
    /// the destination tensor for the captured-graph lm-head matmul
    /// via [`crate::forward::with_lm_head_output_buffer`]. Phase 5
    /// #1082 — see the comment block on
    /// `CapturedBatchedDecodeGraph::_lm_head_output_buffer` for the
    /// full rationale (graph-stable source pointer for the
    /// downstream `slice_set` into `output_logits`).
    #[cfg(feature = "cuda")]
    fn new_lm_head_output_buffer(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            batch > 0,
            "lm-head output buffer requires batch > 0"
        );
        Tensor::zeros_on(device, vec![batch, 1, config.vocab_size], dtype)
            .context("create CUDA graph lm-head output buffer")
    }

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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

    /// `[batch, 1, vocab]` output-logits buffer for batched capture.
    /// (The batched argmax then reduces this to per-row tokens — the
    /// caller of the captured graph either reads the tokens out via
    /// the same in-place mechanism the bs=1 path uses, or runs the LM
    /// head as the post-capture stage on every replay.)
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_output_logits(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "batched output logits require batch > 0");
        Tensor::zeros_on(device, vec![batch, 1, config.vocab_size], dtype)
            .context("create CUDA graph batched output logits")
    }

    /// Per-full-attention-layer paged decode outputs and LSE scratch,
    /// shaped for `[batch, 1, n_heads, head_dim]` and `[batch, n_heads, 1]`.
    /// One element per full-attention layer in the model.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cpu_disables_graphs() {
        let runner = CudaGraphRunner::new(&kiln_tensor::Device::Cpu, true);
        assert!(!runner.is_enabled());
    }

    #[test]
    fn test_new_disabled() {
        let runner = CudaGraphRunner::new(&kiln_tensor::Device::Cpu, false);
        assert!(!runner.is_enabled());
    }

    #[test]
    fn test_invalidate_resets_state() {
        let mut runner = CudaGraphRunner::new(&kiln_tensor::Device::Cpu, false);
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
        let mut runner = CudaGraphRunner::new(&kiln_tensor::Device::Cpu, false);
        runner.invalidate();
        runner.invalidate();
        runner.invalidate();
        assert_eq!(runner.adapter_generation, 3);
    }
}

// SAFETY: CudaGraphRunner is protected by a Mutex in ModelRunner. The inner
// CudaGraph/CudaGraphExec are GPU-side recorded command sequences. Launching a
// graph is thread-safe — the CUDA driver serialises access on the stream.
// The raw pointers (*mut CUgraph_st, *mut CUgraphExec_st) are opaque handles
// to driver-managed objects and are not dereferenced on the CPU side.
unsafe impl Send for CudaGraphRunner {}
unsafe impl Sync for CudaGraphRunner {}
