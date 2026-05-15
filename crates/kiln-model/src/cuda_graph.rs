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
use candle_core::Device;
#[cfg(feature = "cuda")]
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use tracing;

use kiln_core::config::ModelConfig;

use crate::backend::BackendRuntime;
#[cfg(feature = "cuda")]
use crate::forward::PagedDecodeGraphInputs;
#[cfg(feature = "cuda")]
use crate::forward::model_forward_paged_with_graph_inputs;
use crate::forward::{GpuWeights, LinearAttentionState, model_forward_paged};
use crate::lora_loader::LoraWeights;
use crate::paged_kv_cache::PagedKvCache;

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
    fn new(block_table: &BlockTable, paged_cache: &PagedKvCache, seq_len: usize) -> Self {
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
/// stable) bs=1 capture/replay path; `KILN_CUDA_GRAPHS_BATCHED=1`
/// additionally opts into the (in-development) bs>1 capture/replay
/// path. Both must hold for batched graphs to engage; either being
/// off sends the bs>1 caller down the eager batched path. Defaults
/// to off until the batched implementation is fully validated.
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
        paged_cache: &PagedKvCache,
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
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    /// Output logits tensor — its storage is updated in-place during replay.
    output_logits: candle_core::Tensor,
    /// Adapter generation when captured (invalidate on mismatch).
    adapter_gen: u64,
    /// Pre-allocated token-id buffer on GPU (u32, shape [1]).
    /// Updated before each replay so embedding lookup reads the current token
    /// from a graph-stable device pointer.
    token_buffer: Tensor,
    /// Pre-allocated position buffer on GPU (f32, shape [1]).
    /// Updated via cudaMemcpyHtoDAsync before each replay so RoPE sees
    /// the correct position while reading from the same device pointer.
    position_buffer: Tensor,
    /// Pre-allocated padded block table buffer on GPU (u32, shape [1, max_blocks_per_seq]).
    /// Updated before replay so paged attention reads current page metadata from
    /// a graph-stable pointer.
    block_table_buffer: Option<Tensor>,
    /// Pre-allocated actual K/V attention length buffer on GPU (i32, shape [1]).
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
    /// write into capture-time temporary allocations that Candle can free.
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
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    /// `[batch, 1, vocab]` logits — replay writes into this storage.
    output_logits: candle_core::Tensor,
    /// Adapter generation when captured (invalidate on mismatch).
    adapter_gen: u64,
    /// `[batch]` u32 token-id buffer; updated before replay.
    token_buffer: Tensor,
    /// `[batch]` f32 per-row decode position; updated before replay.
    position_buffer: Tensor,
    /// `[batch, max_blocks_per_seq]` u32 padded block table.
    block_table_buffer: Tensor,
    /// `[batch]` i32 per-row K/V length.
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
}

impl CudaGraphRunner {
    /// Create a new graph runner. Enabled only on CUDA devices with the `cuda` feature.
    pub fn new(device: &Device, enabled: bool) -> Self {
        let actually_enabled = enabled && device.is_cuda();
        if actually_enabled {
            tracing::info!("CUDA graphs enabled for decode");
        } else if enabled && !device.is_cuda() {
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
            adapter_generation: 0,
            warmup_done: false,
            #[cfg(feature = "cuda")]
            cache_full_warned: false,
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
        device: &Device,
    ) -> Result<Option<&mut crate::forward::LinearAttentionState>> {
        if !self.enabled {
            return Ok(None);
        }
        if !matches!(device, Device::Cuda(_)) {
            return Ok(None);
        }
        anyhow::ensure!(
            batch_size > 0,
            "persistent batched state requires batch_size > 0"
        );
        if !self.batched_state_pool.contains_key(&batch_size) {
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
        _paged_cache: &PagedKvCache,
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
        paged_cache: &PagedKvCache,
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

        // Phase 1: warmup. The first batched call at each bucket runs
        // eager so Candle / cudarc prime any lazy allocator state
        // before we record the graph. Returning `Ok(None)` here sends
        // the caller back to its eager path, which is exactly what we
        // want: this iteration runs without capture overhead.
        if !self.warmup_done {
            self.warmup_done = true;
            tracing::debug!(
                batch_size,
                max_seqlen_k = key.max_seqlen_k,
                "batched CUDA graph: warmup iteration (eager)"
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
                tracing::warn!(
                    batch_size,
                    max_seqlen_k = key.max_seqlen_k,
                    error = %e,
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
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<candle_core::Tensor> {
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

                    match captured.graph.launch() {
                        Ok(()) => {
                            tracing::debug!(
                                max_seqlen_k = requested_key.max_seqlen_k,
                                max_blocks_per_seq = requested_key.max_blocks_per_seq,
                                "CUDA graph replay succeeded"
                            );
                            return Ok(captured.output_logits.clone());
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
        Self::update_cuda_scalar(token_buffer, &[token_id], "token buffer")
    }

    #[cfg(feature = "cuda")]
    fn update_position_buffer(position_buffer: &Tensor, position: usize) -> Result<()> {
        let pos_f32 = [position as f32];
        Self::update_cuda_scalar(position_buffer, &pos_f32, "position buffer")
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
        Self::update_cuda_scalar(rotary_cos_buffer, cos.as_slice(), "rotary cos buffer")?;
        Self::update_cuda_scalar(rotary_sin_buffer, sin.as_slice(), "rotary sin buffer")?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn update_paged_metadata_buffers(
        block_table_buffer: &Tensor,
        seqused_k_buffer: &Tensor,
        kv_slot_buffer: &Tensor,
        block_table: &BlockTable,
        paged_cache: &PagedKvCache,
        seq_len: usize,
        max_seqlen_k: usize,
    ) -> Result<()> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        Self::update_cuda_scalar(block_table_buffer, padded.as_slice(), "block table buffer")?;
        let attention_len = [(seq_len + 1) as i32];
        Self::update_cuda_scalar(seqused_k_buffer, &attention_len, "seqused_k buffer")?;
        let slot = [block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32];
        Self::update_cuda_scalar(kv_slot_buffer, &slot, "KV slot buffer")?;
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
        Self::update_cuda_scalar(token_buffer, token_ids, "batched token buffer")
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
        Self::update_cuda_scalar(
            position_buffer,
            pos_f32.as_slice(),
            "batched position buffer",
        )
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
        paged_cache: &PagedKvCache,
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
        Self::update_cuda_scalar(
            block_table_buffer,
            block_flat.as_slice(),
            "batched block table buffer",
        )?;
        // seqused_k: per-row (start_pos + 1) as i32.
        let seqused: Vec<i32> = start_positions
            .iter()
            .map(|&p| {
                i32::try_from(p + 1)
                    .context("batched seqused_k buffer: value exceeds i32 range")
            })
            .collect::<Result<Vec<_>>>()?;
        Self::update_cuda_scalar(
            seqused_k_buffer,
            seqused.as_slice(),
            "batched seqused_k buffer",
        )?;
        // KV slots: per-row current write slot.
        let mut slots: Vec<u32> = Vec::with_capacity(block_tables.len());
        for (bt, &pos) in block_tables.iter().zip(start_positions.iter()) {
            slots.push(
                bt.slot_for(pos, paged_cache.block_size())
                    .with_context(|| format!("no slot for decode position {pos}"))?
                    as u32,
            );
        }
        Self::update_cuda_scalar(kv_slot_buffer, slots.as_slice(), "batched KV slot buffer")?;
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
        Self::update_cuda_scalar(
            rotary_cos_buffer,
            cos_flat.as_slice(),
            "batched rotary cos buffer",
        )?;
        Self::update_cuda_scalar(
            rotary_sin_buffer,
            sin_flat.as_slice(),
            "batched rotary sin buffer",
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn update_cuda_scalar<T>(tensor: &Tensor, value: &[T], label: &str) -> Result<()>
    where
        T: candle_core::cuda_backend::cudarc::driver::DeviceRepr
            + candle_core::cuda_backend::CudaDType,
    {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let (storage, _layout) = tensor.storage_and_layout();
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("{label} must be CUDA storage"),
        };

        let stream = cuda_storage.device.cuda_stream();
        let raw_stream = stream.cu_stream();
        let slice = cuda_storage.as_cuda_slice::<T>()?;

        // SAFETY: We write into the scalar buffer before graph replay. No
        // concurrent GPU reads occur between this memcpy and graph launch (the
        // stream is serialized). The device pointer and allocation size are
        // valid because the captured graph owns the tensor.
        unsafe {
            let (dev_ptr, _guard) = slice.device_ptr(&stream);
            candle_core::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                dev_ptr, value, raw_stream,
            )
            .map_err(|e| anyhow::anyhow!("memcpy_htod_async for {label}: {e:?}"))?;
        }

        // Synchronize to ensure the copy completes before graph replay.
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("stream sync after {label} update: {e}"))?;

        Ok(())
    }

    /// Attempt to capture a CUDA graph during a decode forward pass.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_capture(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<candle_core::Tensor> {
        use candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;

        let device = weights.embed_tokens.device();
        let cuda_dev = match device {
            Device::Cuda(d) => d,
            _ => anyhow::bail!("CUDA graphs require a CUDA device"),
        };
        let stream = cuda_dev.cuda_stream();

        // Pre-allocate graph-stable decode tensors BEFORE capture. Their
        // device pointers get baked into the captured graph.
        let token_buffer = Self::new_token_buffer(device, token_id)?;
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        let output_logits = Self::new_output_logits(config, device, weights.embed_tokens.dtype())?;
        let output_logits_for_capture = output_logits.clone();
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
            Self::new_paged_decode_outputs(config, device, weights.embed_tokens.dtype())?
        } else {
            (Vec::new(), Vec::new())
        };
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
                attn_out: &paged_decode_outputs,
                softmax_lse: &paged_decode_lse,
            }),
            _ => None,
        };
        let gdn_decode_outputs = Self::new_gdn_decode_outputs(config, device)?;
        Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;

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
        let logits_result = kiln_gdn_kernel::with_decode_gates_recurrent_outputs(
            gdn_decode_outputs.clone(),
            || {
                let logits = model_forward_paged_with_graph_inputs(
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
                output_logits_for_capture
                    .slice_set(&logits, 0, 0)
                    .context("copy CUDA graph logits into stable output")?;
                Ok::<Tensor, anyhow::Error>(output_logits_for_capture)
            },
        );

        // End capture — instantiates the graph
        let graph_result = stream.end_capture(
            candle_core::cuda_backend::cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );

        // Check forward pass success first
        let logits = logits_result.context("forward pass failed during graph capture")?;

        // Check graph capture success
        match graph_result {
            Ok(Some(graph)) => {
                tracing::info!(
                    "CUDA graph captured for decode ({} layers)",
                    config.num_layers,
                );
                let max_seqlen_k = key.max_seqlen_k;
                self.captured.insert(
                    key,
                    CapturedDecodeGraph {
                        graph,
                        output_logits,
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
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        lora: Option<&LoraWeights>,
    ) -> Result<Vec<u32>> {
        use candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;

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

        let device = weights.embed_tokens.device();
        let cuda_dev = match device {
            Device::Cuda(d) => d,
            _ => anyhow::bail!("CUDA graphs require a CUDA device"),
        };
        let stream = cuda_dev.cuda_stream();
        let adapter_gen = self.adapter_generation;
        let dtype = weights.embed_tokens.dtype();

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

        // Capture + forward inside a scope so the `&mut` borrow on
        // `self.batched_state_pool` (taken by `persistent_batched_state`)
        // ends before we mutate `self.captured_batched` below.
        let captured: CapturedBatchedDecodeGraph = {
            let persistent_state = self
                .persistent_batched_state(batch_size, config, device)?
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
                attn_out: &paged_decode_outputs,
                softmax_lse: &paged_decode_lse,
                output_logits: &output_logits,
                linear_state: persistent_state,
            };

            // Synchronize before capture — the capture window must not
            // race with any in-flight launches from the prior step.
            stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("sync before batched graph capture: {e}"))?;

            stream
                .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| anyhow::anyhow!("begin_capture (batched): {e}"))?;

            // Install pre-allocated GDN recurrent outputs into the
            // GDN kernel's thread-local for the duration of the
            // captured forward. Without this, the GDN decode kernel
            // would `Tensor::zeros(...)` its outputs INSIDE the
            // capture window — those allocations get freed by
            // `AUTO_FREE_ON_LAUNCH` and the graph's recorded
            // pointers go stale on replay. The bs=1 path uses the
            // same mechanism; missing it was the root cause of the
            // observed `CUDA_ERROR_ILLEGAL_ADDRESS` faults at bs>=16
            // in the first wiring attempts.
            let forward_result = kiln_gdn_kernel::with_decode_gates_recurrent_outputs(
                gdn_decode_outputs.clone(),
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
            );

            let graph_result = stream.end_capture(
                candle_core::cuda_backend::cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );

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
                max_seqlen_k: key.max_seqlen_k,
            };
            captured
        };
        // Argmax + DtoH happens OUTSIDE the capture window — `output_logits`
        // holds the captured forward's final-norm + LM-head result.
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
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<candle_core::Tensor> {
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

    #[cfg(feature = "cuda")]
    fn max_cached_graphs() -> usize {
        std::env::var("KILN_CUDA_GRAPH_CACHE_MAX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(8)
    }

    #[cfg(feature = "cuda")]
    fn new_token_buffer(device: &Device, token_id: u32) -> Result<Tensor> {
        Tensor::new(&[token_id], device).context("create CUDA graph token buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_position_buffer(device: &Device, position: usize) -> Result<Tensor> {
        Tensor::new(&[position as f32], device).context("create CUDA graph position buffer")
    }

    /// Allocate a `[batch] u32` token-id buffer with the row 0…batch-1
    /// contents pre-filled from `token_ids`. The captured batched graph
    /// reads its embedding-lookup indices from this device pointer; the
    /// runner refreshes the contents in place via `update_cuda_scalar`
    /// before each replay.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_token_buffer(device: &Device, token_ids: &[u32]) -> Result<Tensor> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "new_batched_token_buffer requires a non-empty batch"
        );
        Tensor::new(token_ids, device).context("create CUDA graph batched token buffer")
    }

    /// Allocate a `[batch] f32` per-row decode-position buffer pre-filled
    /// from `start_positions`. Used by the batched RoPE path under
    /// capture — the captured kernels see the same device pointer and
    /// read whatever the runner writes before each replay.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_position_buffer(device: &Device, start_positions: &[usize]) -> Result<Tensor> {
        anyhow::ensure!(
            !start_positions.is_empty(),
            "new_batched_position_buffer requires a non-empty batch"
        );
        let positions_f32: Vec<f32> = start_positions.iter().map(|&p| p as f32).collect();
        Tensor::new(positions_f32.as_slice(), device)
            .context("create CUDA graph batched position buffer")
    }

    #[cfg(feature = "cuda")]
    fn padded_block_table(
        block_table: &BlockTable,
        paged_cache: &PagedKvCache,
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
        while padded.len() < max_blocks_per_seq {
            let next = padded.last().copied().unwrap_or(0).wrapping_add(1);
            padded.push(next);
        }
        Ok(padded)
    }

    #[cfg(feature = "cuda")]
    fn new_block_table_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCache,
        max_seqlen_k: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        Tensor::new(padded.as_slice(), device)?
            .reshape((1usize, padded.len()))
            .context("create CUDA graph block table buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_seqused_k_buffer(device: &Device, attention_len: usize) -> Result<Tensor> {
        Tensor::new(&[attention_len as i32], device).context("create CUDA graph seqused_k buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_kv_slot_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCache,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let slot = block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32;
        Tensor::new(&[slot], device).context("create CUDA graph KV slot buffer")
    }

    /// Allocate a `[batch, max_blocks_per_seq]` u32 padded block-table
    /// buffer covering every row. Reuses the per-row `padded_block_table`
    /// helper and stacks the rows. Used by the batched capture path so
    /// flash-attention reads page metadata from a graph-stable pointer.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_block_table_buffer(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCache,
        max_seqlen_k: usize,
        device: &Device,
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
        Tensor::new(flat.as_slice(), device)?
            .reshape((block_tables.len(), width))
            .context("create CUDA graph batched block table buffer")
    }

    /// Allocate a `[batch] i32` per-row seqused_k buffer pre-filled from
    /// each row's `start_pos + 1`.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_seqused_k_buffer(
        device: &Device,
        start_positions: &[usize],
    ) -> Result<Tensor> {
        anyhow::ensure!(
            !start_positions.is_empty(),
            "new_batched_seqused_k_buffer requires a non-empty batch"
        );
        let seqused: Vec<i32> = start_positions
            .iter()
            .map(|&p| i32::try_from(p + 1).context("seqused_k exceeds i32 range"))
            .collect::<Result<Vec<_>>>()?;
        Tensor::new(seqused.as_slice(), device)
            .context("create CUDA graph batched seqused_k buffer")
    }

    /// Allocate a `[batch] u32` per-row KV-write-slot buffer for the
    /// current decode step.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_kv_slot_buffer(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCache,
        start_positions: &[usize],
        device: &Device,
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
        Tensor::new(slots.as_slice(), device)
            .context("create CUDA graph batched KV slot buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_rotary_cos_buffer(
        config: &ModelConfig,
        device: &Device,
        position: usize,
    ) -> Result<Tensor> {
        let (cos, _) = Self::rotary_table_values(config, position);
        Tensor::new(cos.as_slice(), device)?
            .reshape((1usize, cos.len()))
            .context("create CUDA graph rotary cos buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_rotary_sin_buffer(
        config: &ModelConfig,
        device: &Device,
        position: usize,
    ) -> Result<Tensor> {
        let (_, sin) = Self::rotary_table_values(config, position);
        Tensor::new(sin.as_slice(), device)?
            .reshape((1usize, sin.len()))
            .context("create CUDA graph rotary sin buffer")
    }

    #[cfg(feature = "cuda")]
    fn new_output_logits(
        config: &ModelConfig,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Tensor> {
        Tensor::zeros((1, 1, config.vocab_size), dtype, device)
            .context("create CUDA graph output logits")
    }

    #[cfg(feature = "cuda")]
    fn new_paged_decode_outputs(
        config: &ModelConfig,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut outputs = Vec::with_capacity(config.num_full_attention_layers);
        let mut lse = Vec::with_capacity(config.num_full_attention_layers);
        for _ in 0..config.num_full_attention_layers {
            outputs.push(
                Tensor::zeros(
                    (1, 1, config.num_attention_heads, config.head_dim),
                    dtype,
                    device,
                )
                .context("create CUDA graph paged decode output")?,
            );
            lse.push(
                Tensor::zeros(
                    (1, config.num_attention_heads, 1),
                    candle_core::DType::F32,
                    device,
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
            if state.dtype() != candle_core::DType::BF16 {
                *state = state
                    .to_dtype(candle_core::DType::BF16)
                    .context("prepare CUDA graph GDN recurrent state")?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn new_gdn_decode_outputs(config: &ModelConfig, device: &Device) -> Result<Vec<Tensor>> {
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let mut outputs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            outputs.push(
                Tensor::zeros(
                    (
                        1,
                        1,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                    ),
                    candle_core::DType::BF16,
                    device,
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
        device: &Device,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "rotary cos buffer requires batch > 0");
        let half = config.rotary_dim() / 2;
        Tensor::zeros((batch, half), candle_core::DType::F32, device)
            .context("create CUDA graph batched rotary cos buffer")
    }

    /// `[batch, rotary_dim/2]` rotary sine buffer initialized at
    /// `position 0`. See `new_batched_rotary_cos_buffer` for the
    /// replay-time refresh semantics.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_rotary_sin_buffer(
        config: &ModelConfig,
        device: &Device,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "rotary sin buffer requires batch > 0");
        let half = config.rotary_dim() / 2;
        Tensor::zeros((batch, half), candle_core::DType::F32, device)
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
        device: &Device,
        dtype: candle_core::DType,
        batch: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(batch > 0, "batched output logits require batch > 0");
        Tensor::zeros((batch, 1, config.vocab_size), dtype, device)
            .context("create CUDA graph batched output logits")
    }

    /// Per-full-attention-layer paged decode outputs and LSE scratch,
    /// shaped for `[batch, 1, n_heads, head_dim]` and `[batch, n_heads, 1]`.
    /// One element per full-attention layer in the model.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn new_batched_paged_decode_outputs(
        config: &ModelConfig,
        device: &Device,
        dtype: candle_core::DType,
        batch: usize,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        anyhow::ensure!(batch > 0, "batched paged decode outputs require batch > 0");
        let mut outputs = Vec::with_capacity(config.num_full_attention_layers);
        let mut lse = Vec::with_capacity(config.num_full_attention_layers);
        for _ in 0..config.num_full_attention_layers {
            outputs.push(
                Tensor::zeros(
                    (batch, 1, config.num_attention_heads, config.head_dim),
                    dtype,
                    device,
                )
                .context("create CUDA graph batched paged decode output")?,
            );
            lse.push(
                Tensor::zeros(
                    (batch, config.num_attention_heads, 1),
                    candle_core::DType::F32,
                    device,
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
        device: &Device,
        batch: usize,
    ) -> Result<Vec<Tensor>> {
        anyhow::ensure!(batch > 0, "batched GDN decode outputs require batch > 0");
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let mut outputs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            outputs.push(
                Tensor::zeros(
                    (
                        batch,
                        1,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                    ),
                    candle_core::DType::BF16,
                    device,
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
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<candle_core::Tensor> {
        let position_buffer = Self::new_position_buffer(weights.embed_tokens.device(), seq_len)?;
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
        let runner = CudaGraphRunner::new(&Device::Cpu, true);
        assert!(!runner.is_enabled());
    }

    #[test]
    fn test_new_disabled() {
        let runner = CudaGraphRunner::new(&Device::Cpu, false);
        assert!(!runner.is_enabled());
    }

    #[test]
    fn test_invalidate_resets_state() {
        let mut runner = CudaGraphRunner::new(&Device::Cpu, false);
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
        let mut runner = CudaGraphRunner::new(&Device::Cpu, false);
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
