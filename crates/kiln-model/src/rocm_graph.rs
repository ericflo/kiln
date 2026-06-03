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
//! Gated on `KILN_ROCM_GRAPHS` (default OFF) and a Rocm device; fully inert
//! otherwise, with graceful eager fallback on any capture/replay failure.

use anyhow::{Context, Result};
use tracing;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;

use crate::backend::BackendRuntime;
use crate::forward::{model_forward_paged, GpuWeights, LinearAttentionState};
use crate::lora_loader::LoraWeights;
use crate::PagedKvCacheKt;

#[cfg(feature = "rocm")]
use crate::forward::{model_forward_paged_hidden_with_graph_inputs, PagedDecodeGraphInputs};
#[cfg(feature = "rocm")]
use std::collections::HashMap;

use kiln_tensor::{Device, Tensor};

/// Whether ROCm HIP-graph decode is requested via `KILN_ROCM_GRAPHS` (default
/// OFF). The sole runtime gate — unlike the CUDA runner there is no separate
/// constructor flag threaded through `new_with_options`.
fn rocm_graphs_env_on() -> bool {
    std::env::var("KILN_ROCM_GRAPHS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on" | "ON"))
        .unwrap_or(false)
}

/// Whether to ATTEMPT capture/replay (vs. eager past warmup). Default ON
/// whenever the runner is enabled — mirrors CUDA, where the single
/// `KILN_CUDA_GRAPHS` flag turns on capture directly (no separate sub-flag).
///
/// This is only ever consulted AFTER the runner is confirmed enabled
/// (`KILN_ROCM_GRAPHS=1`, default off), so out-of-box behavior is unchanged.
/// Capture is fully working: the paged-KV slot write is on-device
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
        // Bucket + size by FA2_KBLOCK_N (=64 for hdim256), matching
        // forward.rs's K_BLOCK_N and padded_block_table exactly — otherwise the
        // captured block-table buffer is sized differently from the table the
        // forward builds, and replay reads OOB.
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let max_seqlen_k = attention_len.div_ceil(kblock_n) * kblock_n;
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

/// A captured HIP graph ready for replay, plus every graph-stable buffer whose
/// device pointer the graph baked in. Mirrors `CapturedDecodeGraph`.
#[cfg(feature = "rocm")]
struct CapturedDecodeGraphRocm {
    /// The source graph — retained because dropping it `hipGraphDestroy`s the
    /// handle; the exec is launched, the graph is kept alive alongside it.
    _graph: kiln_hip::RocmGraph,
    /// The instantiated, launchable graph (AUTO_FREE_ON_LAUNCH).
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
}

/// Runs decode steps through captured HIP graphs when enabled, falling back to
/// eager execution otherwise. ROCm analog of `CudaGraphRunner`.
pub struct RocmGraphRunner {
    enabled: bool,
    adapter_generation: u64,
    warmup_done: bool,
    #[cfg(feature = "rocm")]
    captured: HashMap<RocmGraphKey, CapturedDecodeGraphRocm>,
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
    /// #1082 box-102 BUG-B: request-boundary detection. The captured bs=1 graph
    /// carries GDN recurrent/conv state in its own buffers, evolved in place
    /// across a request's replays; a new request reusing it would run on the
    /// prior request's leftover state. Evict when seq_len is non-contiguous or
    /// the first KV block changed.
    #[cfg(feature = "rocm")]
    last_decode_seq_len: Option<usize>,
    #[cfg(feature = "rocm")]
    last_decode_block0: Option<u32>,
}

impl RocmGraphRunner {
    /// Construct a runner for `device`. Enabled only when `enabled`, the device
    /// is `Device::Rocm`, AND `KILN_ROCM_GRAPHS` is set — otherwise inert.
    pub fn new(device: &Device, enabled: bool) -> Self {
        let is_rocm = matches!(device, Device::Rocm(_));
        let actually_enabled = enabled && is_rocm && rocm_graphs_env_on();
        if actually_enabled {
            tracing::info!("ROCm HIP graphs enabled for decode (KILN_ROCM_GRAPHS)");
        } else if enabled && is_rocm {
            tracing::debug!("ROCm device present but KILN_ROCM_GRAPHS not set — using eager decode");
        }
        Self {
            enabled: actually_enabled,
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
            last_decode_seq_len: None,
            #[cfg(feature = "rocm")]
            last_decode_block0: None,
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
        }
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
    ) -> Result<Tensor> {
        if !self.enabled
            || std::env::var("KILN_FORCE_EAGER_DECODE").ok().as_deref() == Some("1")
        {
            return Self::eager_forward(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            // FP8 paged KV cache is incompatible with HIP graph capture: the
            // per-step on-device quantize allocates its U8 output from the
            // capture arena (a Borrowed, pointer-stable view), and the
            // subsequent `slice_set` into the pool calls `RocmStorage::slice()`
            // on it, which panics on Borrowed storage. CUDA has the same
            // limitation (its FP8 graph-slot write reads the slot index
            // host-side, forcing a sync). Run eager whenever the cache is FP8 —
            // graceful, no capture attempt, parity with CUDA.
            if paged_cache.is_fp8() {
                return Self::eager_forward(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                );
            }

            // Request-boundary eviction (BUG-B): within a bs=1 greedy request
            // seq_len increases by 1 each step and the first KV block is fixed;
            // a new request breaks both. On a boundary, evict so the new request
            // re-captures with its own recurrent state.
            let block0 = block_table.blocks.first().copied();
            let continues = block0.is_some()
                && self.last_decode_seq_len == Some(seq_len.wrapping_sub(1))
                && self.last_decode_block0 == block0;
            if !continues && !self.captured.is_empty() {
                tracing::debug!(
                    seq_len,
                    "ROCm graph: request boundary — evicting captured bs=1 graph"
                );
                self.captured.clear();
            }
            self.last_decode_seq_len = Some(seq_len);
            self.last_decode_block0 = block0;

            // Warmup: first decode step runs eagerly (graph-shaped position
            // buffer) to prime the allocator pools before the first capture.
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
                match Self::eager_forward_with_position_buffer(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                ) {
                    Ok(logits) => return Ok(logits),
                    Err(e) => {
                        tracing::warn!("ROCm graph-shaped warmup failed: {e:#}, plain eager decode");
                    }
                }
                return Self::eager_forward(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                );
            }

            // Capture/replay is gated behind KILN_ROCM_GRAPH_CAPTURE (default
            // off) pending an on-device paged-KV slot write — see
            // rocm_graph_capture_supported(). Without it the runner stays in
            // eager steady-state (transparent vs. plain decode).
            if !rocm_graph_capture_supported() {
                return Self::eager_forward(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                );
            }

            let requested_key = RocmGraphKey::new(block_table, paged_cache, seq_len);

            // Geometry previously found non-capture-safe (host round-trip in its
            // forward) — skip the warm pass + capture attempt and run eager.
            if self.non_capture_safe.contains(&requested_key) {
                return Self::eager_forward(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                );
            }

            // Replay if we have a valid captured graph for this geometry.
            if let Some(captured) = self.captured.get(&requested_key) {
                if captured.adapter_gen == self.adapter_generation {
                    match self.replay(
                        &requested_key, token_id, backend, weights, config, paged_cache,
                        block_table, seq_len,
                    ) {
                        Ok(logits) => {
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(logits);
                        }
                        Err(e) => {
                            tracing::warn!("ROCm graph replay failed: {e:#}, falling back to eager");
                            self.captured.remove(&requested_key);
                            return Self::eager_forward(
                                backend, token_id, weights, config, paged_cache, block_table,
                                seq_len, linear_state, lora,
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
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                );
            }

            // Capture.
            match self.try_capture(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
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
                        backend, token_id, weights, config, paged_cache, block_table, seq_len,
                        linear_state, lora,
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = linear_state;
            Self::eager_forward(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
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
            backend, &[token_id], weights, config, paged_cache, block_table, seq_len,
            Some(linear_state), lora, None,
        )
        .context("eager decode forward pass failed (rocm)")
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
            backend, &[token_id], weights, config, paged_cache, block_table, seq_len,
            Some(linear_state), lora, Some(&position_buffer),
        )
        .context("graph-shaped eager decode forward pass failed (rocm)")
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
        &self,
        key: &RocmGraphKey,
        token_id: u32,
        backend: &dyn BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
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
            config,
            seq_len,
        )?;
        if let (Some(bt), Some(sk), Some(slot)) = (
            captured.block_table_buffer.as_ref(),
            captured.seqused_k_buffer.as_ref(),
            captured.kv_slot_buffer.as_ref(),
        ) {
            Self::update_paged_metadata_buffers(
                bt, sk, slot, block_table, paged_cache, seq_len, captured.max_seqlen_k,
            )?;
        }

        // The per-replay writes above land on the kt DEFAULT stream; the graph
        // launches on its non-default capture stream. Sync the default stream so
        // the writes are visible before launch (else replay reads a stale token).
        if let Some(idx) = captured.token_buffer.device().index() {
            kiln_tensor::rocm_synchronize_default_stream(idx)
                .context("sync per-replay input writes before ROCm graph launch")?;
        }

        captured
            .exec
            .launch(&captured.capture_stream)
            .map_err(|e| anyhow::anyhow!("ROCm graph launch: {e}"))?;
        captured
            .capture_stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync capture stream after replay launch: {e}"))?;

        crate::forward::lm_head_from_hidden_eager(backend, &captured.output_hidden, weights, config)
            .context("eager lm_head on replayed hidden")
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

    fn update_rotary_buffers(
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        config: &ModelConfig,
        position: usize,
    ) -> Result<()> {
        let (cos, sin) = Self::rotary_table_values(config, position);
        kiln_tensor::rocm_write_host_in_place(rotary_cos_buffer, cos.as_slice())
            .context("update ROCm graph rotary cos buffer")?;
        kiln_tensor::rocm_write_host_in_place(rotary_sin_buffer, sin.as_slice())
            .context("update ROCm graph rotary sin buffer")?;
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
        Tensor::from_vec_on(device, padded, vec![1, len]).context("create ROCm graph block table buffer")
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
            .with_context(|| format!("no slot for decode position {seq_len}"))? as u32;
        Tensor::from_vec_on(device, vec![slot], vec![1]).context("create ROCm graph KV slot buffer")
    }

    fn new_rotary_cos_buffer(config: &ModelConfig, device: Device, position: usize) -> Result<Tensor> {
        let (cos, _) = Self::rotary_table_values(config, position);
        let len = cos.len();
        Tensor::from_vec_on(device, cos, vec![1, len]).context("create ROCm graph rotary cos buffer")
    }

    fn new_rotary_sin_buffer(config: &ModelConfig, device: Device, position: usize) -> Result<Tensor> {
        let (_, sin) = Self::rotary_table_values(config, position);
        let len = sin.len();
        Tensor::from_vec_on(device, sin, vec![1, len]).context("create ROCm graph rotary sin buffer")
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
                Tensor::zeros_on(device, vec![1, config.num_attention_heads, 1], kiln_tensor::DType::F32)
                    .context("create ROCm graph paged decode LSE")?,
            );
        }
        Ok((outputs, lse))
    }

    fn prepare_gdn_recurrent_state_for_capture(linear_state: &mut LinearAttentionState) -> Result<()> {
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
                    vec![1, 1, config.linear_num_value_heads, config.linear_value_head_dim],
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
                Some(Self::new_block_table_buffer(block_table, paged_cache, key.max_seqlen_k, device)?),
                Some(Self::new_seqused_k_buffer(device, seq_len + 1)?),
                Some(Self::new_kv_slot_buffer(block_table, paged_cache, seq_len, device)?),
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
            let hidden = model_forward_paged_hidden_with_graph_inputs(
                backend, &[token_id], weights, config, paged_cache, block_table, seq_len,
                Some(linear_state), lora, &token_buffer, &position_buffer, graph_inputs.as_ref(),
            )?;
            kiln_tensor::rocm_slice_set_dim0(&output_hidden, &hidden, 0)
                .context("freeze-pointers warm pass: copy hidden into stable output")?;
            Ok::<(), anyhow::Error>(())
        });
        warm_result.context("freeze-pointers warm (Record) pass failed")?;
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
            return Self::eager_forward(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            );
        }
        // Capture-safe: clear any retry bookkeeping for this geometry.
        self.capture_retry.remove(&key);
        arena.borrow_mut().begin_replay();

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
                    backend, &[token_id], weights, config, paged_cache, block_table, seq_len,
                    Some(linear_state), lora, &token_buffer, &position_buffer, graph_inputs.as_ref(),
                )?;
                kiln_tensor::rocm_slice_set_dim0(&output_hidden, &hidden, 0)
                    .context("ROCm graph: copy kt hidden into stable output_hidden")?;
                Ok::<(), anyhow::Error>(())
            })
        });
        let graph_result = stream.end_capture();
        capture_result.context("forward pass failed during graph capture")?;
        drop(graph_inputs);

        let graph = graph_result.map_err(|e| anyhow::anyhow!("end_capture failed: {e}"))?;
        let exec = graph
            .instantiate()
            .map_err(|e| anyhow::anyhow!("instantiate captured graph: {e}"))?;
        tracing::info!("ROCm HIP graph captured for decode ({} layers)", config.num_layers);

        // Stream capture only RECORDED the forward; launch once now to actually
        // compute this step + advance state, then sync so output_hidden is valid.
        exec.launch(&stream)
            .map_err(|e| anyhow::anyhow!("execute captured decode graph (first run): {e}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after first captured-graph launch: {e}"))?;

        let logits = crate::forward::lm_head_from_hidden_eager(backend, &output_hidden, weights, config)
            .context("eager lm_head on captured hidden (first launch)")?;

        let max_seqlen_k = key.max_seqlen_k;
        let arena_buffers = arena.borrow_mut().take_retained();
        self.captured.insert(
            key,
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
            },
        );
        Ok(logits)
    }

    fn new_token_buffer(device: Device, token_id: u32) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![token_id], vec![1]).context("create ROCm graph token buffer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_off_device() {
        let r = RocmGraphRunner::new(&Device::Cpu, true);
        assert!(!r.is_enabled());
    }

    #[test]
    fn invalidate_bumps_generation_and_resets_warmup() {
        let mut r = RocmGraphRunner::new(&Device::Cpu, true);
        r.warmup_done = true;
        let gen0 = r.adapter_generation;
        r.invalidate();
        assert_eq!(r.adapter_generation, gen0 + 1);
        assert!(!r.warmup_done);
    }
}
