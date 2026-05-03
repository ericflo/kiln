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

use anyhow::{Context, Result};
use candle_core::Device;
#[cfg(feature = "cuda")]
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use tracing;

use kiln_core::config::ModelConfig;

use crate::backend::BackendRuntime;
use crate::forward::{
    model_forward_paged, model_forward_paged_with_graph_inputs, GpuWeights, LinearAttentionState,
};
use crate::lora_loader::LoraWeights;
use crate::paged_kv_cache::PagedKvCache;

use kiln_core::block::BlockTable;

/// Holds a captured CUDA graph ready for replay.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CudaGraphKey {
    seq_len: usize,
    block_table: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl CudaGraphKey {
    fn new(block_table: &BlockTable, seq_len: usize) -> Self {
        Self {
            seq_len,
            block_table: block_table.blocks.clone(),
        }
    }

    fn block_count(&self) -> usize {
        self.block_table.len()
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
    /// Pre-allocated fused GDN decode recurrent outputs, one per linear layer.
    /// Their device pointers are captured by the graph and must stay alive for
    /// replay.
    _gdn_decode_outputs: Vec<Tensor>,
}

/// Manages CUDA graph lifecycle for decode forward passes.
pub struct CudaGraphRunner {
    /// Whether CUDA graphs are enabled.
    enabled: bool,
    /// Captured graphs keyed by graph-unsafe paged metadata.
    #[cfg(feature = "cuda")]
    captured: HashMap<CudaGraphKey, CapturedDecodeGraph>,
    /// Adapter generation counter; incremented on LoRA swap.
    adapter_generation: u64,
    /// Whether warmup is complete.
    warmup_done: bool,
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
            adapter_generation: 0,
            warmup_done: false,
        }
    }

    /// Invalidate the captured graph (call on LoRA adapter swap).
    pub fn invalidate(&mut self) {
        self.adapter_generation += 1;
        self.warmup_done = false;
        #[cfg(feature = "cuda")]
        {
            if !self.captured.is_empty() {
                tracing::debug!(
                    "CUDA graph invalidated (adapter gen={})",
                    self.adapter_generation
                );
            }
            self.captured.clear();
        }
    }

    /// Whether graphs are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
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
        paged_cache: &mut PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<candle_core::Tensor> {
        if !self.enabled {
            return Self::eager_forward(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            );
        }

        // Phase 1: warmup — run eagerly to prime GPU memory pools
        if !self.warmup_done {
            self.warmup_done = true;
            tracing::debug!("CUDA graph: warmup decode step with graph-shaped inputs");
            #[cfg(feature = "cuda")]
            {
                match Self::eager_forward_with_position_buffer(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
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
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            );
        }

        #[cfg(feature = "cuda")]
        {
            let requested_key = CudaGraphKey::new(block_table, seq_len);

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
                            backend, token_id, weights, config, paged_cache, block_table, seq_len,
                            linear_state, lora,
                        );
                    }
                    if let Err(e) = Self::update_position_buffer(&captured.position_buffer, seq_len) {
                        tracing::warn!("Failed to update position buffer: {e}, falling back to eager");
                        self.captured.remove(&requested_key);
                        return Self::eager_forward(
                            backend, token_id, weights, config, paged_cache, block_table, seq_len,
                            linear_state, lora,
                        );
                    }

                    match captured.graph.launch() {
                        Ok(()) => {
                            return Ok(captured.output_logits.clone());
                        }
                        Err(e) => {
                            tracing::warn!("CUDA graph replay failed: {e}, falling back to eager");
                            self.captured.remove(&requested_key);
                            return Self::eager_forward(
                                backend, token_id, weights, config, paged_cache, block_table,
                                seq_len, linear_state, lora,
                            );
                        }
                    }
                } else {
                    // Adapter changed — drop stale graph
                    self.captured.clear();
                }
            } else if !self.captured.is_empty() {
                tracing::debug!(
                    requested_seq_len = requested_key.seq_len,
                    requested_blocks = requested_key.block_count(),
                    cached_graphs = self.captured.len(),
                    "CUDA graph replay miss: paged decode metadata differs from captured graphs"
                );
            }

            if self.captured.len() >= Self::max_cached_graphs() {
                tracing::warn!(
                    cached_graphs = self.captured.len(),
                    requested_seq_len = requested_key.seq_len,
                    requested_blocks = requested_key.block_count(),
                    "CUDA graph capture skipped: paged metadata cache is full; replay requires graph-stable seq_len/KV/block-table inputs"
                );
                return Self::eager_forward(
                    backend, token_id, weights, config, paged_cache, block_table, seq_len,
                    linear_state, lora,
                );
            }

            // Phase 2: capture
            match self.try_capture(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            ) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!("CUDA graph capture failed: {e:#}, using eager decode");
                    self.enabled = false;
                    return Self::eager_forward(
                        backend, token_id, weights, config, paged_cache, block_table, seq_len,
                        linear_state, lora,
                    );
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        Self::eager_forward(
            backend, token_id, weights, config, paged_cache, block_table, seq_len, linear_state,
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
                dev_ptr,
                value,
                raw_stream,
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
        paged_cache: &mut PagedKvCache,
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
        let output_logits =
            Self::new_output_logits(config, device, weights.embed_tokens.dtype())?;
        let output_logits_for_capture = output_logits.clone();
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
                let key = CudaGraphKey::new(block_table, seq_len);
                self.captured.insert(key, CapturedDecodeGraph {
                    graph,
                    output_logits,
                    adapter_gen: self.adapter_generation,
                    token_buffer,
                    position_buffer,
                    _gdn_decode_outputs: gdn_decode_outputs,
                });
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

    /// Eager (non-graph) paged decode.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &mut PagedKvCache,
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
        paged_cache: &mut PagedKvCache,
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
        runner.invalidate();
        assert!(!runner.warmup_done);
        assert_eq!(runner.adapter_generation, 1);
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
