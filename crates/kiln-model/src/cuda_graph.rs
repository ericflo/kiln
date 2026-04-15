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
use candle_core::{Device, Tensor};
use tracing;

use kiln_core::config::ModelConfig;

use crate::forward::{model_forward_paged, GpuWeights, LinearAttentionState};
use crate::lora_loader::LoraWeights;
use crate::paged_kv_cache::PagedKvCache;

use kiln_core::block::BlockTable;

/// Holds a captured CUDA graph ready for replay.
#[cfg(feature = "cuda")]
struct CapturedDecodeGraph {
    /// The instantiated CUDA graph.
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    /// Output logits tensor — its storage is updated in-place during replay.
    output_logits: candle_core::Tensor,
    /// Adapter generation when captured (invalidate on mismatch).
    adapter_gen: u64,
    /// Pre-allocated position buffer on GPU (f32, shape [1]).
    /// Updated via cudaMemcpyHtoDAsync before each replay so RoPE sees
    /// the correct position while reading from the same device pointer.
    position_buffer: Tensor,
}

/// Manages CUDA graph lifecycle for decode forward passes.
pub struct CudaGraphRunner {
    /// Whether CUDA graphs are enabled.
    enabled: bool,
    /// The captured graph.
    #[cfg(feature = "cuda")]
    captured: Option<CapturedDecodeGraph>,
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
            captured: None,
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
            if self.captured.is_some() {
                tracing::debug!(
                    "CUDA graph invalidated (adapter gen={})",
                    self.adapter_generation
                );
            }
            self.captured = None;
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
    pub fn decode_step_paged(
        &mut self,
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
                token_id, weights, config, paged_cache, block_table, seq_len, linear_state, lora,
            );
        }

        // Phase 1: warmup — run eagerly to prime GPU memory pools
        if !self.warmup_done {
            self.warmup_done = true;
            tracing::debug!("CUDA graph: warmup decode step");
            return Self::eager_forward(
                token_id, weights, config, paged_cache, block_table, seq_len, linear_state, lora,
            );
        }

        #[cfg(feature = "cuda")]
        {
            // Phase 3: replay if we have a valid captured graph
            if let Some(ref captured) = self.captured {
                if captured.adapter_gen == self.adapter_generation {
                    // Update position buffer BEFORE graph replay.
                    // The graph's RoPE kernels read from the same GPU pointer,
                    // so updating the data here gives them the correct position.
                    if let Err(e) = Self::update_position_buffer(&captured.position_buffer, seq_len) {
                        tracing::warn!("Failed to update position buffer: {e}, falling back to eager");
                        self.captured = None;
                        self.enabled = false;
                        return Self::eager_forward(
                            token_id, weights, config, paged_cache, block_table, seq_len,
                            linear_state, lora,
                        );
                    }

                    match captured.graph.launch() {
                        Ok(()) => {
                            return Ok(captured.output_logits.clone());
                        }
                        Err(e) => {
                            tracing::warn!("CUDA graph replay failed: {e}, falling back to eager");
                            self.captured = None;
                            self.enabled = false;
                            return Self::eager_forward(
                                token_id, weights, config, paged_cache, block_table, seq_len,
                                linear_state, lora,
                            );
                        }
                    }
                } else {
                    // Adapter changed — drop stale graph
                    self.captured = None;
                }
            }

            // Phase 2: capture
            match self.try_capture(
                token_id, weights, config, paged_cache, block_table, seq_len, linear_state, lora,
            ) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!("CUDA graph capture failed: {e:#}, using eager decode");
                    self.enabled = false;
                    return Self::eager_forward(
                        token_id, weights, config, paged_cache, block_table, seq_len,
                        linear_state, lora,
                    );
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        Self::eager_forward(
            token_id, weights, config, paged_cache, block_table, seq_len, linear_state, lora,
        )
    }

    /// Update the position buffer tensor with a new position value.
    ///
    /// Copies a single f32 value into the existing GPU allocation without
    /// changing the device pointer. This is done outside the CUDA graph so
    /// replayed RoPE kernels read the correct position.
    #[cfg(feature = "cuda")]
    fn update_position_buffer(position_buffer: &Tensor, position: usize) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let pos_f32 = [position as f32];

        let (storage, _layout) = position_buffer.storage_and_layout();
        let cuda_storage = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("position buffer must be CUDA storage"),
        };

        let stream = cuda_storage.device.cuda_stream();
        let raw_stream = stream.cu_stream();
        let slice = cuda_storage.as_cuda_slice::<f32>()?;

        // SAFETY: We write into the position buffer before graph replay.
        // No concurrent GPU reads occur between this memcpy and graph launch
        // (the stream is serialized). The device pointer and allocation size
        // are valid because the position_buffer tensor is alive.
        unsafe {
            let (dev_ptr, _guard) = slice.device_ptr(&stream);
            candle_core::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                dev_ptr,
                &pos_f32,
                raw_stream,
            )
            .map_err(|e| anyhow::anyhow!("memcpy_htod_async for position buffer: {e:?}"))?;
        }

        // Synchronize to ensure the copy completes before graph replay
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("stream sync after position update: {e}"))?;

        Ok(())
    }

    /// Attempt to capture a CUDA graph during a decode forward pass.
    #[cfg(feature = "cuda")]
    fn try_capture(
        &mut self,
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

        // Pre-allocate the position buffer on GPU BEFORE capture.
        // This tensor's device pointer gets baked into the captured graph.
        // On replay, we update its contents via memcpy (outside the graph)
        // so the RoPE kernels see the correct position each step.
        let pos_f32 = seq_len as f32;
        let position_buffer = Tensor::new(&[pos_f32], device)?;

        // Synchronize all pending work before capture
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync before graph capture: {e}"))?;

        // Begin stream capture — all subsequent GPU operations are recorded
        stream
            .begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;

        // Run the forward pass with the pre-allocated position buffer.
        // All kernels are captured, including RoPE which reads from
        // position_buffer's stable GPU address.
        let logits_result = model_forward_paged(
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            Some(&position_buffer),
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
                self.captured = Some(CapturedDecodeGraph {
                    graph,
                    output_logits: logits.clone(),
                    adapter_gen: self.adapter_generation,
                    position_buffer,
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
    fn eager_forward(
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
