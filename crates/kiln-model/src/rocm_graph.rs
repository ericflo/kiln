//! HIP (ROCm) graph capture and replay for decode forward passes — the ROCm
//! twin of [`crate::cuda_graph`] (R.9).
//!
//! During decode each step processes exactly one token with identical tensor
//! shapes, so the kernel sequence can be captured once (`hipStreamBeginCapture`)
//! and replayed (`hipGraphLaunch`) to eliminate per-step host launch overhead.
//! The capture machinery depends on the two foundations landed earlier:
//!   * `kiln_tensor::rocm_write_host_in_place` — refresh per-step inputs through
//!     a graph-stable device pointer (R.9 Phase 1).
//!   * `kiln_tensor::RocmCaptureArena` — freeze every activation pointer the
//!     captured forward touches across capture→replay (R.9 Phase 2).
//!
//! ## Staging
//!
//! This module lands in two stages so the integration is de-risked from the
//! capture internals:
//!   * **Stage A (this commit):** the runner, its `KILN_ROCM_GRAPHS` gate, and
//!     the generate.rs wiring, with capture NOT yet implemented — every decode
//!     step runs eagerly through the same `model_forward_paged` path the
//!     non-graph decode uses. `KILN_ROCM_GRAPHS=1` is therefore a transparent
//!     no-op vs. the default, which the parity test asserts.
//!   * **Stage B:** real `hipStreamBeginCapture` capture + replay, the
//!     graph-stable buffers, the request-boundary eviction, and the off-graph
//!     eager lm_head — mirroring `CudaGraphRunner::try_capture`.
//!
//! The runner falls back to eager execution gracefully on any capture failure
//! and is fully inert (zero behavior change) when `KILN_ROCM_GRAPHS` is unset.

use anyhow::{Context, Result};
use tracing;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;

use crate::backend::BackendRuntime;
use crate::forward::{model_forward_paged, GpuWeights, LinearAttentionState};
use crate::lora_loader::LoraWeights;
use crate::PagedKvCacheKt;

use kiln_tensor::{Device, Tensor};

/// Whether ROCm HIP-graph decode is requested via `KILN_ROCM_GRAPHS`
/// (default OFF). The sole runtime gate for the ROCm graph path — unlike the
/// CUDA runner there is no separate `cuda_graphs` constructor flag threaded
/// through `new_with_options`.
fn rocm_graphs_env_on() -> bool {
    std::env::var("KILN_ROCM_GRAPHS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "on" | "ON"))
        .unwrap_or(false)
}

/// Runs decode steps through captured HIP graphs when enabled, falling back to
/// eager execution otherwise. ROCm analog of [`crate::cuda_graph::CudaGraphRunner`].
///
/// Stage A holds only the eager-path state; Stage B adds the captured-graph map,
/// the `RocmGraphKey`, and the request-boundary eviction fields.
pub struct RocmGraphRunner {
    /// Whether ROCm graphs are enabled (device is Rocm AND `KILN_ROCM_GRAPHS`).
    enabled: bool,
    /// Adapter generation counter; bumped on LoRA swap to invalidate captures.
    adapter_generation: u64,
    /// Whether the one-time warmup decode (priming the allocator pools) ran.
    warmup_done: bool,
}

impl RocmGraphRunner {
    /// Construct a runner for `device`. Enabled only when `enabled`, the device
    /// is `Device::Rocm`, AND `KILN_ROCM_GRAPHS` is set — otherwise inert
    /// (every `decode_step_paged` runs eagerly, identical to the non-graph path).
    pub fn new(device: &Device, enabled: bool) -> Self {
        let is_rocm = matches!(device, Device::Rocm(_));
        let actually_enabled = enabled && is_rocm && rocm_graphs_env_on();
        if actually_enabled {
            tracing::info!("ROCm HIP graphs enabled for decode (KILN_ROCM_GRAPHS)");
        } else if enabled && is_rocm {
            tracing::debug!(
                "ROCm device present but KILN_ROCM_GRAPHS not set — using eager decode"
            );
        }
        Self {
            enabled: actually_enabled,
            adapter_generation: 0,
            warmup_done: false,
        }
    }

    /// Whether captured-graph decode is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Invalidate any captured graphs (LoRA swap changes weight pointers).
    /// Bumps the adapter generation and forces a fresh warmup. Stage B also
    /// clears the captured-graph map here.
    pub fn invalidate(&mut self) {
        self.adapter_generation += 1;
        self.warmup_done = false;
    }

    /// Run one bs=1 paged decode step, returning kt logits `[1, 1, vocab]`.
    ///
    /// Stage A: always eager (capture not yet implemented). The control flow
    /// mirrors the CUDA runner's gates so Stage B can slot capture/replay in
    /// without changing the call sites or the eager-fallback contract.
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

        // Warmup: first decode step runs eagerly (with the graph-shaped position
        // buffer) to prime the allocator pools before the first capture attempt.
        if !self.warmup_done {
            self.warmup_done = true;
            tracing::info!("ROCm graph runner: warmup decode step (KILN_ROCM_GRAPHS active)");
            match Self::eager_forward_with_position_buffer(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            ) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!(
                        "ROCm graph-shaped warmup failed: {e:#}, using plain eager decode"
                    );
                }
            }
            return Self::eager_forward(
                backend, token_id, weights, config, paged_cache, block_table, seq_len,
                linear_state, lora,
            );
        }

        // Stage A: HIP-graph capture/replay is not yet implemented. Run the
        // step eagerly through the same `model_forward_paged` path the
        // non-graph decode uses — byte-identical output, so `KILN_ROCM_GRAPHS`
        // stays a transparent no-op until Stage B lands capture here.
        Self::eager_forward(
            backend, token_id, weights, config, paged_cache, block_table, seq_len,
            linear_state, lora,
        )
    }

    /// Plain eager decode forward — `model_forward_paged` over a single token,
    /// no pre-allocated position buffer. The graph-disabled and capture-failure
    /// fallback path.
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
            None, // no pre-allocated position buffer — created internally
        )
        .context("eager decode forward pass failed (rocm)")
    }

    /// Eager decode forward with a pre-allocated, graph-shaped position buffer —
    /// exercises the same `positions_gpu` input path the captured graph will use,
    /// so the warmup primes the allocator with the capture-shaped allocation
    /// sequence. Mirrors `CudaGraphRunner::eager_forward_with_position_buffer`.
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

    /// Allocate a `[1]` f32 position buffer holding `position` directly on the
    /// kt device. Stage B refreshes its contents in place via
    /// `rocm_write_host_in_place` before each replay.
    fn new_position_buffer(device: Device, position: usize) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![position as f32], vec![1])
            .context("create ROCm graph position buffer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_off_device() {
        // CPU device: never enabled regardless of env.
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
