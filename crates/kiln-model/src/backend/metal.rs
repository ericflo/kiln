//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for GDN and paged-decode.
//!
//! The chokepoint-routed `sdpa` symbol (imported at module level from the
//! kt-side re-export) is an MLX-style fused scaled-dot-product attention
//! kernel with native GQA, BF16, and head dims {32, 64, 72, 80, 96, 128,
//! 256, 512}. For typical transformer head sizes this replaces the vendored
//! CUDA FlashAttention-2 call on Apple Silicon.

use anyhow::{Context, Result};

use super::metal_config::*;
use super::metal_conv1d::{
    metal_causal_conv1d_prefill_bf16_f32_k4, metal_causal_conv1d_update_bf16_f32_k4,
    metal_conv1d_prefill_supports, metal_conv1d_update_supports,
};
use super::metal_core::{kt_metal, kt_metal_alloc};
pub(crate) use super::metal_icb::{MetalPagedDecodeIcbGraph, MetalSingleTokenPagedDecodeIcbGraph};
pub(crate) use super::metal_paged::{
    metal_paged_kv_write_token_major_batch_bf16,
    metal_paged_kv_write_token_major_batch_supports, metal_paged_kv_write_token_major_bf16,
    metal_paged_kv_write_token_major_supports, metal_record_paged_decode_icb_graph,
    metal_record_single_token_paged_decode_icb_graph,
};
use super::metal_paged::{
    metal_paged_attn_decode_contiguous_batch_bf16_d256,
    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256,
    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into,
    metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports,
    metal_paged_attn_decode_contiguous_batch_supports,
    metal_paged_attn_decode_contiguous_bf16_d256, metal_paged_attn_decode_contiguous_supports,
    metal_paged_kv_head_major_read_append_token_major_bf16,
    metal_paged_kv_head_major_read_append_token_major_supports,
    metal_paged_kv_head_major_read_bf16, metal_paged_kv_head_major_read_supports,
};
use super::metal_pipeline::*;
pub(crate) use super::metal_lm_head::{
    metal_lm_head_argmax_bf16, metal_lm_head_argmax_rows_bf16,
    metal_lm_head_argmax_rows_supports, metal_lm_head_argmax_supports, metal_lm_head_bf16,
    metal_lm_head_sample_bf16, metal_lm_head_sample_supports, metal_lm_head_supports,
};
use super::{
    metal_residency, metal_training, BackendRuntime, TrainingCapabilities, TrainingPrecisionPolicy,
};

// Phase 7 #1082: module-level imports for the kt-metal chokepoint types,
// hoisted from ~92 per-function `use` statements so that the chokepoint
// surface in this file is centralized at a single import location. Future
// substrate swaps (e.g. candle → objc2-metal) touch this single import
// block instead of hundreds of scattered fully-qualified references.
use kiln_tensor::metal_types::{MetalCompanion, buffer_o_kt};

#[derive(Debug)]
pub struct MetalBackend {
    /// The kt Metal device this backend dispatches on. (#1082: the
    /// formerly-retained candle `device` field is gone — every trait
    /// method is kt-native, so no candle handle is held.)
    device_kt: kiln_tensor::Device,
    /// Cached at construction to keep env-var reads off per-token support gates.
    disable: MetalKernelDisables,
}

impl MetalBackend {
    pub fn new(device: kiln_tensor::Device) -> Self {
        debug_assert!(
            matches!(device, kiln_tensor::Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self {
            device_kt: device,
            disable: MetalKernelDisables::from_env(),
        }
    }

    pub fn training_capabilities_static() -> TrainingCapabilities {
        metal_training::training_capabilities_static()
    }
}

/// Compile Kiln's custom Metal library and compute pipelines ahead of the
/// first forward pass. Candle kernels still compile lazily inside Candle, but
/// this removes Kiln-owned pipeline setup from the first prewarm/request.
pub fn precompile_custom_kernels(device: &kiln_tensor::Device) -> Result<()> {
    // #1082: kt-native prewarm — derive the companion and drive the pipeline
    // getters through `&dyn MetalPipelineHost` (no candle device).
    let kiln_tensor::Device::Metal(idx) = device else {
        return Ok(());
    };
    let companion = kiln_tensor::primary_metal_companion(*idx)
        .map_err(|e| anyhow::anyhow!("precompile_custom_kernels: companion: {e}"))?;
    let metal_device: &MetalCompanion = &companion;

    metal_shared_library(metal_device)?;
    metal_rms_norm_pipeline(metal_device)?;
    metal_rotary_qk_pipeline(metal_device)?;
    metal_gdn_qk_norm_pipeline(metal_device)?;
    metal_gdn_qk_norm_gqa_pipeline(metal_device)?;
    metal_gdn_decode_qkv_conv_norm_pipeline(metal_device)?;
    metal_gdn_prefill_qkv_conv_split_pipeline(metal_device)?;
    metal_gdn_gates_pipeline(metal_device)?;
    metal_gdn_gates_decay_pipeline(metal_device)?;
    metal_gdn_gates_decay_ab_pipeline(metal_device)?;
    metal_gdn_decode_gates_recurrent_pipeline(metal_device)?;
    metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(metal_device)?;
    metal_gated_rms_norm_pipeline(metal_device)?;
    metal_gdn_in_proj_pipeline(metal_device)?;
    metal_gdn_recurrent_pipeline(metal_device)?;
    metal_gdn_recurrent_prefill_head_last_pipeline(metal_device)?;
    metal_gdn_recurrent_prefill_head_last_decay_pipeline(metal_device)?;
    metal_gdn_forward_substitution_pipeline(metal_device)?;
    metal_gdn_chunk_prep_pipeline(metal_device)?;
    metal_gdn_full_chunk_forward_pipeline(metal_device)?;
    metal_conv1d_prefill_pipeline(metal_device)?;
    metal_conv1d_update_pipeline(metal_device)?;
    metal_lm_head_pipeline(metal_device)?;
    if !metal_lm_head_argmax_disabled() {
        metal_lm_head_argmax_pipeline(metal_device)?;
        if !metal_lm_head_argmax_gpu_reduce_disabled() {
            metal_lm_head_argmax_reduce_pipeline(metal_device)?;
        }
    }
    if !metal_lm_head_argmax_rows_disabled() {
        metal_lm_head_argmax_batch_pipeline(metal_device)?;
        if !metal_lm_head_argmax_gpu_reduce_disabled() {
            metal_lm_head_argmax_reduce_batch_pipeline(metal_device)?;
        }
    }
    if !metal_lm_head_sample_disabled() {
        metal_lm_head_sample_pipeline(metal_device)?;
        metal_lm_head_sample_reduce_pipeline(metal_device)?;
    }
    if !metal_mlp_gate_up_fusion_disabled() {
        metal_mlp_gate_up_pipeline(metal_device)?;
        if !metal_mlp_gate_up_serial_dedicated_disabled() {
            metal_mlp_gate_up_serial_pipeline(metal_device)?;
        }
    }
    metal_mlp_silu_mul_pipeline(metal_device)?;
    if !metal_attn_gate_fusion_disabled() {
        metal_attn_gate_sigmoid_mul_pipeline(metal_device)?;
    }
    if !metal_transposed_coop_gemv_disabled() {
        let default_tile = metal_transposed_coop_gemv_default_tile();
        metal_transposed_coop_gemv_pipeline(metal_device, default_tile)?;
        metal_transposed_coop_gemv_batch_pipeline(metal_device)?;
        if !metal_transposed_coop_gemv_row_quad_tile8_disabled() {
            if !metal_transposed_coop_gemv_row_triple_tile8_disabled() {
                metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(metal_device)?;
            }
            metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(metal_device)?;
        }
        if default_tile != MetalTransposedCoopGemvTile::Tile4 {
            metal_transposed_coop_gemv_pipeline(metal_device, MetalTransposedCoopGemvTile::Tile4)?;
        }
        if !metal_transposed_coop_gemv_tile16_disabled() {
            metal_transposed_coop_gemv_pipeline(metal_device, MetalTransposedCoopGemvTile::Tile16)?;
        }
        if !metal_fused_qkv_proj_disabled() {
            metal_fused_qkv_transposed_coop_gemv_pipeline(metal_device)?;
        }
    }
    if !metal_lora_delta_decode_disabled() {
        metal_lora_hidden_decode_pipeline(metal_device)?;
        metal_lora_add_decode_pipeline(metal_device)?;
    }
    metal_paged_kv_head_major_read_pipeline(metal_device)?;
    metal_paged_kv_head_major_read_append_token_major_pipeline(metal_device)?;
    if !metal_paged_attn_decode_contiguous_disabled() {
        metal_paged_attn_decode_contiguous_pipeline(metal_device)?;
        metal_paged_attn_decode_contiguous_batch_pipeline(metal_device)?;
        metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline(metal_device)?;
    }
    if !metal_paged_kv_write_token_major_disabled() {
        metal_paged_kv_write_token_major_pipeline(metal_device)?;
        metal_paged_kv_write_token_major_batch_pipeline(metal_device)?;
    }
    Ok(())
}

// #1082 DoD-101/102: BackendRuntime decode methods flipped to kt; metal/vulkan impls need matching flip when their builds are restored.
impl BackendRuntime for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn training_precision_policy(&self) -> TrainingPrecisionPolicy {
        metal_training::training_precision_policy()
    }

    // ------------------------------------------------------------------
    // Resident-activation hooks (#1082) — Metal analog of the Vulkan
    // registry. The registry tracks membership only (the kt tensor already
    // owns its GPU buffer); `dispatch_adamw_step` runs a fused on-device
    // AdamW that updates param/m/v in place. Same Ok(true)/Ok(false) and
    // register/has/update/evict/resolve semantics as Vulkan.
    // ------------------------------------------------------------------

    fn supports_resident_activation(&self) -> bool {
        true
    }

    fn register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        metal_residency::register_resident_activation(tensor)
    }

    fn has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool {
        metal_residency::has_resident_activation(tensor)
    }

    fn update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        metal_residency::update_resident_activation(tensor)
    }

    fn evict_resident_activation(&self, tensor: &kiln_tensor::Tensor) {
        metal_residency::evict_resident_activation(tensor);
    }

    fn resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        metal_residency::resolve_resident_activation(tensor, shape, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_adamw_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        first_moment: &kiln_tensor::Tensor,
        second_moment: &kiln_tensor::Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool> {
        // All four operands must be resident: no mixed resident/host update,
        // since that would need a per-call upload and defeat on-device AdamW.
        let all_resident =
            metal_residency::all_registered(&[param, grad, first_moment, second_moment]);
        metal_training::dispatch_adamw_step(
            param,
            grad,
            first_moment,
            second_moment,
            all_resident,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )
    }

    fn supports_linear_decode_sample(&self, top_k: u32) -> bool {
        top_k > 0 && top_k <= METAL_LM_HEAD_SAMPLE_TOP_K_MAX && !metal_lm_head_sample_disabled()
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_decode_sample(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> Result<Option<u32>> {
        if !self.supports_linear_decode_sample(top_k) {
            return Ok(None);
        }
        if !metal_lm_head_sample_supports(x, weight_t, top_k, temperature, history_indices.len()) {
            return Ok(None);
        }
        let greedy =
            kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k);
        let (
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
        ) = if greedy {
            (&[][..], &[][..], 1.0f32, 0.0f32, 0.0f32)
        } else {
            (
                history_indices,
                history_counts,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
            )
        };
        let token = metal_lm_head_sample_bf16(
            x,
            weight_t,
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            temperature.max(f32::MIN_POSITIVE),
            if greedy { 1 } else { top_k },
            top_p,
            min_p,
            seed,
        )
        .context("metal fused linear_decode_sample")?;
        Ok(Some(token))
    }

    fn supports_linear_decode_sample_batch(&self, top_k: &[u32], temperatures: &[f32]) -> bool {
        if top_k.len() != temperatures.len() || top_k.is_empty() || metal_lm_head_sample_disabled()
        {
            return false;
        }
        let mut has_sampled_row = false;
        for (&k, &temp) in top_k.iter().zip(temperatures.iter()) {
            let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
            if greedy {
                continue;
            }
            if !(temp.is_finite() && temp > 0.0 && k > 0 && k <= METAL_LM_HEAD_SAMPLE_TOP_K_MAX) {
                return false;
            }
            has_sampled_row = true;
        }
        has_sampled_row
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_decode_sample_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_rows: &[u32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalties: &[f32],
        presence_penalties: &[f32],
        frequency_penalties: &[f32],
        temperatures: &[f32],
        top_k: &[u32],
        top_p: &[f32],
        min_p: &[f32],
        seeds: &[u64],
    ) -> Result<Option<Vec<u32>>> {
        if !self.supports_linear_decode_sample_batch(top_k, temperatures) {
            return Ok(None);
        }
        let Ok((batch, seq_len, _hidden)) = x.dims3() else {
            return Ok(None);
        };
        if batch == 0 || seq_len != 1 {
            return Ok(None);
        }
        if repetition_penalties.len() != batch
            || presence_penalties.len() != batch
            || frequency_penalties.len() != batch
            || temperatures.len() != batch
            || top_k.len() != batch
            || top_p.len() != batch
            || min_p.len() != batch
            || seeds.len() != batch
            || history_rows.len() != history_indices.len()
            || history_rows.len() != history_counts.len()
        {
            return Ok(None);
        }

        let mut histories = vec![Vec::<(u32, u32)>::new(); batch];
        for ((&row, &idx), &count) in history_rows
            .iter()
            .zip(history_indices.iter())
            .zip(history_counts.iter())
        {
            let row = row as usize;
            if row >= batch {
                return Ok(None);
            }
            histories[row].push((idx, count));
        }
        for row_history in histories.iter_mut() {
            row_history.sort_by_key(|&(idx, _)| idx);
        }

        let mut tokens = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_x = x.narrow(0, row, 1)?.contiguous()?;
            let greedy = kiln_core::sampling::SamplingParams::values_are_effectively_greedy(
                temperatures[row],
                top_k[row],
            );
            let (row_indices, row_counts): (Vec<u32>, Vec<u32>) = if greedy {
                (Vec::new(), Vec::new())
            } else {
                histories[row].iter().copied().unzip()
            };
            let row_temperature = if temperatures[row] == 0.0 {
                1.0
            } else {
                temperatures[row]
            };
            let row_top_k = if greedy { 1 } else { top_k[row] };
            if !metal_lm_head_sample_supports(
                &row_x,
                weight_t,
                row_top_k,
                row_temperature,
                row_indices.len(),
            ) {
                return Ok(None);
            }
            let token = metal_lm_head_sample_bf16(
                &row_x,
                weight_t,
                &row_indices,
                &row_counts,
                if greedy {
                    1.0
                } else {
                    repetition_penalties[row]
                },
                if greedy { 0.0 } else { presence_penalties[row] },
                if greedy {
                    0.0
                } else {
                    frequency_penalties[row]
                },
                row_temperature,
                row_top_k,
                top_p[row],
                min_p[row],
                seeds[row],
            )
            .context("metal fused batched linear_decode_sample row")?;
            tokens.push(token);
        }
        Ok(Some(tokens))
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        std::env::var(DISABLE_METAL_SDPA).is_err()
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        std::env::var(DISABLE_METAL_SDPA).is_err()
    }

    // Note: keep `supports_*` returning true so the planner picks the SDPA
    // path; the per-call gate inside the kernel functions then decides
    // whether the *specific* shape is safe and silently falls back to the
    // naive softmax+matmul path when it isn't.

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn flash_attn_paged_decode_contiguous(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082 forward-flip: trait surface is kt; bridge each kt arg to a
        // candle CPU local (host round-trip, matching this backend's
        // CPU-resident model), then delegate to the unchanged candle helper.
        if !metal_paged_attn_decode_contiguous_supports(
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
        ) {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_bf16_d256(
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
            softmax_scale,
        )
        .context("metal contiguous paged decode attention failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slots: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !metal_paged_attn_decode_contiguous_batch_supports(
            q,
            k_pool,
            v_pool,
            start_slots,
            total_seqlen_k,
        ) {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_batch_bf16_d256(
            q,
            k_pool,
            v_pool,
            start_slots,
            total_seqlen_k,
            softmax_scale,
        )
        .context("metal contiguous paged batch decode attention failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !causal
            || !metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                max_seqlen_k,
                page_block_size,
            )
        {
            return Ok(None);
        }
        let out = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
        )
        .context("metal dyn-seqlen paged batch decode attention failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        graph_outputs: Option<(&kiln_tensor::Tensor, &kiln_tensor::Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !causal
            || !metal_paged_attn_decode_contiguous_batch_dyn_seqlen_supports(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                max_seqlen_k,
                page_block_size,
            )
        {
            return Ok(None);
        }

        if let Some((out, _softmax_lse)) = graph_outputs {
            metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256_into(
                q,
                k_pool,
                v_pool,
                block_table,
                seqused_k,
                out,
                max_seqlen_k,
                page_block_size,
                softmax_scale,
            )
            .context("metal dyn-seqlen paged batch decode attention into graph output failed")?;
            return Ok(Some(out.clone()));
        }

        self.flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
    }

    fn supports_paged_kv_head_major_read(&self) -> bool {
        true
    }

    fn supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
        true
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        !self.disable.conv1d_prefill
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        !self.disable.conv1d_update
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn supports_gdn_gates(&self) -> bool {
        !self.disable.gdn_gates
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        !self.disable.gated_rms_norm
    }

    fn flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if std::env::var(DISABLE_METAL_SDPA).is_ok() {
            return Ok(None);
        }
        // Decline (caller falls back to the portable path) when candle's SDPA
        // can't handle the shape/dtype. Cheaper than surfacing a kernel error
        // from inside the fused path. Guards read the kt arg directly and run
        // BEFORE the candle bridges (#1082 forward-flip).
        if !matches!(
            q.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // Last-axis index via kt-native `rank()` arithmetic so this site no
        // longer names a `candle_core::D::Minus1`-style selector through the
        // chokepoint module (#1082 chokepoint cleanup).
        // `q` here is always at least rank 3 (batch, seq, hidden); the
        // subtraction matches the previous `D::Minus1` semantics.
        let head_dim = q.dim(q.rank() - 1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }
        let q_seq = q.dim(2)?;
        if !metal_sdpa_full_safe_for_q_seq(head_dim, q_seq) {
            return Ok(None);
        }

        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // sdpa(q, k, v, mask, do_causal, scale, softcapping). softcapping=1.0
        // disables it; kiln's prefill path is always causal.
        let out = kiln_tensor::metal_sdpa_last_axis(&q_t, &k_t, &v_t, softmax_scale, causal)
            .context("kt-native metal sdpa (prefill) failed")?;

        let out = out.transpose(1, 2)?.contiguous()?;
        Ok(Some(out))
    }

    fn flash_attn_prefill_head_major(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if std::env::var(DISABLE_METAL_SDPA).is_ok() {
            return Ok(None);
        }
        // Guards read the kt arg directly, BEFORE the candle bridges (#1082).
        if !matches!(
            q.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // Last-axis index via kt-native `rank()` arithmetic; see notes above (#1082 chokepoint).
        let head_dim = q.dim(q.rank() - 1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }
        let q_seq = q.dim(2)?;
        if !metal_sdpa_full_safe_for_q_seq(head_dim, q_seq) {
            return Ok(None);
        }

        let out = kiln_tensor::metal_sdpa_last_axis(q, k, v, softmax_scale, causal)
            .context("kt-native metal sdpa (head-major prefill) failed")?;
        Ok(Some(out))
    }

    /// Gather K/V from the paged pool via `index_select` on the block table,
    /// then call candle's vectorized SDPA (single-query path). The gather
    /// replaces the slow materializing `paged_cache.read` +
    /// naive-softmax+matmul fallback — same result, one fused kernel.
    fn flash_attn_paged_decode(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // Gate on everything SDPA can handle. Pool dtype matches q dtype by
        // construction (both come from the same forward config), so only q
        // needs checking. Guards read the kt arg directly, BEFORE the candle
        // bridges (#1082 forward-flip).
        if !matches!(
            q.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // Last-axis index via kt-native `rank()` arithmetic; see notes above (#1082 chokepoint).
        let head_dim = q.dim(q.rank() - 1)?;
        if !metal_sdpa_supports_head_dim(head_dim) {
            return Ok(None);
        }

        let (batch, q_len, num_heads, _) = q.dims4()?;
        if batch != 1 || q_len != 1 {
            // Multi-sequence paged decode would need a per-sequence gather.
            // Stay on the fallback until the scheduler exercises it.
            return Ok(None);
        }

        let (total_slots, num_kv_heads, _) = k_pool.dims3()?;
        if total_slots % page_block_size != 0 {
            return Ok(None);
        }
        let num_blocks = total_slots / page_block_size;
        let max_blocks_per_seq = block_table.dim(1)?;

        // [num_blocks, block_size, num_kv_heads, head_dim] so index_select on
        // dim 0 gathers a full logical block's slots per physical block id.
        let k_blocks = k_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;
        let v_blocks = v_pool.reshape((num_blocks, page_block_size, num_kv_heads, head_dim))?;

        // The block_table is identical across all 8 full-attention layers in
        // a decode step, but the trait forces us to re-flatten it per call.
        // Threading a pre-flattened handle through the trait would save
        // ~8× redundant flattens per token; defer until the signature can
        // grow a cache parameter.
        let block_ids = block_table.flatten_all()?;

        let k_gathered = k_blocks.index_select(&block_ids, 0)?;
        let v_gathered = v_blocks.index_select(&block_ids, 0)?;

        // [max_blocks_per_seq * block_size, num_kv_heads, head_dim] then
        // narrow to the live KV length.
        let total_gathered = max_blocks_per_seq * page_block_size;
        let k_flat = k_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
        let v_flat = v_gathered.reshape((total_gathered, num_kv_heads, head_dim))?;
        let k_live = k_flat.narrow(0, 0, total_seqlen_k)?;
        let v_live = v_flat.narrow(0, 0, total_seqlen_k)?;

        // SDPA needs [batch, num_heads, seq, head_dim]. Q arrives as
        // [1, 1, num_heads, head_dim]; K/V are [total_seqlen_k, num_kv_heads, head_dim].
        // SDPA handles GQA internally when num_heads % num_kv_heads == 0.
        let q_sdpa = q.transpose(1, 2)?.contiguous()?; // [1, num_heads, 1, head_dim]
        let k_sdpa = k_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // [1, num_kv_heads, total_seqlen_k, head_dim]
        let v_sdpa = v_live.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;

        let out =
            kiln_tensor::metal_sdpa_last_axis(&q_sdpa, &k_sdpa, &v_sdpa, softmax_scale, causal)
                .context("kt-native metal paged sdpa (decode) failed")?;

        // Back to [1, 1, num_heads, head_dim].
        let out = out.transpose(1, 2)?.contiguous()?;
        debug_assert_eq!(out.dims(), &[1, 1, num_heads, head_dim]);
        Ok(Some(out))
    }

    fn paged_kv_head_major_read(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len) {
            return Ok(None);
        }
        let (k_out, v_out) =
            metal_paged_kv_head_major_read_bf16(k_pool, v_pool, start_slot, seq_len)
                .context("metal paged_kv_head_major_read failed")?;
        Ok(Some((k_out, v_out)))
    }

    fn paged_kv_head_major_read_append_token_major(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        prefix_len: usize,
        k_tail: &kiln_tensor::Tensor,
        v_tail: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        if !metal_paged_kv_head_major_read_append_token_major_supports(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        ) {
            return Ok(None);
        }
        let (k_out, v_out) = metal_paged_kv_head_major_read_append_token_major_bf16(
            k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        )
        .context("metal paged_kv_head_major_read_append_token_major failed")?;
        Ok(Some((k_out, v_out)))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly; `conv_state` is mutated
        // in place through its shared UMA buffer (no candle bridge).
        if self.disable.conv1d_prefill
            || !metal_conv1d_prefill_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_prefill_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_prefill kernel failed")?;
        Ok(Some(out))
    }

    fn causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly; `conv_state` is mutated
        // in place through its shared UMA buffer (no candle bridge).
        if self.disable.conv1d_update
            || !metal_conv1d_update_supports(x, weight, conv_state, kernel_size)
        {
            return Ok(None);
        }
        let out = metal_causal_conv1d_update_bf16_f32_k4(x, weight, conv_state, kernel_size)
            .context("metal causal_conv1d_update kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_forward_substitution_supports(a_strict, v_prime, beta)
        {
            return Ok(None);
        }
        let out = match a_strict.dtype() {
            kiln_tensor::DType::BF16 => {
                metal_gdn_forward_substitution_bf16(a_strict, v_prime, beta)
            }
            kiln_tensor::DType::F32 => metal_gdn_forward_substitution_f32(a_strict, v_prime, beta),
            other => anyhow::bail!("unsupported metal gdn_forward_substitution dtype {other:?}"),
        }
        .context("metal gdn_forward_substitution kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_recurrent || !metal_gdn_recurrent_supports(q, k, v, beta, g, state) {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_step kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_chunk_prep(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        {
            return Ok(None);
        }
        let (o0, o1, o2, o3, o4, o5) = metal_gdn_chunk_prep_bf16(g, v, kkt, qkt, ks_entry, q_s)
            .context("metal gdn_chunk_prep kernel failed")?;
        Ok(Some((o0, o1, o2, o3, o4, o5)))
    }

    fn gdn_full_chunk_forward(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_full_chunk_forward_supports(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
            )
        {
            return Ok(None);
        }
        let out =
            metal_gdn_full_chunk_forward_bf16(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
                .context("metal gdn_full_chunk_forward kernel failed")?;
        Ok(Some(out))
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn gdn_full_chunk_forward_head_last_into(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        out: &kiln_tensor::Tensor,
        t_start: usize,
        seq_len: usize,
    ) -> Result<bool> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        // `out` is a caller-owned output buffer written in place by the kernel
        // through its shared UMA buffer; `state` is likewise mutated in place.
        if self.disable.gdn_forward_substitution
            || !metal_gdn_full_chunk_forward_head_last_supports(
                g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
            )
        {
            return Ok(false);
        }
        metal_gdn_full_chunk_forward_head_last_into_bf16(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        )
        .context("metal gdn_full_chunk_forward_head_last_into kernel failed")?;
        Ok(true)
    }

    fn gdn_recurrent_prefill_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_recurrent
            || !metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_prefill_head_last_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_prefill_head_last kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        // The recurrent state is mutated in place through its shared UMA
        // buffer (the &mut state is the same tensor the caller holds).
        if self.disable.gdn_recurrent
            || !metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state)
        {
            return Ok(None);
        }
        let out = metal_gdn_recurrent_prefill_native_head_last_bf16(q, k, v, beta, g, state)
            .context("metal gdn_recurrent_prefill_native_head_last kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_in_proj_decode(
        &self,
        x: &kiln_tensor::Tensor,
        in_proj_qkv_t: &kiln_tensor::Tensor,
        in_proj_z_t: &kiln_tensor::Tensor,
        in_proj_a_t: &kiln_tensor::Tensor,
        in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_in_proj
            || !metal_gdn_in_proj_decode_supports(
                x,
                in_proj_qkv_t,
                in_proj_z_t,
                in_proj_a_t,
                in_proj_b_t,
            )
        {
            return Ok(None);
        }
        let (o0, o1, o2, o3) =
            metal_gdn_in_proj_decode_bf16(x, in_proj_qkv_t, in_proj_z_t, in_proj_a_t, in_proj_b_t)
                .context("metal gdn_in_proj_decode kernel failed")?;
        Ok(Some((o0, o1, o2, o3)))
    }

    fn gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gdn_gates || !metal_gdn_gates_supports(a, b, a_log, dt_bias) {
            return Ok(None);
        }
        let (beta, g) =
            metal_gdn_gates_bf16(a, b, a_log, dt_bias).context("metal gdn_gates kernel failed")?;
        Ok(Some((beta, g)))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if self.disable.gated_rms_norm || !metal_gated_rms_norm_supports(x, z, weight) {
            return Ok(None);
        }
        let out = metal_gated_rms_norm_bf16(x, z, weight, eps as f32)
            .context("metal gated_rms_norm kernel failed")?;
        Ok(Some(out))
    }
}

fn metal_gdn_gates_supports(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if a.shape() != b.shape() {
        return false;
    }
    let Some(&nv) = a.dims().last() else {
        return false;
    };
    if nv == 0 || nv > 256 {
        return false;
    }
    if a_log.dims() != [nv] || dt_bias.dims() != [nv] {
        return false;
    }
    a.elem_count() > 0
}

pub(crate) fn metal_gdn_gates_decay_supports(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> bool {
    !metal_gdn_prefill_decay_recurrent_disabled() && metal_gdn_gates_supports(a, b, a_log, dt_bias)
}

pub(crate) fn metal_gdn_prefill_ab_in_proj_supports(
    x: &kiln_tensor::Tensor,
    in_proj_ab_t: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_prefill_ab_in_proj_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || in_proj_ab_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(in_proj_ab_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !in_proj_ab_t.is_contiguous() || nv == 0 {
        return false;
    }
    let Ok((_batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Some(ab_dim) = nv.checked_mul(2) else {
        return false;
    };
    seq_len > 1 && in_proj_ab_t.dims() == [hidden, ab_dim]
}

pub(crate) fn metal_gdn_gates_decay_ab_supports(
    ab: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_prefill_ab_in_proj_disabled() || metal_gdn_prefill_decay_recurrent_disabled() {
        return false;
    }
    if ab.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(ab.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !ab.is_contiguous() || nv == 0 || nv > 256 {
        return false;
    }
    let Ok((batch, seq_len, channels)) = ab.dims3() else {
        return false;
    };
    let Some(ab_dim) = nv.checked_mul(2) else {
        return false;
    };
    let Some(total) = batch.checked_mul(seq_len).and_then(|n| n.checked_mul(nv)) else {
        return false;
    };
    channels == ab_dim
        && total > 0
        && total <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && a_log.dims() == [nv]
        && dt_bias.dims() == [nv]
}

fn metal_gdn_forward_substitution_supports(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(a_strict.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_prime.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    let dtype = a_strict.dtype();
    if (dtype != kiln_tensor::DType::BF16 && dtype != kiln_tensor::DType::F32)
        || v_prime.dtype() != dtype
        || beta.dtype() != dtype
    {
        return false;
    }
    let Ok((batch, heads, chunk, chunk_cols)) = a_strict.dims4() else {
        return false;
    };
    let Ok((b_v, h_v, c_v, dv)) = v_prime.dims4() else {
        return false;
    };
    let Ok((b_b, h_b, c_b)) = beta.dims3() else {
        return false;
    };

    chunk == chunk_cols
        && (b_v, h_v, c_v) == (batch, heads, chunk)
        && (b_b, h_b, c_b) == (batch, heads, chunk)
        && chunk > 0
        && chunk <= 64
        && dv > 0
        && dv <= 128
}

fn metal_gdn_chunk_prep_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(kkt.device(), kiln_tensor::Device::Metal(_))
        || !matches!(qkt.device(), kiln_tensor::Device::Metal(_))
        || !matches!(ks_entry.device(), kiln_tensor::Device::Metal(_))
        || !matches!(q_s.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if g.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || kkt.dtype() != kiln_tensor::DType::BF16
        || qkt.dtype() != kiln_tensor::DType::BF16
        || ks_entry.dtype() != kiln_tensor::DType::BF16
        || q_s.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    // Full chunks only: this keeps speculative multi-token verification on
    // the already-stable Candle path while accelerating long prompt prefill.
    if chunk != 64 {
        return false;
    }
    let Ok((b_v, h_v, c_v, dv)) = v.dims4() else {
        return false;
    };
    if (b_v, h_v, c_v) != (batch, heads, chunk) || dv == 0 || dv > 128 {
        return false;
    }
    let Ok((b_kkt, h_kkt, c_kkt, c2_kkt)) = kkt.dims4() else {
        return false;
    };
    if (b_kkt, h_kkt, c_kkt, c2_kkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_qkt, h_qkt, c_qkt, c2_qkt)) = qkt.dims4() else {
        return false;
    };
    if (b_qkt, h_qkt, c_qkt, c2_qkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_ks, h_ks, c_ks, dv_ks)) = ks_entry.dims4() else {
        return false;
    };
    if (b_ks, h_ks, c_ks, dv_ks) != (batch, heads, chunk, dv) {
        return false;
    }
    let Ok((b_qs, h_qs, c_qs, dv_qs)) = q_s.dims4() else {
        return false;
    };
    (b_qs, h_qs, c_qs, dv_qs) == (batch, heads, chunk, dv)
}

fn metal_gdn_full_chunk_forward_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if beta.dtype() != kiln_tensor::DType::BF16
        || k_t.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, c_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_kt, h_kt, dk, c_kt)) = k_t.dims4() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_beta, h_beta, c_beta) == (batch, heads, chunk)
        && (b_kt, h_kt, c_kt) == (batch, heads, chunk)
        && (b_state, h_state, dk_state, dv_state) == (batch, heads, dk, dv)
        && chunk == 64
        && dk > 0
        && dk <= 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_full_chunk_forward_strided_inputs_support(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    heads: usize,
) -> bool {
    fn flat_batch_head_ok(stride: &[usize], heads: usize) -> bool {
        stride.len() >= 2
            && stride[1] > 0
            && heads
                .checked_mul(stride[1])
                .is_some_and(|expected| stride[0] == expected)
    }

    fn stride_u32_ok(stride: usize) -> bool {
        stride > 0 && stride <= u32::MAX as usize
    }

    let g_stride = g.layout().strides();
    let v_stride = v.layout().strides();
    let beta_stride = beta.layout().strides();
    let kt_stride = k_t.layout().strides();

    if g_stride.len() != 3 || v_stride.len() != 4 || beta_stride.len() != 3 || kt_stride.len() != 4
    {
        return false;
    }
    if !flat_batch_head_ok(g_stride, heads)
        || !flat_batch_head_ok(v_stride, heads)
        || !flat_batch_head_ok(beta_stride, heads)
        || !flat_batch_head_ok(kt_stride, heads)
    {
        return false;
    }

    // This path only needs to support a time-window narrow. Keep the value
    // dimension contiguous so the per-value lane remains coalesced.
    v_stride[3] == 1
        && [
            g_stride[1],
            g_stride[2],
            v_stride[1],
            v_stride[2],
            v_stride[3],
            beta_stride[1],
            beta_stride[2],
            kt_stride[1],
            kt_stride[2],
            kt_stride[3],
        ]
        .into_iter()
        .all(stride_u32_ok)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn metal_gdn_full_chunk_forward_head_last_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    t_start: usize,
    seq_len: usize,
) -> bool {
    if !metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
        || !matches!(out.device(), kiln_tensor::Device::Metal(_))
        || out.dtype() != kiln_tensor::DType::BF16
        || !out.is_contiguous()
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    out.dims4()
        .is_ok_and(|dims| dims == (batch, seq_len, heads, dv))
        && chunk == 64
        && t_start <= seq_len
        && t_start + chunk <= seq_len
        && metal_gdn_full_chunk_forward_strided_inputs_support(g, v, beta, k_t, heads)
}

fn metal_gdn_recurrent_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, dk)) = q.dims3() else {
        return false;
    };
    let Ok((b_k, h_k, dk_k)) = k.dims3() else {
        return false;
    };
    let Ok((b_v, h_v, dv)) = v.dims3() else {
        return false;
    };
    let Ok((b_b, h_b)) = beta.dims2() else {
        return false;
    };
    let Ok((b_g, h_g)) = g.dims2() else {
        return false;
    };
    let Ok((b_s, h_s, dk_s, dv_s)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, dk_k) == (batch, heads, dk)
        && (b_v, h_v) == (batch, heads)
        && (b_b, h_b) == (batch, heads)
        && (b_g, h_g) == (batch, heads)
        && (b_s, h_s, dk_s, dv_s) == (batch, heads, dk, dv)
        && dk <= 256
        && dv <= 1024
}

const METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN: usize = 2048;

fn metal_gdn_recurrent_prefill_head_last_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, q_heads, seq_len, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, h_k, t_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, v_heads, t_v, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, t_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, h_g, t_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, t_k, dk_k) == (batch, q_heads, seq_len, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, h_beta, t_beta) == (batch, v_heads, seq_len)
        && (b_g, h_g, t_g) == (batch, v_heads, seq_len)
        && (b_state, h_state, dk_state, dv_state) == (batch, v_heads, dk, dv)
        && v_heads >= q_heads
        && v_heads % q_heads == 0
        && seq_len > 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_recurrent_prefill_native_head_last_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, t_beta, h_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, t_g, h_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, t_beta, h_beta) == (batch, seq_len, value_heads)
        && (b_g, t_g, h_g) == (batch, seq_len, value_heads)
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && seq_len >= 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

pub(crate) fn metal_gdn_recurrent_prefill_native_head_last_decay_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    !metal_gdn_prefill_decay_recurrent_disabled()
        && metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, decay, state)
}

pub(crate) fn metal_gdn_decode_gates_recurrent_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_disabled()
        || !matches!(q.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_a, t_a, h_a)) = a.dims3() else {
        return false;
    };
    let Ok((b_b, t_b, h_b)) = b.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    let Some(batch_heads) = batch.checked_mul(value_heads) else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_a, t_a, h_a) == (batch, seq_len, value_heads)
        && (b_b, t_b, h_b) == (batch, seq_len, value_heads)
        && a_log.dims() == [value_heads]
        && dt_bias.dims() == [value_heads]
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && q_heads > 0
        && value_heads > q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && batch_heads <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && value_heads <= u32::MAX as usize
        && state.is_contiguous()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_rmsnorm_disabled() {
        return false;
    }
    if !metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state) {
        return false;
    }
    metal_gated_rms_norm_supports(v, z, weight)
}

fn metal_gated_rms_norm_supports(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(z.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    // x/z are BF16 activations. The norm weight follows the model dtype, which
    // for Qwen3.5 GDN is BF16 (same as the CUDA `gdn_gated_rms_norm_supports_kt`
    // contract). Accept F32 too for callers that pre-promote — the kernel casts
    // the weight to F32 internally either way.
    if x.dtype() != kiln_tensor::DType::BF16
        || z.dtype() != kiln_tensor::DType::BF16
        || !matches!(
            weight.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        )
    {
        return false;
    }
    let Ok((batch, seq_len, heads, hidden)) = x.dims4() else {
        return false;
    };
    let Ok((z_batch, z_seq_len, z_heads, z_hidden)) = z.dims4() else {
        return false;
    };
    if (z_batch, z_seq_len, z_heads, z_hidden) != (batch, seq_len, heads, hidden) {
        return false;
    }
    weight.dims() == [hidden] && hidden <= 1024
}

pub(crate) fn metal_rms_norm_supports(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if metal_rms_norm_disabled() {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    let Some(hidden) = x.dims().last().copied() else {
        return false;
    };
    x.rank() >= 1 && weight.dims() == [hidden] && hidden <= 8192
}

pub(crate) fn metal_gdn_qk_norm_supports(q: &kiln_tensor::Tensor, k: &kiln_tensor::Tensor) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::F32 || k.dtype() != kiln_tensor::DType::F32 {
        return false;
    }
    let Some(hidden) = q.dims().last().copied() else {
        return false;
    };
    q.rank() >= 1 && q.dims() == k.dims() && hidden <= 8192
}

pub(crate) fn metal_gdn_qk_norm_gqa_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::F32 || k.dtype() != kiln_tensor::DType::F32 {
        return false;
    }
    let Ok((_, _, nk, hidden)) = q.dims4() else {
        return false;
    };
    q.dims() == k.dims()
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && hidden <= 8192
        && nv <= u32::MAX as usize
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_supports(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_qkv_conv_norm_disabled() || metal_gdn_qk_norm_disabled() {
        return false;
    }
    if kernel_size != 4 || dk != 128 || dv != 128 {
        return false;
    }
    if !matches!(mixed_qkv.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
        || !matches!(conv_state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if mixed_qkv.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    let Some(rows) = nk
        .checked_add(nk)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_mul(batch))
    else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && channels == expected_channels
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && weight_ok
        && conv_state
            .dims3()
            .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
        && channels <= u32::MAX as usize
        && nk <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && rows <= u32::MAX as usize
}

pub(crate) fn metal_rotary_embedding_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(cos.device(), kiln_tensor::Device::Metal(_))
        || !matches!(sin.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || cos.dtype() != kiln_tensor::DType::F32
        || sin.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    if !q.is_contiguous() || !k.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, q_heads, q_head_dim)) = q.dims4() else {
        return false;
    };
    let Ok(k_dims) = k.dims4() else {
        return false;
    };
    let half_rotary = rotary_dim / 2;
    let table_batch_stride = metal_rotary_table_batch_stride(cos, sin, batch, seq_len, half_rotary);
    let Some(total_q) = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(q_heads))
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    let Some(total_k) = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(k_dims.2))
        .and_then(|n| n.checked_mul(head_dim))
    else {
        return false;
    };
    k_dims.0 == batch
        && k_dims.1 == seq_len
        && k_dims.3 == head_dim
        && q_head_dim == head_dim
        && rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim % 2 == 0
        && table_batch_stride.is_some()
        && batch <= u32::MAX as usize
        && seq_len <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && k_dims.2 <= u32::MAX as usize
        && head_dim <= u32::MAX as usize
        && rotary_dim <= u32::MAX as usize
        && total_q <= u32::MAX as usize
        && total_k <= u32::MAX as usize
        && total_q <= (u32::MAX as usize).saturating_sub(total_k)
}

fn metal_rotary_table_batch_stride(
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    batch: usize,
    seq_len: usize,
    half_rotary: usize,
) -> Option<usize> {
    if cos.dims() != sin.dims() {
        return None;
    }
    match cos.dims() {
        [t, r] if (*t, *r) == (seq_len, half_rotary) => Some(0),
        [b, t, r] if (*b, *t, *r) == (batch, seq_len, half_rotary) => Some(seq_len),
        _ => None,
    }
}



















































pub(crate) fn metal_mlp_gate_up_supports(
    x: &kiln_tensor::Tensor,
    gate_t: &kiln_tensor::Tensor,
    up_t: &kiln_tensor::Tensor,
) -> bool {
    if x.dtype() != kiln_tensor::DType::BF16
        || gate_t.dtype() != kiln_tensor::DType::BF16
        || up_t.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(gate_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(up_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !gate_t.is_contiguous() || !up_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((gate_hidden, intermediate)) = gate_t.dims2() else {
        return false;
    };
    let Ok((up_hidden, up_intermediate)) = up_t.dims2() else {
        return false;
    };
    let Some(rows) = batch.checked_mul(seq_len) else {
        return false;
    };
    let Some(total) = rows.checked_mul(intermediate) else {
        return false;
    };

    rows > 0
        && seq_len == 1
        && hidden == gate_hidden
        && hidden == up_hidden
        && intermediate == up_intermediate
        && hidden <= u32::MAX as usize
        && intermediate <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_mlp_gate_up_bf16(
    x: &kiln_tensor::Tensor,
    gate_t: &kiln_tensor::Tensor,
    up_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_mlp_gate_up_supports(x, gate_t, up_t),
        "metal mlp gate/up supports only BF16 [B,1,H] x [H,I] on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let (_, intermediate) = gate_t.dims2()?;
    let rows = batch * seq_len;
    let row_group_size = if rows == 3
        && !metal_mlp_gate_up_row_pair_disabled()
        && !metal_mlp_gate_up_row_triple_disabled()
    {
        3
    } else if rows >= 5
        && !metal_mlp_gate_up_row_pair_disabled()
        && !metal_mlp_gate_up_row_quad_disabled()
    {
        4
    } else if rows > 1 && !metal_mlp_gate_up_row_pair_disabled() {
        2
    } else {
        1
    };
    let row_groups = rows.div_ceil(row_group_size);
    let total = row_groups * intermediate.div_ceil(2);

    let x_metal = kt_metal(x)?;
    // The kernel writes every row/intermediate element.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, intermediate],
    )?;

    let companion = x_metal.companion()?;
    let encoder = companion.command_encoder()?;

    {
        let gate_metal = kt_metal(gate_t)?;
        let up_metal = kt_metal(up_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 mlp-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let gate_buf = buffer_o_kt(
            gate_metal.buffer().as_ref(),
            gate_t.layout(),
            gate_t.dtype(),
        );
        let up_buf = buffer_o_kt(up_metal.buffer().as_ref(), up_t.layout(), up_t.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let hidden_u32 = hidden as u32;
        let intermediate_u32 = intermediate as u32;

        let serial_vector_safe = rows == 1
            && !metal_mlp_gate_up_serial_vector_load_disabled()
            && intermediate % 2 == 0
            && gate_buf.offset_in_bytes % 4 == 0
            && up_buf.offset_in_bytes % 4 == 0;
        let serial_dedicated = serial_vector_safe && !metal_mlp_gate_up_serial_dedicated_disabled();
        if serial_dedicated {
            let pipeline = metal_mlp_gate_up_serial_pipeline(&*companion)?;
            encoder.set_label("kiln_mlp_gate_up_serial_bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(4, &hidden_u32);
            encoder.set_bytes(5, &intermediate_u32);
        } else {
            let pipeline = metal_mlp_gate_up_pipeline(&*companion)?;
            encoder.set_label("kiln_mlp_gate_up_bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            let rows_u32 = rows as u32;
            let row_pair_mode_u32 = if row_group_size == 1 {
                if serial_vector_safe { 6 } else { 0 }
            } else if row_group_size == 3
                && intermediate % 2 == 0
                && gate_buf.offset_in_bytes % 4 == 0
                && up_buf.offset_in_bytes % 4 == 0
            {
                7
            } else if row_group_size == 4
                && !metal_mlp_gate_up_row_quad_vector_load_disabled()
                && intermediate % 2 == 0
                && gate_buf.offset_in_bytes % 4 == 0
                && up_buf.offset_in_bytes % 4 == 0
            {
                5
            } else {
                row_group_size as u32
            };
            encoder.set_bytes(4, &rows_u32);
            encoder.set_bytes(5, &hidden_u32);
            encoder.set_bytes(6, &intermediate_u32);
            encoder.set_bytes(7, &row_pair_mode_u32);
        }

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_mlp_silu_mul_supports(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> bool {
    if metal_mlp_silu_mul_disabled() {
        return false;
    }
    if gate.dtype() != kiln_tensor::DType::BF16 || up.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(gate.device(), kiln_tensor::Device::Metal(_))
        || !matches!(up.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !gate.is_contiguous() || !up.is_contiguous() || gate.shape() != up.shape() {
        return false;
    }
    gate.elem_count() > 0 && gate.elem_count() <= u32::MAX as usize
}

pub(crate) fn metal_mlp_silu_mul_bf16(
    gate: &kiln_tensor::Tensor,
    up: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_mlp_silu_mul_supports(gate, up),
        "metal mlp silu*mul supports only matching contiguous BF16 Metal tensors"
    );
    let total = gate.elem_count();
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let gate_metal = kt_metal(&gate)?;
    let out = kt_metal_alloc(gate_metal, kiln_tensor::DType::BF16, gate.dims())?;

    let companion = gate_metal.companion()?;
    let pipeline = metal_mlp_silu_mul_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_mlp_silu_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let up_metal = kt_metal(&up)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 mlp-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let gate_buf = buffer_o_kt(gate_metal.buffer().as_ref(), gate.layout(), gate.dtype());
        let up_buf = buffer_o_kt(up_metal.buffer().as_ref(), up.layout(), up.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(up_buf.buffer), up_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let total_u32 = total as u32;
        encoder.set_bytes(3, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_attn_gate_sigmoid_mul_supports(
    x: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> bool {
    if metal_attn_gate_fusion_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || gate.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(gate.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !gate.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((gate_batch, gate_seq_len, gate_hidden)) = gate.dims3() else {
        return false;
    };
    let Some(rows) = batch.checked_mul(seq_len) else {
        return false;
    };
    let Some(total) = rows.checked_mul(hidden) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && gate_batch == batch
        && gate_seq_len == seq_len
        && gate_hidden == hidden
        && total <= u32::MAX as usize
}

pub(crate) fn metal_attn_gate_sigmoid_mul_bf16(
    x: &kiln_tensor::Tensor,
    gate: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_attn_gate_sigmoid_mul_supports(x, gate),
        "metal attn gate sigmoid/mul supports only BF16 [B,1,H] tensors on Metal"
    );
    let (batch, seq_len, hidden) = x.dims3()?;
    let total = batch * seq_len * hidden;

    // The kernel writes every hidden element exactly once.
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, seq_len, hidden])?;

    let companion = x_metal.companion()?;
    let pipeline = metal_attn_gate_sigmoid_mul_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_attn_gate_sigmoid_mul_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let gate_metal = kt_metal(&gate)?;
        let out_metal = kt_metal(&out)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let gate_buf = buffer_o_kt(gate_metal.buffer().as_ref(), gate.layout(), gate.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(gate_buf.buffer), gate_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let total_u32 = total as u32;
        encoder.set_bytes(3, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_transposed_coop_gemv_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((weight_input_dim, output_dim)) = weight_t.dims2() else {
        return false;
    };

    batch == 1
        && seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim == weight_input_dim
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
}

pub(crate) fn metal_transposed_coop_gemv_decode_batch_supports(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_transposed_coop_gemv_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || weight_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !weight_t.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((weight_input_dim, output_dim)) = weight_t.dims2() else {
        return false;
    };
    let Some(total) = batch.checked_mul(output_dim) else {
        return false;
    };

    batch > 1
        && seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim == weight_input_dim
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
        && batch <= u32::MAX as usize
        && total <= u32::MAX as usize
}

pub(crate) fn metal_transposed_coop_gemv_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    if metal_transposed_coop_gemv_decode_batch_supports(x, weight_t) {
        return metal_transposed_coop_gemv_batch_bf16(x, weight_t);
    }

    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;
    metal_transposed_coop_gemv_bf16_with_tile(
        x,
        weight_t,
        metal_transposed_coop_gemv_select_tile(input_dim, output_dim),
    )
}

fn metal_transposed_coop_gemv_bf16_with_tile(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
    tile: MetalTransposedCoopGemvTile,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_supports(x, weight_t),
        "metal transposed coop GEMV supports only BF16 [1,1,K] x [K,N] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;

    let x_metal = kt_metal(&x)?;
    // The kernel writes every output channel exactly once.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = metal_transposed_coop_gemv_pipeline(&*companion, tile)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label(tile.label());
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);

        let cols_per_threadgroup = tile.tile_cols() * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_transposed_coop_gemv_batch_bf16(
    x: &kiln_tensor::Tensor,
    weight_t: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_transposed_coop_gemv_decode_batch_supports(x, weight_t),
        "metal batch transposed coop GEMV supports only BF16 [B,1,K] x [K,N] with B > 1 on Metal"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, output_dim) = weight_t.dims2()?;
    let row_grouping_enabled = batch > 1 && !metal_transposed_coop_gemv_row_pair_disabled();
    let row_quad_enabled =
        row_grouping_enabled && batch >= 3 && !metal_transposed_coop_gemv_row_quad_disabled();
    let row_triple_tile8_enabled = row_quad_enabled
        && batch == 3
        && !metal_transposed_coop_gemv_row_quad_tile8_disabled()
        && !metal_transposed_coop_gemv_row_triple_tile8_disabled();
    let row_quad_tile8_enabled = row_quad_enabled
        && !row_triple_tile8_enabled
        && !metal_transposed_coop_gemv_row_quad_tile8_disabled();
    let row_group_size = if row_triple_tile8_enabled {
        3usize
    } else if row_quad_enabled {
        4usize
    } else if row_grouping_enabled {
        2usize
    } else {
        1usize
    };
    let row_groups = batch.div_ceil(row_group_size);

    let x_metal = kt_metal(&x)?;
    // The kernel writes every batch/output channel exactly once.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let pipeline = if row_triple_tile8_enabled {
        metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(&*companion)?
    } else if row_quad_tile8_enabled {
        metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(&*companion)?
    } else {
        metal_transposed_coop_gemv_batch_pipeline(&*companion)?
    };
    let encoder = companion.command_encoder()?;
    encoder.set_label(if row_triple_tile8_enabled {
        "kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16"
    } else if row_quad_tile8_enabled {
        "kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16"
    } else {
        "kiln_transposed_coop_gemv8_batch_bf16"
    });
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight_t)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(
            w_metal.buffer().as_ref(),
            weight_t.layout(),
            weight_t.dtype(),
        );
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let output_dim_u32 = output_dim as u32;
        encoder.set_bytes(3, &input_dim_u32);
        encoder.set_bytes(4, &output_dim_u32);
        if row_quad_tile8_enabled {
            let batch_u32 = batch as u32;
            encoder.set_bytes(5, &batch_u32);
        } else if !row_triple_tile8_enabled {
            let row_pair_mode_u32 = if row_group_size > 1 { batch as u32 } else { 0 };
            let row_group_size_u32 = row_group_size as u32;
            encoder.set_bytes(5, &row_pair_mode_u32);
            encoder.set_bytes(6, &row_group_size_u32);
        }

        let tile_cols = if row_triple_tile8_enabled || row_quad_tile8_enabled {
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS
        } else if row_quad_enabled {
            METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS
        } else {
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS
        };
        let cols_per_threadgroup = tile_cols * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: output_dim.div_ceil(cols_per_threadgroup),
            height: row_groups,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_fused_qkv_transposed_coop_gemv_supports(
    x: &kiln_tensor::Tensor,
    q_t: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    v_t: &kiln_tensor::Tensor,
) -> bool {
    if metal_fused_qkv_proj_disabled() {
        return false;
    }
    if !metal_transposed_coop_gemv_supports(x, q_t)
        || !metal_transposed_coop_gemv_supports(x, k_t)
        || !metal_transposed_coop_gemv_supports(x, v_t)
    {
        return false;
    }

    let Ok((_, _, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((q_input_dim, _)) = q_t.dims2() else {
        return false;
    };
    let Ok((k_input_dim, _)) = k_t.dims2() else {
        return false;
    };
    let Ok((v_input_dim, _)) = v_t.dims2() else {
        return false;
    };
    input_dim == q_input_dim && input_dim == k_input_dim && input_dim == v_input_dim
}

pub(crate) fn metal_fused_qkv_transposed_coop_gemv_bf16(
    x: &kiln_tensor::Tensor,
    q_t: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    v_t: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_fused_qkv_transposed_coop_gemv_supports(x, q_t, k_t, v_t),
        "metal fused QKV projection supports only BF16 [1,1,K] x [K,Nq/Nk/Nv] on Metal"
    );
    let (_, _, input_dim) = x.dims3()?;
    let (_, q_output_dim) = q_t.dims2()?;
    let (_, k_output_dim) = k_t.dims2()?;
    let (_, v_output_dim) = v_t.dims2()?;

    let total_output_dim = q_output_dim + k_output_dim + v_output_dim;
    // The kernel writes each projection output independently with the existing
    // tile8 cooperative GEMV mapping. Back the three result views with one
    // allocation to avoid repeated small Metal buffer allocations in decode.
    let x_metal = kt_metal(&x)?;
    let fused_out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[1usize, 1usize, total_output_dim],
    )?;
    let q_out = fused_out.narrow(2, 0, q_output_dim)?;
    let k_out = fused_out.narrow(2, q_output_dim, k_output_dim)?;
    let v_out = fused_out.narrow(2, q_output_dim + k_output_dim, v_output_dim)?;

    let companion = x_metal.companion()?;
    let pipeline = metal_fused_qkv_transposed_coop_gemv_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_fused_qkv_transposed_coop_gemv8_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let q_metal = kt_metal(&q_t)?;
        let k_metal = kt_metal(&k_t)?;
        let v_metal = kt_metal(&v_t)?;
        let q_out_metal = kt_metal(&q_out)?;
        let k_out_metal = kt_metal(&k_out)?;
        let v_out_metal = kt_metal(&v_out)?;

        // #1082 Step 5 gemv/matmul-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q_t.layout(), q_t.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k_t.layout(), k_t.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_t.layout(), v_t.dtype());
        let q_out_buf = buffer_o_kt(q_out_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let k_out_buf = buffer_o_kt(k_out_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let v_out_buf = buffer_o_kt(v_out_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(q_out_buf.buffer), q_out_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(k_out_buf.buffer), k_out_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(v_out_buf.buffer), v_out_buf.offset_in_bytes);

        let input_dim_u32 = input_dim as u32;
        let q_output_dim_u32 = q_output_dim as u32;
        let k_output_dim_u32 = k_output_dim as u32;
        let v_output_dim_u32 = v_output_dim as u32;
        encoder.set_bytes(7, &input_dim_u32);
        encoder.set_bytes(8, &q_output_dim_u32);
        encoder.set_bytes(9, &k_output_dim_u32);
        encoder.set_bytes(10, &v_output_dim_u32);

        let cols_per_threadgroup =
            METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
        let q_groups = q_output_dim.div_ceil(cols_per_threadgroup);
        let k_groups = k_output_dim.div_ceil(cols_per_threadgroup);
        let v_groups = v_output_dim.div_ceil(cols_per_threadgroup);
        let total_groups = q_groups + k_groups + v_groups;
        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: total_groups,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: METAL_TRANSPOSED_COOP_GEMV_THREADS,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
}

pub(crate) fn metal_lora_add_decode_supports(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
) -> bool {
    if metal_lora_delta_decode_disabled() {
        return false;
    }
    if base.dtype() != kiln_tensor::DType::BF16
        || x.dtype() != kiln_tensor::DType::BF16
        || a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(base.device(), kiln_tensor::Device::Metal(_))
        || !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !base.is_contiguous() || !x.is_contiguous() || !a.is_contiguous() || !b.is_contiguous() {
        return false;
    }

    let Ok((batch, seq_len, input_dim)) = x.dims3() else {
        return false;
    };
    let Ok((base_batch, base_seq_len, output_dim)) = base.dims3() else {
        return false;
    };
    let Ok((rank, a_input_dim)) = a.dims2() else {
        return false;
    };
    let Ok((b_output_dim, b_rank)) = b.dims2() else {
        return false;
    };
    let Some(total_output) = batch.checked_mul(output_dim) else {
        return false;
    };
    let Some(hidden_total) = batch.checked_mul(rank) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && base_batch == batch
        && base_seq_len == 1
        && input_dim > 0
        && output_dim > 0
        && input_dim >= 1024
        && output_dim >= 1024
        && rank > 0
        && a_input_dim == input_dim
        && b_output_dim == output_dim
        && b_rank == rank
        && batch <= u32::MAX as usize
        && input_dim <= u32::MAX as usize
        && output_dim <= u32::MAX as usize
        && rank <= u32::MAX as usize
        && total_output <= u32::MAX as usize
        && hidden_total <= u32::MAX as usize
}

pub(crate) fn metal_lora_add_decode_bf16(
    base: &kiln_tensor::Tensor,
    x: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    scale: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_lora_add_decode_supports(base, x, a, b),
        "metal LoRA decode add supports only contiguous BF16 Metal base/x/A/B decode tensors"
    );
    let (batch, _, input_dim) = x.dims3()?;
    let (_, _, output_dim) = base.dims3()?;
    let (rank, _) = a.dims2()?;

    let x_metal = kt_metal(&x)?;
    let base_metal = kt_metal(&base)?;
    let hidden = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, rank])?;
    let out = kt_metal_alloc(
        base_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, output_dim],
    )?;

    let companion = x_metal.companion()?;
    let encoder = companion.command_encoder()?;

    {
        let pipeline = metal_lora_hidden_decode_pipeline(&*companion)?;
        encoder.set_label("kiln_lora_hidden_decode_bf16");
        encoder.set_compute_pipeline_state(&pipeline);

        let a_metal = kt_metal(&a)?;
        let hidden_metal = kt_metal(&hidden)?;

        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let hidden_buf = buffer_o_kt(
            hidden_metal.buffer().as_ref(),
            hidden.layout(),
            hidden.dtype(),
        );

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(hidden_buf.buffer), hidden_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let input_dim_u32 = input_dim as u32;
        let rank_u32 = rank as u32;
        encoder.set_bytes(3, &batch_u32);
        encoder.set_bytes(4, &input_dim_u32);
        encoder.set_bytes(5, &rank_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rank,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    {
        let pipeline = metal_lora_add_decode_pipeline(&*companion)?;
        encoder.set_label("kiln_lora_add_decode_bf16");
        encoder.set_compute_pipeline_state(&pipeline);

        let hidden_metal = kt_metal(&hidden)?;
        let b_metal = kt_metal(&b)?;
        let base_metal = kt_metal(&base)?;
        let out_metal = kt_metal(&out)?;

        let hidden_buf = buffer_o_kt(
            hidden_metal.buffer().as_ref(),
            hidden.layout(),
            hidden.dtype(),
        );
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let base_buf = buffer_o_kt(base_metal.buffer().as_ref(), base.layout(), base.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(hidden_buf.buffer), hidden_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(base_buf.buffer), base_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let output_dim_u32 = output_dim as u32;
        let rank_u32 = rank as u32;
        encoder.set_bytes(4, &scale);
        encoder.set_bytes(5, &batch_u32);
        encoder.set_bytes(6, &output_dim_u32);
        encoder.set_bytes(7, &rank_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch * output_dim,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    drop(encoder);
    Ok(out)
}

fn metal_gdn_in_proj_decode_supports(
    x: &kiln_tensor::Tensor,
    qkv_t: &kiln_tensor::Tensor,
    z_t: &kiln_tensor::Tensor,
    a_t: &kiln_tensor::Tensor,
    b_t: &kiln_tensor::Tensor,
) -> bool {
    if x.dtype() != kiln_tensor::DType::BF16
        || qkv_t.dtype() != kiln_tensor::DType::BF16
        || z_t.dtype() != kiln_tensor::DType::BF16
        || a_t.dtype() != kiln_tensor::DType::BF16
        || b_t.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(qkv_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(z_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous()
        || !qkv_t.is_contiguous()
        || !z_t.is_contiguous()
        || !a_t.is_contiguous()
        || !b_t.is_contiguous()
    {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((qkv_hidden, qkv_dim)) = qkv_t.dims2() else {
        return false;
    };
    let Ok((z_hidden, z_dim)) = z_t.dims2() else {
        return false;
    };
    let Ok((a_hidden, nv)) = a_t.dims2() else {
        return false;
    };
    let Ok((b_hidden, b_nv)) = b_t.dims2() else {
        return false;
    };
    let Some(total) = qkv_dim
        .checked_add(z_dim)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_add(b_nv))
    else {
        return false;
    };
    let Some(dispatch_total) = total.checked_mul(batch) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && hidden == qkv_hidden
        && hidden == z_hidden
        && hidden == a_hidden
        && hidden == b_hidden
        && nv == b_nv
        && hidden <= u32::MAX as usize
        && qkv_dim <= u32::MAX as usize
        && z_dim <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && total <= u32::MAX as usize
        && dispatch_total <= u32::MAX as usize
}

fn metal_gdn_in_proj_decode_bf16(
    x: &kiln_tensor::Tensor,
    qkv_t: &kiln_tensor::Tensor,
    z_t: &kiln_tensor::Tensor,
    a_t: &kiln_tensor::Tensor,
    b_t: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_in_proj_decode_supports(x, qkv_t, z_t, a_t, b_t),
        "metal gdn in-proj supports only BF16 [B,1,H] x [H,*] on Metal"
    );
    let (batch, _, hidden) = x.dims3()?;
    let (_, qkv_dim) = qkv_t.dims2()?;
    let (_, z_dim) = z_t.dims2()?;
    let (_, nv) = a_t.dims2()?;
    let output_total = qkv_dim + z_dim + (nv * 2);
    let row_grouping_enabled = batch >= 3 && !metal_gdn_in_proj_row_pair_disabled();
    let row_triple_enabled =
        batch == 3 && row_grouping_enabled && !metal_gdn_in_proj_row_triple_disabled();
    let row_quad_enabled =
        row_grouping_enabled && batch >= 8 && !metal_gdn_in_proj_row_quad_disabled();
    let row_group_size = if row_triple_enabled {
        3usize
    } else if row_quad_enabled {
        4usize
    } else if row_grouping_enabled {
        2usize
    } else {
        1usize
    };
    let x_metal = kt_metal(&x)?;
    // The kernel writes every output element exactly once. Keep bs=1 on one
    // backing allocation, but use separate batch outputs so each `[B,1,N]`
    // tensor remains contiguous for the following fused decode kernels.
    let (qkv_out, z_out, a_out, b_out) = if batch == 1 {
        let proj_out = kt_metal_alloc(
            x_metal,
            kiln_tensor::DType::BF16,
            &[1usize, 1usize, output_total],
        )?;
        (
            proj_out.narrow(2, 0, qkv_dim)?,
            proj_out.narrow(2, qkv_dim, z_dim)?,
            proj_out.narrow(2, qkv_dim + z_dim, nv)?,
            proj_out.narrow(2, qkv_dim + z_dim + nv, nv)?,
        )
    } else {
        (
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, qkv_dim])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, z_dim])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, nv])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, nv])?,
        )
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_gdn_in_proj_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_in_proj_decode_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let qkv_metal = kt_metal(&qkv_t)?;
        let z_metal = kt_metal(&z_t)?;
        let a_metal = kt_metal(&a_t)?;
        let b_metal = kt_metal(&b_t)?;
        let qkv_o_metal = kt_metal(&qkv_out)?;
        let z_o_metal = kt_metal(&z_out)?;
        let a_o_metal = kt_metal(&a_out)?;
        let b_o_metal = kt_metal(&b_out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let qkv_buf = buffer_o_kt(qkv_metal.buffer().as_ref(), qkv_t.layout(), qkv_t.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z_t.layout(), z_t.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a_t.layout(), a_t.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b_t.layout(), b_t.dtype());
        let qkv_o_buf = buffer_o_kt(
            qkv_o_metal.buffer().as_ref(),
            qkv_out.layout(),
            qkv_out.dtype(),
        );
        let z_o_buf = buffer_o_kt(z_o_metal.buffer().as_ref(), z_out.layout(), z_out.dtype());
        let a_o_buf = buffer_o_kt(a_o_metal.buffer().as_ref(), a_out.layout(), a_out.dtype());
        let b_o_buf = buffer_o_kt(b_o_metal.buffer().as_ref(), b_out.layout(), b_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(qkv_buf.buffer), qkv_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qkv_o_buf.buffer), qkv_o_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(z_o_buf.buffer), z_o_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(a_o_buf.buffer), a_o_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(b_o_buf.buffer), b_o_buf.offset_in_bytes);

        let serial_vector_mode = batch == 1
            && !metal_gdn_in_proj_serial_vector_load_disabled()
            && qkv_dim % 2 == 0
            && z_dim % 2 == 0
            && qkv_buf.offset_in_bytes % 4 == 0
            && z_buf.offset_in_bytes % 4 == 0;
        let serial_x2_mode = serial_vector_mode
            && !metal_gdn_in_proj_serial_x2_load_disabled()
            && hidden % 2 == 0
            && x_buf.offset_in_bytes % 4 == 0;
        let dispatch_cols = if serial_vector_mode {
            (qkv_dim / 2) + (z_dim / 2) + (nv * 2)
        } else if batch == 1 {
            output_total
        } else {
            qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + (nv * 2)
        };
        let dispatch_rows = batch.div_ceil(row_group_size);
        let dispatch_total = dispatch_rows * dispatch_cols;

        let hidden_u32 = hidden as u32;
        let qkv_dim_u32 = qkv_dim as u32;
        let z_dim_u32 = z_dim as u32;
        let nv_u32 = nv as u32;
        let batch_u32 = batch as u32;
        let row_pair_mode_u32 = if serial_x2_mode {
            7
        } else if serial_vector_mode {
            6
        } else if row_group_size == 1 {
            0
        } else {
            row_group_size as u32
        };
        encoder.set_bytes(9, &hidden_u32);
        encoder.set_bytes(10, &qkv_dim_u32);
        encoder.set_bytes(11, &z_dim_u32);
        encoder.set_bytes(12, &nv_u32);
        encoder.set_bytes(13, &batch_u32);
        encoder.set_bytes(14, &row_pair_mode_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: dispatch_total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((qkv_out, z_out, a_out, b_out))
}

pub(crate) fn metal_rotary_embedding_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    cos: &kiln_tensor::Tensor,
    sin: &kiln_tensor::Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_rotary_embedding_supports(q, k, cos, sin, head_dim, rotary_dim),
        "metal rotary qk unsupported shape"
    );
    let (batch, seq_len, q_heads, _) = q.dims4()?;
    let (_, _, k_heads, _) = k.dims4()?;
    let table_batch_stride =
        metal_rotary_table_batch_stride(cos, sin, batch, seq_len, rotary_dim / 2)
            .context("metal rotary qk unsupported position table shape")?;
    let q_shape = q.dims().to_vec();
    let k_shape = k.dims().to_vec();
    let q_metal = kt_metal(&q)?;
    let k_metal = kt_metal(&k)?;
    // SAFETY: the kernel dispatch writes every Q output element exactly once.
    let q_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, q_shape.as_slice())?;
    // SAFETY: the kernel dispatch writes every K output element exactly once.
    let k_out = kt_metal_alloc(k_metal, kiln_tensor::DType::BF16, k_shape.as_slice())?;

    let companion = q_metal.companion()?;
    let pipeline = metal_rotary_qk_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rotary_qk_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let cos_metal = kt_metal(&cos)?;
        let sin_metal = kt_metal(&sin)?;
        let q_out_metal = kt_metal(&q_out)?;
        let k_out_metal = kt_metal(&k_out)?;

        // #1082 Step 4 embedding-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let cos_buf = buffer_o_kt(cos_metal.buffer().as_ref(), cos.layout(), cos.dtype());
        let sin_buf = buffer_o_kt(sin_metal.buffer().as_ref(), sin.layout(), sin.dtype());
        let q_out_buf = buffer_o_kt(q_out_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let k_out_buf = buffer_o_kt(k_out_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(cos_buf.buffer), cos_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(sin_buf.buffer), sin_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(q_out_buf.buffer), q_out_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(k_out_buf.buffer), k_out_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let q_heads_u32 = q_heads as u32;
        let k_heads_u32 = k_heads as u32;
        let head_dim_u32 = head_dim as u32;
        let rotary_dim_u32 = rotary_dim as u32;
        let total_q = batch * seq_len * q_heads * head_dim;
        let total_k = batch * seq_len * k_heads * head_dim;
        let total = total_q + total_k;
        let total_q_u32 = total_q as u32;
        let total_u32 = total as u32;
        let table_batch_stride_u32 = table_batch_stride as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &q_heads_u32);
        encoder.set_bytes(9, &k_heads_u32);
        encoder.set_bytes(10, &head_dim_u32);
        encoder.set_bytes(11, &rotary_dim_u32);
        encoder.set_bytes(12, &total_q_u32);
        encoder.set_bytes(13, &total_u32);
        encoder.set_bytes(14, &table_batch_stride_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

pub(crate) fn metal_rms_norm_bf16(
    x: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    let x_dims = x.dims().to_vec();
    let hidden = *x_dims
        .last()
        .context("metal rmsnorm requires rank >= 1 input")?;
    anyhow::ensure!(hidden <= 8192, "metal rmsnorm hidden dim > 8192");
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal rmsnorm shape too large"
    );

    let x = x.contiguous()?;
    let weight = weight.contiguous()?;

    let x_metal = kt_metal(&x)?;
    // The kernel writes every hidden element for every row.
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &x_dims)?;

    if rows == 0 {
        return Ok(out);
    }

    let companion = x_metal.companion()?;
    let pipeline = metal_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 candle-free: buffers + layout + dtype come straight off the
        // kt MetalStorage / kt Tensor — no candle storage_and_layout bridge.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(3, &rows_u32);
        encoder.set_bytes(4, &hidden_u32);
        encoder.set_bytes(5, &eps);
        encoder.set_bytes(6, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_gdn_qk_norm_f32_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    q_scale: f32,
    eps: f32,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let dims = q.dims().to_vec();
    let hidden = *dims
        .last()
        .context("metal gdn qk norm requires rank >= 1 input")?;
    anyhow::ensure!(q.dims() == k.dims(), "metal gdn qk norm shape mismatch");
    anyhow::ensure!(hidden <= 8192, "metal gdn qk norm hidden dim > 8192");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gdn qk norm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // The kernel writes every Q/K element for every row.
    let q_metal = kt_metal(&q)?;
    let q_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, dims.as_slice())?;
    let k_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, dims.as_slice())?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_qk_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &q_scale);
        encoder.set_bytes(7, &eps);
        encoder.set_bytes(8, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

pub(crate) fn metal_gdn_qk_norm_gqa_f32_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    nv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_qk_norm_gqa_supports(q, k, nv),
        "metal gdn qk norm gqa unsupported shape"
    );
    let (batch, seq_len, nk, hidden) = q.dims4()?;
    let gqa_ratio = nv / nk;
    let rows = batch * seq_len * nk;
    anyhow::ensure!(
        rows <= u32::MAX as usize
            && nk <= u32::MAX as usize
            && hidden <= u32::MAX as usize
            && gqa_ratio <= u32::MAX as usize,
        "metal gdn qk norm gqa shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // Each source head writes all replicated value-head outputs.
    let q_metal = kt_metal(&q)?;
    let q_out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, hidden],
    )?;
    let k_out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, hidden],
    )?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_qk_norm_gqa_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_gqa_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        let hidden_u32 = hidden as u32;
        let gqa_ratio_u32 = gqa_ratio as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &nk_u32);
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &hidden_u32);
        encoder.set_bytes(8, &gqa_ratio_u32);
        encoder.set_bytes(9, &q_scale);
        encoder.set_bytes(10, &eps);
        encoder.set_bytes(11, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_bf16(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_decode_qkv_conv_norm_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ),
        "metal gdn decode qkv conv/norm unsupported shape"
    );
    let (batch, _, channels) = mixed_qkv.dims3()?;
    let rows_per_batch = nk + nk + nv;
    let rows = batch * rows_per_batch;
    anyhow::ensure!(
        rows <= u32::MAX as usize,
        "metal gdn decode qkv conv/norm shape too large"
    );

    let mixed_qkv = mixed_qkv.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn decode qkv conv/norm weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }

    // The kernel writes every unexpanded Q/K and V element, and updates each
    // convolution state channel exactly once.
    let mixed_qkv_metal = kt_metal(&mixed_qkv)?;
    let q_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nk, dk],
    )?;
    let k_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nk, dk],
    )?;
    let v_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nv, dv],
    )?;

    let companion = mixed_qkv_metal.companion()?;
    let pipeline = metal_gdn_decode_qkv_conv_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_qkv_conv_norm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let x_metal = kt_metal(&mixed_qkv)?;
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(
            x_metal.buffer().as_ref(),
            mixed_qkv.layout(),
            mixed_qkv.dtype(),
        );
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        encoder.set_bytes(6, &nk_u32);
        encoder.set_bytes(7, &nv_u32);
        encoder.set_bytes(8, &q_scale);
        encoder.set_bytes(9, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rows_per_batch,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
}









fn metal_gdn_gates_bf16(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates input too large"
    );
    anyhow::ensure!(nv <= u32::MAX as usize, "metal gdn_gates nv too large");

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_metal = kt_metal(&a)?;
    // The gates kernel writes every beta/g element.
    let beta = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, &shape)?;
    let g = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, &shape)?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_gates_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(g_buf.buffer), g_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, g))
}

pub(crate) fn metal_gdn_prefill_ab_in_proj_bf16(
    x: &kiln_tensor::Tensor,
    in_proj_ab_t: &kiln_tensor::Tensor,
    nv: usize,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_prefill_ab_in_proj_supports(x, in_proj_ab_t, nv),
        "metal gdn prefill A/B in-proj unsupported shape"
    );
    let ab = x
        .broadcast_matmul(in_proj_ab_t)
        .context("metal gdn prefill A/B in-proj matmul")?;
    let a = ab.narrow(2, 0, nv)?;
    let b = ab.narrow(2, nv, nv)?;
    Ok((ab, a, b))
}

pub(crate) fn metal_gdn_gates_decay_bf16(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_gates_decay_supports(a, b, a_log, dt_bias),
        "metal gdn_gates decay unsupported shape"
    );
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates decay requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates decay input too large"
    );
    anyhow::ensure!(
        nv <= u32::MAX as usize,
        "metal gdn_gates decay nv too large"
    );

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_metal = kt_metal(&a)?;
    let beta = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, shape.as_slice())?;
    let decay = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, shape.as_slice())?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_gates_decay_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_decay_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(decay_buf.buffer), decay_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, decay))
}

pub(crate) fn metal_gdn_gates_decay_ab_bf16(
    ab: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    nv: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_gates_decay_ab_supports(ab, a_log, dt_bias, nv),
        "metal gdn_gates decay A/B unsupported shape"
    );
    let (batch, seq_len, _channels) = ab.dims3()?;
    let total = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(nv))
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates decay A/B input too large"))?;
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates decay A/B input too large"
    );
    anyhow::ensure!(
        nv <= u32::MAX as usize,
        "metal gdn_gates decay A/B nv too large"
    );

    let ab = ab.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let shape = vec![batch, seq_len, nv];
    let ab_metal = kt_metal(&ab)?;
    let beta = kt_metal_alloc(ab_metal, kiln_tensor::DType::BF16, shape.as_slice())?;
    let decay = kt_metal_alloc(ab_metal, kiln_tensor::DType::BF16, shape.as_slice())?;

    let companion = ab_metal.companion()?;
    let pipeline = metal_gdn_gates_decay_ab_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_decay_ab_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let ab_buf = buffer_o_kt(ab_metal.buffer().as_ref(), ab.layout(), ab.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());

        encoder.set_buffer(0, Some(ab_buf.buffer), ab_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(decay_buf.buffer), decay_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(5, &nv_u32);
        encoder.set_bytes(6, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, decay))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state),
        "metal gdn decode gates+recurrent unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_decode_gates_recurrent_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let a_metal = kt_metal(&a)?;
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(9, &batch_heads_u32);
        encoder.set_bytes(10, &dk_u32);
        encoder.set_bytes(11, &dv_u32);
        encoder.set_bytes(12, &value_heads_u32);
        encoder.set_bytes(13, &q_heads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_rmsnorm_supports(
            q, k, v, a, b, a_log, dt_bias, state, z, weight
        ),
        "metal gdn decode gates+recurrent+rmsnorm unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent+rmsnorm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let a_metal = kt_metal(&a)?;
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let state_metal = kt_metal(&state)?;
        let z_metal = kt_metal(&z)?;
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z.layout(), z.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(11, &batch_heads_u32);
        encoder.set_bytes(12, &dk_u32);
        encoder.set_bytes(13, &dv_u32);
        encoder.set_bytes(14, &value_heads_u32);
        encoder.set_bytes(15, &q_heads_u32);
        encoder.set_bytes(16, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: dv,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}



fn metal_gated_rms_norm_bf16(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    let (batch, seq_len, heads, hidden) = x.dims4()?;
    let rows = batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(heads))
        .context("metal gated rmsnorm row count overflow")?;
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gated rmsnorm shape too large"
    );
    anyhow::ensure!(hidden <= 1024, "metal gated rmsnorm hidden dim > 1024");

    let x = x.contiguous()?;
    let z = z.contiguous()?;
    // The kernel binds `weight` as a `device const float*`. The norm weight is
    // BF16 for Qwen3.5 (matches the CUDA path); promote to F32 here so the
    // kernel's buffer type is satisfied. F32 weights pass through unchanged.
    let weight = weight.contiguous()?.to_dtype(kiln_tensor::DType::F32)?;
    let x_metal = kt_metal(&x)?;
    // The kernel writes every hidden element for every row.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, heads, hidden],
    )?;

    if rows == 0 {
        return Ok(out);
    }

    let companion = x_metal.companion()?;
    let pipeline = metal_gated_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gated_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let z_metal = kt_metal(&z)?;
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z.layout(), z.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &eps);
        encoder.set_bytes(7, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}












fn metal_gdn_forward_substitution_bf16(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    // The kernel writes every chunk/value element.
    let a_metal = kt_metal(&a_strict)?;
    let out = kt_metal_alloc(
        a_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_forward_substitution_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v_prime)?;
        let beta_metal = kt_metal(&beta)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_prime.layout(), v_prime.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_forward_substitution_f32(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution f32 shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    let a_metal = kt_metal(&a_strict)?;
    let out = kt_metal_alloc(
        a_metal,
        kiln_tensor::DType::F32,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_forward_substitution_f32_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_f32");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v_prime)?;
        let beta_metal = kt_metal(&beta)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_prime.layout(), v_prime.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_chunk_prep_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    let (batch, heads, chunk_size) = g.dims3()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn chunk-prep shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let g_metal = kt_metal(&g)?;
    // The prep kernel fills each temporary completely before any consumer sees it.
    let a_strict = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, chunk_size],
    )?;
    let b_mask = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, chunk_size],
    )?;
    let v_prime = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;
    let q_s_scaled = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;
    let decay_last_col = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size],
    )?;
    let p_last = kt_metal_alloc(g_metal, kiln_tensor::DType::BF16, &[batch, heads])?;

    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_chunk_prep_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_chunk_prep_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let a_metal = kt_metal(&a_strict)?;
        let b_metal = kt_metal(&b_mask)?;
        let vp_metal = kt_metal(&v_prime)?;
        let qss_metal = kt_metal(&q_s_scaled)?;
        let dl_metal = kt_metal(&decay_last_col)?;
        let pl_metal = kt_metal(&p_last)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b_mask.layout(), b_mask.dtype());
        let vp_buf = buffer_o_kt(
            vp_metal.buffer().as_ref(),
            v_prime.layout(),
            v_prime.dtype(),
        );
        let qss_buf = buffer_o_kt(
            qss_metal.buffer().as_ref(),
            q_s_scaled.layout(),
            q_s_scaled.dtype(),
        );
        let dl_buf = buffer_o_kt(
            dl_metal.buffer().as_ref(),
            decay_last_col.layout(),
            decay_last_col.dtype(),
        );
        let pl_buf = buffer_o_kt(pl_metal.buffer().as_ref(), p_last.layout(), p_last.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(vp_buf.buffer), vp_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(qss_buf.buffer), qss_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(dl_buf.buffer), dl_buf.offset_in_bytes);
        encoder.set_buffer(11, Some(pl_buf.buffer), pl_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(12, &batch_heads_u32);
        encoder.set_bytes(13, &chunk_size_u32);
        encoder.set_bytes(14, &dv_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

fn metal_gdn_full_chunk_forward_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state),
        "metal gdn full-chunk unsupported shape"
    );
    let (batch, heads, chunk_size) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn full-chunk shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let beta = beta.contiguous()?;
    let k_t = k_t.contiguous()?;
    let g_metal = kt_metal(&g)?;
    // The full-chunk kernel writes every output token/head/value element.
    let out = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_full_chunk_forward_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let beta_metal = kt_metal(&beta)?;
        let kt_metal_storage = kt_metal(&k_t)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let kt_buf = buffer_o_kt(
            kt_metal_storage.buffer().as_ref(),
            k_t.layout(),
            k_t.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 0u32;
        let t_start_u32 = 0u32;
        let seq_len_u32 = chunk_size as u32;
        let heads_u32 = heads as u32;
        let g_stride = g.layout().strides();
        let v_stride = v.layout().strides();
        let beta_stride = beta.layout().strides();
        let kt_stride = k_t.layout().strides();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn metal_gdn_full_chunk_forward_head_last_into_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    t_start: usize,
    seq_len: usize,
) -> Result<()> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_head_last_supports(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        ),
        "metal gdn full-chunk head-last unsupported shape"
    );
    let (batch, heads, _) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && t_start <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && heads <= u32::MAX as usize,
        "metal gdn full-chunk head-last shape too large"
    );

    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;

    let g_metal = kt_metal(&g)?;
    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_full_chunk_forward_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let beta_metal = kt_metal(&beta)?;
        let kt_metal_storage = kt_metal(&k_t)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let kt_buf = buffer_o_kt(
            kt_metal_storage.buffer().as_ref(),
            k_t.layout(),
            k_t.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 1u32;
        let t_start_u32 = t_start as u32;
        let seq_len_u32 = seq_len as u32;
        let heads_u32 = heads as u32;
        let g_stride = g.layout().strides();
        let v_stride = v.layout().strides();
        let beta_stride = beta.layout().strides();
        let kt_stride = k_t.layout().strides();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(())
}

fn metal_gdn_recurrent_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // The recurrent kernel writes every batch/head/value element.
    let out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, &[batch, heads, dv])?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &dk_u32);
        encoder.set_bytes(9, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_recurrent_prefill_head_last_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent prefill unsupported shape"
    );
    let (batch, q_heads, seq_len, dk) = q.dims4()?;
    let (_, value_heads, _, _) = v.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 0u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

fn metal_gdn_recurrent_prefill_native_head_last_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent native prefill unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_gdn_recurrent_prefill_native_head_last_decay_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_decay_supports(q, k, v, beta, decay, state),
        "metal gdn recurrent native prefill decay unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill decay shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let decay = decay.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_decay_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_decay_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(decay_buf.buffer), decay_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}






#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_supports(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_prefill_qkv_conv_split_disabled() {
        return false;
    }
    if mixed_qkv.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    if !matches!(mixed_qkv.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
        || !matches!(conv_state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !mixed_qkv.is_contiguous() || !weight.is_contiguous() || !conv_state.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let Ok((s_batch, s_channels, s_width)) = conv_state.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        2 => weight.dims() == [channels, kernel_size],
        3 => weight.dims() == [channels, 1, kernel_size],
        _ => false,
    };
    batch <= u32::MAX as usize
        && seq_len > 1
        && seq_len <= u32::MAX as usize
        && channels == expected_channels
        && channels <= u32::MAX as usize
        && qk_dim <= u32::MAX as usize
        && v_dim <= u32::MAX as usize
        && kernel_size == 4
        && (s_batch, s_channels, s_width) == (batch, channels, 3)
        && weight_ok
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_prefill_qkv_conv_split_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
        ),
        "metal gdn prefill qkv conv-split unsupported shape"
    );
    let (batch, seq_len, channels) = mixed_qkv.dims3()?;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;

    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn prefill qkv conv-split weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    let mixed_qkv_metal = kt_metal(mixed_qkv)?;
    let q = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::F32,
        &[batch, seq_len, nk, dk],
    )?;
    let k = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::F32,
        &[batch, seq_len, nk, dk],
    )?;
    let v = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, dv],
    )?;

    let companion = mixed_qkv_metal.companion()?;
    let pipeline = metal_gdn_prefill_qkv_conv_split_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(conv_state)?;
        let q_metal = kt_metal(&q)?;
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(
            mixed_qkv_metal.buffer().as_ref(),
            mixed_qkv.layout(),
            mixed_qkv.dtype(),
        );
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(v_buf.buffer), v_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let channels_u32 = channels as u32;
        let qk_dim_u32 = qk_dim as u32;
        let v_dim_u32 = v_dim as u32;
        let threads = seq_len.next_power_of_two().clamp(32, 256);
        let threads_u32 = threads as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &channels_u32);
        encoder.set_bytes(9, &qk_dim_u32);
        encoder.set_bytes(10, &v_dim_u32);
        encoder.set_bytes(11, &threads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q, k, v))
}

/// Test/helper: try to initialize a kt Metal device, returning `None` if Metal
/// isn't available or if device discovery panics in a sandboxed runner.
#[doc(hidden)]
pub fn try_new_metal() -> Option<kiln_tensor::Device> {
    let result = std::panic::catch_unwind(|| kiln_tensor::primary_metal_companion(0));
    match result {
        Ok(Ok(_)) => Some(kiln_tensor::Device::Metal(0)),
        Ok(Err(e)) => {
            eprintln!("Metal unavailable: {e}");
            None
        }
        Err(_) => {
            eprintln!("Metal device init panicked (likely CI sandbox with no Metal access)");
            None
        }
    }
}

#[cfg(test)]
mod metal_lm_head_sample_tests {
    use super::*;
    use crate::backend::BackendRuntime;
    use kiln_tensor::{Device, Tensor};
    use std::cmp::Ordering;

    fn metal_device() -> Option<Device> {
        super::try_new_metal()
    }

    fn pattern_bf16(n: usize, seed: u64) -> Vec<half::bf16> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..n {
            s = s
                .wrapping_add(0xA076_1D64_78BD_642F)
                .wrapping_mul(0xE703_7ED1_A0B4_28DB);
            let raw = ((s >> 40) as u32 % 4096) as f32 / 1024.0 - 2.0;
            let trend = (i % 19) as f32 * 0.011;
            out.push(half::bf16::from_f32(raw + trend));
        }
        out
    }

    fn lm_head_logits_for_row(
        x: &[half::bf16],
        weight_t: &[half::bf16],
        row: usize,
        hidden: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut logits = Vec::with_capacity(vocab);
        let row_base = row * hidden;
        for col in 0..vocab {
            let mut acc = 0.0f32;
            for i in 0..hidden {
                acc += x[row_base + i].to_f32() * weight_t[i * vocab + col].to_f32();
            }
            logits.push(half::bf16::from_f32(acc).to_f32());
        }
        logits
    }

    fn raw_argmax(logits: &[f32]) -> u32 {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for (idx, &score) in logits.iter().enumerate() {
            let idx = idx as u32;
            if score > best_score || (score == best_score && idx < best_idx) {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn splitmix_uniform(seed: u64) -> f32 {
        let state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let bits = z ^ (z >> 31);
        let mantissa = ((bits >> 40) & 0xFF_FFFF) as u32;
        mantissa as f32 / 16_777_216.0
    }

    fn unseeded_style_seed(history: &[u32]) -> u64 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let history_hash = history.iter().fold(0xCBF29CE484222325u64, |acc, &token| {
            (acc ^ token as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(history_hash)
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_sample(
        raw_logits: &[f32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> u32 {
        if kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k) {
            return raw_argmax(raw_logits);
        }

        let mut logits = raw_logits.to_vec();
        let rep_active = repetition_penalty.is_finite()
            && repetition_penalty > 0.0
            && (repetition_penalty - 1.0).abs() > f32::EPSILON;
        for (&idx, &count) in history_indices.iter().zip(history_counts.iter()) {
            let Some(score) = logits.get_mut(idx as usize) else {
                continue;
            };
            if rep_active {
                *score = if *score > 0.0 {
                    *score / repetition_penalty
                } else {
                    *score * repetition_penalty
                };
            }
            if presence_penalty.is_finite() && presence_penalty != 0.0 {
                *score -= presence_penalty;
            }
            if frequency_penalty.is_finite() && frequency_penalty != 0.0 {
                *score -= frequency_penalty * count as f32;
            }
        }

        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx as u32, score / temperature))
            .collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        indexed.truncate((top_k as usize).min(indexed.len()).max(1));

        let max_score = indexed[0].1;
        let mut probs: Vec<(u32, f32)> = indexed
            .iter()
            .map(|&(idx, score)| (idx, (score - max_score).exp()))
            .collect();
        let mut sum: f32 = probs.iter().map(|(_, p)| *p).sum();
        if !sum.is_finite() || sum <= 0.0 {
            return indexed[0].0;
        }
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }

        if min_p.is_finite() && min_p > 0.0 {
            let threshold = min_p * probs[0].1;
            probs.retain(|&(_, p)| p >= threshold);
            if probs.is_empty() {
                return indexed[0].0;
            }
            sum = probs.iter().map(|(_, p)| *p).sum();
            if sum > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        if top_p > 0.0 && top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cumsum += *p;
                if cumsum >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            sum = probs.iter().map(|(_, p)| *p).sum();
            if sum > 0.0 {
                for (_, p) in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        let r = splitmix_uniform(seed);
        let mut cumsum = 0.0f32;
        for &(idx, p) in &probs {
            cumsum += p;
            if r < cumsum {
                return idx;
            }
        }
        probs.last().map(|&(idx, _)| idx).unwrap_or(indexed[0].0)
    }

    #[test]
    fn linear_decode_sample_top_k_one_ignores_penalties_and_matches_raw_argmax() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head top_k=1 sample test");
            return Ok(());
        };
        let hidden = 8usize;
        let vocab = 17usize;
        let x_data = pattern_bf16(hidden, 1);
        let weight_data = pattern_bf16(hidden * vocab, 2);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![1, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let backend = MetalBackend::new(dev);
        let logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let want = raw_argmax(&logits);

        let got = backend
            .linear_decode_sample(
                &x,
                &weight_t,
                &[want],
                &[100],
                1.4,
                3.0,
                0.2,
                0.7,
                1,
                0.5,
                0.1,
                0xCAFE_F00D_DEAD_BEEF,
            )?
            .context("Metal backend declined top_k=1 sampled decode")?;
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn metal_lm_head_sample_matches_reference_top_p_min_p_penalties_seeded() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head seeded sample test");
            return Ok(());
        };
        let hidden = 9usize;
        let vocab = 37usize;
        let x_data = pattern_bf16(hidden, 3);
        let weight_data = pattern_bf16(hidden * vocab, 4);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![1, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let history_indices = [2u32, 5, 11, 23];
        let history_counts = [1u32, 3, 2, 4];
        let seed = 0x1234_5678_90AB_CDEF;
        let got = metal_lm_head_sample_bf16(
            &x,
            &weight_t,
            &history_indices,
            &history_counts,
            1.2,
            0.4,
            0.15,
            0.8,
            7,
            0.82,
            0.03,
            seed,
        )?;
        let again = metal_lm_head_sample_bf16(
            &x,
            &weight_t,
            &history_indices,
            &history_counts,
            1.2,
            0.4,
            0.15,
            0.8,
            7,
            0.82,
            0.03,
            seed,
        )?;
        let logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let want = reference_sample(
            &logits,
            &history_indices,
            &history_counts,
            1.2,
            0.4,
            0.15,
            0.8,
            7,
            0.82,
            0.03,
            seed,
        );
        assert_eq!(got, want);
        assert_eq!(again, want, "same seed must be deterministic");
        Ok(())
    }

    #[test]
    fn metal_lm_head_sample_matches_reference_top_k_top_p_unseeded_style_seed() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head unseeded-style sample test");
            return Ok(());
        };
        let hidden = 11usize;
        let vocab = 43usize;
        let x_data = pattern_bf16(hidden, 7);
        let weight_data = pattern_bf16(hidden * vocab, 8);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![1, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let history = [3u32, 5, 3, 17, 5, 29];
        let (history_indices, history_counts): (Vec<u32>, Vec<u32>) =
            [(3u32, 2u32), (5, 2), (17, 1), (29, 1)].into_iter().unzip();
        let seed = unseeded_style_seed(&history);
        let got = metal_lm_head_sample_bf16(
            &x,
            &weight_t,
            &history_indices,
            &history_counts,
            1.0,
            0.0,
            0.0,
            0.95,
            11,
            0.7,
            0.0,
            seed,
        )?;
        let logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let want = reference_sample(
            &logits,
            &history_indices,
            &history_counts,
            1.0,
            0.0,
            0.0,
            0.95,
            11,
            0.7,
            0.0,
            seed,
        );
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn linear_decode_sample_batch_handles_mixed_greedy_and_sampled_rows() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping Metal lm-head batched sample test");
            return Ok(());
        };
        let batch = 2usize;
        let hidden = 10usize;
        let vocab = 41usize;
        let x_data = pattern_bf16(batch * hidden, 5);
        let weight_data = pattern_bf16(hidden * vocab, 6);
        let x = Tensor::from_vec_on(dev, x_data.clone(), vec![batch, 1, hidden])?;
        let weight_t = Tensor::from_vec_on(dev, weight_data.clone(), vec![hidden, vocab])?;
        let backend = MetalBackend::new(dev);

        let tokens = backend
            .linear_decode_sample_batch(
                &x,
                &weight_t,
                &[1, 1, 1],
                &[3, 7, 19],
                &[2, 1, 4],
                &[1.0, 1.15],
                &[0.0, 0.35],
                &[0.0, 0.08],
                &[0.0, 0.9],
                &[0, 6],
                &[1.0, 0.74],
                &[0.0, 0.02],
                &[0xABCD, 0x1234_0000_5678_9999],
            )?
            .context("Metal backend declined batched sampled decode")?;
        assert_eq!(tokens.len(), batch);

        let row0_logits = lm_head_logits_for_row(&x_data, &weight_data, 0, hidden, vocab);
        let row1_logits = lm_head_logits_for_row(&x_data, &weight_data, 1, hidden, vocab);
        let want0 = raw_argmax(&row0_logits);
        let want1 = reference_sample(
            &row1_logits,
            &[3, 7, 19],
            &[2, 1, 4],
            1.15,
            0.35,
            0.08,
            0.9,
            6,
            0.74,
            0.02,
            0x1234_0000_5678_9999,
        );
        assert_eq!(tokens, vec![want0, want1]);
        Ok(())
    }

    #[test]
    fn sample_batch_support_does_not_claim_pure_greedy_batches() {
        let backend = MetalBackend::new(Device::Metal(0));
        assert!(!backend.supports_linear_decode_sample_batch(&[20], &[0.0]));
        assert!(!backend.supports_linear_decode_sample_batch(&[1, 1], &[0.7, 0.8]));
        assert!(backend.supports_linear_decode_sample_batch(&[20, 1], &[0.8, 0.0]));
    }
}

#[cfg(test)]
mod metal_icb_decode_tests {
    use super::*;
    use kiln_tensor::{Device, Tensor};

    fn metal_device() -> Option<Device> {
        kiln_tensor::primary_metal_companion(0)
            .ok()
            .map(|_| Device::Metal(0))
    }

    fn pattern_bf16(n: usize, seed: u64) -> Vec<half::bf16> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in 0..n {
            s = s
                .wrapping_add(0xA076_1D64_78BD_642F)
                .wrapping_mul(0xE703_7ED1_A0B4_28DB);
            let raw = ((s >> 40) as u32 % 1024) as f32 / 4096.0 - 0.125;
            let trend = (i % 17) as f32 * 0.0007;
            out.push(half::bf16::from_f32(raw + trend));
        }
        out
    }

    fn zeroed_bf16(n: usize) -> Vec<half::bf16> {
        vec![half::bf16::ZERO; n]
    }

    fn max_abs_diff_bf16(a: &[half::bf16], b: &[half::bf16]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn single_token_paged_decode_icb_matches_eager_and_updates_slot() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!(
                "Metal unavailable, skipping single_token_paged_decode_icb_matches_eager_and_updates_slot"
            );
            return Ok(());
        };

        let total_slots = 4usize;
        let kv_heads = 4usize;
        let q_heads = 16usize;
        let head_dim = 256usize;
        let pool_elems = total_slots * kv_heads * head_dim;
        let kv_elems = kv_heads * head_dim;
        let q_elems = q_heads * head_dim;
        let out_elems = q_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut k_pool_host = zeroed_bf16(pool_elems);
        let mut v_pool_host = zeroed_bf16(pool_elems);
        let prefix_k = pattern_bf16(2 * kv_elems, 10);
        let prefix_v = pattern_bf16(2 * kv_elems, 11);
        k_pool_host[..2 * kv_elems].copy_from_slice(&prefix_k);
        v_pool_host[..2 * kv_elems].copy_from_slice(&prefix_v);

        let q = Tensor::from_vec_on(
            dev,
            pattern_bf16(q_elems, 12),
            vec![1, 1, q_heads, head_dim],
        )?;
        let k = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 13),
            vec![1, 1, kv_heads, head_dim],
        )?;
        let v = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 14),
            vec![1, 1, kv_heads, head_dim],
        )?;
        let k_pool_eager = Tensor::from_vec_on(
            dev,
            k_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let v_pool_eager = Tensor::from_vec_on(
            dev,
            v_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let k_pool_icb =
            Tensor::from_vec_on(dev, k_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let v_pool_icb =
            Tensor::from_vec_on(dev, v_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let block_table = Tensor::from_vec_on(dev, vec![0u32, 1, 2], vec![1, 3])?;
        let seqused_k = Tensor::from_vec_on(dev, vec![3u32], vec![1])?;
        let out_icb =
            Tensor::from_vec_on(dev, zeroed_bf16(out_elems), vec![1, 1, q_heads, head_dim])?;

        let graph = metal_record_single_token_paged_decode_icb_graph(
            &q,
            &k_pool_icb,
            &v_pool_icb,
            &block_table,
            &seqused_k,
            &out_icb,
            &k,
            &v,
            2,
            3,
            1,
            scale,
        )?;

        metal_paged_kv_write_token_major_bf16(&k_pool_eager, &v_pool_eager, 2, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(2, 3, scale)?;

        let eager_0 = eager.to_vec::<half::bf16>()?;
        let icb_0 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_0, icb_0,
            "first ICB replay must be bit-identical to eager Metal decode"
        );

        let next_k = pattern_bf16(kv_elems, 20);
        let next_v = pattern_bf16(kv_elems, 21);
        kiln_tensor::metal_write_host_in_place(&k, &next_k)?;
        kiln_tensor::metal_write_host_in_place(&v, &next_v)?;
        kiln_tensor::metal_write_host_in_place(&block_table, &[0u32, 1, 3])?;

        metal_paged_kv_write_token_major_bf16(&k_pool_eager, &v_pool_eager, 3, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, 3, scale)?;

        let eager_1 = eager.to_vec::<half::bf16>()?;
        let icb_1 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_1, icb_1,
            "ICB replay after stable-buffer and slot updates must match eager"
        );
        assert_eq!(graph.replay_count(), 2);
        assert!(
            max_abs_diff_bf16(&icb_0, &icb_1) > 0.0,
            "second replay should observe refreshed K/V and metadata"
        );

        Ok(())
    }

    #[test]
    fn batched_paged_decode_icb_matches_eager_and_updates_slots() -> Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!(
                "Metal unavailable, skipping batched_paged_decode_icb_matches_eager_and_updates_slots"
            );
            return Ok(());
        };

        let batch = 2usize;
        let total_slots = 8usize;
        let kv_heads = 4usize;
        let q_heads = 16usize;
        let head_dim = 256usize;
        let pool_row = kv_heads * head_dim;
        let pool_elems = total_slots * pool_row;
        let kv_elems = batch * pool_row;
        let q_elems = batch * q_heads * head_dim;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut k_pool_host = zeroed_bf16(pool_elems);
        let mut v_pool_host = zeroed_bf16(pool_elems);
        for row in 0..batch {
            let block_base = row * 4;
            for prefix_idx in 0..2 {
                let slot = block_base + prefix_idx;
                let dst = slot * pool_row;
                let seed = 100 + (row * 10 + prefix_idx) as u64;
                k_pool_host[dst..dst + pool_row].copy_from_slice(&pattern_bf16(pool_row, seed));
                v_pool_host[dst..dst + pool_row].copy_from_slice(&pattern_bf16(pool_row, seed + 1));
            }
        }

        let q = Tensor::from_vec_on(
            dev,
            pattern_bf16(q_elems, 12),
            vec![batch, 1, q_heads, head_dim],
        )?;
        let k = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 13),
            vec![batch, 1, kv_heads, head_dim],
        )?;
        let v = Tensor::from_vec_on(
            dev,
            pattern_bf16(kv_elems, 14),
            vec![batch, 1, kv_heads, head_dim],
        )?;
        let k_pool_eager = Tensor::from_vec_on(
            dev,
            k_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let v_pool_eager = Tensor::from_vec_on(
            dev,
            v_pool_host.clone(),
            vec![total_slots, kv_heads, head_dim],
        )?;
        let k_pool_icb =
            Tensor::from_vec_on(dev, k_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let v_pool_icb =
            Tensor::from_vec_on(dev, v_pool_host, vec![total_slots, kv_heads, head_dim])?;
        let block_table = Tensor::from_vec_on(dev, vec![0u32, 1, 2, 4, 5, 6], vec![batch, 3])?;
        let seqused_k = Tensor::from_vec_on(dev, vec![3u32, 3], vec![batch])?;
        let slots = Tensor::from_vec_on(dev, vec![2u32, 6], vec![batch])?;
        let out_icb =
            Tensor::from_vec_on(dev, zeroed_bf16(q_elems), vec![batch, 1, q_heads, head_dim])?;

        let graph = metal_record_paged_decode_icb_graph(
            &q,
            &k_pool_icb,
            &v_pool_icb,
            &block_table,
            &seqused_k,
            &out_icb,
            &k,
            &v,
            &slots,
            3,
            1,
            scale,
        )?;

        metal_paged_kv_write_token_major_batch_bf16(&k_pool_eager, &v_pool_eager, &slots, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, scale)?;

        let eager_0 = eager.to_vec::<half::bf16>()?;
        let icb_0 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_0, icb_0,
            "first batched ICB replay must be bit-identical to eager Metal decode"
        );

        let next_k = pattern_bf16(kv_elems, 20);
        let next_v = pattern_bf16(kv_elems, 21);
        kiln_tensor::metal_write_host_in_place(&k, &next_k)?;
        kiln_tensor::metal_write_host_in_place(&v, &next_v)?;
        kiln_tensor::metal_write_host_in_place(&block_table, &[0u32, 1, 3, 4, 5, 7])?;
        kiln_tensor::metal_write_host_in_place(&slots, &[3u32, 7])?;

        metal_paged_kv_write_token_major_batch_bf16(&k_pool_eager, &v_pool_eager, &slots, &k, &v)?;
        let eager = metal_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
            &q,
            &k_pool_eager,
            &v_pool_eager,
            &block_table,
            &seqused_k,
            3,
            1,
            scale,
        )?;
        graph.replay(3, scale)?;

        let eager_1 = eager.to_vec::<half::bf16>()?;
        let icb_1 = out_icb.to_vec::<half::bf16>()?;
        assert_eq!(
            eager_1, icb_1,
            "batched ICB replay after stable slot updates must match eager"
        );
        assert_eq!(graph.replay_count(), 2);
        assert!(
            max_abs_diff_bf16(&icb_0, &icb_1) > 0.0,
            "second batched replay should observe refreshed K/V and metadata"
        );

        Ok(())
    }
}

// ----------------------------------------------------------------------
// On-device AdamW parity (#1082) — the optimizer oracle. A wrong optimizer
// silently corrupts training, so this gate compares the fused Metal
// `dispatch_adamw_step` (registry-resident, in-place) against the host
// reference math (a bit-faithful copy of `kiln_optim::AdamW::step`,
// adamw.rs ~165-181) over several steps, asserting param/m/v match to F32
// tolerance. Lives in a LIVE test module (not the candle-era `cfg(any())`
// block above) so it actually runs on the M1 validator.
#[cfg(test)]
mod adamw_kt_tests {
    use super::*;
    use kiln_tensor::{DType, Device, Tensor};

    /// `Device::Metal(0)` if a Metal device is reachable, else `None`.
    fn metal_device() -> Option<Device> {
        kiln_tensor::primary_metal_companion(0)
            .ok()
            .map(|_| Device::Metal(0))
    }

    /// One in-place AdamW step over f32 host buffers — the reference the
    /// kernel must match. Identical arithmetic + order to
    /// `kiln_optim::AdamW::step`.
    #[allow(clippy::too_many_arguments)]
    fn host_adamw_step(
        param: &mut [f32],
        m: &mut [f32],
        v: &mut [f32],
        grad: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let stepf = step as f32;
        let bc1 = 1.0 - beta1.powf(stepf);
        let bc2 = 1.0 - beta2.powf(stepf);
        for i in 0..param.len() {
            let g = grad[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps));
            param[i] -= lr * weight_decay * param[i];
            param[i] -= update;
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn dispatch_adamw_step_matches_host_reference_f32() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping dispatch_adamw_step_matches_host_reference_f32");
            return Ok(());
        };

        let n = 257usize; // non-multiple of 256 → exercises the tail thread
        let lr = 0.013f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.02f32;
        let steps = 5u32;

        // Deterministic, mildly varied data.
        let param0: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.017) - 2.1).sin() * 0.5)
            .collect();
        // A fresh grad per step keeps the moments moving.
        let grads: Vec<Vec<f32>> = (1..=steps)
            .map(|s| {
                (0..n)
                    .map(|i| ((i as f32 + s as f32 * 1.7) * 0.031).cos() * 0.08)
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Host reference state.
        let mut h_param = param0.clone();
        let mut h_m = vec![0.0f32; n];
        let mut h_v = vec![0.0f32; n];

        // Metal state: param + m + v are persistent across steps (the kernel
        // mutates them in place), so build them once and register them.
        let met_param = Tensor::from_vec_on(dev, param0.clone(), vec![n])?;
        let met_m = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let met_v = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;

        let backend = MetalBackend::new(dev);
        assert!(backend.supports_resident_activation());
        backend.register_resident_activation(&met_param)?;
        backend.register_resident_activation(&met_m)?;
        backend.register_resident_activation(&met_v)?;
        assert!(backend.has_resident_activation(&met_param));
        assert!(backend.has_resident_activation(&met_m));
        assert!(backend.has_resident_activation(&met_v));

        for s in 1..=steps {
            let g = &grads[(s - 1) as usize];
            host_adamw_step(
                &mut h_param,
                &mut h_m,
                &mut h_v,
                g,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            );

            // Fresh grad tensor each step (distinct TensorId), mirroring the
            // trainer registering the grad on the fly.
            let met_grad = Tensor::from_vec_on(dev, g.clone(), vec![n])?;
            backend.register_resident_activation(&met_grad)?;

            let dispatched = backend.dispatch_adamw_step(
                &met_param,
                &met_grad,
                &met_m,
                &met_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            )?;
            assert!(
                dispatched,
                "dispatch_adamw_step must take the on-device path (step {s})"
            );
            backend.evict_resident_activation(&met_grad);
        }

        // Read the device results back to host.
        let g_param: Vec<f32> = met_param.to_device(Device::Cpu)?.to_vec::<f32>()?;
        let g_m: Vec<f32> = met_m.to_device(Device::Cpu)?.to_vec::<f32>()?;
        let g_v: Vec<f32> = met_v.to_device(Device::Cpu)?.to_vec::<f32>()?;

        let tol = 1e-5f32;
        let dp = max_abs_diff(&g_param, &h_param);
        let dm = max_abs_diff(&g_m, &h_m);
        let dv = max_abs_diff(&g_v, &h_v);
        eprintln!(
            "adamw parity over {steps} steps (n={n}): max|Δparam|={dp:e} max|Δm|={dm:e} max|Δv|={dv:e} (tol={tol:e})"
        );
        assert!(dp < tol, "param diverged: max|Δ|={dp:e} >= {tol:e}");
        assert!(dm < tol, "m diverged: max|Δ|={dm:e} >= {tol:e}");
        assert!(dv < tol, "v diverged: max|Δ|={dv:e} >= {tol:e}");

        // resolve_resident_activation must round-trip the in-place-updated
        // buffer (what `sync_to_master` relies on).
        let resolved = backend
            .resolve_resident_activation(&met_param, &[n], DType::F32)?
            .expect("param is resident, resolve must return Some");
        let r_param: Vec<f32> = resolved.to_device(Device::Cpu)?.to_vec::<f32>()?;
        assert!(
            max_abs_diff(&r_param, &g_param) < 1e-6,
            "resolve_resident_activation must reflect the in-place update"
        );

        backend.evict_resident_activation(&met_param);
        backend.evict_resident_activation(&met_m);
        backend.evict_resident_activation(&met_v);
        assert!(!backend.has_resident_activation(&met_param));
        Ok(())
    }

    /// BF16-master reference: mirrors the Metal kernel exactly — read each
    /// operand BF16→f32, run the AdamW math in f32, write the moments + master
    /// back as round-to-nearest BF16 (so the *stored* moments are lossy, the
    /// on-device convention shared with CUDA/Vulkan). Round-to-nearest-even
    /// matches MSL's `(bfloat)` conversion.
    #[allow(clippy::too_many_arguments)]
    fn host_adamw_step_bf16(
        param: &mut [half::bf16],
        m: &mut [half::bf16],
        v: &mut [half::bf16],
        grad: &[half::bf16],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let stepf = step as f32;
        let bc1 = 1.0 - beta1.powf(stepf);
        let bc2 = 1.0 - beta2.powf(stepf);
        for i in 0..param.len() {
            let g = grad[i].to_f32();
            let mf = beta1 * m[i].to_f32() + (1.0 - beta1) * g;
            let vf = beta2 * v[i].to_f32() + (1.0 - beta2) * g * g;
            let m_hat = mf / bc1;
            let v_hat = vf / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps));
            let mut pf = param[i].to_f32();
            pf -= lr * weight_decay * pf;
            pf -= update;
            m[i] = half::bf16::from_f32(mf);
            v[i] = half::bf16::from_f32(vf);
            param[i] = half::bf16::from_f32(pf);
        }
    }

    /// On-device BF16 AdamW (the real LoRA-training dtype) must match the BF16
    /// reference bit-for-bit: same f32 math, same round-to-nearest BF16 store.
    /// This is the on-device path actually exercised by the SFT/GRPO/OPD/GDN
    /// training smokes (their masters are BF16).
    #[test]
    fn dispatch_adamw_step_matches_bf16_reference() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping dispatch_adamw_step_matches_bf16_reference");
            return Ok(());
        };
        let n = 257usize;
        let (lr, beta1, beta2, eps, weight_decay) = (0.013f32, 0.9f32, 0.999f32, 1e-8f32, 0.02f32);
        let steps = 5u32;

        let to_bf16 = |xs: &[f32]| -> Vec<half::bf16> {
            xs.iter().map(|&x| half::bf16::from_f32(x)).collect()
        };
        let param0: Vec<half::bf16> = to_bf16(
            &(0..n)
                .map(|i| ((i as f32 * 0.017) - 2.1).sin() * 0.5)
                .collect::<Vec<_>>(),
        );
        let grads: Vec<Vec<half::bf16>> = (1..=steps)
            .map(|s| {
                to_bf16(
                    &(0..n)
                        .map(|i| ((i as f32 + s as f32 * 1.7) * 0.031).cos() * 0.08)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        let mut h_param = param0.clone();
        let mut h_m = vec![half::bf16::ZERO; n];
        let mut h_v = vec![half::bf16::ZERO; n];

        let met_param = Tensor::from_vec_on(dev, param0.clone(), vec![n])?;
        let met_m = Tensor::from_vec_on(dev, vec![half::bf16::ZERO; n], vec![n])?;
        let met_v = Tensor::from_vec_on(dev, vec![half::bf16::ZERO; n], vec![n])?;
        assert_eq!(met_param.dtype(), DType::BF16);

        let backend = MetalBackend::new(dev);
        backend.register_resident_activation(&met_param)?;
        backend.register_resident_activation(&met_m)?;
        backend.register_resident_activation(&met_v)?;

        for s in 1..=steps {
            let g = &grads[(s - 1) as usize];
            host_adamw_step_bf16(
                &mut h_param,
                &mut h_m,
                &mut h_v,
                g,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            );
            let met_grad = Tensor::from_vec_on(dev, g.clone(), vec![n])?;
            backend.register_resident_activation(&met_grad)?;
            let dispatched = backend.dispatch_adamw_step(
                &met_param,
                &met_grad,
                &met_m,
                &met_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            )?;
            assert!(
                dispatched,
                "BF16 dispatch_adamw_step must take the on-device path (step {s})"
            );
            backend.evict_resident_activation(&met_grad);
        }

        let g_param = met_param.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        let g_m = met_m.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        let g_v = met_v.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        // Bit-exact expected (identical f32 math + round-to-nearest store); allow
        // a hair for any MSL-vs-Rust sqrt/div last-bit nuance.
        let f = |a: &[half::bf16]| a.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let dp = max_abs_diff(&f(&g_param), &f(&h_param));
        let dm = max_abs_diff(&f(&g_m), &f(&h_m));
        let dv = max_abs_diff(&f(&g_v), &f(&h_v));
        eprintln!(
            "adamw bf16 parity (n={n}, {steps} steps): max|Δp|={dp:e} max|Δm|={dm:e} max|Δv|={dv:e}"
        );
        assert!(dp < 1e-2, "bf16 param diverged: {dp:e}");
        assert!(dm < 1e-3, "bf16 m diverged: {dm:e}");
        assert!(dv < 1e-4, "bf16 v diverged: {dv:e}");
        Ok(())
    }

    /// dispatch_adamw_step must decline (Ok(false)) when an operand isn't
    /// resident, so the trainer falls through to the host AdamW.
    #[test]
    fn dispatch_adamw_step_declines_when_not_resident() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            return Ok(());
        };
        let n = 8usize;
        let p = Tensor::from_vec_on(dev, vec![0.1f32; n], vec![n])?;
        let g = Tensor::from_vec_on(dev, vec![0.2f32; n], vec![n])?;
        let m = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let v = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let backend = MetalBackend::new(dev);
        // Nothing registered → decline.
        let dispatched =
            backend.dispatch_adamw_step(&p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?;
        assert!(!dispatched, "must decline when operands aren't resident");
        Ok(())
    }
}
