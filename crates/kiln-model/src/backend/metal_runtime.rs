//! Metal `BackendRuntime` trait implementation.
//!
//! This module keeps trait-surface routing separate from the Metal backend
//! construction/precompile facade and the operation-family command encoders.

use anyhow::{Context, Result};

use super::metal::MetalBackend;
use super::metal_attention::*;
use super::metal_config::*;
use super::metal_conv1d::*;
use super::metal_gdn::*;
use super::metal_lm_head::*;
use super::metal_paged::*;
use super::{
    AttentionBackend, BackendIdentity, BackendMatmulLayout, BackendRuntime, ConvBackend,
    GdnBackend, LinearBackend, OptimizerBackend, PagedKvBackend, ReplayBackend, ResidencyBackend,
    SamplingBackend, StartupBackend, TrainingCapabilities, TrainingLossBackend,
    TrainingPrecisionPolicy, matmul_request_support_rank, matmul_support_from_native,
    metal_residency, metal_training, requested_matmul_layout,
};

impl BackendIdentity for MetalBackend {
    fn runtime_name(&self) -> &'static str {
        "metal"
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        &()
    }
}

impl StartupBackend for MetalBackend {}

#[allow(clippy::too_many_arguments)]
impl ConvBackend for MetalBackend {
    fn runtime_supports_causal_conv1d_prefill(&self) -> bool {
        !self.disable.conv1d_prefill
    }

    fn runtime_supports_causal_conv1d_update(&self) -> bool {
        !self.disable.conv1d_update
    }

    fn runtime_causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native - helpers take kt directly; `conv_state` is mutated
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

    fn runtime_causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // #1082: kt-native - helpers take kt directly; `conv_state` is mutated
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
}

#[allow(clippy::too_many_arguments)]
impl SamplingBackend for MetalBackend {
    fn runtime_supports_linear_decode_sample(&self, top_k: u32) -> bool {
        top_k > 0 && top_k <= METAL_LM_HEAD_SAMPLE_TOP_K_MAX && !metal_lm_head_sample_disabled()
    }

    fn runtime_linear_decode_sample(
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
        if !self.runtime_supports_linear_decode_sample(top_k) {
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

    fn runtime_supports_linear_decode_sample_batch(
        &self,
        top_k: &[u32],
        temperatures: &[f32],
    ) -> bool {
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

    fn runtime_linear_decode_sample_batch(
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
        if !self.runtime_supports_linear_decode_sample_batch(top_k, temperatures) {
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
}

#[allow(clippy::too_many_arguments)]
impl OptimizerBackend for MetalBackend {
    fn runtime_dispatch_adamw_step(
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
        let all_resident = metal_residency::all_registered(
            &self.resident_activation_registry,
            &[param, grad, first_moment, second_moment],
        );
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
}

#[allow(clippy::too_many_arguments)]
impl PagedKvBackend for MetalBackend {
    fn runtime_supports_paged_kv_head_major_read(&self) -> bool {
        true
    }

    fn runtime_supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
        true
    }

    fn runtime_paged_kv_head_major_read(
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

    fn runtime_paged_kv_head_major_read_append_token_major(
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
}

#[allow(clippy::too_many_arguments)]
impl AttentionBackend for MetalBackend {
    fn runtime_supports_flash_attn_prefill(&self) -> bool {
        metal_sdpa_prefill_available()
    }

    fn runtime_supports_flash_attn_prefill_head_major(&self) -> bool {
        metal_sdpa_prefill_available()
    }

    // Note: keep `supports_*` returning true so the planner picks the SDPA
    // path; the per-call gate inside the kernel functions then decides
    // whether the *specific* shape is safe and silently falls back to the
    // naive softmax+matmul path when it isn't.
    fn runtime_supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn runtime_flash_attn_paged_decode_contiguous(
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

    fn runtime_flash_attn_paged_decode_contiguous_batch(
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

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
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

    fn runtime_flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        metal_flash_attn_prefill(q, k, v, softmax_scale, causal)
    }

    fn runtime_flash_attn_prefill_head_major(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        metal_flash_attn_prefill_head_major(q, k, v, softmax_scale, causal)
    }

    /// Gather K/V from the paged pool via `index_select` on the block table,
    /// then call candle's vectorized SDPA (single-query path). The gather
    /// replaces the slow materializing `paged_cache.read` +
    /// naive-softmax+matmul fallback — same result, one fused kernel.
    fn runtime_flash_attn_paged_decode(
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
        metal_flash_attn_paged_decode(
            q,
            k_pool,
            v_pool,
            block_table,
            total_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
    }
}

// #1082 DoD-101/102: BackendRuntime decode methods flipped to kt; metal/vulkan
// impls need matching flip when their builds are restored.
#[allow(clippy::too_many_arguments)]
impl GdnBackend for MetalBackend {
    fn runtime_supports_gdn_forward_substitution(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn runtime_supports_gdn_recurrent_step(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn runtime_supports_gdn_chunk_prep(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn runtime_supports_gdn_full_chunk_forward(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn runtime_supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        !self.disable.gdn_forward_substitution
    }

    fn runtime_supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn runtime_supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        !self.disable.gdn_recurrent
    }

    fn runtime_supports_gdn_gates(&self) -> bool {
        !self.disable.gdn_gates
    }

    fn runtime_supports_gdn_gated_rms_norm(&self) -> bool {
        !self.disable.gated_rms_norm
    }

    fn runtime_gdn_forward_substitution(
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

    fn runtime_gdn_recurrent_step(
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

    fn runtime_gdn_chunk_prep(
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

    fn runtime_gdn_full_chunk_forward(
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
    fn runtime_gdn_full_chunk_forward_head_last_into(
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

    fn runtime_gdn_recurrent_prefill_head_last(
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

    fn runtime_gdn_recurrent_prefill_native_head_last(
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

    fn runtime_gdn_in_proj_decode(
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

    fn runtime_gdn_gates(
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

    fn runtime_gdn_gated_rms_norm(
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

impl LinearBackend for MetalBackend {
    fn runtime_supports_matmul_request(
        &self,
        req: &super::capability::MatmulRequest,
    ) -> super::capability::Support {
        if matmul_request_support_rank(req).is_none() {
            return super::capability::Support::Unsupported;
        }
        matmul_support_from_native(
            req.lhs_dtype == kiln_tensor::DType::BF16
                && matches!(req.epilogue, super::capability::MatmulEpilogue::Identity),
        )
    }

    fn runtime_matmul(
        &self,
        req: &super::capability::MatmulRequest,
        lhs: &kiln_tensor::Tensor,
        rhs: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if !matches!(lhs.device(), kiln_tensor::Device::Metal(_))
            || !matches!(rhs.device(), kiln_tensor::Device::Metal(_))
            || !lhs.is_contiguous()
            || !rhs.is_contiguous()
            || lhs.dtype() != kiln_tensor::DType::BF16
            || req.out_dtype != lhs.dtype()
            || req.lhs_dtype != req.rhs_dtype
        {
            return Ok(None);
        }

        let Some(layout) = requested_matmul_layout(req, lhs, rhs) else {
            return Ok(None);
        };
        let out = match layout {
            BackendMatmulLayout::Plain => kiln_tensor::metal_matmul(lhs, rhs)?,
            BackendMatmulLayout::LhsTransposed => {
                kiln_tensor::metal_matmul_lhs_transposed(lhs, rhs)?
            }
            BackendMatmulLayout::RhsTransposed => {
                kiln_tensor::metal_matmul_rhs_transposed(lhs, rhs)?
            }
            BackendMatmulLayout::BothTransposed => {
                let rank = lhs.rank();
                let lhs_t = lhs.transpose(rank - 2, rank - 1)?.contiguous()?;
                let rhs_t = rhs.transpose(rank - 2, rank - 1)?.contiguous()?;
                kiln_tensor::metal_matmul(&lhs_t, &rhs_t)?
            }
        };
        Ok(Some(out))
    }
}

impl super::residency::ResidentRegistry for MetalBackend {
    fn register_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Result<Option<super::residency::ResidentResource>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        metal_residency::register_resident_activation(&self.resident_activation_registry, tensor)
    }

    fn update_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Result<Option<super::residency::ResidentResource>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        metal_residency::update_resident_activation(&self.resident_activation_registry, tensor)
    }

    fn evict_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) {
        if family == super::residency::ResidentResourceFamily::Activation {
            metal_residency::evict_resident_activation(&self.resident_activation_registry, tensor);
        }
    }

    fn resident_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Option<super::residency::ResidentResource> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return None;
        }
        metal_residency::resident_activation_resource(&self.resident_activation_registry, tensor)
    }

    fn resolve_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        metal_residency::resolve_resident_activation(
            &self.resident_activation_registry,
            tensor,
            shape,
            dtype,
        )
    }
}

impl ResidencyBackend for MetalBackend {
    // ------------------------------------------------------------------
    // Resident-activation hooks (#1082) — Metal analog of the Vulkan
    // registry. The registry tracks membership only (the kt tensor already
    // owns its GPU buffer); `dispatch_adamw_step` runs a fused on-device
    // AdamW that updates param/m/v in place.
    // ------------------------------------------------------------------

    fn runtime_supports_resident_activation(&self) -> bool {
        true
    }
}

impl TrainingLossBackend for MetalBackend {
    fn runtime_training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn runtime_training_precision_policy(&self) -> TrainingPrecisionPolicy {
        metal_training::training_precision_policy()
    }
}

impl BackendRuntime for MetalBackend {}

#[allow(clippy::too_many_arguments)]
impl ReplayBackend for MetalBackend {
    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
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

        AttentionBackend::runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
            self,
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
}
