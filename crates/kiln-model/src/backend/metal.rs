//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for GDN and paged-decode.
//!
//! The chokepoint-routed `sdpa` symbol (imported at module level from the
//! kt-side re-export) is an MLX-style fused scaled-dot-product attention
//! kernel with native GQA, BF16, and head dims {32, 64, 72, 80, 96, 128,
//! 256, 512}. For typical transformer head sizes this replaces the vendored
//! CUDA FlashAttention-2 call on Apple Silicon.

use anyhow::Result;

use super::metal_config::*;
use super::metal_conv1d::{
    metal_causal_conv1d_prefill_bf16_f32_k4, metal_causal_conv1d_update_bf16_f32_k4,
    metal_conv1d_prefill_supports, metal_conv1d_update_supports,
};
pub(crate) use super::metal_dense::{
    metal_attn_gate_sigmoid_mul_bf16, metal_attn_gate_sigmoid_mul_supports,
    metal_fused_qkv_transposed_coop_gemv_bf16,
    metal_fused_qkv_transposed_coop_gemv_supports, metal_lora_add_decode_bf16,
    metal_lora_add_decode_supports, metal_mlp_gate_up_bf16, metal_mlp_gate_up_supports,
    metal_mlp_silu_mul_bf16, metal_mlp_silu_mul_supports, metal_transposed_coop_gemv_bf16,
    metal_transposed_coop_gemv_decode_batch_supports, metal_transposed_coop_gemv_supports,
};
pub(crate) use super::metal_gdn::{
    metal_gdn_decode_gates_recurrent_bf16, metal_gdn_decode_gates_recurrent_rmsnorm_bf16,
    metal_gdn_decode_gates_recurrent_rmsnorm_supports,
    metal_gdn_decode_gates_recurrent_supports, metal_gdn_decode_qkv_conv_norm_bf16,
    metal_gdn_decode_qkv_conv_norm_supports, metal_gdn_gates_decay_ab_bf16,
    metal_gdn_gates_decay_ab_supports, metal_gdn_gates_decay_bf16,
    metal_gdn_gates_decay_supports, metal_gdn_prefill_ab_in_proj_bf16,
    metal_gdn_prefill_ab_in_proj_supports, metal_gdn_prefill_qkv_conv_split_bf16_f32_k4,
    metal_gdn_prefill_qkv_conv_split_supports, metal_gdn_qk_norm_f32_bf16,
    metal_gdn_qk_norm_gqa_f32_bf16, metal_gdn_qk_norm_gqa_supports,
    metal_gdn_qk_norm_supports, metal_gdn_recurrent_prefill_native_head_last_decay_bf16,
    metal_gdn_recurrent_prefill_native_head_last_decay_supports,
};
use super::metal_gdn::{
    metal_gated_rms_norm_bf16, metal_gated_rms_norm_supports, metal_gdn_chunk_prep_bf16,
    metal_gdn_chunk_prep_supports, metal_gdn_forward_substitution_bf16,
    metal_gdn_forward_substitution_f32, metal_gdn_forward_substitution_supports,
    metal_gdn_full_chunk_forward_bf16, metal_gdn_full_chunk_forward_head_last_into_bf16,
    metal_gdn_full_chunk_forward_head_last_supports, metal_gdn_full_chunk_forward_supports,
    metal_gdn_gates_bf16, metal_gdn_gates_supports, metal_gdn_in_proj_decode_bf16,
    metal_gdn_in_proj_decode_supports, metal_gdn_recurrent_bf16, metal_gdn_recurrent_supports,
    metal_gdn_recurrent_prefill_head_last_bf16, metal_gdn_recurrent_prefill_head_last_supports,
    metal_gdn_recurrent_prefill_native_head_last_bf16,
    metal_gdn_recurrent_prefill_native_head_last_supports,
};
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
pub(crate) use super::metal_norm::{
    metal_rms_norm_bf16, metal_rms_norm_supports, metal_rotary_embedding_bf16,
    metal_rotary_embedding_supports,
};
use super::{metal_training, TrainingCapabilities};

// Phase 7 #1082: module-level imports for the kt-metal chokepoint types,
// hoisted from ~92 per-function `use` statements so that the chokepoint
// surface in this file is centralized at a single import location. Future
// substrate swaps (e.g. candle → objc2-metal) touch this single import
// block instead of hundreds of scattered fully-qualified references.
use kiln_tensor::metal_types::MetalCompanion;

#[derive(Debug)]
pub struct MetalBackend {
    /// The kt Metal device this backend dispatches on. (#1082: the
    /// formerly-retained candle `device` field is gone — every trait
    /// method is kt-native, so no candle handle is held.)
    pub(super) device_kt: kiln_tensor::Device,
    /// Cached at construction to keep env-var reads off per-token support gates.
    pub(super) disable: MetalKernelDisables,
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
