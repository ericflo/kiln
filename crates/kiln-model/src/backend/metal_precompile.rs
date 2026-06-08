//! Metal custom-kernel precompile orchestration.
//!
//! Pipeline cache construction lives in `metal_pipeline.rs`; this module owns
//! the startup warmup ordering that asks those caches to build the Kiln-owned
//! Metal library and compute pipelines ahead of the first request.

use anyhow::Result;

use super::metal_config::*;
use super::metal_pipeline::*;
use kiln_tensor::metal_types::MetalCompanion;

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
