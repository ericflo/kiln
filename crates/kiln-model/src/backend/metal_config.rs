//! Metal runtime configuration derived from one immutable kernel policy.

use crate::metal_policy::current_metal_kernel_policy;

pub(super) const METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS: usize = 4;
pub(super) const METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS: usize = 8;
pub(super) const METAL_TRANSPOSED_COOP_GEMV_TILE16_COLS: usize = 16;
pub(super) const METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS: usize = 4;
pub(super) const METAL_TRANSPOSED_COOP_GEMV_THREADS: usize =
    32 * METAL_TRANSPOSED_COOP_GEMV_SIMDGROUPS;
pub(super) const METAL_LM_HEAD_SAMPLE_TOP_K_MAX: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MetalTransposedCoopGemvTile {
    Tile4,
    Tile8,
    Tile16,
}

impl MetalTransposedCoopGemvTile {
    pub(super) fn function_name(self) -> &'static str {
        match self {
            Self::Tile4 => "kiln_transposed_coop_gemv4_bf16",
            Self::Tile8 => "kiln_transposed_coop_gemv8_bf16",
            Self::Tile16 => "kiln_transposed_coop_gemv16_bf16",
        }
    }

    pub(super) fn label(self) -> &'static str {
        self.function_name()
    }

    pub(super) fn tile_cols(self) -> usize {
        match self {
            Self::Tile4 => METAL_TRANSPOSED_COOP_GEMV_TILE4_COLS,
            Self::Tile8 => METAL_TRANSPOSED_COOP_GEMV_TILE8_COLS,
            Self::Tile16 => METAL_TRANSPOSED_COOP_GEMV_TILE16_COLS,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MetalKernelDisables {
    pub(super) conv1d_prefill: bool,
    pub(super) conv1d_update: bool,
    pub(super) gdn_forward_substitution: bool,
    pub(super) gdn_recurrent: bool,
    pub(super) gdn_gates: bool,
    pub(super) gated_rms_norm: bool,
    pub(super) gdn_in_proj: bool,
}

impl MetalKernelDisables {
    pub(super) fn from_policy() -> Self {
        let policy = current_metal_kernel_policy();
        Self {
            conv1d_prefill: !policy.conv1d_prefill,
            conv1d_update: !policy.conv1d_update,
            gdn_forward_substitution: !policy.gdn_forward_substitution,
            gdn_recurrent: !policy.gdn_recurrent,
            gdn_gates: !policy.gdn_gates,
            gated_rms_norm: !policy.gated_rms_norm,
            gdn_in_proj: !policy.gdn_in_proj,
        }
    }
}

/// Mirrors the head-dim whitelist in candle-nn 0.10.2's `Sdpa::custom_op3`.
/// The fallback path absorbs future upstream list drift correctly, just slower.
pub(super) fn metal_sdpa_supports_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
}

/// Qwen3.5-4B prefill path safety gate for the full Metal SDPA kernel.
pub(super) fn metal_sdpa_full_safe_for_q_seq(head_dim: usize, q_seq: usize) -> bool {
    if q_seq <= 8 {
        return true;
    }
    head_dim == 256 && current_metal_kernel_policy().sdpa_full
}

pub(super) fn metal_gdn_qk_norm_disabled() -> bool {
    !current_metal_kernel_policy().gdn_qk_norm
}

pub(super) fn metal_gdn_qkv_conv_norm_disabled() -> bool {
    let policy = current_metal_kernel_policy();
    !policy.conv1d_update || !policy.gdn_qkv_conv_norm
}

pub(super) fn metal_gdn_prefill_qkv_conv_split_disabled() -> bool {
    let policy = current_metal_kernel_policy();
    !policy.conv1d_update || !policy.conv1d_prefill || !policy.gdn_prefill_qkv_conv_split
}

pub(super) fn metal_gdn_in_proj_row_pair_disabled() -> bool {
    !current_metal_kernel_policy().gdn_in_proj_row_pair
}

pub(super) fn metal_gdn_in_proj_row_quad_disabled() -> bool {
    !current_metal_kernel_policy().gdn_in_proj_row_quad
}

pub(super) fn metal_gdn_in_proj_row_triple_disabled() -> bool {
    !current_metal_kernel_policy().gdn_in_proj_row_triple
}

pub(super) fn metal_gdn_in_proj_serial_vector_load_disabled() -> bool {
    !current_metal_kernel_policy().gdn_in_proj_serial_vector_load
}

pub(super) fn metal_gdn_in_proj_serial_x2_load_disabled() -> bool {
    !current_metal_kernel_policy().gdn_in_proj_serial_x2_load
}

pub(super) fn metal_gdn_gates_disabled() -> bool {
    !current_metal_kernel_policy().gdn_gates
}

pub(super) fn metal_gdn_recurrent_disabled() -> bool {
    !current_metal_kernel_policy().gdn_recurrent
}

pub(super) fn metal_gdn_prefill_decay_recurrent_disabled() -> bool {
    metal_gdn_gates_disabled()
        || metal_gdn_recurrent_disabled()
        || !current_metal_kernel_policy().gdn_prefill_decay_recurrent
}

pub(super) fn metal_gdn_prefill_ab_in_proj_disabled() -> bool {
    !current_metal_kernel_policy().gdn_prefill_ab_in_proj
}

pub(super) fn metal_gdn_decode_gates_recurrent_disabled() -> bool {
    metal_gdn_gates_disabled()
        || metal_gdn_recurrent_disabled()
        || !current_metal_kernel_policy().gdn_decode_gates_recurrent
}

pub(super) fn metal_gdn_decode_gates_recurrent_rmsnorm_disabled() -> bool {
    metal_gdn_decode_gates_recurrent_disabled()
        || !current_metal_kernel_policy().gdn_decode_gates_recurrent_rmsnorm
        || !current_metal_kernel_policy().gated_rms_norm
}

pub(super) fn metal_rms_norm_disabled() -> bool {
    !current_metal_kernel_policy().rms_norm
}

pub(crate) fn metal_mlp_gate_up_fusion_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_fusion
}

pub(super) fn metal_mlp_gate_up_row_pair_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_row_pair
}

pub(super) fn metal_mlp_gate_up_row_quad_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_row_quad
}

pub(super) fn metal_mlp_gate_up_row_triple_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_row_triple
}

pub(super) fn metal_mlp_gate_up_row_quad_vector_load_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_row_quad_vector_load
}

pub(super) fn metal_mlp_gate_up_serial_vector_load_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_serial_vector_load
}

pub(super) fn metal_mlp_gate_up_serial_dedicated_disabled() -> bool {
    !current_metal_kernel_policy().mlp_gate_up_serial_dedicated
}

pub(super) fn metal_mlp_silu_mul_disabled() -> bool {
    !current_metal_kernel_policy().mlp_silu_mul
}

pub(super) fn metal_attn_gate_fusion_disabled() -> bool {
    !current_metal_kernel_policy().attn_gate_fusion
}

pub(super) fn metal_fused_qkv_proj_disabled() -> bool {
    !current_metal_kernel_policy().fused_qkv_proj || metal_transposed_coop_gemv_tile8_disabled()
}

pub(super) fn metal_lora_delta_decode_disabled() -> bool {
    !current_metal_kernel_policy().lora_delta_decode
}

pub(super) fn metal_lm_head_argmax_disabled() -> bool {
    // On the Qwen3.5-4B macOS desktop path, the portable materialized
    // last-row projection plus argmax is faster than this custom
    // chunk/reduce kernel.
    // Keep the kernel available for tuning, but require explicit opt-in.
    !current_metal_kernel_policy().lm_head_argmax
}

pub(super) fn metal_lm_head_argmax_rows_disabled() -> bool {
    !current_metal_kernel_policy().lm_head_argmax_rows
}

pub(super) fn metal_lm_head_argmax_gpu_reduce_disabled() -> bool {
    !current_metal_kernel_policy().lm_head_argmax_gpu_reduce
}

pub(super) fn metal_lm_head_sample_disabled() -> bool {
    !current_metal_kernel_policy().lm_head_sample
}

pub(super) fn metal_paged_attn_decode_contiguous_disabled() -> bool {
    !current_metal_kernel_policy().paged_attn_decode_contiguous
}

pub(super) fn metal_paged_kv_write_token_major_disabled() -> bool {
    !current_metal_kernel_policy().paged_kv_write_token_major
}

pub(super) fn metal_transposed_coop_gemv_disabled() -> bool {
    !current_metal_kernel_policy().transposed_coop_gemv
}

pub(super) fn metal_transposed_coop_gemv_tile8_disabled() -> bool {
    !current_metal_kernel_policy().transposed_coop_gemv_tile8
}

pub(super) fn metal_transposed_coop_gemv_tile16_disabled() -> bool {
    metal_transposed_coop_gemv_tile8_disabled()
        || !current_metal_kernel_policy().transposed_coop_gemv_tile16
}

pub(super) fn metal_transposed_coop_gemv_row_pair_disabled() -> bool {
    !current_metal_kernel_policy().transposed_coop_gemv_row_pair
}

pub(super) fn metal_transposed_coop_gemv_row_quad_disabled() -> bool {
    !current_metal_kernel_policy().transposed_coop_gemv_row_quad
}

pub(super) fn metal_transposed_coop_gemv_row_quad_tile8_disabled() -> bool {
    !current_metal_kernel_policy().transposed_coop_gemv_row_quad_tile8
}

pub(super) fn metal_transposed_coop_gemv_row_triple_tile8_disabled() -> bool {
    !current_metal_kernel_policy().transposed_coop_gemv_row_triple_tile8
}

pub(super) fn metal_transposed_coop_gemv_default_tile() -> MetalTransposedCoopGemvTile {
    if metal_transposed_coop_gemv_tile8_disabled() {
        MetalTransposedCoopGemvTile::Tile4
    } else {
        MetalTransposedCoopGemvTile::Tile8
    }
}

pub(super) fn metal_transposed_coop_gemv_select_tile(
    input_dim: usize,
    output_dim: usize,
) -> MetalTransposedCoopGemvTile {
    let default_tile = metal_transposed_coop_gemv_default_tile();
    if default_tile == MetalTransposedCoopGemvTile::Tile8
        && !metal_transposed_coop_gemv_tile16_disabled()
        // Qwen3.5-4B MLP down projection. The wider tile regressed smaller
        // attention-output GEMVs, so keep the selector shape-specific.
        && input_dim == 9216
        && output_dim == 2560
    {
        MetalTransposedCoopGemvTile::Tile16
    } else {
        default_tile
    }
}
