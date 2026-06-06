//! Metal runtime configuration and env-gated support policy.

pub(super) const DISABLE_METAL_SDPA: &str = "KILN_DISABLE_METAL_SDPA";
const DISABLE_METAL_SDPA_FULL: &str = "KILN_DISABLE_METAL_SDPA_FULL";
const DISABLE_METAL_CONV1D_PREFILL: &str = "KILN_DISABLE_METAL_CONV1D_PREFILL";
const DISABLE_FUSED_CONV1D: &str = "KILN_DISABLE_FUSED_CONV1D";
const DISABLE_METAL_FUSED_CONV1D: &str = "KILN_DISABLE_METAL_FUSED_CONV1D";
const DISABLE_GDN_KERNEL: &str = "KILN_DISABLE_GDN_KERNEL";
const DISABLE_FUSED_GDN_GATES: &str = "KILN_DISABLE_FUSED_GDN_GATES";
const DISABLE_METAL_GDN_GATES: &str = "KILN_DISABLE_METAL_GDN_GATES";
const DISABLE_METAL_GDN_FORWARD_SUBSTITUTION: &str = "KILN_DISABLE_METAL_GDN_FORWARD_SUBSTITUTION";
const DISABLE_METAL_GDN_RECURRENT: &str = "KILN_DISABLE_METAL_GDN_RECURRENT";
const DISABLE_METAL_GDN_DECODE_GATES_RECURRENT: &str =
    "KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT";
const DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM: &str =
    "KILN_DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM";
const DISABLE_METAL_GATED_RMSNORM: &str = "KILN_DISABLE_METAL_GATED_RMSNORM";
const DISABLE_METAL_GDN_QK_NORM: &str = "KILN_DISABLE_METAL_GDN_QK_NORM";
const DISABLE_METAL_GDN_QKV_CONV_NORM: &str = "KILN_DISABLE_METAL_GDN_QKV_CONV_NORM";
const DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT: &str =
    "KILN_DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT";
const DISABLE_METAL_GDN_PREFILL_DECAY_RECURRENT: &str =
    "KILN_DISABLE_METAL_GDN_PREFILL_DECAY_RECURRENT";
const DISABLE_METAL_GDN_PREFILL_AB_IN_PROJ: &str = "KILN_DISABLE_METAL_GDN_PREFILL_AB_IN_PROJ";
const DISABLE_RMSNORM_KERNEL: &str = "KILN_DISABLE_RMSNORM_KERNEL";
const DISABLE_METAL_RMSNORM: &str = "KILN_DISABLE_METAL_RMSNORM";
const DISABLE_METAL_MLP_GATE_UP_FUSION: &str = "KILN_DISABLE_METAL_MLP_GATE_UP_FUSION";
const DISABLE_METAL_MLP_GATE_UP_ROW_PAIR: &str = "KILN_DISABLE_METAL_MLP_GATE_UP_ROW_PAIR";
const DISABLE_METAL_MLP_GATE_UP_ROW_QUAD: &str = "KILN_DISABLE_METAL_MLP_GATE_UP_ROW_QUAD";
const DISABLE_METAL_MLP_GATE_UP_ROW_TRIPLE: &str = "KILN_DISABLE_METAL_MLP_GATE_UP_ROW_TRIPLE";
const DISABLE_METAL_MLP_GATE_UP_ROW_QUAD_VECTOR_LOAD: &str =
    "KILN_DISABLE_METAL_MLP_GATE_UP_ROW_QUAD_VECTOR_LOAD";
const DISABLE_METAL_MLP_GATE_UP_SERIAL_VECTOR_LOAD: &str =
    "KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_VECTOR_LOAD";
const DISABLE_METAL_MLP_GATE_UP_SERIAL_DEDICATED: &str =
    "KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_DEDICATED";
const DISABLE_METAL_MLP_SILU_MUL: &str = "KILN_DISABLE_METAL_MLP_SILU_MUL";
const DISABLE_METAL_ATTN_GATE_FUSION: &str = "KILN_DISABLE_METAL_ATTN_GATE_FUSION";
const DISABLE_METAL_FUSED_QKV_PROJ: &str = "KILN_DISABLE_METAL_FUSED_QKV_PROJ";
const DISABLE_METAL_LORA_DELTA_DECODE: &str = "KILN_DISABLE_METAL_LORA_DELTA_DECODE";
const DISABLE_METAL_GDN_IN_PROJ_FUSION: &str = "KILN_DISABLE_METAL_GDN_IN_PROJ_FUSION";
const DISABLE_METAL_GDN_IN_PROJ_ROW_PAIR: &str = "KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_PAIR";
const DISABLE_METAL_GDN_IN_PROJ_ROW_QUAD: &str = "KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_QUAD";
const DISABLE_METAL_GDN_IN_PROJ_ROW_TRIPLE: &str = "KILN_DISABLE_METAL_GDN_IN_PROJ_ROW_TRIPLE";
const DISABLE_METAL_GDN_IN_PROJ_SERIAL_VECTOR_LOAD: &str =
    "KILN_DISABLE_METAL_GDN_IN_PROJ_SERIAL_VECTOR_LOAD";
const DISABLE_METAL_GDN_IN_PROJ_SERIAL_X2_LOAD: &str =
    "KILN_DISABLE_METAL_GDN_IN_PROJ_SERIAL_X2_LOAD";
const ENABLE_METAL_LM_HEAD_ARGMAX: &str = "KILN_ENABLE_METAL_LM_HEAD_ARGMAX";
const DISABLE_METAL_LM_HEAD_ARGMAX: &str = "KILN_DISABLE_METAL_LM_HEAD_ARGMAX";
const DISABLE_METAL_LM_HEAD_ARGMAX_ROWS: &str = "KILN_DISABLE_METAL_LM_HEAD_ARGMAX_ROWS";
const DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE: &str =
    "KILN_DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE";
const DISABLE_METAL_LM_HEAD_SAMPLE: &str = "KILN_DISABLE_METAL_LM_HEAD_SAMPLE";
const DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS: &str =
    "KILN_DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS";
const DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR: &str =
    "KILN_DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV: &str = "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE16: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE16";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_PAIR: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_PAIR";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD_TILE8: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD_TILE8";
const DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_TRIPLE_TILE8: &str =
    "KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_TRIPLE_TILE8";

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
    pub(super) fn from_env() -> Self {
        let gdn_kernel = env_truthy(DISABLE_GDN_KERNEL);
        Self {
            conv1d_prefill: env_truthy(DISABLE_METAL_CONV1D_PREFILL),
            conv1d_update: env_present(DISABLE_FUSED_CONV1D)
                || env_truthy(DISABLE_METAL_FUSED_CONV1D),
            gdn_forward_substitution: gdn_kernel
                || env_truthy(DISABLE_METAL_GDN_FORWARD_SUBSTITUTION),
            gdn_recurrent: gdn_kernel || env_truthy(DISABLE_METAL_GDN_RECURRENT),
            gdn_gates: env_present(DISABLE_FUSED_GDN_GATES) || env_truthy(DISABLE_METAL_GDN_GATES),
            gated_rms_norm: env_truthy(DISABLE_METAL_GATED_RMSNORM),
            gdn_in_proj: gdn_kernel || env_truthy(DISABLE_METAL_GDN_IN_PROJ_FUSION),
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
    head_dim == 256 && !env_truthy(DISABLE_METAL_SDPA_FULL)
}

pub(super) fn metal_gdn_qk_norm_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_QK_NORM)
}

pub(super) fn metal_gdn_qkv_conv_norm_disabled() -> bool {
    env_present(DISABLE_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_GDN_QKV_CONV_NORM)
}

pub(super) fn metal_gdn_prefill_qkv_conv_split_disabled() -> bool {
    env_present(DISABLE_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_CONV1D_PREFILL)
        || env_truthy(DISABLE_METAL_FUSED_CONV1D)
        || env_truthy(DISABLE_METAL_GDN_PREFILL_QKV_CONV_SPLIT)
}

pub(super) fn metal_gdn_in_proj_row_pair_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_IN_PROJ_ROW_PAIR)
}

pub(super) fn metal_gdn_in_proj_row_quad_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_IN_PROJ_ROW_QUAD)
}

pub(super) fn metal_gdn_in_proj_row_triple_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_IN_PROJ_ROW_TRIPLE)
}

pub(super) fn metal_gdn_in_proj_serial_vector_load_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_IN_PROJ_SERIAL_VECTOR_LOAD)
}

pub(super) fn metal_gdn_in_proj_serial_x2_load_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_IN_PROJ_SERIAL_X2_LOAD)
}

pub(super) fn metal_gdn_gates_disabled() -> bool {
    env_present(DISABLE_FUSED_GDN_GATES) || env_truthy(DISABLE_METAL_GDN_GATES)
}

pub(super) fn metal_gdn_recurrent_disabled() -> bool {
    env_truthy(DISABLE_GDN_KERNEL) || env_truthy(DISABLE_METAL_GDN_RECURRENT)
}

pub(super) fn metal_gdn_prefill_decay_recurrent_disabled() -> bool {
    metal_gdn_gates_disabled()
        || metal_gdn_recurrent_disabled()
        || env_truthy(DISABLE_METAL_GDN_PREFILL_DECAY_RECURRENT)
}

pub(super) fn metal_gdn_prefill_ab_in_proj_disabled() -> bool {
    env_truthy(DISABLE_METAL_GDN_PREFILL_AB_IN_PROJ)
}

pub(super) fn metal_gdn_decode_gates_recurrent_disabled() -> bool {
    metal_gdn_gates_disabled()
        || metal_gdn_recurrent_disabled()
        || env_truthy(DISABLE_METAL_GDN_DECODE_GATES_RECURRENT)
}

pub(super) fn metal_gdn_decode_gates_recurrent_rmsnorm_disabled() -> bool {
    metal_gdn_decode_gates_recurrent_disabled()
        || env_truthy(DISABLE_METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM)
        || env_truthy(DISABLE_METAL_GATED_RMSNORM)
}

pub(super) fn metal_rms_norm_disabled() -> bool {
    env_present(DISABLE_RMSNORM_KERNEL) || env_truthy(DISABLE_METAL_RMSNORM)
}

pub(super) fn metal_mlp_gate_up_fusion_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_FUSION)
}

pub(super) fn metal_mlp_gate_up_row_pair_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_ROW_PAIR)
}

pub(super) fn metal_mlp_gate_up_row_quad_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_ROW_QUAD)
}

pub(super) fn metal_mlp_gate_up_row_triple_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_ROW_TRIPLE)
}

pub(super) fn metal_mlp_gate_up_row_quad_vector_load_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_ROW_QUAD_VECTOR_LOAD)
}

pub(super) fn metal_mlp_gate_up_serial_vector_load_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_SERIAL_VECTOR_LOAD)
}

pub(super) fn metal_mlp_gate_up_serial_dedicated_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_GATE_UP_SERIAL_DEDICATED)
}

pub(super) fn metal_mlp_silu_mul_disabled() -> bool {
    env_truthy(DISABLE_METAL_MLP_SILU_MUL)
}

pub(super) fn metal_attn_gate_fusion_disabled() -> bool {
    env_truthy(DISABLE_METAL_ATTN_GATE_FUSION)
}

pub(super) fn metal_fused_qkv_proj_disabled() -> bool {
    env_truthy(DISABLE_METAL_FUSED_QKV_PROJ) || metal_transposed_coop_gemv_tile8_disabled()
}

pub(super) fn metal_lora_delta_decode_disabled() -> bool {
    env_truthy(DISABLE_METAL_LORA_DELTA_DECODE)
}

pub(super) fn metal_lm_head_argmax_disabled() -> bool {
    // On the Qwen3.5-4B macOS desktop path, Candle's materialized last-row
    // projection plus argmax is faster than this custom chunk/reduce kernel.
    // Keep the kernel available for tuning, but require explicit opt-in.
    env_truthy(DISABLE_METAL_LM_HEAD_ARGMAX) || !env_truthy(ENABLE_METAL_LM_HEAD_ARGMAX)
}

pub(super) fn metal_lm_head_argmax_rows_disabled() -> bool {
    env_truthy(DISABLE_METAL_LM_HEAD_ARGMAX) || env_truthy(DISABLE_METAL_LM_HEAD_ARGMAX_ROWS)
}

pub(super) fn metal_lm_head_argmax_gpu_reduce_disabled() -> bool {
    env_truthy(DISABLE_METAL_LM_HEAD_ARGMAX_GPU_REDUCE)
}

pub(super) fn metal_lm_head_sample_disabled() -> bool {
    env_truthy(DISABLE_METAL_LM_HEAD_SAMPLE)
}

pub(super) fn metal_paged_attn_decode_contiguous_disabled() -> bool {
    env_truthy(DISABLE_METAL_PAGED_ATTN_DECODE_CONTIGUOUS)
}

pub(super) fn metal_paged_kv_write_token_major_disabled() -> bool {
    env_truthy(DISABLE_METAL_PAGED_KV_WRITE_TOKEN_MAJOR)
}

pub(super) fn metal_transposed_coop_gemv_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV)
}

pub(super) fn metal_transposed_coop_gemv_tile8_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE8)
}

pub(super) fn metal_transposed_coop_gemv_tile16_disabled() -> bool {
    metal_transposed_coop_gemv_tile8_disabled()
        || env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_TILE16)
}

pub(super) fn metal_transposed_coop_gemv_row_pair_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_PAIR)
}

pub(super) fn metal_transposed_coop_gemv_row_quad_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD)
}

pub(super) fn metal_transposed_coop_gemv_row_quad_tile8_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_QUAD_TILE8)
}

pub(super) fn metal_transposed_coop_gemv_row_triple_tile8_disabled() -> bool {
    env_truthy(DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_TRIPLE_TILE8)
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

fn env_present(var: &str) -> bool {
    std::env::var(var).is_ok()
}

fn env_truthy(var: &str) -> bool {
    matches!(
        std::env::var(var)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}
