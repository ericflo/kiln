use crate::buffer::VulkanBuffer;
use crate::device::VulkanDevice;
use anyhow::{Context, Result};
use ash::vk;
use half::bf16;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

const DEFAULT_MLP_BF16_DOWN_ROWS4_MIN_BATCH: usize = 16;
const DEFAULT_MLP_BF16_GATE_UP_ROWS4_MIN_BATCH: usize = 8;
const DEFAULT_MLP_BF16_ROWS8_MIN_BATCH: usize = 256;
const DEFAULT_MLP_F32_DOWN_ROWS4_MIN_BATCH: usize = 8;
const DEFAULT_FULL_ATTN_QKV_BF16_ROWS4_MIN_BATCH: usize = 2;
const DEFAULT_LINEAR_DECODE_BF16W_ROWS4_MIN_BATCH: usize = 16;
const DEFAULT_LINEAR_DECODE_BF16W_ROWS8_MIN_BATCH: usize = 64;
const DEFAULT_GDN_IN_PROJ_ROWS4_MIN_BATCH: usize = 16;
const DEFAULT_GDN_IN_PROJ_ROWS8_MIN_BATCH: usize = 64;

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn profile_vulkan_mlp_kernel_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("KILN_PROFILE_VULKAN_MLP_KERNEL_STAGES"))
}

pub(crate) fn mlp_bf16_gate_up_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_GATE_UP_ROWS4").is_err())
}

pub(crate) fn mlp_f32_down_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_F32_DOWN_ROWS4").is_err())
}

pub(crate) fn mlp_bf16_down_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_DOWN_ROWS4").is_err())
}

pub(crate) fn mlp_bf16_rows8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_ROWS8").is_err())
}

pub(crate) fn mlp_bf16_rows8_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_MLP_BF16_ROWS8_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MLP_BF16_ROWS8_MIN_BATCH)
    })
}

pub(crate) fn mlp_bf16_gate_up_rows4_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_MLP_BF16_GATE_UP_ROWS4_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MLP_BF16_GATE_UP_ROWS4_MIN_BATCH)
    })
}

pub(crate) fn mlp_bf16_down_rows4_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_MLP_BF16_DOWN_ROWS4_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MLP_BF16_DOWN_ROWS4_MIN_BATCH)
    })
}

pub(crate) fn mlp_f32_down_rows4_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_MLP_F32_DOWN_ROWS4_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_MLP_F32_DOWN_ROWS4_MIN_BATCH)
    })
}

pub(crate) fn linear_decode_bf16w_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE_BF16W_ROWS4").is_err()
            && std::env::var("KILN_DISABLE_VULKAN_LINEAR_BF16W_ROWS4").is_err()
    })
}

pub(crate) fn linear_decode_bf16w_rows8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE_BF16W_ROWS8").is_err()
            && std::env::var("KILN_DISABLE_VULKAN_LINEAR_BF16W_ROWS8").is_err()
    })
}

pub(crate) fn linear_decode_bf16w_rows8_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_LINEAR_BF16_ROWS8_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_LINEAR_DECODE_BF16W_ROWS8_MIN_BATCH)
    })
}

pub(crate) fn linear_decode_bf16w_rows4_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_LINEAR_BF16_ROWS4_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_LINEAR_DECODE_BF16W_ROWS4_MIN_BATCH)
    })
}

pub(crate) fn gdn_in_proj_rows4_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_GDN_IN_PROJ_ROWS4_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_GDN_IN_PROJ_ROWS4_MIN_BATCH)
    })
}

pub(crate) fn gdn_in_proj_rows8_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_GDN_IN_PROJ_ROWS8_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_GDN_IN_PROJ_ROWS8_MIN_BATCH)
    })
}

pub(crate) fn full_attn_qkv_bf16w_rows4_min_batch() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("KILN_VULKAN_FULL_ATTN_QKV_BF16_ROWS4_MIN_BATCH")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_FULL_ATTN_QKV_BF16_ROWS4_MIN_BATCH)
    })
}

const PAGED_ATTN_SPLITK_CHUNKS_B1: usize = 32;
const PAGED_ATTN_SPLITK_CHUNKS_BATCHED: usize = 4;
const PAGED_ATTN_SPLITK_CHUNKS_BATCHED_LONG: usize = 2;
const PAGED_ATTN_SPLITK_LONG_MIN_BLOCKS: usize = 64;
pub(crate) const PAGED_ATTN_SPLITK_MAX_CHUNKS: usize = 256;

pub fn paged_attn_decode_splitk_chunks(batch_size: usize, max_blocks_per_seq: usize) -> usize {
    std::env::var("KILN_VK_PAGED_ATTN_SPLITK_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(if batch_size <= 1 {
            PAGED_ATTN_SPLITK_CHUNKS_B1
        } else if batch_size >= 16 {
            PAGED_ATTN_SPLITK_CHUNKS_BATCHED
        } else if max_blocks_per_seq >= PAGED_ATTN_SPLITK_LONG_MIN_BLOCKS {
            PAGED_ATTN_SPLITK_CHUNKS_BATCHED_LONG
        } else {
            PAGED_ATTN_SPLITK_CHUNKS_BATCHED
        })
        .min(PAGED_ATTN_SPLITK_MAX_CHUNKS)
}

fn paged_attn_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_PAGED_ATTN_SINGLE_SUBMIT").is_err()
    })
}

fn qwen_rmsnorm_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_QWEN_RMSNORM_SINGLE_SUBMIT").is_err()
    })
}

fn gdn_gates_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_GATES_SINGLE_SUBMIT").is_err()
    })
}

fn gdn_gated_norm_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_GATED_NORM_SINGLE_SUBMIT").is_err()
    })
}

fn mlp_gate_up_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_MLP_GATE_UP_SINGLE_SUBMIT").is_err()
    })
}

fn causal_conv1d_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_CAUSAL_CONV1D_SINGLE_SUBMIT").is_err()
    })
}

pub(crate) fn full_attn_qkv_bf16w_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV_BF16W_ROWS4").is_err()
    })
}

fn mlp_chained_dispatch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_CHAINED_DISPATCH").is_err())
}

fn mlp_chained_transfer_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_CHAINED_TRANSFER_SUBMIT").is_err())
}

fn profile_vulkan_gdn_in_proj_kernel_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy("KILN_PROFILE_VULKAN_GDN_IN_PROJ_KERNEL_STAGES"))
}

fn profile_vulkan_gdn_recurrent_kernel_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| env_truthy("KILN_PROFILE_VULKAN_GDN_RECURRENT_KERNEL_STAGES"))
}

#[allow(clippy::too_many_arguments)]
fn finish_vulkan_mlp_kernel_stage_profile(
    stage: &str,
    batch: usize,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    gate_up_bf16_weights: bool,
    down_bf16_weights: bool,
    gate_up_rows2: bool,
    gate_up_rows4: bool,
    down_rows4: bool,
    down_rows2: bool,
    start: Option<Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_vulkan_mlp_kernel_stage stage={stage} batch={batch} hidden={hidden} intermediate={intermediate} out_dim={out_dim} bf16_weights={gate_up_bf16_weights} down_bf16_weights={down_bf16_weights} rows2={gate_up_rows2} gate_up_rows4={gate_up_rows4} down_rows4={down_rows4} down_rows2={down_rows2} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

#[allow(clippy::too_many_arguments)]
fn finish_vulkan_gdn_in_proj_kernel_stage_profile(
    stage: &str,
    batch: usize,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    packed_bf16_weights: bool,
    pair_qkv_z: bool,
    row_group_size: usize,
    single_submit: bool,
    start: Option<Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_vulkan_gdn_in_proj_kernel_stage stage={stage} batch={batch} hidden={hidden} qkv_dim={qkv_dim} z_dim={z_dim} a_dim={a_dim} b_dim={b_dim} packed_bf16_weights={packed_bf16_weights} pair_qkv_z={pair_qkv_z} row_group_size={row_group_size} single_submit={single_submit} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

#[allow(clippy::too_many_arguments)]
fn finish_vulkan_gdn_recurrent_kernel_stage_profile(
    stage: &str,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    parallel_reduce: bool,
    single_submit: bool,
    skip_state_readback: bool,
    start: Option<Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_vulkan_gdn_recurrent_kernel_stage stage={stage} batch={batch} heads={heads} dk={dk} dv={dv} parallel_reduce={parallel_reduce} single_submit={single_submit} skip_state_readback={skip_state_readback} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

fn gdn_decode_host_visible_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_ENABLE_VULKAN_GDN_HOST_VISIBLE_STATE").is_ok())
}

fn gdn_decode_fused_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_ENABLE_VULKAN_GDN_DECODE_FUSED_SINGLE_SUBMIT").is_ok())
}

fn gdn_recurrent_host_visible_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_STATE").is_err()
    })
}

fn gdn_recurrent_host_visible_batch_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_ENABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_BATCH_STATE").is_ok()
    })
}

fn gdn_recurrent_use_host_visible_state(batch: usize) -> bool {
    gdn_recurrent_host_visible_state_enabled()
        && (batch == 1 || gdn_recurrent_host_visible_batch_state_enabled())
}

fn gdn_recurrent_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_SINGLE_SUBMIT").is_err())
}

fn gdn_recurrent_parallel_reduce_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_PARALLEL_REDUCE").is_err())
}

pub(crate) fn use_gdn_recurrent_parallel_reduce(dk: usize, dv: usize) -> bool {
    dk >= 32 && dv > 0 && gdn_recurrent_parallel_reduce_enabled()
}

fn linear_decode_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE_SINGLE_SUBMIT").is_err())
}

fn linear_decode_argmax_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_LINEAR_ARGMAX_SINGLE_SUBMIT").is_err())
}

fn full_attn_qkv_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV_SINGLE_SUBMIT").is_err())
}

fn gdn_in_proj_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_SINGLE_SUBMIT").is_err())
}

pub(crate) fn gdn_in_proj_batch_pair_qkv_z_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_PAIR_QKV_Z").is_err())
}

pub(crate) fn gdn_in_proj_batch_row_pair_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_PAIR").is_err())
}

pub(crate) fn gdn_in_proj_batch_row_quad_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_QUAD").is_err())
}

pub(crate) fn gdn_in_proj_batch_row_octet_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env_truthy("KILN_ENABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_OCTET")
            && std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_OCTET").is_err()
    })
}

fn gdn_gates_batched_transfers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_GATES_BATCHED_TRANSFERS").is_err())
}

fn gdn_gated_norm_batched_uploads_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_GATED_NORM_BATCHED_UPLOADS").is_err()
    })
}

fn gdn_chunk_batched_transfers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNK_BATCHED_TRANSFERS").is_err())
}

fn paged_attn_batched_uploads_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_PAGED_ATTN_BATCHED_UPLOADS").is_err())
}

fn prefill_row_pair_matmul_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_PREFILL_ROW_PAIR_MATMUL").is_err())
}

pub(crate) fn use_prefill_row_pair_matmul(batch: usize) -> bool {
    batch >= 8 && prefill_row_pair_matmul_enabled()
}

/// Pre-create the validated built-in compute pipelines on this Vulkan device.
///
/// SPIR-V bytecode is embedded at build time when `glslc` is available. This
/// function fills the per-device pipeline cache so the first live request does
/// not pay RADV pipeline creation latency on the decode path.
pub fn prewarm_builtin_pipelines(vk_device: &VulkanDevice) -> Result<()> {
    let shaders = [
        ("full_attn_qkv_decode", 5usize, 20u32),
        ("full_attn_qkv_decode_bf16w", 5usize, 20u32),
        ("full_attn_qkv_decode_batched", 5usize, 24u32),
        ("full_attn_qkv_decode_batched_bf16w", 5usize, 24u32),
        ("full_attn_qkv_decode_batched_rows4_bf16w", 5usize, 24u32),
        ("full_attn_qkv_gate_split_batched_bf16w", 8usize, 28u32),
        (
            "full_attn_qkv_gate_split_batched_rows4_bf16w",
            8usize,
            28u32,
        ),
        (
            "full_attn_qkv_gate_split_batched_rows8_bf16w",
            8usize,
            28u32,
        ),
        ("vk_rope_f32", 4, 16),
        ("vk_rope_qk_f32", 6, 20),
        ("vk_rope_q_kv_write_slots_f32", 9, 24),
        ("vk_mul_sigmoid_gate_f32", 3, 4),
        ("qkv_gate_split_batched", 5usize, 16u32),
        ("gdn_gates", 6usize, 8u32),
        ("gdn_decode_gates_recurrent_rmsnorm", 11, 20),
        ("gdn_in_proj_decode", 6, 24),
        ("gdn_in_proj_decode_bf16w", 6, 24),
        ("gdn_in_proj_decode_batched", 6, 28),
        ("gdn_in_proj_decode_batched_bf16w", 6, 28),
        ("gdn_in_proj_decode_batched_pair_qkv_z_bf16w", 6, 28),
        ("gdn_in_proj_split_batched", 5, 20),
        ("gdn_decode_conv_split_batched", 9, 32),
        ("gdn_qkv_split_batched", 4, 12),
        ("gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w", 6, 28),
        ("gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w", 6, 28),
        ("gdn_in_proj_decode_batched_pair_qkv_z_rows8_bf16w", 6, 28),
        ("gdn_gated_rms_norm", 4, 12),
        ("causal_conv1d", 4, 16),
        ("causal_conv1d_state_advance", 2, 16),
        ("gdn_recurrent_prefill", 7, 24),
        ("gdn_recurrent_step_parallel", 7, 24),
        ("gdn_recurrent_qk_norm_step", 7, 24),
        ("l2_norm_per_row", 2, 20),
        ("l2_norm_qk_per_row", 4, 24),
        ("gdn_chunk_prep", 12, 16),
        ("gdn_chunk_scan", 8, 16),
        ("qwen_rmsnorm_forward", 3, 12),
        ("qwen_rmsnorm_qk_combined", 4, 16),
        ("linear_decode", 3, 8),
        ("add_qwen_rmsnorm", 5, 8),
        ("add_qwen_rmsnorm_batched", 5, 12),
        ("linear_decode_bf16w", 3, 8),
        ("linear_decode_bf16w_add_residual", 4, 8),
        ("linear_decode_batched", 3, 12),
        ("linear_decode_batched_bf16w", 3, 12),
        ("linear_decode_batched_bf16w_add_residual", 4, 12),
        ("linear_decode_batched_bf16w_add_residual_rows4", 4, 12),
        ("linear_decode_batched_bf16w_add_residual_rows8", 4, 12),
        ("linear_decode_batched_rows2", 3, 12),
        ("linear_decode_batched_rows4", 3, 12),
        ("linear_decode_batched_rows4_bf16w", 3, 12),
        ("linear_decode_batched_rows8_bf16w", 3, 12),
        ("linear_decode_argmax_blocks", 4, 12),
        ("linear_decode_argmax_blocks_bf16w", 4, 12),
        ("linear_decode_argmax_reduce", 3, 4),
        ("linear_decode_argmax_batched_blocks", 4, 12),
        ("linear_decode_argmax_batched_blocks_bf16w", 4, 12),
        ("linear_decode_argmax_batched_blocks_rows4_bf16w", 4, 16),
        ("linear_decode_argmax_batched_blocks_rows8_bf16w", 4, 16),
        ("linear_decode_argmax_batched_reduce", 3, 4),
        ("mlp_gate_up_decode", 4, 8),
        ("mlp_gate_up_decode_bf16w", 4, 8),
        ("mlp_gate_up_decode_batched", 4, 12),
        ("mlp_gate_up_decode_batched_bf16w", 4, 12),
        ("mlp_gate_up_decode_batched_rows4_bf16w", 4, 12),
        ("mlp_gate_up_decode_batched_rows8_bf16w", 4, 12),
        ("mlp_gate_up_decode_batched_rows2", 4, 12),
        ("paged_kv_write_slot", 4, 8),
        ("paged_kv_write_slots", 5, 12),
        ("paged_attn_decode_batch", 5, 20),
        ("paged_attn_decode_batch_paged", 6, 24),
        ("paged_attn_decode_batch_paged_splitk", 6, 28),
        ("paged_attn_decode_batch_paged_splitk_reduce", 2, 12),
    ];

    for (shader_name, total_bindings, push_bytes) in shaders {
        let glsl_path = format!(
            "{}/csrc/shaders/{}.comp",
            env!("CARGO_MANIFEST_DIR"),
            shader_name
        );
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(&glsl_path)
            .with_context(|| format!("compile Vulkan shader {shader_name}"))?;
        vk_device
            .get_or_create_compute_pipeline(&spirv, total_bindings, push_bytes)
            .with_context(|| format!("create Vulkan pipeline {shader_name}"))?;
    }

    let command_batch_shaders = [
        (crate::shaders::QWEN_RMSNORM_FORWARD, 3usize, 12u32),
        (crate::shaders::QWEN_RMSNORM_QK_COMBINED, 4, 16),
        (crate::shaders::FULL_ATTN_QKV_DECODE_BF16W, 5, 20),
        (crate::shaders::FULL_ATTN_QKV_DECODE_BATCHED_BF16W, 5, 24),
        (
            crate::shaders::FULL_ATTN_QKV_DECODE_BATCHED_ROWS4_BF16W,
            5,
            24,
        ),
        (
            crate::shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_BF16W,
            8,
            28,
        ),
        (
            crate::shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W,
            8,
            28,
        ),
        (
            crate::shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS8_BF16W,
            8,
            28,
        ),
        (crate::shaders::QKV_GATE_SPLIT, 5, 16),
        (crate::shaders::QKV_GATE_SPLIT_BATCHED, 5, 16),
        (crate::shaders::VK_ROPE_F32, 4, 16),
        (crate::shaders::VK_ROPE_QK_F32, 6, 20),
        (crate::shaders::VK_ROPE_Q_KV_WRITE_SLOTS_F32, 9, 24),
        (crate::shaders::PAGED_KV_WRITE_SLOT, 4, 8),
        (crate::shaders::PAGED_KV_WRITE_SLOTS, 5, 12),
        (crate::shaders::PAGED_ATTN_DECODE_BATCH_PAGED, 6, 24),
        (
            crate::shaders::PAGED_ATTN_DECODE_BATCH_PAGED_SPLITK,
            6,
            28,
        ),
        (
            crate::shaders::PAGED_ATTN_DECODE_BATCH_PAGED_SPLITK_REDUCE,
            2,
            12,
        ),
        (crate::shaders::VK_MUL_SIGMOID_GATE_F32, 3, 4),
        (crate::shaders::LINEAR_DECODE_BF16W, 3, 8),
        (crate::shaders::LINEAR_DECODE_BF16W_ADD_RESIDUAL, 4, 8),
        (crate::shaders::LINEAR_DECODE_BATCHED_BF16W, 3, 12),
        (crate::shaders::LINEAR_DECODE_BATCHED_ROWS4_BF16W, 3, 12),
        (
            crate::shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL,
            4,
            12,
        ),
        (
            crate::shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL_ROWS4,
            4,
            12,
        ),
        (
            crate::shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL_ROWS8,
            4,
            12,
        ),
        (crate::shaders::ADD_QWEN_RMSNORM, 5, 8),
        (crate::shaders::ADD_QWEN_RMSNORM_BATCHED, 5, 12),
        (crate::shaders::MLP_GATE_UP_DECODE_BF16W, 4, 8),
        (crate::shaders::MLP_GATE_UP_DECODE_BATCHED_BF16W, 4, 12),
        (
            crate::shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS4_BF16W,
            4,
            12,
        ),
        (
            crate::shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS8_BF16W,
            4,
            12,
        ),
        (crate::shaders::GDN_IN_PROJ_DECODE_BF16W, 6, 24),
        (crate::shaders::GDN_IN_PROJ_SPLIT, 5, 20),
        (crate::shaders::GDN_IN_PROJ_DECODE_BATCHED_BF16W, 6, 28),
        (
            crate::shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_BF16W,
            6,
            28,
        ),
        (
            crate::shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W,
            6,
            28,
        ),
        (
            crate::shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W,
            6,
            28,
        ),
        (
            crate::shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS8_BF16W,
            6,
            28,
        ),
        (crate::shaders::GDN_IN_PROJ_SPLIT_BATCHED, 5, 20),
        (crate::shaders::GDN_DECODE_CONV_SPLIT_BATCHED, 9, 32),
        (crate::shaders::GDN_QKV_SPLIT, 4, 12),
        (crate::shaders::GDN_QKV_SPLIT_BATCHED, 4, 12),
        (crate::shaders::CAUSAL_CONV1D, 4, 16),
        (crate::shaders::CAUSAL_CONV1D_STATE_ADVANCE, 2, 16),
        (crate::shaders::L2_NORM_QK_PER_ROW, 4, 24),
        (crate::shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM, 11, 20),
    ];

    for (shader_path, total_bindings, push_bytes) in command_batch_shaders {
        vk_device
            .get_compute_pipeline_by_path(shader_path, total_bindings, push_bytes)
            .with_context(|| format!("prewarm Vulkan command-batch path {shader_path}"))?;
    }

    let chunkwise_prefill_paths = [
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_narrow_lastdim_f32.comp"),
            2usize,
            16u32,
        ),
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/vk_narrow_lastdim_f32_offset.comp"
            ),
            2,
            24,
        ),
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/vk_narrow_lastdim_bwd_f32.comp"
            ),
            2,
            16,
        ),
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/vk_narrow_lastdim_bwd_f32_offset.comp"
            ),
            2,
            24,
        ),
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_matmul_batched_f32.comp"),
            3,
            16,
        ),
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_transpose_3d_f32.comp"),
            2,
            12,
        ),
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/gdn_chunk_prep.comp"),
            12,
            16,
        ),
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_solve_tri_v2.comp"),
            4,
            16,
        ),
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/vk_broadcast_mul_lastdim.comp"
            ),
            3,
            8,
        ),
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/vk_elementwise_binary_f32.comp"
            ),
            3,
            8,
        ),
        (
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/vk_elementwise_binary_f32_offset.comp"
            ),
            3,
            12,
        ),
    ];

    for (shader_path, total_bindings, push_bytes) in chunkwise_prefill_paths {
        vk_device
            .get_compute_pipeline_by_path(shader_path, total_bindings, push_bytes)
            .with_context(|| format!("prewarm Vulkan chunkwise-prefill path {shader_path}"))?;
    }

    Ok(())
}

/// Candle-free Vulkan compute dispatch — takes raw input byte slices and
/// returns the output buffer as raw bytes. Manages the full lifecycle:
/// create buffers, upload inputs, dispatch, read back output. This is
/// the canonical SPIR-V dispatch entry point for #1082 callers that
/// want no candle types in scope. (#1082)
///
/// `output_elem_size` is the per-element byte size of the output buffer
/// (4 for f32, 2 for bf16/f16, 8 for f64).
pub fn dispatch_kernel_bytes(
    vk_device: &VulkanDevice,
    spirv: &[u8],
    push_constants: &[u32],
    workgroup_count: (u32, u32, u32),
    inputs: &[&[u8]],
    output_shape: &[usize],
    output_elem_size: usize,
) -> Result<Vec<u8>> {
    // Per-axis dispatch grid limit. Use the actual device limit
    // (typically ≈ 2^31 - 1 on AMD/Strix Halo) rather than the
    // Vulkan spec minimum (65535), so we don't bail on legitimate
    // dispatches that the hardware can handle.
    let limit_x = vk_device.max_compute_work_group_count(0);
    let limit_y = vk_device.max_compute_work_group_count(1);
    let limit_z = vk_device.max_compute_work_group_count(2);
    anyhow::ensure!(
        workgroup_count.0 <= limit_x
            && workgroup_count.1 <= limit_y
            && workgroup_count.2 <= limit_z,
        "dispatch_kernel_bytes: workgroup_count {:?} exceeds device per-axis \
         limits ({}, {}, {})",
        workgroup_count,
        limit_x,
        limit_y,
        limit_z
    );
    let device = vk_device.device();
    let queue = vk_device.queue();
    let queue_family_index = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // --- Create output buffer ---
    let elem_count: usize = output_shape.iter().product();
    let output_size = (elem_count * output_elem_size) as u64;
    let output_buffer = VulkanBuffer::create_device_local(device, device_local_mt, output_size)
        .context("failed to create output buffer")?;

    // --- Create input buffers + upload ---
    let mut input_buffers: Vec<VulkanBuffer> = Vec::with_capacity(inputs.len());
    for data in inputs {
        let buf = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .context("failed to create input buffer")?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            queue_family_index,
            &buf,
            data,
        )
        .context("failed to upload input data")?;
        input_buffers.push(buf);
    }

    // --- Build combined binding list (inputs first, then output) ---
    let total_bindings = input_buffers.len() + 1;
    tracing::trace!(
        total_bindings,
        inputs = inputs.len(),
        "Vulkan dispatch start"
    );
    let mut all_handles: Vec<vk::Buffer> = Vec::with_capacity(total_bindings);
    for buf in &input_buffers {
        all_handles.push(buf.handle());
    }
    all_handles.push(output_buffer.handle());

    // --- Shader module ---
    // Copy bytes into a fresh Vec<u32> for guaranteed u32 alignment
    // (see device.rs:601 — `bytemuck::cast_slice::<u8,u32>(spirv)`
    // panics on misaligned input, hit by CI run 26353268949).
    let spirv_words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shader_module_info = vk::ShaderModuleCreateInfo::default()
        .code(&spirv_words)
        ;
    let shader_module = unsafe {
        device
            .create_shader_module(&shader_module_info, None)
            .context("failed to create shader module")?
    };

    // --- Descriptor set layout (STORAGE_BUFFER for all bindings) ---
    let desc_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..total_bindings as u32)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                
        })
        .collect();

    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&desc_bindings)
        ;
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(&set_layout_info, None)
            .context("failed to create descriptor set layout")?
    };

    // --- Pipeline layout ---
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size((push_constants.len() * 4) as u32)
        ;
    let pcr = vec![push_constant_range];
    let set_layouts = vec![set_layout];

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&pcr)
        ;
    let layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .context("failed to create pipeline layout")?
    };

    // --- Compute pipeline ---
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
        ;

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(layout)
        ;

    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(errs, _)| {
                if !errs.is_empty() {
                    anyhow::anyhow!("failed to create compute pipeline: {:?}", errs[0])
                } else {
                    anyhow::anyhow!("failed to create compute pipeline")
                }
            })?
    };
    let pipeline = pipelines[0];

    // --- Descriptor pool + set (STORAGE_BUFFER) ---
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(total_bindings as u32)
        ;
    let pool_sizes = vec![pool_size];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes)
        ;
    let pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .context("failed to create descriptor pool")?
    };

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts)
        ;
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .context("failed to allocate descriptor sets")?
    };
    let descriptor_set = descriptor_sets[0];

    // --- Descriptor writes using STORAGE_BUFFER (no buffer views needed) ---
    {
        // Build DescriptorBufferInfo entries (must outlive WriteDescriptorSet)
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&buf_handle| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf_handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();

        let descriptor_write_infos: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bui)| make_write_descriptor_set_buf(descriptor_set, i as u32, bui))
            .collect();

        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    // --- Command buffer + dispatch ---
    let cmd_pool_info = make_cmd_pool_info(queue_family_index);
    let cmd_pool = unsafe {
        device
            .create_command_pool(&cmd_pool_info, None)
            .context("failed to create command pool")?
    };

    let alloc_info = make_cmd_alloc_info(cmd_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    let begin_info = make_cmd_begin_info();
    unsafe {
        device
            .begin_command_buffer(cmd, &begin_info)
            .context("failed to begin command buffer")?;

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );

        let push_const_bytes = bytemuck::cast_slice(push_constants);
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_const_bytes,
        );

        device.cmd_dispatch(cmd, workgroup_count.0, workgroup_count.1, workgroup_count.2);

        // Memory barrier: flush compute writes so readback sees them
        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }

    // --- Submit + wait ---
    let cmds = vec![cmd];
    let submit_info = make_submit_info(&cmds);
    unsafe {
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .context("failed to submit compute dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for queue")?;
    }

    // --- Read back output ---
    let output_data = VulkanBuffer::read_back(
        device,
        host_visible_mt,
        queue,
        queue_family_index,
        &output_buffer,
    )
    .context("failed to read back output")?;

    // --- Cleanup (input_buffers and output_buffer dropped here) ---
    drop(input_buffers);
    drop(output_buffer);

    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_shader_module(shader_module, None);
        device.free_command_buffers(cmd_pool, &command_buffers);
        device.destroy_command_pool(cmd_pool, None);
    }
    tracing::trace!("Vulkan dispatch complete");

    Ok(output_data)
}

/// CommandPoolCreateInfo via ash 0.38 default+chained builder.
fn make_cmd_pool_info(queue_family_index: u32) -> vk::CommandPoolCreateInfo<'static> {
    vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
}

/// CommandBufferAllocateInfo via default+chained builder.
fn make_cmd_alloc_info(pool: vk::CommandPool) -> vk::CommandBufferAllocateInfo<'static> {
    vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1)
}

/// CommandBufferBeginInfo via default+chained builder.
fn make_cmd_begin_info() -> vk::CommandBufferBeginInfo<'static> {
    vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
}

/// SubmitInfo via default+chained builder.
fn make_submit_info(cmds: &[vk::CommandBuffer]) -> vk::SubmitInfo<'_> {
    vk::SubmitInfo::default().command_buffers(cmds)
}

fn upload_buffers_with_command_pool(
    device: &Arc<ash::Device>,
    host_mem_type: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    uploads: &[(&VulkanBuffer, &[u8])],
) -> Result<()> {
    if uploads.is_empty() {
        return Ok(());
    }

    let mut staging = Vec::with_capacity(uploads.len());
    for (_, data) in uploads {
        let stage = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &stage, data)?;
        staging.push(stage);
    }

    let alloc_info = make_cmd_alloc_info(command_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate batched transfer command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin batched transfer command buffer")?;
        for ((dst, data), stage) in uploads.iter().zip(staging.iter()) {
            device.cmd_copy_buffer(
                cmd,
                stage.handle(),
                dst.handle(),
                &[vk::BufferCopy::default().size(data.len() as u64)],
            );
        }
        device
            .end_command_buffer(cmd)
            .context("failed to end batched transfer command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batched transfer")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batched transfer")?;
        device.free_command_buffers(command_pool, &command_buffers);
    }

    Ok(())
}

fn read_back_buffers_with_command_pool(
    device: &Arc<ash::Device>,
    host_mem_type: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    buffers: &[&VulkanBuffer],
) -> Result<Vec<Vec<u8>>> {
    if buffers.is_empty() {
        return Ok(Vec::new());
    }

    let mut staging = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        staging.push(VulkanBuffer::create_host_visible(
            device,
            host_mem_type,
            buffer.size(),
        )?);
    }

    let alloc_info = make_cmd_alloc_info(command_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate batched readback command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin batched readback command buffer")?;
        for (src, stage) in buffers.iter().zip(staging.iter()) {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                stage.handle(),
                &[vk::BufferCopy::default().size(src.size())],
            );
        }
        device
            .end_command_buffer(cmd)
            .context("failed to end batched readback command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batched readback")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batched readback")?;
        device.free_command_buffers(command_pool, &command_buffers);
    }

    staging
        .iter()
        .map(|stage| VulkanBuffer::read_host_visible(device, stage))
        .collect()
}

/// Create a zero-init WriteDescriptorSet for STORAGE_BUFFER with fixed sType.
fn make_write_descriptor_set_buf(
    dst_set: vk::DescriptorSet,
    dst_binding: u32,
    bui: &vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'_> {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;

    let mut info: MaybeUninit<vk::WriteDescriptorSet> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::WRITE_DESCRIPTOR_SET;
        (*ptr).dst_set = dst_set;
        (*ptr).dst_binding = dst_binding;
        (*ptr).descriptor_count = 1;
        (*ptr).descriptor_type = vk::DescriptorType::STORAGE_BUFFER;
        (*ptr).p_image_info = std::ptr::null();
        (*ptr).p_buffer_info = bui as *const _;
        (*ptr).p_texel_buffer_view = std::ptr::null();
    }
    unsafe { info.assume_init() }
}

/// Create a zero-init MemoryBarrier with fixed sType.
fn make_memory_barrier(
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::MemoryBarrier<'static> {
    vk::MemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)
}

// ===========================================================================
// Candle-free buffer uploads
// ===========================================================================
//
// The candle ↔ Vulkan bridge module (`crate::candle_bridge`) is gone as
// of the final stage of #1082. Its byte-level core,
// `upload_bytes_to_device_buffer`, lives below as a file-private helper
// shared by the public `upload_*_buffer_from_slice` entry points. (#1082)

/// Upload raw bytes as immutable weights into a device-local Vulkan
/// buffer using a transient command pool. Shared core for the
/// candle-free `upload_*_buffer_from_slice` helpers below. (#1082)
fn upload_bytes_to_device_buffer(
    vk_device: &VulkanDevice,
    bytes: &[u8],
    create_ctx: &'static str,
    upload_ctx: &'static str,
) -> Result<VulkanBuffer> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let buffer = VulkanBuffer::create_device_local(device, device_local_mt, bytes.len() as u64)
        .context(create_ctx)?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &buffer,
            bytes,
        )
        .context(upload_ctx)?;
    }
    Ok(buffer)
}

/// Upload an f32 slice as a contiguous immutable weight buffer.
/// Candle-free entry point; callers with host-side f32 data skip the
/// candle Tensor staging step entirely. (#1082)
pub fn upload_f32_buffer_from_slice(
    vk_device: &VulkanDevice,
    data: &[f32],
) -> Result<VulkanBuffer> {
    upload_bytes_to_device_buffer(
        vk_device,
        bytemuck::cast_slice(data),
        "failed to create cached tensor buffer",
        "failed to upload cached tensor buffer",
    )
}

/// Upload a bf16 slice as packed immutable weights into a Vulkan
/// buffer. Two bf16 lanes are packed per u32 word (`(hi << 16) | lo`)
/// to match the `*_bf16w.comp` shader variants. Candle-free entry
/// point. (#1082)
pub fn upload_bf16_packed_buffer_from_slice(
    vk_device: &VulkanDevice,
    data: &[bf16],
) -> Result<VulkanBuffer> {
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));
    for pair in data.chunks(2) {
        let lo = pair[0].to_bits() as u32;
        let hi = pair.get(1).map(|v| v.to_bits() as u32).unwrap_or(0);
        packed.push(lo | (hi << 16));
    }
    upload_bytes_to_device_buffer(
        vk_device,
        bytemuck::cast_slice(&packed),
        "failed to create cached packed bf16 tensor buffer",
        "failed to upload cached packed bf16 tensor buffer",
    )
}

/// Candle-free f32 weights variant of the fused single-token GDN input
/// projection kernel with cached weights.
///
/// Same semantics as the bf16-packed _bytes variant but for f32 weights.
/// (#1082)
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_in_proj_decode_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let out_data = dispatch_gdn_in_proj_decode_cached_impl(
        vk_device,
        x_data,
        batch,
        qkv_weight_t,
        z_weight_t,
        a_weight_t,
        b_weight_t,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        false,
    )?;
    split_gdn_in_proj_bytes(&out_data, batch, qkv_dim, z_dim, a_dim, b_dim)
}

/// Candle-free bf16-packed weights variant of [`dispatch_gdn_in_proj_decode_cached`].
///
/// Takes `x` as raw f32 bytes `[batch, 1, hidden]` and returns the
/// `(qkv, z, a, b)` outputs as raw f32 bytes — each shaped
/// `[batch, 1, *_dim]` in row-major order. The shim reconstructs a CPU
/// Tensor internally so callers can stay candle-free. (#1082)
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_in_proj_decode_cached_bf16_weights_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let out_data = dispatch_gdn_in_proj_decode_cached_impl(
        vk_device,
        x_data,
        batch,
        qkv_weight_t,
        z_weight_t,
        a_weight_t,
        b_weight_t,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        true,
    )?;
    split_gdn_in_proj_bytes(&out_data, batch, qkv_dim, z_dim, a_dim, b_dim)
}

fn dispatch_gdn_in_proj_decode_cached_impl(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    packed_bf16_weights: bool,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    let profile_stages = profile_vulkan_gdn_in_proj_kernel_stages_enabled();
    let total_start = profile_stages.then(Instant::now);
    let total_out = qkv_dim + z_dim + a_dim + b_dim;
    let pair_qkv_z = batch > 1 && gdn_in_proj_batch_pair_qkv_z_enabled();
    let row_grouping =
        packed_bf16_weights && pair_qkv_z && batch >= 3 && gdn_in_proj_batch_row_pair_enabled();
    let row_group_size = if row_grouping
        && batch >= gdn_in_proj_rows8_min_batch()
        && gdn_in_proj_batch_row_octet_enabled()
    {
        8usize
    } else if row_grouping
        && batch >= gdn_in_proj_rows4_min_batch()
        && gdn_in_proj_batch_row_quad_enabled()
    {
        4usize
    } else if row_grouping {
        2usize
    } else {
        1usize
    };
    let dispatch_cols = if pair_qkv_z {
        qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim
    } else {
        total_out
    };
    let single_submit = gdn_in_proj_single_submit_enabled();
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "gdn_in_proj_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );

    let glsl_path = if batch == 1 {
        if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode.comp"
            )
        }
    } else {
        if packed_bf16_weights {
            if row_group_size == 8 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows8_bf16w.comp"
                )
            } else if row_group_size == 4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w.comp"
                )
            } else if row_group_size == 2 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w.comp"
                )
            } else if pair_qkv_z {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_bf16w.comp"
                )
            }
        } else if pair_qkv_z {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched.comp"
            )
        }
    };
    let stage_start = profile_stages.then(Instant::now);
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "shader",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    let mut push_constants = vec![
        hidden as u32,
        qkv_dim as u32,
        z_dim as u32,
        a_dim as u32,
        b_dim as u32,
        total_out as u32,
    ];
    if batch > 1 {
        push_constants.push(batch as u32);
    }
    if single_submit {
        return dispatch_gdn_in_proj_decode_cached_single_submit(
            vk_device,
            qkv_weight_t,
            z_weight_t,
            a_weight_t,
            b_weight_t,
            batch,
            hidden,
            qkv_dim,
            z_dim,
            a_dim,
            b_dim,
            total_out,
            dispatch_cols,
            row_group_size,
            packed_bf16_weights,
            pair_qkv_z,
            profile_stages,
            total_start,
            &spirv,
            &push_constants,
            &x_data,
        );
    }

    let stage_start = profile_stages.then(Instant::now);
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create gdn_in_proj x buffer")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_x_buffer",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    {
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload gdn_in_proj x buffer")?;
        finish_vulkan_gdn_in_proj_kernel_stage_profile(
            "upload_x",
            batch,
            hidden,
            qkv_dim,
            z_dim,
            a_dim,
            b_dim,
            packed_bf16_weights,
            pair_qkv_z,
            row_group_size,
            single_submit,
            stage_start,
        );
    }

    let stage_start = profile_stages.then(Instant::now);
    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * total_out * 4) as u64)
            .context("failed to create gdn_in_proj output buffer")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_out_buffer",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    let all_handles = vec![
        x_buf.handle(),
        qkv_weight_t.handle(),
        z_weight_t.handle(),
        a_weight_t.handle(),
        b_weight_t.handle(),
        out_buf.handle(),
    ];

    let stage_start = profile_stages.then(Instant::now);
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        if batch == 1 {
            total_out.div_ceil(16) as u32
        } else if row_group_size > 1 {
            (batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32
        } else {
            (batch * dispatch_cols.div_ceil(80)) as u32
        },
    )
    .context("gdn_in_proj_decode kernel failed")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "dispatch",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );

    let out_data = {
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back gdn_in_proj output")?;
        finish_vulkan_gdn_in_proj_kernel_stage_profile(
            "readback",
            batch,
            hidden,
            qkv_dim,
            z_dim,
            a_dim,
            b_dim,
            packed_bf16_weights,
            pair_qkv_z,
            row_group_size,
            single_submit,
            stage_start,
        );
        out_data
    };

    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "total",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        total_start,
    );
    let _ = (qkv_dim, z_dim, a_dim, b_dim); // shape metadata consumed at the `_bytes` shim boundary
    Ok(out_data)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_in_proj_decode_cached_single_submit(
    vk_device: &VulkanDevice,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    total_out: usize,
    dispatch_cols: usize,
    row_group_size: usize,
    packed_bf16_weights: bool,
    pair_qkv_z: bool,
    profile_stages: bool,
    total_start: Option<Instant>,
    spirv: &[u8],
    push_constants: &[u32],
    x_data: &[u8],
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let stage_start = profile_stages.then(Instant::now);
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create gdn_in_proj x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create gdn_in_proj x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_x_stage_write",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let out_size = (batch * total_out * 4) as u64;
    let stage_start = profile_stages.then(Instant::now);
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create gdn_in_proj output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create gdn_in_proj output staging buffer")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_out_buffers",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let stage_start = profile_stages.then(Instant::now);
    let all_handles = vec![
        x_buf.handle(),
        qkv_weight_t.handle(),
        z_weight_t.handle(),
        a_weight_t.handle(),
        b_weight_t.handle(),
        out_buf.handle(),
    ];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate gdn_in_proj descriptor set")?[0]
    };
    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "pipeline_descriptor_setup",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    let stage_start = profile_stages.then(Instant::now);
    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::default().size(x_data.len() as u64)],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(
            cmd,
            if batch == 1 {
                total_out.div_ceil(16) as u32
            } else if row_group_size > 1 {
                (batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32
            } else {
                (batch * dispatch_cols.div_ceil(80)) as u32
            },
            1,
            1,
        );
        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn_in_proj single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn_in_proj single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "record_submit_wait",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let stage_start = profile_stages.then(Instant::now);
    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "read_host_visible",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "total",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        total_start,
    );
    Ok(out_data)
}

/// Split the contiguous gdn_in_proj output bytes into per-dim byte
/// slices for the `_bytes` shim callers. The shader writes the dims in
/// SoA order: all qkv batches first, then all z, then all a, then all
/// b. So the split is purely contiguous — `qkv = out[..batch*qkv_dim*4]`,
/// etc. Replaces the older candle-Tensor split helper. (#1082)
fn split_gdn_in_proj_bytes(
    out_data: &[u8],
    batch: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let qkv_bytes = batch * qkv_dim * 4;
    let z_bytes = batch * z_dim * 4;
    let a_bytes = batch * a_dim * 4;
    let b_bytes = batch * b_dim * 4;
    let total = qkv_bytes + z_bytes + a_bytes + b_bytes;
    anyhow::ensure!(
        out_data.len() >= total,
        "gdn_in_proj_decode output slice exceeds readback buffer"
    );
    let mut offset = 0usize;
    let mut take = |len: usize| -> Vec<u8> {
        let end = offset + len;
        let slice = out_data[offset..end].to_vec();
        offset = end;
        slice
    };
    let qkv = take(qkv_bytes);
    let z = take(z_bytes);
    let a = take(a_bytes);
    let b = take(b_bytes);
    Ok((qkv, z, a, b))
}

pub fn dispatch_linear_decode_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<Vec<u8>> {
    dispatch_linear_decode_cached_bytes_core(
        vk_device,
        x_data,
        weight_t,
        batch,
        hidden,
        out_dim,
        packed_bf16_weights,
    )
}

pub fn dispatch_linear_decode_cached_bf16_weights_offset_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_buffer: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    weight_offset: usize,
    full_out_dim: usize,
) -> Result<Vec<u8>> {
    dispatch_linear_decode_cached_bf16_weights_offset_bytes_core(
        vk_device,
        x_data,
        weight_buffer,
        batch,
        hidden,
        out_dim,
        weight_offset,
        full_out_dim,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_linear_decode_cached_bf16_weights_offset_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_buffer: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    weight_offset: usize,
    full_out_dim: usize,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "linear_decode_offset: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    anyhow::ensure!(
        weight_offset + out_dim <= full_out_dim,
        "weight_offset({}) + out_dim({}) overflows full_out_dim({})",
        weight_offset,
        out_dim,
        full_out_dim,
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_offset x buffer")?;
    let out_size = (batch * out_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create linear_decode_offset output buffer")?;

    let all_handles = vec![x_buf.handle(), weight_buffer.handle(), out_buf.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_batched_offset_bf16w.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        hidden as u32,
        out_dim as u32,
        batch as u32,
        weight_offset as u32,
        full_out_dim as u32,
    ];
    let workgroups = (batch * out_dim.div_ceil(32)) as u32;

    if linear_decode_single_submit_enabled() {
        run_compute_pipeline_with_transfer_readback(
            vk_device,
            &x_buf,
            x_data,
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            workgroups,
        )
        .context("linear_decode_batched_offset_bf16w single-submit kernel failed")
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                x_data,
            )
            .context("failed to upload linear_decode_offset x buffer")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("linear_decode_batched_offset_bf16w kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back linear_decode_offset output")
    }
}

pub fn dispatch_qwen_rmsnorm_forward_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<u8>> {
    dispatch_qwen_rmsnorm_forward_bytes_core(vk_device, x_data, weight_data, rows, hidden, eps)
}

fn dispatch_qwen_rmsnorm_forward_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        x_data.len() == rows * hidden * 4,
        "qwen_rmsnorm_forward: x buffer has {} bytes, expected {}",
        x_data.len(),
        rows * hidden * 4
    );
    anyhow::ensure!(
        weight_data.len() == hidden * 4,
        "qwen_rmsnorm_forward: weight buffer has {} bytes, expected {}",
        weight_data.len(),
        hidden * 4
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("qwen_rmsnorm_forward: create x buffer")?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)
            .context("qwen_rmsnorm_forward: create weight buffer")?;
    let out_size = x_data.len() as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("qwen_rmsnorm_forward: create out buffer")?;

    let all_handles = vec![x_buf.handle(), weight_buf.handle(), out_buf.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    // Push constants: rows, hidden, eps. eps is f32 transmuted to u32 bits.
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];
    let workgroups = rows as u32;

    if qwen_rmsnorm_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[(&x_buf, x_data), (&weight_buf, weight_data)],
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            workgroups,
        )
        .context("qwen_rmsnorm_forward: single-submit dispatch")
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                x_data,
            )
            .context("qwen_rmsnorm_forward: upload x")?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &weight_buf,
                weight_data,
            )
            .context("qwen_rmsnorm_forward: upload weight")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("qwen_rmsnorm_forward: kernel dispatch")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("qwen_rmsnorm_forward: read back out")
    }
}

pub fn dispatch_qwen_rmsnorm_backward_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    grad_y_data: &[u8],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<u8>> {
    dispatch_qwen_rmsnorm_backward_bytes_core(
        vk_device,
        x_data,
        weight_data,
        grad_y_data,
        rows,
        hidden,
        eps,
    )
}

fn dispatch_qwen_rmsnorm_backward_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    grad_y_data: &[u8],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<u8>> {
    anyhow::ensure!(
        x_data.len() == rows * hidden * 4,
        "qwen_rmsnorm_backward: x buffer has {} bytes, expected {}",
        x_data.len(),
        rows * hidden * 4
    );
    anyhow::ensure!(
        weight_data.len() == hidden * 4,
        "qwen_rmsnorm_backward: weight buffer has {} bytes, expected {}",
        weight_data.len(),
        hidden * 4
    );
    anyhow::ensure!(
        grad_y_data.len() == rows * hidden * 4,
        "qwen_rmsnorm_backward: grad_y buffer has {} bytes, expected {}",
        grad_y_data.len(),
        rows * hidden * 4
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let out_len = x_data.len();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("qwen_rmsnorm_backward: create x buffer")?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)
            .context("qwen_rmsnorm_backward: create weight buffer")?;
    let grad_y_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, grad_y_data.len() as u64)
            .context("qwen_rmsnorm_backward: create grad_y buffer")?;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_len as u64)
        .context("qwen_rmsnorm_backward: create out buffer")?;

    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            x_data,
        )
        .context("qwen_rmsnorm_backward: upload x")?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &weight_buf,
            weight_data,
        )
        .context("qwen_rmsnorm_backward: upload weight")?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &grad_y_buf,
            grad_y_data,
        )
        .context("qwen_rmsnorm_backward: upload grad_y")?;
    }

    let all_handles = vec![
        x_buf.handle(),
        weight_buf.handle(),
        grad_y_buf.handle(),
        out_buf.handle(),
    ];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_backward.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        rows as u32,
    )
    .context("qwen_rmsnorm_backward: kernel dispatch")?;

    let command_pool = vk_device.transient_command_pool()?;
    VulkanBuffer::read_back_with_command_pool(
        device,
        host_visible_mt,
        queue,
        *command_pool,
        &out_buf,
    )
    .context("qwen_rmsnorm_backward: read back grad_x")
}

pub fn dispatch_linear_decode_cached_bf16_weights_transposed_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_buffer: &VulkanBuffer,
    batch: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<Vec<u8>> {
    dispatch_linear_decode_cached_bf16_weights_transposed_bytes_core(
        vk_device,
        x_data,
        weight_buffer,
        batch,
        k_dim,
        n_dim,
    )
}

fn dispatch_linear_decode_cached_bf16_weights_transposed_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_buffer: &VulkanBuffer,
    batch: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        x_data.len() == batch * k_dim * 4,
        "linear_decode_transposed: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * k_dim * 4
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_transposed x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            x_data,
        )
        .context("failed to upload linear_decode_transposed x buffer")?;
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * n_dim * 4) as u64)
            .context("failed to create linear_decode_transposed output buffer")?;

    let all_handles = vec![x_buf.handle(), weight_buffer.handle(), out_buf.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_batched_transposed_bf16w.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 3] = [k_dim as u32, n_dim as u32, batch as u32];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        (batch * n_dim.div_ceil(32)) as u32,
    )
    .context("linear_decode_batched_transposed_bf16w kernel failed")?;

    let command_pool = vk_device.transient_command_pool()?;
    VulkanBuffer::read_back_with_command_pool(
        device,
        host_visible_mt,
        queue,
        *command_pool,
        &out_buf,
    )
    .context("failed to read back linear_decode_transposed output")
}

fn dispatch_linear_decode_cached_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "linear_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    if linear_decode_single_submit_enabled() {
        return dispatch_linear_decode_cached_single_submit_bytes(
            vk_device,
            weight_t,
            batch,
            hidden,
            out_dim,
            x_data,
            packed_bf16_weights,
        );
    }

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            x_data,
        )
        .context("failed to upload linear_decode x buffer")?;
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * out_dim * 4) as u64)
            .context("failed to create linear_decode output buffer")?;

    let all_handles = vec![x_buf.handle(), weight_t.handle(), out_buf.handle()];
    if batch == 1 {
        let glsl_path = if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode.comp"
            )
        };
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
        let push_constants: [u32; 2] = [hidden as u32, out_dim as u32];
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            out_dim.div_ceil(16) as u32,
        )
        .context("linear_decode kernel failed")?;
    } else {
        let rows4 = packed_bf16_weights
            && batch >= linear_decode_bf16w_rows4_min_batch()
            && linear_decode_bf16w_rows4_enabled();
        let glsl_path = if packed_bf16_weights {
            if rows4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_bf16w.comp"
                )
            }
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched.comp"
            )
        };
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
        let push_constants: [u32; 3] = [hidden as u32, out_dim as u32, batch as u32];
        let workgroups = if rows4 {
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
        } else {
            (batch * out_dim.div_ceil(32)) as u32
        };
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("linear_decode_batched kernel failed")?;
    }

    let command_pool = vk_device.transient_command_pool()?;
    VulkanBuffer::read_back_with_command_pool(
        device,
        host_visible_mt,
        queue,
        *command_pool,
        &out_buf,
    )
    .context("failed to read back linear_decode output")
}

fn dispatch_linear_decode_cached_single_submit_bytes(
    vk_device: &VulkanDevice,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    x_data: &[u8],
    packed_bf16_weights: bool,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create linear_decode x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;

    let out_size = (batch * out_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create linear_decode output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create linear_decode output staging buffer")?;

    let (spirv, push_constants, workgroup_count): (Vec<u8>, Vec<u32>, u32) = if batch == 1 {
        let glsl_path = if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode.comp"
            )
        };
        (
            crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?,
            vec![hidden as u32, out_dim as u32],
            out_dim.div_ceil(16) as u32,
        )
    } else {
        let rows4 = packed_bf16_weights
            && batch >= linear_decode_bf16w_rows4_min_batch()
            && linear_decode_bf16w_rows4_enabled();
        let glsl_path = if packed_bf16_weights {
            if rows4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_bf16w.comp"
                )
            }
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched.comp"
            )
        };
        let workgroups = if rows4 {
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
        } else {
            (batch * out_dim.div_ceil(32)) as u32
        };
        (
            crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?,
            vec![hidden as u32, out_dim as u32, batch as u32],
            workgroups,
        )
    };

    let all_handles = vec![x_buf.handle(), weight_t.handle(), out_buf.handle()];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };
    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::default().size(x_data.len() as u64)],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);
        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit linear_decode single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for linear_decode single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &out_stage)
}

/// Dispatch a single-token transposed linear projection and return argmax.
///
/// This is intended for greedy LM-head decode: the full vocab logits stay on
/// the Vulkan device and only the winning token id is read back.
pub fn dispatch_linear_decode_argmax_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
) -> Result<u32> {
    dispatch_linear_decode_argmax_cached_impl_bytes(
        vk_device,
        x_data,
        weight_t,
        hidden,
        out_dim,
        false,
    )
}

pub fn dispatch_linear_decode_argmax_cached_bf16_weights_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
) -> Result<u32> {
    dispatch_linear_decode_argmax_cached_impl_bytes(
        vk_device,
        x_data,
        weight_t,
        hidden,
        out_dim,
        true,
    )
}

fn dispatch_linear_decode_argmax_cached_impl_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<u32> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(out_dim > 0, "linear argmax: out_dim must be nonzero");
    anyhow::ensure!(
        x_data.len() == hidden * 4,
        "linear argmax: x buffer has {} bytes, expected {}",
        x_data.len(),
        hidden * 4
    );
    if linear_decode_argmax_single_submit_enabled() {
        return dispatch_linear_decode_argmax_cached_single_submit(
            vk_device,
            weight_t,
            hidden,
            out_dim,
            &x_data,
            packed_bf16_weights,
        );
    }

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear argmax x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload linear argmax x buffer")?;
    }

    let block_count = out_dim.div_ceil(16);
    let block_score_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block score buffer")?;
    let block_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block index buffer")?;
    let out_index_buf = VulkanBuffer::create_device_local(device, device_local_mt, 4)
        .context("failed to create linear argmax output index buffer")?;

    let blocks_glsl = if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks.comp"
        )
    };
    let blocks_spirv = crate::pipeline::ShaderPipeline::compile_shader(blocks_glsl)?;
    let block_push: [u32; 3] = [hidden as u32, out_dim as u32, block_count as u32];
    let block_handles = vec![
        x_buf.handle(),
        weight_t.handle(),
        block_score_buf.handle(),
        block_index_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &blocks_spirv,
        &block_handles,
        block_handles.len(),
        &block_push,
        block_count as u32,
    )
    .context("linear_decode_argmax block kernel failed")?;

    let reduce_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_argmax_reduce.comp"
    );
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl)?;
    let reduce_push: [u32; 1] = [block_count as u32];
    let reduce_handles = vec![
        block_score_buf.handle(),
        block_index_buf.handle(),
        out_index_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &reduce_spirv,
        &reduce_handles,
        reduce_handles.len(),
        &reduce_push,
        1,
    )
    .context("linear_decode_argmax reduce kernel failed")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_index_buf,
        )
        .context("failed to read back linear argmax output index")?
    };
    let indices: &[u32] = bytemuck::cast_slice(&out_data);
    indices
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("linear argmax readback was empty"))
}

fn dispatch_linear_decode_argmax_cached_single_submit(
    vk_device: &VulkanDevice,
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
    x_data: &[u8],
    packed_bf16_weights: bool,
) -> Result<u32> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear argmax x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create linear argmax x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;

    let block_count = out_dim.div_ceil(16);
    let block_score_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block score buffer")?;
    let block_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block index buffer")?;
    let out_index_buf = VulkanBuffer::create_device_local(device, device_local_mt, 4)
        .context("failed to create linear argmax output index buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, 4)
        .context("failed to create linear argmax output staging buffer")?;

    let blocks_glsl = if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks.comp"
        )
    };
    let blocks_spirv = crate::pipeline::ShaderPipeline::compile_shader(blocks_glsl)?;
    let block_push: [u32; 3] = [hidden as u32, out_dim as u32, block_count as u32];
    let block_handles = vec![
        x_buf.handle(),
        weight_t.handle(),
        block_score_buf.handle(),
        block_index_buf.handle(),
    ];
    let (block_set_layout, block_layout, block_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &blocks_spirv,
            block_handles.len(),
            (block_push.len() * 4) as u32,
        )?;

    let reduce_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_argmax_reduce.comp"
    );
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl)?;
    let reduce_push: [u32; 1] = [block_count as u32];
    let reduce_handles = vec![
        block_score_buf.handle(),
        block_index_buf.handle(),
        out_index_buf.handle(),
    ];
    let (reduce_set_layout, reduce_layout, reduce_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &reduce_spirv,
            reduce_handles.len(),
            (reduce_push.len() * 4) as u32,
        )?;

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let block_set_layouts = vec![block_set_layout];
    let block_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&block_set_layouts)
                    ,
            )
            .context("failed to allocate linear argmax block descriptor set")?[0]
    };
    let block_buf_infos: Vec<vk::DescriptorBufferInfo> = block_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let block_descriptor_writes: Vec<vk::WriteDescriptorSet> = block_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(block_descriptor_set, i as u32, info))
        .collect();

    let reduce_set_layouts = vec![reduce_set_layout];
    let reduce_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&reduce_set_layouts)
                    ,
            )
            .context("failed to allocate linear argmax reduce descriptor set")?[0]
    };
    let reduce_buf_infos: Vec<vk::DescriptorBufferInfo> = reduce_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let reduce_descriptor_writes: Vec<vk::WriteDescriptorSet> = reduce_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(reduce_descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&block_descriptor_writes, &[]);
        device.update_descriptor_sets(&reduce_descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::default().size(x_data.len() as u64)],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, block_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            block_layout,
            0,
            &[block_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            block_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&block_push),
        );
        device.cmd_dispatch(cmd, block_count as u32, 1, 1);

        let block_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[block_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, reduce_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            reduce_layout,
            0,
            &[reduce_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            reduce_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&reduce_push),
        );
        device.cmd_dispatch(cmd, 1, 1, 1);

        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_index_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(4)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit linear argmax single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for linear argmax single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let indices: &[u32] = bytemuck::cast_slice(&out_data);
    indices
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("linear argmax readback was empty"))
}

/// Single-token transposed linear projection + full Qwen3.5 stochastic
/// sampling, fully fused on the Vulkan device. **Returns only the 4-byte
/// sampled token id — the full-vocab logits never leave GPU memory.**
///
/// Pipeline (all on-device):
/// 1. lm_head matmul → `logits` device buffer of size `[out_dim]`.
/// 2. (Optional) `apply_token_penalties` scatter applies repetition,
///    presence, and frequency penalties at history token indices.
/// 3. `topk_sample` fused kernel does temperature + top-k + softmax +
///    min-p + top-p + seeded categorical sample. Writes 1 u32 token.
/// 4. Read back 4 bytes.
///
/// This is the Vulkan equivalent of CUDA/Metal's on-device sampling
/// path. Replaces the legacy "linear_decode + full vocab readback +
/// host sampler" flow for non-greedy decode steps.
///
/// `top_k` must be ≤ `TOPK_SAMPLE_KERNEL_K_MAX` (= 64). Callers should
/// fall back to the legacy host path for larger top_k requests.
pub const TOPK_SAMPLE_KERNEL_K_MAX: u32 = 64;

#[allow(clippy::too_many_arguments)]
pub fn dispatch_linear_decode_sample_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    packed_bf16_weights: bool,
    hidden: usize,
    out_dim: usize,
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
) -> Result<u32> {
    let device = vk_device.device();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(out_dim > 0, "linear_decode_sample: out_dim must be nonzero");
    anyhow::ensure!(
        top_k > 0 && top_k <= TOPK_SAMPLE_KERNEL_K_MAX,
        "linear_decode_sample: top_k {top_k} out of range (1..={})",
        TOPK_SAMPLE_KERNEL_K_MAX
    );
    anyhow::ensure!(
        history_indices.len() == history_counts.len(),
        "linear_decode_sample: history indices/counts length mismatch ({} vs {})",
        history_indices.len(),
        history_counts.len()
    );
    anyhow::ensure!(
        x_data.len() == hidden * 4,
        "linear_decode_sample: x buffer has {} bytes, expected {}",
        x_data.len(),
        hidden * 4
    );

    // ---- Allocate the device-local buffers ----
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_sample x buffer")?;
    // Logits buffer is `[out_dim]` f32. Stays on device for the entire
    // pipeline — never copied back to host.
    let logits_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (out_dim * 4) as u64)
            .context("failed to create linear_decode_sample logits buffer")?;

    // ---- Step 2: prepare optional penalty buffers before recording ----
    let penalties_active = !history_indices.is_empty()
        && ((repetition_penalty.is_finite() && (repetition_penalty - 1.0).abs() > f32::EPSILON)
            || (presence_penalty.is_finite() && presence_penalty != 0.0)
            || (frequency_penalty.is_finite() && frequency_penalty != 0.0));
    let _history_idx_buf;
    let _history_cnt_buf;
    let mut upload_segments = Vec::with_capacity(if penalties_active { 3 } else { 1 });
    upload_segments.push(x_data);
    if penalties_active {
        let idx_buf = VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            (history_indices.len() * 4) as u64,
        )
        .context("failed to create penalty history-index buffer")?;
        let cnt_buf = VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            (history_counts.len() * 4) as u64,
        )
        .context("failed to create penalty history-count buffer")?;
        upload_segments.push(bytemuck::cast_slice(history_indices));
        upload_segments.push(bytemuck::cast_slice(history_counts));

        // Keep buffers alive until the command batch has completed.
        _history_idx_buf = Some(idx_buf);
        _history_cnt_buf = Some(cnt_buf);
    } else {
        _history_idx_buf = None;
        _history_cnt_buf = None;
    }
    let (upload_stage, upload_offsets) =
        VulkanBuffer::create_host_visible_with_segments(device, host_visible_mt, &upload_segments)
            .context("failed to stage linear_decode_sample uploads")?;

    // ---- Step 3: record lm_head + optional penalties + sample + readback ----
    let out_token_buf = VulkanBuffer::create_device_local(device, device_local_mt, 4)
        .context("failed to create linear_decode_sample out-token buffer")?;
    let out_staging = VulkanBuffer::create_host_visible(device, host_visible_mt, 4)
        .context("failed to create linear_decode_sample output staging buffer")?;

    let lm_glsl = if packed_bf16_weights {
        crate::shaders::LINEAR_DECODE_BF16W
    } else {
        crate::shaders::LINEAR_DECODE
    };
    let mut batch = crate::CommandBatch::new(vk_device)
        .context("linear_decode_sample: create CommandBatch")?;
    let mut upload_copies = Vec::with_capacity(if penalties_active { 3 } else { 1 });
    upload_copies.push((
        &upload_stage,
        &x_buf,
        upload_offsets[0],
        0,
        x_data.len() as u64,
    ));
    if let (Some(idx_buf), Some(cnt_buf)) = (&_history_idx_buf, &_history_cnt_buf) {
        upload_copies.push((
            &upload_stage,
            idx_buf,
            upload_offsets[1],
            0,
            (history_indices.len() * 4) as u64,
        ));
        upload_copies.push((
            &upload_stage,
            cnt_buf,
            upload_offsets[2],
            0,
            (history_counts.len() * 4) as u64,
        ));
    }
    batch
        .record_upload_buffer_regions(&upload_copies)
        .context("linear_decode_sample: record uploads")?;
    batch
        .record_shader(
            lm_glsl,
            &[x_buf.handle(), weight_t.handle(), logits_buf.handle()],
            &[hidden as u32, out_dim as u32],
            crate::Workgroups::OneD(out_dim.div_ceil(16) as u32),
        )
        .context("linear_decode_sample: record lm_head dispatch")?;

    if let (Some(idx_buf), Some(cnt_buf)) = (&_history_idx_buf, &_history_cnt_buf) {
        let n_unique = history_indices.len() as u32;
        batch
            .record_shader(
                crate::shaders::APPLY_TOKEN_PENALTIES,
                &[logits_buf.handle(), idx_buf.handle(), cnt_buf.handle()],
                &[
                    n_unique,
                    out_dim as u32,
                    repetition_penalty.to_bits(),
                    presence_penalty.to_bits(),
                    frequency_penalty.to_bits(),
                ],
                crate::Workgroups::OneD(n_unique.div_ceil(64)),
            )
            .context("linear_decode_sample: record apply_token_penalties dispatch")?;
    }

    let seed_lo = (seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (seed >> 32) as u32;
    // Push constants: u32 vocab_size, u32 top_k, f32 temperature, f32 top_p, f32 min_p, u32 seed_lo, u32 seed_hi
    batch
        .record_shader(
            crate::shaders::TOPK_SAMPLE,
            &[logits_buf.handle(), out_token_buf.handle()],
            &[
                out_dim as u32,
                top_k,
                temperature.to_bits(),
                top_p.to_bits(),
                min_p.to_bits(),
                seed_lo,
                seed_hi,
            ],
            crate::Workgroups::OneD(1),
        )
        .context("linear_decode_sample: record topk_sample dispatch")?;
    batch
        .record_copy_buffer(&out_token_buf, &out_staging, 4)
        .context("linear_decode_sample: record token readback copy")?;
    batch
        .submit_and_wait("linear_decode_sample")
        .context("linear_decode_sample: submit CommandBatch")?;

    // ---- Step 4: map the 4-byte token copied by the command batch ----
    let out_data = out_staging
        .read_mapped(4)
        .context("failed to read mapped linear_decode_sample token")?;
    let tokens: &[u32] = bytemuck::cast_slice(&out_data);
    tokens
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("linear_decode_sample readback was empty"))
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_linear_decode_sample_batch_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    packed_bf16_weights: bool,
    batch: usize,
    hidden: usize,
    out_dim: usize,
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
) -> Result<Vec<u32>> {
    let device = vk_device.device();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(batch > 0, "linear_decode_sample_batch: batch must be nonzero");
    anyhow::ensure!(hidden > 0, "linear_decode_sample_batch: hidden must be nonzero");
    anyhow::ensure!(out_dim > 0, "linear_decode_sample_batch: out_dim must be nonzero");
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "linear_decode_sample_batch: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    anyhow::ensure!(
        history_rows.len() == history_indices.len() && history_indices.len() == history_counts.len(),
        "linear_decode_sample_batch: history row/index/count length mismatch"
    );
    anyhow::ensure!(
        history_rows.iter().all(|&row| (row as usize) < batch),
        "linear_decode_sample_batch: history row out of range"
    );
    anyhow::ensure!(
        repetition_penalties.len() == batch
            && presence_penalties.len() == batch
            && frequency_penalties.len() == batch
            && temperatures.len() == batch
            && top_k.len() == batch
            && top_p.len() == batch
            && min_p.len() == batch
            && seeds.len() == batch,
        "linear_decode_sample_batch: per-row parameter length mismatch"
    );
    for row in 0..batch {
        let temp = temperatures[row];
        let k = top_k[row];
        let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
        anyhow::ensure!(
            greedy || (temp.is_finite() && temp > 0.0),
            "linear_decode_sample_batch: row {row} has invalid temperature {temp}"
        );
        anyhow::ensure!(
            greedy || (k > 0 && k <= TOPK_SAMPLE_KERNEL_K_MAX),
            "linear_decode_sample_batch: row {row} top_k {k} out of range (1..={})",
            TOPK_SAMPLE_KERNEL_K_MAX
        );
    }

    let seed_lo: Vec<u32> = seeds.iter().map(|seed| (*seed & 0xFFFF_FFFF) as u32).collect();
    let seed_hi: Vec<u32> = seeds.iter().map(|seed| (*seed >> 32) as u32).collect();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_sample_batch x buffer")?;
    let logits_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * out_dim * 4) as u64)
            .context("failed to create linear_decode_sample_batch logits buffer")?;
    let top_k_buf = VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
        .context("failed to create linear_decode_sample_batch top_k buffer")?;
    let temperature_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
            .context("failed to create linear_decode_sample_batch temperature buffer")?;
    let top_p_buf = VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
        .context("failed to create linear_decode_sample_batch top_p buffer")?;
    let min_p_buf = VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
        .context("failed to create linear_decode_sample_batch min_p buffer")?;
    let seed_lo_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
            .context("failed to create linear_decode_sample_batch seed_lo buffer")?;
    let seed_hi_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
            .context("failed to create linear_decode_sample_batch seed_hi buffer")?;

    let penalties_active = !history_indices.is_empty();
    let history_row_buf = if penalties_active {
        Some(
            VulkanBuffer::create_device_local(device, device_local_mt, (history_rows.len() * 4) as u64)
                .context("failed to create batched penalty row buffer")?,
        )
    } else {
        None
    };
    let history_idx_buf = if penalties_active {
        Some(
            VulkanBuffer::create_device_local(device, device_local_mt, (history_indices.len() * 4) as u64)
                .context("failed to create batched penalty index buffer")?,
        )
    } else {
        None
    };
    let history_cnt_buf = if penalties_active {
        Some(
            VulkanBuffer::create_device_local(device, device_local_mt, (history_counts.len() * 4) as u64)
                .context("failed to create batched penalty count buffer")?,
        )
    } else {
        None
    };
    let repetition_buf = if penalties_active {
        Some(
            VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
                .context("failed to create batched repetition buffer")?,
        )
    } else {
        None
    };
    let presence_buf = if penalties_active {
        Some(
            VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
                .context("failed to create batched presence buffer")?,
        )
    } else {
        None
    };
    let frequency_buf = if penalties_active {
        Some(
            VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
                .context("failed to create batched frequency buffer")?,
        )
    } else {
        None
    };

    let mut upload_specs: Vec<(&VulkanBuffer, &[u8])> = Vec::with_capacity(16);
    upload_specs.push((&x_buf, x_data));
    upload_specs.push((&top_k_buf, bytemuck::cast_slice(top_k)));
    upload_specs.push((&temperature_buf, bytemuck::cast_slice(temperatures)));
    upload_specs.push((&top_p_buf, bytemuck::cast_slice(top_p)));
    upload_specs.push((&min_p_buf, bytemuck::cast_slice(min_p)));
    upload_specs.push((&seed_lo_buf, bytemuck::cast_slice(&seed_lo)));
    upload_specs.push((&seed_hi_buf, bytemuck::cast_slice(&seed_hi)));
    if penalties_active {
        upload_specs.push((
            history_idx_buf.as_ref().expect("history index buffer"),
            bytemuck::cast_slice(history_indices),
        ));
        upload_specs.push((
            history_cnt_buf.as_ref().expect("history count buffer"),
            bytemuck::cast_slice(history_counts),
        ));
        upload_specs.push((
            history_row_buf.as_ref().expect("history row buffer"),
            bytemuck::cast_slice(history_rows),
        ));
        upload_specs.push((
            repetition_buf.as_ref().expect("repetition buffer"),
            bytemuck::cast_slice(repetition_penalties),
        ));
        upload_specs.push((
            presence_buf.as_ref().expect("presence buffer"),
            bytemuck::cast_slice(presence_penalties),
        ));
        upload_specs.push((
            frequency_buf.as_ref().expect("frequency buffer"),
            bytemuck::cast_slice(frequency_penalties),
        ));
    }
    let upload_segments: Vec<&[u8]> = upload_specs.iter().map(|(_, bytes)| *bytes).collect();
    let (upload_stage, upload_offsets) =
        VulkanBuffer::create_host_visible_with_segments(device, host_visible_mt, &upload_segments)
            .context("failed to stage linear_decode_sample_batch uploads")?;

    let out_token_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
            .context("failed to create linear_decode_sample_batch out-token buffer")?;
    let out_staging =
        VulkanBuffer::create_host_visible(device, host_visible_mt, (batch * 4) as u64)
            .context("failed to create linear_decode_sample_batch output staging buffer")?;

    let rows8 = packed_bf16_weights
        && batch >= linear_decode_bf16w_rows8_min_batch()
        && linear_decode_bf16w_rows8_enabled();
    let rows4 = packed_bf16_weights
        && !rows8
        && batch >= linear_decode_bf16w_rows4_min_batch()
        && linear_decode_bf16w_rows4_enabled();
    let lm_glsl = if packed_bf16_weights {
        if rows8 {
            crate::shaders::LINEAR_DECODE_BATCHED_ROWS8_BF16W
        } else if rows4 {
            crate::shaders::LINEAR_DECODE_BATCHED_ROWS4_BF16W
        } else {
            crate::shaders::LINEAR_DECODE_BATCHED_BF16W
        }
    } else {
        crate::shaders::LINEAR_DECODE_BATCHED
    };
    let row_groups = if rows8 {
        batch.div_ceil(8)
    } else if rows4 {
        batch.div_ceil(4)
    } else {
        batch
    };

    let mut batch_rec = crate::CommandBatch::new(vk_device)
        .context("linear_decode_sample_batch: create CommandBatch")?;
    let upload_copies: Vec<(&VulkanBuffer, &VulkanBuffer, u64, u64, u64)> = upload_specs
        .iter()
        .enumerate()
        .map(|(idx, (dst, bytes))| {
            (&upload_stage, *dst, upload_offsets[idx], 0, bytes.len() as u64)
        })
        .collect();
    batch_rec
        .record_upload_buffer_regions(&upload_copies)
        .context("linear_decode_sample_batch: record uploads")?;
    batch_rec
        .record_shader(
            lm_glsl,
            &[x_buf.handle(), weight_t.handle(), logits_buf.handle()],
            &[hidden as u32, out_dim as u32, batch as u32],
            crate::Workgroups::OneD((row_groups * out_dim.div_ceil(32)) as u32),
        )
        .context("linear_decode_sample_batch: record lm_head dispatch")?;

    if penalties_active {
        batch_rec
            .record_shader(
                crate::shaders::APPLY_TOKEN_PENALTIES_BATCHED,
                &[
                    logits_buf.handle(),
                    history_idx_buf.as_ref().expect("history index buffer").handle(),
                    history_cnt_buf.as_ref().expect("history count buffer").handle(),
                    history_row_buf.as_ref().expect("history row buffer").handle(),
                    repetition_buf.as_ref().expect("repetition buffer").handle(),
                    presence_buf.as_ref().expect("presence buffer").handle(),
                    frequency_buf.as_ref().expect("frequency buffer").handle(),
                ],
                &[history_indices.len() as u32, out_dim as u32, batch as u32],
                crate::Workgroups::OneD((history_indices.len() as u32).div_ceil(64)),
            )
            .context("linear_decode_sample_batch: record batched penalty dispatch")?;
    }

    batch_rec
        .record_shader(
            crate::shaders::TOPK_SAMPLE_BATCHED,
            &[
                logits_buf.handle(),
                out_token_buf.handle(),
                top_k_buf.handle(),
                temperature_buf.handle(),
                top_p_buf.handle(),
                min_p_buf.handle(),
                seed_lo_buf.handle(),
                seed_hi_buf.handle(),
            ],
            &[out_dim as u32, batch as u32],
            crate::Workgroups::ThreeD(1, batch as u32, 1),
        )
        .context("linear_decode_sample_batch: record batched sample dispatch")?;
    batch_rec
        .record_copy_buffer(&out_token_buf, &out_staging, (batch * 4) as u64)
        .context("linear_decode_sample_batch: record token readback copy")?;
    batch_rec
        .submit_and_wait("linear_decode_sample_batch")
        .context("linear_decode_sample_batch: submit CommandBatch")?;

    let out_data = out_staging
        .read_mapped(batch * 4)
        .context("failed to read mapped linear_decode_sample_batch tokens")?;
    let tokens: &[u32] = bytemuck::cast_slice(&out_data);
    anyhow::ensure!(
        tokens.len() >= batch,
        "linear_decode_sample_batch readback returned {} tokens, expected {batch}",
        tokens.len()
    );
    Ok(tokens[..batch].to_vec())
}

/// Dispatch a batched single-token transposed linear projection and return one
/// argmax token per batch row.
///
/// This is intended for greedy batched LM-head decode. It keeps the vocab
/// logits on the Vulkan device and reads back only `[batch]` token ids.
pub fn dispatch_linear_decode_argmax_batched_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Vec<u32>> {
    dispatch_linear_decode_argmax_batched_cached_impl_bytes(
        vk_device, x_data, weight_t, batch, hidden, out_dim, false,
    )
}

pub fn dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Vec<u32>> {
    dispatch_linear_decode_argmax_batched_cached_impl_bytes(
        vk_device, x_data, weight_t, batch, hidden, out_dim, true,
    )
}

fn dispatch_linear_decode_argmax_batched_cached_impl_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<Vec<u32>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(batch > 0, "batched linear argmax: batch must be nonzero");
    anyhow::ensure!(
        out_dim > 0,
        "batched linear argmax: out_dim must be nonzero"
    );
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "batched linear argmax: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create batched linear argmax x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create batched linear argmax x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, &x_data)?;

    let block_count = out_dim.div_ceil(64);
    let total_blocks = batch * block_count;
    let block_score_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (total_blocks * 4) as u64)
            .context("failed to create batched linear argmax block score buffer")?;
    let block_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (total_blocks * 4) as u64)
            .context("failed to create batched linear argmax block index buffer")?;
    let out_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
            .context("failed to create batched linear argmax output index buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, (batch * 4) as u64)
        .context("failed to create batched linear argmax output staging buffer")?;

    let rows8 = packed_bf16_weights
        && batch >= linear_decode_bf16w_rows8_min_batch()
        && linear_decode_bf16w_rows8_enabled();
    let rows4 = packed_bf16_weights
        && !rows8
        && batch >= linear_decode_bf16w_rows4_min_batch()
        && linear_decode_bf16w_rows4_enabled();
    let blocks_glsl = if rows8 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_batched_blocks_rows8_bf16w.comp"
        )
    } else if rows4 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_batched_blocks_rows4_bf16w.comp"
        )
    } else if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_batched_blocks_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_batched_blocks.comp"
        )
    };
    let blocks_spirv = crate::pipeline::ShaderPipeline::compile_shader(blocks_glsl)?;
    let block_push: Vec<u32> = if rows8 || rows4 {
        vec![
            hidden as u32,
            out_dim as u32,
            block_count as u32,
            batch as u32,
        ]
    } else {
        vec![hidden as u32, out_dim as u32, block_count as u32]
    };
    let block_workgroups = if rows8 {
        batch.div_ceil(8) * block_count
    } else if rows4 {
        batch.div_ceil(4) * block_count
    } else {
        total_blocks
    };
    let block_handles = vec![
        x_buf.handle(),
        weight_t.handle(),
        block_score_buf.handle(),
        block_index_buf.handle(),
    ];
    let (block_set_layout, block_layout, block_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &blocks_spirv,
            block_handles.len(),
            (block_push.len() * 4) as u32,
        )?;

    let reduce_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_argmax_batched_reduce.comp"
    );
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl)?;
    let reduce_push: [u32; 1] = [block_count as u32];
    let reduce_handles = vec![
        block_score_buf.handle(),
        block_index_buf.handle(),
        out_index_buf.handle(),
    ];
    let (reduce_set_layout, reduce_layout, reduce_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &reduce_spirv,
            reduce_handles.len(),
            (reduce_push.len() * 4) as u32,
        )?;

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let block_set_layouts = vec![block_set_layout];
    let block_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&block_set_layouts)
                    ,
            )
            .context("failed to allocate batched linear argmax block descriptor set")?[0]
    };
    let block_buf_infos: Vec<vk::DescriptorBufferInfo> = block_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let block_descriptor_writes: Vec<vk::WriteDescriptorSet> = block_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(block_descriptor_set, i as u32, info))
        .collect();

    let reduce_set_layouts = vec![reduce_set_layout];
    let reduce_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&reduce_set_layouts)
                    ,
            )
            .context("failed to allocate batched linear argmax reduce descriptor set")?[0]
    };
    let reduce_buf_infos: Vec<vk::DescriptorBufferInfo> = reduce_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let reduce_descriptor_writes: Vec<vk::WriteDescriptorSet> = reduce_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(reduce_descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&block_descriptor_writes, &[]);
        device.update_descriptor_sets(&reduce_descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::default().size(x_data.len() as u64)],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, block_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            block_layout,
            0,
            &[block_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            block_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&block_push),
        );
        device.cmd_dispatch(cmd, block_workgroups as u32, 1, 1);

        let block_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[block_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, reduce_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            reduce_layout,
            0,
            &[reduce_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            reduce_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&reduce_push),
        );
        device.cmd_dispatch(cmd, batch as u32, 1, 1);

        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_index_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size((batch * 4) as u64)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batched linear argmax dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batched linear argmax dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let indices: &[u32] = bytemuck::cast_slice(&out_data);
    anyhow::ensure!(
        indices.len() >= batch,
        "batched linear argmax readback returned {} indices, expected {batch}",
        indices.len()
    );
    Ok(indices[..batch].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_full_attn_qkv_decode_cached_impl(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    bf16_weights: bool,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        x_data.len() == hidden * 4,
        "full_attn_qkv_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        hidden * 4
    );

    let total_out = q_dim + k_dim + v_dim;
    let glsl_path = if bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        hidden as u32,
        q_dim as u32,
        k_dim as u32,
        v_dim as u32,
        total_out as u32,
    ];
    if full_attn_qkv_single_submit_enabled() {
        return dispatch_full_attn_qkv_decode_cached_single_submit(
            vk_device,
            q_weight_t,
            k_weight_t,
            v_weight_t,
            q_dim,
            k_dim,
            v_dim,
            total_out,
            &spirv,
            &push_constants,
            x_data,
        );
    }

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            x_data,
        )
        .context("failed to upload full_attn_qkv_decode x buffer")?;
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (total_out * 4) as u64)
            .context("failed to create full_attn_qkv_decode output buffer")?;
    let all_handles = vec![
        x_buf.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        out_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        total_out.div_ceil(16) as u32,
    )
    .context("full_attn_qkv_decode kernel failed")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back full_attn_qkv_decode output")?
    };

    Ok(out_data)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_full_attn_qkv_decode_cached_single_submit(
    vk_device: &VulkanDevice,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    total_out: usize,
    spirv: &[u8],
    push_constants: &[u32],
    x_data: &[u8],
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;

    let out_size = (total_out * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create full_attn_qkv_decode output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create full_attn_qkv_decode output staging buffer")?;

    let all_handles = vec![
        x_buf.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        out_buf.handle(),
    ];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate full_attn_qkv_decode descriptor set")?[0]
    };
    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::default().size(x_data.len() as u64)],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, total_out.div_ceil(16) as u32, 1, 1);
        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit full_attn_qkv_decode single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for full_attn_qkv_decode single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let _ = (q_dim, k_dim, v_dim); // shape metadata consumed at the `_bytes` shim boundary
    Ok(out_data)
}

/// Split the contiguous full-attention QKV decode output
/// `[1, 1, q_dim + k_dim + v_dim]` f32 bytes into three per-dim
/// `Vec<u8>` slices for the `_bytes` shim callers. Replaces the older
/// candle-Tensor split helper as part of the kernels.rs candle-free
/// migration. (#1082)
fn split_full_attn_qkv_bytes(
    out_data: &[u8],
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let total_bytes = (q_dim + k_dim + v_dim) * 4;
    anyhow::ensure!(
        out_data.len() >= total_bytes,
        "full_attn_qkv_decode output slice exceeds readback buffer"
    );
    let q_end = q_dim * 4;
    let k_end = q_end + k_dim * 4;
    let v_end = k_end + v_dim * 4;
    Ok((
        out_data[..q_end].to_vec(),
        out_data[q_end..k_end].to_vec(),
        out_data[k_end..v_end].to_vec(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let out_data = dispatch_full_attn_qkv_decode_cached_impl(
        vk_device, x_data, q_weight_t, k_weight_t, v_weight_t, hidden, q_dim, k_dim, v_dim, false,
    )?;
    split_full_attn_qkv_bytes(&out_data, q_dim, k_dim, v_dim)
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_bf16_weights_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let out_data = dispatch_full_attn_qkv_decode_cached_impl(
        vk_device, x_data, q_weight_t, k_weight_t, v_weight_t, hidden, q_dim, k_dim, v_dim, true,
    )?;
    split_full_attn_qkv_bytes(&out_data, q_dim, k_dim, v_dim)
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_batched_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let out_data = dispatch_full_attn_qkv_decode_cached_batched_impl(
        vk_device, x_data, q_weight_t, k_weight_t, v_weight_t, batch, hidden, q_dim, k_dim, v_dim, false,
    )?;
    split_batched_qkv_output(&out_data, batch, q_dim, k_dim, v_dim)
}

pub fn dispatch_full_attn_qkv_decode_cached_batched_bf16_weights_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let out_data = dispatch_full_attn_qkv_decode_cached_batched_impl(
        vk_device,
        x_data,
        q_weight_t,
        k_weight_t,
        v_weight_t,
        batch,
        hidden,
        q_dim,
        k_dim,
        v_dim,
        true,
    )?;
    split_batched_qkv_output(&out_data, batch, q_dim, k_dim, v_dim)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_full_attn_qkv_decode_cached_batched_impl(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    bf16_weights: bool,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(batch > 0, "full_attn_qkv_decode_batched: batch must be > 0");

    let expected_x_bytes = batch
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(4))
        .context("full_attn_qkv_decode_batched: x byte count overflow")?;
    anyhow::ensure!(
        x_data.len() == expected_x_bytes,
        "full_attn_qkv_decode_batched: x buffer has {} bytes, expected {} (batch={}, hidden={})",
        x_data.len(),
        expected_x_bytes,
        batch,
        hidden
    );

    let total_out = q_dim
        .checked_add(k_dim)
        .and_then(|n| n.checked_add(v_dim))
        .context("full_attn_qkv_decode_batched: total_out overflow")?;
    anyhow::ensure!(total_out > 0, "full_attn_qkv_decode_batched: total_out is zero");
    let full_attn_qkv_rows4 = bf16_weights
        && batch >= full_attn_qkv_bf16w_rows4_min_batch()
        && full_attn_qkv_bf16w_rows4_enabled();
    let glsl_path = if bf16_weights {
        if full_attn_qkv_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/full_attn_qkv_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/full_attn_qkv_decode_batched_bf16w.comp"
            )
        }
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode_batched.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 6] = [
        hidden as u32,
        q_dim as u32,
        k_dim as u32,
        v_dim as u32,
        total_out as u32,
        batch as u32,
    ];

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode_batched x buffer")?;

    let out_bytes = batch
        .checked_mul(total_out)
        .and_then(|n| n.checked_mul(4))
        .context("full_attn_qkv_decode_batched: output byte count overflow")?;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_bytes as u64)
        .context("failed to create full_attn_qkv_decode_batched output buffer")?;

    let all_handles = vec![
        x_buf.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        out_buf.handle(),
    ];
    let col_groups = total_out.div_ceil(16);
    let row_groups = if full_attn_qkv_rows4 {
        batch.div_ceil(4)
    } else {
        batch
    };
    let total_groups = row_groups
        .checked_mul(col_groups)
        .context("full_attn_qkv_decode_batched: workgroup count overflow")?;
    let single_submit = full_attn_qkv_single_submit_enabled();
    let out_data = if single_submit {
        run_compute_pipeline_with_transfer_readback(
            vk_device,
            &x_buf,
            x_data,
            &out_buf,
            out_bytes as u64,
            &spirv,
            &all_handles,
            &push_constants,
            total_groups as u32,
        )
        .context("full_attn_qkv_decode_batched single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                x_data,
            )
            .context("failed to upload full_attn_qkv_decode_batched x buffer")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            total_groups as u32,
        )
        .context("full_attn_qkv_decode_batched kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back full_attn_qkv_decode_batched output")?
    };

    let _ = (q_dim, k_dim, v_dim); // shape metadata consumed at the `_bytes` shim boundary
    Ok(out_data)
}

/// Split the contiguous batched `[batch, total_out]` readback buffer into
/// three per-dim `Vec<u8>` arrays of shape `[batch, *_dim]` f32 bytes.
/// The shader writes rows in `(q | k | v)` order per batch element, so
/// we copy row-by-row into three per-dim accumulators.
///
/// Candle-free counterpart of the previous tensor-typed split — the
/// `_bytes` shims consume the three `Vec<u8>` slices directly. (#1082)
fn split_batched_qkv_output(
    out_data: &[u8],
    batch: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let total_out = q_dim + k_dim + v_dim;
    let expected_bytes = batch
        .checked_mul(total_out)
        .and_then(|n| n.checked_mul(4))
        .context("split_batched_qkv_output: byte count overflow")?;
    anyhow::ensure!(
        out_data.len() >= expected_bytes,
        "split_batched_qkv_output: readback has {} bytes, expected {}",
        out_data.len(),
        expected_bytes
    );
    let out_f32: &[f32] = bytemuck::cast_slice(&out_data[..expected_bytes]);

    let mut q_buf: Vec<f32> = Vec::with_capacity(batch * q_dim);
    let mut k_buf: Vec<f32> = Vec::with_capacity(batch * k_dim);
    let mut v_buf: Vec<f32> = Vec::with_capacity(batch * v_dim);
    for row in 0..batch {
        let base = row * total_out;
        q_buf.extend_from_slice(&out_f32[base..base + q_dim]);
        k_buf.extend_from_slice(&out_f32[base + q_dim..base + q_dim + k_dim]);
        v_buf.extend_from_slice(&out_f32[base + q_dim + k_dim..base + total_out]);
    }

    Ok((
        bytemuck::cast_slice(&q_buf).to_vec(),
        bytemuck::cast_slice(&k_buf).to_vec(),
        bytemuck::cast_slice(&v_buf).to_vec(),
    ))
}

/// Dispatch batched paged decode attention over compacted K/V windows.
///
/// `q` is `[batch, 1, num_heads, head_dim]`, `k` and `v` are compact
/// `[batch, max_seqlen, num_kv_heads, head_dim]`, and `seq_lens` gives the
/// active prefix length for each row. Output is `[batch, 1, num_heads,
/// head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attn_decode_batch_f32_bytes(
    vk_device: &VulkanDevice,
    q_data_in: &[u8],
    k_data_in: &[u8],
    v_data_in: &[u8],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    max_seqlen: usize,
    num_kv_heads: usize,
    seq_lens: &[u32],
    softmax_scale: f32,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        q_data_in.len() == batch * 1 * num_heads * head_dim * 4,
        "paged_attn_decode_batch: q bytes {} mismatch expected {}",
        q_data_in.len(),
        batch * num_heads * head_dim * 4,
    );
    anyhow::ensure!(
        k_data_in.len() == batch * max_seqlen * num_kv_heads * head_dim * 4,
        "paged_attn_decode_batch: k bytes {} mismatch expected {}",
        k_data_in.len(),
        batch * max_seqlen * num_kv_heads * head_dim * 4,
    );
    anyhow::ensure!(
        v_data_in.len() == batch * max_seqlen * num_kv_heads * head_dim * 4,
        "paged_attn_decode_batch: v bytes {} mismatch expected {}",
        v_data_in.len(),
        batch * max_seqlen * num_kv_heads * head_dim * 4,
    );
    anyhow::ensure!(
        head_dim <= 256,
        "paged_attn_decode_batch supports head_dim <= 256"
    );
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "paged_attn_decode_batch requires integer GQA ratio"
    );
    anyhow::ensure!(
        seq_lens.len() == batch,
        "paged_attn_decode_batch seq_lens length {} != batch {batch}",
        seq_lens.len()
    );
    for &len in seq_lens {
        anyhow::ensure!(
            len > 0 && len as usize <= max_seqlen,
            "paged_attn_decode_batch invalid row seq_len {len} for max_seqlen {max_seqlen}"
        );
    }

    let q_data: Vec<u8> = q_data_in.to_vec();
    let k_data: Vec<u8> = k_data_in.to_vec();
    let v_data: Vec<u8> = v_data_in.to_vec();
    let seq_data = bytemuck::cast_slice(seq_lens).to_vec();

    let make_input = |data: &[u8], label: &str| -> Result<VulkanBuffer> {
        let buf = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .with_context(|| format!("failed to create paged_attn_decode_batch {label} buffer"))?;
        Ok(buf)
    };
    let q_buf = make_input(&q_data, "q")?;
    let k_buf = make_input(&k_data, "k")?;
    let v_buf = make_input(&v_data, "v")?;
    let seq_buf = make_input(&seq_data, "seq_lens")?;

    let out_size = (batch * num_heads * head_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create paged_attn_decode_batch output buffer")?;
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants = [
        max_seqlen as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
    ];
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        seq_buf.handle(),
        out_buf.handle(),
    ];
    let out_data = if paged_attn_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[
                (&q_buf, &q_data),
                (&k_buf, &k_data),
                (&v_buf, &v_data),
                (&seq_buf, &seq_data),
            ],
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            if paged_attn_batched_uploads_enabled() {
                upload_buffers_with_command_pool(
                    device,
                    host_visible_mt,
                    queue,
                    *command_pool,
                    &[
                        (&q_buf, &q_data),
                        (&k_buf, &k_data),
                        (&v_buf, &v_data),
                        (&seq_buf, &seq_data),
                    ],
                )
                .context("failed to upload paged_attn_decode_batch inputs")?;
            } else {
                for (buf, data, label) in [
                    (&q_buf, &q_data, "q"),
                    (&k_buf, &k_data, "k"),
                    (&v_buf, &v_data, "v"),
                    (&seq_buf, &seq_data, "seq_lens"),
                ] {
                    VulkanBuffer::upload_data_with_command_pool(
                        device,
                        host_visible_mt,
                        queue,
                        *command_pool,
                        buf,
                        data,
                    )
                    .with_context(|| {
                        format!("failed to upload paged_attn_decode_batch {label} buffer")
                    })?;
                }
            }
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back paged_attn_decode_batch output")?
    };
    let _ = (batch, num_heads, head_dim);
    Ok(out_data)
}

/// Paged-pool variant of [`dispatch_paged_attn_decode_batch_f32`].
///
/// Skips the host-side block_table → compacted K/V gather and uploads the
/// raw K/V pool plus the block table; the shader walks the per-row block
/// indices inline. Use when the compacted view would dwarf the pool itself
/// (i.e., when `batch * max_seqlen > total_slots`), which is the typical
/// shape for multi-batch decode at non-trivial context lengths.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attn_decode_batch_paged_f32_bytes(
    vk_device: &VulkanDevice,
    q_data_in: &[u8],
    k_pool_data_in: &[u8],
    v_pool_data_in: &[u8],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    total_slots: usize,
    num_kv_heads: usize,
    block_table_u32: &[u32],
    seq_lens: &[u32],
    max_blocks_per_seq: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        batch > 0 && max_blocks_per_seq > 0 && page_block_size > 0,
        "paged_attn_decode_batch_paged: batch/max_blocks_per_seq/page_block_size must be > 0"
    );
    anyhow::ensure!(
        seq_lens.len() == batch,
        "paged_attn_decode_batch_paged: seq_lens length {} != batch {batch}",
        seq_lens.len()
    );
    anyhow::ensure!(
        block_table_u32.len() == batch * max_blocks_per_seq,
        "paged_attn_decode_batch_paged: block_table length {} != batch*max_blocks_per_seq {}",
        block_table_u32.len(),
        batch * max_blocks_per_seq
    );
    anyhow::ensure!(
        q_data_in.len() == batch * 1 * num_heads * head_dim * 4,
        "paged_attn_decode_batch_paged: q bytes {} mismatch expected {}",
        q_data_in.len(),
        batch * num_heads * head_dim * 4,
    );
    anyhow::ensure!(
        k_pool_data_in.len() == total_slots * num_kv_heads * head_dim * 4,
        "paged_attn_decode_batch_paged: k_pool bytes {} mismatch expected {}",
        k_pool_data_in.len(),
        total_slots * num_kv_heads * head_dim * 4,
    );
    anyhow::ensure!(
        v_pool_data_in.len() == total_slots * num_kv_heads * head_dim * 4,
        "paged_attn_decode_batch_paged: v_pool bytes {} mismatch expected {}",
        v_pool_data_in.len(),
        total_slots * num_kv_heads * head_dim * 4,
    );
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "paged_attn_decode_batch_paged: requires integer GQA ratio"
    );
    for &len in seq_lens {
        anyhow::ensure!(
            len > 0,
            "paged_attn_decode_batch_paged: zero-length seq_len not supported"
        );
    }

    let q_data: Vec<u8> = q_data_in.to_vec();
    let k_data: Vec<u8> = k_pool_data_in.to_vec();
    let v_data: Vec<u8> = v_pool_data_in.to_vec();
    let bt_bytes: Vec<u8> = bytemuck::cast_slice(block_table_u32).to_vec();
    let seq_bytes: Vec<u8> = bytemuck::cast_slice(seq_lens).to_vec();

    let make_input = |data: &[u8], label: &str| -> Result<VulkanBuffer> {
        VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .with_context(|| format!("failed to create paged_attn_decode_batch_paged {label} buffer"))
    };
    let q_buf = make_input(&q_data, "q")?;
    let k_buf = make_input(&k_data, "k_pool")?;
    let v_buf = make_input(&v_data, "v_pool")?;
    let bt_buf = make_input(&bt_bytes, "block_table")?;
    let seq_buf = make_input(&seq_bytes, "seq_lens")?;

    let out_size = (batch * num_heads * head_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create paged_attn_decode_batch_paged output buffer")?;
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants = [
        max_blocks_per_seq as u32,
        page_block_size as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
    ];
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        bt_buf.handle(),
        seq_buf.handle(),
        out_buf.handle(),
    ];
    let out_data = if paged_attn_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[
                (&q_buf, &q_data),
                (&k_buf, &k_data),
                (&v_buf, &v_data),
                (&bt_buf, &bt_bytes),
                (&seq_buf, &seq_bytes),
            ],
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch_paged single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            if paged_attn_batched_uploads_enabled() {
                upload_buffers_with_command_pool(
                    device,
                    host_visible_mt,
                    queue,
                    *command_pool,
                    &[
                        (&q_buf, &q_data),
                        (&k_buf, &k_data),
                        (&v_buf, &v_data),
                        (&bt_buf, &bt_bytes),
                        (&seq_buf, &seq_bytes),
                    ],
                )
                .context("failed to upload paged_attn_decode_batch_paged inputs")?;
            } else {
                for (buf, data, label) in [
                    (&q_buf, &q_data, "q"),
                    (&k_buf, &k_data, "k_pool"),
                    (&v_buf, &v_data, "v_pool"),
                    (&bt_buf, &bt_bytes, "block_table"),
                    (&seq_buf, &seq_bytes, "seq_lens"),
                ] {
                    VulkanBuffer::upload_data_with_command_pool(
                        device,
                        host_visible_mt,
                        queue,
                        *command_pool,
                        buf,
                        data,
                    )
                    .with_context(|| {
                        format!("failed to upload paged_attn_decode_batch_paged {label} buffer")
                    })?;
                }
            }
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch_paged kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back paged_attn_decode_batch_paged output")?
    };
    let _ = (batch, num_heads, head_dim);
    Ok(out_data)
}

/// Split-K paged-pool variant of [`dispatch_paged_attn_decode_batch_paged_f32_bytes`].
///
/// Records the chunk scan and reduction in one command submit when the
/// single-submit path is enabled, so generic paged decode can use the same
/// higher-occupancy attention primitive as the resident decode path.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes(
    vk_device: &VulkanDevice,
    q_data_in: &[u8],
    k_pool_data_in: &[u8],
    v_pool_data_in: &[u8],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    total_slots: usize,
    num_kv_heads: usize,
    block_table_u32: &[u32],
    seq_lens: &[u32],
    max_blocks_per_seq: usize,
    page_block_size: usize,
    softmax_scale: f32,
    num_chunks: usize,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        batch > 0 && max_blocks_per_seq > 0 && page_block_size > 0 && num_chunks > 0,
        "paged_attn_decode_batch_paged_splitk: batch/max_blocks_per_seq/page_block_size/num_chunks must be > 0"
    );
    anyhow::ensure!(
        num_chunks <= PAGED_ATTN_SPLITK_MAX_CHUNKS,
        "paged_attn_decode_batch_paged_splitk: num_chunks {num_chunks} exceeds max {PAGED_ATTN_SPLITK_MAX_CHUNKS}"
    );
    let block_table_expected = batch
        .checked_mul(max_blocks_per_seq)
        .context("paged_attn_decode_batch_paged_splitk block table size overflow")?;
    let q_expected = batch
        .checked_mul(num_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(4))
        .context("paged_attn_decode_batch_paged_splitk q byte size overflow")?;
    let kv_expected = total_slots
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(4))
        .context("paged_attn_decode_batch_paged_splitk kv byte size overflow")?;
    anyhow::ensure!(
        seq_lens.len() == batch,
        "paged_attn_decode_batch_paged_splitk: seq_lens length {} != batch {batch}",
        seq_lens.len()
    );
    anyhow::ensure!(
        block_table_u32.len() == block_table_expected,
        "paged_attn_decode_batch_paged_splitk: block_table length {} != batch*max_blocks_per_seq {}",
        block_table_u32.len(),
        block_table_expected
    );
    anyhow::ensure!(
        q_data_in.len() == q_expected,
        "paged_attn_decode_batch_paged_splitk: q bytes {} mismatch expected {}",
        q_data_in.len(),
        q_expected,
    );
    anyhow::ensure!(
        k_pool_data_in.len() == kv_expected,
        "paged_attn_decode_batch_paged_splitk: k_pool bytes {} mismatch expected {}",
        k_pool_data_in.len(),
        kv_expected,
    );
    anyhow::ensure!(
        v_pool_data_in.len() == kv_expected,
        "paged_attn_decode_batch_paged_splitk: v_pool bytes {} mismatch expected {}",
        v_pool_data_in.len(),
        kv_expected,
    );
    anyhow::ensure!(
        num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "paged_attn_decode_batch_paged_splitk: requires integer GQA ratio"
    );
    for &len in seq_lens {
        anyhow::ensure!(
            len > 0,
            "paged_attn_decode_batch_paged_splitk: zero-length seq_len not supported"
        );
    }

    let q_data: Vec<u8> = q_data_in.to_vec();
    let k_data: Vec<u8> = k_pool_data_in.to_vec();
    let v_data: Vec<u8> = v_pool_data_in.to_vec();
    let bt_bytes: Vec<u8> = bytemuck::cast_slice(block_table_u32).to_vec();
    let seq_bytes: Vec<u8> = bytemuck::cast_slice(seq_lens).to_vec();

    let make_input = |data: &[u8], label: &str| -> Result<VulkanBuffer> {
        VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .with_context(|| {
                format!("failed to create paged_attn_decode_batch_paged_splitk {label} buffer")
            })
    };
    let q_buf = make_input(&q_data, "q")?;
    let k_buf = make_input(&k_data, "k_pool")?;
    let v_buf = make_input(&v_data, "v_pool")?;
    let bt_buf = make_input(&bt_bytes, "block_table")?;
    let seq_buf = make_input(&seq_bytes, "seq_lens")?;

    let partials_stride = 2usize
        .checked_add(head_dim)
        .context("paged_attn_decode_batch_paged_splitk partial stride overflow")?;
    let partials_elems = batch
        .checked_mul(num_heads)
        .and_then(|n| n.checked_mul(num_chunks))
        .and_then(|n| n.checked_mul(partials_stride))
        .context("paged_attn_decode_batch_paged_splitk partial size overflow")?;
    let partials_size = u64::try_from(partials_elems)
        .context("paged_attn_decode_batch_paged_splitk partial size exceeds u64")?
        .checked_mul(4)
        .context("paged_attn_decode_batch_paged_splitk partial bytes overflow")?;
    let partials_buf = VulkanBuffer::create_device_local(device, device_local_mt, partials_size)
        .context("failed to create paged_attn_decode_batch_paged_splitk partials buffer")?;

    let out_size = u64::try_from(q_expected)
        .context("paged_attn_decode_batch_paged_splitk output size exceeds u64")?;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create paged_attn_decode_batch_paged_splitk output buffer")?;

    let split_glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged_splitk.comp"
    );
    let reduce_glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged_splitk_reduce.comp"
    );
    let split_spirv = crate::pipeline::ShaderPipeline::compile_shader(split_glsl_path)?;
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl_path)?;
    let split_push_constants = [
        max_blocks_per_seq as u32,
        page_block_size as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
        num_chunks as u32,
    ];
    let reduce_push_constants = [num_heads as u32, head_dim as u32, num_chunks as u32];
    let split_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        bt_buf.handle(),
        seq_buf.handle(),
        partials_buf.handle(),
    ];
    let reduce_handles = vec![partials_buf.handle(), out_buf.handle()];
    let split_workgroups = batch
        .checked_mul(num_heads)
        .and_then(|n| n.checked_mul(num_chunks))
        .context("paged_attn_decode_batch_paged_splitk workgroup count overflow")?;
    let reduce_workgroups = batch
        .checked_mul(num_heads)
        .context("paged_attn_decode_batch_paged_splitk reduce workgroup count overflow")?;
    let split_workgroups = u32::try_from(split_workgroups)
        .context("paged_attn_decode_batch_paged_splitk workgroup count exceeds u32")?;
    let reduce_workgroups = u32::try_from(reduce_workgroups)
        .context("paged_attn_decode_batch_paged_splitk reduce workgroup count exceeds u32")?;
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        split_workgroups <= limit_x && reduce_workgroups <= limit_x,
        "paged_attn_decode_batch_paged_splitk workgroups exceed device x-axis limit {limit_x}"
    );

    let out_data = if paged_attn_single_submit_enabled() {
        let readbacks = run_two_stage_compute_pipeline_with_transfers(
            vk_device,
            &[
                (&q_buf, &q_data),
                (&k_buf, &k_data),
                (&v_buf, &v_data),
                (&bt_buf, &bt_bytes),
                (&seq_buf, &seq_bytes),
            ],
            &[&out_buf],
            &split_spirv,
            &split_handles,
            &split_push_constants,
            split_workgroups,
            &reduce_spirv,
            &reduce_handles,
            &reduce_push_constants,
            reduce_workgroups,
        )
        .context("paged_attn_decode_batch_paged_splitk single-submit failed")?;
        anyhow::ensure!(
            readbacks.len() == 1,
            "paged_attn_decode_batch_paged_splitk expected one readback, got {}",
            readbacks.len()
        );
        readbacks.into_iter().next().unwrap()
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            if paged_attn_batched_uploads_enabled() {
                upload_buffers_with_command_pool(
                    device,
                    host_visible_mt,
                    queue,
                    *command_pool,
                    &[
                        (&q_buf, &q_data),
                        (&k_buf, &k_data),
                        (&v_buf, &v_data),
                        (&bt_buf, &bt_bytes),
                        (&seq_buf, &seq_bytes),
                    ],
                )
                .context("failed to upload paged_attn_decode_batch_paged_splitk inputs")?;
            } else {
                for (buf, data, label) in [
                    (&q_buf, &q_data, "q"),
                    (&k_buf, &k_data, "k_pool"),
                    (&v_buf, &v_data, "v_pool"),
                    (&bt_buf, &bt_bytes, "block_table"),
                    (&seq_buf, &seq_bytes, "seq_lens"),
                ] {
                    VulkanBuffer::upload_data_with_command_pool(
                        device,
                        host_visible_mt,
                        queue,
                        *command_pool,
                        buf,
                        data,
                    )
                    .with_context(|| {
                        format!(
                            "failed to upload paged_attn_decode_batch_paged_splitk {label} buffer"
                        )
                    })?;
                }
            }
        }
        run_two_stage_compute_pipeline(
            vk_device,
            &split_spirv,
            &split_handles,
            &split_push_constants,
            split_workgroups,
            &reduce_spirv,
            &reduce_handles,
            &reduce_push_constants,
            reduce_workgroups,
        )
        .context("paged_attn_decode_batch_paged_splitk kernels failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back paged_attn_decode_batch_paged_splitk output")?
    };

    Ok(out_data)
}

pub fn dispatch_mlp_gate_up_decode_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    hidden: usize,
    intermediate: usize,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
) -> Result<Vec<u8>> {
    dispatch_mlp_gate_up_decode_cached_bytes_core(
        vk_device,
        x_data,
        batch,
        hidden,
        intermediate,
        gate_weight_t,
        up_weight_t,
    )
}

fn dispatch_mlp_gate_up_decode_cached_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    hidden: usize,
    intermediate: usize,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "mlp_gate_up_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create mlp_gate_up_decode x buffer")?;
    let out_size = (batch * intermediate * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create mlp_gate_up_decode output buffer")?;

    let use_rows2 = use_prefill_row_pair_matmul(batch);
    let glsl_path = if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode.comp"
        )
    } else if use_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let mut push_constants = vec![hidden as u32, intermediate as u32];
    if batch > 1 {
        push_constants.push(batch as u32);
    }
    let all_handles = vec![
        x_buf.handle(),
        gate_weight_t.handle(),
        up_weight_t.handle(),
        out_buf.handle(),
    ];
    let workgroups = if batch == 1 {
        intermediate.div_ceil(64) as u32
    } else if use_rows2 {
        (batch.div_ceil(2) * intermediate.div_ceil(64)) as u32
    } else {
        (batch * intermediate.div_ceil(128)) as u32
    };

    if mlp_gate_up_single_submit_enabled() {
        run_compute_pipeline_with_transfer_readback(
            vk_device,
            &x_buf,
            x_data,
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            workgroups,
        )
        .context("mlp_gate_up_decode single-submit kernel failed")
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                x_data,
            )
            .context("failed to upload mlp_gate_up_decode x buffer")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("mlp_gate_up_decode kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back mlp_gate_up_decode output")
    }
}

/// Dispatch single-token SwiGLU MLP with the hidden activation kept on Vulkan.
///
/// This runs two kernels:
/// 1. `hidden = silu(x @ gate_t) * (x @ up_t)`
/// 2. `out = hidden @ down_t`
///
/// Only the final `[batch, 1, out_dim]` tensor is read back to CPU.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Vec<u8>> {
    dispatch_mlp_decode_cached_impl(
        vk_device,
        x_data,
        batch,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        hidden,
        intermediate,
        out_dim,
        false,
        false,
    )
}

pub fn dispatch_mlp_decode_cached_bf16_weights_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Vec<u8>> {
    dispatch_mlp_decode_cached_impl(
        vk_device,
        x_data,
        batch,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        hidden,
        intermediate,
        out_dim,
        true,
        true,
    )
}

pub fn dispatch_mlp_decode_cached_bf16_gate_up_f32_down_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Vec<u8>> {
    dispatch_mlp_decode_cached_impl(
        vk_device,
        x_data,
        batch,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        hidden,
        intermediate,
        out_dim,
        true,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mlp_decode_cached_impl(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    batch: usize,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    gate_up_bf16_weights: bool,
    down_bf16_weights: bool,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    let profile_stages = profile_vulkan_mlp_kernel_stages_enabled();
    let gate_up_rows2 = !gate_up_bf16_weights && use_prefill_row_pair_matmul(batch);
    // For the all-bf16 MLP, rows4 / rows8 amortization only wins once there
    // are enough rows to keep the SMs full. The default rows8 crossover is
    // runtime-tunable because standalone component timings and full mixed
    // paged token timings can disagree across devices. The Strix Halo default
    // keeps batch 64 on rows4 unless an override proves faster.
    let rows8_path = gate_up_bf16_weights
        && down_bf16_weights
        && batch >= mlp_bf16_rows8_min_batch()
        && mlp_bf16_rows8_enabled();
    let down_bf16_rows4 = down_bf16_weights
        && gate_up_bf16_weights
        && batch >= mlp_bf16_down_rows4_min_batch()
        && !rows8_path
        && mlp_bf16_down_rows4_enabled();
    // gate_up rows4 reuses weights across 4 rows. The intermediate dim is
    // large (9216 for Qwen3.5-4B) so the 64-cols-per-workgroup tiling
    // still leaves plenty of workgroups even at the rows4 4× reduction in
    // row count — making gate_up_rows4 a near-unconditional win for any
    // bf16-weight MLP at batch ≥ 8, independent of which linear-down
    // path takes over. Decouple it from `down_bf16_rows4`.
    let gate_up_rows4 = gate_up_bf16_weights
        && batch >= mlp_bf16_gate_up_rows4_min_batch()
        && !rows8_path
        && mlp_bf16_gate_up_rows4_enabled();
    let down_rows4 =
        gate_up_bf16_weights
            && !down_bf16_weights
            && batch >= mlp_f32_down_rows4_min_batch()
            && mlp_f32_down_rows4_enabled();
    let down_rows2 = !down_bf16_weights && !down_rows4 && use_prefill_row_pair_matmul(batch);
    let chained_dispatch = mlp_chained_dispatch_enabled();
    let chained_transfer_submit = chained_dispatch && mlp_chained_transfer_submit_enabled();
    let total_start = profile_stages.then(Instant::now);
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "mlp_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    let stage_start = profile_stages.then(Instant::now);
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create mlp_decode x buffer")?;
    finish_vulkan_mlp_kernel_stage_profile(
        "create_x_buffer",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    if !chained_transfer_submit {
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload mlp_decode x buffer")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "upload_x",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
    }

    let stage_start = profile_stages.then(Instant::now);
    let hidden_buf = VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        (batch * intermediate * 4) as u64,
    )
    .context("failed to create mlp_decode hidden buffer")?;
    let out_size = (batch * out_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create mlp_decode output buffer")?;
    finish_vulkan_mlp_kernel_stage_profile(
        "create_work_buffers",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );

    let gate_up_glsl = if gate_up_bf16_weights {
        if batch == 1 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_bf16w.comp"
            )
        } else if rows8_path {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_rows8_bf16w.comp"
            )
        } else if gate_up_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_bf16w.comp"
            )
        }
    } else if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode.comp"
        )
    } else if gate_up_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched.comp"
        )
    };
    let stage_start = profile_stages.then(Instant::now);
    let gate_up_spirv = crate::pipeline::ShaderPipeline::compile_shader(gate_up_glsl)?;
    finish_vulkan_mlp_kernel_stage_profile(
        "gate_up_shader",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    let mut gate_up_push = vec![hidden as u32, intermediate as u32];
    if batch > 1 {
        gate_up_push.push(batch as u32);
    }
    let gate_up_handles = vec![
        x_buf.handle(),
        gate_weight_t.handle(),
        up_weight_t.handle(),
        hidden_buf.handle(),
    ];
    let gate_up_workgroups = if batch == 1 {
        intermediate.div_ceil(64) as u32
    } else if rows8_path {
        (batch.div_ceil(8) * intermediate.div_ceil(64)) as u32
    } else if gate_up_rows4 {
        (batch.div_ceil(4) * intermediate.div_ceil(64)) as u32
    } else if gate_up_rows2 {
        (batch.div_ceil(2) * intermediate.div_ceil(64)) as u32
    } else {
        (batch * intermediate.div_ceil(128)) as u32
    };

    let linear_glsl = if down_bf16_weights {
        if batch == 1 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else if rows8_path {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_rows8_bf16w.comp"
            )
        } else if down_bf16_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_bf16w.comp"
            )
        }
    } else if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode.comp"
        )
    } else if down_rows4 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched_rows4.comp"
        )
    } else if down_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched.comp"
        )
    };
    let stage_start = profile_stages.then(Instant::now);
    let linear_spirv = crate::pipeline::ShaderPipeline::compile_shader(linear_glsl)?;
    finish_vulkan_mlp_kernel_stage_profile(
        "down_shader",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    let mut linear_push = vec![intermediate as u32, out_dim as u32];
    if batch > 1 {
        linear_push.push(batch as u32);
    }
    let linear_handles = vec![
        hidden_buf.handle(),
        down_weight_t.handle(),
        out_buf.handle(),
    ];
    let linear_workgroups = if batch == 1 {
        out_dim.div_ceil(16) as u32
    } else if rows8_path {
        (batch.div_ceil(8) * out_dim.div_ceil(32)) as u32
    } else if down_rows4 || down_bf16_rows4 {
        (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
    } else if down_rows2 {
        (batch.div_ceil(2) * out_dim.div_ceil(32)) as u32
    } else {
        (batch * out_dim.div_ceil(32)) as u32
    };

    let out_data = if chained_dispatch {
        if chained_transfer_submit {
            let stage_start = profile_stages.then(Instant::now);
            let out_data = run_two_stage_compute_pipeline_with_transfer_readback(
                vk_device,
                &x_buf,
                &x_data,
                &out_buf,
                out_size,
                &gate_up_spirv,
                &gate_up_handles,
                &gate_up_push,
                gate_up_workgroups,
                &linear_spirv,
                &linear_handles,
                &linear_push,
                linear_workgroups,
            )
            .context("mlp_decode chained transfer + gate/up + down kernels failed")?;
            finish_vulkan_mlp_kernel_stage_profile(
                "chained_transfer_dispatch_readback",
                batch,
                hidden,
                intermediate,
                out_dim,
                gate_up_bf16_weights,
                down_bf16_weights,
                gate_up_rows2,
                gate_up_rows4,
                down_rows4,
                down_rows2,
                stage_start,
            );
            out_data
        } else {
            let stage_start = profile_stages.then(Instant::now);
            run_two_stage_compute_pipeline(
                vk_device,
                &gate_up_spirv,
                &gate_up_handles,
                &gate_up_push,
                gate_up_workgroups,
                &linear_spirv,
                &linear_handles,
                &linear_push,
                linear_workgroups,
            )
            .context("mlp_decode chained gate/up + down kernels failed")?;
            finish_vulkan_mlp_kernel_stage_profile(
                "chained_dispatch",
                batch,
                hidden,
                intermediate,
                out_dim,
                gate_up_bf16_weights,
                down_bf16_weights,
                gate_up_rows2,
                gate_up_rows4,
                down_rows4,
                down_rows2,
                stage_start,
            );
            let stage_start = profile_stages.then(Instant::now);
            let command_pool = vk_device.transient_command_pool()?;
            let out_data = VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &out_buf,
            )
            .context("failed to read back mlp_decode output")?;
            finish_vulkan_mlp_kernel_stage_profile(
                "readback",
                batch,
                hidden,
                intermediate,
                out_dim,
                gate_up_bf16_weights,
                down_bf16_weights,
                gate_up_rows2,
                gate_up_rows4,
                down_rows4,
                down_rows2,
                stage_start,
            );
            out_data
        }
    } else {
        let stage_start = profile_stages.then(Instant::now);
        run_compute_pipeline(
            vk_device,
            &gate_up_spirv,
            &gate_up_handles,
            gate_up_handles.len(),
            &gate_up_push,
            gate_up_workgroups,
        )
        .context("mlp_decode gate/up kernel failed")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "gate_up_dispatch",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
        let stage_start = profile_stages.then(Instant::now);
        run_compute_pipeline(
            vk_device,
            &linear_spirv,
            &linear_handles,
            linear_handles.len(),
            &linear_push,
            linear_workgroups,
        )
        .context("mlp_decode down kernel failed")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "down_dispatch",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back mlp_decode output")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "readback",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
        out_data
    };
    finish_vulkan_mlp_kernel_stage_profile(
        "total",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        total_start,
    );
    Ok(out_data)
}

/// Dispatch fused single-token GDN gates + recurrent update + gated RMSNorm.
/// Bytes-only fused GDN decode dispatch.
///
/// `input_data` must contain exactly 10 byte slices in the order
/// `[q, k, v, a, b, a_log, dt_bias, state, z, weight]`. Output bytes have
/// logical shape `[batch, 1, nv, dv]` in the caller-chosen recurrent dtype.
/// The optional state bytes match the caller-known state shape and dtype;
/// when `skip_state_readback` is true the state slot is `None` (the caller
/// is responsible for keeping the input state buffer if it still wants to
/// rebuild a tensor for it).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_decode_gates_recurrent_rmsnorm_bytes(
    vk_device: &VulkanDevice,
    input_data: &[Vec<u8>],
    batch: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    eps: f32,
    skip_state_readback: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    anyhow::ensure!(
        input_data.len() == 10,
        "gdn_decode fused expects 10 input byte slices, got {}",
        input_data.len()
    );
    anyhow::ensure!(
        dv <= 256,
        "gdn_decode fused: dv {dv} exceeds shader local capacity 256"
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_decode_gates_recurrent_rmsnorm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [nv as u32, dk as u32, dv as u32, eps.to_bits(), batch as u32];

    if gdn_decode_fused_single_submit_enabled() {
        return dispatch_gdn_decode_gates_recurrent_rmsnorm_single_submit_bytes(
            vk_device,
            input_data,
            &spirv,
            push_constants,
            batch,
            nv,
            dv,
            skip_state_readback,
        );
    }

    let use_host_visible_state = gdn_decode_host_visible_state_enabled();
    let mut buffers = Vec::with_capacity(input_data.len());
    for (idx, data) in input_data.iter().enumerate() {
        let buffer = if use_host_visible_state && idx == 7 {
            let buffer =
                VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
            VulkanBuffer::write_host_visible(device, &buffer, data)?;
            buffer
        } else {
            let buffer =
                VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &buffer,
                data,
            )?;
            buffer
        };
        buffers.push(buffer);
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * nv * dv * 4) as u64)
            .context("failed to create gdn_decode fused output buffer")?;

    let mut all_handles: Vec<vk::Buffer> = buffers.iter().map(|buf| buf.handle()).collect();
    all_handles.push(out_buf.handle());

    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        (batch * nv) as u32,
    )
    .context("gdn_decode_gates_recurrent_rmsnorm kernel failed")?;

    let (out_data, state_data) = {
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back gdn_decode fused output")?;
        let state_data = if skip_state_readback {
            None
        } else if use_host_visible_state {
            Some(VulkanBuffer::read_host_visible(device, &buffers[7]))
        } else {
            Some(VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &buffers[7],
            ))
        }
        .transpose()
        .context("failed to read back gdn_decode fused state")?;
        (out_data, state_data)
    };

    Ok((out_data, state_data))
}

/// Dispatch fused GDN decode while keeping recurrent state device-resident.
///
/// The first call uploads `state_data` into a device-local buffer (callers
/// pass `state_data = Some(bytes)` on the cold path). Later calls pass the
/// returned buffer back via `resident_state` and set `state_data = None`,
/// avoiding the full recurrent-state readback/upload pair; only the small
/// normalized output is copied to the CPU. Output bytes have logical shape
/// `[batch, 1, nv, dv]` in the caller-chosen recurrent dtype.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    a_data: &[u8],
    b_data: &[u8],
    a_log_data: &[u8],
    dt_bias_data: &[u8],
    state_data: Option<&[u8]>,
    z_data: &[u8],
    weight_data: &[u8],
    batch: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    eps: f32,
    resident_state: Option<Arc<VulkanBuffer>>,
) -> Result<(Vec<u8>, Arc<VulkanBuffer>)> {
    anyhow::ensure!(
        dv <= 256,
        "gdn_decode fused resident: dv {dv} exceeds shader local capacity 256"
    );
    anyhow::ensure!(
        resident_state.is_some() || state_data.is_some(),
        "gdn_decode fused resident: either resident_state or state_data must be Some"
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_decode_gates_recurrent_rmsnorm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [nv as u32, dk as u32, dv as u32, eps.to_bits(), batch as u32];

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let (q_buf, q_stage) = make_device_and_staging(q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(v_data)?;
    let (a_buf, a_stage) = make_device_and_staging(a_data)?;
    let (b_buf, b_stage) = make_device_and_staging(b_data)?;
    let (a_log_buf, a_log_stage) = make_device_and_staging(a_log_data)?;
    let (dt_bias_buf, dt_bias_stage) = make_device_and_staging(dt_bias_data)?;
    let (z_buf, z_stage) = make_device_and_staging(z_data)?;
    let (weight_buf, weight_stage) = make_device_and_staging(weight_data)?;

    let state_buf = match resident_state {
        Some(buffer) => buffer,
        None => {
            let data = state_data.expect("state data exists when resident state is absent");
            Arc::new(VulkanBuffer::create_device_local(
                device,
                device_local_mt,
                data.len() as u64,
            )?)
        }
    };
    let state_stage = if let Some(data) = state_data {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Some(staging)
    } else {
        None
    };

    let out_size = (batch * nv * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create gdn_decode fused resident output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create gdn_decode fused resident output staging buffer")?;

    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        a_buf.handle(),
        b_buf.handle(),
        a_log_buf.handle(),
        dt_bias_buf.handle(),
        state_buf.handle(),
        z_buf.handle(),
        weight_buf.handle(),
        out_buf.handle(),
    ];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate gdn_decode fused resident descriptor set")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&a_stage, &a_buf, a_data.len() as u64),
            (&b_stage, &b_buf, b_data.len() as u64),
            (&a_log_stage, &a_log_buf, a_log_data.len() as u64),
            (&dt_bias_stage, &dt_bias_buf, dt_bias_data.len() as u64),
            (&z_stage, &z_buf, z_data.len() as u64),
            (&weight_stage, &weight_buf, weight_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::default().size(size)],
            );
        }
        if let (Some(state_stage), Some(state_data)) = (&state_stage, state_data) {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::default()
                    .size(state_data.len() as u64)
                    ],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, (batch * nv) as u32, 1, 1);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn_decode fused resident dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn_decode fused resident dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)
        .context("failed to read back gdn_decode fused resident output")?;
    let _ = (batch, nv, dv);
    Ok((out_data, state_buf))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_decode_gates_recurrent_rmsnorm_single_submit_bytes(
    vk_device: &VulkanDevice,
    input_data: &[Vec<u8>],
    spirv: &[u8],
    push_constants: [u32; 5],
    batch: usize,
    nv: usize,
    dv: usize,
    skip_state_readback: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    anyhow::ensure!(
        input_data.len() == 10,
        "gdn_decode fused single-submit expects 10 inputs, got {}",
        input_data.len()
    );

    let use_host_visible_state = gdn_decode_host_visible_state_enabled();
    let mut buffers = Vec::with_capacity(input_data.len());
    let mut staging = Vec::with_capacity(input_data.len());
    for (idx, data) in input_data.iter().enumerate() {
        if use_host_visible_state && idx == 7 {
            let buffer =
                VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
            VulkanBuffer::write_host_visible(device, &buffer, data)?;
            buffers.push(buffer);
            staging.push(None);
        } else {
            let buffer =
                VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
            let stage =
                VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
            VulkanBuffer::write_host_visible(device, &stage, data)?;
            buffers.push(buffer);
            staging.push(Some(stage));
        }
    }

    let out_size = (batch * nv * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create gdn_decode fused output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create gdn_decode fused output staging buffer")?;

    let mut all_handles: Vec<vk::Buffer> = buffers.iter().map(|buf| buf.handle()).collect();
    all_handles.push(out_buf.handle());
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (idx, stage) in staging.iter().enumerate() {
            let Some(stage) = stage else {
                continue;
            };
            device.cmd_copy_buffer(
                cmd,
                stage.handle(),
                buffers[idx].handle(),
                &[vk::BufferCopy::default()
                    .size(input_data[idx].len() as u64)
                    ],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, (batch * nv) as u32, 1, 1);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );

        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );
        if !use_host_visible_state && !skip_state_readback {
            let state_stage = staging[7]
                .as_ref()
                .expect("state staging exists when state is device-local");
            device.cmd_copy_buffer(
                cmd,
                buffers[7].handle(),
                state_stage.handle(),
                &[vk::BufferCopy::default()
                    .size(input_data[7].len() as u64)
                    ],
            );
        }

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn_decode fused single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn_decode fused single-submit dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)
        .context("failed to read back gdn_decode fused output")?;
    let state_data = if skip_state_readback {
        None
    } else if use_host_visible_state {
        Some(VulkanBuffer::read_host_visible(device, &buffers[7]))
    } else {
        Some(VulkanBuffer::read_host_visible(
            device,
            staging[7]
                .as_ref()
                .expect("state staging exists when state is device-local"),
        ))
    }
    .transpose()
    .context("failed to read back gdn_decode fused state")?;

    let _ = (batch, nv, dv);
    Ok((out_data, state_data))
}

// ---------------------------------------------------------------------------
// Specialized dispatch functions for GDN kernels
// ---------------------------------------------------------------------------

pub fn dispatch_gdn_gates_cached_bytes(
    vk_device: &VulkanDevice,
    a_data: &[u8],
    b_data: &[u8],
    a_log: &VulkanBuffer,
    dt_bias: &VulkanBuffer,
    nv: usize,
    out_shape: &[usize],
) -> Result<(Vec<u8>, Vec<u8>)> {
    dispatch_gdn_gates_cached_bytes_core(
        vk_device, a_data, b_data, a_log, dt_bias, nv, out_shape,
    )
}

fn dispatch_gdn_gates_cached_bytes_core(
    vk_device: &VulkanDevice,
    a_data: &[u8],
    b_data: &[u8],
    a_log: &VulkanBuffer,
    dt_bias: &VulkanBuffer,
    nv: usize,
    out_shape: &[usize],
) -> Result<(Vec<u8>, Vec<u8>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Compile shader
    let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/gdn_gates.comp");
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers
    let a_buf = VulkanBuffer::create_device_local(device, device_local_mt, a_data.len() as u64)?;
    let b_buf = VulkanBuffer::create_device_local(device, device_local_mt, b_data.len() as u64)?;

    // Create output buffers
    let elem_count: usize = out_shape.iter().product();
    let output_size = (elem_count * 4) as u64; // f32
    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;

    // Push constants: total elements, nv
    let push_constants: [u32; 2] = [elem_count as u32, nv as u32];

    // Workgroup count
    let workgroup_count = elem_count.div_ceil(256) as u32;

    // Build descriptor bindings: a=0, b=1, a_log=2, dt_bias=3, beta_out=4, g_out=5
    let all_handles = vec![
        a_buf.handle(),
        b_buf.handle(),
        a_log.handle(),
        dt_bias.handle(),
        beta_buf.handle(),
        g_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    let (beta_data, g_data) = if gdn_gates_single_submit_enabled() {
        let mut outputs = run_compute_pipeline_with_transfers_readbacks(
            vk_device,
            &[(&a_buf, a_data), (&b_buf, b_data)],
            &[(&beta_buf, output_size), (&g_buf, output_size)],
            &spirv,
            &all_handles,
            &push_constants,
            workgroup_count,
        )
        .context("gdn_gates single-submit dispatch")?;
        anyhow::ensure!(
            outputs.len() == 2,
            "gdn_gates single-submit readback returned wrong count"
        );
        let beta_data = outputs.remove(0);
        let g_data = outputs.remove(0);
        (beta_data, g_data)
    } else {
        if gdn_gates_batched_transfers_enabled() {
            let command_pool = vk_device.transient_command_pool()?;
            upload_buffers_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &[(&a_buf, a_data), (&b_buf, b_data)],
            )?;
        } else {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &a_buf,
                a_data,
            )?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &b_buf,
                b_data,
            )?;
        }

        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            total_bindings,
            &push_constants,
            workgroup_count,
        )?;

        if gdn_gates_batched_transfers_enabled() {
            let command_pool = vk_device.transient_command_pool()?;
            let mut data = read_back_buffers_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &[&beta_buf, &g_buf],
            )?;
            anyhow::ensure!(
                data.len() == 2,
                "gdn_gates batched readback returned wrong count"
            );
            (data.remove(0), data.remove(0))
        } else {
            let command_pool = vk_device.transient_command_pool()?;
            let beta_data = VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &beta_buf,
            )?;
            let g_data = VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &g_buf,
            )?;
            (beta_data, g_data)
        }
    };

    // Cleanup
    drop(a_buf);
    drop(b_buf);
    drop(beta_buf);
    drop(g_buf);

    Ok((beta_data, g_data))
}

pub fn dispatch_gdn_gated_rms_norm_cached_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    z_data: &[u8],
    weight: &VulkanBuffer,
    hidden: usize,
    eps: f32,
    out_shape: &[usize],
) -> Result<Vec<u8>> {
    dispatch_gdn_gated_rms_norm_cached_bytes_core(
        vk_device, x_data, z_data, weight, hidden, eps, out_shape,
    )
}

fn dispatch_gdn_gated_rms_norm_cached_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    z_data: &[u8],
    weight: &VulkanBuffer,
    hidden: usize,
    eps: f32,
    out_shape: &[usize],
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_gated_rms_norm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let z_buf = VulkanBuffer::create_device_local(device, device_local_mt, z_data.len() as u64)?;

    // Create output buffer
    let elem_count: usize = out_shape.iter().product();
    let output_size = (elem_count * 4) as u64; // f32
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;

    // Push constants: rows, hidden, eps
    let rows = elem_count / hidden;
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];

    // Workgroup count: one group per row
    let workgroup_count = rows as u32;

    // Build descriptor bindings: x=0, z=1, weight=2, out=3
    let all_handles = vec![
        x_buf.handle(),
        z_buf.handle(),
        weight.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    let output_data = if gdn_gated_norm_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[(&x_buf, x_data), (&z_buf, z_data)],
            &out_buf,
            output_size,
            &spirv,
            &all_handles,
            &push_constants,
            workgroup_count,
        )
        .context("gdn_gated_rms_norm single-submit dispatch")?
    } else {
        if gdn_gated_norm_batched_uploads_enabled() {
            let command_pool = vk_device.transient_command_pool()?;
            upload_buffers_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &[(&x_buf, x_data), (&z_buf, z_data)],
            )?;
        } else {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                x_data,
            )?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &z_buf,
                z_data,
            )?;
        }

        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            total_bindings,
            &push_constants,
            workgroup_count,
        )?;

        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?
    };

    // Cleanup
    drop(x_buf);
    drop(z_buf);
    drop(out_buf);

    Ok(output_data)
}

pub fn dispatch_causal_conv1d_update_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    state_data: &[u8],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    dispatch_causal_conv1d_update_bytes_core(
        vk_device,
        x_data,
        weight_data,
        state_data,
        batch,
        channels,
        seq_len,
        kernel_size,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_causal_conv1d_update_bytes_core(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    state_data: &[u8],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if kernel_size != 4 {
        anyhow::bail!("causal_conv1d: only kernel_size=4 supported");
    }

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Create input buffers (uploads scheduled inside single-submit helper)
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;

    // Create output buffer (f32)
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // ---- Stage 1: causal_conv1d.comp (output only, no state writes) ----
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let total = batch * channels * seq_len;
    let output_wg = total.div_ceil(256) as u32;

    // ---- Stage 2: causal_conv1d_state_advance.comp ----
    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;
    let state_handles: Vec<vk::Buffer> = vec![x_buf.handle(), state_buf.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels).div_ceil(256) as u32;

    let (out_data, state_data_out) = if causal_conv1d_single_submit_enabled() {
        let readbacks = run_two_stage_compute_pipeline_with_transfers(
            vk_device,
            &[
                (&x_buf, x_data),
                (&weight_buf, weight_data),
                (&state_buf, state_data),
            ],
            &[&out_buf, &state_buf],
            &spirv_output,
            &output_handles,
            &output_push,
            output_wg,
            &spirv_state,
            &state_handles,
            &state_push,
            state_wg,
        )
        .context("causal_conv1d_update single-submit failed")?;
        anyhow::ensure!(
            readbacks.len() == 2,
            "causal_conv1d_update single-submit returned wrong readback count"
        );
        let mut iter = readbacks.into_iter();
        (iter.next().unwrap(), iter.next().unwrap())
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                x_data,
            )?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &weight_buf,
                weight_data,
            )?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &state_buf,
                state_data,
            )?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv_output,
            &output_handles,
            output_handles.len(),
            &output_push,
            output_wg,
        )?;
        run_compute_pipeline(
            vk_device,
            &spirv_state,
            &state_handles,
            2,
            &state_push,
            state_wg,
        )?;
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?;
        let state_data_read = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
        )?;
        (out_data, state_data_read)
    };

    // Cleanup
    drop(x_buf);
    drop(weight_buf);
    drop(state_buf);
    drop(out_buf);

    Ok((out_data, state_data_out))
}

/// Dispatch causal_conv1d prefill kernel (multi-token path).
///
/// Depthwise conv1d with kernel_size=4, silu-fused.
/// `x`: `[B, C, T]` bf16. `weight`: `[C, K]` bf16. `conv_state`: `[B, C, K-1]` f32.
/// Returns `out: [B, C, T]` f32 and updates `conv_state` in-place.
///
/// Two-dispatch approach to avoid data races on conv_state:
/// 1. `causal_conv1d.comp` — computes output only (no state writes)
/// 2. `causal_conv1d_state_advance.comp` — advances state per (b, c) pair
pub fn dispatch_causal_conv1d_prefill_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_data: &[u8],
    state_data: &[u8],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if kernel_size != 4 {
        anyhow::bail!("causal_conv1d: only kernel_size=4 supported");
    }

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Create input buffers (uploads scheduled inside single-submit helper)
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;

    // Create output buffer (f32)
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // ---- Stage 1: causal_conv1d.comp (output only, no state writes) ----
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let total = batch * channels * seq_len;
    let output_wg = total.div_ceil(256) as u32;

    // ---- Stage 2: causal_conv1d_state_advance.comp ----
    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;
    let state_handles: Vec<vk::Buffer> = vec![x_buf.handle(), state_buf.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels).div_ceil(256) as u32;

    let (out_data, state_data) = if causal_conv1d_single_submit_enabled() {
        let readbacks = run_two_stage_compute_pipeline_with_transfers(
            vk_device,
            &[
                (&x_buf, &x_data),
                (&weight_buf, &weight_data),
                (&state_buf, &state_data),
            ],
            &[&out_buf, &state_buf],
            &spirv_output,
            &output_handles,
            &output_push,
            output_wg,
            &spirv_state,
            &state_handles,
            &state_push,
            state_wg,
        )
        .context("causal_conv1d_prefill single-submit failed")?;
        anyhow::ensure!(
            readbacks.len() == 2,
            "causal_conv1d_prefill single-submit returned wrong readback count"
        );
        let mut iter = readbacks.into_iter();
        (iter.next().unwrap(), iter.next().unwrap())
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                &x_data,
            )?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &weight_buf,
                &weight_data,
            )?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &state_buf,
                &state_data,
            )?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv_output,
            &output_handles,
            output_handles.len(),
            &output_push,
            output_wg,
        )?;
        run_compute_pipeline(
            vk_device,
            &spirv_state,
            &state_handles,
            2,
            &state_push,
            state_wg,
        )?;
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?;
        let state_data_read = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
        )?;
        (out_data, state_data_read)
    };

    // Cleanup
    drop(x_buf);
    drop(weight_buf);
    drop(state_buf);
    drop(out_buf);

    Ok((out_data, state_data))
}

/// Dispatch causal_conv1d prefill with an immutable cached f32 weight buffer.
///
/// This keeps the old tensor-weight entry point available as a rollback path,
/// while avoiding one per-layer weight upload and folding the two uploads, two
/// compute dispatches, and two readbacks into one command buffer/queue submit.
pub fn dispatch_causal_conv1d_prefill_cached_weight_bytes(
    vk_device: &VulkanDevice,
    x_data: &[u8],
    weight_buf: &VulkanBuffer,
    state_data: &[u8],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if kernel_size != 4 {
        anyhow::bail!("causal_conv1d: only kernel_size=4 supported");
    }

    let device = vk_device.device();
    let device_local_mt = vk_device.device_local_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let output_wg = ((batch * channels * seq_len).div_ceil(256)) as u32;

    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;
    let state_handles: Vec<vk::Buffer> = vec![x_buf.handle(), state_buf.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels).div_ceil(256) as u32;

    let readbacks = run_two_stage_compute_pipeline_with_transfers(
        vk_device,
        &[(&x_buf, &x_data), (&state_buf, &state_data)],
        &[&out_buf, &state_buf],
        &spirv_output,
        &output_handles,
        &output_push,
        output_wg,
        &spirv_state,
        &state_handles,
        &state_push,
        state_wg,
    )
    .context("causal_conv1d prefill cached-weight single-submit failed")?;
    anyhow::ensure!(
        readbacks.len() == 2,
        "causal_conv1d prefill expected 2 readbacks, got {}",
        readbacks.len()
    );
    let out_data = &readbacks[0];
    let state_data = &readbacks[1];

    Ok((out_data.clone(), state_data.clone()))
}

// ---------------------------------------------------------------------------
// Common pipeline build + dispatch helper to reduce code duplication
// ---------------------------------------------------------------------------

/// Dispatch a cached Vulkan compute pipeline and wait for completion.
///
/// This helper is used by causal_conv1d (two-dispatch path) and gdn
/// kernels. Pipeline state is cached on `VulkanDevice`; descriptor sets are
/// allocated from a reusable transient pool and command buffers remain
/// per-dispatch because they depend on live buffers.
pub fn run_compute_pipeline(
    vk_device: &VulkanDevice,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    total_bindings: usize,
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<()> {
    // Use the actual device per-axis limit rather than the Vulkan
    // spec minimum (65535). Real devices typically support much
    // more (AMD/Strix Halo ≈ 2^31 - 1).
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}; caller should \
         split into a multi-axis dispatch via dispatch_kernel"
    );
    let device = vk_device.device();
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        total_bindings,
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];

    anyhow::ensure!(
        total_bindings <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {total_bindings}"
    );
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    // Descriptor writes
    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let descriptor_write_infos: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bui)| make_write_descriptor_set_buf(descriptor_set, i as u32, bui))
            .collect();
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    // Command buffer + dispatch
    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline")?;

    // Cleanup
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(())
}

/// 3D-grid variant of `run_compute_pipeline`. Same caching/descriptor
/// machinery, but dispatches `(x, y, z)` workgroups for shaders that
/// use 2D (transpose) or 3D workgroup layouts.
pub fn run_compute_pipeline_3d(
    vk_device: &VulkanDevice,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    total_bindings: usize,
    push_constants: &[u32],
    workgroup_count: (u32, u32, u32),
) -> Result<()> {
    let (wx, wy, wz) = workgroup_count;
    let limit_x = vk_device.max_compute_work_group_count(0);
    let limit_y = vk_device.max_compute_work_group_count(1);
    let limit_z = vk_device.max_compute_work_group_count(2);
    anyhow::ensure!(
        wx <= limit_x && wy <= limit_y && wz <= limit_z,
        "run_compute_pipeline_3d: workgroups ({wx},{wy},{wz}) exceed device limits \
         ({limit_x},{limit_y},{limit_z})"
    );
    let device = vk_device.device();
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        total_bindings,
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    anyhow::ensure!(
        total_bindings <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {total_bindings}"
    );
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };
    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let descriptor_write_infos: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bui)| make_write_descriptor_set_buf(descriptor_set, i as u32, bui))
            .collect();
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }
    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];
    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, wx, wy, wz);
        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }
    vk_device.submit_and_wait(cmd, "run_compute_pipeline_3d")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }
    Ok(())
}

/// Single-submit upload + dispatch + readback. Sequences a host-to-device
/// copy of `upload_data` into `upload_dst`, runs one compute kernel, then
/// copies `readback_size` bytes from `readback_src` into a host-visible
/// staging buffer — all in one command buffer and one queue submit. Saves
/// the two extra `vkQueueSubmit` + fence-wait round trips the
/// `extract → upload → dispatch → readback` decode kernels otherwise pay
/// per call (≈ 600 µs on NVIDIA Vulkan).
#[allow(clippy::too_many_arguments)]
fn run_compute_pipeline_with_transfer_readback(
    vk_device: &VulkanDevice,
    upload_dst: &VulkanBuffer,
    upload_data: &[u8],
    readback_src: &VulkanBuffer,
    readback_size: u64,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<Vec<u8>> {
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline_with_transfer_readback: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}"
    );
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let upload_stage =
        VulkanBuffer::create_host_visible(device, host_visible_mt, upload_data.len() as u64)
            .context("failed to create transfer-readback upload staging buffer")?;
    VulkanBuffer::write_host_visible(device, &upload_stage, upload_data)?;
    let readback_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, readback_size)
        .context("failed to create transfer-readback readback staging buffer")?;

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    anyhow::ensure!(
        all_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        all_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [set_layout];
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate transfer-readback descriptor set")?[0]
    };

    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfer-readback command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfer-readback command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            upload_stage.handle(),
            upload_dst.handle(),
            &[vk::BufferCopy::default()
                .size(upload_data.len() as u64)
                ],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            readback_src.handle(),
            readback_stage.handle(),
            &[vk::BufferCopy::default().size(readback_size)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end transfer-readback command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline_with_transfer_readback")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfer-readback descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &readback_stage)
        .context("failed to read transfer-readback output")
}

fn create_packed_upload_stage(
    device: &Arc<ash::Device>,
    host_visible_mt: u32,
    uploads: &[(&VulkanBuffer, &[u8])],
    context: &'static str,
) -> Result<(Option<VulkanBuffer>, Vec<u64>)> {
    if uploads.is_empty() {
        return Ok((None, Vec::new()));
    }
    let segments = uploads.iter().map(|(_, data)| *data).collect::<Vec<_>>();
    let (stage, offsets) =
        VulkanBuffer::create_host_visible_with_segments(device, host_visible_mt, &segments)
            .with_context(|| format!("failed to create {context} upload staging buffer"))?;
    Ok((Some(stage), offsets))
}

/// Single-submit multi-upload + dispatch + multi-readback. Variant of
/// `run_compute_pipeline_with_transfer_readback` for kernels that take
/// several disjoint input buffers AND produce several disjoint output
/// buffers (e.g. `dispatch_gdn_gates_cached_bytes`'s beta + g pair). Schedules
/// all host→device copies, the compute dispatch, and every device→host
/// readback into one command buffer.
#[allow(clippy::too_many_arguments)]
fn run_compute_pipeline_with_transfers_readbacks(
    vk_device: &VulkanDevice,
    uploads: &[(&VulkanBuffer, &[u8])],
    readbacks: &[(&VulkanBuffer, u64)],
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<Vec<Vec<u8>>> {
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline_with_transfers_readbacks: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}"
    );
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let (upload_stage, upload_offsets) = create_packed_upload_stage(
        device,
        host_visible_mt,
        uploads,
        "transfers-readbacks",
    )?;
    let mut readback_offsets = Vec::with_capacity(readbacks.len());
    let mut readback_total = 0u64;
    for (idx, (_, size)) in readbacks.iter().enumerate() {
        anyhow::ensure!(
            *size > 0,
            "run_compute_pipeline_with_transfers_readbacks[{idx}]: readback size must be non-zero"
        );
        readback_offsets.push(readback_total);
        readback_total = readback_total.checked_add(*size).ok_or_else(|| {
            anyhow::anyhow!("run_compute_pipeline_with_transfers_readbacks: readback size overflow")
        })?;
    }
    let readback_stage = if readback_total > 0 {
        Some(
            VulkanBuffer::create_host_visible(device, host_visible_mt, readback_total)
                .context("failed to create transfers-readbacks staging buffer")?,
        )
    } else {
        None
    };

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    anyhow::ensure!(
        all_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        all_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [set_layout];
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate transfers-readbacks descriptor set")?[0]
    };

    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfers-readbacks command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfers-readbacks command buffer")?;
        if let Some(stage) = &upload_stage {
            for (idx, (dst, data)) in uploads.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .src_offset(upload_offsets[idx])
                    .size(data.len() as u64);
                device.cmd_copy_buffer(cmd, stage.handle(), dst.handle(), &[copy]);
            }
        }
        if !uploads.is_empty() {
            let upload_barrier = make_memory_barrier(
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_barrier],
                &[],
                &[],
            );
        }

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        if let Some(stage) = &readback_stage {
            for (idx, (src, size)) in readbacks.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .dst_offset(readback_offsets[idx])
                    .size(*size);
                device.cmd_copy_buffer(cmd, src.handle(), stage.handle(), &[copy]);
            }
        }
        device
            .end_command_buffer(cmd)
            .context("failed to end transfers-readbacks command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline_with_transfers_readbacks")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfers-readbacks descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let staging_bytes = if let Some(stage) = readback_stage {
        VulkanBuffer::read_host_visible(device, &stage)
            .context("failed to read transfers-readbacks output")?
    } else {
        Vec::new()
    };
    let mut outputs = Vec::with_capacity(readbacks.len());
    for (idx, (_, size)) in readbacks.iter().enumerate() {
        let start = usize::try_from(readback_offsets[idx])
            .context("run_compute_pipeline_with_transfers_readbacks: offset exceeds usize")?;
        let len = usize::try_from(*size)
            .context("run_compute_pipeline_with_transfers_readbacks: size exceeds usize")?;
        let end = start.checked_add(len).ok_or_else(|| {
            anyhow::anyhow!("run_compute_pipeline_with_transfers_readbacks[{idx}]: slice overflow")
        })?;
        outputs.push(staging_bytes[start..end].to_vec());
    }
    Ok(outputs)
}

/// Single-submit multi-upload + dispatch + readback. Variant of
/// `run_compute_pipeline_with_transfer_readback` for kernels that take
/// several disjoint input buffers (e.g. paged_attn_decode_batch's
/// Q/K/V/seq_lens uploads). Schedules all host→device copies, the compute
/// dispatch, and a single device→host readback into one command buffer.
#[allow(clippy::too_many_arguments)]
fn run_compute_pipeline_with_transfers_readback(
    vk_device: &VulkanDevice,
    uploads: &[(&VulkanBuffer, &[u8])],
    readback_src: &VulkanBuffer,
    readback_size: u64,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<Vec<u8>> {
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline_with_transfers_readback: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}"
    );
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let (upload_stage, upload_offsets) = create_packed_upload_stage(
        device,
        host_visible_mt,
        uploads,
        "transfers-readback",
    )?;
    let readback_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, readback_size)
        .context("failed to create transfers-readback readback staging buffer")?;

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    anyhow::ensure!(
        all_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        all_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [set_layout];
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate transfers-readback descriptor set")?[0]
    };

    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfers-readback command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfers-readback command buffer")?;
        if let Some(stage) = &upload_stage {
            for (idx, (dst, data)) in uploads.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .src_offset(upload_offsets[idx])
                    .size(data.len() as u64);
                device.cmd_copy_buffer(cmd, stage.handle(), dst.handle(), &[copy]);
            }
        }
        if !uploads.is_empty() {
            let upload_barrier = make_memory_barrier(
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_barrier],
                &[],
                &[],
            );
        }

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            readback_src.handle(),
            readback_stage.handle(),
            &[vk::BufferCopy::default().size(readback_size)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end transfers-readback command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline_with_transfers_readback")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfers-readback descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &readback_stage)
        .context("failed to read transfers-readback output")
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage_compute_pipeline(
    vk_device: &VulkanDevice,
    first_spirv: &[u8],
    first_handles: &[vk::Buffer],
    first_push_constants: &[u32],
    first_workgroup_count: u32,
    second_spirv: &[u8],
    second_handles: &[vk::Buffer],
    second_push_constants: &[u32],
    second_workgroup_count: u32,
) -> Result<()> {
    let device = vk_device.device();
    let (first_set_layout, first_layout, first_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            first_spirv,
            first_handles.len(),
            (first_push_constants.len() * 4) as u32,
        )?;
    let (second_set_layout, second_layout, second_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            second_spirv,
            second_handles.len(),
            (second_push_constants.len() * 4) as u32,
        )?;
    anyhow::ensure!(
        first_handles.len() + second_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        first_handles.len() + second_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [first_set_layout, second_set_layout];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate two-stage descriptor sets")?
    };
    let first_descriptor_set = descriptor_sets[0];
    let second_descriptor_set = descriptor_sets[1];

    {
        let first_buf_infos: Vec<vk::DescriptorBufferInfo> = first_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let second_buf_infos: Vec<vk::DescriptorBufferInfo> = second_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let mut descriptor_write_infos: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(first_buf_infos.len() + second_buf_infos.len());
        descriptor_write_infos.extend(
            first_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(first_descriptor_set, i as u32, info)
            }),
        );
        descriptor_write_infos.extend(
            second_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(second_descriptor_set, i as u32, info)
            }),
        );
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate two-stage command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin two-stage command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, first_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            first_layout,
            0,
            &[first_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            first_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(first_push_constants),
        );
        device.cmd_dispatch(cmd, first_workgroup_count, 1, 1);

        let first_to_second_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[first_to_second_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, second_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            second_layout,
            0,
            &[second_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            second_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(second_push_constants),
        );
        device.cmd_dispatch(cmd, second_workgroup_count, 1, 1);

        let second_to_readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[second_to_readback_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end two-stage command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_two_stage_compute_pipeline")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset two-stage transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage_compute_pipeline_with_transfer_readback(
    vk_device: &VulkanDevice,
    upload_dst: &VulkanBuffer,
    upload_data: &[u8],
    readback_src: &VulkanBuffer,
    readback_size: u64,
    first_spirv: &[u8],
    first_handles: &[vk::Buffer],
    first_push_constants: &[u32],
    first_workgroup_count: u32,
    second_spirv: &[u8],
    second_handles: &[vk::Buffer],
    second_push_constants: &[u32],
    second_workgroup_count: u32,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();
    let upload_stage =
        VulkanBuffer::create_host_visible(device, host_visible_mt, upload_data.len() as u64)
            .context("failed to create two-stage upload staging buffer")?;
    VulkanBuffer::write_host_visible(device, &upload_stage, upload_data)?;
    let readback_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, readback_size)
        .context("failed to create two-stage readback staging buffer")?;

    let (first_set_layout, first_layout, first_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            first_spirv,
            first_handles.len(),
            (first_push_constants.len() * 4) as u32,
        )?;
    let (second_set_layout, second_layout, second_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            second_spirv,
            second_handles.len(),
            (second_push_constants.len() * 4) as u32,
        )?;
    anyhow::ensure!(
        first_handles.len() + second_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        first_handles.len() + second_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [first_set_layout, second_set_layout];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate transfer two-stage descriptor sets")?
    };
    let first_descriptor_set = descriptor_sets[0];
    let second_descriptor_set = descriptor_sets[1];

    {
        let first_buf_infos: Vec<vk::DescriptorBufferInfo> = first_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let second_buf_infos: Vec<vk::DescriptorBufferInfo> = second_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let mut descriptor_write_infos: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(first_buf_infos.len() + second_buf_infos.len());
        descriptor_write_infos.extend(
            first_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(first_descriptor_set, i as u32, info)
            }),
        );
        descriptor_write_infos.extend(
            second_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(second_descriptor_set, i as u32, info)
            }),
        );
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfer two-stage command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfer two-stage command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            upload_stage.handle(),
            upload_dst.handle(),
            &[vk::BufferCopy::default()
                .size(upload_data.len() as u64)
                ],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, first_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            first_layout,
            0,
            &[first_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            first_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(first_push_constants),
        );
        device.cmd_dispatch(cmd, first_workgroup_count, 1, 1);

        let first_to_second_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[first_to_second_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, second_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            second_layout,
            0,
            &[second_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            second_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(second_push_constants),
        );
        device.cmd_dispatch(cmd, second_workgroup_count, 1, 1);

        let second_to_readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[second_to_readback_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            readback_src.handle(),
            readback_stage.handle(),
            &[vk::BufferCopy::default().size(readback_size)],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end transfer two-stage command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_two_stage_compute_pipeline_with_transfer_readback")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfer two-stage transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &readback_stage)
        .context("failed to read transfer two-stage output")
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage_compute_pipeline_with_transfers(
    vk_device: &VulkanDevice,
    uploads: &[(&VulkanBuffer, &[u8])],
    readbacks: &[&VulkanBuffer],
    first_spirv: &[u8],
    first_handles: &[vk::Buffer],
    first_push_constants: &[u32],
    first_workgroup_count: u32,
    second_spirv: &[u8],
    second_handles: &[vk::Buffer],
    second_push_constants: &[u32],
    second_workgroup_count: u32,
) -> Result<Vec<Vec<u8>>> {
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let (upload_stage, upload_offsets) =
        create_packed_upload_stage(device, host_visible_mt, uploads, "two-stage")?;

    let mut readback_offsets = Vec::with_capacity(readbacks.len());
    let mut readback_total = 0u64;
    for (idx, buffer) in readbacks.iter().enumerate() {
        anyhow::ensure!(
            buffer.size() > 0,
            "run_two_stage_compute_pipeline_with_transfers[{idx}]: readback size must be non-zero"
        );
        readback_offsets.push(readback_total);
        readback_total = readback_total.checked_add(buffer.size()).ok_or_else(|| {
            anyhow::anyhow!("run_two_stage_compute_pipeline_with_transfers: readback size overflow")
        })?;
    }
    let readback_stage = if readback_total > 0 {
        Some(
            VulkanBuffer::create_host_visible(device, host_visible_mt, readback_total)
                .context("failed to create two-stage readback staging buffer")?,
        )
    } else {
        None
    };

    let (first_set_layout, first_layout, first_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            first_spirv,
            first_handles.len(),
            (first_push_constants.len() * 4) as u32,
        )?;
    let (second_set_layout, second_layout, second_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            second_spirv,
            second_handles.len(),
            (second_push_constants.len() * 4) as u32,
        )?;
    anyhow::ensure!(
        first_handles.len() + second_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        first_handles.len() + second_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [first_set_layout, second_set_layout];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate transfer two-stage descriptor sets")?
    };
    let first_descriptor_set = descriptor_sets[0];
    let second_descriptor_set = descriptor_sets[1];

    {
        let first_buf_infos: Vec<vk::DescriptorBufferInfo> = first_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let second_buf_infos: Vec<vk::DescriptorBufferInfo> = second_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::default()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    
            })
            .collect();
        let mut descriptor_write_infos: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(first_buf_infos.len() + second_buf_infos.len());
        descriptor_write_infos.extend(
            first_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(first_descriptor_set, i as u32, info)
            }),
        );
        descriptor_write_infos.extend(
            second_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(second_descriptor_set, i as u32, info)
            }),
        );
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfer two-stage command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfer two-stage command buffer")?;

        if let Some(stage) = &upload_stage {
            for (idx, (dst, data)) in uploads.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .src_offset(upload_offsets[idx])
                    .size(data.len() as u64);
                device.cmd_copy_buffer(cmd, stage.handle(), dst.handle(), &[copy]);
            }
        }
        if !uploads.is_empty() {
            let upload_barrier = make_memory_barrier(
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_barrier],
                &[],
                &[],
            );
        }

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, first_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            first_layout,
            0,
            &[first_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            first_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(first_push_constants),
        );
        device.cmd_dispatch(cmd, first_workgroup_count, 1, 1);

        let first_to_second_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[first_to_second_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, second_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            second_layout,
            0,
            &[second_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            second_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(second_push_constants),
        );
        device.cmd_dispatch(cmd, second_workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );

        if let Some(stage) = &readback_stage {
            for (idx, src) in readbacks.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .dst_offset(readback_offsets[idx])
                    .size(src.size());
                device.cmd_copy_buffer(cmd, src.handle(), stage.handle(), &[copy]);
            }
        }

        device
            .end_command_buffer(cmd)
            .context("failed to end transfer two-stage command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_two_stage_compute_pipeline_with_transfers")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfer two-stage transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let staging_bytes = if let Some(stage) = readback_stage {
        VulkanBuffer::read_host_visible(device, &stage)
            .context("failed to read transfer two-stage outputs")?
    } else {
        Vec::new()
    };
    let mut outputs = Vec::with_capacity(readbacks.len());
    for (idx, src) in readbacks.iter().enumerate() {
        let start = usize::try_from(readback_offsets[idx])
            .context("run_two_stage_compute_pipeline_with_transfers: offset exceeds usize")?;
        let len = usize::try_from(src.size())
            .context("run_two_stage_compute_pipeline_with_transfers: size exceeds usize")?;
        let end = start.checked_add(len).ok_or_else(|| {
            anyhow::anyhow!("run_two_stage_compute_pipeline_with_transfers[{idx}]: slice overflow")
        })?;
        outputs.push(staging_bytes[start..end].to_vec());
    }
    Ok(outputs)
}

// ---------------------------------------------------------------------------
// GDN forward substitution (triangular solve) kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN forward substitution kernel.
///
/// Computes W = (I + A_strict)^{-1} (beta * V_prime)
/// A_strict: [B,H,C,C] lower-triangular, V_prime: [B,H,C,dv], beta: [B,H,C]
/// Output: W: [B,H,C,dv]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_forward_substitution_bytes(
    vk_device: &VulkanDevice,
    a_strict_bf16: &[u8],
    v_prime_bf16: &[u8],
    beta_bf16: &[u8],
    batch: usize,
    heads: usize,
    chunk: usize,
    dv: usize,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        a_strict_bf16.len() == batch * heads * chunk * chunk * 2,
        "gdn_forward_substitution: a_strict bytes {} mismatch expected {}",
        a_strict_bf16.len(),
        batch * heads * chunk * chunk * 2,
    );
    anyhow::ensure!(
        v_prime_bf16.len() == batch * heads * chunk * dv * 2,
        "gdn_forward_substitution: v_prime bytes {} mismatch expected {}",
        v_prime_bf16.len(),
        batch * heads * chunk * dv * 2,
    );
    anyhow::ensure!(
        beta_bf16.len() == batch * heads * chunk * 2,
        "gdn_forward_substitution: beta bytes {} mismatch expected {}",
        beta_bf16.len(),
        batch * heads * chunk * 2,
    );

    // Compile shader
    let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/solve_tri.comp");
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let a_strict_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, a_strict_bf16.len() as u64)?;
    VulkanBuffer::upload_data(
        device,
        host_visible_mt,
        queue,
        qfi,
        &a_strict_buf,
        a_strict_bf16,
    )?;

    let v_prime_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, v_prime_bf16.len() as u64)?;
    VulkanBuffer::upload_data(
        device,
        host_visible_mt,
        queue,
        qfi,
        &v_prime_buf,
        v_prime_bf16,
    )?;

    let beta_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, beta_bf16.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, beta_bf16)?;

    // Create output buffer (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * chunk * dv;
    let workgroup_count = ((total + 255) / 256) as u32;

    // Bindings: A_strict=0, V_prime=1, beta=2, out=3
    let all_handles = vec![
        a_strict_buf.handle(),
        v_prime_buf.handle(),
        beta_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;

    // Cleanup
    drop(a_strict_buf);
    drop(v_prime_buf);
    drop(beta_buf);
    drop(out_buf);

    Ok(out_data)
}

/// Copy same-sized device-local row buffers into one contiguous batch buffer.
///
/// The layout is byte-row-major: `rows[row]` is copied into
/// `[row * row_size .. (row + 1) * row_size)` in the returned buffer.
/// Used by resident batched decode for both recurrent and conv GDN state.
pub fn copy_device_buffer_rows_to_batch(
    vk_device: &VulkanDevice,
    rows: &[Arc<VulkanBuffer>],
) -> Result<Arc<VulkanBuffer>> {
    anyhow::ensure!(
        !rows.is_empty(),
        "copy_device_buffer_rows_to_batch requires at least one row"
    );
    let row_size = rows[0].size();
    anyhow::ensure!(
        row_size > 0,
        "copy_device_buffer_rows_to_batch row size must be non-zero"
    );
    for (idx, row) in rows.iter().enumerate() {
        anyhow::ensure!(
            row.size() == row_size,
            "copy_device_buffer_rows_to_batch row {idx} size {} != row 0 size {row_size}",
            row.size()
        );
    }

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let batch_buf = Arc::new(VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        row_size * rows.len() as u64,
    )?);

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate row-to-batch copy command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin row-to-batch copy command buffer")?;
        for (row_idx, row) in rows.iter().enumerate() {
            device.cmd_copy_buffer(
                cmd,
                row.handle(),
                batch_buf.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(row_size * row_idx as u64)
                    .size(row_size)
                    ],
            );
        }
        let copy_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[copy_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end row-to-batch copy command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit row-to-batch copy")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for row-to-batch copy")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(batch_buf)
}

/// Split a contiguous device-local batch buffer into same-sized row buffers.
///
/// `batch_buffer` is interpreted as `batch` byte rows of equal size.
pub fn split_device_buffer_batch_rows(
    vk_device: &VulkanDevice,
    batch_buffer: &VulkanBuffer,
    batch: usize,
) -> Result<Vec<Arc<VulkanBuffer>>> {
    anyhow::ensure!(
        batch > 0,
        "split_device_buffer_batch_rows requires a non-zero batch"
    );
    anyhow::ensure!(
        batch_buffer.size() % batch as u64 == 0,
        "split_device_buffer_batch_rows buffer size {} is not divisible by batch {batch}",
        batch_buffer.size()
    );
    let row_size = batch_buffer.size() / batch as u64;
    anyhow::ensure!(
        row_size > 0,
        "split_device_buffer_batch_rows row size must be non-zero"
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let mut rows = Vec::with_capacity(batch);
    for _ in 0..batch {
        rows.push(Arc::new(VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            row_size,
        )?));
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate batch-to-row copy command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin batch-to-row copy command buffer")?;
        let pre_copy_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[pre_copy_barrier],
            &[],
            &[],
        );
        for (row_idx, row) in rows.iter().enumerate() {
            device.cmd_copy_buffer(
                cmd,
                batch_buffer.handle(),
                row.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(row_size * row_idx as u64)
                    .dst_offset(0)
                    .size(row_size)
                    ],
            );
        }
        let post_copy_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[post_copy_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end batch-to-row copy command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batch-to-row copy")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batch-to-row copy")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(rows)
}

// ---------------------------------------------------------------------------
// GDN recurrent step kernel
pub fn copy_gdn_recurrent_state_rows_to_batch(
    vk_device: &VulkanDevice,
    rows: &[Arc<VulkanBuffer>],
) -> Result<Arc<VulkanBuffer>> {
    copy_device_buffer_rows_to_batch(vk_device, rows)
}

pub fn split_gdn_recurrent_state_batch_rows(
    vk_device: &VulkanDevice,
    batch_buffer: &VulkanBuffer,
    batch: usize,
) -> Result<Vec<Arc<VulkanBuffer>>> {
    split_device_buffer_batch_rows(vk_device, batch_buffer, batch)
}

/// Bytes-only single-token GDN recurrent dispatch.
///
/// Inputs are expected to be in `[batch, heads, dk]` / `[batch, heads, dv]`
/// row-major layout matching the regular GQA-expanded path. The caller
/// passes the shape parameters directly. Output bytes are `[batch, heads, dv]`
/// in the kernel's recurrent dtype; the optional state bytes match the
/// caller-known state shape and dtype.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_with_options_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: &[u8],
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    skip_state_readback: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let profile_kernel_stages = profile_vulkan_gdn_recurrent_kernel_stages_enabled();

    let single_submit = gdn_recurrent_single_submit_enabled();
    let parallel_reduce = single_submit && use_gdn_recurrent_parallel_reduce(dk, dv);
    let stage_profile = profile_kernel_stages.then(Instant::now);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "compile_shader",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        single_submit,
        skip_state_readback,
        stage_profile,
    );

    if single_submit {
        return dispatch_gdn_recurrent_step_single_submit_bytes(
            vk_device,
            q_data,
            k_data,
            v_data,
            beta_data,
            g_data,
            state_data,
            &spirv,
            batch,
            heads,
            heads,
            dk,
            dv,
            parallel_reduce,
            skip_state_readback,
            profile_kernel_stages,
            None,
        );
    }

    // Create input buffers + upload.
    let make_input_buf = |data: &[u8]| -> Result<VulkanBuffer> {
        let buf = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &buf,
            data,
        )?;
        Ok(buf)
    };
    let q_buf = make_input_buf(q_data)?;
    let k_buf = make_input_buf(k_data)?;
    let v_buf = make_input_buf(v_data)?;
    let beta_buf = make_input_buf(beta_data)?;
    let g_buf = make_input_buf(g_data)?;
    // State is mutable — upload, dispatch, read back. On Strix Halo, direct
    // host-visible state is faster for batch 1, while batch >1 benefits from
    // device-local state plus explicit staging copies.
    let host_visible_state = gdn_recurrent_use_host_visible_state(batch);
    let state_buf = if host_visible_state {
        VulkanBuffer::create_host_visible(device, host_visible_mt, state_data.len() as u64)?
    } else {
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?
    };
    if host_visible_state {
        VulkanBuffer::write_host_visible(device, &state_buf, state_data)?;
    } else {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
            state_data,
        )?;
    }

    // Create output buffer (f32 shader output, converted to bf16 below).
    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, value heads, seq_len, dk, dv, q/k heads. seq_len
    // is always 1 for this single-token kernel.
    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        heads as u32,
    ];

    // Workgroup count: total elements / 256
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;

    // Bindings: Q=0, K=1, V=2, beta=3, g=4, state=5, out=6
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output and updated state
    let (out_data, state_data_out) = {
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?;
        let state_data_out = if skip_state_readback {
            None
        } else if host_visible_state {
            Some(VulkanBuffer::read_host_visible(device, &state_buf)?)
        } else {
            Some(VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &state_buf,
            )?)
        };
        (out_data, state_data_out)
    };

    // Cleanup
    drop(q_buf);
    drop(k_buf);
    drop(v_buf);
    drop(beta_buf);
    drop(g_buf);
    drop(state_buf);
    drop(out_buf);

    Ok((out_data, state_data_out))
}

/// Dispatch a single-token recurrent step while keeping `state` resident.
///
/// The first call uploads the CPU state into a device-local Vulkan buffer and
/// returns it. Later calls can pass that buffer back and avoid the full state
/// upload/readback pair; only the small recurrent output is copied to the CPU.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_resident_state_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: Option<&[u8]>,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    resident_state: Option<Arc<VulkanBuffer>>,
) -> Result<(Vec<u8>, Arc<VulkanBuffer>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let parallel_reduce = use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let (q_buf, q_stage) = make_device_and_staging(&q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(&k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(&v_data)?;
    let (beta_buf, beta_stage) = make_device_and_staging(&beta_data)?;
    let (g_buf, g_stage) = make_device_and_staging(&g_data)?;

    let state_buf = match resident_state {
        Some(buffer) => buffer,
        None => {
            let data = state_data
                .as_ref()
                .expect("state data exists when resident state is absent");
            Arc::new(VulkanBuffer::create_device_local(
                device,
                device_local_mt,
                data.len() as u64,
            )?)
        }
    };
    let state_stage = if let Some(data) = &state_data {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Some(staging)
    } else {
        None
    };

    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)?;

    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        heads as u32,
    ];
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;
    let dispatch_counts = if parallel_reduce {
        (batch as u32, heads as u32, dv as u32)
    } else {
        (workgroup_count, 1, 1)
    };
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&beta_stage, &beta_buf, beta_data.len() as u64),
            (&g_stage, &g_buf, g_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::default().size(size)],
            );
        }
        if let (Some(state_stage), Some(state_data)) = (&state_stage, &state_data) {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::default()
                    .size(state_data.len() as u64)
                    ],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, dispatch_counts.0, dispatch_counts.1, dispatch_counts.2);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn recurrent resident-state dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn recurrent resident-state dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let _ = (batch, heads, dv);
    Ok((out_data, state_buf))
}

/// Dispatch a native-head single-token recurrent step while keeping `state`
/// resident. `q`/`k` are `[batch, 1, q_heads, dk]`; value-side tensors and
/// state use `heads`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_native_head_last_resident_state_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: Option<&[u8]>,
    batch: usize,
    seq_len: usize,
    q_heads: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    resident_state: Option<Arc<VulkanBuffer>>,
) -> Result<(Vec<u8>, Arc<VulkanBuffer>)> {
    // Caller-supplied shape values (extracted from candle Tensor dims via the kt boundary).
    // Validation that was previously here is the caller's responsibility now, but we still
    // hold derived invariants for the kernel dispatch.
    let (k_batch, k_seq_len, k_heads, k_dk) = (batch, seq_len, q_heads, dk);
    let (v_batch, v_seq_len) = (batch, seq_len);
    let (beta_batch, beta_seq_len, beta_heads) = (batch, seq_len, heads);
    let (g_batch, g_seq_len, g_heads) = (batch, seq_len, heads);
    let (state_batch, state_heads, state_dk, state_dv) = (batch, heads, dk, dv);

    anyhow::ensure!(
        seq_len == 1,
        "native-head resident recurrent expects seq_len=1"
    );
    anyhow::ensure!(
        (k_batch, k_seq_len, k_heads, k_dk) == (batch, seq_len, q_heads, dk),
        "native-head resident recurrent k shape mismatch"
    );
    anyhow::ensure!(
        (v_batch, v_seq_len) == (batch, seq_len),
        "native-head resident recurrent v batch/seq mismatch"
    );
    anyhow::ensure!(
        (beta_batch, beta_seq_len, beta_heads) == (batch, seq_len, heads),
        "native-head resident recurrent beta shape mismatch"
    );
    anyhow::ensure!(
        (g_batch, g_seq_len, g_heads) == (batch, seq_len, heads),
        "native-head resident recurrent g shape mismatch"
    );
    anyhow::ensure!(
        (state_batch, state_heads, state_dk, state_dv) == (batch, heads, dk, dv),
        "native-head resident recurrent state shape mismatch"
    );
    anyhow::ensure!(
        q_heads > 0,
        "native-head resident recurrent q_heads must be positive"
    );
    anyhow::ensure!(
        heads % q_heads == 0,
        "native-head resident recurrent heads {heads} must be divisible by q_heads {q_heads}"
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // q_data, k_data, v_data, beta_data, g_data, state_data are now caller-supplied bytes.

    let parallel_reduce = use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let (q_buf, q_stage) = make_device_and_staging(&q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(&k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(&v_data)?;
    let (beta_buf, beta_stage) = make_device_and_staging(&beta_data)?;
    let (g_buf, g_stage) = make_device_and_staging(&g_data)?;

    let state_buf = match resident_state {
        Some(buffer) => buffer,
        None => {
            let data = state_data
                .as_ref()
                .expect("state data exists when resident state is absent");
            Arc::new(VulkanBuffer::create_device_local(
                device,
                device_local_mt,
                data.len() as u64,
            )?)
        }
    };
    let state_stage = if let Some(data) = &state_data {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Some(staging)
    } else {
        None
    };

    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)?;

    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        q_heads as u32,
    ];
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;
    let dispatch_counts = if parallel_reduce {
        (batch as u32, heads as u32, dv as u32)
    } else {
        (workgroup_count, 1, 1)
    };
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&beta_stage, &beta_buf, beta_data.len() as u64),
            (&g_stage, &g_buf, g_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::default().size(size)],
            );
        }
        if let (Some(state_stage), Some(state_data)) = (&state_stage, &state_data) {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::default()
                    .size(state_data.len() as u64)
                    ],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, dispatch_counts.0, dispatch_counts.1, dispatch_counts.2);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit native-head gdn recurrent resident-state dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for native-head gdn recurrent resident-state dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let out_shape = vec![batch, heads, dv];
    let _ = out_shape;
    Ok((out_data, state_buf))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_recurrent_step_single_submit_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: &[u8],
    spirv: &[u8],
    batch: usize,
    heads: usize,
    q_heads: usize,
    dk: usize,
    dv: usize,
    parallel_reduce: bool,
    skip_state_readback: bool,
    profile_kernel_stages: bool,
    dispatch_counts_override: Option<(u32, u32, u32)>,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let (q_buf, q_stage) = make_device_and_staging(q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(v_data)?;
    let (beta_buf, beta_stage) = make_device_and_staging(beta_data)?;
    let (g_buf, g_stage) = make_device_and_staging(g_data)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "make_input_staging",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let host_visible_state = gdn_recurrent_use_host_visible_state(batch);
    let state_buf = if host_visible_state {
        let buf =
            VulkanBuffer::create_host_visible(device, host_visible_mt, state_data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &buf, state_data)?;
        buf
    } else {
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?
    };
    let state_stage = if host_visible_state {
        None
    } else {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, state_data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, state_data)?;
        Some(staging)
    };
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "make_state_staging",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "create_output_buffers",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        q_heads as u32,
    ];
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;
    let dispatch_counts = if parallel_reduce {
        (batch as u32, heads as u32, dv as u32)
    } else {
        (workgroup_count, 1, 1)
    };
    let dispatch_counts = dispatch_counts_override.unwrap_or(dispatch_counts);
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    ,
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::default()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "pipeline_descriptor_setup",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&beta_stage, &beta_buf, beta_data.len() as u64),
            (&g_stage, &g_buf, g_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::default().size(size)],
            );
        }
        if let Some(state_stage) = &state_stage {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::default()
                    .size(state_data.len() as u64)
                    ],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, dispatch_counts.0, dispatch_counts.1, dispatch_counts.2);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );

        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::default().size(out_size)],
        );
        if !skip_state_readback && let Some(state_stage) = &state_stage {
            device.cmd_copy_buffer(
                cmd,
                state_buf.handle(),
                state_stage.handle(),
                &[vk::BufferCopy::default()
                    .size(state_data.len() as u64)
                    ],
            );
        }

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn recurrent single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn recurrent single-submit dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "record_submit_wait",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "read_output",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );
    let stage_profile = profile_kernel_stages.then(Instant::now);
    let state_data = if skip_state_readback {
        None
    } else if host_visible_state {
        Some(VulkanBuffer::read_host_visible(device, &state_buf)?)
    } else {
        Some(VulkanBuffer::read_host_visible(
            device,
            state_stage
                .as_ref()
                .expect("state staging exists when state is device-local"),
        )?)
    };
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "read_state",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "create_tensors",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );
    Ok((out_data, state_data))
}

/// Dispatch a single-token recurrent step with unexpanded GQA Q/K heads.
///
/// `q` and `k` are `[batch, 1, q_heads, dk]`; `v`, `beta`, and `g` use value
/// heads (`[batch, 1, heads, ...]`). The shader maps each value head to its
/// source Q/K head with `h / (heads / q_heads)`, matching the regular GQA
/// expansion used by the portable path without materializing the repeated Q/K
/// tensors on the host.
///
/// Bytes-only entry point. Caller supplies pre-extracted byte slices and the
/// shape parameters directly. The returned output bytes have logical shape
/// `[batch, heads, dv]` (with the caller responsible for any unsqueeze /
/// reshape on the candle side). Output bytes use the caller-chosen dtype
/// matching the kernel's recurrent dtype; the optional state bytes share
/// the caller-known state shape and dtype.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_native_head_last_with_options_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: &[u8],
    batch: usize,
    q_heads: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    skip_state_readback: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    anyhow::ensure!(
        q_heads > 0,
        "native-head recurrent q_heads must be positive"
    );
    anyhow::ensure!(
        heads % q_heads == 0,
        "native-head recurrent heads {heads} must be divisible by q_heads {q_heads}"
    );

    let parallel_reduce = use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    dispatch_gdn_recurrent_step_single_submit_bytes(
        vk_device,
        q_data,
        k_data,
        v_data,
        beta_data,
        g_data,
        state_data,
        &spirv,
        batch,
        heads,
        q_heads,
        dk,
        dv,
        parallel_reduce,
        skip_state_readback,
        false,
        None,
    )
}

/// Dispatch a single-token recurrent step with unexpanded raw GQA Q/K heads,
/// folding the split path's Q/K L2 normalization into the recurrent shader.
///
/// Bytes-only variant of the former candle-typed entry point. Callers must
/// supply the input byte slices and shape parameters directly. Output bytes
/// are laid out as `[batch, heads, dv]` rows in the kernel's native dtype
/// (`state_dtype`, which is also the dtype of the optional returned state).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_qk_norm_step_native_head_last_with_options_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: &[u8],
    batch: usize,
    q_heads: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    skip_state_readback: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    anyhow::ensure!(
        q_heads > 0,
        "native-head qk-norm recurrent q_heads must be positive"
    );
    anyhow::ensure!(
        heads % q_heads == 0,
        "native-head qk-norm recurrent heads {heads} must be divisible by q_heads {q_heads}"
    );
    anyhow::ensure!(
        dk <= 256 && dv <= 256,
        "native-head qk-norm recurrent supports dk/dv <= 256, got dk={dk} dv={dv}"
    );

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_recurrent_qk_norm_step.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    dispatch_gdn_recurrent_step_single_submit_bytes(
        vk_device,
        q_data,
        k_data,
        v_data,
        beta_data,
        g_data,
        state_data,
        &spirv,
        batch,
        heads,
        q_heads,
        dk,
        dv,
        false,
        skip_state_readback,
        false,
        Some((batch as u32, heads as u32, 1)),
    )
}

// ---------------------------------------------------------------------------
// GDN chunk prep kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN chunk prep kernel.
///
/// Computes: a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last
/// Input: g[B,H,C], v[B,H,C,dv], kkt[B,H,C,C], qkt[B,H,C,C],
///         ks_entry[B,H,C,dv], q_s[B,H,C,dv]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_chunk_prep_bytes(
    vk_device: &VulkanDevice,
    g_data: &[u8],
    v_data: &[u8],
    kkt_data: &[u8],
    qkt_data: &[u8],
    ks_entry_data: &[u8],
    q_s_data: &[u8],
    batch: usize,
    heads: usize,
    chunk: usize,
    dv: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_chunk_prep.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, g_data.len() as u64)?;
    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    let kkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, kkt_data.len() as u64)?;
    let qkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, qkt_data.len() as u64)?;
    let ks_entry_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, ks_entry_data.len() as u64)?;
    let q_s_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, q_s_data.len() as u64)?;
    if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        upload_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[
                (&g_buf, &g_data),
                (&v_buf, &v_data),
                (&kkt_buf, &kkt_data),
                (&qkt_buf, &qkt_data),
                (&ks_entry_buf, &ks_entry_data),
                (&q_s_buf, &q_s_data),
            ],
        )
        .context("failed to upload gdn_chunk_prep inputs")?;
    } else {
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &kkt_buf, &kkt_data)?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &qkt_buf, &qkt_data)?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &ks_entry_buf,
            &ks_entry_data,
        )?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_buf, &q_s_data)?;
    }

    // Create output buffers (f32 shader outputs, converted to bf16 below).
    let cc_size = (batch * heads * chunk * chunk * 4) as u64;
    let cv_size = (batch * heads * chunk * dv * 4) as u64;
    let decay_size = (batch * heads * chunk * 4) as u64;
    let p_last_size = (batch * heads * 4) as u64;
    let a_strict_buf = VulkanBuffer::create_device_local(device, device_local_mt, cc_size)?;
    let b_mask_buf = VulkanBuffer::create_device_local(device, device_local_mt, cc_size)?;
    let v_prime_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let q_s_scaled_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let decay_last_col_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, decay_size)?;
    let p_last_buf = VulkanBuffer::create_device_local(device, device_local_mt, p_last_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * (chunk * chunk + chunk * dv + chunk + 1);
    let workgroup_count = ((total + 255) / 256) as u32;

    // Bindings: g=0, v=1, kkt=2, qkt=3, ks_entry=4, q_s=5,
    //           a_strict=6, b_mask=7, v_prime=8, q_s_scaled=9, decay_last_col=10, p_last=11
    let all_handles = vec![
        g_buf.handle(),
        v_buf.handle(),
        kkt_buf.handle(),
        qkt_buf.handle(),
        ks_entry_buf.handle(),
        q_s_buf.handle(),
        a_strict_buf.handle(),
        b_mask_buf.handle(),
        v_prime_buf.handle(),
        q_s_scaled_buf.handle(),
        decay_last_col_buf.handle(),
        p_last_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back all outputs
    let (
        a_strict_data,
        b_mask_data,
        v_prime_data,
        q_s_scaled_data,
        decay_last_col_data,
        p_last_data,
    ) = if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        let mut data = read_back_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[
                &a_strict_buf,
                &b_mask_buf,
                &v_prime_buf,
                &q_s_scaled_buf,
                &decay_last_col_buf,
                &p_last_buf,
            ],
        )
        .context("failed to read back gdn_chunk_prep outputs")?;
        anyhow::ensure!(
            data.len() == 6,
            "gdn_chunk_prep batched readback returned wrong count"
        );
        (
            data.remove(0),
            data.remove(0),
            data.remove(0),
            data.remove(0),
            data.remove(0),
            data.remove(0),
        )
    } else {
        (
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &a_strict_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &b_mask_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &v_prime_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &q_s_scaled_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &decay_last_col_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &p_last_buf)?,
        )
    };

    // Cleanup
    drop(g_buf);
    drop(v_buf);
    drop(kkt_buf);
    drop(qkt_buf);
    drop(ks_entry_buf);
    drop(q_s_buf);
    drop(a_strict_buf);
    drop(b_mask_buf);
    drop(v_prime_buf);
    drop(q_s_scaled_buf);
    drop(decay_last_col_buf);
    drop(p_last_buf);

    let _ = (batch, heads, chunk, dv);
    Ok((
        a_strict_data,
        b_mask_data,
        v_prime_data,
        q_s_scaled_data,
        decay_last_col_data,
        p_last_data,
    ))
}

// ---------------------------------------------------------------------------
// GDN full chunk forward kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN full chunk forward kernel (fused prep + scan).
///
/// Input: g[B,H,C], v[B,H,C,dv], kkt[B,H,C,C], qkt[B,H,C,C],
///         ks_entry[B,H,C,dv], q_s[B,H,C,dv], beta[B,H,C], k_t[B,H,dk,C]
/// State: [B,H,dk,dv] (in/out)
/// Output: [B,H,C,dv]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_full_chunk_forward_bytes(
    vk_device: &VulkanDevice,
    g_data: &[u8],
    v_data: &[u8],
    kkt_data: &[u8],
    qkt_data: &[u8],
    ks_entry_data: &[u8],
    q_s_data: &[u8],
    beta_data: &[u8],
    k_t_data: &[u8],
    state_data: &[u8],
    batch: usize,
    heads: usize,
    chunk: usize,
    dk: usize,
    dv: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_full_chunk_forward.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    anyhow::ensure!(
        chunk == 64 && dv <= 128,
        "gdn_full_chunk_forward supports chunk=64 and dv<=128, got chunk={chunk} dv={dv}"
    );

    // Create input buffers + upload
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, g_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;

    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;

    let kkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, kkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &kkt_buf, &kkt_data)?;

    let qkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, qkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &qkt_buf, &qkt_data)?;

    let ks_entry_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, ks_entry_data.len() as u64)?;
    VulkanBuffer::upload_data(
        device,
        host_visible_mt,
        queue,
        qfi,
        &ks_entry_buf,
        &ks_entry_data,
    )?;

    let q_s_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, q_s_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_buf, &q_s_data)?;

    let beta_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;

    let k_t_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, k_t_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &k_t_buf, &k_t_data)?;

    // State is mutable — upload, dispatch, read back
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &state_buf, &state_data)?;

    // Create output buffer (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dk, dv
    let push_constants: [u32; 5] = [
        batch as u32,
        heads as u32,
        chunk as u32,
        dk as u32,
        dv as u32,
    ];

    // One workgroup owns one (batch, head) chunk. Threads within the workgroup
    // cooperate over the fixed 64-token chunk and dv lanes.
    let workgroup_count = (batch * heads) as u32;

    // Bindings: g=0, v=1, kkt=2, qkt=3, ks_entry=4, q_s=5, beta=6, k_t=7, state=8, out=9
    let all_handles = vec![
        g_buf.handle(),
        v_buf.handle(),
        kkt_buf.handle(),
        qkt_buf.handle(),
        ks_entry_buf.handle(),
        q_s_buf.handle(),
        beta_buf.handle(),
        k_t_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output and updated state
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let state_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &state_buf)?;

    // Cleanup
    drop(g_buf);
    drop(v_buf);
    drop(kkt_buf);
    drop(qkt_buf);
    drop(ks_entry_buf);
    drop(q_s_buf);
    drop(beta_buf);
    drop(k_t_buf);
    drop(state_buf);
    drop(out_buf);

    let _ = (batch, heads, chunk, dk, dv);
    Ok((out_data, state_data))
}

// ---------------------------------------------------------------------------
// GDN chunk scan kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN chunk scan kernel.
///
/// Performs the scan operation for chunkwise recurrence:
///   1. forward-substitution for W[t]
///   2. intra = B_mask @ W
///   3. out = q_s_scaled + intra
///
/// Input: a_strict[B,H,C,C], b_mask[B,H,C,C], v_prime[B,H,C,dv],
///         q_s_scaled[B,H,C,dv], beta[B,H,C], decay_last_col[B,H,C]
/// Output: out[B,H,C,dv], p_out[B,H,C,dv]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_chunk_scan_bytes(
    vk_device: &VulkanDevice,
    a_strict_data: &[u8],
    b_mask_data: &[u8],
    v_prime_data: &[u8],
    q_s_scaled_data: &[u8],
    beta_data: &[u8],
    decay_last_col_data: &[u8],
    batch: usize,
    heads: usize,
    chunk: usize,
    dv: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_chunk_scan.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let a_strict_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, a_strict_data.len() as u64)?;
    let b_mask_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, b_mask_data.len() as u64)?;
    let v_prime_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, v_prime_data.len() as u64)?;
    let q_s_scaled_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, q_s_scaled_data.len() as u64)?;
    let beta_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    let decay_last_col_buf = VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        decay_last_col_data.len() as u64,
    )?;
    if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        upload_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[
                (&a_strict_buf, &a_strict_data),
                (&b_mask_buf, &b_mask_data),
                (&v_prime_buf, &v_prime_data),
                (&q_s_scaled_buf, &q_s_scaled_data),
                (&beta_buf, &beta_data),
                (&decay_last_col_buf, &decay_last_col_data),
            ],
        )
        .context("failed to upload gdn_chunk_scan inputs")?;
    } else {
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &a_strict_buf,
            &a_strict_data,
        )?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &b_mask_buf,
            &b_mask_data,
        )?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &v_prime_buf,
            &v_prime_data,
        )?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &q_s_scaled_buf,
            &q_s_scaled_data,
        )?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &decay_last_col_buf,
            &decay_last_col_data,
        )?;
    }

    // Create output buffers (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let p_out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * chunk * dv;
    let workgroup_count = ((total + 255) / 256) as u32;

    // Bindings: a_strict=0, b_mask=1, v_prime=2, q_s_scaled=3, beta=4, decay_last_col=5, out=6, p_out=7
    let all_handles = vec![
        a_strict_buf.handle(),
        b_mask_buf.handle(),
        v_prime_buf.handle(),
        q_s_scaled_buf.handle(),
        beta_buf.handle(),
        decay_last_col_buf.handle(),
        out_buf.handle(),
        p_out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back outputs
    let (out_data, p_out_data) = if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        let mut data = read_back_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[&out_buf, &p_out_buf],
        )
        .context("failed to read back gdn_chunk_scan outputs")?;
        anyhow::ensure!(
            data.len() == 2,
            "gdn_chunk_scan batched readback returned wrong count"
        );
        (data.remove(0), data.remove(0))
    } else {
        (
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &p_out_buf)?,
        )
    };

    // Cleanup
    drop(a_strict_buf);
    drop(b_mask_buf);
    drop(v_prime_buf);
    drop(q_s_scaled_buf);
    drop(beta_buf);
    drop(decay_last_col_buf);
    drop(out_buf);
    drop(p_out_buf);

    let _ = (batch, heads, chunk, dv);
    Ok((out_data, p_out_data))
}

/// Scaled dot-product attention forward (prefill), online softmax.
///
/// Inputs `q`, `k`, `v` are F32 row-major `[batch, seq_len, num_heads,
/// head_dim]`. Returns the SDPA output with the same shape and dtype.
///
/// Replaces the buggy `flash_attn.comp` placeholder. The shader runs
/// one workgroup per `(batch, head, q_row)` with 128 threads doing a
/// parallel head_dim reduction per K row, plus the standard online
/// softmax recurrence. No scratch / LSE buffers are written; this is
/// the forward-only path used by training prefill.
///
/// Constraints: `head_dim` must be ≤ 128 (the workgroup size). For
/// Qwen3.5-4B head_dim=128 this is exact; smaller head_dim wastes
/// some threads but produces correct output.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sdpa_prefill_f32_bytes(
    vk_device: &VulkanDevice,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Vec<u8>> {
    let expected_bytes = batch * seq_len * num_heads * head_dim * 4;
    anyhow::ensure!(
        q_data.len() == expected_bytes
            && k_data.len() == expected_bytes
            && v_data.len() == expected_bytes,
        "sdpa_prefill_f32: q/k/v byte lengths ({}, {}, {}) do not match expected {}",
        q_data.len(),
        k_data.len(),
        v_data.len(),
        expected_bytes,
    );
    // The shader spreads head_dim over 128 threads with ELEMS_PER_THREAD=2
    // grid-strided elements each, so it covers head_dim up to 256 (Qwen3.5-4B
    // uses head_dim=256). Bump ELEMS_PER_THREAD in the shader in lockstep to
    // raise this. (#1082 Vulkan SDPA head_dim=256 support.)
    anyhow::ensure!(
        head_dim <= 256,
        "sdpa_prefill_f32: head_dim {head_dim} > 256 (shader covers 128 threads × 2 elems)"
    );
    // Vulkan spec only guarantees `maxComputeWorkGroupCount[i] >= 65535`
    // per axis. The dispatch grid is (seq_len, num_heads, batch); if any
    // axis would exceed that, surface a clear error rather than letting
    // vkCmdDispatch silently drop work or fail with an opaque
    // VK_ERROR_OUT_OF_DEVICE_MEMORY. Use the actual device limit
    // (typically much higher than the spec minimum on AMD/Strix Halo).
    let limit_x = vk_device.max_compute_work_group_count(0) as usize;
    let limit_y = vk_device.max_compute_work_group_count(1) as usize;
    let limit_z = vk_device.max_compute_work_group_count(2) as usize;
    anyhow::ensure!(
        seq_len <= limit_x && num_heads <= limit_y && batch <= limit_z,
        "sdpa_prefill_f32: dispatch grid (seq_len={seq_len}, num_heads={num_heads}, \
         batch={batch}) exceeds device per-axis limits ({limit_x}, {limit_y}, {limit_z})"
    );

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/sdpa_prefill_f32.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("sdpa_prefill_f32: shader compile/load")?;
    let push_constants: [u32; 6] = [
        batch as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
        causal as u32,
    ];
    // Workgroup grid: (q_row, head, batch). Matches gl_WorkGroupID
    // assignments in the shader.
    let workgroup_count = (seq_len as u32, num_heads as u32, batch as u32);
    let output_shape = vec![batch, seq_len, num_heads, head_dim];
    dispatch_kernel_bytes(
        vk_device,
        &spirv,
        &push_constants,
        workgroup_count,
        &[q_data, k_data, v_data],
        &output_shape,
        4,
    )
    .context("sdpa_prefill_f32: dispatch_kernel_bytes")
}

/// Vulkan SGD parameter update step: `param -= lr * grad`, in-place
/// against an existing `VulkanBuffer` (the parameter buffer) using
/// the gradient as a read-only second buffer.
///
/// Phase 4.2 of the residency plan. Used by the trainer once
/// `TrainableLoraParams` have been migrated to registry-resident
/// `VulkanBuffer`s in Phase 4.1; until then, the existing CPU SGD
/// step in `kiln-train::trainer::sgd_step` continues to run.
///
/// Both buffers are flat F32 of length `n_elements`. The dispatch
/// allocates one workgroup per 256 elements; per-step compute is
/// trivially small (3n F32 reads/writes) so no chunking is required
/// even for the largest LoRA Vars (rank=64, hidden=2560 = 164K F32 =
/// 640 KB).
/// BF16 variant of `dispatch_sgd_step_f32`. Both buffers hold
/// packed BF16 (2 bf16 elements per u32) — same layout as the
/// `extract_tensor_packed_bf16_bytes_pub` encoding the residency
/// registry uses for BF16 tensors. One thread per u32 word; each
/// thread updates both lanes via bf16↔f32 bit-expansion.
///
/// Used by the trainer to run SGD on registry-resident LoRA Vars
/// (which are BF16 by convention) without the candle CPU
/// var.set + update_resident_activation re-upload.
pub fn dispatch_sgd_step_bf16(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "sgd_step_bf16: n_elements must be > 0");
    let num_words = n_elements.div_ceil(2);
    let workgroup_count = num_words.div_ceil(256) as u32;
    let limit = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit,
        "sgd_step_bf16: n_elements={n_elements} → {workgroup_count} workgroups \
         (>{limit} device per-axis limit)"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/sgd_step_bf16.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("sgd_step_bf16: shader compile/load")?;
    let push_constants: [u32; 2] = [n_elements as u32, lr.to_bits()];
    let all_handles = vec![param_buffer.handle(), grad_buffer.handle()];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("sgd_step_bf16: kernel dispatch")
}

pub fn dispatch_sgd_step_f32(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "sgd_step_f32: n_elements must be > 0");
    // Vulkan only guarantees `maxComputeWorkGroupCount[i] >= 65535`.
    // The dispatch is `n_elements.div_ceil(256)` workgroups on axis x;
    // Use the actual device limit rather than the spec minimum.
    let limit = vk_device.max_compute_work_group_count(0) as usize;
    anyhow::ensure!(
        n_elements.div_ceil(256) <= limit,
        "sgd_step_f32: n_elements={n_elements} would dispatch \
         {} workgroups (>{limit} device per-axis limit)",
        n_elements.div_ceil(256)
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/sgd_step_f32.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("sgd_step_f32: shader compile/load")?;
    let push_constants: [u32; 2] = [n_elements as u32, lr.to_bits()];
    let all_handles = vec![param_buffer.handle(), grad_buffer.handle()];
    let workgroup_count = n_elements.div_ceil(256) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("sgd_step_f32: kernel dispatch")
}

/// AdamW (decoupled weight decay) for registry-resident BF16 buffers.
///
/// Updates `param`, `m`, and `v` in place. All four buffers (param,
/// grad, m, v) hold packed BF16 (2 bf16 per u32 word) in the
/// `extract_tensor_packed_bf16_bytes_pub` encoding, and must share
/// the same element count `n_elements`. The step counter is 1-indexed
/// (so the first call after `m=v=0` passes `step=1`); host-side this
/// helper computes `bias_correction{1,2} = 1 - beta^step` and ships
/// them via push constants so the shader doesn't need a pow call.
///
/// One thread per u32 word (i.e. two BF16 lanes), 256 threads per
/// workgroup. Per-step cost is ~8n BF16 reads/writes — bandwidth-bound,
/// trivially small even for the largest LoRA Vars.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_adamw_step_bf16(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    first_moment_buffer: &VulkanBuffer,
    second_moment_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "adamw_step_bf16: n_elements must be > 0");
    anyhow::ensure!(step >= 1, "adamw_step_bf16: step must be 1-indexed (>=1)");
    let num_words = n_elements.div_ceil(2);
    let workgroup_count = num_words.div_ceil(256) as u32;
    let limit = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit,
        "adamw_step_bf16: n_elements={n_elements} → {workgroup_count} workgroups \
         (>{limit} device per-axis limit)"
    );
    let bc1 = 1.0_f32 - beta1.powi(step as i32);
    let bc2 = 1.0_f32 - beta2.powi(step as i32);
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/adamw_step_bf16.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("adamw_step_bf16: shader compile/load")?;
    let push_constants: [u32; 9] = [
        n_elements as u32,
        step,
        lr.to_bits(),
        beta1.to_bits(),
        beta2.to_bits(),
        eps.to_bits(),
        weight_decay.to_bits(),
        bc1.to_bits(),
        bc2.to_bits(),
    ];
    let all_handles = vec![
        param_buffer.handle(),
        grad_buffer.handle(),
        first_moment_buffer.handle(),
        second_moment_buffer.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("adamw_step_bf16: kernel dispatch")
}

/// F32 variant of `dispatch_adamw_step_bf16`. Kept for parity with
/// `dispatch_sgd_step_f32`; currently LoRA Vars default to BF16 so
/// this path is exercised mainly by tests.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_adamw_step_f32(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    first_moment_buffer: &VulkanBuffer,
    second_moment_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "adamw_step_f32: n_elements must be > 0");
    anyhow::ensure!(step >= 1, "adamw_step_f32: step must be 1-indexed (>=1)");
    let workgroup_count = n_elements.div_ceil(256) as u32;
    let limit = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit,
        "adamw_step_f32: n_elements={n_elements} → {workgroup_count} workgroups \
         (>{limit} device per-axis limit)"
    );
    let bc1 = 1.0_f32 - beta1.powi(step as i32);
    let bc2 = 1.0_f32 - beta2.powi(step as i32);
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/adamw_step_f32.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("adamw_step_f32: shader compile/load")?;
    let push_constants: [u32; 9] = [
        n_elements as u32,
        step,
        lr.to_bits(),
        beta1.to_bits(),
        beta2.to_bits(),
        eps.to_bits(),
        weight_decay.to_bits(),
        bc1.to_bits(),
        bc2.to_bits(),
    ];
    let all_handles = vec![
        param_buffer.handle(),
        grad_buffer.handle(),
        first_moment_buffer.handle(),
        second_moment_buffer.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("adamw_step_f32: kernel dispatch")
}
