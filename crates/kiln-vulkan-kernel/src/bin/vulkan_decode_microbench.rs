//! Decode-path Vulkan microbench.
//!
//! Measures wall-clock per-iteration latency for the three single-token
//! decode-hot kernels at Qwen3.5-4B shapes: full_attn QKV, GDN in_proj,
//! and MLP gate_up + down. Exercises the same `dispatch_*_cached_*`
//! entry points the production decode loop uses, including host upload
//! of `x` and host readback of the output, so the numbers reflect
//! end-to-end per-call cost.
//!
//! Usage: `cargo run --release -p kiln-vulkan-kernel --bin vulkan_decode_microbench`.

use std::time::Instant;

use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::buffer::VulkanBuffer;
use kiln_vulkan_kernel::device::VulkanDevice;
use kiln_vulkan_kernel::kernels::{
    upload_bf16_packed_buffer_from_slice, upload_f32_buffer_from_slice,
};
use kiln_vulkan_kernel::shaders;

// Used by run_full_step_resident — keep the module-level imports here so the
// helper itself stays terse.

const HIDDEN: usize = 2560;
const Q_DIM: usize = 4096;
const Q_GATE_DIM: usize = 2 * Q_DIM;
const K_DIM: usize = 1024;
const V_DIM: usize = 1024;
const INTERMEDIATE: usize = 9216;
const FULL_ATTN_TOTAL_OUT: usize = Q_GATE_DIM + K_DIM + V_DIM;

// Qwen3.5-4B GDN shapes.
const GDN_NUM_KEY_HEADS: usize = 16;
const GDN_NUM_VALUE_HEADS: usize = 32;
const GDN_HEAD_DIM: usize = 128;
const GDN_QK_DIM: usize = GDN_NUM_KEY_HEADS * GDN_HEAD_DIM;
const GDN_V_DIM: usize = GDN_NUM_VALUE_HEADS * GDN_HEAD_DIM;
const QKV_DIM: usize = 2 * GDN_QK_DIM + GDN_V_DIM;
const Z_DIM: usize = GDN_V_DIM;
const A_DIM: usize = 32;
const B_DIM: usize = 32;

const WARMUP_ITERS: usize = 10;
const TIMED_ITERS: usize = 30;
const REPEATS: usize = 5;
const DEFAULT_BATCHES: &[usize] = &[1, 4, 8, 16, 32, 64];
const MLP_BF16_ROWS8_MIN_BATCH: usize = 256;
const PAGED_ATTN_SPLITK_CHUNKS_B1: usize = 32;
const PAGED_ATTN_SPLITK_CHUNKS_BATCHED: usize = 4;

/// Deterministic flat `Vec<bf16>` weight data for byte/slice dispatch entries.
fn make_bf16_weight_slice(rows: usize, cols: usize) -> Vec<bf16> {
    let n = rows * cols;
    (0..n)
        .map(|i| bf16::from_f32(((i % 31) as f32 - 15.0) * 0.01))
        .collect()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

fn batch_sweep() -> Vec<usize> {
    let Some(parsed) = std::env::var("KILN_VK_MICROBENCH_BATCHES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
    else {
        return DEFAULT_BATCHES.to_vec();
    };
    parsed
}

fn enabled_unless_disabled(name: &str) -> bool {
    std::env::var(name).is_err()
}

fn linear_bf16w_rows4_enabled() -> bool {
    enabled_unless_disabled("KILN_DISABLE_VULKAN_LINEAR_DECODE_BF16W_ROWS4")
        && enabled_unless_disabled("KILN_DISABLE_VULKAN_LINEAR_BF16W_ROWS4")
}

fn paged_attn_splitk_chunks(batch: usize) -> usize {
    std::env::var("KILN_VK_PAGED_ATTN_SPLITK_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(if batch <= 1 {
            PAGED_ATTN_SPLITK_CHUNKS_B1
        } else {
            PAGED_ATTN_SPLITK_CHUNKS_BATCHED
        })
}

fn full_attn_qkv_gate_split_bf16w_plan(batch: usize, total_out: usize) -> (&'static str, u32) {
    let rows4 =
        batch >= 1 && enabled_unless_disabled("KILN_DISABLE_VULKAN_FULL_ATTN_QKV_BF16W_ROWS4");
    let row_groups = if rows4 { batch.div_ceil(4) } else { batch };
    let shader = if rows4 {
        shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W
    } else {
        shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_BF16W
    };
    (shader, (row_groups * total_out.div_ceil(16)) as u32)
}

fn linear_bf16w_batched_plan(batch: usize, out_dim: usize) -> (&'static str, u32) {
    let rows4 = batch >= 32 && linear_bf16w_rows4_enabled();
    let row_groups = if rows4 { batch.div_ceil(4) } else { batch };
    let shader = if rows4 {
        shaders::LINEAR_DECODE_BATCHED_ROWS4_BF16W
    } else {
        shaders::LINEAR_DECODE_BATCHED_BF16W
    };
    (shader, (row_groups * out_dim.div_ceil(32)) as u32)
}

fn mlp_gate_up_bf16w_batched_plan(batch: usize, intermediate: usize) -> (&'static str, u32) {
    let rows8 = batch >= MLP_BF16_ROWS8_MIN_BATCH
        && enabled_unless_disabled("KILN_DISABLE_VULKAN_MLP_BF16_ROWS8");
    let rows4 = batch >= 8
        && !rows8
        && enabled_unless_disabled("KILN_DISABLE_VULKAN_MLP_BF16_GATE_UP_ROWS4");
    if rows8 {
        (
            shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS8_BF16W,
            (batch.div_ceil(8) * intermediate.div_ceil(64)) as u32,
        )
    } else if rows4 {
        (
            shaders::MLP_GATE_UP_DECODE_BATCHED_ROWS4_BF16W,
            (batch.div_ceil(4) * intermediate.div_ceil(64)) as u32,
        )
    } else {
        (
            shaders::MLP_GATE_UP_DECODE_BATCHED_BF16W,
            (batch * intermediate.div_ceil(128)) as u32,
        )
    }
}

fn mlp_down_add_residual_bf16w_batched_plan(batch: usize, out_dim: usize) -> (&'static str, u32) {
    let rows8 = batch >= MLP_BF16_ROWS8_MIN_BATCH
        && enabled_unless_disabled("KILN_DISABLE_VULKAN_MLP_BF16_ROWS8");
    let rows4 =
        batch >= 16 && !rows8 && enabled_unless_disabled("KILN_DISABLE_VULKAN_MLP_BF16_DOWN_ROWS4");
    if rows8 {
        (
            shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL_ROWS8,
            (batch.div_ceil(8) * out_dim.div_ceil(32)) as u32,
        )
    } else if rows4 {
        (
            shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL_ROWS4,
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32,
        )
    } else {
        (
            shaders::LINEAR_DECODE_BATCHED_BF16W_ADD_RESIDUAL,
            (batch * out_dim.div_ceil(32)) as u32,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn gdn_in_proj_bf16w_batched_plan(
    batch: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    total_out: usize,
) -> (&'static str, u32) {
    let pair_qkv_z =
        batch > 1 && enabled_unless_disabled("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_PAIR_QKV_Z");
    let row_grouping = pair_qkv_z
        && batch >= 3
        && enabled_unless_disabled("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_PAIR");
    let row_group_size = if row_grouping
        && batch >= 8
        && enabled_unless_disabled("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_QUAD")
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
    let shader = if row_group_size == 4 {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W
    } else if row_group_size == 2 {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W
    } else if pair_qkv_z {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_BF16W
    } else {
        shaders::GDN_IN_PROJ_DECODE_BATCHED_BF16W
    };
    let row_groups = batch.div_ceil(row_group_size);
    (shader, (row_groups * dispatch_cols.div_ceil(80)) as u32)
}

fn time<F: FnMut() -> Result<()>>(label: &str, batch: usize, mut f: F) -> Result<()> {
    let warmup_iters = env_usize("KILN_VK_MICROBENCH_WARMUP", WARMUP_ITERS);
    let timed_iters = env_usize("KILN_VK_MICROBENCH_TIMED", TIMED_ITERS);
    let repeats = env_usize("KILN_VK_MICROBENCH_REPEATS", REPEATS);
    for _ in 0..warmup_iters {
        f()?;
    }
    // Take the minimum per-iter time across REPEATS independent timed blocks.
    // The fastest block is the cleanest signal of steady-state kernel cost;
    // mean is dragged around by background load and GPU thermal swings.
    let mut best_ns = u128::MAX;
    for _ in 0..repeats {
        let start = Instant::now();
        for _ in 0..timed_iters {
            f()?;
        }
        let elapsed = start.elapsed().as_nanos();
        if elapsed < best_ns {
            best_ns = elapsed;
        }
    }
    let per_iter_us = (best_ns as f64 / timed_iters as f64) / 1_000.0;
    let rows_per_sec = (batch as f64 * timed_iters as f64) / (best_ns as f64 / 1e9);
    println!(
        "{label:<32} batch={batch:>3}  per_iter={per_iter_us:>8.1} us  rows/s={rows_per_sec:>10.0}"
    );
    Ok(())
}

fn run() -> Result<()> {
    let device = VulkanDevice::new()?;
    println!(
        "device: {} ({})",
        device.device_name(),
        device.vendor_string()
    );
    println!();

    // Allow caller to run a single kernel ("mlp_bf16w", "mlp_bf16_gu_f32_d",
    // "full_attn_qkv", "gdn_in_proj") so they can iterate fast without
    // perturbation from sibling tests heating the GPU.
    let only = std::env::args().nth(1);
    let want = |name: &str| only.as_deref().is_none_or(|s| s == name);

    // Pre-upload weights once.
    let q_w = make_bf16_weight_slice(HIDDEN, Q_GATE_DIM);
    let k_w = make_bf16_weight_slice(HIDDEN, K_DIM);
    let v_w = make_bf16_weight_slice(HIDDEN, V_DIM);
    let gate_w = make_bf16_weight_slice(HIDDEN, INTERMEDIATE);
    let up_w = make_bf16_weight_slice(HIDDEN, INTERMEDIATE);
    let down_w = make_bf16_weight_slice(INTERMEDIATE, HIDDEN);
    let down_w_f32: Vec<f32> = down_w.iter().map(|v| v.to_f32()).collect();
    let qkv_w = make_bf16_weight_slice(HIDDEN, QKV_DIM);
    let z_w = make_bf16_weight_slice(HIDDEN, Z_DIM);
    let a_w = make_bf16_weight_slice(HIDDEN, A_DIM);
    let b_w = make_bf16_weight_slice(HIDDEN, B_DIM);

    let q_buf = upload_bf16_packed_buffer_from_slice(&device, &q_w)?;
    let k_buf = upload_bf16_packed_buffer_from_slice(&device, &k_w)?;
    let v_buf = upload_bf16_packed_buffer_from_slice(&device, &v_w)?;
    let gate_buf = upload_bf16_packed_buffer_from_slice(&device, &gate_w)?;
    let up_buf = upload_bf16_packed_buffer_from_slice(&device, &up_w)?;
    let down_buf = upload_bf16_packed_buffer_from_slice(&device, &down_w)?;
    // f32 down buffer for bf16_gate_up_f32_down variant.
    let down_f32_buf = upload_f32_buffer_from_slice(&device, &down_w_f32)?;
    let qkv_buf = upload_bf16_packed_buffer_from_slice(&device, &qkv_w)?;
    let z_buf = upload_bf16_packed_buffer_from_slice(&device, &z_w)?;
    let a_buf = upload_bf16_packed_buffer_from_slice(&device, &a_w)?;
    let b_buf = upload_bf16_packed_buffer_from_slice(&device, &b_w)?;

    let batches = batch_sweep();

    if want("full_attn_qkv") {
        println!("== full_attn QKV+gate (fused, bf16w) ==");
        for &batch in batches.as_slice() {
            // x = zeros [batch, 1, HIDDEN] f32. Bytes-typed dispatch
            // takes &[u8] directly, with no tensor-object staging.
            let x_bytes = vec![0u8; batch * HIDDEN * 4];
            time("full_attn_qkv_decode", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bf16_weights_bytes(
                    &device, &x_bytes, &q_buf, &k_buf, &v_buf, batch, HIDDEN, Q_GATE_DIM, K_DIM,
                    V_DIM,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("mlp_bf16_gu_f32_d") {
        println!("== MLP gate_up + down (bf16 g/u, f32 down) ==");
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * HIDDEN * 4];
            time("mlp_decode_bf16_gu_f32_d", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_gate_up_f32_down_bytes(
                    &device,
                    &x_bytes,
                    batch,
                    &gate_buf,
                    &up_buf,
                    &down_f32_buf,
                    HIDDEN,
                    INTERMEDIATE,
                    HIDDEN,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("mlp_bf16w") {
        println!("== MLP gate_up + down (full bf16) ==");
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * HIDDEN * 4];
            time("mlp_decode_bf16w", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_weights_bytes(
                    &device,
                    &x_bytes,
                    batch,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    HIDDEN,
                    INTERMEDIATE,
                    HIDDEN,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("linear_decode") {
        // Q-out / GDN-out shape: take Q dim → hidden. Exercises the
        // standalone bf16w linear decode used for attention out_proj.
        println!("== linear_decode_cached_bf16w (Q out, q_dim→hidden) ==");
        // Build the bf16w buffer from a bf16 slice.
        let q_out_weight = make_bf16_weight_slice(Q_DIM, HIDDEN);
        let q_out_buf = upload_bf16_packed_buffer_from_slice(&device, &q_out_weight)?;
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * Q_DIM * 4];
            time("linear_decode_bf16w_qout", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bytes(
                    &device, &x_bytes, &q_out_buf, batch, Q_DIM, HIDDEN,
                    /*packed_bf16_weights=*/ true,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("causal_conv1d_update") {
        println!("== causal_conv1d_update (B,C=2048,T=1) ==");
        let channels = 2048usize;
        let kernel_size = 4usize;
        // weight shape [channels, kernel_size], state shape
        // [batch, channels, kernel_size - 1], x shape [batch, channels, 1].
        // All f32. The bytes-typed dispatch takes them as &[u8]. (#1082)
        let weight_bytes = vec![0u8; channels * kernel_size * 4];
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * channels * 1 * 4];
            let state_bytes = vec![0u8; batch * channels * (kernel_size - 1) * 4];
            time("causal_conv1d_update", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update_bytes(
                    &device,
                    &x_bytes,
                    &weight_bytes,
                    &state_bytes,
                    batch,
                    channels,
                    1,
                    kernel_size,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("gdn_gated_norm") {
        println!("== gdn_gated_rms_norm_cached (hidden=2560) ==");
        // Weight = ones(HIDDEN, f32).
        let weight_data: Vec<f32> = vec![1.0; HIDDEN];
        let weight = upload_f32_buffer_from_slice(&device, &weight_data)?;
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * HIDDEN * 4];
            let z_bytes = vec![0u8; batch * HIDDEN * 4];
            time("gdn_gated_norm_cached", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached_bytes(
                    &device,
                    &x_bytes,
                    &z_bytes,
                    &weight,
                    HIDDEN,
                    1e-6,
                    &[batch, 1, HIDDEN],
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("qwen_rmsnorm") {
        println!("== qwen_rmsnorm_forward (hidden=2560 per row) ==");
        // weight = ones(HIDDEN). Both x and weight are passed as raw
        // f32 bytes to the raw-byte dispatch.
        let weight_data: Vec<f32> = vec![1.0; HIDDEN];
        let weight_bytes: &[u8] = bytemuck::cast_slice(&weight_data);
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * HIDDEN * 4];
            time("qwen_rmsnorm_forward", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_forward_bytes(
                    &device,
                    &x_bytes,
                    weight_bytes,
                    /*rows=*/ batch,
                    HIDDEN,
                    1e-6,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("gdn_gates") {
        println!("== gdn_gates_cached (a/b + a_log/dt_bias) ==");
        // Match Qwen3.5 GDN gates: a/b shape [batch, 1, nv]. nv = linear_num_value_heads = 32.
        let nv = 32usize;
        let a_log_w = make_bf16_weight_slice(1, nv);
        let dt_bias_w = make_bf16_weight_slice(1, nv);
        let a_log = upload_bf16_packed_buffer_from_slice(&device, &a_log_w)?;
        let dt_bias = upload_bf16_packed_buffer_from_slice(&device, &dt_bias_w)?;
        for &batch in batches.as_slice() {
            let a_bytes = vec![0u8; batch * 1 * nv * 4];
            let b_bytes = vec![0u8; batch * 1 * nv * 4];
            time("gdn_gates_cached", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached_bytes(
                    &device,
                    &a_bytes,
                    &b_bytes,
                    &a_log,
                    &dt_bias,
                    nv,
                    &[batch, 1, nv],
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("gdn_in_proj") {
        println!("== GDN in_proj (qkv|z|a|b fused, bf16w) ==");
        for &batch in batches.as_slice() {
            let x_bytes = vec![0u8; batch * HIDDEN * 4];
            time("gdn_in_proj_decode", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights_bytes(
                    &device, &x_bytes, batch, &qkv_buf, &z_buf, &a_buf, &b_buf, HIDDEN, QKV_DIM,
                    Z_DIM, A_DIM, B_DIM,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("gdn_block_resident_batched") {
        run_gdn_block_resident_batched(
            &device, &qkv_buf, &z_buf, &a_buf, &b_buf, &gate_buf, &up_buf, &down_buf, &batches,
        )?;
    }
    if want("full_step_resident") {
        run_full_step_resident(
            &device, &q_buf, &k_buf, &v_buf, &gate_buf, &up_buf, &down_buf, &batches,
        )?;
    }
    if want("full_step_resident_batched") {
        run_full_step_resident_batched(
            &device, &q_buf, &k_buf, &v_buf, &gate_buf, &up_buf, &down_buf, &batches,
        )?;
    }
    if want("full_token_resident_batched") {
        run_full_token_resident_batched(
            &device, &q_buf, &k_buf, &v_buf, &gate_buf, &up_buf, &down_buf, &batches,
        )?;
    }
    if want("full_token_resident_mixed_batched") {
        run_full_token_resident_mixed_batched(
            &device, &q_buf, &k_buf, &v_buf, &qkv_buf, &z_buf, &a_buf, &b_buf, &gate_buf, &up_buf,
            &down_buf, &batches, false,
        )?;
    }
    if want("full_token_resident_mixed_paged") {
        run_full_token_resident_mixed_batched(
            &device, &q_buf, &k_buf, &v_buf, &qkv_buf, &z_buf, &a_buf, &b_buf, &gate_buf, &up_buf,
            &down_buf, &batches, true,
        )?;
    }
    if want("full_token_resident_paged") {
        run_full_token_resident_paged(
            &device, &q_buf, &k_buf, &v_buf, &gate_buf, &up_buf, &down_buf, &batches,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_full_token_resident_mixed_batched(
    device: &VulkanDevice,
    q_w: &VulkanBuffer,
    k_w: &VulkanBuffer,
    v_w: &VulkanBuffer,
    gdn_qkv_w: &VulkanBuffer,
    gdn_z_w: &VulkanBuffer,
    gdn_a_w: &VulkanBuffer,
    gdn_b_w: &VulkanBuffer,
    gate_w: &VulkanBuffer,
    up_w: &VulkanBuffer,
    down_w: &VulkanBuffer,
    batches: &[usize],
    use_paged_attention: bool,
) -> Result<()> {
    use kiln_vulkan_kernel::CommandBatch;
    use kiln_vulkan_kernel::VkPagedKvCache;
    use kiln_vulkan_kernel::Workgroups;

    const NUM_LAYERS: usize = 32;
    const FULL_ATTN_LAYERS: usize = 8;
    const GDN_LAYERS: usize = 24;
    let label = if use_paged_attention {
        "full_token_resident_mixed_paged"
    } else {
        "full_token_resident_mixed"
    };
    let attn_mode = if use_paged_attention {
        "paged KV + split-K paged-attn"
    } else {
        "contiguous K/V attention"
    };
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let rotary_dim = 64usize;
    let half_rot = rotary_dim / 2;
    let history_env = if use_paged_attention {
        "KILN_VK_PAGED_HISTORY"
    } else {
        "KILN_VK_ATTENTION_HISTORY"
    };
    let cur_seq_len = env_usize(history_env, 256);
    let max_seqlen = cur_seq_len.max(1);
    let block_size = env_usize("KILN_VK_PAGED_BLOCK_SIZE", 16);
    anyhow::ensure!(block_size > 0, "KILN_VK_PAGED_BLOCK_SIZE must be > 0");
    let blocks_per_seq = (cur_seq_len + 1).div_ceil(block_size).max(1);
    let softmax_scale = (head_dim as f32).sqrt().recip();
    if use_paged_attention {
        println!(
            "== {label} ({FULL_ATTN_LAYERS} full-attn + {GDN_LAYERS} GDN layers, {attn_mode}, history={cur_seq_len}, block={block_size}, 1 cmd-buffer + 1 submit) =="
        );
    } else {
        println!(
            "== {label} ({FULL_ATTN_LAYERS} full-attn + {GDN_LAYERS} GDN layers, {attn_mode}, history={cur_seq_len}, 1 cmd-buffer + 1 submit) =="
        );
    }

    let conv_kernel = 4usize;
    let gdn_in_proj_total = QKV_DIM + Z_DIM + A_DIM + B_DIM;
    let gdn_gqa_ratio = GDN_NUM_VALUE_HEADS / GDN_NUM_KEY_HEADS;
    debug_assert_eq!(GDN_NUM_KEY_HEADS * gdn_gqa_ratio, GDN_NUM_VALUE_HEADS);
    let eps = 1e-6f32;

    let weight_norm = upload_f32_buffer_from_slice(device, &vec![1.0f32; HIDDEN])?;
    let weight_qknorm = upload_f32_buffer_from_slice(device, &vec![1.0f32; head_dim])?;
    let attn_out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(Q_DIM, HIDDEN))?;
    let gdn_recurrent_norm_w = upload_f32_buffer_from_slice(device, &vec![1.0f32; GDN_HEAD_DIM])?;
    let gdn_conv_w = upload_f32_buffer_from_slice(device, &vec![0.0f32; QKV_DIM * conv_kernel])?;
    let gdn_a_log = upload_f32_buffer_from_slice(device, &vec![-1.0f32; GDN_NUM_VALUE_HEADS])?;
    let gdn_dt_bias = upload_f32_buffer_from_slice(device, &vec![0.0f32; GDN_NUM_VALUE_HEADS])?;
    let gdn_out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(GDN_V_DIM, HIDDEN))?;

    let cos_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect();
    let sin_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect();
    let cos_buf = upload_f32_buffer_from_slice(device, &cos_data)?;
    let sin_buf = upload_f32_buffer_from_slice(device, &sin_data)?;

    let rmsnorm_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let rope_shader = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_rope_f32.comp");
    let paged_attn_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let paged_attn_splitk_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged_splitk.comp"
    );
    let paged_attn_reduce_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged_splitk_reduce.comp"
    );
    let kv_write_slots_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_kv_write_slots.comp"
    );
    let mul_sigmoid_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_mul_sigmoid_gate_f32.comp"
    );
    for &batch in batches {
        let mk = |bytes: u64| {
            VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes,
            )
        };
        let mk_zero = |bytes: u64| -> Result<VulkanBuffer> {
            let buf = mk(bytes)?;
            let zeros = vec![0u8; bytes as usize];
            VulkanBuffer::upload_data(
                device.device(),
                device.host_visible_mem_type(),
                device.queue(),
                device.queue_family_index(),
                &buf,
                &zeros,
            )?;
            Ok(buf)
        };
        let mk_dev = |bytes: &[u8]| -> Result<VulkanBuffer> {
            let buf = mk(bytes.len() as u64)?;
            VulkanBuffer::upload_data(
                device.device(),
                device.host_visible_mem_type(),
                device.queue(),
                device.queue_family_index(),
                &buf,
                bytes,
            )?;
            Ok(buf)
        };

        let hidden_bytes = (batch * HIDDEN * 4) as u64;
        let x_buf = mk_zero(hidden_bytes)?;
        let mlp_scratch = mk((batch * INTERMEDIATE * 4) as u64)?;
        let normed_hidden = mk(hidden_bytes)?;
        let residual_hidden = mk(hidden_bytes)?;

        let fa_qkv_combined = mk((batch * FULL_ATTN_TOTAL_OUT * 4) as u64)?;
        let fa_q_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let fa_q_rot = mk((batch * num_heads * head_dim * 4) as u64)?;
        let fa_k_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let fa_k_rot = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let fa_v_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let fa_gate_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_pool = if use_paged_attention {
            None
        } else {
            Some(mk(
                (batch * max_seqlen * num_kv_heads * head_dim * 4) as u64
            )?)
        };
        let v_pool = if use_paged_attention {
            None
        } else {
            Some(mk(
                (batch * max_seqlen * num_kv_heads * head_dim * 4) as u64
            )?)
        };
        let seq_lens_data: Vec<u32> = if use_paged_attention {
            vec![(cur_seq_len + 1) as u32; batch]
        } else {
            vec![max_seqlen as u32; batch]
        };
        let seq_lens_bytes: Vec<u8> = bytemuck::cast_slice(&seq_lens_data).to_vec();
        let seq_lens_buf = mk_dev(&seq_lens_bytes)?;
        let num_chunks = paged_attn_splitk_chunks(batch);
        let paged_cache = if use_paged_attention {
            Some(VkPagedKvCache::new(
                device,
                FULL_ATTN_LAYERS,
                batch * blocks_per_seq,
                block_size,
                num_kv_heads,
                head_dim,
            )?)
        } else {
            None
        };
        let block_table_buf = if use_paged_attention {
            let mut block_ids = Vec::with_capacity(batch * blocks_per_seq);
            for row in 0..batch {
                let row_block_base = row * blocks_per_seq;
                for block in 0..blocks_per_seq {
                    block_ids.push((row_block_base + block) as u32);
                }
            }
            Some(mk_dev(bytemuck::cast_slice(&block_ids))?)
        } else {
            None
        };
        let slots_buf = if use_paged_attention {
            let slots: Vec<u32> = (0..batch)
                .map(|row| (row * blocks_per_seq * block_size + cur_seq_len) as u32)
                .collect();
            Some(mk_dev(bytemuck::cast_slice(&slots))?)
        } else {
            None
        };
        let attn_pre_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_partials = if use_paged_attention {
            let partials_stride = 2 + head_dim;
            Some(mk(
                (batch * num_heads * num_chunks * partials_stride * 4) as u64
            )?)
        } else {
            None
        };
        let attn_post_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_out = mk(hidden_bytes)?;

        let gdn_in_proj_out = mk((batch * gdn_in_proj_total * 4) as u64)?;
        let gdn_z_buf = mk((batch * Z_DIM * 4) as u64)?;
        let gdn_a_buf = mk((batch * A_DIM * 4) as u64)?;
        let gdn_b_buf = mk((batch * B_DIM * 4) as u64)?;
        let gdn_q_buf = mk((batch * GDN_QK_DIM * 4) as u64)?;
        let gdn_k_buf = mk((batch * GDN_QK_DIM * 4) as u64)?;
        let gdn_q_expanded = mk((batch * GDN_V_DIM * 4) as u64)?;
        let gdn_k_expanded = mk((batch * GDN_V_DIM * 4) as u64)?;
        let gdn_v_buf = mk((batch * GDN_V_DIM * 4) as u64)?;
        let gdn_recurrent_state =
            mk_zero((batch * GDN_NUM_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM * 4) as u64)?;
        let gdn_conv_state = mk_zero((batch * QKV_DIM * (conv_kernel - 1) * 4) as u64)?;
        let gdn_gated_norm = mk((batch * GDN_V_DIM * 4) as u64)?;
        let gdn_out = mk(hidden_bytes)?;

        time(label, batch, || {
            let mut b = CommandBatch::new(device)?;
            for layer in 0..NUM_LAYERS {
                if layer % 4 == 3 {
                    b.record_shader(
                        rmsnorm_shader,
                        &[
                            x_buf.handle(),
                            weight_norm.handle(),
                            fa_qkv_combined.handle(),
                        ],
                        &[batch as u32, HIDDEN as u32, eps.to_bits()],
                        Workgroups::OneD(batch as u32),
                    )?;
                    let total_out = FULL_ATTN_TOTAL_OUT;
                    let (qkv_shader, qkv_workgroups) =
                        full_attn_qkv_gate_split_bf16w_plan(batch, total_out);
                    b.record_shader(
                        qkv_shader,
                        &[
                            fa_qkv_combined.handle(),
                            q_w.handle(),
                            k_w.handle(),
                            v_w.handle(),
                            fa_q_buf.handle(),
                            fa_gate_buf.handle(),
                            fa_k_buf.handle(),
                            fa_v_buf.handle(),
                        ],
                        &[
                            HIDDEN as u32,
                            Q_GATE_DIM as u32,
                            K_DIM as u32,
                            V_DIM as u32,
                            total_out as u32,
                            batch as u32,
                            head_dim as u32,
                        ],
                        Workgroups::OneD(qkv_workgroups),
                    )?;
                    b.record_shader(
                        shaders::QWEN_RMSNORM_QK_COMBINED,
                        &[
                            fa_q_buf.handle(),
                            weight_qknorm.handle(),
                            fa_k_buf.handle(),
                            weight_qknorm.handle(),
                        ],
                        &[
                            (batch * num_heads) as u32,
                            (batch * num_kv_heads) as u32,
                            head_dim as u32,
                            eps.to_bits(),
                        ],
                        Workgroups::OneD((batch * (num_heads + num_kv_heads)) as u32),
                    )?;
                    b.record_shader(
                        rope_shader,
                        &[
                            fa_q_buf.handle(),
                            cos_buf.handle(),
                            sin_buf.handle(),
                            fa_q_rot.handle(),
                        ],
                        &[
                            batch as u32,
                            num_heads as u32,
                            head_dim as u32,
                            rotary_dim as u32,
                        ],
                        Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
                    )?;
                    b.record_shader(
                        rope_shader,
                        &[
                            fa_k_buf.handle(),
                            cos_buf.handle(),
                            sin_buf.handle(),
                            fa_k_rot.handle(),
                        ],
                        &[
                            batch as u32,
                            num_kv_heads as u32,
                            head_dim as u32,
                            rotary_dim as u32,
                        ],
                        Workgroups::OneD((batch * num_kv_heads * head_dim).div_ceil(256) as u32),
                    )?;
                    if use_paged_attention {
                        let cache = paged_cache.as_ref().expect("paged cache");
                        let cache_layer = layer / 4;
                        let elements_per_slot = num_kv_heads * head_dim;
                        let k_pool = cache.k_buffer(cache_layer).expect("full-attn layer K pool");
                        let v_pool = cache.v_buffer(cache_layer).expect("full-attn layer V pool");
                        b.record_shader(
                            kv_write_slots_shader,
                            &[
                                fa_k_rot.handle(),
                                fa_v_buf.handle(),
                                slots_buf.as_ref().expect("slots").handle(),
                                k_pool.handle(),
                                v_pool.handle(),
                            ],
                            &[
                                batch as u32,
                                elements_per_slot as u32,
                                cache.total_slots() as u32,
                            ],
                            Workgroups::OneD((batch * elements_per_slot).div_ceil(256) as u32),
                        )?;
                        b.record_shader(
                            paged_attn_splitk_shader,
                            &[
                                fa_q_rot.handle(),
                                k_pool.handle(),
                                v_pool.handle(),
                                block_table_buf.as_ref().expect("block table").handle(),
                                seq_lens_buf.handle(),
                                attn_partials.as_ref().expect("partials").handle(),
                            ],
                            &[
                                blocks_per_seq as u32,
                                block_size as u32,
                                num_heads as u32,
                                num_kv_heads as u32,
                                head_dim as u32,
                                softmax_scale.to_bits(),
                                num_chunks as u32,
                            ],
                            Workgroups::OneD((batch * num_heads * num_chunks) as u32),
                        )?;
                        b.record_shader(
                            paged_attn_reduce_shader,
                            &[
                                attn_partials.as_ref().expect("partials").handle(),
                                attn_pre_gate.handle(),
                            ],
                            &[num_heads as u32, head_dim as u32, num_chunks as u32],
                            Workgroups::OneD((batch * num_heads) as u32),
                        )?;
                    } else {
                        b.record_shader(
                            paged_attn_shader,
                            &[
                                fa_q_rot.handle(),
                                k_pool.as_ref().expect("contiguous K pool").handle(),
                                v_pool.as_ref().expect("contiguous V pool").handle(),
                                seq_lens_buf.handle(),
                                attn_pre_gate.handle(),
                            ],
                            &[
                                max_seqlen as u32,
                                num_heads as u32,
                                num_kv_heads as u32,
                                head_dim as u32,
                                softmax_scale.to_bits(),
                            ],
                            Workgroups::OneD((batch * num_heads) as u32),
                        )?;
                    }
                    b.record_shader(
                        mul_sigmoid_shader,
                        &[
                            attn_pre_gate.handle(),
                            fa_gate_buf.handle(),
                            attn_post_gate.handle(),
                        ],
                        &[(batch * num_heads * head_dim) as u32],
                        Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
                    )?;
                    let (attn_out_shader, attn_out_workgroups) =
                        linear_bf16w_batched_plan(batch, HIDDEN);
                    b.record_shader(
                        attn_out_shader,
                        &[
                            attn_post_gate.handle(),
                            attn_out_w.handle(),
                            attn_out.handle(),
                        ],
                        &[Q_DIM as u32, HIDDEN as u32, batch as u32],
                        Workgroups::OneD(attn_out_workgroups),
                    )?;
                    b.record_shader(
                        shaders::ADD_QWEN_RMSNORM_BATCHED,
                        &[
                            x_buf.handle(),
                            attn_out.handle(),
                            weight_norm.handle(),
                            residual_hidden.handle(),
                            normed_hidden.handle(),
                        ],
                        &[batch as u32, HIDDEN as u32, eps.to_bits()],
                        Workgroups::OneD(batch as u32),
                    )?;
                    let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
                        mlp_gate_up_bf16w_batched_plan(batch, INTERMEDIATE);
                    b.record_shader(
                        mlp_gate_up_shader,
                        &[
                            normed_hidden.handle(),
                            gate_w.handle(),
                            up_w.handle(),
                            mlp_scratch.handle(),
                        ],
                        &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                        Workgroups::OneD(mlp_gate_up_workgroups),
                    )?;
                    let (mlp_down_shader, mlp_down_workgroups) =
                        mlp_down_add_residual_bf16w_batched_plan(batch, HIDDEN);
                    b.record_shader(
                        mlp_down_shader,
                        &[
                            mlp_scratch.handle(),
                            down_w.handle(),
                            residual_hidden.handle(),
                            x_buf.handle(),
                        ],
                        &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                        Workgroups::OneD(mlp_down_workgroups),
                    )?;
                } else {
                    b.record_shader(
                        shaders::QWEN_RMSNORM_FORWARD,
                        &[x_buf.handle(), weight_norm.handle(), normed_hidden.handle()],
                        &[batch as u32, HIDDEN as u32, eps.to_bits()],
                        Workgroups::OneD(batch as u32),
                    )?;
                    let (in_proj_shader, in_proj_workgroups) = gdn_in_proj_bf16w_batched_plan(
                        batch,
                        QKV_DIM,
                        Z_DIM,
                        A_DIM,
                        B_DIM,
                        gdn_in_proj_total,
                    );
                    b.record_shader(
                        in_proj_shader,
                        &[
                            normed_hidden.handle(),
                            gdn_qkv_w.handle(),
                            gdn_z_w.handle(),
                            gdn_a_w.handle(),
                            gdn_b_w.handle(),
                            gdn_in_proj_out.handle(),
                        ],
                        &[
                            HIDDEN as u32,
                            QKV_DIM as u32,
                            Z_DIM as u32,
                            A_DIM as u32,
                            B_DIM as u32,
                            gdn_in_proj_total as u32,
                            batch as u32,
                        ],
                        Workgroups::OneD(in_proj_workgroups),
                    )?;
                    b.record_shader(
                        shaders::GDN_DECODE_CONV_SPLIT_BATCHED,
                        &[
                            gdn_in_proj_out.handle(),
                            gdn_conv_w.handle(),
                            gdn_conv_state.handle(),
                            gdn_q_buf.handle(),
                            gdn_k_buf.handle(),
                            gdn_v_buf.handle(),
                            gdn_z_buf.handle(),
                            gdn_a_buf.handle(),
                            gdn_b_buf.handle(),
                        ],
                        &[
                            batch as u32,
                            QKV_DIM as u32,
                            GDN_QK_DIM as u32,
                            GDN_V_DIM as u32,
                            Z_DIM as u32,
                            A_DIM as u32,
                            B_DIM as u32,
                            conv_kernel as u32,
                        ],
                        Workgroups::OneD((batch * gdn_in_proj_total).div_ceil(256) as u32),
                    )?;
                    let l2_eps = 1e-6f32;
                    let q_scale = 1.0f32 / (GDN_HEAD_DIM as f32).sqrt();
                    b.record_shader(
                        shaders::L2_NORM_QK_PER_ROW,
                        &[
                            gdn_q_buf.handle(),
                            gdn_k_buf.handle(),
                            gdn_q_expanded.handle(),
                            gdn_k_expanded.handle(),
                        ],
                        &[
                            (batch * GDN_NUM_KEY_HEADS) as u32,
                            GDN_HEAD_DIM as u32,
                            l2_eps.to_bits(),
                            q_scale.to_bits(),
                            1.0f32.to_bits(),
                            gdn_gqa_ratio as u32,
                        ],
                        Workgroups::OneD((batch * GDN_NUM_VALUE_HEADS) as u32),
                    )?;
                    b.record_shader(
                        shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
                        &[
                            gdn_q_expanded.handle(),
                            gdn_k_expanded.handle(),
                            gdn_v_buf.handle(),
                            gdn_a_buf.handle(),
                            gdn_b_buf.handle(),
                            gdn_a_log.handle(),
                            gdn_dt_bias.handle(),
                            gdn_recurrent_state.handle(),
                            gdn_z_buf.handle(),
                            gdn_recurrent_norm_w.handle(),
                            gdn_gated_norm.handle(),
                        ],
                        &[
                            GDN_NUM_VALUE_HEADS as u32,
                            GDN_HEAD_DIM as u32,
                            GDN_HEAD_DIM as u32,
                            eps.to_bits(),
                            batch as u32,
                        ],
                        Workgroups::OneD((batch * GDN_NUM_VALUE_HEADS) as u32),
                    )?;
                    let (gdn_out_shader, gdn_out_workgroups) =
                        linear_bf16w_batched_plan(batch, HIDDEN);
                    b.record_shader(
                        gdn_out_shader,
                        &[
                            gdn_gated_norm.handle(),
                            gdn_out_w.handle(),
                            gdn_out.handle(),
                        ],
                        &[GDN_V_DIM as u32, HIDDEN as u32, batch as u32],
                        Workgroups::OneD(gdn_out_workgroups),
                    )?;
                    b.record_shader(
                        shaders::ADD,
                        &[x_buf.handle(), gdn_out.handle(), residual_hidden.handle()],
                        &[(batch * HIDDEN) as u32],
                        Workgroups::OneD((batch * HIDDEN).div_ceil(256) as u32),
                    )?;
                    b.record_shader(
                        shaders::QWEN_RMSNORM_FORWARD,
                        &[
                            residual_hidden.handle(),
                            weight_norm.handle(),
                            normed_hidden.handle(),
                        ],
                        &[batch as u32, HIDDEN as u32, eps.to_bits()],
                        Workgroups::OneD(batch as u32),
                    )?;
                    let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
                        mlp_gate_up_bf16w_batched_plan(batch, INTERMEDIATE);
                    b.record_shader(
                        mlp_gate_up_shader,
                        &[
                            normed_hidden.handle(),
                            gate_w.handle(),
                            up_w.handle(),
                            mlp_scratch.handle(),
                        ],
                        &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                        Workgroups::OneD(mlp_gate_up_workgroups),
                    )?;
                    let (mlp_down_shader, mlp_down_workgroups) =
                        mlp_down_add_residual_bf16w_batched_plan(batch, HIDDEN);
                    b.record_shader(
                        mlp_down_shader,
                        &[
                            mlp_scratch.handle(),
                            down_w.handle(),
                            residual_hidden.handle(),
                            x_buf.handle(),
                        ],
                        &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                        Workgroups::OneD(mlp_down_workgroups),
                    )?;
                }
            }
            b.submit_and_wait("full_token_resident_mixed")?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_gdn_block_resident_batched(
    device: &VulkanDevice,
    qkv_w: &VulkanBuffer,
    z_w: &VulkanBuffer,
    a_w: &VulkanBuffer,
    b_w: &VulkanBuffer,
    gate_w: &VulkanBuffer,
    up_w: &VulkanBuffer,
    down_w: &VulkanBuffer,
    batches: &[usize],
) -> Result<()> {
    use kiln_vulkan_kernel::CommandBatch;
    use kiln_vulkan_kernel::Workgroups;

    println!(
        "== gdn_block_resident_batched (GDN block + MLP, 10 kernels recorded into 1 cmd-buffer + 1 submit) =="
    );

    let conv_kernel = 4usize;
    let in_proj_total = QKV_DIM + Z_DIM + A_DIM + B_DIM;
    let eps = 1e-6f32;
    let gqa_ratio = GDN_NUM_VALUE_HEADS / GDN_NUM_KEY_HEADS;
    debug_assert_eq!(GDN_NUM_KEY_HEADS * gqa_ratio, GDN_NUM_VALUE_HEADS);

    let norm_w = upload_f32_buffer_from_slice(device, &vec![1.0f32; HIDDEN])?;
    let recurrent_norm_w = upload_f32_buffer_from_slice(device, &vec![1.0f32; GDN_HEAD_DIM])?;
    let conv_w = upload_f32_buffer_from_slice(device, &vec![0.0f32; QKV_DIM * conv_kernel])?;
    let a_log = upload_f32_buffer_from_slice(device, &vec![-1.0f32; GDN_NUM_VALUE_HEADS])?;
    let dt_bias = upload_f32_buffer_from_slice(device, &vec![0.0f32; GDN_NUM_VALUE_HEADS])?;
    let out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(GDN_V_DIM, HIDDEN))?;

    for &batch in batches {
        let mk = |bytes: u64| {
            VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes,
            )
        };
        let mk_zero = |bytes: u64| -> Result<VulkanBuffer> {
            let buf = mk(bytes)?;
            let zeros = vec![0u8; bytes as usize];
            VulkanBuffer::upload_data(
                device.device(),
                device.host_visible_mem_type(),
                device.queue(),
                device.queue_family_index(),
                &buf,
                &zeros,
            )?;
            Ok(buf)
        };

        let hidden_bytes = (batch * HIDDEN * 4) as u64;
        let x_buf = mk_zero(hidden_bytes)?;
        let normed_pre = mk(hidden_bytes)?;
        let in_proj_out = mk((batch * in_proj_total * 4) as u64)?;
        let z_buf = mk((batch * Z_DIM * 4) as u64)?;
        let a_buf = mk((batch * A_DIM * 4) as u64)?;
        let b_buf = mk((batch * B_DIM * 4) as u64)?;
        let q_buf = mk((batch * GDN_QK_DIM * 4) as u64)?;
        let k_buf = mk((batch * GDN_QK_DIM * 4) as u64)?;
        let q_expanded = mk((batch * GDN_V_DIM * 4) as u64)?;
        let k_expanded = mk((batch * GDN_V_DIM * 4) as u64)?;
        let v_buf = mk((batch * GDN_V_DIM * 4) as u64)?;
        let recurrent_state =
            mk_zero((batch * GDN_NUM_VALUE_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM * 4) as u64)?;
        let conv_state = mk_zero((batch * QKV_DIM * (conv_kernel - 1) * 4) as u64)?;
        let gated_norm = mk((batch * GDN_V_DIM * 4) as u64)?;
        let gdn_out = mk(hidden_bytes)?;
        let attn_residual = mk(hidden_bytes)?;
        let normed_post = mk(hidden_bytes)?;
        let mlp_scratch = mk((batch * INTERMEDIATE * 4) as u64)?;

        time("gdn_block_resident_batched", batch, || {
            let mut b = CommandBatch::new(device)?;
            b.record_shader(
                shaders::QWEN_RMSNORM_FORWARD,
                &[x_buf.handle(), norm_w.handle(), normed_pre.handle()],
                &[batch as u32, HIDDEN as u32, eps.to_bits()],
                Workgroups::OneD(batch as u32),
            )?;
            let (in_proj_shader, in_proj_workgroups) =
                gdn_in_proj_bf16w_batched_plan(batch, QKV_DIM, Z_DIM, A_DIM, B_DIM, in_proj_total);
            b.record_shader(
                in_proj_shader,
                &[
                    normed_pre.handle(),
                    qkv_w.handle(),
                    z_w.handle(),
                    a_w.handle(),
                    b_w.handle(),
                    in_proj_out.handle(),
                ],
                &[
                    HIDDEN as u32,
                    QKV_DIM as u32,
                    Z_DIM as u32,
                    A_DIM as u32,
                    B_DIM as u32,
                    in_proj_total as u32,
                    batch as u32,
                ],
                Workgroups::OneD(in_proj_workgroups),
            )?;
            b.record_shader(
                shaders::GDN_DECODE_CONV_SPLIT_BATCHED,
                &[
                    in_proj_out.handle(),
                    conv_w.handle(),
                    conv_state.handle(),
                    q_buf.handle(),
                    k_buf.handle(),
                    v_buf.handle(),
                    z_buf.handle(),
                    a_buf.handle(),
                    b_buf.handle(),
                ],
                &[
                    batch as u32,
                    QKV_DIM as u32,
                    GDN_QK_DIM as u32,
                    GDN_V_DIM as u32,
                    Z_DIM as u32,
                    A_DIM as u32,
                    B_DIM as u32,
                    conv_kernel as u32,
                ],
                Workgroups::OneD((batch * in_proj_total).div_ceil(256) as u32),
            )?;
            let l2_eps = 1e-6f32;
            let q_scale = 1.0f32 / (GDN_HEAD_DIM as f32).sqrt();
            b.record_shader(
                shaders::L2_NORM_QK_PER_ROW,
                &[
                    q_buf.handle(),
                    k_buf.handle(),
                    q_expanded.handle(),
                    k_expanded.handle(),
                ],
                &[
                    (batch * GDN_NUM_KEY_HEADS) as u32,
                    GDN_HEAD_DIM as u32,
                    l2_eps.to_bits(),
                    q_scale.to_bits(),
                    1.0f32.to_bits(),
                    gqa_ratio as u32,
                ],
                Workgroups::OneD((batch * GDN_NUM_VALUE_HEADS) as u32),
            )?;
            b.record_shader(
                shaders::GDN_DECODE_GATES_RECURRENT_RMSNORM,
                &[
                    q_expanded.handle(),
                    k_expanded.handle(),
                    v_buf.handle(),
                    a_buf.handle(),
                    b_buf.handle(),
                    a_log.handle(),
                    dt_bias.handle(),
                    recurrent_state.handle(),
                    z_buf.handle(),
                    recurrent_norm_w.handle(),
                    gated_norm.handle(),
                ],
                &[
                    GDN_NUM_VALUE_HEADS as u32,
                    GDN_HEAD_DIM as u32,
                    GDN_HEAD_DIM as u32,
                    eps.to_bits(),
                    batch as u32,
                ],
                Workgroups::OneD((batch * GDN_NUM_VALUE_HEADS) as u32),
            )?;
            let (gdn_out_shader, gdn_out_workgroups) = linear_bf16w_batched_plan(batch, HIDDEN);
            b.record_shader(
                gdn_out_shader,
                &[gated_norm.handle(), out_w.handle(), gdn_out.handle()],
                &[GDN_V_DIM as u32, HIDDEN as u32, batch as u32],
                Workgroups::OneD(gdn_out_workgroups),
            )?;
            b.record_shader(
                shaders::ADD,
                &[x_buf.handle(), gdn_out.handle(), attn_residual.handle()],
                &[(batch * HIDDEN) as u32],
                Workgroups::OneD((batch * HIDDEN).div_ceil(256) as u32),
            )?;
            b.record_shader(
                shaders::QWEN_RMSNORM_FORWARD,
                &[
                    attn_residual.handle(),
                    norm_w.handle(),
                    normed_post.handle(),
                ],
                &[batch as u32, HIDDEN as u32, eps.to_bits()],
                Workgroups::OneD(batch as u32),
            )?;
            let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
                mlp_gate_up_bf16w_batched_plan(batch, INTERMEDIATE);
            b.record_shader(
                mlp_gate_up_shader,
                &[
                    normed_post.handle(),
                    gate_w.handle(),
                    up_w.handle(),
                    mlp_scratch.handle(),
                ],
                &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                Workgroups::OneD(mlp_gate_up_workgroups),
            )?;
            let (mlp_down_shader, mlp_down_workgroups) =
                mlp_down_add_residual_bf16w_batched_plan(batch, HIDDEN);
            b.record_shader(
                mlp_down_shader,
                &[
                    mlp_scratch.handle(),
                    down_w.handle(),
                    attn_residual.handle(),
                    x_buf.handle(),
                ],
                &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                Workgroups::OneD(mlp_down_workgroups),
            )?;
            b.submit_and_wait("gdn_block_resident_batched")?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

/// Full-decode-step microbench using the Vulkan-resident dispatchers
/// (gate (e) of docs/vk_resident_decode_plan.md). Simulates one
/// transformer block at Qwen3.5-4B shapes by chaining six resident
/// dispatchers — qwen_rmsnorm, full_attn QKV, paged_attn, linear_decode
/// (out_proj), qwen_rmsnorm, mlp — through pool slots without any host
/// boundary between them. Compared with `full_attn_qkv` / `mlp_bf16w`
/// in isolation this measures the *full-block* per-step overhead the
/// resident path achieves.
#[allow(clippy::too_many_arguments)]
fn run_full_step_resident(
    device: &VulkanDevice,
    q_w: &VulkanBuffer,
    k_w: &VulkanBuffer,
    v_w: &VulkanBuffer,
    gate_w: &VulkanBuffer,
    up_w: &VulkanBuffer,
    down_w: &VulkanBuffer,
    batches: &[usize],
) -> Result<()> {
    use kiln_vulkan_kernel::DecodeResidentPool;
    use kiln_vulkan_kernel::resident::{
        dispatch_add_resident, dispatch_full_attn_qkv_decode_cached_batched_resident,
        dispatch_full_attn_qkv_decode_cached_resident,
        dispatch_linear_decode_cached_bf16_weights_resident,
        dispatch_mlp_decode_cached_bf16_weights_resident, dispatch_mul_sigmoid_gate_resident,
        dispatch_paged_attn_decode_batch_f32_resident, dispatch_qwen_rmsnorm_forward_resident,
        dispatch_rotary_qk_resident,
    };
    use std::sync::Arc;

    println!(
        "== full_step_resident (rmsnorm → QKV → QK-norm → RoPE → paged_attn → out_gate → out_proj → res → rmsnorm → MLP → res) =="
    );
    // Qwen3.5-4B full-attn shapes from ModelConfig::qwen3_5_4b():
    //   num_attention_heads = 16, num_kv_heads = 4, head_dim = 256,
    //   rotary_percentage of 0.25 → rotary_dim = 64.
    // Q_DIM / K_DIM / V_DIM are the file-level constants 4096/1024/1024,
    // which match num_heads * head_dim = 16 * 256 = 4096 and
    // num_kv_heads * head_dim = 4 * 256 = 1024.
    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let rotary_dim = 64usize;
    let half_rot = rotary_dim / 2;
    let max_seqlen = 256usize; // synthetic KV window
    let softmax_scale = (head_dim as f32).sqrt().recip();

    let dev_arc = Arc::new(VulkanDevice::new()?);
    let pool = DecodeResidentPool::try_new(&dev_arc, HIDDEN, INTERMEDIATE, 64)?
        .expect("RTX 6000 Ada has plenty of room for the resident pool");
    // Upload weights directly from host slices.
    let weight_norm = upload_f32_buffer_from_slice(device, &vec![1.0f32; HIDDEN])?;
    let weight_qknorm = upload_f32_buffer_from_slice(device, &vec![1.0f32; head_dim])?;
    let out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(Q_DIM, HIDDEN))?;

    // Synthetic RoPE cos/sin tables for 1 position (the new decode token).
    let cos_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect();
    let sin_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect();
    let cos_buf = upload_f32_buffer_from_slice(device, &cos_data)?;
    let sin_buf = upload_f32_buffer_from_slice(device, &sin_data)?;

    for &batch in batches {
        // Pre-allocate the per-block intermediate buffers once. In a real
        // decode loop these come from `DecodeResidentPool::acquire()` so
        // they're shared across all 32 layers per step.
        let hidden_bytes = (batch * HIDDEN * 4) as u64;
        let x_buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            hidden_bytes,
        )?;
        let final_out = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            hidden_bytes,
        )?;
        let scratch = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            (batch * INTERMEDIATE * 4) as u64,
        )?;
        let qkv_combined = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            (batch * (Q_DIM + K_DIM + V_DIM) * 4) as u64,
        )?;
        // Reshaped Q / K / V buffers (resident, written as separate slots).
        let q_buf_dim = (batch * num_heads * head_dim * 4) as u64;
        let kv_buf_dim = (batch * num_kv_heads * head_dim * 4) as u64;
        let q_buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            q_buf_dim,
        )?;
        let q_rot = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            q_buf_dim,
        )?;
        let k_buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            kv_buf_dim,
        )?;
        let k_rot = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            kv_buf_dim,
        )?;
        let _v_buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            kv_buf_dim,
        )?;
        let gate_buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            q_buf_dim,
        )?;
        // Synthetic K/V pool (max_seqlen tokens of zeros) — stands in for
        // the paged KV cache. In a real implementation the resident path
        // writes the new K/V into this pool at the per-row block-table
        // slot offset; for the bench we just leave it zeroed.
        let kv_pool_size = (batch * max_seqlen * num_kv_heads * head_dim * 4) as u64;
        let k_pool = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            kv_pool_size,
        )?;
        let v_pool = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            kv_pool_size,
        )?;
        // Per-row sequence-length array (one entry per batch row).
        let seq_lens_data: Vec<u32> = vec![max_seqlen as u32; batch];
        let seq_lens_bytes: Vec<u8> = bytemuck::cast_slice(&seq_lens_data).to_vec();
        let seq_lens_buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            seq_lens_bytes.len() as u64,
        )?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &seq_lens_buf,
            &seq_lens_bytes,
        )?;
        let attn_pre_gate = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            q_buf_dim,
        )?;
        let attn_post_gate = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            q_buf_dim,
        )?;
        let attn_out = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            hidden_bytes,
        )?;
        let attn_residual = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            hidden_bytes,
        )?;

        time("full_step_resident", batch, || {
            pool.reset_cursor();
            // 1) Pre-attn rmsnorm into a pool slot
            let normed1 = pool.acquire();
            dispatch_qwen_rmsnorm_forward_resident(
                device,
                &x_buf,
                &weight_norm,
                &normed1,
                batch,
                HIDDEN,
                1e-6,
            )?;
            // 2) Fused QKV → combined buffer
            if batch == 1 {
                dispatch_full_attn_qkv_decode_cached_resident(
                    device,
                    &normed1,
                    q_w,
                    k_w,
                    v_w,
                    &qkv_combined,
                    HIDDEN,
                    Q_DIM,
                    K_DIM,
                    V_DIM,
                    true,
                )?;
            } else {
                dispatch_full_attn_qkv_decode_cached_batched_resident(
                    device,
                    &normed1,
                    q_w,
                    k_w,
                    v_w,
                    &qkv_combined,
                    batch,
                    HIDDEN,
                    Q_DIM,
                    K_DIM,
                    V_DIM,
                    true,
                )?;
            }
            // For the bench we assume the QKV layout is already
            // de-interleaved into q/k/v slot buffers; in the real wire-up
            // a thin split-or-attention-input-step is added. Here we
            // simulate that step by zero-cost reusing buffers — exact
            // ordering only matters for parity testing (which lives in
            // the parity test, not this latency bench).
            // 3) Per-head QK-norm. rows = batch * heads, hidden = head_dim.
            dispatch_qwen_rmsnorm_forward_resident(
                device,
                &q_buf,
                &weight_qknorm,
                &q_buf, // in-place is fine because the shader's writes don't depend on prior writes within a row
                batch * num_heads,
                head_dim,
                1e-6,
            )?;
            dispatch_qwen_rmsnorm_forward_resident(
                device,
                &k_buf,
                &weight_qknorm,
                &k_buf,
                batch * num_kv_heads,
                head_dim,
                1e-6,
            )?;
            // 4) RoPE on Q and K
            dispatch_rotary_qk_resident(
                device,
                &q_buf,
                &k_buf,
                &cos_buf,
                &sin_buf,
                &q_rot,
                &k_rot,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
            )?;
            // 5) Paged attention against the synthetic K/V pool.
            dispatch_paged_attn_decode_batch_f32_resident(
                device,
                &q_rot,
                &k_pool,
                &v_pool,
                &seq_lens_buf,
                &attn_pre_gate,
                batch,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seqlen,
                softmax_scale,
            )?;
            // 6) Output gate: attn * sigmoid(gate)
            dispatch_mul_sigmoid_gate_resident(
                device,
                &attn_pre_gate,
                &gate_buf,
                &attn_post_gate,
                batch * num_heads * head_dim,
            )?;
            // 7) Attention out_proj: Q_DIM → HIDDEN
            dispatch_linear_decode_cached_bf16_weights_resident(
                device,
                &attn_post_gate,
                &out_w,
                &attn_out,
                batch,
                Q_DIM,
                HIDDEN,
            )?;
            // 8) Residual: x + attn_out
            dispatch_add_resident(device, &x_buf, &attn_out, &attn_residual, batch * HIDDEN)?;
            // 9) Pre-MLP rmsnorm
            let normed2 = pool.acquire();
            dispatch_qwen_rmsnorm_forward_resident(
                device,
                &attn_residual,
                &weight_norm,
                &normed2,
                batch,
                HIDDEN,
                1e-6,
            )?;
            // 10) MLP: SwiGLU
            dispatch_mlp_decode_cached_bf16_weights_resident(
                device,
                &normed2,
                gate_w,
                up_w,
                down_w,
                &scratch,
                &final_out,
                batch,
                HIDDEN,
                INTERMEDIATE,
                HIDDEN,
            )?;
            // 11) Final residual
            dispatch_add_resident(
                device,
                &attn_residual,
                &final_out,
                &x_buf, // overwrite next-layer x
                batch * HIDDEN,
            )?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

/// Same physical block as `run_full_step_resident` but recorded into
/// a single `CommandBatch` and submitted once per iteration. This is
/// the architecturally correct shape for gate (e.1) of the plan: the
/// per-step submit count collapses from `O(layers × kernels)` to
/// `O(1)`. Comparing the two modes gives the direct contribution of
/// queue-submission overhead on the device under test.
#[allow(clippy::too_many_arguments)]
fn run_full_step_resident_batched(
    device: &VulkanDevice,
    q_w: &VulkanBuffer,
    k_w: &VulkanBuffer,
    v_w: &VulkanBuffer,
    gate_w: &VulkanBuffer,
    up_w: &VulkanBuffer,
    down_w: &VulkanBuffer,
    batches: &[usize],
) -> Result<()> {
    use kiln_vulkan_kernel::CommandBatch;
    use kiln_vulkan_kernel::DecodeResidentPool;
    use kiln_vulkan_kernel::Workgroups;
    use std::sync::Arc;

    println!(
        "== full_step_resident_batched (gated-Q full-attn block, recorded into 1 command-buffer + 1 submit) =="
    );

    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let rotary_dim = 64usize;
    let half_rot = rotary_dim / 2;
    let max_seqlen = 256usize;
    let softmax_scale = (head_dim as f32).sqrt().recip();

    let dev_arc = Arc::new(VulkanDevice::new()?);
    let _pool = DecodeResidentPool::try_new(&dev_arc, HIDDEN, INTERMEDIATE, 64)?
        .expect("RTX 6000 Ada has plenty of room for the resident pool");
    // Upload weights directly from host slices.
    let weight_norm = upload_f32_buffer_from_slice(device, &vec![1.0f32; HIDDEN])?;
    let weight_qknorm = upload_f32_buffer_from_slice(device, &vec![1.0f32; head_dim])?;
    let out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(Q_DIM, HIDDEN))?;

    let cos_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect();
    let sin_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect();
    let cos_buf = upload_f32_buffer_from_slice(device, &cos_data)?;
    let sin_buf = upload_f32_buffer_from_slice(device, &sin_data)?;

    // Shader paths reused across every iteration.
    let rmsnorm_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let rope_shader = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_rope_f32.comp");
    let paged_attn_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let mul_sigmoid_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_mul_sigmoid_gate_f32.comp"
    );
    for &batch in batches {
        let hidden_bytes = (batch * HIDDEN * 4) as u64;
        let mk = |bytes: u64| {
            VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes,
            )
        };
        let x_buf = mk(hidden_bytes)?;
        let scratch = mk((batch * INTERMEDIATE * 4) as u64)?;
        let qkv_combined = mk((batch * FULL_ATTN_TOTAL_OUT * 4) as u64)?;
        let q_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let q_rot = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let k_rot = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let v_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let gate_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_pool = mk((batch * max_seqlen * num_kv_heads * head_dim * 4) as u64)?;
        let v_pool = mk((batch * max_seqlen * num_kv_heads * head_dim * 4) as u64)?;
        let seq_lens_data: Vec<u32> = vec![max_seqlen as u32; batch];
        let seq_lens_bytes: Vec<u8> = bytemuck::cast_slice(&seq_lens_data).to_vec();
        let seq_lens_buf = mk(seq_lens_bytes.len() as u64)?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &seq_lens_buf,
            &seq_lens_bytes,
        )?;
        let attn_pre_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_post_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_out = mk(hidden_bytes)?;
        let attn_residual = mk(hidden_bytes)?;

        time("full_step_resident_batched", batch, || {
            let mut b = CommandBatch::new(device)?;

            // 1) Pre-attn rmsnorm
            b.record_shader(
                rmsnorm_shader,
                &[x_buf.handle(), weight_norm.handle(), qkv_combined.handle()],
                &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                Workgroups::OneD(batch as u32),
            )?;
            // 2) QKV projection, writing q/gate/k/v outputs directly.
            let total_out = FULL_ATTN_TOTAL_OUT;
            let (qkv_shader, qkv_workgroups) =
                full_attn_qkv_gate_split_bf16w_plan(batch, total_out);
            b.record_shader(
                qkv_shader,
                &[
                    qkv_combined.handle(),
                    q_w.handle(),
                    k_w.handle(),
                    v_w.handle(),
                    q_buf.handle(),
                    gate_buf.handle(),
                    k_buf.handle(),
                    v_buf.handle(),
                ],
                &[
                    HIDDEN as u32,
                    Q_GATE_DIM as u32,
                    K_DIM as u32,
                    V_DIM as u32,
                    total_out as u32,
                    batch as u32,
                    head_dim as u32,
                ],
                Workgroups::OneD(qkv_workgroups),
            )?;
            b.record_shader(
                shaders::QWEN_RMSNORM_QK_COMBINED,
                &[
                    q_buf.handle(),
                    weight_qknorm.handle(),
                    k_buf.handle(),
                    weight_qknorm.handle(),
                ],
                &[
                    (batch * num_heads) as u32,
                    (batch * num_kv_heads) as u32,
                    head_dim as u32,
                    (1e-6f32).to_bits(),
                ],
                Workgroups::OneD((batch * (num_heads + num_kv_heads)) as u32),
            )?;
            // 4) RoPE on Q
            b.record_shader(
                rope_shader,
                &[
                    q_buf.handle(),
                    cos_buf.handle(),
                    sin_buf.handle(),
                    q_rot.handle(),
                ],
                &[
                    batch as u32,
                    num_heads as u32,
                    head_dim as u32,
                    rotary_dim as u32,
                ],
                Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
            )?;
            b.record_shader(
                rope_shader,
                &[
                    k_buf.handle(),
                    cos_buf.handle(),
                    sin_buf.handle(),
                    k_rot.handle(),
                ],
                &[
                    batch as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    rotary_dim as u32,
                ],
                Workgroups::OneD((batch * num_kv_heads * head_dim).div_ceil(256) as u32),
            )?;
            // 5) Paged attention
            b.record_shader(
                paged_attn_shader,
                &[
                    q_rot.handle(),
                    k_pool.handle(),
                    v_pool.handle(),
                    seq_lens_buf.handle(),
                    attn_pre_gate.handle(),
                ],
                &[
                    max_seqlen as u32,
                    num_heads as u32,
                    num_kv_heads as u32,
                    head_dim as u32,
                    softmax_scale.to_bits(),
                ],
                Workgroups::OneD((batch * num_heads) as u32),
            )?;
            // 6) Output gate
            b.record_shader(
                mul_sigmoid_shader,
                &[
                    attn_pre_gate.handle(),
                    gate_buf.handle(),
                    attn_post_gate.handle(),
                ],
                &[(batch * num_heads * head_dim) as u32],
                Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
            )?;
            // 7) Attention out_proj
            let (attn_out_shader, attn_out_workgroups) = linear_bf16w_batched_plan(batch, HIDDEN);
            b.record_shader(
                attn_out_shader,
                &[attn_post_gate.handle(), out_w.handle(), attn_out.handle()],
                &[Q_DIM as u32, HIDDEN as u32, batch as u32],
                Workgroups::OneD(attn_out_workgroups),
            )?;
            b.record_shader(
                shaders::ADD_QWEN_RMSNORM_BATCHED,
                &[
                    x_buf.handle(),
                    attn_out.handle(),
                    weight_norm.handle(),
                    attn_residual.handle(),
                    qkv_combined.handle(),
                ],
                &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                Workgroups::OneD(batch as u32),
            )?;
            // 10) MLP gate_up
            let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
                mlp_gate_up_bf16w_batched_plan(batch, INTERMEDIATE);
            b.record_shader(
                mlp_gate_up_shader,
                &[
                    qkv_combined.handle(),
                    gate_w.handle(),
                    up_w.handle(),
                    scratch.handle(),
                ],
                &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                Workgroups::OneD(mlp_gate_up_workgroups),
            )?;
            let (mlp_down_shader, mlp_down_workgroups) =
                mlp_down_add_residual_bf16w_batched_plan(batch, HIDDEN);
            b.record_shader(
                mlp_down_shader,
                &[
                    scratch.handle(),
                    down_w.handle(),
                    attn_residual.handle(),
                    x_buf.handle(),
                ],
                &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                Workgroups::OneD(mlp_down_workgroups),
            )?;
            b.submit_and_wait("full_step_resident_batched")?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

/// Full 32-layer decode token recorded into ONE CommandBatch and
/// submitted ONCE per token. This is the gate-(e.1) end-state: per-step
/// submit count is O(1) regardless of layer count. Compared against
/// `full_step_resident_batched` × 32 it measures the residual queue
/// overhead per batch boundary; compared against `full_step_resident` × 32
/// it measures the full resident + batched win against the
/// per-call legacy floor.
#[allow(clippy::too_many_arguments)]
fn run_full_token_resident_batched(
    device: &VulkanDevice,
    q_w: &VulkanBuffer,
    k_w: &VulkanBuffer,
    v_w: &VulkanBuffer,
    gate_w: &VulkanBuffer,
    up_w: &VulkanBuffer,
    down_w: &VulkanBuffer,
    batches: &[usize],
) -> Result<()> {
    use kiln_vulkan_kernel::CommandBatch;
    use kiln_vulkan_kernel::Workgroups;

    const NUM_LAYERS: usize = 32;
    println!(
        "== full_token_resident_batched ({NUM_LAYERS} gated-Q full-attn layers recorded into 1 cmd-buffer + 1 submit) =="
    );

    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let rotary_dim = 64usize;
    let half_rot = rotary_dim / 2;
    let max_seqlen = 256usize;
    let softmax_scale = (head_dim as f32).sqrt().recip();

    // Upload weights directly from host slices.
    let weight_norm = upload_f32_buffer_from_slice(device, &vec![1.0f32; HIDDEN])?;
    let weight_qknorm = upload_f32_buffer_from_slice(device, &vec![1.0f32; head_dim])?;
    let out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(Q_DIM, HIDDEN))?;

    let cos_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect();
    let sin_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect();
    let cos_buf = upload_f32_buffer_from_slice(device, &cos_data)?;
    let sin_buf = upload_f32_buffer_from_slice(device, &sin_data)?;

    let rmsnorm_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let rope_shader = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_rope_f32.comp");
    let paged_attn_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let mul_sigmoid_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_mul_sigmoid_gate_f32.comp"
    );
    for &batch in batches {
        let hidden_bytes = (batch * HIDDEN * 4) as u64;
        let mk = |bytes: u64| {
            VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes,
            )
        };
        let x_buf = mk(hidden_bytes)?;
        let scratch = mk((batch * INTERMEDIATE * 4) as u64)?;
        let qkv_combined = mk((batch * FULL_ATTN_TOTAL_OUT * 4) as u64)?;
        let q_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let q_rot = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let k_rot = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let v_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let gate_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_pool = mk((batch * max_seqlen * num_kv_heads * head_dim * 4) as u64)?;
        let v_pool = mk((batch * max_seqlen * num_kv_heads * head_dim * 4) as u64)?;
        let seq_lens_data: Vec<u32> = vec![max_seqlen as u32; batch];
        let seq_lens_bytes: Vec<u8> = bytemuck::cast_slice(&seq_lens_data).to_vec();
        let seq_lens_buf = mk(seq_lens_bytes.len() as u64)?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &seq_lens_buf,
            &seq_lens_bytes,
        )?;
        let attn_pre_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_post_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_out = mk(hidden_bytes)?;
        let attn_residual = mk(hidden_bytes)?;

        time("full_token_resident_batched", batch, || {
            let mut b = CommandBatch::new(device)?;
            for _layer in 0..NUM_LAYERS {
                // 1) Pre-attn rmsnorm
                b.record_shader(
                    rmsnorm_shader,
                    &[x_buf.handle(), weight_norm.handle(), qkv_combined.handle()],
                    &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                    Workgroups::OneD(batch as u32),
                )?;
                let total_out = FULL_ATTN_TOTAL_OUT;
                let (qkv_shader, qkv_workgroups) =
                    full_attn_qkv_gate_split_bf16w_plan(batch, total_out);
                b.record_shader(
                    qkv_shader,
                    &[
                        qkv_combined.handle(),
                        q_w.handle(),
                        k_w.handle(),
                        v_w.handle(),
                        q_buf.handle(),
                        gate_buf.handle(),
                        k_buf.handle(),
                        v_buf.handle(),
                    ],
                    &[
                        HIDDEN as u32,
                        Q_GATE_DIM as u32,
                        K_DIM as u32,
                        V_DIM as u32,
                        total_out as u32,
                        batch as u32,
                        head_dim as u32,
                    ],
                    Workgroups::OneD(qkv_workgroups),
                )?;
                b.record_shader(
                    shaders::QWEN_RMSNORM_QK_COMBINED,
                    &[
                        q_buf.handle(),
                        weight_qknorm.handle(),
                        k_buf.handle(),
                        weight_qknorm.handle(),
                    ],
                    &[
                        (batch * num_heads) as u32,
                        (batch * num_kv_heads) as u32,
                        head_dim as u32,
                        (1e-6f32).to_bits(),
                    ],
                    Workgroups::OneD((batch * (num_heads + num_kv_heads)) as u32),
                )?;
                b.record_shader(
                    rope_shader,
                    &[
                        q_buf.handle(),
                        cos_buf.handle(),
                        sin_buf.handle(),
                        q_rot.handle(),
                    ],
                    &[
                        batch as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                    ],
                    Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
                )?;
                b.record_shader(
                    rope_shader,
                    &[
                        k_buf.handle(),
                        cos_buf.handle(),
                        sin_buf.handle(),
                        k_rot.handle(),
                    ],
                    &[
                        batch as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                    ],
                    Workgroups::OneD((batch * num_kv_heads * head_dim).div_ceil(256) as u32),
                )?;
                b.record_shader(
                    paged_attn_shader,
                    &[
                        q_rot.handle(),
                        k_pool.handle(),
                        v_pool.handle(),
                        seq_lens_buf.handle(),
                        attn_pre_gate.handle(),
                    ],
                    &[
                        max_seqlen as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        softmax_scale.to_bits(),
                    ],
                    Workgroups::OneD((batch * num_heads) as u32),
                )?;
                b.record_shader(
                    mul_sigmoid_shader,
                    &[
                        attn_pre_gate.handle(),
                        gate_buf.handle(),
                        attn_post_gate.handle(),
                    ],
                    &[(batch * num_heads * head_dim) as u32],
                    Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
                )?;
                let (attn_out_shader, attn_out_workgroups) =
                    linear_bf16w_batched_plan(batch, HIDDEN);
                b.record_shader(
                    attn_out_shader,
                    &[attn_post_gate.handle(), out_w.handle(), attn_out.handle()],
                    &[Q_DIM as u32, HIDDEN as u32, batch as u32],
                    Workgroups::OneD(attn_out_workgroups),
                )?;
                b.record_shader(
                    shaders::ADD_QWEN_RMSNORM_BATCHED,
                    &[
                        x_buf.handle(),
                        attn_out.handle(),
                        weight_norm.handle(),
                        attn_residual.handle(),
                        qkv_combined.handle(),
                    ],
                    &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                    Workgroups::OneD(batch as u32),
                )?;
                let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
                    mlp_gate_up_bf16w_batched_plan(batch, INTERMEDIATE);
                b.record_shader(
                    mlp_gate_up_shader,
                    &[
                        qkv_combined.handle(),
                        gate_w.handle(),
                        up_w.handle(),
                        scratch.handle(),
                    ],
                    &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                    Workgroups::OneD(mlp_gate_up_workgroups),
                )?;
                let (mlp_down_shader, mlp_down_workgroups) =
                    mlp_down_add_residual_bf16w_batched_plan(batch, HIDDEN);
                b.record_shader(
                    mlp_down_shader,
                    &[
                        scratch.handle(),
                        down_w.handle(),
                        attn_residual.handle(),
                        x_buf.handle(),
                    ],
                    &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                    Workgroups::OneD(mlp_down_workgroups),
                )?;
            }
            b.submit_and_wait("full_token_resident_batched")?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

/// Per-token decode bench against a *real* Vulkan-resident paged KV
/// cache: each layer writes its freshly-projected K/V into the
/// VkPagedKvCache via a batched slot write and reads the entire
/// window back via split-K paged attention.
///
/// Distinct from `full_token_resident_batched` which uses a
/// non-paged contiguous K/V pool sized for `max_seqlen` and never
/// actually writes to it — that mode measures only compute + submit.
/// This mode adds the per-step KV-write cost and the paged-attention
/// read cost over real per-row block tables, which is what real
/// Qwen3.5-4B decode pays. The full token still records into one
/// `CommandBatch` and ships as a single submit.
#[allow(clippy::too_many_arguments)]
fn run_full_token_resident_paged(
    device: &VulkanDevice,
    q_w: &VulkanBuffer,
    k_w: &VulkanBuffer,
    v_w: &VulkanBuffer,
    gate_w: &VulkanBuffer,
    up_w: &VulkanBuffer,
    down_w: &VulkanBuffer,
    batches: &[usize],
) -> Result<()> {
    use kiln_vulkan_kernel::CommandBatch;
    use kiln_vulkan_kernel::VkPagedKvCache;
    use kiln_vulkan_kernel::Workgroups;

    const NUM_LAYERS: usize = 32;

    let num_heads = 16usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let rotary_dim = 64usize;
    let half_rot = rotary_dim / 2;
    let cur_seq_len = env_usize("KILN_VK_PAGED_HISTORY", 256);
    let block_size = env_usize("KILN_VK_PAGED_BLOCK_SIZE", 16);
    anyhow::ensure!(block_size > 0, "KILN_VK_PAGED_BLOCK_SIZE must be > 0");
    let blocks_per_seq = (cur_seq_len + 1).div_ceil(block_size).max(1);
    let softmax_scale = (head_dim as f32).sqrt().recip();
    println!(
        "== full_token_resident_paged ({NUM_LAYERS} layers, paged KV cache, history={cur_seq_len}, block={block_size}, 1 KV-write + paged-attn read per layer, 1 submit / token) =="
    );

    // Upload weights directly from host slices.
    let weight_norm = upload_f32_buffer_from_slice(device, &vec![1.0f32; HIDDEN])?;
    let weight_qknorm = upload_f32_buffer_from_slice(device, &vec![1.0f32; head_dim])?;
    let out_w =
        upload_bf16_packed_buffer_from_slice(device, &make_bf16_weight_slice(Q_DIM, HIDDEN))?;

    let cos_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect();
    let sin_data: Vec<f32> = (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect();
    let cos_buf = upload_f32_buffer_from_slice(device, &cos_data)?;
    let sin_buf = upload_f32_buffer_from_slice(device, &sin_data)?;

    let rmsnorm_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let rope_shader = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/vk_rope_f32.comp");
    let paged_attn_splitk_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged_splitk.comp"
    );
    let paged_attn_reduce_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged_splitk_reduce.comp"
    );
    let kv_write_slots_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_kv_write_slots.comp"
    );
    let mul_sigmoid_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_mul_sigmoid_gate_f32.comp"
    );
    let mk_dev = |bytes: &[u8]| -> Result<VulkanBuffer> {
        let buf = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes.len() as u64,
        )?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buf,
            bytes,
        )?;
        Ok(buf)
    };

    for &batch in batches {
        let num_chunks = paged_attn_splitk_chunks(batch);
        let total_blocks = batch * blocks_per_seq;
        let cache = VkPagedKvCache::new(
            device,
            NUM_LAYERS,
            total_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )?;
        let mut block_ids = Vec::with_capacity(batch * blocks_per_seq);
        for row in 0..batch {
            let row_block_base = row * blocks_per_seq;
            for block in 0..blocks_per_seq {
                block_ids.push((row_block_base + block) as u32);
            }
        }
        let block_table_bytes: Vec<u8> = bytemuck::cast_slice(&block_ids).to_vec();
        let block_table_buf = mk_dev(&block_table_bytes)?;
        let seq_lens_data: Vec<u32> = vec![(cur_seq_len + 1) as u32; batch];
        let seq_lens_bytes: Vec<u8> = bytemuck::cast_slice(&seq_lens_data).to_vec();
        let slots_data: Vec<u32> = (0..batch)
            .map(|row| (row * blocks_per_seq * block_size + cur_seq_len) as u32)
            .collect();
        let slots_bytes: Vec<u8> = bytemuck::cast_slice(&slots_data).to_vec();
        let hidden_bytes = (batch * HIDDEN * 4) as u64;
        let mk = |bytes: u64| {
            VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes,
            )
        };
        let x_buf = mk(hidden_bytes)?;
        let scratch = mk((batch * INTERMEDIATE * 4) as u64)?;
        let qkv_combined = mk((batch * FULL_ATTN_TOTAL_OUT * 4) as u64)?;
        let q_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let q_rot = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let k_rot = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let v_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let gate_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let seq_lens_buf = mk_dev(&seq_lens_bytes)?;
        let slots_buf = mk_dev(&slots_bytes)?;
        let attn_pre_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let partials_stride = 2 + head_dim;
        let attn_partials = mk((batch * num_heads * num_chunks * partials_stride * 4) as u64)?;
        let attn_post_gate = mk((batch * num_heads * head_dim * 4) as u64)?;
        let attn_out = mk(hidden_bytes)?;
        let attn_residual = mk(hidden_bytes)?;

        time("full_token_resident_paged", batch, || {
            let mut b = CommandBatch::new(device)?;
            for layer in 0..NUM_LAYERS {
                let k_pool = cache.k_buffer(layer).expect("layer in range");
                let v_pool = cache.v_buffer(layer).expect("layer in range");

                // 1) Pre-attn rmsnorm
                b.record_shader(
                    rmsnorm_shader,
                    &[x_buf.handle(), weight_norm.handle(), qkv_combined.handle()],
                    &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                    Workgroups::OneD(batch as u32),
                )?;
                // 2) QKV projection, writing q/gate/k/v outputs directly.
                let total_out = FULL_ATTN_TOTAL_OUT;
                let (qkv_shader, qkv_workgroups) =
                    full_attn_qkv_gate_split_bf16w_plan(batch, total_out);
                b.record_shader(
                    qkv_shader,
                    &[
                        qkv_combined.handle(),
                        q_w.handle(),
                        k_w.handle(),
                        v_w.handle(),
                        q_buf.handle(),
                        gate_buf.handle(),
                        k_buf.handle(),
                        v_buf.handle(),
                    ],
                    &[
                        HIDDEN as u32,
                        Q_GATE_DIM as u32,
                        K_DIM as u32,
                        V_DIM as u32,
                        total_out as u32,
                        batch as u32,
                        head_dim as u32,
                    ],
                    Workgroups::OneD(qkv_workgroups),
                )?;
                b.record_shader(
                    shaders::QWEN_RMSNORM_QK_COMBINED,
                    &[
                        q_buf.handle(),
                        weight_qknorm.handle(),
                        k_buf.handle(),
                        weight_qknorm.handle(),
                    ],
                    &[
                        (batch * num_heads) as u32,
                        (batch * num_kv_heads) as u32,
                        head_dim as u32,
                        (1e-6f32).to_bits(),
                    ],
                    Workgroups::OneD((batch * (num_heads + num_kv_heads)) as u32),
                )?;
                // 5) RoPE Q
                b.record_shader(
                    rope_shader,
                    &[
                        q_buf.handle(),
                        cos_buf.handle(),
                        sin_buf.handle(),
                        q_rot.handle(),
                    ],
                    &[
                        batch as u32,
                        num_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                    ],
                    Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
                )?;
                // 6) RoPE K
                b.record_shader(
                    rope_shader,
                    &[
                        k_buf.handle(),
                        cos_buf.handle(),
                        sin_buf.handle(),
                        k_rot.handle(),
                    ],
                    &[
                        batch as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        rotary_dim as u32,
                    ],
                    Workgroups::OneD((batch * num_kv_heads * head_dim).div_ceil(256) as u32),
                )?;
                // 7) Write K/V to resident paged pool at each row's slot.
                let elements_per_slot = num_kv_heads * head_dim;
                b.record_shader(
                    kv_write_slots_shader,
                    &[
                        k_rot.handle(),
                        v_buf.handle(),
                        slots_buf.handle(),
                        k_pool.handle(),
                        v_pool.handle(),
                    ],
                    &[
                        batch as u32,
                        elements_per_slot as u32,
                        cache.total_slots() as u32,
                    ],
                    Workgroups::OneD((batch * elements_per_slot).div_ceil(256) as u32),
                )?;
                // 8) Split-K paged attention against the whole pool.
                b.record_shader(
                    paged_attn_splitk_shader,
                    &[
                        q_rot.handle(),
                        k_pool.handle(),
                        v_pool.handle(),
                        block_table_buf.handle(),
                        seq_lens_buf.handle(),
                        attn_partials.handle(),
                    ],
                    &[
                        blocks_per_seq as u32,
                        block_size as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        softmax_scale.to_bits(),
                        num_chunks as u32,
                    ],
                    Workgroups::OneD((batch * num_heads * num_chunks) as u32),
                )?;
                b.record_shader(
                    paged_attn_reduce_shader,
                    &[attn_partials.handle(), attn_pre_gate.handle()],
                    &[num_heads as u32, head_dim as u32, num_chunks as u32],
                    Workgroups::OneD((batch * num_heads) as u32),
                )?;
                // 9) Attention output gate
                b.record_shader(
                    mul_sigmoid_shader,
                    &[
                        attn_pre_gate.handle(),
                        gate_buf.handle(),
                        attn_post_gate.handle(),
                    ],
                    &[(batch * num_heads * head_dim) as u32],
                    Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
                )?;
                // 10) Output projection
                let (attn_out_shader, attn_out_workgroups) =
                    linear_bf16w_batched_plan(batch, HIDDEN);
                b.record_shader(
                    attn_out_shader,
                    &[attn_post_gate.handle(), out_w.handle(), attn_out.handle()],
                    &[Q_DIM as u32, HIDDEN as u32, batch as u32],
                    Workgroups::OneD(attn_out_workgroups),
                )?;
                b.record_shader(
                    shaders::ADD_QWEN_RMSNORM_BATCHED,
                    &[
                        x_buf.handle(),
                        attn_out.handle(),
                        weight_norm.handle(),
                        attn_residual.handle(),
                        qkv_combined.handle(),
                    ],
                    &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                    Workgroups::OneD(batch as u32),
                )?;
                // 13) MLP gate-up
                let (mlp_gate_up_shader, mlp_gate_up_workgroups) =
                    mlp_gate_up_bf16w_batched_plan(batch, INTERMEDIATE);
                b.record_shader(
                    mlp_gate_up_shader,
                    &[
                        qkv_combined.handle(),
                        gate_w.handle(),
                        up_w.handle(),
                        scratch.handle(),
                    ],
                    &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                    Workgroups::OneD(mlp_gate_up_workgroups),
                )?;
                let (mlp_down_shader, mlp_down_workgroups) =
                    mlp_down_add_residual_bf16w_batched_plan(batch, HIDDEN);
                b.record_shader(
                    mlp_down_shader,
                    &[
                        scratch.handle(),
                        down_w.handle(),
                        attn_residual.handle(),
                        x_buf.handle(),
                    ],
                    &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                    Workgroups::OneD(mlp_down_workgroups),
                )?;
            }
            b.submit_and_wait("full_token_resident_paged")?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    run()
}
