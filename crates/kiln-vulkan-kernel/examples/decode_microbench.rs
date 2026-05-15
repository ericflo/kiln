//! Decode-path Vulkan microbench.
//!
//! Measures wall-clock per-iteration latency for the three single-token
//! decode-hot kernels at Qwen3.5-4B shapes: full_attn QKV, GDN in_proj,
//! and MLP gate_up + down. Exercises the same `dispatch_*_cached_*`
//! entry points the production decode loop uses, including host upload
//! of `x` and host readback of the output, so the numbers reflect
//! end-to-end per-call cost.
//!
//! Usage: `cargo run --release --example decode_microbench -p kiln-vulkan-kernel`.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use kiln_vulkan_kernel::buffer::VulkanBuffer;
use kiln_vulkan_kernel::device::VulkanDevice;
use kiln_vulkan_kernel::kernels::{upload_tensor_bf16_packed_buffer, upload_tensor_f32_buffer};

// Used by run_full_step_resident — keep the module-level imports here so the
// helper itself stays terse.

const HIDDEN: usize = 2560;
const Q_DIM: usize = 4096;
const K_DIM: usize = 1024;
const V_DIM: usize = 1024;
const INTERMEDIATE: usize = 9216;

// GDN shapes
const QKV_DIM: usize = 4096; // linear_num_key_heads * head_dim = 16 * 128 = 2048 ish; the layout used by Qwen3.5
const Z_DIM: usize = 4096;
const A_DIM: usize = 32;
const B_DIM: usize = 32;

const WARMUP_ITERS: usize = 10;
const TIMED_ITERS: usize = 30;
const REPEATS: usize = 5;

fn make_bf16_weight(rows: usize, cols: usize) -> Result<Tensor> {
    let n = rows * cols;
    let data: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32(((i % 31) as f32 - 15.0) * 0.01))
        .collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).map_err(Into::into)
}

fn upload_bf16_packed(device: &VulkanDevice, t: &Tensor) -> Result<VulkanBuffer> {
    upload_tensor_bf16_packed_buffer(device, t)
}

fn time<F: FnMut() -> Result<()>>(label: &str, batch: usize, mut f: F) -> Result<()> {
    for _ in 0..WARMUP_ITERS {
        f()?;
    }
    // Take the minimum per-iter time across REPEATS independent timed blocks.
    // The fastest block is the cleanest signal of steady-state kernel cost;
    // mean is dragged around by background load and GPU thermal swings.
    let mut best_ns = u128::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..TIMED_ITERS {
            f()?;
        }
        let elapsed = start.elapsed().as_nanos();
        if elapsed < best_ns {
            best_ns = elapsed;
        }
    }
    let per_iter_us = (best_ns as f64 / TIMED_ITERS as f64) / 1_000.0;
    let rows_per_sec = (batch as f64 * TIMED_ITERS as f64) / (best_ns as f64 / 1e9);
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
    let q_w = make_bf16_weight(HIDDEN, Q_DIM)?;
    let k_w = make_bf16_weight(HIDDEN, K_DIM)?;
    let v_w = make_bf16_weight(HIDDEN, V_DIM)?;
    let gate_w = make_bf16_weight(HIDDEN, INTERMEDIATE)?;
    let up_w = make_bf16_weight(HIDDEN, INTERMEDIATE)?;
    let down_w = make_bf16_weight(INTERMEDIATE, HIDDEN)?;
    let down_w_f32 = down_w.to_dtype(DType::F32)?;
    let qkv_w = make_bf16_weight(HIDDEN, QKV_DIM)?;
    let z_w = make_bf16_weight(HIDDEN, Z_DIM)?;
    let a_w = make_bf16_weight(HIDDEN, A_DIM)?;
    let b_w = make_bf16_weight(HIDDEN, B_DIM)?;

    let q_buf = upload_bf16_packed(&device, &q_w)?;
    let k_buf = upload_bf16_packed(&device, &k_w)?;
    let v_buf = upload_bf16_packed(&device, &v_w)?;
    let gate_buf = upload_bf16_packed(&device, &gate_w)?;
    let up_buf = upload_bf16_packed(&device, &up_w)?;
    let down_buf = upload_bf16_packed(&device, &down_w)?;
    // f32 down buffer for bf16_gate_up_f32_down variant.
    let down_f32_buf = upload_tensor_f32_buffer(&device, &down_w_f32)?;
    let qkv_buf = upload_bf16_packed(&device, &qkv_w)?;
    let z_buf = upload_bf16_packed(&device, &z_w)?;
    let a_buf = upload_bf16_packed(&device, &a_w)?;
    let b_buf = upload_bf16_packed(&device, &b_w)?;

    let batches: [usize; 6] = [1, 4, 8, 16, 32, 64];

    if want("full_attn_qkv") {
        println!("== full_attn QKV (fused, bf16w) ==");
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            time("full_attn_qkv_decode", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bf16_weights(
                    &device, &x, &q_buf, &k_buf, &v_buf, batch, HIDDEN, Q_DIM, K_DIM, V_DIM,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("mlp_bf16_gu_f32_d") {
        println!("== MLP gate_up + down (bf16 g/u, f32 down) ==");
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            time("mlp_decode_bf16_gu_f32_d", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_gate_up_f32_down(
                    &device,
                    &x,
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
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            time("mlp_decode_bf16w", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_weights(
                    &device,
                    &x,
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
        let q_out_buf = upload_bf16_packed(&device, &make_bf16_weight(Q_DIM, HIDDEN)?)?;
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, Q_DIM), DType::F32, &Device::Cpu)?;
            time("linear_decode_bf16w_qout", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights(
                    &device, &x, &q_out_buf, batch, Q_DIM, HIDDEN,
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
        let weight = Tensor::zeros((channels, kernel_size), DType::F32, &Device::Cpu)?;
        for &batch in &batches {
            let x = Tensor::zeros((batch, channels, 1usize), DType::F32, &Device::Cpu)?;
            let state = Tensor::zeros(
                (batch, channels, kernel_size - 1),
                DType::F32,
                &Device::Cpu,
            )?;
            time("causal_conv1d_update", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update(
                    &device, &x, &weight, &state, kernel_size,
                )?;
                Ok(())
            })?;
        }
        println!();
    }

    if want("gdn_gated_norm") {
        println!("== gdn_gated_rms_norm_cached (hidden=2560) ==");
        let weight_t = Tensor::ones(HIDDEN, DType::F32, &Device::Cpu)?;
        let weight = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&device, &weight_t)?;
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            let z = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            time("gdn_gated_norm_cached", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached(
                    &device,
                    &x,
                    &z,
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
        let weight = Tensor::ones(HIDDEN, DType::F32, &Device::Cpu)?;
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            time("qwen_rmsnorm_forward", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_qwen_rmsnorm_forward(
                    &device, &x, &weight, 1e-6,
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
        let a_log = upload_bf16_packed(&device, &make_bf16_weight(1, nv)?)?;
        let dt_bias = upload_bf16_packed(&device, &make_bf16_weight(1, nv)?)?;
        for &batch in &batches {
            let a = Tensor::zeros((batch, 1, nv), DType::F32, &Device::Cpu)?;
            let b = Tensor::zeros((batch, 1, nv), DType::F32, &Device::Cpu)?;
            time("gdn_gates_cached", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached(
                    &device,
                    &a,
                    &b,
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
        for &batch in &batches {
            let x = Tensor::zeros((batch, 1, HIDDEN), DType::F32, &Device::Cpu)?;
            time("gdn_in_proj_decode", batch, || {
                kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights(
                    &device, &x, &qkv_buf, &z_buf, &a_buf, &b_buf, HIDDEN, QKV_DIM, Z_DIM, A_DIM, B_DIM,
                )?;
                Ok(())
            })?;
        }
        println!();
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
    let weight_norm = upload_tensor_f32_buffer(
        device,
        &Tensor::ones(HIDDEN, DType::F32, &Device::Cpu)?,
    )?;
    let weight_qknorm = upload_tensor_f32_buffer(
        device,
        &Tensor::ones(head_dim, DType::F32, &Device::Cpu)?,
    )?;
    let out_w = upload_tensor_bf16_packed_buffer(device, &make_bf16_weight(Q_DIM, HIDDEN)?)?;

    // Synthetic RoPE cos/sin tables for 1 position (the new decode token).
    let cos_t = Tensor::from_vec(
        (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect::<Vec<_>>(),
        (1, half_rot),
        &Device::Cpu,
    )?;
    let sin_t = Tensor::from_vec(
        (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect::<Vec<_>>(),
        (1, half_rot),
        &Device::Cpu,
    )?;
    let cos_buf = upload_tensor_f32_buffer(device, &cos_t)?;
    let sin_buf = upload_tensor_f32_buffer(device, &sin_t)?;

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
        let v_buf = VulkanBuffer::create_device_local(
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
            dispatch_add_resident(
                device,
                &x_buf,
                &attn_out,
                &attn_residual,
                batch * HIDDEN,
            )?;
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
        "== full_step_resident_batched (same 11 kernels, recorded into 1 command-buffer + 1 submit) =="
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
    let weight_norm = upload_tensor_f32_buffer(
        device,
        &Tensor::ones(HIDDEN, DType::F32, &Device::Cpu)?,
    )?;
    let weight_qknorm = upload_tensor_f32_buffer(
        device,
        &Tensor::ones(head_dim, DType::F32, &Device::Cpu)?,
    )?;
    let out_w = upload_tensor_bf16_packed_buffer(device, &make_bf16_weight(Q_DIM, HIDDEN)?)?;

    let cos_t = Tensor::from_vec(
        (0..half_rot).map(|i| ((i as f32) * 0.13).cos()).collect::<Vec<_>>(),
        (1, half_rot),
        &Device::Cpu,
    )?;
    let sin_t = Tensor::from_vec(
        (0..half_rot).map(|i| ((i as f32) * 0.13).sin()).collect::<Vec<_>>(),
        (1, half_rot),
        &Device::Cpu,
    )?;
    let cos_buf = upload_tensor_f32_buffer(device, &cos_t)?;
    let sin_buf = upload_tensor_f32_buffer(device, &sin_t)?;

    // Shader paths reused across every iteration.
    let rmsnorm_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let qkv_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/full_attn_qkv_decode_batched_bf16w.comp"
    );
    let rope_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_rope_f32.comp"
    );
    let paged_attn_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let mul_sigmoid_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/vk_mul_sigmoid_gate_f32.comp"
    );
    let linear_decode_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_batched_bf16w.comp"
    );
    let add_shader = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/add.comp");
    let mlp_gate_up_shader = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/mlp_gate_up_decode_batched_bf16w.comp"
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
        let final_out = mk(hidden_bytes)?;
        let scratch = mk((batch * INTERMEDIATE * 4) as u64)?;
        let qkv_combined = mk((batch * (Q_DIM + K_DIM + V_DIM) * 4) as u64)?;
        let q_buf = mk((batch * num_heads * head_dim * 4) as u64)?;
        let q_rot = mk((batch * num_heads * head_dim * 4) as u64)?;
        let k_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let k_rot = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
        let _v_buf = mk((batch * num_kv_heads * head_dim * 4) as u64)?;
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
            // 2) Fused QKV. Always use batched bf16w shader (covers b>=1).
            let total_out = Q_DIM + K_DIM + V_DIM;
            b.record_shader(
                qkv_shader,
                &[
                    qkv_combined.handle(),
                    q_w.handle(),
                    k_w.handle(),
                    v_w.handle(),
                    qkv_combined.handle(),
                ],
                &[
                    HIDDEN as u32,
                    Q_DIM as u32,
                    K_DIM as u32,
                    V_DIM as u32,
                    total_out as u32,
                    batch as u32,
                ],
                Workgroups::OneD((batch * total_out.div_ceil(16)) as u32),
            )?;
            // 3) Per-head QK norm — same rmsnorm shader, rows=batch*heads.
            b.record_shader(
                rmsnorm_shader,
                &[q_buf.handle(), weight_qknorm.handle(), q_buf.handle()],
                &[(batch * num_heads) as u32, head_dim as u32, (1e-6f32).to_bits()],
                Workgroups::OneD((batch * num_heads) as u32),
            )?;
            b.record_shader(
                rmsnorm_shader,
                &[k_buf.handle(), weight_qknorm.handle(), k_buf.handle()],
                &[(batch * num_kv_heads) as u32, head_dim as u32, (1e-6f32).to_bits()],
                Workgroups::OneD((batch * num_kv_heads) as u32),
            )?;
            // 4) RoPE on Q
            b.record_shader(
                rope_shader,
                &[q_buf.handle(), cos_buf.handle(), sin_buf.handle(), q_rot.handle()],
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
                &[k_buf.handle(), cos_buf.handle(), sin_buf.handle(), k_rot.handle()],
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
                &[attn_pre_gate.handle(), gate_buf.handle(), attn_post_gate.handle()],
                &[(batch * num_heads * head_dim) as u32],
                Workgroups::OneD((batch * num_heads * head_dim).div_ceil(256) as u32),
            )?;
            // 7) Attention out_proj
            b.record_shader(
                linear_decode_shader,
                &[attn_post_gate.handle(), out_w.handle(), attn_out.handle()],
                &[Q_DIM as u32, HIDDEN as u32, batch as u32],
                Workgroups::OneD((batch * HIDDEN.div_ceil(32)) as u32),
            )?;
            // 8) Residual: x + attn_out -> attn_residual
            b.record_shader(
                add_shader,
                &[x_buf.handle(), attn_out.handle(), attn_residual.handle()],
                &[(batch * HIDDEN) as u32],
                Workgroups::OneD((batch * HIDDEN).div_ceil(256) as u32),
            )?;
            // 9) Pre-MLP rmsnorm
            b.record_shader(
                rmsnorm_shader,
                &[attn_residual.handle(), weight_norm.handle(), qkv_combined.handle()],
                &[batch as u32, HIDDEN as u32, (1e-6f32).to_bits()],
                Workgroups::OneD(batch as u32),
            )?;
            // 10) MLP gate_up
            b.record_shader(
                mlp_gate_up_shader,
                &[
                    qkv_combined.handle(),
                    gate_w.handle(),
                    up_w.handle(),
                    scratch.handle(),
                ],
                &[HIDDEN as u32, INTERMEDIATE as u32, batch as u32],
                Workgroups::OneD((batch * INTERMEDIATE.div_ceil(128)) as u32),
            )?;
            // 11) MLP down
            b.record_shader(
                linear_decode_shader,
                &[scratch.handle(), down_w.handle(), final_out.handle()],
                &[INTERMEDIATE as u32, HIDDEN as u32, batch as u32],
                Workgroups::OneD((batch * HIDDEN.div_ceil(32)) as u32),
            )?;
            // 12) Final residual
            b.record_shader(
                add_shader,
                &[attn_residual.handle(), final_out.handle(), x_buf.handle()],
                &[(batch * HIDDEN) as u32],
                Workgroups::OneD((batch * HIDDEN).div_ceil(256) as u32),
            )?;
            b.submit_and_wait("full_step_resident_batched")?;
            Ok(())
        })?;
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    run()
}
