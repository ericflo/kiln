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
    }

    Ok(())
}

fn main() -> Result<()> {
    run()
}
