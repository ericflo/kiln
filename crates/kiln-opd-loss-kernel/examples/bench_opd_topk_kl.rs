//! Throughput benchmark for the fused OPD top-K reverse-KL kernel.
//!
//! Run on a CUDA host (A6000+) with:
//!
//! ```
//! KILN_CUDA_ARCHS=86 cargo run --release --example bench_opd_topk_kl \
//!     --features cuda -p kiln-opd-loss-kernel
//! ```
//!
//! Reports per-shape:
//! - Kernel-path tok/s (raw CUDA via cuda_kernel_forward).
//! - Candle-fallback tok/s (the Phase A reference on CUDA storage).
//! - Speedup ratio.
//!
//! Shapes are chosen to mirror §9.7 of the grand plan: production
//! configuration is K=32, H=2560 (Qwen3.5-4B hidden), T ∈ {512, 4096}.
//! K=16 and bf16 variants are also benchmarked.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use kiln_opd_loss_kernel::{
    opd_top_k_reverse_kl_phase_a_per_position, opd_top_k_reverse_kl_phase_b,
    opd_top_k_reverse_kl_phase_b_per_position,
};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0).context("Cuda device 0")?;
    println!("# kiln-opd-loss-kernel throughput bench");
    println!("# device: {:?}", device);
    println!(
        "# header: shape  K  dtype  iters  kernel_ms  candle_ms  speedup_x  kernel_tok_s"
    );

    // Production-shaped sweeps. Hidden=2560 matches Qwen3.5-4B (the kiln
    // student); vocab matches the actual Qwen tokenizer (~152K but we
    // use 32K to fit comfortably on A6000 within the bench).
    let configs = [
        (256usize, 2560usize, 32_000usize, 32usize, DType::F32),
        (512, 2560, 32_000, 32, DType::F32),
        (1024, 2560, 32_000, 32, DType::F32),
        (4096, 2560, 32_000, 32, DType::F32),
        (1024, 2560, 32_000, 16, DType::F32),
        (1024, 2560, 32_000, 32, DType::BF16),
        (4096, 2560, 32_000, 32, DType::BF16),
    ];

    for &(seq_len, hidden_size, vocab_size, top_k, dtype) in &configs {
        bench_one(&device, seq_len, hidden_size, vocab_size, top_k, dtype)?;
    }
    Ok(())
}

fn bench_one(
    device: &Device,
    seq_len: usize,
    hidden_size: usize,
    vocab_size: usize,
    top_k: usize,
    dtype: DType,
) -> Result<()> {
    // Construct deterministic inputs.
    let hidden_vec: Vec<f32> = (0..(seq_len * hidden_size))
        .map(|i| (i as f32 * 0.0013).sin() * 0.3)
        .collect();
    let hidden_f32 = Tensor::from_vec(hidden_vec, (1, seq_len, hidden_size), device)?;
    let head_vec: Vec<f32> = (0..(hidden_size * vocab_size))
        .map(|i| ((i as f32 + 7.0) * 0.0007).cos() * 0.2)
        .collect();
    let head_f32 = Tensor::from_vec(head_vec, (hidden_size, vocab_size), device)?;
    let hidden = hidden_f32.to_dtype(dtype)?;
    let head_t = head_f32.to_dtype(dtype)?;

    // Every position active (worst case).
    let mask = vec![true; seq_len];
    let mut indices: Vec<u32> = Vec::with_capacity(seq_len * top_k);
    for t in 0..seq_len {
        let mut row: Vec<u32> = (0..top_k as u32)
            .map(|k| ((t * 17 + (k as usize) * 31 + 5) % vocab_size) as u32)
            .collect();
        let mut seen = std::collections::HashSet::new();
        for k in 0..top_k {
            while !seen.insert(row[k]) {
                row[k] = (row[k] + 1) % vocab_size as u32;
            }
        }
        indices.extend_from_slice(&row);
    }
    let mut logprobs: Vec<f32> = Vec::with_capacity(seq_len * top_k);
    for t in 0..seq_len {
        for k in 0..top_k {
            logprobs.push(-((t as f32 + 1.0).ln() + (k as f32) * 0.3));
        }
    }

    let iters = if seq_len >= 1024 { 5 } else { 20 };
    // Warm up once (CUDA graph capture, autotune).
    let _ = opd_top_k_reverse_kl_phase_b_per_position(
        &hidden, &head_t, &indices, &logprobs, &mask, top_k, device, 4096,
    )?;
    device.synchronize()?;

    // Kernel timing.
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = opd_top_k_reverse_kl_phase_b_per_position(
            &hidden, &head_t, &indices, &logprobs, &mask, top_k, device, 4096,
        )?;
    }
    device.synchronize()?;
    let kernel_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Phase A (candle path) timing for the same shapes.
    let _ = opd_top_k_reverse_kl_phase_a_per_position(
        &hidden, &head_t, &indices, &logprobs, &mask, top_k, device,
    )?;
    device.synchronize()?;
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = opd_top_k_reverse_kl_phase_a_per_position(
            &hidden, &head_t, &indices, &logprobs, &mask, top_k, device,
        )?;
    }
    device.synchronize()?;
    let candle_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let speedup = candle_ms / kernel_ms;
    let kernel_tok_s = (seq_len as f64) / (kernel_ms / 1000.0);
    println!(
        "FWD  T={seq_len:5}  H={hidden_size}  V={vocab_size}  K={top_k:2}  {:?}  iters={iters:3}  kernel={kernel_ms:7.3}ms  candle={candle_ms:7.3}ms  {speedup:5.2}x  {kernel_tok_s:9.0} tok/s",
        dtype
    );

    // ----- Backward bench -----
    //
    // To time the backward pass we need an autograd graph rooted at a
    // `Var` holding `hidden`. Building and tearing the graph is a tiny
    // overhead compared to the actual matmul; we time the whole
    // forward+backward together since that's the trainer's hot loop.
    let hidden_var = Var::from_tensor(&hidden)?;
    // Warm up.
    {
        let loss = opd_top_k_reverse_kl_phase_b(
            hidden_var.as_tensor(),
            &head_t,
            &indices,
            &logprobs,
            &mask,
            top_k,
            device,
            4096,
        )?;
        let _grads = loss.backward()?;
    }
    device.synchronize()?;

    // Kernel bwd path (default: kernel ON).
    let t2 = Instant::now();
    for _ in 0..iters {
        let loss = opd_top_k_reverse_kl_phase_b(
            hidden_var.as_tensor(),
            &head_t,
            &indices,
            &logprobs,
            &mask,
            top_k,
            device,
            4096,
        )?;
        let _grads = loss.backward()?;
    }
    device.synchronize()?;
    let bwd_kernel_ms = t2.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Candle bwd path (force kill switch).
    // SAFETY: env mutation is single-threaded for the bench.
    unsafe { std::env::set_var("KILN_DISABLE_OPD_LOSS_KERNEL", "1"); }
    {
        let loss = opd_top_k_reverse_kl_phase_b(
            hidden_var.as_tensor(),
            &head_t,
            &indices,
            &logprobs,
            &mask,
            top_k,
            device,
            4096,
        )?;
        let _grads = loss.backward()?;
    }
    device.synchronize()?;
    let t3 = Instant::now();
    for _ in 0..iters {
        let loss = opd_top_k_reverse_kl_phase_b(
            hidden_var.as_tensor(),
            &head_t,
            &indices,
            &logprobs,
            &mask,
            top_k,
            device,
            4096,
        )?;
        let _grads = loss.backward()?;
    }
    device.synchronize()?;
    let bwd_candle_ms = t3.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    unsafe { std::env::remove_var("KILN_DISABLE_OPD_LOSS_KERNEL"); }

    let bwd_speedup = bwd_candle_ms / bwd_kernel_ms;
    let bwd_tok_s = (seq_len as f64) / (bwd_kernel_ms / 1000.0);
    println!(
        "FWD+BWD T={seq_len:5}  H={hidden_size}  V={vocab_size}  K={top_k:2}  {:?}  iters={iters:3}  kernel={bwd_kernel_ms:7.3}ms  candle={bwd_candle_ms:7.3}ms  {bwd_speedup:5.2}x  {bwd_tok_s:9.0} tok/s",
        dtype
    );
    Ok(())
}
