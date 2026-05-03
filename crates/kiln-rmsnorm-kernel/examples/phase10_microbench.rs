//! Phase 10 §1 — RMSNorm forward+backward microbench.
//!
//! Compares the fused custom-op path (default) against the candle op-chain
//! path (the pre-Phase-10 baseline used when KILN_DISABLE_RMSNORM_BACKWARD=1)
//! at typical Qwen3.5-4B training shapes.
//!
//! Times forward+sum+backward, median of N iterations after warmup, on CUDA.
//! No training framework, no model load — just the kernel.
//!
//! Run:
//!   cargo run --release -p kiln-rmsnorm-kernel --example phase10_microbench

use candle_core::Result;
use candle_core::{DType, Device, Tensor, Var};

fn rms_norm_candle(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms_inv = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rms_inv)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + w_f32)?;
    let out = normed.broadcast_mul(&w_plus_one)?;
    Ok(out.to_dtype(x.dtype())?)
}

fn time_fwd_bwd(
    x: &Var,
    w: &Var,
    fused: bool,
    eps: f32,
    iters: usize,
    device: &Device,
) -> Result<Vec<f64>> {
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.synchronize()?;
        let t0 = std::time::Instant::now();
        let xt = x.as_tensor();
        let wt = w.as_tensor();
        let y = if fused {
            kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd(xt, wt, eps)?
        } else {
            rms_norm_candle(xt, wt, eps as f64)?
        };
        let loss = y.to_dtype(DType::F32)?.sum_all()?;
        let _grads = loss.backward()?;
        device.synchronize()?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(times)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn run_shape(label: &str, b: usize, t: usize, h: usize, device: &Device) -> Result<()> {
    let dtype = DType::BF16;
    // Use candle Var (autograd-tracked) for x and w.
    let x_init = Tensor::randn(0.0_f32, 1.0, (b, t, h), device)?.to_dtype(dtype)?;
    let w_init = Tensor::randn(0.0_f32, 0.02, (h,), device)?.to_dtype(dtype)?;
    let x = Var::from_tensor(&x_init)?;
    let w = Var::from_tensor(&w_init)?;
    let eps = 1e-6_f32;

    // Warmup
    let _ = time_fwd_bwd(&x, &w, true, eps, 5, device)?;
    let _ = time_fwd_bwd(&x, &w, false, eps, 5, device)?;

    // Measure
    let iters = 25;
    let fused_times = time_fwd_bwd(&x, &w, true, eps, iters, device)?;
    let candle_times = time_fwd_bwd(&x, &w, false, eps, iters, device)?;

    let fused_med = median(fused_times.clone());
    let candle_med = median(candle_times.clone());
    let speedup = candle_med / fused_med;

    println!(
        "| {:18} | {} | {} | {} | {:>8.3} | {:>8.3} | {:>6.3}× |",
        label, b, t, h, candle_med, fused_med, speedup
    );
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;

    println!();
    println!("Phase 10 §1 — RMSNorm fwd+sum+backward microbench (BF16, A6000)");
    println!("Median of 25 iters, 5 warmup. Same x/w Var both arms.");
    println!();
    println!("| shape              | B |    T |    H | candle ms | fused ms | speedup |");
    println!("|:-------------------|--:|-----:|-----:|----------:|---------:|--------:|");

    let h = 2560; // Qwen3.5-4B hidden
    run_shape("decode", 1, 1, h, &device)?;
    run_shape("training T=512", 1, 512, h, &device)?;
    run_shape("training T=2048", 1, 2048, h, &device)?;
    run_shape("training T=4096", 1, 4096, h, &device)?;
    run_shape("training T=8192", 1, 8192, h, &device)?;

    println!();
    Ok(())
}
