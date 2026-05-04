//! Kernel-only micro-benchmark for the fused GDN gates kernel.
//!
//! The full kiln-bench requires the Qwen3.5-4B checkpoint. This runs
//! kernel-level timing only — fused kernel vs the candle-op reference
//! chain — at realistic decode and prefill shapes.
//!
//! Decode shape: [1, 1, 32]   (B=1, T=1, nv=32 for Qwen3.5-4B GDN)
//! Prefill shape: [1, 64, 32] (B=1, T=64, nv=32)
//!
//! Honest disclosure: this measures the kernel itself, not E2E tokens/sec.
//! The profile in PR #156 attributed ~18.2% of decode wallclock to
//! `kiln/gdn/gates`; this micro-bench bounds the upper limit of what a
//! fused kernel can win back. The real E2E gain is always smaller
//! because of launch overhead and Amdahl.
//!
//! Run with:
//!   cargo run --release --example gates_bench -p kiln-gdn-kernel

use candle_core::{DType, Device, Tensor};
use kiln_gdn_kernel::{gdn_gates, gdn_gates_supports};
use std::time::Instant;

// candle_nn isn't imported — we hand-roll sigmoid below because the candle_nn
// sigmoid has no CUDA backend in this version.

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((s >> 33) as u32) & 0x7fff_ffff;
            ((bits as f32 / (i32::MAX as f32)) - 0.5) * scale
        })
        .collect()
}

/// Softplus reimplementation matching `kiln-model::forward::softplus` —
/// the exact reference path in Step 6 today.
fn softplus(x: &Tensor) -> anyhow::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let relu_x = x.maximum(&zeros)?;
    let neg_x = x.neg()?;
    let relu_neg_x = neg_x.maximum(&zeros)?;
    let abs_x = (relu_x.clone() + relu_neg_x)?;
    let neg_abs = abs_x.neg()?;
    let log_term = (neg_abs.exp()? + 1.0)?.log()?;
    Ok((relu_x + log_term)?)
}

/// Candle-op reference path: the exact Step-6 chain used in
/// `kiln-model::forward::gated_deltanet_forward` today.
fn candle_chain(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    // beta = sigmoid(b) — stays in bf16. candle_nn::ops::sigmoid has no
    // CUDA kernel, so match forward.rs `cuda_sigmoid`: 1/(1+exp(-x)).
    let neg = b.neg()?;
    let exp_neg = neg.exp()?;
    let one_plus = (exp_neg + 1.0)?;
    let beta = one_plus.recip()?;

    // g = -exp(A_log) * softplus(a + dt_bias), F32 intermediates
    let dt_bias_f = dt_bias.to_dtype(DType::F32)?;
    let a_log_f = a_log.to_dtype(DType::F32)?;
    let a_f32 = a.to_dtype(DType::F32)?;
    let dt_bias_b = dt_bias_f.broadcast_as(a_f32.shape())?;
    let a_biased = (&a_f32 + &dt_bias_b)?;
    let sp = softplus(&a_biased)?;
    let neg_decay = a_log_f.exp()?.neg()?;
    let neg_decay_b = neg_decay.broadcast_as(sp.shape())?;
    let g = (&sp * &neg_decay_b)?;
    let g = g.to_dtype(DType::BF16)?;

    Ok((beta, g))
}

/// Force a CUDA sync by materialising a scalar back to the host.
fn cuda_sync(t: &Tensor) {
    let _ = t.to_device(&Device::Cpu);
}

fn time_path<F>(label: &str, iters: usize, warmup: usize, probe: &Tensor, mut f: F) -> f64
where
    F: FnMut() -> anyhow::Result<Tensor>,
{
    for _ in 0..warmup {
        let out = f().expect("warmup");
        cuda_sync(&out);
    }

    let t0 = Instant::now();
    let mut last: Option<Tensor> = None;
    for _ in 0..iters {
        last = Some(f().expect("timed iter"));
    }
    // Sync on the final output to ensure all enqueued work completes.
    if let Some(t) = last {
        cuda_sync(&t);
    } else {
        cuda_sync(probe);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_iter_us = (elapsed / iters as f64) * 1e6;
    println!("  {label:<30} {per_iter_us:>9.2} µs/iter ({iters} iters)");
    per_iter_us
}

fn run_shape(device: &Device, b: usize, t: usize, nv: usize, label: &str) -> anyhow::Result<()> {
    let rows = b * t;
    let a_host = fill(0xA1 ^ (rows * nv) as u64, rows * nv, 0.5);
    let b_host = fill(0xB2 ^ (rows * nv) as u64, rows * nv, 0.5);
    let a_log_host = fill(0xA_106, nv, 0.2);
    let dt_bias_host = fill(0xDE_B1A5, nv, 0.2);

    let mk = |v: &[f32], shape: &[usize]| -> anyhow::Result<Tensor> {
        Ok(Tensor::from_slice(v, shape, &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(device)?)
    };

    let a = mk(&a_host, &[b, t, nv])?;
    let b_t = mk(&b_host, &[b, t, nv])?;
    let a_log = mk(&a_log_host, &[nv])?;
    let dt_bias = mk(&dt_bias_host, &[nv])?;

    assert!(
        gdn_gates_supports(&a, &b_t, &a_log, &dt_bias),
        "{label} envelope check failed"
    );

    println!("\n== {label}  shape=[{b},{t},{nv}]  rows={rows} ==");

    let fused_us = time_path("fused kiln_gdn_gates_bf16", 1000, 50, &a, || {
        let (beta, g) = gdn_gates(&a, &b_t, &a_log, &dt_bias)?;
        // Force beta + g to materialise — return g so the downstream sync
        // touches the launched kernel's output.
        let _ = beta;
        Ok(g)
    });
    let candle_us = time_path("candle reference chain", 1000, 50, &a, || {
        let (beta, g) = candle_chain(&a, &b_t, &a_log, &dt_bias)?;
        let _ = beta;
        Ok(g)
    });

    let speedup = candle_us / fused_us;
    let saved = candle_us - fused_us;
    println!("  -> speedup fused vs candle: {speedup:.2}x  (saved {saved:.2} µs/iter)");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("No CUDA device available: {e}. Skipping micro-bench.");
            return Ok(());
        }
    };

    println!("kiln-gdn-kernel fused GDN gates micro-bench");
    println!("-----------------------------------------------");
    println!("Kernel: beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)");
    println!("Warmup: 50 iters   Timed: 1000 iters");

    // Decode: the hot path — B=1, T=1, per-layer GDN gate chain.
    run_shape(&device, 1, 1, 32, "decode / Qwen3.5-4B")?;

    // Prefill: moderate T, same nv.
    run_shape(&device, 1, 64, 32, "prefill T=64")?;
    run_shape(&device, 1, 128, 32, "prefill T=128")?;

    // Larger head count envelope check.
    run_shape(&device, 1, 1, 128, "decode / nv=128")?;

    Ok(())
}
