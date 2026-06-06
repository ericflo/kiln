#![cfg(feature = "metal")]
//! Metal matmul microbenchmark (#1082) — measures `ops::matmul` wall-clock
//! at Qwen3.5-4B shapes on Apple Silicon. Before the kiln matrix-core GEMM
//! lands, `matmul` host-falls-back to the CPU; this quantifies that and is
//! the baseline the kiln GEMM must beat. `#[ignore]` — run explicitly:
//!   cargo test -p kiln-tensor --features metal --test metal_matmul_bench -- --ignored --nocapture

use std::time::Instant;
use kiln_tensor::{ops, DType, Device, Tensor};

fn metal() -> Option<Device> {
    kiln_tensor::primary_metal_companion(0).ok().map(|_| Device::Metal(0))
}

fn pat(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| { s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E37_79B9_7F4A_7C15); ((s >> 40) as u32 % 256) as f32 / 256.0 - 0.5 }).collect()
}

fn bf16_metal(data: &[f32], shape: &[usize], dev: Device) -> Tensor {
    ops::cast(&Tensor::from_vec_on(dev, data.to_vec(), shape.to_vec()).unwrap(), DType::BF16).unwrap()
}

fn time_matmul(m: usize, k: usize, n: usize, iters: usize, dev: Device) -> f64 {
    let a = bf16_metal(&pat(m * k, 1), &[m, k], dev);
    let b = bf16_metal(&pat(k * n, 2), &[k, n], dev);
    // warmup (+ sync)
    let _ = ops::matmul(&a, &b).unwrap().to_vec::<half::bf16>().unwrap();
    // Amortize host readback/sync over  dispatches: queue all kernels,
    // sync once at the end (Metal retains the output buffers per command-buffer,
    // so dropped intermediates are safe). Measures GPU kernel time, not the
    // per-call debug-mode host gather.
    let t = Instant::now();
    let mut last = None;
    for _ in 0..iters { last = Some(ops::matmul(&a, &b).unwrap()); }
    let _ = last.unwrap().to_vec::<half::bf16>().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[test]
#[ignore]
fn bench_matmul_qwen_shapes() {
    let Some(dev) = metal() else { eprintln!("no Metal device; skipping"); return; };
    // (metric, label, M, K, N, iters) — small iters for the (slow) CPU-fallback baseline.
    let cases = [
        (
            "decode_qkv_m1_2560x4096_ms",
            "decode QKV   M=1   2560x4096",
            1usize,
            2560usize,
            4096usize,
            50usize,
        ),
        (
            "prefill_qkv_m256_2560x4096_ms",
            "prefill QKV  M=256 2560x4096",
            256,
            2560,
            4096,
            30,
        ),
        (
            "prefill_gate_up_m256_2560x18432_ms",
            "prefill gate||up M=256 2560x18432",
            256,
            2560,
            18432,
            20,
        ),
        (
            "prefill_qkv_m512_2560x4096_ms",
            "prefill QKV  M=512 2560x4096",
            512,
            2560,
            4096,
            20,
        ),
        (
            "lm_head_m1_2560x152064_ms",
            "lm_head      M=1   2560x152064",
            1,
            2560,
            152064,
            20,
        ),
    ];
    println!("\n=== Metal matmul microbench (ops::matmul, BF16) ===");
    for (metric, label, m, k, n, iters) in cases {
        let ms = time_matmul(m, k, n, iters, dev);
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        println!("KILN_LATENCY_METRIC {metric} {ms:.6} ms");
        println!("{label:38} {ms:9.3} ms   {:8.1} GFLOP/s", gflop / (ms / 1000.0));
    }
}
