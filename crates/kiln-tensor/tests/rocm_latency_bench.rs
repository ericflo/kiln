//! ROCm latency microbenchmark for the backend hardware fixture manifest.
//!
//! Measures hipBLASLt-backed `rocm_matmul_into` wall-clock at Qwen3.5-4B-ish
//! dense GEMM shapes. This is intentionally ignored: run it on the named ROCm
//! hardware fixture and check the emitted metrics into the latency artifacts.
//!
//! Run:
//!   /home/ericflo/.cargo/bin/cargo test -p kiln-tensor --features rocm --test rocm_latency_bench -- --ignored --nocapture
#![cfg(feature = "rocm")]

use std::time::Instant;

use half::bf16;
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping ROCm latency fixture bench");
        true
    } else {
        false
    }
}

fn pat_bf16(n: usize, seed: u64) -> Vec<bf16> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_add(0xDEAD_BEEF)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            bf16::from_f32(((state >> 40) as u32 % 256) as f32 / 256.0 - 0.5)
        })
        .collect()
}

fn rocm_bf16(data: Vec<bf16>, shape: &[usize]) -> Tensor {
    Tensor::from_vec_on(Device::Rocm(0), data, shape.to_vec()).expect("ROCm tensor allocation")
}

fn time_matmul_bf16(m: usize, k: usize, n: usize, iters: usize) -> f64 {
    let a = rocm_bf16(pat_bf16(m * k, 1), &[m, k]);
    let b = rocm_bf16(pat_bf16(k * n, 2), &[k, n]);
    let dst = rocm_bf16(vec![bf16::from_f32(0.0); m * n], &[m, n]);

    kiln_tensor::rocm_matmul_into(&a, &b, &dst).expect("warmup rocm_matmul_into");
    kiln_tensor::rocm_synchronize_compute_stream(0).expect("warmup ROCm sync");

    let start = Instant::now();
    for _ in 0..iters {
        kiln_tensor::rocm_matmul_into(&a, &b, &dst).expect("timed rocm_matmul_into");
    }
    kiln_tensor::rocm_synchronize_compute_stream(0).expect("timed ROCm sync");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let host = kiln_tensor::rocm_to_host_copy(&dst).expect("ROCm result copyback");
    let values = host.to_vec::<bf16>().expect("BF16 result decode");
    assert_eq!(values.len(), m * n);
    elapsed_ms
}

#[test]
#[ignore]
fn bench_rocm_matmul_qwen_shapes() {
    if no_rocm() {
        return;
    }

    // Qwen3.5-4B-ish dense GEMMs: hidden=2560, QKV out=4096,
    // gate||up out=18432, vocab/lm_head out=152064.
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
            "lm_head_m1_2560x152064_ms",
            "lm_head      M=1   2560x152064",
            1,
            2560,
            152064,
            20,
        ),
    ];

    println!("\n=== ROCm matmul microbench (rocm_matmul_into, BF16) ===");
    for (metric, label, m, k, n, iters) in cases {
        let ms = time_matmul_bf16(m, k, n, iters);
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        println!("KILN_LATENCY_METRIC {metric} {ms:.6} ms");
        println!(
            "{metric} {ms:.3} ms   {label:38} {:8.1} GFLOP/s",
            gflop / (ms / 1000.0)
        );
    }
}
