#![cfg(feature = "metal")]
//! Metal SDPA microbenchmark (#1082) — measures `metal_sdpa_last_axis`
//! wall-clock at Qwen3.5-4B attention shapes on Apple Silicon. Decode
//! (q_seq=1, growing k_seq) is the memory-bound / long-context case the
//! flash-tile targets; prefill (q_seq>1, causal) is the compute case.
//! `#[ignore]` — run explicitly:
//!   cargo test -p kiln-tensor --features metal --test metal_sdpa_bench -- --ignored --nocapture

use std::time::Instant;
use kiln_tensor::{ops, DType, Device, Tensor};

fn metal() -> Option<Device> {
    kiln_tensor::primary_metal_companion(0).ok().map(|_| Device::Metal(0))
}

fn pat(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0xDEADBEEF).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            ((s >> 40) as u32 % 256) as f32 / 256.0 - 0.5
        })
        .collect()
}

fn bf16(data: Vec<f32>, shape: &[usize], dev: Device) -> Tensor {
    ops::cast(&Tensor::from_vec_on(dev, data, shape.to_vec()).unwrap(), DType::BF16).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn time_sdpa(
    bs: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    iters: usize,
    dev: Device,
) -> f64 {
    let q = bf16(pat(bs * hq * sq * d, 1), &[bs, hq, sq, d], dev);
    let k = bf16(pat(bs * hkv * sk * d, 2), &[bs, hkv, sk, d], dev);
    let v = bf16(pat(bs * hkv * sk * d, 3), &[bs, hkv, sk, d], dev);
    let scale = 1.0f32 / (d as f32).sqrt();
    // warmup + sync
    let _ = kiln_tensor::metal_sdpa_last_axis(&q, &k, &v, scale, causal)
        .unwrap()
        .to_vec::<half::bf16>()
        .unwrap();
    // Amortize host readback: queue all dispatches, sync once at the end.
    let t = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(kiln_tensor::metal_sdpa_last_axis(&q, &k, &v, scale, causal).unwrap());
    }
    let _ = last.unwrap().to_vec::<half::bf16>().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[test]
#[ignore]
fn bench_sdpa_qwen_shapes() {
    let Some(dev) = metal() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // Qwen3.5-4B-ish attention: Hq=32, Hkv=8 (GQA 4:1), head_dim=128.
    let (hq, hkv, d) = (32usize, 8usize, 128usize);
    println!("\n=== Metal SDPA microbench (metal_sdpa_last_axis, BF16, Hq={hq} Hkv={hkv} D={d}) ===");
    // (metric, label, bs, sq, sk, causal, iters)
    let cases = [
        (
            "decode_sq1_sk128_ms",
            "decode  Sq=1   Sk=128 ",
            1usize,
            1usize,
            128usize,
            false,
            100usize,
        ),
        (
            "decode_sq1_sk512_ms",
            "decode  Sq=1   Sk=512 ",
            1,
            1,
            512,
            false,
            100,
        ),
        (
            "decode_sq1_sk2048_ms",
            "decode  Sq=1   Sk=2048",
            1,
            1,
            2048,
            false,
            50,
        ),
        (
            "decode_sq1_sk4096_ms",
            "decode  Sq=1   Sk=4096",
            1,
            1,
            4096,
            false,
            50,
        ),
        (
            "prefill_sq128_sk128_ms",
            "prefill Sq=128 Sk=128 ",
            1,
            128,
            128,
            true,
            50,
        ),
        (
            "prefill_sq512_sk512_ms",
            "prefill Sq=512 Sk=512 ",
            1,
            512,
            512,
            true,
            20,
        ),
    ];
    for (metric, label, bs, sq, sk, causal, iters) in cases {
        let ms = time_sdpa(bs, hq, hkv, sq, sk, d, causal, iters, dev);
        println!("KILN_LATENCY_METRIC {metric} {ms:.6} ms");
        println!("  {label}   {ms:9.3} ms/iter");
    }
}
