#![cfg(feature = "rocm")]
//! Parity test for the ROCm rope kernel (Phase R.5).
//!
//! Builds F32 inputs on the GPU, runs `rocm_rope`, copies the result back to
//! the host, and compares against a CPU reference that mirrors the kernel
//! semantics in `csrc/rope.cu` exactly:
//!
//!   even: x[2i] * cos[s,i] - x[2i+1] * sin[s,i]
//!   odd:  x[2i] * sin[s,i] + x[2i+1] * cos[s,i]
//!
//! with head_dim entries beyond rotary_dim passed through unchanged. RoPE is
//! elementwise (one thread per output element), so a couple of shapes suffice
//! to exercise both the full-rotary and partial-rotary (tail pass-through)
//! paths.

use half::bf16;
use kiln_tensor::{Device, Tensor, rocm_is_available, rocm_to_host_copy};

/// CPU reference for rope over a contiguous `[leading, seq, head_dim]` x with
/// `[seq, rotary_dim/2]` cos/sin.
fn cpu_rope_ref(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    leading: usize,
    seq: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Vec<f32> {
    let pair_count = rotary_dim / 2;
    let mut out = x.to_vec();
    for l in 0..leading {
        for s in 0..seq {
            let base = ((l * seq) + s) * head_dim;
            for i in 0..pair_count {
                let a = x[base + 2 * i];
                let b = x[base + 2 * i + 1];
                let c = cos[s * pair_count + i];
                let si = sin[s * pair_count + i];
                out[base + 2 * i] = a * c - b * si;
                out[base + 2 * i + 1] = a * si + b * c;
            }
            // d >= rotary_dim: pass-through (already copied via to_vec()).
        }
    }
    out
}

fn run_case(leading: usize, seq: usize, head_dim: usize, rotary_dim: usize) {
    let pair_count = rotary_dim / 2;
    let n = leading * seq * head_dim;

    // Deterministic pseudo-data.
    let x: Vec<f32> = (0..n)
        .map(|k| ((k as f32) * 0.013 - 0.7).sin() * 1.5)
        .collect();
    let cos: Vec<f32> = (0..seq * pair_count)
        .map(|k| ((k as f32) * 0.021 + 0.3).cos())
        .collect();
    let sin: Vec<f32> = (0..seq * pair_count)
        .map(|k| ((k as f32) * 0.021 + 0.3).sin())
        .collect();

    let dev = Device::Rocm(0);
    let x_g =
        Tensor::from_vec_on::<f32>(dev, x.clone(), vec![leading, seq, head_dim]).expect("alloc x");
    let cos_g =
        Tensor::from_vec_on::<f32>(dev, cos.clone(), vec![seq, pair_count]).expect("alloc cos");
    let sin_g =
        Tensor::from_vec_on::<f32>(dev, sin.clone(), vec![seq, pair_count]).expect("alloc sin");

    let out_g = kiln_tensor::rocm_rope(&x_g, &cos_g, &sin_g, rotary_dim).expect("rocm_rope");
    let out_host = rocm_to_host_copy(&out_g).expect("copy back");
    let got = out_host.to_vec::<f32>().expect("to_vec");

    let want = cpu_rope_ref(&x, &cos, &sin, leading, seq, head_dim, rotary_dim);

    assert_eq!(got.len(), want.len(), "length mismatch");
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = 1e-5 + 1e-4 * w.abs();
        assert!(
            diff <= tol,
            "mismatch at {idx} (l/s/d decode of leading={leading} seq={seq} head_dim={head_dim} \
             rotary_dim={rotary_dim}): got {g} want {w} diff {diff} tol {tol}"
        );
    }
}

// 8 params: flat mirror of the kernel's argument list (consistent with the
// crate's too_many_arguments allowance for kernel-parity signatures).
#[allow(clippy::too_many_arguments)]
fn cpu_rope_split_half_ref(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = x.to_vec();
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let base = (((b * seq) + s) * heads + h) * head_dim;
                let sched = s * half;
                for i in 0..half {
                    let x1 = x[base + i];
                    let x2 = x[base + half + i];
                    let c = cos[sched + i];
                    let si = sin[sched + i];
                    out[base + i] = x1 * c - x2 * si;
                    out[base + half + i] = x1 * si + x2 * c;
                }
            }
        }
    }
    out
}

fn run_split_half_case_f32(
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) {
    let half = rotary_dim / 2;
    let n = batch * seq * heads * head_dim;
    let x: Vec<f32> = (0..n)
        .map(|k| ((k as f32) * 0.007 - 0.5).sin() * 1.25)
        .collect();
    let cos: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.031 + 0.1).cos())
        .collect();
    let sin: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.031 + 0.1).sin())
        .collect();

    let dev = Device::Rocm(0);
    let x_g = Tensor::from_vec_on::<f32>(dev, x.clone(), vec![batch, seq, heads, head_dim])
        .expect("alloc split-half x");
    let cos_g = Tensor::from_vec_on::<f32>(dev, cos.clone(), vec![seq, half]).expect("alloc cos");
    let sin_g = Tensor::from_vec_on::<f32>(dev, sin.clone(), vec![seq, half]).expect("alloc sin");

    let out_g = kiln_tensor::ops::rope_split_half(&x_g, &cos_g, &sin_g, rotary_dim)
        .expect("rocm rope_split_half");
    let got = rocm_to_host_copy(&out_g)
        .expect("copy split-half")
        .to_vec::<f32>()
        .expect("to_vec split-half");
    let want = cpu_rope_split_half_ref(&x, &cos, &sin, batch, seq, heads, head_dim, rotary_dim);

    assert_eq!(got.len(), want.len(), "split-half length mismatch");
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        let tol = 1e-5 + 1e-4 * w.abs();
        assert!(
            diff <= tol,
            "split-half f32 mismatch at {idx}: got {g} want {w} diff {diff} tol {tol}"
        );
    }
}

fn run_split_half_case_bf16(
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) {
    let half = rotary_dim / 2;
    let n = batch * seq * heads * head_dim;
    let x_f32: Vec<f32> = (0..n)
        .map(|k| ((k as f32) * 0.011 - 0.2).cos() * 1.5)
        .collect();
    let x: Vec<bf16> = x_f32.iter().copied().map(bf16::from_f32).collect();
    let x_ref: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let cos: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.019 + 0.4).cos())
        .collect();
    let sin: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.019 + 0.4).sin())
        .collect();

    let dev = Device::Rocm(0);
    let x_g = Tensor::from_vec_on::<bf16>(dev, x, vec![batch, seq, heads, head_dim])
        .expect("alloc split-half bf16 x");
    let cos_g = Tensor::from_vec_on::<f32>(dev, cos.clone(), vec![seq, half]).expect("alloc cos");
    let sin_g = Tensor::from_vec_on::<f32>(dev, sin.clone(), vec![seq, half]).expect("alloc sin");

    let out_g = kiln_tensor::ops::rope_split_half(&x_g, &cos_g, &sin_g, rotary_dim)
        .expect("rocm rope_split_half bf16");
    let got_bf16 = rocm_to_host_copy(&out_g)
        .expect("copy split-half bf16")
        .to_vec::<bf16>()
        .expect("to_vec split-half bf16");
    let got: Vec<f32> = got_bf16.iter().map(|v| v.to_f32()).collect();
    let want_f32 =
        cpu_rope_split_half_ref(&x_ref, &cos, &sin, batch, seq, heads, head_dim, rotary_dim);
    let want: Vec<f32> = want_f32
        .iter()
        .copied()
        .map(|v| bf16::from_f32(v).to_f32())
        .collect();

    assert_eq!(got.len(), want.len(), "split-half bf16 length mismatch");
    for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "split-half bf16 mismatch at {idx}: got {g} want {w}");
    }
}

#[test]
fn rocm_rope_parity_f32() {
    if !rocm_is_available() {
        eprintln!("rocm_rope_parity_f32: ROCm not available, skipping");
        return;
    }

    // Full-rotary: rotary_dim == head_dim.
    run_case(2, 4, 8, 8);
    // Partial-rotary: tail pass-through (Qwen3.5-style rotary_dim < head_dim).
    run_case(3, 5, 16, 8);
    // Single leading row, larger head_dim, partial rotary.
    run_case(1, 7, 64, 32);
    // leading == 1 implied by rank-2 x is exercised via the rank-3 cases above;
    // wider seq with full rotary to stress the cos/sin indexing.
    run_case(2, 16, 32, 32);
}

#[test]
fn rocm_rope_split_half_rank4_parity() {
    if !rocm_is_available() {
        eprintln!("rocm_rope_split_half_rank4_parity: ROCm not available, skipping");
        return;
    }

    run_split_half_case_f32(2, 5, 3, 16, 8);
    run_split_half_case_f32(1, 17, 4, 64, 32);
    run_split_half_case_bf16(1, 19, 4, 256, 64);
}

#[test]
#[ignore = "production-sized ROCm stress test; requires ~0.5 GiB device memory"]
fn rocm_rope_split_half_production_shape_bf16_is_finite() {
    if !rocm_is_available() {
        eprintln!(
            "rocm_rope_split_half_production_shape_bf16_is_finite: ROCm not available, skipping"
        );
        return;
    }

    let batch = 1;
    let seq = 27_113;
    let heads = 16;
    let head_dim = 256;
    let rotary_dim = 128;
    let half = rotary_dim / 2;
    let n = batch * seq * heads * head_dim;

    let x: Vec<bf16> = (0..n)
        .map(|k| {
            let v = (((k % 2048) as f32) * 0.001 - 1.0).sin() * 2.0;
            bf16::from_f32(v)
        })
        .collect();
    let cos: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.00013 + 0.25).cos())
        .collect();
    let sin: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.00013 + 0.25).sin())
        .collect();

    let dev = Device::Rocm(0);
    let x_g = Tensor::from_vec_on::<bf16>(dev, x, vec![batch, seq, heads, head_dim])
        .expect("alloc production split-half bf16 x");
    let cos_g =
        Tensor::from_vec_on::<f32>(dev, cos, vec![seq, half]).expect("alloc production cos");
    let sin_g =
        Tensor::from_vec_on::<f32>(dev, sin, vec![seq, half]).expect("alloc production sin");

    assert!(x_g.all_finite().expect("x finite"));
    assert!(cos_g.all_finite().expect("cos finite"));
    assert!(sin_g.all_finite().expect("sin finite"));

    let out_g = kiln_tensor::ops::rope_split_half(&x_g, &cos_g, &sin_g, rotary_dim)
        .expect("production rocm rope_split_half bf16");
    assert!(
        out_g
            .all_finite()
            .expect("production rope output finite check"),
        "production split-half RoPE produced a non-finite BF16 output"
    );
}

#[test]
#[ignore = "production-sized ROCm KV-shape stress test; requires ~0.25 GiB device memory"]
fn rocm_rope_split_half_qwen_kv_shape_bf16_is_finite_and_copies_tail() {
    if !rocm_is_available() {
        eprintln!(
            "rocm_rope_split_half_qwen_kv_shape_bf16_is_finite_and_copies_tail: ROCm not available, skipping"
        );
        return;
    }

    let batch = 1;
    let seq = 43_035;
    let heads = 4;
    let head_dim = 256;
    let rotary_dim = 64;
    let half = rotary_dim / 2;
    let n = batch * seq * heads * head_dim;

    let x: Vec<bf16> = (0..n)
        .map(|k| {
            let v = (((k % 4096) as f32) * 0.0007 - 1.3).sin() * 1.75;
            bf16::from_f32(v)
        })
        .collect();
    let cos: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.00011 + 0.17).cos())
        .collect();
    let sin: Vec<f32> = (0..seq * half)
        .map(|k| ((k as f32) * 0.00011 + 0.17).sin())
        .collect();

    let dev = Device::Rocm(0);
    let x_g = Tensor::from_vec_on::<bf16>(dev, x.clone(), vec![batch, seq, heads, head_dim])
        .expect("alloc qwen kv split-half bf16 x");
    let cos_g = Tensor::from_vec_on::<f32>(dev, cos, vec![seq, half]).expect("alloc qwen kv cos");
    let sin_g = Tensor::from_vec_on::<f32>(dev, sin, vec![seq, half]).expect("alloc qwen kv sin");

    assert!(x_g.all_finite().expect("x finite"));
    assert!(cos_g.all_finite().expect("cos finite"));
    assert!(sin_g.all_finite().expect("sin finite"));

    let out_g = kiln_tensor::ops::rope_split_half(&x_g, &cos_g, &sin_g, rotary_dim)
        .expect("qwen kv rocm rope_split_half bf16");
    assert!(
        out_g
            .all_finite()
            .expect("qwen kv rope output finite check"),
        "Qwen KV split-half RoPE produced a non-finite BF16 output"
    );

    let got = rocm_to_host_copy(&out_g)
        .expect("copy qwen kv rope output")
        .to_vec::<bf16>()
        .expect("to_vec qwen kv rope output");
    for &(s, h, d) in &[
        (0usize, 2usize, 214usize),
        (0, 0, rotary_dim),
        (seq / 2, 1, 128),
        (seq - 1, heads - 1, head_dim - 1),
    ] {
        let idx = (((s * heads) + h) * head_dim) + d;
        assert_eq!(
            got[idx], x[idx],
            "pass-through tail mismatch at s={s} h={h} d={d} idx={idx}"
        );
    }
}
