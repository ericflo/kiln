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

use kiln_tensor::{rocm_is_available, rocm_to_host_copy, Device, Tensor};

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
    let x_g = Tensor::from_vec_on::<f32>(dev, x.clone(), vec![leading, seq, head_dim])
        .expect("alloc x");
    let cos_g = Tensor::from_vec_on::<f32>(dev, cos.clone(), vec![seq, pair_count])
        .expect("alloc cos");
    let sin_g = Tensor::from_vec_on::<f32>(dev, sin.clone(), vec![seq, pair_count])
        .expect("alloc sin");

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
