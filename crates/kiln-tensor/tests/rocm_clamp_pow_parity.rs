//! Phase R.5 — CPU-vs-ROCm parity for the clamp/pow scalar-unary kernel.
//!
//! `clamp_pow.cu` is elementwise (one thread per element, no cross-lane
//! reductions), so it has no wave-size hazard — a couple of shapes across both
//! op kinds is sufficient coverage. Compares `rocm_clamp_pow` against a CPU
//! reference (op in F32) at f32 rtol 1e-4 / atol 1e-5. Skips when no ROCm device
//! is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_clamp_pow_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 clamp_pow parity test");
        true
    } else {
        false
    }
}

// Op kinds — match `KIND_CLAMP` / `KIND_POW` in clamp_pow.cu.
const KIND_CLAMP: i32 = 0;
const KIND_POW: i32 = 1;

/// CPU reference for `pow`, mirroring `apply_pow` in clamp_pow.cu: integer
/// exponents go through repeated multiplication (so negative bases are well
/// defined, matching Rust's `f32::powf`); non-integer exponents use `powf`.
fn ref_pow(x: f32, p: f32) -> f32 {
    let p_int = p as i32;
    if p == p_int as f32 {
        if p_int == 0 {
            return 1.0;
        }
        let e = p_int.abs();
        let mut r = 1.0f32;
        for _ in 0..e {
            r *= x;
        }
        if p_int < 0 {
            1.0 / r
        } else {
            r
        }
    } else {
        x.powf(p)
    }
}

/// CPU reference for `clamp` = min(max(x, lo), hi).
fn ref_clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

/// Deterministic value in ~[-5, 5).
fn val(i: usize) -> f32 {
    (((i * 37 + 11) % 1000) as f32) / 100.0 - 5.0
}

fn check(got: &[f32], reference: &[f32], label: &str) {
    assert_eq!(got.len(), reference.len(), "len ({label})");
    for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
        // NaN-aware: the kernel and reference must agree on NaN-ness.
        if rf.is_nan() {
            assert!(g.is_nan(), "{label} idx={i}: got {g} ref NaN");
            continue;
        }
        let diff = (g - rf).abs();
        assert!(
            diff <= 1e-5 + 1e-4 * rf.abs(),
            "clamp_pow mismatch {label} idx={i}: got {g} ref {rf} diff {diff}"
        );
    }
}

#[test]
fn clamp_pow_parity() {
    if no_rocm() {
        return;
    }
    let shapes: [Vec<usize>; 3] = [vec![1], vec![257], vec![5, 33]];

    for shape in &shapes {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(val).collect();

        // --- clamp(x, lo=-2.0, hi=2.0) ---
        {
            let lo = -2.0f32;
            let hi = 2.0f32;
            let reference: Vec<f32> = data.iter().map(|&x| ref_clamp(x, lo, hi)).collect();
            let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), shape.clone())
                .unwrap_or_else(|e| panic!("from_vec_on clamp (shape={shape:?}): {e}"));
            let y = kiln_tensor::rocm_clamp_pow(&t, KIND_CLAMP, lo, hi)
                .unwrap_or_else(|e| panic!("rocm_clamp_pow clamp (shape={shape:?}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&y)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy clamp (shape={shape:?}): {e}"));
            let got = host.to_vec::<f32>().expect("to_vec");
            check(&got, &reference, &format!("clamp shape={shape:?}"));
        }

        // --- pow with several exponents (integer + fractional) ---
        // For the fractional exponent (0.5) some inputs are negative -> powf
        // yields NaN on both sides; the NaN-aware check handles that.
        for &p in &[2.0f32, 3.0, 0.5, -1.0] {
            let reference: Vec<f32> = data.iter().map(|&x| ref_pow(x, p)).collect();
            let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), shape.clone())
                .unwrap_or_else(|e| panic!("from_vec_on pow (p={p}, shape={shape:?}): {e}"));
            // `b` is ignored by the POW kind; pass 0.0 for clarity.
            let y = kiln_tensor::rocm_clamp_pow(&t, KIND_POW, p, 0.0)
                .unwrap_or_else(|e| panic!("rocm_clamp_pow pow (p={p}, shape={shape:?}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&y)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy pow (p={p}, shape={shape:?}): {e}"));
            let got = host.to_vec::<f32>().expect("to_vec");
            check(&got, &reference, &format!("pow p={p} shape={shape:?}"));
        }
    }
    eprintln!("clamp_pow CPU-vs-ROCm parity passed across both kinds and shapes {shapes:?}");
}
