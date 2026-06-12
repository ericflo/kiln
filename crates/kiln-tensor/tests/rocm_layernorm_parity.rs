//! Phase R.5 — the load-bearing wave-size correctness test for LayerNorm.
//!
//! Runs `rocm_layernorm_last_axis` against a CPU reference at row widths that
//! straddle the 32- and 64-lane wavefront boundaries {31,32,33,63,64,65,...}.
//! LayerNorm reduces TWO values (sum + sum-of-squares) per row; a wave64
//! reduction bug compiles cleanly and only manifests numerically at these
//! widths, so a power-of-two-only test would pass a broken kernel. Skips when no
//! ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_layernorm_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 layernorm parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-5, 5) for (row, col).
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// Deterministic per-column weight in ~[0.5, 1.5).
fn wval(c: usize) -> f32 {
    (((c * 37 + 11) % 100) as f32) / 100.0 + 0.5
}

/// Deterministic per-column bias in ~[-0.5, 0.5).
fn bval(c: usize) -> f32 {
    (((c * 53 + 19) % 100) as f32) / 100.0 - 0.5
}

#[test]
fn layernorm_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    // Widths chosen to straddle the 32- and 64-lane boundaries plus a few
    // strided (n_cols > blockDim) cases.
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];
    let n_rows = 6usize;
    let eps = 1e-5f32;

    for &w in &widths {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        let weight: Vec<f32> = (0..w).map(wval).collect();
        let bias: Vec<f32> = (0..w).map(bval).collect();

        // CPU reference: per-row layernorm in F32, single-pass var via
        // E[X^2] - E[X]^2 to mirror the kernel's recipe.
        let mut reference = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            let row = &data[r * w..(r + 1) * w];
            let inv_n = 1.0f32 / (w as f32);
            let mut s = 0.0f32;
            let mut ss = 0.0f32;
            for &x in row {
                s += x;
                ss += x * x;
            }
            let mean = s * inv_n;
            let mut var = ss * inv_n - mean * mean;
            if var < 0.0 {
                var = 0.0;
            }
            let denom = var + eps;
            let inv_std = if denom > 0.0 {
                denom.sqrt().recip()
            } else {
                0.0
            };
            for (c, &x) in row.iter().enumerate() {
                reference.push((x - mean) * inv_std * weight[c] + bias[c]);
            }
        }

        // Device path.
        let x = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on x (w={w}): {e}"));
        let wt = Tensor::from_vec_on(Device::Rocm(0), weight, vec![w])
            .unwrap_or_else(|e| panic!("from_vec_on weight (w={w}): {e}"));
        let bs = Tensor::from_vec_on(Device::Rocm(0), bias, vec![w])
            .unwrap_or_else(|e| panic!("from_vec_on bias (w={w}): {e}"));

        let y = kiln_tensor::rocm_layernorm_last_axis(&x, &wt, &bs, eps)
            .unwrap_or_else(|e| panic!("rocm_layernorm_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-5 + 1e-4 * rf.abs(),
                "layernorm mismatch at width={w} idx={i}: got {g} ref {rf} diff {diff} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }
    }
    eprintln!("layernorm CPU-vs-ROCm parity passed across wavefront-boundary widths {widths:?}");
}
