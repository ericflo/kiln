//! Phase R.5 — wave-size correctness test for `rocm_rmsnorm_last_axis`.
//!
//! Runs the ROCm rmsnorm kernel against a CPU reference at row widths that
//! straddle the 32- and 64-lane wavefront boundaries {31,32,33,63,64,65,...}.
//! rmsnorm's per-row sum-of-squares is a cross-lane reduction — the #1 ROCm
//! hazard. A wave64 reduction bug compiles cleanly and only manifests
//! numerically at these widths, so a power-of-two-only test would pass a broken
//! kernel. Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_rmsnorm_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 rmsnorm parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-5, 5) for (row, col).
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// Deterministic pseudo-random weight in ~[0.5, 1.5) for col c.
fn wval(c: usize) -> f32 {
    (((c * 37 + 11) % 1000) as f32) / 1000.0 + 0.5
}

#[test]
fn rmsnorm_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let eps = 1e-6f32;
    // Widths straddle the 32- and 64-lane wavefront boundaries plus a few
    // strided (n_cols > blockDim) cases — exactly where a wave64 bug shows up.
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];
    let n_rows = 6usize;

    for &w in &widths {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        let weight: Vec<f32> = (0..w).map(wval).collect();

        // CPU reference: per-row RMSNorm with F32 accumulation.
        let mut reference = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            let row = &data[r * w..(r + 1) * w];
            let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
            let mean_sq = sum_sq / w as f32;
            let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
            for (c, &x) in row.iter().enumerate() {
                reference.push(x * inv_rms * weight[c]);
            }
        }

        // Device path.
        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on x (w={w}): {e}"));
        let wt = Tensor::from_vec_on(Device::Rocm(0), weight, vec![w])
            .unwrap_or_else(|e| panic!("from_vec_on weight (w={w}): {e}"));
        let y = kiln_tensor::rocm_rmsnorm_last_axis(&t, &wt, eps)
            .unwrap_or_else(|e| panic!("rocm_rmsnorm_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-5 + 1e-4 * rf.abs(),
                "rmsnorm mismatch at width={w} idx={i}: got {g} ref {rf} diff {diff} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }
    }
    eprintln!(
        "rmsnorm CPU-vs-ROCm parity passed across wavefront-boundary widths {widths:?}"
    );
}
