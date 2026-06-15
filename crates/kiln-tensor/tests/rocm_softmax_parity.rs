//! Phase R.5 — the load-bearing wave-size correctness test.
//!
//! Runs `rocm_softmax_last_axis` against a CPU reference at row widths that
//! straddle the 32- and 64-lane wavefront boundaries {31,32,33,63,64,65,...}.
//! A wave64 reduction bug (the #1 ROCm hazard) compiles cleanly and only
//! manifests numerically at these widths — a power-of-two-only test would pass
//! a broken kernel. Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_softmax_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 softmax parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-5, 5) for (row, col).
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

#[test]
fn softmax_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    // Widths chosen to straddle the 32- and 64-lane boundaries plus a few
    // strided (n_cols > blockDim) cases.
    let widths = [
        1usize, 2, 7, 16, 31, 32, 33, 47, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024,
        1025, 2048,
    ];
    let n_rows = 6usize;

    for &w in &widths {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }

        // CPU reference: numerically-stable softmax per row.
        let mut reference = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            let row = &data[r * w..(r + 1) * w];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            for e in exps {
                reference.push(e / s);
            }
        }

        // Device path.
        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = kiln_tensor::rocm_softmax_last_axis(&t)
            .unwrap_or_else(|e| panic!("rocm_softmax_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-5 + 1e-5 * rf.abs(),
                "softmax mismatch at width={w} idx={i}: got {g} ref {rf} diff {diff} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }
        // Each row must normalize to 1.
        for r in 0..n_rows {
            let s: f32 = got[r * w..(r + 1) * w].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-4,
                "row {r} of width {w} sums to {s}, not 1"
            );
        }
    }
    eprintln!("softmax CPU-vs-ROCm parity passed across wavefront-boundary widths {widths:?}");
}

#[test]
fn softmax_handles_large_causal_neg_inf_rows() {
    if no_rocm() {
        return;
    }

    // Match the long-context SDPA shape class used by Qwen3.5-4B SFT:
    // thousands of KV columns, one finite causal prefix per row, and a
    // -inf suffix that must softmax to exactly zero rather than NaN.
    let width = 7552usize;
    let n_rows = 64usize;
    let mut scores = Vec::with_capacity(n_rows * width);
    let mut mask = Vec::with_capacity(n_rows * width);
    let mut allowed_by_row = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let allowed = ((r * 113) % width).max(1);
        allowed_by_row.push(allowed);
        for c in 0..width {
            scores.push((((r * 131 + c * 17 + 7) % 2000) as f32) / 100.0 - 10.0);
            mask.push(if c < allowed { 0.0 } else { f32::NEG_INFINITY });
        }
    }

    let scores_t =
        Tensor::from_vec_on(Device::Rocm(0), scores, vec![n_rows, width]).expect("scores to ROCm");
    let mask_t =
        Tensor::from_vec_on(Device::Rocm(0), mask, vec![n_rows, width]).expect("mask to ROCm");
    let masked = scores_t.broadcast_add(&mask_t).expect("apply causal mask");
    let y = kiln_tensor::rocm_softmax_last_axis(&masked).expect("rocm softmax");
    let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy");
    let got = host.to_vec::<f32>().expect("to_vec");

    for r in 0..n_rows {
        let row = &got[r * width..(r + 1) * width];
        let allowed = allowed_by_row[r];
        let sum: f32 = row.iter().sum();
        assert!(
            sum.is_finite() && (sum - 1.0).abs() <= 2e-4,
            "row {r} sum={sum} allowed={allowed}"
        );
        for (c, &v) in row.iter().enumerate() {
            assert!(v.is_finite(), "row {r} col {c} is non-finite: {v}");
            if c >= allowed {
                assert_eq!(v, 0.0, "masked row {r} col {c} should be zero");
            }
        }
    }
}
