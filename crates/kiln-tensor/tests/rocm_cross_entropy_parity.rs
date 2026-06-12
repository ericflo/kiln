//! Phase R.5 — wave-size correctness test for the cross_entropy kernel.
//!
//! Runs `rocm_cross_entropy_loss` against a CPU reference at vocab widths that
//! straddle the 32- and 64-lane wavefront boundaries {1,7,31,32,33,63,64,65,...}.
//! A wave64 reduction bug (the #1 ROCm hazard) compiles cleanly and only
//! manifests numerically at these widths — a power-of-two-only test would pass
//! a broken kernel. Cross-entropy's per-row max + log-sum-exp + the finalize
//! batch-sum are all block reductions, so every one of these widths exercises
//! the shared-memory tree path. Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_cross_entropy_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 cross_entropy parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-5, 5) for (row, col).
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// CPU reference: mean over rows of (log_sum_exp - target_logit), F32
/// accumulation, numerically stable.
fn cpu_cross_entropy(logits: &[f32], targets: &[i64], n_rows: usize, n_cols: usize) -> f32 {
    let mut total = 0.0f32;
    for r in 0..n_rows {
        let row = &logits[r * n_cols..(r + 1) * n_cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - m).exp()).sum();
        let lse = m + sum.ln();
        let t = targets[r] as usize;
        total += lse - row[t];
    }
    total / n_rows as f32
}

#[test]
fn cross_entropy_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    // Vocab widths straddling the 32- and 64-lane boundaries plus a few strided
    // (n_cols > blockDim) cases.
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];
    let n_rows = 5usize;

    for &w in &widths {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        // Valid in-range targets, varied per row.
        let targets: Vec<i64> = (0..n_rows).map(|r| ((r * 3 + 1) % w) as i64).collect();

        let reference = cpu_cross_entropy(&data, &targets, n_rows, w);

        // Device path.
        let logits = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on logits (w={w}): {e}"));
        let tgt = Tensor::from_vec_on(Device::Rocm(0), targets.clone(), vec![n_rows])
            .unwrap_or_else(|e| panic!("from_vec_on targets (w={w}): {e}"));

        let loss = kiln_tensor::rocm_cross_entropy_loss(&logits, &tgt)
            .unwrap_or_else(|e| panic!("rocm_cross_entropy_loss (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&loss)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), 1, "scalar output (w={w})");
        let diff = (got[0] - reference).abs();
        assert!(
            diff <= 1e-5 + 1e-4 * reference.abs(),
            "cross_entropy mismatch at width={w}: got {} ref {reference} diff {diff} \
             (a wave64 reduction bug shows up exactly here)",
            got[0]
        );
    }
    eprintln!(
        "cross_entropy CPU-vs-ROCm parity passed across wavefront-boundary widths {widths:?}"
    );
}

/// U32 targets path + a batch large enough to exercise the finalize
/// block-reduce across the 64-lane boundary.
#[test]
fn cross_entropy_parity_u32_targets_and_large_batch() {
    if no_rocm() {
        return;
    }
    let n_rows = 130usize; // > 64 and > 128 so the finalize reduces a full block
    let w = 257usize;

    let mut data = Vec::with_capacity(n_rows * w);
    for r in 0..n_rows {
        for c in 0..w {
            data.push(val(r, c));
        }
    }
    let targets_u32: Vec<u32> = (0..n_rows).map(|r| ((r * 7 + 3) % w) as u32).collect();
    let targets_i64: Vec<i64> = targets_u32.iter().map(|&t| t as i64).collect();

    let reference = cpu_cross_entropy(&data, &targets_i64, n_rows, w);

    let logits =
        Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w]).expect("from_vec_on logits");
    let tgt = Tensor::from_vec_on(Device::Rocm(0), targets_u32, vec![n_rows])
        .expect("from_vec_on u32 targets");

    let loss =
        kiln_tensor::rocm_cross_entropy_loss(&logits, &tgt).expect("rocm_cross_entropy_loss u32");
    let host = kiln_tensor::rocm_to_host_copy(&loss).expect("rocm_to_host_copy");
    let got = host.to_vec::<f32>().expect("to_vec");

    let diff = (got[0] - reference).abs();
    assert!(
        diff <= 1e-5 + 1e-4 * reference.abs(),
        "u32/large-batch cross_entropy mismatch: got {} ref {reference} diff {diff}",
        got[0]
    );
    eprintln!("cross_entropy U32 + large-batch parity passed (n_rows={n_rows}, w={w})");
}
