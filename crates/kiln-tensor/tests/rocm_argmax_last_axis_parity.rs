//! Phase R.5 — wave-size correctness test for `rocm_argmax_last_axis`.
//!
//! Runs `rocm_argmax_last_axis` against a CPU reference at row widths that
//! straddle the 32- and 64-lane wavefront boundaries {31,32,33,63,64,65,...}.
//! argmax reduces a (val, idx) PAIR with a lowest-index tie-break across the
//! block; the original kernel did this with a two-level `__shfl_xor` reduction
//! (BROKEN on AMD wave64). The fixed kernel uses a wave-size-agnostic paired
//! shared-memory tree reduction. A wave64 reduction bug compiles cleanly and
//! only manifests numerically at these widths, so the power-of-two-straddling
//! sweep is load-bearing. Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_argmax_last_axis_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 argmax parity test");
        true
    } else {
        false
    }
}

/// Deterministic pseudo-random value in ~[-5, 5) for (row, col). Distinct enough
/// that most rows have a unique max; widths that force ties exercise the
/// lowest-index tie-break.
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// CPU reference: per-row argmax with lowest-index tie-break (strict `>`),
/// matching `slice.iter().enumerate().max_by(...)` and candle's `argmax`.
fn cpu_argmax(row: &[f32]) -> i64 {
    let mut best_val = f32::NEG_INFINITY;
    let mut best_idx: i64 = 0;
    for (c, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = c as i64;
        }
    }
    best_idx
}

#[test]
fn argmax_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    // Widths chosen to straddle the 32- and 64-lane boundaries plus a few
    // strided (n_cols > blockDim) cases. The exact wavefront-boundary sweep
    // that catches wave64 reduction bugs.
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

        // CPU reference: one argmax index per row.
        let mut reference = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row = &data[r * w..(r + 1) * w];
            reference.push(cpu_argmax(row));
        }

        // Device path.
        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = kiln_tensor::rocm_argmax_last_axis(&t)
            .unwrap_or_else(|e| panic!("rocm_argmax_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<i64>().expect("to_vec i64");

        assert_eq!(got.len(), reference.len(), "width {w}: row count");
        for (r, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                g, rf,
                "argmax mismatch at width={w} row={r}: got {g} ref {rf} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }
    }
    eprintln!("argmax CPU-vs-ROCm parity passed across wavefront-boundary widths {widths:?}");
}

#[test]
fn argmax_parity_tie_break_lowest_index() {
    if no_rocm() {
        return;
    }
    // All-equal row: the lowest index (0) must win on every backend, including
    // across wavefront boundaries.
    for &w in &[1usize, 32, 64, 65, 128, 257] {
        let data = vec![3.5f32; w];
        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![1, w])
            .unwrap_or_else(|e| panic!("from_vec_on tie (w={w}): {e}"));
        let y = kiln_tensor::rocm_argmax_last_axis(&t)
            .unwrap_or_else(|e| panic!("rocm_argmax_last_axis tie (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy tie (w={w}): {e}"));
        let got = host.to_vec::<i64>().expect("to_vec i64");
        assert_eq!(got, vec![0i64], "tie-break must pick lowest index (w={w})");
    }
    eprintln!("argmax lowest-index tie-break parity passed");
}
