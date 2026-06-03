//! Phase R.5 — wave-size correctness parity for the reduce_last_axis kernels.
//!
//! `reduce_last_axis.cu` carried two two-level warp-shuffle reductions
//! (sum-of-squares and sum/mean) that hardcoded a 32-lane warp. On AMD wave64
//! the cross-warp shuffle of "warp 0" self-references lanes 32-63 and faults.
//! The fix routes both through `kiln_block_reduce_sum` (shared-memory tree).
//!
//! This test sweeps last-axis widths that straddle the 32- and 64-lane
//! wavefront boundaries {31,32,33,63,64,65,...}: a wave64 reduction bug
//! compiles cleanly and only manifests numerically here. Skips when no ROCm
//! device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_reduce_last_axis_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 reduce_last_axis parity test");
        true
    } else {
        false
    }
}

/// The wavefront-boundary sweep that catches wave64 bugs.
const WIDTHS: &[usize] = &[
    1, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
];

/// Deterministic pseudo-random value in ~[-5, 5) for (row, col).
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

fn close(got: f32, want: f32) -> bool {
    // f32 rtol 1e-4 / atol 1e-5.
    (got - want).abs() <= 1e-5 + 1e-4 * want.abs()
}

#[test]
fn sum_last_axis_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let n_rows = 6usize;
    for &w in WIDTHS {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        // CPU reference: per-row sum (F32 accumulation).
        let reference: Vec<f32> = (0..n_rows)
            .map(|r| data[r * w..(r + 1) * w].iter().sum())
            .collect();

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = kiln_tensor::rocm_sum_last_axis(&t)
            .unwrap_or_else(|e| panic!("rocm_sum_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "sum width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                close(*g, *rf),
                "sum mismatch at width={w} row={i}: got {g} ref {rf} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }
    }
    eprintln!("sum_last_axis CPU-vs-ROCm parity passed across widths {WIDTHS:?}");
}

#[test]
fn mean_last_axis_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let n_rows = 6usize;
    for &w in WIDTHS {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        let reference: Vec<f32> = (0..n_rows)
            .map(|r| data[r * w..(r + 1) * w].iter().sum::<f32>() / (w as f32))
            .collect();

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = kiln_tensor::rocm_mean_last_axis(&t)
            .unwrap_or_else(|e| panic!("rocm_mean_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "mean width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                close(*g, *rf),
                "mean mismatch at width={w} row={i}: got {g} ref {rf}"
            );
        }
    }
    eprintln!("mean_last_axis CPU-vs-ROCm parity passed across widths {WIDTHS:?}");
}

#[test]
fn sum_squared_last_axis_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let n_rows = 6usize;
    for &w in WIDTHS {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        // CPU reference: per-row sum of squares (F32).
        let reference: Vec<f32> = (0..n_rows)
            .map(|r| data[r * w..(r + 1) * w].iter().map(|&v| v * v).sum())
            .collect();

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = kiln_tensor::rocm_sum_squared_last_axis(&t)
            .unwrap_or_else(|e| panic!("rocm_sum_squared_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "sum_sq width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            // Larger magnitudes here (sum of squares), so use a relative bound.
            assert!(
                close(*g, *rf),
                "sum_sq mismatch at width={w} row={i}: got {g} ref {rf}"
            );
        }
    }
    eprintln!("sum_squared_last_axis CPU-vs-ROCm parity passed across widths {WIDTHS:?}");
}

#[test]
fn l2norm_last_axis_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let eps = 1e-6f32;
    let n_rows = 6usize;
    for &w in WIDTHS {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        // CPU reference: x / sqrt(sum(x^2) + eps), per row.
        let mut reference = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            let row = &data[r * w..(r + 1) * w];
            let ss: f32 = row.iter().map(|&v| v * v).sum::<f32>() + eps;
            let inv = if ss > 0.0 { 1.0 / ss.sqrt() } else { 0.0 };
            for &v in row {
                reference.push(v * inv);
            }
        }

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = kiln_tensor::rocm_l2norm_last_axis(&t, eps)
            .unwrap_or_else(|e| panic!("rocm_l2norm_last_axis (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "l2norm width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                close(*g, *rf),
                "l2norm mismatch at width={w} idx={i}: got {g} ref {rf} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }
    }
    eprintln!("l2norm_last_axis CPU-vs-ROCm parity passed across widths {WIDTHS:?}");
}
