//! Phase R.5b — cumsum / cumprod (scan over the last axis) CPU-vs-ROCm parity.
//!
//! `scan_axis.cu` is the GDN (gated-delta-net) cumsum hot path. The scan does a
//! three-phase contiguous-chunk reduction whose per-thread-total combine runs
//! ENTIRELY in shared memory (`smem[tid]` + `__syncthreads()`), so there are no
//! cross-lane shuffles — it is wave32/wave64-correct by construction. We still
//! sweep row widths straddling the 32/64-lane boundary so a future warp-level
//! fast path can't silently regress wave64.
//!
//! Run:
//!   cargo test -p kiln-tensor --features rocm --test rocm_scan_axis_parity
//!   KILN_ROCM_WAVE64=1 cargo test -p kiln-tensor --features rocm \
//!       --test rocm_scan_axis_parity   # validate under forced wave64
#![cfg(feature = "rocm")]

use kiln_tensor::ops::{cumprod, cumsum};
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5b scan parity test");
        true
    } else {
        false
    }
}

/// Deterministic value in ~[-1, 1) for (row, col). Kept small so cumprod stays
/// numerically tame across long rows.
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 200) as f32) / 100.0 - 1.0
}

// Widths straddling the 32/64-lane wavefront boundaries plus strided
// (n_cols > blockDim) cases.
const WIDTHS: &[usize] = &[
    1, 2, 7, 16, 31, 32, 33, 47, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    2048,
];

#[test]
fn cumsum_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let n_rows = 5usize;
    for &w in WIDTHS {
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }

        // CPU reference: inclusive prefix sum per row, F32 accumulation.
        let mut reference = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            let mut acc = 0.0f32;
            for c in 0..w {
                acc += data[r * w + c];
                reference.push(acc);
            }
        }

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = cumsum(&t, 1).unwrap_or_else(|e| panic!("cumsum (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("to_host (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-4 + 1e-4 * rf.abs(),
                "cumsum mismatch w={w} idx={i}: got {g} ref {rf} diff {diff}"
            );
        }
    }
    eprintln!("cumsum CPU-vs-ROCm parity passed across widths {WIDTHS:?}");
}

#[test]
fn cumprod_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }
    let n_rows = 5usize;
    for &w in WIDTHS {
        // Values near 1.0 so the running product doesn't under/overflow on long
        // rows (keeps the parity tolerance meaningful).
        let mut data = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(0.9 + 0.2 * (((r * 31 + c * 7) % 100) as f32) / 100.0);
            }
        }

        let mut reference = Vec::with_capacity(n_rows * w);
        for r in 0..n_rows {
            let mut acc = 1.0f32;
            for c in 0..w {
                acc *= data[r * w + c];
                reference.push(acc);
            }
        }

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on (w={w}): {e}"));
        let y = cumprod(&t, 1).unwrap_or_else(|e| panic!("cumprod (w={w}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y)
            .unwrap_or_else(|e| panic!("to_host (w={w}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec");

        assert_eq!(got.len(), reference.len(), "width {w}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            let diff = (g - rf).abs();
            assert!(
                diff <= 1e-3 + 1e-3 * rf.abs(),
                "cumprod mismatch w={w} idx={i}: got {g} ref {rf} diff {diff}"
            );
        }
    }
    eprintln!("cumprod CPU-vs-ROCm parity passed across widths {WIDTHS:?}");
}

#[test]
fn cumsum_rank1_known_values() {
    if no_rocm() {
        return;
    }
    let t = Tensor::from_vec_on(Device::Rocm(0), vec![1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
    let y = cumsum(&t, 0).unwrap();
    let host = kiln_tensor::rocm_to_host_copy(&y).unwrap();
    assert_eq!(host.to_vec::<f32>().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}
