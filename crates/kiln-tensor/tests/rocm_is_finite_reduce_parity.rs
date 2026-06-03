//! Phase R.5 parity test for the `is_finite_reduce` kernel.
//!
//! `rocm_is_finite` is a grid-wide `atomicOr` reduction (NOT a cross-lane
//! warp-shuffle reduction), so it is wave-size correct on wave32/wave64 as-is.
//! This test still sweeps the wavefront-boundary widths {1,7,31,32,33,...} that
//! would expose a wave64 bug, and probes a non-finite element planted at every
//! position within the last axis (so a per-lane masking bug at lane 32/63 would
//! show up). Compared against a trivial CPU reference (`all elements finite?`).
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_is_finite_reduce_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 is_finite_reduce parity test");
        true
    } else {
        false
    }
}

/// Deterministic finite value in ~[-5, 5) for (row, col).
fn val(r: usize, c: usize) -> f32 {
    (((r * 131 + c * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// CPU reference: `true` iff every element is finite.
fn cpu_all_finite(data: &[f32]) -> bool {
    data.iter().all(|x| x.is_finite())
}

#[test]
fn is_finite_parity_wavefront_boundary_sweep() {
    if no_rocm() {
        return;
    }

    // Last-axis widths straddling the 32- and 64-lane wavefront boundaries plus
    // strided (width > blockDim) cases.
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];
    let n_rows = 3usize;

    for &w in &widths {
        let total = n_rows * w;

        // ---- Case 1: all finite -> expect Ok(true). ----
        let mut data: Vec<f32> = Vec::with_capacity(total);
        for r in 0..n_rows {
            for c in 0..w {
                data.push(val(r, c));
            }
        }
        let reference = cpu_all_finite(&data);
        let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), vec![n_rows, w])
            .unwrap_or_else(|e| panic!("from_vec_on all-finite (w={w}): {e}"));
        let got = kiln_tensor::rocm_is_finite(&t)
            .unwrap_or_else(|e| panic!("rocm_is_finite all-finite (w={w}): {e}"));
        assert_eq!(
            got, reference,
            "all-finite mismatch at width={w}: got {got} ref {reference}"
        );

        // ---- Case 2: plant a non-finite at a sweep of positions in the last
        // axis (catches a per-lane mask bug at the 32/63 wavefront boundary). ----
        let bad_vals = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        // Probe positions: boundary lanes + last column + a mid column.
        let mut positions: Vec<usize> = vec![0, w.saturating_sub(1)];
        for p in [31usize, 32, 33, 63, 64, 65, w / 2] {
            if p < w {
                positions.push(p);
            }
        }
        positions.sort_unstable();
        positions.dedup();

        for (k, &pos) in positions.iter().enumerate() {
            let bad = bad_vals[k % bad_vals.len()];
            // Place the bad value in a rotating row so we also exercise the
            // grid-strided block walk for the larger widths.
            let bad_row = k % n_rows;
            let mut bad_data = data.clone();
            bad_data[bad_row * w + pos] = bad;

            let reference = cpu_all_finite(&bad_data);
            assert!(
                !reference,
                "test bug: planted {bad} should make CPU ref false (w={w}, pos={pos})"
            );

            let t = Tensor::from_vec_on(Device::Rocm(0), bad_data, vec![n_rows, w])
                .unwrap_or_else(|e| panic!("from_vec_on non-finite (w={w}, pos={pos}): {e}"));
            let got = kiln_tensor::rocm_is_finite(&t)
                .unwrap_or_else(|e| panic!("rocm_is_finite non-finite (w={w}, pos={pos}): {e}"));
            assert_eq!(
                got, reference,
                "non-finite mismatch at width={w} pos={pos} val={bad}: got {got} ref {reference} \
                 (a wave64 / per-lane-mask bug shows up exactly here)"
            );
        }
    }

    eprintln!(
        "is_finite CPU-vs-ROCm parity passed across wavefront-boundary widths {widths:?}"
    );
}
