//! Phase R.5 — wave-size correctness for the arbitrary-axis reductions
//! (`reduce_arbitrary_axis.cu`: sum/mean, min/max, bool all/any).
//!
//! Each output element is one block whose threads reduce a strided slice of the
//! reduced axis, then a block-reduce collapses them. A wave64 reduction bug (the
//! #1 ROCm hazard) compiles cleanly and only manifests numerically when the
//! reduced-axis width straddles the 32/64-lane wavefront boundary, so we sweep
//! axis widths {1,7,31,32,33,63,64,65,96,127,128,129,255,256,257,1000,1024,1025}.
//! The block-reduce path mapped from `kiln_block_reduce_*` is the fix under test.
//! Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_reduce_arbitrary_axis_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 reduce_arbitrary_axis parity test");
        true
    } else {
        false
    }
}

/// Wavefront-boundary sweep for the reduced-axis width.
const AXIS_WIDTHS: [usize; 18] = [
    1, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
];

/// Deterministic pseudo-random value in ~[-5, 5) for index `(o, r, i)`.
fn val(o: usize, r: usize, i: usize) -> f32 {
    (((o * 911 + r * 131 + i * 17 + 7) % 1000) as f32) / 100.0 - 5.0
}

fn approx_eq(got: f32, reference: f32) -> bool {
    // f32 rtol 1e-4 / atol 1e-5.
    (got - reference).abs() <= 1e-5 + 1e-4 * reference.abs()
}

#[test]
fn sum_mean_arbitrary_axis_wavefront_sweep() {
    if no_rocm() {
        return;
    }
    // Reduce axis=1 of a [outer, axis_dim, inner] tensor (a genuine non-last,
    // strided reduction with inner > 1).
    let outer = 3usize;
    let inner = 5usize;

    for &axis_dim in &AXIS_WIDTHS {
        let n = outer * axis_dim * inner;
        let mut data = Vec::with_capacity(n);
        for o in 0..outer {
            for r in 0..axis_dim {
                for i in 0..inner {
                    data.push(val(o, r, i));
                }
            }
        }

        // CPU reference: acc over the reduced axis (stride = inner).
        let mut ref_sum = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0.0f32;
                for r in 0..axis_dim {
                    acc += data[(o * axis_dim + r) * inner + i];
                }
                ref_sum[o * inner + i] = acc;
            }
        }
        let inv = 1.0f32 / (axis_dim as f32);
        let ref_mean: Vec<f32> = ref_sum.iter().map(|&s| s * inv).collect();

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![outer, axis_dim, inner])
            .unwrap_or_else(|e| panic!("from_vec_on (axis_dim={axis_dim}): {e}"));

        // sum
        let y = kiln_tensor::rocm_sum_axis(&t, 1)
            .unwrap_or_else(|e| panic!("rocm_sum_axis (axis_dim={axis_dim}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy sum");
        let got = host.to_vec::<f32>().expect("to_vec sum");
        assert_eq!(got.len(), ref_sum.len(), "sum len (axis_dim={axis_dim})");
        assert_eq!(y.shape().to_vec(), vec![outer, inner], "sum shape");
        for (idx, (g, rf)) in got.iter().zip(ref_sum.iter()).enumerate() {
            assert!(
                approx_eq(*g, *rf),
                "sum mismatch axis_dim={axis_dim} idx={idx}: got {g} ref {rf} \
                 (a wave64 reduction bug shows up exactly here)"
            );
        }

        // mean
        let y = kiln_tensor::rocm_mean_axis(&t, 1)
            .unwrap_or_else(|e| panic!("rocm_mean_axis (axis_dim={axis_dim}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy mean");
        let got = host.to_vec::<f32>().expect("to_vec mean");
        assert_eq!(got.len(), ref_mean.len(), "mean len (axis_dim={axis_dim})");
        for (idx, (g, rf)) in got.iter().zip(ref_mean.iter()).enumerate() {
            assert!(
                approx_eq(*g, *rf),
                "mean mismatch axis_dim={axis_dim} idx={idx}: got {g} ref {rf}"
            );
        }
    }
    eprintln!("sum/mean arbitrary-axis CPU-vs-ROCm parity passed across {AXIS_WIDTHS:?}");
}

#[test]
fn minmax_arbitrary_axis_wavefront_sweep() {
    if no_rocm() {
        return;
    }
    let outer = 3usize;
    let inner = 4usize;

    for &axis_dim in &AXIS_WIDTHS {
        let n = outer * axis_dim * inner;
        let mut data = Vec::with_capacity(n);
        for o in 0..outer {
            for r in 0..axis_dim {
                for i in 0..inner {
                    data.push(val(o, r, i));
                }
            }
        }

        let mut ref_min = vec![f32::INFINITY; outer * inner];
        let mut ref_max = vec![f32::NEG_INFINITY; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut mn = f32::INFINITY;
                let mut mx = f32::NEG_INFINITY;
                for r in 0..axis_dim {
                    let v = data[(o * axis_dim + r) * inner + i];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                ref_min[o * inner + i] = mn;
                ref_max[o * inner + i] = mx;
            }
        }

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![outer, axis_dim, inner])
            .unwrap_or_else(|e| panic!("from_vec_on (axis_dim={axis_dim}): {e}"));

        let y = kiln_tensor::rocm_min_axis(&t, 1)
            .unwrap_or_else(|e| panic!("rocm_min_axis (axis_dim={axis_dim}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy min");
        let got = host.to_vec::<f32>().expect("to_vec min");
        for (idx, (g, rf)) in got.iter().zip(ref_min.iter()).enumerate() {
            assert!(
                approx_eq(*g, *rf),
                "min mismatch axis_dim={axis_dim} idx={idx}: got {g} ref {rf}"
            );
        }

        let y = kiln_tensor::rocm_max_axis(&t, 1)
            .unwrap_or_else(|e| panic!("rocm_max_axis (axis_dim={axis_dim}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy max");
        let got = host.to_vec::<f32>().expect("to_vec max");
        for (idx, (g, rf)) in got.iter().zip(ref_max.iter()).enumerate() {
            assert!(
                approx_eq(*g, *rf),
                "max mismatch axis_dim={axis_dim} idx={idx}: got {g} ref {rf}"
            );
        }
    }
    eprintln!("min/max arbitrary-axis CPU-vs-ROCm parity passed across {AXIS_WIDTHS:?}");
}

#[test]
fn bool_reduce_arbitrary_axis_wavefront_sweep() {
    if no_rocm() {
        return;
    }
    let outer = 3usize;
    let inner = 4usize;

    // Three boolean patterns per (o,i) column to exercise ALL/ANY at the
    // wavefront boundary: "all true", "all false", and "one false near the
    // wave64 boundary" so a lost-lane bug flips the result.
    for &axis_dim in &AXIS_WIDTHS {
        let n = outer * axis_dim * inner;
        let mut data = vec![0u8; n];
        // For column (o, i) pick a pattern by (o*inner + i) % 3.
        for o in 0..outer {
            for i in 0..inner {
                let pattern = (o * inner + i) % 3;
                // The single-false / single-true position lands near the
                // wave64 boundary (lane 32+) when axis_dim is large enough.
                let special = if axis_dim > 33 { 33 } else { axis_dim - 1 };
                for r in 0..axis_dim {
                    let idx = (o * axis_dim + r) * inner + i;
                    let v = match pattern {
                        0 => 1u8,                                  // all true
                        1 => 0u8,                                  // all false
                        _ => {
                            if r == special {
                                0u8
                            } else {
                                1u8
                            }
                        } // one false
                    };
                    data[idx] = v;
                }
            }
        }

        // CPU reference for ALL (kind 0) and ANY (kind 1).
        let mut ref_all = vec![1u8; outer * inner];
        let mut ref_any = vec![0u8; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut all = 1u8;
                let mut any = 0u8;
                for r in 0..axis_dim {
                    let v = data[(o * axis_dim + r) * inner + i];
                    all &= if v != 0 { 1 } else { 0 };
                    any |= if v != 0 { 1 } else { 0 };
                }
                ref_all[o * inner + i] = all;
                ref_any[o * inner + i] = any;
            }
        }

        let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![outer, axis_dim, inner])
            .unwrap_or_else(|e| panic!("from_vec_on u8 (axis_dim={axis_dim}): {e}"));

        // ALL
        let y = kiln_tensor::rocm_bool_reduce_axis(&t, 1, 0)
            .unwrap_or_else(|e| panic!("rocm_bool_reduce_axis ALL (axis_dim={axis_dim}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy all");
        let got = host.to_vec::<u8>().expect("to_vec all");
        assert_eq!(got, ref_all, "ALL mismatch (axis_dim={axis_dim})");

        // ANY
        let y = kiln_tensor::rocm_bool_reduce_axis(&t, 1, 1)
            .unwrap_or_else(|e| panic!("rocm_bool_reduce_axis ANY (axis_dim={axis_dim}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&y).expect("rocm_to_host_copy any");
        let got = host.to_vec::<u8>().expect("to_vec any");
        assert_eq!(got, ref_any, "ANY mismatch (axis_dim={axis_dim})");
    }
    eprintln!("bool all/any arbitrary-axis CPU-vs-ROCm parity passed across {AXIS_WIDTHS:?}");
}
