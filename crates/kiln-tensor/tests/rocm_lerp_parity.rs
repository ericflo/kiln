//! Phase R.5 — element-wise lerp parity test.
//!
//! Runs `rocm_lerp` (out = a + weight*(b - a) over F32) against a CPU reference.
//! `lerp.cu` is a per-element-thread elementwise kernel (no cross-lane
//! reduction), so it is not wave-size sensitive — but we still sweep the
//! 32/64-lane wavefront-boundary widths to catch any tail-masking bug, plus a
//! couple of 2-D shapes and a sweep of weights. Skips when no ROCm device.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_lerp_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 lerp parity test");
        true
    } else {
        false
    }
}

/// Matches `apply_lerp` in lerp.cu one-for-one: a + w * (b - a).
fn cpu_lerp(a: f32, b: f32, w: f32) -> f32 {
    a + w * (b - a)
}

fn close(g: f32, r: f32) -> bool {
    let diff = (g - r).abs();
    diff <= 1e-5 + 1e-4 * r.abs()
}

/// Deterministic pseudo-random values spanning a range with negatives.
fn val_a(i: usize) -> f32 {
    (((i * 31 + 7) % 23) as f32) - 11.0
}
fn val_b(i: usize) -> f32 {
    (((i * 17 + 3) % 19) as f32) - 9.0
}

#[test]
fn lerp_parity_widths_and_weights() {
    if no_rocm() {
        return;
    }

    // Wavefront-boundary widths (catches wave64 tail-masking bugs).
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];
    let weights = [0.0f32, 0.25, 0.5, 1.0, -0.5, 1.5];

    for &w in &widths {
        let a_data: Vec<f32> = (0..w).map(val_a).collect();
        let b_data: Vec<f32> = (0..w).map(val_b).collect();

        for &weight in &weights {
            let reference: Vec<f32> = (0..w)
                .map(|i| cpu_lerp(a_data[i], b_data[i], weight))
                .collect();

            let a = Tensor::from_vec_on(Device::Rocm(0), a_data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on a (w={w}): {e}"));
            let b = Tensor::from_vec_on(Device::Rocm(0), b_data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on b (w={w}): {e}"));

            let out = kiln_tensor::rocm_lerp(&a, &b, weight)
                .unwrap_or_else(|e| panic!("rocm_lerp (w={w} weight={weight}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&out)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w} weight={weight}): {e}"));
            let got = host.to_vec::<f32>().expect("to_vec f32");

            assert_eq!(got.len(), reference.len(), "len (w={w} weight={weight})");
            for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
                assert!(
                    close(*g, *rf),
                    "lerp mismatch at w={w} weight={weight} idx={i}: got {g} ref {rf}"
                );
            }
        }
    }

    eprintln!("lerp CPU-vs-ROCm parity passed across widths {widths:?} and weights {weights:?}");
}

#[test]
fn lerp_parity_2d_shape() {
    if no_rocm() {
        return;
    }

    let rows = 5usize;
    let cols = 96usize;
    let n = rows * cols;

    let a_data: Vec<f32> = (0..n).map(val_a).collect();
    let b_data: Vec<f32> = (0..n).map(val_b).collect();

    for &weight in &[0.3f32, 0.75, 1.25] {
        let reference: Vec<f32> = (0..n)
            .map(|i| cpu_lerp(a_data[i], b_data[i], weight))
            .collect();

        let a = Tensor::from_vec_on(Device::Rocm(0), a_data.clone(), vec![rows, cols])
            .unwrap_or_else(|e| panic!("from_vec_on a 2d: {e}"));
        let b = Tensor::from_vec_on(Device::Rocm(0), b_data.clone(), vec![rows, cols])
            .unwrap_or_else(|e| panic!("from_vec_on b 2d: {e}"));

        let out = kiln_tensor::rocm_lerp(&a, &b, weight)
            .unwrap_or_else(|e| panic!("rocm_lerp 2d (weight={weight}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&out)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy 2d (weight={weight}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec f32");

        assert_eq!(got.len(), n, "2d len weight={weight}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                close(*g, *rf),
                "2d lerp mismatch weight={weight} idx={i}: got {g} ref {rf}"
            );
        }
    }

    eprintln!("lerp 2-D parity passed");
}
