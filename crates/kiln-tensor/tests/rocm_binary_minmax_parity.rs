//! Phase R.5 — element-wise binary minimum / maximum parity test.
//!
//! Runs `rocm_binary_minmax` (min/max over F32) against a CPU reference.
//! `binary_minmax.cu` is a per-element-thread elementwise kernel (one thread
//! per element, no cross-lane reduction), so it is not wave-size sensitive —
//! but we still sweep the 32/64-lane wavefront-boundary widths to catch any
//! tail-masking bug, plus a couple of 2-D shapes. Skips when no ROCm device is
//! present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_binary_minmax_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 binary_minmax parity test");
        true
    } else {
        false
    }
}

// Op kinds — must match binary_minmax.cu (KIND_MINIMUM / KIND_MAXIMUM) and the
// CPU reference `ops::binary_minmax::minimum` / `maximum`.
const KIND_MINIMUM: i32 = 0;
const KIND_MAXIMUM: i32 = 1;

// CPU reference matches the `.cu` contract: Rust `f32::min` / `f32::max`
// semantics (fminf / fmaxf propagate the non-NaN operand).
fn cpu_minmax(kind: i32, a: f32, b: f32) -> f32 {
    match kind {
        KIND_MINIMUM => a.min(b),
        KIND_MAXIMUM => a.max(b),
        _ => unreachable!(),
    }
}

fn approx_eq(got: f32, reference: f32) -> bool {
    if got == reference {
        return true;
    }
    if got.is_nan() && reference.is_nan() {
        return true;
    }
    let diff = (got - reference).abs();
    diff <= 1e-5 + 1e-4 * reference.abs()
}

/// Deterministic pseudo-random value in ~[-4, 4) for index `i`. Tuned to
/// produce frequent exact ties (so both operands are equally likely chosen).
fn val_a(i: usize) -> f32 {
    (((i * 41 + 13) % 9) as f32) - 4.0
}
fn val_b(i: usize) -> f32 {
    (((i * 59 + 7) % 9) as f32) - 4.0
}

#[test]
fn binary_minmax_parity_all_kinds() {
    if no_rocm() {
        return;
    }

    let kinds = [KIND_MINIMUM, KIND_MAXIMUM];
    // Wavefront-boundary widths (catch any wave32/wave64 tail-masking bug).
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];

    for &w in &widths {
        let a_data: Vec<f32> = (0..w).map(val_a).collect();
        let b_data: Vec<f32> = (0..w).map(val_b).collect();

        for &kind in &kinds {
            let reference: Vec<f32> =
                (0..w).map(|i| cpu_minmax(kind, a_data[i], b_data[i])).collect();

            let a = Tensor::from_vec_on(Device::Rocm(0), a_data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on a (w={w}): {e}"));
            let b = Tensor::from_vec_on(Device::Rocm(0), b_data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on b (w={w}): {e}"));

            let out = kiln_tensor::rocm_binary_minmax(&a, &b, kind)
                .unwrap_or_else(|e| panic!("rocm_binary_minmax (w={w} kind={kind}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&out)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w} kind={kind}): {e}"));
            let got = host.to_vec::<f32>().expect("to_vec f32");

            assert_eq!(got.len(), reference.len(), "len (w={w} kind={kind})");
            for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
                assert!(
                    approx_eq(*g, *rf),
                    "binary_minmax mismatch at w={w} kind={kind} idx={i}: got {g} ref {rf}"
                );
            }
        }
    }

    eprintln!("binary_minmax CPU-vs-ROCm parity passed across kinds and widths {widths:?}");
}

#[test]
fn binary_minmax_parity_2d_shape() {
    if no_rocm() {
        return;
    }

    let rows = 5usize;
    let cols = 96usize;
    let n = rows * cols;

    let a_data: Vec<f32> = (0..n).map(val_a).collect();
    let b_data: Vec<f32> = (0..n).map(val_b).collect();

    for &kind in &[KIND_MINIMUM, KIND_MAXIMUM] {
        let reference: Vec<f32> =
            (0..n).map(|i| cpu_minmax(kind, a_data[i], b_data[i])).collect();

        let a = Tensor::from_vec_on(Device::Rocm(0), a_data.clone(), vec![rows, cols])
            .unwrap_or_else(|e| panic!("from_vec_on a 2d: {e}"));
        let b = Tensor::from_vec_on(Device::Rocm(0), b_data.clone(), vec![rows, cols])
            .unwrap_or_else(|e| panic!("from_vec_on b 2d: {e}"));

        let out = kiln_tensor::rocm_binary_minmax(&a, &b, kind)
            .unwrap_or_else(|e| panic!("rocm_binary_minmax 2d (kind={kind}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&out)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy 2d (kind={kind}): {e}"));
        let got = host.to_vec::<f32>().expect("to_vec f32");

        assert_eq!(got.len(), reference.len(), "len 2d (kind={kind})");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert!(
                approx_eq(*g, *rf),
                "binary_minmax 2d mismatch at kind={kind} idx={i}: got {g} ref {rf}"
            );
        }
    }

    eprintln!("binary_minmax 2-D CPU-vs-ROCm parity passed");
}
