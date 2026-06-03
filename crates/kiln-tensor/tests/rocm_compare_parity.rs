//! Phase R.5 — element-wise compare parity test.
//!
//! Runs `rocm_compare` (eq/ne/lt/le/gt/ge over F32) against a CPU reference.
//! `compare.cu` is a per-element-thread elementwise kernel (no cross-lane
//! reduction), so it is not wave-size sensitive — but we still sweep the
//! 32/64-lane wavefront-boundary widths to catch any tail-masking bug, plus a
//! couple of 2-D shapes. Skips when no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_compare_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 compare parity test");
        true
    } else {
        false
    }
}

// Op kinds — must match CmpKind / compare.cu.
const KIND_EQ: i32 = 0;
const KIND_NE: i32 = 1;
const KIND_LT: i32 = 2;
const KIND_LE: i32 = 3;
const KIND_GT: i32 = 4;
const KIND_GE: i32 = 5;

fn cpu_cmp(kind: i32, a: f32, b: f32) -> u8 {
    let r = match kind {
        KIND_EQ => a == b,
        KIND_NE => a != b,
        KIND_LT => a < b,
        KIND_LE => a <= b,
        KIND_GT => a > b,
        KIND_GE => a >= b,
        _ => unreachable!(),
    };
    if r {
        1
    } else {
        0
    }
}

/// Deterministic pseudo-random value in ~[-3, 3) for index `i`. Tuned to
/// produce frequent exact ties (so eq/ne/le/ge exercise the equality branch).
fn val_a(i: usize) -> f32 {
    (((i * 37 + 11) % 7) as f32) - 3.0
}
fn val_b(i: usize) -> f32 {
    (((i * 53 + 5) % 7) as f32) - 3.0
}

#[test]
fn compare_parity_all_kinds() {
    if no_rocm() {
        return;
    }

    let kinds = [KIND_EQ, KIND_NE, KIND_LT, KIND_LE, KIND_GT, KIND_GE];
    // Wavefront-boundary widths plus a couple of strided cases.
    let widths = [
        1usize, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 255, 256, 257, 1000, 1024, 1025,
    ];

    for &w in &widths {
        let a_data: Vec<f32> = (0..w).map(val_a).collect();
        let b_data: Vec<f32> = (0..w).map(val_b).collect();

        for &kind in &kinds {
            let reference: Vec<u8> = (0..w).map(|i| cpu_cmp(kind, a_data[i], b_data[i])).collect();

            let a = Tensor::from_vec_on(Device::Rocm(0), a_data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on a (w={w}): {e}"));
            let b = Tensor::from_vec_on(Device::Rocm(0), b_data.clone(), vec![w])
                .unwrap_or_else(|e| panic!("from_vec_on b (w={w}): {e}"));

            let mask = kiln_tensor::rocm_compare(&a, &b, kind)
                .unwrap_or_else(|e| panic!("rocm_compare (w={w} kind={kind}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&mask)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy (w={w} kind={kind}): {e}"));
            let got = host.to_vec::<u8>().expect("to_vec u8");

            assert_eq!(got.len(), reference.len(), "len (w={w} kind={kind})");
            for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    g, rf,
                    "compare mismatch at w={w} kind={kind} idx={i}: got {g} ref {rf}"
                );
            }
        }
    }

    eprintln!("compare CPU-vs-ROCm parity passed across kinds and widths {widths:?}");
}

#[test]
fn compare_parity_2d_shape() {
    if no_rocm() {
        return;
    }

    let rows = 5usize;
    let cols = 96usize;
    let n = rows * cols;

    let a_data: Vec<f32> = (0..n).map(val_a).collect();
    let b_data: Vec<f32> = (0..n).map(val_b).collect();

    for &kind in &[KIND_LT, KIND_GE, KIND_EQ] {
        let reference: Vec<u8> = (0..n).map(|i| cpu_cmp(kind, a_data[i], b_data[i])).collect();

        let a = Tensor::from_vec_on(Device::Rocm(0), a_data.clone(), vec![rows, cols])
            .unwrap_or_else(|e| panic!("from_vec_on a 2d: {e}"));
        let b = Tensor::from_vec_on(Device::Rocm(0), b_data.clone(), vec![rows, cols])
            .unwrap_or_else(|e| panic!("from_vec_on b 2d: {e}"));

        let mask = kiln_tensor::rocm_compare(&a, &b, kind)
            .unwrap_or_else(|e| panic!("rocm_compare 2d (kind={kind}): {e}"));
        let host = kiln_tensor::rocm_to_host_copy(&mask)
            .unwrap_or_else(|e| panic!("rocm_to_host_copy 2d (kind={kind}): {e}"));
        let got = host.to_vec::<u8>().expect("to_vec u8");

        assert_eq!(got.len(), n, "2d len kind={kind}");
        for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(g, rf, "2d compare mismatch kind={kind} idx={i}");
        }
    }

    eprintln!("compare 2-D parity passed");
}
