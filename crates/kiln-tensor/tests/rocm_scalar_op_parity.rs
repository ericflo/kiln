//! Phase R.5 — CPU-vs-ROCm parity for the tensor-scalar elementwise kernel.
//!
//! `scalar_op.cu` is elementwise (one thread per element, no cross-lane
//! reductions), so it has no wave-size hazard — a couple of shapes across all 8
//! op kinds is sufficient coverage. Compares `rocm_scalar_op` against a CPU
//! reference (op in F32) at f32 rtol 1e-4 / atol 1e-5. Skips when no ROCm device
//! is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_scalar_op_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 scalar_op parity test");
        true
    } else {
        false
    }
}

// Op kinds — match `ScalarKind` / the `apply_scalar` switch in scalar_op.cu.
const KIND_ADD_SCALAR: i32 = 0;
const KIND_SUB_SCALAR: i32 = 1;
const KIND_MUL_SCALAR: i32 = 2;
const KIND_DIV_SCALAR: i32 = 3;
const KIND_SCALAR_MINUS_TENSOR: i32 = 4;
const KIND_SCALAR_DIV_TENSOR: i32 = 5;
const KIND_MAX_WITH_SCALAR: i32 = 6;
const KIND_MIN_WITH_SCALAR: i32 = 7;

/// CPU reference for one op kind (matches the F32 math in scalar_op.cu).
fn apply_scalar(kind: i32, x: f32, c: f32) -> f32 {
    match kind {
        KIND_ADD_SCALAR => x + c,
        KIND_SUB_SCALAR => x - c,
        KIND_MUL_SCALAR => x * c,
        KIND_DIV_SCALAR => x / c,
        KIND_SCALAR_MINUS_TENSOR => c - x,
        KIND_SCALAR_DIV_TENSOR => c / x,
        KIND_MAX_WITH_SCALAR => x.max(c),
        KIND_MIN_WITH_SCALAR => x.min(c),
        _ => unreachable!("unexpected kind {kind}"),
    }
}

/// Deterministic value in ~[-5, 5), kept away from 0 so the div/`c/x` kinds
/// stay well-conditioned.
fn val(i: usize) -> f32 {
    let v = (((i * 37 + 11) % 1000) as f32) / 100.0 - 5.0;
    if v.abs() < 0.5 { v + 1.0 } else { v }
}

#[test]
fn scalar_op_parity_all_kinds() {
    if no_rocm() {
        return;
    }
    let shapes: [Vec<usize>; 3] = [vec![1], vec![257], vec![5, 33]];
    let kinds = [
        KIND_ADD_SCALAR,
        KIND_SUB_SCALAR,
        KIND_MUL_SCALAR,
        KIND_DIV_SCALAR,
        KIND_SCALAR_MINUS_TENSOR,
        KIND_SCALAR_DIV_TENSOR,
        KIND_MAX_WITH_SCALAR,
        KIND_MIN_WITH_SCALAR,
    ];
    let scalar: f32 = 1.75;

    for shape in &shapes {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(val).collect();

        for &kind in &kinds {
            let reference: Vec<f32> = data
                .iter()
                .map(|&x| apply_scalar(kind, x, scalar))
                .collect();

            let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), shape.clone())
                .unwrap_or_else(|e| panic!("from_vec_on (shape={shape:?}): {e}"));
            let y = kiln_tensor::rocm_scalar_op(&t, kind, scalar)
                .unwrap_or_else(|e| panic!("rocm_scalar_op (kind={kind}, shape={shape:?}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&y).unwrap_or_else(|e| {
                panic!("rocm_to_host_copy (kind={kind}, shape={shape:?}): {e}")
            });
            let got = host.to_vec::<f32>().expect("to_vec");

            assert_eq!(
                got.len(),
                reference.len(),
                "len (kind={kind}, shape={shape:?})"
            );
            for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
                let diff = (g - rf).abs();
                assert!(
                    diff <= 1e-5 + 1e-4 * rf.abs(),
                    "scalar_op mismatch kind={kind} shape={shape:?} idx={i}: \
                     got {g} ref {rf} diff {diff}"
                );
            }
        }
    }
    eprintln!("scalar_op CPU-vs-ROCm parity passed across all 8 kinds and shapes {shapes:?}");
}
