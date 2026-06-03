//! Phase R.5 — CPU-vs-ROCm parity for the elementwise unary activation kernel.
//!
//! `activation.cu` is one-thread-per-element with no cross-lane reductions, so
//! there is no wave64 reduction hazard to straddle — a couple of shapes across
//! the core activation kinds (silu/sigmoid/gelu/tanh/relu) suffice. Skips when
//! no ROCm device is present.
//!
//! Run: `cargo test -p kiln-tensor --features rocm --test rocm_activation_parity`
#![cfg(feature = "rocm")]

use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device available; skipping R.5 activation parity test");
        true
    } else {
        false
    }
}

// Op kinds — must match the KIND_* macros in csrc/activation.cu and `UnaryKind`.
const KIND_SILU: i32 = 0;
const KIND_SIGMOID: i32 = 1;
const KIND_GELU: i32 = 2;
const KIND_TANH: i32 = 3;
const KIND_RELU: i32 = 4;

/// Deterministic pseudo-random value in ~[-5, 5) for index i.
fn val(i: usize) -> f32 {
    (((i * 131 + 7) % 1000) as f32) / 100.0 - 5.0
}

/// CPU reference computed in F32 (matches the kernel's F32-math convention).
fn cpu_ref(kind: i32, x: f32) -> f32 {
    match kind {
        KIND_SILU => x / (1.0 + (-x).exp()),
        KIND_SIGMOID => 1.0 / (1.0 + (-x).exp()),
        KIND_GELU => {
            let k = 0.797_884_56_f32; // sqrt(2/pi)
            let inner = k * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        KIND_TANH => x.tanh(),
        KIND_RELU => x.max(0.0),
        _ => unreachable!(),
    }
}

#[test]
fn activation_parity_core_kinds() {
    if no_rocm() {
        return;
    }

    let shapes: &[Vec<usize>] = &[vec![1usize], vec![1024], vec![3, 257], vec![4, 5, 33]];
    let kinds = [KIND_SILU, KIND_SIGMOID, KIND_GELU, KIND_TANH, KIND_RELU];

    for shape in shapes {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(val).collect();

        for &kind in &kinds {
            let reference: Vec<f32> = data.iter().map(|&x| cpu_ref(kind, x)).collect();

            let t = Tensor::from_vec_on(Device::Rocm(0), data.clone(), shape.clone())
                .unwrap_or_else(|e| panic!("from_vec_on (shape={shape:?}): {e}"));
            let y = kiln_tensor::rocm_activation_unary(&t, kind)
                .unwrap_or_else(|e| panic!("rocm_activation_unary (kind={kind}): {e}"));
            let host = kiln_tensor::rocm_to_host_copy(&y)
                .unwrap_or_else(|e| panic!("rocm_to_host_copy (kind={kind}): {e}"));
            let got = host.to_vec::<f32>().expect("to_vec");

            assert_eq!(got.len(), reference.len(), "shape {shape:?} kind {kind}");
            for (i, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
                let diff = (g - rf).abs();
                assert!(
                    diff <= 1e-5 + 1e-4 * rf.abs(),
                    "activation mismatch at shape={shape:?} kind={kind} idx={i}: \
                     got {g} ref {rf} diff {diff}"
                );
            }
        }
    }
    eprintln!("activation CPU-vs-ROCm parity passed across shapes/kinds");
}
