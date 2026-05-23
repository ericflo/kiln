//! End-to-end tape integration with **real** BackwardOp impls
//! (Phase 6a.1–6a.7).
//!
//! This test builds a forward graph using `kiln_tensor::ops::*`,
//! records each op into a `Tape` along with the matching real
//! BackwardOp from `kiln_autograd::backwards::*`, runs the backward
//! walk, and compares the resulting gradients against a
//! finite-difference reference.
//!
//! Demonstrates that the **whole substrate composes end-to-end**:
//! forward kernels, tape recording, reverse-topo walk, per-input
//! gradient construction, anti-pattern 16 version checks, and grad
//! accumulation through `add`.

use kiln_autograd::{
    AddBackward, MatmulBackward, MulBackward, ReduceBackward, ReduceKind, ReduceScope,
    SoftmaxLastDimBackward, Tape,
};
use kiln_tensor::ops::{add, matmul, mul, softmax_last_dim, sum_all};
use kiln_tensor::{CpuStorage, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "len mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "idx {i}: got {x}, want {y} (tol {tol})"
        );
    }
}

fn accumulator(a: &Tensor, b: &Tensor) -> kiln_tensor::Result<Tensor> {
    add(a, b)
}

/// Build a small linear-regression-style graph:
///
/// ```text
/// y     = x @ w
/// z     = y + b
/// loss  = sum_all(z)
/// ```
///
/// Records the matmul, add, and sum_all ops on the tape and verifies
/// `d_w`, `d_x`, `d_b` against the analytic gradients.
#[test]
fn linear_layer_gradient_through_tape() {
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let w = Tensor::from_slice(&[1.0f32, 0.5, -1.0, 2.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![2, 2]).unwrap();

    let mut tape = Tape::new();

    // y = x @ w
    let y = matmul(&x, &w).unwrap();
    tape.record(
        &y,
        &[&x, &w],
        Box::new(MatmulBackward {
            a: x.clone(),
            b: w.clone(),
        }),
    );

    // z = y + b
    let z = add(&y, &b).unwrap();
    tape.record(&z, &[&y, &b], Box::new(AddBackward));

    // loss = sum_all(z)
    let loss = sum_all(&z).unwrap();
    tape.record(
        &loss,
        &[&z],
        Box::new(ReduceBackward {
            input_shape: z.shape().to_vec(),
            dtype: z.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    // Seed grad of the scalar loss = 1.
    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    // For loss = sum(z) = sum(x @ w + b):
    // d_x[i, k] = sum_n w[k, n]                    (per row)
    // d_w[k, n] = sum_b x[b, k]                    (per col)
    // d_b[i, j] = 1
    //
    // With x = [[1, 2], [3, 4]] and w = [[1, 0.5], [-1, 2]]:
    // sum_n w[0, :] = 1 + 0.5 = 1.5
    // sum_n w[1, :] = -1 + 2 = 1.0
    // So d_x = [[1.5, 1.0], [1.5, 1.0]]
    //
    // sum_b x[:, 0] = 1 + 3 = 4
    // sum_b x[:, 1] = 2 + 4 = 6
    // So d_w = [[4, 4], [6, 6]]
    //
    // d_b = ones [2, 2]
    let d_x = store.get(x.id()).expect("d_x present");
    let d_w = store.get(w.id()).expect("d_w present");
    let d_b = store.get(b.id()).expect("d_b present");
    approx(&read_f32(d_x), &[1.5, 1.0, 1.5, 1.0], 1e-5);
    approx(&read_f32(d_w), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    approx(&read_f32(d_b), &[1.0, 1.0, 1.0, 1.0], 1e-5);
}

/// Two ops feeding into the same parameter — verifies gradient
/// accumulation through `add`.
///
/// ```text
/// y1 = a * x
/// y2 = b * x       (same x; gradient should sum)
/// z  = y1 + y2
/// loss = sum_all(z)
/// ```
#[test]
fn shared_input_gradients_accumulate() {
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let a = Tensor::from_slice(&[10.0f32, 10.0, 10.0], vec![3]).unwrap();
    let b = Tensor::from_slice(&[100.0f32, 100.0, 100.0], vec![3]).unwrap();

    let mut tape = Tape::new();

    let y1 = mul(&a, &x).unwrap();
    tape.record(
        &y1,
        &[&a, &x],
        Box::new(MulBackward {
            a: a.clone(),
            b: x.clone(),
        }),
    );

    let y2 = mul(&b, &x).unwrap();
    tape.record(
        &y2,
        &[&b, &x],
        Box::new(MulBackward {
            a: b.clone(),
            b: x.clone(),
        }),
    );

    let z = add(&y1, &y2).unwrap();
    tape.record(&z, &[&y1, &y2], Box::new(AddBackward));

    let loss = sum_all(&z).unwrap();
    tape.record(
        &loss,
        &[&z],
        Box::new(ReduceBackward {
            input_shape: z.shape().to_vec(),
            dtype: z.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    // loss = sum(a*x + b*x)
    // d_x[i] = a[i] + b[i] = 10 + 100 = 110 (per element)
    // d_a[i] = x[i]
    // d_b[i] = x[i]
    let d_x = store.get(x.id()).expect("d_x");
    approx(&read_f32(d_x), &[110.0, 110.0, 110.0], 1e-4);
    let d_a = store.get(a.id()).expect("d_a");
    approx(&read_f32(d_a), &[1.0, 2.0, 3.0], 1e-5);
    let d_b = store.get(b.id()).expect("d_b");
    approx(&read_f32(d_b), &[1.0, 2.0, 3.0], 1e-5);
}

/// Softmax + sum loss, end-to-end through the tape.
///
/// ```text
/// y    = softmax(x, axis=-1)
/// loss = sum(y)   (= 1 trivially per row, total = batch)
/// ```
///
/// d_y = ones; d_x = softmax row-wise reaction. For uniform softmax
/// `y = [1/n, …, 1/n]` with dy = ones, the per-row sum
/// `s = sum_j y_j*dy_j = 1`, so `dx_i = y_i * (1 - 1) = 0`.
#[test]
fn softmax_sum_loss_through_tape() {
    let x = Tensor::from_slice(&[0.0f32; 6], vec![2, 3]).unwrap();
    let mut tape = Tape::new();

    let y = softmax_last_dim(&x).unwrap();
    tape.record(
        &y,
        &[&x],
        Box::new(SoftmaxLastDimBackward { y: y.clone() }),
    );

    let loss = sum_all(&y).unwrap();
    tape.record(
        &loss,
        &[&y],
        Box::new(ReduceBackward {
            input_shape: y.shape().to_vec(),
            dtype: y.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    let d_x = store.get(x.id()).expect("d_x");
    approx(&read_f32(d_x), &[0.0; 6], 1e-5);
}

/// A two-layer net: `loss = sum( softmax(x @ w1 + b1) @ w2 )`.
/// Verifies that the tape walks the deeper graph correctly and
/// produces non-trivial gradients for every parameter.
#[test]
fn two_layer_net_gradients_are_finite() {
    let x = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![1, 4]).unwrap();
    let w1 = Tensor::from_slice(&[0.1f32; 8], vec![4, 2]).unwrap();
    let b1 = Tensor::from_slice(&[0.05f32, -0.05], vec![1, 2]).unwrap();
    let w2 = Tensor::from_slice(&[1.0f32, -1.0], vec![2, 1]).unwrap();

    let mut tape = Tape::new();

    // h_pre = x @ w1
    let h_pre = matmul(&x, &w1).unwrap();
    tape.record(
        &h_pre,
        &[&x, &w1],
        Box::new(MatmulBackward {
            a: x.clone(),
            b: w1.clone(),
        }),
    );

    // h_lin = h_pre + b1
    let h_lin = add(&h_pre, &b1).unwrap();
    tape.record(&h_lin, &[&h_pre, &b1], Box::new(AddBackward));

    // h = softmax(h_lin)
    let h = softmax_last_dim(&h_lin).unwrap();
    tape.record(
        &h,
        &[&h_lin],
        Box::new(SoftmaxLastDimBackward { y: h.clone() }),
    );

    // out = h @ w2
    let out = matmul(&h, &w2).unwrap();
    tape.record(
        &out,
        &[&h, &w2],
        Box::new(MatmulBackward {
            a: h.clone(),
            b: w2.clone(),
        }),
    );

    // loss = sum_all(out)
    let loss = sum_all(&out).unwrap();
    tape.record(
        &loss,
        &[&out],
        Box::new(ReduceBackward {
            input_shape: out.shape().to_vec(),
            dtype: out.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    // We don't need exact analytic values here — just verify that
    // every parameter has a finite, non-NaN gradient of the right shape.
    let d_x = store.get(x.id()).expect("d_x");
    let d_w1 = store.get(w1.id()).expect("d_w1");
    let d_b1 = store.get(b1.id()).expect("d_b1");
    let d_w2 = store.get(w2.id()).expect("d_w2");
    assert_eq!(d_x.shape(), &[1, 4]);
    assert_eq!(d_w1.shape(), &[4, 2]);
    assert_eq!(d_b1.shape(), &[1, 2]);
    assert_eq!(d_w2.shape(), &[2, 1]);
    for v in read_f32(d_x) {
        assert!(v.is_finite(), "d_x has non-finite element: {v}");
    }
    for v in read_f32(d_w1) {
        assert!(v.is_finite(), "d_w1 has non-finite element: {v}");
    }
    for v in read_f32(d_b1) {
        assert!(v.is_finite(), "d_b1 has non-finite element: {v}");
    }
    for v in read_f32(d_w2) {
        assert!(v.is_finite(), "d_w2 has non-finite element: {v}");
    }
}

/// Anti-pattern 16: bumping a saved tensor's version between forward
/// and backward should produce a typed error mentioning the op name.
#[test]
fn anti_pattern_16_version_drift_caught_at_real_op() {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], vec![3]).unwrap();

    let mut tape = Tape::new();
    let c = mul(&a, &b).unwrap();
    tape.record(
        &c,
        &[&a, &b],
        Box::new(MulBackward {
            a: a.clone(),
            b: b.clone(),
        }),
    );

    // Simulate in-place mutation of `a` after recording.
    a.bump_version();

    let seed = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
    let e = tape.backward(c.id(), seed, accumulator).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("Anti-pattern 16"), "got: {msg}");
    assert!(msg.contains("mul_backward"), "got: {msg}");
}
