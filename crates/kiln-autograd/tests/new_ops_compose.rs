//! Integration test for the Phase 1.62-1.66 new ops.
//!
//! Exercises **layer_norm + gelu + broadcast_to + concat + dropout**
//! end-to-end through a forward graph + tape-based backward. Proves
//! the newly added op families compose cleanly with each other and
//! with the existing matmul / add / sum_all stack.
//!
//! No GPU work; pure CPU substrate.

use kiln_autograd::{
    AddBackward, BroadcastToBackward, ConcatBackward, DropoutBackward, GeluBackward,
    LayerNormBackward, MatmulBackward, ReduceBackward, ReduceKind, ReduceScope, Tape,
};
use kiln_tensor::ops::{
    add, broadcast_to, concat, dropout, gelu, layer_norm, matmul, sum_all,
};
use kiln_tensor::{CpuStorage, DType, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn accumulator(a: &Tensor, b: &Tensor) -> kiln_tensor::Result<Tensor> {
    add(a, b)
}

#[test]
fn layernorm_gelu_matmul_compose_through_tape() {
    // pipeline: x [1, 4] → layer_norm → gelu → matmul(W [4, 2]) →
    // bias_bcast [1, 1] → broadcast_to [1, 2] → add → sum_all → loss
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    let ln_w = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
    let ln_b = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap();
    let proj = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], vec![4, 2]).unwrap();
    let bias_scalar = Tensor::from_slice(&[0.1f32], vec![1, 1]).unwrap();
    let eps = 1e-6_f32;

    let mut tape = Tape::new();

    // 1. y_ln = layer_norm(x, ln_w, ln_b, eps)
    let y_ln = layer_norm(&x, &ln_w, &ln_b, eps).unwrap();
    tape.record(
        &y_ln,
        &[&x, &ln_w, &ln_b],
        Box::new(LayerNormBackward {
            x: x.clone(),
            weight: ln_w.clone(),
            eps,
        }),
    );

    // 2. y_gelu = gelu(y_ln)
    let y_gelu = gelu(&y_ln).unwrap();
    tape.record(
        &y_gelu,
        &[&y_ln],
        Box::new(GeluBackward { x: y_ln.clone() }),
    );

    // 3. y_proj = y_gelu @ proj
    let y_proj = matmul(&y_gelu, &proj).unwrap();
    tape.record(
        &y_proj,
        &[&y_gelu, &proj],
        Box::new(MatmulBackward {
            a: y_gelu.clone(),
            b: proj.clone(),
        }),
    );

    // 4. bias_b = broadcast_to(bias_scalar, [1, 2])
    let bias_b = broadcast_to(&bias_scalar, &[1, 2]).unwrap();
    tape.record(
        &bias_b,
        &[&bias_scalar],
        Box::new(BroadcastToBackward {
            input_shape: vec![1, 1],
        }),
    );

    // 5. y_add = y_proj + bias_b
    let y_add = add(&y_proj, &bias_b).unwrap();
    tape.record(&y_add, &[&y_proj, &bias_b], Box::new(AddBackward));

    // 6. loss = sum_all(y_add)
    let loss = sum_all(&y_add).unwrap();
    tape.record(
        &loss,
        &[&y_add],
        Box::new(ReduceBackward {
            input_shape: y_add.shape().to_vec(),
            dtype: y_add.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    // Backward.
    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    // Every leaf must have a finite gradient of the right shape.
    let d_x = store.get(x.id()).expect("d_x");
    assert_eq!(d_x.shape(), &[1, 4]);
    for v in read_f32(d_x) {
        assert!(v.is_finite(), "d_x has non-finite: {v}");
    }

    let d_ln_w = store.get(ln_w.id()).expect("d_ln_w");
    assert_eq!(d_ln_w.shape(), &[4]);

    let d_proj = store.get(proj.id()).expect("d_proj");
    assert_eq!(d_proj.shape(), &[4, 2]);

    let d_bias = store.get(bias_scalar.id()).expect("d_bias");
    assert_eq!(d_bias.shape(), &[1, 1]);
}

#[test]
fn concat_then_matmul_through_tape() {
    // pipeline: concat([a, b], axis=1) → matmul → sum_all → loss
    let a = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
    let b = Tensor::from_slice(&[3.0f32], vec![1, 1]).unwrap();
    let proj = Tensor::from_slice(&[1.0f32, 0.5, 0.5, 1.0, -1.0, 1.0], vec![3, 2]).unwrap();

    let mut tape = Tape::new();

    let cat = concat(&[&a, &b], 1).unwrap();
    tape.record(
        &cat,
        &[&a, &b],
        Box::new(ConcatBackward {
            axis: 1,
            input_axis_sizes: vec![2, 1],
            input_shapes: vec![vec![1, 2], vec![1, 1]],
            dtype: DType::F32,
        }),
    );
    assert_eq!(cat.shape(), &[1, 3]);

    let y = matmul(&cat, &proj).unwrap();
    tape.record(
        &y,
        &[&cat, &proj],
        Box::new(MatmulBackward {
            a: cat.clone(),
            b: proj.clone(),
        }),
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

    // d_a [1, 2] and d_b [1, 1] both produced by ConcatBackward
    // through the tape.
    let d_a = store.get(a.id()).expect("d_a");
    assert_eq!(d_a.shape(), &[1, 2]);
    let d_b = store.get(b.id()).expect("d_b");
    assert_eq!(d_b.shape(), &[1, 1]);

    let d_proj = store.get(proj.id()).expect("d_proj");
    assert_eq!(d_proj.shape(), &[3, 2]);

    // Analytic check: for sum(cat @ proj), d_cat[i,k] = Σ_n proj[k,n].
    // Then ConcatBackward splits d_cat back to d_a, d_b along axis 1.
    //  proj rows: [1+0.5, 0.5+1, -1+1] = [1.5, 1.5, 0]
    //  d_cat = [[1.5, 1.5, 0]]; d_a = [[1.5, 1.5]]; d_b = [[0]]
    let d_a_v = read_f32(d_a);
    let d_b_v = read_f32(d_b);
    assert!(
        (d_a_v[0] - 1.5).abs() < 1e-5,
        "d_a[0]={}, want 1.5",
        d_a_v[0]
    );
    assert!(
        (d_a_v[1] - 1.5).abs() < 1e-5,
        "d_a[1]={}, want 1.5",
        d_a_v[1]
    );
    assert!((d_b_v[0]).abs() < 1e-5, "d_b[0]={}, want 0", d_b_v[0]);
}

#[test]
fn dropout_with_layernorm_train_step() {
    // pipeline: x → layer_norm → dropout(p=0.5, seed=42) → sum_all → loss
    // Backward must produce a finite, shape-correct gradient for x.
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
    let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
    let eps = 1e-6;

    let mut tape = Tape::new();

    let y_ln = layer_norm(&x, &w, &b, eps).unwrap();
    tape.record(
        &y_ln,
        &[&x, &w, &b],
        Box::new(LayerNormBackward {
            x: x.clone(),
            weight: w.clone(),
            eps,
        }),
    );

    let (y_drop, mask) = dropout(&y_ln, 0.5, 42).unwrap();
    tape.record(
        &y_drop,
        &[&y_ln],
        Box::new(DropoutBackward { mask, p: 0.5 }),
    );

    let loss = sum_all(&y_drop).unwrap();
    tape.record(
        &loss,
        &[&y_drop],
        Box::new(ReduceBackward {
            input_shape: y_drop.shape().to_vec(),
            dtype: y_drop.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    let d_x = store.get(x.id()).expect("d_x");
    assert_eq!(d_x.shape(), &[2, 3]);
    for v in read_f32(d_x) {
        assert!(v.is_finite(), "d_x non-finite: {v}");
    }
}

#[test]
fn broadcast_then_elementwise_residual_through_tape() {
    // pipeline: x [2, 3] + broadcast_to(bias [1, 3], [2, 3]) → sum_all
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let bias = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![1, 3]).unwrap();

    let mut tape = Tape::new();
    let bias_b = broadcast_to(&bias, &[2, 3]).unwrap();
    tape.record(
        &bias_b,
        &[&bias],
        Box::new(BroadcastToBackward {
            input_shape: vec![1, 3],
        }),
    );

    let y = add(&x, &bias_b).unwrap();
    tape.record(&y, &[&x, &bias_b], Box::new(AddBackward));

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

    // d_x = ones [2, 3] (since loss = sum(x + ...)).
    let d_x = read_f32(store.get(x.id()).unwrap());
    assert_eq!(d_x, vec![1.0; 6]);
    // d_bias = sum across rows of d_y = [2, 2, 2] (each column sums
    // two rows of ones, then BroadcastToBackward sums along axis 0).
    let d_bias = read_f32(store.get(bias.id()).unwrap());
    assert_eq!(d_bias, vec![2.0, 2.0, 2.0]);
}
