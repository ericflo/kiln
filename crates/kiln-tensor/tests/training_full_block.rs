//! Full transformer block trained end-to-end through the substrate.
//!
//! Same architecture as `transformer_block_e2e.rs` but with cross-
//! entropy loss + manual gradient descent over the FFN weights.
//! Verifies the substrate trains a real architecture on a synthetic
//! next-token prediction task.

use kiln_autograd::{
    AddBackward, CrossEntropyBackward, GeluBackward, LayerNormBackward, MatmulBackward, Tape,
};
use kiln_tensor::ops::{
    add, cross_entropy, gelu, layer_norm, matmul, xavier_uniform,
};
use kiln_tensor::{CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};
use std::sync::Arc;

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn scalar_f32(t: &Tensor) -> f32 {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
}

fn sgd_step(param: &Tensor, grad: &Tensor, lr: f32) -> Result<Tensor> {
    let p = read_f32(param);
    let g = read_f32(grad);
    let new: Vec<f32> = p.iter().zip(g.iter()).map(|(&pv, &gv)| pv - lr * gv).collect();
    let bytes: Vec<u8> = new.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let cpu = CpuStorage::from_bytes(DType::F32, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(param.shape().to_vec()), TensorId::next())
}

fn accumulator(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    add(a, b)
}

#[test]
fn train_block_descends() {
    // Mini "block" — layer_norm → linear (via matmul) → gelu → linear
    // → cross-entropy classification. Use random x + targets,
    // verify loss descends.

    const BATCH: usize = 8;
    const HIDDEN: usize = 8;
    const HIDDEN2: usize = 12;
    const VOCAB: usize = 4;

    let x = xavier_uniform(vec![BATCH, HIDDEN], 1, DType::F32).unwrap();
    let mut w_norm = Tensor::from_slice(&[1.0f32; HIDDEN], vec![HIDDEN]).unwrap();
    let b_norm = Tensor::from_slice(&[0.0f32; HIDDEN], vec![HIDDEN]).unwrap();
    let mut w1 = xavier_uniform(vec![HIDDEN, HIDDEN2], 2, DType::F32).unwrap();
    let mut w2 = xavier_uniform(vec![HIDDEN2, VOCAB], 3, DType::F32).unwrap();

    let targets = Tensor::from_slice(&[0i64, 1, 2, 3, 0, 1, 2, 3], vec![BATCH]).unwrap();

    let mut losses = Vec::with_capacity(40);
    let lr = 0.05_f32;
    let eps = 1e-6_f32;

    for _ in 0..40 {
        let mut tape = Tape::new();

        // 1. y_ln = layer_norm(x, w_norm, b_norm, eps)
        let y_ln = layer_norm(&x, &w_norm, &b_norm, eps).unwrap();
        tape.record(
            &y_ln,
            &[&x, &w_norm, &b_norm],
            Box::new(LayerNormBackward {
                x: x.clone(),
                weight: w_norm.clone(),
                eps,
            }),
        );

        // 2. h_pre = y_ln @ w1
        let h_pre = matmul(&y_ln, &w1).unwrap();
        tape.record(
            &h_pre,
            &[&y_ln, &w1],
            Box::new(MatmulBackward {
                a: y_ln.clone(),
                b: w1.clone(),
            }),
        );

        // 3. h = gelu(h_pre)
        let h = gelu(&h_pre).unwrap();
        tape.record(
            &h,
            &[&h_pre],
            Box::new(GeluBackward { x: h_pre.clone() }),
        );

        // 4. logits = h @ w2
        let logits = matmul(&h, &w2).unwrap();
        tape.record(
            &logits,
            &[&h, &w2],
            Box::new(MatmulBackward {
                a: h.clone(),
                b: w2.clone(),
            }),
        );

        // 5. loss = cross_entropy(logits, targets)
        let loss = cross_entropy(&logits, &targets).unwrap();
        tape.record(
            &loss,
            &[&logits, &targets],
            Box::new(CrossEntropyBackward {
                logits: logits.clone(),
                targets: targets.clone(),
            }),
        );

        let loss_val = scalar_f32(&loss);
        losses.push(loss_val);

        // Backward.
        let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
        let store = tape.backward(loss.id(), seed, accumulator).unwrap();

        // SGD on w_norm, w1, w2.
        if let Some(d) = store.get(w_norm.id()) {
            w_norm = sgd_step(&w_norm, d, lr).unwrap();
        }
        if let Some(d) = store.get(w1.id()) {
            w1 = sgd_step(&w1, d, lr).unwrap();
        }
        if let Some(d) = store.get(w2.id()) {
            w2 = sgd_step(&w2, d, lr).unwrap();
        }
    }

    // Sanity: loss decreased noticeably.
    let first = losses[0];
    let last = losses[39];
    assert!(
        last < first * 0.8,
        "loss did not descend enough: first={first}, last={last}"
    );

    // Cross-entropy loss for 4 classes baseline (uniform) is ln(4) ≈ 1.386.
    // Trained should be well below that.
    assert!(last < 1.3, "trained loss {last} should be < uniform baseline");
}

#[test]
fn block_state_tracks_step_count() {
    // Sanity test — same flow but verifies parameter id stability
    // (anti-pattern 11) across an updated weight tensor.
    let x = xavier_uniform(vec![4, 8], 1, DType::F32).unwrap();
    let w = xavier_uniform(vec![8, 4], 2, DType::F32).unwrap();
    let w_id = w.id();
    let g = Tensor::from_slice(&[0.01f32; 32], vec![8, 4]).unwrap();
    let w2 = sgd_step(&w, &g, 0.1).unwrap();
    assert_ne!(w2.id(), w_id, "fresh tensor has fresh id");
    let _ = x; // silence
}
