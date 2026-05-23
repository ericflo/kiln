//! Full training step end-to-end demo.
//!
//! Composes the entire #1082 substrate on a complete training-step
//! flow:
//!
//! 1. **kiln-tensor**: forward ops (`matmul`, `mean_all`)
//! 2. **kiln-autograd**: `Tape::record` per op, `Tape::backward` walk
//! 3. **kiln-param**: `Parameter` holds master + tensor_id
//! 4. **kiln-optim**: `AdamW::step(parameter, grad)` updates moments
//!
//! Anti-pattern 11 (stable TensorId) + anti-pattern 16 (in-place
//! version detection) compose end-to-end without per-impl awareness.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kiln_autograd::{BackwardOp, Tape};
use kiln_optim::{AdamW, OptimStep};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor as kt;

/// Forward op pretend-derivative: emits an all-ones gradient of the
/// correct *input* shape for each recorded input. The actual values
/// are irrelevant — the test exercises substrate composition, not
/// gradient math. The shapes must match because AdamW's
/// `step(param, grad)` rejects shape mismatches (anti-pattern 16
/// adjacency).
#[derive(Debug)]
struct PassthroughBwd {
    name: &'static str,
    input_shapes: Vec<Vec<usize>>,
    apply_count: Arc<AtomicUsize>,
}

impl BackwardOp for PassthroughBwd {
    fn name(&self) -> &'static str {
        self.name
    }
    fn input_count(&self) -> usize {
        self.input_shapes.len()
    }
    fn apply(&self, _grad_output: &kt::Tensor) -> kt::Result<Vec<Option<kt::Tensor>>> {
        self.apply_count.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .input_shapes
            .iter()
            .map(|shape| {
                let n: usize = shape.iter().product();
                let data = vec![1.0f32; n.max(1)];
                Some(kt::Tensor::from_slice(&data, shape.clone()).unwrap())
            })
            .collect())
    }
}

#[test]
fn full_training_step_substrate_composes_end_to_end() {
    // ── Setup ──────────────────────────────────────────────────────
    // Parameter: a `[2, 2]` weight matrix; master = forward (fp32_ref).
    let weight_fwd =
        kt::Tensor::from_slice(&[1.0f32, 0.5, 0.5, 1.0], vec![2, 2]).unwrap();
    let weight_master =
        kt::Tensor::from_slice(&[1.0f32, 0.5, 0.5, 1.0], vec![2, 2]).unwrap();
    let mut param = Parameter::trainable(
        ForwardStorage::Plain(weight_fwd.clone()),
        weight_master,
        AmpPolicy::fp32_reference(),
    );
    param.set_name("test.weight");
    let param_id = param.tensor_id();

    // Input + target.
    let input = kt::Tensor::from_slice(&[2.0f32, 1.0, 1.0, 2.0], vec![2, 2]).unwrap();

    // ── Forward + tape recording ───────────────────────────────────
    let mut tape = Tape::new();
    let apply_count = Arc::new(AtomicUsize::new(0));

    // Step 1: `out = input @ weight_fwd`
    let out = kt::ops::matmul(&input, &weight_fwd).unwrap();
    tape.record(
        &out,
        &[&input, &weight_fwd],
        Box::new(PassthroughBwd {
            name: "matmul",
            input_shapes: vec![input.shape().to_vec(), weight_fwd.shape().to_vec()],
            apply_count: apply_count.clone(),
        }),
    );

    // Step 2: `loss = mean(out)` (rank-0)
    let loss = kt::ops::mean_all(&out).unwrap();
    tape.record(
        &loss,
        &[&out],
        Box::new(PassthroughBwd {
            name: "mean_all",
            input_shapes: vec![out.shape().to_vec()],
            apply_count: apply_count.clone(),
        }),
    );

    // ── Backward via tape ──────────────────────────────────────────
    // Seed: d(loss)/d(loss) = 1.
    let seed = kt::Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape
        .backward(loss.id(), seed, |a, b| kt::ops::add(a, b))
        .unwrap();
    // PassthroughBwd ran twice (once per recorded op).
    assert_eq!(apply_count.load(Ordering::SeqCst), 2);

    // The weight tensor's grad should be in the store.
    let weight_grad = store.get(weight_fwd.id()).expect(
        "weight_fwd's grad missing from store — the tape walker did not \
         propagate through the matmul op's input list",
    );

    // ── Optimizer step ─────────────────────────────────────────────
    let mut opt = AdamW::default_hp();
    opt.step(&mut param, weight_grad).unwrap();

    let moments = opt.moments(param_id).expect("AdamW moments");
    assert_eq!(moments.step, 1);
    assert_eq!(moments.m.len(), 4);

    // ── End-of-step bookkeeping ─────────────────────────────────────
    // Anti-pattern 16 enforcement: between training steps, the tape
    // MUST be cleared before any in-place mutation. Today the
    // CPU AdamW write-back is a documented no-op (see kiln-optim/
    // src/adamw.rs), so we don't actually bump versions — but the
    // contract is in place.
    tape.clear();
    assert!(tape.is_empty());

    // The parameter's name + content_hash + tensor_id are all stable
    // across the step.
    assert_eq!(param.name(), Some("test.weight"));
    assert_eq!(param.tensor_id(), param_id);
    let _h = param.content_hash().unwrap(); // Just verify it computes.
}

#[test]
fn multi_step_training_loop_preserves_parameter_identity() {
    // 3 training steps; verify AdamW's per-parameter state advances
    // exactly once per step AND parameter identity is stable.
    let weight_fwd = kt::Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
    let weight_master = kt::Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
    let mut param = Parameter::trainable(
        ForwardStorage::Plain(weight_fwd.clone()),
        weight_master,
        AmpPolicy::fp32_reference(),
    );
    let param_id = param.tensor_id();

    let mut opt = AdamW::default_hp();
    let mut tape = Tape::new();
    let grad = kt::Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
    let apply_count = Arc::new(AtomicUsize::new(0));

    for expected_step in 1..=3 {
        // Forward (toy): identity on the weight.
        let out = weight_fwd.clone();
        tape.record(
            &out,
            &[&weight_fwd],
            Box::new(PassthroughBwd {
                name: "identity",
                input_shapes: vec![weight_fwd.shape().to_vec()],
                apply_count: apply_count.clone(),
            }),
        );
        // Backward → store.
        let seed = kt::Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let _store = tape
            .backward(out.id(), seed, |a, b| kt::ops::add(a, b))
            .unwrap();
        // Optimizer step.
        opt.step(&mut param, &grad).unwrap();
        // Verify state advances exactly once per step.
        assert_eq!(opt.moments(param_id).unwrap().step, expected_step);
        // Parameter identity is stable.
        assert_eq!(param.tensor_id(), param_id);
        // Clear tape for next step.
        tape.clear();
    }
    // PassthroughBwd ran 3 times total.
    assert_eq!(apply_count.load(Ordering::SeqCst), 3);
}
