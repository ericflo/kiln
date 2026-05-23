//! kiln-autograd end-to-end backward integration test.
//!
//! Exercises the tape + GradStore + BackwardOp triple on a real
//! op chain (using the Phase 1.15 `add` op as the accumulator) and
//! demonstrates:
//!
//! - Reverse-topo traversal: backward visits ops in reverse of
//!   forward record order.
//! - Grad accumulation: when one input is shared by multiple ops,
//!   gradients sum.
//! - Anti-pattern 16: in-place mutation between forward and backward
//!   surfaces as a typed error with the op name.

use kiln_autograd::{BackwardOp, Tape};
use kiln_tensor as kt;

/// Helper: `add` used as the gradient accumulator. Real op chains
/// in Phase 6a+ use `kiln_tensor::ops::add` directly; this test
/// re-states the closure shape for clarity.
fn accumulate(a: &kt::Tensor, b: &kt::Tensor) -> kt::Result<kt::Tensor> {
    kt::ops::add(a, b)
}

/// Trivial backward op that returns `grad_output` unchanged for each
/// input — the identity backward. Useful for testing tape mechanics
/// without dragging in real backward math.
#[derive(Debug)]
struct IdentityBackward {
    name: &'static str,
    input_count: usize,
}

impl BackwardOp for IdentityBackward {
    fn name(&self) -> &'static str {
        self.name
    }
    fn input_count(&self) -> usize {
        self.input_count
    }
    fn apply(&self, grad_output: &kt::Tensor) -> kt::Result<Vec<Option<kt::Tensor>>> {
        // Each input gets the same upstream grad.
        Ok((0..self.input_count)
            .map(|_| Some(grad_output.clone()))
            .collect())
    }
}

#[test]
fn backward_walks_in_reverse_topo_order_with_real_accumulator() {
    // Forward: a → op1 → b → op2 → c
    // Backward visits op2 then op1.
    let mut tape = Tape::new();
    let a = kt::Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
    let b = kt::Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
    let c = kt::Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();

    tape.record(
        &b,
        &[&a],
        Box::new(IdentityBackward {
            name: "op1",
            input_count: 1,
        }),
    );
    tape.record(
        &c,
        &[&b],
        Box::new(IdentityBackward {
            name: "op2",
            input_count: 1,
        }),
    );

    let seed = kt::Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
    let store = tape.backward(c.id(), seed.clone(), accumulate).unwrap();

    // Identity backward propagates the seed all the way back to `a`.
    let grad_a = store.get(a.id()).expect("a's grad");
    assert_eq!(grad_a.shape(), &[3]);
}

#[test]
fn backward_accumulates_grads_when_input_is_shared() {
    // Forward: a → op1 → b
    //          a → op2 → c
    //          (b, c) → op3 → d
    //
    // Backward: d's grad flows through op3 to both b and c (independently),
    // then through op1 and op2 back to a. The two streams of `a`'s grad
    // must be **summed** via the caller-supplied accumulator.
    let mut tape = Tape::new();
    let a = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    let b = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    let c = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    let d = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();

    tape.record(
        &b,
        &[&a],
        Box::new(IdentityBackward {
            name: "op1_a_to_b",
            input_count: 1,
        }),
    );
    tape.record(
        &c,
        &[&a],
        Box::new(IdentityBackward {
            name: "op2_a_to_c",
            input_count: 1,
        }),
    );
    tape.record(
        &d,
        &[&b, &c],
        Box::new(IdentityBackward {
            name: "op3_bc_to_d",
            input_count: 2,
        }),
    );

    let seed = kt::Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
    let store = tape.backward(d.id(), seed, accumulate).unwrap();

    // a's grad = grad_b + grad_c = seed + seed = 2 * seed.
    // With identity backward, grad_b = grad_c = grad_d = seed.
    let grad_a = store.get(a.id()).expect("a's grad");
    let cpu = grad_a
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let v = f32::from_le_bytes(cpu.as_bytes()[0..4].try_into().unwrap());
    assert!(
        (v - 2.0).abs() < 1e-6,
        "expected grad_a = 2 (1 + 1) via accumulator, got {v}"
    );
}

#[test]
fn backward_anti_pattern_16_detection_end_to_end() {
    // Real-world anti-pattern 16 scenario: optimizer step mutates a
    // parameter in place between forward and backward. The tape walker
    // must surface a typed error with the op name + input index.
    let mut tape = Tape::new();
    let weight = kt::Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let output = kt::Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
    tape.record(
        &output,
        &[&weight],
        Box::new(IdentityBackward {
            name: "matmul",
            input_count: 1,
        }),
    );

    // Simulate `optimizer.step(weight, grad)` between forward and
    // backward via `weight.bump_version()`.
    weight.bump_version();

    let seed = kt::Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
    let e = tape.backward(output.id(), seed, accumulate).unwrap_err();
    let msg = e.to_string();
    assert!(
        msg.contains("Anti-pattern 16"),
        "expected anti-pattern 16 error, got: {msg}"
    );
    assert!(msg.contains("matmul"), "should mention op name; got: {msg}");
    assert!(msg.contains("input 0"));
}

#[test]
fn backward_succeeds_when_no_mutation_between_forward_and_backward() {
    // Happy path: no version drift, backward completes without error.
    let mut tape = Tape::new();
    let x = kt::Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
    let y = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    tape.record(
        &y,
        &[&x],
        Box::new(IdentityBackward {
            name: "clean",
            input_count: 1,
        }),
    );
    let seed = kt::Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
    let _store = tape.backward(y.id(), seed, accumulate).unwrap();
}

#[test]
fn tape_clear_lets_subsequent_step_record_fresh() {
    // The "tape per training step" pattern: record forward, run
    // backward, clear, record next forward. Each step's tape is
    // independent.
    let mut tape = Tape::new();
    let x = kt::Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
    let y = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    tape.record(
        &y,
        &[&x],
        Box::new(IdentityBackward {
            name: "step1",
            input_count: 1,
        }),
    );
    assert_eq!(tape.len(), 1);

    let _ = tape
        .backward(y.id(), kt::Tensor::from_slice(&[1.0f32], vec![1]).unwrap(), accumulate)
        .unwrap();
    tape.clear();
    assert_eq!(tape.len(), 0);

    // Next step: record + backward again, no leftover state.
    let z = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    tape.record(
        &z,
        &[&x],
        Box::new(IdentityBackward {
            name: "step2",
            input_count: 1,
        }),
    );
    let _ = tape
        .backward(z.id(), kt::Tensor::from_slice(&[1.0f32], vec![1]).unwrap(), accumulate)
        .unwrap();
}

#[test]
fn reachable_from_walks_the_full_chain() {
    // Build: a → op1 → b → op2 → c → op3 → d
    // reachable_from(d.id()) → {a, b, c, d}
    let mut tape = Tape::new();
    let a = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    let b = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    let c = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    let d = kt::Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
    for (out, inp, name) in [
        (&b, &a, "op1"),
        (&c, &b, "op2"),
        (&d, &c, "op3"),
    ] {
        tape.record(
            out,
            &[inp],
            Box::new(IdentityBackward {
                name,
                input_count: 1,
            }),
        );
    }
    let reachable = tape.reachable_from(d.id());
    for id in [a.id(), b.id(), c.id(), d.id()] {
        assert!(reachable.contains(&id), "expected {id} in reachable set");
    }
}
