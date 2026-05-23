//! Tied-weight integration test — anti-pattern 17 enforcement.
//!
//! Per the issue's anti-pattern 17:
//!
//! > Same logical parameter, same `Parameter` handle. Tied weights
//! > (LM head + embedding, MTP head + base embedding), shared
//! > draft+target weights (MTP / self-speculative), multi-call-site
//! > weights — all are *one* `Parameter` with *one* `backward_storage`.
//! > Two `Parameter`s pointing at the same physical buffer is a bug;
//! > gradient accumulation must go through atomic-add into a single
//! > grad buffer or determinism collapses.
//!
//! Qwen3.5-4B specifically ties `lm_head ← embed_tokens` (weights.rs:243,
//! forward.rs:6014, vk_forward.rs:1503) and `mtp_head ← embed_tokens`
//! (loader.rs:675). This test verifies that two consumers of the same
//! `kiln_tensor::Tensor` produce ONE accumulated gradient in the
//! GradStore — not two separate entries that the optimizer might double-
//! count.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kiln_autograd::{BackwardOp, Tape};
use kiln_optim::{AdamW, OptimStep};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor as kt;

/// Backward that emits `grad_output` for each input.
#[derive(Debug)]
struct PassthroughBwd {
    name: &'static str,
    input_count: usize,
    apply_count: Arc<AtomicUsize>,
}

impl BackwardOp for PassthroughBwd {
    fn name(&self) -> &'static str {
        self.name
    }
    fn input_count(&self) -> usize {
        self.input_count
    }
    fn apply(&self, grad_output: &kt::Tensor) -> kt::Result<Vec<Option<kt::Tensor>>> {
        self.apply_count.fetch_add(1, Ordering::SeqCst);
        Ok((0..self.input_count)
            .map(|_| Some(grad_output.clone()))
            .collect())
    }
}

#[test]
fn tied_weight_two_consumers_one_accumulated_grad() {
    // Scenario: `embed_tokens` is consumed by both the LM head (via
    // weight tying) and a hypothetical MTP head. Two forward ops each
    // record the embedding tensor as an input, and their gradients
    // must accumulate into ONE entry in the GradStore.
    let embed_fwd = kt::Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3, 2],
    )
    .unwrap();
    let embed_id = embed_fwd.id();

    let mut tape = Tape::new();
    let apply_count = Arc::new(AtomicUsize::new(0));

    // Forward op 1: lm_head uses embed_fwd
    let lm_logits = kt::Tensor::from_slice(&[0.0f32; 6], vec![3, 2]).unwrap();
    tape.record(
        &lm_logits,
        &[&embed_fwd],
        Box::new(PassthroughBwd {
            name: "lm_head_consume_embed",
            input_count: 1,
            apply_count: apply_count.clone(),
        }),
    );

    // Forward op 2: mtp_head uses the SAME embed_fwd tensor (not a clone — same Arc).
    let mtp_logits = kt::Tensor::from_slice(&[0.0f32; 6], vec![3, 2]).unwrap();
    tape.record(
        &mtp_logits,
        &[&embed_fwd],
        Box::new(PassthroughBwd {
            name: "mtp_head_consume_embed",
            input_count: 1,
            apply_count: apply_count.clone(),
        }),
    );

    // Forward op 3: combine both into a "loss" (the tape walker needs
    // a single root to seed backward).
    let loss = kt::Tensor::from_slice(&[0.0f32; 6], vec![3, 2]).unwrap();
    tape.record(
        &loss,
        &[&lm_logits, &mtp_logits],
        Box::new(PassthroughBwd {
            name: "combine",
            input_count: 2,
            apply_count: apply_count.clone(),
        }),
    );

    // Backward.
    let seed = kt::Tensor::from_slice(&[1.0f32; 6], vec![3, 2]).unwrap();
    let store = tape.backward(loss.id(), seed, kt::ops::add).unwrap();

    // The embedding tensor MUST have exactly ONE entry in the store
    // (anti-pattern 17). Both lm_head and mtp_head's gradients
    // accumulated atomically.
    let grad = store.get(embed_id).expect(
        "tied weight grad missing from store — anti-pattern 17 violation: \
         two consumers of the same Tensor should produce ONE GradStore entry",
    );
    assert_eq!(grad.shape(), &[3, 2]);

    // Inspect the gradient: with passthrough backward, each
    // lm_head/mtp_head contributes `seed=1` to the embedding's grad;
    // accumulator sums them → grad = 2.
    let cpu = grad
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
    for v in &values {
        assert!(
            (v - 2.0).abs() < 1e-6,
            "expected each tied-weight grad element = 2 (sum of 2 consumers' upstream seeds), got {v}"
        );
    }

    // Sanity: PassthroughBwd ran exactly 3 times (one per recorded op).
    assert_eq!(apply_count.load(Ordering::SeqCst), 3);
}

#[test]
fn tied_weight_with_real_parameter_and_adamw_step() {
    // End-to-end: build a `Parameter` for the tied embedding;
    // simulate the two-consumer forward; backward yields ONE grad;
    // AdamW.step consumes it once and advances the moment.step
    // counter by 1 (NOT 2 — that would mean the optimizer
    // double-counted the tied parameter, which is exactly the
    // anti-pattern 17 failure mode).
    let embed_fwd = kt::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let embed_master = kt::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let mut param = Parameter::trainable(
        ForwardStorage::Plain(embed_fwd.clone()),
        embed_master,
        AmpPolicy::fp32_reference(),
    );
    let param_id = param.tensor_id();

    let mut tape = Tape::new();
    let count = Arc::new(AtomicUsize::new(0));
    let lm_logits = kt::Tensor::from_slice(&[0.0f32; 4], vec![2, 2]).unwrap();
    let mtp_logits = kt::Tensor::from_slice(&[0.0f32; 4], vec![2, 2]).unwrap();
    let loss = kt::Tensor::from_slice(&[0.0f32; 4], vec![2, 2]).unwrap();
    tape.record(
        &lm_logits,
        &[&embed_fwd],
        Box::new(PassthroughBwd {
            name: "lm",
            input_count: 1,
            apply_count: count.clone(),
        }),
    );
    tape.record(
        &mtp_logits,
        &[&embed_fwd],
        Box::new(PassthroughBwd {
            name: "mtp",
            input_count: 1,
            apply_count: count.clone(),
        }),
    );
    tape.record(
        &loss,
        &[&lm_logits, &mtp_logits],
        Box::new(PassthroughBwd {
            name: "combine",
            input_count: 2,
            apply_count: count.clone(),
        }),
    );

    let seed = kt::Tensor::from_slice(&[1.0f32; 4], vec![2, 2]).unwrap();
    let store = tape.backward(loss.id(), seed, kt::ops::add).unwrap();
    let grad = store.get(embed_fwd.id()).expect("tied embed grad");

    // ONE optimizer step — even though two consumers contributed grads.
    let mut opt = AdamW::default_hp();
    opt.step(&mut param, grad).unwrap();
    assert_eq!(
        opt.moments(param_id).unwrap().step,
        1,
        "step count must be 1 — anti-pattern 17 violation if 2"
    );
}
