//! Bounded single SFT-style training step on `Device::Metal` through the kt-tape
//! substrate.
//!
//! Proves the SFT training step runs end-to-end on Metal in three tiers:
//!   TIER 1 - `try_tape_lora_linear_kt` records the linear+LoRA composite and
//!            backward yields finite, correctly-shaped LoRA A/B grads.
//!   TIER 2 - `try_tape_cross_entropy_from_logits_kt` adds the scalar SFT loss
//!            head and backward yields finite LoRA grads.
//!   TIER 3 - one resident `MetalBackend::dispatch_adamw_step` updates the LoRA
//!            params on device.
//!
//! Tiny tensors (seq=2, hidden=4, vocab=4, rank=2), single bounded step, no
//! training loop, no checkpoint load. Skips unless a Metal device is present.
//!
//! Run on Apple Silicon:
//! `cargo test -p kiln-model --features metal --test metal_sft_step_proof -- --nocapture --test-threads=1`
#![cfg(feature = "metal")]

use kiln_model::backend::metal::MetalBackend;
use kiln_model::backend::{OptimizerBackend, ResidencyBackend};
use kiln_model::lora_loader::LoraProjectionWeights;
use kiln_model::tape_forward::{
    try_tape_cross_entropy_from_logits_kt, try_tape_lora_linear_kt, with_thread_local_tape,
};
use kiln_tensor::{DType, Device, Tensor, ops};

const SEQ: usize = 2;
const HIDDEN: usize = 4;
const OUT: usize = 4;
const RANK: usize = 2;
const LORA_SCALE: f32 = 2.0;

fn metal_enabled(test: &str) -> bool {
    if kiln_tensor::primary_metal_companion(0).is_err() {
        if std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1") {
            panic!("Metal device unavailable while KILN_QUALIFICATION=1 ({test})");
        }
        eprintln!("skip {test}: no Metal device");
        return false;
    }
    true
}

fn read_host_f32(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .expect("D2H")
        .to_vec::<f32>()
        .expect("readback")
}

fn build_lora_fixtures() -> (Tensor, Tensor, LoraProjectionWeights) {
    let x_data: Vec<f32> = (0..SEQ * HIDDEN).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_data: Vec<f32> = (0..HIDDEN * OUT).map(|i| 0.05 * (i as f32) - 0.2).collect();
    let a_data: Vec<f32> = (0..RANK * HIDDEN)
        .map(|i| 0.07 * (i as f32) - 0.1)
        .collect();
    let b_data: Vec<f32> = (0..OUT * RANK).map(|i| 0.03 * (i as f32) + 0.02).collect();
    let x = Tensor::from_vec_on(Device::Metal(0), x_data, vec![SEQ, HIDDEN]).expect("x");
    let weight_t =
        Tensor::from_vec_on(Device::Metal(0), w_data, vec![HIDDEN, OUT]).expect("weight_t");
    let a = Tensor::from_vec_on(Device::Metal(0), a_data, vec![RANK, HIDDEN]).expect("a");
    let b = Tensor::from_vec_on(Device::Metal(0), b_data, vec![OUT, RANK]).expect("b");
    (x, weight_t, LoraProjectionWeights { a, b })
}

fn build_lora_fixtures_3d() -> (Tensor, Tensor, LoraProjectionWeights) {
    let (x2d, weight_t, lora) = build_lora_fixtures();
    let x3d = x2d.reshape(vec![1, SEQ, HIDDEN]).expect("reshape x");
    (x3d, weight_t, lora)
}

#[test]
fn metal_sft_lora_linear_backprops() {
    if !metal_enabled("metal_sft_lora_linear_backprops") {
        return;
    }
    let (x, weight_t, lora) = build_lora_fixtures();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());

    let (out, tape) = with_thread_local_tape(|| {
        try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None on Metal")
    });

    assert_eq!(
        tape.len(),
        4,
        "LoRA-linear recorded {} nodes, expected 4",
        tape.len()
    );
    assert_eq!(out.device(), Device::Metal(0), "forward output left Metal");
    assert_eq!(out.shape(), &[SEQ, OUT], "LoRA-linear output wrong shape");
    let out_v = read_host_f32(&out);
    assert!(
        out_v.iter().all(|v| v.is_finite()),
        "non-finite forward: {out_v:?}"
    );

    let seed = Tensor::from_vec_on(Device::Metal(0), vec![1.0_f32; SEQ * OUT], vec![SEQ, OUT])
        .expect("seed");
    let grads = tape
        .backward(out.id(), seed, |g, z| ops::add(g, z))
        .expect("Tape::backward errored on the LoRA-linear Metal graph");
    let da = grads.get(a_id).expect("no grad keyed on lora.a.id()");
    let db = grads.get(b_id).expect("no grad keyed on lora.b.id()");
    assert_eq!(da.shape(), &[RANK, HIDDEN], "dL/dA wrong shape");
    assert_eq!(db.shape(), &[OUT, RANK], "dL/dB wrong shape");
    let (da_v, db_v) = (read_host_f32(da), read_host_f32(db));
    assert!(
        da_v.iter().chain(db_v.iter()).all(|v| v.is_finite()),
        "non-finite LoRA grads: dA={da_v:?} dB={db_v:?}"
    );
    eprintln!(
        "[Metal TIER1 PASS] tape.len()={} out={out_v:?} dA={da_v:?} dB={db_v:?}",
        tape.len()
    );
}

fn run_sft_forward_loss_backward() -> (f32, Tensor, Tensor, LoraProjectionWeights) {
    let (x, weight_t, lora) = build_lora_fixtures_3d();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());
    let input_ids: Vec<u32> = vec![1, 3];
    let label_mask: Vec<bool> = vec![true; SEQ];

    let (loss, tape) = with_thread_local_tape(|| {
        let logits = try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None on Metal");
        try_tape_cross_entropy_from_logits_kt(&logits, &input_ids, &label_mask)
            .expect("try_tape_cross_entropy_from_logits_kt errored")
            .expect("CE recorder returned None on Metal")
    });

    assert_eq!(loss.device(), Device::Metal(0), "CE loss left Metal");
    let loss_v = read_host_f32(&loss);
    assert_eq!(loss_v.len(), 1, "CE loss is not scalar: {loss_v:?}");
    let loss_scalar = loss_v[0];
    assert!(loss_scalar.is_finite(), "non-finite CE loss: {loss_scalar}");
    assert!(tape.len() >= 5, "expected >=5 nodes, got {}", tape.len());

    let seed = Tensor::from_vec_on(Device::Metal(0), vec![1.0_f32], vec![1]).expect("scalar seed");
    let grads = tape
        .backward(loss.id(), seed, |g, z| ops::add(g, z))
        .expect("Tape::backward errored on the CE+LoRA Metal graph");
    let da = grads.get(a_id).expect("no dA").clone();
    let db = grads.get(b_id).expect("no dB").clone();
    let (da_v, db_v) = (read_host_f32(&da), read_host_f32(&db));
    assert!(
        da_v.iter().chain(db_v.iter()).all(|v| v.is_finite()),
        "non-finite SFT LoRA grads: dA={da_v:?} dB={db_v:?}"
    );
    (loss_scalar, da, db, lora)
}

#[test]
fn metal_sft_cross_entropy_backprops() {
    if !metal_enabled("metal_sft_cross_entropy_backprops") {
        return;
    }
    let (loss, da, db, _lora) = run_sft_forward_loss_backward();
    eprintln!(
        "[Metal TIER2 PASS] SFT CE loss={loss:.6} dA={:?} dB={:?}",
        read_host_f32(&da),
        read_host_f32(&db)
    );
}

fn adamw_one_step_in_place(
    backend: &MetalBackend,
    param: &Tensor,
    grad: &Tensor,
    before: &[f32],
) -> Vec<f32> {
    let n = param.element_count();
    assert_eq!(grad.element_count(), n, "param/grad element-count mismatch");
    let m = Tensor::zeros_on(Device::Metal(0), param.dims().to_vec(), DType::F32).expect("m zeros");
    let v = Tensor::zeros_on(Device::Metal(0), param.dims().to_vec(), DType::F32).expect("v zeros");
    ResidencyBackend::runtime_register_resident_activation(backend, param).expect("register param");
    ResidencyBackend::runtime_register_resident_activation(backend, grad).expect("register grad");
    ResidencyBackend::runtime_register_resident_activation(backend, &m).expect("register m");
    ResidencyBackend::runtime_register_resident_activation(backend, &v).expect("register v");
    let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
        backend, param, grad, &m, &v, 1e-2, 0.9, 0.999, 1e-8, 0.0, 1,
    )
    .expect("dispatch_adamw_step failed");
    assert!(
        dispatched,
        "Metal AdamW should dispatch on resident tensors"
    );
    ResidencyBackend::runtime_evict_resident_activation(backend, grad);
    ResidencyBackend::runtime_evict_resident_activation(backend, &m);
    ResidencyBackend::runtime_evict_resident_activation(backend, &v);

    let after = read_host_f32(param);
    assert_eq!(after.len(), before.len(), "param len changed");
    assert!(
        after.iter().all(|v| v.is_finite()),
        "non-finite param after AdamW: {after:?}"
    );
    after
}

#[test]
fn metal_sft_one_adamw_step_changes_params() {
    if !metal_enabled("metal_sft_one_adamw_step_changes_params") {
        return;
    }
    let (loss, da, db, lora) = run_sft_forward_loss_backward();
    assert!(loss.is_finite(), "non-finite CE loss: {loss}");
    assert_eq!(da.device(), Device::Metal(0), "dA left Metal");
    assert_eq!(db.device(), Device::Metal(0), "dB left Metal");

    let backend = MetalBackend::new(Device::Metal(0));
    let a_before = read_host_f32(&lora.a);
    let b_before = read_host_f32(&lora.b);
    let a_after = adamw_one_step_in_place(&backend, &lora.a, &da, &a_before);
    let b_after = adamw_one_step_in_place(&backend, &lora.b, &db, &b_before);

    let max_abs_delta = |before: &[f32], after: &[f32]| {
        before
            .iter()
            .zip(after)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    };
    let a_delta = max_abs_delta(&a_before, &a_after);
    let b_delta = max_abs_delta(&b_before, &b_after);
    assert!(
        a_delta > 0.0,
        "AdamW did not change LoRA A: before={a_before:?} after={a_after:?}"
    );
    assert!(
        b_delta > 0.0,
        "AdamW did not change LoRA B: before={b_before:?} after={b_after:?}"
    );

    eprintln!(
        "[Metal TIER3 PASS] one full SFT step on Device::Metal(0): CE loss={loss:.6}\n  \
         LoRA A max|delta|={a_delta:.3e}  LoRA B max|delta|={b_delta:.3e}"
    );
}
