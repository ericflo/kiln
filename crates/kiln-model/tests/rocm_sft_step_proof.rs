//! R.9/E2E — bounded single SFT-style training step on `Device::Rocm` through the
//! kt-tape substrate. ROCm twin of `vk_sft_step_proof.rs`.
//!
//! Proves the SFT training step runs end-to-end on ROCm, in three escalating tiers:
//!   TIER 1 — `try_tape_lora_linear_kt` records the linear+LoRA composite on
//!            `Device::Rocm(0)`; `Tape::backward` from the summed output yields
//!            FINITE, correctly-shaped grads for the LoRA A and B leaves.
//!   TIER 2 — `try_tape_cross_entropy_from_logits_kt` as the scalar SFT loss head
//!            over the LoRA-linear logits; backward from the CE scalar → finite
//!            LoRA grads. The real SFT forward+loss+backward on ROCm.
//!   TIER 3 — ONE `adamw_step_f32_kt` step on the LoRA A/B params using the
//!            tier-2 grads; assert the param bytes CHANGED and stayed finite.
//!
//! Tiny tensors (seq=2, hidden=4, vocab=4, rank=2), single bounded step, no loop,
//! no model load. Skips unless a ROCm device is present.
//!
//! Run: `cargo test -p kiln-model --no-default-features --features rocm --test rocm_sft_step_proof -- --nocapture --test-threads=1`
#![cfg(feature = "rocm")]

use kiln_model::lora_loader::LoraProjectionWeights;
use kiln_model::tape_forward::{
    try_tape_cross_entropy_from_logits_kt, try_tape_lora_linear_kt, with_thread_local_tape,
};
use kiln_tensor::{ops, DType, Device, Tensor};

const SEQ: usize = 2;
const HIDDEN: usize = 4;
const OUT: usize = 4; // out_features / vocab
const RANK: usize = 2;
const LORA_SCALE: f32 = 2.0;

fn rocm_enabled(test: &str) -> bool {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("skip {test}: no ROCm device");
        return false;
    }
    true
}

fn read_host_f32(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).expect("D2H").to_vec::<f32>().expect("readback")
}

fn build_lora_fixtures() -> (Tensor, Tensor, LoraProjectionWeights) {
    let x_data: Vec<f32> = (0..SEQ * HIDDEN).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_data: Vec<f32> = (0..HIDDEN * OUT).map(|i| 0.05 * (i as f32) - 0.2).collect();
    let a_data: Vec<f32> = (0..RANK * HIDDEN).map(|i| 0.07 * (i as f32) - 0.1).collect();
    let b_data: Vec<f32> = (0..OUT * RANK).map(|i| 0.03 * (i as f32) + 0.02).collect();
    let x = Tensor::from_vec_on(Device::Rocm(0), x_data, vec![SEQ, HIDDEN]).expect("x");
    let weight_t = Tensor::from_vec_on(Device::Rocm(0), w_data, vec![HIDDEN, OUT]).expect("weight_t");
    let a = Tensor::from_vec_on(Device::Rocm(0), a_data, vec![RANK, HIDDEN]).expect("a");
    let b = Tensor::from_vec_on(Device::Rocm(0), b_data, vec![OUT, RANK]).expect("b");
    (x, weight_t, LoraProjectionWeights { a, b })
}

fn build_lora_fixtures_3d() -> (Tensor, Tensor, LoraProjectionWeights) {
    let (x2d, weight_t, lora) = build_lora_fixtures();
    let x3d = x2d.reshape(vec![1, SEQ, HIDDEN]).expect("reshape x");
    (x3d, weight_t, lora)
}

#[test]
fn rocm_sft_lora_linear_backprops() {
    if !rocm_enabled("rocm_sft_lora_linear_backprops") {
        return;
    }
    let (x, weight_t, lora) = build_lora_fixtures();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());

    let (out, tape) = with_thread_local_tape(|| {
        try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None — recorder did NOT record on ROCm")
    });

    assert_eq!(tape.len(), 4, "LoRA-linear recorded {} nodes, expected 4", tape.len());
    assert_eq!(out.device(), Device::Rocm(0), "forward output left ROCm");
    assert_eq!(out.shape(), &[SEQ, OUT], "LoRA-linear output wrong shape");
    let out_v = read_host_f32(&out);
    assert!(out_v.iter().all(|v| v.is_finite()), "non-finite forward: {out_v:?}");

    let seed = Tensor::from_vec_on(Device::Rocm(0), vec![1.0_f32; SEQ * OUT], vec![SEQ, OUT]).expect("seed");
    let grads = tape
        .backward(out.id(), seed, |g, z| ops::add(g, z))
        .expect("Tape::backward errored on the LoRA-linear ROCm graph");
    let da = grads.get(a_id).expect("no grad keyed on lora.a.id()");
    let db = grads.get(b_id).expect("no grad keyed on lora.b.id()");
    assert_eq!(da.shape(), &[RANK, HIDDEN], "dL/dA wrong shape");
    assert_eq!(db.shape(), &[OUT, RANK], "dL/dB wrong shape");
    let (da_v, db_v) = (read_host_f32(da), read_host_f32(db));
    assert!(
        da_v.iter().chain(db_v.iter()).all(|v| v.is_finite()),
        "non-finite LoRA grads: dA={da_v:?} dB={db_v:?}"
    );
    eprintln!("[ROCm TIER1 PASS] tape.len()={} out={out_v:?} dA={da_v:?} dB={db_v:?}", tape.len());
}

fn run_sft_forward_loss_backward() -> (f32, Tensor, Tensor, LoraProjectionWeights) {
    let (x, weight_t, lora) = build_lora_fixtures_3d();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());
    let input_ids: Vec<u32> = vec![1, 3];
    let label_mask: Vec<bool> = vec![true; SEQ];

    let (loss, tape) = with_thread_local_tape(|| {
        let logits = try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None on ROCm");
        try_tape_cross_entropy_from_logits_kt(&logits, &input_ids, &label_mask)
            .expect("try_tape_cross_entropy_from_logits_kt errored")
            .expect("CE recorder returned None on ROCm")
    });

    assert_eq!(loss.device(), Device::Rocm(0), "CE loss left ROCm");
    let loss_v = read_host_f32(&loss);
    assert_eq!(loss_v.len(), 1, "CE loss is not scalar: {loss_v:?}");
    let loss_scalar = loss_v[0];
    assert!(loss_scalar.is_finite(), "non-finite CE loss: {loss_scalar}");
    assert!(tape.len() >= 5, "expected >=5 nodes, got {}", tape.len());

    let seed = Tensor::from_vec_on(Device::Rocm(0), vec![1.0_f32], vec![1]).expect("scalar seed");
    let grads = tape
        .backward(loss.id(), seed, |g, z| ops::add(g, z))
        .expect("Tape::backward errored on the CE+LoRA ROCm graph");
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
fn rocm_sft_cross_entropy_backprops() {
    if !rocm_enabled("rocm_sft_cross_entropy_backprops") {
        return;
    }
    let (loss, da, db, _lora) = run_sft_forward_loss_backward();
    eprintln!(
        "[ROCm TIER2 PASS] SFT CE loss={loss:.6} dA={:?} dB={:?}",
        read_host_f32(&da),
        read_host_f32(&db)
    );
}

/// ONE AdamW step in place on a ROCm-resident param using its grad, with fresh
/// zeroed m/v. Asserts the param bytes changed and stayed finite.
fn adamw_one_step_in_place(param: &Tensor, grad: &Tensor, before: &[f32]) -> Vec<f32> {
    let n = param.element_count();
    assert_eq!(grad.element_count(), n, "param/grad element-count mismatch");
    let m = Tensor::zeros_on(Device::Rocm(0), param.dims().to_vec(), DType::F32).expect("m zeros");
    let v = Tensor::zeros_on(Device::Rocm(0), param.dims().to_vec(), DType::F32).expect("v zeros");
    // step=1 bias corrections: bc1 = 1-beta1^1, bc2 = 1-beta2^1.
    kiln_rmsnorm_kernel::adamw_step_f32_kt(
        param, grad, &m, &v, 1e-2, 0.9, 0.999, 1e-8, 0.0, 0.1, 0.001,
    )
    .expect("adamw_step_f32_kt failed");
    let after = read_host_f32(param);
    assert_eq!(after.len(), before.len(), "param len changed");
    assert!(after.iter().all(|v| v.is_finite()), "non-finite param after AdamW: {after:?}");
    after
}

#[test]
fn rocm_sft_one_adamw_step_changes_params() {
    if !rocm_enabled("rocm_sft_one_adamw_step_changes_params") {
        return;
    }
    let (loss, da, db, lora) = run_sft_forward_loss_backward();
    assert!(loss.is_finite(), "non-finite CE loss: {loss}");
    assert_eq!(da.device(), Device::Rocm(0), "dA left ROCm");
    assert_eq!(db.device(), Device::Rocm(0), "dB left ROCm");

    let a_before = read_host_f32(&lora.a);
    let b_before = read_host_f32(&lora.b);
    let a_after = adamw_one_step_in_place(&lora.a, &da, &a_before);
    let b_after = adamw_one_step_in_place(&lora.b, &db, &b_before);

    let max_abs_delta = |before: &[f32], after: &[f32]| {
        before.iter().zip(after).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    };
    let a_delta = max_abs_delta(&a_before, &a_after);
    let b_delta = max_abs_delta(&b_before, &b_after);
    assert!(a_delta > 0.0, "AdamW did NOT change LoRA A: before={a_before:?} after={a_after:?}");
    assert!(b_delta > 0.0, "AdamW did NOT change LoRA B: before={b_before:?} after={b_after:?}");

    eprintln!(
        "[ROCm TIER3 PASS] one full SFT step on Device::Rocm(0): CE loss={loss:.6}\n  \
         LoRA A max|Δ|={a_delta:.3e}  LoRA B max|Δ|={b_delta:.3e}"
    );
}
