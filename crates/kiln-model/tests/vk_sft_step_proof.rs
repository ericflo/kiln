//! PR5f — bounded single SFT-style training step on `Device::Vulkan` through the
//! harmonized kt-tape substrate (#1082).
//!
//! CAPSTONE of the PR1–PR5 Vulkan kt-tape harmonization. PR5b/PR5d/PR5e proved
//! the *substrate*: a kt forward RECORDS onto `kiln_autograd::Tape` on
//! `Device::Vulkan` and `Tape::backward` produces correct grads over Vulkan
//! storage (add, rms_norm). This file proves the *SFT training step itself*
//! runs end-to-end on Vulkan through that substrate, in three escalating tiers:
//!
//!   TIER 1 — `try_tape_lora_linear_kt` forward records the linear+LoRA composite
//!            on `Device::Vulkan(0)`; `Tape::backward` from the summed output
//!            yields FINITE, correctly-shaped grads for the LoRA A and B leaves.
//!   TIER 2 — `try_tape_cross_entropy_from_logits_kt` as the scalar SFT loss head
//!            over the LoRA-linear logits; backward from the CE scalar -> finite
//!            LoRA grads. This is the real SFT forward+loss+backward on Vulkan.
//!   TIER 3 — ONE `dispatch_adamw_step_buffers` step on the LoRA A/B param
//!            buffers using the tier-2 grads; assert the param BYTES changed
//!            (read-back before/after) and stayed finite. One complete SFT
//!            training step on Vulkan through the harmonized path.
//!
//! HOST-SAFETY (the host has hard-crashed on long GPU runs): each tier is a
//! SINGLE bounded forward(+loss)+backward(+1 optimizer step) over TINY tensors
//! (seq=2, hidden=4, out/vocab=4, rank=2). NO training loop, NO multi-step
//! iteration, NO checkpoint load. Self-skips unless `KILN_TENSOR_VULKAN_TEST=1`
//! AND a Vulkan device is present (mirrors PR2–PR5 gating). Run named,
//! single-shot, one test at a time:
//!
//!     KILN_TENSOR_VULKAN_TEST=1 \
//!       CARGO_TARGET_DIR=/path/to/kiln/target \
//!       cargo test -p kiln-model --features vulkan \
//!       vk_sft_lora_linear_backprops -- --nocapture --test-threads=1

#![cfg(feature = "vulkan")]

use kiln_model::lora_loader::LoraProjectionWeights;
use kiln_model::tape_forward::{
    try_tape_cross_entropy_from_logits_kt, try_tape_lora_linear_kt, with_thread_local_tape,
};
use kiln_tensor::{Device, Tensor, VulkanStorage, ops};
use kiln_vulkan_kernel::VulkanDevice;

// ----------------------------------------------------------------------------
// Gate + tiny fixtures.
// ----------------------------------------------------------------------------

/// Bounded GPU run is opt-in: `KILN_TENSOR_VULKAN_TEST=1` AND a device present.
fn vk_enabled(test_name: &str) -> bool {
    if std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip {test_name}: KILN_TENSOR_VULKAN_TEST unset");
        return false;
    }
    if !VulkanDevice::probe() {
        eprintln!("skip {test_name}: no Vulkan device");
        return false;
    }
    true
}

// Tiny SFT shapes.
const SEQ: usize = 2; // sequence length (rows)
const HIDDEN: usize = 4; // in_features (k)
const OUT: usize = 4; // out_features (n) — also the vocab for the CE head
const RANK: usize = 2; // LoRA rank
const LORA_SCALE: f32 = 2.0;

fn read_host_f32(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .expect("D2H")
        .to_vec::<f32>()
        .expect("readback")
}

/// Build the tiny SFT fixtures on `Device::Vulkan(0)`:
///   x        : [SEQ, HIDDEN]      (the residual stream input)
///   weight_t : [HIDDEN, OUT]      (FROZEN base weight, transposed)
///   lora.a   : [RANK, HIDDEN]     (trainable LoRA A)
///   lora.b   : [OUT, RANK]        (trainable LoRA B)
fn build_lora_fixtures() -> (Tensor, Tensor, LoraProjectionWeights) {
    let x_data: Vec<f32> = (0..SEQ * HIDDEN).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_data: Vec<f32> = (0..HIDDEN * OUT).map(|i| 0.05 * (i as f32) - 0.2).collect();
    // LoRA A nonzero so the (x@Aᵀ) inner activation is nonzero.
    let a_data: Vec<f32> = (0..RANK * HIDDEN)
        .map(|i| 0.07 * (i as f32) - 0.1)
        .collect();
    // LoRA B nonzero (real adapters init B=0, but a nonzero B exercises the full
    // delta path and gives a nonzero dA as well as dB).
    let b_data: Vec<f32> = (0..OUT * RANK).map(|i| 0.03 * (i as f32) + 0.02).collect();

    let x = Tensor::from_vec_on(Device::Vulkan(0), x_data, vec![SEQ, HIDDEN]).expect("x on Vulkan");
    let weight_t = Tensor::from_vec_on(Device::Vulkan(0), w_data, vec![HIDDEN, OUT])
        .expect("weight_t on Vulkan");
    let a =
        Tensor::from_vec_on(Device::Vulkan(0), a_data, vec![RANK, HIDDEN]).expect("a on Vulkan");
    let b = Tensor::from_vec_on(Device::Vulkan(0), b_data, vec![OUT, RANK]).expect("b on Vulkan");
    (x, weight_t, LoraProjectionWeights { a, b })
}

/// Same fixtures as [`build_lora_fixtures`] but x is rank-3 `[1, SEQ, HIDDEN]`.
/// `try_tape_lora_linear_kt` preserves leading dims, so the recorded output is
/// `[1, SEQ, OUT]` — exactly the `[1, T, V]` shape the CE recorder wants, with
/// the recorder's own trailing `ReshapeBackward` keeping the tape chain
/// connected (no unrecorded reshape between the LoRA-linear output and the CE
/// node, which would dead-end the backward walk at an unrooted tensor id).
fn build_lora_fixtures_3d() -> (Tensor, Tensor, LoraProjectionWeights) {
    let (x2d, weight_t, lora) = build_lora_fixtures();
    let x3d = x2d
        .reshape(vec![1, SEQ, HIDDEN])
        .expect("reshape x -> [1, SEQ, HIDDEN]");
    (x3d, weight_t, lora)
}

// ----------------------------------------------------------------------------
// TIER 1 — LoRA-linear records on Vulkan; backward yields finite LoRA grads.
// ----------------------------------------------------------------------------

/// TIER 1: `try_tape_lora_linear_kt` on `Device::Vulkan(0)` records the
/// linear+LoRA composite (ReshapeBackward, MatmulBackward, LoraDeltaAddBackward,
/// ReshapeBackward) onto the thread-local `Tape`, and `Tape::backward` from the
/// summed output produces FINITE, correctly-shaped grads for the trainable LoRA
/// A `[RANK, HIDDEN]` and B `[OUT, RANK]` leaves over Vulkan storage.
#[test]
fn vk_sft_lora_linear_backprops() {
    let test_name = "vk_sft_lora_linear_backprops";
    if !vk_enabled(test_name) {
        return;
    }
    let (x, weight_t, lora) = build_lora_fixtures();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());

    let (out, tape) = with_thread_local_tape(|| {
        try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None — recorder did NOT record on Vulkan")
    });

    // Recorder fired: reshape(x), matmul, lora-delta-add, reshape(out) = 4 nodes.
    assert_eq!(
        tape.len(),
        4,
        "LoRA-linear recorded {} nodes on Vulkan, expected 4",
        tape.len()
    );
    assert_eq!(
        out.device(),
        Device::Vulkan(0),
        "forward output left Vulkan"
    );
    assert_eq!(out.shape(), &[SEQ, OUT], "LoRA-linear output wrong shape");
    let out_v = read_host_f32(&out);
    assert!(
        out_v.iter().all(|v| v.is_finite()),
        "non-finite LoRA-linear forward: {out_v:?}"
    );

    // Backward: seed dL/dout = ones (sum-of-output scalar loss).
    let seed = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32; SEQ * OUT], vec![SEQ, OUT])
        .expect("seed on Vulkan");
    let grads = tape
        .backward(out.id(), seed, ops::add)
        .expect("Tape::backward errored on the LoRA-linear Vulkan graph");

    let da = grads.get(a_id).expect("no grad keyed on lora.a.id()");
    let db = grads.get(b_id).expect("no grad keyed on lora.b.id()");
    assert_eq!(da.shape(), &[RANK, HIDDEN], "dL/dA wrong shape");
    assert_eq!(db.shape(), &[OUT, RANK], "dL/dB wrong shape");

    let da_v = read_host_f32(da);
    let db_v = read_host_f32(db);
    assert!(
        da_v.iter().chain(db_v.iter()).all(|v| v.is_finite()),
        "non-finite LoRA grads: dA={da_v:?} dB={db_v:?}"
    );
    eprintln!(
        "[PR5f TIER1 PASS] Device::Vulkan(0): tape.len()={} | out={:?} | dA={:?} | dB={:?}",
        tape.len(),
        out_v,
        da_v,
        db_v
    );
}

// ----------------------------------------------------------------------------
// TIER 2 — real SFT forward+loss+backward: LoRA-linear logits -> CE scalar loss.
// ----------------------------------------------------------------------------

/// TIER 2: the real SFT loss path on Vulkan. `try_tape_lora_linear_kt` produces
/// logits `[SEQ, OUT]` (OUT == vocab); reshape to `[1, SEQ, OUT]` and feed
/// `try_tape_cross_entropy_from_logits_kt` as the scalar loss head. `Tape::backward`
/// from the CE scalar produces finite LoRA A/B grads over Vulkan storage. Returns
/// `(loss, da, db)` so TIER 3 can reuse the exact same grads for an optimizer step.
fn run_sft_forward_loss_backward() -> (f32, Vec<f32>, Vec<f32>, LoraProjectionWeights) {
    let (x, weight_t, lora) = build_lora_fixtures_3d();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());

    // input_ids over the OUT-sized vocab; all-true label mask (every shifted
    // next-token position is supervised). seq>=2 required by the CE recorder.
    let input_ids: Vec<u32> = vec![1, 3];
    let label_mask: Vec<bool> = vec![true; SEQ];

    let (loss, tape) = with_thread_local_tape(|| {
        // x is [1, SEQ, HIDDEN] -> recorder emits logits [1, SEQ, OUT] directly,
        // the [1, T, V] shape the CE recorder wants, tape-connected via the
        // recorder's own trailing ReshapeBackward (no unrecorded reshape).
        let logits = try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None on Vulkan");
        try_tape_cross_entropy_from_logits_kt(&logits, &input_ids, &label_mask)
            .expect("try_tape_cross_entropy_from_logits_kt errored")
            .expect("CE recorder returned None on Vulkan")
    });

    assert_eq!(loss.device(), Device::Vulkan(0), "CE loss left Vulkan");
    let loss_v = read_host_f32(&loss);
    assert_eq!(loss_v.len(), 1, "CE loss is not scalar: {loss_v:?}");
    let loss_scalar = loss_v[0];
    assert!(loss_scalar.is_finite(), "non-finite CE loss: {loss_scalar}");
    assert!(
        tape.len() >= 5,
        "expected >=5 recorded nodes (4 LoRA-linear + >=1 CE), got {}",
        tape.len()
    );

    // Backward from the scalar CE loss. Seed dL/dloss = 1 (scalar).
    let seed = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32], vec![1])
        .expect("scalar seed on Vulkan");
    let grads = tape
        .backward(loss.id(), seed, ops::add)
        .expect("Tape::backward errored on the CE+LoRA Vulkan graph");

    let da = grads.get(a_id).expect("no grad keyed on lora.a.id()");
    let db = grads.get(b_id).expect("no grad keyed on lora.b.id()");
    assert_eq!(da.shape(), &[RANK, HIDDEN], "dL/dA wrong shape");
    assert_eq!(db.shape(), &[OUT, RANK], "dL/dB wrong shape");
    let da_v = read_host_f32(da);
    let db_v = read_host_f32(db);
    assert!(
        da_v.iter().chain(db_v.iter()).all(|v| v.is_finite()),
        "non-finite SFT LoRA grads: dA={da_v:?} dB={db_v:?}"
    );
    (loss_scalar, da_v, db_v, lora)
}

#[test]
fn vk_sft_cross_entropy_backprops() {
    let test_name = "vk_sft_cross_entropy_backprops";
    if !vk_enabled(test_name) {
        return;
    }
    let (loss, da_v, db_v, _lora) = run_sft_forward_loss_backward();
    eprintln!(
        "[PR5f TIER2 PASS] Device::Vulkan(0): SFT CE loss={:.6} | dA={:?} | dB={:?}",
        loss, da_v, db_v
    );
}

// ----------------------------------------------------------------------------
// TIER 3 — one complete SFT training step: AdamW updates the LoRA params on GPU.
// ----------------------------------------------------------------------------

/// Borrow the `&VulkanBuffer` + `&VulkanDevice` behind a Vulkan-resident tensor.
fn vk_buffer_of(t: &Tensor) -> &VulkanStorage {
    t.storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .expect("tensor is not Vulkan-backed")
}

/// Run ONE AdamW step in place on `param`'s device buffer using `grad`'s device
/// buffer, with freshly-zeroed m/v device buffers. Asserts the param BYTES
/// changed and stayed finite. Returns the param after the step (read back).
fn adamw_one_step_in_place(param: &Tensor, grad: &Tensor, before: &[f32]) -> Vec<f32> {
    let n = param.element_count();
    assert_eq!(grad.element_count(), n, "param/grad element-count mismatch");
    let pstore = vk_buffer_of(param);
    let gstore = vk_buffer_of(grad);
    let device = pstore.vulkan_device().clone();

    // Fresh zero-initialized m/v device buffers (one optimizer step => step=1).
    let m_store = VulkanStorage::zeros(device.clone(), 0, param.dtype(), n).expect("m zeros");
    let v_store = VulkanStorage::zeros(device.clone(), 0, param.dtype(), n).expect("v zeros");

    kiln_model::backend::vulkan::dispatch_adamw_step_buffers(
        &device,
        pstore.buffer(),
        gstore.buffer(),
        m_store.buffer(),
        v_store.buffer(),
        n,
        1e-2,  // lr — large enough to move the bytes in one step
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // eps
        0.0,   // weight_decay
        1,     // step
    )
    .expect("dispatch_adamw_step_buffers failed");

    let after = read_host_f32(param);
    assert_eq!(after.len(), before.len(), "param len changed");
    assert!(
        after.iter().all(|v| v.is_finite()),
        "non-finite param after AdamW: {after:?}"
    );
    after
}

#[test]
fn vk_sft_one_adamw_step_changes_params() {
    let test_name = "vk_sft_one_adamw_step_changes_params";
    if !vk_enabled(test_name) {
        return;
    }
    // Reuse the exact tier-2 SFT forward+loss+backward to get real grads.
    // We need the grad *tensors* (not just host vecs) on-device for the AdamW
    // dispatch, so re-run the graph here capturing the grad tensors.
    let (x, weight_t, lora) = build_lora_fixtures_3d();
    let (a_id, b_id) = (lora.a.id(), lora.b.id());
    let input_ids: Vec<u32> = vec![1, 3];
    let label_mask: Vec<bool> = vec![true; SEQ];

    let (loss, tape) = with_thread_local_tape(|| {
        let logits = try_tape_lora_linear_kt(&x, &weight_t, Some(&lora), LORA_SCALE)
            .expect("try_tape_lora_linear_kt errored")
            .expect("try_tape_lora_linear_kt returned None on Vulkan");
        try_tape_cross_entropy_from_logits_kt(&logits, &input_ids, &label_mask)
            .expect("CE recorder errored")
            .expect("CE recorder returned None on Vulkan")
    });
    let loss_scalar = read_host_f32(&loss)[0];
    assert!(loss_scalar.is_finite(), "non-finite CE loss: {loss_scalar}");

    let seed = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0_f32], vec![1]).expect("seed");
    let grads = tape
        .backward(loss.id(), seed, ops::add)
        .expect("Tape::backward errored");
    let da = grads.get(a_id).expect("no dA").clone();
    let db = grads.get(b_id).expect("no dB").clone();

    // The grads MUST be on Vulkan for the in-place GPU AdamW dispatch.
    assert_eq!(da.device(), Device::Vulkan(0), "dA left Vulkan");
    assert_eq!(db.device(), Device::Vulkan(0), "dB left Vulkan");

    // Snapshot params BEFORE the step (D2H). The LoRA leaves share storage with
    // the recorder's clones, so the in-place GPU update is visible here.
    let a_before = read_host_f32(&lora.a);
    let b_before = read_host_f32(&lora.b);

    // ONE AdamW step on each LoRA param, in place on the GPU buffer.
    let a_after = adamw_one_step_in_place(&lora.a, &da, &a_before);
    let b_after = adamw_one_step_in_place(&lora.b, &db, &b_before);

    // The bytes must have CHANGED (a real optimizer update happened on GPU).
    let a_delta = a_before
        .iter()
        .zip(a_after.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    let b_delta = b_before
        .iter()
        .zip(b_after.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        a_delta > 0.0,
        "AdamW did NOT change LoRA A bytes: before={a_before:?} after={a_after:?}"
    );
    assert!(
        b_delta > 0.0,
        "AdamW did NOT change LoRA B bytes: before={b_before:?} after={b_after:?}"
    );

    eprintln!(
        "[PR5f TIER3 PASS] one full SFT step on Device::Vulkan(0): \
         CE loss={:.6}\n  LoRA A: before={:?} -> after={:?} (max |Δ|={:.3e})\n  \
         LoRA B: before={:?} -> after={:?} (max |Δ|={:.3e})",
        loss_scalar, a_before, a_after, a_delta, b_before, b_after, b_delta
    );
}
