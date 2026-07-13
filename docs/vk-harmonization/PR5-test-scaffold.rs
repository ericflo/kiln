//! PR5 forward-harmonization test scaffold — **HISTORICAL, DO NOT COPY AS CURRENT TEST CODE.**
//!
//! Issue #1082, branch `feat/vk-tape-harmonization`. This file is a *spec
//! companion* to `docs/vk-harmonization/PR5-spec.md`. It is NOT wired into any
//! Cargo target yet and is NOT expected to compile until PR5's recorder/gate
//! edits AND PR3's `vulkan_fwd` op coverage land. The implementer drops the
//! bounded tests (T1–T7) into `crates/kiln-model/tests/vk_tape_forward_parity.rs`
//! and the reachability smoke (§5.3) into
//! `crates/kiln-server/tests/real_model_integration.rs`.
//!
//! The `KILN_USE_TAPE_*` switches used when this scaffold was written have been
//! removed without aliases or replacement fields. Current recorders are
//! activated solely by `with_thread_local_tape`; an active scope must fail
//! closed when a required recorder cannot accept its operation envelope.
//!
//! Every test is BOUNDED (single op / single tiny forward / epochs:1) per the
//! host-safety ceiling (the dev host has hard-crashed on long runs). Each skips
//! gracefully when no Vulkan device is present, mirroring
//! `crates/kiln-model/tests/vk_resident_decode_parity.rs`.
//!
//! Acceptance thresholds (PR5-spec §5.0):
//!   * forward parity (Vulkan-kt vs vk_forward.rs), F32: max_abs_err ≤ 1e-5, max_rel_err ≤ 1e-4
//!   * FD backward parity, F32: max_abs_err ≤ 1e-3 with eps = 1e-3
//!
//! WIP markers: functions that cannot run until PR3/PR6 land are `#[ignore]`d
//! with a `// FRONTIER:` note naming the blocking op/wiring.

#![cfg(feature = "vulkan")]
#![allow(unused_imports, dead_code, clippy::missing_panics_doc)]

use anyhow::Result;
use kiln_model::backend;
use kiln_tensor::{DType, Device, Tensor};

// NOTE (WIP): under `--features vulkan` with PR5 landed, `tape_forward` is
// compiled in (PR5-spec §2.1). Until then this `use` does not resolve — that is
// expected; the module gate widen is part of PR5.
use kiln_model::tape_forward::{
    try_tape_cross_entropy_from_logits_kt, try_tape_gdn_recurrent_kt, try_tape_rms_norm_kt,
    try_tape_sdpa_fallback_kt, with_active_tape, with_thread_local_tape,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Skip-guard. Mirror of `vk_resident_decode_parity.rs`'s
/// `supports_resident_decode()` early-return and `real_model_integration.rs`'s
/// `try_new_metal().is_none()` skip. Returns `false` (skip) when no Vulkan
/// device is visible, so workspace `cargo test` on a CPU-only host still passes.
fn vulkan_available() -> bool {
    // FRONTIER: exact predicate name to confirm at impl time —
    // `kiln_model::backend::vulkan::vulkan_is_available()` is used by
    // `for_device_kt` (backend.rs:1340). Re-grep before relying on it.
    kiln_model::backend::vulkan::vulkan_is_available()
}

fn vk() -> Device {
    Device::Vulkan(0)
}

/// Build a fixed-seed F32 tensor on Vulkan. Vulkan trains F32 on the hot path
/// (PR5-spec §3.3) — do NOT use BF16 here. Depends on PR2's un-NYI
/// `from_vec_on(Device::Vulkan, ...)`.
fn vk_f32(values: Vec<f32>, shape: Vec<usize>) -> Result<Tensor> {
    Ok(Tensor::from_vec_on(vk(), values, shape)?)
}

/// Deterministic ramp filler so tests are seed-free but reproducible.
fn ramp(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.013 - 0.5) * scale).collect()
}

/// max |a - b| over two host vecs (read back from device first at call site).
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// T1 — core "does the recorder admit Vulkan and record a node now" assertion.
// This is THE proof that PR5's device-gate widen took effect.
// ---------------------------------------------------------------------------

#[test]
fn vk_tape_rms_norm_records_and_backprops() {
    if !vulkan_available() {
        eprintln!("[vk_tape] no Vulkan device — skipping rms_norm record test");
        return;
    }
    let hidden = 8usize;
    let rows = 4usize;
    let x = vk_f32(ramp(rows * hidden, 1.0), vec![1, rows, hidden]).unwrap();
    let weight = vk_f32(ramp(hidden, 0.25), vec![hidden]).unwrap();

    // Open a thread-local tape scope; without it `with_active_tape` returns None
    // and the recorder Ok(None) (PR5-spec R4).
    let recorded = with_thread_local_tape(|| -> Result<bool> {
        let out = try_tape_rms_norm_kt(&x, &weight, 1e-6)?;
        // PR5 core assertion: NOT None on Vulkan anymore.
        assert!(out.is_some(), "rms_norm recorder returned None on Vulkan — device gate not widened");
        let y = out.unwrap();
        assert_eq!(y.shape(), x.shape());
        // Exactly one node recorded.
        let n = with_active_tape(|tape| tape.len()).unwrap_or(0);
        assert_eq!(n, 1, "expected exactly 1 tape node, got {n}");
        Ok(true)
    });
    assert_eq!(recorded.expect("tape scope ran").expect("recorder ok"), true);
}

// ---------------------------------------------------------------------------
// T2 — FD backward parity on Vulkan storage (threshold §5.0 backward).
// FRONTIER: requires PR3 vulkan_fwd for {sum_axis, sqrt, reciprocal,
// broadcast_mul, cast, mul} that RmsNormKtBackward::apply calls
// (tape_forward.rs:253). Until then the analytic backward errors at
// tape.backward() time — keep #[ignore]d.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "FRONTIER: needs PR3 vulkan_fwd for the kt ops in RmsNormKtBackward::apply"]
fn vk_tape_rms_norm_fd_parity() {
    if !vulkan_available() {
        return;
    }
    // Central FD vs analytic dL/dx for a scalar loss L = sum(rms_norm(x, w)).
    // eps = 1e-3, accept max_abs_err ≤ 1e-3 (PR5-spec §5.0).
    //
    // Sketch (fill in at impl time once tape.backward over Vulkan storage runs):
    //   1. analytic: record rms_norm, seed grad_out = ones, tape.backward(), read dL/dx.
    //   2. numeric:  for each i, L(x+eps·e_i) - L(x-eps·e_i) / (2·eps), read back to host.
    //   3. assert max_abs_err(analytic_dx, numeric_dx) ≤ 1e-3.
    //
    // Keep the tensor tiny (rows≤4, hidden≤8) so the O(N) FD sweep is bounded.
    unimplemented!("WIP: fill central-difference loop once Vulkan tape.backward lands");
}

// ---------------------------------------------------------------------------
// T3 — Vulkan ATTENTION backward records via the SDPA fallback (fused flash is
// CUDA-only; PR5-spec §2.2 flash row). Proves the attention edge isn't silently
// dropped on Vulkan.
// ---------------------------------------------------------------------------

#[test]
fn vk_tape_sdpa_fallback_records() {
    if !vulkan_available() {
        return;
    }
    // Tiny [B, nq/nkv, T, d] = [1, 2, 3, 4] q, [1,1,3,4] k/v (GQA 2->1), head_dim=4.
    let (b, nq, nkv, t, d) = (1usize, 2usize, 1usize, 3usize, 4usize);
    let q = vk_f32(ramp(b * nq * t * d, 1.0), vec![b, nq, t, d]).unwrap();
    let k = vk_f32(ramp(b * nkv * t * d, 0.7), vec![b, nkv, t, d]).unwrap();
    let v = vk_f32(ramp(b * nkv * t * d, 0.5), vec![b, nkv, t, d]).unwrap();
    // `out` is the already-computed SDPA output; the recorder just records the
    // backward against it. Shape [b, nq, t, d]. Build a stand-in here.
    let out = vk_f32(ramp(b * nq * t * d, 0.3), vec![b, nq, t, d]).unwrap();

    let ok = with_thread_local_tape(|| -> Result<bool> {
        let res = try_tape_sdpa_fallback_kt(&q, &k, &v, d, &out)?;
        assert!(res.is_some(), "sdpa_fallback recorder returned None on Vulkan — gate not widened");
        let n = with_active_tape(|tape| tape.len()).unwrap_or(0);
        assert_eq!(n, 1, "expected 1 SdpaBackward node, got {n}");
        Ok(true)
    });
    assert!(ok.unwrap().unwrap());
}

// ---------------------------------------------------------------------------
// T4 — Vulkan GDN backward records (GdnRecurrentBackward). Proves the GDN
// recurrence edge survives on Vulkan.
// FRONTIER: try_tape_gdn_recurrent_kt takes &dyn BackendRuntime and calls the
// production recurrence `gdn_recurrent_forward_from_parts` — needs the Vulkan
// backend (for_device_kt(Device::Vulkan)) AND PR3 op coverage for the
// recurrence composite. Keep #[ignore]d until that path runs.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "FRONTIER: needs Vulkan backend + PR3 op coverage for gdn_recurrent_forward_from_parts"]
fn vk_tape_gdn_recurrent_records() {
    if !vulkan_available() {
        return;
    }
    let _backend = backend::for_device_kt(&vk());
    // Build tiny head-FIRST [B, nv, T, dv] q/k/v/beta/g + recurrent_state on
    // Vulkan, then:
    //   let out = try_tape_gdn_recurrent_kt(&*_backend, &q, &k, &v, &beta, &g, &mut state)?;
    //   assert!(out.is_some());
    //   assert_eq!(with_active_tape(|t| t.len()).unwrap_or(0), 1);
    // Shapes per forward.rs gated_deltanet_forward_decode_if. WIP.
    unimplemented!("WIP: build tiny GDN parts on Vulkan and assert GdnRecurrentBackward records");
}

// ---------------------------------------------------------------------------
// T5 — Cross-entropy loss root records on Vulkan (exercises PR2
// from_vec_on(Device::Vulkan,...) for the index helpers, tape_forward.rs:793/817).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "FRONTIER: needs PR2 from_vec_on(Vulkan) + PR3 index_select/log_sum_exp vulkan_fwd"]
fn vk_tape_cross_entropy_records() {
    if !vulkan_available() {
        return;
    }
    let (t, vocab) = (3usize, 8usize); // [1, T, V]
    let logits = vk_f32(ramp(t * vocab, 1.0), vec![1, t, vocab]).unwrap();
    let input_ids: Vec<u32> = vec![1, 2, 3];
    let label_mask: Vec<bool> = vec![true, true, true];

    let ok = with_thread_local_tape(|| -> Result<bool> {
        let res = try_tape_cross_entropy_from_logits_kt(&logits, &input_ids, &label_mask)?;
        assert!(res.is_some(), "CE recorder returned None on Vulkan");
        let loss = res.unwrap();
        assert_eq!(loss.shape().len(), 0, "CE loss must be scalar");
        assert_eq!(with_active_tape(|tape| tape.len()).unwrap_or(0), 1);
        Ok(true)
    });
    assert!(ok.unwrap().unwrap());
}

// ---------------------------------------------------------------------------
// T6 — Fixed-seed per-op FORWARD parity, Vulkan-kt vs vk_forward.rs, BEFORE
// deletion (PR5-spec §5.2). The safety proof that retiring vk_forward.rs does
// not change numerics. BOUNDED: one tiny forward, no training loop.
// FRONTIER: needs PR3 matmul vulkan_fwd; keep #[ignore]d until then.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "FRONTIER: needs PR3 MatmulOp::vulkan_fwd before the harmonized matmul can run"]
fn vk_forward_per_op_parity_pre_delete() {
    if !vulkan_available() {
        return;
    }
    // For each retired vk_forward entry point, run BOTH paths on the SAME
    // fixed-seed weights+input on Device::Vulkan(0) and compare (§5.0 forward):
    //
    //   (a) vk_linear_with_lora           vs  try_tape_lora_linear_kt / ops::matmul
    //   (b) vk_model_forward_loss         vs  model_forward_kt + try_tape_cross_entropy_from_logits_kt
    //   (c) vk_grpo_reference_log_probs.. vs  model_forward_kt -> trainer::token_log_probs
    //
    // Build one tiny VkModelWeights via from_gpu_weights(&gpu_weights) AND the
    // same GpuWeights for model_forward_kt; read both outputs back to host and
    // assert max_abs_err ≤ 1e-5 (logits/loss) / max_rel_err ≤ 1e-4.
    //
    // This file does NOT import vk_forward to avoid coupling the scaffold to the
    // soon-deprecated module; wire the import at impl time:
    //   use kiln_model::vk_forward::{VkModelWeights, vk_linear_with_lora, vk_model_forward_loss};
    unimplemented!("WIP: fixed-seed tiny-model forward parity vs vk_forward.rs");
}

// ---------------------------------------------------------------------------
// T7 — regression guard: the device-AGNOSTIC recorders (no device gate) must
// still record on Vulkan. Catches a future edit that accidentally adds a gate.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "FRONTIER: needs PR3 vulkan_fwd for silu/add/matmul/mul/embedding/swiglu/rope kt ops"]
fn vk_device_agnostic_recorders_still_record() {
    if !vulkan_available() {
        return;
    }
    // Smoke each of try_tape_{silu,add,matmul,embedding,swiglu,mul,rope}_kt on a
    // tiny Vulkan input inside `with_thread_local_tape` and assert Some plus a
    // recorded node.
    // These have NO device gate (tape_forward.rs: silu:143, add:161, rope:184,
    // matmul:384, embedding:410, swiglu:443, mul:470) so they should already
    // record once the module compiles + the kt ops have vulkan_fwd. WIP body.
    unimplemented!("WIP: per-op record smoke for the un-gated recorders on Vulkan");
}

// ===========================================================================
// §5.3 SINGLE reachability smoke — drop into
// crates/kiln-server/tests/real_model_integration.rs, mirroring
// test_real_model_opd_metal (line ~1112). BOUNDED: epochs:1, 2 tiny examples,
// tiny config, F32. This is the ONLY permitted real-model smoke in PR5.
// Multi-epoch / loss-decrease variants are HUMAN-GATED soak steps (§5.4) — do
// NOT add them here.
// ===========================================================================
//
// #[cfg(feature = "vulkan")]
// #[test]
// fn test_real_model_sft_vulkan_reachability() {
//     if !kiln_model::backend::vulkan::vulkan_is_available() {
//         eprintln!("No Vulkan device — skipping SFT-on-Vulkan reachability smoke");
//         return;
//     }
//     let device = Device::Vulkan(0);
//     let config = tiny_config();           // F32 (NOT bf16 — Vulkan trains F32)
//     let weights = tiny_weights(&config, &device);   // not tiny_weights_bf16
//     let tokenizer = test_tokenizer();
//     let examples = vec![ /* the 2 tiny SftExamples from test_real_model_sft_metal */ ];
//     let sft_config = kiln_train::SftConfig {
//         epochs: 1, learning_rate: 1e-3, lora_rank: 2, lora_alpha: 4.0,
//         auto_load: false, seed: Some(0), ..Default::default()
//     };
//     let adapter_dir = tempfile::tempdir().unwrap();
//     let result = kiln_train::trainer::sft_train(
//         &examples, &sft_config, &config, &weights, &tokenizer,
//         adapter_dir.path(), "sft-vulkan-reachability", None, None,
//     );
//     // FRONTIER assertion (mirror test_real_model_opd_metal): either full
//     // success (adapter written + finite loss), OR a DOCUMENTED op-gap. Fail if
//     // it stops anywhere unexpected.
//     match result {
//         Ok(out) => { assert_adapter_written(&out); }
//         Err(e) => {
//             let msg = format!("{e:#}");
//             let documented = msg.contains("vulkan_fwd")    // a kt op missing on Vulkan (PR3)
//                 || msg.contains("NYI")                      // an un-un-NYI'd path
//                 || msg.contains("tape-authoritative");      // scope not yet wired (PR6)
//             assert!(documented, "SFT-on-Vulkan stopped at an UNDOCUMENTED frontier: {msg}");
//             eprintln!("[reachability] SFT-on-Vulkan pinned at documented frontier: {msg}");
//         }
//     }
// }
