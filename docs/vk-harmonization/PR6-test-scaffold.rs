// =============================================================================
// PR6 TEST SCAFFOLD — Vulkan SFT/GRPO/OPD reachability + OPD FD parity (WIP)
// =============================================================================
//
// STATUS: WIP DROP-IN. **DO NOT COMPILE AS-IS.** This file is a scaffold the
// PR6 implementer pastes (in pieces) into the real test crates AFTER the PR6
// gate-widenings land. It mirrors the fully-harmonized Metal template:
//   crates/kiln-server/tests/real_model_integration.rs
//     - test_real_model_sft_metal   (:892)
//     - test_real_model_grpo_metal  (:975)
//     - test_real_model_opd_metal   (:1112)
//     - helpers: loss_capture_cb, assert_loss_decreases, receipt_lora_grad_norms,
//                metal_gpu_guard, assert_adapter_written, tiny_weights_bf16,
//                metal_chat_msg, tiny_config, test_tokenizer
//
// HOW TO USE:
//   1. The three `test_real_model_*_vulkan` fns + the vulkan helpers below go
//      into `crates/kiln-server/tests/real_model_integration.rs`, replacing the
//      `#[cfg(feature = "metal")]` attrs with `#[cfg(feature = "vulkan")]` and
//      the Metal availability/device with their Vulkan equivalents (already done
//      here). The shared helpers `tiny_config`, `test_tokenizer`, `tiny_weights`,
//      `metal_chat_msg`, `tiny_weights_bf16`, `assert_adapter_written`,
//      `loss_capture_cb`, `assert_loss_decreases`, `receipt_lora_grad_norms`
//      already exist in that file under `#[cfg(feature = "metal")]` — either
//      drop the `metal` cfg so both features share them, or duplicate under a
//      `#[cfg(feature = "vulkan")]` block. `metal_chat_msg` is feature-neutral
//      content, reused verbatim as `vk_chat_msg` below.
//   2. The OPD finite-difference parity test goes into
//      `crates/kiln-opd-loss-kernel/src/kt_api.rs` `#[cfg(test)]` mod, next to
//      the existing CUDA/CPU composite FD tests (~:1670/:1763/:1825). It reuses
//      that module's FD helper shape.
//   3. The recorder-coverage assertion is a plain test (no GPU) — put it in
//      `crates/kiln-model/src/tape_forward.rs` `#[cfg(test)]` or a CI grep step.
//
// EVERY GPU test self-skips when no Vulkan device is present
// (`kiln_model::backend::vulkan::vulkan_is_available()`), so the suite is green
// on CI boxes without a GPU. All are additionally `#[ignore]` here so a stray
// `cargo test` cannot launch a GPU run autonomously (host-safety: the dev box
// has hard-crashed on long runs — a human un-ignores + runs these, see
// PR6-spec.md §6.6).
//
// BF16 CAVEAT (PR6-spec.md §7 R1): SFT/GRPO require a BF16 base
// (`base_dtype_supports_tape`). If Vulkan can only train F32 in this release,
// keep `test_real_model_opd_vulkan` enabled (OPD composite accepts F32) and
// leave the SFT/GRPO smokes `#[ignore = "needs BF16 Vulkan base (R1)"]`.
// =============================================================================

#![cfg(feature = "vulkan")]
#![allow(dead_code, unused_imports)] // WIP scaffold

// -----------------------------------------------------------------------------
// PART A — reachability smokes (paste into kiln-server/tests/real_model_integration.rs)
// -----------------------------------------------------------------------------

use kiln_tensor::{DType, Device, Tensor};
use serde_json::Value;

// Feature-neutral chat-message builder (identical body to `metal_chat_msg`).
#[cfg(feature = "vulkan")]
fn vk_chat_msg(role: &str, content: &str) -> kiln_train::ChatMessage {
    kiln_train::ChatMessage {
        role: role.to_string(),
        content: content.to_string(),
    }
}

/// Serialize GPU-heavy Vulkan tests in this binary (mirrors `metal_gpu_guard`).
/// The Vulkan logical device + command pools are process-global; concurrent
/// submission from cargo's parallel test threads perturbs reduction ordering,
/// which the marginal loss-decrease assertions are sensitive to.
#[cfg(feature = "vulkan")]
fn vk_gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

// NOTE: `tiny_config`, `test_tokenizer`, `tiny_weights`, `tiny_weights_bf16`,
// `assert_adapter_written`, `loss_capture_cb`, `assert_loss_decreases`,
// `receipt_lora_grad_norms` are reused from the existing test file. When pasting,
// drop the `#[cfg(feature = "metal")]` on those helpers so `vulkan` builds see
// them too (they contain no Metal-specific API beyond the cfg attr).

/// SFT-on-Vulkan reachability smoke. Mirror of `test_real_model_sft_metal:892`.
/// Exercises `sft_train` -> `standard_forward_backward` ->
/// `standard_forward_backward_tape_authoritative_kt` on `Device::Vulkan(0)`.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "GPU soak — human-gated (PR6-spec.md §6.6); also needs BF16 Vulkan base (R1)"]
fn test_real_model_sft_vulkan() {
    if !kiln_model::backend::vulkan::vulkan_is_available() {
        eprintln!("No Vulkan device — skipping SFT-on-Vulkan smoke");
        return;
    }
    let _gpu = vk_gpu_guard();
    let device = Device::Vulkan(0);
    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;
    let weights = tiny_weights_bf16(&config, &device);
    let tokenizer = test_tokenizer();

    let examples = vec![
        kiln_train::SftExample {
            messages: vec![
                vk_chat_msg("user", "t1 t2 t3"),
                vk_chat_msg("assistant", "t2 t3 t1"),
            ],
        },
        kiln_train::SftExample {
            messages: vec![
                vk_chat_msg("user", "t3 t1"),
                vk_chat_msg("assistant", "t1 t2 t3"),
            ],
        },
    ];
    let sft_config = kiln_train::SftConfig {
        epochs: 3,
        learning_rate: 1e-3,
        lora_rank: 2,
        lora_alpha: 4.0,
        auto_load: false,
        seed: Some(0),
        ..Default::default()
    };

    let (losses, cb) = loss_capture_cb();
    let adapter_dir = tempfile::tempdir().unwrap();
    let out = kiln_train::trainer::sft_train(
        &examples,
        &sft_config,
        &config,
        &weights,
        &tokenizer,
        adapter_dir.path(),
        "sft-vulkan-smoke",
        Some(cb),
        None,
    )
    .expect("SFT training on Device::Vulkan(0) should complete");
    assert_adapter_written(&out);
    assert_loss_decreases(&losses.lock().unwrap());
}

/// GRPO-on-Vulkan reachability smoke. Mirror of `test_real_model_grpo_metal:975`.
/// One pass, `PerSample` aggregation, varied rewards (1.0 / 0.0) so a real PG
/// flows, KL + ECHO off so the step stays tape-authoritative. Asserts grads
/// flowed via `receipt_lora_grad_norms` (the empty-tape failure mode guard).
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "GPU soak — human-gated (PR6-spec.md §6.6); also needs BF16 Vulkan base (R1)"]
fn test_real_model_grpo_vulkan() {
    if !kiln_model::backend::vulkan::vulkan_is_available() {
        eprintln!("No Vulkan device — skipping GRPO-on-Vulkan smoke");
        return;
    }
    let _gpu = vk_gpu_guard();
    let device = Device::Vulkan(0);
    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;
    let weights = tiny_weights_bf16(&config, &device);
    let tokenizer = test_tokenizer();

    let mk_group = |prompt: &str, win: &str, lose: &str| kiln_train::GrpoGroup {
        messages: vec![vk_chat_msg("user", prompt)],
        completions: vec![
            kiln_train::ScoredRollout::legacy(win.to_string(), 1.0),
            kiln_train::ScoredRollout::legacy(lose.to_string(), 0.0),
        ],
    };
    let groups = vec![
        mk_group("t1 t2 t3", "t2 t3 t1", "t3 t1 t2"),
        mk_group("t3 t1", "t1 t2 t3", "t3 t3 t3"),
        mk_group("t2 t1 t3", "t3 t2 t1", "t1 t1 t1"),
    ];

    let mut grpo_config = kiln_train::GrpoConfig {
        learning_rate: 1e-3,
        kl_coeff: 0.0,
        lora_rank: 2,
        lora_alpha: 4.0,
        auto_load: false,
        seed: Some(0),
        dynamic_sampling: false,
        loss_aggregation: kiln_train::LossAggregation::PerSample,
        ..kiln_train::GrpoConfig::default()
    };
    grpo_config.loss.echo = None;
    grpo_config.loss.no_policy_loss = false;

    let (losses, cb) = loss_capture_cb();
    let adapter_dir = tempfile::tempdir().unwrap();
    let out = kiln_train::trainer::grpo_train(
        &groups,
        &grpo_config,
        &config,
        &weights,
        &tokenizer,
        adapter_dir.path(),
        "grpo-vulkan-smoke",
        Some(cb),
        None,
    )
    .expect("GRPO training on Device::Vulkan(0) should complete");

    assert_adapter_written(&out);

    let losses = losses.lock().unwrap();
    assert!(!losses.is_empty(), "GRPO progress callback recorded no losses");
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "all GRPO training losses must be finite, got {losses:?}"
    );

    // Empty/severed-tape guard: a non-zero module count proves the tape walked
    // and deposited grads; a strictly-positive max mean-norm proves real PG
    // signal (mirrors the Metal GRPO smoke at real_model_integration.rs:1065).
    let (grad_norm_modules, max_mean_norm) = receipt_lora_grad_norms(&out);
    assert!(
        grad_norm_modules > 0,
        "GRPO step recorded no LoRA grad norms — gradients did not flow through \
         the kt tape-authoritative path on Vulkan (losses were {losses:?})"
    );
    assert!(
        max_mean_norm > 0.0,
        "GRPO LoRA grad norms are all zero — the tape walked but deposited no \
         signal (modules={grad_norm_modules}, losses={losses:?})"
    );
    eprintln!(
        "[GRPO-VULKAN] losses={losses:?} lora_grad_norm_modules={grad_norm_modules} \
         max_mean_norm={max_mean_norm}"
    );
}

/// OPD-on-Vulkan reachability smoke. Mirror of `test_real_model_opd_metal:1112`.
/// THE single allowed one-step real-model smoke (validation ceiling). Routes
/// `opd_train` -> `opd_step_forward_backward_tape_authoritative` -> the Vulkan
/// forward -> the kt-native OPD scalar-loss tape root
/// (`try_tape_opd_scalar_mean_cuda_kt`) -> the device-agnostic backward composite
/// (`opd_top_k_reverse_kl_phase_b_bwd_composite_kt`, which Vulkan reaches because
/// the CUDA-FFI branch is `Cuda(_)`-only). OPD accepts F32, so this is the
/// R1-independent smoke that proves the substrate even if SFT/GRPO stay BF16-gated.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "GPU soak — human-gated (PR6-spec.md §6.6)"]
fn test_real_model_opd_vulkan() {
    use std::sync::Arc;

    if !kiln_model::backend::vulkan::vulkan_is_available() {
        eprintln!("No Vulkan device — skipping OPD-on-Vulkan smoke");
        return;
    }
    let _gpu = vk_gpu_guard();
    let device = Device::Vulkan(0);
    let mut config = tiny_config();
    config.dtype = kiln_core::config::DType::BF16;
    let weights = tiny_weights_bf16(&config, &device);
    let tokenizer = test_tokenizer();

    let prompts = vec![
        kiln_train::opd::OpdPrompt {
            messages: vec![
                vk_chat_msg("user", "t1 t2 t3"),
                vk_chat_msg("assistant", "t2 t3 t1"),
            ],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        },
        kiln_train::opd::OpdPrompt {
            messages: vec![
                vk_chat_msg("user", "t3 t1"),
                vk_chat_msg("assistant", "t1 t2 t3"),
            ],
            teacher_extra_messages: vec![],
            trajectory: vec![],
        },
    ];

    // K=32 matches the OPD kernel envelope ({16, 32}); tiny vocab_size=32 means
    // the uniform top-K spans the full vocab.
    let teacher: Arc<dyn kiln_train::logit_source::LogitSource> = Arc::new(
        kiln_train::logit_source::DeterministicUniformLogitSource::new(
            "vulkan-smoke-teacher",
            config.vocab_size,
            32,
        ),
    );

    let mut opd_config = kiln_train::opd::OpdConfig {
        learning_rate: 1e-3,
        lora_rank: 2,
        lora_alpha: 4.0,
        auto_load: false,
        seed: Some(0),
        epochs: 1,
        ..Default::default()
    };
    opd_config.training_mode = kiln_train::opd::OpdTrainingMode::OffPolicy;

    let (losses, cb) = loss_capture_cb();
    let adapter_dir = tempfile::tempdir().unwrap();
    let result = kiln_train::opd::opd_train(
        &prompts,
        &opd_config,
        &config,
        &weights,
        &tokenizer,
        teacher,
        adapter_dir.path(),
        "opd-vulkan-smoke",
        Some(cb),
    );

    match result {
        Ok(out) => {
            assert_adapter_written(&out);
            let losses = losses.lock().unwrap();
            assert!(!losses.is_empty(), "OPD progress callback recorded no losses");
            assert!(
                losses.iter().all(|l| l.is_finite()),
                "all OPD training losses must be finite, got {losses:?}"
            );
            let (grad_norm_modules, max_mean_norm) = receipt_lora_grad_norms(&out);
            assert!(
                grad_norm_modules > 0,
                "OPD step recorded no LoRA grad norms — gradients did not flow \
                 through the kt tape-authoritative path on Vulkan"
            );
            eprintln!(
                "[OPD-VULKAN] run completed (Vulkan OPD forward + composite backward): \
                 losses={losses:?} lora_grad_norm_modules={grad_norm_modules} \
                 max_mean_norm={max_mean_norm}"
            );
        }
        Err(e) => {
            // Once PR3/PR5 land the Vulkan forward + the composite ops, OPD must
            // run end-to-end on Vulkan. Any error is a real regression — fail loud.
            let chain = format!("{e:#}");
            panic!(
                "[OPD-VULKAN] OPD-on-Vulkan must run to completion (Vulkan forward \
                 + device-agnostic kt-composite backward). Got error chain:\n{chain}"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// PART B — OPD composite finite-difference parity on Vulkan
//   (paste into crates/kiln-opd-loss-kernel/src/kt_api.rs #[cfg(test)] mod,
//    next to the existing composite FD tests ~:1670/:1763/:1825)
// -----------------------------------------------------------------------------
//
// Acceptance threshold (mirror the Metal OPD gate): max_abs_err <= 1e-5 for F32.
// Reuse the existing module's FD helper / epsilon — DO NOT hand-roll a second FD
// harness; grep the existing test for the central-difference epsilon and the
// `to_device`/`from_vec` builders it uses.
//
// SKELETON (fill the `build_*` + forward-scalar helpers from the existing test):
/*
#[cfg(all(test, feature = "vulkan"))]
mod vulkan_fd_parity {
    use super::*;
    use kiln_tensor::Device as KtDevice;

    const FD_EPS: f32 = 1e-3;        // match the existing composite FD test
    const MAX_ABS_ERR: f32 = 1e-5;   // F32 gate, mirrors the Metal OPD acceptance

    /// Central finite-difference of the scalar OPD loss wrt each hidden[t,h],
    /// compared against the analytic `opd_top_k_reverse_kl_phase_b_bwd_composite_kt`
    /// gradient — run entirely on Device::Vulkan(0). Self-skips without a device.
    #[test]
    #[ignore = "GPU FD parity — human-gated (PR6-spec.md §6.2)"]
    fn opd_composite_bwd_fd_parity_vulkan() {
        if !vulkan_is_available_for_tests() {  // reuse the crate's availability check
            eprintln!("No Vulkan device — skipping OPD composite FD parity");
            return;
        }
        let dev = KtDevice::Vulkan(0);
        let (t, h, v, top_k) = (3usize, 4usize, 32usize, 32usize);

        // hidden [1, T, H], head_t [H, V], teacher top-K metadata, label_mask.
        // Build on CPU then `.to_device(dev)` (PR2 makes Vulkan to_device work).
        let hidden = build_hidden_f32(&[1, t, h], &dev);          // <- existing helper
        let head_t = build_head_t_f32(&[h, v], &dev);             // <- existing helper
        let (idx, lqp, mask) = build_teacher_topk(t, top_k, v);   // <- existing helper

        // grad_loss for ScalarMean is a single-element [1] F32 = 1.0.
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![1])
            .unwrap().to_device(dev).unwrap();

        // Analytic gradient under test.
        let d_hidden = opd_top_k_reverse_kl_phase_b_bwd_composite_kt(
            &hidden, &head_t, &idx, &lqp, &mask, &grad_loss, top_k,
            OpdLossOutputKt::ScalarMean,
        ).expect("composite backward");
        let d_hidden_host = read_to_host_f32(&d_hidden);          // <- vulkan_to_host_copy path

        // Central FD reference: perturb each active hidden[t,h] by ±eps, recompute
        // the scalar OPD loss via the forward composite, dL = (L+ - L-)/(2 eps).
        let mut max_abs_err = 0.0f32;
        for ti in 0..t {
            if !mask[ti] { continue; }
            for hi in 0..h {
                let lp = scalar_loss_with_perturbation(&hidden, ti, hi, FD_EPS, /*..*/);
                let lm = scalar_loss_with_perturbation(&hidden, ti, hi, -FD_EPS, /*..*/);
                let fd = (lp - lm) / (2.0 * FD_EPS);
                let an = d_hidden_host[ti * h + hi];
                max_abs_err = max_abs_err.max((fd - an).abs());
            }
        }
        assert!(
            max_abs_err <= MAX_ABS_ERR,
            "OPD composite Vulkan FD parity: max_abs_err {max_abs_err} > {MAX_ABS_ERR}"
        );
        eprintln!("[OPD-VULKAN-FD] max_abs_err={max_abs_err} (gate {MAX_ABS_ERR})");
    }
}
*/

// -----------------------------------------------------------------------------
// PART C — recorder-coverage assertion (no GPU; gates PR6-over-PR5 ordering)
//   (paste into crates/kiln-model/src/tape_forward.rs #[cfg(test)] mod, or run
//    as a CI grep step — see PR6-spec.md §6.1)
// -----------------------------------------------------------------------------
//
// Proves PR6 was not landed ahead of PR5: every `try_tape_*_kt` device guard in
// tape_forward.rs must accept Vulkan before any orchestration routes into it.
// The literal count (31) is the HEAD count of `Cuda(_) | Metal(_)` guards; after
// PR5 each must also name Vulkan. Update the constant if PR5 adds/removes a
// recorder.
/*
#[cfg(test)]
mod pr6_prereq_guards {
    /// Read the sibling source file at build time and count device guards.
    /// (A grep-based CI step is equivalent and cheaper; this in-crate variant
    /// keeps the invariant next to the code it protects.)
    #[test]
    fn pr6_recorders_accept_vulkan() {
        let src = include_str!("tape_forward.rs");
        let cuda_metal = src.matches("Device::Cuda(_) | kiln_tensor::Device::Metal(_)").count();
        let with_vulkan = src.matches("Device::Vulkan(_)").count();
        // After PR5+PR6, every Cuda|Metal guard also names Vulkan.
        assert!(
            with_vulkan >= cuda_metal && with_vulkan >= 31,
            "tape_forward.rs recorders not Vulkan-widened: {with_vulkan} Vulkan \
             arms vs {cuda_metal} Cuda|Metal guards (expected >= 31). PR6 must \
             not route Vulkan before PR5 widens these recorders."
        );
        // Module cfg must include the vulkan feature.
        assert!(
            src.contains("feature = \"vulkan\""),
            "tape_forward.rs module #![cfg] must include feature = \"vulkan\""
        );
    }
}
*/
