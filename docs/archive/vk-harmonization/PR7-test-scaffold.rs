//! PR7 bounded test scaffold — WIP, DO NOT COMPILE AS-IS.
//!
//! ⚠️  This is a SPEC ARTIFACT shipped with `docs/archive/vk-harmonization/PR7-spec.md`.
//!     It is NOT wired into any crate's `tests/` yet and is intentionally left
//!     OUT of the build. The PR7 implementer drops the relevant test into
//!     `crates/kiln-train/tests/vk_harmonized_save_smoke.rs` and fixes the
//!     `TODO(impl)` markers against the REAL constructor signatures at the
//!     landing commit (every API anchor below was read at branch HEAD and WILL
//!     drift — re-grep before relying on it).
//!
//! Every test here is HOST-SAFE: GPU-gated (skips cleanly without Vulkan),
//! single-op or single-save, NO training loop, NO multi-step, NO long-running
//! binary. This honors the hard host-safety ceiling (the dev host has
//! hard-crashed twice on long training runs).
//!
//! Scope mirror (PR7-spec.md §6.5 / §8):
//!   - §6.5  harmonized-save reachability smoke  (the NEW test PR7 adds)
//!   - §8    adapter / round-trip parity thresholds (F32 max_abs_err ≤ 1e-5)
//!
//! Run individually (NEVER `cargo test --workspace`):
//!   CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
//!     cargo test -p kiln-train --features vulkan vk_harmonized_save -- --ignored --nocapture

#![cfg(feature = "vulkan")]
#![allow(unused_imports, dead_code, unused_variables)]

use std::sync::Arc;

// --- API anchors (HEAD; re-grep, they drift) -------------------------------
//   TrainableLoraParams { layers, rank, alpha }          crates/kiln-train/src/trainer.rs:559
//   TrainableLoraLayerParams { q_proj: Option<(Parameter, Parameter)>, ... } trainer.rs:570
//   TrainableLoraParams::save_peft(&self, dir, num_layers)                   trainer.rs:1203
//   Parameter::trainable(ForwardStorage, master: Tensor, AmpPolicy)          kiln-param/src/parameter.rs:273
//   ForwardStorage::Plain(Tensor)                                            kiln-param/src/parameter.rs:32
//   host_to_vulkan_copy(cpu: &Tensor, device_index) -> Result<Tensor>        kiln-tensor/src/vulkan_storage.rs:277
//   vulkan_to_host_copy(t: &Tensor) -> Result<Tensor>                        kiln-tensor/src/vulkan_storage.rs:372
//   kiln_tensor::safetensors::load_cpu(path) -> HashMap<String, Tensor>      kiln-tensor/src/safetensors.rs:66
// ---------------------------------------------------------------------------

/// Max absolute elementwise error between two f32 slices. Mirrors
/// `vk_opd_parity.rs:156 max_abs_err`. PR7-spec.md §8 gate: F32 ≤ 1e-5.
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_err");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Probe for a usable Vulkan device. Returns the device index (0) or `None`.
/// Mirrors the skip-cleanly pattern in `vk_tensor.rs:483` and
/// `vk_cuda_opd_parity.rs:vk_dev`.
fn vulkan_device_index() -> Option<usize> {
    // TODO(impl): use the real probe. At HEAD the kernel-crate probe is
    // `kiln_vulkan_kernel::VulkanDevice::probe()`. The kt-tensor side likely
    // exposes `kiln_tensor::primary_vulkan_device(0).is_ok()` (mirror of
    // `primary_cuda_context(0)` used in vk_cuda_opd_parity.rs). Pick whichever
    // the harmonized path canonicalizes on by the landing commit.
    //
    //   if kiln_tensor::primary_vulkan_device(0).is_ok() { Some(0) } else { None }
    None
}

// ===========================================================================
// §6.5 — Harmonized-save reachability smoke (the NEW PR7 test)
// ===========================================================================
//
// Proves that, with the fork gone, a LoRA adapter trained on `Device::Vulkan`
// saves through the SHARED `TrainableLoraParams::save_peft` path and reads back
// bit-exact — i.e. `save_vk_lora_adapter` is genuinely unnecessary.
//
// NO training. We construct two tiny LoRA Parameters directly from known host
// data, park them on Vulkan, save, then reload and compare.

#[test]
#[ignore = "WIP PR7 scaffold — wire real constructors before un-ignoring"]
fn vk_harmonized_lora_save_roundtrips_bit_exact() -> anyhow::Result<()> {
    let Some(dev_idx) = vulkan_device_index() else {
        eprintln!("SKIP: no Vulkan device");
        return Ok(());
    };

    // -- 1. Known host LoRA weights ----------------------------------------
    //    rank r=4, one attn q_proj pair. A: [r, in], B: [out, r]. F32 (the
    //    harmonized Vulkan TRAINED dtype — PR7-spec.md §2.3).
    let r = 4usize;
    let in_features = 8usize;
    let out_features = 8usize;
    let a_host: Vec<f32> = (0..r * in_features).map(|i| (i as f32) * 0.013 - 0.2).collect();
    let b_host: Vec<f32> = (0..out_features * r).map(|i| (i as f32) * 0.007 + 0.1).collect();

    // -- 2. Build Vulkan-resident Parameters -------------------------------
    // TODO(impl): the exact construction depends on PR2/PR3 helpers at the
    // landing commit. The intended shape:
    //
    //   use kiln_tensor::{Tensor, Device, DType};
    //   use kiln_param::{Parameter, ForwardStorage, AmpPolicy};
    //   use kiln_tensor::vulkan_storage::host_to_vulkan_copy;
    //
    //   let a_cpu = Tensor::from_vec(a_host.clone(), &[r, in_features], &Device::Cpu)?;
    //   let b_cpu = Tensor::from_vec(b_host.clone(), &[out_features, r], &Device::Cpu)?;
    //   let a_vk = host_to_vulkan_copy(&a_cpu, dev_idx)?;   // vulkan_storage.rs:277
    //   let b_vk = host_to_vulkan_copy(&b_cpu, dev_idx)?;
    //   let mk = |t: Tensor| Parameter::trainable(
    //       ForwardStorage::Plain(t.clone()), t, AmpPolicy::default());   // parameter.rs:273
    //   let q_pair = Some((mk(a_vk), mk(b_vk)));
    //
    //   let layer = TrainableLoraLayerParams { q_proj: q_pair, ..Default::default() }; // trainer.rs:570
    //   let params = TrainableLoraParams { layers: vec![layer], rank: r, alpha: 8.0 }; // trainer.rs:559
    let _ = (r, in_features, out_features, &a_host, &b_host);
    let params = unimplemented!("TODO(impl): build TrainableLoraParams on Vulkan");

    // -- 3. Save through the SHARED path (NOT save_vk_lora_adapter) ---------
    let tmp = tempfile::tempdir()?;
    // params.save_peft(tmp.path(), /*num_layers=*/ 1)?;   // trainer.rs:1203
    let _ = &params;

    // -- 4. Assert the harmonized path wrote what the fork did NOT ----------
    let st = tmp.path().join("adapter_model.safetensors");
    let cfg = tmp.path().join("adapter_config.json");
    assert!(st.exists(), "save_peft must write adapter_model.safetensors");
    assert!(
        cfg.exists(),
        "save_peft writes adapter_config.json — save_vk_lora_adapter omitted it (PR7-spec §3.4)"
    );

    // -- 5. Reload and compare bit-exact (F32 max_abs_err ≤ 1e-5) ----------
    //   let loaded = kiln_tensor::safetensors::load_cpu(&st)?;   // safetensors.rs:66
    //   let a_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
    //   let b_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
    //   let a_back = loaded[a_key].to_vec1::<f32>()?;   // (or .flatten_all()?.to_vec1)
    //   let b_back = loaded[b_key].to_vec1::<f32>()?;
    //   assert!(max_abs_err(&a_back, &a_host) <= 1e-5, "lora_A drift");
    //   assert!(max_abs_err(&b_back, &b_host) <= 1e-5, "lora_B drift");

    Ok(())
}

// ===========================================================================
// §8 — Storage round-trip parity (host -> Vulkan -> host), F32 bit-exact
// ===========================================================================
//
// Independent of save_peft: proves vulkan_to_host_copy is bit-exact, since the
// adapter save reads parameters back through it. This is a thinner guard than
// the PR2 round-trip tests (kiln-tensor/src/tensor.rs:1491/1499) and exists so
// kiln-train's vulkan feature has a self-contained host-safe check.

#[test]
#[ignore = "WIP PR7 scaffold — wire real host_to/from copies before un-ignoring"]
fn vk_storage_roundtrip_f32_bit_exact() -> anyhow::Result<()> {
    let Some(dev_idx) = vulkan_device_index() else {
        eprintln!("SKIP: no Vulkan device");
        return Ok(());
    };
    let host: Vec<f32> = vec![1.0, -2.5, 3.14159, 0.0, 42.0, -0.001];

    // TODO(impl):
    //   use kiln_tensor::{Tensor, Device};
    //   use kiln_tensor::vulkan_storage::{host_to_vulkan_copy, vulkan_to_host_copy};
    //   let cpu = Tensor::from_vec(host.clone(), &[host.len()], &Device::Cpu)?;
    //   let vk = host_to_vulkan_copy(&cpu, dev_idx)?;
    //   let back = vulkan_to_host_copy(&vk)?.to_vec1::<f32>()?;
    //   assert_eq!(back, host, "F32 host<->Vulkan round-trip must be bit-exact");
    let _ = (dev_idx, &host);
    Ok(())
}

// ===========================================================================
// §4.3 — `vk_autograd` removal guard (compile-time, NOT a runtime test)
// ===========================================================================
//
// Not expressible as a normal test: the point is that `vk_autograd`,
// `vk_backward`, and `VkBackwardOp` have NO live consumers. The implementer
// enforces this with the grep in PR7-spec.md §10 step 0/6 and a workspace
// `cargo check`, not with a runtime assertion. Left here as a documented
// reminder so nobody adds a runtime "test" that re-imports the dead symbols.
//
//   grep -rln 'VkBackwardOp\|vk_backward\|VkGradStore\|from_op\|\.grad_fn\|requires_grad' \
//     crates/kiln-vulkan-kernel/src crates/kiln-vulkan-kernel/tests | grep -v /target/
//   # must be EMPTY (doc-comment hits excepted) before removing the autograd half.

// ===========================================================================
// Cross-engine OPD parity (CUDA vs Vulkan) — ALREADY EXISTS, do not duplicate
// ===========================================================================
//
// `crates/kiln-train/tests/vk_cuda_opd_parity.rs` already enforces the §9.2
// contract (F32 ≤ 1e-5 target; 1e-4 abs / 1e-3 rel practical) under
// `#[cfg(all(feature = "cuda", feature = "vulkan"))]` with both GPUs present.
// PR7 keeps it green; it is NOT re-scaffolded here.
