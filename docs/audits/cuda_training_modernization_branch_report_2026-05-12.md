# CUDA Training Modernization Branch Report

**Date:** 2026-05-12
**Branch:** `cuda-training-modernization`
**Reference report:** `docs/audits/vulkan_training_branch_report_2026-05-11.md`
**Baseline:** `844492ab` - `Update Vulkan training branch audit`
**Status:** Phase ledger and first implementation target. This branch is intentionally incremental:
each CUDA training slice must land with tests and a pushed commit before the next larger slice.

## Branch Progress

| Commit | Slice | Evidence |
| --- | --- | --- |
| `c782a854` | Baseline CUDA modernization ledger | This report, derived from the Vulkan branch completion criteria. |
| `6ad88446` | Backend training capability telemetry | `BackendRuntime::training_capabilities`, CUDA/Vulkan profiles, `ModelRunner` startup log. |
| `0b1b8d2f` | CUDA TensorId residency lifecycle registry | CUDA implements register/has/update/evict metadata hooks; no side-buffer ownership claimed. |
| `c67de64f` | Registered CUDA LoRA delta path | `CudaBackend::lora_delta_resident` engages for registered CUDA A/B and delegates to candle CUDA autograd. |
| `6dba1644` | CUDA training projection hook | `CudaBackend::linear_prefill_apply` routes compatible CUDA projection matmuls through the backend seam using candle CUDA autograd. |
| `89926670` | CUDA runtime hook parity tests | Adds A6000-runnable tests for the projection hook and registered LoRA delta hook. |
| `4aee22f9` | CUDA offset training matmul hook | Adds `linear_prefill_apply_offset` for chunked CUDA matmuls, tested but not yet wired into FLCE auto-routing. |
| `b53602df` | CUDA FLCE backend provider gate | Wires CUDA FLCE backend chunk matmul behind `KILN_CUDA_FLCE=1`; default CUDA SFT remains on the existing candle CUDA Phase B chunked loss path until benchmarks justify auto-on. |
| `86291129` | CUDA optimizer dispatch explicit decline | Adds CUDA SGD/AdamW dispatch hooks that log first use and return `false` until CUDA owns a real in-place optimizer update. |
| `6a0824bd` | CUDA attention fast-path training guard | CUDA FlashAttention prefill/paged decode decline autograd-tracked tensors until a CUDA attention op with backward is wired. |
| `4d60b0a9` | CUDA A6000 preflight rejection coverage | Adds a discrete CUDA/A6000 long-context rejection test for the shared fit-before-run estimator. |
| `e1b45394` | CUDA resident optimizer kernels | Adds CUDA SGD and AdamW in-place kernels for registered contiguous CUDA F32/BF16 tensors. |
| `51e1d727` | CUDA optimizer support check | Tightens optimizer dispatch support detection so CUDA only claims supported registered device tensors. |

Local validation so far:

- `cargo test -p kiln-model backend::tests::portable_training_capabilities_are_conservative --lib --quiet` passed after each code slice.
- `cargo test -p kiln-server preflight_rejects_long_context_on_a6000_cuda_budget --lib --quiet` passed for CUDA/discrete fit-gate coverage.
- `cargo test -p kiln-train flce_provider --lib --quiet` passed for the CUDA opt-in gate and Vulkan auto-heuristic tests.
- `cargo test -p kiln-train flce_auto --lib --quiet` passed for the FLCE active-token floor tests.
- `git diff --check` passed before each code commit.
- `cargo fmt --all --check` is blocked in this workspace because the active musl Rust toolchain lacks `rustfmt`.
- Local CUDA-feature tests are blocked in this workspace because `cudarc` cannot find `nvcc`.
- RunPod A6000 validation on pod `pdauarcdn9k62l`, release mode with `KILN_CUDA_ARCHS=86`, passed:
  - `cargo test --release -p kiln-model --features cuda cuda_training_capabilities_do_not_overclaim_native_training --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_resident_activation_registry_lifecycle --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_linear_prefill_apply_matches_candle_cuda_matmul --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_linear_prefill_apply_offset_matches_candle_cuda_chunk --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_registered_lora_delta_matches_candle_cuda_reference --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda test_checkpointed_loss_matches_standard --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda test_flce_parity_vs_naive_loss --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda flce_provider --lib --quiet`
  - `KILN_CUDA_FLCE=1 cargo test --release -p kiln-train --features cuda test_flce_parity_vs_naive_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_optimizer_dispatch_hooks_decline_until_owned_kernel_exists --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_flash_attention_declines_tracked_training_tensors --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_sgd_step_resident_round_trip_f32 --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_adamw_step_resident_round_trip_f32 --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_sgd_and_adamw_resident_round_trip_bf16 --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda test_checkpointed_loss_matches_standard --lib --quiet` re-run after optimizer kernel wiring
  - Debug-mode CUDA test was intentionally rejected after `nvcc -G` hit exit 137 in `kiln-flash-attn`; release mode is the required kiln CUDA path.

## Executive Summary

The Vulkan training modernization branch established the target shape for backend training:

1. reject jobs that cannot fit before training starts;
2. route training math through autograd-safe GPU paths, not decode-only fast paths;
3. chunk vocab, row, batch, and sequence work before dispatch;
4. keep trainable and checkpoint state backend-resident between steps;
5. read back trainable state only at adapter-save/checkpoint boundaries;
6. eventually replace candle's host-storage training contract with a native tensor/autograd stack.

CUDA starts from a different place than Vulkan. The current CUDA backend already benefits from
candle CUDA tensors, FlashAttention, GDN kernels, fused RMSNorm, FLCE Phase B, BF16 LoRA storage,
and several inference/decode fusions. This branch has added CUDA training capability telemetry, a
lightweight TensorId residency registry, autograd-safe candle-CUDA LoRA/projection hooks, explicit
attention training declines, and resident in-place SGD/AdamW kernels for registered CUDA tensors.
It still does **not** have a native CUDA tensor/autograd stack or an end-to-end real-model proof that
adapter saves observe every backend-resident optimizer update.

The CUDA port should therefore not copy Vulkan's buffer-upload mechanics blindly. CUDA candle tensors
already live on the device, so the first useful parity target is to make CUDA training decisions
explicit, testable, and observable:

- decline inference-only kernels for tracked tensors unless they preserve autograd;
- expose CUDA training capabilities separately from decode capabilities;
- add device-resident optimizer/LoRA hooks only when they truly avoid host synchronization;
- preserve the existing long-context safety gates that Phase 10 proved necessary.

## Current CUDA Training Baseline

| Area | Current evidence | Status |
| --- | --- | --- |
| Fit gate | `crates/kiln-server/src/training_preflight.rs` has a shared working-set estimator, treats CUDA/discrete devices as `WeightResidency::SingleCopy`, accepts small A6000 payloads, and rejects 64k-token Qwen3.5-4B SFT on the A6000 budget. | Present, shared with Vulkan. |
| FLCE | `crates/kiln-flce-kernel` implements Phase B chunked-vocab CustomOp; `PHASE10_CLOSURE.md` records T=8192 A40 closure and A6000 prediction. | Present. Must remain default for SFT. |
| RMSNorm training | `forward.rs::rms_norm` routes autograd tensors to `fused_rmsnorm_with_autograd` only behind the 47 GiB gate; small GPUs fall back. | Present with safety gate. |
| Attention training | CUDA FlashAttention prefill/paged-decode hooks now decline tracked tensors; no CUDA attention backward op is claimed yet. | Honest decline present; CUDA attention training op still missing. |
| LoRA precision | `compute_lora_delta` casts A/B to `x.dtype()`; LoRA Vars initialize as BF16. `PHASE10_LORA_PRECISION_STUDY.md` closed performance as null but accepted parity/safety. | Present. |
| CUDA decode LoRA | `CudaBackend::lora_decode_add` declines tracked tensors and only runs the forward-only fused add for inference. | Correct for safety, not a training acceleration. |
| Resident activation registry | CUDA implements `register`, `has`, `update`, and `evict` TensorId metadata hooks while keeping `resolve` conservative unless a caller already owns the tensor. | Present as lifecycle/telemetry registry; no false side-buffer ownership claimed. |
| Device optimizer dispatch | CUDA implements resident in-place SGD and AdamW kernels for registered contiguous CUDA F32/BF16 tensors, with first-use telemetry and fallback declines for unsupported tensors. | Kernel path present; trainer-level engagement and adapter-save readback still need end-to-end proof. |
| Autograd-safe projection backend op | `CudaBackend::linear_prefill_apply` and `linear_prefill_apply_offset` route compatible CUDA matmuls through candle CUDA autograd. | Present for tested shapes; broader layer-routing proof still pending. |
| Native CUDA training stack | No CUDA equivalent of `vk_train.rs`, `vk_tensor.rs`, or `vk_forward.rs`. | Missing. |

## Phase Plan

### Phase C0: Capability and Telemetry Baseline

Goal: make the CUDA training surface explicit before changing math.

Tasks:

- add CUDA training capability logging at startup or first backend use;
- expose which training paths are CUDA-native, candle-CUDA, or declined;
- add tests that CUDA inference-only hooks decline tracked tensors;
- document the exact completion criteria copied from the Vulkan report.

Acceptance:

- CPU-host tests can verify default trait declines without requiring a GPU;
- CUDA-feature builds compile with the new capability surface;
- operator logs can distinguish "safe candle CUDA training path" from "backend-owned CUDA training
  kernel engaged."

### Phase C1: CUDA Resident Registry Semantics

Goal: implement the `BackendRuntime` resident hooks for CUDA without adding false host-copy claims.

Design constraint: candle CUDA tensors are already device-resident. A CUDA "registry" should key
logical training tensors and provide lifecycle/telemetry first; it should only add side buffers when
we need pointer stability, stale-host semantics, or custom kernels.

Tasks:

- implement `supports_resident_activation`, `register_resident_activation`, `has`, `evict`, and
  `update` for CUDA as a lightweight TensorId registry;
- keep `resolve_resident_activation` conservative until there is a real custom buffer source;
- add lifecycle tests that run without a CUDA device if possible, and CUDA-feature tests otherwise;
- wire first-dispatch logs for LoRA and optimizer hooks.

Acceptance:

- register/has/evict/re-register semantics match Vulkan's caller contract;
- no training path silently claims stale candle storage unless CUDA owns an alternate buffer;
- checkpointed training tests still pass with CUDA hooks compiled in.

### Phase C2: Autograd-Safe CUDA LoRA Delta Path

Goal: make CUDA LoRA training engagement explicit and preserve gradients.

Current fallthrough already uses candle CUDA `broadcast_matmul`, which is autograd-safe. The first
CUDA implementation should wrap or report this path rather than replacing it with a leaf tensor.

Tasks:

- implement a CUDA `lora_delta_resident` path only if it returns a tensor connected to `x`, `A`, and
  `B`;
- if the implementation delegates to candle CUDA matmul, label it honestly as candle-CUDA resident;
- add gradient parity and nonzero-LoRA-gradient tests;
- preserve `lora_decode_add` as inference-only.

Acceptance:

- loss backward produces gradients for `A` and `B`;
- forced CUDA resident path matches `compute_lora_delta`;
- tracked tensors never route through `kiln_rmsnorm_kernel::lora_decode_add`.

### Phase C3: Device Optimizer Dispatch

Goal: update trainable state without requiring a host readback between steps.

Tasks:

- add custom CUDA SGD and AdamW kernels or a candle-CUDA in-place equivalent that updates the actual
  canonical tensor storage;
- only return `true` from `dispatch_sgd_step` / `dispatch_adamw_step` when the backend really owns
  the update;
- add F32/BF16 parity, shape-mismatch error, missing-residency decline, and lazy-sync tests.

Acceptance:

- CUDA optimizer step matches CPU reference for F32 and BF16;
- stale-host/current-after-sync behavior is explicit if a side buffer is introduced;
- adapter save sees current LoRA weights.

### Phase C4: Projection and LM-Head Training Dispatch

Goal: add the CUDA equivalent of Vulkan's autograd-safe projection and offset matmul hooks.

Tasks:

- implement `linear_prefill_apply` for training tensors with backward;
- implement `linear_prefill_apply_offset` for FLCE chunks without re-uploading or slicing into
  dangerous submits;
- add output-dim and row chunking ceilings;
- prove q/k/v/o, GDN in-proj, gate/up/down, and lm_head route correctly.

Acceptance:

- forward/backward parity for projection shapes;
- long-shape dispatch is chunked, not monolithic;
- FLCE never materializes `[tokens, vocab]` logits.

### Phase C5: Native CUDA Training Stack

Goal: converge on the same end state as `vk_native_sft_train`, but for CUDA.

Tasks:

- add a CUDA-native tensor/autograd module only after C1-C4 establish the contracts;
- wire Qwen3.5 FullAttention and GDN forward/backward;
- add checkpointed SFT, AdamW, adapter save, and real-model server routing;
- add an arena/pool to avoid allocation churn.

Acceptance:

- one-layer native CUDA smoke loss decreases;
- hybrid Qwen3.5-4B SFT succeeds on a real model;
- planned hot-path readbacks are limited to scalar loss logging and adapter/checkpoint save.

## Commit Discipline

This branch is expected to be long-lived and multi-commit. Each commit should include:

- one phase-sized behavior change or one audit/runbook update;
- a test or compile gate that actually covers the changed behavior;
- a push to the configured remote before starting the next phase.

Current configured `origin` in this workspace is `/data/repo-cache/ericflo/kiln.git`, not a GitHub
HTTPS/SSH remote. Push evidence should therefore record the exact remote used until a GitHub remote
is configured.

## Completion Criteria

CUDA should not be called equivalent to the Vulkan training modernization until the branch proves:

- impossible training jobs are rejected before allocation;
- default SFT uses FLCE or equivalent chunked loss;
- training projections and LoRA deltas are autograd-safe;
- RMSNorm and attention training paths are on GPU for production shapes;
- checkpoint boundaries and trainable LoRA state have explicit CUDA residency semantics;
- SGD and AdamW update CUDA-resident state in place or honestly decline;
- adapter save is the only required trainable-state readback;
- a hybrid Qwen3.5-4B SFT job succeeds on real CUDA hardware;
- operator logs prove the CUDA training paths that engaged;
- parity tests cover forward, backward, fallback, shape errors, BF16/F32, and end-to-end loss
  decrease.
