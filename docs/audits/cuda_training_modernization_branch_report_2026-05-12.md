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
| `d57e5eec` | CUDA optimizer validation ledger | Records A6000 release-mode validation for resident optimizer kernels and updates the CUDA baseline. |
| `8f7041fa` | CUDA trainer optimizer dispatch proof | Adds backend dispatch counters and a CUDA trainer test proving `optimizer_step_from_map` reaches resident SGD/AdamW kernels. |
| `a60ed1e5` | CUDA trainer dispatch validation ledger | Records A6000 release-mode validation for trainer-level optimizer dispatch. |
| `0db3490b` | CUDA optimizer adapter-save proof | Extends the CUDA trainer test to save PEFT safetensors after resident optimizer dispatch and compare saved weights to updated CUDA Vars. |
| `accff468` | CUDA optimizer adapter-save validation ledger | Records A6000 release-mode validation for CUDA optimizer adapter-save behavior. |
| `8ea44f01` | CUDA training projection routing proof | Adds CUDA linear/offset dispatch counters and a trainer test proving projection matmuls plus FLCE chunk matmuls reach backend hooks. |
| `bee4af4f` | CUDA projection routing validation ledger | Records A6000 release-mode validation for CUDA projection/FLCE trainer routing. |
| `39ef69c1` | CUDA attention training fallback proof | Counts tracked FlashAttention declines and extends the CUDA trainer routing test to prove differentiable candle-CUDA attention fallback. |
| `e4fdb98e` | CUDA attention fallback validation ledger | Records A6000 release-mode validation for tracked FlashAttention decline plus candle-CUDA training fallback. |
| `84f29526` | Real Qwen3.5-4B CUDA SFT smoke ledger | Records A6000 release-mode `kiln-bench` validation on downloaded Qwen3.5-4B weights. |
| `1142aabf` | CUDA Qwen SFT smoke script | Adds `scripts/cuda_qwen_sft_smoke.sh` so the real Qwen3.5-4B CUDA one-step SFT validation is repeatable. |
| `d12fcafd` | CUDA Qwen smoke script validation ledger | Records A6000 release-mode validation of the repeatable real Qwen3.5-4B SFT smoke script. |
| `de34a2ee` | CUDA training tensor shell | Adds a CUDA-only `CudaTrainTensor` boundary that rejects CPU tensors and delegates resident SGD/AdamW updates to CUDA kernels. |
| `d9aa352a` | CUDA train tensor metadata | Adds op IDs, parameter `TensorId`, `requires_grad`, and detach metadata to the CUDA training tensor boundary. |
| `d57d499d` | CUDA training autograd scaffold | Adds a CUDA backward-op trait, reverse-topology traversal, per-`TensorId` grad store, and CUDA tensor gradient accumulation for synthetic training graphs. |
| `23efaecb` | CUDA training add backward op | Adds a CUDA tensor add op with a backward rule proving gradients route to both parameters and accumulate through shared inputs. |
| `0edef988` | CUDA training mul backward op | Adds a CUDA tensor multiply op with product-rule backward coverage using saved input values. |
| `3e0e32be` | CUDA training sum backward op | Adds scalar `cuda_sum_all` reduction with broadcast-gradient backward coverage, including `sum(x * x)` graph validation. |
| `1ce54012` | CUDA training matmul backward op | Adds 2D CUDA matmul with backward coverage for `sum(lhs @ rhs)` gradients to both operands. |
| `7a440ddf` | CUDA native SGD step helper | Connects `CudaGradStore` to resident SGD kernels and proves a tiny native CUDA loss decreases after one update. |
| `cca827fe` | CUDA native AdamW step helper | Adds CUDA AdamW config/state helpers, connects `CudaGradStore` to resident AdamW kernels, and proves a tiny native CUDA loss decreases. |
| `aeb64efe` | CUDA native linear train step | Adds a `kiln-train` CUDA helper that runs a minimal linear sum-square AdamW step through native CUDA tensor/autograd primitives. |
| `7399abcb` | CUDA training arena accounting | Adds a conservative `CudaTrainArena` that owns step-lifetime CUDA tensor handles and tracks approximate allocation bytes by dtype/shape. |

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
  - `cargo test --release -p kiln-train --features cuda cuda_optimizer_step_from_map_engages_backend_kernels --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda cuda_optimizer_step_from_map_engages_backend_kernels --lib --quiet` re-run after adding adapter-save safetensors comparison
  - `cargo test --release -p kiln-train --features cuda cuda_training_forward_uses_projection_and_flce_backend_hooks --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda cuda_training_forward_uses_projection_and_flce_backend_hooks --lib --quiet` re-run after adding tracked FlashAttention-decline assertion
  - `KILN_SPEC_METHOD=off KILN_USE_FLCE=1 cargo run --release --features cuda --bin kiln-bench -- --model-path /workspace/qwen3.5-4b --prompt-tokens 8 --max-output-tokens 1 --training-steps 1 --paged --quiet` passed after downloading `Qwen/Qwen3.5-4B` with `hf download`; CUDA backend loaded the real model, completed one SFT step with `loss=1.598035`, `2.85s/step`, and `18952 MB` peak VRAM.
  - `scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --skip-build` passed on the same A6000/model checkout with `loss=1.598035`, `2.65s/step`, and `18952 MB` peak VRAM.
  - `cargo test --release -p kiln-model --features cuda cuda_train_tensor --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_backward --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA autograd scaffolding
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA add backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA multiply backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA sum backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA matmul backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_native_sgd_step_decreases_sum_square_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding native SGD step coverage
  - `cargo test --release -p kiln-model --features cuda cuda_native_adamw_step_decreases_sum_square_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding native AdamW step coverage
  - `cargo test --release -p kiln-train --features cuda cuda_linear_adamw_train_step_decreases_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train_arena --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA training arena accounting
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
It also now has an initial CUDA-only training tensor boundary over candle CUDA storage, including
parameter metadata, detach semantics, a backward-op trait, reverse-topology traversal, and a
per-`TensorId` gradient store for a future CUDA autograd graph. CUDA add, multiply, sum reduction,
and 2D matmul ops prove that the tape can propagate and accumulate gradients through real CUDA
tensor ops, including a product rule, scalar-loss reductions, and projection-shaped matmul
gradients. Native SGD and AdamW helpers now apply those gradients through resident optimizer kernels
and prove tiny loss decreases, and `kiln-train` has a minimal CUDA-native linear AdamW train-step
bridge. A conservative CUDA training arena now owns step-lifetime tensor handles and tracks
approximate allocation bytes. It still does **not** have native CUDA Qwen forward/backward ops or a
custom pooled allocator equivalent to the Vulkan native training path.

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
| Attention training | CUDA FlashAttention prefill/paged-decode hooks decline tracked tensors; the trainer routing test proves tracked FlashAttention is offered, declined, and followed by differentiable candle-CUDA attention fallback. | Honest GPU fallback present; native CUDA attention backward op still missing. |
| LoRA precision | `compute_lora_delta` casts A/B to `x.dtype()`; LoRA Vars initialize as BF16. `PHASE10_LORA_PRECISION_STUDY.md` closed performance as null but accepted parity/safety. | Present. |
| CUDA decode LoRA | `CudaBackend::lora_decode_add` declines tracked tensors and only runs the forward-only fused add for inference. | Correct for safety, not a training acceleration. |
| Resident activation registry | CUDA implements `register`, `has`, `update`, and `evict` TensorId metadata hooks while keeping `resolve` conservative unless a caller already owns the tensor. | Present as lifecycle/telemetry registry; no false side-buffer ownership claimed. |
| Device optimizer dispatch | CUDA implements resident in-place SGD and AdamW kernels for registered contiguous CUDA F32/BF16 tensors, with first-use telemetry, dispatch counters, and fallback declines for unsupported tensors. | Kernel path, trainer-level engagement, saved adapter contents, and one-step real Qwen3.5-4B SFT smoke proven. |
| Autograd-safe projection backend op | `CudaBackend::linear_prefill_apply` and `linear_prefill_apply_offset` route compatible CUDA matmuls through candle CUDA autograd and expose dispatch counters. | Present for direct parity tests, trainer-level projection/FLCE routing, and one-step real-model smoke. |
| Native CUDA training stack | `crates/kiln-model/src/cuda_train.rs` provides an initial CUDA-only tensor shell over candle CUDA storage with op IDs, parameter `TensorId`, `requires_grad`, detach semantics, a backward-op trait, reverse-topology traversal, per-parameter grad storage, CUDA add/mul/sum/matmul backward ops, resident SGD/AdamW optimizer delegation, tiny optimizer loss-decrease proofs, and conservative arena allocation accounting. `crates/kiln-train/src/cuda_train.rs` adds a minimal linear AdamW train-step bridge. There is still no CUDA equivalent of `vk_train.rs`, native Qwen forward/backward ops, or a custom pooled training allocator. | Initial tensor/autograd/optimizer/arena boundary and train-crate bridge present; full native stack missing. |

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
