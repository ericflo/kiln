# Vulkan Training Modernization Branch Report

**Date:** 2026-05-11
**Branch audited:** `main`
**Committed range:** `origin/main..HEAD`
**Base:** `99aba44c` - `Detect Vulkan VK_ERROR_DEVICE_LOST and short-circuit subsequent dispatches (#1021)`
**Head:** `10b96405` - `vk-native training: GPU-resident VkTensor + autograd + Qwen-style stack`
**Committed scope:** 121 commits, 111 files changed, 18,294 insertions, 153 deletions.
**Latest milestone:** `10b96405` commits the vk-native training prototype that was previously
worktree-local: 66 files, 7,853 insertions, and the native tensor/autograd stack described below.

## Executive Summary

This branch is the first complete pass at making Kiln training survivable on commodity GPU memory,
with Vulkan as the reference backend. It attacks the problem from both sides:

1. Stop accepting training jobs that cannot fit on unified-memory hardware.
2. Move the high-FLOP training path to GPU kernels.
3. Bound every dangerous submit by chunking or paging instead of launching monolithic work.
4. Keep trainable state resident on device and read it back only at adapter-save boundaries.
5. Build the next-generation native tensor/autograd path that removes candle's remaining CPU
   autograd storage contract entirely.

The committed stack is an "Option C" bridge: it keeps candle as the public tensor/autograd surface
while routing the expensive training pieces through Vulkan and a device-resident registry. It now
covers projection matmuls, FLCE chunk matmuls, RMSNorm, SDPA prefill, LoRA deltas, SGD, AdamW,
checkpoint activation boundaries, memory preflight, and operator telemetry.

The latest committed vk-native prototype is an "Option A"-style stack proving the end-state
architecture is viable: a `VkTensor` type, eager autograd tape, Vulkan op library, FLCE loss,
transformer-layer forward, and a multi-step AdamW smoke test where loss decreases with all forward
intermediates, gradients, and optimizer buffers living in Vulkan memory. The smoke test runs on a
synthetic one-layer model; production wiring against real Qwen3.5-4B safetensors is enumerated as
follow-up work in `docs/vk_native_training.md` and is not yet committed.

For CUDA and Metal ports, the important lesson is not "copy the Vulkan shaders." The reusable
design is:

- use a backend-owned resident buffer registry keyed by logical tensor ids;
- make training ops autograd-safe instead of inference-only fast paths;
- page/chunk vocab, batch, sequence, and optimizer work before dispatch;
- never materialize `[tokens, vocab]` logits or full-size CPU mirrors when a streamed reduction will
  do;
- make CPU fallback explicit, tested, and visible in telemetry;
- only sync trainable state to candle/host storage when saving an adapter.

## Problem Statement

The original pressure point was Qwen3.5-4B LoRA training on an AMD Strix Halo class unified-memory
APU. The box had roughly 30 GB of system memory, but the Linux DRM VRAM report could describe a much
larger BIOS/GTT carveout. Sizing training against that raw number allowed the server to accept jobs
that could exhaust system RAM.

The compute failure was equally concrete. The SFT repro around `T=918` could route the training
lm_head matmul as a single Vulkan submit:

```text
[918, 2560] @ [2560, 152064] ~= 715 GFLOP
```

On a 40-CU APU, that meant millions of workgroups in one dispatch. The machine hard-froze twice.
No kernel recovery log was left; the display compositor was starved badly enough that the host
needed physical reboot.

The memory failure had two layers:

- the unfused LM head creates a huge `[active_tokens, vocab]` logits tensor and corresponding
  backward storage;
- candle's `CustomOp::cpu_fwd` contract materializes CPU `Tensor` storage for forward outputs even
  when the real computation happened on Vulkan.

The branch therefore pursues a staged plan:

1. reject impossible jobs up front;
2. put the heavy math on GPU;
3. chunk or page every large submit;
4. keep weights, LoRA params, optimizer moments, and checkpoint boundaries resident;
5. eventually replace candle's tensor/autograd storage for training.

## Branch Inventory

The committed diff concentrates in these areas:

| Area | Evidence |
| --- | --- |
| Vulkan backend and training op surface | `crates/kiln-model/src/backend/vulkan.rs`, `vulkan_linear_op.rs`, `vulkan_lora_op.rs` |
| Vulkan kernels | `crates/kiln-vulkan-kernel/src/kernels.rs`, new shaders for linear offset/transposed, RMSNorm, SDPA, SGD, AdamW |
| Trainer integration | `crates/kiln-train/src/trainer.rs`, `crates/kiln-train/src/lib.rs` |
| Memory budget and preflight | `crates/kiln-core/src/vram.rs`, `crates/kiln-server/src/training_preflight.rs` |
| Operator docs and validation | `docs/audits/*2026-05-11.md`, `scripts/phase2_validation_steps_1_2_3.sh`, `CHANGELOG.md` |
| Native vk-training stack | `crates/kiln-vulkan-kernel/src/vk_tensor.rs`, `crates/kiln-vulkan-kernel/src/vk_ops/*`, `crates/kiln-model/src/vk_forward.rs`, `crates/kiln-train/src/vk_train.rs` |

Largest committed files by insertions:

| File | Added lines | Purpose |
| --- | ---: | --- |
| `crates/kiln-model/src/backend/vulkan.rs` | 1860 | Backend registry, dispatch routing, AdamW/SGD, LoRA, SDPA, weight drops |
| `crates/kiln-model/src/backend/vulkan_linear_op.rs` | 1295 | Autograd-safe projection op with chunked fwd/bwd |
| `crates/kiln-train/src/trainer.rs` | 1099 | Trainer residency, FLCE provider, optimizers, checkpoint integration |
| `crates/kiln-vulkan-kernel/src/kernels.rs` | 998 | Kernel helpers and dispatch wrappers |
| `crates/kiln-server/src/training_preflight.rs` | 708 | Fit estimator and HTTP 413 rejection |
| `crates/kiln-model/src/forward.rs` | 545 | Training/inference routing for GPU-backed ops |
| `crates/kiln-vulkan-kernel/src/vk_ops/flce.rs` | 507 | Native Vulkan FLCE forward/backward |
| `crates/kiln-model/src/backend/vulkan_lora_op.rs` | 479 | Autograd-safe LoRA delta op |
| `crates/kiln-vulkan-kernel/src/vk_tensor.rs` | 419 | Native GPU-resident tensor shell |
| `crates/kiln-core/src/vram.rs` | 382 | Unified-memory budget correction |
| `crates/kiln-vulkan-kernel/tests/vk_tensor_parity.rs` | 372 | Native tensor/autograd parity tests |
| `crates/kiln-model/src/vk_forward.rs` | 361 | Native Qwen-style forward stack |
| `crates/kiln-train/tests/vk_train_smoke.rs` | 353 | Native end-to-end training smoke tests |

The latest native-training commit adds 7,853 insertions across 66 files, including 29 `vk_*.comp`
shaders, 17 `vk_ops` Rust modules, 6 Vulkan parity-test files, and
`crates/kiln-train/tests/vk_train_smoke.rs`.

## Timeline by Phase

### Phase 0: Fit Before Running

Key commits:

- `1649a269` - unified-memory APU detection and DRM budget correction.
- `1ba1c0e8` - HTTP 413 training rejection with actionable hint.
- `9bb8dba2` - preflight hardening after live-host crash.
- `ea4512f4` - surface `vram_source` in rejection messages.
- `ebd15fad` - honest GPU memory budget logging and env docs.

The branch stops treating raw DRM VRAM as the truth on unified-memory APUs. `kiln-core::vram` now
distinguishes discrete Linux DRM devices from AMD/Intel unified-memory devices by looking at reported
VRAM, GTT, PCI vendor/class, and `MemTotal`. On unified memory it caps the budget to:

```text
min(drm_reported_vram, MemTotal - reserve)
```

with the reserve defaulting to `max(6 GB, MemTotal / 4)` and overrideable through
`KILN_TRAINING_MEMORY_RESERVE_GB`.

`crates/kiln-server/src/training_preflight.rs` then estimates training working set before accepting
SFT/GRPO jobs. Rejections are HTTP 413 and include the knobs an operator can actually change:
reduce sequence length, raise checkpoint segments, lower memory reserve if safe, or use a smaller
model/rank.

Porting implication for CUDA/Metal:

- CUDA cannot rely only on `nvidia-smi` or CUDA memory APIs if the host is unified memory, WSL, or
  oversubscribed.
- Metal must treat Apple Silicon as system memory first, not discrete VRAM.
- Both ports need a common "effective training budget" object with provenance, not an anonymous
  number.

### Phase 1: Shed Dead CPU Weight Copies

Key commits:

- `52c42fd1` - drop projection originals when Vulkan is active.
- `95ff8676` - stub `embed_tokens` CPU storage on Vulkan-active processes.
- `a15e0f65` - stub pre-transposed bf16 weight caches after upload.
- `f20a2fed` - document remaining candle CPU residency.
- `e0ac5888` and `93c94949` - keep transposed weight stubs shape-preserving after reverting the
  config-derived q/k/v/o shape experiment.

The committed path recognizes that candle has no `Device::Vulkan`. Vulkan-active training still
looks like `Device::Cpu` to many helpers. The branch adds a process-global `vulkan_active()` flag so
model-loading code can drop CPU-only weight copies after Vulkan has uploaded persistent buffers.

The big wins are:

- `embed_tokens` CPU table becomes a one-element BF16 stub after the transposed/uploaded path is
  ready;
- projection-original BF16 tensors are dropped after their transposed kernel views exist;
- pre-transposed BF16 cache tensors can be stubbed after the Vulkan backend rekeys its buffer cache.

The candle residency audit states that multi-GB base-model duplicates are already eliminated on the
Vulkan path. The remaining committed candle CPU residency is structural:

| Category | Approx Qwen3.5-4B footprint | Why it remains in committed bridge |
| --- | ---: | --- |
| LoRA Vars and AdamW moments | about 45 MB | candle `Var` owns shape-sized storage |
| Runtime intermediates per checkpoint segment | about 400 MB peak | `CustomOp::cpu_fwd` must return real `CpuStorage` |

Porting implication:

- CUDA/Metal ports should adopt the same "drop after upload" rule immediately.
- Do not wait for a native tensor rewrite to reclaim dead base weights.
- Any host mirror that is not needed for optimizer math or adapter save should be treated as a bug.

### Phase 2: Autograd-Safe GPU Projection Matmul

Key commits:

- `58add2ff` - `VulkanLinearOp` skeleton and parity tests.
- `4abe85eb` - backward implementation and autograd parity.
- `dd8acb7f` - training projection routing through `track_op()`.
- `42cf91cd` - GDN linear-attention in-projections route through the op.
- `d071fecc` - non-FLCE LM head training matmul routes through the op.
- `53e9c554` - backward uses a transposed Vulkan kernel.

This is the core "do not confuse inference fast path with training fast path" work. The existing
Vulkan inference kernels could produce a tensor, but a training tensor must be connected to candle
autograd. `VulkanLinearOp` is a `CustomOp1` that wraps the Vulkan matmul and implements `bwd`.

Forward:

- accepts training activations shaped like candle's broadcast matmul path;
- uploads or reuses cached F32/BF16-packed transposed weights;
- returns a candle tensor with the correct shape and dtype;
- for BF16-packed weights, can chunk internally along output dimension.

Backward:

- computes `dX = dY @ W.T`;
- reuses the same BF16-packed forward weight buffer;
- chunks oversized backward along batch/row dimension.

This made projection routing possible for q/k/v/o, GDN in-projections, MLP gate/up/down, and the
non-FLCE LM head bridge.

Porting implication:

- CUDA and Metal need training ops, not only decode/prefill inference kernels.
- Each port needs an autograd contract: either a candle `CustomOp` wrapper today or a native
  backend-tensor backward op tomorrow.
- The backward path must be designed at the same time as forward. A forward-only kernel that returns
  a leaf tensor silently breaks training.

### Phase 3: Chunk the Dangerous Work

Key commits:

- `1b8f5f97` - guard `VulkanLinearOp` dispatches after the host-hang repro.
- `9a50164b` - chunk oversized `VulkanLinearOp` dispatches instead of falling to CPU.
- `2ac00877` - tighten ceiling from 100 GFLOP to 20 GFLOP per submit.
- `c279f0d2` - BF16-packed offset linear kernel.
- `c3ca0f5a` - FLCE provider uses offset dispatch.
- `ca4f53ef` - `linear_prefill_apply_offset` sub-chunks instead of bailing.
- `4106f99e`, `c08636db`, `247fd95a` - one-shot trace lines and validation summaries.
- `5d3191a2`, `136c2e0b`, `dd5d4d2e`, `cec1defa`, `b05aa449` - query and enforce real device
  per-axis workgroup limits.

The branch establishes a rule: large training dispatches are paged by construction. The default
ceiling is 20 GFLOP per submit through `KILN_VULKAN_LINEAR_MAX_GFLOP`.

For forward BF16-packed projection:

```text
chunk_out_dim = floor(max_gflop / (2 * rows * hidden))
dispatch x [rows, hidden] against weight[:, chunk_start:chunk_start + chunk_out_dim]
concat chunks on output dim
```

For backward:

```text
chunk_batch = floor(max_gflop / (2 * out_dim * hidden))
dispatch grad_y rows in chunks
concat chunks on row dim
```

For FLCE, the branch avoids re-uploading a narrowed vocab slice by dispatching against offsets into
the full uploaded weight buffer. This is the same idea CUDA and Metal should apply to paged vocab
and paged weight tiles.

Porting implication:

- Every port should expose a generic "max work per submit" guard and make it backend-specific.
- CUDA may rely on streams and preemption differently, but it still needs upper bounds for watchdog,
  WDDM, and commodity laptop responsiveness.
- Metal needs the same protection for unified-memory Macs where display and compute share the GPU.
- Offset-addressable kernels are mandatory for efficient paging. Re-uploading every page defeats the
  design.

### Phase 4: Fused Linear Cross-Entropy as the Default SFT Loss Shape

Key commits:

- `9bbed4f9` - `KILN_USE_FLCE` default-on.
- `977df6f6` - FLCE matmul-provider hook.
- `6c19d3df` - trainer FLCE call sites use provider.
- `0b04b7bc`, `22c746d5`, `6913bc36` - gate, re-gate, then auto-enable by heuristic.
- `6182f746` - lower auto threshold to SFT-relevant `active_count >= 16`.
- `48d6c686` - simplify away unused vocab-size parameter.

FLCE is now treated as a memory and safety primitive, not just a speed optimization. The naive loss
path materializes logits shaped `[active_tokens, vocab]`. With Qwen-class vocabularies, that is the
wrong shape to allocate on commodity hardware.

The FLCE path chunks vocab, computes online log-sum-exp and correct-class gather, and avoids the
full logits tensor. On the Vulkan bridge, each chunk's matmul can be backed by
`linear_prefill_apply_offset`, so the same uploaded LM head buffer is reused across vocab pages.

The auto-engage rule moved from a large `active_count * num_chunks` threshold to a small supervised
token floor. After the host hangs, the comparison changed: FLCE was not merely faster than the
unfused path, it was the safe paged path.

Porting implication:

- CUDA and Metal training should use FLCE or an equivalent paged vocab loss by default.
- Do not build CUDA/Metal modernization around a full `[T, V]` logits allocation.
- The provider interface should take full weight plus chunk metadata so the backend can use offset
  addressing.

### Phase 5: RMSNorm and SDPA Move Off CPU

Key commits:

- `d86c6dc6` - Qwen3.5 RMSNorm forward kernel and routing.
- `fad89e5d` - RMSNorm backward kernel and training opt-in/auto.
- `aa64c155` - auto-enable training RMSNorm at `row_count >= 1024`.
- `f58bf98e` - audit remaining CPU matmul leaks.
- `dc4664ed` - replace buggy FlashAttention placeholder with `sdpa_prefill_f32`.
- `540cfbbf` - wire `flash_attn_prefill_vulkan` to SDPA.
- `02dd31a4` - realistic SDPA parity test.
- `7e7d383c`, `d073780d` - audit docs updated to show SDPA closure and roughly 100 percent hot
  forward FLOPs on GPU.
- `e501663f` - default `KILN_VULKAN_LINEAR` and `KILN_VULKAN_SDPA` on.

RMSNorm matters because training calls it repeatedly and candle decomposes it into several CPU ops.
The branch adds Qwen-style `(1 + weight) * x * rsqrt(mean(x^2) + eps)` forward and a backward kernel
for `dX`. Base RMSNorm weights are frozen in LoRA training, so the committed backward intentionally
does not compute `dW`.

SDPA matters because the full-attention prefill had remaining CPU `Q @ K.T` and `softmax @ V`
matmuls. `sdpa_prefill_f32.comp` runs online softmax with one workgroup per `(batch, head, query)`
and supports head_dim up to 128, matching Qwen3.5-4B.

Porting implication:

- CUDA and Metal ports should not leave "only 56 GFLOP" SDPA CPU leaks in place. Once the major
  projections are GPU-resident, these leftovers become the next bottleneck.
- In the candle bridge, forward-only SDPA must be validated carefully for any route that needs
  q/k/v gradients. The native tensor path should either compose SDPA from differentiable ops or add
  a fused attention backward.
- RMSNorm thresholds should be empirical. Below small row counts, dispatch overhead can beat compute
  savings.

### Phase 6: Resident Activation Registry and Checkpoint Boundaries

Key commits:

- `5d7fed8e` - `BackendRuntime` hooks for resident activations.
- `1f24df91` - Vulkan implementation.
- `765836f2`, `3bbf7645` - checkpoint and GRPO lifecycle wiring.
- `358198c0` - capability bit to gate hooks.
- `a9c80759` - `resolve_resident_activation`.
- `52e8cec6`, `11a0cd57`, `cba2ea40`, `dbcb08b0`, `925bf051`, `3ed8abf0` - resolve and drop
  candle CPU boundary mirrors across monolithic and tiled paths.
- `770f487f` - process-global registry instead of thread-local.
- `c76d1e21`, `fb3cd045`, `a537bac5` - edge-case and re-registration coverage.

The resident activation registry is keyed by candle `TensorId` and stores `Arc<VulkanBuffer>`,
shape, dtype, and element count. It lets training code register a checkpoint boundary, later resolve
it from device memory, and evict it when no longer needed.

This is the bridge version of paged activation memory:

1. checkpoint forward creates a boundary tensor;
2. backend uploads it to the registry;
3. when recomputing a segment, trainer prefers `resolve_resident_activation`;
4. once a boundary has been consumed, trainer evicts the registry entry and replaces the candle
   mirror with a one-element BF16 stub.

The registry became process-global because candle/rayon worker threads need to see the same entries
as the thread that registered them.

Porting implication:

- CUDA/Metal need the same logical registry even if their buffer types differ.
- Keying by logical tensor id is more important than using Vulkan's exact data structure.
- Activation paging should be tied to checkpoint segment lifecycle, not bolted onto kernels.

### Phase 7: LoRA Parameters, LoRA Delta, and Optimizers Become Resident

Key commits:

- `d471ecf6` - register `TrainableLoraParams` in the resident registry.
- `a48d1f8f` - `lora_delta_resident` dispatches on device.
- `bb3cbcef` - keep registry in sync with candle CPU SGD updates.
- `37e2453d` - temporarily gate resident LoRA delta to inference-only.
- `15ffacc0` - make LoRA delta training-safe through `VulkanLoraOp` `CustomOp3`.
- `d3744141` - inference adapters also register A/B tensors.
- `d53d7f78` - evict LoRA Vars at training completion.
- `c016bcb0` - BF16 SGD and trainer end-to-end dispatch.
- `28f7f2ca` - on-device AdamW, moments, trainer wiring.
- `d2eb4da6` - lazy candle-storage sync and no per-step CPU readback.
- `1caf8c32` - propagate dispatch errors instead of silent fallback.

The branch moves the trainable LoRA state into the same resident registry. The forward path computes:

```text
delta = (x @ A.T @ B.T) * scale
```

using BF16-packed registry buffers for `A` and `B`. The final training-safe implementation wraps the
dispatch in `VulkanLoraOp`, a candle `CustomOp3` that returns gradients for `x`, `A`, and `B`.

The optimizer path now prefers device-resident updates:

- `dispatch_sgd_step_f32`
- `dispatch_sgd_step_bf16`
- `dispatch_adamw_step_f32`
- `dispatch_adamw_step_bf16`

AdamW updates `param`, `m`, and `v` in place. Host code computes bias-correction terms and passes
them through push constants. BF16 variants use lane unpack/pack without requiring
`VK_KHR_shader_bfloat16`.

After this point, registry buffers are canonical between training steps. Candle `Var` storage is
allowed to be stale. `TrainableLoraParams::sync_to_candle` and `OptimizerState::sync_to_candle` pull
current bytes back only before `save_peft` or checkpoint serialization.

Porting implication:

- CUDA and Metal should not implement optimizer residency as "dispatch then immediately copy back."
  That preserves the bottleneck.
- The adapter-save boundary is the right host synchronization point.
- BF16 optimizers need F32 math internally, but not necessarily F32 storage.
- Error handling must distinguish "backend declines, use CPU fallback" from "backend accepted but
  parameters are mismatched." The latter should fail loudly.

### Phase 8: Operator Controls, Validation, and Documentation

Key commits:

- `0fa3900b`, `ce6c7812`, `a2c47e40`, `3802cba7`, `13b94bac`, `570572ba` - common env parsing and
  test locking.
- `c68f4a40`, `4a5a6c50`, `51a87d7f`, `120f9b3c`, `333e97a2` - startup profile and one-shot traces.
- `66a7b902`, `e04a1623`, `6ac48d21`, `77aba406`, `a274bd21`, `c47cdd7f`, `0f4856d8` - runbook,
  env reference, validation scripts.
- `2f4a30e7`, `5e6880d6`, `d73ae56d`, `bd0aef9e`, `f3b0d666`, `726061c6`, `c426ae75`, `7e77f58c`,
  `1f93142d` - changelog snapshots.

The branch gives operators and future port authors a control plane:

| Env var | Final behavior |
| --- | --- |
| `KILN_VULKAN_LINEAR` | default on; opt out with `0` |
| `KILN_VULKAN_LINEAR_MAX_GFLOP` | default 20 GFLOP per submit |
| `KILN_VULKAN_SDPA` | default on; opt out with `0` |
| `KILN_VULKAN_FLCE` | tristate; auto engages at `active_count >= 16` |
| `KILN_VULKAN_RMSNORM_TRAINING` | tristate; auto engages at `row_count >= 1024` |
| `KILN_USE_FLCE` | default on |
| `KILN_GPU_MEMORY_GB` | highest-priority GPU memory override |
| `KILN_TRAINING_MEMORY_RESERVE_GB` | reserve for unified-memory hosts |
| `KILN_GRAD_CHECKPOINT_SEGMENTS` | manual activation/checkpoint memory tradeoff |

The validation runbook escalates from tiny SFT to the original T=918 repro. It also calls out trace
lines that prove whether the intended paths fired:

- `GPU memory budget`
- `Vulkan training acceleration profile`
- `VulkanLinearOp::linear_prefill_apply first dispatch`
- `VulkanLinearOp::cpu_fwd first chunked dispatch`
- `VulkanLinearOp::bwd first chunked dispatch`
- `linear_prefill_apply_offset first sub-chunked dispatch`
- `VulkanBackend::register_resident_activation first call`
- `VulkanBackend::dispatch_sgd_step first call`

Porting implication:

- CUDA/Metal modernization should ship with equivalent env controls and startup profile logging.
- "It compiled" is not an operator validation story. The port needs shape-specific trace lines that
  prove the paged path engaged.

### Supporting Work

Not every commit is a kernel or trainer change, but several support commits matter to the branch's
portability:

- `7d4aa6b0` adds `kiln-core::DeviceBuffer`, a shared abstraction for backend-owned buffers.
- `f79b4c9c` moves `VulkanBackend.vulkan_device` behind `Arc`, which is what lets candle
  `CustomOp` instances capture a backend device handle safely.
- `f59861cf` and `8c3772a5` fix and document a transposed-weight-cache writer timing flake, keeping
  tests meaningful while upload/stub behavior changes.
- `b86293ab` and `570572ba` clean up imports and env-mutation locks so feature-gated tests remain
  deterministic.
- `kiln.example.toml` now documents the memory knobs that matter for unified-memory training.
- `docs/site/demo/demo-stream-parser.py` was adjusted alongside the branch docs; it is not part of
  the training kernel path, but it is in the committed diff.

## Committed Architecture

### Dataflow in the Candle Bridge

```text
load model with candle tensors
        |
        | prewarm/upload
        v
backend persistent weight buffers
        |
        | drop or stub unused candle mirrors
        v
training forward
        |
        +-- projection/RMSNorm/SDPA/FLCE/LoRA route through Vulkan kernels
        |
        +-- checkpoint boundaries registered in resident activation registry
        |
        v
candle autograd still owns graph shell
        |
        v
grads
        |
        +-- LoRA params, grads, AdamW moments resolved in registry
        |
        v
on-device SGD/AdamW updates registry buffers
        |
        | only at save_peft/checkpoint
        v
sync registry back to candle and serialize adapter
```

The committed bridge is intentionally incremental. It leaves the public server/trainer API mostly
unchanged while moving the risky work. This is why it is the right reference for CUDA/Metal ports
that need production-safe modernization without forcing a full tensor rewrite immediately.

### BackendRuntime Additions

The branch expands `BackendRuntime` with hooks that CUDA/Metal should mirror:

| Hook | Purpose |
| --- | --- |
| `supports_resident_activation` | advertise real registry support |
| `register_resident_activation` | upload logical tensor into backend registry |
| `evict_resident_activation` | release registry entry |
| `update_resident_activation` | refresh registry bytes after CPU fallback mutation |
| `has_resident_activation` | route only when required operands are resident |
| `resolve_resident_activation` | reconstruct a candle tensor at a boundary |
| `dispatch_sgd_step` | in-place device optimizer step |
| `dispatch_adamw_step` | in-place device AdamW for param/m/v |
| `lora_delta_resident` | resident LoRA delta dispatch |
| `linear_prefill_apply` | autograd-safe projection |
| `linear_prefill_apply_offset` | paged/chunked vocab or weight-slice projection |
| `drop_uploaded_bf16_weights` | release candle CPU mirrors after device upload |

### Kernel Inventory, Committed

| Kernel/helper | Role |
| --- | --- |
| `linear_decode_batched_offset_bf16w.comp` | matmul against a column slice of an uploaded BF16-packed weight |
| `linear_decode_batched_transposed_bf16w.comp` | backward `dX` against the forward weight buffer |
| `qwen_rmsnorm_forward.comp` | Qwen RMSNorm forward |
| `qwen_rmsnorm_backward.comp` | RMSNorm `dX` for frozen base weights |
| `sdpa_prefill_f32.comp` | causal/non-causal F32 SDPA prefill with online softmax |
| `sgd_step_f32.comp` | F32 in-place SGD |
| `sgd_step_bf16.comp` | packed-BF16 in-place SGD |
| `adamw_step_f32.comp` | F32 in-place AdamW |
| `adamw_step_bf16.comp` | packed-BF16 in-place AdamW |
| `dispatch_kernel` and `run_compute_pipeline` guards | enforce real per-axis workgroup limits |

### Test and Validation Coverage

Representative committed tests:

- Vulkan linear forward, BF16-packed forward, backward, offset, transposed, and chunked parity in
  `crates/kiln-model/src/backend/vulkan_linear_op.rs`.
- Vulkan LoRA forward/backward parity in `crates/kiln-model/src/backend/vulkan_lora_op.rs`.
- Resident registry register/evict/resolve/update, SGD, BF16 SGD, AdamW F32/BF16, lazy sync, and
  fallback tests in `crates/kiln-model/src/backend/vulkan.rs`.
- FLCE auto-engage and AdamW CPU fallback tests in `crates/kiln-train/src/trainer.rs`.
- SDPA non-causal, causal, Qwen head_dim=128, and T=64 parity in
  `crates/kiln-vulkan-kernel/tests/gdn_parity.rs`.
- Preflight rejection/acceptance and unified-memory budget tests in
  `crates/kiln-server/src/training_preflight.rs`.
- DRM unified-memory, reserve override, and source display tests in `crates/kiln-core/src/vram.rs`.
- Env truthy/falsy/tristate parser tests in `crates/kiln-core/src/env_flag.rs`.

The runbook remains the required gate for production hardware confidence. Unit tests prove kernel
math and routing contracts; they do not prove the original long-run host-hang repro is closed on
every APU/driver combination.

## Native Vulkan Training Prototype

Commit `10b96405` adds `docs/vk_native_training.md` plus a self-contained vk-native training stack.
This is a working prototype demonstrating the pattern CUDA and Metal can converge toward once their
bridge work is stable; it is not yet a Qwen3.5-4B production training driver.

### What It Adds

| Component | Files |
| --- | --- |
| GPU tensor type | `crates/kiln-vulkan-kernel/src/vk_tensor.rs` |
| Eager autograd tape | `crates/kiln-vulkan-kernel/src/vk_autograd.rs` |
| Op library | `crates/kiln-vulkan-kernel/src/vk_ops/*.rs` |
| Native shaders | `crates/kiln-vulkan-kernel/csrc/shaders/vk_*.comp` |
| Transformer forward | `crates/kiln-model/src/vk_forward.rs` |
| Native training step and adapter save | `crates/kiln-train/src/vk_train.rs` |
| Native parity/smoke tests | `crates/kiln-vulkan-kernel/tests/vk_*_parity.rs`, `crates/kiln-train/tests/vk_train_smoke.rs` |

`VkTensor` wraps:

```rust
Arc<VulkanBuffer>
shape: Vec<usize>
dtype: VkDType
device: Arc<VulkanDevice>
grad_fn: Option<Arc<dyn VkBackwardOp>>
requires_grad: bool
op_id: u64
param_id: Option<TensorId>
```

Clones are cheap `Arc` clones. Buffer lifetime is refcount-driven. Parameter leaves carry candle
`TensorId` values so the prototype can still key optimizer state and adapter serialization
consistently with the existing trainer.

`vk_backward(loss)` performs a DFS topo walk, seeds scalar loss gradient with ones, walks reverse
topological order, accumulates multi-use gradients with a no-grad add kernel, and returns
`VkGradStore` keyed by parameter id.

### Native Op Surface

The prototype includes kernels and backward rules for:

| Op family | Notes |
| --- | --- |
| elementwise add/sub/mul/div | F32 analytic backward |
| sum/mean | reduction plus scalar broadcast backward |
| F32/BF16 casts | precision-drop passthrough backward |
| reshape/transpose | metadata reshape; physical transpose |
| 2D matmul | two-matmul backward |
| batched matmul | batched backward with transpose helpers |
| RMSNorm | analytic backward |
| softmax last dim | analytic backward |
| SiLU | analytic backward |
| RoPE | inverse rotation backward |
| permute rh/hr | inverse permutes |
| repeat KV heads | sum groups backward |
| mask/scale | causal mask and scale ship as in-place no-grad helpers used inside the SDPA composition; a separate `vk_scale` op with proper autograd backward is also exposed and used by the LoRA delta path (without it, scale != 1.0 would silently sever the gradient chain) |
| embedding lookup | F32 and BF16-weight lookup |
| FLCE | chunked online-LSE forward and hidden-gradient backward |
| SDPA prefill | composed from permute (rh→hr) × 3, KV-head repeat × 2, batched transpose, batched matmul (Q@K.T), in-place scale, in-place causal mask, softmax-lastdim, batched matmul (attn@V), and permute (hr→rh) |
| SwiGLU MLP | composed from linears, SiLU, multiply |
| transformer layer | RMSNorm, q/k/v/o LoRA linears, SDPA, MLP, residuals |

The native test set now has 87 GPU-backed parity/trainer tests in the commit message's accounting
(81 kernel/op tests plus 6 trainer smoke tests). The end-to-end smoke test constructs a one-layer
synthetic transformer with 7 LoRA pairs (14 trainable tensors), verifies gradients exist through
full transformer-layer -> FLCE chains, and runs 10 AdamW steps with loss dropping from 3.572 to
2.250.

### Native Prototype Boundaries

The prototype still has explicit host boundaries:

- parameter initialization/upload from candle tensors;
- adapter save through safetensors after `VkTensor::to_candle`;
- one loss-shaping slice currently implemented via readback/reupload in `vk_model_forward_loss`.

The remaining productionization work is already listed in `docs/vk_native_training.md`: real
safetensors upload into `VkModelWeights`, a server route behind a flag, a proper `vk_narrow` op,
BF16 elementwise/accumulation improvements, arena allocation, fused attention backward, mask/scale
gradient cleanup, and native checkpointing.

## CUDA and Metal Port Blueprint

The rest of this report is the direct modernization spec for CUDA and Metal.

### 1. Start With the Same Fit Gate

Port tasks:

- implement or reuse `EffectiveBudget` for each backend;
- report provenance in startup logs;
- account for unified-memory reserve before accepting training jobs;
- make 413 errors say which shape and memory category caused rejection.

Acceptance evidence:

- unit tests for discrete, unified, override, and malformed env cases;
- runbook showing a known-oversized SFT request is rejected before model/training allocation.

### 2. Make the Training Hot Path Autograd-Safe

Port tasks:

- CUDA: wrap projection kernels in a training op with backward, or expose a native tensor op if the
  CUDA path reaches native autograd first.
- Metal: same for MPS/Metal kernels; do not use decode-only fast paths for training tensors.
- Ensure q/k/v/o, GDN in-proj, gate/up/down, and lm_head all route through the training op when
  enabled.

Acceptance evidence:

- forward parity;
- backward parity for `dX`;
- a reused-parameter gradient accumulation test;
- an integration test where LoRA gradients are nonzero after a projection-heavy forward.

### 3. Page Vocab and Dispatch Work

Port tasks:

- implement offset-addressable matmul over a full uploaded weight buffer;
- implement output-dim chunking for forward;
- implement row/batch chunking for backward;
- enforce per-backend submit ceilings;
- log first chunked dispatch with shape, chunk count, and per-chunk work.

Acceptance evidence:

- chunked output equals single-shot output on small shapes;
- safety guard rejects or chunks the original lm_head repro shape;
- long-shape runbook shows compositor responsiveness.

### 4. Make FLCE the Default Loss Shape

Port tasks:

- use FLCE or equivalent paged linear-cross-entropy for SFT;
- avoid allocating `[tokens, vocab]` logits;
- wire provider interface to backend offset matmul;
- keep auto threshold small enough that real SFT always uses the paged path.

Acceptance evidence:

- FLCE parity against naive loss on tiny shapes;
- FLCE engages for T=918-like supervised token counts;
- disabling FLCE is possible for debugging but not the default.

### 5. Move Normalization and Attention

Port tasks:

- implement Qwen RMSNorm forward/backward for training;
- implement SDPA prefill for head_dim 128 and causal masking;
- if native autograd is in scope, implement SDPA backward or fused attention backward;
- keep thresholds/flags backend-specific until hardware validation is done.

Acceptance evidence:

- RMSNorm finite-difference or analytic parity;
- SDPA parity for non-causal, causal, head_dim=128, and longer sequence loops;
- startup profile shows whether attention and norm paths are on.

### 6. Add a Resident Registry

Port tasks:

- store backend buffers keyed by logical tensor id;
- support register, update, resolve, evict, and has operations;
- make lifecycle process-wide or otherwise visible to worker threads;
- use it first for checkpoint boundary states.

Acceptance evidence:

- register/resolve roundtrip;
- re-register after eviction;
- zero-byte no-op;
- checkpointed training parity with and without registry enabled;
- memory drops after boundary mirror stubbing.

### 7. Make LoRA and Optimizer State Device-Resident

Port tasks:

- register LoRA A/B vars at training start;
- route LoRA delta through backend-resident buffers with autograd-safe backward;
- register AdamW `m` and `v` moment buffers;
- dispatch SGD and AdamW in place;
- leave host/candle storage stale between steps;
- sync only before adapter save/checkpoint.

Acceptance evidence:

- LoRA forward/backward parity;
- on-device SGD and AdamW parity for F32 and BF16;
- dispatch declines when operands are not resident;
- dispatch errors on shape mismatch;
- lazy sync test proves host storage is stale after device step and current after explicit sync.

### 8. Drop Host Mirrors Aggressively

Port tasks:

- after upload, stub embeddings, projection originals, and transposed caches when no longer needed;
- rekey backend caches if the logical tensor id changes after stubbing;
- document which mirrors cannot be dropped until native tensors land.

Acceptance evidence:

- tests pin default stubbing decisions;
- memory audit before/after model load;
- adapter save still writes correct LoRA values.

### 9. Treat Telemetry as Part of the API

Port tasks:

- log memory budget and source;
- log training acceleration profile;
- log first dispatch for each major accelerated path;
- include env values and auto decisions in logs;
- keep validation scripts backend-specific but shape-identical.

Acceptance evidence:

- runbook can identify the exact missing path by absence of a trace line;
- env parser accepts the same truthy/falsy spellings across backends.

### 10. Converge on Native Tensor Training

Port tasks:

- build backend-native tensor wrappers for CUDA and Metal;
- implement eager autograd with buffer-owning backward ops;
- move candle to initialization, oracle tests, and serialization boundaries;
- add a memory arena once correctness is stable;
- replace readback/reupload shape hacks with native slice/narrow/view ops.

Acceptance evidence:

- loss decreases in a one-layer native smoke test;
- gradients exist through transformer layer -> FLCE for every LoRA pair;
- no forward intermediate requires host storage;
- adapter save is the only planned LoRA parameter readback.

## Current Known Limits

Committed bridge limits:

- candle still owns graph shells and CPU storage for many runtime intermediates;
- LoRA Var and AdamW moment candle storage cannot be shape-stubbed under candle;
- some backward math in `VulkanLoraOp::bwd` still uses candle CPU matmuls for small-rank analytic
  gradients, though it reads A/B values from registry buffers;
- runbook Step 4 remains the operator-level proof for the original T=918 host-hang repro.

Native prototype limits:

- real Qwen safetensors are not yet lazily uploaded into `VkModelWeights`;
- no server route is wired yet;
- `vk_model_forward_loss` still needs a native `vk_narrow` to remove a readback/reupload slice;
- the in-place causal mask and pre-softmax scale carry no autograd link of their own — gradient
  through them is approximate (close enough that the smoke test loss decreases monotonically, but
  not analytically exact). A clean fix wraps both as autograd ops;
- BF16 elementwise + accumulation, arena allocation, and fused flash-attention with backward all
  remain future work;
- the prototype is Vulkan-only while this report is intended to guide CUDA/Metal parity.

## Artifacts to Read Next

| Artifact | Why it matters |
| --- | --- |
| `CHANGELOG.md` Unreleased section | chronological detail of committed branch behavior |
| `docs/audits/phase2_cpu_matmul_leaks_2026-05-11.md` | original CPU leak audit and final GPU FLOP picture |
| `docs/audits/phase2_hardware_validation_runbook_2026-05-11.md` | operator validation path for host-hang risk |
| `docs/audits/phase2_env_vars_reference_2026-05-11.md` | current flags, defaults, and rollback knobs |
| `docs/audits/candle_cpu_residency_2026-05-11.md` | what candle still keeps and why |
| `docs/vk_native_training.md` | native Vulkan tensor/autograd prototype note |

## Completion Criteria for CUDA and Metal Ports

A CUDA or Metal modernization should not be considered equivalent to this branch until it can show:

- fit preflight rejects impossible unified-memory jobs before training starts;
- default SFT loss uses paged FLCE or equivalent and never allocates full logits;
- projection forward/backward are autograd-safe and chunked;
- SDPA and RMSNorm training paths are on GPU for production shapes;
- checkpoint boundary states can live in a resident registry;
- LoRA A/B and AdamW moments remain backend-resident between steps;
- optimizer steps update backend buffers in place;
- adapter save is the only required trainable-state readback;
- operator logs prove each accelerated path engaged;
- parity tests cover forward, backward, fallbacks, shape errors, BF16/F32, and end-to-end loss
  decrease.

## Commit Ledger

This is the complete committed range audited for this report, oldest first. Subjects are copied
verbatim from `git log origin/main..HEAD --format='%h %s' --reverse`.

```text
9bbed4f9 Default KILN_USE_FLCE on; add explicit opt-out in parity tests
1649a269 Detect unified-memory APUs and correct the DRM-reported VRAM budget
52c42fd1 Auto-drop projection originals when Vulkan is the active backend
1ba1c0e8 Reject training submissions that won't fit with HTTP 413 + actionable hint
7d4aa6b0 Add DeviceBuffer abstraction in kiln-core
9bb8dba2 Harden training preflight after live-host crash on the repro payload
95ff8676 Stub embed_tokens CPU storage on Vulkan-active processes
f79b4c9c Make VulkanBackend.vulkan_device an Arc for op-state capture
58add2ff Add VulkanLinearOp CustomOp1 skeleton + parity tests
4abe85eb Implement VulkanLinearOp backward + autograd parity test
dd8acb7f Wire VulkanLinearOp into training projection path via track_op() routing
42cf91cd Route GDN linear-attention in-projections through VulkanLinearOp
d071fecc Route non-FLCE LM head matmul through VulkanLinearOp on training
977df6f6 Add FLCE matmul-provider hook for routing chunk matmul to Vulkan
6c19d3df Wire BackendFlceProvider into trainer FLCE call sites
0b04b7bc Gate BackendFlceProvider behind KILN_VULKAN_FLCE=1
80899fc7 Default KILN_VULKAN_LINEAR on; add explicit opt-out
c279f0d2 Add buffer-offset bf16-packed linear kernel for chunked dispatch
c3ca0f5a Wire FLCE provider to use linear_prefill_apply_offset; default-on
22c746d5 Re-gate FLCE provider behind opt-in until cross-T heuristic lands
6913bc36 Auto-enable FLCE provider when payload predicts net dispatch win
53e9c554 Route VulkanLinearOp::bwd through transposed Vulkan kernel
d86c6dc6 Add Qwen3.5 RMSNorm Vulkan forward kernel + inference routing
fad89e5d Add Qwen3.5 RMSNorm Vulkan backward kernel + autograd opt-in
aa64c155 Auto-enable Vulkan RMSNorm training path when row_count >= 1024
1b8f5f97 Guard VulkanLinearOp dispatches and revert KILN_VULKAN_LINEAR to opt-in
6182f746 Lower FLCE auto-threshold to engage at SFT-relevant batch sizes
9a50164b Chunk oversized VulkanLinearOp dispatches instead of bailing to CPU
2ac00877 Tighten chunking ceiling from 100 GFLOP to 20 GFLOP per submit
f58bf98e Phase 2.1 audit: CPU-fallback matmul leaks in Vulkan training forward
dc4664ed Add Vulkan SDPA prefill F32 kernel (replaces flash_attn placeholder)
540cfbbf Wire flash_attn_prefill_vulkan to the new SDPA F32 kernel
ebd15fad Phase 5: honest GPU memory budget logging + env-var docs
011f1378 Phase 4.2 prep: Vulkan SGD-step kernel + parity test
5d7fed8e Phase 3.1: BackendRuntime hooks for resident activation registry
02dd31a4 Add SDPA parity test at realistic seq_len (T=64, head_dim=128)
c68f4a40 Log Vulkan training acceleration profile at startup
1f24df91 Phase 3.1: Vulkan impl of resident-activation registry hooks
66a7b902 Phase 5 docs: hardware validation runbook for Phase 2 stack
2f4a30e7 CHANGELOG: Phase 2 Vulkan training hardening (unreleased)
ea4512f4 Surface vram_source in preflight HTTP 413 rejection message
4106f99e Add one-shot tracing for first chunked VulkanLinearOp dispatch
ca4f53ef linear_prefill_apply_offset: sub-chunk instead of bailing to CPU
765836f2 Phase 3.2 step: register/evict boundary activations in checkpointed path
3bbf7645 Phase 3.2 step: same activation registry wiring for GRPO checkpoint path
51a87d7f Trace first register_resident_activation call for operator visibility
e04a1623 Phase 5 docs: canonical env-var reference for Phase 2 acceleration
358198c0 Add supports_resident_activation() capability bit; gate trainer hooks
502d8604 Test: VulkanBackend advertises supports_resident_activation = true
5e6880d6 CHANGELOG: extend Phase 2 entry for the post-runbook commits
6ac48d21 Validation script for runbook Steps 1 + 2 (defaults + KILN_VULKAN_LINEAR)
8c3772a5 Document the queue_cache_write_persists test as nextest-only
f59861cf Fix transposed weight cache writer flake: recv_timeout for initial delay
b86293ab Drop unused BackendStorage import; cfg(test)-gate Device
1b4cd0a6 Add BackendRuntime::dispatch_sgd_step + Vulkan impl
120f9b3c Trace first dispatch_sgd_step call for operator visibility
d73ae56d CHANGELOG: extend Phase 2 entry for SGD trait method + writer fix
775881b8 Test: dispatch_sgd_step shape-mismatch + non-F32 dtype edge cases
77aba406 Extend validation script: add Step 3 (KILN_VULKAN_SDPA=1)
70156545 Update runbook rollback section + done-criteria
7e7d383c Audit doc: mark SDPA action items complete (kernel + wiring + tests)
c47cdd7f Validation script: pre-flight check for required tools
a9c80759 Add BackendRuntime::resolve_resident_activation + Vulkan impl
0fa3900b Add kiln_core::env_flag helper for KILN_* boolean parsing
ce6c7812 Refactor KILN_* env-flag parsers to use kiln_core::env_flag
a2c47e40 Add env_tristate(name) -> Option<bool>; refactor auto-heuristic sites
c76d1e21 register_resident_activation: bail silently on zero-byte tensor
c08636db Add one-shot trace for first offset sub-chunked dispatch
bd0aef9e CHANGELOG: extend Phase 2 entry for env_flag, resolve hook, sub-chunk telemetry
3802cba7 Refactor use_flce() to use kiln_core::env_flag
4a5a6c50 Surface Phase 3.1 + 4.2 status in startup acceleration profile log
fb3cd045 Test: re-register resident activation after eviction
52e8cec6 Phase 3.2: prefer resolve_resident_activation over clone for seg_input
f3b0d666 CHANGELOG: capture Phase 3.2 partial wiring + profile log fields
11a0cd57 Extract segment_input_via_registry_or_clone helper
cba2ea40 Phase 3.2 sub-step: drop candle CPU mirror after registry resolve
d073780d Audit doc: SDPA leak now closed; update FLOP picture to ~100% on GPU
247fd95a Validation script: print per-step trace summary on success
13b94bac env_flag tests: serialize env mutations with ENV_LOCK mutex
333e97a2 Runbook: enumerate trace lines to look for in server log
48d6c686 Drop unused vocab_size parameter from flce_auto_engage
b07ae8a5 Test: dispatch_sgd_step covers all four resident/non-resident combos
a274bd21 Validation script: add --skip-build flag for re-runs
cec1defa sdpa_prefill_f32: guard dispatch grid against Vulkan per-axis limit
b05aa449 sgd_step_f32: same Vulkan per-axis dispatch guard as SDPA
0f4856d8 Runbook: link the Steps 1-3 helper script at the top of the procedure
136c2e0b dispatch_kernel: top-level Vulkan per-axis workgroup-count guard
dd5d4d2e run_compute_pipeline: same Vulkan per-axis workgroup guard
dbcb08b0 Phase 3.2: extend resolve fast-path to tiled recompute paths
925bf051 Phase 3.2: resolve seg_output_var via registry in layer-pair tiled
d471ecf6 Phase 4.1 step 1: register LoRA Vars in resident activation registry
a48d1f8f Phase 4.1 step 2: lora_delta_resident — on-device LoRA delta from registry
bb3cbcef Phase 4.1 step 3: keep registry in sync with candle CPU SGD updates
a537bac5 Test: update_resident_activation overwrites + handles unregistered
37e2453d Gate lora_delta_resident on inference-only to preserve training autograd
3ed8abf0 Phase 3.2 tiled paths: drop boundary CPU mirror after tiled returns
726061c6 CHANGELOG: capture Phase 4.1 partial + tiled-path Phase 3.2 drop
744b909c Test: Phase 4.1 end-to-end — register → mutate → update → forward sees new weights
d53d7f78 Phase 4.1 cleanup: evict LoRA Vars from registry on training completion
570572ba Consolidate env-mutation test lock; deflake vram tests
d3744141 Phase 4.1 step 4: register inference adapters in resident registry
c426ae75 CHANGELOG: capture Phase 4.1 step 4 inference adapter wiring + test lock consolidation
15ffacc0 Phase 4.1 step 5: autograd-safe LoRA delta via CustomOp3 (training-safe)
c016bcb0 Phase 4.x: BF16 SGD kernel + trainer wires dispatch_sgd_step end-to-end
e501663f Default KILN_VULKAN_LINEAR + KILN_VULKAN_SDPA to ON
1f581a0c Docs + validation script: reflect KILN_VULKAN_{LINEAR,SDPA} default-on
7e77f58c CHANGELOG: capture Phase 4.x complete + flipped defaults
aa40d176 Add dispatch_adamw_step trait stub per plan §4.2
5d3191a2 Query actual device dispatch limit instead of spec minimum (65535)
1caf8c32 apply_sgd_update: propagate dispatch_sgd_step errors instead of swallowing
1f93142d CHANGELOG: capture dispatch-limit query, error propagation, AdamW stub
770f487f Make resident activation registry process-global (was thread-local)
d5528e61 Update stale "Phase 4.1 partial" / "inference-only" docstrings
28f7f2ca Implement on-device AdamW (Vulkan + trainer wiring)
d2eb4da6 Lazy candle-storage sync; drop per-step CPU readback
f20a2fed Document candle CPU residency state; pin stub-default tests
a15e0f65 Stub pre-transposed bf16 weight caches after Vulkan upload
434f3628 Trainer: derive LoRA q/k/v/o shapes from config
e0ac5888 Use broadcast_as for shape-preserving stub of transposed weights
93c94949 Revert trainer config-derived q/k/v/o shapes
10b96405 vk-native training: GPU-resident VkTensor + autograd + Qwen-style stack
```
