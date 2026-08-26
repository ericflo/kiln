# Vulkan Training Harmonization Plan (PR1–PR7)

Issue: #1082
Branch: `feat/vk-tape-harmonization`
Status: PR1/PR2 implemented + bounded-validated in this branch; PR3–PR7 specced.

This is the authoritative design doc for collapsing Vulkan training onto the same
kt-tape substrate that CUDA and Metal already share. It was produced from a
multi-agent investigation of the live tree; the file:line references below were
re-verified against the worktree at branch HEAD and are load-bearing.

---

## 1. Thesis: one substrate vs. a parallel fork

CUDA and Metal drive training through **one shared substrate**:

- `kiln_tensor::Tensor` (with CUDA / Metal storage backends)
- `kiln_autograd::Tape` (the autograd graph)
- kt-composite fused ops (FLCE, RMSNorm, OPD, etc.)
- `kiln_optim::AdamW` over `kiln_param::Parameter`
- orchestrated by `kiln-train/src/trainer.rs` (SFT/GRPO) and `kiln-train/src/opd.rs` (OPD)

Vulkan does **not** share this substrate. It runs a fully parallel stack of
roughly 19k LOC:

- `VkTensor` (`crates/kiln-vulkan-kernel/src/vk_tensor.rs`, 568 LOC)
- `vk_autograd` (`crates/kiln-vulkan-kernel/src/vk_autograd.rs`, 172 LOC) plus the
  per-op backward kernels under `crates/kiln-vulkan-kernel/src/vk_ops/`
- `vk_forward` (`crates/kiln-model/src/vk_forward.rs`, 1762 LOC)
- a separate on-device AdamW book
- the orchestrator `vk_train.rs` (`crates/kiln-train/src/vk_train.rs`, **6109 LOC**)

That fork exists for one reason, and removing the reason lets us delete the fork.

---

## 2. Root cause: a single missing storage primitive

`kiln_tensor::Tensor` **cannot be instantiated on `Device::Vulkan`**. Every
constructor returns `Err(... NYI ...)` on the Vulkan arm:

| Constructor          | File:line                                   | Vulkan arm |
| -------------------- | ------------------------------------------- | ---------- |
| `zeros_on`           | `crates/kiln-tensor/src/tensor.rs:237`      | `Err` NYI  |
| `from_vec_on`        | `crates/kiln-tensor/src/tensor.rs:287`      | `Err` NYI  |
| `from_raw_bytes_on`  | `crates/kiln-tensor/src/tensor.rs:351`      | `Err` NYI  |
| `to_device` (→ VK)   | `crates/kiln-tensor/src/tensor.rs:882/909`  | `Err` NYI  |

The reason every one of those is `Err` is that **the one missing primitive is the
`host_to_vulkan_copy` / `vulkan_to_host_copy` pair.** CUDA and Metal each have
theirs; Metal's live at `crates/kiln-tensor/src/metal_storage.rs:482`
(`host_to_metal_copy`) and `:555` (`metal_to_host_copy`). Vulkan has neither.

Without a way to move bytes host↔device through `kiln_tensor::Tensor`, no
`Tensor` can exist on Vulkan, so nothing in the shared substrate
(`Tape`, kt-composite ops, `AdamW`, `trainer.rs`/`opd.rs`) can run on Vulkan.
**That single gap is what forced the entire 19k-LOC fork.**

Secondary gap: `MatmulOp` has no `vulkan_fwd`. `crates/kiln-tensor/src/ops/matmul.rs`
defines `cuda_fwd` (`:113`) and `metal_fwd` (`:138`) but no Vulkan forward, so even
once tensors exist, the single hottest op cannot dispatch on Vulkan.

---

## 3. This is a finish job, not a green field

The substrate-side Vulkan plumbing is already largely built. This plan connects
existing pieces; it does not invent new GPU machinery.

- **VulkanStorage already implements `StorageBackend`** —
  `crates/kiln-tensor/src/vulkan_storage.rs:46` (struct), `:138` (`impl StorageBackend`).
- **24 of 28 `DeviceOps` already have a concrete `vulkan_fwd`.** Confirmed by
  counting concrete impls under `crates/kiln-tensor/src/ops/` (24 with `vulkan_fwd`,
  28 with `cuda_fwd`). The 4 still missing are exactly:
  `ops/matmul.rs`, `ops/reduce.rs`, `ops/scalar.rs`, `ops/silu_mul.rs`.
- **Dispatch already branches Vulkan.** `dispatch1`/`dispatch2`/`dispatch3` in
  `crates/kiln-tensor/src/device_op.rs` (`:158`, `:204`, `:249`) each have a
  `Device::Vulkan(_) => op.vulkan_fwd(...)` arm (`:163`, `:217`, `:268`).
- **The on-device Vulkan AdamW seam is built and tested.**
  `dispatch_adamw_step` is the trait method at
  `crates/kiln-model/src/backend/vulkan.rs:2147` (F32 + BF16, `TensorId`-keyed),
  with passing round-trip tests at `:5628` (F32), `:5711` (BF16 two-step), and
  `:5804` (falls back when not resident).
- **The buffer primitives exist.** `crates/kiln-vulkan-kernel/src/buffer.rs`
  provides `create_device_local` (`:58`), `upload_data` (`:224`), and
  `read_back` (`:558`) — exactly the byte-movers `host_to_vulkan_copy` /
  `vulkan_to_host_copy` need to wrap.
- **`TensorId` is already shared.** `crates/kiln-vulkan-kernel/src/vk_tensor.rs:25`
  does `pub use kiln_tensor_id::TensorId;`, so Vulkan grads key compatibly with
  the shared optimizer book — no id-remapping layer is needed.
- **Metal is fully harmonized and is the template.** Metal SFT/GRPO/OPD all run on
  the shared substrate (see `test_real_model_opd_metal`), and the recent commit
  landing OPD-on-Metal backward (`948bbe0f`) makes Metal the proof-of-pattern that
  this plan mirrors for Vulkan.

---

## 4. The 7-PR plan

Incremental spine from "ship a storage-decoupled optimizer seam first" to a
full-storage end-state. Net effect is roughly **−10k LOC** once the fork is
deleted in PR7. Each PR is independently shippable and ordered so that the
make-or-break perf item (PR3) is reached early and the big LOC-deleting flips
(PR6/PR7) are gated last.

### PR1 — Optimizer seam (storage-decoupled, ship first, numerically identical)

Route the Vulkan weight update through the shared trainer entry point
`apply_adamw_update_kt` (`crates/kiln-train/src/trainer.rs:6482`), which calls
`backend.dispatch_adamw_step` (`trainer.rs:6506`) — the **same** Vulkan AdamW
shader already used today, so this is numerically identical.

Retire:

- `VkAdamWState` / `VkAdamWBook` (`crates/kiln-train/src/vk_train.rs:622`,
  `:670`, builder through `~:707`)
- the 7+ inline `dispatch_adamw_step_f32` call sites in `vk_train.rs`
  (e.g. `:896`, `:2183`, `:2680`), which import from
  `kiln_vulkan_kernel::kernels` (`vk_train.rs:37`)

Key decoupling fact (verified): `register_resident_activation`
(`crates/kiln-model/src/backend/vulkan.rs:1894`) needs only `self.vulkan_device`
(`:1895`), **not** a Vulkan-resident `kiln_tensor::Tensor`. So the AdamW seam can
be exercised before storage lands — PR1 does not depend on PR2.

This PR is **storage-decoupled on purpose**: it is shippable and numerically
identical without any `Tensor`-on-Vulkan support, which is why it ships first.

### PR2 — Storage keystone

Add the missing primitive pair and un-NYI the constructors.

- Implement `host_to_vulkan_copy` / `vulkan_to_host_copy` mirroring
  `host_to_metal_copy` (`metal_storage.rs:482`) and `metal_to_host_copy`
  (`metal_storage.rs:555`), wrapping `VulkanBuffer::create_device_local`
  (`buffer.rs:58`), `upload_data` (`buffer.rs:224`), and `read_back`
  (`buffer.rs:558`).
- Un-NYI the 4 constructors at `tensor.rs:237/287/351` and the `to_device` Vulkan
  arm (`:882/909`).
- Give `VulkanStorage` an `Arc`-of-`VulkanBuffer` so cloned `Tensor`s share device
  memory.
- Flip the negative tests to positive round-trips:
  `zeros_on_metal_errors_until_substrate_lands`-style Vulkan negatives at
  `tensor.rs:1491` and `:1499` become host→device→host round-trip assertions.

**Round-trip correctness is the parity argument for PR2**: a value written from
host, parked on device, and read back must compare bit-exact, op-for-op against
the Metal primitive it mirrors.

### PR3 — `MatmulOp::vulkan_fwd` + zero-copy `VulkanStorage`↔`VkTensor` bridge

The make-or-break perf item.

- Add `MatmulOp::vulkan_fwd` in `ops/matmul.rs` (today it has only `cuda_fwd:113`
  and `metal_fwd:138`).
- Add a **zero-copy bridge** between `VulkanStorage` and `VkTensor` so that an op
  whose forward lives in the SPIR-V leaf layer can read/write the *same* device
  buffer the shared `Tensor` owns.

Without zero-copy, every op would bounce D2H/H2D around each kernel and the
harmonized path would be dramatically slower than the fork. This is the only real
perf lever in the plan (see §6).

### PR4 — Backward into the Tape (the long pole)

Expose the Vulkan backward kernels to the shared autograd graph.

- The Vulkan backward trait is `VkBackwardOp`
  (`crates/kiln-vulkan-kernel/src/vk_tensor.rs:61`), driven by `vk_autograd`
  (`vk_autograd.rs`). The shared graph's trait is `kiln_autograd::BackwardOp`
  (`crates/kiln-autograd/src/backward_op.rs:30`).
- Introduce a thin `VkBwdAdapter` that presents each `VkBackwardOp` as a
  `kiln_autograd::BackwardOp`, migrated **op-family by op-family behind a
  fallback** (an op without a ported adapter falls through to the existing path),
  so the migration never has a flag-day.
- Retire `vk_autograd.rs` once all families are adapted.

### PR5 — Forward harmonization

- Run `model_forward_kt` on `Device::Vulkan`.
- Add Vulkan-resident branches to the `try_tape_*_kt` recorders (the kt-composite
  fused-op recorders) so Vulkan inputs record onto the shared `Tape` like
  CUDA/Metal inputs do.
- Delete `vk_forward.rs` (`crates/kiln-model/src/vk_forward.rs`, 1762 LOC).

### PR6 — Orchestration flip (big LOC win, gated late)

Widen the gates that currently **exclude** Vulkan.

- The tape-authoritative AND gradient-checkpointing gates are
  `#[cfg(any(feature = "cuda", feature = "metal"))]` today — Vulkan is **not** in
  them. Representative sites: `trainer.rs:2478/4869/5001/5990/7217/7237/7267`,
  `opd.rs:2724/2948`, `grpo_candle_shim.rs` (many), `opd_candle_shim.rs:93`.
  PR6 extends these to include `Device::Vulkan`.
- Route Vulkan SFT/GRPO/OPD through the shared `trainer.rs` / `opd.rs`.
- Keep `KILN_VK_NATIVE_TRAINING` as an **opt-out for one release** (it currently
  routes to the native path in `kiln-server`: `main.rs:856`,
  `training_queue.rs:374/416`, `api/training.rs:169`).

Extending the gradient-checkpointing gate is **added scope, not a parity bug**:
that gate excludes Vulkan today, so there is no prior Vulkan checkpointing
behavior to preserve (see §5).

### PR7 — Delete the fork

- Delete `vk_train.rs` (`crates/kiln-train/src/vk_train.rs`, 6109 LOC).
- Collapse `vk_tensor.rs` to a thin shim (the parts still referenced by the
  SPIR-V leaf layer).
- Remove the `KILN_VK_NATIVE_TRAINING` gate.
- **KEEP all SPIR-V kernels.** The ~10k LOC under `vk_ops/` (32 kernel modules:
  attention, matmul, rmsnorm, rope, softmax, flce, opd, gdn_*, …) **are** the
  Vulkan leaf-kernel layer in the harmonized world — they are reclassified, not
  removed.
- Fix the stale docs:
  - `crates/kiln-flce-kernel/src/kt_tape.rs` `envelope_ok` comment (`:85`, `:88`).
  - `crates/kiln-train/src/opd_candle_shim.rs:85-92`, which claims the Metal OPD
    backward `bail!`s — **false at HEAD** (commit `948bbe0f` landed the Metal OPD
    backward; the comment that "the OPD Metal backward is a documented follow-up"
    is now stale).

---

## 5. Risk corrections (from adversarial verification)

Three risks were raised and then disarmed by direct inspection of the tree:

1. **Two-autograd-engine silent corruption is NOT live today.** The Vulkan and
   shared engines are split by device and there is no shared persistent tape
   carried across an in-place mutation, so there is no path for one engine's
   bookkeeping to silently corrupt the other's. This becomes *managed* work at
   PR4 (where the two graphs are deliberately bridged behind the
   op-family-by-op-family fallback), not a latent bug we are shipping on.

2. **Gradient checkpointing currently excludes Vulkan.** The gate is
   `#[cfg(any(feature = "cuda", feature = "metal"))]` and has no Vulkan arm.
   PR6 must **extend** that gate to `Device::Vulkan`. Because there is no existing
   Vulkan checkpointing behavior, this is added scope — not a regression or a
   parity break.

3. **The BF16 train-dtype concern is moot.** Vulkan LoRA weights, activations, and
   grads are **F32 by design** on the hot path. There is no BF16 training cliff to
   fall off; the BF16 AdamW arm (`vulkan.rs:5711` test) exists for parity but the
   trained path is F32.

---

## 6. Performance: parity in steady state

- The ~10k LOC of `vk_ops` SPIR-V kernels are **kept and reclassified** as the
  leaf layer (PR7), so the compute kernels that make Vulkan fast are exactly the
  ones the harmonized path dispatches.
- PR1 dispatches the **same** AdamW shader the fork already used, so the optimizer
  step is unchanged numerically and in cost.
- Vulkan trains in **F32 on the hot path** (no BF16 cliff).
- The **only** real perf lever is the **PR3 zero-copy bridge** landing clean.
  Without it, every leaf op would bounce D2H/H2D and the harmonized path would
  regress badly versus the fork. With it, steady-state throughput is at parity
  because the same kernels run over the same device buffers.

---

## 7. Validation strategy (host-safety constrained)

The development host has **hard-crashed on long training runs**. Validation is
therefore strictly bounded:

- **Allowed:** `cargo check` / `cargo build --features vulkan`; targeted named
  unit / finite-difference tests; at most a single cheap one-step real-model
  reachability smoke if one already exists.
- **Not allowed (human-gated):** full test suites, multi-step training loops,
  long-running binaries, autonomous repros, full GPU soak.

Per-PR validation:

- **PR1/PR2:** implemented + bounded-validated in this branch (named unit tests
  for the AdamW seam: `vulkan.rs:5628/5711/5804`; round-trip storage tests
  replacing the negatives at `tensor.rs:1491/1499`).
- **PR3–PR7:** specced in `docs/archive/vk-harmonization/`; each carries a bounded
  validation recipe (named tests + single-step smokes), with full GPU soak called
  out as a human-gated step rather than run autonomously.

Baseline numbers for the harmonized Vulkan path are tracked in
`bench-results/vulkan-strix-halo-baseline.md`.

---

## 8. Reference index (verified at branch HEAD)

| Claim | Location |
| ----- | -------- |
| `zeros_on` Vulkan NYI | `crates/kiln-tensor/src/tensor.rs:237` |
| `from_vec_on` Vulkan NYI | `crates/kiln-tensor/src/tensor.rs:287` |
| `from_raw_bytes_on` Vulkan NYI | `crates/kiln-tensor/src/tensor.rs:351` |
| Vulkan negative tests to flip | `crates/kiln-tensor/src/tensor.rs:1491,1499` |
| `host_to_metal_copy` (template) | `crates/kiln-tensor/src/metal_storage.rs:482` |
| `metal_to_host_copy` (template) | `crates/kiln-tensor/src/metal_storage.rs:555` |
| `VulkanStorage` struct / `impl StorageBackend` | `crates/kiln-tensor/src/vulkan_storage.rs:46,138` |
| `MatmulOp::cuda_fwd` / `metal_fwd` (no vulkan_fwd) | `crates/kiln-tensor/src/ops/matmul.rs:113,138` |
| Ops missing `vulkan_fwd` | `ops/{matmul,reduce,scalar,silu_mul}.rs` |
| `dispatch1/2/3` Vulkan arms | `crates/kiln-tensor/src/device_op.rs:163,217,268` |
| `dispatch_adamw_step` (trait method) | `crates/kiln-model/src/backend/vulkan.rs:2147` |
| AdamW tests (F32 / BF16 / fallback) | `crates/kiln-model/src/backend/vulkan.rs:5628,5711,5804` |
| `register_resident_activation` (needs only `vulkan_device`) | `crates/kiln-model/src/backend/vulkan.rs:1894` |
| `apply_adamw_update_kt` / `dispatch_adamw_step` call | `crates/kiln-train/src/trainer.rs:6482,6506` |
| `VkAdamWState` / `VkAdamWBook` to retire | `crates/kiln-train/src/vk_train.rs:622,670` |
| `dispatch_adamw_step_f32` inline call sites | `crates/kiln-train/src/vk_train.rs:896,2183,2680` |
| `create_device_local` / `upload_data` / `read_back` | `crates/kiln-vulkan-kernel/src/buffer.rs:58,224,558` |
| `TensorId` re-export (already shared) | `crates/kiln-vulkan-kernel/src/vk_tensor.rs:25` |
| `VkBackwardOp` trait | `crates/kiln-vulkan-kernel/src/vk_tensor.rs:61` |
| `kiln_autograd::BackwardOp` trait | `crates/kiln-autograd/src/backward_op.rs:30` |
| Tape-auth / grad-checkpoint gate (excludes Vulkan) | `crates/kiln-train/src/trainer.rs:2478,4869,5001,5990,7217`; `opd.rs:2724,2948` |
| `vk_forward.rs` to delete (1762 LOC) | `crates/kiln-model/src/vk_forward.rs` |
| `vk_train.rs` to delete (6109 LOC) | `crates/kiln-train/src/vk_train.rs` |
| `vk_ops` SPIR-V leaf kernels (KEEP) | `crates/kiln-vulkan-kernel/src/vk_ops/` |
| Stale `envelope_ok` comment | `crates/kiln-flce-kernel/src/kt_tape.rs:85,88` |
| Stale "Metal OPD backward bails" comment (false at HEAD) | `crates/kiln-train/src/opd_candle_shim.rs:85-92` |
| `KILN_VK_NATIVE_TRAINING` opt-out wiring | `crates/kiln-server/src/{main.rs:856,training_queue.rs:374,api/training.rs:169}` |
