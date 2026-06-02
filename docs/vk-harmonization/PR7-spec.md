# PR7 — Delete the fork (implementation spec)

Issue: #1082
Branch: `feat/vk-tape-harmonization`
Status: SPEC ONLY. Not implemented in this branch. PR1+PR2 are code; PR3–PR7 are specs.
Authoritative design doc: `docs/vulkan-train-harmonization-plan.md` (§4 PR7, §8 reference index).

> **All line numbers below were read from the worktree at branch HEAD and WILL drift.
> Every deletion step ships with the exact `grep` that re-anchors it. Re-run the grep,
> confirm the count, THEN delete. Never delete on a line number alone.**

---

## 0. What PR7 is (and is not)

PR7 is the **mechanical teardown** that runs *after* PR3–PR6 have moved Vulkan
training onto the shared kt-tape substrate and *after* the human-gated GPU soak
has signed off. It deletes the parallel Vulkan training fork and fixes stale
docs. It introduces **no new GPU semantics** and **changes no numerics** — every
compute kernel that makes Vulkan fast is kept.

PR7 deletes:

1. `crates/kiln-train/src/vk_train.rs` (6061 LOC at HEAD; plan says 6109 — re-count, see §3.1).
2. `crates/kiln-train/tests/vk_train_smoke.rs` (3592 LOC) — the fork's smoke tests; they `use kiln_train::vk_train::*` and cannot survive the module deletion.
3. `crates/kiln-model/src/vk_forward.rs` (1762 LOC) — **PR5 already deletes this**; PR7 only confirms no importer survives (see §3.3).
4. `save_vk_lora_adapter` (`vk_train.rs:5990`) — replaced by the shared `LoraParams::save_peft` path (§3.4).
5. The `KILN_VK_NATIVE_TRAINING` / `KILN_VK_NATIVE_GRPO` / `KILN_VK_NATIVE_OPD` opt-out family (§3.5).

PR7 **collapses** (does not delete):

6. `crates/kiln-vulkan-kernel/src/vk_tensor.rs` (568 LOC) — drops the **autograd half** (the training tape), keeps the **leaf-carrier half** the SPIR-V kernels pass around. See §4 — this is the subtle one; read it before touching the file.

PR7 **keeps, untouched**:

7. All SPIR-V under `crates/kiln-vulkan-kernel/src/vk_ops/` (32 kernel modules) and `shaders/`. These are the Vulkan leaf-kernel layer in the harmonized world — reclassified, not removed (plan §4 PR7, §6).

PR7 **fixes stale docs** (§5):

8. `crates/kiln-train/src/opd_candle_shim.rs:88–92` — claims Metal OPD backward `bail!`s. **False at HEAD.**
9. `crates/kiln-opd-loss-kernel/src/kt_tape.rs:92–98` — same false claim (a THIRD site the plan did not list; found during this spec).
10. `crates/kiln-train/src/opd.rs:2943–2947` — same false claim (FOURTH site).
11. `crates/kiln-flce-kernel/src/kt_tape.rs:84–110` — `envelope_ok` comment; see §5.4 for the *honest, verify-first* instruction (it is genuinely CUDA-only at HEAD).
12. `crates/kiln-train/src/opd.rs:771` doc comment referencing the deleted `vk_native_opd_train`.

---

## 1. Prerequisites (hard ordering — do not start PR7 until all are true)

PR7 is gated last on purpose. Before the first deletion:

| Prereq | Why PR7 needs it | How to confirm |
| ------ | ---------------- | -------------- |
| **PR2 landed** | `Tensor`-on-Vulkan storage exists (`host_to_vulkan_copy:277` / `vulkan_to_host_copy:372` / `VulkanStorage::from_arc_buffer:130` in `crates/kiln-tensor/src/vulkan_storage.rs`). Already in this branch (commit `b94feeac`). | `grep -n 'pub fn host_to_vulkan_copy\|pub fn vulkan_to_host_copy' crates/kiln-tensor/src/vulkan_storage.rs` |
| **PR3 landed** | `MatmulOp::vulkan_fwd` + zero-copy `VulkanStorage`↔`VkTensor` bridge, so the shared `Tensor` and the SPIR-V leaf kernels share one device buffer. | `grep -n 'fn vulkan_fwd' crates/kiln-tensor/src/ops/matmul.rs` |
| **PR4 landed (THE pole)** | The shared autograd graph (`kiln_autograd::Tape` / `BackwardOp`) drives the Vulkan backward kernels. **`vk_autograd.rs` and all `vk_backward` call sites are retired.** Without this, the `vk_tensor.rs` collapse in §4 is impossible (the autograd half still has live consumers). | `grep -rn 'vk_autograd\|vk_backward' --include='*.rs' crates/ \| grep -v /target/` returns **only** doc-comment hits, no live `use`/call. |
| **PR5 landed** | `model_forward_kt` runs on `Device::Vulkan`; `vk_forward.rs` deleted. | `vk_forward.rs` does not exist OR is unreferenced (§3.3 grep). |
| **PR6 landed + ONE release elapsed** | SFT/GRPO/OPD route through shared `trainer.rs`/`opd.rs` on Vulkan; the cuda/metal cfg gates include `Device::Vulkan`; `KILN_VK_NATIVE_TRAINING` kept as opt-out for that release. | `grep -n 'Device::Vulkan' crates/kiln-train/src/trainer.rs` shows the widened gates. |
| **GPU soak signed off** | Human-gated end-to-end Vulkan SFT+GRPO+OPD on real Qwen3.5 matches the fork's saved adapters within tolerance (§7.2). | Soak report attached to the PR. |

**If any prereq is not provably true, STOP. PR7 is the wrong PR to land first.**

What PR7 unblocks: the **−10k LOC net** for the whole plan (plan §4) lands here.
After PR7, Vulkan looks exactly like Metal: **no `*_train.rs`, no `*_forward.rs`,
no parallel tensor/autograd stack** — just `VulkanStorage` + the shared substrate
+ the SPIR-V leaf kernels. Metal is the proof this end-state is reachable (it has
**no** `metal_train.rs` / `metal_forward.rs` / `metal_tensor.rs` — confirmed by
`find`; that absence is the template).

---

## 2. Data / ownership model (read before editing)

The collapse hinges on three invariants. The implementer must preserve all three.

### 2.1 Arc-of-`VulkanBuffer` lifetime

- `VkTensorInner.storage: Arc<VulkanBuffer>` (`vk_tensor.rs:72`). Cloning a `VkTensor`
  is an `Arc::clone` of the inner shell (`vk_tensor.rs:99`, `detach:143`), so device
  memory is refcount-driven. **Do not change this.** The leaf kernels rely on cheap
  `VkTensor` clone == buffer-Arc bump.
- PR2's `VulkanStorage::from_arc_buffer` (`vulkan_storage.rs:130`) already hands a
  shared `Arc<VulkanBuffer>` into the kt `Tensor`. PR3's zero-copy bridge makes the
  shared `Tensor` and a `VkTensor` view point at the **same** `Arc<VulkanBuffer>`.
  PR7 must **not** introduce any host bounce (D2H/H2D) on this seam — that would
  silently regress the only perf lever in the plan (plan §6).

### 2.2 `TensorId` preservation

- `crates/kiln-vulkan-kernel/src/vk_tensor.rs:25` does `pub use kiln_tensor_id::TensorId;`.
  This is the **dependency-free leaf** id type shared with `kiln_tensor`, `kiln_autograd`,
  and `kiln_optim`. **Keep this re-export** even in the collapsed shim — the SPIR-V
  ops still key buffers by id, and downstream crates import `TensorId` through it.
- The optimizer book is `TensorId`-keyed (`backend/vulkan.rs:2147 dispatch_adamw_step`).
  PR1 already routes Vulkan AdamW through the shared seam keyed by the same `TensorId`,
  so grad→param identity is already compatible. PR7 changes nothing here.

### 2.3 dtype / contiguity

- `VkDType { F32, Bf16 }` (`vk_tensor.rs:29`). The harmonized Vulkan **trained** path
  is **F32 on the hot path** (plan §5.3, §6). BF16 buffers are packed as u16 pairs
  viewed as `u32` words (`vk_tensor.rs:10–12`, `device_buffer_bytes:43`). The BF16 AdamW
  arm exists for parity (`backend/vulkan.rs:5711` test) but the trained path is F32 —
  **there is no BF16 training cliff** (plan §5.3). PR7 must not "simplify away" BF16
  packing: the inference decode path and the resident-activation registry still use it.
- Storage is **always C-contiguous** (`vk_tensor.rs:6–8`). Reshape is metadata-only;
  transpose/strided views are physical moves with their own dispatch. PR7 preserves this.
- `to_bytes()` (`vk_tensor.rs:419`) truncates to `num_elements * dtype.byte_size()` to
  strip the BF16 padding word. Any save path that survives PR7 must keep that truncation
  (the shared `save_peft` reads through kt `Tensor` → CPU contiguous, which already does).

---

## 3. File-by-file change spec

### 3.1 Delete `crates/kiln-train/src/vk_train.rs`

**Pre-delete grep (re-anchor + count):**
```
wc -l crates/kiln-train/src/vk_train.rs            # expect ~6061 (plan said 6109; re-count)
grep -rn 'vk_train::' --include='*.rs' crates/ | grep -v /target/ | grep -v 'src/vk_train.rs'
```
At HEAD the live external references are exactly:
- `crates/kiln-server/src/training_queue.rs:376` → `vk_native_sft_train`
- `crates/kiln-server/src/training_queue.rs:517` → `vk_native_grpo_train_jsonl`
- `crates/kiln-server/src/training_queue.rs:593` → `vk_native_grpo_train`
- `crates/kiln-server/src/training_queue.rs:812` → `vk_native_opd_train`
- `crates/kiln-train/src/opd.rs:771` → **doc comment only** (`pub(crate) struct TokenizedOpdPrompt`)
- `crates/kiln-train/tests/vk_train_smoke.rs` → the smoke suite (deleted in §3.2)

**Required-before-delete (these are PR6 work; PR7 only verifies they are gone):**
All four `training_queue.rs` call sites must already be replaced by the shared
`trainer::sft_train` / `trainer::grpo_train` / `opd::opd_train` routes (the fork's
`#[cfg(feature = "vulkan")] return kiln_train::vk_train::vk_native_*` blocks removed).
PR6 owns that rewrite; PR7 asserts it: the grep above must return **zero** live
`vk_train::` call sites (doc comments excepted) before deletion.

**Changes:**
1. `rm crates/kiln-train/src/vk_train.rs`.
2. `crates/kiln-train/src/lib.rs:64-65` — remove:
   ```rust
   #[cfg(feature = "vulkan")]
   pub mod vk_train;
   ```
3. `crates/kiln-train/src/opd.rs:771` — reword the doc comment on `TokenizedOpdPrompt`
   to drop the `crate::vk_train::vk_native_opd_train` reference. **Keep** `pub(crate)`
   visibility (the candle/kt `opd_train` still uses it). Suggested: "Shared OPD
   tokenization output used by `opd_train` (the VK-native path was removed in PR7)."

**Post-delete verify:** `grep -rn 'vk_train' --include='*.rs' crates/ | grep -v /target/`
returns nothing but unrelated string-literal log lines, if any. Then
`CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target cargo check -p kiln-train --features vulkan`.

### 3.2 Delete `crates/kiln-train/tests/vk_train_smoke.rs`

**Pre-delete grep:**
```
grep -n 'use kiln_train::vk_train' crates/kiln-train/tests/vk_train_smoke.rs
```
The file imports `kiln_train::vk_train::{...}` (line ~37, ~1785) and exercises
`save_vk_lora_adapter`, `allocate_adamw_state`, `vk_backward`, etc. — all fork APIs.
It cannot compile once `vk_train` is gone.

**Change:** `rm crates/kiln-train/tests/vk_train_smoke.rs`. Its coverage (forward,
backward FD, AdamW round-trip, adapter save) is replaced by:
- the kept SPIR-V parity tests under `crates/kiln-vulkan-kernel/tests/` (forward+backward FD),
- the PR2 storage round-trip tests in `crates/kiln-tensor` (host↔device),
- the PR1 AdamW seam tests (`backend/vulkan.rs:5628/5711/5804`),
- the shared `save_peft` adapter test (§3.4),
- and the new bounded reachability test in §6.

Do **not** port the fork-specific tests verbatim — they assert against the
fork's own `VkTensor` training tape, which no longer exists.

### 3.3 Confirm `crates/kiln-model/src/vk_forward.rs` is gone (PR5 deletes it)

**Grep:**
```
grep -rln 'vk_forward::' --include='*.rs' crates/ | grep -v /target/ | grep -v 'vk_forward.rs'
```
At HEAD the only importers are `vk_train.rs` and `vk_train_smoke.rs` — both deleted
above. So after §3.1/§3.2, `vk_forward.rs` has **zero importers** and PR5's deletion
(`rm crates/kiln-model/src/vk_forward.rs` + remove `pub mod vk_forward;` at
`crates/kiln-model/src/lib.rs:39`) is unblocked.

**PR7 responsibility:** if PR5 left `vk_forward.rs` in place (e.g. it was only
*unreferenced* but not removed), PR7 removes it and its `lib.rs:39` `pub mod`.
Before removing, re-run the grep and confirm zero importers. The 1762-LOC count
in the plan is the delete size; re-`wc -l` to confirm.

**Leaf-helper note:** the plan mentions "remaining `vk_forward.rs` leaf helpers not
reused." At HEAD every `pub fn` in `vk_forward.rs` (`vk_linear_with_lora:270`,
`vk_full_attention_*`, `vk_transformer_layer*`, `vk_model_forward_*`, `vk_step_backward:1365`,
…) is consumed **only** by `vk_train.rs`. None is reused by the harmonized path
(`backend/vulkan.rs` calls `vk_ops::*` directly, e.g. `vk_gdn_chunkwise_forward_no_grad`,
not `vk_forward::*`). So there are **no leaf helpers to migrate** — the whole file goes.
Confirm with: `grep -rn 'vk_forward::' crates/kiln-model/src/backend/vulkan.rs` → empty.

### 3.4 Remove `save_vk_lora_adapter`, use the shared safetensors path

**Anchor:**
```
grep -n 'fn save_vk_lora_adapter\|save_vk_lora_adapter(' crates/kiln-train/src/vk_train.rs
```
At HEAD: definition at `vk_train.rs:5990`; **10** call sites inside `vk_train.rs`
(`:4361, :4379, :4934, :4958, :5451, :5481, :5871, :5927`, the def `:5990`, and the
context-string `:6058`). Every call site is inside `vk_train.rs`, so deleting the file
(§3.1) removes them all — there is **no external caller** to rewire.

**The shared replacement is already live:** `LoraParams::save_peft`
(`crates/kiln-train/src/trainer.rs:1203`). It:
- writes `adapter_config.json` (r, lora_alpha, target_modules, peft_type) — which
  `save_vk_lora_adapter` **omitted** (`vk_train.rs:6059` only had a TODO comment),
- reads each `Parameter`'s primary kt tensor → CPU contiguous → `kiln_tensor::safetensors::save_cpu`,
- writes the adapter output receipt (`adapter_output::write_adapter_output_receipt`).

Because PR6 routes Vulkan SFT/GRPO/OPD through the shared trainer, the trained LoRA
weights are already `kiln_param::Parameter`s on `Device::Vulkan`, and `save_peft`'s
`to_device(Cpu).contiguous()` path (`trainer.rs:1233`) uses PR2's `vulkan_to_host_copy`
to read them back. **No Vulkan-specific save code is needed.** The safetensors key
layout is byte-identical (`base_model.model.model.layers.{i}.{sub}.{name}.lora_{A,B}.weight`).

**Acceptance:** the adapter produced by the harmonized Vulkan SFT path must be
byte-comparable to the fork's `save_vk_lora_adapter` output for the same trained
weights (modulo `save_peft` *adding* `adapter_config.json` + receipt, which is an
improvement). This comparison is a soak-gated step (§7.2), not a unit test.

### 3.5 Remove the `KILN_VK_NATIVE_*` opt-out family

This is the highest-care deletion: the flag is **not a single string**, and one of
its reads is a **weight-layout decision in `forward.rs`, not routing** (the trap).

**Anchor the full surface:**
```
grep -rn 'KILN_VK_NATIVE_TRAINING\|KILN_VK_NATIVE_GRPO\|KILN_VK_NATIVE_OPD' --include='*.rs' crates/ | grep -v /target/
```

**Routing reads (remove with PR6's route already done):**
- `crates/kiln-server/src/training_queue.rs` — `vk_native_sft_enabled:415`,
  `vk_native_grpo_enabled:432`, `vk_native_opd_enabled:446`, the routing blocks
  `:369–396` (SFT), `:515/593` (GRPO), `:810–823` (OPD), and the test
  `:2572 vk_native_grpo_defaults_to_vulkan_backend_and_honors_override`
  (+ `remove_var` at `:2576`).
- `crates/kiln-server/src/api/training.rs` — `vk_native_sft_enabled:168`,
  `vk_native_grpo_enabled:188`, and call sites `:378/432`.
- `crates/kiln-server/src/main.rs` — `vk_native_training_enabled:855`.

Once routing is unconditional-through-shared-trainer (PR6), all of these collapse:
delete the `vk_native_*_enabled` helpers and their callers; the request simply goes
to `trainer::sft_train` / `grpo_train` / `opd::opd_train` regardless of backend
(exactly the CPU/CUDA/Metal path).

**Capability string:** `crates/kiln-model/src/backend/vulkan.rs:1808`
```rust
native_training: "vk_native_sft_train/vk_native_grpo_train enabled by default on Vulkan",
```
Update to reflect the harmonized world, e.g.
`native_training: "kt-tape shared trainer (no native fork; #1082 PR7)"`.

**THE TRAP — `forward.rs` weight-layout reads (NOT routing):**
```
grep -n 'KILN_VK_NATIVE_TRAINING' crates/kiln-model/src/forward.rs
```
- `keep_projection_originals_enabled` (`forward.rs:5367`):
  `if crate::backend::vulkan_active() || env_enabled("KILN_VK_NATIVE_TRAINING") { return true; }`
  — the env is an **extra** trigger on top of `vulkan_active()`. Removing the env
  arm is safe **only because** `vulkan_active()` already covers every Vulkan process
  (confirmed by the comment at `:5364–5366`). Drop the `|| env_enabled(...)` and keep
  the `vulkan_active()` arm.
- `drop_projection_transposes_enabled` (`forward.rs:5393`):
  `env_enabled("KILN_VK_NATIVE_TRAINING") && !env_enabled("KILN_KEEP_PROJECTION_TRANSPOSES")`
  — here the env is the **ONLY** trigger. If you delete `KILN_VK_NATIVE_TRAINING`
  outright, this function becomes permanently `false` (transposes never dropped),
  which silently changes Vulkan weight layout and **wastes VRAM** keeping originals
  the harmonized trainer would otherwise drop.
  **Required action:** replace the env trigger with a non-env signal that the
  harmonized trainer is the sole consumer of these weights — e.g. a `GpuWeights`
  flag set when `model_forward_kt` is invoked for training on Vulkan, or thread the
  decision through the trainer config. **Do NOT** simply delete the env and leave
  the function dead-`false`. This needs the implementer to pick the replacement
  signal and is the single most likely place to introduce a silent regression.
  Tests to keep green: `forward.rs:26417–26447` exercise this gate via env today;
  rewrite them against the chosen non-env signal.
- The `KILN_VK_NATIVE_PHASE_TIMING` read (`forward.rs:23583`) is a **separate
  profiling flag**, NOT part of the opt-out family — **leave it alone.**

**Post-delete verify:**
```
grep -rn 'KILN_VK_NATIVE_TRAINING\|KILN_VK_NATIVE_GRPO\|KILN_VK_NATIVE_OPD' --include='*.rs' crates/ | grep -v /target/
```
must return empty (the `_PHASE_TIMING` flag is a different string and remains).
Then `cargo check -p kiln-server --features vulkan` and `-p kiln-model --features vulkan`.

---

## 4. Collapse `vk_tensor.rs` — the subtle one (read fully)

**The plan says "collapse `vk_tensor.rs` to the thin Arc-of-VulkanBuffer shim the
bridge needs." Taken literally that is NOT achievable in PR7, and an engineer who
tries it will break ~30 SPIR-V op modules. Here is the real state and the real target.**

### 4.1 Why a literal "Arc<VulkanBuffer> shim" does not work

The SPIR-V leaf kernels under `vk_ops/` consume and produce **`VkTensor`**, not bare
buffers. Reference density (HEAD):

```
grep -c 'VkTensor' crates/kiln-vulkan-kernel/src/vk_ops/*.rs
```
e.g. `gdn_chunkwise.rs` = 137 `VkTensor` refs, `gdn_chunk_bwd.rs` = 75, `attention.rs` = 54.
Even the **inference** path uses `VkTensor`: `backend/vulkan.rs:4568/4584/4603` call
`vk_gdn_chunkwise_forward_no_grad(q: &VkTensor, …) -> Result<VkTensor>`
(`gdn_chunkwise.rs:1131`). So `VkTensor` is the leaf layer's **tensor carrier type**
and must survive PR7. Reducing it to `Arc<VulkanBuffer>` would force rewriting every
`vk_ops` signature — that is a different, much larger PR and is out of scope.

### 4.2 The real target: drop the autograd half, keep the carrier half

`VkTensor` has two distinct responsibilities. PR4 already moves the autograd
responsibility onto `kiln_autograd::Tape`. PR7 removes the now-dead autograd half:

**KEEP (the leaf carrier the SPIR-V kernels need):**
- `struct VkTensor(Arc<VkTensorInner>)` and `VkTensorInner` fields: `storage`,
  `shape`, `dtype`, `device` (`vk_tensor.rs:71–83` minus the autograd fields below).
- `VkDType` + `byte_size` + `device_buffer_bytes` (`:29–51`).
- `pub use kiln_tensor_id::TensorId;` (`:25`) — still needed (§2.2).
- Accessors: `shape/dtype/device/buffer/num_elements/byte_size` (`:102–140`).
- Constructors used by the leaf kernels + the zero-copy bridge: `from_buffer:179`,
  `alloc_uninit:219`, `from_f32_slice:239`, `from_f32_slice_as_bf16:279`,
  `from_bytes:377`, `to_bytes:419`, `to_vec_f32:441`.

**REMOVE (the training tape — its only consumers were `vk_autograd` + `vk_train`,
both deleted by PR4/PR7):**
- The `VkBackwardOp` trait (`:61–68`) **iff** PR4 retired every impl (see §4.3).
- `next_op_id`/`NEXT_OP_ID` (`:53–58`) — autograd topo ordering only.
- `VkTensorInner` fields `grad_fn`, `requires_grad`, `op_id`, `param_id` (`:76–82`).
- Methods `op_id:118`, `requires_grad:122`, `param_id:126`, `grad_fn:130`, `detach:143`,
  `from_op:158`, `parameter:199`, `parameter_from_f32_slice:332`,
  `parameter_from_f32_slice_as_bf16:351`, `fresh_param_id:323`.
- Update the `Debug` impl (`:85–95`) to drop the `requires_grad/op_id/is_param/grad_fn` fields.

Net: `vk_tensor.rs` drops from 568 LOC to a buffer+shape+dtype carrier (~250–300 LOC).
That **is** the "thin shim" — thin meaning "no autograd," not "just a buffer."

### 4.3 Hard dependency: §4.2 is blocked on PR4 finishing the op rewrite

The `from_op`+`grad_fn` machinery is referenced by **every** `vk_ops` forward that
builds a backward node (`grep -rln 'VkBackwardOp\|from_op\|grad_fn' crates/kiln-vulkan-kernel/src/vk_ops/`
→ ~26 modules) AND by the SPIR-V **parity tests**
(`crates/kiln-vulkan-kernel/tests/vk_*_parity.rs` drive backward via
`vk_autograd::vk_backward`, e.g. `vk_matmul_parity.rs:155`). **PR4 must have already**:
1. rewired each `vk_ops` forward so its `_no_grad` buffer path is the only path the
   harmonized inference/training uses, and recorded backward onto `kiln_autograd::Tape`
   instead of attaching a `VkBackwardOp` grad_fn, and
2. re-pointed the `vk_*_parity.rs` FD tests to drive the SPIR-V backward **kernels**
   directly (buffer-in/buffer-out), not through `vk_backward`.

**PR7 verification before removing the `VkBackwardOp` trait:**
```
grep -rln 'VkBackwardOp\|vk_backward\|VkGradStore\|from_op\|\.grad_fn\|requires_grad' \
  crates/kiln-vulkan-kernel/src crates/kiln-vulkan-kernel/tests | grep -v /target/
```
If this returns **any** live (non-doc) hit, **STOP** — PR4 is not done and the
collapse is premature. Remove only the symbols that grep proves dead. If PR4 left
the parity tests on `vk_backward`, the honest move is: keep the minimal
`VkBackwardOp`+`vk_autograd` test-only harness behind `#[cfg(test)]`/a test module
and file a follow-up, rather than break SPIR-V backward coverage. **Document whichever
you do in the PR description.**

### 4.4 lib.rs exports to prune (`crates/kiln-vulkan-kernel/src/lib.rs`)
- `:24 pub mod vk_autograd;` — remove iff §4.3 grep is clean (PR4 may already have).
- `:162 pub use vk_autograd::{VkGradStore, vk_backward};` — remove with it.
- Keep `:28 pub mod vk_tensor;` and `:25 pub mod vk_ops;`.

---

## 5. Stale-doc fixes (verified against HEAD)

The plan listed two stale sites; verification found **four** copies of the same false
claim plus the FLCE comment. Fix all of them; do not fabricate the FLCE rewording —
see §5.4.

### 5.1 The false claim (THREE identical copies): "Metal OPD backward bail!s"

**Ground truth at HEAD:** `CudaOpdTopKReverseKlPhaseBBackward::apply`
(`crates/kiln-opd-loss-kernel/src/kt_tape.rs:166–245`) does **NOT** bail on Metal.
Its `#[cfg(feature = "cuda")]` arm (`:197–220`) routes CUDA through the fused FFI
kernel; the fall-through arm (`:222–237`) routes **CPU and Metal** through the
**device-agnostic** `opd_top_k_reverse_kl_phase_b_bwd_composite_kt` (pure `kiln_tensor`
ops, FD-validated). Commit `948bbe0f` ("OPD training on Apple Metal") landed this.
So every comment claiming the Metal OPD backward `bail!`s is **false**.

Fix all three:
1. **`crates/kiln-train/src/opd_candle_shim.rs:88–92`** — currently:
   > "On Metal the FORWARD + loss record onto the tape; the recorded backward
   > (`CudaOpdTopKReverseKlPhaseBBackward::apply`) is CUDA-FFI-only and `bail!`s
   > on Metal — the OPD Metal backward is a documented follow-up."
   Replace with: on Metal/CPU the recorded backward runs the device-agnostic
   kt-composite `opd_top_k_reverse_kl_phase_b_bwd_composite_kt`; only CUDA uses the
   fused FFI kernel. No follow-up pending.
2. **`crates/kiln-opd-loss-kernel/src/kt_tape.rs:92–98`** (the `envelope_ok` comment) —
   same correction: drop "still CUDA-FFI-only — on Metal it `bail!`s …documented
   follow-up." State that Metal/CPU use the composite backward.
3. **`crates/kiln-train/src/opd.rs:2943–2947`** (doc on
   `opd_step_forward_backward_tape_authoritative`) — same correction.

(Re-anchor each with `grep -rn 'CudaOpdTopKReverseKlPhaseBBackward::apply' --include='*.rs' crates/ | grep -v /target/`.)

### 5.2 (Optional, in-scope) add Vulkan to the OPD backward envelope note

`envelope_ok` (`kt_tape.rs:99`) gates `Cuda(_) | Metal(_)`. If PR6 added a Vulkan arm
to `opd_step_forward_backward_tape_authoritative`'s cfg gate, the envelope and its
comment should mention Vulkan too. **Verify the gate first** (`grep -n 'Device::Vulkan' crates/kiln-opd-loss-kernel/src/kt_tape.rs`);
only widen the comment to match the code that actually exists.

### 5.3 opd.rs cross-reference cleanup

`opd.rs:771` doc comment names the deleted `vk_native_opd_train` — already covered in §3.1.

### 5.4 FLCE `envelope_ok` comment (`crates/kiln-flce-kernel/src/kt_tape.rs:84–110`) — VERIFY, don't assume

Plan §8 flags this as stale. **Honest finding:** at HEAD the FLCE `envelope_ok`
(`kt_tape.rs:91–110`) genuinely checks `Cuda(_)` **only** (`:90`), and the FLCE backward
`CudaFlcePhaseBBackward::apply` (`:169–202`) calls
`fused_linear_cross_entropy_phase_b_backward_kt` (`kt_api.rs:474`) with **no Metal/CPU
composite fork** visible — unlike OPD. So the comment "CUDA + dtype in {F32, BF16} + …"
may be **accurate**, not stale.

**Required of the implementer:** before editing, verify whether FLCE backward is
device-agnostic at the PR7 landing commit:
```
grep -n 'KtDevice::Metal\|Metal(_)\|Cpu\|composite' crates/kiln-flce-kernel/src/kt_tape.rs crates/kiln-flce-kernel/src/kt_api.rs
```
- If FLCE backward is now device-agnostic (a Metal/CPU path landed after this spec was
  written): widen `envelope_ok` + comment to match OPD (`Cuda(_) | Metal(_) | Vulkan`).
- If FLCE is still CUDA-only: the comment is **correct** — leave it, and instead fix
  only the wording the plan actually objected to (e.g. a reference to a renamed
  `shim_envelope_ok`/`flce_candle_shim` — note `crates/kiln-train/src/flce_candle_shim.rs`
  **does not exist** at HEAD, so the comment's `kiln_train::flce_candle_shim::shim_envelope_ok`
  cross-reference at `:85` IS stale and should be corrected to the real predicate location).

**Do not fabricate a device-coverage claim for FLCE you cannot confirm by reading the code.**

---

## 6. Bounded test plan (runs without long training)

All tests below are **host-safe**: named, finite, no multi-step training, no
long-running binaries. Run individually with
`CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target cargo test -p <crate> --features vulkan <test_name> -- --nocapture`.

### 6.1 Compile gates (the primary PR7 acceptance)
- `cargo check -p kiln-train --features vulkan` — proves `vk_train` deletion + `lib.rs`
  edit + `opd.rs` doc reword compile.
- `cargo check -p kiln-model --features vulkan` — proves `vk_forward` removal +
  `forward.rs` flag rework compile.
- `cargo check -p kiln-server --features vulkan` — proves the `KILN_VK_NATIVE_*`
  routing removal compiles.
- `cargo check -p kiln-vulkan-kernel --features vulkan` — proves the `vk_tensor.rs`
  collapse + `vk_autograd` removal compile.
- `cargo check -p kiln-opd-loss-kernel`, `-p kiln-flce-kernel` — proves the doc edits
  compile (doctests if any).

### 6.2 Kept SPIR-V parity tests (NO new kernels — these must still pass unchanged)
These validate the kept leaf kernels. After PR4 re-points them off `vk_backward`
(§4.3), they exercise the SPIR-V forward+backward **kernels** directly. PR7 must not
regress them:
- `crates/kiln-vulkan-kernel/tests/vk_opd_parity.rs` — F32 abs ≤ **1e-4**, bf16w ≤ 5e-2
  (`vk_opd_parity.rs:6–7,210`). This is the Vulkan mirror of the Metal OPD gate.
- `vk_matmul_parity.rs`, `vk_rmsnorm_parity.rs`, `vk_softmax_parity.rs`,
  `vk_attention_parity.rs`, `vk_flce_parity.rs`, `vk_gdn_foundation_parity.rs`,
  `vk_gdn_chunkwise_parity.rs`, `vk_tensor_parity.rs`.
- **Skip-cleanly when no Vulkan GPU** (existing pattern: `VulkanDevice::probe()` →
  early `return`).

### 6.3 Cross-engine OPD parity (the Metal-OPD-gate mirror)
- `crates/kiln-train/tests/vk_cuda_opd_parity.rs` — already enforces the §9.2 contract:
  same `(hidden, teacher_topk_logprobs, top_k_indices)` → KL within tolerance across
  CUDA/Vulkan. Runs only under `#[cfg(all(feature = "cuda", feature = "vulkan"))]` with
  both GPUs present; skips with a printed reason otherwise (`vk_cuda_opd_parity.rs:16`).
  PR7 keeps this green. **Acceptance threshold: F32 max_abs_err ≤ 1e-5** (the §9.2
  grand-plan contract; the file uses 1e-4 abs / 1e-3 rel to absorb fmadd/associativity
  drift — keep that practical bound, and document any tightening to 1e-5 as the target).

### 6.4 Storage round-trip (PR2 tests — must still pass)
- The PR2 host→device→host round-trip tests in `crates/kiln-tensor` (the flipped
  negatives at `tensor.rs:1491/1499`) — confirm `vulkan_to_host_copy` still bit-exact,
  since `save_peft` reads adapters through it.

### 6.5 NEW bounded reachability smoke (add in PR7)
Add **one** cheap, single-step, GPU-gated test asserting the harmonized save path
works end-to-end without the fork. Suggested home:
`crates/kiln-train/tests/vk_harmonized_save_smoke.rs` (scaffold provided —
`docs/vk-harmonization/PR7-test-scaffold.rs`, **WIP / `#[ignore]`**). It:
1. probes Vulkan; skips if absent,
2. builds two tiny LoRA `Parameter`s on `Device::Vulkan` (via PR2 `host_to_vulkan_copy`),
3. wraps them in a one-layer `LoraParams`,
4. calls `LoraParams::save_peft` to a tmp dir,
5. asserts `adapter_model.safetensors` + `adapter_config.json` exist and the
   `lora_A/lora_B` tensors read back bit-exact vs the host source.
**No training loop. No multi-step. Single op + one save.** Mark `#[ignore]` until the
implementer wires the real `LoraParams` constructor signature (it WILL drift).

### 6.6 Flag-removal unit tests
- Rewrite `forward.rs:26417–26447` (the transpose-drop gate tests) against the
  **non-env** signal chosen in §3.5, asserting keep-originals stays on for Vulkan and
  drop-transposes engages only when the trainer is the sole consumer.
- Delete `training_queue.rs:2572 vk_native_grpo_defaults_to_vulkan_backend_and_honors_override`
  (its subject — env-default routing — is gone).

---

## 7. Human-gated GPU soak (NOT run autonomously — describe, do not execute)

> The dev host has **hard-crashed twice on long training runs**. The steps below are
> for a human operator on a soak box, explicitly separated from §6. **An autonomous
> agent must NOT run these.**

### 7.1 End-to-end harmonized training (multi-step)
On a Vulkan GPU (Strix Halo baseline in `bench-results/vulkan-strix-halo-baseline.md`):
1. Run a bounded real-model SFT (e.g. 20 steps, tiny dataset) through the **shared**
   `trainer::sft_train` on `Device::Vulkan` (post-PR6, no `KILN_VK_NATIVE_TRAINING`).
2. Repeat for GRPO (`grpo_train`) and OPD (`opd_train`).
3. Confirm loss decreases and no NaN/Inf; capture tok/s vs the fork baseline (parity
   expected — same SPIR-V kernels over the same device buffers, plan §6).

### 7.2 Adapter-equivalence gate (the real correctness check)
1. Train the **same** seed/data/steps on the **fork** (last commit before PR7) and on
   the **harmonized** path.
2. Compare the two `adapter_model.safetensors`: per-tensor `max_abs_err` on
   `lora_A.weight`/`lora_B.weight`. **Accept F32 max_abs_err ≤ 1e-5** (mirrors the
   Metal OPD gate). Differences above that, but explained purely by op associativity,
   require justification; anything structural is a fail.
3. Confirm the harmonized output **adds** `adapter_config.json` + the output receipt
   that `save_vk_lora_adapter` omitted, and that both load via `lora_loader`.

### 7.3 Inference-after-train smoke
Load the harmonized adapter into `kiln serve` on Vulkan; run a few chat completions;
confirm coherent output (no shape/transpose regression from the §3.5 weight-layout
flag rework — the exact failure mode the `forward.rs:5381–5394` comment warns about).

---

## 8. Parity acceptance thresholds (single source of truth)

| Path | dtype | Threshold | Source |
| ---- | ----- | --------- | ------ |
| OPD top-K reverse-KL fwd/bwd (Vulkan vs oracle) | F32 | max_abs ≤ **1e-4** abs / 1e-4 rel | `vk_opd_parity.rs:6–7,210` |
| same | bf16w | ≤ 5e-2 | `vk_opd_parity.rs:210` |
| Cross-engine OPD (CUDA vs Vulkan) | F32 | **≤ 1e-5** contract (1e-4 abs / 1e-3 rel practical) | `vk_cuda_opd_parity.rs:4–10`, plan §9.2 |
| Adapter equivalence (fork vs harmonized) | F32 | max_abs_err ≤ **1e-5** (Metal OPD gate mirror) | §7.2 |
| Storage round-trip (host↔device) | F32 | **bit-exact** | PR2 round-trip tests |
| Storage round-trip | BF16 | host `bf16::from_f32` rounding only (≤ 1/256) | `vk_tensor.rs:506` test |

The headline gate the task asks for — **"mirror the Metal OPD gate: max_abs_err ~1e-5
F32"** — is row 4 (adapter equivalence) and the 1e-5 contract target in row 3.

---

## 9. Open risks + de-risking

1. **`vk_tensor.rs` collapse is gated on PR4 being truly finished (§4.3).** If PR4 left
   the `vk_*_parity.rs` tests driving `vk_backward`, removing `VkBackwardOp` breaks
   SPIR-V backward coverage. *De-risk:* the §4.3 grep is the go/no-go; if not clean,
   keep a `#[cfg(test)]` minimal autograd harness and file a follow-up rather than
   delete blindly. Honest partial > broken.

2. **The `forward.rs` weight-layout flag (§3.5 trap).** Deleting `KILN_VK_NATIVE_TRAINING`
   silently neuters `drop_projection_transposes_enabled` (always-false), changing
   Vulkan weight layout / VRAM. *De-risk:* replace with an explicit non-env trainer
   signal **before** removing the env; keep `forward.rs:26417–26447` green against it;
   §7.3 inference smoke catches a regression.

3. **FLCE comment may not actually be stale (§5.4).** Editing it to claim Metal coverage
   that does not exist would *create* a false doc. *De-risk:* verify FLCE backward
   device coverage at the landing commit; only the `flce_candle_shim` cross-reference
   (a path that does not exist at HEAD) is provably stale.

4. **Adapter-equivalence drift (§7.2).** Op associativity differences (fused FFI vs
   kt-composite, fmadd ordering) can push `max_abs_err` above 1e-5 even when both are
   "correct." *De-risk:* compare against the same-seed fork run, attribute any delta to
   a named associativity source, and gate on the 1e-4 practical bound if 1e-5 proves
   unreachable — document the choice in the PR.

5. **Plan LOC counts drift (6109 vs 6061, etc.).** *De-risk:* every delete step in §3
   carries a `wc -l` / `grep -c` re-anchor; trust the grep, not the doc number.

6. **Hidden external consumer of a deleted symbol.** A crate outside the greps above
   (examples, benches, server tests) could import `vk_train`/`save_vk_lora_adapter`.
   *De-risk:* run the §3 post-delete greps across the **whole** workspace
   (`grep -rn … crates/ examples/ 2>/dev/null | grep -v /target/`) and a full
   `cargo check --workspace --features vulkan` before merge (compile-only, host-safe).

---

## 10. Deletion checklist (ordered; each line gated by its grep)

```
# 0. PREREQS — abort if any prints live hits / fails
grep -rn 'vk_train::'  --include='*.rs' crates/ | grep -v /target/ | grep -v 'src/vk_train.rs' | grep -v 'tests/vk_train_smoke'   # only opd.rs:771 doc-comment allowed
grep -rn 'vk_autograd\|vk_backward' --include='*.rs' crates/ | grep -v /target/                                                   # PR4 must be done (doc-only OK)
grep -n  'Device::Vulkan' crates/kiln-train/src/trainer.rs                                                                        # PR6 widened gates present

# 1. vk_train.rs
wc -l crates/kiln-train/src/vk_train.rs
rm    crates/kiln-train/src/vk_train.rs
#     edit crates/kiln-train/src/lib.rs:64-65  (remove cfg+pub mod vk_train)
#     edit crates/kiln-train/src/opd.rs:771    (reword doc comment)

# 2. smoke test
rm    crates/kiln-train/tests/vk_train_smoke.rs

# 3. vk_forward.rs (if PR5 left it)
grep -rln 'vk_forward::' --include='*.rs' crates/ | grep -v /target/ | grep -v 'vk_forward.rs'   # must be empty
rm    crates/kiln-model/src/vk_forward.rs        # if present
#     edit crates/kiln-model/src/lib.rs:39       (remove pub mod vk_forward)

# 4. save_vk_lora_adapter — removed by deleting vk_train.rs; confirm no external caller
grep -rn 'save_vk_lora_adapter' --include='*.rs' crates/ | grep -v /target/   # empty after step 1+2

# 5. KILN_VK_NATIVE_* opt-out family
grep -rn 'KILN_VK_NATIVE_TRAINING\|KILN_VK_NATIVE_GRPO\|KILN_VK_NATIVE_OPD' --include='*.rs' crates/ | grep -v /target/
#     edit training_queue.rs / api/training.rs / main.rs (remove routing + helpers + tests)
#     edit backend/vulkan.rs:1808 (capability string)
#     edit forward.rs:5367 (drop || env_enabled arm) and forward.rs:5393 (replace env trigger — §3.5 TRAP)
#     rewrite forward.rs:26417-26447 tests against the non-env signal
#     DO NOT touch forward.rs:23583 KILN_VK_NATIVE_PHASE_TIMING

# 6. vk_tensor.rs collapse (only after §4.3 grep is clean)
grep -rln 'VkBackwardOp\|from_op\|\.grad_fn\|requires_grad\|VkGradStore' crates/kiln-vulkan-kernel/src crates/kiln-vulkan-kernel/tests | grep -v /target/
#     edit vk_tensor.rs (remove autograd half — §4.2), keep carrier half + TensorId re-export
#     edit lib.rs:24,162 (remove vk_autograd mod + re-export) — if PR4 left them
#     rm crates/kiln-vulkan-kernel/src/vk_autograd.rs — if PR4 left it

# 7. stale docs
#     opd_candle_shim.rs:88-92, opd-loss-kernel/kt_tape.rs:92-98, opd.rs:2943-2947  (§5.1)
#     flce-kernel/kt_tape.rs:84-110  (§5.4 — VERIFY FIRST)

# 8. compile gates (host-safe)
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target cargo check --workspace --features vulkan
```

---

## 11. Reference index (read from worktree HEAD; will drift — re-grep)

| Item | Location (HEAD) |
| ---- | --------------- |
| `vk_train.rs` (delete) | `crates/kiln-train/src/vk_train.rs` (~6061 LOC) |
| `vk_train` module decl | `crates/kiln-train/src/lib.rs:64-65` |
| `vk_train_smoke.rs` (delete) | `crates/kiln-train/tests/vk_train_smoke.rs` (3592 LOC) |
| `save_vk_lora_adapter` def | `crates/kiln-train/src/vk_train.rs:5990` (10 call sites, all in-file) |
| shared save path (replacement) | `crates/kiln-train/src/trainer.rs:1203 LoraParams::save_peft` |
| `vk_forward.rs` (delete, PR5) | `crates/kiln-model/src/vk_forward.rs` (1762 LOC), decl `lib.rs:39` |
| `vk_tensor.rs` (collapse) | `crates/kiln-vulkan-kernel/src/vk_tensor.rs` (568 LOC) |
| `VkTensorInner` autograd fields | `vk_tensor.rs:76-82` (grad_fn, requires_grad, op_id, param_id) |
| `VkBackwardOp` trait | `vk_tensor.rs:61-68` |
| `TensorId` re-export (KEEP) | `vk_tensor.rs:25` |
| `vk_autograd` mod + re-export | `crates/kiln-vulkan-kernel/src/lib.rs:24,162` |
| `vk_ops` SPIR-V leaf kernels (KEEP) | `crates/kiln-vulkan-kernel/src/vk_ops/` (32 modules) |
| inference uses `VkTensor` carrier | `backend/vulkan.rs:4568/4584/4603` → `gdn_chunkwise.rs:1131` |
| native-training capability string | `crates/kiln-model/src/backend/vulkan.rs:1808` |
| `KILN_VK_NATIVE_*` routing | `training_queue.rs:374/415/432/446/517/593/812`, `api/training.rs:168/188`, `main.rs:855` |
| `KILN_VK_NATIVE_TRAINING` weight-layout TRAP | `forward.rs:5367` (keep-originals), `5393` (drop-transposes), tests `26417-26447` |
| `KILN_VK_NATIVE_PHASE_TIMING` (LEAVE) | `forward.rs:23583` |
| Metal OPD backward (NOT bail) | `crates/kiln-opd-loss-kernel/src/kt_tape.rs:166-245` (composite arm `:222-237`) |
| stale "bail!s on Metal" (3 copies) | `opd_candle_shim.rs:88-92`, `opd-loss-kernel/kt_tape.rs:92-98`, `opd.rs:2943-2947` |
| FLCE `envelope_ok` (VERIFY) | `crates/kiln-flce-kernel/src/kt_tape.rs:84-110`; missing `flce_candle_shim.rs` makes `:85` cross-ref stale |
| OPD parity thresholds | `crates/kiln-vulkan-kernel/tests/vk_opd_parity.rs:6-7,210` |
| cross-engine OPD parity | `crates/kiln-train/tests/vk_cuda_opd_parity.rs:4-16` |
| PR2 storage round-trip | `crates/kiln-tensor/src/vulkan_storage.rs:277,372,130`; tests `tensor.rs:1491,1499` |
