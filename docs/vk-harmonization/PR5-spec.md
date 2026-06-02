# PR5 — Forward harmonization: `model_forward_kt` + `try_tape_*_kt` recorders on `Device::Vulkan`

Issue: #1082
Branch: `feat/vk-tape-harmonization`
Status: **SPEC ONLY — not implemented.** PR1/PR2 are code in this branch; PR3–PR7 are specs.
Authoritative parent plan: `docs/vulkan-train-harmonization-plan.md` (§4 "PR5", §5, §6, §7).

> ⚠ **ADVERSARIAL-REVIEW CORRECTION (load-bearing — read before implementing).**
> The §0 thesis — *"backward just works once the forward records, because the
> `*Backward` composites bottom out in device-agnostic `kiln_tensor::ops::*`"* — is
> **materially incomplete for Vulkan.** Several ops those composites call have **no
> `vulkan_fwd` AND no Vulkan host-fallback**, so they **hard-error** at
> `tape.backward()` time:
> - `reduce.rs::sum_axis` — CUDA-only (`Ok(None)` on Vulkan)
> - `scalar.rs::{mul,add,sub}_scalar` — cpu+cuda only
> - `ops/log_softmax.rs::log_softmax_last_dim` — explicit CUDA + Metal fast paths, **no Vulkan arm**
>
> The very example cited as proof, `RmsNormKtBackward::apply` (`tape_forward.rs:257`),
> calls `sum_axis`, `mul_scalar`, `add_scalar`, `sub`, `mul`. Critically,
> `dispatch1`/`dispatch2` (`device_op.rs:170-192`) host-fallback **only for Metal**;
> for Vulkan an `Ok(None)` op falls to `cpu_fwd` on Vulkan storage and **hard-errors**.
> Metal's backward "just worked" *partly because of* that Metal-only host fallback
> Vulkan deliberately lacks.
>
> **This is an unscoped op-port sub-project that NO PR currently owns.** Resolve it
> one of two ways (see `REVIEW-NOTES.md` §"[major] op-coverage"): **(B, recommended
> first)** add the Metal-style Vulkan host-fallback in `dispatch1/dispatch2` so
> Ok(None) ops bounce D2H→`cpu_fwd`→H2D (correct-but-slow, makes the thesis true
> immediately, mirrors how Metal shipped), then **(A)** port the hot ops
> (`sum_axis`, scalar, `log_softmax_last_dim`, possibly `scatter_add`/`broadcast`)
> to `vulkan_fwd` for perf. Assign ownership: fold into PR3's scope or insert a
> dedicated **PR3.5 — Vulkan backward op-coverage** between PR3 and PR4.

> All file:line anchors below were grepped against the worktree at branch HEAD
> (`b94feeac`). **Line numbers drift** — every anchor is paired with a stable
> grep token. Re-confirm with the grep before editing, then edit by token.

---

## 0. One-paragraph thesis

`model_forward_kt` (`crates/kiln-model/src/forward.rs:22737`) is already
device-agnostic: it drives the forward through `backend`, `kiln_tensor::ops::*`,
and the `crate::tape_forward::try_tape_*_kt` recorders, all of which dispatch by
the *tensor's own* `Device`. After PR2 (Vulkan-resident `kiln_tensor::Tensor`)
and PR3 (`MatmulOp::vulkan_fwd` + the zero-copy `VulkanStorage`↔`VkTensor`
bridge), the shared forward *can physically run* on `Device::Vulkan(_)`. Two
things still block a **trainable** Vulkan forward, and both live in
`crates/kiln-model/src/tape_forward.rs`:

1. The whole `tape_forward` module is `#[cfg(any(feature = "cuda", feature =
   "metal"))]` (`lib.rs:33`), so it is **absent** from a `--features vulkan`
   build. `model_forward_kt`'s calls into it would not compile.
2. Even with the module present, **16 of its recorders hard-gate the device to
   `Device::Cuda(_) | Device::Metal(_)`** and return `Ok(None)` on Vulkan. A
   recorder that returns `None` records **nothing** on the `Tape`: the forward
   value is still produced (via the non-tape fallthrough), but the backward node
   is silently dropped, so GDN / flash / attention / rms-norm / lora / CE
   gradients **never reach the optimizer**. This is the "silently will not
   record" failure the task calls out.

PR5 (a) widens the module gate and the 16 device gates to admit
`Device::Vulkan(_)`, (b) proves per-op forward parity of the harmonized path
against the legacy `vk_forward.rs` *before* deleting it, and (c) retires
`vk_forward.rs` (1762 LOC: `vk_model_forward_loss`, `vk_linear_with_lora`, the
`GpuWeights`→`VkModelWeights` bridge, and the GRPO reference-logprob reimpl).

**Why the backward "just works" once the forward records.** Every backward op
recorded by these recorders (`RmsNormKtBackward`, `FlashAttnBackward`,
`SdpaBackward`, `GdnRecurrentBackward`, `GdnL2NormScaleBackward`,
`GdnGatedRmsNormBackward`, `GqaExpandBackward`, `CrossEntropyFromLogitsKtBackward`,
`LoraDeltaAddBackward`, `CausalConv1dBackward`, `CastBackward`, `NarrowBackward`)
has an `apply()` that is **device-agnostic**: it bottoms out in
`kiln_tensor::ops::*` (and `crate::forward` / `crate::backend` composites) that
dispatch by the grad tensor's own device. Verified concretely for
`RmsNormKtBackward::apply` (`tape_forward.rs:253`): every line is
`kiln_tensor::ops::{cast,mul,sum_axis,sqrt,reciprocal,...}` / `broadcast_mul` /
`reshape` — **no `Device::Cuda` / `Device::Metal` branch.** So the *only* thing
keeping the Vulkan backward from running is the forward-side device gate
refusing to record. Mirror of the Metal lane exactly:
`948bbe0f`/`b9aa7219` widened the same gates `cuda → any(cuda, metal)` and the
Metal SFT/GRPO/OPD backward then ran with no per-op backward port (the kt
composites carried it). **The hard PR5 dependency is therefore PR3**: those
device-agnostic backward composites need `vulkan_fwd` for each kt op they call.
PR5 does not port any backward kernel; it unblocks the recorder and relies on
PR3's op coverage.

---

## 1. Prerequisite PRs and what PR5 unblocks

| Needs (must land first) | Why |
| --- | --- |
| **PR2** — Vulkan-resident `Tensor` (host↔device copy pair, un-NYI constructors). | `model_forward_kt` builds intermediate `Tensor`s on the input's device; without PR2 they `Err` NYI on Vulkan. Landed in this branch (`b94feeac`). |
| **PR3** — `MatmulOp::vulkan_fwd` + zero-copy `VulkanStorage`↔`VkTensor` bridge. | The forward's hottest op and every backward composite call `kiln_tensor::ops::matmul`; the backward composites (`RmsNormKtBackward` etc.) call `mul/sum_axis/sqrt/cast/...`. Each needs `vulkan_fwd`. Without zero-copy each op bounces D2H/H2D and the harmonized path regresses badly vs. the fork (parent plan §6). |
| **PR4** *(soft)* — `VkBwdAdapter` for any op family **not** covered by a device-agnostic kt composite. | The recorders PR5 widens all record device-agnostic kt backwards, so PR5 does **not** strictly need PR4. PR4 matters only if a future op family lacks a kt-composite backward. Note this in the PR5 PR description; do not block on it. |

**PR5 unblocks:**

- **PR6** (orchestration flip): once `model_forward_kt` + recorders run + record
  on Vulkan, PR6 can widen the tape-authoritative / grad-checkpoint gates
  (`trainer.rs:2478/4869/5001/5990/7217/...`, `opd.rs:2724/2948`) to include
  `Device::Vulkan` and route Vulkan SFT/GRPO/OPD through the shared `trainer.rs`
  / `opd.rs`. PR6 is the LOC win; it is meaningless until PR5's forward records.
- **PR7** (delete the fork): PR5 deletes `vk_forward.rs`; PR7 then deletes
  `vk_train.rs` (its sole consumer — see §4.4) and the `KILN_VK_NATIVE_TRAINING`
  gate.

---

## 2. Exact file targets (grep-confirmed at HEAD `b94feeac`)

### 2.1 `crates/kiln-model/src/lib.rs` — module gate (THE keystone edit)

- **Anchor:** `lib.rs:33–34`
  ```rust
  #[cfg(any(feature = "cuda", feature = "metal"))]
  pub mod tape_forward;
  ```
  grep token: `pub mod tape_forward;`
- **Change:** widen the cfg to include vulkan:
  ```rust
  #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
  pub mod tape_forward;
  ```
- **Consequence to verify (do not skip):** `tape_forward.rs:103` has the *same*
  module-inner gate `#![cfg(any(feature = "cuda", feature = "metal"))]`. Widen
  it identically (grep token: `#![cfg(any(feature = "cuda", feature = "metal"))]`).
  Then `cargo check -p kiln-model --features vulkan` will surface every symbol
  `tape_forward` imports that is itself cuda/metal-gated. Expect at least:
  - `use kiln_autograd::{... MatmulBackward, ... SiluBackward, ...}`
    (`tape_forward.rs:106`) — `kiln_autograd` backward structs are
    device-agnostic; confirm they are not feature-gated out under vulkan.
  - `use crate::forward::{gdn_*_backward_no_grad, gdn_recurrent_forward_from_parts,
    sdpa_fallback_backward_no_grad, GDN_CHUNK_SIZE}` (`tape_forward.rs:113–117`)
    — these `crate::forward` composites must be reachable under `--features
    vulkan`. If any is cuda/metal-gated, widen its gate too (same one-line
    pattern). **This is the most likely compile-break of the whole PR; budget
    for a short cascade of gate widenings, each `cuda,metal → cuda,metal,vulkan`.**
  - `kiln_flash_attn::flash_attn_fwd_kt` (`tape_forward.rs:1527`) is reached only
    inside `#[cfg(feature = "cuda")]` (see §2.2 flash row) — it does **not** need
    a vulkan path; the Vulkan attention backward records via the SDPA fallback.

### 2.2 `crates/kiln-model/src/tape_forward.rs` — 16 device-gated recorders

**Mechanical rule:** every occurrence of
```rust
kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Metal(_)
```
inside a recorder's precondition becomes
```rust
kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Metal(_) | kiln_tensor::Device::Vulkan(_)
```
grep token to enumerate every site:
`grep -n "Device::Cuda(_) | kiln_tensor::Device::Metal(_)" crates/kiln-model/src/tape_forward.rs`
(31 in-condition occurrences across the 16 functions; HEAD line list below — **re-grep, do not trust the numbers**). Use `replace_all` on the exact token string; it is identical at every site, so a single global replace is correct **except** the flash-attn site (handle per §2.2 flash row).

| # | Recorder fn | grep token (fn name) | HEAD gate line(s) | Note |
|---|-------------|----------------------|-------------------|------|
| 1 | `try_tape_rms_norm_kt` | `fn try_tape_rms_norm_kt` | 339,340 | QK-norm + pre-attn norm; closes the LoRA-grad-empty seam (§ doc-comment at `:315`). |
| 2 | `try_tape_transpose_kt` | `fn try_tape_transpose_kt` | 508 | head-FIRST→head-LAST chaining gap in GDN. |
| 3 | `try_tape_reshape_kt` | `fn try_tape_reshape_kt` | 543 | |
| 4 | `try_tape_cross_entropy_from_logits_kt` | `fn try_tape_cross_entropy_from_logits_kt` | 749 | SFT loss root. Also builds `KtTensor::from_vec_on(device, ...)` at `:793,:817` — needs PR2 Vulkan `from_vec_on`. |
| 5 | `try_tape_lora_add_kt` | `fn try_tape_lora_add_kt` | 935,936,937,938 | 4-tensor gate (base,x,proj.a,proj.b). |
| 6 | `try_tape_lora_linear_kt` | `fn try_tape_lora_linear_kt` | 1146,1147 | fused linear+LoRA; the common projection path. |
| 7 | `try_tape_flash_attn_kt` | `fn try_tape_flash_attn_kt` | 1490,1491,1492 | **SPECIAL — see flash row below. Do NOT widen.** |
| 8 | `try_tape_gdn_recurrent_kt` | `fn try_tape_gdn_recurrent_kt` | 1736 | GDN recurrence; backward = `GdnRecurrentBackward`. |
| 9 | `tape_record_gdn_recurrent_kt` | `fn tape_record_gdn_recurrent_kt` | 1809 | `matches!(device, ...)` form (gates the passed `device`, not `.device()`). Same widen. |
| 10 | `try_tape_causal_conv1d_prefill_kt` | `fn try_tape_causal_conv1d_prefill_kt` | 2049,2050,2051 | **SPECIAL — same `cfg(not(cuda))→Ok(None)` split as flash (body at `:2042/2047`). GDN causal-conv backward is CUDA-only; on a vulkan-only build this is ALREADY a no-op. Do NOT rely on widening 2049–2051 (dead under vulkan-only). See conv row below.** |
| 11 | `try_tape_gdn_l2_norm_scale_kt` | `fn try_tape_gdn_l2_norm_scale_kt` | 2176 | GDN step-4/5 L2 qk-norm. |
| 12 | `try_tape_gdn_gated_rms_norm_kt` | `fn try_tape_gdn_gated_rms_norm_kt` | 2277 | GDN step-8 gated RMSNorm. |
| 13 | `try_tape_cast_kt` | `fn try_tape_cast_kt` | 2391,2392 | |
| 14 | `try_tape_narrow_kt` | `fn try_tape_narrow_kt` | 2511,2512 | |
| 15 | `try_tape_gqa_expand_kt` | `fn try_tape_gqa_expand_kt` | 2621,2622 | GQA KV-head broadcast. |
| 16 | `try_tape_sdpa_fallback_kt` | `fn try_tape_sdpa_fallback_kt` | 2806,2807,2808,2809 | **This is the Vulkan attention backward path** (fused flash is CUDA-only). |

**`try_tape_flash_attn_kt` — special handling (DO NOT widen its gate):**
The body is split (`tape_forward.rs:1480`):
```rust
#[cfg(not(feature = "cuda"))] { let _ = (...); return Ok(None); }
#[cfg(feature = "cuda")] { ... q.dtype()==BF16 ... Cuda|Metal gate ... flash_attn_fwd_kt ... }
```
On a vulkan-only build (`not(feature = "cuda")`) this recorder is **already a
no-op `return Ok(None)`** — correct, because fused FlashAttention-2 is a CUDA
kernel with no Vulkan equivalent. Vulkan attention records its backward through
**`try_tape_sdpa_fallback_kt`** (#16) instead, exactly as Metal does. **Leave
the flash gate at `Cuda | Metal`.** A global `replace_all` would touch lines
1490–1492 inside the `cfg(feature="cuda")` arm; that is harmless (dead under
vulkan-only) but misleading. Preferred: do the global `replace_all`, then
manually revert the three flash-attn lines (1490–1492) back to `Cuda | Metal`
and leave the explanatory comment at `:1477–1479`.

**`try_tape_causal_conv1d_prefill_kt` — special handling (like flash, #7):**
The body is split the same way (`tape_forward.rs:2042`):
```rust
#[cfg(not(feature = "cuda"))] { let _ = (...); return Ok(None); }  // GDN conv bwd is CUDA-only
#[cfg(feature = "cuda")] { ... Cuda|Metal gate ... }
```
On a vulkan-only build this recorder is **already a no-op** because the GDN
causal-conv1d **backward** kernel is CUDA-only (comment at `:2040`) — exactly the
state on Metal (GDN-conv training is not supported there either). So the GDN
depthwise-conv edge does **not** record on Vulkan, matching Metal. **Do not
attempt to make it record in PR5** — that needs a device-agnostic conv1d backward
composite (a separate op-port, out of PR5 scope). A global `replace_all` touches
2049–2051 (dead under vulkan-only); harmless but, like flash, you may revert them
to `Cuda | Metal` to avoid implying Vulkan support. **Note this gap explicitly in
the PR description: "GDN causal-conv1d backward stays CUDA-only on Vulkan, same as
Metal; record-coverage is a follow-up op-port."**

**Recorders that need NO edit (already device-agnostic — confirm by absence of a
device gate):** `try_tape_silu_kt` (143), `try_tape_add_kt` (161),
`try_tape_rope_kt` (184), `try_tape_matmul_kt` (384), `try_tape_embedding_kt`
(410), `try_tape_swiglu_kt` (443), `try_tape_mul_kt` (470). These dispatch
purely through `kiln_tensor::ops::*` and will record on Vulkan as soon as the
module compiles. **List them in the PR description as "covered for free".**

### 2.3 `crates/kiln-model/src/vk_forward.rs` — retire (1762 LOC)

- **Anchor:** whole file; module decl `lib.rs:38–39`
  ```rust
  #[cfg(feature = "vulkan")]
  pub mod vk_forward;
  ```
  grep token: `pub mod vk_forward;`
- Public surface to be retired (grep `pub fn|pub struct` in the file):
  `VkLoraPair`(59), `VkLoraLayer`(128), `VkFullAttentionWeights`(160),
  `VkLinearAttentionWeights`(183), `VkModelWeights`(214) + `impl
  from_gpu_weights`(1515) [the **GpuWeights→VkModelWeights bridge**],
  `vk_linear_with_lora`(270), `vk_model_forward_loss`(846) +
  `_with_state`(1292) + `_masked_with_state`(1313),
  `vk_model_forward_final_norm_with_state`(857), the **GRPO ref-logprob reimpl**
  (`VkGrpoReferencePrefix`(931), `vk_grpo_reference_prefill_prompt`(1010),
  `vk_grpo_reference_log_probs_from_prefix`(1184),
  `vk_grpo_reference_log_probs_full_sequence`(1226)), and
  `vk_step_backward`(1365) / `vk_count_gdn_layers`(1355) /
  `vk_compute_rope_tables*`(340,349) / `vk_*_layer*` / `vk_*_mlp_*`.
- **Sequencing constraint (load-bearing):** `vk_forward.rs`'s ONLY consumer is
  `crates/kiln-train/src/vk_train.rs` (confirmed: `vk_train.rs:31` `use
  kiln_model::vk_forward::{...}` + ~40 call sites of `vk_linear_with_lora` /
  `vk_model_forward_loss*` / `vk_step_backward`). `vk_train.rs` is the 6109-LOC
  fork deleted in **PR7**. Therefore **PR5 cannot `rm vk_forward.rs` while
  `vk_train.rs` still imports it** — that would break `cargo build --features
  vulkan`. Resolve with one of two options; the spec mandates **Option A**:
  - **Option A (chosen — keep PR5 self-contained & green):** In PR5, do NOT
    delete the file. Instead (i) move it behind a temporary
    `#[cfg(feature = "kiln_vk_native_training")]`-style internal cfg OR keep it
    compiled but mark the whole module `#[deprecated(note = "superseded by
    model_forward_kt on Device::Vulkan; deleted in PR7 with vk_train.rs")]`, and
    (ii) land the **per-op forward parity scaffold** (§5.2) that pins
    Vulkan-kt-forward == `vk_forward.rs` BEFORE deletion. The actual `rm` of both
    `vk_forward.rs` and `vk_train.rs` happens together in **PR7**, because they
    are one dependency unit. Update the parent plan's "PR5 deletes
    vk_forward.rs" line to "PR5 proves parity + deprecates; PR7 deletes
    vk_forward.rs with vk_train.rs" — they cannot be separated without stubbing
    40 call sites.
  - **Option B (only if PR6 already routed vk_train off vk_forward):** if, by the
    time PR5 lands, PR6 has already re-pointed `vk_train.rs`'s SFT/GRPO/OPD
    entry points at the shared `trainer.rs`/`opd.rs` and nothing imports
    `vk_forward`, then PR5 may `rm` it outright. Verify with
    `grep -rn "vk_forward" crates/ | grep -v vk_forward.rs` returning empty
    before deleting.

  **Decision recorded:** ship Option A. The 1762-LOC deletion is *attributed* to
  PR5 in the plan but *physically executed* in PR7 alongside `vk_train.rs`; PR5's
  deliverable is the parity proof + deprecation + the recorder/forward
  harmonization that makes the deletion safe.

---

## 3. Data / ownership model (the part that bites)

### 3.1 `Arc`-of-`VulkanBuffer` lifetime
PR2 gave `VulkanStorage` an `Arc<VulkanBuffer>` (`vulkan_storage.rs`, parent plan
§PR2) so cloned `Tensor`s share device memory. PR5's recorders **clone inputs
into the backward op** (`x: x.clone()`, `q: q.clone()`, etc. — e.g.
`tape_forward.rs:371,397,1533`). A `Tensor::clone()` is an `Arc::clone` of the
storage, so the saved-tensor set in each `*Backward` holds **shared, refcounted**
device buffers — the device memory lives until the last `Tape` node referencing
it drops. **Invariant the implementer must preserve:** no recorder may save a
**non-contiguous view** whose lifetime outlives its base in a way the Vulkan
allocator can't track. The existing recorders already force `.contiguous()`
before saving where needed (`try_tape_transpose_kt:519`, `lora_linear:1194`);
keep that. Do not add `to_device`/`to_dtype` round-trips in the hot path — they
allocate fresh `Arc<VulkanBuffer>`s.

### 3.2 `TensorId` preservation (the chaining contract)
The tape connects producer→consumer by `TensorId`: a recorder's recorded output
id must be the id that flows into the next recorder's input (`tape_forward.rs`
doc at `:1774`, GDN). `kiln_tensor::Tensor::clone()` preserves `.id()`; the
Vulkan `TensorId` is already the shared `kiln_tensor_id::TensorId` (re-exported
at `vk_tensor.rs:25`). **Invariant:** PR5 introduces **no id remapping**. The
recorders already pass the original `&x` (not a copy with a new id) as the tape
input (`tape.record(&y, &[x, weight], ...)` at `:367`). Widening the device gate
does not touch this — Vulkan ids chain identically to CUDA/Metal ids. The only
place to watch: `try_tape_cross_entropy_from_logits_kt` builds *new* index
tensors via `from_vec_on(device, ...)` (`:793,:817`); those are **non-diff
helper inputs** (`active_idx`, `flat_indices`), not tape inputs — they must NOT
be recorded. They are not today; keep it.

### 3.3 dtype / contiguity envelope
The recorders' dtype gates are **device-independent** and stay as-is:
- rms_norm / lora: `BF16 | F32`, `x.dtype()==weight.dtype()`, contiguous
  (`:347–350`, `:945–953`).
- flash: `BF16` only (CUDA-only path; irrelevant on Vulkan).
- **Vulkan trains in F32 on the hot path** (parent plan §5.3, §6): Vulkan LoRA
  weights / activations / grads are F32 by design, so on Vulkan these recorders
  admit via the `F32` arm. **No BF16 cliff** — do not add a BF16-specific Vulkan
  branch. The BF16 arm stays for parity but the trained Vulkan path is F32.
- Contiguity: Vulkan kt ops (PR3) require contiguous inputs for matmul exactly as
  CUDA/Metal do; the recorders already enforce `is_contiguous()` (`:349`,
  `:1185`, `:1493`). No change.

### 3.4 Who owns the forward's intermediate tensors
`model_forward_kt` owns its intermediates on the stack; they drop at function
exit unless captured by a `*Backward` saved in the active `Tape`. Under a
tape-authoritative scope the `Tape` (thread-local, `with_thread_local_tape`)
holds the saved `Arc<VulkanBuffer>`s alive across the forward→backward boundary.
**Invariant:** the Vulkan forward MUST run inside the same
`with_thread_local_tape` / `with_tape_authoritative_scope_kt` scope the CUDA/Metal
forward uses, or recorders silently `Ok(None)` (`with_active_tape` returns
`None` with no scope). PR6 wires that scope for Vulkan; PR5's bounded reachability
test (§5.3) must establish a scope explicitly to exercise recording.

---

## 4. Concrete change list (ordered, copy-pasteable intent)

1. **`lib.rs:33`** — `#[cfg(any(feature = "cuda", feature = "metal"))]` →
   `#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]` on
   `pub mod tape_forward;`.
2. **`tape_forward.rs:103`** — same widen on the module-inner `#![cfg(...)]`.
3. **`tape_forward.rs`** — `replace_all` the 31 in-condition occurrences of
   `kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Metal(_)` →
   `kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Metal(_) |
   kiln_tensor::Device::Vulkan(_)`; **then revert the flash-attn lines
   (1490–1492) AND the causal-conv1d lines (2049–2051)** to `Cuda | Metal` —
   both are CUDA-only-backward no-ops under vulkan-only, so widening them is
   dead/misleading (§2.2 flash row + conv row). Net recorders gaining a real
   Vulkan path: **14** (16 gated − flash − conv).
4. **Cascade fix** — run `cargo check -p kiln-model --features vulkan`; for each
   "cannot find / not compiled" on a `crate::forward::*` or `kiln_autograd::*`
   symbol that `tape_forward` imports, widen that item's `#[cfg(any(feature =
   "cuda", feature = "metal"))]` to add `feature = "vulkan"`. Likely set:
   `gdn_recurrent_forward_from_parts`, `gdn_*_backward_no_grad`,
   `sdpa_fallback_backward_no_grad`, `GDN_CHUNK_SIZE` in `forward.rs`. Each is a
   one-line gate widen; **do not change their bodies.**
5. **`vk_forward.rs` / `lib.rs:38`** — apply Option A (§2.3): add
   `#[deprecated(note = "...deleted in PR7 with vk_train.rs")]` to the module and
   do NOT delete. (Deletion is PR7.)
6. **Tests** — add the bounded scaffold (§5). Do not run long paths.

---

## 5. Test plan

### 5.0 Parity acceptance thresholds (mirror the Metal OPD gate)
- **Forward parity (Vulkan-kt vs. `vk_forward.rs`), F32:** `max_abs_err ≤ 1e-5`
  elementwise, AND `max_rel_err ≤ 1e-4` on the lm-head logits (mirrors the
  resident-decode gate `vk_resident_decode_parity.rs:7` `≤ 1e-4` rel and the
  Metal OPD analytic-backward FD tolerance `1e-5`, `opd.rs:715`). Justification:
  both paths bottom out in the **same `vk_ops` SPIR-V kernels** over the same
  device buffers (parent plan §6), so F32 forward should be near-bit-exact; the
  `1e-5` abs / `1e-4` rel band only absorbs reduction-order differences between
  the legacy `vk_*` op order and the kt-op order.
- **FD backward parity (recorder → `*Backward::apply`), F32:** central finite
  difference, `max_abs_err ≤ 1e-3` on grads with `eps = 1e-3` (the standard
  FD-vs-analytic band used by the existing CPU autograd FD tests, e.g.
  `test_gdn_recurrent_backward_no_grad_matches_autograd_cpu` referenced at
  `tape_forward.rs:1575`). This validates the device-agnostic `*Backward` runs
  correctly *on Vulkan storage*, not just CPU.

### 5.1 BOUNDED tests (run without long training — these are the gate)
All are **single-step / single-op**, build a tiny tensor on `Device::Vulkan(0)`,
and skip gracefully when `vulkan::vulkan_is_available()` is false (mirror
`vk_resident_decode_parity.rs:57` `supports_resident_decode()` skip). Place in
`crates/kiln-model/tests/` (new file, see scaffold) under `#![cfg(feature =
"vulkan")]`.

- **T1 `vk_tape_rms_norm_records_and_backprops`** — build F32 `x`,`weight` on
  Vulkan; open a `with_thread_local_tape` scope; call `try_tape_rms_norm_kt`;
  assert it returns `Some` (NOT `None` — proves the gate now admits Vulkan) and
  the tape recorded exactly 1 node; run `tape.backward(seed)` and assert
  `dL/dx`, `dL/dweight` are finite and the right shapes. **This is the core
  "does it record on Vulkan now" assertion.**
- **T2 `vk_tape_rms_norm_fd_parity`** — same op, FD vs. analytic grad, threshold
  §5.0 backward. (Bounded: one op, hidden≈8, rows≈4.)
- **T3 `vk_tape_sdpa_fallback_records`** — tiny q/k/v on Vulkan; assert
  `try_tape_sdpa_fallback_kt` returns `Some` + records `SdpaBackward` (proves the
  **Vulkan attention backward** path, since fused flash is CUDA-only).
- **T4 `vk_tape_gdn_recurrent_records`** — tiny GDN parts on Vulkan; assert
  `try_tape_gdn_recurrent_kt`/`tape_record_gdn_recurrent_kt` records
  `GdnRecurrentBackward` (proves the **GDN backward** records on Vulkan).
- **T5 `vk_tape_cross_entropy_records`** — `[1,T,V]` logits on Vulkan; assert
  `try_tape_cross_entropy_from_logits_kt` returns `Some` scalar loss + 1 node
  (exercises PR2 `from_vec_on(Device::Vulkan,...)` for the index helpers).
- **T6 `vk_forward_per_op_parity_pre_delete`** — see §5.2.
- **T7 `vk_device_agnostic_recorders_still_record`** — sanity that
  `try_tape_{silu,add,matmul,embedding,swiglu,mul,rope}_kt` (no device gate)
  return `Some` on Vulkan (catches a regression if a future edit accidentally
  adds a gate).

Each Tn is one op or one tiny forward; none loops, none loads a real checkpoint,
none trains. Run named: `cargo test -p kiln-model --features vulkan
vk_tape_ -- --nocapture` (timeout 300s).

### 5.2 Fixed-seed per-op forward parity scaffold (BEFORE deletion) — T6
The mandated "Vulkan-kt forward vs `vk_forward.rs` before deletion" check. For
each retired `vk_forward` entry point, run BOTH on the **same fixed-seed weights
+ input** on `Device::Vulkan(0)` and compare:

| `vk_forward.rs` fn | Harmonized equivalent | Compare |
| --- | --- | --- |
| `vk_linear_with_lora` (`:270`) | `try_tape_lora_linear_kt` path / `kiln_tensor::ops::matmul` + LoRA | logits `max_abs_err ≤ 1e-5` |
| `vk_model_forward_loss` (`:846`) | `model_forward_kt` + `try_tape_cross_entropy_from_logits_kt` | scalar loss `abs_err ≤ 1e-5` |
| `vk_grpo_reference_log_probs_full_sequence` (`:1226`) | `model_forward_kt` → token logprobs (shared `trainer::token_log_probs`) | per-token logprob `max_abs_err ≤ 1e-5` |

Build a single tiny `VkModelWeights` via `from_gpu_weights` (`:1515`) AND the
same `GpuWeights` for `model_forward_kt`; assert the two logit/loss tensors agree
within §5.0. **This is the safety proof that retiring `vk_forward.rs` does not
change numerics.** It is bounded: one forward pass, one tiny model, no training
loop. Seed via `SftConfig{seed:Some(0)}`-style fixed init (mirror
`real_model_integration.rs:924`). Mark `#[ignore]` until PR3 lands `matmul
vulkan_fwd` (without it the harmonized matmul can't run); the scaffold compiles
and documents the gate.

### 5.3 Single cheap reachability smoke (AT MOST ONE — host-safety ceiling)
Mirror `test_real_model_opd_metal` (`real_model_integration.rs:1112`) exactly,
swapping `Device::Metal(0)`→`Device::Vulkan(0)` and the guard
`backend::metal::try_new_metal().is_none()` →
`!kiln_model::backend::vulkan::vulkan_is_available()`. **One** new test
`test_real_model_sft_vulkan_reachability` that:
- builds the tiny model on `Device::Vulkan(0)`, F32 (NOT bf16 — Vulkan trains
  F32, §3.3; use `tiny_weights` not `tiny_weights_bf16`),
- runs `sft_train` for **`epochs: 1`** on the 2 tiny examples,
- **frontier assertion** (the OPD-test pattern): either it completes with an
  adapter written + finite loss (full success), OR it stops at a *documented*
  op-gap (a kt op still missing `vulkan_fwd` from PR3) — and the test asserts the
  error message matches one of the documented gaps, **failing if it stops
  anywhere unexpected.** This pins the frontier without a long run.
- **HOST-SAFETY:** `epochs:1`, 2 examples, tiny config. This is the single
  permitted real-model smoke. **Do NOT** add a multi-epoch or loss-decrease
  variant in PR5 — that is a human-gated soak step (§5.4).

### 5.4 Human-gated GPU-soak steps (explicitly SEPARATE — do NOT run autonomously)
These are described for the human operator, NOT executed by the implementer
(the host has hard-crashed on long runs — `MEMORY.md`, parent plan §7):
1. Full Vulkan SFT loss-decrease soak: `test_real_model_sft_vulkan` with
   `epochs:3` + `assert_loss_decreases` (the Metal SFT analog,
   `real_model_integration.rs:892`). Run by a human on Strix Halo.
2. Full Vulkan GRPO soak: `lora_grad_norms` receipt check (Metal analog `:975`)
   — proves the tape walked + deposited nonzero LoRA grads on Vulkan.
3. Full Vulkan OPD soak (once PR4 backward families, if any, land).
4. End-to-end real Qwen3.5-4B checkpoint forward parity
   (`vk_resident_decode_parity.rs` style, set `KILN_RESIDENT_DECODE_PARITY_MODEL`).
5. Steady-state throughput regression vs. the fork baseline
   (`bench-results/vulkan-strix-halo-baseline.md`) — confirms PR3 zero-copy held.

**The implementer runs only §5.1–§5.3 (bounded). §5.4 is handed to the human.**

---

## 6. Open risks + de-risking

1. **R1 — Module-gate cascade is larger than expected.** Widening
   `tape_forward`'s gate may pull in a long tail of cuda/metal-gated
   `crate::forward` helpers. *De-risk:* this is a compile-only failure surfaced
   by `cargo check -p kiln-model --features vulkan` in seconds; fix is mechanical
   gate-widening, body-untouched. Bounded, host-safe. If a pulled-in helper has a
   genuinely cuda-only body (an FFI symbol), STOP and report it as a frontier gap
   — do not stub it.
2. **R2 — A backward composite calls a kt op with no `vulkan_fwd` (PR3 gap).**
   E.g. `RmsNormKtBackward::apply` calls `sum_axis`, `sqrt`, `reciprocal`,
   `broadcast_mul`. If any lacks `vulkan_fwd`, the recorded backward errors at
   `tape.backward()` time (not record time). *De-risk:* T2's FD test surfaces it
   immediately on a single op; the §5.3 frontier assertion catches it in the
   smoke. Report the missing op as a PR3 follow-up, do not work around it.
3. **R3 — `vk_forward.rs` cannot be deleted in PR5 (vk_train dependency).**
   Already resolved: §2.3 Option A defers the `rm` to PR7. *Risk if ignored:*
   `cargo build --features vulkan` breaks. The parity scaffold (§5.2) is the
   independent deliverable that de-risks the eventual deletion.
4. **R4 — Recording without a scope silently no-ops.** If PR5's reachability
   smoke forgets to open a tape-authoritative scope, recorders return `Ok(None)`
   and the test "passes" while recording nothing. *De-risk:* T1/T3/T4/T5 assert
   `Some` + `tape.len()==1` (positive-record assertions), not just "no panic".
   The frontier smoke must run under the same scope `sft_train` opens for
   CUDA/Metal — verify the scope is Vulkan-reachable (PR6 wires it; if PR6 isn't
   in yet, the smoke is `#[ignore]` with a note).
5. **R5 — Two-autograd-engine corruption.** Parent plan §5.1: not live (engines
   split by device, no shared persistent tape across in-place mutation). PR5 does
   not bridge the two engines (that's PR4). *De-risk:* PR5's Vulkan forward
   records onto the **shared** `kiln_autograd::Tape` only; the legacy
   `vk_autograd` is not invoked on the harmonized path. No new sharing introduced.
6. **R6 — F32 vs. BF16 envelope mismatch.** A recorder admits Vulkan only via the
   `F32` dtype arm; if a caller hands BF16 Vulkan tensors the recorder returns
   `None` and silently drops the node. *De-risk:* Vulkan trains F32 by design
   (§3.3); T7 + the frontier smoke use F32. If a BF16 Vulkan input ever appears,
   it is a caller bug, not a recorder bug — assert F32 at the smoke's weight
   construction.

---

## 7. Done-when (PR5 acceptance)

- [ ] `cargo check -p kiln-model --features vulkan` is clean with `tape_forward`
      compiled in (module + inner gate widened, cascade fixed).
- [ ] All 16 device-gated recorders admit `Device::Vulkan(_)` (flash-attn
      excepted, by design).
- [ ] T1–T5, T7 pass on hardware with a Vulkan device (skip gracefully without).
- [ ] T6 parity scaffold compiles; runs green where PR3 op coverage allows,
      `#[ignore]` + documented otherwise.
- [ ] §5.3 single reachability smoke either completes or pins a documented
      frontier op-gap.
- [ ] `vk_forward.rs` deprecated (NOT deleted — PR7), parity proof landed.
- [ ] §5.4 soak steps written up for the human; none run autonomously.
