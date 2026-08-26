# PR6 — Orchestration flip: route Vulkan SFT/GRPO/OPD through `trainer.rs` / `opd.rs`

Issue: #1082
Branch: `feat/vk-tape-harmonization`
Status: **SPEC ONLY — not implemented.** PR1/PR2 are landed in this branch; PR3–PR7 are specced.
Parent plan: `vulkan-train-harmonization-plan.md` §4 PR6, §5 risk #2.

> All `file:line` anchors below were grepped against the worktree at branch HEAD
> (`b94feeac`). **Line numbers drift** — every anchor names the enclosing
> function / `cfg` / string so the implementer re-greps the symbol, not the
> number. Each site lists the exact symbol to search.

---

## 0. One-paragraph thesis

PR6 is the **big LOC-deleting flip, gated late**. It does *not* invent any GPU
machinery. It **widens the device gates** that currently route SFT/GRPO/OPD onto
the shared kt-tape substrate so that `Device::Vulkan(_)` is accepted exactly the
way `Device::Cuda(_)` and `Device::Metal(_)` already are, and **flips the
server-side default** so Vulkan training jobs go through `kiln_train::trainer` /
`kiln_train::opd` instead of `kiln_train::vk_train`. Metal is the fully-harmonized
template at every site; PR6 is "add `| kiln_tensor::Device::Vulkan(_)` next to
the existing `Device::Metal(_)` arm, and add `feature = "vulkan"` next to the
existing `feature = "metal"` cfg." The fork (`vk_train.rs`, `vk_forward.rs`) is
NOT deleted here — that is PR7. PR6 leaves `KILN_VK_NATIVE_TRAINING` as an
explicit **opt-OUT for one release**.

---

## 1. Prerequisites (PR6 is unsafe to land before these)

PR6 only flips orchestration; it assumes the substrate underneath already
records and differentiates Vulkan tensors. The hard dependency is **PR5**, with
PR2/PR3/PR4 transitively required:

| Prereq | What it must deliver before PR6 routes | Verify |
| ------ | -------------------------------------- | ------ |
| **PR2** (landed) | `kiln_tensor::Tensor` instantiable on `Device::Vulkan`; `host_to_vulkan_copy`/`vulkan_to_host_copy`. | `crates/kiln-tensor/src/vulkan_storage.rs:277` (`host_to_vulkan_copy`), `:~360` (`vulkan_to_host_copy`); `VulkanStorage` holds `Arc<VulkanBuffer>` (`:54`). |
| **PR3** | `MatmulOp::vulkan_fwd` + zero-copy `VulkanStorage`↔`VkTensor` bridge. | `crates/kiln-tensor/src/ops/matmul.rs` has a `vulkan_fwd`. |
| **PR4** | Vulkan backward ops presented to `kiln_autograd::Tape` (`VkBwdAdapter`). | tape walk produces non-empty grads for a Vulkan forward. |
| **PR5** | **The 31 `try_tape_*_kt` recorders accept Vulkan inputs.** Today every recorder in `crates/kiln-model/src/tape_forward.rs` is module-gated `#![cfg(any(feature = "cuda", feature = "metal"))]` (`:103`) and each device guard is `matches!(.., Device::Cuda(_) \| Device::Metal(_))` (31 sites — grep `Device::Cuda(_) \| kiln_tensor::Device::Metal(_)`). PR5 must extend the module cfg to include `feature = "vulkan"` and add `\| Device::Vulkan(_)` to every guard, **and** add `\| feature="vulkan"` to `model_forward_kt`'s Vulkan forward branch. | `grep -c 'Device::Vulkan' crates/kiln-model/src/tape_forward.rs` must be **31, not 0**. |

**Hard gate for PR6 start:** `grep -c 'Device::Vulkan' crates/kiln-model/src/tape_forward.rs == 31`
and `crates/kiln-kt-bridge/src/tape_bridge.rs:27` module cfg already includes
`feature = "vulkan"` (PR6 owns that one line — see §3.3). If the recorders still
decline Vulkan inputs (return `None`), every routed step will hit the
`returned None` bail and PR6 looks "done" but trains nothing. **Do not skip the
recorder-coverage assertion in §6.**

What PR6 unblocks: **PR7** (delete `vk_train.rs` 6109 LOC + `vk_forward.rs` 1762
LOC, remove the `KILN_VK_NATIVE_TRAINING` gate). PR6 is the last gate before the
fork is dead code.

---

## 2. Scope summary (what PR6 changes, in one list)

1. **SFT tape-auth gate** — `standard_forward_backward` (`trainer.rs`, the
   `matches!(device, Cuda \| Metal)` ensure at `:7592`) → add `Vulkan`.
2. **SFT gradient-checkpointing gate** — `checkpointed_forward_backward_tape_authoritative_kt`
   is `#[cfg(any(cuda, metal))]` (`:7380`) **and** its call site at the SFT loop
   (`:2478`, the `if let Some(ref segs)` arm) is cfg-gated `(cuda, metal)`. Vulkan
   is **explicitly excluded today** (plan §5 risk #2). PR6 extends BOTH.
3. **GRPO tape-auth gate** — `tape_auth_eligible` (`trainer.rs:4870`,
   `matches!(device, Cuda \| Metal)`) → add `Vulkan`; the per-completion dispatch
   cfg-block at `:5001` → add `feature = "vulkan"`; the function
   `grpo_step_forward_backward_tape_authoritative_kt` (`:7662`, `#[cfg(any(cuda,
   metal))]`) → add `feature = "vulkan"`; `tape_authoritative_enabled` (`:7217`)
   and `base_dtype_supports_tape` (`:7237`) cfg → add `feature = "vulkan"`.
4. **GRPO scalar-loss adapter gate** — `try_tape_grpo_pg_loss_from_logits_kt`
   (`grpo_candle_shim.rs:495` `#[cfg(any(cuda, metal))]` + the device guard at
   `:516-519`) → add `feature = "vulkan"` and `| Device::Vulkan(_)`.
5. **OPD tape-auth gate** — `opd_step_forward_backward_tape_authoritative`
   (`opd.rs:2948`, `#[cfg(any(cuda, metal))]`) and its dispatch cfg-block at
   `:2724` → add `feature = "vulkan"`.
6. **OPD scalar-loss adapter gate** — `try_tape_opd_scalar_mean_cuda_kt`
   (`opd_candle_shim.rs:93` `#[cfg(any(cuda, metal))]`) → add `feature = "vulkan"`.
7. **OPD kernel envelope** — `envelope_ok` in
   `crates/kiln-opd-loss-kernel/src/kt_tape.rs:91` (device guard at `:99`,
   `Cuda \| Metal`) → add `| Device::Vulkan(_)`; **and** the device-agnostic
   composite's host-upload helpers (`kt_api.rs` `opd_top_k_reverse_kl_phase_b_bwd_composite_kt`,
   `upload_u32`/`upload_f32` at `:788`/`:805`) must grow a `Vulkan` arm or a
   documented host-bounce (see §4 op-coverage audit — this is the one *real* code
   gap, not a one-line gate).
8. **`with_tape_authoritative_scope_kt`** — `tape_bridge.rs:198`, module cfg
   `#![cfg(any(cuda, metal))]` at `:27` → add `feature = "vulkan"`.
9. **Server routing default** — flip `vk_native_*_enabled` so the default
   in-process path is the shared trainer, keeping `KILN_VK_NATIVE_TRAINING` as
   opt-OUT (`main.rs:855`, `api/training.rs:168/188`, `training_queue.rs:415/432/446`,
   plus the routing branches `training_queue.rs:369/510/588/805`). Update the
   capability string `backend/vulkan.rs:1808`.
10. **Reachability smokes** — add `test_real_model_{sft,grpo,opd}_vulkan` mirroring
    the Metal trio in `crates/kiln-server/tests/real_model_integration.rs`.

---

## 3. Exact change at each site

> Convention used throughout: where Metal reads
> `kiln_tensor::Device::Cuda(_) | kiln_tensor::Device::Metal(_)` add
> `| kiln_tensor::Device::Vulkan(_)`; where it reads
> `#[cfg(any(feature = "cuda", feature = "metal"))]` change to
> `#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]`
> (and the negative `not(any(...))` arms get `feature = "vulkan"` added in lock-step).

### 3.1 `crates/kiln-train/src/trainer.rs`

| Anchor (grep) | Current | Change |
| ------------- | ------- | ------ |
| `fn standard_forward_backward` ensure! at `:7592` | `matches!(device, Cuda(_) \| Metal(_))` | add `\| Vulkan(_)`. Update the bail string `:7593` ("requires a CUDA device") to "CUDA, Metal, or Vulkan". |
| `standard_forward_backward` cfg block `:7589` + `not(...)` `:7614` | `cfg(any(cuda, metal))` | add `feature = "vulkan"` to both arms; update the `:7626` bail text. |
| `fn tape_authoritative_enabled` cfg `:7217` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. |
| `fn base_dtype_supports_tape` cfg `:7237` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. **Note**: this returns `true` only for a BF16 base. Vulkan's hot path is **F32 by design** (plan §5 risk #3). See §7 risk R1 — this gate will REJECT F32-Vulkan training unless reconciled. |
| `fn standard_forward_backward_tape_authoritative_kt` cfg `:7267` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. |
| `fn grpo_step_forward_backward_tape_authoritative_kt` cfg `:7662` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. |
| `fn checkpointed_forward_backward_tape_authoritative_kt` cfg `:7380` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. (Plan §5 risk #2: this is **added scope**, not a parity bug — no prior Vulkan checkpointing behavior exists.) |
| SFT loop checkpoint dispatch `:2478` (`if let Some(ref segs)` true-arm) + `not(...)` `:2493` | `cfg(any(cuda, metal))` | add `feature = "vulkan"` to both; update the `:2502` bail text. |
| GRPO `let tape_auth_eligible = ...` at `:4870` (`matches!(device, Cuda \| Metal)`) + the cfg pair `:4869`/`:4880` | gated + `Cuda \| Metal` | add `\| Vulkan(_)` to the `matches!` and `feature = "vulkan"` to both cfg arms. **This is the `tape_auth_eligible` single-source-of-truth** that the §5 ensure! (`:4990`) and the checkpoint-bypass both read. |
| GRPO per-completion dispatch cfg `:5001` + `not(...)` `:5024` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`; the `unreachable!` text at `:5030` updates to mention vulkan. |
| `fn unused-on-non-gpu` helper cfg `:5990` | `cfg(any(cuda, metal))` | add `feature = "vulkan"` (grep the symbol at that line; it is one of the GRPO support fns — flip it in lock-step so the `vulkan`-only build compiles). |

**Total trainer.rs cfg sites to flip:** 10 `cfg(any(...))` + their 3 `not(any(...))`
counterparts (grep `cfg(any(feature = "cuda", feature = "metal"))` → 10 hits;
`cfg(not(any(...)))` → 3 hits). Plus 2 `matches!(device, ...)` device guards
(`:4873`, `:7592`).

### 3.2 `crates/kiln-train/src/opd.rs`

| Anchor (grep) | Current | Change |
| ------------- | ------- | ------ |
| `opd_train` dispatch block `:2724` (`#[cfg(any(cuda, metal))]`) + `not(...)` `:2771` | gated | add `feature = "vulkan"`; update the `:2773` bail text ("requires a CUDA or Metal build" → "+ Vulkan"). |
| `fn opd_step_forward_backward_tape_authoritative` cfg `:2948` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. Update the doc-comment at `:2941-2947` (it currently says the OPD backward `bail!`s on Metal — that is **already stale at HEAD**: the backward routes Metal/CPU through the device-agnostic composite; extend the comment to state Vulkan routes through the same composite). |

**Total opd.rs cfg sites:** 3 `cfg(any(...))` (grep count = 3) + the matching
`not(...)` arm at `:2771`.

### 3.3 `crates/kiln-kt-bridge/src/tape_bridge.rs`

| Anchor | Current | Change |
| ------ | ------- | ------ |
| Module cfg `:27` `#![cfg(any(feature = "cuda", feature = "metal"))]` | gated | → `#![cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]`. This single line makes `with_tape_authoritative_scope_kt` (`:198`), `with_io_mapping_scope` (`:156`), `with_tape_segment_backward_scope` (`:272`), and `register_input_mapping_kt` (`:133`) compile into the `vulkan`-only build. **The function bodies are device-agnostic** — they operate on `kiln_tensor::Tensor` / `kiln_autograd::Tape` / `KtTensorId` and contain zero `Device::` matches, so no body change is needed (verified: the only ids are `loss_kt.id()`, `KtTensorId::from_raw`, `ones_like`, `tape.backward_with_seeds`). |

### 3.4 `crates/kiln-train/src/grpo_candle_shim.rs`

| Anchor | Current | Change |
| ------ | ------- | ------ |
| `fn try_tape_grpo_pg_loss_from_logits_kt` cfg `:495` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. |
| device guard `:516-519` (`logits_kt.device()` `Cuda \| Metal`) | `Cuda \| Metal` | add `\| Vulkan(_)`. |
| The other `#[cfg(any(cuda, metal))]` in this file (8 sites: `:85,87,89,93,108,130,198,229`) | gated | add `feature = "vulkan"` to **all** — they are the supporting `token_log_probs` / `grpo_loss` / backward-op decls the routed path calls; flip in lock-step so the `vulkan`-only build links. (grep `cfg(any(feature = "cuda", feature = "metal"))`.) |

### 3.5 `crates/kiln-train/src/opd_candle_shim.rs`

| Anchor | Current | Change |
| ------ | ------- | ------ |
| `fn try_tape_opd_scalar_mean_cuda_kt` cfg `:93` | `cfg(any(cuda, metal))` | add `feature = "vulkan"`. |
| doc comment `:85-92` ("the recorded backward … is CUDA-FFI-only and `bail!`s on Metal — documented follow-up") | **stale at HEAD** | rewrite: the backward routes CPU/Metal/**Vulkan** through the device-agnostic composite `opd_top_k_reverse_kl_phase_b_bwd_composite_kt`; only `Device::Cuda(_)` uses the FFI kernel. (PR7 also flags this; PR6 fixes it here since PR6 makes it Vulkan-reachable.) |

### 3.6 `crates/kiln-opd-loss-kernel/src/kt_tape.rs`

| Anchor | Current | Change |
| ------ | ------- | ------ |
| `fn envelope_ok` device guard `:99` (`Cuda \| Metal`) | `Cuda \| Metal` | add `\| Vulkan(_)`. Update the comment block `:92-98` (claims the recorded backward `bail!`s off-CUDA — stale; it routes through the composite). |
| `impl BackwardOp for CudaOpdTopKReverseKlPhaseBBackward::apply` `:179` | `#[cfg(feature = "cuda")] if matches!(self.hidden.device(), Cuda(_))` then composite fallback (`:223`) | **No gate change needed** — the FFI branch is already `cfg(feature="cuda")`-AND-`Cuda(_)`-guarded, so a Vulkan tensor naturally falls through to the composite at `:223`. The composite is the path Vulkan uses. (Confirm by reading: the `#[cfg(feature = "cuda")] if ...` block returns early only for CUDA; everything else reaches `opd_top_k_reverse_kl_phase_b_bwd_composite_kt`.) |

### 3.7 `crates/kiln-opd-loss-kernel/src/kt_api.rs` — **the one real code gap**

`opd_top_k_reverse_kl_phase_b_bwd_composite_kt` (`:752`) is the device-agnostic
OPD backward Vulkan will run. Its two host→device upload closures only have
`Cpu` / `Cuda` / `Metal` arms:

```
let upload_u32 = |vals, shape| match dev {           // :788
    KtDevice::Cpu => KtTensor::from_vec(...),
    #[cfg(feature = "cuda")] KtDevice::Cuda(i) => KtTensor::cuda_from_slice(...),
    #[cfg(feature = "metal")] KtDevice::Metal(_) => from_vec(...).to_device(dev),
    #[allow(unreachable_patterns)] other => Err("unsupported device ..."),  // ← Vulkan lands here
};
```

A `Device::Vulkan` tensor will hit the `other =>` arm and **error**. Same for
`upload_f32` (`:805`). PR6 must add a Vulkan arm to **both**, mirroring the Metal
arm (host `from_vec` then `to_device(dev)` — PR2 made `to_device(Vulkan)` work):

```rust
#[cfg(feature = "vulkan")]
KtDevice::Vulkan(_) => KtTensor::from_vec(vals.to_vec(), shape)
    .and_then(|t| t.to_device(dev))
    .map_err(OpdLossError::Kt),
```

**Op-coverage note (host-bounce, documented):** the composite already does a
host bounce for `grad_loss` (`:845-860`, reads `grad_loss` to CPU via
`to_device(Cpu)` + `CpuStorage` downcast). On Vulkan that path requires
`vulkan_to_host_copy` (PR2, landed). The composite's *compute* ops
(`index_select`, `matmul`, `log_softmax_last_dim`, `exp`, `sub`, `mul`,
`sum_axis`, `broadcast_to`, `scatter_add`, `cast`, `to_f32`, `squeeze`,
`unsqueeze`, `reshape`, `permute`, `contiguous`) run on `dev` — see §4 for the
per-op coverage audit. The `KtDevice::Cpu` short-circuit zeros buffer at
`:826-834` already `to_device(dev)`-bounces, so it works on Vulkan for free once
the upload arms exist.

### 3.8 `crates/kiln-server/src/main.rs`, `api/training.rs`, `training_queue.rs`

The server currently **defaults Vulkan to the native fork**. PR6 inverts the
default to the shared trainer while keeping `KILN_VK_NATIVE_TRAINING` as the
explicit opt-OUT for one release.

| Anchor | Current behavior | Change |
| ------ | ---------------- | ------ |
| `fn vk_native_training_enabled` `main.rs:855` | `env_tristate(KILN_VK_NATIVE_TRAINING).unwrap_or_else(\|\| is_vulkan)` → **default ON on Vulkan** (only affects prewarm skip). | Flip default branch to `false` (so prewarm runs normally). Keep `Some(enabled) => enabled` so an explicit set still works. |
| `fn vk_native_sft_enabled` `api/training.rs:168` | `None => runner == "vulkan"` (default native). | Flip the `None` arm to `false`. The explicit `Some(enabled)` arm is unchanged (opt-out: set `KILN_VK_NATIVE_TRAINING=1` to restore native). |
| `fn vk_native_grpo_enabled` `api/training.rs:188` | falls back to `vk_native_sft_enabled`. | unchanged (inherits the flip). |
| `fn vk_native_sft_enabled` `training_queue.rs:415` | `None => backend_name == "vulkan"`. | Flip `None` arm to `false`. |
| `fn vk_native_grpo_enabled` `training_queue.rs:432`, `fn vk_native_opd_enabled` `:446` | fall back to `vk_native_sft_enabled`. | unchanged (inherit). |
| Routing branches: `training_queue.rs:369` (`if vk_native` → `vk_native_sft_train`), `:510` (`vk_native_grpo_train_jsonl`), `:588` (`vk_native_grpo_train`), `:805` (`vk_native_opd_train`) | route to fork when flag set. | **No structural change** — once the `_enabled` helpers default `false`, these branches are simply not taken by default. The `trainer::sft_train` / `grpo_train` / `opd::opd_train` fall-through (`:397`, `:614`, `:846`) becomes the default Vulkan path. |
| Capability string `backend/vulkan.rs:1808` `native_training: "vk_native_..._train enabled by default on Vulkan"` | stale once flipped. | Update to: `"shared kt-tape trainer (trainer.rs / opd.rs) by default; KILN_VK_NATIVE_TRAINING=1 opts back into vk_native_*_train for one release"`. |
| Unit tests `training_queue.rs:2581-2592` (`vk_native_grpo_enabled("vulkan")` asserts `true` by default) | encode old default. | **Update these assertions** to the new default (`vk_native_grpo_enabled("vulkan")` with no env → `false`; with `KILN_VK_NATIVE_TRAINING=1` → `true`). This is the bounded unit test that proves the flip (see §6). |

**Do NOT** delete `vk_native_sft_train` / `vk_native_grpo_train` /
`vk_native_opd_train` or any `vk_train.rs` symbol — they remain reachable via the
opt-out flag through this release. Deletion is PR7.

---

## 4. Op-coverage audit (GRPO ref-logprobs + OPD composites on `Device::Vulkan`)

PR6's correctness rests on every op the routed composites touch having a
`Device::Vulkan` path **or** a documented host-bounce. The forward
(`model_forward_kt`, `model_forward_no_head`) coverage is **PR5's** contract; PR6
audits the **loss-root + backward composites** it newly reaches.

### 4.1 GRPO reference log-probs (`trainer.rs` GRPO path)

GRPO computes `ref_log_probs` via `model_forward_no_head` +
`selected_log_probs_from_normed_hidden_chunked` (`:4943-4961`) and the policy via
`grpo_step_forward_backward_tape_authoritative_kt` →
`try_tape_grpo_pg_loss_from_logits_kt` → `token_log_probs` + `grpo_loss`. These
are all `kiln_tensor` ops that route through `dispatch1/2/3` (which already have
`Device::Vulkan` arms — plan §3, `device_op.rs:163,217,268`). **PR6 op-coverage
requirement:** the device-agnostic `token_log_probs` / `grpo_loss` composites
(`grpo_candle_shim.rs`) use `log_softmax`, `gather`/`index_select`, elementwise
`sub`/`mul`, `sum`. PR5 must have landed the Vulkan `vulkan_fwd` for any of these
still on the `ops/{reduce,scalar,silu_mul}.rs` missing list (plan §2). **Audit
action:** before routing, grep each op the GRPO loss composite calls and confirm
it has a `vulkan_fwd` or a host-bounce; list any gap as a blocking TODO.

### 4.2 OPD top-K reverse-KL backward composite (`kt_api.rs:752`)

This is the device-agnostic backward Vulkan runs (the CUDA FFI kernel is skipped
for non-CUDA). Per-op coverage table — fill the "Vulkan path" column by grepping
each op's `vulkan_fwd` impl under `crates/kiln-tensor/src/ops/` during
implementation:

| Op (in composite) | Call site | Vulkan path needed | Notes |
| ----------------- | --------- | ------------------ | ----- |
| `to_f32` / `cast` | `:845,895,896,930-933` | `cast` `vulkan_fwd` | F32↔BF16. |
| host readback of `grad_loss` | `:845-860` | `vulkan_to_host_copy` (PR2) | **documented host-bounce**, tiny `[1]`/`[T_active]`. |
| `from_vec` + `to_device(Vulkan)` uploads | `upload_u32/f32` `:788,805` | **add Vulkan arm (§3.7)** | the one code gap. |
| `index_select` | `:894,899` | `index_select` `vulkan_fwd` | gather head rows + active rows. |
| `matmul` | `:904,927` | **`MatmulOp::vulkan_fwd` (PR3)** | batched `[T,1,H]@[T,H,K]` + `[T,H,K]@[T,K,1]`. |
| `log_softmax_last_dim` | `:907,908` | `log_softmax` Vulkan kernel | Metal got `metal_log_softmax_last_axis`; Vulkan needs the equivalent (verify a Vulkan `log_softmax` SPIR-V exists under `vk_ops/softmax`). **If missing → blocking TODO.** |
| `exp` | `:909` | `exp` `vulkan_fwd` | |
| `sub` / `mul` | `:910,918,919,921` | scalar/elementwise `vulkan_fwd` | on the `ops/scalar.rs` missing list (plan §2) — **verify PR5 closed it**. |
| `sum_axis` | `:913` | reduce `vulkan_fwd` | on the `ops/reduce.rs` missing list — **verify PR5 closed it**. |
| `broadcast_to` | `:915,920` | broadcast `vulkan_fwd` | |
| `reshape`/`permute`/`squeeze`/`unsqueeze`/`contiguous` | many | metadata/contig `vulkan_fwd` | `permute(...).contiguous()` at `:901` forces a real copy — needs a Vulkan contiguous path. |
| `scatter_add` | `:935` | `scatter_add` `vulkan_fwd` | scatters active rows into `[T,H]`. **Verify a Vulkan path exists; if not, document a host-bounce.** |

**Deliverable of this audit:** a checklist in the PR description enumerating each
op above with its confirmed `vulkan_fwd` symbol **or** a one-line host-bounce
justification. Any op with neither is a **blocking TODO** that belongs in PR3/PR5,
not PR6 — PR6 must not paper over a missing kernel with a silent host bounce that
tanks perf (plan §6: the only perf lever is the PR3 zero-copy bridge; an
unplanned per-op D2H/H2D in the OPD backward would regress badly).

### 4.3 SFT loss root (`try_tape_cross_entropy_from_logits_kt`)

`tape_forward.rs:731`, device guard `:749` (`Cuda \| Metal`). This is **PR5's**
gate to widen (it lives in the PR5-owned module). PR6 only routes into it; confirm
PR5 added `| Vulkan(_)` at `:749` and the FLCE-free CE-from-logits path runs on
Vulkan storage. **Note:** SFT does *not* use the FLCE `kt_tape` envelope
(`crates/kiln-flce-kernel/src/kt_tape.rs:88`, which is `Cuda`-only and which PR6
does **not** touch — SFT's loss root is the CE-from-logits adapter, not FLCE
kt-tape). Do not widen the FLCE kt_tape gate in PR6; it is out of scope and
CUDA-only by design here.

---

## 5. Data / ownership model (load-bearing invariants)

These are the invariants the flip must preserve; mirror exactly how CUDA/Metal
already satisfy them.

1. **`Arc<VulkanBuffer>` lifetime.** `VulkanStorage` owns `buffer: Arc<VulkanBuffer>`
   (`vulkan_storage.rs:54`) and `vulkan_device: Arc<VulkanDevice>` (`:61`). A
   cloned `Tensor` shares device memory via the `Arc` refcount (PR2). The tape
   saves inputs by `.clone()` (e.g. `CudaOpdTopKReverseKlPhaseBBackward { hidden:
   ... }` holds `KtTensor` clones — `kt_tape.rs:146`), which is an `Arc` bump, not
   a copy. **Invariant:** the buffer must outlive the backward walk. The tape
   holds its own `Arc` clones of every saved input, so the buffer survives even if
   the forward's `Tensor` is dropped — same as CUDA. No change; just don't break
   it by introducing a non-`Arc` view.
2. **`TensorId` preservation.** `crates/kiln-vulkan-kernel/src/vk_tensor.rs:25`
   re-exports `kiln_tensor_id::TensorId`, so Vulkan grads key compatibly with the
   shared `GradStore` and the AdamW book (plan §3). The whole tape-auth grad
   plumbing keys by `Parameter::tensor_id().as_raw()` (`trainer.rs:7329-7344`) and
   the `KT_PARAM_DEPOSIT_TAG` namespace tag (`tape_bridge.rs:142`). **Invariant:**
   a LoRA `Parameter` on Vulkan must produce a stable `tensor_id()` across the
   forward→backward→optimizer hop. This already holds for CUDA/Metal because the
   id is storage-independent; verify the Vulkan `Tensor` constructors (PR2)
   preserve the id through `to_device`.
3. **dtype.** GRPO/SFT tape adapters are **BF16-only** (`base_dtype_supports_tape`,
   `trainer.rs:7237`). The OPD composite accepts **F32 or BF16** (`validate_inputs_kt`
   envelope). Vulkan's hot path is **F32 by design** (plan §5 risk #3). This is the
   central tension — see §7 R1. **Invariant for the OPD lane:** F32-Vulkan flows
   through the composite cleanly (F32 ∈ envelope). **Invariant for the SFT/GRPO
   lane:** they require BF16 base — a BF16-base Vulkan model is required for those
   two to route, OR R1's resolution.
4. **Contiguity.** The OPD composite calls `.contiguous()` after `permute`
   (`kt_api.rs:901`) and after each matmul (`:904,927`). Vulkan `contiguous` must
   produce a genuinely packed buffer (PR3/PR5). The matmul forward (PR3) must
   accept the contiguous layout the composite produces. **Invariant:** every
   `vulkan_fwd` the composite hits assumes contiguous F32 inputs unless its
   kernel documents strided support.

---

## 6. Bounded test plan (runs WITHOUT long training — host-safety constrained)

> The host has hard-crashed on long runs. **Everything here is bounded:** named
> unit tests, finite-difference parity on a single tiny tensor, and AT MOST one
> single-step reachability smoke per loss type. Full GPU soak is **§6.4,
> human-gated, NOT run autonomously.**

### 6.1 Prereq-coverage assertion (cheap, no GPU)

A compile-time / grep guard that PR6 didn't route ahead of PR5:

* **`pr6_recorders_accept_vulkan`** (a `#[test]` in `kiln-model` or a CI grep
  step): assert `grep -c 'Device::Vulkan' crates/kiln-model/src/tape_forward.rs
  == 31` and that `tape_bridge.rs:27` module cfg contains `feature = "vulkan"`.
  Fails loudly if PR6 is landed on top of an un-widened recorder layer.

### 6.2 OPD composite finite-difference parity on Vulkan (the numeric gate)

Mirror the existing FD test for the composite. The CUDA/CPU FD harness already
lives at `crates/kiln-opd-loss-kernel/src/kt_api.rs` tests
(`opd_top_k_reverse_kl_phase_b_bwd_composite_kt` validated against a central FD
in the test at `:1670`/`:1763`/`:1825`). PR6 adds a **Vulkan-device variant**,
gated `#[cfg(feature = "vulkan")]` and `#[ignore]` until a Vulkan device is
present (the test self-skips via `vulkan_is_available()`):

* **`opd_composite_bwd_fd_parity_vulkan`**: build tiny `hidden [1,T,H]`,
  `head_t [H,V]` F32 on `Device::Vulkan(0)`; run the composite backward and a
  central finite-difference reference (perturb each `hidden[t,h]` by ±ε, recompute
  the scalar OPD loss via the forward composite, `(L+ − L−)/2ε`).
* **Acceptance threshold (mirror the Metal OPD gate):** `max_abs_err ≤ 1e-5` for
  F32 (the same tolerance the existing composite FD test uses). For BF16, use the
  looser BF16 tolerance the existing test uses (grep the existing test's epsilon).

### 6.3 Single-step reachability smokes (mirror the Metal trio)

Add to `crates/kiln-server/tests/real_model_integration.rs`, each
`#[cfg(feature = "vulkan")]`, each self-skipping when
`kiln_model::backend::vulkan::vulkan_is_available()` is `false` (the file already
uses that guard at `:196`). These are **single-pass** (epochs ≤ 3, one tiny
group/prompt) — bounded, not a training loop:

* **`test_real_model_sft_vulkan`** — mirror `test_real_model_sft_metal:892`. One
  tiny BF16 model, 2 examples, `epochs: 3`, assert adapter written + losses finite
  + decreasing.
* **`test_real_model_grpo_vulkan`** — mirror `test_real_model_grpo_metal:975`.
  3 groups, `PerSample`, KL off, ECHO off; assert adapter written + finite losses
  + `receipt_lora_grad_norms(out).0 > 0` (grads flowed) + `max_mean_norm > 0`.
* **`test_real_model_opd_vulkan`** — mirror `test_real_model_opd_metal:1112`.
  `DeterministicUniformLogitSource`, K=32, off-policy, `epochs: 1`; assert adapter
  written + finite losses + grad-norm modules > 0. **This is the single allowed
  one-step real-model smoke** (per the validation ceiling).

The scaffold for these three is in `PR6-test-scaffold.rs` (drop-in, marked WIP /
`#[ignore]`). **Note the BF16 caveat:** SFT/GRPO need a BF16 base
(`base_dtype_supports_tape`), so the smokes use `tiny_weights_bf16` exactly like
Metal — but R1 (§7) must be resolved first if Vulkan can only run F32. If R1
lands as "OPD-on-F32-Vulkan first, SFT/GRPO BF16 later", ship `test_real_model_opd_vulkan`
in PR6 and mark the SFT/GRPO smokes `#[ignore = "needs BF16 Vulkan (R1)"]`.

### 6.4 Server-routing unit tests (cheap, no GPU)

* Update `training_queue.rs:2581-2592` `vk_native_grpo_enabled` assertions to the
  **new default**: no env → `false`; `KILN_VK_NATIVE_TRAINING=1` → `true`;
  `=0` → `false`. Add an analogous `vk_native_opd_enabled` case. These are the
  bounded proof that the default flipped without breaking the opt-out.

### 6.5 Build gates (the only cargo PR6 runs autonomously)

* `CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target cargo check -p kiln-train --features vulkan`
* `... cargo check -p kiln-server --features vulkan`
* `... cargo check -p kiln-opd-loss-kernel --features vulkan`
* `... cargo check -p kiln-kt-bridge --features vulkan`
* Run the named tests from §6.1/§6.4 (pure-CPU, bounded):
  `... cargo test -p kiln-server --features vulkan vk_native -- --nocapture`

### 6.6 HUMAN-GATED GPU soak (NOT run by the implementer agent)

Explicitly separated. A human runs these on the Strix Halo box, watching for the
host-crash failure mode:

1. The three `test_real_model_*_vulkan` smokes against a real Vulkan device
   (single-step each — still bounded, but GPU-resident).
2. A short end-to-end server run: `POST /v1/training/sft` with a 2-example
   dataset, default flags, confirm it routes through `trainer::sft_train` (log
   line: NOT "routing to vk_native_sft_train") and writes an adapter.
3. A 1-2 step OPD run on a real BF16 model to confirm grad-norms are non-zero and
   losses finite. Compare the first-step loss against the `vk_native_opd_train`
   fork (`KILN_VK_NATIVE_TRAINING=1`) within `1e-3` relative as a coarse parity
   check.
4. Capture before/after decode + step throughput into
   `bench-results/vulkan-strix-halo-baseline.md` (the harmonized path must not
   regress vs. the fork — plan §6).

---

## 7. Open risks + de-risking

**R1 — BF16 vs F32 train-dtype mismatch (highest).** `base_dtype_supports_tape`
(`trainer.rs:7237`) gates SFT/GRPO to a **BF16 base**, but Vulkan's hot path is
**F32 by design** (plan §5 risk #3). If a production Vulkan model is F32,
`tape_auth_eligible` (GRPO) and the SFT ensure! will be `true` (device matches)
but `base_dtype_supports_tape` returns `false`, and SFT bails at `:7597`. *OPD is
fine* (its composite envelope accepts F32). *De-risk:* (a) **Confirm the actual
Vulkan production base dtype.** The plan asserts F32 hot-path but also that the
BF16 AdamW arm exists "for parity"; clarify whether the trained Vulkan base is
BF16 (then SFT/GRPO route as-is) or F32 (then PR6 must either land an F32 kt-tape
adapter path for Vulkan or scope SFT/GRPO-on-Vulkan to a follow-up and ship only
OPD + the gate-widening). (b) Until resolved, ship **OPD-on-Vulkan** end-to-end
(F32-clean) and land SFT/GRPO gate-widening but mark their smokes
`#[ignore = "needs BF16 Vulkan base (R1)"]`. This is the honest partial: the LOC
flip is staged, OPD proves the substrate, SFT/GRPO follow once dtype is settled.

**R2 — silent `None`-decline = "trains nothing" false-green.** Every routed
loss-root adapter returns `Ok(None)` when its envelope declines (e.g. recorder
still CUDA/Metal-only, or wrong dtype), and the trainer turns that into a bail —
but a subtly mis-widened gate could route a step that records an **empty tape**
and deposits zero grads while still "completing." *De-risk:* the §6.3 smokes
assert `receipt_lora_grad_norms(out).0 > 0` AND `max_mean_norm > 0` (exactly the
Metal GRPO smoke's two-part assertion at `real_model_integration.rs:1065-1074`) —
an empty/severed tape fails loudly. The §6.1 recorder-coverage assertion catches
the most common cause before any GPU runs.

**R3 — OPD composite host-bounce perf cliff.** §4.2 lists ops that, if missing a
`vulkan_fwd`, would silently host-bounce inside the OPD backward (D2H→CPU op→H2D
per op), tanking step time (plan §6). *De-risk:* the §4.2 audit is a hard
deliverable — every op must show a confirmed `vulkan_fwd` symbol or an explicit,
justified host-bounce. A blocking TODO for any unbacked op routes the work to
PR3/PR5, not a silent regression in PR6.

**R4 — gradient checkpointing is brand-new on Vulkan.** Widening the
`checkpointed_forward_backward_tape_authoritative_kt` gate (`:7380` + call site
`:2478`) enables a path that **never ran on Vulkan** (plan §5 risk #2: added
scope, not a parity break). The per-segment fresh-tape design
(`with_tape_segment_backward_scope`, `tape_bridge.rs:272`) must hold an
`Arc<VulkanBuffer>` boundary across segments. *De-risk:* a bounded named unit test
**`ckpt_kt_two_segment_chains_input_grad_vulkan`** that runs the 2-segment
checkpoint reverse on a tiny Vulkan model and asserts the segment-input grad
chains (non-empty `kt_grads.get(seg_input_id)`), without a full training loop.
Alternatively, ship PR6 with Vulkan checkpointing **gated off** (route Vulkan SFT
to the non-checkpointed `standard_forward_backward` branch by leaving `segments =
None` for Vulkan) and land checkpointing as a fast-follow — a legitimate scope cut
since no prior behavior is lost.

**R5 — server default flip surprises an operator mid-release.** Flipping
`vk_native_*_enabled` defaults changes which trainer a Vulkan box uses with no
config change. *De-risk:* `KILN_VK_NATIVE_TRAINING=1` is the documented one-release
opt-OUT (kept live, fork not deleted until PR7). Emit a one-time `tracing::info!`
at job start naming the chosen path ("routing Vulkan SFT through shared kt-tape
trainer; set KILN_VK_NATIVE_TRAINING=1 to use the legacy native path"). The §6.4
unit-test update locks the new default into CI.

**R6 — `vulkan`-only build link errors from half-flipped cfgs.** The
`cfg(any(cuda, metal))` sites come in matched pairs (positive + `not(...)`) and
fan out across 5 crates (trainer, opd, kt-bridge, both candle-shims, opd-loss-kernel).
Missing one in a `vulkan`-only build (no cuda, no metal) yields either a missing
symbol or an `unreachable!` reached. *De-risk:* the §6.5 `cargo check --features
vulkan` per-crate is the mechanical gate; grep-confirm the post-edit counts:
`trainer.rs` should have **0** remaining bare `cfg(any(feature = "cuda", feature =
"metal"))` (all become the 3-feature form), likewise opd.rs / both shims /
kt-tape.rs / kt-bridge module cfg.

---

## 8. Implementation order (within PR6)

1. `kt-bridge/tape_bridge.rs:27` module cfg (one line) — unblocks the scope fn for
   the vulkan build.
2. `opd-loss-kernel`: `kt_tape.rs:99` envelope + `kt_api.rs:788/805` upload Vulkan
   arms — the only real code change; land + FD test (§6.2) first.
3. `kiln-train`: all `trainer.rs` / `opd.rs` / both candle-shim cfg + `matches!`
   widenings (§3.1/3.2/3.4/3.5).
4. `kiln-server`: default flip (§3.8) + unit-test updates (§6.4).
5. Reachability smokes (§6.3) + recorder-coverage assertion (§6.1).
6. `cargo check --features vulkan` across the 5 crates (§6.5); hand off the §6.6
   GPU soak to a human.

---

## 9. Reference index (re-grep at implementation time)

| Claim | Location | grep token |
| ----- | -------- | ---------- |
| SFT tape-auth ensure (`Cuda \| Metal`) | `trainer.rs:7592` | `kt tape-authoritative SFT requires a CUDA` |
| GRPO `tape_auth_eligible` (`Cuda \| Metal`) | `trainer.rs:4870` | `let tape_auth_eligible = tape_authoritative_enabled()` |
| Grad-checkpoint fn cfg (excludes Vulkan) | `trainer.rs:7380` | `fn checkpointed_forward_backward_tape_authoritative_kt` |
| Grad-checkpoint SFT call site | `trainer.rs:2478` | `if let Some(ref segs) = segments` |
| `tape_authoritative_enabled` cfg | `trainer.rs:7217` | `pub(crate) fn tape_authoritative_enabled` |
| `base_dtype_supports_tape` (BF16-only) | `trainer.rs:7237` | `fn base_dtype_supports_tape` |
| GRPO producer fn cfg | `trainer.rs:7662` | `fn grpo_step_forward_backward_tape_authoritative_kt` |
| SFT producer fn cfg | `trainer.rs:7267` | `fn standard_forward_backward_tape_authoritative_kt` |
| OPD dispatch cfg block | `opd.rs:2724` | `opd_step_forward_backward_tape_authoritative(` |
| OPD producer fn cfg + stale comment | `opd.rs:2948` | `fn opd_step_forward_backward_tape_authoritative` |
| `with_tape_authoritative_scope_kt` + module cfg | `tape_bridge.rs:198`, module cfg `:27` | `pub fn with_tape_authoritative_scope_kt` |
| GRPO loss adapter cfg + device guard | `grpo_candle_shim.rs:495,516` | `fn try_tape_grpo_pg_loss_from_logits_kt` |
| OPD loss adapter cfg + stale comment | `opd_candle_shim.rs:93,85-92` | `fn try_tape_opd_scalar_mean_cuda_kt` |
| OPD envelope device guard (`Cuda \| Metal`) | `kt_tape.rs:99` | `fn envelope_ok` |
| OPD composite upload helpers (no Vulkan arm) | `kt_api.rs:788,805` | `let upload_u32` |
| OPD composite (device-agnostic backward) | `kt_api.rs:752` | `pub fn opd_top_k_reverse_kl_phase_b_bwd_composite_kt` |
| OPD backward composite fallback (no gate change) | `kt_tape.rs:179,198,223` | `impl BackwardOp for CudaOpdTopKReverseKlPhaseBBackward` |
| PR5 recorder module cfg (31 Cuda\|Metal guards) | `tape_forward.rs:103` + 31 guards | `Device::Cuda(_) \| kiln_tensor::Device::Metal(_)` |
| SFT CE loss root device guard (PR5-owned) | `tape_forward.rs:749` | `fn try_tape_cross_entropy_from_logits_kt` |
| Server prewarm default-native | `main.rs:855` | `fn vk_native_training_enabled` |
| Server SFT/GRPO default-native | `api/training.rs:168,188` | `fn vk_native_sft_enabled` |
| Queue SFT/GRPO/OPD default-native + routing | `training_queue.rs:415,432,446` + `369,510,588,805` | `fn vk_native_sft_enabled` |
| Queue routing unit tests (encode old default) | `training_queue.rs:2581-2592` | `vk_native_grpo_enabled("vulkan")` |
| Capability string (default-on Vulkan) | `backend/vulkan.rs:1808` | `native_training: "vk_native` |
| `vulkan_is_available()` (test skip guard) | `backend/vulkan.rs:5059` | `pub fn vulkan_is_available` |
| `host_to_vulkan_copy` / Arc buffer (PR2) | `vulkan_storage.rs:277,54` | `pub fn host_to_vulkan_copy` |
| `TensorId` re-export (shared keying) | `vk_tensor.rs:25` | `pub use kiln_tensor_id::TensorId` |
| Metal reachability trio (template) | `real_model_integration.rs:892,975,1112` | `fn test_real_model_sft_metal` |
| `env_tristate` helper | `kiln-core/src/env_flag.rs:44` | `pub fn env_tristate` |
