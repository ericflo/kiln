# Issue #1082 — `forward.rs` bare-`Tensor` candle→kt type-flip: execution plan

Authoritative roadmap for the *remaining ref-reducing* candle-removal work in
kiln-model + kiln-train. Grounded in source at HEAD 2026-05-29 (post `fe81330f`).
Supersedes the looser "type-flip" framing in earlier notes.

## State entering this plan

- **CP-4 tape-authoritative training is DONE + default-on** (`ac3bd054`),
  finite-diff-validated as *more correct* than candle (candle `loss.backward()`
  silently drops full-attn + GDN-conv grads).
- **The method façade is in place** (`fe81330f` + `ce/1082-kt-facade-finish`):
  `kiln_tensor::Tensor` now has the candle-style method surface
  (matmul/exp/to_dtype/broadcast_*/dims/sum/keepdim/affine/sqr/is_finite/to_vec2…),
  so the math/view/ctor/readback surface is **mechanically flippable**.
- Hub-flip regions 1–3 landed (`d8048283`): embedding/lm_head, SwiGLU MLP, GQA-SDPA
  matmuls are kt-internal (gated, conservative).

## Two source facts that reframe the work

1. **The candle-autograd "island" (`Var`/`Var::from_tensor`/`.backward()`/`.as_tensor()`)
   is TEST-ONLY.** It lives entirely in `forward.rs`'s `#[cfg(test)] mod tests`
   (block at ~26659; `Var` uses 31420–31841). Production `forward.rs` has **zero**
   `Var`, **zero** real `.backward()` calls (the 8 greps are doc-comments), **zero**
   `.as_tensor()`. The test mod stays candle (`use candle_core::…Var` locally) and is
   not part of the production flip.
2. **`model_forward_kt` (23449) has ZERO production callers.** Every consumer
   (`generate.rs`, `server/bench.rs`, `trainer.rs`, `opd.rs`, `speculative.rs`) calls
   the candle `model_forward` (23308) and expects a `candle_core::Tensor`. The public
   boundary is candle on **both** inference and training sides.

So the real production blockers are: the **`CustomOp{1,2,3}` + `track_op()` +
`storage_and_layout()` cluster**, the **candle-typed `model_forward` signature**, the
**`GpuWeights` bare-`Tensor` fields**, and — the deepest one — the **candle `GradStore`
grad-delivery plumbing** the optimizer still reads.

## ⚠️ The flip is gated on a kt-native training substrate (the true long pole)

Even the **default-on tape-authoritative** path still calls candle `loss.backward()`
on the *detached* loss to obtain a `GradStore`, then inserts the kt-tape grads keyed by
each LoRA `Var`'s candle `TensorId` (`trainer.rs:10756–10783`). The kt-native
`Tape::backward → kt-GradStore` substrate exists (`tape_step.rs`) but is **not wired to
production `Parameter`s**. The `KILN_USE_TAPE_AUTHORITATIVE=0` opt-out **and the CPU
device path** (forced regardless of the flag, device-gate at 10809) are pure candle
`loss.backward()`.

**Consequence:** flipping `forward.rs` to kt severs the candle autograd graph that
`model_forward`'s candle-typed return feeds into — which breaks *both* the
candle-authoritative opt-out *and* the current tape path's GradStore harvest. Therefore
the atomic flip (Increment 4) **cannot land** until the training side is fully
kt-native (Increment 0 below). DELETE-the-candle-path is **not yet safe**; ISOLATE is.

This is why removing candle from kiln-model/kiln-train is a *training-substrate* project
as much as a *tensor-type* one.

## Increment sequence

Classification: **[PREP]** still candle, compiles, no behavior change ·
**[SUBSTRATE]** training-side kt-native wiring · **[ATOMIC]** the coupled type-flip ·
**[DROP]** removes dep/vendor.

### Increment 0 — [SUBSTRATE] kt-native `Tape::backward → GradStore → Parameter` (THE prerequisite)
Wire the kt-native tape backward to deliver grads into a kt-keyed `GradStore` consumed
by the optimizer, removing the candle `loss.backward()` GradStore harvest at
`trainer.rs:10756–10783`. Then remove the `KILN_USE_TAPE_AUTHORITATIVE=0` opt-out and
give the CPU training path a kt route (or CUDA-gate it). Files: `trainer.rs:10697–10874`,
`tape_step.rs`, `cd_types.rs` (GradStore/Var aliases), `kiln-autograd`. **Largest +
highest-design step; gates everything below.** Validate: CP-4 100-step loss+Adam-moment
convergence + finite-diff grad gate, fully kt (no candle backward in the loop).

### Increment 1 — [PREP] Migrate the 30 `track_op()` guards to the tape/bridge predicate
`forward.rs` — centralize through `any_tensor_tracks_op` (2233); rewrite the ~30
production `!x.track_op()` fast-path guards (94, 2180, 2234, 2709, 3573, 4024-4121,
7772-7841, 10948-11071, 11330, 11538-11701, 12588, 16215, 16328, 17878, 18224, 22312)
to the `tape_recording_active = tape_forward_enabled() && bridge_scope_active()` template
already shipped at 16210. **DO this only after Increment 0** — while the candle-auth path
exists, `track_op()` legitimately disables fast-paths under candle training, and a naive
replacement would fire fast-paths on that path. Once candle-auth is gone, `track_op()`
is dead. Validate: CP-4 parity gate (this changes fast-path selection).

### Increment 2 — [PREP] Un-alias the 7 CUDA + 1 Vulkan `CustomOp` impls to explicit `candle_core::Tensor`
`forward.rs` — inside `CudaLoraAddF32` (2958), `CudaLoraLinearBf16` (3080),
`CudaLoraAddBf16` (3331), `CudaSigmoidMulTrainingBf16` (3674),
`CudaFlashAttentionTrainingBf16` (4520), `CudaRotaryOneBf16` (10683), `VulkanRmsNormOp`
(8031) and their `apply_op*` wrappers (2750, 2852, 2946, 3075, 3664, 4500, 8218, 8882),
spell every `Tensor` as `candle_core::Tensor`. Pure type-spelling; isolates them from the
alias so the flip skips them. (These get DELETED in a later phase once candle-auth is
fully gone; ISOLATE keeps them compiling meanwhile.) Validate: compiles under
cuda+vulkan; training smoke.

### Increment 3 — [PREP] Add `GpuLinearAttentionWeights::*_kt` accessors + bridge the 48 BackendRuntime call sites
`GpuFullAttentionWeights`/`GpuFfnWeights` already have `*_t_kt` accessors (5105–5346);
**`GpuLinearAttentionWeights` does not** — add `in_proj_*_t_kt`/`out_proj_t_kt`/`conv1d_kt`
(mirror the 5105 borrow pattern). The `BackendRuntime` trait (`backend/mod.rs`) is fully
`candle_core::Tensor`-qualified (235 refs, NO alias) so the flip does **not** touch it;
the 48 forward.rs call sites that cross the seam just need `kt_tensor_to_candle_cuda_copy`
/ `_borrow` adapters (primitives already used 219× in forward.rs). Validate: decode +
marlin + resident parity.

### Increment 4 — [ATOMIC] Flip the alias + GpuWeights fields + cd_types + `model_forward` signature (single commit)
`forward.rs:25/57` (drop `Tensor` from the candle `use`; promote `kiln_tensor::Tensor` to
bare `Tensor`); GpuWeights/MtpGpuWeights/GpuFullAttentionWeights/GpuLinearAttentionWeights/
GpuFfnWeights fields → kt; `model_forward` (23308) internals → kt (`model_forward_kt`
becomes identity); `cd_types.rs:30` aliases + their `vk_train.rs:1683+` field reads →
kt, lockstep. **Mutually type-dependent — cannot be split.** Test mod stays candle.
**Single highest-risk step.** Validate: full `cargo nextest -p kiln-model -p kiln-train
--features cuda` + tape_forward_parity + CP-4 parity + generate/server e2e inference parity.

### Increment 5 — [ATOMIC] Propagate kt through `generate.rs`/`server`/`speculative.rs`
Flip their `model_forward` consumption to kt (or keep a candle-returning shim and defer).
Validate: server smoke + inference parity.

### Increment 6 — [ATOMIC] Flip the `BackendRuntime` trait + 4 backends; remove Increment-3 bridges
`backend/mod.rs` (41 methods) + cpu/cuda/metal/vulkan + vulkan_linear_op.rs/vulkan_lora_op.rs.
Validate: per-backend parity (vk resident decode, CUDA decode, metal CI).

### Increment 7 — [DROP] Remove candle from kiln-model + kiln-train Cargo.toml; delete vendor
Gated on Increment 0 (no residual candle `Var`/`backward`/`GradStore`). Delete the candle
CustomOp impls + test-mod oracles' candle (or keep a kt oracle). Then
`rm -rf vendor/candle-core`; `cargo tree -i candle-core` empty. Validate: full suite all backends.

## Bottom line

- **Realistic increments: 8** (0–7). Increment **0** (kt-native training substrate) is the
  true long pole and gates the flip; Increment **4** (alias + GpuWeights + cd_types +
  model_forward, ~1173 sites) is the coupled-atomic highest-risk type change.
- Increments **2 + 3** are safe, behavior-preserving PREP that can land anytime
  (independent of Increment 0).
- The autograd "island" is test-only; the production gate is Increment 0's GradStore
  wiring + the 30 `track_op()` guards (Increment 1), for which CP-4 already shipped the
  replacement pattern.

**Key files:** `crates/kiln-model/src/forward.rs` (25, 57, 2233, 2958–10727, 4758–5346,
16210, 23308–23474), `crates/kiln-model/src/backend/mod.rs`, `crates/kiln-train/src/{cd_types.rs,
trainer.rs (10697–10874), vk_train.rs (1683+), tape_step.rs}`, `crates/kiln-kt-bridge/src/{lib.rs,
tape_bridge.rs (bridge_scope_active)}`, `crates/kiln-autograd/`.
