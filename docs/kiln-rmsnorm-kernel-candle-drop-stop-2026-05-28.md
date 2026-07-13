# `kiln-rmsnorm-kernel` candle-core dep removal — STOP (audit 2026-05-28)

> **Historical snapshot, not current operating guidance.** This document records
> migration state from May 2026. The `KILN_USE_TAPE_*` and
> `KILN_USE_TAPE_AUTHORITATIVE` switches mentioned below were removed without
> aliases or replacement fields. Current GPU training uses an internal tape
> scope as its sole routing authority. See [Configuration](./CONFIGURATION.md)
> and [Native SFT Profile](./NATIVE_SFT_PROFILE.md) for current behavior.

## TL;DR

The `kiln-rmsnorm-kernel` crate **cannot** drop `candle-core` /
`candle-nn` to `[dev-dependencies]` at HEAD (`main@3a1f8d02`). The
sibling rmsnorm-kt-tape pilot from `895162ca` carved out exactly **one**
kt-typed entry (`fused_rmsnorm_via_kt_tape`) inside the crate, but the
other ~15 public functions in `lib.rs` are still candle-typed and serve
as the production CUDA backend for `kiln-model`'s candle-typed
training/inference graph.

`grep -rn "candle_core\|candle_nn" crates/kiln-rmsnorm-kernel/src/`
returns **151 hits** across the four source files. The distribution
makes the blocker obvious:

| File | candle refs | Status |
|------|------------:|--------|
| `src/lib.rs` | 132 | **All in production code.** Zero `#[cfg(test)]` boundary in the file. 15 candle-typed `pub fn`s. |
| `src/kt_forward_op.rs` | 18 | **Production shim.** `fused_rmsnorm_via_kt_forward_op` is the live caller from `kiln-model::forward::rms_norm`. |
| `src/kt_tape.rs` | 1 | Doc comment only. The function body is fully kt-typed (pilot from `895162ca`). |
| `src/kt_api.rs` | 0 | Fully kt-typed. The Phase-7 endpoint shape. |
| **total** | **151** | |

The pilot already moved the cleanest entry (rmsnorm with autograd) onto
the kt-tape; the architectural blocker for moving the rest is the
**same CP-4 substrate** documented in:

* [`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
* [`opd-loss-kernel-candle-removal-stop-2026-05-28.md`](./opd-loss-kernel-candle-removal-stop-2026-05-28.md)

This STOP records the dep-level audit so future planning cycles don't
re-queue the dev-deps move from stale assumptions about how much of
the crate is already kt-typed.

## What the resume task framed

> "Evaluate whether `kiln-rmsnorm-kernel`'s `candle-core` and
> `candle-nn` deps can be moved to `[dev-dependencies]` (or dropped
> entirely). The kt_tape.rs pilot already exists (commit 895162ca); we
> need to know what production candle refs remain."
>
> "**OUTCOME A: Most candle refs are bridge/production, can't drop yet**
> — Write a STOP doc describing total ref count by file, the pattern of
> who calls these functions from kiln-model, what's needed to migrate,
> and reference what the OPD-loss-kernel STOP doc identifies for the
> same architectural blocker."
>
> "**OUTCOME B: Most production refs already through kt_tape.rs /
> KtForwardOp shim** — try moving candle to dev-deps and verify on
> pod."

The audit at `main@3a1f8d02` lands squarely on **Outcome A**. 132 of
151 candle refs are in `lib.rs` production code, with 23 candle-typed
call sites in `kiln-model` / `kiln-train` consuming the public API.
Moving the deps to `[dev-dependencies]` would not compile in either
the kernel crate or any of its consumers.

## Audit at HEAD (`main@3a1f8d02`)

### 1. Crate's current `[dependencies]`

`crates/kiln-rmsnorm-kernel/Cargo.toml`:

```toml
[dependencies]
candle-core = { workspace = true, features = ["cuda"] }
half = "2"
# Phase 7 prep — kiln-tensor-typed surface alongside candle-typed.
kiln-tensor = { workspace = true, features = ["cuda"] }
kiln-kt-bridge = { workspace = true, features = ["cuda"] }
# Phase 6a/CP-4 (#1082): kt-tape RMSNorm shim records onto kiln_autograd::Tape.
kiln-autograd = { workspace = true }
```

`candle-nn` is **already not** a direct `[dependencies]` entry — only
`candle-core`. The two `candle-nn` mentions in this STOP's TL;DR refer
to the task framing ("candle-core and candle-nn"); the only candle
crate to evaluate dropping from this crate is `candle-core`.

### 2. lib.rs — 15 candle-typed `pub fn`s in production code

`grep -n "^pub fn" crates/kiln-rmsnorm-kernel/src/lib.rs` gives:

```
409: pub fn supports(x: &Tensor, weight: &Tensor) -> bool
421: pub fn rotary_one_bf16_storage(...)
530: pub fn supports_rotary_one_bwd_bf16(...)
568: pub fn rotary_one_bwd_bf16(...)
753: pub fn supports_sigmoid_mul(x: &Tensor, gate: &Tensor) -> bool
768: pub fn fused_sigmoid_mul_storage(...)
862: pub fn matmul_f32_bf16w(lhs: &Tensor, weight: &Tensor) -> Result<Tensor>
889: pub fn matmul_f32_bf16w_bwd_lhs(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor>
947: pub fn lora_add_inplace_f32_storage(...)
1037: pub fn lora_add_bf16_storage(...)
1135: pub fn causal_depthwise_conv1d_f32(...)
1257: pub fn causal_depthwise_conv1d_f32_inplace(...)
1378: pub fn causal_depthwise_conv1d_f32_bwd_input(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor>
1479: pub fn causal_depthwise_conv1d_f32_bwd_weight(...)
1605: pub fn causal_depthwise_conv1d_f32_bwd_state(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor>
```

All 15 use the candle-typed top imports at line 77:

```rust
use candle_core::{
    CudaStorage, DType, Device, Layout, Result, Tensor, backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
};
```

There is **no `#[cfg(test)]` module boundary in lib.rs**. Every one of
the 132 candle refs is reachable in `cargo build` (non-test) builds.
Confirmed via `grep -n "cfg(test)\|mod tests\|#\[test\]"
crates/kiln-rmsnorm-kernel/src/lib.rs` returning only a single doc
comment match (line 1699).

### 3. kt_forward_op.rs — production shim wraps the candle-typed CustomOp

`crates/kiln-rmsnorm-kernel/src/kt_forward_op.rs:117`:

```rust
pub fn fused_rmsnorm_via_kt_forward_op(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    ...
    cuda_via_kt_forward_op(x, weight, eps)
}
```

This is the **live production caller** wired from
`kiln-model::forward::rms_norm` at `crates/kiln-model/src/forward.rs:7278`:

```rust
return kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_forward_op(x, weight, eps as f32)
    .context("fused_rmsnorm_via_kt_forward_op shim failed");
```

Internally it constructs a `KtForwardOp2` from `kiln-kt-bridge` over
the kt-typed forward+backward (`fused_rmsnorm_kt` /
`fused_rmsnorm_backward_kt`), but the **outer signature is
candle-typed** so the candle autograd graph picks up the result for
`loss.backward()`. The 18 candle refs in this file are:

* The `use candle_core::{Tensor, Result, ...}` imports.
* `candle_core::bail!(...)` for out-of-envelope errors.
* The `&Tensor` / `Result<Tensor>` signatures of the public function.
* The `KtForwardOp2` constructor consuming candle tensors via
  `kiln-kt-bridge` (which itself is a candle ↔ kt bridge — see
  `kt_tensor_from_candle_cuda_borrow` / `kt_tensor_to_candle_cuda_copy`).

### 4. kt_tape.rs — already candle-free in code, one doc comment

`crates/kiln-rmsnorm-kernel/src/kt_tape.rs:249`:

```rust
pub fn fused_rmsnorm_via_kt_tape(
    x: &KtTensor,
    weight: &KtTensor,
    eps: f32,
    tape: &mut Tape,
) -> KtResult<KtTensor>
```

Pure kt-typed. The single "candle" hit in this file is a doc comment
inside the function-level docstring (line 244) referencing
`candle_core::backprop::GradStore` as part of the explanation for why
the production caller hasn't flipped to this entry yet (see the
companion STOP `rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`
for the CP-4 substrate blocker).

The `#[cfg(test)] mod tests` block at line 283 also imports `KtTensor`
through `super::*` — no candle. The pilot is exactly what its commit
message claimed: a parallel kt-shim that mirrors the kt-forward-op
behavior on the kt side without touching the candle path.

### 5. kt_api.rs — zero candle refs

`crates/kiln-rmsnorm-kernel/src/kt_api.rs` is the kt-typed surface
where the FFI symbols are wrapped. 1911 lines, **0** candle hits. This
is the Phase-7 endpoint shape — the rest of the crate is supposed to
look like this once kiln-model migrates off candle.

### 6. Consumer crates — 23 candle-typed call sites

`grep -rn "kiln_rmsnorm_kernel::" crates/kiln-model/src/ crates/kiln-train/src/`
filtered to non-comment, non-`_kt` lines returns 23 call sites:

| Path | Symbol | Type |
|------|--------|------|
| `kiln-model/src/cuda_train.rs:1277,1351,2089` | `matmul_f32_bf16w` | candle |
| `kiln-model/src/cuda_train.rs:5544,5596` | `matmul_f32_bf16w_bwd_lhs` | candle |
| `kiln-model/src/cuda_train.rs:2909` | `causal_depthwise_conv1d_f32` | candle |
| `kiln-model/src/cuda_train.rs:3011` | `causal_depthwise_conv1d_f32_inplace` | candle |
| `kiln-model/src/cuda_train.rs:5676,5735` | `causal_depthwise_conv1d_f32_bwd_input` | candle |
| `kiln-model/src/cuda_train.rs:5685` | `causal_depthwise_conv1d_f32_bwd_weight` | candle |
| `kiln-model/src/cuda_train.rs:5698` | `causal_depthwise_conv1d_f32_bwd_state` | candle |
| `kiln-model/src/forward.rs:2902` | `lora_add_inplace_f32_storage` | candle |
| `kiln-model/src/forward.rs:3063,3278` | `lora_add_bf16_storage` | candle |
| `kiln-model/src/forward.rs:3524` | `supports_sigmoid_mul` | candle |
| `kiln-model/src/forward.rs:3573` | `fused_sigmoid_mul_storage` | candle |
| `kiln-model/src/forward.rs:7251` | `supports` | candle |
| `kiln-model/src/forward.rs:7278` | `fused_rmsnorm_via_kt_forward_op` | candle (shim) |
| `kiln-model/src/forward.rs:10190` | `rotary_one_bf16_storage` | candle |
| `kiln-model/src/forward.rs:10214,32136` | `supports_rotary_one_bwd_bf16` | candle |
| `kiln-model/src/forward.rs:10241,32141` | `rotary_one_bwd_bf16` | candle |

For comparison, the kt-typed call sites (`*_kt`) from the same grep
returns 39 hits. The candle-typed surface is smaller than the kt-typed
surface in raw call count — but it's not zero, and every one of those
23 call sites lives inside a function whose own signature already
takes/returns candle `Tensor`s. The production training and inference
graphs in `kiln-model::forward` and `kiln-model::cuda_train` are
candle-native; the kernel crate's candle entries are the
launch/backward primitives those graphs invoke. There is no way to
drop the dep without simultaneously migrating those graphs.

### 7. Consumer crates' own candle ↔ kt seam

The `tape_forward.rs` adapter introduced in wave-12 / `deed13a8` is
the model's current bridge between the candle graph and the kt-tape:

`crates/kiln-model/src/tape_forward.rs:140-148`:

```rust
if !kiln_rmsnorm_kernel::supports_rmsnorm_kt(&x_kt, &w_kt) {
    ...
}
kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape(&x_kt, &w_kt, eps, tape)
```

This calls the **kt-typed** pilot. The candle tensors are borrowed
into kt via `kt_tensor_from_candle_cuda_borrow` before the kt
function is invoked. This adapter is wired into `try_tape_rms_norm_cuda`
but, per the sibling STOP, is only reached when both
`KILN_USE_TAPE_FORWARD=1` and a thread-local `Tape` scope are active —
neither of which is the default training-loop configuration today
(the trainer still drives `loss.backward()`).

So the kt-tape pilot is **wired but not load-bearing** at HEAD. The
production rms_norm gradient still flows through the candle-typed
`fused_rmsnorm_via_kt_forward_op` shim, which in turn requires
`candle-core` to live in `[dependencies]`.

## What a hypothetical "candle → dev-deps" move would break

A trial run of moving `candle-core` to `[dev-dependencies]` in
`crates/kiln-rmsnorm-kernel/Cargo.toml` would fail at compile time in
**both** the kernel crate and every consumer:

1. **Kernel crate fails to compile.** `lib.rs:77` imports `candle_core::{...}`
   at module scope. `kt_forward_op.rs` does the same. Dev-deps are not
   visible to non-test compilation units, so `cargo build` of the
   crate itself stops at unresolved-import.
2. **kiln-model fails to compile.** 23 candle-typed call sites in
   `forward.rs` / `cuda_train.rs` resolve to functions whose signatures
   live in `lib.rs` (e.g. `matmul_f32_bf16w(&Tensor, &Tensor) ->
   Result<Tensor>`). When the kernel crate refuses to expose those
   signatures, the consumer's imports break first.
3. **kiln-train fails transitively.** `kiln-train` depends on
   `kiln-model`, so even though it has zero direct candle-typed
   call sites against `kiln_rmsnorm_kernel`, the upstream break
   propagates.
4. **The kt-only path is still gated.** `try_tape_rms_norm_cuda` (the
   only kiln-model entry that already routes through the kt-tape
   pilot) is opt-in behind `KILN_USE_TAPE_FORWARD` + thread-local tape
   scope. Dropping the candle path would force every training run to
   set those env vars and refactor `loss.backward()` first — that is
   the CP-4 substrate work, not a Cargo.toml edit.

There is no `legacy-candle` feature-flag escape hatch that would help
either, because the consumer call sites are unconditional. A feature
flag could only help if `kiln-model` already had a kt-typed parallel
signature for each of the 15 candle-typed `pub fn`s and could
conditionally pick the kt one — which is exactly the work that
hasn't happened yet.

## What unblocks the dev-deps move

The same CP-4 substrate that unblocks the rmsnorm production-caller
flip and the OPD-loss-kernel candle-core drop unblocks this one too.
Sequencing:

1. **CP-4 substrate lands in `kiln-train`** — `kiln_autograd::Var`
   wraps `kiln_tensor::Tensor`, the per-step training loop opens a
   `with_thread_local_tape` scope, and `loss.backward()` is replaced
   by `tape.backward(loss_id, seed, accumulator)`.
2. **kiln-model production paths port off candle.** Each of the 23
   candle-typed call sites against `kiln_rmsnorm_kernel` needs a
   kt-typed sibling in `kiln-model` whose signature is `KtTensor`
   in, `KtTensor` out. The kt-typed sibling already exists in
   `kt_api.rs` for the launch-only entries (e.g. `matmul_f32_bf16w_kt`
   does not exist yet; equivalents like `fused_mlp_silu_mul_packed_kt`
   do). Sites that take/return `candle_core::Storage` would need
   richer kt counterparts; the kt API today doesn't expose the
   non-canonical-tensor storage path.
3. **`kt_forward_op.rs` deletes** in the same PR that flips
   `kiln-model::forward::rms_norm` from `fused_rmsnorm_via_kt_forward_op`
   to a `Tape`-aware call through the kt pilot.
4. **lib.rs slims to launch wrappers only.** The candle-typed
   `pub fn`s either get kt-typed siblings (preferred) or get
   deleted along with their consumers in kiln-model.
5. **At that point, `candle-core` exits `[dependencies]` and
   becomes a `[dev-dependencies]` entry** for any remaining
   candle-typed parity tests / benches. If those tests are also
   migrated to kt, candle-core leaves the crate entirely.

This sequencing matches the existing roadmap in
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) §"Top 3
next-tasks" #1 (CP-4 substrate) and the parallel STOPs for the
rmsnorm production-caller flip and the OPD-loss-kernel dep drop.

## What this commit ships

Pure documentation STOP. No code changes in `crates/kiln-rmsnorm-kernel/`.

The kt-tape pilot remains parallel-shim posture; the kt-typed surface
in `kt_api.rs` continues to grow alongside the candle-typed launch
wrappers in `lib.rs` as wave-12/wave-13 migrate more model paths.

## Cross-references

* [`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
  — production-caller flip blocker (same CP-4 substrate gate).
* [`opd-loss-kernel-candle-removal-stop-2026-05-28.md`](./opd-loss-kernel-candle-removal-stop-2026-05-28.md)
  — sibling crate dep-drop blocker (identical pattern: kt pilot exists,
  candle CustomOp still production, dep stays).
* [`kiln-server-candle-removal-stop-2026-05-27.md`](./kiln-server-candle-removal-stop-2026-05-27.md)
  — same "blocked by substrate" shape one layer up the stack.
* [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) §"kt-autograd
  readiness" and CP-4.
* Pilot commit: `895162ca` — "kiln-rmsnorm-kernel: pilot kt-tape
  backward port for fused RMSNorm (#1082)".
* HEAD at audit: `3a1f8d02`.
