# `kiln-model::forward::rms_norm` kt-tape production-caller flip — STOP (audit 2026-05-28)

## TL;DR

The pilot in `895162ca` added a parallel `fused_rmsnorm_via_kt_tape(x,
weight, eps, &mut Tape)` entry alongside the existing
`fused_rmsnorm_via_kt_forward_op(x, weight, eps)` shim. The pilot
commit message explicitly punted the production-caller flip
("production-caller migration lands once the wider kiln-model autograd
substrate adopts kt-tape — keeping the two paths in parallel for now
matches the 'parallel shim, flip when ready' rollout cadence").

This STOP records the resume-task audit at `main@20a5885c` that
confirms the flip **cannot land at the `rms_norm` call site itself**.
The production caller is `kiln-model::forward::rms_norm(x: &Tensor,
weight: &Tensor, eps: f64) -> Result<Tensor>` (candle `Tensor` in,
candle `Tensor` out) with no `&mut Tape` in scope and no `kiln_autograd`
dependency in the crate. The kt-tape entry requires `&mut Tape` because
it `tape.record(...)` a `BackwardOp` for `Tape::backward` to execute
later.

The architectural gap is **CP-4 — kt-typed autograd `Var`/`Tape`
adoption substrate**, already named in
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) as the highest-
leverage remaining item ("multi-PR substrate work
(`kiln_tensor::Tensor::tape_id`, `kiln_autograd::Var`, cross-crate
parity tests for ≥1k training steps)"). Pure-STOP is the right outcome
here. No `kiln-model` code changes in this commit.

## What the resume task framed

> "Investigate whether the production rms_norm caller can be flipped to
> use `kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape(...)` instead of
> `fused_rmsnorm_via_kt_forward_op(...)`. The surrounding caller would
> need a `&mut Tape` in scope. … Even partial: adding a Tape parameter
> to one forward function is progress."

The framing's "even partial" hope is what this STOP investigates. The
audit below shows that adding a `&mut Tape` parameter to `rms_norm` is
not progress — it's a vacuous change that compiles but has no caller
willing to thread the parameter through (because no caller has a Tape
either, and so on transitively all the way up to the training-step
boundary in `kiln-train::trainer`). The actual blocker is structural:
the training loop is built on `loss.backward()` via
`candle_core::backprop::GradStore`, not on `Tape::backward`.

## Audit at HEAD (`main@20a5885c`)

### 1. The pilot entry signature

`crates/kiln-rmsnorm-kernel/src/kt_tape.rs:237`:

```rust
pub fn fused_rmsnorm_via_kt_tape(
    x: &KtTensor,
    weight: &KtTensor,
    eps: f32,
    tape: &mut Tape,
) -> KtResult<KtTensor>
```

* Inputs and output are `kiln_tensor::Tensor` (alias `KtTensor`), not
  `candle_core::Tensor`.
* Records a `CudaFusedRmsNormBackward` node on `&mut Tape` so a later
  `Tape::backward(loss_id, seed_grad, accumulator)` call can walk it.
* Returns an `Err` outside the kt envelope — the pilot expects the
  caller to pre-check via `supports_rmsnorm_kt(x, weight)` exactly like
  `fused_rmsnorm_via_kt_forward_op` does.

### 2. The production caller

`crates/kiln-model/src/forward.rs:7135`:

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let kernel_disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        let bwd_disabled = std::env::var("KILN_DISABLE_RMSNORM_BACKWARD").is_ok();
        if !kernel_disabled && !bwd_disabled {
            if !x.track_op() && !weight.track_op() {
                // … forward-only kt path (inference) …
                return kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt) /* … */;
            }
            if should_use_fused_rmsnorm() && kiln_rmsnorm_kernel::supports(x, weight) {
                // Production caller — currently routes through the kt-forward-op shim.
                return kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_forward_op(x, weight, eps as f32)
                    .context("fused_rmsnorm_via_kt_forward_op shim failed");
            }
        }
    }
    // metal / vulkan / fallback paths …
    rms_norm_fallback(x, weight, eps)
}
```

* Signature is `(x: &candle_core::Tensor, weight: &candle_core::Tensor,
  eps: f64) -> Result<candle_core::Tensor>`.
* Returns a candle `Tensor` whose autograd lineage is consumed by the
  caller's eventual `loss.backward()` via `candle_core::backprop`.
* No `&mut Tape` parameter, no `Tape` value in scope, no
  `kiln_autograd::Tape` import.

`grep -rln "kiln_autograd\|kiln-autograd" crates/kiln-model/` returns
**zero files**. The `kiln-model` crate `Cargo.toml` has no dependency
on `kiln-autograd` and no transitive path to it.

### 3. The caller's callers — 20+ candle-typed call sites

`grep -n "\brms_norm(" crates/kiln-model/src/forward.rs` shows the
20+ production call sites:

```
7135: pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
10296:        rms_norm(hidden, &layer.post_attention_layernorm, config.rms_norm_eps)?
11882:    let normed = rms_norm(hidden, &weights.final_norm, config.rms_norm_eps)?;
13634:        rms_norm(hidden, &layer.input_layernorm, config.rms_norm_eps)?
13656:    rms_norm(hidden, &layer.input_layernorm, config.rms_norm_eps)
16239:    let q = rms_norm(&q, &attn_weights.q_norm, rms_norm_eps)?;
16280:    let k = rms_norm(&k, &attn_weights.k_norm, rms_norm_eps)?;
16387,16388,16522,16582,16791,16792,18271,18272,18905,18906,19974: …
```

Every one of these is inside a function whose own signature takes
candle `Tensor`s (e.g. `attention_block`, `pre_fc_norm_block`,
`mlp_block`, `final_norm`). None of them have a `&mut Tape`.

Threading a Tape through `rms_norm` therefore implies threading a Tape
through *every transitive caller* up to the training step root in
`kiln-train::trainer`. That is the CP-4 substrate work, not a "one
forward function" tweak.

### 4. The training loop is built on `loss.backward()`

`crates/kiln-train/src/trainer.rs:4781`:

```rust
let grads = loss.backward().context("GRPO+ECHO backward pass")?;
```

`grep -n "loss\.backward\|backprop::GradStore" crates/kiln-train/src/trainer.rs`
returns ~10 sites, all flowing through `candle_core::backprop::GradStore`
— the candle-side autograd tape that `BackpropOp` nodes register into
when a candle op runs with at least one `track_op() == true` input.

`kiln_autograd::Tape::backward(loss_id, seed_grad, accumulator)` is a
completely separate graph datastructure. It cannot drive
`loss.backward()`'s walk, and `loss.backward()` cannot consume nodes
recorded on a `kiln_autograd::Tape`. The two autograd worlds are
disjoint until CP-4 lands a `Var` substrate that bridges them.

This is **the documented Phase 6a/CP-4 blocker** in
`CANDLE_REMOVAL_PLAN.md` line 209-218:

> "**CP-4 — kt-typed autograd `Var` / `Tape` adoption substrate.** Now
> the highest-leverage remaining item. The `KtForwardOp` shim is a
> stop-gap that wraps fused kt forward+backward inside candle's
> autograd graph; converting the rest of the training loop (`Var` /
> `loss.backward()` → `kiln_autograd::Tape::backward`) is the
> precondition for Tier-4 (kt-bridge deletion). Already sketched in
> `CANDLE_REMOVAL_PLAN.md` §'kt-autograd autograd-interop blocker
> (2026-05-27)' — multi-PR substrate work (`kiln_tensor::Tensor::tape_id`,
> `kiln_autograd::Var`, cross-crate parity tests for ≥1k training
> steps)."

### 5. The pilot's own commit message agrees

From `895162ca`'s message:

> "The production caller in `kiln-model::forward::rms_norm` still uses
> the candle-CustomOp path (`fused_rmsnorm_via_kt_forward_op`). The
> production-caller migration lands once the wider kiln-model autograd
> substrate adopts kt-tape — keeping the two paths in parallel for now
> matches the 'parallel shim, flip when ready' rollout cadence the
> issue authorises."

The pilot author explicitly identified the same gap. This audit
confirms it has not closed in the ~24 hours since the pilot landed.

### 6. Same pattern across the other Phase 6a pilots

The companion FLCE kt-tape pilot (`5a78a0ef`) hit the analogous wall:
follow-up commit `f83ec4c1` had to mark the two CUDA-gated E2E tests
`#[ignore]` for a separate "kt-substrate gap" (index-op constructors
honouring parent device) before the test surface compiled cleanly.
The OPD kt-tape port (`5478e64f`) landed without a production-caller
flip for the same reason — `kiln-train::opd` consumes
`opd_top_k_reverse_kl_phase_a_per_position` (candle-typed) and feeds
the result into `loss.backward()`.

This is a consistent pattern: every Phase 6a/CP-4 kt-tape pilot lands a
parallel shim and leaves the production caller on the candle-side
`CustomOp` route. The next-leg work in *all three* (rmsnorm, FLCE, OPD)
is the same kt-autograd `Var`/`Tape` substrate, not a per-kernel flip.

## What "even partial" would look like — and why it isn't progress

Hypothetical "partial" patch: change

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor>
```

to

```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64, tape: &mut Tape) -> Result<Tensor>
```

and route the CUDA + autograd-tracked branch through
`fused_rmsnorm_via_kt_tape(x_kt, w_kt, eps as f32, tape)` after
borrowing both inputs through `kt_tensor_from_candle_cuda_borrow`,
then `kt_tensor_to_candle_cuda_copy` on the way out.

Why this is **not** progress:

1. **It pollutes 20+ call sites with a parameter no caller is willing
   to construct.** Every transitive caller would have to either accept
   a `&mut Tape` itself (cascading further up) or instantiate a
   throw-away local `Tape`. A throw-away local `Tape` is a no-op
   because `Tape::backward` is never called on it — the gradient still
   flows through candle's `BackpropOp` chain on the returned candle
   tensor (which is what consumes the gradient in
   `loss.backward()`).
2. **The recorded `BackwardOp` node would never execute.** A `Tape`
   only matters when `Tape::backward(loss_id, …)` is later called. The
   training loop currently calls `candle_core::Tensor::backward`, which
   walks candle's `BackpropOp` graph — it has no path to a
   `kiln_autograd::Tape`. So a `Tape::record` inside `rms_norm` would
   accumulate an orphan node that gets dropped at function-return time
   if the Tape is local, or accumulates indefinitely if the Tape is
   passed in. Either way, the actual backward executes via the
   existing `KtForwardOp2` CustomOp path attached to the returned
   candle tensor.
3. **It bit-rots faster than it ages.** The kt-tape entry insists on
   the CUDA + BF16 + contiguous + hidden ≤ 8192 envelope. Out-of-
   envelope cases would still need to fall back to
   `fused_rmsnorm_via_kt_forward_op` or `rms_norm_fallback`. The
   resulting double-dispatch chain (kt-tape attempt → kt-forward-op
   fallback → candle fallback) is more code, more env-var matrices,
   and more parity surface to maintain — for zero numerical benefit
   (same FFI symbols on the happy path).

A flip that adds parameters without changing the gradient path is
substrate work theatre. The pilot was correct to stop where it
stopped.

## What unblocks this

The flip becomes safe and meaningful when **one** of these two
substrate-level changes lands:

### Option A — Full `Var`/`Tape` adoption in kiln-train (the documented CP-4 path)

* `kiln_autograd::Var` wraps a `kiln_tensor::Tensor` and carries a
  `tape_id` plus a `Arc<RefCell<Tape>>` handle.
* `kiln-train::trainer` ports every per-step forward from
  `candle_core::Tensor` to `kiln_autograd::Var` (or a sibling type
  that boxes the candle/kt seam during the migration window).
* `loss.backward()` is replaced by `tape.backward(loss_id, seed_grad,
  …)`. The accumulator passed in is
  `kiln_tensor::ops::add` (or a wrapper).
* Every kt-tape pilot's production caller (rmsnorm, FLCE, OPD, future
  GDN/Conv1d/FlashAttn ports) flips at the same time, because each is
  now driven by a `&mut Tape` borrowed from the per-step training
  loop.

This is the multi-PR substrate work named in
`CANDLE_REMOVAL_PLAN.md` line 209-218.

### Option B — Bridge `kiln_autograd::Tape` into candle's `BackpropOp` (the stop-gap shim)

* New `kiln-kt-bridge::TapeBackpropOp` wraps a `&mut Tape` and a
  candle `BackpropOp` so that `loss.backward()` walks
  `Tape::backward` as a side-effect of walking the candle graph.
* This is structurally what `KtForwardOp{1,2,3}` already does in the
  opposite direction (candle CustomOp wrapping a kt-typed backward).
  Doing it both ways simultaneously is a complexity blowup and was
  rejected as a path in `CANDLE_REMOVAL_PLAN.md` line 1172:
  > "There is no need for kt-autograd's `Tape` to drive candle's
  > backprop … `Tape`/`BackwardOp` replace the candle
  > `Op::CustomOp{1,2,3}`."
* Listed for completeness only. **Do not implement Option B.**

### Why a tape-aware intermediate wrapper inside `rms_norm` is also not the answer

A hypothetical `rms_norm_with_tape(x: &Tensor, weight: &Tensor, eps:
f64, tape: &mut Tape) -> Result<Tensor>` that runs the kt-tape forward,
records onto `tape`, then exits — same problem. The tape is local to
the caller's frame, the candle return value carries no kt-tape lineage,
and the caller's `loss.backward()` still drives candle's `GradStore`.
The tape parameter becomes dead weight.

## Decision

**STOP.** No code change in this commit. The pilot's parallel-shim
posture is correct and well-documented; this STOP records the resume-
task audit that confirms the flip is gated on CP-4 substrate and not
something a one-function patch can unblock.

When CP-4 lands (`kiln_autograd::Var` substrate + `kiln-train::trainer`
port off `loss.backward()`), the flip becomes a one-line change at
`forward.rs:7172`:

```rust
// before
return kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_forward_op(x, weight, eps as f32)
    .context("fused_rmsnorm_via_kt_forward_op shim failed");

// after (post-CP-4)
return kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape(x_kt, w_kt, eps as f32, tape)
    .map(/* kt → candle for Phase 6→7 transition window */)
    .context("fused_rmsnorm_via_kt_tape failed");
```

At that point `fused_rmsnorm_via_kt_forward_op` and
`crates/kiln-rmsnorm-kernel/src/kt_forward_op.rs` can be deleted in
the same PR, completing the Phase 7 rmsnorm-crate candle dep drop
named in `CANDLE_REMOVAL_PLAN.md` line 341.

## Cross-references

* Pilot commit: `895162ca` —
  "kiln-rmsnorm-kernel: pilot kt-tape backward port for fused RMSNorm (#1082)"
* Companion FLCE pilot: `5a78a0ef` + follow-up `f83ec4c1`
* Companion OPD pilot: `5478e64f`
* CP-4 doc: [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md)
  §"Top 3 next-tasks" #1 (line 209-218)
* Related STOP doc precedent:
  [`kiln-server-candle-removal-stop-2026-05-27.md`](./kiln-server-candle-removal-stop-2026-05-27.md)
  (same "blocked-by-substrate" shape)
* Related STOP doc precedent:
  [`lora-bwd-kt-migration-stop-2026-05-27.md`](./lora-bwd-kt-migration-stop-2026-05-27.md)
  (same "two patterns don't mix" shape)

## Addendum (Wave-13 / 2026-05-28) — extending the substrate hook to OPD and FLCE

After this STOP landed, two wave-12 commits extended the substrate
groundwork:

* `deed13a8` — `kiln-model: add experimental KILN_USE_TAPE_FORWARD
  env-gated tape pipeline (#1082)` — added the thread-local-tape
  scope (`with_thread_local_tape` / `with_active_tape` / the
  `KILN_USE_TAPE_FORWARD` env tristate) and the candle-typed
  `try_tape_rms_norm_cuda` adapter wired into
  `kiln-model::forward::rms_norm`.
* `54159a76` — `kiln-model: add tape_forward parity test for
  KILN_USE_TAPE_FORWARD path (#1082)` — bit-exact parity test
  between the tape-routed forward and the kt-shim forward.

Wave-13 (this section's date) extends the same opt-in pattern to OPD
and FLCE so the substrate hook is in place across all three Phase-6a
kernels:

* `b2702ce0` — `kiln-autograd: promote thread-local Tape scope
  (with_thread_local_tape / with_active_tape) into kiln-autograd
  (#1082)`. Moves the thread-local-tape machinery out of `kiln-model`
  so the OPD and FLCE kernel crates (and their `kiln-train` callers)
  can route through the same handle without taking a `kiln-model`
  dependency. `kiln-model::tape_forward` re-exports the symbols for
  back-compat.
* `79d4873c` — `kiln-model: refactor tape_forward to re-export
  kiln-autograd::tape_scope (#1082)`. Companion to the previous; the
  rmsnorm adapter `try_tape_rms_norm_cuda` stays in `kiln-model`
  because it depends on `kiln-rmsnorm-kernel` and `kiln-kt-bridge`.
* `e6b8c3a3` — `kiln-opd-loss-kernel + kiln-train: add
  try_tape_opd_per_position_cuda adapter; wire opd_step_loss to
  short-circuit when tape scope open (#1082)`. Mirror of the rmsnorm
  adapter for the OPD production caller at
  `kiln-train::opd::opd_step_loss`.
* `20173dbb` — `kiln-flce-kernel: add try_tape_flce_phase_b_cuda
  adapter; wire dispatch to short-circuit when tape scope open
  (#1082)`. Mirror for the FLCE production caller at
  `fused_linear_cross_entropy_dispatch_with_provider`. The
  `provider.is_none()` guard preserves the Vulkan FLCE escape hatch.

### What Wave-13 does and does not do

Wave-13 ships the substrate hook for OPD and FLCE: the opt-in
`KILN_USE_TAPE_FORWARD` env var + a thread-local `Tape` scope around
the training step is now sufficient to route any of {rmsnorm, OPD,
FLCE} through the kt-tape pilot. The production callers
short-circuit through the tape adapter when both gates are open and
fall through to the existing kt-forward-op shims otherwise.

Wave-13 does **not** delete any of:

* `crates/kiln-rmsnorm-kernel/src/kt_forward_op.rs`
* `crates/kiln-opd-loss-kernel/src/kt_forward_op.rs`
* `crates/kiln-flce-kernel/src/kt_forward_op.rs`

The architectural reason (this STOP's main body) has not changed:
the candle-autograd training loop (`loss.backward()` over
`candle_core::backprop::GradStore`) cannot consume nodes recorded on
a `kiln_autograd::Tape`. With `KILN_USE_TAPE_FORWARD` unset (the
default), every adapter returns `Ok(None)` and the kt-shim path is
exactly the one that ran before Wave-13. The shim CANNOT be deleted
until `kiln-train` ports its `loss.backward()` callers onto
`Tape::backward` (CP-4 substrate). Until then, the shim is the only
path that drives gradients through candle's autograd graph correctly.

### What unblocks the deletion

The same Option A from the main body: `kiln_autograd::Var` substrate
+ `kiln-train::trainer` port off `loss.backward()`. With Wave-13 in
place, the CP-4 work no longer has to thread `&mut Tape` through 20+
candle-typed forward functions in `kiln-model::forward`. It needs to:

1. Open a `with_thread_local_tape` scope around each training step
   in `kiln-train::trainer`.
2. Set `KILN_USE_TAPE_FORWARD=1` in the training-step config.
3. Replace `loss.backward()` with `tape.backward(loss_id, ...)`.

At that point the three `kt_forward_op.rs` modules become dead code
and can be deleted in a single PR. The `try_tape_*_cuda` adapters
become the unconditional path (the env gate + scope check fall away
once the substrate is the only training path).
