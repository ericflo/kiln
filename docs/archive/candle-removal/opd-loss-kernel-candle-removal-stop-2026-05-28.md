# `kiln-opd-loss-kernel` candle-core dep removal — STOP (audit 2026-05-28)

> **Historical snapshot, not current operating guidance.** This document records
> migration state from May 2026. The `KILN_USE_TAPE_*` and
> `KILN_USE_TAPE_AUTHORITATIVE` switches mentioned below were removed without
> aliases or replacement fields. Current GPU training uses an internal tape
> scope as its sole routing authority. See [Configuration](../../CONFIGURATION.md)
> and [Native SFT Profile](../../NATIVE_SFT_PROFILE.md) for current behavior.

## TL;DR

After Wave-13's `try_tape_opd_per_position_cuda` (commit `e6b8c3a3`) and
the kt-forward-op shim's production caller flip
(`opd_top_k_reverse_kl_per_position_via_kt_forward_op`,
in `crates/kiln-train/src/opd.rs::opd_step_loss`), the OPD kernel crate
**still cannot drop `candle-core` from `[dependencies]`**.

This STOP records the resume-task aggressive audit that confirms the
preconditions for candle-core removal are **not** met. The kt-tape
adapter is opt-in (gated by `KILN_USE_TAPE_FORWARD=1` + an active
thread-local `kiln_autograd::Tape` scope), the kt-forward-op shim is
the default production fallback **and** is a candle `CustomOp1` via
`kiln_kt_bridge::forward_op::KtForwardOp1`, and three other modules in
the crate still take or return `candle_core::Tensor` for the
production caller contract:

| File                                  | Why it still needs candle                                                                                                                                              |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/kt_forward_op.rs`                | Production caller. Wraps a `KtForwardOp1` (candle `CustomOp1`) for the OPD trainer; takes `&Tensor` in, returns `Tensor`. Forward + backward closures move candle tensors across the kt boundary. |
| `src/tape_forward.rs`                 | Tape-gated production short-circuit. Takes `&Tensor` in (the trainer's candle student-hidden) and returns `Option<Tensor>` (candle copy of the kt result).             |
| `src/lib.rs::opd_top_k_reverse_kl_phase_a_per_position` + `per_position_phase_a` | Pure-candle reference path used as the fallback inside `kt_forward_op` when the kt-shim envelope (`K ∈ {16, 32}`, `dtype ∈ {F32, BF16}`, CUDA, matching head_t dtype) is not satisfied. |

The `OpdLossCustomOp` candle `CustomOp1` in `src/phase_b.rs` is
**dead in production** — the kt-shim wraps `KtForwardOp1` (defined in
`kiln-kt-bridge`), not `OpdLossCustomOp`. The only remaining callers
of `phase_b::opd_top_k_reverse_kl_phase_b{,_per_position}` (and
transitively `OpdLossCustomOp`) are test-only:

* `crates/kiln-train/src/opd.rs::opd_train_synthetic_validation`
  (`#[cfg(test)] pub` smoke for the AdamW + autograd loop).
* `crates/kiln-train/tests/vk_cuda_opd_parity.rs::run_cuda_per_position`
  (the §9.2 grand-plan CUDA-vs-Vulkan parity gate).

The aggressive option in the resume task framing — *"check if
`OpdLossCustomOp` can be REPLACED entirely by the tape_forward adapter
(`e6b8c3a3`)"* — does not unblock candle-core removal even when the
test-only callers are migrated to the production shim, because the
kt-forward-op shim itself is still a candle `CustomOp1`.

## What the resume task framed

> "Aggressive option: check if `OpdLossCustomOp` can be REPLACED
> entirely by the tape_forward adapter (`e6b8c3a3`). If production
> callers can go through `try_tape_opd_per_position_cuda` and the
> candle CustomOp is dead, delete it.
>
> Steps:
> 1. Read `crates/kiln-opd-loss-kernel/` after wave 13 work
> 2. Audit `OpdLossCustomOp` callers — production vs tests
> 3. If production tape adapter covers all paths: delete `OpdLossCustomOp` + `kt_forward_op.rs`
> 4. Try drop candle-core from Cargo.toml `[dependencies]`
>
> If blocked, STOP-doc the exact remaining piece."

This audit fills in step 4's "if blocked" branch.

## Audit at HEAD (`main@7b0e9dbd`)

### 1. Production caller — `kiln-train::opd::opd_step_loss`

`crates/kiln-train/src/opd.rs:1258-1294`:

```rust
let per_position_kl = {
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = kiln_opd_loss_kernel::try_tape_opd_per_position_cuda(
            student_hidden, head_t, &teacher_topk_indices,
            &teacher_topk_logprobs, &label_mask, resolved_top_k,
        )? {
            out
        } else {
            opd_top_k_reverse_kl_per_position_via_kt_forward_op(
                student_hidden, head_t, &teacher_topk_indices,
                &teacher_topk_logprobs, &label_mask, resolved_top_k, &device,
            )?
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        opd_top_k_reverse_kl_per_position_via_kt_forward_op(
            student_hidden, head_t, &teacher_topk_indices,
            &teacher_topk_logprobs, &label_mask, resolved_top_k, &device,
        )?
    }
};
```

The tape adapter (`try_tape_opd_per_position_cuda`) is only reached
when:

1. `KILN_USE_TAPE_FORWARD=1` is set in the trainer's env (default off),
   AND
2. `kiln_autograd::with_active_tape(...)` finds a thread-local `Tape`
   scope on the current thread (today: only set inside the kt-tape
   substrate tests in `crates/kiln-rmsnorm-kernel/`,
   `crates/kiln-flce-kernel/`, etc.).

Otherwise the trainer falls through to
`opd_top_k_reverse_kl_per_position_via_kt_forward_op`, which is a
**candle `CustomOp1`** wrapper around the kt-typed forward + backward
(see `crates/kiln-opd-loss-kernel/src/kt_forward_op.rs:155-466`). The
forward closure borrows the candle CUDA tensors into kt with
`kt_tensor_from_candle_cuda_borrow`, runs the kt composite
(`opd_top_k_reverse_kl_per_position_kt`), and copies the result back
to candle with `kt_tensor_to_candle_cuda_copy`. The backward closure
runs the fused CUDA kernel (`opd_top_k_reverse_kl_phase_b_bwd_kt`)
through the same kt borrow/copy bridge.

The kt-forward-op shim is the candle CustomOp the production caller
relies on. Removing it would require either:

* Flipping production to **always** use the tape adapter (needs a
  caller-side `&mut Tape` scope wrapping the full step, and a
  `Tape::backward(loss_id, seed, accumulator)` driver replacing
  `loss.backward()` in the trainer), or
* Migrating `kiln-train::opd::opd_step_loss` and every downstream
  consumer (`opd_train`, `stable_opd_loss_step`, `vk_cuda_opd_parity`,
  …) to a pure-kt signature.

Both are CP-4 (kt-typed autograd `Var`/`Tape` adoption substrate)
work, already named in [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md)
as the highest-leverage remaining item.

### 2. `phase_b::OpdLossCustomOp` — dead in production

The Wave-9 kt-bwd-bridge work (commit `0c1be227`) and the kt-forward-op
shim (`#1082`) replaced the production usage of `OpdLossCustomOp` with
`KtForwardOp1`. The only remaining callers of the public re-exports
`opd_top_k_reverse_kl_phase_b{,_per_position}` (which transitively
build `OpdLossCustomOp`) are:

* `crates/kiln-train/src/opd.rs:1857` — inside
  `#[cfg(test)] pub fn opd_train_synthetic_validation`, the
  AdamW-driven KL-down smoke test.
* `crates/kiln-train/tests/vk_cuda_opd_parity.rs:113` — the
  CUDA-vs-Vulkan §9.2 parity gate.

`compute_per_position_metrics` / `PerPositionMetrics` (also in
`phase_b.rs`) have **no callers outside the crate** — re-exported but
unused.

### 3. The unused fused FWD CUDA kernel

`kiln_opd_topk_kl_fwd_{bf16,f32}` (defined in
`crates/kiln-opd-loss-kernel/csrc/opd_topk_kl.cu:401-440`) is **only**
called by `OpdLossCustomOp::cuda_kernel_forward`. The production
forward path runs the kt composite
(`per_position_forward_kt` in `kt_api.rs`), not the fused FWD
kernel. The fused BWD kernel `kiln_opd_topk_kl_bwd_{bf16,f32}` is
still live — both the candle path (`cuda_kernel_backward`) and the kt
path (`opd_top_k_reverse_kl_phase_b_bwd_kt`) call it.

`kiln_opd_topk_metrics_{bf16,f32}` is only called by
`cuda_compute_per_position_metrics` in `phase_b.rs`, which is itself
only reached through the unused `compute_per_position_metrics` entry.

### 4. `lib.rs::opd_top_k_reverse_kl_phase_a_per_position`

The kt-forward-op shim falls through to this candle reference path
when the input envelope is outside `(K ∈ {16, 32}, dtype ∈ {F32, BF16},
CUDA, head_t dtype matches hidden dtype)`. Removing it would either:

* Widen the kt-shim envelope to cover all OPD trainer inputs
  (currently restricted to the CUDA fast path; CPU and `K = 8` are
  the holdouts), OR
* Mark the shim as CUDA-only and refuse to compile on non-CUDA
  targets, then route the CPU OPD path through the kt-typed
  `opd_top_k_reverse_kl_per_position_kt` directly (still candle-free
  on the consumer side, but `kiln-train` would need to upload
  via `kiln-kt-bridge` from CPU candle tensors — which today only
  supports CUDA borrow).

Either route is wider than this resume task scope.

## What this commit ships

This is a pure documentation STOP. No code changes in
`crates/kiln-opd-loss-kernel/`. The candle-core dep stays in
`Cargo.toml`. The next steps below are the orthogonal cleanups the
aggressive audit identified — they reduce the candle surface in the
crate without dropping the dep:

1. Migrate the two test callers
   (`opd_train_synthetic_validation`,
   `vk_cuda_opd_parity::run_cuda_per_position`) to use
   `opd_top_k_reverse_kl_per_position_via_kt_forward_op` (the
   production shim). Output is bit-different from the fused FWD
   kernel up to f32 associativity (composite kt ops vs single
   fused CUDA kernel), but the existing test tolerances accommodate
   this (`1e-4` abs / `1e-3` rel).
2. Delete `phase_b.rs` after the test migration. This drops
   `OpdLossCustomOp`, the `opd_top_k_reverse_kl_phase_b{,_per_position}`
   re-exports, and `compute_per_position_metrics` /
   `PerPositionMetrics`.
3. Drop `kiln_opd_topk_kl_fwd_{bf16,f32}` and
   `kiln_opd_topk_metrics_{bf16,f32}` from
   `csrc/opd_topk_kl.cu`. The fused BWD kernel stays.

These cleanups are independent of the candle-core dep removal and
can land on a separate PR.

## When this STOP unblocks

When the trainer (`kiln-train::trainer::sft_train` and
`kiln-train::opd::opd_train`) is migrated to drive backward via
`Tape::backward(loss_id, seed, accumulator)` instead of
`loss.backward()`. Today the trainer uses
`<candle_nn::optim::AdamW as candle_nn::Optimizer>::backward_step`
which internally calls `loss.backward()` and walks the candle
autograd graph. A kt-tape-driven loop requires:

* A `&mut Tape` threaded through every kernel call in the step
  (forward, KL loss, RmsNorm, FLCE, OPD, attention, MLP, ...).
* `kiln_autograd::Var<KtTensor>` (CP-4 substrate item) wrapping
  every LoRA parameter so the tape can accumulate gradients
  symmetrically with candle's `Var<Tensor>`.
* An AdamW-equivalent optimizer that consumes
  `&[(Var<KtTensor>, KtTensor)]` instead of
  `&[Var<Tensor>]`.

That's the same CP-4 substrate block that `rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`
records. Both kernels (rmsnorm + OPD) are waiting on the same
training-loop refactor.

## Post-wave-14 re-audit (2026-05-28, this PR)

Wave-14 (commit `84549819`) deleted the dead `OpdLossCustomOp` +
fused-FWD / metrics kernels (~1880 LOC). A resume task re-asked the
same question this STOP records — *"with `OpdLossCustomOp` gone, can
candle-core finally move to `[dev-dependencies]` (or drop)?"* — under
the framing that the deletion might have removed the last production
candle dependency. The re-audit at HEAD (after `73109cbe`,
`94ce67da`) confirms the conclusion is unchanged:

* Total candle refs in `crates/kiln-opd-loss-kernel/src/` after
  wave-14: **27** (down from ~80+ pre-wave-14). All 27 surviving
  refs are in **production** module bodies, not `#[cfg(test)]`:
  * `src/lib.rs` — 2 refs. The candle Phase A reference path
    (`opd_top_k_reverse_kl_phase_a_per_position` + helpers) is the
    fallback inside `kt_forward_op.rs` when the kt envelope rejects
    inputs. Still live in production.
  * `src/kt_forward_op.rs` — 24 refs. The production caller in
    `kiln-train::opd::opd_step_loss` (line 1280, 1293, 1877) calls
    `opd_top_k_reverse_kl_per_position_via_kt_forward_op`, which
    wraps `KtForwardOp1` (a candle `CustomOp1`) and moves candle
    `Tensor`s across the kt bridge. All 24 refs are above line 474
    (the `#[cfg(test)]` block start) — i.e. in the production module
    body.
  * `src/tape_forward.rs` — 2 refs. `try_tape_opd_per_position_cuda`
    takes `&Tensor` and returns `Option<Tensor>` so the trainer's
    candle-autograd loop can splice in the kt-tape result. Module is
    `#![cfg(feature = "cuda")]` (CUDA-only) but still production
    when `KILN_USE_TAPE_FORWARD=1` + a thread-local `Tape` scope is
    active.
  * `src/phase_b.rs` — 1 ref. Only a `use candle_core::DType;` for
    the `cuda_kernel_supports` envelope helper. **Could** be ported
    to `kiln_tensor::DType` independently, but that's a separate
    micro-refactor and doesn't unblock the dep removal because the
    other three files above all still take/return candle `Tensor`.
  * Two doc-comment refs in `src/lib.rs` and `src/kt_api.rs` — just
    text, not code.

* `cargo check -p kiln-opd-loss-kernel` (CPU, default features)
  passes locally at this HEAD. CUDA `cargo check -p
  kiln-opd-loss-kernel --features cuda` could not be re-verified on
  pod for this re-audit (`ce kiln-pod-acquire` returned HTTP 500
  "no instances available" twice in a row from RunPod GraphQL —
  capacity exhausted in the A6000 pool today). Since this is a
  docs-only commit with no Cargo.toml or `.rs` changes, the pod
  verification adds no signal beyond the local CPU check and the
  CUDA build was already validated earlier in (#1082) post-wave-14
  (commit `73109cbe`, the env-var test serialization fix, requires
  the CUDA feature to compile and builds successfully under
  `cargo check -p kiln-opd-loss-kernel --features cuda --tests` per
  the wave-13 pod runs documented in
  `docs/archive/candle-removal/kt-tape-substrate-landed-in-kiln-train-2026-05-28.md`).

* No new candle test-only refs were introduced by wave-14. There
  is no surviving candle ref that would benefit from a
  `[dev-dependencies]` move; every remaining ref is on a path the
  default `cargo build` exercises.

**Conclusion (unchanged from initial audit)**: candle-core stays
in `[dependencies]`. The blocker is **CP-4** — training-loop
refactor from `loss.backward()` to `Tape::backward(...)` — same
substrate gate `rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`
records. When CP-4 lands, `kt_forward_op.rs` + `tape_forward.rs` +
the Phase A candle reference can all be deleted in one pass and the
candle-core dep drops with them.

The orthogonal cleanup #4 — porting the lone `use candle_core::DType;`
in `phase_b.rs` to `kiln_tensor::DType` — is mentioned for
completeness but explicitly out of scope for this STOP. It's a
one-line change that doesn't unblock the dep removal and is better
done atomically with the CP-4 flip.

## Cross-references

* [`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
  — sibling STOP for rmsnorm's kt-tape production-caller flip.
* [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) — section
  "kt-autograd readiness" and CP-4.
* [`kt-tape-substrate-landed-in-kiln-train-2026-05-28.md`](./kt-tape-substrate-landed-in-kiln-train-2026-05-28.md)
  — Wave-13 substrate addendum.
* Commit `e6b8c3a3` — Wave-13 `try_tape_opd_per_position_cuda`.
* Commit `0c1be227` — `OpdLossCustomOp::bwd` kt-bridge migration.
* Commit `84549819` — Wave-14 deletion of dead `OpdLossCustomOp` +
  fused-FWD / metrics kernels (~1880 LOC).
* Commit `c8c341b2` — initial wave-14 STOP doc.

---

## Update — 2026-05-28 (post CP-4 bridge landing)

The CP-4 substrate's first end-to-end production wiring landed in
`675e0dea` (SFT `standard_forward_backward` wrapped in
`kiln_kt_bridge::tape_bridge::with_tape_scope_emit_to_grad_store`).
The bridge keeps `loss.backward()` intact and merges kt-tape
gradients into candle's GradStore via registered
`(kt_id ↔ candle_id)` IO mappings — see the
[`rmsnorm-kt-tape-production-caller-stop`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
"Update — 2026-05-28 (post CP-4 bridge landing)" section for the
4-step bridge walk.

The opd-loss kernel benefits identically:

- The opd-loss kt-tape adapter `try_tape_opd_per_position_cuda`
  (`e6b8c3a3`) becomes the recording path when
  `KILN_USE_TAPE_FORWARD=1` is set inside an SFT step. It registers
  IO mappings so the bridge can route the opd-loss gradient back
  into the candle GradStore on the matching candle TensorId.
- The `KtForwardOp1`-based shim (`kt_forward_op.rs`) is the default
  path when the env gate is unset.
- The kt-tape adapter still no-ops outside a bridge scope (returns
  `Ok(None)` and falls through to the shim), so callers that build
  a candle-only graph continue to work.

The candle-core Cargo dep **still cannot drop** because the shim
(`kt_forward_op.rs`) and Phase-A reference path (`lib.rs`) both name
candle types in their public surface — same blocker as in the main
body. The deletion moment moves to **when production training
defaults to `KILN_USE_TAPE_FORWARD=1`** and the kt-tape adapter
becomes the unconditional path. At that point both `kt_forward_op.rs`
and the candle-typed Phase-A fallback become dead code.

See [`candle-removal-status-2026-05-28-pm.md`](./candle-removal-status-2026-05-28-pm.md)
for the current per-crate state.
