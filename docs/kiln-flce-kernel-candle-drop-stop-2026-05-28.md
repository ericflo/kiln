# `kiln-flce-kernel` candle-core dep removal — STOP (audit 2026-05-28)

## TL;DR

After the Phase 6a/CP-4 kt-tape pilot port (commit `5a78a0ef` — see
`crates/kiln-flce-kernel/src/kt_tape.rs`) and the production-caller flip
to the kt-forward-op shim (`fused_linear_cross_entropy_phase_b_via_kt_forward_op`,
wired into `crates/kiln-train/src/trainer.rs::sft_train` via
`fused_linear_cross_entropy_dispatch`), the FLCE kernel crate **still
cannot drop `candle-core` from `[dependencies]`**.

This STOP records the evaluation that confirms the preconditions for
candle-core removal are **not** met. The kt-tape adapter
(`try_tape_flce_phase_b_cuda`) is opt-in (gated by
`KILN_USE_TAPE_FORWARD=1` + an active thread-local `kiln_autograd::Tape`
scope); the kt-forward-op shim is the default production fallback **and**
is itself a candle `CustomOp1` via `kiln_kt_bridge::forward_op::KtForwardOp1`;
and three other modules in the crate still take or return
`candle_core::Tensor` for the production-caller contract:

| File                                  | Why it still needs candle                                                                                                                                              |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/lib.rs::fused_linear_cross_entropy{,_dispatch{,_with_provider}}` | Public production API. Takes `&Tensor` in, returns `Tensor`. Phase-A reference (`fused_linear_cross_entropy`) is pure-candle and is reachable via `KILN_FLCE_PHASE_A=1`. Dispatch fns are the entry the trainer calls. |
| `src/phase_b.rs::FlceCustomOp` + `fused_linear_cross_entropy_phase_b{,_with_provider}` | The Phase-B production candle `CustomOp1`. CPU/CUDA/Metal `*_fwd` methods + `bwd` are all keyed on `candle_core::{CpuStorage, CudaStorage, MetalStorage}`. The kt-bridge backward (`fused_linear_cross_entropy_phase_b_backward_via_kt_bridge`) is the bwd fast path but it still lives inside this candle CustomOp's `bwd()`. |
| `src/kt_forward_op.rs::fused_linear_cross_entropy_phase_b_via_kt_forward_op` | Default production caller. Wraps `KtForwardOp1` (a candle `CustomOp1`) over the kt-typed forward+backward; takes `&Tensor` in, returns `Tensor`. Forward + backward closures move candle tensors across the kt boundary via `kt_tensor_from_candle_cuda_borrow` / `kt_tensor_to_candle_cuda_copy`. |
| `src/tape_forward.rs::try_tape_flce_phase_b_cuda` | Tape-gated production short-circuit. Takes `&Tensor` in (the trainer's candle hidden) and returns `Option<Tensor>` (candle copy of the kt result). |

The pure-kt code paths (`src/kt_api.rs` and `src/kt_tape.rs`) are
already candle-free in their **bodies**: `kt_api.rs` has only two
candle references and both are in module-doc comments; `kt_tape.rs`
has zero non-doc references. But neither of these are reachable from
the production trainer without going through one of the candle-typed
entries above.

## Numbers (candle refs by file, kiln HEAD `3a1f8d02`)

```
crates/kiln-flce-kernel/src/phase_b.rs       — 22 refs (production, CustomOp1 + kt-bridge bwd)
crates/kiln-flce-kernel/src/kt_forward_op.rs — 15 refs (production, KtForwardOp1 shim closures)
crates/kiln-flce-kernel/src/tape_forward.rs  —  2 refs (1 doc, 1 `use candle_core::Tensor`)
crates/kiln-flce-kernel/src/lib.rs           —  2 refs (1 doc, 1 `use candle_core::{D, DType, Device, Tensor}`)
crates/kiln-flce-kernel/src/kt_api.rs        —  2 refs (both doc-only — no actual use of candle)
crates/kiln-flce-kernel/src/kt_tape.rs       —  0 refs in code; 14 in module-doc comments
                                              -----
                                              43 total grep hits
                                              ~40 are production code (use sites + impls)
                                              ~3-4 are doc-comment mentions of candle types
```

The `[dev-dependencies]` line `candle-nn = { workspace = true }` is
**dead** — `grep -rn 'candle_nn\|candle-nn' crates/kiln-flce-kernel/`
returns only the `Cargo.toml` line itself. No test, bench, or src file
uses it. This dev-dep can be dropped independently of the
`[dependencies] candle-core` blocker; that small win is shipped in
the same PR as this STOP doc (see the trailing "What this commit
ships" section).

## What the resume task asked

> "Evaluate whether `kiln-flce-kernel`'s `candle-core` and `candle-nn`
> deps can be moved to `[dev-dependencies]` (or dropped entirely). The
> kt_tape.rs pilot already exists (commit `5a78a0ef`); we need to know
> what production candle refs remain.
>
> [...]
>
> **OUTCOME A**: Most candle refs are bridge/production, can't drop
> yet → write a STOP doc describing the pattern of who calls these
> functions from kiln-train and kiln-model, and what's needed to
> migrate (e.g. KtForwardOp shim adoption, kt-typed signatures).
> Reference `opd-loss-kernel-candle-removal-stop-2026-05-28.md` for
> the same architectural pattern.
>
> **OUTCOME B**: Most production refs already through kt_tape.rs /
> KtForwardOp shim → try moving candle to dev-deps."

This is **Outcome A**. The kt-tape pilot landed in `kt_tape.rs` but
the production caller in `kiln-train` still uses the candle-typed
`fused_linear_cross_entropy_dispatch` (verified at
`crates/kiln-train/src/trainer.rs:52-56`).

## Audit at HEAD (`main@3a1f8d02`)

### 1. Production caller — `kiln-train::trainer::sft_train`

`crates/kiln-train/src/trainer.rs:52-56`:

```rust
use kiln_flce_kernel::fused_linear_cross_entropy;
use kiln_flce_kernel::{
    DEFAULT_CHUNK_SIZE, FlceMatmulProvider, FlceProvider, fused_linear_cross_entropy_dispatch,
    fused_linear_cross_entropy_dispatch_with_provider,
};
```

The trainer pulls in four candle-typed FLCE entries:

* `fused_linear_cross_entropy` — the Phase-A pure-candle reference. Still
  re-exported even though the dispatch fn route defaults to Phase B; the
  trainer's `sft_train` calls into `fused_linear_cross_entropy_dispatch`,
  which delegates to either Phase A (env opt-in) or
  `fused_linear_cross_entropy_phase_b_via_kt_forward_op` (default Phase B).
* `FlceProvider` / `FlceMatmulProvider` — the candle-typed matmul
  provider trait the Vulkan FLCE escape (`kiln-vulkan-kernel`)
  implements to route per-chunk matmuls through Vulkan. Removing
  this hook is its own larger migration (the kt-typed twin
  `FlceMatmulProviderKt` already exists in `kt_api.rs` but
  kiln-vulkan-kernel hasn't migrated).
* `fused_linear_cross_entropy_dispatch{,_with_provider}` — the
  central production dispatch fn. Both `hidden: &Tensor` and the
  return type are `candle_core::Tensor`.

The kt-tape adapter (`try_tape_flce_phase_b_cuda`) is **only** reached
inside `fused_linear_cross_entropy_dispatch_with_provider` when:

1. `KILN_USE_TAPE_FORWARD=1` is set (the env tristate in
   `kiln_autograd::tape_forward_enabled()`; default off), AND
2. `kiln_autograd::with_active_tape(...)` finds a thread-local `Tape`
   scope on the current thread (today: only set by the kt-tape
   substrate tests in `crates/kiln-rmsnorm-kernel/`,
   `crates/kiln-flce-kernel/`, `crates/kiln-opd-loss-kernel/`).

Otherwise the trainer falls through to
`fused_linear_cross_entropy_phase_b_via_kt_forward_op`, which is a
**candle `CustomOp1`** wrapper (`KtForwardOp1`) around the kt-typed
forward + backward (see
`crates/kiln-flce-kernel/src/kt_forward_op.rs:264-439`). The forward
closure runs the candle Phase-B body
(`fused_linear_cross_entropy_phase_b_with_provider`); the backward
closure borrows candle CUDA tensors into kt with
`kt_tensor_from_candle_cuda_borrow`, runs the fused CUDA kernel
(`fused_linear_cross_entropy_phase_b_backward_kt`), and copies the
result back to candle with `kt_tensor_to_candle_cuda_copy`.

The kt-forward-op shim is the candle `CustomOp1` the production
caller relies on. Removing it would require either:

* Flipping production to **always** use the tape adapter (needs a
  caller-side `&mut Tape` scope wrapping the full step, and a
  `Tape::backward(loss_id, seed, accumulator)` driver replacing
  `loss.backward()` in the trainer), or
* Migrating `kiln-train::trainer::sft_train` and every downstream
  consumer (`opd_train`, the Vulkan FLCE provider, the per-step
  metrics computation, ...) to a pure-kt signature.

Both are CP-4 (kt-typed autograd `Var`/`Tape` adoption substrate)
work, already named in
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) as the
highest-leverage remaining item.

### 2. `phase_b::FlceCustomOp` — production-callable, not dead

Unlike the OPD case (`opd-loss-kernel-candle-removal-stop-2026-05-28.md`,
section §2: "phase_b::OpdLossCustomOp — dead in production"),
`FlceCustomOp` is **still production-callable** from the trainer
through three routes:

1. **Default Phase B path inside the kt-forward-op shim**: the shim's
   forward closure calls `fused_linear_cross_entropy_phase_b_with_provider`
   directly. See `kt_forward_op.rs:317-340`. The module docs explain
   this is intentional: the kt-typed forward
   (`fused_linear_cross_entropy_phase_b_kt`) has substrate gaps at
   production-trainer shapes that surface as cross-device
   `index_select` errors — currently kt-tensor's index constructors
   build CPU tensors regardless of the parent tensor's device. (See
   the `#[ignore]` markers on `kt_tape.rs::forward_records_tape_node_when_cuda_available`
   and `kt_tape.rs::backward_apply_returns_dhidden_shape_and_none_for_head`
   — same root cause.)
2. **Provider-bound chunk matmuls**: when an `FlceProvider` is bound
   (the trainer's Vulkan FLCE escape hatch), the kt-forward-op shim
   delegates straight to
   `fused_linear_cross_entropy_phase_b_with_provider` (the candle
   `CustomOp1`) because the shim has no provider plumbing.
3. **Kill switch fallback**: when `KILN_DISABLE_FLCE_KT_FORWARD_OP=1`
   is set, the shim falls back to the candle `CustomOp1` path. This
   is the kt parity-test / reversibility escape hatch and matches the
   precedent set by `KILN_DISABLE_FLCE_BWD_KT_BRIDGE`,
   `KILN_DISABLE_OPD_KT_FORWARD_OP`, `KILN_DISABLE_RMSNORM_KERNEL`,
   `KILN_DISABLE_FUSED_CONV1D`, etc.

The `FlceCustomOp::bwd()` (`phase_b.rs:305-358`) already routes
through the kt-bridge backward fast path
(`fused_linear_cross_entropy_phase_b_backward_via_kt_bridge`, commit
`ab2da23f`) for CUDA inputs — but the outer `CustomOp1` is still a
candle node attached to the candle autograd graph. The kt-bridge
backward returns a candle `Tensor` that gets handed back to candle's
`GradStore::store_grad` machinery.

### 3. `lib.rs::fused_linear_cross_entropy` — the Phase-A reference

The Phase-A candle reference (`fused_linear_cross_entropy`,
`lib.rs:259-451`) is the parity-test oracle for Phase B and the
opt-in escape hatch for debugging via `KILN_FLCE_PHASE_A=1`. It is
pure-candle: chunks the `[active, hidden] @ [hidden, chunk_len]`
matmul through `candle_core::Tensor::matmul`, runs the chunked
log-sum-exp reduction over candle ops (`max_keepdim`, `exp`,
`sum_keepdim`, `broadcast_as`, `index_add`, ...), and lets candle
autograd flow through the entire chunk loop.

Removing it would either:

* Delete the parity oracle outright — possible only once the
  kt-typed `fused_linear_cross_entropy_phase_b_kt` body is exercised
  end-to-end at production-trainer shapes on every test matrix
  variant. Today's `kt_tape.rs` CUDA E2E tests are `#[ignore]`-d on
  the kt-substrate index-op cross-device gap (see §2 above), so
  the kt-typed forward isn't ready to be the sole parity reference.
* Re-implement the Phase-A reference over `kiln_tensor::Tensor`
  inside `kt_api.rs`. This is the "natural next step" the
  `kt_api.rs` module docs themselves call out (line 11-17):
  > "kiln-flce-kernel does **not** yet have any raw-CUDA FFI to wrap:
  > today's Phase A/B forward+backward run on `candle_core::Tensor`
  > ops (matmul, exp, max_keepdim, ...). The kt-typed surface this
  > module defines is therefore the *migration target*."

The kt-typed twin already exists
(`fused_linear_cross_entropy_phase_b_kt` in `kt_api.rs`) but it
shares the same kt-substrate index-op cross-device gap as the
`KtForwardOp1` forward closure (see §2). Until that gap closes, the
Phase-A candle reference can't be deleted without losing the parity
oracle for Phase B at production shapes.

### 4. `kiln_kt_bridge`-only refs in `phase_b.rs` and `kt_forward_op.rs`

15+ of the `phase_b.rs` candle refs and most of the
`kt_forward_op.rs` candle refs are inside the kt-bridge backward fast
path: `kt_tensor_from_candle_cuda_borrow` / `kt_tensor_to_candle_cuda_copy`
calls and their candle-error-formatting glue. These can ONLY go away
once the public function signature stops taking/returning
`candle_core::Tensor` — i.e. the production caller in `kiln-train`
migrates to a pure-kt signature. Without that, the bridge is the
load-bearing seam and the candle refs are the surface area on the
bridge's candle side.

## What this commit ships

Two changes:

1. **This STOP doc** — pure documentation. No code change in
   `crates/kiln-flce-kernel/src/`. The candle-core dep stays in
   `[dependencies]`.
2. **`candle-nn` dev-dep removal** — `candle-nn = { workspace = true }`
   in `[dev-dependencies]` is unused. `grep -rn 'candle_nn\|candle-nn'
   crates/kiln-flce-kernel/` returns only the `Cargo.toml` line. This
   is a small no-risk cleanup that aligns with the #1082 epic
   (every dropped candle reference is a win, even a dev-dep).

The orthogonal cleanups the audit identified — none of which unblock
candle-core removal but each reduces the candle surface in the crate
— are listed below for completeness:

* **Move the Phase-A candle reference under `#[cfg(test)]`** — the
  Phase-A path is currently reachable in production via
  `KILN_FLCE_PHASE_A=1`. Making it test-only would shave one
  production candle reference (`use candle_core::{D, DType, Device, Tensor};`
  in `lib.rs`). But the env opt-in is the documented parity-debug
  escape, so cfg-gating it is a behaviour change Eric should
  greenlight first.
* **Inline `fused_linear_cross_entropy` re-export at use sites** —
  the trainer's `use kiln_flce_kernel::fused_linear_cross_entropy;`
  (`trainer.rs:52`) is the only external caller of the Phase-A
  reference. Confirming the call-site list is empty would let the
  re-export drop from `lib.rs`. Possible but small; defer until
  candle-core removal is in flight.

Both are independent of the candle-core dep blocker and can land on
separate PRs.

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
  `&[(Var<KtTensor>, KtTensor)]` instead of `&[Var<Tensor>]`.
* Closure of the kt-substrate index-op cross-device gap that
  currently forces `kt_tape.rs`'s CUDA E2E tests to be
  `#[ignore]`-d (see §2 above).

That is the same CP-4 substrate block that
[`opd-loss-kernel-candle-removal-stop-2026-05-28.md`](./opd-loss-kernel-candle-removal-stop-2026-05-28.md)
and [`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
record as the upstream blocker. **Three** kernels (rmsnorm + OPD + flce)
are now stacked behind the same training-loop refactor.

## Cross-references

* [`opd-loss-kernel-candle-removal-stop-2026-05-28.md`](./opd-loss-kernel-candle-removal-stop-2026-05-28.md)
  — sibling STOP for OPD's identical pattern.
* [`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
  — sibling STOP for rmsnorm's kt-tape production-caller flip.
* [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) — section
  "kt-autograd readiness" and CP-4.
* [`kt-tape-substrate-landed-in-kiln-train-2026-05-28.md`](./kt-tape-substrate-landed-in-kiln-train-2026-05-28.md)
  — Wave-13 substrate addendum.
* Commit `5a78a0ef` — Phase 6a/CP-4 FLCE kt-tape pilot port
  (`kt_tape.rs`).
* Commit `72339698` — production caller flip to
  `fused_linear_cross_entropy_phase_b_via_kt_forward_op`.
* Commit `ab2da23f` — `FlceCustomOp::bwd()` kt-bridge backward
  migration.
