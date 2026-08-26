# CP-4 kt-tape substrate — first lift into `kiln-train` (2026-05-28)

## TL;DR

`crates/kiln-train/src/tape_step.rs` now exists. It's the first
training entry in the `kiln-train` crate that runs **forward +
backward + SGD update entirely on `kiln_autograd::Tape`** — no
`candle_core::Tensor`, no `loss.backward()`, no
`candle_core::backprop::GradStore`. Two parallel entries land in this
substrate seed:

* `linear_step_via_tape(LinearStepInputs) -> Result<StepOutput>` —
  1-parameter linear regression (matmul + bias-add + MSE) with a
  scalar loss and per-element SGD update.
* `mlp_step_via_tape(MlpStepInputs) -> Result<MlpStepOutput>` —
  2-parameter MLP (matmul + silu + matmul + MSE) demonstrating the
  multi-layer composition real training needs.

The production trainer in `crate::trainer` still drives
`loss.backward()`. The kt-tape entries sit **alongside** it, not in
place of it — same pattern as the three kernel pilots
(`fused_rmsnorm_via_kt_tape` etc.) sitting alongside their candle
CustomOp shims.

## Why this is the right next step

The
[`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
audit at `main@20a5885c` concluded that **no per-call-site flip in
`kiln-model::forward::rms_norm` can land** until `kiln-train` itself
adopts the `kiln_autograd::Tape` substrate. Threading a `&mut Tape`
through `rms_norm` alone is substrate-work theatre — the tape would
be either local (no caller frame holds it long enough to consume)
or transitively threaded up through 20+ caller sites until the
training-loop root in `kiln-train::trainer::*` (which is the actual
substrate gap).

This commit pair closes the smallest non-empty subset of the gap:
the **substrate-introduction layer** inside `kiln-train`. The next
PRs extend it incrementally without revisiting whether the substrate
"works" at all.

## What this substrate proves

Concretely, after these two commits land:

1. **Dependency wiring**: `kiln-train` *can* depend on
   `kiln-autograd` without breaking the candle-typed trainer.
   No feature-flag collisions, no name conflicts, no transitive
   build problems. `cargo check -p kiln-train` and `cargo check`
   (full workspace) both pass cleanly.
2. **End-to-end composition**: a real supervised-learning step
   composes on `kiln_tensor::Tensor` + `kiln_autograd::Tape` +
   `kiln_tensor::ops::{matmul, add, sub, mul, silu, sum_all}` +
   `kiln_autograd::{MatmulBackward, AddBackward, SubBackward,
   MulBackward, SiluBackward, ReduceBackward}`. Backward walks
   produce correct gradients (loss descends, XOR converges).
3. **Per-parameter id keying**: `Tape::backward`'s returned
   `GradStore` keys gradients on the **original forward `TensorId`**.
   That's the same key `kiln_param::Parameter` and `kiln_optim` use
   for the optimizer slot — so the substrate plugs cleanly into the
   existing parameter machinery once the trainer's forward path is
   ported.
4. **Multi-layer disambiguation**: when two distinct parameters
   (`w1` and `w2`) flow through the same op type (`MatmulBackward`),
   the GradStore keys them correctly. The walker isn't conflating
   ids by op-type — it tracks per-input-position ids.
5. **Non-linear composition**: silu's gradient signal flows back
   through both layers, exercising the saved-input `BackwardOp`
   pattern (the same pattern `CudaFusedRmsNormBackward` and the FLCE
   `CrossEntropyBackward` use to save forward state by `Arc` clone).

## What's NOT in the substrate yet (the next-PR punch list)

In priority order, with the gap each PR closes:

### Next PR 1 — `Parameter` integration in `tape_step`

Substitute `&Tensor` inputs for `kiln_param::Parameter` slots. The
step's SGD helper rebinds via the Parameter's anti-pattern-11
storage-variant API. Tests verify the parameter id is stable across
the step (the storage-variant transitions don't change `TensorId`),
which is the property kiln-optim's Adam state slot depends on.

**Why this PR first**: every subsequent extension (optimizer state,
GPU dispatch, model-forward port) needs `Parameter`s — without them,
each PR forks on two substrate concerns at once.

**Files touched**: `crates/kiln-train/src/tape_step.rs` (extend
`MlpStepInputs` with `&[Parameter]`), add `kiln-param` /
`kiln-optim` workspace deps to `crates/kiln-train/Cargo.toml`.

### Next PR 2 — Adam/AdamW via `kiln-optim`

Replace the local CPU SGD helper with a `kiln-optim` call. The
optimizer is already kt-typed (`crates/kiln-optim/src/adam.rs`); the
substrate just needs to pass it the `Parameter` + grad pair from
the post-`Tape::backward` store.

**Files touched**: `crates/kiln-train/src/tape_step.rs` (swap
`sgd_step_cpu` for `kiln_optim::AdamW::step`). Drops the local
`sgd_step_cpu` + `read_f32_vec` helpers from the file.

### Next PR 3 — GPU dispatch via the kernel pilots

Re-route the matmul + silu + (eventually) rmsnorm through the kt-tape
kernel pilots (`fused_rmsnorm_via_kt_tape`,
`fused_linear_cross_entropy_phase_b_via_kt_tape`, etc.) when the
inputs land on a CUDA device with the right envelope. The CPU CPU-F32
path stays as the reference.

**Files touched**: `crates/kiln-train/src/tape_step.rs` (add
`#[cfg(feature = "cuda")]` branches that call the kernel pilots'
tape entries). Wire `kiln-rmsnorm-kernel/cuda`,
`kiln-flce-kernel/cuda`, and `kiln-opd-loss-kernel/cuda` features.

### Next PR 4 — Port `kiln_model::forward` ops onto the tape

The bulk of CP-4. Every `rms_norm` / `matmul` / `silu` / `embedding`
call site in `kiln-model/src/forward.rs` needs a `tape.record`
companion (or, if the op is purely inference-time with no
backward signal, an explicit `// no tape needed: inference-only`
comment). This is multi-PR work in its own right, one forward-op
family per PR.

**Critical insight from this commit's tests**: `Tape::backward`
drains intermediate-output grad entries as it walks (the
`grads.remove(&node.output_id)` line in `crates/kiln-autograd/src/tape.rs:171`).
The final GradStore is keyed on **leaves + loss seed**, not on every
recorded tape node's output. Future PRs that want recompute /
activation-checkpoint policy can't lean on the GradStore for
intermediate-id lookup — they need to hold references to the
`TapeNode`s themselves. This was surfaced by an over-strict
assertion in `mlp_grad_store_keys_include_both_layer_parameters`
that initially expected ≥ 6 GradStore entries; the assertion was
relaxed to ≥ 4 (the leaves) and the docstring documents the
substrate behavior.

### Next PR 5 — Flip the kernel pilots' production callers

Once PRs 1-4 land, the production caller in
`kiln-model::forward::rms_norm` can finally route through
`fused_rmsnorm_via_kt_tape` instead of
`fused_rmsnorm_via_kt_forward_op`. Same for FLCE and OPD.

At that point `kiln-rmsnorm-kernel/src/kt_forward_op.rs` and the
candle CustomOp wrapper layer can be deleted — completing the Phase 7
rmsnorm-crate / FLCE-crate / OPD-crate candle-dep drop named in
`CANDLE_REMOVAL_PLAN.md` line 341.

## Sibling pilots' status (for cross-reference)

| Pilot                                 | Commit       | Status |
| ------------------------------------- | ------------ | ------ |
| `fused_rmsnorm_via_kt_tape`           | `895162ca`   | Landed, parallel to candle CustomOp shim. Production-caller flip blocked on this substrate. |
| FLCE phase B `_via_kt_tape`           | `5a78a0ef`   | Landed, parallel to candle CustomOp shim. CUDA-gated E2E tests `#[ignore]`d in follow-up `f83ec4c1` for a separate index-op gap. |
| OPD `_via_kt_tape`                    | `5478e64f`   | Landed, parallel to candle CustomOp shim. Production caller in `kiln-train::opd` still uses candle path. |
| `kiln-train::tape_step::linear_step_via_tape` | **`73ac1c3c`** (this PR pair) | **First substrate seed inside `kiln-train`. Documented here.** |
| `kiln-train::tape_step::mlp_step_via_tape` | **`c5726a32`** (this PR pair) | **Multi-layer substrate proof. Documented here.** |

## Validation

* `cargo check -p kiln-train` — clean.
* `cargo check` (full workspace, no features) — clean.
* `cargo test -p kiln-train --lib tape_step::` on A6000 pod —
  **7/7 tests pass** (4 from `73ac1c3c`, 3 from `c5726a32`), no
  hardware required:

      running 7 tests
      test tape_step::tests::module_does_not_depend_on_candle_in_signatures ... ok
      test tape_step::tests::grad_store_keys_include_all_inputs ... ok
      test tape_step::tests::mlp_step_produces_per_layer_gradients ... ok
      test tape_step::tests::single_step_produces_parameter_gradient ... ok
      test tape_step::tests::mlp_grad_store_keys_include_both_layer_parameters ... ok
      test tape_step::tests::iterated_steps_drive_loss_down ... ok
      test tape_step::tests::mlp_step_can_learn_xor ... ok
      test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured;
                      268 filtered out; finished in 0.14s

* `cargo check --features cuda` on A6000 pod OOM'd during nvcc
  compile of an unrelated `kiln-flash-attn` `.cu` file (status 137 —
  system resource limit, not a compile error in `tape_step`). The
  substrate itself is CPU-only and gated behind no feature flag, so
  the CUDA build OOM is orthogonal — `tape_step` adds no
  `#[cfg(feature = "cuda")]` blocks and no new CUDA dependencies.

## Decision

**Substrate seed is in.** The next CP-4 PR can extend it without
rediscovering whether the substrate works at all.

## Cross-references

* CP-4 STOP doc:
  [`rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`](./rmsnorm-kt-tape-production-caller-stop-2026-05-28.md)
* CP-4 design context:
  [`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) §"Top 3
  next-tasks" #1 (line 209-218)
* Companion pilots: see the table above.
* Sibling test that proves the same recipe at the `kiln-autograd`
  boundary: `crates/kiln-autograd/tests/training_loop_descent.rs`.
* Implementation: `crates/kiln-train/src/tape_step.rs`.
