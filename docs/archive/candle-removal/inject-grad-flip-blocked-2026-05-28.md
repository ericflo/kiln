# STOP: `InjectTensorGradient` call-site flip blocked on substrate design

**Status:** Blocked — substrate (`9b2eda8e`) is incompatible with the
production call-site pattern. Caller flip + struct deletion cannot
proceed without a substrate revision.

**Issue:** `#1082` (candle-core removal epic, CP-4 step 4).

**Author:** Cloud Eric, 2026-05-28.

**Predecessor docs:**
- [`candle-removal-status-2026-05-28-pm.md`](./candle-removal-status-2026-05-28-pm.md) (dashboard)
- Substrate commit `9b2eda8e` ("kiln-kt-bridge: kt-tape substrate for
  InjectTensorGradient replacement").
- SFT bridge wrap precedent: `675e0dea`.

## TL;DR

The substrate built in `9b2eda8e` produces a **bit-equivalent** grad for
`arg.id()` in the candle `GradStore` AFTER `loss.backward()` returns
(via `insert_or_add_by_raw` running post-backward inside
`with_tape_scope_emit_to_grad_store`). The parity test
(`crates/kiln-train/tests/inject_gradient_parity.rs`) passes because it
uses `arg = Var::from_tensor(...)` — the arg IS the leaf, so candle's
backward walk terminates at `arg` and the post-hoc update is the final
value the test reads.

The production call sites in `trainer.rs:8068, 8220, 8366, 8376, 8446,
8454` use `arg = <intermediate tensor produced by candle ops above
seg_input_var>`. For these sites, candle's backward walk must
**propagate** the grad for `arg.id()` through the upstream candle ops
to the actual `Var` leaves (`pre_o_tile_var`, `seg_input_var`, etc.).
The walk consumes whatever value `grads[arg.id()]` holds **during**
the walk. The substrate's shim returns `zeros_like(arg)` from
`CustomOp1::bwd`, so candle propagates zeros to every upstream Var.
The post-hoc `insert_or_add_by_raw` then sets `grads[arg.id()] =
upstream_tile`, but by then the upstream Vars' grads are already wrong
(derived from zeros).

This is a wiring incompatibility, not a numerical one. It also explains
why the parity test passed without exercising this path: the test
queries `grads.get(arg_var)` directly (arg IS the only Var in scope),
while every production call site queries `grads.get(pre_o_tile_var)` /
`grads.get(seg_input_var)` (downstream-of-arg leaves).

## Reproduction recipe

Read each call site in `crates/kiln-train/src/trainer.rs`:

- **`full_attention_single_layer_tiled_mlp_reverse`** (line 7869),
  six `apply_op1` sites:
  - `out_proj_tile.apply_op1(...)` at L8068 — backward reads
    `grads.get(pre_o_tile_var)` at L8079.
  - `pre_o.apply_op1(...)` at L8220 — backward reads
    `grads.get(q_var)`, `grads.get(k_var)`, `grads.get(v_var)`,
    `grads.get(gate_var)` at L8229 / L8240 / L8253 / L8259.
  - `q.apply_op1(...)` at L8366 and `gate.apply_op1(...)` at L8376 —
    backward reads `grads.get(seg_input_var)` at L8394 (after combining
    two inject terms via `+`).
  - `k.apply_op1(...)` at L8446 and `v.apply_op1(...)` at L8454 —
    backward reads `grads.get(seg_input_var)` at L8472 (after combining
    two inject terms via `+`).

In every case the **inject site's arg** (`out_proj_tile`, `pre_o`,
`q`, `gate`, `k`, `v`) is an intermediate, and the downstream code
reads grad against a Var that is an ANCESTOR of arg in the candle
graph. That ancestor's grad is computed by candle backward walking
**through** the inject site's arg.

Candle backprop loop (vendored at `vendor/candle-core/src/backprop.rs`
L644-648):

```rust
Op::CustomOp1(arg, c) => {
    if let Some(arg_grad) = c.bwd(arg, node, &grad)? {
        grads.insert_or_add(arg, arg_grad)?
    }
}
```

Then the loop iterates to the **next** node (in topological order),
which is whatever op produced `arg`. That op's bwd is called with
`grad = grads.get(arg)?` — i.e., the value `bwd` just inserted. If
`bwd` inserted zeros (as the shim does), the upstream walk uses
zeros, and the final Var grads are zeros.

The substrate's post-hoc `insert_or_add_by_raw` (`tape_bridge.rs`
L458-503) runs **after** `loss.backward()` returns
(`with_tape_scope_emit_to_grad_store`, L327-419), so the upstream
walk has already completed by the time `grads[arg.id()]` gets its
real value.

## Why the parity test missed this

```rust
fn run_candle_baseline(arg_var: &Var, upstream: &Tensor) -> GradStore {
    let injected = arg_var
        .as_tensor()
        .apply_op1(InjectTensorGradientRef { upstream: upstream.clone() })
        .expect("apply_op1");
    injected.backward().expect("candle backward")
}
```

`arg_var.as_tensor()` IS a Var-tracked tensor; backward walks scalar
→ CustomOp1 bwd → `insert_or_add(arg_var.id(), grad)`. There is no
upstream op above `arg_var` in the test's graph, so the walk
terminates here. The post-hoc `insert_or_add_by_raw` running after
backward yields the same final value as a direct `Some(upstream)`
return.

The test asserts:

```rust
let grad_b = grads_b
    .get(arg_var_b.as_tensor())  // <-- arg_var, not an upstream Var
    .expect("kt-tape bridge must produce a grad for arg")
    .clone();
```

i.e. it reads grad against the SAME tensor that was the inject arg.
This is precisely the case where the substrate's post-hoc-mutation
strategy is bit-equivalent to a direct bwd return. It does not cover
the production case where the inject arg is upstream of the queried
Var.

## What a working substrate needs

Two viable paths, both substrate-level changes (not trainer.rs
changes):

### Option 1: shim returns the real grad

Make `InjectGradientCandleShim::bwd` return the actual injected
upstream grad instead of zeros. The bridge can record an
`InjectGradientBackward` on the kt tape for parity coverage but does
NOT need to splice the value back via `insert_or_add_by_raw` — candle
already has the right value in `grads[arg.id()]` after the shim
returns, and the upstream walk will propagate it correctly.

The candle adapter would essentially become a thin shim equivalent to
the existing `InjectTensorGradient`, plus a kt-tape recording for
audit/debug purposes. That kills the bridge's "tape walk + IO mapping
inserts" complexity for this op but keeps the kt-side recording for
future trainer flips.

**Tradeoff:** the bridge no longer drives any real kt-side
computation for this op — it's a candle-typed op with a kt-side
debug trace. That's likely fine for the immediate goal (delete the
trainer.rs candle CustomOp1 impl) but doesn't move CP-4 forward in
the "kt does the math" direction.

### Option 2: inject_gradient_kt drives backward directly

Make `inject_gradient_kt` return a candle Tensor whose `.backward()`
seeds the upstream walk correctly. One way: have the candle adapter
itself be a `CustomOp1` whose `bwd` returns the precomputed upstream
grad (i.e. exactly what `InjectTensorGradient` does), and route the
kt-tape record + grad emission as a side-effect (for parity testing
+ future migration tracking).

This is essentially "make inject_gradient_kt a drop-in API
replacement for `InjectTensorGradient`, with kt-tape recording as a
side channel." Migrates trainer.rs cleanly but the candle adapter
keeps the candle bwd contract — which means a candle `CustomOp1`
impl still lives **in kiln-kt-bridge**, just no longer in
**kiln-train**.

**Tradeoff:** the candle CustomOp1 impl moves from kiln-train to
kiln-kt-bridge. kiln-train's last production candle_core ref goes
away (real progress on the per-crate dashboard). kiln-kt-bridge is
already the by-design candle-keeper, so this is the natural home for
the shim.

### Recommendation: Option 2

Option 2 is the cleaner endpoint. The dashboard categorises
kiln-kt-bridge as "Yes (by-design)" — a Tier-5 deletion target after
CP-4 closeout. Adding a candle `CustomOp1` impl there is on-brand for
the bridge's role. The candle-typed adapter's API stays
`inject_gradient_kt(arg, upstream) -> Result<CandleTensor>`; the
implementation flips from "shim + post-hoc insert" to
"InjectTensorGradient-equivalent + kt-tape side-record".

## What this PR was supposed to do

From the task description:

> Flip the two `InjectTensorGradient::apply_op1` call sites in
> `crates/kiln-train/src/trainer.rs` (lines 8068 and 8220) to use
> `kiln_kt_bridge::tape_bridge::inject_gradient_kt`, then delete the
> `InjectTensorGradient` struct + `impl candle_core::CustomOp1`.

Two task-description correctness issues caught during investigation:

1. **There are six call sites**, not two. All six are inside
   `full_attention_single_layer_tiled_mlp_reverse` (lines 8068, 8220,
   8366, 8376, 8446, 8454). The substrate commit's own message says
   "six call sites are untouched." The task description and the
   dashboard ("2 sites at trainer.rs:8068, 8220") both undercount.
2. **The struct cannot be deleted** without flipping all six sites
   (compilation fails otherwise). Even with substrate fixes, this is
   a one-shot delete.

## What changes in this PR

This PR only adds this STOP doc — no source changes. Specifically NOT
done:

- No `InjectTensorGradient` deletion (struct + impl remain at
  `trainer.rs:7795-7866`).
- No call-site rewrite.
- No substrate change to `kiln-kt-bridge::tape_bridge`.

The unblocking work belongs in a follow-up PR that:

1. Picks Option 1 or Option 2 above (or proposes a third).
2. Implements the substrate change with a new parity test that
   exercises the upstream-Var-grad case (arg is intermediate).
3. Then does the trainer.rs flip across all six sites.
4. Then deletes the struct.

## Cost protection

This investigation was done locally on the read-only
`/workspace/sessions/...` clone — **zero RunPod spend**. The previous
two crashed sessions on this task accumulated cost without producing
working code; this STOP doc is the cheapest path to unblock the next
agent.

## Related notes

- `kernel-vendor-precondition-check` — preflight before pod spend.
- `phase6-kernel-vendor-preflight-pattern` — doc-only redirect PR
  pattern.

The pattern here is similar to phase6's kernel-vendor doc-only
redirect PRs (#131, #163, #164, #170): we landed a $0 doc PR
explaining why the substrate work needs to change before the caller
flip can land. This protects the next agent from re-discovering the
same wiring issue at pod-cost.
