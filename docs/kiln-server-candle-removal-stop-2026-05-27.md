# `kiln-server` candle removal — STOP (audit 2026-05-27)

## TL;DR

`kiln-server`'s candle footprint **cannot be reduced in isolation**. Every
candle reference in the crate is plumbing for a `kiln-model` (or
`kiln-train`) public API whose signature requires `candle_core::Device`,
`candle_core::DType`, or `candle_core::Tensor`. Until Tier 3 of
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) — `kiln-model`'s
forward pass and `ModelRunner`/`GpuWeights` surface — migrates,
"removing candle from `kiln-server`" only relocates the same types one
hop up the stack.

This STOP documents the audit at `main@72339698` so the next attempt
doesn't repeat the trace. Pure-STOP is the right outcome here.

## What the resume task framed

> "`kiln-server` currently imports candle in: `src/device.rs`,
> `src/state.rs` (25 candle imports), `src/cli.rs`, … the easy bucket
> includes `candle_core::Device` references that map to a kiln-tensor
> `Device` type cleanly."

The framing's "25 candle imports in state.rs" is the line-reference
count, not the `use` statement count. The actual `use` statements
inventory at HEAD is below; all are forced by downstream API signatures.

## Audit at HEAD (`main@72339698`)

`grep -rn "candle_core\|candle_nn" crates/kiln-server/` returns
**61 references across 9 files**:

| File | candle refs | What forces candle |
| --- | --- | --- |
| `src/state.rs` | 25 | `ModelRunner` (taken by value in `AppState::new_real`); `PagedKvCache::new(_, _, _, _, DType, &Device, _)`; `runner.weights.embed_tokens.device()` returning `&candle_core::Device`; `LinearAttentionState::new(&ModelConfig, &candle_core::Device)`. |
| `src/bench.rs` | 19 | `runtime_backend::for_device(&candle_core::Device)`, `GpuWeights::from_model_weights(&_, &_, &candle_core::Device)`, every `cast` to `candle_core::DType::{BF16,F16,F32}` to match `GpuWeights` activation dtype, `model_forward(_, _, &GpuWeights, _, _, &mut LinearAttentionState, _)` taking candle tensors. |
| `src/main.rs` | 6 | `precompile_metal_custom_kernels(device: &candle_core::Device)` is gated on `matches!(device, candle_core::Device::Metal(_))` — `device` came from `device::select_device()`. |
| `src/api/completions.rs` | 6 | `log_softmax_last_dim(x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor>` — the prompt-logprobs path operates directly on candle tensors produced by `kiln_model::forward::model_forward`. |
| `src/device.rs` | 1 | `pub fn select_device() -> Result<candle_core::Device>` — return type is forced by every downstream consumer. |
| `examples/flce_phase_a_validation_bench.rs` | 1 | `select_device()` returns candle Device → `GpuWeights::from_model_weights(_, _, &device)`. |
| `examples/flce_preflight_bench.rs` | 1 | Same. |
| `examples/phase10_rmsnorm_bench.rs` | 1 | Same. |
| `tests/real_model_integration.rs` | 1 | Constructs candle Tensors directly to feed into `kiln_model` forward APIs that take `&Tensor`. |

There are exactly **6 `use candle_*` lines** in the crate, but the
remaining ~55 references are fully-qualified `candle_core::*` paths in
function signatures and `matches!(device, candle_core::Device::*)`
checks. Removing the `use` statements would not delete a single
dependency.

## Why nothing migrates cleanly today

**1. `select_device()`'s return type is the load-bearing one.** Every
caller (`main.rs` lines 678/818/819/837/840/855, `bench.rs:114`, all 3
examples) takes the returned candle Device and passes it straight into
a `kiln-model` constructor:

```rust
// crates/kiln-server/examples/flce_phase_a_validation_bench.rs:466
let device = kiln_server::device::select_device()?;
if matches!(device, Device::Cpu) { ... }
let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)?;
```

`GpuWeights::from_model_weights` is in `kiln-model` and takes
`&candle_core::Device`. Switching `select_device` to return
`kiln_tensor::Device` would require either (a) a translation layer that
re-creates a candle Device (and then we still depend on candle), or
(b) flipping `GpuWeights::from_model_weights` and every other consumer
to kt-typed Devices in the same PR — which is the Tier 3
`kiln-model` migration this STOP is downstream of.

**2. `runner.weights.embed_tokens.device()` is candle's own
accessor.** In `state.rs:1556`, `main.rs:677`, and
`completions.rs:3604`, `kiln-server` reads the Device off a candle
Tensor that lives inside `GpuWeights`. That accessor returns
`&candle_core::Device` because the underlying field is a candle
Tensor. Until `GpuWeights` is migrated off candle Tensors (Tier 3),
`kiln-server` is forced to handle candle Devices.

**3. `PagedKvCache::new` and `LinearAttentionState::new` are candle-
typed.** In `state.rs:3320-3327` and `completions.rs:3607`,
`kiln-server` constructs paged KV cache / linear-attn state
through `kiln-model` constructors that take `candle_core::DType` and
`&candle_core::Device`. Same Tier-3 dependency.

**4. `log_softmax_last_dim` operates on candle tensors.** In
`completions.rs:3583`, the prompt-logprobs path receives a candle
`Tensor` from `kiln_model::forward::model_forward(...)` (returns
candle Tensor) and runs `log_sum_exp`/`broadcast_as`/`to_dtype` on it.
That's not adapter glue — it's an in-place numerical computation that
would have to be rewritten against `kiln-tensor::ops::*` *and* swap
the upstream `model_forward` to return a kt-typed Tensor. Both halves
are Tier 3.

**5. The `bench.rs` `DType` `match` ladders are forced by
`GpuWeights`.** The five `match dtype { ... => candle_core::DType::BF16,
... }` ladders in `bench.rs` exist to convert the kt-native
`kiln_core::config::DType` into the candle DType that
`GpuWeights::cast_activations_to(...)` and friends require. Same
shape: until `GpuWeights` is kt-typed, these ladders stay.

## The temptation that doesn't pay off

A naive partial migration would look like:

```rust
// crates/kiln-server/src/device.rs
pub fn select_device() -> Result<kiln_tensor::Device> { ... }

// callers (main.rs, bench.rs, examples)
let kt_device = select_device()?;
let candle_device = kt_device_to_candle(&kt_device)?;  // new helper
let gpu_weights = GpuWeights::from_model_weights(&w, &cfg, &candle_device)?;
```

This **does not reduce the candle dependency**. It adds a new
translation helper (`kt_device_to_candle`) that itself imports
`candle_core`, leaves every downstream signature unchanged, and adds
boilerplate at every call site. The `Cargo.toml` line
`candle-core = { workspace = true }` would still be required by
`kiln-server` for the helper, and by every consumer of `GpuWeights`
for the API surface. The same logic applies to wrapping `DType`.

The only honest partial would be deleting `use candle_core::Device` /
`use candle_core::DType` and replacing each occurrence with a fully-
qualified `candle_core::Device` / `candle_core::DType`. That changes
zero behaviour, zero dependencies, and zero candle references — pure
churn.

## What unblocks this work

`kiln-server`'s candle removal becomes mechanical once the Tier 3
`kiln-model` migration in
[`CANDLE_REMOVAL_PLAN.md`](./CANDLE_REMOVAL_PLAN.md) lands:

1. **`GpuWeights` → `KtWeights` (or equivalent)** — embed_tokens,
   q/k/v/o projections, MLP weights all live as `kiln_tensor::Tensor`,
   exposing a `.device() -> kiln_tensor::Device` accessor.
2. **`ModelRunner::new(..., device: kiln_tensor::Device)`** — so
   `state.rs::new_real` takes a kt Device and the
   `is_metal_device`/`device_needs_inference_prewarm`/`runtime_used_vram_for_device`
   helpers match on `kiln_tensor::Device::Metal(_)` instead.
3. **`kiln_model::forward::model_forward` → kt-typed return** — so
   the prompt-logprobs path in `completions.rs` can drop
   `candle_core::Tensor` and use `kiln_tensor::Tensor` +
   `kiln_tensor::ops::log_softmax_last_axis` (or equivalent — this
   primitive may need to be added; see
   `bench-results/parity-tolerance.csv` for the parity surface).
4. **`PagedKvCache::new(..., dtype: kiln_tensor::DType, device:
   &kiln_tensor::Device, ...)`** — the only remaining state.rs
   plumbing is the `kiln_core::config::DType → kiln_tensor::DType`
   adapter (a trivial 3-arm match).
5. **`GpuWeights::from_model_weights(..., &kiln_tensor::Device)` +
   `runtime_backend::for_device(&kiln_tensor::Device)`** — unblocks
   `bench.rs` and the three examples in one pass.

Once those land, `kiln-server`'s migration is:

- `src/device.rs`: change return type from `candle_core::Device` to
  `kiln_tensor::Device`; the cfg-gated `kiln_tensor::cuda_is_available()`
  / `kiln_tensor::metal_is_available()` probes already exist and stay.
- `src/state.rs`: replace `use candle_core::DType;` with `use
  kiln_tensor::DType;`; replace 8 `candle_core::Device::*` matches with
  `kiln_tensor::Device::*`; replace the `DType::F32` arg to
  `PagedKvCache::new` with `kiln_tensor::DType::F32`.
- `src/bench.rs`: drop 5 `kiln_core::config::DType → candle_core::DType`
  ladders (replaced by a single `kt_dtype_from_kc(...)` helper); change
  `runtime_backend_for_bench` signature to `&kiln_tensor::Device`.
- `src/main.rs`: 6 `matches!(device, candle_core::Device::Metal(_))`
  checks flip to `kiln_tensor::Device::Metal(_)`.
- `src/api/completions.rs`: `log_softmax_last_dim` flips to
  `&kiln_tensor::Tensor`; the prompt-logprobs `to_dtype(F32)` flips to
  `kiln_tensor::DType::F32`.
- `tests/real_model_integration.rs` + 3 examples: candle Tensor / Device
  construction flips to kt equivalents.
- `Cargo.toml`: drop `candle-core = { workspace = true }` from both
  `[dependencies]` and `[dev-dependencies]`.

All of that is **one mechanical PR after Tier 3 lands**. Trying to do
any piece of it earlier just shuffles the dependency without reducing
it.

## Workspace cross-reference

| Crate | Tier in plan | Status |
| --- | --- | --- |
| `kiln-server` | downstream of Tier 3 | **this STOP** |
| `kiln-model` | Tier 3 | 40+ `try_kt_*` opt-in gates landed; `PagedKvCacheKt` partial; ~10 decode sites remain |
| `kiln-train` | downstream of Tier 3 | per-step `try_kt_*` migrations in progress; `OpdLossCustomOp::bwd` + `FlceCustomOp::bwd` already on kt bridge |
| `kiln-kt-bridge` | Tier 4 | by design last to migrate |

`kiln-server`'s 61 candle references will collapse to **zero** as a
direct consequence of Tier 3 closing — there is no preparatory work in
`kiln-server` that buys speedup of Tier 3.

## Decision

**Pure-STOP. No code changes in this pass.**

The audit confirms the situation at `main@72339698`. The next agent
attempting to "remove candle from kiln-server" should:

1. Re-read this STOP doc.
2. Re-run `grep -rn "candle_core\|candle_nn" crates/kiln-server/ | wc -l`.
   If the number is still ~60, Tier 3 hasn't closed — STOP applies.
3. If the number has dropped (because Tier 3 progressed), the
   migration above becomes mechanical; follow the "What unblocks this
   work" section in order.
4. Do **not** attempt a "swap `use` statements" cosmetic pass — it
   reduces zero dependencies and adds zero value.

This STOP follows the same pattern as
`docs/lora-bwd-kt-migration-stop-2026-05-27.md`: re-audit at HEAD,
confirm the prior tier sequencing, refuse to ship cosmetic churn.
