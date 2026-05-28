# `kiln-train` `candle-core` Cargo dep still required — note (audit 2026-05-28 pm)

**Status:** Informational follow-up to commits `a86e9b12` (InjectTensorGradient
flip + struct delete) and the broader Option-2 substrate work that landed
between `e2f8723c` and `a86e9b12`.

**Issue:** `#1082` (candle-core removal epic).

## TL;DR

The dashboard ([`candle-removal-status-2026-05-28-pm.md`](./candle-removal-status-2026-05-28-pm.md))
described kiln-train's blocker as "**One production ref:** `impl
candle_core::CustomOp1 for InjectTensorGradient` in trainer.rs." That
was technically correct — the trait impl was the only `candle_core::*`
*direct* code reference in trainer.rs — but it understated the broader
crate-level dependency: `crates/kiln-train/src/cd_types.rs` is a
production module that holds `pub(crate)` type aliases for every
candle type the trainer uses:

```rust
pub(crate) type Tensor    = candle_core::Tensor;
pub(crate) type Var       = candle_core::Var;
pub(crate) type CdDevice  = candle_core::Device;
pub(crate) type DType     = candle_core::DType;
pub(crate) type Shape     = candle_core::Shape;
pub(crate) type GradStore = candle_core::backprop::GradStore;
pub(crate) type TensorId  = candle_core::TensorId;
pub(crate) type D         = candle_core::D;
pub(crate) type CdResult<T> = candle_core::Result<T>;
pub(crate) type CpuStorage = candle_core::CpuStorage;
#[cfg(feature = "cuda")]
pub(crate) type CudaStorage = candle_core::CudaStorage;
pub(crate) type Layout     = candle_core::Layout;
```

…plus the safetensors load/save shims and a `cd_bail!` macro that
wraps `::candle_core::bail!`.

The trainer.rs body uses these aliases everywhere (`let x: Tensor =
...`, `fn foo(device: &CdDevice) -> Result<Var>`, etc.). They're
*production* code paths, not dev-only. So as long as cd_types.rs
imports candle types in production, `candle-core` cannot move to
`[dev-dependencies]` in kiln-train.

## What landed (real progress)

The `a86e9b12` flip is real progress, despite the misleading dashboard
framing:

1. **Deleted the `impl candle_core::CustomOp1 for InjectTensorGradient`**
   in trainer.rs — the kt-tape replacement
   (`kiln_kt_bridge::inject_grad_shim::inject_gradient_via_shim`)
   is now the production path at all 6 call sites.
2. **Validated the Option-2 substrate end-to-end** with the
   intermediate-arg parity test (`dd2eb4f3`). The shim's bwd
   returning `upstream` directly is bit-equivalent to the historical
   `InjectTensorGradient::bwd` for every queryable `Var` (root + any
   downstream-of-arg Var in the candle graph).
3. **Hoisted the shim out of `tape_bridge` into `inject_grad_shim`**
   so non-cuda builds compile the flip too (`a6531830`).
4. **Made the kt-tape side-channel optional** in the cuda-only
   `inject_gradient_kt` adapter (`07afd64a`): no active tape is no
   longer an error.

## What's still required for kiln-train `candle-core` → dev-deps

The cd_types facade is the only remaining production seam. Two
plausible migration paths:

### Path A — replace cd_types aliases with kt-native types

`Tensor` → `kiln_tensor::Tensor`, `Var` → kt-native Var (substrate
exists), `Device` → `kiln_tensor::Device`, `DType` → `kiln_tensor::DType`,
etc. Every trainer.rs call site that names a cd_types alias would
need to migrate to the kt-side surface. This is a massive
mechanical migration — hundreds of call sites — and it's the same
work that has to happen anyway as part of the full Phase 7 close.

### Path B — keep candle-core in [dependencies], move just the specific cuda + nn pieces

Examine which candle-core features cd_types actually needs in
non-test code. If the production cd_types surface ONLY needs the
candle-core types (Tensor, Var, Device, DType, Layout) and not the
cuda/cpu storage backends or autograd, candle-core could
theoretically be feature-gated differently. But cd_types references
`backprop::GradStore` (autograd) and uses the safetensors helpers,
so this path doesn't materially shrink the dep.

### Recommendation

Neither path is a quick follow-up. Path A is the right endpoint but
needs to be planned as a multi-PR migration (sized similarly to
kiln-model's metal_types swap plan). For the dashboard, the
honest representation is:

> kiln-train candle-core `[dependencies]` dep is gated on
> migrating `cd_types::*` aliases from `candle_core::*` to
> `kiln_tensor::*`. Substrate (kt-native Tensor/Var/Device/DType)
> exists; this is a mechanical N-site migration similar in shape to
> the metal_types swap plan for kiln-tensor.

## Dashboard correction

The blocker row for kiln-train in
[`candle-removal-status-2026-05-28-pm.md`](./candle-removal-status-2026-05-28-pm.md)
should be updated to reflect this. The CP-4 InjectTensorGradient
deletion **is** complete — the dashboard's bullet 2 should mark ✅
and the kiln-train blocker text should pivot to "cd_types facade
migration".

## Cross-references

- [`inject-grad-flip-blocked-2026-05-28.md`](./inject-grad-flip-blocked-2026-05-28.md)
  — STOP doc that diagnosed the Option-1 substrate failure.
- Commit `e2f8723c` — Option 2 substrate revision.
- Commit `07afd64a` — IO mapping removal + no-tape soft-fail.
- Commit `a6531830` — shim hoist to non-cuda module.
- Commit `dd2eb4f3` — intermediate-arg parity test.
- Commit `a86e9b12` — 6-site flip + struct + impl delete.
