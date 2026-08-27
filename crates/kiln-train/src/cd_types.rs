//! (#1082) Per-crate type facade for [`crate::trainer`] — Wave E4 of the
//! candle-drop.
//!
//! **Post-flip semantics (matches the rest of the workspace).** The
//! forward.rs type-flip established the codebase convention that bare
//! `Tensor` / `Device` / `DType` / `TensorId` mean **kt**
//! (`kiln_tensor::*`); candle now survives only as explicit
//! `candle_core::*` islands bridged via `kiln_kt_bridge`. This module
//! used to alias those bare names to `candle_core::*` so `trainer.rs`
//! could keep candle types under a facade. Wave E4 flips the
//! load-bearing aliases (`Tensor`, `Device`, `DType`, `TensorId`) to
//! **kt** types, in lockstep with the `trainer.rs` / `opd.rs`
//! Var→`kiln_param::Parameter` + candle-grad-removal migration (Wave
//! E1/E2). The candle-API-specific aliases (`Shape`, `D`, `CdResult`)
//! and the candle constructor / safetensors shims were all retired with
//! the candle drop: the aliases are kt-pinned and the shims are gone.
//!
//! The handful of items that were intrinsically candle-API-specific —
//! the candle safetensors I/O shims and the candle generic-constructor
//! helpers (`NdArray` / `WithDType` bounds) — are GONE: their callers in
//! `trainer.rs` migrated off candle (Wave E1), and the shims were deleted
//! with the candle drop.
//!
//! `Var` is gone: LoRA parameters migrate to `kiln_param::Parameter`
//! (the trainer holds `Option<(Parameter, Parameter)>` per projection),
//! so no facade alias for `candle_core::Var` is needed.
//!
//! `cd_tensor_id_to_kt` is gone: with `TensorId` now aliased to the
//! kt id directly (and `Parameter::tensor_id()` already returning a kt
//! id), the bridge is the identity function and every call site keys on
//! kt ids natively.

// ---------------------------------------------------------------------------
// (#1082) Wave E4 — bare type aliases now resolve to kt, matching the
// workspace-wide post-flip convention (`Tensor`/`Device`/`DType` = kt).
//
// These are NOT `use candle_*` imports — they are `pub(crate) type`
// aliases onto `kiln_tensor::*`.
// ---------------------------------------------------------------------------

pub(crate) type Tensor = kiln_tensor::Tensor;
pub(crate) type Device = kiln_tensor::Device;
pub(crate) type DType = kiln_tensor::DType;

/// kt `Shape` / `D` (#1082). Named in the helper signatures in
/// `trainer.rs` (e.g. `Into<Shape>` bounds on `zeros_*`, `D::Minus1` axis
/// args). Bare kt code uses `kiln_tensor::{Shape, D}` directly; these
/// aliases resolve to the same kt types.
pub(crate) type Shape = kiln_tensor::Shape;
pub(crate) type D = kiln_tensor::D;

// (#1082) The `GradStore` candle-island alias
// (`candle_core::backprop::GradStore`) is GONE: Wave E1 routed every
// `.backward()` path in `trainer.rs` through the kt tape's
// `kiln_autograd::GradStore`, which is referenced fully-qualified at all
// call sites. No bare `GradStore` resolves to this facade anymore, so the
// alias had zero callers and was removed.
