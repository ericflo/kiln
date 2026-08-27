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
//! E1/E2). The remaining candle-API-specific aliases (`Shape`, `D`,
//! `CdResult`) and the candle constructor / safetensors shims stay
//! pinned to explicit `candle_core::*` as a candle island (see below).
//!
//! The handful of items that are intrinsically candle-API-specific —
//! the candle safetensors I/O shims and the candle generic-constructor
//! helpers (`NdArray` / `WithDType` bounds) — stay pinned to explicit
//! `candle_core::*` here as a **candle island**. They have no 1:1 kt
//! equivalent and their callers in `trainer.rs` are being migrated off
//! candle in the same wave (E1); this facade keeps compiling in the
//! interim, and the island is deleted once the last caller flips.
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
// keep: reserved for the Wave E4 kt facade — this module's #1082 E4 doc
// names `TensorId` as one of the four load-bearing aliases (alongside the
// live `Tensor`, `Device`, `DType`); deleting only this alias would break
// the documented facade invariant. The `cd_tensor_id_to_kt` bridge was
// retired the same wave because `TensorId` is already the kt id.
#[allow(dead_code)]
pub(crate) type TensorId = kiln_tensor::TensorId;

/// candle `Shape` / `D` — candle island (#1082). Still named in the
/// candle-authoritative helper signatures in `trainer.rs` (e.g.
/// `Into<Shape>` bounds on candle `zeros_*`, `D::Minus1` axis args on
/// candle ops). Bare kt code uses `kiln_tensor::{Shape, D}` directly;
/// these aliases stay candle-pinned until Wave E1 flips those helpers.
pub(crate) type Shape = kiln_tensor::Shape;
pub(crate) type D = kiln_tensor::D;

// (#1082) The `GradStore` candle-island alias
// (`candle_core::backprop::GradStore`) is GONE: Wave E1 routed every
// `.backward()` path in `trainer.rs` through the kt tape's
// `kiln_autograd::GradStore`, which is referenced fully-qualified at all
// call sites. No bare `GradStore` resolves to this facade anymore, so the
// alias had zero callers and was removed.
