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
//! load-bearing aliases (`Tensor`, `CdDevice`, `DType`, `TensorId`) to
//! **kt** types, in lockstep with the `trainer.rs` / `opd.rs`
//! Var→`kiln_param::Parameter` + candle-grad-removal migration (Wave
//! E1/E2). The remaining candle-API-specific aliases (`Shape`, `D`,
//! `GradStore`, `CdResult`) and the candle constructor / safetensors
//! shims stay pinned to explicit `candle_core::*` as a candle island
//! (see below).
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

use std::path::Path;

// ---------------------------------------------------------------------------
// (#1082) Wave E4 — bare type aliases now resolve to kt, matching the
// workspace-wide post-flip convention (`Tensor`/`Device`/`DType` = kt).
//
// These are NOT `use candle_*` imports — they are `pub(crate) type`
// aliases onto `kiln_tensor::*`.
// ---------------------------------------------------------------------------

pub(crate) type Tensor = kiln_tensor::Tensor;
pub(crate) type CdDevice = kiln_tensor::Device;
pub(crate) type DType = kiln_tensor::DType;
pub(crate) type TensorId = kiln_tensor::TensorId;

// ---------------------------------------------------------------------------
// (#1082) Candle island — safetensors I/O shims + generic constructors.
//
// These are intrinsically candle-API-specific (candle's `safetensors`
// module, candle's `NdArray` / `WithDType` trait bounds) and have no
// drop-in kt counterpart. They stay as explicit `candle_core::*` paths
// (a candle island) while their `trainer.rs` callers are migrated off
// candle in Wave E1. Delete once those callers flip.
//
// NOTE: `CdResult<T>` is candle's `Result`, used only by these shims.
// ---------------------------------------------------------------------------

/// candle `Result` — used by the candle-island safetensors shims below.
pub(crate) type CdResult<T> = candle_core::Result<T>;

/// candle `Shape` / `D` — candle island (#1082). Still named in the
/// candle-authoritative helper signatures in `trainer.rs` (e.g.
/// `Into<Shape>` bounds on candle `zeros_*`, `D::Minus1` axis args on
/// candle ops). Bare kt code uses `kiln_tensor::{Shape, D}` directly;
/// these aliases stay candle-pinned until Wave E1 flips those helpers.
pub(crate) type Shape = kiln_tensor::Shape;
pub(crate) type D = kiln_tensor::D;

/// candle backprop gradient store — candle island (#1082). The
/// candle-authoritative `.backward()` paths in `trainer.rs` still hand
/// these around (`grads: &GradStore`); Wave E1 routes those through the
/// kt tape's `kiln_autograd::GradStore` and this alias is deleted. The
/// kt grad store is referenced fully-qualified (`kiln_autograd::GradStore`)
/// at the already-migrated call sites, so there is no name clash.
pub(crate) type GradStore = candle_core::backprop::GradStore;

/// Allocate a candle Tensor from an in-memory `NdArray` value (scalar /
/// slice / array). Candle island (#1082) — `trainer.rs` E1 migrates the
/// caller off candle; the helper goes when the last caller flips.
#[inline]
pub(crate) fn tensor_new<A: candle_core::NdArray>(
    value: A,
    device: &candle_core::Device,
) -> anyhow::Result<candle_core::Tensor> {
    Ok(candle_core::Tensor::new(value, device)?)
}

/// Allocate a candle Tensor from a Vec + shape on `device`. Candle
/// island (#1082) — see [`tensor_new`].
#[inline]
pub(crate) fn tensor_from_vec<T: candle_core::WithDType, S: Into<candle_core::Shape>>(
    values: Vec<T>,
    shape: S,
    device: &candle_core::Device,
) -> anyhow::Result<candle_core::Tensor> {
    Ok(candle_core::Tensor::from_vec(values, shape, device)?)
}

/// Load a safetensors file into a HashMap<String, candle Tensor> on
/// `device`. Candle island (#1082) — adapter on-disk format; migrates
/// to `kt::safetensors::load_cpu` when the trainer's candle adapter I/O
/// flips.
#[inline]
pub(crate) fn safetensors_load_file(
    path: &Path,
    device: &candle_core::Device,
) -> CdResult<std::collections::HashMap<String, candle_core::Tensor>> {
    candle_core::safetensors::load(path, device)
}

/// Save a HashMap<String, candle Tensor> as a safetensors file at
/// `path`. Candle island (#1082) — see [`safetensors_load_file`].
#[inline]
pub(crate) fn safetensors_save_file(
    tensors: &std::collections::HashMap<String, candle_core::Tensor>,
    path: &Path,
) -> CdResult<()> {
    candle_core::safetensors::save(tensors, path)
}
