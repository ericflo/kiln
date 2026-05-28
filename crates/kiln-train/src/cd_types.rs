//! (#1082) Centralised candle facade for [`crate::trainer`].
//!
//! This module is the single place that mentions `candle_core::` paths on
//! behalf of `trainer.rs`. Every type alias, generic constructor helper,
//! and safetensors I/O shim that previously embedded `candle_core::` in
//! `trainer.rs` now lives here. Re-exporting these via `pub(crate) use
//! crate::cd_types::*;` lets `trainer.rs` reach zero direct `candle_core::`
//! references for everything except the one item that genuinely cannot be
//! pulled across a module boundary cleanly: the `impl candle_core::CustomOp1
//! for InjectTensorGradient` block, whose trait impl must live next to the
//! struct.
//!
//! Audit invariants this module preserves:
//!   * `trainer.rs` has zero `use candle_*` imports at module top — only
//!     `use crate::cd_types::*;`.
//!   * All candle paths are confined to this file plus the single
//!     `CustomOp1` trait impl and its cd_bail macro in `trainer.rs`.
//!   * The `cd_bail!` macro is hoisted to the crate root via
//!     `#[macro_export]` so call sites keep writing `cd_bail!(...)`.

use std::path::Path;

// ---------------------------------------------------------------------------
// Type aliases (moved from trainer.rs, formerly lines 451..=462).
//
// These are NOT `use candle_*` imports — they are `pub(crate) type` aliases
// local to the kiln-train crate.
// ---------------------------------------------------------------------------

pub(crate) type Tensor = candle_core::Tensor;
pub(crate) type Var = candle_core::Var;
pub(crate) type CdDevice = candle_core::Device;
pub(crate) type DType = candle_core::DType;
pub(crate) type Shape = candle_core::Shape;
pub(crate) type GradStore = candle_core::backprop::GradStore;
pub(crate) type TensorId = candle_core::TensorId;
pub(crate) type D = candle_core::D;
pub(crate) type CdResult<T> = candle_core::Result<T>;
pub(crate) type CpuStorage = candle_core::CpuStorage;
// `CudaStorage` is only consumed inside `#[cfg(feature = "cuda")]`
// `cuda_fwd` in trainer.rs, so the alias is dead code without the
// cuda feature on. Suppress that warning rather than cfg-gating the
// alias itself (cfg-gating leaks `cfg(feature = "cuda")` into trainer.rs).
#[allow(dead_code)]
pub(crate) type CudaStorage = candle_core::CudaStorage;
pub(crate) type Layout = candle_core::Layout;
// Note: candle's `CustomOp1` is a trait and cannot be type-aliased on
// stable Rust without the `trait_alias` feature, so the lone `impl
// candle_core::CustomOp1 for InjectTensorGradient` block in `trainer.rs`
// keeps the full path inline.

// ---------------------------------------------------------------------------
// Generic constructor helpers (moved from trainer.rs, formerly lines 333
// and 344). These keep `candle_core::NdArray` / `candle_core::WithDType`
// bounds confined to this file.
// ---------------------------------------------------------------------------

/// Allocate a candle Tensor from an in-memory `NdArray` value (scalar /
/// slice / array). Consolidates the `Tensor::new(value, device)`
/// constructor (~58 sites pre-consolidation).
#[inline]
pub(crate) fn tensor_new<A: candle_core::NdArray>(
    value: A,
    device: &CdDevice,
) -> anyhow::Result<Tensor> {
    Ok(Tensor::new(value, device)?)
}

/// Allocate a candle Tensor from a Vec + shape on `device`. Consolidates
/// the `Tensor::from_vec(values, shape, device)` constructor
/// (~25 sites pre-consolidation).
#[inline]
pub(crate) fn tensor_from_vec<T: candle_core::WithDType, S: Into<Shape>>(
    values: Vec<T>,
    shape: S,
    device: &CdDevice,
) -> anyhow::Result<Tensor> {
    Ok(Tensor::from_vec(values, shape, device)?)
}

// ---------------------------------------------------------------------------
// safetensors I/O shims (moved from trainer.rs, formerly lines 471 and 482).
// Confines `candle_core::safetensors::{load,save}` to this file.
// ---------------------------------------------------------------------------

/// Load a safetensors file into a HashMap<String, Tensor> on `device`.
/// Consolidates the candle safetensors::load(path, device) call site
/// (~4 sites in adapter I/O + tests).
#[inline]
pub(crate) fn safetensors_load_file(
    path: &Path,
    device: &CdDevice,
) -> CdResult<std::collections::HashMap<String, Tensor>> {
    candle_core::safetensors::load(path, device)
}

/// Save a HashMap<String, Tensor> as a safetensors file at `path`.
/// Consolidates the candle safetensors::save(tensors, path) call site
/// (~1 site in adapter I/O).
#[inline]
pub(crate) fn safetensors_save_file(
    tensors: &std::collections::HashMap<String, Tensor>,
    path: &Path,
) -> CdResult<()> {
    candle_core::safetensors::save(tensors, path)
}

// ---------------------------------------------------------------------------
// Macro shim for `candle_core::bail!`. Confines the candle bail path to
// this file. Re-exported via `pub(crate) use cd_bail` so call sites can
// `use crate::cd_types::cd_bail;` and keep writing `cd_bail!(...)`.
// ---------------------------------------------------------------------------

/// Helper macro shim wrapping candle's `bail!`. Lets call sites write
/// `cd_bail!(...)` instead of the full candle path.
macro_rules! cd_bail {
    ($($t:tt)*) => { ::candle_core::bail!($($t)*) };
}
pub(crate) use cd_bail;
