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
// (#1082) Removed dead aliases `CpuStorage`, `CudaStorage`, `Layout`:
// every production use of these types in `kiln-train` already imports
// the kt-native counterpart directly (`kiln_tensor::{CpuStorage,
// CudaStorage, Layout}`), so the cd_types facade no longer needs to
// re-expose them. This is the first pilot step in the
// `cd_types::* -> kiln_tensor::*` migration tracked in
// `docs/kiln-train-candle-core-deps-still-required-2026-05-28.md`.
// Note: candle's `CustomOp1` is a trait and cannot be type-aliased on
// stable Rust without the `trait_alias` feature, so the lone `impl
// candle_core::CustomOp1 for InjectTensorGradient` block in `trainer.rs`
// keeps the full path inline.

// ---------------------------------------------------------------------------
// (#1082) `TensorId` migration substrate — second pilot in the
// `cd_types::* -> kiln_tensor::*` migration that began with the
// `CpuStorage` / `CudaStorage` / `Layout` pruning above.
//
// The kt-native `kiln_tensor_id::TensorId` (wraps `u64`) is the migration
// target for the legacy `TensorId = candle_core::TensorId` alias (wraps
// `usize`). Both types are value-equal, API-compatible identity
// wrappers; the kt side is already in production use in `vk_train.rs`
// (LoRA / boundary-leaf parameter ids) and `tape_step.rs` (kt-tape
// gradient store keys), and the candle side is what `Tensor::id()`
// currently returns inside `trainer.rs` / `cuda_train.rs` / `opd.rs`
// (~30 call sites collectively).
//
// **This PR is substrate-only.** The legacy `TensorId` alias above is
// intentionally unchanged so the existing ~30 `cd_types::TensorId` call
// sites in `trainer.rs` / `cuda_train.rs` / `opd.rs` keep compiling.
// Future PRs migrate individual call sites by inserting
// `cd_tensor_id_to_kt(tensor.id())` at the candle-graph boundary; once
// every site is flipped, the candle alias is deleted and `KtTensorId`
// is renamed back to `TensorId`.
//
// Migration sketch for a future call site:
//
// ```rust,ignore
// // Before:
// use crate::cd_types::{Tensor, TensorId};
// let mut moments: HashMap<TensorId, AdamWMoments> = HashMap::new();
// moments.insert(var.as_tensor().id(), AdamWMoments::default());
//
// // After:
// use crate::cd_types::{Tensor, KtTensorId, cd_tensor_id_to_kt};
// let mut moments: HashMap<KtTensorId, AdamWMoments> = HashMap::new();
// moments.insert(
//     cd_tensor_id_to_kt(var.as_tensor().id()),
//     AdamWMoments::default(),
// );
// ```
// ---------------------------------------------------------------------------

/// kt-native `TensorId` alias — the migration target for the legacy
/// [`TensorId`] alias above.
///
/// Identical to `kiln_tensor::TensorId` (`kiln-tensor` re-exports
/// `kiln_tensor_id::TensorId` from its leaf module). Carried in
/// `cd_types` so call sites stay rooted in the facade module while the
/// migration is in flight; once the legacy [`TensorId`] alias is
/// deleted, this alias gets renamed back to `TensorId`.
pub(crate) type KtTensorId = kiln_tensor_id::TensorId;

/// Bridge a candle `Tensor::id()` return value into the kt-native
/// `kiln_tensor_id::TensorId` space.
///
/// The two id types are stable, opaque, value-equal identity wrappers
/// (candle wraps `usize`, kt wraps `u64`). A `usize -> u64` widening is
/// safe on every target kiln supports (64-bit hosts; CUDA / Metal /
/// Vulkan / CPU all assume 64-bit pointers).
///
/// Used at the candle-graph boundary in the `TensorId` migration: any
/// `HashMap<TensorId, _>` keyed on `tensor.id()` can switch to
/// `HashMap<KtTensorId, _>` by routing the key through this helper.
///
/// The conversion preserves equality: two candle `TensorId` values
/// compare equal iff their bridged `KtTensorId` values compare equal
/// (each candle id maps to exactly one kt id under `from_raw(... as
/// u64)`, and a `usize -> u64` widening is injective on 64-bit hosts).
/// Cross-space collisions with ids minted via `KtTensorId::next()`
/// elsewhere in the process are not relevant here — the bridged value
/// is only ever compared against other bridged values inside the same
/// `HashMap<KtTensorId, _>` instance that a migrated call site owns.
#[inline]
pub(crate) fn cd_tensor_id_to_kt(id: TensorId) -> KtTensorId {
    // `candle_core::TensorId::as_raw()` returns the raw `usize`; the
    // `as u64` cast is the only conversion needed (all kiln targets
    // are 64-bit). `KtTensorId::from_raw` is a `const fn` round-trip
    // helper documented as appropriate for serialization / id
    // bridging (see `kiln-tensor-id/src/lib.rs` `from_raw` doc).
    KtTensorId::from_raw(id.as_raw() as u64)
}

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
// (#1082) The `cd_bail!` macro shim (wrapping `candle_core::bail!`) was
// removed: its sole call site was the `InjectTensorGradient::bwd` candle
// `CustomOp1` impl in `trainer.rs`, which the forward.rs type-flip deleted in
// favour of `kiln_kt_bridge::inject_grad_shim::inject_gradient_via_shim`. With
// no remaining call sites the macro + its `pub(crate) use` were dead (unused-
// macro / unused-import warnings), so they are gone. Re-add a kt-native bail
// shim here if a future kt-tape adapter needs one.
// ---------------------------------------------------------------------------
// (#1082) Tests for the `cd_tensor_id_to_kt` migration substrate.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `cd_tensor_id_to_kt` preserves equality: any two candle ids that
    /// compare equal map to two kt ids that also compare equal, and
    /// any two distinct candle ids map to two distinct kt ids. This is
    /// the substrate guarantee callers rely on when swapping
    /// `HashMap<TensorId, _>` for `HashMap<KtTensorId, _>`.
    #[test]
    fn cd_tensor_id_to_kt_preserves_equality() {
        // candle's `TensorId::new()` is private (`fn new` in
        // `vendor/candle-core/src/tensor.rs:14`); the only way to mint
        // one in a test is to create a tensor and pull `tensor.id()`.
        // Doing that requires a device + dtype, which would drag CPU
        // device construction into a leaf-module test. Instead, we
        // exercise the conversion via candle Tensors that we know
        // share / don't share an id (Tensor::clone preserves the
        // TensorId, Tensor::new mints a fresh one — see
        // `vendor/candle-core/src/tensor.rs:175,2080`).
        use candle_core::{Device, Tensor as CdTensor};
        let device = Device::Cpu;
        let a = CdTensor::new(&[1.0_f32, 2.0, 3.0], &device).unwrap();
        let a_clone = a.clone(); // Arc-clone: id is preserved.
        let b = CdTensor::new(&[4.0_f32, 5.0, 6.0], &device).unwrap();

        // Aliasing pair: candle equal -> kt equal.
        assert_eq!(a.id(), a_clone.id());
        assert_eq!(cd_tensor_id_to_kt(a.id()), cd_tensor_id_to_kt(a_clone.id()));

        // Distinct pair: candle distinct -> kt distinct.
        assert_ne!(a.id(), b.id());
        assert_ne!(cd_tensor_id_to_kt(a.id()), cd_tensor_id_to_kt(b.id()));
    }

    /// Bridged ids are stable under repeated conversion: calling
    /// `cd_tensor_id_to_kt` twice on the same candle id yields the
    /// same kt id (i.e. the helper is pure / deterministic, not
    /// minting fresh ids on each call).
    #[test]
    fn cd_tensor_id_to_kt_is_deterministic() {
        use candle_core::{Device, Tensor as CdTensor};
        let device = Device::Cpu;
        let t = CdTensor::new(&[1.0_f32], &device).unwrap();
        let id = t.id();
        let kt_a = cd_tensor_id_to_kt(id);
        let kt_b = cd_tensor_id_to_kt(id);
        assert_eq!(kt_a, kt_b);
        // And the raw payload matches the documented `usize -> u64`
        // widening contract.
        assert_eq!(kt_a.as_raw(), id.as_raw() as u64);
    }
}
