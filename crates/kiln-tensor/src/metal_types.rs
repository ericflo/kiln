//! Candle-free type aliases for the metal-rs–style types used by every
//! kernel-crate FFI site in `kiln-model::backend::metal`.
//!
//! # Why this module exists (#1082 Phase 7)
//!
//! Roughly all `candle_metal_kernels::*` references in
//! `kiln-model::backend::metal` are simple uses of two type names —
//! `ComputePipeline` (the pipeline-state object) and `Library` (the
//! compiled MSL library handle). Each call site looks like:
//!
//! ```text
//! fn metal_X_pipeline(
//!     device: &candle_core::metal_backend::MetalDevice,
//! ) -> Result<candle_metal_kernels::metal::ComputePipeline> {
//!     use candle_metal_kernels::metal::ComputePipeline;
//!     ...
//! }
//! ```
//!
//! Funneling those through one re-export consolidates the
//! `candle_metal_kernels::*` surface area to a single chokepoint that
//! lives in `kiln-tensor`. Once every caller imports
//! `kiln_tensor::metal_types::{ComputePipeline, Library}`, the
//! underlying substrate can swap (e.g. to a direct `objc2_metal::*`
//! protocol-object pair, or to a non-candle `metal-rs` wrapper) without
//! touching the ~92 kernel-helper functions in `metal.rs`.
//!
//! The wire types are identical to candle's re-export — they ARE the
//! same `candle-metal-kernels::metal::{ComputePipeline, Library}`
//! types, just reached through a `kiln_tensor::` path so that the
//! kernel helpers no longer name `candle_metal_kernels` in their
//! signatures or bodies. This is the same bookkeeping pattern that
//! `kiln-tensor::metal_storage::metal_device_handle()` uses on the
//! storage side: keep the candle-shipped substrate in place for now
//! while the call-site footprint drops to zero.
//!
//! # Phase 7 follow-up
//!
//! When the kernel crates migrate off candle-metal-kernels'
//! pipeline/library abstractions (e.g. to objc2_metal directly), this
//! module flips its `pub use candle_metal_kernels::metal::*` line to
//! point at the new substrate. The metal.rs migration that converts
//! the existing 92 sites to `kiln_tensor::metal_types::*` is the
//! enabling step — until those imports are all centralized here, the
//! substrate swap must touch every helper individually.

#![cfg(feature = "metal")]

/// MSL compute-pipeline-state handle. Re-exported from candle's
/// in-house objc2 wrapper today; flips to a direct `objc2_metal`
/// `Retained<ProtocolObject<dyn MTLComputePipelineState>>` when the
/// kernel crates retire the candle bridge.
pub use candle_metal_kernels::metal::ComputePipeline;

/// MSL library handle. Re-exported from candle's in-house objc2
/// wrapper today; flips to a direct `objc2_metal`
/// `Retained<ProtocolObject<dyn MTLLibrary>>` when the kernel crates
/// retire the candle bridge.
pub use candle_metal_kernels::metal::Library;

/// `BufferOffset` — `{ buffer: &Buffer, offset_in_bytes: usize }`
/// pair consumed by every `candle_metal_kernels::call_*` MSL kernel
/// entry point as a positional argument.
///
/// Re-exported from `candle_metal_kernels::utils::BufferOffset` today;
/// when the kernel crates retire the candle bridge, this flips to a
/// kt-native struct holding `(Retained<ProtocolObject<dyn MTLBuffer>>,
/// usize)` without touching the ~232 call sites in
/// `kiln-model::backend::metal` that construct one.
pub use candle_metal_kernels::utils::BufferOffset;

/// Build a `BufferOffset` from a candle `Buffer` + `Layout` + `DType`.
///
/// This is a mirror of `candle_core::metal_backend::buffer_o` — same
/// formula, same wire types, but reachable through the `kiln_tensor`
/// path so that callers in `kiln-model::backend::metal` no longer
/// name `candle_core::metal_backend::*` in their bodies.
///
/// The formula is the candle-bundled one (`l.start_offset() *
/// dtype.size_in_bytes()`); inlining it here gives the same byte
/// offset that `candle_metal_kernels::call_*` kernels expect.
///
/// # Phase 7 follow-up
///
/// When the kernel crates retire the candle bridge, this helper
/// gains a kt-native overload that takes a `kiln_tensor::Layout` +
/// `kiln_tensor::DType` directly; the candle-typed signature stays
/// as the legacy entry point until every caller migrates.
///
/// # Why we re-implement rather than re-export
///
/// `candle_core::metal_backend::buffer_o` lives behind candle-core's
/// `metal` feature; re-exporting it through `pub use` would lock the
/// chokepoint to candle-core's API surface forever. Re-implementing
/// the trivial formula here keeps the chokepoint free to evolve.
#[inline]
pub fn buffer_o<'a>(
    buffer: &'a candle_metal_kernels::metal::Buffer,
    l: &candle_core::Layout,
    dtype: candle_core::DType,
) -> BufferOffset<'a> {
    BufferOffset {
        buffer,
        offset_in_bytes: l.start_offset() * dtype.size_in_bytes(),
    }
}
