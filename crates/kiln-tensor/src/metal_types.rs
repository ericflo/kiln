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
//!
//! # Native objc2-metal substrate (#1082 Phase 7 lift, partial)
//!
//! The `Raw*` aliases below expose the direct `objc2_metal::MTL*`
//! protocol-object handles that sit underneath every
//! `candle_metal_kernels::metal::*` wrapper. They are the substrate
//! the chokepoint will eventually flip its existing aliases to point at
//! — adding them now (parallel to the candle re-exports) lets future
//! call-site migrations land on a kt-native objc2-metal path without
//! re-introducing a candle string anywhere on the import line. The
//! `Retained<ProtocolObject<dyn MTL*>>` shape is the same one
//! `candle_metal_kernels::metal::{ComputePipeline, Library, Buffer,
//! Device}` hold internally (via `.raw` fields backed by the same
//! `objc2-metal 0.3` crate), so passing a `Raw*` to `candle_metal_kernels::call_*`
//! works as soon as the candle wrapper exposes `From<Raw*>` conversions
//! (already in place for `Device` and `Buffer`; pipeline/library wrap
//! the same Retained internally).

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

/// Candle `MetalDevice` re-export — used by every pipeline-builder
/// helper in `kiln-model::backend::metal` as a function parameter
/// (`device: &kiln_tensor::metal_types::MetalDevice`).
///
/// The wire type is candle's `MetalDevice` wrapper; the kt-side path
/// lets the model crate stop naming `candle_core::metal_backend::*`
/// in its helper signatures so the eventual substrate lift to a raw
/// `MTLDevice` + `MTLCommandQueue` handle pair is a single edit
/// here, not 46 separate signature edits in `metal.rs`.
///
/// # Phase 7 follow-up
///
/// Substrate lift retires the candle wrapper itself — this re-export
/// then flips to a `kiln_tensor::MetalRawDevice` companion that holds
/// `(Retained<ProtocolObject<dyn MTLDevice>>,
///   Retained<ProtocolObject<dyn MTLCommandQueue>>)` directly. The
/// helper-function bodies (which only call `device.id()`,
/// `device.device()`, `device.command_buffer()`, etc.) get migrated
/// alongside that flip.
pub use candle_core::metal_backend::MetalDevice;

/// Candle `DeviceId` re-export — the `usize` newtype used as the cache
/// key in every per-device pipeline / library `HashMap` in the
/// `metal_*_pipeline` helpers. Same Phase-7 rationale as `MetalDevice`:
/// keep the wire type identical, move the path through `kiln_tensor`.
pub use candle_core::metal_backend::DeviceId;

/// Candle `Storage` enum re-export — pattern-matched in every kernel
/// helper to downcast `Storage::Metal(s) => s` and extract the
/// candle `MetalStorage` for the `.buffer()` / `.dtype()` accessors.
/// Phase 7 substrate-lift replaces the match arm with a direct
/// `kiln_tensor::MetalStorage` downcast off `kiln_tensor::Storage`
/// (already in flight on the kt-side); the chokepoint here is the
/// path-naming bookkeeping step.
pub use candle_core::Storage;

// `D` re-export retired (#1082): the production `kiln-model::backend::metal`
// call sites now compute the last-axis index via kt-native `tensor.rank() - 1`
// arithmetic, and the test helpers in that file import `candle_core::D`
// directly since they operate on `candle_core::Tensor` already.

/// Candle `sdpa` re-export — the MLX-style fused scaled-dot-product
/// attention kernel shipped in `candle_nn::ops::sdpa`. Used in 9
/// callsites across `kiln-model::backend::metal` (3 in the metal
/// SDPA dispatch + 6 in test/parity helpers).
///
/// Same Phase-7 chokepoint rationale as the other re-exports: keep
/// the wire signature identical, move the path through `kiln_tensor`
/// so callers stop naming `candle_nn::ops::sdpa` directly. The
/// substrate-lift step replaces this re-export with a kt-native
/// fused SDPA op that takes `kiln_tensor::Tensor` arguments.
pub use candle_nn::ops::sdpa;

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

// ----------------------------------------------------------------------
// Native objc2-metal substrate (#1082 Phase 7 — partial lift)
// ----------------------------------------------------------------------
//
// Every `candle_metal_kernels::metal::*` wrapper above ultimately holds
// a `Retained<ProtocolObject<dyn MTL*>>` from the `objc2-metal 0.3`
// crate. The `Raw*` aliases below expose those handles directly —
// they are the substrate the existing chokepoint aliases (`ComputePipeline`,
// `Library`, `MetalDevice`, etc.) will eventually flip to point at, once
// the kernel-crate FFI sites in `kiln-model::backend::metal` migrate off
// the candle wrapper signatures (`device.command_buffer()`,
// `device.kernels()`, `pipeline.set_compute_pipeline_state(...)`).
//
// Adding them here in parallel with the candle re-exports above does not
// break any existing call site — the candle aliases stay live — and gives
// future migration commits a kt-native objc2-metal path to land on
// without re-introducing a `candle_metal_kernels::metal::*` string
// anywhere on the import line. The eventual chokepoint flip replaces
// `pub use candle_metal_kernels::metal::ComputePipeline` with
// `pub type ComputePipeline = RawComputePipelineState`, etc.
//
// # Why the objc2-metal version is pinned to candle's
//
// `candle-metal-kernels 0.10.2` depends on `objc2-metal = "0.3.2"`.
// kiln-tensor's `metal` feature pulls the same `objc2-metal = "0.3"`,
// so cargo unifies on one version and the `Retained<ProtocolObject<dyn
// MTL*>>` types these aliases name are exactly the ones candle's
// wrappers hold internally. That bit-identity is what lets a future
// caller pass a `RawComputePipelineState` straight to
// `candle_metal_kernels::call_*` without a `From<>` shim — the wire
// type that crosses the FFI is the same Retained handle either way.

/// MSL compute-pipeline-state handle as a raw `objc2-metal` protocol
/// object. The eventual substrate for `ComputePipeline`.
///
/// This is the same Retained handle that
/// `candle_metal_kernels::metal::ComputePipeline` holds internally,
/// just reached without the candle wrapper layer. Future kernel-helper
/// migrations can pivot away from `pipeline.set_compute_pipeline_state(...)`
/// (a candle wrapper method) onto direct
/// `MTLComputeCommandEncoder::setComputePipelineState_` calls against
/// the `Retained` handle.
pub type RawComputePipelineState = objc2::rc::Retained<
    objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
>;

/// MSL library handle as a raw `objc2-metal` protocol object. The
/// eventual substrate for `Library`.
pub type RawLibrary =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLLibrary>>;

/// MTLBuffer handle as a raw `objc2-metal` protocol object. The
/// eventual substrate the existing `candle_metal_kernels::metal::Buffer`
/// wrapper flips to point at.
///
/// On Apple Silicon UMA this is the same Retained handle that backs
/// every `MetalStorage::buffer()` accessor today — flipping the
/// chokepoint here is a renaming, not a re-allocation.
pub type RawBuffer =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// MTLDevice handle as a raw `objc2-metal` protocol object. The
/// eventual substrate for `MetalDevice` once the kernel-helper
/// signatures retire the candle wrapper's `.command_buffer()` /
/// `.kernels()` accessors in favor of direct
/// `MTLCommandQueue::commandBuffer` + a kt-side `Kernels` cache.
pub type RawDevice =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>;

/// MTLCommandQueue handle as a raw `objc2-metal` protocol object.
/// The eventual companion that lands on `MetalStorage` alongside
/// `metal_handle: MetalRawDevice` so that the in-storage substrate
/// ops in `metal_storage.rs` can build `MTLCommandBuffer`s without
/// derivating a candle `MetalDevice` per call (the
/// `primary_metal_device` shim retires once this lands).
pub type RawCommandQueue =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>;
