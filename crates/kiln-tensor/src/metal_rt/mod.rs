//! Kiln-owned objc2-metal substrate (`metal_rt`) — the candle-free
//! replacement for `candle_metal_kernels::metal` (#1082 final step).
//!
//! This module vendors candle-metal-kernels 0.10.2's thin objc2 wrapper
//! over the Apple Metal protocol objects — the only part of
//! `candle_metal_kernels` kiln still consumed after every compute kernel
//! (matmul / cast / unary / binary / softmax / rmsnorm / layernorm /
//! index_select / sdpa) was reimplemented as kiln-owned MSL in
//! [`crate::metal_matmul`] + [`crate::metal_kernels`].
//!
//! The vendored types are pure objc2 / objc2-metal / objc2-foundation
//! boilerplate (no candle, no MLX, no MSL): the [`Device`] / [`Buffer`] /
//! [`CommandQueue`] handles, the [`Commands`] command-buffer pool
//! (semaphore / recycle / batch-50 logic), [`ComputeCommandEncoder`] /
//! [`BlitCommandEncoder`], [`ComputePipeline`], [`Library`] / [`Function`]
//! / [`ConstantValues`]. The wire types are bit-identical to candle's
//! (both wrap `Retained<ProtocolObject<dyn MTL*>>` from the same
//! `objc2-metal 0.3.2` crate) — this is a vendor-and-rename, not a
//! reimplementation.
//!
//! candle's `MetalKernelError` is replaced by the local [`MetalRtError`]
//! (only the four variants the substrate actually constructs:
//! `FailedToCreateResource`, `LoadFunctionError`, `CommandBufferError`,
//! `LockError`). Callers in `kiln-tensor` map it to [`crate::Error::Msg`]
//! via its `Debug`/`Display` impl exactly as they did with candle's.
//!
//! Source: candle-metal-kernels 0.10.2 `src/metal/` (MIT/Apache-2.0).

#![cfg(feature = "metal")]
// Vendored substrate: the wrapper exposes the full objc2-metal surface
// (blit encoder, buffer-with-data, `enqueue`, `did_modify_range`, device
// classification, function constants, ...) even though `kiln-tensor` only
// drives a subset today. Keeping the complete API verbatim minimizes
// divergence from the known-good candle-metal-kernels source.
#![allow(dead_code)]

pub mod buffer;
pub mod command_buffer;
pub mod commands;
pub mod compute_pipeline;
pub mod device;
pub mod encoder;
pub mod indirect_command_buffer;
pub mod library;

pub use buffer::*;
pub use command_buffer::*;
pub use commands::*;
pub use compute_pipeline::*;
pub use device::*;
pub use encoder::*;
pub use indirect_command_buffer::*;
pub use library::*;

/// Substrate-level error for the vendored objc2-metal wrapper.
///
/// Mirrors the subset of `candle_metal_kernels::MetalKernelError` that the
/// vendored Device / Library / Commands paths actually construct. Callers
/// map it into [`crate::Error::Msg`] via its `Debug` formatting (the same
/// `{e:?}` shape they used against candle's error).
#[derive(thiserror::Error, Debug)]
pub enum MetalRtError {
    /// A `MTLCommandBuffer` finished in the `Error` status.
    #[error("Command buffer had following error: {0}")]
    CommandBufferError(String),
    /// A `Mutex`/`RwLock` guarding pool state was poisoned.
    #[error("Could not lock resource: {0}")]
    LockError(String),
    /// `MTLLibrary::newFunctionWithName` returned nil / errored.
    #[error("Error while loading function: {0}")]
    LoadFunctionError(String),
    /// A `MTLDevice`/`MTLCommandQueue` returned nil from a
    /// resource-creation call (buffer, command buffer, queue).
    #[error("Failed to create metal resource: {0}")]
    FailedToCreateResource(String),
}

impl<T> From<std::sync::PoisonError<T>> for MetalRtError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}
