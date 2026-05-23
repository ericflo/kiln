//! kiln-tensor — in-house Tensor + Storage substrate for kiln.
//!
//! # Status
//!
//! **Phase 1.1 scaffold** — see [GitHub issue #1082][epic] for the full
//! migration plan. Today this crate ships only [`Error`], [`Result`],
//! and the [`bail!`] macro. Subsequent Phase 1 PRs add `DType`,
//! `TensorId`, `Layout`, `Storage`, and `Tensor`.
//!
//! # Public-API stability
//!
//! `kiln-tensor` is **internal-only — no semver commitment**. Other
//! `kiln-*` crates may break against minor version bumps until candle is
//! removed and the surface stabilizes (Phase 7 of #1082). Once Phase 7
//! lands and the migration is complete, the public API of this crate
//! freezes against the next major version bump.
//!
//! [epic]: https://github.com/ericflo/kiln/issues/1082

#![deny(missing_debug_implementations)]
#![warn(rust_2018_idioms)]

mod activation_registry;
mod determinism;
mod device;
mod device_op;
mod dtype;
mod element;
mod error;
mod layout;
pub mod ops;
pub mod profile;
pub mod safetensors;
mod storage;
mod tensor;
mod tensor_id;

#[cfg(feature = "cuda")]
mod cuda_storage;
#[cfg(feature = "metal")]
mod metal_storage;
#[cfg(feature = "vulkan")]
mod vulkan_storage;

pub use activation_registry::{
    selective_recompute_recommendation, Activation, ActivationId, ActivationKind, ActivationRef,
    OffloadPolicy, RecomputeRecommendation,
};
pub use determinism::{deterministic_enabled, Determinism, DeterministicCache, DETERMINISTIC_CACHED};
pub use device::{Backend, Device};
pub use device_op::{dispatch1, dispatch2, dispatch3, BackwardOp, DeviceOp1, DeviceOp2, DeviceOp3};
pub use dtype::DType;
pub use element::Element;
pub use error::{Error, Result};
pub use layout::Layout;
pub use storage::{cpu_zeros, CpuStorage, Storage, StorageBackend};
pub use tensor::Tensor;
pub use tensor_id::TensorId;

#[cfg(feature = "cuda")]
pub use cuda_storage::{cuda_zeros, CudaStorage};
#[cfg(feature = "metal")]
pub use metal_storage::{metal_zeros, MetalStorage};
#[cfg(feature = "vulkan")]
pub use vulkan_storage::{vulkan_zeros, VulkanStorage};
