//! `kiln_tensor::TensorId` — re-export of the canonical
//! [`kiln_tensor_id::TensorId`].
//!
//! The implementation moved to the leaf `kiln-tensor-id` crate so that
//! `kiln-vulkan-kernel` can also depend on it without creating the
//! `kiln-tensor -> kiln-vulkan-kernel -> kiln-tensor` cargo path-dep
//! cycle. This re-export preserves the existing `kiln_tensor::TensorId`
//! import path so all downstream callers (kiln-model, kiln-train,
//! kiln-autograd, kiln-kt-bridge, etc.) keep compiling unchanged.
//!
//! New code should prefer `kiln_tensor_id::TensorId` directly when the
//! caller doesn't otherwise need any of the storage / tensor types from
//! `kiln-tensor`.

pub use kiln_tensor_id::TensorId;
