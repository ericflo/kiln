//! Typed errors for the capture / replay path.

use kiln_tensor::TensorId;

/// Errors a per-backend `CapturedGraph` impl may raise.
#[derive(thiserror::Error, Debug)]
pub enum CaptureError {
    /// Attempted to allocate during the capture window (anti-pattern:
    /// `Frozen` mode disallows allocations).
    #[error(
        "CaptureError: allocation attempted during frozen capture (op: {op:?}, \
         requested {requested_bytes} bytes). Pre-warm the allocator pool before \
         capture_session.begin()."
    )]
    AllocationDuringFreeze {
        op: String,
        requested_bytes: usize,
    },

    /// A pinned pointer referenced by the captured graph no longer
    /// resolves to a live allocation (anti-pattern: dangling pointer
    /// across capture/replay boundary).
    #[error(
        "CaptureError: pinned pointer for tensor {tensor_id:?} is no longer live; \
         the captured graph would dereference freed memory. Likely cause: the \
         Tensor was dropped between capture and replay."
    )]
    DanglingPointer { tensor_id: TensorId },

    /// Replay called on a graph that wasn't successfully captured.
    #[error("CaptureError: replay called on an uncaptured graph")]
    NotCaptured,

    /// Per-backend driver error (CUDA / Metal / Vulkan returned an
    /// error during `cuGraphLaunch` / `executeCommandBuffer` etc.).
    #[error("CaptureError: backend driver error: {0}")]
    Backend(String),

    /// Underlying kiln_tensor error.
    #[error("CaptureError: tensor: {0}")]
    Tensor(#[from] kiln_tensor::Error),
}
