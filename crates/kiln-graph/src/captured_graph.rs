//! `CapturedGraph` — the per-backend trait every capture impl implements.

use kiln_tensor::Backend;

use crate::CaptureError;

/// One captured graph, ready to be replayed.
///
/// Implementations:
///
/// - `kiln-graph-cuda::CudaCapturedGraph` (Phase 5.x) — wraps
///   `cudarc::driver::CudaGraphExec`.
/// - `kiln-graph-metal::MetalCapturedGraph` (Phase 5.x) — wraps
///   `MTLIndirectCommandBuffer`.
/// - `kiln-graph-vulkan::VulkanCapturedGraph` (Phase 5.x) — extends
///   `kiln-vulkan-kernel::cmd_batch.rs`.
pub trait CapturedGraph: Send + Sync + std::fmt::Debug {
    /// Stable backend tag.
    fn backend(&self) -> Backend;

    /// Replay the captured commands. Returns `CaptureError` on per-
    /// backend driver failure or dangling-pointer detection (under
    /// debug builds).
    fn replay(&self) -> Result<(), CaptureError>;

    /// Number of times `replay()` has been called on this instance.
    /// Used by `bench-results/` reports to attribute captured-graph
    /// runtime cost.
    fn replay_count(&self) -> u64;

    /// Estimated VRAM footprint of this graph's scratch pool (the
    /// pre-warmed slab indexed by tensor handle). Reported by the
    /// Phase 5 `bench-results/graph-family-vram.csv`.
    fn scratch_bytes(&self) -> usize;
}
