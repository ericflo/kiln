//! Three-mode allocator policy per the Phase 1 issue bullet.
//!
//! > **Allocator design — three modes, one interface.**
//! > `kiln-tensor::Allocator` has three explicit modes: (1) `Owned`
//! > (per-allocation `cudaMalloc` / `vkAllocateMemory` — slow, used
//! > only at startup); (2) `Pool` (slab-allocated from a pre-sized
//! > device-local pool, the steady-state path; mirrors Vulkan's
//! > `buffer_pool.rs` and `decode_resident_pool.rs`); (3) `Frozen`
//! > (no allocation; pulls from a pre-warmed slab indexed by
//! > tensor-handle — used during `capture()` … `replay()`).

/// Three-mode allocator policy.
///
/// Used by:
/// - kiln-tensor's storage allocator (`CudaStorage::zeros` /
///   `VulkanStorage::zeros` / `MetalStorage::zeros`) when the
///   allocator-aware constructor lands in Phase 1.x.
/// - kiln-graph's `CaptureSession` to set `Frozen` mode for the
///   duration of `capture()` … `replay()` so the per-tensor `Arc`s
///   don't churn between capture and replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AllocatorMode {
    /// Per-allocation `cudaMalloc` / `MTLDevice::newBuffer` /
    /// `vkAllocateMemory`. Slow; used only at model load.
    Owned,
    /// Slab-allocated from a pre-sized device-local pool. The
    /// steady-state decode path. Mirrors Vulkan's `buffer_pool.rs`
    /// and `decode_resident_pool.rs` (MIN_SLOTS=3, PREFERRED_SLOTS=4,
    /// 1% of device-local heap).
    Pool,
    /// No allocation. Pulls from a pre-warmed slab indexed by
    /// tensor-handle. Active for the duration of
    /// `CaptureSession::begin()` … `CaptureSession::end()` so the
    /// recorded device pointers stay valid across replays.
    Frozen,
}

impl AllocatorMode {
    /// Stable short name for logging + JSON.
    pub const fn name(self) -> &'static str {
        match self {
            AllocatorMode::Owned => "owned",
            AllocatorMode::Pool => "pool",
            AllocatorMode::Frozen => "frozen",
        }
    }

    /// `true` iff this mode permits new allocations to be served.
    /// `Frozen` returns `false` — under freeze, every needed buffer
    /// MUST come from the pre-warmed slab.
    pub const fn allows_alloc(self) -> bool {
        !matches!(self, AllocatorMode::Frozen)
    }
}

impl Default for AllocatorMode {
    fn default() -> Self {
        AllocatorMode::Pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_strings() {
        assert_eq!(AllocatorMode::Owned.name(), "owned");
        assert_eq!(AllocatorMode::Pool.name(), "pool");
        assert_eq!(AllocatorMode::Frozen.name(), "frozen");
    }

    #[test]
    fn frozen_disallows_alloc() {
        assert!(AllocatorMode::Owned.allows_alloc());
        assert!(AllocatorMode::Pool.allows_alloc());
        assert!(!AllocatorMode::Frozen.allows_alloc());
    }

    #[test]
    fn default_is_pool() {
        assert_eq!(AllocatorMode::default(), AllocatorMode::Pool);
    }
}
