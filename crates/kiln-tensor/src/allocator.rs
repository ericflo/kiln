//! Backend-agnostic [`Allocator`] trait.
//!
//! Per the Phase 1 issue bullet:
//!
//! > **Allocator design — three modes, one interface.**
//! > `kiln-tensor::Allocator` has three explicit modes: (1) `Owned`
//! > (per-allocation `cudaMalloc` / `vkAllocateMemory` — slow, used
//! > only at startup); (2) `Pool` (slab-allocated from a pre-sized
//! > device-local pool, the steady-state path; mirrors Vulkan's
//! > `buffer_pool.rs` and `decode_resident_pool.rs`); (3) `Frozen`
//! > (no allocation; pulls from a pre-warmed slab indexed by
//! > tensor-handle — used during `capture()` … `replay()`).
//!
//! And:
//!
//! > **One allocator-pool design across Vulkan / CUDA / Metal.** ...
//! > All three present the same `Allocator` trait. Single bookkeeping
//! > codepath.
//!
//! # Phase 1.27 scope (this PR)
//!
//! Backend-agnostic [`Allocator`] trait + the [`AllocatorMode`] enum
//! (canonical home — `kiln-graph` re-exports). Per-backend impls
//! (CUDA `cudaMemPool_t`, Metal `MTLHeap`, Vulkan `buffer_pool.rs`)
//! plug in via this trait in subsequent PRs.
//!
//! # Memory accounting hook
//!
//! [`Allocator::reserved_bytes`] returns the bytes currently held by
//! the allocator's pool. Reported by `kiln-core::vram::detect_vram`
//! so the auto-sizer subtracts allocator-owned memory from the
//! available-for-KV-cache budget. The issue calls this out:
//!
//! > `Device::reserved_bytes()` per backend — authoritative
//! > "bytes held by kiln-tensor" so `kiln-core::vram` keeps
//! > auto-sizing the KV cache without undercount when we own
//! > buffer pools.

use crate::{DType, Device, Error, Result, Storage};

/// Three-mode allocator policy. Canonical home.
///
/// `kiln-graph` re-exports this so the Phase 5 capture surface and
/// the kiln-tensor allocator share one enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AllocatorMode {
    /// Per-allocation `cudaMalloc` / `MTLDevice::newBuffer` /
    /// `vkAllocateMemory`. Slow; used only at model load.
    Owned,
    /// Slab-allocated from a pre-sized device-local pool. The
    /// steady-state decode path. Mirrors Vulkan's `buffer_pool.rs`
    /// + `decode_resident_pool.rs`.
    Pool,
    /// No allocation. Pulls from a pre-warmed slab indexed by
    /// tensor-handle. Active for the duration of `capture()` …
    /// `replay()` so the recorded device pointers stay valid across
    /// replays.
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

    /// `true` iff this mode permits new allocations.
    pub const fn allows_alloc(self) -> bool {
        !matches!(self, AllocatorMode::Frozen)
    }
}

impl Default for AllocatorMode {
    fn default() -> Self {
        AllocatorMode::Pool
    }
}

impl core::fmt::Display for AllocatorMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Backend-agnostic allocator. Per-backend impls present the same
/// API; the Phase 5 capture session takes `&mut dyn Allocator` to
/// flip mode to `Frozen` for the capture window.
///
/// Implementations:
///
/// - `kiln-tensor::CpuAllocator` (Phase 1.27.x) — straight `Vec<u8>`
///   allocations; `Pool` mode is a free-list cache.
/// - `kiln-tensor::CudaAllocator` (Phase 1.x) — wraps
///   `cudaMemPool_t` (`cudaMallocFromPoolAsync` for steady-state
///   `Pool` mode, native CUDA-graph capture friendly).
/// - `kiln-tensor::MetalAllocator` (Phase 1.x) — wraps `MTLHeap`.
/// - `kiln-tensor::VulkanAllocator` (Phase 1.x) — lifts
///   `kiln-vulkan-kernel::buffer_pool::BufferPool`.
pub trait Allocator: Send + Sync + core::fmt::Debug {
    /// The device this allocator manages.
    fn device(&self) -> Device;

    /// Current mode.
    fn mode(&self) -> AllocatorMode;

    /// Set the allocator's mode.
    ///
    /// Phase 5's `CaptureSession::begin()` calls
    /// `set_mode(AllocatorMode::Frozen)` for the capture window;
    /// `CaptureSession::end()` restores the prior mode (typically
    /// `Pool`).
    ///
    /// Returns `Err` if the transition is illegal (e.g. transitioning
    /// from `Owned` to `Frozen` without a `Pool` warmup pass — the
    /// `Frozen` slab needs to exist first).
    fn set_mode(&mut self, mode: AllocatorMode) -> Result<()>;

    /// Allocate (or look up from the Frozen slab) a [`Storage`] with
    /// `n_elements` of `dtype`.
    ///
    /// Under `Owned`: equivalent to per-storage `*_zeros` constructors.
    /// Under `Pool`: pulls from the pre-sized device-local pool.
    /// Under `Frozen`: looks up by `(dtype, n_elements)` key — if not
    /// in the warm slab, returns `Err`.
    fn alloc(&mut self, dtype: DType, n_elements: usize) -> Result<Storage>;

    /// Total bytes the allocator's pool currently holds (sum of all
    /// outstanding slab + live allocations).
    ///
    /// Reported by `kiln-core::vram::detect_vram` so the KV-cache
    /// auto-sizer accounts for allocator-owned memory.
    fn reserved_bytes(&self) -> usize;

    /// Peak `reserved_bytes` since allocator construction. Useful for
    /// profile output + Phase 9 regression tracking.
    fn peak_reserved_bytes(&self) -> usize;
}

/// Error helper: standardize the "mode forbids alloc" message so
/// per-backend impls report identical errors.
pub fn allocator_frozen_error(op: &str, requested_bytes: usize) -> Error {
    Error::Msg(format!(
        "Allocator: cannot allocate {requested_bytes} bytes in Frozen mode \
         (op: {op}). Pre-warm the pool before entering the capture window."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_mode_names() {
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

    #[test]
    fn allocator_mode_display() {
        assert_eq!(format!("{}", AllocatorMode::Owned), "owned");
        assert_eq!(format!("{}", AllocatorMode::Frozen), "frozen");
    }

    #[test]
    fn frozen_error_helper_includes_request() {
        let e = allocator_frozen_error("test/op", 1024);
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("test/op"));
        assert!(s.contains("Pre-warm"));
    }
}
