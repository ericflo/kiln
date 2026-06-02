//! `WorkspacePool` — per-handle cublasLt workspace allocator.
//!
//! Per the Phase 2 issue bullet:
//!
//! > Matmul with explicit algo cache, **workspace pool**, optional
//! > split-K, optional fused-bias-and-activation epilogue.
//!
//! Backend-agnostic data type — the per-backend impl (CUDA cudarc
//! buffer / Metal MTLBuffer / Vulkan vk::Buffer) wraps this with the
//! actual device allocation. This module's job is the **policy**:
//!
//! - Cap workspace at a configurable limit (default 32 MiB matching the
//!   Phase 0.8 probe).
//! - Track per-`MatmulHandle` peak usage for `bench-results/` reports.
//! - Allow reuse across calls on the same stream (zero re-alloc cost in
//!   the steady state).
//!
//! # No-cuda build
//!
//! This module compiles on every host. It carries no GPU allocations —
//! only the byte budget + per-handle high-water marks. The per-backend
//! `MatmulHandle::workspace_buffer()` method (Phase 2.x) consults this
//! policy.

/// Per-handle workspace policy.
#[derive(Debug, Clone, Copy)]
pub struct WorkspacePool {
    /// Maximum allowed workspace bytes per matmul. Cublasrt's
    /// heuristic respects this cap when picking an algo.
    pub max_bytes: u64,
    /// High-water mark observed across all matmuls served by this
    /// handle. Reported by `bench-results/`.
    pub peak_bytes: u64,
    /// Number of matmul calls served. Reported by `bench-results/`.
    pub call_count: u64,
}

impl WorkspacePool {
    /// Default per-handle policy. 32 MiB cap matches the Phase 0.8
    /// `cublaslt_mlp_probe` exemplar.
    pub const DEFAULT_MAX_BYTES: u64 = 32 * 1024 * 1024;

    /// Construct with the default cap.
    pub fn new() -> Self {
        WorkspacePool {
            max_bytes: Self::DEFAULT_MAX_BYTES,
            peak_bytes: 0,
            call_count: 0,
        }
    }

    /// Construct with an explicit cap (used by tests + callers that
    /// know their workload's working-set size).
    pub fn with_cap(max_bytes: u64) -> Self {
        WorkspacePool {
            max_bytes,
            peak_bytes: 0,
            call_count: 0,
        }
    }

    /// Record one matmul call's workspace usage. Updates the peak.
    pub fn record(&mut self, used_bytes: u64) {
        self.call_count += 1;
        if used_bytes > self.peak_bytes {
            self.peak_bytes = used_bytes;
        }
    }

    /// Returns `true` iff the algo's requested workspace fits within
    /// the per-handle cap.
    pub fn allows(&self, requested_bytes: u64) -> bool {
        requested_bytes <= self.max_bytes
    }

    /// Reset the high-water mark + call counter. Used between
    /// bench harness phases.
    pub fn reset(&mut self) {
        self.peak_bytes = 0;
        self.call_count = 0;
    }
}

impl Default for WorkspacePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_cap_is_32_mib() {
        let p = WorkspacePool::new();
        assert_eq!(p.max_bytes, 32 * 1024 * 1024);
    }

    #[test]
    fn record_updates_peak_and_count() {
        let mut p = WorkspacePool::new();
        p.record(1024);
        p.record(2048);
        p.record(512);
        assert_eq!(p.peak_bytes, 2048);
        assert_eq!(p.call_count, 3);
    }

    #[test]
    fn allows_respects_cap() {
        let p = WorkspacePool::with_cap(1024);
        assert!(p.allows(0));
        assert!(p.allows(512));
        assert!(p.allows(1024));
        assert!(!p.allows(1025));
        assert!(!p.allows(u64::MAX));
    }

    #[test]
    fn reset_zeros_counters_not_cap() {
        let mut p = WorkspacePool::with_cap(2048);
        p.record(1024);
        p.reset();
        assert_eq!(p.peak_bytes, 0);
        assert_eq!(p.call_count, 0);
        assert_eq!(p.max_bytes, 2048);
    }
}
