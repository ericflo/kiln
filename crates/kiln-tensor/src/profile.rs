//! Profile counters for the kiln-tensor migration (anti-pattern 2).
//!
//! Every explicit [`Tensor::contiguous`](crate::Tensor::contiguous) and
//! every kernel that internally materializes a copy emits a
//! [`kiln_profile_contiguous_copy`] event. The counter is aggregated
//! by `bench-results/` regression scripts and surfaces as a
//! "copies per token" metric — anti-pattern 2 mandates that
//! regressions on the 712-call layout surface in `forward.rs` (the
//! `.contiguous()` / `.narrow(` / `.reshape(` / `.transpose(` calls
//! Phase 0.1 captured) are **visible without re-reading nsys traces**.
//!
//! # Counter shape
//!
//! Today: a process-global atomic `u64` counter per event kind. Phase
//! 1.x adds per-call-site labels once the `kiln_nvtx` integration
//! hooks here.
//!
//! # Threading model
//!
//! `Relaxed` ordering — we don't synchronize anything through the
//! counter, only count events. The cost is one atomic add per
//! materializing copy, sub-nanosecond on the hot path.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::Backend;

#[derive(Debug)]
struct CopyCounter {
    value: AtomicU64,
}

impl CopyCounter {
    const fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    #[inline]
    fn emit(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    fn count(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }
}

/// Counter for explicit + implicit contiguous copies.
///
/// Bumped by [`emit_contiguous_copy`] and read by
/// [`contiguous_copy_count`] / [`reset_contiguous_copy_count`].
static CONTIGUOUS_COPY_COUNTER: CopyCounter = CopyCounter::new();

static METAL_DEVICE_OP1_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static METAL_DEVICE_OP2_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static METAL_DEVICE_OP3_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static VULKAN_DEVICE_OP1_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static VULKAN_DEVICE_OP2_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static VULKAN_DEVICE_OP3_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ROCM_DEVICE_OP1_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ROCM_DEVICE_OP2_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ROCM_DEVICE_OP3_HOST_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Per-backend counts of generic [`DeviceOp`](crate::DeviceOp1) CPU
/// round trips after a native backend method returned `Ok(None)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceOpHostFallbackCounts {
    pub metal_op1: u64,
    pub metal_op2: u64,
    pub metal_op3: u64,
    pub vulkan_op1: u64,
    pub vulkan_op2: u64,
    pub vulkan_op3: u64,
    pub rocm_op1: u64,
    pub rocm_op2: u64,
    pub rocm_op3: u64,
}

impl DeviceOpHostFallbackCounts {
    /// Sum all backend/arity fallback counters.
    pub fn total(self) -> u64 {
        self.metal_op1
            + self.metal_op2
            + self.metal_op3
            + self.vulkan_op1
            + self.vulkan_op2
            + self.vulkan_op3
            + self.rocm_op1
            + self.rocm_op2
            + self.rocm_op3
    }
}

fn device_op_host_fallback_counter(backend: Backend, arity: u8) -> Option<&'static AtomicU64> {
    match (backend, arity) {
        (Backend::Metal, 1) => Some(&METAL_DEVICE_OP1_HOST_FALLBACKS),
        (Backend::Metal, 2) => Some(&METAL_DEVICE_OP2_HOST_FALLBACKS),
        (Backend::Metal, 3) => Some(&METAL_DEVICE_OP3_HOST_FALLBACKS),
        (Backend::Vulkan, 1) => Some(&VULKAN_DEVICE_OP1_HOST_FALLBACKS),
        (Backend::Vulkan, 2) => Some(&VULKAN_DEVICE_OP2_HOST_FALLBACKS),
        (Backend::Vulkan, 3) => Some(&VULKAN_DEVICE_OP3_HOST_FALLBACKS),
        (Backend::Rocm, 1) => Some(&ROCM_DEVICE_OP1_HOST_FALLBACKS),
        (Backend::Rocm, 2) => Some(&ROCM_DEVICE_OP2_HOST_FALLBACKS),
        (Backend::Rocm, 3) => Some(&ROCM_DEVICE_OP3_HOST_FALLBACKS),
        _ => None,
    }
}

/// Bump the host-fallback counter for a backend/arity pair.
///
/// CPU and CUDA are ignored: CPU has no fallback and CUDA keeps the
/// strict "missing native kernel errors loudly" contract.
pub fn emit_device_op_host_fallback(backend: Backend, arity: u8) {
    if let Some(counter) = device_op_host_fallback_counter(backend, arity) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// Current generic DeviceOp host-fallback counts.
pub fn device_op_host_fallback_counts() -> DeviceOpHostFallbackCounts {
    DeviceOpHostFallbackCounts {
        metal_op1: METAL_DEVICE_OP1_HOST_FALLBACKS.load(Ordering::Relaxed),
        metal_op2: METAL_DEVICE_OP2_HOST_FALLBACKS.load(Ordering::Relaxed),
        metal_op3: METAL_DEVICE_OP3_HOST_FALLBACKS.load(Ordering::Relaxed),
        vulkan_op1: VULKAN_DEVICE_OP1_HOST_FALLBACKS.load(Ordering::Relaxed),
        vulkan_op2: VULKAN_DEVICE_OP2_HOST_FALLBACKS.load(Ordering::Relaxed),
        vulkan_op3: VULKAN_DEVICE_OP3_HOST_FALLBACKS.load(Ordering::Relaxed),
        rocm_op1: ROCM_DEVICE_OP1_HOST_FALLBACKS.load(Ordering::Relaxed),
        rocm_op2: ROCM_DEVICE_OP2_HOST_FALLBACKS.load(Ordering::Relaxed),
        rocm_op3: ROCM_DEVICE_OP3_HOST_FALLBACKS.load(Ordering::Relaxed),
    }
}

/// Reset generic DeviceOp host-fallback counters and return previous values.
pub fn reset_device_op_host_fallback_counts() -> DeviceOpHostFallbackCounts {
    DeviceOpHostFallbackCounts {
        metal_op1: METAL_DEVICE_OP1_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        metal_op2: METAL_DEVICE_OP2_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        metal_op3: METAL_DEVICE_OP3_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        vulkan_op1: VULKAN_DEVICE_OP1_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        vulkan_op2: VULKAN_DEVICE_OP2_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        vulkan_op3: VULKAN_DEVICE_OP3_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        rocm_op1: ROCM_DEVICE_OP1_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        rocm_op2: ROCM_DEVICE_OP2_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
        rocm_op3: ROCM_DEVICE_OP3_HOST_FALLBACKS.swap(0, Ordering::Relaxed),
    }
}

/// Bump the contiguous-copy counter by one.
///
/// Called from:
/// - `Tensor::contiguous()` whenever it actually allocates a fresh
///   buffer (the "fast path" early-return on already-contiguous
///   tensors skips this).
/// - Per-backend storage impls that internally materialize a copy
///   inside an op (e.g. CPU `view.contiguous` walks the strided view).
///
/// **Anti-pattern 2 contract**: every site that copies bytes due to a
/// stride/layout mismatch must call this. Adding a new copy in a
/// kernel without calling this is a Phase 9 audit failure.
#[inline]
pub fn emit_contiguous_copy() {
    CONTIGUOUS_COPY_COUNTER.emit();
}

/// Current counter value (process lifetime).
pub fn contiguous_copy_count() -> u64 {
    CONTIGUOUS_COPY_COUNTER.count()
}

/// Reset the counter to zero and return the previous value.
/// Used by bench harnesses around a single decode/prefill iteration
/// to compute the "copies per token" metric.
pub fn reset_contiguous_copy_count() -> u64 {
    CONTIGUOUS_COPY_COUNTER.reset()
}

/// RAII guard that captures the counter delta over a scope. Used by
/// bench harnesses around a single iteration to compute the
/// "copies per token" metric without manually pairing
/// `contiguous_copy_count()` reads.
///
/// ```ignore
/// use kiln_tensor::profile::CopyScope;
///
/// let scope = CopyScope::start();
/// // ... run decode iteration ...
/// let copies_this_iter = scope.finish();
/// ```
#[derive(Debug)]
#[must_use = "CopyScope only measures; call `finish()` to read the delta"]
pub struct CopyScope {
    start: u64,
}

impl CopyScope {
    /// Take a snapshot of the current counter.
    pub fn start() -> Self {
        CopyScope {
            start: contiguous_copy_count(),
        }
    }

    /// Compute and return the delta since [`start`](CopyScope::start).
    pub fn finish(self) -> u64 {
        contiguous_copy_count().saturating_sub(self.start)
    }
}

/// Process-global mutex for tests that voluntarily coordinate around
/// the `contiguous_copy_count` counter. Exposed at crate visibility
/// under `#[cfg(test)]` so tests in OTHER modules (e.g.
/// `tensor::tests`) can grab the same lock. Tests must still avoid
/// absolute assertions on the global counter because other tests can
/// legitimately emit copy events without taking this lock.
#[cfg(test)]
pub(crate) fn counter_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_and_read() {
        let counter = CopyCounter::new();
        counter.emit();
        counter.emit();
        counter.emit();
        assert_eq!(counter.count(), 3);
    }

    #[test]
    fn reset_returns_previous() {
        let counter = CopyCounter::new();
        counter.emit();
        counter.emit();
        let prev = counter.reset();
        assert_eq!(prev, 2);
        assert_eq!(counter.count(), 0);
    }

    #[test]
    fn scope_measures_delta() {
        let scope = CopyScope::start();
        emit_contiguous_copy();
        emit_contiguous_copy();
        emit_contiguous_copy();
        assert!(scope.finish() >= 3);
    }

    #[test]
    fn scope_starts_at_current_count() {
        let before = contiguous_copy_count();
        let scope = CopyScope::start();
        let after = contiguous_copy_count();
        assert!(scope.start >= before);
        assert!(scope.start <= after);
    }

    #[test]
    fn device_op_host_fallback_counts_are_backend_and_arity_specific() {
        let _guard = counter_test_lock();
        reset_device_op_host_fallback_counts();

        emit_device_op_host_fallback(Backend::Metal, 1);
        emit_device_op_host_fallback(Backend::Metal, 1);
        emit_device_op_host_fallback(Backend::Metal, 3);
        emit_device_op_host_fallback(Backend::Vulkan, 2);
        emit_device_op_host_fallback(Backend::Rocm, 3);

        // CPU, CUDA, and unsupported arities must remain outside this
        // correctness-fallback signal.
        emit_device_op_host_fallback(Backend::Cpu, 1);
        emit_device_op_host_fallback(Backend::Cuda, 1);
        emit_device_op_host_fallback(Backend::Metal, 4);

        let counts = device_op_host_fallback_counts();
        assert_eq!(counts.metal_op1, 2);
        assert_eq!(counts.metal_op2, 0);
        assert_eq!(counts.metal_op3, 1);
        assert_eq!(counts.vulkan_op1, 0);
        assert_eq!(counts.vulkan_op2, 1);
        assert_eq!(counts.vulkan_op3, 0);
        assert_eq!(counts.rocm_op1, 0);
        assert_eq!(counts.rocm_op2, 0);
        assert_eq!(counts.rocm_op3, 1);
        assert_eq!(counts.total(), 5);

        let previous = reset_device_op_host_fallback_counts();
        assert_eq!(previous, counts);
        assert_eq!(device_op_host_fallback_counts().total(), 0);
    }
}
