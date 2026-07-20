//! `kiln_tensor::Determinism` — per-op determinism classification.
//!
//! Operationalizes the determinism stance from PROFILING.md (Phase 0.3
//! of #1082). Every kiln-tensor op declares one of these in its
//! metadata; a [`Determinism::ToleranceBounded`] backward op without a
//! matching row in `bench-results/parity-tolerance.csv` is a Phase 9
//! audit failure.
//!
//! # User-facing contract
//!
//! The owning application selects one immutable process-wide policy value that
//! tensor and kernel implementations can consult. This module does not claim
//! that every `tolerance_bounded` op currently consumes the selector, nor does
//! it configure external library controls such as `CUBLAS_WORKSPACE_CONFIG`.
//!
//! ```ignore
//! use kiln_tensor::determinism::deterministic_enabled;
//!
//! fn embedding_bwd_impl(...) {
//!     if deterministic_enabled() {
//!         deterministic_atomic_add_path(...)
//!     } else {
//!         fast_atomic_add_path(...)
//!     }
//! }
//! ```

use core::fmt;

/// Classification of an op's determinism property.
///
/// Anchored to the four categories documented in PROFILING.md's
/// "Determinism stance" section:
///
/// 1. cuBLAS workspace (constructive under `:4096:8`)
/// 2. atomicAdd in bwd (tolerance-bounded; deterministic variant
///    available under deterministic execution policy)
/// 3. Reduction order in softmax / RMSNorm / cross-entropy bwd
///    (constructive when fixed-tree)
/// 4. Warp-shuffle reductions (constructive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Determinism {
    /// Bit-identical across runs on the same hardware + dtype + input.
    /// Examples: shape-only ops (reshape / transpose / permute /
    /// narrow), elementwise (silu / sigmoid / add / mul), fixed-tree
    /// reductions (softmax fwd, RMSNorm fwd), RoPE, conv1d.
    Constructive,
    /// Order-dependent; bounded by the per-dtype tolerance row in
    /// `bench-results/parity-tolerance.csv`. The `dtype_band_key`
    /// identifies the row.
    ///
    /// Examples: atomicAdd in embedding bwd, dW path in matmul bwd
    /// (when the backend uses per-column atomic accumulation),
    /// flash-attn bwd (deterministic variant available under build
    /// flag).
    ToleranceBounded {
        /// Key into `bench-results/parity-tolerance.csv` — the
        /// `category` column. Example: `"atomic-bwd"` for embedding
        /// gradient accumulation; `"matmul-bwd-atomic"` for the dW
        /// path; `"attention"` for flash-attn bwd.
        dtype_band_key: &'static str,
    },
}

impl Determinism {
    /// Is this op bit-identical across runs?
    pub const fn is_constructive(self) -> bool {
        matches!(self, Determinism::Constructive)
    }

    /// Does this op require a parity-tolerance row?
    pub const fn is_tolerance_bounded(self) -> bool {
        !self.is_constructive()
    }

    /// For tolerance-bounded ops, the band key (CSV `category` column).
    /// Returns `None` for constructive ops.
    pub const fn band_key(self) -> Option<&'static str> {
        match self {
            Determinism::Constructive => None,
            Determinism::ToleranceBounded { dtype_band_key } => Some(dtype_band_key),
        }
    }
}

impl fmt::Display for Determinism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Determinism::Constructive => f.write_str("constructive"),
            Determinism::ToleranceBounded { dtype_band_key } => {
                write!(f, "tolerance-bounded({dtype_band_key})")
            }
        }
    }
}

/// Return the process-lifetime deterministic runtime selection.
///
/// Standalone tensor consumers receive the documented `false` default. Servers
/// validate their typed configuration and call [`DeterministicCache::configure`]
/// before initializing any tensor runtime.
pub fn deterministic_enabled() -> bool {
    DETERMINISTIC_CACHED.is_on()
}

/// Process-lifetime deterministic selection. Standalone use resolves to false;
/// typed server startup may configure it before the first operation.
///
/// Hot-path entry points should call this:
///
/// ```ignore
/// if kiln_tensor::determinism::DETERMINISTIC_CACHED.is_on() {
///     deterministic_atomic_add(...);
/// } else {
///     fast_atomic_add(...);
/// }
/// ```
pub static DETERMINISTIC_CACHED: DeterministicCache = DeterministicCache::new();

/// Sub-byte-state cache for deterministic execution policy.
///
/// Encodes three states in a single `AtomicU8`:
///   - 0 = unconfigured (resolve to false on first `is_on()`)
///   - 1 = off
///   - 2 = on
#[derive(Debug)]
pub struct DeterministicCache {
    state: core::sync::atomic::AtomicU8,
}

impl DeterministicCache {
    pub const fn new() -> Self {
        DeterministicCache {
            state: core::sync::atomic::AtomicU8::new(0),
        }
    }

    /// Read the cached value, populating from env on the first call.
    pub fn is_on(&self) -> bool {
        use core::sync::atomic::Ordering;
        let v = self.state.load(Ordering::Relaxed);
        match v {
            0 => {
                match self
                    .state
                    .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => false,
                    Err(active) => active == 2,
                }
            }
            1 => false,
            _ => true,
        }
    }

    /// Fix the process-lifetime selection before tensor initialization.
    ///
    /// Repeating the same selection is harmless. A conflicting selection is
    /// rejected because changing deterministic kernels after an operation has
    /// run would make one process internally irreproducible.
    pub fn configure(
        &self,
        enabled: bool,
    ) -> core::result::Result<(), DeterministicConfigurationError> {
        use core::sync::atomic::Ordering;
        let requested = if enabled { 2 } else { 1 };
        match self
            .state
            .compare_exchange(0, requested, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => Ok(()),
            Err(active) if active == requested => Ok(()),
            Err(active) => Err(DeterministicConfigurationError {
                requested: enabled,
                active: active == 2,
            }),
        }
    }

    /// Force-set the cache. **Tests only**.
    #[doc(hidden)]
    pub fn _force_for_test(&self, on: bool) {
        use core::sync::atomic::Ordering;
        self.state.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    }

    /// Reset to the unread state. **Tests only**.
    #[doc(hidden)]
    pub fn _reset_for_test(&self) {
        use core::sync::atomic::Ordering;
        self.state.store(0, Ordering::Relaxed);
    }
}

/// A process attempted to change deterministic tensor behavior after it was
/// already fixed by startup configuration or first use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterministicConfigurationError {
    pub requested: bool,
    pub active: bool,
}

impl fmt::Display for DeterministicConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "deterministic tensor runtime is already configured as {}; cannot change it to {}",
            self.active, self.requested
        )
    }
}

impl std::error::Error for DeterministicConfigurationError {}

impl Default for DeterministicCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructive_classification() {
        let d = Determinism::Constructive;
        assert!(d.is_constructive());
        assert!(!d.is_tolerance_bounded());
        assert_eq!(d.band_key(), None);
        assert_eq!(format!("{d}"), "constructive");
    }

    #[test]
    fn tolerance_bounded_classification() {
        let d = Determinism::ToleranceBounded {
            dtype_band_key: "atomic-bwd",
        };
        assert!(!d.is_constructive());
        assert!(d.is_tolerance_bounded());
        assert_eq!(d.band_key(), Some("atomic-bwd"));
        assert_eq!(format!("{d}"), "tolerance-bounded(atomic-bwd)");
    }

    #[test]
    fn deterministic_cache_force_and_read() {
        let cache = DeterministicCache::new();
        cache._force_for_test(true);
        assert!(cache.is_on());
        cache._force_for_test(false);
        assert!(!cache.is_on());
    }

    #[test]
    fn deterministic_cache_configuration_is_immutable() {
        let cache = DeterministicCache::new();
        cache.configure(true).unwrap();
        cache.configure(true).unwrap();
        assert_eq!(
            cache.configure(false).unwrap_err(),
            DeterministicConfigurationError {
                requested: false,
                active: true,
            }
        );
        assert!(cache.is_on());
    }

    #[test]
    fn deterministic_cache_defaults_off_until_configured() {
        let cache = DeterministicCache::new();
        assert!(!cache.is_on());
        cache._reset_for_test();
        cache.configure(false).unwrap();
        assert!(!cache.is_on());
    }
}
