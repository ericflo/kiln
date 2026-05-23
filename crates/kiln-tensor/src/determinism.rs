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
//! Setting `KILN_DETERMINISTIC=1` enables the deterministic variant of
//! every `tolerance_bounded` op (cuBLAS workspace pin, deterministic
//! atomicAdd embedding-bwd, deterministic-reduction-tree softmax / RMSNorm /
//! cross-entropy bwd, no warp-shuffle + cross-block reduction).
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
///    available under `KILN_DETERMINISTIC=1`)
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

/// Read `KILN_DETERMINISTIC` from the environment.
///
/// Returns `true` iff the env var is `"1"`, `"true"`, or `"yes"`
/// (case-insensitive, whitespace-trimmed). Mirrors `kiln_core::env_flag`
/// — kiln-tensor cannot depend on kiln-core directly (kiln-core depends
/// on kiln-tensor in subsequent phases), so we inline the same predicate.
///
/// Hot-path callers should cache the result rather than calling this
/// per-op: env-var lookup is a syscall + a `String` allocation. See
/// [`DETERMINISTIC_CACHED`].
pub fn deterministic_enabled() -> bool {
    match std::env::var("KILN_DETERMINISTIC").ok().as_deref() {
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes")
        }
        None => false,
    }
}

/// Cached, lazy `deterministic_enabled()`. Reads the env var once
/// per process; subsequent calls are a relaxed atomic load.
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

/// Sub-byte-state cache for the `KILN_DETERMINISTIC` env var.
///
/// Encodes three states in a single `AtomicU8`:
///   - 0 = unread (read from env on first `is_on()`)
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
                let on = deterministic_enabled();
                let encoded = if on { 2 } else { 1 };
                // Best-effort CAS — race-tolerant since both racers
                // read the same env var and write the same value.
                let _ = self.state.compare_exchange(
                    0,
                    encoded,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                on
            }
            1 => false,
            _ => true,
        }
    }

    /// Force-set the cache. **Tests only** — production callers must
    /// read the env var via [`is_on`].
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
    fn deterministic_cache_reads_env_lazily() {
        let cache = DeterministicCache::new();
        // Reset to unread, then ensure subsequent is_on() reads the
        // ambient env. Test serially via a process-global lock to
        // avoid racing other env-mutating tests.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap();
        cache._reset_for_test();
        // SAFETY: documented in std::env::set_var — single-threaded in
        // this lock-guarded test.
        unsafe { std::env::remove_var("KILN_DETERMINISTIC") };
        assert!(!cache.is_on());

        cache._reset_for_test();
        unsafe { std::env::set_var("KILN_DETERMINISTIC", "1") };
        assert!(cache.is_on());

        cache._reset_for_test();
        unsafe { std::env::set_var("KILN_DETERMINISTIC", "no") };
        assert!(!cache.is_on());

        unsafe { std::env::remove_var("KILN_DETERMINISTIC") };
    }
}
