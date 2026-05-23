//! `kiln_tensor::TensorId` — a stable identity for every Tensor / Parameter.
//!
//! The id is the migration target for `candle_core::TensorId` (30 call sites
//! per the Phase 0.1 audit; `vk_autograd.rs:24-66`'s `VkGradStore` already
//! keys on it).
//!
//! # Why stability matters (anti-pattern 11)
//!
//! From the issue:
//!
//! > **One Parameter, one TensorId — stable across variants.** Forward-
//! > quantized + backward-master + transposed cache + LoRA delta all live
//! > behind a single `Parameter` keyed on one `kiln_tensor::TensorId`. The
//! > id is assigned at `Parameter` construction and is stable when a
//! > Marlin/FP8 forward variant is added (or LoRA delta is hot-swapped).
//! > The optimizer never sees four tensors where there is one logical
//! > weight; `cuda_graph` capture never sees pointer churn from adapter
//! > swap; AdamW moments don't get orphaned on weight-form transitions.
//!
//! The implementation: a process-global atomic counter. Ids are unique
//! within a process lifetime; they are **not** stable across process
//! restarts (the master / checkpoint-loaded form keys the optimizer state
//! by parameter *name*, not by `TensorId`).

use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

/// Stable identity for a Tensor / Parameter.
///
/// Two `TensorId` values compare equal iff they were produced by the
/// same [`TensorId::next`] call. The id is intentionally **opaque** —
/// callers must not derive arithmetic relationships from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(u64);

impl TensorId {
    /// Produce a fresh id. Thread-safe; uses a process-global atomic
    /// counter with `Relaxed` ordering (we don't synchronize anything
    /// through the counter, only emit unique values).
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        TensorId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Construct a `TensorId` from a raw u64. **Only for serialization
    /// round-trips and tests** — call sites that need a new id must use
    /// [`next`](TensorId::next).
    ///
    /// Note: there is no uniqueness guarantee here; if you pass an id
    /// that matches an existing live tensor, equality checks (e.g. the
    /// optimizer's `HashMap<TensorId, AdamWMoments>` lookups) silently
    /// alias.
    pub const fn from_raw(raw: u64) -> Self {
        TensorId(raw)
    }

    /// Extract the raw u64. Symmetric to [`from_raw`](TensorId::from_raw).
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t#{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_returns_unique_ids() {
        let a = TensorId::next();
        let b = TensorId::next();
        let c = TensorId::next();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        // Monotonically increasing (counter starts at 1 and increments
        // by 1 per call; even with neighboring tests racing, b > a + 0).
        assert!(b.as_raw() > a.as_raw());
        assert!(c.as_raw() > b.as_raw());
    }

    #[test]
    fn from_raw_roundtrip() {
        for raw in [0u64, 1, 42, u64::MAX] {
            assert_eq!(TensorId::from_raw(raw).as_raw(), raw);
        }
    }

    #[test]
    fn display_prefixes_with_t() {
        assert_eq!(format!("{}", TensorId::from_raw(7)), "t#7");
    }

    #[test]
    fn equality_is_value_equality() {
        let a = TensorId::from_raw(123);
        let b = TensorId::from_raw(123);
        let c = TensorId::from_raw(124);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
