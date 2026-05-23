//! `CpuAllocator` — the first concrete [`Allocator`] impl.
//!
//! CPU is the canonical numerical reference (per the issue's DoD), so
//! the allocator runs on every host without needing a GPU stack. It
//! is also the first user of the Phase 5 capture surface — the
//! `CpuAllocator` + `kiln-graph::CaptureSession` pair is a fully-
//! testable smoke harness for the dangling-pointer audit + the
//! Frozen-mode allocation rejection.
//!
//! # Mode semantics
//!
//! - **`Owned`** — every `alloc()` call returns a fresh `CpuStorage`.
//!   No internal pool. Most expensive but simplest. Default startup
//!   mode (`set_mode(Pool)` happens once the workload settles).
//! - **`Pool`** — a free-list cache keyed on `(dtype, n_elements)`.
//!   `alloc()` reuses any cached storage with the matching key;
//!   otherwise it falls back to a fresh `Vec<u8>` and tracks the
//!   peak. **Storages "returned" to the pool live in the cache until
//!   the allocator is dropped.**
//! - **`Frozen`** — `alloc()` only succeeds for keys already in the
//!   cache (pre-warmed via a prior `Pool` mode pass). Returns
//!   [`allocator_frozen_error`] otherwise.
//!
//! # Pool semantics today
//!
//! Today's `Pool` cache is a no-op LRU — every alloc returns a fresh
//! Vec because we have no `release()` API to put a storage *back* into
//! the cache. The free-list is populated only via [`CpuAllocator::warm`],
//! which the Phase 5 capture session calls before flipping to Frozen.
//!
//! A proper "drop-into-cache" hook lands in Phase 1.x's
//! [`crate::Storage`] interior-mutability story.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, CpuStorage, DType, Device, Error, Result,
    Storage,
};

/// CPU allocator. See module doc.
#[derive(Debug)]
pub struct CpuAllocator {
    mode: AllocatorMode,
    /// Free-list cache keyed on `(dtype, n_elements)`. Vec because
    /// duplicate keys are allowed (each slot is a separate storage).
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl CpuAllocator {
    /// Construct in `Owned` mode (the conservative startup mode).
    pub fn new() -> Self {
        CpuAllocator {
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    /// Construct directly in a given mode. Used by tests + by callers
    /// that know they're entering a specific phase.
    pub fn with_mode(mode: AllocatorMode) -> Self {
        let mut a = Self::new();
        a.mode = mode;
        a
    }

    /// Pre-warm the pool: allocate `count` storages of `(dtype,
    /// n_elements)` and stash them in the cache so subsequent
    /// `alloc()` calls (under any mode) can pull from there.
    ///
    /// Phase 5's `CaptureSession::begin()` calls this for every
    /// `(dtype, n_elements)` the captured graph needs before
    /// flipping mode to `Frozen`.
    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) {
        let key = (dtype, n_elements);
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry(key).or_default();
        for _ in 0..count {
            let cpu = CpuStorage::zeros(dtype, n_elements);
            let storage: Storage = Arc::new(cpu);
            slot.push(storage);
            self.reserved_bytes += bytes_per;
        }
        if self.reserved_bytes > self.peak_reserved_bytes {
            self.peak_reserved_bytes = self.reserved_bytes;
        }
    }

    /// Number of cached storages for a given key (testing / introspection).
    pub fn cache_len(&self, dtype: DType, n_elements: usize) -> usize {
        self.cache
            .get(&(dtype, n_elements))
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

impl Default for CpuAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl Allocator for CpuAllocator {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn mode(&self) -> AllocatorMode {
        self.mode
    }

    fn set_mode(&mut self, mode: AllocatorMode) -> Result<()> {
        // Phase 1.28 accepts every transition. The "Owned -> Frozen
        // without warmup" guard is a soft check (we return Frozen
        // errors lazily on alloc) rather than blocking the transition
        // — keeps the API ergonomic for tests and for the Phase 5
        // capture session that may legitimately go
        // `Owned -> Frozen` after a `warm()` pass.
        self.mode = mode;
        Ok(())
    }

    fn alloc(&mut self, dtype: DType, n_elements: usize) -> Result<Storage> {
        let key = (dtype, n_elements);
        // 1. Try the cache first regardless of mode — cheaper than alloc.
        if let Some(slot) = self.cache.get_mut(&key) {
            if let Some(s) = slot.pop() {
                // Storage came from the cache; reserved_bytes already
                // accounts for it.
                return Ok(s);
            }
        }
        // 2. Cache miss.
        match self.mode {
            AllocatorMode::Frozen => Err(allocator_frozen_error(
                "CpuAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                let cpu = CpuStorage::zeros(dtype, n_elements);
                let storage: Storage = Arc::new(cpu);
                let bytes = dtype.packed_buffer_bytes(n_elements);
                self.reserved_bytes += bytes;
                if self.reserved_bytes > self.peak_reserved_bytes {
                    self.peak_reserved_bytes = self.reserved_bytes;
                }
                Ok(storage)
            }
        }
    }

    fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }

    fn peak_reserved_bytes(&self) -> usize {
        self.peak_reserved_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_in_owned_mode() {
        let a = CpuAllocator::new();
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Cpu);
        assert_eq!(a.reserved_bytes(), 0);
        assert_eq!(a.peak_reserved_bytes(), 0);
    }

    #[test]
    fn owned_alloc_increments_reserved() {
        let mut a = CpuAllocator::new();
        let s = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(s.byte_len(), 64);
        assert_eq!(a.reserved_bytes(), 64);
        assert_eq!(a.peak_reserved_bytes(), 64);
    }

    #[test]
    fn pool_alloc_serves_from_warm_cache() {
        let mut a = CpuAllocator::with_mode(AllocatorMode::Pool);
        a.warm(DType::BF16, 32, 2);
        assert_eq!(a.cache_len(DType::BF16, 32), 2);
        let _s1 = a.alloc(DType::BF16, 32).unwrap();
        assert_eq!(a.cache_len(DType::BF16, 32), 1);
        let _s2 = a.alloc(DType::BF16, 32).unwrap();
        assert_eq!(a.cache_len(DType::BF16, 32), 0);
    }

    #[test]
    fn pool_alloc_falls_back_to_fresh_on_cache_miss() {
        let mut a = CpuAllocator::with_mode(AllocatorMode::Pool);
        let _s = a.alloc(DType::F32, 8).unwrap();
        // No warmup → cache empty → fresh alloc; reserved_bytes
        // reflects the fresh alloc.
        assert!(a.reserved_bytes() >= 32);
    }

    #[test]
    fn frozen_alloc_rejects_cache_miss() {
        let mut a = CpuAllocator::with_mode(AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(e.to_string().contains("Pre-warm"));
        assert!(e.to_string().contains("Frozen"));
    }

    #[test]
    fn frozen_alloc_serves_from_pre_warmed_cache() {
        let mut a = CpuAllocator::new(); // start in Owned
        a.warm(DType::F32, 16, 1); // warm one slot
        a.set_mode(AllocatorMode::Frozen).unwrap();
        let s = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(s.byte_len(), 64);
        // Now cache is empty → next alloc errors.
        let e = a.alloc(DType::F32, 16).unwrap_err();
        assert!(e.to_string().contains("Frozen"));
    }

    #[test]
    fn warm_tracks_reserved_bytes() {
        let mut a = CpuAllocator::new();
        a.warm(DType::F32, 16, 3);
        assert_eq!(a.reserved_bytes(), 3 * 64);
        assert_eq!(a.peak_reserved_bytes(), 3 * 64);
    }

    #[test]
    fn peak_reserved_only_grows() {
        let mut a = CpuAllocator::new();
        a.warm(DType::F32, 8, 1);
        let peak_after_first = a.peak_reserved_bytes();
        a.warm(DType::F32, 8, 2);
        assert!(a.peak_reserved_bytes() >= peak_after_first);
    }

    #[test]
    fn set_mode_transitions_succeed() {
        let mut a = CpuAllocator::new();
        for m in [
            AllocatorMode::Owned,
            AllocatorMode::Pool,
            AllocatorMode::Frozen,
            AllocatorMode::Pool,
            AllocatorMode::Owned,
        ] {
            a.set_mode(m).unwrap();
            assert_eq!(a.mode(), m);
        }
    }

    #[test]
    fn packed_dtype_byte_accounting() {
        let mut a = CpuAllocator::new();
        a.warm(DType::Int4Packed, 16, 1);
        // 16 packed elements → 8 bytes.
        assert_eq!(a.reserved_bytes(), 8);
    }
}
