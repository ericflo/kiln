//! `RocmAllocator` — the ROCm [`Allocator`] impl.
//!
//! Wraps an `Arc<kiln_hip::RocmContext>` (the same handle every ROCm
//! kernel crate uses) and produces [`RocmStorage`] allocations through
//! it.
//!
//! Phase R.3 of #1082 ships:
//!
//! - Owned/Pool/Frozen mode handling identical to [`crate::CpuAllocator`]
//! - `warm(dtype, n_elements, count)` for pre-Frozen pool population
//!   (used by Phase 5's [`crate::CaptureSession`] before flipping mode)
//! - `reserved_bytes` / `peak_reserved_bytes` accounting
//!
//! GPU-only tests live behind `KILN_TENSOR_ROCM_TEST=1` so the CI
//! compile path (which links `rocm` against `kiln-hip` but has no
//! GPU) doesn't spuriously fail.
//!
//! # candle-free by construction
//!
//! `RocmAllocator` carries only `Arc<kiln_hip::RocmContext>` on the
//! ROCm-handle side — there is no candle device on the ROCm backend.
//! Every internal allocation site routes through this context handle
//! via [`RocmStorage::zeros_ctx`], which allocates straight through
//! kiln-hip's `active_rocm_stream(ctx).alloc_zeros(byte_len)`.

use std::collections::HashMap;
use std::sync::Arc;

use kiln_hip::RocmContext;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, DType, Device, Result, RocmStorage, Storage,
};

#[derive(Debug)]
pub struct RocmAllocator {
    /// kiln-hip `RocmContext` — the **only** ROCm-side handle this
    /// allocator carries.
    ///
    /// Used to allocate device memory via
    /// `active_rocm_stream(ctx).alloc_zeros(byte_len)` through the
    /// [`RocmStorage::zeros_ctx`] entry. There is no candle device on
    /// the ROCm backend; callers that need a stream handle read it from
    /// the produced `RocmStorage`.
    ctx: Arc<RocmContext>,
    /// ROCm device ordinal — matches the ordinal of `ctx`'s owning
    /// device.
    device_index: usize,
    mode: AllocatorMode,
    /// Free-list cache keyed on `(dtype, n_elements)`. See
    /// [`CpuAllocator`] for the mode contract.
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl RocmAllocator {
    /// Construct in `Owned` mode bound to `ctx` at the given ROCm
    /// ordinal — the canonical, candle-free constructor entry.
    pub fn new_ctx(ctx: Arc<RocmContext>, device_index: usize) -> Self {
        RocmAllocator {
            ctx,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    /// Construct directly in a given mode (tests, capture session) —
    /// **candle-free** entry.
    pub fn with_mode_ctx(
        ctx: Arc<RocmContext>,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new_ctx(ctx, device_index);
        a.mode = mode;
        a
    }

    /// Pre-warm the pool: allocate `count` storages of `(dtype,
    /// n_elements)` and stash them in the cache so subsequent
    /// `alloc()` calls (under any mode) can pull from there.
    ///
    /// Phase 5's `CaptureSession::begin()` calls this for every
    /// `(dtype, n_elements)` the captured graph needs before
    /// flipping to `Frozen`.
    ///
    /// Internally routes through [`RocmStorage::zeros_ctx`] (the
    /// candle-free `Arc<RocmContext>` allocation entry). The actual ROCm
    /// allocation goes straight through kiln-hip's
    /// `active_rocm_stream(ctx).alloc_zeros` with no candle device
    /// involvement.
    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            let rocm =
                RocmStorage::zeros_ctx(&self.ctx, self.device_index, dtype, n_elements)?;
            let storage: Storage = Arc::new(rocm);
            slot.push(storage);
            self.reserved_bytes += bytes_per;
        }
        if self.reserved_bytes > self.peak_reserved_bytes {
            self.peak_reserved_bytes = self.reserved_bytes;
        }
        Ok(())
    }

    /// Number of cached storages for a given key (testing / introspection).
    pub fn cache_len(&self, dtype: DType, n_elements: usize) -> usize {
        self.cache
            .get(&(dtype, n_elements))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Borrow the kiln-hip `RocmContext` handle this allocator was
    /// constructed with. The same handle used for every internal
    /// allocation via `active_rocm_stream(ctx).alloc_zeros` (see
    /// [`Self::warm`] / `Allocator::alloc`).
    pub fn context(&self) -> &Arc<RocmContext> {
        &self.ctx
    }

    /// The ROCm device ordinal this allocator is bound to.
    pub fn device_index(&self) -> usize {
        self.device_index
    }
}

impl Allocator for RocmAllocator {
    fn device(&self) -> Device {
        Device::Rocm(self.device_index)
    }

    fn mode(&self) -> AllocatorMode {
        self.mode
    }

    fn set_mode(&mut self, mode: AllocatorMode) -> Result<()> {
        // Same accepting transitions as CpuAllocator. The Frozen check
        // is enforced on alloc cache-miss, not at mode change.
        self.mode = mode;
        Ok(())
    }

    fn alloc(&mut self, dtype: DType, n_elements: usize) -> Result<Storage> {
        // Try the cache first.
        if let Some(slot) = self.cache.get_mut(&(dtype, n_elements)) {
            if let Some(s) = slot.pop() {
                return Ok(s);
            }
        }
        // Cache miss.
        match self.mode {
            AllocatorMode::Frozen => Err(allocator_frozen_error(
                "RocmAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                // Route through the candle-free zeros_ctx entry — the
                // actual kiln-hip allocation skips any candle wrapper
                // entirely.
                let rocm = RocmStorage::zeros_ctx(
                    &self.ctx,
                    self.device_index,
                    dtype,
                    n_elements,
                )?;
                let storage: Storage = Arc::new(rocm);
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

    fn rocm_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_ROCM_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_rocm_ctx() -> Option<Arc<RocmContext>> {
        if !rocm_test_enabled() {
            return None;
        }
        RocmContext::new(0).ok()
    }

    #[test]
    fn rocm_allocator_starts_in_owned_mode() {
        let Some(ctx) = maybe_rocm_ctx() else {
            eprintln!("skip: KILN_TENSOR_ROCM_TEST unset or no GPU");
            return;
        };
        let a = RocmAllocator::new_ctx(ctx, 0);
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Rocm(0));
        assert_eq!(a.reserved_bytes(), 0);
    }

    #[test]
    fn rocm_owned_alloc_increments_reserved() {
        let Some(ctx) = maybe_rocm_ctx() else {
            eprintln!("skip: KILN_TENSOR_ROCM_TEST unset or no GPU");
            return;
        };
        let mut a = RocmAllocator::new_ctx(ctx, 0);
        let s = a.alloc(DType::BF16, 32).unwrap();
        assert_eq!(s.dtype(), DType::BF16);
        assert_eq!(s.byte_len(), 64);
        assert_eq!(a.reserved_bytes(), 64);
        assert_eq!(a.peak_reserved_bytes(), 64);
    }

    #[test]
    fn rocm_pool_alloc_serves_from_warm_cache() {
        let Some(ctx) = maybe_rocm_ctx() else {
            eprintln!("skip: KILN_TENSOR_ROCM_TEST unset or no GPU");
            return;
        };
        let mut a = RocmAllocator::with_mode_ctx(ctx, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 16, 2).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 2);
        let _s1 = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 1);
        let _s2 = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 0);
    }

    #[test]
    fn rocm_frozen_alloc_fails_on_cache_miss() {
        let Some(ctx) = maybe_rocm_ctx() else {
            eprintln!("skip: KILN_TENSOR_ROCM_TEST unset or no GPU");
            return;
        };
        let mut a = RocmAllocator::with_mode_ctx(ctx, 0, AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(
            e.to_string().contains("RocmAllocator::alloc"),
            "got: {e}"
        );
    }

    #[test]
    fn rocm_frozen_alloc_succeeds_when_warm_cache_has_match() {
        let Some(ctx) = maybe_rocm_ctx() else {
            eprintln!("skip: KILN_TENSOR_ROCM_TEST unset or no GPU");
            return;
        };
        let mut a = RocmAllocator::with_mode_ctx(ctx, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 4, 1).unwrap();
        a.set_mode(AllocatorMode::Frozen).unwrap();
        let s = a.alloc(DType::F32, 4).unwrap();
        assert_eq!(s.byte_len(), 16);
    }

    #[test]
    fn rocm_set_mode_transitions_freely() {
        let Some(ctx) = maybe_rocm_ctx() else {
            eprintln!("skip: KILN_TENSOR_ROCM_TEST unset or no GPU");
            return;
        };
        let mut a = RocmAllocator::new_ctx(ctx, 0);
        a.set_mode(AllocatorMode::Pool).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Pool);
        a.set_mode(AllocatorMode::Frozen).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Frozen);
        a.set_mode(AllocatorMode::Owned).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Owned);
    }
}
