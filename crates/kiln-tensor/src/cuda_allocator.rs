//! `CudaAllocator` — the CUDA [`Allocator`] impl.
//!
//! Wraps an `Arc<candle_core::cuda_backend::CudaDevice>` (the same
//! handle every CUDA kernel crate uses) and produces [`CudaStorage`]
//! allocations through it.
//!
//! Phase 2.1.1 of #1082 ships:
//!
//! - Owned/Pool/Frozen mode handling identical to [`crate::CpuAllocator`]
//! - `warm(dtype, n_elements, count)` for pre-Frozen pool population
//!   (used by Phase 5's [`crate::CaptureSession`] before flipping mode)
//! - `reserved_bytes` / `peak_reserved_bytes` accounting
//!
//! GPU-only tests live behind `KILN_TENSOR_CUDA_TEST=1` so the CI
//! compile path (which links `cuda` against `candle-core` but has no
//! GPU) doesn't spuriously fail.
//!
//! # Phase 7 candle-removal — partial UNBLOCK (allocation path candle-free)
//!
//! As of the b39f5712 commit (`kiln-tensor: add CudaStorage::context()
//! accessor`), [`CudaStorage::context()`] returns the same
//! `Arc<cudarc::driver::CudaContext>` that candle's `CudaDevice` wraps
//! internally — derived via `candle_device.cuda_stream().context()
//! .clone()`. **The bridge from candle's `Arc<CudaDevice>` to a
//! cudarc-typed `Arc<CudaContext>` is now reachable without storage
//! changes.**
//!
//! Subsequent commits:
//! - 03b8a34c (`CudaAllocator gains Arc<CudaContext> companion field`)
//!   grew this struct with a second `ctx: Arc<CudaContext>` field
//!   populated from `candle_device.cuda_stream().context()` at
//!   construction time, and added a `context()` accessor for
//!   downstream callers that only need the cudarc handle.
//! - d3caf46b (`add CudaStorage::zeros_ctx parallel constructor`)
//!   landed the candle-free allocation entry on the storage side.
//! - **this commit**: flipped this allocator's `warm()` and `alloc()`
//!   internal call sites from `CudaStorage::zeros(candle_device, ...)`
//!   to `CudaStorage::zeros_ctx(&self.ctx, ...)`. The actual cudarc
//!   `alloc_zeros::<u8>` call is now reached through this allocator's
//!   `Arc<CudaContext>` companion field, NOT through the candle
//!   `CudaDevice::alloc_zeros` wrapper. The `candle_device` field is
//!   no longer load-bearing for allocation; it is only kept around so
//!   the existing `candle_device()` getter keeps compiling for
//!   downstream callers (Phase 5's `CaptureSession` etc.) that have
//!   not yet migrated to `.context()`.
//!
//! The **field flip** that fully removes the candle-typed
//! `candle_device: Arc<CudaDevice>` is still blocked on the storage-
//! side `candle_device` field flip (the produced `CudaStorage` still
//! carries a candle device for downstream FFI sites that read
//! `cuda_stream()`), but every downstream caller can now migrate to
//! `.context()` independently — and this allocator no longer creates
//! any new candle-flavored allocations internally.
//!
//! # Original audit (call-graph dependencies)
//!
//! The original plan was for Phase 7 to swap `Arc<CudaDevice>` for
//! `Arc<cudarc::driver::CudaContext>` here. After auditing the actual
//! call graph (#1082 phase 7 audit), that swap **cannot land in this
//! file in isolation**:
//!
//! 1. Every code path in `CudaAllocator` that touches the device
//!    handle (both `warm()` and `alloc()`) immediately forwards it to
//!    `CudaStorage::zeros(candle_device, device_index, dtype, n)`.
//! 2. `CudaStorage::zeros` itself stores `Arc<CudaDevice>` on the
//!    returned `CudaStorage` and uses it for both `alloc_zeros::<u8>`
//!    and the `cuda_stream()` accessor that every kernel-crate FFI
//!    site reads.
//! 3. There is no `Arc<CudaContext> → Arc<CudaDevice>` adapter — the
//!    relationship runs the other way: candle's `CudaDevice` *wraps*
//!    a `cudarc::driver::CudaContext`. Holding a `CudaContext` here
//!    would force this allocator to construct a fresh `CudaDevice`
//!    per `alloc()` call, which would break stream affinity with
//!    every kernel-crate handle that was set up against the original
//!    `CudaDevice`.
//! 4. `CudaAllocator::new` / `with_mode` have **zero external
//!    callers** today (`grep -rn 'CudaAllocator::new' crates/` —
//!    only `pub use cuda_allocator::CudaAllocator` in `lib.rs`).
//!    Swapping the constructor signature alone would not free any
//!    downstream from candle either.
//!
//! Therefore Phase 7 candle removal for the CUDA allocator surface
//! is **blocked on `CudaStorage::zeros` migrating first** to take an
//! `Arc<cudarc::driver::CudaContext>` (and the rest of `CudaStorage`
//! losing its `candle_device: Arc<CudaDevice>` field — see the parallel
//! STOP note in `cuda_storage.rs` line 343 about the substrate
//! transition for the `candle_device` field becoming
//! `Arc<cudarc::driver::CudaContext>`). Once `CudaStorage` accepts a
//! `CudaContext`, this allocator becomes a trivial type-substitution
//! that recompiles with no behavior change (the only thing changing
//! is the type of the field stored next to the `device_index`).
//!
//! Order-of-operations for the lift (tracked under #1082):
//!
//! 0. **DONE (b39f5712)**: Add `CudaStorage::context() ->
//!    Arc<CudaContext>` accessor. Derives the cudarc context from
//!    candle's existing `CudaDevice::cuda_stream().context().clone()`
//!    with no struct change. This is the read-side bridge.
//! 1. Add a parallel `CudaStorage::zeros_kt(ctx: Arc<CudaContext>, ...)`
//!    that allocates via `ctx.default_stream().alloc_zeros::<u8>` and
//!    stores `Arc<CudaContext>` instead of `Arc<CudaDevice>` on
//!    `CudaStorage` (likely via an internal enum + dual accessors so
//!    the existing kernel-crate FFI sites keep compiling unchanged).
//! 2. Migrate the dozen-plus kernel-crate FFI call sites that reach
//!    `.candle_device().cuda_stream()` to a `cuda_stream_raw()` /
//!    `CUstream` accessor that already exists on `CudaStorage`.
//! 3. Flip `CudaStorage::zeros` itself to take `Arc<CudaContext>`,
//!    drop the `candle_device` field.
//! 4. Then this file changes one type (`candle_device: Arc<CudaDevice>`
//!    → `ctx: Arc<cudarc::driver::CudaContext>`) and one import
//!    (`use cudarc::driver::CudaContext;` replacing
//!    `use candle_core::cuda_backend::CudaDevice;`).
//!
//! Interim moves (additive, can land in any order once step 0 is in):
//!
//! - Grow this allocator's struct with a second `ctx: Arc<CudaContext>`
//!   field populated from `candle_device.cuda_stream().context()` at
//!   construction. Add a `ctx()` accessor. Downstream callers that
//!   only need the context (rather than the full candle wrapper) can
//!   migrate without waiting for the storage-side field flip.
//!
//! Doing step 4 first (the swap implied by the prior docstring) would
//! either require an `Arc<CudaContext>` → `Arc<CudaDevice>` conversion
//! at every `alloc()` call (impossible without holding the
//! `CudaDevice` somewhere) or would silently break stream affinity
//! across the kernel-crate FFI surface. Both are worse than holding
//! the substrate stable until step 1 lands.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::cuda_backend::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::CudaContext;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, CudaStorage, DType, Device, Result, Storage,
};

#[derive(Debug)]
pub struct CudaAllocator {
    /// Candle CUDA device handle. Held for stream affinity + the
    /// `alloc_zeros::<u8>` helper. The Phase 7 swap to a direct
    /// cudarc `CudaContext` field is still blocked on
    /// `CudaStorage::zeros` migrating first (step 1 in the top-of-
    /// file order-of-operations); the *read*-side bridge is now
    /// available via [`CudaStorage::context()`] (step 0, landed in
    /// b39f5712), so downstream callers can already migrate to
    /// `.context()` without waiting for this field to flip.
    candle_device: Arc<CudaDevice>,
    /// Cudarc `CudaContext` companion handle, derived from
    /// `candle_device.cuda_stream().context().clone()` at construction
    /// time — the **same** underlying CUDA primary context as
    /// `candle_device`, just exposed without the candle wrapper.
    ///
    /// Held alongside `candle_device` so callers that only need the
    /// context can read it from this allocator (via [`Self::context()`])
    /// without going through candle. This is the interim "additive"
    /// move ahead of the field flip — downstream consumers can migrate
    /// off `.candle_device()` to `.context()` site-by-site, and when
    /// the storage-side refactor (step 3 of the top-of-file order)
    /// lands, this allocator's `candle_device` field can be dropped in
    /// favor of just this `ctx` handle.
    ctx: Arc<CudaContext>,
    /// CUDA device index — matches the index of `candle_device`'s
    /// owning context.
    device_index: usize,
    mode: AllocatorMode,
    /// Free-list cache keyed on `(dtype, n_elements)`. See
    /// [`CpuAllocator`] for the mode contract.
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl CudaAllocator {
    /// Construct in `Owned` mode bound to `candle_device` at the
    /// given CUDA index. The cudarc-typed `ctx` companion is derived
    /// from the candle device internally — see the `ctx` field doc.
    pub fn new(candle_device: Arc<CudaDevice>, device_index: usize) -> Self {
        let ctx = candle_device.cuda_stream().context().clone();
        CudaAllocator {
            candle_device,
            ctx,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    /// Construct directly in a given mode (tests, capture session).
    pub fn with_mode(
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new(candle_device, device_index);
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
    /// Internally routes through [`CudaStorage::zeros_ctx`] (the
    /// candle-free `Arc<CudaContext>` allocation entry, landed in
    /// d3caf46b) instead of [`CudaStorage::zeros`]. The actual CUDA
    /// allocation goes straight through cudarc's
    /// `ctx.default_stream().alloc_zeros::<u8>` with no candle device
    /// involvement; the produced `CudaStorage` still carries a
    /// back-compat `candle_device` field for downstream FFI sites
    /// that read `cuda_stream()` (those sites migrate to
    /// `cuda_stream_raw()` in a follow-up sweep).
    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            let cuda =
                CudaStorage::zeros_ctx(&self.ctx, self.device_index, dtype, n_elements)?;
            let storage: Storage = Arc::new(cuda);
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

    /// Borrow the underlying candle device handle.
    pub fn candle_device(&self) -> &Arc<CudaDevice> {
        &self.candle_device
    }

    /// Borrow the cudarc `CudaContext` companion handle (the same
    /// underlying CUDA primary context as `candle_device`, exposed
    /// without the candle wrapper). See the `ctx` field doc for the
    /// migration rationale.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl Allocator for CudaAllocator {
    fn device(&self) -> Device {
        Device::Cuda(self.device_index)
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
                "CudaAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                // Route through the candle-free zeros_ctx entry
                // (d3caf46b) — the actual cudarc allocation skips the
                // candle `CudaDevice::alloc_zeros` wrapper entirely.
                let cuda = CudaStorage::zeros_ctx(
                    &self.ctx,
                    self.device_index,
                    dtype,
                    n_elements,
                )?;
                let storage: Storage = Arc::new(cuda);
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
    use candle_core::Device as CandleDevice;

    fn cuda_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_CUDA_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_cuda_device() -> Option<Arc<CudaDevice>> {
        if !cuda_test_enabled() {
            return None;
        }
        match CandleDevice::new_cuda(0).ok()? {
            CandleDevice::Cuda(d) => Some(Arc::new(d)),
            _ => None,
        }
    }

    #[test]
    fn cuda_allocator_starts_in_owned_mode() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let a = CudaAllocator::new(dev, 0);
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Cuda(0));
        assert_eq!(a.reserved_bytes(), 0);
    }

    #[test]
    fn cuda_owned_alloc_increments_reserved() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::new(dev, 0);
        let s = a.alloc(DType::BF16, 32).unwrap();
        assert_eq!(s.dtype(), DType::BF16);
        assert_eq!(s.byte_len(), 64);
        assert_eq!(a.reserved_bytes(), 64);
        assert_eq!(a.peak_reserved_bytes(), 64);
    }

    #[test]
    fn cuda_pool_alloc_serves_from_warm_cache() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::with_mode(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 16, 2).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 2);
        let _s1 = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 1);
        let _s2 = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 0);
    }

    #[test]
    fn cuda_frozen_alloc_fails_on_cache_miss() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::with_mode(dev, 0, AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(
            e.to_string().contains("CudaAllocator::alloc"),
            "got: {e}"
        );
    }

    #[test]
    fn cuda_frozen_alloc_succeeds_when_warm_cache_has_match() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::with_mode(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 4, 1).unwrap();
        a.set_mode(AllocatorMode::Frozen).unwrap();
        let s = a.alloc(DType::F32, 4).unwrap();
        assert_eq!(s.byte_len(), 16);
    }

    #[test]
    fn cuda_set_mode_transitions_freely() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::new(dev, 0);
        a.set_mode(AllocatorMode::Pool).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Pool);
        a.set_mode(AllocatorMode::Frozen).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Frozen);
        a.set_mode(AllocatorMode::Owned).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Owned);
    }
}
