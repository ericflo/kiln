//! `MetalAllocator` — the Metal [`Allocator`] impl.
//!
//! Mirrors [`crate::CudaAllocator`]'s contract: Owned/Pool/Frozen
//! mode handling, `warm()` for pre-Frozen pool population,
//! `reserved_bytes` / `peak_reserved_bytes` accounting. Backed by
//! [`MetalStorage::zeros_kt`] through the candle-free metal-rs
//! `Device` plumbing.
//!
//! GPU-only tests live behind `KILN_TENSOR_METAL_TEST=1` so the CI
//! compile path (which links `metal` against `candle-core` but has no
//! Metal device) doesn't spuriously fail.
//!
//! # Phase 7 candle-removal — CP-1 complete (allocator AND storage)
//!
//! Both the allocator surface and the storage's owned state are now
//! candle-free at the field level. `MetalAllocator::warm()` /
//! `alloc()` route through [`MetalStorage::zeros_kt`] — the candle-
//! free metal-rs entry — and the produced `MetalStorage` itself
//! holds only a `metal_handle: MetalRawDevice` (no candle wrapper in
//! its field state any more, dropped in the CP-1 final lift commit).
//!
//! **Mirror of the CudaAllocator + CudaStorage CP-1 chain** (commits
//! 03b8a34c add allocator companion, e2bddd72 swap allocation to
//! `zeros_ctx`, 6155e2e6 drop `CudaAllocator::candle_device` field,
//! fcc9bac7 drop CudaAllocator back-compat constructors, then on the
//! storage side b39f5712 add `context()` accessor, db916383 migrate
//! ops to `.context()`, 5c3cd353 drop `CudaStorage::candle_device`
//! field, 876e17da delete candle-typed back-compat constructors).
//! The Metal side now mirrors all of these shapes.
//!
//! Internal residual at the substrate boundary: the 7 in-file
//! substrate ops in `metal_storage.rs` still derive a candle
//! `MetalDevice` per call via [`crate::primary_metal_companion`] for
//! `kernels()` and `command_encoder()` access — those are the
//! candle-cached MSL pipeline collection and command-buffer pool
//! used by `candle_metal_kernels::call_*` FFI. The follow-up
//! substrate lift moves an `Arc<Kernels>` + `CommandQueue` companion
//! onto `MetalStorage` so the per-op call becomes a cheap field
//! clone; that lift is out of scope for the CP-1 commits.

use std::collections::HashMap;
use std::sync::Arc;

use crate::metal_rt::Device as MetalRawDevice;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, DType, Device, MetalStorage, Result, Storage,
};

#[derive(Debug)]
pub struct MetalAllocator {
    /// Metal-rs `Device` handle — the **candle-free** device handle
    /// this allocator forwards into [`MetalStorage::zeros_kt`] on
    /// every `warm()` / `alloc()`.
    ///
    /// The previously-held `candle_device: Arc<MetalDevice>` field
    /// was dropped alongside the allocation-path swap from
    /// `MetalStorage::zeros(candle_device, ...)` to
    /// `MetalStorage::zeros_kt(&metal_device_handle, ...)`. External
    /// callers needing a candle wrapper can derive one on demand via
    /// [`crate::primary_metal_companion`] using the stored
    /// `device_index`, or call `MetalStorage::candle_device()` on a
    /// produced storage (which itself derives via
    /// `primary_metal_device(device_index)` after the #1082 CP-1
    /// final lift — MetalStorage no longer holds a candle wrapper in
    /// its field state either).
    ///
    /// Mirror of [`crate::CudaAllocator::context`] (commit
    /// 03b8a34c added the companion; commit 6155e2e6 dropped the
    /// candle field; this allocator now matches that shape).
    metal_device_handle: MetalRawDevice,
    device_index: usize,
    mode: AllocatorMode,
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl MetalAllocator {
    /// Construct in `Owned` mode bound to `metal_device_handle` at the
    /// given Metal device index — the canonical, **candle-free**
    /// constructor entry.
    ///
    /// The previously-shipped candle-flavored `new` / `with_mode`
    /// entries (which extracted the metal-rs handle from a candle
    /// `MetalDevice`) were removed alongside the `metal_allocator`
    /// candle import drop (#1082); the only callers were in-source
    /// `#[cfg(test)]` and have been migrated to this entry. External
    /// callers needing a candle wrapper can derive one on demand via
    /// [`crate::primary_metal_companion`].
    ///
    /// Mirror of [`crate::CudaAllocator::new_ctx`].
    pub fn new_ctx(metal_device_handle: MetalRawDevice, device_index: usize) -> Self {
        MetalAllocator {
            metal_device_handle,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    /// Construct directly in a given mode (tests, capture session) —
    /// the **candle-free** entry.
    ///
    /// Mirror of [`crate::CudaAllocator::with_mode_ctx`].
    pub fn with_mode_ctx(
        metal_device_handle: MetalRawDevice,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new_ctx(metal_device_handle, device_index);
        a.mode = mode;
        a
    }

    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            // Route through the candle-free zeros_kt entry — after the
            // CP-1 final lift the resulting MetalStorage stores only a
            // metal-rs MetalRawDevice in its field state (the candle
            // wrapper, if needed downstream, is derived on demand via
            // `MetalStorage::candle_device()` -> `primary_metal_device`).
            // This allocator never touches the candle wrapper.
            let metal = MetalStorage::zeros_kt(
                &self.metal_device_handle,
                self.device_index,
                dtype,
                n_elements,
            )?;
            let storage: Storage = Arc::new(metal);
            slot.push(storage);
            self.reserved_bytes += bytes_per;
        }
        if self.reserved_bytes > self.peak_reserved_bytes {
            self.peak_reserved_bytes = self.reserved_bytes;
        }
        Ok(())
    }

    pub fn cache_len(&self, dtype: DType, n_elements: usize) -> usize {
        self.cache
            .get(&(dtype, n_elements))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Borrow the metal-rs `Device` companion handle — the canonical,
    /// candle-free device-handle accessor. Callers that need a candle
    /// `MetalDevice` wrapper can derive one on demand via
    /// [`crate::primary_metal_companion`] using
    /// [`Self::device_index()`], or call `.candle_device()` on a
    /// produced `MetalStorage` (which after the #1082 CP-1 final lift
    /// also derives via `primary_metal_device(device_index)` since
    /// the storage no longer holds a candle wrapper in its field).
    pub fn metal_device_handle(&self) -> &MetalRawDevice {
        &self.metal_device_handle
    }

    /// The Metal device ordinal this allocator is bound to.
    pub fn device_index(&self) -> usize {
        self.device_index
    }
}

impl Allocator for MetalAllocator {
    fn device(&self) -> Device {
        Device::Metal(self.device_index)
    }
    fn mode(&self) -> AllocatorMode {
        self.mode
    }
    fn set_mode(&mut self, mode: AllocatorMode) -> Result<()> {
        self.mode = mode;
        Ok(())
    }
    fn alloc(&mut self, dtype: DType, n_elements: usize) -> Result<Storage> {
        if let Some(slot) = self.cache.get_mut(&(dtype, n_elements)) {
            if let Some(s) = slot.pop() {
                return Ok(s);
            }
        }
        match self.mode {
            AllocatorMode::Frozen => Err(allocator_frozen_error(
                "MetalAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                // Route through the candle-free zeros_kt entry — see
                // the matching note in `warm()`.
                let metal = MetalStorage::zeros_kt(
                    &self.metal_device_handle,
                    self.device_index,
                    dtype,
                    n_elements,
                )?;
                let storage: Storage = Arc::new(metal);
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

    fn metal_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_METAL_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_metal_raw_device() -> Option<MetalRawDevice> {
        if !metal_test_enabled() {
            return None;
        }
        MetalRawDevice::system_default()
    }

    #[test]
    fn metal_allocator_starts_in_owned_mode() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let a = MetalAllocator::new_ctx(dev, 0);
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Metal(0));
        assert_eq!(a.reserved_bytes(), 0);
    }

    #[test]
    fn metal_pool_warm_serves_alloc() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let mut a = MetalAllocator::with_mode_ctx(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 16, 2).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 2);
        let _s = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 1);
    }

    #[test]
    fn metal_frozen_alloc_fails_on_cache_miss() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let mut a = MetalAllocator::with_mode_ctx(dev, 0, AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(e.to_string().contains("MetalAllocator::alloc"), "got: {e}");
    }
}
