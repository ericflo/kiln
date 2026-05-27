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
//! # Phase 7 candle-removal — partial UNBLOCK (allocation path candle-free)
//!
//! The allocation path is now candle-free at this allocator's
//! surface. Both `warm()` and `alloc()` route through
//! [`MetalStorage::zeros_kt`] — the candle-free metal-rs entry — and
//! the previously-held `candle_device: Arc<MetalDevice>` field has
//! been dropped. `MetalAllocator` now carries only the metal-rs
//! `metal_device_handle: MetalRawDevice` companion on the device-
//! handle side.
//!
//! **Mirror of the CudaAllocator chain** (commits 03b8a34c add
//! companion, e2bddd72 swap allocation to `zeros_ctx`, 6155e2e6
//! drop `candle_device` field, fcc9bac7 drop back-compat
//! constructors). The Metal side now mirrors all four shapes —
//! `metal_device_handle` companion already lived here from the
//! additive step, the allocation swap + field drop both land in this
//! commit, and the back-compat constructor drop follows in a sibling
//! commit.
//!
//! **CP-1 on the allocator side is now structurally complete.** The
//! produced `MetalStorage` still carries a candle `MetalDevice` for
//! kernel-crate FFI (every `(*candle_device_arc).clone()` site in
//! `kiln-model::backend::metal` feeds `candle_metal_kernels::*` by
//! value). The storage-side field drop (CP-2) is gated on migrating
//! those ~232 FFI sites to a raw `MTLDevice` + `MTLCommandQueue`
//! surface — that is the next, much larger refactor and stays out
//! of scope here. The internal residual at the substrate boundary
//! is now confined to [`crate::primary_metal_device`] inside
//! `MetalStorage::zeros_kt`, which derives an `Arc<MetalDevice>`
//! to populate the storage's back-compat field; that helper retires
//! when the storage-side field is dropped.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::MetalDevice;
use candle_metal_kernels::metal::Device as MetalRawDevice;

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
    /// [`crate::primary_metal_device`] using the stored
    /// `device_index`, or read it from the produced
    /// `MetalStorage.candle_device()` (which is still load-bearing
    /// for the kernel-crate FFI sites that consume `MetalDevice` by
    /// value — see `metal_storage.rs` field doc).
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
    /// Construct in `Owned` mode bound to `candle_device` at the
    /// given Metal device index.
    ///
    /// Back-compat shim: extracts the metal-rs `Device` companion via
    /// `candle_device.metal_device().clone()` and forwards to the
    /// candle-free allocation path. The `candle_device` argument is
    /// otherwise unused — the allocator no longer stores it. See the
    /// `metal_device_handle` field doc.
    pub fn new(candle_device: Arc<MetalDevice>, device_index: usize) -> Self {
        let metal_device_handle = candle_device.metal_device().clone();
        MetalAllocator {
            metal_device_handle,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    pub fn with_mode(
        candle_device: Arc<MetalDevice>,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new(candle_device, device_index);
        a.mode = mode;
        a
    }

    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            // Route through the candle-free zeros_kt entry — the
            // resulting MetalStorage still carries a candle MetalDevice
            // internally (derived via primary_metal_device for the
            // back-compat field), but this allocator no longer touches
            // the candle wrapper directly.
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
    /// [`crate::primary_metal_device`] using
    /// [`Self::device_index()`], or read it from a produced
    /// `MetalStorage.candle_device()`.
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
    use candle_core::Device as CandleDevice;

    fn metal_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_METAL_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_metal_device() -> Option<Arc<MetalDevice>> {
        if !metal_test_enabled() {
            return None;
        }
        match CandleDevice::new_metal(0).ok()? {
            CandleDevice::Metal(d) => Some(Arc::new(d)),
            _ => None,
        }
    }

    #[test]
    fn metal_allocator_starts_in_owned_mode() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let a = MetalAllocator::new(dev, 0);
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Metal(0));
        assert_eq!(a.reserved_bytes(), 0);
    }

    #[test]
    fn metal_pool_warm_serves_alloc() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let mut a = MetalAllocator::with_mode(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 16, 2).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 2);
        let _s = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 1);
    }

    #[test]
    fn metal_frozen_alloc_fails_on_cache_miss() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let mut a = MetalAllocator::with_mode(dev, 0, AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(e.to_string().contains("MetalAllocator::alloc"), "got: {e}");
    }
}
