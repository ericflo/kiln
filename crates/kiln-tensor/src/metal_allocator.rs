//! `MetalAllocator` — the Metal [`Allocator`] impl.
//!
//! Mirrors [`crate::CudaAllocator`]'s contract: Owned/Pool/Frozen
//! mode handling, `warm()` for pre-Frozen pool population,
//! `reserved_bytes` / `peak_reserved_bytes` accounting. Backed by
//! [`MetalStorage::zeros`] through the existing
//! `Arc<MetalDevice>` plumbing.
//!
//! GPU-only tests live behind `KILN_TENSOR_METAL_TEST=1` so the CI
//! compile path (which links `metal` against `candle-core` but has no
//! Metal device) doesn't spuriously fail.
//!
//! # Phase 7 candle-removal — STOP (blocked on `MetalStorage::zeros`)
//!
//! The original plan was for Phase 7 to swap `Arc<MetalDevice>` for a
//! direct metal-rs `Arc<MTLDevice>` + command-queue handle pair here.
//! After auditing the actual call graph (#1082 phase 7 audit), that
//! swap **cannot land in this file in isolation** — it is the exact
//! same shape as the [`crate::CudaAllocator`] STOP:
//!
//! 1. Every code path in `MetalAllocator` that touches the device
//!    handle (both `warm()` and `alloc()`) immediately forwards it to
//!    `MetalStorage::zeros(candle_device, device_index, dtype, n)`.
//! 2. `MetalStorage::zeros` itself stores `Arc<MetalDevice>` on the
//!    returned `MetalStorage` and uses it for
//!    `candle_device.allocate_zeros(byte_len)` — which reaches into
//!    candle's internal buffer cache (`MetalDevice.buffers:
//!    RwLock<HashMap<...>>`), allocates with `RESOURCE_OPTIONS`
//!    (`MTLStorageModeShared`), and zero-fills via a blit-command
//!    encoder owned by the `MetalDevice`. None of that state lives in
//!    metal-rs / objc2-metal directly — it is candle's own buffer-cache
//!    and command-queue plumbing layered on top of `MTLDevice`.
//! 3. The downstream `MetalStorage` accessor `candle_device()` is read
//!    by every kernel call site in `crates/kiln-tensor/src/metal_storage.rs`
//!    (e.g. `(*candle_device_arc).clone()` passed to
//!    `candle_metal_kernels::*`). Those FFI sites take a
//!    `MetalDevice` by value, not an `MTLDevice`. Swapping the type
//!    on `MetalAllocator` alone would force a fresh `MetalDevice`
//!    construction per `alloc()` call, which would break command-queue
//!    affinity with every kernel-crate handle that was set up against
//!    the original `MetalDevice` — and would lose the shared buffer
//!    cache that `allocate_zeros` relies on.
//! 4. `MetalAllocator::new` / `with_mode` have **zero external
//!    callers** today (`grep -rn 'MetalAllocator::new' crates/` —
//!    only the in-file constructors plus `pub use
//!    metal_allocator::MetalAllocator` in `lib.rs`). Swapping the
//!    constructor signature alone would not free any downstream from
//!    candle either.
//!
//! Therefore Phase 7 candle removal for the Metal allocator surface
//! is **blocked on `MetalStorage::zeros` migrating first** to take an
//! `Arc<metal::Device>` (and the rest of `MetalStorage` losing its
//! `candle_device: Arc<MetalDevice>` field). Once `MetalStorage`
//! accepts a raw `MTLDevice` + command-queue pair, this allocator
//! becomes a trivial type-substitution that recompiles with no
//! behavior change (the only thing changing is the type of the field
//! stored next to the `device_index`).
//!
//! Order-of-operations for the lift (tracked under #1082):
//!
//! 1. Add a parallel `MetalStorage::zeros_kt(dev: Arc<metal::Device>,
//!    queue: Arc<metal::CommandQueue>, ...)` that allocates a
//!    `Shared`-mode `MTLBuffer` directly via
//!    `dev.new_buffer(size, MTLResourceStorageModeShared)` and zero-
//!    fills via its own blit-command-encoder against `queue`,
//!    bypassing `MetalDevice.buffers` entirely (or carrying a kiln-
//!    owned cache).
//! 2. Migrate the multi-site kernel-crate FFI surface in
//!    `metal_storage.rs` (every `(*candle_device_arc).clone()` site
//!    feeding `candle_metal_kernels::*`) to take an `MTLDevice` +
//!    `CommandQueue` pair instead of a `MetalDevice`.
//! 3. Flip `MetalStorage::zeros` itself to take
//!    `Arc<metal::Device>` + `Arc<metal::CommandQueue>`, drop the
//!    `candle_device` field.
//! 4. Then this file changes one type
//!    (`candle_device: Arc<MetalDevice>` →
//!    `device: Arc<metal::Device>` + `queue: Arc<metal::CommandQueue>`)
//!    and one import (`use candle_metal_kernels::metal::Device;` or
//!    `use objc2_metal::ProtocolObject<dyn MTLDevice>;` replacing
//!    `use candle_core::MetalDevice;`).
//!
//! Doing step 4 first (the swap implied by the prior docstring) would
//! either require an `Arc<MTLDevice>` → `Arc<MetalDevice>` conversion
//! at every `alloc()` call (impossible without holding the
//! `MetalDevice` somewhere) or would silently break command-queue
//! affinity + the shared buffer cache across the kernel-crate FFI
//! surface. Both are worse than holding the substrate stable until
//! step 1 lands.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::MetalDevice;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, DType, Device, MetalStorage, Result, Storage,
};

#[derive(Debug)]
pub struct MetalAllocator {
    /// Candle Metal device handle. Held for command-queue affinity +
    /// the `allocate_zeros` helper (which reaches candle's internal
    /// buffer cache + blit-encoder). The Phase 7 swap to a direct
    /// metal-rs `MTLDevice` + `CommandQueue` pair is blocked on
    /// `MetalStorage::zeros` migrating first — see the STOP note at
    /// the top of this file.
    candle_device: Arc<MetalDevice>,
    device_index: usize,
    mode: AllocatorMode,
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl MetalAllocator {
    pub fn new(candle_device: Arc<MetalDevice>, device_index: usize) -> Self {
        MetalAllocator {
            candle_device,
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
            let metal = MetalStorage::zeros(
                self.candle_device.clone(),
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

    pub fn candle_device(&self) -> &Arc<MetalDevice> {
        &self.candle_device
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
                let metal = MetalStorage::zeros(
                    self.candle_device.clone(),
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
