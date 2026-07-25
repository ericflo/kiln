//! CUDA storage impl behind the `cuda` feature flag.
//!
//! Wraps `cudarc::driver::CudaSlice<u8>` (the actual buffer) + dtype +
//! `Arc<candle_core::cuda_backend::CudaDevice>` for stream affinity.
//!
//! # Anti-pattern 1 compliance
//!
//! Per the issue:
//!
//! > `kiln-tensor` is not a candle wrapper. Storage is
//! > `cudarc::CudaSlice` directly. No `candle_core::Tensor` field on
//! > `kiln_tensor::Tensor`.
//!
//! `CudaStorage` does **not** hold a `candle_core::Tensor`. The buffer
//! is a `CudaSlice<u8>` we own. The candle `CudaDevice` is held only
//! for its `cuda_stream()` accessor + its `alloc_zeros::<T>` helper;
//! that's the same pattern in use across `kiln-rmsnorm-kernel`,
//! `kiln-gdn-kernel`, `kiln-marlin-gemm`, `kiln-flash-attn`, etc.
//! Phase 7 of #1082 (candle removal) replaces `Arc<CudaDevice>` with a
//! direct `Arc<cudarc::driver::CudaContext>` + `Arc<CudaStream>`.
//!
//! # Phase 1.6 scope (storage layer only)
//!
//! - `zeros(device, dtype, n_elements)` — async device alloc + memset.
//! - `from_slice(device, dtype, slice)` — take ownership of an
//!   existing `CudaSlice<u8>` allocated through candle's device. This
//!   is the FFI seam that today's kernel crates plug into.
//! - `StorageBackend` impl — `device() / dtype() / byte_len() / as_any()`.
//! - `slice()` / `slice_mut()` accessors for the existing kernel-crate
//!   FFI sites that want raw byte pointers.
//!
//! Math ops, H2D/D2H helpers, pinned-host staging — separate later PRs.

use std::any::Any;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUdeviceptr;

use crate::{DType, Device, Error, Result, StorageBackend};

/// Owner of a CUDA byte buffer. Either kt owns the allocation
/// outright (`Owned`) or kt is sharing a buffer that some other type
/// owns (`Borrowed` — e.g. a candle `CudaStorage` held alive via the
/// `_keep_alive` Arc).
///
/// The Borrowed variant is the foundation for the Phase 7 zero-copy
/// candle→kt adapter: it lets a kt-Tensor wrap a candle Tensor's
/// device buffer without copying, while the Arc keeps the candle side
/// alive for as long as the kt side needs the bytes. Drop semantics:
/// dropping a Borrowed `CudaStorage` just decrements the keep-alive
/// Arc — it never frees the device memory directly.
pub(crate) enum SliceOwner {
    Owned(CudaSlice<u8>),
    /// Borrowed view over an externally-owned CUDA buffer.
    ///
    /// `_keep_alive` is an opaque Arc that must outlive every read
    /// from `ptr`. Typically holds an Arc-wrapped `candle::Storage` so
    /// the candle side's CudaSlice<T> Drop runs only after kt drops
    /// its references.
    Borrowed {
        ptr: CUdeviceptr,
        byte_len: usize,
        _keep_alive: Arc<dyn Any + Send + Sync>,
    },
}

impl std::fmt::Debug for SliceOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(s) => f.debug_struct("Owned").field("len", &s.len()).finish(),
            Self::Borrowed { ptr, byte_len, .. } => f
                .debug_struct("Borrowed")
                .field("ptr", &format_args!("0x{ptr:x}"))
                .field("byte_len", byte_len)
                .finish(),
        }
    }
}

/// CUDA-backed storage. Byte-typed; dtype carried alongside for dispatch.
///
/// The handed-down `CudaSlice<u8>` is allocated via cudarc directly
/// (`CudaContext::default_stream().alloc_zeros::<u8>`); the storage no
/// longer holds an `Arc<CudaDevice>` field — the candle wrapper, when
/// needed by legacy FFI sites, is derived on-demand from `device_index`
/// via [`primary_cuda_device`]. This was the #1082 CP-1 final lift:
/// dropping the candle device field in favor of the cudarc context.
///
/// Storage can be either owned (allocated by kt) or borrowed (sharing
/// an external CUDA buffer with a keep-alive Arc) — see [`SliceOwner`].
#[derive(Debug)]
pub struct CudaStorage {
    /// Device-index variant of [`Device`]. Stored explicitly so
    /// `StorageBackend::device()` is O(1) (no context-query syscall).
    device: Device,
    /// Element dtype tag.
    dtype: DType,
    /// The byte buffer (owned or borrowed).
    slice: SliceOwner,
    /// Cudarc CUDA context this storage was allocated on. Held for
    /// stream affinity — every kernel-launch path reads
    /// `self.ctx.default_stream()` to get the primary stream handle.
    /// Replaces the previous `candle_device: Arc<CudaDevice>` field
    /// (#1082 CP-1 final lift). With the `.candle_device()` accessor
    /// removed (#1082 aggressive cleanup), callers that need a candle
    /// `Arc<CudaDevice>` must derive one externally via
    /// [`primary_cuda_device`] from `self.device()`.
    ctx: Arc<CudaContext>,
}

impl CudaStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on the cudarc
    /// `CudaContext` `ctx`, **candle-free** in the allocation path.
    ///
    /// Buffer is zero-initialized via `ctx.default_stream()
    /// .alloc_zeros::<u8>(byte_len)` — i.e. straight through cudarc with
    /// no `candle_core::cuda_backend::CudaDevice` involvement on the
    /// allocation side.
    ///
    /// `device_index` is the CUDA device ordinal — must match the
    /// ordinal of `ctx`'s owning device. Stored as the
    /// [`Device::Cuda`] variant.
    ///
    /// #1082: this is now the **sole** zeros constructor on `CudaStorage`.
    /// The candle-typed `Self::zeros(candle_device, ...)` back-compat
    /// wrapper has been deleted; the free function [`cuda_zeros`] still
    /// accepts a candle device for external callers (kiln-model) and
    /// internally derives the ctx and routes here.
    pub fn zeros_ctx(
        ctx: &Arc<CudaContext>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        // #1082 freeze-pointers (boxes 98-101): when a capture arena is
        // active on this thread, route through it — the forward gets a
        // Borrowed view into an arena-retained buffer whose device pointer
        // stays valid across every graph replay. No-op (None) outside capture.
        if let Some(result) = crate::capture_arena_alloc(dtype, n_elements, true) {
            return result;
        }
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        // Candle-free allocation through cudarc. #1082 CUDA-graph fix:
        // route through the thread-local active stream so the alloc is
        // captured on the capture stream during graph capture; outside a
        // capture scope this is exactly `ctx.default_stream()`.
        let slice = crate::active_cuda_stream(ctx)
            .alloc_zeros::<u8>(byte_len)
            .map_err(|e| {
                Error::Msg(format!(
                    "CudaStorage::zeros_ctx: active_cuda_stream(ctx).alloc_zeros::<u8>({byte_len}) \
                     failed: {e:?}"
                ))
            })?;
        // #1082 CP-1 final lift: field is `ctx: Arc<CudaContext>` — no
        // candle device materialization needed in the storage.
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            ctx: ctx.clone(),
        })
    }

    /// (#1082 perf — Pattern A) Allocate `n_elements` of `dtype`
    /// **UNINITIALIZED** on the cudarc `CudaContext` `ctx`.
    ///
    /// Identical to [`Self::zeros_ctx`] except it skips the
    /// `cudaMemsetAsync` zero-fill (`alloc_zeros` → `alloc`). That zero-fill
    /// is pure waste whenever the producer overwrites 100% of the buffer —
    /// e.g. a GEMM with `beta = 0` (`Epilogue::Identity`), or a full-tensor
    /// elementwise / cast / copy that writes every element. The audit flagged
    /// "alloc-zeros then fully overwrite" as the highest-frequency baked-in
    /// waste on the decode + train paths.
    ///
    /// # Caller contract
    /// The returned storage's device bytes are uninitialized; the caller MUST
    /// fully overwrite the buffer before any read. For accumulation /
    /// partial-write outputs (read-before-write), use [`Self::zeros_ctx`].
    pub fn alloc_uninit_ctx(
        ctx: &Arc<CudaContext>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        // #1082 freeze-pointers: route uninitialized allocs through the active
        // capture arena too (zero = false → no captured memset). No-op outside
        // capture.
        if let Some(result) = crate::capture_arena_alloc(dtype, n_elements, false) {
            return result;
        }
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        // SAFETY: cudarc's `alloc` returns uninitialized device memory. The
        // caller contract (documented above) requires a full overwrite before
        // any read, so the uninitialized contents are never observed.
        // #1082 CUDA-graph fix: route through the thread-local active
        // stream so the alloc is captured on the capture stream during
        // graph capture; outside a capture scope this is exactly
        // `ctx.default_stream()`.
        let slice = unsafe { crate::active_cuda_stream(ctx).alloc::<u8>(byte_len) }.map_err(|e| {
            Error::Msg(format!(
                "CudaStorage::alloc_uninit_ctx: active_cuda_stream(ctx).alloc::<u8>({byte_len}) \
                 failed: {e:?}"
            ))
        })?;
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            ctx: ctx.clone(),
        })
    }

    /// Wrap an existing `CudaSlice<u8>` allocated by the caller —
    /// **candle-free** entry point.
    ///
    /// Validates slice length against `dtype.size_in_bytes()` for
    /// non-packed dtypes (must be a multiple); packed dtypes have no
    /// per-element alignment. Takes a cudarc `Arc<CudaContext>` directly
    /// — no candle `CudaDevice` materialization happens on the storage
    /// side.
    ///
    /// #1082: this is now the **sole** from-slice constructor on
    /// `CudaStorage`. The candle-typed `Self::from_slice(candle_device, ...)`
    /// back-compat wrapper has been deleted.
    pub fn from_slice_ctx(
        ctx: &Arc<CudaContext>,
        device_index: usize,
        dtype: DType,
        slice: CudaSlice<u8>,
    ) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !slice.len().is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "CudaStorage::from_slice_ctx: slice len {} is not a multiple of \
                     size_in_bytes({:?}) = {}",
                    slice.len(),
                    dtype,
                    per
                )));
            }
        }
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            ctx: ctx.clone(),
        })
    }

    /// Wrap an externally-owned CUDA buffer as a kt `CudaStorage`
    /// without copying — **candle-free** entry point.
    ///
    /// `keep_alive` is an opaque Arc that must outlive every read
    /// from `device_ptr`. Typical pattern: pass an Arc-wrapped candle
    /// `Storage::Cuda(...)` so the candle Tensor's underlying
    /// `CudaSlice<T>` drop runs after this storage's last reference.
    ///
    /// `device_ptr` + `byte_len` describe the borrowed region. The
    /// caller is responsible for the byte_len matching dtype × element
    /// count (this constructor does the same alignment check as
    /// [`Self::from_slice_ctx`]).
    ///
    /// The Phase 7 zero-copy candle→kt adapter is the canonical
    /// caller. Kernel-crate kt-API sites that reach `.slice()` will
    /// panic on a borrowed storage — they must migrate to the
    /// dtype/owner-aware accessor that lands alongside the adapter.
    ///
    /// #1082: this is now the **sole** from-borrowed constructor on
    /// `CudaStorage`. The candle-typed `Self::from_borrowed(candle_device, ...)`
    /// back-compat wrapper has been deleted.
    pub fn from_borrowed_ctx(
        ctx: &Arc<CudaContext>,
        device_index: usize,
        dtype: DType,
        device_ptr: CUdeviceptr,
        byte_len: usize,
        keep_alive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !byte_len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "CudaStorage::from_borrowed_ctx: byte_len {byte_len} is not a multiple of \
                     size_in_bytes({dtype:?}) = {per}"
                )));
            }
        }
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice: SliceOwner::Borrowed {
                ptr: device_ptr,
                byte_len,
                _keep_alive: keep_alive,
            },
            ctx: ctx.clone(),
        })
    }

    /// Whether this storage owns its underlying CUDA buffer (`true`)
    /// or just borrows it from an external Arc keep-alive (`false`).
    pub fn is_owned(&self) -> bool {
        matches!(self.slice, SliceOwner::Owned(_))
    }

    /// Whether this storage borrows its underlying CUDA buffer from
    /// an external owner (Phase 7 candle adapter), as opposed to
    /// owning its own allocation.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.slice, SliceOwner::Borrowed { .. })
    }

    /// Borrow the underlying byte slice. The existing kernel-crate
    /// FFI sites that want the raw device pointer reach this then
    /// call `.device_ptr(&stream)` per the cudarc 0.19 pattern.
    ///
    /// **Panics** if this is a `Borrowed` storage (there is no
    /// `CudaSlice<u8>` to return — call sites must use the dtype/
    /// owner-aware raw-pointer accessor that lands alongside the
    /// Phase 7 zero-copy adapter migration).
    pub fn slice(&self) -> &CudaSlice<u8> {
        match &self.slice {
            SliceOwner::Owned(s) => s,
            SliceOwner::Borrowed { .. } => panic!(
                "CudaStorage::slice() called on Borrowed storage; call sites must use the \
                 raw-pointer accessor that supports both owners"
            ),
        }
    }

    /// Mutable borrow for in-place ops. Bumps no version counter at
    /// this layer (anti-pattern 16 versioning is a Tensor-level concern,
    /// enforced once `kiln-autograd` lands).
    ///
    /// **Panics** if this is a `Borrowed` storage — borrowed buffers
    /// are not safe to mutate through the kt side (the external owner
    /// dictates write semantics).
    pub fn slice_mut(&mut self) -> &mut CudaSlice<u8> {
        match &mut self.slice {
            SliceOwner::Owned(s) => s,
            SliceOwner::Borrowed { .. } => panic!(
                "CudaStorage::slice_mut() called on Borrowed storage; borrowed buffers are \
                 read-only through kt"
            ),
        }
    }

    /// Raw device pointer at the start of the storage's byte buffer.
    /// Works for both `Owned` and `Borrowed` variants.
    ///
    /// Returns `(ptr, byte_len)`. Callers typically add the
    /// kt-Tensor's `layout.start_offset() * dtype.size_in_bytes()` to
    /// reach the active region.
    ///
    /// Note: this returns a raw `CUdeviceptr` without a sync guard.
    /// Callers writing through the pointer must respect kiln's stream
    /// affinity — they are already in `unsafe` FFI territory.
    pub fn device_ptr_raw(&self) -> (CUdeviceptr, usize) {
        match &self.slice {
            SliceOwner::Owned(s) => {
                use cudarc::driver::DevicePtr;
                // Use the active stream's device_ptr just to extract the raw
                // bits; the SyncOnDrop is dropped immediately, recording
                // nothing. #1082 CUDA-graph fix: outside a capture scope this
                // is exactly `self.ctx.default_stream()`.
                let stream = crate::active_cuda_stream(&self.ctx);
                let (ptr, _g) = s.device_ptr(&stream);
                (ptr, s.len())
            }
            SliceOwner::Borrowed { ptr, byte_len, .. } => (*ptr, *byte_len),
        }
    }

    // #1082: the previous private `device_index()` helper was retired
    // alongside the public `candle_device()` accessor. Callers should
    // pattern-match on `self.device()` directly (or read `self.context()`
    // when they only need stream affinity).

    /// The underlying cudarc `CudaContext` this storage was allocated
    /// on — **candle-free passthrough**.
    ///
    /// After the #1082 CP-1 final lift the storage holds an
    /// `Arc<CudaContext>` directly, so this is a cheap Arc clone —
    /// no candle wrapper materialization, no `cuda_stream()` chain.
    pub fn context(&self) -> Arc<CudaContext> {
        self.ctx.clone()
    }

    /// Raw CUDA stream pointer for FFI dispatch — **candle-free
    /// passthrough**.
    ///
    /// Returns the underlying `CUstream` handle cast to
    /// `*mut core::ffi::c_void`, which is the type every kernel
    /// crate's FFI declaration expects for the `stream` argument.
    /// Callers don't need a `candle_core` or `cudarc` dependency
    /// to plumb the stream into a kernel launch — they just take
    /// the raw pointer and pass it through.
    ///
    /// This is the substrate-side accessor that unblocks dropping
    /// `candle-core` from the kernel crates' `[dependencies]` blocks
    /// (#1082 Tier 1 closure). Without it, every `kt_api.rs` is
    /// forced to import `candle_core::cuda_backend::cudarc::driver::
    /// DevicePtr` just to call `candle_device().cuda_stream()
    /// .cu_stream() as *mut c_void`.
    ///
    /// Stream lifetime: the returned pointer is valid for the
    /// lifetime of `self`. Callers passing it to a CUDA FFI must
    /// not store it past the borrow.
    pub fn cuda_stream_raw(&self) -> *mut core::ffi::c_void {
        // #1082 CUDA-graph fix: resolve through the thread-local active
        // stream so kernel launches land on the capture stream during
        // graph capture. Outside a capture scope this returns
        // `self.ctx.default_stream()` — identical to the prior behavior.
        let stream = crate::active_cuda_stream(&self.ctx);
        stream.cu_stream() as *mut core::ffi::c_void
    }

    /// Crate-internal accessor for the slice owner. Used by sibling
    /// FFI modules (`fp8.rs`, etc.) that need to extract a device
    /// pointer without going through `device_ptr_raw` (which discards
    /// the stream guard).
    pub(crate) fn slice_owner(&self) -> &SliceOwner {
        &self.slice
    }
}

/// Construct the primary `Arc<cudarc::driver::CudaContext>` for the
/// given device index — **fully candle-free** accessor.
///
/// Wraps `cudarc::driver::CudaContext::new(device_index)`. Returns
/// `Err` if the requested CUDA device isn't available (no driver, no
/// GPU at that ordinal, etc.).
///
/// This is the #1082 replacement for [`primary_cuda_device`]: where
/// the old helper bounced through `candle_core::Device::new_cuda`
/// (which itself just calls `CudaContext::new` under the hood),
/// `primary_cuda_context` skips the candle round-trip entirely. The
/// resulting `Arc<CudaContext>` is the same primary-context retain
/// every kernel crate already uses via `device.cuda_stream().context()`.
///
/// **Use this for:**
/// - `tests/*.rs` CUDA-availability probes:
///   `primary_cuda_context(0).is_ok()` instead of
///   `primary_cuda_device(0).is_ok()`.
/// - Any candle-free call site that needs a `CudaContext` to drive
///   `default_stream().alloc_*` or `memcpy_*` directly.
///
/// `primary_cuda_device` stays around only as long as
/// `kiln-kt-bridge::to_candle` needs a candle `CudaDevice` for its
/// `candle_core::Tensor::zeros` allocation.
#[cfg(feature = "cuda")]
pub fn primary_cuda_context(device_index: usize) -> Result<Arc<CudaContext>> {
    CudaContext::new(device_index)
        .map_err(|e| Error::Msg(format!("primary_cuda_context({device_index}): {e}")))
}

/// Device-reported `(free, total)` CUDA memory via `cuMemGetInfo`. The CUDA
/// analog of [`crate::rocm_mem_get_info`] — driver ground truth for the active
/// GPU, used to back the memory governor on discrete NVIDIA cards.
pub fn cuda_mem_get_info(device_index: usize) -> Result<(usize, usize)> {
    let ctx = primary_cuda_context(device_index)?;
    ctx.bind_to_thread()
        .map_err(|e| Error::Msg(format!("cuda_mem_get_info bind({device_index}): {e}")))?;
    ctx.mem_get_info()
        .map_err(|e| Error::Msg(format!("cuda_mem_get_info({device_index}): {e}")))
}

/// Return pooled-but-unused CUDA VRAM to the OS, keeping at least
/// `min_keep_bytes` cached. The CUDA analog of [`crate::rocm_trim_pool`]: the
/// governor's reclaim hook for NVIDIA. **Device-synchronizes first** so no
/// in-flight kernel is reading a block being released (race-free, mirroring the
/// ROCm path). On a DISCRETE GPU `cuMemPoolTrimTo` actually returns VRAM to the
/// OS so a coexisting process / training reservation gets headroom; best-effort
/// no-op when the runtime has no stream-ordered mempool (the pool is empty).
pub fn cuda_trim_pool(device_index: usize, min_keep_bytes: usize) -> Result<()> {
    use cudarc::driver::sys;
    let ctx = primary_cuda_context(device_index)?;
    ctx.bind_to_thread()
        .map_err(|e| Error::Msg(format!("cuda_trim_pool bind({device_index}): {e}")))?;
    // Drain in-flight work before releasing pooled pages (see rocm_trim_pool).
    crate::cuda_synchronize_context_for(device_index, &ctx, crate::CudaSyncReason::MemoryReclaim)
        .map_err(|e| Error::Msg(format!("cuda_trim_pool sync({device_index}): {e}")))?;
    let dev = ctx.cu_device();
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: `dev` is a valid CUdevice from the live context; `pool` is an
    // out-param. Both calls are best-effort — failure is ignored (no mempool).
    unsafe {
        if sys::cuDeviceGetDefaultMemPool(&mut pool, dev) == sys::CUresult::CUDA_SUCCESS
            && !pool.is_null()
        {
            let _ = sys::cuMemPoolTrimTo(pool, min_keep_bytes);
        }
    }
    Ok(())
}

/// Make the device's default stream-ordered mempool HOARD freed allocations
/// instead of returning them to the OS at every stream sync.
///
/// cudarc allocates via `cuMemAllocAsync` (the default mempool) and frees via
/// `cuMemFreeAsync`. The pool's `RELEASE_THRESHOLD` defaults to 0, which means
/// the driver releases all unused pages back to the OS on each synchronize — so
/// every alloc/free churn pays an OS round-trip AND [`cuda_trim_pool`] has
/// nothing left to reclaim (the governor's CUDA reclaimer is a redundant no-op).
///
/// Raising the threshold (to `u64::MAX` here) makes the pool keep freed pages
/// for fast reuse (the perf win), and turns [`cuda_trim_pool`] into the real,
/// governor-driven release valve: under memory pressure the reclaimer trims the
/// hoarded pool back to the OS for a coexisting process. This is the same
/// hoard-then-reclaim-under-pressure model the ROCm path targets.
///
/// Idempotent and best-effort: a failure (no mempool support) leaves the default
/// behaviour, which is still correct (just less perf-y / reclaimer redundant).
pub fn cuda_set_pool_release_threshold(device_index: usize, threshold_bytes: u64) -> Result<()> {
    use cudarc::driver::sys;
    let ctx = primary_cuda_context(device_index)?;
    ctx.bind_to_thread().map_err(|e| {
        Error::Msg(format!(
            "cuda_set_pool_release_threshold bind({device_index}): {e}"
        ))
    })?;
    let dev = ctx.cu_device();
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    let mut threshold = threshold_bytes;
    // SAFETY: `dev` is a valid CUdevice from the live context; `pool` is an
    // out-param; `&mut threshold` outlives the SetAttribute call. Best-effort.
    unsafe {
        if sys::cuDeviceGetDefaultMemPool(&mut pool, dev) == sys::CUresult::CUDA_SUCCESS
            && !pool.is_null()
        {
            let _ = sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut threshold as *mut u64 as *mut std::ffi::c_void,
            );
        }
    }
    Ok(())
}

impl StorageBackend for CudaStorage {
    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        match &self.slice {
            SliceOwner::Owned(s) => s.len(),
            SliceOwner::Borrowed { byte_len, .. } => *byte_len,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a fresh [`crate::Storage`] handle (`Arc<dyn StorageBackend>`)
/// holding a [`CudaStorage`]. Convenience constructor matching the CPU
/// `cpu_zeros` helper.
///
/// **#1082 candle-removal**: this is now the only public `cuda_zeros*`
/// entry. The previous candle-typed
/// `cuda_zeros(Arc<CudaDevice>, usize, DType, usize)` wrapper was
/// deleted in wave 13 of the #1082 sweep; its sole external caller
/// (`kiln-kt-bridge::kt_tensor_from_candle_cuda_copy`) was migrated
/// to this candle-free variant.
///
/// Allocations route through [`CudaStorage::zeros_ctx`], which uses
/// `ctx.default_stream().alloc_zeros::<u8>()` directly via cudarc —
/// no candle alloc path is touched.
#[cfg(feature = "cuda")]
pub fn cuda_zeros_ctx(
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    // Derive the cudarc CudaContext for the given ordinal directly —
    // primary_cuda_context(device_index) just calls
    // CudaContext::new(device_index), which is exactly the primary-
    // context retain candle_core::Device::new_cuda used to perform.
    let ctx = primary_cuda_context(device_index)?;
    let storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
}

// ----------------------------------------------------------------------
// CUDA-side Tensor::contiguous (Phase 1 substrate op)
// ----------------------------------------------------------------------
//
// The kernel itself lives in `csrc/contiguous.cu` and is compiled by
// `build.rs` when `--features cuda` is on. See the kernel header
// comment for the algorithm and launch shape.

unsafe extern "C" {
    fn kiln_contiguous_copy_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        shape: *const i64,
        strides_e: *const i64,
        rank: i32,
        bytes_per_elem: i32,
        n_elements: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_index_select_dim0_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        row_bytes: i64,
        n_indices: i64,
        src_n_rows: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_index_select_axis_n_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        right_bytes: i64,
        ids_dim: i64,
        src_dim: i64,
        left_size: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_lerp_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        weight: f32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_binary_minmax_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_elementwise_binary_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_activation_unary_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_scalar_op_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        c: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_cast_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        n_elements: i64,
        cast_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_softmax_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_log_softmax_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_log_softmax_last_axis_f32_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        input_dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_sum_squared_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_is_finite_storage_async(
        x: *const core::ffi::c_void,
        out_flag: *mut core::ffi::c_void,
        n_elements: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_argmax_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_topk_last_axis_async(
        x: *const core::ffi::c_void,
        out_vals: *mut core::ffi::c_void,
        out_indices: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        k: i32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_prompt_logprobs_async(
        x: *const core::ffi::c_void,
        observed_ids: *const i64,
        out_row_max: *mut core::ffi::c_void,
        out_log_sum: *mut core::ffi::c_void,
        out_observed_logit: *mut core::ffi::c_void,
        out_observed_rank: *mut core::ffi::c_void,
        out_top_logits: *mut core::ffi::c_void,
        out_top_indices: *mut core::ffi::c_void,
        out_invalid_kind: *mut core::ffi::c_void,
        out_invalid_column: *mut core::ffi::c_void,
        out_invalid_value: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        top_k: i32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_scan_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_l2norm_apply_async(
        x: *const core::ffi::c_void,
        sum_sq_f32: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        eps: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_rmsnorm_last_axis_async(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        eps: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_layernorm_last_axis_async(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        bias: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        eps: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_masked_fill_u8_async(
        x: *const core::ffi::c_void,
        mask: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        fill_value: f32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_dropout_async(
        x: *const core::ffi::c_void,
        y: *mut core::ffi::c_void,
        mask: *mut core::ffi::c_void,
        n_elements: i64,
        p: f32,
        inv_keep: f32,
        seed: u64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_scatter_add_dim0_async(
        updates: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        n_indices: i64,
        row_inner: i64,
        target_dim: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_cross_entropy_loss_async(
        logits: *const core::ffi::c_void,
        targets: *const core::ffi::c_void,
        row_loss_f32: *mut core::ffi::c_void,
        row_err_i32: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        targets_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_flce_grad_logits_chunk_f32_async(
        logits: *mut core::ffi::c_void,
        labels: *const core::ffi::c_void,
        global_max: *const core::ffi::c_void,
        global_sumexp: *const core::ffi::c_void,
        num_active: i64,
        chunk_len: i64,
        chunk_start: i64,
        scale: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_grpo_grad_logits_chunk_f32_async(
        logits: *mut core::ffi::c_void,
        labels: *const core::ffi::c_void,
        global_max: *const core::ffi::c_void,
        global_sumexp: *const core::ffi::c_void,
        coeff: *const core::ffi::c_void,
        num_active: i64,
        chunk_len: i64,
        chunk_start: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_sum_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        divisor: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_sum_arbitrary_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        outer: i64,
        axis_dim: i64,
        inner: i64,
        divisor: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_minmax_arbitrary_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        outer: i64,
        axis_dim: i64,
        inner: i64,
        kind: i32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_bool_reduce_arbitrary_axis_async(
        mask: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        outer: i64,
        axis_dim: i64,
        inner: i64,
        kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_concat_async(
        dst: *mut core::ffi::c_void,
        src_ptrs: *const *const core::ffi::c_void,
        t_axis_lens: *const i64,
        n_inputs: i32,
        outer: i64,
        inner_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_rope_async(
        x_in: *const core::ffi::c_void,
        x_out: *mut core::ffi::c_void,
        cos: *const core::ffi::c_void,
        sin: *const core::ffi::c_void,
        leading: i64,
        seq: i64,
        head_dim: i64,
        pair_count: i64,
        x_dtype: i32,
        cs_dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_rope_split_half_4d_async(
        x_in: *const core::ffi::c_void,
        x_out: *mut core::ffi::c_void,
        cos: *const core::ffi::c_void,
        sin: *const core::ffi::c_void,
        batch: i64,
        seq: i64,
        heads: i64,
        head_dim: i64,
        rotary_dim: i64,
        x_dtype: i32,
        cs_dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_clamp_pow_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        a: f32,
        b: f32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_compare_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_where_select_async(
        mask: *const core::ffi::c_void,
        t: *const core::ffi::c_void,
        f: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_diagonal_extract_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n: i64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_diag_build_async(
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n: i64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// CUDA-side stride-aware contiguous() — produces a new kt-Tensor with
/// the same logical shape + dtype as `src` but row-major contiguous
/// strides and `start_offset = 0`.
///
/// Routes through the `kiln_contiguous_copy_async` kernel. Caller is
/// the dispatch site in `Tensor::contiguous()`.
///
/// # Materializes transposed / permuted views (#1082)
///
/// `Tensor::transpose(d0, d1)` and `Tensor::permute(&axes)` are
/// zero-copy layout ops — they only permute `Layout::shape` and
/// `Layout::strides` (see `Layout::{transpose,permute}` in
/// `kiln-tensor::layout`). The transposed/permuted view shares
/// storage with its parent.
///
/// Calling `.contiguous()` on such a view dispatches here, and
/// `kiln_contiguous_copy_async` materializes the permuted layout
/// because it is fully stride-aware: per output element it
/// unflattens the linear index against the output shape and
/// accumulates the source byte offset via the input's element
/// strides plus `layout.start_offset()`. As a result, there is
/// **no dedicated `cuda_transpose` / `cuda_permute` kernel**, and
/// no `TransposeOp` / `PermuteOp` exists in `kiln-tensor::ops`.
/// Parity is locked in by
/// `crates/kiln-kt-bridge/tests/cuda_transpose_parity.rs`.
///
/// Errors:
/// - Source must be CUDA storage (downcast to `CudaStorage`).
/// - Packed dtypes are not supported (Marlin / Int4Packed / Fp4Packed
///   have no element-wise interpretation).
/// - Rank must be ≤ 8 (matches the kernel's `MAX_RANK`).
#[cfg(feature = "cuda")]
pub fn cuda_contiguous(src: &crate::Tensor) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if src.dtype().is_packed() {
        return Err(crate::Error::Msg(
            "cuda_contiguous: packed dtype not supported".to_string(),
        ));
    }

    let layout = src.layout();
    let shape = src.shape();
    let strides_elems = src.strides();
    let rank = shape.len();
    if rank > 8 {
        return Err(crate::Error::Msg(format!(
            "cuda_contiguous: rank {rank} exceeds kernel MAX_RANK=8"
        )));
    }
    let n_elements = src.element_count();
    let bpe = src.dtype().size_in_bytes();

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_contiguous: source must be CUDA storage".to_string())
        })?;

    // Allocate the destination — contiguous, same dtype, same shape.
    let ctx = src_storage.context();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };
    let dst_storage = CudaStorage::zeros_ctx(&ctx, device_index, src.dtype(), n_elements)?;

    // Extract raw device pointers. Source base + start_offset; dst
    // base.
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let src_base = match &src_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    // Source pointer at the live region's start (start_offset applied).
    let src_byte_off = (layout.start_offset() * bpe) as u64;
    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    // Marshal shape + element strides to i64 vecs.
    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let strides_i64: Vec<i64> = strides_elems.iter().map(|&s| s as i64).collect();

    let status = unsafe {
        kiln_contiguous_copy_async(
            src_ptr,
            dst_ptr,
            shape_i64.as_ptr(),
            strides_i64.as_ptr(),
            rank as i32,
            bpe as i32,
            n_elements as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_contiguous: kiln_contiguous_copy_async returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_contiguous: wrap: {e}")))
}

/// In-place `slice_set` along dim 0: copy `src` (contiguous) into `dst`'s
/// existing device buffer starting at outer-axis `offset`.
///
/// #1082: the kt counterpart of `candle_core::Tensor::slice_set` for the
/// CUDA-graph output buffer + GDN resident-state row writes. Every kiln
/// call site uses dim 0, so this is a single contiguous device→device
/// memcpy: dim 0 is the outermost stride, so an offset along it is a flat
/// byte offset (`offset * inner * bpe`) and the `src` block is contiguous.
/// Writes through `dst`'s raw device pointer — the caller owns the
/// destination (graph buffer / resident state), so aliasing is intentional;
/// the [`crate::Tensor::slice_set`] wrapper validates shapes/dtype/device
/// and bumps the version counter afterward. `src`/`dst` must be contiguous
/// and same-dtype/device (checked by the wrapper).
#[cfg(feature = "cuda")]
pub fn cuda_slice_set_dim0(dst: &crate::Tensor, src: &crate::Tensor, offset: usize) -> Result<()> {
    use cudarc::driver::DevicePtr;

    if dst.dtype().is_packed() {
        return Err(crate::Error::Msg(
            "cuda_slice_set: packed dtype not supported".to_string(),
        ));
    }
    let bpe = dst.dtype().size_in_bytes();
    // inner = product of dims after the outer axis (the per-row block size).
    let inner: usize = dst.dims().iter().skip(1).product();
    let src_n = src.element_count();

    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_slice_set: dst must be CUDA storage".to_string()))?;
    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_slice_set: src must be CUDA storage".to_string()))?;

    let ctx = dst_storage.context();
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let src_base = match &src_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let src_byte_off = (src.layout().start_offset() * bpe) as u64;
    let dst_byte_off = ((offset * inner) * bpe) as u64;
    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let dst_ptr = (dst_base + dst_byte_off) as *mut core::ffi::c_void;

    // Flat contiguous copy of `src_n` elements (shape [src_n], stride [1])
    // into dst at the computed byte offset. Reuses the contiguous-copy
    // kernel that backs `cuda_contiguous`.
    let shape_i64 = [src_n as i64];
    let strides_i64 = [1i64];
    let status = unsafe {
        kiln_contiguous_copy_async(
            src_ptr,
            dst_ptr,
            shape_i64.as_ptr(),
            strides_i64.as_ptr(),
            1,
            bpe as i32,
            src_n as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_slice_set: kiln_contiguous_copy_async returned status {status}"
        )));
    }
    // `slice_set` is an in-place API whose source tensor may be dropped as soon
    // as this function returns. The copy kernel is async, so outside graph
    // capture drain the active stream before Rust can release/reuse `src`.
    if !crate::capture_arena_active() {
        crate::cuda_synchronize_tensor_stream_for(
            dst,
            &stream,
            crate::CudaSyncReason::AllocationLifetime,
        )
        .map_err(|e| crate::Error::Msg(format!("cuda_slice_set: {e}")))?;
    }
    Ok(())
}

/// In-place host→device write into an existing CUDA tensor's buffer,
/// preserving its device pointer.
///
/// #1082 CUDA-graph refresh: the kt counterpart of the candle
/// `memcpy_htod_async`-into-the-buffer's-device-pointer dance the CUDA
/// graph runner used to perform on its candle scalar buffers
/// (`update_cuda_scalar`). The captured graph bakes `dst`'s device
/// pointer; replay refreshes the contents through THIS function so the
/// pointer never changes and the recorded kernels read the new values.
///
/// Unlike [`host_to_cuda_copy`] (which allocates a fresh device buffer),
/// this writes through `dst`'s already-allocated storage. `dst` must be
/// CUDA-backed, contiguous, `start_offset == 0`, and own exactly
/// `host.len()` elements of its element type `E` (the element type must
/// match `dst`'s dtype byte width). The copy runs on the kt active
/// stream — inside a [`crate::with_active_cuda_stream`] scope it lands on
/// the capture/replay stream (so a refresh issued during capture is
/// recorded into the graph); outside one it's the context default stream.
/// A stream synchronize follows so the write completes before the
/// subsequent graph launch reads it (mirrors the old candle path).
#[cfg(feature = "cuda")]
pub fn cuda_write_host_in_place<E: crate::Element>(dst: &crate::Tensor, host: &[E]) -> Result<()> {
    if dst.dtype().is_packed() {
        return Err(crate::Error::Msg(
            "cuda_write_host_in_place: packed dtype not supported".to_string(),
        ));
    }
    if E::DTYPE.size_in_bytes() != dst.dtype().size_in_bytes() {
        return Err(crate::Error::Msg(format!(
            "cuda_write_host_in_place: element byte width {} != dst dtype {} byte width {}",
            E::DTYPE.size_in_bytes(),
            dst.dtype(),
            dst.dtype().size_in_bytes()
        )));
    }
    if !dst.is_contiguous() || dst.layout().start_offset() != 0 {
        return Err(crate::Error::Msg(
            "cuda_write_host_in_place: dst must be contiguous with start_offset == 0".to_string(),
        ));
    }
    let n = dst.element_count();
    if host.len() != n {
        return Err(crate::Error::Msg(format!(
            "cuda_write_host_in_place: host len {} != dst element count {n}",
            host.len()
        )));
    }

    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_write_host_in_place: dst must be CUDA storage".to_string())
        })?;

    let ctx = dst_storage.context();
    let stream = crate::active_cuda_stream(&ctx);
    let (dst_ptr, _byte_len) = dst_storage.device_ptr_raw();

    // SAFETY: `dst_ptr` is the start of `dst`'s contiguous device buffer
    // (validated start_offset==0); `host` has exactly `n` elements of the
    // same byte width as `dst`'s dtype, so the copy stays inside the
    // allocation. The copy is issued on the kt active stream — the same
    // stream the captured graph runs on — so there is no concurrent read
    // between this write and the graph launch. The synchronize below
    // ensures completion before any subsequent launch reads the buffer.
    unsafe {
        cudarc::driver::result::memcpy_htod_async(dst_ptr, host, stream.cu_stream()).map_err(
            |e| {
                crate::Error::Msg(format!(
                    "cuda_write_host_in_place: memcpy_htod_async: {e:?}"
                ))
            },
        )?;
    }
    crate::cuda_synchronize_tensor_stream_for(dst, &stream, crate::CudaSyncReason::InPlaceMutation)
        .map_err(|e| crate::Error::Msg(format!("cuda_write_host_in_place: {e}")))?;
    Ok(())
}

/// Synchronize the context default stream for `device_index`.
///
/// #1082 CUDA-graph fix: the graph-stable buffers are filled by
/// `Tensor::from_vec_on` (→ `host_to_cuda_copy` → `clone_htod`), whose
/// H2D `memcpy_htod_async` lands on the kt **default** stream. Graph
/// capture begins on a *separate* (non-default) capture stream, so the
/// pre-capture `capture_stream.synchronize()` does NOT cover those
/// fills. Call this before `begin_capture` so the buffers' initial
/// contents are guaranteed visible to the captured forward (matches the
/// candle path, where `Tensor::new` allocated on the capture stream so a
/// single capture-stream sync sufficed).
#[cfg(feature = "cuda")]
pub fn cuda_synchronize_default_stream(device_index: usize) -> Result<()> {
    crate::cuda_synchronize_default_stream_for(
        device_index,
        crate::CudaSyncReason::ExplicitStreamDrain,
    )
}

/// CUDA-side `index_select(src, axis=0, indices)` — gather along the
/// outer axis of a CUDA tensor.
///
/// Both `src` and `indices` must be CUDA-backed and contiguous.
/// `indices` must be U32 (the kernel reads it as `*const u32`).
///
/// Returns a freshly-allocated contiguous output of shape
/// `[indices.element_count(), src.shape()[1..]]` with the same dtype
/// as `src`.
///
/// Errors:
/// - src/indices must be CUDA storage
/// - src/indices must be contiguous
/// - src.rank() must be >= 1
/// - indices.dtype() == U32
/// - packed dtypes not supported
#[cfg(feature = "cuda")]
pub fn cuda_index_select_dim0(
    src: &crate::Tensor,
    indices: &crate::Tensor,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if src.dtype().is_packed() {
        return Err(crate::Error::Msg(
            "cuda_index_select_dim0: packed dtype not supported".to_string(),
        ));
    }
    if indices.dtype() != crate::DType::U32 {
        return Err(crate::Error::Msg(format!(
            "cuda_index_select_dim0: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !src.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_index_select_dim0: src must be contiguous (call .contiguous()? first)"
                .to_string(),
        ));
    }
    if !indices.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_index_select_dim0: indices must be contiguous".to_string(),
        ));
    }

    let src_shape = src.shape();
    if src_shape.is_empty() {
        return Err(crate::Error::Msg(
            "cuda_index_select_dim0: src must have rank >= 1".to_string(),
        ));
    }
    let src_n_rows = src_shape[0];
    let inner: usize = src_shape[1..].iter().product();
    let bpe = src.dtype().size_in_bytes();
    let row_bytes = (inner * bpe) as i64;

    let n_indices = indices.element_count();
    let mut out_shape = vec![n_indices];
    out_shape.extend_from_slice(&src_shape[1..]);
    let n_out_elements = n_indices * inner;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_index_select_dim0: src must be CUDA storage".to_string())
        })?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_index_select_dim0: indices must be CUDA storage".to_string())
        })?;

    let ctx = src_storage.context();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };
    let dst_storage = CudaStorage::zeros_ctx(&ctx, device_index, src.dtype(), n_out_elements)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let src_base = match &src_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let idx_base = match &idx_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let src_byte_off = (src.layout().start_offset() * bpe) as u64;
    let idx_byte_off = (indices.layout().start_offset() * crate::DType::U32.size_in_bytes()) as u64;

    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let idx_ptr = (idx_base + idx_byte_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_index_select_dim0_async(
            src_ptr,
            dst_ptr,
            idx_ptr,
            row_bytes,
            n_indices as i64,
            src_n_rows as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_index_select_dim0: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_index_select_dim0: wrap: {e}")))
}

/// CUDA-side `index_select` along an arbitrary axis.
///
/// Generalizes [`cuda_index_select_dim0`] to gather slices from any
/// axis of `src`. Output shape is
/// `src.shape[..axis] ++ indices.shape ++ src.shape[axis+1..]`.
///
/// Requirements:
/// - `src` is CUDA-backed and contiguous.
/// - `indices` is CUDA-backed, contiguous, U32 dtype.
/// - `src.rank() >= 1` and `axis < src.rank()`.
/// - Packed dtypes (e.g. quantized) are not supported.
///
/// `axis == 0` is dispatched through the same kernel; callers that
/// want the legacy dim0 fast path (single-axis blocks) should call
/// [`cuda_index_select_dim0`] directly.
#[cfg(feature = "cuda")]
pub fn cuda_index_select_axis_n(
    src: &crate::Tensor,
    axis: usize,
    indices: &crate::Tensor,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if src.dtype().is_packed() {
        return Err(crate::Error::Msg(
            "cuda_index_select_axis_n: packed dtype not supported".to_string(),
        ));
    }
    if indices.dtype() != crate::DType::U32 {
        return Err(crate::Error::Msg(format!(
            "cuda_index_select_axis_n: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !src.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_index_select_axis_n: src must be contiguous (call .contiguous()? first)"
                .to_string(),
        ));
    }
    if !indices.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_index_select_axis_n: indices must be contiguous".to_string(),
        ));
    }

    let src_shape = src.shape();
    if src_shape.is_empty() {
        return Err(crate::Error::Msg(
            "cuda_index_select_axis_n: src must have rank >= 1".to_string(),
        ));
    }
    if axis >= src_shape.len() {
        return Err(crate::Error::Msg(format!(
            "cuda_index_select_axis_n: axis {axis} out of bounds (src rank {})",
            src_shape.len()
        )));
    }

    let src_dim = src_shape[axis];
    let left_size: usize = src_shape[..axis].iter().product();
    let right_size: usize = src_shape[axis + 1..].iter().product();
    let bpe = src.dtype().size_in_bytes();
    let right_bytes = (right_size * bpe) as i64;
    let ids_dim = indices.element_count();

    let mut out_shape: Vec<usize> = src_shape[..axis].to_vec();
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&src_shape[axis + 1..]);
    let n_out_elements = left_size * ids_dim * right_size;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_index_select_axis_n: src must be CUDA storage".to_string())
        })?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_index_select_axis_n: indices must be CUDA storage".to_string())
        })?;

    let ctx = src_storage.context();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };
    let dst_storage = CudaStorage::zeros_ctx(&ctx, device_index, src.dtype(), n_out_elements)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let src_base = match &src_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let idx_base = match &idx_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let src_byte_off = (src.layout().start_offset() * bpe) as u64;
    let idx_byte_off = (indices.layout().start_offset() * crate::DType::U32.size_in_bytes()) as u64;

    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let idx_ptr = (idx_base + idx_byte_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_index_select_axis_n_async(
            src_ptr,
            dst_ptr,
            idx_ptr,
            right_bytes,
            ids_dim as i64,
            src_dim as i64,
            left_size as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_index_select_axis_n: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_index_select_axis_n: wrap: {e}")))
}

/// CUDA-side element-wise binary op: `out[i] = op(a[i], b[i])` for
/// matching-shape contiguous CUDA tensors.
///
/// `kind` encodes the op (0=Add, 1=Sub, 2=Mul, 3=Div). Dtype is
/// inferred from `a.dtype()`; must be F32 / BF16 / F16. Both inputs
/// must be contiguous and on the same CUDA device.
///
/// Returns a fresh contiguous output of the same shape and dtype.
#[cfg(feature = "cuda")]
pub fn cuda_elementwise_binary(
    a: &crate::Tensor,
    b: &crate::Tensor,
    kind: i32,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if a.shape() != b.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_elementwise_binary: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_elementwise_binary: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_elementwise_binary: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_elementwise_binary: contiguous inputs required".to_string(),
        ));
    }

    let n = a.element_count();
    let bpe = dtype.size_in_bytes();

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_elementwise_binary: a must be CUDA".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_elementwise_binary: b must be CUDA".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): elementwise binary writes the full output
    // (out[i] = op(a[i], b[i]) for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let a_base = match &a_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let b_base = match &b_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let a_off = (a.layout().start_offset() * bpe) as u64;
    let b_off = (b.layout().start_offset() * bpe) as u64;

    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_elementwise_binary_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_elementwise_binary: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(a.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_elementwise_binary: wrap: {e}")))
}

/// CUDA-side elementwise binary minimum / maximum (#1082).
///
/// `kind` encodes the op: 0 = minimum, 1 = maximum. Both tensors must
/// share shape + dtype (F32 / BF16 / F16) and be contiguous and on
/// CUDA. NaN propagates via `fminf` / `fmaxf` semantics — the non-NaN
/// operand wins when one side is NaN. Matches the CPU reference in
/// `ops::binary_minmax::minimum` / `maximum`.
#[cfg(feature = "cuda")]
pub fn cuda_binary_minmax(
    a: &crate::Tensor,
    b: &crate::Tensor,
    kind: i32,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    if a.shape() != b.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_binary_minmax: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_binary_minmax: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_binary_minmax: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_binary_minmax: contiguous inputs required".to_string(),
        ));
    }
    if kind != 0 && kind != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_binary_minmax: kind must be 0 (min) or 1 (max), got {kind}"
        )));
    }

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_binary_minmax: a must be CUDA".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_binary_minmax: b must be CUDA".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let n = a.element_count();
    // #1082 (perf, Pattern A): binary min/max writes the full output
    // (out[i] = min/max(a[i], b[i]) for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let a_base = match &a_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let b_base = match &b_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let per = dtype.size_in_bytes();
    let a_off = (a.layout().start_offset() * per) as u64;
    let b_off = (b.layout().start_offset() * per) as u64;
    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_binary_minmax_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_binary_minmax: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(a.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_binary_minmax: wrap: {e}")))
}

/// CUDA-side linear interpolation `out = a + weight * (b - a)` (#1082).
///
/// Element-wise. Both tensors must share shape + dtype (F32 / BF16 /
/// F16) and be contiguous and on CUDA. Mirrors the CPU reference in
/// `ops::lerp::lerp` (which evaluates the same `a + weight * (b - a)`
/// expression in F32).
#[cfg(feature = "cuda")]
pub fn cuda_lerp(a: &crate::Tensor, b: &crate::Tensor, weight: f32) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    if a.shape() != b.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_lerp: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_lerp: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_lerp: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_lerp: contiguous inputs required".to_string(),
        ));
    }

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_lerp: a must be CUDA".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_lerp: b must be CUDA".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let n = a.element_count();
    // #1082 (perf, Pattern A): lerp writes the full output
    // (out[i] = a[i] + w*(b[i]-a[i]) for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let a_base = match &a_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let b_base = match &b_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let per = dtype.size_in_bytes();
    let a_off = (a.layout().start_offset() * per) as u64;
    let b_off = (b.layout().start_offset() * per) as u64;
    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_lerp_async(
            a_ptr, b_ptr, out_ptr, n as i64, weight, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_lerp: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(a.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_lerp: wrap: {e}")))
}

/// CUDA-side unary activation: `out[i] = activation(x[i])`.
///
/// `kind` encodes the op (0=Silu, 1=Sigmoid, 2=Gelu, 3=Tanh, 4=Relu;
/// 5=Log, 6=Exp, 7=Sin, 8=Cos, 9=Tan, 10=Sinh, 11=Cosh, 12=Neg, 13=Abs, 14=Sqrt;
/// 15=Log2, 16=Log10, 17=Log1p, 18=Asin, 19=Acos, 20=Atan, 21=Atanh;
/// 22=Recip, 23=Sign, 24=Floor, 25=Ceil, 26=Round, 27=Trunc, 28=Rsqrt).
/// Dtype inferred from `x.dtype()`; must be F32/BF16/F16. Input must
/// be contiguous and on CUDA.
#[cfg(feature = "cuda")]
pub fn cuda_activation_unary(x: &crate::Tensor, kind: i32) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_activation_unary: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_activation_unary: contiguous input required".to_string(),
        ));
    }

    let n = x.element_count();
    let bpe = dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_activation_unary: x must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): unary activation writes the full output
    // (out[i] = f(x[i]) for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_activation_unary_async(x_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_activation_unary: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_activation_unary: wrap: {e}")))
}

/// CUDA-side dtype cast: `dst[i] = (TargetDType)(src[i])`.
///
/// Supports the same float↔float matrix as the CPU CastOp:
/// F32↔BF16↔F16 (6 directions). Integer round-trips (U32↔I64) stay
/// CPU-only since the few call sites have host data.
///
/// Source must be contiguous and CUDA-backed; output is contiguous
/// with the target dtype.
#[cfg(feature = "cuda")]
pub fn cuda_cast(src: &crate::Tensor, target: crate::DType) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    let from = src.dtype();
    if from == target {
        return src
            .contiguous()
            .map_err(|e| crate::Error::Msg(format!("cuda_cast: no-op contiguous: {e}")));
    }
    let cast_tag: i32 = match (from, target) {
        (crate::DType::F32, crate::DType::BF16) => 0,
        (crate::DType::F32, crate::DType::F16) => 1,
        (crate::DType::BF16, crate::DType::F32) => 2,
        (crate::DType::BF16, crate::DType::F16) => 3,
        (crate::DType::F16, crate::DType::F32) => 4,
        (crate::DType::F16, crate::DType::BF16) => 5,
        _ => {
            return Err(crate::Error::Msg(format!(
                "cuda_cast: unsupported pair {from} -> {target} \
                 (CUDA path supports F32↔BF16↔F16 only)"
            )));
        }
    };
    if !src.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_cast: contiguous input required".to_string(),
        ));
    }

    let n = src.element_count();
    let from_bpe = from.size_in_bytes();

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_cast: src must be CUDA".to_string()))?;

    let ctx = src_storage.context();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): the cast kernel writes every one of `n` output
    // elements (`out[i] = cast(src[i])`), so the output is fully overwritten
    // before any read — allocate uninitialized to skip the cudaMemsetAsync.
    let dst_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, target, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let src_base = match &src_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let src_off = (src.layout().start_offset() * from_bpe) as u64;
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    let status = unsafe { kiln_cast_async(src_ptr, dst_ptr, n as i64, cast_tag, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_cast: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_cast: wrap: {e}")))
}

// ----------------------------------------------------------------------
// Tests are GPU-only — gated by KILN_TENSOR_CUDA_TEST=1 so a host with
// cudarc + candle's cuda feature compiled in but no actual GPU doesn't
// spuriously fail.
// ----------------------------------------------------------------------

/// CUDA-side `scatter_add(updates, axis=0, indices, target_dim)` — inverse
/// of [`cuda_index_select_dim0`]. Mutates a pre-zeroed `out` tensor in
/// place by atomically adding each `updates` row to the position
/// indicated by `indices`.
///
/// This is the embedding-backward / gather-grad accumulation
/// primitive. Substrate-only (Phase 4 of #1082): the higher-level
/// `ScatterAddOp::cuda_fwd` allocates a fresh zero-filled output and
/// calls this function to perform the scatter.
///
/// # Arguments
///
/// - `out` — destination tensor, shape `[target_dim, ...inner]`, must
///   be pre-zeroed and CUDA-backed.
/// - `indices` — U32 CUDA tensor, shape `[n_indices]`.
/// - `updates` — values to scatter, shape `[n_indices, ...inner]`,
///   same dtype as `out`.
///
/// # Determinism
///
/// Uses `atomicAdd` per output cell. When two index positions collide
/// on the same target row, the addition order is non-deterministic.
/// This is the documented "atomic-bwd" tolerance band on
/// [`crate::ops::scatter_add::ScatterAddOp`].
///
/// # Errors
///
/// - `out` / `updates` / `indices` must be CUDA storage.
/// - All inputs must be contiguous.
/// - `indices.dtype() == U32`.
/// - `out` / `updates` dtype must be F32 or BF16.
/// - `out.rank() >= 1`, `updates.rank() >= 1`.
/// - `out.shape()[1..] == updates.shape()[1..]`.
/// - `updates.shape()[0] == indices.element_count()`.
#[cfg(feature = "cuda")]
pub fn cuda_scatter_add_dim0(
    out: &crate::Tensor,
    indices: &crate::Tensor,
    updates: &crate::Tensor,
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    // ---- dtype + shape validation ----
    if out.dtype() != updates.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_scatter_add_dim0: out dtype {} != updates dtype {}",
            out.dtype(),
            updates.dtype()
        )));
    }
    let dtype_tag: i32 = match out.dtype() {
        DType::F32 => 0,
        DType::BF16 => 1,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_scatter_add_dim0: unsupported dtype {other} (F32/BF16 only)"
            )));
        }
    };
    if indices.dtype() != DType::U32 {
        return Err(crate::Error::Msg(format!(
            "cuda_scatter_add_dim0: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !out.is_contiguous() || !updates.is_contiguous() || !indices.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_scatter_add_dim0: out/updates/indices must be contiguous".to_string(),
        ));
    }
    let out_shape = out.shape();
    let upd_shape = updates.shape();
    if out_shape.is_empty() || upd_shape.is_empty() {
        return Err(crate::Error::Msg(
            "cuda_scatter_add_dim0: out and updates must have rank >= 1".to_string(),
        ));
    }
    if out_shape[1..] != upd_shape[1..] {
        return Err(crate::Error::Msg(format!(
            "cuda_scatter_add_dim0: inner shape mismatch out={:?} updates={:?}",
            out_shape, upd_shape
        )));
    }
    let n_indices = indices.element_count();
    if upd_shape[0] != n_indices {
        return Err(crate::Error::Msg(format!(
            "cuda_scatter_add_dim0: updates.shape[0]={} != indices.len()={}",
            upd_shape[0], n_indices
        )));
    }
    let target_dim = out_shape[0];
    let row_inner: usize = out_shape[1..].iter().product::<usize>().max(1);

    // ---- storage downcasts ----
    let out_storage = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_scatter_add_dim0: out must be CUDA storage".to_string())
        })?;
    let upd_storage = updates
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_scatter_add_dim0: updates must be CUDA storage".to_string())
        })?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_scatter_add_dim0: indices must be CUDA storage".to_string())
        })?;

    let ctx = out_storage.context();
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let upd_base = match &upd_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let idx_base = match &idx_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let bpe = out.dtype().size_in_bytes();
    let upd_byte_off = (updates.layout().start_offset() * bpe) as u64;
    let idx_byte_off = (indices.layout().start_offset() * DType::U32.size_in_bytes()) as u64;
    let out_byte_off = (out.layout().start_offset() * bpe) as u64;

    let upd_ptr = (upd_base + upd_byte_off) as *const core::ffi::c_void;
    let idx_ptr = (idx_base + idx_byte_off) as *const core::ffi::c_void;
    let out_ptr = (out_base + out_byte_off) as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_scatter_add_dim0_async(
            upd_ptr,
            out_ptr,
            idx_ptr,
            n_indices as i64,
            row_inner as i64,
            target_dim as i64,
            dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_scatter_add_dim0: FFI returned status {status}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cuda_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_CUDA_TEST").ok().as_deref() == Some("1")
    }

    /// Acquire a primary cudarc [`CudaContext`] for tests that need to
    /// allocate device storage directly.
    ///
    /// #1082 CP-1: candle-free. The context comes from
    /// [`primary_cuda_context`], not a candle `Device::new_cuda`
    /// round-trip, so this test module compiles with `candle-core`
    /// dropped from the crate's `cuda` feature.
    ///
    /// Returns `None` if `KILN_TENSOR_CUDA_TEST` is unset OR the host
    /// has no visible CUDA device.
    fn maybe_cuda_ctx() -> Option<Arc<CudaContext>> {
        if !cuda_test_enabled() {
            return None;
        }
        primary_cuda_context(0).ok()
    }

    /// #1082: `Tensor::slice_set` dim-0 in-place write — the CUDA-graph
    /// output-buffer + GDN resident-state row-write primitive. Verifies
    /// the targeted row is overwritten and the others are untouched.
    #[test]
    fn slice_set_dim0_writes_targeted_row() {
        let Some(_dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let dev = Device::Cuda(0);
        // dst = 3x2 zeros; src = 1x2 [7, 8]; write into row 1.
        let dst = crate::Tensor::from_vec_on(dev, vec![0f32; 6], vec![3, 2]).unwrap();
        let src = crate::Tensor::from_vec_on(dev, vec![7f32, 8f32], vec![1, 2]).unwrap();
        dst.slice_set(&src, 0usize, 1).unwrap();
        let got = dst.to_vec2::<f32>().unwrap();
        assert_eq!(
            got,
            vec![vec![0.0, 0.0], vec![7.0, 8.0], vec![0.0, 0.0]],
            "slice_set must overwrite only row 1"
        );

        // offset 0 (the graph-output-buffer pattern): full overwrite.
        let dst2 = crate::Tensor::from_vec_on(dev, vec![9f32; 4], vec![2, 2]).unwrap();
        let src2 = crate::Tensor::from_vec_on(dev, vec![1f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        dst2.slice_set(&src2, 0usize, 0).unwrap();
        assert_eq!(
            dst2.to_vec2::<f32>().unwrap(),
            vec![vec![1.0, 2.0], vec![3.0, 4.0]]
        );
    }

    /// #1082: dim>0 and shape-overflow are rejected (only dim 0 is wired).
    #[test]
    fn slice_set_rejects_unsupported_dim_and_overflow() {
        let Some(_dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let dev = Device::Cuda(0);
        let dst = crate::Tensor::from_vec_on(dev, vec![0f32; 6], vec![3, 2]).unwrap();
        let src = crate::Tensor::from_vec_on(dev, vec![1f32, 2.0], vec![1, 2]).unwrap();
        assert!(dst.slice_set(&src, 1usize, 0).is_err(), "dim 1 must error");
        assert!(
            dst.slice_set(&src, 0usize, 3).is_err(),
            "offset overflow must error"
        );
    }

    #[test]
    fn zeros_round_sizes() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let ctx = dev.clone();
        let storage = CudaStorage::zeros_ctx(&ctx, 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Cuda(0));
        assert_eq!(storage.dtype(), DType::BF16);
        assert_eq!(storage.byte_len(), 128);

        let storage = CudaStorage::zeros_ctx(&ctx, 0, DType::Int4Packed, 16).unwrap();
        assert_eq!(storage.byte_len(), 8); // 16 elements packed -> 8 bytes
    }

    #[test]
    fn from_slice_validates_alignment() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let ctx = dev.clone();
        let slice = dev.default_stream().alloc_zeros::<u8>(17).unwrap();
        let err = CudaStorage::from_slice_ctx(&ctx, 0, DType::F32, slice).unwrap_err();
        assert!(err.to_string().contains("not a multiple"));
    }

    #[test]
    fn cuda_zeros_returns_arc_storage() {
        let Some(_dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        // (#1082) `cuda_zeros(Arc<CudaDevice>, ...)` was deleted in
        // a1f1c5bb; the test now exercises the candle-free
        // `cuda_zeros_ctx` entry which is the surviving public API.
        let s: crate::Storage = cuda_zeros_ctx(0, DType::F32, 4).unwrap();
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.byte_len(), 16);
        assert_eq!(s.device(), Device::Cuda(0));
        // Downcast to ensure the concrete type is CudaStorage.
        let cuda = s.as_any().downcast_ref::<CudaStorage>().expect("downcast");
        assert_eq!(cuda.slice().len(), 16);
    }

    // --- cuda_is_finite (#1082 Phase 9 substrate) -------------------

    /// Build a CUDA F32 tensor from a host slice via host_to_cuda_copy.
    fn cuda_f32_from_slice(_ctx: Arc<CudaContext>, values: &[f32]) -> crate::Tensor {
        let cpu = crate::Tensor::from_slice(values, vec![values.len()]).unwrap();
        crate::host_to_cuda_copy(&cpu, 0).unwrap()
    }

    #[test]
    fn cuda_is_finite_all_finite_f32_returns_true() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let t = cuda_f32_from_slice(dev, &[1.0, 2.0, -3.5, 0.0, 1e30]);
        assert!(super::cuda_is_finite(&t).unwrap());
        // Also exercise the Tensor::all_finite() path that routes here.
        assert!(t.all_finite().unwrap());
    }

    #[test]
    fn cuda_is_finite_nan_f32_returns_false() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let t = cuda_f32_from_slice(dev, &[1.0, f32::NAN, 3.0]);
        assert!(!super::cuda_is_finite(&t).unwrap());
        assert!(!t.all_finite().unwrap());
    }

    #[test]
    fn cuda_is_finite_pos_inf_f32_returns_false() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let t = cuda_f32_from_slice(dev, &[1.0, f32::INFINITY, 3.0]);
        assert!(!super::cuda_is_finite(&t).unwrap());
    }

    #[test]
    fn cuda_is_finite_neg_inf_f32_returns_false() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let t = cuda_f32_from_slice(dev, &[1.0, f32::NEG_INFINITY, 3.0]);
        assert!(!super::cuda_is_finite(&t).unwrap());
    }

    #[test]
    fn cuda_is_finite_bf16_nan_returns_false() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let _ = dev;
        // Build BF16 by casting an F32 with a NaN.
        let cpu_f32 = crate::Tensor::from_slice(&[1.0f32, f32::NAN, 2.0], vec![3]).unwrap();
        let cuda_f32 = crate::host_to_cuda_copy(&cpu_f32, 0).unwrap();
        let bf16 = crate::cuda_cast(&cuda_f32, DType::BF16).unwrap();
        assert_eq!(bf16.dtype(), DType::BF16);
        assert!(!super::cuda_is_finite(&bf16).unwrap());

        // Sanity: an all-finite BF16 tensor returns true.
        let cpu_ok = crate::Tensor::from_slice(&[1.0f32, 2.0, -0.5], vec![3]).unwrap();
        let cuda_ok = crate::host_to_cuda_copy(&cpu_ok, 0).unwrap();
        let bf16_ok = crate::cuda_cast(&cuda_ok, DType::BF16).unwrap();
        assert!(super::cuda_is_finite(&bf16_ok).unwrap());
    }

    #[test]
    fn cuda_is_finite_f16_inf_returns_false() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let _ = dev;
        let cpu_f32 = crate::Tensor::from_slice(&[1.0f32, f32::INFINITY, 2.0], vec![3]).unwrap();
        let cuda_f32 = crate::host_to_cuda_copy(&cpu_f32, 0).unwrap();
        let f16 = crate::cuda_cast(&cuda_f32, DType::F16).unwrap();
        assert_eq!(f16.dtype(), DType::F16);
        assert!(!super::cuda_is_finite(&f16).unwrap());
    }

    #[test]
    fn cuda_is_finite_integer_dtype_is_vacuously_true() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        // Build a CUDA U32 storage with a couple values; cuda_is_finite
        // short-circuits before touching the kernel.
        let ctx = dev.clone();
        let storage = CudaStorage::zeros_ctx(&ctx, 0, DType::U32, 4).unwrap();
        let storage_arc: crate::Storage = Arc::new(storage);
        let t = crate::Tensor::from_parts(
            storage_arc,
            crate::Layout::contiguous(vec![4]),
            crate::TensorId::next(),
        )
        .unwrap();
        assert!(super::cuda_is_finite(&t).unwrap());
        assert!(t.all_finite().unwrap());
    }

    #[test]
    fn cuda_is_finite_non_contiguous_uses_contig_path() {
        let Some(dev) = maybe_cuda_ctx() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        // 2x2 with one NaN at logical [1, 0]; transpose to make it
        // non-contiguous. cuda_is_finite contiguifies internally.
        let _ = dev;
        let cpu_f32 = crate::Tensor::from_slice(&[1.0f32, 2.0, f32::NAN, 4.0], vec![2, 2]).unwrap();
        let cuda_f32 = crate::host_to_cuda_copy(&cpu_f32, 0).unwrap();
        let tt = cuda_f32.transpose(0, 1).unwrap();
        assert!(!super::cuda_is_finite(&tt).unwrap());
    }
}
#[cfg(feature = "cuda")]
type CudaLastAxisNormalizationKernel = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    i64,
    i64,
    i32,
    *mut core::ffi::c_void,
) -> i32;

#[cfg(feature = "cuda")]
fn cuda_last_axis_normalization(
    x: &crate::Tensor,
    label: &str,
    kernel: CudaLastAxisNormalizationKernel,
    output_dtype: crate::DType,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "{label}: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{label}: input must be contiguous"
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: input must have rank >= 1"
        )));
    }
    let shape = x.shape();
    let trailing_dim = shape[rank - 1];
    if trailing_dim == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: trailing axis must be non-empty"
        )));
    }
    let n_cols = i64::try_from(trailing_dim)
        .map_err(|_| crate::Error::Msg(format!("{label}: trailing axis exceeds i64")))?;
    let n_rows = i64::try_from(x.element_count() / trailing_dim)
        .map_err(|_| crate::Error::Msg(format!("{label}: row count exceeds i64")))?;
    if n_rows > i64::from(i32::MAX) {
        return Err(crate::Error::Msg(format!(
            "{label}: row count {n_rows} exceeds kernel grid limit {}",
            i32::MAX
        )));
    }

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{label}: input must be CUDA")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Normalization kernels write every output element, so skip the zero-fill.
    let out_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, output_dtype, x.element_count())?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe { kernel(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}

/// CUDA softmax over the trailing axis (Phase 4 substrate op).
///
/// Operates on a contiguous `[..., D]` tensor; produces a fresh
/// contiguous tensor of the same shape and dtype with each
/// `[..., :]` row normalized to a probability distribution.
///
/// Routes through `kiln_softmax_last_axis_async` in
/// `csrc/softmax.cu`. F32/BF16/F16 supported.
#[cfg(feature = "cuda")]
pub fn cuda_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    cuda_last_axis_normalization(
        x,
        "cuda_softmax_last_axis",
        kiln_softmax_last_axis_async,
        x.dtype(),
    )
}

/// Numerically stable CUDA log-softmax over the trailing axis.
///
/// The fused kernel forms `x - max(x) - log(sum(exp(x - max(x))))`
/// directly in F32 arithmetic and performs one output allocation. It never
/// materializes probabilities whose underflow could corrupt a representable
/// log-probability before `log` is applied.
#[cfg(feature = "cuda")]
pub fn cuda_log_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    cuda_last_axis_normalization(
        x,
        "cuda_log_softmax_last_axis",
        kiln_log_softmax_last_axis_async,
        x.dtype(),
    )
}

/// Numerically stable CUDA log-softmax with direct F32 output.
///
/// The fused kernel reads F32/BF16/F16 input in place, accumulates in F32,
/// and writes one freshly allocated F32 output. It does not allocate a casted
/// input or round the result through the input dtype.
#[cfg(feature = "cuda")]
pub fn cuda_log_softmax_last_axis_f32(x: &crate::Tensor) -> Result<crate::Tensor> {
    cuda_last_axis_normalization(
        x,
        "cuda_log_softmax_last_axis_f32",
        kiln_log_softmax_last_axis_f32_async,
        crate::DType::F32,
    )
}

/// CUDA-side D2H copy: copy a CUDA-backed kt-Tensor's bytes into a
/// freshly-allocated CPU-backed kt-Tensor.
///
/// Phase 1 substrate op — closes a longstanding test-infrastructure
/// gap. The previous workaround was `kt-bridge`'s
/// `kt_tensor_to_candle_cuda_copy(&t).to_dtype(...).to_vec1::<f32>()`
/// chain. `cuda_to_host_copy` returns a kt-Tensor directly.
///
/// The candle-side `CudaStream::memcpy_dtoh` synchronizes against
/// any pending writes on the device's default stream before
/// returning. The returned CPU tensor is guaranteed to see the
/// latest results of any kernel previously launched on that stream.
///
/// The output has the same logical shape + dtype as `src`. Layout
/// is row-major contiguous (`start_offset = 0`); any non-contiguous
/// input is implicitly contiguified into the destination via
/// [`cuda_contiguous`].
///
/// Errors:
/// - Source must be CUDA storage.
/// - Packed dtypes (Marlin / Int4 / Fp4) are not supported.
#[cfg(feature = "cuda")]
pub fn cuda_to_host_copy(src: &crate::Tensor) -> Result<crate::Tensor> {
    if src.dtype().is_packed() {
        return Err(crate::Error::Msg(format!(
            "cuda_to_host_copy: packed dtype {} not supported",
            src.dtype()
        )));
    }

    // Force a contiguous, start_offset=0 CUDA buffer first.
    // `cuda_contiguous` handles both Owned and Borrowed inputs and
    // produces an Owned output, which has a usable `slice()`.
    let contig = cuda_contiguous(src)?;

    let contig_storage = contig
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_to_host_copy: contiguous'd storage must be CudaStorage".to_string(),
            )
        })?;

    let dtype = src.dtype();
    let n_elements = src.element_count();
    let byte_len = dtype.packed_buffer_bytes(n_elements);

    // Issue the D2H memcpy through the cudarc default stream. The
    // stream's `memcpy_dtoh` synchronizes on completion (per cudarc
    // 0.19's CudaStream semantics). Pulling the stream from
    // contig_storage.context() instead of .candle_device().cuda_stream()
    // retires another .candle_device() read (#1082).
    let slice = contig_storage.slice();
    let mut host_bytes = vec![0u8; byte_len];
    // #1082 CUDA-graph fix: route through the thread-local active stream
    // (outside a capture scope this is exactly `ctx.default_stream()`).
    let ctx = contig_storage.context();
    let stream = crate::active_cuda_stream(&ctx);
    stream
        .memcpy_dtoh(slice, &mut host_bytes)
        .map_err(|e| crate::Error::Msg(format!("cuda_to_host_copy: memcpy_dtoh failed: {e:?}")))?;

    let cpu_storage = crate::CpuStorage::from_bytes(dtype, host_bytes)?;
    let storage_arc: crate::Storage = Arc::new(cpu_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Phase 9 substrate op (#1082) — "any non-finite?" tensor-wide
/// reduction on a CUDA-resident tensor. Returns `Ok(true)` if every
/// element is finite (no NaN, no `+Inf`, no `-Inf`), `Ok(false)`
/// otherwise.
///
/// Routes through `kiln_is_finite_storage_async` in
/// `csrc/is_finite_reduce.cu`. Supported dtypes: F32, BF16, F16,
/// F8E4M3, F8E5M2. Integer dtypes (U8/U32/I64) and packed dtypes
/// (Int4Packed/Fp4Packed) are handled by the caller (vacuously
/// finite — no NaN/Inf representation).
///
/// The kernel atomic-OR's a single u32 device buffer; we issue exactly
/// one 4-byte D2H to read the flag back. Cost vs. the pre-Phase-9
/// `cuda_to_host_copy(self).all_finite()` bridge:
///
/// | tensor bytes | D2H bytes (old) | D2H bytes (new) |
/// |--------------|-----------------|-----------------|
/// | N            | N               | 4               |
///
/// For a typical hidden-state gradient (`[1, 1024, 2560]` BF16 ≈ 5 MB)
/// scanned once per backward op when tape anomaly detection is enabled, the
/// bridge paid ~5 MB of D2H per node; the kernel pays 4 bytes.
///
/// Non-contiguous inputs are contiguified into the kernel input via
/// [`cuda_contiguous`] before launching (matching the convention of
/// other kt-CUDA reductions like [`cuda_sum_squared_last_axis`] /
/// [`cuda_to_host_copy`]).
#[cfg(feature = "cuda")]
pub fn cuda_is_finite(src: &crate::Tensor) -> Result<bool> {
    use crate::DType;
    use cudarc::driver::DevicePtr;

    let dtype = src.dtype();
    // Integer + packed dtypes have no NaN/Inf — vacuously finite.
    // Tensor::all_finite() also early-returns for these; we replicate
    // the contract here so direct callers of `cuda_is_finite` get the
    // same answer.
    if matches!(dtype, DType::U8 | DType::U32 | DType::I64) {
        return Ok(true);
    }
    if dtype.is_packed() {
        return Ok(true);
    }

    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        DType::F8E4M3 => 3,
        DType::F8E5M2 => 4,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_is_finite: unsupported dtype {other}"
            )));
        }
    };

    // Force a contiguous, `start_offset = 0` device buffer. The
    // kernel walks `[0..n_elements)` directly; non-contiguous strided
    // inputs would otherwise need a separate stride-walking kernel.
    let contig = cuda_contiguous(src)?;
    let contig_storage = contig
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_is_finite: contiguous'd storage must be CudaStorage".to_string(),
            )
        })?;
    let ctx = contig_storage.context();
    let device_index = match contig_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("cuda_is_finite: contig storage device must be Cuda"),
    };

    // 1-element U32 device buffer (4 bytes, zero-init). Kernel
    // atomic-ORs a `1` into it on first non-finite hit.
    let flag_storage = CudaStorage::zeros_ctx(&ctx, device_index, DType::U32, 1)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &contig_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let flag_base = match &flag_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    // `cuda_contiguous` produces start_offset == 0, so no byte_off math.
    let x_ptr = x_base as *const core::ffi::c_void;
    let flag_ptr = flag_base as *mut core::ffi::c_void;

    let n_elements = src.element_count() as i64;
    let status =
        unsafe { kiln_is_finite_storage_async(x_ptr, flag_ptr, n_elements, dtype_tag, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_is_finite: FFI returned status {status}"
        )));
    }

    // Read the 4-byte flag back. `memcpy_dtoh` synchronizes against
    // the launch on the same stream.
    let flag_slice = match &flag_storage.slice {
        SliceOwner::Owned(s) => s,
        SliceOwner::Borrowed { .. } => unreachable!(),
    };
    let mut flag_host = [0u8; 4];
    stream
        .memcpy_dtoh(flag_slice, &mut flag_host)
        .map_err(|e| crate::Error::Msg(format!("cuda_is_finite: flag D2H failed: {e:?}")))?;
    let flag = u32::from_le_bytes(flag_host);
    Ok(flag == 0)
}

/// Host → CUDA copy: copy a CPU-backed kt-Tensor's bytes onto a
/// CUDA device, returning a new CUDA-backed kt-Tensor.
///
/// Phase 1 substrate op — the sibling of [`cuda_to_host_copy`].
/// Together they close the host↔device round-trip surface.
///
/// `device_index` identifies the destination CUDA device. The
/// output layout is row-major contiguous (`start_offset = 0`);
/// non-contiguous inputs are silently contiguified into the
/// destination.
///
/// **#1082 candle-removal**: this entry is fully candle-free. The
/// destination `Arc<CudaContext>` is derived internally via
/// [`primary_cuda_context`] — no candle `CudaDevice` is materialized
/// anywhere along the path. The previous candle-typed
/// `host_to_cuda_copy(src, Arc<CudaDevice>, usize)` signature was
/// deleted as part of the wave 13 push to drop candle-core from
/// `kiln-tensor`'s mandatory `[dependencies]`.
///
/// Errors:
/// - Source must be CPU storage.
/// - Packed dtypes (Marlin / Int4 / Fp4) are not supported.
#[cfg(feature = "cuda")]
pub fn host_to_cuda_copy(src: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    if src.dtype().is_packed() {
        return Err(crate::Error::Msg(format!(
            "host_to_cuda_copy: packed dtype {} not supported",
            src.dtype()
        )));
    }
    let _cpu_storage = src
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("host_to_cuda_copy: source must be CPU storage".to_string())
        })?;

    let dtype = src.dtype();
    let n_elements = src.element_count();
    let byte_len = dtype.packed_buffer_bytes(n_elements);

    // Materialize contiguous host bytes. For non-contiguous CPU
    // inputs we'd need a stride-aware copy; cpu_storage.as_bytes()
    // is the raw byte buffer, valid only for contiguous start_offset=0
    // tensors. If the source isn't already that shape, do the
    // contiguify on host first via the existing CPU op.
    let contig_src = if src.is_contiguous() && src.layout().start_offset() == 0 {
        src.clone()
    } else {
        // CPU `Tensor::contiguous()` — does a strided byte copy on
        // the host into a fresh CpuStorage.
        src.contiguous()?
    };
    let contig_cpu = contig_src
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("host_to_cuda_copy: contig src must be CPU storage".to_string())
        })?;
    let bytes = contig_cpu.as_bytes();
    if bytes.len() != byte_len {
        return Err(crate::Error::Msg(format!(
            "host_to_cuda_copy: src byte_len {} != expected {}",
            bytes.len(),
            byte_len
        )));
    }

    // CUDA graph freeze-pointers: host-created tensors inside the captured
    // forward must not record `cudaMallocAsync` plus a pageable H2D copy whose
    // source is this short-lived CPU tensor. Record mode uploads and retains a
    // stable device buffer; replay/capture mode validates identical host bytes
    // and reuses it without emitting allocation, memcpy, or free graph nodes.
    if let Some(storage) = crate::capture_arena_from_host(dtype, n_elements, bytes) {
        let storage_arc: crate::Storage = Arc::new(storage?);
        return crate::Tensor::from_parts(
            storage_arc,
            crate::Layout::contiguous(src.shape().to_vec()),
            crate::TensorId::next(),
        );
    }

    // Allocate the device buffer + issue H2D memcpy via the primary
    // cudarc context's default stream. No candle device is involved
    // anywhere along this path: `primary_cuda_context(device_index)`
    // just calls `cudarc::driver::CudaContext::new(device_index)`,
    // which is the same primary-context retain candle's
    // `Device::new_cuda` performs under the hood.
    let ctx = primary_cuda_context(device_index)?;
    let stream = crate::active_cuda_stream(&ctx);
    let device_slice = stream
        .clone_htod(bytes)
        .map_err(|e| crate::Error::Msg(format!("host_to_cuda_copy: clone_htod failed: {e:?}")))?;
    let cuda_storage = CudaStorage::from_slice_ctx(&ctx, device_index, dtype, device_slice)?;

    let storage_arc: crate::Storage = Arc::new(cuda_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Host → CUDA copy — back-compat alias for [`host_to_cuda_copy`].
///
/// Historically this was the candle-free parallel to a candle-typed
/// `host_to_cuda_copy(src, Arc<CudaDevice>, usize)`. With wave 13 of
/// the #1082 sweep, the candle-typed entry was deleted and the
/// canonical signature is now itself candle-free — so `host_to_cuda_copy_ctx`
/// just forwards to `host_to_cuda_copy`. Existing callers continue
/// to work unchanged; new code should call `host_to_cuda_copy`
/// directly.
#[cfg(feature = "cuda")]
pub fn host_to_cuda_copy_ctx(src: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    host_to_cuda_copy(src, device_index)
}

/// CUDA per-row sum-of-squares reduction over the trailing axis
/// (Phase 4 substrate op).
///
/// For a contiguous `[..., D]` input, produces a contiguous F32
/// output of shape `[...]` (one rank less). The reduction runs in
/// F32 regardless of the input dtype; the output is always F32 so
/// downstream composition can reuse the result without an extra cast.
///
/// Routes through `kiln_sum_squared_last_axis_async` in
/// `csrc/reduce_last_axis.cu`. F32/BF16/F16 supported.
#[cfg(feature = "cuda")]
pub fn cuda_sum_squared_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_sum_squared_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_sum_squared_last_axis: input must be contiguous".to_string(),
        ));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_sum_squared_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_sum_squared_last_axis: input must be CUDA".to_string())
        })?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output is always F32.
    // #1082 (perf, Pattern A): the sum-of-squares kernel writes every
    // output row (out[row] = Σ_c x[row,c]^2 for all n_rows rows; lane 0
    // of each per-row block stores unconditionally) and the output is
    // exactly n_rows elements; uninit skips the memset.
    let out_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::F32, n_rows as usize)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_squared_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_sum_squared_last_axis: FFI returned status {status}"
        )));
    }

    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// CUDA L2 normalization over the trailing axis (Phase 4 substrate
/// op).
///
/// For a contiguous `[..., D]` input, produces a contiguous output of
/// the same shape and dtype with each row scaled by
/// `1 / sqrt(sum_d x[..., d]^2 + eps)`.
///
/// Internally:
///   1. Per-row sum-of-squares via `kiln_sum_squared_last_axis_async`
///      (F32 accumulator).
///   2. Per-element scale + cast back via `kiln_l2norm_apply_async`
///      (rsqrt + multiply, no second pass over `sum_sq`).
///
/// F32/BF16/F16 supported.
#[cfg(feature = "cuda")]
pub fn cuda_l2norm_last_axis(x: &crate::Tensor, eps: f32) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_l2norm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_l2norm_last_axis: input must be contiguous".to_string(),
        ));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_l2norm_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    // Phase 1: produce per-row sum-of-squares (F32, shape [..rows]).
    let sum_sq = cuda_sum_squared_last_axis(x)?;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_l2norm_last_axis: input must be CUDA".to_string())
        })?;
    let sum_sq_storage = sum_sq
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_l2norm_last_axis: sum_sq must be CUDA (internal invariant)".to_string(),
            )
        })?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): the l2norm-apply kernel writes every
    // element of the output (out[row, c] = x[row,c] * inv_norm for all
    // rows × cols) and the output is exactly x.element_count(); uninit
    // skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let sum_sq_base = match &sum_sq_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let sum_sq_off = (sum_sq.layout().start_offset() * crate::DType::F32.size_in_bytes()) as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let sum_sq_ptr = (sum_sq_base + sum_sq_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_l2norm_apply_async(
            x_ptr, sum_sq_ptr, out_ptr, n_rows, n_cols, eps, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_l2norm_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}

/// CUDA RMSNorm over the trailing axis (Phase 4 substrate op).
///
/// Operates on a contiguous `[..., D]` tensor with a contiguous
/// rank-1 `weight: [D]`; produces a fresh contiguous tensor of the
/// same shape and dtype as `x`. Each row is normalized as:
///
/// ```text
/// mean_sq = (1/D) * sum_c x[..., c]^2
/// inv_rms = 1 / sqrt(mean_sq + eps)
/// out[..., c] = x[..., c] * inv_rms * weight[c]
/// ```
///
/// All accumulation happens in F32 regardless of input dtype; output
/// is cast back to `x`'s dtype. F32 / BF16 / F16 supported.
///
/// `weight.dtype()` must match `x.dtype()`.
///
/// Routes through `kiln_rmsnorm_last_axis_async` in `csrc/rmsnorm.cu`.
#[cfg(feature = "cuda")]
pub fn cuda_rmsnorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_rmsnorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_rmsnorm_last_axis: input must be contiguous".to_string(),
        ));
    }
    if !weight.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_rmsnorm_last_axis: weight must be contiguous".to_string(),
        ));
    }
    if weight.dtype() != dtype {
        return Err(crate::Error::Msg(format!(
            "cuda_rmsnorm_last_axis: weight dtype {} != x dtype {}",
            weight.dtype(),
            dtype
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_rmsnorm_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if weight.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_rmsnorm_last_axis: weight must be rank-1, got rank {}",
            weight.rank()
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    if weight.shape()[0] as i64 != n_cols {
        return Err(crate::Error::Msg(format!(
            "cuda_rmsnorm_last_axis: weight len {} != x trailing axis {}",
            weight.shape()[0],
            n_cols
        )));
    }
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rmsnorm_last_axis: x must be CUDA".to_string()))?;
    let weight_storage = weight
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_rmsnorm_last_axis: weight must be CUDA".to_string())
        })?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): rmsnorm writes every element of the
    // last-axis output (Pass 2 stores out[row, c] for all rows × cols);
    // uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let weight_base = match &weight_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let w_off = (weight.layout().start_offset() * dtype.size_in_bytes()) as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let weight_ptr = (weight_base + w_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_rmsnorm_last_axis_async(
            x_ptr, weight_ptr, out_ptr, n_rows, n_cols, eps, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_rmsnorm_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}

/// CUDA LayerNorm over the trailing axis (Phase 4 substrate op).
///
/// Operates on a contiguous `[..., D]` tensor with contiguous rank-1
/// `weight: [D]` and `bias: [D]`; produces a fresh contiguous tensor
/// of the same shape and dtype as `x`. Each row is normalized as:
///
/// ```text
/// mean = (1/D) * sum_c x[..., c]
/// var  = (1/D) * sum_c (x[..., c] - mean)^2
/// inv_std = 1 / sqrt(var + eps)
/// out[..., c] = (x[..., c] - mean) * inv_std * weight[c] + bias[c]
/// ```
///
/// All accumulation happens in F32 regardless of input dtype; output
/// is cast back to `x`'s dtype. F32 / BF16 / F16 supported.
///
/// `weight.dtype()` and `bias.dtype()` must match `x.dtype()`.
///
/// Routes through `kiln_layernorm_last_axis_async` in
/// `csrc/layernorm.cu`.
#[cfg(feature = "cuda")]
pub fn cuda_layernorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    bias: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_layernorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_layernorm_last_axis: all inputs must be contiguous".to_string(),
        ));
    }
    if weight.dtype() != dtype || bias.dtype() != dtype {
        return Err(crate::Error::Msg(format!(
            "cuda_layernorm_last_axis: dtype mismatch x={} weight={} bias={}",
            dtype,
            weight.dtype(),
            bias.dtype()
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_layernorm_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if weight.rank() != 1 || bias.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_layernorm_last_axis: weight/bias must be rank-1, got {}/{}",
            weight.rank(),
            bias.rank()
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    if weight.shape()[0] as i64 != n_cols || bias.shape()[0] as i64 != n_cols {
        return Err(crate::Error::Msg(format!(
            "cuda_layernorm_last_axis: weight len {} / bias len {} != x trailing axis {}",
            weight.shape()[0],
            bias.shape()[0],
            n_cols
        )));
    }
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_layernorm_last_axis: x must be CUDA".to_string()))?;
    let weight_storage = weight
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_layernorm_last_axis: weight must be CUDA".to_string())
        })?;
    let bias_storage = bias
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_layernorm_last_axis: bias must be CUDA".to_string())
        })?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): layernorm writes every element of the
    // last-axis output (Pass 2 stores out[row, c] for all rows × cols);
    // uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let weight_base = match &weight_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let bias_base = match &bias_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let w_off = (weight.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let b_off = (bias.layout().start_offset() * dtype.size_in_bytes()) as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let weight_ptr = (weight_base + w_off) as *const core::ffi::c_void;
    let bias_ptr = (bias_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_layernorm_last_axis_async(
            x_ptr, weight_ptr, bias_ptr, out_ptr, n_rows, n_cols, eps, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_layernorm_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}

/// CUDA masked-fill (Phase 4 substrate op).
///
/// `out[i] = (mask[i] != 0) ? fill_value : x[i]` over a contiguous
/// `x` with shape `S` and dtype F32/BF16/F16, and a contiguous `mask`
/// with shape `S` and dtype U8. `fill_value` is `f32` and cast to
/// `x`'s dtype on store.
///
/// Routes through `kiln_masked_fill_u8_async` in
/// `csrc/masked_fill.cu`. Mirrors the CPU `MaskedFillOp::cpu_fwd`
/// (see `kiln-tensor/src/ops/mask.rs`).
#[cfg(feature = "cuda")]
pub fn cuda_masked_fill(
    x: &crate::Tensor,
    mask: &crate::Tensor,
    fill_value: f32,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    if x.shape() != mask.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_masked_fill: shape mismatch x={:?} mask={:?}",
            x.shape(),
            mask.shape()
        )));
    }
    if mask.dtype() != crate::DType::U8 {
        return Err(crate::Error::Msg(format!(
            "cuda_masked_fill: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_masked_fill: unsupported x dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() || !mask.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_masked_fill: contiguous inputs required".to_string(),
        ));
    }

    let n = x.element_count();
    let x_bpe = dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_masked_fill: x must be CUDA".to_string()))?;
    let mask_storage = mask
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_masked_fill: mask must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let mask_base = match &mask_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    // mask dtype is U8 → bpe = 1.
    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let mask_off = mask.layout().start_offset() as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let mask_ptr = (mask_base + mask_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_masked_fill_u8_async(
            x_ptr, mask_ptr, out_ptr, n as i64, fill_value, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_masked_fill: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_masked_fill: wrap: {e}")))
}

/// CUDA FLCE backward chunk helper.
///
/// Mutates a fresh F32 logits chunk `[num_active, chunk_len]` in-place into
/// `(softmax - one_hot) * scale`, using the per-row global log-sum-exp state
/// saved by the FLCE forward. This removes the generic broadcast/exp/div/mul
/// and sparse one-hot scatter sequence from the long-context FLCE backward.
#[cfg(feature = "cuda")]
pub fn cuda_flce_grad_logits_chunk_inplace(
    logits: &crate::Tensor,
    labels: &crate::Tensor,
    global_max: &crate::Tensor,
    global_sumexp: &crate::Tensor,
    chunk_start: usize,
    scale: f32,
) -> Result<()> {
    if logits.dtype() != crate::DType::F32 {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: logits dtype must be F32, got {}",
            logits.dtype()
        )));
    }
    if labels.dtype() != crate::DType::U32 {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: labels dtype must be U32, got {}",
            labels.dtype()
        )));
    }
    if global_max.dtype() != crate::DType::F32 || global_sumexp.dtype() != crate::DType::F32 {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: global stats must be F32, got max={} sumexp={}",
            global_max.dtype(),
            global_sumexp.dtype()
        )));
    }
    if logits.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: logits must be rank-2 [active, chunk], got {:?}",
            logits.shape()
        )));
    }
    let num_active = logits.shape()[0];
    let chunk_len = logits.shape()[1];
    if labels.shape() != [num_active] {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: labels shape {:?} != [{num_active}]",
            labels.shape()
        )));
    }
    if global_max.shape() != [num_active] || global_sumexp.shape() != [num_active] {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: global stat shapes max={:?} sumexp={:?} != [{num_active}]",
            global_max.shape(),
            global_sumexp.shape()
        )));
    }
    if !logits.is_contiguous()
        || !labels.is_contiguous()
        || !global_max.is_contiguous()
        || !global_sumexp.is_contiguous()
    {
        return Err(crate::Error::Msg(
            "cuda_flce_grad_logits_chunk_inplace: inputs must be contiguous".to_string(),
        ));
    }

    let logits_storage = logits
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_flce_grad_logits_chunk_inplace: logits must be CUDA".to_string(),
            )
        })?;
    let labels_storage = labels
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_flce_grad_logits_chunk_inplace: labels must be CUDA".to_string(),
            )
        })?;
    let max_storage = global_max
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_flce_grad_logits_chunk_inplace: global_max must be CUDA".to_string(),
            )
        })?;
    let sumexp_storage = global_sumexp
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_flce_grad_logits_chunk_inplace: global_sumexp must be CUDA".to_string(),
            )
        })?;

    if matches!(&logits_storage.slice, SliceOwner::Borrowed { .. }) {
        return Err(crate::Error::Msg(
            "cuda_flce_grad_logits_chunk_inplace: borrowed logits storage cannot be mutated"
                .to_string(),
        ));
    }
    if logits_storage.device != labels_storage.device
        || logits_storage.device != max_storage.device
        || logits_storage.device != sumexp_storage.device
    {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: device mismatch logits={} labels={} max={} sumexp={}",
            logits_storage.device, labels_storage.device, max_storage.device, sumexp_storage.device
        )));
    }

    let ctx = logits_storage.context();
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let (logits_base, _) = logits_storage.device_ptr_raw();
    let (labels_base, _) = labels_storage.device_ptr_raw();
    let (max_base, _) = max_storage.device_ptr_raw();
    let (sumexp_base, _) = sumexp_storage.device_ptr_raw();

    let f32_bpe = crate::DType::F32.size_in_bytes();
    let logits_ptr =
        (logits_base + (logits.layout().start_offset() * f32_bpe) as u64) as *mut core::ffi::c_void;
    let labels_ptr = (labels_base
        + (labels.layout().start_offset() * crate::DType::U32.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let max_ptr = (max_base + (global_max.layout().start_offset() * f32_bpe) as u64)
        as *const core::ffi::c_void;
    let sumexp_ptr = (sumexp_base + (global_sumexp.layout().start_offset() * f32_bpe) as u64)
        as *const core::ffi::c_void;

    let status = unsafe {
        kiln_flce_grad_logits_chunk_f32_async(
            logits_ptr,
            labels_ptr,
            max_ptr,
            sumexp_ptr,
            num_active as i64,
            chunk_len as i64,
            chunk_start as i64,
            scale,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_flce_grad_logits_chunk_inplace: FFI returned status {status}"
        )));
    }
    Ok(())
}

/// CUDA GRPO backward chunk helper.
///
/// Mutates a fresh F32 logits chunk `[num_active, chunk_len]` in-place into
/// `coeff[row] * (one_hot - softmax)`, using per-row global softmax stats saved
/// by the chunked selected-log-prob forward. This removes the generic
/// broadcast/exp/div/mul and sparse label-scatter sequence from the long-context
/// GRPO-family backward before the existing cuBLAS hidden-gradient matmul.
#[cfg(feature = "cuda")]
pub fn cuda_grpo_grad_logits_chunk_inplace(
    logits: &crate::Tensor,
    labels: &crate::Tensor,
    global_max: &crate::Tensor,
    global_sumexp: &crate::Tensor,
    coeff: &crate::Tensor,
    chunk_start: usize,
) -> Result<()> {
    if logits.dtype() != crate::DType::F32 {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: logits dtype must be F32, got {}",
            logits.dtype()
        )));
    }
    if labels.dtype() != crate::DType::U32 {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: labels dtype must be U32, got {}",
            labels.dtype()
        )));
    }
    if global_max.dtype() != crate::DType::F32
        || global_sumexp.dtype() != crate::DType::F32
        || coeff.dtype() != crate::DType::F32
    {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: global stats and coeff must be F32, got max={} sumexp={} coeff={}",
            global_max.dtype(),
            global_sumexp.dtype(),
            coeff.dtype()
        )));
    }
    if logits.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: logits must be rank-2 [active, chunk], got {:?}",
            logits.shape()
        )));
    }
    let num_active = logits.shape()[0];
    let chunk_len = logits.shape()[1];
    if labels.shape() != [num_active] {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: labels shape {:?} != [{num_active}]",
            labels.shape()
        )));
    }
    if global_max.elem_count() != num_active
        || global_sumexp.elem_count() != num_active
        || coeff.elem_count() != num_active
    {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: per-row tensor element counts max={} sumexp={} coeff={} != {num_active}",
            global_max.elem_count(),
            global_sumexp.elem_count(),
            coeff.elem_count()
        )));
    }
    if !logits.is_contiguous()
        || !labels.is_contiguous()
        || !global_max.is_contiguous()
        || !global_sumexp.is_contiguous()
        || !coeff.is_contiguous()
    {
        return Err(crate::Error::Msg(
            "cuda_grpo_grad_logits_chunk_inplace: inputs must be contiguous".to_string(),
        ));
    }

    let logits_storage = logits
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_grpo_grad_logits_chunk_inplace: logits must be CUDA".to_string(),
            )
        })?;
    let labels_storage = labels
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_grpo_grad_logits_chunk_inplace: labels must be CUDA".to_string(),
            )
        })?;
    let max_storage = global_max
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_grpo_grad_logits_chunk_inplace: global_max must be CUDA".to_string(),
            )
        })?;
    let sumexp_storage = global_sumexp
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_grpo_grad_logits_chunk_inplace: global_sumexp must be CUDA".to_string(),
            )
        })?;
    let coeff_storage = coeff
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_grpo_grad_logits_chunk_inplace: coeff must be CUDA".to_string())
        })?;

    if matches!(&logits_storage.slice, SliceOwner::Borrowed { .. }) {
        return Err(crate::Error::Msg(
            "cuda_grpo_grad_logits_chunk_inplace: borrowed logits storage cannot be mutated"
                .to_string(),
        ));
    }
    if logits_storage.device != labels_storage.device
        || logits_storage.device != max_storage.device
        || logits_storage.device != sumexp_storage.device
        || logits_storage.device != coeff_storage.device
    {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: device mismatch logits={} labels={} max={} sumexp={} coeff={}",
            logits_storage.device,
            labels_storage.device,
            max_storage.device,
            sumexp_storage.device,
            coeff_storage.device
        )));
    }

    let ctx = logits_storage.context();
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let (logits_base, _) = logits_storage.device_ptr_raw();
    let (labels_base, _) = labels_storage.device_ptr_raw();
    let (max_base, _) = max_storage.device_ptr_raw();
    let (sumexp_base, _) = sumexp_storage.device_ptr_raw();
    let (coeff_base, _) = coeff_storage.device_ptr_raw();

    let f32_bpe = crate::DType::F32.size_in_bytes();
    let logits_ptr =
        (logits_base + (logits.layout().start_offset() * f32_bpe) as u64) as *mut core::ffi::c_void;
    let labels_ptr = (labels_base
        + (labels.layout().start_offset() * crate::DType::U32.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let max_ptr = (max_base + (global_max.layout().start_offset() * f32_bpe) as u64)
        as *const core::ffi::c_void;
    let sumexp_ptr = (sumexp_base + (global_sumexp.layout().start_offset() * f32_bpe) as u64)
        as *const core::ffi::c_void;
    let coeff_ptr =
        (coeff_base + (coeff.layout().start_offset() * f32_bpe) as u64) as *const core::ffi::c_void;

    let status = unsafe {
        kiln_grpo_grad_logits_chunk_f32_async(
            logits_ptr,
            labels_ptr,
            max_ptr,
            sumexp_ptr,
            coeff_ptr,
            num_active as i64,
            chunk_len as i64,
            chunk_start as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_grpo_grad_logits_chunk_inplace: FFI returned status {status}"
        )));
    }
    Ok(())
}

/// CUDA argmax over the trailing axis (Phase 4 substrate op).
///
/// Operates on a contiguous `[..., D]` tensor; produces a fresh
/// contiguous tensor of shape `[...]` (trailing axis dropped) with
/// `I64` element type containing per-row argmax indices.
///
/// Routes through `kiln_argmax_last_axis_async` in
/// `csrc/argmax_last_axis.cu`. F32/BF16/F16 inputs supported. Ties
/// break to the lowest index — same convention as
/// `candle_core::Tensor::argmax` and kt's CPU `argmax_last_dim`.
#[cfg(feature = "cuda")]
pub fn cuda_argmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let in_dtype = x.dtype();
    let dtype_tag: i32 = match in_dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_argmax_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_argmax_last_axis: input must be contiguous".to_string(),
        ));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_argmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_argmax_last_axis: input must be CUDA".to_string())
        })?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    // Output: shape = leading axes, dtype = I64.
    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let out_elem_count: usize = out_shape.iter().product::<usize>().max(1);
    let out_storage =
        CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::I64, out_elem_count)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * in_dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_argmax_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_argmax_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// Select top-k `(value, index)` pairs from a contiguous rank-1 CUDA
/// F32/BF16/F16 tensor. The full vocabulary stays resident; only `k` F32
/// values and I64 indices cross to the host. Ranking is descending by value
/// with lower-index tie breaking, matching the host fallback.
#[cfg(feature = "cuda")]
pub fn cuda_topk_last_axis(x: &crate::Tensor, k: usize) -> Result<(Vec<f32>, Vec<u32>)> {
    let (values, indices, _) = cuda_topk_last_axis_profiled(x, k)?;
    Ok((values, indices))
}

/// [`cuda_topk_last_axis`] plus the summed wall time of its two existing D2H
/// copies, excluding kernel launch and host decoding.
#[cfg(feature = "cuda")]
pub fn cuda_topk_last_axis_profiled(
    x: &crate::Tensor,
    k: usize,
) -> Result<(Vec<f32>, Vec<u32>, Duration)> {
    use cudarc::driver::DevicePtr;

    let in_dtype = x.dtype();
    let dtype_tag: i32 = match in_dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_topk_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_topk_last_axis: input must be contiguous".to_string(),
        ));
    }
    if x.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_topk_last_axis: input must be rank 1, got rank {}",
            x.rank()
        )));
    }
    let vocab = x.dims()[0];
    if vocab == 0 {
        return Ok((Vec::new(), Vec::new(), Duration::ZERO));
    }
    let k = k.min(vocab);
    if k == 0 {
        return Ok((Vec::new(), Vec::new(), Duration::ZERO));
    }

    let n_cols = vocab as i64;
    let n_rows = 1_i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_topk_last_axis: input must be CUDA".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    // Tiny on-device outputs: [k] F32 values + [k] I64 indices. The
    // kernel writes every slot, but the buffers are small enough that
    // alloc_uninit's correctness contract is trivially satisfied; use
    // zeros_ctx for defensiveness against any row-too-short edge.
    let vals_storage = CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::F32, k)?;
    let idx_storage = CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::I64, k)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let vals_base = match &vals_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let idx_base = match &idx_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * in_dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let vals_ptr = vals_base as *mut core::ffi::c_void;
    let idx_ptr = idx_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_topk_last_axis_async(
            x_ptr, vals_ptr, idx_ptr, n_rows, n_cols, k as i32, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_topk_last_axis: FFI returned status {status}"
        )));
    }

    // Small D2H: only k floats + k i64 indices cross the bus. Each
    // `memcpy_dtoh` synchronizes against the kernel launch on the same
    // stream.
    let vals_slice = match &vals_storage.slice {
        SliceOwner::Owned(s) => s,
        SliceOwner::Borrowed { .. } => unreachable!(),
    };
    let idx_slice = match &idx_storage.slice {
        SliceOwner::Owned(s) => s,
        SliceOwner::Borrowed { .. } => unreachable!(),
    };

    let mut vals_bytes = vec![0u8; k * 4];
    let values_readback_started = Instant::now();
    stream
        .memcpy_dtoh(vals_slice, &mut vals_bytes)
        .map_err(|e| crate::Error::Msg(format!("cuda_topk_last_axis: values D2H failed: {e:?}")))?;
    let mut readback_duration = values_readback_started.elapsed();
    let mut idx_bytes = vec![0u8; k * 8];
    let indices_readback_started = Instant::now();
    stream.memcpy_dtoh(idx_slice, &mut idx_bytes).map_err(|e| {
        crate::Error::Msg(format!("cuda_topk_last_axis: indices D2H failed: {e:?}"))
    })?;
    readback_duration = readback_duration.saturating_add(indices_readback_started.elapsed());

    let mut values = Vec::with_capacity(k);
    for chunk in vals_bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut indices = Vec::with_capacity(k);
    for chunk in idx_bytes.chunks_exact(8) {
        let i64v = i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        indices.push(i64v as u32);
    }

    Ok((values, indices, readback_duration))
}

/// Select compact prompt-logprob rows without copying full vocabulary rows.
///
/// `logits` must be contiguous rank-2 CUDA F32/BF16/F16. Every input logit and
/// every derived F32 log-probability is checked for finiteness on device. The
/// host receives only normalization scalars, observed values/ranks, validation
/// diagnostics, and `top_k` `(logit, token_id)` pairs per row.
#[cfg(feature = "cuda")]
pub fn cuda_prompt_logprobs(
    logits: &crate::Tensor,
    observed_token_ids: &[u32],
    top_k: usize,
) -> Result<Vec<crate::DevicePromptLogprobRow>> {
    const NAME: &str = "cuda_prompt_logprobs";
    let dtype_tag = match logits.dtype() {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        dtype => {
            return Err(crate::Error::Msg(format!(
                "{NAME}: unsupported dtype {dtype}"
            )));
        }
    };
    if !logits.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{NAME}: logits must be contiguous"
        )));
    }
    if logits.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "{NAME}: logits must be rank 2, got rank {}",
            logits.rank()
        )));
    }
    let n_rows = logits.dims()[0];
    let n_cols = logits.dims()[1];
    if n_cols == 0 {
        return Err(crate::Error::Msg(format!(
            "{NAME}: vocabulary width must be nonzero"
        )));
    }
    if observed_token_ids.len() != n_rows {
        return Err(crate::Error::Msg(format!(
            "{NAME}: observed token count {} did not equal row count {n_rows}",
            observed_token_ids.len()
        )));
    }
    if top_k > n_cols {
        return Err(crate::Error::Msg(format!(
            "{NAME}: top_k {top_k} exceeded vocabulary width {n_cols}"
        )));
    }
    let top_k_i32 = i32::try_from(top_k)
        .map_err(|_| crate::Error::Msg(format!("{NAME}: top_k {top_k} did not fit i32")))?;
    for (row, &token_id) in observed_token_ids.iter().enumerate() {
        if token_id as usize >= n_cols {
            return Err(crate::Error::Msg(format!(
                "{NAME}: row {row} observed token id {token_id} was outside vocabulary width {n_cols}"
            )));
        }
    }
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    let n_rows_i64 = i64::try_from(n_rows)
        .map_err(|_| crate::Error::Msg(format!("{NAME}: row count {n_rows} did not fit i64")))?;
    let n_cols_i64 = i64::try_from(n_cols).map_err(|_| {
        crate::Error::Msg(format!("{NAME}: vocabulary width {n_cols} did not fit i64"))
    })?;

    let storage = logits
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{NAME}: logits must use CUDA storage")))?;
    let ctx = storage.context();
    let device_index = match logits.device() {
        crate::Device::Cuda(index) => index,
        device => {
            return Err(crate::Error::Msg(format!(
                "{NAME}: logits must be CUDA, got {device}"
            )));
        }
    };
    let observed_values = observed_token_ids
        .iter()
        .map(|&token_id| i64::from(token_id))
        .collect::<Vec<_>>();
    let observed_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::I64, n_rows)?;
    let observed = crate::Tensor::from_parts(
        Arc::new(observed_storage),
        crate::Layout::contiguous(vec![n_rows]),
        crate::TensorId::next(),
    )?;
    crate::cuda_write_host_in_place(&observed, &observed_values)?;
    let observed_storage = observed
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{NAME}: observed IDs must use CUDA storage")))?;

    let top_count = n_rows
        .checked_mul(top_k)
        .ok_or_else(|| crate::Error::Msg(format!("{NAME}: top-k output length overflow")))?;
    let row_max_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::F32, n_rows)?;
    let log_sum_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::F32, n_rows)?;
    let observed_logit_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::F32, n_rows)?;
    let observed_rank_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::I64, n_rows)?;
    let top_logits_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::F32, top_count.max(1))?;
    let top_indices_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::I64, top_count.max(1))?;
    let invalid_kind_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::U32, n_rows)?;
    let invalid_column_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::I64, n_rows)?;
    let invalid_value_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::F32, n_rows)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let (x_base, _) = storage.device_ptr_raw();
    let (observed_base, _) = observed_storage.device_ptr_raw();
    let (row_max_base, _) = row_max_storage.device_ptr_raw();
    let (log_sum_base, _) = log_sum_storage.device_ptr_raw();
    let (observed_logit_base, _) = observed_logit_storage.device_ptr_raw();
    let (observed_rank_base, _) = observed_rank_storage.device_ptr_raw();
    let (top_logits_base, _) = top_logits_storage.device_ptr_raw();
    let (top_indices_base, _) = top_indices_storage.device_ptr_raw();
    let (invalid_kind_base, _) = invalid_kind_storage.device_ptr_raw();
    let (invalid_column_base, _) = invalid_column_storage.device_ptr_raw();
    let (invalid_value_base, _) = invalid_value_storage.device_ptr_raw();
    let x_offset = (logits.layout().start_offset() * logits.dtype().size_in_bytes()) as u64;

    let status = unsafe {
        kiln_prompt_logprobs_async(
            (x_base + x_offset) as *const core::ffi::c_void,
            observed_base as *const i64,
            row_max_base as *mut core::ffi::c_void,
            log_sum_base as *mut core::ffi::c_void,
            observed_logit_base as *mut core::ffi::c_void,
            observed_rank_base as *mut core::ffi::c_void,
            top_logits_base as *mut core::ffi::c_void,
            top_indices_base as *mut core::ffi::c_void,
            invalid_kind_base as *mut core::ffi::c_void,
            invalid_column_base as *mut core::ffi::c_void,
            invalid_value_base as *mut core::ffi::c_void,
            n_rows_i64,
            n_cols_i64,
            top_k_i32,
            dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{NAME}: FFI returned status {status}"
        )));
    }

    fn flat_tensor(storage: CudaStorage, len: usize) -> Result<crate::Tensor> {
        crate::Tensor::from_parts(
            Arc::new(storage),
            crate::Layout::contiguous(vec![len]),
            crate::TensorId::next(),
        )
    }
    let row_maxes = flat_tensor(row_max_storage, n_rows)?.to_vec1::<f32>()?;
    let log_sums = flat_tensor(log_sum_storage, n_rows)?.to_vec1::<f32>()?;
    let observed_logits = flat_tensor(observed_logit_storage, n_rows)?.to_vec1::<f32>()?;
    let observed_ranks = flat_tensor(observed_rank_storage, n_rows)?.to_vec1::<i64>()?;
    let top_logits = if top_count == 0 {
        Vec::new()
    } else {
        flat_tensor(top_logits_storage, top_count)?.to_vec1::<f32>()?
    };
    let top_indices = if top_count == 0 {
        Vec::new()
    } else {
        flat_tensor(top_indices_storage, top_count)?.to_vec1::<i64>()?
    };
    let invalid_kinds = flat_tensor(invalid_kind_storage, n_rows)?.to_vec1::<u32>()?;
    let invalid_columns = flat_tensor(invalid_column_storage, n_rows)?.to_vec1::<i64>()?;
    let invalid_values = flat_tensor(invalid_value_storage, n_rows)?.to_vec1::<f32>()?;
    let _input_keepalive = (&observed, logits);

    crate::prompt_logprobs::finish_device_prompt_logprob_rows(
        NAME,
        n_rows,
        n_cols,
        top_k,
        row_maxes,
        log_sums,
        observed_logits,
        observed_ranks,
        top_logits,
        top_indices,
        invalid_kinds,
        invalid_columns,
        invalid_values,
    )
}

/// CUDA cross-entropy loss (Phase 4 substrate op).
///
/// Mirrors the CPU reference in
/// `crates/kiln-tensor/src/ops/cross_entropy.rs`:
///
/// ```text
/// for each batch b:
///     m = max_v logits[b, v]
///     log_sum_exp = m + log(sum_v exp(logits[b, v] - m))
///     loss_b = log_sum_exp - logits[b, targets[b]]
/// loss = mean_b loss_b
/// ```
///
/// `logits` is `[batch, vocab]` (F32 / BF16 / F16), `targets` is
/// `[batch]` (I64 or U32). Output is a rank-0 scalar at the logits
/// dtype.
///
/// Implementation: dispatches `kiln_cross_entropy_loss_async` in
/// `csrc/cross_entropy.cu`. The kernel does the row-wise log-sum-exp
/// + target-logit subtraction into a per-row F32 scratch buffer,
/// then a single-block finalize sums and divides by batch.
///
/// Target-index validation runs on-device: an out-of-range target
/// (negative or `>= vocab`) sets a device-side error flag which we
/// read back after sync and surface as `Result::Err`. An all-`-inf`
/// row similarly errors out (loss undefined).
#[cfg(feature = "cuda")]
pub fn cuda_cross_entropy_loss(
    logits: &crate::Tensor,
    targets: &crate::Tensor,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = logits.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_cross_entropy_loss: logits dtype must be F32/BF16/F16, got {other}"
            )));
        }
    };
    let targets_tag: i32 = match targets.dtype() {
        crate::DType::I64 => 0,
        crate::DType::U32 => 1,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_cross_entropy_loss: targets dtype must be I64/U32, got {other}"
            )));
        }
    };
    if logits.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_cross_entropy_loss: logits must be rank-2 [batch, vocab], got shape {:?}",
            logits.shape()
        )));
    }
    if targets.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_cross_entropy_loss: targets must be rank-1 [batch], got shape {:?}",
            targets.shape()
        )));
    }
    if !logits.is_contiguous() || !targets.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_cross_entropy_loss: both inputs must be contiguous".to_string(),
        ));
    }
    let shape = logits.shape();
    let batch = shape[0];
    let vocab = shape[1];
    if batch == 0 {
        return Err(crate::Error::Msg(
            "cuda_cross_entropy_loss: batch dim is 0 — mean is undefined".to_string(),
        ));
    }
    if targets.shape()[0] != batch {
        return Err(crate::Error::Msg(format!(
            "cuda_cross_entropy_loss: batch mismatch — logits has batch={batch}, targets has batch={}",
            targets.shape()[0]
        )));
    }

    let logits_storage = logits
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_cross_entropy_loss: logits must be CUDA storage".to_string())
        })?;
    let targets_storage = targets
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_cross_entropy_loss: targets must be CUDA storage".to_string())
        })?;

    let ctx = logits_storage.context();
    let device_index = match logits_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    // Allocate per-row F32 scratch and a 4-byte error flag. Both
    // start zeroed.
    let row_loss_storage = CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::F32, batch)?;
    // For row_err we use a 1-element U32 buffer (4 bytes, zero-init).
    let row_err_storage = CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::U32, 1)?;

    // Scalar output buffer (1 element at the input dtype). Reusing
    // `CudaStorage::zeros` to get a zero-initialized buffer; the
    // kernel overwrites it.
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, 1)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let logits_base = match &logits_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let targets_base = match &targets_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let row_loss_base = match &row_loss_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let row_err_base = match &row_err_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let logits_off = (logits.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let targets_off = (targets.layout().start_offset() * targets.dtype().size_in_bytes()) as u64;
    let logits_ptr = (logits_base + logits_off) as *const core::ffi::c_void;
    let targets_ptr = (targets_base + targets_off) as *const core::ffi::c_void;
    let row_loss_ptr = row_loss_base as *mut core::ffi::c_void;
    let row_err_ptr = row_err_base as *mut core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_cross_entropy_loss_async(
            logits_ptr,
            targets_ptr,
            row_loss_ptr,
            row_err_ptr,
            out_ptr,
            batch as i64,
            vocab as i64,
            dtype_tag,
            targets_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_cross_entropy_loss: FFI returned status {status}"
        )));
    }

    // Read back the row-error flag to surface validation failures.
    // This forces a sync but is the only way to surface a Result-level
    // error from on-device checks. We copy a single u32.
    let row_err_slice = match &row_err_storage.slice {
        SliceOwner::Owned(s) => s,
        SliceOwner::Borrowed { .. } => unreachable!(),
    };
    let mut err_host = [0u8; 4];
    stream
        .memcpy_dtoh(row_err_slice, &mut err_host)
        .map_err(|e| {
            crate::Error::Msg(format!(
                "cuda_cross_entropy_loss: row_err D2H failed: {e:?}"
            ))
        })?;
    let err_code = u32::from_le_bytes(err_host);
    match err_code {
        0 => {}
        1 => {
            return Err(crate::Error::Msg(format!(
                "cuda_cross_entropy_loss: target out of range (vocab={vocab})"
            )));
        }
        2 => {
            return Err(crate::Error::Msg(
                "cuda_cross_entropy_loss: row has no finite logits (all -inf?); loss is undefined"
                    .to_string(),
            ));
        }
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_cross_entropy_loss: unknown row_err code {other}"
            )));
        }
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(Vec::<usize>::new()),
        crate::TensorId::next(),
    )
}

/// CUDA per-row sum/mean over the trailing axis (Phase 4 substrate op).
///
/// For a contiguous `[..., D]` input, produces a contiguous output
/// of shape `[...]` (axis removed) at the same dtype. F32 accumulation
/// internally, cast back to input dtype on store. F32/BF16/F16
/// supported.
///
/// Routes through `kiln_sum_last_axis_async` in
/// `csrc/reduce_last_axis.cu`. `divisor` is applied in F32 before
/// the cast (so the mean path passes `1.0 / n_cols` and gets bit-
/// identical results to a separate divide kernel).
fn cuda_reduce_last_axis_impl(
    x: &crate::Tensor,
    divisor: f32,
    label: &str,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "{label}: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{label}: input must be contiguous"
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: input must have rank >= 1"
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{label}: input must be CUDA")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output: same dtype as input, shape = leading axes.
    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let out_elem_count: usize = out_shape.iter().product::<usize>().max(1);
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, out_elem_count)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_last_axis_async(
            x_ptr, out_ptr, n_rows, n_cols, divisor, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// `out = sum(x, axis=-1)` on CUDA — produces a tensor of shape
/// `x.shape[..-1]` at the same dtype as `x`. F32 accumulation.
#[cfg(feature = "cuda")]
pub fn cuda_sum_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    cuda_reduce_last_axis_impl(x, 1.0, "cuda_sum_last_axis")
}

/// `out = mean(x, axis=-1)` on CUDA — produces a tensor of shape
/// `x.shape[..-1]` at the same dtype as `x`. F32 accumulation;
/// divide-by-N applied in F32 before the cast back.
#[cfg(feature = "cuda")]
pub fn cuda_mean_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    if x.rank() == 0 {
        return Err(crate::Error::Msg(
            "cuda_mean_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    let n_cols = x.shape()[x.rank() - 1];
    if n_cols == 0 {
        return Err(crate::Error::Msg(
            "cuda_mean_last_axis: trailing dim is 0; mean is undefined".to_string(),
        ));
    }
    let inv = 1.0_f32 / (n_cols as f32);
    cuda_reduce_last_axis_impl(x, inv, "cuda_mean_last_axis")
}

/// Shared implementation behind [`cuda_sum_axis`] / [`cuda_mean_axis`].
///
/// Reduces `x` over a single (non-last) axis. Routes through
/// `kiln_sum_arbitrary_axis_async` in `csrc/reduce_arbitrary_axis.cu`,
/// which dispatches one block per (outer, inner) output element.
///
/// `divisor` is applied in F32 before the cast back:
/// - sum  ⇒ `divisor = 1.0`
/// - mean ⇒ `divisor = 1.0 / axis_dim`
///
/// Issue #1082.
#[cfg(feature = "cuda")]
fn cuda_reduce_arbitrary_axis_impl(
    x: &crate::Tensor,
    axis: usize,
    divisor: f32,
    label: &str,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "{label}: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{label}: input must be contiguous"
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: input must have rank >= 1"
        )));
    }
    if axis >= rank {
        return Err(crate::Error::Msg(format!(
            "{label}: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let shape = x.shape();
    let axis_dim = shape[axis] as i64;
    let outer: i64 = shape[..axis].iter().product::<usize>() as i64;
    let inner: i64 = shape[axis + 1..].iter().product::<usize>() as i64;
    let outer = outer.max(1);
    let inner = inner.max(1);

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{label}: input must be CUDA")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output: same dtype, shape = input shape with `axis` removed.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, out_elem_count)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_arbitrary_axis_async(
            x_ptr, out_ptr, outer, axis_dim, inner, divisor, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// `out = sum(x, axis=A)` on CUDA — produces a tensor of shape
/// `x.shape` with axis `A` removed, at the same dtype as `x`.
/// Issue #1082: generalises the last-axis path to any axis.
#[cfg(feature = "cuda")]
pub fn cuda_sum_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    let rank = x.rank();
    if rank > 0 && axis == rank - 1 {
        return cuda_sum_last_axis(x);
    }
    cuda_reduce_arbitrary_axis_impl(x, axis, 1.0, "cuda_sum_axis")
}

/// `out = mean(x, axis=A)` on CUDA — produces a tensor of shape
/// `x.shape` with axis `A` removed, at the same dtype as `x`.
/// Issue #1082: generalises the last-axis path to any axis.
#[cfg(feature = "cuda")]
pub fn cuda_mean_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    let rank = x.rank();
    if rank > 0 && axis == rank - 1 {
        return cuda_mean_last_axis(x);
    }
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_mean_axis: input must have rank >= 1".to_string(),
        ));
    }
    if axis >= rank {
        return Err(crate::Error::Msg(format!(
            "cuda_mean_axis: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let axis_dim = x.shape()[axis];
    if axis_dim == 0 {
        return Err(crate::Error::Msg(
            "cuda_mean_axis: axis dim is 0; mean is undefined".to_string(),
        ));
    }
    let inv = 1.0_f32 / (axis_dim as f32);
    cuda_reduce_arbitrary_axis_impl(x, axis, inv, "cuda_mean_axis")
}

/// Shared implementation behind [`cuda_min_axis`] / [`cuda_max_axis`].
///
/// Reduces `x` over a single axis by min or max. Routes through
/// `kiln_minmax_arbitrary_axis_async` in `csrc/reduce_arbitrary_axis.cu`,
/// which uses a fixed warp-shuffle + cross-warp tree (constructive
/// determinism — matches the CPU reference `ops::max_axis` /
/// `ops::min_axis` which walks the axis in a fixed iteration order).
///
/// F32 accumulation throughout; cast back to T on the final store.
/// `kind == 0` is MIN, `kind == 1` is MAX.
///
/// Issue #1082.
#[cfg(feature = "cuda")]
fn cuda_minmax_arbitrary_axis_impl(
    x: &crate::Tensor,
    axis: usize,
    kind: i32,
    label: &str,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "{label}: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{label}: input must be contiguous"
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: input must have rank >= 1"
        )));
    }
    if axis >= rank {
        return Err(crate::Error::Msg(format!(
            "{label}: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let shape = x.shape();
    let axis_dim = shape[axis] as i64;
    if axis_dim == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: axis dim is 0; {label} is undefined"
        )));
    }
    let outer: i64 = shape[..axis].iter().product::<usize>() as i64;
    let inner: i64 = shape[axis + 1..].iter().product::<usize>() as i64;
    let outer = outer.max(1);
    let inner = inner.max(1);

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{label}: input must be CUDA")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, out_elem_count)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_minmax_arbitrary_axis_async(
            x_ptr, out_ptr, outer, axis_dim, inner, kind, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// `out = min(x, axis=A)` on CUDA — produces a tensor of shape
/// `x.shape` with axis `A` removed, at the same dtype as `x`.
/// Issue #1082.
#[cfg(feature = "cuda")]
pub fn cuda_min_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    cuda_minmax_arbitrary_axis_impl(x, axis, 0, "cuda_min_axis")
}

/// `out = max(x, axis=A)` on CUDA — produces a tensor of shape
/// `x.shape` with axis `A` removed, at the same dtype as `x`.
/// Issue #1082.
#[cfg(feature = "cuda")]
pub fn cuda_max_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    cuda_minmax_arbitrary_axis_impl(x, axis, 1, "cuda_max_axis")
}

/// `out = all(mask, axis=A)` on CUDA — U8 mask in, U8 result of shape
/// `mask.shape` with axis `A` removed.
/// Issue #1082.
#[cfg(feature = "cuda")]
pub fn cuda_bool_reduce_axis(
    mask: &crate::Tensor,
    axis: usize,
    kind: u8, // 0 = ALL, 1 = ANY
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let label = if kind == 0 {
        "cuda_all_axis"
    } else {
        "cuda_any_axis"
    };
    if mask.dtype() != crate::DType::U8 {
        return Err(crate::Error::Msg(format!(
            "{label}: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    if !mask.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{label}: mask must be contiguous"
        )));
    }
    let rank = mask.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: mask must have rank >= 1"
        )));
    }
    if axis >= rank {
        return Err(crate::Error::Msg(format!(
            "{label}: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let shape = mask.shape();
    let axis_dim = shape[axis] as i64;
    let outer: i64 = shape[..axis].iter().product::<usize>() as i64;
    let inner: i64 = shape[axis + 1..].iter().product::<usize>() as i64;
    let outer = outer.max(1);
    let inner = inner.max(1);

    let x_storage = mask
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{label}: input must be CUDA")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::U8, out_elem_count)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (mask.layout().start_offset() * crate::DType::U8.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_bool_reduce_arbitrary_axis_async(
            x_ptr,
            out_ptr,
            outer,
            axis_dim,
            inner,
            kind as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// CUDA-side `concat(inputs, axis)` — concatenate `inputs` along
/// `axis` into a freshly-allocated contiguous output.
///
/// Mirrors the CPU reference in `crates/kiln-tensor/src/ops/concat.rs`
/// byte-for-byte: per-outer-slab, each input's axis-slab is copied
/// into the running output offset. The kernel
/// `kiln_concat_async` performs all the copies in a single launch.
///
/// Requirements:
/// - At least one input (and no more than 32).
/// - All inputs must be CUDA-backed and contiguous.
/// - All inputs must share dtype and the same shape except along `axis`.
/// - `axis < rank`, all input ranks equal.
/// - Packed dtypes are not supported.
#[cfg(feature = "cuda")]
pub fn cuda_concat(inputs: &[&crate::Tensor], axis: usize) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if inputs.is_empty() {
        return Err(crate::Error::Msg(
            "cuda_concat: at least one input required".to_string(),
        ));
    }
    if inputs.len() > 32 {
        return Err(crate::Error::Msg(format!(
            "cuda_concat: too many inputs ({}); MAX_INPUTS=32",
            inputs.len()
        )));
    }

    let rank = inputs[0].rank();
    if axis >= rank {
        return Err(crate::Error::Msg(format!(
            "cuda_concat: axis {axis} out of range for rank-{rank} inputs"
        )));
    }
    let dtype = inputs[0].dtype();
    if dtype.is_packed() {
        return Err(crate::Error::Msg(format!(
            "cuda_concat: packed dtype {dtype} is not supported"
        )));
    }
    let bpe = dtype.size_in_bytes();
    if bpe == 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_concat: zero-size dtype {dtype}"
        )));
    }

    for (i, t) in inputs.iter().enumerate() {
        if t.rank() != rank {
            return Err(crate::Error::Msg(format!(
                "cuda_concat: input {i} rank {} != input 0 rank {rank}",
                t.rank()
            )));
        }
        if t.dtype() != dtype {
            return Err(crate::Error::Msg(format!(
                "cuda_concat: input {i} dtype {} != input 0 dtype {dtype}",
                t.dtype()
            )));
        }
        if !t.is_contiguous() {
            return Err(crate::Error::Msg(format!(
                "cuda_concat: input {i} must be contiguous"
            )));
        }
        for (d, (&a, &b)) in t.shape().iter().zip(inputs[0].shape()).enumerate() {
            if d != axis && a != b {
                return Err(crate::Error::Msg(format!(
                    "cuda_concat: input {i} shape {:?} differs from input 0 shape {:?} along axis {d}",
                    t.shape(),
                    inputs[0].shape()
                )));
            }
        }
    }

    // Output shape: input 0's shape with axis dim replaced by sum.
    let mut out_shape = inputs[0].shape().to_vec();
    let axis_total: usize = inputs.iter().map(|t| t.shape()[axis]).sum();
    out_shape[axis] = axis_total;

    let outer: usize = out_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = out_shape[axis + 1..].iter().product::<usize>().max(1);
    let inner_bytes = (inner * bpe) as i64;

    // Pull candle_device + device index from first input.
    let first_storage = inputs[0]
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_concat: input 0 must be CUDA storage".to_string())
        })?;
    let ctx = first_storage.context();
    let device_index = match first_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };

    // Allocate destination — same device, same dtype, total elements.
    let n_out_elements: usize = out_shape.iter().product();
    let dst_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n_out_elements)?;

    // Collect per-input source pointers (base + start_offset bytes).
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let mut src_ptrs: Vec<*const core::ffi::c_void> = Vec::with_capacity(inputs.len());
    let mut t_axis_lens: Vec<i64> = Vec::with_capacity(inputs.len());
    // Hold guards so any borrowed device-pointer lifetime stays alive
    // through the kernel launch.
    for t in inputs {
        let st = t
            .storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                crate::Error::Msg("cuda_concat: input must be CUDA storage".to_string())
            })?;
        let base = match &st.slice {
            SliceOwner::Owned(s) => {
                let (p, _g) = s.device_ptr(&stream);
                p
            }
            SliceOwner::Borrowed { ptr, .. } => *ptr,
        };
        let off = (t.layout().start_offset() * bpe) as u64;
        src_ptrs.push((base + off) as *const core::ffi::c_void);
        t_axis_lens.push(t.shape()[axis] as i64);
    }

    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_concat_async(
            dst_ptr,
            src_ptrs.as_ptr(),
            t_axis_lens.as_ptr(),
            inputs.len() as i32,
            outer as i64,
            inner_bytes,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_concat: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

/// CUDA-side rotary position embedding (RoPE) — applies a per-position
/// 2-D rotation to the first `rotary_dim` of each row's `head_dim`,
/// passing through the remaining `head_dim - rotary_dim` entries.
///
/// Mirrors [`crate::ops::rope::RopeOp::cpu_fwd`]:
///   x:   [..., seq, head_dim], dtype F32/BF16/F16
///   cos: [seq, rotary_dim/2], dtype F32/BF16/F16
///   sin: [seq, rotary_dim/2], dtype F32/BF16/F16
///
/// `rotary_dim` is taken from `2 * cos.shape[-1]` (must equal
/// `2 * sin.shape[-1]`); the rotated region is the leading
/// `rotary_dim` of each row's `head_dim`.
///
/// All inputs must be contiguous, on the same CUDA device, with the
/// shape constraints validated in `ops/rope.rs::validate`. Returns a
/// fresh contiguous output tensor of the same shape + dtype as `x`.
#[cfg(feature = "cuda")]
pub fn cuda_rope(
    x: &crate::Tensor,
    cos: &crate::Tensor,
    sin: &crate::Tensor,
    rotary_dim: usize,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    // Shape / dtype validation (mirrors ops/rope.rs::validate).
    if x.rank() < 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: x must have rank >= 2, got shape {:?}",
            x.shape()
        )));
    }
    if cos.rank() != 2 || sin.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: cos / sin must be rank-2, got cos={:?} sin={:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.shape() != sin.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: cos / sin shape mismatch {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.dtype() != sin.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: cos / sin dtype mismatch {} vs {}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    let head_dim = x.shape()[x.rank() - 1];
    let seq = x.shape()[x.rank() - 2];
    if cos.shape()[0] != seq {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: cos.shape[0] ({}) != x seq ({seq})",
            cos.shape()[0]
        )));
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: rotary_dim must be positive and even, got {rotary_dim}"
        )));
    }
    if rotary_dim > head_dim {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: rotary_dim ({rotary_dim}) > head_dim ({head_dim})"
        )));
    }
    if cos.shape()[1] * 2 != rotary_dim {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: cos.shape[1] ({}) * 2 != rotary_dim ({rotary_dim})",
            cos.shape()[1]
        )));
    }
    if !x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_rope: contiguous inputs required".to_string(),
        ));
    }
    if !matches!(
        x.dtype(),
        crate::DType::F32 | crate::DType::BF16 | crate::DType::F16
    ) {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: x dtype must be F32/BF16/F16, got {}",
            x.dtype()
        )));
    }
    if !matches!(
        cos.dtype(),
        crate::DType::F32 | crate::DType::BF16 | crate::DType::F16
    ) {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: cos / sin dtype must be F32/BF16/F16, got {}",
            cos.dtype()
        )));
    }

    let x_dtype = x.dtype();
    let cs_dtype = cos.dtype();
    let x_dtype_tag: i32 = match x_dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        _ => unreachable!(),
    };
    let cs_dtype_tag: i32 = match cs_dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        _ => unreachable!(),
    };

    let pair_count = rotary_dim / 2;
    let leading: usize = x.shape()[..x.rank() - 2].iter().product::<usize>().max(1);
    let n = x.element_count();
    let x_bpe = x_dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rope: x must be CUDA".to_string()))?;
    let cos_storage = cos
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rope: cos must be CUDA".to_string()))?;
    let sin_storage = sin
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rope: sin must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, x_dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let cs_bpe = cs_dtype.size_in_bytes();

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let cos_base = match &cos_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let sin_base = match &sin_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let cos_off = (cos.layout().start_offset() * cs_bpe) as u64;
    let sin_off = (sin.layout().start_offset() * cs_bpe) as u64;

    let x_in_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let cos_ptr = (cos_base + cos_off) as *const core::ffi::c_void;
    let sin_ptr = (sin_base + sin_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    // The kernel writes the entire output: rotated values for the first
    // rotary_dim of each row's head_dim, and a pass-through copy for the
    // tail (head_dim - rotary_dim trailing entries). No prior memcpy needed.

    let status = unsafe {
        kiln_rope_async(
            x_in_ptr,
            out_ptr,
            cos_ptr,
            sin_ptr,
            leading as i64,
            seq as i64,
            head_dim as i64,
            pair_count as i64,
            x_dtype_tag,
            cs_dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_rope: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_rope: wrap: {e}")))
}

/// CUDA split-half/GPT-NeoX RoPE for `[batch, seq, heads, head_dim]` tensors.
#[cfg(feature = "cuda")]
pub fn cuda_rope_split_half(
    x: &crate::Tensor,
    cos: &crate::Tensor,
    sin: &crate::Tensor,
    rotary_dim: usize,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if x.rank() != 4 {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: x must be rank-4 [batch, seq, heads, head_dim], got {:?}",
            x.shape()
        )));
    }
    if cos.rank() != 2 || sin.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: cos / sin must be rank-2, got cos={:?} sin={:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.shape() != sin.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: cos / sin shape mismatch {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.dtype() != sin.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: cos / sin dtype mismatch {} vs {}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    let shape = x.shape();
    let (batch, seq, heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    if cos.shape()[0] != seq {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: cos.shape[0] ({}) != x seq ({seq})",
            cos.shape()[0]
        )));
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: rotary_dim must be positive and even, got {rotary_dim}"
        )));
    }
    if rotary_dim > head_dim {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: rotary_dim ({rotary_dim}) > head_dim ({head_dim})"
        )));
    }
    if cos.shape()[1] * 2 != rotary_dim {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: cos.shape[1] ({}) * 2 != rotary_dim ({rotary_dim})",
            cos.shape()[1]
        )));
    }
    if !x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_rope_split_half: contiguous inputs required".to_string(),
        ));
    }
    if !matches!(
        x.dtype(),
        crate::DType::F32 | crate::DType::BF16 | crate::DType::F16
    ) {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: x dtype must be F32/BF16/F16, got {}",
            x.dtype()
        )));
    }
    if !matches!(
        cos.dtype(),
        crate::DType::F32 | crate::DType::BF16 | crate::DType::F16
    ) {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: cos / sin dtype must be F32/BF16/F16, got {}",
            cos.dtype()
        )));
    }

    let x_dtype = x.dtype();
    let cs_dtype = cos.dtype();
    let x_dtype_tag: i32 = match x_dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        _ => unreachable!(),
    };
    let cs_dtype_tag: i32 = match cs_dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        _ => unreachable!(),
    };

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rope_split_half: x must be CUDA".to_string()))?;
    let cos_storage = cos
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rope_split_half: cos must be CUDA".to_string()))?;
    let sin_storage = sin
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_rope_split_half: sin must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::alloc_uninit_ctx(&ctx, device_index, x_dtype, x.element_count())?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_bpe = x_dtype.size_in_bytes();
    let cs_bpe = cs_dtype.size_in_bytes();
    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let cos_base = match &cos_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let sin_base = match &sin_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let cos_off = (cos.layout().start_offset() * cs_bpe) as u64;
    let sin_off = (sin.layout().start_offset() * cs_bpe) as u64;

    let status = unsafe {
        kiln_rope_split_half_4d_async(
            (x_base + x_off) as *const core::ffi::c_void,
            out_base as *mut core::ffi::c_void,
            (cos_base + cos_off) as *const core::ffi::c_void,
            (sin_base + sin_off) as *const core::ffi::c_void,
            batch as i64,
            seq as i64,
            heads as i64,
            head_dim as i64,
            rotary_dim as i64,
            x_dtype_tag,
            cs_dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_rope_split_half: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_rope_split_half: wrap: {e}")))
}

/// CUDA inverted dropout (training-time).
///
/// Produces `(y, mask)` where `y[i] = (rand_i >= p) ? x[i] / (1 - p) : 0`
/// and `mask[i]` is the U8 survival indicator. Per-element RNG: a
/// splitmix64-style hash of `(seed, i)` — fully deterministic given
/// `seed`. NOT bit-identical to the CPU op's sequential RNG (which
/// chains splitmix64 state), but matches the same distribution +
/// scaling contract.
///
/// Supports F32 / BF16 / F16 contiguous inputs.
///
/// Routes through `kiln_dropout_async` in `csrc/dropout.cu`. Mirrors
/// the CPU forward in `kiln-tensor/src/ops/dropout.rs`.
#[cfg(feature = "cuda")]
pub fn cuda_dropout(
    x: &crate::Tensor,
    p: f32,
    seed: u64,
) -> Result<(crate::Tensor, crate::Tensor)> {
    use cudarc::driver::DevicePtr;

    if !(0.0..1.0).contains(&p) {
        return Err(crate::Error::Msg(format!(
            "cuda_dropout: p must be in [0, 1), got {p}"
        )));
    }
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_dropout: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_dropout: input must be contiguous".to_string(),
        ));
    }

    let n = x.element_count();
    let x_bpe = dtype.size_in_bytes();
    let inv_keep = if p == 0.0 { 1.0 } else { 1.0 / (1.0 - p) };

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_dropout: x must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    let y_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n)?;
    let mask_storage = CudaStorage::zeros_ctx(&ctx, device_index, crate::DType::U8, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let y_base = match &y_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let mask_base = match &mask_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let y_ptr = y_base as *mut core::ffi::c_void;
    let mask_ptr = mask_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_dropout_async(
            x_ptr, y_ptr, mask_ptr, n as i64, p, inv_keep, seed, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_dropout: FFI returned status {status}"
        )));
    }

    let y_arc: crate::Storage = Arc::new(y_storage);
    let mask_arc: crate::Storage = Arc::new(mask_storage);
    let y = crate::Tensor::from_parts(
        y_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_dropout: wrap y: {e}")))?;
    let mask = crate::Tensor::from_parts(
        mask_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_dropout: wrap mask: {e}")))?;
    Ok((y, mask))
}

/// CUDA-side tensor-scalar elementwise op: `out[i] = op(x[i], c)`.
///
/// `kind` encodes the op:
/// - `0`: `AddScalar` (`x + c`)
/// - `1`: `SubScalar` (`x - c`)
/// - `2`: `MulScalar` (`x * c`)
/// - `3`: `DivScalar` (`x / c`)
/// - `4`: `ScalarMinusTensor` (`c - x`)
/// - `5`: `ScalarDivTensor` (`c / x`)
/// - `6`: `MaxWithScalar` (`max(x, c)`)
/// - `7`: `MinWithScalar` (`min(x, c)`)
///
/// Dtype inferred from `x.dtype()`; must be F32/BF16/F16. Input must be
/// contiguous and on CUDA. Output is a fresh contiguous tensor of the same
/// shape and dtype.
///
/// Wired by `ScalarOp::cuda_fwd` in `kiln-tensor/src/ops/scalar.rs` (#1082).
#[cfg(feature = "cuda")]
pub fn cuda_scalar_op(x: &crate::Tensor, kind: i32, c: f32) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_scalar_op: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_scalar_op: contiguous input required".to_string(),
        ));
    }
    if !(0..=7).contains(&kind) {
        return Err(crate::Error::Msg(format!(
            "cuda_scalar_op: kind {kind} out of range 0..=7"
        )));
    }

    let n = x.element_count();
    let bpe = dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_scalar_op: x must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): scalar op writes the full output
    // (out[i] = f(x[i], c) for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status =
        unsafe { kiln_scalar_op_async(x_ptr, out_ptr, n as i64, kind, dtype_tag, c, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_scalar_op: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_scalar_op: wrap: {e}")))
}

/// CUDA-side scalar-parameterized unary: `out[i] = f(x[i]; a, b)` where
/// `kind` selects:
///   0 → `clamp(x, lo=a, hi=b)`
///   1 → `pow(x, p=a)` (the `b` parameter is ignored)
///
/// Dtype inferred from `x.dtype()`; must be F32/BF16/F16. Input must
/// be contiguous and on CUDA. Math is promoted to F32 internally and
/// narrowed back to storage dtype, matching the kt-tensor numerical
/// reference. (#1082)
#[cfg(feature = "cuda")]
pub fn cuda_clamp_pow(x: &crate::Tensor, kind: i32, a: f32, b: f32) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_clamp_pow: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_clamp_pow: contiguous input required".to_string(),
        ));
    }

    let n = x.element_count();
    let bpe = dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_clamp_pow: x must be CUDA".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // #1082 (perf, Pattern A): clamp/pow writes the full output
    // (out[i] = clamp/pow(x[i]) for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_clamp_pow_async(x_ptr, out_ptr, n as i64, kind, a, b, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_clamp_pow: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(x.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_clamp_pow: wrap: {e}")))
}

/// CUDA-side element-wise comparison: `out[i] = (op(a[i], b[i])) ? 1 : 0`.
///
/// `kind` encodes the op (0=Eq, 1=Ne, 2=Lt, 3=Le, 4=Gt, 5=Ge). Dtype
/// is inferred from `a.dtype()`; must be F32/BF16/F16. Both inputs
/// must be contiguous and on the same CUDA device.
///
/// Returns a fresh contiguous U8 tensor of the same shape — useful
/// as a mask for `masked_fill` / `where_select`. (#1082)
#[cfg(feature = "cuda")]
pub fn cuda_compare(a: &crate::Tensor, b: &crate::Tensor, kind: i32) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if a.shape() != b.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_compare: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_compare: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_compare: unsupported dtype {other}"
            )));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_compare: contiguous inputs required".to_string(),
        ));
    }

    let n = a.element_count();
    let bpe = dtype.size_in_bytes();

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_compare: a must be CUDA".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_compare: b must be CUDA".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output is U8 (one byte per element).
    // #1082 (perf, Pattern A): compare writes the full output
    // (out[i] = cmp(a[i], b[i]) ? 1 : 0 for all n); uninit skips the memset.
    let out_storage = CudaStorage::alloc_uninit_ctx(&ctx, device_index, crate::DType::U8, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let a_base = match &a_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let b_base = match &b_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let a_off = (a.layout().start_offset() * bpe) as u64;
    let b_off = (b.layout().start_offset() * bpe) as u64;

    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status =
        unsafe { kiln_compare_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_compare: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(a.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_compare: wrap: {e}")))
}

/// CUDA-side ternary mask-based select: `out[i] = mask[i] != 0 ? t[i] : f[i]`.
///
/// `mask` must be U8 on CUDA. `t` / `f` must share dtype (F32/BF16/F16),
/// shape, and CUDA device. All inputs must be contiguous. Output is a
/// fresh contiguous tensor of the same shape + dtype as `t`/`f`. (#1082)
#[cfg(feature = "cuda")]
pub fn cuda_where_select(
    mask: &crate::Tensor,
    t: &crate::Tensor,
    f: &crate::Tensor,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if mask.shape() != t.shape() || t.shape() != f.shape() {
        return Err(crate::Error::Msg(format!(
            "cuda_where_select: shape mismatch mask={:?} t={:?} f={:?}",
            mask.shape(),
            t.shape(),
            f.shape()
        )));
    }
    if mask.dtype() != crate::DType::U8 {
        return Err(crate::Error::Msg(format!(
            "cuda_where_select: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    if t.dtype() != f.dtype() {
        return Err(crate::Error::Msg(format!(
            "cuda_where_select: t/f dtype mismatch t={} f={}",
            t.dtype(),
            f.dtype()
        )));
    }
    let dtype = t.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_where_select: unsupported dtype {other}"
            )));
        }
    };
    if !mask.is_contiguous() || !t.is_contiguous() || !f.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_where_select: contiguous inputs required".to_string(),
        ));
    }

    let n = t.element_count();
    let bpe = dtype.size_in_bytes();

    let mask_storage = mask
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_where_select: mask must be CUDA".to_string()))?;
    let t_storage = t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_where_select: t must be CUDA".to_string()))?;
    let f_storage = f
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_where_select: f must be CUDA".to_string()))?;

    let ctx = t_storage.context();
    let device_index = match t_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let mask_base = match &mask_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let t_base = match &t_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let f_base = match &f_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    // mask is U8 — one byte per element regardless of dtype.
    let mask_off = (mask.layout().start_offset()) as u64;
    let t_off = (t.layout().start_offset() * bpe) as u64;
    let f_off = (f.layout().start_offset() * bpe) as u64;

    let mask_ptr = (mask_base + mask_off) as *const core::ffi::c_void;
    let t_ptr = (t_base + t_off) as *const core::ffi::c_void;
    let f_ptr = (f_base + f_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_where_select_async(
            mask_ptr, t_ptr, f_ptr, out_ptr, n as i64, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_where_select: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(t.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_where_select: wrap: {e}")))
}

/// CUDA-side extract of the main diagonal of an `[n, n]` square matrix.
///
/// Input `x` must be CUDA-resident, contiguous, square rank-2, dtype
/// F32/BF16/F16. Returns a fresh `[n]` tensor. (#1082)
#[cfg(feature = "cuda")]
pub fn cuda_diagonal_extract(x: &crate::Tensor) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if x.rank() != 2 {
        return Err(crate::Error::Msg(format!(
            "cuda_diagonal_extract: input must be rank-2, got {:?}",
            x.shape()
        )));
    }
    let n = x.shape()[0];
    if x.shape()[1] != n {
        return Err(crate::Error::Msg(format!(
            "cuda_diagonal_extract: input must be square, got {:?}",
            x.shape()
        )));
    }
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_diagonal_extract: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_diagonal_extract: input must be contiguous".to_string(),
        ));
    }

    let bpe = dtype.size_in_bytes();
    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_diagonal_extract: x must be CUDA".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status =
        unsafe { kiln_diagonal_extract_async(x_ptr, out_ptr, n as i64, dtype_tag, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_diagonal_extract: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(vec![n]),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_diagonal_extract: wrap: {e}")))
}

/// CUDA-side construction of a diagonal matrix from a rank-1 vector.
///
/// Input `v` of length `n` produces a fresh `[n, n]` zero-initialized
/// tensor with `v` placed on the main diagonal. F32/BF16/F16. (#1082)
#[cfg(feature = "cuda")]
pub fn cuda_diag_build(v: &crate::Tensor) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;
    use std::any::Any as _;

    if v.rank() != 1 {
        return Err(crate::Error::Msg(format!(
            "cuda_diag_build: input must be rank-1, got {:?}",
            v.shape()
        )));
    }
    let dtype = v.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_diag_build: unsupported dtype {other}"
            )));
        }
    };
    if !v.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_diag_build: input must be contiguous".to_string(),
        ));
    }

    let n = v.element_count();
    let bpe = dtype.size_in_bytes();
    let v_storage = v
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_diag_build: v must be CUDA".to_string()))?;
    let ctx = v_storage.context();
    let device_index = match v_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Pre-zero the output; the kernel only writes the n diagonal entries.
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, n * n)?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let v_base = match &v_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let v_off = (v.layout().start_offset() * bpe) as u64;
    let v_ptr = (v_base + v_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe { kiln_diag_build_async(v_ptr, out_ptr, n as i64, dtype_tag, raw_stream) };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_diag_build: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(vec![n, n]),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_diag_build: wrap: {e}")))
}

/// CUDA cumulative sum along the trailing axis (Phase 6 scan kernel,
/// #1082).
///
/// Operates on a contiguous `[..., D]` tensor; produces a fresh
/// contiguous tensor of the same shape and dtype with each row scanned
/// (inclusive prefix sum) over the trailing axis.
///
/// Routes through `kiln_scan_last_axis_async(..., kind=0)` in
/// `csrc/scan_axis.cu`. F32/BF16/F16 supported. F32 accumulation.
#[cfg(feature = "cuda")]
pub fn cuda_cumsum_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    cuda_scan_axis_impl(x, axis, 0, "cuda_cumsum_axis")
}

/// CUDA cumulative product along the trailing axis (Phase 6 scan
/// kernel, #1082).
///
/// Operates on a contiguous `[..., D]` tensor; produces a fresh
/// contiguous tensor of the same shape and dtype with each row scanned
/// (inclusive prefix product) over the trailing axis.
///
/// Routes through `kiln_scan_last_axis_async(..., kind=1)` in
/// `csrc/scan_axis.cu`. F32/BF16/F16 supported. F32 accumulation.
#[cfg(feature = "cuda")]
pub fn cuda_cumprod_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    cuda_scan_axis_impl(x, axis, 1, "cuda_cumprod_axis")
}

#[cfg(feature = "cuda")]
fn cuda_scan_axis_impl(
    x: &crate::Tensor,
    axis: usize,
    kind: i32,
    label: &str,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "{label}: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(format!(
            "{label}: input must be contiguous"
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: input must have rank >= 1"
        )));
    }
    if axis != rank - 1 {
        return Err(crate::Error::Msg(format!(
            "{label}: only last-axis scan supported (axis={axis}, rank={rank})"
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = if shape[rank - 1] == 0 {
        0
    } else {
        (x.element_count() / shape[rank - 1]) as i64
    };

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg(format!("{label}: input must be CUDA")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, dtype, x.element_count())?;

    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_base = match &x_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_scan_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, kind, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "{label}: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}
