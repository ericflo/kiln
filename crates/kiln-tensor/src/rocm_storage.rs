//! ROCm/HIP storage impl behind the `rocm` feature flag (Phase R.3).
//!
//! The candle-free ROCm analog of [`crate::cuda_storage`]: wraps a
//! `kiln_hip::RocmSlice` (the device buffer) + dtype + `Arc<RocmContext>` for
//! stream affinity. `RocmStorage` is the `StorageBackend` impl that makes
//! `Device::Rocm` tensors allocatable; the math ops that consume it land in
//! Phase R.5 (the hipcc-compiled `csrc/*.cu` kernels).
//!
//! This file mirrors `cuda_storage.rs`'s storage core 1:1, swapping
//! `cudarc::driver::{CudaContext, CudaStream, CudaSlice}` for
//! `kiln_hip::{RocmContext, RocmStream, RocmSlice}` and routing every stream
//! resolution through [`crate::active_rocm_stream`]. The capture-arena fast
//! path (CUDA-graph freeze-pointers) is deferred to Phase R.9.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use kiln_hip::{RocmContext, RocmSlice};

/// Diagnostic: counts host<->device round-trips (each one synchronizes its
/// stream via `memcpy_dtoh`/`memcpy_htod`). When `KILN_ROCM_PROFILE` is set,
/// `rocm_to_host_copy` / `host_to_rocm_copy` bump these and emit a periodic
/// line so we can see whether the prefill hot path is host-bound.
pub static ROCM_DTOH_COUNT: AtomicU64 = AtomicU64::new(0);
pub static ROCM_HTOD_COUNT: AtomicU64 = AtomicU64::new(0);

/// Current host→device copy count. The HIP-graph capture path snapshots this
/// across the warm forward to detect a host round-trip (`host_to_rocm_copy`),
/// which is illegal inside `hipStreamBeginCapture` — so any forward that does
/// one is not capture-safe and the capture is skipped for that geometry.
pub fn rocm_htod_count() -> u64 {
    ROCM_HTOD_COUNT.load(Ordering::Relaxed)
}

#[inline]
fn rocm_profile_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("KILN_ROCM_PROFILE").is_ok())
}

/// Diagnostic (KILN_ROCM_BT=1): print a symbolized backtrace the FIRST time each
/// distinct (direction, shape) host round-trip happens past warmup, so we can
/// localize every remaining decode-region sync (the hard prerequisite for HIP
/// graph capture). Build with `RUSTFLAGS=-Cdebuginfo=1` so frames symbolize.
fn rocm_bt_once(dir: &str, shape: &[usize], total: u64) {
    if total < 1000 {
        return; // skip weight-load / warmup
    }
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("KILN_ROCM_BT").is_ok()) {
        return;
    }
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let key = format!("{dir}{shape:?}");
    let mut seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new())).lock().unwrap();
    if seen.len() < 16 && seen.insert(key) {
        eprintln!(
            "[rocm-bt] {dir} {shape:?}:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
}

/// Diagnostic: tallies how often each generic `DeviceOp` falls through to the
/// CPU host-fallback path (a full D2H→cpu_fwd→H2D round-trip per call). When
/// `KILN_ROCM_PROFILE` is set, the running tally per op-name is printed every
/// 100 fallbacks so we can see which ops dominate the host-bound prefill.
pub fn rocm_log_host_fallback(op_name: &str, shape: &[usize]) {
    if !rocm_profile_on() {
        return;
    }
    static TALLY: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    let n = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let mut map = TALLY.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    *map.entry(op_name.to_string()).or_insert(0) += 1;
    if n % 100 == 0 {
        let mut v: Vec<(String, u64)> = map.iter().map(|(k, c)| (k.clone(), *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<String> = v.iter().take(12).map(|(k, c)| format!("{k}={c}")).collect();
        eprintln!(
            "[rocm-fallback] total={n} last={op_name}{shape:?} top: {}",
            top.join(" ")
        );
    }
}

use crate::{DType, Device, Error, Result, StorageBackend};

// The ROCm-side kernel launchers live in `csrc/*.cu`, compiled by
// `build.rs::build_rocm()` into `libkiln_tensor_rocm_ops.a` (same stable C ABI
// as the CUDA build). Phase R.3 uses only the contiguity-copy launcher; the
// rest join as their kernels gain the Phase R.5 wave-size fix.
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

    fn kiln_softmax_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Owner of a ROCm byte buffer — either kt owns the allocation (`Owned`) or kt
/// shares a buffer owned elsewhere (`Borrowed`, kept alive by an opaque Arc).
/// The Borrowed variant is the foundation for the zero-copy kt-bridge adapter
/// (Phase R.4): dropping it just decrements the keep-alive Arc, never freeing
/// device memory directly.
pub(crate) enum SliceOwner {
    Owned(RocmSlice),
    /// Borrowed view over an externally-owned HIP buffer. `ptr` is the raw
    /// device address (`hipDeviceptr_t` normalized to a `u64`). `_keep_alive`
    /// must outlive every read from `ptr`.
    Borrowed {
        ptr: u64,
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

/// ROCm-backed storage. Byte-typed; dtype carried alongside for dispatch.
///
/// Either owns its `RocmSlice` allocation or borrows an external HIP buffer via
/// a keep-alive Arc (see [`SliceOwner`]). Holds an `Arc<RocmContext>` for stream
/// affinity — every kernel-launch path resolves
/// `crate::active_rocm_stream(&self.ctx)` to get the live stream handle.
#[derive(Debug)]
pub struct RocmStorage {
    device: Device,
    dtype: DType,
    slice: SliceOwner,
    ctx: Arc<RocmContext>,
}

impl RocmStorage {
    /// Allocate `n_elements` of `dtype`, zero-initialized, on `ctx`. Routes
    /// through the thread-local active stream so the alloc is captured on the
    /// capture stream during HIP-graph capture (Phase R.9); outside capture
    /// this is exactly `ctx.default_stream()`.
    pub fn zeros_ctx(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        // HIP-graph freeze-pointers (R.9): while a capture arena is active on
        // this thread, route through it (Borrowed view into a pre-allocated,
        // pointer-stable arena buffer) instead of a fresh hipMallocAsync.
        if let Some(result) = crate::rocm_capture_arena_alloc(dtype, n_elements, true) {
            return result;
        }
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let slice = crate::active_rocm_stream(ctx)
            .alloc_zeros(byte_len)
            .map_err(|e| {
                Error::Msg(format!(
                    "RocmStorage::zeros_ctx: active_rocm_stream(ctx).alloc_zeros({byte_len}) \
                     failed: {e:?}"
                ))
            })?;
        Ok(RocmStorage {
            device: Device::Rocm(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            ctx: ctx.clone(),
        })
    }

    /// Allocate `n_elements` of `dtype` **UNINITIALIZED** on `ctx` — skips the
    /// zero-fill. The caller MUST fully overwrite the buffer before any read
    /// (use [`Self::zeros_ctx`] for read-before-write / accumulation outputs).
    pub fn alloc_uninit_ctx(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        // HIP-graph freeze-pointers (R.9): see `zeros_ctx`. `zero = false` here
        // — the arena hands out an uninitialized Borrowed view on Record, and on
        // Replay re-zeros only under KILN_ARENA_FORCE_ZERO.
        if let Some(result) = crate::rocm_capture_arena_alloc(dtype, n_elements, false) {
            return result;
        }
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let slice = crate::active_rocm_stream(ctx).alloc(byte_len).map_err(|e| {
            Error::Msg(format!(
                "RocmStorage::alloc_uninit_ctx: active_rocm_stream(ctx).alloc({byte_len}) \
                 failed: {e:?}"
            ))
        })?;
        Ok(RocmStorage {
            device: Device::Rocm(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            ctx: ctx.clone(),
        })
    }

    /// Wrap an existing `RocmSlice` allocated by the caller. Validates length
    /// against `dtype.size_in_bytes()` for non-packed dtypes.
    pub fn from_slice_ctx(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        dtype: DType,
        slice: RocmSlice,
    ) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !slice.len().is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "RocmStorage::from_slice_ctx: slice len {} is not a multiple of \
                     size_in_bytes({:?}) = {}",
                    slice.len(),
                    dtype,
                    per
                )));
            }
        }
        Ok(RocmStorage {
            device: Device::Rocm(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            ctx: ctx.clone(),
        })
    }

    /// Wrap an externally-owned HIP buffer as a kt `RocmStorage` without
    /// copying. `keep_alive` must outlive every read from `device_ptr`.
    pub fn from_borrowed_ctx(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        dtype: DType,
        device_ptr: u64,
        byte_len: usize,
        keep_alive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !byte_len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "RocmStorage::from_borrowed_ctx: byte_len {byte_len} is not a multiple of \
                     size_in_bytes({dtype:?}) = {per}"
                )));
            }
        }
        Ok(RocmStorage {
            device: Device::Rocm(device_index),
            dtype,
            slice: SliceOwner::Borrowed {
                ptr: device_ptr,
                byte_len,
                _keep_alive: keep_alive,
            },
            ctx: ctx.clone(),
        })
    }

    /// Whether this storage owns its underlying HIP buffer.
    pub fn is_owned(&self) -> bool {
        matches!(self.slice, SliceOwner::Owned(_))
    }

    /// Whether this storage borrows its buffer from an external owner.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.slice, SliceOwner::Borrowed { .. })
    }

    /// Borrow the underlying `RocmSlice`. **Panics** on a `Borrowed` storage —
    /// use [`Self::device_ptr_raw`] for the owner-agnostic raw pointer.
    pub fn slice(&self) -> &RocmSlice {
        match &self.slice {
            SliceOwner::Owned(s) => s,
            SliceOwner::Borrowed { .. } => panic!(
                "RocmStorage::slice() called on Borrowed storage; use device_ptr_raw() which \
                 supports both owners"
            ),
        }
    }

    /// Mutable borrow for in-place ops. **Panics** on a `Borrowed` storage —
    /// borrowed buffers are read-only through kt.
    pub fn slice_mut(&mut self) -> &mut RocmSlice {
        match &mut self.slice {
            SliceOwner::Owned(s) => s,
            SliceOwner::Borrowed { .. } => panic!(
                "RocmStorage::slice_mut() called on Borrowed storage; borrowed buffers are \
                 read-only through kt"
            ),
        }
    }

    /// Raw device pointer (`hipDeviceptr_t` as `u64`) + byte length, for both
    /// `Owned` and `Borrowed` variants. Callers add the kt-Tensor's
    /// `layout.start_offset() * dtype.size_in_bytes()` to reach the live region.
    pub fn device_ptr_raw(&self) -> (u64, usize) {
        match &self.slice {
            SliceOwner::Owned(s) => (s.device_ptr() as u64, s.len()),
            SliceOwner::Borrowed { ptr, byte_len, .. } => (*ptr, *byte_len),
        }
    }

    /// The `RocmContext` this storage was allocated on — cheap Arc clone.
    pub fn context(&self) -> Arc<RocmContext> {
        self.ctx.clone()
    }

    /// Raw HIP stream pointer for FFI dispatch — the `stream` argument every
    /// kernel-crate FFI declaration expects (`*mut c_void`). Resolves through
    /// the thread-local active stream (capture stream during HIP-graph capture;
    /// `ctx.default_stream()` otherwise).
    pub fn rocm_stream_raw(&self) -> *mut core::ffi::c_void {
        crate::active_rocm_stream(&self.ctx).hip_stream() as *mut core::ffi::c_void
    }

    /// Crate-internal accessor for the slice owner.
    pub(crate) fn slice_owner(&self) -> &SliceOwner {
        &self.slice
    }
}

/// The primary `Arc<RocmContext>` for `device_index` — the ROCm analog of
/// `primary_cuda_context`. **Cached per device** (process-global): every caller
/// for a given ordinal shares ONE context and therefore ONE default stream.
///
/// This caching is load-bearing for correctness, not just perf: HIP's runtime
/// API has no implicit primary-context retain like cudarc's, so without the
/// cache each allocation would mint a fresh context + stream. An output
/// tensor's async zeroing memset and the kernel that writes it would then run on
/// unordered streams and race (nondeterministically zeroing valid results).
/// One cached context per device serializes alloc/memset/kernel/readback on the
/// shared default stream, matching the CUDA backend's single-primary-stream
/// behavior. `Err` if the device isn't available.
pub fn primary_rocm_context(device_index: usize) -> Result<Arc<RocmContext>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<RocmContext>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache
        .lock()
        .map_err(|_| Error::Msg("primary_rocm_context: cache mutex poisoned".to_string()))?;
    if let Some(ctx) = map.get(&device_index) {
        return Ok(Arc::clone(ctx));
    }
    let ctx = RocmContext::new(device_index)
        .map_err(|e| Error::Msg(format!("primary_rocm_context({device_index}): {e}")))?;
    map.insert(device_index, Arc::clone(&ctx));
    Ok(ctx)
}

/// Whether a HIP runtime + at least one AMD device is present. The ROCm analog
/// of `cuda_is_available`; swallows driver errors into `false`.
pub fn rocm_is_available() -> bool {
    kiln_hip::is_available()
}

impl StorageBackend for RocmStorage {
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

/// Construct a fresh [`crate::Storage`] holding a zeroed [`RocmStorage`].
/// The ROCm analog of `cuda_zeros_ctx`.
pub fn rocm_zeros_ctx(device_index: usize, dtype: DType, n_elements: usize) -> Result<crate::Storage> {
    let ctx = primary_rocm_context(device_index)?;
    let storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
}

/// Block until all work on `device_index`'s device completes. ROCm analog of
/// `cuda_synchronize_default_stream`.
pub fn rocm_synchronize_default_stream(device_index: usize) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    ctx.synchronize()
        .map_err(|e| Error::Msg(format!("rocm_synchronize_default_stream({device_index}): {e:?}")))
}

/// Block until all work on the ACTIVE compute stream completes
/// (`hipStreamSynchronize`, not the device-wide `hipDeviceSynchronize`). Cheaper
/// than [`rocm_synchronize_default_stream`] when other (e.g. hipBLASLt-internal)
/// streams have pending work we don't need to wait on.
pub fn rocm_synchronize_compute_stream(device_index: usize) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    crate::active_rocm_stream(&ctx)
        .synchronize()
        .map_err(|e| Error::Msg(format!("rocm_synchronize_compute_stream({device_index}): {e:?}")))
}

/// Refresh `dst`'s contents in place from a host slice WITHOUT reallocating —
/// the ROCm analog of [`crate::cuda_write_host_in_place`].
///
/// The HIP-graph replay path (R.9) bakes `dst`'s device pointer into the
/// captured graph; replay refreshes the per-step inputs (token id, position,
/// rotary tables, paged-KV metadata) through THIS function so the pointer
/// never changes and the recorded kernels read the new values. Unlike
/// [`host_to_rocm_copy`] (which mints a fresh device buffer with a NEW
/// pointer), this writes through `dst`'s already-allocated storage.
///
/// `dst` must be ROCm-backed, contiguous, `start_offset == 0`, and own exactly
/// `host.len()` elements whose element type `E` matches `dst`'s dtype byte
/// width. The copy runs on the kt active stream (the default stream on the
/// replay path) and is issued WITHOUT a trailing synchronize — the per-token
/// replay refreshes ~7 of these buffers, and a sync after each one dominated the
/// replay cost (decode dropped to ~5 tok/s). Ordering is instead provided by the
/// single `rocm_synchronize_default_stream` the replay path issues before the
/// graph launch, which is the actual cross-stream guarantee.
///
/// Host-buffer safety: the staged `Vec<u8>` is pageable, and
/// `hipMemcpyHtoDAsync` copies pageable host memory into a pinned staging buffer
/// SYNCHRONOUSLY before returning (it cannot DMA pageable memory directly), so
/// the local buffer is fully consumed by the time this function returns — only
/// the device-side write remains queued. No dangling host read.
#[cfg(feature = "rocm")]
pub fn rocm_write_host_in_place<E: crate::Element>(
    dst: &crate::Tensor,
    host: &[E],
) -> Result<()> {
    if dst.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_write_host_in_place: packed dtype not supported".to_string(),
        ));
    }
    if E::DTYPE.size_in_bytes() != dst.dtype().size_in_bytes() {
        return Err(Error::Msg(format!(
            "rocm_write_host_in_place: element byte width {} != dst dtype {} byte width {}",
            E::DTYPE.size_in_bytes(),
            dst.dtype(),
            dst.dtype().size_in_bytes()
        )));
    }
    if !dst.is_contiguous() || dst.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "rocm_write_host_in_place: dst must be contiguous with start_offset == 0".to_string(),
        ));
    }
    let n = dst.element_count();
    if host.len() != n {
        return Err(Error::Msg(format!(
            "rocm_write_host_in_place: host len {} != dst element count {n}",
            host.len()
        )));
    }

    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_write_host_in_place: dst must be ROCm storage".to_string())
        })?;

    let ctx = dst_storage.context();
    let stream = crate::active_rocm_stream(&ctx);
    let (dst_base, _byte_len) = dst_storage.device_ptr_raw();
    let bytes = E::to_bytes(host);

    // SAFETY: `dst_base` is the start of `dst`'s contiguous device buffer
    // (validated start_offset == 0); `bytes` is exactly `n * size_in_bytes`
    // bytes, matching the destination region, so the copy stays inside the
    // allocation. The copy is issued on the kt active stream — the same stream
    // the captured graph runs on during replay — and is not synchronized here
    // beyond the explicit stream sync below.
    unsafe {
        stream
            .memcpy_htod_raw_async(dst_base as *mut core::ffi::c_void, &bytes)
            .map_err(|e| {
                Error::Msg(format!(
                    "rocm_write_host_in_place: memcpy_htod_raw_async: {e:?}"
                ))
            })?;
    }
    // No trailing synchronize — see the doc comment. The replay path syncs the
    // default stream once before the graph launch.
    Ok(())
}

// ----------------------------------------------------------------------
// ROCm-side Tensor::contiguous — the first storage→kernel op, proving the
// hipcc FFI path end-to-end (contiguous.cu is compiled in build_rocm()).
// ----------------------------------------------------------------------

/// Stride-aware copy of a (possibly non-contiguous) ROCm storage into a fresh
/// contiguous output. ROCm analog of `cuda_contiguous`.
pub fn rocm_contiguous(src: &crate::Tensor) -> Result<crate::Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_contiguous: packed dtype not supported".to_string(),
        ));
    }

    let layout = src.layout();
    let shape = src.shape();
    let strides_elems = src.strides();
    let rank = shape.len();
    if rank > 8 {
        return Err(Error::Msg(format!(
            "rocm_contiguous: rank {rank} exceeds kernel MAX_RANK=8"
        )));
    }
    let n_elements = src.element_count();
    let bpe = src.dtype().size_in_bytes();

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_contiguous: source must be ROCm storage".to_string()))?;

    let ctx = src_storage.context();
    let device_index = match src_storage.device {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };
    let dst_storage = RocmStorage::zeros_ctx(&ctx, device_index, src.dtype(), n_elements)?;

    let raw_stream = src_storage.rocm_stream_raw();

    let (src_base, _) = src_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();

    let src_byte_off = (layout.start_offset() * bpe) as u64;
    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

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
        return Err(Error::Msg(format!(
            "rocm_contiguous: kiln_contiguous_copy_async returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_contiguous: wrap: {e}")))
}

/// In-place `slice_set` along dim 0 on a ROCm tensor: flat-copy `src` into
/// `dst`'s existing device buffer starting at outer-axis `offset`. ROCm analog
/// of `cuda_slice_set_dim0` (reuses the contiguity-copy kernel for the d2d write
/// at the computed byte offset). The GDN decode resident-state restore is the
/// production caller.
pub fn rocm_slice_set_dim0(dst: &crate::Tensor, src: &crate::Tensor, offset: usize) -> Result<()> {
    if dst.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_slice_set: packed dtype not supported".to_string(),
        ));
    }
    let bpe = dst.dtype().size_in_bytes();
    let inner: usize = dst.dims().iter().skip(1).product();
    let src_n = src.element_count();

    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_slice_set: dst must be ROCm storage".to_string()))?;
    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_slice_set: src must be ROCm storage".to_string()))?;

    let raw_stream = dst_storage.rocm_stream_raw();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();
    let src_byte_off = (src.layout().start_offset() * bpe) as u64;
    let dst_byte_off = ((offset * inner) * bpe) as u64;
    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let dst_ptr = (dst_base + dst_byte_off) as *mut core::ffi::c_void;

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
        return Err(Error::Msg(format!(
            "rocm_slice_set: kiln_contiguous_copy_async returned status {status}"
        )));
    }
    Ok(())
}

/// Copy a ROCm-resident tensor back to host (CPU) storage. ROCm analog of
/// `cuda_to_host_copy`.
pub fn rocm_to_host_copy(src: &crate::Tensor) -> Result<crate::Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(format!(
            "rocm_to_host_copy: packed dtype {} not supported",
            src.dtype()
        )));
    }

    // Force a contiguous, start_offset=0 device buffer first (Owned output with
    // a usable `slice()`).
    let contig = rocm_contiguous(src)?;
    let contig_storage = contig
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_to_host_copy: contiguous'd storage must be RocmStorage".to_string())
        })?;

    let dtype = src.dtype();
    let ctx = contig_storage.context();
    let stream = crate::active_rocm_stream(&ctx);
    let host_bytes = stream
        .memcpy_dtoh(contig_storage.slice())
        .map_err(|e| Error::Msg(format!("rocm_to_host_copy: memcpy_dtoh failed: {e:?}")))?;

    if rocm_profile_on() {
        let n = ROCM_DTOH_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        rocm_bt_once("dtoh", src.shape(), n);
        if n % 200 == 0 {
            eprintln!(
                "[rocm-profile] dtoh={} htod={} (last shape {:?})",
                n,
                ROCM_HTOD_COUNT.load(Ordering::Relaxed),
                src.shape()
            );
        }
    }

    let cpu_storage = crate::CpuStorage::from_bytes(dtype, host_bytes)?;
    let storage_arc: crate::Storage = Arc::new(cpu_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Copy a contiguous host (CPU) tensor up to a fresh ROCm buffer on
/// `device_index`. ROCm analog of `host_to_cuda_copy`.
pub fn host_to_rocm_copy(src: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(format!(
            "host_to_rocm_copy: packed dtype {} not supported",
            src.dtype()
        )));
    }
    let _cpu_storage = src
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| Error::Msg("host_to_rocm_copy: source must be CPU storage".to_string()))?;

    let dtype = src.dtype();
    let n_elements = src.element_count();
    let byte_len = dtype.packed_buffer_bytes(n_elements);

    let contig_src = if src.is_contiguous() && src.layout().start_offset() == 0 {
        src.clone()
    } else {
        src.contiguous()?
    };
    let contig_cpu = contig_src
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| Error::Msg("host_to_rocm_copy: contig src must be CPU storage".to_string()))?;
    let bytes = contig_cpu.as_bytes();
    if bytes.len() != byte_len {
        return Err(Error::Msg(format!(
            "host_to_rocm_copy: src byte_len {} != expected {}",
            bytes.len(),
            byte_len
        )));
    }

    let ctx = primary_rocm_context(device_index)?;
    let stream = crate::active_rocm_stream(&ctx);
    let device_slice = stream
        .clone_htod(bytes)
        .map_err(|e| Error::Msg(format!("host_to_rocm_copy: clone_htod failed: {e:?}")))?;

    // Always count (cheap atomic) — the HIP-graph capture-safety check reads this
    // via `rocm_htod_count()` to detect a host round-trip during the warm pass.
    // The profiling OUTPUT below stays gated behind KILN_ROCM_PROFILE.
    let n = ROCM_HTOD_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if rocm_profile_on() {
        rocm_bt_once("htod", src.shape(), n);
        if n % 200 == 0 {
            eprintln!(
                "[rocm-profile] htod={} dtoh={} (last shape {:?})",
                n,
                ROCM_DTOH_COUNT.load(Ordering::Relaxed),
                src.shape()
            );
        }
    }

    let rocm_storage = RocmStorage::from_slice_ctx(&ctx, device_index, dtype, device_slice)?;

    let storage_arc: crate::Storage = Arc::new(rocm_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Host → ROCm copy — back-compat alias for [`host_to_rocm_copy`], mirroring
/// `host_to_cuda_copy_ctx`.
pub fn host_to_rocm_copy_ctx(src: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    host_to_rocm_copy(src, device_index)
}

/// Softmax over the last axis of a contiguous ROCm tensor. ROCm analog of
/// `cuda_softmax_last_axis`, routing through the wave-size-fixed `softmax.cu`
/// kernel (Phase R.5). F32 / BF16 / F16; F32 accumulation throughout.
pub fn rocm_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_softmax_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_softmax_last_axis: input must be contiguous".to_string(),
        ));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(
            "rocm_softmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_softmax_last_axis: input must be ROCm".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };
    // Softmax writes every output element (Pass 3), so skip the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_softmax_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_softmax_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}
