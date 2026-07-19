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

use kiln_hip::{
    RocmContext, RocmExecutionPolicy, RocmSlice, RocmStreamId, RocmStreamSubmission,
    RocmSyncReason, RocmSyncTelemetrySnapshot,
};

/// Process-wide count of successful host-to-device copies.
///
/// Capture-safety decisions use [`with_rocm_htod_observer`] instead: this
/// aggregate can change because another thread or ROCm device made progress.
pub static ROCM_HTOD_COUNT: AtomicU64 = AtomicU64::new(0);

const ROCM_HTOD_MAX_UNIQUE_SITES: usize = 32;

/// One bounded, source-attributed host-to-ROCm transfer site observed inside a
/// dynamic capture-safety scope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RocmHtodCopySite {
    /// Rust source file at the outermost tracked upload call site.
    pub source_file: &'static str,
    /// One-indexed source line at the outermost tracked upload call site.
    pub source_line: u32,
    /// One-indexed source column at the outermost tracked upload call site.
    pub source_column: u32,
    /// Element type copied to the device.
    pub dtype: crate::DType,
    /// Number of tensor elements in each aggregated copy.
    pub elements_per_copy: u64,
    /// Number of bytes in each aggregated copy.
    pub bytes_per_copy: u64,
    /// Number of matching copies observed at this site.
    pub copy_count: u64,
    /// Total bytes copied by all matching copies at this site.
    pub total_bytes: u64,
}

/// Bounded host-to-ROCm transfer evidence for one dynamic observation scope.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RocmHtodObservation {
    /// Total number of matching copies observed in the scope.
    pub copy_count: u64,
    /// Total number of bytes copied in the scope.
    pub total_bytes: u64,
    /// Per-site aggregates, bounded by `ROCM_HTOD_MAX_UNIQUE_SITES`.
    pub sites: Vec<RocmHtodCopySite>,
    /// Copies omitted after the unique-site bound was reached.
    pub unattributed_copy_count: u64,
    /// Bytes copied by sites omitted after the unique-site bound was reached.
    pub unattributed_bytes: u64,
}

#[derive(Clone)]
struct RocmHtodObserverState {
    device_index: usize,
    observation: RocmHtodObservation,
}

thread_local! {
    static ROCM_HTOD_OBSERVERS: std::cell::RefCell<Vec<RocmHtodObserverState>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct RocmHtodObserverGuard {
    depth: usize,
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl Drop for RocmHtodObserverGuard {
    fn drop(&mut self) {
        ROCM_HTOD_OBSERVERS.with(|observers| observers.borrow_mut().truncate(self.depth));
    }
}

/// Run `operation` while counting successful host-to-ROCm copies issued on this
/// thread for exactly `device_index`. Unrelated threads and devices cannot
/// create false graph-capture safety failures.
pub fn with_rocm_htod_observer<R>(device_index: usize, operation: impl FnOnce() -> R) -> (R, u64) {
    let (output, observation) = with_rocm_htod_observer_detailed(device_index, operation);
    (output, observation.copy_count)
}

/// Run `operation` while collecting bounded, source-attributed successful
/// host-to-ROCm copies issued on this thread for exactly `device_index`.
pub fn with_rocm_htod_observer_detailed<R>(
    device_index: usize,
    operation: impl FnOnce() -> R,
) -> (R, RocmHtodObservation) {
    let depth = ROCM_HTOD_OBSERVERS.with(|observers| {
        let mut observers = observers.borrow_mut();
        let depth = observers.len();
        observers.push(RocmHtodObserverState {
            device_index,
            observation: RocmHtodObservation::default(),
        });
        depth
    });
    let guard = RocmHtodObserverGuard {
        depth,
        _not_send: std::marker::PhantomData,
    };
    let output = operation();
    let observation =
        ROCM_HTOD_OBSERVERS.with(|observers| observers.borrow()[depth].observation.clone());
    drop(guard);
    (output, observation)
}

#[track_caller]
#[cfg(test)]
fn record_rocm_htod(device_index: usize) {
    record_rocm_htod_copy(
        device_index,
        0,
        crate::DType::U8,
        0,
        std::panic::Location::caller(),
    );
}

fn record_rocm_htod_copy(
    device_index: usize,
    byte_len: u64,
    dtype: crate::DType,
    element_count: u64,
    source: &'static std::panic::Location<'static>,
) {
    ROCM_HTOD_OBSERVERS.with(|observers| {
        for observer in observers.borrow_mut().iter_mut() {
            if observer.device_index == device_index {
                let observation = &mut observer.observation;
                observation.copy_count = observation.copy_count.saturating_add(1);
                observation.total_bytes = observation.total_bytes.saturating_add(byte_len);
                if let Some(site) = observation.sites.iter_mut().find(|site| {
                    site.source_file == source.file()
                        && site.source_line == source.line()
                        && site.source_column == source.column()
                        && site.dtype == dtype
                        && site.elements_per_copy == element_count
                        && site.bytes_per_copy == byte_len
                }) {
                    site.copy_count = site.copy_count.saturating_add(1);
                    site.total_bytes = site.total_bytes.saturating_add(byte_len);
                } else if observation.sites.len() < ROCM_HTOD_MAX_UNIQUE_SITES {
                    observation.sites.push(RocmHtodCopySite {
                        source_file: source.file(),
                        source_line: source.line(),
                        source_column: source.column(),
                        dtype,
                        elements_per_copy: element_count,
                        bytes_per_copy: byte_len,
                        copy_count: 1,
                        total_bytes: byte_len,
                    });
                } else {
                    observation.unattributed_copy_count =
                        observation.unattributed_copy_count.saturating_add(1);
                    observation.unattributed_bytes =
                        observation.unattributed_bytes.saturating_add(byte_len);
                }
            }
        }
    });
}

pub fn rocm_htod_count() -> u64 {
    ROCM_HTOD_COUNT.load(Ordering::Relaxed)
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

    fn kiln_transpose_4d_12_copy_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        bsz: i64,
        heads: i64,
        seq: i64,
        dim: i64,
        stride_b: i64,
        stride_h: i64,
        stride_t: i64,
        bytes_per_elem: i32,
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
    fn validate_context_device(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        operation: &'static str,
    ) -> Result<()> {
        ctx.ensure_execution_available(operation)
            .map_err(|error| Error::Msg(format!("{operation}: {error}")))?;
        if ctx.ordinal() != device_index {
            return Err(Error::Msg(format!(
                "{operation}: context device {} does not match requested device {device_index}",
                ctx.ordinal()
            )));
        }
        Ok(())
    }

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
        Self::validate_context_device(ctx, device_index, "RocmStorage::zeros_ctx")?;
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
        Self::validate_context_device(ctx, device_index, "RocmStorage::alloc_uninit_ctx")?;
        // HIP-graph freeze-pointers (R.9): see `zeros_ctx`. `zero = false` here
        // — the arena hands out an uninitialized Borrowed view on Record, and on
        // Replay re-zeros only under KILN_ARENA_FORCE_ZERO.
        if let Some(result) = crate::rocm_capture_arena_alloc(dtype, n_elements, false) {
            return result;
        }
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let slice = crate::active_rocm_stream(ctx)
            .alloc(byte_len)
            .map_err(|e| {
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
    ///
    /// # Safety
    /// The caller must prove all writes that initialize `slice` are complete or
    /// ordered before the primary context's default stream consumes this
    /// storage. Safe kiln call sites use a synchronizing host-to-device copy.
    pub unsafe fn from_slice_ctx(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        dtype: DType,
        slice: RocmSlice,
    ) -> Result<Self> {
        Self::validate_context_device(ctx, device_index, "RocmStorage::from_slice_ctx")?;
        if slice.stream().ordinal() != device_index {
            return Err(Error::Msg(format!(
                "RocmStorage::from_slice_ctx: allocation stream device {} does not match requested device {device_index}",
                slice.stream().ordinal()
            )));
        }
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
    ///
    /// # Safety
    /// The caller must establish producer-to-consumer ordering for
    /// `device_ptr`. Kiln uses this only for capture-arena views whose owned
    /// buffer and all consumers are explicitly ordered on the graph stream.
    pub unsafe fn from_borrowed_ctx(
        ctx: &Arc<RocmContext>,
        device_index: usize,
        dtype: DType,
        device_ptr: u64,
        byte_len: usize,
        keep_alive: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self> {
        Self::validate_context_device(ctx, device_index, "RocmStorage::from_borrowed_ctx")?;
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

    /// Opaque identity of the stream that would receive this storage's next
    /// operation. This is safe for equality checks only and grants no execution
    /// admission.
    pub fn rocm_stream_id(&self) -> RocmStreamId {
        crate::active_rocm_stream(&self.ctx).id()
    }

    /// Opaque identity of this storage context's owning default stream.
    pub fn rocm_owner_stream_id(&self) -> RocmStreamId {
        self.ctx.default_stream().id()
    }

    /// Acquire a typed HIP stream submission for one external FFI dispatch.
    ///
    /// Resolves through the thread-local active stream (capture stream during
    /// HIP-graph capture; `ctx.default_stream()` otherwise), but refuses to
    /// expose a launch handle after cleanup has been quarantined. The returned
    /// token must remain alive until the external C call returns, which makes
    /// admission linearizable with a concurrent quarantine transition. Callers
    /// must then consume it with `complete` or `quarantine`; an unclassified
    /// drop fails closed and permanently quarantines the device.
    pub fn rocm_stream_submission(&self) -> Result<RocmStreamSubmission> {
        crate::active_rocm_stream(&self.ctx)
            .execution_submission("RocmStorage::rocm_stream_submission")
            .map_err(|error| {
                Error::Msg(format!(
                    "RocmStorage::rocm_stream_submission: stream unavailable: {error}"
                ))
            })
    }

    /// Whether dropping this storage after an asynchronously launched consumer
    /// is ordered behind that consumer on `stream`.
    fn owner_release_is_ordered_on(&self, stream: &Arc<kiln_hip::RocmStream>) -> bool {
        match &self.slice {
            SliceOwner::Owned(slice) => Arc::ptr_eq(slice.stream(), stream),
            // The opaque external owner may free on another stream or by a
            // synchronous API as soon as its keep-alive Arc is dropped.
            SliceOwner::Borrowed { .. } => false,
        }
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
fn primary_rocm_context_cache() -> &'static Mutex<HashMap<usize, Arc<RocmContext>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<RocmContext>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn validate_primary_rocm_context_policy(
    device_index: usize,
    existing: RocmExecutionPolicy,
    requested: RocmExecutionPolicy,
) -> Result<()> {
    if existing == requested {
        return Ok(());
    }
    Err(Error::Msg(format!(
        "primary ROCm context for device {device_index} already uses execution policy {existing:?}; \
         requested {requested:?}. Install the policy before model or tensor initialization"
    )))
}

pub fn primary_rocm_context(device_index: usize) -> Result<Arc<RocmContext>> {
    let cache = primary_rocm_context_cache();
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

/// Install `execution_policy` while creating the primary context, or validate
/// that an already-created primary context has the same immutable policy.
///
/// Runtime configuration must call this before model or tensor initialization.
/// A conflicting late request fails instead of silently mixing synchronization
/// disciplines within one device's shared stream.
pub fn primary_rocm_context_with_execution_policy(
    device_index: usize,
    execution_policy: RocmExecutionPolicy,
) -> Result<Arc<RocmContext>> {
    let cache = primary_rocm_context_cache();
    let mut map = cache.lock().map_err(|_| {
        Error::Msg("primary_rocm_context_with_execution_policy: cache mutex poisoned".to_string())
    })?;
    if let Some(ctx) = map.get(&device_index) {
        validate_primary_rocm_context_policy(
            device_index,
            ctx.execution_policy(),
            execution_policy,
        )?;
        return Ok(Arc::clone(ctx));
    }
    let ctx =
        RocmContext::new_with_execution_policy(device_index, execution_policy).map_err(|e| {
            Error::Msg(format!(
                "primary_rocm_context_with_execution_policy({device_index}): {e}"
            ))
        })?;
    map.insert(device_index, Arc::clone(&ctx));
    Ok(ctx)
}

/// Effective immutable execution policy for the primary ROCm context.
pub fn rocm_execution_policy(device_index: usize) -> Result<RocmExecutionPolicy> {
    Ok(primary_rocm_context(device_index)?.execution_policy())
}

/// Point-in-time fixed-cardinality synchronization telemetry for a ROCm device.
pub fn rocm_sync_telemetry_snapshot(device_index: usize) -> Result<RocmSyncTelemetrySnapshot> {
    Ok(primary_rocm_context(device_index)?.sync_telemetry_snapshot())
}

/// Whether failed synchronization recovery has quarantined further execution
/// and destruction for the primary ROCm context.
pub fn rocm_cleanup_quarantined(device_index: usize) -> Result<bool> {
    Ok(primary_rocm_context(device_index)?.cleanup_quarantined())
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
pub fn rocm_zeros_ctx(
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let ctx = primary_rocm_context(device_index)?;
    let storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
}

/// Block until all work on `device_index`'s device completes. ROCm analog of
/// `cuda_synchronize_default_stream`.
pub fn rocm_synchronize_default_stream(device_index: usize) -> Result<()> {
    rocm_synchronize_device_for(device_index, RocmSyncReason::ExplicitDeviceDrain)
}

/// Always drain all work on `device_index`, accounting it under `reason`.
pub fn rocm_synchronize_device_for(device_index: usize, reason: RocmSyncReason) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    ctx.synchronize_device_for(reason).map_err(|e| {
        Error::Msg(format!(
            "rocm_synchronize_device_for({device_index}, {}): {e:?}",
            reason.as_str()
        ))
    })
}

/// Synchronize generated work before it becomes visible outside the backend.
///
/// Legacy policy drains the device. Stream-ordered policy drains the primary
/// stream, which graph replay explicitly orders after its capture stream.
pub fn rocm_synchronize_external_yield(device_index: usize) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    let producer_stream = crate::active_rocm_stream::last_rocm_producer_stream(&ctx)
        .unwrap_or_else(|| ctx.default_stream());
    let result = ctx
        .synchronize_external_yield_for(&producer_stream)
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_synchronize_external_yield({device_index}): {e:?}"
            ))
        });
    if result.is_ok() {
        crate::active_rocm_stream::clear_last_rocm_producer_stream(&ctx, &producer_stream);
    }
    result
}

/// Settle an explicitly identified ROCm producer stream before external yield.
/// This is the cross-thread embedding form; ordinary same-thread dispatch uses
/// [`rocm_synchronize_external_yield`] and its remembered producer stream.
pub fn rocm_synchronize_external_yield_for_stream(
    device_index: usize,
    producer_stream: &Arc<kiln_hip::RocmStream>,
) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    let result = ctx
        .synchronize_external_yield_for(producer_stream)
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_synchronize_external_yield_for_stream({device_index}): {e:?}"
            ))
        });
    if result.is_ok() {
        crate::active_rocm_stream::clear_last_rocm_producer_stream(&ctx, producer_stream);
    }
    result
}

/// `(reserved, used)` bytes of device `device_index`'s stream-ordered memory
/// pool — PROCESS-ISOLATED (only kiln's pool), unlike the all-process DRM
/// counters. `reserved` is kiln's VRAM high-water mark; `used` is what's live.
/// The right signal for "is freed KV being reused vs growing our footprint."
pub fn rocm_pool_stats(device_index: usize) -> Result<(u64, u64)> {
    let ctx = primary_rocm_context(device_index)?;
    ctx.pool_stats()
        .map_err(|e| Error::Msg(format!("rocm_pool_stats({device_index}): {e:?}")))
}

/// Device-reported `(free, total)` ROCm memory via `hipMemGetInfo`. This is the
/// most accurate "free VRAM on the active GPU" signal — it reports the actual
/// pool the GPU allocates from (a discrete card's VRAM, or an APU's GART/UMA
/// carveout), which on a unified APU like Strix Halo is DISTINCT from host
/// `MemAvailable` (GPU buffers come from the carveout, not system RAM). Use this
/// to back the memory governor with ground truth for the device in use.
pub fn rocm_mem_get_info(device_index: usize) -> Result<(usize, usize)> {
    let ctx = primary_rocm_context(device_index)?;
    ctx.mem_get_info()
        .map_err(|e| Error::Msg(format!("rocm_mem_get_info({device_index}): {e:?}")))
}

/// Return pooled-but-unused ROCm VRAM to the OS, keeping at least
/// `min_keep_bytes` cached. The memory-pressure reclaim hook: when the governor
/// sees a coexisting process needs VRAM, this hands kiln's freed pool blocks
/// back. Synchronizes only when pool statistics show reclaimable spare bytes;
/// returns the measured reduction in reserved bytes (see
/// [`kiln_hip::RocmContext::trim_pool`]).
pub fn rocm_trim_pool(device_index: usize, min_keep_bytes: usize) -> Result<u64> {
    let ctx = primary_rocm_context(device_index)?;
    ctx.trim_pool(min_keep_bytes)
        .map_err(|e| Error::Msg(format!("rocm_trim_pool({device_index}): {e:?}")))
}

/// Block until all work on the ACTIVE compute stream completes
/// (`hipStreamSynchronize`, not the device-wide `hipDeviceSynchronize`). Cheaper
/// than [`rocm_synchronize_default_stream`] when other (e.g. hipBLASLt-internal)
/// streams have pending work we don't need to wait on.
pub fn rocm_synchronize_compute_stream(device_index: usize) -> Result<()> {
    rocm_synchronize_compute_stream_for(device_index, RocmSyncReason::ExplicitStreamDrain)
}

/// Always drain the active compute stream, accounting it under `reason`.
pub fn rocm_synchronize_compute_stream_for(
    device_index: usize,
    reason: RocmSyncReason,
) -> Result<()> {
    // HIP rejects hipStreamSynchronize while that stream is being captured.
    // Graph capture installs the arena and routes every kt op onto one active
    // capture stream; same-stream ordering plus the graph runner's explicit
    // pre-capture syncs provide the needed handoff guarantees there.
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let ctx = primary_rocm_context(device_index)?;
    let stream = crate::active_rocm_stream(&ctx);
    ctx.synchronize_stream_for(&stream, reason).map_err(|e| {
        Error::Msg(format!(
            "rocm_synchronize_compute_stream_for({device_index}, {}): {e:?}",
            reason.as_str()
        ))
    })
}

/// Order a proven same-stream producer/consumer dependency.
///
/// Legacy policy performs the historical active-stream wait. Stream-ordered
/// policy relies on FIFO ordering and records the omitted barrier in telemetry.
pub fn rocm_synchronize_same_stream_dependency(
    device_index: usize,
    reason: RocmSyncReason,
) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    rocm_synchronize_context_same_stream_dependency(&ctx, reason)
}

pub(crate) fn rocm_synchronize_context_same_stream_dependency(
    ctx: &Arc<RocmContext>,
    reason: RocmSyncReason,
) -> Result<()> {
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let stream = crate::active_rocm_stream(&ctx);
    ctx.synchronize_same_stream_dependency(&stream, reason)
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_synchronize_same_stream_dependency({}, {}): {e:?}",
                ctx.ordinal(),
                reason.as_str()
            ))
        })
}

/// Order a post-launch boundary while also protecting the lifetime of every
/// input owner read by that launch.
///
/// Stream-ordered mode may omit the host wait only when every owned input was
/// allocated on the active stream. A borrowed owner or allocation from another
/// stream requires an active-stream wait before the caller can drop it.
pub(crate) fn rocm_synchronize_context_same_stream_dependency_with_inputs(
    ctx: &Arc<RocmContext>,
    inputs: &[&RocmStorage],
    reason: RocmSyncReason,
) -> Result<()> {
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let stream = crate::active_rocm_stream(ctx);
    let owners_are_stream_ordered = !inputs.is_empty()
        && inputs
            .iter()
            .all(|storage| storage.owner_release_is_ordered_on(&stream));
    let result = if ctx.execution_policy().synchronization_mode
        == crate::RocmSynchronizationMode::StreamOrdered
        && !owners_are_stream_ordered
    {
        ctx.synchronize_stream_for(&stream, reason)
    } else {
        ctx.synchronize_same_stream_dependency(&stream, reason)
    };
    result.map_err(|e| {
        Error::Msg(format!(
            "rocm_synchronize_same_stream_dependency_with_inputs({}, {}): {e:?}",
            ctx.ordinal(),
            reason.as_str()
        ))
    })
}

/// Preserve a historical device-wide barrier in legacy mode and omit it in
/// stream-ordered mode for a proven same-stream dependency.
pub fn rocm_synchronize_legacy_device_same_stream_dependency(
    device_index: usize,
    reason: RocmSyncReason,
) -> Result<()> {
    let ctx = primary_rocm_context(device_index)?;
    rocm_synchronize_context_legacy_device_same_stream_dependency(&ctx, reason)
}

pub(crate) fn rocm_synchronize_context_legacy_device_same_stream_dependency(
    ctx: &Arc<RocmContext>,
    reason: RocmSyncReason,
) -> Result<()> {
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let stream = crate::active_rocm_stream(ctx);
    ctx.synchronize_legacy_device_same_stream_dependency(&stream, reason)
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_synchronize_legacy_device_same_stream_dependency({}, {}): \
                 {e:?}",
                ctx.ordinal(),
                reason.as_str()
            ))
        })
}

/// Preserve a historical device-wide legacy boundary while allowing a
/// stream-ordered skip only when all asynchronously consumed owners will also
/// be released on the active stream.
pub(crate) fn rocm_synchronize_context_legacy_device_same_stream_dependency_with_inputs(
    ctx: &Arc<RocmContext>,
    inputs: &[&RocmStorage],
    reason: RocmSyncReason,
) -> Result<()> {
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let stream = crate::active_rocm_stream(ctx);
    let owners_are_stream_ordered = !inputs.is_empty()
        && inputs
            .iter()
            .all(|storage| storage.owner_release_is_ordered_on(&stream));
    let result = if ctx.execution_policy().synchronization_mode
        == crate::RocmSynchronizationMode::StreamOrdered
        && !owners_are_stream_ordered
    {
        ctx.synchronize_stream_for(&stream, reason)
    } else {
        ctx.synchronize_legacy_device_same_stream_dependency(&stream, reason)
    };
    result.map_err(|e| {
        Error::Msg(format!(
            "rocm_synchronize_legacy_device_same_stream_dependency_with_inputs({}, {}): {e:?}",
            ctx.ordinal(),
            reason.as_str()
        ))
    })
}

/// Block until the active stream for a ROCm tensor's actual storage context
/// completes. Prefer this at tensor handoff sites where the tensor may have
/// been allocated on a context already in hand, rather than re-acquiring the
/// primary context by device index.
pub fn rocm_synchronize_tensor_stream(t: &crate::Tensor) -> Result<()> {
    rocm_synchronize_tensor_stream_for(t, RocmSyncReason::TensorHandoff)
}

/// Always drain a ROCm tensor's active storage stream under `reason`.
pub fn rocm_synchronize_tensor_stream_for(t: &crate::Tensor, reason: RocmSyncReason) -> Result<()> {
    let storage = t
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_synchronize_tensor_stream: tensor must be ROCm".into()))?;
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let ctx = storage.context();
    let stream = crate::active_rocm_stream(&ctx);
    ctx.synchronize_stream_for(&stream, reason).map_err(|e| {
        Error::Msg(format!(
            "rocm_synchronize_tensor_stream_for({}, {}): {e:?}",
            t.device(),
            reason.as_str()
        ))
    })
}

/// Order a proven same-stream dependency using a tensor's storage context.
pub fn rocm_synchronize_tensor_same_stream_dependency(
    t: &crate::Tensor,
    reason: RocmSyncReason,
) -> Result<()> {
    let storage = t
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_synchronize_tensor_same_stream_dependency: tensor must be ROCm".into())
        })?;
    if crate::rocm_capture_arena_active() {
        return Ok(());
    }
    let ctx = storage.context();
    rocm_synchronize_context_same_stream_dependency_with_inputs(&ctx, &[storage], reason)
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
/// replay cost (decode dropped to ~5 tok/s). Ordering is instead provided by an
/// input-ready event recorded on the default stream and awaited by the capture
/// stream before graph launch.
///
/// Host-buffer safety: `E::to_bytes` creates an ordinary, unregistered
/// `Vec<u8>`. The generic ROCm `hipMemcpyAsync` contract states that an
/// unpinned host source is consumed synchronously before the call returns, so
/// the local staging buffer cannot become a dangling runtime read. Stream
/// ordering is still preserved without adding a trailing wait here.
#[cfg(feature = "rocm")]
pub fn rocm_write_host_in_place<E: crate::Element>(dst: &crate::Tensor, host: &[E]) -> Result<()> {
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
    // the captured graph runs on during replay. The fresh `Vec` is pageable,
    // satisfying `memcpy_htod_raw_async`'s synchronous-source contract.
    unsafe {
        stream
            .memcpy_htod_raw_async(dst_base as *mut core::ffi::c_void, &bytes)
            .map_err(|e| {
                Error::Msg(format!(
                    "rocm_write_host_in_place: memcpy_htod_raw_async: {e:?}"
                ))
            })?;
    }
    // No trailing synchronize — see the doc comment. Replay hands this write to
    // the capture stream through an explicit input-ready event.
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
    let dst_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, src.dtype(), n_elements)?;

    let (src_base, _) = src_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();

    let src_byte_off = (layout.start_offset() * bpe) as u64;
    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    if rocm_view_is_physically_compact(shape, strides_elems) {
        let copy_bytes = n_elements
            .checked_mul(bpe)
            .ok_or_else(|| Error::Msg("rocm_contiguous: copy byte count overflow".to_string()))?;
        if copy_bytes > 0 {
            let stream = crate::active_rocm_stream(&ctx);
            unsafe {
                stream
                    .memcpy_dtod_raw_async(dst_ptr, src_ptr, copy_bytes)
                    .map_err(|e| {
                        Error::Msg(format!("rocm_contiguous: dense D2D copy failed: {e:?}"))
                    })?;
            }
            rocm_synchronize_context_same_stream_dependency_with_inputs(
                &ctx,
                &[src_storage],
                RocmSyncReason::ContiguousOutput,
            )
            .map_err(|e| {
                Error::Msg(format!(
                    "rocm_contiguous: synchronize after dense D2D copy: {e:?}"
                ))
            })?;
        }
        let storage_arc: crate::Storage = Arc::new(dst_storage);
        return crate::Tensor::from_parts(
            storage_arc,
            crate::Layout::contiguous(shape.to_vec()),
            crate::TensorId::next(),
        )
        .map_err(|e| Error::Msg(format!("rocm_contiguous: wrap: {e}")));
    }

    if let Some((bsz, heads, seq, dim, stride_b, stride_h, stride_t)) =
        rocm_view_is_4d_axis12_transpose(shape, strides_elems)
    {
        let stream_submission = src_storage.rocm_stream_submission()?;
        let raw_stream = stream_submission.raw_stream();
        let status = unsafe {
            kiln_transpose_4d_12_copy_async(
                src_ptr,
                dst_ptr,
                bsz as i64,
                heads as i64,
                seq as i64,
                dim as i64,
                stride_b as i64,
                stride_h as i64,
                stride_t as i64,
                bpe as i32,
                raw_stream,
            )
        };
        if status != 0 {
            stream_submission.quarantine();
            return Err(Error::Msg(format!(
                "rocm_contiguous: kiln_transpose_4d_12_copy_async returned status {status}"
            )));
        }
        stream_submission.complete();
        rocm_synchronize_context_same_stream_dependency_with_inputs(
            &ctx,
            &[src_storage],
            RocmSyncReason::ContiguousOutput,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_contiguous: synchronize after 4d axis12 transpose copy: {e:?}"
            ))
        })?;
        let storage_arc: crate::Storage = Arc::new(dst_storage);
        return crate::Tensor::from_parts(
            storage_arc,
            crate::Layout::contiguous(shape.to_vec()),
            crate::TensorId::next(),
        )
        .map_err(|e| Error::Msg(format!("rocm_contiguous: wrap: {e}")));
    }

    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let strides_i64: Vec<i64> = strides_elems.iter().map(|&s| s as i64).collect();
    let stream_submission = src_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();

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
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_contiguous: kiln_contiguous_copy_async returned status {status}"
        )));
    }
    stream_submission.complete();
    rocm_synchronize_context_same_stream_dependency_with_inputs(
        &ctx,
        &[src_storage],
        RocmSyncReason::ContiguousOutput,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "rocm_contiguous: synchronize after strided D2D copy: {e:?}"
        ))
    })?;

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_contiguous: wrap: {e}")))
}

fn rocm_view_is_physically_compact(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }

    let mut expected = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides.iter()).rev() {
        if dim <= 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        expected = match expected.checked_mul(dim) {
            Some(v) => v,
            None => return false,
        };
    }
    true
}

fn rocm_view_is_4d_axis12_transpose(
    shape: &[usize],
    strides: &[usize],
) -> Option<(usize, usize, usize, usize, usize, usize, usize)> {
    if shape.len() != 4 || strides.len() != 4 {
        return None;
    }

    let bsz = shape[0];
    let heads = shape[1];
    let seq = shape[2];
    let dim = shape[3];
    if heads == 0 || seq == 0 || dim == 0 || strides[3] != 1 {
        return None;
    }

    let expected_head_stride = dim;
    let expected_seq_stride = heads.checked_mul(dim)?;
    if strides[1] != expected_head_stride || strides[2] != expected_seq_stride {
        return None;
    }

    Some((bsz, heads, seq, dim, strides[0], strides[1], strides[2]))
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
    if dst.dtype() != src.dtype() {
        return Err(Error::Msg(format!(
            "rocm_slice_set: dtype mismatch dst={} src={}",
            dst.dtype(),
            src.dtype()
        )));
    }
    if dst.device() != src.device() {
        return Err(Error::Msg(format!(
            "rocm_slice_set: device mismatch dst={} src={}",
            dst.device(),
            src.device()
        )));
    }
    if !dst.is_contiguous() || !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_slice_set: dst and src must be contiguous".to_string(),
        ));
    }
    let bpe = dst.dtype().size_in_bytes();
    let inner: usize = dst.dims().iter().skip(1).product();
    let src_n = src.element_count();
    if inner == 0 {
        return Err(Error::Msg(
            "rocm_slice_set: zero-size destination row".to_string(),
        ));
    }
    if src_n % inner != 0 {
        return Err(Error::Msg(format!(
            "rocm_slice_set: src element count {src_n} is not a whole number of dst rows \
             (inner={inner})"
        )));
    }
    let rows = src_n / inner;
    let dst_rows = dst.dims().first().copied().unwrap_or(0);
    if offset > dst_rows || rows > dst_rows.saturating_sub(offset) {
        return Err(Error::Msg(format!(
            "rocm_slice_set: rows [{offset}, {}) out of bounds for dst rows {dst_rows}",
            offset + rows
        )));
    }

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

    let ctx = dst_storage.context();
    let stream = crate::active_rocm_stream(&ctx);
    let stream_submission = dst_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();
    let src_byte_off = (src.layout().start_offset() * bpe) as u64;
    let dst_byte_off = ((dst.layout().start_offset() + offset * inner) * bpe) as u64;
    let src_ptr = (src_base + src_byte_off) as *const core::ffi::c_void;
    let dst_ptr = (dst_base + dst_byte_off) as *mut core::ffi::c_void;

    // Flat contiguous copy of `src_n` elements (shape [src_n], stride [1])
    // into dst at the computed byte offset. This mirrors the CUDA helper and
    // avoids gfx115x raw D2D-copy corruption seen in long-context concat.
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
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_slice_set: kiln_contiguous_copy_async returned status {status}"
        )));
    }
    stream_submission.complete();
    // `slice_set` is an in-place API whose source tensor may be dropped as soon
    // as this function returns. The ROCm copy is async and RocmSlice frees are
    // stream-ordered, so drain the copy outside graph capture before releasing
    // the caller back to Rust. This matches the API's visible mutation
    // semantics and prevents row-tiled reducers from copying from freed tiles.
    if !crate::rocm_capture_arena_active() {
        ctx.synchronize_stream_for(&stream, RocmSyncReason::InPlaceMutation)
            .map_err(|e| Error::Msg(format!("rocm_slice_set: synchronize after copy: {e:?}")))?;
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

    let dtype = src.dtype();
    let expected_byte_len = dtype.packed_buffer_bytes(src.element_count());
    if rocm_view_is_physically_compact(src.shape(), src.strides()) {
        let src_storage = src
            .storage()
            .as_any()
            .downcast_ref::<RocmStorage>()
            .ok_or_else(|| Error::Msg("rocm_to_host_copy: source must be ROCm storage".into()))?;
        let byte_offset = src
            .layout()
            .start_offset()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Msg("rocm_to_host_copy: byte offset overflow".into()))?;
        let byte_end = byte_offset
            .checked_add(expected_byte_len)
            .ok_or_else(|| Error::Msg("rocm_to_host_copy: byte range overflow".into()))?;
        let (base, storage_byte_len) = src_storage.device_ptr_raw();
        if byte_end > storage_byte_len {
            return Err(Error::Msg(format!(
                "rocm_to_host_copy: compact view byte range {byte_offset}..{byte_end} exceeds storage length {storage_byte_len}"
            )));
        }
        let src_ptr = base
            .checked_add(byte_offset as u64)
            .ok_or_else(|| Error::Msg("rocm_to_host_copy: device pointer overflow".into()))?;
        let ctx = src_storage.context();
        let stream = crate::active_rocm_stream(&ctx);
        // SAFETY: `Tensor::from_parts` validates every layout against physical
        // storage, and the checked range above proves this compact view stays
        // within that live allocation. The tensor keeps the storage alive
        // through the synchronized copy.
        let host_bytes = unsafe {
            stream.memcpy_dtoh_raw(src_ptr as *const core::ffi::c_void, expected_byte_len)
        }
        .map_err(|e| {
            Error::Msg(format!(
                "rocm_to_host_copy: direct range copy failed: {e:?}"
            ))
        })?;
        let cpu_storage = crate::CpuStorage::from_bytes(dtype, host_bytes)?;
        let storage_arc: crate::Storage = Arc::new(cpu_storage);
        return crate::Tensor::from_parts(
            storage_arc,
            crate::Layout::contiguous(src.shape().to_vec()),
            crate::TensorId::next(),
        );
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

    let ctx = contig_storage.context();
    let stream = crate::active_rocm_stream(&ctx);
    let host_bytes = stream
        .memcpy_dtoh(contig_storage.slice())
        .map_err(|e| Error::Msg(format!("rocm_to_host_copy: memcpy_dtoh failed: {e:?}")))?;

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
#[track_caller]
pub fn host_to_rocm_copy(src: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    let ctx = primary_rocm_context(device_index)?;
    host_to_rocm_copy_with_context(src, &ctx)
}

/// Copy a contiguous host tensor to a fresh buffer owned by an explicit ROCm
/// context. This is the policy-preserving upload path for isolated runtimes and
/// tests that must not mutate process-global configuration.
#[track_caller]
pub fn host_to_rocm_copy_with_context(
    src: &crate::Tensor,
    ctx: &Arc<RocmContext>,
) -> Result<crate::Tensor> {
    let device_index = ctx.ordinal();
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
        .ok_or_else(|| {
            Error::Msg("host_to_rocm_copy: contig src must be CPU storage".to_string())
        })?;
    let bytes = contig_cpu.as_bytes();
    if bytes.len() != byte_len {
        return Err(Error::Msg(format!(
            "host_to_rocm_copy: src byte_len {} != expected {}",
            bytes.len(),
            byte_len
        )));
    }

    let stream = crate::active_rocm_stream(ctx);
    let device_slice = stream
        .clone_htod(bytes)
        .map_err(|e| Error::Msg(format!("host_to_rocm_copy: clone_htod failed: {e:?}")))?;

    // Notify capture-safety observers scoped to this thread and device. Keep the
    // process-wide aggregate for compatibility with the public counter API.
    record_rocm_htod_copy(
        device_index,
        u64::try_from(byte_len).unwrap_or(u64::MAX),
        dtype,
        u64::try_from(n_elements).unwrap_or(u64::MAX),
        std::panic::Location::caller(),
    );
    ROCM_HTOD_COUNT.fetch_add(1, Ordering::Relaxed);

    // SAFETY: clone_htod synchronizes its stream before returning, so the
    // wrapped slice is fully initialized before any tensor consumer can run.
    let rocm_storage =
        unsafe { RocmStorage::from_slice_ctx(ctx, device_index, dtype, device_slice) }?;

    let storage_arc: crate::Storage = Arc::new(rocm_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Host → ROCm copy — back-compat alias for [`host_to_rocm_copy`], mirroring
/// `host_to_cuda_copy_ctx`.
#[track_caller]
pub fn host_to_rocm_copy_ctx(src: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    host_to_rocm_copy(src, device_index)
}

type RocmLastAxisNormalizationKernel = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    i64,
    i64,
    i32,
    *mut core::ffi::c_void,
) -> i32;

fn rocm_last_axis_normalization(
    x: &crate::Tensor,
    label: &str,
    kernel: RocmLastAxisNormalizationKernel,
    output_dtype: DType,
) -> Result<crate::Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!("{label}: unsupported dtype {other}")));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(format!("{label}: input must be contiguous")));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    let shape = x.shape();
    let trailing_dim = shape[rank - 1];
    if trailing_dim == 0 {
        return Err(Error::Msg(format!(
            "{label}: trailing axis must be non-empty"
        )));
    }
    let n_cols = i64::try_from(trailing_dim)
        .map_err(|_| Error::Msg(format!("{label}: trailing axis exceeds i64")))?;
    let n_rows = i64::try_from(x.element_count() / trailing_dim)
        .map_err(|_| Error::Msg(format!("{label}: row count exceeds i64")))?;
    if n_rows > i64::from(i32::MAX) {
        return Err(Error::Msg(format!(
            "{label}: row count {n_rows} exceeds kernel grid limit {}",
            i32::MAX
        )));
    }

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be ROCm")))?;
    let ctx = x_storage.context();
    let device_index = match x_storage.device {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };
    // Normalization kernels write every output element, so skip the zero-fill.
    let out_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, output_dtype, x.element_count())?;

    let stream_submission = x_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe { kernel(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
    )
}

/// Softmax over the last axis of a contiguous ROCm tensor. ROCm analog of
/// `cuda_softmax_last_axis`, routing through the wave-size-fixed `softmax.cu`
/// kernel (Phase R.5). F32 / BF16 / F16; F32 accumulation throughout.
pub fn rocm_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    rocm_last_axis_normalization(
        x,
        "rocm_softmax_last_axis",
        kiln_softmax_last_axis_async,
        x.dtype(),
    )
}

/// Numerically stable ROCm log-softmax over the trailing axis.
///
/// The fused kernel forms `x - max(x) - log(sum(exp(x - max(x))))`
/// directly in F32 arithmetic and performs one output allocation. It never
/// materializes probabilities whose underflow could corrupt a representable
/// log-probability before `log` is applied.
pub fn rocm_log_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    rocm_last_axis_normalization(
        x,
        "rocm_log_softmax_last_axis",
        kiln_log_softmax_last_axis_async,
        x.dtype(),
    )
}

/// Numerically stable ROCm log-softmax with direct F32 output.
///
/// The fused kernel reads F32/BF16/F16 input in place, accumulates in F32,
/// and writes one freshly allocated F32 output. It does not allocate a casted
/// input or round the result through the input dtype.
pub fn rocm_log_softmax_last_axis_f32(x: &crate::Tensor) -> Result<crate::Tensor> {
    rocm_last_axis_normalization(
        x,
        "rocm_log_softmax_last_axis_f32",
        kiln_log_softmax_last_axis_f32_async,
        DType::F32,
    )
}

#[cfg(test)]
mod execution_policy_tests {
    use super::*;
    use kiln_hip::RocmSynchronizationMode;

    #[test]
    fn primary_context_policy_rejects_late_mismatch() {
        let legacy = RocmExecutionPolicy::default();
        let stream_ordered = RocmExecutionPolicy::new(RocmSynchronizationMode::StreamOrdered);

        validate_primary_rocm_context_policy(0, legacy, legacy).expect("matching policy");
        let error = validate_primary_rocm_context_policy(0, legacy, stream_ordered)
            .expect_err("conflicting policy must fail");
        assert!(
            error
                .to_string()
                .contains("before model or tensor initialization")
        );
    }

    #[test]
    fn htod_observer_is_scoped_to_thread_and_device() {
        let ((), count) = with_rocm_htod_observer(3, || {
            record_rocm_htod(4);
            std::thread::spawn(|| record_rocm_htod(3))
                .join()
                .expect("unrelated observer thread");
            record_rocm_htod(3);
        });

        assert_eq!(count, 1);
    }

    #[test]
    fn nested_htod_observers_count_matching_dynamic_scopes() {
        let ((inner_same, inner_other), outer) = with_rocm_htod_observer(0, || {
            record_rocm_htod(0);
            let ((), inner_same) = with_rocm_htod_observer(0, || record_rocm_htod(0));
            let ((), inner_other) = with_rocm_htod_observer(1, || {
                record_rocm_htod(0);
                record_rocm_htod(1);
            });
            (inner_same, inner_other)
        });

        assert_eq!(outer, 3);
        assert_eq!(inner_same, 1);
        assert_eq!(inner_other, 1);
    }

    #[test]
    fn detailed_htod_observer_aggregates_copy_sites_and_bytes() {
        let source = std::panic::Location::caller();
        let ((), observation) = with_rocm_htod_observer_detailed(2, || {
            record_rocm_htod_copy(2, 16, DType::F32, 4, source);
            record_rocm_htod_copy(2, 16, DType::F32, 4, source);
            record_rocm_htod_copy(3, 64, DType::BF16, 32, source);
        });

        assert_eq!(observation.copy_count, 2);
        assert_eq!(observation.total_bytes, 32);
        assert_eq!(observation.unattributed_copy_count, 0);
        assert_eq!(observation.unattributed_bytes, 0);
        assert_eq!(observation.sites.len(), 1);
        let site = observation.sites[0];
        assert_eq!(site.source_file, source.file());
        assert_eq!(site.source_line, source.line());
        assert_eq!(site.source_column, source.column());
        assert_eq!(site.dtype, DType::F32);
        assert_eq!(site.elements_per_copy, 4);
        assert_eq!(site.bytes_per_copy, 16);
        assert_eq!(site.copy_count, 2);
        assert_eq!(site.total_bytes, 32);
    }

    #[test]
    fn detailed_htod_observer_bounds_unique_sites_and_accounts_for_overflow() {
        let source = std::panic::Location::caller();
        let extra_sites = 5_u64;
        let ((), observation) = with_rocm_htod_observer_detailed(2, || {
            for element_count in 1..=(ROCM_HTOD_MAX_UNIQUE_SITES as u64 + extra_sites) {
                record_rocm_htod_copy(
                    2,
                    element_count * DType::F32.size_in_bytes() as u64,
                    DType::F32,
                    element_count,
                    source,
                );
            }
        });

        assert_eq!(
            observation.copy_count,
            ROCM_HTOD_MAX_UNIQUE_SITES as u64 + extra_sites
        );
        assert_eq!(observation.sites.len(), ROCM_HTOD_MAX_UNIQUE_SITES);
        assert_eq!(observation.unattributed_copy_count, extra_sites);
        let first_omitted = ROCM_HTOD_MAX_UNIQUE_SITES as u64 + 1;
        let last_omitted = ROCM_HTOD_MAX_UNIQUE_SITES as u64 + extra_sites;
        let omitted_elements = (first_omitted + last_omitted) * extra_sites / 2;
        assert_eq!(
            observation.unattributed_bytes,
            omitted_elements * DType::F32.size_in_bytes() as u64
        );
        assert_eq!(
            observation.total_bytes,
            (1..=last_omitted)
                .map(|elements| elements * DType::F32.size_in_bytes() as u64)
                .sum::<u64>()
        );
    }

    #[test]
    fn htod_observer_guard_cleans_up_after_panic() {
        let panic = std::panic::catch_unwind(|| {
            let _ = with_rocm_htod_observer(7, || {
                record_rocm_htod(7);
                panic!("observer cleanup probe");
            });
        });
        assert!(panic.is_err());

        record_rocm_htod(7);
        let ((), count) = with_rocm_htod_observer(7, || record_rocm_htod(7));
        assert_eq!(count, 1);
    }
}
