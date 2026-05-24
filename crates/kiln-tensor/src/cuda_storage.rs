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

use candle_core::cuda_backend::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::cuda_backend::cudarc::driver::sys::CUdeviceptr;

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
            Self::Owned(s) => f
                .debug_struct("Owned")
                .field("len", &s.len())
                .finish(),
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
/// The handed-down `CudaSlice<u8>` is allocated via candle's
/// `CudaDevice` accessor today; Phase 7 swaps that for a direct cudarc
/// `CudaContext::default_stream().alloc_zeros::<u8>` once the candle
/// dep is gone.
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
    /// Candle CUDA device handle. Held for stream affinity (Phase 1.x
    /// `StreamPlanner` reads it) and for the in-flight kernel-crate
    /// FFI calls that take `&CudaDevice` as their first argument.
    candle_device: Arc<CudaDevice>,
}

impl CudaStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on
    /// `candle_device`. Buffer is zero-initialized via candle's
    /// `alloc_zeros::<u8>(n)`.
    ///
    /// `device_index` is the CUDA device index — must match the index
    /// of the candle device's owning context. Stored as the
    /// [`Device::Cuda`] variant.
    pub fn zeros(
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let slice = candle_device
            .alloc_zeros::<u8>(byte_len)
            .map_err(|e| {
                Error::Msg(format!("CudaStorage::zeros: alloc_zeros<u8>({byte_len}) failed: {e:?}"))
            })?;
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice: SliceOwner::Owned(slice),
            candle_device,
        })
    }

    /// Wrap an existing `CudaSlice<u8>` allocated by the caller.
    ///
    /// Validates the slice length against
    /// `dtype.size_in_bytes()` for non-packed dtypes (must be a
    /// multiple); packed dtypes have no per-element alignment.
    pub fn from_slice(
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        dtype: DType,
        slice: CudaSlice<u8>,
    ) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !slice.len().is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "CudaStorage::from_slice: slice len {} is not a multiple of \
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
            candle_device,
        })
    }

    /// Wrap an externally-owned CUDA buffer as a kt `CudaStorage`
    /// without copying.
    ///
    /// `keep_alive` is an opaque Arc that must outlive every read
    /// from `device_ptr`. Typical pattern: pass an Arc-wrapped candle
    /// `Storage::Cuda(...)` so the candle Tensor's underlying
    /// `CudaSlice<T>` drop runs after this storage's last reference.
    ///
    /// `device_ptr` + `byte_len` describe the borrowed region. The
    /// caller is responsible for the byte_len matching dtype × element
    /// count (this constructor does the same alignment check as
    /// [`Self::from_slice`]).
    ///
    /// The Phase 7 zero-copy candle→kt adapter is the canonical
    /// caller. Kernel-crate kt-API sites that reach `.slice()` will
    /// panic on a borrowed storage — they must migrate to the
    /// dtype/owner-aware accessor that lands alongside the adapter.
    pub fn from_borrowed(
        candle_device: Arc<CudaDevice>,
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
                    "CudaStorage::from_borrowed: byte_len {byte_len} is not a multiple of \
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
            candle_device,
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
                use candle_core::cuda_backend::cudarc::driver::DevicePtr;
                // Use a default-stream device_ptr just to extract the raw bits;
                // the SyncOnDrop is dropped immediately, recording nothing.
                let stream = self.candle_device.cuda_stream();
                let (ptr, _g) = s.device_ptr(&stream);
                (ptr, s.len())
            }
            SliceOwner::Borrowed { ptr, byte_len, .. } => (*ptr, *byte_len),
        }
    }

    /// The candle CUDA device handle this storage was allocated on.
    /// Used by FFI sites + the Phase 1.x `StreamPlanner` to read
    /// stream affinity.
    pub fn candle_device(&self) -> &Arc<CudaDevice> {
        &self.candle_device
    }
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
pub fn cuda_zeros(
    candle_device: Arc<CudaDevice>,
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let storage = CudaStorage::zeros(candle_device, device_index, dtype, n_elements)?;
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

    fn kiln_elementwise_binary_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
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
/// Errors:
/// - Source must be CUDA storage (downcast to `CudaStorage`).
/// - Packed dtypes are not supported (Marlin / Int4Packed / Fp4Packed
///   have no element-wise interpretation).
/// - Rank must be ≤ 8 (matches the kernel's `MAX_RANK`).
#[cfg(feature = "cuda")]
pub fn cuda_contiguous(src: &crate::Tensor) -> Result<crate::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let candle_device = src_storage.candle_device.clone();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };
    let dst_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        src.dtype(),
        n_elements,
    )?;

    // Extract raw device pointers. Source base + start_offset; dst
    // base.
    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
            crate::Error::Msg(
                "cuda_index_select_dim0: indices must be CUDA storage".to_string(),
            )
        })?;

    let candle_device = src_storage.candle_device.clone();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };
    let dst_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        src.dtype(),
        n_out_elements,
    )?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = a_storage.candle_device.clone();
    let device_index = match a_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        dtype,
        n,
    )?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let a_off = (a.layout().start_offset() * bpe) as u64;
    let b_off = (b.layout().start_offset() * bpe) as u64;

    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_elementwise_binary_async(
            a_ptr,
            b_ptr,
            out_ptr,
            n as i64,
            kind,
            dtype_tag,
            raw_stream,
        )
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

// ----------------------------------------------------------------------
// Tests are GPU-only — gated by KILN_TENSOR_CUDA_TEST=1 so a host with
// cudarc + candle's cuda feature compiled in but no actual GPU doesn't
// spuriously fail.
// ----------------------------------------------------------------------

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
    fn zeros_round_sizes() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let storage = CudaStorage::zeros(dev.clone(), 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Cuda(0));
        assert_eq!(storage.dtype(), DType::BF16);
        assert_eq!(storage.byte_len(), 128);

        let storage = CudaStorage::zeros(dev, 0, DType::Int4Packed, 16).unwrap();
        assert_eq!(storage.byte_len(), 8); // 16 elements packed -> 8 bytes
    }

    #[test]
    fn from_slice_validates_alignment() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let slice = dev.alloc_zeros::<u8>(17).unwrap();
        let err = CudaStorage::from_slice(dev.clone(), 0, DType::F32, slice).unwrap_err();
        assert!(err.to_string().contains("not a multiple"));
    }

    #[test]
    fn cuda_zeros_returns_arc_storage() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let s: crate::Storage = cuda_zeros(dev, 0, DType::F32, 4).unwrap();
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.byte_len(), 16);
        assert_eq!(s.device(), Device::Cuda(0));
        // Downcast to ensure the concrete type is CudaStorage.
        let cuda = s.as_any().downcast_ref::<CudaStorage>().expect("downcast");
        assert_eq!(cuda.slice().len(), 16);
    }
}
