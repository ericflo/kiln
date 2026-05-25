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

    fn kiln_sum_squared_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
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

/// CUDA-side unary activation: `out[i] = activation(x[i])`.
///
/// `kind` encodes the op (0=Silu, 1=Sigmoid, 2=Gelu, 3=Tanh, 4=Relu;
/// 5=Log, 6=Exp, 7=Sin, 8=Cos, 9=Tan, 10=Sinh, 11=Cosh, 12=Neg, 13=Abs, 14=Sqrt;
/// 15=Log2, 16=Log10, 17=Log1p, 18=Asin, 19=Acos, 20=Atan, 21=Atanh).
/// Dtype inferred from `x.dtype()`; must be F32/BF16/F16. Input must
/// be contiguous and on CUDA.
#[cfg(feature = "cuda")]
pub fn cuda_activation_unary(x: &crate::Tensor, kind: i32) -> Result<crate::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_activation_unary_async(
            x_ptr,
            out_ptr,
            n as i64,
            kind,
            dtype_tag,
            raw_stream,
        )
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = src_storage.candle_device.clone();
    let device_index = match src_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let dst_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        target,
        n,
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
    let dst_base = match &dst_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let src_off = (src.layout().start_offset() * from_bpe) as u64;
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_cast_async(src_ptr, dst_ptr, n as i64, cast_tag, raw_stream)
    };
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = out_storage.candle_device.clone();
    let stream = candle_device.cuda_stream();
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        crate::DType::F32 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F16 => 2,
        other => {
            return Err(crate::Error::Msg(format!(
                "cuda_softmax_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_softmax_last_axis: input must be contiguous".to_string(),
        ));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(crate::Error::Msg(
            "cuda_softmax_last_axis: input must have rank ≥ 1".to_string(),
        ));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg("cuda_softmax_last_axis: input must be CUDA".to_string()),
    )?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, x.element_count())?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_softmax_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_softmax_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape.to_vec()),
        crate::TensorId::next(),
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

    let candle_device = contig_storage.candle_device().clone();
    let dtype = src.dtype();
    let n_elements = src.element_count();
    let byte_len = dtype.packed_buffer_bytes(n_elements);

    // Issue the D2H memcpy through candle's stream wrapper. The
    // stream's `memcpy_dtoh` synchronizes on completion (per cudarc
    // 0.19's CudaStream semantics).
    let slice = contig_storage.slice();
    let mut host_bytes = vec![0u8; byte_len];
    let stream = candle_device.cuda_stream();
    stream
        .memcpy_dtoh(slice, &mut host_bytes)
        .map_err(|e| {
            crate::Error::Msg(format!("cuda_to_host_copy: memcpy_dtoh failed: {e:?}"))
        })?;

    let cpu_storage = crate::CpuStorage::from_bytes(dtype, host_bytes)?;
    let storage_arc: crate::Storage = Arc::new(cpu_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
}


/// Host → CUDA copy: copy a CPU-backed kt-Tensor's bytes onto a
/// CUDA device, returning a new CUDA-backed kt-Tensor.
///
/// Phase 1 substrate op — the sibling of [`cuda_to_host_copy`].
/// Together they close the host↔device round-trip surface.
///
/// `candle_device` and `device_index` together identify the
/// destination CUDA device. The output layout is row-major
/// contiguous (`start_offset = 0`); non-contiguous inputs are
/// silently contiguified into the destination.
///
/// Errors:
/// - Source must be CPU storage.
/// - Packed dtypes (Marlin / Int4 / Fp4) are not supported.
#[cfg(feature = "cuda")]
pub fn host_to_cuda_copy(
    src: &crate::Tensor,
    candle_device: Arc<CudaDevice>,
    device_index: usize,
) -> Result<crate::Tensor> {
    if src.dtype().is_packed() {
        return Err(crate::Error::Msg(format!(
            "host_to_cuda_copy: packed dtype {} not supported",
            src.dtype()
        )));
    }
    let cpu_storage = src
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

    // Allocate the device buffer + issue H2D memcpy. We use
    // candle's `clone_htod` which wraps cudarc's H2D for a slice;
    // it returns a fresh CudaSlice<u8> ready to wrap.
    let device_slice = {
        let stream = candle_device.cuda_stream();
        stream
            .clone_htod(bytes)
            .map_err(|e| {
                crate::Error::Msg(format!("host_to_cuda_copy: clone_htod failed: {e:?}"))
            })?
    };
    let cuda_storage =
        CudaStorage::from_slice(candle_device, device_index, dtype, device_slice)?;

    let storage_arc: crate::Storage = Arc::new(cuda_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg("cuda_sum_squared_last_axis: input must be CUDA".to_string()),
    )?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output is always F32.
    let out_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        crate::DType::F32,
        n_rows as usize,
    )?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg("cuda_l2norm_last_axis: input must be CUDA".to_string()),
    )?;
    let sum_sq_storage = sum_sq
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg(
                "cuda_l2norm_last_axis: sum_sq must be CUDA (internal invariant)".to_string(),
            )
        })?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, x.element_count())?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let sum_sq_off = (sum_sq.layout().start_offset() * crate::DType::F32.size_in_bytes()) as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let sum_sq_ptr = (sum_sq_base + sum_sq_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_l2norm_apply_async(
            x_ptr,
            sum_sq_ptr,
            out_ptr,
            n_rows,
            n_cols,
            eps,
            dtype_tag,
            raw_stream,
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg("cuda_rmsnorm_last_axis: x must be CUDA".to_string()),
    )?;
    let weight_storage = weight
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_rmsnorm_last_axis: weight must be CUDA".to_string())
        })?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, x.element_count())?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let w_off = (weight.layout().start_offset() * dtype.size_in_bytes()) as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let weight_ptr = (weight_base + w_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_rmsnorm_last_axis_async(
            x_ptr,
            weight_ptr,
            out_ptr,
            n_rows,
            n_cols,
            eps,
            dtype_tag,
            raw_stream,
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg("cuda_layernorm_last_axis: x must be CUDA".to_string()),
    )?;
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
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, x.element_count())?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
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
            x_ptr,
            weight_ptr,
            bias_ptr,
            out_ptr,
            n_rows,
            n_cols,
            eps,
            dtype_tag,
            raw_stream,
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros(candle_device.clone(), device_index, dtype, n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    // mask dtype is U8 → bpe = 1.
    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let mask_off = mask.layout().start_offset() as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let mask_ptr = (mask_base + mask_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_masked_fill_u8_async(
            x_ptr,
            mask_ptr,
            out_ptr,
            n as i64,
            fill_value,
            dtype_tag,
            raw_stream,
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg("cuda_argmax_last_axis: input must be CUDA".to_string()),
    )?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    // Output: shape = leading axes, dtype = I64.
    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let out_elem_count: usize = out_shape.iter().product::<usize>().max(1);
    let out_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        crate::DType::I64,
        out_elem_count,
    )?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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
            crate::Error::Msg(
                "cuda_cross_entropy_loss: targets must be CUDA storage".to_string(),
            )
        })?;

    let candle_device = logits_storage.candle_device.clone();
    let device_index = match logits_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    // Allocate per-row F32 scratch and a 4-byte error flag. Both
    // start zeroed.
    let row_loss_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        crate::DType::F32,
        batch,
    )?;
    // For row_err we use a 1-element U32 buffer (4 bytes, zero-init).
    let row_err_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        crate::DType::U32,
        1,
    )?;

    // Scalar output buffer (1 element at the input dtype). Reusing
    // `CudaStorage::zeros` to get a zero-initialized buffer; the
    // kernel overwrites it.
    let out_storage = CudaStorage::zeros(candle_device.clone(), device_index, dtype, 1)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };
    let row_err_base = match &row_err_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };
    let out_base = match &out_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
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
fn cuda_reduce_last_axis_impl(x: &crate::Tensor, divisor: f32, label: &str) -> Result<crate::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage = x.storage().as_any().downcast_ref::<CudaStorage>().ok_or_else(
        || crate::Error::Msg(format!("{label}: input must be CUDA")),
    )?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output: same dtype as input, shape = leading axes.
    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let out_elem_count: usize = out_shape.iter().product::<usize>().max(1);
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, out_elem_count)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, divisor, dtype_tag, raw_stream)
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let x_storage =
        x.storage()
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| {
                crate::Error::Msg(format!("{label}: input must be CUDA"))
            })?;
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output: same dtype, shape = input shape with `axis` removed.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, out_elem_count)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_arbitrary_axis_async(
            x_ptr,
            out_ptr,
            outer,
            axis_dim,
            inner,
            divisor,
            dtype_tag,
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

/// `out = all(mask, axis=A)` on CUDA — U8 mask in, U8 result of shape
/// `mask.shape` with axis `A` removed.
/// Issue #1082.
#[cfg(feature = "cuda")]
pub fn cuda_bool_reduce_axis(
    mask: &crate::Tensor,
    axis: usize,
    kind: u8, // 0 = ALL, 1 = ANY
) -> Result<crate::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        crate::DType::U8,
        out_elem_count,
    )?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let candle_device = first_storage.candle_device.clone();
    let device_index = match first_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!("CudaStorage::device is always Cuda"),
    };

    // Allocate destination — same device, same dtype, total elements.
    let n_out_elements: usize = out_shape.iter().product();
    let dst_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, n_out_elements)?;

    // Collect per-input source pointers (base + start_offset bytes).
    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let leading: usize = x.shape()[..x.rank() - 2]
        .iter()
        .product::<usize>()
        .max(1);
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

    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros(candle_device.clone(), device_index, x_dtype, n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

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

    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };

    let y_storage = CudaStorage::zeros(candle_device.clone(), device_index, dtype, n)?;
    let mask_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, crate::DType::U8, n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };
    let mask_base = match &mask_storage.slice {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let y_ptr = y_base as *mut core::ffi::c_void;
    let mask_ptr = mask_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_dropout_async(
            x_ptr,
            y_ptr,
            mask_ptr,
            n as i64,
            p,
            inv_keep,
            seed,
            dtype_tag,
            raw_stream,
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_scalar_op_async(
            x_ptr,
            out_ptr,
            n as i64,
            kind,
            dtype_tag,
            c,
            raw_stream,
        )
    };
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
pub fn cuda_clamp_pow(
    x: &crate::Tensor,
    kind: i32,
    a: f32,
    b: f32,
) -> Result<crate::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage = CudaStorage::zeros(candle_device.clone(), device_index, dtype, n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_clamp_pow_async(
            x_ptr,
            out_ptr,
            n as i64,
            kind,
            a,
            b,
            dtype_tag,
            raw_stream,
        )
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
pub fn cuda_compare(
    a: &crate::Tensor,
    b: &crate::Tensor,
    kind: i32,
) -> Result<crate::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = a_storage.candle_device.clone();
    let device_index = match a_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Output is U8 (one byte per element).
    let out_storage = CudaStorage::zeros(
        candle_device.clone(),
        device_index,
        crate::DType::U8,
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
        kiln_compare_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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

    let candle_device = t_storage.candle_device.clone();
    let device_index = match t_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
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
        kiln_where_select_async(mask_ptr, t_ptr, f_ptr, out_ptr, n as i64, dtype_tag, raw_stream)
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let candle_device = x_storage.candle_device.clone();
    let device_index = match x_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_diagonal_extract_async(x_ptr, out_ptr, n as i64, dtype_tag, raw_stream)
    };
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
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
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
    let candle_device = v_storage.candle_device.clone();
    let device_index = match v_storage.device {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    // Pre-zero the output; the kernel only writes the n diagonal entries.
    let out_storage =
        CudaStorage::zeros(candle_device.clone(), device_index, dtype, n * n)?;

    let stream = candle_device.cuda_stream();
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
        SliceOwner::Borrowed { .. } => unreachable!("cuda_zeros produces Owned"),
    };

    let v_off = (v.layout().start_offset() * bpe) as u64;
    let v_ptr = (v_base + v_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_diag_build_async(v_ptr, out_ptr, n as i64, dtype_tag, raw_stream)
    };
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
