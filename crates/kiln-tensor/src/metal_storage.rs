//! Metal storage impl behind the `metal` feature flag.
//!
//! Wraps `Arc<metal::Buffer>` (the actual buffer) + dtype +
//! `Arc<candle_core::metal_backend::MetalDevice>` for command-queue
//! affinity. The `metal` crate (Apple's MTLBuffer binding) is reached
//! through candle's re-export; Phase 7 of #1082 (candle removal)
//! replaces `MetalDevice` with a direct `MTLDevice` + command-queue
//! handle pair.
//!
//! # Anti-pattern 1 compliance
//!
//! Per the issue:
//!
//! > `kiln-tensor` is not a candle wrapper. Storage is
//! > `metal::Buffer` directly.
//!
//! `MetalStorage` does not hold a `candle_core::Tensor`. The buffer is
//! `Arc<metal::Buffer>` we own (allocated via candle's
//! `MetalDevice::allocate_zeros` — the same allocator path used by
//! candle internally; the `Arc<Buffer>` is then fully ours).
//!
//! # Apple Silicon UMA invariant
//!
//! Per the issue's Phase 1 bullet:
//!
//! > **Apple Silicon UMA zero-copy invariant**: on M-series, CPU and
//! > GPU share physical memory; `MTLStorageModeShared` buffers are
//! > addressable from both. kiln-tensor exposes `Tensor::is_unified_memory()`
//! > and `Tensor::as_host_slice()` (zero-copy on UMA, errors elsewhere)
//! > so the safetensors loader and the optimizer don't pay a copy
//! > round-trip on Mac. Discrete-GPU Macs (Pro/Studio with M-Ultra)
//! > are still UMA — no host pinning needed.
//!
//! Candle's `allocate_zeros` returns a `Shared`-mode buffer (the
//! `RESOURCE_OPTIONS` constant in `vendor/candle-core/src/metal_backend`).
//! `MetalStorage::is_unified_memory()` returns true; the zero-copy
//! host accessor lands in a follow-up PR (it needs a stride/layout
//! check that this PR keeps off the critical path).

use std::any::Any;
use std::sync::Arc;

use candle_core::metal_backend::MetalDevice;
// `candle_metal_kernels` is its own crate — candle-core does NOT
// re-export it under `metal_backend`. Depend on it directly under the
// `metal` feature so this path resolves.
use candle_metal_kernels::metal::Buffer as MetalBuffer;
use candle_metal_kernels::metal::Device as MetalRawDevice;

use crate::{DType, Device, Error, Result, StorageBackend};

/// Metal-backed storage. Byte-typed; dtype carried alongside for dispatch.
///
/// Holds an `Arc<metal::Buffer>` directly (anti-pattern 1). The
/// candle `MetalDevice` is held for command-queue affinity and for
/// the `allocate_zeros` / `new_buffer` accessors that the existing
/// kernel paths in `kiln-model::backend::metal` already use.
#[derive(Debug)]
pub struct MetalStorage {
    device: Device,
    dtype: DType,
    buffer: Arc<MetalBuffer>,
    candle_device: Arc<MetalDevice>,
}

impl MetalStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on
    /// `candle_device`. Zero-initialized via candle's blit-encoder
    /// fill (the same path used by `MetalDevice::allocate_zeros`).
    ///
    /// `device_index` is the Metal device index (always 0 on Apple
    /// Silicon today; Multi-GPU Macs would use 1+).
    pub fn zeros(
        candle_device: Arc<MetalDevice>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let buffer = candle_device.allocate_zeros(byte_len).map_err(|e| {
            Error::Msg(format!(
                "MetalStorage::zeros: allocate_zeros({byte_len}) failed: {e:?}"
            ))
        })?;
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer,
            candle_device,
        })
    }

    /// Allocate `n_elements` worth of bytes for `dtype` on the metal-rs
    /// `device`, **candle-free** in the allocation path.
    ///
    /// Buffer is allocated through
    /// `device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)`
    /// (Apple Silicon UMA — host and GPU share physical memory). Zero
    /// initialization happens via a direct `core::ptr::write_bytes` on
    /// the buffer's UMA `.contents()` pointer; no blit-command-encoder
    /// is required because Shared-mode buffers are CPU-addressable.
    ///
    /// `device_index` is the Metal device ordinal — must match the
    /// ordinal of `device`'s owning system device. Stored as the
    /// [`Device::Metal`] variant.
    ///
    /// # Why the storage's `candle_device` field is still populated
    ///
    /// The internal `candle_device: Arc<MetalDevice>` field is still
    /// load-bearing for downstream callers: every kernel-crate FFI
    /// site in `kiln-model::backend::metal` clones it as
    /// `(*candle_device_arc).clone()` to plumb a `MetalDevice` by
    /// value into `candle_metal_kernels::*` calls, and the candle
    /// `MetalDevice` wrapper caches compute pipelines + holds the
    /// shared command-queue that production code re-uses. Until every
    /// caller migrates to a raw-metal-rs FFI surface (`MTLDevice` +
    /// `MTLCommandQueue` directly), this constructor derives the
    /// candle wrapper from [`primary_metal_device`] (which calls
    /// `candle_core::Device::new_metal(device_index)`) so the
    /// resulting `MetalStorage` is drop-in-compatible with every
    /// existing downstream call site.
    ///
    /// # Device-affinity contract
    ///
    /// Both `device` (the metal-rs handle the caller passes) and the
    /// candle `MetalDevice` returned by `primary_metal_device(device_index)`
    /// wrap the same `MTLDevice` protocol object for the given
    /// ordinal — candle's `MetalDevice::new` (via `Device::all()`)
    /// resolves the same registry-ID-indexed physical GPU that
    /// metal-rs's `Device::system_default()` / `Device::all()` returns.
    /// The new buffer is therefore addressable by every kernel-crate
    /// FFI that consumes `candle_device.metal_device()`.
    ///
    /// # UMA + zero-init safety
    ///
    /// Apple Silicon UMA guarantees that an `MTLStorageModeShared`
    /// buffer's `contents()` pointer is CPU-addressable and points at
    /// the same physical bytes the GPU sees. `core::ptr::write_bytes`
    /// (memset) on that pointer with value 0 is well-defined for any
    /// `byte_len` allocation — there is no `MTLBuffer didModifyRange`
    /// requirement on Shared mode (unlike Managed mode on Intel
    /// Macs). For `byte_len == 0`, we explicitly skip the alloc + fill
    /// and synthesize a 1-byte placeholder buffer to match candle's
    /// `allocate_zeros(0)` semantics (which goes through
    /// `buf_size(0) = 1.next_power_of_two() = 1`).
    ///
    /// # Future direction
    ///
    /// Once every kernel-crate FFI site migrates to a raw-metal-rs
    /// `MTLDevice` + `MTLCommandQueue` surface (out of scope here),
    /// the storage's `candle_device` field can be dropped entirely
    /// and this constructor stops needing to call
    /// [`primary_metal_device`]. At that point `Self::zeros` is
    /// folded into this entry. See the order-of-operations doc in
    /// `metal_allocator.rs` lines 56-78.
    ///
    /// Mirror of [`crate::CudaStorage::zeros_ctx`] (commit d3caf46b) —
    /// same shape, same rationale (the parallel-constructor step of
    /// the CP-1/CP-2 substrate lift documented in
    /// `docs/issue-1082-tier-4-5-roadmap-2026-05-27.md`).
    pub fn zeros_kt(
        device: &MetalRawDevice,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        use candle_metal_kernels::metal::MTLResourceOptions;

        let byte_len = dtype.packed_buffer_bytes(n_elements);
        // Candle-free allocation through metal-rs. Apple's MTLDevice
        // rejects newBufferWithLength:options: for length=0 (returns
        // nil), so round up to 1 byte to match candle's buf_size(0) = 1
        // behavior. The dtype-derived byte_len on the StorageBackend
        // side still reads from buffer.length(), so the 0-len case is
        // self-consistent (the byte_len reported by the StorageBackend
        // will be 1, matching what candle's zeros() returns today).
        let alloc_len = byte_len.max(1);
        let buffer = device
            .new_buffer(alloc_len, MTLResourceOptions::StorageModeShared)
            .map_err(|e| {
                Error::Msg(format!(
                    "MetalStorage::zeros_kt: device.new_buffer({alloc_len}, Shared) \
                     failed: {e:?}"
                ))
            })?;
        // Zero-fill via UMA contents pointer — no command-queue
        // required on Shared-mode buffers.
        //
        // SAFETY: `buffer.contents()` returns a non-null `*mut u8` for
        // Shared-mode buffers on Apple Silicon UMA. `alloc_len` is the
        // exact length passed to `newBufferWithLength:options:`, so the
        // write_bytes call stays within the buffer's allocation. The
        // buffer is single-owner (just freshly allocated, no Arc clone
        // outstanding yet) so there are no aliasing concerns.
        unsafe {
            core::ptr::write_bytes(buffer.contents(), 0u8, alloc_len);
        }
        // Derive the candle device wrapper for the back-compat field.
        // This is the one residual candle dependency in this path; it
        // goes away in the future `candle_device` field-removal step.
        let candle_device = primary_metal_device(device_index)?;
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer: Arc::new(buffer),
            candle_device,
        })
    }

    /// Wrap an existing `Arc<metal::Buffer>` allocated by the caller.
    ///
    /// Validates the buffer length against `dtype.size_in_bytes()`
    /// for non-packed dtypes.
    pub fn from_buffer(
        candle_device: Arc<MetalDevice>,
        device_index: usize,
        dtype: DType,
        buffer: Arc<MetalBuffer>,
    ) -> Result<Self> {
        let len = buffer.length() as usize;
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "MetalStorage::from_buffer: buffer len {len} is not a multiple of \
                     size_in_bytes({:?}) = {per}",
                    dtype
                )));
            }
        }
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer,
            candle_device,
        })
    }

    /// Borrow the underlying buffer. The existing kernel-crate FFI
    /// sites in `kiln-model::backend::metal` plug in via this
    /// accessor (mirrors `candle_core::metal_backend::buffer_o` 232
    /// call sites from Phase 0.1's audit).
    pub fn buffer(&self) -> &Arc<MetalBuffer> {
        &self.buffer
    }

    /// Borrow the candle Metal device — same handle the existing
    /// kernels in `kiln-model::backend::metal` consume.
    pub fn candle_device(&self) -> &Arc<MetalDevice> {
        &self.candle_device
    }

    /// The underlying metal-rs `Device` this storage was allocated
    /// on — **candle-free passthrough**.
    ///
    /// Returns an owned [`candle_metal_kernels::metal::Device`] (cheap
    /// `Retained<ProtocolObject<dyn MTLDevice>>` clone via NSObject
    /// `retain`). The returned device wraps the SAME `MTLDevice`
    /// protocol object that candle's `MetalDevice` holds internally —
    /// candle's `MetalDevice::metal_device()` returns `&Device` from
    /// the same `candle_metal_kernels::metal` crate that this re-export
    /// uses, so the wire type is identical.
    ///
    /// This is the substrate-side accessor that unblocks the #1082
    /// Phase 7 CP-2 migration of `MetalAllocator` (and other
    /// downstream callers) to hold `metal-rs::Device` directly without
    /// depending on candle's `MetalDevice` wrapper. The internal
    /// storage field continues to hold the candle `Arc<MetalDevice>`
    /// for now (so every existing kernel-crate FFI site reading
    /// `self.candle_device.clone()` keeps working unchanged); the
    /// field flip to a raw `Arc<MetalRawDevice>` is a follow-up step
    /// that can land after all callers have migrated to read
    /// `.metal_device_handle()` instead of `.candle_device()`.
    ///
    /// Mirror of [`crate::CudaStorage::context`] — same shape, same
    /// rationale (the read-bridge step of the CP-1/CP-2 substrate lift
    /// documented in `docs/issue-1082-tier-4-5-roadmap-2026-05-27.md`).
    pub fn metal_device_handle(&self) -> MetalRawDevice {
        self.candle_device.metal_device().clone()
    }

    /// Returns `true` iff this storage's buffer is in a UMA-compatible
    /// storage mode (shared / managed).
    ///
    /// On Apple Silicon, every Metal device is UMA and every buffer
    /// candle's `MetalDevice::allocate_zeros` hands out is in
    /// `MTLStorageModeShared`; `from_buffer` callers must also pass a
    /// Shared/Managed buffer (the constructor's contract). Since Metal
    /// is only supported on Apple Silicon hosts, this is unconditionally
    /// `true` — querying the buffer's actual storage mode would require
    /// reaching through `candle_metal_kernels::metal::Buffer` to the
    /// inner `dyn MTLBuffer` protocol object, which `Buffer` does not
    /// expose. Revisit when supporting Intel Macs or Private-mode
    /// buffers becomes a goal.
    pub fn is_unified_memory(&self) -> bool {
        true
    }
}

impl StorageBackend for MetalStorage {
    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        self.buffer.length() as usize
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Resolve the primary candle `MetalDevice` for the given Metal device
/// ordinal — public mirror of [`crate::primary_cuda_device`] for the
/// Metal backend.
///
/// Calls `candle_core::Device::new_metal(device_index)` and unwraps to
/// `Arc<MetalDevice>`. Used by [`MetalStorage::zeros_kt`] to populate
/// the back-compat `candle_device` field on storage allocated via the
/// candle-free metal-rs path; also exposed publicly so downstream test
/// code can construct Metal tensors candle-free at the construction
/// boundary.
///
/// # Phase 7 (#1082) note
///
/// Once every kernel-crate FFI site migrates to a raw-metal-rs
/// `MTLDevice` + `MTLCommandQueue` surface, the storage-side
/// `candle_device` field disappears and this helper retires. Until
/// then it stays as the single residual candle hook in the
/// substrate's allocation path.
#[allow(dead_code)]
pub fn primary_metal_device(device_index: usize) -> Result<Arc<MetalDevice>> {
    match candle_core::Device::new_metal(device_index)
        .map_err(|e| Error::Msg(format!("primary_metal_device({device_index}): {e}")))?
    {
        candle_core::Device::Metal(d) => Ok(Arc::new(d)),
        _ => Err(Error::Msg(format!(
            "primary_metal_device({device_index}): expected Metal device"
        ))),
    }
}


// ----------------------------------------------------------------------
// metal_softmax_last_axis — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal softmax over the trailing axis. Mirrors the role of
/// [`crate::cuda_softmax_last_axis`] for the Metal backend.
///
/// Operates on a contiguous `[..., D]` Metal-backed tensor; produces a
/// fresh contiguous tensor of the same shape and dtype with each
/// `[..., :]` row normalized to a probability distribution.
///
/// # Implementation
///
/// Delegates to candle's `candle_nn::ops::softmax_last_dim`, which
/// dispatches to candle's Metal kernels under the hood. We wrap the
/// kt-Tensor's `Arc<metal::Buffer>` in a candle [`candle_core::MetalStorage`]
/// (sharing the underlying MTLBuffer — zero-copy on Apple Silicon UMA),
/// build a candle Tensor via the public [`candle_core::Tensor::from_storage`],
/// run the softmax, then wrap the result's MTLBuffer back into a kt
/// `MetalStorage`. On UMA, both the input and output Arcs point to the
/// same underlying MTLBuffer (refcounted by Objective-C `retain`), so
/// no host/device round-trip occurs.
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16`
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` must hold
///
/// # Errors
///
/// Returns [`Error::Msg`] if the storage isn't `MetalStorage`, the
/// dtype is unsupported, the layout is non-contiguous, or the
/// underlying candle call fails (e.g. shape/dtype mismatch the candle
/// kernel cannot handle).
///
/// # Phase 7 (#1082) note
///
/// This is the canonical kt-API Metal softmax for Phase 4 (substrate
/// ops); Phase 7 candle removal replaces the candle-typed inner call
/// with a direct `candle_metal_kernels::call_last_softmax` or a vendored
/// MSL kernel. The public signature (`metal_softmax_last_axis(&Tensor)
/// -> Result<Tensor>`) does not change.
pub fn metal_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    use candle_core::{op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage, Storage as CandleStorage, Tensor as CandleTensor};

    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    let candle_dtype = match dtype {
        DType::F32 => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
        other => {
            return Err(Error::Msg(format!(
                "metal_softmax_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if x.rank() == 0 {
        return Err(Error::Msg(
            "metal_softmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "metal_softmax_last_axis: input must be contiguous".to_string(),
        ));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_softmax_last_axis: input must be Metal-backed".to_string())
        })?;

    let candle_device_arc = kt_metal.candle_device().clone();
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();

    // ---- Wrap kt buffer in a candle MetalStorage (shares underlying MTLBuffer) ----
    //
    // `candle_core::MetalStorage::new(buffer, device, count, dtype)` takes:
    //   - buffer: Arc<metal::Buffer>  (same type kt-tensor holds)
    //   - device: MetalDevice         (cloned from the Arc — cheap, NSObject retain)
    //   - count: usize                (element count)
    //   - dtype: candle DType
    //
    // The resulting candle MetalStorage shares the MTLBuffer with kt-tensor's
    // MetalStorage (Apple Silicon UMA: a single physical allocation, two Arc
    // handles, one MTLBuffer retain-count).
    let candle_in_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal.buffer()),
        (*candle_device_arc).clone(),
        element_count,
        candle_dtype,
    );
    let candle_storage = CandleStorage::Metal(candle_in_storage);
    let candle_in: CandleTensor = CandleTensor::from_storage(
        candle_storage,
        shape.as_slice(),
        BackpropOp::none(),
        /*is_variable=*/ false,
    );

    // ---- Dispatch through candle_nn ----
    //
    // `candle_nn::ops::softmax_last_dim` invokes candle's Metal softmax
    // kernel (via the same path candle_nn::ops::softmax(_, -1) takes).
    let candle_out: CandleTensor = candle_nn::ops::softmax_last_dim(&candle_in).map_err(|e| {
        Error::Msg(format!(
            "metal_softmax_last_axis: candle_nn::ops::softmax_last_dim failed: {e}"
        ))
    })?;

    // candle's softmax may return a non-contiguous result depending on the
    // kernel path; force contiguity so the resulting kt tensor's stride
    // assumption holds.
    let candle_out = candle_out.contiguous().map_err(|e| {
        Error::Msg(format!(
            "metal_softmax_last_axis: candle contiguous failed: {e}"
        ))
    })?;

    // ---- Extract the output buffer back into kt-Tensor space ----
    //
    // candle's `Tensor::storage_and_layout` returns a `RwLockReadGuard`
    // over a `candle_core::Storage`. Match on the Metal variant and
    // extract the underlying MTLBuffer via `.buffer()`. We then wrap
    // a clone in an `Arc<metal::Buffer>` — `metal::Buffer: Clone`
    // performs an NSObject retain (refcount bump), so kt-Tensor's new
    // Arc points to the same MTLBuffer the candle result owns.
    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_softmax_last_axis: candle softmax returned non-Metal storage \
                 (unexpected — candle_nn::ops::softmax_last_dim preserves device)"
                    .to_string(),
            ));
        }
    };

    // `candle_out_metal.buffer()` returns `&metal::Buffer`. The `metal`
    // crate's `Buffer` is a thin Objective-C handle (`ProtocolObject<MTLBuffer>`);
    // its `Clone` impl is an NSObject `retain`, so cloning is cheap and
    // refcounts the underlying MTLBuffer. Wrapping in `Arc::new(...)`
    // gives kt-Tensor its own Arc<Buffer> sharing the MTLBuffer with
    // the candle result. When the candle Tensor `candle_out` drops at
    // the end of this function, the MTLBuffer survives (kt-side retain
    // still holds it).
    let out_buffer_clone = candle_out_metal.buffer().to_owned();
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> = Arc::new(out_buffer_clone);

    let out_storage = MetalStorage::from_buffer(
        candle_device_arc,
        device_index,
        dtype,
        out_buffer_arc,
    )?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    // Drop the candle read guard before we construct the kt tensor.
    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_rmsnorm_last_axis — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal RMSNorm over the trailing axis. Mirrors the role of
/// [`crate::cuda_rmsnorm_last_axis`] for the Metal backend.
///
/// Operates on contiguous `[..., D]` and `[D]` Metal-backed tensors and
/// produces a fresh contiguous tensor with each `[..., :]` row
/// normalized by its row-RMS and scaled per-element by `weight`:
/// `y = w * x / sqrt(mean(x^2) + eps)`.
///
/// # Implementation
///
/// Delegates to candle's `candle_nn::ops::rms_norm`, which dispatches
/// through candle's Metal RMSNorm kernel under the hood (the same path
/// candle's `RmsNorm` layer takes). We wrap both kt-Tensors'
/// `Arc<metal::Buffer>` handles in candle [`candle_core::MetalStorage`]
/// values (sharing the underlying MTLBuffers — zero-copy on UMA),
/// build candle Tensors via [`candle_core::Tensor::from_storage`], run
/// `rms_norm`, then wrap the result's MTLBuffer back into a kt
/// `MetalStorage`. Both input and output Arcs point at the same
/// MTLBuffers (refcounted by Objective-C `retain`); no host/device
/// round-trip on Apple Silicon.
///
/// # Requirements
///
/// - `x` and `weight` must both be backed by [`MetalStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16` (and equal to
///   `weight.dtype()`)
/// - `x.rank() >= 1`, `weight.rank() == 1`
/// - `weight.shape()[0] == *x.shape().last().unwrap()`
/// - both inputs contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or candle error.
///
/// # Phase 7 (#1082) note
///
/// Phase 7 candle removal replaces the candle inner call with a
/// `candle_metal_kernels::call_rms_norm` (or vendored MSL kernel).
/// Public signature stays the same.
pub fn metal_rmsnorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use candle_core::{
        op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage,
        Storage as CandleStorage, Tensor as CandleTensor,
    };

    let dtype = x.dtype();
    let candle_dtype = match dtype {
        DType::F32 => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
        other => {
            return Err(Error::Msg(format!(
                "metal_rmsnorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if weight.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_rmsnorm_last_axis: weight dtype {} != x dtype {dtype}",
            weight.dtype()
        )));
    }
    if x.rank() == 0 || weight.rank() != 1 {
        return Err(Error::Msg(format!(
            "metal_rmsnorm_last_axis: rank constraints failed (x.rank={}, weight.rank={})",
            x.rank(),
            weight.rank()
        )));
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return Err(Error::Msg(
            "metal_rmsnorm_last_axis: inputs must be contiguous".to_string(),
        ));
    }
    let hidden = *x.shape().last().unwrap();
    if weight.shape().first().copied() != Some(hidden) {
        return Err(Error::Msg(format!(
            "metal_rmsnorm_last_axis: weight.shape()[0] {:?} != x last-dim {hidden}",
            weight.shape()
        )));
    }

    let kt_metal_x = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_rmsnorm_last_axis: x must be Metal-backed".to_string())
        })?;
    let kt_metal_w = weight
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_rmsnorm_last_axis: weight must be Metal-backed".to_string())
        })?;

    let candle_device_arc = kt_metal_x.candle_device().clone();
    let device_index = match kt_metal_x.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count_x: usize = x.element_count();
    let element_count_w: usize = weight.element_count();

    let candle_x_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_x.buffer()),
        (*candle_device_arc).clone(),
        element_count_x,
        candle_dtype,
    );
    let candle_x: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_x_storage),
        shape.as_slice(),
        BackpropOp::none(),
        false,
    );

    let candle_w_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_w.buffer()),
        (*candle_device_arc).clone(),
        element_count_w,
        candle_dtype,
    );
    let candle_w: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_w_storage),
        &[hidden],
        BackpropOp::none(),
        false,
    );

    let candle_out: CandleTensor =
        candle_nn::ops::rms_norm(&candle_x, &candle_w, eps).map_err(|e| {
            Error::Msg(format!(
                "metal_rmsnorm_last_axis: candle_nn::ops::rms_norm failed: {e}"
            ))
        })?;

    let candle_out = candle_out.contiguous().map_err(|e| {
        Error::Msg(format!(
            "metal_rmsnorm_last_axis: candle contiguous failed: {e}"
        ))
    })?;

    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_rmsnorm_last_axis: candle rms_norm returned non-Metal storage \
                 (unexpected — candle_nn::ops::rms_norm preserves device)"
                    .to_string(),
            ));
        }
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());

    let out_storage =
        MetalStorage::from_buffer(candle_device_arc, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_layernorm_last_axis — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal LayerNorm over the trailing axis. Mirrors the role of
/// [`crate::cuda_layernorm_last_axis`] for the Metal backend.
///
/// Operates on contiguous `[..., D]`, `[D]`, `[D]` Metal-backed
/// tensors and produces a fresh contiguous tensor:
/// `y = ((x - mean) / sqrt(var + eps)) * weight + bias`.
///
/// # Implementation
///
/// Delegates to `candle_nn::ops::layer_norm`, which dispatches through
/// candle's Metal LayerNorm kernel. Bridges all three kt MetalStorages
/// -> candle MetalStorages via `Arc<metal::Buffer>` clone (NSObject
/// retain), runs `layer_norm`, then wraps the result MTLBuffer back
/// into a kt `MetalStorage`. Zero-copy on Apple Silicon UMA — same
/// pattern as `metal_softmax_last_axis` and `metal_rmsnorm_last_axis`.
///
/// # Requirements
///
/// - `x`, `weight`, `bias` must all be Metal-backed
/// - all three share dtype in {F32, BF16, F16}
/// - `x.rank() >= 1`, `weight.rank() == 1`, `bias.rank() == 1`
/// - `weight.shape()[0] == bias.shape()[0] == *x.shape().last().unwrap()`
/// - all three contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or candle error.
pub fn metal_layernorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    bias: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use candle_core::{
        op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage,
        Storage as CandleStorage, Tensor as CandleTensor,
    };

    let dtype = x.dtype();
    let candle_dtype = match dtype {
        DType::F32 => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
        other => {
            return Err(Error::Msg(format!(
                "metal_layernorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if weight.dtype() != dtype || bias.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_layernorm_last_axis: weight dtype {} / bias dtype {} != x dtype {dtype}",
            weight.dtype(),
            bias.dtype()
        )));
    }
    if x.rank() == 0 || weight.rank() != 1 || bias.rank() != 1 {
        return Err(Error::Msg(format!(
            "metal_layernorm_last_axis: rank constraints failed (x={}, w={}, b={})",
            x.rank(),
            weight.rank(),
            bias.rank()
        )));
    }
    if !x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() {
        return Err(Error::Msg(
            "metal_layernorm_last_axis: inputs must be contiguous".to_string(),
        ));
    }
    let hidden = *x.shape().last().unwrap();
    if weight.shape().first().copied() != Some(hidden)
        || bias.shape().first().copied() != Some(hidden)
    {
        return Err(Error::Msg(format!(
            "metal_layernorm_last_axis: weight/bias shapes ({:?}, {:?}) != x last-dim {hidden}",
            weight.shape(),
            bias.shape()
        )));
    }

    let kt_metal_x = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_layernorm_last_axis: x must be Metal-backed".to_string())
        })?;
    let kt_metal_w = weight
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_layernorm_last_axis: weight must be Metal-backed".to_string())
        })?;
    let kt_metal_b = bias
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_layernorm_last_axis: bias must be Metal-backed".to_string())
        })?;

    let candle_device_arc = kt_metal_x.candle_device().clone();
    let device_index = match kt_metal_x.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count_x = x.element_count();
    let element_count_w = weight.element_count();
    let element_count_b = bias.element_count();

    let candle_x_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_x.buffer()),
        (*candle_device_arc).clone(),
        element_count_x,
        candle_dtype,
    );
    let candle_x: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_x_storage),
        shape.as_slice(),
        BackpropOp::none(),
        false,
    );

    let candle_w_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_w.buffer()),
        (*candle_device_arc).clone(),
        element_count_w,
        candle_dtype,
    );
    let candle_w: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_w_storage),
        &[hidden],
        BackpropOp::none(),
        false,
    );

    let candle_b_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_b.buffer()),
        (*candle_device_arc).clone(),
        element_count_b,
        candle_dtype,
    );
    let candle_b: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_b_storage),
        &[hidden],
        BackpropOp::none(),
        false,
    );

    let candle_out: CandleTensor =
        candle_nn::ops::layer_norm(&candle_x, &candle_w, &candle_b, eps).map_err(|e| {
            Error::Msg(format!(
                "metal_layernorm_last_axis: candle_nn::ops::layer_norm failed: {e}"
            ))
        })?;

    let candle_out = candle_out.contiguous().map_err(|e| {
        Error::Msg(format!(
            "metal_layernorm_last_axis: candle contiguous failed: {e}"
        ))
    })?;

    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_layernorm_last_axis: candle layer_norm returned non-Metal storage"
                    .to_string(),
            ));
        }
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());

    let out_storage =
        MetalStorage::from_buffer(candle_device_arc, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_index_select_dim0 — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal index_select along axis 0. Mirrors the role of
/// [`crate::cuda_index_select_dim0`] for the Metal backend.
///
/// Dispatches through candle's production Metal `index_select` kernel
/// (`candle_metal_kernels::call_index_select`, the same path
/// `candle_core::Tensor::index_select(...)` takes).
///
/// Given:
///   - `input: [vocab_size, hidden]` or `[axis_dim, ...]` (rank >= 1)
///   - `indices: [N]` or higher-rank, dtype U32
///
/// Produces a contiguous `[indices.shape, ...input.shape[1..]]` tensor
/// with the same dtype as `input`.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Same wrap-share-retain pattern as the other metal_* helpers — one
/// physical allocation per side.
///
/// # Requirements
///
/// - `input` must be backed by [`MetalStorage`]
/// - `indices` must be backed by [`MetalStorage`]
/// - `input` and `indices` both contiguous
/// - `indices.dtype() == U32` (matches CUDA's `cuda_index_select_dim0`)
/// - `input.dtype()` in {F32, BF16, F16} (candle's Metal index_select
///   supports these; integer/packed dtypes return Err here and the
///   op's metal_fwd falls through to CPU.)
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported dtype, non-contiguous layout,
/// non-Metal storage, or candle kernel error.
pub fn metal_index_select_dim0(
    input: &crate::Tensor,
    indices: &crate::Tensor,
) -> Result<crate::Tensor> {
    use candle_core::{
        op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage,
        Storage as CandleStorage, Tensor as CandleTensor,
    };

    let dtype = input.dtype();
    let candle_input_dtype = match dtype {
        DType::F32 => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
        other => {
            return Err(Error::Msg(format!(
                "metal_index_select_dim0: unsupported input dtype {other} \
                 (float triple only)"
            )));
        }
    };
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "metal_index_select_dim0: indices dtype must be U32 (got {})",
            indices.dtype()
        )));
    }
    if !input.is_contiguous() || !indices.is_contiguous() {
        return Err(Error::Msg(
            "metal_index_select_dim0: inputs must be contiguous".to_string(),
        ));
    }
    if input.rank() == 0 {
        return Err(Error::Msg(
            "metal_index_select_dim0: input must have rank >= 1".to_string(),
        ));
    }

    let kt_metal_in = input
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_index_select_dim0: input must be Metal-backed".to_string())
        })?;
    let kt_metal_ids = indices
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_index_select_dim0: indices must be Metal-backed".to_string())
        })?;

    let candle_device_arc = kt_metal_in.candle_device().clone();
    let device_index = match kt_metal_in.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let in_shape: Vec<usize> = input.shape().to_vec();
    let in_element_count: usize = input.element_count();
    let ids_shape: Vec<usize> = indices.shape().to_vec();
    let ids_element_count: usize = indices.element_count();

    // ---- Wrap kt buffers in candle MetalStorages ----
    let candle_in_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_in.buffer()),
        (*candle_device_arc).clone(),
        in_element_count,
        candle_input_dtype,
    );
    let candle_in: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_in_storage),
        in_shape.as_slice(),
        BackpropOp::none(),
        false,
    );
    let candle_ids_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_ids.buffer()),
        (*candle_device_arc).clone(),
        ids_element_count,
        CandleDType::U32,
    );
    // candle's index_select accepts multi-dim indices; flatten internally.
    let candle_ids: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_ids_storage),
        ids_shape.as_slice(),
        BackpropOp::none(),
        false,
    );

    // ---- Dispatch ----
    let candle_out: CandleTensor = candle_in.index_select(&candle_ids, 0).map_err(|e| {
        Error::Msg(format!(
            "metal_index_select_dim0: candle index_select failed: {e}"
        ))
    })?;

    let candle_out = candle_out.contiguous().map_err(|e| {
        Error::Msg(format!(
            "metal_index_select_dim0: candle contiguous failed: {e}"
        ))
    })?;

    let out_shape: Vec<usize> = candle_out.shape().dims().to_vec();
    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_index_select_dim0: candle index_select returned non-Metal storage"
                    .to_string(),
            ));
        }
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());

    let out_storage =
        MetalStorage::from_buffer(candle_device_arc, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_cast — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal dtype cast. Mirrors the role of [`crate::cuda_cast`] for the
/// Metal backend.
///
/// Dispatches through candle's production Metal `to_dtype` kernel
/// (the same path `candle_core::Tensor::to_dtype(...)` takes). Covers
/// F32 <-> BF16 <-> F16 — the float triple. Integer round-trips
/// (U32 <-> I64) stay on the CPU fallback for now.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Same pattern as `metal_softmax_last_axis` and friends:
/// - Wrap kt's `MetalStorage` buffer in a candle `MetalStorage`
///   (shares the same `MTLBuffer`).
/// - Run candle's Metal `to_dtype` kernel on the wrapped tensor.
/// - Reach into the candle output's storage, clone the `metal::Buffer`
///   (an NSObject `retain`), and wrap it back into a kt `MetalStorage`.
///
/// One physical allocation per side; no host bounce.
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.is_contiguous()`
/// - `(x.dtype(), to)` in the supported float triple
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported dtype pair, non-contiguous
/// layout, non-Metal storage, or candle kernel error.
pub fn metal_cast(x: &crate::Tensor, to: DType) -> Result<crate::Tensor> {
    use candle_core::{
        op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage,
        Storage as CandleStorage, Tensor as CandleTensor,
    };

    let from = x.dtype();
    let dtype_to_candle = |d: DType| -> Result<CandleDType> {
        match d {
            DType::F32 => Ok(CandleDType::F32),
            DType::BF16 => Ok(CandleDType::BF16),
            DType::F16 => Ok(CandleDType::F16),
            other => Err(Error::Msg(format!(
                "metal_cast: dtype {other} cannot be mapped to candle DType \
                 (float triple only)"
            ))),
        }
    };
    let candle_from = dtype_to_candle(from)?;
    let candle_to = dtype_to_candle(to)?;

    if !x.is_contiguous() {
        return Err(Error::Msg(
            "metal_cast: input must be contiguous".to_string(),
        ));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_cast: input must be Metal-backed".to_string()))?;

    let candle_device_arc = kt_metal.candle_device().clone();
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();

    // ---- Wrap kt buffer in a candle MetalStorage (shares MTLBuffer) ----
    let candle_in_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal.buffer()),
        (*candle_device_arc).clone(),
        element_count,
        candle_from,
    );
    let candle_storage = CandleStorage::Metal(candle_in_storage);
    let candle_in: CandleTensor = CandleTensor::from_storage(
        candle_storage,
        shape.as_slice(),
        BackpropOp::none(),
        /*is_variable=*/ false,
    );

    // ---- Dispatch through candle ----
    let candle_out: CandleTensor = candle_in.to_dtype(candle_to).map_err(|e| {
        Error::Msg(format!(
            "metal_cast: candle to_dtype({candle_from:?} -> {candle_to:?}) failed: {e}"
        ))
    })?;
    let candle_out = candle_out
        .contiguous()
        .map_err(|e| Error::Msg(format!("metal_cast: candle contiguous failed: {e}")))?;

    // ---- Extract output buffer back into kt-Tensor space ----
    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_cast: candle to_dtype returned non-Metal storage \
                 (unexpected — preserves device)"
                    .to_string(),
            ));
        }
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());

    let out_storage = MetalStorage::from_buffer(candle_device_arc, device_index, to, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_elementwise_binary — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal element-wise binary op (add/sub/mul/div). Mirrors the role of
/// [`crate::cuda_elementwise_binary`] for the Metal backend.
///
/// Dispatches on `kind_tag` (matches the CUDA tags in
/// `ElementwiseOp::cuda_fwd`):
///   - `0` -> Add via `candle::Tensor::add(&a, &b)`
///   - `1` -> Sub via `candle::Tensor::sub(&a, &b)`
///   - `2` -> Mul via `candle::Tensor::mul(&a, &b)`
///   - `3` -> Div via `candle::Tensor::div(&a, &b)`
///
/// candle's pointwise binary ops go through the production Metal
/// shaders (`affine_*` / `binary_*` kernels in
/// `vendor/candle-metal-kernels`). Covers F32 / BF16 / F16. Both inputs
/// must share shape and dtype.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Same pattern as `metal_softmax_last_axis` and friends — wrap each
/// kt MetalStorage buffer in a candle MetalStorage (shared MTLBuffer),
/// dispatch, then `retain`-clone the candle output buffer back into a
/// fresh kt MetalStorage.
///
/// # Requirements
///
/// - `a` and `b` must both be backed by [`MetalStorage`]
/// - `a.dtype() == b.dtype()` and dtype in {F32, BF16, F16}
/// - `a.shape() == b.shape()` (no broadcasting yet)
/// - both contiguous
/// - `kind_tag` in {0, 1, 2, 3}
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported kind, dtype, non-contiguous
/// layout, non-Metal storage, shape mismatch, or candle kernel error.
pub fn metal_elementwise_binary(
    a: &crate::Tensor,
    b: &crate::Tensor,
    kind_tag: i32,
) -> Result<crate::Tensor> {
    use candle_core::{
        op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage,
        Storage as CandleStorage, Tensor as CandleTensor,
    };

    if !matches!(kind_tag, 0 | 1 | 2 | 3) {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: kind_tag {kind_tag} not supported \
             (only 0=Add, 1=Sub, 2=Mul, 3=Div)"
        )));
    }
    let dtype = a.dtype();
    let candle_dtype = match dtype {
        DType::F32 => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
        other => {
            return Err(Error::Msg(format!(
                "metal_elementwise_binary: unsupported dtype {other}"
            )));
        }
    };
    if b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: dtype mismatch a={dtype} b={}",
            b.dtype()
        )));
    }
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "metal_elementwise_binary: inputs must be contiguous".to_string(),
        ));
    }

    let kt_metal_a = a
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_elementwise_binary: a must be Metal-backed".to_string()))?;
    let kt_metal_b = b
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_elementwise_binary: b must be Metal-backed".to_string()))?;

    let candle_device_arc = kt_metal_a.candle_device().clone();
    let device_index = match kt_metal_a.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = a.shape().to_vec();
    let element_count: usize = a.element_count();

    // ---- Wrap kt buffers in candle MetalStorages ----
    let candle_a_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_a.buffer()),
        (*candle_device_arc).clone(),
        element_count,
        candle_dtype,
    );
    let candle_a: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_a_storage),
        shape.as_slice(),
        BackpropOp::none(),
        false,
    );
    let candle_b_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal_b.buffer()),
        (*candle_device_arc).clone(),
        element_count,
        candle_dtype,
    );
    let candle_b: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_b_storage),
        shape.as_slice(),
        BackpropOp::none(),
        false,
    );

    // ---- Dispatch ----
    let candle_out: CandleTensor = match kind_tag {
        0 => (&candle_a + &candle_b),
        1 => (&candle_a - &candle_b),
        2 => (&candle_a * &candle_b),
        3 => (&candle_a / &candle_b),
        _ => unreachable!("gated above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "metal_elementwise_binary: candle binary op (kind={kind_tag}) failed: {e}"
        ))
    })?;

    let candle_out = candle_out.contiguous().map_err(|e| {
        Error::Msg(format!(
            "metal_elementwise_binary: candle contiguous failed: {e}"
        ))
    })?;

    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_elementwise_binary: candle returned non-Metal storage \
                 (unexpected — binary ops preserve device)"
                    .to_string(),
            ));
        }
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());

    let out_storage =
        MetalStorage::from_buffer(candle_device_arc, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_activation_unary — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal unary activation. Mirrors the role of
/// [`crate::cuda_activation_unary`] for the Metal backend.
///
/// Dispatches on `kind_tag` (matches the CUDA tags in
/// `ActivationOp::cuda_fwd` and `UnaryArithKind::cuda_kind_tag`):
///   - `0`  -> Silu via `candle::Tensor::silu(&x)` (activation)
///   - `1`  -> Sigmoid — not natively in candle's `UnaryOp`. This
///     wrapper rejects kind=1; callers must fall through to CPU until
///     a real Metal sigmoid kernel lands.
///   - `2`  -> Gelu via `candle::Tensor::gelu(&x)` (tanh approximation,
///     matches the CPU/CUDA formula)
///   - `3`  -> Tanh via `candle::Tensor::tanh(&x)`
///   - `4`  -> Relu via `candle::Tensor::relu(&x)`
///   - `5`  -> Ln via `candle::Tensor::log(&x)` (unary arith)
///   - `6`  -> Exp via `candle::Tensor::exp(&x)` (unary arith)
///   - `12` -> Neg via `candle::Tensor::neg(&x)` (unary arith)
///   - `13` -> Abs via `candle::Tensor::abs(&x)` (unary arith)
///   - `14` -> Sqrt via `candle::Tensor::sqrt(&x)` (unary arith)
///
/// candle's pointwise unary ops go through the production Metal
/// shaders (`unary_*` kernels in `vendor/candle-metal-kernels`). Covers
/// F32 / BF16 / F16.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Same pattern as `metal_softmax_last_axis` and friends — wrap kt
/// MetalStorage buffer in a candle MetalStorage (shared MTLBuffer),
/// dispatch, then `retain`-clone the candle output buffer back into a
/// fresh kt MetalStorage.
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.dtype()` in {F32, BF16, F16}
/// - `x.is_contiguous()`
/// - `kind_tag` in {0, 2, 3, 4} (Sigmoid=1 is rejected pending kernel)
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported kind, dtype, non-contiguous
/// layout, non-Metal storage, or candle kernel error.
pub fn metal_activation_unary(x: &crate::Tensor, kind_tag: i32) -> Result<crate::Tensor> {
    use candle_core::{
        op::BackpropOp, DType as CandleDType, MetalStorage as CandleMetalStorage,
        Storage as CandleStorage, Tensor as CandleTensor,
    };

    if !matches!(
        kind_tag,
        0 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 13 | 14 | 22 | 23 | 24 | 25 | 26
    ) {
        return Err(Error::Msg(format!(
            "metal_activation_unary: kind_tag {kind_tag} not supported on Metal today \
             (0=Silu, 2=Gelu, 3=Tanh, 4=Relu, 5=Ln, 6=Exp, 12=Neg, 13=Abs, 14=Sqrt; \
             Sigmoid=1 has no candle UnaryOp — falls through to CPU until a Metal \
             sigmoid kernel lands)"
        )));
    }
    let dtype = x.dtype();
    let candle_dtype = match dtype {
        DType::F32 => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16 => CandleDType::F16,
        other => {
            return Err(Error::Msg(format!(
                "metal_activation_unary: unsupported dtype {other} (F32/BF16/F16 only)"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "metal_activation_unary: input must be contiguous".to_string(),
        ));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_activation_unary: input must be Metal-backed".to_string())
        })?;

    let candle_device_arc = kt_metal.candle_device().clone();
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();

    // ---- Wrap kt buffer in candle MetalStorage ----
    let candle_in_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal.buffer()),
        (*candle_device_arc).clone(),
        element_count,
        candle_dtype,
    );
    let candle_in: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_in_storage),
        shape.as_slice(),
        BackpropOp::none(),
        false,
    );

    // ---- Dispatch through candle's Metal unary kernel ----
    let candle_out: CandleTensor = match kind_tag {
        0 => candle_in.silu(),
        2 => candle_in.gelu(),
        3 => candle_in.tanh(),
        4 => candle_in.relu(),
        5 => candle_in.log(),
        6 => candle_in.exp(),
        12 => candle_in.neg(),
        13 => candle_in.abs(),
        14 => candle_in.sqrt(),
        7 => candle_in.sin(),
        8 => candle_in.cos(),
        22 => candle_in.recip(),
        23 => candle_in.sign(),
        24 => candle_in.floor(),
        25 => candle_in.ceil(),
        26 => candle_in.round(),
        _ => unreachable!("gated above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "metal_activation_unary: candle unary op (kind={kind_tag}) failed: {e}"
        ))
    })?;

    let candle_out = candle_out.contiguous().map_err(|e| {
        Error::Msg(format!(
            "metal_activation_unary: candle contiguous failed: {e}"
        ))
    })?;

    let (out_storage_guard, _out_layout) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => {
            return Err(Error::Msg(
                "metal_activation_unary: candle returned non-Metal storage \
                 (unexpected — unary ops preserve device)"
                    .to_string(),
            ));
        }
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());

    let out_storage =
        MetalStorage::from_buffer(candle_device_arc, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    drop(out_storage_guard);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_metal_kernels::metal::MTLResourceOptions;

    fn metal_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_METAL_TEST").ok().as_deref() == Some("1")
    }

    /// Candle-free Metal device handle for the tests below.
    ///
    /// Mirrors the `maybe_metal_raw_device` pattern in
    /// `metal_allocator.rs` — uses `MetalRawDevice::system_default()`
    /// (the `candle_metal_kernels::metal` re-export of Apple's `metal`
    /// crate's `Device::system_default`) so the test mod does not need
    /// `candle_core::Device::new_metal(0)`. This drops the
    /// `use candle_core::Device as CandleDevice` import that the test
    /// mod previously carried — one less candle hook in
    /// `metal_storage.rs` (#1082 candle removal).
    fn maybe_metal_raw_device() -> Option<MetalRawDevice> {
        if !metal_test_enabled() {
            return None;
        }
        MetalRawDevice::system_default()
    }

    #[test]
    fn zeros_round_sizes() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        // Exercise the candle-free metal-rs allocation path.
        // `zeros_kt` round-trips through `device.new_buffer` directly
        // (no candle blit-encoder), matching what `MetalAllocator`
        // already uses in production. The metal-rs `new_buffer` does
        // NOT round up to slab sizes the way candle's `allocate_zeros`
        // did, so `byte_len()` reports exactly the dtype-derived
        // byte_len now (BF16 * 64 = 128 bytes exactly).
        let storage = MetalStorage::zeros_kt(&dev, 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Metal(0));
        assert_eq!(storage.dtype(), DType::BF16);
        assert!(storage.byte_len() >= 128);

        let storage = MetalStorage::zeros_kt(&dev, 0, DType::Int4Packed, 16).unwrap();
        assert!(storage.byte_len() >= 8);
    }

    #[test]
    fn from_buffer_validates_alignment() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        // 17 bytes is not a multiple of f32 (4). metal-rs does NOT
        // round up `new_buffer`-sized allocations the way candle's
        // `allocate_zeros` did (which went through candle's slab
        // cache), so `raw_len` here equals 17 — the unaligned-len
        // branch of the test below is the one that fires.
        //
        // `MetalStorage::from_buffer` still requires `Arc<MetalDevice>`
        // (the back-compat field is load-bearing for downstream
        // kernel-crate FFI sites that consume `.candle_device()`); we
        // derive that wrapper via the public `primary_metal_device`
        // helper rather than reaching for `candle_core::Device::new_metal`
        // directly. The helper is documented as the substrate's single
        // residual candle hook — the same hook `MetalStorage::zeros_kt`
        // already calls internally, so this test does not add to the
        // candle surface.
        let candle_dev = primary_metal_device(0).unwrap();
        let small = dev
            .new_buffer(17, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let raw_len = small.length() as usize;
        let small_arc = Arc::new(small);
        let result = MetalStorage::from_buffer(candle_dev, 0, DType::F32, small_arc);
        if raw_len.is_multiple_of(4) {
            // metal-rs rounded up (host-specific behavior); validation passes.
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().to_string().contains("not a multiple"));
        }
    }
}
