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

/// Construct a fresh [`crate::Storage`] handle holding a [`MetalStorage`].
pub fn metal_zeros(
    candle_device: Arc<MetalDevice>,
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let storage = MetalStorage::zeros(candle_device, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
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
    fn zeros_round_sizes() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let storage = MetalStorage::zeros(dev.clone(), 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Metal(0));
        assert_eq!(storage.dtype(), DType::BF16);
        // Candle's metal allocator rounds up to its slab size; only assert
        // that the byte_len is *at least* what we asked for.
        assert!(storage.byte_len() >= 128);

        let storage = MetalStorage::zeros(dev, 0, DType::Int4Packed, 16).unwrap();
        assert!(storage.byte_len() >= 8);
    }

    #[test]
    fn from_buffer_validates_alignment() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        // 17 bytes is not a multiple of f32 (4). The allocator may round
        // up our request, so this only tests the *post-round-up* size
        // when we explicitly pass a 17-byte buffer.
        let small = dev.allocate_zeros(17).unwrap();
        let raw_len = small.length() as usize;
        let result = MetalStorage::from_buffer(dev, 0, DType::F32, small);
        if raw_len.is_multiple_of(4) {
            // Allocator rounded up; the validation passes.
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().to_string().contains("not a multiple"));
        }
    }

    #[test]
    fn metal_zeros_returns_arc_storage() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let s: crate::Storage = metal_zeros(dev, 0, DType::F32, 4).unwrap();
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.device(), Device::Metal(0));
        let metal_s = s.as_any().downcast_ref::<MetalStorage>().expect("downcast");
        // UMA invariant: Shared-mode buffer (Apple Silicon default).
        assert!(metal_s.is_unified_memory());
    }
}
