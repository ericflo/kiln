//! Vulkan storage impl behind the `vulkan` feature flag.
//!
//! Lifts `kiln_vulkan_kernel::VulkanBuffer` + `VulkanDevice`. Unlike
//! [`CudaStorage`](crate::CudaStorage) and
//! [`MetalStorage`](crate::MetalStorage), this storage carries **no
//! transitive candle dependency** — `kiln-vulkan-kernel` is already
//! candle-free (uses `ash` + custom GLSL compute) so Phase 7 of #1082
//! does not need to replace anything here.
//!
//! # Anti-pattern 1 compliance
//!
//! `VulkanStorage` holds a `VulkanBuffer` directly. The buffer wraps
//! `vk::Buffer + vk::DeviceMemory + Arc<ash::Device>` and is owned by
//! us. No candle types appear in the type signature.
//!
//! # Phase 1.8 scope
//!
//! Storage-layer only:
//!
//! - `zeros(device, dtype, n_elements)` — device-local buffer alloc.
//! - `from_buffer(device, dtype, buffer)` — wrap an existing
//!   `VulkanBuffer` allocated by `kiln-vulkan-kernel`.
//! - `StorageBackend` impl returning `Device::Vulkan(idx)`, dtype,
//!   byte len.
//! - `buffer()` / `vulkan_device()` accessors for FFI sites in
//!   `kiln-vulkan-kernel`'s kernel-dispatch layer.
//!
//! H2D/D2H staging, the autograd tape integration, and `vk_tensor.rs`'s
//! lift into `kiln-tensor` proper are subsequent PRs.

use std::any::Any;
use std::sync::Arc;

use kiln_vulkan_kernel::buffer::VulkanBuffer;
use kiln_vulkan_kernel::device::VulkanDevice;

use crate::{DType, Device, Error, Result, StorageBackend};

/// Vulkan-backed storage. Byte-typed; dtype carried alongside for
/// dispatch.
///
/// `VulkanBuffer` is **not** `Clone` — Drop frees the underlying
/// `vk::DeviceMemory`. The expected handle is `Arc<dyn StorageBackend>`
/// from the [`crate::Storage`] alias.
#[derive(Debug)]
pub struct VulkanStorage {
    device: Device,
    dtype: DType,
    buffer: VulkanBuffer,
    /// Cached byte length. We re-record this on construction because
    /// `VulkanBuffer`'s `size: u64` field is private upstream; we know
    /// the value at allocate time and store it to avoid an upstream
    /// API change (which would require touching kiln-vulkan-kernel
    /// outside this PR's scope).
    byte_len: usize,
    vulkan_device: Arc<VulkanDevice>,
}

impl VulkanStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on
    /// `vulkan_device`. Uses
    /// `VulkanBuffer::create_device_local(device.device(),
    /// device.device_local_mem_type(), byte_len)`.
    ///
    /// `device_index` is the Vulkan physical-device index — stored as
    /// the [`Device::Vulkan`] variant. The `vulkan_device` argument's
    /// index is the source of truth; we record it explicitly so the
    /// `device()` accessor is O(1).
    pub fn zeros(
        vulkan_device: Arc<VulkanDevice>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let buffer = VulkanBuffer::create_device_local(
            vulkan_device.device(),
            vulkan_device.device_local_mem_type(),
            byte_len as u64,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "VulkanStorage::zeros: create_device_local({byte_len}) failed: {e}"
            ))
        })?;
        Ok(VulkanStorage {
            device: Device::Vulkan(device_index),
            dtype,
            buffer,
            byte_len,
            vulkan_device,
        })
    }

    /// Wrap an existing `VulkanBuffer` allocated by the caller.
    ///
    /// Validates the buffer length against `dtype.size_in_bytes()`
    /// for non-packed dtypes.
    pub fn from_buffer(
        vulkan_device: Arc<VulkanDevice>,
        device_index: usize,
        dtype: DType,
        buffer: VulkanBuffer,
        size_bytes: u64,
    ) -> Result<Self> {
        let len = size_bytes as usize;
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "VulkanStorage::from_buffer: buffer len {len} is not a multiple of \
                     size_in_bytes({:?}) = {per}",
                    dtype
                )));
            }
        }
        Ok(VulkanStorage {
            device: Device::Vulkan(device_index),
            dtype,
            buffer,
            byte_len: len,
            vulkan_device,
        })
    }

    /// Borrow the underlying VulkanBuffer.
    pub fn buffer(&self) -> &VulkanBuffer {
        &self.buffer
    }

    /// Borrow the VulkanDevice — the queue/dispatch handle the existing
    /// `kiln-vulkan-kernel::kernels::*` entry points consume.
    pub fn vulkan_device(&self) -> &Arc<VulkanDevice> {
        &self.vulkan_device
    }
}

impl StorageBackend for VulkanStorage {
    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a fresh [`crate::Storage`] handle holding a [`VulkanStorage`].
pub fn vulkan_zeros(
    vulkan_device: Arc<VulkanDevice>,
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let storage = VulkanStorage::zeros(vulkan_device, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
}

// ----------------------------------------------------------------------
// vulkan_softmax_last_axis — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan softmax over the trailing axis. Mirrors the role of
/// [`crate::cuda_softmax_last_axis`] for the Vulkan backend.
///
/// Operates on a contiguous `[..., D]` Vulkan-backed tensor; produces a
/// fresh contiguous tensor of the same shape and dtype with each
/// `[..., :]` row normalized to a probability distribution.
///
/// # Implementation
///
/// Delegates to `kiln_vulkan_kernel::vk_ops::softmax::vk_softmax_lastdim_no_grad`,
/// the production F32 softmax kernel (two-pass max → exp+sum → divide).
///
/// The current path bridges between kt's `VulkanStorage` (which owns
/// `VulkanBuffer` directly) and the kernel's `VkTensor` (which holds
/// `Arc<VulkanBuffer>`) via D2H read-back + H2D re-upload at each
/// boundary. The data round-trips through the host even though both
/// sides are GPU-resident — this is functionally correct (kernel runs
/// on-device) but adds a per-call host bounce.
///
/// # Performance follow-up (#1082)
///
/// The cleanest fix is to land a zero-copy bridge, e.g. one of:
///   1. Add `VkTensor::from_kt_storage(&VulkanStorage) -> VkTensor` that
///      shares the underlying `vk::Buffer` handle without copying. Needs
///      an upstream `VkTensor` constructor that accepts a borrowed
///      `VulkanBuffer` (or an `Arc<VulkanBuffer>` cloned from one we own
///      cooperatively).
///   2. Add a kt-side `from_arc_buffer` constructor to `VulkanStorage`
///      so the kernel result's `Arc<VulkanBuffer>` can be wrapped
///      directly. The Arc count survives the kernel call and ownership
///      transfers cleanly back to kt.
///   3. Expose `kiln_vulkan_kernel::vk_ops::softmax::dispatch_softmax_fwd`
///      (currently `pub(crate)`) so kt can dispatch the shader against
///      kt-side `vk::Buffer` handles directly, no `VkTensor` involved.
///
/// All three avoid the H2D+D2H round-trip in this wrapper.
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `x.dtype()` must be `F32` (kernel is F32-only; BF16/F16 needs cast
///   or a widened `VkDType` per the softmax-op TODOs)
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` must hold
///
/// # Errors
///
/// Returns [`Error::Msg`] if the storage isn't `VulkanStorage`, the
/// dtype is unsupported, the layout is non-contiguous, or the
/// underlying kernel call fails.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::softmax::vk_softmax_lastdim_no_grad;
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_softmax_last_axis: unsupported dtype {dtype} (kernel is F32-only; \
             BF16/F16 needs a cast wrapper or widened VkDType — see TODO)"
        )));
    }
    if x.rank() == 0 {
        return Err(Error::Msg(
            "vulkan_softmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_softmax_last_axis: input must be contiguous".to_string(),
        ));
    }

    let kt_vk = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_softmax_last_axis: input must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk.vulkan_device());
    let device_index = match kt_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();
    let byte_len = kt_vk.byte_len();

    // ---- D2H: read kt buffer back to host bytes ----
    let bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: D2H read_back of input failed: {e}"
        ))
    })?;

    // ---- H2D: upload bytes into a fresh VkTensor leaf ----
    let vk_dtype = match dtype {
        DType::F32 => VkDType::F32,
        // Unreachable: gated above. Kept exhaustive for clarity.
        other => {
            return Err(Error::Msg(format!(
                "vulkan_softmax_last_axis: dtype {other} cannot be mapped to VkDType"
            )));
        }
    };
    let vk_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: device-local alloc for VkTensor input failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_buffer,
        &bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: H2D upload of VkTensor input failed: {e}"
        ))
    })?;
    let vk_in = VkTensor::from_buffer(
        Arc::new(vk_buffer),
        shape.clone(),
        vk_dtype,
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch the production Vulkan softmax kernel ----
    let vk_out = vk_softmax_lastdim_no_grad(&vk_in).map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: kernel dispatch failed: {e}"
        ))
    })?;

    // ---- D2H: read kernel result back to host bytes ----
    let out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: D2H read_back of kernel result failed: {e}"
        ))
    })?;

    // ---- H2D: upload result bytes into a fresh kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        byte_len as u64,
    )?;

    let _ = element_count; // shape is the source of truth; element_count kept for symmetry with cuda path

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_rmsnorm_last_axis — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan RMSNorm over the trailing axis. Mirrors the role of
/// [`crate::cuda_rmsnorm_last_axis`] for the Vulkan backend.
///
/// Operates on a contiguous `[..., D]` Vulkan-backed tensor and a
/// contiguous `[D]` Vulkan-backed `weight` tensor; produces a fresh
/// contiguous tensor of the same shape and dtype with each `[..., :]`
/// row normalized by its row-RMS and scaled per-element by `weight`.
///
/// # Implementation
///
/// Delegates to `kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm_no_grad`,
/// the production F32 RMSNorm kernel (matches the QwenRmsNorm shader at
/// `csrc/shaders/qwen_rmsnorm_forward.comp`). The kernel is F32-only;
/// BF16/F16 inputs return an `Error::Msg` here and the op falls through
/// to the CPU path before reaching this wrapper.
///
/// Note: the Qwen shader computes `(1 + w_shader) * x / sqrt(mean(x^2)
/// + eps)`. The standard form (and the kt CPU reference path in
/// `RmsNormOp::cpu_fwd`) is `w * x / sqrt(mean(x^2) + eps)`. To match
/// the standard form, this wrapper pre-subtracts 1.0 from each weight
/// element when staging it for the kernel.
///
/// The current path bridges between kt's `VulkanStorage` (which owns
/// `VulkanBuffer` directly) and the kernel's `VkTensor` (which holds
/// `Arc<VulkanBuffer>`) via D2H read-back + H2D re-upload at each
/// boundary, matching [`vulkan_softmax_last_axis`]. Zero-copy follow-up
/// is the same set of three bridges documented there.
///
/// # Requirements
///
/// - `x` and `weight` must both be backed by [`VulkanStorage`]
/// - `x.dtype() == weight.dtype() == F32`
/// - `x.rank() >= 1`, `weight.rank() == 1`
/// - `weight.shape()[0] == *x.shape().last().unwrap()`
/// - both inputs contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure, storage downcast
/// failure, or kernel dispatch error.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_rmsnorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::rmsnorm::vk_rmsnorm_no_grad;
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: unsupported dtype {dtype} (kernel is F32-only; \
             BF16/F16 needs a cast wrapper or widened VkDType)"
        )));
    }
    if weight.dtype() != dtype {
        return Err(Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: weight dtype {} != x dtype {dtype}",
            weight.dtype()
        )));
    }
    if x.rank() == 0 || weight.rank() != 1 {
        return Err(Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: rank constraints failed (x.rank={}, weight.rank={})",
            x.rank(),
            weight.rank()
        )));
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_rmsnorm_last_axis: inputs must be contiguous".to_string(),
        ));
    }
    let hidden = *x.shape().last().unwrap();
    if weight.shape().first().copied() != Some(hidden) {
        return Err(Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: weight.shape()[0] {:?} != x last-dim {hidden}",
            weight.shape()
        )));
    }

    let kt_vk_x = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_rmsnorm_last_axis: x must be Vulkan-backed".to_string())
        })?;
    let kt_vk_w = weight
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_rmsnorm_last_axis: weight must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk_x.vulkan_device());
    let device_index = match kt_vk_x.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let x_byte_len = kt_vk_x.byte_len();
    let w_byte_len = kt_vk_w.byte_len();

    // ---- D2H x ----
    let x_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk_x.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: D2H read_back of x failed: {e}"
        ))
    })?;

    // ---- D2H weight, then transform `w -> w - 1.0` so the QwenRMSNorm
    // shader's `(1 + w_shader) * ...` semantics matches kt's standard
    // `w * ...` reference (see CPU `RmsNormOp::cpu_fwd`).
    let w_bytes_orig = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk_w.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: D2H read_back of weight failed: {e}"
        ))
    })?;
    let w_bytes_adj: Vec<u8> = {
        // F32-only path (gated above): subtract 1.0 from each element.
        let n = w_bytes_orig.len() / 4;
        let mut out = Vec::with_capacity(w_bytes_orig.len());
        for i in 0..n {
            let chunk = &w_bytes_orig[i * 4..(i + 1) * 4];
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            let adj = v - 1.0_f32;
            out.extend_from_slice(&adj.to_le_bytes());
        }
        out
    };

    // ---- H2D into VkTensors ----
    let vk_dtype = VkDType::F32;

    let vk_x_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        x_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: device-local alloc for VkTensor x failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_x_buffer,
        &x_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: H2D upload of VkTensor x failed: {e}"
        ))
    })?;
    let vk_x = VkTensor::from_buffer(
        Arc::new(vk_x_buffer),
        shape.clone(),
        vk_dtype,
        Arc::clone(&vulkan_device),
    );

    let vk_w_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        w_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: device-local alloc for VkTensor weight failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_w_buffer,
        &w_bytes_adj,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: H2D upload of VkTensor weight failed: {e}"
        ))
    })?;
    let vk_w = VkTensor::from_buffer(
        Arc::new(vk_w_buffer),
        vec![hidden],
        vk_dtype,
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch the production Vulkan RMSNorm kernel ----
    let vk_out = vk_rmsnorm_no_grad(&vk_x, &vk_w, eps).map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: kernel dispatch failed: {e}"
        ))
    })?;

    // ---- D2H kernel result ----
    let out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: D2H read_back of kernel result failed: {e}"
        ))
    })?;

    // ---- H2D into kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        x_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        x_byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_l2norm_last_axis — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan L2-norm over the trailing axis. Mirrors the role of
/// [`crate::cuda_l2norm_last_axis`] for the Vulkan backend.
///
/// Operates on a contiguous `[..., D]` Vulkan-backed tensor and
/// produces a fresh tensor of the same shape with each row L2-normalized:
/// `y = x / sqrt(sum(x^2) + eps)`.
///
/// # Implementation
///
/// Delegates to `kiln_vulkan_kernel::vk_ops::l2norm::vk_l2_norm_lastdim_no_grad`,
/// the production F32 L2-norm shader. Carries `scale = 1.0` (the
/// shader API supports fused-scale for QK-norm call sites; pure L2
/// norm uses 1.0).
///
/// # Limitations
///
/// The shader caps `hidden_dim <= 256` (see `check_l2norm_shape` in
/// `vk_ops/l2norm.rs`). Inputs with a larger trailing dim return
/// `Error::Msg` here and the op falls through to the CPU path before
/// reaching this wrapper.
///
/// Bridges between kt's `VulkanStorage` and `VkTensor` via D2H+H2D
/// round-trip, matching [`vulkan_softmax_last_axis`].
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `x.dtype() == F32`
/// - `x.rank() >= 1`
/// - `x.is_contiguous()`
/// - `*x.shape().last().unwrap() <= 256` (shader limit)
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or kernel error.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_l2norm_last_axis(x: &crate::Tensor, eps: f32) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::l2norm::vk_l2_norm_lastdim_no_grad;
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    let dtype = x.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_l2norm_last_axis: unsupported dtype {dtype} (kernel is F32-only)"
        )));
    }
    if x.rank() == 0 {
        return Err(Error::Msg(
            "vulkan_l2norm_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_l2norm_last_axis: input must be contiguous".to_string(),
        ));
    }
    let hidden = *x.shape().last().unwrap();
    if hidden == 0 || hidden > 256 {
        return Err(Error::Msg(format!(
            "vulkan_l2norm_last_axis: hidden dim {hidden} exceeds shader cap 256"
        )));
    }

    let kt_vk = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_l2norm_last_axis: input must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk.vulkan_device());
    let device_index = match kt_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let byte_len = kt_vk.byte_len();

    // ---- D2H ----
    let bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: D2H read_back failed: {e}"
        ))
    })?;

    // ---- H2D into VkTensor ----
    let vk_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: device-local alloc for VkTensor input failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_buffer,
        &bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: H2D upload of VkTensor input failed: {e}"
        ))
    })?;
    let vk_in = VkTensor::from_buffer(
        Arc::new(vk_buffer),
        shape.clone(),
        VkDType::F32,
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch ----
    let vk_out = vk_l2_norm_lastdim_no_grad(&vk_in, /*scale=*/ 1.0_f32, eps).map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: kernel dispatch failed: {e}"
        ))
    })?;

    // ---- D2H result ----
    let out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: D2H read_back of kernel result failed: {e}"
        ))
    })?;

    // ---- H2D into kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_activation_unary — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan unary activation. Mirrors the role of
/// [`crate::cuda_activation_unary`] for the Vulkan backend.
///
/// Dispatches on `kind_tag` (matches the CUDA tags in
/// `ActivationOp::cuda_fwd`):
///   - `0` -> SiLU  (`x * sigmoid(x)`)
///   - `1` -> Sigmoid (`1 / (1 + exp(-x))`)
///
/// Other kinds (Gelu=2, Tanh=3, Relu=4) currently return
/// `Error::Msg`; the op's `vulkan_fwd` gates on the supported tags and
/// falls through to CPU for the rest. F32-only on Vulkan today; the
/// underlying `vk_silu_no_grad` / `vk_sigmoid_no_grad` shaders are
/// F32-only.
///
/// # Implementation
///
/// Bridges between kt's `VulkanStorage` and the kernel's `VkTensor`
/// via D2H read-back + H2D re-upload at each boundary, matching the
/// softmax / rmsnorm / l2norm wires. Zero-copy follow-up is the same
/// set of three bridges documented on `vulkan_softmax_last_axis`.
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `x.dtype() == F32`
/// - `x.is_contiguous()`
/// - `kind_tag` in {0, 1}
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported kind, dtype, non-contiguous
/// layout, non-Vulkan storage, or kernel error.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_activation_unary(x: &crate::Tensor, kind_tag: i32) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::sigmoid::vk_sigmoid_no_grad;
    use kiln_vulkan_kernel::vk_ops::silu::vk_silu_no_grad;
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    if !matches!(kind_tag, 0 | 1) {
        return Err(Error::Msg(format!(
            "vulkan_activation_unary: kind_tag {kind_tag} not supported on Vulkan today \
             (only 0=Silu, 1=Sigmoid have shaders; Gelu/Tanh/Relu need new kernels)"
        )));
    }
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_activation_unary: unsupported dtype {dtype} (F32-only today; \
             BF16/F16 need cast wrappers or widened VkDType)"
        )));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_activation_unary: input must be contiguous".to_string(),
        ));
    }

    let kt_vk = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_activation_unary: input must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk.vulkan_device());
    let device_index = match kt_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let byte_len = kt_vk.byte_len();

    // ---- D2H ----
    let bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: D2H read_back failed: {e}"
        ))
    })?;

    // ---- H2D into VkTensor ----
    let vk_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: device-local alloc for VkTensor input failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_buffer,
        &bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: H2D upload of VkTensor input failed: {e}"
        ))
    })?;
    let vk_in = VkTensor::from_buffer(
        Arc::new(vk_buffer),
        shape.clone(),
        VkDType::F32,
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch ----
    let vk_out = match kind_tag {
        0 => vk_silu_no_grad(&vk_in),
        1 => vk_sigmoid_no_grad(&vk_in),
        _ => unreachable!("gated above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: kernel dispatch (kind={kind_tag}) failed: {e}"
        ))
    })?;

    // ---- D2H result ----
    let out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: D2H read_back of kernel result failed: {e}"
        ))
    })?;

    // ---- H2D into kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_activation_unary: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_index_select_dim0 — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan index_select along axis 0. Mirrors the role of
/// [`crate::cuda_index_select_dim0`] for the Vulkan backend.
///
/// Dispatches through the production
/// `kiln_vulkan_kernel::vk_ops::embedding::vk_embedding_lookup_f32`
/// shader — the dim0 gather kernel currently exposed for embedding
/// lookup. The shader handles arbitrary `[axis_dim, inner]` 2-D
/// input (collapsing higher-rank input's inner dims), F32 input only.
///
/// Given:
///   - `input: [axis_dim, ...]` (rank >= 1), F32
///   - `indices: rank >= 1`, U32 (or kt's VulkanStorage with U32 bytes)
///
/// Produces a contiguous tensor with shape
/// `[indices.shape, ...input.shape[1..]]` and dtype F32. Multi-dim
/// indices are handled by flattening the index buffer before dispatch
/// and reshaping the kernel's `[num_tokens, hidden]` output back.
///
/// # Implementation
///
/// Bridges between kt's `VulkanStorage` and the kernel's `VkTensor`
/// via D2H read-back + H2D re-upload at each boundary. The indices
/// buffer is re-uploaded as a `VkDType::F32`-placeholder VkTensor
/// whose underlying bytes are the raw U32 token IDs (matches the
/// `upload_u32_ids` helper convention in `vk_ops::embedding`). The
/// shader interprets the buffer as `u32[]` regardless of placeholder
/// dtype.
///
/// # Requirements
///
/// - `input` and `indices` must both be backed by [`VulkanStorage`]
/// - `input.dtype() == F32`
/// - `indices.dtype() == U32`
/// - `input.rank() >= 1`, `indices.rank() >= 1`
/// - both contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported dtype, non-contiguous layout,
/// non-Vulkan storage, or kernel error.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_index_select_dim0(
    input: &crate::Tensor,
    indices: &crate::Tensor,
) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::embedding::vk_embedding_lookup_f32;
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    // ---- Validate kt-side preconditions ----
    let dtype = input.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_index_select_dim0: unsupported input dtype {dtype} \
             (only F32 has a Vulkan shader today; BF16 has `vk_embedding_lookup_bf16` \
              but emits F32 output — needs a separate wrapper path)"
        )));
    }
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "vulkan_index_select_dim0: indices dtype must be U32 (got {})",
            indices.dtype()
        )));
    }
    if input.rank() == 0 || indices.rank() == 0 {
        return Err(Error::Msg(format!(
            "vulkan_index_select_dim0: rank constraints failed \
             (input.rank={}, indices.rank={})",
            input.rank(),
            indices.rank()
        )));
    }
    if !input.is_contiguous() || !indices.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_index_select_dim0: inputs must be contiguous".to_string(),
        ));
    }

    let kt_vk_in = input
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_index_select_dim0: input must be Vulkan-backed".to_string())
        })?;
    let kt_vk_ids = indices
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_index_select_dim0: indices must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk_in.vulkan_device());
    let device_index = match kt_vk_in.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let in_shape: Vec<usize> = input.shape().to_vec();
    let vocab_size = in_shape[0];
    let hidden: usize = in_shape[1..].iter().product();
    let in_byte_len = kt_vk_in.byte_len();
    let ids_byte_len = kt_vk_ids.byte_len();
    let n_indices = indices.element_count();

    // ---- D2H input weights ----
    let in_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk_in.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: D2H read_back of input failed: {e}"
        ))
    })?;

    // ---- D2H ids ----
    let ids_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk_ids.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: D2H read_back of ids failed: {e}"
        ))
    })?;

    // ---- H2D input as F32 VkTensor with shape [vocab, hidden] (flatten inner) ----
    let weight_shape = vec![vocab_size, hidden];
    let vk_in_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        in_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: device-local alloc for VkTensor input failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_in_buffer,
        &in_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: H2D upload of VkTensor input failed: {e}"
        ))
    })?;
    let vk_in = VkTensor::from_buffer(
        Arc::new(vk_in_buffer),
        weight_shape,
        VkDType::F32,
        Arc::clone(&vulkan_device),
    );

    // ---- H2D ids as F32-placeholder VkTensor (buffer holds raw u32 bytes)
    // The vk_embedding_lookup_f32 kernel reads the ids buffer as u32[]
    // regardless of placeholder dtype — same convention as `upload_u32_ids`.
    let vk_ids_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        ids_byte_len.max(4) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: device-local alloc for VkTensor ids failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_ids_buffer,
        &ids_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: H2D upload of VkTensor ids failed: {e}"
        ))
    })?;
    let vk_ids = VkTensor::from_buffer(
        Arc::new(vk_ids_buffer),
        vec![n_indices],
        VkDType::F32, // placeholder; buffer holds u32 bytes per upload_u32_ids convention
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch ----
    let vk_out = vk_embedding_lookup_f32(&vk_in, &vk_ids, vocab_size, hidden).map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: vk_embedding_lookup_f32 dispatch failed: {e}"
        ))
    })?;

    // ---- D2H result ----
    let out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: D2H read_back of kernel result failed: {e}"
        ))
    })?;

    // ---- Compute final kt output shape ----
    // Final shape = indices.shape ++ input.shape[1..]
    let mut out_shape: Vec<usize> = indices.shape().to_vec();
    out_shape.extend_from_slice(&in_shape[1..]);
    let out_byte_len = n_indices * hidden * 4; // F32 only

    // Truncate any trailing padding (in case the kernel's output buffer
    // is over-allocated; mirrors the safety guard in vulkan_cast).
    let trimmed_bytes = if out_bytes.len() > out_byte_len {
        out_bytes[..out_byte_len].to_vec()
    } else if out_bytes.len() < out_byte_len {
        return Err(Error::Msg(format!(
            "vulkan_index_select_dim0: kernel produced {} bytes, expected {}",
            out_bytes.len(),
            out_byte_len
        )));
    } else {
        out_bytes
    };

    // ---- H2D into kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        out_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &trimmed_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_index_select_dim0: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        out_byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_cast — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan dtype cast (F32 <-> BF16). Mirrors the role of
/// [`crate::cuda_cast`] for the Vulkan backend.
///
/// Dispatches through the production
/// `kiln_vulkan_kernel::vk_ops::cast::vk_cast_f32_to_bf16_no_grad`
/// and `vk_cast_bf16_to_f32_no_grad` shaders. Only the F32 <-> BF16
/// pair has shaders today; F16 round-trips, integer casts, and
/// same-dtype no-ops are all rejected here and the op's `vulkan_fwd`
/// gates accordingly.
///
/// # Implementation
///
/// Bridges between kt's `VulkanStorage` and the kernel's `VkTensor`
/// via D2H read-back + H2D re-upload at each boundary, matching the
/// softmax / rmsnorm / l2norm / activation / elementwise wires.
/// Zero-copy follow-up is the same set of three bridges documented on
/// `vulkan_softmax_last_axis`.
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `(x.dtype(), to)` in {(F32, BF16), (BF16, F32)}
/// - `x.is_contiguous()`
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported dtype pair, non-contiguous
/// layout, non-Vulkan storage, or kernel error.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_cast(x: &crate::Tensor, to: DType) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::cast::{
        vk_cast_bf16_to_f32_no_grad, vk_cast_f32_to_bf16_no_grad,
    };
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    let from = x.dtype();
    let (vk_from, vk_to) = match (from, to) {
        (DType::F32, DType::BF16) => (VkDType::F32, VkDType::Bf16),
        (DType::BF16, DType::F32) => (VkDType::Bf16, VkDType::F32),
        _ => {
            return Err(Error::Msg(format!(
                "vulkan_cast: dtype pair {from} -> {to} not supported \
                 (only F32 <-> BF16 has Vulkan shaders today)"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_cast: input must be contiguous".to_string(),
        ));
    }

    let kt_vk = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_cast: input must be Vulkan-backed".to_string()))?;

    let vulkan_device = Arc::clone(kt_vk.vulkan_device());
    let device_index = match kt_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let n_elements = x.element_count();
    let in_byte_len = kt_vk.byte_len();
    let out_per = to.size_in_bytes();
    let out_byte_len = n_elements * out_per;

    // ---- D2H ----
    let in_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk.buffer(),
    )
    .map_err(|e| Error::Msg(format!("vulkan_cast: D2H read_back failed: {e}")))?;

    // ---- H2D into VkTensor (source dtype) ----
    let vk_in_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        in_byte_len.max(1) as u64,
    )
    .map_err(|e| Error::Msg(format!("vulkan_cast: device-local alloc for VkTensor failed: {e}")))?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_in_buffer,
        &in_bytes,
    )
    .map_err(|e| Error::Msg(format!("vulkan_cast: H2D upload of VkTensor input failed: {e}")))?;
    let vk_in = VkTensor::from_buffer(
        Arc::new(vk_in_buffer),
        shape.clone(),
        vk_from,
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch ----
    let vk_out = match (vk_from, vk_to) {
        (VkDType::F32, VkDType::Bf16) => vk_cast_f32_to_bf16_no_grad(&vk_in),
        (VkDType::Bf16, VkDType::F32) => vk_cast_bf16_to_f32_no_grad(&vk_in),
        _ => unreachable!("gated above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_cast: kernel dispatch ({vk_from:?} -> {vk_to:?}) failed: {e}"
        ))
    })?;

    // ---- D2H result. The output buffer may be padded for u32 alignment
    // when the target is BF16 — only the leading `out_byte_len` bytes
    // are the actual elements; the trailing padding (if any) is unused.
    let mut out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| Error::Msg(format!("vulkan_cast: D2H read_back of result failed: {e}")))?;
    if out_bytes.len() > out_byte_len {
        out_bytes.truncate(out_byte_len);
    } else if out_bytes.len() < out_byte_len {
        return Err(Error::Msg(format!(
            "vulkan_cast: kernel produced {} bytes, expected at least {}",
            out_bytes.len(),
            out_byte_len
        )));
    }

    // ---- H2D into kt VulkanStorage (target dtype) ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        out_byte_len.max(1) as u64,
    )
    .map_err(|e| Error::Msg(format!("vulkan_cast: device-local alloc for kt output failed: {e}")))?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| Error::Msg(format!("vulkan_cast: H2D upload of kt output failed: {e}")))?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        to,
        out_buffer,
        out_byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_elementwise_binary — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan element-wise binary op (add/sub/mul/div). Mirrors the role of
/// [`crate::cuda_elementwise_binary`] for the Vulkan backend.
///
/// Dispatches on `kind_tag` (matches the CUDA tags in
/// `ElementwiseOp::cuda_fwd`):
///   - `0` -> Add  (`a + b`)
///   - `1` -> Sub  (`a - b`)
///   - `2` -> Mul  (`a * b`)
///   - `3` -> Div  (`a / b`)
///
/// F32-only on Vulkan today; the underlying
/// `vk_elementwise_binary_f32` shader (and its `_offset` tiled variant)
/// supports F32 only. BF16/F16 fall through to CPU until matching
/// shaders land.
///
/// # Implementation
///
/// Bridges between kt's `VulkanStorage` and the kernel's `VkTensor`
/// via D2H read-back + H2D re-upload at each boundary, matching the
/// softmax / rmsnorm / l2norm wires. Zero-copy follow-up is the same
/// set of three bridges documented on `vulkan_softmax_last_axis`.
///
/// # Requirements
///
/// - `a` and `b` must both be backed by [`VulkanStorage`]
/// - `a.dtype() == b.dtype() == F32`
/// - `a.shape() == b.shape()` (no broadcasting yet)
/// - both contiguous
/// - `kind_tag` in {0, 1, 2, 3}
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported kind, dtype, non-contiguous
/// layout, non-Vulkan storage, shape mismatch, or kernel error.
#[allow(clippy::needless_range_loop)]
pub fn vulkan_elementwise_binary(
    a: &crate::Tensor,
    b: &crate::Tensor,
    kind_tag: i32,
) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::elementwise::{
        vk_add_no_grad, vk_div_no_grad, vk_mul_no_grad, vk_sub_no_grad,
    };
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    if !matches!(kind_tag, 0 | 1 | 2 | 3) {
        return Err(Error::Msg(format!(
            "vulkan_elementwise_binary: kind_tag {kind_tag} not supported \
             (only 0=Add, 1=Sub, 2=Mul, 3=Div have shaders)"
        )));
    }
    let dtype = a.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_elementwise_binary: unsupported dtype {dtype} (F32-only today; \
             BF16/F16 need cast wrappers or widened shaders)"
        )));
    }
    if b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "vulkan_elementwise_binary: dtype mismatch a={dtype} b={}",
            b.dtype()
        )));
    }
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "vulkan_elementwise_binary: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_elementwise_binary: inputs must be contiguous".to_string(),
        ));
    }

    let kt_vk_a = a
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_elementwise_binary: a must be Vulkan-backed".to_string())
        })?;
    let kt_vk_b = b
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_elementwise_binary: b must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk_a.vulkan_device());
    let device_index = match kt_vk_a.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };
    let shape: Vec<usize> = a.shape().to_vec();
    let a_byte_len = kt_vk_a.byte_len();
    let b_byte_len = kt_vk_b.byte_len();

    // ---- D2H a ----
    let a_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk_a.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: D2H read_back of a failed: {e}"
        ))
    })?;

    // ---- D2H b ----
    let b_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk_b.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: D2H read_back of b failed: {e}"
        ))
    })?;

    // ---- H2D into VkTensors ----
    let vk_dtype = VkDType::F32;

    let vk_a_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        a_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: device-local alloc for VkTensor a failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_a_buffer,
        &a_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: H2D upload of VkTensor a failed: {e}"
        ))
    })?;
    let vk_a = VkTensor::from_buffer(
        Arc::new(vk_a_buffer),
        shape.clone(),
        vk_dtype,
        Arc::clone(&vulkan_device),
    );

    let vk_b_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        b_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: device-local alloc for VkTensor b failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_b_buffer,
        &b_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: H2D upload of VkTensor b failed: {e}"
        ))
    })?;
    let vk_b = VkTensor::from_buffer(
        Arc::new(vk_b_buffer),
        shape.clone(),
        vk_dtype,
        Arc::clone(&vulkan_device),
    );

    // ---- Dispatch ----
    let vk_out = match kind_tag {
        0 => vk_add_no_grad(&vk_a, &vk_b),
        1 => vk_sub_no_grad(&vk_a, &vk_b),
        2 => vk_mul_no_grad(&vk_a, &vk_b),
        3 => vk_div_no_grad(&vk_a, &vk_b),
        _ => unreachable!("gated above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: kernel dispatch (kind={kind_tag}) failed: {e}"
        ))
    })?;

    // ---- D2H result ----
    let out_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk_out.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: D2H read_back of kernel result failed: {e}"
        ))
    })?;

    // ---- H2D into kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        a_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_elementwise_binary: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        a_byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_argmax_last_axis — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan argmax over the trailing axis. Mirrors the role of
/// [`crate::cuda_argmax_last_axis`] for the Vulkan backend.
///
/// Operates on a contiguous `[..., D]` Vulkan-backed tensor of dtype
/// F32 / BF16 / F16 and produces a fresh contiguous I64-typed
/// Vulkan-backed tensor of shape `[...]` (trailing axis dropped).
/// Tie-break is lowest-index-wins, matching the CPU and CUDA paths.
///
/// # Implementation: D2H + CPU compute + H2D bridge
///
/// `kiln-vulkan-kernel` does not currently expose a generic
/// `argmax_last_dim` SPIR-V shader (the `linear_decode_argmax_*`
/// family is the fused matmul-then-argmax greedy decoder, not a
/// drop-in reduction kernel). Until that shader lands, this wrapper
/// ships the same D2H-read + CPU-compute + H2D-upload bridge used by
/// [`vulkan_softmax_last_axis`]'s pre-kernel staging path — the
/// storage is GPU-resident on both sides of the call, but the
/// reduction itself runs on the host.
///
/// This is functionally identical to the `Ok(None)` fallback (the
/// dispatcher would route to CPU) but keeps the storage round-trip
/// fully visible at the dispatch site instead of silently dropping
/// off the Vulkan path mid-graph. Once a real SPIR-V argmax kernel
/// lands in `kiln_vulkan_kernel::vk_ops::reduce` (or a new
/// `vk_ops::argmax` module), swap the host-side scan below for a
/// `dispatch_simple(...)` call — the surrounding D2H/H2D scaffolding
/// can stay or shrink to a zero-copy bridge per the softmax wrapper's
/// rustdoc.
///
/// # Performance follow-up (#1082)
///
/// See [`vulkan_softmax_last_axis`] for the three zero-copy bridges
/// proposed for the broader kt <-> kiln-vulkan-kernel seam. Applying
/// any of them here removes the round-trip; replacing the inner CPU
/// scan with a SPIR-V dispatch removes the host compute.
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16`
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` must hold
///
/// # Errors
///
/// Returns [`Error::Msg`] if the storage isn't `VulkanStorage`, the
/// dtype is unsupported, the layout is non-contiguous, the rank is 0,
/// or the underlying buffer transfer fails.
pub fn vulkan_argmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "vulkan_argmax_last_axis: unsupported dtype {dtype} (F32/BF16/F16 only)"
        )));
    }
    if x.rank() == 0 {
        return Err(Error::Msg(
            "vulkan_argmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_argmax_last_axis: input must be contiguous".to_string(),
        ));
    }

    let kt_vk = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_argmax_last_axis: input must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk.vulkan_device());
    let device_index = match kt_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let hidden = *shape.last().unwrap();
    let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

    // ---- D2H: read kt buffer back to host bytes ----
    let in_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_vk.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_argmax_last_axis: D2H read_back of input failed: {e}"
        ))
    })?;

    // ---- Host-side argmax over each row (lowest-index tie-break) ----
    let mut out_indices: Vec<i64> = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..hidden {
            let v = read_one_f32(dtype, &in_bytes, r * hidden + i);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
            // Tie: keep best_idx (lowest-index-wins).
        }
        out_indices.push(best_idx as i64);
    }
    let out_bytes: Vec<u8> = out_indices
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    let out_byte_len = out_bytes.len();

    // ---- H2D: upload I64 result bytes into a fresh kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        out_byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_argmax_last_axis: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_argmax_last_axis: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        DType::I64,
        out_buffer,
        out_byte_len as u64,
    )?;

    // Output shape drops the trailing axis. Rank-1 input -> rank-0 output.
    let out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

fn read_one_f32(dtype: DType, bytes: &[u8], i: usize) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
        }
        _ => unreachable!("vulkan_argmax_last_axis: read_one_f32 called with non-float dtype"),
    }
}

// ----------------------------------------------------------------------
// vulkan_masked_fill — Phase 4 Vulkan substrate op (#1082)
// ----------------------------------------------------------------------

/// Vulkan masked-fill. Mirrors the role of [`crate::cuda_masked_fill`]
/// for the Vulkan backend.
///
/// `out[i] = (mask[i] != 0) ? fill_value : x[i]` over a contiguous
/// `x` with shape `S` and dtype F32 / BF16 / F16, and a contiguous
/// `mask` with shape `S` and dtype U8. `fill_value` is `f32` and cast
/// to `x`'s dtype on store.
///
/// # Implementation: D2H + CPU compute + H2D bridge
///
/// `kiln-vulkan-kernel::vk_ops::mask` currently exposes
/// `vk_causal_mask_inplace` (additive `-1e30` for `k > q + offset`)
/// and `vk_scale_inplace`, but no generic `masked_fill` with a U8
/// buffer + arbitrary `fill_value`. Until that shader lands, this
/// wrapper ships the same D2H-read + CPU-compute + H2D-upload bridge
/// used by [`vulkan_softmax_last_axis`]'s pre-kernel staging path —
/// the storage is GPU-resident on both sides of the call, but the
/// pointwise where-style ternary runs on the host.
///
/// This is functionally identical to the `Ok(None)` fallback (the
/// dispatcher would route to CPU) but keeps the storage round-trip
/// visible at the dispatch site instead of silently dropping off the
/// Vulkan path mid-graph. Once a real SPIR-V `masked_fill` shader
/// lands in `kiln_vulkan_kernel::vk_ops::mask`, swap the host-side
/// pointwise loop below for a `dispatch_simple(...)` call — the
/// surrounding D2H/H2D scaffolding can stay or shrink to a zero-copy
/// bridge per the softmax wrapper's rustdoc.
///
/// # Performance follow-up (#1082)
///
/// See [`vulkan_softmax_last_axis`] for the three zero-copy bridges
/// proposed for the broader kt <-> kiln-vulkan-kernel seam. Applying
/// any of them here removes the round-trip; replacing the inner CPU
/// loop with a SPIR-V dispatch removes the host compute. The shader
/// itself is the simplest in `vk_ops::` — pure pointwise, one
/// work-item per element, no reductions, no shared memory.
///
/// # Requirements
///
/// - `x` and `mask` must be backed by [`VulkanStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16`
/// - `mask.dtype()` must be `U8`
/// - `x.shape() == mask.shape()`
/// - both must be contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] if any input isn't `VulkanStorage`, dtypes
/// are unsupported, shapes mismatch, layouts are non-contiguous, or
/// the underlying buffer transfer fails.
pub fn vulkan_masked_fill(
    x: &crate::Tensor,
    mask: &crate::Tensor,
    fill_value: f32,
) -> Result<crate::Tensor> {
    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    if x.shape() != mask.shape() {
        return Err(Error::Msg(format!(
            "vulkan_masked_fill: shape mismatch x={:?} mask={:?}",
            x.shape(),
            mask.shape()
        )));
    }
    if mask.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "vulkan_masked_fill: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "vulkan_masked_fill: unsupported x dtype {dtype} (F32/BF16/F16 only)"
        )));
    }
    if !x.is_contiguous() || !mask.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_masked_fill: x and mask must be contiguous".to_string(),
        ));
    }

    let kt_x = x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_masked_fill: x must be Vulkan-backed".to_string())
        })?;
    let kt_mask = mask
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_masked_fill: mask must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_x.vulkan_device());
    let device_index = match kt_x.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let n = x.element_count();
    let per = dtype.size_in_bytes();
    let byte_len = kt_x.byte_len();

    // ---- D2H: read kt x + mask buffers back to host bytes ----
    let x_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_x.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_masked_fill: D2H read_back of x failed: {e}"
        ))
    })?;
    let m_bytes = kiln_vulkan_kernel::buffer::VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        kt_mask.buffer(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_masked_fill: D2H read_back of mask failed: {e}"
        ))
    })?;

    // ---- Host-side pointwise: where(mask != 0, fill_value, x) ----
    // Output dtype matches input dtype; fill_value casts on store.
    let mut out_bytes = vec![0u8; n * per];
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let v = if m_bytes[i] != 0 {
                    fill_value
                } else {
                    f32::from_le_bytes(x_bytes[i * 4..i * 4 + 4].try_into().unwrap())
                };
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let v = if m_bytes[i] != 0 {
                    fill_value
                } else {
                    half::bf16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32()
                };
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let v = if m_bytes[i] != 0 {
                    fill_value
                } else {
                    half::f16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32()
                };
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!("vulkan_masked_fill: dtype gated above"),
    }

    // ---- H2D: upload result bytes into a fresh kt VulkanStorage ----
    let out_buffer = kiln_vulkan_kernel::buffer::VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        byte_len.max(1) as u64,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_masked_fill: device-local alloc for kt output failed: {e}"
        ))
    })?;
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &out_buffer,
        &out_bytes,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_masked_fill: H2D upload of kt output failed: {e}"
        ))
    })?;
    let out_storage = VulkanStorage::from_buffer(
        vulkan_device,
        device_index,
        dtype,
        out_buffer,
        byte_len as u64,
    )?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vulkan_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_vulkan_device() -> Option<Arc<VulkanDevice>> {
        if !vulkan_test_enabled() {
            return None;
        }
        VulkanDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn zeros_constructs() {
        let Some(dev) = maybe_vulkan_device() else {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        };
        let storage = VulkanStorage::zeros(dev, 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Vulkan(0));
        assert_eq!(storage.dtype(), DType::BF16);
    }
}
