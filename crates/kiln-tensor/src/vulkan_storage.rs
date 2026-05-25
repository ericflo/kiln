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
