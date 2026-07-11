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
/// `vk::DeviceMemory`. We therefore hold it behind an `Arc` so the
/// buffer's `Drop` (which frees the `vk::DeviceMemory` exactly once)
/// fires when the last handle is gone, and so a kernel result's
/// `Arc<VulkanBuffer>` can be wrapped directly via
/// [`VulkanStorage::from_arc_buffer`] with no D2H/H2D bounce. The
/// outer handle is `Arc<dyn StorageBackend>` from the [`crate::Storage`]
/// alias.
#[derive(Debug)]
pub struct VulkanStorage {
    device: Device,
    dtype: DType,
    buffer: Arc<VulkanBuffer>,
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
        let alloc_len = byte_len.max(1);
        let buffer = VulkanBuffer::create_device_local(
            vulkan_device.device(),
            vulkan_device.device_local_mem_type(),
            alloc_len as u64,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "VulkanStorage::zeros: create_device_local({alloc_len}) failed: {e}"
            ))
        })?;
        if byte_len > 0 {
            let zeros = vec![0u8; byte_len];
            VulkanBuffer::upload_data(
                vulkan_device.device(),
                vulkan_device.host_visible_mem_type(),
                vulkan_device.queue(),
                vulkan_device.queue_family_index(),
                &buffer,
                &zeros,
            )
            .map_err(|e| {
                Error::Msg(format!("VulkanStorage::zeros: H2D zero upload failed: {e}"))
            })?;
        }
        Ok(VulkanStorage {
            device: Device::Vulkan(device_index),
            dtype,
            buffer: Arc::new(buffer),
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
        Self::from_arc_buffer(
            vulkan_device,
            device_index,
            dtype,
            Arc::new(buffer),
            size_bytes,
        )
    }

    /// Wrap an existing `Arc<VulkanBuffer>` (e.g. a kernel result that
    /// already lives behind an `Arc`) **without** copying the device
    /// memory. The `Arc` refcount is bumped; the underlying
    /// `vk::DeviceMemory` is freed exactly once when the last clone of
    /// this `Arc` is dropped.
    ///
    /// This is the zero-copy bridge between the kernel layer
    /// (`VkTensor` holds `Arc<VulkanBuffer>`) and kt storage referenced
    /// by the `vulkan_softmax_last_axis` TODO. Validates the buffer
    /// length against `dtype.size_in_bytes()` for non-packed dtypes.
    pub fn from_arc_buffer(
        vulkan_device: Arc<VulkanDevice>,
        device_index: usize,
        dtype: DType,
        buffer: Arc<VulkanBuffer>,
        size_bytes: u64,
    ) -> Result<Self> {
        let len = size_bytes as usize;
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "VulkanStorage::from_arc_buffer: buffer len {len} is not a multiple of \
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

    /// Clone the underlying `Arc<VulkanBuffer>` handle (refcount bump,
    /// no device copy) — for zero-copy bridges into the kernel layer.
    pub fn buffer_arc(&self) -> Arc<VulkanBuffer> {
        Arc::clone(&self.buffer)
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
// Zero-copy VulkanStorage <-> VkTensor bridge (#1082 PR3b)
// ----------------------------------------------------------------------
//
// These two helpers replace the documented D2H + H2D host bounce in the
// `vulkan_*_last_axis` wrappers above with an `Arc<VulkanBuffer>` refcount
// bump in each direction. The kernel layer (`VkTensor`) and kt storage
// (`VulkanStorage`) both hold the buffer behind an `Arc`, so handing the
// buffer across the boundary is a pointer share, not a copy — the device
// memory is never staged through the host.
//
// (Types referenced by fully-qualified path so this module-level code does
// not clash with the function-local `use ...VkTensor` imports below.)

/// Map a kt [`DType`] to the kernel's `VkDType`. Returns `None` for
/// dtypes the Vulkan kernel layer cannot represent (anything but F32 /
/// BF16 today); callers gate on this and fall through to the host path.
fn kt_dtype_to_vk(dtype: DType) -> Option<kiln_vulkan_kernel::vk_tensor::VkDType> {
    use kiln_vulkan_kernel::vk_tensor::VkDType;
    match dtype {
        DType::F32 => Some(VkDType::F32),
        DType::BF16 => Some(VkDType::Bf16),
        _ => None,
    }
}

/// Extract the `Device::Vulkan(i)` ordinal of a Vulkan-backed tensor, or
/// error with `op`'s name if `t` is not [`VulkanStorage`]-backed. Used by the
/// zero-copy wrappers to stamp the output storage's device ordinal.
fn vulkan_device_index(t: &crate::Tensor, op: &str) -> Result<usize> {
    match t
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg(format!("{op}: input must be Vulkan-backed")))?
        .device()
    {
        Device::Vulkan(i) => Ok(i),
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    }
}

/// Map a kernel `VkDType` back to the kt [`DType`].
fn vk_dtype_to_kt(dtype: kiln_vulkan_kernel::vk_tensor::VkDType) -> DType {
    use kiln_vulkan_kernel::vk_tensor::VkDType;
    match dtype {
        VkDType::F32 => DType::F32,
        VkDType::Bf16 => DType::BF16,
    }
}

/// Zero-copy bridge: view a Vulkan-backed kt [`crate::Tensor`] as a
/// `VkTensor` for the kernel layer, sharing the underlying
/// `vk::DeviceMemory` (no D2H/H2D). The `Arc<VulkanBuffer>` refcount is
/// bumped; the device memory is freed exactly once when the last handle
/// on either side drops.
///
/// # Requirements
///
/// - `t` must be backed by [`VulkanStorage`]
/// - `t.is_contiguous()` and `t.layout().start_offset() == 0` (the
///   `VkTensor` model is always whole-buffer C-contiguous; a strided or
///   offset view does not share a clean buffer image)
/// - `t.dtype()` must map to a `VkDType` (F32/BF16)
///
/// Shape and dtype are preserved exactly. The `VulkanBuffer`'s physical
/// allocation may be pool-bucket-rounded larger than the logical element
/// range; the kernel only addresses `shape * dtype` elements, so the
/// rounding is harmless here.
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, non-contiguous / offset
/// layout, or unsupported dtype.
pub fn vk_tensor_from_kt(t: &crate::Tensor) -> Result<kiln_vulkan_kernel::vk_tensor::VkTensor> {
    use kiln_vulkan_kernel::vk_tensor::VkTensor;
    let kt_vk = t
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vk_tensor_from_kt: tensor must be Vulkan-backed".to_string()))?;
    if !t.is_contiguous() {
        return Err(Error::Msg(
            "vk_tensor_from_kt: tensor must be contiguous".to_string(),
        ));
    }
    if t.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "vk_tensor_from_kt: tensor must have start_offset == 0 (VkTensor is whole-buffer)"
                .to_string(),
        ));
    }
    let vk_dtype = kt_dtype_to_vk(t.dtype()).ok_or_else(|| {
        Error::Msg(format!(
            "vk_tensor_from_kt: dtype {} has no VkDType mapping (F32/BF16 only)",
            t.dtype()
        ))
    })?;
    // Zero-copy: clone the Arc handle (refcount bump, no device copy).
    let buffer = kt_vk.buffer_arc();
    let device = Arc::clone(kt_vk.vulkan_device());
    Ok(VkTensor::from_buffer(
        buffer,
        t.shape().to_vec(),
        vk_dtype,
        device,
    ))
}

/// Zero-copy bridge: wrap a kernel `VkTensor` result as a Vulkan-backed kt
/// [`crate::Tensor`], sharing the underlying `vk::DeviceMemory` (no
/// D2H/H2D).
///
/// CRITICAL: the kt-side logical byte length recorded on the
/// [`VulkanStorage`] is `n_elements * dtype.size_in_bytes()`, computed
/// from the `VkTensor`'s shape + dtype — **not** the `VulkanBuffer`'s
/// physical allocation size, which the kernel buffer pool bucket-rounds
/// up (to >= 64 KiB). Recording the rounded size would mis-report
/// `byte_len()` and corrupt any downstream byte-range slice / readback.
///
/// `device_index` is the [`Device::Vulkan`] ordinal to stamp on the
/// resulting storage (the `VkTensor` does not carry one; callers pass the
/// ordinal of the inputs they bridged from).
///
/// # Errors
///
/// Returns [`Error::Msg`] if the resulting [`crate::Tensor`] cannot be
/// assembled.
pub fn kt_tensor_from_vk(
    vk: &kiln_vulkan_kernel::vk_tensor::VkTensor,
    device_index: usize,
) -> Result<crate::Tensor> {
    let dtype = vk_dtype_to_kt(vk.dtype());
    let shape = vk.shape().to_vec();
    let n_elements: usize = shape.iter().product();
    // Logical byte length — NOT VulkanBuffer::size() (pool-rounded).
    let byte_len = n_elements * dtype.size_in_bytes();
    let device = Arc::clone(vk.device());
    // Zero-copy: clone the Arc handle (refcount bump, no device copy).
    let buffer = Arc::clone(vk.buffer());
    let storage =
        VulkanStorage::from_arc_buffer(device, device_index, dtype, buffer, byte_len as u64)?;
    let storage_arc: crate::Storage = Arc::new(storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// vulkan_matmul — Phase 3b perf-critical GEMM (#1082)
// ----------------------------------------------------------------------

/// Vulkan rank-2 F32 GEMM `[M, K] @ [K, N] = [M, N]`. Mirrors the role of
/// [`crate::cuda_matmul`] / [`crate::metal_matmul`] for the Vulkan backend.
///
/// Bridges both inputs to `VkTensor` zero-copy (see [`vk_tensor_from_kt`]),
/// dispatches the production
/// `kiln_vulkan_kernel::vk_ops::matmul::vk_matmul_no_grad` shader, and
/// bridges the result back zero-copy (see [`kt_tensor_from_vk`]). No D2H /
/// H2D round-trip — the data stays GPU-resident end to end.
///
/// # Requirements
///
/// - `a` and `b` both backed by [`VulkanStorage`] (same device)
/// - `a.dtype() == b.dtype() == F32` (the kernel is F32-only today)
/// - both rank-2 and contiguous, with matching contraction dim
///
/// Callers (`MatmulOp::vulkan_fwd`) gate these preconditions and return
/// `Ok(None)` (host fallback) for anything this kernel does not cover
/// (batched / higher-rank, BF16/F16, non-contiguous). This function still
/// validates and errors loudly if called with an unsupported shape.
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, dtype/shape violations,
/// or kernel dispatch failure.
pub fn vulkan_matmul(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul_no_grad;

    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "vulkan_matmul: F32-only kernel (got a={}, b={})",
            a.dtype(),
            b.dtype()
        )));
    }
    if a.rank() != 2 || b.rank() != 2 {
        return Err(Error::Msg(format!(
            "vulkan_matmul: rank-2 only (got a.rank={}, b.rank={})",
            a.rank(),
            b.rank()
        )));
    }

    let device_index = match a
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_matmul: a must be Vulkan-backed".to_string()))?
        .device()
    {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let vk_a = vk_tensor_from_kt(a)?;
    let vk_b = vk_tensor_from_kt(b)?;
    let vk_out = vk_matmul_no_grad(&vk_a, &vk_b)
        .map_err(|e| Error::Msg(format!("vulkan_matmul: kernel dispatch failed: {e}")))?;
    kt_tensor_from_vk(&vk_out, device_index)
}

// ----------------------------------------------------------------------
// vulkan_matmul_batched — batched (rank ≥ 3) F32 GEMM, fully resident
// ----------------------------------------------------------------------

/// Vulkan batched F32 GEMM: `a:[..., M, K] @ b:[..., K, N] = [..., M, N]`,
/// where the leading axes (everything but the trailing two) form the batch.
///
/// This is the rank ≥ 3 companion to [`vulkan_matmul`]. The attention core
/// (`Q·Kᵀ` and `scores·V`, typically rank-4 `[B, H, S, D]`) and several GDN
/// composites issue batched matmuls; without this the kt
/// [`crate::ops::matmul::MatmulOp::vulkan_fwd`] returned `Ok(None)` for
/// rank > 2 and `dispatch2` ran the GEMM on the **CPU**, then copied the
/// result back to Vulkan — a host round-trip on the inference hot path.
///
/// The leading batch axes are flattened to a single dim so the underlying
/// rank-3 kernel (`vk_ops::matmul_batched::vk_matmul_batched_no_grad`,
/// already proven in the GDN chunkwise paths) can run every batch slice
/// independently; the result is reshaped back to the caller's leading dims.
/// Both bridge legs are zero-copy (shared `vk::DeviceMemory`); no D2H/H2D.
///
/// # Requirements
///
/// - `a`, `b` both [`VulkanStorage`]-backed on the same device, F32,
///   contiguous, `start_offset == 0`, equal rank ≥ 3
/// - matching leading (batch) axes and contraction dim
///
/// Callers ([`MatmulOp::vulkan_fwd`]) gate these and fall back to the host
/// reference otherwise. This function re-validates and errors loudly on a
/// shape/dtype violation.
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, dtype/rank/shape
/// violations, or kernel dispatch failure.
pub fn vulkan_matmul_batched(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul_batched::{
        vk_matmul_batched_bf16_no_grad, vk_matmul_batched_no_grad,
    };
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16) || b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "vulkan_matmul_batched: F32/BF16 equal-dtype kernel (got a={}, b={})",
            a.dtype(),
            b.dtype()
        )));
    }
    let vk_dtype = match dtype {
        DType::F32 => VkDType::F32,
        DType::BF16 => VkDType::Bf16,
        _ => unreachable!("dtype gated to F32/BF16 above"),
    };
    let (ar, br) = (a.rank(), b.rank());
    if ar < 3 || br != ar {
        return Err(Error::Msg(format!(
            "vulkan_matmul_batched: rank ≥ 3 and equal ranks required (a.rank={ar}, b.rank={br})"
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_matmul_batched: both inputs must be contiguous".to_string(),
        ));
    }
    if a.layout().start_offset() != 0 || b.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "vulkan_matmul_batched: both inputs must have start_offset == 0 (whole-buffer)"
                .to_string(),
        ));
    }

    let a_vk = a
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_matmul_batched: a must be Vulkan-backed".to_string()))?;
    let b_vk = b
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_matmul_batched: b must be Vulkan-backed".to_string()))?;

    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[ar - 2];
    let k = a_shape[ar - 1];
    let kk = b_shape[br - 2];
    let n = b_shape[br - 1];
    if k != kk {
        return Err(Error::Msg(format!(
            "vulkan_matmul_batched: contraction-dim mismatch (a K={k} vs b K={kk})"
        )));
    }
    let batch_a: usize = a_shape[..ar - 2].iter().product::<usize>().max(1);
    let batch_b: usize = b_shape[..br - 2].iter().product::<usize>().max(1);
    if batch_a != batch_b {
        return Err(Error::Msg(format!(
            "vulkan_matmul_batched: batch mismatch (a={batch_a} vs b={batch_b}) for shapes {a_shape:?} / {b_shape:?}"
        )));
    }

    let device_index = match a_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    // Zero-copy: wrap each kt buffer as a rank-3 VkTensor with the batch
    // axes flattened. Contiguous + start_offset==0 (checked above) makes the
    // flattened view exact — same bytes, different logical shape.
    let vk_a = VkTensor::from_buffer(
        a_vk.buffer_arc(),
        vec![batch_a, m, k],
        vk_dtype,
        Arc::clone(a_vk.vulkan_device()),
    );
    let vk_b = VkTensor::from_buffer(
        b_vk.buffer_arc(),
        vec![batch_b, kk, n],
        vk_dtype,
        Arc::clone(b_vk.vulkan_device()),
    );
    // Both kernels accumulate in F32 and return an F32 [batch, m, n] result;
    // the BF16 kernel reads bf16-packed inputs and writes F32 (no write race).
    let vk_out = match dtype {
        DType::F32 => vk_matmul_batched_no_grad(&vk_a, &vk_b),
        DType::BF16 => vk_matmul_batched_bf16_no_grad(&vk_a, &vk_b),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_matmul_batched: kernel dispatch failed: {e}"
        ))
    })?;
    // vk_out is F32 [batch, m, n]; restore the caller's leading axes.
    let mut out_shape: Vec<usize> = a_shape[..ar - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_f32 = kt_tensor_from_vk(&vk_out, device_index)?.reshape(out_shape)?;
    // Match the input dtype: BF16 inputs → BF16 output (resident vulkan_cast).
    match dtype {
        DType::F32 => Ok(out_f32),
        DType::BF16 => out_f32.to_dtype(DType::BF16),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
}

// ----------------------------------------------------------------------
// vulkan_matmul_lhs_transposed — resident `a^T @ b` GEMM
// ----------------------------------------------------------------------

/// Vulkan resident GEMM `a^T @ b`, where `a:[..., K, M]` and
/// `b:[..., K, N]`, returning `[..., M, N]`.
///
/// This is the Vulkan counterpart to `cuda_matmul_lhs_transposed` and avoids
/// allocating the physical `a.transpose(-2, -1).contiguous()` image in LoRA and
/// matmul backward. Rank-2 inputs are routed as a batch of one; higher-rank
/// inputs flatten their leading axes for the batched shader.
pub fn vulkan_matmul_lhs_transposed(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul_batched::{
        vk_matmul_lhs_t_batched_bf16_no_grad, vk_matmul_lhs_t_batched_no_grad,
    };
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16) || b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "vulkan_matmul_lhs_transposed: F32/BF16 equal-dtype kernel (got a={}, b={})",
            a.dtype(),
            b.dtype()
        )));
    }
    let vk_dtype = match dtype {
        DType::F32 => VkDType::F32,
        DType::BF16 => VkDType::Bf16,
        _ => unreachable!("dtype gated to F32/BF16 above"),
    };
    let (ar, br) = (a.rank(), b.rank());
    if ar < 2 || br != ar {
        return Err(Error::Msg(format!(
            "vulkan_matmul_lhs_transposed: rank >= 2 and equal ranks required (a.rank={ar}, b.rank={br})"
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_matmul_lhs_transposed: both inputs must be contiguous".to_string(),
        ));
    }
    if a.layout().start_offset() != 0 || b.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "vulkan_matmul_lhs_transposed: both inputs must have start_offset == 0 (whole-buffer)"
                .to_string(),
        ));
    }

    let a_vk = a
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_matmul_lhs_transposed: a must be Vulkan-backed".to_string())
        })?;
    let b_vk = b
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_matmul_lhs_transposed: b must be Vulkan-backed".to_string())
        })?;

    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..ar - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(Error::Msg(format!(
                "vulkan_matmul_lhs_transposed: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let k = a_shape[ar - 2];
    let m = a_shape[ar - 1];
    let kk = b_shape[br - 2];
    let n = b_shape[br - 1];
    if k != kk {
        return Err(Error::Msg(format!(
            "vulkan_matmul_lhs_transposed: contraction-dim mismatch (a K={k} vs b K={kk})"
        )));
    }

    let batch: usize = a_shape[..ar - 2].iter().product::<usize>().max(1);
    let device_index = match a_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let vk_a = VkTensor::from_buffer(
        a_vk.buffer_arc(),
        vec![batch, k, m],
        vk_dtype,
        Arc::clone(a_vk.vulkan_device()),
    );
    let vk_b = VkTensor::from_buffer(
        b_vk.buffer_arc(),
        vec![batch, kk, n],
        vk_dtype,
        Arc::clone(b_vk.vulkan_device()),
    );
    let vk_out = match dtype {
        DType::F32 => vk_matmul_lhs_t_batched_no_grad(&vk_a, &vk_b),
        DType::BF16 => vk_matmul_lhs_t_batched_bf16_no_grad(&vk_a, &vk_b),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_matmul_lhs_transposed: kernel dispatch failed: {e}"
        ))
    })?;

    let mut out_shape: Vec<usize> = a_shape[..ar - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_f32 = kt_tensor_from_vk(&vk_out, device_index)?.reshape(out_shape)?;
    match dtype {
        DType::F32 => Ok(out_f32),
        DType::BF16 => out_f32.to_dtype(DType::BF16),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
}

// ----------------------------------------------------------------------
// vulkan_matmul_rhs_transposed — resident `a @ b^T` GEMM
// ----------------------------------------------------------------------

/// Vulkan resident GEMM `a @ b^T`, where `a:[..., M, K]` and
/// `b:[..., N, K]`, returning `[..., M, N]`.
///
/// This is the Vulkan counterpart to `cuda_matmul_rhs_transposed` and avoids
/// allocating the physical `b.transpose(-2, -1).contiguous()` image in matmul
/// backward and chunked loss gradients. Rank-2 inputs are routed as a batch of
/// one; higher-rank inputs flatten their leading axes for the batched shader.
pub fn vulkan_matmul_rhs_transposed(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul_batched::{
        vk_matmul_rhs_t_batched_bf16_no_grad, vk_matmul_rhs_t_batched_no_grad,
    };
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};

    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16) || b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "vulkan_matmul_rhs_transposed: F32/BF16 equal-dtype kernel (got a={}, b={})",
            a.dtype(),
            b.dtype()
        )));
    }
    let vk_dtype = match dtype {
        DType::F32 => VkDType::F32,
        DType::BF16 => VkDType::Bf16,
        _ => unreachable!("dtype gated to F32/BF16 above"),
    };
    let (ar, br) = (a.rank(), b.rank());
    if ar < 2 || br != ar {
        return Err(Error::Msg(format!(
            "vulkan_matmul_rhs_transposed: rank >= 2 and equal ranks required (a.rank={ar}, b.rank={br})"
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "vulkan_matmul_rhs_transposed: both inputs must be contiguous".to_string(),
        ));
    }
    if a.layout().start_offset() != 0 || b.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "vulkan_matmul_rhs_transposed: both inputs must have start_offset == 0 (whole-buffer)"
                .to_string(),
        ));
    }

    let a_vk = a
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_matmul_rhs_transposed: a must be Vulkan-backed".to_string())
        })?;
    let b_vk = b
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_matmul_rhs_transposed: b must be Vulkan-backed".to_string())
        })?;

    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..ar - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(Error::Msg(format!(
                "vulkan_matmul_rhs_transposed: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let m = a_shape[ar - 2];
    let k = a_shape[ar - 1];
    let n = b_shape[br - 2];
    let kk = b_shape[br - 1];
    if k != kk {
        return Err(Error::Msg(format!(
            "vulkan_matmul_rhs_transposed: contraction-dim mismatch (a K={k} vs b K={kk})"
        )));
    }

    let batch: usize = a_shape[..ar - 2].iter().product::<usize>().max(1);
    let device_index = match a_vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let vk_a = VkTensor::from_buffer(
        a_vk.buffer_arc(),
        vec![batch, m, k],
        vk_dtype,
        Arc::clone(a_vk.vulkan_device()),
    );
    let vk_b = VkTensor::from_buffer(
        b_vk.buffer_arc(),
        vec![batch, n, kk],
        vk_dtype,
        Arc::clone(b_vk.vulkan_device()),
    );
    let vk_out = match dtype {
        DType::F32 => vk_matmul_rhs_t_batched_no_grad(&vk_a, &vk_b),
        DType::BF16 => vk_matmul_rhs_t_batched_bf16_no_grad(&vk_a, &vk_b),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_matmul_rhs_transposed: kernel dispatch failed: {e}"
        ))
    })?;

    let mut out_shape: Vec<usize> = a_shape[..ar - 2].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    let out_f32 = kt_tensor_from_vk(&vk_out, device_index)?.reshape(out_shape)?;
    match dtype {
        DType::F32 => Ok(out_f32),
        DType::BF16 => out_f32.to_dtype(DType::BF16),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
}

// ----------------------------------------------------------------------
// vulkan_contiguous — on-device strided gather (kills the contiguous() bounce)
// ----------------------------------------------------------------------

/// Materialize a non-contiguous Vulkan tensor into a fresh contiguous Vulkan
/// tensor entirely on-device, via the `vk_gather_contiguous_{f32,bf16}`
/// kernels. This replaces the correctness-first host bounce in
/// [`crate::Tensor::contiguous`] (D2H per-element gather + H2D re-upload) —
/// the single most common CPU round-trip on the Vulkan inference path (every
/// `transpose().contiguous()`, `narrow().contiguous()`, GQA expand, …).
///
/// The input may be any strided/offset layout; its raw `shape` / element
/// `strides` / element `start_offset` are passed straight to the gather
/// kernel, which reads from the **whole** underlying buffer.
///
/// # Requirements / fallback
///
/// - F32 or BF16, Vulkan-backed, `rank <= vk_ops::contiguous_gather::MAX_RANK`
///   (8). The caller ([`crate::Tensor::contiguous`]) gates these and keeps the
///   host path for anything outside the envelope (other dtypes, rank > 8).
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, unsupported dtype, over-rank
/// input, or kernel dispatch failure.
pub fn vulkan_contiguous(t: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::contiguous_gather::{
        MAX_RANK, vk_gather_contiguous_bf16, vk_gather_contiguous_f32,
    };

    if !matches!(t.dtype(), DType::F32 | DType::BF16) {
        return Err(Error::Msg(format!(
            "vulkan_contiguous: F32/BF16-only gather (got {})",
            t.dtype()
        )));
    }
    let rank = t.rank();
    if rank > MAX_RANK {
        return Err(Error::Msg(format!(
            "vulkan_contiguous: rank {rank} exceeds gather MAX_RANK {MAX_RANK}"
        )));
    }
    let vk = t
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_contiguous: tensor must be Vulkan-backed".to_string()))?;
    let device_index = match vk.device() {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let device = vk.vulkan_device();
    // The whole underlying buffer; `strides`/`start_offset` index into it.
    let src = vk.buffer_arc();
    let shape = t.shape().to_vec();
    let strides = t.strides().to_vec();
    let start_offset = t.layout().start_offset();

    let vk_out = match t.dtype() {
        DType::F32 => vk_gather_contiguous_f32(device, &src, &shape, &strides, start_offset),
        DType::BF16 => vk_gather_contiguous_bf16(device, &src, &shape, &strides, start_offset),
        _ => unreachable!("dtype gated to F32/BF16 above"),
    }
    .map_err(|e| Error::Msg(format!("vulkan_contiguous: kernel dispatch failed: {e}")))?;
    kt_tensor_from_vk(&vk_out, device_index)
}

// ----------------------------------------------------------------------
// vulkan_matmul_bf16w — #1443 step1: F32-act × BF16-weight mixed-precision GEMM
// ----------------------------------------------------------------------

/// Whether `t` is backed by [`VulkanStorage`] (resident on the GPU pool).
fn is_vulkan_backed(t: &crate::Tensor) -> bool {
    t.storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .is_some()
}

/// One-shot WARN for the non-resident mixed-precision fallback so hybrid
/// (CPU-storage) training runs are honest about taking the slow path.
fn warn_bf16w_host_fallback_once() {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "WARN vulkan_matmul_bf16w: inputs are not Vulkan-resident (hybrid \
             CPU-storage training) — falling back to a host cast-to-F32 \
             matmul. Correct but slow; the Vulkan residency port removes \
             this path."
        );
    });
}

/// Vulkan mixed-precision linear `out = x @ W.T` where `x` is an F32
/// activation `[rows, K]` and `W` is a **frozen** BF16-packed weight in the
/// transposed `[N, K]` layout (`N = out_dim`, `K = hidden`). Returns an F32
/// `[rows, N]` tensor.
///
/// This is the foundational kt-level op for #1443 (store frozen base weights
/// in BF16 on Vulkan to halve base VRAM, keeping activations/LoRA/grads F32).
/// The kt [`crate::ops::matmul`] requires equal input dtypes, so F32-act ×
/// BF16-weight cannot run through it; this bridge fills that gap by calling the
/// dedicated `vk_matmul_bf16w` kernels.
///
/// Bridges both inputs to `VkTensor` zero-copy ([`vk_tensor_from_kt`]) — the
/// BF16 weight bridges cleanly because the PR3b bridge already gates F32|BF16 —
/// dispatches `kiln_vulkan_kernel::vk_ops::matmul_bf16w::vk_matmul_bf16w_no_grad`
/// (which tiles general PREFILL row counts internally via the
/// `vk_matmul_bf16w_fwd_rows` shader), and bridges the F32 result back zero-copy
/// ([`kt_tensor_from_vk`], which records the **logical** byte length, not the
/// pool-rounded physical allocation). No D2H/H2D round-trip.
///
/// # Requirements
///
/// - `x` and `weight_t` both backed by [`VulkanStorage`] (same device)
/// - `x.dtype() == F32`, `weight_t.dtype() == BF16`
/// - both rank-2 and contiguous (`start_offset == 0`), with
///   `x.shape()[1] == weight_t.shape()[1]` (shared contraction dim `K`)
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, dtype/shape violations, or
/// kernel dispatch failure.
pub fn vulkan_matmul_bf16w(x: &crate::Tensor, weight_t: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul_bf16w::vk_matmul_bf16w_no_grad;

    if x.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w: x must be F32 (got {})",
            x.dtype()
        )));
    }
    if weight_t.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w: weight must be BF16 (got {})",
            weight_t.dtype()
        )));
    }
    if x.rank() != 2 || weight_t.rank() != 2 {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w: rank-2 only (got x.rank={}, weight.rank={})",
            x.rank(),
            weight_t.rank()
        )));
    }
    if x.shape()[1] != weight_t.shape()[1] {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w: contraction dim mismatch: x.shape[1]={} vs weight.shape[1]={}",
            x.shape()[1],
            weight_t.shape()[1]
        )));
    }
    // Hybrid (CPU-storage) training fallback: the kernel needs
    // Vulkan-resident buffers; until the residency port, production
    // training tensors are kt CPU-storage. Cast the frozen weight to F32
    // and run the equal-dtype matmul — bitwise-equivalent math (the
    // kernel also widens BF16 to F32 in-shader), just slower.
    if !is_vulkan_backed(x) || !is_vulkan_backed(weight_t) {
        warn_bf16w_host_fallback_once();
        // Hybrid residency can mix devices (vulkan-resident activations,
        // CPU-storage weights). Compute on host — moving the small
        // activation is far cheaper than uploading the full F32 weight —
        // then return to x's device.
        let x_dev = x.device();
        let x_host = x.to_device(Device::Cpu)?;
        let w_f32 = weight_t.to_device(Device::Cpu)?.to_dtype(DType::F32)?;
        let y = crate::ops::matmul_rhs_transposed(&x_host, &w_f32)?;
        return y.to_device(x_dev);
    }
    let device_index = vulkan_device_index(x, "vulkan_matmul_bf16w")?;

    let vk_x = vk_tensor_from_kt(x)?;
    let vk_w = vk_tensor_from_kt(weight_t)?;
    let vk_out = vk_matmul_bf16w_no_grad(&vk_x, &vk_w)
        .map_err(|e| Error::Msg(format!("vulkan_matmul_bf16w: kernel dispatch failed: {e}")))?;
    kt_tensor_from_vk(&vk_out, device_index)
}

/// Vulkan mixed-precision linear backward `dx = grad_out @ W` for the frozen
/// BF16 weight `W` shaped `[N, K]`. Companion to [`vulkan_matmul_bf16w`]: given
/// the upstream F32 gradient `grad_out` `[rows, N]`, returns `dx` `[rows, K]`
/// F32. The weight is FROZEN — there is **no** `dW` (the recorder returns
/// `None` for the weight slot).
///
/// Bridges both inputs to `VkTensor` zero-copy, dispatches
/// `vk_matmul_bf16w_bwd_no_grad` (tiled over prefill rows internally), and
/// bridges `dx` back zero-copy.
///
/// # Requirements
///
/// - `grad_out` and `weight_t` both [`VulkanStorage`]-backed (same device)
/// - `grad_out.dtype() == F32`, `weight_t.dtype() == BF16`
/// - both rank-2 and contiguous (`start_offset == 0`), with
///   `grad_out.shape()[1] == weight_t.shape()[0]` (shared `N = out_dim`)
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, dtype/shape violations, or
/// kernel dispatch failure.
pub fn vulkan_matmul_bf16w_bwd(
    grad_out: &crate::Tensor,
    weight_t: &crate::Tensor,
) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul_bf16w::vk_matmul_bf16w_bwd_no_grad;

    if grad_out.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w_bwd: grad_out must be F32 (got {})",
            grad_out.dtype()
        )));
    }
    if weight_t.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w_bwd: weight must be BF16 (got {})",
            weight_t.dtype()
        )));
    }
    if grad_out.rank() != 2 || weight_t.rank() != 2 {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w_bwd: rank-2 only (got grad_out.rank={}, weight.rank={})",
            grad_out.rank(),
            weight_t.rank()
        )));
    }
    if grad_out.shape()[1] != weight_t.shape()[0] {
        return Err(Error::Msg(format!(
            "vulkan_matmul_bf16w_bwd: dim mismatch: grad_out.shape[1]={} vs weight.shape[0]={}",
            grad_out.shape()[1],
            weight_t.shape()[0]
        )));
    }
    // Hybrid (CPU-storage) fallback — see `vulkan_matmul_bf16w`.
    if !is_vulkan_backed(grad_out) || !is_vulkan_backed(weight_t) {
        warn_bf16w_host_fallback_once();
        let go_dev = grad_out.device();
        let go_host = grad_out.to_device(Device::Cpu)?;
        let w_f32 = weight_t.to_device(Device::Cpu)?.to_dtype(DType::F32)?;
        let dx = go_host.matmul(&w_f32)?;
        return dx.to_device(go_dev);
    }
    let device_index = vulkan_device_index(grad_out, "vulkan_matmul_bf16w_bwd")?;

    let vk_grad = vk_tensor_from_kt(grad_out)?;
    let vk_w = vk_tensor_from_kt(weight_t)?;
    let vk_dx = vk_matmul_bf16w_bwd_no_grad(&vk_grad, &vk_w).map_err(|e| {
        Error::Msg(format!(
            "vulkan_matmul_bf16w_bwd: kernel dispatch failed: {e}"
        ))
    })?;
    kt_tensor_from_vk(&vk_dx, device_index)
}

// ----------------------------------------------------------------------
// vulkan_scale — Phase 3c hot-op port: tensor * scalar (#1082)
// ----------------------------------------------------------------------

/// Vulkan elementwise scalar-multiply `out = x * scale`. Backs
/// `ScalarOp`'s `mul_scalar` (scale = c) and `div_scalar` (scale = 1/c)
/// hot paths that the backward composites call.
///
/// Bridges the input to `VkTensor` zero-copy ([`vk_tensor_from_kt`]),
/// dispatches the production
/// `kiln_vulkan_kernel::vk_ops::mask::vk_scale_no_grad` shader
/// (`vk_scale_inplace_f32`), and bridges the result back zero-copy
/// ([`kt_tensor_from_vk`]). No D2H / H2D round-trip — the data stays
/// GPU-resident end to end.
///
/// `add_scalar` / `sub_scalar` have **no** corresponding scalar-bias
/// Vulkan kernel today, so `ScalarOp::vulkan_fwd` returns `Ok(None)` for
/// those and the dispatch host-fallback (PR3a) covers them.
///
/// # Requirements
///
/// - `x` backed by [`VulkanStorage`]
/// - `x.dtype() == F32` (the kernel is F32-only today)
/// - `x` contiguous with `start_offset == 0`
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, dtype/layout violations,
/// or kernel dispatch failure.
pub fn vulkan_scale(x: &crate::Tensor, scale: f32) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::mask::vk_scale_no_grad;

    if x.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "vulkan_scale: F32-only kernel (got {})",
            x.dtype()
        )));
    }
    let device_index = match x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_scale: input must be Vulkan-backed".to_string()))?
        .device()
    {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let vk_in = vk_tensor_from_kt(x)?;
    let vk_out = vk_scale_no_grad(&vk_in, scale)
        .map_err(|e| Error::Msg(format!("vulkan_scale: kernel dispatch failed: {e}")))?;
    kt_tensor_from_vk(&vk_out, device_index)
}

// ----------------------------------------------------------------------
// vulkan_sum_all / vulkan_mean_all — Phase 3c hot-op port (#1082)
// ----------------------------------------------------------------------

/// Vulkan all-elements reduction `out = sum(x)` (or `mean(x)`), producing
/// a rank-0 (scalar) F32 tensor matching `ReduceOp(All)`'s CPU output
/// shape.
///
/// Bridges the input to `VkTensor` zero-copy ([`vk_tensor_from_kt`]),
/// dispatches the production two-pass tree reduction
/// (`kiln_vulkan_kernel::vk_ops::reduce::vk_sum_all_no_grad` /
/// `vk_mean_all`), and bridges the `[1]`-shaped result back zero-copy,
/// then reshapes to rank-0 (the kernel emits shape `[1]`; the kt CPU
/// reference emits a 0-D scalar — the reshape is a metadata-only view, no
/// device copy).
///
/// Only the **All** reduction scope has a Vulkan kernel. `sum_axis` /
/// `mean_axis` (single-axis, keepdim=false) have no axis-reduce shader, so
/// `ReduceOp::vulkan_fwd` returns `Ok(None)` for those and the dispatch
/// host-fallback (PR3a) covers them.
///
/// # Requirements
///
/// - `x` backed by [`VulkanStorage`], `x.dtype() == F32`, non-empty
/// - `x` contiguous with `start_offset == 0`
///
/// # Errors
///
/// Returns [`Error::Msg`] on non-Vulkan storage, dtype violations, empty
/// input, or kernel dispatch failure.
fn vulkan_reduce_all(x: &crate::Tensor, mean: bool) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::reduce::{vk_mean_all, vk_sum_all_no_grad};

    let op = if mean {
        "vulkan_mean_all"
    } else {
        "vulkan_sum_all"
    };
    if x.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "{op}: F32-only kernel (got {})",
            x.dtype()
        )));
    }
    if x.element_count() == 0 {
        return Err(Error::Msg(format!("{op}: empty tensor")));
    }
    let device_index = match x
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg(format!("{op}: input must be Vulkan-backed")))?
        .device()
    {
        Device::Vulkan(i) => i,
        _ => unreachable!("VulkanStorage::device() returns Device::Vulkan"),
    };

    let vk_in = vk_tensor_from_kt(x)?;
    let vk_out = if mean {
        vk_mean_all(&vk_in)
    } else {
        vk_sum_all_no_grad(&vk_in)
    }
    .map_err(|e| Error::Msg(format!("{op}: kernel dispatch failed: {e}")))?;

    // Kernel emits a `[1]`-shaped scalar; reshape to rank-0 to match the
    // CPU reference's 0-D output. Metadata-only — no device copy.
    let scalar1 = kt_tensor_from_vk(&vk_out, device_index)?;
    scalar1.reshape(Vec::<usize>::new())
}

/// Vulkan `sum(x)` over all elements → rank-0 scalar. See
/// [`vulkan_reduce_all`].
pub fn vulkan_sum_all(x: &crate::Tensor) -> Result<crate::Tensor> {
    vulkan_reduce_all(x, false)
}

/// Vulkan `mean(x)` over all elements → rank-0 scalar. See
/// [`vulkan_reduce_all`].
pub fn vulkan_mean_all(x: &crate::Tensor) -> Result<crate::Tensor> {
    vulkan_reduce_all(x, true)
}

/// Vulkan implementation of [`crate::Tensor::all_finite`]. F32 and BF16
/// inputs are reduced entirely on-device and only the final u32 flag is read
/// back. Other floating formats use the existing host copy as a correctness
/// fallback until their storage formats have native Vulkan kernels.
pub fn vulkan_is_finite(x: &crate::Tensor) -> Result<bool> {
    use kiln_vulkan_kernel::vk_ops::reduce::vk_all_finite;

    if x.element_count() == 0 {
        return Ok(true);
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16) {
        return vulkan_to_host_copy(x)?.all_finite();
    }

    // VkTensor addresses a whole contiguous buffer. A contiguous narrow can
    // still carry a non-zero offset, so force the resident gather in either
    // case instead of relying on Tensor::contiguous's metadata fast path.
    let materialized;
    let input = if x.is_contiguous() && x.layout().start_offset() == 0 {
        x
    } else {
        materialized = vulkan_contiguous(x)?;
        &materialized
    };
    let vk = vk_tensor_from_kt(input)?;
    vk_all_finite(&vk)
        .map_err(|e| Error::Msg(format!("vulkan_is_finite: kernel dispatch failed: {e}")))
}

// ----------------------------------------------------------------------
// Host ↔ Vulkan I/O — candle-free device staging (#1082 PR2)
// ----------------------------------------------------------------------
//
// These are the Vulkan arms of `Tensor::{from_vec_on, from_raw_bytes_on,
// zeros_on}` (host→Vulkan) and `Tensor::to_device(Cpu)` / `Tensor::to_vec`
// (Vulkan→host) — the exact counterparts of `host_to_metal_copy` /
// `metal_to_host_copy`. They are the storage keystone every Vulkan
// parity test needs (you cannot A/B a Vulkan op against the CPU
// reference without constructing a Vulkan input from host data and
// reading the Vulkan output back).
//
// Unlike Apple-Silicon UMA, a Vulkan `DEVICE_LOCAL` buffer is **not**
// CPU-addressable, so the "copy" is a real H2D / D2H transfer through a
// host-visible staging buffer — exactly what
// `VulkanBuffer::{upload_data, read_back}` already implement. Those
// primitives are pure byte-blob movers; this layer owns the
// dtype / packed-byte / contiguity contract (mirrors
// `host_to_metal_copy`: CPU-backed source, materialized contiguous,
// byte-range checks).

use std::sync::{Mutex, OnceLock};

/// Process-global cache of one logical [`VulkanDevice`] per device
/// ordinal — the Vulkan analogue of [`crate::primary_metal_companion`].
///
/// Logical-device creation (`VulkanDevice::new`) is tens of
/// milliseconds and allocates queues, so the host↔device copy helpers
/// and the `Device::Vulkan(i)` tensor constructors share a single
/// cached device rather than spinning up a fresh one per call.
///
/// Note: `VulkanDevice::new` performs its own best-GPU / env-driven
/// physical-device selection and does not currently take an explicit
/// ordinal, so all ordinals presently resolve to the same selected
/// physical device; we still key the cache by `device_index` so the
/// `Device::Vulkan(idx)` recorded on the storage matches what the
/// caller asked for and a future multi-GPU selector slots in cleanly.
///
/// # Errors
///
/// Returns [`Error::Msg`] if no Vulkan device can be created.
pub fn primary_vulkan_device(device_index: usize) -> Result<Arc<VulkanDevice>> {
    static DEVICES: OnceLock<Mutex<std::collections::HashMap<usize, Arc<VulkanDevice>>>> =
        OnceLock::new();
    let map_mutex = DEVICES.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut map = map_mutex.lock().map_err(|_| {
        Error::Msg("primary_vulkan_device: device cache mutex poisoned".to_string())
    })?;
    if let Some(dev) = map.get(&device_index) {
        return Ok(Arc::clone(dev));
    }
    let dev = Arc::new(VulkanDevice::new().map_err(|e| {
        Error::Msg(format!(
            "primary_vulkan_device({device_index}): VulkanDevice::new() failed: {e}"
        ))
    })?);
    map.insert(device_index, Arc::clone(&dev));
    Ok(dev)
}

/// Block until the primary Vulkan queue for `device_index` is idle.
///
/// This is the Vulkan counterpart to `cuda_synchronize_default_stream` /
/// `rocm_synchronize_compute_stream` for cross-kernel handoff points where a
/// later backend-specific consumer must see all previously submitted work.
pub fn vulkan_synchronize_queue(device_index: usize) -> Result<()> {
    let vulkan_device = primary_vulkan_device(device_index)?;
    unsafe {
        vulkan_device
            .device()
            .queue_wait_idle(vulkan_device.queue())
            .map_err(|e| {
                Error::Msg(format!(
                    "vulkan_synchronize_queue({device_index}): queue_wait_idle failed: {e:?}"
                ))
            })
    }
}

/// Upload a host (CPU-resident) tensor to a fresh `DEVICE_LOCAL`
/// Vulkan buffer on `device_index`. **Candle-core-free.**
///
/// The result is a contiguous, `start_offset == 0` Vulkan tensor in
/// logical row-major order. The source is materialized contiguous on
/// the host first (cheap when already contiguous), so any input layout
/// is accepted. Mirrors [`crate::host_to_metal_copy`].
///
/// # Errors
///
/// Returns [`Error::Msg`] if `cpu` is not [`crate::CpuStorage`]-backed,
/// no Vulkan device exists at `device_index`, or buffer allocation /
/// upload fails.
pub fn host_to_vulkan_copy(cpu: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    use crate::CpuStorage;

    // Materialize a packed, logical-row-major byte image on the host.
    let contig = cpu.contiguous()?;
    let dtype = contig.dtype();
    let cpu_storage = contig
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            Error::Msg("host_to_vulkan_copy: source tensor must be CPU-backed".to_string())
        })?;
    let all_bytes = cpu_storage.as_bytes();
    // A contiguous tensor may still carry a non-zero start_offset (a
    // narrowed-but-contiguous view shares the parent buffer); slice to
    // the logical element range so the Vulkan buffer holds exactly this
    // tensor.
    let n_elems = contig.element_count();
    let per = dtype.size_in_bytes();
    let byte_len = if dtype.is_packed() {
        all_bytes.len()
    } else {
        n_elems * per
    };
    let start_bytes = contig.layout().start_offset() * per;
    let end_bytes = start_bytes + byte_len;
    if end_bytes > all_bytes.len() {
        return Err(Error::Msg(format!(
            "host_to_vulkan_copy: byte range {start_bytes}..{end_bytes} exceeds CPU storage \
             length {}",
            all_bytes.len()
        )));
    }
    let src = &all_bytes[start_bytes..end_bytes];

    let vulkan_device = primary_vulkan_device(device_index)?;
    // Allocate at least 1 byte: a zero-length Vulkan buffer is invalid.
    let alloc_len = byte_len.max(1) as u64;
    let buffer = VulkanBuffer::create_device_local(
        vulkan_device.device(),
        vulkan_device.device_local_mem_type(),
        alloc_len,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "host_to_vulkan_copy: create_device_local({alloc_len}) failed: {e}"
        ))
    })?;
    // H2D: stage `src` and copy into the device-local buffer. `src` may
    // be empty (a zero-element tensor); skip the transfer in that case
    // since the buffer was allocated at the 1-byte floor purely to be
    // a valid handle.
    if !src.is_empty() {
        VulkanBuffer::upload_data(
            vulkan_device.device(),
            vulkan_device.host_visible_mem_type(),
            vulkan_device.queue(),
            vulkan_device.queue_family_index(),
            &buffer,
            src,
        )
        .map_err(|e| Error::Msg(format!("host_to_vulkan_copy: H2D upload failed: {e}")))?;
    }

    let storage =
        VulkanStorage::from_buffer(vulkan_device, device_index, dtype, buffer, byte_len as u64)?;
    crate::Tensor::from_parts(
        Arc::new(storage),
        crate::Layout::contiguous(contig.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Read a Vulkan tensor back to a fresh CPU tensor, packed contiguous in
/// logical row-major order. **Candle-core-free.** Mirrors
/// [`crate::metal_to_host_copy`].
///
/// D2H-reads the whole device buffer through a host-visible staging
/// buffer (`VulkanBuffer::read_back`, which submits + waits on the
/// queue, so prior GPU writes are visible), then gathers the logical
/// elements — handling any strided / offset view via a host-side
/// gather, exactly as the Metal readback does.
///
/// # Errors
///
/// Returns [`Error::Msg`] if the tensor is not [`VulkanStorage`]-backed
/// or the read-back fails.
pub fn vulkan_to_host_copy(t: &crate::Tensor) -> Result<crate::Tensor> {
    use crate::CpuStorage;

    let vk = t
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_to_host_copy: tensor must be Vulkan-backed".to_string())
        })?;
    let vulkan_device = vk.vulkan_device();

    // D2H: pull the device buffer's bytes back to the host. read_back
    // submits and waits on the queue, so GPU writes are visible.
    let backing = VulkanBuffer::read_back(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        vk.buffer(),
    )
    .map_err(|e| Error::Msg(format!("vulkan_to_host_copy: D2H read_back failed: {e}")))?;
    let buf_len = backing.len();

    let dtype = t.dtype();
    let layout = t.layout();
    let n_elems = t.element_count();

    if dtype.is_packed() {
        // Packed dtypes have no per-element stride math; copy the
        // addressed byte image directly (Vulkan packed-dtype tensors are
        // always whole contiguous buffers in the current code paths).
        let storage = CpuStorage::from_bytes(dtype, backing)?;
        return crate::Tensor::from_parts(
            Arc::new(storage),
            crate::Layout::contiguous(t.shape().to_vec()),
            crate::TensorId::next(),
        );
    }

    let per = dtype.size_in_bytes();
    let start = layout.start_offset();
    let mut out = vec![0u8; n_elems * per];
    if layout.is_contiguous() {
        let s = start * per;
        let e = s + n_elems * per;
        if e > buf_len {
            return Err(Error::Msg(format!(
                "vulkan_to_host_copy: contiguous range {s}..{e} exceeds buffer length {buf_len}"
            )));
        }
        out.copy_from_slice(&backing[s..e]);
    } else {
        // Strided / permuted view: gather each logical element by
        // walking the multi-dimensional index against the layout
        // strides (host readback is rare; per-element gather is fine).
        let dims = layout.shape();
        let strides = layout.strides();
        let rank = dims.len();
        let mut idx = vec![0usize; rank];
        for logical in 0..n_elems {
            let mut phys = start;
            for d in 0..rank {
                phys += idx[d] * strides[d];
            }
            let s = phys * per;
            let d_off = logical * per;
            if s + per > buf_len {
                return Err(Error::Msg(format!(
                    "vulkan_to_host_copy: element offset {s}..{} exceeds buffer length {buf_len}",
                    s + per
                )));
            }
            out[d_off..d_off + per].copy_from_slice(&backing[s..s + per]);
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    let storage = CpuStorage::from_bytes(dtype, out)?;
    crate::Tensor::from_parts(
        Arc::new(storage),
        crate::Layout::contiguous(t.shape().to_vec()),
        crate::TensorId::next(),
    )
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
/// **Zero-copy (#1082 PR3b bridge).** The input kt buffer is shared into the
/// kernel as a `VkTensor` via [`vk_tensor_from_kt`] (an `Arc<VulkanBuffer>`
/// refcount bump — no D2H), the kernel runs on-device, and the result buffer
/// is wrapped straight back as a kt `VulkanStorage` via [`kt_tensor_from_vk`]
/// (no D2H/H2D). The data stays GPU-resident end to end; the former host
/// bounce is gone.
///
/// The resulting kt tensor's `VulkanStorage::byte_len()` is the **logical**
/// element-range size, while the underlying pooled `VulkanBuffer` may be
/// bucket-rounded larger. Every downstream readback consumer
/// (`vulkan_to_host_copy`, the input reads in the non-zero-copy wrappers)
/// slices to the logical `byte_len`, so the rounding is invisible.
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `x.dtype()` must be `F32` (kernel is F32-only; BF16/F16 needs cast
///   or a widened `VkDType` per the softmax-op TODOs)
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` and `x.layout().start_offset() == 0` (the zero-copy
///   `VkTensor` bridge is whole-buffer; `vk_tensor_from_kt` enforces this)
///
/// # Errors
///
/// Returns [`Error::Msg`] if the storage isn't `VulkanStorage`, the
/// dtype is unsupported, the layout is non-contiguous / offset, or the
/// underlying kernel call fails.
pub fn vulkan_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::softmax::vk_softmax_lastdim_no_grad;

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
    let device_index = vulkan_device_index(x, "vulkan_softmax_last_axis")?;

    // ---- Zero-copy: share the kt buffer into the kernel, dispatch, wrap the
    // result back — no D2H/H2D host bounce. `vk_tensor_from_kt` gates
    // contiguous + start_offset == 0 + F32/BF16 and errors otherwise (the
    // dispatcher then host-falls-back). ----
    let vk_in = vk_tensor_from_kt(x)?;
    let vk_out = vk_softmax_lastdim_no_grad(&vk_in).map_err(|e| {
        Error::Msg(format!(
            "vulkan_softmax_last_axis: kernel dispatch failed: {e}"
        ))
    })?;
    kt_tensor_from_vk(&vk_out, device_index)
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

    let kt_vk_w = weight
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_rmsnorm_last_axis: weight must be Vulkan-backed".to_string())
        })?;

    let vulkan_device = Arc::clone(kt_vk_w.vulkan_device());
    let device_index = vulkan_device_index(x, "vulkan_rmsnorm_last_axis")?;

    let w_byte_len = kt_vk_w.byte_len();

    // ---- Zero-copy bridge for `x` (the large activation tensor) — no host
    // bounce. `vk_tensor_from_kt` enforces contiguous + start_offset == 0 +
    // F32. ----
    let vk_x = vk_tensor_from_kt(x)?;

    // ---- Weight: the QwenRMSNorm shader computes `(1 + w_shader) * x / ...`,
    // but kt's reference (`RmsNormOp::cpu_fwd`) is `w * x / ...`, so the weight
    // must be staged as `w - 1.0`. That host-side subtraction cannot be a pure
    // Arc-share, so the weight (only `[hidden]` floats — tiny vs. `x`) takes a
    // bounded D2H → adjust → H2D. The dominant `x`/output traffic stays
    // zero-copy. ----
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
    // `read_back` returns the buffer's *physical* (pool-bucket-rounded) byte
    // image, which is >= the logical `w_byte_len` if the weight is itself a
    // zero-copy kernel output. Operate on exactly the logical element range.
    let w_logical = w_byte_len.min(w_bytes_orig.len());
    let w_bytes_adj: Vec<u8> = {
        // F32-only path (gated above): subtract 1.0 from each element.
        let n = w_logical / 4;
        let mut out = Vec::with_capacity(w_logical);
        for i in 0..n {
            let chunk = &w_bytes_orig[i * 4..(i + 1) * 4];
            let v = f32::from_le_bytes(chunk.try_into().unwrap());
            let adj = v - 1.0_f32;
            out.extend_from_slice(&adj.to_le_bytes());
        }
        out
    };

    let vk_dtype = VkDType::F32;
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

    // ---- Dispatch, then wrap the result back zero-copy (no output bounce). ----
    let vk_out = vk_rmsnorm_no_grad(&vk_x, &vk_w, eps).map_err(|e| {
        Error::Msg(format!(
            "vulkan_rmsnorm_last_axis: kernel dispatch failed: {e}"
        ))
    })?;
    kt_tensor_from_vk(&vk_out, device_index)
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
/// **Zero-copy (#1082 PR3b bridge).** Shares the kt input buffer into the
/// kernel via [`vk_tensor_from_kt`], dispatches on-device, and wraps the
/// result straight back via [`kt_tensor_from_vk`] — no D2H/H2D host bounce.
///
/// # Requirements
///
/// - `x` must be backed by [`VulkanStorage`]
/// - `x.dtype() == F32`
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` and `x.layout().start_offset() == 0`
/// - `*x.shape().last().unwrap() <= 256` (shader limit)
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or kernel error.
pub fn vulkan_l2norm_last_axis(x: &crate::Tensor, eps: f32) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::l2norm::vk_l2_norm_lastdim_no_grad;

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
    let hidden = *x.shape().last().unwrap();
    if hidden == 0 || hidden > 256 {
        return Err(Error::Msg(format!(
            "vulkan_l2norm_last_axis: hidden dim {hidden} exceeds shader cap 256"
        )));
    }
    let device_index = vulkan_device_index(x, "vulkan_l2norm_last_axis")?;

    // ---- Zero-copy bridge (no host bounce). `vk_tensor_from_kt` enforces
    // contiguous + start_offset == 0 + F32. ----
    let vk_in = vk_tensor_from_kt(x)?;
    let vk_out = vk_l2_norm_lastdim_no_grad(&vk_in, /*scale=*/ 1.0_f32, eps).map_err(|e| {
        Error::Msg(format!(
            "vulkan_l2norm_last_axis: kernel dispatch failed: {e}"
        ))
    })?;
    kt_tensor_from_vk(&vk_out, device_index)
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
    let device_index = vulkan_device_index(x, "vulkan_activation_unary")?;

    // ---- Zero-copy bridge (no host bounce). `vk_tensor_from_kt` enforces
    // contiguous + start_offset == 0 + F32. ----
    let vk_in = vk_tensor_from_kt(x)?;
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
    kt_tensor_from_vk(&vk_out, device_index)
}

/// Generic unary elementwise math on Vulkan (neg/exp/ln/sqrt/abs/recip/
/// sign/floor/.../sin/cos/tan/tanh/gelu/relu and scalar add/mul). `op`
/// selects the function (see
/// `kiln_vulkan_kernel::vk_ops::unary_elementwise::op`); `param0` is the
/// scalar operand for ADD_SCALAR/MUL_SCALAR and ignored otherwise.
///
/// Zero-copy: bridges through `vk_tensor_from_kt` / `kt_tensor_from_vk`
/// (no host bounce), so the kt `DeviceOp1` impls that route here stay
/// fully GPU-resident. F32-only and contiguous (enforced by the bridge);
/// callers (the `vulkan_fwd` impls) gate on those before dispatching.
pub fn vulkan_unary_math(x: &crate::Tensor, op: u32, param0: f32) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::unary_elementwise::vk_unary_elementwise_f32;

    let dtype = x.dtype();
    if !matches!(dtype, DType::F32) {
        return Err(Error::Msg(format!(
            "vulkan_unary_math: unsupported dtype {dtype} (F32-only today)"
        )));
    }
    let device_index = vulkan_device_index(x, "vulkan_unary_math")?;
    let vk_in = vk_tensor_from_kt(x)?;
    let vk_out = vk_unary_elementwise_f32(&vk_in, op, param0).map_err(|e| {
        Error::Msg(format!(
            "vulkan_unary_math: kernel dispatch (op={op}) failed: {e}"
        ))
    })?;
    kt_tensor_from_vk(&vk_out, device_index)
}

/// Vulkan in-place slice-set along dim 0: `dst[offset .. offset+src.dim0] = src`.
///
/// The Vulkan arm of [`crate::Tensor::slice_set`]. Both tensors are
/// Vulkan-backed and contiguous (the dim-0 slice-set contract), so this is a
/// pure device-to-device `vkCmdCopyBuffer` of the contiguous row range at the
/// computed byte offset — no host bounce. dtype-agnostic via raw
/// `VulkanStorage::buffer()` access, so it serves the BF16 paged-KV pool write
/// as well as F32 callers.
pub fn vulkan_slice_set_dim0(
    dst: &crate::Tensor,
    src: &crate::Tensor,
    offset: usize,
) -> Result<()> {
    let dst_vk = dst
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_slice_set_dim0: dst must be Vulkan-backed".to_string())
        })?;
    let src_vk = src
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| {
            Error::Msg("vulkan_slice_set_dim0: src must be Vulkan-backed".to_string())
        })?;
    let vulkan_device = dst_vk.vulkan_device();
    let bpe = dst.dtype().size_in_bytes() as u64;
    // `inner` = product of all dims except dim 0 (row size in elements).
    let inner: u64 = dst.dims().iter().skip(1).product::<usize>() as u64;
    let n_bytes = (src.element_count() as u64) * bpe;
    let dst_byte_off = (offset as u64) * inner * bpe;
    let src_byte_off = (src.layout().start_offset() as u64) * bpe;
    kiln_vulkan_kernel::buffer::VulkanBuffer::copy_buffer_region(
        vulkan_device.device(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        src_vk.buffer(),
        src_byte_off,
        dst_vk.buffer(),
        dst_byte_off,
        n_bytes,
    )
    .map_err(|e| Error::Msg(format!("vulkan_slice_set_dim0: device copy failed: {e}")))?;
    Ok(())
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
    // Pool-overflow guard (see vulkan_cast): clamp the upload to the logical
    // `in_byte_len` in case the input is a zero-copy kernel output whose
    // physical buffer is bucket-rounded larger (PR3b bridge).
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_in_buffer,
        &in_bytes[..in_byte_len.min(in_bytes.len())],
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
    // Pool-overflow guard: clamp to the logical `ids_byte_len`.
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_ids_buffer,
        &ids_bytes[..ids_byte_len.min(ids_bytes.len())],
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
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_cast: device-local alloc for VkTensor failed: {e}"
        ))
    })?;
    // Pool-overflow guard: `read_back` returns the buffer's *physical*
    // (bucket-rounded) byte image, which is >= the logical `in_byte_len` if the
    // input is a zero-copy kernel output (PR3b bridge). `vk_in_buffer` is sized
    // at `in_byte_len`; clamp the upload to the logical bytes so we never write
    // past the destination (RADV GPUVM PERMISSION_FAULTS).
    kiln_vulkan_kernel::buffer::VulkanBuffer::upload_data(
        vulkan_device.device(),
        vulkan_device.host_visible_mem_type(),
        vulkan_device.queue(),
        vulkan_device.queue_family_index(),
        &vk_in_buffer,
        &in_bytes[..in_byte_len.min(in_bytes.len())],
    )
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_cast: H2D upload of VkTensor input failed: {e}"
        ))
    })?;
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
    .map_err(|e| {
        Error::Msg(format!(
            "vulkan_cast: device-local alloc for kt output failed: {e}"
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
    let device_index = vulkan_device_index(a, "vulkan_elementwise_binary")?;

    // ---- Zero-copy bridge for BOTH inputs (no host bounce).
    // `vk_tensor_from_kt` enforces contiguous + start_offset == 0 + F32 for
    // each; either failing errors out and the dispatcher host-falls-back. ----
    let vk_a = vk_tensor_from_kt(a)?;
    let vk_b = vk_tensor_from_kt(b)?;
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
    kt_tensor_from_vk(&vk_out, device_index)
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
    let out_bytes: Vec<u8> = out_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
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
        .ok_or_else(|| Error::Msg("vulkan_masked_fill: x must be Vulkan-backed".to_string()))?;
    let kt_mask = mask
        .storage()
        .as_any()
        .downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vulkan_masked_fill: mask must be Vulkan-backed".to_string()))?;

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
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let v = if m_bytes[i] != 0 {
                    fill_value
                } else {
                    half::f16::from_le_bytes(x_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
                };
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
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
    fn vulkan_all_finite_reduces_f32_bf16_and_views() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }

        let finite = vk_f32(vec![1.0, -2.5, 0.0, f32::MIN_POSITIVE], vec![2, 2]);
        assert!(finite.all_finite().expect("finite F32 reduction"));

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let tensor = vk_f32(vec![1.0, bad, 3.0, 4.0], vec![2, 2]);
            assert!(!tensor.all_finite().expect("non-finite F32 reduction"));
            assert!(
                !tensor
                    .transpose(0, 1)
                    .expect("transpose view")
                    .all_finite()
                    .expect("non-contiguous F32 reduction")
            );
        }

        let offset = vk_f32(vec![f32::NAN, 1.0, 2.0, 3.0], vec![4])
            .narrow(0, 1, 3)
            .expect("offset narrow view");
        assert!(offset.all_finite().expect("offset view reduction"));

        let bf16_bytes = [
            0x80, 0x3f, // 1.0
            0x00, 0xc0, // -2.0
            0x80, 0x7f, // +Inf
            0xc0, 0x7f, // NaN
        ];
        let bf16 = crate::Tensor::from_raw_bytes_on(
            Device::Vulkan(0),
            DType::BF16,
            bf16_bytes.to_vec(),
            vec![4],
        )
        .expect("BF16 Vulkan tensor");
        assert!(!bf16.all_finite().expect("non-finite BF16 reduction"));
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

    #[test]
    fn zeros_read_back_as_zero_f32() {
        if !vulkan_test_enabled() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset");
            return;
        }
        if primary_vulkan_device(0).is_err() {
            eprintln!("skip: no Vulkan device");
            return;
        }
        let t = crate::Tensor::zeros_on(Device::Vulkan(0), vec![32], DType::F32).unwrap();
        let got = t.to_device(Device::Cpu).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(got, vec![0.0; 32]);
    }

    /// PR2 keystone (#1082): a host→Vulkan→host round-trip must preserve
    /// the exact bytes, shape, and dtype for an F32 tensor — and the
    /// intermediate tensor must actually live on `Device::Vulkan`.
    ///
    /// Skips when `KILN_TENSOR_VULKAN_TEST != 1` or no Vulkan device is
    /// present (CI has no GPU). Tiny tensors only — bounded validation,
    /// no training. Exercises the public constructor / `to_device` path
    /// (`from_vec_on` → `host_to_vulkan_copy`, then `to_device(Cpu)` →
    /// `vulkan_to_host_copy`) plus the raw-bytes BF16 path.
    #[test]
    fn host_vulkan_host_roundtrip_preserves_bytes() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }

        // ---- F32 round-trip via from_vec_on / to_device ----
        let data: Vec<f32> = vec![-2.5, 0.0, 1.0, 3.5, 42.0, -0.125];
        let shape = vec![2usize, 3usize];
        let dev = Device::Vulkan(0);

        let vk = crate::Tensor::from_vec_on(dev, data.clone(), shape.clone())
            .expect("from_vec_on(Vulkan) should construct a Vulkan tensor");
        assert_eq!(vk.device(), dev, "intermediate tensor must be on Vulkan");
        assert_eq!(vk.dtype(), DType::F32);
        assert_eq!(vk.shape(), shape.as_slice());

        let host = vk
            .to_device(Device::Cpu)
            .expect("to_device(Cpu) should D2H-copy the Vulkan tensor");
        assert_eq!(host.device(), Device::Cpu);
        assert_eq!(host.dtype(), DType::F32);
        assert_eq!(host.shape(), shape.as_slice());

        let got: Vec<f32> = host.to_vec().expect("read F32 tensor back to host Vec");
        assert_eq!(got, data, "F32 host→Vulkan→host must be byte-identical");

        // ---- BF16 round-trip via from_raw_bytes_on (raw LE bytes) ----
        // Two bf16 elements: 1.0 = 0x3F80, -2.0 = 0xC000 (LE byte order).
        let bf16_bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0xC0];
        let bf16_shape = vec![2usize];
        let vk_bf16 = crate::Tensor::from_raw_bytes_on(
            dev,
            DType::BF16,
            bf16_bytes.clone(),
            bf16_shape.clone(),
        )
        .expect("from_raw_bytes_on(Vulkan, BF16) should construct a Vulkan tensor");
        assert_eq!(vk_bf16.device(), dev);
        assert_eq!(vk_bf16.dtype(), DType::BF16);
        assert_eq!(vk_bf16.shape(), bf16_shape.as_slice());

        let host_bf16 = crate::vulkan_to_host_copy(&vk_bf16)
            .expect("vulkan_to_host_copy should D2H-copy the BF16 tensor");
        assert_eq!(host_bf16.device(), Device::Cpu);
        assert_eq!(host_bf16.dtype(), DType::BF16);
        let host_bf16_cpu = host_bf16
            .storage()
            .as_any()
            .downcast_ref::<crate::CpuStorage>()
            .expect("BF16 readback must be CPU-backed");
        assert_eq!(
            host_bf16_cpu.as_bytes(),
            bf16_bytes.as_slice(),
            "BF16 host→Vulkan→host must be byte-identical"
        );
    }

    // ---- PR5e: pool-size GPUVM write-fault regression guards ----
    //
    // Each of the bounce wrappers below read their kernel output back from
    // a `pool_alloc_f32` buffer, whose `.size` is bucket-rounded UP (64 KB
    // granularity at small sizes). The destination kt `out_buffer` is sized
    // at the *logical* byte length. Before the fix, the full bucket-sized
    // `out_bytes` was uploaded into the logical-sized buffer, overrunning it
    // and raising a RADV GPUVM write fault. Tiny tensors are the WORST case
    // (logical ~32 B vs. a 64 KB bucket), so a tiny [rows, hidden] F32 tensor
    // is the strongest single-shot reproducer. Each test asserts the readback
    // is finite and matches the CPU reference within a small tolerance.
    //
    // Bounded validation only: single op, single shot, tiny tensors. Skips
    // when KILN_TENSOR_VULKAN_TEST != 1 or no Vulkan device is present.

    fn vk_f32(data: Vec<f32>, shape: Vec<usize>) -> crate::Tensor {
        crate::Tensor::from_vec_on(Device::Vulkan(0), data, shape)
            .expect("from_vec_on(Vulkan, F32)")
    }

    fn read_vk_f32(t: &crate::Tensor) -> Vec<f32> {
        t.to_device(Device::Cpu)
            .expect("to_device(Cpu)")
            .to_vec()
            .expect("to_vec F32")
    }

    fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn vulkan_softmax_pool_overflow_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let rows = 2usize;
        let hidden = 4usize;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 2.5];
        let x = vk_f32(data.clone(), vec![rows, hidden]);
        let out = super::vulkan_softmax_last_axis(&x).expect("vulkan_softmax_last_axis");
        let got = read_vk_f32(&out);
        // CPU reference: row-wise softmax over the trailing axis.
        let mut want = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let row = &data[r * hidden..(r + 1) * hidden];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for i in 0..hidden {
                want[r * hidden + i] = exps[i] / sum;
            }
        }
        assert!(
            got.iter().all(|v| v.is_finite()),
            "softmax output not finite: {got:?}"
        );
        let err = max_abs_err(&got, &want);
        eprintln!("vulkan_softmax_pool_overflow_parity: max_abs_err = {err:e}");
        assert!(
            err < 1e-5,
            "softmax max_abs_err {err} too large; got={got:?} want={want:?}"
        );
    }

    /// PR3b zero-copy invariant (#1082): a wrapper that bridged its result
    /// back via `kt_tensor_from_vk` must record the **logical** element-range
    /// byte length on the output `VulkanStorage`, NOT the pooled
    /// `VulkanBuffer`'s bucket-rounded physical size. For a tiny tensor the
    /// pool bucket is >= 64 KiB, so if the bounce path (which sized the output
    /// at the logical length) had silently come back, or if `kt_tensor_from_vk`
    /// recorded the bucket size, this assertion catches it. Recording the
    /// bucket size would also overflow/garble any downstream byte-range slice
    /// — this is the load-bearing consumer invariant.
    #[test]
    fn vulkan_softmax_zero_copy_records_logical_byte_len() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let rows = 2usize;
        let hidden = 4usize;
        let logical_bytes = rows * hidden * 4; // 32 B, far below the >= 64 KiB pool bucket
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 2.5];
        let x = vk_f32(data, vec![rows, hidden]);
        let out = super::vulkan_softmax_last_axis(&x).expect("vulkan_softmax_last_axis");

        // The result is whole-buffer C-contiguous (zero-copy bridge invariant).
        assert!(
            out.is_contiguous(),
            "zero-copy softmax output must be contiguous"
        );
        assert_eq!(
            out.layout().start_offset(),
            0,
            "zero-copy softmax output must have start_offset == 0"
        );

        // The kt storage must report the LOGICAL byte length, not the pooled
        // bucket. `StorageBackend::byte_len()` is the source of truth.
        let storage = out.storage();
        let vk_storage = storage
            .as_any()
            .downcast_ref::<VulkanStorage>()
            .expect("zero-copy softmax output must be Vulkan-backed");
        assert_eq!(
            vk_storage.byte_len(),
            logical_bytes,
            "zero-copy output recorded byte_len {} but logical is {logical_bytes} \
             (pool bucket leaked into byte_len, or a bounce path returned)",
            vk_storage.byte_len()
        );

        // And the readback through the logical-slicing consumer is exact.
        let got = read_vk_f32(&out);
        assert_eq!(
            got.len(),
            rows * hidden,
            "readback length must be logical, not bucketed"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "softmax output not finite: {got:?}"
        );
    }

    #[test]
    fn vulkan_l2norm_pool_overflow_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let rows = 2usize;
        let hidden = 4usize;
        let eps = 1e-6f32;
        let data: Vec<f32> = vec![3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0];
        let x = vk_f32(data.clone(), vec![rows, hidden]);
        let out = super::vulkan_l2norm_last_axis(&x, eps).expect("vulkan_l2norm_last_axis");
        let got = read_vk_f32(&out);
        // CPU reference matches the shader convention exactly (vk_ops/l2norm.rs):
        //   y = scale * x / sqrt(sum(x^2) + eps),  scale = 1.0 (true L2, no /hidden).
        let mut want = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let row = &data[r * hidden..(r + 1) * hidden];
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0f32 / (ss + eps).sqrt();
            for i in 0..hidden {
                want[r * hidden + i] = row[i] * inv;
            }
        }
        assert!(
            got.iter().all(|v| v.is_finite()),
            "l2norm output not finite: {got:?}"
        );
        let err = max_abs_err(&got, &want);
        eprintln!("vulkan_l2norm_pool_overflow_parity: max_abs_err = {err:e}");
        assert!(
            err < 1e-5,
            "l2norm max_abs_err {err} too large; got={got:?} want={want:?}"
        );
    }

    #[test]
    fn vulkan_activation_silu_pool_overflow_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let data: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 1.0, 3.0];
        let x = vk_f32(data.clone(), vec![2, 3]);
        // kind_tag 0 = Silu
        let out = super::vulkan_activation_unary(&x, 0).expect("vulkan_activation_unary(silu)");
        let got = read_vk_f32(&out);
        // CPU reference: silu(x) = x * sigmoid(x).
        let want: Vec<f32> = data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "silu output not finite: {got:?}"
        );
        let err = max_abs_err(&got, &want);
        eprintln!("vulkan_activation_silu_pool_overflow_parity: max_abs_err = {err:e}");
        assert!(
            err < 1e-5,
            "silu max_abs_err {err} too large; got={got:?} want={want:?}"
        );
    }

    #[test]
    fn vulkan_elementwise_add_pool_overflow_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, -1.0, -2.0, -3.0];
        let a = vk_f32(a_data.clone(), vec![2, 3]);
        let b = vk_f32(b_data.clone(), vec![2, 3]);
        // kind_tag 0 = Add
        let out =
            super::vulkan_elementwise_binary(&a, &b, 0).expect("vulkan_elementwise_binary(add)");
        let got = read_vk_f32(&out);
        let want: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "add output not finite: {got:?}"
        );
        let err = max_abs_err(&got, &want);
        eprintln!("vulkan_elementwise_add_pool_overflow_parity: max_abs_err = {err:e}");
        assert!(
            err < 1e-5,
            "add max_abs_err {err} too large; got={got:?} want={want:?}"
        );
    }

    // ------------------------------------------------------------------
    // #1443 step1: F32-act × BF16-weight mixed-precision matmul.
    //
    // `vulkan_matmul_bf16w(x, W)` computes `out = x @ W.T` for an F32
    // activation `x [rows, K]` and a frozen BF16 weight `W [N, K]`. The
    // reference is the SAME BF16 weight cast back to F32 (so both paths see
    // the same BF16-rounded weight values), transposed to `[K, N]`, then run
    // through the F32 `vulkan_matmul` (PR3b). Tolerance ~2e-2 covers the BF16
    // weight precision. Bounded, single-shot tiny GPU test.
    // ------------------------------------------------------------------

    /// Build a BF16 Vulkan weight from F32 host data of shape `[n, k]`.
    fn vk_bf16_weight(data: Vec<f32>, n: usize, k: usize) -> crate::Tensor {
        let w_f32 = vk_f32(data, vec![n, k]);
        super::vulkan_cast(&w_f32, DType::BF16).expect("vulkan_cast f32->bf16 weight")
    }

    #[test]
    fn vulkan_matmul_bf16w_parity_2d() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        // x [rows=4, K=8] F32, W [N=6, K=8] BF16. out = x @ W.T = [4, 6].
        let (rows, k, n) = (4usize, 8usize, 6usize);
        let x_data: Vec<f32> = (0..rows * k).map(|i| (i as f32) * 0.1 - 1.3).collect();
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.25 - 0.5).collect();

        let x = vk_f32(x_data.clone(), vec![rows, k]);
        let w_bf16 = vk_bf16_weight(w_data.clone(), n, k);
        assert_eq!(w_bf16.dtype(), DType::BF16);

        // Mixed-precision path under test.
        let out = super::vulkan_matmul_bf16w(&x, &w_bf16).expect("vulkan_matmul_bf16w");
        assert_eq!(out.shape(), &[rows, n]);
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(
            out.device(),
            Device::Vulkan(0),
            "result must stay on Vulkan"
        );
        let got = read_vk_f32(&out);

        // Reference: same BF16 weight cast back to F32, transposed to [K, N],
        // then F32 vulkan_matmul (PR3b). This sees the identical BF16-rounded
        // weight, so any residual error is the bf16w kernel's accumulation.
        let w_f32_ref = super::vulkan_cast(&w_bf16, DType::F32).expect("vulkan_cast bf16->f32 ref");
        let w_t_ref = w_f32_ref
            .transpose(0, 1)
            .expect("transpose [N,K]->[K,N]")
            .contiguous()
            .expect("contiguous w_t");
        let ref_out = super::vulkan_matmul(&x, &w_t_ref).expect("vulkan_matmul reference");
        let want = read_vk_f32(&ref_out);

        assert!(
            got.iter().all(|v| v.is_finite()),
            "bf16w out not finite: {got:?}"
        );
        let err = max_abs_err(&got, &want);
        eprintln!("vulkan_matmul_bf16w_parity_2d: max_abs_err = {err:e}");
        assert!(
            err < 2e-2,
            "bf16w matmul diverges from F32-cast reference: max_abs_err={err}; got={got:?} want={want:?}"
        );
    }

    /// FD gradient check: central-difference `dL/dx` (loss = sum(out)) vs the
    /// `vulkan_matmul_bf16w_bwd` adjoint `dx = grad_out @ W` (grad_out = ones).
    /// Confirms the recorded backward's dx is correct. The weight is frozen —
    /// there is no `dW` to check (the recorder returns `None` for it).
    #[test]
    fn vulkan_matmul_bf16w_fd_dx() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let (rows, k, n) = (4usize, 8usize, 6usize);
        let x_data: Vec<f32> = (0..rows * k).map(|i| (i as f32) * 0.07 - 0.9).collect();
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.2 - 0.3).collect();

        let w_bf16 = vk_bf16_weight(w_data.clone(), n, k);

        // Analytic dx via the backward kernel. loss = sum(out) => grad_out = ones.
        let grad_out = vk_f32(vec![1.0f32; rows * n], vec![rows, n]);
        let dx =
            super::vulkan_matmul_bf16w_bwd(&grad_out, &w_bf16).expect("vulkan_matmul_bf16w_bwd");
        assert_eq!(dx.shape(), &[rows, k]);
        assert_eq!(dx.device(), Device::Vulkan(0));
        let dx_v = read_vk_f32(&dx);

        // Central-difference each x element through the forward.
        let eps = 1e-2f32;
        let loss_at = |xd: &[f32]| -> f32 {
            let x = vk_f32(xd.to_vec(), vec![rows, k]);
            let out = super::vulkan_matmul_bf16w(&x, &w_bf16).expect("fd forward");
            read_vk_f32(&out).iter().sum()
        };
        let mut fd = vec![0.0f32; rows * k];
        for i in 0..rows * k {
            let mut xp = x_data.clone();
            let mut xm = x_data.clone();
            xp[i] += eps;
            xm[i] -= eps;
            fd[i] = (loss_at(&xp) - loss_at(&xm)) / (2.0 * eps);
        }

        let err = max_abs_err(&dx_v, &fd);
        eprintln!("vulkan_matmul_bf16w_fd_dx: max_abs_err = {err:e}");
        assert!(
            dx_v.iter().all(|v| v.is_finite()),
            "dx not finite: {dx_v:?}"
        );
        assert!(
            err < 2e-2,
            "bf16w dx diverges from finite-difference: max_abs_err={err}; analytic={dx_v:?} fd={fd:?}"
        );
    }

    #[test]
    fn vulkan_matmul_lhs_transposed_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let (k, m, n) = (33usize, 17usize, 19usize);
        let a_data: Vec<f32> = (0..k * m).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.027).cos()).collect();

        let a_cpu = crate::Tensor::from_vec(a_data.clone(), vec![k, m]).unwrap();
        let b_cpu = crate::Tensor::from_vec(b_data.clone(), vec![k, n]).unwrap();
        let a_vk = vk_f32(a_data.clone(), vec![k, m]);
        let b_vk = vk_f32(b_data.clone(), vec![k, n]);
        let want = crate::ops::matmul_lhs_transposed(&a_cpu, &b_cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let out = crate::ops::matmul_lhs_transposed(&a_vk, &b_vk).expect("vulkan lhs_t f32");
        assert_eq!(out.device(), Device::Vulkan(0));
        assert_eq!(out.shape(), &[m, n]);
        let got = read_vk_f32(&out);
        let err = max_abs_err(&got, &want);
        assert!(err < 1e-4, "vulkan lhs_t f32 max_abs_err={err}");

        let a_cpu_bf16 = crate::ops::cast(&a_cpu, DType::BF16).unwrap();
        let b_cpu_bf16 = crate::ops::cast(&b_cpu, DType::BF16).unwrap();
        let want_bf16_t = crate::ops::matmul_lhs_transposed(&a_cpu_bf16, &b_cpu_bf16).unwrap();
        let want_bf16 = crate::ops::cast(&want_bf16_t, DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let a_vk_bf16 = super::vulkan_cast(&a_vk, DType::BF16).expect("vulkan cast a bf16");
        let b_vk_bf16 = super::vulkan_cast(&b_vk, DType::BF16).expect("vulkan cast b bf16");
        let out_bf16 =
            crate::ops::matmul_lhs_transposed(&a_vk_bf16, &b_vk_bf16).expect("vulkan lhs_t bf16");
        assert_eq!(out_bf16.device(), Device::Vulkan(0));
        assert_eq!(out_bf16.dtype(), DType::BF16);
        assert_eq!(out_bf16.shape(), &[m, n]);
        let got_bf16 = super::vulkan_cast(&out_bf16, DType::F32)
            .expect("vulkan cast lhs_t bf16 out")
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let err = max_abs_err(&got_bf16, &want_bf16);
        assert!(err < 3e-2, "vulkan lhs_t bf16 max_abs_err={err}");
    }

    #[test]
    fn vulkan_matmul_rhs_transposed_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let (m, k, n) = (17usize, 33usize, 19usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.027).cos()).collect();

        let a_cpu = crate::Tensor::from_vec(a_data.clone(), vec![m, k]).unwrap();
        let b_cpu = crate::Tensor::from_vec(b_data.clone(), vec![n, k]).unwrap();
        let a_vk = vk_f32(a_data.clone(), vec![m, k]);
        let b_vk = vk_f32(b_data.clone(), vec![n, k]);
        let want = crate::ops::matmul_rhs_transposed(&a_cpu, &b_cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let out = crate::ops::matmul_rhs_transposed(&a_vk, &b_vk).expect("vulkan rhs_t f32");
        assert_eq!(out.device(), Device::Vulkan(0));
        assert_eq!(out.shape(), &[m, n]);
        let got = read_vk_f32(&out);
        let err = max_abs_err(&got, &want);
        assert!(err < 1e-4, "vulkan rhs_t f32 max_abs_err={err}");

        let a_cpu_bf16 = crate::ops::cast(&a_cpu, DType::BF16).unwrap();
        let b_cpu_bf16 = crate::ops::cast(&b_cpu, DType::BF16).unwrap();
        let want_bf16_t = crate::ops::matmul_rhs_transposed(&a_cpu_bf16, &b_cpu_bf16).unwrap();
        let want_bf16 = crate::ops::cast(&want_bf16_t, DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let a_vk_bf16 = super::vulkan_cast(&a_vk, DType::BF16).expect("vulkan cast a bf16");
        let b_vk_bf16 = super::vulkan_cast(&b_vk, DType::BF16).expect("vulkan cast b bf16");
        let out_bf16 =
            crate::ops::matmul_rhs_transposed(&a_vk_bf16, &b_vk_bf16).expect("vulkan rhs_t bf16");
        assert_eq!(out_bf16.device(), Device::Vulkan(0));
        assert_eq!(out_bf16.dtype(), DType::BF16);
        assert_eq!(out_bf16.shape(), &[m, n]);
        let got_bf16 = super::vulkan_cast(&out_bf16, DType::F32)
            .expect("vulkan cast rhs_t bf16 out")
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let err = max_abs_err(&got_bf16, &want_bf16);
        assert!(err < 3e-2, "vulkan rhs_t bf16 max_abs_err={err}");
    }

    /// On-device `contiguous()` (the `vk_gather_contiguous_f32` strided
    /// gather) must byte-match the CPU reference for the layouts that show up
    /// on the inference path: last-two-axis transpose (rank-2 and rank-4 =
    /// attention `k.t()`), a strided `narrow` (non-zero start_offset), and a
    /// broadcast-expand (stride-0 axis). The result must stay on Vulkan — no
    /// host bounce.
    #[test]
    fn vulkan_contiguous_gather_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let dev = Device::Vulkan(0);

        // Build a CPU tensor + the identical Vulkan tensor, apply `view` to
        // both, and check the on-device contiguous() matches the CPU ref AND
        // stays resident.
        fn check(
            dev: Device,
            data: &[f32],
            shape: &[usize],
            label: &str,
            view: impl Fn(&crate::Tensor) -> crate::Tensor,
        ) {
            let cpu = crate::Tensor::from_slice(data, shape.to_vec()).unwrap();
            let cpu_ref: Vec<f32> = view(&cpu).contiguous().unwrap().to_vec().unwrap();

            let vk = crate::Tensor::from_vec_on(dev, data.to_vec(), shape.to_vec()).unwrap();
            let vk_view = view(&vk);
            assert!(
                !vk_view.is_contiguous(),
                "{label}: view should be non-contiguous to exercise the gather"
            );
            let vk_contig = vk_view.contiguous().unwrap();
            assert_eq!(
                vk_contig.device(),
                dev,
                "{label}: result must stay on Vulkan"
            );
            assert!(
                vk_contig.is_contiguous(),
                "{label}: result must be contiguous"
            );
            let got: Vec<f32> = vk_contig.to_device(Device::Cpu).unwrap().to_vec().unwrap();
            assert_eq!(got.len(), cpu_ref.len(), "{label}: element count mismatch");
            let mut max_abs = 0.0f32;
            for (g, r) in got.iter().zip(cpu_ref.iter()) {
                max_abs = max_abs.max((g - r).abs());
            }
            eprintln!("vulkan_contiguous[{label}]: max_abs_err = {max_abs:e}");
            assert!(
                max_abs == 0.0,
                "{label}: gather diverges from CPU ref (max_abs={max_abs})"
            );
        }

        // rank-2 transpose: [3,4] -> view [4,3]
        let d2: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        check(dev, &d2, &[3, 4], "transpose_2d", |t| t.t().unwrap());

        // rank-4 attention k.t(): [2,3,5,4] -> [2,3,4,5]
        let d4: Vec<f32> = (0..2 * 3 * 5 * 4)
            .map(|i| (i % 13) as f32 * 0.25 - 1.0)
            .collect();
        check(dev, &d4, &[2, 3, 5, 4], "transpose_last2_rank4", |t| {
            t.transpose(2, 3).unwrap()
        });

        // strided narrow (non-zero start_offset): [4,6] -> narrow axis1 [4,3]@2
        let dn: Vec<f32> = (0..24).map(|i| i as f32).collect();
        check(dev, &dn, &[4, 6], "narrow_offset", |t| {
            t.narrow(1, 2, 3).unwrap()
        });

        // NOTE: broadcast/expand views (stride-0 axes) are reported
        // `is_contiguous() == true` by the kt layout regardless of which axis
        // is expanded, so `contiguous()` short-circuits and they never reach
        // the gather — they can't be exercised through this path and aren't
        // tested here. The gather covers genuinely-strided layouts (transpose,
        // narrow), validated above.
    }

    /// BF16 on-device `contiguous()` (the `vk_gather_contiguous_bf16` packed
    /// gather) must match the CPU reference. BF16 elements are packed into
    /// u32 words (one writer per output word) — this exercises the lane
    /// pack/unpack on transpose / narrow / odd-length-row layouts. Values are
    /// chosen exactly representable in BF16 so the comparison is byte-exact.
    #[test]
    fn vulkan_contiguous_gather_bf16_parity() {
        if maybe_vulkan_device().is_none() {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        }
        let dev = Device::Vulkan(0);

        fn check_bf16(
            dev: Device,
            data: &[f32],
            shape: &[usize],
            label: &str,
            view: impl Fn(&crate::Tensor) -> crate::Tensor,
        ) {
            // Build BF16 tensors on both devices from the same F32 source.
            let cpu = crate::Tensor::from_slice(data, shape.to_vec())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let cpu_ref: Vec<f32> = view(&cpu)
                .contiguous()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec()
                .unwrap();

            let vk = crate::Tensor::from_vec_on(dev, data.to_vec(), shape.to_vec())
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            assert_eq!(vk.device(), dev, "{label}: bf16 cast must stay on Vulkan");
            let vk_view = view(&vk);
            assert!(
                !vk_view.is_contiguous(),
                "{label}: view should be non-contiguous to exercise the gather"
            );
            let vk_contig = vk_view.contiguous().unwrap();
            assert_eq!(
                vk_contig.device(),
                dev,
                "{label}: result must stay on Vulkan"
            );
            assert_eq!(
                vk_contig.dtype(),
                DType::BF16,
                "{label}: dtype must be BF16"
            );
            assert!(
                vk_contig.is_contiguous(),
                "{label}: result must be contiguous"
            );
            let got: Vec<f32> = vk_contig
                .to_device(Device::Cpu)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec()
                .unwrap();
            assert_eq!(got.len(), cpu_ref.len(), "{label}: element count mismatch");
            let mut max_abs = 0.0f32;
            for (g, r) in got.iter().zip(cpu_ref.iter()) {
                max_abs = max_abs.max((g - r).abs());
            }
            eprintln!("vulkan_contiguous_bf16[{label}]: max_abs_err = {max_abs:e}");
            assert!(
                max_abs == 0.0,
                "{label}: bf16 gather diverges from CPU ref (max_abs={max_abs})"
            );
        }

        // BF16-exact integer values.
        // rank-2 transpose with ODD element count per word boundary: [3,5]->[5,3]
        let d2: Vec<f32> = (0..15).map(|i| i as f32).collect();
        check_bf16(dev, &d2, &[3, 5], "bf16_transpose_2d_odd", |t| {
            t.t().unwrap()
        });

        // rank-4 last-two transpose (attention k.t()): [2,3,5,4]->[2,3,4,5]
        let d4: Vec<f32> = (0..2 * 3 * 5 * 4).map(|i| (i % 64) as f32).collect();
        check_bf16(dev, &d4, &[2, 3, 5, 4], "bf16_transpose_last2_rank4", |t| {
            t.transpose(2, 3).unwrap()
        });

        // strided narrow (non-zero start_offset): [4,6]->[4,3]@2
        let dn: Vec<f32> = (0..24).map(|i| i as f32).collect();
        check_bf16(dev, &dn, &[4, 6], "bf16_narrow_offset", |t| {
            t.narrow(1, 2, 3).unwrap()
        });
    }
}
