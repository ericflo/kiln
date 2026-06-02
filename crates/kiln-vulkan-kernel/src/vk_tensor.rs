//! `VkTensor`: a GPU-resident tensor for vk-native training.
//!
//! Wraps an `Arc<VulkanBuffer>` with shape/dtype metadata and an optional
//! autograd link (`grad_fn`). Cloning is `Arc::clone` of the inner shell —
//! cheap, refcount-driven storage lifetime.
//!
//! Storage is always C-contiguous. Reshape is metadata-only; transpose and
//! other strided views are *physical* moves (their own dispatch).
//!
//! BF16 buffers are packed as a contiguous u16 sequence; shaders interpret
//! them as `u32[]` with 2 BF16 lanes per word (same convention as
//! `adamw_step_bf16.comp` and the resident-activation registry).

use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use half::bf16;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Re-export so callers outside `vk_tensor.rs` (e.g. `vk_autograd.rs`)
/// can refer to the parameter-id type without an explicit candle import.
/// Sourced from the dependency-free leaf crate `kiln-tensor-id`, which
/// breaks the would-be `kiln-tensor <-> kiln-vulkan-kernel` cargo path
/// cycle (see `kiln-tensor-id/src/lib.rs` for the cycle analysis). (#1082)
pub use kiln_tensor_id::TensorId;

/// Element type of a `VkTensor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VkDType {
    F32,
    Bf16,
}

impl VkDType {
    pub fn byte_size(self) -> usize {
        match self {
            VkDType::F32 => 4,
            VkDType::Bf16 => 2,
        }
    }
}

fn device_buffer_bytes(n_elements: usize, dtype: VkDType) -> usize {
    let mut bytes = (n_elements * dtype.byte_size()).max(dtype.byte_size());
    if dtype == VkDType::Bf16 {
        // BF16 kernels view storage as u32 words containing two logical
        // lanes. Round odd element counts up so the final word is addressable.
        bytes = ((bytes + 3) / 4) * 4;
    }
    bytes
}

/// Monotonic op-id allocator. Used for autograd topo ordering.
static NEXT_OP_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_op_id() -> u64 {
    NEXT_OP_ID.fetch_add(1, Ordering::Relaxed)
}

/// Backward op interface — one impl per forward op family.
pub trait VkBackwardOp: Send + Sync + std::fmt::Debug {
    fn op_name(&self) -> &'static str;
    /// Saved input tensors. Backward iterates these and returns one
    /// gradient per input (or `None` to skip).
    fn input_refs(&self) -> &[VkTensor];
    /// Compute input gradients given the gradient at this node's output.
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>>;
}

/// Inner shell. Refcounted; outer `VkTensor` is just an `Arc` clone.
pub struct VkTensorInner {
    pub(crate) storage: Arc<VulkanBuffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: VkDType,
    pub(crate) device: Arc<VulkanDevice>,
    pub(crate) grad_fn: Option<Arc<dyn VkBackwardOp>>,
    pub(crate) requires_grad: bool,
    pub(crate) op_id: u64,
    /// Set on parameter leaves so `backward()` can return gradients
    /// keyed by candle's `TensorId` (matches the existing optimizer
    /// dispatch path).
    pub(crate) param_id: Option<TensorId>,
}

impl std::fmt::Debug for VkTensorInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("requires_grad", &self.requires_grad)
            .field("op_id", &self.op_id)
            .field("is_param", &self.param_id.is_some())
            .field("grad_fn", &self.grad_fn.as_ref().map(|o| o.op_name()))
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct VkTensor(pub(crate) Arc<VkTensorInner>);

impl VkTensor {
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn dtype(&self) -> VkDType {
        self.0.dtype
    }

    pub fn device(&self) -> &Arc<VulkanDevice> {
        &self.0.device
    }

    pub fn buffer(&self) -> &Arc<VulkanBuffer> {
        &self.0.storage
    }

    pub fn op_id(&self) -> u64 {
        self.0.op_id
    }

    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    pub fn param_id(&self) -> Option<TensorId> {
        self.0.param_id
    }

    pub fn grad_fn(&self) -> Option<&Arc<dyn VkBackwardOp>> {
        self.0.grad_fn.as_ref()
    }

    pub fn num_elements(&self) -> usize {
        self.0.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.0.dtype.byte_size()
    }

    /// Strip the autograd link; result behaves like a fresh leaf.
    pub fn detach(&self) -> Self {
        VkTensor(Arc::new(VkTensorInner {
            storage: Arc::clone(&self.0.storage),
            shape: self.0.shape.clone(),
            dtype: self.0.dtype,
            device: Arc::clone(&self.0.device),
            grad_fn: None,
            requires_grad: false,
            op_id: next_op_id(),
            param_id: None,
        }))
    }

    /// Construct an output VkTensor from an op (buffer + shape + grad_fn).
    /// `requires_grad` is set if `grad_fn` is provided.
    pub fn from_op(
        storage: Arc<VulkanBuffer>,
        shape: Vec<usize>,
        dtype: VkDType,
        device: Arc<VulkanDevice>,
        grad_fn: Option<Arc<dyn VkBackwardOp>>,
    ) -> Self {
        let requires_grad = grad_fn.is_some();
        VkTensor(Arc::new(VkTensorInner {
            storage,
            shape,
            dtype,
            device,
            grad_fn,
            requires_grad,
            op_id: next_op_id(),
            param_id: None,
        }))
    }

    /// Construct a leaf VkTensor from an already-allocated buffer.
    pub fn from_buffer(
        storage: Arc<VulkanBuffer>,
        shape: Vec<usize>,
        dtype: VkDType,
        device: Arc<VulkanDevice>,
    ) -> Self {
        VkTensor(Arc::new(VkTensorInner {
            storage,
            shape,
            dtype,
            device,
            grad_fn: None,
            requires_grad: false,
            op_id: next_op_id(),
            param_id: None,
        }))
    }

    /// Construct a parameter leaf — `requires_grad=true`, gradients
    /// returned keyed by `param_id` from `backward()`.
    pub fn parameter(
        storage: Arc<VulkanBuffer>,
        shape: Vec<usize>,
        dtype: VkDType,
        device: Arc<VulkanDevice>,
        param_id: TensorId,
    ) -> Self {
        VkTensor(Arc::new(VkTensorInner {
            storage,
            shape,
            dtype,
            device,
            grad_fn: None,
            requires_grad: true,
            op_id: next_op_id(),
            param_id: Some(param_id),
        }))
    }

    /// Allocate a fresh device-local buffer (filled by the caller).
    pub fn alloc_uninit(
        device: Arc<VulkanDevice>,
        shape: Vec<usize>,
        dtype: VkDType,
    ) -> Result<Self> {
        let nelem: usize = shape.iter().product();
        let bytes = nelem * dtype.byte_size();
        let buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            bytes.max(1) as u64,
        )
        .context("VkTensor::alloc_uninit: device-local buffer")?;
        Ok(Self::from_buffer(Arc::new(buffer), shape, dtype, device))
    }

    /// Upload an f32 slice as a fresh F32 VkTensor leaf. Candle-free
    /// constructor for tests/examples that only need bytes → GPU; the
    /// F32 fast-path that used to live behind a `VkTensor::from_candle`
    /// candle-Tensor bridge before that bridge was deleted in #1082.
    pub fn from_f32_slice(
        data: &[f32],
        shape: Vec<usize>,
        device: Arc<VulkanDevice>,
    ) -> Result<Self> {
        let nelem: usize = shape.iter().product();
        anyhow::ensure!(
            data.len() == nelem,
            "VkTensor::from_f32_slice: {} elements for shape {:?} (expected {})",
            data.len(),
            shape,
            nelem
        );
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            device_buffer_bytes(nelem, VkDType::F32) as u64,
        )
        .context("VkTensor::from_f32_slice: device-local buffer")?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buffer,
            bytes,
        )
        .context("VkTensor::from_f32_slice: upload")?;
        Ok(Self::from_buffer(
            Arc::new(buffer),
            shape,
            VkDType::F32,
            device,
        ))
    }

    /// Upload an f32 source slice converted to BF16 as a fresh BF16
    /// VkTensor leaf. Candle-free; mirrors the BF16 path of
    /// [`Self::from_candle`] after a `to_dtype(BF16)` cast. (#1082)
    pub fn from_f32_slice_as_bf16(
        data: &[f32],
        shape: Vec<usize>,
        device: Arc<VulkanDevice>,
    ) -> Result<Self> {
        let nelem: usize = shape.iter().product();
        anyhow::ensure!(
            data.len() == nelem,
            "VkTensor::from_f32_slice_as_bf16: {} elements for shape {:?} (expected {})",
            data.len(),
            shape,
            nelem
        );
        let mut bytes: Vec<u8> = Vec::with_capacity(nelem * 2);
        for &v in data {
            bytes.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
        }
        let buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            device_buffer_bytes(nelem, VkDType::Bf16) as u64,
        )
        .context("VkTensor::from_f32_slice_as_bf16: device-local buffer")?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buffer,
            &bytes,
        )
        .context("VkTensor::from_f32_slice_as_bf16: upload")?;
        Ok(Self::from_buffer(
            Arc::new(buffer),
            shape,
            VkDType::Bf16,
            device,
        ))
    }

    /// Mint a fresh `TensorId` for use as a parameter id, without
    /// requiring callers to import candle. Now backed by the
    /// `kiln-tensor-id` leaf crate's atomic counter — no candle round
    /// trip required. (#1082)
    pub fn fresh_param_id() -> TensorId {
        TensorId::next()
    }

    /// Build a parameter leaf directly from an f32 slice. Candle-free
    /// replacement for the `Tensor::from_vec → VkTensor::from_candle →
    /// VkTensor::parameter` pattern used in tests/examples that only
    /// need a parameter for autograd. Mints a fresh `TensorId` via
    /// [`Self::fresh_param_id`]. (#1082)
    pub fn parameter_from_f32_slice(
        data: &[f32],
        shape: Vec<usize>,
        device: Arc<VulkanDevice>,
    ) -> Result<Self> {
        let leaf = Self::from_f32_slice(data, shape, device)?;
        Ok(Self::parameter(
            Arc::clone(leaf.buffer()),
            leaf.shape().to_vec(),
            leaf.dtype(),
            Arc::clone(leaf.device()),
            Self::fresh_param_id(),
        ))
    }

    /// Build a BF16 parameter leaf directly from an f32 slice (host-side
    /// `bf16::from_f32` cast). Candle-free replacement for the
    /// `Tensor::from_vec().to_dtype(BF16) → VkTensor::from_candle →
    /// VkTensor::parameter` pattern. (#1082)
    pub fn parameter_from_f32_slice_as_bf16(
        data: &[f32],
        shape: Vec<usize>,
        device: Arc<VulkanDevice>,
    ) -> Result<Self> {
        let leaf = Self::from_f32_slice_as_bf16(data, shape, device)?;
        Ok(Self::parameter(
            Arc::clone(leaf.buffer()),
            leaf.shape().to_vec(),
            leaf.dtype(),
            Arc::clone(leaf.device()),
            Self::fresh_param_id(),
        ))
    }

    /// Upload raw little-endian bytes as a fresh `VkTensor` leaf.
    ///
    /// Candle-free general-purpose upload boundary: the caller lays the
    /// bytes out as the dtype expects (F32 = 4 bytes/element LE, BF16 =
    /// 2 bytes/element LE u16 bits) and we round the device allocation
    /// up to the BF16 word-pair size where needed. This is the canonical
    /// upload boundary for code that already has a flat byte buffer in
    /// the right shape — the kt host→Vulkan staging path
    /// (`kiln-tensor::host_to_vulkan_copy`) hands a packed byte image to
    /// this constructor. (The former candle bridge that also used it was
    /// deleted with `kiln-model::vk_forward` in PR7.) (#1082)
    pub fn from_bytes(
        bytes: &[u8],
        shape: Vec<usize>,
        dtype: VkDType,
        device: Arc<VulkanDevice>,
    ) -> Result<Self> {
        let nelem: usize = shape.iter().product();
        let expected = nelem * dtype.byte_size();
        anyhow::ensure!(
            bytes.len() == expected,
            "VkTensor::from_bytes: got {} bytes for shape {:?} dtype {:?} (expected {})",
            bytes.len(),
            shape,
            dtype,
            expected
        );
        let buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            device_buffer_bytes(nelem, dtype) as u64,
        )
        .context("VkTensor::from_bytes: device-local buffer")?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buffer,
            bytes,
        )
        .context("VkTensor::from_bytes: upload")?;
        Ok(Self::from_buffer(Arc::new(buffer), shape, dtype, device))
    }

    /// Read back raw little-endian bytes for this tensor. Candle-free
    /// counterpart to the now-deleted `to_candle`: callers that just
    /// want the flat byte buffer (e.g. to write a safetensors file,
    /// compare against a reference byte slice, or hand to a candle
    /// `Tensor::from_raw_buffer` at a higher layer that still owns
    /// candle) skip the candle round-trip. Bytes are truncated to
    /// `num_elements() * dtype.byte_size()` to strip the padding word
    /// that BF16 device buffers may carry. (#1082)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let dev = &self.0.device;
        let mut bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            &self.0.storage,
        )
        .context("VkTensor::to_bytes: read_back")?;
        let logical = self.num_elements() * self.0.dtype.byte_size();
        anyhow::ensure!(
            bytes.len() >= logical,
            "VkTensor::to_bytes: buffer holds {} bytes, expected at least {}",
            bytes.len(),
            logical
        );
        bytes.truncate(logical);
        Ok(bytes)
    }

    /// Read back as a flat Vec<f32>, converting BF16 to F32. For tests.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let dev = &self.0.device;
        let bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            &self.0.storage,
        )
        .context("VkTensor::to_vec_f32: read_back")?;
        let nelem = self.num_elements();
        match self.0.dtype {
            VkDType::F32 => {
                let mut data = Vec::with_capacity(nelem);
                for i in 0..nelem {
                    let off = i * 4;
                    data.push(f32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]));
                }
                Ok(data)
            }
            VkDType::Bf16 => {
                let mut data = Vec::with_capacity(nelem);
                for i in 0..nelem {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    data.push(f32::from_bits((bits as u32) << 16));
                }
                Ok(data)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vk_dev() -> Option<Arc<VulkanDevice>> {
        VulkanDevice::probe()
            .then(|| VulkanDevice::new().ok())
            .flatten()
            .map(Arc::new)
    }

    #[test]
    fn vk_tensor_f32_roundtrip() {
        let Some(dev) = vk_dev() else { return };
        let vt = VkTensor::from_f32_slice(
            &[1.0_f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            Arc::clone(&dev),
        )
        .unwrap();
        assert_eq!(vt.shape(), &[2, 2]);
        assert_eq!(vt.dtype(), VkDType::F32);
        let back = vt.to_vec_f32().unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vk_tensor_bf16_roundtrip() {
        let Some(dev) = vk_dev() else { return };
        let data: Vec<f32> = vec![1.0, 2.5, -3.25, 0.125];
        let vt =
            VkTensor::from_f32_slice_as_bf16(&data, vec![4], Arc::clone(&dev)).unwrap();
        let back = vt.to_vec_f32().unwrap();
        for (i, expected) in data.iter().enumerate() {
            // BF16 round-trip via host `bf16::from_f32` introduces the
            // usual ~1/256 mantissa rounding.
            let expected_bf16 = bf16::from_f32(*expected).to_f32();
            assert!(
                (back[i] - expected_bf16).abs() < 1e-6,
                "idx {i}: {} vs {}",
                back[i],
                expected_bf16
            );
        }
    }

    #[test]
    fn detach_clears_grad_fn() {
        let Some(dev) = vk_dev() else { return };
        let vt = VkTensor::from_f32_slice(&[1.0_f32], vec![1], dev).unwrap();
        let detached = vt.detach();
        assert!(detached.grad_fn().is_none());
        assert!(!detached.requires_grad());
    }

    #[test]
    fn vk_tensor_from_bytes_f32_matches_from_f32_slice() {
        let Some(dev) = vk_dev() else { return };
        let data: Vec<f32> = vec![3.14, -1.0, 2.718, 0.0];
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let vt_bytes = VkTensor::from_bytes(
            &bytes,
            vec![data.len()],
            VkDType::F32,
            Arc::clone(&dev),
        )
        .unwrap();
        let vt_slice =
            VkTensor::from_f32_slice(&data, vec![data.len()], Arc::clone(&dev)).unwrap();
        let from_bytes = vt_bytes.to_vec_f32().unwrap();
        let from_slice = vt_slice.to_vec_f32().unwrap();
        assert_eq!(from_bytes, from_slice);
        assert_eq!(from_bytes, data);
    }

    #[test]
    fn vk_tensor_to_bytes_truncates_to_logical_size() {
        let Some(dev) = vk_dev() else { return };
        // 3 BF16 elements = 6 logical bytes, but the device buffer is
        // padded to a 4-byte boundary internally. `to_bytes()` must
        // truncate to 6 bytes.
        let data = vec![1.0_f32, 2.0, 3.0];
        let vt = VkTensor::from_f32_slice_as_bf16(&data, vec![3], dev).unwrap();
        let bytes = vt.to_bytes().unwrap();
        assert_eq!(bytes.len(), 6);
    }
}
