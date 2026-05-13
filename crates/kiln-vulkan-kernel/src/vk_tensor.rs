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
use candle_core::{CpuStorage, DType, Device, Storage, Tensor, TensorId};
use half::bf16;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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

    pub fn to_candle(self) -> DType {
        match self {
            VkDType::F32 => DType::F32,
            VkDType::Bf16 => DType::BF16,
        }
    }

    pub fn from_candle(d: DType) -> Result<Self> {
        match d {
            DType::F32 => Ok(VkDType::F32),
            DType::BF16 => Ok(VkDType::Bf16),
            other => anyhow::bail!("VkTensor: unsupported candle dtype {:?}", other),
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

    /// Upload a candle Tensor to GPU as a fresh VkTensor leaf.
    ///
    /// Only F32 and BF16 are accepted. The tensor is forced contiguous;
    /// strided views are materialized first.
    pub fn from_candle(t: &Tensor, device: Arc<VulkanDevice>) -> Result<Self> {
        let dtype = VkDType::from_candle(t.dtype())?;
        if let Some((bytes_vec, shape)) = contiguous_cpu_tensor_bytes(t, dtype)? {
            let nelem = shape.iter().product();
            let buffer = VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                device_buffer_bytes(nelem, dtype) as u64,
            )
            .context("VkTensor::from_candle: device-local buffer")?;
            VulkanBuffer::upload_data(
                device.device(),
                device.host_visible_mem_type(),
                device.queue(),
                device.queue_family_index(),
                &buffer,
                &bytes_vec,
            )
            .context("VkTensor::from_candle: upload")?;
            return Ok(Self::from_buffer(Arc::new(buffer), shape, dtype, device));
        }

        let t = t
            .contiguous()
            .context("VkTensor::from_candle: contiguous")?;
        let shape: Vec<usize> = t.dims().to_vec();
        let nelem: usize = shape.iter().product();
        let bytes_vec = match dtype {
            VkDType::F32 => {
                let data: Vec<f32> = t
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .context("VkTensor::from_candle: f32 readout")?;
                let mut bytes = Vec::with_capacity(nelem * 4);
                for v in data {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            }
            VkDType::Bf16 => {
                let data: Vec<bf16> = t
                    .flatten_all()?
                    .to_vec1::<bf16>()
                    .context("VkTensor::from_candle: bf16 readout")?;
                let mut bytes = Vec::with_capacity(nelem * 2);
                for v in data {
                    bytes.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                bytes
            }
        };
        let buffer = VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            device_buffer_bytes(nelem, dtype) as u64,
        )
        .context("VkTensor::from_candle: device-local buffer")?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buffer,
            &bytes_vec,
        )
        .context("VkTensor::from_candle: upload")?;
        Ok(Self::from_buffer(Arc::new(buffer), shape, dtype, device))
    }

    /// Read back to a candle Tensor on CPU. Used at save/parity boundaries.
    pub fn to_candle(&self) -> Result<Tensor> {
        let dev = &self.0.device;
        let bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            &self.0.storage,
        )
        .context("VkTensor::to_candle: read_back")?;
        let nelem = self.num_elements();
        match self.0.dtype {
            VkDType::F32 => {
                anyhow::ensure!(
                    bytes.len() >= nelem * 4,
                    "VkTensor::to_candle f32: buffer holds {} bytes, expected {}",
                    bytes.len(),
                    nelem * 4
                );
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
                Tensor::from_vec(data, self.0.shape.clone(), &Device::Cpu)
                    .context("VkTensor::to_candle: f32 tensor build")
            }
            VkDType::Bf16 => {
                anyhow::ensure!(
                    bytes.len() >= nelem * 2,
                    "VkTensor::to_candle bf16: buffer holds {} bytes, expected {}",
                    bytes.len(),
                    nelem * 2
                );
                let mut data = Vec::with_capacity(nelem);
                for i in 0..nelem {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                    data.push(bf16::from_bits(bits));
                }
                Tensor::from_vec(data, self.0.shape.clone(), &Device::Cpu)
                    .context("VkTensor::to_candle: bf16 tensor build")
            }
        }
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

fn contiguous_cpu_tensor_bytes(
    tensor: &Tensor,
    dtype: VkDType,
) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
    let (storage, layout) = tensor.storage_and_layout();
    let Some((start, end)) = layout.contiguous_offsets() else {
        return Ok(None);
    };
    let shape = layout.shape().dims().to_vec();
    match (&*storage, dtype) {
        (Storage::Cpu(CpuStorage::F32(data)), VkDType::F32) => {
            anyhow::ensure!(
                end <= data.len(),
                "VkTensor::from_candle: f32 CPU storage range {start}..{end} exceeds len {}",
                data.len()
            );
            Ok(Some((
                bytemuck::cast_slice(&data[start..end]).to_vec(),
                shape,
            )))
        }
        (Storage::Cpu(CpuStorage::BF16(data)), VkDType::Bf16) => {
            anyhow::ensure!(
                end <= data.len(),
                "VkTensor::from_candle: bf16 CPU storage range {start}..{end} exceeds len {}",
                data.len()
            );
            let slice = &data[start..end];
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(slice),
                )
                .to_vec()
            };
            Ok(Some((bytes, shape)))
        }
        _ => Ok(None),
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
        let t = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        let vt = VkTensor::from_candle(&t, Arc::clone(&dev)).unwrap();
        assert_eq!(vt.shape(), &[2, 2]);
        assert_eq!(vt.dtype(), VkDType::F32);
        let back = vt.to_vec_f32().unwrap();
        assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vk_tensor_bf16_roundtrip() {
        let Some(dev) = vk_dev() else { return };
        let data: Vec<bf16> = vec![1.0, 2.5, -3.25, 0.125]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let t = Tensor::from_vec(data.clone(), (4,), &Device::Cpu).unwrap();
        let vt = VkTensor::from_candle(&t, Arc::clone(&dev)).unwrap();
        let back = vt.to_vec_f32().unwrap();
        for (i, expected) in data.iter().enumerate() {
            assert!(
                (back[i] - expected.to_f32()).abs() < 1e-6,
                "idx {i}: {} vs {}",
                back[i],
                expected
            );
        }
    }

    #[test]
    fn detach_clears_grad_fn() {
        let Some(dev) = vk_dev() else { return };
        let t = Tensor::from_vec(vec![1.0_f32], (1,), &Device::Cpu).unwrap();
        let vt = VkTensor::from_candle(&t, dev).unwrap();
        let detached = vt.detach();
        assert!(detached.grad_fn().is_none());
        assert!(!detached.requires_grad());
    }
}
