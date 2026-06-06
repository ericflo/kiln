//! Vulkan kt tensor byte/upload/readback bridge helpers.

use anyhow::{Context, Result};
use std::sync::Arc;

struct F32TensorUpload<'a> {
    bytes: &'a [u8],
    shape: Vec<usize>,
}

fn cpu_contiguous_f32_tensor_upload_vk(t: &kiln_tensor::Tensor) -> Option<F32TensorUpload<'_>> {
    if !matches!(t.device(), kiln_tensor::Device::Cpu)
        || t.dtype() != kiln_tensor::DType::F32
        || !t.is_contiguous()
    {
        return None;
    }

    let per = t.dtype().size_in_bytes();
    let n = t.element_count();
    let start = t.layout().start_offset().checked_mul(per)?;
    let len = n.checked_mul(per)?;
    let end = start.checked_add(len)?;
    let storage = t
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()?;
    let bytes = storage.as_bytes().get(start..end)?;
    Some(F32TensorUpload {
        bytes,
        shape: t.shape().to_vec(),
    })
}

fn upload_f32_tensors_from_cpu_bytes_vk(
    vk_device: &Arc<kiln_vulkan_kernel::VulkanDevice>,
    specs: &[F32TensorUpload<'_>],
) -> Result<Vec<kiln_vulkan_kernel::vk_tensor::VkTensor>> {
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};
    use kiln_vulkan_kernel::VulkanBuffer;

    let mut buffers = Vec::with_capacity(specs.len());
    for spec in specs {
        let nelem = spec.shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("Vulkan F32 upload shape overflow"))
        })?;
        let expected = nelem
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("Vulkan F32 upload byte-size overflow"))?;
        anyhow::ensure!(
            spec.bytes.len() == expected,
            "Vulkan F32 upload byte length mismatch: got {}, expected {} for shape {:?}",
            spec.bytes.len(),
            expected,
            spec.shape
        );
        let buffer = VulkanBuffer::create_device_local(
            vk_device.device(),
            vk_device.device_local_mem_type(),
            expected.max(std::mem::size_of::<f32>()) as u64,
        )
        .context("Vulkan F32 batch upload: create device-local buffer")?;
        buffers.push(Arc::new(buffer));
    }

    let uploads: Vec<(&VulkanBuffer, &[u8])> = buffers
        .iter()
        .zip(specs.iter())
        .map(|(buffer, spec)| (buffer.as_ref(), spec.bytes))
        .collect();
    VulkanBuffer::upload_data_batch(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &uploads,
    )
    .context("Vulkan F32 batch upload")?;

    Ok(buffers
        .into_iter()
        .zip(specs.iter())
        .map(|(buffer, spec)| {
            VkTensor::from_buffer(
                buffer,
                spec.shape.clone(),
                VkDType::F32,
                Arc::clone(vk_device),
            )
        })
        .collect())
}

pub(super) fn upload_gdn_chunkwise_inputs_from_cpu_bytes_vk(
    vk_device: &Arc<kiln_vulkan_kernel::VulkanDevice>,
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> Result<Option<[kiln_vulkan_kernel::vk_tensor::VkTensor; 6]>> {
    let Some(q_upload) = cpu_contiguous_f32_tensor_upload_vk(q) else {
        return Ok(None);
    };
    let Some(k_upload) = cpu_contiguous_f32_tensor_upload_vk(k) else {
        return Ok(None);
    };
    let Some(v_upload) = cpu_contiguous_f32_tensor_upload_vk(v) else {
        return Ok(None);
    };
    let Some(beta_upload) = cpu_contiguous_f32_tensor_upload_vk(beta) else {
        return Ok(None);
    };
    let Some(g_upload) = cpu_contiguous_f32_tensor_upload_vk(g) else {
        return Ok(None);
    };
    let Some(state_upload) = cpu_contiguous_f32_tensor_upload_vk(state) else {
        return Ok(None);
    };
    let uploads = [
        q_upload,
        k_upload,
        v_upload,
        beta_upload,
        g_upload,
        state_upload,
    ];
    let tensors = upload_f32_tensors_from_cpu_bytes_vk(vk_device, &uploads)?;
    tensors.try_into().map(Some).map_err(|tensors: Vec<_>| {
        anyhow::anyhow!(
            "Vulkan GDN chunkwise input upload produced {} tensors, expected 6",
            tensors.len()
        )
    })
}

fn cpu_f32_tensor_from_vk_bytes_vk(
    tensor: &kiln_vulkan_kernel::vk_tensor::VkTensor,
    mut bytes: Vec<u8>,
    context: &'static str,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        tensor.dtype() == kiln_vulkan_kernel::vk_tensor::VkDType::F32,
        "{context}: expected F32 VkTensor, got {:?}",
        tensor.dtype()
    );
    let logical = tensor
        .num_elements()
        .checked_mul(tensor.dtype().byte_size())
        .ok_or_else(|| anyhow::anyhow!("{context}: byte-size overflow"))?;
    anyhow::ensure!(
        bytes.len() >= logical,
        "{context}: readback produced {} bytes, expected at least {}",
        bytes.len(),
        logical
    );
    bytes.truncate(logical);
    let shape = tensor.shape().to_vec();
    kiln_tensor::Tensor::from_raw_bytes_on(
        kiln_tensor::Device::Cpu,
        kiln_tensor::DType::F32,
        bytes,
        shape,
    )
    .map_err(|e| anyhow::anyhow!("{context}: rebuild CPU kt tensor from bytes: {e}"))
}

#[cfg(test)]
fn vk_f32_tensor_to_cpu_tensor_vk(
    tensor: &kiln_vulkan_kernel::vk_tensor::VkTensor,
    context: &'static str,
) -> Result<kiln_tensor::Tensor> {
    let bytes = tensor
        .to_bytes()
        .with_context(|| format!("{context}: read bytes"))?;
    cpu_f32_tensor_from_vk_bytes_vk(tensor, bytes, context)
}

pub(super) fn vk_f32_tensors_to_cpu_tensors_batched_vk(
    tensors: &[(&kiln_vulkan_kernel::vk_tensor::VkTensor, &'static str)],
) -> Result<Vec<kiln_tensor::Tensor>> {
    use kiln_vulkan_kernel::VulkanBuffer;

    if tensors.is_empty() {
        return Ok(Vec::new());
    }

    let vk_device = tensors[0].0.device();
    for (tensor, context) in tensors {
        anyhow::ensure!(
            tensor.dtype() == kiln_vulkan_kernel::vk_tensor::VkDType::F32,
            "{context}: expected F32 VkTensor, got {:?}",
            tensor.dtype()
        );
        anyhow::ensure!(
            Arc::ptr_eq(tensor.device(), vk_device),
            "{context}: batched readback requires one Vulkan device"
        );
    }

    let buffers: Vec<&VulkanBuffer> = tensors
        .iter()
        .map(|(tensor, _)| tensor.buffer().as_ref())
        .collect();
    let raw_bytes = VulkanBuffer::read_back_batch(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        &buffers,
    )
    .context("Vulkan F32 batch readback")?;
    anyhow::ensure!(
        raw_bytes.len() == tensors.len(),
        "Vulkan F32 batch readback returned {} buffers, expected {}",
        raw_bytes.len(),
        tensors.len()
    );

    let mut out = Vec::with_capacity(tensors.len());
    for (idx, bytes) in raw_bytes.into_iter().enumerate() {
        let (tensor, context) = tensors[idx];
        out.push(cpu_f32_tensor_from_vk_bytes_vk(tensor, bytes, context)?);
    }
    Ok(out)
}

/// kt-native f32 byte + shape extraction.
/// shape straight from a kt tensor, no candle bridge. (#1082)
#[inline]
pub(super) fn kt_tensor_to_f32_bytes_with_shape(
    tensor: &kiln_tensor::Tensor,
) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape: Vec<usize> = tensor.shape().to_vec();
    let f32_data = tensor
        .flatten_all()
        .context("kt flatten_all")?
        .to_dtype(kiln_tensor::DType::F32)
        .context("kt to f32")?
        .to_vec1::<f32>()
        .context("kt to_vec1 f32")?;
    Ok((bytemuck::cast_slice(&f32_data).to_vec(), shape))
}

/// Resident single-sequence (batch==1) causal prefill SDPA — the buffer-based,
/// fully on-device replacement for the bytes path in `flash_attn_prefill_vulkan`.
///
/// q/k/v arrive `[1, seq_len, num_heads, head_dim]` (k/v already GQA-expanded
/// to `num_heads` by the caller). Squeeze the batch axis to the rank-3
/// `[seq, heads, head_dim]` layout `vk_sdpa_prefill` consumes, bridge zero-copy
/// (F32, contiguous, whole-buffer), run the fused on-device SDPA, bridge the
/// result back, and restore the `[1, seq, heads, head_dim]` shape + input
/// dtype. No D2H/H2D round-trip.
#[allow(clippy::too_many_arguments)]
pub(super) fn resident_sdpa_prefill_b1(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<kiln_tensor::Tensor> {
    use kiln_tensor::{kt_tensor_from_vk, vk_tensor_from_kt, DType, Device};

    // F32, [seq, heads, head_dim], contiguous, start_offset==0 (the bridge
    // contract). `to_dtype`/`contiguous` are resident on Vulkan (no host hop);
    // both are no-ops when the activation is already F32 + contiguous.
    let prep = |t: &kiln_tensor::Tensor| -> Result<kiln_tensor::Tensor> {
        Ok(t.to_dtype(DType::F32)?
            .reshape((seq_len, num_heads, head_dim))?
            .contiguous()?)
    };
    let qf = prep(q)?;
    let kf = prep(k)?;
    let vf = prep(v)?;
    let device_index = match qf.device() {
        Device::Vulkan(i) => i,
        _ => anyhow::bail!("resident_sdpa_prefill_b1: not Vulkan-resident after prep"),
    };

    let vk_q = vk_tensor_from_kt(&qf)?;
    let vk_k = vk_tensor_from_kt(&kf)?;
    let vk_v = vk_tensor_from_kt(&vf)?;
    let vk_out =
        kiln_vulkan_kernel::vk_ops::attention::vk_sdpa_prefill(&vk_q, &vk_k, &vk_v, scale)?;
    let out = kt_tensor_from_vk(&vk_out, device_index)?
        .reshape((1usize, seq_len, num_heads, head_dim))?;
    Ok(if q.dtype() == DType::F32 {
        out
    } else {
        out.to_dtype(q.dtype())?
    })
}

/// kt-native: wrap f32 bytes as a kt tensor
/// (CPU-host, the Vulkan activation residency), no candle bridge. (#1082)
#[inline]
pub(super) fn kt_tensor_from_f32_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: kiln_tensor::DType,
) -> Result<kiln_tensor::Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    let t = kiln_tensor::Tensor::from_vec(f32_data.to_vec(), shape.to_vec())
        .map_err(|e| anyhow::anyhow!("kt_tensor_from_f32_bytes: from_vec: {e}"))?;
    if dtype == kiln_tensor::DType::F32 {
        Ok(t)
    } else {
        t.to_dtype(dtype)
            .map_err(|e| anyhow::anyhow!("kt_tensor_from_f32_bytes: to_dtype: {e}"))
    }
}

/// kt-native packed bf16 extraction with shape — mirrors the tuple shape of
/// `kiln_vulkan_kernel::kernels::extract_tensor_packed_bf16_bytes_pub`
/// so call sites that use the `.0` (bytes) projection stay identical.
#[inline]
pub(super) fn kt_tensor_to_packed_bf16_bytes_with_shape(
    tensor: &kiln_tensor::Tensor,
) -> Result<(Vec<u8>, Vec<usize>)> {
    anyhow::ensure!(
        tensor.dtype() == kiln_tensor::DType::BF16,
        "packed bf16 upload requires BF16 tensor, got {:?}",
        tensor.dtype()
    );
    let shape: Vec<usize> = tensor.shape().to_vec();
    let flat = tensor.flatten_all().context("failed to flatten tensor")?;
    let bf16_data = flat
        .to_vec1::<half::bf16>()
        .context("failed to extract bf16 data")?;
    let mut packed = Vec::with_capacity(bf16_data.len().div_ceil(2));
    for pair in bf16_data.chunks(2) {
        let lo = pair[0].to_bits() as u32;
        let hi = pair.get(1).map(|v| v.to_bits() as u32).unwrap_or(0);
        packed.push(lo | (hi << 16));
    }
    Ok((bytemuck::cast_slice(&packed).to_vec(), shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_contiguous_f32_tensor_upload_borrows_exact_bytes() -> Result<()> {
        let tensor = kiln_tensor::Tensor::from_slice(&[1.25f32, -2.5, 3.75], vec![3])?;
        let upload = cpu_contiguous_f32_tensor_upload_vk(&tensor)
            .expect("contiguous CPU F32 tensor should expose upload bytes");
        let expected = [1.25f32, -2.5, 3.75]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<u8>>();

        assert_eq!(upload.shape, vec![3]);
        assert_eq!(upload.bytes, expected.as_slice());
        Ok(())
    }

    #[test]
    fn cpu_contiguous_f32_tensor_upload_rejects_non_f32() -> Result<()> {
        let tensor = kiln_tensor::Tensor::from_slice(
            &[half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)],
            vec![2],
        )?;
        assert!(cpu_contiguous_f32_tensor_upload_vk(&tensor).is_none());
        Ok(())
    }

    #[test]
    fn gdn_chunkwise_batched_input_upload_round_trips_on_vulkan() -> Result<()> {
        let Ok(vk_device) = kiln_vulkan_kernel::VulkanDevice::new() else {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let q_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let k_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let v_data = vec![9.0f32, 10.0, 11.0, 12.0];
        let beta_data = vec![0.25f32, 0.5];
        let g_data = vec![0.75f32, 1.0];
        let state_data = vec![13.0f32, 14.0, 15.0, 16.0];
        let q = kiln_tensor::Tensor::from_slice(&q_data, vec![1, 1, 2, 2])?;
        let k = kiln_tensor::Tensor::from_slice(&k_data, vec![1, 1, 2, 2])?;
        let v = kiln_tensor::Tensor::from_slice(&v_data, vec![1, 1, 2, 2])?;
        let beta = kiln_tensor::Tensor::from_slice(&beta_data, vec![1, 1, 2])?;
        let g = kiln_tensor::Tensor::from_slice(&g_data, vec![1, 1, 2])?;
        let state = kiln_tensor::Tensor::from_slice(&state_data, vec![1, 1, 2, 2])?;

        let [q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk] =
            upload_gdn_chunkwise_inputs_from_cpu_bytes_vk(
                &vk_device, &q, &k, &v, &beta, &g, &state,
            )?
            .expect("contiguous F32 inputs should use batched upload");

        assert_eq!(q_vk.to_vec_f32()?, q_data);
        assert_eq!(k_vk.to_vec_f32()?, k_data);
        assert_eq!(v_vk.to_vec_f32()?, v_data);
        assert_eq!(beta_vk.to_vec_f32()?, beta_data);
        assert_eq!(g_vk.to_vec_f32()?, g_data);
        assert_eq!(state_vk.to_vec_f32()?, state_data);
        Ok(())
    }

    #[test]
    fn vk_f32_tensor_to_cpu_tensor_rebuilds_from_raw_bytes() -> Result<()> {
        let Ok(vk_device) = kiln_vulkan_kernel::VulkanDevice::new() else {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        };
        let data = vec![1.5f32, -2.25, 3.75, 4.5];
        let vk_tensor = kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice(
            &data,
            vec![2, 2],
            Arc::new(vk_device),
        )?;
        let cpu_tensor = vk_f32_tensor_to_cpu_tensor_vk(&vk_tensor, "test raw byte readback")?;

        assert_eq!(cpu_tensor.shape(), &[2, 2]);
        assert_eq!(cpu_tensor.flatten_all()?.to_vec1::<f32>()?, data);
        Ok(())
    }

    #[test]
    fn vk_f32_tensors_to_cpu_tensors_batched_rebuilds_from_raw_bytes() -> Result<()> {
        let Ok(vk_device) = kiln_vulkan_kernel::VulkanDevice::new() else {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let out_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let state_data = vec![5.5f32, 6.5, 7.5, 8.5, 9.5, 10.5];
        let out_vk = kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice(
            &out_data,
            vec![2, 2],
            Arc::clone(&vk_device),
        )?;
        let state_vk = kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice(
            &state_data,
            vec![1, 2, 3],
            vk_device,
        )?;

        let [out_cpu, state_cpu]: [kiln_tensor::Tensor; 2] =
            vk_f32_tensors_to_cpu_tensors_batched_vk(&[
                (&out_vk, "test batched output readback"),
                (&state_vk, "test batched state readback"),
            ])?
            .try_into()
            .map_err(|readbacks: Vec<_>| {
                anyhow::anyhow!("test batched readback returned {} tensors", readbacks.len())
            })?;

        assert_eq!(out_cpu.shape(), &[2, 2]);
        assert_eq!(state_cpu.shape(), &[1, 2, 3]);
        assert_eq!(out_cpu.flatten_all()?.to_vec1::<f32>()?, out_data);
        assert_eq!(state_cpu.flatten_all()?.to_vec1::<f32>()?, state_data);
        Ok(())
    }
}
