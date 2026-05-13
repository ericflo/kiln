//! Per-layer Gated DeltaNet (GDN) recurrent + conv state.
//!
//! Mirrors candle's `LinearAttentionState` (forward.rs:1207) but holds
//! raw `Arc<VulkanBuffer>` so the GDN forward path can read/write
//! state in place without bouncing through candle Tensors.
//!
//! For training:
//!   - Each example starts from zero state.
//!   - State is a *leaf* in the autograd graph — `requires_grad =
//!     false`, the dS_in for the very first chunk is dropped (we
//!     don't train the initial recurrent state).
//!   - Phase 5 wires `vk_gdn_chunkwise` to thread state across chunks.

use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

/// One layer's recurrent + conv state buffers.
pub struct VkGdnLayerState {
    /// Recurrent state S, shape [batch, num_value_heads, head_dim_k, head_dim_v].
    /// Stored as F32 (matches candle training-time recurrent dtype).
    pub recurrent_state: Arc<VulkanBuffer>,
    pub recurrent_n_elements: usize,

    /// Conv1d sliding window, shape [batch, conv_channels, kernel_size - 1].
    /// Stored as F32 to match the conv1d kernel input dtype.
    pub conv_state: Arc<VulkanBuffer>,
    pub conv_n_elements: usize,
}

/// Whole-model GDN state — one per linear-attention layer.
///
/// FullAttention layers do not occupy entries here; the indexing is by
/// linear-attention layer index, not by overall layer index. Callers
/// must maintain the mapping (typically a `Vec<usize>` of GDN layer
/// indices, or simpler: only push to this vec for layers whose
/// `VkLayerWeights` is `LinearAttention`).
pub struct VkLinearAttentionState {
    pub layers: Vec<VkGdnLayerState>,
}

fn alloc_zeroed_f32(device: &Arc<VulkanDevice>, n_elements: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n_elements * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("VkLinearAttentionState: alloc state buffer")?;
    let zeros = vec![0u8; bytes];
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &buf,
        &zeros,
    )?;
    Ok(Arc::new(buf))
}

fn copy_device_buffer(
    device: &Arc<VulkanDevice>,
    src: &Arc<VulkanBuffer>,
    bytes: usize,
    label: &str,
) -> Result<Arc<VulkanBuffer>> {
    let dst = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes.max(4) as u64,
    )
    .with_context(|| format!("{label}: alloc dst buffer"))?;

    let command_pool = device.transient_command_pool()?;
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(*command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.device().handle(), &alloc_info, 1)
            .with_context(|| format!("{label}: allocate command buffer"))?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .device()
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().build())
            .with_context(|| format!("{label}: begin command buffer"))?;
        device.device().cmd_copy_buffer(
            cmd,
            src.handle(),
            dst.handle(),
            &[vk::BufferCopy::builder().size(bytes.max(4) as u64).build()],
        );
        device
            .device()
            .end_command_buffer(cmd)
            .with_context(|| format!("{label}: end command buffer"))?;
    }
    device.submit_and_wait(cmd, label)?;
    unsafe {
        device
            .device()
            .free_command_buffers(*command_pool, &command_buffers);
    }

    Ok(Arc::new(dst))
}

impl VkLinearAttentionState {
    /// Create fresh zero-initialized state for `num_gdn_layers` GDN
    /// layers, with the per-layer dimensions provided.
    ///
    /// Shapes per layer:
    ///   - recurrent_state: [batch, heads_v, head_dim_k, head_dim_v]
    ///   - conv_state:      [batch, conv_channels, kernel_size - 1]
    pub fn zeros(
        device: &Arc<VulkanDevice>,
        num_gdn_layers: usize,
        batch: usize,
        heads_v: usize,
        head_dim_k: usize,
        head_dim_v: usize,
        conv_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let recurrent_n = batch * heads_v * head_dim_k * head_dim_v;
        let state_len = kernel_size.saturating_sub(1).max(1);
        let conv_n = batch * conv_channels * state_len;
        let mut layers = Vec::with_capacity(num_gdn_layers);
        for _ in 0..num_gdn_layers {
            layers.push(VkGdnLayerState {
                recurrent_state: alloc_zeroed_f32(device, recurrent_n)?,
                recurrent_n_elements: recurrent_n,
                conv_state: alloc_zeroed_f32(device, conv_n)?,
                conv_n_elements: conv_n,
            });
        }
        Ok(Self { layers })
    }

    /// Capture a branchable GPU-resident copy of every GDN state buffer.
    ///
    /// GRPO reference scoring needs to prefill the shared prompt once and then
    /// evaluate several completion branches from the identical prompt state.
    /// This copy stays entirely on the Vulkan device; it does not read state
    /// back through CPU memory.
    pub fn snapshot(&self, device: &Arc<VulkanDevice>) -> Result<Self> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            layers.push(VkGdnLayerState {
                recurrent_state: copy_device_buffer(
                    device,
                    &layer.recurrent_state,
                    layer.recurrent_n_elements * 4,
                    &format!("VkLinearAttentionState snapshot recurrent layer {idx}"),
                )?,
                recurrent_n_elements: layer.recurrent_n_elements,
                conv_state: copy_device_buffer(
                    device,
                    &layer.conv_state,
                    layer.conv_n_elements * 4,
                    &format!("VkLinearAttentionState snapshot conv layer {idx}"),
                )?,
                conv_n_elements: layer.conv_n_elements,
            });
        }
        Ok(Self { layers })
    }
}
