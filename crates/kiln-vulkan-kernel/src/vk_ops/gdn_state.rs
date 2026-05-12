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
}
