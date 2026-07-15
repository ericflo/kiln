//! Vulkan resident decode session and linear-attention state helpers.
//!
//! These helpers track which K/V rows and linear-attention recurrent buffers
//! are already resident across decode steps. They remain inherent
//! `VulkanBackend` methods so forward/resident-decode call sites keep the same
//! method surface while `backend/vulkan.rs` stays focused on trait routing.

use anyhow::{Context, Result};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::vulkan::VulkanBackend;

impl VulkanBackend {
    /// Test (without mutating) whether the given full-attention layer
    /// has already been seeded into the resident KV cache for this
    /// session. Returns true after the first successful call to
    /// `mark_full_attn_layer_seeded` for the same `layer_idx`.
    pub fn full_attn_layer_seeded(&self, layer_idx: usize) -> bool {
        match self.seeded_full_attn_layers.lock() {
            Ok(g) => g.contains(&layer_idx),
            Err(_) => false,
        }
    }

    /// Mark the given full-attention layer as having been seeded into
    /// the resident KV cache for this session.
    pub fn mark_full_attn_layer_seeded(&self, layer_idx: usize) {
        if let Ok(mut g) = self.seeded_full_attn_layers.lock() {
            g.insert(layer_idx);
        }
    }

    /// Reset the seeded-layer set. Tests / multi-session callers call
    /// this when the kt paged cache may have been reset between
    /// resident decode calls; otherwise the resident path keeps reusing
    /// stale K/V state.
    pub fn reset_full_attn_seeded(&self) {
        if let Ok(mut g) = self.seeded_full_attn_layers.lock() {
            g.clear();
        }
        if let Ok(mut g) = self.seeded_resident_decode_rows.lock() {
            g.clear();
        }
    }

    pub fn resident_decode_row_seeded(&self, layer_idx: usize, row_id: u64) -> bool {
        match self.seeded_resident_decode_rows.lock() {
            Ok(g) => g.contains(&(layer_idx, row_id)),
            Err(_) => false,
        }
    }

    pub fn mark_resident_decode_row_seeded(&self, layer_idx: usize, row_id: u64) {
        if let Ok(mut g) = self.seeded_resident_decode_rows.lock() {
            g.insert((layer_idx, row_id));
        }
    }

    pub fn reset_resident_decode_row_seeded(&self) {
        if let Ok(mut g) = self.seeded_resident_decode_rows.lock() {
            g.clear();
        }
    }

    /// Note this resident decode call's `start_pos`. Within one
    /// request the resident path advances `start_pos` by 1 per token;
    /// a discontinuity (first call after server boot, or a new
    /// request whose first decode step doesn't follow the previous
    /// request's last step) signals a fresh session. At that point
    /// we clear the per-layer seeded flags so the next per-layer call
    /// re-seeds the resident `VkPagedKvCache` from this request's
    /// prefill. Returns `true` when a new session was detected.
    ///
    /// Without this, a second `/v1/chat/completions` request reuses
    /// the persistent `VkPagedKvCache` slot data the previous request
    /// wrote (because `seeded_full_attn_layers`, keyed only by layer
    /// index, is stuck at `true` from request 1). The model then
    /// reasons about the prior request's prompt.
    pub fn note_resident_session(&self, start_pos: usize) -> bool {
        let mut last = match self.last_resident_start_pos.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let is_new_session = match *last {
            // Same `start_pos` = another layer's call within the same
            // decode token. Same `start_pos + 1` = the next decode
            // step in the same request. Anything else = boundary.
            Some(prev) => start_pos != prev && start_pos != prev.wrapping_add(1),
            None => true,
        };
        // Only advance on a strictly-incrementing step so multi-layer
        // calls within the same token don't trigger a spurious reset.
        if match *last {
            Some(prev) => start_pos == prev.wrapping_add(1) || is_new_session,
            None => true,
        } {
            *last = Some(start_pos);
        }
        drop(last);
        if is_new_session {
            self.reset_full_attn_seeded();
            self.reset_linear_attn_seeded();
        }
        is_new_session
    }

    pub fn reset_linear_attn_seeded(&self) {
        if let Ok(mut g) = self.seeded_linear_attn_layers_kt.lock() {
            g.clear();
        }
    }

    pub fn linear_attn_recurrent_state_buffer_kt(
        &self,
        key: kiln_tensor::TensorId,
        bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .linear_attn_recurrent_state_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("kt recurrent state mutex poisoned"))?;
        if let Some(buf) = g.get(&key) {
            if buf.size() >= bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let arc = kiln_vulkan_kernel::buffer_pool::pool_alloc_device_local(dev, bytes)
            .context("acquire kt linear-attn recurrent state buffer")?;
        g.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    pub fn linear_attn_conv_state_buffer_kt(
        &self,
        key: kiln_tensor::TensorId,
        bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .linear_attn_conv_state_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("kt conv state mutex poisoned"))?;
        if let Some(buf) = g.get(&key) {
            if buf.size() >= bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let arc = kiln_vulkan_kernel::buffer_pool::pool_alloc_device_local(dev, bytes)
            .context("acquire kt linear-attn conv state buffer")?;
        g.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    pub fn linear_attn_layer_seeded_kt(&self, key: kiln_tensor::TensorId) -> bool {
        match self.seeded_linear_attn_layers_kt.lock() {
            Ok(g) => g.contains(&key),
            Err(_) => false,
        }
    }

    pub fn mark_linear_attn_layer_seeded_kt(&self, key: kiln_tensor::TensorId) {
        if let Ok(mut g) = self.seeded_linear_attn_layers_kt.lock() {
            g.insert(key);
        }
    }

    fn assemble_linear_attn_state_batch_kt(
        &self,
        state_map: &Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
        row_bytes: u64,
        label: &'static str,
    ) -> Result<bool> {
        if row_keys.is_empty() {
            return Ok(false);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        let row_buffers = {
            let g = state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?;
            let mut out = Vec::with_capacity(row_keys.len());
            for key in row_keys {
                let Some(buf) = g.get(key) else {
                    return Ok(false);
                };
                out.push(Arc::clone(buf));
            }
            out
        };
        let existing_batch = state_map
            .lock()
            .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?
            .get(&batch_key)
            .cloned();
        if let Some(batch_buffer) = existing_batch {
            kiln_vulkan_kernel::kernels::copy_device_buffer_rows_to_existing_batch(
                vk_device,
                &row_buffers,
                batch_buffer.as_ref(),
                row_bytes,
            )
            .with_context(|| format!("refresh kt {label} state batch rows"))?;
        } else {
            let batch_buffer = kiln_vulkan_kernel::kernels::copy_device_buffer_rows_to_batch(
                vk_device,
                &row_buffers,
                row_bytes,
            )
            .with_context(|| format!("assemble kt {label} state batch rows"))?;
            state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?
                .insert(batch_key, batch_buffer);
        }
        Ok(true)
    }

    fn scatter_linear_attn_state_batch_kt(
        &self,
        state_map: &Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
        row_bytes: u64,
        label: &'static str,
    ) -> Result<bool> {
        if row_keys.is_empty() {
            return Ok(false);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        let batch_buffer = {
            let g = state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?;
            let Some(buf) = g.get(&batch_key) else {
                return Ok(false);
            };
            Arc::clone(buf)
        };
        let row_buffers = {
            let g = state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?;
            let mut out = Vec::with_capacity(row_keys.len());
            for key in row_keys {
                let Some(buf) = g.get(key) else {
                    return Ok(false);
                };
                out.push(Arc::clone(buf));
            }
            out
        };
        kiln_vulkan_kernel::kernels::copy_device_buffer_batch_to_rows(
            vk_device,
            &batch_buffer,
            &row_buffers,
            row_bytes,
        )
        .with_context(|| format!("scatter kt {label} state batch rows"))?;
        Ok(true)
    }

    fn assemble_linear_attn_recurrent_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
        row_bytes: u64,
    ) -> Result<bool> {
        self.assemble_linear_attn_state_batch_kt(
            &self.linear_attn_recurrent_state_kt,
            row_keys,
            batch_key,
            row_bytes,
            "recurrent",
        )
    }

    fn assemble_linear_attn_conv_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
        row_bytes: u64,
    ) -> Result<bool> {
        self.assemble_linear_attn_state_batch_kt(
            &self.linear_attn_conv_state_kt,
            row_keys,
            batch_key,
            row_bytes,
            "conv",
        )
    }

    fn scatter_linear_attn_recurrent_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
        row_bytes: u64,
    ) -> Result<bool> {
        self.scatter_linear_attn_state_batch_kt(
            &self.linear_attn_recurrent_state_kt,
            batch_key,
            row_keys,
            row_bytes,
            "recurrent",
        )
    }

    fn scatter_linear_attn_conv_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
        row_bytes: u64,
    ) -> Result<bool> {
        self.scatter_linear_attn_state_batch_kt(
            &self.linear_attn_conv_state_kt,
            batch_key,
            row_keys,
            row_bytes,
            "conv",
        )
    }

    pub fn assemble_linear_attn_gdn_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
        recurrent_row_bytes: u64,
        conv_row_bytes: u64,
    ) -> Result<bool> {
        let recurrent_ok = self.assemble_linear_attn_recurrent_state_batch_kt(
            row_keys,
            batch_key,
            recurrent_row_bytes,
        )?;
        if !recurrent_ok {
            return Ok(false);
        }
        let conv_ok =
            self.assemble_linear_attn_conv_state_batch_kt(row_keys, batch_key, conv_row_bytes)?;
        if !conv_ok {
            return Ok(false);
        }
        self.mark_linear_attn_layer_seeded_kt(batch_key);
        Ok(true)
    }

    pub fn scatter_linear_attn_gdn_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
        recurrent_row_bytes: u64,
        conv_row_bytes: u64,
    ) -> Result<bool> {
        let recurrent_ok = self.scatter_linear_attn_recurrent_state_batch_kt(
            batch_key,
            row_keys,
            recurrent_row_bytes,
        )?;
        if !recurrent_ok {
            return Ok(false);
        }
        let conv_ok =
            self.scatter_linear_attn_conv_state_batch_kt(batch_key, row_keys, conv_row_bytes)?;
        if !conv_ok {
            return Ok(false);
        }
        if let Ok(mut seeded) = self.seeded_linear_attn_layers_kt.lock() {
            for key in row_keys {
                seeded.insert(*key);
            }
        }
        Ok(true)
    }

    pub fn seed_linear_attn_gdn_state_kt(
        &self,
        recurrent_t: &kiln_tensor::Tensor,
        conv_t: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        let key = recurrent_t.id();
        let recurrent_bytes = (recurrent_t.elem_count() * std::mem::size_of::<f32>()) as u64;
        let conv_bytes = (conv_t.elem_count() * std::mem::size_of::<f32>()) as u64;
        let recurrent_buf = self.linear_attn_recurrent_state_buffer_kt(key, recurrent_bytes)?;
        let conv_buf = self.linear_attn_conv_state_buffer_kt(key, conv_bytes)?;
        crate::vk_decode_resident::seed_recurrent_state_kt(vk_device, &recurrent_buf, recurrent_t)?;
        crate::vk_decode_resident::seed_conv_state_kt(vk_device, &conv_buf, conv_t)?;
        self.mark_linear_attn_layer_seeded_kt(key);
        Ok(true)
    }

    pub fn has_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) -> bool {
        if !self.linear_attn_layer_seeded_kt(key) {
            return false;
        }
        let recurrent_present = self
            .linear_attn_recurrent_state_kt
            .lock()
            .map(|g| g.contains_key(&key))
            .unwrap_or(false);
        let conv_present = self
            .linear_attn_conv_state_kt
            .lock()
            .map(|g| g.contains_key(&key))
            .unwrap_or(false);
        recurrent_present && conv_present
    }

    /// Release both halves of one kt-resident linear-attention state.
    ///
    /// The recurrent tensor ID is the shared ownership key for the recurrent
    /// and convolution maps. Keeping either entry after its request or cached
    /// assembled batch is dropped retains device memory indefinitely.
    pub fn evict_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) {
        self.linear_attn_recurrent_state_kt
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&key);
        self.linear_attn_conv_state_kt
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&key);
        self.seeded_linear_attn_layers_kt
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&key);
    }

    #[cfg(test)]
    fn linear_attn_gdn_state_entry_counts(&self) -> (usize, usize, usize) {
        let recurrent = self
            .linear_attn_recurrent_state_kt
            .lock()
            .map(|states| states.len())
            .unwrap_or_default();
        let conv = self
            .linear_attn_conv_state_kt
            .lock()
            .map(|states| states.len())
            .unwrap_or_default();
        let seeded = self
            .seeded_linear_attn_layers_kt
            .lock()
            .map(|states| states.len())
            .unwrap_or_default();
        (recurrent, conv, seeded)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::VulkanBackend;

    #[test]
    fn evict_linear_attention_state_releases_both_buffers_and_seed_marker() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping live state-lifecycle test");
            return Ok(());
        }

        let recurrent = kiln_tensor::Tensor::from_vec(vec![1.0f32; 16], (1, 2, 2, 4))?;
        let conv = kiln_tensor::Tensor::from_vec(vec![2.0f32; 8], (1, 4, 2))?;
        assert!(backend.seed_linear_attn_gdn_state_kt(&recurrent, &conv)?);
        assert!(backend.has_linear_attn_gdn_state_kt(recurrent.id()));
        assert_eq!(backend.linear_attn_gdn_state_entry_counts(), (1, 1, 1));

        backend.evict_linear_attn_gdn_state_kt(recurrent.id());
        assert!(!backend.has_linear_attn_gdn_state_kt(recurrent.id()));
        assert_eq!(backend.linear_attn_gdn_state_entry_counts(), (0, 0, 0));

        // Terminal cleanup is intentionally idempotent so overlapping error
        // and cancellation fences cannot resurrect or double-own state.
        backend.evict_linear_attn_gdn_state_kt(recurrent.id());
        assert_eq!(backend.linear_attn_gdn_state_entry_counts(), (0, 0, 0));
        Ok(())
    }
}
