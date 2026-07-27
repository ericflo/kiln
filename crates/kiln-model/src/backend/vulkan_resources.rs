//! Vulkan resident resource cache helpers.
//!
//! These inherent `VulkanBackend` methods manage long-lived Vulkan buffers and
//! caches used by resident decode paths. Keeping them here leaves
//! `backend/vulkan.rs` focused on backend construction and `BackendRuntime`
//! routing while preserving the existing public method surface.

use anyhow::{Context, Result};

use std::sync::Arc;

use super::vulkan::VulkanBackend;

impl VulkanBackend {
    /// Lazily construct (and cache) the resident-decode buffer ring.
    ///
    /// Returns `Some(&pool)` when the ring fits within 1% of the
    /// device-local heap and every slot allocation succeeds.
    /// Returns `None` (after a one-time `tracing::warn!`) when the
    /// device cannot fit the minimum three slots. The outcome is cached so the
    /// per-call kt `kiln_tensor::Tensor` fallback does not re-probe on every
    /// decode step.
    pub fn decode_resident_pool(
        &self,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Option<&Arc<kiln_vulkan_kernel::DecodeResidentPool>> {
        let dev = self.vulkan_device.as_ref()?;
        self.decode_resident_pool
            .get_or_init(|| {
                match kiln_vulkan_kernel::DecodeResidentPool::try_new(
                    dev,
                    max_hidden,
                    max_intermediate,
                    max_batch,
                ) {
                    Ok(Some(pool)) => Some(Arc::new(pool)),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Vulkan-resident decode pool construction errored; \
                             falling back to per-call kt Tensor path"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Lazily construct (and cache) the Vulkan-resident paged KV cache
    /// for the given geometry.
    ///
    /// `num_full_attn_layers`, `num_blocks`, `block_size`, `num_kv_heads`,
    /// `head_dim` mirror the legacy `PagedKvCache::new` geometry. The
    /// resident cache is a device-local sibling laid out element-for-element
    /// compatible with the existing paged-attn shaders.
    ///
    /// Returns `Some(&cache)` when the device allocation succeeds. Returns
    /// `None` (with a one-time `tracing::warn!`) when the device can't fit
    /// the geometry; callers fall back to the legacy CPU-backed pool.
    /// The `None` outcome is cached on the backend so subsequent calls
    /// don't re-probe.
    pub fn vk_paged_kv_cache(
        &self,
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Option<&Arc<kiln_vulkan_kernel::VkPagedKvCache>> {
        let dev = self.vulkan_device.as_ref()?;
        self.vk_paged_kv_cache
            .get_or_init(|| {
                match kiln_vulkan_kernel::VkPagedKvCache::try_new(
                    dev,
                    num_full_attn_layers,
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_dim,
                ) {
                    Ok(Some(cache)) => Some(Arc::new(cache)),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Vulkan-resident paged KV cache construction errored; \
                             falling back to legacy CPU-backed pool"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Acquire (or lazily create) a persistent
    /// [`VulkanBuffer`](kiln_vulkan_kernel::VulkanBuffer) under the given role
    /// key, sized to at least `min_bytes`. The same buffer is returned on every
    /// subsequent call with the same role, so the resident decode block helpers
    /// pay zero allocation cost on the steady-state hot path.
    ///
    /// If a previously-cached buffer for the role is too small for
    /// the new `min_bytes` it is replaced.
    pub fn acquire_resident_scratch(
        &self,
        role: &'static str,
        min_bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .resident_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("resident scratch mutex poisoned"))?;
        if let Some(buf) = g.get(role) {
            if buf.size() >= min_bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let buf = kiln_vulkan_kernel::VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            min_bytes.max(4),
        )
        .with_context(|| format!("alloc resident scratch '{role}'"))?;
        let arc = Arc::new(buf);
        g.insert(role, Arc::clone(&arc));
        Ok(arc)
    }

    /// Host-visible variant of `acquire_resident_scratch`. Used by the
    /// native decode orchestrator to keep a persistent readback
    /// staging buffer (for logits), folding the readback's
    /// `cmd_copy_buffer` into the main `CommandBatch` so the post-
    /// submit step is just a `map_memory` rather than a fresh queue
    /// submission.
    pub fn acquire_resident_scratch_host_visible(
        &self,
        role: &'static str,
        min_bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .resident_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("resident scratch mutex poisoned"))?;
        if let Some(buf) = g.get(role) {
            if buf.size() >= min_bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let buf = kiln_vulkan_kernel::VulkanBuffer::create_host_visible(
            dev.device(),
            dev.host_visible_mem_type(),
            min_bytes.max(4),
        )
        .with_context(|| format!("alloc host-visible resident scratch '{role}'"))?;
        let arc = Arc::new(buf);
        g.insert(role, Arc::clone(&arc));
        Ok(arc)
    }
}
