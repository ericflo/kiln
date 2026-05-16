//! Vulkan-resident paged KV cache for the resident decode path.
//!
//! Gate (b)/(e) of `docs/vk_resident_decode_plan.md`: the legacy
//! [`PagedKvCache`] in `kiln-model` stores its `(k_pool, v_pool)`
//! tensors on the candle CPU device on Vulkan (there is no candle
//! Vulkan device). That means every paged-attention read on Vulkan
//! today crosses host ↔ device twice per decode step — once to
//! upload the freshly-projected K/V slot and once to upload the
//! whole window into the paged-attn kernel.
//!
//! `VkPagedKvCache` is the device-resident sibling: one device-local
//! `VulkanBuffer` per `(layer, K|V)` pair, laid out as
//! `[total_slots, num_kv_heads, head_dim]` in f32 — the same shape
//! the legacy pool uses, the same layout `paged_attn_decode_batch.comp`
//! and `paged_attn_decode_batch_paged.comp` already read from. The
//! resident decode path threads these buffers through every layer
//! without ever round-tripping K/V across the PCIe boundary.
//!
//! Lifetime model: the cache is owned by `VulkanBackend` and
//! lazy-initialized via [`Self::try_new`] on the first resident decode
//! step. Sizing is derived from the model config (number of
//! full-attention layers, paged-cache block geometry); the cache
//! holds device memory of
//! `2 × num_layers × total_slots × num_kv_heads × head_dim × 4`
//! bytes. On Qwen3.5-4B with the standard `2048 × 16` block pool and
//! 8 full-attention layers that's
//! `2 × 8 × 32768 × 4 × 256 × 4 ≈ 2 GiB`, well inside an RTX 6000 Ada
//! 48 GiB heap.
//!
//! This module owns the *storage*; per-step writes / reads of K/V
//! are dispatched through resident kernels in `resident.rs`.

use anyhow::{Context, Result};
use std::sync::Arc;

use crate::{VulkanBuffer, VulkanDevice};

/// Bytes per element of the resident pool. f32 keeps the layout
/// directly compatible with `paged_attn_decode_batch{,_paged}.comp`
/// (binding 1/2 declares `readonly buffer KBuf { float data_k[]; }`)
/// without a bf16-aware shader variant.
const BYTES_PER_ELEMENT: u64 = 4;

/// Vulkan-resident parallel paged KV cache.
///
/// Mirrors the legacy [`kiln_model::paged_kv_cache::PagedKvCache`]'s
/// `layers: Vec<(Tensor, Tensor)>` layout, but each pool is a
/// device-local [`VulkanBuffer`] of f32 elements rather than a
/// candle CPU tensor.
pub struct VkPagedKvCache {
    /// Per full-attention layer, the K pool buffer in VRAM.
    k_layers: Vec<Arc<VulkanBuffer>>,
    /// Per full-attention layer, the V pool buffer in VRAM.
    v_layers: Vec<Arc<VulkanBuffer>>,
    /// Total slot count = `num_blocks * block_size`.
    total_slots: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
}

impl std::fmt::Debug for VkPagedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkPagedKvCache")
            .field("num_layers", &self.k_layers.len())
            .field("total_slots", &self.total_slots)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("block_size", &self.block_size)
            .field("bytes_per_layer", &self.bytes_per_layer())
            .finish()
    }
}

impl VkPagedKvCache {
    /// Allocate a zero-initialized cache with the requested geometry.
    ///
    /// Layout: each `k_layers[layer]` and `v_layers[layer]` is a
    /// device-local f32 buffer of `total_slots × num_kv_heads × head_dim`
    /// floats, matching the legacy paged pool element-for-element.
    /// `block_size` is recorded so callers (resident write kernels)
    /// can resolve `(start_pos, block_table) → slot` without
    /// duplicating the legacy cache's geometry knowledge.
    pub fn new(
        device: &VulkanDevice,
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            num_full_attn_layers > 0,
            "VkPagedKvCache: num_full_attn_layers must be > 0"
        );
        anyhow::ensure!(num_blocks > 0, "VkPagedKvCache: num_blocks must be > 0");
        anyhow::ensure!(block_size > 0, "VkPagedKvCache: block_size must be > 0");
        anyhow::ensure!(num_kv_heads > 0, "VkPagedKvCache: num_kv_heads must be > 0");
        anyhow::ensure!(head_dim > 0, "VkPagedKvCache: head_dim must be > 0");

        let total_slots = num_blocks
            .checked_mul(block_size)
            .context("VkPagedKvCache: total_slots overflowed")?;
        let elems_per_layer = total_slots
            .checked_mul(num_kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .context("VkPagedKvCache: elements per layer overflowed")?;
        let bytes_per_layer = (elems_per_layer as u64)
            .checked_mul(BYTES_PER_ELEMENT)
            .context("VkPagedKvCache: bytes per layer overflowed")?;

        let mut k_layers = Vec::with_capacity(num_full_attn_layers);
        let mut v_layers = Vec::with_capacity(num_full_attn_layers);
        for layer_idx in 0..num_full_attn_layers {
            let k = VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes_per_layer,
            )
            .with_context(|| format!("VkPagedKvCache: allocate K pool for layer {layer_idx}"))?;
            let v = VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                bytes_per_layer,
            )
            .with_context(|| format!("VkPagedKvCache: allocate V pool for layer {layer_idx}"))?;
            k_layers.push(Arc::new(k));
            v_layers.push(Arc::new(v));
        }

        Ok(Self {
            k_layers,
            v_layers,
            total_slots,
            num_kv_heads,
            head_dim,
            block_size,
        })
    }

    /// Try to allocate the cache; returns `Ok(None)` if the device
    /// can't fit the geometry. Callers fall back to the legacy CPU
    /// path on `None`. Distinct from `new` so the lazy first-use
    /// init in `VulkanBackend` has a non-panicking branch.
    pub fn try_new(
        device: &VulkanDevice,
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Option<Self>> {
        match Self::new(
            device,
            num_full_attn_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        ) {
            Ok(cache) => Ok(Some(cache)),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Vulkan-resident paged KV cache allocation failed; falling back to \
                     legacy CPU-backed pool"
                );
                Ok(None)
            }
        }
    }

    pub fn num_layers(&self) -> usize {
        self.k_layers.len()
    }

    pub fn total_slots(&self) -> usize {
        self.total_slots
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Bytes per pool buffer (K or V): `total_slots × num_kv_heads × head_dim × 4`.
    pub fn bytes_per_layer(&self) -> u64 {
        (self.total_slots as u64)
            * (self.num_kv_heads as u64)
            * (self.head_dim as u64)
            * BYTES_PER_ELEMENT
    }

    /// f32 element count per slot: `num_kv_heads × head_dim`.
    pub fn elements_per_slot(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Get the K pool buffer for a layer.
    pub fn k_buffer(&self, layer_idx: usize) -> Option<&Arc<VulkanBuffer>> {
        self.k_layers.get(layer_idx)
    }

    /// Get the V pool buffer for a layer.
    pub fn v_buffer(&self, layer_idx: usize) -> Option<&Arc<VulkanBuffer>> {
        self.v_layers.get(layer_idx)
    }

    /// Seed one layer's K/V pool from caller-provided f32 buffers.
    ///
    /// `k_pool_bytes` and `v_pool_bytes` each carry
    /// `bytes_per_layer()` bytes of f32 data laid out as the legacy
    /// `[total_slots, num_kv_heads, head_dim]` pool. This is the
    /// migration step from a CPU-resident pool: copy the entire layer
    /// once, then all subsequent decode writes go through the
    /// resident slot-write kernel.
    /// Upload one block's worth of K/V slots into the resident pool at
    /// `block_idx`. `k_block_bytes` and `v_block_bytes` must each be
    /// `block_size * num_kv_heads * head_dim * 4` bytes (one block's
    /// f32 payload). Used by the slot-range-aware seed path so a fresh
    /// request only pays for the blocks its block_table references —
    /// not the multi-GB full pool slab.
    pub fn upload_layer_block_from_f32(
        &self,
        device: &VulkanDevice,
        layer_idx: usize,
        block_idx: usize,
        k_block_bytes: &[u8],
        v_block_bytes: &[u8],
    ) -> Result<()> {
        let elements_per_slot = self.elements_per_slot();
        let block_bytes = (self.block_size * elements_per_slot) as u64 * BYTES_PER_ELEMENT;
        anyhow::ensure!(
            k_block_bytes.len() as u64 == block_bytes,
            "VkPagedKvCache::upload_layer_block_from_f32: k payload is {} bytes, expected {block_bytes}",
            k_block_bytes.len()
        );
        anyhow::ensure!(
            v_block_bytes.len() as u64 == block_bytes,
            "VkPagedKvCache::upload_layer_block_from_f32: v payload is {} bytes, expected {block_bytes}",
            v_block_bytes.len()
        );
        let total_blocks = self.total_slots / self.block_size;
        anyhow::ensure!(
            block_idx < total_blocks,
            "VkPagedKvCache::upload_layer_block_from_f32: block_idx {block_idx} >= total_blocks {total_blocks}"
        );
        let k = self
            .k_layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer_idx {layer_idx} out of range"))?;
        let v = self
            .v_layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer_idx {layer_idx} out of range"))?;
        let dst_offset = (block_idx as u64) * block_bytes;
        VulkanBuffer::upload_data_at_offset(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            k,
            dst_offset,
            k_block_bytes,
        )
        .context("VkPagedKvCache: K block upload")?;
        VulkanBuffer::upload_data_at_offset(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            v,
            dst_offset,
            v_block_bytes,
        )
        .context("VkPagedKvCache: V block upload")?;
        Ok(())
    }

    pub fn upload_layer_from_f32(
        &self,
        device: &VulkanDevice,
        layer_idx: usize,
        k_pool_bytes: &[u8],
        v_pool_bytes: &[u8],
    ) -> Result<()> {
        let need = self.bytes_per_layer() as usize;
        anyhow::ensure!(
            k_pool_bytes.len() == need,
            "VkPagedKvCache::upload_layer_from_f32: k payload is {} bytes, expected {need}",
            k_pool_bytes.len()
        );
        anyhow::ensure!(
            v_pool_bytes.len() == need,
            "VkPagedKvCache::upload_layer_from_f32: v payload is {} bytes, expected {need}",
            v_pool_bytes.len()
        );
        let k = self
            .k_layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer_idx {layer_idx} out of range"))?;
        let v = self
            .v_layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("layer_idx {layer_idx} out of range"))?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            k,
            k_pool_bytes,
        )
        .context("VkPagedKvCache: K layer upload")?;
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            v,
            v_pool_bytes,
        )
        .context("VkPagedKvCache: V layer upload")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vk_paged_kv_cache_constructs_when_device_up() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        let arc = Arc::new(dev);
        // Use a tiny config — the test just verifies allocation succeeds
        // and the geometry is recorded correctly. The real Qwen3.5-4B
        // sizing is exercised end-to-end by the parity test.
        let num_layers = 4;
        let num_blocks = 8;
        let block_size = 16;
        let num_kv_heads = 4;
        let head_dim = 64;
        let cache = VkPagedKvCache::new(
            &arc,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )
        .expect("vk paged kv cache should allocate when device is up");
        assert_eq!(cache.num_layers(), num_layers);
        assert_eq!(cache.total_slots(), num_blocks * block_size);
        assert_eq!(cache.num_kv_heads(), num_kv_heads);
        assert_eq!(cache.head_dim(), head_dim);
        assert_eq!(cache.block_size(), block_size);
        let expected_bytes =
            (num_blocks * block_size * num_kv_heads * head_dim) as u64 * BYTES_PER_ELEMENT;
        assert_eq!(cache.bytes_per_layer(), expected_bytes);
        assert_eq!(cache.elements_per_slot(), num_kv_heads * head_dim);
        for layer_idx in 0..num_layers {
            assert!(cache.k_buffer(layer_idx).is_some());
            assert!(cache.v_buffer(layer_idx).is_some());
            assert_eq!(cache.k_buffer(layer_idx).unwrap().size(), expected_bytes);
            assert_eq!(cache.v_buffer(layer_idx).unwrap().size(), expected_bytes);
        }
        assert!(cache.k_buffer(num_layers).is_none());
    }

    #[test]
    fn vk_paged_kv_cache_try_new_returns_some_on_normal_device() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        let arc = Arc::new(dev);
        let cache = VkPagedKvCache::try_new(&arc, 2, 4, 16, 2, 32)
            .expect("try_new should not propagate the underlying error");
        assert!(cache.is_some());
    }

    #[test]
    fn vk_paged_kv_cache_rejects_zero_dimensions() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        let arc = Arc::new(dev);
        assert!(VkPagedKvCache::new(&arc, 0, 4, 16, 2, 32).is_err());
        assert!(VkPagedKvCache::new(&arc, 2, 0, 16, 2, 32).is_err());
        assert!(VkPagedKvCache::new(&arc, 2, 4, 0, 2, 32).is_err());
        assert!(VkPagedKvCache::new(&arc, 2, 4, 16, 0, 32).is_err());
        assert!(VkPagedKvCache::new(&arc, 2, 4, 16, 2, 0).is_err());
    }
}
