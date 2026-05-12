//! Recycling buffer pool to eliminate per-dispatch allocator churn.
//!
//! In the vk-native training step we issue ~5K dispatches per step,
//! each typically allocating a fresh device-local buffer for its
//! output. With ~10K total alloc/free per step, the kernel-level DRM
//! allocator on Strix Halo dominates step time. This pool short-
//! circuits that:
//!
//!   - `pool_alloc(device, bytes)` returns an `Arc<VulkanBuffer>`.
//!     The pool holds an internal `Arc` on every buffer it has ever
//!     handed out, so when the caller drops their `Arc` the buffer
//!     stays alive (refcount = 1, only-pool).
//!   - The next `pool_alloc(device, bytes)` for the same byte-size
//!     scans for an entry with `Arc::strong_count == 1` (only the
//!     pool holds it = "free") and returns a clone. No vkAllocateMemory.
//!   - Cache hit cost: lock contention + linear scan of the size
//!     bucket. Both are tiny vs. vkAllocateMemory + vkBindBufferMemory.
//!
//! Memory: peak grows to the size of the working set across one
//! training step. For the SFT smoke test that's ~3 GB of recycled
//! scratch buffers — a worthwhile tradeoff for the throughput win.
//!
//! Key choice: bytes (not element count). All callers go through this
//! function, so size buckets are stable.

use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Default)]
struct PoolInner {
    /// Per-(device, byte-size) FIFO of recycled buffers. Keying by the
    /// raw device handle ensures a buffer allocated for one logical
    /// VulkanDevice is never reused on another (e.g., across tests
    /// that each construct their own device).
    by_device_bytes: HashMap<(vk::Device, u64), Vec<Arc<VulkanBuffer>>>,
}

static POOL: OnceLock<Mutex<PoolInner>> = OnceLock::new();

fn pool() -> &'static Mutex<PoolInner> {
    POOL.get_or_init(|| Mutex::new(PoolInner::default()))
}

/// Allocate (or recycle) a device-local buffer of `bytes` bytes.
///
/// The returned `Arc<VulkanBuffer>` shares storage with the internal
/// pool; when the caller's last clone drops, the buffer is recycled
/// (NOT freed). Use `pool_drain` if you need to release everything
/// at the end of training.
pub fn pool_alloc_device_local(
    device: &Arc<VulkanDevice>,
    bytes: u64,
) -> Result<Arc<VulkanBuffer>> {
    // Round to the next power-of-two-ish bucket to reduce fragmentation.
    // Buckets: round up to multiples of 64 KB at small sizes, larger
    // multiples for big buffers. Empirically the GDN training step has
    // ~30 distinct sizes at typical T=918 — small enough that we don't
    // need exact-match buckets.
    let bucket = bucket_for(bytes);
    let dev_handle = device.device().handle();
    let key = (dev_handle, bucket);
    let mut inner = pool().lock().unwrap();
    if let Some(slots) = inner.by_device_bytes.get_mut(&key) {
        for slot in slots.iter() {
            if Arc::strong_count(slot) == 1 {
                return Ok(Arc::clone(slot));
            }
        }
    }
    drop(inner); // release lock before vkAllocateMemory

    let buf =
        VulkanBuffer::create_device_local(device.device(), device.device_local_mem_type(), bucket)
            .context("pool_alloc_device_local: vkAllocateMemory")?;
    let arc = Arc::new(buf);

    let mut inner = pool().lock().unwrap();
    inner
        .by_device_bytes
        .entry(key)
        .or_default()
        .push(Arc::clone(&arc));
    Ok(arc)
}

fn bucket_for(bytes: u64) -> u64 {
    let bytes = bytes.max(4);
    if bytes <= 65_536 {
        // 64 KB granularity for small buffers
        ((bytes + 65_535) / 65_536) * 65_536
    } else if bytes <= 4 * 1024 * 1024 {
        // 256 KB granularity up to 4 MB
        ((bytes + 262_143) / 262_144) * 262_144
    } else {
        // 4 MB granularity for big buffers
        ((bytes + (4 * 1024 * 1024 - 1)) / (4 * 1024 * 1024)) * (4 * 1024 * 1024)
    }
}

/// Convenience wrapper: allocate `n` F32 elements (`n * 4` bytes).
pub fn pool_alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n as u64).saturating_mul(4).max(4);
    pool_alloc_device_local(device, bytes)
}

/// Stats for diagnostics: returns (num_buckets, total_buffers, total_bytes).
pub fn pool_stats() -> (usize, usize, u64) {
    let inner = pool().lock().unwrap();
    let mut total_bufs = 0usize;
    let mut total_bytes = 0u64;
    for ((_dev, size), slots) in inner.by_device_bytes.iter() {
        total_bufs += slots.len();
        total_bytes += size * (slots.len() as u64);
    }
    (inner.by_device_bytes.len(), total_bufs, total_bytes)
}

/// Drop all pooled buffers. Calls vkFreeMemory on each via the
/// `VulkanBuffer::Drop`. Use at end of training to release VRAM.
pub fn pool_drain() {
    let mut inner = pool().lock().unwrap();
    inner.by_device_bytes.clear();
}

/// Drop all pooled buffers belonging to a specific Vulkan device.
/// Called from `VulkanDevice::Drop` to release the buffers BEFORE
/// the device's descriptor/command pools tear down — otherwise the
/// pool would hold dangling Arc<VulkanBuffer> entries whose
/// underlying buffers are still valid but whose owning device has
/// no live descriptor pool to bind them to.
pub fn pool_drop_for_device(dev: vk::Device) {
    let mut inner = pool().lock().unwrap();
    inner.by_device_bytes.retain(|(d, _), _| *d != dev);
}
