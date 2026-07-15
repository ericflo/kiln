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
    /// Per-(device, host-mem-type, byte-size) FIFO of recycled
    /// *host-visible* staging buffers. Kept separate from
    /// `by_device_bytes` because these carry different memory
    /// properties (HOST_VISIBLE) and must never be handed to a
    /// device-local request (or vice versa). Recycling these kills the
    /// per-`read_back` `vkAllocateMemory` (`amdgpu_bo_alloc` kernel
    /// ioctl, tens-to-hundreds of MB each) that dominated the Vulkan
    /// decode-weight prewarm — hundreds of weights × one fresh staging
    /// BO apiece.
    host_by_device_bytes: HashMap<(vk::Device, u32, u64), Vec<Arc<VulkanBuffer>>>,
    /// Last 256 MiB high-water bucket emitted for each logical device.
    reported_high_water_buckets: HashMap<vk::Device, u64>,
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
    maybe_report_pool_growth(&mut inner, dev_handle);
    Ok(arc)
}

/// Allocate (or recycle) a **host-visible** staging buffer of `bytes`
/// bytes for D2H read-back (and small H2D staging).
///
/// Mirrors [`pool_alloc_device_local`] but keys on the host memory
/// type as well, and takes the raw `ash::Device` + memory-type index
/// directly (what `VulkanBuffer::read_back` already has in hand) so it
/// needs no `VulkanDevice`. The returned `Arc<VulkanBuffer>` is
/// recycled when the caller's last clone drops — so a `read_back` that
/// grabs one, maps/copies, and returns hands it straight back to the
/// pool for the next call. The returned buffer may be *larger* than
/// `bytes` (bucket-rounded); callers must copy/read only the bytes
/// they need (`read_back` already does — it copies `src.size` and maps
/// `WHOLE_SIZE` but reads only `src.size`).
pub fn pool_alloc_host_visible(
    device: &Arc<ash::Device>,
    host_mem_type: u32,
    bytes: u64,
) -> Result<Arc<VulkanBuffer>> {
    // `bucket_for` always rounds UP (and floors at 4 bytes), so the buffer
    // we hand back is always >= `bytes`. Callers (`read_back`) rely on this:
    // they copy/read only the logical bytes, so a bucket-larger buffer is
    // fine, but a SMALLER one would truncate the D2H copy. The debug_asserts
    // below pin that invariant against any future `bucket_for` change.
    let bucket = bucket_for(bytes);
    let dev_handle = device.handle();
    let key = (dev_handle, host_mem_type, bucket);
    let mut inner = pool().lock().unwrap();
    if let Some(slots) = inner.host_by_device_bytes.get_mut(&key) {
        for slot in slots.iter() {
            // Each caller gets a DISTINCT Arc (a free slot here, or a fresh
            // alloc below), so concurrent `read_back`s never map/unmap the
            // same VkDeviceMemory at once. The host memory type is
            // HOST_COHERENT (see `VulkanDevice` mem-type selection), so the
            // mapped read after `queue_wait_idle` needs no explicit
            // vkInvalidateMappedMemoryRanges.
            if Arc::strong_count(slot) == 1 {
                debug_assert!(
                    slot.size() >= bytes,
                    "pooled host buffer {} < requested {bytes}",
                    slot.size()
                );
                return Ok(Arc::clone(slot));
            }
        }
    }
    drop(inner); // release lock before vkAllocateMemory

    let buf = VulkanBuffer::create_host_visible(device, host_mem_type, bucket)
        .context("pool_alloc_host_visible: vkAllocateMemory")?;
    debug_assert!(
        buf.size() >= bytes,
        "fresh host buffer {} < requested {bytes}",
        buf.size()
    );
    let arc = Arc::new(buf);

    let mut inner = pool().lock().unwrap();
    inner
        .host_by_device_bytes
        .entry(key)
        .or_default()
        .push(Arc::clone(&arc));
    maybe_report_pool_growth(&mut inner, dev_handle);
    Ok(arc)
}

const POOL_HIGH_WATER_REPORT_STEP_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    pub bucket_count: usize,
    pub buffer_count: usize,
    pub total_bytes: u64,
    pub free_buffer_count: usize,
    pub free_bytes: u64,
}

impl BufferPoolStats {
    pub fn borrowed_buffer_count(self) -> usize {
        self.buffer_count.saturating_sub(self.free_buffer_count)
    }

    pub fn borrowed_bytes(self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
}

fn stats_for_device(inner: &PoolInner, device: vk::Device) -> BufferPoolStats {
    let mut stats = BufferPoolStats::default();
    for ((entry_device, size), slots) in &inner.by_device_bytes {
        if *entry_device != device {
            continue;
        }
        stats.bucket_count += 1;
        stats.buffer_count += slots.len();
        stats.total_bytes = stats
            .total_bytes
            .saturating_add(size.saturating_mul(slots.len() as u64));
        for slot in slots {
            if Arc::strong_count(slot) == 1 {
                stats.free_buffer_count += 1;
                stats.free_bytes = stats.free_bytes.saturating_add(*size);
            }
        }
    }
    for ((entry_device, _memory_type, size), slots) in &inner.host_by_device_bytes {
        if *entry_device != device {
            continue;
        }
        stats.bucket_count += 1;
        stats.buffer_count += slots.len();
        stats.total_bytes = stats
            .total_bytes
            .saturating_add(size.saturating_mul(slots.len() as u64));
        for slot in slots {
            if Arc::strong_count(slot) == 1 {
                stats.free_buffer_count += 1;
                stats.free_bytes = stats.free_bytes.saturating_add(*size);
            }
        }
    }
    stats
}

fn maybe_report_pool_growth(inner: &mut PoolInner, device: vk::Device) {
    let stats = stats_for_device(inner, device);
    let high_water_bucket = stats.total_bytes / POOL_HIGH_WATER_REPORT_STEP_BYTES;
    let reported = inner.reported_high_water_buckets.entry(device).or_default();
    if high_water_bucket == 0 || high_water_bucket <= *reported {
        return;
    }
    *reported = high_water_bucket;
    tracing::info!(
        event = "vulkan_buffer_pool_high_water",
        bucket_count = stats.bucket_count,
        buffer_count = stats.buffer_count,
        total_bytes = stats.total_bytes,
        free_buffer_count = stats.free_buffer_count,
        free_bytes = stats.free_bytes,
        borrowed_buffer_count = stats.borrowed_buffer_count(),
        borrowed_bytes = stats.borrowed_bytes(),
        "Vulkan recycling buffer pool reached a new high-water mark"
    );
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
/// Counts both the device-local and host-visible staging pools.
pub fn pool_stats() -> (usize, usize, u64) {
    let inner = pool().lock().unwrap();
    let mut total_bufs = 0usize;
    let mut total_bytes = 0u64;
    for ((_dev, size), slots) in inner.by_device_bytes.iter() {
        total_bufs += slots.len();
        total_bytes += size * (slots.len() as u64);
    }
    for ((_dev, _mt, size), slots) in inner.host_by_device_bytes.iter() {
        total_bufs += slots.len();
        total_bytes += size * (slots.len() as u64);
    }
    let buckets = inner.by_device_bytes.len() + inner.host_by_device_bytes.len();
    (buckets, total_bufs, total_bytes)
}

/// Return recycling-pool ownership telemetry for one logical Vulkan device.
pub fn pool_stats_for_device(device: vk::Device) -> BufferPoolStats {
    stats_for_device(&pool().lock().unwrap(), device)
}

/// Drop all pooled buffers. Calls vkFreeMemory on each via the
/// `VulkanBuffer::Drop`. Use at end of training to release VRAM.
pub fn pool_drain() {
    let mut inner = pool().lock().unwrap();
    inner.by_device_bytes.clear();
    inner.host_by_device_bytes.clear();
    inner.reported_high_water_buckets.clear();
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
    inner.host_by_device_bytes.retain(|(d, _, _), _| *d != dev);
    inner.reported_high_water_buckets.remove(&dev);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detailed_stats_derive_borrowed_ownership_without_underflow() {
        let stats = BufferPoolStats {
            bucket_count: 3,
            buffer_count: 8,
            total_bytes: 1024,
            free_buffer_count: 5,
            free_bytes: 640,
        };
        assert_eq!(stats.borrowed_buffer_count(), 3);
        assert_eq!(stats.borrowed_bytes(), 384);

        let inconsistent = BufferPoolStats {
            free_buffer_count: 2,
            free_bytes: 2,
            ..BufferPoolStats::default()
        };
        assert_eq!(inconsistent.borrowed_buffer_count(), 0);
        assert_eq!(inconsistent.borrowed_bytes(), 0);
    }

    #[test]
    fn allocation_buckets_never_round_down() {
        for bytes in [0, 1, 65_536, 65_537, 4 * 1024 * 1024, 4 * 1024 * 1024 + 1] {
            assert!(bucket_for(bytes) >= bytes.max(4));
        }
    }
}
