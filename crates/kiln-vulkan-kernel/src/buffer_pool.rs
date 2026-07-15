//! Recycling buffer pool to eliminate per-dispatch allocator churn.
//!
//! In the vk-native training step we issue ~5K dispatches per step,
//! each typically allocating a fresh device-local buffer for its
//! output. With ~10K total alloc/free per step, the kernel-level DRM
//! allocator on Strix Halo dominates step time. This pool short-
//! circuits that:
//!
//!   - `pool_alloc(device, bytes)` returns an `Arc<VulkanBuffer>`.
//!     The pool holds an internal `Arc` on admitted buffers, so when the
//!     caller drops their `Arc` the buffer stays available for reuse
//!     (refcount = 1, only-pool).
//!   - The next `pool_alloc(device, bytes)` for the same byte-size
//!     scans for an entry with `Arc::strong_count == 1` (only the
//!     pool holds it = "free") and returns a clone. No vkAllocateMemory.
//!   - Cache hit cost: lock contention + linear scan of the size
//!     bucket. Both are tiny vs. vkAllocateMemory + vkBindBufferMemory.
//!
//! Memory: retention is capped process-wide. Admission evicts the
//! oldest idle entries until the new buffer fits; a buffer that cannot
//! fit remains caller-owned and is freed normally. Active borrowers are
//! never evicted, and pressure reclaim can release idle entries.
//!
//! Key choice: bytes (not element count). All callers go through this
//! function, so size buckets are stable.

use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Default process-wide retained scratch budget. Active work may allocate
/// beyond it, but excess buffers are freed when their final caller drops them.
pub const DEFAULT_MAX_RETAINED_BYTES: u64 = 3 * 1024 * 1024 * 1024;

static MAX_RETAINED_BYTES: AtomicU64 = AtomicU64::new(DEFAULT_MAX_RETAINED_BYTES);

#[derive(Debug)]
struct PooledBuffer {
    buffer: Arc<VulkanBuffer>,
    last_used: u64,
}

#[derive(Default)]
struct PoolInner {
    /// Per-(device, byte-size) FIFO of recycled buffers. Keying by the
    /// raw device handle ensures a buffer allocated for one logical
    /// VulkanDevice is never reused on another (e.g., across tests
    /// that each construct their own device).
    by_device_bytes: HashMap<(vk::Device, u64), Vec<PooledBuffer>>,
    /// Per-(device, host-mem-type, byte-size) FIFO of recycled
    /// *host-visible* staging buffers. Kept separate from
    /// `by_device_bytes` because these carry different memory
    /// properties (HOST_VISIBLE) and must never be handed to a
    /// device-local request (or vice versa). Recycling these kills the
    /// per-`read_back` `vkAllocateMemory` (`amdgpu_bo_alloc` kernel
    /// ioctl, tens-to-hundreds of MB each) that dominated the Vulkan
    /// decode-weight prewarm — hundreds of weights × one fresh staging
    /// BO apiece.
    host_by_device_bytes: HashMap<(vk::Device, u32, u64), Vec<PooledBuffer>>,
    /// Last 256 MiB high-water bucket emitted for each logical device.
    reported_high_water_buckets: HashMap<vk::Device, u64>,
    use_clock: u64,
    cache_hits: u64,
    cache_misses: u64,
    eviction_count: u64,
    evicted_bytes: u64,
    uncached_allocation_count: u64,
    uncached_allocated_bytes: u64,
}

static POOL: OnceLock<Mutex<PoolInner>> = OnceLock::new();

fn pool() -> &'static Mutex<PoolInner> {
    POOL.get_or_init(|| Mutex::new(PoolInner::default()))
}

impl PoolInner {
    fn next_use(&mut self) -> u64 {
        self.use_clock = self.use_clock.wrapping_add(1).max(1);
        self.use_clock
    }
}

/// Allocate (or recycle) a device-local buffer of `bytes` bytes.
///
/// The returned `Arc<VulkanBuffer>` shares storage with the internal
/// pool; when the caller's last clone drops, an admitted buffer is
/// recycled. Buffers beyond the configured cap are freed normally.
pub fn pool_alloc_device_local(device: &VulkanDevice, bytes: u64) -> Result<Arc<VulkanBuffer>> {
    // Round to the next power-of-two-ish bucket to reduce fragmentation.
    // Buckets: round up to multiples of 64 KB at small sizes, larger
    // multiples for big buffers. Empirically the GDN training step has
    // ~30 distinct sizes at typical T=918 — small enough that we don't
    // need exact-match buckets.
    let bucket = bucket_for(bytes);
    let dev_handle = device.device().handle();
    let key = (dev_handle, bucket);
    let mut inner = pool().lock().unwrap();
    let last_used = inner.next_use();
    if let Some(slots) = inner.by_device_bytes.get_mut(&key) {
        for slot in slots.iter_mut() {
            if Arc::strong_count(&slot.buffer) == 1 {
                slot.last_used = last_used;
                let buffer = Arc::clone(&slot.buffer);
                inner.cache_hits = inner.cache_hits.saturating_add(1);
                return Ok(buffer);
            }
        }
    }
    inner.cache_misses = inner.cache_misses.saturating_add(1);
    drop(inner); // release lock before vkAllocateMemory

    let buf =
        VulkanBuffer::create_device_local(device.device(), device.device_local_mem_type(), bucket)
            .context("pool_alloc_device_local: vkAllocateMemory")?;
    let arc = Arc::new(buf);

    let mut inner = pool().lock().unwrap();
    let evicted = retain_new_buffer(
        &mut inner,
        PoolKey::Device(key),
        Arc::clone(&arc),
        MAX_RETAINED_BYTES.load(Ordering::Relaxed),
    );
    if Arc::strong_count(&arc) == 1 {
        inner.uncached_allocation_count = inner.uncached_allocation_count.saturating_add(1);
        inner.uncached_allocated_bytes = inner
            .uncached_allocated_bytes
            .saturating_add(arc.allocation_size());
    } else {
        maybe_report_pool_growth(&mut inner, dev_handle);
    }
    drop(inner);
    drop(evicted);
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
    let last_used = inner.next_use();
    if let Some(slots) = inner.host_by_device_bytes.get_mut(&key) {
        for slot in slots.iter_mut() {
            // Each caller gets a DISTINCT Arc (a free slot here, or a fresh
            // alloc below), so concurrent `read_back`s never map/unmap the
            // same VkDeviceMemory at once. The host memory type is
            // HOST_COHERENT (see `VulkanDevice` mem-type selection), so the
            // mapped read after `queue_wait_idle` needs no explicit
            // vkInvalidateMappedMemoryRanges.
            if Arc::strong_count(&slot.buffer) == 1 {
                debug_assert!(
                    slot.buffer.size() >= bytes,
                    "pooled host buffer {} < requested {bytes}",
                    slot.buffer.size()
                );
                slot.last_used = last_used;
                let buffer = Arc::clone(&slot.buffer);
                inner.cache_hits = inner.cache_hits.saturating_add(1);
                return Ok(buffer);
            }
        }
    }
    inner.cache_misses = inner.cache_misses.saturating_add(1);
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
    let evicted = retain_new_buffer(
        &mut inner,
        PoolKey::Host(key),
        Arc::clone(&arc),
        MAX_RETAINED_BYTES.load(Ordering::Relaxed),
    );
    if Arc::strong_count(&arc) == 1 {
        inner.uncached_allocation_count = inner.uncached_allocation_count.saturating_add(1);
        inner.uncached_allocated_bytes = inner
            .uncached_allocated_bytes
            .saturating_add(arc.allocation_size());
    } else {
        maybe_report_pool_growth(&mut inner, dev_handle);
    }
    drop(inner);
    drop(evicted);
    Ok(arc)
}

const POOL_HIGH_WATER_REPORT_STEP_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    pub max_retained_bytes: u64,
    pub bucket_count: usize,
    pub buffer_count: usize,
    pub total_bytes: u64,
    pub free_buffer_count: usize,
    pub free_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub eviction_count: u64,
    pub evicted_bytes: u64,
    pub uncached_allocation_count: u64,
    pub uncached_allocated_bytes: u64,
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
    let mut stats = BufferPoolStats {
        max_retained_bytes: MAX_RETAINED_BYTES.load(Ordering::Relaxed),
        cache_hits: inner.cache_hits,
        cache_misses: inner.cache_misses,
        eviction_count: inner.eviction_count,
        evicted_bytes: inner.evicted_bytes,
        uncached_allocation_count: inner.uncached_allocation_count,
        uncached_allocated_bytes: inner.uncached_allocated_bytes,
        ..BufferPoolStats::default()
    };
    for ((entry_device, _size), slots) in &inner.by_device_bytes {
        if *entry_device != device {
            continue;
        }
        stats.bucket_count += 1;
        stats.buffer_count += slots.len();
        for slot in slots {
            let allocation_size = slot.buffer.allocation_size();
            stats.total_bytes = stats.total_bytes.saturating_add(allocation_size);
            if Arc::strong_count(&slot.buffer) == 1 {
                stats.free_buffer_count += 1;
                stats.free_bytes = stats.free_bytes.saturating_add(allocation_size);
            }
        }
    }
    for ((entry_device, _memory_type, _size), slots) in &inner.host_by_device_bytes {
        if *entry_device != device {
            continue;
        }
        stats.bucket_count += 1;
        stats.buffer_count += slots.len();
        for slot in slots {
            let allocation_size = slot.buffer.allocation_size();
            stats.total_bytes = stats.total_bytes.saturating_add(allocation_size);
            if Arc::strong_count(&slot.buffer) == 1 {
                stats.free_buffer_count += 1;
                stats.free_bytes = stats.free_bytes.saturating_add(allocation_size);
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

#[derive(Debug, Clone, Copy)]
enum PoolKey {
    Device((vk::Device, u64)),
    Host((vk::Device, u32, u64)),
}

#[derive(Debug, Clone, Copy)]
struct EvictionCandidate {
    key: PoolKey,
    index: usize,
    last_used: u64,
}

fn oldest_free_buffer(inner: &PoolInner) -> Option<EvictionCandidate> {
    let mut oldest: Option<EvictionCandidate> = None;
    for (key, slots) in &inner.by_device_bytes {
        for (index, slot) in slots.iter().enumerate() {
            if Arc::strong_count(&slot.buffer) != 1 {
                continue;
            }
            let candidate = EvictionCandidate {
                key: PoolKey::Device(*key),
                index,
                last_used: slot.last_used,
            };
            if oldest.is_none_or(|current| candidate.last_used < current.last_used) {
                oldest = Some(candidate);
            }
        }
    }
    for (key, slots) in &inner.host_by_device_bytes {
        for (index, slot) in slots.iter().enumerate() {
            if Arc::strong_count(&slot.buffer) != 1 {
                continue;
            }
            let candidate = EvictionCandidate {
                key: PoolKey::Host(*key),
                index,
                last_used: slot.last_used,
            };
            if oldest.is_none_or(|current| candidate.last_used < current.last_used) {
                oldest = Some(candidate);
            }
        }
    }
    oldest
}

fn remove_candidate(inner: &mut PoolInner, candidate: EvictionCandidate) -> Arc<VulkanBuffer> {
    match candidate.key {
        PoolKey::Device(key) => {
            let (buffer, empty) = {
                let slots = inner
                    .by_device_bytes
                    .get_mut(&key)
                    .expect("device eviction key disappeared");
                let buffer = slots.swap_remove(candidate.index).buffer;
                (buffer, slots.is_empty())
            };
            if empty {
                inner.by_device_bytes.remove(&key);
            }
            buffer
        }
        PoolKey::Host(key) => {
            let (buffer, empty) = {
                let slots = inner
                    .host_by_device_bytes
                    .get_mut(&key)
                    .expect("host eviction key disappeared");
                let buffer = slots.swap_remove(candidate.index).buffer;
                (buffer, slots.is_empty())
            };
            if empty {
                inner.host_by_device_bytes.remove(&key);
            }
            buffer
        }
    }
}

fn retained_bytes(inner: &PoolInner) -> u64 {
    inner
        .by_device_bytes
        .values()
        .chain(inner.host_by_device_bytes.values())
        .flat_map(|slots| slots.iter())
        .fold(0u64, |total, slot| {
            total.saturating_add(slot.buffer.allocation_size())
        })
}

fn evict_free_until(
    inner: &mut PoolInner,
    mut should_continue: impl FnMut(u64) -> bool,
) -> Vec<Arc<VulkanBuffer>> {
    let mut removed = Vec::new();
    let mut freed = 0u64;
    while should_continue(freed) {
        let Some(candidate) = oldest_free_buffer(inner) else {
            break;
        };
        let buffer = remove_candidate(inner, candidate);
        let bytes = buffer.allocation_size();
        freed = freed.saturating_add(bytes);
        inner.eviction_count = inner.eviction_count.saturating_add(1);
        inner.evicted_bytes = inner.evicted_bytes.saturating_add(bytes);
        removed.push(buffer);
    }
    removed
}

fn retain_new_buffer(
    inner: &mut PoolInner,
    key: PoolKey,
    buffer: Arc<VulkanBuffer>,
    limit: u64,
) -> Vec<Arc<VulkanBuffer>> {
    let allocation_size = buffer.allocation_size();
    if allocation_size > limit {
        return Vec::new();
    }

    let current = retained_bytes(inner);
    let free = free_retained_bytes(inner);
    if current.saturating_sub(free).saturating_add(allocation_size) > limit {
        return Vec::new();
    }
    let required = current
        .saturating_add(allocation_size)
        .saturating_sub(limit);
    let evicted = evict_free_until(inner, |freed| freed < required);
    if retained_bytes(inner).saturating_add(allocation_size) <= limit {
        let last_used = inner.next_use();
        let slot = PooledBuffer { buffer, last_used };
        match key {
            PoolKey::Device(key) => inner.by_device_bytes.entry(key).or_default().push(slot),
            PoolKey::Host(key) => inner
                .host_by_device_bytes
                .entry(key)
                .or_default()
                .push(slot),
        }
    }
    evicted
}

fn free_retained_bytes(inner: &PoolInner) -> u64 {
    inner
        .by_device_bytes
        .values()
        .chain(inner.host_by_device_bytes.values())
        .flat_map(|slots| slots.iter())
        .filter(|slot| Arc::strong_count(&slot.buffer) == 1)
        .fold(0u64, |total, slot| {
            total.saturating_add(slot.buffer.allocation_size())
        })
}

/// Convenience wrapper: allocate `n` F32 elements (`n * 4` bytes).
pub fn pool_alloc_f32(device: &VulkanDevice, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n as u64).saturating_mul(4).max(4);
    pool_alloc_device_local(device, bytes)
}

/// Process-wide recycler state across device-local and host-visible buckets.
pub fn pool_stats() -> BufferPoolStats {
    let inner = pool().lock().unwrap();
    let mut stats = BufferPoolStats {
        max_retained_bytes: MAX_RETAINED_BYTES.load(Ordering::Relaxed),
        bucket_count: inner.by_device_bytes.len() + inner.host_by_device_bytes.len(),
        cache_hits: inner.cache_hits,
        cache_misses: inner.cache_misses,
        eviction_count: inner.eviction_count,
        evicted_bytes: inner.evicted_bytes,
        uncached_allocation_count: inner.uncached_allocation_count,
        uncached_allocated_bytes: inner.uncached_allocated_bytes,
        ..BufferPoolStats::default()
    };
    for slot in inner
        .by_device_bytes
        .values()
        .chain(inner.host_by_device_bytes.values())
        .flat_map(|slots| slots.iter())
    {
        let bytes = slot.buffer.allocation_size();
        stats.buffer_count += 1;
        stats.total_bytes = stats.total_bytes.saturating_add(bytes);
        if Arc::strong_count(&slot.buffer) == 1 {
            stats.free_buffer_count += 1;
            stats.free_bytes = stats.free_bytes.saturating_add(bytes);
        }
    }
    stats
}

/// Return recycling-pool ownership telemetry for one logical Vulkan device.
pub fn pool_stats_for_device(device: vk::Device) -> BufferPoolStats {
    stats_for_device(&pool().lock().unwrap(), device)
}

/// Install the retained-byte budget used by subsequent pool allocations.
/// Any currently idle excess is released immediately.
pub fn pool_configure_max_retained_bytes(max_retained_bytes: u64) -> u64 {
    MAX_RETAINED_BYTES.store(max_retained_bytes, Ordering::Relaxed);
    let mut inner = pool().lock().unwrap();
    let target = retained_bytes(&inner).saturating_sub(max_retained_bytes);
    let evicted = evict_free_until(&mut inner, |freed| freed < target);
    let freed = evicted.iter().fold(0u64, |total, buffer| {
        total.saturating_add(buffer.allocation_size())
    });
    drop(inner);
    drop(evicted);
    freed
}

/// Release at least `target_bytes` of idle cached buffers when available,
/// oldest first. Borrowed buffers remain owned by their callers.
pub fn pool_trim_free(target_bytes: u64) -> u64 {
    if target_bytes == 0 {
        return 0;
    }
    let mut inner = pool().lock().unwrap();
    let evicted = evict_free_until(&mut inner, |freed| freed < target_bytes);
    let freed = evicted.iter().fold(0u64, |total, buffer| {
        total.saturating_add(buffer.allocation_size())
    });
    drop(inner);
    drop(evicted);
    freed
}

/// Drop all pooled buffers. Calls vkFreeMemory on each via the
/// `VulkanBuffer::Drop`. Use at end of training to release VRAM.
pub fn pool_drain() {
    let mut inner = pool().lock().unwrap();
    let device = std::mem::take(&mut inner.by_device_bytes);
    let host = std::mem::take(&mut inner.host_by_device_bytes);
    inner.reported_high_water_buckets.clear();
    drop(inner);
    drop(device);
    drop(host);
}

/// Drop all pooled buffers belonging to a specific Vulkan device.
/// Called from `VulkanDevice::Drop` to release the buffers BEFORE
/// the device's descriptor/command pools tear down — otherwise the
/// pool would hold dangling Arc<VulkanBuffer> entries whose
/// underlying buffers are still valid but whose owning device has
/// no live descriptor pool to bind them to.
pub fn pool_drop_for_device(dev: vk::Device) {
    let mut inner = pool().lock().unwrap();
    let device_keys = inner
        .by_device_bytes
        .keys()
        .filter(|(device, _)| *device == dev)
        .copied()
        .collect::<Vec<_>>();
    let host_keys = inner
        .host_by_device_bytes
        .keys()
        .filter(|(device, _, _)| *device == dev)
        .copied()
        .collect::<Vec<_>>();
    let device_buffers = device_keys
        .into_iter()
        .filter_map(|key| inner.by_device_bytes.remove(&key))
        .collect::<Vec<_>>();
    let host_buffers = host_keys
        .into_iter()
        .filter_map(|key| inner.host_by_device_bytes.remove(&key))
        .collect::<Vec<_>>();
    inner.reported_high_water_buckets.remove(&dev);
    drop(inner);
    drop(device_buffers);
    drop(host_buffers);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detailed_stats_derive_borrowed_ownership_without_underflow() {
        let stats = BufferPoolStats {
            max_retained_bytes: 2048,
            bucket_count: 3,
            buffer_count: 8,
            total_bytes: 1024,
            free_buffer_count: 5,
            free_bytes: 640,
            ..BufferPoolStats::default()
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

    #[test]
    fn bounded_retention_evicts_only_idle_buffers() -> Result<()> {
        let device = Arc::new(VulkanDevice::new()?);
        let handle = device.device().handle();
        let mut inner = PoolInner::default();

        let first = Arc::new(VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            64 * 1024,
        )?);
        let second = Arc::new(VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            128 * 1024,
        )?);
        let limit = second.allocation_size();
        assert!(first.allocation_size() <= limit);

        let evicted = retain_new_buffer(
            &mut inner,
            PoolKey::Device((handle, 64 * 1024)),
            Arc::clone(&first),
            limit,
        );
        assert!(evicted.is_empty());
        drop(first);

        let evicted = retain_new_buffer(
            &mut inner,
            PoolKey::Device((handle, 128 * 1024)),
            Arc::clone(&second),
            limit,
        );
        assert_eq!(evicted.len(), 1);
        assert!(retained_bytes(&inner) <= limit);
        drop(evicted);

        let borrowed_trim = evict_free_until(&mut inner, |freed| freed < limit);
        assert!(borrowed_trim.is_empty());
        assert_eq!(retained_bytes(&inner), second.allocation_size());

        let overflow = Arc::new(VulkanBuffer::create_device_local(
            device.device(),
            device.device_local_mem_type(),
            64 * 1024,
        )?);
        let rejected = retain_new_buffer(
            &mut inner,
            PoolKey::Device((handle, 64 * 1024)),
            Arc::clone(&overflow),
            limit,
        );
        assert!(rejected.is_empty());
        assert_eq!(retained_bytes(&inner), second.allocation_size());
        drop(overflow);

        drop(second);
        let idle_trim = evict_free_until(&mut inner, |freed| freed < limit);
        assert_eq!(idle_trim.len(), 1);
        assert_eq!(retained_bytes(&inner), 0);
        drop(idle_trim);
        drop(inner);
        drop(device);
        Ok(())
    }
}
