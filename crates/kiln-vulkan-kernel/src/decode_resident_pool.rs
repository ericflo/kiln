//! Fixed-size ring of intermediate device-local buffers for the
//! Vulkan-resident decode path.
//!
//! Gate (b) of `docs/vk_resident_decode_plan.md`: a small fixed ring
//! of 3–4 reusable `VulkanBuffer`s sized to
//! `max(hidden, intermediate) × max_batch × 4` bytes — ≈ 2.3 MiB per
//! slot on Qwen3.5-4B at `max_batch = 64`, ≈ 10 MiB total. The pool
//! autosizes from `VulkanDevice::device_local_heap_bytes()` and caps
//! itself at 1 % of that. On a device that can't even fit the
//! 3-slot minimum — Strix Halo iGPU with its 16 GiB unified-memory
//! budget that the model already nearly fills, or any smaller
//! integrated GPU — `try_new` returns `Ok(None)`, the dispatcher
//! emits `tracing::warn!` once, and the call site transparently
//! falls back to the existing per-call `Tensor`-shaped path.
//!
//! This is **deliberately** distinct from the general-purpose
//! `buffer_pool` (training-side recycler keyed by byte bucket):
//!  - Training reuses thousands of distinct sizes per step; this
//!    pool reuses a single size across every layer of decode.
//!  - Training pool grows; this pool is fixed-size and bounded by
//!    1 % of device-local memory.
//!  - Training pool entries are recycled by Arc-strong-count; this
//!    pool round-robins through a fixed cursor.

use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fixed ring of reusable intermediate `VulkanBuffer`s ping-ponged
/// across layer boundaries of one Vulkan-resident decode step.
///
/// See module docs.
pub struct DecodeResidentPool {
    slots: Vec<Arc<VulkanBuffer>>,
    slot_bytes: u64,
    cursor: AtomicUsize,
}

impl std::fmt::Debug for DecodeResidentPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeResidentPool")
            .field("num_slots", &self.slots.len())
            .field("slot_bytes", &self.slot_bytes)
            .finish()
    }
}

/// Minimum slot count below which we fall back to the per-call
/// Tensor path. Three is enough to triple-buffer one Vulkan kernel
/// reading one input and writing one output while a *next* output
/// is staged.
const MIN_SLOTS: usize = 3;

/// Preferred slot count. Four is enough to thread the SwiGLU MLP's
/// `gate`, `up`, `down`, and the layer residual buffer through a
/// single decode block without re-using any buffer that is still
/// in-flight on the same submit.
const PREFERRED_SLOTS: usize = 4;

/// Fraction of the device-local heap the pool is allowed to claim.
/// 1 % was picked so the pool is invisible against multi-GB weight
/// footprints: on a 24 GiB / 48 GiB GPU it is 240 MiB / 480 MiB
/// ceiling — orders of magnitude above the ≈ 10 MiB the ring needs
/// at Qwen3.5-4B `max_batch=64`. On a 16 GiB shared-memory iGPU the
/// 160 MiB cap is also comfortably above the ring footprint, so
/// `try_new` only fails when the entire 1 % budget has already been
/// claimed by something else (extremely tight UMA).
const BUDGET_FRACTION: f64 = 0.01;

impl DecodeResidentPool {
    /// Try to construct a ring sized to
    /// `max(hidden, intermediate) * max_batch * 4` bytes per slot.
    ///
    /// Returns `Ok(None)` (with a one-time `tracing::warn!`) when the
    /// device cannot fit even the minimum `MIN_SLOTS` slots within
    /// 1 % of the device-local heap, or when any of the slot
    /// allocations themselves fails. Callers transparently fall back
    /// to the existing per-call Tensor path on `None`.
    pub fn try_new(
        device: &Arc<VulkanDevice>,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Result<Option<Self>> {
        let elems = max_hidden
            .max(max_intermediate)
            .saturating_mul(max_batch.max(1));
        // f32 (4 bytes) — the resident path keeps everything in f32 at
        // the intermediate stage; the bf16 weight buffers are not part
        // of this pool.
        let slot_bytes = (elems as u64).saturating_mul(4).max(4);

        let heap = device.device_local_heap_bytes();
        let budget = if heap > 0 {
            ((heap as f64) * BUDGET_FRACTION).floor() as u64
        } else {
            // Driver reports no heap size: skip the budget gate and
            // attempt the preferred ring directly. If physical
            // allocation fails the slot-allocation loop below will
            // catch it.
            (PREFERRED_SLOTS as u64).saturating_mul(slot_bytes)
        };

        if (MIN_SLOTS as u64).saturating_mul(slot_bytes) > budget {
            tracing::warn!(
                slot_bytes,
                budget,
                heap_bytes = heap,
                "Vulkan-resident decode pool cannot fit minimum {MIN_SLOTS} slots within \
                 {pct:.1}% of the device-local heap; falling back to per-call Tensor path",
                pct = BUDGET_FRACTION * 100.0,
            );
            return Ok(None);
        }

        let mut num_slots = if slot_bytes == 0 {
            PREFERRED_SLOTS
        } else {
            ((budget / slot_bytes) as usize).min(PREFERRED_SLOTS)
        };
        if num_slots < MIN_SLOTS {
            num_slots = MIN_SLOTS;
        }

        let mut slots: Vec<Arc<VulkanBuffer>> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            match VulkanBuffer::create_device_local(
                device.device(),
                device.device_local_mem_type(),
                slot_bytes,
            ) {
                Ok(buf) => slots.push(Arc::new(buf)),
                Err(e) => {
                    tracing::warn!(
                        slot = i,
                        slot_bytes,
                        error = %e,
                        "Vulkan-resident decode pool slot allocation failed; falling back \
                         to per-call Tensor path"
                    );
                    return Ok(None);
                }
            }
        }

        tracing::info!(
            num_slots,
            slot_bytes,
            total_bytes = slot_bytes * (num_slots as u64),
            heap_bytes = heap,
            "Vulkan-resident decode pool ready"
        );

        Ok(Some(Self {
            slots,
            slot_bytes,
            cursor: AtomicUsize::new(0),
        }))
    }

    /// Bytes per slot. Useful for kernel callers that need to assert
    /// their `out_dim * batch * 4` payload fits before dispatch.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Number of buffers in the ring.
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Round-robin acquire the next slot in the ring. Returns an
    /// `Arc<VulkanBuffer>` that the caller holds across one dispatch
    /// and then drops; the next `acquire()` returns the *next* slot
    /// in the ring, ensuring two consecutive `acquire()` calls never
    /// hand out the same buffer.
    ///
    /// The ring size is chosen so that across one transformer-block
    /// (read-input, write-norm, read-norm, write-attn, etc.) every
    /// active read+write pair lands on distinct slots.
    pub fn acquire(&self) -> Arc<VulkanBuffer> {
        let n = self.slots.len();
        // Empty pool shouldn't happen if try_new succeeded but guard.
        debug_assert!(n > 0, "DecodeResidentPool::acquire on empty pool");
        let i = self.cursor.fetch_add(1, Ordering::Relaxed) % n;
        Arc::clone(&self.slots[i])
    }

    /// Reset the cursor to slot 0. Call this at the start of every
    /// decode step so the ring's per-layer ping-pong is deterministic
    /// across steps and easier to reason about in dispatcher code.
    pub fn reset_cursor(&self) {
        self.cursor.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_on_unavailable_device_is_skipped() {
        // No assertions needed — this test mainly exists so the module
        // compiles and is exercised when the Vulkan device is up; the
        // real budget / allocation logic is exercised in
        // `decode_resident_pool_constructs_when_device_up` below.
    }

    #[test]
    fn decode_resident_pool_constructs_when_device_up() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        let arc = Arc::new(dev);
        // Qwen3.5-4B shapes: hidden=2560, intermediate=9216, max_batch=64.
        let pool = DecodeResidentPool::try_new(&arc, 2560, 9216, 64)
            .expect("try_new should not error when the device is up");
        let pool = pool.expect("pool should fit on the test GPU");
        assert!(
            (MIN_SLOTS..=PREFERRED_SLOTS).contains(&pool.num_slots()),
            "ring size {} should fall within [{MIN_SLOTS}, {PREFERRED_SLOTS}]",
            pool.num_slots(),
        );
        assert_eq!(
            pool.slot_bytes() as usize,
            9216 * 64 * 4,
            "slot bytes should match max(hidden,intermediate) * max_batch * 4"
        );
        // Round-robin: across 2*num_slots acquires, we must touch every
        // slot at least once.
        let mut seen = vec![false; pool.num_slots()];
        for _ in 0..2 * pool.num_slots() {
            let buf = pool.acquire();
            for (i, slot) in pool.slots.iter().enumerate() {
                if Arc::ptr_eq(slot, &buf) {
                    seen[i] = true;
                    break;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "ring should round-robin all slots");
    }

    #[test]
    fn decode_resident_pool_reset_cursor_restarts_ring() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        let arc = Arc::new(dev);
        let pool = DecodeResidentPool::try_new(&arc, 64, 64, 1)
            .unwrap()
            .expect("small pool should fit");
        let first = pool.acquire();
        let _second = pool.acquire();
        pool.reset_cursor();
        let first_again = pool.acquire();
        assert!(
            Arc::ptr_eq(&first, &first_again),
            "after reset_cursor, the next acquire must return the same slot as the first acquire \
             of the previous burst"
        );
    }
}
