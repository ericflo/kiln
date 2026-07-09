//! KV cache autoscaler (#24/#26) — the training/inference VRAM arbiter.
//!
//! A background policy thread that watches the [`MemoryGovernor`]'s live,
//! all-process memory view and physically resizes the paged-KV cache to match:
//! SHRINK when VRAM gets tight (a coexisting training run / process needs it —
//! the freed KV bytes return to the device pool for that workload to reuse), and
//! GROW back toward the startup size when headroom returns. This is what makes
//! "most workloads are 100% inference, but training dynamically takes VRAM and
//! gives it back — never OOM" real.
//!
//! The actual resize runs on the batching-engine actor at its between-steps
//! barrier (via `BatchingEngineHandle::resize_kv_blocking`), which takes
//! exclusive GPU access so no decode/training kernel races the pool swap. This
//! thread only decides the target and rate-limits.
//!
//! Conservative by design — a control loop that thrashes would stall decode
//! (each resize briefly blocks the GPU). Hysteresis (distinct shrink/grow
//! thresholds), a step cap, a minimum floor, and a cooldown prevent oscillation.
//! Disable with `KILN_KV_AUTOSCALE=0`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kiln_memory::{MemoryGovernor, MemoryPressure};
use kiln_model::{GpuAllocatorMemoryProbePolicy, PagedKvCacheKt};

use crate::batching_engine::BatchingEngineHandle;

/// Startup state exposed through `/health`. This distinguishes an operator
/// request from a control loop that actually owns a usable device KV pool.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct KvAutoscalerState {
    pub requested: bool,
    pub enabled: bool,
    pub state: &'static str,
    pub reason: &'static str,
    pub start_blocks: Option<usize>,
    pub min_blocks: Option<usize>,
    pub bytes_per_block: Option<usize>,
}

impl KvAutoscalerState {
    pub fn unavailable(reason: &'static str) -> Self {
        Self {
            requested: !is_disabled(),
            enabled: false,
            state: "unavailable",
            reason,
            start_blocks: None,
            min_blocks: None,
            bytes_per_block: None,
        }
    }

    fn disabled() -> Self {
        Self {
            requested: false,
            enabled: false,
            state: "disabled",
            reason: "environment",
            start_blocks: None,
            min_blocks: None,
            bytes_per_block: None,
        }
    }
}

/// How often the policy re-evaluates.
const TICK: Duration = Duration::from_secs(2);
/// Minimum spacing between two resizes — a resize blocks the GPU briefly, so we
/// never react faster than this even if pressure swings.
const COOLDOWN: Duration = Duration::from_secs(8);
/// Target free-VRAM headroom the policy steers toward, as a multiple of one
/// block's bytes. Below `SHRINK` we free KV; above `GROW` (and below the
/// startup size) we reclaim it. The gap is the anti-thrash dead-band.
const HEADROOM_SHRINK_BLOCKS: u64 = 64;
const HEADROOM_GROW_BLOCKS: u64 = 512;
/// Never resize by more than this fraction of the current size in one step, so a
/// transient spike can't collapse the cache.
const MAX_STEP_FRACTION: f64 = 0.35;

#[derive(Clone, Copy)]
struct Bounds {
    /// Startup block count — the grow ceiling.
    max_blocks: usize,
    /// Never shrink below this (keeps inference viable even under pressure).
    min_blocks: usize,
}

fn is_disabled() -> bool {
    matches!(
        std::env::var("KILN_KV_AUTOSCALE").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("off") | Ok("OFF") | Ok("no")
    )
}

/// Spawn the autoscaler thread. No-op (returns without spawning) if disabled, or
/// if the cache can't report its geometry. Idempotent is the CALLER's concern.
pub fn spawn(
    engine: BatchingEngineHandle,
    paged_cache: Arc<PagedKvCacheKt>,
    gpu_allocator_memory_probe_policy: GpuAllocatorMemoryProbePolicy,
) -> KvAutoscalerState {
    if is_disabled() {
        tracing::info!("KV autoscaler disabled (KILN_KV_AUTOSCALE=0)");
        return KvAutoscalerState::disabled();
    }
    let bytes_per_block = paged_cache.bytes_per_block();
    let start_blocks = paged_cache.num_blocks();
    if bytes_per_block == 0 || start_blocks == 0 {
        tracing::info!("KV autoscaler not started: cache reports no geometry");
        return KvAutoscalerState::unavailable("cache_geometry");
    }
    // Keep at least a quarter of the startup cache (and >= 1) under any pressure.
    let bounds = Bounds {
        max_blocks: start_blocks,
        min_blocks: (start_blocks / 4).max(1),
    };
    tracing::info!(
        start_blocks,
        min_blocks = bounds.min_blocks,
        bytes_per_block,
        "KV autoscaler started (set KILN_KV_AUTOSCALE=0 to disable)"
    );

    std::thread::Builder::new()
        .name("kiln-kv-autoscaler".to_string())
        .spawn(move || {
            run(
                engine,
                paged_cache,
                gpu_allocator_memory_probe_policy,
                bytes_per_block as u64,
                bounds,
            )
        })
        .expect("spawn kv autoscaler");

    KvAutoscalerState {
        requested: true,
        enabled: true,
        state: "enabled",
        reason: "active",
        start_blocks: Some(start_blocks),
        min_blocks: Some(bounds.min_blocks),
        bytes_per_block: Some(bytes_per_block),
    }
}

fn run(
    engine: BatchingEngineHandle,
    paged_cache: Arc<PagedKvCacheKt>,
    gpu_allocator_memory_probe_policy: GpuAllocatorMemoryProbePolicy,
    bytes_per_block: u64,
    bounds: Bounds,
) {
    let gov = MemoryGovernor::global();
    let mut last_resize = Instant::now() - COOLDOWN;

    // Verification/debug knob: KILN_KV_FORCE_BLOCKS=N performs ONE resize to N
    // blocks at startup (then normal policy resumes). Lets an operator confirm
    // the full resize path end-to-end on a box that isn't under real pressure.
    if let Some(target) = std::env::var("KILN_KV_FORCE_BLOCKS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        let requested = target;
        let cur = paged_cache.num_blocks();
        let avail = live_safe_available_bytes(gpu_allocator_memory_probe_policy, gov, &paged_cache);
        let target = cap_grow_target_by_available(cur, requested, avail, bytes_per_block);
        if target < requested {
            tracing::warn!(
                requested,
                capped = target,
                cur,
                avail_mb = avail / (1024 * 1024),
                "KV autoscaler forced grow capped by live allocator memory"
            );
        }
        match engine.resize_kv_blocking(target) {
            Ok(achieved) => {
                last_resize = Instant::now();
                tracing::info!(
                    requested = target,
                    achieved,
                    "KV autoscaler FORCED resize (KILN_KV_FORCE_BLOCKS)"
                );
            }
            Err(err) => tracing::warn!(error = %err, "KV autoscaler forced resize failed"),
        }
    }

    loop {
        // Event-driven (#35): wake immediately when the budget changes (a
        // training reservation taken → shrink; a job ends/reservation drops →
        // grow KV back) rather than always waiting out the poll tick. The TICK
        // timeout still backstops EXTERNAL changes (a coexisting process) that
        // only the periodic probe sees.
        gov.wait_for_change(TICK);
        if last_resize.elapsed() < COOLDOWN {
            continue;
        }
        let cur = paged_cache.num_blocks();
        let avail = live_safe_available_bytes(gpu_allocator_memory_probe_policy, gov, &paged_cache);
        let pressure = gov.pressure();

        let target = decide_target(cur, avail, bytes_per_block, pressure, bounds);
        let Some(target) = target else { continue };

        match engine.resize_kv_blocking(target) {
            Ok(achieved) => {
                last_resize = Instant::now();
                if achieved != cur {
                    tracing::info!(
                        from = cur,
                        to = achieved,
                        requested = target,
                        pressure = ?pressure,
                        avail_mb = avail / (1024 * 1024),
                        "KV autoscaler resized cache"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "KV autoscaler resize failed");
                last_resize = Instant::now(); // back off on error too
            }
        }
    }
}

fn live_safe_available_bytes(
    gpu_allocator_memory_probe_policy: GpuAllocatorMemoryProbePolicy,
    gov: &MemoryGovernor,
    paged_cache: &PagedKvCacheKt,
) -> u64 {
    // available_bytes is the governor's all-process free-VRAM estimate minus
    // soft reservations. CUDA/ROCm also expose the allocator heap the KV tensors
    // actually grow from; use the stricter signal when present so an optimistic
    // OS snapshot cannot drive a backend allocation failure.
    let governor_avail = gov.available_bytes();
    let allocator_avail = paged_cache.device().and_then(|device| {
        crate::device_memory::allocator_safe_available_bytes(
            gpu_allocator_memory_probe_policy,
            gov,
            &device,
        )
    });
    allocator_avail
        .map(|allocator| governor_avail.min(allocator))
        .unwrap_or(governor_avail)
}

fn cap_grow_target_by_available(
    cur: usize,
    target: usize,
    avail: u64,
    bytes_per_block: u64,
) -> usize {
    if target <= cur || bytes_per_block == 0 {
        return target;
    }
    let grow_blocks = (avail / bytes_per_block) as usize;
    target.min(cur.saturating_add(grow_blocks))
}

/// Pure policy: given the current block count, available VRAM, per-block bytes,
/// pressure, and bounds, return the new target block count — or `None` to hold.
/// Hysteresis: shrink only when clearly tight, grow only when clearly roomy.
fn decide_target(
    cur: usize,
    avail: u64,
    bytes_per_block: u64,
    pressure: MemoryPressure,
    bounds: Bounds,
) -> Option<usize> {
    let avail_blocks = avail / bytes_per_block;
    let max_step = ((cur as f64) * MAX_STEP_FRACTION).ceil() as usize;

    // SHRINK: pressure is high OR free headroom has fallen below the low mark.
    let tight = matches!(pressure, MemoryPressure::Tight | MemoryPressure::Critical)
        || avail_blocks < HEADROOM_SHRINK_BLOCKS;
    if tight && cur > bounds.min_blocks {
        // Free enough blocks to restore the low headroom, capped by the step.
        let deficit = HEADROOM_SHRINK_BLOCKS.saturating_sub(avail_blocks) as usize;
        let step = deficit.clamp(1, max_step);
        let target = cur.saturating_sub(step).max(bounds.min_blocks);
        return (target < cur).then_some(target);
    }

    // GROW: pressure is comfortable AND there's room well above the high mark,
    // and we're below the startup size.
    let roomy =
        matches!(pressure, MemoryPressure::Comfortable) && avail_blocks > HEADROOM_GROW_BLOCKS;
    if roomy && cur < bounds.max_blocks {
        // Use the surplus above the high mark, capped by step and the ceiling.
        let surplus = (avail_blocks - HEADROOM_GROW_BLOCKS) as usize;
        let step = surplus.clamp(1, max_step);
        let target = (cur + step).min(bounds.max_blocks);
        return (target > cur).then_some(target);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const BPB: u64 = 10 * 1024 * 1024; // 10 MB/block
    fn bounds() -> Bounds {
        Bounds {
            max_blocks: 1000,
            min_blocks: 250,
        }
    }

    #[test]
    fn shrinks_under_critical_pressure() {
        // Tiny headroom → shrink, but never below the floor, never beyond step.
        let t = decide_target(1000, 0, BPB, MemoryPressure::Critical, bounds());
        let t = t.expect("should shrink");
        assert!(t < 1000 && t >= 250, "target {t}");
        assert!(1000 - t <= 350, "step cap (35% of 1000)");
    }

    #[test]
    fn does_not_shrink_below_floor() {
        let t = decide_target(250, 0, BPB, MemoryPressure::Critical, bounds());
        assert_eq!(t, None, "already at floor");
    }

    #[test]
    fn grows_when_comfortable_with_headroom() {
        // 5000 blocks of headroom, comfortable, cur below ceiling → grow.
        let avail = 5000 * BPB;
        let t = decide_target(300, avail, BPB, MemoryPressure::Comfortable, bounds());
        let t = t.expect("should grow");
        assert!(t > 300 && t <= 1000, "target {t}");
    }

    #[test]
    fn holds_in_the_deadband() {
        // Moderate pressure, headroom between the marks → no change (no thrash).
        let avail = 200 * BPB; // between SHRINK(64) and GROW(512)
        assert_eq!(
            decide_target(500, avail, BPB, MemoryPressure::Moderate, bounds()),
            None
        );
    }

    #[test]
    fn does_not_grow_past_ceiling() {
        let avail = 99999 * BPB;
        assert_eq!(
            decide_target(1000, avail, BPB, MemoryPressure::Comfortable, bounds()),
            None
        );
    }

    #[test]
    fn forced_grow_target_is_capped_by_available_blocks() {
        assert_eq!(cap_grow_target_by_available(500, 900, 120 * BPB, BPB), 620);
        assert_eq!(cap_grow_target_by_available(500, 450, 0, BPB), 450);
    }
}
