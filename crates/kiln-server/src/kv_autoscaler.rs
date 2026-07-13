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
//! The actual resize runs on the batching-engine actor after active requests
//! drain (via `BatchingEngineHandle::resize_kv_blocking`), which takes exclusive
//! GPU access so no decode/training kernel races the pool swap. This thread only
//! decides the target, reserves replacement headroom, and rate-limits.
//!
//! Conservative by design — a control loop that thrashes would stall decode
//! (each resize briefly blocks the GPU). Hysteresis (distinct shrink/grow
//! thresholds), a step cap, a minimum floor, and a cooldown prevent oscillation.
//! Disable with `KILN_KV_AUTOSCALE=0`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kiln_memory::{MemoryGovernor, MemoryPressure, Reservation};
use kiln_model::{GpuAllocatorMemoryProbePolicy, PagedKvCacheKt};

use crate::batching_engine::{BatchingEngineHandle, KvResizeReason};

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
/// Allocation or headroom failures increase the next retry delay up to this
/// bound. A memory-change notification still wakes the thread, but cannot turn
/// a persistent failure into a two-second allocation loop.
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(128);
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

/// A replacement pool that is known to fit in currently available staging
/// memory. Physical resize keeps the complete old pool alive until commit, so
/// `replacement_bytes` is the full target size, not a grow delta.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KvResizeStagingPlan {
    pub target_blocks: usize,
    pub replacement_bytes: u64,
}

/// Fit a requested replacement into staging headroom without crossing the
/// caller's minimum acceptable target. For shrink, the caller controls how much
/// deeper than the policy request this transaction may go; for grow, it passes
/// `current_blocks + 1` so a partial but useful grow is allowed.
pub(crate) fn plan_resize_with_staging_headroom(
    current_blocks: usize,
    requested_blocks: usize,
    minimum_target_blocks: usize,
    staging_available_bytes: u64,
    bytes_per_block: u64,
) -> Option<KvResizeStagingPlan> {
    if bytes_per_block == 0 || requested_blocks == 0 || requested_blocks == current_blocks {
        return None;
    }
    let max_staged_blocks =
        usize::try_from(staging_available_bytes / bytes_per_block).unwrap_or(usize::MAX);
    let target_blocks = requested_blocks.min(max_staged_blocks);
    if target_blocks < minimum_target_blocks || target_blocks == current_blocks {
        return None;
    }
    if requested_blocks < current_blocks && target_blocks >= current_blocks {
        return None;
    }
    if requested_blocks > current_blocks && target_blocks <= current_blocks {
        return None;
    }
    Some(KvResizeStagingPlan {
        target_blocks,
        replacement_bytes: (target_blocks as u64).saturating_mul(bytes_per_block),
    })
}

struct ReservedKvResize<'a> {
    plan: KvResizeStagingPlan,
    _reservation: Reservation<'a>,
    replanned_after_contention: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvResizeReservationFailure {
    InsufficientHeadroom {
        staging_available_bytes: u64,
    },
    Contended {
        initial_plan: KvResizeStagingPlan,
        replanned_plan: Option<KvResizeStagingPlan>,
        refreshed_staging_available_bytes: u64,
    },
}

/// Plan and atomically claim the complete replacement pool. If another
/// in-process consumer wins memory after the initial observation, replan once
/// from a fresh, governor-capped budget. A second loss is returned to the
/// control loop so its bounded retry backoff applies; physical allocation must
/// never proceed under an unchecked reservation.
fn plan_and_reserve_resize_with_replan<'a>(
    governor: &'a MemoryGovernor,
    current_blocks: usize,
    requested_blocks: usize,
    minimum_target_blocks: usize,
    initial_staging_available_bytes: u64,
    bytes_per_block: u64,
    refresh_staging_available_bytes: impl FnOnce() -> u64,
) -> Result<ReservedKvResize<'a>, KvResizeReservationFailure> {
    let Some(initial_plan) = plan_resize_with_staging_headroom(
        current_blocks,
        requested_blocks,
        minimum_target_blocks,
        initial_staging_available_bytes,
        bytes_per_block,
    ) else {
        return Err(KvResizeReservationFailure::InsufficientHeadroom {
            staging_available_bytes: initial_staging_available_bytes,
        });
    };

    if let Some(reservation) = governor.try_reserve_cached(initial_plan.replacement_bytes) {
        return Ok(ReservedKvResize {
            plan: initial_plan,
            _reservation: reservation,
            replanned_after_contention: false,
        });
    }

    let refreshed_staging_available_bytes =
        refresh_staging_available_bytes().min(governor.cached_observation().available_bytes);
    let replanned_plan = plan_resize_with_staging_headroom(
        current_blocks,
        requested_blocks,
        minimum_target_blocks,
        refreshed_staging_available_bytes,
        bytes_per_block,
    );
    if let Some(plan) = replanned_plan {
        if let Some(reservation) = governor.try_reserve_cached(plan.replacement_bytes) {
            return Ok(ReservedKvResize {
                plan,
                _reservation: reservation,
                replanned_after_contention: true,
            });
        }
    }

    Err(KvResizeReservationFailure::Contended {
        initial_plan,
        replanned_plan,
        refreshed_staging_available_bytes,
    })
}

fn next_retry_backoff(current: Duration) -> Duration {
    current.saturating_mul(2).min(MAX_RETRY_BACKOFF)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResizeMemorySnapshot {
    /// Conservative policy input: respect both the process-wide budget and the
    /// backend allocator heap.
    policy_available_bytes: u64,
    /// Actual replacement-allocation input. ROCm includes reusable HIP-pool
    /// spare here, which the OS-wide snapshot reports as unavailable.
    staging_available_bytes: u64,
    pressure: MemoryPressure,
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
    let mut next_attempt = Instant::now();
    let mut retry_backoff = COOLDOWN;

    // Verification/debug knob: KILN_KV_FORCE_BLOCKS=N performs ONE resize to N
    // blocks at startup (then normal policy resumes). Lets an operator confirm
    // the full resize path end-to-end on a box that isn't under real pressure.
    if let Some(target) = std::env::var("KILN_KV_FORCE_BLOCKS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        let requested = target;
        let cur = paged_cache.num_blocks();
        let memory =
            live_resize_memory_snapshot(gpu_allocator_memory_probe_policy, gov, &paged_cache);
        // Forced shrink is exact: silently shrinking below an operator's target
        // would be surprising. Forced grow retains the prior partial-grow
        // behavior, now capped by the full replacement size rather than delta.
        let minimum_target = if requested < cur {
            requested
        } else {
            cur.saturating_add(1)
        };
        if requested == cur {
            tracing::debug!(
                requested,
                "KV autoscaler forced resize is already satisfied"
            );
        } else {
            let reservation_attempt = plan_and_reserve_resize_with_replan(
                gov,
                cur,
                requested,
                minimum_target,
                memory.staging_available_bytes,
                bytes_per_block,
                || {
                    live_resize_memory_snapshot(
                        gpu_allocator_memory_probe_policy,
                        gov,
                        &paged_cache,
                    )
                    .staging_available_bytes
                },
            );
            match reservation_attempt {
                Ok(claim) => {
                    let plan = claim.plan;
                    if plan.target_blocks < requested {
                        tracing::warn!(
                            requested,
                            capped = plan.target_blocks,
                            cur,
                            replanned_after_contention = claim.replanned_after_contention,
                            staging_available_mb = memory.staging_available_bytes / (1024 * 1024),
                            replacement_mb = plan.replacement_bytes / (1024 * 1024),
                            "KV autoscaler forced grow capped by replacement-pool staging headroom"
                        );
                    }
                    match engine
                        .resize_kv_blocking(plan.target_blocks, KvResizeReason::ForcedConfiguration)
                    {
                        Ok(achieved) => {
                            next_attempt = Instant::now() + COOLDOWN;
                            retry_backoff = COOLDOWN;
                            tracing::info!(
                                requested,
                                planned = plan.target_blocks,
                                achieved,
                                replacement_mb = plan.replacement_bytes / (1024 * 1024),
                                "KV autoscaler FORCED resize (KILN_KV_FORCE_BLOCKS)"
                            );
                        }
                        Err(err) => {
                            let applied_backoff = retry_backoff;
                            next_attempt = Instant::now() + applied_backoff;
                            retry_backoff = next_retry_backoff(retry_backoff);
                            tracing::warn!(
                                error = %err,
                                requested,
                                planned = plan.target_blocks,
                                retry_after_ms = applied_backoff.as_millis() as u64,
                                "KV autoscaler forced resize failed"
                            );
                        }
                    }
                }
                Err(failure) => {
                    let applied_backoff = retry_backoff;
                    next_attempt = Instant::now() + applied_backoff;
                    retry_backoff = next_retry_backoff(retry_backoff);
                    match failure {
                        KvResizeReservationFailure::InsufficientHeadroom {
                            staging_available_bytes,
                        } => tracing::warn!(
                            requested,
                            cur,
                            staging_available_mb = staging_available_bytes / (1024 * 1024),
                            bytes_per_block,
                            retry_after_ms = applied_backoff.as_millis() as u64,
                            "KV autoscaler forced resize skipped: full replacement pool lacks staging headroom"
                        ),
                        KvResizeReservationFailure::Contended {
                            initial_plan,
                            replanned_plan,
                            refreshed_staging_available_bytes,
                        } => tracing::warn!(
                            requested,
                            cur,
                            initial_planned = initial_plan.target_blocks,
                            replanned = replanned_plan.map(|plan| plan.target_blocks),
                            refreshed_staging_available_mb =
                                refreshed_staging_available_bytes / (1024 * 1024),
                            retry_after_ms = applied_backoff.as_millis() as u64,
                            "KV autoscaler forced resize lost its staging reservation; backing off"
                        ),
                    }
                }
            }
        }
    }

    loop {
        // Event-driven (#35): wake immediately when the budget changes (a
        // training reservation taken → shrink; a job ends/reservation drops →
        // grow KV back) rather than always waiting out the poll tick. The TICK
        // timeout still backstops EXTERNAL changes (a coexisting process) that
        // only the periodic probe sees.
        gov.wait_for_change(TICK);
        if Instant::now() < next_attempt {
            continue;
        }
        let cur = paged_cache.num_blocks();
        let memory =
            live_resize_memory_snapshot(gpu_allocator_memory_probe_policy, gov, &paged_cache);
        let requested = decide_target(
            cur,
            memory.policy_available_bytes,
            bytes_per_block,
            memory.pressure,
            bounds,
        );
        let Some(requested) = requested else {
            retry_backoff = COOLDOWN;
            continue;
        };
        let minimum_target = if requested < cur {
            cur.saturating_sub(max_step_blocks(cur))
                .max(bounds.min_blocks)
        } else {
            cur.saturating_add(1)
        };
        let claim = match plan_and_reserve_resize_with_replan(
            gov,
            cur,
            requested,
            minimum_target,
            memory.staging_available_bytes,
            bytes_per_block,
            || {
                live_resize_memory_snapshot(gpu_allocator_memory_probe_policy, gov, &paged_cache)
                    .staging_available_bytes
            },
        ) {
            Ok(claim) => claim,
            Err(failure) => {
                let applied_backoff = retry_backoff;
                next_attempt = Instant::now() + applied_backoff;
                retry_backoff = next_retry_backoff(retry_backoff);
                match failure {
                    KvResizeReservationFailure::InsufficientHeadroom {
                        staging_available_bytes,
                    } => tracing::warn!(
                        from = cur,
                        requested,
                        minimum_target,
                        pressure = ?memory.pressure,
                        policy_available_mb = memory.policy_available_bytes / (1024 * 1024),
                        staging_available_mb = staging_available_bytes / (1024 * 1024),
                        retry_after_ms = applied_backoff.as_millis() as u64,
                        "KV autoscaler resize skipped: full replacement pool lacks staging headroom"
                    ),
                    KvResizeReservationFailure::Contended {
                        initial_plan,
                        replanned_plan,
                        refreshed_staging_available_bytes,
                    } => tracing::warn!(
                        from = cur,
                        requested,
                        minimum_target,
                        initial_planned = initial_plan.target_blocks,
                        replanned = replanned_plan.map(|plan| plan.target_blocks),
                        pressure = ?memory.pressure,
                        refreshed_staging_available_mb =
                            refreshed_staging_available_bytes / (1024 * 1024),
                        retry_after_ms = applied_backoff.as_millis() as u64,
                        "KV autoscaler resize lost its staging reservation; backing off"
                    ),
                }
                continue;
            }
        };
        let plan = claim.plan;
        if plan.target_blocks != requested {
            tracing::info!(
                from = cur,
                requested,
                planned = plan.target_blocks,
                minimum_target,
                replanned_after_contention = claim.replanned_after_contention,
                staging_available_mb = memory.staging_available_bytes / (1024 * 1024),
                replacement_mb = plan.replacement_bytes / (1024 * 1024),
                "KV autoscaler adjusted target to fit transactional staging headroom"
            );
        }

        match engine.resize_kv_blocking(plan.target_blocks, KvResizeReason::AutomaticMemoryPolicy) {
            Ok(achieved) => {
                next_attempt = Instant::now() + COOLDOWN;
                retry_backoff = COOLDOWN;
                if achieved != cur {
                    tracing::info!(
                        from = cur,
                        to = achieved,
                        requested,
                        planned = plan.target_blocks,
                        pressure = ?memory.pressure,
                        policy_available_mb = memory.policy_available_bytes / (1024 * 1024),
                        staging_available_mb = memory.staging_available_bytes / (1024 * 1024),
                        replacement_mb = plan.replacement_bytes / (1024 * 1024),
                        "KV autoscaler resized cache"
                    );
                } else {
                    let applied_backoff = retry_backoff;
                    next_attempt = Instant::now() + applied_backoff;
                    retry_backoff = next_retry_backoff(retry_backoff);
                    tracing::warn!(
                        from = cur,
                        requested,
                        planned = plan.target_blocks,
                        retry_after_ms = applied_backoff.as_millis() as u64,
                        "KV autoscaler resize produced no capacity change; backing off"
                    );
                }
            }
            Err(err) => {
                let applied_backoff = retry_backoff;
                next_attempt = Instant::now() + applied_backoff;
                retry_backoff = next_retry_backoff(retry_backoff);
                tracing::warn!(
                    error = %err,
                    from = cur,
                    requested,
                    planned = plan.target_blocks,
                    replacement_mb = plan.replacement_bytes / (1024 * 1024),
                    retry_after_ms = applied_backoff.as_millis() as u64,
                    "KV autoscaler resize failed"
                );
            }
        }
    }
}

fn live_resize_memory_snapshot(
    gpu_allocator_memory_probe_policy: GpuAllocatorMemoryProbePolicy,
    gov: &MemoryGovernor,
    paged_cache: &PagedKvCacheKt,
) -> ResizeMemorySnapshot {
    // available_bytes is the governor's all-process free-VRAM estimate minus
    // soft reservations. CUDA/ROCm also expose the allocator heap the KV tensors
    // actually grow from; use the stricter signal when present so an optimistic
    // OS snapshot cannot drive a backend allocation failure.
    let published = gov.cached_observation();
    let governor_avail = published.available_bytes;
    let allocator_avail = paged_cache.device().and_then(|device| {
        crate::device_memory::allocator_safe_available_bytes_with_soft_reserved(
            gpu_allocator_memory_probe_policy,
            &device,
            gov.config().floor_bytes,
            published.soft_reserved_bytes,
        )
    });
    ResizeMemorySnapshot {
        policy_available_bytes: allocator_avail
            .map(|allocator| governor_avail.min(allocator))
            .unwrap_or(governor_avail),
        staging_available_bytes: allocator_avail.unwrap_or(governor_avail),
        pressure: published.pressure,
    }
}

fn max_step_blocks(current_blocks: usize) -> usize {
    ((current_blocks as f64) * MAX_STEP_FRACTION).ceil() as usize
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
    let max_step = max_step_blocks(cur);

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
    use std::sync::Mutex;

    use kiln_memory::{GovernorConfig, MemorySnapshot, MemorySource};

    use super::*;

    const BPB: u64 = 10 * 1024 * 1024; // 10 MB/block
    fn bounds() -> Bounds {
        Bounds {
            max_blocks: 1000,
            min_blocks: 250,
        }
    }

    struct FixedMemorySource(Mutex<MemorySnapshot>);

    impl MemorySource for FixedMemorySource {
        fn probe(&self) -> MemorySnapshot {
            *self.0.lock().unwrap()
        }
    }

    fn reservation_governor(available_blocks: u64) -> MemoryGovernor {
        MemoryGovernor::with_source(
            Box::new(FixedMemorySource(Mutex::new(MemorySnapshot {
                total_bytes: available_blocks * BPB,
                used_bytes: 0,
                free_bytes: available_blocks * BPB,
                source: kiln_memory::vram::VramSource::None,
                unified: false,
                observations: Default::default(),
            }))),
            GovernorConfig {
                floor_bytes: 0,
                ..GovernorConfig::default()
            },
        )
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
    fn grow_requires_the_full_replacement_pool_not_only_the_delta() {
        assert_eq!(
            plan_resize_with_staging_headroom(500, 900, 501, 120 * BPB, BPB),
            None,
            "120 blocks of free memory cannot stage any grow from 500 blocks"
        );
        assert_eq!(
            plan_resize_with_staging_headroom(500, 900, 501, 700 * BPB, BPB),
            Some(KvResizeStagingPlan {
                target_blocks: 700,
                replacement_bytes: 700 * BPB,
            })
        );
    }

    #[test]
    fn shrink_can_deepen_within_the_explicit_step_bound_to_fit_staging() {
        assert_eq!(
            plan_resize_with_staging_headroom(1000, 900, 650, 800 * BPB, BPB),
            Some(KvResizeStagingPlan {
                target_blocks: 800,
                replacement_bytes: 800 * BPB,
            })
        );
        assert_eq!(
            plan_resize_with_staging_headroom(1000, 900, 650, 600 * BPB, BPB),
            None,
            "staging must not force a shrink beyond the caller's step bound"
        );
    }

    #[test]
    fn exact_shrink_refuses_to_silently_cross_the_requested_target() {
        assert_eq!(
            plan_resize_with_staging_headroom(1000, 900, 900, 800 * BPB, BPB),
            None
        );
    }

    #[test]
    fn shrink_replans_and_atomically_reserves_the_full_replacement() {
        let governor = reservation_governor(1_000);
        let competing = governor.reserve(200 * BPB);

        let claim =
            plan_and_reserve_resize_with_replan(&governor, 1_000, 900, 650, 900 * BPB, BPB, || {
                900 * BPB
            })
            .expect("the stale 900-block plan should replan to the live 800-block budget");

        assert_eq!(claim.plan.target_blocks, 800);
        assert_eq!(claim._reservation.bytes(), 800 * BPB);
        assert!(claim.replanned_after_contention);
        assert_eq!(governor.soft_reserved_bytes(), 1_000 * BPB);
        drop(claim);
        assert_eq!(governor.soft_reserved_bytes(), 200 * BPB);
        drop(competing);
    }

    #[test]
    fn grow_replans_and_atomically_reserves_the_full_replacement() {
        let governor = reservation_governor(1_000);
        let competing = governor.reserve(300 * BPB);

        let claim =
            plan_and_reserve_resize_with_replan(&governor, 500, 900, 501, 900 * BPB, BPB, || {
                900 * BPB
            })
            .expect("the stale 900-block plan should replan to the live 700-block budget");

        assert_eq!(claim.plan.target_blocks, 700);
        assert_eq!(claim._reservation.bytes(), 700 * BPB);
        assert!(claim.replanned_after_contention);
        assert_eq!(governor.soft_reserved_bytes(), 1_000 * BPB);
        drop(claim);
        drop(competing);
    }

    #[test]
    fn second_reservation_loss_returns_a_backoff_outcome_without_allocating() {
        let governor = reservation_governor(1_000);
        let competing = governor.reserve(500 * BPB);

        let failure =
            plan_and_reserve_resize_with_replan(&governor, 500, 900, 501, 900 * BPB, BPB, || {
                900 * BPB
            })
            .err()
            .expect("no useful grow fits after the stale reservation is rejected");

        assert_eq!(
            failure,
            KvResizeReservationFailure::Contended {
                initial_plan: KvResizeStagingPlan {
                    target_blocks: 900,
                    replacement_bytes: 900 * BPB,
                },
                replanned_plan: None,
                refreshed_staging_available_bytes: 500 * BPB,
            }
        );
        assert_eq!(governor.soft_reserved_bytes(), 500 * BPB);
        drop(competing);
    }

    #[test]
    fn staging_plan_rejects_zero_geometry_and_no_op_targets() {
        assert_eq!(
            plan_resize_with_staging_headroom(500, 500, 1, 1000 * BPB, BPB),
            None
        );
        assert_eq!(
            plan_resize_with_staging_headroom(500, 400, 1, 1000 * BPB, 0),
            None
        );
        assert_eq!(
            plan_resize_with_staging_headroom(500, 0, 0, 1000 * BPB, BPB),
            None
        );
    }

    #[test]
    fn resize_retry_backoff_is_exponential_and_bounded() {
        let mut delay = COOLDOWN;
        assert_eq!(delay, Duration::from_secs(8));
        for expected in [16, 32, 64, 128, 128] {
            delay = next_retry_backoff(delay);
            assert_eq!(delay, Duration::from_secs(expected));
        }
    }
}
