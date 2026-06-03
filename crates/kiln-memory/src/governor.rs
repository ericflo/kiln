//! The [`MemoryGovernor`] — continuous, cross-engine memory awareness.
//!
//! [`crate::vram::current_memory_snapshot`] answers "how much memory is free
//! *right now*", but calling it on every allocation would be wasteful (it spawns
//! `nvidia-smi` / reads sysfs). The governor wraps it in a cheap-to-read,
//! TTL-cached view and layers on the two things every other subsystem needs:
//!
//! * a **pressure level** ([`MemoryPressure`]) derived from the live free
//!   fraction, so allocators / the KV sizer / the arbiter can react *before*
//!   they hit an OOM, and
//! * an **available-budget** calculation that respects a safety floor and any
//!   *soft reservations* — memory a consumer has announced it's about to need
//!   (e.g. a training job's activation peak) but hasn't allocated yet, so it
//!   doesn't get double-handed-out to inference in the meantime.
//!
//! It is backend-agnostic: it only ever reads [`MemorySnapshot`]s, which are
//! themselves OS-level and unified-memory-aware. A single process-wide instance
//! is available via [`MemoryGovernor::global`] so the allocator and other
//! deep-in-the-stack callers can consult it without threading an `Arc` around.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A reclaim hook: "release up to `target` bytes of pooled/cached device memory
/// back to the OS; return how much you freed (0 if unknown/none)." Registered by
/// the allocator layer; invoked by the governor under memory pressure.
pub type Reclaimer = Box<dyn Fn(u64) -> u64 + Send + Sync>;

use crate::vram::{current_memory_snapshot, MemorySnapshot};

/// Source of [`MemorySnapshot`]s. Abstracted so tests can drive the governor
/// with synthetic memory states (no GPU required) and so an integration could
/// later swap in a device-API probe (`hipMemGetInfo`/`cudaMemGetInfo`) without
/// touching the governor itself.
pub trait MemorySource: Send + Sync {
    fn probe(&self) -> MemorySnapshot;
}

/// Default source: the OS-level [`current_memory_snapshot`].
#[derive(Debug, Default, Clone, Copy)]
pub struct OsProbe;

impl MemorySource for OsProbe {
    fn probe(&self) -> MemorySnapshot {
        current_memory_snapshot()
    }
}

/// How tight memory is right now, as a coarse signal for policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Plenty of headroom — grow freely, capture graphs, skip checkpointing.
    Comfortable,
    /// Getting full — stop growing, prefer modest allocations.
    Moderate,
    /// Nearly full — shrink caches, return pool memory, engage checkpointing.
    Tight,
    /// On the edge — refuse new non-essential allocations, evict aggressively.
    Critical,
}

impl MemoryPressure {
    /// True once the system should start *releasing* memory (Tight or worse).
    pub fn should_reclaim(self) -> bool {
        matches!(self, MemoryPressure::Tight | MemoryPressure::Critical)
    }
}

/// Governor tuning. Defaults are conservative and env-overridable via
/// [`GovernorConfig::from_env`].
#[derive(Debug, Clone, Copy)]
pub struct GovernorConfig {
    /// Minimum interval between OS probes; reads inside this window return the
    /// cached snapshot. Keeps `nvidia-smi`/sysfs probing off the hot path.
    pub ttl: Duration,
    /// Bytes never handed out — headroom left for the OS, other apps, and
    /// allocator fragmentation slack. `available_bytes` subtracts this.
    pub floor_bytes: u64,
    /// `free/total` at/below which pressure is [`MemoryPressure::Tight`].
    pub tight_frac: f64,
    /// `free/total` at/below which pressure is [`MemoryPressure::Critical`].
    pub critical_frac: f64,
    /// `free/total` at/above which pressure is [`MemoryPressure::Comfortable`].
    pub comfortable_frac: f64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        GovernorConfig {
            ttl: Duration::from_millis(500),
            floor_bytes: 1024 * 1024 * 1024, // 1 GiB
            critical_frac: 0.05,
            tight_frac: 0.10,
            comfortable_frac: 0.25,
        }
    }
}

impl GovernorConfig {
    /// Build from defaults with optional env overrides:
    /// * `KILN_MEMORY_FLOOR_GB` — safety floor in GiB.
    /// * `KILN_MEMORY_PROBE_MS` — probe TTL in milliseconds.
    pub fn from_env() -> Self {
        let mut cfg = GovernorConfig::default();
        if let Ok(v) = std::env::var("KILN_MEMORY_FLOOR_GB") {
            if let Ok(gb) = v.parse::<f64>() {
                cfg.floor_bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
            }
        }
        if let Ok(v) = std::env::var("KILN_MEMORY_PROBE_MS") {
            if let Ok(ms) = v.parse::<u64>() {
                cfg.ttl = Duration::from_millis(ms);
            }
        }
        cfg
    }
}

struct State {
    cached: MemorySnapshot,
    sampled_at: Instant,
}

/// Continuous, shared memory-awareness for the whole process.
///
/// Cheap to read (TTL-cached), backend-agnostic, and the single source every
/// dynamic-memory consumer should consult. Construct one and share it via
/// `Arc`, or use [`MemoryGovernor::global`] for the process-wide default.
pub struct MemoryGovernor {
    source: Box<dyn MemorySource>,
    cfg: GovernorConfig,
    state: Mutex<State>,
    /// Soft reservations: memory announced-but-not-yet-allocated. Summed and
    /// subtracted from `available_bytes` so two consumers can't both plan to
    /// use the same free bytes. Released via the [`Reservation`] guard.
    soft_reserved: AtomicU64,
    /// Reclaim hooks registered by the allocator layer (return pooled VRAM to
    /// the OS). Invoked under pressure by [`Self::reclaim`] / the monitor.
    reclaimers: Mutex<Vec<Reclaimer>>,
    /// Guards [`Self::start_monitor`] against spawning more than one thread.
    monitor_started: AtomicBool,
}

impl MemoryGovernor {
    /// Construct with an explicit source + config (used by tests and custom
    /// integrations).
    pub fn with_source(source: Box<dyn MemorySource>, cfg: GovernorConfig) -> Self {
        let cached = source.probe();
        MemoryGovernor {
            source,
            cfg,
            state: Mutex::new(State {
                cached,
                sampled_at: Instant::now(),
            }),
            soft_reserved: AtomicU64::new(0),
            reclaimers: Mutex::new(Vec::new()),
            monitor_started: AtomicBool::new(false),
        }
    }

    /// The default governor: OS probe + env-tuned config.
    pub fn new() -> Self {
        Self::with_source(Box::new(OsProbe), GovernorConfig::from_env())
    }

    /// The process-wide governor. Lazily initialized on first use.
    pub fn global() -> &'static MemoryGovernor {
        static GLOBAL: OnceLock<MemoryGovernor> = OnceLock::new();
        GLOBAL.get_or_init(MemoryGovernor::new)
    }

    /// Latest snapshot, re-probing only if the cached one is older than the TTL.
    pub fn snapshot(&self) -> MemorySnapshot {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.sampled_at.elapsed() >= self.cfg.ttl {
            st.cached = self.source.probe();
            st.sampled_at = Instant::now();
        }
        st.cached
    }

    /// Force a fresh probe now (bypasses the TTL). Use after a large
    /// alloc/free when the next decision needs ground truth.
    pub fn refresh(&self) -> MemorySnapshot {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.cached = self.source.probe();
        st.sampled_at = Instant::now();
        st.cached
    }

    /// Current pressure level from the live free fraction.
    pub fn pressure(&self) -> MemoryPressure {
        let s = self.snapshot();
        if s.total_bytes == 0 {
            // No device detected — can't reason about pressure; treat as
            // comfortable so we never block on a detection gap.
            return MemoryPressure::Comfortable;
        }
        let frac = s.free_bytes as f64 / s.total_bytes as f64;
        if frac <= self.cfg.critical_frac {
            MemoryPressure::Critical
        } else if frac <= self.cfg.tight_frac {
            MemoryPressure::Tight
        } else if frac < self.cfg.comfortable_frac {
            MemoryPressure::Moderate
        } else {
            MemoryPressure::Comfortable
        }
    }

    /// Bytes a new allocation may safely claim right now: live free, minus the
    /// safety floor, minus outstanding soft reservations. Saturates at 0.
    pub fn available_bytes(&self) -> u64 {
        let free = self.snapshot().free_bytes;
        let reserved = self.soft_reserved.load(Ordering::Relaxed);
        free.saturating_sub(self.cfg.floor_bytes)
            .saturating_sub(reserved)
    }

    /// Whether `bytes` fits within [`Self::available_bytes`] right now.
    pub fn can_fit(&self, bytes: u64) -> bool {
        self.available_bytes() >= bytes
    }

    /// Announce an intent to allocate `bytes` soon (training activation peak, a
    /// KV-pool grow, a graph capture). The reservation is subtracted from
    /// `available_bytes` until the returned guard drops, so concurrent
    /// consumers see an honest budget. This does NOT itself allocate.
    pub fn reserve(&self, bytes: u64) -> Reservation<'_> {
        self.soft_reserved.fetch_add(bytes, Ordering::Relaxed);
        Reservation {
            governor: self,
            bytes,
        }
    }

    /// Total outstanding soft reservations (for logging / introspection).
    pub fn soft_reserved_bytes(&self) -> u64 {
        self.soft_reserved.load(Ordering::Relaxed)
    }

    pub fn config(&self) -> &GovernorConfig {
        &self.cfg
    }

    /// Register a reclaim hook (the allocator layer's "return pooled VRAM to the
    /// OS" function). Invoked under memory pressure so kiln gives memory back to
    /// a coexisting process instead of hoarding it. Multiple hooks may register
    /// (e.g. one per device pool); they're called in registration order.
    pub fn register_reclaimer<F: Fn(u64) -> u64 + Send + Sync + 'static>(&self, f: F) {
        self.reclaimers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Box::new(f));
    }

    /// Invoke registered reclaimers to free up to `target_bytes`, then re-probe.
    /// Returns total bytes freed (best-effort; a hook may report 0 if it can't
    /// measure). A no-op with no reclaimers registered.
    pub fn reclaim(&self, target_bytes: u64) -> u64 {
        let mut freed = 0u64;
        {
            let hooks = self.reclaimers.lock().unwrap_or_else(|e| e.into_inner());
            for hook in hooks.iter() {
                if freed >= target_bytes {
                    break;
                }
                freed = freed.saturating_add(hook(target_bytes.saturating_sub(freed)));
            }
        }
        if freed > 0 {
            self.refresh(); // ground truth after returning memory to the OS
        }
        freed
    }

    /// Reclaim only if under memory pressure (Tight/Critical) — the policy the
    /// background monitor applies. Targets enough to climb back to the
    /// comfortable free fraction. Returns bytes freed (0 if not needed).
    pub fn maybe_reclaim(&self) -> u64 {
        if !self.pressure().should_reclaim() {
            return 0;
        }
        let s = self.snapshot();
        let want_free = ((s.total_bytes as f64) * self.cfg.comfortable_frac) as u64;
        let target = want_free.saturating_sub(s.free_bytes).max(1);
        self.reclaim(target)
    }

    /// Spawn a background thread that watches pressure and auto-reclaims, turning
    /// the one-shot probe into *continuous* self-adjustment: if a coexisting job
    /// (or kiln itself) drives memory tight, kiln returns pooled VRAM to the OS
    /// without anyone asking. Idempotent — starts at most one thread. Requires a
    /// `'static` governor (use [`MemoryGovernor::global`]).
    pub fn start_monitor(&'static self) {
        if self.monitor_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let interval = self.cfg.ttl.max(Duration::from_secs(2));
        std::thread::Builder::new()
            .name("kiln-mem-governor".into())
            .spawn(move || loop {
                std::thread::sleep(interval);
                let pressure = self.pressure();
                if pressure.should_reclaim() {
                    let freed = self.maybe_reclaim();
                    let s = self.snapshot();
                    tracing::info!(
                        ?pressure,
                        freed_mb = freed / (1024 * 1024),
                        free_gb = s.free_bytes as f64 / 1e9,
                        total_gb = s.total_bytes as f64 / 1e9,
                        "memory governor: reclaimed under pressure"
                    );
                }
            })
            .ok();
    }
}

impl Default for MemoryGovernor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryGovernor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGovernor")
            .field("cfg", &self.cfg)
            .field("soft_reserved", &self.soft_reserved_bytes())
            .finish_non_exhaustive()
    }
}

/// RAII guard for a soft reservation made via [`MemoryGovernor::reserve`].
/// Releases the reservation (decrements the governor's outstanding total) on
/// drop, so a consumer that plans an allocation and then either completes it
/// (the memory now shows up in the live probe) or abandons it both end with a
/// correct budget.
pub struct Reservation<'a> {
    governor: &'a MemoryGovernor,
    bytes: u64,
}

impl Reservation<'_> {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.governor
            .soft_reserved
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vram::VramSource;

    /// A fixed, injectable snapshot source for deterministic tests.
    struct Fixed(std::sync::Mutex<MemorySnapshot>);
    impl Fixed {
        fn new(total: u64, free: u64) -> Self {
            Fixed(std::sync::Mutex::new(snap(total, free)))
        }
        fn set_free(&self, free: u64) {
            let mut s = self.0.lock().unwrap();
            s.free_bytes = free;
            s.used_bytes = s.total_bytes.saturating_sub(free);
        }
    }
    impl MemorySource for Fixed {
        fn probe(&self) -> MemorySnapshot {
            *self.0.lock().unwrap()
        }
    }

    fn snap(total: u64, free: u64) -> MemorySnapshot {
        MemorySnapshot {
            total_bytes: total,
            used_bytes: total.saturating_sub(free),
            free_bytes: free,
            source: VramSource::NvidiaSmi,
            unified: false,
        }
    }

    const GB: u64 = 1024 * 1024 * 1024;

    fn gov(total: u64, free: u64) -> MemoryGovernor {
        MemoryGovernor::with_source(Box::new(Fixed::new(total, free)), GovernorConfig::default())
    }

    #[test]
    fn pressure_tracks_free_fraction() {
        // 24 GiB total, default thresholds (crit 5%, tight 10%, comfy 25%).
        assert_eq!(gov(24 * GB, 12 * GB).pressure(), MemoryPressure::Comfortable);
        assert_eq!(gov(24 * GB, 4 * GB).pressure(), MemoryPressure::Moderate); // ~17%
        assert_eq!(gov(24 * GB, 2 * GB).pressure(), MemoryPressure::Tight); // ~8%
        assert_eq!(gov(24 * GB, GB).pressure(), MemoryPressure::Critical); // ~4%
    }

    #[test]
    fn no_device_is_comfortable_not_a_panic() {
        assert_eq!(gov(0, 0).pressure(), MemoryPressure::Comfortable);
    }

    #[test]
    fn available_subtracts_floor() {
        // 16 GiB free, 1 GiB floor -> 15 GiB available.
        let g = gov(24 * GB, 16 * GB);
        assert_eq!(g.available_bytes(), 15 * GB);
        assert!(g.can_fit(15 * GB));
        assert!(!g.can_fit(15 * GB + 1));
    }

    #[test]
    fn soft_reservation_reduces_available_then_releases() {
        let g = gov(24 * GB, 16 * GB);
        assert_eq!(g.available_bytes(), 15 * GB);
        {
            let _r = g.reserve(10 * GB);
            assert_eq!(g.soft_reserved_bytes(), 10 * GB);
            assert_eq!(g.available_bytes(), 5 * GB);
            assert!(!g.can_fit(6 * GB));
        }
        // Guard dropped -> reservation released.
        assert_eq!(g.soft_reserved_bytes(), 0);
        assert_eq!(g.available_bytes(), 15 * GB);
    }

    /// Manual on-hardware smoke (`cargo test -p kiln-memory --
    /// --ignored --nocapture live_governor_smoke`): prints the REAL governor
    /// state on this machine, including whatever other processes are using the
    /// device. Ignored by default (hardware-dependent, no fixed assertion).
    #[test]
    #[ignore]
    fn live_governor_smoke() {
        let g = MemoryGovernor::new();
        let s = g.snapshot();
        eprintln!(
            "LIVE: total={:.1}GB used={:.1}GB free={:.1}GB unified={} source={} \
             | pressure={:?} available(after floor)={:.1}GB",
            s.total_bytes as f64 / 1e9,
            s.used_bytes as f64 / 1e9,
            s.free_bytes as f64 / 1e9,
            s.unified,
            s.source,
            g.pressure(),
            g.available_bytes() as f64 / 1e9,
        );
    }

    #[test]
    fn coexisting_process_reduces_our_budget() {
        // The probe is SYSTEM-WIDE (nvidia-smi memory.used / DRM mem_info_vram_used
        // / unified MemAvailable all count every process), so a neighbour using
        // the GPU shrinks what kiln may claim — we never assume we own the device.
        // Start with 20 GiB free on a 24 GiB device.
        let src = std::sync::Arc::new(Fixed::new(24 * GB, 20 * GB));
        struct Shared(std::sync::Arc<Fixed>);
        impl MemorySource for Shared {
            fn probe(&self) -> MemorySnapshot {
                self.0.probe()
            }
        }
        let cfg = GovernorConfig {
            ttl: Duration::from_millis(0),
            ..GovernorConfig::default()
        };
        let g = MemoryGovernor::with_source(Box::new(Shared(src.clone())), cfg);
        assert_eq!(g.available_bytes(), 19 * GB); // 20 free - 1 floor
        assert_eq!(g.pressure(), MemoryPressure::Comfortable);

        // A neighbouring process grabs 18 GiB of VRAM -> only 2 GiB free now.
        src.set_free(2 * GB);
        // Our available budget collapses accordingly (continuous awareness, not
        // a startup snapshot), and pressure rises so the system reclaims.
        assert_eq!(g.available_bytes(), GB); // 2 free - 1 floor
        assert_eq!(g.pressure(), MemoryPressure::Tight); // 2/24 = 8.3% <= 10%
        assert!(g.pressure().should_reclaim());
        assert!(!g.can_fit(5 * GB));
    }

    #[test]
    fn reclaim_invokes_hooks_under_pressure_and_recovers() {
        let src = std::sync::Arc::new(Fixed::new(24 * GB, GB)); // 1/24 ≈ 4% -> Critical
        struct Shared(std::sync::Arc<Fixed>);
        impl MemorySource for Shared {
            fn probe(&self) -> MemorySnapshot {
                self.0.probe()
            }
        }
        let cfg = GovernorConfig {
            ttl: Duration::from_millis(0),
            ..GovernorConfig::default()
        };
        let g = MemoryGovernor::with_source(Box::new(Shared(src.clone())), cfg);
        assert!(g.pressure().should_reclaim());

        // A reclaimer that "returns memory to the OS" — modelled as free jumping
        // back up — and reports the bytes it freed.
        let calls = std::sync::Arc::new(AtomicU64::new(0));
        let (src2, calls2) = (src.clone(), calls.clone());
        g.register_reclaimer(move |target| {
            calls2.fetch_add(1, Ordering::SeqCst);
            src2.set_free(12 * GB);
            target
        });

        let freed = g.maybe_reclaim();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "reclaimer must be invoked");
        assert!(freed > 0);
        // The post-reclaim re-probe sees the freed memory -> back to comfortable.
        assert_eq!(g.pressure(), MemoryPressure::Comfortable);

        // No pressure -> reclaimer NOT called again.
        let freed2 = g.maybe_reclaim();
        assert_eq!(freed2, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn snapshot_refresh_picks_up_changes() {
        let src = std::sync::Arc::new(Fixed::new(24 * GB, 16 * GB));
        // ttl=0 so every read re-probes.
        let cfg = GovernorConfig {
            ttl: Duration::from_millis(0),
            ..GovernorConfig::default()
        };
        struct Shared(std::sync::Arc<Fixed>);
        impl MemorySource for Shared {
            fn probe(&self) -> MemorySnapshot {
                self.0.probe()
            }
        }
        let g = MemoryGovernor::with_source(Box::new(Shared(src.clone())), cfg);
        assert_eq!(g.snapshot().free_bytes, 16 * GB);
        src.set_free(4 * GB);
        assert_eq!(g.snapshot().free_bytes, 4 * GB);
        assert_eq!(g.pressure(), MemoryPressure::Moderate);
    }
}
