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
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A reclaim hook: "release up to `target` bytes of pooled/cached device memory
/// back to the OS; return how much you freed (0 if unknown/none)." Registered by
/// the allocator layer; invoked by the governor under memory pressure.
pub type Reclaimer = Box<dyn Fn(u64) -> u64 + Send + Sync>;

use crate::vram::{
    MemorySnapshot, VramProbeSelector, current_memory_snapshot_for, try_current_memory_snapshot_for,
};

/// Source of [`MemorySnapshot`]s. Abstracted so tests can drive the governor
/// with synthetic memory states (no GPU required) and so an integration could
/// later swap in a device-API probe (`hipMemGetInfo`/`cudaMemGetInfo`) without
/// touching the governor itself.
pub trait MemorySource: Send + Sync {
    fn probe(&self) -> MemorySnapshot;

    /// Fallible form for sources that can distinguish a failed observation
    /// from a real zero-capacity snapshot.
    fn try_probe(&self) -> Option<MemorySnapshot> {
        Some(self.probe())
    }
}

/// Default source: the OS-level [`current_memory_snapshot`].
#[derive(Debug, Clone, Copy)]
pub struct OsProbe {
    selector: VramProbeSelector,
}

impl OsProbe {
    pub fn new(selector: VramProbeSelector) -> Self {
        Self { selector }
    }

    pub fn selector(self) -> VramProbeSelector {
        self.selector
    }
}

impl Default for OsProbe {
    fn default() -> Self {
        Self::new(VramProbeSelector::Auto)
    }
}

impl MemorySource for OsProbe {
    fn probe(&self) -> MemorySnapshot {
        current_memory_snapshot_for(self.selector)
    }

    fn try_probe(&self) -> Option<MemorySnapshot> {
        try_current_memory_snapshot_for(self.selector)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalGovernorConfiguration {
    pub selector: VramProbeSelector,
    pub governor: GovernorConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GlobalGovernorConfigurationError {
    AlreadyConfigured {
        existing: GlobalGovernorConfiguration,
        requested: GlobalGovernorConfiguration,
    },
    AlreadyInitialized {
        existing: GlobalGovernorConfiguration,
        requested: GlobalGovernorConfiguration,
    },
}

impl std::fmt::Display for GlobalGovernorConfigurationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyConfigured {
                existing,
                requested,
            } => write!(
                f,
                "global memory governor already configured as {existing:?}; cannot change to {requested:?}"
            ),
            Self::AlreadyInitialized {
                existing,
                requested,
            } => write!(
                f,
                "global memory governor initialized with {existing:?}; cannot change to {requested:?}"
            ),
        }
    }
}

impl std::error::Error for GlobalGovernorConfigurationError {}

#[derive(Debug, Clone, Copy)]
struct GlobalGovernorState {
    configured: Option<GlobalGovernorConfiguration>,
    initialized: Option<GlobalGovernorConfiguration>,
}

impl GlobalGovernorState {
    const fn new() -> Self {
        Self {
            configured: None,
            initialized: None,
        }
    }

    fn configure(
        &mut self,
        requested: GlobalGovernorConfiguration,
    ) -> Result<(), GlobalGovernorConfigurationError> {
        if let Some(existing) = self.initialized {
            return if existing == requested {
                Ok(())
            } else {
                Err(GlobalGovernorConfigurationError::AlreadyInitialized {
                    existing,
                    requested,
                })
            };
        }
        if let Some(existing) = self.configured {
            if existing != requested {
                return Err(GlobalGovernorConfigurationError::AlreadyConfigured {
                    existing,
                    requested,
                });
            }
            return Ok(());
        }
        self.configured = Some(requested);
        Ok(())
    }

    fn initialize(&mut self) -> GlobalGovernorConfiguration {
        let configuration = self.configured.unwrap_or(GlobalGovernorConfiguration {
            selector: VramProbeSelector::Auto,
            governor: GovernorConfig::default_const(),
        });
        self.initialized = Some(configuration);
        configuration
    }
}

static GLOBAL_GOVERNOR_STATE: Mutex<GlobalGovernorState> = Mutex::new(GlobalGovernorState::new());
static GLOBAL_GOVERNOR: OnceLock<MemoryGovernor> = OnceLock::new();

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

/// One probe-free, internally coherent view of the governor's published state.
///
/// `available_bytes` is derived from `snapshot` and `soft_reserved_bytes` in
/// this value, rather than loading the reservation counter a second time. A
/// caller therefore never observes an available budget from one reservation
/// generation alongside a reservation total from another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryGovernorObservation {
    pub snapshot: MemorySnapshot,
    pub available_bytes: u64,
    pub soft_reserved_bytes: u64,
    pub pressure: MemoryPressure,
    pub sample_status: CachedSampleStatus,
}

/// Controls whether memory observation may mutate backend allocator state.
/// `Off` is the stable default because a reclaim hook may synchronize a live
/// accelerator. The other modes require an explicit operator choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryReclaimMode {
    Off,
    OnDemand,
    Automatic,
}

impl MemoryReclaimMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "on-demand" => Ok(Self::OnDemand),
            "automatic" => Ok(Self::Automatic),
            _ => Err(format!(
                "memory.reclaim_mode must be one of off, on-demand, automatic; got {value:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::OnDemand => "on-demand",
            Self::Automatic => "automatic",
        }
    }
}

impl MemoryPressure {
    /// True once the system should start *releasing* memory (Tight or worse).
    pub fn should_reclaim(self) -> bool {
        matches!(self, MemoryPressure::Tight | MemoryPressure::Critical)
    }
}

const AUTOMATIC_RECLAIM_MIN_BACKOFF: Duration = Duration::from_secs(2);
const AUTOMATIC_RECLAIM_SUCCESS_COOLDOWN: Duration = Duration::from_secs(8);
const AUTOMATIC_RECLAIM_MAX_BACKOFF: Duration = Duration::from_secs(128);
/// Cached admission must not trust an observation indefinitely if a sampler is
/// stalled or dead. Four normal sample intervals tolerate scheduler jitter and
/// the bounded two-second NVIDIA probe; five seconds is the minimum deadline
/// for faster sysfs-backed configurations.
const MIN_CACHED_SAMPLE_MAX_AGE: Duration = Duration::from_secs(5);
const CACHED_SAMPLE_TTL_MULTIPLIER: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutomaticReclaimScheduleUpdate {
    retry_after: Duration,
    zero_yield_streak: u32,
    report_zero_yield: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutomaticReclaimDecision {
    Idle,
    BackingOff,
    Attempt,
}

/// Per-pressure-episode retry state for the automatic monitor. Tight or
/// Critical pressure arms an episode; Moderate remains armed so reclaim does
/// not chatter at the trigger threshold, and Comfortable is the sole reset.
#[derive(Debug)]
struct AutomaticReclaimSchedule {
    armed: bool,
    next_attempt_at: Option<Instant>,
    zero_yield_backoff: Duration,
    zero_yield_streak: u32,
}

impl AutomaticReclaimSchedule {
    fn new() -> Self {
        Self {
            armed: false,
            next_attempt_at: None,
            zero_yield_backoff: AUTOMATIC_RECLAIM_MIN_BACKOFF,
            zero_yield_streak: 0,
        }
    }

    fn decision(&mut self, pressure: MemoryPressure, now: Instant) -> AutomaticReclaimDecision {
        if pressure == MemoryPressure::Comfortable {
            self.reset();
            return AutomaticReclaimDecision::Idle;
        }
        if !self.armed {
            if !pressure.should_reclaim() {
                return AutomaticReclaimDecision::Idle;
            }
            self.armed = true;
        }
        if self
            .next_attempt_at
            .is_some_and(|next_attempt_at| now < next_attempt_at)
        {
            AutomaticReclaimDecision::BackingOff
        } else {
            AutomaticReclaimDecision::Attempt
        }
    }

    fn record_result(
        &mut self,
        now: Instant,
        reclaimed_bytes: u64,
    ) -> AutomaticReclaimScheduleUpdate {
        let (retry_after, report_zero_yield) = if reclaimed_bytes > 0 {
            self.zero_yield_backoff = AUTOMATIC_RECLAIM_MIN_BACKOFF;
            self.zero_yield_streak = 0;
            (AUTOMATIC_RECLAIM_SUCCESS_COOLDOWN, false)
        } else {
            self.zero_yield_streak = self.zero_yield_streak.saturating_add(1);
            let retry_after = self.zero_yield_backoff;
            self.zero_yield_backoff = self
                .zero_yield_backoff
                .saturating_mul(2)
                .min(AUTOMATIC_RECLAIM_MAX_BACKOFF);
            (retry_after, self.zero_yield_streak == 1)
        };
        self.next_attempt_at = Some(now + retry_after);
        AutomaticReclaimScheduleUpdate {
            retry_after,
            zero_yield_streak: self.zero_yield_streak,
            report_zero_yield,
        }
    }

    fn reset(&mut self) {
        self.armed = false;
        self.next_attempt_at = None;
        self.zero_yield_backoff = AUTOMATIC_RECLAIM_MIN_BACKOFF;
        self.zero_yield_streak = 0;
    }
}

/// Typed governor tuning. Defaults are conservative; startup configuration is
/// injected once through [`MemoryGovernor::configure_global`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorConfig {
    /// Minimum interval between OS probes; reads inside this window return the
    /// cached snapshot. Keeps `nvidia-smi`/sysfs probing off the hot path.
    pub ttl: Duration,
    /// Bytes never handed out — headroom left for the OS, other apps, and
    /// allocator fragmentation slack. `available_bytes` subtracts this.
    pub floor_bytes: u64,
    /// Optional process-level capacity ceiling resolved from typed startup
    /// configuration. `Some(0)` is an intentional fail-closed limit.
    pub capacity_limit_bytes: Option<u64>,
    /// `free/total` at/below which pressure is [`MemoryPressure::Tight`].
    pub tight_frac: f64,
    /// `free/total` at/below which pressure is [`MemoryPressure::Critical`].
    pub critical_frac: f64,
    /// `free/total` at/above which pressure is [`MemoryPressure::Comfortable`].
    pub comfortable_frac: f64,
    /// Whether backend reclaim hooks are disabled, on-demand, or periodic.
    pub reclaim_mode: MemoryReclaimMode,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self::default_const()
    }
}

impl GovernorConfig {
    pub const fn default_const() -> Self {
        Self {
            ttl: Duration::from_millis(500),
            floor_bytes: 1024 * 1024 * 1024, // 1 GiB
            capacity_limit_bytes: None,
            critical_frac: 0.05,
            tight_frac: 0.10,
            comfortable_frac: 0.25,
            reclaim_mode: MemoryReclaimMode::Off,
        }
    }
}

struct State {
    cached: MemorySnapshot,
    sampled_at: Instant,
}

/// Liveness and freshness of the observation backing cached admission.
///
/// Cached readers never perform driver or operating-system I/O. Once
/// `healthy` is false, [`MemoryGovernor::cached_snapshot`] retains capacity and
/// raw diagnostics but publishes zero free bytes so availability, pressure,
/// and checked reservations all fail closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedSampleStatus {
    pub age: Duration,
    pub max_age: Duration,
    pub stale: bool,
    pub sampler_required: bool,
    pub sampler_running: bool,
    pub healthy: bool,
}

impl Default for CachedSampleStatus {
    fn default() -> Self {
        Self {
            age: Duration::ZERO,
            max_age: Duration::ZERO,
            stale: true,
            sampler_required: false,
            sampler_running: false,
            healthy: false,
        }
    }
}

/// Bounded process-lifetime observability for the opt-in automatic reclaim
/// monitor. Counters describe completed decisions; the last-attempt fields are
/// overwritten rather than retaining an unbounded event history.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AutomaticReclaimStats {
    pub attempts: u64,
    pub successful_attempts: u64,
    pub zero_yield_attempts: u64,
    pub suppressed_attempts: u64,
    pub reclaimed_bytes: u64,
    pub last_target_bytes: u64,
    pub last_reclaimed_bytes: u64,
    pub last_duration_us: u64,
    pub retry_after_ms: u64,
    pub zero_yield_streak: u64,
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
    /// Serializes potentially slow source probes without blocking readers of
    /// the last published snapshot.
    probe_lock: Mutex<()>,
    /// Soft reservations: memory announced-but-not-yet-allocated. Summed and
    /// subtracted from `available_bytes` so two consumers can't both plan to
    /// use the same free bytes. Released via the [`Reservation`] guard.
    soft_reserved: AtomicU64,
    /// Number of live unconditional reservations whose requested debt did not
    /// fit completely in `soft_reserved`. While nonzero, budget readers report
    /// `u64::MAX` reserved and checked reservations fail closed. Each guard
    /// still remembers its represented portion, so arbitrary drop order cannot
    /// expose headroom while an overflowed whole-operation debt remains live.
    soft_reservation_overflow_debts: AtomicU64,
    /// Seqlock-style reservation observation metadata. Even values are stable;
    /// an odd value means one writer is updating the represented and overflow
    /// debt counters. Readers retry if the generation changes during capture.
    reservation_generation: AtomicU64,
    /// Reclaim hooks registered by the allocator layer (return pooled VRAM to
    /// the OS). Invoked under pressure by [`Self::reclaim`] / the monitor.
    reclaimers: Mutex<Vec<Reclaimer>>,
    /// Guards [`Self::start_monitor`] against spawning more than one thread.
    monitor_started: AtomicBool,
    /// Guards the probe-only publisher used by hot-path readers. This sampler
    /// never reclaims or allocates; it performs driver/OS I/O off request and
    /// inference threads and publishes into `state.cached`.
    sampler_started: AtomicBool,
    /// Set once cached admission has been delegated to the background sampler.
    /// A stopped worker then fails cached readers closed immediately rather
    /// than leaving the last successful free-memory figure live indefinitely.
    sampler_required: AtomicBool,
    /// Current worker liveness. A lifecycle guard clears this, and
    /// `sampler_started`, if the worker ever exits unexpectedly.
    sampler_running: AtomicBool,
    automatic_reclaim_stats: Mutex<AutomaticReclaimStats>,
    /// Event wake-up for budget-change consumers (the KV autoscaler). The
    /// `u64` is a monotonically-increasing generation bumped on every
    /// [`notify_change`](Self::notify_change); waiters block on the `Condvar`
    /// until the generation advances or their timeout fires. This makes
    /// grow-back EVENT-DRIVEN: when a training reservation drops (job ends),
    /// the autoscaler wakes immediately to reclaim KV instead of waiting out
    /// its poll tick. The timeout still backstops EXTERNAL changes (a
    /// coexisting process freeing VRAM) that only the periodic probe sees.
    wake: (Mutex<u64>, Condvar),
}

/// Clears worker liveness even if the sampler thread unwinds outside the
/// per-probe panic boundary. Resetting `sampler_started` permits an explicit
/// later restart instead of permanently reporting a dead worker as started.
struct SamplerLifecycleGuard<'a> {
    governor: &'a MemoryGovernor,
}

struct ReservationUpdateGuard<'a> {
    governor: &'a MemoryGovernor,
    stable_generation: u64,
}

impl<'a> ReservationUpdateGuard<'a> {
    fn new(governor: &'a MemoryGovernor) -> Self {
        loop {
            let generation = governor.reservation_generation.load(Ordering::Acquire);
            if generation & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            if governor
                .reservation_generation
                .compare_exchange_weak(
                    generation,
                    generation.wrapping_add(1),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Self {
                    governor,
                    stable_generation: generation,
                };
            }
        }
    }
}

impl Drop for ReservationUpdateGuard<'_> {
    fn drop(&mut self) {
        self.governor
            .reservation_generation
            .store(self.stable_generation.wrapping_add(2), Ordering::Release);
    }
}

impl Drop for SamplerLifecycleGuard<'_> {
    fn drop(&mut self) {
        self.governor
            .sampler_running
            .store(false, Ordering::Release);
        self.governor
            .sampler_started
            .store(false, Ordering::Release);
        self.governor.notify_change();
    }
}

impl MemoryGovernor {
    /// Construct with an explicit source + config (used by tests and custom
    /// integrations).
    pub fn with_source(source: Box<dyn MemorySource>, cfg: GovernorConfig) -> Self {
        let cached = source.try_probe().unwrap_or_else(|| {
            let mut failed = current_memory_snapshot_for(VramProbeSelector::None);
            failed.observations.probe_failed = true;
            failed
        });
        MemoryGovernor {
            source,
            cfg,
            state: Mutex::new(State {
                cached,
                sampled_at: Instant::now(),
            }),
            probe_lock: Mutex::new(()),
            soft_reserved: AtomicU64::new(0),
            soft_reservation_overflow_debts: AtomicU64::new(0),
            reservation_generation: AtomicU64::new(0),
            reclaimers: Mutex::new(Vec::new()),
            monitor_started: AtomicBool::new(false),
            sampler_started: AtomicBool::new(false),
            sampler_required: AtomicBool::new(false),
            sampler_running: AtomicBool::new(false),
            automatic_reclaim_stats: Mutex::new(AutomaticReclaimStats::default()),
            wake: (Mutex::new(0), Condvar::new()),
        }
    }

    /// Wake any consumers blocked in [`Self::wait_for_change`] — the budget or
    /// pressure just changed (a reservation taken/released, or the monitor saw a
    /// pressure transition). Cheap; safe to over-call (waiters re-evaluate).
    pub fn notify_change(&self) {
        {
            let mut wake_gen = self.wake.0.lock().unwrap_or_else(|e| e.into_inner());
            *wake_gen = wake_gen.wrapping_add(1);
        }
        self.wake.1.notify_all();
    }

    /// Block until the memory budget/pressure changes (via [`Self::notify_change`])
    /// or `timeout` elapses, whichever comes first. The KV autoscaler uses this
    /// in place of a fixed sleep so it reacts PROMPTLY to a training job ending
    /// (reservation drop → grow KV back) while still polling every `timeout` for
    /// external changes the event path can't see.
    pub fn wait_for_change(&self, timeout: Duration) {
        let wake_gen = self.wake.0.lock().unwrap_or_else(|e| e.into_inner());
        let start = *wake_gen;
        // wait_timeout_while re-checks the predicate across spurious wakeups.
        // The returned guard drops at the end of this scope (we don't read it).
        let (_guard, _timed_out) = self
            .wake
            .1
            .wait_timeout_while(wake_gen, timeout, |g| *g == start)
            .unwrap_or_else(|e| e.into_inner());
    }

    pub fn with_selector(selector: VramProbeSelector, cfg: GovernorConfig) -> Self {
        Self::with_source(Box::new(OsProbe::new(selector)), cfg)
    }

    /// Configure the process-wide live probe and policy. This must run before
    /// the first call to [`Self::global`]; repeated identical configuration is
    /// idempotent, including after initialization.
    pub fn configure_global(
        selector: VramProbeSelector,
        governor: GovernorConfig,
    ) -> Result<(), GlobalGovernorConfigurationError> {
        GLOBAL_GOVERNOR_STATE
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .configure(GlobalGovernorConfiguration { selector, governor })
    }

    /// Current configured/initialized global policy, or conservative defaults.
    pub fn global_configuration() -> GlobalGovernorConfiguration {
        let state = GLOBAL_GOVERNOR_STATE
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state
            .initialized
            .or(state.configured)
            .unwrap_or(GlobalGovernorConfiguration {
                selector: VramProbeSelector::Auto,
                governor: GovernorConfig::default_const(),
            })
    }

    /// A standalone governor using the configured policy or typed defaults.
    pub fn new() -> Self {
        let configuration = Self::global_configuration();
        Self::with_selector(configuration.selector, configuration.governor)
    }

    /// The process-wide governor. Lazily initialized on first use.
    pub fn global() -> &'static MemoryGovernor {
        GLOBAL_GOVERNOR.get_or_init(|| {
            let configuration = GLOBAL_GOVERNOR_STATE
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .initialize();
            Self::with_selector(configuration.selector, configuration.governor)
        })
    }

    /// Last process-wide snapshot if the governor has already been initialized.
    /// Unlike [`Self::global`], this never initializes a probe and therefore
    /// cannot perform synchronous driver or OS I/O on the caller.
    pub fn try_global_cached_snapshot() -> Option<MemorySnapshot> {
        GLOBAL_GOVERNOR
            .get()
            .map(|governor| governor.cached_snapshot())
    }

    /// One coherent process-wide observation, if the governor is initialized.
    /// This never initializes or probes the global source.
    pub fn try_global_cached_observation() -> Option<MemoryGovernorObservation> {
        GLOBAL_GOVERNOR
            .get()
            .map(|governor| governor.cached_observation())
    }

    /// Last process-wide admissible allocation budget, if initialized.
    ///
    /// This is the hot-path counterpart to [`Self::available_bytes`]: it uses
    /// the sampler-published snapshot and subtracts the safety floor and all
    /// outstanding soft reservations without performing a probe.
    pub fn try_global_cached_available_bytes() -> Option<u64> {
        GLOBAL_GOVERNOR
            .get()
            .map(|governor| governor.cached_available_bytes())
    }

    /// Atomically reserve bytes from the last sampler-published global budget.
    ///
    /// Unlike [`Self::reserve`], this fails when the request would exceed live
    /// free memory after the safety floor and existing reservations. The guard
    /// keeps concurrent operation planners from authorizing the same bytes.
    pub fn try_global_cached_reserve(bytes: u64) -> Option<Reservation<'static>> {
        GLOBAL_GOVERNOR.get()?.try_reserve_cached(bytes)
    }

    /// Last process-wide pressure state without initializing or probing.
    pub fn try_global_cached_pressure() -> Option<MemoryPressure> {
        GLOBAL_GOVERNOR.get().map(|governor| {
            let snapshot = governor.cached_snapshot();
            governor.pressure_for_snapshot(snapshot)
        })
    }

    /// Latest snapshot, re-probing only if the cached one is older than the TTL.
    pub fn snapshot(&self) -> MemorySnapshot {
        {
            let st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if st.sampled_at.elapsed() < self.cfg.ttl {
                return self.apply_capacity_limit(st.cached);
            }
        }
        self.refresh()
    }

    /// Return the last published snapshot without performing OS or driver I/O.
    /// Request handlers and inference hot paths must use this form; explicit
    /// refresh ownership belongs at startup or in the background monitor. A
    /// stale sample or stopped required sampler retains its capacity and raw
    /// diagnostics but is projected as fully used so admission fails closed.
    pub fn cached_snapshot(&self) -> MemorySnapshot {
        let st = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let status = self.cached_sample_status_for(&st);
        self.cached_snapshot_for_status(st.cached, status)
    }

    /// Capture the published snapshot and derive every reservation-sensitive
    /// field from one atomic reservation-counter load. This performs no probe.
    pub fn cached_observation(&self) -> MemoryGovernorObservation {
        let st = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let sample_status = self.cached_sample_status_for(&st);
        let snapshot = self.cached_snapshot_for_status(st.cached, sample_status);
        let soft_reserved_bytes = self.soft_reserved_bytes_for_budget();
        MemoryGovernorObservation {
            snapshot,
            available_bytes: snapshot
                .free_bytes
                .saturating_sub(self.cfg.floor_bytes)
                .saturating_sub(soft_reserved_bytes),
            soft_reserved_bytes,
            pressure: self.pressure_for_snapshot(snapshot),
            sample_status,
        }
    }

    /// Freshness and worker liveness for the last cached observation. This is a
    /// lock-only diagnostic and never probes the driver or operating system.
    pub fn cached_sample_status(&self) -> CachedSampleStatus {
        let st = self.state.lock().unwrap_or_else(|error| error.into_inner());
        self.cached_sample_status_for(&st)
    }

    /// Bytes a hot-path allocation may claim from the last published sample.
    /// This never performs OS or driver I/O.
    pub fn cached_available_bytes(&self) -> u64 {
        self.cached_observation().available_bytes
    }

    /// Force a fresh probe now (bypasses the TTL). Use after a large
    /// alloc/free when the next decision needs ground truth.
    pub fn refresh(&self) -> MemorySnapshot {
        let _probe_guard = self
            .probe_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let snapshot = self.source.try_probe();
        self.publish_probe_result(snapshot)
    }

    fn publish_probe_result(&self, snapshot: Option<MemorySnapshot>) -> MemorySnapshot {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        st.cached = match snapshot {
            Some(snapshot) => snapshot,
            None => {
                let mut failed = st.cached;
                failed.used_bytes = failed.total_bytes;
                failed.free_bytes = 0;
                failed.observations.probe_failed = true;
                failed
            }
        };
        st.sampled_at = Instant::now();
        self.apply_capacity_limit(st.cached)
    }

    fn publish_sampler_panic(&self) -> MemorySnapshot {
        self.publish_probe_result(None)
    }

    fn sampler_refresh_once(&self) -> MemorySnapshot {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.refresh())) {
            Ok(snapshot) => snapshot,
            Err(payload) => {
                let panic = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("non-string panic payload");
                tracing::error!(
                    panic,
                    "memory sampler probe panicked; publishing zero-free failure and continuing"
                );
                self.publish_sampler_panic()
            }
        }
    }

    fn cached_sample_max_age(&self) -> Duration {
        self.cfg
            .ttl
            .saturating_mul(CACHED_SAMPLE_TTL_MULTIPLIER)
            .max(MIN_CACHED_SAMPLE_MAX_AGE)
    }

    fn cached_sample_status_for(&self, state: &State) -> CachedSampleStatus {
        let age = state.sampled_at.elapsed();
        let max_age = self.cached_sample_max_age();
        let stale = age > max_age;
        let sampler_required = self.sampler_required.load(Ordering::Acquire);
        let sampler_running = self.sampler_running.load(Ordering::Acquire);
        let healthy = !state.cached.observations.probe_failed
            && !stale
            && (!sampler_required || sampler_running);
        CachedSampleStatus {
            age,
            max_age,
            stale,
            sampler_required,
            sampler_running,
            healthy,
        }
    }

    fn cached_snapshot_for_status(
        &self,
        snapshot: MemorySnapshot,
        status: CachedSampleStatus,
    ) -> MemorySnapshot {
        let mut snapshot = self.apply_capacity_limit(snapshot);
        if !status.healthy {
            snapshot.used_bytes = snapshot.total_bytes;
            snapshot.free_bytes = 0;
        }
        snapshot
    }

    fn apply_capacity_limit(&self, mut snapshot: MemorySnapshot) -> MemorySnapshot {
        let Some(limit) = self.cfg.capacity_limit_bytes else {
            return snapshot;
        };
        snapshot.total_bytes = snapshot.total_bytes.min(limit);
        snapshot.used_bytes = snapshot.used_bytes.min(snapshot.total_bytes);
        snapshot.free_bytes = snapshot
            .free_bytes
            .min(snapshot.total_bytes.saturating_sub(snapshot.used_bytes));
        snapshot
    }

    /// Current pressure level from the live free fraction.
    pub fn pressure(&self) -> MemoryPressure {
        self.pressure_for_snapshot(self.snapshot())
    }

    /// Derive pressure from an already-selected snapshot without probing.
    pub fn pressure_for_snapshot(&self, s: MemorySnapshot) -> MemoryPressure {
        if s.total_bytes == 0 {
            // A selected probe that cannot establish capacity must fail
            // closed. `available_bytes` is already zero; Critical also stops
            // pressure-only consumers from growing through a detection gap.
            return MemoryPressure::Critical;
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

    fn reclaim_target_to_comfortable(&self, s: MemorySnapshot) -> u64 {
        let want_free = ((s.total_bytes as f64) * self.cfg.comfortable_frac) as u64;
        want_free.saturating_sub(s.free_bytes).max(1)
    }

    /// Bytes a new allocation may safely claim right now: live free, minus the
    /// safety floor, minus outstanding soft reservations. Saturates at 0.
    pub fn available_bytes(&self) -> u64 {
        self.available_bytes_for_snapshot(self.snapshot())
    }

    /// Derive an admissible allocation budget from an existing snapshot.
    /// The live safety floor and outstanding soft reservations are applied.
    pub fn available_bytes_for_snapshot(&self, snapshot: MemorySnapshot) -> u64 {
        let reserved = self.soft_reserved_bytes_for_budget();
        snapshot
            .free_bytes
            .saturating_sub(self.cfg.floor_bytes)
            .saturating_sub(reserved)
    }

    /// Whether `bytes` fits within [`Self::available_bytes`] right now.
    pub fn can_fit(&self, bytes: u64) -> bool {
        self.available_bytes() >= bytes
    }

    /// Announce unconditional debt for the whole lifetime of an operation (for
    /// example, a training activation peak). The reservation is subtracted
    /// from `available_bytes` until the returned guard drops. Unlike
    /// [`Self::try_reserve_cached`], this intentionally does not enforce the
    /// current published budget; callers authorizing a physical allocation
    /// must use the checked API.
    ///
    /// Debt arithmetic saturates without losing the whole-operation intent. If
    /// the represented total reaches `u64::MAX`, budget readers remain failed
    /// closed until every overflowed reservation guard has been released.
    pub fn reserve(&self, bytes: u64) -> Reservation<'_> {
        let _update = ReservationUpdateGuard::new(self);
        let mut reserved = self.soft_reserved.load(Ordering::Acquire);
        let (represented_bytes, overflowed) = loop {
            let represented_bytes = bytes.min(u64::MAX - reserved);
            let next = reserved + represented_bytes;
            match self.soft_reserved.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break (represented_bytes, represented_bytes < bytes),
                Err(actual) => reserved = actual,
            }
        };
        if overflowed {
            self.soft_reservation_overflow_debts
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |debts| {
                    debts.checked_add(1)
                })
                .expect("memory governor overflow-debt guard count exceeded u64::MAX");
        }
        drop(_update);
        // Budget dropped — wake the autoscaler so it can shrink KV promptly.
        self.notify_change();
        Reservation {
            governor: self,
            bytes,
            represented_bytes,
            overflowed,
        }
    }

    /// Reserve from the current published budget with atomic check-and-claim.
    ///
    /// The snapshot itself may subsequently change because another process can
    /// allocate device memory, but in-process planners cannot overcommit the
    /// same observation: the compare/exchange includes every live reservation.
    pub fn try_reserve_cached(&self, bytes: u64) -> Option<Reservation<'_>> {
        self.try_reserve_cached_with_credit(bytes, 0)
    }

    fn try_reserve_cached_with_credit(
        &self,
        bytes: u64,
        credited_reservation_bytes: u64,
    ) -> Option<Reservation<'_>> {
        let snapshot = self.cached_snapshot();
        let allocatable = snapshot.free_bytes.saturating_sub(self.cfg.floor_bytes);
        let reservation_limit = allocatable.checked_add(credited_reservation_bytes)?;
        let _update = ReservationUpdateGuard::new(self);
        if bytes > 0 && self.soft_reservation_overflow_debts.load(Ordering::Acquire) > 0 {
            return None;
        }
        let mut reserved = self.soft_reserved.load(Ordering::Acquire);
        loop {
            if bytes > 0 && self.soft_reservation_overflow_debts.load(Ordering::Acquire) > 0 {
                return None;
            }
            let next = reserved.checked_add(bytes)?;
            if next > reservation_limit {
                return None;
            }
            match self.soft_reserved.compare_exchange_weak(
                reserved,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let reservation = Reservation {
                        governor: self,
                        bytes,
                        represented_bytes: bytes,
                        overflowed: false,
                    };
                    drop(_update);
                    if bytes > 0 {
                        self.notify_change();
                    }
                    return Some(reservation);
                }
                Err(actual) => reserved = actual,
            }
        }
    }

    /// Total outstanding soft reservations (for logging / introspection).
    pub fn soft_reserved_bytes(&self) -> u64 {
        self.soft_reserved_bytes_for_budget()
    }

    fn soft_reserved_bytes_for_budget(&self) -> u64 {
        loop {
            let generation = self.reservation_generation.load(Ordering::Acquire);
            if generation & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let represented = self.soft_reserved.load(Ordering::Acquire);
            let overflowed = self.soft_reservation_overflow_debts.load(Ordering::Acquire) > 0;
            if self.reservation_generation.load(Ordering::Acquire) == generation {
                return if overflowed { u64::MAX } else { represented };
            }
            std::hint::spin_loop();
        }
    }

    pub fn config(&self) -> &GovernorConfig {
        &self.cfg
    }

    /// True only after the opt-in background monitor has spawned.
    pub fn monitor_started(&self) -> bool {
        self.monitor_started.load(Ordering::Acquire)
    }

    /// Start the probe-only background publisher. Hot paths read
    /// [`Self::cached_snapshot`] and never perform the refresh themselves.
    /// Idempotent and process-lifetime, like [`Self::start_monitor`].
    pub fn start_sampler(&'static self) -> bool {
        if self
            .sampler_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return self.sampler_running.load(Ordering::Acquire);
        }
        self.sampler_required.store(true, Ordering::Release);
        self.sampler_running.store(true, Ordering::Release);
        let interval = self.cfg.ttl.max(Duration::from_millis(100));
        if let Err(error) = std::thread::Builder::new()
            .name("kiln-mem-sampler".into())
            .spawn(move || {
                let _lifecycle = SamplerLifecycleGuard { governor: self };
                loop {
                    std::thread::sleep(interval);
                    let before = self.cached_snapshot();
                    let after = self.sampler_refresh_once();
                    if before != after {
                        self.notify_change();
                    }
                }
            })
        {
            self.sampler_running.store(false, Ordering::Release);
            self.sampler_started.store(false, Ordering::Release);
            tracing::error!(error = %error, "failed to start memory sampler");
            return false;
        }
        true
    }

    pub fn automatic_reclaim_stats(&self) -> AutomaticReclaimStats {
        *self
            .automatic_reclaim_stats
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn record_automatic_reclaim_suppressed(&self) {
        let mut stats = self
            .automatic_reclaim_stats
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        stats.suppressed_attempts = stats.suppressed_attempts.saturating_add(1);
    }

    fn record_automatic_reclaim_result(
        &self,
        target_bytes: u64,
        reclaimed_bytes: u64,
        duration: Duration,
        update: AutomaticReclaimScheduleUpdate,
    ) {
        let mut stats = self
            .automatic_reclaim_stats
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        stats.attempts = stats.attempts.saturating_add(1);
        if reclaimed_bytes > 0 {
            stats.successful_attempts = stats.successful_attempts.saturating_add(1);
        } else {
            stats.zero_yield_attempts = stats.zero_yield_attempts.saturating_add(1);
        }
        stats.reclaimed_bytes = stats.reclaimed_bytes.saturating_add(reclaimed_bytes);
        stats.last_target_bytes = target_bytes;
        stats.last_reclaimed_bytes = reclaimed_bytes;
        stats.last_duration_us = duration.as_micros().min(u64::MAX as u128) as u64;
        stats.retry_after_ms = update.retry_after.as_millis().min(u64::MAX as u128) as u64;
        stats.zero_yield_streak = u64::from(update.zero_yield_streak);
    }

    fn clear_automatic_reclaim_backoff(&self) {
        let mut stats = self
            .automatic_reclaim_stats
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        stats.retry_after_ms = 0;
        stats.zero_yield_streak = 0;
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
        if self.cfg.reclaim_mode == MemoryReclaimMode::Off {
            return 0;
        }
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
        let s = self.snapshot();
        if !self.pressure_for_snapshot(s).should_reclaim() {
            return 0;
        }
        self.reclaim(self.reclaim_target_to_comfortable(s))
    }

    /// Spawn a background thread that watches pressure and auto-reclaims, turning
    /// the one-shot probe into *continuous* self-adjustment: if a coexisting job
    /// (or kiln itself) drives memory tight, kiln returns pooled VRAM to the OS
    /// without anyone asking. Idempotent — starts at most one thread. Requires a
    /// `'static` governor (use [`MemoryGovernor::global`]).
    pub fn start_monitor(&'static self) -> bool {
        if self.cfg.reclaim_mode != MemoryReclaimMode::Automatic {
            tracing::info!(
                mode = self.cfg.reclaim_mode.as_str(),
                "memory governor automatic reclaim monitor not started"
            );
            return false;
        }
        if self
            .monitor_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return true;
        }
        let interval = self.cfg.ttl.max(Duration::from_secs(2));
        if let Err(err) = std::thread::Builder::new()
            .name("kiln-mem-governor".into())
            .spawn(move || {
                let mut schedule = AutomaticReclaimSchedule::new();
                loop {
                    std::thread::sleep(interval);
                    let before = self.snapshot();
                    let pressure_before = self.pressure_for_snapshot(before);
                    let now = Instant::now();
                    match schedule.decision(pressure_before, now) {
                        AutomaticReclaimDecision::Idle => {
                            if pressure_before == MemoryPressure::Comfortable {
                                self.clear_automatic_reclaim_backoff();
                            }
                            continue;
                        }
                        AutomaticReclaimDecision::BackingOff => {
                            self.record_automatic_reclaim_suppressed();
                            continue;
                        }
                        AutomaticReclaimDecision::Attempt => {}
                    }

                    // An armed episode remains eligible through Moderate
                    // pressure and stops only at Comfortable. This separates
                    // the trigger and recovery thresholds and avoids threshold
                    // chatter.
                    let target_bytes = self.reclaim_target_to_comfortable(before);
                    let started = Instant::now();
                    let reclaimed_bytes = self.reclaim(target_bytes);
                    let duration = started.elapsed();
                    let duration_ms = duration.as_secs_f64() * 1000.0;
                    let after = self.snapshot();
                    let pressure_after = self.pressure_for_snapshot(after);
                    let update = schedule.record_result(Instant::now(), reclaimed_bytes);
                    self.record_automatic_reclaim_result(
                        target_bytes,
                        reclaimed_bytes,
                        duration,
                        update,
                    );
                    let retry_after_ms = update.retry_after.as_millis() as u64;

                    if reclaimed_bytes > 0 {
                        tracing::info!(
                            reason = "automatic_pressure",
                            ?pressure_before,
                            ?pressure_after,
                            target_bytes,
                            reclaimed_bytes,
                            free_before_bytes = before.free_bytes,
                            free_after_bytes = after.free_bytes,
                            total_bytes = after.total_bytes,
                            duration_ms,
                            retry_after_ms,
                            "memory governor automatic reclaim completed"
                        );
                    } else if update.report_zero_yield {
                        tracing::info!(
                            reason = "automatic_pressure_zero_yield",
                            ?pressure_before,
                            ?pressure_after,
                            target_bytes,
                            reclaimed_bytes,
                            free_before_bytes = before.free_bytes,
                            free_after_bytes = after.free_bytes,
                            total_bytes = after.total_bytes,
                            duration_ms,
                            retry_after_ms,
                            zero_yield_streak = update.zero_yield_streak,
                            "memory governor automatic reclaim yielded no bytes; backing off"
                        );
                    } else {
                        tracing::debug!(
                            reason = "automatic_pressure_zero_yield",
                            ?pressure_before,
                            ?pressure_after,
                            target_bytes,
                            reclaimed_bytes,
                            free_before_bytes = before.free_bytes,
                            free_after_bytes = after.free_bytes,
                            total_bytes = after.total_bytes,
                            duration_ms,
                            retry_after_ms,
                            zero_yield_streak = update.zero_yield_streak,
                            "memory governor automatic reclaim still yielded no bytes"
                        );
                    }
                }
            })
        {
            self.monitor_started.store(false, Ordering::Release);
            tracing::error!(error = %err, "failed to start memory governor monitor");
            return false;
        }
        true
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
    /// Requested debt, exposed to the caller even if the aggregate counter had
    /// to saturate.
    bytes: u64,
    /// Portion actually added to `soft_reserved`; drop subtracts exactly this.
    represented_bytes: u64,
    /// Whether this guard carries unrepresented overflow debt.
    overflowed: bool,
}

impl<'a> Reservation<'a> {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Atomically claim a temporary allocation that replaces, rather than
    /// overlaps, the future allocation represented by this guard.
    ///
    /// The caller must keep the original operation unallocated until the
    /// returned guard drops. This is intended for transactional pool
    /// replacement performed while a queued job's working-set reservation is
    /// already live. Only this guard's represented bytes are credited; all
    /// other reservations remain fully charged.
    pub fn try_reserve_replacement_cached(&self, bytes: u64) -> Option<Reservation<'a>> {
        if self.overflowed {
            return None;
        }
        self.governor
            .try_reserve_cached_with_credit(bytes, self.represented_bytes)
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        let _update = ReservationUpdateGuard::new(self.governor);
        self.governor
            .soft_reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                reserved.checked_sub(self.represented_bytes)
            })
            .expect("memory governor soft-reservation counter underflow");
        if self.overflowed {
            self.governor
                .soft_reservation_overflow_debts
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |debts| {
                    debts.checked_sub(1)
                })
                .expect("memory governor overflow-debt guard count underflow");
        }
        drop(_update);
        // Budget freed (e.g. a training job ended) — wake the autoscaler so KV
        // grows back immediately instead of on the next poll tick.
        self.governor.notify_change();
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
            observations: Default::default(),
        }
    }

    const GB: u64 = 1024 * 1024 * 1024;

    fn gov(total: u64, free: u64) -> MemoryGovernor {
        MemoryGovernor::with_source(Box::new(Fixed::new(total, free)), GovernorConfig::default())
    }

    #[test]
    fn probe_selector_is_carried_into_os_source() {
        let probe = OsProbe::new(VramProbeSelector::None);
        assert_eq!(probe.selector(), VramProbeSelector::None);
        assert_eq!(probe.probe().total_bytes, 0);
    }

    #[test]
    fn global_configuration_is_idempotent_but_immutable() {
        let mut state = GlobalGovernorState::new();
        let amd = GlobalGovernorConfiguration {
            selector: VramProbeSelector::LinuxDrm {
                index: 0,
                vendor: Some(crate::vram::LinuxDrmVendor::Amd),
            },
            governor: GovernorConfig::default(),
        };
        assert_eq!(state.configure(amd), Ok(()));
        assert_eq!(state.configure(amd), Ok(()));
        let nvidia = GlobalGovernorConfiguration {
            selector: VramProbeSelector::Nvidia(0),
            governor: GovernorConfig::default(),
        };
        assert!(matches!(
            state.configure(nvidia),
            Err(GlobalGovernorConfigurationError::AlreadyConfigured { .. })
        ));
        assert_eq!(state.initialize(), amd);
        assert_eq!(state.configure(amd), Ok(()));
        assert!(matches!(
            state.configure(nvidia),
            Err(GlobalGovernorConfigurationError::AlreadyInitialized { .. })
        ));

        let mut changed_policy = amd;
        changed_policy.governor.floor_bytes *= 2;
        assert!(matches!(
            state.configure(changed_policy),
            Err(GlobalGovernorConfigurationError::AlreadyInitialized { .. })
        ));
    }

    #[test]
    fn pressure_tracks_free_fraction() {
        // 24 GiB total, default thresholds (crit 5%, tight 10%, comfy 25%).
        assert_eq!(
            gov(24 * GB, 12 * GB).pressure(),
            MemoryPressure::Comfortable
        );
        assert_eq!(gov(24 * GB, 4 * GB).pressure(), MemoryPressure::Moderate); // ~17%
        assert_eq!(gov(24 * GB, 2 * GB).pressure(), MemoryPressure::Tight); // ~8%
        assert_eq!(gov(24 * GB, GB).pressure(), MemoryPressure::Critical); // ~4%
    }

    #[test]
    fn reclaim_mode_is_strict() {
        assert_eq!(
            MemoryReclaimMode::parse("off").unwrap(),
            MemoryReclaimMode::Off
        );
        assert_eq!(
            MemoryReclaimMode::parse("ON-DEMAND").unwrap(),
            MemoryReclaimMode::OnDemand
        );
        assert_eq!(
            MemoryReclaimMode::parse(" automatic ").unwrap(),
            MemoryReclaimMode::Automatic
        );
        assert!(MemoryReclaimMode::parse("true").is_err());
        assert!(MemoryReclaimMode::parse("").is_err());
    }

    #[test]
    fn automatic_reclaim_hysteresis_stays_armed_until_comfortable() {
        let mut schedule = AutomaticReclaimSchedule::new();
        let start = Instant::now();

        assert_eq!(
            schedule.decision(MemoryPressure::Moderate, start),
            AutomaticReclaimDecision::Idle
        );
        assert_eq!(
            schedule.decision(MemoryPressure::Tight, start),
            AutomaticReclaimDecision::Attempt
        );
        let first = schedule.record_result(start, 0);
        assert_eq!(first.retry_after, Duration::from_secs(2));
        assert!(first.report_zero_yield);

        assert_eq!(
            schedule.decision(MemoryPressure::Moderate, start + Duration::from_secs(1)),
            AutomaticReclaimDecision::BackingOff
        );
        assert_eq!(
            schedule.decision(MemoryPressure::Moderate, start + Duration::from_secs(2)),
            AutomaticReclaimDecision::Attempt
        );

        assert_eq!(
            schedule.decision(MemoryPressure::Comfortable, start + Duration::from_secs(2)),
            AutomaticReclaimDecision::Idle
        );
        assert_eq!(
            schedule.decision(MemoryPressure::Moderate, start + Duration::from_secs(3)),
            AutomaticReclaimDecision::Idle
        );
    }

    #[test]
    fn automatic_reclaim_zero_yield_backoff_is_bounded_and_quiet() {
        let mut schedule = AutomaticReclaimSchedule::new();
        let mut now = Instant::now();
        assert_eq!(
            schedule.decision(MemoryPressure::Critical, now),
            AutomaticReclaimDecision::Attempt
        );

        for (index, expected_secs) in [2, 4, 8, 16, 32, 64, 128, 128].into_iter().enumerate() {
            let update = schedule.record_result(now, 0);
            assert_eq!(update.retry_after, Duration::from_secs(expected_secs));
            assert_eq!(update.zero_yield_streak, (index + 1) as u32);
            assert_eq!(update.report_zero_yield, index == 0);
            assert_eq!(
                schedule.decision(
                    MemoryPressure::Critical,
                    now + update.retry_after - Duration::from_millis(1)
                ),
                AutomaticReclaimDecision::BackingOff
            );
            now += update.retry_after;
            assert_eq!(
                schedule.decision(MemoryPressure::Critical, now),
                AutomaticReclaimDecision::Attempt
            );
        }
    }

    #[test]
    fn automatic_reclaim_success_applies_cooldown_and_resets_zero_yield_backoff() {
        let mut schedule = AutomaticReclaimSchedule::new();
        let start = Instant::now();
        assert_eq!(
            schedule.decision(MemoryPressure::Tight, start),
            AutomaticReclaimDecision::Attempt
        );

        let zero = schedule.record_result(start, 0);
        assert_eq!(zero.retry_after, Duration::from_secs(2));
        let retry_at = start + zero.retry_after;
        assert_eq!(
            schedule.decision(MemoryPressure::Tight, retry_at),
            AutomaticReclaimDecision::Attempt
        );

        let success = schedule.record_result(retry_at, GB);
        assert_eq!(success.retry_after, Duration::from_secs(8));
        assert_eq!(success.zero_yield_streak, 0);
        assert!(!success.report_zero_yield);
        assert_eq!(
            schedule.decision(MemoryPressure::Tight, retry_at + Duration::from_secs(7)),
            AutomaticReclaimDecision::BackingOff
        );

        let after_cooldown = retry_at + Duration::from_secs(8);
        assert_eq!(
            schedule.decision(MemoryPressure::Tight, after_cooldown),
            AutomaticReclaimDecision::Attempt
        );
        let next_zero = schedule.record_result(after_cooldown, 0);
        assert_eq!(next_zero.retry_after, Duration::from_secs(2));
        assert_eq!(next_zero.zero_yield_streak, 1);
        assert!(next_zero.report_zero_yield);
    }

    #[test]
    fn automatic_reclaim_stats_report_actual_yield_and_suppression() {
        let g = gov(24 * GB, 2 * GB);
        assert_eq!(
            g.automatic_reclaim_stats(),
            AutomaticReclaimStats::default()
        );

        g.record_automatic_reclaim_suppressed();
        g.record_automatic_reclaim_suppressed();
        g.record_automatic_reclaim_result(
            4 * GB,
            GB,
            Duration::from_micros(1_500),
            AutomaticReclaimScheduleUpdate {
                retry_after: Duration::from_secs(8),
                zero_yield_streak: 0,
                report_zero_yield: false,
            },
        );
        g.record_automatic_reclaim_result(
            3 * GB,
            0,
            Duration::from_micros(250),
            AutomaticReclaimScheduleUpdate {
                retry_after: Duration::from_secs(2),
                zero_yield_streak: 1,
                report_zero_yield: true,
            },
        );

        assert_eq!(
            g.automatic_reclaim_stats(),
            AutomaticReclaimStats {
                attempts: 2,
                successful_attempts: 1,
                zero_yield_attempts: 1,
                suppressed_attempts: 2,
                reclaimed_bytes: GB,
                last_target_bytes: 3 * GB,
                last_reclaimed_bytes: 0,
                last_duration_us: 250,
                retry_after_ms: 2_000,
                zero_yield_streak: 1,
            }
        );
        g.clear_automatic_reclaim_backoff();
        let cleared = g.automatic_reclaim_stats();
        assert_eq!(cleared.retry_after_ms, 0);
        assert_eq!(cleared.zero_yield_streak, 0);
        assert_eq!(cleared.reclaimed_bytes, GB);
    }

    #[test]
    fn reclaim_is_off_by_default_and_disabled_monitor_does_not_start() {
        let cfg = GovernorConfig::default();
        assert_eq!(cfg.reclaim_mode, MemoryReclaimMode::Off);
        let g = Box::leak(Box::new(MemoryGovernor::with_source(
            Box::new(Fixed::new(24 * GB, GB)),
            cfg,
        )));
        let calls = std::sync::Arc::new(AtomicU64::new(0));
        let calls_for_hook = calls.clone();
        g.register_reclaimer(move |_| {
            calls_for_hook.fetch_add(1, Ordering::SeqCst);
            GB
        });
        assert_eq!(g.reclaim(GB), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!g.start_monitor());
        assert!(!g.monitor_started());
    }

    #[test]
    fn missing_selected_device_fails_closed() {
        let governor = gov(0, 0);
        assert_eq!(governor.pressure(), MemoryPressure::Critical);
        assert_eq!(governor.available_bytes(), 0);
    }

    #[test]
    fn configured_capacity_limit_constrains_snapshot_and_admission() {
        const GB: u64 = 1024 * 1024 * 1024;
        let cfg = GovernorConfig {
            capacity_limit_bytes: Some(16 * GB),
            ..GovernorConfig::default()
        };
        let governor = MemoryGovernor::with_source(Box::new(Fixed::new(24 * GB, 20 * GB)), cfg);

        let snapshot = governor.snapshot();
        assert_eq!(snapshot.total_bytes, 16 * GB);
        assert_eq!(snapshot.used_bytes, 4 * GB);
        assert_eq!(snapshot.free_bytes, 12 * GB);
        assert_eq!(governor.available_bytes(), 11 * GB);
    }

    #[test]
    fn cached_snapshot_does_not_wait_for_refresh_probe() {
        struct BlockingRefreshSource {
            calls: AtomicU64,
            started: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>>,
            release: std::sync::Arc<(std::sync::Mutex<bool>, Condvar)>,
        }

        impl MemorySource for BlockingRefreshSource {
            fn probe(&self) -> MemorySnapshot {
                if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return snap(24 * GB, 16 * GB);
                }
                if let Some(started) = self.started.lock().unwrap().take() {
                    let _ = started.send(());
                }
                let (released, wake) = &*self.release;
                let mut released = released.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
                snap(24 * GB, 8 * GB)
            }
        }

        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let release = std::sync::Arc::new((std::sync::Mutex::new(false), Condvar::new()));
        let governor = std::sync::Arc::new(MemoryGovernor::with_source(
            Box::new(BlockingRefreshSource {
                calls: AtomicU64::new(0),
                started: std::sync::Mutex::new(Some(started_tx)),
                release: release.clone(),
            }),
            GovernorConfig::default(),
        ));
        let refresher = {
            let governor = governor.clone();
            std::thread::spawn(move || governor.refresh())
        };
        started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("refresh probe did not start");

        let (cached_tx, cached_rx) = std::sync::mpsc::channel();
        let cached_reader = {
            let governor = governor.clone();
            std::thread::spawn(move || {
                let _ = cached_tx.send(governor.cached_snapshot());
            })
        };
        let cached = cached_rx
            .recv_timeout(Duration::from_millis(250))
            .expect("cached reader blocked behind the in-flight probe");
        assert_eq!(cached.free_bytes, 16 * GB);

        let (released, wake) = &*release;
        *released.lock().unwrap() = true;
        wake.notify_all();
        cached_reader.join().unwrap();
        assert_eq!(refresher.join().unwrap().free_bytes, 8 * GB);
    }

    #[test]
    fn failed_probe_preserves_capacity_but_forces_zero_free_until_recovery() {
        struct TransientFailure {
            calls: AtomicU64,
        }

        impl MemorySource for TransientFailure {
            fn probe(&self) -> MemorySnapshot {
                snap(24 * GB, 16 * GB)
            }

            fn try_probe(&self) -> Option<MemorySnapshot> {
                match self.calls.fetch_add(1, Ordering::SeqCst) {
                    0 => Some(snap(24 * GB, 16 * GB)),
                    1 => None,
                    _ => Some(snap(24 * GB, 8 * GB)),
                }
            }
        }

        let governor = MemoryGovernor::with_source(
            Box::new(TransientFailure {
                calls: AtomicU64::new(0),
            }),
            GovernorConfig::default(),
        );
        let failed = governor.refresh();
        assert_eq!(failed.total_bytes, 24 * GB);
        assert_eq!(failed.used_bytes, 24 * GB);
        assert_eq!(failed.free_bytes, 0);
        assert_eq!(failed.source, VramSource::NvidiaSmi);
        assert!(failed.observations.probe_failed);
        assert_eq!(governor.cached_available_bytes(), 0);

        let recovered = governor.refresh();
        assert_eq!(recovered.free_bytes, 8 * GB);
        assert!(!recovered.observations.probe_failed);
    }

    #[test]
    fn stale_cached_sample_fails_availability_pressure_and_reservation_closed() {
        let governor = gov(24 * GB, 16 * GB);
        {
            let mut state = governor.state.lock().unwrap();
            state.sampled_at = Instant::now()
                .checked_sub(governor.cached_sample_max_age() + Duration::from_millis(1))
                .unwrap();
        }

        let observation = governor.cached_observation();
        assert!(observation.sample_status.stale);
        assert!(!observation.sample_status.healthy);
        assert_eq!(observation.snapshot.total_bytes, 24 * GB);
        assert_eq!(observation.snapshot.used_bytes, 24 * GB);
        assert_eq!(observation.snapshot.free_bytes, 0);
        assert_eq!(observation.available_bytes, 0);
        assert_eq!(observation.pressure, MemoryPressure::Critical);
        assert!(governor.try_reserve_cached(1).is_none());

        let refreshed = governor.refresh();
        assert_eq!(refreshed.free_bytes, 16 * GB);
        assert!(governor.cached_sample_status().healthy);
    }

    #[test]
    fn stopped_required_sampler_fails_cached_sample_closed_and_can_reset_start_flag() {
        let governor = gov(24 * GB, 16 * GB);
        governor.sampler_required.store(true, Ordering::Release);
        governor.sampler_started.store(true, Ordering::Release);
        governor.sampler_running.store(true, Ordering::Release);

        {
            let _lifecycle = SamplerLifecycleGuard {
                governor: &governor,
            };
        }

        assert!(!governor.sampler_started.load(Ordering::Acquire));
        let status = governor.cached_sample_status();
        assert!(status.sampler_required);
        assert!(!status.sampler_running);
        assert!(!status.healthy);
        assert_eq!(governor.cached_snapshot().free_bytes, 0);
        assert_eq!(governor.cached_available_bytes(), 0);
    }

    #[test]
    fn sampler_probe_panic_publishes_failure_and_next_probe_recovers() {
        struct PanicOnce {
            calls: AtomicU64,
        }

        impl MemorySource for PanicOnce {
            fn probe(&self) -> MemorySnapshot {
                match self.calls.fetch_add(1, Ordering::SeqCst) {
                    0 => snap(24 * GB, 16 * GB),
                    1 => panic!("synthetic sampler panic"),
                    _ => snap(24 * GB, 8 * GB),
                }
            }
        }

        let governor = MemoryGovernor::with_source(
            Box::new(PanicOnce {
                calls: AtomicU64::new(0),
            }),
            GovernorConfig::default(),
        );
        let failed = governor.sampler_refresh_once();
        assert!(failed.observations.probe_failed);
        assert_eq!(failed.free_bytes, 0);
        assert!(!governor.cached_sample_status().healthy);

        let recovered = governor.sampler_refresh_once();
        assert_eq!(recovered.free_bytes, 8 * GB);
        assert!(!recovered.observations.probe_failed);
        assert!(governor.cached_sample_status().healthy);
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
        assert_eq!(g.cached_available_bytes(), 15 * GB);
        {
            let _r = g.reserve(10 * GB);
            assert_eq!(g.soft_reserved_bytes(), 10 * GB);
            assert_eq!(g.available_bytes(), 5 * GB);
            assert_eq!(g.cached_available_bytes(), 5 * GB);
            assert!(!g.can_fit(6 * GB));
        }
        // Guard dropped -> reservation released.
        assert_eq!(g.soft_reserved_bytes(), 0);
        assert_eq!(g.available_bytes(), 15 * GB);
        assert_eq!(g.cached_available_bytes(), 15 * GB);
    }

    #[test]
    fn unconditional_reservation_overflow_stays_failed_closed_in_any_drop_order() {
        let g = gov(u64::MAX, u64::MAX);
        let base = g.reserve(u64::MAX - 5);
        let overflow = g.reserve(10);

        assert_eq!(base.bytes(), u64::MAX - 5);
        assert_eq!(overflow.bytes(), 10);
        assert_eq!(g.soft_reserved_bytes(), u64::MAX);
        assert_eq!(g.cached_available_bytes(), 0);

        // The mostly represented guard may finish before the overflowed guard.
        // Hidden debt must still keep admission closed rather than exposing the
        // five represented bytes as available.
        drop(base);
        assert_eq!(g.soft_reserved_bytes(), u64::MAX);
        assert!(g.try_reserve_cached(1).is_none());

        drop(overflow);
        assert_eq!(g.soft_reserved_bytes(), 0);
        assert!(g.try_reserve_cached(1).is_some());
    }

    #[test]
    fn cached_observation_keeps_budget_and_reservation_generation_coherent() {
        let governor = std::sync::Arc::new(gov(24 * GB, 16 * GB));
        let writer = {
            let governor = std::sync::Arc::clone(&governor);
            std::thread::spawn(move || {
                for _ in 0..2_000 {
                    let reservation = governor.reserve(GB);
                    std::thread::yield_now();
                    drop(reservation);
                }
            })
        };

        for _ in 0..2_000 {
            let observation = governor.cached_observation();
            let expected_available = observation
                .snapshot
                .free_bytes
                .saturating_sub(governor.config().floor_bytes)
                .saturating_sub(observation.soft_reserved_bytes);
            assert_eq!(observation.available_bytes, expected_available);
            assert_eq!(
                observation.pressure,
                governor.pressure_for_snapshot(observation.snapshot)
            );
            std::thread::yield_now();
        }
        writer.join().unwrap();
        assert_eq!(governor.soft_reserved_bytes(), 0);
    }

    #[test]
    fn checked_cached_reservation_never_exceeds_published_budget() {
        let g = gov(24 * GB, 16 * GB);
        let first = g
            .try_reserve_cached(10 * GB)
            .expect("ten GiB fits the fifteen-GiB published budget");
        let second = g
            .try_reserve_cached(5 * GB)
            .expect("the remaining five GiB fits exactly");
        assert!(g.try_reserve_cached(1).is_none());
        drop(first);
        assert_eq!(g.cached_available_bytes(), 10 * GB);
        drop(second);
        assert_eq!(g.cached_available_bytes(), 15 * GB);
    }

    #[test]
    fn replacement_reservation_credits_only_its_original_guard() {
        let g = gov(24 * GB, 16 * GB); // 15 GiB after the default floor.
        let original_operation = g.reserve(10 * GB);
        let other_operation = g.reserve(2 * GB);

        assert!(g.try_reserve_cached(4 * GB).is_none());
        assert!(
            original_operation
                .try_reserve_replacement_cached(14 * GB)
                .is_none(),
            "the unrelated two-GiB reservation must remain charged"
        );
        let replacement = original_operation
            .try_reserve_replacement_cached(13 * GB)
            .expect("the replacement may reuse only the original ten-GiB reservation");
        assert_eq!(replacement.bytes(), 13 * GB);
        assert_eq!(g.soft_reserved_bytes(), 25 * GB);
        assert_eq!(g.cached_available_bytes(), 0);

        drop(replacement);
        drop(other_operation);
        drop(original_operation);
        assert_eq!(g.soft_reserved_bytes(), 0);
    }

    #[test]
    fn checked_cached_reservation_is_atomic_across_planners() {
        let g = Box::leak(Box::new(gov(24 * GB, 6 * GB)));
        let mut workers = Vec::new();
        for _ in 0..8 {
            workers.push(std::thread::spawn(move || g.try_reserve_cached(GB)));
        }
        let reservations = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            reservations
                .iter()
                .filter(|result| result.is_some())
                .count(),
            5,
            "the five-GiB post-floor budget must admit exactly five concurrent claims"
        );
        drop(reservations);
        assert_eq!(g.soft_reserved_bytes(), 0);
    }

    #[test]
    fn reservation_change_wakes_waiter_for_growback() {
        use std::sync::Arc;
        let g = Arc::new(gov(24 * GB, 16 * GB));
        let res = g.reserve(4 * GB);
        let g2 = g.clone();
        let start = Instant::now();
        // A waiter that would block the full 10s timeout absent an event.
        let waiter = std::thread::spawn(move || g2.wait_for_change(Duration::from_secs(10)));
        std::thread::sleep(Duration::from_millis(50));
        drop(res); // Reservation::drop -> notify_change -> waiter must wake.
        waiter.join().unwrap();
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "autoscaler waiter did NOT wake on reservation drop — grow-back would be late"
        );
        assert_eq!(g.soft_reserved_bytes(), 0);
    }

    #[test]
    fn wait_for_change_times_out_without_event() {
        let g = gov(24 * GB, 16 * GB);
        let start = Instant::now();
        g.wait_for_change(Duration::from_millis(120)); // no notify -> returns ~timeout
        let waited = start.elapsed();
        assert!(
            waited >= Duration::from_millis(100),
            "returned too early: {waited:?}"
        );
        assert!(
            waited < Duration::from_secs(2),
            "timeout overshot: {waited:?}"
        );
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
            reclaim_mode: MemoryReclaimMode::OnDemand,
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
