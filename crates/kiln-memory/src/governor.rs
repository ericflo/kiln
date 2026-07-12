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

use crate::vram::{MemorySnapshot, current_memory_snapshot};

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

pub const MEMORY_RECLAIM_MODE_ENV: &str = "KILN_MEMORY_RECLAIM_MODE";

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
                "{MEMORY_RECLAIM_MODE_ENV} must be one of off, on-demand, automatic; got {value:?}"
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
    /// Whether backend reclaim hooks are disabled, on-demand, or periodic.
    pub reclaim_mode: MemoryReclaimMode,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        GovernorConfig {
            ttl: Duration::from_millis(500),
            floor_bytes: 1024 * 1024 * 1024, // 1 GiB
            critical_frac: 0.05,
            tight_frac: 0.10,
            comfortable_frac: 0.25,
            reclaim_mode: MemoryReclaimMode::Off,
        }
    }
}

impl GovernorConfig {
    /// Build from defaults with optional env overrides:
    /// * `KILN_MEMORY_FLOOR_GB` — safety floor in GiB.
    /// * `KILN_MEMORY_PROBE_MS` — probe TTL in milliseconds.
    /// * `KILN_MEMORY_RECLAIM_MODE` — `off`, `on-demand`, or `automatic`.
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
        if let Some(v) = std::env::var_os(MEMORY_RECLAIM_MODE_ENV) {
            let v = v
                .into_string()
                .unwrap_or_else(|_| panic!("{MEMORY_RECLAIM_MODE_ENV} must contain valid UTF-8"));
            cfg.reclaim_mode =
                MemoryReclaimMode::parse(&v).unwrap_or_else(|error| panic!("{error}"));
        }
        cfg
    }
}

struct State {
    cached: MemorySnapshot,
    sampled_at: Instant,
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
    /// Soft reservations: memory announced-but-not-yet-allocated. Summed and
    /// subtracted from `available_bytes` so two consumers can't both plan to
    /// use the same free bytes. Released via the [`Reservation`] guard.
    soft_reserved: AtomicU64,
    /// Reclaim hooks registered by the allocator layer (return pooled VRAM to
    /// the OS). Invoked under pressure by [`Self::reclaim`] / the monitor.
    reclaimers: Mutex<Vec<Reclaimer>>,
    /// Guards [`Self::start_monitor`] against spawning more than one thread.
    monitor_started: AtomicBool,
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
        self.pressure_for_snapshot(self.snapshot())
    }

    fn pressure_for_snapshot(&self, s: MemorySnapshot) -> MemoryPressure {
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

    fn reclaim_target_to_comfortable(&self, s: MemorySnapshot) -> u64 {
        let want_free = ((s.total_bytes as f64) * self.cfg.comfortable_frac) as u64;
        want_free.saturating_sub(s.free_bytes).max(1)
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
        // Budget dropped — wake the autoscaler so it can shrink KV promptly.
        self.notify_change();
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

    /// True only after the opt-in background monitor has spawned.
    pub fn monitor_started(&self) -> bool {
        self.monitor_started.load(Ordering::Acquire)
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
        }
    }

    const GB: u64 = 1024 * 1024 * 1024;

    fn gov(total: u64, free: u64) -> MemoryGovernor {
        MemoryGovernor::with_source(Box::new(Fixed::new(total, free)), GovernorConfig::default())
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
