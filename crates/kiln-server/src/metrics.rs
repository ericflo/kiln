//! Prometheus metrics collection for kiln.
//!
//! Uses atomic counters and gauges — no external dependencies.
//! The `/metrics` endpoint renders all metrics in Prometheus text exposition format.

use kiln_core::thinking_budget::ThinkingBudgetSource;
use kiln_model::{DecodeBatcherStats, ExternalYieldSyncStats};
use kiln_scheduler::PrefixCacheStats;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::batching_engine::BatchingEngineSnapshot;
use crate::latency_observability::{
    ABSOLUTE_STALL_THRESHOLD, LatencyPhaseTimings, LatencyStallReason, RequestLatencyDiagnostics,
    TokenGapObservation,
};
use crate::memory_observability::CachedMemoryGovernorObservation;
use crate::recent_requests::RequestThinkingBudget;

const LATENCY_BUCKETS_SECONDS: [f64; 13] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

const LATENCY_BUCKETS_US: [u64; LATENCY_BUCKETS_SECONDS.len()] = [
    5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000,
    10_000_000, 30_000_000, 60_000_000,
];

const REQUEST_LATENCY_BUCKETS_SECONDS: [f64; 16] = [
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 420.0, 600.0,
    900.0,
];

const REQUEST_LATENCY_BUCKETS_US: [u64; REQUEST_LATENCY_BUCKETS_SECONDS.len()] = [
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    30_000_000,
    60_000_000,
    120_000_000,
    180_000_000,
    240_000_000,
    300_000_000,
    420_000_000,
    600_000_000,
    900_000_000,
];

const THINKING_BUDGET_TOKEN_BUCKETS: [u64; 17] = [
    0, 1, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072,
];
const THINKING_BUDGET_TIME_BUCKETS_MS: [u64; 16] = [
    0, 1, 10, 50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000, 300_000, 3_600_000,
    86_400_000,
];
const THINKING_BUDGET_SOURCES: [ThinkingBudgetSource; 5] = [
    ThinkingBudgetSource::Request,
    ThinkingBudgetSource::ServerDefault,
    ThinkingBudgetSource::RequestUnlimited,
    ThinkingBudgetSource::Unlimited,
    ThinkingBudgetSource::Unknown,
];
const THINKING_BUDGET_OUTCOMES: [&str; 9] = [
    "unconfigured",
    "inert",
    "natural_close",
    "tokens",
    "time",
    "max_tokens",
    "unclosed",
    "interrupted",
    "unresolved",
];

const LATENCY_PHASES: [&str; 20] = [
    "actor_queue",
    "actor_admission",
    "tokenization",
    "prefill",
    "decode",
    "actor_cycle_idle",
    "sampling",
    "readback",
    "response_delivery",
    "handler_queue",
    "client_delivery",
    "gpu_lock_wait",
    "graph_capture",
    "graph_replay",
    "synchronization",
    "resize",
    "trim",
    "adapter",
    "training",
    "unexplained",
];

struct AtomicLatencyHistogram {
    count: AtomicU64,
    sum_us: AtomicU64,
    buckets: [AtomicU64; LATENCY_BUCKETS_US.len() + 1],
}

impl AtomicLatencyHistogram {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    fn observe_milliseconds(&self, milliseconds: f64) {
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return;
        }
        let microseconds = (milliseconds * 1_000.0).min(u64::MAX as f64) as u64;
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add(microseconds, Ordering::Relaxed);
        observe_bucket(&self.buckets, &LATENCY_BUCKETS_US, microseconds);
    }
}

/// Atomically tracked metrics for the kiln server.
pub struct Metrics {
    // Inference counters
    pub requests_ok: AtomicU64,
    pub requests_error: AtomicU64,
    pub requests_timeout: AtomicU64,
    pub requests_rejected: AtomicU64,

    /// Total tokens generated across all requests.
    pub tokens_generated: AtomicU64,

    /// Currently in-flight inference requests.
    pub active_requests: AtomicU64,

    /// Peak concurrently in-flight inference requests since process start.
    pub active_requests_peak: AtomicU64,

    /// Prompt/prefix prefill tokens completed by currently in-flight requests.
    pub request_prefill_tokens_completed: Arc<AtomicU64>,

    /// Request duration tracking (simple: count + sum in microseconds).
    pub request_duration_count: AtomicU64,
    pub request_duration_sum_us: AtomicU64,
    pub request_duration_buckets: [AtomicU64; REQUEST_LATENCY_BUCKETS_US.len() + 1],

    /// Time spent in model prefill before decode starts, in microseconds.
    pub prefill_duration_count: AtomicU64,
    pub prefill_duration_sum_us: AtomicU64,
    pub prefill_duration_buckets: [AtomicU64; LATENCY_BUCKETS_US.len() + 1],

    /// Time spent in token decode after prefill, in microseconds.
    pub decode_duration_count: AtomicU64,
    pub decode_duration_sum_us: AtomicU64,
    pub decode_duration_buckets: [AtomicU64; LATENCY_BUCKETS_US.len() + 1],

    request_ttft: AtomicLatencyHistogram,
    token_itl: AtomicLatencyHistogram,
    latency_phases: [AtomicLatencyHistogram; LATENCY_PHASES.len()],
    token_stall_reasons: [AtomicU64; LatencyStallReason::ALL.len()],

    /// Fixed-cardinality thinking-budget telemetry. Effective numeric limits
    /// are histograms, never labels; source and outcome labels come only from
    /// the closed sets above.
    thinking_budget_token_sources: [AtomicU64; THINKING_BUDGET_SOURCES.len()],
    thinking_budget_time_sources: [AtomicU64; THINKING_BUDGET_SOURCES.len()],
    thinking_budget_outcomes: [AtomicU64; THINKING_BUDGET_OUTCOMES.len()],
    thinking_budget_token_limit_count: AtomicU64,
    thinking_budget_token_limit_sum: AtomicU64,
    thinking_budget_token_limit_buckets: [AtomicU64; THINKING_BUDGET_TOKEN_BUCKETS.len() + 1],
    thinking_budget_time_limit_count: AtomicU64,
    thinking_budget_time_limit_sum_ms: AtomicU64,
    thinking_budget_time_limit_buckets: [AtomicU64; THINKING_BUDGET_TIME_BUCKETS_MS.len() + 1],

    // Training counters
    pub training_sft_completed: AtomicU64,
    pub training_sft_failed: AtomicU64,
    pub training_sft_cancelled: AtomicU64,
    pub training_grpo_completed: AtomicU64,
    pub training_grpo_failed: AtomicU64,
    pub training_grpo_cancelled: AtomicU64,
    pub training_opd_completed: AtomicU64,
    pub training_opd_failed: AtomicU64,
    pub training_opd_cancelled: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_ok: AtomicU64::new(0),
            requests_error: AtomicU64::new(0),
            requests_timeout: AtomicU64::new(0),
            requests_rejected: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            active_requests_peak: AtomicU64::new(0),
            request_prefill_tokens_completed: Arc::new(AtomicU64::new(0)),
            request_duration_count: AtomicU64::new(0),
            request_duration_sum_us: AtomicU64::new(0),
            request_duration_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            prefill_duration_count: AtomicU64::new(0),
            prefill_duration_sum_us: AtomicU64::new(0),
            prefill_duration_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            decode_duration_count: AtomicU64::new(0),
            decode_duration_sum_us: AtomicU64::new(0),
            decode_duration_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            request_ttft: AtomicLatencyHistogram::new(),
            token_itl: AtomicLatencyHistogram::new(),
            latency_phases: std::array::from_fn(|_| AtomicLatencyHistogram::new()),
            token_stall_reasons: std::array::from_fn(|_| AtomicU64::new(0)),
            thinking_budget_token_sources: std::array::from_fn(|_| AtomicU64::new(0)),
            thinking_budget_time_sources: std::array::from_fn(|_| AtomicU64::new(0)),
            thinking_budget_outcomes: std::array::from_fn(|_| AtomicU64::new(0)),
            thinking_budget_token_limit_count: AtomicU64::new(0),
            thinking_budget_token_limit_sum: AtomicU64::new(0),
            thinking_budget_token_limit_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            thinking_budget_time_limit_count: AtomicU64::new(0),
            thinking_budget_time_limit_sum_ms: AtomicU64::new(0),
            thinking_budget_time_limit_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            training_sft_completed: AtomicU64::new(0),
            training_sft_failed: AtomicU64::new(0),
            training_sft_cancelled: AtomicU64::new(0),
            training_grpo_completed: AtomicU64::new(0),
            training_grpo_failed: AtomicU64::new(0),
            training_grpo_cancelled: AtomicU64::new(0),
            training_opd_completed: AtomicU64::new(0),
            training_opd_failed: AtomicU64::new(0),
            training_opd_cancelled: AtomicU64::new(0),
        }
    }

    /// Increment the request counter for the given status.
    pub fn inc_request(&self, status: RequestStatus) {
        match status {
            RequestStatus::Ok => self.requests_ok.fetch_add(1, Ordering::Relaxed),
            RequestStatus::Error => self.requests_error.fetch_add(1, Ordering::Relaxed),
            RequestStatus::Timeout => self.requests_timeout.fetch_add(1, Ordering::Relaxed),
            RequestStatus::Rejected => self.requests_rejected.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Record a completed request duration in seconds.
    pub fn observe_duration(&self, secs: f64) {
        self.request_duration_count.fetch_add(1, Ordering::Relaxed);
        let us = (secs * 1_000_000.0) as u64;
        self.request_duration_sum_us
            .fetch_add(us, Ordering::Relaxed);
        observe_bucket(
            &self.request_duration_buckets,
            &REQUEST_LATENCY_BUCKETS_US,
            us,
        );
    }

    /// Record model prefill duration in seconds.
    pub fn observe_prefill_duration(&self, secs: f64) {
        self.prefill_duration_count.fetch_add(1, Ordering::Relaxed);
        let us = (secs * 1_000_000.0) as u64;
        self.prefill_duration_sum_us
            .fetch_add(us, Ordering::Relaxed);
        observe_bucket(&self.prefill_duration_buckets, &LATENCY_BUCKETS_US, us);
    }

    /// Record model decode duration in seconds.
    pub fn observe_decode_duration(&self, secs: f64) {
        self.decode_duration_count.fetch_add(1, Ordering::Relaxed);
        let us = (secs * 1_000_000.0) as u64;
        self.decode_duration_sum_us.fetch_add(us, Ordering::Relaxed);
        observe_bucket(&self.decode_duration_buckets, &LATENCY_BUCKETS_US, us);
    }

    /// Record one request-local token gap. The histogram includes every gap;
    /// the reason counter uses the fixed 250ms absolute floor so its meaning is
    /// stable across scrapes. The rolling stats endpoint additionally applies
    /// the request-sensitive `max(250ms, 5*p50)` stall gate.
    pub fn observe_token_gap(&self, observation: TokenGapObservation) {
        self.token_itl
            .observe_milliseconds(observation.gap.as_secs_f64() * 1_000.0);
        if observation.gap >= ABSOLUTE_STALL_THRESHOLD {
            self.token_stall_reasons[observation.reason.index()].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record the bounded phase summary attached to one completed request.
    /// Missing backend subphases are not converted to zero observations.
    pub fn observe_request_latency(&self, diagnostics: &RequestLatencyDiagnostics) {
        if let Some(ttft_ms) = diagnostics.ttft_ms {
            self.request_ttft.observe_milliseconds(ttft_ms);
        }
        for (index, value) in latency_phase_values(&diagnostics.phases)
            .into_iter()
            .enumerate()
        {
            if let Some(milliseconds) = value {
                self.latency_phases[index].observe_milliseconds(milliseconds);
            }
        }
    }

    /// Record the effective thinking-budget configuration and one bounded
    /// outcome category for a completed recent-request record.
    pub fn observe_thinking_budget(&self, budget: &RequestThinkingBudget, finish_reason: &str) {
        let token_source = thinking_budget_source_index(&budget.tokens_source);
        self.thinking_budget_token_sources[token_source].fetch_add(1, Ordering::Relaxed);
        let time_source = thinking_budget_source_index(&budget.time_source);
        self.thinking_budget_time_sources[time_source].fetch_add(1, Ordering::Relaxed);

        if let Some(limit) = budget.max_tokens {
            self.thinking_budget_token_limit_count
                .fetch_add(1, Ordering::Relaxed);
            self.thinking_budget_token_limit_sum
                .fetch_add(limit, Ordering::Relaxed);
            observe_bucket(
                &self.thinking_budget_token_limit_buckets,
                &THINKING_BUDGET_TOKEN_BUCKETS,
                limit,
            );
        }
        if let Some(limit_ms) = budget.max_time_ms {
            self.thinking_budget_time_limit_count
                .fetch_add(1, Ordering::Relaxed);
            self.thinking_budget_time_limit_sum_ms
                .fetch_add(limit_ms, Ordering::Relaxed);
            observe_bucket(
                &self.thinking_budget_time_limit_buckets,
                &THINKING_BUDGET_TIME_BUCKETS_MS,
                limit_ms,
            );
        }

        let outcome = thinking_budget_outcome(budget, finish_reason);
        let outcome_index = THINKING_BUDGET_OUTCOMES
            .iter()
            .position(|candidate| *candidate == outcome)
            .expect("thinking-budget outcome must use the closed metric vocabulary");
        self.thinking_budget_outcomes[outcome_index].fetch_add(1, Ordering::Relaxed);
    }

    /// Add generated token count.
    pub fn add_tokens(&self, n: u64) {
        self.tokens_generated.fetch_add(n, Ordering::Relaxed);
    }

    /// Increment active requests (call on request entry).
    pub fn inc_active(&self) {
        let active = self.active_requests.fetch_add(1, Ordering::Relaxed) + 1;
        update_peak(&self.active_requests_peak, active);
    }

    /// Decrement active requests (call on request exit).
    pub fn dec_active(&self) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record a training job completion.
    pub fn inc_training(&self, job_type: TrainingMetricType, status: TrainingMetricStatus) {
        match (job_type, status) {
            (TrainingMetricType::Sft, TrainingMetricStatus::Completed) => {
                self.training_sft_completed.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Sft, TrainingMetricStatus::Failed) => {
                self.training_sft_failed.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Sft, TrainingMetricStatus::Cancelled) => {
                self.training_sft_cancelled.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Grpo, TrainingMetricStatus::Completed) => {
                self.training_grpo_completed.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Grpo, TrainingMetricStatus::Failed) => {
                self.training_grpo_failed.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Grpo, TrainingMetricStatus::Cancelled) => {
                self.training_grpo_cancelled.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Opd, TrainingMetricStatus::Completed) => {
                self.training_opd_completed.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Opd, TrainingMetricStatus::Failed) => {
                self.training_opd_failed.fetch_add(1, Ordering::Relaxed);
            }
            (TrainingMetricType::Opd, TrainingMetricStatus::Cancelled) => {
                self.training_opd_cancelled.fetch_add(1, Ordering::Relaxed);
            }
        };
    }

    /// Render all metrics in Prometheus text exposition format.
    ///
    /// Dynamic gauges (scheduler state, GPU memory, training active, adapter) are
    /// passed in as `SnapshotGauges` because they come from shared state that the
    /// metrics struct itself doesn't own.
    pub fn render(&self, gauges: &SnapshotGauges) -> String {
        let mut out = String::with_capacity(2048);

        // --- Inference ---
        out.push_str("# HELP kiln_requests_total Total inference requests.\n");
        out.push_str("# TYPE kiln_requests_total counter\n");
        prom_counter(
            &mut out,
            "kiln_requests_total",
            "status",
            "ok",
            self.requests_ok.load(Ordering::Relaxed),
        );

        out.push_str("# HELP kiln_backend_quarantined Whether inference is disabled because backend completion became unknown.\n");
        out.push_str("# TYPE kiln_backend_quarantined gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_backend_quarantined {}",
                u8::from(gauges.backend_quarantined)
            ),
        );
        out.push_str("# HELP kiln_backend_external_yield_sync_calls_total Backend completion waits before external progress or resource reuse.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_calls_total counter\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_failures_total Failed backend completion waits.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_failures_total counter\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_seconds_total Total wall time spent proving backend completion.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_seconds_total counter\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_max_seconds Maximum observed backend completion wait.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_max_seconds gauge\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_slow_total Synchronization calls taking at least 100 milliseconds.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_slow_total counter\n");
        for stats in &gauges.external_yield_sync {
            prom_counter(
                &mut out,
                "kiln_backend_external_yield_sync_calls_total",
                "boundary",
                &stats.boundary,
                stats.calls,
            );
            prom_counter(
                &mut out,
                "kiln_backend_external_yield_sync_failures_total",
                "boundary",
                &stats.boundary,
                stats.failures,
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_backend_external_yield_sync_seconds_total{{boundary=\"{}\"}} {}",
                    stats.boundary,
                    stats.total_micros as f64 / 1_000_000.0
                ),
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_backend_external_yield_sync_max_seconds{{boundary=\"{}\"}} {}",
                    stats.boundary,
                    stats.max_micros as f64 / 1_000_000.0
                ),
            );
            prom_counter(
                &mut out,
                "kiln_backend_external_yield_sync_slow_total",
                "boundary",
                &stats.boundary,
                stats.slow_calls,
            );
        }
        out.push_str(
            "# HELP kiln_rocm_synchronization_active Whether the selected model backend is ROCm.\n",
        );
        out.push_str("# TYPE kiln_rocm_synchronization_active gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rocm_synchronization_active {}",
                u8::from(gauges.rocm_synchronization.active)
            ),
        );
        out.push_str("# HELP kiln_rocm_synchronization_telemetry_available Whether the selected ROCm context supplied an atomic telemetry snapshot.\n");
        out.push_str("# TYPE kiln_rocm_synchronization_telemetry_available gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rocm_synchronization_telemetry_available {}",
                u8::from(gauges.rocm_synchronization.telemetry_available)
            ),
        );
        out.push_str("# HELP kiln_rocm_cleanup_quarantined Whether failed ROCm recovery made further execution and resource destruction unsafe. Valid only when synchronization telemetry is available.\n");
        out.push_str("# TYPE kiln_rocm_cleanup_quarantined gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rocm_cleanup_quarantined {}",
                u8::from(gauges.rocm_synchronization.cleanup_quarantined)
            ),
        );
        out.push_str("# HELP kiln_rocm_synchronization_policy_info Immutable ROCm synchronization policy selected at startup.\n");
        out.push_str("# TYPE kiln_rocm_synchronization_policy_info gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rocm_synchronization_policy_info{{mode=\"{}\"}} 1",
                gauges.rocm_synchronization_mode
            ),
        );
        out.push_str("# HELP kiln_rocm_synchronizations_total ROCm host wait attempts by fixed reason and scope.\n");
        out.push_str("# TYPE kiln_rocm_synchronizations_total counter\n");
        out.push_str("# HELP kiln_rocm_synchronization_wait_seconds_total Host wall time spent in ROCm waits by fixed reason.\n");
        out.push_str("# TYPE kiln_rocm_synchronization_wait_seconds_total counter\n");
        out.push_str("# HELP kiln_rocm_synchronization_skipped_total Same-stream ROCm barriers omitted by stream-ordered policy.\n");
        out.push_str("# TYPE kiln_rocm_synchronization_skipped_total counter\n");
        for stats in &gauges.rocm_synchronization.reasons {
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_synchronizations_total{{reason=\"{}\",scope=\"device\"}} {}",
                    stats.reason, stats.device_wait_count
                ),
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_synchronizations_total{{reason=\"{}\",scope=\"stream\"}} {}",
                    stats.reason, stats.stream_wait_count
                ),
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_synchronization_wait_seconds_total{{reason=\"{}\"}} {}",
                    stats.reason,
                    stats.waited_ns as f64 / 1_000_000_000.0
                ),
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_synchronization_skipped_total{{reason=\"{}\"}} {}",
                    stats.reason, stats.skipped_count
                ),
            );
        }
        out.push_str("# HELP kiln_rocm_graph_telemetry_available Whether a nonblocking ROCm graph-runner snapshot was available.\n");
        out.push_str("# TYPE kiln_rocm_graph_telemetry_available gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rocm_graph_telemetry_available {}",
                u8::from(gauges.rocm_graph.is_some())
            ),
        );
        out.push_str("# HELP kiln_rocm_graph_snapshot_unavailable Whether the full graph snapshot is unavailable for a closed reason.\n");
        out.push_str("# TYPE kiln_rocm_graph_snapshot_unavailable gauge\n");
        for reason in crate::rocm_graph_observability::RocmGraphUnavailableReason::ALL {
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_snapshot_unavailable{{reason=\"{}\"}} {}",
                    reason.as_str(),
                    u8::from(gauges.rocm_graph_unavailable_reason == Some(reason))
                ),
            );
        }
        out.push_str("# HELP kiln_rocm_graph_phase_telemetry_available Whether ROCm graph phase telemetry independent of the model and graph-runner locks was available.\n");
        out.push_str("# TYPE kiln_rocm_graph_phase_telemetry_available gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rocm_graph_phase_telemetry_available {}",
                u8::from(gauges.rocm_graph_telemetry.is_some())
            ),
        );
        out.push_str("# HELP kiln_rocm_graph_phase_telemetry_unavailable Whether graph phase telemetry independent of the model and graph-runner locks is unavailable for a closed reason.\n");
        out.push_str("# TYPE kiln_rocm_graph_phase_telemetry_unavailable gauge\n");
        for reason in crate::rocm_graph_observability::RocmGraphUnavailableReason::ALL {
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_phase_telemetry_unavailable{{reason=\"{}\"}} {}",
                    reason.as_str(),
                    u8::from(gauges.rocm_graph_telemetry_unavailable_reason == Some(reason))
                ),
            );
        }
        if let Some(telemetry) = gauges.rocm_graph_telemetry {
            let graph_phases = [
                (
                    "pre_candidate_headroom",
                    kiln_model::RocmGraphPhase::PreCandidateHeadroom,
                    telemetry.pre_candidate_headroom_phase,
                ),
                (
                    "candidate_warm",
                    kiln_model::RocmGraphPhase::CandidateWarm,
                    telemetry.candidate_warm_phase,
                ),
                (
                    "pre_native_reservation",
                    kiln_model::RocmGraphPhase::PreNativeReservation,
                    telemetry.pre_native_reservation_phase,
                ),
                (
                    "native_capture",
                    kiln_model::RocmGraphPhase::NativeCapture,
                    telemetry.native_capture_phase,
                ),
                (
                    "rejected_candidate_cleanup",
                    kiln_model::RocmGraphPhase::RejectedCandidateCleanup,
                    telemetry.rejected_candidate_cleanup_phase,
                ),
            ];
            out.push_str("# HELP kiln_rocm_graph_current_phase Current ROCm graph candidate lifecycle phase as a closed one-hot gauge.\n");
            out.push_str("# TYPE kiln_rocm_graph_current_phase gauge\n");
            for (phase, phase_kind, _) in graph_phases {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_current_phase{{phase=\"{phase}\"}} {}",
                        u8::from(telemetry.current_phase == Some(phase_kind))
                    ),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_current_phase_elapsed_seconds Monotonic elapsed time in the active ROCm graph phase, or zero while idle.\n");
            out.push_str("# TYPE kiln_rocm_graph_current_phase_elapsed_seconds gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_current_phase_elapsed_seconds {:.6}",
                    telemetry.current_phase_elapsed_micros as f64 / 1_000_000.0
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_transient_candidate_bytes Exact requested physical bytes in the last and largest measured pre-admission graph candidate.\n");
            out.push_str("# TYPE kiln_rocm_graph_transient_candidate_bytes gauge\n");
            for (kind, bytes) in [
                ("last", telemetry.last_transient_candidate_bytes),
                ("peak", telemetry.peak_transient_candidate_bytes),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_transient_candidate_bytes{{kind=\"{kind}\"}} {bytes}"
                    ),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_phase_calls_total ROCm graph candidate lifecycle calls by fixed phase.\n");
            out.push_str("# TYPE kiln_rocm_graph_phase_calls_total counter\n");
            out.push_str("# HELP kiln_rocm_graph_phase_slow_total ROCm graph candidate phase calls taking at least 100 ms.\n");
            out.push_str("# TYPE kiln_rocm_graph_phase_slow_total counter\n");
            out.push_str("# HELP kiln_rocm_graph_phase_duration_seconds_total Accumulated ROCm graph candidate phase duration.\n");
            out.push_str("# TYPE kiln_rocm_graph_phase_duration_seconds_total counter\n");
            out.push_str("# HELP kiln_rocm_graph_phase_duration_seconds_max Longest observed ROCm graph candidate phase duration.\n");
            out.push_str("# TYPE kiln_rocm_graph_phase_duration_seconds_max gauge\n");
            for (phase, _, stats) in graph_phases {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_phase_calls_total{{phase=\"{phase}\"}} {}",
                        stats.calls
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_phase_slow_total{{phase=\"{phase}\"}} {}",
                        stats.slow
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_phase_duration_seconds_total{{phase=\"{phase}\"}} {:.6}",
                        stats.total_duration_micros as f64 / 1_000_000.0
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_phase_duration_seconds_max{{phase=\"{phase}\"}} {:.6}",
                        stats.max_duration_micros as f64 / 1_000_000.0
                    ),
                );
            }
        }
        if let Some(graph) = gauges.rocm_graph {
            out.push_str(
                "# HELP kiln_rocm_graph_state ROCm graph policy and circuit-breaker state.\n",
            );
            out.push_str("# TYPE kiln_rocm_graph_state gauge\n");
            for (name, value) in [
                ("requested", graph.requested),
                ("capture_requested", graph.capture_requested),
                ("enabled", graph.enabled),
                ("capture_enabled", graph.capture_enabled),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_state{{kind=\"{name}\"}} {}",
                        u8::from(value)
                    ),
                );
            }
            out.push_str(
                "# HELP kiln_rocm_graph_cache_entries Native graphs currently retained.\n",
            );
            out.push_str("# TYPE kiln_rocm_graph_cache_entries gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_cache_entries {}",
                    graph.captured_graph_count
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_cache_entry_limit Configured maximum retained native graphs.\n");
            out.push_str("# TYPE kiln_rocm_graph_cache_entry_limit gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_cache_entry_limit {}",
                    graph.max_cached_graphs
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_slots Persistent ROCm graph owner slots by live assignment state.\n");
            out.push_str("# TYPE kiln_rocm_graph_slots gauge\n");
            for (state, count) in [
                ("total", graph.graph_slot_count),
                ("active", graph.active_graph_slot_count),
                ("idle", graph.idle_graph_slot_count),
            ] {
                push_line(
                    &mut out,
                    &format!("kiln_rocm_graph_slots{{state=\"{state}\"}} {count}"),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_tracked_decode_owners Decode continuity timelines currently retained by the ROCm graph runner.\n");
            out.push_str("# TYPE kiln_rocm_graph_tracked_decode_owners gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_tracked_decode_owners {}",
                    graph.tracked_decode_owner_count
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_owner_lifecycle_total ROCm graph owner and persistent-slot lifecycle events.\n");
            out.push_str("# TYPE kiln_rocm_graph_owner_lifecycle_total counter\n");
            for (event, count) in [
                ("decode_release", graph.decode_owner_release_count),
                ("graph_release", graph.decode_owner_graph_release_count),
                ("slot_create", graph.graph_slot_create_count),
                ("slot_reuse", graph.graph_slot_reuse_count),
            ] {
                push_line(
                    &mut out,
                    &format!("kiln_rocm_graph_owner_lifecycle_total{{event=\"{event}\"}} {count}"),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_retained_bytes Deduplicated requested physical bytes retained by ROCm graph resources.\n");
            out.push_str("# TYPE kiln_rocm_graph_retained_bytes gauge\n");
            for (kind, bytes) in [
                ("stable_io", graph.retained_stable_io_bytes),
                ("capture_arena", graph.retained_capture_arena_bytes),
                ("blaslt_workspace", graph.retained_blaslt_workspace_bytes),
                ("slot_state", graph.retained_slot_state_bytes),
                ("total", graph.retained_bytes),
                ("peak", graph.peak_retained_bytes),
                ("quarantined", graph.quarantined_retained_bytes),
            ] {
                push_line(
                    &mut out,
                    &format!("kiln_rocm_graph_retained_bytes{{kind=\"{kind}\"}} {bytes}"),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_retained_byte_limit Configured requested-physical-byte limit for retained ROCm graph resources.\n");
            out.push_str("# TYPE kiln_rocm_graph_retained_byte_limit gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_retained_byte_limit {}",
                    graph.max_retained_bytes
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_retained_byte_accounting_complete Whether every retained tensor mapped to exact ROCm allocation metadata.\n");
            out.push_str("# TYPE kiln_rocm_graph_retained_byte_accounting_complete gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_retained_byte_accounting_complete {}",
                    u8::from(graph.retained_bytes_accounting_complete)
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_opaque_native_objects HIP graph, executable, stream, and event objects whose driver bytes are not queryable.\n");
            out.push_str("# TYPE kiln_rocm_graph_opaque_native_objects gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_opaque_native_objects {}",
                    graph.opaque_native_object_count
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_cache_admissions_total Successful admissions into the bounded ROCm graph cache.\n");
            out.push_str("# TYPE kiln_rocm_graph_cache_admissions_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_cache_admissions_total {}",
                    graph.cache_admission_successes
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_cache_evictions_total Native graph entries released after safe device settlement.\n");
            out.push_str("# TYPE kiln_rocm_graph_cache_evictions_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_cache_evictions_total {}",
                    graph.cache_evictions
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_cache_evicted_bytes_total Requested physical bytes released by successful graph-cache evictions.\n");
            out.push_str("# TYPE kiln_rocm_graph_cache_evicted_bytes_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_cache_evicted_bytes_total {}",
                    graph.cache_evicted_bytes
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_cache_evictions_by_cause_total Graph-cache evictions by the closed ownership-removal cause.\n");
            out.push_str("# TYPE kiln_rocm_graph_cache_evictions_by_cause_total counter\n");
            for (cause, count) in [
                ("budget", graph.budget_evictions),
                ("pressure", graph.pressure_evictions),
                ("invalidation", graph.invalidation_evictions),
                ("recovery", graph.recovery_evictions),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_cache_evictions_by_cause_total{{cause=\"{cause}\"}} {count}"
                    ),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_cache_admission_rejections_total Successfully launched candidate graphs rejected by exact post-capture admission.\n");
            out.push_str("# TYPE kiln_rocm_graph_cache_admission_rejections_total counter\n");
            for (reason, count) in [
                ("entry_capacity", graph.entry_capacity_rejections),
                ("byte_budget", graph.byte_budget_rejections),
                (
                    "accounting_incomplete",
                    graph.accounting_incomplete_rejections,
                ),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_cache_admission_rejections_total{{reason=\"{reason}\"}} {count}"
                    ),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_pre_capture_skips_total Candidate captures skipped before native capture by the closed policy, accounting, capacity, or memory-governor reason.\n");
            out.push_str("# TYPE kiln_rocm_graph_pre_capture_skips_total counter\n");
            for (reason, count) in [
                ("entry_capacity", graph.pre_capture_entry_capacity_skips),
                ("byte_budget", graph.pre_capture_byte_budget_skips),
                (
                    "accounting_incomplete",
                    graph.pre_capture_accounting_incomplete_skips,
                ),
                (
                    "memory_reservation_denied",
                    graph.pre_capture_memory_reservation_denied_skips,
                ),
                (
                    "memory_governor_selector_mismatch",
                    graph.memory_governor_selector_mismatch_skips,
                ),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_pre_capture_skips_total{{reason=\"{reason}\"}} {count}"
                    ),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_capture_attempts_total ROCm graph capture state-machine attempts, including candidates deferred before native capture.\n");
            out.push_str("# TYPE kiln_rocm_graph_capture_attempts_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_capture_attempts_total {}",
                    graph.capture_attempts
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_capture_outcomes_total ROCm graph capture state-machine outcomes, including pre-native deferrals.\n");
            out.push_str("# TYPE kiln_rocm_graph_capture_outcomes_total counter\n");
            for (outcome, count) in [
                ("success", graph.capture_successes),
                ("deferred", graph.capture_deferrals),
                ("failure", graph.capture_failures),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_capture_outcomes_total{{outcome=\"{outcome}\"}} {count}"
                    ),
                );
            }
            out.push_str(
                "# HELP kiln_rocm_graph_replay_attempts_total Native ROCm graph replay attempts.\n",
            );
            out.push_str("# TYPE kiln_rocm_graph_replay_attempts_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_replay_attempts_total {}",
                    graph.replay_attempts
                ),
            );
            out.push_str(
                "# HELP kiln_rocm_graph_replay_outcomes_total Native ROCm graph replay outcomes.\n",
            );
            out.push_str("# TYPE kiln_rocm_graph_replay_outcomes_total counter\n");
            for (outcome, count) in [
                ("success", graph.replay_successes),
                ("failure", graph.replay_failures),
            ] {
                push_line(
                    &mut out,
                    &format!(
                        "kiln_rocm_graph_replay_outcomes_total{{outcome=\"{outcome}\"}} {count}"
                    ),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_fallbacks_total Eager ROCm graph fallbacks by closed reason.\n");
            out.push_str("# TYPE kiln_rocm_graph_fallbacks_total counter\n");
            for (reason, count) in [
                (
                    "multi_row_batch_unsupported",
                    graph.fallbacks.multi_row_batch_unsupported,
                ),
                (
                    "cold_cache_host_round_trip",
                    graph.fallbacks.cold_cache_host_round_trip,
                ),
                (
                    "persistent_host_round_trip",
                    graph.fallbacks.persistent_host_round_trip,
                ),
                (
                    "shape_dependent_attention",
                    graph.fallbacks.shape_dependent_attention,
                ),
                ("graph_cache_capacity", graph.fallbacks.graph_cache_capacity),
                (
                    "graph_cache_byte_budget",
                    graph.fallbacks.graph_cache_byte_budget,
                ),
                (
                    "graph_accounting_incomplete",
                    graph.fallbacks.graph_accounting_incomplete,
                ),
                (
                    "moderate_memory_pressure",
                    graph.fallbacks.moderate_memory_pressure,
                ),
                (
                    "tight_memory_pressure",
                    graph.fallbacks.tight_memory_pressure,
                ),
                (
                    "critical_memory_pressure",
                    graph.fallbacks.critical_memory_pressure,
                ),
                (
                    "memory_reservation_denied",
                    graph.fallbacks.memory_reservation_denied,
                ),
                (
                    "memory_governor_selector_mismatch",
                    graph.fallbacks.memory_governor_selector_mismatch,
                ),
                ("capture_failure", graph.fallbacks.capture_failure),
                ("replay_failure", graph.fallbacks.replay_failure),
            ] {
                push_line(
                    &mut out,
                    &format!("kiln_rocm_graph_fallbacks_total{{reason=\"{reason}\"}} {count}"),
                );
            }
            out.push_str("# HELP kiln_rocm_graph_fallback_slow_total Eager ROCm graph fallbacks whose complete fallback path took at least 100 ms.\n");
            out.push_str("# TYPE kiln_rocm_graph_fallback_slow_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_fallback_slow_total {}",
                    graph.fallbacks.slow
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_fallback_duration_seconds_total Accumulated end-to-end eager ROCm graph fallback duration.\n");
            out.push_str("# TYPE kiln_rocm_graph_fallback_duration_seconds_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_fallback_duration_seconds_total {:.6}",
                    graph.fallbacks.total_duration_micros as f64 / 1_000_000.0
                ),
            );
            out.push_str("# HELP kiln_rocm_graph_fallback_duration_seconds_max Longest observed end-to-end eager ROCm graph fallback duration.\n");
            out.push_str("# TYPE kiln_rocm_graph_fallback_duration_seconds_max gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_rocm_graph_fallback_duration_seconds_max {:.6}",
                    graph.fallbacks.max_duration_micros as f64 / 1_000_000.0
                ),
            );
        }
        prom_counter(
            &mut out,
            "kiln_requests_total",
            "status",
            "error",
            self.requests_error.load(Ordering::Relaxed),
        );
        prom_counter(
            &mut out,
            "kiln_requests_total",
            "status",
            "timeout",
            self.requests_timeout.load(Ordering::Relaxed),
        );
        prom_counter(
            &mut out,
            "kiln_requests_total",
            "status",
            "rejected",
            self.requests_rejected.load(Ordering::Relaxed),
        );

        out.push_str("# HELP kiln_request_duration_seconds Request latency.\n");
        out.push_str("# TYPE kiln_request_duration_seconds histogram\n");
        let count = self.request_duration_count.load(Ordering::Relaxed);
        let sum_us = self.request_duration_sum_us.load(Ordering::Relaxed);
        render_histogram_buckets(
            &mut out,
            "kiln_request_duration_seconds",
            &REQUEST_LATENCY_BUCKETS_SECONDS,
            &self.request_duration_buckets,
        );
        push_line(
            &mut out,
            &format!("kiln_request_duration_seconds_count {count}"),
        );
        push_line(
            &mut out,
            &format!(
                "kiln_request_duration_seconds_sum {:.6}",
                sum_us as f64 / 1_000_000.0
            ),
        );

        out.push_str(
            "# HELP kiln_request_prefill_duration_seconds Model prefill latency before decode.\n",
        );
        out.push_str("# TYPE kiln_request_prefill_duration_seconds histogram\n");
        let prefill_count = self.prefill_duration_count.load(Ordering::Relaxed);
        let prefill_sum_us = self.prefill_duration_sum_us.load(Ordering::Relaxed);
        render_histogram_buckets(
            &mut out,
            "kiln_request_prefill_duration_seconds",
            &LATENCY_BUCKETS_SECONDS,
            &self.prefill_duration_buckets,
        );
        push_line(
            &mut out,
            &format!("kiln_request_prefill_duration_seconds_count {prefill_count}"),
        );
        push_line(
            &mut out,
            &format!(
                "kiln_request_prefill_duration_seconds_sum {:.6}",
                prefill_sum_us as f64 / 1_000_000.0
            ),
        );

        out.push_str(
            "# HELP kiln_request_decode_duration_seconds Model decode latency after prefill.\n",
        );
        out.push_str("# TYPE kiln_request_decode_duration_seconds histogram\n");
        let decode_count = self.decode_duration_count.load(Ordering::Relaxed);
        let decode_sum_us = self.decode_duration_sum_us.load(Ordering::Relaxed);
        render_histogram_buckets(
            &mut out,
            "kiln_request_decode_duration_seconds",
            &LATENCY_BUCKETS_SECONDS,
            &self.decode_duration_buckets,
        );
        push_line(
            &mut out,
            &format!("kiln_request_decode_duration_seconds_count {decode_count}"),
        );
        push_line(
            &mut out,
            &format!(
                "kiln_request_decode_duration_seconds_sum {:.6}",
                decode_sum_us as f64 / 1_000_000.0
            ),
        );

        out.push_str("# HELP kiln_request_ttft_seconds Time from request receipt until the first model token became ready.\n");
        out.push_str("# TYPE kiln_request_ttft_seconds histogram\n");
        render_atomic_latency_histogram(
            &mut out,
            "kiln_request_ttft_seconds",
            None,
            &self.request_ttft,
        );

        out.push_str("# HELP kiln_token_itl_seconds Request-local inter-token latency; concurrent requests are never cross-paired.\n");
        out.push_str("# TYPE kiln_token_itl_seconds histogram\n");
        render_atomic_latency_histogram(&mut out, "kiln_token_itl_seconds", None, &self.token_itl);

        out.push_str("# HELP kiln_request_latency_phase_seconds Observed request phase duration. Missing unsupported subphases emit no sample.\n");
        out.push_str("# TYPE kiln_request_latency_phase_seconds histogram\n");
        for (phase, histogram) in LATENCY_PHASES.iter().zip(&self.latency_phases) {
            render_atomic_latency_histogram(
                &mut out,
                "kiln_request_latency_phase_seconds",
                Some(("phase", phase)),
                histogram,
            );
        }

        out.push_str("# HELP kiln_token_stalls_total Request-local token gaps at or above the fixed 250ms absolute stall floor, by bounded dominant reason.\n");
        out.push_str("# TYPE kiln_token_stalls_total counter\n");
        for reason in LatencyStallReason::ALL {
            prom_counter(
                &mut out,
                "kiln_token_stalls_total",
                "reason",
                reason.as_str(),
                self.token_stall_reasons[reason.index()].load(Ordering::Relaxed),
            );
        }

        out.push_str("# HELP kiln_thinking_budget_source_total Recorded chat completion thinking-budget provenance by dimension.\n");
        out.push_str("# TYPE kiln_thinking_budget_source_total counter\n");
        for (index, source) in THINKING_BUDGET_SOURCES.iter().enumerate() {
            prom_counter2(
                &mut out,
                "kiln_thinking_budget_source_total",
                "dimension",
                "tokens",
                "source",
                source.as_str(),
                self.thinking_budget_token_sources[index].load(Ordering::Relaxed),
            );
            prom_counter2(
                &mut out,
                "kiln_thinking_budget_source_total",
                "dimension",
                "time",
                "source",
                source.as_str(),
                self.thinking_budget_time_sources[index].load(Ordering::Relaxed),
            );
        }

        out.push_str("# HELP kiln_thinking_budget_outcomes_total Recorded chat completion thinking-budget outcomes.\n");
        out.push_str("# TYPE kiln_thinking_budget_outcomes_total counter\n");
        for (index, outcome) in THINKING_BUDGET_OUTCOMES.iter().enumerate() {
            prom_counter(
                &mut out,
                "kiln_thinking_budget_outcomes_total",
                "outcome",
                outcome,
                self.thinking_budget_outcomes[index].load(Ordering::Relaxed),
            );
        }

        out.push_str("# HELP kiln_thinking_budget_effective_tokens Effective configured thinking-token limits on recorded chat completions.\n");
        out.push_str("# TYPE kiln_thinking_budget_effective_tokens histogram\n");
        render_integer_histogram_buckets(
            &mut out,
            "kiln_thinking_budget_effective_tokens",
            &THINKING_BUDGET_TOKEN_BUCKETS,
            &self.thinking_budget_token_limit_buckets,
        );
        push_line(
            &mut out,
            &format!(
                "kiln_thinking_budget_effective_tokens_count {}",
                self.thinking_budget_token_limit_count
                    .load(Ordering::Relaxed)
            ),
        );
        push_line(
            &mut out,
            &format!(
                "kiln_thinking_budget_effective_tokens_sum {}",
                self.thinking_budget_token_limit_sum.load(Ordering::Relaxed)
            ),
        );

        out.push_str("# HELP kiln_thinking_budget_effective_seconds Effective configured thinking-time limits on recorded chat completions.\n");
        out.push_str("# TYPE kiln_thinking_budget_effective_seconds histogram\n");
        render_millisecond_histogram_buckets_as_seconds(
            &mut out,
            "kiln_thinking_budget_effective_seconds",
            &THINKING_BUDGET_TIME_BUCKETS_MS,
            &self.thinking_budget_time_limit_buckets,
        );
        push_line(
            &mut out,
            &format!(
                "kiln_thinking_budget_effective_seconds_count {}",
                self.thinking_budget_time_limit_count
                    .load(Ordering::Relaxed)
            ),
        );
        push_line(
            &mut out,
            &format!(
                "kiln_thinking_budget_effective_seconds_sum {:.6}",
                self.thinking_budget_time_limit_sum_ms
                    .load(Ordering::Relaxed) as f64
                    / 1_000.0
            ),
        );

        out.push_str("# HELP kiln_tokens_generated_total Total tokens generated.\n");
        out.push_str("# TYPE kiln_tokens_generated_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_tokens_generated_total {}",
                self.tokens_generated.load(Ordering::Relaxed)
            ),
        );

        out.push_str(
            "# HELP kiln_rendered_prompt_cache_lookups_total Rendered prompt cache lookups.\n",
        );
        out.push_str("# TYPE kiln_rendered_prompt_cache_lookups_total counter\n");
        prom_counter(
            &mut out,
            "kiln_rendered_prompt_cache_lookups_total",
            "result",
            "hit",
            gauges.rendered_prompt_cache_hits,
        );
        prom_counter(
            &mut out,
            "kiln_rendered_prompt_cache_lookups_total",
            "result",
            "miss",
            gauges.rendered_prompt_cache_misses,
        );

        out.push_str("# HELP kiln_rendered_prompt_cache_entries Rendered prompt cache entries.\n");
        out.push_str("# TYPE kiln_rendered_prompt_cache_entries gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_rendered_prompt_cache_entries {}",
                gauges.rendered_prompt_cache_entries
            ),
        );

        out.push_str(
            "# HELP kiln_prompt_token_cache_lookups_total Rendered prompt token cache lookups.\n",
        );
        out.push_str("# TYPE kiln_prompt_token_cache_lookups_total counter\n");
        prom_counter(
            &mut out,
            "kiln_prompt_token_cache_lookups_total",
            "result",
            "hit",
            gauges.prompt_token_cache_hits,
        );
        prom_counter(
            &mut out,
            "kiln_prompt_token_cache_lookups_total",
            "result",
            "miss",
            gauges.prompt_token_cache_misses,
        );

        out.push_str(
            "# HELP kiln_prompt_token_cache_entries Rendered prompt token cache entries.\n",
        );
        out.push_str("# TYPE kiln_prompt_token_cache_entries gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prompt_token_cache_entries {}",
                gauges.prompt_token_cache_entries
            ),
        );

        out.push_str("# HELP kiln_active_requests Currently in-flight requests.\n");
        out.push_str("# TYPE kiln_active_requests gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_active_requests {}",
                self.active_requests.load(Ordering::Relaxed)
            ),
        );

        out.push_str(
            "# HELP kiln_active_requests_peak Peak in-flight requests since process start.\n",
        );
        out.push_str("# TYPE kiln_active_requests_peak gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_active_requests_peak {}",
                self.active_requests_peak.load(Ordering::Relaxed)
            ),
        );

        out.push_str("# HELP kiln_request_prefill_tokens_completed Prompt/prefix prefill tokens completed by currently in-flight requests.\n");
        out.push_str("# TYPE kiln_request_prefill_tokens_completed gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_request_prefill_tokens_completed {}",
                self.request_prefill_tokens_completed
                    .load(Ordering::Relaxed)
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_enabled Whether the live greedy decode batcher is enabled.\n");
        out.push_str("# TYPE kiln_decode_batcher_enabled gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_enabled {}",
                if gauges.decode_batcher_enabled { 1 } else { 0 }
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_jobs_total Live greedy decode batcher jobs.\n");
        out.push_str("# TYPE kiln_decode_batcher_jobs_total counter\n");
        prom_counter(
            &mut out,
            "kiln_decode_batcher_jobs_total",
            "result",
            "submitted",
            gauges.decode_batcher.submitted_jobs as u64,
        );
        prom_counter(
            &mut out,
            "kiln_decode_batcher_jobs_total",
            "result",
            "runner_busy",
            gauges.decode_batcher.runner_busy_jobs as u64,
        );
        prom_counter(
            &mut out,
            "kiln_decode_batcher_jobs_total",
            "result",
            "failed",
            gauges.decode_batcher.failed_jobs as u64,
        );

        out.push_str("# HELP kiln_decode_batcher_batches_total Live greedy decode batches executed by the rendezvous worker.\n");
        out.push_str("# TYPE kiln_decode_batcher_batches_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_batches_total {}",
                gauges.decode_batcher.executed_batches
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_rows_total Live greedy decode rows executed by the rendezvous worker.\n");
        out.push_str("# TYPE kiln_decode_batcher_rows_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_rows_total {}",
                gauges.decode_batcher.executed_rows
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_runner_calls_total ModelRunner decode calls issued by the live greedy decode batcher, including rowwise retry attempts.\n");
        out.push_str("# TYPE kiln_decode_batcher_runner_calls_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_runner_calls_total {}",
                gauges.decode_batcher.runner_calls
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_runner_calls_per_token ModelRunner decode calls per live greedy decode token row; lower than 1.0 means batching amortized calls across rows.\n");
        out.push_str("# TYPE kiln_decode_batcher_runner_calls_per_token gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_runner_calls_per_token {:.6}",
                gauges
                    .decode_batcher
                    .runner_calls_per_token()
                    .unwrap_or(0.0)
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_max_runner_calls_per_token Maximum ModelRunner decode calls any token row observed in one live greedy decode worker batch.\n");
        out.push_str("# TYPE kiln_decode_batcher_max_runner_calls_per_token gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_max_runner_calls_per_token {}",
                gauges.decode_batcher.max_runner_calls_per_token
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_runner_call_budget_per_token Phase 8 sentinel budget for maximum ModelRunner decode calls per live greedy decode token row.\n");
        out.push_str("# TYPE kiln_decode_batcher_runner_call_budget_per_token gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_runner_call_budget_per_token {}",
                gauges.decode_batcher.runner_call_budget_per_token()
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_runner_call_budget_exceeded Whether the observed max runner calls per token exceeded the Phase 8 sentinel budget.\n");
        out.push_str("# TYPE kiln_decode_batcher_runner_call_budget_exceeded gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_runner_call_budget_exceeded {}",
                usize::from(gauges.decode_batcher.runner_call_budget_exceeded())
            ),
        );

        out.push_str("# HELP kiln_decode_batcher_max_observed_batch Largest live greedy decode batch observed since process start.\n");
        out.push_str("# TYPE kiln_decode_batcher_max_observed_batch gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_decode_batcher_max_observed_batch {}",
                gauges.decode_batcher.max_observed_batch
            ),
        );

        out.push_str("# HELP kiln_batching_engine_enabled Whether the real-model batching engine actor is enabled.\n");
        out.push_str("# TYPE kiln_batching_engine_enabled gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_enabled {}",
                if gauges.batching_engine_enabled { 1 } else { 0 }
            ),
        );

        out.push_str("# HELP kiln_batching_engine_snapshot_age_seconds Seconds since the batching actor last published its cached control-plane snapshot. This increases while a decode forward or another long actor operation is in flight.\n");
        out.push_str("# TYPE kiln_batching_engine_snapshot_age_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_snapshot_age_seconds {:.6}",
                gauges.batching_engine.snapshot_age_ms as f64 / 1_000.0
            ),
        );

        out.push_str("# HELP kiln_batching_engine_actor_cycle_idle_configured_seconds Configured cooperative safe-boundary idle after batching actor cycles that advanced model work.\n");
        out.push_str("# TYPE kiln_batching_engine_actor_cycle_idle_configured_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_actor_cycle_idle_configured_seconds {:.6}",
                gauges.batching_engine.actor_cycle_idle_ms as f64 / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_actor_cycle_idle_active Whether the batching actor is currently inside a cooperative cycle-idle wait.\n");
        out.push_str("# TYPE kiln_batching_engine_actor_cycle_idle_active gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_actor_cycle_idle_active {}",
                u8::from(gauges.batching_engine.actor_cycle_idle_active)
            ),
        );
        out.push_str("# HELP kiln_batching_engine_actor_cycle_idles_total Cooperative safe-boundary waits entered since actor startup.\n");
        out.push_str("# TYPE kiln_batching_engine_actor_cycle_idles_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_actor_cycle_idles_total {}",
                gauges.batching_engine.actor_cycle_idle_count
            ),
        );
        out.push_str("# HELP kiln_batching_engine_actor_cycle_idle_seconds_total Cumulative observed cooperative cycle-idle wall time.\n");
        out.push_str("# TYPE kiln_batching_engine_actor_cycle_idle_seconds_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_actor_cycle_idle_seconds_total {:.6}",
                gauges.batching_engine.total_actor_cycle_idle_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_actor_cycle_idle_max_seconds Largest observed cooperative cycle-idle wall time since actor startup.\n");
        out.push_str("# TYPE kiln_batching_engine_actor_cycle_idle_max_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_actor_cycle_idle_max_seconds {:.6}",
                gauges.batching_engine.max_actor_cycle_idle_ms / 1_000.0
            ),
        );

        out.push_str("# HELP kiln_batching_engine_queue_depth Requests waiting inside the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_queue_depth gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_queue_depth {}",
                gauges.batching_engine.queue_depth
            ),
        );

        out.push_str("# HELP kiln_batching_engine_active_decode Requests currently active inside the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_active_decode gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_active_decode {}",
                gauges.batching_engine.active_decode
            ),
        );

        out.push_str("# HELP kiln_batching_engine_active_prefill Requests retaining partial resumable-prefill state.\n");
        out.push_str("# TYPE kiln_batching_engine_active_prefill gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_active_prefill {}",
                gauges.batching_engine.active_prefill
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefix_cache_enabled Whether cross-request prompt, KV, and recurrent-state prefix reuse is correctness-qualified and admitted.\n");
        out.push_str("# TYPE kiln_batching_engine_prefix_cache_enabled gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefix_cache_enabled {}",
                u8::from(gauges.batching_engine.prefix_cache_enabled)
            ),
        );

        out.push_str("# HELP kiln_batching_engine_resident_prefill_enabled Whether native resident token-prefill is correctness-qualified and admitted.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_enabled gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_enabled {}",
                u8::from(gauges.batching_engine.resident_prefill_enabled)
            ),
        );

        out.push_str("# HELP kiln_batching_engine_active_resident_prefill Prefill rows whose newest KV positions require the resident Vulkan route.\n");
        out.push_str("# TYPE kiln_batching_engine_active_resident_prefill gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_active_resident_prefill {}",
                gauges.batching_engine.active_resident_prefill
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_batch_tokens Effective combined decode-plus-prefill token budget per actor cycle.\n");
        out.push_str("# TYPE kiln_batching_engine_max_batch_tokens gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_batch_tokens {}",
                gauges.batching_engine.max_batch_tokens
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_prefill_tokens_per_cycle Effective prompt-token ceiling between decode cohorts.\n");
        out.push_str("# TYPE kiln_batching_engine_max_prefill_tokens_per_cycle gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_prefill_tokens_per_cycle {}",
                gauges.batching_engine.max_prefill_tokens_per_cycle
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_prefill_layers_per_cycle Effective transformer-layer ceiling between decode cohorts.\n");
        out.push_str("# TYPE kiln_batching_engine_max_prefill_layers_per_cycle gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_prefill_layers_per_cycle {}",
                gauges.batching_engine.max_prefill_layers_per_cycle
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_admission_quantum Maximum queued requests prefilled before yielding to decode.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_admission_quantum gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_admission_quantum {}",
                gauges.batching_engine.max_prefill_admission_quantum
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_staging_slots Bounded short-prefill slots beyond the ordinary decode-width slots.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_staging_slots gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_staging_slots {}",
                gauges.batching_engine.max_prefill_staging_slots
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_active_requests Total ordinary plus short-prefill staging capacity.\n");
        out.push_str("# TYPE kiln_batching_engine_max_active_requests gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_active_requests {}",
                gauges.batching_engine.max_active_requests
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_staging_priority_burst Maximum staged-priority turns before a mandatory global prefill turn.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_staging_priority_burst gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_staging_priority_burst {}",
                gauges.batching_engine.max_prefill_staging_priority_burst
            ),
        );

        out.push_str("# HELP kiln_batching_engine_active_staged_requests Requests currently owning a short-prefill staging slot.\n");
        out.push_str("# TYPE kiln_batching_engine_active_staged_requests gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_active_staged_requests {}",
                gauges.batching_engine.active_staged_requests
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_observed_active_requests Largest total active set observed since process start.\n");
        out.push_str("# TYPE kiln_batching_engine_max_observed_active_requests gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_observed_active_requests {}",
                gauges.batching_engine.max_observed_active_requests
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_decode_batch Effective concurrent decode-row ceiling after startup policy and token-budget constraints.\n");
        out.push_str("# TYPE kiln_batching_engine_max_decode_batch gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_decode_batch {}",
                gauges.batching_engine.max_decode_batch
            ),
        );

        out.push_str("# HELP kiln_batching_engine_last_batch_size Last decode batch size selected by the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_last_batch_size gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_last_batch_size {}",
                gauges.batching_engine.last_batch_size
            ),
        );

        out.push_str("# HELP kiln_batching_engine_max_observed_batch Largest decode batch selected by the real-model batching engine since process start.\n");
        out.push_str("# TYPE kiln_batching_engine_max_observed_batch gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_max_observed_batch {}",
                gauges.batching_engine.max_observed_batch_size
            ),
        );

        out.push_str("# HELP kiln_batching_engine_last_forward_ms Last decode forward wall time in milliseconds.\n");
        out.push_str("# TYPE kiln_batching_engine_last_forward_ms gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_last_forward_ms {:.6}",
                gauges.batching_engine.last_forward_ms
            ),
        );

        out.push_str("# HELP kiln_batching_engine_decode_forward_seconds_total Cumulative wall time spent inside decode forwards.\n");
        out.push_str("# TYPE kiln_batching_engine_decode_forward_seconds_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_decode_forward_seconds_total {:.6}",
                gauges.batching_engine.total_decode_forward_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_decode_forward_max_seconds Maximum decode-forward wall time observed since process start.\n");
        out.push_str("# TYPE kiln_batching_engine_decode_forward_max_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_decode_forward_max_seconds {:.6}",
                gauges.batching_engine.max_decode_forward_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_slow_decode_forwards_total Decode forwards taking at least 100 milliseconds.\n");
        out.push_str("# TYPE kiln_batching_engine_slow_decode_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_slow_decode_forwards_total {}",
                gauges.batching_engine.slow_decode_forward_count
            ),
        );

        out.push_str(
            "# HELP kiln_batching_engine_last_prefill_ms Last prefill wall time in milliseconds.\n",
        );
        out.push_str("# TYPE kiln_batching_engine_last_prefill_ms gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_last_prefill_ms {:.6}",
                gauges.batching_engine.last_prefill_ms
            ),
        );

        out.push_str("# HELP kiln_batching_engine_last_prefill_tokens Prompt tokens processed by the most recent bounded prefill forward.\n");
        out.push_str("# TYPE kiln_batching_engine_last_prefill_tokens gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_last_prefill_tokens {}",
                gauges.batching_engine.last_prefill_tokens
            ),
        );

        out.push_str("# HELP kiln_batching_engine_last_prefill_layers Transformer layers processed by the most recent bounded prefill forward.\n");
        out.push_str("# TYPE kiln_batching_engine_last_prefill_layers gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_last_prefill_layers {}",
                gauges.batching_engine.last_prefill_layers
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_forward_seconds_total Cumulative wall time spent inside bounded prefill forwards.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_forward_seconds_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_forward_seconds_total {:.6}",
                gauges.batching_engine.total_prefill_forward_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_prefill_forward_max_seconds Maximum bounded-prefill wall time observed since process start.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_forward_max_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_forward_max_seconds {:.6}",
                gauges.batching_engine.max_prefill_forward_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_slow_prefill_forwards_total Bounded prefill forwards taking at least 100 milliseconds.\n");
        out.push_str("# TYPE kiln_batching_engine_slow_prefill_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_slow_prefill_forwards_total {}",
                gauges.batching_engine.slow_prefill_forward_count
            ),
        );

        out.push_str("# HELP kiln_batching_engine_last_admission_seconds Wall time of the most recent request-admission preparation call.\n");
        out.push_str("# TYPE kiln_batching_engine_last_admission_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_last_admission_seconds {:.6}",
                gauges.batching_engine.last_admission_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_admission_seconds_total Cumulative wall time spent preparing admitted requests.\n");
        out.push_str("# TYPE kiln_batching_engine_admission_seconds_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_admission_seconds_total {:.6}",
                gauges.batching_engine.total_admission_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_admission_max_seconds Maximum request-admission preparation wall time observed since process start.\n");
        out.push_str("# TYPE kiln_batching_engine_admission_max_seconds gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_admission_max_seconds {:.6}",
                gauges.batching_engine.max_admission_ms / 1_000.0
            ),
        );
        out.push_str("# HELP kiln_batching_engine_admission_calls_total Request-admission preparation calls.\n");
        out.push_str("# TYPE kiln_batching_engine_admission_calls_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_admission_calls_total {}",
                gauges.batching_engine.total_admission_calls
            ),
        );
        out.push_str("# HELP kiln_batching_engine_slow_admissions_total Request-admission preparation calls taking at least 100 milliseconds.\n");
        out.push_str("# TYPE kiln_batching_engine_slow_admissions_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_slow_admissions_total {}",
                gauges.batching_engine.slow_admission_count
            ),
        );

        out.push_str("# HELP kiln_batching_engine_decode_forwards_total Decode forward calls issued by the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_decode_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_decode_forwards_total {}",
                gauges.batching_engine.total_decode_forwards
            ),
        );

        out.push_str("# HELP kiln_batching_engine_batched_decode_forwards_total Decode forward calls issued with batch size greater than one.\n");
        out.push_str("# TYPE kiln_batching_engine_batched_decode_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_batched_decode_forwards_total {}",
                gauges.batching_engine.total_batched_decode_forwards
            ),
        );

        out.push_str("# HELP kiln_batching_engine_decode_rows_total Decode rows submitted by the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_decode_rows_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_decode_rows_total {}",
                gauges.batching_engine.total_decode_rows
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_admission_cycles_total Prefill admission rounds that admitted at least one request.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_admission_cycles_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_admission_cycles_total {}",
                gauges.batching_engine.total_prefill_admission_cycles
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_forwards_total Bounded prompt-forward quanta issued by the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_forwards_total {}",
                gauges.batching_engine.total_prefill_forwards
            ),
        );

        out.push_str("# HELP kiln_batching_engine_resident_prefill_attempts_total Native resident token-prefill calls attempted by the batching actor.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_attempts_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_attempts_total {}",
                gauges.batching_engine.total_resident_prefill_attempts
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_forwards_total Successful native resident token-prefill forwards.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_forwards_total {}",
                gauges.batching_engine.total_resident_prefill_forwards
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_initial_declines_total Mutation-free native declines before any selected row entered resident authority.\n");
        out.push_str(
            "# TYPE kiln_batching_engine_resident_prefill_initial_declines_total counter\n",
        );
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_initial_declines_total {}",
                gauges
                    .batching_engine
                    .total_resident_prefill_initial_declines
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_route_failures_total Fail-closed native errors, post-entry declines, or progress-contract violations.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_route_failures_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_route_failures_total {}",
                gauges.batching_engine.total_resident_prefill_route_failures
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_rows_total Prompt rows advanced by successful native resident forwards.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_rows_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_rows_total {}",
                gauges.batching_engine.total_resident_prefill_rows
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_completed_rows_total Prompt rows completed by native resident forwards.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_completed_rows_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_completed_rows_total {}",
                gauges.batching_engine.total_resident_prefill_completed_rows
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_last_batch_size Rows in the most recent successful native resident prefill forward.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_last_batch_size gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_last_batch_size {}",
                gauges.batching_engine.last_resident_prefill_batch_size
            ),
        );
        out.push_str("# HELP kiln_batching_engine_resident_prefill_max_batch_size Largest successful native resident prefill cohort since process start.\n");
        out.push_str("# TYPE kiln_batching_engine_resident_prefill_max_batch_size gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_resident_prefill_max_batch_size {}",
                gauges.batching_engine.max_resident_prefill_batch_size
            ),
        );

        out.push_str("# HELP kiln_batching_engine_decode_tokens_total Decode tokens emitted by the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_decode_tokens_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_decode_tokens_total {}",
                gauges.batching_engine.total_decode_tokens
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_tokens_total Prompt tokens prefilled by the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_tokens_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_tokens_total {}",
                gauges.batching_engine.total_prefill_tokens
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_layers_total Transformer layers processed by bounded prompt forwards.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_layers_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_layers_total {}",
                gauges.batching_engine.total_prefill_layers
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_layer_yields_total Bounded prefill forwards that retained a token chunk and yielded between layers.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_layer_yields_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_layer_yields_total {}",
                gauges.batching_engine.total_prefill_layer_yields
            ),
        );

        out.push_str("# HELP kiln_batching_engine_short_prefill_priority_forwards_total Bounded short-tail prefill service opportunities used.\n");
        out.push_str("# TYPE kiln_batching_engine_short_prefill_priority_forwards_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_short_prefill_priority_forwards_total {}",
                gauges.batching_engine.total_short_prefill_priority_forwards
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_staging_priority_forwards_total Priority opportunities assigned to rotating staged prefills.\n");
        out.push_str(
            "# TYPE kiln_batching_engine_prefill_staging_priority_forwards_total counter\n",
        );
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_staging_priority_forwards_total {}",
                gauges
                    .batching_engine
                    .total_prefill_staging_priority_forwards
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefill_staging_admissions_total Requests admitted through the bounded short-prefill staging lane.\n");
        out.push_str("# TYPE kiln_batching_engine_prefill_staging_admissions_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefill_staging_admissions_total {}",
                gauges.batching_engine.total_prefill_staging_admissions
            ),
        );

        out.push_str(
            "# HELP kiln_batching_engine_errors_total Real-model batching engine errors.\n",
        );
        out.push_str("# TYPE kiln_batching_engine_errors_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_errors_total {}",
                gauges.batching_engine.total_errors
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_delivery_in_flight Requests with a response batch currently owned by the delivery worker.\n");
        out.push_str("# TYPE kiln_batching_engine_response_delivery_in_flight gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_delivery_in_flight {}",
                gauges.batching_engine.response_delivery_in_flight
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_delivery_backpressured Response batches currently waiting for capacity in a per-request response channel.\n");
        out.push_str("# TYPE kiln_batching_engine_response_delivery_backpressured gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_delivery_backpressured {}",
                gauges.batching_engine.response_delivery_backpressured
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_delivery_pending_terminal Terminal response batches awaiting ordered delivery.\n");
        out.push_str("# TYPE kiln_batching_engine_response_delivery_pending_terminal gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_delivery_pending_terminal {}",
                gauges.batching_engine.response_delivery_pending_terminal
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_backpressure_events_total Response batches that encountered a full per-request response channel.\n");
        out.push_str("# TYPE kiln_batching_engine_response_backpressure_events_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_backpressure_events_total {}",
                gauges.batching_engine.response_backpressure_events
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_backpressure_wait_seconds_total Cumulative time spent waiting for full per-request response channels to drain.\n");
        out.push_str(
            "# TYPE kiln_batching_engine_response_backpressure_wait_seconds_total counter\n",
        );
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_backpressure_wait_seconds_total {:.6}",
                gauges.batching_engine.response_backpressure_wait_ms as f64 / 1_000.0
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_stall_evictions_total Requests evicted after their response channel remained full for the configured grace window.\n");
        out.push_str("# TYPE kiln_batching_engine_response_stall_evictions_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_stall_evictions_total {}",
                gauges.batching_engine.response_stall_evictions
            ),
        );

        out.push_str("# HELP kiln_batching_engine_response_channel_closed_total Active requests discarded because their response receiver was closed.\n");
        out.push_str("# TYPE kiln_batching_engine_response_channel_closed_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_response_channel_closed_total {}",
                gauges.batching_engine.response_channel_closed
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefix_deferred_waiting Last strong-snapshot sample of queued requests held back because an active same-adapter request can become their reusable strict prefix. Cached control-plane reads do not rescan prefixes.\n");
        out.push_str("# TYPE kiln_batching_engine_prefix_deferred_waiting gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefix_deferred_waiting {}",
                gauges.batching_engine.prefix_deferred_waiting
            ),
        );

        out.push_str("# HELP kiln_batching_engine_prefix_admission_deferrals_total Prefix-aware admission deferral observations in the real-model batching engine.\n");
        out.push_str("# TYPE kiln_batching_engine_prefix_admission_deferrals_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batching_engine_prefix_admission_deferrals_total {}",
                gauges.batching_engine.prefix_admission_deferrals
            ),
        );

        // --- Scheduler ---
        out.push_str("# HELP kiln_scheduler_waiting Requests waiting to be scheduled.\n");
        out.push_str("# TYPE kiln_scheduler_waiting gauge\n");
        push_line(
            &mut out,
            &format!("kiln_scheduler_waiting {}", gauges.scheduler_waiting),
        );

        out.push_str("# HELP kiln_scheduler_running Requests currently generating.\n");
        out.push_str("# TYPE kiln_scheduler_running gauge\n");
        push_line(
            &mut out,
            &format!("kiln_scheduler_running {}", gauges.scheduler_running),
        );

        out.push_str("# HELP kiln_blocks_used KV cache blocks in use.\n");
        out.push_str("# TYPE kiln_blocks_used gauge\n");
        push_line(
            &mut out,
            &format!("kiln_blocks_used {}", gauges.blocks_used),
        );

        out.push_str("# HELP kiln_blocks_total Total KV cache blocks.\n");
        out.push_str("# TYPE kiln_blocks_total gauge\n");
        push_line(
            &mut out,
            &format!("kiln_blocks_total {}", gauges.blocks_total),
        );

        // --- GPU Memory ---
        out.push_str("# HELP kiln_vram_total_bytes Total GPU VRAM.\n");
        out.push_str("# TYPE kiln_vram_total_bytes gauge\n");
        push_line(
            &mut out,
            &format!("kiln_vram_total_bytes {}", gauges.vram_total),
        );

        out.push_str(
            "# HELP kiln_vram_model_bytes Model/device residency used for memory budgeting.\n",
        );
        out.push_str("# TYPE kiln_vram_model_bytes gauge\n");
        push_line(
            &mut out,
            &format!("kiln_vram_model_bytes {}", gauges.vram_model),
        );

        out.push_str(
            "# HELP kiln_vram_model_estimated_bytes Static model parameter memory estimate.\n",
        );
        out.push_str("# TYPE kiln_vram_model_estimated_bytes gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_vram_model_estimated_bytes {}",
                gauges.vram_model_estimated
            ),
        );

        out.push_str("# HELP kiln_vram_post_load_used_bytes CUDA memory.used snapshot after model load before KV allocation.\n");
        out.push_str("# TYPE kiln_vram_post_load_used_bytes gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_vram_post_load_used_bytes {}",
                gauges.vram_post_load_used
            ),
        );

        out.push_str("# HELP kiln_vram_prefill_peak_used_bytes Highest CUDA memory.used observed immediately after prefill/generation boundaries.\n");
        out.push_str("# TYPE kiln_vram_prefill_peak_used_bytes gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_vram_prefill_peak_used_bytes {}",
                gauges.vram_prefill_peak_used
            ),
        );

        out.push_str("# HELP kiln_vram_kv_cache_bytes KV cache memory.\n");
        out.push_str("# TYPE kiln_vram_kv_cache_bytes gauge\n");
        push_line(
            &mut out,
            &format!("kiln_vram_kv_cache_bytes {}", gauges.vram_kv_cache),
        );

        out.push_str("# HELP kiln_vram_training_budget_bytes Training memory budget.\n");
        out.push_str("# TYPE kiln_vram_training_budget_bytes gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_vram_training_budget_bytes {}",
                gauges.vram_training_budget
            ),
        );

        if let Some(stats) = gauges.vulkan_buffers {
            out.push_str("# HELP kiln_vulkan_buffer_live_buffers Vulkan buffers with live bound allocations in this process.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_live_buffers gauge\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_live_buffers",
                "memory",
                "device_local",
                stats.live_device_local_buffers,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_live_buffers",
                "memory",
                "host_visible",
                stats.live_host_visible_buffers,
            );
            out.push_str("# HELP kiln_vulkan_buffer_live_bytes Vulkan memory bytes currently bound to live buffers in this process.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_live_bytes gauge\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_live_bytes",
                "memory",
                "device_local",
                stats.live_device_local_bytes,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_live_bytes",
                "memory",
                "host_visible",
                stats.live_host_visible_bytes,
            );
            out.push_str("# HELP kiln_vulkan_buffer_peak_live_bytes Process-lifetime high-water mark of memory bytes bound to live Vulkan buffers.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_peak_live_bytes gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_vulkan_buffer_peak_live_bytes {}",
                    stats.peak_live_bytes
                ),
            );
            out.push_str("# HELP kiln_vulkan_buffer_allocations_total Vulkan buffer memory allocations completed by this process.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_allocations_total counter\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_allocations_total",
                "memory",
                "device_local",
                stats.device_local_allocations,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_allocations_total",
                "memory",
                "host_visible",
                stats.host_visible_allocations,
            );
            out.push_str("# HELP kiln_vulkan_buffer_allocated_bytes_total Vulkan buffer memory bytes cumulatively allocated by this process.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_allocated_bytes_total counter\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_allocated_bytes_total",
                "memory",
                "device_local",
                stats.device_local_allocated_bytes,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_allocated_bytes_total",
                "memory",
                "host_visible",
                stats.host_visible_allocated_bytes,
            );
            out.push_str("# HELP kiln_vulkan_buffer_frees_total Vulkan buffer memory allocations freed by this process after destroying their bound buffers.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_frees_total counter\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_frees_total",
                "memory",
                "device_local",
                stats.device_local_frees,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_frees_total",
                "memory",
                "host_visible",
                stats.host_visible_frees,
            );
            out.push_str("# HELP kiln_vulkan_buffer_freed_bytes_total Vulkan buffer memory bytes cumulatively freed by this process.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_freed_bytes_total counter\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_freed_bytes_total",
                "memory",
                "device_local",
                stats.device_local_freed_bytes,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_freed_bytes_total",
                "memory",
                "host_visible",
                stats.host_visible_freed_bytes,
            );
        }

        let stats = gauges.batched_state_cache;
        out.push_str("# HELP kiln_batched_recurrent_state_cache_entry Whether a parked batched recurrent-state cache entry exists.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_entry gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batched_recurrent_state_cache_entry {}",
                u8::from(stats.entry_present)
            ),
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_rows Parked recurrent-state rows by ownership kind.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_rows gauge\n");
        prom_counter(
            &mut out,
            "kiln_batched_recurrent_state_cache_rows",
            "kind",
            "capacity",
            stats.capacity_rows as u64,
        );
        prom_counter(
            &mut out,
            "kiln_batched_recurrent_state_cache_rows",
            "kind",
            "logical",
            stats.logical_rows as u64,
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_resident Whether every parked GDN buffer is backend-resident.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_resident gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batched_recurrent_state_cache_resident {}",
                u8::from(stats.resident)
            ),
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_leases Batched recurrent-state leases by kind.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_leases gauge\n");
        prom_counter(
            &mut out,
            "kiln_batched_recurrent_state_cache_leases",
            "kind",
            "active",
            stats.active_leases,
        );
        prom_counter(
            &mut out,
            "kiln_batched_recurrent_state_cache_leases",
            "kind",
            "max",
            stats.max_active_leases,
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_takes_total Batched recurrent-state cache checkouts by result.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_takes_total counter\n");
        for (result, value) in [
            ("hit", stats.take_hit_count),
            ("miss", stats.take_miss_count),
        ] {
            prom_counter(
                &mut out,
                "kiln_batched_recurrent_state_cache_takes_total",
                "result",
                result,
                value,
            );
        }
        out.push_str("# HELP kiln_batched_recurrent_state_cache_misses_while_leased_total Empty cache checkouts while another recurrent-state lease is active.\n");
        out.push_str(
            "# TYPE kiln_batched_recurrent_state_cache_misses_while_leased_total counter\n",
        );
        push_line(
            &mut out,
            &format!(
                "kiln_batched_recurrent_state_cache_misses_while_leased_total {}",
                stats.take_miss_while_leased_count
            ),
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_reuses_total Batched recurrent-state cache reuse operations by kind.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_reuses_total counter\n");
        for (kind, value) in [
            ("exact", stats.exact_reuse_count),
            ("resident_capacity", stats.resident_capacity_reuse_count),
            ("prefix_view", stats.resident_prefix_view_count),
            ("refresh", stats.resident_refresh_count),
        ] {
            prom_counter(
                &mut out,
                "kiln_batched_recurrent_state_cache_reuses_total",
                "kind",
                kind,
                value,
            );
        }
        out.push_str("# HELP kiln_batched_recurrent_state_cache_assemblies_total Fresh batched recurrent-state assemblies.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_assemblies_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batched_recurrent_state_cache_assemblies_total {}",
                stats.fresh_assembly_count
            ),
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_rejections_total Checked-out entries rejected for reuse by reason.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_rejections_total counter\n");
        for (reason, value) in [
            ("missing_row_ids", stats.rejected_missing_row_ids_count),
            ("nonresident_rows", stats.rejected_nonresident_rows_count),
            ("nonresident_cache", stats.rejected_nonresident_cache_count),
            (
                "insufficient_capacity",
                stats.rejected_insufficient_capacity_count,
            ),
        ] {
            prom_counter(
                &mut out,
                "kiln_batched_recurrent_state_cache_rejections_total",
                "reason",
                reason,
                value,
            );
        }
        out.push_str("# HELP kiln_batched_recurrent_state_cache_parks_total Batched recurrent-state leases returned to the cache.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_parks_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batched_recurrent_state_cache_parks_total {}",
                stats.park_count
            ),
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_invalidations_total Explicit adapter or model lifecycle invalidations.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_invalidations_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_batched_recurrent_state_cache_invalidations_total {}",
                stats.explicit_invalidation_count
            ),
        );
        out.push_str("# HELP kiln_batched_recurrent_state_cache_completed_rows_total Completed cached rows by ownership action.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_completed_rows_total counter\n");
        for (action, value) in [
            ("preserve", stats.completed_row_preservation_count),
            ("evict", stats.completed_row_eviction_count),
        ] {
            prom_counter(
                &mut out,
                "kiln_batched_recurrent_state_cache_completed_rows_total",
                "action",
                action,
                value,
            );
        }
        out.push_str("# HELP kiln_batched_recurrent_state_cache_evictions_total Batched recurrent-state ownership releases by reason.\n");
        out.push_str("# TYPE kiln_batched_recurrent_state_cache_evictions_total counter\n");
        for (reason, value) in [
            ("park_replacement", stats.park_replacement_eviction_count),
            (
                "explicit_invalidation",
                stats.explicit_invalidation_eviction_count,
            ),
            ("completed_row", stats.completed_row_eviction_count),
            ("lease_drop", stats.lease_drop_eviction_count),
        ] {
            prom_counter(
                &mut out,
                "kiln_batched_recurrent_state_cache_evictions_total",
                "reason",
                reason,
                value,
            );
        }
        out.push_str("# HELP kiln_prefix_cache_snapshot_suppressions_total Block-aligned prefix snapshots rejected because backend-resident state was newer than the logical cache state.\n");
        out.push_str("# TYPE kiln_prefix_cache_snapshot_suppressions_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_snapshot_suppressions_total {}",
                stats.resident_prefix_snapshot_suppression_count
            ),
        );

        if let Some(stats) = gauges.vulkan_buffer_pool {
            out.push_str("# HELP kiln_vulkan_buffer_pool_limit_bytes Configured maximum bytes retained by the Vulkan scratch recycler.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_limit_bytes gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_vulkan_buffer_pool_limit_bytes {}",
                    stats.max_retained_bytes
                ),
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_bytes Vulkan scratch bytes retained by ownership state.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_bytes gauge\n");
            for (state, value) in [
                ("retained", stats.total_bytes),
                ("free", stats.free_bytes),
                ("borrowed", stats.borrowed_bytes()),
            ] {
                prom_counter(
                    &mut out,
                    "kiln_vulkan_buffer_pool_bytes",
                    "state",
                    state,
                    value,
                );
            }
            out.push_str("# HELP kiln_vulkan_buffer_pool_buffers Vulkan scratch buffers retained by ownership state.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_buffers gauge\n");
            for (state, value) in [
                ("retained", stats.buffer_count),
                ("free", stats.free_buffer_count),
                ("borrowed", stats.borrowed_buffer_count()),
            ] {
                prom_counter(
                    &mut out,
                    "kiln_vulkan_buffer_pool_buffers",
                    "state",
                    state,
                    value as u64,
                );
            }
            out.push_str("# HELP kiln_vulkan_buffer_pool_buckets Number of retained Vulkan recycler buckets.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_buckets gauge\n");
            push_line(
                &mut out,
                &format!("kiln_vulkan_buffer_pool_buckets {}", stats.bucket_count),
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_requests_total Vulkan recycler lookups by result.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_requests_total counter\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_pool_requests_total",
                "result",
                "hit",
                stats.cache_hits,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_pool_requests_total",
                "result",
                "miss",
                stats.cache_misses,
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_misses_total Vulkan recycler cache misses by allocation route.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_misses_total counter\n");
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_pool_misses_total",
                "route",
                "device_local",
                stats.device_local_cache_misses,
            );
            prom_counter(
                &mut out,
                "kiln_vulkan_buffer_pool_misses_total",
                "route",
                "host_visible",
                stats.host_visible_cache_misses,
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_evictions_total Idle Vulkan recycler buffers released for capacity or pressure.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_evictions_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_vulkan_buffer_pool_evictions_total {}",
                    stats.eviction_count
                ),
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_evicted_bytes_total Vulkan recycler bytes cumulatively released for capacity or pressure.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_evicted_bytes_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_vulkan_buffer_pool_evicted_bytes_total {}",
                    stats.evicted_bytes
                ),
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_uncached_allocations_total Vulkan scratch allocations not retained because the recycler cap had no room.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_uncached_allocations_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_vulkan_buffer_pool_uncached_allocations_total {}",
                    stats.uncached_allocation_count
                ),
            );
            out.push_str("# HELP kiln_vulkan_buffer_pool_uncached_allocated_bytes_total Vulkan scratch bytes cumulatively allocated outside the retained recycler.\n");
            out.push_str("# TYPE kiln_vulkan_buffer_pool_uncached_allocated_bytes_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_vulkan_buffer_pool_uncached_allocated_bytes_total {}",
                    stats.uncached_allocated_bytes
                ),
            );
        }

        let stats = gauges.resident_recurrent_state;
        out.push_str("# HELP kiln_gdn_recurrent_state_resident_entries Direct backend-private GDN recurrent-state buffers currently owned by resumable prefill or scoped decode.\n");
        out.push_str("# TYPE kiln_gdn_recurrent_state_resident_entries gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_gdn_recurrent_state_resident_entries {}",
                stats.entry_count
            ),
        );
        out.push_str("# HELP kiln_gdn_recurrent_state_resident_bytes Direct backend-private GDN recurrent-state bytes by buffer or allocation accounting kind.\n");
        out.push_str("# TYPE kiln_gdn_recurrent_state_resident_bytes gauge\n");
        for (kind, value) in [
            ("buffer", stats.buffer_bytes),
            ("allocation", stats.allocation_bytes),
        ] {
            prom_counter(
                &mut out,
                "kiln_gdn_recurrent_state_resident_bytes",
                "kind",
                kind,
                value,
            );
        }

        // --- Prefix cache ---
        out.push_str("# HELP kiln_prefix_cache_lookups_total Total prefix cache lookups.\n");
        out.push_str("# TYPE kiln_prefix_cache_lookups_total counter\n");
        prom_counter(
            &mut out,
            "kiln_prefix_cache_lookups_total",
            "result",
            "hit",
            gauges.prefix_cache.lookup_hits,
        );
        prom_counter(
            &mut out,
            "kiln_prefix_cache_lookups_total",
            "result",
            "miss",
            gauges.prefix_cache.lookup_misses,
        );

        out.push_str("# HELP kiln_prefix_cache_hit_tokens_total Total prompt tokens skipped by prefix cache hits.\n");
        out.push_str("# TYPE kiln_prefix_cache_hit_tokens_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_hit_tokens_total {}",
                gauges.prefix_cache.hit_tokens
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_hit_blocks_total Total KV blocks reused by prefix cache hits.\n");
        out.push_str("# TYPE kiln_prefix_cache_hit_blocks_total counter\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_hit_blocks_total {}",
                gauges.prefix_cache.hit_blocks
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_cached_blocks KV blocks currently retained by the prefix cache.\n");
        out.push_str("# TYPE kiln_prefix_cache_cached_blocks gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_cached_blocks {}",
                gauges.prefix_cache.cached_blocks
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_max_blocks Maximum KV blocks retainable by the prefix cache.\n");
        out.push_str("# TYPE kiln_prefix_cache_max_blocks gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_max_blocks {}",
                gauges.prefix_cache.max_blocks
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_cached_entries Prefix-cache entries currently retaining GDN state snapshots.\n");
        out.push_str("# TYPE kiln_prefix_cache_cached_entries gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_cached_entries {}",
                gauges.prefix_cache.cached_entries
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_max_entries Maximum prefix-cache entries that may retain GDN state snapshots.\n");
        out.push_str("# TYPE kiln_prefix_cache_max_entries gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_max_entries {}",
                gauges.prefix_cache.max_entries
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_active_leases In-flight requests currently pinning prefix-cache entries.\n");
        out.push_str("# TYPE kiln_prefix_cache_active_leases gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_active_leases {}",
                gauges.prefix_cache.active_leases
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_pending_release_entries Invalidated prefix-cache entries awaiting their final active lease.\n");
        out.push_str("# TYPE kiln_prefix_cache_pending_release_entries gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_pending_release_entries {}",
                gauges.prefix_cache.pending_release_entries
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_state_bytes Device memory retained by cached GDN state snapshots.\n");
        out.push_str("# TYPE kiln_prefix_cache_state_bytes gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_state_bytes {}",
                gauges.prefix_cache.cached_state_bytes
            ),
        );

        out.push_str("# HELP kiln_prefix_cache_max_state_bytes Maximum device memory retainable by cached GDN state snapshots.\n");
        out.push_str("# TYPE kiln_prefix_cache_max_state_bytes gauge\n");
        push_line(
            &mut out,
            &format!(
                "kiln_prefix_cache_max_state_bytes {}",
                gauges.prefix_cache.max_state_bytes
            ),
        );

        // --- Training ---
        out.push_str("# HELP kiln_training_jobs_total Total training jobs.\n");
        out.push_str("# TYPE kiln_training_jobs_total counter\n");
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "sft",
            "status",
            "completed",
            self.training_sft_completed.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "sft",
            "status",
            "failed",
            self.training_sft_failed.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "sft",
            "status",
            "cancelled",
            self.training_sft_cancelled.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "grpo",
            "status",
            "completed",
            self.training_grpo_completed.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "grpo",
            "status",
            "failed",
            self.training_grpo_failed.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "grpo",
            "status",
            "cancelled",
            self.training_grpo_cancelled.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "opd",
            "status",
            "completed",
            self.training_opd_completed.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "opd",
            "status",
            "failed",
            self.training_opd_failed.load(Ordering::Relaxed),
        );
        prom_counter2(
            &mut out,
            "kiln_training_jobs_total",
            "type",
            "opd",
            "status",
            "cancelled",
            self.training_opd_cancelled.load(Ordering::Relaxed),
        );

        out.push_str("# HELP kiln_training_active Currently running training job.\n");
        out.push_str("# TYPE kiln_training_active gauge\n");
        push_line(
            &mut out,
            &format!("kiln_training_active {}", gauges.training_active),
        );

        // --- Adapter ---
        out.push_str("# HELP kiln_active_adapter Currently loaded adapter.\n");
        out.push_str("# TYPE kiln_active_adapter gauge\n");
        if let Some(ref name) = gauges.active_adapter {
            push_line(
                &mut out,
                &format!("kiln_active_adapter{{name=\"{name}\"}} 1"),
            );
        } else {
            push_line(&mut out, "kiln_active_adapter{name=\"base\"} 1");
        }

        // --- Live GPU memory (the governor's all-process view) ---
        // Scrapeable counterpart to /health's `live` block: total / used (ALL
        // processes, incl. a coexisting llama.cpp / vLLM job) / free / available
        // / soft-reserved bytes, plus a pressure level (0=Comfortable, 1=Moderate,
        // 2=Tight, 3=Critical). Lets Prometheus/Grafana alert on memory pressure
        // and watch coexistence without shelling into nvtop.
        {
            let memory = gauges.memory_governor;
            let s = memory.snapshot;
            out.push_str(
                "# HELP kiln_gpu_memory_probe_failed Whether the latest selected-device memory probe failed (1=yes).\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_probe_failed gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_gpu_memory_probe_failed {}",
                    u8::from(s.observations.probe_failed)
                ),
            );
            let sample = memory.sample_status;
            out.push_str(
                "# HELP kiln_gpu_memory_sample_healthy Whether the cached memory sample is healthy for admission (1=yes).\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_sample_healthy gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_gpu_memory_sample_healthy {}",
                    u8::from(sample.healthy)
                ),
            );
            out.push_str(
                "# HELP kiln_gpu_memory_sample_stale Whether the cached memory sample exceeded its maximum age (1=yes).\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_sample_stale gauge\n");
            push_line(
                &mut out,
                &format!("kiln_gpu_memory_sample_stale {}", u8::from(sample.stale)),
            );
            out.push_str(
                "# HELP kiln_gpu_memory_sample_age_seconds Age of the cached memory sample in seconds.\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_sample_age_seconds gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_gpu_memory_sample_age_seconds {}",
                    sample.age.as_secs_f64()
                ),
            );
            out.push_str(
                "# HELP kiln_gpu_memory_sample_max_age_seconds Maximum healthy cached memory sample age in seconds.\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_sample_max_age_seconds gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_gpu_memory_sample_max_age_seconds {}",
                    sample.max_age.as_secs_f64()
                ),
            );
            out.push_str(
                "# HELP kiln_gpu_memory_sampler_required Whether cached admission requires the background sampler (1=yes).\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_sampler_required gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_gpu_memory_sampler_required {}",
                    u8::from(sample.sampler_required)
                ),
            );
            out.push_str(
                "# HELP kiln_gpu_memory_sampler_running Whether the required background sampler is currently running (1=yes).\n",
            );
            out.push_str("# TYPE kiln_gpu_memory_sampler_running gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_gpu_memory_sampler_running {}",
                    u8::from(sample.sampler_running)
                ),
            );
            if let Some(tier) = s.observations.host_backed {
                out.push_str(
                    "# HELP kiln_gpu_host_backed_memory_bytes Safe host-backed accelerator memory by kind.\n",
                );
                out.push_str("# TYPE kiln_gpu_host_backed_memory_bytes gauge\n");
                push_line(
                    &mut out,
                    &format!(
                        "kiln_gpu_host_backed_memory_bytes{{kind=\"total\"}} {}",
                        tier.total_bytes
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_gpu_host_backed_memory_bytes{{kind=\"used\"}} {}",
                        tier.used_bytes
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_gpu_host_backed_memory_bytes{{kind=\"free\"}} {}",
                        tier.free_bytes
                    ),
                );
            }
            if s.total_bytes > 0 {
                out.push_str(
                    "# HELP kiln_gpu_memory_bytes Live GPU memory by kind (all-process driver view).\n",
                );
                out.push_str("# TYPE kiln_gpu_memory_bytes gauge\n");
                push_line(
                    &mut out,
                    &format!("kiln_gpu_memory_bytes{{kind=\"total\"}} {}", s.total_bytes),
                );
                push_line(
                    &mut out,
                    &format!("kiln_gpu_memory_bytes{{kind=\"used\"}} {}", s.used_bytes),
                );
                push_line(
                    &mut out,
                    &format!("kiln_gpu_memory_bytes{{kind=\"free\"}} {}", s.free_bytes),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_gpu_memory_bytes{{kind=\"available\"}} {}",
                        memory.available_bytes
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_gpu_memory_bytes{{kind=\"soft_reserved\"}} {}",
                        memory.soft_reserved_bytes
                    ),
                );
                out.push_str(
                    "# HELP kiln_gpu_memory_pressure GPU memory pressure (0=Comfortable 1=Moderate 2=Tight 3=Critical).\n",
                );
                out.push_str("# TYPE kiln_gpu_memory_pressure gauge\n");
                let level = match memory.pressure {
                    kiln_memory::MemoryPressure::Comfortable => 0,
                    kiln_memory::MemoryPressure::Moderate => 1,
                    kiln_memory::MemoryPressure::Tight => 2,
                    kiln_memory::MemoryPressure::Critical => 3,
                };
                push_line(&mut out, &format!("kiln_gpu_memory_pressure {level}"));
            }

            let reclaim = memory.automatic_reclaim;
            out.push_str(
                "# HELP kiln_memory_reclaim_attempts_total Automatic memory reclaim attempts by outcome.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaim_attempts_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_attempts_total{{outcome=\"reclaimed\"}} {}",
                    reclaim.successful_attempts
                ),
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_attempts_total{{outcome=\"zero_yield\"}} {}",
                    reclaim.zero_yield_attempts
                ),
            );
            out.push_str(
                "# HELP kiln_memory_reclaim_suppressed_total Automatic reclaim actions suppressed by cooldown or backoff.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaim_suppressed_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_suppressed_total {}",
                    reclaim.suppressed_attempts
                ),
            );
            out.push_str(
                "# HELP kiln_memory_reclaimed_bytes_total Bytes actually returned by automatic reclaim.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaimed_bytes_total counter\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaimed_bytes_total {}",
                    reclaim.reclaimed_bytes
                ),
            );
            out.push_str(
                "# HELP kiln_memory_reclaim_last_bytes Target and actual bytes for the last automatic reclaim attempt.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaim_last_bytes gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_last_bytes{{kind=\"target\"}} {}",
                    reclaim.last_target_bytes
                ),
            );
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_last_bytes{{kind=\"reclaimed\"}} {}",
                    reclaim.last_reclaimed_bytes
                ),
            );
            out.push_str(
                "# HELP kiln_memory_reclaim_last_duration_seconds Duration of the last automatic reclaim attempt.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaim_last_duration_seconds gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_last_duration_seconds {}",
                    reclaim.last_duration_us as f64 / 1_000_000.0
                ),
            );
            out.push_str(
                "# HELP kiln_memory_reclaim_retry_after_seconds Scheduled delay after the last automatic reclaim attempt.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaim_retry_after_seconds gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_retry_after_seconds {}",
                    reclaim.retry_after_ms as f64 / 1_000.0
                ),
            );
            out.push_str(
                "# HELP kiln_memory_reclaim_zero_yield_streak Consecutive automatic reclaim attempts returning zero bytes.\n",
            );
            out.push_str("# TYPE kiln_memory_reclaim_zero_yield_streak gauge\n");
            push_line(
                &mut out,
                &format!(
                    "kiln_memory_reclaim_zero_yield_streak {}",
                    reclaim.zero_yield_streak
                ),
            );
        }

        out
    }
}

/// Dynamic gauge values snapshotted at render time.
pub struct SnapshotGauges {
    pub(crate) memory_governor: CachedMemoryGovernorObservation,
    pub backend_quarantined: bool,
    pub external_yield_sync: Vec<ExternalYieldSyncStats>,
    pub rocm_synchronization_mode: &'static str,
    pub rocm_synchronization: crate::accelerator_runtime::RocmSynchronizationRuntimeStats,
    pub rocm_graph: Option<kiln_model::RocmGraphStats>,
    pub(crate) rocm_graph_unavailable_reason:
        Option<crate::rocm_graph_observability::RocmGraphUnavailableReason>,
    pub rocm_graph_telemetry: Option<kiln_model::RocmGraphLiveTelemetry>,
    pub(crate) rocm_graph_telemetry_unavailable_reason:
        Option<crate::rocm_graph_observability::RocmGraphUnavailableReason>,
    pub scheduler_waiting: usize,
    pub scheduler_running: usize,
    pub blocks_used: usize,
    pub blocks_total: usize,
    pub vram_total: u64,
    pub vram_model: u64,
    pub vram_model_estimated: u64,
    pub vram_post_load_used: u64,
    pub vram_prefill_peak_used: u64,
    pub vram_kv_cache: u64,
    pub vram_training_budget: u64,
    pub vulkan_buffers: Option<kiln_model::VulkanBufferAllocationStats>,
    pub vulkan_buffer_pool: Option<kiln_model::VulkanBufferPoolStats>,
    pub batched_state_cache: kiln_model::BatchedStateCacheStats,
    pub resident_recurrent_state: kiln_model::GdnRecurrentStateResidencyStats,
    pub prefix_cache: PrefixCacheStats,
    pub rendered_prompt_cache_hits: u64,
    pub rendered_prompt_cache_misses: u64,
    pub rendered_prompt_cache_entries: usize,
    pub prompt_token_cache_hits: u64,
    pub prompt_token_cache_misses: u64,
    pub prompt_token_cache_entries: usize,
    pub decode_batcher_enabled: bool,
    pub decode_batcher: DecodeBatcherStats,
    pub batching_engine_enabled: bool,
    pub batching_engine: BatchingEngineSnapshot,
    pub training_active: u8,
    pub active_adapter: Option<String>,
}

pub enum RequestStatus {
    Ok,
    Error,
    Timeout,
    Rejected,
}

#[derive(Clone, Copy)]
pub enum TrainingMetricType {
    Sft,
    Grpo,
    Opd,
}

#[derive(Clone, Copy)]
pub enum TrainingMetricStatus {
    Completed,
    Failed,
    Cancelled,
}

fn push_line(out: &mut String, line: &str) {
    out.push_str(line);
    out.push('\n');
}

fn observe_bucket(buckets: &[AtomicU64], bounds: &[u64], value: u64) {
    let index = bounds
        .iter()
        .position(|&bound| value <= bound)
        .unwrap_or(bounds.len());
    buckets[index].fetch_add(1, Ordering::Relaxed);
}

fn thinking_budget_source_index(source: &ThinkingBudgetSource) -> usize {
    THINKING_BUDGET_SOURCES
        .iter()
        .position(|candidate| candidate == source)
        .unwrap_or(THINKING_BUDGET_SOURCES.len() - 1)
}

fn thinking_budget_outcome(budget: &RequestThinkingBudget, finish_reason: &str) -> &'static str {
    if !budget.configured {
        return "unconfigured";
    }
    match budget.applied {
        None => return "unresolved",
        Some(false) => return "inert",
        Some(true) => {}
    }

    match budget.trigger.as_deref() {
        Some("tokens") => return "tokens",
        Some("time") => return "time",
        Some("max_tokens") => return "max_tokens",
        _ => {}
    }
    if budget.triggered == Some(false) && budget.closed == Some(true) {
        return "natural_close";
    }
    if matches!(finish_reason, "error" | "timeout" | "client_disconnect") {
        return "interrupted";
    }
    if budget.closed == Some(false) {
        return "unclosed";
    }
    "unresolved"
}

fn update_peak(peak: &AtomicU64, value: u64) {
    let mut current = peak.load(Ordering::Relaxed);
    while value > current {
        match peak.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

fn latency_phase_values(phases: &LatencyPhaseTimings) -> [Option<f64>; LATENCY_PHASES.len()] {
    [
        phases.actor_queue_ms,
        phases.actor_admission_ms,
        phases.tokenization_ms,
        phases.prefill_ms,
        phases.decode_ms,
        phases.actor_cycle_idle_ms,
        phases.sampling_ms,
        phases.readback_ms,
        phases.response_delivery_ms,
        phases.handler_queue_ms,
        phases.client_delivery_ms,
        phases.gpu_lock_wait_ms,
        phases.graph_capture_ms,
        phases.graph_replay_ms,
        phases.synchronization_ms,
        phases.resize_ms,
        phases.trim_ms,
        phases.adapter_ms,
        phases.training_ms,
        phases.unexplained_ms,
    ]
}

fn render_atomic_latency_histogram(
    out: &mut String,
    name: &str,
    label: Option<(&str, &str)>,
    histogram: &AtomicLatencyHistogram,
) {
    let mut cumulative = 0_u64;
    for (index, bound) in LATENCY_BUCKETS_SECONDS.iter().enumerate() {
        cumulative = cumulative.saturating_add(histogram.buckets[index].load(Ordering::Relaxed));
        match label {
            Some((key, value)) => push_line(
                out,
                &format!("{name}_bucket{{{key}=\"{value}\",le=\"{bound}\"}} {cumulative}"),
            ),
            None => push_line(
                out,
                &format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}"),
            ),
        }
    }
    cumulative = cumulative
        .saturating_add(histogram.buckets[LATENCY_BUCKETS_SECONDS.len()].load(Ordering::Relaxed));
    let count = histogram.count.load(Ordering::Relaxed);
    let sum = histogram.sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    match label {
        Some((key, value)) => {
            push_line(
                out,
                &format!("{name}_bucket{{{key}=\"{value}\",le=\"+Inf\"}} {cumulative}"),
            );
            push_line(out, &format!("{name}_count{{{key}=\"{value}\"}} {count}"));
            push_line(out, &format!("{name}_sum{{{key}=\"{value}\"}} {sum:.6}"));
        }
        None => {
            push_line(out, &format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}"));
            push_line(out, &format!("{name}_count {count}"));
            push_line(out, &format!("{name}_sum {sum:.6}"));
        }
    }
}

fn render_histogram_buckets(
    out: &mut String,
    name: &str,
    bounds_seconds: &[f64],
    buckets: &[AtomicU64],
) {
    let mut cumulative = 0;
    for (idx, bound) in bounds_seconds.iter().enumerate() {
        cumulative += buckets[idx].load(Ordering::Relaxed);
        push_line(
            out,
            &format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}"),
        );
    }
    cumulative += buckets[bounds_seconds.len()].load(Ordering::Relaxed);
    push_line(out, &format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}"));
}

fn render_integer_histogram_buckets(
    out: &mut String,
    name: &str,
    bounds: &[u64],
    buckets: &[AtomicU64],
) {
    let mut cumulative = 0;
    for (idx, bound) in bounds.iter().enumerate() {
        cumulative += buckets[idx].load(Ordering::Relaxed);
        push_line(
            out,
            &format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}"),
        );
    }
    cumulative += buckets[bounds.len()].load(Ordering::Relaxed);
    push_line(out, &format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}"));
}

fn render_millisecond_histogram_buckets_as_seconds(
    out: &mut String,
    name: &str,
    bounds_ms: &[u64],
    buckets: &[AtomicU64],
) {
    let mut cumulative = 0;
    for (idx, bound_ms) in bounds_ms.iter().enumerate() {
        cumulative += buckets[idx].load(Ordering::Relaxed);
        let bound_seconds = *bound_ms as f64 / 1_000.0;
        push_line(
            out,
            &format!("{name}_bucket{{le=\"{bound_seconds}\"}} {cumulative}"),
        );
    }
    cumulative += buckets[bounds_ms.len()].load(Ordering::Relaxed);
    push_line(out, &format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}"));
}

fn prom_counter(out: &mut String, name: &str, label: &str, value: &str, count: u64) {
    out.push_str(&format!("{name}{{{label}=\"{value}\"}} {count}\n"));
}

fn prom_counter2(out: &mut String, name: &str, l1: &str, v1: &str, l2: &str, v2: &str, count: u64) {
    out.push_str(&format!("{name}{{{l1}=\"{v1}\",{l2}=\"{v2}\"}} {count}\n"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigValueSource;

    fn test_memory_governor_observation() -> CachedMemoryGovernorObservation {
        CachedMemoryGovernorObservation {
            snapshot: kiln_memory::MemorySnapshot {
                total_bytes: 24_000_000_000,
                used_bytes: 18_000_000_000,
                free_bytes: 6_000_000_000,
                source: kiln_memory::vram::VramSource::LinuxDrmSysfs,
                unified: false,
                observations: kiln_memory::MemorySnapshotObservations {
                    host_backed: Some(kiln_memory::MemoryTierSnapshot {
                        total_bytes: 8_000_000_000,
                        used_bytes: 5_000_000_000,
                        free_bytes: 3_000_000_000,
                    }),
                    ..kiln_memory::MemorySnapshotObservations::default()
                },
            },
            available_bytes: 4_000_000_000,
            soft_reserved_bytes: 2_000_000_000,
            pressure: kiln_memory::MemoryPressure::Tight,
            sample_status: kiln_memory::CachedSampleStatus {
                age: std::time::Duration::from_millis(2_500),
                max_age: std::time::Duration::from_secs(10),
                stale: false,
                sampler_required: true,
                sampler_running: true,
                healthy: true,
            },
            automatic_monitor_enabled: false,
            automatic_reclaim: kiln_memory::AutomaticReclaimStats::default(),
        }
    }

    fn test_thinking_budget(
        configured: bool,
        tokens_source: &str,
        time_source: &str,
    ) -> RequestThinkingBudget {
        RequestThinkingBudget {
            configured,
            max_tokens: None,
            max_time_ms: None,
            tokens_source: tokens_source.into(),
            time_source: time_source.into(),
            applied: None,
            triggered: None,
            trigger: None,
            closed: None,
            thinking_tokens: None,
            thinking_time_ms: None,
        }
    }

    #[test]
    fn test_metrics_render() {
        let m = Metrics::new();
        m.inc_request(RequestStatus::Ok);
        m.inc_request(RequestStatus::Ok);
        m.inc_request(RequestStatus::Error);
        m.observe_duration(0.5);
        m.observe_prefill_duration(0.25);
        m.observe_decode_duration(0.75);
        m.observe_token_gap(TokenGapObservation {
            gap: std::time::Duration::from_millis(300),
            reason: LatencyStallReason::ActorPrefill,
            attributed_duration: std::time::Duration::from_millis(280),
        });
        m.observe_request_latency(&RequestLatencyDiagnostics {
            emitted_tokens: 2,
            gap_samples: 1,
            retained_gap_samples: 1,
            gap_samples_truncated: false,
            ttft_ms: Some(40.0),
            itl_ms_p50: Some(300.0),
            itl_ms_p99: Some(300.0),
            itl_ms_p999: Some(300.0),
            max_itl_ms: Some(300.0),
            stall_threshold_ms: Some(250.0),
            stall_count: 1,
            unexplained_stall_count: 0,
            stall_reasons: crate::latency_observability::LatencyStallReasonCounts {
                actor_prefill: 1,
                ..Default::default()
            },
            phases: LatencyPhaseTimings {
                tokenization_ms: Some(4.0),
                prefill_ms: Some(280.0),
                ..Default::default()
            },
        });
        m.add_tokens(100);
        m.inc_active();
        m.inc_active();
        m.dec_active();
        m.request_prefill_tokens_completed
            .store(8192, std::sync::atomic::Ordering::Relaxed);

        let mut token_closed = test_thinking_budget(true, "request", "server_default");
        token_closed.max_tokens = Some(64);
        token_closed.max_time_ms = Some(1_500);
        token_closed.applied = Some(true);
        token_closed.triggered = Some(true);
        token_closed.trigger = Some("tokens".to_string());
        token_closed.closed = Some(true);
        m.observe_thinking_budget(&token_closed, "stop");

        let mut natural_close =
            test_thinking_budget(true, "hostile-user-controlled-source", "request_unlimited");
        natural_close.max_tokens = Some(32);
        natural_close.applied = Some(true);
        natural_close.triggered = Some(false);
        natural_close.closed = Some(true);
        m.observe_thinking_budget(&natural_close, "stop");

        let mut interrupted = test_thinking_budget(true, "request", "request");
        interrupted.max_time_ms = Some(250);
        interrupted.applied = Some(true);
        m.observe_thinking_budget(&interrupted, "error");

        let mut inert = test_thinking_budget(true, "unlimited", "unlimited");
        inert.applied = Some(false);
        m.observe_thinking_budget(&inert, "stop");

        let gauges = SnapshotGauges {
            memory_governor: test_memory_governor_observation(),
            backend_quarantined: true,
            external_yield_sync: vec![ExternalYieldSyncStats {
                boundary: "batched decode step".to_string(),
                calls: 4,
                failures: 1,
                total_micros: 250_000,
                max_micros: 125_000,
                slow_calls: 1,
            }],
            rocm_synchronization_mode: "stream_ordered",
            rocm_synchronization: crate::accelerator_runtime::RocmSynchronizationRuntimeStats {
                active: true,
                telemetry_available: true,
                cleanup_quarantined: true,
                telemetry_error: None,
                total_device_wait_count: 2,
                total_stream_wait_count: 3,
                total_waited_ns: 250_000_000,
                total_skipped_count: 7,
                reasons: vec![crate::accelerator_runtime::RocmSynchronizationReasonStats {
                    reason: "external_yield",
                    device_wait_count: 2,
                    stream_wait_count: 3,
                    waited_ns: 250_000_000,
                    skipped_count: 7,
                }],
            },
            rocm_graph: Some(kiln_model::RocmGraphStats {
                requested: true,
                capture_requested: true,
                enabled: true,
                capture_enabled: true,
                max_cached_graphs: 8,
                max_retained_bytes: 1_073_741_824,
                capture_attempts: 8,
                capture_successes: 6,
                capture_deferrals: 1,
                capture_failures: 1,
                replay_attempts: 11,
                replay_successes: 10,
                replay_failures: 1,
                failures: 2,
                decode_owner_release_count: 3,
                decode_owner_graph_release_count: 2,
                graph_slot_create_count: 4,
                graph_slot_reuse_count: 5,
                cache_admission_successes: 4,
                cache_evictions: 2,
                cache_evicted_bytes: 201_326_592,
                budget_evictions: 1,
                pressure_evictions: 1,
                invalidation_evictions: 0,
                recovery_evictions: 0,
                entry_capacity_rejections: 1,
                byte_budget_rejections: 1,
                accounting_incomplete_rejections: 0,
                pre_capture_entry_capacity_skips: 2,
                pre_capture_byte_budget_skips: 3,
                pre_capture_accounting_incomplete_skips: 1,
                pre_capture_memory_reservation_denied_skips: 4,
                memory_governor_selector_mismatch_skips: 2,
                captured_graph_count: 2,
                graph_slot_count: 3,
                active_graph_slot_count: 2,
                idle_graph_slot_count: 1,
                tracked_decode_owner_count: 2,
                retained_stable_io_bytes: 33_554_432,
                retained_capture_arena_bytes: 134_217_728,
                retained_blaslt_workspace_bytes: 67_108_864,
                retained_slot_state_bytes: 16_777_216,
                retained_bytes: 251_658_240,
                peak_retained_bytes: 536_870_912,
                opaque_native_object_count: 10,
                retained_bytes_accounting_complete: true,
                pre_candidate_headroom_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 10,
                    slow: 1,
                    total_duration_micros: 225_000,
                    max_duration_micros: 125_000,
                },
                candidate_warm_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 8,
                    slow: 2,
                    total_duration_micros: 450_000,
                    max_duration_micros: 250_000,
                },
                pre_native_reservation_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 6,
                    slow: 1,
                    total_duration_micros: 175_000,
                    max_duration_micros: 125_000,
                },
                native_capture_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 6,
                    slow: 1,
                    total_duration_micros: 325_000,
                    max_duration_micros: 200_000,
                },
                rejected_candidate_cleanup_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 2,
                    slow: 0,
                    total_duration_micros: 35_000,
                    max_duration_micros: 20_000,
                },
                last_transient_candidate_bytes: 100_663_296,
                peak_transient_candidate_bytes: 167_772_160,
                fallbacks: kiln_model::RocmGraphFallbackStats {
                    total: 11,
                    multi_row_batch_unsupported: 2,
                    graph_cache_byte_budget: 2,
                    graph_accounting_incomplete: 1,
                    moderate_memory_pressure: 1,
                    tight_memory_pressure: 3,
                    memory_reservation_denied: 1,
                    memory_governor_selector_mismatch: 1,
                    slow: 2,
                    total_duration_micros: 425_000,
                    max_duration_micros: 275_000,
                    ..kiln_model::RocmGraphFallbackStats::default()
                },
                ..kiln_model::RocmGraphStats::default()
            }),
            rocm_graph_unavailable_reason: None,
            rocm_graph_telemetry: Some(kiln_model::RocmGraphLiveTelemetry {
                current_phase: Some(kiln_model::RocmGraphPhase::CandidateWarm),
                current_phase_elapsed_micros: 80_000,
                pre_candidate_headroom_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 10,
                    slow: 1,
                    total_duration_micros: 225_000,
                    max_duration_micros: 125_000,
                },
                candidate_warm_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 8,
                    slow: 2,
                    total_duration_micros: 450_000,
                    max_duration_micros: 250_000,
                },
                pre_native_reservation_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 6,
                    slow: 1,
                    total_duration_micros: 175_000,
                    max_duration_micros: 125_000,
                },
                native_capture_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 6,
                    slow: 1,
                    total_duration_micros: 325_000,
                    max_duration_micros: 200_000,
                },
                rejected_candidate_cleanup_phase: kiln_model::RocmGraphPhaseStats {
                    calls: 2,
                    slow: 0,
                    total_duration_micros: 35_000,
                    max_duration_micros: 20_000,
                },
                last_transient_candidate_bytes: 100_663_296,
                peak_transient_candidate_bytes: 167_772_160,
            }),
            rocm_graph_telemetry_unavailable_reason: None,
            scheduler_waiting: 3,
            scheduler_running: 1,
            blocks_used: 10,
            blocks_total: 256,
            vram_total: 24_000_000_000,
            vram_model: 9_000_000_000,
            vram_model_estimated: 8_000_000_000,
            vram_post_load_used: 9_000_000_000,
            vram_prefill_peak_used: 19_000_000_000,
            vram_kv_cache: 2_000_000_000,
            vram_training_budget: 14_000_000_000,
            vulkan_buffers: Some(kiln_model::VulkanBufferAllocationStats {
                live_device_local_buffers: 11,
                live_device_local_bytes: 101,
                live_host_visible_buffers: 12,
                live_host_visible_bytes: 102,
                peak_live_bytes: 203,
                device_local_allocations: 21,
                device_local_allocated_bytes: 201,
                device_local_frees: 10,
                device_local_freed_bytes: 100,
                host_visible_allocations: 22,
                host_visible_allocated_bytes: 202,
                host_visible_frees: 10,
                host_visible_freed_bytes: 100,
            }),
            vulkan_buffer_pool: Some(kiln_model::VulkanBufferPoolStats {
                max_retained_bytes: 4096,
                bucket_count: 3,
                buffer_count: 8,
                total_bytes: 3072,
                free_buffer_count: 5,
                free_bytes: 2048,
                cache_hits: 100,
                cache_misses: 9,
                device_local_cache_misses: 7,
                host_visible_cache_misses: 2,
                last_cache_miss: kiln_model::VulkanBufferPoolCacheMiss {
                    sequence: 9,
                    route: kiln_model::VulkanBufferPoolCacheMissRoute::DeviceLocal,
                    requested_bytes: 20_000_000,
                    bucket_bytes: 20_971_520,
                    caller_file: "crates/kiln-tensor/src/vulkan_storage.rs",
                    caller_line: 1234,
                },
                eviction_count: 2,
                evicted_bytes: 512,
                uncached_allocation_count: 1,
                uncached_allocated_bytes: 256,
            }),
            resident_recurrent_state: kiln_model::GdnRecurrentStateResidencyStats {
                entry_count: 24,
                buffer_bytes: 1_572_864,
                allocation_bytes: 1_671_168,
            },
            batched_state_cache: kiln_model::BatchedStateCacheStats {
                entry_present: true,
                capacity_rows: 8,
                logical_rows: 4,
                resident: true,
                active_leases: 1,
                max_active_leases: 3,
                take_hit_count: 19,
                take_miss_count: 5,
                take_miss_while_leased_count: 4,
                exact_reuse_count: 7,
                resident_capacity_reuse_count: 8,
                resident_prefix_view_count: 6,
                resident_refresh_count: 5,
                fresh_assembly_count: 5,
                rejected_missing_row_ids_count: 1,
                rejected_nonresident_rows_count: 2,
                rejected_nonresident_cache_count: 3,
                rejected_insufficient_capacity_count: 4,
                park_count: 20,
                park_replacement_eviction_count: 4,
                explicit_invalidation_count: 2,
                explicit_invalidation_eviction_count: 1,
                completed_row_preservation_count: 11,
                completed_row_eviction_count: 1,
                lease_drop_eviction_count: 6,
                resident_prefix_snapshot_suppression_count: 9,
            },
            prefix_cache: PrefixCacheStats {
                lookup_hits: 7,
                lookup_misses: 3,
                hit_tokens: 112,
                hit_blocks: 7,
                cached_blocks: 64,
                max_blocks: 128,
                cached_entries: 4,
                max_entries: 8,
                cached_state_bytes: 196,
                max_state_bytes: 392,
                active_leases: 2,
                pending_release_entries: 1,
            },
            rendered_prompt_cache_hits: 6,
            rendered_prompt_cache_misses: 3,
            rendered_prompt_cache_entries: 5,
            prompt_token_cache_hits: 5,
            prompt_token_cache_misses: 2,
            prompt_token_cache_entries: 4,
            decode_batcher_enabled: true,
            decode_batcher: DecodeBatcherStats {
                submitted_jobs: 4,
                executed_batches: 2,
                executed_rows: 4,
                runner_calls: 2,
                max_runner_calls_per_token: 1,
                max_observed_batch: 3,
                runner_busy_jobs: 1,
                failed_jobs: 0,
            },
            batching_engine_enabled: true,
            batching_engine: BatchingEngineSnapshot {
                snapshot_age_ms: 1_250,
                actor_cycle_idle_ms: 75,
                actor_cycle_idle_source: ConfigValueSource::ConfigFile,
                actor_cycle_idle_active: true,
                actor_cycle_idle_count: 5,
                total_actor_cycle_idle_ms: 375.0,
                max_actor_cycle_idle_ms: 80.0,
                queue_depth: 2,
                active_decode: 3,
                active_prefill: 2,
                prefix_cache_enabled: true,
                resident_prefill_enabled: true,
                active_resident_prefill: 1,
                max_batch_tokens: 256,
                max_prefill_tokens_per_cycle: 64,
                max_prefill_layers_per_cycle: 4,
                max_prefill_admission_quantum: 2,
                max_prefill_staging_slots: 2,
                max_active_requests: 10,
                max_prefill_staging_priority_burst: 4,
                max_decode_batch: 8,
                active_staged_requests: 2,
                max_observed_active_requests: 10,
                last_batch_size: 3,
                max_observed_batch_size: 4,
                last_forward_ms: 12.5,
                max_decode_forward_ms: 125.0,
                total_decode_forward_ms: 1_250.0,
                slow_decode_forward_count: 2,
                last_prefill_ms: 250.0,
                max_prefill_forward_ms: 625.0,
                total_prefill_forward_ms: 2_500.0,
                slow_prefill_forward_count: 4,
                last_prefill_tokens: 253,
                last_prefill_layers: 4,
                last_admission_ms: 25.0,
                max_admission_ms: 150.0,
                total_admission_ms: 500.0,
                total_admission_calls: 8,
                slow_admission_count: 1,
                total_decode_forwards: 17,
                total_batched_decode_forwards: 15,
                total_decode_rows: 48,
                total_prefill_admission_cycles: 6,
                total_prefill_forwards: 12,
                total_resident_prefill_attempts: 11,
                total_resident_prefill_forwards: 9,
                total_resident_prefill_initial_declines: 2,
                total_resident_prefill_route_failures: 0,
                total_resident_prefill_rows: 61,
                total_resident_prefill_completed_rows: 7,
                last_resident_prefill_batch_size: 6,
                max_resident_prefill_batch_size: 8,
                total_decode_tokens: 128,
                total_prefill_tokens: 8192,
                total_prefill_layers: 48,
                total_prefill_layer_yields: 36,
                total_short_prefill_priority_forwards: 9,
                total_prefill_staging_priority_forwards: 3,
                total_prefill_staging_admissions: 4,
                total_errors: 1,
                response_delivery_in_flight: 3,
                response_delivery_backpressured: 2,
                response_delivery_pending_terminal: 1,
                response_backpressure_events: 2,
                response_backpressure_wait_ms: 750,
                response_stall_evictions: 1,
                response_channel_closed: 3,
                prefix_deferred_waiting: 1,
                prefix_admission_deferrals: 4,
                ..BatchingEngineSnapshot::default()
            },
            training_active: 0,
            active_adapter: Some("my-adapter".to_string()),
        };

        let output = m.render(&gauges);

        assert!(output.contains("kiln_requests_total{status=\"ok\"} 2"));
        assert!(output.contains("kiln_requests_total{status=\"error\"} 1"));
        assert!(output.contains("kiln_tokens_generated_total 100"));
        assert!(output.contains("kiln_active_requests 1"));
        assert!(output.contains("kiln_active_requests_peak 2"));
        assert!(output.contains("kiln_request_prefill_tokens_completed 8192"));
        assert!(output.contains("kiln_scheduler_waiting 3"));
        assert!(output.contains("kiln_blocks_total 256"));
        assert!(output.contains("kiln_vram_total_bytes 24000000000"));
        assert!(output.contains("kiln_vram_model_bytes 9000000000"));
        assert!(output.contains("kiln_vram_model_estimated_bytes 8000000000"));
        assert!(output.contains("kiln_vram_post_load_used_bytes 9000000000"));
        assert!(output.contains("kiln_vram_prefill_peak_used_bytes 19000000000"));
        assert!(output.contains("kiln_gpu_memory_bytes{kind=\"total\"} 24000000000"));
        assert!(output.contains("kiln_gpu_memory_bytes{kind=\"available\"} 4000000000"));
        assert!(output.contains("kiln_gpu_memory_bytes{kind=\"soft_reserved\"} 2000000000"));
        assert!(output.contains("kiln_gpu_memory_pressure 2"));
        assert!(output.contains("kiln_gpu_memory_probe_failed 0"));
        assert!(output.contains("kiln_gpu_memory_sample_healthy 1"));
        assert!(output.contains("kiln_gpu_memory_sample_stale 0"));
        assert!(output.contains("kiln_gpu_memory_sample_age_seconds 2.5"));
        assert!(output.contains("kiln_gpu_memory_sample_max_age_seconds 10"));
        assert!(output.contains("kiln_gpu_memory_sampler_required 1"));
        assert!(output.contains("kiln_gpu_memory_sampler_running 1"));
        assert!(output.contains("kiln_gpu_host_backed_memory_bytes{kind=\"free\"} 3000000000"));
        assert!(output.contains("kiln_memory_reclaim_attempts_total{outcome=\"reclaimed\"} 0"));
        assert!(output.contains("kiln_memory_reclaim_attempts_total{outcome=\"zero_yield\"} 0"));
        assert!(output.contains("kiln_memory_reclaim_suppressed_total 0"));
        assert!(output.contains("kiln_memory_reclaimed_bytes_total 0"));
        assert!(output.contains("kiln_memory_reclaim_last_bytes{kind=\"target\"} 0"));
        assert!(output.contains("kiln_memory_reclaim_last_bytes{kind=\"reclaimed\"} 0"));
        assert!(output.contains("kiln_memory_reclaim_last_duration_seconds 0"));
        assert!(output.contains("kiln_memory_reclaim_retry_after_seconds 0"));
        assert!(output.contains("kiln_memory_reclaim_zero_yield_streak 0"));
        assert!(output.contains("kiln_backend_quarantined 1"));
        assert!(output.contains(
            "kiln_backend_external_yield_sync_calls_total{boundary=\"batched decode step\"} 4"
        ));
        assert!(output.contains(
            "kiln_backend_external_yield_sync_failures_total{boundary=\"batched decode step\"} 1"
        ));
        assert!(output.contains(
            "kiln_backend_external_yield_sync_seconds_total{boundary=\"batched decode step\"} 0.25"
        ));
        assert!(output.contains(
            "kiln_backend_external_yield_sync_max_seconds{boundary=\"batched decode step\"} 0.125"
        ));
        assert!(output.contains(
            "kiln_backend_external_yield_sync_slow_total{boundary=\"batched decode step\"} 1"
        ));
        assert!(
            output.contains("kiln_rocm_synchronization_policy_info{mode=\"stream_ordered\"} 1")
        );
        assert!(output.contains("kiln_rocm_cleanup_quarantined 1"));
        assert!(output.contains(
            "kiln_rocm_synchronizations_total{reason=\"external_yield\",scope=\"device\"} 2"
        ));
        assert!(output.contains(
            "kiln_rocm_synchronizations_total{reason=\"external_yield\",scope=\"stream\"} 3"
        ));
        assert!(output.contains(
            "kiln_rocm_synchronization_wait_seconds_total{reason=\"external_yield\"} 0.25"
        ));
        assert!(
            output.contains("kiln_rocm_synchronization_skipped_total{reason=\"external_yield\"} 7")
        );
        assert!(output.contains("kiln_rocm_graph_telemetry_available 1"));
        assert!(
            output.contains("kiln_rocm_graph_snapshot_unavailable{reason=\"model_runner_busy\"} 0")
        );
        assert!(output.contains("kiln_rocm_graph_phase_telemetry_available 1"));
        assert!(output.contains(
            "kiln_rocm_graph_phase_telemetry_unavailable{reason=\"backend_without_graph_runner\"} 0"
        ));
        assert!(output.contains("kiln_rocm_graph_current_phase{phase=\"candidate_warm\"} 1"));
        assert!(output.contains("kiln_rocm_graph_current_phase{phase=\"native_capture\"} 0"));
        assert!(output.contains("kiln_rocm_graph_current_phase_elapsed_seconds 0.080000"));
        assert!(output.contains("kiln_rocm_graph_state{kind=\"capture_enabled\"} 1"));
        assert!(output.contains("kiln_rocm_graph_cache_entries 2"));
        assert!(output.contains("kiln_rocm_graph_cache_entry_limit 8"));
        assert!(output.contains("kiln_rocm_graph_retained_bytes{kind=\"total\"} 251658240"));
        assert!(output.contains("kiln_rocm_graph_retained_byte_limit 1073741824"));
        assert!(output.contains("kiln_rocm_graph_retained_byte_accounting_complete 1"));
        assert!(output.contains("kiln_rocm_graph_slots{state=\"active\"} 2"));
        assert!(output.contains("kiln_rocm_graph_tracked_decode_owners 2"));
        assert!(output.contains("kiln_rocm_graph_owner_lifecycle_total{event=\"slot_reuse\"} 5"));
        assert!(output.contains("kiln_rocm_graph_opaque_native_objects 10"));
        assert!(
            output.contains("kiln_rocm_graph_transient_candidate_bytes{kind=\"last\"} 100663296")
        );
        assert!(
            output.contains("kiln_rocm_graph_transient_candidate_bytes{kind=\"peak\"} 167772160")
        );
        assert!(
            output
                .contains("kiln_rocm_graph_phase_calls_total{phase=\"pre_candidate_headroom\"} 10")
        );
        assert!(output.contains("kiln_rocm_graph_phase_calls_total{phase=\"candidate_warm\"} 8"));
        assert!(
            output.contains("kiln_rocm_graph_phase_slow_total{phase=\"pre_native_reservation\"} 1")
        );
        assert!(output.contains(
            "kiln_rocm_graph_phase_duration_seconds_total{phase=\"native_capture\"} 0.325000"
        ));
        assert!(output.contains(
            "kiln_rocm_graph_phase_duration_seconds_max{phase=\"rejected_candidate_cleanup\"} 0.020000"
        ));
        assert!(output.contains("kiln_rocm_graph_cache_admissions_total 4"));
        assert!(output.contains("kiln_rocm_graph_cache_evictions_total 2"));
        assert!(output.contains("kiln_rocm_graph_cache_evicted_bytes_total 201326592"));
        assert!(
            output.contains("kiln_rocm_graph_cache_evictions_by_cause_total{cause=\"budget\"} 1")
        );
        assert!(
            output.contains("kiln_rocm_graph_cache_evictions_by_cause_total{cause=\"recovery\"} 0")
        );
        assert!(output.contains(
            "kiln_rocm_graph_cache_admission_rejections_total{reason=\"byte_budget\"} 1"
        ));
        assert!(output.contains(
            "kiln_rocm_graph_pre_capture_skips_total{reason=\"accounting_incomplete\"} 1"
        ));
        assert!(output.contains(
            "kiln_rocm_graph_pre_capture_skips_total{reason=\"memory_reservation_denied\"} 4"
        ));
        assert!(output.contains(
            "kiln_rocm_graph_pre_capture_skips_total{reason=\"memory_governor_selector_mismatch\"} 2"
        ));
        assert!(output.contains("kiln_rocm_graph_capture_attempts_total 8"));
        assert!(output.contains("kiln_rocm_graph_capture_outcomes_total{outcome=\"deferred\"} 1"));
        assert!(output.contains("kiln_rocm_graph_replay_attempts_total 11"));
        assert!(
            output.contains(
                "kiln_rocm_graph_fallbacks_total{reason=\"multi_row_batch_unsupported\"} 2"
            )
        );
        assert!(
            output
                .contains("kiln_rocm_graph_fallbacks_total{reason=\"graph_cache_byte_budget\"} 2")
        );
        assert!(
            output.contains("kiln_rocm_graph_fallbacks_total{reason=\"tight_memory_pressure\"} 3")
        );
        assert!(
            output
                .contains("kiln_rocm_graph_fallbacks_total{reason=\"moderate_memory_pressure\"} 1")
        );
        assert!(
            output.contains(
                "kiln_rocm_graph_fallbacks_total{reason=\"memory_reservation_denied\"} 1"
            )
        );
        assert!(output.contains(
            "kiln_rocm_graph_fallbacks_total{reason=\"memory_governor_selector_mismatch\"} 1"
        ));
        assert!(output.contains("kiln_rocm_graph_fallback_slow_total 2"));
        assert!(output.contains("kiln_rocm_graph_fallback_duration_seconds_total 0.425000"));
        assert!(output.contains("kiln_rocm_graph_fallback_duration_seconds_max 0.275000"));
        assert!(output.contains("kiln_vulkan_buffer_live_buffers{memory=\"device_local\"} 11"));
        assert!(output.contains("kiln_vulkan_buffer_live_bytes{memory=\"host_visible\"} 102"));
        assert!(output.contains("kiln_vulkan_buffer_peak_live_bytes 203"));
        assert!(output.contains("kiln_gdn_recurrent_state_resident_entries 24"));
        assert!(
            output.contains("kiln_gdn_recurrent_state_resident_bytes{kind=\"buffer\"} 1572864")
        );
        assert!(
            output.contains("kiln_gdn_recurrent_state_resident_bytes{kind=\"allocation\"} 1671168")
        );
        assert!(
            output.contains("kiln_vulkan_buffer_allocations_total{memory=\"device_local\"} 21")
        );
        assert!(
            output
                .contains("kiln_vulkan_buffer_allocated_bytes_total{memory=\"host_visible\"} 202")
        );
        assert!(output.contains("kiln_vulkan_buffer_frees_total{memory=\"device_local\"} 10"));
        assert!(
            output.contains("kiln_vulkan_buffer_freed_bytes_total{memory=\"host_visible\"} 100")
        );
        assert!(output.contains("kiln_vulkan_buffer_pool_misses_total{route=\"device_local\"} 7"));
        assert!(output.contains("kiln_vulkan_buffer_pool_misses_total{route=\"host_visible\"} 2"));
        for expected in [
            "kiln_batched_recurrent_state_cache_entry 1",
            "kiln_batched_recurrent_state_cache_rows{kind=\"capacity\"} 8",
            "kiln_batched_recurrent_state_cache_rows{kind=\"logical\"} 4",
            "kiln_batched_recurrent_state_cache_resident 1",
            "kiln_batched_recurrent_state_cache_leases{kind=\"active\"} 1",
            "kiln_batched_recurrent_state_cache_leases{kind=\"max\"} 3",
            "kiln_batched_recurrent_state_cache_takes_total{result=\"hit\"} 19",
            "kiln_batched_recurrent_state_cache_takes_total{result=\"miss\"} 5",
            "kiln_batched_recurrent_state_cache_misses_while_leased_total 4",
            "kiln_batched_recurrent_state_cache_reuses_total{kind=\"exact\"} 7",
            "kiln_batched_recurrent_state_cache_reuses_total{kind=\"resident_capacity\"} 8",
            "kiln_batched_recurrent_state_cache_reuses_total{kind=\"prefix_view\"} 6",
            "kiln_batched_recurrent_state_cache_reuses_total{kind=\"refresh\"} 5",
            "kiln_batched_recurrent_state_cache_assemblies_total 5",
            "kiln_batched_recurrent_state_cache_rejections_total{reason=\"missing_row_ids\"} 1",
            "kiln_batched_recurrent_state_cache_rejections_total{reason=\"nonresident_rows\"} 2",
            "kiln_batched_recurrent_state_cache_rejections_total{reason=\"nonresident_cache\"} 3",
            "kiln_batched_recurrent_state_cache_rejections_total{reason=\"insufficient_capacity\"} 4",
            "kiln_batched_recurrent_state_cache_parks_total 20",
            "kiln_batched_recurrent_state_cache_invalidations_total 2",
            "kiln_batched_recurrent_state_cache_completed_rows_total{action=\"preserve\"} 11",
            "kiln_batched_recurrent_state_cache_completed_rows_total{action=\"evict\"} 1",
            "kiln_batched_recurrent_state_cache_evictions_total{reason=\"park_replacement\"} 4",
            "kiln_batched_recurrent_state_cache_evictions_total{reason=\"explicit_invalidation\"} 1",
            "kiln_batched_recurrent_state_cache_evictions_total{reason=\"completed_row\"} 1",
            "kiln_batched_recurrent_state_cache_evictions_total{reason=\"lease_drop\"} 6",
            "kiln_prefix_cache_snapshot_suppressions_total 9",
        ] {
            assert!(output.contains(expected), "missing metric: {expected}");
        }
        assert!(output.contains("kiln_prefix_cache_lookups_total{result=\"hit\"} 7"));
        assert!(output.contains("kiln_prefix_cache_lookups_total{result=\"miss\"} 3"));
        assert!(output.contains("kiln_prefix_cache_hit_tokens_total 112"));
        assert!(output.contains("kiln_prefix_cache_hit_blocks_total 7"));
        assert!(output.contains("kiln_prefix_cache_cached_blocks 64"));
        assert!(output.contains("kiln_prefix_cache_max_blocks 128"));
        assert!(output.contains("kiln_prefix_cache_cached_entries 4"));
        assert!(output.contains("kiln_prefix_cache_max_entries 8"));
        assert!(output.contains("kiln_prefix_cache_active_leases 2"));
        assert!(output.contains("kiln_prefix_cache_pending_release_entries 1"));
        assert!(output.contains("kiln_prefix_cache_state_bytes 196"));
        assert!(output.contains("kiln_prefix_cache_max_state_bytes 392"));
        assert!(output.contains("kiln_rendered_prompt_cache_lookups_total{result=\"hit\"} 6"));
        assert!(output.contains("kiln_rendered_prompt_cache_lookups_total{result=\"miss\"} 3"));
        assert!(output.contains("kiln_rendered_prompt_cache_entries 5"));
        assert!(output.contains("kiln_prompt_token_cache_lookups_total{result=\"hit\"} 5"));
        assert!(output.contains("kiln_prompt_token_cache_lookups_total{result=\"miss\"} 2"));
        assert!(output.contains("kiln_prompt_token_cache_entries 4"));
        assert!(output.contains("kiln_batching_engine_enabled 1"));
        assert!(output.contains("kiln_batching_engine_snapshot_age_seconds 1.250000"));
        assert!(
            output.contains("kiln_batching_engine_actor_cycle_idle_configured_seconds 0.075000")
        );
        assert!(output.contains("kiln_batching_engine_actor_cycle_idle_active 1"));
        assert!(output.contains("kiln_batching_engine_actor_cycle_idles_total 5"));
        assert!(output.contains("kiln_batching_engine_actor_cycle_idle_seconds_total 0.375000"));
        assert!(output.contains("kiln_batching_engine_actor_cycle_idle_max_seconds 0.080000"));
        assert!(output.contains("kiln_batching_engine_queue_depth 2"));
        assert!(output.contains("kiln_batching_engine_active_decode 3"));
        assert!(output.contains("kiln_batching_engine_active_prefill 2"));
        assert!(output.contains("kiln_batching_engine_prefix_cache_enabled 1"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_enabled 1"));
        assert!(output.contains("kiln_batching_engine_active_resident_prefill 1"));
        assert!(output.contains("kiln_batching_engine_max_batch_tokens 256"));
        assert!(output.contains("kiln_batching_engine_max_prefill_tokens_per_cycle 64"));
        assert!(output.contains("kiln_batching_engine_max_prefill_layers_per_cycle 4"));
        assert!(output.contains("kiln_batching_engine_prefill_admission_quantum 2"));
        assert!(output.contains("kiln_batching_engine_prefill_staging_slots 2"));
        assert!(output.contains("kiln_batching_engine_max_active_requests 10"));
        assert!(output.contains("kiln_batching_engine_prefill_staging_priority_burst 4"));
        assert!(output.contains("kiln_batching_engine_active_staged_requests 2"));
        assert!(output.contains("kiln_batching_engine_max_observed_active_requests 10"));
        assert!(output.contains("kiln_batching_engine_max_decode_batch 8"));
        assert!(output.contains("kiln_batching_engine_last_batch_size 3"));
        assert!(output.contains("kiln_batching_engine_max_observed_batch 4"));
        assert!(output.contains("kiln_batching_engine_last_forward_ms 12.500000"));
        assert!(output.contains("kiln_batching_engine_decode_forward_seconds_total 1.250000"));
        assert!(output.contains("kiln_batching_engine_decode_forward_max_seconds 0.125000"));
        assert!(output.contains("kiln_batching_engine_slow_decode_forwards_total 2"));
        assert!(output.contains("kiln_batching_engine_last_prefill_ms 250.000000"));
        assert!(output.contains("kiln_batching_engine_last_prefill_tokens 253"));
        assert!(output.contains("kiln_batching_engine_last_prefill_layers 4"));
        assert!(output.contains("kiln_batching_engine_prefill_forward_seconds_total 2.500000"));
        assert!(output.contains("kiln_batching_engine_prefill_forward_max_seconds 0.625000"));
        assert!(output.contains("kiln_batching_engine_slow_prefill_forwards_total 4"));
        assert!(output.contains("kiln_batching_engine_last_admission_seconds 0.025000"));
        assert!(output.contains("kiln_batching_engine_admission_seconds_total 0.500000"));
        assert!(output.contains("kiln_batching_engine_admission_max_seconds 0.150000"));
        assert!(output.contains("kiln_batching_engine_admission_calls_total 8"));
        assert!(output.contains("kiln_batching_engine_slow_admissions_total 1"));
        assert!(output.contains("kiln_batching_engine_decode_forwards_total 17"));
        assert!(output.contains("kiln_batching_engine_batched_decode_forwards_total 15"));
        assert!(output.contains("kiln_batching_engine_decode_rows_total 48"));
        assert!(output.contains("kiln_decode_batcher_runner_calls_total 2"));
        assert!(output.contains("kiln_decode_batcher_runner_calls_per_token 0.500000"));
        assert!(output.contains("kiln_decode_batcher_max_runner_calls_per_token 1"));
        assert!(output.contains("kiln_decode_batcher_runner_call_budget_per_token 2"));
        assert!(output.contains("kiln_decode_batcher_runner_call_budget_exceeded 0"));
        assert!(output.contains("kiln_batching_engine_prefill_admission_cycles_total 6"));
        assert!(output.contains("kiln_batching_engine_prefill_staging_priority_forwards_total 3"));
        assert!(output.contains("kiln_batching_engine_prefill_staging_admissions_total 4"));
        assert!(output.contains("kiln_batching_engine_prefill_forwards_total 12"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_attempts_total 11"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_forwards_total 9"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_initial_declines_total 2"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_route_failures_total 0"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_rows_total 61"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_completed_rows_total 7"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_last_batch_size 6"));
        assert!(output.contains("kiln_batching_engine_resident_prefill_max_batch_size 8"));
        assert!(output.contains("kiln_batching_engine_decode_tokens_total 128"));
        assert!(output.contains("kiln_batching_engine_prefill_tokens_total 8192"));
        assert!(output.contains("kiln_batching_engine_prefill_layers_total 48"));
        assert!(output.contains("kiln_batching_engine_prefill_layer_yields_total 36"));
        assert!(output.contains("kiln_batching_engine_short_prefill_priority_forwards_total 9"));
        assert!(output.contains("kiln_batching_engine_errors_total 1"));
        assert!(output.contains("# TYPE kiln_batching_engine_response_delivery_in_flight gauge"));
        assert!(output.contains("kiln_batching_engine_response_delivery_in_flight 3"));
        assert!(
            output.contains("# TYPE kiln_batching_engine_response_delivery_backpressured gauge")
        );
        assert!(output.contains("kiln_batching_engine_response_delivery_backpressured 2"));
        assert!(
            output.contains("# TYPE kiln_batching_engine_response_delivery_pending_terminal gauge")
        );
        assert!(output.contains("kiln_batching_engine_response_delivery_pending_terminal 1"));
        assert!(output.contains("kiln_batching_engine_response_backpressure_events_total 2"));
        assert!(
            output
                .contains("kiln_batching_engine_response_backpressure_wait_seconds_total 0.750000")
        );
        assert!(output.contains("kiln_batching_engine_response_stall_evictions_total 1"));
        assert!(output.contains("kiln_batching_engine_response_channel_closed_total 3"));
        assert!(output.contains("kiln_batching_engine_prefix_deferred_waiting 1"));
        assert!(output.contains("kiln_batching_engine_prefix_admission_deferrals_total 4"));
        assert!(output.contains("kiln_active_adapter{name=\"my-adapter\"} 1"));
        assert!(output.contains(r#"kiln_request_duration_seconds_bucket{le="0.5"} 1"#));
        assert!(output.contains(r#"kiln_request_duration_seconds_bucket{le="+Inf"} 1"#));
        assert!(output.contains("kiln_request_duration_seconds_count 1"));
        assert!(output.contains("kiln_request_duration_seconds_sum 0.5"));
        assert!(output.contains(r#"kiln_request_prefill_duration_seconds_bucket{le="0.25"} 1"#));
        assert!(output.contains("kiln_request_prefill_duration_seconds_count 1"));
        assert!(output.contains("kiln_request_prefill_duration_seconds_sum 0.25"));
        assert!(output.contains(r#"kiln_request_decode_duration_seconds_bucket{le="1"} 1"#));
        assert!(output.contains("kiln_request_decode_duration_seconds_count 1"));
        assert!(output.contains("kiln_request_decode_duration_seconds_sum 0.75"));
        assert!(output.contains(r#"kiln_request_ttft_seconds_bucket{le="0.05"} 1"#));
        assert!(output.contains("kiln_request_ttft_seconds_count 1"));
        assert!(output.contains(r#"kiln_token_itl_seconds_bucket{le="0.5"} 1"#));
        assert!(output.contains(
            r#"kiln_request_latency_phase_seconds_bucket{phase="tokenization",le="0.005"} 1"#
        ));
        assert!(output.contains(r#"kiln_token_stalls_total{reason="actor_prefill"} 1"#));
        assert!(output.contains(
            "kiln_thinking_budget_source_total{dimension=\"tokens\",source=\"request\"} 2"
        ));
        assert!(output.contains(
            "kiln_thinking_budget_source_total{dimension=\"tokens\",source=\"unknown\"} 1"
        ));
        assert!(output.contains(
            "kiln_thinking_budget_source_total{dimension=\"time\",source=\"request_unlimited\"} 1"
        ));
        assert!(!output.contains("hostile-user-controlled-source"));
        assert!(output.contains("kiln_thinking_budget_outcomes_total{outcome=\"tokens\"} 1"));
        assert!(
            output.contains("kiln_thinking_budget_outcomes_total{outcome=\"natural_close\"} 1")
        );
        assert!(output.contains("kiln_thinking_budget_outcomes_total{outcome=\"interrupted\"} 1"));
        assert!(output.contains("kiln_thinking_budget_outcomes_total{outcome=\"inert\"} 1"));
        assert!(output.contains("kiln_thinking_budget_effective_tokens_bucket{le=\"32\"} 1"));
        assert!(output.contains("kiln_thinking_budget_effective_tokens_bucket{le=\"64\"} 2"));
        assert!(output.contains("kiln_thinking_budget_effective_tokens_count 2"));
        assert!(output.contains("kiln_thinking_budget_effective_tokens_sum 96"));
        assert!(output.contains("kiln_thinking_budget_effective_seconds_bucket{le=\"0.25\"} 1"));
        assert!(output.contains("kiln_thinking_budget_effective_seconds_bucket{le=\"2\"} 2"));
        assert!(output.contains("kiln_thinking_budget_effective_seconds_count 2"));
        assert!(output.contains("kiln_thinking_budget_effective_seconds_sum 1.750000"));
    }

    #[test]
    fn thinking_budget_outcomes_use_only_the_closed_vocabulary() {
        let mut budget = test_thinking_budget(false, "unlimited", "unlimited");
        assert_eq!(thinking_budget_outcome(&budget, "stop"), "unconfigured");

        budget.configured = true;
        assert_eq!(thinking_budget_outcome(&budget, "error"), "unresolved");
        budget.applied = Some(false);
        assert_eq!(thinking_budget_outcome(&budget, "stop"), "inert");

        budget.applied = Some(true);
        budget.closed = Some(false);
        assert_eq!(thinking_budget_outcome(&budget, "length"), "unclosed");
        assert_eq!(thinking_budget_outcome(&budget, "timeout"), "interrupted");

        budget.closed = Some(true);
        budget.triggered = Some(false);
        assert_eq!(thinking_budget_outcome(&budget, "stop"), "natural_close");

        budget.triggered = Some(true);
        for trigger in ["tokens", "time", "max_tokens"] {
            budget.trigger = Some(trigger.to_string());
            assert_eq!(thinking_budget_outcome(&budget, "stop"), trigger);
        }
        budget.trigger = Some("future-user-supplied-value".to_string());
        budget.triggered = None;
        budget.closed = None;
        assert_eq!(thinking_budget_outcome(&budget, "stop"), "unresolved");
    }

    #[test]
    fn test_base_adapter_rendering() {
        let m = Metrics::new();
        let mut gauges = SnapshotGauges {
            memory_governor: CachedMemoryGovernorObservation::default(),
            backend_quarantined: false,
            external_yield_sync: Vec::new(),
            rocm_synchronization_mode: "legacy_host_barriers",
            rocm_synchronization:
                crate::accelerator_runtime::RocmSynchronizationRuntimeStats::default(),
            rocm_graph: Some(kiln_model::RocmGraphStats::default()),
            rocm_graph_unavailable_reason: None,
            rocm_graph_telemetry: Some(kiln_model::RocmGraphLiveTelemetry::default()),
            rocm_graph_telemetry_unavailable_reason: None,
            scheduler_waiting: 0,
            scheduler_running: 0,
            blocks_used: 0,
            blocks_total: 0,
            vram_total: 0,
            vram_model: 0,
            vram_model_estimated: 0,
            vram_post_load_used: 0,
            vram_prefill_peak_used: 0,
            vram_kv_cache: 0,
            vram_training_budget: 0,
            vulkan_buffers: None,
            vulkan_buffer_pool: None,
            batched_state_cache: kiln_model::BatchedStateCacheStats::default(),
            resident_recurrent_state: kiln_model::GdnRecurrentStateResidencyStats::default(),
            prefix_cache: PrefixCacheStats::default(),
            rendered_prompt_cache_hits: 0,
            rendered_prompt_cache_misses: 0,
            rendered_prompt_cache_entries: 0,
            prompt_token_cache_hits: 0,
            prompt_token_cache_misses: 0,
            prompt_token_cache_entries: 0,
            decode_batcher_enabled: false,
            decode_batcher: DecodeBatcherStats::default(),
            batching_engine_enabled: false,
            batching_engine: BatchingEngineSnapshot::default(),
            training_active: 0,
            active_adapter: None,
        };
        let output = m.render(&gauges);
        assert!(output.contains("kiln_active_adapter{name=\"base\"} 1"));

        gauges.rocm_graph = None;
        gauges.rocm_graph_unavailable_reason =
            Some(crate::rocm_graph_observability::RocmGraphUnavailableReason::ModelRunnerBusy);
        gauges.rocm_graph_telemetry = Some(kiln_model::RocmGraphLiveTelemetry {
            current_phase: Some(kiln_model::RocmGraphPhase::NativeCapture),
            current_phase_elapsed_micros: 125_000,
            ..kiln_model::RocmGraphLiveTelemetry::default()
        });
        let full_snapshot_busy = m.render(&gauges);
        assert!(full_snapshot_busy.contains("kiln_rocm_graph_telemetry_available 0"));
        assert!(full_snapshot_busy.contains("kiln_rocm_graph_phase_telemetry_available 1"));
        assert!(
            full_snapshot_busy
                .contains("kiln_rocm_graph_snapshot_unavailable{reason=\"model_runner_busy\"} 1")
        );
        assert!(
            full_snapshot_busy
                .contains("kiln_rocm_graph_current_phase{phase=\"native_capture\"} 1")
        );
        assert!(
            full_snapshot_busy.contains("kiln_rocm_graph_current_phase_elapsed_seconds 0.125000")
        );
        assert!(!full_snapshot_busy.contains("kiln_rocm_graph_cache_entries "));

        gauges.rocm_graph_telemetry = None;
        gauges.rocm_graph_telemetry_unavailable_reason = Some(
            crate::rocm_graph_observability::RocmGraphUnavailableReason::BackendWithoutGraphRunner,
        );
        let unavailable = m.render(&gauges);
        assert!(unavailable.contains("kiln_rocm_graph_telemetry_available 0"));
        assert!(unavailable.contains("kiln_rocm_graph_phase_telemetry_available 0"));
        assert!(
            unavailable
                .contains("kiln_rocm_graph_snapshot_unavailable{reason=\"model_runner_busy\"} 1")
        );
        assert!(unavailable.contains(
            "kiln_rocm_graph_phase_telemetry_unavailable{reason=\"backend_without_graph_runner\"} 1"
        ));
        assert!(!unavailable.contains("kiln_rocm_graph_cache_entries "));
        assert!(!unavailable.contains("kiln_rocm_graph_current_phase{"));
    }
}
