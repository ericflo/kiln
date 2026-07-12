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
        out.push_str("# HELP kiln_backend_external_yield_sync_calls_total Device-wide synchronization calls before external progress or resource reuse.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_calls_total counter\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_failures_total Failed device-wide synchronization calls.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_failures_total counter\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_seconds_total Total wall time spent in device-wide synchronization.\n");
        out.push_str("# TYPE kiln_backend_external_yield_sync_seconds_total counter\n");
        out.push_str("# HELP kiln_backend_external_yield_sync_max_seconds Maximum observed device-wide synchronization time.\n");
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
            let g = kiln_memory::MemoryGovernor::global();
            let s = g.snapshot();
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
                        g.available_bytes()
                    ),
                );
                push_line(
                    &mut out,
                    &format!(
                        "kiln_gpu_memory_bytes{{kind=\"soft_reserved\"}} {}",
                        g.soft_reserved_bytes()
                    ),
                );
                out.push_str(
                    "# HELP kiln_gpu_memory_pressure GPU memory pressure (0=Comfortable 1=Moderate 2=Tight 3=Critical).\n",
                );
                out.push_str("# TYPE kiln_gpu_memory_pressure gauge\n");
                let level = match g.pressure() {
                    kiln_memory::MemoryPressure::Comfortable => 0,
                    kiln_memory::MemoryPressure::Moderate => 1,
                    kiln_memory::MemoryPressure::Tight => 2,
                    kiln_memory::MemoryPressure::Critical => 3,
                };
                push_line(&mut out, &format!("kiln_gpu_memory_pressure {level}"));
            }
        }

        out
    }
}

/// Dynamic gauge values snapshotted at render time.
pub struct SnapshotGauges {
    pub backend_quarantined: bool,
    pub external_yield_sync: Vec<ExternalYieldSyncStats>,
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
            backend_quarantined: true,
            external_yield_sync: vec![ExternalYieldSyncStats {
                boundary: "batched decode step".to_string(),
                calls: 4,
                failures: 1,
                total_micros: 250_000,
                max_micros: 125_000,
                slow_calls: 1,
            }],
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
                queue_depth: 2,
                active_decode: 3,
                active_prefill: 2,
                max_batch_tokens: 256,
                max_prefill_tokens_per_cycle: 64,
                max_prefill_layers_per_cycle: 4,
                max_prefill_admission_quantum: 2,
                max_decode_batch: 8,
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
                total_decode_tokens: 128,
                total_prefill_tokens: 8192,
                total_prefill_layers: 48,
                total_prefill_layer_yields: 36,
                total_short_prefill_priority_forwards: 9,
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
        assert!(output.contains("kiln_batching_engine_queue_depth 2"));
        assert!(output.contains("kiln_batching_engine_active_decode 3"));
        assert!(output.contains("kiln_batching_engine_active_prefill 2"));
        assert!(output.contains("kiln_batching_engine_max_batch_tokens 256"));
        assert!(output.contains("kiln_batching_engine_max_prefill_tokens_per_cycle 64"));
        assert!(output.contains("kiln_batching_engine_max_prefill_layers_per_cycle 4"));
        assert!(output.contains("kiln_batching_engine_prefill_admission_quantum 2"));
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
        assert!(output.contains("kiln_batching_engine_prefill_forwards_total 12"));
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
        let gauges = SnapshotGauges {
            backend_quarantined: false,
            external_yield_sync: Vec::new(),
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
    }
}
