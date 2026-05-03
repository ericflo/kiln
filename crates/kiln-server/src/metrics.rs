//! Prometheus metrics collection for kiln.
//!
//! Uses atomic counters and gauges — no external dependencies.
//! The `/metrics` endpoint renders all metrics in Prometheus text exposition format.

use kiln_scheduler::PrefixCacheStats;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

    // Training counters
    pub training_sft_completed: AtomicU64,
    pub training_sft_failed: AtomicU64,
    pub training_sft_cancelled: AtomicU64,
    pub training_grpo_completed: AtomicU64,
    pub training_grpo_failed: AtomicU64,
    pub training_grpo_cancelled: AtomicU64,
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
            training_sft_completed: AtomicU64::new(0),
            training_sft_failed: AtomicU64::new(0),
            training_sft_cancelled: AtomicU64::new(0),
            training_grpo_completed: AtomicU64::new(0),
            training_grpo_failed: AtomicU64::new(0),
            training_grpo_cancelled: AtomicU64::new(0),
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

        out
    }
}

/// Dynamic gauge values snapshotted at render time.
pub struct SnapshotGauges {
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

fn observe_bucket(buckets: &[AtomicU64], bounds_us: &[u64], value_us: u64) {
    let index = bounds_us
        .iter()
        .position(|&bound| value_us <= bound)
        .unwrap_or(bounds_us.len());
    buckets[index].fetch_add(1, Ordering::Relaxed);
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

fn prom_counter(out: &mut String, name: &str, label: &str, value: &str, count: u64) {
    out.push_str(&format!("{name}{{{label}=\"{value}\"}} {count}\n"));
}

fn prom_counter2(out: &mut String, name: &str, l1: &str, v1: &str, l2: &str, v2: &str, count: u64) {
    out.push_str(&format!("{name}{{{l1}=\"{v1}\",{l2}=\"{v2}\"}} {count}\n"));
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let gauges = SnapshotGauges {
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
            },
            rendered_prompt_cache_hits: 6,
            rendered_prompt_cache_misses: 3,
            rendered_prompt_cache_entries: 5,
            prompt_token_cache_hits: 5,
            prompt_token_cache_misses: 2,
            prompt_token_cache_entries: 4,
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
        assert!(output.contains("kiln_prefix_cache_lookups_total{result=\"hit\"} 7"));
        assert!(output.contains("kiln_prefix_cache_lookups_total{result=\"miss\"} 3"));
        assert!(output.contains("kiln_prefix_cache_hit_tokens_total 112"));
        assert!(output.contains("kiln_prefix_cache_hit_blocks_total 7"));
        assert!(output.contains("kiln_prefix_cache_cached_blocks 64"));
        assert!(output.contains("kiln_prefix_cache_max_blocks 128"));
        assert!(output.contains("kiln_prefix_cache_cached_entries 4"));
        assert!(output.contains("kiln_prefix_cache_max_entries 8"));
        assert!(output.contains("kiln_prefix_cache_state_bytes 196"));
        assert!(output.contains("kiln_prefix_cache_max_state_bytes 392"));
        assert!(output.contains("kiln_rendered_prompt_cache_lookups_total{result=\"hit\"} 6"));
        assert!(output.contains("kiln_rendered_prompt_cache_lookups_total{result=\"miss\"} 3"));
        assert!(output.contains("kiln_rendered_prompt_cache_entries 5"));
        assert!(output.contains("kiln_prompt_token_cache_lookups_total{result=\"hit\"} 5"));
        assert!(output.contains("kiln_prompt_token_cache_lookups_total{result=\"miss\"} 2"));
        assert!(output.contains("kiln_prompt_token_cache_entries 4"));
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
    }

    #[test]
    fn test_base_adapter_rendering() {
        let m = Metrics::new();
        let gauges = SnapshotGauges {
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
            training_active: 0,
            active_adapter: None,
        };
        let output = m.render(&gauges);
        assert!(output.contains("kiln_active_adapter{name=\"base\"} 1"));
    }
}
