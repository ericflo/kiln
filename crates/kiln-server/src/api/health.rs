use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use kiln_core::config_hashes::ConfigHashes;
use kiln_scheduler::PrefixCacheStats;
use serde::Serialize;
use std::sync::atomic::Ordering;

use crate::batching_engine::BatchingEngineSnapshot;
use crate::config::{ConfigValueSource, ModelDefaultsProfile};
use crate::recent_requests::RequestRecord;
use crate::state::{AppState, ModelBackend};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_seconds: u64,
    model: String,
    backend: &'static str,
    http: HttpRuntimeInfo,
    model_defaults_profile: ModelDefaultsProfile,
    eval_mode: bool,
    default_thinking_enabled: Option<bool>,
    default_thinking_budget_tokens: Option<usize>,
    default_thinking_budget_ms: Option<u64>,
    fold_reasoning_into_content: bool,
    config_hashes: ConfigHashes,
    active_adapter: Option<String>,
    loaded_adapter: Option<String>,
    loaded_adapter_count: usize,
    adapters_loaded: usize,
    request_count: u64,
    requests: RequestMetrics,
    recent_requests: RecentRequestMetrics,
    scheduler: Option<SchedulerStats>,
    /// [agent] self_improve scheduler status (None = not armed).
    #[serde(skip_serializing_if = "Option::is_none")]
    self_improve_scheduler: Option<crate::state::SelfImproveSchedulerStatus>,
    gpu_memory: Option<GpuMemoryInfo>,
    prefix_cache: PrefixCacheInfo,
    prompt_caches: PromptCachesInfo,
    decode_runtime: DecodeRuntimeInfo,
    training: TrainingInfo,
    checks: Vec<HealthCheck>,
}

#[derive(Serialize)]
struct HttpRuntimeInfo {
    /// Resolved per-connection `SO_SNDBUF` request; null leaves the OS default.
    send_buffer_requested_bytes: Option<usize>,
    /// Raw listener `getsockopt(SO_SNDBUF)` result captured before readiness.
    send_buffer_kernel_readback_bytes: Option<usize>,
    /// Preflight result after normalizing platform-specific accounting.
    send_buffer_effective_bytes: Option<usize>,
}

#[derive(Serialize)]
struct SchedulerStats {
    waiting: usize,
    running: usize,
    blocks_used: usize,
    blocks_free: usize,
    blocks_total: usize,
}

#[derive(Serialize)]
struct GpuMemoryInfo {
    total_vram_bytes: u64,
    model_bytes: u64,
    estimated_model_bytes: u64,
    post_load_used_bytes: u64,
    peak_prefill_used_bytes: u64,
    kv_cache_bytes: u64,
    training_budget_bytes: u64,
    allocated_bytes: u64,
    reserved_bytes: u64,
    total_vram_gb: f64,
    model_gb: f64,
    estimated_model_gb: f64,
    post_load_used_gb: f64,
    peak_prefill_used_gb: f64,
    kv_cache_gb: f64,
    training_budget_gb: f64,
    allocated_gb: f64,
    reserved_gb: f64,
    inference_memory_fraction: f64,
    /// The memory governor's LIVE, all-process view right now (driver counters /
    /// MemAvailable) — what kiln actually sees, including any coexisting GPU job.
    /// Distinct from the static startup budget above. `None` on CPU / when
    /// undetectable.
    live: Option<LiveMemory>,
}

#[derive(Serialize)]
struct LiveMemory {
    total_gb: f64,
    /// Used by ALL processes right now (includes coexisting GPU jobs).
    used_gb: f64,
    free_gb: f64,
    /// Free − safety floor − soft reservations: what kiln may allocate now.
    available_gb: f64,
    /// Soft reservations announced by training/other planned allocations.
    soft_reserved_gb: f64,
    /// Comfortable | Moderate | Tight | Critical.
    pressure: String,
    /// Probe provenance (e.g. linux-drm-sysfs, nvidia-smi).
    source: String,
    /// True when GPU shares system RAM (APU / Apple Silicon).
    unified: bool,
}

#[derive(Serialize)]
struct RequestMetrics {
    total: u64,
    ok: u64,
    error: u64,
    timeout: u64,
    rejected: u64,
    active: u64,
    active_peak: u64,
}

#[derive(Serialize)]
struct RecentRequestMetrics {
    retained: usize,
    capacity: usize,
    latency_ms: LatencyPercentiles,
    tokens_per_second: f64,
    timeout_count: u64,
    error_count: u64,
    last_error: Option<LastErrorSummary>,
}

#[derive(Serialize)]
struct LatencyPercentiles {
    p50: f64,
    p95: f64,
    p99: f64,
}

#[derive(Serialize)]
struct LastErrorSummary {
    id: String,
    timestamp_unix_ms: u64,
    finish_reason: String,
    error: Option<String>,
    duration_ms: u64,
    adapter: Option<String>,
}

#[derive(Serialize)]
struct PrefixCacheInfo {
    enabled: bool,
    lookup_hits: u64,
    lookup_misses: u64,
    hit_tokens: u64,
    hit_blocks: u64,
    cached_blocks: usize,
    max_blocks: usize,
    cached_entries: usize,
    max_entries: usize,
    cached_state_bytes: u64,
    max_state_bytes: u64,
    block_utilization: f64,
    entry_utilization: f64,
    state_utilization: f64,
}

#[derive(Serialize)]
struct PromptCachesInfo {
    rendered_prompt: PromptCacheInfo,
    prompt_token: PromptCacheInfo,
}

#[derive(Serialize)]
struct PromptCacheInfo {
    hits: u64,
    misses: u64,
    entries: usize,
}

#[derive(Serialize)]
struct DecodeRuntimeInfo {
    cuda_graphs: GraphInfo,
    rocm_graphs: RocmGraphInfo,
    metal_graphs: GraphInfo,
    kv_autoscaler: crate::kv_autoscaler::KvAutoscalerState,
    memory_governor: MemoryGovernorRuntimeInfo,
    decode_batcher: Option<DecodeBatcherInfo>,
    batching_engine: Option<BatchingEngineInfo>,
}

#[derive(Serialize)]
struct MemoryGovernorRuntimeInfo {
    reclaim_mode: &'static str,
    automatic_monitor_enabled: bool,
    source: &'static str,
}

fn memory_governor_runtime_info() -> MemoryGovernorRuntimeInfo {
    let governor = kiln_memory::MemoryGovernor::global();
    let enabled = governor.monitor_started();
    MemoryGovernorRuntimeInfo {
        reclaim_mode: governor.config().reclaim_mode.as_str(),
        automatic_monitor_enabled: enabled,
        source: if std::env::var_os(kiln_memory::MEMORY_RECLAIM_MODE_ENV).is_some() {
            "environment"
        } else {
            "default"
        },
    }
}

#[derive(Serialize)]
struct GraphInfo {
    enabled: Option<bool>,
    state: &'static str,
}

fn graph_info(enabled: Option<bool>) -> GraphInfo {
    GraphInfo {
        enabled,
        state: match enabled {
            Some(true) => "enabled",
            Some(false) => "disabled",
            None => "busy",
        },
    }
}

#[derive(Serialize)]
struct RocmGraphInfo {
    requested: Option<bool>,
    capture_requested: Option<bool>,
    enabled: Option<bool>,
    capture_enabled: Option<bool>,
    state: &'static str,
    capture_attempts: Option<u64>,
    capture_successes: Option<u64>,
    capture_deferrals: Option<u64>,
    capture_failures: Option<u64>,
    replay_attempts: Option<u64>,
    replay_successes: Option<u64>,
    replay_failures: Option<u64>,
    failures: Option<u64>,
    decode_owner_release_count: Option<u64>,
    decode_owner_graph_release_count: Option<u64>,
    captured_graph_count: Option<usize>,
    tracked_decode_owner_count: Option<usize>,
}

fn rocm_graph_info(stats: Option<kiln_model::RocmGraphStats>) -> RocmGraphInfo {
    let enabled = stats.map(|snapshot| snapshot.enabled);
    RocmGraphInfo {
        requested: stats.map(|snapshot| snapshot.requested),
        capture_requested: stats.map(|snapshot| snapshot.capture_requested),
        enabled,
        capture_enabled: stats.map(|snapshot| snapshot.capture_enabled),
        state: match enabled {
            Some(true) => "enabled",
            Some(false) => "disabled",
            None => "busy",
        },
        capture_attempts: stats.map(|snapshot| snapshot.capture_attempts),
        capture_successes: stats.map(|snapshot| snapshot.capture_successes),
        capture_deferrals: stats.map(|snapshot| snapshot.capture_deferrals),
        capture_failures: stats.map(|snapshot| snapshot.capture_failures),
        replay_attempts: stats.map(|snapshot| snapshot.replay_attempts),
        replay_successes: stats.map(|snapshot| snapshot.replay_successes),
        replay_failures: stats.map(|snapshot| snapshot.replay_failures),
        failures: stats.map(|snapshot| snapshot.failures),
        decode_owner_release_count: stats.map(|snapshot| snapshot.decode_owner_release_count),
        decode_owner_graph_release_count: stats
            .map(|snapshot| snapshot.decode_owner_graph_release_count),
        captured_graph_count: stats.map(|snapshot| snapshot.captured_graph_count),
        tracked_decode_owner_count: stats.map(|snapshot| snapshot.tracked_decode_owner_count),
    }
}

#[derive(Serialize)]
struct DecodeBatcherInfo {
    submitted_jobs: usize,
    executed_batches: usize,
    executed_rows: usize,
    runner_calls: usize,
    runner_calls_per_token: Option<f64>,
    max_runner_calls_per_token: usize,
    runner_call_budget_per_token: usize,
    runner_call_budget_exceeded: bool,
    max_observed_batch: usize,
    runner_busy_jobs: usize,
    failed_jobs: usize,
}

#[derive(Serialize)]
struct BatchingEngineInfo {
    snapshot_age_ms: u64,
    stream_stall_grace_ms: u64,
    stream_stall_grace_source: ConfigValueSource,
    accepting: bool,
    queue_depth: usize,
    active_decode: usize,
    max_prefill_admission_quantum: usize,
    current_batch_size: usize,
    last_batch_size: usize,
    max_observed_batch_size: usize,
    last_forward_ms: f64,
    last_prefill_ms: f64,
    total_decode_forwards: u64,
    total_batched_decode_forwards: u64,
    total_decode_rows: u64,
    total_prefill_admission_cycles: u64,
    total_decode_tokens: u64,
    total_prefill_tokens: u64,
    total_errors: u64,
    response_delivery_in_flight: usize,
    response_delivery_backpressured: usize,
    response_delivery_pending_terminal: usize,
    response_backpressure_events: u64,
    response_backpressure_wait_ms: u64,
    response_stall_evictions: u64,
    response_channel_closed: u64,
    adapter_groups_waiting: usize,
    prefix_deferred_waiting: usize,
    prefix_admission_deferrals: u64,
}

#[derive(Serialize)]
struct TrainingInfo {
    active_job: Option<ActiveJobInfo>,
    queued: usize,
}

#[derive(Serialize)]
struct ActiveJobInfo {
    job_id: String,
    progress: f32,
}

#[derive(Serialize)]
struct HealthCheck {
    name: &'static str,
    pass: bool,
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let uptime_seconds = state.started_at.elapsed().as_secs();

    // Adapter info
    let active_adapter = state.active_adapter_name.read().unwrap().clone();
    let loaded_adapter = state.loaded_adapter_name.read().unwrap().clone();
    let loaded_adapter_count = usize::from(loaded_adapter.is_some());

    let adapter_dir = state.adapter_dir.clone();
    let adapters_loaded = tokio::task::spawn_blocking(move || count_adapter_dirs(&adapter_dir))
        .await
        .unwrap_or(0);

    // Scheduler stats (works for both Mock and Real backends)
    let (
        backend_name,
        scheduler_stats,
        prefix_cache,
        decode_batcher,
        batching_engine,
        cuda_graphs,
        rocm_graphs,
        metal_graphs,
        model_loaded,
    ) = match state.backend.as_ref() {
        ModelBackend::Mock { scheduler, .. } => {
            let sched = scheduler.lock().await;
            let bm = sched.block_manager();
            (
                "mock",
                Some(SchedulerStats {
                    waiting: sched.num_waiting(),
                    running: sched.num_running(),
                    blocks_used: bm.num_used(),
                    blocks_free: bm.num_free(),
                    blocks_total: bm.num_blocks(),
                }),
                PrefixCacheInfo::from(sched.prefix_cache_stats()),
                None,
                None,
                graph_info(Some(false)),
                rocm_graph_info(Some(kiln_model::RocmGraphStats::default())),
                graph_info(Some(false)),
                true,
            )
        }
        ModelBackend::Real {
            runner,
            block_manager,
            prefix_cache,
            batching_engine,
            decode_batcher,
            ..
        } => {
            let scheduler_stats = {
                let bm = block_manager.lock().unwrap();
                Some(SchedulerStats {
                    waiting: 0,
                    running: 0,
                    blocks_used: bm.num_used(),
                    blocks_free: bm.num_free(),
                    blocks_total: bm.num_blocks(),
                })
            };
            let prefix_cache = {
                let cache = prefix_cache.lock().unwrap();
                PrefixCacheInfo::from(cache.stats())
            };
            let decode_batcher = decode_batcher
                .as_ref()
                .map(|batcher| DecodeBatcherInfo::from(batcher.stats()));
            let batching_engine = match batching_engine {
                Some(engine) => Some(BatchingEngineInfo::from(engine.cached_snapshot())),
                None => None,
            };
            let (cuda_graph_enabled, rocm_graph_stats, metal_graph_enabled) =
                match runner.try_read() {
                    Ok(runner) => (
                        runner.cuda_graph_enabled().ok(),
                        runner.rocm_graph_stats().ok(),
                        runner.metal_graph_enabled().ok(),
                    ),
                    Err(_) => (None, None, None),
                };
            (
                "model",
                scheduler_stats,
                prefix_cache,
                decode_batcher,
                batching_engine,
                graph_info(cuda_graph_enabled),
                rocm_graph_info(rocm_graph_stats),
                graph_info(metal_graph_enabled),
                true,
            )
        }
    };

    let requests = request_metrics(&state);
    let request_count = requests.total;
    let recent_requests = recent_request_metrics(&state);
    let prompt_caches = prompt_caches(&state);
    let decode_runtime = DecodeRuntimeInfo {
        cuda_graphs,
        rocm_graphs,
        metal_graphs,
        kv_autoscaler: state.kv_autoscaler,
        memory_governor: memory_governor_runtime_info(),
        decode_batcher,
        batching_engine,
    };

    // GPU memory
    let gpu_memory = if state.memory_budget.total_vram_bytes > 0 {
        let b = &state.memory_budget;
        let peak_prefill_used_bytes = b.peak_prefill_used_vram_bytes();
        let allocated_bytes = b.model_memory_bytes.saturating_add(b.kv_cache_bytes);
        let reserved_bytes = allocated_bytes.saturating_add(b.training_budget_bytes);
        // The governor's LIVE view right now — what kiln actually sees, including
        // a coexisting llama.cpp / vLLM / training job (the probe is all-process).
        let live = {
            let g = kiln_memory::MemoryGovernor::global();
            let s = g.snapshot();
            (s.total_bytes > 0).then(|| LiveMemory {
                total_gb: s.total_bytes as f64 / 1e9,
                used_gb: s.used_bytes as f64 / 1e9,
                free_gb: s.free_bytes as f64 / 1e9,
                available_gb: g.available_bytes() as f64 / 1e9,
                soft_reserved_gb: g.soft_reserved_bytes() as f64 / 1e9,
                pressure: format!("{:?}", g.pressure()),
                source: s.source.to_string(),
                unified: s.unified,
            })
        };
        Some(GpuMemoryInfo {
            total_vram_bytes: b.total_vram_bytes,
            model_bytes: b.model_memory_bytes,
            estimated_model_bytes: b.estimated_model_memory_bytes,
            post_load_used_bytes: b.post_load_used_vram_bytes,
            peak_prefill_used_bytes,
            kv_cache_bytes: b.kv_cache_bytes,
            training_budget_bytes: b.training_budget_bytes,
            allocated_bytes,
            reserved_bytes,
            total_vram_gb: b.total_vram_bytes as f64 / 1e9,
            model_gb: b.model_memory_bytes as f64 / 1e9,
            estimated_model_gb: b.estimated_model_memory_bytes as f64 / 1e9,
            post_load_used_gb: b.post_load_used_vram_bytes as f64 / 1e9,
            peak_prefill_used_gb: peak_prefill_used_bytes as f64 / 1e9,
            kv_cache_gb: b.kv_cache_bytes as f64 / 1e9,
            training_budget_gb: b.training_budget_bytes as f64 / 1e9,
            allocated_gb: allocated_bytes as f64 / 1e9,
            reserved_gb: reserved_bytes as f64 / 1e9,
            inference_memory_fraction: b.inference_memory_fraction,
            live,
        })
    } else {
        None
    };

    // Training info
    let (active_job, _running_count) = {
        let jobs = state.training_jobs.read().unwrap();
        let active = jobs
            .values()
            .find(|j| j.state == kiln_train::TrainingState::Running);
        let active_info = active.map(|j| ActiveJobInfo {
            job_id: j.job_id.clone(),
            progress: j.progress,
        });
        let running = if active.is_some() { 1 } else { 0 };
        (active_info, running)
    };
    let queued = state.training_queue.lock().unwrap().len();

    let training = TrainingInfo { active_job, queued };

    // Health checks
    let scheduler_responsive = scheduler_stats.is_some();
    let inference_prewarm_complete = state.inference_prewarm_complete.load(Ordering::Acquire);
    let checks = vec![
        HealthCheck {
            name: "model_loaded",
            pass: model_loaded,
        },
        HealthCheck {
            name: "scheduler_responsive",
            pass: scheduler_responsive,
        },
        HealthCheck {
            name: "inference_prewarm_complete",
            pass: inference_prewarm_complete,
        },
    ];

    let serving_ready = model_loaded && scheduler_responsive;
    let status = if serving_ready { "ok" } else { "degraded" };

    let response = HealthResponse {
        status,
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds,
        model: format!(
            "{} ({}L, {}H, {}KV)",
            state.served_model_id,
            state.model_config.num_layers,
            state.model_config.num_attention_heads,
            state.model_config.num_kv_heads,
        ),
        backend: backend_name,
        http: HttpRuntimeInfo {
            send_buffer_requested_bytes: state.http_send_buffer_bytes,
            send_buffer_kernel_readback_bytes: state.http_send_buffer_preflight_actual_bytes,
            send_buffer_effective_bytes: state.http_send_buffer_preflight_effective_bytes,
        },
        model_defaults_profile: state.model_defaults_profile,
        eval_mode: state.eval_mode,
        default_thinking_enabled: state.default_thinking_enabled,
        default_thinking_budget_tokens: state.default_thinking_budget_tokens,
        default_thinking_budget_ms: state.default_thinking_budget_ms,
        fold_reasoning_into_content: state.fold_reasoning_into_content,
        config_hashes: state.config_hashes.clone(),
        active_adapter,
        loaded_adapter,
        loaded_adapter_count,
        adapters_loaded,
        request_count,
        requests,
        recent_requests,
        scheduler: scheduler_stats,
        self_improve_scheduler: state.self_improve_scheduler.read().unwrap().clone(),
        gpu_memory,
        prefix_cache,
        prompt_caches,
        decode_runtime,
        training,
        checks,
    };

    if serving_ready {
        (StatusCode::OK, Json(response)).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
    }
}

fn count_adapter_dirs(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                // Dot-prefixed names are kiln internals (.composed,
                // .upload-tmp-*, .eval, .requests, .kiln-jobs), not adapters.
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .count()
        })
        .unwrap_or(0)
}

fn request_metrics(state: &AppState) -> RequestMetrics {
    let ok = state.metrics.requests_ok.load(Ordering::Relaxed);
    let error = state.metrics.requests_error.load(Ordering::Relaxed);
    let timeout = state.metrics.requests_timeout.load(Ordering::Relaxed);
    let rejected = state.metrics.requests_rejected.load(Ordering::Relaxed);

    RequestMetrics {
        total: ok
            .saturating_add(error)
            .saturating_add(timeout)
            .saturating_add(rejected),
        ok,
        error,
        timeout,
        rejected,
        active: state.metrics.active_requests.load(Ordering::Relaxed),
        active_peak: state.metrics.active_requests_peak.load(Ordering::Relaxed),
    }
}

fn recent_request_metrics(state: &AppState) -> RecentRequestMetrics {
    let (records, capacity) = match state.recent_requests.lock() {
        Ok(ring) => (ring.snapshot(), ring.capacity()),
        Err(poisoned) => {
            let ring = poisoned.into_inner();
            (ring.snapshot(), ring.capacity())
        }
    };

    summarize_recent_requests(records, capacity)
}

fn summarize_recent_requests(records: Vec<RequestRecord>, capacity: usize) -> RecentRequestMetrics {
    let mut latencies: Vec<u64> = records.iter().map(|r| r.duration_ms).collect();
    latencies.sort_unstable();

    let completion_tokens = records
        .iter()
        .map(|r| r.completion_tokens as u64)
        .sum::<u64>();
    let duration_ms = records.iter().map(|r| r.duration_ms).sum::<u64>();

    RecentRequestMetrics {
        retained: records.len(),
        capacity,
        latency_ms: LatencyPercentiles {
            p50: percentile_u64(&latencies, 0.50),
            p95: percentile_u64(&latencies, 0.95),
            p99: percentile_u64(&latencies, 0.99),
        },
        tokens_per_second: if duration_ms > 0 {
            completion_tokens as f64 / (duration_ms as f64 / 1000.0)
        } else {
            0.0
        },
        timeout_count: records.iter().filter(|r| is_timeout_record(r)).count() as u64,
        error_count: records.iter().filter(|r| is_error_record(r)).count() as u64,
        last_error: records
            .iter()
            .find(|r| is_timeout_record(r) || is_error_record(r))
            .map(|r| LastErrorSummary {
                id: r.id.clone(),
                timestamp_unix_ms: r.timestamp_unix_ms,
                finish_reason: r.finish_reason.clone(),
                error: r.error.clone(),
                duration_ms: r.duration_ms,
                adapter: r.adapter.clone(),
            }),
    }
}

fn percentile_u64(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo] as f64
    } else {
        let frac = rank - lo as f64;
        sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac
    }
}

fn is_timeout_record(record: &RequestRecord) -> bool {
    record.finish_reason == "timeout"
}

fn is_error_record(record: &RequestRecord) -> bool {
    record.error.is_some() || record.finish_reason == "error"
}

fn prompt_caches(state: &AppState) -> PromptCachesInfo {
    let (rendered_prompt_hits, rendered_prompt_misses, rendered_prompt_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    let (prompt_token_hits, prompt_token_misses, prompt_token_entries) =
        state.prompt_token_cache.lock().unwrap().stats();

    PromptCachesInfo {
        rendered_prompt: PromptCacheInfo {
            hits: rendered_prompt_hits,
            misses: rendered_prompt_misses,
            entries: rendered_prompt_entries,
        },
        prompt_token: PromptCacheInfo {
            hits: prompt_token_hits,
            misses: prompt_token_misses,
            entries: prompt_token_entries,
        },
    }
}

fn utilization(used: u64, capacity: u64) -> f64 {
    if capacity == 0 {
        0.0
    } else {
        used as f64 / capacity as f64
    }
}

impl From<PrefixCacheStats> for PrefixCacheInfo {
    fn from(stats: PrefixCacheStats) -> Self {
        Self {
            enabled: stats.max_blocks > 0 || stats.max_entries > 0 || stats.max_state_bytes > 0,
            lookup_hits: stats.lookup_hits,
            lookup_misses: stats.lookup_misses,
            hit_tokens: stats.hit_tokens,
            hit_blocks: stats.hit_blocks,
            cached_blocks: stats.cached_blocks,
            max_blocks: stats.max_blocks,
            cached_entries: stats.cached_entries,
            max_entries: stats.max_entries,
            cached_state_bytes: stats.cached_state_bytes,
            max_state_bytes: stats.max_state_bytes,
            block_utilization: utilization(stats.cached_blocks as u64, stats.max_blocks as u64),
            entry_utilization: utilization(stats.cached_entries as u64, stats.max_entries as u64),
            state_utilization: utilization(stats.cached_state_bytes, stats.max_state_bytes),
        }
    }
}

impl From<kiln_model::DecodeBatcherStats> for DecodeBatcherInfo {
    fn from(stats: kiln_model::DecodeBatcherStats) -> Self {
        Self {
            submitted_jobs: stats.submitted_jobs,
            executed_batches: stats.executed_batches,
            executed_rows: stats.executed_rows,
            runner_calls: stats.runner_calls,
            runner_calls_per_token: stats.runner_calls_per_token(),
            max_runner_calls_per_token: stats.max_runner_calls_per_token,
            runner_call_budget_per_token: stats.runner_call_budget_per_token(),
            runner_call_budget_exceeded: stats.runner_call_budget_exceeded(),
            max_observed_batch: stats.max_observed_batch,
            runner_busy_jobs: stats.runner_busy_jobs,
            failed_jobs: stats.failed_jobs,
        }
    }
}

impl From<BatchingEngineSnapshot> for BatchingEngineInfo {
    fn from(snapshot: BatchingEngineSnapshot) -> Self {
        Self {
            snapshot_age_ms: snapshot.snapshot_age_ms,
            stream_stall_grace_ms: snapshot.stream_stall_grace_ms,
            stream_stall_grace_source: snapshot.stream_stall_grace_source,
            accepting: snapshot.accepting,
            queue_depth: snapshot.queue_depth,
            active_decode: snapshot.active_decode,
            max_prefill_admission_quantum: snapshot.max_prefill_admission_quantum,
            current_batch_size: snapshot.current_batch_size,
            last_batch_size: snapshot.last_batch_size,
            max_observed_batch_size: snapshot.max_observed_batch_size,
            last_forward_ms: snapshot.last_forward_ms,
            last_prefill_ms: snapshot.last_prefill_ms,
            total_decode_forwards: snapshot.total_decode_forwards,
            total_batched_decode_forwards: snapshot.total_batched_decode_forwards,
            total_decode_rows: snapshot.total_decode_rows,
            total_prefill_admission_cycles: snapshot.total_prefill_admission_cycles,
            total_decode_tokens: snapshot.total_decode_tokens,
            total_prefill_tokens: snapshot.total_prefill_tokens,
            total_errors: snapshot.total_errors,
            response_delivery_in_flight: snapshot.response_delivery_in_flight,
            response_delivery_backpressured: snapshot.response_delivery_backpressured,
            response_delivery_pending_terminal: snapshot.response_delivery_pending_terminal,
            response_backpressure_events: snapshot.response_backpressure_events,
            response_backpressure_wait_ms: snapshot.response_backpressure_wait_ms,
            response_stall_evictions: snapshot.response_stall_evictions,
            response_channel_closed: snapshot.response_channel_closed,
            adapter_groups_waiting: snapshot.adapter_groups_waiting,
            prefix_deferred_waiting: snapshot.prefix_deferred_waiting,
            prefix_admission_deferrals: snapshot.prefix_admission_deferrals,
        }
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use axum::body::Body;
    use axum::http::Request;
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn make_test_state() -> AppState {
        let config = ModelConfig::qwen3_5_4b();
        let sched_config = SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        };
        let scheduler = Scheduler::new(sched_config, 256);
        let engine = MockEngine::new(config.clone());
        let tokenizer = crate::api::test_tokenizer();
        AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            300,
            "Qwen3.5-4B".to_string(),
        )
    }

    #[test]
    fn rocm_graph_health_preserves_runtime_snapshot() {
        let info = rocm_graph_info(Some(kiln_model::RocmGraphStats {
            requested: true,
            capture_requested: true,
            enabled: false,
            capture_enabled: false,
            capture_attempts: 4,
            capture_successes: 2,
            capture_deferrals: 1,
            capture_failures: 1,
            replay_attempts: 9,
            replay_successes: 8,
            replay_failures: 1,
            failures: 2,
            decode_owner_release_count: 3,
            decode_owner_graph_release_count: 4,
            captured_graph_count: 0,
            tracked_decode_owner_count: 1,
        }));
        let json = serde_json::to_value(info).unwrap();

        assert_eq!(json["requested"], true);
        assert_eq!(json["capture_requested"], true);
        assert_eq!(json["enabled"], false);
        assert_eq!(json["capture_enabled"], false);
        assert_eq!(json["state"], "disabled");
        assert_eq!(json["capture_attempts"], 4);
        assert_eq!(json["capture_successes"], 2);
        assert_eq!(json["capture_deferrals"], 1);
        assert_eq!(json["capture_failures"], 1);
        assert_eq!(json["replay_attempts"], 9);
        assert_eq!(json["replay_successes"], 8);
        assert_eq!(json["replay_failures"], 1);
        assert_eq!(json["failures"], 2);
        assert_eq!(json["decode_owner_release_count"], 3);
        assert_eq!(json["decode_owner_graph_release_count"], 4);
        assert_eq!(json["captured_graph_count"], 0);
        assert_eq!(json["tracked_decode_owner_count"], 1);
    }

    #[test]
    fn batching_engine_health_preserves_delivery_counters() {
        let info = BatchingEngineInfo::from(BatchingEngineSnapshot {
            snapshot_age_ms: 125,
            stream_stall_grace_ms: 750,
            stream_stall_grace_source: ConfigValueSource::Environment,
            response_backpressure_events: 3,
            response_backpressure_wait_ms: 750,
            response_stall_evictions: 2,
            response_channel_closed: 5,
            response_delivery_in_flight: 7,
            response_delivery_backpressured: 2,
            response_delivery_pending_terminal: 1,
            ..BatchingEngineSnapshot::default()
        });
        let json = serde_json::to_value(info).unwrap();

        assert_eq!(json["snapshot_age_ms"], 125);
        assert_eq!(json["stream_stall_grace_ms"], 750);
        assert_eq!(json["stream_stall_grace_source"], "environment");
        assert_eq!(json["response_delivery_in_flight"], 7);
        assert_eq!(json["response_delivery_backpressured"], 2);
        assert_eq!(json["response_delivery_pending_terminal"], 1);
        assert_eq!(json["response_backpressure_events"], 3);
        assert_eq!(json["response_backpressure_wait_ms"], 750);
        assert_eq!(json["response_stall_evictions"], 2);
        assert_eq!(json["response_channel_closed"], 5);
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let state = make_test_state();
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["status"], "ok");
        assert!(json["uptime_seconds"].is_number());
        assert!(json["model"].as_str().unwrap().contains("Qwen3.5-4B"));
        assert_eq!(json["backend"], "mock");
        assert!(json["http"]["send_buffer_requested_bytes"].is_null());
        assert!(json["http"]["send_buffer_kernel_readback_bytes"].is_null());
        assert!(json["http"]["send_buffer_effective_bytes"].is_null());
        assert_eq!(json["model_defaults_profile"]["name"], "Qwen3.5-4B");
        assert_eq!(
            json["model_defaults_profile"]["canonical_model_id"],
            "Qwen/Qwen3.5-4B"
        );
        assert_eq!(
            json["model_defaults_profile"]["canonical_served_model_id"],
            "Qwen3.5-4B"
        );
        assert!(json["model_defaults_profile"]["server_default_thinking_enabled"].is_null());
        assert_eq!(
            json["model_defaults_profile"]["template_default_thinking_enabled"],
            true
        );
        assert_eq!(
            json["model_defaults_profile"]["eval_mode_default_thinking_enabled"],
            false
        );
        assert_eq!(
            json["model_defaults_profile"]["supports_enable_thinking_kwarg"],
            true
        );
        assert_eq!(
            json["model_defaults_profile"]["supports_tool_chat_template"],
            true
        );
        assert_eq!(json["eval_mode"], false);
        assert!(json["default_thinking_enabled"].is_null());
        assert_eq!(json["fold_reasoning_into_content"], false);
        assert!(
            json["config_hashes"]["model_config_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(
            json["config_hashes"]["tokenizer_config_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(json["config_hashes"]["chat_template_hash"].is_null());
        assert!(json["config_hashes"]["kiln_env_config_hash"].is_null());
        assert!(json["active_adapter"].is_null());
        assert!(json["loaded_adapter"].is_null());
        assert_eq!(json["loaded_adapter_count"], 0);
        assert_eq!(json["adapters_loaded"], 0);
        assert_eq!(json["request_count"], 0);
        assert_eq!(json["requests"]["total"], 0);
        assert!(json["recent_requests"].is_object());
        assert!(json["scheduler"].is_object());
        assert!(json["prefix_cache"].is_object());
        assert!(json["prompt_caches"].is_object());
        assert!(json["decode_runtime"].is_object());
        for backend in ["cuda_graphs", "rocm_graphs", "metal_graphs"] {
            assert_eq!(json["decode_runtime"][backend]["enabled"], false);
            assert_eq!(json["decode_runtime"][backend]["state"], "disabled");
        }
        let rocm_graphs = &json["decode_runtime"]["rocm_graphs"];
        assert_eq!(rocm_graphs["requested"], false);
        assert_eq!(rocm_graphs["capture_requested"], false);
        assert_eq!(rocm_graphs["capture_enabled"], false);
        for counter in [
            "capture_attempts",
            "capture_successes",
            "capture_deferrals",
            "capture_failures",
            "replay_attempts",
            "replay_successes",
            "replay_failures",
            "failures",
            "decode_owner_release_count",
            "decode_owner_graph_release_count",
            "captured_graph_count",
            "tracked_decode_owner_count",
        ] {
            assert_eq!(rocm_graphs[counter], 0, "unexpected {counter}");
        }
        assert_eq!(json["decode_runtime"]["kv_autoscaler"]["enabled"], false);
        assert_eq!(
            json["decode_runtime"]["kv_autoscaler"]["reason"],
            "mock_backend"
        );
        assert_eq!(
            json["decode_runtime"]["memory_governor"]["automatic_monitor_enabled"],
            false
        );
        assert_eq!(
            json["decode_runtime"]["memory_governor"]["reclaim_mode"],
            "off"
        );
        assert!(json["training"].is_object());
        assert_eq!(json["training"]["queued"], 0);
        assert!(json["training"]["active_job"].is_null());
        assert!(json["checks"].is_array());
        let checks = json["checks"].as_array().unwrap();
        assert!(checks.iter().all(|c| c["pass"] == true));
    }

    #[tokio::test]
    async fn test_health_reports_effective_http_send_buffer() {
        let mut state = make_test_state();
        state.http_send_buffer_bytes = Some(4096);
        state.http_send_buffer_preflight_actual_bytes = Some(8192);
        state.http_send_buffer_preflight_effective_bytes = Some(4096);
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["http"]["send_buffer_requested_bytes"], 4096);
        assert_eq!(json["http"]["send_buffer_kernel_readback_bytes"], 8192);
        assert_eq!(json["http"]["send_buffer_effective_bytes"], 4096);
    }

    #[tokio::test]
    async fn test_health_v1_alias() {
        let state = make_test_state();
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_reports_active_adapter_and_request_count() {
        let mut state = make_test_state();
        state.eval_mode = true;
        state.default_thinking_enabled = Some(false);
        state.fold_reasoning_into_content = true;
        *state.active_adapter_name.write().unwrap() = Some("eval-adapter".to_string());
        state.metrics.inc_request(crate::metrics::RequestStatus::Ok);
        state
            .metrics
            .inc_request(crate::metrics::RequestStatus::Timeout);
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["active_adapter"], "eval-adapter");
        assert_eq!(json["eval_mode"], true);
        assert_eq!(json["default_thinking_enabled"], false);
        assert_eq!(json["fold_reasoning_into_content"], true);
        assert_eq!(json["request_count"], 2);
        assert_eq!(json["requests"]["total"], 2);
        assert_eq!(json["requests"]["ok"], 1);
        assert_eq!(json["requests"]["timeout"], 1);
    }

    #[tokio::test]
    async fn test_health_ok_while_inference_prewarm_is_running() {
        let state = make_test_state();
        state
            .inference_prewarm_complete
            .store(false, Ordering::Release);
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["status"], "ok");
        let checks = json["checks"].as_array().unwrap();
        let prewarm = checks
            .iter()
            .find(|c| c["name"] == "inference_prewarm_complete")
            .unwrap();
        assert_eq!(prewarm["pass"], false);
    }

    #[tokio::test]
    async fn test_health_scheduler_stats_present() {
        let state = make_test_state();
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let sched = &json["scheduler"];
        assert_eq!(sched["waiting"], 0);
        assert_eq!(sched["running"], 0);
        assert!(sched["blocks_total"].as_u64().unwrap() > 0);
        assert_eq!(
            sched["blocks_used"].as_u64().unwrap() + sched["blocks_free"].as_u64().unwrap(),
            sched["blocks_total"].as_u64().unwrap()
        );
    }
}
