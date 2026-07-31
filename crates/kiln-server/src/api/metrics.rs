//! GET /metrics — Prometheus text exposition format.

use axum::{
    Router,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
    routing::get,
};

use crate::batching_engine::BatchingEngineSnapshot;
use crate::memory_observability::CachedMemoryGovernorObservation;
use crate::metrics::SnapshotGauges;
use crate::state::{AppState, ModelBackend};
use kiln_train::TrainingState;

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let cuda_synchronization =
        crate::accelerator_runtime::cuda_synchronization_runtime_stats(state.model_weight_device);
    let rocm_synchronization = state.observe_rocm_runtime_health();
    let rocm_graph_observation = crate::rocm_graph_observability::observe_rocm_graphs(&state);
    // Snapshot scheduler gauges.
    let (
        scheduler_waiting,
        scheduler_running,
        blocks_used,
        blocks_total,
        prefix_cache,
        batching_engine_enabled,
        batching_engine,
        backend_health_quarantined,
        external_yield_sync,
        batched_state_cache,
        resident_recurrent_state,
    ) = match state.backend.as_ref() {
        ModelBackend::Mock { scheduler, .. } => {
            let sched = scheduler.lock().await;
            let bm = sched.block_manager();
            (
                sched.num_waiting(),
                sched.num_running(),
                bm.num_used(),
                bm.num_blocks(),
                sched.prefix_cache_stats(),
                false,
                BatchingEngineSnapshot::default(),
                false,
                Vec::new(),
                kiln_model::BatchedStateCacheStats::default(),
                kiln_model::GdnRecurrentStateResidencyStats::default(),
            )
        }
        ModelBackend::Real {
            block_manager,
            prefix_cache,
            backend_health,
            batching_engine,
            runner,
            ..
        } => {
            let (blocks_used, blocks_total) = {
                let bm = block_manager.lock().unwrap();
                (bm.num_used(), bm.num_blocks())
            };
            let prefix_cache = {
                let cache = prefix_cache.lock().unwrap();
                cache.stats()
            };
            let batching_engine_snapshot = batching_engine.cached_snapshot();
            let backend_health_snapshot = backend_health.snapshot();
            let runner = runner
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                0,
                0,
                blocks_used,
                blocks_total,
                prefix_cache,
                true,
                batching_engine_snapshot,
                backend_health_snapshot.quarantined,
                backend_health.external_yield_sync_stats(),
                runner.batched_state_cache_stats(),
                runner.gdn_recurrent_state_residency_stats(),
            )
        }
    };

    // Training active?
    let training_active = {
        let jobs = state.training_jobs.read().unwrap();
        if jobs.values().any(|j| j.state == TrainingState::Running) {
            1
        } else {
            0
        }
    };

    let active_adapter = state.active_adapter_name.read().unwrap().clone();
    let (openenv_runs_active, openenv_runs_queued, openenv_runs_tracked) =
        state.openenv_runs.counts();
    let (rendered_prompt_cache_hits, rendered_prompt_cache_misses, rendered_prompt_cache_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    let (prompt_token_cache_hits, prompt_token_cache_misses, prompt_token_cache_entries) =
        state.prompt_token_cache.lock().unwrap().stats();
    let gauges = SnapshotGauges {
        memory_governor: CachedMemoryGovernorObservation::capture_global_for(
            state.vram_probe_selector,
        ),
        backend_quarantined: backend_health_quarantined || rocm_synchronization.cleanup_quarantined,
        external_yield_sync,
        cuda_synchronization,
        rocm_synchronization_mode: state
            .accelerator_runtime_policy
            .rocm_synchronization_mode
            .effective
            .as_str(),
        rocm_synchronization,
        rocm_graph: rocm_graph_observation.stats,
        rocm_graph_unavailable_reason: rocm_graph_observation.stats_unavailable_reason,
        rocm_graph_telemetry: rocm_graph_observation.telemetry,
        rocm_graph_telemetry_unavailable_reason: rocm_graph_observation
            .telemetry_unavailable_reason,
        rocm_w8_lm_head: kiln_model::rocm_w8_proj::stats(),
        scheduler_waiting,
        scheduler_running,
        blocks_used,
        blocks_total,
        vram_total: state.memory_budget.total_vram_bytes,
        vram_model: state.memory_budget.model_memory_bytes,
        vram_model_estimated: state.memory_budget.estimated_model_memory_bytes,
        vram_post_load_used: state.memory_budget.post_load_used_vram_bytes,
        vram_prefill_peak_used: state.memory_budget.peak_prefill_used_vram_bytes(),
        vram_kv_cache: state.memory_budget.kv_cache_bytes,
        vram_training_budget: state.memory_budget.training_budget_bytes,
        vulkan_buffers: kiln_model::vulkan_buffer_allocation_stats(),
        vulkan_buffer_pool: kiln_model::vulkan_buffer_pool_stats(),
        batched_state_cache,
        resident_recurrent_state,
        prefix_cache,
        rendered_prompt_cache_hits,
        rendered_prompt_cache_misses,
        rendered_prompt_cache_entries,
        prompt_token_cache_hits,
        prompt_token_cache_misses,
        prompt_token_cache_entries,
        batching_engine_enabled,
        batching_engine,
        training_active,
        openenv_runs_active,
        openenv_runs_queued,
        openenv_runs_tracked,
        active_adapter,
    };

    let body = state.metrics.render(&gauges);

    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/metrics", get(metrics_handler))
}
