//! GET /metrics — Prometheus text exposition format.

use axum::{
    Router,
    extract::State,
    http::{StatusCode, header},
    response::IntoResponse,
    routing::get,
};

use crate::batching_engine::BatchingEngineSnapshot;
use crate::metrics::SnapshotGauges;
use crate::state::{AppState, ModelBackend};
use kiln_train::TrainingState;

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Snapshot scheduler gauges.
    let (
        scheduler_waiting,
        scheduler_running,
        blocks_used,
        blocks_total,
        prefix_cache,
        decode_batcher_enabled,
        decode_batcher,
        batching_engine_enabled,
        batching_engine,
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
                kiln_model::DecodeBatcherStats::default(),
                false,
                BatchingEngineSnapshot::default(),
            )
        }
        ModelBackend::Real {
            block_manager,
            prefix_cache,
            batching_engine,
            decode_batcher,
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
            let batcher_stats = decode_batcher
                .as_ref()
                .map(|batcher| batcher.stats())
                .unwrap_or_default();
            let batching_engine_snapshot = match batching_engine {
                Some(engine) => engine.snapshot().await.unwrap_or_default(),
                None => BatchingEngineSnapshot::default(),
            };
            (
                0,
                0,
                blocks_used,
                blocks_total,
                prefix_cache,
                decode_batcher.is_some(),
                batcher_stats,
                batching_engine.is_some(),
                batching_engine_snapshot,
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
    let (rendered_prompt_cache_hits, rendered_prompt_cache_misses, rendered_prompt_cache_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    let (prompt_token_cache_hits, prompt_token_cache_misses, prompt_token_cache_entries) =
        state.prompt_token_cache.lock().unwrap().stats();

    let gauges = SnapshotGauges {
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
        prefix_cache,
        rendered_prompt_cache_hits,
        rendered_prompt_cache_misses,
        rendered_prompt_cache_entries,
        prompt_token_cache_hits,
        prompt_token_cache_misses,
        prompt_token_cache_entries,
        decode_batcher_enabled,
        decode_batcher,
        batching_engine_enabled,
        batching_engine,
        training_active,
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
