//! GET /metrics — Prometheus text exposition format.

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};

use crate::metrics::SnapshotGauges;
use crate::state::{AppState, ModelBackend};
use kiln_train::TrainingState;

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    // Snapshot scheduler gauges.
    let (scheduler_waiting, scheduler_running, blocks_used, blocks_total) =
        match state.backend.as_ref() {
            ModelBackend::Mock { scheduler, .. } => {
                let sched = scheduler.lock().await;
                let bm = sched.block_manager();
                (
                    sched.num_waiting(),
                    sched.num_running(),
                    bm.num_used(),
                    bm.num_blocks(),
                )
            }
            ModelBackend::Real { block_manager, .. } => {
                let bm = block_manager.lock().unwrap();
                (0, 0, bm.num_used(), bm.num_blocks())
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

    let gauges = SnapshotGauges {
        scheduler_waiting,
        scheduler_running,
        blocks_used,
        blocks_total,
        vram_total: state.memory_budget.total_vram_bytes,
        vram_model: state.memory_budget.model_memory_bytes,
        vram_kv_cache: state.memory_budget.kv_cache_bytes,
        vram_training_budget: state.memory_budget.training_budget_bytes,
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
