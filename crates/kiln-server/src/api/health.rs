use axum::{extract::State, routing::get, Json, Router};
use serde::Serialize;

use crate::state::{AppState, ModelBackend};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    model: String,
    backend: &'static str,
    scheduler: Option<SchedulerStats>,
}

#[derive(Serialize)]
struct SchedulerStats {
    waiting: usize,
    running: usize,
    blocks_used: usize,
    blocks_free: usize,
    blocks_total: usize,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let (backend_name, scheduler_stats) = match state.backend.as_ref() {
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
            )
        }
        ModelBackend::Real(_) => ("model", None),
    };

    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        model: format!(
            "qwen3.5-4b ({}L, {}H, {}KV)",
            state.model_config.num_layers,
            state.model_config.num_attention_heads,
            state.model_config.num_kv_heads,
        ),
        backend: backend_name,
        scheduler: scheduler_stats,
    })
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
}
