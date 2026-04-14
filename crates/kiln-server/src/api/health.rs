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
    gpu_memory: Option<GpuMemoryInfo>,
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
    total_vram_gb: f64,
    model_gb: f64,
    kv_cache_gb: f64,
    training_budget_gb: f64,
    inference_memory_fraction: f64,
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
        ModelBackend::Real { .. } => ("model", None),
    };

    let gpu_memory = if state.memory_budget.total_vram_bytes > 0 {
        let b = &state.memory_budget;
        Some(GpuMemoryInfo {
            total_vram_gb: b.total_vram_bytes as f64 / 1e9,
            model_gb: b.model_memory_bytes as f64 / 1e9,
            kv_cache_gb: b.kv_cache_bytes as f64 / 1e9,
            training_budget_gb: b.training_budget_bytes as f64 / 1e9,
            inference_memory_fraction: b.inference_memory_fraction,
        })
    } else {
        None
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
        gpu_memory,
    })
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
}
