use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

use crate::state::{AppState, ModelBackend};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_seconds: u64,
    model: String,
    backend: &'static str,
    active_adapter: Option<String>,
    adapters_loaded: usize,
    scheduler: Option<SchedulerStats>,
    gpu_memory: Option<GpuMemoryInfo>,
    training: TrainingInfo,
    checks: Vec<HealthCheck>,
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
    let active_adapter = state
        .active_adapter_name
        .read()
        .unwrap()
        .clone();

    let adapters_loaded = std::fs::read_dir(&state.adapter_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .count()
        })
        .unwrap_or(0);

    // Scheduler stats (works for both Mock and Real backends)
    let (backend_name, scheduler_stats, model_loaded) = match state.backend.as_ref() {
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
                true,
            )
        }
        ModelBackend::Real { block_manager, .. } => {
            let bm = block_manager.lock().unwrap();
            (
                "model",
                Some(SchedulerStats {
                    waiting: 0,
                    running: 0,
                    blocks_used: bm.num_used(),
                    blocks_free: bm.num_free(),
                    blocks_total: bm.num_blocks(),
                }),
                true,
            )
        }
    };

    // GPU memory
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

    // Training info
    let (active_job, running_count) = {
        let jobs = state.training_jobs.read().unwrap();
        let active = jobs.values().find(|j| {
            j.state == kiln_train::TrainingState::Running
        });
        let active_info = active.map(|j| ActiveJobInfo {
            job_id: j.job_id.clone(),
            progress: j.progress,
        });
        let running = if active.is_some() { 1 } else { 0 };
        (active_info, running)
    };
    let queued = state.training_queue.lock().unwrap().len();

    let training = TrainingInfo {
        active_job,
        queued,
    };

    // Health checks
    let scheduler_responsive = scheduler_stats.is_some();
    let checks = vec![
        HealthCheck {
            name: "model_loaded",
            pass: model_loaded,
        },
        HealthCheck {
            name: "scheduler_responsive",
            pass: scheduler_responsive,
        },
    ];

    let all_pass = checks.iter().all(|c| c.pass);
    let status = if all_pass { "ok" } else { "degraded" };

    let response = HealthResponse {
        status,
        version: env!("CARGO_PKG_VERSION"),
        uptime_seconds,
        model: format!(
            "qwen3.5-4b ({}L, {}H, {}KV)",
            state.model_config.num_layers,
            state.model_config.num_attention_heads,
            state.model_config.num_kv_heads,
        ),
        backend: backend_name,
        active_adapter,
        adapters_loaded,
        scheduler: scheduler_stats,
        gpu_memory,
        training,
        checks,
    };

    if all_pass {
        (StatusCode::OK, Json(response)).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
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
        let tokenizer =
            kiln_core::tokenizer::KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B").unwrap();
        AppState::new_mock(config, scheduler, Arc::new(engine), tokenizer, 300)
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let state = make_test_state();
        let app = routes().with_state(state);

        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["status"], "ok");
        assert!(json["uptime_seconds"].is_number());
        assert!(json["model"].as_str().unwrap().contains("qwen3.5-4b"));
        assert_eq!(json["backend"], "mock");
        assert!(json["active_adapter"].is_null());
        assert_eq!(json["adapters_loaded"], 0);
        assert!(json["scheduler"].is_object());
        assert!(json["training"].is_object());
        assert_eq!(json["training"]["queued"], 0);
        assert!(json["training"]["active_job"].is_null());
        assert!(json["checks"].is_array());
        let checks = json["checks"].as_array().unwrap();
        assert!(checks.iter().all(|c| c["pass"] == true));
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
    async fn test_health_scheduler_stats_present() {
        let state = make_test_state();
        let app = routes().with_state(state);

        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
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
