use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::config::ServingProfileDiagnostics;
use crate::state::{AppState, ModelBackend};

#[derive(Serialize)]
struct ConfigResponse {
    serving_profile: ServingProfileDiagnostics,
    vram: VramConfig,
    kv_cache: KvCacheConfig,
    training: TrainingConfig,
    memory_budget: MemoryBudgetConfig,
    generation: GenerationConfig,
}

#[derive(Serialize)]
struct VramConfig {
    detected_gb: f64,
    source: String,
}

#[derive(Serialize)]
struct KvCacheConfig {
    num_blocks: usize,
    num_blocks_source: &'static str,
    fp8_enabled: bool,
}

#[derive(Serialize)]
struct TrainingConfig {
    checkpoint_segments: usize,
    checkpoint_segments_source: &'static str,
    checkpointing_enabled: bool,
}

#[derive(Serialize)]
struct MemoryBudgetConfig {
    total_vram_gb: f64,
    model_gb: f64,
    kv_cache_gb: f64,
    training_budget_gb: f64,
    inference_memory_fraction: f64,
}

#[derive(Serialize)]
struct GenerationConfig {
    default_thinking_enabled: Option<bool>,
    default_thinking_budget_tokens: Option<usize>,
    default_thinking_budget_ms: Option<u64>,
    fold_reasoning_into_content: bool,
}

async fn get_config(State(state): State<AppState>) -> Json<ConfigResponse> {
    let vram = &state.vram_info;

    // Get actual num_blocks from the running backend
    let (num_blocks, num_blocks_source) = match state.backend.as_ref() {
        ModelBackend::Real { block_manager, .. } => {
            let bm = block_manager.lock().unwrap();
            let source = if std::env::var("KILN_NUM_BLOCKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .is_some()
            {
                "KILN_NUM_BLOCKS"
            } else {
                "auto"
            };
            (bm.num_blocks(), source)
        }
        ModelBackend::Mock { scheduler, .. } => {
            let sched = scheduler.try_lock();
            match sched {
                Ok(s) => (s.block_manager().num_blocks(), "mock"),
                Err(_) => (0, "unknown"),
            }
        }
    };

    // Determine checkpoint segments
    let ckpt = kiln_train::CheckpointConfig::from_env(state.model_config.num_layers);
    let segments_source = if std::env::var("KILN_GRAD_CHECKPOINT_SEGMENTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some()
    {
        "KILN_GRAD_CHECKPOINT_SEGMENTS"
    } else if ckpt.auto_configured {
        "auto"
    } else {
        "default"
    };

    let b = &state.memory_budget;

    Json(ConfigResponse {
        serving_profile: state.serving_profile.diagnostics(),
        vram: VramConfig {
            detected_gb: vram.total_bytes as f64 / 1e9,
            source: vram.source.to_string(),
        },
        kv_cache: KvCacheConfig {
            num_blocks,
            num_blocks_source,
            fp8_enabled: match state.backend.as_ref() {
                ModelBackend::Real { paged_cache, .. } => paged_cache.is_fp8(),
                ModelBackend::Mock { .. } => false,
            },
        },
        training: TrainingConfig {
            checkpoint_segments: ckpt.num_segments,
            checkpoint_segments_source: segments_source,
            checkpointing_enabled: ckpt.enabled,
        },
        memory_budget: MemoryBudgetConfig {
            total_vram_gb: b.total_vram_bytes as f64 / 1e9,
            model_gb: b.model_memory_bytes as f64 / 1e9,
            kv_cache_gb: b.kv_cache_bytes as f64 / 1e9,
            training_budget_gb: b.training_budget_bytes as f64 / 1e9,
            inference_memory_fraction: b.inference_memory_fraction,
        },
        generation: GenerationConfig {
            default_thinking_enabled: state.default_thinking_enabled,
            default_thinking_budget_tokens: state.default_thinking_budget_tokens,
            default_thinking_budget_ms: state.default_thinking_budget_ms,
            fold_reasoning_into_content: state.fold_reasoning_into_content,
        },
    })
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/config", get(get_config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use std::sync::Arc;
    use tower::ServiceExt;

    fn make_test_state() -> AppState {
        let model_config = ModelConfig::qwen3_5_4b();
        let scheduler = Scheduler::new(SchedulerConfig::default(), 256);
        AppState::new_mock(
            model_config.clone(),
            scheduler,
            Arc::new(MockEngine::new(model_config)),
            crate::api::test_tokenizer(),
            300,
            "Qwen3.5-4B".to_string(),
        )
    }

    #[tokio::test]
    async fn config_reports_profile_provenance_and_every_effective_policy() {
        let mut state = make_test_state();
        state.serving_profile = crate::config::ServingProfileSetting::new(
            crate::config::ServingProfile::Experimental,
            crate::config::ConfigValueSource::Environment,
        );
        let app = routes().with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["serving_profile"]["profile"], "experimental");
        assert_eq!(json["serving_profile"]["source"], "environment");
        assert_eq!(json["serving_profile"]["immutable_after_startup"], true);
        assert_eq!(json["serving_profile"]["request_overrides_allowed"], false);
        let policy = &json["serving_profile"]["effective_policy"];
        for field in [
            "inference_admission",
            "training_gpu_ownership",
            "adapter_weight_transitions",
            "dynamic_kv_resize",
            "allocator_reclaim",
            "live_graph_capture",
        ] {
            assert_eq!(policy[field], true, "unexpected {field}");
        }
        assert_eq!(policy["exclusive_gpu_behavior"], "writer_priority");
    }
}
