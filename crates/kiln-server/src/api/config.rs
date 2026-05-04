use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::state::{AppState, ModelBackend};

#[derive(Serialize)]
struct ConfigResponse {
    vram: VramConfig,
    kv_cache: KvCacheConfig,
    training: TrainingConfig,
    memory_budget: MemoryBudgetConfig,
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
        vram: VramConfig {
            detected_gb: vram.total_bytes as f64 / 1e9,
            source: vram.source.to_string(),
        },
        kv_cache: KvCacheConfig {
            num_blocks,
            num_blocks_source,
            fp8_enabled: match state.backend.as_ref() {
                ModelBackend::Real { paged_cache, .. } => paged_cache.lock().unwrap().is_fp8(),
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
    })
}

pub fn routes() -> Router<AppState> {
    Router::new().route("/v1/config", get(get_config))
}
