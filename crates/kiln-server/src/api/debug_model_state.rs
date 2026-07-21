use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use kiln_core::config_hashes::ConfigHashes;
use kiln_core::execution_provenance::ExecutionProvenanceV1;
use kiln_core::model_provenance::BaseWeightShardManifest;
use kiln_scheduler::PrefixCacheStats;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::batching_engine::BatchingEngineSnapshot;
use crate::config::{
    BatchingRuntimeConfig, ConfigValueSource, DecodeRuntimeConfig, ModelDefaultsProfile,
    StreamingPrefillRuntimeConfig,
};
use crate::state::{AppState, ModelBackend};

#[derive(Serialize)]
struct DebugDisabledResponse {
    error: &'static str,
    enable_with: &'static str,
}

#[derive(Serialize)]
struct ModelStateResponse {
    model: ModelDebugState,
    adapters: AdapterDebugState,
    config_hashes: ConfigHashes,
    http: HttpDebugState,
    decode_runtime: DecodeRuntimeConfig,
    accelerator_runtime: crate::config::ResolvedAcceleratorRuntimePolicy,
    cuda_graphs: super::config::CudaGraphConfigResponse,
    rocm_synchronization: crate::accelerator_runtime::RocmSynchronizationRuntimeStats,
    rocm_graphs: Option<kiln_model::RocmGraphStats>,
    rocm_graphs_unavailable_reason:
        Option<crate::rocm_graph_observability::RocmGraphUnavailableReason>,
    rocm_graph_telemetry: Option<kiln_model::RocmGraphLiveTelemetry>,
    rocm_graph_telemetry_unavailable_reason:
        Option<crate::rocm_graph_observability::RocmGraphUnavailableReason>,
    kv_autoscaler: crate::kv_autoscaler::KvAutoscalerState,
    streaming_prefill: StreamingPrefillRuntimeConfig,
    training: TrainingDebugState,
    batching_engine: BatchingEngineDebugState,
    thinking: ThinkingDebugState,
    caches: CacheDebugState,
}

#[derive(Serialize)]
struct TrainingDebugState {
    checkpoint_boundary_policy: kiln_train::CheckpointBoundaryPolicy,
}

#[derive(Serialize)]
struct ModelDebugState {
    path: Option<String>,
    served_model_id: String,
    defaults_profile: ModelDefaultsProfile,
    num_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    max_position_embeddings: usize,
    base_weight_shard_manifest: Option<BaseWeightShardManifest>,
    execution_provenance: Option<ExecutionProvenanceV1>,
    backend_runtime: BackendRuntimeDebugState,
}

#[derive(Serialize)]
struct BackendRuntimeDebugState {
    healthy: bool,
    quarantined: bool,
    reason: Option<String>,
    restart_required: bool,
    external_yield_sync: Vec<kiln_model::ExternalYieldSyncStats>,
}

#[derive(Serialize)]
struct AdapterDebugState {
    adapter_dir: String,
    active_adapter: Option<String>,
    loaded_adapter: Option<String>,
    loaded_adapter_revision: Option<String>,
    loaded_adapters: Vec<LoadedAdapterDebugState>,
    available_adapter_count: usize,
    load_errors: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct LoadedAdapterDebugState {
    name: String,
    path: String,
    adapter_model_sha256: Option<String>,
}

#[derive(Serialize)]
struct HttpDebugState {
    /// Resolved request after TOML and environment precedence.
    send_buffer_requested_bytes: Option<usize>,
    /// Raw listener `getsockopt(SO_SNDBUF)` result captured before readiness.
    send_buffer_kernel_readback_bytes: Option<usize>,
    /// Preflight result after normalizing platform-specific accounting.
    send_buffer_effective_bytes: Option<usize>,
}

#[derive(Serialize)]
struct BatchingEngineDebugState {
    backend: &'static str,
    enabled: bool,
    configuration: BatchingRuntimeConfig,
    snapshot: Option<BatchingEngineSnapshotDebug>,
}

#[derive(Serialize)]
struct BatchingEngineSnapshotDebug {
    snapshot_age_ms: u64,
    stream_stall_grace_ms: u64,
    stream_stall_grace_source: ConfigValueSource,
    actor_cycle_idle_ms: u64,
    actor_cycle_idle_source: ConfigValueSource,
    actor_cycle_idle_active: bool,
    actor_cycle_idle_count: u64,
    total_actor_cycle_idle_ms: f64,
    max_actor_cycle_idle_ms: f64,
    accepting: bool,
    queue_depth: usize,
    active_decode: usize,
    active_prefill: usize,
    prefix_cache_enabled: bool,
    resident_prefill_enabled: bool,
    active_resident_prefill: usize,
    max_batch_tokens: usize,
    max_batch_tokens_source: ConfigValueSource,
    max_prefill_tokens_per_cycle: usize,
    max_prefill_tokens_per_cycle_source: ConfigValueSource,
    max_prefill_layers_per_cycle: usize,
    max_prefill_layers_per_cycle_source: ConfigValueSource,
    max_prefill_admission_quantum: usize,
    max_prefill_staging_slots: usize,
    max_active_requests: usize,
    max_prefill_staging_priority_burst: usize,
    max_decode_batch: usize,
    active_staged_requests: usize,
    max_observed_active_requests: usize,
    current_batch_size: usize,
    last_batch_size: usize,
    max_observed_batch_size: usize,
    last_forward_ms: f64,
    max_decode_forward_ms: f64,
    total_decode_forward_ms: f64,
    slow_decode_forward_count: u64,
    last_prefill_ms: f64,
    max_prefill_forward_ms: f64,
    total_prefill_forward_ms: f64,
    slow_prefill_forward_count: u64,
    last_prefill_tokens: usize,
    last_prefill_layers: usize,
    last_admission_ms: f64,
    max_admission_ms: f64,
    total_admission_ms: f64,
    total_admission_calls: u64,
    slow_admission_count: u64,
    total_decode_forwards: u64,
    total_batched_decode_forwards: u64,
    total_decode_rows: u64,
    total_prefill_admission_cycles: u64,
    total_prefill_forwards: u64,
    total_resident_prefill_attempts: u64,
    total_resident_prefill_forwards: u64,
    total_resident_prefill_initial_declines: u64,
    total_resident_prefill_route_failures: u64,
    total_resident_prefill_rows: u64,
    total_resident_prefill_completed_rows: u64,
    last_resident_prefill_batch_size: usize,
    max_resident_prefill_batch_size: usize,
    total_decode_tokens: u64,
    total_prefill_tokens: u64,
    total_prefill_layers: u64,
    total_prefill_layer_yields: u64,
    total_short_prefill_priority_forwards: u64,
    total_prefill_staging_priority_forwards: u64,
    total_prefill_staging_admissions: u64,
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
struct ThinkingDebugState {
    eval_mode: bool,
    default_thinking_enabled: Option<bool>,
    default_thinking_budget_tokens: Option<usize>,
    default_thinking_budget_ms: Option<u64>,
    profile_server_default_thinking_enabled: Option<bool>,
    template_default_thinking_enabled: bool,
    eval_mode_default_thinking_enabled: bool,
}

#[derive(Serialize)]
struct CacheDebugState {
    deterministic_completion_entries: usize,
    deterministic_chat_request_entries: usize,
    deterministic_chat_choices_entries: usize,
    deterministic_batch_entries: usize,
    batched_recurrent_state: kiln_model::BatchedStateCacheStats,
    resident_recurrent_state: kiln_model::GdnRecurrentStateResidencyStats,
    rendered_prompt: PromptCacheDebugState,
    prompt_token: PromptCacheDebugState,
    prefix_cache: PrefixCacheDebugState,
}

#[derive(Serialize)]
struct PromptCacheDebugState {
    hits: u64,
    misses: u64,
    entries: usize,
}

#[derive(Serialize)]
struct PrefixCacheDebugState {
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
    active_leases: usize,
    pending_release_entries: usize,
}

async fn model_state(State(state): State<AppState>) -> Response {
    if !debug_model_state_enabled(&state) {
        return (
            StatusCode::FORBIDDEN,
            Json(DebugDisabledResponse {
                error: "debug endpoint disabled",
                enable_with: "set server.debug_model_state=true or server.eval_mode=true",
            }),
        )
            .into_response();
    }

    if let Some(provenance) = state.execution_provenance.as_deref()
        && let Err(error) = provenance.validate()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "invalid execution provenance",
                "detail": error.to_string(),
            })),
        )
            .into_response();
    }

    Json(build_model_state_response(&state).await).into_response()
}

fn debug_model_state_enabled(state: &AppState) -> bool {
    state.eval_mode || state.debug_model_state
}

async fn build_model_state_response(state: &AppState) -> ModelStateResponse {
    let rocm_synchronization = state.observe_rocm_runtime_health();
    let rocm_graph_observation = crate::rocm_graph_observability::observe_rocm_graphs(state);
    ModelStateResponse {
        model: model_debug_state(state, &rocm_synchronization),
        adapters: adapter_debug_state(state),
        config_hashes: state.config_hashes.clone(),
        http: HttpDebugState {
            send_buffer_requested_bytes: state.http_send_buffer_bytes,
            send_buffer_kernel_readback_bytes: state.http_send_buffer_preflight_actual_bytes,
            send_buffer_effective_bytes: state.http_send_buffer_preflight_effective_bytes,
        },
        decode_runtime: state.decode_runtime_config,
        accelerator_runtime: state.accelerator_runtime_policy,
        cuda_graphs: super::config::cuda_graph_config_response(state),
        rocm_synchronization,
        rocm_graphs: rocm_graph_observation.stats,
        rocm_graphs_unavailable_reason: rocm_graph_observation.stats_unavailable_reason,
        rocm_graph_telemetry: rocm_graph_observation.telemetry,
        rocm_graph_telemetry_unavailable_reason: rocm_graph_observation
            .telemetry_unavailable_reason,
        kv_autoscaler: state.kv_autoscaler,
        streaming_prefill: state.streaming_prefill_runtime_config,
        training: TrainingDebugState {
            checkpoint_boundary_policy: state.training_runtime.checkpoint_boundary_policy(),
        },
        batching_engine: batching_engine_state(state).await,
        thinking: thinking_state(state),
        caches: cache_state(state),
    }
}

fn model_debug_state(
    state: &AppState,
    rocm_synchronization: &crate::accelerator_runtime::RocmSynchronizationRuntimeStats,
) -> ModelDebugState {
    let mut backend_runtime = match state.backend.as_ref() {
        ModelBackend::Mock { .. } => BackendRuntimeDebugState {
            healthy: true,
            quarantined: false,
            reason: None,
            restart_required: false,
            external_yield_sync: Vec::new(),
        },
        ModelBackend::Real { backend_health, .. } => {
            let snapshot = backend_health.snapshot();
            BackendRuntimeDebugState {
                healthy: !snapshot.quarantined,
                quarantined: snapshot.quarantined,
                reason: snapshot.reason,
                restart_required: snapshot.quarantined,
                external_yield_sync: backend_health.external_yield_sync_stats(),
            }
        }
    };
    if let Some(reason) = rocm_synchronization.fail_closed_reason() {
        backend_runtime.healthy = false;
        backend_runtime.quarantined = true;
        backend_runtime.restart_required = true;
        if backend_runtime.reason.is_none() {
            backend_runtime.reason = Some(reason);
        }
    }
    ModelDebugState {
        path: state
            .model_path
            .as_ref()
            .map(|path| path.display().to_string()),
        served_model_id: state.served_model_id.clone(),
        defaults_profile: state.model_defaults_profile,
        num_layers: state.model_config.num_layers,
        num_attention_heads: state.model_config.num_attention_heads,
        num_kv_heads: state.model_config.num_kv_heads,
        max_position_embeddings: state.model_config.max_position_embeddings,
        base_weight_shard_manifest: state.base_weight_shard_manifest.as_deref().cloned(),
        execution_provenance: state.execution_provenance.as_deref().cloned(),
        backend_runtime,
    }
}

fn adapter_debug_state(state: &AppState) -> AdapterDebugState {
    let active_adapter = state.active_adapter_name.read().unwrap().clone();
    let loaded_identity = state.loaded_adapter_identity();
    let loaded_adapter = loaded_identity
        .as_ref()
        .map(|identity| identity.name.clone());
    let loaded_adapter_revision = loaded_identity.map(|identity| identity.content_revision);
    let load_errors: BTreeMap<String, String> = state
        .adapter_load_errors
        .read()
        .unwrap()
        .iter()
        .map(|(name, err)| (name.clone(), err.clone()))
        .collect();
    let mut loaded_names = Vec::new();
    if let Some(name) = active_adapter.as_ref() {
        loaded_names.push(name.clone());
    }
    if let Some(name) = loaded_adapter.as_ref()
        && !loaded_names.iter().any(|existing| existing == name)
    {
        loaded_names.push(name.clone());
    }
    let loaded_adapters = loaded_names
        .into_iter()
        .map(|name| loaded_adapter_state(&state.adapter_dir, name))
        .collect();

    AdapterDebugState {
        adapter_dir: state.adapter_dir.display().to_string(),
        active_adapter,
        loaded_adapter,
        loaded_adapter_revision,
        loaded_adapters,
        available_adapter_count: count_adapter_dirs(&state.adapter_dir),
        load_errors,
    }
}

fn loaded_adapter_state(adapter_dir: &Path, name: String) -> LoadedAdapterDebugState {
    let path = adapter_dir.join(&name);
    let weights_path = path.join("adapter_model.safetensors");
    LoadedAdapterDebugState {
        name,
        path: path.display().to_string(),
        adapter_model_sha256: sha256_file(&weights_path).ok(),
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!(
        "sha256:{}",
        hex_digest(hasher.finalize().as_slice())
    ))
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn count_adapter_dirs(adapter_dir: &Path) -> usize {
    std::fs::read_dir(adapter_dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.path().is_dir())
                .count()
        })
        .unwrap_or(0)
}

async fn batching_engine_state(state: &AppState) -> BatchingEngineDebugState {
    match state.backend.as_ref() {
        ModelBackend::Mock { .. } => BatchingEngineDebugState {
            backend: "mock",
            enabled: false,
            configuration: state.batching_runtime_config,
            snapshot: None,
        },
        ModelBackend::Real {
            batching_engine, ..
        } => BatchingEngineDebugState {
            backend: "model",
            enabled: true,
            configuration: state.batching_runtime_config,
            snapshot: Some(batching_engine.cached_snapshot().into()),
        },
    }
}

fn thinking_state(state: &AppState) -> ThinkingDebugState {
    ThinkingDebugState {
        eval_mode: state.eval_mode,
        default_thinking_enabled: state.default_thinking_enabled,
        default_thinking_budget_tokens: state.default_thinking_budget_tokens,
        default_thinking_budget_ms: state.default_thinking_budget_ms,
        profile_server_default_thinking_enabled: state
            .model_defaults_profile
            .server_default_thinking_enabled,
        template_default_thinking_enabled: state
            .model_defaults_profile
            .template_default_thinking_enabled,
        eval_mode_default_thinking_enabled: state
            .model_defaults_profile
            .eval_mode_default_thinking_enabled,
    }
}

fn cache_state(state: &AppState) -> CacheDebugState {
    let (rendered_prompt_hits, rendered_prompt_misses, rendered_prompt_entries) =
        state.rendered_prompt_cache.lock().unwrap().stats();
    let (prompt_token_hits, prompt_token_misses, prompt_token_entries) =
        state.prompt_token_cache.lock().unwrap().stats();
    let (batched_recurrent_state, resident_recurrent_state) = match state.backend.as_ref() {
        ModelBackend::Mock { .. } => (
            kiln_model::BatchedStateCacheStats::default(),
            kiln_model::GdnRecurrentStateResidencyStats::default(),
        ),
        ModelBackend::Real { runner, .. } => {
            let runner = runner
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (
                runner.batched_state_cache_stats(),
                runner.gdn_recurrent_state_residency_stats(),
            )
        }
    };

    CacheDebugState {
        deterministic_completion_entries: state.completion_cache.lock().unwrap().stats(),
        deterministic_chat_request_entries: state.chat_request_cache.lock().unwrap().stats(),
        deterministic_chat_choices_entries: state.chat_choices_cache.lock().unwrap().stats(),
        deterministic_batch_entries: state.batch_cache.lock().unwrap().stats(),
        batched_recurrent_state,
        resident_recurrent_state,
        rendered_prompt: PromptCacheDebugState {
            hits: rendered_prompt_hits,
            misses: rendered_prompt_misses,
            entries: rendered_prompt_entries,
        },
        prompt_token: PromptCacheDebugState {
            hits: prompt_token_hits,
            misses: prompt_token_misses,
            entries: prompt_token_entries,
        },
        prefix_cache: prefix_cache_state(state),
    }
}

fn prefix_cache_state(state: &AppState) -> PrefixCacheDebugState {
    let stats = match state.backend.as_ref() {
        ModelBackend::Mock { scheduler, .. } => scheduler
            .try_lock()
            .map(|sched| sched.prefix_cache_stats())
            .unwrap_or_default(),
        ModelBackend::Real { prefix_cache, .. } => prefix_cache.lock().unwrap().stats(),
    };
    PrefixCacheDebugState::from(stats)
}

impl From<PrefixCacheStats> for PrefixCacheDebugState {
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
            active_leases: stats.active_leases,
            pending_release_entries: stats.pending_release_entries,
        }
    }
}

impl From<BatchingEngineSnapshot> for BatchingEngineSnapshotDebug {
    fn from(snapshot: BatchingEngineSnapshot) -> Self {
        Self {
            snapshot_age_ms: snapshot.snapshot_age_ms,
            stream_stall_grace_ms: snapshot.stream_stall_grace_ms,
            stream_stall_grace_source: snapshot.stream_stall_grace_source,
            actor_cycle_idle_ms: snapshot.actor_cycle_idle_ms,
            actor_cycle_idle_source: snapshot.actor_cycle_idle_source,
            actor_cycle_idle_active: snapshot.actor_cycle_idle_active,
            actor_cycle_idle_count: snapshot.actor_cycle_idle_count,
            total_actor_cycle_idle_ms: snapshot.total_actor_cycle_idle_ms,
            max_actor_cycle_idle_ms: snapshot.max_actor_cycle_idle_ms,
            accepting: snapshot.accepting,
            queue_depth: snapshot.queue_depth,
            active_decode: snapshot.active_decode,
            active_prefill: snapshot.active_prefill,
            prefix_cache_enabled: snapshot.prefix_cache_enabled,
            resident_prefill_enabled: snapshot.resident_prefill_enabled,
            active_resident_prefill: snapshot.active_resident_prefill,
            max_batch_tokens: snapshot.max_batch_tokens,
            max_batch_tokens_source: snapshot.max_batch_tokens_source,
            max_prefill_tokens_per_cycle: snapshot.max_prefill_tokens_per_cycle,
            max_prefill_tokens_per_cycle_source: snapshot.max_prefill_tokens_per_cycle_source,
            max_prefill_layers_per_cycle: snapshot.max_prefill_layers_per_cycle,
            max_prefill_layers_per_cycle_source: snapshot.max_prefill_layers_per_cycle_source,
            max_prefill_admission_quantum: snapshot.max_prefill_admission_quantum,
            max_prefill_staging_slots: snapshot.max_prefill_staging_slots,
            max_active_requests: snapshot.max_active_requests,
            max_prefill_staging_priority_burst: snapshot.max_prefill_staging_priority_burst,
            max_decode_batch: snapshot.max_decode_batch,
            active_staged_requests: snapshot.active_staged_requests,
            max_observed_active_requests: snapshot.max_observed_active_requests,
            current_batch_size: snapshot.current_batch_size,
            last_batch_size: snapshot.last_batch_size,
            max_observed_batch_size: snapshot.max_observed_batch_size,
            last_forward_ms: snapshot.last_forward_ms,
            max_decode_forward_ms: snapshot.max_decode_forward_ms,
            total_decode_forward_ms: snapshot.total_decode_forward_ms,
            slow_decode_forward_count: snapshot.slow_decode_forward_count,
            last_prefill_ms: snapshot.last_prefill_ms,
            max_prefill_forward_ms: snapshot.max_prefill_forward_ms,
            total_prefill_forward_ms: snapshot.total_prefill_forward_ms,
            slow_prefill_forward_count: snapshot.slow_prefill_forward_count,
            last_prefill_tokens: snapshot.last_prefill_tokens,
            last_prefill_layers: snapshot.last_prefill_layers,
            last_admission_ms: snapshot.last_admission_ms,
            max_admission_ms: snapshot.max_admission_ms,
            total_admission_ms: snapshot.total_admission_ms,
            total_admission_calls: snapshot.total_admission_calls,
            slow_admission_count: snapshot.slow_admission_count,
            total_decode_forwards: snapshot.total_decode_forwards,
            total_batched_decode_forwards: snapshot.total_batched_decode_forwards,
            total_decode_rows: snapshot.total_decode_rows,
            total_prefill_admission_cycles: snapshot.total_prefill_admission_cycles,
            total_prefill_forwards: snapshot.total_prefill_forwards,
            total_resident_prefill_attempts: snapshot.total_resident_prefill_attempts,
            total_resident_prefill_forwards: snapshot.total_resident_prefill_forwards,
            total_resident_prefill_initial_declines: snapshot
                .total_resident_prefill_initial_declines,
            total_resident_prefill_route_failures: snapshot.total_resident_prefill_route_failures,
            total_resident_prefill_rows: snapshot.total_resident_prefill_rows,
            total_resident_prefill_completed_rows: snapshot.total_resident_prefill_completed_rows,
            last_resident_prefill_batch_size: snapshot.last_resident_prefill_batch_size,
            max_resident_prefill_batch_size: snapshot.max_resident_prefill_batch_size,
            total_decode_tokens: snapshot.total_decode_tokens,
            total_prefill_tokens: snapshot.total_prefill_tokens,
            total_prefill_layers: snapshot.total_prefill_layers,
            total_prefill_layer_yields: snapshot.total_prefill_layer_yields,
            total_short_prefill_priority_forwards: snapshot.total_short_prefill_priority_forwards,
            total_prefill_staging_priority_forwards: snapshot
                .total_prefill_staging_priority_forwards,
            total_prefill_staging_admissions: snapshot.total_prefill_staging_admissions,
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
    Router::new().route("/v1/debug/model-state", get(model_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use kiln_core::config::ModelConfig;
    use kiln_model::engine::MockEngine;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use std::path::PathBuf;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn make_test_state(adapter_dir: PathBuf) -> AppState {
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
        let mut state = AppState::new_mock(
            config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            300,
            "Qwen3.5-4B".to_string(),
        );
        state.adapter_dir = adapter_dir;
        state.model_path = Some(PathBuf::from("/models/Qwen3.5-4B"));
        state
    }

    #[tokio::test]
    async fn debug_model_state_requires_typed_access_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_test_state(tmp.path().to_path_buf());
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/debug/model-state")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn debug_model_state_accepts_typed_debug_policy_without_eval_mode() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = make_test_state(tmp.path().to_path_buf());
        state.debug_model_state = true;
        let app = routes().with_state(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/debug/model-state")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn debug_model_state_reports_active_adapter_and_config_without_prompts() {
        let tmp = tempfile::tempdir().unwrap();
        let adapter_dir = tmp.path().join("eval-adapter");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        std::fs::write(
            adapter_dir.join("adapter_model.safetensors"),
            b"adapter bytes",
        )
        .unwrap();

        let mut state = make_test_state(tmp.path().to_path_buf());
        state.eval_mode = true;
        state.http_send_buffer_bytes = Some(4096);
        state.http_send_buffer_preflight_actual_bytes = Some(8192);
        state.http_send_buffer_preflight_effective_bytes = Some(4096);
        state.default_thinking_enabled = Some(false);
        let checkpoint_boundary_policy = kiln_train::CheckpointBoundaryPolicy::from_parts(
            kiln_train::CheckpointBoundaryRecomputeMode::Enabled,
            16_384,
            Some(11),
            4 * 1024 * 1024 * 1024,
        )
        .unwrap();
        state.training_runtime = state
            .training_runtime
            .with_checkpoint_boundary_policy(checkpoint_boundary_policy);
        let shard_manifest = BaseWeightShardManifest::new(vec![
            kiln_core::model_provenance::BaseWeightShardIdentity::from_digest(
                "model.safetensors",
                456,
                [0xcd; 32],
            )
            .unwrap(),
        ])
        .unwrap();
        state.base_weight_shard_manifest = Some(Arc::new(shard_manifest.clone()));
        let execution_provenance = crate::execution_provenance::test_execution_provenance();
        state.execution_provenance = Some(Arc::new(execution_provenance.clone()));
        *state.active_adapter_name.write().unwrap() = Some("eval-adapter".to_string());
        *state.loaded_adapter.write().unwrap() = Some(crate::state::LoadedAdapterIdentity {
            name: "eval-adapter".to_string(),
            content_revision: "a".repeat(64),
        });
        state.adapter_load_errors.write().unwrap().insert(
            "bad-adapter".to_string(),
            "missing adapter_config.json".to_string(),
        );
        {
            let mut recent = state.recent_requests.lock().unwrap();
            recent.record(crate::recent_requests::RequestRecord {
                user_agent: None,
                id: "secret-id".to_string(),
                prompt_preview: "secret prompt".to_string(),
                prompt_full: Some("full secret prompt".to_string()),
                completion_preview: "secret completion".to_string(),
                ..Default::default()
            });
        }
        let expected_streaming_prefill =
            serde_json::to_value(state.streaming_prefill_runtime_config).unwrap();
        let expected_checkpoint_boundary_policy =
            serde_json::to_value(checkpoint_boundary_policy).unwrap();

        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/debug/model-state")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["streaming_prefill"], expected_streaming_prefill);
        assert_eq!(
            json["training"]["checkpoint_boundary_policy"],
            expected_checkpoint_boundary_policy
        );
        assert_eq!(json["model"]["path"], "/models/Qwen3.5-4B");
        assert_eq!(json["model"]["served_model_id"], "Qwen3.5-4B");
        assert_eq!(json["model"]["defaults_profile"]["name"], "Qwen3.5-4B");
        assert_eq!(
            json["model"]["base_weight_shard_manifest"]["aggregate_sha256"],
            shard_manifest.aggregate_sha256
        );
        assert_eq!(
            json["model"]["base_weight_shard_manifest"]["shards"][0]["filename"],
            "model.safetensors"
        );
        assert_eq!(
            json["model"]["base_weight_shard_manifest"]["shards"][0]["size_bytes"],
            456
        );
        assert_eq!(
            json["model"]["execution_provenance"]["provenance_sha256"],
            execution_provenance.provenance_sha256
        );
        assert_eq!(
            json["model"]["execution_provenance"]["backend"]["device"],
            "cpu"
        );
        assert_eq!(json["adapters"]["active_adapter"], "eval-adapter");
        assert_eq!(json["adapters"]["loaded_adapter"], "eval-adapter");
        assert_eq!(json["adapters"]["loaded_adapter_revision"], "a".repeat(64));
        assert_eq!(json["adapters"]["available_adapter_count"], 1);
        assert_eq!(
            json["adapters"]["load_errors"]["bad-adapter"],
            "missing adapter_config.json"
        );
        assert_eq!(
            json["adapters"]["loaded_adapters"][0]["name"],
            "eval-adapter"
        );
        assert!(
            json["adapters"]["loaded_adapters"][0]["adapter_model_sha256"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(
            json["config_hashes"]["model_config_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert_eq!(json["thinking"]["eval_mode"], true);
        assert_eq!(json["thinking"]["default_thinking_enabled"], false);
        assert_eq!(
            json["thinking"]["eval_mode_default_thinking_enabled"],
            false
        );
        assert_eq!(json["batching_engine"]["backend"], "mock");
        assert_eq!(json["batching_engine"]["enabled"], false);
        assert_eq!(
            json["batching_engine"]["configuration"]["prefix_aware_admission"]["enabled"],
            true
        );
        assert_eq!(
            json["batching_engine"]["configuration"]["actor_cycle_idle"]["milliseconds"],
            0
        );
        assert_eq!(
            json["batching_engine"]["configuration"]["actor_cycle_idle"]["source"],
            "default"
        );
        assert_eq!(
            json["batching_engine"]["configuration"]["actor_cycle_idle"]["command_poll_milliseconds"],
            5
        );
        assert_eq!(json["kv_autoscaler"]["requested"], true);
        assert_eq!(json["kv_autoscaler"]["requested_source"], "default");
        assert!(json["kv_autoscaler"]["force_blocks"].is_null());
        assert_eq!(json["kv_autoscaler"]["force_blocks_source"], "default");
        assert_eq!(json["kv_autoscaler"]["state"], "unavailable");
        assert_eq!(json["kv_autoscaler"]["reason"], "mock_backend");
        assert_eq!(json["http"]["send_buffer_requested_bytes"], 4096);
        assert_eq!(json["http"]["send_buffer_kernel_readback_bytes"], 8192);
        assert_eq!(json["http"]["send_buffer_effective_bytes"], 4096);
        assert_eq!(json["decode_runtime"]["deterministic"]["enabled"], false);
        assert_eq!(json["decode_runtime"]["max_decode_batch"]["effective"], 8);
        assert_eq!(
            json["accelerator_runtime"]["schema_id"],
            "kiln.accelerator-runtime-policy.v15"
        );
        assert_eq!(
            json["accelerator_runtime"]["vulkan_kernel_policy_schema_id"],
            "kiln.vulkan-kernel-policy.v3"
        );
        assert_eq!(
            json["accelerator_runtime"]["vulkan_device_policy_schema_id"],
            "kiln.vulkan-device-policy.v1"
        );
        assert_eq!(
            json["accelerator_runtime"]["kt_api_mode"]["effective"],
            "auto"
        );
        assert_eq!(
            json["accelerator_runtime"]["full_attention_score_budget_mib"]["effective"],
            kiln_model::DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB
        );
        assert_eq!(
            json["accelerator_runtime"]["cuda_kernel_profile"]["effective"],
            "native_default"
        );
        assert_eq!(
            json["accelerator_runtime"]["cuda_marlin_profile"]["effective"],
            "disabled"
        );
        assert_eq!(
            json["accelerator_runtime"]["cuda_flash_backward_mode"]["effective"],
            "fast"
        );
        assert_eq!(
            json["accelerator_runtime"]["metal_kernel_profile"]["effective"],
            "native_default"
        );
        assert!(json["accelerator_runtime"]["vulkan_device_index"]["effective"].is_null());
        assert_eq!(
            json["accelerator_runtime"]["vulkan_validation"]["effective"],
            false
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_graph_cache_max_bytes"]["effective"],
            crate::config::DEFAULT_ROCM_GRAPH_CACHE_MAX_BYTES
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_strided_batched_matmul_mode"]["effective"],
            "auto"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_bf16_matmul_output_mode"]["effective"],
            "auto"
        );
        assert_eq!(
            json["accelerator_runtime"]["rocm_kernel_profile"]["effective"],
            "qualified"
        );
        assert_eq!(json["cuda_graphs"]["requested"], true);
        assert_eq!(
            json["cuda_graphs"]["capture_allowed_by_serving_profile"],
            false
        );
        assert_eq!(json["cuda_graphs"]["effective_policy_enabled"], false);
        assert_eq!(json["cuda_graphs"]["max_cached_graphs"], 8);
        assert_eq!(json["cuda_graphs"]["stable_paged_metadata"], true);
        assert_eq!(json["cuda_graphs"]["batched_capture_available"], false);
        assert_eq!(json["cuda_graphs"]["restart_required_to_change"], true);
        assert_eq!(json["rocm_synchronization"]["active"], false);
        assert_eq!(json["rocm_synchronization"]["cleanup_quarantined"], false);
        assert!(json["rocm_graphs"].is_null());
        assert_eq!(
            json["streaming_prefill"]["dispatch"]["configured_mode"],
            "auto"
        );
        assert_eq!(
            json["streaming_prefill"]["dispatch"]["effective"]["policy"],
            "never"
        );
        assert_eq!(json["streaming_prefill"]["tile_tokens"]["effective"], 8192);
        assert!(json["caches"]["rendered_prompt"].is_object());
        assert!(json["caches"]["prefix_cache"].is_object());
        assert!(json["caches"]["batched_recurrent_state"].is_object());
        assert_eq!(
            json["caches"]["batched_recurrent_state"]["entry_present"],
            false
        );
        assert_eq!(json["caches"]["resident_recurrent_state"]["entry_count"], 0);
        assert_eq!(
            json["caches"]["resident_recurrent_state"]["buffer_bytes"],
            0
        );
        assert_eq!(
            json["caches"]["resident_recurrent_state"]["allocation_bytes"],
            0
        );
        assert_eq!(
            json["caches"]["batched_recurrent_state"]["active_leases"],
            0
        );
        assert_eq!(
            json["caches"]["batched_recurrent_state"]["park_replacement_eviction_count"],
            0
        );
        assert_eq!(json["caches"]["prefix_cache"]["active_leases"], 0);
        assert_eq!(json["caches"]["prefix_cache"]["pending_release_entries"], 0);

        let serialized = serde_json::to_string(&json).unwrap();
        assert!(!serialized.contains("secret prompt"));
        assert!(!serialized.contains("full secret prompt"));
        assert!(!serialized.contains("secret completion"));
    }

    #[tokio::test]
    async fn debug_model_state_rejects_tampered_execution_provenance() {
        let tmp = tempfile::tempdir().unwrap();
        let mut state = make_test_state(tmp.path().to_path_buf());
        state.eval_mode = true;
        let mut provenance = crate::execution_provenance::test_execution_provenance();
        provenance.backend.device = "tampered".into();
        state.execution_provenance = Some(Arc::new(provenance));

        let response = routes()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .uri("/v1/debug/model-state")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"], "invalid execution provenance");
    }

    #[test]
    fn batching_engine_debug_preserves_stream_stall_policy() {
        let debug = BatchingEngineSnapshotDebug::from(BatchingEngineSnapshot {
            stream_stall_grace_ms: 500,
            stream_stall_grace_source: ConfigValueSource::ConfigFile,
            actor_cycle_idle_ms: 50,
            actor_cycle_idle_source: ConfigValueSource::Environment,
            actor_cycle_idle_active: false,
            actor_cycle_idle_count: 6,
            total_actor_cycle_idle_ms: 302.0,
            max_actor_cycle_idle_ms: 52.0,
            active_prefill: 3,
            prefix_cache_enabled: true,
            resident_prefill_enabled: true,
            active_resident_prefill: 2,
            max_batch_tokens: 128,
            max_batch_tokens_source: ConfigValueSource::Environment,
            max_prefill_tokens_per_cycle: 32,
            max_prefill_tokens_per_cycle_source: ConfigValueSource::ConfigFile,
            max_prefill_layers_per_cycle: 3,
            max_prefill_layers_per_cycle_source: ConfigValueSource::Environment,
            max_prefill_staging_slots: 4,
            max_active_requests: 16,
            max_prefill_staging_priority_burst: 4,
            max_decode_batch: 12,
            active_staged_requests: 2,
            max_observed_active_requests: 15,
            last_prefill_tokens: 124,
            last_prefill_layers: 3,
            max_decode_forward_ms: 115.0,
            total_decode_forward_ms: 460.0,
            slow_decode_forward_count: 2,
            max_prefill_forward_ms: 575.0,
            total_prefill_forward_ms: 1_725.0,
            slow_prefill_forward_count: 3,
            max_admission_ms: 150.0,
            total_admission_ms: 225.0,
            total_admission_calls: 4,
            slow_admission_count: 1,
            total_prefill_forwards: 11,
            total_resident_prefill_attempts: 10,
            total_resident_prefill_forwards: 9,
            total_resident_prefill_initial_declines: 1,
            total_resident_prefill_route_failures: 0,
            total_resident_prefill_rows: 53,
            total_resident_prefill_completed_rows: 6,
            last_resident_prefill_batch_size: 5,
            max_resident_prefill_batch_size: 7,
            total_prefill_layers: 33,
            total_prefill_layer_yields: 22,
            total_short_prefill_priority_forwards: 6,
            total_prefill_staging_priority_forwards: 3,
            total_prefill_staging_admissions: 5,
            response_delivery_in_flight: 4,
            response_delivery_backpressured: 2,
            response_delivery_pending_terminal: 1,
            ..BatchingEngineSnapshot::default()
        });
        let json = serde_json::to_value(debug).unwrap();

        assert_eq!(json["stream_stall_grace_ms"], 500);
        assert_eq!(json["stream_stall_grace_source"], "config_file");
        assert_eq!(json["actor_cycle_idle_ms"], 50);
        assert_eq!(json["actor_cycle_idle_source"], "environment");
        assert_eq!(json["actor_cycle_idle_active"], false);
        assert_eq!(json["actor_cycle_idle_count"], 6);
        assert_eq!(json["total_actor_cycle_idle_ms"], 302.0);
        assert_eq!(json["max_actor_cycle_idle_ms"], 52.0);
        assert_eq!(json["active_prefill"], 3);
        assert_eq!(json["prefix_cache_enabled"], true);
        assert_eq!(json["resident_prefill_enabled"], true);
        assert_eq!(json["active_resident_prefill"], 2);
        assert_eq!(json["max_batch_tokens"], 128);
        assert_eq!(json["max_batch_tokens_source"], "environment");
        assert_eq!(json["max_prefill_tokens_per_cycle"], 32);
        assert_eq!(json["max_prefill_tokens_per_cycle_source"], "config_file");
        assert_eq!(json["max_prefill_layers_per_cycle"], 3);
        assert_eq!(json["max_prefill_layers_per_cycle_source"], "environment");
        assert_eq!(json["max_prefill_staging_slots"], 4);
        assert_eq!(json["max_active_requests"], 16);
        assert_eq!(json["max_prefill_staging_priority_burst"], 4);
        assert_eq!(json["max_decode_batch"], 12);
        assert_eq!(json["active_staged_requests"], 2);
        assert_eq!(json["max_observed_active_requests"], 15);
        assert_eq!(json["last_prefill_tokens"], 124);
        assert_eq!(json["last_prefill_layers"], 3);
        assert_eq!(json["max_decode_forward_ms"], 115.0);
        assert_eq!(json["total_decode_forward_ms"], 460.0);
        assert_eq!(json["slow_decode_forward_count"], 2);
        assert_eq!(json["max_prefill_forward_ms"], 575.0);
        assert_eq!(json["total_prefill_forward_ms"], 1_725.0);
        assert_eq!(json["slow_prefill_forward_count"], 3);
        assert_eq!(json["max_admission_ms"], 150.0);
        assert_eq!(json["total_admission_ms"], 225.0);
        assert_eq!(json["total_admission_calls"], 4);
        assert_eq!(json["slow_admission_count"], 1);
        assert_eq!(json["total_prefill_forwards"], 11);
        assert_eq!(json["total_resident_prefill_attempts"], 10);
        assert_eq!(json["total_resident_prefill_forwards"], 9);
        assert_eq!(json["total_resident_prefill_initial_declines"], 1);
        assert_eq!(json["total_resident_prefill_route_failures"], 0);
        assert_eq!(json["total_resident_prefill_rows"], 53);
        assert_eq!(json["total_resident_prefill_completed_rows"], 6);
        assert_eq!(json["last_resident_prefill_batch_size"], 5);
        assert_eq!(json["max_resident_prefill_batch_size"], 7);
        assert_eq!(json["total_prefill_layers"], 33);
        assert_eq!(json["total_prefill_layer_yields"], 22);
        assert_eq!(json["total_short_prefill_priority_forwards"], 6);
        assert_eq!(json["total_prefill_staging_priority_forwards"], 3);
        assert_eq!(json["total_prefill_staging_admissions"], 5);
        assert_eq!(json["response_delivery_in_flight"], 4);
        assert_eq!(json["response_delivery_backpressured"], 2);
        assert_eq!(json["response_delivery_pending_terminal"], 1);
    }
}
