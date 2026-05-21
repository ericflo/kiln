use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use kiln_core::config_hashes::ConfigHashes;
use kiln_scheduler::PrefixCacheStats;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::batching_engine::BatchingEngineSnapshot;
use crate::config::ModelDefaultsProfile;
use crate::state::{AppState, ModelBackend};

const DEBUG_ENDPOINT_ENV: &str = "KILN_DEBUG_ENDPOINTS";

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
    env_flags: BTreeMap<&'static str, EnvFlagState>,
    batching_engine: BatchingEngineDebugState,
    thinking: ThinkingDebugState,
    caches: CacheDebugState,
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
}

#[derive(Serialize)]
struct AdapterDebugState {
    adapter_dir: String,
    active_adapter: Option<String>,
    loaded_adapter: Option<String>,
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
struct EnvFlagState {
    present: bool,
    value: Option<String>,
}

#[derive(Serialize)]
struct BatchingEngineDebugState {
    backend: &'static str,
    enabled: bool,
    snapshot: Option<BatchingEngineSnapshotDebug>,
    decode_batcher: Option<DecodeBatcherDebug>,
}

#[derive(Serialize)]
struct BatchingEngineSnapshotDebug {
    accepting: bool,
    queue_depth: usize,
    active_decode: usize,
    current_batch_size: usize,
    last_batch_size: usize,
    last_forward_ms: f64,
    last_prefill_ms: f64,
    total_decode_tokens: u64,
    total_prefill_tokens: u64,
    total_errors: u64,
    adapter_groups_waiting: usize,
    prefix_deferred_waiting: usize,
    prefix_admission_deferrals: u64,
}

#[derive(Serialize)]
struct DecodeBatcherDebug {
    submitted_jobs: usize,
    executed_batches: usize,
    executed_rows: usize,
    max_observed_batch: usize,
    runner_busy_jobs: usize,
    failed_jobs: usize,
}

#[derive(Serialize)]
struct ThinkingDebugState {
    eval_mode: bool,
    default_thinking_enabled: Option<bool>,
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
}

async fn model_state(State(state): State<AppState>) -> Response {
    if !debug_model_state_enabled(&state) {
        return (
            StatusCode::FORBIDDEN,
            Json(DebugDisabledResponse {
                error: "debug endpoint disabled",
                enable_with: "set KILN_DEBUG_ENDPOINTS=1 or run with eval_mode=true",
            }),
        )
            .into_response();
    }

    Json(build_model_state_response(&state).await).into_response()
}

fn debug_model_state_enabled(state: &AppState) -> bool {
    state.eval_mode
        || std::env::var(DEBUG_ENDPOINT_ENV)
            .ok()
            .as_deref()
            .is_some_and(is_truthy)
}

fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

async fn build_model_state_response(state: &AppState) -> ModelStateResponse {
    ModelStateResponse {
        model: model_debug_state(state),
        adapters: adapter_debug_state(state),
        config_hashes: state.config_hashes.clone(),
        env_flags: selected_env_flags(),
        batching_engine: batching_engine_state(state).await,
        thinking: thinking_state(state),
        caches: cache_state(state),
    }
}

fn model_debug_state(state: &AppState) -> ModelDebugState {
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
    }
}

fn adapter_debug_state(state: &AppState) -> AdapterDebugState {
    let active_adapter = state.active_adapter_name.read().unwrap().clone();
    let loaded_adapter = state.loaded_adapter_name.read().unwrap().clone();
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
    Ok(format!("sha256:{}", hex_digest(hasher.finalize().as_slice())))
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

fn selected_env_flags() -> BTreeMap<&'static str, EnvFlagState> {
    [
        DEBUG_ENDPOINT_ENV,
        "KILN_MODEL_PATH",
        "KILN_MODEL_ID",
        "KILN_TOKENIZER_PATH",
        "KILN_ADAPTER_DIR",
        "KILN_SERVED_MODEL_ID",
        "KILN_EVAL_MODE",
        "KILN_DEFAULT_THINKING_ENABLED",
        "KILN_DEFAULT_NO_THINK",
        "KILN_BATCHING_ENGINE",
        "KILN_CUDA_GRAPHS",
        "KILN_KV_CACHE_FP8",
        "KILN_PREFIX_CACHE_ENABLED",
        "KILN_NUM_BLOCKS",
        "KILN_INFERENCE_MEMORY_FRACTION",
    ]
    .into_iter()
    .map(|name| {
        let value = std::env::var(name).ok();
        (
            name,
            EnvFlagState {
                present: value.is_some(),
                value,
            },
        )
    })
    .collect()
}

async fn batching_engine_state(state: &AppState) -> BatchingEngineDebugState {
    match state.backend.as_ref() {
        ModelBackend::Mock { .. } => BatchingEngineDebugState {
            backend: "mock",
            enabled: false,
            snapshot: None,
            decode_batcher: None,
        },
        ModelBackend::Real {
            batching_engine,
            decode_batcher,
            ..
        } => {
            let snapshot = match batching_engine {
                Some(engine) => engine.snapshot().await.ok().map(Into::into),
                None => None,
            };
            BatchingEngineDebugState {
                backend: "model",
                enabled: batching_engine.is_some(),
                snapshot,
                decode_batcher: decode_batcher.as_ref().map(|batcher| {
                    let stats = batcher.stats();
                    DecodeBatcherDebug {
                        submitted_jobs: stats.submitted_jobs,
                        executed_batches: stats.executed_batches,
                        executed_rows: stats.executed_rows,
                        max_observed_batch: stats.max_observed_batch,
                        runner_busy_jobs: stats.runner_busy_jobs,
                        failed_jobs: stats.failed_jobs,
                    }
                }),
            }
        }
    }
}

fn thinking_state(state: &AppState) -> ThinkingDebugState {
    ThinkingDebugState {
        eval_mode: state.eval_mode,
        default_thinking_enabled: state.default_thinking_enabled,
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

    CacheDebugState {
        deterministic_completion_entries: state.completion_cache.lock().unwrap().stats(),
        deterministic_chat_request_entries: state.chat_request_cache.lock().unwrap().stats(),
        deterministic_chat_choices_entries: state.chat_choices_cache.lock().unwrap().stats(),
        deterministic_batch_entries: state.batch_cache.lock().unwrap().stats(),
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
        }
    }
}

impl From<BatchingEngineSnapshot> for BatchingEngineSnapshotDebug {
    fn from(snapshot: BatchingEngineSnapshot) -> Self {
        Self {
            accepting: snapshot.accepting,
            queue_depth: snapshot.queue_depth,
            active_decode: snapshot.active_decode,
            current_batch_size: snapshot.current_batch_size,
            last_batch_size: snapshot.last_batch_size,
            last_forward_ms: snapshot.last_forward_ms,
            last_prefill_ms: snapshot.last_prefill_ms,
            total_decode_tokens: snapshot.total_decode_tokens,
            total_prefill_tokens: snapshot.total_prefill_tokens,
            total_errors: snapshot.total_errors,
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
    async fn debug_model_state_requires_eval_or_debug_env() {
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
    async fn debug_model_state_reports_active_adapter_and_config_without_prompts() {
        let tmp = tempfile::tempdir().unwrap();
        let adapter_dir = tmp.path().join("eval-adapter");
        std::fs::create_dir_all(&adapter_dir).unwrap();
        std::fs::write(adapter_dir.join("adapter_model.safetensors"), b"adapter bytes").unwrap();

        let mut state = make_test_state(tmp.path().to_path_buf());
        state.eval_mode = true;
        state.default_thinking_enabled = Some(false);
        *state.active_adapter_name.write().unwrap() = Some("eval-adapter".to_string());
        *state.loaded_adapter_name.write().unwrap() = Some("eval-adapter".to_string());
        state
            .adapter_load_errors
            .write()
            .unwrap()
            .insert("bad-adapter".to_string(), "missing adapter_config.json".to_string());
        {
            let mut recent = state.recent_requests.lock().unwrap();
            recent.record(crate::recent_requests::RequestRecord {
                id: "secret-id".to_string(),
                prompt_preview: "secret prompt".to_string(),
                prompt_full: Some("full secret prompt".to_string()),
                completion_preview: "secret completion".to_string(),
                ..Default::default()
            });
        }

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

        assert_eq!(json["model"]["path"], "/models/Qwen3.5-4B");
        assert_eq!(json["model"]["served_model_id"], "Qwen3.5-4B");
        assert_eq!(json["model"]["defaults_profile"]["name"], "Qwen3.5-4B");
        assert_eq!(json["adapters"]["active_adapter"], "eval-adapter");
        assert_eq!(json["adapters"]["loaded_adapter"], "eval-adapter");
        assert_eq!(json["adapters"]["available_adapter_count"], 1);
        assert_eq!(
            json["adapters"]["load_errors"]["bad-adapter"],
            "missing adapter_config.json"
        );
        assert_eq!(json["adapters"]["loaded_adapters"][0]["name"], "eval-adapter");
        assert!(json["adapters"]["loaded_adapters"][0]["adapter_model_sha256"]
            .as_str()
            .unwrap()
            .starts_with("sha256:"));
        assert!(json["config_hashes"]["model_config_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:"));
        assert_eq!(json["thinking"]["eval_mode"], true);
        assert_eq!(json["thinking"]["default_thinking_enabled"], false);
        assert_eq!(json["thinking"]["eval_mode_default_thinking_enabled"], false);
        assert_eq!(json["batching_engine"]["backend"], "mock");
        assert_eq!(json["batching_engine"]["enabled"], false);
        assert!(json["caches"]["rendered_prompt"].is_object());
        assert!(json["caches"]["prefix_cache"].is_object());

        let serialized = serde_json::to_string(&json).unwrap();
        assert!(!serialized.contains("secret prompt"));
        assert!(!serialized.contains("full secret prompt"));
        assert!(!serialized.contains("secret completion"));
    }
}
