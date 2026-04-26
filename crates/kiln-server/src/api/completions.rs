use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use std::path::{Path, PathBuf};
use kiln_core::request::Request;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::ChatMessage;
use kiln_model::adapter_merge::{merge_concat, PeftLora};
use kiln_model::lora_loader::LoraWeights;
use kiln_model::{GenerationOutput, ModelRunner, PagedPrefixReuse, SpeculativeConfig, StreamEvent};

use crate::config::{SpecMethod, SpeculativeDecodingConfig};
use crate::error::ApiError;
use crate::metrics::RequestStatus;
use crate::recent_requests::{RequestRecord, now_unix_ms, truncate_chars};
use crate::state::{AppState, ModelBackend, RealPrefixCache};

/// Max characters retained in the prompt preview for the recent-requests panel.
const PROMPT_PREVIEW_MAX_CHARS: usize = 120;
/// Max characters retained in the completion preview for the recent-requests panel.
const COMPLETION_PREVIEW_MAX_CHARS: usize = 200;

/// Pull the most recent user-authored message text from a request, falling
/// back to the very last message if no user role is present. Returns an empty
/// string if there are no messages.
fn last_user_message_text(req: &ChatCompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| req.messages.last())
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

/// If the rendered chat-template prompt prefilled an opening reasoning tag
/// into the assistant turn (Qwen3.5's official template ends with
/// `<|im_start|>assistant\n<think>\n` whenever `enable_thinking` isn't
/// explicitly false), return the literal text the model is continuing from
/// so the caller can re-emit it on the wire. Without this, the model
/// generates "Thinking Process: ..." with no opening `<think>` tag and any
/// client/UI that splits on `<think>...</think>` to render the chain of
/// thought separately fails to detect a reasoning block at all.
///
/// Conservative: only fires when the prompt ends with the exact `<think>\n`
/// suffix Qwen3.5 emits, so non-reasoning templates and the bare ChatML
/// fallback are unaffected.
fn prefilled_assistant_opener(prompt_text: &str) -> Option<&'static str> {
    if prompt_text.ends_with("<think>\n") {
        Some("<think>\n")
    } else {
        None
    }
}

/// Push a [`RequestRecord`] into the dashboard's recent-requests ring. Logs a
/// warning if the lock is poisoned but otherwise never panics — request
/// recording must not fail the user's request.
fn record_recent_request(state: &AppState, record: RequestRecord) {
    match state.recent_requests.lock() {
        Ok(mut ring) => ring.record(record),
        Err(poisoned) => poisoned.into_inner().record(record),
    }
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// When omitted, the server falls back to its configured `served_model_id`.
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Kiln extension: which LoRA adapter to use for this request.
    #[serde(default)]
    pub adapter: Option<String>,
    /// Kiln extension: stack multiple LoRA adapters with per-source scaling.
    /// Mutually exclusive with `adapter`. The composed adapter is merged once
    /// (via `merge_concat`) and cached on disk under `adapter_dir/.composed/`,
    /// keyed by a hash of the (name, scale) pairs.
    #[serde(default)]
    pub adapters: Option<Vec<AdapterRef>>,
}

/// A single source adapter for per-request composition.
#[derive(Debug, Deserialize)]
pub struct AdapterRef {
    pub name: String,
    pub scale: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(deserialize_with = "deserialize_content")]
    pub content: String,
}

/// Accept `content` as either a plain string or an OpenAI-style array of
/// content parts (`[{"type": "text", "text": "..."}, ...]`). Text parts are
/// concatenated in order; non-text parts are ignored since kiln is text-only.
fn deserialize_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Content {
        Text(String),
        Parts(Vec<serde_json::Value>),
    }

    match Content::deserialize(deserializer)? {
        Content::Text(s) => Ok(s),
        Content::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                let Some(obj) = part.as_object() else {
                    continue;
                };
                let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if ty == "text" {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        out.push_str(text);
                    }
                }
            }
            Ok(out)
        }
    }
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI-compatible streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone)]
enum ResolvedSpeculativeMode {
    Off,
    SkipLayer(SpeculativeConfig),
    Mtp,
}

const MTP_MAX_PROMPT_TOKENS_DEFAULT: usize = 128;
const LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT: usize = 1024;
const LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL: usize = 4096;
const LONG_PROMPT_SKIP_LAYER_MIN_OUTPUT_TOKENS_DEFAULT: usize = 32;

fn native_mtp_enabled_for_metal() -> bool {
    std::env::var("KILN_ENABLE_METAL_NATIVE_MTP")
        .ok()
        .as_deref()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

fn native_mtp_allowed_for_state(state: &AppState) -> bool {
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return true;
    };
    let runner_guard = runner.read().unwrap();
    match runner_guard.weights.embed_tokens.device() {
        candle_core::Device::Metal(_) => native_mtp_enabled_for_metal(),
        _ => true,
    }
}

fn long_prompt_skip_layer_min_prompt_tokens_for_state(state: &AppState) -> usize {
    let ModelBackend::Real { runner, .. } = state.backend.as_ref() else {
        return LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT;
    };
    let runner_guard = runner.read().unwrap();
    match runner_guard.weights.embed_tokens.device() {
        candle_core::Device::Metal(_) => LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL,
        _ => LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
    }
}

fn resolve_skip_layer_config(
    model_config: &kiln_core::config::ModelConfig,
    speculative: &SpeculativeDecodingConfig,
) -> Option<SpeculativeConfig> {
    let config = SpeculativeConfig {
        num_speculative_tokens: speculative.num_speculative_tokens,
        draft_layers: speculative.draft_layers,
    };

    config.validate(model_config).ok().map(|_| config)
}

fn resolve_speculative_mode_from_config(
    model_config: &kiln_core::config::ModelConfig,
    speculative: &SpeculativeDecodingConfig,
    sampling: &SamplingParams,
    prompt_tokens: usize,
    mtp_supported: bool,
    has_active_lora: bool,
    native_mtp_allowed: bool,
    long_prompt_skip_layer_min_prompt_tokens: usize,
) -> ResolvedSpeculativeMode {
    let skip_layer = resolve_skip_layer_config(model_config, speculative);

    match speculative.effective_method() {
        SpecMethod::Off => ResolvedSpeculativeMode::Off,
        SpecMethod::SkipLayer => skip_layer
            .map(ResolvedSpeculativeMode::SkipLayer)
            .unwrap_or(ResolvedSpeculativeMode::Off),
        SpecMethod::Mtp => {
            let greedy_without_lora = sampling.temperature == 0.0 && !has_active_lora;
            if mtp_supported
                && native_mtp_allowed
                && greedy_without_lora
                && prompt_tokens <= MTP_MAX_PROMPT_TOKENS_DEFAULT
            {
                ResolvedSpeculativeMode::Mtp
            } else if greedy_without_lora
                && prompt_tokens >= long_prompt_skip_layer_min_prompt_tokens
                && sampling.max_tokens >= LONG_PROMPT_SKIP_LAYER_MIN_OUTPUT_TOKENS_DEFAULT
            {
                skip_layer
                    .map(ResolvedSpeculativeMode::SkipLayer)
                    .unwrap_or(ResolvedSpeculativeMode::Off)
            } else {
                ResolvedSpeculativeMode::Off
            }
        }
    }
}

fn resolve_speculative_mode(
    state: &AppState,
    sampling: &SamplingParams,
    prompt_tokens: usize,
    mtp_supported: bool,
    has_active_lora: bool,
) -> ResolvedSpeculativeMode {
    let speculative = SpeculativeDecodingConfig::from_env();
    resolve_speculative_mode_from_config(
        &state.model_config,
        &speculative,
        sampling,
        prompt_tokens,
        mtp_supported,
        has_active_lora,
        native_mtp_allowed_for_state(state),
        long_prompt_skip_layer_min_prompt_tokens_for_state(state),
    )
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = std::time::Instant::now();
    state.metrics.inc_active();

    let result = chat_completions_inner(&state, req).await;

    state.metrics.dec_active();
    let elapsed = start.elapsed().as_secs_f64();
    state.metrics.observe_duration(elapsed);

    match &result {
        Ok(_) => state.metrics.inc_request(RequestStatus::Ok),
        Err(e) => {
            if e.status == StatusCode::REQUEST_TIMEOUT {
                state.metrics.inc_request(RequestStatus::Timeout);
            } else {
                state.metrics.inc_request(RequestStatus::Error);
            }
        }
    }

    result
}

async fn chat_completions_inner(
    state: &AppState,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    // Captured at the top of the request so the recent-requests panel reflects
    // wall-clock time including chat-template formatting and tokenization, not
    // just generation. Streaming and non-streaming paths both consume this.
    let request_start = std::time::Instant::now();

    // Convert request messages to ChatMessage for template formatting
    let chat_messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    // Apply chat template and tokenize
    let prompt_text = state
        .tokenizer
        .apply_chat_template(&chat_messages)
        .map_err(ApiError::chat_template_failed)?;
    let prompt_tokens = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(ApiError::tokenization_failed)?;

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        max_tokens: req.max_tokens.unwrap_or(2048),
        stop: req.stop.clone().unwrap_or_default(),
        seed: req.seed,
        ..Default::default()
    };

    // Validate adapter / adapters mutual exclusion. Done up front (before
    // backend dispatch) so 400-on-misuse is observable from any backend.
    if req.adapter.is_some() && req.adapters.is_some() {
        return Err(ApiError::invalid_compose_request(
            "specify either 'adapter' (single name) or 'adapters' (list), not both",
        ));
    }
    if let Some(ref list) = req.adapters {
        if list.is_empty() {
            return Err(ApiError::invalid_compose_request(
                "'adapters' must be a non-empty list when present",
            ));
        }
        for src in list {
            validate_compose_name(&src.name)?;
        }
    }

    // If `adapters` is set, synthesize (or reuse cached) composed adapter on
    // disk. Runs regardless of backend so the cache is populated even in mock
    // mode tests; only the actual hot-swap is gated on the Real backend.
    let composed_target: Option<ComposedTarget> = if let Some(list) = req.adapters.as_deref() {
        Some(synthesize_composed_adapter(&state.adapter_dir, list).await?)
    } else {
        None
    };

    // Ensure the correct LoRA adapter is active for this request.
    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if let Some(ref target) = composed_target {
            ensure_composed_adapter_swap(state, runner, target).await?;
        } else {
            ensure_adapter(state, runner, &req.adapter).await?;
        }
    }

    if req.stream {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
                prefix_cache,
            } => {
                generate_real_streaming(
                    state,
                    runner,
                    block_manager,
                    paged_cache,
                    prefix_cache,
                    &prompt_text,
                    &prompt_tokens,
                    &sampling,
                    &req,
                    request_start,
                )
                .await
            }
            ModelBackend::Mock { .. } => Err(ApiError::streaming_not_supported_mock()),
        }
    } else {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
                prefix_cache,
            } => {
                let resp = generate_real(
                    state,
                    runner,
                    block_manager,
                    paged_cache,
                    prefix_cache,
                    &prompt_text,
                    &prompt_tokens,
                    &sampling,
                    &req,
                    request_start,
                )
                .await?;
                // Count generated tokens for metrics.
                state
                    .metrics
                    .add_tokens(resp.usage.completion_tokens as u64);
                Ok(Json(resp).into_response())
            }
            ModelBackend::Mock { scheduler, engine } => {
                let resp = generate_mock(
                    state,
                    scheduler,
                    engine,
                    &prompt_text,
                    &sampling,
                    &req,
                    request_start,
                )
                .await?;
                state
                    .metrics
                    .add_tokens(resp.usage.completion_tokens as u64);
                Ok(Json(resp).into_response())
            }
        }
    }
}

/// Ensure the correct LoRA adapter is active for the given request.
///
/// Compares `req_adapter` against the currently active adapter name. If they
/// differ, loads/unloads the adapter using the two-phase RwLock pattern
/// (read config, load weights outside lock, write-lock to swap).
async fn ensure_adapter(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    req_adapter: &Option<String>,
) -> Result<(), ApiError> {
    let current = state.active_adapter_name.read().unwrap().clone();
    if *req_adapter == current {
        return Ok(());
    }

    match req_adapter {
        Some(name) => {
            // Resolve adapter path
            let adapter_path = if Path::new(name).is_absolute() {
                std::path::PathBuf::from(name)
            } else {
                state.adapter_dir.join(name)
            };
            if !adapter_path.exists() {
                return Err(ApiError::adapter_not_found(name));
            }

            // Two-phase load: read device/num_layers, load weights, then swap.
            let (device, num_layers) = {
                let guard = runner.read().unwrap();
                (
                    guard.weights.embed_tokens.device().clone(),
                    guard.config.num_layers,
                )
            };

            let runner = runner.clone();
            let adapter_name = name.clone();
            let active_name = state.active_adapter_name.clone();

            tokio::task::spawn_blocking(move || {
                let lora = LoraWeights::load(&adapter_path, num_layers, &device)
                    .map_err(|e| format!("{e}"))?;
                let mut guard = runner.write().unwrap();
                guard.swap_lora(Some(lora));
                *active_name.write().unwrap() = Some(adapter_name);
                Ok::<(), String>(())
            })
            .await
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?
            .map_err(ApiError::adapter_load_failed)?;
            state.clear_real_prefix_cache();
        }
        None => {
            // Revert to base model.
            let runner = runner.clone();
            let active_name = state.active_adapter_name.clone();

            tokio::task::spawn_blocking(move || {
                let mut guard = runner.write().unwrap();
                guard.swap_lora(None);
                *active_name.write().unwrap() = None;
            })
            .await
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?;
            state.clear_real_prefix_cache();
        }
    }

    Ok(())
}

/// Disk handle for a composed adapter ready to be loaded.
#[derive(Debug, Clone)]
struct ComposedTarget {
    /// Stable cache name embedded in `active_adapter_name` once swapped in,
    /// e.g. `"__composed:abc123..."`. Used for cache-hit comparison and as
    /// the prefix-cache adapter key.
    active_name: String,
    /// On-disk directory holding the synthesized PEFT adapter.
    cache_dir: PathBuf,
}

/// Validate a single source-adapter name from an `adapters: [...]` request.
///
/// Names must be a single path segment with no separators or traversal — same
/// rules as `validate_adapter_name` in `api/adapters.rs`. Centralized here so
/// `chat_completions` can return a 404-shaped error consistent with the
/// existing single-adapter path (`adapter_not_found`).
fn validate_compose_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || Path::new(name).is_absolute()
    {
        return Err(ApiError::invalid_adapter_name(name));
    }
    Ok(())
}

/// Compute a stable hex hash for an `adapters` composition spec.
///
/// Hashes the sorted list of `"<name>@<scale>"` pairs with `DefaultHasher`
/// (deterministic SipHash-1-3 with key (0,0)). Used as the cache directory
/// name and the suffix of `active_adapter_name`.
fn composition_hash(adapters: &[AdapterRef]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut entries: Vec<String> = adapters
        .iter()
        .map(|a| format!("{}@{}", a.name, a.scale))
        .collect();
    entries.sort();

    let mut hasher = DefaultHasher::new();
    for e in &entries {
        e.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Synthesize (or reuse the on-disk cache for) a composed adapter spec.
///
/// On first call for a given hash, loads each source adapter, runs
/// `merge_concat`, and writes the result under `<adapter_dir>/.composed/<hash>/`.
/// On subsequent calls for the same hash, returns immediately without touching
/// the source files. Source-adapter lookup uses the same path resolution as
/// `ensure_adapter`: each `name` is treated as a single segment under
/// `adapter_dir`, missing sources surface as 404.
async fn synthesize_composed_adapter(
    adapter_dir: &Path,
    adapters: &[AdapterRef],
) -> Result<ComposedTarget, ApiError> {
    let hash = composition_hash(adapters);
    let active_name = format!("__composed:{hash}");
    let composed_root = adapter_dir.join(".composed");
    let cache_dir = composed_root.join(&hash);

    if cache_dir.exists() {
        return Ok(ComposedTarget {
            active_name,
            cache_dir,
        });
    }

    // Confirm every source exists before doing any merge work.
    let mut source_paths: Vec<(String, f32, PathBuf)> = Vec::with_capacity(adapters.len());
    for src in adapters {
        let path = adapter_dir.join(&src.name);
        if !path.exists() || !path.is_dir() {
            return Err(ApiError::adapter_not_found(&src.name));
        }
        source_paths.push((src.name.clone(), src.scale, path));
    }

    let composed_root = composed_root.clone();
    let cache_dir_for_task = cache_dir.clone();
    let merge_result = tokio::task::spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&composed_root)
            .map_err(|e| format!("creating composed-cache dir: {e}"))?;

        let mut loaded: Vec<(PeftLora, f32)> = Vec::with_capacity(source_paths.len());
        for (name, scale, path) in source_paths {
            let adapter = PeftLora::load(&path)
                .map_err(|e| format!("loading source '{name}' from {}: {e}", path.display()))?;
            loaded.push((adapter, scale));
        }

        let refs: Vec<(&PeftLora, f32)> = loaded.iter().map(|(a, s)| (a, *s)).collect();
        let merged = merge_concat(&refs).map_err(|e| format!("merge_concat: {e}"))?;
        merged
            .save(&cache_dir_for_task)
            .map_err(|e| format!("saving composed adapter: {e}"))?;
        Ok(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?;

    if let Err(msg) = merge_result {
        // Best-effort cleanup if we partially wrote anything before failing.
        let _ = std::fs::remove_dir_all(&cache_dir);
        return Err(ApiError::adapter_merge_failed(msg));
    }

    Ok(ComposedTarget {
        active_name,
        cache_dir,
    })
}

/// Hot-swap the runner onto a synthesized composed adapter.
///
/// Mirrors `ensure_adapter`'s two-phase RwLock pattern (read device + num
/// layers, load weights outside any lock, then write-lock to swap). No-op if
/// the composed adapter is already active.
async fn ensure_composed_adapter_swap(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    target: &ComposedTarget,
) -> Result<(), ApiError> {
    {
        let current = state.active_adapter_name.read().unwrap();
        if current.as_deref() == Some(target.active_name.as_str()) {
            return Ok(());
        }
    }

    let (device, num_layers) = {
        let guard = runner.read().unwrap();
        (
            guard.weights.embed_tokens.device().clone(),
            guard.config.num_layers,
        )
    };

    let runner = runner.clone();
    let active_name = state.active_adapter_name.clone();
    let cache_dir = target.cache_dir.clone();
    let composed_active = target.active_name.clone();

    tokio::task::spawn_blocking(move || {
        let lora = LoraWeights::load(&cache_dir, num_layers, &device)
            .map_err(|e| format!("{e}"))?;
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
        *active_name.write().unwrap() = Some(composed_active);
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?
    .map_err(ApiError::adapter_load_failed)?;
    state.clear_real_prefix_cache();

    Ok(())
}

/// Generate using the real ModelRunner with paged KV cache.
async fn generate_real(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<std::sync::Mutex<kiln_model::PagedKvCache>>,
    prefix_cache: &std::sync::Arc<std::sync::Mutex<RealPrefixCache>>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_token_count = prompt_tokens.len();

    let (mtp_supported, has_active_lora) = {
        let guard = runner.read().unwrap();
        (guard.weights.mtp.is_some(), guard.active_lora().is_some())
    };
    let speculative_mode = resolve_speculative_mode(
        state,
        sampling,
        prompt_token_count,
        mtp_supported,
        has_active_lora,
    );

    // ModelRunner.generate_paged() is CPU-bound; run on a blocking thread to
    // avoid starving the tokio runtime.
    let runner = runner.clone();
    let bm = block_manager.clone();
    let pc = paged_cache.clone();
    let prefix_cache = prefix_cache.clone();
    let prompt = prompt_text.to_owned();
    let prompt_tokens = prompt_tokens.to_vec();
    let params = sampling.clone();
    // For prefix-cache keying. After ensure_adapter / ensure_composed_adapter_swap,
    // state.active_adapter_name reflects the actually-loaded adapter (a
    // `__composed:<hash>` name when the request used `adapters: [...]`), which
    // is what we want as the cache key — distinct compositions and the base
    // model must not share cached blocks.
    let adapter = state.active_adapter_name.read().unwrap().clone();

    let gpu_lock = state.gpu_lock.clone();
    let timeout = state.request_timeout;
    let generation = tokio::task::spawn_blocking(move || {
        // Acquire GPU coordination read lock — allows concurrent inference,
        // but blocks while training holds the write lock.
        let _gpu_guard = gpu_lock.read().unwrap();
        let runner_guard = runner.read().unwrap();
        match speculative_mode {
            ResolvedSpeculativeMode::Off => {
                let prefix_enabled = {
                    let cache = prefix_cache.lock().unwrap();
                    cache.is_enabled()
                };
                if !prefix_enabled {
                    runner_guard.generate_paged_shared_tokens(
                        &prompt_tokens,
                        &params,
                        bm.as_ref(),
                        pc.as_ref(),
                    )
                } else {
                    let hit = {
                        let mut cache = prefix_cache.lock().unwrap();
                        cache.lookup(&adapter, &prompt_tokens)?
                    };
                    let hit_entry_id = hit.as_ref().map(|hit| hit.entry_id);
                    let cached_prefix = hit.map(|hit| PagedPrefixReuse {
                        cached_tokens: hit.cached_tokens,
                        block_ids: hit.block_ids,
                        linear_state: hit.linear_state,
                    });

                    let result = runner_guard.generate_paged_shared_tokens_with_prefix_cache(
                        &prompt_tokens,
                        &params,
                        bm.as_ref(),
                        pc.as_ref(),
                        cached_prefix,
                    );

                    let mut output = match result {
                        Ok(output) => output,
                        Err(err) => {
                            if let Some(entry_id) = hit_entry_id {
                                let mut cache = prefix_cache.lock().unwrap();
                                cache.release_hit(entry_id);
                            }
                            return Err(err);
                        }
                    };
                    let registration = output.registration.take();
                    let allocated_blocks = std::mem::take(&mut output.allocated_blocks);
                    let mut retained_blocks = Vec::new();
                    let mut evicted_blocks = Vec::new();
                    {
                        let mut cache = prefix_cache.lock().unwrap();
                        if let Some(entry_id) = hit_entry_id {
                            cache.release_hit(entry_id);
                        }
                        if let Some(registration) = registration {
                            let outcome = cache.register(adapter.clone(), registration);
                            retained_blocks = outcome.retained_blocks;
                            evicted_blocks = outcome.evicted_blocks;
                        }
                    }

                    let mut blocks_to_free: Vec<u32> = allocated_blocks
                        .into_iter()
                        .filter(|block_id| !retained_blocks.contains(block_id))
                        .collect();
                    blocks_to_free.extend(evicted_blocks);
                    if !blocks_to_free.is_empty() {
                        let mut bm_guard = bm.lock().unwrap();
                        bm_guard.free_all(&blocks_to_free);
                    }

                    Ok(output.output)
                }
            }
            ResolvedSpeculativeMode::SkipLayer(spec_config) => {
                if params.temperature == 0.0 {
                    runner_guard.generate_paged_speculative_shared_tokens(
                        &prompt_tokens,
                        &params,
                        bm.as_ref(),
                        pc.as_ref(),
                        &spec_config,
                    )
                } else {
                    let flat_spec_config = SpeculativeConfig {
                        num_speculative_tokens: spec_config.num_speculative_tokens.min(4),
                        draft_layers: spec_config.draft_layers,
                    };
                    runner_guard.generate_speculative(&prompt, &params, &flat_spec_config)
                }
            }
            ResolvedSpeculativeMode::Mtp => {
                let output = runner_guard.generate_mtp_speculative(&prompt, &params)?;
                Ok(GenerationOutput {
                    text: output.text,
                    token_ids: output.token_ids,
                    finish_reason: output.finish_reason,
                })
            }
        }
    });

    let output = match tokio::time::timeout(timeout, generation).await {
        Ok(join_result) => join_result
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?
            .map_err(ApiError::generation_failed)?,
        Err(_) => {
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
    };

    let finish_reason = match output.finish_reason {
        kiln_model::FinishReason::Eos => "stop",
        kiln_model::FinishReason::MaxTokens => "length",
        kiln_model::FinishReason::StopSequence(_) => "stop",
    };

    let now = now_epoch();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_tokens = output.token_ids.len();
    // See `prefilled_assistant_opener` — Qwen3.5's chat template prefills
    // `<think>\n` into the assistant turn so the model never re-emits the
    // opening tag. Re-attach it here so the JSON response carries a
    // well-formed `<think>...</think>` block, matching the streaming path
    // and what vLLM/SGLang serve for the same template.
    let completion_text = match prefilled_assistant_opener(prompt_text) {
        Some(opener) => format!("{opener}{}", output.text),
        None => output.text,
    };

    record_recent_request(
        state,
        RequestRecord {
            id: id.clone(),
            timestamp_unix_ms: now_unix_ms(),
            model: model.clone(),
            prompt_preview: truncate_chars(&last_user_message_text(req), PROMPT_PREVIEW_MAX_CHARS),
            completion_preview: truncate_chars(&completion_text, COMPLETION_PREVIEW_MAX_CHARS),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            streamed: false,
            finish_reason: finish_reason.to_string(),
        },
    );

    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: completion_text,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
    })
}

/// Generate using the real ModelRunner with SSE streaming and paged KV cache.
///
/// Handles two cancellation paths:
/// 1. **Client disconnect**: When the SSE client drops the connection, the async
///    mpsc `tx.send()` fails. The forwarding task then drops `sync_rx`, which
///    causes `tx.send()` in the generation loop to fail, stopping generation.
/// 2. **Request timeout**: A `tokio::time::sleep` future races against the
///    forwarding loop. On timeout, the task drops `sync_rx`, stopping generation,
///    and sends an error event to the client.
async fn generate_real_streaming(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<std::sync::Mutex<kiln_model::PagedKvCache>>,
    prefix_cache: &std::sync::Arc<std::sync::Mutex<RealPrefixCache>>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
) -> Result<Response, ApiError> {
    let (mtp_supported, has_active_lora) = {
        let guard = runner.read().unwrap();
        (guard.weights.mtp.is_some(), guard.active_lora().is_some())
    };
    let speculative_mode = resolve_speculative_mode(
        state,
        sampling,
        prompt_tokens.len(),
        mtp_supported,
        has_active_lora,
    );

    let runner = runner.clone();
    let bm = block_manager.clone();
    let pc = paged_cache.clone();
    let prefix_cache = prefix_cache.clone();
    let prompt = prompt_text.to_owned();
    let prompt_token_count = prompt_tokens.len();
    let prompt_tokens = prompt_tokens.to_vec();
    let params = sampling.clone();
    // For prefix-cache keying. After ensure_adapter / ensure_composed_adapter_swap,
    // state.active_adapter_name reflects the actually-loaded adapter (a
    // `__composed:<hash>` name when the request used `adapters: [...]`), which
    // is what we want as the cache key — distinct compositions and the base
    // model must not share cached blocks.
    let adapter = state.active_adapter_name.read().unwrap().clone();
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_epoch();
    let gpu_lock = state.gpu_lock.clone();
    let timeout = state.request_timeout;
    let decode_stats = state.decode_stats.clone();
    let recent_requests = state.recent_requests.clone();
    let prompt_preview = truncate_chars(&last_user_message_text(req), PROMPT_PREVIEW_MAX_CHARS);

    // Use a tokio mpsc channel to bridge sync generation -> async SSE stream.
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);

    // Spawn a task that runs the blocking generation and converts to SSE events.
    tokio::task::spawn({
        let id = completion_id.clone();
        let model = model.clone();
        async move {
            // Accumulate the assistant content so we can store a preview in the
            // recent-requests ring once generation completes (or times out, or
            // the client disconnects).
            let mut completion_buf = String::new();
            let mut completion_token_count: u32 = 0;

            let record = |finish_reason: String,
                          completion: &str,
                          completion_tokens: u32| {
                let record = RequestRecord {
                    id: id.clone(),
                    timestamp_unix_ms: now_unix_ms(),
                    model: model.clone(),
                    prompt_preview: prompt_preview.clone(),
                    completion_preview: truncate_chars(
                        completion,
                        COMPLETION_PREVIEW_MAX_CHARS,
                    ),
                    prompt_tokens: prompt_token_count as u32,
                    completion_tokens,
                    duration_ms: request_start.elapsed().as_millis() as u64,
                    streamed: true,
                    finish_reason,
                };
                match recent_requests.lock() {
                    Ok(mut ring) => ring.record(record),
                    Err(poisoned) => poisoned.into_inner().record(record),
                }
            };

            // Send initial role chunk
            let role_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            if tx
                .send(Event::default().data(serde_json::to_string(&role_chunk).unwrap()))
                .await
                .is_err()
            {
                record(
                    "client_disconnect".to_string(),
                    &completion_buf,
                    completion_token_count,
                );
                return;
            }

            // Qwen3.5's chat template prefills `<think>\n` into the assistant
            // turn so the model continues directly with reasoning content
            // (never re-emitting the opening tag). Without help, OpenAI-compat
            // clients see a stream that opens mid-thought and have no way to
            // detect a `<think>...</think>` block. Re-emit the prefilled tag
            // as a synthetic content chunk so the wire format is well-formed
            // — `<think>\n` + model output + (model emits) `</think>\n` +
            // answer — matching what vLLM/SGLang do for the same template.
            if let Some(opener) = prefilled_assistant_opener(&prompt) {
                let opener_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(opener.to_string()),
                        },
                        finish_reason: None,
                    }],
                };
                if tx
                    .send(Event::default().data(serde_json::to_string(&opener_chunk).unwrap()))
                    .await
                    .is_err()
                {
                    record(
                        "client_disconnect".to_string(),
                        &completion_buf,
                        completion_token_count,
                    );
                    return;
                }
                completion_buf.push_str(opener);
            }

            let sync_rx = match speculative_mode {
                ResolvedSpeculativeMode::Off => {
                    // Spawn-via-blocking so that prefill (which is itself a
                    // blocking GPU call) runs off the async runtime, then
                    // hands the receiver back. The actual decode loop runs on
                    // its own `std::thread` inside `spawn_streaming_*`, so
                    // this `spawn_blocking` returns as soon as prefill emits
                    // the first token — *not* after `max_tokens` steps. That
                    // is what makes `stream: true` actually stream over SSE
                    // instead of buffering everything until generation is
                    // done.
                    match tokio::task::spawn_blocking(move || {
                        let _gpu_guard = gpu_lock.read().unwrap();
                        let prefix_enabled = {
                            let cache = prefix_cache.lock().unwrap();
                            cache.is_enabled()
                        };
                        if !prefix_enabled {
                            kiln_model::ModelRunner::spawn_streaming_paged_shared_tokens(
                                runner.clone(),
                                prompt_tokens.clone(),
                                params.clone(),
                                bm.clone(),
                                pc.clone(),
                            )
                        } else {
                            let hit = {
                                let mut cache = prefix_cache.lock().unwrap();
                                cache.lookup(&adapter, &prompt_tokens)?
                            };
                            let hit_entry_id = hit.as_ref().map(|hit| hit.entry_id);
                            let cached_prefix = hit.map(|hit| PagedPrefixReuse {
                                cached_tokens: hit.cached_tokens,
                                block_ids: hit.block_ids,
                                linear_state: hit.linear_state,
                            });

                            let result = kiln_model::ModelRunner::
                                spawn_streaming_paged_shared_tokens_with_prefix_cache(
                                    runner.clone(),
                                    prompt_tokens.clone(),
                                    params.clone(),
                                    bm.clone(),
                                    pc.clone(),
                                    cached_prefix,
                                );

                            let mut output = match result {
                                Ok(output) => output,
                                Err(err) => {
                                    if let Some(entry_id) = hit_entry_id {
                                        let mut cache = prefix_cache.lock().unwrap();
                                        cache.release_hit(entry_id);
                                    }
                                    return Err(err);
                                }
                            };
                            let registration = output.registration.take();
                            let allocated_blocks = std::mem::take(&mut output.allocated_blocks);
                            let block_free_signal = output.block_free_signal.take();
                            let mut retained_blocks = Vec::new();
                            let mut evicted_blocks = Vec::new();
                            {
                                let mut cache = prefix_cache.lock().unwrap();
                                if let Some(entry_id) = hit_entry_id {
                                    cache.release_hit(entry_id);
                                }
                                if let Some(registration) = registration {
                                    let outcome = cache.register(adapter.clone(), registration);
                                    retained_blocks = outcome.retained_blocks;
                                    evicted_blocks = outcome.evicted_blocks;
                                }
                            }

                            let mut blocks_to_free: Vec<u32> = allocated_blocks
                                .into_iter()
                                .filter(|block_id| !retained_blocks.contains(block_id))
                                .collect();
                            blocks_to_free.extend(evicted_blocks);
                            // For the threaded streaming path, NEVER free
                            // here — the spawned decode worker is still
                            // reading these block ids for KV. Hand the
                            // computed set to the worker via its
                            // rendezvous channel; it frees after the
                            // decode loop finishes. Calling
                            // `bm.free_all` synchronously here was the
                            // root cause of the second-and-later same-
                            // prompt regression to "毎回毎回..." — the
                            // BlockManager handed those same block ids
                            // back out to the next request before the
                            // running decode loop was done with them.
                            // The legacy synchronous path still leaves
                            // `block_free_signal == None` and gets the
                            // immediate-free behavior below.
                            if let Some(signal) = block_free_signal {
                                let _ = signal.send(blocks_to_free);
                            } else if !blocks_to_free.is_empty() {
                                let mut bm_guard = bm.lock().unwrap();
                                bm_guard.free_all(&blocks_to_free);
                            }

                            Ok(output.receiver)
                        }
                    })
                    .await
                    {
                        Ok(Ok(rx)) => rx,
                        _ => {
                            record(
                                "error".to_string(),
                                &completion_buf,
                                completion_token_count,
                            );
                            let _ = tx.send(Event::default().data("[DONE]")).await;
                            return;
                        }
                    }
                }
                ResolvedSpeculativeMode::SkipLayer(spec_config) => {
                    match tokio::task::spawn_blocking(move || {
                        let _gpu_guard = gpu_lock.read().unwrap();
                        let runner_guard = runner.read().unwrap();
                        if params.temperature == 0.0 {
                            runner_guard.generate_streaming_paged_speculative_shared_tokens(
                                &prompt_tokens,
                                &params,
                                bm.as_ref(),
                                pc.as_ref(),
                                &spec_config,
                            )
                        } else {
                            let flat_spec_config = SpeculativeConfig {
                                num_speculative_tokens: spec_config.num_speculative_tokens.min(4),
                                draft_layers: spec_config.draft_layers,
                            };
                            runner_guard.generate_streaming_speculative(
                                &prompt,
                                &params,
                                &flat_spec_config,
                            )
                        }
                    })
                    .await
                    {
                        Ok(Ok(rx)) => rx,
                        _ => {
                            record(
                                "error".to_string(),
                                &completion_buf,
                                completion_token_count,
                            );
                            let _ = tx.send(Event::default().data("[DONE]")).await;
                            return;
                        }
                    }
                }
                ResolvedSpeculativeMode::Mtp => {
                    match tokio::task::spawn_blocking(move || {
                        let _gpu_guard = gpu_lock.read().unwrap();
                        let runner_guard = runner.read().unwrap();
                        runner_guard.generate_streaming_mtp_speculative(&prompt, &params)
                    })
                    .await
                    {
                        Ok(Ok(rx)) => rx,
                        _ => {
                            record(
                                "error".to_string(),
                                &completion_buf,
                                completion_token_count,
                            );
                            let _ = tx.send(Event::default().data("[DONE]")).await;
                            return;
                        }
                    }
                }
            };

            // Forward token events as SSE, racing against a timeout.
            // When the timeout fires or client disconnects, we drop `sync_rx`,
            // which causes `tx.send()` in the generation loop to fail, stopping
            // generation and freeing KV cache blocks.
            //
            // We wrap `sync_rx.recv()` in `spawn_blocking` so it doesn't block
            // the async runtime. The receiver is moved into each spawn_blocking
            // call and returned along with the result so we can reuse it.
            let mut maybe_rx = Some(sync_rx);
            let mut timed_out = false;
            let deadline = tokio::time::Instant::now() + timeout;

            loop {
                let rx_inner = match maybe_rx.take() {
                    Some(r) => r,
                    None => break,
                };

                // Race: recv the next token vs. request timeout
                let recv_handle = tokio::task::spawn_blocking(move || {
                    let result = rx_inner.recv();
                    (rx_inner, result)
                });

                tokio::select! {
                    join_result = recv_handle => {
                        match join_result {
                            Ok((rx_back, Ok(StreamEvent::Token(token)))) => {
                                maybe_rx = Some(rx_back);
                                if let Ok(mut stats) = decode_stats.lock() {
                                    stats.record_token(std::time::Instant::now());
                                }
                                completion_token_count = completion_token_count.saturating_add(1);
                                // Cap the buffered preview text so an unbounded
                                // generation doesn't keep allocating.
                                if completion_buf.chars().count() < COMPLETION_PREVIEW_MAX_CHARS + 16 {
                                    completion_buf.push_str(&token.text);
                                }
                                let chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk",
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: Some(token.text),
                                        },
                                        finish_reason: None,
                                    }],
                                };
                                if tx
                                    .send(
                                        Event::default()
                                            .data(serde_json::to_string(&chunk).unwrap()),
                                    )
                                    .await
                                    .is_err()
                                {
                                    // Client disconnected — drop rx to stop generation
                                    record(
                                        "client_disconnect".to_string(),
                                        &completion_buf,
                                        completion_token_count,
                                    );
                                    return;
                                }
                            }
                            Ok((_, Ok(StreamEvent::Done(done)))) => {
                                let finish = match done.finish_reason {
                                    kiln_model::FinishReason::Eos => "stop",
                                    kiln_model::FinishReason::MaxTokens => "length",
                                    kiln_model::FinishReason::StopSequence(_) => "stop",
                                };
                                let chunk = ChatCompletionChunk {
                                    id: id.clone(),
                                    object: "chat.completion.chunk",
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                        },
                                        finish_reason: Some(finish.to_string()),
                                    }],
                                };
                                let _ = tx
                                    .send(
                                        Event::default()
                                            .data(serde_json::to_string(&chunk).unwrap()),
                                    )
                                    .await;
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                record(
                                    finish.to_string(),
                                    &completion_buf,
                                    completion_token_count,
                                );
                                return;
                            }
                            _ => {
                                // Channel closed or join error
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                record(
                                    "error".to_string(),
                                    &completion_buf,
                                    completion_token_count,
                                );
                                return;
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out = true;
                        // recv_handle is dropped, which will abort the
                        // spawn_blocking task. The sync_rx inside it will
                        // be dropped, causing the generation loop to stop.
                        break;
                    }
                }
            }

            if timed_out {
                let error_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("timeout".to_string()),
                    }],
                };
                let _ = tx
                    .send(Event::default().data(serde_json::to_string(&error_chunk).unwrap()))
                    .await;
                let _ = tx.send(Event::default().data("[DONE]")).await;
                record(
                    "timeout".to_string(),
                    &completion_buf,
                    completion_token_count,
                );
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(Ok::<_, std::convert::Infallible>);

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Generate using the mock engine + scheduler loop (existing behavior).
async fn generate_mock(
    state: &AppState,
    scheduler: &tokio::sync::Mutex<kiln_scheduler::Scheduler>,
    engine: &std::sync::Arc<dyn kiln_model::engine::Engine>,
    prompt_text: &str,
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_tokens = state
        .tokenizer
        .encode(prompt_text)
        .map_err(ApiError::tokenization_failed)?;

    let prompt_token_count = prompt_tokens.len();
    let request = Request::new(prompt_tokens, sampling.clone(), req.adapter.clone());
    let request_id = request.id;

    // Add to scheduler
    {
        let mut sched = scheduler.lock().await;
        sched.add_request(request);
    }

    // Run scheduler steps until this request completes.
    let max_steps = 100;
    let mut output_tokens = Vec::new();

    for _ in 0..max_steps {
        let mut sched = scheduler.lock().await;
        let step_output = sched.step();

        if step_output.scheduled.is_empty() {
            break;
        }

        // Build batch input
        let batch = kiln_model::engine::BatchInput {
            token_ids: vec![0; step_output.total_tokens],
            seqlens: step_output.scheduled.iter().map(|s| s.num_tokens).collect(),
            slot_mapping: vec![0; step_output.total_tokens],
            block_tables: step_output.scheduled.iter().map(|_| vec![0]).collect(),
            is_prefill: step_output.scheduled.iter().map(|s| s.is_prefill).collect(),
            request_ids: step_output.scheduled.iter().map(|s| s.request_id).collect(),
        };

        let engine_output = engine.step(&batch).map_err(ApiError::generation_failed)?;

        for (rid, token, finished) in &engine_output.results {
            if *rid == request_id {
                if let Some(t) = token {
                    output_tokens.push(*t);
                }
            }

            let prefill_processed = step_output
                .scheduled
                .iter()
                .find(|s| s.request_id == *rid && s.is_prefill)
                .map(|s| s.num_tokens);

            let finished = *finished || output_tokens.len() >= 20; // Mock: stop after 20 tokens
            sched.update_request(rid, *token, finished, prefill_processed);
        }

        // Check if our request is done
        if let Some(req) = sched.get_request(&request_id) {
            if matches!(
                req.state,
                kiln_core::request::RequestState::Complete
                    | kiln_core::request::RequestState::Cancelled
            ) {
                break;
            }
        } else {
            break;
        }
    }

    // Decode output tokens
    let completion_text = state
        .tokenizer
        .decode(&output_tokens)
        .unwrap_or_else(|_| format!("[{} tokens, decode failed]", output_tokens.len()));

    let now = now_epoch();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_tokens = output_tokens.len();

    record_recent_request(
        state,
        RequestRecord {
            id: id.clone(),
            timestamp_unix_ms: now_unix_ms(),
            model: model.clone(),
            prompt_preview: truncate_chars(&last_user_message_text(req), PROMPT_PREVIEW_MAX_CHARS),
            completion_preview: truncate_chars(&completion_text, COMPLETION_PREVIEW_MAX_CHARS),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            streamed: false,
            finish_reason: "stop".to_string(),
        },
    );

    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: completion_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
    })
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Maximum number of completions a single batch request may produce.
/// Total outputs = `prompts.len() * n.unwrap_or(1)`. Above this cap the
/// request is rejected with `batch_too_large` (400) so a runaway client
/// cannot pin the engine for an unbounded number of iterations.
const BATCH_MAX_TOTAL_OUTPUTS: usize = 64;

/// Batch completion request — generate completions for many prompts (and/or
/// many completions per prompt) in a single HTTP round-trip.
///
/// Designed for the GRPO loop: groups of `n` completions per prompt are
/// the unit of advantage normalization, and issuing N separate HTTP requests
/// per group adds non-trivial overhead. With this endpoint a GRPO worker
/// posts the whole group in one call and the iteration-level scheduler
/// batches the underlying prefill/decode steps.
///
/// `stream: true` is not supported on this endpoint — for v1 we only return
/// the aggregated final result. Per-prompt adapter override is also a future
/// extension; for v1 the entire batch shares a single adapter (or none, or a
/// single composition).
#[derive(Debug, Deserialize)]
pub struct BatchCompletionRequest {
    /// When omitted, the server falls back to its configured `served_model_id`.
    #[serde(default)]
    pub model: Option<String>,
    /// One messages array per prompt. Total outputs returned =
    /// `prompts.len() * n.unwrap_or(1)`.
    pub prompts: Vec<Vec<Message>>,
    /// Number of completions to generate per prompt. Defaults to 1.
    /// Must be >= 1 when set.
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Base seed. When set, each completion's effective seed is
    /// `seed.wrapping_add((prompt_index * n + completion_index) as u64)`
    /// so completions are deterministic across runs but distinct within
    /// a group — without that, identical prompts plus a fixed seed would
    /// produce identical outputs even at temperature > 0.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Single LoRA adapter applied to every prompt in the batch.
    /// Mutually exclusive with `adapters`.
    #[serde(default)]
    pub adapter: Option<String>,
    /// Composition spec applied once for the entire batch (same shape and
    /// caching as `/v1/chat/completions`). Mutually exclusive with `adapter`.
    /// Per-prompt adapter override is a future extension.
    #[serde(default)]
    pub adapters: Option<Vec<AdapterRef>>,
}

/// Aggregated batch response. `completions.len() == prompts.len() * n`.
#[derive(Debug, Serialize)]
pub struct BatchCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub completions: Vec<BatchCompletionItem>,
    /// Sum of per-completion usage. `prompt_tokens` counts each prompt once
    /// per completion (so a prompt with `n=4` contributes its prompt token
    /// count 4×), matching how a client would sum N independent calls.
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct BatchCompletionItem {
    pub prompt_index: usize,
    pub completion_index: usize,
    pub text: String,
    pub finish_reason: String,
    pub usage: Usage,
}

async fn batch_completions(
    State(state): State<AppState>,
    Json(req): Json<BatchCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = std::time::Instant::now();
    state.metrics.inc_active();

    let result = batch_completions_inner(&state, req).await;

    state.metrics.dec_active();
    let elapsed = start.elapsed().as_secs_f64();
    state.metrics.observe_duration(elapsed);

    match &result {
        Ok(_) => state.metrics.inc_request(RequestStatus::Ok),
        Err(e) => {
            if e.status == StatusCode::REQUEST_TIMEOUT {
                state.metrics.inc_request(RequestStatus::Timeout);
            } else {
                state.metrics.inc_request(RequestStatus::Error);
            }
        }
    }

    result
}

async fn batch_completions_inner(
    state: &AppState,
    req: BatchCompletionRequest,
) -> Result<Response, ApiError> {
    let n_per = req.n.unwrap_or(1);

    if req.prompts.is_empty() {
        return Err(ApiError::batch_invalid_request(
            "'prompts' must contain at least one messages array",
        ));
    }
    if n_per == 0 {
        return Err(ApiError::batch_invalid_request(
            "'n' must be >= 1 when set",
        ));
    }
    let total_outputs = req.prompts.len().saturating_mul(n_per);
    if total_outputs > BATCH_MAX_TOTAL_OUTPUTS {
        return Err(ApiError::batch_too_large(
            total_outputs,
            BATCH_MAX_TOTAL_OUTPUTS,
        ));
    }

    // Adapter validation. Same rules as the single-completion endpoint, but
    // applied once for the whole batch — per-prompt adapter override is a
    // future extension.
    if req.adapter.is_some() && req.adapters.is_some() {
        return Err(ApiError::invalid_compose_request(
            "specify either 'adapter' (single name) or 'adapters' (list), not both",
        ));
    }
    if let Some(ref list) = req.adapters {
        if list.is_empty() {
            return Err(ApiError::invalid_compose_request(
                "'adapters' must be a non-empty list when present",
            ));
        }
        for src in list {
            validate_compose_name(&src.name)?;
        }
    }

    // Resolve adapter once for the entire batch. After this returns,
    // state.active_adapter_name reflects the loaded adapter and every
    // synthesized per-output ChatCompletionRequest below leaves
    // `adapter`/`adapters` as None — generate_real reads the active adapter
    // from state, not from the request.
    let composed_target: Option<ComposedTarget> = if let Some(list) = req.adapters.as_deref() {
        Some(synthesize_composed_adapter(&state.adapter_dir, list).await?)
    } else {
        None
    };

    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if let Some(ref target) = composed_target {
            ensure_composed_adapter_swap(state, runner, target).await?;
        } else {
            ensure_adapter(state, runner, &req.adapter).await?;
        }
    }

    // Spawn one task per (prompt, completion) pair. Each task synthesizes a
    // ChatCompletionRequest with this prompt's messages and a derived seed,
    // then dispatches through the existing generate_real / generate_mock
    // path. The iteration-level scheduler is what actually batches concurrent
    // requests; we do not introduce a new code path inside the engine.
    let mut handles = Vec::with_capacity(total_outputs);
    for (prompt_idx, prompt_messages) in req.prompts.iter().enumerate() {
        for completion_idx in 0..n_per {
            let derived_seed = req
                .seed
                .map(|s| s.wrapping_add((prompt_idx * n_per + completion_idx) as u64));
            let messages: Vec<Message> = prompt_messages
                .iter()
                .map(|m| Message {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();
            let synth_req = ChatCompletionRequest {
                model: req.model.clone(),
                messages,
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                max_tokens: req.max_tokens,
                stream: false,
                stop: req.stop.clone(),
                seed: derived_seed,
                adapter: None,
                adapters: None,
            };
            let state_clone = state.clone();
            handles.push(tokio::spawn(async move {
                generate_one_response(&state_clone, synth_req).await
            }));
        }
    }

    let mut completions = Vec::with_capacity(total_outputs);
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (idx, handle) in handles.into_iter().enumerate() {
        let prompt_index = idx / n_per;
        let completion_index = idx % n_per;
        let resp = match handle.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err(ApiError::internal(format!(
                    "batch task join error: {e}"
                )));
            }
        };
        let choice = resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::internal("generate returned a response with no choices"))?;
        total_prompt_tokens = total_prompt_tokens.saturating_add(resp.usage.prompt_tokens);
        total_completion_tokens =
            total_completion_tokens.saturating_add(resp.usage.completion_tokens);
        completions.push(BatchCompletionItem {
            prompt_index,
            completion_index,
            text: choice.message.content,
            finish_reason: choice.finish_reason,
            usage: resp.usage,
        });
    }

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());

    Ok(Json(BatchCompletionResponse {
        id: format!("batchcmpl-{}", Uuid::new_v4()),
        object: "batch.completion",
        created: now_epoch(),
        model,
        completions,
        usage: Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens.saturating_add(total_completion_tokens),
        },
    })
    .into_response())
}

/// Run a single non-streaming completion against whichever backend is loaded.
/// Used by the batch endpoint to fan out N synthesized single-completion
/// requests in parallel.
///
/// The adapter is intentionally not re-resolved here — the caller (the batch
/// handler) resolves the adapter once for the whole batch. This avoids
/// pointless write-locking and re-loading the same adapter N times.
async fn generate_one_response(
    state: &AppState,
    req: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    let request_start = std::time::Instant::now();

    let chat_messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    let prompt_text = state
        .tokenizer
        .apply_chat_template(&chat_messages)
        .map_err(ApiError::chat_template_failed)?;
    let prompt_tokens = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(ApiError::tokenization_failed)?;

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        max_tokens: req.max_tokens.unwrap_or(2048),
        stop: req.stop.clone().unwrap_or_default(),
        seed: req.seed,
        ..Default::default()
    };

    match state.backend.as_ref() {
        ModelBackend::Real {
            runner,
            block_manager,
            paged_cache,
            prefix_cache,
        } => {
            let resp = generate_real(
                state,
                runner,
                block_manager,
                paged_cache,
                prefix_cache,
                &prompt_text,
                &prompt_tokens,
                &sampling,
                &req,
                request_start,
            )
            .await?;
            state
                .metrics
                .add_tokens(resp.usage.completion_tokens as u64);
            Ok(resp)
        }
        ModelBackend::Mock { scheduler, engine } => {
            let resp = generate_mock(
                state,
                scheduler,
                engine,
                &prompt_text,
                &sampling,
                &req,
                request_start,
            )
            .await?;
            state
                .metrics
                .add_tokens(resp.usage.completion_tokens as u64);
            Ok(resp)
        }
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions/batch", post(batch_completions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpecMethod;
    use kiln_core::config::ModelConfig;

    fn parse_request(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("request should deserialize")
    }

    fn make_sampling(temperature: f32) -> SamplingParams {
        SamplingParams {
            temperature,
            ..Default::default()
        }
    }

    #[test]
    fn content_accepts_plain_string() {
        let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}]}"#);
        assert_eq!(req.messages[0].content, "hello");
    }

    #[test]
    fn content_accepts_text_parts_array() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}]}"#,
        );
        assert_eq!(req.messages[0].content, "hello world");
    }

    #[test]
    fn content_ignores_non_text_parts() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"describe: "},{"type":"image_url","image_url":{"url":"https://example.com/a.png"}},{"type":"text","text":"done"}]}]}"#,
        );
        assert_eq!(req.messages[0].content, "describe: done");
    }

    #[test]
    fn content_empty_array_is_empty_string() {
        let req = parse_request(r#"{"messages":[{"role":"user","content":[]}]}"#);
        assert_eq!(req.messages[0].content, "");
    }

    #[test]
    fn content_mixed_messages_in_same_request() {
        let req = parse_request(
            r#"{"messages":[{"role":"system","content":"be nice"},{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        );
        assert_eq!(req.messages[0].content, "be nice");
        assert_eq!(req.messages[1].content, "hi");
    }

    #[test]
    fn speculative_toggle_defaults_to_skip_layer() {
        let cfg = SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Off,
            num_speculative_tokens: 4,
            draft_layers: 8,
        };
        let mode = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(1.0),
            16,
            false,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );

        match mode {
            ResolvedSpeculativeMode::SkipLayer(spec) => {
                assert_eq!(spec.num_speculative_tokens, 4);
                assert_eq!(spec.draft_layers, 8);
            }
            _ => panic!("desktop toggle should resolve to skip-layer"),
        }
    }

    #[test]
    fn mtp_requires_greedy_request_and_no_lora() {
        let cfg = SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Mtp,
            num_speculative_tokens: 4,
            draft_layers: 8,
        };

        let greedy = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            16,
            true,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(greedy, ResolvedSpeculativeMode::Mtp));

        let sampled = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.7),
            16,
            true,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(sampled, ResolvedSpeculativeMode::Off));

        let with_lora = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            16,
            true,
            true,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(with_lora, ResolvedSpeculativeMode::Off));

        let long_prompt = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
            true,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(long_prompt, ResolvedSpeculativeMode::SkipLayer(_)));

        let medium_prompt = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            512,
            true,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(medium_prompt, ResolvedSpeculativeMode::Off));

        let mut short_output_sampling = make_sampling(0.0);
        short_output_sampling.max_tokens = LONG_PROMPT_SKIP_LAYER_MIN_OUTPUT_TOKENS_DEFAULT - 1;
        let long_prompt_short_output = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &short_output_sampling,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
            true,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(
            long_prompt_short_output,
            ResolvedSpeculativeMode::Off
        ));
    }

    #[test]
    fn mtp_short_prompt_stays_off_when_native_mtp_is_disallowed() {
        let cfg = SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Mtp,
            num_speculative_tokens: 4,
            draft_layers: 8,
        };

        let mode = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            64,
            true,
            false,
            false,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );
        assert!(matches!(mode, ResolvedSpeculativeMode::Off));
    }

    #[test]
    fn invalid_skip_layer_config_falls_back_to_off() {
        let cfg = SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::SkipLayer,
            num_speculative_tokens: 4,
            draft_layers: ModelConfig::qwen3_5_4b().num_layers,
        };
        let mode = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(1.0),
            16,
            false,
            false,
            true,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
        );

        assert!(matches!(mode, ResolvedSpeculativeMode::Off));
    }

    #[test]
    fn mtp_metal_medium_prompt_stays_off_until_4096() {
        let cfg = SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Mtp,
            num_speculative_tokens: 4,
            draft_layers: 8,
        };

        let mode = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            2048,
            true,
            false,
            false,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL,
        );
        assert!(matches!(mode, ResolvedSpeculativeMode::Off));
    }

    #[test]
    fn mtp_metal_4096_prompt_falls_back_to_skip_layer() {
        let cfg = SpeculativeDecodingConfig {
            enabled: true,
            method: SpecMethod::Mtp,
            num_speculative_tokens: 4,
            draft_layers: 8,
        };

        let mode = resolve_speculative_mode_from_config(
            &ModelConfig::qwen3_5_4b(),
            &cfg,
            &make_sampling(0.0),
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL,
            true,
            false,
            false,
            LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL,
        );
        assert!(matches!(mode, ResolvedSpeculativeMode::SkipLayer(_)));
    }

    // ── Batch completion endpoint ───────────────────────────────────

    fn parse_batch_request(json: &str) -> BatchCompletionRequest {
        serde_json::from_str(json).expect("batch request should deserialize")
    }

    fn make_batch_test_state() -> AppState {
        let config = ModelConfig::qwen3_5_4b();
        let sched_config = kiln_scheduler::SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
        };
        let scheduler = kiln_scheduler::Scheduler::new(sched_config, 256);
        let engine = kiln_model::engine::MockEngine::new(config.clone());
        let tokenizer = crate::api::test_tokenizer();
        AppState::new_mock(
            config,
            scheduler,
            std::sync::Arc::new(engine),
            tokenizer,
            300,
            "kiln-test".to_string(),
        )
    }

    /// Build a minimal request body, invoke the route, and return (status, body).
    async fn batch_post(state: AppState, body_json: &str) -> (axum::http::StatusCode, serde_json::Value) {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions/batch")
                    .header("content-type", "application/json")
                    .body(Body::from(body_json.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    #[test]
    fn batch_request_parses_minimal_shape() {
        let req = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"hi"}]]}"#,
        );
        assert_eq!(req.prompts.len(), 1);
        assert_eq!(req.prompts[0][0].role, "user");
        assert_eq!(req.prompts[0][0].content, "hi");
        assert!(req.n.is_none());
        assert!(req.seed.is_none());
        assert!(req.adapter.is_none());
        assert!(req.adapters.is_none());
    }

    #[test]
    fn batch_request_parses_full_shape() {
        let req = parse_batch_request(
            r#"{
                "prompts":[
                    [{"role":"user","content":"a"}],
                    [{"role":"user","content":"b"}]
                ],
                "n":4,
                "temperature":0.7,
                "top_p":0.95,
                "top_k":40,
                "max_tokens":32,
                "stop":["\n\n"],
                "seed":1234,
                "adapter":"my-adapter"
            }"#,
        );
        assert_eq!(req.prompts.len(), 2);
        assert_eq!(req.n, Some(4));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.95));
        assert_eq!(req.top_k, Some(40));
        assert_eq!(req.max_tokens, Some(32));
        assert_eq!(req.stop.as_deref(), Some(&["\n\n".to_string()][..]));
        assert_eq!(req.seed, Some(1234));
        assert_eq!(req.adapter.as_deref(), Some("my-adapter"));
    }

    #[tokio::test]
    async fn batch_rejects_empty_prompts() {
        let (status, body) =
            batch_post(make_batch_test_state(), r#"{"prompts":[]}"#).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "batch_invalid_request");
    }

    #[tokio::test]
    async fn batch_rejects_zero_n() {
        let (status, body) = batch_post(
            make_batch_test_state(),
            r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":0}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "batch_invalid_request");
    }

    #[tokio::test]
    async fn batch_rejects_too_many_outputs() {
        // 65 prompts * 1 = 65 > BATCH_MAX_TOTAL_OUTPUTS (64)
        let prompts: Vec<serde_json::Value> = (0..65)
            .map(|_| serde_json::json!([{"role":"user","content":"hi"}]))
            .collect();
        let req_body = serde_json::json!({"prompts": prompts}).to_string();
        let (status, body) = batch_post(make_batch_test_state(), &req_body).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "batch_too_large");
    }

    #[tokio::test]
    async fn batch_rejects_too_many_outputs_via_n_multiplier() {
        // 8 prompts * 9 = 72 > 64 — proves the cap counts the product, not just prompts.len().
        let prompts: Vec<serde_json::Value> = (0..8)
            .map(|_| serde_json::json!([{"role":"user","content":"hi"}]))
            .collect();
        let req_body = serde_json::json!({"prompts": prompts, "n": 9}).to_string();
        let (status, body) = batch_post(make_batch_test_state(), &req_body).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "batch_too_large");
    }

    #[tokio::test]
    async fn batch_rejects_adapter_and_adapters_together() {
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"hi"}]],
            "adapter": "single",
            "adapters": [{"name":"a","scale":1.0}]
        })
        .to_string();
        let (status, body) = batch_post(make_batch_test_state(), &body).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_compose_request");
    }

    #[tokio::test]
    async fn batch_rejects_empty_adapters_list() {
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"hi"}]],
            "adapters": []
        })
        .to_string();
        let (status, body) = batch_post(make_batch_test_state(), &body).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "invalid_compose_request");
    }

    #[test]
    fn batch_seed_derivation_is_distinct_per_output() {
        // Verifies the documented derivation: per-output seed =
        // base.wrapping_add(prompt_idx * n + completion_idx).
        let base: u64 = 42;
        let n_per = 3;
        let mut seeds = std::collections::HashSet::new();
        for prompt_idx in 0..2usize {
            for completion_idx in 0..n_per {
                let derived = base.wrapping_add((prompt_idx * n_per + completion_idx) as u64);
                assert!(
                    seeds.insert(derived),
                    "seed {derived} for ({prompt_idx},{completion_idx}) collides with an earlier output"
                );
            }
        }
        assert_eq!(seeds.len(), 2 * n_per);
    }

    #[test]
    fn batch_response_object_field_is_batch_completion() {
        // Lock down the discriminator string clients will key on so we don't
        // accidentally rename it.
        let resp = BatchCompletionResponse {
            id: "batchcmpl-test".to_string(),
            object: "batch.completion",
            created: 0,
            model: "kiln-test".to_string(),
            completions: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "batch.completion");
    }
}
