use anyhow::Context;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use tokio_stream::StreamExt;
#[cfg(test)]
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use kiln_core::config_hashes::ConfigHashes;
use kiln_core::request::Request;
use kiln_core::sampling::{SamplingParams, ThinkingBudget, ThinkingBudgetStatus};
use kiln_core::thinking_budget::{
    EffectiveThinkingBudget, ThinkingBudgetDefaults, ThinkingBudgetOutcome,
    ThinkingBudgetOverride as BudgetOverride, ThinkingBudgetOverrides, ThinkingBudgetRecord,
    ThinkingBudgetScope, ThinkingBudgetSource,
};
use kiln_core::token::TokenId;
use kiln_core::tokenizer::{ChatMessage, ChatTemplateOptions, TokenizerError};
use kiln_eval::qwen3::ParsedToolCall;
use kiln_model::adapter_merge::{PeftLora, merge_concat};
use kiln_model::{CancelHandle, ModelRunner};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::batching_engine::{EngineActionTokenSource, EngineEvent, EngineRequest};
use crate::error::ApiError;
use crate::gpu_coordination::write_guard_while_healthy_async;
use crate::latency_observability::{
    EngineTokenTiming, RequestLatencyDiagnostics, RequestLatencyTracker, TokenPhaseDurations,
};
use crate::memory_observability::CachedMemoryGovernorObservation;
use crate::metrics::RequestStatus;
use crate::recent_requests::{
    FULL_BODY_MAX_CHARS, RequestRecord, RequestThinkingBudget, now_unix_ms, truncate_chars,
};
use crate::state::{
    AppState, DeterministicBatchCache, DeterministicBatchCacheClaim, DeterministicBatchCacheItem,
    DeterministicBatchCacheKey, DeterministicBatchCacheValue, DeterministicBatchInFlightState,
    DeterministicCacheClaimId, DeterministicCacheKey, DeterministicChatChoicesCache,
    DeterministicChatChoicesCacheClaim, DeterministicChatChoicesCacheProbe,
    DeterministicChatChoicesCacheValue, DeterministicChatChoicesInFlightState,
    DeterministicChatRequestCache, DeterministicChatRequestCacheClaim,
    DeterministicChatRequestCacheProbe, DeterministicChatRequestCacheValue,
    DeterministicChatRequestInFlightState, DeterministicCompletionCacheClaim,
    DeterministicCompletionCacheKey, DeterministicCompletionCacheProbe,
    DeterministicCompletionCacheValue, DeterministicCompletionInFlightState, LoadedAdapterIdentity,
    ModelBackend,
};
use crate::teacher_identity::{
    MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES, MAX_COMPLETION_PROMPT_LOGPROBS,
    MAX_COMPLETION_PROMPT_TOKENS, MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS,
    PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET,
};

mod adapters;
mod batch;
mod cache_lifecycle;
mod finalization;
mod generation;
mod preparation;
mod prompt_logprobs;
mod schema;
mod streaming;
mod validation;

use adapters::*;
use batch::{CHAT_MAX_CHOICES, MAX_COMPOSE_ADAPTERS, batch_completions};
use cache_lifecycle::*;
use finalization::*;
use generation::*;
use preparation::*;
pub(crate) use preparation::{encode_prompt_tokens, render_prompt_text};
use prompt_logprobs::*;
pub use schema::*;
use streaming::*;
pub(crate) use validation::stop_sequence_conflicts_with_thinking_close;
use validation::*;

// Preserve the crate-visible wire-type surface while behavior moves into `batch`.
#[allow(unused_imports)]
pub use batch::{
    BatchCompletionItem, BatchCompletionMetadata, BatchCompletionRequest, BatchCompletionResponse,
};

/// Max characters retained in the prompt preview for the recent-requests panel.
const PROMPT_PREVIEW_MAX_CHARS: usize = 120;
/// Max characters retained in the completion preview for the recent-requests panel.
const COMPLETION_PREVIEW_MAX_CHARS: usize = 200;
const QWEN_TOOL_CALL_OPEN_TAG: &str = "<tool_call>";
const QWEN_TOOL_CALL_CLOSE_TAG: &str = "</tool_call>";
const MOCK_COMPLETION_TOKEN_LIMIT: usize = 20;

/// Push a [`RequestRecord`] into the dashboard's recent-requests ring. Logs a
/// warning if the lock is poisoned but otherwise never panics; request
/// recording must not fail the user's request.
fn record_recent_request(state: &AppState, record: RequestRecord) {
    maybe_log_slow_chat_completion(state, &record);
    if let Some(latency) = record.latency.as_ref() {
        state.metrics.observe_request_latency(latency);
    }
    if let Some(budget) = record.thinking_budget.as_ref() {
        state
            .metrics
            .observe_thinking_budget(budget, &record.finish_reason);
    }
    match state.recent_requests.lock() {
        Ok(mut ring) => ring.record(record),
        Err(poisoned) => poisoned.into_inner().record(record),
    }
}

fn record_failed_chat_completion(
    state: &AppState,
    req: &ChatCompletionRequest,
    model: &str,
    prompt_text: &str,
    request_start: std::time::Instant,
    prompt_tokens: usize,
    error: &ApiError,
) {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let thinking_budget = recent_thinking_budget_from_metadata(
        &thinking_budget_metadata_for_request(state, req, prompt_starts_in_reasoning(prompt_text)),
    );
    record_recent_request(
        state,
        RequestRecord {
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            prompt_tokens: prompt_tokens.min(u32::MAX as usize) as u32,
            completion_tokens: 0,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: "error".to_string(),
            error: Some(error.to_string()),
            thinking_mode: Some(thinking_mode_for_request(req).to_string()),
            prefix_cache: Some("unknown".to_string()),
            thinking_budget: Some(thinking_budget),
            ..request_record_from_req(state, req, &id, model, false)
        },
    );
}

fn adapter_header_value(adapter: Option<String>) -> HeaderValue {
    HeaderValue::from_str(adapter.as_deref().unwrap_or("base"))
        .unwrap_or_else(|_| HeaderValue::from_static("invalid"))
}

fn response_with_loaded_adapter_identity(
    mut response: Response,
    adapter: &Option<LoadedAdapterIdentity>,
) -> Response {
    response.headers_mut().insert(
        HeaderName::from_static("x-kiln-loaded-adapter"),
        adapter_header_value(adapter.as_ref().map(|identity| identity.name.clone())),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-kiln-loaded-adapter-revision"),
        adapter_header_value(
            adapter
                .as_ref()
                .map(|identity| identity.content_revision.clone()),
        ),
    );
    response
}

fn response_with_runtime_headers(state: &AppState, mut response: Response) -> Response {
    let eval_mode = if state.eval_mode {
        HeaderValue::from_static("true")
    } else {
        HeaderValue::from_static("false")
    };
    response
        .headers_mut()
        .insert(HeaderName::from_static("x-kiln-eval-mode"), eval_mode);
    response.headers_mut().insert(
        HeaderName::from_static("x-kiln-active-adapter"),
        adapter_header_value(state.active_adapter_name.read().unwrap().clone()),
    );
    if !response
        .headers()
        .contains_key("x-kiln-loaded-adapter-revision")
    {
        response =
            response_with_loaded_adapter_identity(response, &state.loaded_adapter_identity());
    }
    response
}

fn maybe_log_slow_chat_completion(state: &AppState, record: &RequestRecord) {
    let Some(values) = slow_request_log_values(state, record) else {
        return;
    };
    tracing::warn!(
        target: "kiln_server::slow_request",
        request_id = %values.request_id,
        adapter = %values.adapter,
        prompt_tokens = values.prompt_tokens,
        max_output_tokens = values.max_output_tokens,
        generated_tokens = values.generated_tokens,
        elapsed_ms = values.elapsed_ms,
        threshold_ms = values.threshold_ms,
        ttft_ms = ?values.ttft_ms,
        model_prefill_ms = ?values.model_prefill_ms,
        model_decode_ms = ?values.model_decode_ms,
        batching_engine_state = %values.batching_engine_state,
        thinking_mode = %values.thinking_mode,
        cuda_graph_state = %values.cuda_graph_state,
        prefix_cache = %values.prefix_cache,
        finish_reason = %values.finish_reason,
        error = %values.error,
        streamed = values.streamed,
        "slow chat completion"
    );
}

#[derive(Debug)]
struct SlowRequestLogValues {
    request_id: String,
    adapter: String,
    prompt_tokens: u32,
    max_output_tokens: u32,
    generated_tokens: u32,
    elapsed_ms: u64,
    threshold_ms: u64,
    ttft_ms: Option<u64>,
    model_prefill_ms: Option<u64>,
    model_decode_ms: Option<u64>,
    batching_engine_state: &'static str,
    thinking_mode: String,
    cuda_graph_state: &'static str,
    prefix_cache: String,
    finish_reason: String,
    error: String,
    streamed: bool,
}

fn slow_request_log_values(
    state: &AppState,
    record: &RequestRecord,
) -> Option<SlowRequestLogValues> {
    let Some(threshold) = state.slow_request_warn_threshold else {
        return None;
    };
    let elapsed = std::time::Duration::from_millis(record.duration_ms);
    if elapsed < threshold {
        return None;
    }

    let (batching_engine_state, cuda_graph_state) = slow_request_runtime_state(state);
    Some(SlowRequestLogValues {
        request_id: record.id.clone(),
        adapter: record
            .adapter
            .clone()
            .unwrap_or_else(|| "server_default_or_base".to_string()),
        prompt_tokens: record.prompt_tokens,
        max_output_tokens: record.max_tokens.unwrap_or(0),
        generated_tokens: record.completion_tokens,
        elapsed_ms: record.duration_ms,
        threshold_ms: threshold.as_millis() as u64,
        ttft_ms: record.ttft_ms,
        model_prefill_ms: record.model_prefill_ms,
        model_decode_ms: record.model_decode_ms,
        batching_engine_state,
        thinking_mode: record
            .thinking_mode
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        cuda_graph_state,
        prefix_cache: record
            .prefix_cache
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        finish_reason: record.finish_reason.clone(),
        error: record.error.clone().unwrap_or_default(),
        streamed: record.streamed,
    })
}

fn slow_request_runtime_state(state: &AppState) -> (&'static str, &'static str) {
    match state.backend.as_ref() {
        ModelBackend::Mock { .. } => ("mock", "not_applicable"),
        ModelBackend::Real { runner, .. } => {
            let cuda_graph = runner
                .try_read()
                .ok()
                .and_then(|runner| runner.cuda_graph_enabled().ok());
            let cuda_graph = match cuda_graph {
                Some(true) => "enabled",
                Some(false) => "disabled",
                None => "busy",
            };
            ("enabled", cuda_graph)
        }
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = std::time::Instant::now();
    let streaming = req.stream;
    if let Err(err) = ensure_backend_admission(&state) {
        state.metrics.inc_request(RequestStatus::Rejected);
        return Err(err);
    }
    // Attribute this request to the calling agent (pi / opencode / SDK / curl)
    // so the dashboard can show per-client traffic. Header-only, never trusted
    // from the body.
    req.user_agent = extract_user_agent(&headers);
    req.client = extract_client(&headers);
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

    if state.eval_mode && !streaming {
        state.clear_eval_mode_transient_state();
    }

    result.map(|response| response_with_runtime_headers(&state, response))
}

/// Execute an OpenEnv policy action through the authoritative chat handler
/// without requiring the server to call its own listening socket.
///
/// This intentionally preserves normal inference admission, adapter loading,
/// request attribution, metrics, timeout behavior, and response construction.
pub(crate) async fn openenv_chat_completion(
    state: &AppState,
    body: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let request: ChatCompletionRequest =
        serde_json::from_value(body).context("decode OpenEnv policy chat request")?;
    let mut headers = HeaderMap::new();
    headers.insert("x-kiln-client", HeaderValue::from_static("openenv"));
    let response = chat_completions(State(state.clone()), headers, Json(request))
        .await
        .map_err(|error| {
            anyhow::anyhow!(
                "Kiln action generation failed with {} {}: {}",
                error.status,
                error.code,
                error.message
            )
        })?;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), CHAT_BODY_LIMIT)
        .await
        .context("read in-process OpenEnv policy response")?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).context("decode in-process OpenEnv policy response")?;
    anyhow::ensure!(
        status.is_success(),
        "Kiln action generation returned HTTP {status}: {}",
        serde_json::to_string(&value).unwrap_or_default()
    );
    Ok(value)
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<TextCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = std::time::Instant::now();
    if let Err(err) = ensure_backend_admission(&state) {
        state.metrics.inc_request(RequestStatus::Rejected);
        return Err(err);
    }
    state.metrics.inc_active();

    let result = completions_inner(&state, req).await;

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

    result.map(|response| response_with_runtime_headers(&state, response))
}

async fn completions_inner(
    state: &AppState,
    req: TextCompletionRequest,
) -> Result<Response, ApiError> {
    if let Some(requested_model) = req.model.as_deref()
        && requested_model != state.served_model_id
    {
        return Err(ApiError::completion_invalid_request(format!(
            "model {requested_model:?} is not served by this process; expected {:?}",
            state.served_model_id
        )));
    }
    if req.stream {
        return Err(ApiError::completion_invalid_request(
            "stream=true is not supported on /v1/completions prompt-logprobs requests",
        ));
    }
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::completion_invalid_request(
            "'n' must be 1 when set; batched text completions are not supported",
        ));
    }
    let max_tokens = req.max_tokens.unwrap_or(1);
    if max_tokens > 1 {
        return Err(ApiError::completion_invalid_request(
            "only prompt-logprobs mode is supported; set max_tokens to 0 or 1",
        ));
    }
    let top_k = req.prompt_logprobs.ok_or_else(|| {
        ApiError::completion_invalid_request(
            "prompt_logprobs is required; generation-only text completions are not supported",
        )
    })?;
    validate_prompt_logprobs_top_k(state, top_k)?;
    let prompt_tokens =
        tokens_for_text_completion_prompt(state, &req.prompt, req.add_special_tokens)?;
    if prompt_tokens.is_empty() {
        return Err(ApiError::completion_invalid_request(
            "prompt must tokenize to at least one token",
        ));
    }
    // Raw token-ID prompts bypass the tokenizer, so out-of-range IDs would
    // otherwise reach the embedding lookup. Prompt length and total candidate
    // count are bounded independently: T=4096 and K=256 would exceed one
    // million decoded map entries even though each individual field is valid.
    if prompt_tokens.len() > MAX_COMPLETION_PROMPT_TOKENS {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt is {} tokens; prompt-logprobs requests are capped at \
             {MAX_COMPLETION_PROMPT_TOKENS} tokens",
            prompt_tokens.len()
        )));
    }
    if prompt_tokens.len() > state.model_config.max_position_embeddings {
        return Err(ApiError::context_length_exceeded(
            state.model_config.max_position_embeddings,
            prompt_tokens.len(),
            0,
        ));
    }
    let vocab_size = state.model_config.vocab_size;
    if let Some(bad) = prompt_tokens.iter().find(|&&t| (t as usize) >= vocab_size) {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt token id {bad} is out of range for vocab size {vocab_size}"
        )));
    }
    let scored_positions = prompt_tokens.len().saturating_sub(1);
    let candidate_upper_bound = scored_positions
        .checked_mul(top_k.saturating_add(1))
        .unwrap_or(usize::MAX);
    if candidate_upper_bound > MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt_logprobs response may contain {candidate_upper_bound} candidates; \
             kiln caps prompt-logprob responses at {MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES}"
        )));
    }

    let system_fingerprint = completion_system_fingerprint(state)?;
    let prompt_logprobs = match state.backend.as_ref() {
        ModelBackend::Mock { .. } => mock_prompt_logprobs(state, &prompt_tokens, top_k)?,
        ModelBackend::Real { runner, .. } => {
            real_prompt_logprobs(state, runner, &prompt_tokens, top_k).await?
        }
    };

    let model = state.served_model_id.clone();
    let response = TextCompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion",
        created: now_epoch(),
        model,
        system_fingerprint,
        choices: vec![TextCompletionChoice {
            index: 0,
            text: String::new(),
            finish_reason: "length".to_string(),
            prompt_logprobs: Some(prompt_logprobs),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: 0,
            total_tokens: prompt_tokens.len(),
        },
    };
    Ok(Json(response).into_response())
}

fn fresh_rollout_seed() -> u64 {
    let bytes = *Uuid::new_v4().as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn resolve_rollout_seed(req: &mut ChatCompletionRequest) {
    if req.rollout_provenance && req.seed.is_none() {
        req.seed = Some(fresh_rollout_seed());
    }
}

fn validate_rollout_provenance_admission(
    state: &AppState,
    req: &ChatCompletionRequest,
    n_per: usize,
) -> Result<(), ApiError> {
    if !req.rollout_provenance {
        return Ok(());
    }
    if req.ignore_eos {
        return Err(ApiError::rollout_provenance_unavailable(
            "ignore_eos=true is not represented by the current behavior-policy provenance schema",
        ));
    }
    if req.stream {
        return Err(ApiError::rollout_provenance_unavailable(
            "stream=true cannot atomically return the final token/action record",
        ));
    }
    if n_per != 1 {
        return Err(ApiError::rollout_provenance_unavailable(
            "n must be exactly 1",
        ));
    }
    if req.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        || req
            .tool_choice
            .as_ref()
            .is_some_and(|choice| !choice.is_null())
    {
        return Err(ApiError::rollout_provenance_unavailable(
            "request tools and tool_choice can produce tool-call outputs that the scored training payload cannot yet represent exactly",
        ));
    }
    match state.backend.as_ref() {
        ModelBackend::Real { .. } => {}
        ModelBackend::Mock { .. } => {
            return Err(ApiError::rollout_provenance_unavailable(
                "the mock backend has no behavior-policy identity or token probabilities",
            ));
        }
    }
    if state.base_teacher_identity.is_none() {
        return Err(ApiError::rollout_provenance_unavailable(
            "the real-model server did not publish a content-addressed base policy identity",
        ));
    }
    Ok(())
}

fn validate_rollout_provenance_generation_capacity(
    req: &ChatCompletionRequest,
    sampling: &SamplingParams,
) -> Result<(), ApiError> {
    if !req.rollout_provenance {
        return Ok(());
    }
    if sampling.max_tokens == 0 {
        return Err(ApiError::chat_invalid_request(
            "rollout_provenance=true requires an effective max_tokens greater than zero",
        ));
    }
    if let Some(budget) = sampling
        .thinking_budget
        .as_ref()
        .filter(|budget| sampling.max_tokens <= budget.close_token_count())
    {
        return Err(ApiError::chat_invalid_request(format!(
            "rollout_provenance=true requires effective max_tokens greater than the active tokenizer's {}-token thinking close sequence so at least one sampled action can be recorded",
            budget.close_token_count()
        )));
    }
    Ok(())
}

async fn chat_completions_inner(
    state: &AppState,
    mut req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let adapter_request_id = Uuid::new_v4().to_string();
    // Captured at the top of the request so the recent-requests panel reflects
    // wall-clock time including chat-template formatting and tokenization, not
    // just generation. Streaming and non-streaming paths both consume this.
    let request_start = std::time::Instant::now();

    let n_per = req.n.unwrap_or(1);
    if n_per == 0 {
        return Err(ApiError::chat_invalid_request("'n' must be >= 1 when set"));
    }
    if n_per > CHAT_MAX_CHOICES {
        return Err(ApiError::chat_invalid_request(format!(
            "'n' would produce {n_per} choices, which exceeds the cap of {CHAT_MAX_CHOICES}"
        )));
    }
    if n_per > 1 && req.stream {
        return Err(ApiError::chat_invalid_request(
            "'n' > 1 is not supported with stream=true",
        ));
    }

    apply_eval_mode_chat_defaults(state, &mut req);
    validate_rollout_provenance_admission(state, &req, n_per)?;
    resolve_rollout_seed(&mut req);
    let mut sampling = sampling_params_for_chat_request(&req);
    let effective_thinking_budget = effective_thinking_budget_for_request(state, &req);

    // Validate adapter / adapters mutual exclusion. Done up front (before
    // backend dispatch) so 400-on-misuse is observable from any backend.
    if req.adapter.is_explicit() && req.adapters.is_some() {
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
        if list.len() > MAX_COMPOSE_ADAPTERS {
            return Err(ApiError::invalid_compose_request(format!(
                "'adapters' list length {} exceeds maximum of {}",
                list.len(),
                MAX_COMPOSE_ADAPTERS,
            )));
        }
        for src in list {
            validate_compose_name(&src.name)?;
        }
    }

    if n_per > 1 {
        let stable_default_adapter = stable_default_adapter_identity(state);
        let cache_adapter = stable_default_adapter
            .clone()
            .unwrap_or_else(|| state.loaded_adapter_identity());
        let mut chat_choices_cache_key = if effective_thinking_budget.max_time_ms.is_some() {
            None
        } else {
            deterministic_chat_choices_cache_key_with_vocab_size_and_fold(
                &req,
                n_per,
                &sampling,
                state.model_config.vocab_size,
                fold_reasoning_into_content_for_request(state, &req),
                effective_thinking_budget.max_tokens,
            )?
            .map(|request| state.deterministic_cache_key(cache_adapter.clone(), request))
        };
        let can_hit_chat_choices_cache_before_adapter_work = stable_default_adapter.is_some();
        let mut chat_choices_cache_owner = None;
        if can_hit_chat_choices_cache_before_adapter_work
            && let Some(key) = chat_choices_cache_key.as_ref()
        {
            let claim = state.chat_choices_cache.lock().unwrap().claim(key);
            match claim {
                DeterministicChatChoicesCacheClaim::Hit(cached) => {
                    let resp =
                        response_from_cached_chat_choices(state, &req, request_start, cached);
                    store_chat_request_cache_from_chat_choices_response(
                        state,
                        &cache_adapter,
                        &req,
                        &resp,
                        state.model_config.vocab_size,
                    )?;
                    return Ok(response_with_loaded_adapter_identity(
                        Json(resp).into_response(),
                        &cache_adapter,
                    ));
                }
                DeterministicChatChoicesCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_chat_choices(receiver).await {
                        let resp =
                            response_from_cached_chat_choices(state, &req, request_start, cached);
                        store_chat_request_cache_from_chat_choices_response(
                            state,
                            &cache_adapter,
                            &req,
                            &resp,
                            state.model_config.vocab_size,
                        )?;
                        return Ok(response_with_loaded_adapter_identity(
                            Json(resp).into_response(),
                            &cache_adapter,
                        ));
                    }
                }
                DeterministicChatChoicesCacheClaim::Owner(claim_id) => {
                    chat_choices_cache_owner = Some(ChatChoicesCacheOwnerGuard::new(
                        state.chat_choices_cache.clone(),
                        key.clone(),
                        claim_id,
                    ));
                }
            }
        }

        if can_hit_chat_choices_cache_before_adapter_work
            && let Some(resp) = zero_chat_choices_response_from_request_cache_hit(
                state,
                &cache_adapter,
                &req,
                request_start,
                n_per,
                state.model_config.vocab_size,
            )
            .await?
        {
            finish_chat_choices_cache(
                state,
                chat_choices_cache_key,
                chat_choices_cache_owner.take(),
                &resp,
            );
            return Ok(response_with_loaded_adapter_identity(
                Json(resp).into_response(),
                &cache_adapter,
            ));
        }

        let (resp, loaded_adapter) =
            generate_multi_chat_response(state, &req, request_start, n_per).await?;
        if let Some(key) = chat_choices_cache_key.as_mut() {
            let rebound =
                state.deterministic_cache_key(loaded_adapter.clone(), key.request.clone());
            if chat_choices_cache_owner
                .as_ref()
                .is_some_and(|owner| !owner.matches_key(&rebound))
            {
                drop(chat_choices_cache_owner.take());
            }
            *key = rebound;
        }
        store_chat_request_cache_from_chat_choices_response(
            state,
            &loaded_adapter,
            &req,
            &resp,
            state.model_config.vocab_size,
        )?;
        finish_chat_choices_cache(
            state,
            chat_choices_cache_key,
            chat_choices_cache_owner.take(),
            &resp,
        );
        return Ok(response_with_loaded_adapter_identity(
            Json(resp).into_response(),
            &loaded_adapter,
        ));
    }

    let stable_default_adapter = stable_default_adapter_identity(state);
    let cache_adapter = stable_default_adapter
        .clone()
        .unwrap_or_else(|| state.loaded_adapter_identity());
    let mut chat_request_cache_key =
        if effective_thinking_budget.max_time_ms.is_some() || req.rollout_provenance {
            None
        } else {
            deterministic_chat_request_cache_key_with_vocab_size_and_fold(
                &req,
                &sampling,
                state.model_config.vocab_size,
                fold_reasoning_into_content_for_request(state, &req),
                effective_thinking_budget.max_tokens,
            )?
            .map(|request| state.deterministic_cache_key(cache_adapter.clone(), request))
        };
    let can_hit_chat_request_cache_before_adapter_work = stable_default_adapter.is_some();
    let mut chat_request_cache_owner = None;
    if can_hit_chat_request_cache_before_adapter_work
        && let Some(key) = chat_request_cache_key.as_ref()
    {
        if req.stream {
            let claim = state.chat_request_cache.lock().unwrap().claim(key);
            match claim {
                DeterministicChatRequestCacheClaim::Hit(cached) => {
                    return Ok(response_with_loaded_adapter_identity(
                        streaming_response_from_cached_chat_request(
                            state,
                            &req,
                            request_start,
                            cached,
                        ),
                        &cache_adapter,
                    ));
                }
                DeterministicChatRequestCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_chat_request(receiver).await {
                        return Ok(response_with_loaded_adapter_identity(
                            streaming_response_from_cached_chat_request(
                                state,
                                &req,
                                request_start,
                                cached,
                            ),
                            &cache_adapter,
                        ));
                    }
                }
                DeterministicChatRequestCacheClaim::Owner(claim_id) => {
                    chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                        state.chat_request_cache.clone(),
                        key.clone(),
                        claim_id,
                    ));
                }
            }
        } else {
            let claim = state.chat_request_cache.lock().unwrap().claim(key);
            match claim {
                DeterministicChatRequestCacheClaim::Hit(cached) => {
                    let resp =
                        response_from_cached_chat_request(state, &req, request_start, cached);
                    return Ok(response_with_loaded_adapter_identity(
                        Json(resp).into_response(),
                        &cache_adapter,
                    ));
                }
                DeterministicChatRequestCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_chat_request(receiver).await {
                        let resp =
                            response_from_cached_chat_request(state, &req, request_start, cached);
                        return Ok(response_with_loaded_adapter_identity(
                            Json(resp).into_response(),
                            &cache_adapter,
                        ));
                    }
                }
                DeterministicChatRequestCacheClaim::Owner(claim_id) => {
                    chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                        state.chat_request_cache.clone(),
                        key.clone(),
                        claim_id,
                    ));
                }
            }
        }
    }

    // Apply chat template and tokenize
    let prompt_text = render_prompt_text(
        state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        req.chat_template_kwargs.as_ref(),
    )?;
    let tokenization_started_at = std::time::Instant::now();
    let prompt_tokens = encode_prompt_tokens(state, &prompt_text)?;
    let tokenization_duration = tokenization_started_at.elapsed();
    enforce_context_window(state, &mut sampling, prompt_tokens.len())?;
    configure_thinking_budget_for_prompt(state, &req, &prompt_text, &mut sampling)?;
    validate_rollout_provenance_generation_capacity(&req, &sampling)?;

    if sampling.max_tokens == 0 {
        let cache_value = DeterministicChatRequestCacheValue {
            prompt_tokens: prompt_tokens.len(),
            completion: DeterministicCompletionCacheValue {
                text: String::new(),
                reasoning_content: None,
                tool_calls: None,
                finish_reason: "length".to_string(),
                completion_tokens: 0,
                thinking_budget_status: None,
            },
        };
        if let Some(owner) = chat_request_cache_owner.take() {
            owner.complete(cache_value.clone());
        } else if let Some(key) = chat_request_cache_key.clone() {
            state
                .chat_request_cache
                .lock()
                .unwrap()
                .insert(key, cache_value);
        }
        if req.stream {
            return Ok(response_with_loaded_adapter_identity(
                empty_chat_completion_streaming_response(
                    state,
                    &req,
                    prompt_tokens.len(),
                    request_start,
                ),
                &cache_adapter,
            ));
        }
        let mut resp =
            empty_chat_completion_response(state, &req, prompt_tokens.len(), request_start);
        resp.metadata = chat_completion_metadata_from_prompt(state, &req, &prompt_text);
        attach_chat_performance_metadata(
            state,
            &req,
            &mut resp,
            request_start,
            Some(std::time::Duration::ZERO),
            Some(std::time::Duration::ZERO),
            Some(std::time::Duration::ZERO),
        );
        return Ok(response_with_loaded_adapter_identity(
            Json(resp).into_response(),
            &cache_adapter,
        ));
    }

    // If `adapters` is set, synthesize (or reuse cached) composed adapter on
    // disk. Runs regardless of backend so the cache is populated even in mock
    // mode tests; only the actual hot-swap is gated on the Real backend.
    let has_composed_adapter = if let Some(list) = req.adapters.as_deref() {
        ensure_composed_adapter_for_request(state, list).await?;
        true
    } else {
        false
    };

    // Ensure the correct LoRA adapter is active for this request.
    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if !has_composed_adapter {
            ensure_adapter(state, runner, &req.adapter, &adapter_request_id).await?;
        }
    }

    let request_adapter = state.loaded_adapter_identity();
    if let Some(key) = chat_request_cache_key.as_mut() {
        let rebound = state.deterministic_cache_key(request_adapter.clone(), key.request.clone());
        if chat_request_cache_owner
            .as_ref()
            .is_some_and(|owner| !owner.matches_key(&rebound))
        {
            drop(chat_request_cache_owner.take());
        }
        *key = rebound;
    }

    let completion_cache_key = if req.rollout_provenance {
        None
    } else {
        deterministic_completion_cache_key_for_adapter(
            state,
            request_adapter.clone(),
            &prompt_tokens,
            &sampling,
            fold_reasoning_into_content_for_request(state, &req),
        )
    };
    let mut completion_cache_owner = None;
    if let Some(key) = completion_cache_key.as_ref() {
        if req.stream {
            let probe = state.completion_cache.lock().unwrap().probe(key);
            match probe {
                DeterministicCompletionCacheProbe::Hit(cached) => {
                    let chat_cache_value =
                        chat_request_cache_value_from_completion(prompt_tokens.len(), cached);
                    finish_chat_request_cache_value(
                        state,
                        chat_request_cache_key.clone(),
                        chat_request_cache_owner.take(),
                        chat_cache_value.clone(),
                    );
                    return Ok(response_with_loaded_adapter_identity(
                        streaming_response_from_cached_chat_request(
                            state,
                            &req,
                            request_start,
                            chat_cache_value,
                        ),
                        &request_adapter,
                    ));
                }
                DeterministicCompletionCacheProbe::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_completion(receiver).await {
                        let chat_cache_value =
                            chat_request_cache_value_from_completion(prompt_tokens.len(), cached);
                        finish_chat_request_cache_value(
                            state,
                            chat_request_cache_key.clone(),
                            chat_request_cache_owner.take(),
                            chat_cache_value.clone(),
                        );
                        return Ok(response_with_loaded_adapter_identity(
                            streaming_response_from_cached_chat_request(
                                state,
                                &req,
                                request_start,
                                chat_cache_value,
                            ),
                            &request_adapter,
                        ));
                    }
                }
                DeterministicCompletionCacheProbe::Miss => {}
            }
        } else {
            let claim = state.completion_cache.lock().unwrap().claim(key);
            match claim {
                DeterministicCompletionCacheClaim::Hit(cached) => {
                    let resp = response_from_cached_completion(
                        state,
                        &req,
                        prompt_tokens.len(),
                        request_start,
                        cached,
                    );
                    finish_chat_request_cache(
                        state,
                        chat_request_cache_key.clone(),
                        chat_request_cache_owner.take(),
                        &resp,
                    );
                    return Ok(response_with_loaded_adapter_identity(
                        Json(resp).into_response(),
                        &request_adapter,
                    ));
                }
                DeterministicCompletionCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_completion(receiver).await {
                        let resp = response_from_cached_completion(
                            state,
                            &req,
                            prompt_tokens.len(),
                            request_start,
                            cached,
                        );
                        finish_chat_request_cache(
                            state,
                            chat_request_cache_key.clone(),
                            chat_request_cache_owner.take(),
                            &resp,
                        );
                        return Ok(response_with_loaded_adapter_identity(
                            Json(resp).into_response(),
                            &request_adapter,
                        ));
                    }
                }
                DeterministicCompletionCacheClaim::Owner(claim_id) => {
                    completion_cache_owner = Some(claim_id);
                }
            }
        }
    }

    let result = if req.stream {
        match state.backend.as_ref() {
            ModelBackend::Real {
                batching_engine, ..
            } => {
                generate_real_batched_streaming(
                    state,
                    batching_engine,
                    request_adapter.clone(),
                    &prompt_text,
                    &prompt_tokens,
                    &sampling,
                    &req,
                    request_start,
                    Some(tokenization_duration),
                )
                .await
            }
            ModelBackend::Mock { .. } => Err(ApiError::streaming_not_supported_mock()),
        }
    } else {
        match state.backend.as_ref() {
            ModelBackend::Real {
                batching_engine, ..
            } => {
                let generation = generate_real_batched(
                    state,
                    batching_engine,
                    request_adapter.clone(),
                    &prompt_text,
                    &prompt_tokens,
                    &sampling,
                    &req,
                    request_start,
                    Some(tokenization_duration),
                )
                .await;
                let resp = match generation {
                    Ok(resp) => resp,
                    Err(err) => {
                        let model = req
                            .model
                            .clone()
                            .unwrap_or_else(|| state.served_model_id.clone());
                        record_failed_chat_completion(
                            state,
                            &req,
                            &model,
                            &prompt_text,
                            request_start,
                            prompt_tokens.len(),
                            &err,
                        );
                        if let (Some(claim_id), Some(key)) =
                            (completion_cache_owner, completion_cache_key.as_ref())
                        {
                            fail_deterministic_completion_owner(state, key, claim_id);
                        }
                        return Err(err);
                    }
                };
                if let Some(key) = completion_cache_key.clone() {
                    if let Some(claim_id) = completion_cache_owner {
                        complete_deterministic_completion_owner(state, key, claim_id, &resp);
                    } else {
                        store_deterministic_completion(state, key, &resp);
                    }
                }
                finish_chat_request_cache(
                    state,
                    chat_request_cache_key.clone(),
                    chat_request_cache_owner.take(),
                    &resp,
                );
                // Count generated tokens for metrics.
                state
                    .metrics
                    .add_tokens(resp.usage.completion_tokens as u64);
                Ok(Json(resp).into_response())
            }
            ModelBackend::Mock { scheduler, engine } => {
                let generation = generate_mock(
                    state,
                    scheduler,
                    engine,
                    &prompt_text,
                    &prompt_tokens,
                    &sampling,
                    &req,
                    request_start,
                )
                .await;
                let resp = match generation {
                    Ok(resp) => resp,
                    Err(err) => {
                        let model = req
                            .model
                            .clone()
                            .unwrap_or_else(|| state.served_model_id.clone());
                        record_failed_chat_completion(
                            state,
                            &req,
                            &model,
                            &prompt_text,
                            request_start,
                            prompt_tokens.len(),
                            &err,
                        );
                        if let (Some(claim_id), Some(key)) =
                            (completion_cache_owner, completion_cache_key.as_ref())
                        {
                            fail_deterministic_completion_owner(state, key, claim_id);
                        }
                        return Err(err);
                    }
                };
                if let Some(key) = completion_cache_key.clone() {
                    if let Some(claim_id) = completion_cache_owner {
                        complete_deterministic_completion_owner(state, key, claim_id, &resp);
                    } else {
                        store_deterministic_completion(state, key, &resp);
                    }
                }
                finish_chat_request_cache(
                    state,
                    chat_request_cache_key.clone(),
                    chat_request_cache_owner.take(),
                    &resp,
                );
                state
                    .metrics
                    .add_tokens(resp.usage.completion_tokens as u64);
                Ok(Json(resp).into_response())
            }
        }
    };
    if req.stream {
        if let Err(error) = &result {
            let model = req
                .model
                .clone()
                .unwrap_or_else(|| state.served_model_id.clone());
            record_failed_chat_completion(
                state,
                &req,
                &model,
                &prompt_text,
                request_start,
                prompt_tokens.len(),
                error,
            );
        }
    }
    result.map(|response| response_with_loaded_adapter_identity(response, &request_adapter))
}

/// Ensure the model runner has the adapter required for this chat request.
///
/// Missing `adapter` selects the server default without changing it. Explicit
/// `null`/`""` selects base for this request. A name selects that adapter for
/// this request. Only the adapter load/unload endpoints mutate the default.
fn rollout_sha256(value: &str) -> String {
    if value.starts_with("sha256:") {
        value.to_string()
    } else {
        format!("sha256:{value}")
    }
}

fn build_rollout_provenance(
    state: &AppState,
    req: &ChatCompletionRequest,
    adapter: Option<&LoadedAdapterIdentity>,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    output: &crate::batching_engine::BatchedGenerationOutput,
    scored_text: &str,
) -> Result<kiln_train::RolloutProvenanceV1, ApiError> {
    let build = || -> anyhow::Result<kiln_train::RolloutProvenanceV1> {
        let trace = output
            .action_tokens
            .as_ref()
            .context("batching engine omitted the requested behavior action trace")?;
        anyhow::ensure!(
            trace.len() == output.completion_tokens,
            "behavior action trace has {} entries for {} completion tokens",
            trace.len(),
            output.completion_tokens
        );
        let terminal_eos = matches!(&output.finish_reason, kiln_model::FinishReason::Eos);
        anyhow::ensure!(
            output.completion_tokens == output.token_ids.len() + usize::from(terminal_eos),
            "batching output reports {} completion tokens for {} visible token IDs and terminal_eos={terminal_eos}",
            output.completion_tokens,
            output.token_ids.len()
        );

        let action_tokens = trace
            .iter()
            .enumerate()
            .map(|(generated_index, action)| -> anyhow::Result<_> {
                anyhow::ensure!(
                    action.generated_index == generated_index,
                    "behavior action trace index {} appears at generated position {generated_index}",
                    action.generated_index
                );
                if let Some(&visible_token) = output.token_ids.get(generated_index) {
                    anyhow::ensure!(
                        visible_token == action.token_id,
                        "behavior action token {} differs from generated token {visible_token} at position {generated_index}",
                        action.token_id
                    );
                } else {
                    anyhow::ensure!(
                        terminal_eos && generated_index == output.token_ids.len(),
                        "behavior action token {} at position {generated_index} has no generated-token counterpart",
                        action.token_id
                    );
                    anyhow::ensure!(
                        state.tokenizer.eos_token_ids().contains(&action.token_id),
                        "terminal behavior action token {} is not an EOS token",
                        action.token_id
                    );
                }
                let sequence_index = prompt_tokens
                    .len()
                    .checked_add(generated_index)
                    .context("rollout action sequence index overflow")?;
                Ok(match action.source {
                    EngineActionTokenSource::Sampled => {
                        kiln_train::RolloutActionTokenV1::sampled(
                            sequence_index,
                            action.token_id,
                            f64::from(action.behavior_logprob.context(
                                "sampled behavior action is missing its selected-token log-probability",
                            )?),
                        )
                    }
                    EngineActionTokenSource::Forced => {
                        anyhow::ensure!(
                            action.behavior_logprob.is_none(),
                            "forced behavior action unexpectedly carries a log-probability"
                        );
                        kiln_train::RolloutActionTokenV1::forced(sequence_index, action.token_id)
                    }
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let base = state
            .base_teacher_identity
            .as_deref()
            .context("server base behavior-policy identity is unavailable")?;
        let behavior_policy = kiln_train::RolloutBehaviorPolicyIdentityV1 {
            served_model_id: base.served_model_id().to_string(),
            base_model_sha256: rollout_sha256(base.base_model_sha256()),
            adapter: adapter.map(|adapter| kiln_train::RolloutAdapterIdentityV1 {
                name: adapter.name.clone(),
                content_sha256: rollout_sha256(&adapter.content_revision),
            }),
            inference_config_sha256: rollout_sha256(base.inference_config_sha256()),
            implementation: base.implementation().to_string(),
        };
        let tokenizer = kiln_train::RolloutTokenizerIdentityV1 {
            vocab_sha256: rollout_sha256(base.tokenizer_vocab_sha256()),
            config_sha256: rollout_sha256(base.tokenizer_config_sha256()),
            chat_template_sha256: state
                .tokenizer
                .chat_template_sha256()
                .context("server tokenizer has no chat-template identity")?,
        };
        let thinking_budget =
            sampling
                .thinking_budget
                .as_ref()
                .map(|budget| kiln_train::RolloutThinkingBudgetV1 {
                    max_tokens: budget.max_tokens(),
                    max_time_ms: budget.max_time().map(duration_ms_u64),
                    close_token_ids: budget.close_token_ids().to_vec(),
                });
        let sampling_config = kiln_train::RolloutSamplingConfigV1 {
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
            min_p: sampling.min_p,
            max_tokens: sampling.max_tokens,
            repetition_penalty: sampling.repetition_penalty,
            presence_penalty: sampling.presence_penalty,
            frequency_penalty: sampling.frequency_penalty,
            stop: sampling.stop.clone(),
            thinking_budget,
        };
        let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
        let template_invocation = kiln_train::RolloutChatTemplateInvocationV1 {
            tools: normalized_tools.map_or_else(Vec::new, <[_]>::to_vec),
            tool_choice: normalized_tool_choice_for_cache(
                normalized_tools,
                req.tool_choice.as_ref(),
            )
            .cloned(),
            template_kwargs: effective_chat_template_kwargs(
                state.default_thinking_enabled,
                req.chat_template_kwargs.as_ref(),
            ),
        };
        let prompt_messages = req.messages.iter().map(message_to_chat).collect::<Vec<_>>();
        let prompt_messages_sha256 = kiln_train::rollout_prompt_messages_sha256(&prompt_messages)
            .map_err(anyhow::Error::msg)?;
        let scored_payload = kiln_train::ScoredRollout::legacy(scored_text.to_string(), 0.0);
        let scored_payload_sha256 = kiln_train::scored_rollout_payload_sha256(&scored_payload)
            .map_err(anyhow::Error::msg)?;
        let mut input_token_ids = Vec::with_capacity(
            prompt_tokens
                .len()
                .checked_add(trace.len())
                .context("rollout input token count overflow")?,
        );
        input_token_ids.extend_from_slice(prompt_tokens);
        input_token_ids.extend(trace.iter().map(|action| action.token_id));
        let generation_backend = match state.backend.as_ref() {
            ModelBackend::Real { runner, .. } => runner
                .read()
                .map_err(|_| anyhow::anyhow!("model runner lock poisoned"))?
                .backend_name()
                .to_string(),
            ModelBackend::Mock { .. } => anyhow::bail!("mock backend cannot emit provenance"),
        };

        kiln_train::RolloutProvenanceV1::new(
            input_token_ids,
            prompt_tokens.len(),
            prompt_messages_sha256,
            scored_payload_sha256,
            action_tokens,
            behavior_policy,
            tokenizer,
            sampling_config,
            sampling
                .seed
                .context("rollout sampling seed was not resolved")?,
            generation_backend,
        )
        .and_then(|provenance| provenance.with_template_invocation(template_invocation))
        .map_err(anyhow::Error::msg)
    };

    build().map_err(ApiError::generation_failed)
}

/// Generate using the real ModelRunner with paged KV cache.
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
fn request_values_are_effectively_greedy(temperature: Option<f32>, top_k: Option<u32>) -> bool {
    SamplingParams::values_are_effectively_greedy(
        requested_or_default_temperature(temperature),
        requested_or_default_top_k(top_k),
    )
}

fn batch_can_clone_deterministic_completions(req: &BatchCompletionRequest) -> bool {
    req.n.unwrap_or(1) > 1
        && request_values_are_effectively_greedy(req.temperature, req.top_k)
        && !batch_request_allows_tool_call_parsing(req)
}

fn batch_can_clone_identical_prompt_groups(req: &BatchCompletionRequest) -> bool {
    request_values_are_effectively_greedy(req.temperature, req.top_k)
        && !batch_request_allows_tool_call_parsing(req)
}

async fn generate_multi_chat_response(
    state: &AppState,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    n_per: usize,
) -> Result<(ChatCompletionResponse, Option<LoadedAdapterIdentity>), ApiError> {
    if chat_request_max_tokens(req) == 0 {
        let prompt_text = render_prompt_text(
            state,
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            req.chat_template_kwargs.as_ref(),
        )?;
        let prompt_tokens = encode_prompt_tokens(state, &prompt_text)?;
        let mut resp = response_from_cached_completion(
            state,
            req,
            prompt_tokens.len(),
            request_start,
            DeterministicCompletionCacheValue {
                text: String::new(),
                reasoning_content: None,
                tool_calls: None,
                finish_reason: "length".to_string(),
                completion_tokens: 0,
                thinking_budget_status: None,
            },
        );
        resp.metadata = chat_completion_metadata_from_prompt(state, req, &prompt_text);
        attach_chat_performance_metadata(
            state,
            req,
            &mut resp,
            request_start,
            Some(std::time::Duration::ZERO),
            Some(std::time::Duration::ZERO),
            Some(std::time::Duration::ZERO),
        );
        let response = chat_response_from_multi_responses(
            state,
            req,
            request_start,
            vec![(0, resp)],
            n_per,
            true,
        )?;
        return Ok((response, state.loaded_adapter_identity()));
    }

    let has_composed_adapter = if let Some(list) = req.adapters.as_deref() {
        ensure_composed_adapter_for_request(state, list).await?;
        true
    } else {
        false
    };

    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if !has_composed_adapter {
            ensure_adapter(state, runner, &req.adapter, &Uuid::new_v4().to_string()).await?;
        }
    }
    let request_adapter = state.loaded_adapter_identity();

    let clone_greedy_choices = effective_thinking_budget_for_request(state, req)
        .max_time_ms
        .is_none()
        && request_values_are_effectively_greedy(req.temperature, req.top_k);
    let completion_count = if clone_greedy_choices { 1 } else { n_per };
    let mut responses = Vec::with_capacity(n_per);
    let stop = normalized_stop_option_for_synthetic_request(req.stop.as_deref());
    let tools = normalized_tools_option_for_synthetic_request(req.tools.as_deref());
    let tool_choice = normalized_tool_choice_option_for_synthetic_request(
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
    );
    for completion_idx in 0..completion_count {
        let derived_seed = req
            .seed
            .map(|seed| seed.wrapping_add(completion_idx as u64));
        let synth_req = ChatCompletionRequest {
            model: req.model.clone(),
            messages: req.messages.clone(),
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            n: None,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            repetition_penalty: req.repetition_penalty,
            sampling_preset: req.sampling_preset.clone(),
            max_tokens: req.max_tokens,
            max_completion_tokens: req.max_completion_tokens,
            ignore_eos: req.ignore_eos,
            thinking_budget_tokens: req.thinking_budget_tokens,
            thinking_budget_ms: req.thinking_budget_ms,
            stream: false,
            stream_options: None,
            stop: stop.clone(),
            seed: derived_seed,
            adapter: ChatAdapterSelection::Default,
            adapters: None,
            tools: tools.clone(),
            tool_choice: tool_choice.clone(),
            chat_template_kwargs: req.chat_template_kwargs.clone(),
            fold_reasoning_into_content: req.fold_reasoning_into_content,
            include_performance: req.include_performance,
            include_config_hashes: req.include_config_hashes,
            rollout_provenance: false,
        };
        let resp = generate_one_response(state, synth_req, request_adapter.clone()).await?;
        responses.push((completion_idx, resp));
    }

    let response = chat_response_from_multi_responses(
        state,
        req,
        request_start,
        responses,
        n_per,
        clone_greedy_choices,
    )?;
    Ok((response, request_adapter))
}

fn chat_response_from_multi_responses(
    state: &AppState,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    mut responses: Vec<(usize, ChatCompletionResponse)>,
    n_per: usize,
    clone_first_response: bool,
) -> Result<ChatCompletionResponse, ApiError> {
    if clone_first_response {
        let first = responses
            .first()
            .map(|(_, resp)| resp.clone())
            .ok_or_else(|| ApiError::internal("chat n clone path produced no response"))?;
        for completion_idx in 1..n_per {
            responses.push((completion_idx, first.clone()));
        }
    }

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let mut prompt_tokens = None;
    let mut completion_tokens = 0usize;
    let mut metadata = None;
    let mut choices = Vec::with_capacity(n_per);
    for (completion_idx, resp) in responses {
        prompt_tokens.get_or_insert(resp.usage.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(resp.usage.completion_tokens);
        metadata.get_or_insert_with(|| resp.metadata.clone());
        let choice =
            resp.choices.into_iter().next().ok_or_else(|| {
                ApiError::internal("generate returned a response with no choices")
            })?;
        choices.push(Choice {
            index: completion_idx,
            message: choice.message,
            finish_reason: choice.finish_reason,
            thinking_budget: choice.thinking_budget,
            rollout_provenance: choice.rollout_provenance,
            completion_tokens: choice.completion_tokens,
        });
    }
    choices.sort_by_key(|choice| choice.index);

    let prompt_tokens = prompt_tokens.unwrap_or(0);
    let mut response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: now_epoch(),
        model,
        choices,
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        },
        metadata: metadata.unwrap_or_else(|| chat_completion_metadata_from_request(state, req)),
    };
    attach_chat_performance_metadata(state, req, &mut response, request_start, None, None, None);
    Ok(response)
}

/// Run a single non-streaming completion against whichever backend is loaded.
/// Used by the batch endpoint to fan out N synthesized single-completion
/// requests in parallel.
///
/// The adapter is intentionally not re-resolved here — the caller (the batch
/// handler) resolves the adapter once for the whole batch. This avoids
/// pointless write-locking and re-loading the same adapter N times.
/// Resolve a [`SamplingParams`] from a chat completion request. The
/// starting point is the Qwen3.5 thinking-general profile (or a named
/// preset if the request specified one); explicit fields on the request
/// override the preset values, so callers that send no sampling fields
/// get the model-card recommendation by default.
fn sampling_params_for_chat_request(req: &ChatCompletionRequest) -> SamplingParams {
    let mut base = preset_or_default_for_request(req);
    if let Some(t) = req.temperature {
        base.temperature = t;
    }
    if let Some(p) = req.top_p {
        base.top_p = p;
    }
    if let Some(k) = req.top_k {
        base.top_k = k;
    }
    if let Some(mp) = req.min_p {
        base.min_p = mp;
    }
    if let Some(pp) = req.presence_penalty {
        base.presence_penalty = pp;
    }
    if let Some(fp) = req.frequency_penalty {
        base.frequency_penalty = fp;
    }
    if let Some(rp) = req.repetition_penalty {
        base.repetition_penalty = rp;
    }
    base.max_tokens = chat_request_max_tokens(req);
    base.ignore_eos = req.ignore_eos;
    base.stop = stop_sequences_for_chat_generation(req);
    base.seed = req.seed;
    base
}

// NOTE: batch-request → SamplingParams conversion lives implicitly in
// the batch handler, which builds synthetic per-prompt
// `ChatCompletionRequest`s and routes them through
// `sampling_params_for_chat_request`. The previously-separate
// `sampling_params_for_batch_request` was dead code — every batch
// field already plumbs through the synthetic-request path.

/// Default values used to fill in omitted request fields when computing
/// cache keys. These MUST match the Qwen3.5-thinking-general start point
/// used by [`sampling_params_for_chat_request`] — if they diverge, cache
/// hits will silently use stale completions sampled at different
/// settings than the current request would produce.
pub(crate) fn requested_or_default_temperature(v: Option<f32>) -> f32 {
    v.unwrap_or(1.0)
}
pub(crate) fn requested_or_default_top_p(v: Option<f32>) -> f32 {
    v.unwrap_or(0.95)
}
pub(crate) fn requested_or_default_top_k(v: Option<u32>) -> u32 {
    v.unwrap_or(20)
}
pub(crate) fn requested_or_default_min_p(v: Option<f32>) -> f32 {
    v.unwrap_or(0.0)
}
pub(crate) fn requested_or_default_presence_penalty(v: Option<f32>) -> f32 {
    v.unwrap_or(1.5)
}
pub(crate) fn requested_or_default_frequency_penalty(v: Option<f32>) -> f32 {
    v.unwrap_or(0.0)
}
pub(crate) fn requested_or_default_repetition_penalty(v: Option<f32>) -> f32 {
    v.unwrap_or(1.0)
}

/// Map a `sampling_preset` string to its corresponding [`SamplingParams`]
/// starting point. Unknown values silently fall back to the Qwen3.5
/// thinking-general default — same shape a user gets with no preset.
/// Default profile selection for a chat request. An explicit
/// `sampling_preset` always wins; otherwise TOOLS-BEARING requests get
/// the Qwen3.5 thinking-coding profile (temperature 0.6,
/// presence_penalty 0.0) instead of thinking-general (1.0 / 1.5):
/// presence_penalty punishes re-emitting tokens, which for code means
/// punishing every reuse of an identifier — kiln's own preset docs call
/// the general profile wrong for code, yet every preset-less pi request
/// (pi ALWAYS sends tools) was getting it. Explicit per-field values on
/// the request still override the profile either way.
fn preset_or_default_for_request(req: &ChatCompletionRequest) -> SamplingParams {
    if req.sampling_preset.is_none() && req.tools.as_ref().is_some_and(|tools| !tools.is_empty()) {
        return SamplingParams::qwen3_thinking_coding();
    }
    preset_or_default(req.sampling_preset.as_deref())
}

fn preset_or_default(name: Option<&str>) -> SamplingParams {
    match name.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("greedy") => SamplingParams::greedy(),
        Some("qwen3-thinking-coding") | Some("qwen3.5-thinking-coding") => {
            SamplingParams::qwen3_thinking_coding()
        }
        Some("qwen3-non-thinking-general") | Some("qwen3.5-non-thinking-general") => {
            SamplingParams::qwen3_non_thinking_general()
        }
        Some("qwen3-non-thinking-reasoning") | Some("qwen3.5-non-thinking-reasoning") => {
            SamplingParams::qwen3_non_thinking_reasoning()
        }
        _ => SamplingParams::qwen3_thinking_general(),
    }
}

async fn generate_one_response(
    state: &AppState,
    mut req: ChatCompletionRequest,
    request_adapter: Option<LoadedAdapterIdentity>,
) -> Result<ChatCompletionResponse, ApiError> {
    let request_start = std::time::Instant::now();
    apply_eval_mode_chat_defaults(state, &mut req);
    let mut sampling = sampling_params_for_chat_request(&req);

    let chat_request_cache_key = if effective_thinking_budget_for_request(state, &req)
        .max_time_ms
        .is_some()
    {
        None
    } else {
        deterministic_chat_request_cache_key_with_vocab_size_and_fold(
            &req,
            &sampling,
            state.model_config.vocab_size,
            fold_reasoning_into_content_for_request(state, &req),
            effective_thinking_budget_for_request(state, &req).max_tokens,
        )?
        .map(|request| state.deterministic_cache_key(request_adapter.clone(), request))
    };
    let mut chat_request_cache_owner = None;
    if let Some(key) = chat_request_cache_key.as_ref() {
        let claim = state.chat_request_cache.lock().unwrap().claim(key);
        match claim {
            DeterministicChatRequestCacheClaim::Hit(cached) => {
                return Ok(response_from_cached_chat_request(
                    state,
                    &req,
                    request_start,
                    cached,
                ));
            }
            DeterministicChatRequestCacheClaim::Wait(receiver) => {
                if let Some(cached) = wait_for_deterministic_chat_request(receiver).await {
                    return Ok(response_from_cached_chat_request(
                        state,
                        &req,
                        request_start,
                        cached,
                    ));
                }
            }
            DeterministicChatRequestCacheClaim::Owner(claim_id) => {
                chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                    state.chat_request_cache.clone(),
                    key.clone(),
                    claim_id,
                ));
            }
        }
    }

    let prompt_text = render_prompt_text(
        state,
        &req.messages,
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        req.chat_template_kwargs.as_ref(),
    )?;
    let prompt_tokens = encode_prompt_tokens(state, &prompt_text)?;
    enforce_context_window(state, &mut sampling, prompt_tokens.len())?;
    configure_thinking_budget_for_prompt(state, &req, &prompt_text, &mut sampling)?;

    generate_one_prepared_response(
        state,
        &req,
        request_start,
        &sampling,
        request_adapter,
        chat_request_cache_key,
        chat_request_cache_owner,
        &prompt_text,
        &prompt_tokens,
    )
    .await
}

async fn generate_one_prepared_prompt_response(
    state: &AppState,
    mut req: ChatCompletionRequest,
    request_adapter: Option<LoadedAdapterIdentity>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
) -> Result<ChatCompletionResponse, ApiError> {
    let request_start = std::time::Instant::now();
    apply_eval_mode_chat_defaults(state, &mut req);
    let mut sampling = sampling_params_for_chat_request(&req);

    let chat_request_cache_key = if effective_thinking_budget_for_request(state, &req)
        .max_time_ms
        .is_some()
    {
        None
    } else {
        deterministic_chat_request_cache_key_with_vocab_size_and_fold(
            &req,
            &sampling,
            state.model_config.vocab_size,
            fold_reasoning_into_content_for_request(state, &req),
            effective_thinking_budget_for_request(state, &req).max_tokens,
        )?
        .map(|request| state.deterministic_cache_key(request_adapter.clone(), request))
    };
    let mut chat_request_cache_owner = None;
    if let Some(key) = chat_request_cache_key.as_ref() {
        let claim = state.chat_request_cache.lock().unwrap().claim(key);
        match claim {
            DeterministicChatRequestCacheClaim::Hit(cached) => {
                return Ok(response_from_cached_chat_request(
                    state,
                    &req,
                    request_start,
                    cached,
                ));
            }
            DeterministicChatRequestCacheClaim::Wait(receiver) => {
                if let Some(cached) = wait_for_deterministic_chat_request(receiver).await {
                    return Ok(response_from_cached_chat_request(
                        state,
                        &req,
                        request_start,
                        cached,
                    ));
                }
            }
            DeterministicChatRequestCacheClaim::Owner(claim_id) => {
                chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                    state.chat_request_cache.clone(),
                    key.clone(),
                    claim_id,
                ));
            }
        }
    }

    enforce_context_window(state, &mut sampling, prompt_tokens.len())?;
    configure_thinking_budget_for_prompt(state, &req, prompt_text, &mut sampling)?;

    generate_one_prepared_response(
        state,
        &req,
        request_start,
        &sampling,
        request_adapter,
        chat_request_cache_key,
        chat_request_cache_owner,
        prompt_text,
        prompt_tokens,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn generate_one_prepared_response(
    state: &AppState,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    sampling: &SamplingParams,
    request_adapter: Option<LoadedAdapterIdentity>,
    chat_request_cache_key: Option<DeterministicCacheKey>,
    mut chat_request_cache_owner: Option<ChatRequestCacheOwnerGuard>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
) -> Result<ChatCompletionResponse, ApiError> {
    let completion_cache_key = deterministic_completion_cache_key_for_adapter(
        state,
        request_adapter.clone(),
        prompt_tokens,
        sampling,
        fold_reasoning_into_content_for_request(state, req),
    );
    let mut completion_cache_owner = None;
    if let Some(key) = completion_cache_key.as_ref() {
        let claim = state.completion_cache.lock().unwrap().claim(key);
        match claim {
            DeterministicCompletionCacheClaim::Hit(cached) => {
                let resp = response_from_cached_completion(
                    state,
                    &req,
                    prompt_tokens.len(),
                    request_start,
                    cached,
                );
                finish_chat_request_cache(
                    state,
                    chat_request_cache_key.clone(),
                    chat_request_cache_owner.take(),
                    &resp,
                );
                return Ok(resp);
            }
            DeterministicCompletionCacheClaim::Wait(receiver) => {
                if let Some(cached) = wait_for_deterministic_completion(receiver).await {
                    let resp = response_from_cached_completion(
                        state,
                        &req,
                        prompt_tokens.len(),
                        request_start,
                        cached,
                    );
                    finish_chat_request_cache(
                        state,
                        chat_request_cache_key.clone(),
                        chat_request_cache_owner.take(),
                        &resp,
                    );
                    return Ok(resp);
                }
            }
            DeterministicCompletionCacheClaim::Owner(claim_id) => {
                completion_cache_owner = Some(claim_id);
            }
        }
    }

    match state.backend.as_ref() {
        ModelBackend::Real {
            batching_engine, ..
        } => {
            let generation = generate_real_batched(
                state,
                batching_engine,
                request_adapter.clone(),
                prompt_text,
                prompt_tokens,
                sampling,
                req,
                request_start,
                None,
            )
            .await;
            let resp = match generation {
                Ok(resp) => resp,
                Err(err) => {
                    let model = req
                        .model
                        .clone()
                        .unwrap_or_else(|| state.served_model_id.clone());
                    record_failed_chat_completion(
                        state,
                        req,
                        &model,
                        prompt_text,
                        request_start,
                        prompt_tokens.len(),
                        &err,
                    );
                    if let (Some(claim_id), Some(key)) =
                        (completion_cache_owner, completion_cache_key.as_ref())
                    {
                        fail_deterministic_completion_owner(state, key, claim_id);
                    }
                    return Err(err);
                }
            };
            if let Some(key) = completion_cache_key.clone() {
                if let Some(claim_id) = completion_cache_owner {
                    complete_deterministic_completion_owner(state, key, claim_id, &resp);
                } else {
                    store_deterministic_completion(state, key, &resp);
                }
            }
            state
                .metrics
                .add_tokens(resp.usage.completion_tokens as u64);
            finish_chat_request_cache(
                state,
                chat_request_cache_key.clone(),
                chat_request_cache_owner.take(),
                &resp,
            );
            Ok(resp)
        }
        ModelBackend::Mock { scheduler, engine } => {
            let generation = generate_mock(
                state,
                scheduler,
                engine,
                prompt_text,
                prompt_tokens,
                sampling,
                req,
                request_start,
            )
            .await;
            let resp = match generation {
                Ok(resp) => resp,
                Err(err) => {
                    let model = req
                        .model
                        .clone()
                        .unwrap_or_else(|| state.served_model_id.clone());
                    record_failed_chat_completion(
                        state,
                        req,
                        &model,
                        prompt_text,
                        request_start,
                        prompt_tokens.len(),
                        &err,
                    );
                    if let (Some(claim_id), Some(key)) =
                        (completion_cache_owner, completion_cache_key.as_ref())
                    {
                        fail_deterministic_completion_owner(state, key, claim_id);
                    }
                    return Err(err);
                }
            };
            if let Some(key) = completion_cache_key.clone() {
                if let Some(claim_id) = completion_cache_owner {
                    complete_deterministic_completion_owner(state, key, claim_id, &resp);
                } else {
                    store_deterministic_completion(state, key, &resp);
                }
            }
            state
                .metrics
                .add_tokens(resp.usage.completion_tokens as u64);
            finish_chat_request_cache(
                state,
                chat_request_cache_key.clone(),
                chat_request_cache_owner.take(),
                &resp,
            );
            Ok(resp)
        }
    }
}

/// Body-size cap for /v1/chat/completions (audit LOW §1).
/// 8 MiB is generous for chat payloads while bounding memory DoS via JSON extraction.
const CHAT_BODY_LIMIT: usize = 8 * 1024 * 1024;
/// Body-size cap for /v1/completions (vLLM-compatible prompt-logprobs).
const COMPLETIONS_BODY_LIMIT: usize = 8 * 1024 * 1024;
/// Body-size cap for /v1/completions/batch (audit LOW §1).
/// 8 MiB accommodates batched prompts; per-request adapter composition is separately capped at 16.
const BATCH_BODY_LIMIT: usize = 8 * 1024 * 1024;

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(DefaultBodyLimit::max(CHAT_BODY_LIMIT)),
        )
        .route(
            "/v1/completions",
            post(completions).layer(DefaultBodyLimit::max(COMPLETIONS_BODY_LIMIT)),
        )
        .route(
            "/v1/completions/batch",
            post(batch_completions).layer(DefaultBodyLimit::max(BATCH_BODY_LIMIT)),
        )
}

#[cfg(test)]
mod tests;
