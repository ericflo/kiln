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
    ModelBackend, gpu_coordination_write_guard_while_healthy_async,
};

/// Snapshot a default adapter that is already the exact revision published by
/// the runner. `None` means a transient per-request override is loaded, so a
/// default-request cache lookup must wait until adapter resolution completes.
fn stable_default_adapter_identity(state: &AppState) -> Option<Option<LoadedAdapterIdentity>> {
    let default_name = state.active_adapter_name.read().unwrap().clone();
    let loaded = state.loaded_adapter_identity();
    match (default_name.as_deref(), loaded) {
        (None, None) => Some(None),
        (Some(name), Some(identity)) if identity.name == name => Some(Some(identity)),
        _ => None,
    }
}
use crate::teacher_identity::{
    MAX_COMPLETION_PROMPT_LOGPROB_CANDIDATES, MAX_COMPLETION_PROMPT_LOGPROBS,
    MAX_COMPLETION_PROMPT_TOKENS, MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS,
    PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET,
};

/// Max characters retained in the prompt preview for the recent-requests panel.
const PROMPT_PREVIEW_MAX_CHARS: usize = 120;
/// Max characters retained in the completion preview for the recent-requests panel.
const COMPLETION_PREVIEW_MAX_CHARS: usize = 200;
const QWEN_TOOL_CALL_OPEN_TAG: &str = "<tool_call>";
const QWEN_TOOL_CALL_CLOSE_TAG: &str = "</tool_call>";
const MOCK_COMPLETION_TOKEN_LIMIT: usize = 20;

fn observe_post_prefill_vram(
    memory_budget: &std::sync::Arc<crate::state::GpuMemoryBudget>,
    selector: kiln_memory::VramProbeSelector,
) {
    let observation = CachedMemoryGovernorObservation::capture_global_for(selector);
    if observation.sample_status.healthy
        && observation.snapshot.total_bytes > 0
        && !observation.snapshot.observations.probe_failed
    {
        memory_budget.observe_prefill_used_vram_bytes(observation.snapshot.used_bytes);
    }
}

fn ensure_backend_admission(state: &AppState) -> Result<(), ApiError> {
    state
        .ensure_backend_healthy()
        .map_err(ApiError::backend_quarantined)?;
    state
        .ensure_inference_admission_allowed()
        .map_err(|_| ApiError::inference_disabled_by_profile(state.serving_profile.profile()))
}

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

fn thinking_mode_for_request(req: &ChatCompletionRequest) -> &'static str {
    match req
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
    {
        Some(serde_json::Value::Bool(true)) => "explicit_enabled",
        Some(serde_json::Value::Bool(false)) => "explicit_disabled",
        Some(_) => "custom",
        None => "template_default",
    }
}

fn request_thinking_enabled(req: &ChatCompletionRequest) -> Option<bool> {
    req.chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(|value| value.as_bool())
}

fn effective_thinking_enabled_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> Option<bool> {
    request_thinking_enabled(req).or(state.default_thinking_enabled)
}

fn effective_thinking_budget_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> EffectiveThinkingBudget {
    resolve_effective_thinking_budget(state, req.thinking_budget_tokens, req.thinking_budget_ms)
}

fn resolve_effective_thinking_budget(
    state: &AppState,
    tokens: BudgetOverride<usize>,
    ms: BudgetOverride<u64>,
) -> EffectiveThinkingBudget {
    EffectiveThinkingBudget::resolve(
        ThinkingBudgetOverrides {
            tokens,
            time_ms: ms,
        },
        ThinkingBudgetDefaults {
            tokens: state.default_thinking_budget_tokens,
            time_ms: state.default_thinking_budget_ms,
        },
        ThinkingBudgetScope::Request,
    )
}

fn thinking_budget_metadata_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
    starts_in_reasoning: bool,
) -> ThinkingBudgetMetadata {
    let effective = effective_thinking_budget_for_request(state, req);
    let applied = effective.configured() && starts_in_reasoning && chat_request_max_tokens(req) > 0;
    ThinkingBudgetRecord::from_effective(effective, applied)
}

fn attach_thinking_budget_outcome(
    status: Option<ThinkingBudgetStatus>,
    response: &mut ChatCompletionResponse,
) {
    let Some(status) = status else {
        return;
    };
    apply_thinking_budget_status_to_metadata(&mut response.metadata.thinking_budget, status);
    if let Some(choice) = response.choices.first_mut() {
        choice.thinking_budget = Some(status);
    }
}

fn apply_thinking_budget_status_to_metadata(
    metadata: &mut ThinkingBudgetMetadata,
    status: ThinkingBudgetStatus,
) {
    metadata.set_outcome(status.into());
}

fn unresolved_request_thinking_budget(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> RequestThinkingBudget {
    let configuration = ThinkingBudgetConfigurationMetadata::from(
        effective_thinking_budget_for_request(state, req),
    );
    RequestThinkingBudget {
        configured: configuration.configured,
        max_tokens: configuration
            .max_tokens
            .map(|value| u64::try_from(value).unwrap_or(u64::MAX)),
        max_time_ms: configuration.max_time_ms,
        tokens_source: configuration.tokens_source,
        time_source: configuration.time_source,
        applied: (!configuration.configured).then_some(false),
        triggered: None,
        trigger: None,
        closed: None,
        thinking_tokens: None,
        thinking_time_ms: None,
    }
}

fn recent_thinking_budget_from_metadata(
    metadata: &ThinkingBudgetMetadata,
) -> RequestThinkingBudget {
    let outcome = metadata.outcome;
    RequestThinkingBudget {
        configured: metadata.configured,
        max_tokens: metadata
            .max_tokens
            .map(|value| u64::try_from(value).unwrap_or(u64::MAX)),
        max_time_ms: metadata.max_time_ms,
        tokens_source: metadata.tokens_source,
        time_source: metadata.time_source,
        applied: Some(metadata.applied),
        triggered: outcome.map(|outcome| outcome.triggered),
        trigger: outcome
            .and_then(|outcome| outcome.trigger)
            .map(|trigger| trigger.as_str().to_string()),
        closed: outcome.map(|outcome| outcome.closed),
        thinking_tokens: outcome
            .map(|outcome| outcome.thinking_tokens)
            .map(|value| u64::try_from(value).unwrap_or(u64::MAX)),
        thinking_time_ms: outcome.map(|outcome| outcome.thinking_time_ms),
    }
}

fn recent_thinking_budget_with_status(
    metadata: &ThinkingBudgetMetadata,
    status: Option<ThinkingBudgetStatus>,
) -> RequestThinkingBudget {
    let mut metadata = metadata.clone();
    if let Some(status) = status {
        metadata.applied = true;
        apply_thinking_budget_status_to_metadata(&mut metadata, status);
    }
    recent_thinking_budget_from_metadata(&metadata)
}

fn attach_cached_thinking_budget_outcome(response: &mut ChatCompletionResponse) {
    if let Some(status) = response
        .choices
        .first()
        .and_then(|choice| choice.thinking_budget)
    {
        response.metadata.thinking_enabled = true;
        response.metadata.thinking_mode = "reasoning".to_string();
        response.metadata.thinking_budget.applied = true;
        apply_thinking_budget_status_to_metadata(&mut response.metadata.thinking_budget, status);
    }
}

fn configure_thinking_budget_for_prompt(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_text: &str,
    sampling: &mut SamplingParams,
) -> Result<(), ApiError> {
    if !prompt_starts_in_reasoning(prompt_text) || sampling.max_tokens == 0 {
        sampling.thinking_budget = None;
        return Ok(());
    }
    let effective = effective_thinking_budget_for_request(state, req);
    if !effective.configured() {
        sampling.thinking_budget = None;
        return Ok(());
    }

    if let Some(stop) = sampling
        .stop
        .iter()
        .find(|stop| stop_sequence_conflicts_with_thinking_close(stop))
    {
        return Err(ApiError::chat_invalid_request(format!(
            "stop sequence {stop:?} conflicts with the forced {REASONING_CLOSE_TAG:?} thinking close sequence"
        )));
    }

    let close_token_ids = state
        .tokenizer
        .encode(REASONING_CLOSE_TAG)
        .map_err(ApiError::tokenization_failed)?;
    let decoded_close = state
        .tokenizer
        .decode(&close_token_ids)
        .map_err(ApiError::tokenization_failed)?;
    if decoded_close != REASONING_CLOSE_TAG {
        return Err(ApiError::chat_invalid_request(format!(
            "the active tokenizer cannot reproduce {REASONING_CLOSE_TAG:?} as a forced token sequence"
        )));
    }
    validate_thinking_budget_completion_capacity(sampling.max_tokens, close_token_ids.len())?;
    let max_completion_tokens = match state.backend.as_ref() {
        ModelBackend::Mock { .. } => sampling.max_tokens.min(MOCK_COMPLETION_TOKEN_LIMIT),
        ModelBackend::Real { .. } => sampling.max_tokens,
    };
    sampling.thinking_budget = Some(
        ThinkingBudget::new(
            effective.max_tokens,
            effective.max_time_ms.map(std::time::Duration::from_millis),
            max_completion_tokens,
            close_token_ids,
        )
        .map_err(|error| ApiError::chat_invalid_request(error.to_string()))?,
    );
    Ok(())
}

pub(crate) fn stop_sequence_conflicts_with_thinking_close(stop: &str) -> bool {
    if stop.is_empty() || REASONING_CLOSE_TAG.contains(stop) || stop.contains(REASONING_CLOSE_TAG) {
        return true;
    }

    let stop = stop.as_bytes();
    let close = REASONING_CLOSE_TAG.as_bytes();
    (1..=stop.len().min(close.len()))
        .any(|len| stop.ends_with(&close[..len]) || close.ends_with(&stop[..len]))
}

fn validate_thinking_budget_completion_capacity(
    effective_max_tokens: usize,
    close_token_count: usize,
) -> Result<(), ApiError> {
    if close_token_count > effective_max_tokens {
        return Err(ApiError::chat_invalid_request(format!(
            "effective max_tokens {effective_max_tokens} cannot fit the active tokenizer's {close_token_count}-token {REASONING_CLOSE_TAG:?} thinking close sequence"
        )));
    }
    Ok(())
}

fn fold_reasoning_into_content_for_request(state: &AppState, req: &ChatCompletionRequest) -> bool {
    req.fold_reasoning_into_content
        .unwrap_or(state.fold_reasoning_into_content)
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn thinking_source_for_request(state: &AppState, req: &ChatCompletionRequest) -> &'static str {
    match req
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
    {
        Some(serde_json::Value::Bool(_)) => "request",
        Some(_) => "custom",
        None if state.default_thinking_enabled.is_some() => "server_default",
        None => "template_default",
    }
}

fn thinking_mode_for_prompt(prompt_text: &str) -> &'static str {
    if prompt_starts_in_reasoning(prompt_text) {
        "reasoning"
    } else {
        "non_reasoning"
    }
}

fn effective_chat_template_kwargs(
    default_thinking_enabled: Option<bool>,
    chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut kwargs = chat_template_kwargs.cloned().unwrap_or_default();
    if let Some(enabled) = default_thinking_enabled {
        kwargs
            .entry("enable_thinking".to_string())
            .or_insert(serde_json::Value::Bool(enabled));
    }
    kwargs
}

fn chat_completion_metadata_from_prompt(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_text: &str,
) -> ChatCompletionMetadata {
    ChatCompletionMetadata {
        thinking_enabled: prompt_starts_in_reasoning(prompt_text),
        thinking_mode: thinking_mode_for_prompt(prompt_text).to_string(),
        thinking_source: thinking_source_for_request(state, req),
        default_thinking_enabled: state.default_thinking_enabled,
        final_content_empty: false,
        content_empty_reason: None,
        reasoning_folded_into_content: false,
        thinking_budget: thinking_budget_metadata_for_request(
            state,
            req,
            prompt_starts_in_reasoning(prompt_text),
        ),
        config_hashes: None,
        performance: None,
    }
}

fn chat_completion_metadata_from_prompt_and_output(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_text: &str,
    output: &AssistantOutputParts,
) -> ChatCompletionMetadata {
    let fold_reasoning = fold_reasoning_into_content_for_request(state, req);
    ChatCompletionMetadata {
        thinking_enabled: prompt_starts_in_reasoning(prompt_text),
        thinking_mode: thinking_mode_for_prompt(prompt_text).to_string(),
        thinking_source: thinking_source_for_request(state, req),
        default_thinking_enabled: state.default_thinking_enabled,
        final_content_empty: output.content.is_empty(),
        content_empty_reason: content_empty_reason(output),
        reasoning_folded_into_content: fold_reasoning
            && output
                .reasoning_content
                .as_deref()
                .is_some_and(|text| !text.is_empty()),
        thinking_budget: thinking_budget_metadata_for_request(
            state,
            req,
            prompt_starts_in_reasoning(prompt_text),
        ),
        config_hashes: None,
        performance: None,
    }
}

fn chat_completion_metadata_from_cached_output(
    state: &AppState,
    req: &ChatCompletionRequest,
    cached_output: &AssistantOutputParts,
) -> ChatCompletionMetadata {
    let thinking_enabled =
        effective_thinking_enabled_for_request(state, req).unwrap_or_else(|| {
            cached_output
                .reasoning_content
                .as_deref()
                .is_some_and(|text| !text.is_empty())
        });
    ChatCompletionMetadata {
        thinking_enabled,
        thinking_mode: if thinking_enabled {
            "reasoning".to_string()
        } else {
            "non_reasoning".to_string()
        },
        thinking_source: thinking_source_for_request(state, req),
        default_thinking_enabled: state.default_thinking_enabled,
        final_content_empty: cached_output.content.is_empty(),
        content_empty_reason: content_empty_reason(cached_output),
        reasoning_folded_into_content: fold_reasoning_into_content_for_request(state, req)
            && cached_output
                .reasoning_content
                .as_deref()
                .is_some_and(|text| !text.is_empty()),
        thinking_budget: thinking_budget_metadata_for_request(state, req, thinking_enabled),
        config_hashes: None,
        performance: None,
    }
}

fn chat_completion_metadata_from_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> ChatCompletionMetadata {
    let thinking_enabled = effective_thinking_enabled_for_request(state, req).unwrap_or(false);
    ChatCompletionMetadata {
        thinking_enabled,
        thinking_mode: if thinking_enabled {
            "reasoning".to_string()
        } else {
            "non_reasoning".to_string()
        },
        thinking_source: thinking_source_for_request(state, req),
        default_thinking_enabled: state.default_thinking_enabled,
        final_content_empty: false,
        content_empty_reason: None,
        reasoning_folded_into_content: false,
        thinking_budget: thinking_budget_metadata_for_request(state, req, thinking_enabled),
        config_hashes: None,
        performance: None,
    }
}

fn chat_performance_metadata_enabled(state: &AppState, req: &ChatCompletionRequest) -> bool {
    req.include_performance
        .unwrap_or(state.chat_performance_metadata)
}

fn chat_config_hash_metadata_enabled(state: &AppState, req: &ChatCompletionRequest) -> bool {
    req.include_config_hashes
        .unwrap_or(state.chat_config_hash_metadata)
}

fn duration_ms_f64(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn duration_ms_u64(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn adapter_used_for_performance_metadata(state: &AppState) -> String {
    state
        .loaded_adapter_name()
        .unwrap_or_else(|| "base".to_string())
}

fn decode_tokens_per_sec_for_performance_metadata(
    completion_tokens: usize,
    total_latency: std::time::Duration,
    ttft: Option<std::time::Duration>,
    decode_duration: Option<std::time::Duration>,
) -> Option<f64> {
    if completion_tokens == 0 {
        return Some(0.0);
    }

    let decode_secs = decode_duration
        .filter(|duration| !duration.is_zero())
        .map(|duration| duration.as_secs_f64())
        .or_else(|| {
            ttft.and_then(|ttft| {
                total_latency
                    .checked_sub(ttft)
                    .filter(|duration| !duration.is_zero())
                    .map(|duration| duration.as_secs_f64())
            })
        })
        .or_else(|| (!total_latency.is_zero()).then_some(total_latency.as_secs_f64()))?;

    Some(completion_tokens as f64 / decode_secs)
}

fn attach_chat_performance_metadata(
    state: &AppState,
    req: &ChatCompletionRequest,
    resp: &mut ChatCompletionResponse,
    request_start: std::time::Instant,
    ttft: Option<std::time::Duration>,
    prefill_duration: Option<std::time::Duration>,
    decode_duration: Option<std::time::Duration>,
) {
    if chat_config_hash_metadata_enabled(state, req) {
        resp.metadata.config_hashes = Some(state.config_hashes.clone());
    }

    if !chat_performance_metadata_enabled(state, req) {
        return;
    }

    let total_latency = request_start.elapsed();
    let finish_reason = resp
        .choices
        .first()
        .map(|choice| choice.finish_reason.clone())
        .unwrap_or_else(|| "unknown".to_string());
    resp.metadata.performance = Some(ChatCompletionPerformanceMetadata {
        prompt_tokens: resp.usage.prompt_tokens,
        completion_tokens: resp.usage.completion_tokens,
        ttft_ms: ttft.map(duration_ms_f64),
        prefill_ms: prefill_duration.map(duration_ms_f64),
        actor_queue_ms: None,
        actor_admission_ms: None,
        actor_prefill_wall_ms: None,
        resident_prefill_used: None,
        decode_ms: decode_duration.map(duration_ms_f64),
        total_latency_ms: duration_ms_f64(total_latency),
        decode_tokens_per_sec: decode_tokens_per_sec_for_performance_metadata(
            resp.usage.completion_tokens,
            total_latency,
            ttft,
            decode_duration,
        ),
        adapter_used: adapter_used_for_performance_metadata(state),
        thinking_mode: resp.metadata.thinking_mode.clone(),
        finish_reason,
        latency: None,
    });
}

fn attach_batched_actor_performance_metadata(
    resp: &mut ChatCompletionResponse,
    actor_queue_duration: std::time::Duration,
    actor_admission_duration: std::time::Duration,
    actor_prefill_wall_duration: Option<std::time::Duration>,
    resident_prefill_used: bool,
) {
    let Some(performance) = resp.metadata.performance.as_mut() else {
        return;
    };
    performance.actor_queue_ms = Some(duration_ms_f64(actor_queue_duration));
    performance.actor_admission_ms = Some(duration_ms_f64(actor_admission_duration));
    performance.actor_prefill_wall_ms = actor_prefill_wall_duration.map(duration_ms_f64);
    performance.resident_prefill_used = Some(resident_prefill_used);
}

fn content_empty_reason(output: &AssistantOutputParts) -> Option<&'static str> {
    if !output.content.is_empty() {
        return None;
    }
    if output
        .reasoning_content
        .as_deref()
        .is_some_and(|text| !text.is_empty())
    {
        return Some("reasoning_without_final_content");
    }
    if output
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        return Some("tool_call");
    }
    Some("no_content")
}

fn ensure_eval_mode_thinking_default(
    chat_template_kwargs: &mut Option<serde_json::Map<String, serde_json::Value>>,
    enabled: bool,
) {
    let kwargs = chat_template_kwargs.get_or_insert_with(serde_json::Map::new);
    kwargs
        .entry("enable_thinking".to_string())
        .or_insert(serde_json::Value::Bool(enabled));
}

fn apply_eval_mode_chat_defaults(state: &AppState, req: &mut ChatCompletionRequest) {
    if !state.eval_mode {
        return;
    }

    if req.temperature.is_none() {
        req.temperature = Some(0.0);
    }
    if req.top_p.is_none() {
        req.top_p = Some(1.0);
    }
    if req.top_k.is_none() {
        req.top_k = Some(0);
    }
    if req.min_p.is_none() {
        req.min_p = Some(0.0);
    }
    if req.presence_penalty.is_none() {
        req.presence_penalty = Some(0.0);
    }
    if req.frequency_penalty.is_none() {
        req.frequency_penalty = Some(0.0);
    }
    if req.repetition_penalty.is_none() {
        req.repetition_penalty = Some(1.0);
    }
    if req.seed.is_none() {
        req.seed = Some(0);
    }
    ensure_eval_mode_thinking_default(
        &mut req.chat_template_kwargs,
        state
            .model_defaults_profile
            .eval_mode_default_thinking_enabled,
    );
}

fn apply_eval_mode_batch_defaults(state: &AppState, req: &mut BatchCompletionRequest) {
    if !state.eval_mode {
        return;
    }

    if req.temperature.is_none() {
        req.temperature = Some(0.0);
    }
    if req.top_p.is_none() {
        req.top_p = Some(1.0);
    }
    if req.top_k.is_none() {
        req.top_k = Some(0);
    }
    if req.min_p.is_none() {
        req.min_p = Some(0.0);
    }
    if req.presence_penalty.is_none() {
        req.presence_penalty = Some(0.0);
    }
    if req.frequency_penalty.is_none() {
        req.frequency_penalty = Some(0.0);
    }
    if req.repetition_penalty.is_none() {
        req.repetition_penalty = Some(1.0);
    }
    if req.seed.is_none() {
        req.seed = Some(0);
    }
    ensure_eval_mode_thinking_default(
        &mut req.chat_template_kwargs,
        state
            .model_defaults_profile
            .eval_mode_default_thinking_enabled,
    );
}

/// Build a [`RequestRecord`] pre-populated with everything we know from the
/// request side: id, model, adapter, sampling knobs, prompt preview + full
/// body. Each call site overrides the response-side fields (completion text,
/// token counts, duration, finish reason, optional ttft / error).
fn request_record_from_req(
    state: &AppState,
    req: &ChatCompletionRequest,
    id: &str,
    model: &str,
    streamed: bool,
) -> RequestRecord {
    let prompt = last_user_message_text(req);
    RequestRecord {
        id: id.to_owned(),
        timestamp_unix_ms: now_unix_ms(),
        model: model.to_owned(),
        prompt_preview: truncate_chars(&prompt, PROMPT_PREVIEW_MAX_CHARS),
        prompt_full: Some(truncate_chars(&prompt, FULL_BODY_MAX_CHARS)),
        streamed,
        adapter: req.adapter.request_adapter_name(),
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: Some(chat_request_max_tokens(req).min(u32::MAX as usize) as u32),
        thinking_mode: Some(thinking_mode_for_request(req).to_string()),
        prefix_cache: Some("unknown".to_string()),
        user_agent: req.user_agent.clone(),
        client: req.client.clone(),
        thinking_budget: Some(unresolved_request_thinking_budget(state, req)),
        ..RequestRecord::default()
    }
}

/// Extract the calling client's `User-Agent` for per-agent attribution on the
/// /ui dashboard. Bounded so a hostile header can't bloat the ring.
fn extract_user_agent(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(200).collect())
}

/// Extract the `X-Kiln-Client` self-identification header. The /ui dashboard
/// sends `dashboard` on its own inference traffic so the journey strip /
/// Connect panel can tell it apart from a real agent. Bounded like
/// `extract_user_agent` so a hostile header can't bloat the ring.
fn extract_client(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-kiln-client")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(64).collect())
}

const REASONING_OPEN_TAG: &str = "<think>\n";
const REASONING_CLOSE_TAG: &str = "</think>";

/// True when the rendered chat-template prompt prefilled the opening reasoning
/// tag into the assistant turn (Qwen3.5's official template ends with
/// `<|im_start|>assistant\n<think>\n` whenever `enable_thinking` isn't
/// explicitly false). The model continues directly with chain-of-thought
/// content, never re-emitting the opening tag, then closes with `</think>`
/// before the actual answer. Used to initialize the reasoning splitter into
/// the "currently inside <think>...</think>" state on the very first token.
///
/// Conservative: only fires when the prompt ends with the exact `<think>\n`
/// suffix Qwen3.5 emits, so the bare ChatML fallback and any
/// non-reasoning template stay on the OpenAI-shaped pure-`content` path.
fn prompt_starts_in_reasoning(prompt_text: &str) -> bool {
    prompt_text.trim_end().ends_with("<think>")
}

/// Per-stream parser that splits incremental decode-token text into
/// `reasoning_content` and `content` deltas across a `</think>` boundary.
/// Mirrors the wire format llama.cpp's `--reasoning-format=deepseek` ships:
/// each chunk's `delta` carries at most one of `{reasoning_content, content}`
/// depending on which side of the close tag the new token landed on.
///
/// The close tag can straddle multiple decode-token boundaries (BPE tokenizers
/// regularly split it across three or more pieces — e.g. `</`, `think`, `>`),
/// so we buffer up to `len("</think>") - 1` characters of "could-be tag tail"
/// in `pending` and only flush them once we've seen enough to disambiguate.
/// `flush` drains the tail when generation finishes (EOS, max_tokens, stop
/// sequence, client disconnect) so no characters are silently swallowed.
struct ReasoningSplitter {
    in_reasoning: bool,
    pending: String,
}

#[derive(Default, Debug)]
struct ReasoningChunk {
    reasoning: Option<String>,
    content: Option<String>,
}

impl ReasoningChunk {
    fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }
}

impl ReasoningSplitter {
    fn new(starts_in_reasoning: bool) -> Self {
        Self {
            in_reasoning: starts_in_reasoning,
            pending: String::new(),
        }
    }

    fn push(&mut self, token: &str) -> ReasoningChunk {
        if !self.in_reasoning {
            // Already past `</think>`, everything streams as content.
            if token.is_empty() {
                return ReasoningChunk::default();
            }
            return ReasoningChunk {
                content: Some(token.to_string()),
                ..Default::default()
            };
        }

        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(token);

        if let Some(idx) = buf.find(REASONING_CLOSE_TAG) {
            let before = buf[..idx].to_string();
            let after = buf[idx + REASONING_CLOSE_TAG.len()..].to_string();
            self.in_reasoning = false;
            let mut out = ReasoningChunk::default();
            if !before.is_empty() {
                out.reasoning = Some(before);
            }
            if !after.is_empty() {
                out.content = Some(after);
            }
            return out;
        }

        // No full close tag — but the tail may be a partial prefix of one.
        // Keep the longest such suffix in `pending` so the next push can
        // complete the match instead of leaking a literal "</" or "</thi"
        // into reasoning_content.
        for k in (1..REASONING_CLOSE_TAG.len()).rev() {
            if buf.len() >= k && buf.ends_with(&REASONING_CLOSE_TAG[..k]) {
                let emit_len = buf.len() - k;
                self.pending = buf[emit_len..].to_string();
                if emit_len == 0 {
                    return ReasoningChunk::default();
                }
                return ReasoningChunk {
                    reasoning: Some(buf[..emit_len].to_string()),
                    ..Default::default()
                };
            }
        }

        ReasoningChunk {
            reasoning: Some(buf),
            ..Default::default()
        }
    }

    /// Drain whatever is buffered at end-of-stream — necessary when generation
    /// stops while we're still holding partial-tag bytes that turned out not
    /// to be a tag. Without this, those bytes vanish from the response.
    fn flush(&mut self) -> ReasoningChunk {
        if self.pending.is_empty() {
            return ReasoningChunk::default();
        }
        let buf = std::mem::take(&mut self.pending);
        if self.in_reasoning {
            ReasoningChunk {
                reasoning: Some(buf),
                ..Default::default()
            }
        } else {
            ReasoningChunk {
                content: Some(buf),
                ..Default::default()
            }
        }
    }
}

/// Send a [`ReasoningChunk`] over the SSE channel as one or two
/// [`ChatCompletionChunk`]s — one per populated channel — so each chunk's
/// `delta` only ever carries one of `content` or `reasoning_content` (the
/// llama.cpp shape; mixing both in the same delta confuses
/// content-aware UIs that switch panels on each delta key). Returns `false`
/// when the SSE receiver was dropped mid-send so the caller can record the
/// disconnect and stop.
async fn emit_reasoning_chunk(
    tx: &tokio::sync::mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    chunk: ReasoningChunk,
    completion_preview_buf: &mut String,
    reasoning_buf: &mut String,
    content_buf: &mut String,
) -> bool {
    if chunk.is_empty() {
        return true;
    }
    if let Some(text) = chunk.reasoning {
        // Reasoning content also feeds the dashboard preview when the
        // answer hasn't started yet — without this the preview is blank
        // until the model emits `</think>`, which can be hundreds of
        // tokens in.
        if completion_preview_buf.chars().count() < COMPLETION_PREVIEW_MAX_CHARS + 16 {
            completion_preview_buf.push_str(&text);
        }
        reasoning_buf.push_str(&text);
        let event = ChatCompletionChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: Some(text),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        if tx
            .send(Event::default().data(serde_json::to_string(&event).unwrap()))
            .await
            .is_err()
        {
            return false;
        }
    }
    if let Some(text) = chunk.content {
        if completion_preview_buf.chars().count() < COMPLETION_PREVIEW_MAX_CHARS + 16 {
            completion_preview_buf.push_str(&text);
        }
        content_buf.push_str(&text);
        let event = ChatCompletionChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(text),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        if tx
            .send(Event::default().data(serde_json::to_string(&event).unwrap()))
            .await
            .is_err()
        {
            return false;
        }
    }
    true
}

async fn emit_tool_calls_chunk(
    tx: &tokio::sync::mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    tool_calls: &[serde_json::Value],
) -> bool {
    let event = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(tool_call_deltas_from_openai_calls(tool_calls)),
            },
            finish_reason: None,
        }],
    };
    tx.send(Event::default().data(serde_json::to_string(&event).unwrap()))
        .await
        .is_ok()
}

async fn emit_content_chunk(
    tx: &tokio::sync::mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    content: String,
) -> bool {
    if content.is_empty() {
        return true;
    }
    let event = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(content),
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };
    tx.send(Event::default().data(serde_json::to_string(&event).unwrap()))
        .await
        .is_ok()
}

#[allow(clippy::too_many_arguments)]
/// Cap on the whitespace-run holdback preceding a possible tool tag.
const TOOL_TAG_WS_HOLDBACK_MAX: usize = 64;

/// Streams content EAGERLY on tools-bearing requests, holding back only
/// the longest tail that could still become `<tool_call>` (plus the
/// whitespace run before it, because the finish path `trim_end()`s the
/// pre-tag content — holding the whitespace means the wire never shows
/// bytes the final content retracts). Flips to full buffering once the
/// tag confirms.
///
/// Before this gate, `buffer_tool_content` withheld the ENTIRE content
/// channel until Done on every tools-bearing request — and pi always
/// sends tools, so the daily driver never saw a token stream.
struct ToolCallGate {
    enabled: bool,
    confirmed: bool,
    /// Bytes of `content_buf` already emitted on the wire.
    streamed: usize,
}

impl ToolCallGate {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            confirmed: false,
            streamed: 0,
        }
    }

    /// Call after appending new text to `content_buf`. Returns the byte
    /// range of `content_buf` now safe to stream (empty while holding
    /// back or buffering a confirmed tag).
    fn advance(&mut self, content_buf: &str) -> std::ops::Range<usize> {
        if !self.enabled {
            let r = self.streamed..content_buf.len();
            self.streamed = content_buf.len();
            return r;
        }
        if self.confirmed {
            return self.streamed..self.streamed;
        }
        let tail = &content_buf[self.streamed..];
        if let Some(i) = tail.find(QWEN_TOOL_CALL_OPEN_TAG) {
            self.confirmed = true;
            // The finish path trims trailing whitespace off the pre-tag
            // content; never emit bytes the final content would retract.
            let emit_end = self.streamed + tail[..i].trim_end().len();
            let r = self.streamed..emit_end;
            self.streamed = emit_end;
            return r;
        }
        // Longest suffix that is a proper prefix of the open tag…
        let mut hold = 0usize;
        for k in (1..QWEN_TOOL_CALL_OPEN_TAG.len()).rev() {
            if k <= tail.len() && tail.ends_with(&QWEN_TOOL_CALL_OPEN_TAG[..k]) {
                hold = k;
                break;
            }
        }
        // …extended left over the whitespace run before it (capped).
        let mut idx = tail.len() - hold;
        let mut ws = 0usize;
        while idx > 0 && ws < TOOL_TAG_WS_HOLDBACK_MAX {
            let prev = tail[..idx].chars().next_back().unwrap_or('x');
            if matches!(prev, ' ' | '\t' | '\r' | '\n') {
                idx -= prev.len_utf8();
                ws += 1;
            } else {
                break;
            }
        }
        let emit_end = self.streamed + idx;
        let r = self.streamed..emit_end;
        self.streamed = emit_end;
        r
    }

    fn confirmed(&self) -> bool {
        self.confirmed
    }

    /// Unstreamed remainder for end-of-stream emission.
    fn unsent<'a>(&self, content_buf: &'a str) -> &'a str {
        &content_buf[self.streamed.min(content_buf.len())..]
    }

    fn mark_all_sent(&mut self, content_buf: &str) {
        self.streamed = content_buf.len();
    }
}

/// The OpenAI `stream_options.include_usage` final chunk: empty
/// `choices`, populated `usage`, emitted after the finish chunk and
/// before `[DONE]`.
fn usage_chunk_json(
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
    .to_string()
}

fn finalized_thinking_budget_status(
    budget: Option<&ThinkingBudget>,
    completion_tokens: usize,
) -> Option<ThinkingBudgetStatus> {
    let mut status = budget.map(ThinkingBudget::status)?;
    if !status.closed && status.trigger.is_none() {
        status.thinking_tokens = completion_tokens;
    }
    Some(status)
}

fn streaming_finish_chunk_json(
    chunk: &ChatCompletionChunk,
    thinking_budget: &ThinkingBudgetMetadata,
    performance: Option<&ChatCompletionPerformanceMetadata>,
) -> String {
    let mut value = serde_json::to_value(chunk).expect("chat completion chunk must serialize");
    if let Some(object) = value.as_object_mut() {
        let mut metadata = serde_json::json!({"thinking_budget": thinking_budget});
        if let (Some(performance), Some(metadata)) = (performance, metadata.as_object_mut()) {
            metadata.insert(
                "performance".to_string(),
                serde_json::to_value(performance)
                    .expect("chat performance metadata must serialize"),
            );
        }
        object.insert("metadata".to_string(), metadata);
    }
    serde_json::to_string(&value).expect("chat completion chunk JSON must serialize")
}

#[derive(Debug, Serialize)]
struct StreamingTokenTiming {
    object: &'static str,
    source: &'static str,
    token_index: u32,
    token_id: u32,
    ready_ms: f64,
    producer_delivered_ms: f64,
    handler_received_ms: f64,
    body_enqueued_ms: f64,
    response_delivery_ms: f64,
    handler_queue_ms: f64,
    queue_delay_ms: f64,
    client_delivery_ms: f64,
    blocking_phase: Option<&'static str>,
    blocking_phase_ms: Option<f64>,
}

fn instant_delta_ms(start: std::time::Instant, end: std::time::Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1_000.0
}

fn streaming_token_timing_enabled(req: &ChatCompletionRequest) -> bool {
    req.include_performance == Some(true)
}

fn streaming_token_timing_json(
    enabled: bool,
    token_index: u32,
    token_id: u32,
    request_start: std::time::Instant,
    timing: EngineTokenTiming,
    handler_received_at: std::time::Instant,
    body_enqueued_at: std::time::Instant,
    gap: Option<crate::latency_observability::TokenGapObservation>,
) -> Option<String> {
    enabled.then(|| {
        let producer_delivered_at = timing.producer_delivered_at.unwrap_or(timing.ready_at);
        serde_json::to_string(&StreamingTokenTiming {
            object: "kiln.token_timing",
            source: "batching_engine",
            token_index,
            token_id,
            ready_ms: instant_delta_ms(request_start, timing.ready_at),
            producer_delivered_ms: instant_delta_ms(request_start, producer_delivered_at),
            handler_received_ms: instant_delta_ms(request_start, handler_received_at),
            body_enqueued_ms: instant_delta_ms(request_start, body_enqueued_at),
            response_delivery_ms: instant_delta_ms(timing.ready_at, producer_delivered_at),
            handler_queue_ms: instant_delta_ms(producer_delivered_at, handler_received_at),
            queue_delay_ms: instant_delta_ms(timing.ready_at, handler_received_at),
            client_delivery_ms: instant_delta_ms(handler_received_at, body_enqueued_at),
            blocking_phase: gap.map(|observation| observation.reason.as_str()),
            blocking_phase_ms: gap
                .map(|observation| duration_ms_f64(observation.attributed_duration)),
        })
        .expect("token timing payload must serialize")
    })
}

async fn emit_or_buffer_reasoning_chunk(
    tx: &tokio::sync::mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    chunk: ReasoningChunk,
    completion_preview_buf: &mut String,
    reasoning_buf: &mut String,
    content_buf: &mut String,
    tool_gate: &mut ToolCallGate,
) -> bool {
    if tx.is_closed() {
        return false;
    }

    let ReasoningChunk { reasoning, content } = chunk;
    if let Some(text) = reasoning {
        let reasoning_only = ReasoningChunk {
            reasoning: Some(text),
            content: None,
        };
        if !emit_reasoning_chunk(
            tx,
            id,
            created,
            model,
            reasoning_only,
            completion_preview_buf,
            reasoning_buf,
            content_buf,
        )
        .await
        {
            return false;
        }
    }
    if let Some(text) = content {
        if completion_preview_buf.chars().count() < COMPLETION_PREVIEW_MAX_CHARS + 16 {
            completion_preview_buf.push_str(&text);
        }
        content_buf.push_str(&text);
        // Eager content streaming: emit whatever the tool gate clears
        // (everything for tool-less requests; everything up to the
        // ws+`<tool_call>`-prefix holdback otherwise).
        let r = tool_gate.advance(content_buf);
        if !r.is_empty() {
            let delta = content_buf[r].to_string();
            if !emit_content_chunk(tx, id, created, model, delta).await {
                return false;
            }
        }
    }
    !tx.is_closed()
}

/// End-of-stream salvage for the timeout/error exits of the streaming
/// handlers. Drains the reasoning splitter through the emit-or-buffer
/// path, then — when tool-call buffering held generated content back
/// from the client — parses the buffered text and emits it as a
/// `tool_calls` delta (complete call) or a plain `content` delta
/// (partial text) ahead of the caller's finish chunk, so an aborted
/// stream never silently drops output the model already produced.
///
/// Returns the parsed parts so the caller can pick the finish reason
/// (`finish_reason` stays as passed unless a complete tool call parsed,
/// which flips it to `"tool_calls"`) and record the full salvaged
/// completion; returns `None` when no content was buffered. Sends are
/// best-effort — a disconnected client must not stop the caller from
/// recording the salvaged output.
const STREAM_TAIL_FLUSH_GRACE: std::time::Duration = std::time::Duration::from_millis(100);

#[allow(clippy::too_many_arguments)]
async fn flush_buffered_stream_tail(
    tx: &tokio::sync::mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    reasoning_splitter: &mut ReasoningSplitter,
    completion_preview_buf: &mut String,
    reasoning_buf: &mut String,
    content_buf: &mut String,
    tool_gate: &mut ToolCallGate,
    finish_reason: &str,
) -> Option<AssistantOutputParts> {
    let flush_deadline = tokio::time::Instant::now() + STREAM_TAIL_FLUSH_GRACE;
    let trailing = reasoning_splitter.flush();
    let _ = tokio::time::timeout_at(
        flush_deadline,
        emit_or_buffer_reasoning_chunk(
            tx,
            id,
            created,
            model,
            trailing,
            completion_preview_buf,
            reasoning_buf,
            content_buf,
            tool_gate,
        ),
    )
    .await;

    if !tool_gate.enabled || content_buf.is_empty() {
        return None;
    }

    let reasoning_content = if reasoning_buf.is_empty() {
        None
    } else {
        Some(reasoning_buf.clone())
    };
    let assistant_output = assistant_output_from_split_parts_with_tool_parsing(
        true,
        reasoning_content,
        content_buf.clone(),
        finish_reason,
    );
    if let Some(tool_calls) = assistant_output.tool_calls.as_deref() {
        // Pre-tag content already streamed eagerly (the gate's trim_end
        // holdback makes the wire equal the parsed content exactly) —
        // emit nothing extra, just the calls.
        let _ = tokio::time::timeout_at(
            flush_deadline,
            emit_tool_calls_chunk(tx, id, created, model, tool_calls),
        )
        .await;
    } else {
        // Malformed/unclosed tag (the gate confirmed and buffered) or a
        // pending holdback tail: the client receives the UNSENT suffix
        // exactly once — never a replay of eagerly-streamed bytes.
        let unsent = tool_gate.unsent(content_buf).to_string();
        if !unsent.is_empty() {
            let _ = tokio::time::timeout_at(
                flush_deadline,
                emit_content_chunk(tx, id, created, model, unsent),
            )
            .await;
        }
    }
    tool_gate.mark_all_sent(content_buf);
    Some(assistant_output)
}

/// Preserve the full salvaged completion for recent-request diagnostics.
/// Terminal classification stays owned by the caller, so a parsed tool call
/// cannot disguise an error or timeout as a successful `tool_calls` finish.
fn stream_tail_record_completion(
    tail: Option<AssistantOutputParts>,
    fallback_completion: &str,
) -> String {
    match tail {
        Some(parts) => parts.preview_source().to_string(),
        None => fallback_completion.to_string(),
    }
}

/// OpenAI semantics: a matched stop sequence terminates the content
/// BEFORE the stop text — the stop string itself must never reach the
/// client. Kiln's engine detects the match on the decoded text but kept
/// emitting the full buffer, so agent harnesses using stop markers
/// (`"Observation:"`, `"</tool_call>"`, …) received the marker glued to
/// the content and their parsers saw phantom delimiters.
fn truncate_at_matched_stop<'a>(
    text: &'a str,
    finish_reason: &kiln_model::FinishReason,
) -> &'a str {
    if let kiln_model::FinishReason::StopSequence(stop) = finish_reason {
        if !stop.is_empty() {
            if let Some(idx) = text.find(stop.as_str()) {
                return &text[..idx];
            }
        }
    }
    text
}

/// Non-streaming variant: split a fully-generated response text into
/// `(reasoning_content, content)` around the same `</think>` boundary the
/// streaming splitter handles. Returns `(None, raw)` when the prompt did not
/// prefill `<think>\n` so non-reasoning models keep emitting plain content.
fn split_reasoning_response(model_output: &str, prompt_text: &str) -> (Option<String>, String) {
    if !prompt_starts_in_reasoning(prompt_text) {
        return (None, model_output.to_string());
    }
    match model_output.find(REASONING_CLOSE_TAG) {
        Some(idx) => {
            let reasoning = model_output[..idx].to_string();
            let content = model_output[idx + REASONING_CLOSE_TAG.len()..].to_string();
            let reasoning_opt = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            };
            (reasoning_opt, content)
        }
        None => (Some(model_output.to_string()), String::new()),
    }
}

#[derive(Debug)]
struct AssistantOutputParts {
    content: String,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<serde_json::Value>>,
    finish_reason: String,
}

impl AssistantOutputParts {
    fn preview_source(&self) -> &str {
        if self.content.is_empty() {
            self.reasoning_content.as_deref().unwrap_or("")
        } else {
            self.content.as_str()
        }
    }
}

fn folded_reasoning_content(reasoning: &str, content: &str) -> String {
    let folded = format!("{REASONING_OPEN_TAG}{reasoning}{REASONING_CLOSE_TAG}");
    if content.is_empty() {
        folded
    } else {
        format!("{folded}\n\n{content}")
    }
}

fn unfold_reasoning_from_content(content: &str, reasoning: &str) -> String {
    let folded_prefix = format!("{REASONING_OPEN_TAG}{reasoning}{REASONING_CLOSE_TAG}");
    let Some(rest) = content.strip_prefix(&folded_prefix) else {
        return content.to_string();
    };
    rest.strip_prefix("\n\n").unwrap_or(rest).to_string()
}

fn response_content_for_cache(content: &str, reasoning: Option<&str>) -> String {
    match reasoning {
        Some(reasoning) if !reasoning.is_empty() => {
            unfold_reasoning_from_content(content, reasoning)
        }
        _ => content.to_string(),
    }
}

fn content_with_reasoning_policy(
    content: String,
    reasoning: Option<&str>,
    fold_reasoning_into_content: bool,
) -> String {
    if fold_reasoning_into_content
        && let Some(reasoning) = reasoning
        && !reasoning.is_empty()
    {
        return folded_reasoning_content(reasoning, &content);
    }
    content
}

fn apply_reasoning_content_policy(
    mut output: AssistantOutputParts,
    fold_reasoning_into_content: bool,
) -> AssistantOutputParts {
    output.content = content_with_reasoning_policy(
        output.content,
        output.reasoning_content.as_deref(),
        fold_reasoning_into_content,
    );
    output
}

struct PrefillProgressGuard(CancelHandle);

impl PrefillProgressGuard {
    fn new(cancel: CancelHandle) -> Self {
        Self(cancel)
    }
}

impl Drop for PrefillProgressGuard {
    fn drop(&mut self) {
        // This guard is task-scoped. Normal exits have already settled their
        // worker, so a late cancellation is harmless; unwind exits need the
        // signal so a detached blocking task cannot keep decoding unchecked.
        self.0.cancel();
        self.0.clear_prefill_progress();
    }
}

struct CancelOnDrop(CancelHandle);

impl CancelOnDrop {
    fn new(cancel: CancelHandle) -> Self {
        Self(cancel)
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        // A dropped HTTP future must still tell any queued or running blocking
        // worker to stop at its next cooperative boundary.
        self.0.cancel();
    }
}

fn tool_call_parsing_allowed(
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> bool {
    normalized_tools_for_cache(tools).is_some()
        && !matches!(tool_choice.and_then(|value| value.as_str()), Some("none"))
}

fn request_allows_tool_call_parsing(req: &ChatCompletionRequest) -> bool {
    tool_call_parsing_allowed(req.tools.as_deref(), req.tool_choice.as_ref())
}

fn batch_request_allows_tool_call_parsing(req: &BatchCompletionRequest) -> bool {
    tool_call_parsing_allowed(req.tools.as_deref(), req.tool_choice.as_ref())
}

fn assistant_output_from_model_output(
    req: &ChatCompletionRequest,
    model_output: &str,
    prompt_text: &str,
    finish_reason: &str,
) -> AssistantOutputParts {
    let (reasoning_content, content) = split_reasoning_response(model_output, prompt_text);
    assistant_output_from_split_parts(req, reasoning_content, content, finish_reason)
}

/// Parse-then-truncate: run tool-call parsing on the UNTRUNCATED engine
/// text, applying the stop truncation only when no tool calls parse.
///
/// Order matters: tools-bearing requests carry an implicit
/// `</tool_call>` stop, and the Qwen3 XML extractor REQUIRES the close
/// tag — truncating first (#1510's original shape) stripped the tag from
/// every stop-finished tool call, so the call failed to parse and the
/// raw XML leaked into `content`.
fn assistant_output_from_model_output_stop_aware(
    req: &ChatCompletionRequest,
    model_output: &str,
    prompt_text: &str,
    wire_finish: &str,
    engine_finish: &kiln_model::FinishReason,
) -> AssistantOutputParts {
    let parsed = assistant_output_from_model_output(req, model_output, prompt_text, wire_finish);
    if parsed.tool_calls.is_some() {
        return parsed;
    }
    let mut parsed = parsed;
    parsed.content = truncate_at_matched_stop(&parsed.content, engine_finish).to_string();
    if parsed.content.is_empty() {
        if let Some(reasoning) = parsed.reasoning_content.take() {
            let truncated = truncate_at_matched_stop(&reasoning, engine_finish).to_string();
            parsed.reasoning_content = (!truncated.is_empty()).then_some(truncated);
        }
    }
    parsed
}

fn assistant_output_from_cached_parts(
    req: &ChatCompletionRequest,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<serde_json::Value>>,
    finish_reason: String,
) -> AssistantOutputParts {
    if let Some(tool_calls) = tool_calls.filter(|calls| !calls.is_empty()) {
        return AssistantOutputParts {
            content,
            reasoning_content,
            tool_calls: Some(tool_calls),
            finish_reason: "tool_calls".to_string(),
        };
    }

    assistant_output_from_split_parts(req, reasoning_content, content, &finish_reason)
}

fn assistant_output_from_split_parts(
    req: &ChatCompletionRequest,
    reasoning_content: Option<String>,
    content: String,
    finish_reason: &str,
) -> AssistantOutputParts {
    assistant_output_from_split_parts_with_tool_parsing(
        request_allows_tool_call_parsing(req),
        reasoning_content,
        content,
        finish_reason,
    )
}

/// Stream-finish parsing with stop reconstruction: the emit gates strip
/// the matched stop from `content_buf` (it must never reach the wire or
/// the cache), but the implicit `</tool_call>` stop is part of the XML
/// grammar — the extractor REQUIRES the close tag. Re-append the matched
/// stop FOR PARSING ONLY; when no calls parse, fall back to the clean
/// buffer so the stop text can't leak into content.
fn stream_assistant_output_with_stop_reconstruction(
    buffer_tool_content: bool,
    reasoning_content: Option<String>,
    content_buf: &str,
    matched_stop: Option<&str>,
    finish: &str,
) -> AssistantOutputParts {
    if buffer_tool_content {
        if let Some(stop) = matched_stop {
            let reconstructed = format!("{content_buf}{stop}");
            let out = assistant_output_from_split_parts_with_tool_parsing(
                true,
                reasoning_content.clone(),
                reconstructed,
                finish,
            );
            if out.tool_calls.is_some() {
                return out;
            }
        }
    }
    assistant_output_from_split_parts_with_tool_parsing(
        buffer_tool_content,
        reasoning_content,
        content_buf.to_string(),
        finish,
    )
}

fn assistant_output_from_split_parts_with_tool_parsing(
    allow_tool_call_parsing: bool,
    reasoning_content: Option<String>,
    content: String,
    finish_reason: &str,
) -> AssistantOutputParts {
    if allow_tool_call_parsing {
        let parsed_content_calls = kiln_eval::qwen3::extract_tool_calls(&content);
        let content_has_tool_calls = !parsed_content_calls.is_empty();
        let parsed_calls = if content_has_tool_calls {
            parsed_content_calls
        } else {
            reasoning_content
                .as_deref()
                .map(kiln_eval::qwen3::extract_tool_calls)
                .unwrap_or_default()
        };
        if !parsed_calls.is_empty() {
            let content = if content_has_tool_calls {
                content_before_qwen_tool_call(&content).unwrap_or_default()
            } else {
                content
            };
            let reasoning_content = reasoning_content
                .map(|text| strip_qwen_tool_call_blocks(&text))
                .filter(|text| !text.trim().is_empty());
            return AssistantOutputParts {
                content,
                reasoning_content,
                tool_calls: Some(openai_tool_calls_from_qwen(&parsed_calls)),
                finish_reason: "tool_calls".to_string(),
            };
        }
    }

    AssistantOutputParts {
        content,
        reasoning_content,
        tool_calls: None,
        finish_reason: finish_reason.to_string(),
    }
}

fn content_before_qwen_tool_call(text: &str) -> Option<String> {
    let idx = text.find(QWEN_TOOL_CALL_OPEN_TAG)?;
    Some(text[..idx].trim_end().to_string())
}

fn strip_qwen_tool_call_blocks(text: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0usize;
    while cursor < text.len() {
        let Some(open_rel) = text[cursor..].find(QWEN_TOOL_CALL_OPEN_TAG) else {
            out.push_str(&text[cursor..]);
            break;
        };
        let open_abs = cursor + open_rel;
        out.push_str(&text[cursor..open_abs]);
        let body_start = open_abs + QWEN_TOOL_CALL_OPEN_TAG.len();
        let Some(close_rel) = text[body_start..].find(QWEN_TOOL_CALL_CLOSE_TAG) else {
            out.push_str(&text[open_abs..]);
            break;
        };
        cursor = body_start + close_rel + QWEN_TOOL_CALL_CLOSE_TAG.len();
    }
    out.trim_end().to_string()
}

fn openai_tool_calls_from_qwen(calls: &[ParsedToolCall]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .map(|call| {
            let arguments =
                serde_json::to_string(&serde_json::Value::Object(call.arguments.clone()))
                    .unwrap_or_else(|_| "{}".to_string());
            serde_json::json!({
                "id": format!("call_{}", Uuid::new_v4().simple()),
                "type": "function",
                "function": {
                    "name": call.name.clone(),
                    "arguments": arguments,
                },
            })
        })
        .collect()
}

fn tool_call_deltas_from_openai_calls(calls: &[serde_json::Value]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let mut value = call.clone();
            if let serde_json::Value::Object(ref mut object) = value {
                object.insert(
                    "index".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(index)),
                );
            }
            value
        })
        .collect()
}

/// Push a [`RequestRecord`] into the dashboard's recent-requests ring. Logs a
/// warning if the lock is poisoned but otherwise never panics — request
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

#[derive(Serialize)]
struct ChatPromptMessageCacheKey<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Cow<'a, [serde_json::Value]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

fn message_cache_keys(messages: &[Message]) -> Vec<ChatPromptMessageCacheKey<'_>> {
    messages
        .iter()
        .map(|message| ChatPromptMessageCacheKey {
            role: &message.role,
            content: &message.content,
            tool_calls: normalized_message_tool_calls_for_cache(message.tool_calls.as_deref()),
            name: message.name.as_deref(),
            tool_call_id: message.tool_call_id.as_deref(),
        })
        .collect()
}

#[derive(Serialize)]
struct RenderedPromptCacheKey<'a> {
    messages: &'a [ChatPromptMessageCacheKey<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [serde_json::Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

fn normalized_tools_for_cache(tools: Option<&[serde_json::Value]>) -> Option<&[serde_json::Value]> {
    tools.filter(|tools| !tools.is_empty())
}

fn normalized_tools_option_for_synthetic_request(
    tools: Option<&[serde_json::Value]>,
) -> Option<Vec<serde_json::Value>> {
    normalized_tools_for_cache(tools).map(Vec::from)
}

fn normalized_tool_choice_for_cache<'a>(
    normalized_tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&'a serde_json::Value>,
) -> Option<&'a serde_json::Value> {
    if normalized_tools.is_none()
        && matches!(
            tool_choice.and_then(|value| value.as_str()),
            Some("auto" | "none")
        )
    {
        return None;
    }
    tool_choice
}

fn normalized_chat_template_kwargs_for_cache(
    chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    chat_template_kwargs.filter(|kwargs| !kwargs.is_empty())
}

fn normalized_tool_choice_option_for_synthetic_request(
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let normalized_tools = normalized_tools_for_cache(tools);
    normalized_tool_choice_for_cache(normalized_tools, tool_choice).cloned()
}

fn normalized_message_tool_calls_for_cache(
    tool_calls: Option<&[serde_json::Value]>,
) -> Option<Cow<'_, [serde_json::Value]>> {
    let tool_calls = tool_calls.filter(|tool_calls| !tool_calls.is_empty())?;
    let mut normalized: Option<Vec<serde_json::Value>> = None;

    for (index, tool_call) in tool_calls.iter().enumerate() {
        if let Some(normalized_tool_call) = normalized_tool_call_for_cache(tool_call) {
            let values = normalized.get_or_insert_with(|| tool_calls[..index].to_vec());
            values.push(normalized_tool_call);
        } else if let Some(values) = normalized.as_mut() {
            values.push(tool_call.clone());
        }
    }

    Some(match normalized {
        Some(values) => Cow::Owned(values),
        None => Cow::Borrowed(tool_calls),
    })
}

fn normalized_tool_call_for_cache(tool_call: &serde_json::Value) -> Option<serde_json::Value> {
    let serde_json::Value::Object(object) = tool_call else {
        return None;
    };
    let mut normalized: Option<serde_json::Map<String, serde_json::Value>> = None;

    if let Some(arguments) = parsed_json_argument_for_cache(object.get("arguments")) {
        normalized
            .get_or_insert_with(|| object.clone())
            .insert("arguments".to_string(), arguments);
    }

    if let Some(function) = object
        .get("function")
        .and_then(normalized_tool_call_function_for_cache)
    {
        normalized
            .get_or_insert_with(|| object.clone())
            .insert("function".to_string(), function);
    }

    normalized.map(serde_json::Value::Object)
}

fn normalized_tool_call_function_for_cache(
    function: &serde_json::Value,
) -> Option<serde_json::Value> {
    let serde_json::Value::Object(object) = function else {
        return None;
    };
    let arguments = parsed_json_argument_for_cache(object.get("arguments"))?;
    let mut normalized = object.clone();
    normalized.insert("arguments".to_string(), arguments);
    Some(serde_json::Value::Object(normalized))
}

fn parsed_json_argument_for_cache(value: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    serde_json::from_str(value?.as_str()?).ok()
}

pub(crate) fn render_prompt_text(
    state: &AppState,
    messages: &[Message],
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
    chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<String, ApiError> {
    let merged_chat_template_kwargs =
        effective_chat_template_kwargs(state.default_thinking_enabled, chat_template_kwargs);
    let normalized_tools = normalized_tools_for_cache(tools);
    let normalized_tool_choice = normalized_tool_choice_for_cache(normalized_tools, tool_choice);
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(Some(&merged_chat_template_kwargs));
    let message_keys = message_cache_keys(messages);
    let key = serde_json::to_string(&RenderedPromptCacheKey {
        messages: &message_keys,
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
    })
    .map_err(|err| ApiError::internal(format!("failed to key rendered prompt cache: {err}")))?;

    if let Some(prompt_text) = state.rendered_prompt_cache.lock().unwrap().get(&key) {
        return Ok(prompt_text);
    }

    let chat_messages: Vec<ChatMessage> = messages.iter().map(message_to_chat).collect();
    let chat_template_options =
        chat_template_options_from_kwargs(Some(&merged_chat_template_kwargs));
    let prompt_text = state
        .tokenizer
        .apply_chat_template_full_with_options(
            &chat_messages,
            normalized_tools,
            normalized_tool_choice,
            chat_template_options,
        )
        .map_err(ApiError::chat_template_failed)?;
    state
        .rendered_prompt_cache
        .lock()
        .unwrap()
        .insert(key, prompt_text.clone());
    Ok(prompt_text)
}

fn chat_template_options_from_kwargs(
    chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
) -> ChatTemplateOptions {
    ChatTemplateOptions {
        template_kwargs: chat_template_kwargs.cloned().unwrap_or_default(),
    }
}

pub(crate) fn encode_prompt_tokens(
    state: &AppState,
    prompt_text: &str,
) -> Result<Vec<TokenId>, ApiError> {
    if let Some(tokens) = state.prompt_token_cache.lock().unwrap().get(prompt_text) {
        return Ok(tokens);
    }

    let tokens = state
        .tokenizer
        .encode(prompt_text)
        .map_err(ApiError::tokenization_failed)?;
    state
        .prompt_token_cache
        .lock()
        .unwrap()
        .insert(prompt_text.to_string(), tokens.clone());
    Ok(tokens)
}

/// The serving context ceiling: the model's positional limit capped by
/// what the KV pool can physically hold. `None` = no enforcible signal
/// (mock backend).
pub(crate) fn serving_context_ceiling(state: &AppState) -> Option<usize> {
    let model_max = state.model_config.max_position_embeddings;
    match state.backend.as_ref() {
        ModelBackend::Real { block_manager, .. } => {
            let bm = block_manager.lock().unwrap();
            let kv_max = bm.num_blocks().saturating_mul(bm.block_size());
            Some(model_max.min(kv_max))
        }
        ModelBackend::Mock { .. } => None,
    }
}

/// Long agent sessions used to die in opaque 500s exactly when the
/// conversation grew: nothing validated the prompt against the model's
/// positional limit or the KV pool, so BlockManager OOM surfaced as a
/// generic "Retry the request" — and agent harnesses (pi included) key
/// their auto-compaction off an OpenAI-style 400 with code
/// `context_length_exceeded`, which kiln never sent.
///
/// - Prompt alone ≥ ceiling → the 400 the harness can act on, with the
///   counts in the message (OpenAI wording shape).
/// - Prompt fits but prompt+max_tokens overflows → clamp max_tokens to
///   the remaining window; the truncation is visible as
///   finish_reason="length".
pub(crate) fn enforce_context_window(
    state: &AppState,
    sampling: &mut SamplingParams,
    prompt_token_count: usize,
) -> Result<(), ApiError> {
    enforce_context_window_with_ceiling(
        serving_context_ceiling(state),
        sampling,
        prompt_token_count,
    )
}

pub(crate) fn enforce_context_window_with_ceiling(
    ceiling: Option<usize>,
    sampling: &mut SamplingParams,
    prompt_token_count: usize,
) -> Result<(), ApiError> {
    let Some(ceiling) = ceiling else {
        return Ok(());
    };
    if prompt_token_count >= ceiling {
        return Err(ApiError::context_length_exceeded(
            ceiling,
            prompt_token_count,
            sampling.max_tokens,
        ));
    }
    let remaining = ceiling - prompt_token_count;
    if sampling.max_tokens > remaining {
        tracing::debug!(
            prompt_tokens = prompt_token_count,
            requested_max_tokens = sampling.max_tokens,
            clamped_max_tokens = remaining,
            ceiling,
            "clamping max_tokens to the remaining context window"
        );
        sampling.max_tokens = remaining;
    }
    Ok(())
}

fn deterministic_completion_cache_key_for_adapter(
    state: &AppState,
    adapter: Option<LoadedAdapterIdentity>,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    fold_reasoning_into_content: bool,
) -> Option<DeterministicCompletionCacheKey> {
    if sampling
        .thinking_budget
        .as_ref()
        .and_then(ThinkingBudget::max_time)
        .is_some()
    {
        return None;
    }
    let greedy = sampling.is_effectively_greedy();
    if !greedy && sampling.seed.is_none() {
        return None;
    }

    // Greedy argmax does not consult RNG, sampling filters, or token
    // penalties (`sample_step` short-circuits before the penalty pass —
    // see kiln-model/src/sampling.rs), so normalize those fields to
    // maximize equivalent cache hits. Seeded sampling is replayable and
    // must keep every parameter that changes the token path.
    let (
        temperature_bits,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    ) = if greedy {
        (
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0,
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            None,
        )
    } else {
        (
            sampling.temperature.to_bits(),
            normalized_top_p_bits_for_cache(sampling.top_p),
            normalized_top_k_for_cache(sampling.top_k, state.model_config.vocab_size),
            normalized_min_p_bits_for_cache(sampling.min_p),
            normalized_penalty_bits_for_cache(sampling.presence_penalty, 0.0),
            normalized_penalty_bits_for_cache(sampling.frequency_penalty, 0.0),
            normalized_penalty_bits_for_cache(sampling.repetition_penalty, 1.0),
            sampling.seed,
        )
    };

    let (global_generation, adapter_generation) = state.deterministic_cache_fence(&adapter);
    Some(DeterministicCompletionCacheKey {
        adapter,
        global_generation,
        adapter_generation,
        prompt_tokens: prompt_tokens.to_vec(),
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(sampling),
        thinking_budget_tokens: sampling
            .thinking_budget
            .as_ref()
            .and_then(ThinkingBudget::max_tokens),
        stop: normalized_stop_for_cache(&sampling.stop),
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
        fold_reasoning_into_content,
    })
}

#[cfg(test)]
fn deterministic_completion_cache_key(
    state: &AppState,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    fold_reasoning_into_content: bool,
) -> Option<DeterministicCompletionCacheKey> {
    deterministic_completion_cache_key_for_adapter(
        state,
        state.loaded_adapter_identity(),
        prompt_tokens,
        sampling,
        fold_reasoning_into_content,
    )
}

#[derive(Serialize)]
struct DeterministicChatRequestCacheKey<'a> {
    messages: &'a [ChatPromptMessageCacheKey<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [serde_json::Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "is_false")]
    fold_reasoning_into_content: bool,
    temperature_bits: u32,
    max_tokens: usize,
    #[serde(skip_serializing_if = "is_false")]
    ignore_eos: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_tokens: Option<usize>,
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
    min_p_bits: u32,
    presence_penalty_bits: u32,
    frequency_penalty_bits: u32,
    repetition_penalty_bits: u32,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct DeterministicChatChoicesCacheKey<'a> {
    messages: &'a [ChatPromptMessageCacheKey<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [serde_json::Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "is_false")]
    fold_reasoning_into_content: bool,
    n: usize,
    temperature_bits: u32,
    max_tokens: usize,
    #[serde(skip_serializing_if = "is_false")]
    ignore_eos: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_tokens: Option<usize>,
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
    min_p_bits: u32,
    presence_penalty_bits: u32,
    frequency_penalty_bits: u32,
    repetition_penalty_bits: u32,
    seed: Option<u64>,
}

#[cfg(test)]
fn deterministic_chat_request_cache_key(
    req: &ChatCompletionRequest,
    sampling: &SamplingParams,
) -> Result<Option<String>, ApiError> {
    deterministic_chat_request_cache_key_with_vocab_size(req, sampling, usize::MAX)
}

#[cfg(test)]
fn deterministic_chat_request_cache_key_with_vocab_size(
    req: &ChatCompletionRequest,
    sampling: &SamplingParams,
    vocab_size: usize,
) -> Result<Option<String>, ApiError> {
    deterministic_chat_request_cache_key_with_vocab_size_and_fold(
        req,
        sampling,
        vocab_size,
        req.fold_reasoning_into_content.unwrap_or(false),
        request_token_budget_without_server_default(req),
    )
}

fn request_token_budget_without_server_default(req: &ChatCompletionRequest) -> Option<usize> {
    match req.thinking_budget_tokens {
        BudgetOverride::Limited(value) => Some(value),
        BudgetOverride::Inherit | BudgetOverride::Unlimited => None,
    }
}

fn deterministic_chat_request_cache_key_with_vocab_size_and_fold(
    req: &ChatCompletionRequest,
    sampling: &SamplingParams,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
    thinking_budget_tokens: Option<usize>,
) -> Result<Option<String>, ApiError> {
    if req.rollout_provenance {
        return Ok(None);
    }
    if req.n.unwrap_or(1) != 1 {
        return Ok(None);
    }

    if req.adapter.is_explicit() || req.adapters.is_some() {
        return Ok(None);
    }

    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let NormalizedRequestSamplingKey {
        temperature_bits,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    } = normalized_deterministic_request_sampling_key(sampling, vocab_size);

    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());
    let message_keys = message_cache_keys(&req.messages);

    serde_json::to_string(&DeterministicChatRequestCacheKey {
        messages: &message_keys,
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
        fold_reasoning_into_content,
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(sampling),
        thinking_budget_tokens,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key chat request cache: {err}")))
}

/// Resolve a raw chat request's sampling for cache keying by deriving
/// from [`sampling_params_for_chat_request`] itself — lockstep BY
/// CONSTRUCTION (any future profile/preset logic flows into keys
/// automatically; the old per-field duplicate defaults could silently
/// diverge). Key semantics preserved: raw request stops (not the
/// generation-augmented set) and the caller's seed.
fn chat_request_sampling_for_cache_key(
    req: &ChatCompletionRequest,
    seed: Option<u64>,
) -> SamplingParams {
    let mut params = sampling_params_for_chat_request(req);
    params.stop = req.stop.clone().unwrap_or_default();
    params.seed = seed;
    params
}

/// Batch-request twin of [`chat_request_sampling_for_cache_key`].
fn batch_request_sampling_for_cache_key(
    req: &BatchCompletionRequest,
    seed: Option<u64>,
) -> SamplingParams {
    SamplingParams {
        temperature: requested_or_default_temperature(req.temperature),
        top_p: requested_or_default_top_p(req.top_p),
        top_k: requested_or_default_top_k(req.top_k),
        min_p: requested_or_default_min_p(req.min_p),
        max_tokens: batch_request_max_tokens(req),
        ignore_eos: req.ignore_eos,
        repetition_penalty: requested_or_default_repetition_penalty(req.repetition_penalty),
        presence_penalty: requested_or_default_presence_penalty(req.presence_penalty),
        frequency_penalty: requested_or_default_frequency_penalty(req.frequency_penalty),
        stop: req.stop.clone().unwrap_or_default(),
        seed,
        thinking_budget: None,
    }
}

fn batch_token_budget_without_server_default(req: &BatchCompletionRequest) -> Option<usize> {
    match req.thinking_budget_tokens {
        BudgetOverride::Limited(value) => Some(value),
        BudgetOverride::Inherit | BudgetOverride::Unlimited => None,
    }
}

fn deterministic_chat_request_cache_key_from_chat_choice_with_vocab_size_and_fold(
    req: &ChatCompletionRequest,
    seed: Option<u64>,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Result<Option<String>, ApiError> {
    if req.adapter.is_explicit() || req.adapters.is_some() {
        return Ok(None);
    }

    let sampling = chat_request_sampling_for_cache_key(req, seed);
    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let NormalizedRequestSamplingKey {
        temperature_bits,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    } = normalized_deterministic_request_sampling_key(&sampling, vocab_size);
    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());
    let message_keys = message_cache_keys(&req.messages);

    serde_json::to_string(&DeterministicChatRequestCacheKey {
        messages: &message_keys,
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
        fold_reasoning_into_content,
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(&sampling),
        thinking_budget_tokens: request_token_budget_without_server_default(req),
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key chat choice request cache: {err}")))
}

#[cfg(test)]
fn deterministic_chat_choices_cache_key(
    req: &ChatCompletionRequest,
    n_per: usize,
    sampling: &SamplingParams,
) -> Result<Option<String>, ApiError> {
    deterministic_chat_choices_cache_key_with_vocab_size(req, n_per, sampling, usize::MAX)
}

#[cfg(test)]
fn deterministic_chat_choices_cache_key_with_vocab_size(
    req: &ChatCompletionRequest,
    n_per: usize,
    sampling: &SamplingParams,
    vocab_size: usize,
) -> Result<Option<String>, ApiError> {
    deterministic_chat_choices_cache_key_with_vocab_size_and_fold(
        req,
        n_per,
        sampling,
        vocab_size,
        req.fold_reasoning_into_content.unwrap_or(false),
        request_token_budget_without_server_default(req),
    )
}

fn deterministic_chat_choices_cache_key_with_vocab_size_and_fold(
    req: &ChatCompletionRequest,
    n_per: usize,
    sampling: &SamplingParams,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
    thinking_budget_tokens: Option<usize>,
) -> Result<Option<String>, ApiError> {
    if n_per <= 1 || req.adapter.is_explicit() || req.adapters.is_some() {
        return Ok(None);
    }

    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let NormalizedRequestSamplingKey {
        temperature_bits,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    } = normalized_deterministic_request_sampling_key(sampling, vocab_size);

    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());
    let message_keys = message_cache_keys(&req.messages);

    serde_json::to_string(&DeterministicChatChoicesCacheKey {
        messages: &message_keys,
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
        fold_reasoning_into_content,
        n: n_per,
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(sampling),
        thinking_budget_tokens,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key chat choices cache: {err}")))
}

fn deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
    req: &BatchCompletionRequest,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Result<Option<String>, ApiError> {
    if req.prompts.len() != 1 || req.adapter.is_some() || req.adapters.is_some() {
        return Ok(None);
    }

    deterministic_chat_choices_cache_key_from_batch_prompt_with_vocab_size_and_fold(
        req,
        &req.prompts[0],
        req.seed,
        vocab_size,
        fold_reasoning_into_content,
    )
}

fn deterministic_chat_choices_cache_key_from_batch_prompt_with_vocab_size_and_fold(
    req: &BatchCompletionRequest,
    messages: &[Message],
    seed: Option<u64>,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Result<Option<String>, ApiError> {
    if req.adapter.is_some() || req.adapters.is_some() {
        return Ok(None);
    }

    let n_per = req.n.unwrap_or(1);
    if n_per <= 1 {
        return Ok(None);
    }

    let sampling = batch_request_sampling_for_cache_key(req, seed);
    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let NormalizedRequestSamplingKey {
        temperature_bits,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    } = normalized_deterministic_request_sampling_key(&sampling, vocab_size);
    let message_keys = batch_synth_message_cache_keys(messages);
    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

    serde_json::to_string(&DeterministicChatChoicesCacheKey {
        messages: &message_keys,
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
        fold_reasoning_into_content,
        n: n_per,
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(&sampling),
        thinking_budget_tokens: batch_token_budget_without_server_default(req),
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key batch chat choices cache: {err}")))
}

fn batch_synth_message_cache_keys(messages: &[Message]) -> Vec<ChatPromptMessageCacheKey<'_>> {
    messages
        .iter()
        .map(|message| ChatPromptMessageCacheKey {
            role: &message.role,
            content: &message.content,
            tool_calls: normalized_message_tool_calls_for_cache(message.tool_calls.as_deref()),
            name: message.name.as_deref(),
            tool_call_id: message.tool_call_id.as_deref(),
        })
        .collect()
}

fn deterministic_chat_request_cache_key_from_batch_prompt_with_vocab_size_and_fold(
    req: &BatchCompletionRequest,
    messages: &[Message],
    seed: Option<u64>,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Result<Option<String>, ApiError> {
    if req.adapter.is_some() || req.adapters.is_some() {
        return Ok(None);
    }

    let sampling = batch_request_sampling_for_cache_key(req, seed);
    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let NormalizedRequestSamplingKey {
        temperature_bits,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    } = normalized_deterministic_request_sampling_key(&sampling, vocab_size);
    let message_keys = batch_synth_message_cache_keys(messages);
    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

    serde_json::to_string(&DeterministicChatRequestCacheKey {
        messages: &message_keys,
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
        fold_reasoning_into_content,
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(&sampling),
        thinking_budget_tokens: batch_token_budget_without_server_default(req),
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key batch chat request cache: {err}")))
}

/// Sampling fields of a deterministic request cache key, normalized so
/// that equivalent spellings of the same token path share entries.
struct NormalizedRequestSamplingKey {
    temperature_bits: u32,
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
    min_p_bits: u32,
    presence_penalty_bits: u32,
    frequency_penalty_bits: u32,
    repetition_penalty_bits: u32,
    seed: Option<u64>,
}

impl NormalizedRequestSamplingKey {
    /// Greedy argmax (and the zero-output case) never consults the RNG,
    /// the sampling filters, or the token penalties — `sample_step`
    /// short-circuits before the penalty pass (kiln-model/src/sampling.rs)
    /// — so pin every such field to its no-op spelling.
    fn greedy(stop: Vec<String>) -> Self {
        Self {
            temperature_bits: 0.0f32.to_bits(),
            stop,
            top_p_bits: 1.0f32.to_bits(),
            top_k: 0,
            min_p_bits: 0.0f32.to_bits(),
            presence_penalty_bits: 0.0f32.to_bits(),
            frequency_penalty_bits: 0.0f32.to_bits(),
            repetition_penalty_bits: 1.0f32.to_bits(),
            seed: None,
        }
    }
}

fn normalized_deterministic_request_sampling_key(
    sampling: &SamplingParams,
    vocab_size: usize,
) -> NormalizedRequestSamplingKey {
    if sampling.max_tokens == 0 {
        return NormalizedRequestSamplingKey::greedy(Vec::new());
    }

    let stop = normalized_stop_for_cache(&sampling.stop);
    if sampling.is_effectively_greedy() {
        return NormalizedRequestSamplingKey::greedy(stop);
    }

    NormalizedRequestSamplingKey {
        temperature_bits: sampling.temperature.to_bits(),
        stop,
        top_p_bits: normalized_top_p_bits_for_cache(sampling.top_p),
        top_k: normalized_top_k_for_cache(sampling.top_k, vocab_size),
        min_p_bits: normalized_min_p_bits_for_cache(sampling.min_p),
        presence_penalty_bits: normalized_penalty_bits_for_cache(sampling.presence_penalty, 0.0),
        frequency_penalty_bits: normalized_penalty_bits_for_cache(sampling.frequency_penalty, 0.0),
        repetition_penalty_bits: normalized_penalty_bits_for_cache(
            sampling.repetition_penalty,
            1.0,
        ),
        seed: sampling.seed,
    }
}

fn normalized_top_p_bits_for_cache(top_p: f32) -> u32 {
    if SamplingParams::top_p_disables_nucleus_filter(top_p) {
        1.0f32.to_bits()
    } else {
        top_p.to_bits()
    }
}

fn normalized_min_p_bits_for_cache(min_p: f32) -> u32 {
    if SamplingParams::min_p_is_disabled(min_p) {
        0.0f32.to_bits()
    } else {
        min_p.to_bits()
    }
}

fn normalized_ignore_eos_for_cache(sampling: &SamplingParams) -> bool {
    sampling.max_tokens != 0 && sampling.ignore_eos
}

/// Fold alternate spellings of a penalty's no-op value (`-0.0` for the
/// subtractive penalties) onto canonical no-op bits so equivalent
/// requests share cache entries.
fn normalized_penalty_bits_for_cache(value: f32, no_op: f32) -> u32 {
    if value == no_op {
        no_op.to_bits()
    } else {
        value.to_bits()
    }
}

fn normalized_top_k_for_cache(top_k: u32, vocab_size: usize) -> u32 {
    if top_k != 0 && (top_k as usize) >= vocab_size {
        0
    } else {
        top_k
    }
}

fn normalized_stop_for_cache(stop: &[String]) -> Vec<String> {
    if stop.is_empty() {
        return Vec::new();
    }
    if stop.iter().any(|value| value.is_empty()) {
        return vec![String::new()];
    }
    let mut normalized = stop.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    normalized.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    let mut minimal: Vec<String> = Vec::with_capacity(normalized.len());
    for value in normalized {
        if minimal.iter().any(|kept| value.contains(kept)) {
            continue;
        }
        minimal.push(value);
    }
    minimal
}

fn normalized_stop_for_generation(stop: Option<&[String]>) -> Vec<String> {
    stop.map(normalized_stop_for_cache).unwrap_or_default()
}

fn stop_sequences_for_chat_generation(req: &ChatCompletionRequest) -> Vec<String> {
    let mut stop = normalized_stop_for_generation(req.stop.as_deref());
    if request_allows_tool_call_parsing(req)
        && !stop.iter().any(|value| {
            value.is_empty()
                || value.as_str() == QWEN_TOOL_CALL_CLOSE_TAG
                || QWEN_TOOL_CALL_CLOSE_TAG.contains(value.as_str())
        })
    {
        stop.push(QWEN_TOOL_CALL_CLOSE_TAG.to_string());
    }
    stop
}

fn normalized_stop_option_for_synthetic_request(stop: Option<&[String]>) -> Option<Vec<String>> {
    let stop = normalized_stop_for_generation(stop);
    if stop.is_empty() { None } else { Some(stop) }
}

fn resolved_max_tokens(max_tokens: Option<usize>, max_completion_tokens: Option<usize>) -> usize {
    max_tokens.or(max_completion_tokens).unwrap_or(2048)
}

fn chat_request_max_tokens(req: &ChatCompletionRequest) -> usize {
    resolved_max_tokens(req.max_tokens, req.max_completion_tokens)
}

fn batch_request_max_tokens(req: &BatchCompletionRequest) -> usize {
    resolved_max_tokens(req.max_tokens, req.max_completion_tokens)
}

fn effective_batch_thinking_budget_for_request(
    state: &AppState,
    req: &BatchCompletionRequest,
) -> EffectiveThinkingBudget {
    resolve_effective_thinking_budget(state, req.thinking_budget_tokens, req.thinking_budget_ms)
}

fn batch_completion_metadata_for_request(
    state: &AppState,
    req: &BatchCompletionRequest,
) -> BatchCompletionMetadata {
    BatchCompletionMetadata {
        thinking_budget: effective_batch_thinking_budget_for_request(state, req).into(),
    }
}

fn response_from_cached_chat_request(
    state: &AppState,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    cached: DeterministicChatRequestCacheValue,
) -> ChatCompletionResponse {
    response_from_cached_completion(
        state,
        req,
        cached.prompt_tokens,
        request_start,
        cached.completion,
    )
}

fn streaming_response_from_cached_chat_request(
    state: &AppState,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    cached: DeterministicChatRequestCacheValue,
) -> Response {
    streaming_response_from_cached_completion(
        state,
        req,
        cached.prompt_tokens,
        request_start,
        cached.completion,
    )
}

fn chat_request_cache_value_from_response(
    resp: &ChatCompletionResponse,
) -> Option<DeterministicChatRequestCacheValue> {
    Some(chat_request_cache_value_from_completion(
        resp.usage.prompt_tokens,
        cache_value_from_response(resp)?,
    ))
}

fn chat_request_cache_value_from_completion(
    prompt_tokens: usize,
    completion: DeterministicCompletionCacheValue,
) -> DeterministicChatRequestCacheValue {
    DeterministicChatRequestCacheValue {
        prompt_tokens,
        completion,
    }
}

fn chat_request_cache_value_from_choice(
    prompt_tokens: usize,
    choice: &Choice,
) -> DeterministicChatRequestCacheValue {
    DeterministicChatRequestCacheValue {
        prompt_tokens,
        completion: DeterministicCompletionCacheValue {
            text: response_content_for_cache(
                &choice.message.content,
                choice.message.reasoning_content.as_deref(),
            ),
            reasoning_content: choice.message.reasoning_content.clone(),
            tool_calls: choice.message.tool_calls.clone(),
            finish_reason: choice.finish_reason.clone(),
            completion_tokens: choice.completion_tokens,
            thinking_budget_status: choice.thinking_budget,
        },
    }
}

fn store_chat_request_cache_from_chat_choices_response(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &ChatCompletionRequest,
    resp: &ChatCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    if effective_thinking_budget_for_request(state, req)
        .max_time_ms
        .is_some()
        || chat_request_max_tokens(req) != 0
        || req.adapter.is_explicit()
        || req.adapters.is_some()
    {
        return Ok(());
    }
    let mut entries = Vec::with_capacity(resp.choices.len());
    let mut seen_keys = std::collections::HashSet::new();
    for choice in &resp.choices {
        let seed = req.seed.map(|seed| seed.wrapping_add(choice.index as u64));
        let Some(key) =
            deterministic_chat_request_cache_key_from_chat_choice_with_vocab_size_and_fold(
                req,
                seed,
                vocab_size,
                fold_reasoning_into_content_for_request(state, req),
            )?
        else {
            continue;
        };
        if seen_keys.insert(key.clone()) {
            entries.push((
                key,
                chat_request_cache_value_from_choice(resp.usage.prompt_tokens, choice),
            ));
        }
    }

    let mut cache = state.chat_request_cache.lock().unwrap();
    for (key, value) in entries {
        cache.insert(state.deterministic_cache_key(adapter.clone(), key), value);
    }
    Ok(())
}

async fn zero_chat_choices_response_from_request_cache_hit(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    n_per: usize,
    vocab_size: usize,
) -> Result<Option<ChatCompletionResponse>, ApiError> {
    if chat_request_max_tokens(req) != 0 || req.adapter.is_explicit() || req.adapters.is_some() {
        return Ok(None);
    }

    let Some(key) = deterministic_chat_request_cache_key_from_chat_choice_with_vocab_size_and_fold(
        req,
        req.seed,
        vocab_size,
        fold_reasoning_into_content_for_request(state, req),
    )?
    else {
        return Ok(None);
    };

    let key = state.deterministic_cache_key(adapter.clone(), key);
    let probe = state.chat_request_cache.lock().unwrap().probe(&key);
    let cached = match probe {
        DeterministicChatRequestCacheProbe::Hit(cached) => cached,
        DeterministicChatRequestCacheProbe::Wait(receiver) => {
            let Some(cached) = wait_for_deterministic_chat_request(receiver).await else {
                return Ok(None);
            };
            cached
        }
        DeterministicChatRequestCacheProbe::Miss => return Ok(None),
    };

    let resp = response_from_cached_chat_request(state, req, request_start, cached);
    chat_response_from_multi_responses(state, req, request_start, vec![(0, resp)], n_per, true)
        .map(Some)
}

fn chat_choices_cache_value_from_response(
    resp: &ChatCompletionResponse,
) -> Option<DeterministicChatChoicesCacheValue> {
    let completions = resp
        .choices
        .iter()
        .map(|choice| DeterministicCompletionCacheValue {
            text: response_content_for_cache(
                &choice.message.content,
                choice.message.reasoning_content.as_deref(),
            ),
            reasoning_content: choice.message.reasoning_content.clone(),
            tool_calls: choice.message.tool_calls.clone(),
            finish_reason: choice.finish_reason.clone(),
            completion_tokens: choice.completion_tokens,
            thinking_budget_status: choice.thinking_budget,
        })
        .collect();

    Some(DeterministicChatChoicesCacheValue {
        prompt_tokens: resp.usage.prompt_tokens,
        completions,
    })
}

fn response_from_cached_completion(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_token_count: usize,
    request_start: std::time::Instant,
    cached: DeterministicCompletionCacheValue,
) -> ChatCompletionResponse {
    let now = now_epoch();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());

    let completion_tokens = cached.completion_tokens;
    let thinking_budget_status = cached.thinking_budget_status;
    let cached_output = assistant_output_from_cached_parts(
        req,
        cached.text,
        cached.reasoning_content,
        cached.tool_calls,
        cached.finish_reason,
    );
    let metadata = chat_completion_metadata_from_cached_output(state, req, &cached_output);
    let cached_output = apply_reasoning_content_policy(
        cached_output,
        fold_reasoning_into_content_for_request(state, req),
    );
    let preview_source = cached_output.preview_source();
    record_recent_request(
        state,
        RequestRecord {
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: cached_output.finish_reason.clone(),
            prefix_cache: Some("completion_cache_hit".to_string()),
            thinking_budget: Some(recent_thinking_budget_with_status(
                &metadata.thinking_budget,
                thinking_budget_status,
            )),
            ..request_record_from_req(state, req, &id, &model, false)
        },
    );

    let finish_reason = cached_output.finish_reason.clone();
    let mut response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: cached_output.content,
                reasoning_content: cached_output.reasoning_content,
                tool_calls: cached_output.tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason,
            thinking_budget: thinking_budget_status,
            rollout_provenance: None,
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    attach_cached_thinking_budget_outcome(&mut response);
    attach_chat_performance_metadata(
        state,
        req,
        &mut response,
        request_start,
        Some(std::time::Duration::ZERO),
        Some(std::time::Duration::ZERO),
        Some(std::time::Duration::ZERO),
    );
    response
}

fn response_from_cached_chat_choices(
    state: &AppState,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    cached: DeterministicChatChoicesCacheValue,
) -> ChatCompletionResponse {
    let now = now_epoch();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());

    let cached_completions = cached
        .completions
        .into_iter()
        .map(|completion| {
            let completion_tokens = completion.completion_tokens;
            let thinking_budget_status = completion.thinking_budget_status;
            let output = assistant_output_from_cached_parts(
                req,
                completion.text,
                completion.reasoning_content,
                completion.tool_calls,
                completion.finish_reason,
            );
            (completion_tokens, thinking_budget_status, output)
        })
        .collect::<Vec<_>>();
    let metadata = cached_completions
        .first()
        .map(|(_, _, output)| chat_completion_metadata_from_cached_output(state, req, output))
        .unwrap_or_else(|| chat_completion_metadata_from_request(state, req));
    let fold_reasoning = fold_reasoning_into_content_for_request(state, req);
    let cached_completions = cached_completions
        .into_iter()
        .map(|(completion_tokens, thinking_budget_status, output)| {
            (
                completion_tokens,
                thinking_budget_status,
                apply_reasoning_content_policy(output, fold_reasoning),
            )
        })
        .collect::<Vec<_>>();
    let preview_source = cached_completions
        .first()
        .map(|(_, _, output)| output.preview_source())
        .unwrap_or("");
    let completion_tokens = cached_completions
        .iter()
        .map(|(completion_tokens, _, _)| *completion_tokens)
        .sum::<usize>();
    let recent_thinking_budget = recent_thinking_budget_with_status(
        &metadata.thinking_budget,
        cached_completions
            .first()
            .and_then(|(_, status, _)| *status),
    );
    record_recent_request(
        state,
        RequestRecord {
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: cached.prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: cached_completions
                .first()
                .map(|(_, _, output)| output.finish_reason.clone())
                .unwrap_or_else(|| "length".to_string()),
            prefix_cache: Some("chat_choices_cache_hit".to_string()),
            thinking_budget: Some(recent_thinking_budget),
            ..request_record_from_req(state, req, &id, &model, false)
        },
    );

    let choices = cached_completions
        .into_iter()
        .enumerate()
        .map(
            |(index, (completion_tokens, thinking_budget_status, output))| Choice {
                index,
                message: Message {
                    role: "assistant".to_string(),
                    content: output.content,
                    reasoning_content: output.reasoning_content,
                    tool_calls: output.tool_calls,
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: output.finish_reason,
                thinking_budget: thinking_budget_status,
                rollout_provenance: None,
                completion_tokens,
            },
        )
        .collect();

    let mut response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model,
        choices,
        usage: Usage {
            prompt_tokens: cached.prompt_tokens,
            completion_tokens,
            total_tokens: cached.prompt_tokens.saturating_add(completion_tokens),
        },
        metadata,
    };
    attach_cached_thinking_budget_outcome(&mut response);
    attach_chat_performance_metadata(
        state,
        req,
        &mut response,
        request_start,
        Some(std::time::Duration::ZERO),
        Some(std::time::Duration::ZERO),
        Some(std::time::Duration::ZERO),
    );
    response
}

fn streaming_response_from_cached_completion(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_token_count: usize,
    request_start: std::time::Instant,
    cached: DeterministicCompletionCacheValue,
) -> Response {
    let created = now_epoch();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_tokens = cached.completion_tokens;
    let thinking_budget_status = cached.thinking_budget_status;
    let cached_output = assistant_output_from_cached_parts(
        req,
        cached.text,
        cached.reasoning_content,
        cached.tool_calls,
        cached.finish_reason,
    );
    let preview_source = cached_output.preview_source();
    let mut thinking_budget_metadata =
        chat_completion_metadata_from_cached_output(state, req, &cached_output).thinking_budget;
    if let Some(status) = thinking_budget_status {
        thinking_budget_metadata.applied = true;
        apply_thinking_budget_status_to_metadata(&mut thinking_budget_metadata, status);
    }

    record_recent_request(
        state,
        RequestRecord {
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: cached_output.finish_reason.clone(),
            prefix_cache: Some("completion_cache_hit".to_string()),
            thinking_budget: Some(recent_thinking_budget_from_metadata(
                &thinking_budget_metadata,
            )),
            ..request_record_from_req(state, req, &id, &model, true)
        },
    );

    let mut events = Vec::with_capacity(5);
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
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
        }],
    };
    events.push(Event::default().data(serde_json::to_string(&role_chunk).unwrap()));

    if let Some(reasoning) = cached_output.reasoning_content {
        if !reasoning.is_empty() {
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
                        reasoning_content: Some(reasoning),
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            };
            events.push(Event::default().data(serde_json::to_string(&chunk).unwrap()));
        }
    }

    if let Some(tool_calls) = cached_output.tool_calls.as_deref() {
        if !cached_output.content.is_empty() {
            let chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(cached_output.content.clone()),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            };
            events.push(Event::default().data(serde_json::to_string(&chunk).unwrap()));
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
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(tool_call_deltas_from_openai_calls(tool_calls)),
                },
                finish_reason: None,
            }],
        };
        events.push(Event::default().data(serde_json::to_string(&chunk).unwrap()));
    } else if !cached_output.content.is_empty() {
        let chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(cached_output.content),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        events.push(Event::default().data(serde_json::to_string(&chunk).unwrap()));
    }

    let done_chunk = ChatCompletionChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: Some(cached_output.finish_reason),
        }],
    };
    events.push(Event::default().data(streaming_finish_chunk_json(
        &done_chunk,
        &thinking_budget_metadata,
        None,
    )));
    events.push(Event::default().data("[DONE]"));

    let stream = tokio_stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn empty_chat_completion_response(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_token_count: usize,
    request_start: std::time::Instant,
) -> ChatCompletionResponse {
    response_from_cached_completion(
        state,
        req,
        prompt_token_count,
        request_start,
        DeterministicCompletionCacheValue {
            text: String::new(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "length".to_string(),
            completion_tokens: 0,
            thinking_budget_status: None,
        },
    )
}

fn empty_chat_completion_streaming_response(
    state: &AppState,
    req: &ChatCompletionRequest,
    prompt_token_count: usize,
    request_start: std::time::Instant,
) -> Response {
    streaming_response_from_cached_completion(
        state,
        req,
        prompt_token_count,
        request_start,
        DeterministicCompletionCacheValue {
            text: String::new(),
            reasoning_content: None,
            tool_calls: None,
            finish_reason: "length".to_string(),
            completion_tokens: 0,
            thinking_budget_status: None,
        },
    )
}

fn cache_value_from_response(
    resp: &ChatCompletionResponse,
) -> Option<DeterministicCompletionCacheValue> {
    let choice = resp.choices.first()?;
    Some(DeterministicCompletionCacheValue {
        text: response_content_for_cache(
            &choice.message.content,
            choice.message.reasoning_content.as_deref(),
        ),
        reasoning_content: choice.message.reasoning_content.clone(),
        tool_calls: choice.message.tool_calls.clone(),
        finish_reason: choice.finish_reason.clone(),
        completion_tokens: resp.usage.completion_tokens,
        thinking_budget_status: choice.thinking_budget,
    })
}

fn store_deterministic_completion(
    state: &AppState,
    key: DeterministicCompletionCacheKey,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = cache_value_from_response(resp) else {
        return;
    };
    state
        .completion_cache
        .lock()
        .unwrap()
        .insert_complete_value(key, value);
}

fn complete_deterministic_completion_owner(
    state: &AppState,
    key: DeterministicCompletionCacheKey,
    claim_id: DeterministicCacheClaimId,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = cache_value_from_response(resp) else {
        state.completion_cache.lock().unwrap().fail(&key, claim_id);
        return;
    };
    state
        .completion_cache
        .lock()
        .unwrap()
        .complete(key, claim_id, value);
}

fn fail_deterministic_completion_owner(
    state: &AppState,
    key: &DeterministicCompletionCacheKey,
    claim_id: DeterministicCacheClaimId,
) {
    state.completion_cache.lock().unwrap().fail(key, claim_id);
}

async fn wait_for_deterministic_completion(
    mut receiver: tokio::sync::watch::Receiver<DeterministicCompletionInFlightState>,
) -> Option<DeterministicCompletionCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicCompletionInFlightState::Pending => {}
            DeterministicCompletionInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

struct ChatRequestCacheOwnerGuard {
    cache: std::sync::Arc<std::sync::Mutex<DeterministicChatRequestCache>>,
    key: DeterministicCacheKey,
    claim_id: DeterministicCacheClaimId,
    active: bool,
}

impl ChatRequestCacheOwnerGuard {
    fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicChatRequestCache>>,
        key: DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> Self {
        Self {
            cache,
            key,
            claim_id,
            active: true,
        }
    }

    fn complete(mut self, value: DeterministicChatRequestCacheValue) {
        self.cache
            .lock()
            .unwrap()
            .complete(self.key.clone(), self.claim_id, value);
        self.active = false;
    }

    fn matches_key(&self, key: &DeterministicCacheKey) -> bool {
        &self.key == key
    }
}

impl Drop for ChatRequestCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key, self.claim_id);
        }
    }
}

async fn wait_for_deterministic_chat_request(
    mut receiver: tokio::sync::watch::Receiver<DeterministicChatRequestInFlightState>,
) -> Option<DeterministicChatRequestCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicChatRequestInFlightState::Pending => {}
            DeterministicChatRequestInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

struct ChatChoicesCacheOwnerGuard {
    cache: std::sync::Arc<std::sync::Mutex<DeterministicChatChoicesCache>>,
    key: DeterministicCacheKey,
    claim_id: DeterministicCacheClaimId,
    active: bool,
}

impl ChatChoicesCacheOwnerGuard {
    fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicChatChoicesCache>>,
        key: DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> Self {
        Self {
            cache,
            key,
            claim_id,
            active: true,
        }
    }

    fn complete(mut self, value: DeterministicChatChoicesCacheValue) {
        self.cache
            .lock()
            .unwrap()
            .complete(self.key.clone(), self.claim_id, value);
        self.active = false;
    }

    fn matches_key(&self, key: &DeterministicCacheKey) -> bool {
        &self.key == key
    }
}

impl Drop for ChatChoicesCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key, self.claim_id);
        }
    }
}

async fn wait_for_deterministic_chat_choices(
    mut receiver: tokio::sync::watch::Receiver<DeterministicChatChoicesInFlightState>,
) -> Option<DeterministicChatChoicesCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicChatChoicesInFlightState::Pending => {}
            DeterministicChatChoicesInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

fn finish_chat_request_cache(
    state: &AppState,
    key: Option<DeterministicCacheKey>,
    owner: Option<ChatRequestCacheOwnerGuard>,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = chat_request_cache_value_from_response(resp) else {
        return;
    };
    finish_chat_request_cache_value(state, key, owner, value);
}

fn finish_chat_request_cache_value(
    state: &AppState,
    key: Option<DeterministicCacheKey>,
    owner: Option<ChatRequestCacheOwnerGuard>,
    value: DeterministicChatRequestCacheValue,
) {
    if let Some(owner) = owner {
        owner.complete(value);
    } else if let Some(key) = key {
        state.chat_request_cache.lock().unwrap().insert(key, value);
    }
}

fn finish_chat_choices_cache(
    state: &AppState,
    key: Option<DeterministicCacheKey>,
    owner: Option<ChatChoicesCacheOwnerGuard>,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = chat_choices_cache_value_from_response(resp) else {
        return;
    };
    if let Some(owner) = owner {
        owner.complete(value);
    } else if let Some(key) = key {
        state.chat_choices_cache.lock().unwrap().insert(key, value);
    }
}

struct BatchCacheOwnerGuard {
    cache: std::sync::Arc<std::sync::Mutex<DeterministicBatchCache>>,
    key: DeterministicBatchCacheKey,
    claim_id: DeterministicCacheClaimId,
    active: bool,
}

impl BatchCacheOwnerGuard {
    fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicBatchCache>>,
        key: DeterministicBatchCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> Self {
        Self {
            cache,
            key,
            claim_id,
            active: true,
        }
    }

    fn complete(mut self, value: DeterministicBatchCacheValue) {
        self.cache
            .lock()
            .unwrap()
            .complete(self.key.clone(), self.claim_id, value);
        self.active = false;
    }

    fn matches_key(&self, key: &DeterministicBatchCacheKey) -> bool {
        &self.key == key
    }
}

impl Drop for BatchCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key, self.claim_id);
        }
    }
}

async fn wait_for_deterministic_batch(
    mut receiver: tokio::sync::watch::Receiver<DeterministicBatchInFlightState>,
) -> Option<DeterministicBatchCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicBatchInFlightState::Pending => {}
            DeterministicBatchInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatAdapterSelection {
    /// The request omitted `adapter`; use the server default as-is.
    Default,
    /// The request explicitly selected base model for this request.
    Base,
    /// The request explicitly selected a named adapter for this request.
    Named(String),
}

impl Default for ChatAdapterSelection {
    fn default() -> Self {
        Self::Default
    }
}

impl ChatAdapterSelection {
    fn is_explicit(&self) -> bool {
        !matches!(self, Self::Default)
    }

    fn request_adapter_name(&self) -> Option<String> {
        match self {
            Self::Named(name) => Some(name.clone()),
            Self::Default | Self::Base => None,
        }
    }

    fn target_adapter_name(&self, default_adapter: Option<String>) -> Option<String> {
        match self {
            Self::Default => default_adapter,
            Self::Base => None,
            Self::Named(name) => Some(name.clone()),
        }
    }

    fn reason(&self) -> &'static str {
        match self {
            Self::Default => "chat_adapter_missing_use_default",
            Self::Base => "chat_adapter_explicit_base",
            Self::Named(_) => "chat_adapter_explicit_name",
        }
    }
}

fn deserialize_chat_adapter_selection<'de, D>(
    deserializer: D,
) -> Result<ChatAdapterSelection, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(match value {
        None => ChatAdapterSelection::Base,
        Some(name) if name.is_empty() => ChatAdapterSelection::Base,
        Some(name) => ChatAdapterSelection::Named(name),
    })
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// When omitted, the server falls back to its configured `served_model_id`.
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    /// Calling client's `User-Agent`, injected by the handler (never from the
    /// JSON body) so the /ui dashboard can attribute traffic per agent.
    #[serde(skip)]
    pub user_agent: Option<String>,
    /// First-party self-identification from the `X-Kiln-Client` header,
    /// injected by the handler (never from the JSON body). The /ui dashboard
    /// sends `dashboard` on its own traffic (Test connection, Playground,
    /// Compare) so onboarding milestones don't mistake it for an agent.
    #[serde(skip)]
    pub client: Option<String>,
    /// Number of completions to generate for this prompt. Defaults to 1.
    /// Non-streaming `n>1` reuses the same single-output fast paths below.
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Min-p sampling — drop tokens below `min_p * max_prob`. 0.0 = off.
    /// Qwen3.5's recommended default is 0.0 for all four sampling profiles.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// OpenAI-style presence penalty (-2.0 ..= 2.0). For each token id
    /// emitted at least once, subtract `presence_penalty` from its logit.
    /// Defaults to 1.5 (Qwen3.5 thinking-general recommendation) when
    /// omitted — set explicitly to 0.0 to disable.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// OpenAI-style frequency penalty (-2.0 ..= 2.0). For each token id
    /// in the generated prefix, subtract `frequency_penalty * count`
    /// from its logit. Default 0.0.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// HuggingFace-style repetition penalty (1.0 = no-op). Default 1.0.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Kiln extension: pick a Qwen3.5-recommended sampling profile in
    /// one shot. Recognized values: `"qwen3-thinking-general"`,
    /// `"qwen3-thinking-coding"`, `"qwen3-non-thinking-general"`,
    /// `"qwen3-non-thinking-reasoning"`, `"greedy"`. Explicit
    /// `temperature`/`top_p`/etc still override the preset.
    #[serde(default)]
    pub sampling_preset: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI newer-name alias for `max_tokens`. `max_tokens` wins when both
    /// are present to preserve existing request behavior.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// Kiln/vLLM-compatible extension: treat tokenizer EOS ids as ordinary
    /// generated tokens. Generation remains bounded by `max_tokens`, and
    /// explicit stop sequences still apply.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Kiln extension: maximum reasoning tokens before the server forces the
    /// tokenizer's `</think>` sequence into the model context. Omitted
    /// inherits the server default; `null` disables the token limit; `0`
    /// closes immediately.
    #[serde(default)]
    pub thinking_budget_tokens: BudgetOverride<usize>,
    /// Kiln extension: reasoning wall-clock budget in milliseconds. The clock
    /// begins at the first decode candidate, after queueing and prefill.
    /// Omitted inherits the server default; `null` disables the time limit.
    #[serde(default)]
    pub thinking_budget_ms: BudgetOverride<u64>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI `stream_options`. `include_usage: true` appends a final
    /// chunk (empty `choices`) carrying the request's token usage before
    /// `[DONE]` — agent clients meter cost from it.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, deserialize_with = "deserialize_optional_stop")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Kiln extension: which LoRA adapter to use for this request.
    ///
    /// Missing means "use the server default"; explicit `null` or `""` means
    /// "use base for this request only"; a non-empty string means "use this
    /// adapter for this request only".
    #[serde(default, deserialize_with = "deserialize_chat_adapter_selection")]
    pub adapter: ChatAdapterSelection,
    /// Kiln extension: stack multiple LoRA adapters with per-source scaling.
    /// Mutually exclusive with `adapter`. The composed adapter is merged once
    /// (via `merge_concat`) and cached on disk under `adapter_dir/.composed/`,
    /// keyed by a hash of the (name, scale) pairs.
    #[serde(default)]
    pub adapters: Option<Vec<AdapterRef>>,
    /// OpenAI-style tool/function definitions. Forwarded as opaque JSON into
    /// the chat template's Jinja context as `tools`. Templates that branch on
    /// `{% if tools %}` (e.g. Qwen3.5-4B's official template emits its
    /// `<tools>` schemas + tool-calling prelude only when this is set) require
    /// this round-trip — without it the model never sees the tool schemas at
    /// inference time and can't produce tool calls at all.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI-style `tool_choice` (`"none" | "auto" | "required"` or an object
    /// naming a specific tool). Accepted at the API edge so OpenAI clients
    /// don't see "unknown field" errors; threaded into the template context as
    /// `tool_choice` so HF templates that branch on it render correctly. Kiln
    /// itself does not enforce the choice at the sampler — that's caller
    /// responsibility for now.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Kiln extension: additional top-level variables forwarded into the
    /// HuggingFace Jinja chat-template context. This lets callers configure
    /// template-specific switches such as Qwen's `enable_thinking=false`
    /// without smuggling control flags through user-visible prompt text.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Kiln extension: duplicate separated reasoning text into `content` for
    /// compatibility with clients that treat empty content as no response.
    /// Defaults to the server setting, which is off by default.
    #[serde(default)]
    pub fold_reasoning_into_content: Option<bool>,
    /// Kiln extension: include request-scoped performance counters in the
    /// stable `metadata.performance` response field. An explicit `true` on a
    /// real-model stream also emits a distinct `kiln.token_timing` SSE object
    /// for each model token. When omitted, the server config default
    /// decides final-response metadata but does not opt the client into custom
    /// per-token SSE objects.
    #[serde(default)]
    pub include_performance: Option<bool>,
    /// Kiln extension: include model/tokenizer/template/config hashes in the
    /// stable `metadata.config_hashes` response field. When omitted, the
    /// server config default decides.
    #[serde(default)]
    pub include_config_hashes: Option<bool>,
    /// Kiln extension: capture an exact, behavior-policy-bound rollout record
    /// suitable for GRPO importance correction. This correctness-first path is
    /// non-streaming, single-choice, and requires the batching engine.
    #[serde(default)]
    pub rollout_provenance: bool,
}

/// A single source adapter for per-request composition.
#[derive(Debug, Deserialize)]
pub struct AdapterRef {
    pub name: String,
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(
        default,
        deserialize_with = "kiln_core::tokenizer::deserialize_chat_content"
    )]
    pub content: String,
    /// llama.cpp / DeepSeek-style chain-of-thought channel. Populated for
    /// reasoning models (Qwen3.5, DeepSeek R1, …) when the model emitted a
    /// `<think>...</think>` block; carries the inside of that block while
    /// `content` carries only the post-`</think>` answer. Skipped on the
    /// wire when empty so non-reasoning responses stay byte-identical to
    /// the OpenAI shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls emitted by the assistant on a prior turn (OpenAI shape:
    /// `[{"id": "call_…", "type": "function", "function": {"name": …,
    /// "arguments": "…"}}, …]`). Round-tripped into the chat template so
    /// multi-turn tool-use conversations render correctly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Function name on assistant messages with named function calls, OR the
    /// tool name on `role: "tool"` messages. Some templates branch on this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// On `role: "tool"` messages, identifies which assistant `tool_calls[*]`
    /// entry this message responds to. Required by OpenAI for multi-tool
    /// assistant turns; templates use it to pair the tool response with the
    /// originating tool call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Convert an API `Message` to the core tokenizer's `ChatMessage`, propagating
/// the OpenAI tool fields (`tool_calls`, `name`, `tool_call_id`) so the chat
/// template renders past assistant tool calls and `role: "tool"` responses.
fn message_to_chat(m: &Message) -> ChatMessage {
    ChatMessage {
        role: m.role.clone(),
        content: m.content.clone(),
        tool_calls: m
            .tool_calls
            .as_ref()
            .filter(|tool_calls| !tool_calls.is_empty())
            .cloned(),
        name: m.name.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }
}

/// Accept OpenAI `stop` as either a single string, an array of strings, `null`,
/// or missing. Internally the sampler and deterministic cache keys use a list.
fn deserialize_optional_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stop {
        One(String),
        Many(Vec<String>),
    }

    Ok(match Option::<Stop>::deserialize(deserializer)? {
        None => None,
        Some(Stop::One(stop)) => Some(vec![stop]),
        Some(Stop::Many(stops)) => Some(stops),
    })
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub metadata: ChatCompletionMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionMetadata {
    pub thinking_enabled: bool,
    pub thinking_mode: String,
    pub thinking_source: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_thinking_enabled: Option<bool>,
    pub final_content_empty: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_empty_reason: Option<&'static str>,
    pub reasoning_folded_into_content: bool,
    pub thinking_budget: ThinkingBudgetMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_hashes: Option<ConfigHashes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance: Option<ChatCompletionPerformanceMetadata>,
}

pub type ThinkingBudgetMetadata = ThinkingBudgetRecord;

/// Request-wide thinking-budget configuration and provenance. Batch outcomes
/// remain completion-specific and are reported on each completion item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ThinkingBudgetConfigurationMetadata {
    pub configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_time_ms: Option<u64>,
    pub tokens_source: ThinkingBudgetSource,
    pub time_source: ThinkingBudgetSource,
}

impl From<EffectiveThinkingBudget> for ThinkingBudgetConfigurationMetadata {
    fn from(effective: EffectiveThinkingBudget) -> Self {
        Self {
            configured: effective.configured(),
            max_tokens: effective.max_tokens,
            max_time_ms: effective.max_time_ms,
            tokens_source: effective.tokens_source,
            time_source: effective.time_source,
        }
    }
}

impl Default for ThinkingBudgetConfigurationMetadata {
    fn default() -> Self {
        Self {
            configured: false,
            max_tokens: None,
            max_time_ms: None,
            tokens_source: ThinkingBudgetSource::Unlimited,
            time_source: ThinkingBudgetSource::Unlimited,
        }
    }
}

fn serialize_optional_thinking_budget_status<S>(
    status: &Option<ThinkingBudgetStatus>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match status {
        Some(status) => ThinkingBudgetOutcome::from(status).serialize(serializer),
        None => serializer.serialize_none(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionPerformanceMetadata {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub actor_queue_ms: Option<f64>,
    pub actor_admission_ms: Option<f64>,
    pub actor_prefill_wall_ms: Option<f64>,
    pub resident_prefill_used: Option<bool>,
    pub decode_ms: Option<f64>,
    pub total_latency_ms: f64,
    pub decode_tokens_per_sec: Option<f64>,
    pub adapter_used: String,
    pub thinking_mode: String,
    pub finish_reason: String,
    pub latency: Option<RequestLatencyDiagnostics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_thinking_budget_status"
    )]
    pub thinking_budget: Option<ThinkingBudgetStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollout_provenance: Option<kiln_train::RolloutProvenanceV1>,
    #[serde(skip)]
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TextCompletionPrompt {
    Text(String),
    TokenIds(Vec<TokenId>),
}

const fn completion_add_special_tokens_default() -> bool {
    true
}

/// vLLM-shaped text-completions request subset.
///
/// Kiln supports this endpoint's prompt-logprobs mode so `RemoteTeacher`
/// clients can query a kiln-served teacher through the same `/v1/completions`
/// shape they use for vLLM/sglang. `prompt_logprobs` accepts `0..=256`. The
/// first position is `null`; every later position contains the observed prompt
/// token plus the requested top K, yielding K entries when the observed token
/// is already top K and K+1 otherwise. Extra observed tokens report their
/// full-vocabulary rank. Scores are F32 log-softmax values from the preceding
/// logits row, and token display uses only preceding actual prompt tokens to
/// complete split UTF-8 sequences.
///
/// This is a bounded, correctness-first subset: text prompts default to
/// tokenizer special-token insertion; `-1` all-vocabulary requests are not
/// accepted; candidate output is capped at 65,536 entries; and real scoring is
/// serialized under exclusive GPU admission. `model` must match the served
/// base model, and an active LoRA is rejected until an adapter content revision
/// can be pinned in the request and response identity. CUDA and ROCm validate
/// and select on device, transferring only O(TK) compact results. Vulkan,
/// Metal, and CPU retain the correctness-first O(TV) host fallback within the
/// same 64 MiB / 32-row projection-chunk bound.
///
/// Token display is fail closed: an unknown token ID or decoder failure returns
/// `tokenization_error` instead of a successful response containing an empty
/// fallback token. Model output is also validated before rendering;
/// vocabulary-width or non-finite-value corruption returns `generation_error`
/// rather than a short map or JSON `null` value.
#[derive(Debug, Deserialize)]
pub struct TextCompletionRequest {
    /// When omitted, the server falls back to its configured `served_model_id`.
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: TextCompletionPrompt,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub prompt_logprobs: Option<usize>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    /// vLLM-compatible text-prompt default. Ignored for token-ID prompts.
    #[serde(default = "completion_add_special_tokens_default")]
    pub add_special_tokens: bool,
}

#[derive(Debug, Serialize)]
pub struct TextCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    /// Canonical, content-addressed teacher identity. Mock responses retain
    /// the field as JSON null so clients cannot mistake a model alias for an
    /// authoritative identity.
    pub system_fingerprint: Option<String>,
    pub choices: Vec<TextCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct TextCompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<Vec<Option<PromptLogprobMap>>>,
}

pub type PromptLogprobMap = BTreeMap<String, PromptLogprobEntry>;

#[derive(Debug, Clone, Serialize)]
pub struct PromptLogprobEntry {
    pub logprob: f32,
    pub rank: usize,
    pub decoded_token: String,
}

fn completion_usage_tokens(
    visible_token_count: usize,
    finish_reason: &kiln_model::FinishReason,
) -> usize {
    visible_token_count + usize::from(matches!(finish_reason, kiln_model::FinishReason::Eos))
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
    /// Streaming counterpart of [`Message::reasoning_content`]. Each chunk
    /// emits at most one of `reasoning_content` (while inside a
    /// `<think>...</think>` block) or `content` (after the close tag).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
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

fn completion_system_fingerprint(state: &AppState) -> Result<Option<String>, ApiError> {
    canonical_completion_fingerprint(
        state.base_teacher_identity.as_deref(),
        matches!(state.backend.as_ref(), ModelBackend::Real { .. }),
    )
}

fn canonical_completion_fingerprint(
    identity: Option<&kiln_train::TeacherIdentityV1>,
    required: bool,
) -> Result<Option<String>, ApiError> {
    match (identity, required) {
        (Some(identity), _) => Ok(Some(identity.fingerprint())),
        (None, false) => Ok(None),
        (None, true) => Err(ApiError::internal(
            "real prompt-logprob backend has no verified base teacher identity; restart with a loader-owned model source revision",
        )),
    }
}

fn prompt_logprob_projection_chunk_tokens(vocab_size: usize) -> usize {
    let bytes_per_row = vocab_size
        .saturating_mul(2 * std::mem::size_of::<f32>())
        .max(1);
    (PROMPT_LOGPROB_PROJECTION_BYTE_BUDGET / bytes_per_row)
        .clamp(1, MAX_PROMPT_LOGPROB_PROJECTION_CHUNK_TOKENS)
}

fn validate_prompt_logprobs_top_k(state: &AppState, top_k: usize) -> Result<(), ApiError> {
    if top_k > MAX_COMPLETION_PROMPT_LOGPROBS {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt_logprobs {top_k} exceeds kiln's cap of {MAX_COMPLETION_PROMPT_LOGPROBS}"
        )));
    }
    if top_k > state.model_config.vocab_size {
        return Err(ApiError::completion_invalid_request(format!(
            "prompt_logprobs {top_k} exceeds vocab size {}",
            state.model_config.vocab_size
        )));
    }
    Ok(())
}

fn tokens_for_text_completion_prompt(
    state: &AppState,
    prompt: &TextCompletionPrompt,
    add_special_tokens: bool,
) -> Result<Vec<TokenId>, ApiError> {
    match prompt {
        TextCompletionPrompt::TokenIds(tokens) => Ok(tokens.clone()),
        TextCompletionPrompt::Text(text) => state
            .tokenizer
            .encode_with_special_tokens(text, add_special_tokens)
            .map_err(ApiError::tokenization_failed),
    }
}

fn decode_prompt_logprob_token(
    token_id: TokenId,
    preceding_actual_ids: &[TokenId],
    decode_token: impl FnOnce(TokenId, &[TokenId]) -> Result<String, TokenizerError>,
) -> Result<String, ApiError> {
    decode_token(token_id, preceding_actual_ids).map_err(|err| {
        ApiError::tokenization_failed(format!(
            "failed to render prompt-logprob token id {token_id}: {err}"
        ))
    })
}

#[derive(Clone, Copy)]
struct ValidatedPromptLogprobRow<'a> {
    values: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
struct CompactPromptLogprobEntry {
    token_id: TokenId,
    logprob: f32,
    rank: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct CompactPromptLogprobSelection {
    entries: Vec<CompactPromptLogprobEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PromptLogprobRankCandidate {
    token_id: TokenId,
    logit: f32,
}

impl Eq for PromptLogprobRankCandidate {}

impl Ord for PromptLogprobRankCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap::peek returns the greatest item. Reverse score order and
        // keep token-ID order forward so the root is the worst retained
        // candidate: lowest logit, then highest token ID.
        other
            .logit
            .partial_cmp(&self.logit)
            .expect("prompt-logprob rank candidates are validated finite")
            .then_with(|| self.token_id.cmp(&other.token_id))
    }
}

impl PartialOrd for PromptLogprobRankCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
fn top_k_logprob_map_with_decoder(
    logit_row: &[f32],
    logprob_row: &[f32],
    expected_vocab_size: usize,
    observed_token_id: TokenId,
    top_k: usize,
    preceding_actual_ids: &[TokenId],
    mut decode_token: impl FnMut(TokenId, &[TokenId]) -> Result<String, TokenizerError>,
) -> Result<PromptLogprobMap, ApiError> {
    let logit_row = validate_prompt_logprob_row(logit_row, expected_vocab_size)?;
    let logprob_row = validate_prompt_logprob_row(logprob_row, expected_vocab_size)?;
    let selection = select_prompt_logprobs_from_validated_rows(
        logit_row,
        logprob_row,
        observed_token_id,
        top_k,
    )?;
    prompt_logprob_map_from_selection_with_decoder(
        &selection,
        preceding_actual_ids,
        &mut decode_token,
    )
}

fn validate_prompt_logprob_row<'a>(
    row: &'a [f32],
    expected_vocab_size: usize,
) -> Result<ValidatedPromptLogprobRow<'a>, ApiError> {
    if row.len() != expected_vocab_size {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs row width {} did not match model vocabulary size {expected_vocab_size}",
            row.len()
        )));
    }

    if let Some((token_id, value)) = row.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs row contained non-finite value {value:?} at token id {token_id}"
        )));
    }

    Ok(ValidatedPromptLogprobRow { values: row })
}

fn select_prompt_logprobs_from_validated_rows(
    logit_row: ValidatedPromptLogprobRow<'_>,
    logprob_row: ValidatedPromptLogprobRow<'_>,
    observed_token_id: TokenId,
    top_k: usize,
) -> Result<CompactPromptLogprobSelection, ApiError> {
    let logits = logit_row.values;
    let logprobs = logprob_row.values;
    if logits.len() != logprobs.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs logits width {} did not match log-probability width {}",
            logits.len(),
            logprobs.len()
        )));
    }
    if top_k > logits.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "requested {top_k} prompt-logprob candidates from a vocabulary row with width {}",
            logits.len()
        )));
    }
    let observed_index = observed_token_id as usize;
    let observed_logit = logits.get(observed_index).copied().ok_or_else(|| {
        ApiError::generation_failed(anyhow::anyhow!(
            "observed prompt token id {observed_token_id} was outside vocabulary row width {}",
            logits.len()
        ))
    })?;
    let observed_logprob = logprobs[observed_index];

    if top_k == 0 {
        let observed_rank = logits
            .iter()
            .filter(|&&logit| logit >= observed_logit)
            .count();
        return Ok(CompactPromptLogprobSelection {
            entries: vec![CompactPromptLogprobEntry {
                token_id: observed_token_id,
                logprob: observed_logprob,
                rank: observed_rank,
            }],
        });
    }

    let compare_rank = |a: &(TokenId, f32), b: &(TokenId, f32)| {
        b.1.partial_cmp(&a.1)
            .expect("validated prompt-logprob rows contain only finite values")
            .then_with(|| a.0.cmp(&b.0))
    };
    let mut heap = std::collections::BinaryHeap::with_capacity(top_k);
    for (token_id, &logit) in logits.iter().enumerate() {
        let candidate = PromptLogprobRankCandidate {
            token_id: token_id as TokenId,
            logit,
        };
        if heap.len() < top_k {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|worst| {
            compare_rank(
                &(candidate.token_id, candidate.logit),
                &(worst.token_id, worst.logit),
            )
            .is_lt()
        }) {
            heap.pop();
            heap.push(candidate);
        }
    }
    let mut pairs = heap
        .into_iter()
        .map(|candidate| (candidate.token_id, candidate.logit))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by(compare_rank);

    let observed_top_rank = pairs
        .iter()
        .position(|(token_id, _)| *token_id == observed_token_id)
        .map(|rank| rank + 1);
    let observed_rank = observed_top_rank.unwrap_or_else(|| {
        logits
            .iter()
            .filter(|&&logit| logit >= observed_logit)
            .count()
    });

    // vLLM inserts the observed token first and then its top-K alternatives.
    // A duplicate observed token in top-K overwrites the value at the same key;
    // represent that final distinct-token result directly here.
    let mut entries = Vec::with_capacity(top_k + usize::from(observed_top_rank.is_none()));
    entries.push(CompactPromptLogprobEntry {
        token_id: observed_token_id,
        logprob: observed_logprob,
        rank: observed_rank,
    });
    for (rank, (token_id, _)) in pairs.into_iter().enumerate() {
        if token_id != observed_token_id {
            entries.push(CompactPromptLogprobEntry {
                token_id,
                logprob: logprobs[token_id as usize],
                rank: rank + 1,
            });
        }
    }

    let expected_len = top_k + usize::from(observed_top_rank.is_none());
    if entries.len() != expected_len {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs selection returned {} distinct candidates instead of {expected_len}",
            entries.len()
        )));
    }
    Ok(CompactPromptLogprobSelection { entries })
}

fn select_prompt_logprobs_from_device_row(
    row: &kiln_tensor::DevicePromptLogprobRow,
    expected_vocab_size: usize,
    observed_token_id: TokenId,
    top_k: usize,
) -> Result<CompactPromptLogprobSelection, ApiError> {
    if top_k > expected_vocab_size {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "requested {top_k} prompt-logprob candidates from a vocabulary row with width {expected_vocab_size}"
        )));
    }
    if observed_token_id as usize >= expected_vocab_size {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "observed prompt token id {observed_token_id} was outside vocabulary row width {expected_vocab_size}"
        )));
    }
    if row.candidates.len() != top_k {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob selection returned {} candidates instead of {top_k}",
            row.candidates.len()
        )));
    }
    if !row.observed_logit.is_finite() || !row.observed_logprob.is_finite() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob selection returned non-finite observed values"
        )));
    }
    if !(1..=expected_vocab_size).contains(&row.observed_full_rank) {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob observed rank {} was outside 1..={expected_vocab_size}",
            row.observed_full_rank
        )));
    }

    let observed_top_rank = row
        .candidates
        .iter()
        .position(|candidate| candidate.token_id == observed_token_id)
        .map(|rank| rank + 1);
    if let Some(rank) = observed_top_rank {
        let candidate = &row.candidates[rank - 1];
        if candidate.logit != row.observed_logit || candidate.logprob != row.observed_logprob {
            return Err(ApiError::generation_failed(anyhow::anyhow!(
                "device prompt-logprob observed candidate disagreed with observed row statistics"
            )));
        }
    }

    let observed_rank = observed_top_rank.unwrap_or(row.observed_full_rank);
    let mut entries = Vec::with_capacity(top_k + usize::from(observed_top_rank.is_none()));
    entries.push(CompactPromptLogprobEntry {
        token_id: observed_token_id,
        logprob: row.observed_logprob,
        rank: observed_rank,
    });
    for (rank, candidate) in row.candidates.iter().enumerate() {
        if candidate.token_id as usize >= expected_vocab_size {
            return Err(ApiError::generation_failed(anyhow::anyhow!(
                "device prompt-logprob candidate token id {} was outside vocabulary row width {expected_vocab_size}",
                candidate.token_id
            )));
        }
        if !candidate.logit.is_finite() || !candidate.logprob.is_finite() {
            return Err(ApiError::generation_failed(anyhow::anyhow!(
                "device prompt-logprob candidate at rank {} was non-finite",
                rank + 1
            )));
        }
        if candidate.token_id != observed_token_id {
            entries.push(CompactPromptLogprobEntry {
                token_id: candidate.token_id,
                logprob: candidate.logprob,
                rank: rank + 1,
            });
        }
    }

    let expected_len = top_k + usize::from(observed_top_rank.is_none());
    if entries.len() != expected_len {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "device prompt-logprob selection returned {} distinct candidates instead of {expected_len}",
            entries.len()
        )));
    }
    Ok(CompactPromptLogprobSelection { entries })
}

fn prompt_logprob_map_from_selection(
    state: &AppState,
    selection: &CompactPromptLogprobSelection,
    preceding_actual_ids: &[TokenId],
) -> Result<PromptLogprobMap, ApiError> {
    let mut decode_token = |token_id, context: &[TokenId]| {
        state
            .tokenizer
            .decode_token_for_display_with_context(token_id, context)
    };
    prompt_logprob_map_from_selection_with_decoder(
        selection,
        preceding_actual_ids,
        &mut decode_token,
    )
}

fn prompt_logprob_map_from_selection_with_decoder(
    selection: &CompactPromptLogprobSelection,
    preceding_actual_ids: &[TokenId],
    decode_token: &mut impl FnMut(TokenId, &[TokenId]) -> Result<String, TokenizerError>,
) -> Result<PromptLogprobMap, ApiError> {
    let mut out = BTreeMap::new();
    for entry in &selection.entries {
        out.insert(
            entry.token_id.to_string(),
            PromptLogprobEntry {
                logprob: entry.logprob,
                rank: entry.rank,
                decoded_token: decode_prompt_logprob_token(
                    entry.token_id,
                    preceding_actual_ids,
                    &mut *decode_token,
                )?,
            },
        );
    }
    if out.len() != selection.entries.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs rendering collapsed distinct token ids"
        )));
    }
    Ok(out)
}

fn prompt_logprobs_from_selections(
    state: &AppState,
    prompt_tokens: &[TokenId],
    selections: &[CompactPromptLogprobSelection],
    deadline: Option<tokio::time::Instant>,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let expected_selections = prompt_tokens.len().saturating_sub(1);
    if selections.len() != expected_selections {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs selection count {} did not match scored prompt position count {expected_selections}",
            selections.len()
        )));
    }

    let mut out = Vec::with_capacity(prompt_tokens.len());
    out.push(None);
    for (selection_index, selection) in selections.iter().enumerate() {
        if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
            return Err(ApiError::request_timeout(state.request_timeout.as_secs()));
        }
        let prompt_position = selection_index + 1;
        out.push(Some(prompt_logprob_map_from_selection(
            state,
            selection,
            &prompt_tokens[..prompt_position],
        )?));
    }
    Ok(out)
}

#[cfg(test)]
fn prompt_logprobs_from_rows(
    state: &AppState,
    prompt_tokens: &[TokenId],
    logit_rows: &[Vec<f32>],
    logprob_rows: &[Vec<f32>],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let expected_rows = prompt_tokens.len().saturating_sub(1);
    if logit_rows.len() != expected_rows || logprob_rows.len() != expected_rows {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs logits/log-probability row counts {}/{} did not match scored prompt position count {expected_rows}",
            logit_rows.len(),
            logprob_rows.len()
        )));
    }

    let validated_logits = logit_rows
        .iter()
        .map(|row| validate_prompt_logprob_row(row, state.model_config.vocab_size))
        .collect::<Result<Vec<_>, _>>()?;
    let validated_rows = logprob_rows
        .iter()
        .map(|row| validate_prompt_logprob_row(row, state.model_config.vocab_size))
        .collect::<Result<Vec<_>, _>>()?;

    let mut selections = Vec::with_capacity(prompt_tokens.len().saturating_sub(1));
    for pos in 1..prompt_tokens.len() {
        selections.push(select_prompt_logprobs_from_validated_rows(
            validated_logits[pos - 1],
            validated_rows[pos - 1],
            prompt_tokens[pos],
            top_k,
        )?);
    }
    prompt_logprobs_from_selections(state, prompt_tokens, &selections, None)
}

fn mock_prompt_logprobs(
    state: &AppState,
    prompt_tokens: &[TokenId],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let vocab_size = state.model_config.vocab_size.max(1);
    let mut selections = Vec::with_capacity(prompt_tokens.len().saturating_sub(1));
    for pos in 1..prompt_tokens.len() {
        // The mock distribution is score-descending in token-id order:
        // logprob(token_id) = -token_id. This keeps behavior deterministic
        // while exercising the same K/K+1 observed-token rule as real models.
        let observed_token_id = prompt_tokens[pos];
        let observed_in_top_k = (observed_token_id as usize) < top_k;
        let mut entries = Vec::with_capacity(top_k + usize::from(!observed_in_top_k));
        entries.push(CompactPromptLogprobEntry {
            token_id: observed_token_id,
            logprob: -(observed_token_id as f32),
            rank: observed_token_id as usize + 1,
        });
        for rank in 0..top_k.min(vocab_size) {
            let token_id = rank as TokenId;
            if token_id != observed_token_id {
                entries.push(CompactPromptLogprobEntry {
                    token_id,
                    logprob: -(token_id as f32),
                    rank: rank + 1,
                });
            }
        }
        selections.push(CompactPromptLogprobSelection { entries });
    }
    prompt_logprobs_from_selections(state, prompt_tokens, &selections, None)
}

fn prompt_logprob_tensor_rows_to_f32(
    tensor: &kiln_tensor::Tensor,
) -> anyhow::Result<Vec<Vec<f32>>> {
    match tensor.dtype() {
        kiln_tensor::DType::F32 => tensor.to_vec2::<f32>(),
        kiln_tensor::DType::BF16 => tensor.to_vec2::<half::bf16>().map(|rows| {
            rows.into_iter()
                .map(|row| row.into_iter().map(half::bf16::to_f32).collect())
                .collect()
        }),
        kiln_tensor::DType::F16 => tensor.to_vec2::<half::f16>().map(|rows| {
            rows.into_iter()
                .map(|row| row.into_iter().map(half::f16::to_f32).collect())
                .collect()
        }),
        dtype => anyhow::bail!("prompt-logprobs logits had unsupported dtype {dtype}"),
    }
    .context("prompt-logprobs tensor rows to F32 host values")
}

fn ensure_prompt_logprob_scoring_active(cancel: &CancelHandle) -> anyhow::Result<()> {
    if cancel.is_cancelled() {
        anyhow::bail!("prompt-logprobs scoring cancelled");
    }
    Ok(())
}

async fn validate_prompt_logprob_runner_admission(
    runner: &std::sync::RwLock<ModelRunner>,
    deadline: tokio::time::Instant,
    timeout: std::time::Duration,
) -> Result<(), ApiError> {
    loop {
        match runner.try_read() {
            Ok(runner_guard) => {
                runner_guard
                    .ensure_backend_healthy()
                    .map_err(ApiError::generation_failed)?;
                if runner_guard.active_lora().is_some() {
                    return Err(ApiError::completion_invalid_request(
                        "prompt-logprobs scoring is base-model only until adapter revision identity is pinned; unload the active adapter",
                    ));
                }
                return Ok(());
            }
            Err(std::sync::TryLockError::Poisoned(error)) => {
                return Err(ApiError::internal(format!(
                    "prompt-logprobs runner lock poisoned: {error}"
                )));
            }
            Err(std::sync::TryLockError::WouldBlock) => {}
        }

        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
        tokio::time::sleep_until(std::cmp::min(
            deadline,
            now + std::time::Duration::from_millis(1),
        ))
        .await;
    }
}

fn prompt_logprob_runner_read<'a>(
    runner: &'a std::sync::RwLock<ModelRunner>,
    cancel: &CancelHandle,
) -> anyhow::Result<std::sync::RwLockReadGuard<'a, ModelRunner>> {
    loop {
        ensure_prompt_logprob_scoring_active(cancel)?;
        match runner.try_read() {
            Ok(runner_guard) => return Ok(runner_guard),
            Err(std::sync::TryLockError::Poisoned(error)) => {
                anyhow::bail!("prompt-logprobs runner lock poisoned: {error}")
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

struct PromptLogprobWorkerOwnership {
    _gpu_guard: tokio::sync::OwnedRwLockWriteGuard<()>,
    linear_state: Option<kiln_model::forward::LinearAttentionState>,
    normalized_hidden: Option<kiln_tensor::Tensor>,
    hidden_chunk: Option<kiln_tensor::Tensor>,
    logits: Option<kiln_tensor::Tensor>,
    cpu_logits: Option<kiln_tensor::Tensor>,
    logits_2d: Option<kiln_tensor::Tensor>,
    log_probs: Option<kiln_tensor::Tensor>,
    log_probs_2d: Option<kiln_tensor::Tensor>,
}

impl PromptLogprobWorkerOwnership {
    fn new(gpu_guard: tokio::sync::OwnedRwLockWriteGuard<()>) -> Self {
        Self {
            _gpu_guard: gpu_guard,
            linear_state: None,
            normalized_hidden: None,
            hidden_chunk: None,
            logits: None,
            cpu_logits: None,
            logits_2d: None,
            log_probs: None,
            log_probs_2d: None,
        }
    }

    fn clear_completed_chunk(&mut self) {
        self.log_probs_2d = None;
        self.log_probs = None;
        self.logits_2d = None;
        self.cpu_logits = None;
        self.logits = None;
        self.hidden_chunk = None;
    }
}

fn run_prompt_logprob_worker_with_panic_fence<T, O>(
    backend_health: &kiln_model::BackendHealthHandle,
    mut ownership: O,
    work: impl FnOnce(&mut O) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&mut ownership))) {
        Ok(result) => {
            if backend_health.snapshot().quarantined {
                std::mem::forget(ownership);
            }
            result
        }
        Err(_) => {
            let reason = "prompt-logprobs scorer or settlement panicked; backend completion and GPU ownership are unknown";
            backend_health.quarantine(reason);
            std::mem::forget(ownership);
            Err(anyhow::anyhow!(reason))
        }
    }
}

fn score_real_prompt_logprob_rows(
    runner: &ModelRunner,
    prompt_tokens: &[TokenId],
    scored_tokens: &[TokenId],
    expected_vocab_size: usize,
    top_k: usize,
    cancel: &CancelHandle,
    metrics: &crate::metrics::Metrics,
    ownership: &mut PromptLogprobWorkerOwnership,
) -> anyhow::Result<Vec<CompactPromptLogprobSelection>> {
    runner.ensure_backend_healthy()?;
    if runner.active_lora().is_some() {
        anyhow::bail!("prompt-logprobs active adapter changed after base-model admission");
    }
    if prompt_tokens.len() != scored_tokens.len().saturating_add(1) {
        anyhow::bail!(
            "prompt-logprobs prompt/scored token counts {}/{} did not preserve the causal one-row offset",
            prompt_tokens.len(),
            scored_tokens.len()
        );
    }

    let device = runner.weights.device_kt();
    let backend = runner.backend_runtime();
    ownership.linear_state = Some(
        kiln_model::forward::LinearAttentionState::new_with_batch_for_inference_runtime(
            &runner.config,
            1,
            &device,
            backend,
        )?,
    );
    let normalized_hidden = kiln_model::forward::model_forward_no_head_with_policy(
        backend,
        scored_tokens,
        &runner.weights,
        &runner.config,
        ownership.linear_state.as_mut(),
        None,
        runner.streaming_prefill_policy(),
    )
    .context("prompt-logprobs forward pass")?;
    ownership.normalized_hidden = Some(normalized_hidden);
    ensure_prompt_logprob_scoring_active(cancel)?;

    let (batch_size, scored_sequence_len, hidden_size) = ownership
        .normalized_hidden
        .as_ref()
        .context("prompt-logprobs normalized hidden owner missing")?
        .dims3()
        .context("prompt-logprobs normalized hidden shape")?;
    if batch_size != 1
        || scored_sequence_len != scored_tokens.len()
        || hidden_size != runner.config.hidden_size
    {
        anyhow::bail!(
            "prompt-logprobs normalized hidden shape {:?} did not match expected [1, {}, {}]",
            ownership
                .normalized_hidden
                .as_ref()
                .context("prompt-logprobs normalized hidden owner missing")?
                .dims(),
            scored_tokens.len(),
            runner.config.hidden_size
        );
    }

    let mut selections = Vec::with_capacity(scored_sequence_len);
    let mut validated_row_count = 0usize;
    let projection_chunk_tokens = prompt_logprob_projection_chunk_tokens(expected_vocab_size);
    for chunk_start in (0..scored_sequence_len).step_by(projection_chunk_tokens) {
        ensure_prompt_logprob_scoring_active(cancel)?;
        let chunk_len = (scored_sequence_len - chunk_start).min(projection_chunk_tokens);
        ownership.hidden_chunk = Some(
            ownership
                .normalized_hidden
                .as_ref()
                .context("prompt-logprobs normalized hidden owner missing")?
                .narrow(1, chunk_start, chunk_len)
                .context("prompt-logprobs narrow normalized hidden chunk")?
                .contiguous()
                .context("prompt-logprobs normalized hidden chunk contiguous")?,
        );
        ownership.logits = Some(
            kiln_model::forward::model_forward_project_normalized_hidden(
                backend,
                ownership
                    .hidden_chunk
                    .as_ref()
                    .context("prompt-logprobs hidden chunk owner missing")?,
                &runner.weights,
            )
            .with_context(|| {
                format!(
                    "prompt-logprobs LM-head projection for rows {chunk_start}..{}",
                    chunk_start + chunk_len
                )
            })?,
        );
        let expected_logits_shape = [1, chunk_len, expected_vocab_size];
        let logits = ownership
            .logits
            .as_ref()
            .context("prompt-logprobs logits owner missing")?;
        if logits.dims() != expected_logits_shape {
            anyhow::bail!(
                "prompt-logprobs logits shape {:?} did not match expected {:?}",
                logits.dims(),
                expected_logits_shape
            );
        }

        ownership.logits_2d = Some(
            logits
                .squeeze(0)
                .context("prompt-logprobs squeeze logits batch dim")?,
        );
        let observed_token_ids = &prompt_tokens[chunk_start + 1..chunk_start + chunk_len + 1];
        let logits_2d = ownership
            .logits_2d
            .as_ref()
            .context("prompt-logprobs logits view owner missing")?;
        let device_rows: Option<Vec<kiln_tensor::DevicePromptLogprobRow>> = match logits_2d.device()
        {
            #[cfg(feature = "cuda")]
            kiln_tensor::Device::Cuda(_) => Some(
                kiln_tensor::cuda_prompt_logprobs(logits_2d, observed_token_ids, top_k)
                    .context("CUDA compact prompt-logprob selection")?,
            ),
            #[cfg(feature = "rocm")]
            kiln_tensor::Device::Rocm(_) => Some(
                kiln_tensor::rocm_prompt_logprobs(logits_2d, observed_token_ids, top_k)
                    .context("ROCm compact prompt-logprob selection")?,
            ),
            _ => None,
        };

        let selection_route;
        if let Some(device_rows) = device_rows {
            selection_route = crate::metrics::PromptLogprobSelectionRoute::CompactDevice;
            if device_rows.len() != chunk_len {
                anyhow::bail!(
                    "compact prompt-logprob selection returned {} rows instead of {chunk_len}",
                    device_rows.len()
                );
            }
            for (chunk_row, row) in device_rows.iter().enumerate() {
                ensure_prompt_logprob_scoring_active(cancel)?;
                selections.push(
                    select_prompt_logprobs_from_device_row(
                        row,
                        expected_vocab_size,
                        observed_token_ids[chunk_row],
                        top_k,
                    )
                    .map_err(|error| anyhow::anyhow!(error.message))?,
                );
                validated_row_count += 1;
            }
        } else {
            selection_route = crate::metrics::PromptLogprobSelectionRoute::BoundedHostFallback;
            // Selection and full rank use original logits. F32 log-softmax
            // values can collapse distinct far-tail logits after the LSE
            // subtraction, so ranking the rendered values is not equivalent.
            // Vulkan's log-softmax is host-backed; keep that result on CPU
            // instead of bouncing it back to the device and immediately out.
            let host_logit_rows = if matches!(logits.device(), kiln_tensor::Device::Vulkan(_)) {
                ownership.cpu_logits = Some(
                    logits
                        .to_device(kiln_tensor::Device::Cpu)
                        .context("prompt-logprobs Vulkan logits to host")?,
                );
                ownership.logits_2d = Some(
                    ownership
                        .cpu_logits
                        .as_ref()
                        .context("prompt-logprobs CPU logits owner missing")?
                        .squeeze(0)
                        .context("prompt-logprobs squeeze CPU logits batch dim")?,
                );
                let host_rows = prompt_logprob_tensor_rows_to_f32(
                    ownership
                        .logits_2d
                        .as_ref()
                        .context("prompt-logprobs logits view owner missing")?,
                )?;
                ownership.log_probs = Some(
                    kiln_tensor::ops::log_softmax_last_dim_f32(
                        ownership
                            .cpu_logits
                            .as_ref()
                            .context("prompt-logprobs CPU logits owner missing")?,
                    )
                    .context("prompt-logprobs host log_softmax_last_dim_f32")?,
                );
                host_rows
            } else {
                let host_rows = prompt_logprob_tensor_rows_to_f32(
                    ownership
                        .logits_2d
                        .as_ref()
                        .context("prompt-logprobs logits view owner missing")?,
                )?;
                ownership.log_probs = Some(
                    kiln_tensor::ops::log_softmax_last_dim_f32(logits)
                        .context("prompt-logprobs log_softmax_last_dim_f32")?,
                );
                host_rows
            };
            let log_probs = ownership
                .log_probs
                .as_ref()
                .context("prompt-logprobs log-probability owner missing")?;
            if log_probs.dims() != expected_logits_shape
                || log_probs.dtype() != kiln_tensor::DType::F32
            {
                anyhow::bail!(
                    "prompt-logprobs F32 log-softmax produced shape {:?} and dtype {:?}; expected {:?} and F32",
                    log_probs.dims(),
                    log_probs.dtype(),
                    expected_logits_shape
                );
            }
            ownership.log_probs_2d = Some(
                log_probs
                    .squeeze(0)
                    .context("prompt-logprobs squeeze log-probability batch dim")?,
            );
            let host_logprob_rows = ownership
                .log_probs_2d
                .as_ref()
                .context("prompt-logprobs log-probability view owner missing")?
                .to_vec2::<f32>()
                .context("prompt-logprobs F32 chunk to host")?;

            ensure_prompt_logprob_scoring_active(cancel)?;
            if host_logit_rows.len() != chunk_len || host_logprob_rows.len() != chunk_len {
                anyhow::bail!(
                    "prompt-logprobs host chunks returned logits/log-probability row counts {}/{} instead of {chunk_len}",
                    host_logit_rows.len(),
                    host_logprob_rows.len()
                );
            }
            for (chunk_row, (logit_row, logprob_row)) in
                host_logit_rows.iter().zip(&host_logprob_rows).enumerate()
            {
                ensure_prompt_logprob_scoring_active(cancel)?;
                let global_row = chunk_start + chunk_row;
                let validated_logits = validate_prompt_logprob_row(logit_row, expected_vocab_size)
                    .map_err(|error| anyhow::anyhow!(error.message))?;
                let validated_logprobs =
                    validate_prompt_logprob_row(logprob_row, expected_vocab_size)
                        .map_err(|error| anyhow::anyhow!(error.message))?;
                selections.push(
                    select_prompt_logprobs_from_validated_rows(
                        validated_logits,
                        validated_logprobs,
                        prompt_tokens[global_row + 1],
                        top_k,
                    )
                    .map_err(|error| anyhow::anyhow!(error.message))?,
                );
                validated_row_count += 1;
            }
        }
        // Host readback proves the requested values are available, but the
        // backend may own auxiliary stream/cache work. Use the repository's
        // explicit external-yield fence before dropping this chunk's device
        // owners and reusing their bounded memory budget.
        runner.synchronize_external_yield("prompt-logprobs projection chunk")?;
        metrics.record_prompt_logprob_selection(selection_route, chunk_len);
        ownership.clear_completed_chunk();
    }

    if validated_row_count != scored_sequence_len {
        anyhow::bail!(
            "prompt-logprobs validated {validated_row_count} rows instead of {scored_sequence_len}"
        );
    }
    Ok(selections)
}

async fn real_prompt_logprobs(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    prompt_tokens: &[TokenId],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    if prompt_tokens.len() == 1 {
        return Ok(vec![None]);
    }

    let timeout = state.request_timeout;
    let deadline = tokio::time::Instant::now() + timeout;
    let cancel = CancelHandle::new();
    let _cancel_on_drop = CancelOnDrop::new(cancel.clone());
    let backend_health = match state.backend.as_ref() {
        ModelBackend::Real { backend_health, .. } => backend_health.clone(),
        ModelBackend::Mock { .. } => {
            return Err(ApiError::internal(
                "real prompt-logprobs scorer received the mock backend",
            ));
        }
    };
    backend_health
        .ensure_healthy()
        .map_err(ApiError::backend_quarantined)?;
    validate_prompt_logprob_runner_admission(runner, deadline, timeout).await?;
    // Prompt scoring performs a full prefill plus vocabulary-wide projection.
    // Take exclusive GPU admission so concurrent scoring/generation/training
    // cannot multiply its bounded scratch residency into an OOM or a visible
    // mid-inference allocator stall.
    let gpu_guard = match tokio::time::timeout_at(
        deadline,
        gpu_coordination_write_guard_while_healthy_async(&state.gpu_lock, &backend_health),
    )
    .await
    {
        Ok(Ok(guard)) => guard,
        Ok(Err(error)) => return Err(ApiError::backend_quarantined(error)),
        Err(_) => return Err(ApiError::request_timeout(timeout.as_secs())),
    };

    let runner = runner.clone();
    let prompt_tokens_owned = prompt_tokens.to_vec();
    let scored_tokens = prompt_tokens[..prompt_tokens.len() - 1].to_vec();
    let expected_vocab_size = state.model_config.vocab_size;
    let cancel_inner = cancel.clone();
    let worker_backend_health = backend_health.clone();
    let metrics = std::sync::Arc::clone(&state.metrics);
    let handle = tokio::task::spawn_blocking(
        move || -> anyhow::Result<Vec<CompactPromptLogprobSelection>> {
            run_prompt_logprob_worker_with_panic_fence(
                &worker_backend_health,
                PromptLogprobWorkerOwnership::new(gpu_guard),
                |ownership| {
                    ensure_prompt_logprob_scoring_active(&cancel_inner)?;
                    let runner_guard = prompt_logprob_runner_read(&runner, &cancel_inner)?;
                    let scoring_result = score_real_prompt_logprob_rows(
                        &runner_guard,
                        &prompt_tokens_owned,
                        &scored_tokens,
                        expected_vocab_size,
                        top_k,
                        &cancel_inner,
                        &metrics,
                        ownership,
                    );

                    // Every ordinary result, including cancellation and
                    // validation failure, must settle submitted backend work
                    // before exclusive admission becomes reusable.
                    let synchronized =
                        runner_guard.synchronize_external_yield("prompt-logprobs scoring");
                    drop(runner_guard);
                    synchronized?;
                    scoring_result
                },
            )
        },
    );

    tokio::pin!(handle);
    let selections = match tokio::time::timeout_at(deadline, &mut handle).await {
        Ok(Ok(Ok(selections))) => selections,
        Ok(Ok(Err(err))) => return Err(ApiError::generation_failed(err)),
        Ok(Err(err)) => {
            backend_health.quarantine(format!(
                "prompt-logprobs blocking worker terminated without settlement: {err}"
            ));
            return Err(ApiError::internal(format!("join error: {err}")));
        }
        Err(_) => {
            cancel.cancel();
            if let Err(err) = handle.await {
                backend_health.quarantine(format!(
                    "timed-out prompt-logprobs worker terminated without settlement: {err}"
                ));
            }
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
    };
    prompt_logprobs_from_selections(state, prompt_tokens, &selections, Some(deadline))
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
async fn ensure_adapter(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    req_adapter: &ChatAdapterSelection,
    request_id: &str,
) -> Result<(), ApiError> {
    let target = req_adapter.target_adapter_name(state.active_adapter_name.read().unwrap().clone());
    ensure_runtime_adapter(state, runner, target, request_id, req_adapter.reason()).await
}

async fn ensure_batch_adapter(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    req_adapter: &Option<String>,
    request_id: &str,
) -> Result<(), ApiError> {
    let target = req_adapter
        .clone()
        .or_else(|| state.active_adapter_name.read().unwrap().clone());
    ensure_runtime_adapter(
        state,
        runner,
        target,
        request_id,
        if req_adapter.is_some() {
            "batch_adapter_explicit_name"
        } else {
            "batch_adapter_missing_use_default"
        },
    )
    .await
}

async fn ensure_runtime_adapter(
    state: &AppState,
    _runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    target_adapter: Option<String>,
    request_id: &str,
    reason: &str,
) -> Result<(), ApiError> {
    let current = state.loaded_adapter_name();
    if target_adapter == current {
        return Ok(());
    }

    let target = match target_adapter.clone() {
        Some(name) => {
            validate_compose_name(&name)?;
            if !state.adapter_dir.join(&name).exists() {
                return Err(ApiError::adapter_not_found(&name));
            }
            crate::adapter_swap::SwapTarget::Named(name)
        }
        None => crate::adapter_swap::SwapTarget::Base,
    };

    // The actual weight flip happens at the batching engine's
    // between-requests barrier (see `adapter_swap`), so a streaming
    // request can never continue mid-generation on different weights.
    crate::adapter_swap::swap_runtime_adapter(
        state,
        crate::adapter_swap::SwapRequest {
            target,
            content_changed: false,
            default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Preserve,
            reason: "per_request_adapter",
        },
    )
    .await
    .map_err(ApiError::adapter_load_failed)?;

    tracing::info!(
        request_id = %request_id,
        old_adapter = ?current,
        new_adapter = ?target_adapter,
        reason = reason,
        "adapter transition"
    );
    if state.eval_mode {
        tracing::warn!(
            request_id = %request_id,
            old_adapter = ?current,
            new_adapter = ?target_adapter,
            reason = reason,
            "adapter transition during eval mode"
        );
    }

    Ok(())
}

/// Disk handle for a composed adapter ready to be loaded.
#[derive(Debug, Clone)]
struct ComposedTarget {
    /// Stable cache name embedded in the loaded-adapter identity once swapped in,
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

type ResolvedCompositionSource = (String, f32, PathBuf, String);

/// Resolve every source and bind the cache key to its exact PEFT revision.
/// The caller owns the adapter mutation guard, so config and weight identities
/// cannot change between hashing and the merge loader opening them.
fn resolve_composition_sources(
    adapter_dir: &Path,
    adapters: &[(String, f32)],
) -> Result<(String, Vec<ResolvedCompositionSource>), ApiError> {
    use sha2::{Digest, Sha256};

    let mut sources = Vec::with_capacity(adapters.len());
    for (name, scale) in adapters {
        let path = adapter_dir.join(name);
        if !path.is_dir() {
            return Err(ApiError::adapter_not_found(name));
        }
        let content_revision = kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&path)
            .map_err(|error| {
                ApiError::adapter_merge_failed(format!(
                    "resolve exact source revision for '{}' at {}: {error:#}",
                    name,
                    path.display()
                ))
            })?
            .content_revision();
        sources.push((name.clone(), *scale, path, content_revision));
    }

    let mut canonical: Vec<_> = sources
        .iter()
        .map(|(name, scale, _, revision)| (name.as_str(), scale.to_bits(), revision.as_str()))
        .collect();
    canonical.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(b"kiln-composed-adapter-v2\0");
    for (name, scale_bits, revision) in canonical {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(scale_bits.to_le_bytes());
        hasher.update((revision.len() as u64).to_le_bytes());
        hasher.update(revision.as_bytes());
    }
    let hash = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((hash, sources))
}

/// Synthesize (or reuse the on-disk cache for) a composed adapter spec.
///
/// On first call for a given source-revision hash, loads each source adapter, runs
/// `merge_concat`, and writes the result under `<adapter_dir>/.composed/<hash>/`.
/// Source-adapter lookup uses the same single-segment path resolution as
/// `ensure_adapter`; missing sources surface as 404. Publication uses a hidden
/// staging directory and one rename, so a failed merge never leaves a cache
/// hit that looks complete.
async fn synthesize_composed_adapter_locked(
    adapter_dir: &Path,
    adapters: &[AdapterRef],
    _serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) -> Result<ComposedTarget, ApiError> {
    let adapter_dir = adapter_dir.to_path_buf();
    let adapters: Vec<_> = adapters
        .iter()
        .map(|source| (source.name.clone(), source.scale))
        .collect();
    tokio::task::spawn_blocking(move || {
        synthesize_composed_adapter_blocking(&adapter_dir, &adapters)
    })
    .await
    .map_err(|error| ApiError::internal(format!("join composed-adapter publisher: {error}")))?
}

/// Blocking half of [`synthesize_composed_adapter_locked`]. Keeping every
/// filesystem read and CPU merge in one blocking task prevents a large LoRA
/// fingerprint from stalling an async request worker.
fn synthesize_composed_adapter_blocking(
    adapter_dir: &Path,
    adapters: &[(String, f32)],
) -> Result<ComposedTarget, ApiError> {
    let (hash, source_paths) = resolve_composition_sources(adapter_dir, adapters)?;
    let active_name = format!("__composed:{hash}");
    let composed_root = adapter_dir.join(".composed");
    let cache_dir = composed_root.join(&hash);

    if cache_dir.exists() {
        if let Err(error) =
            kiln_model::lora_loader::LoraSourceIdentity::from_adapter_dir(&cache_dir)
        {
            tracing::warn!(
                cache_dir = %cache_dir.display(),
                error = %format!("{error:#}"),
                "discarding incomplete composed-adapter cache entry"
            );
            std::fs::remove_dir_all(&cache_dir).map_err(|remove_error| {
                ApiError::adapter_merge_failed(format!(
                    "remove incomplete composed cache {}: {remove_error}",
                    cache_dir.display()
                ))
            })?;
        } else {
            // Cache hit: refresh the directory's mtime so LRU eviction treats this
            // entry as recently used. Best-effort — a failure does not block the
            // request, and stale mtimes only mean slightly less-accurate LRU
            // ordering.
            let now = filetime::FileTime::from_system_time(std::time::SystemTime::now());
            if let Err(e) = filetime::set_file_mtime(&cache_dir, now) {
                tracing::warn!(
                    cache_dir = %cache_dir.display(),
                    error = %e,
                    "failed to refresh composed-cache mtime on hit (LRU may be slightly off)"
                );
            }
            return Ok(ComposedTarget {
                active_name,
                cache_dir,
            });
        }
    }

    std::fs::create_dir_all(&composed_root).map_err(|error| {
        ApiError::adapter_merge_failed(format!("creating composed-cache dir: {error}"))
    })?;
    let staging = tempfile::Builder::new()
        .prefix(".compose-tmp-")
        .tempdir_in(&composed_root)
        .map_err(|error| {
            ApiError::adapter_merge_failed(format!("creating composed-cache staging dir: {error}"))
        })?;
    let staging_output = staging.path().join("adapter");

    let mut loaded: Vec<(PeftLora, f32)> = Vec::with_capacity(source_paths.len());
    for (name, scale, path, _revision) in source_paths {
        let adapter = PeftLora::load(&path).map_err(|error| {
            ApiError::adapter_merge_failed(format!(
                "loading source '{name}' from {}: {error}",
                path.display()
            ))
        })?;
        loaded.push((adapter, scale));
    }

    let refs: Vec<(&PeftLora, f32)> = loaded
        .iter()
        .map(|(adapter, scale)| (adapter, *scale))
        .collect();
    let merged = merge_concat(&refs)
        .map_err(|error| ApiError::adapter_merge_failed(format!("merge_concat: {error}")))?;
    merged.save(&staging_output).map_err(|error| {
        ApiError::adapter_merge_failed(format!("saving composed adapter: {error}"))
    })?;
    std::fs::rename(&staging_output, &cache_dir).map_err(|error| {
        ApiError::adapter_merge_failed(format!(
            "publishing composed adapter {}: {error}",
            cache_dir.display()
        ))
    })?;

    Ok(ComposedTarget {
        active_name,
        cache_dir,
    })
}

/// LRU-evict entries from `composed_root` until total entries `<= max_entries`
/// and total bytes `<= max_bytes`. Either bound being `None` disables that
/// dimension; if both are `None` the function is a no-op.
///
/// Eviction is best-effort: individual `remove_dir_all` failures are logged
/// and the loop continues. Hidden / non-directory entries (anything whose
/// name starts with `.`) are skipped — kiln only writes hash-named
/// subdirectories under `.composed/`, but a stray file should not be picked
/// for eviction.
///
/// Closes audit LOW §8 / roadmap item 8 (PR #620 capped uploaded adapters
/// but explicitly excluded this cache pending this LRU pass).
fn evict_composed_cache_lru(
    composed_root: &Path,
    max_bytes: Option<u64>,
    max_entries: Option<u64>,
    protected: &Path,
    _serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) {
    if max_bytes.is_none() && max_entries.is_none() {
        return;
    }
    let read_dir = match std::fs::read_dir(composed_root) {
        Ok(rd) => rd,
        Err(_) => return, // Parent gone or unreadable — nothing to evict.
    };

    // Gather (path, mtime, size) for each cache entry. `mtime` is read via
    // `std::fs::Metadata::modified()`; if unavailable we fall back to
    // `UNIX_EPOCH` so the entry sorts as oldest and gets evicted first.
    let mut entries: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
    let mut total_bytes: u64 = 0;
    for entry in read_dir.flatten() {
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy();
        // Skip hidden / sentinel files (names starting with `.`). All real
        // entries are 64-hex-digit revision hashes.
        if name_lossy.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let meta = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.file_type().is_dir() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        let size = composed_entry_size_bytes(&path);
        total_bytes = total_bytes.saturating_add(size);
        entries.push((path, mtime, size));
    }

    // Oldest first.
    entries.sort_by(|a, b| a.1.cmp(&b.1));

    let mut total_entries = entries.len() as u64;
    let mut iter = entries.into_iter();
    while (max_entries.is_some_and(|cap| total_entries > cap))
        || (max_bytes.is_some_and(|cap| total_bytes > cap))
    {
        let (path, _mtime, size) = match iter.next() {
            Some(e) => e,
            None => break, // Caps still exceeded but nothing left to evict.
        };
        if path == protected {
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                total_entries = total_entries.saturating_sub(1);
                total_bytes = total_bytes.saturating_sub(size);
                tracing::info!(
                    evicted = %path.display(),
                    freed_bytes = size,
                    "composed-adapter cache LRU eviction"
                );
            }
            Err(e) => {
                tracing::warn!(
                    cache_dir = %path.display(),
                    error = %e,
                    "failed to evict composed-cache entry (will retry next eviction)"
                );
                // Don't decrement — couldn't free this one.
            }
        }
    }
}

/// Recursively sum regular-file byte sizes under a composed-cache entry.
/// Mirrors the conservative best-effort spirit of
/// `dir_size_recursive` in `api/adapters.rs` — symlinks and stat errors
/// count as zero.
fn composed_entry_size_bytes(root: &Path) -> u64 {
    let meta = match std::fs::symlink_metadata(root) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    if meta.file_type().is_file() {
        return meta.len();
    }
    if !meta.file_type().is_dir() {
        return 0;
    }
    let read_dir = match std::fs::read_dir(root) {
        Ok(rd) => rd,
        Err(_) => return 0,
    };
    let mut total: u64 = 0;
    for entry in read_dir.flatten() {
        total = total.saturating_add(composed_entry_size_bytes(&entry.path()));
    }
    total
}

/// Hot-swap the runner onto a synthesized composed adapter.
///
/// Same barrier semantics as `ensure_runtime_adapter` — composed names are
/// content-hashed (`__composed:<hash>`), so they never alias stale cache
/// entries and `content_changed` stays false. No-op if already active.
async fn ensure_composed_adapter_swap_locked(
    state: &AppState,
    target: &ComposedTarget,
    serial: &crate::adapter_swap::AdapterMutationGuard<'_>,
) -> Result<(), ApiError> {
    {
        let current = state.loaded_adapter_identity();
        if current.as_ref().map(|identity| identity.name.as_str())
            == Some(target.active_name.as_str())
        {
            return Ok(());
        }
    }

    crate::adapter_swap::swap_runtime_adapter_locked(
        state,
        crate::adapter_swap::SwapRequest {
            target: crate::adapter_swap::SwapTarget::Resolved {
                active_name: target.active_name.clone(),
                dir: target.cache_dir.clone(),
            },
            content_changed: false,
            default_adapter: crate::adapter_swap::DefaultAdapterUpdate::Preserve,
            reason: "composed_adapter",
        },
        serial,
    )
    .await
    .map_err(ApiError::adapter_load_failed)?;
    if state.eval_mode {
        tracing::warn!(
            adapter = %target.active_name,
            "composed adapter transition during eval mode"
        );
    }

    Ok(())
}

/// Resolve, synthesize, publish, load, and evict a composed adapter while one
/// mutation guard covers every disk and loaded-weight transition.
async fn ensure_composed_adapter_for_request(
    state: &AppState,
    adapters: &[AdapterRef],
) -> Result<(), ApiError> {
    let serial = crate::adapter_swap::adapter_mutation_guard(state)
        .await
        .map_err(ApiError::adapter_load_failed)?;
    let target = synthesize_composed_adapter_locked(&state.adapter_dir, adapters, &serial).await?;
    if matches!(state.backend.as_ref(), ModelBackend::Real { .. }) {
        ensure_composed_adapter_swap_locked(state, &target, &serial).await?;
    }
    evict_composed_cache_lru(
        &state.adapter_dir.join(".composed"),
        state.composed_cache_max_bytes,
        state.composed_cache_max_entries,
        &target.cache_dir,
        &serial,
    );
    Ok(())
}

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
async fn generate_real_batched(
    state: &AppState,
    batching_engine: &crate::batching_engine::BatchingEngineHandle,
    adapter: Option<LoadedAdapterIdentity>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    tokenization_duration: Option<std::time::Duration>,
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_token_count = prompt_tokens.len();
    let request_id = Uuid::new_v4();
    let cancel = CancelHandle::with_prefill_progress_gauge(
        state.metrics.request_prefill_tokens_completed.clone(),
    );
    let mut events = batching_engine
        .enqueue(EngineRequest {
            request_id,
            prompt_tokens: prompt_tokens.to_vec(),
            sampling: sampling.clone(),
            adapter: adapter.clone(),
            capture_behavior_logprobs: req.rollout_provenance,
            cancel: cancel.clone(),
        })
        .await
        .map_err(ApiError::generation_failed)?;

    let timeout = state.request_timeout;
    let mut first_token_at: Option<std::time::Instant> = None;
    let mut latency_tracker = RequestLatencyTracker::new(request_start, tokenization_duration);
    let collect = async {
        loop {
            match events.recv().await {
                Some(EngineEvent::Token { timing, .. }) => {
                    first_token_at.get_or_insert(timing.ready_at);
                    let observed_at = std::time::Instant::now();
                    if let Some(gap) = latency_tracker.record_token(timing, observed_at) {
                        state.metrics.observe_token_gap(gap);
                        if let Ok(mut stats) = state.decode_stats.lock() {
                            stats.record_gap(observed_at, gap);
                        }
                    }
                }
                Some(EngineEvent::Done { output }) => break Ok(output),
                Some(EngineEvent::Error(err)) => break Err(anyhow::anyhow!(err)),
                None => break Err(anyhow::anyhow!("batching engine response channel closed")),
            }
        }
    };

    let output = match tokio::time::timeout(timeout, collect).await {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => {
            cancel.clear_prefill_progress();
            tracing::error!(error = %format!("{err:#}"), "batched real generation failed");
            return Err(ApiError::generation_failed(err));
        }
        Err(_) => {
            cancel.cancel();
            let _ = batching_engine.cancel(request_id).await;
            cancel.clear_prefill_progress();
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
    };
    cancel.clear_prefill_progress();
    state
        .metrics
        .observe_prefill_duration(output.prefill_duration.as_secs_f64());
    state
        .metrics
        .observe_decode_duration(output.decode_duration.as_secs_f64());
    observe_post_prefill_vram(&state.memory_budget, state.vram_probe_selector);

    let actor_queue_duration = output.actor_queue_duration;
    let actor_admission_duration = output.actor_admission_duration;
    let actor_prefill_wall_duration = output.actor_prefill_wall_duration;
    let latency_diagnostics = latency_tracker.diagnostics();
    let finish_reason = match &output.finish_reason {
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
    let completion_tokens = output.completion_tokens;
    let thinking_budget_status =
        finalized_thinking_budget_status(sampling.thinking_budget.as_ref(), completion_tokens);
    let assistant_output = assistant_output_from_model_output_stop_aware(
        req,
        &output.text,
        prompt_text,
        finish_reason,
        &output.finish_reason,
    );
    let metadata =
        chat_completion_metadata_from_prompt_and_output(state, req, prompt_text, &assistant_output);
    let assistant_output = apply_reasoning_content_policy(
        assistant_output,
        fold_reasoning_into_content_for_request(state, req),
    );
    let rollout_provenance = if req.rollout_provenance {
        Some(build_rollout_provenance(
            state,
            req,
            adapter.as_ref(),
            prompt_tokens,
            sampling,
            &output,
            &assistant_output.content,
        )?)
    } else {
        None
    };
    let preview_source = assistant_output.preview_source();
    let ttft = first_token_at.map(|instant| instant.duration_since(request_start));
    record_recent_request(
        state,
        RequestRecord {
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: assistant_output.finish_reason.clone(),
            ttft_ms: ttft.map(duration_ms_u64),
            model_prefill_ms: Some(duration_ms_u64(output.prefill_duration)),
            model_decode_ms: Some(duration_ms_u64(output.decode_duration)),
            thinking_mode: Some(thinking_mode_for_prompt(prompt_text).to_string()),
            prefix_cache: Some("batching_engine".to_string()),
            thinking_budget: Some(recent_thinking_budget_with_status(
                &metadata.thinking_budget,
                thinking_budget_status,
            )),
            latency: Some(latency_diagnostics.clone()),
            ..request_record_from_req(state, req, &id, &model, false)
        },
    );

    let finish_reason = assistant_output.finish_reason.clone();
    let mut response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: assistant_output.content,
                reasoning_content: assistant_output.reasoning_content,
                tool_calls: assistant_output.tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason,
            thinking_budget: None,
            rollout_provenance,
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    attach_thinking_budget_outcome(thinking_budget_status, &mut response);
    attach_chat_performance_metadata(
        state,
        req,
        &mut response,
        request_start,
        ttft,
        Some(output.prefill_duration),
        Some(output.decode_duration),
    );
    attach_batched_actor_performance_metadata(
        &mut response,
        actor_queue_duration,
        actor_admission_duration,
        actor_prefill_wall_duration,
        output.resident_prefill_used,
    );
    if let Some(performance) = response.metadata.performance.as_mut() {
        performance.latency = Some(latency_diagnostics);
    }
    Ok(response)
}

#[derive(Debug, Default)]
enum StreamTerminalState {
    #[default]
    Pending,
    Complete(std::collections::VecDeque<Event>),
    Failed(String),
    Consumed,
}

#[derive(Clone, Default)]
struct StreamTerminal {
    state: std::sync::Arc<std::sync::Mutex<StreamTerminalState>>,
}

impl StreamTerminal {
    fn fail(&self, message: impl Into<String>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, StreamTerminalState::Pending) {
            *state = StreamTerminalState::Failed(message.into());
        }
    }

    fn complete(&self, events: std::collections::VecDeque<Event>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, StreamTerminalState::Pending) {
            *state = StreamTerminalState::Complete(events);
        }
    }

    fn take_events(&self) -> std::collections::VecDeque<Event> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match std::mem::replace(&mut *state, StreamTerminalState::Consumed) {
            StreamTerminalState::Pending => stream_generation_error_events(
                "streaming response producer ended without a terminal state".to_string(),
            ),
            StreamTerminalState::Failed(message) => stream_generation_error_events(message),
            StreamTerminalState::Complete(events) => events,
            StreamTerminalState::Consumed => std::collections::VecDeque::new(),
        }
    }
}

fn stream_generation_error_events(message: String) -> std::collections::VecDeque<Event> {
    let payload = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "code": "generation_error"
        }
    });
    [
        Event::default().data(payload.to_string()),
        Event::default().data("[DONE]"),
    ]
    .into()
}

fn stream_with_terminal(
    rx: tokio::sync::mpsc::Receiver<Event>,
    terminal: StreamTerminal,
) -> impl futures::Stream<Item = Result<Event, std::convert::Infallible>> {
    futures::stream::unfold(
        (rx, terminal, std::collections::VecDeque::new()),
        |(mut rx, terminal, mut pending)| async move {
            if let Some(event) = pending.pop_front() {
                return Some((Ok(event), (rx, terminal, pending)));
            }

            match rx.recv().await {
                Some(event) => Some((Ok(event), (rx, terminal, pending))),
                None => {
                    pending = terminal.take_events();
                    pending
                        .pop_front()
                        .map(|event| (Ok(event), (rx, terminal, pending)))
                }
            }
        },
    )
}

// A success tail currently emits at most eight events (two residual chunks,
// two splitter-flush chunks, one tool/content chunk, finish, usage, and DONE).
// Keep headroom so terminal construction can never wait on an undrained queue.
const STREAM_TERMINAL_EVENT_CAPACITY: usize = 16;

fn drain_terminal_event_buffer(
    mut rx: tokio::sync::mpsc::Receiver<Event>,
) -> std::collections::VecDeque<Event> {
    let mut events = std::collections::VecDeque::new();
    while let Ok(event) = rx.try_recv() {
        events.push_back(event);
    }
    events
}

async fn generate_real_batched_streaming(
    state: &AppState,
    batching_engine: &crate::batching_engine::BatchingEngineHandle,
    adapter: Option<LoadedAdapterIdentity>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    tokenization_duration: Option<std::time::Duration>,
) -> Result<Response, ApiError> {
    let prompt_token_count = prompt_tokens.len();
    let request_id = Uuid::new_v4();
    let cancel = CancelHandle::with_prefill_progress_gauge(
        state.metrics.request_prefill_tokens_completed.clone(),
    );
    let events = batching_engine
        .enqueue(EngineRequest {
            request_id,
            prompt_tokens: prompt_tokens.to_vec(),
            sampling: sampling.clone(),
            adapter,
            capture_behavior_logprobs: false,
            cancel: cancel.clone(),
        })
        .await
        .map_err(ApiError::generation_failed)?;
    tracing::info!(
        event = "stream_request_bound",
        request_id = %request_id,
        client = req.client.as_deref().unwrap_or(""),
        "stream_request_bound"
    );

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_epoch();
    let timeout = state.request_timeout;
    let tokenizer = state.tokenizer.clone();
    let metrics = state.metrics.clone();
    let decode_stats = state.decode_stats.clone();
    let state_for_record = state.clone();
    let prompt_text_full = last_user_message_text(req);
    let prompt_preview = truncate_chars(&prompt_text_full, PROMPT_PREVIEW_MAX_CHARS);
    let prompt_full = truncate_chars(&prompt_text_full, FULL_BODY_MAX_CHARS);
    let req_adapter = req.adapter.request_adapter_name();
    let req_user_agent = req.user_agent.clone();
    let req_client = req.client.clone();
    let req_temperature = req.temperature;
    let req_top_p = req.top_p;
    let req_max_tokens = Some(chat_request_max_tokens(req).min(u32::MAX as usize) as u32);
    let thinking_mode = thinking_mode_for_prompt(prompt_text).to_string();
    let prompt_starts_in_reasoning = prompt_starts_in_reasoning(prompt_text);
    let buffer_tool_content = request_allows_tool_call_parsing(req);
    let mut tool_gate = ToolCallGate::new(buffer_tool_content);
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let include_token_timing = streaming_token_timing_enabled(req);
    let include_performance = chat_performance_metadata_enabled(state, req);
    let performance_adapter_used = adapter_used_for_performance_metadata(state);
    let batching_engine = batching_engine.clone();
    let stop_sequences = sampling.stop.clone();
    let thinking_budget = sampling.thinking_budget.clone();
    let stream_thinking_budget_metadata =
        thinking_budget_metadata_for_request(state, req, prompt_starts_in_reasoning);
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);
    let stream_terminal = StreamTerminal::default();
    let stream_terminal_for_task = stream_terminal.clone();

    tokio::task::spawn({
        let id = completion_id.clone();
        let model = model.clone();
        async move {
            let _prefill_progress_guard = PrefillProgressGuard::new(cancel.clone());
            let mut events = events;
            let mut completion_buf = String::new();
            let mut reasoning_buf = String::new();
            let mut content_buf = String::new();
            let mut completion_token_count: u32 = 0;
            let mut generated_tokens: Vec<TokenId> = Vec::new();
            let mut first_token_ready_at: Option<std::time::Instant> = None;
            let latency_tracker = std::sync::Mutex::new(RequestLatencyTracker::new(
                request_start,
                tokenization_duration,
            ));
            // Server-side emit gates for the engine path (the engine emits
            // raw token ids; the server decodes): incremental detokenizer
            // (no U+FFFD on multi-byte chars, bounded decode window instead
            // of the old O(n²) full-prefix re-decode) + stop holdback (the
            // matched stop never reaches the wire).
            let mut detok = kiln_model::stream_text::IncrementalDetokenizer::new();
            let mut stop_gate = kiln_model::stream_text::StopTailGate::new(&stop_sequences);
            let record_error = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
            let record_error_for_record = record_error.clone();
            let record = |finish_reason: String, completion: &str, completion_tokens: u32| {
                let latency = latency_tracker.lock().unwrap().diagnostics();
                let error =
                    record_error_for_record
                        .lock()
                        .unwrap()
                        .clone()
                        .or_else(|| match finish_reason.as_str() {
                            "error" => Some("streaming generation failed".to_string()),
                            "timeout" => Some("streaming generation timed out".to_string()),
                            _ => None,
                        });
                let record = RequestRecord {
                    user_agent: req_user_agent.clone(),
                    client: req_client.clone(),
                    id: id.clone(),
                    timestamp_unix_ms: now_unix_ms(),
                    model: model.clone(),
                    prompt_preview: prompt_preview.clone(),
                    prompt_full: Some(prompt_full.clone()),
                    completion_preview: truncate_chars(completion, COMPLETION_PREVIEW_MAX_CHARS),
                    completion_full: Some(truncate_chars(completion, FULL_BODY_MAX_CHARS)),
                    prompt_tokens: prompt_token_count as u32,
                    completion_tokens,
                    duration_ms: request_start.elapsed().as_millis() as u64,
                    streamed: true,
                    finish_reason,
                    thinking_mode: Some(thinking_mode.clone()),
                    prefix_cache: Some("batching_engine".to_string()),
                    adapter: req_adapter.clone(),
                    temperature: req_temperature,
                    top_p: req_top_p,
                    max_tokens: req_max_tokens,
                    ttft_ms: latency
                        .ttft_ms
                        .map(|milliseconds| milliseconds.min(u64::MAX as f64) as u64),
                    model_prefill_ms: None,
                    model_decode_ms: None,
                    error,
                    thinking_budget: Some(recent_thinking_budget_with_status(
                        &stream_thinking_budget_metadata,
                        finalized_thinking_budget_status(
                            thinking_budget.as_ref(),
                            completion_tokens as usize,
                        ),
                    )),
                    latency: Some(latency),
                };
                record_recent_request(&state_for_record, record);
            };

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
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                }],
            };
            if tx
                .send(Event::default().data(serde_json::to_string(&role_chunk).unwrap()))
                .await
                .is_err()
            {
                cancel.cancel();
                let _ = batching_engine.cancel(request_id).await;
                record(
                    "client_disconnect".to_string(),
                    &completion_buf,
                    completion_token_count,
                );
                return;
            }

            let mut reasoning_splitter = ReasoningSplitter::new(prompt_starts_in_reasoning);
            let deadline =
                tokio::time::Instant::now() + timeout.saturating_sub(request_start.elapsed());
            // Initial `false` is dead under the current control flow —
            // every non-timeout branch returns from the closure rather
            // than breaking the loop, so `timed_out` is only ever read
            // *after* the deadline arm assigns `true`. Kept for clarity.
            #[allow(unused_assignments)]
            let mut timed_out = false;

            loop {
                if tx.is_closed() {
                    cancel.cancel();
                    let _ = batching_engine.cancel(request_id).await;
                    record(
                        "client_disconnect".to_string(),
                        &completion_buf,
                        completion_token_count,
                    );
                    return;
                }

                tokio::select! {
                    _ = tx.closed() => {
                        cancel.cancel();
                        let _ = batching_engine.cancel(request_id).await;
                        record(
                            "client_disconnect".to_string(),
                            &completion_buf,
                            completion_token_count,
                        );
                        return;
                    }
                    event = events.recv() => {
                        match event {
                            Some(EngineEvent::Token { token, timing }) => {
                                let handler_received_at = std::time::Instant::now();
                                first_token_ready_at.get_or_insert(timing.ready_at);
                                generated_tokens.push(token);
                                completion_token_count = completion_token_count.saturating_add(1);
                                metrics.add_tokens(1);
                                let gap = latency_tracker
                                    .lock()
                                    .unwrap()
                                    .record_token(timing, handler_received_at);
                                if let Some(gap) = gap {
                                    metrics.observe_token_gap(gap);
                                    if let Ok(mut stats) = decode_stats.lock() {
                                        stats.record_gap(handler_received_at, gap);
                                    }
                                }

                                let delta = match detok.next_delta(&tokenizer, &generated_tokens) {
                                    Ok(delta) => delta,
                                    Err(error) => {
                                        let error =
                                            format!("streaming detokenization failed: {error}");
                                        *record_error.lock().unwrap() = Some(error.clone());
                                        stream_terminal_for_task.fail(error.clone());
                                        cancel.cancel();
                                        let _ = batching_engine.cancel(request_id).await;
                                        let tail = flush_buffered_stream_tail(
                                            &tx,
                                            &id,
                                            created,
                                            &model,
                                            &mut reasoning_splitter,
                                            &mut completion_buf,
                                            &mut reasoning_buf,
                                            &mut content_buf,
                                            &mut tool_gate,
                                            "error",
                                        )
                                        .await;
                                        let record_completion =
                                            stream_tail_record_completion(tail, &completion_buf);
                                        record(
                                            "error".to_string(),
                                            &record_completion,
                                            completion_token_count,
                                        );
                                        return;
                                    }
                                };
                                let scan = stop_gate.push(&delta);
                                let chunk = reasoning_splitter.push(&scan.emit);
                                // The emit awaits channel capacity — a client
                                // that stops reading (zero TCP window) would
                                // otherwise park this task INSIDE the select
                                // arm, where the request-deadline arm can never
                                // fire. Cap the wait at the same deadline.
                                let emitted = tokio::time::timeout_at(
                                    deadline,
                                    emit_or_buffer_reasoning_chunk(
                                        &tx,
                                        &id,
                                        created,
                                        &model,
                                        chunk,
                                        &mut completion_buf,
                                        &mut reasoning_buf,
                                        &mut content_buf,
                                        &mut tool_gate,
                                    ),
                                );
                                match emitted.await {
                                    Err(_) => {
                                        // Deadline elapsed while the client
                                        // refused bytes — same terminal path
                                        // as the deadline arm.
                                        timed_out = true;
                                        break;
                                    }
                                    Ok(false) => {
                                        cancel.cancel();
                                        let _ = batching_engine.cancel(request_id).await;
                                        record(
                                            "client_disconnect".to_string(),
                                            &completion_buf,
                                            completion_token_count,
                                        );
                                        return;
                                    }
                                    Ok(true) => {
                                        let body_enqueued_at = std::time::Instant::now();
                                        latency_tracker
                                            .lock()
                                            .unwrap()
                                            .record_client_delivery(
                                                handler_received_at,
                                                body_enqueued_at,
                                            );
                                        if let Some(timing_payload) = streaming_token_timing_json(
                                            include_token_timing,
                                            completion_token_count,
                                            token,
                                            request_start,
                                            timing,
                                            handler_received_at,
                                            body_enqueued_at,
                                            gap,
                                        ) {
                                            match tokio::time::timeout_at(
                                                deadline,
                                                tx.send(Event::default().data(timing_payload)),
                                            )
                                            .await
                                            {
                                                Err(_) => {
                                                    timed_out = true;
                                                    break;
                                                }
                                                Ok(Err(_)) => {
                                                    cancel.cancel();
                                                    let _ = batching_engine.cancel(request_id).await;
                                                    record(
                                                        "client_disconnect".to_string(),
                                                        &completion_buf,
                                                        completion_token_count,
                                                    );
                                                    return;
                                                }
                                                Ok(Ok(())) => {}
                                            }
                                        }
                                    }
                                }
                            }
                            Some(EngineEvent::Done { output }) => {
                                let finish = match output.finish_reason {
                                    kiln_model::FinishReason::Eos => "stop",
                                    kiln_model::FinishReason::MaxTokens => "length",
                                    kiln_model::FinishReason::StopSequence(_) => "stop",
                                };
                                let (terminal_tx, terminal_rx) =
                                    tokio::sync::mpsc::channel(STREAM_TERMINAL_EVENT_CAPACITY);
                                // Flush the emit gates: detokenizer residue
                                // passes the stop gate (a stop can complete
                                // inside held bytes), then the splitter —
                                // BEFORE the splitter's own flush.
                                {
                                    let residual = match detok.flush(&tokenizer, &generated_tokens) {
                                        Ok(residual) => residual,
                                        Err(error) => {
                                            let error = format!(
                                                "streaming detokenizer flush failed: {error}"
                                            );
                                            *record_error.lock().unwrap() = Some(error.clone());
                                            stream_terminal_for_task.fail(error.clone());
                                            let tail = flush_buffered_stream_tail(
                                                &tx,
                                                &id,
                                                created,
                                                &model,
                                                &mut reasoning_splitter,
                                                &mut completion_buf,
                                                &mut reasoning_buf,
                                                &mut content_buf,
                                                &mut tool_gate,
                                                "error",
                                            )
                                            .await;
                                            let record_completion =
                                                stream_tail_record_completion(
                                                    tail,
                                                    &completion_buf,
                                                );
                                            record(
                                                "error".to_string(),
                                                &record_completion,
                                                completion_token_count,
                                            );
                                            return;
                                        }
                                    };
                                    let scan = stop_gate.push(&residual);
                                    let mut gate_tail = scan.emit;
                                    if scan.matched_stop.is_none() {
                                        gate_tail.push_str(&stop_gate.flush());
                                    }
                                    if !gate_tail.is_empty() {
                                        let chunk = reasoning_splitter.push(&gate_tail);
                                        if !emit_or_buffer_reasoning_chunk(
                                            &terminal_tx,
                                            &id,
                                            created,
                                            &model,
                                            chunk,
                                            &mut completion_buf,
                                            &mut reasoning_buf,
                                            &mut content_buf,
                                            &mut tool_gate,
                                        )
                                        .await
                                        {
                                            record(
                                                "client_disconnect".to_string(),
                                                &completion_buf,
                                                completion_token_count,
                                            );
                                            return;
                                        }
                                    }
                                }
                                let trailing = reasoning_splitter.flush();
                                if !emit_or_buffer_reasoning_chunk(
                                    &terminal_tx,
                                    &id,
                                    created,
                                    &model,
                                    trailing,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    &mut tool_gate,
                                )
                                .await
                                {
                                    record(
                                        "client_disconnect".to_string(),
                                        &completion_buf,
                                        completion_token_count,
                                    );
                                    return;
                                }
                                let reasoning_content = if reasoning_buf.is_empty() {
                                    None
                                } else {
                                    Some(reasoning_buf.clone())
                                };
                                let matched_stop = stop_gate
                                    .matched()
                                    .map(str::to_string)
                                    .or(match &output.finish_reason {
                                        kiln_model::FinishReason::StopSequence(s) => {
                                            Some(s.clone())
                                        }
                                        _ => None,
                                    });
                                let assistant_output =
                                    stream_assistant_output_with_stop_reconstruction(
                                        buffer_tool_content,
                                        reasoning_content,
                                        &content_buf,
                                        matched_stop.as_deref(),
                                        finish,
                                    );
                                // Pre-tag content already streamed eagerly —
                                // the gate's trim_end holdback makes the wire
                                // equal the parsed content; emitting it again
                                // would duplicate every tool-call preamble.
                                if let Some(tool_calls) = assistant_output.tool_calls.as_deref() {
                                    if !emit_tool_calls_chunk(
                                        &terminal_tx,
                                        &id,
                                        created,
                                        &model,
                                        tool_calls,
                                    )
                                    .await
                                    {
                                        record(
                                            "client_disconnect".to_string(),
                                            &completion_buf,
                                            completion_token_count,
                                        );
                                        return;
                                    }
                                } else if buffer_tool_content
                                    && !tool_gate.unsent(&content_buf).is_empty()
                                    && !emit_content_chunk(
                                        &terminal_tx,
                                        &id,
                                        created,
                                        &model,
                                        tool_gate.unsent(&content_buf).to_string(),
                                    )
                                    .await
                                {
                                    record(
                                        "client_disconnect".to_string(),
                                        &completion_buf,
                                        completion_token_count,
                                    );
                                    return;
                                }
                                let finish = assistant_output.finish_reason.clone();
                                let record_completion = assistant_output.preview_source().to_string();
                                let total_latency = request_start.elapsed();
                                let ttft = first_token_ready_at
                                    .map(|ready_at| ready_at.saturating_duration_since(request_start));
                                let latency_diagnostics =
                                    latency_tracker.lock().unwrap().diagnostics();
                                let performance = include_performance.then(|| {
                                    ChatCompletionPerformanceMetadata {
                                        prompt_tokens: prompt_token_count,
                                        completion_tokens: output.completion_tokens,
                                        ttft_ms: ttft.map(duration_ms_f64),
                                        prefill_ms: Some(duration_ms_f64(output.prefill_duration)),
                                        actor_queue_ms: Some(duration_ms_f64(
                                            output.actor_queue_duration,
                                        )),
                                        actor_admission_ms: Some(duration_ms_f64(
                                            output.actor_admission_duration,
                                        )),
                                        actor_prefill_wall_ms: output
                                            .actor_prefill_wall_duration
                                            .map(duration_ms_f64),
                                        resident_prefill_used: Some(
                                            output.resident_prefill_used,
                                        ),
                                        decode_ms: Some(duration_ms_f64(output.decode_duration)),
                                        total_latency_ms: duration_ms_f64(total_latency),
                                        decode_tokens_per_sec:
                                            decode_tokens_per_sec_for_performance_metadata(
                                                output.completion_tokens,
                                                total_latency,
                                                ttft,
                                                Some(output.decode_duration),
                                            ),
                                        adapter_used: performance_adapter_used.clone(),
                                        thinking_mode: thinking_mode.clone(),
                                        finish_reason: finish.clone(),
                                        latency: Some(latency_diagnostics.clone()),
                                    }
                                });
                                let mut thinking_budget_metadata =
                                    stream_thinking_budget_metadata.clone();
                                if let Some(status) = finalized_thinking_budget_status(
                                    thinking_budget.as_ref(),
                                    completion_token_count as usize,
                                ) {
                                    apply_thinking_budget_status_to_metadata(
                                        &mut thinking_budget_metadata,
                                        status,
                                    );
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
                                            content: None,
                                            reasoning_content: None,
                                            tool_calls: None,
                                        },
                                        finish_reason: Some(finish.clone()),
                                    }],
                                };
                                drop(terminal_tx);
                                let mut terminal_events =
                                    drain_terminal_event_buffer(terminal_rx);
                                terminal_events.push_back(
                                    Event::default().data(streaming_finish_chunk_json(
                                        &chunk,
                                        &thinking_budget_metadata,
                                        performance.as_ref(),
                                    )),
                                );
                                if include_usage {
                                    terminal_events.push_back(
                                        Event::default().data(usage_chunk_json(
                                            &id,
                                            created,
                                            &model,
                                            prompt_token_count as u32,
                                            output.completion_tokens as u32,
                                        )),
                                    );
                                }
                                terminal_events.push_back(Event::default().data("[DONE]"));
                                record(
                                    finish,
                                    &record_completion,
                                    output.completion_tokens as u32,
                                );
                                stream_terminal_for_task.complete(terminal_events);
                                return;
                            }
                            Some(EngineEvent::Error(err)) => {
                                tracing::error!(error = %err, "batched streaming generation failed");
                                let error = err.to_string();
                                *record_error.lock().unwrap() = Some(error.clone());
                                stream_terminal_for_task.fail(error.clone());
                                let tail = flush_buffered_stream_tail(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    &mut reasoning_splitter,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    &mut tool_gate,
                                    "error",
                                )
                                .await;
                                let record_completion =
                                    stream_tail_record_completion(tail, &completion_buf);
                                record(
                                    "error".to_string(),
                                    &record_completion,
                                    completion_token_count,
                                );
                                return;
                            }
                            None => {
                                let error =
                                    "batched generation worker ended without a terminal event"
                                        .to_string();
                                *record_error.lock().unwrap() = Some(error.clone());
                                stream_terminal_for_task.fail(error.clone());
                                let tail = flush_buffered_stream_tail(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    &mut reasoning_splitter,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    &mut tool_gate,
                                    "error",
                                )
                                .await;
                                let record_completion =
                                    stream_tail_record_completion(tail, &completion_buf);
                                record(
                                    "error".to_string(),
                                    &record_completion,
                                    completion_token_count,
                                );
                                return;
                            }
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out = true;
                        break;
                    }
                }
            }

            if timed_out {
                let error = format!(
                    "streaming generation timed out after {} ms",
                    timeout.as_millis()
                );
                *record_error.lock().unwrap() = Some(error.clone());
                stream_terminal_for_task.fail(error.clone());
                cancel.cancel();
                let _ = batching_engine.cancel(request_id).await;
                let tail = flush_buffered_stream_tail(
                    &tx,
                    &id,
                    created,
                    &model,
                    &mut reasoning_splitter,
                    &mut completion_buf,
                    &mut reasoning_buf,
                    &mut content_buf,
                    &mut tool_gate,
                    "timeout",
                )
                .await;
                let record_completion = stream_tail_record_completion(tail, &completion_buf);
                record(
                    "timeout".to_string(),
                    &record_completion,
                    completion_token_count,
                );
            }
        }
    });

    let stream = stream_with_terminal(rx, stream_terminal);

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn generate_mock(
    state: &AppState,
    scheduler: &tokio::sync::Mutex<kiln_scheduler::Scheduler>,
    engine: &std::sync::Arc<dyn kiln_model::engine::Engine>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_token_count = prompt_tokens.len();
    let request = Request::new(
        prompt_tokens.to_vec(),
        sampling.clone(),
        req.adapter.request_adapter_name(),
    );
    let request_id = request.id;

    // Add to scheduler
    {
        let mut sched = scheduler.lock().await;
        sched.add_request(request);
    }

    // Run scheduler steps until this request completes.
    let max_steps = 100;
    let mut output_tokens = Vec::new();
    let mut first_token_at: Option<std::time::Instant> = None;

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
            let accepted_token = if *rid == request_id {
                token.map(|token| sampling.apply_thinking_budget(&output_tokens, token))
            } else {
                *token
            };
            if *rid == request_id {
                if let Some(t) = accepted_token {
                    first_token_at.get_or_insert_with(std::time::Instant::now);
                    output_tokens.push(t);
                }
            }

            let prefill_processed = step_output
                .scheduled
                .iter()
                .find(|s| s.request_id == *rid && s.is_prefill)
                .map(|s| s.num_tokens);

            let finished = *finished
                || output_tokens.len() >= sampling.max_tokens.min(MOCK_COMPLETION_TOKEN_LIMIT);
            sched.update_request(rid, accepted_token, finished, prefill_processed);
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
    let thinking_budget_status =
        finalized_thinking_budget_status(sampling.thinking_budget.as_ref(), completion_tokens);
    let assistant_output = assistant_output_from_split_parts(req, None, completion_text, "stop");
    let metadata =
        chat_completion_metadata_from_prompt_and_output(state, req, prompt_text, &assistant_output);
    let assistant_output = apply_reasoning_content_policy(
        assistant_output,
        fold_reasoning_into_content_for_request(state, req),
    );

    record_recent_request(
        state,
        RequestRecord {
            user_agent: req.user_agent.clone(),
            client: req.client.clone(),
            completion_preview: truncate_chars(
                assistant_output.preview_source(),
                COMPLETION_PREVIEW_MAX_CHARS,
            ),
            completion_full: Some(truncate_chars(
                assistant_output.preview_source(),
                FULL_BODY_MAX_CHARS,
            )),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: assistant_output.finish_reason.clone(),
            thinking_mode: Some("mock".to_string()),
            prefix_cache: Some("not_applicable".to_string()),
            thinking_budget: Some(recent_thinking_budget_with_status(
                &metadata.thinking_budget,
                thinking_budget_status,
            )),
            ..request_record_from_req(state, req, &id, &model, false)
        },
    );

    let finish_reason = assistant_output.finish_reason.clone();
    let mut response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: assistant_output.content,
                // Mock backend never emits a reasoning block.
                reasoning_content: assistant_output.reasoning_content,
                tool_calls: assistant_output.tool_calls,
                name: None,
                tool_call_id: None,
            },
            finish_reason,
            thinking_budget: None,
            rollout_provenance: None,
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    attach_thinking_budget_outcome(thinking_budget_status, &mut response);
    let ttft = first_token_at.map(|instant| instant.duration_since(request_start));
    attach_chat_performance_metadata(state, req, &mut response, request_start, ttft, None, None);
    Ok(response)
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

/// Maximum number of choices a single chat completion request may produce.
const CHAT_MAX_CHOICES: usize = BATCH_MAX_TOTAL_OUTPUTS;

/// Maximum number of source adapters allowed in a single compose request
/// (`adapters: [...]` on `/v1/chat/completions` and `/v1/completions/batch`).
/// Caps the cheapest DoS shape from §6 of `docs/audits/security-audit-v0.1.md`:
/// each entry triggers a safetensors read and an N-way `merge_concat`, so an
/// unbounded list lets a single request pin CPU + I/O for arbitrarily long.
const MAX_COMPOSE_ADAPTERS: usize = 16;

/// Batch completion request — generate completions for many prompts (and/or
/// many completions per prompt) in a single HTTP round-trip.
///
/// Designed for the GRPO loop: groups of `n` completions per prompt are
/// the unit of advantage normalization, and issuing N separate HTTP requests
/// per group adds non-trivial overhead. With this endpoint a GRPO worker
/// posts the whole group in one call; completions for the same prompt are run
/// in prompt-local order so prefix-cache registration from the first output can
/// remove duplicate prefill work from the rest of the group.
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
    /// See [`ChatCompletionRequest::min_p`].
    #[serde(default)]
    pub min_p: Option<f32>,
    /// See [`ChatCompletionRequest::presence_penalty`].
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// See [`ChatCompletionRequest::frequency_penalty`].
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// See [`ChatCompletionRequest::repetition_penalty`].
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// See [`ChatCompletionRequest::sampling_preset`].
    #[serde(default)]
    pub sampling_preset: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// OpenAI newer-name alias for `max_tokens`. `max_tokens` wins when both
    /// are present to preserve existing request behavior.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    /// See [`ChatCompletionRequest::ignore_eos`].
    #[serde(default)]
    pub ignore_eos: bool,
    /// Shared thinking token budget for every completion in the batch. Uses
    /// the same omitted/null/number semantics as the chat endpoint.
    #[serde(default)]
    pub thinking_budget_tokens: BudgetOverride<usize>,
    /// Shared thinking wall-clock budget (milliseconds) for every completion.
    #[serde(default)]
    pub thinking_budget_ms: BudgetOverride<u64>,
    #[serde(default, deserialize_with = "deserialize_optional_stop")]
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
    /// OpenAI-style tool/function definitions shared by every prompt in the
    /// batch. Forwarded to the same chat template context as the chat endpoint.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI-style `tool_choice` shared by every prompt in the batch.
    /// Forwarded to the same chat template context as the chat endpoint.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Kiln extension: additional top-level variables forwarded into the
    /// HuggingFace Jinja chat-template context for every prompt in the batch.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
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
    pub metadata: BatchCompletionMetadata,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BatchCompletionMetadata {
    pub thinking_budget: ThinkingBudgetConfigurationMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatchCompletionItem {
    pub prompt_index: usize,
    pub completion_index: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    pub finish_reason: String,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_optional_thinking_budget_status"
    )]
    pub thinking_budget: Option<ThinkingBudgetStatus>,
    pub usage: Usage,
}

struct BatchPromptGroup {
    messages: Vec<Message>,
    prompt_indices: Vec<usize>,
}

#[derive(Serialize)]
struct BatchPromptMessageCacheKey<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Cow<'a, [serde_json::Value]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

#[derive(Serialize)]
struct DeterministicBatchCacheKeyWire<'a> {
    prompts: Vec<Vec<BatchPromptMessageCacheKey<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [serde_json::Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "is_false")]
    fold_reasoning_into_content: bool,
    n: usize,
    temperature_bits: u32,
    max_tokens: usize,
    #[serde(skip_serializing_if = "is_false")]
    ignore_eos: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget_tokens: Option<usize>,
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
    min_p_bits: u32,
    presence_penalty_bits: u32,
    frequency_penalty_bits: u32,
    repetition_penalty_bits: u32,
    seed: Option<u64>,
}

fn batch_prompt_cache_key(messages: &[Message]) -> Vec<BatchPromptMessageCacheKey<'_>> {
    messages
        .iter()
        .map(|m| BatchPromptMessageCacheKey {
            role: &m.role,
            content: &m.content,
            tool_calls: normalized_message_tool_calls_for_cache(m.tool_calls.as_deref()),
            name: m.name.as_deref(),
            tool_call_id: m.tool_call_id.as_deref(),
        })
        .collect()
}

fn batch_synth_messages(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .map(|m| Message {
            role: m.role.clone(),
            content: m.content.clone(),
            // Input reasoning is not rendered by the chat endpoint either, but
            // tool metadata is part of the visible template conversation.
            reasoning_content: None,
            tool_calls: m
                .tool_calls
                .as_ref()
                .filter(|tool_calls| !tool_calls.is_empty())
                .cloned(),
            name: m.name.clone(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

fn batch_prompt_groups(prompts: &[Vec<Message>]) -> Vec<BatchPromptGroup> {
    let mut group_by_key = std::collections::HashMap::new();
    let mut groups: Vec<BatchPromptGroup> = Vec::new();

    for (prompt_index, messages) in prompts.iter().enumerate() {
        let key = serde_json::to_string(&batch_prompt_cache_key(messages))
            .expect("serializing batch prompt group key should not fail");
        let group_index = match group_by_key.get(&key).copied() {
            Some(group_index) => group_index,
            None => {
                let group_index = groups.len();
                groups.push(BatchPromptGroup {
                    messages: batch_synth_messages(messages),
                    prompt_indices: Vec::new(),
                });
                group_by_key.insert(key, group_index);
                group_index
            }
        };

        groups[group_index].prompt_indices.push(prompt_index);
    }

    groups
}

#[cfg(test)]
fn deterministic_batch_cache_key(
    req: &BatchCompletionRequest,
    total_outputs: usize,
) -> Option<String> {
    deterministic_batch_cache_key_with_vocab_size(req, total_outputs, usize::MAX)
}

#[cfg(test)]
fn deterministic_batch_cache_key_with_vocab_size(
    req: &BatchCompletionRequest,
    total_outputs: usize,
    vocab_size: usize,
) -> Option<String> {
    deterministic_batch_cache_key_with_vocab_size_and_fold(
        req,
        total_outputs,
        vocab_size,
        false,
        batch_token_budget_without_server_default(req),
    )
}

fn deterministic_batch_cache_key_with_vocab_size_and_fold(
    req: &BatchCompletionRequest,
    total_outputs: usize,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
    thinking_budget_tokens: Option<usize>,
) -> Option<String> {
    if total_outputs == 0 || req.adapter.is_some() || req.adapters.is_some() {
        return None;
    }

    let sampling = batch_request_sampling_for_cache_key(req, req.seed);
    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return None;
    }
    let NormalizedRequestSamplingKey {
        temperature_bits,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    } = normalized_deterministic_request_sampling_key(&sampling, vocab_size);
    let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
    let normalized_tool_choice =
        normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
    let normalized_chat_template_kwargs =
        normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

    let key = DeterministicBatchCacheKeyWire {
        prompts: req
            .prompts
            .iter()
            .map(|messages| batch_prompt_cache_key(messages))
            .collect(),
        tools: normalized_tools,
        tool_choice: normalized_tool_choice,
        chat_template_kwargs: normalized_chat_template_kwargs,
        fold_reasoning_into_content,
        n: req.n.unwrap_or(1),
        temperature_bits,
        max_tokens: sampling.max_tokens,
        ignore_eos: normalized_ignore_eos_for_cache(&sampling),
        thinking_budget_tokens,
        stop,
        top_p_bits,
        top_k,
        min_p_bits,
        presence_penalty_bits,
        frequency_penalty_bits,
        repetition_penalty_bits,
        seed,
    };
    Some(serde_json::to_string(&key).expect("serializing batch cache key should not fail"))
}

fn batch_response_from_cached_value(
    state: &AppState,
    req: &BatchCompletionRequest,
    cached: DeterministicBatchCacheValue,
) -> BatchCompletionResponse {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completions = cached
        .completions
        .into_iter()
        .map(|item| {
            let thinking_budget = item.thinking_budget_status;
            let reasoning_content = item.reasoning_content;
            BatchCompletionItem {
                prompt_index: item.prompt_index,
                completion_index: item.completion_index,
                text: content_with_reasoning_policy(
                    item.text,
                    reasoning_content.as_deref(),
                    state.fold_reasoning_into_content,
                ),
                reasoning_content,
                tool_calls: item.tool_calls,
                finish_reason: item.finish_reason,
                thinking_budget,
                usage: Usage {
                    prompt_tokens: item.prompt_tokens,
                    completion_tokens: item.completion_tokens,
                    total_tokens: item.prompt_tokens.saturating_add(item.completion_tokens),
                },
            }
        })
        .collect();

    BatchCompletionResponse {
        id: format!("batchcmpl-{}", Uuid::new_v4()),
        object: "batch.completion",
        created: now_epoch(),
        model,
        completions,
        usage: Usage {
            prompt_tokens: cached.prompt_tokens,
            completion_tokens: cached.completion_tokens,
            total_tokens: cached
                .prompt_tokens
                .saturating_add(cached.completion_tokens),
        },
        metadata: batch_completion_metadata_for_request(state, req),
    }
}

fn batch_response_from_cached_chat_choices(
    state: &AppState,
    req: &BatchCompletionRequest,
    cached: DeterministicChatChoicesCacheValue,
) -> BatchCompletionResponse {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let prompt_tokens_per_choice = cached.prompt_tokens;
    let mut total_completion_tokens = 0usize;
    let completions = cached
        .completions
        .into_iter()
        .enumerate()
        .map(|(completion_index, completion)| {
            total_completion_tokens =
                total_completion_tokens.saturating_add(completion.completion_tokens);
            let thinking_budget = completion.thinking_budget_status;
            let reasoning_content = completion.reasoning_content;
            BatchCompletionItem {
                prompt_index: 0,
                completion_index,
                text: content_with_reasoning_policy(
                    completion.text,
                    reasoning_content.as_deref(),
                    state.fold_reasoning_into_content,
                ),
                reasoning_content,
                tool_calls: completion.tool_calls,
                finish_reason: completion.finish_reason,
                thinking_budget,
                usage: Usage {
                    prompt_tokens: prompt_tokens_per_choice,
                    completion_tokens: completion.completion_tokens,
                    total_tokens: prompt_tokens_per_choice
                        .saturating_add(completion.completion_tokens),
                },
            }
        })
        .collect::<Vec<_>>();
    let total_prompt_tokens = prompt_tokens_per_choice.saturating_mul(completions.len());

    BatchCompletionResponse {
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
        metadata: batch_completion_metadata_for_request(state, req),
    }
}

fn batch_response_from_cached_chat_choice_groups(
    state: &AppState,
    req: &BatchCompletionRequest,
    cached_by_prompt: Vec<DeterministicChatChoicesCacheValue>,
) -> BatchCompletionResponse {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;
    let completion_count = cached_by_prompt
        .iter()
        .map(|cached| cached.completions.len())
        .sum();
    let mut completions = Vec::with_capacity(completion_count);

    for (prompt_index, cached) in cached_by_prompt.into_iter().enumerate() {
        total_prompt_tokens = total_prompt_tokens.saturating_add(
            cached
                .prompt_tokens
                .saturating_mul(cached.completions.len()),
        );
        for (completion_index, completion) in cached.completions.into_iter().enumerate() {
            total_completion_tokens =
                total_completion_tokens.saturating_add(completion.completion_tokens);
            let thinking_budget = completion.thinking_budget_status;
            let reasoning_content = completion.reasoning_content;
            completions.push(BatchCompletionItem {
                prompt_index,
                completion_index,
                text: content_with_reasoning_policy(
                    completion.text,
                    reasoning_content.as_deref(),
                    state.fold_reasoning_into_content,
                ),
                reasoning_content,
                tool_calls: completion.tool_calls,
                finish_reason: completion.finish_reason,
                thinking_budget,
                usage: Usage {
                    prompt_tokens: cached.prompt_tokens,
                    completion_tokens: completion.completion_tokens,
                    total_tokens: cached
                        .prompt_tokens
                        .saturating_add(completion.completion_tokens),
                },
            });
        }
    }

    BatchCompletionResponse {
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
        metadata: batch_completion_metadata_for_request(state, req),
    }
}

fn batch_response_from_cached_chat_requests(
    state: &AppState,
    req: &BatchCompletionRequest,
    cached: Vec<DeterministicChatRequestCacheValue>,
) -> BatchCompletionResponse {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;
    let completions = cached
        .into_iter()
        .enumerate()
        .map(|(prompt_index, cached)| {
            let completion = cached.completion;
            total_prompt_tokens = total_prompt_tokens.saturating_add(cached.prompt_tokens);
            total_completion_tokens =
                total_completion_tokens.saturating_add(completion.completion_tokens);
            let thinking_budget = completion.thinking_budget_status;
            let reasoning_content = completion.reasoning_content;
            BatchCompletionItem {
                prompt_index,
                completion_index: 0,
                text: content_with_reasoning_policy(
                    completion.text,
                    reasoning_content.as_deref(),
                    state.fold_reasoning_into_content,
                ),
                reasoning_content,
                tool_calls: completion.tool_calls,
                finish_reason: completion.finish_reason,
                thinking_budget,
                usage: Usage {
                    prompt_tokens: cached.prompt_tokens,
                    completion_tokens: completion.completion_tokens,
                    total_tokens: cached
                        .prompt_tokens
                        .saturating_add(completion.completion_tokens),
                },
            }
        })
        .collect();

    BatchCompletionResponse {
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
        metadata: batch_completion_metadata_for_request(state, req),
    }
}

fn batch_response_from_chat_request_cache_hits(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &BatchCompletionRequest,
    vocab_size: usize,
) -> Result<Option<BatchCompletionResponse>, ApiError> {
    let budget = effective_batch_thinking_budget_for_request(state, req);
    if budget.configured()
        || req.n.unwrap_or(1) != 1
        || req.adapter.is_some()
        || req.adapters.is_some()
    {
        return Ok(None);
    }

    let mut keys = Vec::with_capacity(req.prompts.len());
    for (prompt_index, messages) in req.prompts.iter().enumerate() {
        let seed = req.seed.map(|seed| seed.wrapping_add(prompt_index as u64));
        let Some(key) =
            deterministic_chat_request_cache_key_from_batch_prompt_with_vocab_size_and_fold(
                req,
                messages,
                seed,
                vocab_size,
                state.fold_reasoning_into_content,
            )?
        else {
            return Ok(None);
        };
        keys.push(state.deterministic_cache_key(adapter.clone(), key));
    }

    let mut cached = Vec::with_capacity(keys.len());
    {
        let mut cache = state.chat_request_cache.lock().unwrap();
        for key in keys {
            match cache.probe(&key) {
                DeterministicChatRequestCacheProbe::Hit(value) => cached.push(value),
                DeterministicChatRequestCacheProbe::Wait(_)
                | DeterministicChatRequestCacheProbe::Miss => return Ok(None),
            }
        }
    }

    Ok(Some(batch_response_from_cached_chat_requests(
        state, req, cached,
    )))
}

fn batch_response_from_chat_choices_cache_hits(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &BatchCompletionRequest,
    vocab_size: usize,
) -> Result<Option<BatchCompletionResponse>, ApiError> {
    let n_per = req.n.unwrap_or(1);
    let budget = effective_batch_thinking_budget_for_request(state, req);
    if budget.configured() || n_per <= 1 || req.adapter.is_some() || req.adapters.is_some() {
        return Ok(None);
    }

    let mut keys = Vec::with_capacity(req.prompts.len());
    for (prompt_index, messages) in req.prompts.iter().enumerate() {
        let seed = req
            .seed
            .map(|seed| seed.wrapping_add((prompt_index * n_per) as u64));
        let Some(key) =
            deterministic_chat_choices_cache_key_from_batch_prompt_with_vocab_size_and_fold(
                req,
                messages,
                seed,
                vocab_size,
                state.fold_reasoning_into_content,
            )?
        else {
            return Ok(None);
        };
        keys.push(state.deterministic_cache_key(adapter.clone(), key));
    }

    let mut cached_by_prompt = Vec::with_capacity(keys.len());
    {
        let mut cache = state.chat_choices_cache.lock().unwrap();
        for key in keys {
            match cache.probe(&key) {
                DeterministicChatChoicesCacheProbe::Hit(value)
                    if value.completions.len() == n_per =>
                {
                    cached_by_prompt.push(value)
                }
                DeterministicChatChoicesCacheProbe::Hit(_)
                | DeterministicChatChoicesCacheProbe::Wait(_)
                | DeterministicChatChoicesCacheProbe::Miss => return Ok(None),
            }
        }
    }

    Ok(Some(batch_response_from_cached_chat_choice_groups(
        state,
        req,
        cached_by_prompt,
    )))
}

fn cache_value_from_batch_response(resp: &BatchCompletionResponse) -> DeterministicBatchCacheValue {
    DeterministicBatchCacheValue {
        completions: resp
            .completions
            .iter()
            .map(|item| DeterministicBatchCacheItem {
                prompt_index: item.prompt_index,
                completion_index: item.completion_index,
                text: response_content_for_cache(&item.text, item.reasoning_content.as_deref()),
                reasoning_content: item.reasoning_content.clone(),
                tool_calls: item.tool_calls.clone(),
                finish_reason: item.finish_reason.clone(),
                prompt_tokens: item.usage.prompt_tokens,
                completion_tokens: item.usage.completion_tokens,
                thinking_budget_status: item.thinking_budget,
            })
            .collect(),
        prompt_tokens: resp.usage.prompt_tokens,
        completion_tokens: resp.usage.completion_tokens,
    }
}

fn chat_request_cache_value_from_batch_item(
    item: &BatchCompletionItem,
) -> Option<DeterministicChatRequestCacheValue> {
    Some(DeterministicChatRequestCacheValue {
        prompt_tokens: item.usage.prompt_tokens,
        completion: DeterministicCompletionCacheValue {
            text: response_content_for_cache(&item.text, item.reasoning_content.as_deref()),
            reasoning_content: item.reasoning_content.clone(),
            tool_calls: item.tool_calls.clone(),
            finish_reason: item.finish_reason.clone(),
            completion_tokens: item.usage.completion_tokens,
            thinking_budget_status: item.thinking_budget,
        },
    })
}

fn store_chat_request_cache_from_batch_response(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &BatchCompletionRequest,
    resp: &BatchCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    let n_per = req.n.unwrap_or(1);
    let budget = effective_batch_thinking_budget_for_request(state, req);
    if budget.configured() || n_per == 0 || req.adapter.is_some() || req.adapters.is_some() {
        return Ok(());
    }

    let mut items_by_prompt = std::iter::repeat_with(|| Vec::with_capacity(n_per))
        .take(req.prompts.len())
        .collect::<Vec<_>>();
    for item in &resp.completions {
        let Some(slot) = items_by_prompt.get_mut(item.prompt_index) else {
            return Ok(());
        };
        if item.completion_index >= n_per || slot.len() >= n_per {
            return Ok(());
        }
        slot.push(item);
    }

    let mut entries = Vec::with_capacity(resp.completions.len().min(req.prompts.len()));
    let mut seen_keys = std::collections::HashSet::new();
    for (prompt_index, mut items) in items_by_prompt.into_iter().enumerate() {
        if items.len() != n_per {
            return Ok(());
        }
        items.sort_by_key(|item| item.completion_index);
        for (expected, item) in items.into_iter().enumerate() {
            if item.completion_index != expected {
                return Ok(());
            }
            let seed = req
                .seed
                .map(|seed| seed.wrapping_add((prompt_index * n_per + expected) as u64));
            let Some(key) =
                deterministic_chat_request_cache_key_from_batch_prompt_with_vocab_size_and_fold(
                    req,
                    &req.prompts[prompt_index],
                    seed,
                    vocab_size,
                    state.fold_reasoning_into_content,
                )?
            else {
                continue;
            };
            if seen_keys.insert(key.clone())
                && let Some(value) = chat_request_cache_value_from_batch_item(item)
            {
                entries.push((key, value));
            }
        }
    }

    let mut cache = state.chat_request_cache.lock().unwrap();
    for (key, value) in entries {
        cache.insert(state.deterministic_cache_key(adapter.clone(), key), value);
    }
    Ok(())
}

fn chat_choices_cache_value_from_batch_items(
    mut items: Vec<&BatchCompletionItem>,
    n_per: usize,
) -> Option<DeterministicChatChoicesCacheValue> {
    if items.len() != n_per {
        return None;
    }
    items.sort_by_key(|item| item.completion_index);
    for (expected, item) in items.iter().enumerate() {
        if item.completion_index != expected {
            return None;
        }
    }
    let prompt_tokens = items.first()?.usage.prompt_tokens;
    if items
        .iter()
        .any(|item| item.usage.prompt_tokens != prompt_tokens)
    {
        return None;
    }
    Some(DeterministicChatChoicesCacheValue {
        prompt_tokens,
        completions: items
            .into_iter()
            .map(|item| DeterministicCompletionCacheValue {
                text: response_content_for_cache(&item.text, item.reasoning_content.as_deref()),
                reasoning_content: item.reasoning_content.clone(),
                tool_calls: item.tool_calls.clone(),
                finish_reason: item.finish_reason.clone(),
                completion_tokens: item.usage.completion_tokens,
                thinking_budget_status: item.thinking_budget,
            })
            .collect(),
    })
}

fn store_chat_choices_cache_from_batch_response(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &BatchCompletionRequest,
    resp: &BatchCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    let n_per = req.n.unwrap_or(1);
    let budget = effective_batch_thinking_budget_for_request(state, req);
    if budget.configured() || n_per <= 1 || req.adapter.is_some() || req.adapters.is_some() {
        return Ok(());
    }

    let mut items_by_prompt = std::iter::repeat_with(|| Vec::with_capacity(n_per))
        .take(req.prompts.len())
        .collect::<Vec<_>>();
    for item in &resp.completions {
        let Some(slot) = items_by_prompt.get_mut(item.prompt_index) else {
            return Ok(());
        };
        slot.push(item);
    }

    let mut entries = Vec::with_capacity(items_by_prompt.len());
    for (prompt_index, items) in items_by_prompt.into_iter().enumerate() {
        let Some(value) = chat_choices_cache_value_from_batch_items(items, n_per) else {
            return Ok(());
        };
        let seed = req
            .seed
            .map(|seed| seed.wrapping_add((prompt_index * n_per) as u64));
        let Some(key) =
            deterministic_chat_choices_cache_key_from_batch_prompt_with_vocab_size_and_fold(
                req,
                &req.prompts[prompt_index],
                seed,
                vocab_size,
                state.fold_reasoning_into_content,
            )?
        else {
            continue;
        };
        entries.push((key, value));
    }

    let mut cache = state.chat_choices_cache.lock().unwrap();
    for (key, value) in entries {
        cache.insert(state.deterministic_cache_key(adapter.clone(), key), value);
    }
    Ok(())
}

fn store_chat_caches_from_batch_response(
    state: &AppState,
    adapter: &Option<LoadedAdapterIdentity>,
    req: &BatchCompletionRequest,
    resp: &BatchCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    let budget = effective_batch_thinking_budget_for_request(state, req);
    if budget.configured() {
        return Ok(());
    }
    store_chat_request_cache_from_batch_response(state, adapter, req, resp, vocab_size)?;
    store_chat_choices_cache_from_batch_response(state, adapter, req, resp, vocab_size)?;
    Ok(())
}

async fn batch_completions(
    State(state): State<AppState>,
    Json(req): Json<BatchCompletionRequest>,
) -> Result<Response, ApiError> {
    let start = std::time::Instant::now();
    if let Err(err) = ensure_backend_admission(&state) {
        state.metrics.inc_request(RequestStatus::Rejected);
        return Err(err);
    }
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

    if state.eval_mode {
        state.clear_eval_mode_transient_state();
    }

    result.map(|response| response_with_runtime_headers(&state, response))
}

async fn batch_completions_inner(
    state: &AppState,
    mut req: BatchCompletionRequest,
) -> Result<Response, ApiError> {
    let n_per = req.n.unwrap_or(1);

    if req.prompts.is_empty() {
        return Err(ApiError::batch_invalid_request(
            "'prompts' must contain at least one messages array",
        ));
    }
    if n_per == 0 {
        return Err(ApiError::batch_invalid_request("'n' must be >= 1 when set"));
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

    apply_eval_mode_batch_defaults(state, &mut req);
    let effective_thinking_budget = effective_batch_thinking_budget_for_request(state, &req);
    let stable_default_adapter = stable_default_adapter_identity(state);
    let cache_adapter = stable_default_adapter
        .clone()
        .unwrap_or_else(|| state.loaded_adapter_identity());

    let mut batch_cache_key = if effective_thinking_budget.max_time_ms.is_some() {
        None
    } else {
        deterministic_batch_cache_key_with_vocab_size_and_fold(
            &req,
            total_outputs,
            state.model_config.vocab_size,
            state.fold_reasoning_into_content,
            effective_thinking_budget.max_tokens,
        )
        .map(|request| state.deterministic_cache_key(cache_adapter.clone(), request))
    };
    let can_hit_batch_cache_before_adapter_work = stable_default_adapter.is_some();
    let mut batch_cache_owner = None;
    if can_hit_batch_cache_before_adapter_work && let Some(key) = batch_cache_key.as_ref() {
        let claim = state.batch_cache.lock().unwrap().claim(key);
        match claim {
            DeterministicBatchCacheClaim::Hit(cached) => {
                let resp = batch_response_from_cached_value(state, &req, cached);
                store_chat_caches_from_batch_response(
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
            DeterministicBatchCacheClaim::Wait(receiver) => {
                if let Some(cached) = wait_for_deterministic_batch(receiver).await {
                    let resp = batch_response_from_cached_value(state, &req, cached);
                    store_chat_caches_from_batch_response(
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
            DeterministicBatchCacheClaim::Owner(claim_id) => {
                batch_cache_owner = Some(BatchCacheOwnerGuard::new(
                    state.batch_cache.clone(),
                    key.clone(),
                    claim_id,
                ));
            }
        }
    }

    let chat_choices_cache_key = if effective_thinking_budget.configured() {
        None
    } else {
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &req,
            state.model_config.vocab_size,
            state.fold_reasoning_into_content,
        )?
        .map(|request| state.deterministic_cache_key(cache_adapter.clone(), request))
    };
    if can_hit_batch_cache_before_adapter_work && let Some(key) = chat_choices_cache_key.as_ref() {
        let probe = state.chat_choices_cache.lock().unwrap().probe(key);
        match probe {
            DeterministicChatChoicesCacheProbe::Hit(cached) => {
                let resp = batch_response_from_cached_chat_choices(state, &req, cached);
                let cache_value = cache_value_from_batch_response(&resp);
                if let Some(owner) = batch_cache_owner.take() {
                    owner.complete(cache_value);
                } else if let Some(key) = batch_cache_key.clone() {
                    state.batch_cache.lock().unwrap().insert(key, cache_value);
                }
                store_chat_request_cache_from_batch_response(
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
            DeterministicChatChoicesCacheProbe::Wait(receiver) => {
                if let Some(cached) = wait_for_deterministic_chat_choices(receiver).await {
                    let resp = batch_response_from_cached_chat_choices(state, &req, cached);
                    let cache_value = cache_value_from_batch_response(&resp);
                    if let Some(owner) = batch_cache_owner.take() {
                        owner.complete(cache_value);
                    } else if let Some(key) = batch_cache_key.clone() {
                        state.batch_cache.lock().unwrap().insert(key, cache_value);
                    }
                    store_chat_request_cache_from_batch_response(
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
            DeterministicChatChoicesCacheProbe::Miss => {}
        }
    }

    if can_hit_batch_cache_before_adapter_work
        && let Some(resp) = batch_response_from_chat_choices_cache_hits(
            state,
            &cache_adapter,
            &req,
            state.model_config.vocab_size,
        )?
    {
        let cache_value = cache_value_from_batch_response(&resp);
        if let Some(owner) = batch_cache_owner.take() {
            owner.complete(cache_value);
        } else if let Some(key) = batch_cache_key.clone() {
            state.batch_cache.lock().unwrap().insert(key, cache_value);
        }
        store_chat_request_cache_from_batch_response(
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

    if can_hit_batch_cache_before_adapter_work
        && let Some(resp) = batch_response_from_chat_request_cache_hits(
            state,
            &cache_adapter,
            &req,
            state.model_config.vocab_size,
        )?
    {
        let cache_value = cache_value_from_batch_response(&resp);
        if let Some(owner) = batch_cache_owner.take() {
            owner.complete(cache_value);
        } else if let Some(key) = batch_cache_key.clone() {
            state.batch_cache.lock().unwrap().insert(key, cache_value);
        }
        return Ok(response_with_loaded_adapter_identity(
            Json(resp).into_response(),
            &cache_adapter,
        ));
    }

    if batch_request_max_tokens(&req) == 0 {
        let mut completions_by_prompt: Vec<Option<Vec<BatchCompletionItem>>> =
            std::iter::repeat_with(|| None)
                .take(req.prompts.len())
                .collect();
        let mut total_prompt_tokens = 0usize;
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| state.served_model_id.clone());
        let created = now_epoch();

        for prompt_group in batch_prompt_groups(&req.prompts) {
            let prompt_text = render_prompt_text(
                state,
                &prompt_group.messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                req.chat_template_kwargs.as_ref(),
            )?;
            let prompt_tokens = encode_prompt_tokens(state, &prompt_text)?;
            let prompt_token_count = prompt_tokens.len();

            for prompt_index in prompt_group.prompt_indices {
                total_prompt_tokens =
                    total_prompt_tokens.saturating_add(prompt_token_count.saturating_mul(n_per));
                let mut items = Vec::with_capacity(n_per);
                for completion_index in 0..n_per {
                    items.push(BatchCompletionItem {
                        prompt_index,
                        completion_index,
                        text: String::new(),
                        reasoning_content: None,
                        tool_calls: None,
                        finish_reason: "length".to_string(),
                        thinking_budget: None,
                        usage: Usage {
                            prompt_tokens: prompt_token_count,
                            completion_tokens: 0,
                            total_tokens: prompt_token_count,
                        },
                    });
                }
                if let Some(slot) = completions_by_prompt.get_mut(prompt_index) {
                    *slot = Some(items);
                } else {
                    return Err(ApiError::internal(format!(
                        "batch zero-token path returned out-of-range prompt index {prompt_index}"
                    )));
                }
            }
        }

        let mut completions = Vec::with_capacity(total_outputs);
        for (prompt_index, items) in completions_by_prompt.into_iter().enumerate() {
            let items = items.ok_or_else(|| {
                ApiError::internal(format!(
                    "batch zero-token path did not return prompt index {prompt_index}"
                ))
            })?;
            completions.extend(items);
        }

        let resp = BatchCompletionResponse {
            id: format!("batchcmpl-{}", Uuid::new_v4()),
            object: "batch.completion",
            created,
            model,
            completions,
            usage: Usage {
                prompt_tokens: total_prompt_tokens,
                completion_tokens: 0,
                total_tokens: total_prompt_tokens,
            },
            metadata: batch_completion_metadata_for_request(state, &req),
        };
        let cache_value = cache_value_from_batch_response(&resp);
        if let Some(owner) = batch_cache_owner.take() {
            owner.complete(cache_value);
        } else if let Some(key) = batch_cache_key.clone() {
            state.batch_cache.lock().unwrap().insert(key, cache_value);
        }
        store_chat_request_cache_from_batch_response(
            state,
            &cache_adapter,
            &req,
            &resp,
            state.model_config.vocab_size,
        )?;
        store_chat_choices_cache_from_batch_response(
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

    // Resolve adapter once for the entire batch. After this returns,
    // The loaded-adapter identity reflects the loaded adapter and every
    // synthesized per-output ChatCompletionRequest below leaves
    // `adapter`/`adapters` as None — generate_real reads the active adapter
    // from state, not from the request.
    let has_composed_adapter = if let Some(list) = req.adapters.as_deref() {
        ensure_composed_adapter_for_request(state, list).await?;
        true
    } else {
        false
    };

    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if !has_composed_adapter {
            ensure_batch_adapter(state, runner, &req.adapter, &Uuid::new_v4().to_string()).await?;
        }
    }

    let request_adapter = state.loaded_adapter_identity();
    if let Some(key) = batch_cache_key.as_mut() {
        let rebound = state.deterministic_cache_key(request_adapter.clone(), key.request.clone());
        if batch_cache_owner
            .as_ref()
            .is_some_and(|owner| !owner.matches_key(&rebound))
        {
            drop(batch_cache_owner.take());
        }
        *key = rebound;
    }

    // Spawn one task per distinct rendered prompt, then run duplicates in that
    // group sequentially. Different prompts still run concurrently, while
    // duplicate prompt groups let the first physical generation register exact
    // prefix-cache state before later sampled completions look it up. Greedy
    // (`temp=0`) duplicates go further: the output is deterministic, so one
    // decode can serve each prompt-local `n` group and every identical prompt
    // row in the group.
    let clone_greedy_completions = effective_thinking_budget.max_time_ms.is_none()
        && batch_can_clone_deterministic_completions(&req);
    let clone_greedy_prompt_groups = effective_thinking_budget.max_time_ms.is_none()
        && batch_can_clone_identical_prompt_groups(&req);
    let prompt_count = req.prompts.len();
    let prompt_groups = batch_prompt_groups(&req.prompts);
    let prepare_prompt_groups = !clone_greedy_completions;
    let mut handles = Vec::with_capacity(prompt_groups.len());
    for prompt_group in prompt_groups {
        let state_clone = state.clone();
        let request_adapter = request_adapter.clone();
        let model = req.model.clone();
        let stop = normalized_stop_option_for_synthetic_request(req.stop.as_deref());
        let temperature = req.temperature;
        let top_p = req.top_p;
        let top_k = req.top_k;
        let min_p = req.min_p;
        let presence_penalty = req.presence_penalty;
        let frequency_penalty = req.frequency_penalty;
        let repetition_penalty = req.repetition_penalty;
        let sampling_preset = req.sampling_preset.clone();
        let max_tokens = req.max_tokens;
        let max_completion_tokens = req.max_completion_tokens;
        let thinking_budget_tokens = req.thinking_budget_tokens;
        let thinking_budget_ms = req.thinking_budget_ms;
        let ignore_eos = req.ignore_eos;
        let seed = req.seed;
        let tools = normalized_tools_option_for_synthetic_request(req.tools.as_deref());
        let tool_choice = normalized_tool_choice_option_for_synthetic_request(
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
        );
        let chat_template_kwargs = req.chat_template_kwargs.clone();

        handles.push(tokio::spawn(async move {
            let BatchPromptGroup {
                messages,
                prompt_indices,
            } = prompt_group;
            let prepared_prompt = if prepare_prompt_groups {
                let prompt_text = render_prompt_text(
                    &state_clone,
                    &messages,
                    tools.as_deref(),
                    tool_choice.as_ref(),
                    chat_template_kwargs.as_ref(),
                )?;
                let prompt_tokens = encode_prompt_tokens(&state_clone, &prompt_text)?;
                Some((prompt_text, prompt_tokens))
            } else {
                None
            };
            let mut group_responses = Vec::with_capacity(prompt_indices.len());
            let clone_prompt_group = clone_greedy_prompt_groups && prompt_indices.len() > 1;
            let mut cloned_group_response: Option<Vec<(usize, ChatCompletionResponse)>> = None;
            for prompt_index in prompt_indices {
                if let Some(cloned) = cloned_group_response.as_ref() {
                    group_responses.push((prompt_index, cloned.clone()));
                    continue;
                }

                let mut responses = Vec::with_capacity(n_per);
                let completion_count = if clone_greedy_completions { 1 } else { n_per };
                for completion_idx in 0..completion_count {
                    let derived_seed = seed
                        .map(|s| s.wrapping_add((prompt_index * n_per + completion_idx) as u64));
                    let synth_req = ChatCompletionRequest {
                        model: model.clone(),
                        messages: messages.clone(),
                        user_agent: None,
                        client: None,
                        n: None,
                        temperature,
                        top_p,
                        top_k,
                        min_p,
                        presence_penalty,
                        frequency_penalty,
                        repetition_penalty,
                        sampling_preset: sampling_preset.clone(),
                        max_tokens,
                        max_completion_tokens,
                        ignore_eos,
                        thinking_budget_tokens,
                        thinking_budget_ms,
                        stream: false,
                        stream_options: None,
                        stop: stop.clone(),
                        seed: derived_seed,
                        adapter: ChatAdapterSelection::Default,
                        adapters: None,
                        tools: tools.clone(),
                        tool_choice: tool_choice.clone(),
                        chat_template_kwargs: chat_template_kwargs.clone(),
                        fold_reasoning_into_content: None,
                        include_performance: None,
                        include_config_hashes: None,
                        rollout_provenance: false,
                    };
                    let resp = if let Some((prompt_text, prompt_tokens)) = prepared_prompt.as_ref()
                    {
                        generate_one_prepared_prompt_response(
                            &state_clone,
                            synth_req,
                            request_adapter.clone(),
                            prompt_text,
                            prompt_tokens,
                        )
                        .await?
                    } else {
                        generate_one_response(&state_clone, synth_req, request_adapter.clone())
                            .await?
                    };
                    responses.push((completion_idx, resp));
                }
                if clone_greedy_completions {
                    let first =
                        responses
                            .first()
                            .map(|(_, resp)| resp.clone())
                            .ok_or_else(|| {
                                ApiError::internal("greedy clone path produced no response")
                            })?;
                    responses.reserve(n_per.saturating_sub(1));
                    for completion_idx in 1..n_per {
                        responses.push((completion_idx, first.clone()));
                    }
                }
                if clone_prompt_group {
                    cloned_group_response = Some(responses.clone());
                }
                group_responses.push((prompt_index, responses));
            }
            Ok::<Vec<(usize, Vec<(usize, ChatCompletionResponse)>)>, ApiError>(group_responses)
        }));
    }

    let mut responses_by_prompt: Vec<Option<Vec<(usize, ChatCompletionResponse)>>> =
        std::iter::repeat_with(|| None).take(prompt_count).collect();

    for handle in handles {
        let group_responses = match handle.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err(ApiError::internal(format!("batch task join error: {e}")));
            }
        };

        for (prompt_index, responses) in group_responses {
            if let Some(slot) = responses_by_prompt.get_mut(prompt_index) {
                *slot = Some(responses);
            } else {
                return Err(ApiError::internal(format!(
                    "batch task returned out-of-range prompt index {prompt_index}"
                )));
            }
        }
    }

    let mut completions = Vec::with_capacity(total_outputs);
    let mut total_prompt_tokens = 0usize;
    let mut total_completion_tokens = 0usize;

    for (prompt_index, responses) in responses_by_prompt.into_iter().enumerate() {
        let responses = responses.ok_or_else(|| {
            ApiError::internal(format!(
                "batch task did not return prompt index {prompt_index}"
            ))
        })?;

        for (completion_index, resp) in responses {
            let choice = resp.choices.into_iter().next().ok_or_else(|| {
                ApiError::internal("generate returned a response with no choices")
            })?;
            total_prompt_tokens = total_prompt_tokens.saturating_add(resp.usage.prompt_tokens);
            total_completion_tokens =
                total_completion_tokens.saturating_add(resp.usage.completion_tokens);
            completions.push(BatchCompletionItem {
                prompt_index,
                completion_index,
                text: choice.message.content,
                reasoning_content: choice.message.reasoning_content,
                tool_calls: choice.message.tool_calls,
                finish_reason: choice.finish_reason,
                thinking_budget: choice.thinking_budget,
                usage: resp.usage,
            });
        }
    }

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());

    let resp = BatchCompletionResponse {
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
        metadata: batch_completion_metadata_for_request(state, &req),
    };
    let cache_value = cache_value_from_batch_response(&resp);
    if let Some(owner) = batch_cache_owner.take() {
        owner.complete(cache_value);
    } else if let Some(key) = batch_cache_key {
        state.batch_cache.lock().unwrap().insert(key, cache_value);
    }
    store_chat_choices_cache_from_batch_response(
        state,
        &request_adapter,
        &req,
        &resp,
        state.model_config.vocab_size,
    )?;
    Ok(response_with_loaded_adapter_identity(
        Json(resp).into_response(),
        &request_adapter,
    ))
}

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
