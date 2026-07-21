use super::*;

// Request admission, effective thinking policy, defaults, and response metadata
// are resolved together so every route reports the policy it actually ran.

/// Snapshot a default adapter that is already the exact revision published by
/// the runner. `None` means a transient per-request override is loaded, so a
/// default-request cache lookup must wait until adapter resolution completes.
pub(super) fn stable_default_adapter_identity(
    state: &AppState,
) -> Option<Option<LoadedAdapterIdentity>> {
    let default_name = state.active_adapter_name.read().unwrap().clone();
    let loaded = state.loaded_adapter_identity();
    match (default_name.as_deref(), loaded) {
        (None, None) => Some(None),
        (Some(name), Some(identity)) if identity.name == name => Some(Some(identity)),
        _ => None,
    }
}

pub(super) fn observe_post_prefill_vram(
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

pub(super) fn ensure_backend_admission(state: &AppState) -> Result<(), ApiError> {
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
pub(super) fn last_user_message_text(req: &ChatCompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| req.messages.last())
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

pub(super) fn thinking_mode_for_request(req: &ChatCompletionRequest) -> &'static str {
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

pub(super) fn request_thinking_enabled(req: &ChatCompletionRequest) -> Option<bool> {
    req.chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(|value| value.as_bool())
}

pub(super) fn effective_thinking_enabled_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> Option<bool> {
    request_thinking_enabled(req).or(state.default_thinking_enabled)
}

pub(super) fn effective_thinking_budget_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> EffectiveThinkingBudget {
    resolve_effective_thinking_budget(state, req.thinking_budget_tokens, req.thinking_budget_ms)
}

pub(super) fn resolve_effective_thinking_budget(
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

pub(super) fn thinking_budget_metadata_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
    starts_in_reasoning: bool,
) -> ThinkingBudgetMetadata {
    let effective = effective_thinking_budget_for_request(state, req);
    let applied = effective.configured() && starts_in_reasoning && chat_request_max_tokens(req) > 0;
    ThinkingBudgetRecord::from_effective(effective, applied)
}

pub(super) fn attach_thinking_budget_outcome(
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

pub(super) fn apply_thinking_budget_status_to_metadata(
    metadata: &mut ThinkingBudgetMetadata,
    status: ThinkingBudgetStatus,
) {
    metadata.set_outcome(status.into());
}

pub(super) fn unresolved_request_thinking_budget(
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

pub(super) fn recent_thinking_budget_from_metadata(
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

pub(super) fn recent_thinking_budget_with_status(
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

pub(super) fn attach_cached_thinking_budget_outcome(response: &mut ChatCompletionResponse) {
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

pub(super) fn configure_thinking_budget_for_prompt(
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

pub(super) fn validate_thinking_budget_completion_capacity(
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

pub(super) fn fold_reasoning_into_content_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> bool {
    req.fold_reasoning_into_content
        .unwrap_or(state.fold_reasoning_into_content)
}

pub(super) fn is_false(value: &bool) -> bool {
    !*value
}

pub(super) fn thinking_source_for_request(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> &'static str {
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

pub(super) fn thinking_mode_for_prompt(prompt_text: &str) -> &'static str {
    if prompt_starts_in_reasoning(prompt_text) {
        "reasoning"
    } else {
        "non_reasoning"
    }
}

pub(super) fn effective_chat_template_kwargs(
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

pub(super) fn chat_completion_metadata_from_prompt(
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

pub(super) fn chat_completion_metadata_from_prompt_and_output(
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

pub(super) fn chat_completion_metadata_from_cached_output(
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

pub(super) fn chat_completion_metadata_from_request(
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

pub(super) fn chat_performance_metadata_enabled(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> bool {
    req.include_performance
        .unwrap_or(state.chat_performance_metadata)
}

pub(super) fn chat_config_hash_metadata_enabled(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> bool {
    req.include_config_hashes
        .unwrap_or(state.chat_config_hash_metadata)
}

pub(super) fn duration_ms_f64(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

pub(super) fn duration_ms_u64(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

pub(super) fn adapter_used_for_performance_metadata(state: &AppState) -> String {
    state
        .loaded_adapter_name()
        .unwrap_or_else(|| "base".to_string())
}

pub(super) fn decode_tokens_per_sec_for_performance_metadata(
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

pub(super) fn attach_chat_performance_metadata(
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

pub(super) fn attach_batched_actor_performance_metadata(
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

pub(super) fn content_empty_reason(output: &AssistantOutputParts) -> Option<&'static str> {
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

pub(super) fn ensure_eval_mode_thinking_default(
    chat_template_kwargs: &mut Option<serde_json::Map<String, serde_json::Value>>,
    enabled: bool,
) {
    let kwargs = chat_template_kwargs.get_or_insert_with(serde_json::Map::new);
    kwargs
        .entry("enable_thinking".to_string())
        .or_insert(serde_json::Value::Bool(enabled));
}

pub(super) fn apply_eval_mode_chat_defaults(state: &AppState, req: &mut ChatCompletionRequest) {
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

pub(super) fn apply_eval_mode_batch_defaults(state: &AppState, req: &mut BatchCompletionRequest) {
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
pub(super) fn request_record_from_req(
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
pub(super) fn extract_user_agent(headers: &HeaderMap) -> Option<String> {
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
pub(super) fn extract_client(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-kiln-client")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(64).collect())
}

pub(super) const REASONING_OPEN_TAG: &str = "<think>\n";
pub(super) const REASONING_CLOSE_TAG: &str = "</think>";
