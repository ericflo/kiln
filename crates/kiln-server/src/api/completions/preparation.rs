use super::*;

#[derive(Serialize)]
pub(super) struct ChatPromptMessageCacheKey<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Cow<'a, [serde_json::Value]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

pub(super) fn message_cache_keys(messages: &[Message]) -> Vec<ChatPromptMessageCacheKey<'_>> {
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
pub(super) struct RenderedPromptCacheKey<'a> {
    messages: &'a [ChatPromptMessageCacheKey<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [serde_json::Value]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

pub(super) fn normalized_tools_for_cache(
    tools: Option<&[serde_json::Value]>,
) -> Option<&[serde_json::Value]> {
    tools.filter(|tools| !tools.is_empty())
}

pub(super) fn normalized_tools_option_for_synthetic_request(
    tools: Option<&[serde_json::Value]>,
) -> Option<Vec<serde_json::Value>> {
    normalized_tools_for_cache(tools).map(Vec::from)
}

pub(super) fn normalized_tool_choice_for_cache<'a>(
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

pub(super) fn normalized_chat_template_kwargs_for_cache(
    chat_template_kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    chat_template_kwargs.filter(|kwargs| !kwargs.is_empty())
}

pub(super) fn normalized_tool_choice_option_for_synthetic_request(
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let normalized_tools = normalized_tools_for_cache(tools);
    normalized_tool_choice_for_cache(normalized_tools, tool_choice).cloned()
}

pub(super) fn normalized_message_tool_calls_for_cache(
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

pub(super) fn normalized_tool_call_for_cache(
    tool_call: &serde_json::Value,
) -> Option<serde_json::Value> {
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

pub(super) fn normalized_tool_call_function_for_cache(
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

pub(super) fn parsed_json_argument_for_cache(
    value: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
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

pub(super) fn chat_template_options_from_kwargs(
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

pub(super) fn deterministic_completion_cache_key_for_adapter(
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
pub(super) fn deterministic_completion_cache_key(
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
pub(super) struct DeterministicChatRequestCacheKey<'a> {
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
pub(super) struct DeterministicChatChoicesCacheKey<'a> {
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
pub(super) fn deterministic_chat_request_cache_key(
    req: &ChatCompletionRequest,
    sampling: &SamplingParams,
) -> Result<Option<String>, ApiError> {
    deterministic_chat_request_cache_key_with_vocab_size(req, sampling, usize::MAX)
}

#[cfg(test)]
pub(super) fn deterministic_chat_request_cache_key_with_vocab_size(
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

pub(super) fn request_token_budget_without_server_default(
    req: &ChatCompletionRequest,
) -> Option<usize> {
    match req.thinking_budget_tokens {
        BudgetOverride::Limited(value) => Some(value),
        BudgetOverride::Inherit | BudgetOverride::Unlimited => None,
    }
}

pub(super) fn deterministic_chat_request_cache_key_with_vocab_size_and_fold(
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
pub(super) fn chat_request_sampling_for_cache_key(
    req: &ChatCompletionRequest,
    seed: Option<u64>,
) -> SamplingParams {
    let mut params = sampling_params_for_chat_request(req);
    params.stop = req.stop.clone().unwrap_or_default();
    params.seed = seed;
    params
}

/// Batch-request twin of [`chat_request_sampling_for_cache_key`].
pub(super) fn batch_request_sampling_for_cache_key(
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

pub(super) fn batch_token_budget_without_server_default(
    req: &BatchCompletionRequest,
) -> Option<usize> {
    match req.thinking_budget_tokens {
        BudgetOverride::Limited(value) => Some(value),
        BudgetOverride::Inherit | BudgetOverride::Unlimited => None,
    }
}

pub(super) fn deterministic_chat_request_cache_key_from_chat_choice_with_vocab_size_and_fold(
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
pub(super) fn deterministic_chat_choices_cache_key(
    req: &ChatCompletionRequest,
    n_per: usize,
    sampling: &SamplingParams,
) -> Result<Option<String>, ApiError> {
    deterministic_chat_choices_cache_key_with_vocab_size(req, n_per, sampling, usize::MAX)
}

#[cfg(test)]
pub(super) fn deterministic_chat_choices_cache_key_with_vocab_size(
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

pub(super) fn deterministic_chat_choices_cache_key_with_vocab_size_and_fold(
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

pub(super) fn deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
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

pub(super) fn deterministic_chat_choices_cache_key_from_batch_prompt_with_vocab_size_and_fold(
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

pub(super) fn batch_synth_message_cache_keys(
    messages: &[Message],
) -> Vec<ChatPromptMessageCacheKey<'_>> {
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

pub(super) fn deterministic_chat_request_cache_key_from_batch_prompt_with_vocab_size_and_fold(
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
pub(super) struct NormalizedRequestSamplingKey {
    pub(super) temperature_bits: u32,
    pub(super) stop: Vec<String>,
    pub(super) top_p_bits: u32,
    pub(super) top_k: u32,
    pub(super) min_p_bits: u32,
    pub(super) presence_penalty_bits: u32,
    pub(super) frequency_penalty_bits: u32,
    pub(super) repetition_penalty_bits: u32,
    pub(super) seed: Option<u64>,
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

pub(super) fn normalized_deterministic_request_sampling_key(
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

pub(super) fn normalized_top_p_bits_for_cache(top_p: f32) -> u32 {
    if SamplingParams::top_p_disables_nucleus_filter(top_p) {
        1.0f32.to_bits()
    } else {
        top_p.to_bits()
    }
}

pub(super) fn normalized_min_p_bits_for_cache(min_p: f32) -> u32 {
    if SamplingParams::min_p_is_disabled(min_p) {
        0.0f32.to_bits()
    } else {
        min_p.to_bits()
    }
}

pub(super) fn normalized_ignore_eos_for_cache(sampling: &SamplingParams) -> bool {
    sampling.max_tokens != 0 && sampling.ignore_eos
}

/// Fold alternate spellings of a penalty's no-op value (`-0.0` for the
/// subtractive penalties) onto canonical no-op bits so equivalent
/// requests share cache entries.
pub(super) fn normalized_penalty_bits_for_cache(value: f32, no_op: f32) -> u32 {
    if value == no_op {
        no_op.to_bits()
    } else {
        value.to_bits()
    }
}

pub(super) fn normalized_top_k_for_cache(top_k: u32, vocab_size: usize) -> u32 {
    if top_k != 0 && (top_k as usize) >= vocab_size {
        0
    } else {
        top_k
    }
}

pub(super) fn normalized_stop_for_cache(stop: &[String]) -> Vec<String> {
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

pub(super) fn normalized_stop_for_generation(stop: Option<&[String]>) -> Vec<String> {
    stop.map(normalized_stop_for_cache).unwrap_or_default()
}

pub(super) fn stop_sequences_for_chat_generation(req: &ChatCompletionRequest) -> Vec<String> {
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

pub(super) fn normalized_stop_option_for_synthetic_request(
    stop: Option<&[String]>,
) -> Option<Vec<String>> {
    let stop = normalized_stop_for_generation(stop);
    if stop.is_empty() { None } else { Some(stop) }
}

pub(super) fn resolved_max_tokens(
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
) -> usize {
    max_tokens.or(max_completion_tokens).unwrap_or(2048)
}

pub(super) fn chat_request_max_tokens(req: &ChatCompletionRequest) -> usize {
    resolved_max_tokens(req.max_tokens, req.max_completion_tokens)
}

pub(super) fn batch_request_max_tokens(req: &BatchCompletionRequest) -> usize {
    resolved_max_tokens(req.max_tokens, req.max_completion_tokens)
}

pub(super) fn effective_batch_thinking_budget_for_request(
    state: &AppState,
    req: &BatchCompletionRequest,
) -> EffectiveThinkingBudget {
    resolve_effective_thinking_budget(state, req.thinking_budget_tokens, req.thinking_budget_ms)
}

pub(super) fn batch_completion_metadata_for_request(
    state: &AppState,
    req: &BatchCompletionRequest,
) -> BatchCompletionMetadata {
    BatchCompletionMetadata {
        thinking_budget: effective_batch_thinking_budget_for_request(state, req).into(),
    }
}
