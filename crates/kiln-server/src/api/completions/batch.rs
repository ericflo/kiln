use super::*;

pub(super) const BATCH_MAX_TOTAL_OUTPUTS: usize = 64;

/// Maximum number of choices a single chat completion request may produce.
pub(super) const CHAT_MAX_CHOICES: usize = BATCH_MAX_TOTAL_OUTPUTS;

/// Maximum number of source adapters allowed in a single compose request
/// (`adapters: [...]` on `/v1/chat/completions` and `/v1/completions/batch`).
/// Caps the cheapest DoS shape from §6 of `docs/audits/security-audit-v0.1.md`:
/// each entry triggers a safetensors read and an N-way `merge_concat`, so an
/// unbounded list lets a single request pin CPU + I/O for arbitrarily long.
pub(super) const MAX_COMPOSE_ADAPTERS: usize = 16;

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

pub(super) struct BatchPromptGroup {
    pub(super) messages: Vec<Message>,
    pub(super) prompt_indices: Vec<usize>,
}

#[derive(Serialize)]
pub(super) struct BatchPromptMessageCacheKey<'a> {
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
pub(super) struct DeterministicBatchCacheKeyWire<'a> {
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

pub(super) fn batch_prompt_cache_key(messages: &[Message]) -> Vec<BatchPromptMessageCacheKey<'_>> {
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

pub(super) fn batch_synth_messages(messages: &[Message]) -> Vec<Message> {
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

pub(super) fn batch_prompt_groups(prompts: &[Vec<Message>]) -> Vec<BatchPromptGroup> {
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
pub(super) fn deterministic_batch_cache_key(
    req: &BatchCompletionRequest,
    total_outputs: usize,
) -> Option<String> {
    deterministic_batch_cache_key_with_vocab_size(req, total_outputs, usize::MAX)
}

#[cfg(test)]
pub(super) fn deterministic_batch_cache_key_with_vocab_size(
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

pub(super) fn deterministic_batch_cache_key_with_vocab_size_and_fold(
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

pub(super) fn batch_response_from_cached_value(
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

pub(super) fn batch_response_from_cached_chat_choices(
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

pub(super) fn batch_response_from_cached_chat_choice_groups(
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

pub(super) fn batch_response_from_cached_chat_requests(
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

pub(super) fn batch_response_from_chat_request_cache_hits(
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

pub(super) fn batch_response_from_chat_choices_cache_hits(
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

pub(super) fn cache_value_from_batch_response(
    resp: &BatchCompletionResponse,
) -> DeterministicBatchCacheValue {
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

pub(super) fn chat_request_cache_value_from_batch_item(
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

pub(super) fn store_chat_request_cache_from_batch_response(
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

pub(super) fn chat_choices_cache_value_from_batch_items(
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

pub(super) fn store_chat_choices_cache_from_batch_response(
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

pub(super) fn store_chat_caches_from_batch_response(
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

pub(super) async fn batch_completions(
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

pub(super) async fn batch_completions_inner(
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
