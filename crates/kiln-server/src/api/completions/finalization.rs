use super::*;

pub(super) fn response_from_cached_chat_request(
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

pub(super) fn streaming_response_from_cached_chat_request(
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

pub(super) fn chat_request_cache_value_from_response(
    resp: &ChatCompletionResponse,
) -> Option<DeterministicChatRequestCacheValue> {
    Some(chat_request_cache_value_from_completion(
        resp.usage.prompt_tokens,
        cache_value_from_response(resp)?,
    ))
}

pub(super) fn chat_request_cache_value_from_completion(
    prompt_tokens: usize,
    completion: DeterministicCompletionCacheValue,
) -> DeterministicChatRequestCacheValue {
    DeterministicChatRequestCacheValue {
        prompt_tokens,
        completion,
    }
}

pub(super) fn chat_request_cache_value_from_choice(
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

pub(super) fn store_chat_request_cache_from_chat_choices_response(
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

pub(super) async fn zero_chat_choices_response_from_request_cache_hit(
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

pub(super) fn chat_choices_cache_value_from_response(
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

pub(super) fn response_from_cached_completion(
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

pub(super) fn response_from_cached_chat_choices(
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

pub(super) fn streaming_response_from_cached_completion(
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

pub(super) fn empty_chat_completion_response(
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

pub(super) fn empty_chat_completion_streaming_response(
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
