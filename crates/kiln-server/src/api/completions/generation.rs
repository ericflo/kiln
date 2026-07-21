use super::*;

pub(super) async fn generate_real_batched(
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
pub(super) enum StreamTerminalState {
    #[default]
    Pending,
    Complete(std::collections::VecDeque<Event>),
    Failed(String),
    Consumed,
}

#[derive(Clone, Default)]
pub(super) struct StreamTerminal {
    state: std::sync::Arc<std::sync::Mutex<StreamTerminalState>>,
}

impl StreamTerminal {
    pub(super) fn fail(&self, message: impl Into<String>) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, StreamTerminalState::Pending) {
            *state = StreamTerminalState::Failed(message.into());
        }
    }

    pub(super) fn complete(&self, events: std::collections::VecDeque<Event>) {
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

pub(super) fn stream_generation_error_events(message: String) -> std::collections::VecDeque<Event> {
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

pub(super) fn stream_with_terminal(
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
pub(super) const STREAM_TERMINAL_EVENT_CAPACITY: usize = 16;

pub(super) fn drain_terminal_event_buffer(
    mut rx: tokio::sync::mpsc::Receiver<Event>,
) -> std::collections::VecDeque<Event> {
    let mut events = std::collections::VecDeque::new();
    while let Ok(event) = rx.try_recv() {
        events.push_back(event);
    }
    events
}

pub(super) async fn generate_real_batched_streaming(
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

pub(super) async fn generate_mock(
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
