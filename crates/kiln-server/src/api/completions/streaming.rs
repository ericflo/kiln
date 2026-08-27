use super::*;

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
pub(super) fn prompt_starts_in_reasoning(prompt_text: &str) -> bool {
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
pub(super) struct ReasoningSplitter {
    in_reasoning: bool,
    pending: String,
}

#[derive(Default, Debug)]
pub(super) struct ReasoningChunk {
    pub(super) reasoning: Option<String>,
    pub(super) content: Option<String>,
}

impl ReasoningChunk {
    pub(super) fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none()
    }
}

impl ReasoningSplitter {
    pub(super) fn new(starts_in_reasoning: bool) -> Self {
        Self {
            in_reasoning: starts_in_reasoning,
            pending: String::new(),
        }
    }

    pub(super) fn push(&mut self, token: &str) -> ReasoningChunk {
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
    pub(super) fn flush(&mut self) -> ReasoningChunk {
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
// The flat argument list mirrors the CLI-flag/API field set 1:1; a parameter struct would obscure that correspondence, and changing the signature would be a breaking API change.
#[allow(clippy::too_many_arguments)]
pub(super) async fn emit_reasoning_chunk(
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

pub(super) async fn emit_tool_calls_chunk(
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

pub(super) async fn emit_content_chunk(
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
pub(super) struct ToolCallGate {
    enabled: bool,
    confirmed: bool,
    /// Bytes of `content_buf` already emitted on the wire.
    streamed: usize,
}

impl ToolCallGate {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            confirmed: false,
            streamed: 0,
        }
    }

    /// Call after appending new text to `content_buf`. Returns the byte
    /// range of `content_buf` now safe to stream (empty while holding
    /// back or buffering a confirmed tag).
    pub(super) fn advance(&mut self, content_buf: &str) -> std::ops::Range<usize> {
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

    // Test-only: lets the tool-gate tests (completions/tests/mod.rs) assert
    // the gate fired exactly once; the streaming loop only reads the field.
    #[allow(dead_code)]
    pub(super) fn confirmed(&self) -> bool {
        self.confirmed
    }

    /// Unstreamed remainder for end-of-stream emission.
    pub(super) fn unsent<'a>(&self, content_buf: &'a str) -> &'a str {
        &content_buf[self.streamed.min(content_buf.len())..]
    }

    pub(super) fn mark_all_sent(&mut self, content_buf: &str) {
        self.streamed = content_buf.len();
    }
}

/// The OpenAI `stream_options.include_usage` final chunk: empty
/// `choices`, populated `usage`, emitted after the finish chunk and
/// before `[DONE]`.
pub(super) fn usage_chunk_json(
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

pub(super) fn finalized_thinking_budget_status(
    budget: Option<&ThinkingBudget>,
    completion_tokens: usize,
) -> Option<ThinkingBudgetStatus> {
    let mut status = budget.map(ThinkingBudget::status)?;
    if !status.closed && status.trigger.is_none() {
        status.thinking_tokens = completion_tokens;
    }
    Some(status)
}

pub(super) fn streaming_finish_chunk_json(
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
pub(super) struct StreamingTokenTiming {
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

pub(super) fn instant_delta_ms(start: std::time::Instant, end: std::time::Instant) -> f64 {
    end.saturating_duration_since(start).as_secs_f64() * 1_000.0
}

pub(super) fn streaming_token_timing_enabled(req: &ChatCompletionRequest) -> bool {
    req.include_performance == Some(true)
}

// The flat argument list mirrors the CLI-flag/API field set 1:1; a parameter struct would obscure that correspondence, and changing the signature would be a breaking API change.
#[allow(clippy::too_many_arguments)]
pub(super) fn streaming_token_timing_json(
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

// The flat argument list mirrors the CLI-flag/API field set 1:1; a parameter struct would obscure that correspondence, and changing the signature would be a breaking API change.
#[allow(clippy::too_many_arguments)]
pub(super) async fn emit_or_buffer_reasoning_chunk(
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
pub(super) async fn flush_buffered_stream_tail(
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
pub(super) fn stream_tail_record_completion(
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
pub(super) fn truncate_at_matched_stop<'a>(
    text: &'a str,
    finish_reason: &kiln_model::FinishReason,
) -> &'a str {
    if let kiln_model::FinishReason::StopSequence(stop) = finish_reason
        && !stop.is_empty()
        && let Some(idx) = text.find(stop.as_str())
    {
        return &text[..idx];
    }
    text
}

/// Non-streaming variant: split a fully-generated response text into
/// `(reasoning_content, content)` around the same `</think>` boundary the
/// streaming splitter handles. Returns `(None, raw)` when the prompt did not
/// prefill `<think>\n` so non-reasoning models keep emitting plain content.
pub(super) fn split_reasoning_response(
    model_output: &str,
    prompt_text: &str,
) -> (Option<String>, String) {
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
pub(super) struct AssistantOutputParts {
    pub(super) content: String,
    pub(super) reasoning_content: Option<String>,
    pub(super) tool_calls: Option<Vec<serde_json::Value>>,
    pub(super) finish_reason: String,
}

impl AssistantOutputParts {
    pub(super) fn preview_source(&self) -> &str {
        if self.content.is_empty() {
            self.reasoning_content.as_deref().unwrap_or("")
        } else {
            self.content.as_str()
        }
    }
}

pub(super) fn folded_reasoning_content(reasoning: &str, content: &str) -> String {
    let folded = format!("{REASONING_OPEN_TAG}{reasoning}{REASONING_CLOSE_TAG}");
    if content.is_empty() {
        folded
    } else {
        format!("{folded}\n\n{content}")
    }
}

pub(super) fn unfold_reasoning_from_content(content: &str, reasoning: &str) -> String {
    let folded_prefix = format!("{REASONING_OPEN_TAG}{reasoning}{REASONING_CLOSE_TAG}");
    let Some(rest) = content.strip_prefix(&folded_prefix) else {
        return content.to_string();
    };
    rest.strip_prefix("\n\n").unwrap_or(rest).to_string()
}

pub(super) fn response_content_for_cache(content: &str, reasoning: Option<&str>) -> String {
    match reasoning {
        Some(reasoning) if !reasoning.is_empty() => {
            unfold_reasoning_from_content(content, reasoning)
        }
        _ => content.to_string(),
    }
}

pub(super) fn content_with_reasoning_policy(
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

pub(super) fn apply_reasoning_content_policy(
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

pub(super) struct PrefillProgressGuard(CancelHandle);

impl PrefillProgressGuard {
    pub(super) fn new(cancel: CancelHandle) -> Self {
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

pub(super) struct CancelOnDrop(CancelHandle);

impl CancelOnDrop {
    pub(super) fn new(cancel: CancelHandle) -> Self {
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

pub(super) fn tool_call_parsing_allowed(
    tools: Option<&[serde_json::Value]>,
    tool_choice: Option<&serde_json::Value>,
) -> bool {
    normalized_tools_for_cache(tools).is_some()
        && !matches!(tool_choice.and_then(|value| value.as_str()), Some("none"))
}

pub(super) fn request_allows_tool_call_parsing(req: &ChatCompletionRequest) -> bool {
    tool_call_parsing_allowed(req.tools.as_deref(), req.tool_choice.as_ref())
}

pub(super) fn batch_request_allows_tool_call_parsing(req: &BatchCompletionRequest) -> bool {
    tool_call_parsing_allowed(req.tools.as_deref(), req.tool_choice.as_ref())
}

pub(super) fn assistant_output_from_model_output(
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
pub(super) fn assistant_output_from_model_output_stop_aware(
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
    if parsed.content.is_empty()
        && let Some(reasoning) = parsed.reasoning_content.take()
    {
        let truncated = truncate_at_matched_stop(&reasoning, engine_finish).to_string();
        parsed.reasoning_content = (!truncated.is_empty()).then_some(truncated);
    }
    parsed
}

pub(super) fn assistant_output_from_cached_parts(
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

pub(super) fn assistant_output_from_split_parts(
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
pub(super) fn stream_assistant_output_with_stop_reconstruction(
    buffer_tool_content: bool,
    reasoning_content: Option<String>,
    content_buf: &str,
    matched_stop: Option<&str>,
    finish: &str,
) -> AssistantOutputParts {
    if buffer_tool_content && let Some(stop) = matched_stop {
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
    assistant_output_from_split_parts_with_tool_parsing(
        buffer_tool_content,
        reasoning_content,
        content_buf.to_string(),
        finish,
    )
}

pub(super) fn assistant_output_from_split_parts_with_tool_parsing(
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

pub(super) fn content_before_qwen_tool_call(text: &str) -> Option<String> {
    let idx = text.find(QWEN_TOOL_CALL_OPEN_TAG)?;
    Some(text[..idx].trim_end().to_string())
}

pub(super) fn strip_qwen_tool_call_blocks(text: &str) -> String {
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

pub(super) fn openai_tool_calls_from_qwen(calls: &[ParsedToolCall]) -> Vec<serde_json::Value> {
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

pub(super) fn tool_call_deltas_from_openai_calls(
    calls: &[serde_json::Value],
) -> Vec<serde_json::Value> {
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
