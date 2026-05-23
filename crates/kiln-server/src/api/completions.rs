use anyhow::Context;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use kiln_core::config_hashes::ConfigHashes;
use kiln_core::request::Request;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::{ChatMessage, ChatTemplateOptions};
use kiln_eval::qwen3::ParsedToolCall;
use kiln_model::adapter_merge::{PeftLora, merge_concat};
use kiln_model::lora_loader::LoraWeights;
use kiln_model::{
    CancelHandle, GenerationOutput, ModelRunner, PagedPrefixReuse, SpeculativeConfig, StreamEvent,
};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::batching_engine::{EngineEvent, EngineRequest};
use crate::config::{SpecMethod, SpeculativeDecodingConfig};
use crate::error::ApiError;
use crate::metrics::RequestStatus;
use crate::recent_requests::{FULL_BODY_MAX_CHARS, RequestRecord, now_unix_ms, truncate_chars};
use crate::state::{
    AppState, DeterministicBatchCache, DeterministicBatchCacheClaim, DeterministicBatchCacheItem,
    DeterministicBatchCacheKey, DeterministicBatchCacheValue, DeterministicBatchInFlightState,
    DeterministicChatChoicesCache, DeterministicChatChoicesCacheClaim,
    DeterministicChatChoicesCacheProbe, DeterministicChatChoicesCacheValue,
    DeterministicChatChoicesInFlightState, DeterministicChatRequestCache,
    DeterministicChatRequestCacheClaim, DeterministicChatRequestCacheProbe,
    DeterministicChatRequestCacheValue, DeterministicChatRequestInFlightState,
    DeterministicCompletionCacheClaim, DeterministicCompletionCacheKey,
    DeterministicCompletionCacheProbe, DeterministicCompletionCacheValue,
    DeterministicCompletionInFlightState, ModelBackend, RealPrefixCache,
};

/// Max characters retained in the prompt preview for the recent-requests panel.
const PROMPT_PREVIEW_MAX_CHARS: usize = 120;
/// Max characters retained in the completion preview for the recent-requests panel.
const COMPLETION_PREVIEW_MAX_CHARS: usize = 200;
const QWEN_TOOL_CALL_OPEN_TAG: &str = "<tool_call>";
const QWEN_TOOL_CALL_CLOSE_TAG: &str = "</tool_call>";

fn observe_post_prefill_vram(memory_budget: &std::sync::Arc<crate::state::GpuMemoryBudget>) {
    if let Some(bytes) = kiln_core::vram::detect_used_vram_bytes() {
        memory_budget.observe_prefill_used_vram_bytes(bytes);
    }
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

fn adapter_used_for_performance_metadata(state: &AppState) -> String {
    state
        .loaded_adapter_name
        .read()
        .unwrap()
        .clone()
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
    });
}

struct TimedGenerationOutput {
    output: GenerationOutput,
    ttft: Option<std::time::Duration>,
    decode_duration: Option<std::time::Duration>,
}

impl TimedGenerationOutput {
    fn without_timings(output: GenerationOutput) -> Self {
        Self {
            output,
            ttft: None,
            decode_duration: None,
        }
    }
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
        ..RequestRecord::default()
    }
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
    prompt_text.ends_with(REASONING_OPEN_TAG)
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
async fn emit_or_buffer_reasoning_chunk(
    tx: &tokio::sync::mpsc::Sender<Event>,
    id: &str,
    created: u64,
    model: &str,
    chunk: ReasoningChunk,
    completion_preview_buf: &mut String,
    reasoning_buf: &mut String,
    content_buf: &mut String,
    buffer_content: bool,
) -> bool {
    if tx.is_closed() {
        return false;
    }

    if !buffer_content {
        return emit_reasoning_chunk(
            tx,
            id,
            created,
            model,
            chunk,
            completion_preview_buf,
            reasoning_buf,
            content_buf,
        )
        .await;
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
    }
    !tx.is_closed()
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
        self.0.clear_prefill_progress();
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
    match state.recent_requests.lock() {
        Ok(mut ring) => ring.record(record),
        Err(poisoned) => poisoned.into_inner().record(record),
    }
}

fn record_failed_chat_completion(
    state: &AppState,
    req: &ChatCompletionRequest,
    model: &str,
    request_start: std::time::Instant,
    prompt_tokens: usize,
    error: &ApiError,
) {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    record_recent_request(
        state,
        RequestRecord {
            prompt_tokens: prompt_tokens.min(u32::MAX as usize) as u32,
            completion_tokens: 0,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: "error".to_string(),
            error: Some(error.to_string()),
            thinking_mode: Some(thinking_mode_for_request(req).to_string()),
            prefix_cache: Some("unknown".to_string()),
            ..request_record_from_req(req, &id, model, false)
        },
    );
}

fn adapter_header_value(adapter: Option<String>) -> HeaderValue {
    HeaderValue::from_str(adapter.as_deref().unwrap_or("base"))
        .unwrap_or_else(|_| HeaderValue::from_static("invalid"))
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
    response.headers_mut().insert(
        HeaderName::from_static("x-kiln-loaded-adapter"),
        adapter_header_value(state.loaded_adapter_name.read().unwrap().clone()),
    );
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
        ModelBackend::Real {
            runner,
            batching_engine,
            ..
        } => {
            let batching = if batching_engine.is_some() {
                "enabled"
            } else {
                "disabled"
            };
            let cuda_graph = runner
                .try_read()
                .ok()
                .and_then(|runner| runner.cuda_graph_enabled().ok());
            let cuda_graph = match cuda_graph {
                Some(true) => "enabled",
                Some(false) => "disabled",
                None => "busy",
            };
            (batching, cuda_graph)
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

fn deterministic_completion_cache_key(
    state: &AppState,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    fold_reasoning_into_content: bool,
) -> Option<DeterministicCompletionCacheKey> {
    let greedy = sampling.is_effectively_greedy();
    if !greedy && sampling.seed.is_none() {
        return None;
    }

    // Greedy argmax does not consult RNG or sampling filters, so normalize
    // those fields to maximize equivalent cache hits. Seeded sampling is
    // replayable and must keep every parameter that changes the token path.
    let (temperature_bits, top_p_bits, top_k, seed) = if greedy {
        (0.0f32.to_bits(), 1.0f32.to_bits(), 0, None)
    } else {
        (
            sampling.temperature.to_bits(),
            normalized_top_p_bits_for_cache(sampling.top_p),
            normalized_top_k_for_cache(sampling.top_k, state.model_config.vocab_size),
            sampling.seed,
        )
    };

    Some(DeterministicCompletionCacheKey {
        adapter: state.loaded_adapter_name.read().unwrap().clone(),
        prompt_tokens: prompt_tokens.to_vec(),
        temperature_bits,
        max_tokens: sampling.max_tokens,
        stop: normalized_stop_for_cache(&sampling.stop),
        top_p_bits,
        top_k,
        seed,
        fold_reasoning_into_content,
    })
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
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
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
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
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
    )
}

fn deterministic_chat_request_cache_key_with_vocab_size_and_fold(
    req: &ChatCompletionRequest,
    sampling: &SamplingParams,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Result<Option<String>, ApiError> {
    if req.n.unwrap_or(1) != 1 {
        return Ok(None);
    }

    if req.adapter.is_explicit() || req.adapters.is_some() {
        return Ok(None);
    }

    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let (temperature_bits, stop, top_p_bits, top_k, seed) =
        normalized_deterministic_request_sampling_key(
            sampling.temperature,
            sampling.max_tokens,
            &sampling.stop,
            sampling.top_p,
            sampling.top_k,
            sampling.seed,
            vocab_size,
        );

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
        stop,
        top_p_bits,
        top_k,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key chat request cache: {err}")))
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

    let temperature = requested_or_default_temperature(req.temperature);
    let top_k = requested_or_default_top_k(req.top_k);
    let max_tokens = chat_request_max_tokens(req);
    if max_tokens != 0
        && !SamplingParams::values_are_effectively_greedy(temperature, top_k)
        && seed.is_none()
    {
        return Ok(None);
    }

    let stop = req.stop.as_deref().unwrap_or(&[]);
    let (temperature_bits, stop, top_p_bits, top_k, seed) =
        normalized_deterministic_request_sampling_key(
            temperature,
            max_tokens,
            stop,
            requested_or_default_top_p(req.top_p),
            top_k,
            seed,
            vocab_size,
        );
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
        max_tokens,
        stop,
        top_p_bits,
        top_k,
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
    )
}

fn deterministic_chat_choices_cache_key_with_vocab_size_and_fold(
    req: &ChatCompletionRequest,
    n_per: usize,
    sampling: &SamplingParams,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Result<Option<String>, ApiError> {
    if n_per <= 1 || req.adapter.is_explicit() || req.adapters.is_some() {
        return Ok(None);
    }

    if sampling.max_tokens != 0 && !sampling.is_effectively_greedy() && sampling.seed.is_none() {
        return Ok(None);
    }

    let (temperature_bits, stop, top_p_bits, top_k, seed) =
        normalized_deterministic_request_sampling_key(
            sampling.temperature,
            sampling.max_tokens,
            &sampling.stop,
            sampling.top_p,
            sampling.top_k,
            sampling.seed,
            vocab_size,
        );

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
        stop,
        top_p_bits,
        top_k,
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

    let temperature = requested_or_default_temperature(req.temperature);
    let top_k = requested_or_default_top_k(req.top_k);
    let max_tokens = batch_request_max_tokens(req);
    if max_tokens != 0
        && !SamplingParams::values_are_effectively_greedy(temperature, top_k)
        && seed.is_none()
    {
        return Ok(None);
    }

    let stop = req.stop.as_deref().unwrap_or(&[]);
    let (temperature_bits, stop, top_p_bits, top_k, seed) =
        normalized_deterministic_request_sampling_key(
            temperature,
            max_tokens,
            stop,
            requested_or_default_top_p(req.top_p),
            top_k,
            seed,
            vocab_size,
        );
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
        max_tokens,
        stop,
        top_p_bits,
        top_k,
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

    let temperature = requested_or_default_temperature(req.temperature);
    let top_k = requested_or_default_top_k(req.top_k);
    let max_tokens = batch_request_max_tokens(req);
    if max_tokens != 0
        && !SamplingParams::values_are_effectively_greedy(temperature, top_k)
        && seed.is_none()
    {
        return Ok(None);
    }

    let stop = req.stop.as_deref().unwrap_or(&[]);
    let (temperature_bits, stop, top_p_bits, top_k, seed) =
        normalized_deterministic_request_sampling_key(
            temperature,
            max_tokens,
            stop,
            requested_or_default_top_p(req.top_p),
            top_k,
            seed,
            vocab_size,
        );
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
        max_tokens,
        stop,
        top_p_bits,
        top_k,
        seed,
    })
    .map(Some)
    .map_err(|err| ApiError::internal(format!("failed to key batch chat request cache: {err}")))
}

fn normalized_deterministic_request_sampling_key(
    temperature: f32,
    max_tokens: usize,
    stop: &[String],
    top_p: f32,
    top_k: u32,
    seed: Option<u64>,
    vocab_size: usize,
) -> (u32, Vec<String>, u32, u32, Option<u64>) {
    if max_tokens == 0 {
        return (0.0f32.to_bits(), Vec::new(), 1.0f32.to_bits(), 0, None);
    }

    let stop = normalized_stop_for_cache(stop);
    if SamplingParams::values_are_effectively_greedy(temperature, top_k) {
        return (0.0f32.to_bits(), stop, 1.0f32.to_bits(), 0, None);
    }

    (
        temperature.to_bits(),
        stop,
        normalized_top_p_bits_for_cache(top_p),
        normalized_top_k_for_cache(top_k, vocab_size),
        seed,
    )
}

fn normalized_top_p_bits_for_cache(top_p: f32) -> u32 {
    if SamplingParams::top_p_disables_nucleus_filter(top_p) {
        1.0f32.to_bits()
    } else {
        top_p.to_bits()
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
        },
    }
}

fn store_chat_request_cache_from_chat_choices_response(
    state: &AppState,
    req: &ChatCompletionRequest,
    resp: &ChatCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    if chat_request_max_tokens(req) != 0 || req.adapter.is_explicit() || req.adapters.is_some() {
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
        cache.insert(key, value);
    }
    Ok(())
}

async fn zero_chat_choices_response_from_request_cache_hit(
    state: &AppState,
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
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: cached_output.finish_reason.clone(),
            prefix_cache: Some("completion_cache_hit".to_string()),
            ..request_record_from_req(req, &id, &model, false)
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
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    attach_chat_performance_metadata(
        state,
        req,
        &mut response,
        request_start,
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
            let output = assistant_output_from_cached_parts(
                req,
                completion.text,
                completion.reasoning_content,
                completion.tool_calls,
                completion.finish_reason,
            );
            (completion_tokens, output)
        })
        .collect::<Vec<_>>();
    let metadata = cached_completions
        .first()
        .map(|(_, output)| chat_completion_metadata_from_cached_output(state, req, output))
        .unwrap_or_else(|| chat_completion_metadata_from_request(state, req));
    let fold_reasoning = fold_reasoning_into_content_for_request(state, req);
    let cached_completions = cached_completions
        .into_iter()
        .map(|(completion_tokens, output)| {
            (
                completion_tokens,
                apply_reasoning_content_policy(output, fold_reasoning),
            )
        })
        .collect::<Vec<_>>();
    let preview_source = cached_completions
        .first()
        .map(|(_, output)| output.preview_source())
        .unwrap_or("");
    let completion_tokens = cached_completions
        .iter()
        .map(|(completion_tokens, _)| *completion_tokens)
        .sum::<usize>();
    record_recent_request(
        state,
        RequestRecord {
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: cached.prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: cached_completions
                .first()
                .map(|(_, output)| output.finish_reason.clone())
                .unwrap_or_else(|| "length".to_string()),
            prefix_cache: Some("chat_choices_cache_hit".to_string()),
            ..request_record_from_req(req, &id, &model, false)
        },
    );

    let choices = cached_completions
        .into_iter()
        .enumerate()
        .map(|(index, (completion_tokens, output))| Choice {
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
            completion_tokens,
        })
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
    attach_chat_performance_metadata(
        state,
        req,
        &mut response,
        request_start,
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
    let cached_output = assistant_output_from_cached_parts(
        req,
        cached.text,
        cached.reasoning_content,
        cached.tool_calls,
        cached.finish_reason,
    );
    let preview_source = cached_output.preview_source();

    record_recent_request(
        state,
        RequestRecord {
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: cached_output.finish_reason.clone(),
            prefix_cache: Some("completion_cache_hit".to_string()),
            ..request_record_from_req(req, &id, &model, true)
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
    events.push(Event::default().data(serde_json::to_string(&done_chunk).unwrap()));
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
    resp: &ChatCompletionResponse,
) {
    let Some(value) = cache_value_from_response(resp) else {
        state.completion_cache.lock().unwrap().fail(&key);
        return;
    };
    state.completion_cache.lock().unwrap().complete(key, value);
}

fn fail_deterministic_completion_owner(state: &AppState, key: &DeterministicCompletionCacheKey) {
    state.completion_cache.lock().unwrap().fail(key);
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
    key: String,
    active: bool,
}

impl ChatRequestCacheOwnerGuard {
    fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicChatRequestCache>>,
        key: String,
    ) -> Self {
        Self {
            cache,
            key,
            active: true,
        }
    }

    fn complete(mut self, value: DeterministicChatRequestCacheValue) {
        self.cache.lock().unwrap().complete(self.key.clone(), value);
        self.active = false;
    }
}

impl Drop for ChatRequestCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key);
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
    key: String,
    active: bool,
}

impl ChatChoicesCacheOwnerGuard {
    fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicChatChoicesCache>>,
        key: String,
    ) -> Self {
        Self {
            cache,
            key,
            active: true,
        }
    }

    fn complete(mut self, value: DeterministicChatChoicesCacheValue) {
        self.cache.lock().unwrap().complete(self.key.clone(), value);
        self.active = false;
    }
}

impl Drop for ChatChoicesCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key);
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
    key: Option<String>,
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
    key: Option<String>,
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
    key: Option<String>,
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
    active: bool,
}

impl BatchCacheOwnerGuard {
    fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicBatchCache>>,
        key: DeterministicBatchCacheKey,
    ) -> Self {
        Self {
            cache,
            key,
            active: true,
        }
    }

    fn complete(mut self, value: DeterministicBatchCacheValue) {
        self.cache.lock().unwrap().complete(self.key.clone(), value);
        self.active = false;
    }
}

impl Drop for BatchCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key);
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

/// OpenAI-compatible chat completion request.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// When omitted, the server falls back to its configured `served_model_id`.
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
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
    #[serde(default)]
    pub stream: bool,
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
    /// stable `metadata.performance` response field. When omitted, the server
    /// config default decides.
    #[serde(default)]
    pub include_performance: Option<bool>,
    /// Kiln extension: include model/tokenizer/template/config hashes in the
    /// stable `metadata.config_hashes` response field. When omitted, the
    /// server config default decides.
    #[serde(default)]
    pub include_config_hashes: Option<bool>,
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
    #[serde(default, deserialize_with = "deserialize_optional_content")]
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

/// Accept `content` as either a plain string, an OpenAI-style array of
/// content parts (`[{"type": "text", "text": "..."}, ...]`), `null`, or
/// missing. Text parts are concatenated in order; non-text parts are ignored
/// since kiln is text-only. `null` and missing both yield an empty string —
/// the assistant-tool-calls-only OpenAI shape (`{"role": "assistant",
/// "content": null, "tool_calls": [...]}`) deserializes cleanly.
fn deserialize_optional_content<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Content {
        Text(String),
        Parts(Vec<serde_json::Value>),
    }

    // `Option<T>` handles both the missing-key case (via `#[serde(default)]`
    // on the field) and an explicit `null` value, falling through to the
    // untagged enum for plain strings and arrays.
    match Option::<Content>::deserialize(deserializer)? {
        None => Ok(String::new()),
        Some(Content::Text(s)) => Ok(s),
        Some(Content::Parts(parts)) => {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_hashes: Option<ConfigHashes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance: Option<ChatCompletionPerformanceMetadata>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionPerformanceMetadata {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: Option<f64>,
    pub total_latency_ms: f64,
    pub decode_tokens_per_sec: Option<f64>,
    pub adapter_used: String,
    pub thinking_mode: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
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

/// vLLM-compatible text-completions request subset.
///
/// Kiln supports this endpoint's prompt-logprobs mode so `RemoteTeacher`
/// clients can query a kiln-served teacher through the same `/v1/completions`
/// shape they use for vLLM/sglang.
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
}

#[derive(Debug, Serialize)]
pub struct TextCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
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
    let streaming = req.stream;
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
    let prompt_tokens = tokens_for_text_completion_prompt(state, &req.prompt)?;
    if prompt_tokens.is_empty() {
        return Err(ApiError::completion_invalid_request(
            "prompt must tokenize to at least one token",
        ));
    }

    let prompt_logprobs = match state.backend.as_ref() {
        ModelBackend::Mock { .. } => mock_prompt_logprobs(state, &prompt_tokens, top_k),
        ModelBackend::Real { runner, .. } => {
            real_prompt_logprobs(state, runner, &prompt_tokens, top_k).await?
        }
    };

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let response = TextCompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion",
        created: now_epoch(),
        model,
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

const MAX_COMPLETION_PROMPT_LOGPROBS: usize = 256;

fn validate_prompt_logprobs_top_k(state: &AppState, top_k: usize) -> Result<(), ApiError> {
    if top_k == 0 {
        return Err(ApiError::completion_invalid_request(
            "prompt_logprobs must be greater than 0",
        ));
    }
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
) -> Result<Vec<TokenId>, ApiError> {
    match prompt {
        TextCompletionPrompt::TokenIds(tokens) => Ok(tokens.clone()),
        TextCompletionPrompt::Text(text) => state
            .tokenizer
            .encode(text)
            .map_err(ApiError::tokenization_failed),
    }
}

fn decoded_token_for_logprob(state: &AppState, token_id: TokenId) -> String {
    state.tokenizer.decode(&[token_id]).unwrap_or_default()
}

fn top_k_logprob_map(state: &AppState, row: &[f32], top_k: usize) -> PromptLogprobMap {
    let mut pairs: Vec<(TokenId, f32)> = row
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, logprob)| (idx as TokenId, logprob))
        .collect();
    if top_k < pairs.len() {
        pairs.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(top_k);
    }
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = BTreeMap::new();
    for (rank, (token_id, logprob)) in pairs.into_iter().enumerate() {
        out.insert(
            token_id.to_string(),
            PromptLogprobEntry {
                logprob,
                rank: rank + 1,
                decoded_token: decoded_token_for_logprob(state, token_id),
            },
        );
    }
    out
}

fn prompt_logprobs_from_rows(
    state: &AppState,
    prompt_tokens: &[TokenId],
    logprob_rows: &[Vec<f32>],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    if logprob_rows.len() != prompt_tokens.len() {
        return Err(ApiError::generation_failed(anyhow::anyhow!(
            "prompt-logprobs row count {} did not match prompt token count {}",
            logprob_rows.len(),
            prompt_tokens.len()
        )));
    }

    let mut out = Vec::with_capacity(prompt_tokens.len());
    out.push(None);
    for pos in 1..prompt_tokens.len() {
        out.push(Some(top_k_logprob_map(
            state,
            &logprob_rows[pos - 1],
            top_k,
        )));
    }
    Ok(out)
}

fn mock_prompt_logprobs(
    state: &AppState,
    prompt_tokens: &[TokenId],
    top_k: usize,
) -> Vec<Option<PromptLogprobMap>> {
    let vocab_size = state.model_config.vocab_size.max(1);
    let mut out = Vec::with_capacity(prompt_tokens.len());
    out.push(None);
    for pos in 1..prompt_tokens.len() {
        let mut map = BTreeMap::new();
        for rank in 0..top_k {
            let token_id = prompt_tokens[pos].wrapping_add(rank as u32) as usize % vocab_size;
            let token_id = token_id as TokenId;
            map.insert(
                token_id.to_string(),
                PromptLogprobEntry {
                    logprob: -(rank as f32),
                    rank: rank + 1,
                    decoded_token: decoded_token_for_logprob(state, token_id),
                },
            );
        }
        out.push(Some(map));
    }
    out
}

fn log_softmax_last_dim(x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
    let lse = x
        .log_sum_exp(candle_core::D::Minus1)?
        .unsqueeze(candle_core::D::Minus1)?;
    let broadcast = lse.broadcast_as(x.shape())?;
    Ok((x - broadcast)?)
}

async fn real_prompt_logprobs(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    prompt_tokens: &[TokenId],
    top_k: usize,
) -> Result<Vec<Option<PromptLogprobMap>>, ApiError> {
    let runner = runner.clone();
    let prompt_tokens_owned = prompt_tokens.to_vec();
    let gpu_lock = state.gpu_lock.clone();
    let timeout = state.request_timeout;
    let handle = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Vec<f32>>> {
        let _gpu_guard = gpu_lock.read().unwrap();
        let runner_guard = runner.read().unwrap();
        let device = runner_guard.weights.embed_tokens.device().clone();
        let backend = kiln_model::backend::for_device(&device);
        let mut linear_state =
            kiln_model::forward::LinearAttentionState::new(&runner_guard.config, &device)?;
        let logits = kiln_model::forward::model_forward(
            &*backend,
            &prompt_tokens_owned,
            &runner_guard.weights,
            &runner_guard.config,
            None,
            Some(&mut linear_state),
            runner_guard.active_lora(),
        )
        .context("prompt-logprobs forward pass")?;
        let log_probs = log_softmax_last_dim(&logits).context("prompt-logprobs log_softmax")?;
        let log_probs_2d = log_probs
            .squeeze(0)
            .context("prompt-logprobs squeeze batch dim")?
            .to_dtype(candle_core::DType::F32)
            .context("prompt-logprobs to_dtype f32")?;
        log_probs_2d
            .to_vec2::<f32>()
            .context("prompt-logprobs to host")
    });

    let rows = match tokio::time::timeout(timeout, handle).await {
        Ok(Ok(Ok(rows))) => rows,
        Ok(Ok(Err(err))) => return Err(ApiError::generation_failed(err)),
        Ok(Err(err)) => return Err(ApiError::internal(format!("join error: {err}"))),
        Err(_) => return Err(ApiError::request_timeout(timeout.as_secs())),
    };
    prompt_logprobs_from_rows(state, prompt_tokens, &rows, top_k)
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
    let sampling = sampling_params_for_chat_request(&req);

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
        let chat_choices_cache_key = deterministic_chat_choices_cache_key_with_vocab_size_and_fold(
            &req,
            n_per,
            &sampling,
            state.model_config.vocab_size,
            fold_reasoning_into_content_for_request(state, &req),
        )?;
        let can_hit_chat_choices_cache_before_adapter_work =
            state.active_adapter_name.read().unwrap().is_none();
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
                        &req,
                        &resp,
                        state.model_config.vocab_size,
                    )?;
                    return Ok(Json(resp).into_response());
                }
                DeterministicChatChoicesCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_chat_choices(receiver).await {
                        let resp =
                            response_from_cached_chat_choices(state, &req, request_start, cached);
                        store_chat_request_cache_from_chat_choices_response(
                            state,
                            &req,
                            &resp,
                            state.model_config.vocab_size,
                        )?;
                        return Ok(Json(resp).into_response());
                    }
                }
                DeterministicChatChoicesCacheClaim::Owner => {
                    chat_choices_cache_owner = Some(ChatChoicesCacheOwnerGuard::new(
                        state.chat_choices_cache.clone(),
                        key.clone(),
                    ));
                }
            }
        }

        if can_hit_chat_choices_cache_before_adapter_work
            && let Some(resp) = zero_chat_choices_response_from_request_cache_hit(
                state,
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
            return Ok(Json(resp).into_response());
        }

        let resp = generate_multi_chat_response(state, &req, request_start, n_per).await?;
        store_chat_request_cache_from_chat_choices_response(
            state,
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
        return Ok(Json(resp).into_response());
    }

    let chat_request_cache_key = deterministic_chat_request_cache_key_with_vocab_size_and_fold(
        &req,
        &sampling,
        state.model_config.vocab_size,
        fold_reasoning_into_content_for_request(state, &req),
    )?;
    let can_hit_chat_request_cache_before_adapter_work =
        state.active_adapter_name.read().unwrap().is_none();
    let mut chat_request_cache_owner = None;
    if can_hit_chat_request_cache_before_adapter_work
        && let Some(key) = chat_request_cache_key.as_ref()
    {
        if req.stream {
            let claim = state.chat_request_cache.lock().unwrap().claim(key);
            match claim {
                DeterministicChatRequestCacheClaim::Hit(cached) => {
                    return Ok(streaming_response_from_cached_chat_request(
                        state,
                        &req,
                        request_start,
                        cached,
                    ));
                }
                DeterministicChatRequestCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_chat_request(receiver).await {
                        return Ok(streaming_response_from_cached_chat_request(
                            state,
                            &req,
                            request_start,
                            cached,
                        ));
                    }
                }
                DeterministicChatRequestCacheClaim::Owner => {
                    chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                        state.chat_request_cache.clone(),
                        key.clone(),
                    ));
                }
            }
        } else {
            let claim = state.chat_request_cache.lock().unwrap().claim(key);
            match claim {
                DeterministicChatRequestCacheClaim::Hit(cached) => {
                    let resp =
                        response_from_cached_chat_request(state, &req, request_start, cached);
                    return Ok(Json(resp).into_response());
                }
                DeterministicChatRequestCacheClaim::Wait(receiver) => {
                    if let Some(cached) = wait_for_deterministic_chat_request(receiver).await {
                        let resp =
                            response_from_cached_chat_request(state, &req, request_start, cached);
                        return Ok(Json(resp).into_response());
                    }
                }
                DeterministicChatRequestCacheClaim::Owner => {
                    chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                        state.chat_request_cache.clone(),
                        key.clone(),
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
    let prompt_tokens = encode_prompt_tokens(state, &prompt_text)?;

    if sampling.max_tokens == 0 {
        let cache_value = DeterministicChatRequestCacheValue {
            prompt_tokens: prompt_tokens.len(),
            completion: DeterministicCompletionCacheValue {
                text: String::new(),
                reasoning_content: None,
                tool_calls: None,
                finish_reason: "length".to_string(),
                completion_tokens: 0,
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
            return Ok(empty_chat_completion_streaming_response(
                state,
                &req,
                prompt_tokens.len(),
                request_start,
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
        );
        return Ok(Json(resp).into_response());
    }

    // If `adapters` is set, synthesize (or reuse cached) composed adapter on
    // disk. Runs regardless of backend so the cache is populated even in mock
    // mode tests; only the actual hot-swap is gated on the Real backend.
    let composed_target: Option<ComposedTarget> = if let Some(list) = req.adapters.as_deref() {
        Some(
            synthesize_composed_adapter(
                &state.adapter_dir,
                list,
                state.composed_cache_max_bytes,
                state.composed_cache_max_entries,
            )
            .await?,
        )
    } else {
        None
    };

    // Ensure the correct LoRA adapter is active for this request.
    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if let Some(ref target) = composed_target {
            ensure_composed_adapter_swap(state, runner, target).await?;
        } else {
            ensure_adapter(state, runner, &req.adapter, &adapter_request_id).await?;
        }
    }

    let completion_cache_key = deterministic_completion_cache_key(
        state,
        &prompt_tokens,
        &sampling,
        fold_reasoning_into_content_for_request(state, &req),
    );
    let mut completion_cache_owner = false;
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
                    return Ok(streaming_response_from_cached_chat_request(
                        state,
                        &req,
                        request_start,
                        chat_cache_value,
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
                        return Ok(streaming_response_from_cached_chat_request(
                            state,
                            &req,
                            request_start,
                            chat_cache_value,
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
                    return Ok(Json(resp).into_response());
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
                        return Ok(Json(resp).into_response());
                    }
                }
                DeterministicCompletionCacheClaim::Owner => {
                    completion_cache_owner = true;
                }
            }
        }
    }

    if req.stream {
        match state.backend.as_ref() {
            ModelBackend::Real {
                runner,
                block_manager,
                paged_cache,
                prefix_cache,
                batching_engine,
                decode_batcher,
            } => {
                if let Some(batching_engine) = batching_engine {
                    generate_real_batched_streaming(
                        state,
                        batching_engine,
                        &prompt_text,
                        &prompt_tokens,
                        &sampling,
                        &req,
                        request_start,
                    )
                    .await
                } else {
                    generate_real_streaming(
                        state,
                        runner,
                        block_manager,
                        paged_cache,
                        prefix_cache,
                        decode_batcher,
                        &prompt_text,
                        &prompt_tokens,
                        &sampling,
                        &req,
                        request_start,
                        completion_cache_key.clone(),
                        chat_request_cache_key.clone(),
                        chat_request_cache_owner.take(),
                    )
                    .await
                }
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
                batching_engine,
                ..
            } => {
                let generation = if let Some(batching_engine) = batching_engine {
                    generate_real_batched(
                        state,
                        batching_engine,
                        &prompt_text,
                        &prompt_tokens,
                        &sampling,
                        &req,
                        request_start,
                    )
                    .await
                } else {
                    generate_real(
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
                };
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
                            request_start,
                            prompt_tokens.len(),
                            &err,
                        );
                        if completion_cache_owner && let Some(key) = completion_cache_key.as_ref() {
                            fail_deterministic_completion_owner(state, key);
                        }
                        return Err(err);
                    }
                };
                if let Some(key) = completion_cache_key.clone() {
                    if completion_cache_owner {
                        complete_deterministic_completion_owner(state, key, &resp);
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
                            request_start,
                            prompt_tokens.len(),
                            &err,
                        );
                        if completion_cache_owner && let Some(key) = completion_cache_key.as_ref() {
                            fail_deterministic_completion_owner(state, key);
                        }
                        return Err(err);
                    }
                };
                if let Some(key) = completion_cache_key.clone() {
                    if completion_cache_owner {
                        complete_deterministic_completion_owner(state, key, &resp);
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
    }
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
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    target_adapter: Option<String>,
    request_id: &str,
    reason: &str,
) -> Result<(), ApiError> {
    let current = state.loaded_adapter_name.read().unwrap().clone();
    if target_adapter == current {
        return Ok(());
    }

    match target_adapter.clone() {
        Some(name) => {
            validate_compose_name(&name)?;
            // Resolve adapter path
            let adapter_path = state.adapter_dir.join(&name);
            if !adapter_path.exists() {
                return Err(ApiError::adapter_not_found(&name));
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
            let loaded_name = state.loaded_adapter_name.clone();

            tokio::task::spawn_blocking(move || {
                let lora = LoraWeights::load(&adapter_path, num_layers, &device)
                    .map_err(|e| format!("{e}"))?;
                let mut guard = runner.write().unwrap();
                guard.swap_lora(Some(lora));
                *loaded_name.write().unwrap() = Some(adapter_name);
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
            let loaded_name = state.loaded_adapter_name.clone();

            tokio::task::spawn_blocking(move || {
                let mut guard = runner.write().unwrap();
                guard.swap_lora(None);
                *loaded_name.write().unwrap() = None;
            })
            .await
            .map_err(|e| ApiError::internal(format!("join error: {e}")))?;
            state.clear_real_prefix_cache();
        }
    }

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
    /// Stable cache name embedded in `loaded_adapter_name` once swapped in,
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
/// name and the suffix of `loaded_adapter_name`.
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
///
/// After a fresh synthesize, runs LRU eviction over the parent `.composed/`
/// directory to keep total entries below `max_entries` and total bytes below
/// `max_bytes` (oldest mtime first). Either limit set to `None` disables that
/// dimension. Eviction is best-effort — failures are logged and the request
/// still succeeds. Cache hits also refresh the entry's mtime so reuse counts
/// as recency for LRU ordering.
async fn synthesize_composed_adapter(
    adapter_dir: &Path,
    adapters: &[AdapterRef],
    max_bytes: Option<u64>,
    max_entries: Option<u64>,
) -> Result<ComposedTarget, ApiError> {
    let hash = composition_hash(adapters);
    let active_name = format!("__composed:{hash}");
    let composed_root = adapter_dir.join(".composed");
    let cache_dir = composed_root.join(&hash);

    if cache_dir.exists() {
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

    // Confirm every source exists before doing any merge work.
    let mut source_paths: Vec<(String, f32, PathBuf)> = Vec::with_capacity(adapters.len());
    for src in adapters {
        let path = adapter_dir.join(&src.name);
        if !path.exists() || !path.is_dir() {
            return Err(ApiError::adapter_not_found(&src.name));
        }
        source_paths.push((src.name.clone(), src.scale, path));
    }

    let composed_root_for_task = composed_root.clone();
    let cache_dir_for_task = cache_dir.clone();
    let merge_result = tokio::task::spawn_blocking(move || -> Result<(), String> {
        std::fs::create_dir_all(&composed_root_for_task)
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
        // LRU eviction runs in the same blocking task — keeps the request
        // path off the runtime and avoids racing with another synthesize for
        // the same root within this request.
        evict_composed_cache_lru(&composed_root_for_task, max_bytes, max_entries);
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
        // entries are 16-hex-digit hash directories.
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
/// Mirrors `ensure_adapter`'s two-phase RwLock pattern (read device + num
/// layers, load weights outside any lock, then write-lock to swap). No-op if
/// the composed adapter is already active.
async fn ensure_composed_adapter_swap(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<ModelRunner>>,
    target: &ComposedTarget,
) -> Result<(), ApiError> {
    {
        let current = state.loaded_adapter_name.read().unwrap();
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
    let active_name = state.loaded_adapter_name.clone();
    let cache_dir = target.cache_dir.clone();
    let composed_active = target.active_name.clone();

    tokio::task::spawn_blocking(move || {
        let lora =
            LoraWeights::load(&cache_dir, num_layers, &device).map_err(|e| format!("{e}"))?;
        let mut guard = runner.write().unwrap();
        guard.swap_lora(Some(lora));
        *active_name.write().unwrap() = Some(composed_active);
        Ok::<(), String>(())
    })
    .await
    .map_err(|e| ApiError::internal(format!("join error: {e}")))?
    .map_err(ApiError::adapter_load_failed)?;
    state.clear_real_prefix_cache();
    if state.eval_mode {
        tracing::warn!(
            adapter = %target.active_name,
            "composed adapter transition during eval mode"
        );
    }

    Ok(())
}

/// Generate using the real ModelRunner with paged KV cache.
async fn generate_real_batched(
    state: &AppState,
    batching_engine: &crate::batching_engine::BatchingEngineHandle,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
) -> Result<ChatCompletionResponse, ApiError> {
    let prompt_token_count = prompt_tokens.len();
    let request_id = Uuid::new_v4();
    let cancel = CancelHandle::with_prefill_progress_gauge(
        state.metrics.request_prefill_tokens_completed.clone(),
    );
    let adapter = state.loaded_adapter_name.read().unwrap().clone();
    let mut events = batching_engine
        .enqueue(EngineRequest {
            request_id,
            prompt_tokens: prompt_tokens.to_vec(),
            sampling: sampling.clone(),
            adapter,
            cancel: cancel.clone(),
        })
        .await
        .map_err(ApiError::generation_failed)?;

    let timeout = state.request_timeout;
    let mut first_token_at: Option<std::time::Instant> = None;
    let collect = async {
        loop {
            match events.recv().await {
                Some(EngineEvent::Token(_token)) => {
                    first_token_at.get_or_insert_with(std::time::Instant::now);
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
    observe_post_prefill_vram(&state.memory_budget);

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
    let completion_tokens =
        completion_usage_tokens(output.completion_tokens, &output.finish_reason);
    let assistant_output =
        assistant_output_from_model_output(req, &output.text, prompt_text, finish_reason);
    let metadata =
        chat_completion_metadata_from_prompt_and_output(state, req, prompt_text, &assistant_output);
    let assistant_output = apply_reasoning_content_policy(
        assistant_output,
        fold_reasoning_into_content_for_request(state, req),
    );
    let preview_source = assistant_output.preview_source();
    record_recent_request(
        state,
        RequestRecord {
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: assistant_output.finish_reason.clone(),
            thinking_mode: Some(thinking_mode_for_prompt(prompt_text).to_string()),
            prefix_cache: Some("batching_engine".to_string()),
            ..request_record_from_req(req, &id, &model, false)
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
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    let ttft = first_token_at.map(|instant| instant.duration_since(request_start));
    attach_chat_performance_metadata(state, req, &mut response, request_start, ttft, None);
    Ok(response)
}

async fn generate_real_batched_streaming(
    state: &AppState,
    batching_engine: &crate::batching_engine::BatchingEngineHandle,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
) -> Result<Response, ApiError> {
    let prompt_token_count = prompt_tokens.len();
    let request_id = Uuid::new_v4();
    let cancel = CancelHandle::with_prefill_progress_gauge(
        state.metrics.request_prefill_tokens_completed.clone(),
    );
    let adapter = state.loaded_adapter_name.read().unwrap().clone();
    let events = batching_engine
        .enqueue(EngineRequest {
            request_id,
            prompt_tokens: prompt_tokens.to_vec(),
            sampling: sampling.clone(),
            adapter,
            cancel: cancel.clone(),
        })
        .await
        .map_err(ApiError::generation_failed)?;

    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_epoch();
    let timeout = state.request_timeout;
    let tokenizer = state.tokenizer.clone();
    let metrics = state.metrics.clone();
    let state_for_record = state.clone();
    let prompt_text_full = last_user_message_text(req);
    let prompt_preview = truncate_chars(&prompt_text_full, PROMPT_PREVIEW_MAX_CHARS);
    let prompt_full = truncate_chars(&prompt_text_full, FULL_BODY_MAX_CHARS);
    let req_adapter = req.adapter.request_adapter_name();
    let req_temperature = req.temperature;
    let req_top_p = req.top_p;
    let req_max_tokens = Some(chat_request_max_tokens(req).min(u32::MAX as usize) as u32);
    let thinking_mode = thinking_mode_for_prompt(prompt_text).to_string();
    let prompt_starts_in_reasoning = prompt_starts_in_reasoning(prompt_text);
    let buffer_tool_content = request_allows_tool_call_parsing(req);
    let batching_engine = batching_engine.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);

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
            let mut decoded_prefix = String::new();
            let record = |finish_reason: String, completion: &str, completion_tokens: u32| {
                let record = RequestRecord {
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
                    ttft_ms: None,
                    error: None,
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
            let deadline = tokio::time::Instant::now() + timeout;
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
                            Some(EngineEvent::Token(token)) => {
                                generated_tokens.push(token);
                                completion_token_count = completion_token_count.saturating_add(1);
                                metrics.add_tokens(1);

                                let decoded = tokenizer
                                    .decode(&generated_tokens)
                                    .unwrap_or_else(|_| decoded_prefix.clone());
                                let delta = decoded
                                    .strip_prefix(&decoded_prefix)
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| tokenizer.decode(&[token]).unwrap_or_default());
                                decoded_prefix = decoded;
                                let chunk = reasoning_splitter.push(&delta);
                                if !emit_or_buffer_reasoning_chunk(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    chunk,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    buffer_tool_content,
                                )
                                .await
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
                            }
                            Some(EngineEvent::Done { output }) => {
                                let finish = match output.finish_reason {
                                    kiln_model::FinishReason::Eos => "stop",
                                    kiln_model::FinishReason::MaxTokens => "length",
                                    kiln_model::FinishReason::StopSequence(_) => "stop",
                                };
                                let trailing = reasoning_splitter.flush();
                                if !emit_or_buffer_reasoning_chunk(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    trailing,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    buffer_tool_content,
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
                                let assistant_output =
                                    assistant_output_from_split_parts_with_tool_parsing(
                                        buffer_tool_content,
                                        reasoning_content,
                                        content_buf.clone(),
                                        finish,
                                    );
                                if assistant_output.tool_calls.is_some()
                                    && !assistant_output.content.is_empty()
                                    && !emit_content_chunk(
                                        &tx,
                                        &id,
                                        created,
                                        &model,
                                        assistant_output.content.clone(),
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
                                if let Some(tool_calls) = assistant_output.tool_calls.as_deref() {
                                    if !emit_tool_calls_chunk(
                                        &tx,
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
                                    && !assistant_output.content.is_empty()
                                    && !emit_content_chunk(
                                        &tx,
                                        &id,
                                        created,
                                        &model,
                                        assistant_output.content.clone(),
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
                                let _ = tx
                                    .send(Event::default().data(serde_json::to_string(&chunk).unwrap()))
                                    .await;
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                record(
                                    finish,
                                    &record_completion,
                                    output.completion_tokens as u32,
                                );
                                return;
                            }
                            Some(EngineEvent::Error(err)) => {
                                tracing::error!(error = %err, "batched streaming generation failed");
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                record(
                                    "error".to_string(),
                                    &completion_buf,
                                    completion_token_count,
                                );
                                return;
                            }
                            None => {
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
                        break;
                    }
                }
            }

            if timed_out {
                cancel.cancel();
                let _ = batching_engine.cancel(request_id).await;
                let trailing = reasoning_splitter.flush();
                let _ = emit_or_buffer_reasoning_chunk(
                    &tx,
                    &id,
                    created,
                    &model,
                    trailing,
                    &mut completion_buf,
                    &mut reasoning_buf,
                    &mut content_buf,
                    buffer_tool_content,
                )
                .await;
                let timeout_chunk = ChatCompletionChunk {
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
                        finish_reason: Some("timeout".to_string()),
                    }],
                };
                let _ = tx
                    .send(Event::default().data(serde_json::to_string(&timeout_chunk).unwrap()))
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

async fn generate_real(
    state: &AppState,
    runner: &std::sync::Arc<std::sync::RwLock<kiln_model::ModelRunner>>,
    block_manager: &std::sync::Arc<std::sync::Mutex<kiln_core::block::BlockManager>>,
    paged_cache: &std::sync::Arc<kiln_model::PagedKvCache>,
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
    // state.loaded_adapter_name reflects the actually-loaded adapter (a
    // `__composed:<hash>` name when the request used `adapters: [...]`), which
    // is what we want as the cache key — distinct compositions and the base
    // model must not share cached blocks.
    let adapter = state.loaded_adapter_name.read().unwrap().clone();

    let gpu_lock = state.gpu_lock.clone();
    let memory_budget = state.memory_budget.clone();
    let metrics = state.metrics.clone();
    let timeout = state.request_timeout;
    let prefix_cache_diagnostic = std::sync::Arc::new(std::sync::Mutex::new("unknown"));
    let prefix_cache_diagnostic_inner = prefix_cache_diagnostic.clone();
    // Cooperative cancellation: `tokio::time::timeout` cancels the outer
    // future, but `spawn_blocking` does not honor that — the closure keeps
    // running on the blocking pool, holding `runner.read()` and
    // `prefix_cache.lock()` for the rest of generation. The next request
    // races against still-held state and 5xx's. On timeout we signal this
    // handle and `.await` the join handle so locks release before we
    // respond. See issue #664.
    let cancel = CancelHandle::with_prefill_progress_gauge(
        state.metrics.request_prefill_tokens_completed.clone(),
    );
    let cancel_inner = cancel.clone();
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
                    *prefix_cache_diagnostic_inner.lock().unwrap() = "disabled";
                    runner_guard
                        .generate_paged_shared_tokens(
                            &prompt_tokens,
                            &params,
                            bm.as_ref(),
                            pc.as_ref(),
                            Some(&cancel_inner),
                        )
                        .map(TimedGenerationOutput::without_timings)
                } else {
                    let (hit, should_register_on_miss) = {
                        let mut cache = prefix_cache.lock().unwrap();
                        let should_lookup = cache.should_lookup_prompt(&prompt_tokens);
                        let hit = if should_lookup {
                            cache.lookup(&adapter, &prompt_tokens)?
                        } else {
                            None
                        };
                        let should_register = cache.should_register_prompt(&prompt_tokens);
                        (hit, should_register)
                    };
                    if hit.is_none() && !should_register_on_miss {
                        *prefix_cache_diagnostic_inner.lock().unwrap() = "skipped";
                        return runner_guard
                            .generate_paged_shared_tokens(
                                &prompt_tokens,
                                &params,
                                bm.as_ref(),
                                pc.as_ref(),
                                Some(&cancel_inner),
                            )
                            .map(TimedGenerationOutput::without_timings);
                    }

                    *prefix_cache_diagnostic_inner.lock().unwrap() =
                        if hit.is_some() { "hit" } else { "miss" };
                    let hit_entry_id = hit.as_ref().map(|hit| hit.entry_id);
                    let cached_prefix = hit.map(|hit| PagedPrefixReuse {
                        cached_tokens: hit.cached_tokens,
                        block_ids: hit.block_ids,
                        linear_state: hit.linear_state,
                        next_token: hit.next_token,
                    });

                    let result = runner_guard.generate_paged_shared_tokens_with_prefix_cache(
                        &prompt_tokens,
                        &params,
                        bm.as_ref(),
                        pc.as_ref(),
                        cached_prefix,
                        Some(&cancel_inner),
                    );

                    let mut output = match result {
                        Ok(output) => {
                            metrics.observe_prefill_duration(output.prefill_duration.as_secs_f64());
                            metrics.observe_decode_duration(output.decode_duration.as_secs_f64());
                            output
                        }
                        Err(err) => {
                            if let Some(entry_id) = hit_entry_id {
                                let mut cache = prefix_cache.lock().unwrap();
                                cache.release_hit(entry_id);
                            }
                            return Err(err);
                        }
                    };
                    let registration = output.registration.take();
                    let extra_registrations = std::mem::take(&mut output.extra_registrations);
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
                            retained_blocks.extend(outcome.retained_blocks);
                            evicted_blocks.extend(outcome.evicted_blocks);
                        }
                        for registration in extra_registrations {
                            let outcome = cache.register(adapter.clone(), registration);
                            retained_blocks.extend(outcome.retained_blocks);
                            evicted_blocks.extend(outcome.evicted_blocks);
                        }
                    }

                    let mut blocks_to_free: Vec<u32> = allocated_blocks
                        .into_iter()
                        .filter(|block_id| !retained_blocks.contains(block_id))
                        .collect();
                    blocks_to_free.extend(evicted_blocks);
                    // #673: never free a block that the prefix cache has just
                    // claimed for the new entry, and never queue the same
                    // block twice in one free call.
                    debug_assert!(
                        blocks_to_free
                            .iter()
                            .all(|id| !retained_blocks.contains(id)),
                        "blocks_to_free overlaps retained_blocks: free={blocks_to_free:?} retained={retained_blocks:?}",
                    );
                    debug_assert!(
                        {
                            let mut seen =
                                std::collections::HashSet::with_capacity(blocks_to_free.len());
                            blocks_to_free.iter().all(|id| seen.insert(*id))
                        },
                        "blocks_to_free contains duplicate block IDs: {blocks_to_free:?}",
                    );
                    if !blocks_to_free.is_empty() {
                        let mut bm_guard = bm.lock().unwrap();
                        bm_guard.free_all(&blocks_to_free);
                    }

                    Ok(TimedGenerationOutput {
                        output: output.output,
                        ttft: Some(output.prefill_duration),
                        decode_duration: Some(output.decode_duration),
                    })
                }
            }
            ResolvedSpeculativeMode::SkipLayer(spec_config) => {
                *prefix_cache_diagnostic_inner.lock().unwrap() = "not_used_speculative";
                if params.temperature == 0.0 {
                    runner_guard
                        .generate_paged_speculative_shared_tokens(
                            &prompt_tokens,
                            &params,
                            bm.as_ref(),
                            pc.as_ref(),
                            &spec_config,
                            Some(&cancel_inner),
                        )
                        .map(TimedGenerationOutput::without_timings)
                } else {
                    let flat_spec_config = SpeculativeConfig {
                        num_speculative_tokens: spec_config.num_speculative_tokens.min(4),
                        draft_layers: spec_config.draft_layers,
                    };
                    runner_guard
                        .generate_speculative(&prompt, &params, &flat_spec_config)
                        .map(TimedGenerationOutput::without_timings)
                }
            }
            ResolvedSpeculativeMode::Mtp => {
                *prefix_cache_diagnostic_inner.lock().unwrap() = "not_used_speculative";
                let output = runner_guard.generate_mtp_speculative(&prompt, &params)?;
                Ok(TimedGenerationOutput::without_timings(GenerationOutput {
                    text: output.text,
                    token_ids: output.token_ids,
                    finish_reason: output.finish_reason,
                }))
            }
        }
    });

    tokio::pin!(generation);
    let output = match tokio::time::timeout(timeout, &mut generation).await {
        Ok(join_result) => match join_result {
            Ok(Ok(output)) => {
                observe_post_prefill_vram(&memory_budget);
                cancel.clear_prefill_progress();
                output
            }
            Ok(Err(err)) => {
                observe_post_prefill_vram(&memory_budget);
                cancel.clear_prefill_progress();
                tracing::error!(error = %format!("{err:#}"), "real generation failed");
                return Err(ApiError::generation_failed(err));
            }
            Err(err) => {
                observe_post_prefill_vram(&memory_budget);
                cancel.clear_prefill_progress();
                return Err(ApiError::internal(format!("join error: {err}")));
            }
        },
        Err(_) => {
            // Signal cooperative cancellation, then await the join handle so
            // the spawn_blocking closure releases `runner.read()` /
            // `prefix_cache.lock()` before we return. Without this drain,
            // subsequent /v1/chat/completions requests race against still-
            // held state and 5xx with "prefill f..." (issue #664). The
            // decode loops poll `is_cancelled()` between tokens, so this
            // typically returns within one decode step after we signal.
            cancel.cancel();
            let _ = generation.await;
            cancel.clear_prefill_progress();
            return Err(ApiError::request_timeout(timeout.as_secs()));
        }
    };

    let ttft = output.ttft;
    let decode_duration = output.decode_duration;
    let output = output.output;

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
    let completion_tokens = completion_usage_tokens(output.token_ids.len(), &output.finish_reason);
    let prefix_cache = *prefix_cache_diagnostic.lock().unwrap();
    // Qwen3.5's chat template prefills `<think>\n` into the assistant turn,
    // so the model emits chain-of-thought directly and closes with
    // `</think>` before the actual answer. Split into the llama.cpp /
    // DeepSeek-shaped `(reasoning_content, content)` pair so OpenAI-compat
    // clients can render the two channels separately. For non-reasoning
    // templates this returns `(None, output.text)` and the response shape
    // is byte-identical to before.
    let assistant_output =
        assistant_output_from_model_output(req, &output.text, prompt_text, finish_reason);
    let metadata =
        chat_completion_metadata_from_prompt_and_output(state, req, prompt_text, &assistant_output);
    let assistant_output = apply_reasoning_content_policy(
        assistant_output,
        fold_reasoning_into_content_for_request(state, req),
    );

    // Recent-requests preview wants the user-visible answer, but the
    // reasoning often dominates the first few hundred chars. Show
    // reasoning when the answer is empty (still mid-thought at
    // max_tokens cutoff) so the dashboard isn't blank.
    let preview_source = assistant_output.preview_source();
    record_recent_request(
        state,
        RequestRecord {
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
            completion_full: Some(truncate_chars(preview_source, FULL_BODY_MAX_CHARS)),
            prompt_tokens: prompt_token_count as u32,
            completion_tokens: completion_tokens as u32,
            duration_ms: request_start.elapsed().as_millis() as u64,
            finish_reason: assistant_output.finish_reason.clone(),
            thinking_mode: Some(thinking_mode_for_prompt(prompt_text).to_string()),
            prefix_cache: Some(prefix_cache.to_string()),
            ..request_record_from_req(req, &id, &model, false)
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
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    attach_chat_performance_metadata(
        state,
        req,
        &mut response,
        request_start,
        ttft,
        decode_duration,
    );
    Ok(response)
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
    paged_cache: &std::sync::Arc<kiln_model::PagedKvCache>,
    prefix_cache: &std::sync::Arc<std::sync::Mutex<RealPrefixCache>>,
    decode_batcher: &Option<std::sync::Arc<kiln_model::DecodeBatcher>>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
    sampling: &SamplingParams,
    req: &ChatCompletionRequest,
    request_start: std::time::Instant,
    completion_cache_key: Option<DeterministicCompletionCacheKey>,
    chat_request_cache_key: Option<String>,
    chat_request_cache_owner: Option<ChatRequestCacheOwnerGuard>,
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
    let decode_batcher = decode_batcher.clone();
    let prompt = prompt_text.to_owned();
    let prompt_token_count = prompt_tokens.len();
    let prompt_tokens = prompt_tokens.to_vec();
    let params = sampling.clone();
    // For prefix-cache keying. After ensure_adapter / ensure_composed_adapter_swap,
    // state.loaded_adapter_name reflects the actually-loaded adapter (a
    // `__composed:<hash>` name when the request used `adapters: [...]`), which
    // is what we want as the cache key — distinct compositions and the base
    // model must not share cached blocks.
    let adapter = state.loaded_adapter_name.read().unwrap().clone();
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.served_model_id.clone());
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = now_epoch();
    let gpu_lock = state.gpu_lock.clone();
    let memory_budget = state.memory_budget.clone();
    let timeout = state.request_timeout;
    let decode_stats = state.decode_stats.clone();
    let metrics = state.metrics.clone();
    let state_for_record = state.clone();
    let completion_cache = state.completion_cache.clone();
    let chat_request_cache = state.chat_request_cache.clone();
    let prompt_text_full = last_user_message_text(req);
    let prompt_preview = truncate_chars(&prompt_text_full, PROMPT_PREVIEW_MAX_CHARS);
    let prompt_full = truncate_chars(&prompt_text_full, FULL_BODY_MAX_CHARS);
    let req_adapter = req.adapter.request_adapter_name();
    let req_temperature = req.temperature;
    let req_top_p = req.top_p;
    let req_max_tokens = Some(chat_request_max_tokens(req).min(u32::MAX as usize) as u32);
    let buffer_tool_content = request_allows_tool_call_parsing(req);
    let thinking_mode = thinking_mode_for_prompt(&prompt).to_string();
    let prefix_cache_diagnostic = std::sync::Arc::new(std::sync::Mutex::new("unknown"));

    // Use a tokio mpsc channel to bridge sync generation -> async SSE stream.
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);

    // Spawn a task that runs the blocking generation and converts to SSE events.
    tokio::task::spawn({
        let id = completion_id.clone();
        let model = model.clone();
        let mut chat_request_cache_owner = chat_request_cache_owner;
        async move {
            // Accumulate the assistant content so we can store a preview in the
            // recent-requests ring once generation completes (or times out, or
            // the client disconnects).
            let mut completion_buf = String::new();
            let mut reasoning_buf = String::new();
            let mut content_buf = String::new();
            let mut completion_token_count: u32 = 0;

            let record = |finish_reason: String, completion: &str, completion_tokens: u32| {
                let record = RequestRecord {
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
                    prefix_cache: Some((*prefix_cache_diagnostic.lock().unwrap()).to_string()),
                    adapter: req_adapter.clone(),
                    temperature: req_temperature,
                    top_p: req_top_p,
                    max_tokens: req_max_tokens,
                    ttft_ms: None,
                    error: None,
                };
                record_recent_request(&state_for_record, record);
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
                record(
                    "client_disconnect".to_string(),
                    &completion_buf,
                    completion_token_count,
                );
                return;
            }

            // Reasoning splitter — Qwen3.5's chat template prefills `<think>\n`
            // into the assistant turn so the model continues mid-thought and
            // closes with `</think>` before the actual answer. We split that
            // stream into `reasoning_content` deltas (until `</think>`) and
            // `content` deltas (after). For non-reasoning templates the
            // splitter starts in the "not in reasoning" state and forwards
            // every token as `content`, identical to the previous wire shape.
            let mut reasoning_splitter =
                ReasoningSplitter::new(prompt_starts_in_reasoning(&prompt));

            let sync_rx = match speculative_mode {
                ResolvedSpeculativeMode::Off => {
                    let prefix_cache_diagnostic = prefix_cache_diagnostic.clone();
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
                            *prefix_cache_diagnostic.lock().unwrap() = "disabled";
                            kiln_model::ModelRunner::spawn_streaming_paged_shared_tokens(
                                runner.clone(),
                                prompt_tokens.clone(),
                                params.clone(),
                                bm.clone(),
                                pc.clone(),
                                decode_batcher.clone(),
                            )
                        } else {
                            let (hit, should_register_on_miss) = {
                                let mut cache = prefix_cache.lock().unwrap();
                                let should_lookup = cache.should_lookup_prompt(&prompt_tokens);
                                let hit = if should_lookup {
                                    cache.lookup(&adapter, &prompt_tokens)?
                                } else {
                                    None
                                };
                                let should_register = cache.should_register_prompt(&prompt_tokens);
                                (hit, should_register)
                            };
                            if hit.is_none() && !should_register_on_miss {
                                *prefix_cache_diagnostic.lock().unwrap() = "skipped";
                                return kiln_model::ModelRunner::spawn_streaming_paged_shared_tokens(
                                    runner.clone(),
                                    prompt_tokens.clone(),
                                    params.clone(),
                                    bm.clone(),
                                    pc.clone(),
                                    decode_batcher.clone(),
                                );
                            }
                            *prefix_cache_diagnostic.lock().unwrap() = if hit.is_some() {
                                "hit"
                            } else {
                                "miss"
                            };
                            let hit_entry_id = hit.as_ref().map(|hit| hit.entry_id);
                            let cached_prefix = hit.map(|hit| PagedPrefixReuse {
                                cached_tokens: hit.cached_tokens,
                                block_ids: hit.block_ids,
                                linear_state: hit.linear_state,
                                next_token: hit.next_token,
                            });

                            let result = kiln_model::ModelRunner::
                                spawn_streaming_paged_shared_tokens_with_prefix_cache(
                                    runner.clone(),
                                    prompt_tokens.clone(),
                                    params.clone(),
                                    bm.clone(),
                                    pc.clone(),
                                    cached_prefix,
                                    decode_batcher.clone(),
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
                            let extra_registrations =
                                std::mem::take(&mut output.extra_registrations);
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
                                    retained_blocks.extend(outcome.retained_blocks);
                                    evicted_blocks.extend(outcome.evicted_blocks);
                                }
                                for registration in extra_registrations {
                                    let outcome = cache.register(adapter.clone(), registration);
                                    retained_blocks.extend(outcome.retained_blocks);
                                    evicted_blocks.extend(outcome.evicted_blocks);
                                }
                            }

                            let mut blocks_to_free: Vec<u32> = allocated_blocks
                                .into_iter()
                                .filter(|block_id| !retained_blocks.contains(block_id))
                                .collect();
                            blocks_to_free.extend(evicted_blocks);
                            // #673: never free a block that the prefix cache
                            // has just claimed for the new entry, and never
                            // queue the same block twice in one free call —
                            // even on the deferred (channel) path.
                            debug_assert!(
                                blocks_to_free
                                    .iter()
                                    .all(|id| !retained_blocks.contains(id)),
                                "blocks_to_free overlaps retained_blocks: free={blocks_to_free:?} retained={retained_blocks:?}",
                            );
                            debug_assert!(
                                {
                                    let mut seen = std::collections::HashSet::with_capacity(
                                        blocks_to_free.len(),
                                    );
                                    blocks_to_free.iter().all(|id| seen.insert(*id))
                                },
                                "blocks_to_free contains duplicate block IDs: {blocks_to_free:?}",
                            );
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
                        Ok(Ok(rx)) => {
                            observe_post_prefill_vram(&memory_budget);
                            rx
                        }
                        _ => {
                            observe_post_prefill_vram(&memory_budget);
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
                    let prefix_cache_diagnostic = prefix_cache_diagnostic.clone();
                    match tokio::task::spawn_blocking(move || {
                        let _gpu_guard = gpu_lock.read().unwrap();
                        let runner_guard = runner.read().unwrap();
                        *prefix_cache_diagnostic.lock().unwrap() = "not_used_speculative";
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
                            record("error".to_string(), &completion_buf, completion_token_count);
                            let _ = tx.send(Event::default().data("[DONE]")).await;
                            return;
                        }
                    }
                }
                ResolvedSpeculativeMode::Mtp => {
                    let prefix_cache_diagnostic = prefix_cache_diagnostic.clone();
                    match tokio::task::spawn_blocking(move || {
                        let _gpu_guard = gpu_lock.read().unwrap();
                        let runner_guard = runner.read().unwrap();
                        *prefix_cache_diagnostic.lock().unwrap() = "not_used_speculative";
                        runner_guard.generate_streaming_mtp_speculative(&prompt, &params)
                    })
                    .await
                    {
                        Ok(Ok(rx)) => rx,
                        _ => {
                            record("error".to_string(), &completion_buf, completion_token_count);
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
            // Initial `false` is dead under the current control flow —
            // every non-timeout branch returns from the closure rather
            // than breaking the loop, so `timed_out` is only ever read
            // *after* the deadline arm assigns `true`. Kept for clarity.
            #[allow(unused_assignments)]
            let mut timed_out = false;
            let deadline = tokio::time::Instant::now() + timeout;

            loop {
                if tx.is_closed() {
                    record(
                        "client_disconnect".to_string(),
                        &completion_buf,
                        completion_token_count,
                    );
                    return;
                }

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
                    _ = tx.closed() => {
                        record(
                            "client_disconnect".to_string(),
                            &completion_buf,
                            completion_token_count,
                        );
                        return;
                    }
                    join_result = recv_handle => {
                        match join_result {
                            Ok((rx_back, Ok(StreamEvent::Token(token)))) => {
                                maybe_rx = Some(rx_back);
                                if let Ok(mut stats) = decode_stats.lock() {
                                    stats.record_token(std::time::Instant::now());
                                }
                                metrics.add_tokens(1);
                                completion_token_count = completion_token_count.saturating_add(1);
                                // Route this token through the reasoning
                                // splitter — at most one of `content` or
                                // `reasoning_content` lands in the delta,
                                // depending on which side of `</think>` the
                                // splitter currently sits on. A token that
                                // straddles the boundary may emit both in
                                // separate chunks (rare; emitted in order).
                                let chunk = reasoning_splitter.push(&token.text);
                                if !emit_or_buffer_reasoning_chunk(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    chunk,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    buffer_tool_content,
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
                            Ok((_, Ok(StreamEvent::Done(done)))) => {
                                let finish = match done.finish_reason {
                                    kiln_model::FinishReason::Eos => "stop",
                                    kiln_model::FinishReason::MaxTokens => "length",
                                    kiln_model::FinishReason::StopSequence(_) => "stop",
                                };
                                // Drain any partial-tag tail buffered in the
                                // splitter before signaling end-of-stream so
                                // we don't silently swallow up-to-7 chars
                                // that turned out not to be `</think>`.
                                let trailing = reasoning_splitter.flush();
                                if !emit_or_buffer_reasoning_chunk(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    trailing,
                                    &mut completion_buf,
                                    &mut reasoning_buf,
                                    &mut content_buf,
                                    buffer_tool_content,
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
                                let assistant_output =
                                    assistant_output_from_split_parts_with_tool_parsing(
                                        buffer_tool_content,
                                        reasoning_content,
                                        content_buf.clone(),
                                        finish,
                                    );
                                if assistant_output.tool_calls.is_some()
                                    && !assistant_output.content.is_empty()
                                    && !emit_content_chunk(
                                        &tx,
                                        &id,
                                        created,
                                        &model,
                                        assistant_output.content.clone(),
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
                                if let Some(tool_calls) = assistant_output.tool_calls.as_deref() {
                                    if !emit_tool_calls_chunk(
                                        &tx,
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
                                    && !assistant_output.content.is_empty()
                                    && !emit_content_chunk(
                                        &tx,
                                        &id,
                                        created,
                                        &model,
                                        assistant_output.content.clone(),
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
                                let _ = tx
                                    .send(
                                        Event::default()
                                            .data(serde_json::to_string(&chunk).unwrap()),
                                    )
                                    .await;
                                let _ = tx.send(Event::default().data("[DONE]")).await;
                                let cache_value = DeterministicCompletionCacheValue {
                                    text: assistant_output.content.clone(),
                                    reasoning_content: assistant_output.reasoning_content.clone(),
                                    tool_calls: assistant_output.tool_calls.clone(),
                                    finish_reason: finish.clone(),
                                    completion_tokens: completion_token_count as usize,
                                };
                                if let Some(key) = completion_cache_key.clone() {
                                    completion_cache
                                        .lock()
                                        .unwrap()
                                        .insert_complete_value(key, cache_value.clone());
                                }
                                let chat_cache_value = DeterministicChatRequestCacheValue {
                                    prompt_tokens: prompt_token_count,
                                    completion: cache_value,
                                };
                                if let Some(owner) = chat_request_cache_owner.take() {
                                    owner.complete(chat_cache_value);
                                } else if let Some(key) = chat_request_cache_key.clone() {
                                    chat_request_cache
                                        .lock()
                                        .unwrap()
                                        .insert(key, chat_cache_value);
                                }
                                record(
                                    finish,
                                    &record_completion,
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
                // Drain any pending partial-tag tail before the timeout
                // chunk so the client doesn't lose those bytes.
                let trailing = reasoning_splitter.flush();
                let _ = emit_or_buffer_reasoning_chunk(
                    &tx,
                    &id,
                    created,
                    &model,
                    trailing,
                    &mut completion_buf,
                    &mut reasoning_buf,
                    &mut content_buf,
                    buffer_tool_content,
                )
                .await;
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
                            reasoning_content: None,
                            tool_calls: None,
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
            if *rid == request_id {
                if let Some(t) = token {
                    first_token_at.get_or_insert_with(std::time::Instant::now);
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
            ..request_record_from_req(req, &id, &model, false)
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
            completion_tokens,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
        metadata,
    };
    let ttft = first_token_at.map(|instant| instant.duration_since(request_start));
    attach_chat_performance_metadata(state, req, &mut response, request_start, ttft, None);
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
    stop: Vec<String>,
    top_p_bits: u32,
    top_k: u32,
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

fn batch_prepared_prompts_disabled() -> bool {
    matches!(
        std::env::var("KILN_DISABLE_BATCH_PREPARED_PROMPTS")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

#[cfg(test)]
fn deterministic_batch_cache_key(
    req: &BatchCompletionRequest,
    total_outputs: usize,
) -> Option<DeterministicBatchCacheKey> {
    deterministic_batch_cache_key_with_vocab_size(req, total_outputs, usize::MAX)
}

#[cfg(test)]
fn deterministic_batch_cache_key_with_vocab_size(
    req: &BatchCompletionRequest,
    total_outputs: usize,
    vocab_size: usize,
) -> Option<DeterministicBatchCacheKey> {
    deterministic_batch_cache_key_with_vocab_size_and_fold(req, total_outputs, vocab_size, false)
}

fn deterministic_batch_cache_key_with_vocab_size_and_fold(
    req: &BatchCompletionRequest,
    total_outputs: usize,
    vocab_size: usize,
    fold_reasoning_into_content: bool,
) -> Option<DeterministicBatchCacheKey> {
    if total_outputs == 0 || req.adapter.is_some() || req.adapters.is_some() {
        return None;
    }

    let temperature = requested_or_default_temperature(req.temperature);
    let top_k = requested_or_default_top_k(req.top_k);
    let max_tokens = batch_request_max_tokens(req);
    if max_tokens != 0
        && !SamplingParams::values_are_effectively_greedy(temperature, top_k)
        && req.seed.is_none()
    {
        return None;
    }
    let (temperature_bits, stop, top_p_bits, top_k, seed) =
        normalized_deterministic_request_sampling_key(
            temperature,
            max_tokens,
            &req.stop.clone().unwrap_or_default(),
            requested_or_default_top_p(req.top_p),
            top_k,
            req.seed,
            vocab_size,
        );
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
        max_tokens,
        stop,
        top_p_bits,
        top_k,
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
    }
}

fn batch_response_from_chat_request_cache_hits(
    state: &AppState,
    req: &BatchCompletionRequest,
    vocab_size: usize,
) -> Result<Option<BatchCompletionResponse>, ApiError> {
    if req.n.unwrap_or(1) != 1 || req.adapter.is_some() || req.adapters.is_some() {
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
        keys.push(key);
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
    req: &BatchCompletionRequest,
    vocab_size: usize,
) -> Result<Option<BatchCompletionResponse>, ApiError> {
    let n_per = req.n.unwrap_or(1);
    if n_per <= 1 || req.adapter.is_some() || req.adapters.is_some() {
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
        keys.push(key);
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
        },
    })
}

fn store_chat_request_cache_from_batch_response(
    state: &AppState,
    req: &BatchCompletionRequest,
    resp: &BatchCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    let n_per = req.n.unwrap_or(1);
    if n_per == 0 || req.adapter.is_some() || req.adapters.is_some() {
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
        cache.insert(key, value);
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
            })
            .collect(),
    })
}

fn store_chat_choices_cache_from_batch_response(
    state: &AppState,
    req: &BatchCompletionRequest,
    resp: &BatchCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    let n_per = req.n.unwrap_or(1);
    if n_per <= 1 || req.adapter.is_some() || req.adapters.is_some() {
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
        cache.insert(key, value);
    }
    Ok(())
}

fn store_chat_caches_from_batch_response(
    state: &AppState,
    req: &BatchCompletionRequest,
    resp: &BatchCompletionResponse,
    vocab_size: usize,
) -> Result<(), ApiError> {
    store_chat_request_cache_from_batch_response(state, req, resp, vocab_size)?;
    store_chat_choices_cache_from_batch_response(state, req, resp, vocab_size)?;
    Ok(())
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

    let batch_cache_key = deterministic_batch_cache_key_with_vocab_size_and_fold(
        &req,
        total_outputs,
        state.model_config.vocab_size,
        state.fold_reasoning_into_content,
    );
    let can_hit_batch_cache_before_adapter_work =
        state.active_adapter_name.read().unwrap().is_none();
    let mut batch_cache_owner = None;
    if can_hit_batch_cache_before_adapter_work && let Some(key) = batch_cache_key.as_ref() {
        let claim = state.batch_cache.lock().unwrap().claim(key);
        match claim {
            DeterministicBatchCacheClaim::Hit(cached) => {
                let resp = batch_response_from_cached_value(state, &req, cached);
                store_chat_caches_from_batch_response(
                    state,
                    &req,
                    &resp,
                    state.model_config.vocab_size,
                )?;
                return Ok(Json(resp).into_response());
            }
            DeterministicBatchCacheClaim::Wait(receiver) => {
                if let Some(cached) = wait_for_deterministic_batch(receiver).await {
                    let resp = batch_response_from_cached_value(state, &req, cached);
                    store_chat_caches_from_batch_response(
                        state,
                        &req,
                        &resp,
                        state.model_config.vocab_size,
                    )?;
                    return Ok(Json(resp).into_response());
                }
            }
            DeterministicBatchCacheClaim::Owner => {
                batch_cache_owner = Some(BatchCacheOwnerGuard::new(
                    state.batch_cache.clone(),
                    key.clone(),
                ));
            }
        }
    }

    let chat_choices_cache_key =
        deterministic_chat_choices_cache_key_from_single_prompt_batch_with_vocab_size_and_fold(
            &req,
            state.model_config.vocab_size,
            state.fold_reasoning_into_content,
        )?;
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
                    &req,
                    &resp,
                    state.model_config.vocab_size,
                )?;
                return Ok(Json(resp).into_response());
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
                        &req,
                        &resp,
                        state.model_config.vocab_size,
                    )?;
                    return Ok(Json(resp).into_response());
                }
            }
            DeterministicChatChoicesCacheProbe::Miss => {}
        }
    }

    if can_hit_batch_cache_before_adapter_work
        && let Some(resp) =
            batch_response_from_chat_choices_cache_hits(state, &req, state.model_config.vocab_size)?
    {
        let cache_value = cache_value_from_batch_response(&resp);
        if let Some(owner) = batch_cache_owner.take() {
            owner.complete(cache_value);
        } else if let Some(key) = batch_cache_key.clone() {
            state.batch_cache.lock().unwrap().insert(key, cache_value);
        }
        store_chat_request_cache_from_batch_response(
            state,
            &req,
            &resp,
            state.model_config.vocab_size,
        )?;
        return Ok(Json(resp).into_response());
    }

    if can_hit_batch_cache_before_adapter_work
        && let Some(resp) =
            batch_response_from_chat_request_cache_hits(state, &req, state.model_config.vocab_size)?
    {
        let cache_value = cache_value_from_batch_response(&resp);
        if let Some(owner) = batch_cache_owner.take() {
            owner.complete(cache_value);
        } else if let Some(key) = batch_cache_key.clone() {
            state.batch_cache.lock().unwrap().insert(key, cache_value);
        }
        return Ok(Json(resp).into_response());
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
        };
        let cache_value = cache_value_from_batch_response(&resp);
        if let Some(owner) = batch_cache_owner.take() {
            owner.complete(cache_value);
        } else if let Some(key) = batch_cache_key.clone() {
            state.batch_cache.lock().unwrap().insert(key, cache_value);
        }
        store_chat_request_cache_from_batch_response(
            state,
            &req,
            &resp,
            state.model_config.vocab_size,
        )?;
        store_chat_choices_cache_from_batch_response(
            state,
            &req,
            &resp,
            state.model_config.vocab_size,
        )?;
        return Ok(Json(resp).into_response());
    }

    // Resolve adapter once for the entire batch. After this returns,
    // state.loaded_adapter_name reflects the loaded adapter and every
    // synthesized per-output ChatCompletionRequest below leaves
    // `adapter`/`adapters` as None — generate_real reads the active adapter
    // from state, not from the request.
    let composed_target: Option<ComposedTarget> = if let Some(list) = req.adapters.as_deref() {
        Some(
            synthesize_composed_adapter(
                &state.adapter_dir,
                list,
                state.composed_cache_max_bytes,
                state.composed_cache_max_entries,
            )
            .await?,
        )
    } else {
        None
    };

    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if let Some(ref target) = composed_target {
            ensure_composed_adapter_swap(state, runner, target).await?;
        } else {
            ensure_batch_adapter(state, runner, &req.adapter, &Uuid::new_v4().to_string()).await?;
        }
    }

    // Spawn one task per distinct rendered prompt, then run duplicates in that
    // group sequentially. Different prompts still run concurrently, while
    // duplicate prompt groups let the first physical generation register exact
    // prefix-cache state before later sampled completions look it up. Greedy
    // (`temp=0`) duplicates go further: the output is deterministic, so one
    // decode can serve each prompt-local `n` group and every identical prompt
    // row in the group.
    let clone_greedy_completions = batch_can_clone_deterministic_completions(&req);
    let clone_greedy_prompt_groups = batch_can_clone_identical_prompt_groups(&req);
    let prompt_count = req.prompts.len();
    let prompt_groups = batch_prompt_groups(&req.prompts);
    let prepare_prompt_groups = !batch_prepared_prompts_disabled() && !clone_greedy_completions;
    let mut handles = Vec::with_capacity(prompt_groups.len());
    for prompt_group in prompt_groups {
        let state_clone = state.clone();
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
                        stream: false,
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
                    };
                    let resp = if let Some((prompt_text, prompt_tokens)) = prepared_prompt.as_ref()
                    {
                        generate_one_prepared_prompt_response(
                            &state_clone,
                            synth_req,
                            prompt_text,
                            prompt_tokens,
                        )
                        .await?
                    } else {
                        generate_one_response(&state_clone, synth_req).await?
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
    };
    let cache_value = cache_value_from_batch_response(&resp);
    if let Some(owner) = batch_cache_owner.take() {
        owner.complete(cache_value);
    } else if let Some(key) = batch_cache_key {
        state.batch_cache.lock().unwrap().insert(key, cache_value);
    }
    store_chat_choices_cache_from_batch_response(
        state,
        &req,
        &resp,
        state.model_config.vocab_size,
    )?;
    Ok(Json(resp).into_response())
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
) -> Result<ChatCompletionResponse, ApiError> {
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
        );
        return chat_response_from_multi_responses(
            state,
            req,
            request_start,
            vec![(0, resp)],
            n_per,
            true,
        );
    }

    let composed_target: Option<ComposedTarget> = if let Some(list) = req.adapters.as_deref() {
        Some(
            synthesize_composed_adapter(
                &state.adapter_dir,
                list,
                state.composed_cache_max_bytes,
                state.composed_cache_max_entries,
            )
            .await?,
        )
    } else {
        None
    };

    if let ModelBackend::Real { runner, .. } = state.backend.as_ref() {
        if let Some(ref target) = composed_target {
            ensure_composed_adapter_swap(state, runner, target).await?;
        } else {
            ensure_adapter(state, runner, &req.adapter, &Uuid::new_v4().to_string()).await?;
        }
    }

    let clone_greedy_choices = request_values_are_effectively_greedy(req.temperature, req.top_k);
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
            stream: false,
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
        };
        let resp = generate_one_response(state, synth_req).await?;
        responses.push((completion_idx, resp));
    }

    chat_response_from_multi_responses(
        state,
        req,
        request_start,
        responses,
        n_per,
        clone_greedy_choices,
    )
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
    attach_chat_performance_metadata(state, req, &mut response, request_start, None, None);
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
    let mut base = preset_or_default(req.sampling_preset.as_deref());
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

/// Map a `sampling_preset` string to its corresponding [`SamplingParams`]
/// starting point. Unknown values silently fall back to the Qwen3.5
/// thinking-general default — same shape a user gets with no preset.
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
) -> Result<ChatCompletionResponse, ApiError> {
    let request_start = std::time::Instant::now();
    apply_eval_mode_chat_defaults(state, &mut req);
    let sampling = sampling_params_for_chat_request(&req);

    let chat_request_cache_key = deterministic_chat_request_cache_key_with_vocab_size_and_fold(
        &req,
        &sampling,
        state.model_config.vocab_size,
        fold_reasoning_into_content_for_request(state, &req),
    )?;
    let can_hit_chat_request_cache_before_prompt_work =
        state.loaded_adapter_name.read().unwrap().is_none();
    let mut chat_request_cache_owner = None;
    if can_hit_chat_request_cache_before_prompt_work
        && let Some(key) = chat_request_cache_key.as_ref()
    {
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
            DeterministicChatRequestCacheClaim::Owner => {
                chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                    state.chat_request_cache.clone(),
                    key.clone(),
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

    generate_one_prepared_response(
        state,
        &req,
        request_start,
        &sampling,
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
    prompt_text: &str,
    prompt_tokens: &[TokenId],
) -> Result<ChatCompletionResponse, ApiError> {
    let request_start = std::time::Instant::now();
    apply_eval_mode_chat_defaults(state, &mut req);
    let sampling = sampling_params_for_chat_request(&req);

    let chat_request_cache_key = deterministic_chat_request_cache_key_with_vocab_size_and_fold(
        &req,
        &sampling,
        state.model_config.vocab_size,
        fold_reasoning_into_content_for_request(state, &req),
    )?;
    let can_hit_chat_request_cache_before_prompt_work =
        state.loaded_adapter_name.read().unwrap().is_none();
    let mut chat_request_cache_owner = None;
    if can_hit_chat_request_cache_before_prompt_work
        && let Some(key) = chat_request_cache_key.as_ref()
    {
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
            DeterministicChatRequestCacheClaim::Owner => {
                chat_request_cache_owner = Some(ChatRequestCacheOwnerGuard::new(
                    state.chat_request_cache.clone(),
                    key.clone(),
                ));
            }
        }
    }

    generate_one_prepared_response(
        state,
        &req,
        request_start,
        &sampling,
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
    chat_request_cache_key: Option<String>,
    mut chat_request_cache_owner: Option<ChatRequestCacheOwnerGuard>,
    prompt_text: &str,
    prompt_tokens: &[TokenId],
) -> Result<ChatCompletionResponse, ApiError> {
    let completion_cache_key = deterministic_completion_cache_key(
        state,
        prompt_tokens,
        sampling,
        fold_reasoning_into_content_for_request(state, req),
    );
    let mut completion_cache_owner = false;
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
            DeterministicCompletionCacheClaim::Owner => {
                completion_cache_owner = true;
            }
        }
    }

    match state.backend.as_ref() {
        ModelBackend::Real {
            runner,
            block_manager,
            paged_cache,
            prefix_cache,
            batching_engine,
            ..
        } => {
            let generation = if let Some(batching_engine) = batching_engine {
                generate_real_batched(
                    state,
                    batching_engine,
                    prompt_text,
                    prompt_tokens,
                    sampling,
                    req,
                    request_start,
                )
                .await
            } else {
                generate_real(
                    state,
                    runner,
                    block_manager,
                    paged_cache,
                    prefix_cache,
                    prompt_text,
                    prompt_tokens,
                    sampling,
                    req,
                    request_start,
                )
                .await
            };
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
                        request_start,
                        prompt_tokens.len(),
                        &err,
                    );
                    if completion_cache_owner && let Some(key) = completion_cache_key.as_ref() {
                        fail_deterministic_completion_owner(state, key);
                    }
                    return Err(err);
                }
            };
            if let Some(key) = completion_cache_key.clone() {
                if completion_cache_owner {
                    complete_deterministic_completion_owner(state, key, &resp);
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
                        request_start,
                        prompt_tokens.len(),
                        &err,
                    );
                    if completion_cache_owner && let Some(key) = completion_cache_key.as_ref() {
                        fail_deterministic_completion_owner(state, key);
                    }
                    return Err(err);
                }
            };
            if let Some(key) = completion_cache_key.clone() {
                if completion_cache_owner {
                    complete_deterministic_completion_owner(state, key, &resp);
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
mod tests {
    use super::*;
    use crate::config::SpecMethod;
    use kiln_core::config::ModelConfig;

    fn parse_request(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("request should deserialize")
    }

    fn make_qwen_template_test_state() -> AppState {
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
        let template =
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
        let tokenizer = crate::api::test_tokenizer().with_chat_template(template.to_string());
        AppState::new_mock(
            config,
            scheduler,
            std::sync::Arc::new(engine),
            tokenizer,
            300,
            "kiln-test".to_string(),
        )
    }

    fn make_sampling(temperature: f32) -> SamplingParams {
        SamplingParams {
            temperature,
            ..Default::default()
        }
    }

    #[test]
    fn completion_usage_counts_terminal_eos_token() {
        assert_eq!(
            completion_usage_tokens(0, &kiln_model::FinishReason::Eos),
            1
        );
        assert_eq!(
            completion_usage_tokens(3, &kiln_model::FinishReason::Eos),
            4
        );
        assert_eq!(
            completion_usage_tokens(3, &kiln_model::FinishReason::MaxTokens),
            3
        );
        assert_eq!(
            completion_usage_tokens(
                3,
                &kiln_model::FinishReason::StopSequence("stop".to_string())
            ),
            3
        );
    }

    #[test]
    fn content_accepts_plain_string() {
        let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}]}"#);
        assert_eq!(req.messages[0].content, "hello");
    }

    #[test]
    fn chat_adapter_missing_uses_server_default() {
        let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}]}"#);
        assert_eq!(req.adapter, ChatAdapterSelection::Default);
        assert_eq!(
            req.adapter
                .target_adapter_name(Some("loaded-a".to_string())),
            Some("loaded-a".to_string())
        );
        assert!(!req.adapter.is_explicit());
    }

    #[tokio::test]
    async fn chat_adapter_missing_regression_does_not_unload_active_adapter_http_path() {
        let state = make_batch_test_state();
        *state.active_adapter_name.write().unwrap() = Some("loaded-a".to_string());
        *state.loaded_adapter_name.write().unwrap() = Some("loaded-a".to_string());
        let state_for_assert = state.clone();

        let (status, body) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"adapter default"}],"max_tokens":0}"#,
        )
        .await;

        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(
            state_for_assert
                .active_adapter_name
                .read()
                .unwrap()
                .as_deref(),
            Some("loaded-a"),
            "regression: omitted chat `adapter` must not unload server default"
        );
        assert_eq!(
            state_for_assert
                .loaded_adapter_name
                .read()
                .unwrap()
                .as_deref(),
            Some("loaded-a"),
            "regression: omitted chat `adapter` must not unload runtime adapter"
        );
    }

    #[test]
    fn chat_adapter_null_selects_base_for_request() {
        let req =
            parse_request(r#"{"messages":[{"role":"user","content":"hello"}],"adapter":null}"#);
        assert_eq!(req.adapter, ChatAdapterSelection::Base);
        assert_eq!(
            req.adapter
                .target_adapter_name(Some("loaded-a".to_string())),
            None
        );
        assert!(req.adapter.is_explicit());
    }

    #[test]
    fn chat_adapter_empty_selects_base_for_request() {
        let req = parse_request(r#"{"messages":[{"role":"user","content":"hello"}],"adapter":""}"#);
        assert_eq!(req.adapter, ChatAdapterSelection::Base);
        assert_eq!(
            req.adapter
                .target_adapter_name(Some("loaded-a".to_string())),
            None
        );
        assert!(req.adapter.is_explicit());
    }

    #[test]
    fn chat_adapter_name_selects_named_adapter_for_request() {
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":"hello"}],"adapter":"my-adapter"}"#,
        );
        assert_eq!(
            req.adapter,
            ChatAdapterSelection::Named("my-adapter".to_string())
        );
        assert_eq!(
            req.adapter
                .target_adapter_name(Some("loaded-a".to_string())),
            Some("my-adapter".to_string())
        );
        assert!(req.adapter.is_explicit());
    }

    #[test]
    fn chat_adapter_rejects_invalid_value_type() {
        let err = serde_json::from_str::<ChatCompletionRequest>(
            r#"{"messages":[{"role":"user","content":"hello"}],"adapter":7}"#,
        )
        .expect_err("numeric adapter should not deserialize");
        assert!(err.to_string().contains("invalid type"));
    }

    #[test]
    fn chat_adapter_rejects_invalid_name_shape() {
        assert!(validate_compose_name("../escape").is_err());
        assert!(validate_compose_name("/tmp/adapter").is_err());
        assert!(validate_compose_name("valid-name").is_ok());
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
    fn stop_accepts_string_or_array() {
        let chat_string =
            parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"stop":"END"}"#);
        assert_eq!(chat_string.stop.as_deref(), Some(&["END".to_string()][..]));

        let chat_array =
            parse_request(r#"{"messages":[{"role":"user","content":"hi"}],"stop":["END","DONE"]}"#);
        assert_eq!(
            chat_array.stop.as_deref(),
            Some(&["END".to_string(), "DONE".to_string()][..])
        );

        let batch_string =
            parse_batch_request(r#"{"prompts":[[{"role":"user","content":"hi"}]],"stop":"END"}"#);
        assert_eq!(batch_string.stop.as_deref(), Some(&["END".to_string()][..]));
    }

    #[test]
    fn max_completion_tokens_alias_resolves_like_max_tokens() {
        let alias = parse_request(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":7}"#,
        );
        assert_eq!(alias.max_tokens, None);
        assert_eq!(alias.max_completion_tokens, Some(7));
        assert_eq!(chat_request_max_tokens(&alias), 7);

        let both = parse_request(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":3,"max_completion_tokens":7}"#,
        );
        assert_eq!(chat_request_max_tokens(&both), 3);

        let batch_alias = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"hi"}]],"max_completion_tokens":7}"#,
        );
        assert_eq!(batch_alias.max_tokens, None);
        assert_eq!(batch_alias.max_completion_tokens, Some(7));
        assert_eq!(batch_request_max_tokens(&batch_alias), 7);
    }

    #[test]
    fn tools_round_trip_preserved_on_request() {
        let json = r#"{
            "messages":[{"role":"user","content":"run ls"}],
            "tools":[
                {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object"}}},
                {"type":"function","function":{"name":"Read","description":"Read a file","parameters":{"type":"object"}}}
            ],
            "tool_choice":"auto"
        }"#;
        let req = parse_request(json);
        let tools = req.tools.expect("tools should deserialize");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["function"]["name"], "Bash");
        assert_eq!(tools[1]["function"]["name"], "Read");
        assert_eq!(
            req.tool_choice.as_ref().and_then(|v| v.as_str()),
            Some("auto")
        );
    }

    /// CI hardening for kiln#659: pins the FULL chain
    /// JSON → `ChatCompletionRequest` → `message_to_chat` → `apply_chat_template_full`
    /// against the production bundled Qwen3.5-4B chat template.
    ///
    /// Existing per-layer tests cover only one half each:
    ///  - `tools_round_trip_preserved_on_request` (above) pins JSON deserialization.
    ///  - `kiln_core::tokenizer::test_qwen35_4b_chat_template_renders_tools_and_tool_calls`
    ///    pins rendering with hand-built `ChatMessage` values.
    ///
    /// Neither exercises the seam where production bugs 1 (missing `tojson` filter)
    /// and 3 (`arguments` left as JSON-encoded string instead of dict) actually lived
    /// in PR #632 and shipped to main. A regression in `message_to_chat` mapping or
    /// in how `req.tools.as_deref()` flows through `apply_chat_template_full` would
    /// not surface in either of those tests, but it would break this one.
    #[test]
    fn tools_bearing_chat_completion_renders_via_qwen35_4b_template() {
        // Wire-shape JSON exactly as `/v1/chat/completions` receives. Five
        // turns (system → user → assistant-with-tool_calls → tool → user)
        // exercise the multi-step-tool branch in the Qwen3.5 template, and
        // both tools have non-trivial `parameters.properties` so `tojson`,
        // `|items`, and `|length` filters all run against real data.
        let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Show me what's in /etc."},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_42", "type": "function", "function": {
                        "name": "Bash",
                        "arguments": "{\"command\": \"ls /etc\"}"
                    }}
                ]},
                {"role": "tool", "name": "Bash", "tool_call_id": "call_42",
                 "content": "hosts\nresolv.conf\nshadow"},
                {"role": "user", "content": "Now read /etc/hosts."}
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run"}
                        },
                        "required": ["command"]
                    }
                }},
                {"type": "function", "function": {
                    "name": "Read",
                    "description": "Read a file from disk",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Filesystem path"}
                        },
                        "required": ["path"]
                    }
                }}
            ],
            "tool_choice": "auto"
        }"#;

        // Step 1: deserialize wire payload (pins serde + content-shape parsing
        // including `content: null` on the assistant tool-calls turn).
        let req = parse_request(json);
        assert_eq!(
            req.messages.len(),
            5,
            "fixture must exercise multi-turn shape"
        );
        assert_eq!(
            req.tools.as_ref().map(|t| t.len()),
            Some(2),
            "tools array must round-trip through deserialization"
        );

        // Step 2: load the production bundled Qwen3.5-4B chat template — the
        // canonical template every kiln user actually hits at runtime. Path is
        // relative to this source file (crates/kiln-server/src/api/...).
        let template =
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
        let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());

        // Step 3: wire EXACTLY as `chat_completions_inner` does (see the
        // `let chat_messages = ... map(message_to_chat) ...` /
        // `apply_chat_template_full(..., req.tools.as_deref(), req.tool_choice.as_ref())`
        // pair near the top of that function). Drift between this test and the
        // production wiring would defeat the point of the smoke test.
        let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();
        let prompt = tok
            .apply_chat_template_full(
                &chat_messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
            )
            .expect(
                "Qwen3.5-4B chat template must render the wire-shape \
                 tools+tool_calls payload without error",
            );

        // Bug 1 (tojson filter on `tools` array): if minijinja lacks the
        // `json` feature, the render fails outright. The `<tools>` block plus
        // both function names appearing prove `tools | tojson` produced
        // valid JSON for both definitions.
        assert!(
            prompt.contains("<tools>"),
            "tools block missing — `tools | tojson` regression? prompt was {prompt:?}"
        );
        assert!(
            prompt.contains("\"Bash\""),
            "Bash tool not serialized into <tools> block: {prompt:?}"
        );
        assert!(
            prompt.contains("\"Read\""),
            "Read tool not serialized into <tools> block: {prompt:?}"
        );

        // Bug 3 (arguments as JSON-encoded string vs dict): the Qwen template
        // iterates `tool_call.arguments | items`. If kiln passes the wire
        // form (`"{\"command\":\"ls /etc\"}"`) through unchanged, minijinja
        // rejects it with "cannot convert value into pairs". The
        // `<parameter=command>` block proves arguments were promoted to a
        // dict before render.
        assert!(
            prompt.contains("<function=Bash>"),
            "prior assistant tool_call did not render in pi-XML form: {prompt:?}"
        );
        assert!(
            prompt.contains("<parameter=command>"),
            "tool_call arguments were not iterated as dict — \
             string-form regression? {prompt:?}"
        );
        assert!(
            prompt.contains("ls /etc"),
            "argument value missing from rendered tool_call: {prompt:?}"
        );

        // `role: "tool"` must wrap as `<tool_response>...</tool_response>` —
        // proves `message_to_chat` propagated `tool_call_id` / `name` so the
        // template's `{%- elif message.role == "tool" %}` branch fires.
        assert!(
            prompt.contains("<tool_response>"),
            "tool response role did not render through message_to_chat wiring: {prompt:?}"
        );
        assert!(
            prompt.contains("hosts"),
            "tool response content did not render inside <tool_response>: {prompt:?}"
        );

        // Follow-up user turn must survive past the template's
        // `multi_step_tool` / `last_query_index` scan; otherwise the template
        // raises "No user query found in messages." and the render errors.
        assert!(
            prompt.contains("Now read /etc/hosts."),
            "follow-up user turn missing — last_query_index regression? {prompt:?}"
        );
    }

    #[test]
    fn qwen35_chat_template_kwargs_disable_thinking() {
        let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "Answer with exactly one word."}
            ],
            "chat_template_kwargs": {"enable_thinking": false},
            "temperature": 0.0,
            "max_tokens": 12
        }"#;
        let req = parse_request(json);
        let template =
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
        let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
        let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();

        let prompt = tok
            .apply_chat_template_full_with_options(
                &chat_messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                chat_template_options_from_kwargs(req.chat_template_kwargs.as_ref()),
            )
            .expect("Qwen3.5 prompt with chat_template_kwargs should render");

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "enable_thinking=false should pre-close the reasoning block: {prompt:?}"
        );
        assert!(
            !prompt_starts_in_reasoning(&prompt),
            "pre-closed reasoning prompt should not put generated text into reasoning_content"
        );
        let (reasoning_content, content) = split_reasoning_response("Blue", &prompt);
        assert_eq!(reasoning_content, None);
        assert_eq!(content, "Blue");
    }

    #[test]
    fn qwen35_tool_requests_keep_template_default_thinking() {
        let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "List files."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 12
        }"#;
        let req = parse_request(json);
        let template =
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
        let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
        let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();
        let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
        let normalized_tool_choice =
            normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
        let normalized_chat_template_kwargs =
            normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

        let prompt = tok
            .apply_chat_template_full_with_options(
                &chat_messages,
                normalized_tools,
                normalized_tool_choice,
                chat_template_options_from_kwargs(normalized_chat_template_kwargs),
            )
            .expect("Qwen3.5 tools prompt should render");

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "tool requests should use Qwen's template default thinking mode: {prompt:?}"
        );
        assert!(
            prompt_starts_in_reasoning(&prompt),
            "default tool requests should still split generated text as reasoning_content"
        );
    }

    #[test]
    fn qwen35_tool_requests_honor_explicit_disable_thinking() {
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"List files."}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "chat_template_kwargs":{"enable_thinking":false},
                "temperature":0.0,
                "max_tokens":12
            }"#,
        );
        let template =
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
        let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
        let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();
        let normalized_tools = normalized_tools_for_cache(req.tools.as_deref());
        let normalized_tool_choice =
            normalized_tool_choice_for_cache(normalized_tools, req.tool_choice.as_ref());
        let normalized_chat_template_kwargs =
            normalized_chat_template_kwargs_for_cache(req.chat_template_kwargs.as_ref());

        let prompt = tok
            .apply_chat_template_full_with_options(
                &chat_messages,
                normalized_tools,
                normalized_tool_choice,
                chat_template_options_from_kwargs(normalized_chat_template_kwargs),
            )
            .expect("Qwen3.5 explicit non-thinking tools prompt should render");

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "explicit enable_thinking=false should pre-close thinking: {prompt:?}"
        );
        assert!(
            !prompt_starts_in_reasoning(&prompt),
            "explicit enable_thinking=false should stream generated text as content"
        );
    }

    #[test]
    fn qwen35_no_think_text_is_not_a_control_flag() {
        let json = r#"{
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "Explain what the literal /no_think tag means."}
            ],
            "temperature": 0.0,
            "max_tokens": 12
        }"#;
        let req = parse_request(json);
        let template =
            include_str!("../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja");
        let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());
        let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();

        let prompt = tok
            .apply_chat_template_full_with_options(
                &chat_messages,
                req.tools.as_deref(),
                req.tool_choice.as_ref(),
                chat_template_options_from_kwargs(req.chat_template_kwargs.as_ref()),
            )
            .expect("Qwen3.5 prompt should render");

        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "plain prompt text must not disable template thinking: {prompt:?}"
        );
        assert!(
            prompt_starts_in_reasoning(&prompt),
            "open reasoning prompt should still split generated text as reasoning_content"
        );
    }

    #[test]
    fn qwen35_server_default_thinking_can_be_disabled_and_overridden() {
        let mut state = make_qwen_template_test_state();
        state.default_thinking_enabled = Some(false);
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"List files."}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"auto",
                "temperature":0.0,
                "max_tokens":12
            }"#,
        );

        let prompt = render_prompt_text(
            &state,
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            req.chat_template_kwargs.as_ref(),
        )
        .expect("Qwen3.5 prompt with server default should render");
        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "server default thinking=false should pre-close thinking: {prompt:?}"
        );
        let metadata = chat_completion_metadata_from_prompt(&state, &req, &prompt);
        assert!(!metadata.thinking_enabled);
        assert_eq!(metadata.thinking_mode, "non_reasoning");
        assert_eq!(metadata.thinking_source, "server_default");

        let override_req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"List files."}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"auto",
                "chat_template_kwargs":{"enable_thinking":true},
                "temperature":0.0,
                "max_tokens":12
            }"#,
        );
        let override_prompt = render_prompt_text(
            &state,
            &override_req.messages,
            override_req.tools.as_deref(),
            override_req.tool_choice.as_ref(),
            override_req.chat_template_kwargs.as_ref(),
        )
        .expect("Qwen3.5 prompt with request override should render");
        assert!(
            override_prompt.ends_with("<|im_start|>assistant\n<think>\n"),
            "request enable_thinking=true should override server default: {override_prompt:?}"
        );
        let override_metadata =
            chat_completion_metadata_from_prompt(&state, &override_req, &override_prompt);
        assert!(override_metadata.thinking_enabled);
        assert_eq!(override_metadata.thinking_mode, "reasoning");
        assert_eq!(override_metadata.thinking_source, "request");
    }

    #[test]
    fn qwen35_thinking_off_short_prompt_regression_splits_normal_content() {
        let mut state = make_qwen_template_test_state();
        state.default_thinking_enabled = Some(false);
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"Answer with exactly two words."}],
                "temperature":0.0,
                "max_tokens":8
            }"#,
        );

        let prompt = render_prompt_text(
            &state,
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            req.chat_template_kwargs.as_ref(),
        )
        .expect("Qwen3.5 non-thinking short prompt should render");
        assert!(
            !prompt_starts_in_reasoning(&prompt),
            "regression: thinking-off prompt must pre-close the reasoning block"
        );
        let (reasoning_content, content) = split_reasoning_response("Quality control", &prompt);
        assert_eq!(
            reasoning_content, None,
            "regression: thinking-off Qwen output should not be hidden in reasoning_content"
        );
        assert_eq!(
            content, "Quality control",
            "regression: thinking-off short prompt should produce normal content"
        );
    }

    #[test]
    fn reasoning_only_output_serializes_reasoning_channel_and_empty_reason_metadata() {
        let state = make_qwen_template_test_state();
        let req = parse_request(
            r#"{"messages":[{"role":"user","content":"think only"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let prompt_text = "<|im_start|>assistant\n<think>\n";
        let assistant_output = assistant_output_from_split_parts(
            &req,
            Some("Need one more step.".to_string()),
            String::new(),
            "length",
        );
        let metadata = chat_completion_metadata_from_prompt_and_output(
            &state,
            &req,
            prompt_text,
            &assistant_output,
        );
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 0,
            model: "kiln-test".to_string(),
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
                finish_reason: assistant_output.finish_reason,
                completion_tokens: 4,
            }],
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            },
            metadata,
        };

        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "");
        assert_eq!(
            json["choices"][0]["message"]["reasoning_content"],
            "Need one more step."
        );
        assert_eq!(json["metadata"]["final_content_empty"], true);
        assert_eq!(
            json["metadata"]["content_empty_reason"],
            "reasoning_without_final_content"
        );
        assert_eq!(json["metadata"]["reasoning_folded_into_content"], false);
    }

    #[test]
    fn reasoning_folding_duplicates_reasoning_into_content_without_hiding_channel() {
        let state = make_qwen_template_test_state();
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"think only"}],
                "fold_reasoning_into_content":true,
                "temperature":0.0,
                "max_tokens":4
            }"#,
        );
        let prompt_text = "<|im_start|>assistant\n<think>\n";
        let assistant_output = assistant_output_from_split_parts(
            &req,
            Some("Need one more step.".to_string()),
            String::new(),
            "length",
        );
        let metadata = chat_completion_metadata_from_prompt_and_output(
            &state,
            &req,
            prompt_text,
            &assistant_output,
        );
        let assistant_output = apply_reasoning_content_policy(
            assistant_output,
            fold_reasoning_into_content_for_request(&state, &req),
        );

        assert_eq!(
            assistant_output.content,
            "<think>\nNeed one more step.</think>"
        );
        assert_eq!(
            assistant_output.reasoning_content.as_deref(),
            Some("Need one more step.")
        );
        assert!(metadata.final_content_empty);
        assert_eq!(
            metadata.content_empty_reason,
            Some("reasoning_without_final_content")
        );
        assert!(metadata.reasoning_folded_into_content);
    }

    #[test]
    fn request_reasoning_fold_override_wins_over_server_default() {
        let mut state = make_qwen_template_test_state();
        state.fold_reasoning_into_content = true;
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"think only"}],
                "fold_reasoning_into_content":false,
                "temperature":0.0,
                "max_tokens":4
            }"#,
        );
        let assistant_output = assistant_output_from_split_parts(
            &req,
            Some("Private scratchpad.".to_string()),
            String::new(),
            "length",
        );
        let metadata = chat_completion_metadata_from_prompt_and_output(
            &state,
            &req,
            "<|im_start|>assistant\n<think>\n",
            &assistant_output,
        );
        let assistant_output = apply_reasoning_content_policy(
            assistant_output,
            fold_reasoning_into_content_for_request(&state, &req),
        );

        assert_eq!(assistant_output.content, "");
        assert_eq!(
            assistant_output.reasoning_content.as_deref(),
            Some("Private scratchpad.")
        );
        assert!(!metadata.reasoning_folded_into_content);
    }

    #[test]
    fn folded_reasoning_is_unfolded_before_cache_storage() {
        let content = folded_reasoning_content("Scratchpad", "Final answer");
        assert_eq!(
            response_content_for_cache(&content, Some("Scratchpad")),
            "Final answer"
        );

        let response = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 0,
            model: "kiln-test".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content,
                    reasoning_content: Some("Scratchpad".to_string()),
                    tool_calls: None,
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".to_string(),
                completion_tokens: 3,
            }],
            usage: Usage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            },
            metadata: ChatCompletionMetadata {
                thinking_enabled: true,
                thinking_mode: "reasoning".to_string(),
                thinking_source: "template_default",
                default_thinking_enabled: None,
                final_content_empty: false,
                content_empty_reason: None,
                reasoning_folded_into_content: true,
                config_hashes: None,
                performance: None,
            },
        };

        let cached = cache_value_from_response(&response).unwrap();
        assert_eq!(cached.text, "Final answer");
        assert_eq!(cached.reasoning_content.as_deref(), Some("Scratchpad"));
    }

    #[test]
    fn message_tool_calls_round_trip_preserved() {
        let json = r#"{
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call_42","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"ls\"}"}}
                ]},
                {"role":"tool","name":"Bash","tool_call_id":"call_42","content":"file.txt"}
            ]
        }"#;
        let req = parse_request(json);
        // Assistant message with content:null lands as empty string + tool_calls populated.
        assert_eq!(req.messages[0].role, "assistant");
        assert_eq!(req.messages[0].content, "");
        let calls = req.messages[0]
            .tool_calls
            .as_ref()
            .expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_42");
        assert_eq!(calls[0]["function"]["name"], "Bash");
        // Tool response with id + name.
        assert_eq!(req.messages[1].role, "tool");
        assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_42"));
        assert_eq!(req.messages[1].name.as_deref(), Some("Bash"));
        assert_eq!(req.messages[1].content, "file.txt");
    }

    #[test]
    fn tools_absent_request_keeps_options_none() {
        let req = parse_request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert!(
            req.tools.is_none(),
            "tools should default to None when absent"
        );
        assert!(
            req.tool_choice.is_none(),
            "tool_choice should default to None"
        );
        assert!(req.messages[0].tool_calls.is_none());
        assert!(req.messages[0].name.is_none());
        assert!(req.messages[0].tool_call_id.is_none());
    }

    #[test]
    fn message_to_chat_propagates_tool_fields() {
        let m = Message {
            role: "tool".to_string(),
            content: "ok".to_string(),
            reasoning_content: None,
            tool_calls: None,
            name: Some("Bash".to_string()),
            tool_call_id: Some("call_1".to_string()),
        };
        let chat = message_to_chat(&m);
        assert_eq!(chat.role, "tool");
        assert_eq!(chat.content, "ok");
        assert_eq!(chat.name.as_deref(), Some("Bash"));
        assert_eq!(chat.tool_call_id.as_deref(), Some("call_1"));
        assert!(chat.tool_calls.is_none());
    }

    #[test]
    fn message_to_chat_omits_empty_tool_calls() {
        let empty = Message {
            role: "assistant".to_string(),
            content: "ok".to_string(),
            reasoning_content: None,
            tool_calls: Some(Vec::new()),
            name: None,
            tool_call_id: None,
        };
        assert!(
            message_to_chat(&empty).tool_calls.is_none(),
            "empty message tool_calls should render like omitted tool_calls"
        );

        let non_empty = Message {
            role: "assistant".to_string(),
            content: "ok".to_string(),
            reasoning_content: None,
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "Lookup", "arguments": "{}"}
            })]),
            name: None,
            tool_call_id: None,
        };
        assert_eq!(
            message_to_chat(&non_empty).tool_calls.unwrap().len(),
            1,
            "non-empty message tool_calls must still reach the template"
        );
    }

    #[test]
    fn qwen_xml_tool_call_output_becomes_openai_tool_calls() {
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"list files"}],
                "tools":[{
                    "type":"function",
                    "function":{
                        "name":"bash",
                        "parameters":{
                            "type":"object",
                            "properties":{"command":{"type":"string"}},
                            "required":["command"]
                        }
                    }
                }],
                "tool_choice":"auto"
            }"#,
        );
        let raw = "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>";
        let output = assistant_output_from_split_parts(&req, None, raw.to_string(), "stop");

        assert_eq!(output.finish_reason, "tool_calls");
        assert_eq!(output.content, "");
        let calls = output.tool_calls.expect("tool_calls should be populated");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(calls[0]["function"]["name"], "bash");
        let args: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["command"], "ls -la");

        let deltas = tool_call_deltas_from_openai_calls(&calls);
        assert_eq!(deltas[0]["index"], 0);
        assert_eq!(deltas[0]["function"]["name"], "bash");
    }

    #[test]
    fn qwen_xml_tool_call_output_respects_tool_choice_none() {
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"say hi"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"none"
            }"#,
        );
        let raw = "<tool_call>\n<function=bash>\n</function>\n</tool_call>";
        let output = assistant_output_from_split_parts(&req, None, raw.to_string(), "stop");

        assert_eq!(output.finish_reason, "stop");
        assert!(output.tool_calls.is_none());
        assert!(output.content.contains("<tool_call>"));
    }

    #[test]
    fn tool_requests_stop_generation_at_qwen_tool_call_close() {
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"list files"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "stop":["omega","alpha","omega"]
            }"#,
        );
        let sampling = sampling_params_for_chat_request(&req);

        assert_eq!(
            sampling.stop,
            vec![
                "alpha".to_string(),
                "omega".to_string(),
                QWEN_TOOL_CALL_CLOSE_TAG.to_string(),
            ],
            "tools-capable requests should stop as soon as a complete Qwen XML tool call is generated"
        );
    }

    #[test]
    fn qwen_tool_stop_is_not_added_when_tools_are_disabled() {
        let tool_choice_none = parse_request(
            r#"{
                "messages":[{"role":"user","content":"say hi"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"none"
            }"#,
        );
        assert!(
            !sampling_params_for_chat_request(&tool_choice_none)
                .stop
                .contains(&QWEN_TOOL_CALL_CLOSE_TAG.to_string()),
            "tool_choice=none should keep model output text-shaped"
        );

        let explicit_shorter_stop = parse_request(
            r#"{
                "messages":[{"role":"user","content":"say hi"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "stop":"tool_call"
            }"#,
        );
        assert_eq!(
            sampling_params_for_chat_request(&explicit_shorter_stop).stop,
            vec!["tool_call".to_string()],
            "user-supplied stops that would fire inside the close tag should remain authoritative"
        );
    }

    #[test]
    fn qwen_xml_tool_call_after_reasoning_is_not_leaked_as_content() {
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"inspect the repo"}],
                "tools":[{
                    "type":"function",
                    "function":{
                        "name":"bash",
                        "parameters":{
                            "type":"object",
                            "properties":{"command":{"type":"string"}},
                            "required":["command"]
                        }
                    }
                }]
            }"#,
        );
        let prompt_text = "<|im_start|>assistant\n<think>\n";
        let raw = "The user wants a directory listing.</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>";
        let output = assistant_output_from_model_output(&req, raw, prompt_text, "stop");

        assert_eq!(output.finish_reason, "tool_calls");
        assert_eq!(output.content, "");
        assert_eq!(
            output.reasoning_content.as_deref(),
            Some("The user wants a directory listing.")
        );
        assert!(output.tool_calls.is_some());
    }

    #[test]
    fn qwen_xml_tool_call_inside_unclosed_reasoning_is_not_cached_as_text() {
        let req = parse_request(
            r#"{
                "messages":[{"role":"user","content":"inspect the repo"}],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]
            }"#,
        );
        let prompt_text = "<|im_start|>assistant\n<think>\n";
        let raw = "Need a command.\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
        let output = assistant_output_from_model_output(&req, raw, prompt_text, "stop");

        assert_eq!(output.finish_reason, "tool_calls");
        assert_eq!(output.content, "");
        assert_eq!(output.reasoning_content.as_deref(), Some("Need a command."));
        assert!(output.tool_calls.is_some());
    }

    #[tokio::test]
    async fn buffered_tool_stream_detects_closed_client_without_sending() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let mut completion_preview_buf = String::new();
        let mut reasoning_buf = String::new();
        let mut content_buf = String::new();

        let ok = emit_or_buffer_reasoning_chunk(
            &tx,
            "chatcmpl-test",
            0,
            "kiln-test",
            ReasoningChunk {
                reasoning: None,
                content: Some("<tool_call>".to_string()),
            },
            &mut completion_preview_buf,
            &mut reasoning_buf,
            &mut content_buf,
            true,
        )
        .await;

        assert!(
            !ok,
            "buffered tool-call content must still notice dropped SSE clients"
        );
        assert!(content_buf.is_empty());
    }

    #[test]
    fn tool_call_responses_round_trip_through_chat_caches() {
        let tool_calls = vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
        })];
        let resp = ChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            object: "chat.completion",
            created: 0,
            model: "kiln-test".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: Some(tool_calls.clone()),
                    name: None,
                    tool_call_id: None,
                },
                finish_reason: "tool_calls".to_string(),
                completion_tokens: 8,
            }],
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 8,
                total_tokens: 12,
            },
            metadata: ChatCompletionMetadata {
                thinking_enabled: false,
                thinking_mode: "non_reasoning".to_string(),
                thinking_source: "template_default",
                default_thinking_enabled: None,
                final_content_empty: true,
                content_empty_reason: Some("tool_call"),
                reasoning_folded_into_content: false,
                config_hashes: None,
                performance: None,
            },
        };

        assert_eq!(
            cache_value_from_response(&resp)
                .unwrap()
                .tool_calls
                .as_ref(),
            Some(&tool_calls)
        );
        assert_eq!(
            chat_request_cache_value_from_response(&resp)
                .unwrap()
                .completion
                .tool_calls
                .as_ref(),
            Some(&tool_calls)
        );
        assert_eq!(
            chat_choices_cache_value_from_response(&resp)
                .unwrap()
                .completions[0]
                .tool_calls
                .as_ref(),
            Some(&tool_calls)
        );
    }

    #[test]
    fn batch_cached_chat_layers_preserve_tool_calls() {
        let state = make_batch_test_state();
        let req = parse_batch_request(
            r#"{
                "prompts":[[{"role":"user","content":"inspect"}]],
                "n":1,
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]
            }"#,
        );
        let tool_calls = vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"}
        })];
        let completion = DeterministicCompletionCacheValue {
            text: String::new(),
            reasoning_content: Some("Need the working directory.".to_string()),
            tool_calls: Some(tool_calls.clone()),
            finish_reason: "tool_calls".to_string(),
            completion_tokens: 7,
        };

        let from_choices = batch_response_from_cached_chat_choices(
            &state,
            &req,
            DeterministicChatChoicesCacheValue {
                prompt_tokens: 11,
                completions: vec![completion.clone()],
            },
        );
        assert_eq!(
            from_choices.completions[0].tool_calls.as_ref(),
            Some(&tool_calls)
        );

        let from_requests = batch_response_from_cached_chat_requests(
            &state,
            &req,
            vec![DeterministicChatRequestCacheValue {
                prompt_tokens: 11,
                completion,
            }],
        );
        assert_eq!(
            from_requests.completions[0].tool_calls.as_ref(),
            Some(&tool_calls)
        );
    }

    #[test]
    fn batch_items_serialize_openai_tool_calls() {
        let raw = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>";
        let output = assistant_output_from_split_parts_with_tool_parsing(
            true,
            None,
            raw.to_string(),
            "stop",
        );
        let item = BatchCompletionItem {
            prompt_index: 0,
            completion_index: 0,
            text: output.content,
            reasoning_content: output.reasoning_content,
            tool_calls: output.tool_calls,
            finish_reason: output.finish_reason,
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            },
        };

        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["text"], "");
        assert_eq!(json["finish_reason"], "tool_calls");
        assert_eq!(json["tool_calls"][0]["type"], "function");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "bash");
        let args: serde_json::Value = serde_json::from_str(
            json["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args["command"], "pwd");
        assert!(
            json.get("reasoning_content").is_none(),
            "batch responses should omit reasoning_content when absent"
        );
    }

    #[test]
    fn batch_items_serialize_reasoning_content_when_present() {
        let item = BatchCompletionItem {
            prompt_index: 0,
            completion_index: 0,
            text: String::new(),
            reasoning_content: Some("Still thinking.".to_string()),
            tool_calls: None,
            finish_reason: "length".to_string(),
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            },
        };

        let json = serde_json::to_value(&item).unwrap();
        assert_eq!(json["text"], "");
        assert_eq!(json["reasoning_content"], "Still thinking.");
    }

    #[test]
    fn batch_cache_value_preserves_tool_calls() {
        let tool_calls = vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
        })];
        let resp = BatchCompletionResponse {
            id: "batchcmpl-test".to_string(),
            object: "batch.completion",
            created: 0,
            model: "kiln-test".to_string(),
            completions: vec![BatchCompletionItem {
                prompt_index: 0,
                completion_index: 0,
                text: String::new(),
                reasoning_content: None,
                tool_calls: Some(tool_calls.clone()),
                finish_reason: "tool_calls".to_string(),
                usage: Usage {
                    prompt_tokens: 4,
                    completion_tokens: 8,
                    total_tokens: 12,
                },
            }],
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 8,
                total_tokens: 12,
            },
        };
        let cached = cache_value_from_batch_response(&resp);
        let req = parse_batch_request(
            r#"{
                "prompts":[[{"role":"user","content":"list files"}]],
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]
            }"#,
        );
        let rehydrated = batch_response_from_cached_value(&make_batch_test_state(), &req, cached);

        assert_eq!(
            rehydrated.completions[0].tool_calls.as_ref(),
            Some(&tool_calls)
        );
        assert_eq!(rehydrated.completions[0].finish_reason, "tool_calls");
    }

    #[test]
    fn batch_tool_call_items_round_trip_through_chat_caches() {
        let tool_calls = vec![serde_json::json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
        })];
        let item = BatchCompletionItem {
            prompt_index: 0,
            completion_index: 0,
            text: String::new(),
            reasoning_content: None,
            tool_calls: Some(tool_calls.clone()),
            finish_reason: "tool_calls".to_string(),
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 8,
                total_tokens: 12,
            },
        };

        assert_eq!(
            chat_request_cache_value_from_batch_item(&item)
                .unwrap()
                .completion
                .tool_calls
                .as_ref(),
            Some(&tool_calls)
        );
        assert_eq!(
            chat_choices_cache_value_from_batch_items(vec![&item], 1)
                .unwrap()
                .completions[0]
                .tool_calls
                .as_ref(),
            Some(&tool_calls)
        );
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

    #[test]
    fn slow_request_log_values_respect_threshold_and_redact_prompt_text() {
        let mut state = make_batch_test_state();
        state.slow_request_warn_threshold = Some(std::time::Duration::from_millis(50));
        let mut record = RequestRecord {
            id: "chatcmpl-test".to_string(),
            timestamp_unix_ms: 0,
            model: "kiln-test".to_string(),
            prompt_preview: "secret prompt".to_string(),
            completion_preview: "ok".to_string(),
            prompt_tokens: 12,
            completion_tokens: 4,
            duration_ms: 49,
            streamed: false,
            finish_reason: "stop".to_string(),
            adapter: Some("adapter-a".to_string()),
            max_tokens: Some(8),
            thinking_mode: Some("reasoning".to_string()),
            prefix_cache: Some("hit".to_string()),
            prompt_full: Some("secret full prompt".to_string()),
            ..RequestRecord::default()
        };

        assert!(slow_request_log_values(&state, &record).is_none());

        record.duration_ms = 50;
        let values = slow_request_log_values(&state, &record).unwrap();
        assert_eq!(values.request_id, "chatcmpl-test");
        assert_eq!(values.adapter, "adapter-a");
        assert_eq!(values.prompt_tokens, 12);
        assert_eq!(values.max_output_tokens, 8);
        assert_eq!(values.generated_tokens, 4);
        assert_eq!(values.elapsed_ms, 50);
        assert_eq!(values.threshold_ms, 50);
        assert_eq!(values.thinking_mode, "reasoning");
        assert_eq!(values.prefix_cache, "hit");
        assert_eq!(values.finish_reason, "stop");

        let debug = format!("{values:?}");
        assert!(!debug.contains("secret prompt"));
        assert!(!debug.contains("secret full prompt"));
    }

    /// Build a minimal request body, invoke the route, and return (status, body).
    async fn batch_post(
        state: AppState,
        body_json: &str,
    ) -> (axum::http::StatusCode, serde_json::Value) {
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

    async fn chat_post(
        state: AppState,
        body_json: &str,
    ) -> (axum::http::StatusCode, serde_json::Value) {
        use axum::body::to_bytes;

        let resp = chat_post_raw(state, body_json).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    async fn chat_post_raw(state: AppState, body_json: &str) -> Response {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let app = routes().with_state(state);
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body_json.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn completion_post(
        state: AppState,
        body_json: &str,
    ) -> (axum::http::StatusCode, serde_json::Value) {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
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

    async fn chat_post_text(state: AppState, body_json: &str) -> (axum::http::StatusCode, String) {
        use axum::body::{Body, to_bytes};
        use axum::http::Request;
        use tower::ServiceExt;

        let app = routes().with_state(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body_json.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn chat_rejects_zero_n() {
        let (status, body) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"hi"}],"n":0}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "chat_invalid_request");
    }

    #[tokio::test]
    async fn chat_rejects_streaming_multi_choice() {
        let (status, body) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"hi"}],"n":2,"stream":true}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "chat_invalid_request");
    }

    #[tokio::test]
    async fn completions_prompt_logprobs_accepts_token_id_prompt() {
        let body = serde_json::json!({
            "model": "Qwen3.5-4B",
            "prompt": [11, 22, 33],
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 3
        })
        .to_string();

        let (status, json) = completion_post(make_batch_test_state(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["usage"]["prompt_tokens"], 3);
        assert_eq!(json["usage"]["completion_tokens"], 0);
        let prompt_logprobs = json["choices"][0]["prompt_logprobs"].as_array().unwrap();
        assert_eq!(prompt_logprobs.len(), 3);
        assert!(prompt_logprobs[0].is_null());
        assert_eq!(prompt_logprobs[1].as_object().unwrap().len(), 3);
        assert_eq!(prompt_logprobs[2].as_object().unwrap().len(), 3);
        assert!(
            prompt_logprobs[1]
                .as_object()
                .unwrap()
                .values()
                .all(|entry| entry["logprob"].is_number() && entry["rank"].is_number())
        );
    }

    #[tokio::test]
    async fn completions_prompt_logprobs_rejects_missing_prompt_logprobs() {
        let body = serde_json::json!({
            "prompt": [11, 22, 33],
            "max_tokens": 1
        })
        .to_string();

        let (status, json) = completion_post(make_batch_test_state(), &body).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "completion_invalid_request");
    }

    #[tokio::test]
    async fn completions_prompt_logprobs_rejects_topk_above_cap() {
        let body = serde_json::json!({
            "prompt": [11, 22, 33],
            "max_tokens": 1,
            "prompt_logprobs": MAX_COMPLETION_PROMPT_LOGPROBS + 1
        })
        .to_string();

        let (status, json) = completion_post(make_batch_test_state(), &body).await;
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "completion_invalid_request");
    }

    #[tokio::test]
    async fn chat_response_metadata_reports_server_thinking_default() {
        let mut state = make_batch_test_state();
        state.default_thinking_enabled = Some(false);
        let body = r#"{"messages":[{"role":"user","content":"metadata"}],"max_tokens":0}"#;

        let (status, json) = chat_post(state, body).await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(json["metadata"]["thinking_enabled"], false);
        assert_eq!(json["metadata"]["thinking_mode"], "non_reasoning");
        assert_eq!(json["metadata"]["thinking_source"], "server_default");
        assert_eq!(json["metadata"]["default_thinking_enabled"], false);
    }

    #[tokio::test]
    async fn chat_performance_metadata_is_omitted_by_default() {
        let (status, json) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"perf default"}],"max_tokens":0}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert!(
            json["metadata"].get("performance").is_none(),
            "performance metadata should be opt-in by default: {json}"
        );
    }

    #[tokio::test]
    async fn chat_config_hash_metadata_can_be_enabled_by_request_flag() {
        let mut state = make_batch_test_state();
        state.config_hashes.kiln_env_config_hash = Some(format!("sha256:{}", "a".repeat(64)));

        let (status, json) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"hash request"}],"max_tokens":0,"include_config_hashes":true}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        let hashes = &json["metadata"]["config_hashes"];
        assert!(
            hashes["model_config_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(
            hashes["tokenizer_config_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(hashes["chat_template_hash"].is_null());
        assert!(
            hashes["kiln_env_config_hash"]
                .as_str()
                .unwrap()
                .starts_with("sha256:")
        );
        assert!(
            json["metadata"].get("performance").is_none(),
            "config hash debug metadata must not imply performance metadata"
        );
    }

    #[tokio::test]
    async fn chat_config_hash_metadata_can_be_enabled_by_server_config() {
        let mut state = make_batch_test_state();
        state.chat_config_hash_metadata = true;

        let (status, json) = chat_post(
            state.clone(),
            r#"{"messages":[{"role":"user","content":"hash server"}],"max_tokens":0}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        assert!(json["metadata"]["config_hashes"].is_object());

        let (status, json) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"hash server disabled"}],"max_tokens":0,"include_config_hashes":false}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        assert!(
            json["metadata"].get("config_hashes").is_none(),
            "request flag should be able to disable the server default"
        );
    }

    #[tokio::test]
    async fn chat_performance_metadata_can_be_enabled_by_request_flag() {
        let (status, json) = chat_post(
            make_batch_test_state(),
            r#"{"messages":[{"role":"user","content":"perf request"}],"max_tokens":0,"include_performance":true}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        let perf = &json["metadata"]["performance"];
        assert_eq!(perf["prompt_tokens"], json["usage"]["prompt_tokens"]);
        assert_eq!(perf["completion_tokens"], 0);
        assert_eq!(perf["ttft_ms"], 0.0);
        assert!(perf["total_latency_ms"].as_f64().unwrap() >= 0.0);
        assert_eq!(perf["decode_tokens_per_sec"], 0.0);
        assert_eq!(perf["adapter_used"], "base");
        assert_eq!(perf["thinking_mode"], json["metadata"]["thinking_mode"]);
        assert_eq!(perf["finish_reason"], "length");
    }

    #[tokio::test]
    async fn chat_performance_metadata_can_be_enabled_by_server_config() {
        let mut state = make_batch_test_state();
        state.chat_performance_metadata = true;

        let (status, json) = chat_post(
            state.clone(),
            r#"{"messages":[{"role":"user","content":"perf server"}],"max_tokens":0}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        assert!(json["metadata"]["performance"].is_object());

        let (status, json) = chat_post(
            state,
            r#"{"messages":[{"role":"user","content":"perf server disabled"}],"max_tokens":0,"include_performance":false}"#,
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK, "{json}");
        assert!(
            json["metadata"].get("performance").is_none(),
            "request flag should be able to disable the server default"
        );
    }

    #[tokio::test]
    async fn eval_mode_chat_completion_sets_headers_defaults_and_resets_caches() {
        use axum::body::to_bytes;

        let mut state = make_batch_test_state();
        state.eval_mode = true;
        *state.active_adapter_name.write().unwrap() = Some("eval-adapter".to_string());
        let state_for_assert = state.clone();
        let body = r#"{"messages":[{"role":"user","content":"eval direct"}],"max_tokens":2}"#;

        for _ in 0..16 {
            let resp = chat_post_raw(state.clone(), body).await;
            assert_eq!(resp.status(), axum::http::StatusCode::OK);
            assert_eq!(resp.headers().get("x-kiln-eval-mode").unwrap(), "true");
            assert_eq!(
                resp.headers().get("x-kiln-active-adapter").unwrap(),
                "eval-adapter"
            );
            assert_eq!(resp.headers().get("x-kiln-loaded-adapter").unwrap(), "base");
            let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["object"], "chat.completion");
        }

        let recent = state_for_assert.recent_requests.lock().unwrap().snapshot();
        assert_eq!(recent.len(), 16);
        let latest = recent.first().unwrap();
        assert_eq!(latest.temperature, Some(0.0));
        assert_eq!(latest.top_p, Some(1.0));
        assert_eq!(latest.max_tokens, Some(2));
        assert_eq!(latest.thinking_mode.as_deref(), Some("mock"));

        assert_eq!(state_for_assert.completion_cache.lock().unwrap().stats(), 0);
        assert_eq!(
            state_for_assert.chat_request_cache.lock().unwrap().stats(),
            0
        );
        assert_eq!(
            state_for_assert.chat_choices_cache.lock().unwrap().stats(),
            0
        );
        assert_eq!(state_for_assert.batch_cache.lock().unwrap().stats(), 0);
        assert_eq!(
            state_for_assert
                .rendered_prompt_cache
                .lock()
                .unwrap()
                .stats()
                .2,
            0
        );
        assert_eq!(
            state_for_assert
                .prompt_token_cache
                .lock()
                .unwrap()
                .stats()
                .2,
            0
        );
    }

    #[test]
    fn eval_mode_defaults_are_deterministic_and_disable_thinking_unless_overridden() {
        let mut state = make_batch_test_state();
        state.eval_mode = true;
        let mut req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();

        apply_eval_mode_chat_defaults(&state, &mut req);
        assert_eq!(req.temperature, Some(0.0));
        assert_eq!(req.top_p, Some(1.0));
        assert_eq!(req.top_k, Some(0));
        assert_eq!(req.seed, Some(0));
        assert_eq!(
            req.chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("enable_thinking")),
            Some(&serde_json::Value::Bool(false))
        );

        let mut explicit: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7,"chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
        apply_eval_mode_chat_defaults(&state, &mut explicit);
        assert_eq!(explicit.temperature, Some(0.7));
        assert_eq!(
            explicit
                .chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("enable_thinking")),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn eval_mode_thinking_default_comes_from_model_profile() {
        let mut state = make_batch_test_state();
        state.eval_mode = true;
        state.model_defaults_profile = crate::config::ModelDefaultsProfile {
            eval_mode_default_thinking_enabled: true,
            ..crate::config::ModelDefaultsProfile::qwen3_5_4b()
        };
        let mut req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();

        apply_eval_mode_chat_defaults(&state, &mut req);

        assert_eq!(
            req.chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.get("enable_thinking")),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn batch_request_parses_minimal_shape() {
        let req = parse_batch_request(r#"{"prompts":[[{"role":"user","content":"hi"}]]}"#);
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
                "adapter":"my-adapter",
                "tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],
                "tool_choice":"auto"
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
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.tool_choice.as_ref().unwrap(), "auto");
    }

    #[tokio::test]
    async fn batch_rejects_empty_prompts() {
        let (status, body) = batch_post(make_batch_test_state(), r#"{"prompts":[]}"#).await;
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

    #[tokio::test]
    async fn batch_rejects_oversized_adapters_list() {
        // 17 entries > MAX_COMPOSE_ADAPTERS (16) — caps audit MEDIUM §6 DoS.
        let adapters: Vec<serde_json::Value> = (0..17)
            .map(|_| serde_json::json!({"name": "a", "scale": 1.0}))
            .collect();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"hi"}]],
            "adapters": adapters,
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
    fn batch_prompt_groups_coalesce_duplicate_plain_prompts() {
        let req = parse_batch_request(
            r#"{
                "prompts": [
                    [{"role":"user","content":"same"}],
                    [{"role":"user","content":"different"}],
                    [{"role":"user","content":"same"}]
                ],
                "temperature": 0.7
            }"#,
        );

        let groups = batch_prompt_groups(&req.prompts);
        let grouped_indices: Vec<Vec<usize>> = groups
            .iter()
            .map(|group| group.prompt_indices.clone())
            .collect();
        assert_eq!(grouped_indices, vec![vec![0, 2], vec![1]]);
        assert_eq!(
            groups[0].messages.len(),
            1,
            "duplicate prompt group should store one synthesized message vector"
        );
    }

    #[tokio::test]
    async fn batch_duplicate_prompt_grouping_preserves_response_order() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"same"}],
                [{"role":"user","content":"different"}],
                [{"role":"user","content":"same"}]
            ],
            "temperature": 0.7,
            "max_tokens": 2,
            "seed": 9
        })
        .to_string();

        let (status, body) = batch_post(state, &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let prompt_indices: Vec<u64> = body["completions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["prompt_index"].as_u64().unwrap())
            .collect();
        assert_eq!(
            prompt_indices,
            vec![0, 1, 2],
            "grouped duplicate prompts must not reorder the public batch response"
        );
    }

    #[test]
    fn batch_deterministic_clone_gate_requires_explicit_greedy_multi_completion() {
        let greedy_multi = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":2,"temperature":0.0}"#,
        );
        assert!(batch_can_clone_deterministic_completions(&greedy_multi));
        assert!(batch_can_clone_identical_prompt_groups(&greedy_multi));

        let greedy_single = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":1,"temperature":0.0}"#,
        );
        assert!(!batch_can_clone_deterministic_completions(&greedy_single));
        assert!(batch_can_clone_identical_prompt_groups(&greedy_single));

        let sampled_multi = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":2,"temperature":0.7}"#,
        );
        assert!(!batch_can_clone_deterministic_completions(&sampled_multi));
        assert!(!batch_can_clone_identical_prompt_groups(&sampled_multi));

        let default_temperature =
            parse_batch_request(r#"{"prompts":[[{"role":"user","content":"hi"}]],"n":2}"#);
        assert!(!batch_can_clone_deterministic_completions(
            &default_temperature
        ));
        assert!(!batch_can_clone_identical_prompt_groups(
            &default_temperature
        ));
    }

    #[test]
    fn deterministic_completion_cache_key_accepts_replayable_sampling_only() {
        let state = make_batch_test_state();
        let prompt_tokens = vec![1, 2, 3];

        let unseeded_sampled = SamplingParams {
            temperature: 0.7,
            max_tokens: 4,
            ..Default::default()
        };
        assert!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &unseeded_sampled, false)
                .is_none(),
            "unseeded sampled decoding must stay uncached because it is intentionally random"
        );

        let seeded_sampled_1 = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 4,
            seed: Some(1),
            ..Default::default()
        };
        let seeded_sampled_2 = SamplingParams {
            seed: Some(2),
            ..seeded_sampled_1.clone()
        };
        let seeded_sampled_different_temperature = SamplingParams {
            temperature: 0.8,
            ..seeded_sampled_1.clone()
        };
        assert!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_1, false)
                .is_some(),
            "seeded sampled decoding is replayable and can use the completion cache"
        );
        assert_ne!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_1, false),
            deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_2, false),
            "sampled decoding must split cache entries by seed"
        );
        assert_ne!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_sampled_1, false),
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_sampled_different_temperature,
                false
            ),
            "sampled decoding must split cache entries by temperature"
        );
        let seeded_full_distribution = SamplingParams {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 4,
            seed: Some(1),
            ..SamplingParams::greedy()
        };
        let seeded_top_p_above_one = SamplingParams {
            top_p: 1.5,
            ..seeded_full_distribution.clone()
        };
        let seeded_top_p_zero = SamplingParams {
            top_p: 0.0,
            ..seeded_full_distribution.clone()
        };
        let seeded_top_p_negative = SamplingParams {
            top_p: -0.5,
            ..seeded_full_distribution.clone()
        };
        let seeded_top_k_disabled = SamplingParams {
            top_k: state.model_config.vocab_size as u32,
            ..seeded_full_distribution.clone()
        };
        assert_eq!(
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_full_distribution,
                false
            ),
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_top_p_above_one,
                false
            ),
            "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded sampling should share completion-cache entries"
        );
        assert_eq!(
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_full_distribution,
                false
            ),
            deterministic_completion_cache_key(&state, &prompt_tokens, &seeded_top_p_zero, false),
            "top_p=0 disables nucleus filtering, so full-distribution seeded sampling should share completion-cache entries"
        );
        assert_eq!(
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_full_distribution,
                false
            ),
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_top_p_negative,
                false
            ),
            "negative top_p disables nucleus filtering, so full-distribution seeded sampling should share completion-cache entries"
        );
        assert_eq!(
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_full_distribution,
                false
            ),
            deterministic_completion_cache_key(
                &state,
                &prompt_tokens,
                &seeded_top_k_disabled,
                false
            ),
            "top_k >= model vocab size is disabled, so full-distribution seeded sampling should share completion-cache entries"
        );

        let greedy_seed_1 = SamplingParams {
            temperature: 0.0,
            top_p: 0.8,
            top_k: 17,
            max_tokens: 4,
            seed: Some(1),
            ..Default::default()
        };
        let greedy_seed_2 = SamplingParams {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 0,
            max_tokens: 4,
            seed: Some(2),
            ..Default::default()
        };

        assert_eq!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_seed_1, false),
            deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_seed_2, false),
            "greedy decoding is seed/filter-independent, so seed/top-p/top-k must not split cache entries"
        );
        let top_k_one = SamplingParams {
            temperature: 0.7,
            top_p: 0.2,
            top_k: 1,
            max_tokens: 4,
            ..Default::default()
        };
        assert_eq!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &greedy_seed_1, false),
            deterministic_completion_cache_key(&state, &prompt_tokens, &top_k_one, false),
            "top_k=1 is effectively greedy, so seed/top-p/temperature must not split completion-cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_ignores_stream_flag() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let non_streaming = parse_request(
            r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let streaming = parse_request(
            r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4,"stream":true}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&non_streaming, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&streaming, &sampling).unwrap(),
            "streaming and non-streaming deterministic chat requests share the same cached payload"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_skips_multi_choice_requests() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let single_choice = parse_request(
            r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4,"n":1}"#,
        );
        let multi_choice = parse_request(
            r#"{"messages":[{"role":"user","content":"same cached chat"}],"temperature":0.0,"max_tokens":4,"n":2}"#,
        );

        assert!(
            deterministic_chat_request_cache_key(&single_choice, &sampling)
                .unwrap()
                .is_some(),
            "n=1 should keep the single-choice request cache path"
        );
        assert!(
            deterministic_chat_request_cache_key(&multi_choice, &sampling)
                .unwrap()
                .is_none(),
            "the single-choice request cache value cannot represent top-level n>1 responses"
        );
    }

    #[test]
    fn deterministic_chat_choices_cache_key_normalizes_replayable_multi_choice_requests() {
        let greedy_a = parse_request(
            r#"{"messages":[{"role":"user","content":"same greedy choices"}],"n":4,"temperature":0.0,"top_p":0.8,"top_k":17,"max_tokens":4,"seed":1}"#,
        );
        let greedy_b = parse_request(
            r#"{"messages":[{"role":"user","content":"same greedy choices"}],"n":4,"temperature":0.0,"top_p":0.95,"top_k":0,"max_tokens":4,"seed":2}"#,
        );
        let greedy_sampling_a = SamplingParams {
            temperature: 0.0,
            top_p: 0.8,
            top_k: 17,
            max_tokens: 4,
            seed: Some(1),
            ..Default::default()
        };
        let greedy_sampling_b = SamplingParams {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 0,
            max_tokens: 4,
            seed: Some(2),
            ..Default::default()
        };

        assert_eq!(
            deterministic_chat_choices_cache_key(&greedy_a, 4, &greedy_sampling_a).unwrap(),
            deterministic_chat_choices_cache_key(&greedy_b, 4, &greedy_sampling_b).unwrap(),
            "greedy chat n>1 cache keys should ignore seed/top-p/top-k"
        );
        assert_ne!(
            deterministic_chat_choices_cache_key(&greedy_a, 4, &greedy_sampling_a).unwrap(),
            deterministic_chat_choices_cache_key(&greedy_a, 3, &greedy_sampling_a).unwrap(),
            "different n values must not share top-level chat choices cache entries"
        );
        let top_k_one = parse_request(
            r#"{"messages":[{"role":"user","content":"same greedy choices"}],"n":4,"temperature":0.7,"top_p":0.2,"top_k":1,"max_tokens":4}"#,
        );
        let top_k_one_sampling = SamplingParams {
            temperature: 0.7,
            top_p: 0.2,
            top_k: 1,
            max_tokens: 4,
            ..Default::default()
        };
        assert_eq!(
            deterministic_chat_choices_cache_key(&greedy_a, 4, &greedy_sampling_a).unwrap(),
            deterministic_chat_choices_cache_key(&top_k_one, 4, &top_k_one_sampling).unwrap(),
            "top_k=1 is effectively greedy, so it should share chat choices cache entries with temperature=0"
        );

        let sampled_a = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices"}],"n":4,"temperature":0.7,"top_p":0.9,"max_tokens":4,"seed":1}"#,
        );
        let sampled_b = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices"}],"n":4,"temperature":0.7,"top_p":0.9,"max_tokens":4,"seed":2}"#,
        );
        let sampled_unseeded = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices"}],"n":4,"temperature":0.7,"top_p":0.9,"max_tokens":4}"#,
        );
        let sampled_sampling_a = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 4,
            seed: Some(1),
            ..Default::default()
        };
        let sampled_sampling_b = SamplingParams {
            seed: Some(2),
            ..sampled_sampling_a.clone()
        };
        let sampled_sampling_unseeded = SamplingParams {
            seed: None,
            ..sampled_sampling_a.clone()
        };

        assert_ne!(
            deterministic_chat_choices_cache_key(&sampled_a, 4, &sampled_sampling_a).unwrap(),
            deterministic_chat_choices_cache_key(&sampled_b, 4, &sampled_sampling_b).unwrap(),
            "seeded sampled chat n>1 cache keys should split by base seed"
        );
        let sampled_full_distribution = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":1.0,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_p_above_one = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":1.5,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_p_zero = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":0.0,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_k_disabled = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled choices full top p"}],"n":4,"temperature":0.7,"top_p":1.0,"top_k":248320,"max_tokens":4,"seed":1}"#,
        );
        let sampled_full_distribution_sampling = SamplingParams {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 4,
            seed: Some(1),
            ..SamplingParams::greedy()
        };
        let sampled_top_p_above_one_sampling = SamplingParams {
            top_p: 1.5,
            ..sampled_full_distribution_sampling.clone()
        };
        let sampled_top_p_zero_sampling = SamplingParams {
            top_p: 0.0,
            ..sampled_full_distribution_sampling.clone()
        };
        let sampled_top_k_disabled_sampling = SamplingParams {
            top_k: ModelConfig::qwen3_5_4b().vocab_size as u32,
            ..sampled_full_distribution_sampling.clone()
        };
        let model_vocab_size = ModelConfig::qwen3_5_4b().vocab_size;
        assert_eq!(
            deterministic_chat_choices_cache_key(
                &sampled_full_distribution,
                4,
                &sampled_full_distribution_sampling
            )
            .unwrap(),
            deterministic_chat_choices_cache_key(
                &sampled_top_p_above_one,
                4,
                &sampled_top_p_above_one_sampling
            )
            .unwrap(),
            "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded chat choices should share cache entries"
        );
        assert_eq!(
            deterministic_chat_choices_cache_key_with_vocab_size(
                &sampled_full_distribution,
                4,
                &sampled_full_distribution_sampling,
                model_vocab_size
            )
            .unwrap(),
            deterministic_chat_choices_cache_key_with_vocab_size(
                &sampled_top_p_zero,
                4,
                &sampled_top_p_zero_sampling,
                model_vocab_size
            )
            .unwrap(),
            "top_p=0 disables nucleus filtering, so full-distribution seeded chat choices should share cache entries"
        );
        assert_eq!(
            deterministic_chat_choices_cache_key_with_vocab_size(
                &sampled_full_distribution,
                4,
                &sampled_full_distribution_sampling,
                model_vocab_size
            )
            .unwrap(),
            deterministic_chat_choices_cache_key_with_vocab_size(
                &sampled_top_k_disabled,
                4,
                &sampled_top_k_disabled_sampling,
                model_vocab_size
            )
            .unwrap(),
            "top_k >= model vocab size is disabled, so full-distribution seeded chat choices should share cache entries"
        );
        assert!(
            deterministic_chat_choices_cache_key(&sampled_unseeded, 4, &sampled_sampling_unseeded)
                .unwrap()
                .is_none(),
            "unseeded sampled chat n>1 requests are intentionally random and must not be cached"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_equivalent_sampling_fields() {
        let greedy_a = parse_request(
            r#"{"messages":[{"role":"user","content":"same greedy chat"}],"temperature":0.0,"top_p":0.8,"top_k":17,"max_tokens":4,"seed":1}"#,
        );
        let greedy_b = parse_request(
            r#"{"messages":[{"role":"user","content":"same greedy chat"}],"temperature":0.0,"top_p":0.95,"top_k":0,"max_tokens":4,"seed":2}"#,
        );
        let greedy_sampling_a = SamplingParams {
            temperature: 0.0,
            top_p: 0.8,
            top_k: 17,
            max_tokens: 4,
            seed: Some(1),
            ..Default::default()
        };
        let greedy_sampling_b = SamplingParams {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 0,
            max_tokens: 4,
            seed: Some(2),
            ..Default::default()
        };

        assert_eq!(
            deterministic_chat_request_cache_key(&greedy_a, &greedy_sampling_a).unwrap(),
            deterministic_chat_request_cache_key(&greedy_b, &greedy_sampling_b).unwrap(),
            "greedy request-cache keys should ignore seed/top-p/top-k"
        );
        let top_k_one = parse_request(
            r#"{"messages":[{"role":"user","content":"same greedy chat"}],"temperature":0.7,"top_p":0.2,"top_k":1,"max_tokens":4}"#,
        );
        let top_k_one_sampling = SamplingParams {
            temperature: 0.7,
            top_p: 0.2,
            top_k: 1,
            max_tokens: 4,
            ..Default::default()
        };

        assert_eq!(
            deterministic_chat_request_cache_key(&greedy_a, &greedy_sampling_a).unwrap(),
            deterministic_chat_request_cache_key(&top_k_one, &top_k_one_sampling).unwrap(),
            "top_k=1 request-cache keys should normalize to greedy"
        );

        let zero_a = parse_request(
            r#"{"messages":[{"role":"user","content":"same zero chat"}],"temperature":0.7,"top_p":0.8,"top_k":17,"max_tokens":0,"stop":["x"],"seed":1}"#,
        );
        let zero_b = parse_request(
            r#"{"messages":[{"role":"user","content":"same zero chat"}],"temperature":0.2,"top_p":0.95,"top_k":0,"max_tokens":0,"stop":["y"],"seed":2}"#,
        );
        let zero_sampling_a = SamplingParams {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 17,
            max_tokens: 0,
            stop: vec!["x".to_string()],
            seed: Some(1),
            ..Default::default()
        };
        let zero_sampling_b = SamplingParams {
            temperature: 0.2,
            top_p: 0.95,
            top_k: 0,
            max_tokens: 0,
            stop: vec!["y".to_string()],
            seed: Some(2),
            ..Default::default()
        };

        assert_eq!(
            deterministic_chat_request_cache_key(&zero_a, &zero_sampling_a).unwrap(),
            deterministic_chat_request_cache_key(&zero_b, &zero_sampling_b).unwrap(),
            "max_tokens=0 request-cache keys should ignore generation-only sampling fields"
        );

        let sampled_a = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled chat"}],"temperature":0.7,"max_tokens":4,"seed":1}"#,
        );
        let sampled_b = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled chat"}],"temperature":0.7,"max_tokens":4,"seed":2}"#,
        );
        let sampled_sampling_a = SamplingParams {
            temperature: 0.7,
            max_tokens: 4,
            seed: Some(1),
            ..Default::default()
        };
        let sampled_sampling_b = SamplingParams {
            temperature: 0.7,
            max_tokens: 4,
            seed: Some(2),
            ..Default::default()
        };

        assert_ne!(
            deterministic_chat_request_cache_key(&sampled_a, &sampled_sampling_a).unwrap(),
            deterministic_chat_request_cache_key(&sampled_b, &sampled_sampling_b).unwrap(),
            "seeded sampled request-cache keys must still split by seed"
        );
        // top_k=0 explicit in the JSON because the kiln default is 20 —
        // this test is specifically validating top_k=disabled vs
        // top_k>=vocab equivalence, so we have to opt out of the model
        // default to get the "disabled" semantics it asserts on.
        let sampled_full_distribution = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":1.0,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_p_above_one = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":1.5,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_p_zero = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":0.0,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_k_disabled = parse_request(
            r#"{"messages":[{"role":"user","content":"sampled chat full top p"}],"temperature":0.7,"top_p":1.0,"top_k":248320,"max_tokens":4,"seed":1}"#,
        );
        let sampled_full_distribution_sampling = SamplingParams {
            temperature: 0.7,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 4,
            seed: Some(1),
            ..SamplingParams::greedy()
        };
        let sampled_top_p_above_one_sampling = SamplingParams {
            top_p: 1.5,
            ..sampled_full_distribution_sampling.clone()
        };
        let sampled_top_p_zero_sampling = SamplingParams {
            top_p: 0.0,
            ..sampled_full_distribution_sampling.clone()
        };
        let sampled_top_k_disabled_sampling = SamplingParams {
            top_k: ModelConfig::qwen3_5_4b().vocab_size as u32,
            ..sampled_full_distribution_sampling.clone()
        };
        let model_vocab_size = ModelConfig::qwen3_5_4b().vocab_size;
        assert_eq!(
            deterministic_chat_request_cache_key(
                &sampled_full_distribution,
                &sampled_full_distribution_sampling
            )
            .unwrap(),
            deterministic_chat_request_cache_key(
                &sampled_top_p_above_one,
                &sampled_top_p_above_one_sampling
            )
            .unwrap(),
            "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded chat requests should share cache entries"
        );
        assert_eq!(
            deterministic_chat_request_cache_key_with_vocab_size(
                &sampled_full_distribution,
                &sampled_full_distribution_sampling,
                model_vocab_size
            )
            .unwrap(),
            deterministic_chat_request_cache_key_with_vocab_size(
                &sampled_top_p_zero,
                &sampled_top_p_zero_sampling,
                model_vocab_size
            )
            .unwrap(),
            "top_p=0 disables nucleus filtering, so full-distribution seeded chat requests should share cache entries"
        );
        assert_eq!(
            deterministic_chat_request_cache_key_with_vocab_size(
                &sampled_full_distribution,
                &sampled_full_distribution_sampling,
                model_vocab_size
            )
            .unwrap(),
            deterministic_chat_request_cache_key_with_vocab_size(
                &sampled_top_k_disabled,
                &sampled_top_k_disabled_sampling,
                model_vocab_size
            )
            .unwrap(),
            "top_k >= model vocab size is disabled, so full-distribution seeded chat requests should share cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_max_completion_tokens_alias() {
        let max_tokens = parse_request(
            r#"{"messages":[{"role":"user","content":"same max token alias"}],"temperature":0.0,"max_tokens":4,"seed":1}"#,
        );
        let alias = parse_request(
            r#"{"messages":[{"role":"user","content":"same max token alias"}],"temperature":0.0,"max_completion_tokens":4,"seed":2}"#,
        );
        let max_tokens_sampling = SamplingParams {
            temperature: max_tokens.temperature.unwrap_or(1.0),
            max_tokens: chat_request_max_tokens(&max_tokens),
            seed: max_tokens.seed,
            ..Default::default()
        };
        let alias_sampling = SamplingParams {
            temperature: alias.temperature.unwrap_or(1.0),
            max_tokens: chat_request_max_tokens(&alias),
            seed: alias.seed,
            ..Default::default()
        };

        assert_eq!(
            deterministic_chat_request_cache_key(&max_tokens, &max_tokens_sampling).unwrap(),
            deterministic_chat_request_cache_key(&alias, &alias_sampling).unwrap(),
            "max_completion_tokens should share deterministic request-cache entries with max_tokens"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_ignores_default_openai_option_fields() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let plain = parse_request(
            r#"{"messages":[{"role":"user","content":"same default options"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let defaults = parse_request(
            r#"{"messages":[{"role":"user","content":"same default options"}],"temperature":0.0,"max_tokens":4,"n":1,"response_format":{"type":"text"},"parallel_tool_calls":true,"user":"client-a","metadata":{"trace_id":"ignored"},"store":false,"service_tier":"auto","logprobs":false,"top_logprobs":0,"frequency_penalty":0.0,"presence_penalty":0.0,"stream_options":{"include_usage":false}}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&defaults, &sampling).unwrap(),
            "default OpenAI option fields should not split deterministic chat request-cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_includes_chat_template_kwargs() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let default_req = parse_request(
            r#"{"messages":[{"role":"user","content":"same template kwargs"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let empty_kwargs = parse_request(
            r#"{"messages":[{"role":"user","content":"same template kwargs"}],"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{}}"#,
        );
        let no_think = parse_request(
            r#"{"messages":[{"role":"user","content":"same template kwargs"}],"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&default_req, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&empty_kwargs, &sampling).unwrap(),
            "empty chat_template_kwargs should normalize to omitted"
        );
        assert_ne!(
            deterministic_chat_request_cache_key(&default_req, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&no_think, &sampling).unwrap(),
            "template kwargs change rendered prompts and must split request-cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_keeps_tool_template_kwargs_explicit() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let omitted_kwargs = parse_request(
            r#"{"messages":[{"role":"user","content":"same tool default kwargs"}],"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#,
        );
        let explicit_no_think = parse_request(
            r#"{"messages":[{"role":"user","content":"same tool default kwargs"}],"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":false}}"#,
        );
        let explicit_think = parse_request(
            r#"{"messages":[{"role":"user","content":"same tool default kwargs"}],"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":true}}"#,
        );

        assert_ne!(
            deterministic_chat_request_cache_key(&omitted_kwargs, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&explicit_no_think, &sampling).unwrap(),
            "tool requests must not reinterpret omitted kwargs as enable_thinking=false"
        );
        assert_ne!(
            deterministic_chat_request_cache_key(&explicit_no_think, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&explicit_think, &sampling).unwrap(),
            "explicit enable_thinking values must keep separate cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_empty_tools() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let omitted_tools = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tools"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let empty_tools = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tools"}],"tools":[],"temperature":0.0,"max_tokens":4}"#,
        );
        let real_tool = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tools"}],"tools":[{"type":"function","function":{"name":"Search","parameters":{"type":"object","properties":{}}}}],"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&omitted_tools, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&empty_tools, &sampling).unwrap(),
            "empty tools should not split request-cache entries from omitted tools"
        );
        assert_ne!(
            deterministic_chat_request_cache_key(&omitted_tools, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&real_tool, &sampling).unwrap(),
            "non-empty tools must still split request-cache entries"
        );
        assert_eq!(
            normalized_tools_option_for_synthetic_request(empty_tools.tools.as_deref()),
            None,
            "synthetic fanout should drop empty tools before cloning"
        );
        assert_eq!(
            normalized_tool_choice_option_for_synthetic_request(
                empty_tools.tools.as_deref(),
                empty_tools.tool_choice.as_ref(),
            ),
            None,
            "synthetic fanout should drop absent tool_choice with empty tools"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_no_tool_auto_choice() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let omitted_choice = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let auto_without_tools = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tool_choice":"auto","temperature":0.0,"max_tokens":4}"#,
        );
        let none_with_empty_tools = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tools":[],"tool_choice":"none","temperature":0.0,"max_tokens":4}"#,
        );
        let required_without_tools = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tool_choice":"required","temperature":0.0,"max_tokens":4}"#,
        );
        let real_tool_auto = parse_request(
            r#"{"messages":[{"role":"user","content":"same no-op tool choice"}],"tools":[{"type":"function","function":{"name":"Search","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto","temperature":0.0,"max_tokens":4}"#,
        );

        let omitted_key = deterministic_chat_request_cache_key(&omitted_choice, &sampling).unwrap();
        assert_eq!(
            omitted_key,
            deterministic_chat_request_cache_key(&auto_without_tools, &sampling).unwrap(),
            "tool_choice=auto without tools should not split request-cache entries"
        );
        assert_eq!(
            omitted_key,
            deterministic_chat_request_cache_key(&none_with_empty_tools, &sampling).unwrap(),
            "tool_choice=none with empty tools should not split request-cache entries"
        );
        assert_ne!(
            omitted_key,
            deterministic_chat_request_cache_key(&required_without_tools, &sampling).unwrap(),
            "tool_choice=required without tools stays distinct because it is not a no-op choice"
        );
        assert_ne!(
            omitted_key,
            deterministic_chat_request_cache_key(&real_tool_auto, &sampling).unwrap(),
            "non-empty tools must still split request-cache entries"
        );
        assert_eq!(
            normalized_tool_choice_option_for_synthetic_request(
                auto_without_tools.tools.as_deref(),
                auto_without_tools.tool_choice.as_ref(),
            ),
            None,
            "synthetic fanout should drop no-tool tool_choice=auto before cloning"
        );
        assert_eq!(
            normalized_tool_choice_option_for_synthetic_request(
                none_with_empty_tools.tools.as_deref(),
                none_with_empty_tools.tool_choice.as_ref(),
            ),
            None,
            "synthetic fanout should drop no-tool tool_choice=none before cloning"
        );
        assert_eq!(
            normalized_tool_choice_option_for_synthetic_request(
                required_without_tools.tools.as_deref(),
                required_without_tools.tool_choice.as_ref(),
            ),
            required_without_tools.tool_choice,
            "synthetic fanout must keep non-no-op required tool_choice"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_ignores_input_reasoning_content() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let without_reasoning = parse_request(
            r#"{"messages":[{"role":"user","content":"same rendered prompt"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let with_reasoning = parse_request(
            r#"{"messages":[{"role":"user","content":"same rendered prompt","reasoning_content":"ignored by renderer"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let with_name = parse_request(
            r#"{"messages":[{"role":"user","content":"same rendered prompt","name":"distinct"}],"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&without_reasoning, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&with_reasoning, &sampling).unwrap(),
            "input reasoning_content is not rendered and should not split request-cache entries"
        );
        assert_ne!(
            deterministic_chat_request_cache_key(&without_reasoning, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&with_name, &sampling).unwrap(),
            "fields propagated to the renderer must still split request-cache entries"
        );
    }

    #[test]
    fn deterministic_cache_keys_include_reasoning_fold_mode() {
        let state = make_qwen_template_test_state();
        let prompt_tokens = vec![1, 2, 3];
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let plain = parse_request(
            r#"{"messages":[{"role":"user","content":"same prompt"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let folded = parse_request(
            r#"{"messages":[{"role":"user","content":"same prompt"}],"fold_reasoning_into_content":true,"temperature":0.0,"max_tokens":4}"#,
        );

        assert_ne!(
            deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&folded, &sampling).unwrap(),
            "request-cache entries must split when content folding changes response shape"
        );
        assert_ne!(
            deterministic_chat_choices_cache_key(&plain, 2, &sampling).unwrap(),
            deterministic_chat_choices_cache_key(&folded, 2, &sampling).unwrap(),
            "choices-cache entries must split when content folding changes response shape"
        );
        assert_ne!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &sampling, false),
            deterministic_completion_cache_key(&state, &prompt_tokens, &sampling, true),
            "completion-cache entries must split when content folding changes response shape"
        );

        let batch = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch prompt"}]],"temperature":0.0,"max_tokens":4}"#,
        );
        assert_ne!(
            deterministic_batch_cache_key_with_vocab_size_and_fold(&batch, 1, usize::MAX, false),
            deterministic_batch_cache_key_with_vocab_size_and_fold(&batch, 1, usize::MAX, true),
            "batch-cache entries must split when server folding changes response shape"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_text_content_parts() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let plain = parse_request(
            r#"{"messages":[{"role":"user","content":"same text parts"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let parts = parse_request(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"same "},{"type":"text","text":"text parts"}]}],"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&parts, &sampling).unwrap(),
            "equivalent OpenAI text content parts should not split request-cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_ignores_non_text_content_parts() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let plain = parse_request(
            r#"{"messages":[{"role":"user","content":"same visible text"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let parts = parse_request(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"same visible "},{"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},{"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},{"type":"text","text":"text"}]}],"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_chat_request_cache_key(&plain, &sampling).unwrap(),
            deterministic_chat_request_cache_key(&parts, &sampling).unwrap(),
            "non-text content parts are ignored by the text-only deserializer and should not split request-cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_empty_message_tool_calls() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let omitted = parse_request(
            r#"{"messages":[{"role":"user","content":"same empty message tool calls"},{"role":"assistant","content":"ok"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let empty_tool_calls = parse_request(
            r#"{"messages":[{"role":"user","content":"same empty message tool calls"},{"role":"assistant","content":"ok","tool_calls":[]},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let non_empty_tool_calls = parse_request(
            r#"{"messages":[{"role":"user","content":"same empty message tool calls"},{"role":"assistant","content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{}"}}]},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );

        let omitted_key = deterministic_chat_request_cache_key(&omitted, &sampling).unwrap();
        assert_eq!(
            omitted_key,
            deterministic_chat_request_cache_key(&empty_tool_calls, &sampling).unwrap(),
            "empty message tool_calls should not split request-cache entries"
        );
        assert_ne!(
            omitted_key,
            deterministic_chat_request_cache_key(&non_empty_tool_calls, &sampling).unwrap(),
            "non-empty message tool_calls must still split request-cache entries"
        );
    }

    #[test]
    fn deterministic_chat_request_cache_key_normalizes_tool_call_argument_json_strings() {
        let sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        };
        let compact = parse_request(
            r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{\"query\":\"cache\",\"limit\":2}"}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let whitespace = parse_request(
            r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{ \"limit\" : 2, \"query\" : \"cache\" }"}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let structured = parse_request(
            r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":{"limit":2,"query":"cache"}}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );
        let non_json = parse_request(
            r#"{"messages":[{"role":"user","content":"normalize tool args"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"not-json"}}]},{"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},{"role":"user","content":"continue"}],"temperature":0.0,"max_tokens":4}"#,
        );

        let compact_key = deterministic_chat_request_cache_key(&compact, &sampling).unwrap();
        assert_eq!(
            compact_key,
            deterministic_chat_request_cache_key(&whitespace, &sampling).unwrap(),
            "JSON-equivalent tool_call argument strings should not split request-cache entries"
        );
        assert_eq!(
            compact_key,
            deterministic_chat_request_cache_key(&structured, &sampling).unwrap(),
            "parsed and structured tool_call arguments render equivalently"
        );
        assert_ne!(
            compact_key,
            deterministic_chat_request_cache_key(&non_json, &sampling).unwrap(),
            "non-JSON argument strings must stay distinct"
        );
    }

    #[test]
    fn deterministic_cache_keys_normalize_stop_sequence_sets() {
        let state = make_batch_test_state();
        let prompt_tokens = vec![1, 2, 3];
        let stop_a = vec![
            "omega".to_string(),
            "alpha".to_string(),
            "omega".to_string(),
        ];
        let stop_b = vec!["alpha".to_string(), "omega".to_string()];
        let completion_a = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            stop: stop_a.clone(),
            ..Default::default()
        };
        let completion_b = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            stop: stop_b.clone(),
            ..Default::default()
        };
        assert_eq!(
            deterministic_completion_cache_key(&state, &prompt_tokens, &completion_a, false),
            deterministic_completion_cache_key(&state, &prompt_tokens, &completion_b, false),
            "stop sequence order and duplicates should not split completion-cache entries"
        );

        let chat_a = parse_request(
            r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":["omega","alpha","omega"]}"#,
        );
        let chat_b = parse_request(
            r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":["alpha","omega"]}"#,
        );
        assert_eq!(
            deterministic_chat_request_cache_key(&chat_a, &completion_a).unwrap(),
            deterministic_chat_request_cache_key(&chat_b, &completion_b).unwrap(),
            "stop sequence order and duplicates should not split chat request-cache entries"
        );
        let chat_string = parse_request(
            r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":"alpha"}"#,
        );
        let chat_single_list = parse_request(
            r#"{"messages":[{"role":"user","content":"same stop set"}],"temperature":0.0,"max_tokens":4,"stop":["alpha"]}"#,
        );
        let single_stop_sampling = SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            stop: vec!["alpha".to_string()],
            ..Default::default()
        };
        assert_eq!(
            deterministic_chat_request_cache_key(&chat_string, &single_stop_sampling).unwrap(),
            deterministic_chat_request_cache_key(&chat_single_list, &single_stop_sampling).unwrap(),
            "single-string stop should share chat request-cache entries with a one-item stop list"
        );

        let batch_a = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":["omega","alpha","omega"]}"#,
        );
        let batch_b = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":["alpha","omega"]}"#,
        );
        assert_eq!(
            deterministic_batch_cache_key(&batch_a, 1),
            deterministic_batch_cache_key(&batch_b, 1),
            "stop sequence order and duplicates should not split batch-cache entries"
        );
        let batch_string = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":"alpha"}"#,
        );
        let batch_single_list = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same stop set"}]],"n":1,"temperature":0.0,"max_tokens":4,"stop":["alpha"]}"#,
        );
        assert_eq!(
            deterministic_batch_cache_key(&batch_string, 1),
            deterministic_batch_cache_key(&batch_single_list, 1),
            "single-string stop should share batch-cache entries with a one-item stop list"
        );

        assert_eq!(
            normalized_stop_for_cache(&["x".to_string(), String::new()]),
            vec![String::new()],
            "an empty stop sequence dominates other stop strings"
        );
        assert_eq!(
            normalized_stop_for_cache(&[
                "omega".to_string(),
                "alpha-extra".to_string(),
                "alpha".to_string(),
                "omega-tail".to_string(),
            ]),
            vec!["alpha".to_string(), "omega".to_string()],
            "a shorter stop sequence dominates longer stops that start with it"
        );
        assert_eq!(
            normalized_stop_for_cache(&[
                "prefix-needle-suffix".to_string(),
                "needle".to_string(),
                "az".to_string(),
                "xaz".to_string(),
            ]),
            vec!["az".to_string(), "needle".to_string()],
            "a shorter stop sequence dominates longer stops that contain it anywhere"
        );
        assert_eq!(
            normalized_stop_for_generation(Some(&[
                "prefix-needle-suffix".to_string(),
                "needle".to_string(),
            ])),
            vec!["needle".to_string()],
            "fresh generation should use the same canonical stop list as replay keys"
        );
        assert_eq!(
            normalized_stop_option_for_synthetic_request(Some(&[
                "prefix-needle-suffix".to_string(),
                "needle".to_string(),
            ])),
            Some(vec!["needle".to_string()]),
            "synthetic fanout requests should clone only canonical stop strings"
        );
        assert_eq!(
            normalized_stop_option_for_synthetic_request(Some(&[] as &[String])),
            None,
            "synthetic requests should preserve no-stop as None"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_normalizes_equivalent_sampling_fields() {
        let greedy_a = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same greedy batch"}]],"n":1,"temperature":0.0,"top_p":0.8,"top_k":17,"max_tokens":4,"seed":1}"#,
        );
        let greedy_b = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same greedy batch"}]],"n":1,"temperature":0.0,"top_p":0.95,"top_k":0,"max_tokens":4,"seed":2}"#,
        );
        assert_eq!(
            deterministic_batch_cache_key(&greedy_a, 1),
            deterministic_batch_cache_key(&greedy_b, 1),
            "greedy batch-cache keys should ignore seed/top-p/top-k"
        );
        let top_k_one = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same greedy batch"}]],"n":1,"temperature":0.7,"top_p":0.2,"top_k":1,"max_tokens":4}"#,
        );
        assert_eq!(
            deterministic_batch_cache_key(&greedy_a, 1),
            deterministic_batch_cache_key(&top_k_one, 1),
            "top_k=1 batch-cache keys should normalize to greedy"
        );

        let zero_a = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same zero batch"}]],"n":1,"temperature":0.7,"top_p":0.8,"top_k":17,"max_tokens":0,"stop":["x"],"seed":1}"#,
        );
        let zero_b = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same zero batch"}]],"n":1,"temperature":0.2,"top_p":0.95,"top_k":0,"max_tokens":0,"stop":["y"],"seed":2}"#,
        );
        assert_eq!(
            deterministic_batch_cache_key(&zero_a, 1),
            deterministic_batch_cache_key(&zero_b, 1),
            "max_tokens=0 batch-cache keys should ignore generation-only sampling fields"
        );

        let sampled_a = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"sampled batch"}]],"n":1,"temperature":0.7,"max_tokens":4,"seed":1}"#,
        );
        let sampled_b = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"sampled batch"}]],"n":1,"temperature":0.7,"max_tokens":4,"seed":2}"#,
        );
        assert_ne!(
            deterministic_batch_cache_key(&sampled_a, 1),
            deterministic_batch_cache_key(&sampled_b, 1),
            "seeded sampled batch-cache keys must still split by seed"
        );
        let sampled_full_distribution = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":1.0,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_p_above_one = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":1.5,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_p_zero = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":0.0,"top_k":0,"max_tokens":4,"seed":1}"#,
        );
        let sampled_top_k_disabled = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"sampled batch full top p"}]],"n":1,"temperature":0.7,"top_p":1.0,"top_k":248320,"max_tokens":4,"seed":1}"#,
        );
        assert_eq!(
            deterministic_batch_cache_key(&sampled_full_distribution, 1),
            deterministic_batch_cache_key(&sampled_top_p_above_one, 1),
            "top_p >= 1.0 disables nucleus filtering, so full-distribution seeded batches should share cache entries"
        );
        assert_eq!(
            deterministic_batch_cache_key(&sampled_full_distribution, 1),
            deterministic_batch_cache_key(&sampled_top_p_zero, 1),
            "top_p=0 disables nucleus filtering, so full-distribution seeded batches should share cache entries"
        );
        let model_vocab_size = ModelConfig::qwen3_5_4b().vocab_size;
        assert_eq!(
            deterministic_batch_cache_key_with_vocab_size(
                &sampled_full_distribution,
                1,
                model_vocab_size
            ),
            deterministic_batch_cache_key_with_vocab_size(
                &sampled_top_k_disabled,
                1,
                model_vocab_size
            ),
            "top_k >= model vocab size is disabled, so full-distribution seeded batches should share cache entries"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_normalizes_max_completion_tokens_alias() {
        let max_tokens = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch max token alias"}]],"n":2,"temperature":0.0,"max_tokens":4,"seed":1}"#,
        );
        let alias = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch max token alias"}]],"n":2,"temperature":0.0,"max_completion_tokens":4,"seed":2}"#,
        );

        assert_eq!(
            deterministic_batch_cache_key(&max_tokens, 2),
            deterministic_batch_cache_key(&alias, 2),
            "max_completion_tokens should share deterministic batch-cache entries with max_tokens"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_ignores_default_openai_option_fields() {
        let plain = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch default options"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        let defaults = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch default options"}]],"n":2,"temperature":0.0,"max_tokens":4,"response_format":{"type":"text"},"parallel_tool_calls":true,"user":"client-a","metadata":{"trace_id":"ignored"},"store":false,"service_tier":"auto","logprobs":false,"top_logprobs":0,"frequency_penalty":0.0,"presence_penalty":0.0,"stream_options":{"include_usage":false}}"#,
        );

        assert_eq!(
            deterministic_batch_cache_key(&plain, 2),
            deterministic_batch_cache_key(&defaults, 2),
            "default OpenAI option fields should not split deterministic batch-cache entries"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_includes_chat_template_kwargs() {
        let default_req = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch template kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        let empty_kwargs = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch template kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{}}"#,
        );
        let no_think = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch template kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}"#,
        );

        assert_eq!(
            deterministic_batch_cache_key(&default_req, 2),
            deterministic_batch_cache_key(&empty_kwargs, 2),
            "empty batch chat_template_kwargs should normalize to omitted"
        );
        assert_ne!(
            deterministic_batch_cache_key(&default_req, 2),
            deterministic_batch_cache_key(&no_think, 2),
            "template kwargs change rendered prompts and must split batch-cache entries"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_keeps_tool_template_kwargs_explicit() {
        let omitted_kwargs = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch tool default kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#,
        );
        let explicit_no_think = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch tool default kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":false}}"#,
        );
        let explicit_think = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch tool default kwargs"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"auto","chat_template_kwargs":{"enable_thinking":true}}"#,
        );

        assert_ne!(
            deterministic_batch_cache_key(&omitted_kwargs, 2),
            deterministic_batch_cache_key(&explicit_no_think, 2),
            "batch tool requests must not reinterpret omitted kwargs as enable_thinking=false"
        );
        assert_ne!(
            deterministic_batch_cache_key(&explicit_no_think, 2),
            deterministic_batch_cache_key(&explicit_think, 2),
            "explicit batch enable_thinking values must keep separate cache entries"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_includes_tools_and_tool_choice() {
        let plain = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch tools"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        let with_tools = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch tools"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}]}"#,
        );
        let with_tools_required = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch tools"}]],"n":2,"temperature":0.0,"max_tokens":4,"tools":[{"type":"function","function":{"name":"bash","parameters":{"type":"object"}}}],"tool_choice":"required"}"#,
        );

        assert_ne!(
            deterministic_batch_cache_key(&plain, 2),
            deterministic_batch_cache_key(&with_tools, 2),
            "top-level tools change the rendered batch prompt and must split cache entries"
        );
        assert_ne!(
            deterministic_batch_cache_key(&with_tools, 2),
            deterministic_batch_cache_key(&with_tools_required, 2),
            "tool_choice is template-visible when tools are present"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_normalizes_text_content_parts() {
        let plain = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch text parts"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        let parts = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":[{"type":"text","text":"same batch "},{"type":"text","text":"text parts"}]}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_batch_cache_key(&plain, 2),
            deterministic_batch_cache_key(&parts, 2),
            "equivalent OpenAI text content parts should not split batch-cache entries"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_ignores_non_text_content_parts() {
        let plain = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch visible text"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        let parts = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":[{"type":"text","text":"same batch visible "},{"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},{"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},{"type":"text","text":"text"}]}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_batch_cache_key(&plain, 2),
            deterministic_batch_cache_key(&parts, 2),
            "non-text content parts are ignored by the text-only deserializer and should not split batch-cache entries"
        );
    }

    #[test]
    fn deterministic_batch_cache_key_ignores_unrendered_message_metadata() {
        let plain = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch metadata"},{"role":"assistant","content":"ok"},{"role":"user","content":"continue"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        let metadata = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch metadata","reasoning_content":"ignored by batch renderer"},{"role":"assistant","content":"ok","tool_calls":[]},{"role":"user","content":"continue"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );

        assert_eq!(
            deterministic_batch_cache_key(&plain, 2),
            deterministic_batch_cache_key(&metadata, 2),
            "reasoning_content and empty tool_calls should not split batch-cache entries"
        );
        let non_empty_tool_calls = parse_batch_request(
            r#"{"prompts":[[{"role":"user","content":"same batch metadata"},{"role":"assistant","content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Lookup","arguments":"{}"}}]},{"role":"user","content":"continue"}]],"n":2,"temperature":0.0,"max_tokens":4}"#,
        );
        assert_ne!(
            deterministic_batch_cache_key(&plain, 2),
            deterministic_batch_cache_key(&non_empty_tool_calls, 2),
            "historical assistant tool calls are rendered and must split batch-cache entries"
        );
    }

    #[tokio::test]
    async fn batch_greedy_n_clones_one_physical_completion() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"hi"}]],
            "n": 3,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 7
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let completions = body["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 3);
        assert_eq!(completions[0]["completion_index"], 0);
        assert_eq!(completions[1]["completion_index"], 1);
        assert_eq!(completions[2]["completion_index"], 2);
        assert_eq!(completions[1]["text"], completions[0]["text"]);
        assert_eq!(completions[2]["text"], completions[0]["text"]);

        let one_completion_tokens = completions[0]["usage"]["completion_tokens"]
            .as_u64()
            .unwrap();
        assert_eq!(
            body["usage"]["completion_tokens"].as_u64().unwrap(),
            one_completion_tokens * 3
        );
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            one_completion_tokens,
            "metrics should count physical model decode work, not cloned logical completions"
        );
        assert_eq!(
            state.recent_requests.lock().unwrap().len(),
            1,
            "only the physical generation should enter the recent-request ring"
        );
    }

    #[tokio::test]
    async fn batch_greedy_duplicate_prompts_clone_one_physical_completion() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"same greedy prompt"}],
                [{"role":"user","content":"same greedy prompt"}]
            ],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 7
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let completions = body["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 4);
        let logical_positions: Vec<(u64, u64)> = completions
            .iter()
            .map(|item| {
                (
                    item["prompt_index"].as_u64().unwrap(),
                    item["completion_index"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(logical_positions, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);

        let first_text = completions[0]["text"].clone();
        assert!(
            completions.iter().all(|item| item["text"] == first_text),
            "all identical greedy prompt rows should clone the same deterministic text"
        );

        let one_completion_tokens = completions[0]["usage"]["completion_tokens"]
            .as_u64()
            .unwrap();
        assert_eq!(
            body["usage"]["completion_tokens"].as_u64().unwrap(),
            one_completion_tokens * 4
        );
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            one_completion_tokens,
            "duplicate greedy prompt rows should perform one physical generation total"
        );
        assert_eq!(
            state.recent_requests.lock().unwrap().len(),
            1,
            "cloned duplicate prompt rows should not create extra synthetic recent requests"
        );

        let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!(render_misses, 1);
        assert_eq!(
            render_hits, 0,
            "cloned duplicate prompt rows should skip render-cache lookups entirely"
        );

        let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(token_misses, 1);
        assert_eq!(
            token_hits, 0,
            "cloned duplicate prompt rows should skip token-cache lookups entirely"
        );
    }

    #[tokio::test]
    async fn batch_top_k_one_clones_single_physical_completion() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top k one greedy batch"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 0.2,
            "top_k": 1,
            "max_tokens": 4
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top k one greedy batch"}]],
            "n": 4,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 1,
            "max_tokens": 4
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let completions = body["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 4);
        let first_text = completions[0]["text"].clone();
        assert!(
            completions.iter().all(|item| item["text"] == first_text),
            "top_k=1 completions should clone the single greedy physical output"
        );

        let physical_completion_tokens = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(physical_completion_tokens > 0);
        assert_eq!(
            body["usage"]["completion_tokens"].as_u64().unwrap(),
            physical_completion_tokens * 4,
            "top_k=1 batch n usage should count logical completions while model work stays single-output"
        );

        let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!((render_hits, render_misses), (0, 1));
        let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!((token_hits, token_misses), (0, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
        assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "unseeded top_k=1 batch repeat should hit the whole-batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_k=1 batch-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_k=1 batch-cache hit should return before prompt-token lookup"
        );
        assert_eq!(body["usage"], repeat["usage"]);
        assert_eq!(body["completions"], repeat["completions"]);
    }

    #[tokio::test]
    async fn batch_top_p_above_one_hits_seeded_full_distribution_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top p full distribution batch cache"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 4,
            "seed": 140
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top p full distribution batch cache"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 1.5,
            "max_tokens": 4,
            "seed": 140
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
        assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_p >= 1.0 seeded batch repeat should hit the whole-batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_p >= 1.0 batch-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_p >= 1.0 batch-cache hit should return before prompt-token lookup"
        );
        assert_eq!(body["usage"], repeat["usage"]);
        assert_eq!(body["completions"], repeat["completions"]);
    }

    #[tokio::test]
    async fn batch_top_p_zero_hits_seeded_full_distribution_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top p zero full distribution batch cache"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 4,
            "seed": 144
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top p zero full distribution batch cache"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 0.0,
            "max_tokens": 4,
            "seed": 144
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
        assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_p=0 seeded batch repeat should hit the whole-batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_p=0 batch-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_p=0 batch-cache hit should return before prompt-token lookup"
        );
        assert_eq!(body["usage"], repeat["usage"]);
        assert_eq!(body["completions"], repeat["completions"]);
    }

    #[tokio::test]
    async fn batch_top_k_oversized_hits_seeded_full_distribution_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let disabled_top_k = state.model_config.vocab_size as u32;
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top k oversized full distribution batch cache"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": disabled_top_k,
            "max_tokens": 4,
            "seed": 148
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"top k oversized full distribution batch cache"}]],
            "n": 4,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 4,
            "seed": 148
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_repeat, repeat) = batch_post(state.clone(), &repeat_body).await;
        assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_k >= vocab seeded batch repeat should hit the whole-batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_k >= vocab batch-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_k >= vocab batch-cache hit should return before prompt-token lookup"
        );
        assert_eq!(body["usage"], repeat["usage"]);
        assert_eq!(body["completions"], repeat["completions"]);
    }

    #[tokio::test]
    async fn chat_multi_choice_greedy_clones_single_physical_completion() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"same greedy chat choices"}],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 7
        })
        .to_string();

        let (status, body) = chat_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let choices = body["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 4);
        let choice_indices: Vec<u64> = choices
            .iter()
            .map(|choice| choice["index"].as_u64().unwrap())
            .collect();
        assert_eq!(choice_indices, vec![0, 1, 2, 3]);

        let first_message = choices[0]["message"].clone();
        assert!(
            choices
                .iter()
                .all(|choice| choice["message"] == first_message),
            "all greedy choices should clone the same deterministic message"
        );

        let physical_completion_tokens = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(physical_completion_tokens > 0);
        assert_eq!(
            body["usage"]["completion_tokens"].as_u64().unwrap(),
            physical_completion_tokens * 4,
            "logical chat n usage should count every returned choice"
        );
        assert_eq!(
            state.recent_requests.lock().unwrap().len(),
            1,
            "cloned chat choices should not create extra synthetic recent requests"
        );

        let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!(render_misses, 1);
        assert_eq!(
            render_hits, 0,
            "cloned chat choices should skip render-cache lookups entirely"
        );

        let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(token_misses, 1);
        assert_eq!(
            token_hits, 0,
            "cloned chat choices should skip token-cache lookups entirely"
        );
    }

    #[tokio::test]
    async fn chat_multi_choice_top_k_one_clones_single_physical_completion() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"top k one greedy chat choices"}],
            "n": 4,
            "temperature": 0.7,
            "top_p": 0.2,
            "top_k": 1,
            "max_tokens": 4
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "messages": [{"role":"user","content":"top k one greedy chat choices"}],
            "n": 4,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 1,
            "max_tokens": 4
        })
        .to_string();

        let (status, body) = chat_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let choices = body["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 4);
        let first_message = choices[0]["message"].clone();
        assert!(
            choices
                .iter()
                .all(|choice| choice["message"] == first_message),
            "top_k=1 chat choices should clone the single greedy physical output"
        );

        let physical_completion_tokens = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(physical_completion_tokens > 0);
        assert_eq!(
            body["usage"]["completion_tokens"].as_u64().unwrap(),
            physical_completion_tokens * 4,
            "top_k=1 chat n usage should count logical choices while model work stays single-output"
        );

        let (render_hits, render_misses, _) = state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!((render_hits, render_misses), (0, 1));
        let (token_hits, token_misses, _) = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!((token_hits, token_misses), (0, 1));
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        let (status_repeat, repeat) = chat_post(state.clone(), &repeat_body).await;
        assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "unseeded top_k=1 chat n repeat should hit the top-level choices cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_k=1 chat choices hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_k=1 chat choices hit should return before prompt-token lookup"
        );
        assert_eq!(body["usage"], repeat["usage"]);
        assert_eq!(body["choices"], repeat["choices"]);
    }

    #[tokio::test]
    async fn chat_multi_choice_repeated_greedy_hits_top_level_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let first_body = serde_json::json!({
            "messages": [{"role":"user","content":"cache chat n choices"}],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let second_body = serde_json::json!({
            "messages": [{"role":"user","content":"cache chat n choices"}],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &first_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &second_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "second greedy chat n request should hit the top-level choices cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top-level chat n cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top-level chat n cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn chat_multi_choice_repeated_seeded_sampled_hits_top_level_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"cache sampled chat n choices"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        assert_eq!(first["choices"].as_array().unwrap().len(), 3);
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(
            render_stats_after_first,
            (2, 1, 1),
            "first sampled chat n request should render once and reuse the rendered prompt for later choices"
        );
        assert_eq!(
            token_stats_after_first,
            (2, 1, 1),
            "first sampled chat n request should tokenize once and reuse tokens for later choices"
        );
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 3);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "second seeded sampled chat n request should hit the top-level choices cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top-level sampled chat n cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top-level sampled chat n cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn concurrent_chat_multi_choice_singleflights_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"chat n choices singleflight"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (first, second) = tokio::join!(
            chat_post(state.clone(), &body),
            chat_post(state.clone(), &body)
        );
        let (status_first, first) = first;
        let (status_second, second) = second;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            (2, 1, 1),
            "concurrent duplicate chat n request should render once and reuse for later choices only"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            (2, 1, 1),
            "concurrent duplicate chat n request should tokenize once and reuse for later choices only"
        );
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            first["usage"]["completion_tokens"].as_u64().unwrap(),
            "concurrent duplicate chat n request should do one top-level set of physical completions"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
        assert_ne!(
            first["id"], second["id"],
            "singleflight replay should still get a fresh chat response id"
        );
    }

    #[tokio::test]
    async fn single_prompt_batch_hits_chat_choices_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let chat_body = serde_json::json!({
            "messages": [{"role":"user","content":"share sampled chat n with batch"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"share sampled chat n with batch"}]],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
        assert_eq!(chat["choices"].as_array().unwrap().len(), 3);
        let generated_after_chat = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_chat > 0);
        let render_stats_after_chat = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chat = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 0);

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_chat,
            "equivalent one-prompt batch should hit the chat choices cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chat,
            "batch-from-chat-choices hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chat,
            "batch-from-chat-choices hit should return before prompt-token lookup"
        );
        assert_eq!(
            state.batch_cache.lock().unwrap().stats(),
            1,
            "chat choices hit should also populate the batch cache"
        );

        assert_eq!(
            batch["usage"]["prompt_tokens"].as_u64().unwrap(),
            chat["usage"]["prompt_tokens"].as_u64().unwrap() * 3
        );
        assert_eq!(
            batch["usage"]["completion_tokens"],
            chat["usage"]["completion_tokens"]
        );
        for (choice, completion) in chat["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(batch["completions"].as_array().unwrap())
        {
            assert_eq!(completion["text"], choice["message"]["content"]);
            assert_eq!(completion["finish_reason"], choice["finish_reason"]);
        }
    }

    #[tokio::test]
    async fn single_prompt_batch_from_choices_cache_rehydrates_request_cache_before_single_chat_work()
     {
        let state = make_batch_test_state();
        let chat_n_body = serde_json::json!({
            "messages": [{"role":"user","content":"choices batch rehydrates request single"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 193
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"choices batch rehydrates request single"}]],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 193
        })
        .to_string();
        let chat_one_body = serde_json::json!({
            "messages": [{"role":"user","content":"choices batch rehydrates request single"}],
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();

        let (status_chat_n, chat_n) = chat_post(state.clone(), &chat_n_body).await;
        assert_eq!(status_chat_n, axum::http::StatusCode::OK, "{chat_n}");
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
        let render_stats_after_chat_n = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chat_n = state.prompt_token_cache.lock().unwrap().stats();

        *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chat_n,
            "batch choices-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chat_n,
            "batch choices-cache hit should return before prompt-token lookup"
        );
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            1,
            "batch from choices cache should rehydrate the normalized single-chat request entry"
        );

        let (status_chat_one, chat_one) = chat_post(state.clone(), &chat_one_body).await;
        assert_eq!(status_chat_one, axum::http::StatusCode::OK, "{chat_one}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chat_n,
            "rehydrated single-chat request hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chat_n,
            "rehydrated single-chat request hit should return before prompt-token lookup"
        );
        assert_eq!(chat_one["usage"], batch["completions"][0]["usage"]);
        assert_eq!(chat_one["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn single_prompt_batch_populates_chat_choices_cache_before_chat_work() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"share sampled batch n with chat"}]],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let chat_body = serde_json::json!({
            "messages": [{"role":"user","content":"share sampled batch n with chat"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(batch["completions"].as_array().unwrap().len(), 3);
        let generated_after_batch = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_batch > 0);
        let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

        let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_batch,
            "equivalent chat n request should hit the chat choices cache populated by batch"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_batch,
            "chat-from-batch-choices hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_batch,
            "chat-from-batch-choices hit should return before prompt-token lookup"
        );

        assert_eq!(
            chat["usage"]["prompt_tokens"].as_u64().unwrap() * 3,
            batch["usage"]["prompt_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            chat["usage"]["completion_tokens"],
            batch["usage"]["completion_tokens"]
        );
        for (choice, completion) in chat["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(batch["completions"].as_array().unwrap())
        {
            assert_eq!(choice["message"]["content"], completion["text"]);
            assert_eq!(choice["finish_reason"], completion["finish_reason"]);
        }
    }

    #[tokio::test]
    async fn batch_repeated_greedy_request_hits_batch_cache_before_completion_cache_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"cache me"}]],
            "n": 1,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            generated_after_first > 0,
            "first request should perform physical generation"
        );
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "second identical greedy request should be served from batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch cache hit should return before rendered-prompt cache lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch cache hit should return before prompt-token cache lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(
            first["completions"][0]["text"],
            second["completions"][0]["text"]
        );
        assert_ne!(
            first["id"], second["id"],
            "cached responses should still get a fresh batch response id"
        );
        assert_eq!(
            state.recent_requests.lock().unwrap().len(),
            1,
            "early batch-cache hits should not synthesize per-output recent requests"
        );
    }

    #[tokio::test]
    async fn batch_text_content_parts_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let plain_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch text content parts should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let parts_body = serde_json::json!({
            "prompts": [[{
                "role":"user",
                "content":[
                    {"type":"text","text":"batch text content parts "},
                    {"type":"text","text":"should be no-op"}
                ]
            }]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &plain_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &parts_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "equivalent text content parts should reuse the batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch text content parts cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch text content parts cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_non_text_content_parts_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let plain_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch non-text content parts should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let parts_body = serde_json::json!({
            "prompts": [[{
                "role":"user",
                "content":[
                    {"type":"text","text":"batch non-text content parts "},
                    {"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},
                    {"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},
                    {"type":"text","text":"should be no-op"}
                ]
            }]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &plain_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &parts_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "non-text content parts should reuse the batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch non-text content parts cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch non-text content parts cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_max_completion_tokens_alias_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let max_tokens_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch max completion tokens alias should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let alias_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch max completion tokens alias should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_completion_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &max_tokens_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &alias_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "max_completion_tokens should reuse the max_tokens batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch max_completion_tokens cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch max_completion_tokens cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_stop_string_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let list_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch stop string should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop"],
            "seed": 123
        })
        .to_string();
        let string_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch stop string should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": "never-match-stop",
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &list_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &string_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "single-string stop should reuse the one-item list batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch stop-string cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch stop-string cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_dominated_stop_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let redundant_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch dominated stop should be no-op"}]],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop", "never-match-stop-suffix"],
            "seed": 123
        })
        .to_string();
        let minimal_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch dominated stop should be no-op"}]],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop"],
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &redundant_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &minimal_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "dominated stop sequences should reuse the minimal stop batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "dominated-stop batch-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "dominated-stop batch-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_substring_dominated_stop_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let redundant_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch substring dominated stop should be no-op"}]],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["prefix-never-match-stop-suffix", "never-match-stop"],
            "seed": 123
        })
        .to_string();
        let minimal_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch substring dominated stop should be no-op"}]],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop"],
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &redundant_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &minimal_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "substring-dominated stop sequences should reuse the minimal stop batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "substring-dominated-stop batch-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "substring-dominated-stop batch-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_default_openai_option_fields_hit_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let plain_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch default OpenAI options should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let defaults_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"batch default OpenAI options should be no-op"}]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999,
            "response_format": {"type":"text"},
            "parallel_tool_calls": true,
            "user": "client-a",
            "metadata": {"trace_id":"ignored"},
            "store": false,
            "service_tier": "auto",
            "logprobs": false,
            "top_logprobs": 0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream_options": {"include_usage": false}
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &plain_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &defaults_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "default OpenAI options should reuse the batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch default-option cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch default-option cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_unrendered_message_metadata_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let plain_body = serde_json::json!({
            "prompts": [[
                {"role":"user","content":"batch ignored metadata should be no-op"},
                {"role":"assistant","content":"ok"},
                {"role":"user","content":"continue"}
            ]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let metadata_body = serde_json::json!({
            "prompts": [[
                {
                    "role":"user",
                    "content":"batch ignored metadata should be no-op",
                    "reasoning_content":"ignored by batch renderer"
                },
                {"role":"assistant","content":"ok","tool_calls":[]},
                {"role":"user","content":"continue"}
            ]],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &plain_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &metadata_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "batch ignored message metadata should reuse the batch cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch ignored metadata cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch ignored metadata cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
    }

    #[tokio::test]
    async fn batch_single_output_hits_chat_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let chat_body = serde_json::json!({
            "messages": [{"role":"user","content":"cross endpoint chat to batch"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"cross endpoint chat to batch"}]],
            "n": 1,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
        let generated_after_chat = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_chat > 0);
        let render_stats_after_chat = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chat = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_chat, (0, 1, 1));
        assert_eq!(token_stats_after_chat, (0, 1, 1));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_chat,
            "single-output batch should reuse the chat request cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chat,
            "batch should hit chat request cache before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chat,
            "batch should hit chat request cache before prompt-token lookup"
        );
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(chat["usage"], batch["usage"]);
        assert_eq!(
            chat["choices"][0]["message"]["content"],
            batch["completions"][0]["text"]
        );
    }

    #[tokio::test]
    async fn multi_prompt_batch_hits_chat_request_cache_before_fanout_work() {
        let state = make_batch_test_state();
        let chat_a_body = serde_json::json!({
            "messages": [{"role":"user","content":"cross endpoint cached prompt a"}],
            "temperature": 0.7,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let chat_b_body = serde_json::json!({
            "messages": [{"role":"user","content":"cross endpoint cached prompt b"}],
            "temperature": 0.7,
            "max_tokens": 4,
            "seed": 124
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"cross endpoint cached prompt a"}],
                [{"role":"user","content":"cross endpoint cached prompt b"}]
            ],
            "n": 1,
            "temperature": 0.7,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_a, chat_a) = chat_post(state.clone(), &chat_a_body).await;
        assert_eq!(status_a, axum::http::StatusCode::OK, "{chat_a}");
        let (status_b, chat_b) = chat_post(state.clone(), &chat_b_body).await;
        assert_eq!(status_b, axum::http::StatusCode::OK, "{chat_b}");
        let generated_after_chats = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_chats > 0);
        let render_stats_after_chats = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chats = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_chats, (0, 2, 2));
        assert_eq!(token_stats_after_chats, (0, 2, 2));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_chats,
            "multi-prompt batch should reuse per-prompt chat request-cache hits"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chats,
            "multi-prompt batch should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chats,
            "multi-prompt batch should return before prompt-token lookup"
        );
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(
            chat_a["choices"][0]["message"]["content"],
            batch["completions"][0]["text"]
        );
        assert_eq!(
            chat_b["choices"][0]["message"]["content"],
            batch["completions"][1]["text"]
        );
        assert_eq!(
            batch["usage"]["prompt_tokens"],
            chat_a["usage"]["prompt_tokens"].as_u64().unwrap()
                + chat_b["usage"]["prompt_tokens"].as_u64().unwrap()
        );
    }

    #[tokio::test]
    async fn multi_prompt_batch_hits_chat_choices_cache_before_fanout_work() {
        let state = make_batch_test_state();
        let chat_a_body = serde_json::json!({
            "messages": [{"role":"user","content":"cross endpoint cached choices prompt a"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 700
        })
        .to_string();
        let chat_b_body = serde_json::json!({
            "messages": [{"role":"user","content":"cross endpoint cached choices prompt b"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 703
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"cross endpoint cached choices prompt a"}],
                [{"role":"user","content":"cross endpoint cached choices prompt b"}]
            ],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 700
        })
        .to_string();

        let (status_a, chat_a) = chat_post(state.clone(), &chat_a_body).await;
        assert_eq!(status_a, axum::http::StatusCode::OK, "{chat_a}");
        let (status_b, chat_b) = chat_post(state.clone(), &chat_b_body).await;
        assert_eq!(status_b, axum::http::StatusCode::OK, "{chat_b}");
        assert_eq!(chat_a["choices"].as_array().unwrap().len(), 3);
        assert_eq!(chat_b["choices"].as_array().unwrap().len(), 3);
        let generated_after_chats = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_chats > 0);
        let render_stats_after_chats = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chats = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_chats, (4, 2, 2));
        assert_eq!(token_stats_after_chats, (4, 2, 2));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 6);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 2);

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_chats,
            "multi-prompt n batch should reuse per-prompt chat choices-cache hits"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chats,
            "multi-prompt n batch should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chats,
            "multi-prompt n batch should return before prompt-token lookup"
        );
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(batch["completions"].as_array().unwrap().len(), 6);
        assert_eq!(
            batch["usage"]["prompt_tokens"].as_u64().unwrap(),
            (chat_a["usage"]["prompt_tokens"].as_u64().unwrap()
                + chat_b["usage"]["prompt_tokens"].as_u64().unwrap())
                * 3
        );
        assert_eq!(
            batch["usage"]["completion_tokens"].as_u64().unwrap(),
            chat_a["usage"]["completion_tokens"].as_u64().unwrap()
                + chat_b["usage"]["completion_tokens"].as_u64().unwrap()
        );
        for (choice, completion) in chat_a["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&batch["completions"].as_array().unwrap()[0..3])
        {
            assert_eq!(completion["prompt_index"], 0);
            assert_eq!(completion["text"], choice["message"]["content"]);
            assert_eq!(completion["finish_reason"], choice["finish_reason"]);
        }
        for (choice, completion) in chat_b["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&batch["completions"].as_array().unwrap()[3..6])
        {
            assert_eq!(completion["prompt_index"], 1);
            assert_eq!(completion["text"], choice["message"]["content"]);
            assert_eq!(completion["finish_reason"], choice["finish_reason"]);
        }
    }

    #[tokio::test]
    async fn multi_prompt_batch_from_choices_cache_rehydrates_request_cache_before_chat_work() {
        let state = make_batch_test_state();
        let chat_a_body = serde_json::json!({
            "messages": [{"role":"user","content":"multi choices rehydrate request prompt a"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 194
        })
        .to_string();
        let chat_b_body = serde_json::json!({
            "messages": [{"role":"user","content":"multi choices rehydrate request prompt b"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 197
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"multi choices rehydrate request prompt a"}],
                [{"role":"user","content":"multi choices rehydrate request prompt b"}]
            ],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 194
        })
        .to_string();
        let chat_b_single_body = serde_json::json!({
            "messages": [{"role":"user","content":"multi choices rehydrate request prompt b"}],
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();

        let (status_a, chat_a) = chat_post(state.clone(), &chat_a_body).await;
        assert_eq!(status_a, axum::http::StatusCode::OK, "{chat_a}");
        let (status_b, chat_b) = chat_post(state.clone(), &chat_b_body).await;
        assert_eq!(status_b, axum::http::StatusCode::OK, "{chat_b}");
        let render_stats_after_chats = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chats = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_chats, (0, 2, 2));
        assert_eq!(token_stats_after_chats, (0, 2, 2));
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 2);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);

        *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chats,
            "multi-prompt batch choices-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chats,
            "multi-prompt batch choices-cache hit should return before prompt-token lookup"
        );
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            2,
            "multi-prompt batch from choices cache should rehydrate one request entry per prompt"
        );

        let (status_chat, chat_b_single) = chat_post(state.clone(), &chat_b_single_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b_single}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chats,
            "rehydrated prompt-b request hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chats,
            "rehydrated prompt-b request hit should return before prompt-token lookup"
        );
        assert_eq!(chat_b_single["usage"], batch["completions"][3]["usage"]);
        assert_eq!(chat_b_single["choices"][0]["finish_reason"], "length");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn multi_prompt_batch_populates_chat_choices_cache_before_chat_work() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"batch populates choices prompt a"}],
                [{"role":"user","content":"batch populates choices prompt b"}]
            ],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 800
        })
        .to_string();
        let chat_b_body = serde_json::json!({
            "messages": [{"role":"user","content":"batch populates choices prompt b"}],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 803
        })
        .to_string();

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(batch["completions"].as_array().unwrap().len(), 6);
        let generated_after_batch = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_batch > 0);
        let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_batch, (0, 2, 2));
        assert_eq!(token_stats_after_batch, (0, 2, 2));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(
            state.chat_choices_cache.lock().unwrap().stats(),
            2,
            "multi-prompt batch should populate one chat choices entry per prompt"
        );

        let (status_chat, chat_b) = chat_post(state.clone(), &chat_b_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_batch,
            "equivalent chat n request should hit choices cache populated by multi-prompt batch"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_batch,
            "batch-populated chat choices hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_batch,
            "batch-populated chat choices hit should return before prompt-token lookup"
        );
        assert_eq!(
            state.chat_choices_cache.lock().unwrap().stats(),
            2,
            "chat hit should not need to create a new choices-cache entry"
        );
        assert_eq!(
            chat_b["usage"]["prompt_tokens"].as_u64().unwrap(),
            batch["completions"][3]["usage"]["prompt_tokens"]
                .as_u64()
                .unwrap()
        );
        assert_eq!(
            chat_b["usage"]["completion_tokens"].as_u64().unwrap(),
            batch["completions"].as_array().unwrap()[3..6]
                .iter()
                .map(|item| item["usage"]["completion_tokens"].as_u64().unwrap())
                .sum::<u64>()
        );
        for (choice, completion) in chat_b["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&batch["completions"].as_array().unwrap()[3..6])
        {
            assert_eq!(choice["message"]["content"], completion["text"]);
            assert_eq!(choice["finish_reason"], completion["finish_reason"]);
        }
    }

    #[tokio::test]
    async fn chat_hits_request_cache_populated_by_single_output_batch() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"cross endpoint batch to chat"}]],
            "n": 1,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let chat_body = serde_json::json!({
            "messages": [{"role":"user","content":"cross endpoint batch to chat"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        let generated_after_batch = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_batch > 0);
        let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_batch, (0, 1, 1));
        assert_eq!(token_stats_after_batch, (0, 1, 1));
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            1,
            "batch generation should populate the equivalent chat request cache"
        );

        let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_batch,
            "chat should reuse the request cache populated by the batch request"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_batch,
            "chat should hit request cache before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_batch,
            "chat should hit request cache before prompt-token lookup"
        );
        assert_eq!(batch["usage"], chat["usage"]);
        assert_eq!(
            batch["completions"][0]["text"],
            chat["choices"][0]["message"]["content"]
        );
    }

    #[tokio::test]
    async fn multi_prompt_zero_batch_populates_chat_request_cache_before_chat_work() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"zero batch feeds chat prompt a"}],
                [{"role":"user","content":"zero batch feeds chat prompt b"}]
            ],
            "n": 1,
            "temperature": 0.7,
            "max_tokens": 0,
            "seed": 900
        })
        .to_string();
        let chat_b_body = serde_json::json!({
            "messages": [{"role":"user","content":"zero batch feeds chat prompt b"}],
            "temperature": 0.7,
            "max_tokens": 0,
            "seed": 901
        })
        .to_string();

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(batch["completions"].as_array().unwrap().len(), 2);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_batch, (0, 2, 2));
        assert_eq!(token_stats_after_batch, (0, 2, 2));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            2,
            "zero-token multi-prompt batch should populate one chat request entry per prompt"
        );

        let (status_chat, chat_b) = chat_post(state.clone(), &chat_b_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "zero-token chat hit should not generate model tokens"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_batch,
            "batch-populated chat request hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_batch,
            "batch-populated chat request hit should return before prompt-token lookup"
        );
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            2,
            "chat hit should not need to create a new request-cache entry"
        );
        assert_eq!(
            chat_b["usage"]["prompt_tokens"],
            batch["completions"][1]["usage"]["prompt_tokens"]
        );
        assert_eq!(chat_b["usage"]["completion_tokens"], 0);
        assert_eq!(
            chat_b["choices"][0]["message"]["content"],
            batch["completions"][1]["text"]
        );
        assert_eq!(
            chat_b["choices"][0]["finish_reason"],
            batch["completions"][1]["finish_reason"]
        );
    }

    #[tokio::test]
    async fn multi_output_zero_batch_populates_chat_request_cache_before_single_chat_work() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"zero batch n feeds single chat prompt a"}],
                [{"role":"user","content":"zero batch n feeds single chat prompt b"}]
            ],
            "n": 3,
            "temperature": 0.7,
            "max_tokens": 0,
            "seed": 950
        })
        .to_string();
        let chat_b_body = serde_json::json!({
            "messages": [{"role":"user","content":"zero batch n feeds single chat prompt b"}],
            "temperature": 0.7,
            "max_tokens": 0,
            "seed": 955
        })
        .to_string();

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(batch["completions"].as_array().unwrap().len(), 6);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_batch, (0, 2, 2));
        assert_eq!(token_stats_after_batch, (0, 2, 2));
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            2,
            "zero-token n>1 batch should populate one request entry per prompt after key normalization"
        );
        assert_eq!(
            state.chat_choices_cache.lock().unwrap().stats(),
            2,
            "zero-token n>1 batch should still populate one choices entry per prompt"
        );

        let (status_chat, chat_b) = chat_post(state.clone(), &chat_b_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_b}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "zero-token chat hit should not generate model tokens"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_batch,
            "batch-populated request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_batch,
            "batch-populated request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);
        assert_eq!(
            chat_b["usage"]["prompt_tokens"],
            batch["completions"][3]["usage"]["prompt_tokens"]
        );
        assert_eq!(
            chat_b["choices"][0]["message"]["content"],
            batch["completions"][3]["text"]
        );
        assert_eq!(
            chat_b["choices"][0]["finish_reason"],
            batch["completions"][3]["finish_reason"]
        );
    }

    #[tokio::test]
    async fn greedy_multi_output_batch_clones_cached_chat_response_before_prompt_work() {
        let state = make_batch_test_state();
        let chat_body = serde_json::json!({
            "messages": [{"role":"user","content":"cached chat fans out to greedy batch"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"cached chat fans out to greedy batch"}]],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
        let generated_after_chat = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_chat > 0);
        let render_stats_after_chat = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_chat = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_chat, (0, 1, 1));
        assert_eq!(token_stats_after_chat, (0, 1, 1));

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_chat,
            "greedy n>1 batch should clone the cached chat response without model work"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_chat,
            "greedy n>1 batch should hit chat request cache before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_chat,
            "greedy n>1 batch should hit chat request cache before prompt-token lookup"
        );

        let completions = batch["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 4);
        for completion in completions {
            assert_eq!(
                completion["text"], chat["choices"][0]["message"]["content"],
                "each logical greedy batch output should clone the cached chat content"
            );
            assert_eq!(completion["usage"], chat["usage"]);
        }
        assert_eq!(
            batch["usage"]["prompt_tokens"],
            4 * chat["usage"]["prompt_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            batch["usage"]["completion_tokens"],
            4 * chat["usage"]["completion_tokens"].as_u64().unwrap()
        );
    }

    #[tokio::test]
    async fn chat_hits_request_cache_populated_by_greedy_multi_output_batch() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [[{"role":"user","content":"cached greedy batch feeds chat"}]],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let chat_body = serde_json::json!({
            "messages": [{"role":"user","content":"cached greedy batch feeds chat"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_batch, batch) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_batch, axum::http::StatusCode::OK, "{batch}");
        let generated_after_batch = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_batch > 0);
        let render_stats_after_batch = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_batch = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_batch, (0, 1, 1));
        assert_eq!(token_stats_after_batch, (0, 1, 1));
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            1,
            "the one physical greedy generation should populate the equivalent chat cache"
        );

        let (status_chat, chat) = chat_post(state.clone(), &chat_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_batch,
            "chat should reuse the multi-output batch's physical cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_batch,
            "chat should hit request cache before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_batch,
            "chat should hit request cache before prompt-token lookup"
        );

        let completions = batch["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 4);
        assert_eq!(
            completions[0]["text"],
            chat["choices"][0]["message"]["content"]
        );
        assert_eq!(completions[0]["usage"], chat["usage"]);
        assert_eq!(
            batch["usage"]["prompt_tokens"],
            4 * chat["usage"]["prompt_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            batch["usage"]["completion_tokens"],
            4 * chat["usage"]["completion_tokens"].as_u64().unwrap()
        );
    }

    #[tokio::test]
    async fn repeated_multi_output_zero_batch_hits_batch_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"batch cache zero one"}],
                [{"role":"user","content":"batch cache zero two"}]
            ],
            "n": 2,
            "temperature": 0.7,
            "max_tokens": 0
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 2, 2));
        assert_eq!(token_stats_after_first, (0, 2, 2));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = batch_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch cache hit should return before rendered-prompt cache lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch cache hit should return before prompt-token cache lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
        assert_ne!(
            first["id"], second["id"],
            "cached batch responses should still get a fresh batch id"
        );
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn batch_cache_hit_rehydrates_chat_request_cache_before_chat_work() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"batch hit rehydrates request one"}],
                [{"role":"user","content":"batch hit rehydrates request two"}]
            ],
            "n": 1,
            "temperature": 0.7,
            "max_tokens": 0,
            "seed": 187
        })
        .to_string();
        let chat_two_body = serde_json::json!({
            "messages": [{"role":"user","content":"batch hit rehydrates request two"}],
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 2, 2));
        assert_eq!(token_stats_after_first, (0, 2, 2));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 2);

        *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

        let (status_second, second) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch cache rehydration should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch cache rehydration should return before prompt-token lookup"
        );
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            2,
            "batch cache hit should rehydrate one request-cache entry per prompt"
        );

        let (status_chat, chat_two) = chat_post(state.clone(), &chat_two_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_two}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "rehydrated request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "rehydrated request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(chat_two["usage"], second["completions"][1]["usage"]);
        assert_eq!(chat_two["choices"][0]["finish_reason"], "length");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn batch_cache_hit_rehydrates_chat_choices_cache_before_chat_n_work() {
        let state = make_batch_test_state();
        let batch_body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"batch hit rehydrates choices one"}],
                [{"role":"user","content":"batch hit rehydrates choices two"}]
            ],
            "n": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 188
        })
        .to_string();
        let chat_two_body = serde_json::json!({
            "messages": [{"role":"user","content":"batch hit rehydrates choices two"}],
            "n": 3,
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        assert_eq!(first["completions"].as_array().unwrap().len(), 6);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 2, 2));
        assert_eq!(token_stats_after_first, (0, 2, 2));
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 2);

        *state.chat_choices_cache.lock().unwrap() = DeterministicChatChoicesCache::new(64);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 0);

        let (status_second, second) = batch_post(state.clone(), &batch_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "batch cache choices rehydration should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "batch cache choices rehydration should return before prompt-token lookup"
        );
        assert_eq!(
            state.chat_choices_cache.lock().unwrap().stats(),
            2,
            "batch cache hit should rehydrate one choices-cache entry per prompt"
        );

        let (status_chat, chat_two) = chat_post(state.clone(), &chat_two_body).await;
        assert_eq!(status_chat, axum::http::StatusCode::OK, "{chat_two}");
        assert_eq!(chat_two["choices"].as_array().unwrap().len(), 3);
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "rehydrated choices-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "rehydrated choices-cache hit should return before prompt-token lookup"
        );
        assert_eq!(
            chat_two["usage"]["prompt_tokens"],
            second["completions"][3]["usage"]["prompt_tokens"]
        );
        assert_eq!(
            chat_two["usage"]["completion_tokens"].as_u64().unwrap(),
            second["completions"].as_array().unwrap()[3..6]
                .iter()
                .map(|item| item["usage"]["completion_tokens"].as_u64().unwrap())
                .sum::<u64>()
        );
    }

    #[tokio::test]
    async fn concurrent_multi_output_greedy_batch_singleflights_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"batch singleflight one"}],
                [{"role":"user","content":"batch singleflight two"}]
            ],
            "n": 2,
            "temperature": 0.0,
            "max_tokens": 2
        })
        .to_string();

        let (first, second) = tokio::join!(
            batch_post(state.clone(), &body),
            batch_post(state.clone(), &body)
        );
        let (status_first, first) = first;
        let (status_second, second) = second;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            (0, 2, 2),
            "concurrent duplicate batch should do prompt rendering once per unique prompt"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            (0, 2, 2),
            "concurrent duplicate batch should tokenize once per unique prompt"
        );
        assert_eq!(state.batch_cache.lock().unwrap().stats(), 1);
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["completions"], second["completions"]);
        assert_ne!(
            first["id"], second["id"],
            "singleflight replay should still get a fresh batch id"
        );
    }

    #[tokio::test]
    async fn batch_repeated_seeded_sampled_request_hits_completion_cache() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"cache seeded sample"}]],
            "n": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_first, first) = batch_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);

        let (status_second, second) = batch_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "second identical seeded sampled batch request should be served from full completion cache"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(
            first["completions"][0]["text"],
            second["completions"][0]["text"]
        );
    }

    #[tokio::test]
    async fn chat_repeated_greedy_request_hits_completion_cache() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"cache me once"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);

        let (status_second, second) = chat_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "second identical greedy chat request should be served from full completion cache"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(
            first["choices"][0]["message"]["content"],
            second["choices"][0]["message"]["content"]
        );
        assert_ne!(
            first["id"], second["id"],
            "cached chat responses should still get a fresh response id"
        );
        assert_eq!(state.recent_requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn chat_repeated_seeded_sampled_request_hits_completion_cache() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"cache sampled once"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);

        let (status_second, second) = chat_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "second identical seeded sampled chat request should be served from full completion cache"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(
            first["choices"][0]["message"]["content"],
            second["choices"][0]["message"]["content"]
        );
    }

    #[tokio::test]
    async fn chat_repeated_unseeded_sampled_request_does_not_use_completion_cache() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"do not cache random sample"}],
            "temperature": 0.7,
            "max_tokens": 4
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);

        let (status_second, second) = chat_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed)
                > generated_after_first,
            "unseeded sampled requests should keep performing physical generation"
        );
    }

    #[tokio::test]
    async fn chat_streaming_repeated_greedy_request_uses_completion_cache() {
        let state = make_batch_test_state();
        let base = serde_json::json!({
            "messages": [{"role":"user","content":"cache me as a stream"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        });

        let (status_first, first) = chat_post(state.clone(), &base.to_string()).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let mut stream_body_json = base;
        stream_body_json["stream"] = serde_json::Value::Bool(true);
        let (status_second, body) =
            chat_post_text(state.clone(), &stream_body_json.to_string()).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{body}");
        assert!(
            body.contains("chat.completion.chunk") && body.contains("[DONE]"),
            "cached streaming response should be SSE-shaped: {body}"
        );
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "cached streaming response should not perform model generation"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "streaming request-cache hit should return before rendered-prompt cache lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "streaming request-cache hit should return before prompt-token cache lookup"
        );
        assert_eq!(state.recent_requests.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn chat_streaming_completion_cache_hit_populates_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base = serde_json::json!({
            "messages": [{"role":"user","content":"stream lower cache should promote"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        });
        let req = parse_request(&base.to_string());
        let sampling = SamplingParams {
            temperature: requested_or_default_temperature(req.temperature),
            top_p: requested_or_default_top_p(req.top_p),
            top_k: requested_or_default_top_k(req.top_k),
            max_tokens: chat_request_max_tokens(&req),
            stop: normalized_stop_for_generation(req.stop.as_deref()),
            seed: req.seed,
            ..Default::default()
        };
        let prompt_text = render_prompt_text(
            &state,
            &req.messages,
            req.tools.as_deref(),
            req.tool_choice.as_ref(),
            req.chat_template_kwargs.as_ref(),
        )
        .unwrap();
        let prompt_tokens = encode_prompt_tokens(&state, &prompt_text).unwrap();
        let completion_cache_key =
            deterministic_completion_cache_key(&state, &prompt_tokens, &sampling, false)
                .expect("greedy request should be deterministic");
        state
            .completion_cache
            .lock()
            .unwrap()
            .insert_complete_value(
                completion_cache_key,
                DeterministicCompletionCacheValue {
                    text: "cached lower-layer stream".to_string(),
                    reasoning_content: None,
                    tool_calls: None,
                    finish_reason: "stop".to_string(),
                    completion_tokens: 3,
                },
            );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            (0, 1, 1)
        );
        assert_eq!(state.prompt_token_cache.lock().unwrap().stats(), (0, 1, 1));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

        let mut stream_body = base.clone();
        stream_body["stream"] = serde_json::Value::Bool(true);
        let (status_stream, stream) = chat_post_text(state.clone(), &stream_body.to_string()).await;
        assert_eq!(status_stream, axum::http::StatusCode::OK, "{stream}");
        assert!(stream.contains("cached lower-layer stream"));
        assert!(stream.contains("[DONE]"));
        let render_stats_after_stream = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_stream = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(
            render_stats_after_stream,
            (1, 1, 1),
            "streaming lower-cache hit still has to render before promotion"
        );
        assert_eq!(
            token_stats_after_stream,
            (1, 1, 1),
            "streaming lower-cache hit still has to tokenize before promotion"
        );
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            1,
            "streaming completion-cache hit should promote to chat request cache"
        );

        let (status_second, second) = chat_post(state.clone(), &base.to_string()).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            second["choices"][0]["message"]["content"],
            "cached lower-layer stream"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_stream,
            "promoted request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_stream,
            "promoted request-cache hit should return before prompt-token lookup"
        );
    }

    #[tokio::test]
    async fn chat_zero_max_tokens_returns_without_generation() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"do not decode"}],
            "temperature": 0.7,
            "max_tokens": 0
        })
        .to_string();

        let (status, body) = chat_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 0);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "max_tokens=0 should not enter model generation"
        );
        assert_eq!(state.recent_requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn chat_zero_max_completion_tokens_alias_returns_without_generation() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"do not decode alias"}],
            "temperature": 0.7,
            "max_completion_tokens": 0
        })
        .to_string();

        let (status, body) = chat_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 0);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "max_completion_tokens=0 should not enter model generation"
        );
        assert_eq!(state.recent_requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn repeated_zero_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"request cache avoids prompt work"}],
            "temperature": 0.7,
            "max_tokens": 0
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "chat request cache hit should return before rendered-prompt cache lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "chat request cache hit should return before prompt-token cache lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
        assert_ne!(
            first["id"], second["id"],
            "cached chat request should still get a fresh response id"
        );
    }

    #[tokio::test]
    async fn multi_choice_zero_chat_populates_request_cache_before_single_chat_work() {
        let state = make_batch_test_state();
        let multi_body = serde_json::json!({
            "messages": [{"role":"user","content":"zero chat n feeds single chat"}],
            "n": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 100
        })
        .to_string();
        let single_body = serde_json::json!({
            "messages": [{"role":"user","content":"zero chat n feeds single chat"}],
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();

        let (status_multi, multi) = chat_post(state.clone(), &multi_body).await;
        assert_eq!(status_multi, axum::http::StatusCode::OK, "{multi}");
        assert_eq!(multi["choices"].as_array().unwrap().len(), 4);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let render_stats_after_multi = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_multi = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_multi, (0, 1, 1));
        assert_eq!(token_stats_after_multi, (0, 1, 1));
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            1,
            "zero-token chat n should populate one normalized request-cache entry"
        );
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

        let (status_single, single) = chat_post(state.clone(), &single_body).await;
        assert_eq!(status_single, axum::http::StatusCode::OK, "{single}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "zero-token single chat hit should not generate model tokens"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_multi,
            "chat n-populated request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_multi,
            "chat n-populated request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
        assert_eq!(multi["usage"], single["usage"]);
        assert_eq!(multi["choices"][0], single["choices"][0]);
    }

    #[tokio::test]
    async fn chat_choices_cache_hit_rehydrates_request_cache_before_single_chat_work() {
        let state = make_batch_test_state();
        let multi_body = serde_json::json!({
            "messages": [{"role":"user","content":"choices hit feeds single chat"}],
            "n": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 199
        })
        .to_string();
        let single_body = serde_json::json!({
            "messages": [{"role":"user","content":"choices hit feeds single chat"}],
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();

        let (status_multi, multi) = chat_post(state.clone(), &multi_body).await;
        assert_eq!(status_multi, axum::http::StatusCode::OK, "{multi}");
        assert_eq!(multi["choices"].as_array().unwrap().len(), 4);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let render_stats_after_populate = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_populate = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_populate, (0, 1, 1));
        assert_eq!(token_stats_after_populate, (0, 1, 1));
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        *state.chat_request_cache.lock().unwrap() = DeterministicChatRequestCache::new(128);
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 0);

        let (status_hit, hit) = chat_post(state.clone(), &multi_body).await;
        assert_eq!(status_hit, axum::http::StatusCode::OK, "{hit}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_populate,
            "choices-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_populate,
            "choices-cache hit should return before prompt-token lookup"
        );
        assert_eq!(multi["usage"], hit["usage"]);
        assert_eq!(multi["choices"], hit["choices"]);
        assert_eq!(
            state.chat_request_cache.lock().unwrap().stats(),
            1,
            "choices-cache hit should rehydrate normalized request-cache entries"
        );

        let (status_single, single) = chat_post(state.clone(), &single_body).await;
        assert_eq!(status_single, axum::http::StatusCode::OK, "{single}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_populate,
            "rehydrated request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_populate,
            "rehydrated request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "zero-token single chat hit should not generate model tokens"
        );
        assert_eq!(hit["usage"], single["usage"]);
        assert_eq!(hit["choices"][0], single["choices"][0]);
    }

    #[tokio::test]
    async fn multi_choice_zero_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let single_body = serde_json::json!({
            "messages": [{"role":"user","content":"single chat feeds zero chat n"}],
            "temperature": 0.2,
            "top_p": 0.1,
            "max_tokens": 0,
            "seed": 999
        })
        .to_string();
        let multi_body = serde_json::json!({
            "messages": [{"role":"user","content":"single chat feeds zero chat n"}],
            "n": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 0,
            "seed": 202
        })
        .to_string();

        let (status_single, single) = chat_post(state.clone(), &single_body).await;
        assert_eq!(status_single, axum::http::StatusCode::OK, "{single}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let render_stats_after_single = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_single = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_single, (0, 1, 1));
        assert_eq!(token_stats_after_single, (0, 1, 1));
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 0);

        let (status_multi, multi) = chat_post(state.clone(), &multi_body).await;
        assert_eq!(status_multi, axum::http::StatusCode::OK, "{multi}");
        assert_eq!(multi["choices"].as_array().unwrap().len(), 4);
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_single,
            "zero-token chat n should hit request cache before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_single,
            "zero-token chat n should hit request cache before prompt-token lookup"
        );
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "zero-token chat n request-cache hit should not generate model tokens"
        );
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);
        assert_eq!(single["usage"], multi["usage"]);
        assert_eq!(single["choices"][0], multi["choices"][0]);

        let (status_repeat, repeat) = chat_post(state.clone(), &multi_body).await;
        assert_eq!(status_repeat, axum::http::StatusCode::OK, "{repeat}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_single,
            "request-cache synthesized choices should seed the choices cache"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_single,
            "choices-cache repeat should still avoid prompt-token lookup"
        );
        assert_eq!(multi["usage"], repeat["usage"]);
        assert_eq!(multi["choices"], repeat["choices"]);
    }

    #[tokio::test]
    async fn top_k_one_chat_hits_greedy_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let greedy_body = serde_json::json!({
            "messages": [{"role":"user","content":"top k one request cache"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 1
        })
        .to_string();
        let top_k_one_body = serde_json::json!({
            "messages": [{"role":"user","content":"top k one request cache"}],
            "temperature": 0.7,
            "top_p": 0.2,
            "top_k": 1,
            "max_tokens": 4
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &greedy_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &top_k_one_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_k=1 should hit the equivalent greedy request cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_k=1 request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_k=1 request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn top_p_above_one_chat_hits_seeded_full_distribution_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"top p full distribution request cache"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 4,
            "seed": 140
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "messages": [{"role":"user","content":"top p full distribution request cache"}],
            "temperature": 0.7,
            "top_p": 1.5,
            "max_tokens": 4,
            "seed": 140
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &repeat_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_p >= 1.0 seeded chat repeat should hit the equivalent request cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_p >= 1.0 request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_p >= 1.0 request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn top_p_zero_chat_hits_seeded_full_distribution_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"top p zero full distribution request cache"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 4,
            "seed": 144
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "messages": [{"role":"user","content":"top p zero full distribution request cache"}],
            "temperature": 0.7,
            "top_p": 0.0,
            "max_tokens": 4,
            "seed": 144
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &repeat_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_p=0 seeded chat repeat should hit the equivalent request cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_p=0 request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_p=0 request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn top_k_oversized_chat_hits_seeded_full_distribution_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let disabled_top_k = state.model_config.vocab_size as u32;
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"top k oversized full distribution request cache"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": disabled_top_k,
            "max_tokens": 4,
            "seed": 148
        })
        .to_string();
        let repeat_body = serde_json::json!({
            "messages": [{"role":"user","content":"top k oversized full distribution request cache"}],
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 4,
            "seed": 148
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &repeat_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "top_k >= vocab seeded chat repeat should hit the equivalent request cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "top_k >= vocab request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "top_k >= vocab request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn empty_tools_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"empty tools should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let empty_tools_body = serde_json::json!({
            "messages": [{"role":"user","content":"empty tools should be no-op"}],
            "tools": [],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &empty_tools_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "empty-tools request should reuse the no-tools cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "empty-tools request cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "empty-tools request cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn no_tool_auto_choice_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"auto tool choice should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let no_op_choice_body = serde_json::json!({
            "messages": [{"role":"user","content":"auto tool choice should be no-op"}],
            "tools": [],
            "tool_choice": "auto",
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &no_op_choice_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "no-tool auto choice should reuse the no-tools cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "no-tool auto choice request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "no-tool auto choice request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn no_tool_none_choice_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"none tool choice should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let no_op_choice_body = serde_json::json!({
            "messages": [{"role":"user","content":"none tool choice should be no-op"}],
            "tools": [],
            "tool_choice": "none",
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &no_op_choice_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "no-tool none choice should reuse the no-tools cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "no-tool none choice request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "no-tool none choice request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn no_tool_none_choice_chat_multi_choice_hits_choices_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"none tool choice chat n should be no-op"}],
            "n": 4,
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let no_op_choice_body = serde_json::json!({
            "messages": [{"role":"user","content":"none tool choice chat n should be no-op"}],
            "n": 4,
            "tools": [],
            "tool_choice": "none",
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_choices_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &no_op_choice_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "no-tool none choice should reuse the no-tools chat choices cache"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "no-tool none choices-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "no-tool none choices-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn input_reasoning_content_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"input reasoning content should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let reasoning_body = serde_json::json!({
            "messages": [{
                "role":"user",
                "content":"input reasoning content should be no-op",
                "reasoning_content":"ignored by prompt rendering"
            }],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &reasoning_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "input reasoning_content should reuse the rendered-equivalent cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "input reasoning_content request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "input reasoning_content request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn text_content_parts_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"text content parts should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let parts_body = serde_json::json!({
            "messages": [{
                "role":"user",
                "content":[
                    {"type":"text","text":"text content parts "},
                    {"type":"text","text":"should be no-op"}
                ]
            }],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &parts_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "equivalent text content parts should reuse the cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "text content parts request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "text content parts request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn non_text_content_parts_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [{"role":"user","content":"non-text content parts should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let parts_body = serde_json::json!({
            "messages": [{
                "role":"user",
                "content":[
                    {"type":"text","text":"non-text content parts "},
                    {"type":"image_url","image_url":{"url":"https://example.invalid/ignored.png"}},
                    {"type":"input_audio","input_audio":{"data":"ignored","format":"wav"}},
                    {"type":"text","text":"should be no-op"}
                ]
            }],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &parts_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "non-text content parts should reuse the cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "non-text content parts request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "non-text content parts request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn max_completion_tokens_alias_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let max_tokens_body = serde_json::json!({
            "messages": [{"role":"user","content":"max completion tokens alias should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let alias_body = serde_json::json!({
            "messages": [{"role":"user","content":"max completion tokens alias should be no-op"}],
            "temperature": 0.0,
            "max_completion_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &max_tokens_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &alias_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "max_completion_tokens should reuse the max_tokens cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "max_completion_tokens request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "max_completion_tokens request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn stop_string_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let list_body = serde_json::json!({
            "messages": [{"role":"user","content":"stop string should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop"],
            "seed": 123
        })
        .to_string();
        let string_body = serde_json::json!({
            "messages": [{"role":"user","content":"stop string should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": "never-match-stop",
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &list_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &string_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "single-string stop should reuse the one-item list cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "stop-string request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "stop-string request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn dominated_stop_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let redundant_body = serde_json::json!({
            "messages": [{"role":"user","content":"dominated stop should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop", "never-match-stop-suffix"],
            "seed": 123
        })
        .to_string();
        let minimal_body = serde_json::json!({
            "messages": [{"role":"user","content":"dominated stop should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop"],
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &redundant_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &minimal_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "dominated stop sequences should reuse the minimal stop cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "dominated-stop request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "dominated-stop request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn substring_dominated_stop_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let redundant_body = serde_json::json!({
            "messages": [{"role":"user","content":"substring dominated stop should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["prefix-never-match-stop-suffix", "never-match-stop"],
            "seed": 123
        })
        .to_string();
        let minimal_body = serde_json::json!({
            "messages": [{"role":"user","content":"substring dominated stop should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "stop": ["never-match-stop"],
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &redundant_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);

        let (status_second, second) = chat_post(state.clone(), &minimal_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "substring-dominated stop sequences should reuse the minimal stop cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "substring-dominated-stop request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "substring-dominated-stop request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn default_openai_option_fields_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let plain_body = serde_json::json!({
            "messages": [{"role":"user","content":"default OpenAI options should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let defaults_body = serde_json::json!({
            "messages": [{"role":"user","content":"default OpenAI options should be no-op"}],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999,
            "n": 1,
            "response_format": {"type":"text"},
            "parallel_tool_calls": true,
            "user": "client-a",
            "metadata": {"trace_id":"ignored"},
            "store": false,
            "service_tier": "auto",
            "logprobs": false,
            "top_logprobs": 0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream_options": {"include_usage": false}
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &plain_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &defaults_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "default OpenAI options should reuse the cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "default-option request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "default-option request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn empty_message_tool_calls_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [
                {"role":"user","content":"empty message tool calls should be no-op"},
                {"role":"assistant","content":"ok"},
                {"role":"user","content":"continue"}
            ],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let empty_tool_calls_body = serde_json::json!({
            "messages": [
                {"role":"user","content":"empty message tool calls should be no-op"},
                {"role":"assistant","content":"ok","tool_calls":[]},
                {"role":"user","content":"continue"}
            ],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &empty_tool_calls_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "empty message tool_calls should reuse the rendered-equivalent cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "empty message tool_calls request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "empty message tool_calls request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn tool_call_argument_json_string_chat_hits_request_cache_before_prompt_work() {
        let state = make_batch_test_state();
        let base_body = serde_json::json!({
            "messages": [
                {"role":"user","content":"tool call argument JSON should be canonical"},
                {"role":"assistant","content":null,"tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"Lookup","arguments":"{\"query\":\"cache\",\"limit\":2}"}
                }]},
                {"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},
                {"role":"user","content":"continue"}
            ],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 123
        })
        .to_string();
        let whitespace_args_body = serde_json::json!({
            "messages": [
                {"role":"user","content":"tool call argument JSON should be canonical"},
                {"role":"assistant","content":null,"tool_calls":[{
                    "id":"call_1",
                    "type":"function",
                    "function":{"name":"Lookup","arguments":"{ \"limit\" : 2, \"query\" : \"cache\" }"}
                }]},
                {"role":"tool","content":"done","name":"Lookup","tool_call_id":"call_1"},
                {"role":"user","content":"continue"}
            ],
            "temperature": 0.0,
            "max_tokens": 4,
            "seed": 999
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &base_body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let generated_after_first = state
            .metrics
            .tokens_generated
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(generated_after_first > 0);
        let render_stats_after_first = state.rendered_prompt_cache.lock().unwrap().stats();
        let token_stats_after_first = state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(render_stats_after_first, (0, 1, 1));
        assert_eq!(token_stats_after_first, (0, 1, 1));

        let (status_second, second) = chat_post(state.clone(), &whitespace_args_body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            generated_after_first,
            "JSON-equivalent tool_call arguments should reuse the cached response"
        );
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            render_stats_after_first,
            "tool_call argument JSON request-cache hit should return before rendered-prompt lookup"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            token_stats_after_first,
            "tool_call argument JSON request-cache hit should return before prompt-token lookup"
        );
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
    }

    #[tokio::test]
    async fn concurrent_zero_chat_singleflights_before_prompt_work() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"chat request singleflight"}],
            "temperature": 0.0,
            "max_tokens": 0
        })
        .to_string();

        let (first, second) = tokio::join!(
            chat_post(state.clone(), &body),
            chat_post(state.clone(), &body)
        );
        let (status_first, first) = first;
        let (status_second, second) = second;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");
        assert_eq!(
            state.rendered_prompt_cache.lock().unwrap().stats(),
            (0, 1, 1),
            "concurrent identical chat requests should render once"
        );
        assert_eq!(
            state.prompt_token_cache.lock().unwrap().stats(),
            (0, 1, 1),
            "concurrent identical chat requests should tokenize once"
        );
        assert_eq!(state.chat_request_cache.lock().unwrap().stats(), 1);
        assert_eq!(first["usage"], second["usage"]);
        assert_eq!(first["choices"], second["choices"]);
        assert_ne!(first["id"], second["id"]);
    }

    #[tokio::test]
    async fn repeated_chat_prompt_reuses_prompt_token_cache() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"tokenize this once"}],
            "temperature": 0.7,
            "max_tokens": 1
        })
        .to_string();

        let (status_first, first) = chat_post(state.clone(), &body).await;
        assert_eq!(status_first, axum::http::StatusCode::OK, "{first}");
        let (status_second, second) = chat_post(state.clone(), &body).await;
        assert_eq!(status_second, axum::http::StatusCode::OK, "{second}");

        let (render_hits, render_misses, render_entries) =
            state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!(
            render_misses, 1,
            "first chat request should miss rendered-prompt cache"
        );
        assert_eq!(
            render_hits, 1,
            "second identical chat request should hit rendered-prompt cache"
        );
        assert_eq!(render_entries, 1);

        let (token_hits, token_misses, token_entries) =
            state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(
            token_misses, 1,
            "first rendered prompt should miss token cache"
        );
        assert_eq!(
            token_hits, 1,
            "second identical rendered prompt should hit token cache"
        );
        assert_eq!(token_entries, 1);
    }

    #[tokio::test]
    async fn chat_streaming_zero_max_tokens_returns_sse_without_generation() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "messages": [{"role":"user","content":"do not stream decode"}],
            "temperature": 0.0,
            "max_tokens": 0,
            "stream": true
        })
        .to_string();

        let (status, body) = chat_post_text(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("\"finish_reason\":\"length\""));
        assert!(body.contains("[DONE]"));
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "streaming max_tokens=0 should not enter model generation"
        );
        assert_eq!(state.recent_requests.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn batch_zero_max_tokens_returns_without_generation() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"zero one"}],
                [{"role":"user","content":"zero two"}]
            ],
            "n": 2,
            "temperature": 0.7,
            "max_tokens": 0
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let completions = body["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 4);
        assert!(completions.iter().all(|item| item["text"] == ""));
        assert!(
            completions
                .iter()
                .all(|item| item["finish_reason"] == "length")
        );
        assert_eq!(body["usage"]["completion_tokens"], 0);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "batch max_tokens=0 should not enter model generation"
        );
    }

    #[tokio::test]
    async fn batch_zero_max_completion_tokens_alias_returns_without_generation() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"zero alias one"}],
                [{"role":"user","content":"zero alias two"}]
            ],
            "n": 2,
            "temperature": 0.7,
            "max_completion_tokens": 0
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        let completions = body["completions"].as_array().unwrap();
        assert_eq!(completions.len(), 4);
        assert!(completions.iter().all(|item| item["text"] == ""));
        assert!(
            completions
                .iter()
                .all(|item| item["finish_reason"] == "length")
        );
        assert_eq!(body["usage"]["completion_tokens"], 0);
        assert_eq!(
            state
                .metrics
                .tokens_generated
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "batch max_completion_tokens=0 should not enter model generation"
        );
    }

    #[tokio::test]
    async fn duplicate_batch_zero_prompts_skip_repeated_render_and_tokenize() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [
                [{"role":"user","content":"same token prompt"}],
                [{"role":"user","content":"same token prompt"}]
            ],
            "n": 1,
            "temperature": 0.7,
            "max_tokens": 0
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["completions"].as_array().unwrap().len(), 2);
        assert_eq!(body["completions"][0]["prompt_index"], 0);
        assert_eq!(body["completions"][1]["prompt_index"], 1);

        let (render_hits, render_misses, render_entries) =
            state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!(
            render_misses, 1,
            "duplicate zero-token batch prompts should render once"
        );
        assert_eq!(
            render_hits, 0,
            "duplicate zero-token batch prompts should skip repeated render lookups"
        );
        assert_eq!(render_entries, 1);

        let (token_hits, token_misses, token_entries) =
            state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(
            token_misses, 1,
            "duplicate zero-token batch prompts should tokenize once"
        );
        assert_eq!(
            token_hits, 0,
            "duplicate zero-token batch prompts should skip repeated token lookups"
        );
        assert_eq!(token_entries, 1);
    }

    #[tokio::test]
    async fn batch_multi_sample_prepares_prompt_once_per_group() {
        let state = make_batch_test_state();
        let body = serde_json::json!({
            "prompts": [[{"role":"user","content":"same sampled prompt"}]],
            "n": 3,
            "temperature": 0.7,
            "max_tokens": 1,
            "seed": 282
        })
        .to_string();

        let (status, body) = batch_post(state.clone(), &body).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{body}");
        assert_eq!(body["completions"].as_array().unwrap().len(), 3);

        let (render_hits, render_misses, render_entries) =
            state.rendered_prompt_cache.lock().unwrap().stats();
        assert_eq!(
            (render_hits, render_misses, render_entries),
            (0, 1, 1),
            "sampled n>1 batch should prepare the shared prompt once"
        );

        let (token_hits, token_misses, token_entries) =
            state.prompt_token_cache.lock().unwrap().stats();
        assert_eq!(
            (token_hits, token_misses, token_entries),
            (0, 1, 1),
            "sampled n>1 batch should tokenize the shared prompt once"
        );
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
