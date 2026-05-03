use axum::extract::{DefaultBodyLimit, State};
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
use kiln_model::{
    CancelHandle, GenerationOutput, ModelRunner, PagedPrefixReuse, SpeculativeConfig, StreamEvent,
};

use crate::config::{SpecMethod, SpeculativeDecodingConfig};
use crate::error::ApiError;
use crate::metrics::RequestStatus;
use crate::recent_requests::{RequestRecord, now_unix_ms, truncate_chars};
use crate::state::{AppState, ModelBackend, RealPrefixCache};

/// Max characters retained in the prompt preview for the recent-requests panel.
const PROMPT_PREVIEW_MAX_CHARS: usize = 120;
/// Max characters retained in the completion preview for the recent-requests panel.
const COMPLETION_PREVIEW_MAX_CHARS: usize = 200;

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
    completion_buf: &mut String,
) -> bool {
    if chunk.is_empty() {
        return true;
    }
    if let Some(text) = chunk.reasoning {
        // Reasoning content also feeds the dashboard preview when the
        // answer hasn't started yet — without this the preview is blank
        // until the model emits `</think>`, which can be hundreds of
        // tokens in.
        if completion_buf.chars().count() < COMPLETION_PREVIEW_MAX_CHARS + 16 {
            completion_buf.push_str(&text);
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
                    content: None,
                    reasoning_content: Some(text),
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
        if completion_buf.chars().count() < COMPLETION_PREVIEW_MAX_CHARS + 16 {
            completion_buf.push_str(&text);
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
                    content: Some(text),
                    reasoning_content: None,
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

/// Non-streaming variant: split a fully-generated response text into
/// `(reasoning_content, content)` around the same `</think>` boundary the
/// streaming splitter handles. Returns `(None, raw)` when the prompt did not
/// prefill `<think>\n` so non-reasoning models keep emitting plain content.
fn split_reasoning_response(
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
        tool_calls: m.tool_calls.clone(),
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
    /// Streaming counterpart of [`Message::reasoning_content`]. Each chunk
    /// emits at most one of `reasoning_content` (while inside a
    /// `<think>...</think>` block) or `content` (after the close tag).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
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

    // Convert request messages to ChatMessage for template formatting,
    // forwarding `tool_calls` / `name` / `tool_call_id` so multi-turn
    // tool-use conversations render correctly. Tools schema is threaded
    // separately via `apply_chat_template_with_tools`.
    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();

    // Apply chat template and tokenize
    let prompt_text = state
        .tokenizer
        .apply_chat_template_full(&chat_messages, req.tools.as_deref(), req.tool_choice.as_ref())
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
    let memory_budget = state.memory_budget.clone();
    let metrics = state.metrics.clone();
    let timeout = state.request_timeout;
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
                    runner_guard.generate_paged_shared_tokens(
                        &prompt_tokens,
                        &params,
                        bm.as_ref(),
                        pc.as_ref(),
                        Some(&cancel_inner),
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
                        Some(&cancel_inner),
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
    // Qwen3.5's chat template prefills `<think>\n` into the assistant turn,
    // so the model emits chain-of-thought directly and closes with
    // `</think>` before the actual answer. Split into the llama.cpp /
    // DeepSeek-shaped `(reasoning_content, content)` pair so OpenAI-compat
    // clients can render the two channels separately. For non-reasoning
    // templates this returns `(None, output.text)` and the response shape
    // is byte-identical to before.
    let (reasoning_content, completion_text) =
        split_reasoning_response(&output.text, prompt_text);

    // Recent-requests preview wants the user-visible answer, but the
    // reasoning often dominates the first few hundred chars. Show
    // reasoning when the answer is empty (still mid-thought at
    // max_tokens cutoff) so the dashboard isn't blank.
    let preview_source = if completion_text.is_empty() {
        reasoning_content.as_deref().unwrap_or("")
    } else {
        completion_text.as_str()
    };
    record_recent_request(
        state,
        RequestRecord {
            id: id.clone(),
            timestamp_unix_ms: now_unix_ms(),
            model: model.clone(),
            prompt_preview: truncate_chars(&last_user_message_text(req), PROMPT_PREVIEW_MAX_CHARS),
            completion_preview: truncate_chars(preview_source, COMPLETION_PREVIEW_MAX_CHARS),
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
                reasoning_content,
                tool_calls: None,
                name: None,
                tool_call_id: None,
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
    let memory_budget = state.memory_budget.clone();
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
                        reasoning_content: None,
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
                                // Route this token through the reasoning
                                // splitter — at most one of `content` or
                                // `reasoning_content` lands in the delta,
                                // depending on which side of `</think>` the
                                // splitter currently sits on. A token that
                                // straddles the boundary may emit both in
                                // separate chunks (rare; emitted in order).
                                let chunk = reasoning_splitter.push(&token.text);
                                if !emit_reasoning_chunk(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    chunk,
                                    &mut completion_buf,
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
                                if !emit_reasoning_chunk(
                                    &tx,
                                    &id,
                                    created,
                                    &model,
                                    trailing,
                                    &mut completion_buf,
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
                // Drain any pending partial-tag tail before the timeout
                // chunk so the client doesn't lose those bytes.
                let trailing = reasoning_splitter.flush();
                let _ = emit_reasoning_chunk(
                    &tx,
                    &id,
                    created,
                    &model,
                    trailing,
                    &mut completion_buf,
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
                // Mock backend never emits a reasoning block.
                reasoning_content: None,
                tool_calls: None,
                name: None,
                tool_call_id: None,
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
/// posts the whole group in one call. Real serving currently dispatches each
/// output through the normal chat-completions path; the continuous-batching
/// rebuild wires real decode scheduling through the production backend in
/// phases.
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

    // Resolve adapter once for the entire batch. After this returns,
    // state.active_adapter_name reflects the loaded adapter and every
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
            ensure_adapter(state, runner, &req.adapter).await?;
        }
    }

    // Spawn one task per (prompt, completion) pair. Each task synthesizes a
    // ChatCompletionRequest with this prompt's messages and a derived seed,
    // then dispatches through the existing generate_real / generate_mock
    // path. This endpoint is an API-level batching convenience, not a separate
    // engine-level batch path.
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
                    // Batch input messages never carry historical reasoning;
                    // the chat template renders them as plain text turns.
                    reasoning_content: None,
                    tool_calls: None,
                    name: None,
                    tool_call_id: None,
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
                tools: None,
                tool_choice: None,
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

    let chat_messages: Vec<ChatMessage> = req.messages.iter().map(message_to_chat).collect();

    let prompt_text = state
        .tokenizer
        .apply_chat_template_full(&chat_messages, req.tools.as_deref(), req.tool_choice.as_ref())
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

/// Body-size cap for /v1/chat/completions (audit LOW §1).
/// 8 MiB is generous for chat payloads while bounding memory DoS via JSON extraction.
const CHAT_BODY_LIMIT: usize = 8 * 1024 * 1024;
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
        assert_eq!(req.tool_choice.as_ref().and_then(|v| v.as_str()), Some("auto"));
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
        assert_eq!(req.messages.len(), 5, "fixture must exercise multi-turn shape");
        assert_eq!(
            req.tools.as_ref().map(|t| t.len()),
            Some(2),
            "tools array must round-trip through deserialization"
        );

        // Step 2: load the production bundled Qwen3.5-4B chat template — the
        // canonical template every kiln user actually hits at runtime. Path is
        // relative to this source file (crates/kiln-server/src/api/...).
        let template = include_str!(
            "../../../kiln-core/test_fixtures/qwen35_4b_chat_template.jinja"
        );
        let tok = crate::api::test_tokenizer().with_chat_template(template.to_string());

        // Step 3: wire EXACTLY as `chat_completions_inner` does (see the
        // `let chat_messages = ... map(message_to_chat) ...` /
        // `apply_chat_template_full(..., req.tools.as_deref(), req.tool_choice.as_ref())`
        // pair near the top of that function). Drift between this test and the
        // production wiring would defeat the point of the smoke test.
        let chat_messages: Vec<ChatMessage> =
            req.messages.iter().map(message_to_chat).collect();
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
        let calls = req.messages[0].tool_calls.as_ref().expect("tool_calls present");
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
        assert!(req.tools.is_none(), "tools should default to None when absent");
        assert!(req.tool_choice.is_none(), "tool_choice should default to None");
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
