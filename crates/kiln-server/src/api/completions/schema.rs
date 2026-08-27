use super::*;

/// OpenAI-compatible chat completion request adapter selection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ChatAdapterSelection {
    /// The request omitted `adapter`; use the server default as-is.
    #[default]
    Default,
    /// The request explicitly selected base model for this request.
    Base,
    /// The request explicitly selected a named adapter for this request.
    Named(String),
}

impl ChatAdapterSelection {
    pub(super) fn is_explicit(&self) -> bool {
        !matches!(self, Self::Default)
    }

    pub(super) fn request_adapter_name(&self) -> Option<String> {
        match self {
            Self::Named(name) => Some(name.clone()),
            Self::Default | Self::Base => None,
        }
    }

    pub(super) fn target_adapter_name(&self, default_adapter: Option<String>) -> Option<String> {
        match self {
            Self::Default => default_adapter,
            Self::Base => None,
            Self::Named(name) => Some(name.clone()),
        }
    }

    pub(super) fn reason(&self) -> &'static str {
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
pub(super) fn message_to_chat(m: &Message) -> ChatMessage {
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
pub(super) fn deserialize_optional_stop<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error>
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

pub(super) fn serialize_optional_thinking_budget_status<S>(
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

// Test-only: pins the EOS-inclusive usage convention in `mod tests`
// (completions/tests/mod.rs `completion_usage_counts_terminal_eos_token`);
// the live streaming path uses `batching_engine::completion_usage_tokens`.
#[allow(dead_code)]
pub(super) fn completion_usage_tokens(
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
