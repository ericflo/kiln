use crate::token::TokenId;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("failed to encode text: {0}")]
    Encode(String),
    #[error("failed to decode tokens: {0}")]
    Decode(String),
    #[error("chat template error: {0}")]
    ChatTemplate(String),
}

/// A chat message for template formatting.
///
/// `tool_calls`, `name`, and `tool_call_id` are skipped on serialize when None
/// so plain chat conversations render byte-identically to the pre-tools shape.
/// When populated they let Qwen3.5-style chat templates render past assistant
/// tool calls and `role: "tool"` responses correctly via `{% if message.tool_calls %}`
/// branches.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Wraps the HuggingFace tokenizers crate for Kiln's tokenization needs.
#[derive(Clone)]
pub struct KilnTokenizer {
    inner: Tokenizer,
    chat_template: Option<String>,
}

impl KilnTokenizer {
    /// Load a tokenizer from the HuggingFace Hub by model ID (e.g. "Qwen/Qwen3.5-4B").
    pub fn from_pretrained(model_id: &str) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_pretrained(model_id, None)
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            chat_template: None,
        })
    }

    /// Create from a pre-built `Tokenizer` instance.
    pub fn from_inner(inner: Tokenizer) -> Self {
        Self {
            inner,
            chat_template: None,
        }
    }

    /// Create from raw JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner =
            Tokenizer::from_bytes(bytes).map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            chat_template: None,
        })
    }

    /// Load a tokenizer from a local tokenizer.json file.
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path).map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            chat_template: None,
        })
    }

    /// Set a Jinja2 chat template string explicitly (from tokenizer_config.json).
    pub fn with_chat_template(mut self, template: String) -> Self {
        self.chat_template = Some(template);
        self
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<TokenId>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[TokenId]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Apply the chat template to format messages into a prompt string.
    ///
    /// If a Jinja2 template was set via [`with_chat_template`], renders it with
    /// minijinja. Otherwise falls back to plain ChatML framing — a minimal
    /// subset that omits Qwen3.5's XML `<function=...>` tool calls, `<think>`
    /// reasoning, and `<tool_response>` handling. Load the model's real
    /// `chat_template` via `with_chat_template` for production use.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String, TokenizerError> {
        self.apply_chat_template_with_tools(messages, None)
    }

    /// Same as [`apply_chat_template`] but also threads OpenAI-style tool/function
    /// definitions into the Jinja chat-template context as `tools`. Templates
    /// that branch on `{% if tools %}` (e.g. Qwen3.5's official template) emit
    /// their tool-calling prelude only when this is `Some(non-empty)`. Pass
    /// `None` (or an empty slice) for plain chat.
    ///
    /// The `tools` slice is forwarded as opaque JSON: kiln does not validate
    /// the OpenAI tool schema, it just hands the values to the template.
    pub fn apply_chat_template_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String, TokenizerError> {
        self.apply_chat_template_full(messages, tools, None)
    }

    /// Full-context variant: also threads `tool_choice` into the Jinja
    /// context. Most templates ignore `tool_choice`; this exists so OpenAI
    /// clients that send `"none" | "auto" | "required"` (or a named-function
    /// object) round-trip through templates that DO consult it without losing
    /// the field. Pass `None` to omit.
    pub fn apply_chat_template_full(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        tool_choice: Option<&serde_json::Value>,
    ) -> Result<String, TokenizerError> {
        match &self.chat_template {
            Some(template) => self.render_jinja_template(template, messages, tools, tool_choice),
            None => Ok(Self::apply_chatml(messages)),
        }
    }

    /// Get the EOS token IDs from the tokenizer's added tokens.
    pub fn eos_token_ids(&self) -> Vec<TokenId> {
        let mut eos_ids = Vec::new();
        for (id, token) in self.inner.get_added_tokens_decoder() {
            let content = token.content.as_str();
            if token.special
                && (content == "<|endoftext|>"
                    || content == "<|im_end|>"
                    || content == "<|end|>"
                    || content == "</s>")
            {
                eos_ids.push(id);
            }
        }
        eos_ids.sort();
        eos_ids
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn render_jinja_template(
        &self,
        template: &str,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        tool_choice: Option<&serde_json::Value>,
    ) -> Result<String, TokenizerError> {
        // Most HF chat templates (Qwen2.5/3.5, Llama 3.1, Mistral v0.3) iterate
        // or `tojson`-serialize `tool_call.function.arguments` as a dict, so we
        // deserialize the JSON-encoded OpenAI-wire-format string in-place
        // before rendering. DeepSeek V3 is the outlier: it concatenates
        // `arguments` directly into a string-fenced ` ```json ... ``` ` body,
        // which minijinja rejects as "string + map" when arguments has been
        // promoted to a dict. Try the dict-deserialized form first (matches
        // HF default convention) and retry with arguments left as the raw
        // OpenAI-wire JSON string only if minijinja explicitly rejects the
        // map+string combination — preserves all existing behavior while
        // making DeepSeek-style templates render without hand-editing the
        // vendored chat_template.jinja.
        match self.render_jinja_template_with(
            template,
            messages,
            tools,
            tool_choice,
            /* deserialize_arguments = */ true,
        ) {
            Err(TokenizerError::ChatTemplate(msg))
                if msg.contains("string and map") || msg.contains("map and string") =>
            {
                self.render_jinja_template_with(
                    template,
                    messages,
                    tools,
                    tool_choice,
                    /* deserialize_arguments = */ false,
                )
            }
            other => other,
        }
    }

    fn render_jinja_template_with(
        &self,
        template: &str,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        tool_choice: Option<&serde_json::Value>,
        deserialize_arguments: bool,
    ) -> Result<String, TokenizerError> {
        let mut env = minijinja::Environment::new();
        // Real chat templates (e.g. Qwen3.5's `chat_template.jinja`) call
        // Python-flavored string methods like `.startswith()`, `.endswith()`,
        // `.split()`, and `.lstrip()` that minijinja does not implement
        // natively. `pycompat::unknown_method_callback` adds the standard
        // Python `str`/`list`/`dict` methods on demand so HF templates render
        // unmodified.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // `raise_exception("...")` is a Jinja2 idiom HF templates rely on for
        // input validation. Without it Qwen3.5's template aborts immediately.
        env.add_function("raise_exception", |msg: String| -> Result<(), minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });
        env.add_template("chat", template)
            .map_err(|e| TokenizerError::ChatTemplate(e.to_string()))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| TokenizerError::ChatTemplate(e.to_string()))?;

        // Empty `tools` is treated as "no tools" so `{% if tools %}` branches
        // do not fire — matches HuggingFace's tokenizer convention and avoids
        // emitting an empty `<tools></tools>` block.
        let tools_value = tools
            .filter(|t| !t.is_empty())
            .map(|t| serde_json::Value::Array(t.to_vec()))
            .unwrap_or(serde_json::Value::Null);
        let tool_choice_value = tool_choice
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        // Deserialize tool_call arguments from JSON-string to structured
        // serde_json::Value before handing the messages to the Jinja
        // template. The OpenAI Chat Completions spec defines
        // `tool_calls[*].function.arguments` as a JSON-encoded string
        // (https://platform.openai.com/docs/api-reference/chat/object), so
        // OpenAI-compatible clients (and the kiln-server completions API)
        // forward it as a string. HuggingFace chat templates — including
        // Qwen3.5-4B's bundled `chat_template.jinja` — iterate it as a dict
        // via `{% for k, v in tool_call.arguments|items %}`. Without this
        // conversion the `|items` filter throws
        // "cannot convert value into pairs" on every multi-turn request that
        // carries prior assistant tool_calls.
        //
        // Both the OpenAI `{function: {name, arguments}}` envelope and the
        // flatter `{name, arguments}` shape some templates assume are
        // handled. If `arguments` is already a JSON object/array (i.e. the
        // caller pre-parsed it), it's left alone. When `deserialize_arguments`
        // is false (DeepSeek V3 fallback path), the JSON-string form is
        // preserved verbatim so templates can concatenate it as a string.
        let messages_value: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut v = serde_json::to_value(m).unwrap_or(serde_json::Value::Null);
                let has_tool_calls = if let Some(tcs) =
                    v.get_mut("tool_calls").and_then(|t| t.as_array_mut())
                {
                    if deserialize_arguments {
                        for tc in tcs.iter_mut() {
                            deserialize_arguments_in_place(tc);
                            if let Some(function) = tc.get_mut("function") {
                                deserialize_arguments_in_place(function);
                            }
                        }
                    }
                    true
                } else {
                    false
                };
                // OpenAI Chat Completions spec sends `content: null` for
                // assistant messages carrying `tool_calls`. ChatMessage uses
                // `content: String` (with `""` as the empty placeholder) for
                // ergonomic Rust callers, so the serialized form would emit
                // `""` even when the OpenAI wire shape is `null`. Some HF
                // chat templates branch on `content is none` to decide
                // whether to emit the tool_calls block — DeepSeek V3 is the
                // canonical example: empty string is NOT `none`, so without
                // this normalization the tool_calls block is silently dropped
                // and the assistant turn renders as a content-less
                // `<｜Assistant｜><｜end▁of▁sentence｜>` stub. Templates that
                // test `if message.content` are unaffected since both `""`
                // and `null` are falsy in Jinja.
                if has_tool_calls
                    && v.get("content").and_then(|c| c.as_str()) == Some("")
                {
                    v["content"] = serde_json::Value::Null;
                }
                v
            })
            .collect();

        // `bos_token` / `eos_token` are referenced by many HF chat templates
        // — Mistral 7B Instruct v0.3, for example, embeds `eos_token` in
        // `+` expressions like `" " + message["content"]|trim + eos_token`
        // and `"]" + eos_token`. Minijinja's default Undefined behavior
        // treats `string + undefined` as an error (only standalone
        // `{{ undefined }}` renders empty), so leaving these out crashes
        // any render that hits an assistant turn under Mistral. Default
        // both to empty strings so the template renders structurally; the
        // tokenizer's own `add_special_tokens=true` path handles real
        // BOS/EOS injection during encode (template-emitted token strings
        // would otherwise risk double-tokenization). Threading the actual
        // model-specific token strings is a follow-up — when needed,
        // populate from `KilnTokenizer::inner.get_added_tokens_decoder()`
        // similar to `eos_token_ids`.
        tmpl.render(minijinja::context! {
            messages => messages_value,
            tools => tools_value,
            tool_choice => tool_choice_value,
            add_generation_prompt => true,
            bos_token => "",
            eos_token => "",
        })
        .map_err(|e| TokenizerError::ChatTemplate(e.to_string()))
    }

    /// Plain ChatML framing fallback: `<|im_start|>role\n...<|im_end|>\n`.
    /// Qwen3.5's real template adds XML `<function=...>` tool calls, `<think>`
    /// reasoning, and `<tool_response>` wrapping on top of this — none of
    /// which is implemented here.
    fn apply_chatml(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }
}

/// If `value` has an `arguments` field that is a JSON-encoded string, replace
/// it in place with the parsed `serde_json::Value`. Used to convert the OpenAI
/// `tool_calls[*].function.arguments` string form to the dict form HF chat
/// templates iterate via `arguments|items`. A no-op when `arguments` is missing,
/// already structured, or a non-JSON string (left untouched in that case so the
/// chat template can decide how to handle it).
fn deserialize_arguments_in_place(value: &mut serde_json::Value) {
    let Some(args) = value.get_mut("arguments") else {
        return;
    };
    let Some(s) = args.as_str() else {
        return;
    };
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
        *args = parsed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid tokenizer JSON for offline tests (BPE with tiny vocab).
    fn minimal_tokenizer() -> KilnTokenizer {
        let json = br#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
            }
        }"#;
        KilnTokenizer {
            inner: Tokenizer::from_bytes(json.as_slice()).unwrap(),
            chat_template: None,
        }
    }

    #[test]
    fn test_chatml_single_message() {
        let tok = minimal_tokenizer();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            ..Default::default()
        }];
        let prompt = tok.apply_chat_template(&messages).unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_chatml_multi_turn() {
        let tok = minimal_tokenizer();
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "How are you?".to_string(),
                ..Default::default()
            },
        ];
        let prompt = tok.apply_chat_template(&messages).unwrap();
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_encode_decode_minimal() {
        let tok = minimal_tokenizer();
        // With our minimal BPE vocab, "ab" encodes to [0, 1]
        let ids = tok.encode("ab").unwrap();
        assert_eq!(ids, vec![0, 1]);
        // Decode roundtrip — BPE may insert spaces between tokens
        let decoded = tok.decode(&ids).unwrap();
        assert!(decoded.contains('a') && decoded.contains('b'));
    }

    #[test]
    fn test_vocab_size() {
        let tok = minimal_tokenizer();
        assert_eq!(tok.vocab_size(), 2);
    }

    #[test]
    #[ignore] // Requires network access to HuggingFace Hub
    fn test_from_pretrained() {
        let tok = KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B").unwrap();
        assert!(tok.vocab_size() > 100_000);
    }

    #[test]
    #[ignore] // Requires network access to HuggingFace Hub
    fn test_encode_decode_roundtrip_pretrained() {
        let tok = KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B").unwrap();
        let text = "Hello, world! This is a test of the tokenizer.";
        let ids = tok.encode(text).unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    #[ignore] // Requires network access to HuggingFace Hub
    fn test_eos_token_ids_pretrained() {
        let tok = KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B").unwrap();
        let eos = tok.eos_token_ids();
        assert!(!eos.is_empty(), "Should find at least one EOS token");
    }

    /// Tiny chat template that exercises the same surface real HF templates
    /// rely on: iterates `messages`, branches on `message.tool_calls` and on
    /// the top-level `tools` context entry. Used to prove that both halves of
    /// the tools wiring actually reach the Jinja renderer.
    const TOOLS_TEST_TEMPLATE: &str = "\
{%- if tools %}<tools>\
{%- for t in tools %}{{ t.name }};{% endfor %}\
</tools>{% endif -%}\
{%- for message in messages -%}\
[{{ message.role }}]\
{%- if message.tool_calls %}TOOLCALLS:\
{%- for tc in message.tool_calls %}{{ tc.name }};{% endfor %}{%- endif -%}\
{%- if message.tool_call_id %}TCID:{{ message.tool_call_id }};{% endif -%}\
{%- if message.name %}NAME:{{ message.name }};{% endif -%}\
{{ message.content }}\
{%- endfor -%}";

    fn tokenizer_with_template(template: &str) -> KilnTokenizer {
        let json = br#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
            }
        }"#;
        KilnTokenizer {
            inner: Tokenizer::from_bytes(json.as_slice()).unwrap(),
            chat_template: Some(template.to_string()),
        }
    }

    #[test]
    fn test_jinja_renders_tools_when_present() {
        let tok = tokenizer_with_template(TOOLS_TEST_TEMPLATE);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Run ls".to_string(),
            ..Default::default()
        }];
        let tools = vec![
            serde_json::json!({"name": "Bash", "description": "Run a command"}),
            serde_json::json!({"name": "Read", "description": "Read a file"}),
        ];
        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .unwrap();
        assert!(
            prompt.contains("<tools>"),
            "tools block missing from prompt: {prompt:?}"
        );
        assert!(prompt.contains("Bash;"), "tool name missing: {prompt:?}");
        assert!(prompt.contains("Read;"), "tool name missing: {prompt:?}");
    }

    #[test]
    fn test_jinja_omits_tools_block_when_none() {
        let tok = tokenizer_with_template(TOOLS_TEST_TEMPLATE);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            ..Default::default()
        }];
        let prompt = tok.apply_chat_template_with_tools(&messages, None).unwrap();
        assert!(
            !prompt.contains("<tools>"),
            "tools block leaked into prompt: {prompt:?}"
        );
        // Default apply_chat_template (no tools arg) must behave identically.
        let prompt_default = tok.apply_chat_template(&messages).unwrap();
        assert_eq!(prompt, prompt_default);
    }

    #[test]
    fn test_jinja_omits_tools_block_when_empty_slice() {
        let tok = tokenizer_with_template(TOOLS_TEST_TEMPLATE);
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
            ..Default::default()
        }];
        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&[]))
            .unwrap();
        assert!(
            !prompt.contains("<tools>"),
            "empty tools slice should not render tools block: {prompt:?}"
        );
    }

    #[test]
    fn test_jinja_renders_message_tool_calls_and_tool_responses() {
        let tok = tokenizer_with_template(TOOLS_TEST_TEMPLATE);
        let messages = vec![
            ChatMessage {
                role: "assistant".to_string(),
                content: "calling".to_string(),
                tool_calls: Some(vec![serde_json::json!({"name": "Bash"})]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "ok".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_42".to_string()),
                ..Default::default()
            },
        ];
        let prompt = tok.apply_chat_template(&messages).unwrap();
        assert!(
            prompt.contains("TOOLCALLS:Bash;"),
            "assistant tool_calls missing: {prompt:?}"
        );
        assert!(
            prompt.contains("TCID:call_42;"),
            "tool_call_id missing: {prompt:?}"
        );
        assert!(prompt.contains("NAME:Bash;"), "name missing: {prompt:?}");
    }

    #[test]
    fn test_chat_message_default_skips_optional_fields_in_serialize() {
        let m = ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
            ..Default::default()
        };
        let v = serde_json::to_value(&m).unwrap();
        // Optional fields must be absent when None so plain conversations
        // round-trip identically to the pre-tools shape.
        assert!(v.get("tool_calls").is_none());
        assert!(v.get("name").is_none());
        assert!(v.get("tool_call_id").is_none());
        assert_eq!(v["role"], "user");
        assert_eq!(v["content"], "hi");
    }

    /// `tojson` filter must be available on the minijinja env. Bundled HF
    /// chat templates (Qwen3.5, Llama-3, Mistral) call it unconditionally
    /// when rendering tools. If minijinja is depended on without the `json`
    /// feature, this filter is missing and every tools-bearing render
    /// errors with `unknown filter: tojson`. Regression test for the
    /// "structurally wired but feature-gated off" failure mode in PR #632.
    #[test]
    fn test_jinja_tojson_filter_is_available() {
        let template = "{{ data | tojson }}";
        let tok = tokenizer_with_template(template);
        let prompt = tok.apply_chat_template(&[]).unwrap();
        // `data` is unset in the template context, so `tojson` of `Undefined`
        // emits `null` — that's fine, what matters is the filter resolved.
        assert!(
            prompt.contains("null"),
            "tojson filter did not produce expected output: {prompt:?}"
        );

        // Also exercise the realistic shape: tools array forwarded into the
        // template, each tool rendered via `| tojson`. Mirrors line 50 of
        // Qwen3.5-4B's chat_template.jinja.
        let tools_template = "\
{%- for tool in tools -%}\
T:{{ tool | tojson }};\
{%- endfor -%}";
        let tok = tokenizer_with_template(tools_template);
        let tools = vec![serde_json::json!({"name": "Bash", "description": "Run a command"})];
        let prompt = tok
            .apply_chat_template_with_tools(&[], Some(&tools))
            .unwrap();
        assert!(
            prompt.contains("\"name\":\"Bash\""),
            "tojson did not serialize tool dict: {prompt:?}"
        );
    }

    /// Multi-turn conversation with prior assistant `tool_calls`. The Qwen3.5
    /// chat template iterates `tool_call.arguments|items`, which requires a
    /// dict. The OpenAI Chat Completions spec sends arguments as a
    /// JSON-encoded string. Without the in-place deserialization the render
    /// crashes with `cannot convert value into pairs`. Both the
    /// `{function: {...}}` and flat `{name, arguments}` envelopes must work.
    #[test]
    fn test_jinja_tool_call_arguments_string_is_iterable_as_dict() {
        // Mirrors line 120 of Qwen3.5-4B's chat_template.jinja:
        //   {%- for args_name, args_value in tool_call.arguments|items %}
        let template = "\
{%- for message in messages -%}\
[{{ message.role }}]\
{%- if message.tool_calls -%}\
{%- for tc in message.tool_calls -%}\
{%- set args = tc.function.arguments if tc.function is defined else tc.arguments -%}\
{%- for k, v in args | items -%}\
ARG:{{ k }}={{ v }};\
{%- endfor -%}\
{%- endfor -%}\
{%- endif -%}\
{{ message.content }}\
{%- endfor -%}";
        let tok = tokenizer_with_template(template);
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "list files".to_string(),
                ..Default::default()
            },
            // OpenAI envelope: tool_calls[*] = {id, type, function: {name, arguments: "<json string>"}}
            ChatMessage {
                role: "assistant".to_string(),
                content: "".to_string(),
                tool_calls: Some(vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": r#"{"command": "ls", "cwd": "/tmp"}"#
                    }
                })]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "a.txt".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "thanks".to_string(),
                ..Default::default()
            },
        ];
        let prompt = tok.apply_chat_template(&messages).expect(
            "render failed — likely arguments-as-string not deserialized to dict before |items",
        );
        assert!(
            prompt.contains("ARG:command=ls;"),
            "argument key/value missing: {prompt:?}"
        );
        assert!(
            prompt.contains("ARG:cwd=/tmp;"),
            "second argument missing: {prompt:?}"
        );

        // Also exercise the flat envelope (no nested function key).
        let messages_flat = vec![ChatMessage {
            role: "assistant".to_string(),
            content: "".to_string(),
            tool_calls: Some(vec![serde_json::json!({
                "name": "Read",
                "arguments": r#"{"path": "/etc/hosts"}"#
            })]),
            ..Default::default()
        }];
        let prompt = tok.apply_chat_template(&messages_flat).expect("flat-envelope render failed");
        assert!(
            prompt.contains("ARG:path=/etc/hosts;"),
            "flat-envelope argument missing: {prompt:?}"
        );
    }

    /// End-to-end render of Qwen3.5-4B's actual bundled `chat_template.jinja`
    /// against a realistic tools-bearing multi-turn conversation. This is the
    /// load-bearing fixture that pins both fixes from this PR (the `json`
    /// feature on minijinja and the `arguments`-string-to-dict
    /// deserialization). Either bug regressing causes this test to fail with
    /// `unknown filter: tojson` or `cannot convert value into pairs`.
    #[test]
    fn test_qwen35_4b_chat_template_renders_tools_and_tool_calls() {
        let template =
            include_str!("../test_fixtures/qwen35_4b_chat_template.jinja");
        let tok = tokenizer_with_template(template);

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }),
        ];

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a coding agent.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Show me what's in /etc.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "".to_string(),
                tool_calls: Some(vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": r#"{"command": "ls /etc"}"#
                    }
                })]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "hosts\nresolv.conf".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Now read /etc/hosts.".to_string(),
                ..Default::default()
            },
        ];

        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .expect("Qwen3.5-4B chat template rendered without error");

        // Tools block must appear (proves `tools | tojson` worked).
        assert!(prompt.contains("<tools>"), "tools block missing: prompt was {prompt:?}");
        assert!(prompt.contains("\"Bash\""), "Bash tool not serialized");
        assert!(prompt.contains("\"Read\""), "Read tool not serialized");

        // Past assistant tool_call must render in pi-XML form (proves the
        // arguments-as-dict path worked).
        assert!(
            prompt.contains("<function=Bash>"),
            "prior assistant tool_call did not render in Qwen pi-XML: {prompt:?}"
        );
        assert!(
            prompt.contains("<parameter=command>"),
            "tool_call argument not iterated as dict: {prompt:?}"
        );
        assert!(
            prompt.contains("ls /etc"),
            "argument value missing from rendered tool_call: {prompt:?}"
        );

        // Tool response wrapping must appear too.
        assert!(
            prompt.contains("<tool_response>"),
            "tool response not wrapped: {prompt:?}"
        );
    }

    /// End-to-end render of Meta's official Llama 3.1 8B Instruct
    /// `chat_template.jinja` against a tools-bearing multi-turn conversation.
    /// Companion to the Qwen3.5-4B render test — extends per-template
    /// fixture coverage to a second high-traffic model family with a
    /// distinct tool-call shape (`<|start_header_id|>` framing,
    /// `{"name": ..., "parameters": ...}` wire form via `arguments | tojson`).
    /// Closes kiln#657.
    #[test]
    fn test_llama31_8b_instruct_chat_template_renders_tools_and_tool_calls() {
        let template = include_str!(
            "../test_fixtures/llama31_8b_instruct_chat_template.jinja"
        );
        let tok = tokenizer_with_template(template);

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"}
                    },
                    "required": ["command"]
                }
            }
        })];

        // Llama 3.1's template raises if tool_calls.length != 1, so the
        // stress conversation has exactly one prior assistant tool_call.
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a coding agent.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Show me what's in /etc.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "".to_string(),
                tool_calls: Some(vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": r#"{"command": "ls /etc"}"#
                    }
                })]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "hosts\nresolv.conf".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Now read /etc/hosts.".to_string(),
                ..Default::default()
            },
        ];

        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .expect("Llama 3.1 chat template rendered without error");

        // Llama 3.1's distinctive header framing must appear.
        assert!(
            prompt.contains("<|start_header_id|>system<|end_header_id|>"),
            "Llama 3.1 system header missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<|eot_id|>"),
            "Llama 3.1 eot marker missing: {prompt:?}"
        );

        // Tools block must be serialized (template uses `t | tojson(indent=4)`,
        // proving minijinja's `json` feature is wired correctly).
        assert!(
            prompt.contains("\"name\": \"Bash\""),
            "Bash tool not serialized into prompt: {prompt:?}"
        );

        // Past assistant tool_call must render in Llama 3.1's wire form,
        // proving the arguments-as-dict path worked through `tojson`.
        assert!(
            prompt.contains("<|start_header_id|>assistant<|end_header_id|>"),
            "assistant header missing: {prompt:?}"
        );
        assert!(
            prompt.contains("\"parameters\""),
            "Llama 3.1 parameters key missing from rendered tool_call: {prompt:?}"
        );
        assert!(
            prompt.contains("\"command\""),
            "tool_call argument key missing — `arguments` likely not deserialized to dict: {prompt:?}"
        );
        assert!(
            prompt.contains("ls /etc"),
            "argument value missing from rendered tool_call: {prompt:?}"
        );

        // Tool response must be framed as ipython turn (Llama 3.1 maps
        // role=tool onto its ipython header).
        assert!(
            prompt.contains("<|start_header_id|>ipython<|end_header_id|>"),
            "Llama 3.1 ipython framing for tool response missing: {prompt:?}"
        );

        // Tool response content must round-trip into the rendered prompt.
        assert!(
            prompt.contains("hosts"),
            "tool response content missing from prompt: {prompt:?}"
        );
    }

    /// End-to-end render of Qwen2.5 7B Instruct's official
    /// `chat_template.jinja` against a tools-bearing multi-turn conversation.
    /// Distinct from the Qwen3.5-4B fixture: Qwen2.5 emits
    /// `<tool_call>{"name":...,"arguments":...}</tool_call>` JSON inline
    /// (vs. Qwen3.5's `<function=...><parameter=...>` pi-XML), so it
    /// exercises a different code path through `arguments | tojson`.
    /// Closes kiln#657.
    #[test]
    fn test_qwen25_7b_instruct_chat_template_renders_tools_and_tool_calls() {
        let template = include_str!(
            "../test_fixtures/qwen25_7b_instruct_chat_template.jinja"
        );
        let tok = tokenizer_with_template(template);

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }),
        ];

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a coding agent.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Show me what's in /etc.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "".to_string(),
                tool_calls: Some(vec![serde_json::json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": r#"{"command": "ls /etc"}"#
                    }
                })]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "hosts\nresolv.conf".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Now read /etc/hosts.".to_string(),
                ..Default::default()
            },
        ];

        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .expect("Qwen2.5 chat template rendered without error");

        // Tools block must be serialized via `tool | tojson` — proves
        // minijinja's `json` feature is wired and the template's `<tools>`
        // wrapper is firing.
        assert!(
            prompt.contains("<tools>"),
            "Qwen2.5 tools wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("\"Bash\""),
            "Bash tool not serialized: {prompt:?}"
        );
        assert!(
            prompt.contains("\"Read\""),
            "Read tool not serialized: {prompt:?}"
        );

        // Past assistant tool_call must render in Qwen2.5's `<tool_call>`
        // JSON wire form. Argument keys/values must round-trip — proves
        // `arguments | tojson` saw a dict, not a JSON-encoded string.
        assert!(
            prompt.contains("<tool_call>"),
            "Qwen2.5 tool_call wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("\"name\": \"Bash\""),
            "tool_call name missing from rendered prompt: {prompt:?}"
        );
        assert!(
            prompt.contains("\"command\""),
            "tool_call argument key missing — `arguments` likely not deserialized to dict: {prompt:?}"
        );
        assert!(
            prompt.contains("ls /etc"),
            "argument value missing from rendered tool_call: {prompt:?}"
        );
        assert!(
            prompt.contains("</tool_call>"),
            "Qwen2.5 tool_call closing tag missing: {prompt:?}"
        );

        // Tool response must be wrapped in `<tool_response>` framing.
        assert!(
            prompt.contains("<tool_response>"),
            "Qwen2.5 tool_response wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("hosts"),
            "tool response content missing from prompt: {prompt:?}"
        );
    }

    /// End-to-end render of Mistral 7B Instruct v0.3's official
    /// `chat_template.jinja` against a tools-bearing multi-turn conversation.
    /// Distinct from the Llama 3.1 / Qwen2.5 fixtures: Mistral uses
    /// `[INST]` / `[/INST]` user framing with `[AVAILABLE_TOOLS]` /
    /// `[/AVAILABLE_TOOLS]` for the tool list, `[TOOL_CALLS]` for assistant
    /// tool-call shape (JSON list with `tool_call.function|tojson` then a
    /// post-fixed `"id": "..."` field), and `[TOOL_RESULTS]` /
    /// `[/TOOL_RESULTS]` for tool responses. The template enforces a strict
    /// 9-character tool_call.id constraint and inlines the system message
    /// into the LAST user `[INST]` block (rather than emitting a separate
    /// system turn). Closes the Mistral coverage gap deferred from #658.
    #[test]
    fn test_mistral_7b_instruct_v03_chat_template_renders_tools_and_tool_calls() {
        let template = include_str!(
            "../test_fixtures/mistral_7b_instruct_v03_chat_template.jinja"
        );
        let tok = tokenizer_with_template(template);

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }),
        ];

        // Mistral's template enforces strict user/assistant alternation
        // across all NON-tool / NON-tool_calls messages (tool messages and
        // assistant-with-tool_calls turns are invisible to the alternation
        // counter). We need a final user turn so the system message inlines
        // into a `[INST]` block via `loop.last and system_message is defined`.
        // Tool call IDs must be exactly 9 alphanumeric characters or the
        // template raises.
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a coding agent.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Show me what's in /etc.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "".to_string(),
                tool_calls: Some(vec![serde_json::json!({
                    "id": "call12345",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": r#"{"command": "ls /etc"}"#
                    }
                })]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "hosts resolv.conf".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call12345".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Found two files.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Now read /etc/hosts.".to_string(),
                ..Default::default()
            },
        ];

        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .expect("Mistral v0.3 chat template rendered without error");

        // Mistral's distinctive `[INST]` / `[/INST]` user framing must appear.
        assert!(
            prompt.contains("[INST] "),
            "Mistral [INST] user framing missing: {prompt:?}"
        );
        assert!(
            prompt.contains("[/INST]"),
            "Mistral [/INST] closing missing: {prompt:?}"
        );

        // System message inlines into the LAST user `[INST]` block (Mistral
        // does not emit a separate system turn). Proves the
        // `loop.last and system_message is defined` branch fires.
        assert!(
            prompt.contains("You are a coding agent."),
            "Mistral system_message not inlined into final [INST]: {prompt:?}"
        );
        assert!(
            prompt.contains("Now read /etc/hosts."),
            "Mistral final user content missing: {prompt:?}"
        );

        // Tools must be serialized via `[AVAILABLE_TOOLS]` framing on the
        // LAST user message — proves the per-tool key/value walk and `tojson`
        // both fired (parameters is a dict, not a string).
        assert!(
            prompt.contains("[AVAILABLE_TOOLS]"),
            "Mistral [AVAILABLE_TOOLS] wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("[/AVAILABLE_TOOLS]"),
            "Mistral [/AVAILABLE_TOOLS] closing missing: {prompt:?}"
        );
        assert!(
            prompt.contains("\"name\": \"Bash\""),
            "Bash tool not serialized into prompt: {prompt:?}"
        );
        assert!(
            prompt.contains("\"name\": \"Read\""),
            "Read tool not serialized into prompt: {prompt:?}"
        );
        assert!(
            prompt.contains("\"parameters\""),
            "Mistral parameters key missing — `tojson` likely failed: {prompt:?}"
        );

        // Past assistant tool_call must render in Mistral's `[TOOL_CALLS]`
        // JSON-list wire form. Argument keys/values must round-trip — proves
        // `tool_call.function|tojson` saw `arguments` as a dict, not a
        // JSON-encoded string (regression guard for kiln#632 / #653).
        assert!(
            prompt.contains("[TOOL_CALLS]"),
            "Mistral [TOOL_CALLS] wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("\"command\""),
            "tool_call argument key missing — `arguments` likely not deserialized to dict: {prompt:?}"
        );
        assert!(
            prompt.contains("ls /etc"),
            "argument value missing from rendered tool_call: {prompt:?}"
        );
        assert!(
            prompt.contains("\"id\": \"call12345\""),
            "Mistral tool_call.id not appended to JSON object: {prompt:?}"
        );

        // Tool response must be wrapped in `[TOOL_RESULTS]` framing with the
        // matching `call_id` round-tripping the prior assistant tool_call.id.
        assert!(
            prompt.contains("[TOOL_RESULTS]"),
            "Mistral [TOOL_RESULTS] wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("[/TOOL_RESULTS]"),
            "Mistral [/TOOL_RESULTS] closing missing: {prompt:?}"
        );
        assert!(
            prompt.contains("\"call_id\": \"call12345\""),
            "Mistral tool response call_id missing or mismatched: {prompt:?}"
        );
        assert!(
            prompt.contains("hosts"),
            "tool response content missing from prompt: {prompt:?}"
        );

        // Inter-turn assistant text content (between tool result and final
        // user) must round-trip — proves the plain assistant content branch
        // (` content + eos_token`) is hit when `tool_calls` is absent.
        assert!(
            prompt.contains("Found two files."),
            "Inter-turn assistant text content missing: {prompt:?}"
        );
    }

    /// End-to-end render of DeepSeek V3's official `chat_template`
    /// (extracted from `tokenizer_config.json`) against a tools-bearing
    /// multi-turn conversation. Distinct from the Llama 3.1 / Qwen2.5 /
    /// Mistral fixtures: DeepSeek uses **full-width unicode bracket**
    /// framing (`<｜User｜>`, `<｜Assistant｜>`, `<｜tool▁calls▁begin｜>`,
    /// `<｜tool▁call▁begin｜>`, `<｜tool▁sep｜>`,
    /// `<｜tool▁outputs▁begin｜>`, `<｜tool▁output▁begin｜>`) — exercising
    /// minijinja unicode handling in a way none of the existing fixtures
    /// do. It branches on `message['content'] is none` (NOT
    /// `tool_calls is defined` as Mistral does, NOR `tool_calls.length` as
    /// Llama 3.1 does) to decide whether to emit the tool_calls block,
    /// which means assistant tool_call turns require `content: null` on the
    /// wire — the in-pipeline empty-content-with-tool_calls -> null
    /// normalization handles the gap. The template aggregates ALL system
    /// messages into a single `ns.system_prompt` emitted ONCE immediately
    /// after `bos_token` (no per-turn role framing), and tool definitions
    /// (`tools` parameter) are NOT serialized into the prompt at all
    /// (DeepSeek expects them via system message). Closes the DeepSeek
    /// coverage gap on top of #658 / #660.
    #[test]
    fn test_deepseek_v3_chat_template_renders_tools_and_tool_calls() {
        let template = include_str!(
            "../test_fixtures/deepseek_v3_chat_template.jinja"
        );
        let tok = tokenizer_with_template(template);

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }),
        ];

        // DeepSeek's template only emits `<｜tool▁calls▁end｜>` from the
        // SECOND tool_call iteration onwards (the close tag lives inside
        // the `{%- else %}` branch — a quirk of the upstream template).
        // Pass two tool calls so both `<｜tool▁call▁begin｜>` paths fire
        // and the closing `<｜tool▁calls▁end｜>` token is exercised. Two
        // tool messages follow (one per call), then a plain assistant
        // text turn (which clears `ns.is_tool` via the
        // `<｜tool▁outputs▁end｜>` branch), then a final user turn so
        // `add_generation_prompt=true` appends a trailing `<｜Assistant｜>`.
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a coding agent.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Show me what's in /etc and read /etc/hosts."
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                // OpenAI spec sends `null` content alongside tool_calls;
                // DeepSeek's template branches on `content is none`. We
                // pass `""` here and rely on the in-pipeline
                // empty-content-with-tool_calls -> null normalization to
                // flip the template branch correctly.
                content: "".to_string(),
                tool_calls: Some(vec![
                    serde_json::json!({
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": r#"{"command": "ls /etc"}"#
                        }
                    }),
                    serde_json::json!({
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": r#"{"path": "/etc/hosts"}"#
                        }
                    }),
                ]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "hosts resolv.conf".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "127.0.0.1 localhost".to_string(),
                name: Some("Read".to_string()),
                tool_call_id: Some("call_2".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Found two files; localhost loopback in hosts."
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Now check resolv.conf.".to_string(),
                ..Default::default()
            },
        ];

        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .expect("DeepSeek V3 chat template rendered without error");

        // DeepSeek's distinctive full-width unicode bracket framing must
        // appear. No `[INST]`, no `<|start_header_id|>` — these tokens are
        // entirely distinct from the Llama / Qwen / Mistral families.
        assert!(
            prompt.contains("<｜User｜>"),
            "DeepSeek <｜User｜> framing missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜Assistant｜>"),
            "DeepSeek <｜Assistant｜> framing missing: {prompt:?}"
        );

        // System message is aggregated into `ns.system_prompt` and emitted
        // ONCE immediately after `bos_token`, with no per-turn role framing.
        assert!(
            prompt.contains("You are a coding agent."),
            "DeepSeek system_prompt aggregation failed: {prompt:?}"
        );

        // User content roundtrip — both turns must appear.
        assert!(
            prompt.contains("Show me what's in /etc and read /etc/hosts."),
            "DeepSeek first user content missing: {prompt:?}"
        );
        assert!(
            prompt.contains("Now check resolv.conf."),
            "DeepSeek final user content missing: {prompt:?}"
        );

        // Past assistant tool_calls must render in DeepSeek's distinctive
        // `<｜tool▁calls▁begin｜>` ... `<｜tool▁calls▁end｜>` framing with
        // per-call `<｜tool▁call▁begin｜>` ... `<｜tool▁call▁end｜>`
        // wrappers. The `<｜tool▁calls▁end｜>` close tag only fires from
        // the second iteration onwards (template quirk) — passing 2
        // tool_calls exercises it.
        assert!(
            prompt.contains("<｜tool▁calls▁begin｜>"),
            "DeepSeek <｜tool▁calls▁begin｜> framing missing — likely the assistant tool_calls branch (requires `content is none`) was skipped: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜tool▁calls▁end｜>"),
            "DeepSeek <｜tool▁calls▁end｜> close tag missing — only emitted from 2nd tool_call iteration: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜tool▁call▁begin｜>"),
            "DeepSeek per-call <｜tool▁call▁begin｜> wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜tool▁call▁end｜>"),
            "DeepSeek per-call <｜tool▁call▁end｜> wrapper missing: {prompt:?}"
        );

        // `<｜tool▁sep｜>` separates tool type and function name within
        // each call — distinctive vs. Mistral (no per-call separator) and
        // Llama (`<|python_tag|>`). Both function names must round-trip
        // into the rendered tool_call body.
        assert!(
            prompt.contains("<｜tool▁sep｜>"),
            "DeepSeek <｜tool▁sep｜> missing: {prompt:?}"
        );
        assert!(
            prompt.contains("Bash"),
            "Bash tool name missing from rendered tool_call: {prompt:?}"
        );
        assert!(
            prompt.contains("Read"),
            "Read tool name missing from rendered tool_call: {prompt:?}"
        );

        // Argument values must round-trip into the rendered tool_call body
        // (template embeds them inside ` ```json ... ``` ` fences).
        assert!(
            prompt.contains("ls /etc"),
            "Bash tool_call argument value missing: {prompt:?}"
        );
        assert!(
            prompt.contains("/etc/hosts"),
            "Read tool_call argument value missing: {prompt:?}"
        );
        assert!(
            prompt.contains("```json"),
            "DeepSeek ```json argument fence missing: {prompt:?}"
        );

        // Tool responses must be wrapped in `<｜tool▁outputs▁begin｜>` ...
        // `<｜tool▁outputs▁end｜>` framing, with per-output
        // `<｜tool▁output▁begin｜>` ... `<｜tool▁output▁end｜>` wrappers.
        // The outputs-end tag is emitted by the next non-tool message
        // (here: the inter-turn assistant text) via the `if ns.is_tool`
        // branch.
        assert!(
            prompt.contains("<｜tool▁outputs▁begin｜>"),
            "DeepSeek <｜tool▁outputs▁begin｜> framing missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜tool▁outputs▁end｜>"),
            "DeepSeek <｜tool▁outputs▁end｜> close tag missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜tool▁output▁begin｜>"),
            "DeepSeek per-output <｜tool▁output▁begin｜> wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<｜tool▁output▁end｜>"),
            "DeepSeek per-output <｜tool▁output▁end｜> wrapper missing: {prompt:?}"
        );
        assert!(
            prompt.contains("hosts resolv.conf"),
            "Bash tool response content missing: {prompt:?}"
        );
        assert!(
            prompt.contains("127.0.0.1 localhost"),
            "Read tool response content missing: {prompt:?}"
        );

        // Inter-turn assistant text content (between tool result and
        // final user) must round-trip — proves the plain assistant
        // content branch (`content is not none` and not `ns.is_tool`)
        // is hit when the intervening tool messages already cleared
        // `is_tool` via the `<｜tool▁outputs▁end｜>` path.
        assert!(
            prompt.contains("Found two files"),
            "Inter-turn assistant text content missing: {prompt:?}"
        );

        // `add_generation_prompt=true` (default in apply_chat_template_*)
        // plus a final user turn must append a trailing `<｜Assistant｜>`
        // for the model to continue from. Use `ends_with` because the
        // `<｜Assistant｜>` token also appears mid-prompt inside the
        // tool_calls block — only the trailing instance is the
        // generation marker.
        assert!(
            prompt.trim_end().ends_with("<｜Assistant｜>"),
            "DeepSeek trailing <｜Assistant｜> generation marker missing: {prompt:?}"
        );
    }

    /// End-to-end render of NousResearch/Hermes-3-Llama-3.1-8B's official
    /// `tool_use` chat template against a tools-bearing multi-turn
    /// conversation. Hermes-3 is a distinct template lineage from the five
    /// fixtures already vendored (Qwen3.5-4B, Llama 3.1, Qwen2.5, Mistral
    /// 7B v0.3, DeepSeek V3): it pre-injects a hardcoded `<|im_start|>system`
    /// turn that frames the tools list inside `<tools></tools>` XML wrappers
    /// and instructs the model to emit `<tool_call>{json}</tool_call>` inline
    /// JSON tags (NOT Mistral's `[TOOL_CALLS]`, NOT Llama's
    /// `<|python_tag|>`, NOT DeepSeek's full-width unicode bracket framing).
    /// Tool responses are wrapped in `<tool_response></tool_response>`
    /// blocks. The template's distinctive trick: consecutive `tool` messages
    /// pack into a SINGLE `<|im_start|>tool ... <|im_end|>` turn — the
    /// `<|im_start|>tool` header only re-emits when
    /// `loop.previtem.role != "tool"`, and the closing `<|im_end|>` is
    /// driven by `loop.nextitem.role != "tool"`. This exercises minijinja's
    /// `loop.previtem` / `loop.nextitem` accessors in a way none of the
    /// previously-vendored fixtures do (Hermes is the only one that depends
    /// on both for correct rendering). Closes the Hermes / inline-XML
    /// tool-call coverage gap on top of #658 / #660 / #661.
    #[test]
    fn test_hermes3_llama31_chat_template_renders_tools_and_tool_calls() {
        let template = include_str!(
            "../test_fixtures/hermes3_llama31_chat_template.jinja"
        );
        let tok = tokenizer_with_template(template);

        let tools = vec![
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"}
                        },
                        "required": ["command"]
                    }
                }
            }),
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"}
                        },
                        "required": ["path"]
                    }
                }
            }),
        ];

        // Two tool_calls in a single assistant turn (both must render as
        // sibling `<tool_call>...</tool_call>` blocks within ONE
        // `<|im_start|>assistant ... <|im_end|>` envelope) followed by two
        // tool messages (which must pack into ONE `<|im_start|>tool ...
        // <|im_end|>` turn via the `loop.previtem.role` / `loop.nextitem.role`
        // accessors). A plain assistant text turn then a final user turn
        // exercises the inter-turn assistant path AND drives
        // `add_generation_prompt=true` to append the trailing
        // `<|im_start|>assistant\n` generation marker.
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a coding agent.".to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Show me what's in /etc and read /etc/hosts."
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                // Hermes-3 branches the assistant emit on
                // `tool_calls is not defined` (NOT `content is none` like
                // DeepSeek V3), so the empty-content normalization isn't
                // strictly required here — but the in-pipeline normalization
                // still flips `""` to `null`, and the template must handle
                // both shapes. Pass `""` to confirm the assistant tool_calls
                // branch fires regardless of normalization state.
                content: "".to_string(),
                tool_calls: Some(vec![
                    serde_json::json!({
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": r#"{"command": "ls /etc"}"#
                        }
                    }),
                    serde_json::json!({
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": r#"{"path": "/etc/hosts"}"#
                        }
                    }),
                ]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "hosts resolv.conf".to_string(),
                name: Some("Bash".to_string()),
                tool_call_id: Some("call_1".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".to_string(),
                content: "127.0.0.1 localhost".to_string(),
                name: Some("Read".to_string()),
                tool_call_id: Some("call_2".to_string()),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Found two files; localhost loopback in hosts."
                    .to_string(),
                ..Default::default()
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Now check resolv.conf.".to_string(),
                ..Default::default()
            },
        ];

        let prompt = tok
            .apply_chat_template_with_tools(&messages, Some(&tools))
            .expect("Hermes-3 chat template rendered without error");

        // Hermes-3 pre-injects a hardcoded `<|im_start|>system` block at the
        // top of the prompt that frames the tools inside `<tools></tools>`
        // and embeds the "function calling AI model" prelude. None of the
        // other vendored fixtures (Qwen3.5, Llama 3.1, Qwen2.5, Mistral,
        // DeepSeek V3) emit this prelude — it's the load-bearing signal that
        // we're rendering the `tool_use` template variant correctly.
        assert!(
            prompt.contains("function calling AI model"),
            "Hermes-3 hardcoded function-calling system prelude missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<tools>"),
            "Hermes-3 <tools> wrapper missing — tool list block did not emit: {prompt:?}"
        );
        assert!(
            prompt.contains("</tools>"),
            "Hermes-3 </tools> closing wrapper missing: {prompt:?}"
        );

        // Both tool definitions must serialize into the `<tools>` block —
        // proves the `for tool in tools` loop and `tool.parameters|tojson`
        // path both fired (regression guard for the kiln#632 minijinja
        // `json` feature gap).
        assert!(
            prompt.contains("\"name\": \"Bash\""),
            "Bash tool not serialized into Hermes-3 tools block — tojson likely failed: {prompt:?}"
        );
        assert!(
            prompt.contains("\"name\": \"Read\""),
            "Read tool not serialized into Hermes-3 tools block: {prompt:?}"
        );

        // ChatML framing must round-trip user/system/assistant turns. The
        // user-supplied system message renders as a SECOND `<|im_start|>system`
        // turn (in addition to the hardcoded prelude — Hermes does not
        // aggregate them).
        assert!(
            prompt.contains("<|im_start|>system\nYou are a coding agent.<|im_end|>"),
            "Hermes-3 user-supplied system turn missing or misframed: {prompt:?}"
        );
        assert!(
            prompt.contains("<|im_start|>user\nShow me what's in /etc and read /etc/hosts.<|im_end|>"),
            "Hermes-3 first user turn missing or misframed: {prompt:?}"
        );
        assert!(
            prompt.contains("<|im_start|>user\nNow check resolv.conf.<|im_end|>"),
            "Hermes-3 final user turn missing or misframed: {prompt:?}"
        );

        // Past assistant tool_calls must render as sibling
        // `<tool_call>{json}</tool_call>` blocks within a SINGLE
        // `<|im_start|>assistant ... <|im_end|>` envelope. Both inline JSON
        // tags must appear, and both function names + argument values must
        // round-trip — proves `tool_call.arguments|tojson` saw a dict (not a
        // JSON-encoded string), which is the kiln#632 / #653 regression
        // class this fixture pins.
        assert!(
            prompt.contains("<tool_call>"),
            "Hermes-3 inline <tool_call> tag missing — assistant tool_calls branch did not fire: {prompt:?}"
        );
        assert!(
            prompt.contains("</tool_call>"),
            "Hermes-3 inline </tool_call> close tag missing: {prompt:?}"
        );
        assert!(
            prompt.contains(r#""name": "Bash""#),
            "Bash tool_call name missing from rendered tool_call body: {prompt:?}"
        );
        assert!(
            prompt.contains(r#""name": "Read""#),
            "Read tool_call name missing from rendered tool_call body: {prompt:?}"
        );
        assert!(
            prompt.contains("ls /etc"),
            "Bash tool_call argument value missing — `arguments` likely not deserialized to dict: {prompt:?}"
        );
        assert!(
            prompt.contains("/etc/hosts"),
            "Read tool_call argument value missing: {prompt:?}"
        );

        // Tool responses must wrap in `<tool_response></tool_response>`
        // blocks INSIDE a single `<|im_start|>tool ... <|im_end|>` turn.
        // Hermes-3's distinctive trick: the `<|im_start|>tool` header is
        // gated on `loop.previtem and loop.previtem.role != "tool"`, so two
        // consecutive tool messages emit only ONE header (not two separate
        // tool turns). Likewise, the closing `<|im_end|>` is driven by
        // `loop.nextitem.role != "tool"`. Both content strings must
        // round-trip into their respective `<tool_response>` bodies.
        assert!(
            prompt.contains("<|im_start|>tool\n<tool_response>"),
            "Hermes-3 single `<|im_start|>tool` header for first tool message missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<tool_response>\nhosts resolv.conf"),
            "Hermes-3 first tool_response content missing: {prompt:?}"
        );
        assert!(
            prompt.contains("<tool_response>\n127.0.0.1 localhost"),
            "Hermes-3 second tool_response content missing — likely the second tool message emitted its own `<|im_start|>tool` instead of packing into the prior turn: {prompt:?}"
        );
        // The two tool_response blocks must share ONE `<|im_start|>tool`
        // turn, so the prompt must contain exactly one such header — count
        // it directly. Two separate occurrences would mean
        // `loop.previtem.role` is broken under minijinja.
        assert_eq!(
            prompt.matches("<|im_start|>tool\n").count(),
            1,
            "Hermes-3 emitted multiple `<|im_start|>tool` headers — `loop.previtem.role` likely not honored: {prompt:?}"
        );

        // Inter-turn assistant text content (between tool result and final
        // user) must round-trip — proves the plain assistant content branch
        // (`message.role == "assistant" and message.tool_calls is not
        // defined`) fires when the prior tool turn already closed.
        assert!(
            prompt.contains("<|im_start|>assistant\nFound two files; localhost loopback in hosts.<|im_end|>"),
            "Hermes-3 inter-turn assistant text content missing or misframed: {prompt:?}"
        );

        // `add_generation_prompt=true` (default in apply_chat_template_*)
        // must append a trailing `<|im_start|>assistant\n` for the model to
        // continue from. Use `ends_with` because `<|im_start|>assistant`
        // also appears mid-prompt inside the assistant tool_calls turn —
        // only the trailing instance is the generation marker.
        assert!(
            prompt.trim_end().ends_with("<|im_start|>assistant"),
            "Hermes-3 trailing `<|im_start|>assistant` generation marker missing: {prompt:?}"
        );
    }

    /// `deserialize_arguments_in_place` is a no-op when arguments is already
    /// a dict (caller pre-parsed) or missing entirely, and silently leaves
    /// non-JSON strings untouched (the template can decide what to do).
    #[test]
    fn test_deserialize_arguments_in_place_handles_edge_cases() {
        // Already-parsed dict: untouched.
        let mut v = serde_json::json!({"arguments": {"a": 1}});
        deserialize_arguments_in_place(&mut v);
        assert_eq!(v["arguments"]["a"], 1);

        // No arguments field: untouched.
        let mut v = serde_json::json!({"name": "Bash"});
        deserialize_arguments_in_place(&mut v);
        assert_eq!(v.get("arguments"), None);

        // Non-JSON string: left as-is so template logic sees it.
        let mut v = serde_json::json!({"arguments": "not-json-at-all"});
        deserialize_arguments_in_place(&mut v);
        assert_eq!(v["arguments"], "not-json-at-all");

        // Empty object as JSON string: parses to empty dict.
        let mut v = serde_json::json!({"arguments": "{}"});
        deserialize_arguments_in_place(&mut v);
        assert_eq!(v["arguments"], serde_json::json!({}));
    }
}
