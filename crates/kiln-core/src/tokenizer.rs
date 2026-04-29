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

        tmpl.render(minijinja::context! {
            messages => messages,
            tools => tools_value,
            tool_choice => tool_choice_value,
            add_generation_prompt => true,
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
}
