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
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
        match &self.chat_template {
            Some(template) => self.render_jinja_template(template, messages),
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
    ) -> Result<String, TokenizerError> {
        let mut env = minijinja::Environment::new();
        env.add_template("chat", template)
            .map_err(|e| TokenizerError::ChatTemplate(e.to_string()))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| TokenizerError::ChatTemplate(e.to_string()))?;

        tmpl.render(minijinja::context! {
            messages => messages,
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
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "How are you?".to_string(),
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
}
