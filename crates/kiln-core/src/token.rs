/// A token ID in the model's vocabulary.
pub type TokenId = u32;

/// Special token IDs — populated from tokenizer config at load time.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<TokenId>,
    pub eos_token_ids: Vec<TokenId>,
    pub pad_token_id: Option<TokenId>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token_id: None,
            eos_token_ids: vec![],
            pad_token_id: None,
        }
    }
}
