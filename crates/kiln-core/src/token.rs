/// A token ID in the model's vocabulary.
pub type TokenId = u32;

/// Special token IDs — populated from tokenizer config at load time.
#[derive(Debug, Clone, Default)]
pub struct SpecialTokens {
    pub bos_token_id: Option<TokenId>,
    pub eos_token_ids: Vec<TokenId>,
    pub pad_token_id: Option<TokenId>,
}
