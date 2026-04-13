use crate::sampling::SamplingParams;
use crate::token::TokenId;
use uuid::Uuid;

/// Unique identifier for an inference request.
pub type RequestId = Uuid;

/// The lifecycle state of an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting in queue, not yet scheduled.
    Waiting,
    /// Prefill in progress (may be chunked across multiple iterations).
    Prefilling {
        /// How many prompt tokens have been processed so far.
        tokens_processed: usize,
    },
    /// Actively generating tokens.
    Decoding,
    /// Generation complete (hit EOS, max_tokens, or stop sequence).
    Complete,
    /// Cancelled by the client.
    Cancelled,
}

/// A single inference request tracked by the scheduler.
#[derive(Debug)]
pub struct Request {
    pub id: RequestId,
    pub prompt_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,
    pub sampling_params: SamplingParams,
    pub state: RequestState,

    /// Physical block IDs allocated for this request's KV cache.
    pub block_ids: Vec<u32>,

    /// Timestamp when the request was received.
    pub created_at: std::time::Instant,

    /// Which LoRA adapter (if any) this request uses.
    /// None = base model only.
    pub adapter_id: Option<String>,
}

impl Request {
    pub fn new(
        prompt_tokens: Vec<TokenId>,
        sampling_params: SamplingParams,
        adapter_id: Option<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_params,
            state: RequestState::Waiting,
            block_ids: Vec::new(),
            created_at: std::time::Instant::now(),
            adapter_id,
        }
    }

    /// Total tokens consumed so far (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// How many prompt tokens still need prefilling.
    pub fn remaining_prefill(&self) -> usize {
        match self.state {
            RequestState::Prefilling { tokens_processed } => {
                self.prompt_tokens.len().saturating_sub(tokens_processed)
            }
            RequestState::Waiting => self.prompt_tokens.len(),
            _ => 0,
        }
    }

    /// How many KV cache blocks this request currently needs.
    pub fn blocks_needed(&self, block_size: usize) -> usize {
        (self.total_tokens() + block_size - 1) / block_size
    }
}
