use anyhow::Result;
use kiln_core::config::ModelConfig;
use kiln_core::request::RequestId;
use kiln_core::token::TokenId;

/// Output from one iteration of the engine.
#[derive(Debug)]
pub struct StepOutput {
    /// For each request in the batch: (request_id, new_token, finished).
    pub results: Vec<(RequestId, Option<TokenId>, bool)>,
}

/// Trait for the model execution engine.
///
/// The engine handles the actual forward pass — loading weights, running attention,
/// sampling tokens. The scheduler decides WHAT to run; the engine decides HOW.
///
/// Phase 1: mock engine that returns random tokens (for testing the scheduler + server).
/// Phase 2: real Qwen3 inference via candle or CUDA kernels.
pub trait Engine: Send + Sync {
    /// Run one forward pass for the given batch.
    ///
    /// `batch` contains (request_id, token_ids, slot_mappings) for each request
    /// in the current iteration. The engine computes attention, applies LoRA deltas,
    /// samples tokens, and returns results.
    fn step(&self, batch: &BatchInput) -> Result<StepOutput>;

    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;

    /// Load a LoRA adapter from the given path. Returns an adapter ID.
    fn load_adapter(&self, path: &str) -> Result<String>;

    /// Swap the active LoRA adapter. Atomic at iteration boundary.
    fn activate_adapter(&self, adapter_id: Option<&str>) -> Result<()>;
}

/// Input batch for one forward pass.
#[derive(Debug)]
pub struct BatchInput {
    /// Flat array of token IDs across all sequences in the batch.
    pub token_ids: Vec<TokenId>,
    /// Cumulative sequence lengths for ragged batching.
    /// seqlens[i] = number of tokens in sequence i.
    pub seqlens: Vec<usize>,
    /// Per-token physical slot index for KV cache writes.
    pub slot_mapping: Vec<usize>,
    /// Per-sequence block tables for KV cache reads during attention.
    pub block_tables: Vec<Vec<u32>>,
    /// Which sequences are prefilling (true) vs decoding (false).
    pub is_prefill: Vec<bool>,
    /// Request IDs in the same order as seqlens.
    pub request_ids: Vec<RequestId>,
}

/// Mock engine for testing — generates random tokens.
pub struct MockEngine {
    config: ModelConfig,
}

impl MockEngine {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl Engine for MockEngine {
    fn step(&self, batch: &BatchInput) -> Result<StepOutput> {
        let mut results = Vec::new();
        for (i, &req_id) in batch.request_ids.iter().enumerate() {
            // For prefill-only passes, don't emit a token yet (unless this is the last chunk)
            // For decode, always emit a token
            if batch.is_prefill[i] {
                // Emit a token at end of prefill to transition to decode
                results.push((req_id, Some(42), false));
            } else {
                // Random-ish decode token, never EOS for now
                let token = ((req_id.as_u128() % 1000) as u32) + 100;
                results.push((req_id, Some(token), false));
            }
        }
        Ok(StepOutput { results })
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load_adapter(&self, _path: &str) -> Result<String> {
        Ok("mock-adapter".to_string())
    }

    fn activate_adapter(&self, _adapter_id: Option<&str>) -> Result<()> {
        Ok(())
    }
}
