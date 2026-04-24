//! End-to-end autoregressive text generation pipeline.
//!
//! Wires together tokenizer, model weights, forward pass, and sampling into
//! a `ModelRunner` that accepts text prompts and produces text output.

use anyhow::{Context, Result};
use candle_core::DType;
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;

use crate::backend::{self, BackendRuntime};
use crate::cuda_graph::CudaGraphRunner;
use crate::forward::{
    GpuWeights, LinearAttentionState, model_forward, model_forward_paged,
    model_forward_paged_last_token, model_forward_paged_last_token_with_last_hidden,
    model_forward_paged_next_token_greedy, model_forward_paged_streaming,
    model_forward_paged_streaming_last_token_with_last_hidden, streaming_prefill_enabled_for,
};
use crate::kv_cache::KvCache;
use crate::lora_loader::LoraWeights;
use crate::paged_kv_cache::PagedKvCache;
use crate::sampling::{greedy_sample, sample_with_params};
use crate::speculative::{
    SpeculativeConfig, speculative_decode_step, speculative_decode_step_paged_greedy,
    speculative_mtp_decode_step,
};

use kiln_core::block::{BlockManager, BlockTable};

/// Holds loaded model weights and tokenizer, provides text generation.
pub struct ModelRunner {
    pub weights: GpuWeights,
    pub tokenizer: KilnTokenizer,
    pub config: ModelConfig,
    /// EOS token IDs cached from the tokenizer.
    eos_token_ids: Vec<TokenId>,
    /// Currently active LoRA adapter weights (None = base model only).
    active_lora: Option<LoraWeights>,
    /// CUDA graph runner for accelerated decode steps.
    /// Uses Mutex for interior mutability (graph state changes during &self generation).
    cuda_graph: Mutex<CudaGraphRunner>,
    backend: Arc<dyn BackendRuntime>,
}

/// Output from a generation call.
#[derive(Debug)]
pub struct GenerationOutput {
    /// The generated text (not including the prompt).
    pub text: String,
    /// The generated token IDs (not including prompt tokens).
    pub token_ids: Vec<TokenId>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// Output from a native MTP speculative generation call.
///
/// Carries everything [`GenerationOutput`] does plus the per-call MTP draft
/// accept/reject counters used by bench reporting to compute α (acceptance
/// rate = `draft_accepted_count / total_draft_attempts`).
#[derive(Debug)]
pub struct MtpGenerationOutput {
    /// The generated text (not including the prompt).
    pub text: String,
    /// The generated token IDs (not including prompt tokens).
    pub token_ids: Vec<TokenId>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// How many MTP draft tokens were accepted across the decode loop.
    pub draft_accepted_count: usize,
    /// How many MTP draft attempts were made (one per [`speculative_mtp_decode_step`] call).
    pub total_draft_attempts: usize,
}

/// A single token emitted during streaming generation.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The generated token ID.
    pub token_id: TokenId,
    /// The decoded text for this token.
    pub text: String,
}

/// Final event sent when streaming generation completes.
#[derive(Debug, Clone)]
pub struct StreamDone {
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Total number of generated tokens.
    pub completion_tokens: usize,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new token was generated.
    Token(StreamToken),
    /// Generation is complete.
    Done(StreamDone),
}

enum StreamTokenDisposition {
    Continue,
    Finished(FinishReason),
    ReceiverDropped,
}

struct SharedBlockReservation<'a> {
    block_manager: &'a Mutex<BlockManager>,
    block_ids: Vec<u32>,
}

impl Drop for SharedBlockReservation<'_> {
    fn drop(&mut self) {
        if self.block_ids.is_empty() {
            return;
        }
        match self.block_manager.lock() {
            Ok(mut guard) => guard.free_all(&self.block_ids),
            Err(e) => tracing::error!("failed to lock block manager to free blocks: {e}"),
        }
    }
}

fn lock_block_manager(
    block_manager: &Mutex<BlockManager>,
) -> Result<std::sync::MutexGuard<'_, BlockManager>> {
    block_manager
        .lock()
        .map_err(|e| anyhow::anyhow!("failed to lock block manager: {e}"))
}

fn lock_paged_cache(
    paged_cache: &Mutex<PagedKvCache>,
) -> Result<std::sync::MutexGuard<'_, PagedKvCache>> {
    paged_cache
        .lock()
        .map_err(|e| anyhow::anyhow!("failed to lock paged KV cache: {e}"))
}

fn emit_stream_token(
    tx: &mpsc::Sender<StreamEvent>,
    tokenizer: &KilnTokenizer,
    generated_tokens: &mut Vec<TokenId>,
    token: TokenId,
    stop_sequences: &[String],
) -> StreamTokenDisposition {
    generated_tokens.push(token);

    let token_text = tokenizer.decode(&[token]).unwrap_or_default();
    if tx
        .send(StreamEvent::Token(StreamToken {
            token_id: token,
            text: token_text,
        }))
        .is_err()
    {
        return StreamTokenDisposition::ReceiverDropped;
    }

    if !stop_sequences.is_empty() {
        if let Ok(text) = tokenizer.decode(generated_tokens) {
            for stop_seq in stop_sequences {
                if text.contains(stop_seq.as_str()) {
                    return StreamTokenDisposition::Finished(FinishReason::StopSequence(
                        stop_seq.clone(),
                    ));
                }
            }
        }
    }

    StreamTokenDisposition::Continue
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token.
    Eos,
    /// Reached max_tokens limit.
    MaxTokens,
    /// Hit a stop sequence in the decoded text.
    StopSequence(String),
}

impl ModelRunner {
    /// Create a new ModelRunner from pre-loaded weights, tokenizer, and config.
    ///
    /// CUDA graphs for decode are enabled by default on CUDA devices.
    /// Pass `cuda_graphs: false` to disable.
    pub fn new(weights: GpuWeights, tokenizer: KilnTokenizer, config: ModelConfig) -> Self {
        Self::new_with_options(weights, tokenizer, config, true)
    }

    /// Create a new ModelRunner with explicit CUDA graph control.
    pub fn new_with_options(
        weights: GpuWeights,
        tokenizer: KilnTokenizer,
        config: ModelConfig,
        cuda_graphs: bool,
    ) -> Self {
        let eos_token_ids = tokenizer.eos_token_ids();
        let device = weights.embed_tokens.device().clone();
        let cuda_graph = CudaGraphRunner::new(&device, cuda_graphs);
        let backend = backend::for_device(&device);
        Self {
            weights,
            tokenizer,
            config,
            eos_token_ids,
            active_lora: None,
            cuda_graph: Mutex::new(cuda_graph),
            backend,
        }
    }

    /// Load a LoRA adapter from a PEFT-compatible directory.
    ///
    /// The directory must contain `adapter_config.json` and `adapter_model.safetensors`.
    /// Replaces any previously loaded adapter.
    pub fn load_adapter(&mut self, path: &Path) -> Result<()> {
        let device = self.weights.embed_tokens.device().clone();
        let num_layers = self.config.num_layers;
        let lora =
            LoraWeights::load(path, num_layers, &device).context("failed to load LoRA adapter")?;
        self.active_lora = Some(lora);
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        Ok(())
    }

    /// Unload the currently active LoRA adapter, reverting to base model.
    pub fn unload_adapter(&mut self) {
        self.active_lora = None;
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
    }

    /// Returns a reference to the active LoRA weights, if any.
    pub fn active_lora(&self) -> Option<&LoraWeights> {
        self.active_lora.as_ref()
    }

    /// Atomically swap the active LoRA adapter.
    ///
    /// Pass `Some(lora)` to activate pre-loaded weights, or `None` to revert to
    /// the base model. Designed for use with `RwLock`: load weights outside the
    /// lock, then take a brief write lock to call this method.
    ///
    /// Invalidates any captured CUDA graph since the adapter change alters
    /// weight tensor pointers embedded in the graph.
    pub fn swap_lora(&mut self, lora: Option<LoraWeights>) {
        self.active_lora = lora;
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
    }

    fn snapshot_draft_linear_state(
        &self,
        linear_state: &LinearAttentionState,
        spec_config: &SpeculativeConfig,
    ) -> Result<LinearAttentionState> {
        let draft_linear_layers = self
            .weights
            .linear_attention_layers_in_prefix(spec_config.draft_layers);
        linear_state
            .snapshot_for_decode_rollback_prefix(draft_linear_layers)
            .context("clone draft linear-attention prefix from skip-layer prefill")
    }

    /// Generate text from a prompt string.
    ///
    /// Tokenizes the prompt, runs the autoregressive generation loop,
    /// and decodes the output tokens back to text.
    pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens(&prompt_tokens, params)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Create a new KV cache sized for this model.
    fn new_kv_cache(&self, max_seq_len: usize) -> Result<KvCache> {
        let dtype = match self.config.dtype {
            kiln_core::config::DType::BF16 => DType::BF16,
            kiln_core::config::DType::FP16 => DType::F16,
            kiln_core::config::DType::FP32 => DType::F32,
        };
        let device = self.weights.embed_tokens.device();
        KvCache::new(
            self.config.num_full_attention_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_seq_len,
            dtype,
            device,
        )
    }

    /// Create a new linear attention state for GDN layers.
    fn new_linear_state(&self) -> Result<LinearAttentionState> {
        let device = self.weights.embed_tokens.device();
        LinearAttentionState::new(&self.config, device)
    }

    /// Generate text token-by-token, sending each token to a channel as it is produced.
    ///
    /// Returns an `mpsc::Receiver<StreamEvent>` that yields `Token` events
    /// followed by a final `Done` event.  The generation runs synchronously
    /// on the calling thread (caller should use `spawn_blocking`).
    pub fn generate_streaming(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let (tx, rx) = mpsc::channel();

        let max_total = prompt_tokens.len() + params.max_tokens;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: run forward pass on all prompt tokens at once
        let logits = model_forward(
            &*self.backend,
            &prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        // Sample first token from the last position's logits
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;

        let mut next_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            generated_tokens.push(next_token);

            // Decode this token's text
            let token_text = self.tokenizer.decode(&[next_token]).unwrap_or_default();

            // Send token event; if receiver dropped, stop early
            if tx
                .send(StreamEvent::Token(StreamToken {
                    token_id: next_token,
                    text: token_text,
                }))
                .is_err()
            {
                return Ok(rx);
            }

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            finish_reason = FinishReason::StopSequence(stop_seq.clone());
                            let _ = tx.send(StreamEvent::Done(StreamDone {
                                finish_reason,
                                completion_tokens: generated_tokens.len(),
                            }));
                            return Ok(rx);
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            // Decode step: forward pass on just the new token
            let logits = model_forward(
                &*self.backend,
                &[next_token],
                &self.weights,
                &self.config,
                Some(&mut kv_cache),
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("decode forward pass failed")?;
            kv_cache.advance(1);

            next_token = if params.temperature == 0.0 {
                greedy_sample(&logits)?
            } else {
                sample_with_params(
                    &logits,
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    step_seed,
                )?
            };
        }

        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
        }));

        Ok(rx)
    }

    /// Autoregressive generation loop operating on token IDs.
    ///
    /// 1. Prefill: run forward pass on the full prompt to get first next-token logits.
    /// 2. Decode: repeatedly sample a token, run forward on just the new token.
    /// 3. Stop on EOS, max_tokens, or stop sequence.
    pub fn generate_from_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let max_total = prompt_tokens.len() + params.max_tokens;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: run forward pass on all prompt tokens at once
        let logits = model_forward(
            &*self.backend,
            prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        // Sample first token from the last position's logits
        let mut next_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )?
        };

        for _step in 0..params.max_tokens {
            // Advance seed for next step
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            // Check stop sequences against decoded text so far
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            // Decode step: forward pass on just the new token (KV cache has all previous)
            let logits = model_forward(
                &*self.backend,
                &[next_token],
                &self.weights,
                &self.config,
                Some(&mut kv_cache),
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("decode forward pass failed")?;
            kv_cache.advance(1);

            // Sample next token from the new logits
            next_token = if params.temperature == 0.0 {
                greedy_sample(&logits)?
            } else {
                sample_with_params(
                    &logits,
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    step_seed,
                )?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Compute the number of blocks needed for a given number of tokens.
    fn blocks_needed(num_tokens: usize, block_size: usize) -> usize {
        (num_tokens + block_size - 1) / block_size
    }

    /// Generate text from a prompt using paged KV cache backed by a BlockManager.
    ///
    /// This is the memory-efficient path: blocks are allocated on demand from the
    /// shared BlockManager pool and freed when generation completes.
    pub fn generate_paged(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &mut PagedKvCache,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output =
            self.generate_from_tokens_paged(&prompt_tokens, params, block_manager, paged_cache)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Autoregressive generation using paged KV cache.
    ///
    /// Allocates blocks from `block_manager` as needed and frees them when done.
    pub fn generate_from_tokens_paged(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &mut PagedKvCache,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = block_manager.block_size();
        let max_total = prompt_tokens.len() + params.max_tokens;

        // Pre-allocate blocks for the maximum possible sequence length
        let num_blocks = Self::blocks_needed(max_total, block_size);
        let allocated_blocks = block_manager
            .allocate(num_blocks)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut block_table = BlockTable::new();
        for &block_id in &allocated_blocks {
            block_table.push(block_id);
        }

        // Run generation with paged cache; free blocks on completion (or error)
        let result =
            self.generate_from_tokens_paged_inner(prompt_tokens, params, paged_cache, &block_table);

        // Always free allocated blocks
        block_manager.free_all(&allocated_blocks);

        result
    }

    /// Generate text from a prompt using shared paged-cache state protected by
    /// short-lived mutexes.
    ///
    /// On backends with CUDA graph replay enabled we preserve the existing
    /// whole-request lock scope because the graph state is runner-global.
    /// On non-CUDA desktop paths (Metal / CPU), blocks are reserved up front,
    /// the block manager is released immediately, and the paged cache is locked
    /// only around prefill / decode forward passes so concurrent requests can
    /// interleave between decode steps.
    pub fn generate_paged_shared(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_paged_shared(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Same as [`generate_paged_shared`], but accepts an already-tokenized
    /// prompt so API callers do not render/tokenize the same prompt twice.
    pub fn generate_paged_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
    ) -> Result<GenerationOutput> {
        let output = self.generate_from_tokens_paged_shared(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    fn generate_from_tokens_paged_shared(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let cuda_graph_enabled = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled();
        if cuda_graph_enabled {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            return self.generate_from_tokens_paged(
                prompt_tokens,
                params,
                &mut bm_guard,
                &mut pc_guard,
            );
        }

        let max_total = prompt_tokens.len() + params.max_tokens;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_paged_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
        );

        drop(reservation);
        result
    }

    fn decode_next_token_paged_interleaved(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &Mutex<PagedKvCache>,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        step_seed: Option<u64>,
    ) -> Result<TokenId> {
        if params.temperature == 0.0
            && matches!(self.backend.device(), candle_core::Device::Metal(_))
        {
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            return model_forward_paged_next_token_greedy(
                &*self.backend,
                input_token,
                &self.weights,
                &self.config,
                &mut pc_guard,
                block_table,
                seq_len,
                Some(linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("greedy decode forward pass (paged) failed");
        }

        let logits = {
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            model_forward_paged(
                &*self.backend,
                &[input_token],
                &self.weights,
                &self.config,
                &mut pc_guard,
                block_table,
                seq_len,
                Some(linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("decode forward pass (paged) failed")?
        };

        if params.temperature == 0.0 {
            greedy_sample(&logits)
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )
        }
    }

    pub fn generate_paged_speculative_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(
            params.temperature == 0.0,
            "paged skip-layer speculative decode is greedy-only"
        );
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let output = self.generate_from_tokens_paged_speculative_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            spec_config,
        );

        drop(reservation);

        let output = output?;
        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    fn generate_from_tokens_paged_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &Mutex<PagedKvCache>,
        block_table: &BlockTable,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (paged, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?
            }
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        let mut next_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = self.decode_next_token_paged_interleaved(
                params,
                next_token,
                paged_cache,
                block_table,
                seq_len,
                &mut linear_state,
                step_seed,
            )?;
            seq_len += 1;
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    fn generate_from_tokens_paged_speculative_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &Mutex<PagedKvCache>,
        block_table: &BlockTable,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (paged skip-layer, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged skip-layer) failed")?
            }
        };

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut base_pos = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut last_token = greedy_sample(&logits)?;

        loop {
            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            if self.eos_token_ids.contains(&last_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(last_token);
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: spec_config.num_speculative_tokens.min(remaining),
                draft_layers: spec_config.draft_layers,
            };

            let result = {
                let mut pc_guard = lock_paged_cache(paged_cache)?;
                speculative_decode_step_paged_greedy(
                    &*self.backend,
                    last_token,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    base_pos,
                    &mut linear_state,
                    &mut draft_linear_state,
                    &effective_config,
                    params,
                    &self.eos_token_ids,
                    self.active_lora.as_ref(),
                )
                .context("paged skip-layer speculative decode step failed")?
            };
            base_pos += result.base_advance;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);
                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();
            if result.hit_eos {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Inner generation loop using paged KV cache (blocks already allocated).
    fn generate_from_tokens_paged_inner(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &mut PagedKvCache,
        block_table: &BlockTable,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        // Prefill: forward pass on all prompt tokens (never uses CUDA graphs).
        // Long Metal prompts use tiled streaming prefill by default; env
        // overrides can force either path.
        let logits = if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
            model_forward_paged_streaming(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("prefill forward pass (paged, streaming) failed")?
        } else {
            model_forward_paged_last_token(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("prefill forward pass (paged) failed")?
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        // Acquire the CUDA graph runner for decode steps
        let mut graph_runner = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;

        // Sample first token
        let mut next_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.temperature == 0.0
                && matches!(self.backend.device(), candle_core::Device::Metal(_))
            {
                let token = model_forward_paged_next_token_greedy(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    block_table,
                    seq_len,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )?;
                seq_len += 1;
                token
            } else {
                // Decode step: use CUDA graph runner (captures/replays when enabled)
                let logits = graph_runner.decode_step_paged(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    block_table,
                    seq_len,
                    &mut linear_state,
                    self.active_lora.as_ref(),
                )?;
                seq_len += 1;
                sample_with_params(
                    &logits,
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    step_seed,
                )?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Generate text using self-speculative decoding (skip-layer draft).
    ///
    /// The first `spec_config.draft_layers` layers of the model propose candidate
    /// tokens, and the full model verifies them in a single forward pass. Accepted
    /// tokens are emitted in batches, giving 1.5-2.5x decode speedup.
    ///
    /// Falls back to standard generation if speculative config is invalid.
    pub fn generate_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_speculative(&prompt_tokens, params, spec_config)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Speculative generation loop operating on token IDs.
    ///
    /// 1. Prefill: standard full-model forward pass on the prompt.
    /// 2. Decode: draft K tokens with first N layers, verify with full model,
    ///    accept/reject via rejection sampling.
    pub fn generate_from_tokens_speculative(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        use rand::SeedableRng;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        // Verification writes the full speculative window (`last_token + k`)
        // before the loop commits accepted tokens, so flat KV needs temporary
        // headroom beyond the user-visible max token budget.
        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: full model forward pass on all prompt tokens
        let logits = model_forward(
            &*self.backend,
            prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        // Sample first token from prefill logits
        let mut last_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed,
            )?
        };

        loop {
            // Check if we've hit max_tokens
            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            // Check for EOS
            if self.eos_token_ids.contains(&last_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(last_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            // Run one speculative decode step
            let remaining = params.max_tokens - generated_tokens.len();
            let effective_k = spec_config.num_speculative_tokens.min(remaining);
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: effective_k,
                draft_layers: spec_config.draft_layers,
            };

            let result = speculative_decode_step(
                &*self.backend,
                last_token,
                &self.weights,
                &self.config,
                &mut kv_cache,
                &mut linear_state,
                &mut draft_linear_state,
                &effective_config,
                params,
                &self.eos_token_ids,
                &mut rng,
                self.active_lora.as_ref(),
            )
            .context("speculative decode step failed")?;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }
                // No tokens accepted and no EOS — shouldn't happen normally,
                // but fall back to sampling from the verification logits.
                // Break to avoid infinite loop.
                break;
            }

            // Add accepted tokens (except the last one which becomes last_token)
            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);

                // Check stop sequences after each token
                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Generate text using native MTP (Multi-Token Prediction) speculative decoding.
    ///
    /// Uses the model's pretrained MTP head to draft a single candidate token per
    /// step (Qwen3.5-4B ships `num_nextn_predict_layers=1`), which the base model
    /// then verifies in a fused forward pass that emits both the draft-position
    /// target and a bonus token for the accept case.
    ///
    /// Requires the checkpoint to carry `mtp.*` tensors; returns an error
    /// otherwise. Greedy-only (temperature == 0); the stochastic
    /// rejection-sampling variant is a follow-up.
    ///
    /// Reports α (acceptance rate) via the returned [`MtpGenerationOutput`] so
    /// bench callers can publish it alongside throughput numbers.
    pub fn generate_mtp_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<MtpGenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_mtp_speculative(&prompt_tokens, params)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(MtpGenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
            draft_accepted_count: output.draft_accepted_count,
            total_draft_attempts: output.total_draft_attempts,
        })
    }

    /// Native MTP speculative generation operating on token IDs.
    ///
    /// 1. Prefill: paged forward pass on the prompt that returns both logits and
    ///    the last-row pre-final-norm hidden state (`h_prev`).
    /// 2. Decode: per iteration, call [`speculative_mtp_decode_step`] which
    ///    drafts via the MTP head, verifies via the base model, and reports the
    ///    accepted tokens plus advanced positions for the next call.
    ///
    /// Two paged caches are used: the base cache (sized for the model's
    /// full-attention layers) and a 1-layer MTP cache. They have independent
    /// position counters because the MTP layer only commits a slot on accept.
    pub fn generate_from_tokens_mtp_speculative(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<MtpGenerationOutput> {
        use rand::SeedableRng;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        anyhow::ensure!(
            self.weights.mtp.is_some(),
            "generate_mtp_speculative requires the checkpoint to carry mtp.* tensors \
             (Qwen3.5-4B native MTP head)"
        );
        anyhow::ensure!(
            params.temperature == 0.0,
            "generate_mtp_speculative currently only supports greedy decoding (temperature == 0)"
        );

        // Block size matches the kiln-core default + the bench convention.
        const BLOCK_SIZE: usize = 16;

        let max_total = prompt_tokens.len() + params.max_tokens;
        let device = self.weights.embed_tokens.device();
        let dtype = match self.config.dtype {
            kiln_core::config::DType::BF16 => DType::BF16,
            kiln_core::config::DType::FP16 => DType::F16,
            kiln_core::config::DType::FP32 => DType::F32,
        };

        // Two independent paged caches:
        //   * `base_cache` covers the model's full-attention layers.
        //   * `mtp_cache` is a single-layer cache for the MTP block.
        // Each gets its own block table mapping logical block i -> physical i.
        let num_blocks = Self::blocks_needed(max_total, BLOCK_SIZE);
        let mut base_cache = PagedKvCache::new(
            self.config.num_full_attention_layers,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            device,
        )?;
        let mut mtp_cache = PagedKvCache::new(
            1,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            device,
        )?;
        let mut base_block_table = BlockTable::new();
        let mut mtp_block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            base_block_table.push(i);
            mtp_block_table.push(i);
        }

        let mut linear_state = self.new_linear_state()?;

        // Prefill: feed the prompt through the base model and capture the
        // post-final-norm last hidden row as the seed `h_prev`.
        let (prefill_logits, mut h_prev) =
            if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
                model_forward_paged_streaming_last_token_with_last_hidden(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("mtp streaming prefill forward pass failed")?
            } else {
                model_forward_paged_last_token_with_last_hidden(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("mtp prefill forward pass failed")?
            };

        // The last-row logits drive the first emitted token (same as the
        // skip-layer path).
        let prefill_last = prefill_logits.squeeze(1)?;
        let mut last_token = greedy_sample(&prefill_last)?;

        let mut base_pos = prompt_tokens.len();
        let mut mtp_pos = 0usize;
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut draft_accepted_count: usize = 0;
        let mut total_draft_attempts: usize = 0;

        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            if self.eos_token_ids.contains(&last_token) {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            generated_tokens.push(last_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(MtpGenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                draft_accepted_count,
                                total_draft_attempts,
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            total_draft_attempts += 1;
            let mut replay_prefix =
                Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
            replay_prefix.extend_from_slice(prompt_tokens);
            replay_prefix.extend_from_slice(&generated_tokens);
            crate::mtp_debug::set_h_main_replay_prefix_tokens(&replay_prefix);
            let result = speculative_mtp_decode_step(
                &*self.backend,
                last_token,
                &h_prev,
                &self.weights,
                &self.config,
                &mut base_cache,
                &base_block_table,
                base_pos,
                &mut linear_state,
                &mut mtp_cache,
                &mtp_block_table,
                mtp_pos,
                params,
                &self.eos_token_ids,
                &mut rng,
            );
            crate::mtp_debug::clear_h_main_replay_prefix_tokens();
            let result = result.context("mtp speculative decode step failed")?;

            if result.draft_accepted {
                draft_accepted_count += 1;
            }
            base_pos += result.base_advance;
            mtp_pos += result.mtp_advance;
            h_prev = result.new_h_prev;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(MtpGenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                        draft_accepted_count,
                        total_draft_attempts,
                    });
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);

                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(MtpGenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                    draft_accepted_count,
                                    total_draft_attempts,
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(MtpGenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                        draft_accepted_count,
                        total_draft_attempts,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }
        }

        Ok(MtpGenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
            draft_accepted_count,
            total_draft_attempts,
        })
    }

    /// Streaming self-speculative decoding (skip-layer draft).
    ///
    /// Mirrors [`generate_from_tokens_speculative`] but emits committed tokens
    /// incrementally through the returned channel so the SSE desktop path can
    /// benefit from the existing speculative setting.
    pub fn generate_streaming_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        use rand::SeedableRng;

        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let (tx, rx) = mpsc::channel();
        // Verification writes the full speculative window (`last_token + k`)
        // before the loop commits accepted tokens, so flat KV needs temporary
        // headroom beyond the user-visible max token budget.
        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        let logits = model_forward(
            &*self.backend,
            &prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let mut last_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed,
            )?
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.eos_token_ids.contains(&last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut generated_tokens,
                last_token,
                &params.stop,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    let completion_tokens = generated_tokens.len();
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens,
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_k = spec_config.num_speculative_tokens.min(remaining);
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: effective_k,
                draft_layers: spec_config.draft_layers,
            };

            let result = speculative_decode_step(
                &*self.backend,
                last_token,
                &self.weights,
                &self.config,
                &mut kv_cache,
                &mut linear_state,
                &mut draft_linear_state,
                &effective_config,
                params,
                &self.eos_token_ids,
                &mut rng,
                self.active_lora.as_ref(),
            )
            .context("speculative decode step failed")?;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut generated_tokens,
                    token,
                    &params.stop,
                ) {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        let completion_tokens = generated_tokens.len();
                        let _ = tx.send(StreamEvent::Done(StreamDone {
                            finish_reason: reason,
                            completion_tokens,
                        }));
                        return Ok(rx);
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
        }));

        Ok(rx)
    }

    /// Streaming native-MTP speculative decoding.
    ///
    /// Mirrors [`generate_from_tokens_mtp_speculative`] but emits committed
    /// tokens as they are accepted so the desktop streaming path can use MTP
    /// when the checkpoint and request settings allow it.
    pub fn generate_streaming_mtp_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        use rand::SeedableRng;

        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        anyhow::ensure!(
            self.weights.mtp.is_some(),
            "generate_streaming_mtp_speculative requires the checkpoint to carry mtp.* tensors \
             (Qwen3.5-4B native MTP head)"
        );
        anyhow::ensure!(
            params.temperature == 0.0,
            "generate_streaming_mtp_speculative currently only supports greedy decoding \
             (temperature == 0)"
        );

        const BLOCK_SIZE: usize = 16;

        let max_total = prompt_tokens.len() + params.max_tokens;
        let device = self.weights.embed_tokens.device();
        let dtype = match self.config.dtype {
            kiln_core::config::DType::BF16 => DType::BF16,
            kiln_core::config::DType::FP16 => DType::F16,
            kiln_core::config::DType::FP32 => DType::F32,
        };

        let num_blocks = Self::blocks_needed(max_total, BLOCK_SIZE);
        let mut base_cache = PagedKvCache::new(
            self.config.num_full_attention_layers,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            device,
        )?;
        let mut mtp_cache = PagedKvCache::new(
            1,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            device,
        )?;
        let mut base_block_table = BlockTable::new();
        let mut mtp_block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            base_block_table.push(i);
            mtp_block_table.push(i);
        }

        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let (prefill_logits, mut h_prev) =
            if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
                model_forward_paged_streaming_last_token_with_last_hidden(
                    &*self.backend,
                    &prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("mtp streaming prefill forward pass failed")?
            } else {
                model_forward_paged_last_token_with_last_hidden(
                    &*self.backend,
                    &prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut base_cache,
                    &base_block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("mtp prefill forward pass failed")?
            };

        let prefill_last = prefill_logits.squeeze(1)?;
        let mut last_token = greedy_sample(&prefill_last)?;

        let mut base_pos = prompt_tokens.len();
        let mut mtp_pos = 0usize;
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.eos_token_ids.contains(&last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut generated_tokens,
                last_token,
                &params.stop,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    let completion_tokens = generated_tokens.len();
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens,
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let mut replay_prefix =
                Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
            replay_prefix.extend_from_slice(&prompt_tokens);
            replay_prefix.extend_from_slice(&generated_tokens);
            crate::mtp_debug::set_h_main_replay_prefix_tokens(&replay_prefix);
            let result = speculative_mtp_decode_step(
                &*self.backend,
                last_token,
                &h_prev,
                &self.weights,
                &self.config,
                &mut base_cache,
                &base_block_table,
                base_pos,
                &mut linear_state,
                &mut mtp_cache,
                &mtp_block_table,
                mtp_pos,
                params,
                &self.eos_token_ids,
                &mut rng,
            );
            crate::mtp_debug::clear_h_main_replay_prefix_tokens();
            let result = result.context("mtp speculative decode step failed")?;

            base_pos += result.base_advance;
            mtp_pos += result.mtp_advance;
            h_prev = result.new_h_prev;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut generated_tokens,
                    token,
                    &params.stop,
                ) {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        let completion_tokens = generated_tokens.len();
                        let _ = tx.send(StreamEvent::Done(StreamDone {
                            finish_reason: reason,
                            completion_tokens,
                        }));
                        return Ok(rx);
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
        }));

        Ok(rx)
    }

    /// Streaming generation using shared paged-cache state protected by
    /// short-lived mutexes.
    ///
    /// Mirrors [`generate_paged_shared`]: CUDA graph-enabled runtimes keep the
    /// existing whole-request lock scope, while non-CUDA desktop paths reserve
    /// blocks up front and lock the paged cache only around prefill / decode
    /// forward passes.
    pub fn generate_streaming_paged_shared(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        self.generate_from_tokens_streaming_paged_shared(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    /// Streaming variant of [`generate_paged_shared_tokens`].
    pub fn generate_streaming_paged_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.generate_from_tokens_streaming_paged_shared(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    pub fn generate_streaming_paged_speculative_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(
            params.temperature == 0.0,
            "paged skip-layer speculative streaming is greedy-only"
        );
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_streaming_paged_speculative_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            spec_config,
        );

        drop(reservation);
        result
    }

    fn generate_from_tokens_streaming_paged_shared(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &Mutex<PagedKvCache>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let cuda_graph_enabled = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled();
        if cuda_graph_enabled {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            return self.generate_from_tokens_streaming_paged_locked(
                prompt_tokens,
                params,
                &mut bm_guard,
                &mut pc_guard,
            );
        }

        let max_total = prompt_tokens.len() + params.max_tokens;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_streaming_paged_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
        );

        drop(reservation);
        result
    }

    fn generate_from_tokens_streaming_paged_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &Mutex<PagedKvCache>,
        block_table: &BlockTable,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (paged, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?
            }
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;

        let mut next_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut generated_tokens,
                next_token,
                &params.stop,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    finish_reason = reason;
                    break;
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = self.decode_next_token_paged_interleaved(
                params,
                next_token,
                paged_cache,
                block_table,
                seq_len,
                &mut linear_state,
                step_seed,
            )?;
            seq_len += 1;
        }

        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
        }));

        Ok(rx)
    }

    fn generate_from_tokens_streaming_paged_speculative_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &Mutex<PagedKvCache>,
        block_table: &BlockTable,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let mut pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (streaming paged skip-layer, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (streaming paged skip-layer) failed")?
            }
        };

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut base_pos = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut last_token = greedy_sample(&logits)?;

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.eos_token_ids.contains(&last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut generated_tokens,
                last_token,
                &params.stop,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    finish_reason = reason;
                    break;
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: spec_config.num_speculative_tokens.min(remaining),
                draft_layers: spec_config.draft_layers,
            };

            let result = {
                let mut pc_guard = lock_paged_cache(paged_cache)?;
                speculative_decode_step_paged_greedy(
                    &*self.backend,
                    last_token,
                    &self.weights,
                    &self.config,
                    &mut pc_guard,
                    block_table,
                    base_pos,
                    &mut linear_state,
                    &mut draft_linear_state,
                    &effective_config,
                    params,
                    &self.eos_token_ids,
                    self.active_lora.as_ref(),
                )
                .context("streaming paged skip-layer speculative decode step failed")?
            };
            base_pos += result.base_advance;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut generated_tokens,
                    token,
                    &params.stop,
                ) {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        finish_reason = reason;
                        break;
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();
            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
        }));

        Ok(rx)
    }

    /// Streaming generation using paged KV cache.
    ///
    /// Same as [`generate_streaming`] but uses paged KV cache for memory-efficient
    /// serving with the BlockManager.
    pub fn generate_streaming_paged(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &mut PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        self.generate_from_tokens_streaming_paged_locked(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    fn generate_from_tokens_streaming_paged_locked(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &mut PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = block_manager.block_size();
        let max_total = prompt_tokens.len() + params.max_tokens;

        let num_blocks = Self::blocks_needed(max_total, block_size);
        let allocated_blocks = block_manager
            .allocate(num_blocks)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut block_table = BlockTable::new();
        for &block_id in &allocated_blocks {
            block_table.push(block_id);
        }

        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        // Prefill. Long Metal prompts use tiled streaming prefill by default;
        // env overrides can force either path.
        let prefill_result = if streaming_prefill_enabled_for(self.backend.device(), prompt_tokens.len()) {
            model_forward_paged_streaming(
                &*self.backend,
                &prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                &block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
        } else {
            model_forward_paged_last_token(
                &*self.backend,
                &prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                &block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
        };
        let logits = match prefill_result {
            Ok(l) => l,
            Err(e) => {
                block_manager.free_all(&allocated_blocks);
                return Err(e.context("prefill forward pass (paged) failed"));
            }
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;

        // Acquire CUDA graph runner for decode steps
        let mut graph_runner = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;

        let mut next_token = if params.temperature == 0.0 {
            greedy_sample(&logits)?
        } else {
            sample_with_params(
                &logits,
                params.temperature,
                params.top_p,
                params.top_k,
                step_seed,
            )?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            generated_tokens.push(next_token);

            let token_text = self.tokenizer.decode(&[next_token]).unwrap_or_default();

            if tx
                .send(StreamEvent::Token(StreamToken {
                    token_id: next_token,
                    text: token_text,
                }))
                .is_err()
            {
                block_manager.free_all(&allocated_blocks);
                return Ok(rx);
            }

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            finish_reason = FinishReason::StopSequence(stop_seq.clone());
                            let _ = tx.send(StreamEvent::Done(StreamDone {
                                finish_reason,
                                completion_tokens: generated_tokens.len(),
                            }));
                            block_manager.free_all(&allocated_blocks);
                            return Ok(rx);
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.temperature == 0.0
                && matches!(self.backend.device(), candle_core::Device::Metal(_))
            {
                let token = match model_forward_paged_next_token_greedy(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    &block_table,
                    seq_len,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                ) {
                    Ok(token) => token,
                    Err(e) => {
                        block_manager.free_all(&allocated_blocks);
                        return Err(e.context("decode forward pass (paged greedy) failed"));
                    }
                };
                seq_len += 1;
                token
            } else {
                // Decode step: use CUDA graph runner
                let logits = match graph_runner.decode_step_paged(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    &block_table,
                    seq_len,
                    &mut linear_state,
                    self.active_lora.as_ref(),
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        block_manager.free_all(&allocated_blocks);
                        return Err(e.context("decode forward pass (paged) failed"));
                    }
                };
                seq_len += 1;
                sample_with_params(
                    &logits,
                    params.temperature,
                    params.top_p,
                    params.top_k,
                    step_seed,
                )?
            };
        }

        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
        }));

        block_manager.free_all(&allocated_blocks);

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Create a tiny model config for testing.
    fn tiny_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 8,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1, // every layer is full attention
            attn_output_gate: false,
            linear_num_key_heads: 0,
            linear_key_head_dim: 0,
            linear_num_value_heads: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            partial_rotary_factor: 1.0,
        }
    }

    /// Create random GPU weights matching the tiny config.
    fn tiny_weights(config: &ModelConfig, device: &Device) -> GpuWeights {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;

        let embed = Tensor::randn(0.0_f32, 0.02, (vocab, h), device).unwrap();
        let embed_t = embed.t().unwrap().contiguous().unwrap();
        let final_norm = Tensor::zeros((h,), candle_core::DType::F32, device).unwrap();

        let q_proj = Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, h), device).unwrap();
        let k_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device).unwrap();
        let v_proj = Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device).unwrap();
        let o_proj = Tensor::randn(0.0_f32, 0.02, (h, num_heads * head_dim), device).unwrap();
        let q_proj_t = q_proj.t().unwrap().contiguous().unwrap();
        let k_proj_t = k_proj.t().unwrap().contiguous().unwrap();
        let v_proj_t = v_proj.t().unwrap().contiguous().unwrap();
        let o_proj_t = o_proj.t().unwrap().contiguous().unwrap();

        let gate_proj = Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap();
        let up_proj = Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap();
        let down_proj = Tensor::randn(0.0_f32, 0.02, (h, inter), device).unwrap();
        let gate_proj_t = gate_proj.t().unwrap().contiguous().unwrap();
        let up_proj_t = up_proj.t().unwrap().contiguous().unwrap();
        let down_proj_t = down_proj.t().unwrap().contiguous().unwrap();

        let layer = crate::forward::GpuLayerWeights {
            input_layernorm: Tensor::zeros((h,), candle_core::DType::F32, device).unwrap(),
            post_attention_layernorm: Tensor::zeros((h,), candle_core::DType::F32, device).unwrap(),
            attention: crate::forward::GpuAttentionWeights::Full(
                crate::forward::GpuFullAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros((head_dim,), candle_core::DType::F32, device).unwrap(),
                    k_norm: Tensor::zeros((head_dim,), candle_core::DType::F32, device).unwrap(),
                    q_proj_t,
                    k_proj_t,
                    v_proj_t,
                    o_proj_t,
                    q_proj_marlin: None,
                },
            ),
            mlp: crate::forward::GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
            },
        };

        let rotary_inv_freq =
            crate::forward::compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)
                .unwrap();

        GpuWeights {
            embed_tokens: embed,
            embed_tokens_t: embed_t,
            layers: vec![layer],
            final_norm,
            rotary_inv_freq,
            mtp: None,
        }
    }

    /// Create a minimal tokenizer for testing.
    fn test_tokenizer() -> KilnTokenizer {
        // Build a BPE tokenizer with a small vocab that maps single chars to token IDs.
        // We need at least vocab_size=32 tokens for the tiny model.
        let mut vocab = std::collections::HashMap::new();
        for i in 0u32..32 {
            let c = format!("t{i}");
            vocab.insert(c, i);
        }

        let json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            },
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<|endoftext|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ]
        });

        let bytes = serde_json::to_vec(&json).unwrap();
        KilnTokenizer::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn test_generate_from_tokens_max_tokens() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config);

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        let output = runner.generate_from_tokens(&[1, 2, 3], &params)?;

        assert_eq!(output.token_ids.len(), 5);
        assert_eq!(output.finish_reason, FinishReason::MaxTokens);
        // All generated tokens should be valid vocab indices
        for &t in &output.token_ids {
            assert!((t as usize) < 32, "token {t} out of vocab range");
        }

        Ok(())
    }

    #[test]
    fn test_generate_from_tokens_deterministic() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;

        // Create identical weights for two runs
        let weights1 = tiny_weights(&config, &device);
        let tokenizer1 = test_tokenizer();

        // We can't easily get identical random weights, so test with the same runner
        let runner = ModelRunner::new(weights1, tokenizer1, config.clone());

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        };

        let out1 = runner.generate_from_tokens(&[1, 2], &params)?;
        let out2 = runner.generate_from_tokens(&[1, 2], &params)?;

        assert_eq!(
            out1.token_ids, out2.token_ids,
            "greedy decoding should be deterministic"
        );

        Ok(())
    }

    #[test]
    fn test_eos_detection() -> Result<()> {
        // Test that when the model produces an EOS token, generation stops.
        // We do this by generating with the tiny random model and verifying
        // that the output is bounded and the finish reason is correct.
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let mut runner = ModelRunner::new(weights, tokenizer, config);

        // First, generate normally and verify max_tokens works
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        };
        let output = runner.generate_from_tokens(&[1, 2, 3], &params)?;
        assert!(output.token_ids.len() <= 3);

        // Now, set ALL tokens as EOS tokens — generation must stop at step 0
        runner.eos_token_ids = (0u32..32).collect();

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 100,
            ..Default::default()
        };
        let output = runner.generate_from_tokens(&[1, 2], &params)?;
        assert_eq!(output.finish_reason, FinishReason::Eos);
        assert!(
            output.token_ids.is_empty(),
            "all tokens are EOS, should stop immediately, got {:?}",
            output.token_ids
        );

        Ok(())
    }

    #[test]
    fn test_generate_from_tokens_with_temperature() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config);

        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            max_tokens: 5,
            seed: Some(42),
            ..Default::default()
        };

        let out1 = runner.generate_from_tokens(&[1, 2], &params)?;
        let out2 = runner.generate_from_tokens(&[1, 2], &params)?;

        // Same seed should give same results
        assert_eq!(
            out1.token_ids, out2.token_ids,
            "same seed should be deterministic"
        );
        assert_eq!(out1.token_ids.len(), 5);

        Ok(())
    }

    #[test]
    fn test_generate_speculative_max_tokens() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config);

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        // Use 1 draft layer (the tiny model has 1 layer, so draft_layers must be < 1)
        // Since our tiny model only has 1 layer, we can't test speculative decoding
        // with it (need at least 2 layers). Test validation instead.
        let spec_config = SpeculativeConfig {
            num_speculative_tokens: 2,
            draft_layers: 1, // == num_layers, should fail validation
        };

        let result = runner.generate_from_tokens_speculative(&[1, 2, 3], &params, &spec_config);
        assert!(result.is_err(), "draft_layers must be < num_layers");

        Ok(())
    }

    #[test]
    fn test_empty_prompt_errors() {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config);
        let params = SamplingParams::default();

        let result = runner.generate_from_tokens(&[], &params);
        assert!(result.is_err(), "empty prompt should error");
    }

    #[test]
    fn test_generate_paged_max_tokens() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config.clone());

        let block_size = 4;
        let num_blocks = 16; // enough for prompt + max_tokens
        let mut block_manager = BlockManager::new(num_blocks, block_size);
        let mut paged_cache = PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            candle_core::DType::F32,
            &device,
        )?;

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        let output = runner.generate_from_tokens_paged(
            &[1, 2, 3],
            &params,
            &mut block_manager,
            &mut paged_cache,
        )?;

        assert_eq!(output.token_ids.len(), 5);
        assert_eq!(output.finish_reason, FinishReason::MaxTokens);
        for &t in &output.token_ids {
            assert!((t as usize) < 32, "token {t} out of vocab range");
        }

        // Blocks should be freed after generation
        assert_eq!(block_manager.num_free(), num_blocks);

        Ok(())
    }

    #[test]
    fn test_generate_paged_shared_max_tokens() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config.clone());

        let block_size = 4;
        let num_blocks = 16;
        let block_manager = Mutex::new(BlockManager::new(num_blocks, block_size));
        let paged_cache = Mutex::new(PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            candle_core::DType::F32,
            &device,
        )?);

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        let output = runner.generate_from_tokens_paged_shared(
            &[1, 2, 3],
            &params,
            &block_manager,
            &paged_cache,
        )?;

        assert_eq!(output.token_ids.len(), 5);
        assert_eq!(output.finish_reason, FinishReason::MaxTokens);
        assert_eq!(block_manager.lock().unwrap().num_free(), num_blocks);

        Ok(())
    }

    #[test]
    fn test_paged_vs_contiguous_equivalence() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config.clone());

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 5,
            ..Default::default()
        };

        // Generate with contiguous cache
        let contiguous_output = runner.generate_from_tokens(&[1, 2, 3], &params)?;

        // Generate with paged cache
        let block_size = 4;
        let num_blocks = 16;
        let mut block_manager = BlockManager::new(num_blocks, block_size);
        let mut paged_cache = PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            candle_core::DType::F32,
            &device,
        )?;

        let paged_output = runner.generate_from_tokens_paged(
            &[1, 2, 3],
            &params,
            &mut block_manager,
            &mut paged_cache,
        )?;

        // Both paths should produce identical tokens with greedy sampling
        assert_eq!(
            contiguous_output.token_ids, paged_output.token_ids,
            "paged and contiguous paths should produce identical output with greedy sampling"
        );
        assert_eq!(contiguous_output.finish_reason, paged_output.finish_reason);

        Ok(())
    }

    #[test]
    fn test_generate_streaming_paged() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = ModelRunner::new(weights, tokenizer, config.clone());

        let block_size = 4;
        let num_blocks = 16;
        let mut block_manager = BlockManager::new(num_blocks, block_size);
        let mut paged_cache = PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            candle_core::DType::F32,
            &device,
        )?;

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        };

        // Test that paged generation produces tokens incrementally using
        // the inner paged generation path directly with token IDs.
        let output = runner.generate_from_tokens_paged(
            &[1, 2, 3],
            &params,
            &mut block_manager,
            &mut paged_cache,
        )?;

        assert_eq!(output.token_ids.len(), 3);
        assert_eq!(output.finish_reason, FinishReason::MaxTokens);

        // Blocks freed
        assert_eq!(block_manager.num_free(), num_blocks);

        Ok(())
    }

    #[test]
    fn test_generate_paged_shared_concurrent_requests_complete() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let runner = std::sync::Arc::new(ModelRunner::new(weights, tokenizer, config.clone()));
        let block_size = 4;
        let num_blocks = 16;
        let block_manager =
            std::sync::Arc::new(Mutex::new(BlockManager::new(num_blocks, block_size)));
        let paged_cache = std::sync::Arc::new(Mutex::new(PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            candle_core::DType::F32,
            &device,
        )?));

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 3,
            ..Default::default()
        };

        let handle_a = {
            let runner = runner.clone();
            let block_manager = block_manager.clone();
            let paged_cache = paged_cache.clone();
            let params = params.clone();
            std::thread::spawn(move || {
                runner.generate_from_tokens_paged_shared(
                    &[1, 2, 3],
                    &params,
                    block_manager.as_ref(),
                    paged_cache.as_ref(),
                )
            })
        };
        let handle_b = {
            let runner = runner.clone();
            let block_manager = block_manager.clone();
            let paged_cache = paged_cache.clone();
            let params = params.clone();
            std::thread::spawn(move || {
                runner.generate_from_tokens_paged_shared(
                    &[4, 5, 6],
                    &params,
                    block_manager.as_ref(),
                    paged_cache.as_ref(),
                )
            })
        };

        let output_a = handle_a.join().unwrap()?;
        let output_b = handle_b.join().unwrap()?;

        assert_eq!(output_a.token_ids.len(), 3);
        assert_eq!(output_b.token_ids.len(), 3);
        assert_eq!(block_manager.lock().unwrap().num_free(), num_blocks);

        Ok(())
    }

    #[test]
    fn test_paged_eos_detection() -> Result<()> {
        let config = tiny_config();
        let device = Device::Cpu;
        let weights = tiny_weights(&config, &device);
        let tokenizer = test_tokenizer();

        let mut runner = ModelRunner::new(weights, tokenizer, config.clone());

        // Set ALL tokens as EOS
        runner.eos_token_ids = (0u32..32).collect();

        let block_size = 4;
        // Need enough blocks: prompt(2) + max_tokens(100) = 102 tokens, 102/4 = 26 blocks
        let num_blocks = 32;
        let mut block_manager = BlockManager::new(num_blocks, block_size);
        let mut paged_cache = PagedKvCache::new(
            config.num_full_attention_layers,
            num_blocks,
            block_size,
            config.num_kv_heads,
            config.head_dim,
            candle_core::DType::F32,
            &device,
        )?;

        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: 100,
            ..Default::default()
        };

        let output = runner.generate_from_tokens_paged(
            &[1, 2],
            &params,
            &mut block_manager,
            &mut paged_cache,
        )?;

        assert_eq!(output.finish_reason, FinishReason::Eos);
        assert!(
            output.token_ids.is_empty(),
            "all tokens are EOS, should stop immediately"
        );

        // Blocks freed
        assert_eq!(block_manager.num_free(), num_blocks);

        Ok(())
    }
}
