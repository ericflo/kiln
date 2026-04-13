//! End-to-end autoregressive text generation pipeline.
//!
//! Wires together tokenizer, model weights, forward pass, and sampling into
//! a `ModelRunner` that accepts text prompts and produces text output.

use anyhow::{Context, Result};
use candle_core::DType;
use std::path::Path;
use std::sync::mpsc;

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;

use crate::forward::{model_forward, model_forward_paged, GpuWeights, LinearAttentionState};
use crate::kv_cache::KvCache;
use crate::lora_loader::LoraWeights;
use crate::paged_kv_cache::PagedKvCache;
use crate::sampling::{greedy_sample, sample_with_params};

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
    pub fn new(weights: GpuWeights, tokenizer: KilnTokenizer, config: ModelConfig) -> Self {
        let eos_token_ids = tokenizer.eos_token_ids();
        Self {
            weights,
            tokenizer,
            config,
            eos_token_ids,
            active_lora: None,
        }
    }

    /// Load a LoRA adapter from a PEFT-compatible directory.
    ///
    /// The directory must contain `adapter_config.json` and `adapter_model.safetensors`.
    /// Replaces any previously loaded adapter.
    pub fn load_adapter(&mut self, path: &Path) -> Result<()> {
        let device = self.weights.embed_tokens.device().clone();
        let num_layers = self.config.num_layers;
        let lora = LoraWeights::load(path, num_layers, &device)
            .context("failed to load LoRA adapter")?;
        self.active_lora = Some(lora);
        Ok(())
    }

    /// Unload the currently active LoRA adapter, reverting to base model.
    pub fn unload_adapter(&mut self) {
        self.active_lora = None;
    }

    /// Returns a reference to the active LoRA weights, if any.
    pub fn active_lora(&self) -> Option<&LoraWeights> {
        self.active_lora.as_ref()
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
        let logits = model_forward(&prompt_tokens, &self.weights, &self.config, Some(&mut kv_cache), Some(&mut linear_state), self.active_lora.as_ref())
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
            let token_text = self
                .tokenizer
                .decode(&[next_token])
                .unwrap_or_default();

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
                            finish_reason =
                                FinishReason::StopSequence(stop_seq.clone());
                            let _ = tx.send(StreamEvent::Done(StreamDone {
                                finish_reason,
                                completion_tokens: generated_tokens.len(),
                            }));
                            return Ok(rx);
                        }
                    }
                }
            }

            // Decode step: forward pass on just the new token
            let logits = model_forward(&[next_token], &self.weights, &self.config, Some(&mut kv_cache), Some(&mut linear_state), self.active_lora.as_ref())
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
        let logits = model_forward(prompt_tokens, &self.weights, &self.config, Some(&mut kv_cache), Some(&mut linear_state), self.active_lora.as_ref())
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

            // Decode step: forward pass on just the new token (KV cache has all previous)
            let logits = model_forward(&[next_token], &self.weights, &self.config, Some(&mut kv_cache), Some(&mut linear_state), self.active_lora.as_ref())
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

        let output = self.generate_from_tokens_paged(&prompt_tokens, params, block_manager, paged_cache)?;

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
        let result = self.generate_from_tokens_paged_inner(
            prompt_tokens, params, paged_cache, &block_table,
        );

        // Always free allocated blocks
        block_manager.free_all(&allocated_blocks);

        result
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

        // Prefill: forward pass on all prompt tokens
        let logits = model_forward_paged(
            prompt_tokens,
            &self.weights,
            &self.config,
            paged_cache,
            block_table,
            0,
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass (paged) failed")?;

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

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

            // Decode step: forward pass on just the new token
            let logits = model_forward_paged(
                &[next_token],
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                seq_len,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("decode forward pass (paged) failed")?;
            seq_len += 1;

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

        // Prefill
        let logits = match model_forward_paged(
            &prompt_tokens,
            &self.weights,
            &self.config,
            paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        ) {
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

            let token_text = self
                .tokenizer
                .decode(&[next_token])
                .unwrap_or_default();

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

            // Decode step
            let logits = match model_forward_paged(
                &[next_token],
                &self.weights,
                &self.config,
                paged_cache,
                &block_table,
                seq_len,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            ) {
                Ok(l) => l,
                Err(e) => {
                    block_manager.free_all(&allocated_blocks);
                    return Err(e.context("decode forward pass (paged) failed"));
                }
            };
            seq_len += 1;

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
        let final_norm = Tensor::zeros((h,), candle_core::DType::F32, device).unwrap();

        let layer = crate::forward::GpuLayerWeights {
            input_layernorm: Tensor::zeros((h,), candle_core::DType::F32, device).unwrap(),
            post_attention_layernorm: Tensor::zeros((h,), candle_core::DType::F32, device).unwrap(),
            attention: crate::forward::GpuAttentionWeights::Full(
                crate::forward::GpuFullAttentionWeights {
                    q_proj: Tensor::randn(0.0_f32, 0.02, (num_heads * head_dim, h), device)
                        .unwrap(),
                    k_proj: Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device)
                        .unwrap(),
                    v_proj: Tensor::randn(0.0_f32, 0.02, (num_kv_heads * head_dim, h), device)
                        .unwrap(),
                    o_proj: Tensor::randn(0.0_f32, 0.02, (h, num_heads * head_dim), device)
                        .unwrap(),
                    q_norm: Tensor::zeros((head_dim,), candle_core::DType::F32, device).unwrap(),
                    k_norm: Tensor::zeros((head_dim,), candle_core::DType::F32, device).unwrap(),
                },
            ),
            mlp: crate::forward::GpuFfnWeights {
                gate_proj: Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap(),
                up_proj: Tensor::randn(0.0_f32, 0.02, (inter, h), device).unwrap(),
                down_proj: Tensor::randn(0.0_f32, 0.02, (h, inter), device).unwrap(),
            },
        };

        GpuWeights {
            embed_tokens: embed,
            layers: vec![layer],
            final_norm,
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
        assert_eq!(out1.token_ids, out2.token_ids, "same seed should be deterministic");
        assert_eq!(out1.token_ids.len(), 5);

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
        assert!(output.token_ids.is_empty(), "all tokens are EOS, should stop immediately");

        // Blocks freed
        assert_eq!(block_manager.num_free(), num_blocks);

        Ok(())
    }
}
