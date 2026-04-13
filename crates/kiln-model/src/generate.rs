//! End-to-end autoregressive text generation pipeline.
//!
//! Wires together tokenizer, model weights, forward pass, and sampling into
//! a `ModelRunner` that accepts text prompts and produces text output.

use anyhow::{Context, Result};
use std::sync::mpsc;

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;

use crate::forward::{model_forward, GpuWeights};
use crate::sampling::{greedy_sample, sample_with_params};

/// Holds loaded model weights and tokenizer, provides text generation.
pub struct ModelRunner {
    pub weights: GpuWeights,
    pub tokenizer: KilnTokenizer,
    pub config: ModelConfig,
    /// EOS token IDs cached from the tokenizer.
    eos_token_ids: Vec<TokenId>,
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
        }
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

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut all_tokens: Vec<TokenId> = prompt_tokens.to_vec();
        let mut step_seed = params.seed;

        let mut finish_reason = FinishReason::MaxTokens;

        for _step in 0..params.max_tokens {
            let logits = model_forward(&all_tokens, &self.weights, &self.config)
                .context("forward pass failed")?;

            let next_token = if params.temperature == 0.0 {
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

            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

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
                            // Send done and return
                            let _ = tx.send(StreamEvent::Done(StreamDone {
                                finish_reason,
                                completion_tokens: generated_tokens.len(),
                            }));
                            return Ok(rx);
                        }
                    }
                }
            }
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
    /// 2. Decode: repeatedly sample a token, append it, run forward on just the new token.
    /// 3. Stop on EOS, max_tokens, or stop sequence.
    pub fn generate_from_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut all_tokens: Vec<TokenId> = prompt_tokens.to_vec();

        // Track the RNG seed: for deterministic generation with temperature > 0,
        // we increment the seed for each step so the sequence is reproducible
        // but each step samples differently.
        let mut step_seed = params.seed;

        for _step in 0..params.max_tokens {
            // Run forward pass on all tokens so far.
            // In a production system this would use KV cache to only compute
            // the new token's attention, but for correctness-first we recompute
            // the full sequence each step.
            let logits = model_forward(&all_tokens, &self.weights, &self.config)
                .context("forward pass failed")?;

            // Sample next token from the last position's logits
            let next_token = if params.temperature == 0.0 {
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

            // Advance seed for next step
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(), // caller fills this in
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

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
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
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
        let final_norm = Tensor::ones((h,), candle_core::DType::F32, device).unwrap();

        let layer = crate::forward::GpuLayerWeights {
            input_layernorm: Tensor::ones((h,), candle_core::DType::F32, device).unwrap(),
            post_attention_layernorm: Tensor::ones((h,), candle_core::DType::F32, device).unwrap(),
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
                    q_norm: Tensor::ones((head_dim,), candle_core::DType::F32, device).unwrap(),
                    k_norm: Tensor::ones((head_dim,), candle_core::DType::F32, device).unwrap(),
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
}
