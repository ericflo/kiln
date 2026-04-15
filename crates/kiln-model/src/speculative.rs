//! Self-speculative decoding using skip-layer drafting.
//!
//! Instead of loading a separate smaller draft model (which would double VRAM),
//! we use the first N layers of the main model as a lightweight "draft model."
//! The draft proposes K candidate tokens, then the full model verifies them in
//! a single forward pass, accepting correct predictions via rejection sampling.
//!
//! Expected speedup: 1.5-2.5x for autoregressive decode, depending on the
//! acceptance rate (which depends on how well the first N layers predict the
//! full model's output).

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;

use crate::forward::{
    model_forward, model_forward_embed, model_forward_head, model_forward_segment,
    GpuWeights, LinearAttentionState,
};
use crate::kv_cache::KvCache;
use crate::sampling::{greedy_sample, sample_with_params};

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per speculation step.
    /// Higher values amortize verification cost but risk more rejections.
    /// Typical range: 3-8. Default: 4.
    pub num_speculative_tokens: usize,

    /// Number of layers to use for the draft model (skip-layer approach).
    /// Uses the first `draft_layers` layers of the main model.
    /// Default: 8 (25% of Qwen3.5-4B's 32 layers).
    pub draft_layers: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            draft_layers: 8,
        }
    }
}

impl SpeculativeConfig {
    /// Validate the config against the model configuration.
    pub fn validate(&self, model_config: &ModelConfig) -> Result<()> {
        anyhow::ensure!(
            self.num_speculative_tokens > 0,
            "num_speculative_tokens must be > 0, got {}",
            self.num_speculative_tokens
        );
        anyhow::ensure!(
            self.draft_layers > 0 && self.draft_layers < model_config.num_layers,
            "draft_layers must be in [1, {}), got {}",
            model_config.num_layers,
            self.draft_layers
        );
        Ok(())
    }
}

/// Result of one speculative decoding step.
#[derive(Debug)]
pub struct SpeculativeStepResult {
    /// Tokens accepted (and possibly one resampled token at the rejection point).
    pub accepted_tokens: Vec<TokenId>,
    /// Whether an EOS token was encountered among the accepted tokens.
    pub hit_eos: bool,
}

/// Run a draft forward pass using only the first N layers of the model.
///
/// This produces lower-quality logits but runs much faster since it skips
/// most of the model's layers.
fn draft_forward(
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &ModelConfig,
    draft_layers: usize,
    linear_state: &mut LinearAttentionState,
) -> Result<Tensor> {
    // Embed tokens
    let (hidden, positions) = model_forward_embed(token_ids, weights)?;

    // Run only the first `draft_layers` layers
    let hidden = model_forward_segment(
        hidden,
        weights,
        config,
        &positions,
        0,
        draft_layers,
        Some(linear_state),
        None, // no LoRA for draft (speed over quality)
    )?;

    // Project to vocab
    model_forward_head(&hidden, weights, config)
}

/// Compute softmax probabilities from logits for the last position.
///
/// Returns a Vec of probabilities indexed by token ID.
fn logits_to_probs(logits: &Tensor, temperature: f32) -> Result<Vec<f32>> {
    let dims = logits.dims();

    // Extract logits for the last position
    let last_logits = if dims.len() >= 2 {
        let seq_len = dims[dims.len() - 2];
        logits
            .narrow(dims.len() - 2, seq_len - 1, 1)?
            .squeeze(dims.len() - 2)?
    } else {
        logits.clone()
    };

    let flat = last_logits.flatten_all()?.to_dtype(DType::F32)?;
    let mut vals = flat.to_vec1::<f32>()?;

    // Apply temperature
    if temperature > 0.0 && temperature != 1.0 {
        for v in vals.iter_mut() {
            *v /= temperature;
        }
    }

    // Softmax
    let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    Ok(probs)
}

/// Perform rejection sampling for speculative decoding.
///
/// For each draft token, we accept it with probability min(1, p_target / p_draft).
/// If rejected, we resample from the adjusted distribution max(0, p_target - p_draft).
///
/// This guarantees the output distribution exactly matches the target model,
/// regardless of draft model quality.
pub fn rejection_sample(
    draft_token: TokenId,
    draft_probs: &[f32],
    target_probs: &[f32],
    rng: &mut StdRng,
) -> Result<(bool, Option<TokenId>)> {
    let draft_idx = draft_token as usize;
    anyhow::ensure!(
        draft_idx < draft_probs.len() && draft_idx < target_probs.len(),
        "token {} out of range (draft_probs: {}, target_probs: {})",
        draft_token,
        draft_probs.len(),
        target_probs.len()
    );

    let p_draft = draft_probs[draft_idx];
    let p_target = target_probs[draft_idx];

    // Accept with probability min(1, p_target / p_draft)
    let accept_prob = if p_draft > 0.0 {
        (p_target / p_draft).min(1.0)
    } else if p_target > 0.0 {
        // Draft assigned 0 probability but target didn't — reject and resample
        0.0
    } else {
        // Both are 0 — accept (token won't matter, but keeps the math clean)
        1.0
    };

    let r: f32 = rng.r#gen();

    if r < accept_prob {
        // Accept the draft token
        Ok((true, None))
    } else {
        // Reject: resample from adjusted distribution max(0, p_target - p_draft)
        let resampled = resample_adjusted(target_probs, draft_probs, rng)?;
        Ok((false, Some(resampled)))
    }
}

/// Resample from the adjusted distribution: normalize(max(0, p_target - p_draft)).
fn resample_adjusted(
    target_probs: &[f32],
    draft_probs: &[f32],
    rng: &mut StdRng,
) -> Result<TokenId> {
    let len = target_probs.len().min(draft_probs.len());
    let mut adjusted: Vec<f32> = Vec::with_capacity(len);
    for i in 0..len {
        adjusted.push((target_probs[i] - draft_probs[i]).max(0.0));
    }

    let sum: f32 = adjusted.iter().sum();
    if sum <= 0.0 {
        // Fallback: sample from target distribution directly
        let r: f32 = rng.r#gen();
        let mut cumsum = 0.0;
        for (i, &p) in target_probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return Ok(i as TokenId);
            }
        }
        return Ok((target_probs.len() - 1) as TokenId);
    }

    // Normalize and sample
    let r: f32 = rng.r#gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for (i, &p) in adjusted.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as TokenId);
        }
    }

    Ok((len - 1) as TokenId)
}

/// Initialize the draft model's linear attention state by running the prompt
/// through the draft layers. The logits output is discarded; only the state
/// mutation matters.
pub fn draft_forward_for_state_init(
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &ModelConfig,
    draft_layers: usize,
    linear_state: &mut LinearAttentionState,
) -> Result<()> {
    let _ = draft_forward(token_ids, weights, config, draft_layers, linear_state)?;
    Ok(())
}

/// Run one speculative decoding step: draft K tokens, verify with full model.
///
/// `kv_cache`: the KV cache for the full model (already populated up to current position).
/// `linear_state`: linear attention state for the full model.
/// `draft_linear_state`: separate linear attention state for the draft model.
/// `last_token`: the last accepted token (input to both draft and verify).
///
/// Returns the accepted tokens and whether EOS was hit.
pub fn speculative_decode_step(
    last_token: TokenId,
    weights: &GpuWeights,
    config: &ModelConfig,
    kv_cache: &mut KvCache,
    linear_state: &mut LinearAttentionState,
    draft_linear_state: &mut LinearAttentionState,
    spec_config: &SpeculativeConfig,
    params: &SamplingParams,
    eos_token_ids: &[TokenId],
    rng: &mut StdRng,
    lora: Option<&crate::lora_loader::LoraWeights>,
) -> Result<SpeculativeStepResult> {
    let k = spec_config.num_speculative_tokens;
    let temperature = params.temperature;

    // Phase 1: Draft — generate K candidate tokens using the first N layers.
    // We need a separate KV cache snapshot for the draft model since it runs
    // independently. For self-speculative with shared weights, we clone the
    // linear state but don't use a KV cache for the draft (it uses the segment
    // forward which doesn't need one).
    let mut draft_tokens: Vec<TokenId> = Vec::with_capacity(k);
    let mut draft_probs_list: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut current_token = last_token;

    for _ in 0..k {
        let draft_logits = draft_forward(
            &[current_token],
            weights,
            config,
            spec_config.draft_layers,
            draft_linear_state,
        )
        .context("draft forward pass failed")?;

        let probs = if temperature > 0.0 {
            logits_to_probs(&draft_logits, temperature)?
        } else {
            logits_to_probs(&draft_logits, 1.0)?
        };

        // Sample from draft
        let draft_token = if temperature == 0.0 {
            greedy_sample(&draft_logits)?
        } else {
            sample_with_params(
                &draft_logits,
                temperature,
                params.top_p,
                params.top_k,
                None, // use rng directly for speculative, not the seed
            )?
        };

        draft_probs_list.push(probs);
        draft_tokens.push(draft_token);
        current_token = draft_token;
    }

    // Phase 2: Verify — run the full model on all K+1 positions in one pass.
    // Input: [last_token, draft_token_0, ..., draft_token_{K-1}]
    // Output: logits for K+1 positions
    let mut verify_input: Vec<TokenId> = Vec::with_capacity(k + 1);
    verify_input.push(last_token);
    verify_input.extend_from_slice(&draft_tokens);

    let verify_logits = model_forward(
        &verify_input,
        weights,
        config,
        Some(kv_cache),
        Some(linear_state),
        lora,
    )
    .context("verification forward pass failed")?;

    // The verify_logits has shape [1, K+1, vocab_size].
    // Position i gives logits for predicting token at position i+1.
    // So verify_logits[0] predicts what should follow last_token (i.e., verifies draft_tokens[0]).

    let mut accepted_tokens: Vec<TokenId> = Vec::new();
    let mut hit_eos = false;

    for i in 0..k {
        // Extract target probabilities for position i
        let pos_logits = verify_logits.narrow(1, i, 1)?.squeeze(1)?;
        let target_probs = if temperature > 0.0 {
            logits_to_probs(&pos_logits, temperature)?
        } else {
            logits_to_probs(&pos_logits, 1.0)?
        };

        let draft_token = draft_tokens[i];

        // Check EOS before acceptance
        if eos_token_ids.contains(&draft_token) {
            // If draft proposed EOS, check if target agrees
            if temperature == 0.0 {
                let target_token = greedy_sample(&pos_logits)?;
                if eos_token_ids.contains(&target_token) {
                    hit_eos = true;
                    break;
                }
                // Target disagrees with EOS — accept target's token instead
                accepted_tokens.push(target_token);
            } else {
                let (accepted, resampled) = rejection_sample(
                    draft_token,
                    &draft_probs_list[i],
                    &target_probs,
                    rng,
                )?;
                if accepted {
                    hit_eos = true;
                    break;
                }
                if let Some(token) = resampled {
                    if eos_token_ids.contains(&token) {
                        hit_eos = true;
                    } else {
                        accepted_tokens.push(token);
                    }
                }
            }
            break;
        }

        if temperature == 0.0 {
            // Greedy verification: accept if target agrees
            let target_token = greedy_sample(&pos_logits)?;
            if target_token == draft_token {
                accepted_tokens.push(draft_token);
            } else {
                // Reject: use the target's token instead
                if eos_token_ids.contains(&target_token) {
                    hit_eos = true;
                } else {
                    accepted_tokens.push(target_token);
                }
                break;
            }
        } else {
            // Stochastic verification via rejection sampling
            let (accepted, resampled) = rejection_sample(
                draft_token,
                &draft_probs_list[i],
                &target_probs,
                rng,
            )?;

            if accepted {
                accepted_tokens.push(draft_token);
            } else {
                if let Some(token) = resampled {
                    if eos_token_ids.contains(&token) {
                        hit_eos = true;
                    } else {
                        accepted_tokens.push(token);
                    }
                }
                break;
            }
        }
    }

    // Bonus token: if ALL K draft tokens were accepted, sample one more from
    // the last verification position (position K).
    if accepted_tokens.len() == k && !hit_eos {
        let bonus_logits = verify_logits.narrow(1, k, 1)?.squeeze(1)?;
        let bonus_token = if temperature == 0.0 {
            greedy_sample(&bonus_logits)?
        } else {
            sample_with_params(
                &bonus_logits,
                temperature,
                params.top_p,
                params.top_k,
                None,
            )?
        };

        if eos_token_ids.contains(&bonus_token) {
            hit_eos = true;
        } else {
            accepted_tokens.push(bonus_token);
        }
    }

    // Advance KV cache by the number of tokens we verified (K+1),
    // but we'll need to roll back to only the accepted count.
    // Actually, model_forward already wrote all K+1 positions into the cache.
    // We need to advance by only the accepted count + 1 (for last_token).
    // The KV cache advance is handled by the caller based on accepted_tokens.len().
    kv_cache.advance(1 + accepted_tokens.len());

    Ok(SpeculativeStepResult {
        accepted_tokens,
        hit_eos,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn test_rejection_sample_identical_distributions() {
        // When draft == target, acceptance rate should be 100%
        let probs = vec![0.1, 0.3, 0.4, 0.2];
        let mut rng = make_rng(42);

        for _ in 0..100 {
            for token in 0..4u32 {
                let (accepted, _) = rejection_sample(token, &probs, &probs, &mut rng).unwrap();
                assert!(accepted, "identical distributions should always accept");
            }
        }
    }

    #[test]
    fn test_rejection_sample_target_concentrated() {
        // Target puts all mass on token 0, draft is uniform
        let draft_probs = vec![0.25, 0.25, 0.25, 0.25];
        let target_probs = vec![1.0, 0.0, 0.0, 0.0];
        let mut rng = make_rng(42);

        // Token 0 should always be accepted (p_target/p_draft = 4.0, clamped to 1.0)
        for _ in 0..50 {
            let (accepted, _) = rejection_sample(0, &draft_probs, &target_probs, &mut rng).unwrap();
            assert!(accepted, "token 0 should always be accepted when target concentrates on it");
        }

        // Token 1 should always be rejected (p_target = 0)
        for _ in 0..50 {
            let (accepted, resampled) =
                rejection_sample(1, &draft_probs, &target_probs, &mut rng).unwrap();
            assert!(!accepted, "token 1 should always be rejected when target assigns 0 prob");
            // Resampled token should always be 0 (the only token with mass in adjusted dist)
            assert_eq!(resampled, Some(0), "resampled should be token 0");
        }
    }

    #[test]
    fn test_rejection_sample_draft_zero_target_nonzero() {
        // Draft assigns 0 prob, target assigns nonzero — should reject
        let draft_probs = vec![0.0, 0.5, 0.5, 0.0];
        let target_probs = vec![0.3, 0.2, 0.2, 0.3];
        let mut rng = make_rng(42);

        let (accepted, resampled) =
            rejection_sample(0, &draft_probs, &target_probs, &mut rng).unwrap();
        assert!(!accepted, "should reject when draft assigns 0 probability");
        assert!(resampled.is_some(), "should provide a resampled token");
    }

    #[test]
    fn test_rejection_sample_both_zero() {
        let draft_probs = vec![0.0, 0.5, 0.5, 0.0];
        let target_probs = vec![0.0, 0.6, 0.4, 0.0];
        let mut rng = make_rng(42);

        // Both are 0 for token 0 — should accept
        let (accepted, _) = rejection_sample(0, &draft_probs, &target_probs, &mut rng).unwrap();
        assert!(accepted, "both zero should accept");
    }

    #[test]
    fn test_logits_to_probs_sums_to_one() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 0.5], &device).unwrap();
        let probs = logits_to_probs(&logits, 1.0).unwrap();

        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probabilities should sum to 1.0, got {sum}"
        );

        // All probabilities should be non-negative
        for &p in &probs {
            assert!(p >= 0.0, "probability should be non-negative, got {p}");
        }
    }

    #[test]
    fn test_logits_to_probs_temperature_effect() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 1.0], &device).unwrap();

        // Low temperature should make distribution more peaked
        let probs_low = logits_to_probs(&logits, 0.1).unwrap();
        let probs_high = logits_to_probs(&logits, 10.0).unwrap();

        // At low temp, the max prob should be higher
        let max_low = probs_low.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_high = probs_high.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_low > max_high,
            "lower temperature should produce more peaked distribution: {max_low} vs {max_high}"
        );
    }

    #[test]
    fn test_logits_to_probs_2d() {
        let device = candle_core::Device::Cpu;
        // [seq_len=2, vocab_size=3]
        let logits = Tensor::new(
            &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &device,
        )
        .unwrap()
        .reshape((2, 3))
        .unwrap();

        // Should extract last position (4.0, 5.0, 6.0)
        let probs = logits_to_probs(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        // Token 2 (logit 6.0) should have highest probability
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_speculative_config_validation() {
        let model_config = ModelConfig::qwen3_5_4b();

        // Valid config
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            draft_layers: 8,
        };
        assert!(config.validate(&model_config).is_ok());

        // Zero speculative tokens
        let config = SpeculativeConfig {
            num_speculative_tokens: 0,
            draft_layers: 8,
        };
        assert!(config.validate(&model_config).is_err());

        // Zero draft layers
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            draft_layers: 0,
        };
        assert!(config.validate(&model_config).is_err());

        // Draft layers >= total layers
        let config = SpeculativeConfig {
            num_speculative_tokens: 4,
            draft_layers: 32,
        };
        assert!(config.validate(&model_config).is_err());
    }

    #[test]
    fn test_resample_adjusted_fallback() {
        // When target == draft, adjusted is all zeros — should fallback to target
        let target = vec![0.3, 0.3, 0.4];
        let draft = vec![0.3, 0.3, 0.4];
        let mut rng = make_rng(42);

        let token = resample_adjusted(&target, &draft, &mut rng).unwrap();
        assert!((token as usize) < 3, "resampled token should be in range");
    }

    #[test]
    fn test_resample_adjusted_concentrates() {
        // Target has mass where draft doesn't
        let target = vec![0.0, 0.0, 1.0];
        let draft = vec![0.5, 0.5, 0.0];
        let mut rng = make_rng(42);

        // Adjusted = max(0, [0-0.5, 0-0.5, 1-0]) = [0, 0, 1]
        for _ in 0..20 {
            let token = resample_adjusted(&target, &draft, &mut rng).unwrap();
            assert_eq!(token, 2, "should always resample token 2");
        }
    }
}
