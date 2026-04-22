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
use rand::Rng;
use rand::rngs::StdRng;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;

use crate::backend::BackendRuntime;
use crate::c1_attr;
use crate::forward::{
    GpuWeights, LinearAttentionState, model_forward, model_forward_embed, model_forward_head,
    model_forward_paged, model_forward_paged_with_last_hidden, model_forward_segment,
    mtp_forward_step,
};
use crate::kv_cache::KvCache;
use crate::paged_kv_cache::PagedKvCache;
use crate::sampling::{greedy_sample, greedy_sample_rows, sample_with_params};

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per speculation step.
    /// Higher values amortize verification cost but risk more rejections.
    /// Typical range depends on the verifier path. The paged greedy path uses
    /// a large default window to amortize full-model verification.
    pub num_speculative_tokens: usize,

    /// Number of layers to use for the draft model (skip-layer approach).
    /// Uses the first `draft_layers` layers of the main model.
    /// Default: 8 (25% of Qwen3.5-4B's 32 layers).
    pub draft_layers: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 256,
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
fn draft_forward_hidden(
    backend: &dyn BackendRuntime,
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
        backend,
        hidden,
        weights,
        config,
        &positions,
        0,
        draft_layers,
        Some(linear_state),
        None, // no LoRA for draft (speed over quality)
    )?;

    Ok(hidden)
}

fn draft_forward(
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &ModelConfig,
    draft_layers: usize,
    linear_state: &mut LinearAttentionState,
) -> Result<Tensor> {
    let hidden = draft_forward_hidden(
        backend,
        token_ids,
        weights,
        config,
        draft_layers,
        linear_state,
    )?;
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
    backend: &dyn BackendRuntime,
    token_ids: &[u32],
    weights: &GpuWeights,
    config: &ModelConfig,
    draft_layers: usize,
    linear_state: &mut LinearAttentionState,
) -> Result<()> {
    let _ = draft_forward_hidden(
        backend,
        token_ids,
        weights,
        config,
        draft_layers,
        linear_state,
    )?;
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
#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step(
    backend: &dyn BackendRuntime,
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
    let mut draft_probs_list: Vec<Vec<f32>> = if temperature > 0.0 {
        Vec::with_capacity(k)
    } else {
        Vec::new()
    };
    let mut current_token = last_token;

    for _ in 0..k {
        let draft_logits = draft_forward(
            backend,
            &[current_token],
            weights,
            config,
            spec_config.draft_layers,
            draft_linear_state,
        )
        .context("draft forward pass failed")?;

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

        if temperature > 0.0 {
            draft_probs_list.push(logits_to_probs(&draft_logits, temperature)?);
        }
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
        backend,
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
    let greedy_target_tokens = if temperature == 0.0 {
        let tokens = greedy_sample_rows(&verify_logits)
            .context("batched verification greedy sampling failed")?;
        anyhow::ensure!(
            tokens.len() == k + 1,
            "verifier returned {} target tokens for window {}",
            tokens.len(),
            k + 1
        );
        Some(tokens)
    } else {
        None
    };

    let mut accepted_tokens: Vec<TokenId> = Vec::new();
    let mut hit_eos = false;

    for i in 0..k {
        let draft_token = draft_tokens[i];

        // Check EOS before acceptance
        if eos_token_ids.contains(&draft_token) {
            // If draft proposed EOS, check if target agrees
            if temperature == 0.0 {
                let target_token = greedy_target_tokens.as_ref().expect("greedy tokens")[i];
                if eos_token_ids.contains(&target_token) {
                    hit_eos = true;
                    break;
                }
                // Target disagrees with EOS — accept target's token instead
                accepted_tokens.push(target_token);
            } else {
                let pos_logits = verify_logits.narrow(1, i, 1)?.squeeze(1)?;
                let target_probs = logits_to_probs(&pos_logits, temperature)?;
                let (accepted, resampled) =
                    rejection_sample(draft_token, &draft_probs_list[i], &target_probs, rng)?;
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
            let target_token = greedy_target_tokens.as_ref().expect("greedy tokens")[i];
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
            let pos_logits = verify_logits.narrow(1, i, 1)?.squeeze(1)?;
            let target_probs = logits_to_probs(&pos_logits, temperature)?;
            let (accepted, resampled) =
                rejection_sample(draft_token, &draft_probs_list[i], &target_probs, rng)?;

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
        let bonus_token = if temperature == 0.0 {
            greedy_target_tokens.as_ref().expect("greedy tokens")[k]
        } else {
            let bonus_logits = verify_logits.narrow(1, k, 1)?.squeeze(1)?;
            sample_with_params(&bonus_logits, temperature, params.top_p, params.top_k, None)?
        };

        if eos_token_ids.contains(&bonus_token) {
            hit_eos = true;
        } else {
            accepted_tokens.push(bonus_token);
        }
    }

    // `model_forward` wrote the whole verification window, but only the
    // committed prefix is valid for the next step. `accepted_tokens` includes
    // the target/bonus token that will be fed as `last_token` next time, so
    // advancing by exactly this count keeps the cache aligned while leaving
    // rejected/stale slots to be overwritten.
    kv_cache.advance(accepted_tokens.len());

    Ok(SpeculativeStepResult {
        accepted_tokens,
        hit_eos,
    })
}

/// Result of one paged greedy skip-layer speculative decoding step.
#[derive(Debug)]
pub struct PagedSpeculativeStepResult {
    pub accepted_tokens: Vec<TokenId>,
    pub hit_eos: bool,
    pub base_advance: usize,
    pub accepted_draft_tokens: usize,
    pub attempted_draft_tokens: usize,
}

/// Greedy-only skip-layer speculative decode on the production paged KV path.
///
/// This mirrors [`speculative_decode_step`] but avoids the flat `KvCache` path
/// and skips probability-vector materialization that greedy acceptance does not
/// use. On rejection it restores both base and draft linear-attention state,
/// then replays only the committed prefix so the next step is exact.
#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step_paged_greedy(
    backend: &dyn BackendRuntime,
    last_token: TokenId,
    weights: &GpuWeights,
    config: &ModelConfig,
    paged_cache: &mut PagedKvCache,
    block_table: &BlockTable,
    base_pos: usize,
    linear_state: &mut LinearAttentionState,
    draft_linear_state: &mut LinearAttentionState,
    spec_config: &SpeculativeConfig,
    params: &SamplingParams,
    eos_token_ids: &[TokenId],
    lora: Option<&crate::lora_loader::LoraWeights>,
) -> Result<PagedSpeculativeStepResult> {
    anyhow::ensure!(
        params.temperature == 0.0,
        "paged skip-layer speculative decode is greedy-only"
    );

    let k = spec_config.num_speculative_tokens;
    let base_state_snapshot = linear_state
        .snapshot_for_decode_rollback()
        .context("snapshot base linear attention state before paged skip-layer verify")?;
    let draft_state_snapshot = draft_linear_state
        .snapshot_for_decode_rollback()
        .context("snapshot draft linear attention state before paged skip-layer draft")?;

    let mut draft_tokens: Vec<TokenId> = Vec::with_capacity(k);
    let mut current_token = last_token;
    for _ in 0..k {
        let draft_logits = draft_forward(
            backend,
            &[current_token],
            weights,
            config,
            spec_config.draft_layers,
            draft_linear_state,
        )
        .context("paged skip-layer draft forward pass failed")?;
        let draft_token = greedy_sample(&draft_logits)?;
        draft_tokens.push(draft_token);
        current_token = draft_token;
    }

    let mut verify_input: Vec<TokenId> = Vec::with_capacity(k + 1);
    verify_input.push(last_token);
    verify_input.extend_from_slice(&draft_tokens);
    let verify_logits = model_forward_paged(
        backend,
        &verify_input,
        weights,
        config,
        paged_cache,
        block_table,
        base_pos,
        Some(linear_state),
        lora,
        None,
    )
    .context("paged skip-layer verification forward pass failed")?;
    let target_tokens = greedy_sample_rows(&verify_logits)
        .context("paged skip-layer batched verification greedy sampling failed")?;
    anyhow::ensure!(
        target_tokens.len() == k + 1,
        "paged skip-layer verifier returned {} target tokens for window {}",
        target_tokens.len(),
        k + 1
    );

    let mut accepted_tokens: Vec<TokenId> = Vec::with_capacity(k + 1);
    let mut hit_eos = false;
    let mut accepted_draft_tokens = 0usize;
    let mut replay_input_len: Option<usize> = None;

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let target_token = target_tokens[i];

        if target_token == draft_token {
            accepted_draft_tokens += 1;
            if eos_token_ids.contains(&draft_token) {
                hit_eos = true;
                break;
            }
            accepted_tokens.push(draft_token);
        } else {
            replay_input_len = Some(i + 1);
            if eos_token_ids.contains(&target_token) {
                hit_eos = true;
            } else {
                accepted_tokens.push(target_token);
            }
            break;
        }
    }

    if accepted_draft_tokens == k && !hit_eos {
        let bonus_token = target_tokens[k];
        if eos_token_ids.contains(&bonus_token) {
            hit_eos = true;
        } else {
            accepted_tokens.push(bonus_token);
        }
    }

    if let Some(replay_len) = replay_input_len {
        linear_state
            .restore_from_decode_rollback(&base_state_snapshot)
            .context("restore base linear attention state after paged skip-layer rejection")?;
        draft_linear_state
            .restore_from_decode_rollback(&draft_state_snapshot)
            .context("restore draft linear attention state after paged skip-layer rejection")?;

        let committed_prefix = &verify_input[..replay_len];
        let _ = model_forward_paged(
            backend,
            committed_prefix,
            weights,
            config,
            paged_cache,
            block_table,
            base_pos,
            Some(linear_state),
            lora,
            None,
        )
        .context("replay base prefix after paged skip-layer rejection failed")?;
        draft_forward_for_state_init(
            backend,
            committed_prefix,
            weights,
            config,
            spec_config.draft_layers,
            draft_linear_state,
        )
        .context("replay draft prefix after paged skip-layer rejection failed")?;
    }

    let base_advance = accepted_tokens.len();
    Ok(PagedSpeculativeStepResult {
        accepted_tokens,
        hit_eos,
        base_advance,
        accepted_draft_tokens,
        attempted_draft_tokens: k,
    })
}

/// Result of one native MTP (Multi-Token Prediction) speculative decoding step.
///
/// In addition to the accepted tokens and EOS flag produced by
/// [`SpeculativeStepResult`], MTP also threads a new `h_prev` (post-final-norm
/// hidden state from the base-model verify pass) and advances the MTP
/// position counter independently of the base position counter.
#[derive(Debug)]
pub struct MtpSpeculativeStepResult {
    /// Tokens accepted in this step: up to `[draft, bonus]` on accept, or
    /// `[target_at_0]` on reject.
    pub accepted_tokens: Vec<TokenId>,
    /// Whether an EOS token was encountered in the accepted run.
    pub hit_eos: bool,
    /// Post-final-norm hidden state to feed into the NEXT MTP step.
    ///
    /// On ACCEPT this is the hidden at the draft token's position in the
    /// verify pass, which predicts the bonus. On REJECT this is the hidden
    /// after replaying only `last_token`, so the next draft starts from exact
    /// base-model GDN state.
    pub new_h_prev: Tensor,
    /// How many KV-cache slots the BASE model consumed this step
    /// (`= accepted_tokens.len()` when no EOS cut the step short, and
    /// matches `accepted_tokens.len()` either way: ACCEPT emits 2, REJECT
    /// emits 1). The caller adds this to `base_pos` for the next iteration.
    pub base_advance: usize,
    /// How many KV-cache slots the MTP layer consumed this step. 1 on ACCEPT,
    /// 0 on REJECT. The caller adds this to `mtp_pos`; the rejected draft's
    /// KV write stays in the slot and is overwritten on the next call (which
    /// will target the same position).
    pub mtp_advance: usize,
    /// Whether the draft was accepted (for tracking the acceptance rate α
    /// used by bench reporting). Used exclusively for diagnostics.
    pub draft_accepted: bool,
}

/// Run one native MTP (k=1) speculative decoding step.
///
/// Implements the `draft → verify → accept/reject` pattern from the vLLM
/// `qwen3_next_mtp` reference specialised to k=1 (Qwen3.5-4B ships
/// `num_nextn_predict_layers=1`, so a single MTP draft per iteration is the
/// exact architecture).
///
/// Flow per iteration:
///
/// 1. Draft: `mtp_forward_step(last_token, h_prev, ...)` → `mtp_logits` at
///    `mtp_pos`; greedy sample → `draft_token`.
/// 2. Verify: `model_forward_paged_with_last_hidden([last_token, draft_token],
///    base_cache, base_pos, ...)` → `(verify_logits[1, 2, V], new_hidden[1, 1, H])`.
///    This writes base KV at positions `[base_pos, base_pos + 1]`.
/// 3. Compare: `target_at_0 = argmax(verify_logits[:, 0, :])`; on match,
///    ACCEPT and emit `[draft_token, bonus]` where `bonus =
///    argmax(verify_logits[:, 1, :])`; otherwise REJECT and emit
///    `[target_at_0]`.
/// 4. Advance counters: on ACCEPT, `base_pos += 2`, `mtp_pos += 1`; on REJECT,
///    `base_pos += 1`, `mtp_pos unchanged` (next call overwrites the same MTP
///    KV slot; the base KV at `base_pos + 1` is also written but stale and is
///    overwritten on the next iteration when the corrected token is processed
///    as the new `last_token`).
/// 5. Return the new `h_prev`: the draft position on accept, or the replayed
///    `last_token` position on reject.
///
/// Greedy-only path (temperature == 0). The stochastic rejection-sampling
/// variant is a follow-up; this implementation takes the minimum viable path
/// to produce a correct decode loop plus measurable acceptance rate.
#[allow(clippy::too_many_arguments)]
pub fn speculative_mtp_decode_step(
    backend: &dyn BackendRuntime,
    last_token: TokenId,
    h_prev: &Tensor,
    weights: &GpuWeights,
    config: &ModelConfig,
    base_cache: &mut PagedKvCache,
    base_block_table: &BlockTable,
    base_pos: usize,
    linear_state: &mut LinearAttentionState,
    mtp_cache: &mut PagedKvCache,
    mtp_block_table: &BlockTable,
    mtp_pos: usize,
    _params: &SamplingParams,
    eos_token_ids: &[TokenId],
    _rng: &mut StdRng,
) -> Result<MtpSpeculativeStepResult> {
    // 1. Draft: one MTP step produces a single candidate next token.
    //
    // We pass both `base_pos` (the absolute sequence index the draft token
    // will occupy in the base stream) and `mtp_pos` (the MTP-cache slot
    // index). The inner block uses `base_pos + mtp_pos` for RoPE — so the
    // MTP head sees the same rotation angles the main GQA block would have
    // applied at that absolute position — and keeps `mtp_pos` as the
    // write slot in the isolated MTP paged cache. See Phase B7a evidence
    // in PR #276: without this, `post_layer` drifts monotonically with
    // `mtp_pos` (cos_sim 0.999977 → 0.999531 → 0.997527 at pos 0/1/2).
    let (mtp_logits, _mtp_hidden) = mtp_forward_step(
        backend,
        last_token,
        h_prev,
        weights,
        config,
        mtp_cache,
        mtp_block_table,
        base_pos,
        mtp_pos,
    )
    .context("mtp draft step failed")?;
    let draft_token = greedy_sample(&mtp_logits).context("mtp draft sampling failed")?;

    // 2. Verify the draft with one two-token base-model pass. This is the only
    // k=1 MTP shape that can amortize base-model overhead: on accept the pass
    // both verifies `draft_token` and produces the bonus-token logits. On
    // rejection we restore the GDN state snapshot and replay the one committed
    // token so the base recurrent state remains exact.
    let linear_state_snapshot = linear_state
        .snapshot_for_decode_rollback()
        .context("snapshot linear attention state before MTP verify")?;
    let verify_input = [last_token, draft_token];
    let (verify_logits, hidden_after_draft) = model_forward_paged_with_last_hidden(
        backend,
        &verify_input,
        weights,
        config,
        base_cache,
        base_block_table,
        base_pos,
        Some(linear_state),
        None, // no LoRA on the verify pass — keep parity with scaffolding.
        None, // positions_gpu: let the forward pass build positions internally.
    )
    .context("mtp two-token verify forward failed")?;

    // Position 0 predicts what follows `last_token`; position 1 predicts what
    // follows the accepted draft token (the speculative bonus).
    let verify_pos0 = verify_logits.narrow(1, 0, 1)?.squeeze(1)?;
    let target_at_0 = greedy_sample(&verify_pos0).context("verify pos-0 sampling failed")?;

    // 3. Accept / reject decision. Greedy compare.
    let mut accepted_tokens: Vec<TokenId> = Vec::new();
    let mut hit_eos = false;
    let draft_accepted = target_at_0 == draft_token;

    // Phase C1 — MTP acceptance-rate attribution (greedy).
    // Under greedy decoding `accepted == (mtp_top1 == main_top1)` must hold
    // by construction; recording both independently lets the CSV analyzer
    // verify that invariant and attribute any low α to either an MTP head
    // bug (tokens disagree) or a verification/sampling bug (tokens agree
    // but accept flips false). Off by default; enabled with
    // `KILN_C1_ATTR_PATH=<path>`.
    if c1_attr::is_enabled() {
        // top_k=1 extraction is ~O(V) on host but only runs when the env
        // var is set, so production decode pays nothing.
        let draft_top1 = crate::mtp_debug::top_k_logits(&mtp_logits, 1);
        let main_top1 = crate::mtp_debug::top_k_logits(&verify_pos0, 1);
        let (mtp_top1_logit, main_top1_logit) = match (draft_top1, main_top1) {
            (Ok(d), Ok(m)) => (
                d.first().map(|p| p.1).unwrap_or(f32::NAN),
                m.first().map(|p| p.1).unwrap_or(f32::NAN),
            ),
            _ => (f32::NAN, f32::NAN),
        };
        c1_attr::push_row(c1_attr::C1Row {
            step_idx: c1_attr::next_step_idx(),
            pos_in_k: 0, // Qwen3.5-4B k=1 MTP: one draft per step.
            base_pos,
            mtp_pos,
            last_token,
            mtp_top1: draft_token,
            mtp_top1_logit,
            main_top1: target_at_0,
            main_top1_logit,
            accepted: draft_accepted,
            topk_match: draft_token == target_at_0,
        });
    }

    // Optional Phase B instrumentation. Off by default; enabled with
    // `KILN_MTP_DEBUG=1`. Logs the verify pos-0 top-5 alongside the draft so
    // a 16-step trace can be diffed against an HF reference run on the same
    // prompt. The `mtp_draft` line emitted from `mtp_forward_step` and this
    // `mtp_verify` line share `mtp_pos` so they can be paired by grep.
    if crate::mtp_debug::is_enabled() {
        let v_top = crate::mtp_debug::top_k_logits(&verify_pos0, 5)
            .map(|t| crate::mtp_debug::format_top_k(&t))
            .unwrap_or_else(|e| format!("<top_k err: {e}>"));
        let v_norm = crate::mtp_debug::tensor_l2_norm(&verify_pos0).unwrap_or(f32::NAN);
        tracing::info!(
            target: "kiln::mtp_debug",
            mtp_pos = mtp_pos,
            base_pos = base_pos,
            last_token = last_token,
            draft_token = draft_token,
            target_at_0 = target_at_0,
            accepted = draft_accepted,
            verify_pos0_l2 = v_norm,
            verify_pos0_top5 = %v_top,
            "mtp_verify"
        );
    }

    let mut new_h_prev = hidden_after_draft.clone();
    let (base_advance, mtp_advance) = if draft_accepted {
        // ACCEPT: draft_token matches target. Emit [draft, bonus].
        if eos_token_ids.contains(&draft_token) {
            hit_eos = true;
            // Generation stops here, so there is no need to run the bonus
            // forward just to materialize state that will never be read.
            (1, 1)
        } else {
            accepted_tokens.push(draft_token);
            let verify_pos1 = verify_logits.narrow(1, 1, 1)?.squeeze(1)?;
            let bonus = greedy_sample(&verify_pos1).context("bonus sampling failed")?;
            if eos_token_ids.contains(&bonus) {
                hit_eos = true;
            } else {
                accepted_tokens.push(bonus);
            }
            // Base consumed 2 slots (last_token + draft). MTP consumed 1 slot.
            (2, 1)
        }
    } else {
        // Restore to the state before the optimistic two-token verifier, then
        // replay exactly the committed token. The stale base KV written at
        // `base_pos + 1` is outside the next attention window and will be
        // overwritten if that slot becomes live later.
        linear_state
            .restore_from_decode_rollback(&linear_state_snapshot)
            .context("restore linear attention state after MTP rejection")?;
        let verify_input0 = [last_token];
        let (_verify_logits0, hidden_after_last) = model_forward_paged_with_last_hidden(
            backend,
            &verify_input0,
            weights,
            config,
            base_cache,
            base_block_table,
            base_pos,
            Some(linear_state),
            None,
            None,
        )
        .context("mtp rejection replay forward failed")?;
        new_h_prev = hidden_after_last;

        // REJECT: emit the target's token at position 0.
        if eos_token_ids.contains(&target_at_0) {
            hit_eos = true;
        } else {
            accepted_tokens.push(target_at_0);
        }
        // Base consumed exactly last_token. The rejected draft was never fed
        // through the base model, so no recurrent-state snapshot/restore is
        // needed and the KV cache has no stale base_pos+1 slot.
        (1, 0)
    };

    Ok(MtpSpeculativeStepResult {
        accepted_tokens,
        hit_eos,
        new_h_prev,
        base_advance,
        mtp_advance,
        draft_accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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
            assert!(
                accepted,
                "token 0 should always be accepted when target concentrates on it"
            );
        }

        // Token 1 should always be rejected (p_target = 0)
        for _ in 0..50 {
            let (accepted, resampled) =
                rejection_sample(1, &draft_probs, &target_probs, &mut rng).unwrap();
            assert!(
                !accepted,
                "token 1 should always be rejected when target assigns 0 prob"
            );
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
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
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
