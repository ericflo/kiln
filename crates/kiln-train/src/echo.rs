//! ECHO — Environment Cross-entropy Hybrid Objective.
//!
//! Adds a length-normalized cross-entropy loss on environment-observation
//! tokens to the standard GRPO policy-gradient loss on action tokens. Shares
//! the same forward pass as the policy update; differs only in which mask
//! gathers the logits.
//!
//! Paper: `docs/papers/echo/echo_paper.md` (Shrivastava, Awadallah,
//! Papailiopoulos — MSR AI Frontiers, 2026).
//! Plan: `docs/plans/echo-integration-plan.md` §3.1, §B.3.
//!
//! ## Loss math
//!
//! For an agentic rollout with token sequence `x_{1..T}`:
//!   - `action_mask` marks positions where the model generated tokens
//!     (assistant turns) — targets of GRPO policy gradient.
//!   - `env_mask` marks positions where the environment produced tokens
//!     (tool results, command output) — targets of ECHO env-CE.
//!
//! The env-CE term is
//!
//!     L_env(θ; O') = - (1 / |O|) · Σ_{t ∈ O'} log p_θ(x_t | x_{<t})
//!
//! where `O'` is the env_mask-active positions (after warning_filter trim)
//! and `|O|` is the **total** observation length (including filtered-out
//! warnings; paper §3.1). The division by `|O|` rather than `|O'|` makes
//! the term auto-anneal as the model learns terminal structure — `L_env`
//! falls fast, the term shrinks naturally without a schedule.
//!
//! ## Implementation
//!
//! `echo_step_loss` calls `kiln_flce_kernel::fused_linear_cross_entropy_dispatch`
//! with `env_mask` as the label mask. The FLCE kernel chunks the vocab-dim
//! matmul (default chunk 4096) and never materializes the full `[T, V]`
//! logit tensor, so peak memory stays roughly equal to the GRPO baseline.
//! The kernel returns the mean over active positions, which `echo_step_loss`
//! rescales by `|O'| / |O|` to produce the paper §3.1 normalization.
//!
//! Backend coverage is inherited from the kernel: CUDA Phase B (CustomOp1
//! with analytic backward), Metal Phase B, Vulkan Phase B, CPU Phase A.
//! Vulkan-native training in `crate::vk_train` runs on `VkTensor`s, not
//! candle tensors, and has its own ECHO implementation (Phase 1 follow-up).
//!
//! ## Shape mirror with OPD
//!
//! Signature deliberately mirrors `crate::opd::opd_step_loss` so the
//! `LossConfig { policy, echo, opd }` composition is structural: two
//! parallel `step_loss` calls with parallel inputs and outputs.

use anyhow::{Context, Result};
use candle_core::Tensor;

use kiln_flce_kernel::{
    DEFAULT_CHUNK_SIZE, FlceProvider, fused_linear_cross_entropy_dispatch_with_provider,
};

/// Inputs to the ECHO auxiliary cross-entropy loss term.
///
/// Constructed by the trainer per-rollout from a `TokenizedGrpoCompletion`
/// (the `input_ids`, `env_mask`, `total_obs_len` fields) plus the
/// per-rollout `student_hidden` (post-final-RMSNorm hidden states) and
/// `head_t` (LM head transposed weight). The matmul provider is optional
/// — used to route the per-chunk matmul through the active backend's
/// fast path (Vulkan / CUDA / Metal). Without a provider, the kernel
/// falls back to candle's portable matmul.
pub struct EchoStepInputs<'a> {
    /// Token IDs for the full conversation. Used by FLCE for the
    /// next-token-shift target lookup.
    pub tokens: &'a [u32],
    /// True at env-observation positions that contribute to the loss.
    /// Length == `tokens.len()`. The kernel applies the standard
    /// next-token shift internally (positions `i` where `env_mask[i+1]`
    /// is true contribute the CE for predicting `tokens[i+1]` from
    /// `hidden[i]`).
    pub env_mask: &'a [bool],
    /// Post-final-RMSNorm hidden states from the policy forward.
    /// Shape: `[1, seq_len, hidden_size]`.
    pub student_hidden: &'a Tensor,
    /// LM head weight transposed. Shape: `[hidden_size, vocab_size]`.
    /// Matches kiln's `embed_tokens_t` layout.
    pub head_t: &'a Tensor,
    /// Total observation length `|O|` for paper §3.1 length normalization.
    /// Counts every Observation segment token regardless of warning_filter
    /// trimming. When `env_mask` is a strict subset of the observation
    /// segments (warning_filter active), `|O'| < |O|`; the rescaling here
    /// keeps the normalization right.
    pub total_obs_len: usize,
    /// Chunk size along the vocab dim. Use
    /// [`kiln_flce_kernel::DEFAULT_CHUNK_SIZE`] (4096) unless tuning.
    pub chunk_size: usize,
    /// Optional per-chunk matmul provider for backend acceleration.
    /// `None` falls back to candle's portable matmul.
    pub provider: Option<FlceProvider>,
}

/// Output of `echo_step_loss`.
///
/// `mean_ce` is the autograd-tracked scalar that the trainer adds to the
/// total loss as `λ_echo · mean_ce`. `env_count` is the number of active
/// env-positions that contributed (i.e. `|O'|` after warning_filter
/// trimming) — used for diagnostics + the `lambda_effective` stream.
#[derive(Debug)]
pub struct EchoStepOutputs {
    /// The scalar env-CE loss term, paper §3.1 normalization applied:
    ///
    ///     mean_ce = - (1 / |O|) · Σ_{t ∈ O'} log p_θ(x_t | x_{<t})
    ///
    /// Autograd-tracked off `student_hidden`'s parents, so
    /// `mean_ce.backward()` flows gradients into the LoRA parameters
    /// the trainer is optimizing.
    pub mean_ce: Tensor,
    /// `|O'|` — number of active env-positions after the next-token shift
    /// and warning_filter trim. Used for diagnostics; not required for the
    /// loss math.
    pub env_count: usize,
}

/// Compute the ECHO auxiliary cross-entropy loss term.
///
/// Returns `Ok(None)` when there are no active env-positions (legacy
/// single-turn rollouts, or trajectories whose Observation segments are
/// fully consumed by the warning_filter trim). The caller short-circuits
/// this case to skip the FLCE call entirely — there's no point invoking
/// a kernel on an empty mask, and adding `0.0 · 0.0` to the loss is just
/// noise on the gradient bookkeeping.
///
/// Returns `Ok(Some(EchoStepOutputs))` when at least one position
/// contributes; the trainer should add `λ_echo · outputs.mean_ce` to the
/// total loss before `.backward()`.
pub fn echo_step_loss(inputs: EchoStepInputs<'_>) -> Result<Option<EchoStepOutputs>> {
    let EchoStepInputs {
        tokens,
        env_mask,
        student_hidden,
        head_t,
        total_obs_len,
        chunk_size,
        provider,
    } = inputs;

    anyhow::ensure!(
        tokens.len() == env_mask.len(),
        "echo_step_loss: tokens / env_mask length mismatch ({} vs {})",
        tokens.len(),
        env_mask.len()
    );

    // The kernel applies a next-token shift internally: position `i`
    // contributes the CE for predicting `tokens[i+1]` when
    // `env_mask[i+1] == true`. We count the same way here so the rescale
    // matches the kernel's output.
    let shifted = env_mask.get(1..).unwrap_or(&[]);
    let env_count = shifted.iter().filter(|&&b| b).count();
    if env_count == 0 {
        return Ok(None);
    }
    if total_obs_len == 0 {
        // Trajectory has env-mask bits but |O|=0; that's a builder bug.
        // Refuse rather than divide by zero.
        anyhow::bail!(
            "echo_step_loss: env_count={env_count} but total_obs_len=0 — \
             trajectory_mask should record full Observation segment widths"
        );
    }

    let chunk = if chunk_size == 0 {
        DEFAULT_CHUNK_SIZE
    } else {
        chunk_size
    };

    let device = student_hidden.device();
    let mean_over_active = fused_linear_cross_entropy_dispatch_with_provider(
        student_hidden,
        head_t,
        tokens,
        env_mask,
        device,
        chunk,
        provider,
    )
    .context("FLCE dispatch for ECHO env-CE")?;

    // The kernel returns the mean over `env_count` active positions:
    //     mean_over_active = (1 / |O'|) · Σ_{t ∈ O'} CE_t
    // Paper §3.1 wants normalization by |O|:
    //     paper_mean_ce = (1 / |O|) · Σ_{t ∈ O'} CE_t
    //                   = mean_over_active · (|O'| / |O|)
    // The `|O'| / |O|` rescale is an autograd-compatible affine op.
    let scale = env_count as f64 / total_obs_len as f64;
    let mean_ce = mean_over_active
        .affine(scale, 0.0)
        .context("scale ECHO mean_ce by |O'|/|O| for paper §3.1 normalization")?;

    Ok(Some(EchoStepOutputs { mean_ce, env_count }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Smoke test: empty env_mask returns Ok(None) without touching the
    /// FLCE kernel.
    #[test]
    fn echo_step_loss_empty_mask_short_circuits() -> Result<()> {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((1, 4, 8), DType::F32, &device)?;
        let head_t = Tensor::zeros((8, 16), DType::F32, &device)?;
        let tokens = vec![0u32, 1, 2, 3];
        let env_mask = vec![false; 4];
        let result = echo_step_loss(EchoStepInputs {
            tokens: &tokens,
            env_mask: &env_mask,
            student_hidden: &hidden,
            head_t: &head_t,
            total_obs_len: 0,
            chunk_size: 0,
            provider: None,
        })?;
        assert!(result.is_none());
        Ok(())
    }

    /// Length mismatch is caught with a clear error.
    #[test]
    fn echo_step_loss_length_mismatch_errors() -> Result<()> {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((1, 4, 8), DType::F32, &device)?;
        let head_t = Tensor::zeros((8, 16), DType::F32, &device)?;
        let tokens = vec![0u32, 1, 2, 3];
        let env_mask = vec![false, true, false]; // length 3 != 4
        let err = echo_step_loss(EchoStepInputs {
            tokens: &tokens,
            env_mask: &env_mask,
            student_hidden: &hidden,
            head_t: &head_t,
            total_obs_len: 1,
            chunk_size: 0,
            provider: None,
        })
        .unwrap_err();
        assert!(err.to_string().contains("length mismatch"));
        Ok(())
    }

    /// total_obs_len=0 with active env_mask is a builder bug; refuse.
    #[test]
    fn echo_step_loss_zero_obs_len_with_active_mask_errors() -> Result<()> {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((1, 4, 8), DType::F32, &device)?;
        let head_t = Tensor::zeros((8, 16), DType::F32, &device)?;
        let tokens = vec![0u32, 1, 2, 3];
        // Bit at index 2 is active under next-token shift (env_mask[i+1] true at i+1=2).
        let env_mask = vec![false, false, true, false];
        let err = echo_step_loss(EchoStepInputs {
            tokens: &tokens,
            env_mask: &env_mask,
            student_hidden: &hidden,
            head_t: &head_t,
            total_obs_len: 0,
            chunk_size: 0,
            provider: None,
        })
        .unwrap_err();
        assert!(err.to_string().contains("total_obs_len=0"));
        Ok(())
    }

    /// End-to-end: build a small hidden / head_t pair, mark a few env
    /// positions, verify the returned mean_ce is finite and that the
    /// normalization rescale fires (mean_ce should differ from the raw
    /// kernel output when total_obs_len != env_count).
    #[test]
    fn echo_step_loss_paper_normalization_rescales_correctly() -> Result<()> {
        let device = Device::Cpu;
        let seq_len = 8;
        let hidden_size = 4;
        let vocab = 8;

        // Tiny random hidden + head. Use a fixed pattern so the test is
        // deterministic.
        let hidden_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32) * 0.01).collect();
        let hidden = Tensor::from_vec(hidden_data, (1, seq_len, hidden_size), &device)?
            .to_dtype(DType::F32)?;
        let head_data: Vec<f32> = (0..hidden_size * vocab).map(|i| (i as f32) * 0.1).collect();
        let head_t = Tensor::from_vec(head_data, (hidden_size, vocab), &device)?
            .to_dtype(DType::F32)?;

        let tokens: Vec<u32> = (0..seq_len).map(|i| (i as u32) % vocab as u32).collect();

        // 4 env positions out of seq_len=8 (after next-token shift, 4 active rows).
        let env_mask = vec![false, true, true, false, false, true, true, false];

        // Test 1: env_count == total_obs_len. Scale is 1.0, paper_mean == kernel_mean.
        let out_eq = echo_step_loss(EchoStepInputs {
            tokens: &tokens,
            env_mask: &env_mask,
            student_hidden: &hidden,
            head_t: &head_t,
            total_obs_len: 4,
            chunk_size: 0,
            provider: None,
        })?
        .unwrap();
        assert_eq!(out_eq.env_count, 4);
        let mean_eq = out_eq.mean_ce.to_scalar::<f32>()?;
        assert!(mean_eq.is_finite() && mean_eq > 0.0);

        // Test 2: total_obs_len = 2 * env_count. Paper mean should be half
        // the kernel mean (scale = 4/8 = 0.5).
        let out_half = echo_step_loss(EchoStepInputs {
            tokens: &tokens,
            env_mask: &env_mask,
            student_hidden: &hidden,
            head_t: &head_t,
            total_obs_len: 8,
            chunk_size: 0,
            provider: None,
        })?
        .unwrap();
        let mean_half = out_half.mean_ce.to_scalar::<f32>()?;
        let ratio = mean_eq / mean_half;
        assert!(
            (ratio - 2.0).abs() < 1e-4,
            "expected 2x ratio, got {ratio} (mean_eq={mean_eq}, mean_half={mean_half})"
        );
        Ok(())
    }
}
