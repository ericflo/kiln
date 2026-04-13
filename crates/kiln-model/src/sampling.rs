//! Token sampling strategies for autoregressive generation.
//!
//! Provides greedy (argmax) and parameterized (temperature + top-k + top-p) sampling
//! over logits produced by the model forward pass.

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Greedy sampling: return the token ID with the highest logit for the last position.
///
/// `logits`: tensor of shape [..., vocab_size]. Only the last position is sampled.
///
/// Returns the token ID (index of the maximum logit).
pub fn greedy_sample(logits: &Tensor) -> Result<u32> {
    let dims = logits.dims();
    let _vocab_size = *dims.last().context("logits must have at least 1 dimension")?;

    // Extract logits for the last position
    let last_logits = if dims.len() >= 2 {
        let seq_len = dims[dims.len() - 2];
        logits
            .narrow(dims.len() - 2, seq_len - 1, 1)?
            .squeeze(dims.len() - 2)?
    } else {
        logits.clone()
    };

    // Flatten to 1-D [vocab_size]
    let flat = last_logits.flatten_all()?;
    let vals = flat.to_vec1::<f32>().or_else(|_| {
        flat.to_dtype(DType::F32)?.to_vec1::<f32>()
    })?;

    let (max_idx, _) = vals
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .context("empty logits")?;

    Ok(max_idx as u32)
}

/// Parameterized sampling with temperature, top-k, and top-p (nucleus) filtering.
///
/// `logits`: tensor of shape [..., vocab_size]. Only the last position is sampled.
/// `temperature`: scaling factor for logits (lower = more deterministic). If 0.0, uses greedy.
/// `top_p`: nucleus sampling threshold in (0, 1]. Only the smallest set of tokens whose
///          cumulative probability exceeds `top_p` are considered.
/// `top_k`: if > 0, only the top-k highest-probability tokens are considered.
/// `seed`: optional RNG seed for reproducibility.
///
/// Returns the sampled token ID.
pub fn sample_with_params(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: Option<u64>,
) -> Result<u32> {
    if temperature == 0.0 {
        return greedy_sample(logits);
    }

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

    // Flatten to 1-D and convert to f32
    let flat = last_logits.flatten_all()?.to_dtype(DType::F32)?;
    let mut vals = flat.to_vec1::<f32>()?;

    // Apply temperature
    for v in vals.iter_mut() {
        *v /= temperature;
    }

    // Build (index, logit) pairs
    let mut indexed: Vec<(u32, f32)> = vals.iter().copied().enumerate().map(|(i, v)| (i as u32, v)).collect();

    // Sort descending by logit
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k filtering
    if top_k > 0 && (top_k as usize) < indexed.len() {
        indexed.truncate(top_k as usize);
    }

    // Softmax over remaining candidates
    let max_logit = indexed[0].1;
    let mut probs: Vec<(u32, f32)> = indexed
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p (nucleus) filtering
    if top_p > 0.0 && top_p < 1.0 {
        let mut cumsum = 0.0_f32;
        let mut cutoff = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        // Renormalize
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }

    // Categorical sampling
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let r: f32 = rng.r#gen();
    let mut cumsum = 0.0_f32;
    for &(idx, p) in &probs {
        cumsum += p;
        if r < cumsum {
            return Ok(idx);
        }
    }

    // Fallback to last candidate (numerical edge case)
    Ok(probs.last().context("no candidates after filtering")?.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_greedy_sample_1d() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        let token = greedy_sample(&logits)?;
        assert_eq!(token, 1); // index of 5.0
        Ok(())
    }

    #[test]
    fn test_greedy_sample_2d() -> Result<()> {
        let device = Device::Cpu;
        // [seq_len=3, vocab_size=4] — should sample from last position
        let logits = Tensor::new(
            &[
                1.0_f32, 2.0, 3.0, 4.0, // position 0
                5.0, 6.0, 7.0, 8.0,     // position 1
                0.1, 0.2, 9.0, 0.3,     // position 2 (last) — max at index 2
            ],
            &device,
        )?
        .reshape((3, 4))?;
        let token = greedy_sample(&logits)?;
        assert_eq!(token, 2); // index of 9.0 in last row
        Ok(())
    }

    #[test]
    fn test_greedy_sample_3d() -> Result<()> {
        let device = Device::Cpu;
        // [batch=1, seq_len=2, vocab_size=3]
        let logits = Tensor::new(
            &[
                1.0_f32, 2.0, 3.0, // position 0
                7.0, 5.0, 6.0,     // position 1 (last) — max at index 0
            ],
            &device,
        )?
        .reshape((1, 2, 3))?;
        let token = greedy_sample(&logits)?;
        assert_eq!(token, 0); // index of 7.0 in last row
        Ok(())
    }

    #[test]
    fn test_sample_temperature_zero_is_greedy() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        let token = sample_with_params(&logits, 0.0, 1.0, 0, Some(42))?;
        assert_eq!(token, 1); // same as greedy
        Ok(())
    }

    #[test]
    fn test_sample_very_low_temperature_is_near_greedy() -> Result<()> {
        let device = Device::Cpu;
        // With very low temperature, sampling should consistently pick the max
        let logits = Tensor::new(&[1.0_f32, 10.0, 3.0, 2.0], &device)?;
        for seed in 0..20 {
            let token = sample_with_params(&logits, 0.01, 1.0, 0, Some(seed))?;
            assert_eq!(token, 1, "with temp=0.01 and seed={seed}, expected greedy result");
        }
        Ok(())
    }

    #[test]
    fn test_top_k_filtering() -> Result<()> {
        let device = Device::Cpu;
        // With top_k=2, only the 2 highest logits should be candidates
        let logits = Tensor::new(&[1.0_f32, 10.0, 8.0, 2.0, 0.5], &device)?;
        for seed in 0..50 {
            let token = sample_with_params(&logits, 1.0, 1.0, 2, Some(seed))?;
            assert!(
                token == 1 || token == 2,
                "top_k=2 should only produce tokens 1 or 2, got {token} with seed={seed}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_top_p_filtering() -> Result<()> {
        let device = Device::Cpu;
        // Logits designed so that after softmax, token 0 has ~99.5% probability
        let logits = Tensor::new(&[10.0_f32, 0.0, 0.0, 0.0], &device)?;
        for seed in 0..20 {
            let token = sample_with_params(&logits, 1.0, 0.95, 0, Some(seed))?;
            // With top_p=0.95, token 0 alone exceeds the threshold
            assert_eq!(token, 0, "top_p=0.95 with dominant logit should pick token 0, got {token}");
        }
        Ok(())
    }

    #[test]
    fn test_sample_with_seed_is_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 2.5], &device)?;
        let t1 = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        let t2 = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        assert_eq!(t1, t2, "same seed should produce same result");
        Ok(())
    }
}
