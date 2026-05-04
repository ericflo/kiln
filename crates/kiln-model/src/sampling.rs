//! Token sampling strategies for autoregressive generation.
//!
//! Provides greedy (argmax) and parameterized (temperature + top-k + top-p) sampling
//! over logits produced by the model forward pass.
//!
//! # On-device sampling
//!
//! Sampling stays on-device as much as possible to eliminate the per-token DtoH stall
//! that dominated decode time (see `PROFILING.md`). Specifically:
//!
//! * Greedy argmax runs on the device, and only the resulting scalar token ID (1 u32,
//!   4 bytes) crosses PCIe. Previously the full `[vocab_size]` logits tensor (608 KB
//!   at vocab=151,936 bf16/f32) was pulled to host every decoded token.
//! * For the top-k path, the logits are sorted on-device and only the top-k
//!   (value, index) pairs are transferred. If the on-device sort fails (e.g. shared
//!   memory limits on very large vocab), we fall back to transferring the full
//!   distribution and sorting on host — behaviourally identical, just slower.
//! * The default full-vocab temperature path can stay on-device on GPU
//!   backends with Gumbel-max + argmax, so only the final scalar token ID
//!   crosses PCIe. CPU keeps the existing host sampler.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::sampling::gumbel_softmax;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Extract the last-position logits from a `[..., vocab_size]` tensor and flatten
/// them to a 1-D `[vocab_size]` tensor that still lives on the original device.
fn last_position_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    let last_logits = if dims.len() >= 2 {
        let seq_len = dims[dims.len() - 2];
        logits
            .narrow(dims.len() - 2, seq_len - 1, 1)?
            .squeeze(dims.len() - 2)?
    } else {
        logits.clone()
    };
    Ok(last_logits.flatten_all()?)
}

/// Greedy sampling: return the token ID with the highest logit for the last position.
///
/// Uses an on-device argmax so that only the scalar token ID (4 bytes) crosses PCIe,
/// eliminating the per-token DtoH stall of the full vocab tensor.
///
/// `logits`: tensor of shape `[..., vocab_size]`. Only the last position is sampled.
///
/// Returns the token ID (index of the maximum logit).
pub fn greedy_sample(logits: &Tensor) -> Result<u32> {
    let flat = last_position_logits(logits)?;
    // Argmax stays on device; only the scalar u32 token ID is transferred to host.
    let idx = flat.argmax(0)?.to_scalar::<u32>()?;
    Ok(idx)
}

/// Greedy sampling for every row in a `[..., vocab_size]` logits tensor.
///
/// This runs one on-device argmax over the vocab dimension and transfers only
/// the resulting token IDs. It is useful for batched verification paths where
/// repeatedly narrowing rows and scalar-sampling would add one device op and
/// one synchronization per verified position.
pub fn greedy_sample_rows(logits: &Tensor) -> Result<Vec<u32>> {
    let dims = logits.dims();
    anyhow::ensure!(!dims.is_empty(), "logits tensor must have at least one dim");
    let vocab_dim = dims.len() - 1;
    let ids = logits.argmax(vocab_dim)?.flatten_all()?;
    Ok(ids.to_vec1::<u32>()?)
}


/// Sample one token for every batch row in a logits tensor.
///
/// Fast paths cover the hot serving cases without per-row synchronization:
/// greedy rows use one argmax over the vocab dimension, while identical
/// unseeded full-distribution sampling uses one batched Gumbel-softmax over the
/// vocab dimension. More specialized top-k/top-p/seeded rows fall back to the
/// scalar sampler for correctness.
pub fn sample_rows_with_params(
    logits: &Tensor,
    params: &[(f32, f32, u32, Option<u64>)],
) -> Result<Vec<u32>> {
    if params.is_empty() {
        return Ok(Vec::new());
    }

    let dims = logits.dims();
    anyhow::ensure!(!dims.is_empty(), "logits tensor must have at least one dim");
    let vocab_dim = dims.len() - 1;
    let vocab_size = dims[vocab_dim];
    let row_count = logits.elem_count() / vocab_size;
    anyhow::ensure!(
        row_count == params.len(),
        "batched sampler row count {row_count} != params length {}",
        params.len()
    );

    if params.iter().all(|&(temperature, _top_p, top_k, _seed)| {
        kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k)
    }) {
        return greedy_sample_rows(logits);
    }

    let (temperature, top_p, top_k, seed) = params[0];
    if seed.is_none()
        && params.iter().all(|&candidate| candidate == params[0])
        && !kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k)
        && kiln_core::sampling::SamplingParams::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
        && matches!(logits.device(), Device::Cuda(_) | Device::Metal(_))
    {
        let scaled = logits
            .to_dtype(DType::F32)?
            .affine(1.0 / temperature as f64, 0.0)?;
        let sampled = gumbel_softmax(&scaled, 1.0, vocab_dim)?.flatten_all()?;
        return Ok(sampled.to_vec1::<u32>()?);
    }

    let mut sampled = Vec::with_capacity(params.len());
    for (row, &(temperature, top_p, top_k, seed)) in params.iter().enumerate() {
        let row_logits = logits.narrow(0, row, 1)?;
        sampled.push(sample_with_params(
            &row_logits,
            temperature,
            top_p,
            top_k,
            seed,
        )?);
    }
    Ok(sampled)
}

/// Parameterized sampling with temperature, top-k, and top-p (nucleus) filtering.
///
/// `logits`: tensor of shape `[..., vocab_size]`. Only the last position is sampled.
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
    if kiln_core::sampling::SamplingParams::values_are_effectively_greedy(temperature, top_k) {
        return greedy_sample(logits);
    }

    let flat = last_position_logits(logits)?;
    let vocab_size = flat.dims1()?;

    // Apply temperature on-device; result stays on the original device.
    let scaled = flat
        .to_dtype(DType::F32)?
        .affine(1.0 / temperature as f64, 0.0)?;

    // Default sampling stays on-device for GPU backends. This keeps the
    // default desktop path off the full-vocab DtoH transfer.
    if seed.is_none()
        && kiln_core::sampling::SamplingParams::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
        && matches!(scaled.device(), Device::Cuda(_) | Device::Metal(_))
    {
        let sampled = gumbel_softmax(&scaled, 1.0, 0)?;
        return Ok(sampled.to_scalar::<u32>()?);
    }

    // Full-vocab temperature sampling with nucleus filtering disabled still
    // needs one host transfer for seeded categorical RNG, but it does not need
    // an O(V log V) sort because no rank-based filtering is requested.
    if kiln_core::sampling::SamplingParams::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
    {
        return sample_full_distribution_unsorted(&scaled, seed);
    }

    // Fetch a descending (index, logit) list, truncated to top_k when active.
    // When top_k selects a real subset of the vocab, we try to sort on-device and
    // transfer only the top_k pairs (e.g. 50 floats + 50 u32 indices = 400 B vs
    // 608 KB for the full vocab at vocab=151,936). If the device sort fails (e.g.
    // shared-memory limits on large vocabs), we fall back to a full-vocab transfer
    // and host sort — correctness is preserved, only the speedup is forfeited.
    let indexed: Vec<(u32, f32)> = if top_k > 0 && (top_k as usize) < vocab_size {
        match try_topk_on_device(&scaled, top_k as usize) {
            Ok(pairs) => pairs,
            Err(_) => topk_via_host_sort(&scaled, Some(top_k as usize))?,
        }
    } else {
        // top_k == 0 (or >= vocab): we need the full distribution on host for the
        // subsequent softmax / top-p / categorical stages.
        topk_via_host_sort(&scaled, None)?
    };

    if indexed.is_empty() {
        anyhow::bail!("no candidates after filtering");
    }

    // Softmax over remaining candidates (numerically stable via max subtraction).
    let max_logit = indexed[0].1;
    let mut probs: Vec<(u32, f32)> = indexed
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p (nucleus) filtering.
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
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }

    // Categorical sampling (host-side; candle has no GPU categorical RNG).
    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };

    let r: f32 = rng.random();
    let mut cumsum = 0.0_f32;
    for &(idx, p) in &probs {
        cumsum += p;
        if r < cumsum {
            return Ok(idx);
        }
    }

    // Numerical edge case: cumsum < r due to rounding. Return the last candidate.
    Ok(probs.last().context("no candidates after filtering")?.0)
}

/// Sample from the full temperature-scaled distribution without sorting.
///
/// This is the fast path for the API/UI defaults: `temperature > 0`,
/// `top_p = 1`, and `top_k = 0`.
fn sample_full_distribution_unsorted(scaled: &Tensor, seed: Option<u64>) -> Result<u32> {
    let values: Vec<f32> = scaled.to_vec1()?;
    if values.is_empty() {
        anyhow::bail!("empty logits distribution");
    }

    let max_logit = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
    if !max_logit.is_finite() {
        return Ok((values.len() - 1) as u32);
    }
    let mut sum = 0.0_f32;
    let mut probs = Vec::with_capacity(values.len());
    for &logit in &values {
        let p = if logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0
        };
        sum += p;
        probs.push(p);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Ok((values.len() - 1) as u32);
    }

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };
    let threshold = rng.random::<f32>() * sum;
    let mut cumsum = 0.0_f32;
    for (idx, p) in probs.into_iter().enumerate() {
        cumsum += p;
        if threshold < cumsum {
            return Ok(idx as u32);
        }
    }

    Ok((values.len() - 1) as u32)
}

/// Sort `scaled` descending on-device, transfer only the top-k `(index, value)`
/// pairs, and return them in descending order.
///
/// Fails if the device sort kernel cannot handle this tensor (e.g. insufficient
/// shared memory for very large last-dim sizes on CUDA). Callers should catch the
/// error and fall back to a host sort over the full vocab.
fn try_topk_on_device(scaled: &Tensor, top_k: usize) -> Result<Vec<(u32, f32)>> {
    // `asc = false` -> descending sort. Returns (sorted_values, sorted_indices).
    let (sorted_vals, sorted_indices) = scaled.sort_last_dim(false)?;
    let top_vals = sorted_vals.narrow(0, 0, top_k)?;
    let top_idx = sorted_indices.narrow(0, 0, top_k)?;
    // Transfer only top_k floats + top_k u32s. Typically 50 * 8 B = 400 B vs 608 KB.
    let values: Vec<f32> = top_vals.to_vec1()?;
    let indices: Vec<u32> = top_idx.to_vec1()?;
    Ok(indices.into_iter().zip(values).collect())
}

/// Pull the full distribution to host, sort descending, and optionally truncate
/// to the top-k entries. Used as a fallback when the on-device sort cannot run.
fn topk_via_host_sort(scaled: &Tensor, top_k: Option<usize>) -> Result<Vec<(u32, f32)>> {
    let values: Vec<f32> = scaled.to_vec1()?;
    let mut indexed: Vec<(u32, f32)> = values
        .into_iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = top_k {
        if k < indexed.len() {
            indexed.truncate(k);
        }
    }
    Ok(indexed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "metal")]
    use crate::backend::metal::try_new_metal;
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
                5.0, 6.0, 7.0, 8.0, // position 1
                0.1, 0.2, 9.0, 0.3, // position 2 (last) — max at index 2
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
                7.0, 5.0, 6.0, // position 1 (last) — max at index 0
            ],
            &device,
        )?
        .reshape((1, 2, 3))?;
        let token = greedy_sample(&logits)?;
        assert_eq!(token, 0); // index of 7.0 in last row
        Ok(())
    }

    #[test]
    fn test_greedy_sample_rows_2d() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[
                0.1_f32, 0.9, 0.2, // max index 1
                3.0, 1.0, 2.0, // max index 0
                -1.0, -0.5, -0.25, // max index 2
            ],
            &device,
        )?
        .reshape((3, 3))?;
        assert_eq!(greedy_sample_rows(&logits)?, vec![1, 0, 2]);
        Ok(())
    }

    #[test]
    fn test_greedy_sample_rows_3d_flattens_prefix_dims() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[
                0.1_f32, 0.9, 0.2, // max index 1
                3.0, 1.0, 2.0, // max index 0
                -1.0, -0.5, -0.25, // max index 2
                4.0, 5.0, 1.0, // max index 1
            ],
            &device,
        )?
        .reshape((2, 2, 3))?;
        assert_eq!(greedy_sample_rows(&logits)?, vec![1, 0, 2, 1]);
        Ok(())
    }

    #[test]
    fn test_greedy_matches_naive_argmax() -> Result<()> {
        // On a realistic-sized vector with distinct values, on-device argmax must
        // match a naive host argmax. This guards the core correctness invariant
        // of the migration away from `to_vec1` + host `max_by`.
        let device = Device::Cpu;
        let values: Vec<f32> = (0..2048)
            .map(|i| ((i as f32) * 0.137).sin() * 7.5 + (i as f32) * 0.001)
            .collect();
        let expected = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0 as u32;
        let logits = Tensor::new(values.as_slice(), &device)?;
        assert_eq!(greedy_sample(&logits)?, expected);
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
    fn test_sample_top_k_one_is_greedy() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        for seed in 0..20 {
            let token = sample_with_params(&logits, 0.8, 0.2, 1, Some(seed))?;
            assert_eq!(
                token, 1,
                "top_k=1 should pick the argmax regardless of seed/top_p"
            );
        }
        Ok(())
    }

    #[test]
    fn test_sample_very_low_temperature_is_near_greedy() -> Result<()> {
        let device = Device::Cpu;
        // With very low temperature, sampling should consistently pick the max
        let logits = Tensor::new(&[1.0_f32, 10.0, 3.0, 2.0], &device)?;
        for seed in 0..20 {
            let token = sample_with_params(&logits, 0.01, 1.0, 0, Some(seed))?;
            assert_eq!(
                token, 1,
                "with temp=0.01 and seed={seed}, expected greedy result"
            );
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
    fn test_top_k_matches_host_topk() -> Result<()> {
        // The on-device sort + narrow path must select the same top-k set as a
        // naive host-side sort over the full vocab.
        let device = Device::Cpu;
        let values: Vec<f32> = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 9.0, 0.5, 6.0, 2.5, 4.5];
        let logits = Tensor::new(values.as_slice(), &device)?;

        // Expected top-3 indices (descending by logit): 9.0->7, 8.0->3, 7.0->5
        let mut expected: Vec<(u32, f32)> = values
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let expected_top3: Vec<u32> = expected.iter().take(3).map(|(i, _)| *i).collect();

        // With top_k=3 and deterministic seeds, every sampled token must lie in
        // the expected top-3 set.
        for seed in 0..80 {
            let token = sample_with_params(&logits, 1.0, 1.0, 3, Some(seed))?;
            assert!(
                expected_top3.contains(&token),
                "top_k=3 produced token {token} outside expected set {expected_top3:?} (seed={seed})"
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
            assert_eq!(
                token, 0,
                "top_p=0.95 with dominant logit should pick token 0, got {token}"
            );
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

    #[test]
    fn test_full_distribution_sampling_tolerates_non_finite_logits() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[f32::NAN, f32::NEG_INFINITY, f32::NAN], &device)?;
        let token = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        assert_eq!(token, 2);
        Ok(())
    }

    #[test]
    fn test_top_p_outside_range_is_full_distribution() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 2.5, 0.25, -0.5], &device)?;
        for seed in 0..80 {
            let full = sample_with_params(&logits, 1.0, 1.0, 0, Some(seed))?;
            let zero = sample_with_params(&logits, 1.0, 0.0, 0, Some(seed))?;
            let negative = sample_with_params(&logits, 1.0, -0.5, 0, Some(seed))?;
            let above_one = sample_with_params(&logits, 1.0, 1.5, 0, Some(seed))?;
            assert_eq!(
                zero, full,
                "top_p=0 disables nucleus filtering and should match top_p=1 for seed={seed}"
            );
            assert_eq!(
                negative, full,
                "negative top_p disables nucleus filtering and should match top_p=1 for seed={seed}"
            );
            assert_eq!(
                above_one, full,
                "top_p>1 disables nucleus filtering and should match top_p=1 for seed={seed}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_sample_with_seed_deterministic_with_topk() -> Result<()> {
        // Determinism must also hold when the top-k on-device path is used.
        let device = Device::Cpu;
        let values: Vec<f32> = (0..512).map(|i| (i as f32 * 0.09).cos() * 3.0).collect();
        let logits = Tensor::new(values.as_slice(), &device)?;
        for seed in [1_u64, 42, 7777, 123456] {
            let a = sample_with_params(&logits, 1.0, 0.9, 50, Some(seed))?;
            let b = sample_with_params(&logits, 1.0, 0.9, 50, Some(seed))?;
            assert_eq!(a, b, "same seed must produce same token (seed={seed})");
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_default_path_seed_is_deterministic_on_metal() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        let a = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        let b = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        assert_eq!(
            a, b,
            "same seed must yield same token on Metal default path"
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_default_path_no_seed_samples_on_metal() -> Result<()> {
        let Some(device) = try_new_metal() else {
            return Ok(());
        };
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        let token = sample_with_params(&logits, 1.0, 1.0, 0, None)?;
        assert!(token < 4, "sampled token out of range: {token}");
        Ok(())
    }
}
