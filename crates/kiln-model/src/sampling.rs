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

/// Sample one decode step with full Qwen3.5 sampling support. Wraps
/// [`sample_with_full_params`] with a stepwise seed override so callers
/// can advance the seed per token without cloning `SamplingParams` at
/// every call site.
///
/// `step_seed` overrides `params.seed` for this step only (it does NOT
/// mutate `params`). Pass `None` for unseeded sampling.
/// `history` is the slice of generated token ids so far — pass `&[]`
/// for the first decode token (no penalties apply).
pub fn sample_step(
    logits: &Tensor,
    params: &kiln_core::sampling::SamplingParams,
    step_seed: Option<u64>,
    history: &[u32],
) -> Result<u32> {
    if params.is_effectively_greedy() {
        return greedy_sample(logits);
    }
    // Cheap struct copy — the only field that ever differs per step is
    // `seed`. Cloning the small `Vec<String>` stop list is negligible
    // compared to the forward pass that produced the logits.
    let mut effective = params.clone();
    effective.seed = step_seed;
    sample_with_full_params(logits, &effective, history)
}

/// Comprehensive sampler. Applies, in order:
///
/// 1. Repetition penalty (HF-style, sign-conditional).
/// 2. Presence + frequency penalty (OpenAI-style, subtractive).
/// 3. Temperature scaling.
/// 4. Top-k filtering.
/// 5. Min-p filtering.
/// 6. Top-p (nucleus) filtering.
/// 7. Categorical sample.
///
/// **Performance**: the penalties are applied on-device via `index_add`
/// over the small set of *unique* history token ids — typical history
/// is hundreds of tokens with maybe ~tens of unique ids active at any
/// given step, so the on-device work is microseconds. After the
/// penalty pass the sampler routes through the existing fast top-k
/// path, transferring only top_k (idx, value) pairs to host. The
/// host-side work that remains (softmax over top_k, min_p, top_p,
/// categorical) operates on tens of values, not the full vocab.
///
/// When every penalty is a no-op AND `min_p == 0`, this short-circuits
/// to the legacy [`sample_with_params`] for byte-identical behavior
/// with the pre-Qwen3.5 sampler.
pub fn sample_with_full_params(
    logits: &Tensor,
    params: &kiln_core::sampling::SamplingParams,
    token_history: &[u32],
) -> Result<u32> {
    use kiln_core::sampling::SamplingParams as SP;

    if params.is_effectively_greedy() {
        return greedy_sample(logits);
    }
    let penalties_no_op = params.token_penalties_are_no_op() || token_history.is_empty();
    let min_p_no_op = SP::min_p_is_disabled(params.min_p);
    if penalties_no_op && min_p_no_op {
        return sample_with_params(
            logits,
            params.temperature,
            params.top_p,
            params.top_k,
            params.seed,
        );
    }

    // Apply penalties on-device — produces a new logits tensor with
    // history-token logits adjusted. The bulk of the vocab is
    // untouched, so this scales with |unique history| not |vocab|.
    let adjusted_logits = if penalties_no_op {
        last_position_logits(logits)?
    } else {
        apply_penalties_on_device(
            logits,
            token_history,
            params.repetition_penalty,
            params.presence_penalty,
            params.frequency_penalty,
        )?
    };

    // Now route through the existing top-k fast path with min_p added
    // as a post-top-k host-side filter.
    sample_from_adjusted_logits(
        &adjusted_logits,
        params.temperature,
        params.top_p,
        params.top_k,
        params.min_p,
        params.seed,
    )
}

/// Apply repetition / presence / frequency penalties to the logits on
/// the device that holds them. Returns a new tensor of the same shape
/// as `last_position_logits(logits)`.
///
/// Strategy: gather the small slice of current logits at the unique
/// history token ids, compute the post-penalty values on host (a few
/// hundred floats max), then index_add the delta back into the logits
/// tensor on-device.
fn apply_penalties_on_device(
    logits: &Tensor,
    history: &[u32],
    repetition: f32,
    presence: f32,
    frequency: f32,
) -> Result<Tensor> {
    let flat = last_position_logits(logits)?;
    let flat = flat.to_dtype(DType::F32)?;
    let device = flat.device().clone();

    // Count unique history tokens.
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::with_capacity(history.len());
    for &t in history {
        *counts.entry(t).or_default() += 1;
    }
    if counts.is_empty() {
        return Ok(flat);
    }
    // Stable ordering for the indices tensor — keeps the on-device
    // scatter deterministic across runs.
    let mut unique: Vec<u32> = counts.keys().copied().collect();
    unique.sort_unstable();

    // Gather current logit values for those token ids.
    let indices = Tensor::new(unique.as_slice(), &device)?;
    let current: Vec<f32> = flat.index_select(&indices, 0)?.to_vec1()?;

    let rep_active = repetition.is_finite()
        && repetition > 0.0
        && (repetition - 1.0).abs() > f32::EPSILON;
    let presence_active = presence.is_finite() && presence != 0.0;
    let frequency_active = frequency.is_finite() && frequency != 0.0;

    let mut deltas: Vec<f32> = Vec::with_capacity(unique.len());
    for (i, &tok) in unique.iter().enumerate() {
        let orig = current[i];
        let mut new = orig;
        if rep_active {
            new = if new > 0.0 { new / repetition } else { new * repetition };
        }
        if presence_active {
            new -= presence;
        }
        if frequency_active {
            let count = counts.get(&tok).copied().unwrap_or(0);
            new -= frequency * count as f32;
        }
        deltas.push(new - orig);
    }

    let delta_tensor = Tensor::new(deltas.as_slice(), &device)?;
    // `index_add` returns a new tensor with `source` added at the given
    // `indices` along dim 0. Available on every backend candle ships
    // (CPU, CUDA, Metal, Vulkan-via-candle), so no backend-specific
    // branching needed here.
    Ok(flat.index_add(&indices, &delta_tensor, 0)?)
}

/// Sample from an already-temperature-pre-scaling logits tensor with
/// the standard top-k → softmax → min_p → top_p → categorical pipeline.
/// Mirrors the legacy [`sample_with_params`] fast paths bit-for-bit
/// when `min_p == 0`, then adds the host-side min_p filter on the
/// truncated top-k subset.
fn sample_from_adjusted_logits(
    flat_logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    seed: Option<u64>,
) -> Result<u32> {
    use kiln_core::sampling::SamplingParams as SP;
    if SP::values_are_effectively_greedy(temperature, top_k) {
        return greedy_sample(flat_logits);
    }
    let vocab_size = flat_logits.dims1()?;

    let scaled = flat_logits.affine(1.0 / temperature as f64, 0.0)?;

    // Default sampling stays on-device for GPU backends when there's no
    // filter active AT ALL — this is the gumbel-softmax fast path.
    let min_p_no_op = SP::min_p_is_disabled(min_p);
    if seed.is_none()
        && SP::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
        && min_p_no_op
        && matches!(scaled.device(), Device::Cuda(_) | Device::Metal(_))
    {
        let sampled = gumbel_softmax(&scaled, 1.0, 0)?;
        return Ok(sampled.to_scalar::<u32>()?);
    }
    if SP::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
        && min_p_no_op
    {
        return sample_full_distribution_unsorted(&scaled, seed);
    }

    // Fetch top-k (idx, logit) pairs — same machinery the legacy
    // sampler uses. Min_p is a host-side filter applied after softmax.
    let indexed: Vec<(u32, f32)> = if top_k > 0 && (top_k as usize) < vocab_size {
        match try_topk_on_device(&scaled, top_k as usize) {
            Ok(pairs) => pairs,
            Err(_) => topk_via_host_sort(&scaled, Some(top_k as usize))?,
        }
    } else {
        topk_via_host_sort(&scaled, None)?
    };
    if indexed.is_empty() {
        anyhow::bail!("no candidates after filtering");
    }

    // Stable softmax over the truncated set.
    let max_logit = indexed[0].1;
    let mut probs: Vec<(u32, f32)> = indexed
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_logit).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Ok(probs.first().map(|&(idx, _)| idx).unwrap_or(0));
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Min-p filtering, applied to the post-softmax top-k subset.
    if !min_p_no_op {
        let pmax = probs.first().map(|&(_, p)| p).unwrap_or(0.0);
        let threshold = min_p * pmax;
        probs.retain(|&(_, p)| p >= threshold);
        if probs.is_empty() {
            return Ok(indexed[0].0);
        }
        let s: f32 = probs.iter().map(|(_, p)| p).sum();
        if s > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= s;
            }
        }
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
        let s: f32 = probs.iter().map(|(_, p)| p).sum();
        if s > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= s;
            }
        }
    }

    // Categorical sample.
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
    Ok(probs.last().context("no candidates after filtering")?.0)
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

/// Pull the full distribution to host and return the top-k `(idx, value)`
/// pairs in descending order. When `top_k` is small (under ~1024 — the
/// regime Qwen3.5's default of 20 always falls into) we use a min-heap
/// based partial selection in O(V log K) instead of a full O(V log V)
/// sort. For Qwen3.5-4B (V=152,064, K=20) this is ~4× fewer comparisons.
///
/// When `top_k` is `None` or large, falls back to the full sort. The
/// CPU/Vulkan fast path benefits the most from this — CUDA/Metal use
/// `try_topk_on_device` which never reaches this fallback under normal
/// operation.
fn topk_via_host_sort(scaled: &Tensor, top_k: Option<usize>) -> Result<Vec<(u32, f32)>> {
    let values: Vec<f32> = scaled.to_vec1()?;
    let vocab = values.len();

    // Heap-based partial selection. Only worth the bookkeeping when
    // k << V; the crossover is roughly k < log2(V) — i.e. for our
    // 152k vocab, k < 17 starts saving meaningful time. We pick a
    // wider threshold of 1024 to also cover users who set top_k higher
    // than Qwen3.5's default of 20.
    if let Some(k) = top_k.filter(|&k| k > 0 && k < vocab && k <= 1024) {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Copy, Clone)]
        struct MinEntry(f32, u32);
        impl PartialEq for MinEntry {
            fn eq(&self, o: &Self) -> bool { self.0 == o.0 && self.1 == o.1 }
        }
        impl Eq for MinEntry {}
        // BinaryHeap is a max-heap; invert ordering so we get a min-heap.
        impl PartialOrd for MinEntry {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for MinEntry {
            fn cmp(&self, o: &Self) -> Ordering {
                // Reversed: smaller value is "greater" in heap terms so
                // that pop() yields the smallest. NaN-safe.
                o.0.partial_cmp(&self.0)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| o.1.cmp(&self.1))
            }
        }

        let mut heap: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(k + 1);
        for (i, &v) in values.iter().enumerate() {
            if heap.len() < k {
                heap.push(MinEntry(v, i as u32));
            } else if let Some(min) = heap.peek() {
                // Push only if this value beats the current heap min.
                if v.partial_cmp(&min.0) == Some(Ordering::Greater) {
                    heap.pop();
                    heap.push(MinEntry(v, i as u32));
                }
            }
        }
        // `into_sorted_vec` returns entries in *ascending* order of our
        // reversed Ord — which is descending by actual value. That's
        // exactly the order we want (largest first), no reversal needed.
        let out: Vec<(u32, f32)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|e| (e.1, e.0))
            .collect();
        return Ok(out);
    }

    // Full sort fallback (top_k unset or larger than the heap threshold).
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

    // ---- sample_with_full_params + penalty tests ---------------------------

    fn full_params_with_seed(
        seed: u64,
    ) -> kiln_core::sampling::SamplingParams {
        kiln_core::sampling::SamplingParams {
            seed: Some(seed),
            ..kiln_core::sampling::SamplingParams::greedy()
        }
    }

    #[test]
    fn test_full_params_greedy_short_circuits() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        let mut params = full_params_with_seed(42);
        params.temperature = 0.0;
        let token = sample_with_full_params(&logits, &params, &[])?;
        assert_eq!(token, 1, "temperature=0 must be greedy");
        Ok(())
    }

    #[test]
    fn test_full_params_no_op_path_matches_legacy() -> Result<()> {
        // With penalties off, min_p=0, the full sampler must produce the
        // same token as the legacy sample_with_params for any given seed.
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], &device)?;
        let mut params = full_params_with_seed(123);
        params.temperature = 1.0;
        params.top_p = 1.0;
        params.top_k = 0;
        params.min_p = 0.0;
        params.repetition_penalty = 1.0;
        params.presence_penalty = 0.0;
        params.frequency_penalty = 0.0;
        let history: Vec<u32> = vec![1, 2, 3]; // ignored when penalties no-op
        let a = sample_with_full_params(&logits, &params, &history)?;
        let b = sample_with_params(&logits, 1.0, 1.0, 0, Some(123))?;
        assert_eq!(a, b, "no-op penalty path must match legacy sampler");
        Ok(())
    }

    #[test]
    fn test_min_p_drops_low_probability_tokens() -> Result<()> {
        let device = Device::Cpu;
        // Token 0 dominates the distribution (~99%); tokens 1-3 are tiny.
        let logits = Tensor::new(&[10.0_f32, 0.0, 0.0, 0.0], &device)?;
        let mut params = full_params_with_seed(7);
        params.temperature = 1.0;
        params.top_p = 1.0;
        params.top_k = 0;
        params.min_p = 0.5; // require >= 50% of max probability
        for seed in 0..20 {
            params.seed = Some(seed);
            let token = sample_with_full_params(&logits, &params, &[])?;
            assert_eq!(token, 0, "min_p=0.5 should drop all tokens but 0");
        }
        Ok(())
    }

    #[test]
    fn test_repetition_penalty_avoids_repeated_token() -> Result<()> {
        let device = Device::Cpu;
        // Token 1 is the natural argmax. With a strong repetition
        // penalty AND token 1 in history, the sampler should prefer
        // another token.
        let logits = Tensor::new(&[2.0_f32, 5.0, 4.0, 1.0], &device)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1e-6; // near-greedy so the result is dominated by the highest logit
        params.top_p = 1.0;
        params.top_k = 0;
        params.repetition_penalty = 100.0; // crush the repeated-token logit
        let token = sample_with_full_params(&logits, &params, &[1])?;
        assert_ne!(
            token, 1,
            "strong repetition penalty must move us off the repeated token"
        );
        Ok(())
    }

    #[test]
    fn test_presence_penalty_suppresses_seen_tokens() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[2.0_f32, 5.0, 4.0, 1.0], &device)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1e-6;
        params.top_p = 1.0;
        params.top_k = 0;
        params.presence_penalty = 10.0; // massive subtraction
        // Token 1 was emitted once before. After presence penalty,
        // logit[1] becomes 5 - 10 = -5, well below logit[2] = 4.
        let token = sample_with_full_params(&logits, &params, &[1])?;
        assert_eq!(
            token, 2,
            "presence penalty should redirect to next-best token"
        );
        Ok(())
    }

    #[test]
    fn test_frequency_penalty_scales_with_count() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[2.0_f32, 5.0, 4.0, 1.0], &device)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1e-6;
        params.top_p = 1.0;
        params.top_k = 0;
        params.frequency_penalty = 1.0;
        // History: token 1 appears 5 times. Logit becomes 5 - 5 = 0;
        // token 2 still at 4 wins.
        let token = sample_with_full_params(&logits, &params, &[1, 1, 1, 1, 1])?;
        assert_eq!(
            token, 2,
            "frequency penalty must scale with count of occurrences"
        );
        Ok(())
    }

    #[test]
    fn test_combined_penalties_compose() -> Result<()> {
        // Confirm that repetition + presence + frequency stack
        // additively on the same token without one stomping the other.
        let device = Device::Cpu;
        let logits = Tensor::new(&[10.0_f32, 1.0, 1.0, 1.0], &device)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1e-6;
        params.top_p = 1.0;
        params.top_k = 0;
        // History: token 0 appears 3 times. With rep=2.0, presence=1.0,
        // frequency=1.0:
        //   logit[0] = (10 / 2) - 1 - 3*1 = 5 - 4 = 1, equal to the others.
        // Sampler should now no longer prefer token 0.
        params.repetition_penalty = 2.0;
        params.presence_penalty = 1.0;
        params.frequency_penalty = 1.0;
        let token = sample_with_full_params(&logits, &params, &[0, 0, 0])?;
        assert_ne!(
            token, 0,
            "combined penalties must dethrone the dominant repeated token"
        );
        Ok(())
    }

    #[test]
    fn test_min_p_combined_with_top_k() -> Result<()> {
        let device = Device::Cpu;
        // 6 tokens with descending logits.
        let logits = Tensor::new(&[5.0_f32, 4.5, 4.0, 1.0, 0.5, 0.0], &device)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1.0;
        params.top_p = 1.0;
        params.top_k = 4;
        params.min_p = 0.3;
        // Top-4 considered: [0, 1, 2, 3]. min_p drops low-prob tail.
        for seed in 0..40 {
            params.seed = Some(seed);
            let token = sample_with_full_params(&logits, &params, &[])?;
            // The very-low-probability tokens (indices 3+) shouldn't
            // make the cut.
            assert!(
                token < 3,
                "min_p combined with top_k must yield a high-prob token (got {token}, seed={seed})"
            );
        }
        Ok(())
    }

    #[test]
    fn test_empty_history_with_penalties_no_op() -> Result<()> {
        // No generated tokens yet → penalties should be inert.
        let device = Device::Cpu;
        let logits = Tensor::new(&[10.0_f32, 1.0, 1.0, 1.0], &device)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1e-6;
        params.repetition_penalty = 100.0;
        params.presence_penalty = 100.0;
        params.frequency_penalty = 100.0;
        let token = sample_with_full_params(&logits, &params, &[])?;
        assert_eq!(token, 0, "empty history must leave logits untouched");
        Ok(())
    }

    #[test]
    fn test_partial_topk_heap_matches_full_sort() -> Result<()> {
        // The heap-based partial-top-k path must produce the same
        // (index, value) pairs as the legacy full-sort path. Run on a
        // realistic-sized vocab to exercise the actual code path.
        let device = Device::Cpu;
        let values: Vec<f32> = (0..152_064)
            .map(|i| ((i as f32) * 0.137).sin() * 7.5 + (i as f32) * 0.0001)
            .collect();
        let logits = Tensor::new(values.as_slice(), &device)?;
        for &k in &[1, 20, 50, 200, 1024] {
            let heap = topk_via_host_sort(&logits, Some(k))?;
            let mut full: Vec<(u32, f32)> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            full.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            full.truncate(k);
            assert_eq!(
                heap.len(),
                full.len(),
                "heap path produced wrong length for k={k}"
            );
            // Compare only the *set* of indices since equal-value
            // ties can break either direction across the two paths.
            let heap_set: std::collections::BTreeSet<u32> =
                heap.iter().map(|&(i, _)| i).collect();
            let full_set: std::collections::BTreeSet<u32> =
                full.iter().map(|&(i, _)| i).collect();
            assert_eq!(
                heap_set, full_set,
                "heap top-k disagrees with full-sort top-k at k={k}"
            );
            // Values should be in descending order.
            for w in heap.windows(2) {
                assert!(
                    w[0].1 >= w[1].1,
                    "heap output not descending at k={k}: {:?}",
                    w
                );
            }
        }
        Ok(())
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_penalty_path_perf_budget() -> Result<()> {
        // The penalty pass for the common Qwen3.5 case (presence_penalty=1.5,
        // a few hundred history tokens, top_k=20) must complete in well
        // under the time of a single decode forward pass. We measure
        // the host-side cost only — real-backend perf is similar or
        // better since `index_add` runs on-device on CUDA/Metal.
        //
        // Debug builds are 10-20× slower than release; the test is
        // gated on release-only via `cfg(not(debug_assertions))`.
        let device = Device::Cpu;
        let values: Vec<f32> = (0..152_064).map(|i| (i as f32 * 0.001).sin()).collect();
        let logits = Tensor::new(values.as_slice(), &device)?;
        let history: Vec<u32> = (0..500).map(|i| (i * 17 + 3) as u32).collect();
        let mut params = full_params_with_seed(1);
        params.temperature = 1.0;
        params.top_p = 0.95;
        params.top_k = 20;
        params.min_p = 0.0;
        params.presence_penalty = 1.5;

        let start = std::time::Instant::now();
        for seed in 0..32 {
            params.seed = Some(seed);
            let _ = sample_with_full_params(&logits, &params, &history)?;
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / 32;
        // 5ms host budget on a 152k vocab is generous. On a modern CPU
        // the partial-top-k heap + small history scatter lands well
        // under 1 ms per token.
        assert!(
            per_call < std::time::Duration::from_millis(5),
            "penalty path took {:?} per call (release-mode budget 5ms) — regression?",
            per_call,
        );
        Ok(())
    }

    #[test]
    fn test_seed_determinism_with_full_params() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 2.5, 0.5, -1.0], &device)?;
        let mut params = full_params_with_seed(42);
        params.temperature = 0.8;
        params.top_p = 0.9;
        params.top_k = 4;
        params.min_p = 0.05;
        params.presence_penalty = 1.5;
        let history = vec![2, 3, 2];
        let a = sample_with_full_params(&logits, &params, &history)?;
        let b = sample_with_full_params(&logits, &params, &history)?;
        assert_eq!(a, b, "same seed must produce same token under full params");
        Ok(())
    }
}
