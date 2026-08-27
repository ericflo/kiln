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

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

// #1082: bare `Tensor`/`Device`/`DType`/`D` resolve to `kiln_tensor` —
// candle has been removed from the sampler. Mirrors `forward.rs`.
use kiln_tensor::{DType, Device, Tensor};

/// One token sampled from the fully resolved behavior distribution.
///
/// `logprob` is the natural logarithm of the token's probability after
/// penalties, temperature, top-k, min-p, and top-p have all been applied and
/// the surviving distribution has been renormalized. Deterministic sampling
/// has log-probability zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledToken {
    pub token_id: u32,
    pub logprob: f32,
}

impl SampledToken {
    fn deterministic(token_id: u32) -> Self {
        Self {
            token_id,
            logprob: 0.0,
        }
    }
}

pub(crate) fn unique_history_counts_for_batch_sample(history: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts = std::collections::BTreeMap::<u32, u32>::new();
    for &token in history {
        *counts.entry(token).or_default() += 1;
    }
    counts.into_iter().unzip()
}

pub(crate) fn sample_seed_for_batch_row(step_seed: Option<u64>, history: &[u32]) -> u64 {
    step_seed.unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        let history_hash = history.iter().fold(0xCBF29CE484222325u64, |acc, &token| {
            (acc ^ token as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(history_hash)
    })
}

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
    // Phase 7 (#1082): contiguous CUDA logits of {F32, BF16, F16}
    // take the kt 1-D argmax path by default after last-position
    // flattening. Falls through to the generic argmax when any
    // compatibility precondition fails. `flat` is already a kt tensor,
    // so the candle->kt bridge is gone — pass it to the kt helper directly.
    #[cfg(feature = "cuda")]
    if flat.is_contiguous() {
        if let Some(idx) = crate::forward::try_kt_argmax_1d(&flat)? {
            return Ok(idx);
        }
    }
    // ROCm fast path: `argmax` runs on-device (ArgmaxOp::rocm_fwd) and yields an
    // I64 scalar; read it back directly. The generic `.to_dtype(U32)` path below
    // would cast I64->U32, which is NOT in the ROCm cast matrix (F32<->BF16<->F16
    // only), so it host-round-trips every token (D2H I64 -> CPU cast -> H2D U32 ->
    // D2H U32 for the scalar read — an upload we immediately re-download).
    // Reading the I64 scalar directly is a single D2H, no upload.
    #[cfg(feature = "rocm")]
    if matches!(flat.device(), kiln_tensor::Device::Rocm(_)) {
        let indices = flat.argmax(0)?.flatten_all()?;
        let idx_i64 =
            crate::execution_phase::profile_accelerator_readback(indices.device(), || {
                indices.to_vec1::<i64>()
            })?;
        return Ok(idx_i64[0] as u32);
    }
    // Argmax stays on device; only the scalar token ID is transferred to
    // host. kt `argmax` yields an I64 index tensor (no implicit cast on
    // readback), so cast to U32 before the `to_scalar::<u32>()` host read.
    let index = flat.argmax(0)?.to_dtype(DType::U32)?;
    let idx = crate::execution_phase::profile_accelerator_readback(index.device(), || {
        index.to_scalar::<u32>()
    })?;
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
    // Phase 7 (#1082): contiguous CUDA logits of {F32, BF16, F16}
    // with rank >= 1 take the kt row-argmax path by default. This
    // replaces the generic `argmax(vocab_dim) + flatten_all +
    // to_vec1::<u32>()` composite with a single fused kernel + one
    // I64->u32 host copy. Falls through to the generic composite when
    // any compatibility precondition fails. `logits` is already a kt
    // tensor, so the candle->kt bridge is gone.
    #[cfg(feature = "cuda")]
    if logits.is_contiguous() {
        if let Some(ids) = crate::forward::try_kt_sampling_argmax_rows(logits)? {
            return Ok(ids);
        }
    }
    let vocab_dim = dims.len() - 1;
    // ROCm fast path: read the on-device I64 argmax indices directly. The
    // generic `.to_dtype(U32)` below isn't in the ROCm cast matrix and would
    // host-round-trip (upload-then-download) per call. Read I64 and convert
    // host-side instead.
    #[cfg(feature = "rocm")]
    if matches!(logits.device(), kiln_tensor::Device::Rocm(_)) {
        let indices = logits.argmax(vocab_dim)?.flatten_all()?;
        let ids_i64 =
            crate::execution_phase::profile_accelerator_readback(indices.device(), || {
                indices.to_vec1::<i64>()
            })?;
        return Ok(ids_i64.into_iter().map(|v| v as u32).collect());
    }
    // kt `argmax` yields an I64 index tensor; cast to U32 to preserve the
    // `to_vec1::<u32>()` host read path (candle's `argmax` returned U32).
    let ids = logits
        .argmax(vocab_dim)?
        .flatten_all()?
        .to_dtype(DType::U32)?;
    Ok(crate::execution_phase::profile_accelerator_readback(
        ids.device(),
        || ids.to_vec1::<u32>(),
    )?)
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

/// Sample one decode step and return the exact selected-token behavior
/// log-probability. Stochastic traced sampling requires a resolved seed so the
/// token and its provenance can be reproduced independently of process-local
/// entropy sources.
pub fn sample_step_with_logprob(
    logits: &Tensor,
    params: &kiln_core::sampling::SamplingParams,
    step_seed: Option<u64>,
    history: &[u32],
) -> Result<SampledToken> {
    if params.is_effectively_greedy() {
        return Ok(SampledToken::deterministic(greedy_sample(logits)?));
    }
    anyhow::ensure!(
        step_seed.is_some(),
        "sampling with behavior log-probability capture requires a resolved per-step seed"
    );
    let mut effective = params.clone();
    effective.seed = step_seed;
    sample_with_full_params_and_logprob(logits, &effective, history)
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

/// Comprehensive sampling plus the selected token's normalized
/// post-filter log-probability.
///
/// The seeded draw and probability lookup share one materialized effective
/// distribution, so trace capture does not repeat filtering or transfer the
/// candidate set twice.
pub fn sample_with_full_params_and_logprob(
    logits: &Tensor,
    params: &kiln_core::sampling::SamplingParams,
    token_history: &[u32],
) -> Result<SampledToken> {
    if params.is_effectively_greedy() {
        return Ok(SampledToken::deterministic(greedy_sample(logits)?));
    }
    anyhow::ensure!(
        params.seed.is_some(),
        "sampling with behavior log-probability capture requires a resolved seed"
    );
    let seed = params.seed.expect("resolved seed checked above");
    let penalties_no_op = params.token_penalties_are_no_op() || token_history.is_empty();
    let min_p_no_op = kiln_core::sampling::SamplingParams::min_p_is_disabled(params.min_p);
    if penalties_no_op && min_p_no_op {
        return sample_with_params_and_logprob(
            logits,
            params.temperature,
            params.top_p,
            params.top_k,
            seed,
        );
    }
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
    sample_from_adjusted_logits_with_logprob(
        &adjusted_logits,
        params.temperature,
        params.top_p,
        params.top_k,
        params.min_p,
        seed,
    )
}

/// Temperature/top-k/top-p sampling with a resolved seed and exact selected
/// behavior log-probability.
pub fn sample_with_params_and_logprob(
    logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: u64,
) -> Result<SampledToken> {
    use kiln_core::sampling::SamplingParams as SP;

    if SP::values_are_effectively_greedy(temperature, top_k) {
        return Ok(SampledToken::deterministic(greedy_sample(logits)?));
    }
    let flat = last_position_logits(logits)?.to_dtype(DType::F32)?;
    sample_from_adjusted_logits_with_logprob(&flat, temperature, top_p, top_k, 0.0, seed)
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
    let device = flat.device();

    #[cfg(feature = "cuda")]
    if let Some(out) =
        try_kt_apply_penalties_on_device(&flat, history, repetition, presence, frequency)?
    {
        return Ok(out);
    }

    let (counts, unique) = sorted_history_counts(history);
    if counts.is_empty() {
        return Ok(flat);
    }

    // Gather current logit values for those token ids.
    let indices = Tensor::new(unique.as_slice(), device)?;
    let selected = flat.index_select(&indices, 0)?;
    let current: Vec<f32> =
        crate::execution_phase::profile_accelerator_readback(selected.device(), || {
            selected.to_vec1()
        })?;
    let deltas = penalty_deltas(&unique, &counts, &current, repetition, presence, frequency);

    let delta_tensor = Tensor::new(deltas.as_slice(), device)?;
    // `index_add` returns a new tensor with `source` added at the given
    // `indices` along dim 0. kt's `index_add` composes scatter_add + add,
    // both of which dispatch to the holding device (CPU/CUDA/Metal), so
    // no backend-specific branching is needed here.
    Ok(flat.index_add(&indices, &delta_tensor, 0)?)
}

fn sorted_history_counts(history: &[u32]) -> (std::collections::BTreeMap<u32, u32>, Vec<u32>) {
    let mut counts = std::collections::BTreeMap::new();
    for &t in history {
        *counts.entry(t).or_default() += 1;
    }
    let unique = counts.keys().copied().collect();
    (counts, unique)
}

fn penalty_deltas(
    unique: &[u32],
    counts: &std::collections::BTreeMap<u32, u32>,
    current: &[f32],
    repetition: f32,
    presence: f32,
    frequency: f32,
) -> Vec<f32> {
    let rep_active =
        repetition.is_finite() && repetition > 0.0 && (repetition - 1.0).abs() > f32::EPSILON;
    let presence_active = presence.is_finite() && presence != 0.0;
    let frequency_active = frequency.is_finite() && frequency != 0.0;

    let mut deltas = Vec::with_capacity(unique.len());
    for (i, &tok) in unique.iter().enumerate() {
        let orig = current[i];
        let mut new = orig;
        if rep_active {
            new = if new > 0.0 {
                new / repetition
            } else {
                new * repetition
            };
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
    deltas
}

#[cfg(feature = "cuda")]
fn try_kt_apply_penalties_on_device(
    flat: &Tensor,
    history: &[u32],
    repetition: f32,
    presence: f32,
    frequency: f32,
) -> Result<Option<Tensor>> {
    if !matches!(flat.device(), Device::Cuda(_))
        || flat.dtype() != DType::F32
        || !flat.is_contiguous()
        || flat.rank() != 1
    {
        return Ok(None);
    }

    let (counts, unique) = sorted_history_counts(history);
    if counts.is_empty() {
        return Ok(Some(flat.clone()));
    }

    kiln_nvtx::range!(c"kiln/sampling_penalties_kt");

    // #1082: `flat` is already a kt CUDA tensor — no candle bridge. Build
    // the index/delta tensors on the same device and drive the on-device
    // gather + scatter-add directly. `cuda_scatter_add_dim0` mutates
    // `out_kt` in place, so `copy()` an owned deep copy of `flat` to
    // scatter into (a bare `contiguous()`/`clone()` would alias `flat`'s
    // storage since it is already contiguous, corrupting the caller).
    let device = flat.device();
    let indices_kt = Tensor::new(unique.as_slice(), device)?;
    let out_kt = match flat.copy() {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let current_kt = match kiln_tensor::cuda_index_select_dim0(&out_kt, &indices_kt) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let current: Vec<f32> =
        crate::execution_phase::profile_accelerator_readback(current_kt.device(), || {
            current_kt.to_vec1()
        })?;
    let deltas = penalty_deltas(&unique, &counts, &current, repetition, presence, frequency);
    let delta_kt = Tensor::new(deltas.as_slice(), device)?;
    if kiln_tensor::cuda_scatter_add_dim0(&out_kt, &indices_kt, &delta_kt).is_err() {
        return Ok(None);
    }
    Ok(Some(out_kt))
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
    let vocab_size = flat_logits.dims()[0];

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
        let sampled = kiln_tensor::ops::gumbel_softmax_sample(&scaled, 1.0, 0)?;
        return Ok(crate::execution_phase::profile_accelerator_readback(
            sampled.device(),
            || sampled.to_scalar::<u32>(),
        )?);
    }
    if SP::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
        && min_p_no_op
    {
        return sample_full_distribution_unsorted(&scaled, seed);
    }

    filtered_distribution(&scaled, top_p, top_k, min_p)?.sample(seed)
}

fn sample_from_adjusted_logits_with_logprob(
    flat_logits: &Tensor,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    min_p: f32,
    seed: u64,
) -> Result<SampledToken> {
    use kiln_core::sampling::SamplingParams as SP;

    if SP::values_are_effectively_greedy(temperature, top_k) {
        return Ok(SampledToken::deterministic(greedy_sample(flat_logits)?));
    }
    let scaled = flat_logits.affine(1.0 / temperature as f64, 0.0)?;
    let vocab_size = flat_logits.dims()[0];
    let min_p_no_op = SP::min_p_is_disabled(min_p);
    if SP::top_p_disables_nucleus_filter(top_p)
        && (top_k == 0 || top_k as usize >= vocab_size)
        && min_p_no_op
    {
        sample_full_distribution_unsorted_with_logprob(&scaled, seed)
    } else {
        filtered_distribution(&scaled, top_p, top_k, min_p)?.sample_with_logprob(Some(seed))
    }
}

enum EffectiveDistribution {
    Deterministic(u32),
    Categorical(Vec<(u32, f32)>),
}

impl EffectiveDistribution {
    fn sample(&self, seed: Option<u64>) -> Result<u32> {
        match self {
            Self::Deterministic(token_id) => Ok(*token_id),
            Self::Categorical(probs) => {
                let mut rng: StdRng = match seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => rand::make_rng::<StdRng>(),
                };
                let random: f32 = rng.random();
                let mut cumulative = 0.0_f32;
                for &(token_id, probability) in probs {
                    cumulative += probability;
                    if random < cumulative {
                        return Ok(token_id);
                    }
                }
                Ok(probs.last().context("no candidates after filtering")?.0)
            }
        }
    }

    fn sample_with_logprob(&self, seed: Option<u64>) -> Result<SampledToken> {
        match self {
            Self::Deterministic(token_id) => Ok(SampledToken::deterministic(*token_id)),
            Self::Categorical(probs) => {
                let mut rng: StdRng = match seed {
                    Some(seed) => StdRng::seed_from_u64(seed),
                    None => rand::make_rng::<StdRng>(),
                };
                let random: f32 = rng.random();
                let mut cumulative = 0.0_f32;
                for &(token_id, probability) in probs {
                    cumulative += probability;
                    if random < cumulative {
                        return sampled_token_from_probability(token_id, probability);
                    }
                }
                let &(token_id, probability) =
                    probs.last().context("no candidates after filtering")?;
                sampled_token_from_probability(token_id, probability)
            }
        }
    }
}

fn sampled_token_from_probability(token_id: u32, probability: f32) -> Result<SampledToken> {
    anyhow::ensure!(
        probability.is_finite() && probability > 0.0 && probability <= 1.0 + 1e-6,
        "sampled token {token_id} has invalid effective probability {probability}"
    );
    Ok(SampledToken {
        token_id,
        logprob: probability.min(1.0).ln(),
    })
}

fn filtered_distribution(
    scaled: &Tensor,
    top_p: f32,
    top_k: u32,
    min_p: f32,
) -> Result<EffectiveDistribution> {
    use kiln_core::sampling::SamplingParams as SP;

    let vocab_size = scaled.dims()[0];
    let min_p_no_op = SP::min_p_is_disabled(min_p);
    // Fetch top-k (idx, logit) pairs — same machinery the token-only sampler
    // uses. Min-p is a host-side filter applied after softmax.
    let indexed: Vec<(u32, f32)> = if top_k > 0 && (top_k as usize) < vocab_size {
        match try_topk_on_device(scaled, top_k as usize) {
            Ok(pairs) => pairs,
            Err(_) => topk_via_host_sort(scaled, Some(top_k as usize))?,
        }
    } else {
        topk_via_host_sort(scaled, None)?
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
        return Ok(EffectiveDistribution::Deterministic(
            probs.first().map(|&(idx, _)| idx).unwrap_or(0),
        ));
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
            return Ok(EffectiveDistribution::Deterministic(indexed[0].0));
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

    Ok(EffectiveDistribution::Categorical(probs))
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
    let flat = last_position_logits(logits)?.to_dtype(DType::F32)?;
    sample_from_adjusted_logits(&flat, temperature, top_p, top_k, 0.0, seed)
}

/// Sample from the full temperature-scaled distribution without sorting.
///
/// This is the fast path for the API/UI defaults: `temperature > 0`,
/// `top_p = 1`, and `top_k = 0`.
fn sample_full_distribution_unsorted(scaled: &Tensor, seed: Option<u64>) -> Result<u32> {
    let (weights, fallback_idx) = full_distribution_weights(scaled)?;
    let Some(weights) = weights else {
        return Ok(fallback_idx);
    };
    sample_from_distribution_weights(&weights, seed, fallback_idx)
}

fn sample_full_distribution_unsorted_with_logprob(
    scaled: &Tensor,
    seed: u64,
) -> Result<SampledToken> {
    let (weights, fallback_idx) = full_distribution_weights(scaled)?;
    let Some(weights) = weights else {
        return Ok(SampledToken::deterministic(fallback_idx));
    };
    sample_from_distribution_weights_with_logprob(&weights, Some(seed), fallback_idx)
}

fn full_distribution_weights(scaled: &Tensor) -> Result<(Option<Vec<f32>>, u32)> {
    #[cfg(feature = "cuda")]
    if let Some(weights) = try_kt_full_distribution_probs(scaled)? {
        if weights.is_empty() {
            anyhow::bail!("empty logits distribution");
        }
        let fallback_idx = (weights.len() - 1) as u32;
        return Ok((Some(weights), fallback_idx));
    }

    let values: Vec<f32> =
        crate::execution_phase::profile_accelerator_readback(scaled.device(), || scaled.to_vec1())?;
    if values.is_empty() {
        anyhow::bail!("empty logits distribution");
    }

    let fallback_idx = (values.len() - 1) as u32;
    Ok((softmax_probs_from_logits(&values), fallback_idx))
}

fn softmax_probs_from_logits(values: &[f32]) -> Option<Vec<f32>> {
    let max_logit = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
    if !max_logit.is_finite() {
        return None;
    }

    let mut weights = Vec::with_capacity(values.len());
    for &logit in values {
        weights.push(if logit.is_finite() {
            (logit - max_logit).exp()
        } else {
            0.0
        });
    }
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for weight in weights.iter_mut() {
        *weight /= sum;
    }
    Some(weights)
}

fn sample_from_distribution_weights(
    weights: &[f32],
    seed: Option<u64>,
    fallback_idx: u32,
) -> Result<u32> {
    if weights.is_empty() {
        anyhow::bail!("empty logits distribution");
    }
    let sum: f32 = weights
        .iter()
        .copied()
        .filter(|weight| weight.is_finite() && *weight > 0.0)
        .sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Ok(fallback_idx);
    }

    let mut rng: StdRng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => rand::make_rng::<StdRng>(),
    };
    let threshold = rng.random::<f32>() * sum;
    let mut cumulative = 0.0_f32;
    for (index, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        cumulative += weight;
        if threshold < cumulative {
            return Ok(index as u32);
        }
    }
    Ok(fallback_idx)
}

fn sample_from_distribution_weights_with_logprob(
    weights: &[f32],
    seed: Option<u64>,
    fallback_idx: u32,
) -> Result<SampledToken> {
    if weights.is_empty() {
        anyhow::bail!("empty logits distribution");
    }
    let sum: f32 = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Ok(SampledToken::deterministic(fallback_idx));
    }

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => rand::make_rng::<StdRng>(),
    };
    let threshold = rng.random::<f32>() * sum;
    let mut cumsum = 0.0_f32;
    for (idx, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        cumsum += weight;
        if threshold < cumsum {
            return sampled_token_from_probability(idx as u32, weight / sum);
        }
    }

    let fallback_probability = weights
        .get(fallback_idx as usize)
        .copied()
        .unwrap_or_default();
    sampled_token_from_probability(fallback_idx, fallback_probability / sum)
}

#[cfg(feature = "cuda")]
fn try_kt_full_distribution_probs(scaled: &Tensor) -> Result<Option<Vec<f32>>> {
    if !matches!(scaled.device(), Device::Cuda(_))
        || scaled.dtype() != DType::F32
        || !scaled.is_contiguous()
        || scaled.rank() != 1
    {
        return Ok(None);
    }

    kiln_nvtx::range!(c"kiln/sampling_softmax_kt");

    // #1082: `scaled` is already a kt CUDA tensor — no candle bridge.
    // Drive the on-device softmax directly and read the probs back to host.
    let probs_kt = match kiln_tensor::cuda_softmax_last_axis(scaled) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    Ok(Some(crate::execution_phase::profile_accelerator_readback(
        probs_kt.device(),
        || probs_kt.to_vec1::<f32>(),
    )?))
}

/// Select the top-k of `scaled` on-device, transferring only the `k`
/// `(index, value)` pairs back to host in descending rank order
/// (descending value, ties broken by lower index).
///
/// On CUDA this routes through `kiln_tensor::cuda_topk_last_axis`, which
/// keeps the full `[V]` row resident on the device and copies just
/// `~k * 12` bytes over PCIe (#1082 perf-fix H9). On CPU/Vulkan/Metal —
/// where a host read is local, not a PCIe transfer — it returns `Err` so
/// callers fall back to `topk_via_host_sort`. Also returns `Err` if the
/// CUDA op itself fails, so the host fallback preserves correctness.
pub fn try_topk_on_device(scaled: &Tensor, top_k: usize) -> Result<Vec<(u32, f32)>> {
    // #1082 perf-fix H9: on CUDA, select the top-k on-device and transfer
    // ONLY the ~k (value, index) pairs (~k * 12 bytes) back to host —
    // instead of the `topk_via_host_sort` fallback's full-[V] f32
    // `to_vec1()` DtoH (~970 KB at V=248320) EVERY decoded token.
    //
    // The on-device kernel matches the host fallback's ranking exactly:
    // descending value, ties broken by lower index. So callers see
    // behaviourally identical `(idx, value)` pairs whichever path runs.
    //
    // On CPU/Vulkan/Metal we bail so callers take `topk_via_host_sort`,
    // whose `to_vec1()` is a local read on those backends (no PCIe hop).
    #[cfg(feature = "cuda")]
    {
        if matches!(scaled.device(), Device::Cuda(_))
            && scaled.is_contiguous()
            && scaled.rank() == 1
            && matches!(scaled.dtype(), DType::F32 | DType::BF16 | DType::F16)
        {
            kiln_nvtx::range!(c"kiln/sampling_topk_kt");
            let (values, indices, readback_duration) =
                kiln_tensor::cuda_topk_last_axis_profiled(scaled, top_k)?;
            crate::execution_phase::observe_profiled_readback(readback_duration);
            let pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
            return Ok(pairs);
        }
    }
    // ROCm: same on-device top-k as CUDA — keep the full [V] row resident and
    // transfer only the k (value, index) pairs, instead of the host-sort
    // fallback's full-[V] D2H every sampled token. Ranking is bit-identical to
    // `topk_via_host_sort` (descending value, ties → lower index).
    #[cfg(feature = "rocm")]
    {
        if matches!(scaled.device(), Device::Rocm(_))
            && scaled.is_contiguous()
            && scaled.rank() == 1
            && matches!(scaled.dtype(), DType::F32 | DType::BF16 | DType::F16)
        {
            let (values, indices, readback_duration) =
                kiln_tensor::rocm_topk_last_axis_profiled(scaled, top_k)?;
            crate::execution_phase::observe_profiled_readback(readback_duration);
            let pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
            return Ok(pairs);
        }
    }
    let _ = (scaled, top_k);
    anyhow::bail!(
        "try_topk_on_device: no on-device top-k for this backend; falling back to host sort (#1082)"
    )
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
    let values: Vec<f32> =
        crate::execution_phase::profile_accelerator_readback(scaled.device(), || scaled.to_vec1())?;
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
            fn eq(&self, o: &Self) -> bool {
                self.0 == o.0 && self.1 == o.1
            }
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
    if let Some(k) = top_k
        && k < indexed.len()
    {
        indexed.truncate(k);
    }
    Ok(indexed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "metal")]
    use crate::backend::metal::try_new_metal;

    #[test]
    fn test_greedy_sample_1d() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], device)?;
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
            device,
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
            device,
        )?
        .reshape((1, 2, 3))?;
        let token = greedy_sample(&logits)?;
        assert_eq!(token, 0); // index of 7.0 in last row
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_greedy_sample_kt_default_matches_expected() -> Result<()> {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!(
                "CUDA unavailable, skipping test_cuda_greedy_sample_kt_default_matches_expected"
            );
            return Ok(());
        }
        let device = Device::Cuda(0);

        let values = [
            9.0_f32, 1.0, 2.0, 3.0, // ignored non-last position
            0.0, 4.0, 8.0, 7.0, // max index 2
        ];
        let logits = Tensor::new(&values, device)?.reshape((2, 4))?;
        let flat = last_position_logits(&logits)?;

        // #1082: `flat` is already a kt CUDA tensor — pass it straight to the
        // kt helper, no candle->kt bridge.
        assert_eq!(
            crate::forward::try_kt_argmax_1d(&flat.contiguous()?)?,
            Some(2)
        );
        assert_eq!(greedy_sample(&logits)?, 2);

        let bf16_logits = logits.to_dtype(DType::BF16)?;
        let bf16_flat = last_position_logits(&bf16_logits)?;
        assert_eq!(
            crate::forward::try_kt_argmax_1d(&bf16_flat.contiguous()?)?,
            Some(2)
        );
        assert_eq!(greedy_sample(&bf16_logits)?, 2);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_topk_last_axis_matches_host_sort() -> Result<()> {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("CUDA unavailable, skipping test_cuda_topk_last_axis_matches_host_sort");
            return Ok(());
        }
        let device = Device::Cuda(0);

        // Includes a tie (two 4.0s at idx 6 and 11) to exercise the
        // lowest-index tie-break, matching the host fallback.
        let values: Vec<f32> = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 9.0, 0.5, 6.0, 2.5, 4.0];
        let logits = Tensor::new(values.as_slice(), device)?;

        for k in [1usize, 3, 7, 12] {
            let (vals, idxs) = kiln_tensor::cuda_topk_last_axis(&logits, k)?;
            assert_eq!(vals.len(), k);
            assert_eq!(idxs.len(), k);

            // Host reference: descending value, ties broken by lower index.
            let mut expected: Vec<(u32, f32)> = values
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (i as u32, v))
                .collect();
            expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
            expected.truncate(k);

            for (slot, ((gi, gv), (ei, ev))) in idxs
                .iter()
                .zip(&vals)
                .zip(expected.iter().map(|&(i, v)| (i, v)))
                .enumerate()
            {
                assert_eq!(*gi, ei, "k={k} slot={slot}: index mismatch");
                assert!(
                    (gv - ev).abs() < 1e-5,
                    "k={k} slot={slot}: value {gv} != expected {ev}"
                );
            }
        }
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
            device,
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
            device,
        )?
        .reshape((2, 2, 3))?;
        assert_eq!(greedy_sample_rows(&logits)?, vec![1, 0, 2, 1]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_greedy_sample_rows_kt_default_matches_expected() -> Result<()> {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!(
                "CUDA unavailable, skipping test_cuda_greedy_sample_rows_kt_default_matches_expected"
            );
            return Ok(());
        }
        let device = Device::Cuda(0);

        let values = [
            1.0_f32, 2.0, 9.0, 0.0, // max index 2
            7.0, 4.0, 3.0, 1.0, // max index 0
            -1.0, -2.0, -3.0, 0.0, // max index 3
            0.0, 5.0, 4.0, 3.0, // max index 1
        ];
        let expected = vec![2, 0, 3, 1];
        let logits = Tensor::new(&values, device)?.reshape((2, 2, 4))?;

        // #1082: `logits` is already a kt CUDA tensor — pass it straight to
        // the kt helper, no candle->kt bridge.
        assert_eq!(
            crate::forward::try_kt_sampling_argmax_rows(&logits.contiguous()?)?,
            Some(expected.clone())
        );
        assert_eq!(greedy_sample_rows(&logits)?, expected);

        let bf16_logits = logits.to_dtype(DType::BF16)?;
        assert_eq!(
            crate::forward::try_kt_sampling_argmax_rows(&bf16_logits.contiguous()?)?,
            Some(vec![2, 0, 3, 1])
        );
        assert_eq!(greedy_sample_rows(&bf16_logits)?, vec![2, 0, 3, 1]);
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
        let logits = Tensor::new(values.as_slice(), device)?;
        assert_eq!(greedy_sample(&logits)?, expected);
        Ok(())
    }

    #[test]
    fn test_sample_temperature_zero_is_greedy() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], device)?;
        let token = sample_with_params(&logits, 0.0, 1.0, 0, Some(42))?;
        assert_eq!(token, 1); // same as greedy
        Ok(())
    }

    #[test]
    fn test_sample_top_k_one_is_greedy() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], device)?;
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
        let logits = Tensor::new(&[1.0_f32, 10.0, 3.0, 2.0], device)?;
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
        let logits = Tensor::new(&[1.0_f32, 10.0, 8.0, 2.0, 0.5], device)?;
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
        let logits = Tensor::new(values.as_slice(), device)?;

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
        let logits = Tensor::new(&[10.0_f32, 0.0, 0.0, 0.0], device)?;
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
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 2.5], device)?;
        let t1 = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        let t2 = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        assert_eq!(t1, t2, "same seed should produce same result");
        Ok(())
    }

    #[test]
    fn test_full_distribution_sampling_tolerates_non_finite_logits() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[f32::NAN, f32::NEG_INFINITY, f32::NAN], device)?;
        let token = sample_with_params(&logits, 1.0, 1.0, 0, Some(12345))?;
        assert_eq!(token, 2);
        Ok(())
    }

    #[test]
    fn test_top_p_outside_range_is_full_distribution() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 2.5, 0.25, -0.5], device)?;
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
        let logits = Tensor::new(values.as_slice(), device)?;
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

    fn full_params_with_seed(seed: u64) -> kiln_core::sampling::SamplingParams {
        kiln_core::sampling::SamplingParams {
            seed: Some(seed),
            ..kiln_core::sampling::SamplingParams::greedy()
        }
    }

    #[test]
    fn test_full_params_greedy_short_circuits() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], device)?;
        let mut params = full_params_with_seed(42);
        params.temperature = 0.0;
        let token = sample_with_full_params(&logits, &params, &[])?;
        assert_eq!(token, 1, "temperature=0 must be greedy");
        Ok(())
    }

    #[test]
    fn traced_greedy_sampling_has_unit_probability() -> Result<()> {
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], Device::Cpu)?;
        let mut params = full_params_with_seed(42);
        params.temperature = 0.0;
        let sampled = sample_with_full_params_and_logprob(&logits, &params, &[])?;
        assert_eq!(sampled, SampledToken::deterministic(1));
        Ok(())
    }

    #[test]
    fn traced_sampling_matches_token_path_and_full_distribution_probability() -> Result<()> {
        let values = [0.0_f32, 1.0, 2.0, -1.0];
        let logits = Tensor::new(&values, Device::Cpu)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1.0;
        params.top_p = 1.0;
        params.top_k = 0;
        params.min_p = 0.0;

        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = values.map(|value| (value - max).exp());
        let normalizer: f32 = weights.iter().sum();
        for seed in 0..32 {
            params.seed = Some(seed);
            let expected_token = sample_with_full_params(&logits, &params, &[])?;
            let sampled = sample_with_full_params_and_logprob(&logits, &params, &[])?;
            assert_eq!(sampled.token_id, expected_token, "seed {seed}");
            let expected_logprob = (weights[expected_token as usize] / normalizer).ln();
            assert!(
                (sampled.logprob - expected_logprob).abs() <= 1e-6,
                "seed {seed}: got {}, expected {expected_logprob}",
                sampled.logprob
            );
        }
        Ok(())
    }

    #[test]
    fn traced_sampling_reports_probability_after_penalties_and_filters() -> Result<()> {
        let logits = Tensor::new(&[3.0_f32, 2.0, 1.0, 0.0], Device::Cpu)?;
        let mut params = full_params_with_seed(0);
        params.temperature = 1.0;
        params.top_k = 3;
        params.min_p = 0.3;
        params.top_p = 0.75;
        params.frequency_penalty = 0.75;
        let history = [0_u32, 0];

        // The frequency penalty changes token 0's logit from 3.0 to 1.5.
        // Top-k/min-p retain logits [2.0, 1.5, 1.0], then top-p retains the
        // first two and renormalizes them to softmax([2.0, 1.5]).
        let p_token_1 = 1.0 / (1.0 + (-0.5_f32).exp());
        let p_token_0 = 1.0 - p_token_1;
        for seed in 0..32 {
            params.seed = Some(seed);
            let expected_token = sample_with_full_params(&logits, &params, &history)?;
            let sampled = sample_with_full_params_and_logprob(&logits, &params, &history)?;
            assert_eq!(sampled.token_id, expected_token, "seed {seed}");
            let expected_probability = match sampled.token_id {
                1 => p_token_1,
                0 => p_token_0,
                token => panic!("filtered distribution emitted token {token}"),
            };
            assert!(
                (sampled.logprob - expected_probability.ln()).abs() <= 1e-6,
                "seed {seed}: got {}, expected {}",
                sampled.logprob,
                expected_probability.ln()
            );
        }

        params.seed = None;
        let error = sample_with_full_params_and_logprob(&logits, &params, &history).unwrap_err();
        assert!(error.to_string().contains("requires a resolved seed"));
        Ok(())
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn traced_sampling_rocm_topk_matches_cpu_token_and_logprob() -> Result<()> {
        if !kiln_tensor::rocm_is_available() {
            eprintln!("ROCm unavailable, skipping traced sampling parity");
            return Ok(());
        }
        let values: Vec<f32> = (0..257)
            .map(|index| ((index as f32 + 0.25) * 0.173).sin() * 3.0 + index as f32 * 0.0001)
            .collect();
        let cpu_logits = Tensor::new(values.as_slice(), Device::Cpu)?;
        let rocm_logits = Tensor::new(values.as_slice(), Device::Rocm(0))?;
        let history = [3_u32, 3, 17, 41, 41, 41];
        let mut params = full_params_with_seed(0);
        params.temperature = 0.8;
        params.top_k = 20;
        params.top_p = 0.85;
        params.min_p = 0.05;
        params.repetition_penalty = 1.1;
        params.presence_penalty = 0.2;
        params.frequency_penalty = 0.05;

        for seed in 0..16 {
            params.seed = Some(seed);
            let expected = sample_with_full_params_and_logprob(&cpu_logits, &params, &history)?;
            let actual = sample_with_full_params_and_logprob(&rocm_logits, &params, &history)?;
            assert_eq!(actual.token_id, expected.token_id, "seed {seed}");
            assert!(
                (actual.logprob - expected.logprob).abs() <= 1e-5,
                "seed {seed}: ROCm logprob {} != CPU {}",
                actual.logprob,
                expected.logprob
            );
        }

        params.seed = Some(91);
        let expected = sample_with_full_params_and_logprob(&cpu_logits, &params, &history)?;
        let profiled = crate::execution_phase::profile_readback_invocation(|| {
            sample_with_full_params_and_logprob(&rocm_logits, &params, &history)
        })?;
        assert_eq!(profiled.value.token_id, expected.token_id);
        assert!(
            profiled
                .readback_duration
                .is_some_and(|duration| !duration.is_zero()),
            "ROCm penalty/top-k sampling must expose its existing readbacks"
        );

        let expected_greedy = greedy_sample(&cpu_logits)?;
        let profiled_greedy =
            crate::execution_phase::profile_readback_invocation(|| greedy_sample(&rocm_logits))?;
        assert_eq!(profiled_greedy.value, expected_greedy);
        assert!(
            profiled_greedy
                .readback_duration
                .is_some_and(|duration| !duration.is_zero()),
            "ROCm greedy sampling must expose its existing scalar readback"
        );
        Ok(())
    }

    #[test]
    fn test_full_params_no_op_path_matches_legacy() -> Result<()> {
        // With penalties off, min_p=0, the full sampler must produce the
        // same token as the legacy sample_with_params for any given seed.
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 5.0, 3.0, 2.0], device)?;
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
    #[cfg(feature = "cuda")]
    fn test_cuda_sampling_penalties_kt_default_matches_candle_path() -> Result<()> {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!(
                "CUDA unavailable, skipping test_cuda_sampling_penalties_kt_default_matches_candle_path"
            );
            return Ok(());
        }
        let cuda = Device::Cuda(0);
        let cpu = Device::Cpu;
        let values = [
            0.0_f32, 5.0, -2.0, 3.0, 1.0, 0.5, // ignored non-last position
            0.5, 8.0, -4.0, 2.0, 1.5, 0.0, // sampled last position
        ];
        let history = [1_u32, 1, 2, 4];
        let repetition = 2.0;
        let presence = 0.5;
        let frequency = 0.25;

        let cuda_logits = Tensor::new(&values, cuda)?.reshape((2, 6))?;
        let got =
            apply_penalties_on_device(&cuda_logits, &history, repetition, presence, frequency)?;

        let cuda_flat = last_position_logits(&cuda_logits)?.to_dtype(DType::F32)?;
        let got_direct = try_kt_apply_penalties_on_device(
            &cuda_flat, &history, repetition, presence, frequency,
        )?
        .context("expected CUDA kt penalty path to run")?;

        let cpu_logits = Tensor::new(&values, cpu)?.reshape((2, 6))?;
        let expected =
            apply_penalties_on_device(&cpu_logits, &history, repetition, presence, frequency)?;

        let got = got.to_vec1::<f32>()?;
        let got_direct = got_direct.to_vec1::<f32>()?;
        let expected = expected.to_vec1::<f32>()?;
        assert_eq!(got.len(), expected.len());
        assert_eq!(got_direct.len(), expected.len());
        for (idx, ((g, gd), e)) in got
            .iter()
            .zip(got_direct.iter())
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() <= 1e-6,
                "penalty logit mismatch at {idx}: got {g}, expected {e}"
            );
            assert!(
                (gd - e).abs() <= 1e-6,
                "direct kt penalty logit mismatch at {idx}: got {gd}, expected {e}"
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_sampling_softmax_kt_helper_matches_host_probs() -> Result<()> {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!(
                "CUDA unavailable, skipping test_cuda_sampling_softmax_kt_helper_matches_host_probs"
            );
            return Ok(());
        }
        let cuda = Device::Cuda(0);
        let values = [0.0_f32, 2.0, -1.0, 6.0, 1.0, -3.0];
        let logits = Tensor::new(&values, cuda)?;

        let got = try_kt_full_distribution_probs(&logits)?
            .context("expected CUDA kt sampler softmax path to run")?;
        let expected = softmax_probs_from_logits(&values).context("host softmax probs")?;
        assert_eq!(got.len(), expected.len());
        for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-5,
                "sampling softmax mismatch at {idx}: got {g}, expected {e}"
            );
        }

        let fallback_idx = (values.len() - 1) as u32;
        let got_token = sample_from_distribution_weights(&got, Some(123), fallback_idx)?;
        let expected_token = sample_from_distribution_weights(&expected, Some(123), fallback_idx)?;
        assert_eq!(got_token, expected_token);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_full_distribution_sampler_default_matches_cpu() -> Result<()> {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!(
                "CUDA unavailable, skipping test_cuda_full_distribution_sampler_default_matches_cpu"
            );
            return Ok(());
        }
        let cuda = Device::Cuda(0);
        let cpu = Device::Cpu;
        let values = [0.0_f32, 2.0, -1.0, 6.0, 1.0, -3.0];
        let cuda_logits = Tensor::new(&values, cuda)?;
        let cpu_logits = Tensor::new(&values, cpu)?;

        for seed in 0..32 {
            let got = sample_full_distribution_unsorted(&cuda_logits, Some(seed))?;
            let expected = sample_full_distribution_unsorted(&cpu_logits, Some(seed))?;
            assert_eq!(
                got, expected,
                "full-distribution token mismatch at seed {seed}: got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_min_p_drops_low_probability_tokens() -> Result<()> {
        let device = Device::Cpu;
        // Token 0 dominates the distribution (~99%); tokens 1-3 are tiny.
        let logits = Tensor::new(&[10.0_f32, 0.0, 0.0, 0.0], device)?;
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
        let logits = Tensor::new(&[2.0_f32, 5.0, 4.0, 1.0], device)?;
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
        let logits = Tensor::new(&[2.0_f32, 5.0, 4.0, 1.0], device)?;
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
        let logits = Tensor::new(&[2.0_f32, 5.0, 4.0, 1.0], device)?;
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
        let logits = Tensor::new(&[10.0_f32, 1.0, 1.0, 1.0], device)?;
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
        let logits = Tensor::new(&[5.0_f32, 4.5, 4.0, 1.0, 0.5, 0.0], device)?;
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
        let logits = Tensor::new(&[10.0_f32, 1.0, 1.0, 1.0], device)?;
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
        let logits = Tensor::new(values.as_slice(), device)?;
        for &k in &[1, 20, 50, 200, 1024] {
            let heap = topk_via_host_sort(&logits, Some(k))?;
            let mut full: Vec<(u32, f32)> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            full.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            full.truncate(k);
            assert_eq!(
                heap.len(),
                full.len(),
                "heap path produced wrong length for k={k}"
            );
            // Compare only the *set* of indices since equal-value
            // ties can break either direction across the two paths.
            let heap_set: std::collections::BTreeSet<u32> = heap.iter().map(|&(i, _)| i).collect();
            let full_set: std::collections::BTreeSet<u32> = full.iter().map(|&(i, _)| i).collect();
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
        // Coarse cross-hardware tripwire: the penalty pass must stay well
        // under a single decode forward step (~20 ms for Qwen3.5-4B), so an
        // algorithmic regression (e.g. an accidental O(vocab*history) scan
        // instead of the partial-top-k heap + small history scatter) trips
        // this, while normal host-CPU variation does not. Measured steady
        // state is ~7 ms/call on an A6000 pod host CPU (152k vocab, 500-token
        // history, top_k=20), so the 20 ms budget leaves ~3x headroom.
        // Release-only (`cfg(not(debug_assertions))`); never runs on the
        // debug-mode GHA CI.
        assert!(
            per_call < std::time::Duration::from_millis(20),
            "penalty path took {:?} per call (release-mode budget 20ms) — algorithmic regression?",
            per_call,
        );
        Ok(())
    }

    #[test]
    fn test_seed_determinism_with_full_params() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0_f32, 2.0, 3.0, 2.5, 0.5, -1.0], device)?;
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
