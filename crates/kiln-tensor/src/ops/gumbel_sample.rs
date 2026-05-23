//! Gumbel-max categorical sample — the terminal step of the
//! sampler chain.
//!
//! After every `LogitProcessor` has masked / scaled the logits,
//! Gumbel-max turns them into a single sampled token id per batch
//! row in one fused elementwise + argmax kernel.
//!
//! # Algorithm
//!
//! For each row `r` of logits `[B, V]`:
//!
//! 1. Draw `g_v ~ Gumbel(0, 1)` independently for each vocab index
//!    `v`. Sampling Gumbel is two stdlib calls:
//!    `g_v = -ln(-ln(u_v))` where `u_v ~ Uniform(0, 1)` (using
//!    `(0, 1)` open interval to keep `ln` defined).
//! 2. Add to logits: `s_v = logit_v + g_v`.
//! 3. Token id = `argmax_v s_v`.
//!
//! This is the **Gumbel-max trick**: it is provably equivalent to
//! drawing one token from `softmax(logits)`, but requires no
//! normalization, no exp, no cumsum — just one elementwise add and
//! an argmax. Maps perfectly to a single GPU kernel.
//!
//! # Why this op and not a regular categorical sample?
//!
//! - **No softmax**: the `-inf` masks from upstream processors (top-K,
//!   top-P, ngram-block, etc.) work seamlessly. `-inf + Gumbel = -inf`
//!   so masked tokens are guaranteed never selected.
//! - **One pass**: standard categorical sample is two passes
//!   (softmax + cumsum + uniform draw + linear search). Gumbel is one
//!   pass and easily fused with the chain's prior op.
//! - **No numerical issues**: softmax with masked logits can produce
//!   `0/0` in the exp/sum step if every token has `-inf`. Gumbel-max
//!   short-circuits to the argmax of the few survivors automatically.
//!
//! # Determinism
//!
//! `ToleranceBounded { dtype_band_key: "gumbel_argmax" }`. The argmax
//! is constructive given the noise draws, but the noise draws are
//! seed-dependent and the parity envelope must account for `u → ln`
//! precision differences between CPU/CUDA/Metal/Vulkan.
//!
//! In tests we use a fixed splitmix64 seed for deterministic output.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};
use std::sync::Arc;

/// Single-step Gumbel-max categorical sampler.
///
/// Sample one token id per batch row. Returns an `[B]` `I64` tensor.
#[derive(Debug)]
pub struct GumbelSampler {
    rng: Mutex<u64>,
}

impl GumbelSampler {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xCAFEBABEDEADBEEFu64);
        GumbelSampler {
            rng: Mutex::new(seed.max(1)),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        GumbelSampler {
            rng: Mutex::new(seed.max(1)),
        }
    }

    /// Sample one token id per row of `logits: [B, V]`.
    /// Returns `[B]` `I64` token ids.
    pub fn sample(&self, logits: &Tensor) -> Result<Tensor> {
        if logits.rank() != 2 {
            bail!(
                "GumbelSampler: logits must be rank-2 [batch, vocab], got {:?}",
                logits.shape()
            );
        }
        if !logits.is_contiguous() {
            bail!("GumbelSampler: logits must be contiguous");
        }
        if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "GumbelSampler: dtype must be F32/BF16/F16, got {}",
                logits.dtype()
            );
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let cpu = logits
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("GumbelSampler: storage must be CpuStorage on CPU"))?;
        let bytes = cpu.as_bytes();
        let dtype = logits.dtype();
        let per = dtype.size_in_bytes();

        let mut out = Vec::with_capacity(batch);
        let mut rng = self.rng.lock().unwrap();
        for b in 0..batch {
            let mut best_idx: i64 = 0;
            let mut best_score = f32::NEG_INFINITY;
            let mut all_neg_inf = true;
            for v in 0..vocab {
                let i = b * vocab + v;
                let logit = match dtype {
                    DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
                    DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                    DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                    _ => unreachable!(),
                };
                let g = gumbel_sample(&mut rng);
                if logit.is_finite() {
                    all_neg_inf = false;
                    let score = logit + g;
                    if score > best_score {
                        best_score = score;
                        best_idx = v as i64;
                    }
                }
            }
            // Safety net: if every logit was -inf (shouldn't happen
            // after a healthy LogitProcessor chain, but we guard
            // anyway), fall back to index 0.
            if all_neg_inf {
                best_idx = 0;
            }
            out.push(best_idx);
        }

        let bytes_out: Vec<u8> = out.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let cpu_out = CpuStorage::from_bytes(DType::I64, bytes_out)?;
        let storage: Storage = Arc::new(cpu_out);
        Tensor::from_parts(storage, Layout::contiguous(vec![batch]), TensorId::next())
    }
}

impl Default for GumbelSampler {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------
// RNG helpers
// ----------------------------------------------------------------------

fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15u64);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
    z ^ (z >> 31)
}

fn uniform_open(state: &mut u64) -> f64 {
    // Open (0, 1) — never returns 0 (which would make ln(u) -inf).
    // Use 53-bit precision to give Gumbel high tail fidelity.
    let r = splitmix64_step(state) >> 11; // top 53 bits
    let denom = (1u64 << 53) as f64;
    (r as f64 + 0.5) / denom
}

fn gumbel_sample(state: &mut u64) -> f32 {
    let u = uniform_open(state);
    // u in (0, 1) → ln(u) in (-inf, 0) → -ln(u) in (0, inf) → ln(-ln(u)) in R
    let g = -(-u.ln()).ln();
    g as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_i64(t: &Tensor) -> Vec<i64> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn gumbel_sharp_distribution_picks_argmax_with_high_probability() {
        // Logits [100, 0, 0, 0] — index 0 should be the sample
        // essentially always.
        let logits = Tensor::from_slice(&[100.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let s = GumbelSampler::with_seed(1);
        let out = s.sample(&logits).unwrap();
        let ids = read_i64(&out);
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn gumbel_output_shape_and_dtype() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let s = GumbelSampler::with_seed(42);
        let out = s.sample(&logits).unwrap();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out.dtype(), DType::I64);
    }

    #[test]
    fn gumbel_respects_neg_inf_mask() {
        // If only index 2 is finite, the sampler must always pick 2.
        let logits = Tensor::from_slice(
            &[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY],
            vec![1, 4],
        )
        .unwrap();
        for seed in 1..50 {
            let s = GumbelSampler::with_seed(seed);
            let out = s.sample(&logits).unwrap();
            assert_eq!(read_i64(&out), vec![2], "seed {seed}");
        }
    }

    #[test]
    fn gumbel_deterministic_with_same_seed() {
        let logits =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 1.0, 1.5, 2.0, 2.5, 3.0], vec![2, 5])
                .unwrap();
        let s1 = GumbelSampler::with_seed(99);
        let s2 = GumbelSampler::with_seed(99);
        let r1 = read_i64(&s1.sample(&logits).unwrap());
        let r2 = read_i64(&s2.sample(&logits).unwrap());
        assert_eq!(r1, r2);
    }

    #[test]
    fn gumbel_different_seeds_diverge() {
        // Flat distribution → samples are RNG-dominated. Over many seeds
        // we expect to see > 1 distinct token id.
        let logits = Tensor::from_slice(&vec![1.0f32; 32], vec![1, 32]).unwrap();
        let mut tokens = std::collections::HashSet::new();
        for seed in 1..200 {
            let s = GumbelSampler::with_seed(seed);
            let id = read_i64(&s.sample(&logits).unwrap())[0];
            tokens.insert(id);
        }
        assert!(
            tokens.len() > 8,
            "expected diversity from RNG; got {} tokens",
            tokens.len()
        );
    }

    #[test]
    fn gumbel_distribution_approximates_softmax() {
        // Logits [ln(0.7), ln(0.2), ln(0.1)] → softmax = [0.7, 0.2, 0.1].
        // Sample 10000 times; empirical frequencies should be close.
        let logits = Tensor::from_slice(
            &[0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln()],
            vec![1, 3],
        )
        .unwrap();
        let s = GumbelSampler::with_seed(123);
        let mut counts = [0usize; 3];
        for _ in 0..10_000 {
            let id = read_i64(&s.sample(&logits).unwrap())[0];
            counts[id as usize] += 1;
        }
        let total = 10_000.0;
        let f0 = counts[0] as f32 / total;
        let f1 = counts[1] as f32 / total;
        let f2 = counts[2] as f32 / total;
        assert!(
            (f0 - 0.7).abs() < 0.02,
            "expected ~0.7 for index 0, got {f0}"
        );
        assert!(
            (f1 - 0.2).abs() < 0.02,
            "expected ~0.2 for index 1, got {f1}"
        );
        assert!(
            (f2 - 0.1).abs() < 0.02,
            "expected ~0.1 for index 2, got {f2}"
        );
    }

    #[test]
    fn gumbel_rejects_rank_1() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let s = GumbelSampler::with_seed(1);
        let e = s.sample(&logits).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }

    #[test]
    fn gumbel_rejects_bad_dtype() {
        let logits = Tensor::from_slice(&[1u32, 2, 3], vec![1, 3]).unwrap();
        let s = GumbelSampler::with_seed(1);
        let e = s.sample(&logits).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn gumbel_bf16_input_path() {
        let bf: Vec<half::bf16> = [0.1f32, 100.0, 0.2]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let logits = Tensor::from_slice(&bf, vec![1, 3]).unwrap();
        let s = GumbelSampler::with_seed(7);
        let out = s.sample(&logits).unwrap();
        assert_eq!(read_i64(&out), vec![1]);
    }

    #[test]
    fn gumbel_all_neg_inf_falls_back_to_index_zero() {
        // Pathological input — every logit is masked. Sampler returns 0.
        let logits = Tensor::from_slice(
            &[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
            vec![1, 3],
        )
        .unwrap();
        let s = GumbelSampler::with_seed(1);
        let out = s.sample(&logits).unwrap();
        assert_eq!(read_i64(&out), vec![0]);
    }

    #[test]
    fn gumbel_per_batch_independent() {
        // Two rows with strongly different argmaxes — both should be
        // selected correctly even though the sampler shares one RNG.
        let logits =
            Tensor::from_slice(&[100.0f32, 0.0, 0.0, 0.0, 0.0, 100.0], vec![2, 3]).unwrap();
        let s = GumbelSampler::with_seed(1);
        let out = s.sample(&logits).unwrap();
        assert_eq!(read_i64(&out), vec![0, 2]);
    }
}
