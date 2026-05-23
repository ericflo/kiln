//! Modern sampler `LogitProcessor`s: `min_p` + `typical_p`.
//!
//! Two newer alternatives to nucleus sampling that compose with the
//! Phase 1.48 chain via the [`LogitProcessor`] trait.
//!
//! # min_p (Nguyen et al. 2024)
//!
//! Keeps every token whose probability `P(t) >= p * P(top)` — i.e.
//! a relative threshold based on the most-likely token's probability.
//! Unlike top-p which is cumulative, min-p is a per-token threshold;
//! it produces smaller candidate sets on peaked distributions and
//! larger sets on flat ones.
//!
//! # typical_p (Meister et al. 2022, "Locally Typical Sampling")
//!
//! Targets entries with information content close to the conditional
//! entropy. Computes per-token surprisal `s_i = -log P(i)` and
//! selects the smallest set of tokens whose cumulative probability
//! reaches `p` AND whose surprisal is close to `H = -sum P(i) log P(i)`.
//!
//! Both produce a mask that's intersected with the logits (mask
//! out-of-set entries with -inf).

use crate::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};
use std::sync::Arc;

use super::logit_processor::LogitProcessor;

// ----------------------------------------------------------------------
// Shared helpers
// ----------------------------------------------------------------------

fn validate_logits(logits: &Tensor, name: &str) -> Result<()> {
    if logits.rank() != 2 {
        bail!("{name}: logits must be rank-2 [batch, vocab], got {:?}", logits.shape());
    }
    if !logits.is_contiguous() {
        bail!("{name}: logits must be contiguous");
    }
    if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", logits.dtype());
    }
    Ok(())
}

fn load_all_rows_f32(logits: &Tensor, batch: usize, vocab: usize) -> Result<Vec<Vec<f32>>> {
    let cpu = logits
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("logit_modern: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let dtype = logits.dtype();
    let per = dtype.size_in_bytes();
    let mut rows = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = Vec::with_capacity(vocab);
        let start = b * vocab * per;
        for v in 0..vocab {
            let chunk = &bytes[start + v * per..start + (v + 1) * per];
            let val = match dtype {
                DType::F32 => f32::from_le_bytes(chunk.try_into().unwrap()),
                DType::BF16 => half::bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32(),
                DType::F16 => half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32(),
                _ => unreachable!(),
            };
            row.push(val);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn store_rows(dtype: DType, shape: &[usize], rows: &[Vec<f32>]) -> Result<Tensor> {
    let batch = rows.len();
    let vocab = rows.first().map(|r| r.len()).unwrap_or(0);
    let per = dtype.size_in_bytes();
    let mut out = vec![0u8; batch * vocab * per];
    for (b, row) in rows.iter().enumerate() {
        for (v, &val) in row.iter().enumerate() {
            let offset = (b * vocab + v) * per;
            match dtype {
                DType::F32 => out[offset..offset + 4].copy_from_slice(&val.to_le_bytes()),
                DType::BF16 => out[offset..offset + 2]
                    .copy_from_slice(&half::bf16::from_f32(val).to_le_bytes()),
                DType::F16 => out[offset..offset + 2]
                    .copy_from_slice(&half::f16::from_f32(val).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

/// Compute softmax probabilities of a single row (max-subtracted for
/// numerical stability). Returns a Vec of length `row.len()`.
fn softmax_row(row: &[f32]) -> Vec<f32> {
    let max_v = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !max_v.is_finite() {
        return vec![1.0 / row.len() as f32; row.len()];
    }
    let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ----------------------------------------------------------------------
// MinPProcessor
// ----------------------------------------------------------------------

/// Min-p sampling (Nguyen et al. 2024).
///
/// Keeps every token with `P(t) >= p * P(top)`. The smaller `p`, the
/// larger the candidate set on peaked distributions.
#[derive(Debug, Clone, Copy)]
pub struct MinPProcessor {
    pub p: f32,
}

impl MinPProcessor {
    pub const fn new(p: f32) -> Self {
        MinPProcessor { p }
    }
}

impl LogitProcessor for MinPProcessor {
    fn name(&self) -> &'static str {
        "min_p"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "MinPProcessor")?;
        if !(0.0 < self.p && self.p <= 1.0) {
            bail!("MinPProcessor: p must be in (0, 1], got {}", self.p);
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for row in rows.iter_mut() {
            let probs = softmax_row(row);
            let p_max = probs.iter().fold(0.0_f32, |a, &b| a.max(b));
            let threshold = self.p * p_max;
            for (i, &prob) in probs.iter().enumerate() {
                if prob < threshold {
                    row[i] = f32::NEG_INFINITY;
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

// ----------------------------------------------------------------------
// TypicalPProcessor
// ----------------------------------------------------------------------

/// Locally-typical sampling (Meister et al. 2022).
///
/// Sorts tokens by `|surprisal - entropy|` ascending; keeps the
/// smallest prefix whose cumulative probability reaches `p`.
#[derive(Debug, Clone, Copy)]
pub struct TypicalPProcessor {
    pub p: f32,
}

impl TypicalPProcessor {
    pub const fn new(p: f32) -> Self {
        TypicalPProcessor { p }
    }
}

impl LogitProcessor for TypicalPProcessor {
    fn name(&self) -> &'static str {
        "typical_p"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "TypicalPProcessor")?;
        if !(0.0 < self.p && self.p <= 1.0) {
            bail!("TypicalPProcessor: p must be in (0, 1], got {}", self.p);
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for row in rows.iter_mut() {
            let probs = softmax_row(row);
            // Surprisal = -log P(i)
            let surprisals: Vec<f32> = probs.iter().map(|&p| -p.max(1e-30).ln()).collect();
            // Entropy = sum P log(1/P) = -sum P log P
            let entropy: f32 = probs
                .iter()
                .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
                .sum();
            // Distance from entropy
            let dist: Vec<f32> = surprisals.iter().map(|&s| (s - entropy).abs()).collect();
            // Sort indices by dist ascending
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(std::cmp::Ordering::Equal));
            // Walk sorted indices, accumulate probability, keep while
            // cumulative < p; then keep the one that crosses p (so we
            // always select at least one token).
            let mut keep = vec![false; vocab];
            let mut acc = 0.0_f32;
            for &i in &idx {
                acc += probs[i];
                keep[i] = true;
                if acc >= self.p {
                    break;
                }
            }
            for (i, k) in keep.iter().enumerate() {
                if !*k {
                    row[i] = f32::NEG_INFINITY;
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_rows(t: &Tensor, batch: usize, vocab: usize) -> Vec<Vec<f32>> {
        load_all_rows_f32(t, batch, vocab).unwrap()
    }

    // ─── MinP ──────────────────────────────────────────────────────

    #[test]
    fn min_p_keeps_all_when_p_is_small() {
        // Uniform distribution; p=0.5 keeps everything (every prob =
        // 1/4 = 0.25 ≥ 0.5 * 0.25 = 0.125).
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let out = MinPProcessor::new(0.5).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        for &v in &rows[0] {
            assert!(v.is_finite(), "uniform + p=0.5 should keep all; got {v}");
        }
    }

    #[test]
    fn min_p_keeps_only_top_when_p_is_one() {
        // p=1.0 → threshold = P(top); only the top-1 survives (strict).
        // logits [1, 2, 3]: softmax peaks at index 2.
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let out = MinPProcessor::new(1.0).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert!(rows[0][1].is_infinite() && rows[0][1] < 0.0);
        assert_eq!(rows[0][2], 3.0); // top-1 survives
    }

    #[test]
    fn min_p_peaked_distribution_filters_tail() {
        // Logits with a clear peak: [10, 0, 0, 0]; softmax ~ [1, 0, 0, 0]
        // → with p=0.1, threshold ≈ 0.1; only index 0 survives.
        let logits = Tensor::from_slice(&[10.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let out = MinPProcessor::new(0.1).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert_eq!(rows[0][0], 10.0);
        for i in 1..4 {
            assert!(rows[0][i].is_infinite() && rows[0][i] < 0.0);
        }
    }

    #[test]
    fn min_p_invalid_p_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        assert!(MinPProcessor::new(0.0).apply(&logits).is_err());
        assert!(MinPProcessor::new(1.5).apply(&logits).is_err());
    }

    // ─── TypicalP ──────────────────────────────────────────────────

    #[test]
    fn typical_p_uniform_keeps_at_least_one() {
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let out = TypicalPProcessor::new(0.5).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        let kept = rows[0].iter().filter(|v| v.is_finite()).count();
        assert!(kept >= 1, "typical_p must always keep ≥ 1");
    }

    #[test]
    fn typical_p_one_keeps_everything() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let out = TypicalPProcessor::new(1.0).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        for v in &rows[0] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn typical_p_invalid_p_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        assert!(TypicalPProcessor::new(0.0).apply(&logits).is_err());
        assert!(TypicalPProcessor::new(-0.5).apply(&logits).is_err());
    }

    #[test]
    fn names() {
        assert_eq!(MinPProcessor::new(0.5).name(), "min_p");
        assert_eq!(TypicalPProcessor::new(0.5).name(), "typical_p");
    }
}
