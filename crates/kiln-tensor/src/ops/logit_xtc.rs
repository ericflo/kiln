//! XTC ("Exclude Top Choices") logit processor.
//!
//! Per the original proposal (oobabooga/text-generation-webui, 2024):
//!
//! 1. With probability `probability`, do nothing this step (XTC is
//!    stochastic — it fires intermittently, not every step).
//! 2. Otherwise, find every token whose softmax probability exceeds
//!    `threshold`. If at most one token clears the bar, do nothing.
//! 3. Among those above-threshold tokens, sort by probability
//!    descending and mask every one *except the last* (the lowest-
//!    probability one that is still above the threshold).
//!
//! The intuition: top-K / top-P / Mirostat all *keep* the
//! highest-probability tokens. XTC inverts that — when it fires, it
//! deliberately *removes* the safe, boring high-probability tokens
//! to force the sampler into the long tail.
//!
//! # RNG
//!
//! `LogitProcessor::apply` takes `&self`, so we keep an internal
//! splitmix64 state behind a `Mutex<u64>`. Constructed with an
//! explicit seed for deterministic testing; the seedless constructor
//! draws a fresh seed from `SystemTime`.
//!
//! splitmix64 is the same kernel `XorShift*` family seeders use —
//! no `rand` crate dependency required.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};
use std::sync::Arc;

use super::logit_processor::LogitProcessor;

// ----------------------------------------------------------------------
// Local helpers (mirrors of logit_misc).
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
        .ok_or_else(|| Error::from_str("xtc: storage must be CpuStorage"))?;
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

/// Tiny splitmix64. One step per call.
fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15u64);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
    z ^ (z >> 31)
}

fn next_f32(state: &mut u64) -> f32 {
    // Take the top 24 bits → divide by 2^24 → uniform on [0, 1).
    let x = splitmix64_step(state);
    let bits = (x >> 40) as u32;
    bits as f32 / (1u32 << 24) as f32
}

// ----------------------------------------------------------------------
// XtcProcessor
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct XtcProcessor {
    /// Probability threshold. Only tokens whose softmax p exceeds
    /// this value are eligible for exclusion.
    pub threshold: f32,
    /// Probability that XTC fires on any given step.
    pub probability: f32,
    /// Internal splitmix64 RNG state.
    rng: Mutex<u64>,
}

impl XtcProcessor {
    pub fn new(threshold: f32, probability: f32) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEADBEEFCAFEBABEu64);
        XtcProcessor {
            threshold,
            probability,
            rng: Mutex::new(seed.max(1)),
        }
    }

    pub fn with_seed(threshold: f32, probability: f32, seed: u64) -> Self {
        XtcProcessor {
            threshold,
            probability,
            rng: Mutex::new(seed.max(1)),
        }
    }
}

impl LogitProcessor for XtcProcessor {
    fn name(&self) -> &'static str {
        "xtc"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "XtcProcessor")?;
        if !(0.0 <= self.threshold && self.threshold < 1.0) {
            bail!(
                "XtcProcessor: threshold must be in [0, 1), got {}",
                self.threshold
            );
        }
        if !(0.0 <= self.probability && self.probability <= 1.0) {
            bail!(
                "XtcProcessor: probability must be in [0, 1], got {}",
                self.probability
            );
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for row in rows.iter_mut() {
            // Independent fire decision per batch row.
            let r = {
                let mut s = self.rng.lock().unwrap();
                next_f32(&mut s)
            };
            if r >= self.probability {
                continue; // skip XTC this row
            }
            // Compute softmax probabilities to find above-threshold set.
            let max_v = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_v).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();
            // Find indices above threshold.
            let mut above: Vec<usize> = (0..vocab).filter(|&i| probs[i] > self.threshold).collect();
            if above.len() < 2 {
                continue; // need at least 2 above threshold to "exclude top choices"
            }
            // Sort by probability descending.
            above.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            // Mask everyone except the LAST (lowest probability above threshold).
            for &i in &above[..above.len() - 1] {
                row[i] = f32::NEG_INFINITY;
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

    /// Find a seed that makes XTC fire on the next call for the given probability.
    fn fire_seed(probability: f32) -> u64 {
        let mut s: u64 = 1;
        loop {
            let mut snapshot = s;
            let r = next_f32(&mut snapshot);
            if r < probability {
                return s;
            }
            s += 1;
        }
    }

    /// Find a seed that makes XTC skip the next call.
    fn skip_seed(probability: f32) -> u64 {
        let mut s: u64 = 1;
        loop {
            let mut snapshot = s;
            let r = next_f32(&mut snapshot);
            if r >= probability {
                return s;
            }
            s += 1;
        }
    }

    #[test]
    fn xtc_fires_and_masks_top_choices() {
        // Distribution: [5.0, 4.0, 3.0, -10.0]. Softmax probs roughly:
        // 0.67, 0.24, 0.09, ~0. Threshold 0.1 → indices 0 and 1 above.
        // XTC fires → mask index 0 (highest), keep index 1 (lowest
        // above threshold), keep 2 and 3 (below threshold) unchanged.
        let logits = Tensor::from_slice(&[5.0f32, 4.0, 3.0, -10.0], vec![1, 4]).unwrap();
        let seed = fire_seed(1.0);
        let p = XtcProcessor::with_seed(0.1, 1.0, seed);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert_eq!(rows[0][1], 4.0);
        assert_eq!(rows[0][2], 3.0);
        assert_eq!(rows[0][3], -10.0);
    }

    #[test]
    fn xtc_skips_when_rng_above_probability() {
        let logits = Tensor::from_slice(&[5.0f32, 4.0, 3.0], vec![1, 3]).unwrap();
        // probability=0.0 → never fires.
        let p = XtcProcessor::with_seed(0.1, 0.0, 1);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn xtc_noop_when_only_one_above_threshold() {
        // Sharply peaked distribution → only the top is above threshold.
        let logits = Tensor::from_slice(&[10.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let seed = fire_seed(1.0);
        let p = XtcProcessor::with_seed(0.1, 1.0, seed);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        // Should be identity — XTC only acts when ≥2 are above threshold.
        for (got, exp) in rows[0].iter().zip([10.0, 0.0, 0.0, 0.0]) {
            assert_eq!(*got, exp);
        }
    }

    #[test]
    fn xtc_noop_when_no_tokens_above_threshold() {
        // Threshold 0.5; with 1000 tokens of equal logit, each is 0.001
        // probability — way below 0.5.
        let logits = Tensor::from_slice(&vec![1.0f32; 100], vec![1, 100]).unwrap();
        let seed = fire_seed(1.0);
        let p = XtcProcessor::with_seed(0.5, 1.0, seed);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 100);
        for v in &rows[0] {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn xtc_three_above_threshold_masks_top_two() {
        // Probabilities target: ~0.50, ~0.27, ~0.18, ~0.05 with threshold
        // 0.10 → first three are above. XTC masks the top 2, keeps the
        // third.
        let logits = Tensor::from_slice(&[3.0f32, 2.4, 2.0, 1.0], vec![1, 4]).unwrap();
        let seed = fire_seed(1.0);
        let p = XtcProcessor::with_seed(0.1, 1.0, seed);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert!(rows[0][1].is_infinite() && rows[0][1] < 0.0);
        assert_eq!(rows[0][2], 2.0);
        assert_eq!(rows[0][3], 1.0);
    }

    #[test]
    fn xtc_invalid_threshold_errors() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = XtcProcessor::with_seed(1.0, 0.5, 1).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("threshold"));
        let e = XtcProcessor::with_seed(-0.1, 0.5, 1).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("threshold"));
    }

    #[test]
    fn xtc_invalid_probability_errors() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = XtcProcessor::with_seed(0.1, 1.5, 1).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("probability"));
        let e = XtcProcessor::with_seed(0.1, -0.1, 1).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("probability"));
    }

    #[test]
    fn xtc_name() {
        assert_eq!(XtcProcessor::with_seed(0.1, 0.5, 1).name(), "xtc");
    }

    #[test]
    fn xtc_deterministic_with_same_seed() {
        let logits = Tensor::from_slice(&[3.0f32, 2.4, 2.0], vec![1, 3]).unwrap();
        let p1 = XtcProcessor::with_seed(0.1, 0.5, 42);
        let p2 = XtcProcessor::with_seed(0.1, 0.5, 42);
        let r1 = p1.apply(&logits).unwrap();
        let r2 = p2.apply(&logits).unwrap();
        let rows1 = read_rows(&r1, 1, 3);
        let rows2 = read_rows(&r2, 1, 3);
        assert_eq!(rows1, rows2);
    }
}
