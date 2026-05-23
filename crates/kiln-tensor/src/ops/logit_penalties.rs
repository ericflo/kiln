//! Repetition / frequency / presence penalty `LogitProcessor`s.
//!
//! Three standard sampler penalties for repetition control, exposed
//! as composable [`crate::ops::logit_processor::LogitProcessor`]
//! impls. They share a common `history: Vec<Vec<u32>>` input — one
//! list of previously-emitted tokens per batch row.
//!
//! # Repetition penalty (OpenAI-style)
//!
//! For each token id `t` in `history[b]`:
//! - If `logits[b, t] > 0`: `logits[b, t] /= penalty`
//! - If `logits[b, t] < 0`: `logits[b, t] *= penalty`
//!
//! `penalty = 1.0` is no-op; `penalty > 1.0` discourages repetition;
//! `penalty < 1.0` encourages it (rare).
//!
//! # Frequency penalty (OpenAI-style)
//!
//! `logits[b, t] -= alpha * count(t in history[b])`. Each occurrence
//! adds another `alpha` subtraction.
//!
//! # Presence penalty (OpenAI-style)
//!
//! `logits[b, t] -= alpha` for every distinct `t` that appears in
//! `history[b]` (regardless of count).
//!
//! All three are designed to compose: a typical chain runs
//! `repetition → frequency → presence → temperature → top_k → top_p`.

use crate::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};
use std::sync::Arc;

use super::logit_processor::LogitProcessor;

// ----------------------------------------------------------------------
// Common helpers (shared with logit_processor.rs through duplication
// for now; a per-backend hoist lands later).
// ----------------------------------------------------------------------

fn validate_logits(logits: &Tensor, history: &[Vec<u32>], name: &str) -> Result<()> {
    if logits.rank() != 2 {
        bail!(
            "{name}: logits must be rank-2 [batch, vocab], got shape {:?}",
            logits.shape()
        );
    }
    if !logits.is_contiguous() {
        bail!("{name}: logits must be contiguous");
    }
    if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "{name}: dtype must be F32/BF16/F16, got {}",
            logits.dtype()
        );
    }
    let batch = logits.shape()[0];
    if history.len() != batch {
        bail!(
            "{name}: history.len()={} must equal logits batch={batch}",
            history.len()
        );
    }
    Ok(())
}

fn load_all_rows_f32(logits: &Tensor, batch: usize, vocab: usize) -> Result<Vec<Vec<f32>>> {
    let cpu = logits
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("logit_penalties: storage must be CpuStorage"))?;
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
                DType::BF16 => {
                    half::bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32()
                }
                DType::F16 => {
                    half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32()
                }
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

// ----------------------------------------------------------------------
// RepetitionPenaltyProcessor
// ----------------------------------------------------------------------

/// OpenAI-style repetition penalty. `penalty=1.0` is no-op.
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    penalty: f32,
    history: Vec<Vec<u32>>,
}

impl RepetitionPenaltyProcessor {
    pub fn new(penalty: f32, history: Vec<Vec<u32>>) -> Self {
        RepetitionPenaltyProcessor { penalty, history }
    }
    pub fn penalty(&self) -> f32 {
        self.penalty
    }
    pub fn history(&self) -> &[Vec<u32>] {
        &self.history
    }
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn name(&self) -> &'static str {
        "penalty_repetition"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, &self.history, "RepetitionPenaltyProcessor")?;
        if self.penalty <= 0.0 {
            bail!(
                "RepetitionPenaltyProcessor: penalty must be > 0, got {}",
                self.penalty
            );
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for (b, row) in rows.iter_mut().enumerate() {
            for &id in &self.history[b] {
                if (id as usize) < vocab {
                    let v = row[id as usize];
                    let new = if v > 0.0 {
                        v / self.penalty
                    } else {
                        v * self.penalty
                    };
                    row[id as usize] = new;
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

// ----------------------------------------------------------------------
// FrequencyPenaltyProcessor
// ----------------------------------------------------------------------

/// OpenAI-style frequency penalty: subtract `alpha * count(t in history)`
/// from `logits[t]`.
#[derive(Debug, Clone)]
pub struct FrequencyPenaltyProcessor {
    alpha: f32,
    history: Vec<Vec<u32>>,
}

impl FrequencyPenaltyProcessor {
    pub fn new(alpha: f32, history: Vec<Vec<u32>>) -> Self {
        FrequencyPenaltyProcessor { alpha, history }
    }
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

impl LogitProcessor for FrequencyPenaltyProcessor {
    fn name(&self) -> &'static str {
        "penalty_frequency"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, &self.history, "FrequencyPenaltyProcessor")?;
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for (b, row) in rows.iter_mut().enumerate() {
            // Count occurrences in this batch row's history.
            let mut counts: std::collections::HashMap<u32, u32> =
                std::collections::HashMap::new();
            for &id in &self.history[b] {
                *counts.entry(id).or_insert(0) += 1;
            }
            for (id, count) in counts {
                if (id as usize) < vocab {
                    row[id as usize] -= self.alpha * count as f32;
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

// ----------------------------------------------------------------------
// PresencePenaltyProcessor
// ----------------------------------------------------------------------

/// OpenAI-style presence penalty: subtract `alpha` for every distinct
/// token id that appears in `history` (regardless of count).
#[derive(Debug, Clone)]
pub struct PresencePenaltyProcessor {
    alpha: f32,
    history: Vec<Vec<u32>>,
}

impl PresencePenaltyProcessor {
    pub fn new(alpha: f32, history: Vec<Vec<u32>>) -> Self {
        PresencePenaltyProcessor { alpha, history }
    }
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

impl LogitProcessor for PresencePenaltyProcessor {
    fn name(&self) -> &'static str {
        "penalty_presence"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, &self.history, "PresencePenaltyProcessor")?;
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for (b, row) in rows.iter_mut().enumerate() {
            let distinct: std::collections::HashSet<u32> =
                self.history[b].iter().copied().collect();
            for id in distinct {
                if (id as usize) < vocab {
                    row[id as usize] -= self.alpha;
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

    #[test]
    fn repetition_penalty_divides_positive() {
        // Logits [1.0, 2.0, -1.0]; history [0] with penalty 2.0:
        //   logits[0] = 1.0 / 2.0 = 0.5
        let logits = Tensor::from_slice(&[1.0f32, 2.0, -1.0], vec![1, 3]).unwrap();
        let p = RepetitionPenaltyProcessor::new(2.0, vec![vec![0]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert!((rows[0][0] - 0.5).abs() < 1e-6);
        assert!((rows[0][1] - 2.0).abs() < 1e-6);
        assert!((rows[0][2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_multiplies_negative() {
        // logits[-1.0]; history [0] with penalty 2.0:
        //   logits[0] = -1.0 * 2.0 = -2.0
        let logits = Tensor::from_slice(&[-1.0f32, 2.0], vec![1, 2]).unwrap();
        let p = RepetitionPenaltyProcessor::new(2.0, vec![vec![0]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 2);
        assert!((rows[0][0] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_one_is_noop() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let p = RepetitionPenaltyProcessor::new(1.0, vec![vec![0, 1, 2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn frequency_penalty_subtracts_alpha_times_count() {
        // logits [1.0, 1.0, 1.0]; history [0, 0, 1] with alpha=0.5:
        //   logits[0] -= 0.5 * 2 = 1.0 → 1.0 - 1.0 = 0.0
        //   logits[1] -= 0.5 * 1 = 0.5 → 1.0 - 0.5 = 0.5
        //   logits[2] unchanged
        let logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![1, 3]).unwrap();
        let p = FrequencyPenaltyProcessor::new(0.5, vec![vec![0, 0, 1]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert!((rows[0][0] - 0.0).abs() < 1e-6);
        assert!((rows[0][1] - 0.5).abs() < 1e-6);
        assert!((rows[0][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn presence_penalty_distinct_only() {
        // logits [1.0, 1.0, 1.0]; history [0, 0, 1] with alpha=0.5:
        //   distinct = {0, 1}
        //   logits[0] -= 0.5 → 0.5
        //   logits[1] -= 0.5 → 0.5
        //   logits[2] unchanged
        let logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![1, 3]).unwrap();
        let p = PresencePenaltyProcessor::new(0.5, vec![vec![0, 0, 1]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert!((rows[0][0] - 0.5).abs() < 1e-6);
        assert!((rows[0][1] - 0.5).abs() < 1e-6);
        assert!((rows[0][2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn multi_batch_independent_histories() {
        // Two batches with different histories.
        let logits = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0],
            vec![2, 3],
        )
        .unwrap();
        let p = RepetitionPenaltyProcessor::new(2.0, vec![vec![0], vec![2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 2, 3);
        // Batch 0: logits[0] = 1.0 / 2.0 = 0.5
        assert!((rows[0][0] - 0.5).abs() < 1e-6);
        // Batch 1: logits[2] = 30.0 / 2.0 = 15.0
        assert!((rows[1][2] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_history_id_silently_skipped() {
        // History id 99 > vocab=3 → silently skipped (typical when
        // history was tokenized against a different vocab or contains
        // BOS/EOS that don't map onto the model's logit space).
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let p = RepetitionPenaltyProcessor::new(2.0, vec![vec![99]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn history_len_must_match_batch() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let p = RepetitionPenaltyProcessor::new(2.0, vec![vec![], vec![]]);
        let e = p.apply(&logits).unwrap_err();
        assert!(e.to_string().contains("history.len()"));
    }

    #[test]
    fn repetition_penalty_zero_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let p = RepetitionPenaltyProcessor::new(0.0, vec![vec![]]);
        let e = p.apply(&logits).unwrap_err();
        assert!(e.to_string().contains("must be > 0"));
    }

    #[test]
    fn op_names() {
        assert_eq!(
            RepetitionPenaltyProcessor::new(1.5, vec![vec![]]).name(),
            "penalty_repetition"
        );
        assert_eq!(
            FrequencyPenaltyProcessor::new(0.5, vec![vec![]]).name(),
            "penalty_frequency"
        );
        assert_eq!(
            PresencePenaltyProcessor::new(0.5, vec![vec![]]).name(),
            "penalty_presence"
        );
    }
}
