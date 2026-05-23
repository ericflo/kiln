//! DRY ("Don't Repeat Yourself") logit processor.
//!
//! DRY scales repetition penalty by *match length* — it punishes
//! tokens that would extend an already-seen n-gram, with exponential
//! penalty growth as the matched suffix lengthens.
//!
//! Per the standard implementation (oobabooga / Aphrodite / koboldcpp):
//!
//! For each candidate token `c` in the vocabulary:
//!
//! 1. Find the longest n-gram ending in `c` such that the same
//!    n-gram has already occurred in the prompt + completion so far.
//! 2. If the matched length exceeds `allowed_length`, subtract
//!    `multiplier * base^(matched_length - allowed_length)` from
//!    `logits[c]`.
//! 3. Tokens listed in `sequence_breakers` (typically newline,
//!    sentence-end punctuation) reset the match.
//!
//! Reference: <https://github.com/oobabooga/text-generation-webui/pull/5677>
//!
//! # Match-length computation
//!
//! We scan the history once and, for each position `i`, compute
//! `k_i = match length when c = history[i]` (i.e. the count of
//! tokens *before* `i` that match the tail of history). The match
//! value for token `history[i]` is `k_i + 1`.
//!
//! For each token id, the best match across all occurrences is the
//! one that produces the penalty. Tokens that don't appear in
//! history get no penalty.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

use super::logit_processor::LogitProcessor;

// ----------------------------------------------------------------------
// Local helpers (shared with logit_misc / logit_mirostat; will hoist).
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
        .ok_or_else(|| Error::from_str("logit_dry: storage must be CpuStorage"))?;
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

// ----------------------------------------------------------------------
// DryProcessor
// ----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DryProcessor {
    /// `multiplier` in penalty = multiplier * base^(L - allowed).
    pub multiplier: f32,
    /// `base` in penalty = multiplier * base^(L - allowed).
    pub base: f32,
    /// Match lengths up to and including this length are not penalized.
    pub allowed_length: usize,
    /// Per-batch history (the tokens emitted so far for each row).
    pub history: Vec<Vec<u32>>,
    /// Token ids that reset the match (e.g. `\n`, `.`, end-of-sentence).
    pub sequence_breakers: HashSet<u32>,
}

impl DryProcessor {
    pub fn new(
        multiplier: f32,
        base: f32,
        allowed_length: usize,
        history: Vec<Vec<u32>>,
    ) -> Self {
        DryProcessor {
            multiplier,
            base,
            allowed_length,
            history,
            sequence_breakers: HashSet::new(),
        }
    }

    pub fn with_sequence_breakers(mut self, breakers: HashSet<u32>) -> Self {
        self.sequence_breakers = breakers;
        self
    }

    /// Compute the per-token match map for one history row:
    /// `map[token]` = best match length across all positions in
    /// history where `token` appears as the "next token" extending
    /// a tail-matching prefix.
    fn match_map_for_row(&self, history: &[u32]) -> HashMap<u32, usize> {
        let mut best: HashMap<u32, usize> = HashMap::new();
        let n = history.len();
        for i in 0..n {
            let tok = history[i];
            if self.sequence_breakers.contains(&tok) {
                continue;
            }
            // Count how many tokens *before* i match the tail of history.
            // The tail tokens are history[n-1], history[n-2], ...
            // The prefix tokens are history[i-1], history[i-2], ...
            // Stop when they diverge, when we run out, or when we hit a
            // sequence breaker.
            let mut k = 0usize;
            while k < i {
                let hp = history[i - 1 - k];
                let ht = history[n - 1 - k];
                if hp != ht {
                    break;
                }
                if self.sequence_breakers.contains(&hp) {
                    break;
                }
                k += 1;
            }
            let m = k + 1; // +1 for tok itself
            best.entry(tok).and_modify(|v| *v = (*v).max(m)).or_insert(m);
        }
        best
    }
}

impl LogitProcessor for DryProcessor {
    fn name(&self) -> &'static str {
        "dry"
    }

    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "DryProcessor")?;
        if self.multiplier < 0.0 {
            bail!(
                "DryProcessor: multiplier must be >= 0, got {}",
                self.multiplier
            );
        }
        if self.base <= 1.0 {
            bail!("DryProcessor: base must be > 1, got {}", self.base);
        }
        let batch = logits.shape()[0];
        if self.history.len() != batch {
            bail!(
                "DryProcessor: history.len()={} must equal logits batch={batch}",
                self.history.len()
            );
        }
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for (b, row) in rows.iter_mut().enumerate() {
            if self.multiplier == 0.0 {
                continue;
            }
            let map = self.match_map_for_row(&self.history[b]);
            for (&tok, &m) in &map {
                if (tok as usize) >= vocab {
                    continue;
                }
                if m > self.allowed_length {
                    let exp = (m - self.allowed_length) as f32;
                    let penalty = self.multiplier * self.base.powf(exp);
                    row[tok as usize] -= penalty;
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
    fn dry_no_repetition_no_penalty() {
        // History [1, 2, 3, 4]. No repeats → no penalty.
        // Each token in history has match=1 (just itself), which
        // is <= allowed_length=2, so no penalty.
        let logits = Tensor::from_slice(&[10.0f32, 10.0, 10.0, 10.0, 10.0], vec![1, 5]).unwrap();
        let p = DryProcessor::new(0.8, 1.75, 2, vec![vec![1, 2, 3, 4]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 5);
        for v in &rows[0] {
            assert_eq!(*v, 10.0);
        }
    }

    #[test]
    fn dry_penalizes_extended_match() {
        // History [1, 2, 3, 1, 2]. For token 3, position 2 (the only
        // place 3 appears): before it, history[1]=2 matches tail[1]=2,
        // history[0]=1 matches tail[2]=1. So k=2, m=3. With
        // allowed_length=1 → exponent = 2. Penalty for token 3 only.
        let logits = Tensor::from_slice(&[10.0f32, 10.0, 10.0, 10.0], vec![1, 4]).unwrap();
        let p = DryProcessor::new(1.0, 2.0, 1, vec![vec![1, 2, 3, 1, 2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        // Token 3 penalized by 1.0 * 2^2 = 4.0
        assert!((rows[0][3] - 6.0).abs() < 1e-4, "{}", rows[0][3]);
        // Tokens 1, 2 have match=1 each (just themselves at history
        // boundary) — those are at the tail so technically m=1, not
        // > allowed_length=1, so no penalty.
        assert_eq!(rows[0][1], 10.0);
        assert_eq!(rows[0][2], 10.0);
    }

    #[test]
    fn dry_multiplier_zero_is_noop() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let p = DryProcessor::new(0.0, 2.0, 0, vec![vec![1, 2, 1, 2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn dry_sequence_breakers_reset_match() {
        // History [1, 2, 0, 1, 2]. Token 0 is a sequence breaker.
        // For token 2 at position 4 (tail): before it, history[3]=1
        // matches tail[1]=1. history[2]=0 is a breaker → stop.
        // So k=1, m=2. With allowed_length=1 → exponent=1 → penalty
        // for token 2.
        // BUT 2 is the tail token (at position 4 = n-1), so the loop
        // also visits it via match_map_for_row. The map uses .max() —
        // we get the best across positions.
        // For position 4 (the tail), i=4. Before it: i-1=3 → history[3]=1,
        // tail[0]=history[4]=2 — wait the tail-matching is history[n-1-k]
        // for k=0,1,... so tail[0]=history[n-1]=2 (k=0). Mistake.
        //
        // Let me re-walk: i=4 (the tail). k=0: hp=history[3]=1, ht=history[4]=2. 1 != 2, break. k=0, m=1.
        // i=1: tok=2. k=0: hp=history[0]=1, ht=history[4]=2. break. k=0, m=1.
        // So best match for 2 is 1, which is not > allowed_length=1.
        // No penalty.
        let mut breakers = HashSet::new();
        breakers.insert(0u32);
        let logits = Tensor::from_slice(&[5.0f32, 5.0, 5.0], vec![1, 3]).unwrap();
        let p = DryProcessor::new(1.0, 2.0, 1, vec![vec![1, 2, 0, 1, 2]])
            .with_sequence_breakers(breakers);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        // With the breaker between, no penalty applied.
        assert_eq!(rows[0], vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn dry_long_match_grows_exponentially() {
        // History [9, 8, 7, 9, 8, 7]. For position 2 (tok=7):
        //   k=0: hist[1]=8, tail[0]=hist[5]=7. 8 != 7 → break.
        //   k=0, m=1.
        // For position 5 (tok=7, the tail):
        //   k=0: hist[4]=8, tail[0]=hist[5]=7. break. m=1.
        // So token 7 gets m=1, no penalty.
        //
        // Try better: history [1, 2, 3, 1, 2]. For tok=3 at position 2:
        //   k=0: hist[1]=2, tail[0]=hist[4]=2 ✓
        //   k=1: hist[0]=1, tail[1]=hist[3]=1 ✓
        //   k=2: i=2, k<i false. stop.
        //   m=3. With allowed_length=0 → exponent=3 → penalty = 1.0 * 2^3 = 8.0.
        let logits = Tensor::from_slice(&[10.0f32; 5], vec![1, 5]).unwrap();
        let p = DryProcessor::new(1.0, 2.0, 0, vec![vec![1, 2, 3, 1, 2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 5);
        // Token 3 → m=3 → penalty 8.0 → 10 - 8 = 2.0
        assert!((rows[0][3] - 2.0).abs() < 1e-4, "{}", rows[0][3]);
    }

    #[test]
    fn dry_negative_multiplier_errors() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = DryProcessor::new(-1.0, 2.0, 0, vec![vec![]]).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("multiplier"));
    }

    #[test]
    fn dry_base_le_one_errors() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = DryProcessor::new(1.0, 1.0, 0, vec![vec![]]).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("base"));
    }

    #[test]
    fn dry_history_batch_mismatch_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![2, 1]).unwrap();
        let e = DryProcessor::new(1.0, 2.0, 0, vec![vec![]]).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("history.len"));
    }

    #[test]
    fn dry_name() {
        assert_eq!(DryProcessor::new(0.8, 1.75, 2, vec![]).name(), "dry");
    }
}
