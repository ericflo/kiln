//! `NgramBlockProcessor` + `LogitBiasProcessor`.
//!
//! Two more sampler-chain steps:
//!
//! - **NgramBlock** — given a per-batch token history and an
//!   n-gram size, mask any token that would form a forbidden
//!   n-gram with the last `n-1` history tokens. The forbidden
//!   set is whatever n-grams have appeared in the history.
//! - **LogitBias** — a per-token-id additive bias map. Used by the
//!   OpenAI chat API's `logit_bias` parameter for forced-token
//!   inclusion / exclusion + grammar-shaped boosts.

use std::collections::{HashMap, HashSet};

use crate::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};
use std::sync::Arc;

use super::logit_processor::LogitProcessor;

// ----------------------------------------------------------------------
// Shared helpers (one more duplication; Phase 2.x hoist to a util crate
// once the kernel ports stabilize).
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
        .ok_or_else(|| Error::from_str("logit_misc: storage must be CpuStorage"))?;
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
// NgramBlockProcessor
// ----------------------------------------------------------------------

/// Block any token that would complete an n-gram that has already
/// appeared in this batch row's history.
///
/// Algorithm: walk the history, collect all n-grams as
/// `(prefix_tail, last_token)` pairs. At sample time, look at the
/// last `n-1` tokens of the history: if `(history_tail, candidate)`
/// is in the seen-set, mask candidate with -inf.
///
/// `n = 1` is a degenerate case — every history token gets blocked
/// (the no-prefix lookup matches the empty prefix). We reject `n < 2`.
#[derive(Debug, Clone)]
pub struct NgramBlockProcessor {
    n: usize,
    history: Vec<Vec<u32>>,
}

impl NgramBlockProcessor {
    pub fn new(n: usize, history: Vec<Vec<u32>>) -> Self {
        NgramBlockProcessor { n, history }
    }
    pub fn n(&self) -> usize {
        self.n
    }
}

impl LogitProcessor for NgramBlockProcessor {
    fn name(&self) -> &'static str {
        "ngram_block"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "NgramBlockProcessor")?;
        if self.n < 2 {
            bail!("NgramBlockProcessor: n must be ≥ 2, got {}", self.n);
        }
        let batch = logits.shape()[0];
        if self.history.len() != batch {
            bail!(
                "NgramBlockProcessor: history.len()={} must equal logits batch={batch}",
                self.history.len()
            );
        }
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        let prefix_len = self.n - 1;
        for (b, row) in rows.iter_mut().enumerate() {
            let hist = &self.history[b];
            // Collect the set of (prefix, last) pairs seen in history.
            let mut seen: HashSet<(Vec<u32>, u32)> = HashSet::new();
            if hist.len() >= self.n {
                for start in 0..=hist.len() - self.n {
                    let prefix = hist[start..start + prefix_len].to_vec();
                    let last = hist[start + prefix_len];
                    seen.insert((prefix, last));
                }
            }
            // For the upcoming sample, the prefix is the LAST `prefix_len`
            // tokens of the history. If it's shorter than that, no block.
            if hist.len() >= prefix_len {
                let tail = hist[hist.len() - prefix_len..].to_vec();
                for candidate in 0u32..vocab as u32 {
                    if seen.contains(&(tail.clone(), candidate)) {
                        row[candidate as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

// ----------------------------------------------------------------------
// LogitBiasProcessor
// ----------------------------------------------------------------------

/// Per-token-id additive bias. Matches OpenAI's `logit_bias` API:
/// each entry adds `bias[id]` to `logits[id]` for every batch row.
/// `bias=-inf` is "ban this token entirely"; `bias=+inf` is
/// "always emit this token" (with caveats — top-K may still mask it
/// if downstream).
#[derive(Debug, Clone)]
pub struct LogitBiasProcessor {
    bias: HashMap<u32, f32>,
}

impl LogitBiasProcessor {
    pub fn new(bias: HashMap<u32, f32>) -> Self {
        LogitBiasProcessor { bias }
    }
}

impl LogitProcessor for LogitBiasProcessor {
    fn name(&self) -> &'static str {
        "logit_bias"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "LogitBiasProcessor")?;
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for row in rows.iter_mut() {
            for (&id, &b) in &self.bias {
                if (id as usize) < vocab {
                    row[id as usize] += b;
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

    // ─── NgramBlock ────────────────────────────────────────────────

    #[test]
    fn ngram_block_2gram_blocks_repeated_token() {
        // History: [1, 2, 3, 2]. n=2 → block any (prefix, last) seen.
        // Pairs seen: (1)→2, (2)→3, (3)→2. Tail = [2]. Block
        // candidate 3 (because (2, 3) is in seen).
        let logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![1, 4]).unwrap();
        let p = NgramBlockProcessor::new(2, vec![vec![1, 2, 3, 2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert_eq!(rows[0][0], 1.0);
        assert_eq!(rows[0][1], 1.0);
        assert_eq!(rows[0][2], 1.0);
        assert!(rows[0][3].is_infinite() && rows[0][3] < 0.0);
    }

    #[test]
    fn ngram_block_3gram() {
        // History [1, 2, 3, 1, 2]. n=3 → pairs ((1,2)→3, (2,3)→1, (3,1)→2).
        // Tail = [1, 2]. (1, 2, 3) seen → block 3.
        let logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![1, 4]).unwrap();
        let p = NgramBlockProcessor::new(3, vec![vec![1, 2, 3, 1, 2]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert!(rows[0][3].is_infinite() && rows[0][3] < 0.0);
        // Others unchanged.
        for i in 0..3 {
            assert_eq!(rows[0][i], 1.0);
        }
    }

    #[test]
    fn ngram_block_short_history_is_noop() {
        // History [1]; n=2 → no n-grams can have been seen.
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let p = NgramBlockProcessor::new(2, vec![vec![1]]);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn ngram_block_n_less_than_2_errors() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = NgramBlockProcessor::new(1, vec![vec![]]).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("n must be ≥ 2"));
    }

    // ─── LogitBias ─────────────────────────────────────────────────

    #[test]
    fn logit_bias_adds_per_token() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let mut bias = HashMap::new();
        bias.insert(0u32, 10.0);
        bias.insert(2u32, -100.0);
        let p = LogitBiasProcessor::new(bias);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0][0], 11.0);
        assert_eq!(rows[0][1], 2.0);
        assert_eq!(rows[0][2], -97.0);
    }

    #[test]
    fn logit_bias_neg_inf_bans_token() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let mut bias = HashMap::new();
        bias.insert(0u32, f32::NEG_INFINITY);
        let p = LogitBiasProcessor::new(bias);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert_eq!(rows[0][1], 2.0);
        assert_eq!(rows[0][2], 3.0);
    }

    #[test]
    fn logit_bias_out_of_range_id_silently_skipped() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let mut bias = HashMap::new();
        bias.insert(99u32, 1000.0);
        let p = LogitBiasProcessor::new(bias);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 2);
        assert_eq!(rows[0], vec![1.0, 2.0]);
    }

    #[test]
    fn names() {
        assert_eq!(
            NgramBlockProcessor::new(3, vec![vec![]]).name(),
            "ngram_block"
        );
        assert_eq!(LogitBiasProcessor::new(HashMap::new()).name(), "logit_bias");
    }
}
