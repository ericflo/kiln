//! `LogitProcessor` chain — composable sampler operating on logits
//! in place.
//!
//! Per the Phase 4 issue bullet:
//!
//! > **`LogitProcessor` chain as a first-class abstraction.** Audit
//! > shows the current sampler at `crates/kiln-core/src/sampling.rs`
//! > has top-k, top-p (nucleus), min-p, temperature, and the three
//! > penalties — but they're hand-wired, not composed. Define
//! > `LogitProcessor` as a stackable trait operating on a
//! > `Tensor<[batch, vocab]>` in place, dispatched on the StreamPlanner.
//! >
//! > The chain order, fixed for Qwen3.5-4B:
//! > `penalties → ngram-block → grammar-mask → logit_bias →
//! > temperature → top-k → top-p / min-p / typical-p / Mirostat →
//! > categorical sample (Gumbel-max)`. Each processor is one device
//! > kernel + one parity test against a CPU reference.
//!
//! # Phase 1.48 scope
//!
//! Backend-agnostic CPU reference + the three highest-traffic
//! processors:
//!
//! - [`TemperatureProcessor`] — divides logits by `temperature`.
//!   `temperature = 0` greedy-paths (handled by `argmax_last_dim`,
//!   not by this chain).
//! - [`TopKProcessor`] — masks all but the top-K logits with `-inf`.
//! - [`TopPProcessor`] — nucleus sampling; masks logits outside the
//!   smallest set whose cumulative softmax probability ≥ `p`.
//!
//! The remaining chain steps (penalties, ngram-block, grammar-mask,
//! logit_bias, min-p, typical-p, Mirostat, DRY, XTC, categorical
//! sample) land in subsequent PRs as additional `LogitProcessor`
//! impls.
//!
//! # Composition
//!
//! `LogitProcessorChain` carries `Vec<Box<dyn LogitProcessor>>` and
//! applies each in order:
//!
//! ```rust,no_run
//! use kiln_tensor::ops::logit_processor::*;
//! use kiln_tensor::Tensor;
//!
//! let logits = Tensor::from_slice(&[0.0f32; 16], vec![1, 16]).unwrap();
//! let chain = LogitProcessorChain::new(vec![
//!     Box::new(TemperatureProcessor::new(0.7)),
//!     Box::new(TopKProcessor::new(50)),
//!     Box::new(TopPProcessor::new(0.95)),
//! ]);
//! let _post = chain.apply(&logits).unwrap();
//! ```
//!
//! The output `Tensor` is a fresh allocation today; an in-place
//! variant lands when `kiln_tensor::Tensor` exposes interior
//! mutability (Phase 1.x version-counter story).

use crate::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Stackable logit processor. Each impl applies a per-batch-row
/// transform to logits `[batch, vocab]`.
///
/// `name()` is used by the Phase 9 audit + by trace labels.
pub trait LogitProcessor: Send + Sync + std::fmt::Debug {
    /// Stable name (e.g. "temperature", "top_k", "top_p").
    fn name(&self) -> &'static str;

    /// Apply the processor. Returns a fresh `Tensor` of the same
    /// shape + dtype as `logits`.
    fn apply(&self, logits: &Tensor) -> Result<Tensor>;
}

/// Chain of [`LogitProcessor`]s applied in declaration order.
#[derive(Debug)]
pub struct LogitProcessorChain {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl LogitProcessorChain {
    pub fn new(processors: Vec<Box<dyn LogitProcessor>>) -> Self {
        LogitProcessorChain { processors }
    }

    pub fn empty() -> Self {
        LogitProcessorChain { processors: Vec::new() }
    }

    pub fn push(&mut self, p: Box<dyn LogitProcessor>) {
        self.processors.push(p);
    }

    pub fn len(&self) -> usize {
        self.processors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }

    /// Apply every processor in order. Returns the final logits.
    pub fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        let mut current = logits.clone();
        for p in &self.processors {
            current = p.apply(&current)?;
        }
        Ok(current)
    }

    /// Borrow the ordered names of processors in the chain. Used by
    /// trace labels and by `bench-results/` reports.
    pub fn names(&self) -> Vec<&'static str> {
        self.processors.iter().map(|p| p.name()).collect()
    }
}

// ----------------------------------------------------------------------
// TemperatureProcessor
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    pub const fn new(temperature: f32) -> Self {
        TemperatureProcessor { temperature }
    }
}

impl LogitProcessor for TemperatureProcessor {
    fn name(&self) -> &'static str {
        "temperature"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if !logits.is_contiguous() {
            bail!("TemperatureProcessor: logits must be contiguous");
        }
        if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "TemperatureProcessor: dtype must be F32/BF16/F16, got {}",
                logits.dtype()
            );
        }
        if self.temperature == 0.0 {
            bail!(
                "TemperatureProcessor: temperature=0 — use argmax_last_dim for greedy decoding"
            );
        }
        if self.temperature <= 0.0 {
            bail!(
                "TemperatureProcessor: temperature must be positive, got {}",
                self.temperature
            );
        }
        let inv_t = 1.0_f32 / self.temperature;
        scale_all(logits, inv_t)
    }
}

// ----------------------------------------------------------------------
// TopKProcessor
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TopKProcessor {
    pub k: usize,
}

impl TopKProcessor {
    pub const fn new(k: usize) -> Self {
        TopKProcessor { k }
    }
}

impl LogitProcessor for TopKProcessor {
    fn name(&self) -> &'static str {
        "top_k"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if logits.rank() != 2 {
            bail!(
                "TopKProcessor: logits must be rank-2 [batch, vocab], got shape {:?}",
                logits.shape()
            );
        }
        if !logits.is_contiguous() {
            bail!("TopKProcessor: logits must be contiguous");
        }
        if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "TopKProcessor: dtype must be F32/BF16/F16, got {}",
                logits.dtype()
            );
        }
        if self.k == 0 {
            bail!("TopKProcessor: k=0 would mask every logit — use a positive k");
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let k = self.k.min(vocab);
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for row in rows.iter_mut() {
            // Find the k-th largest value via simple O(n log n) sort.
            // (Quickselect optimization is a per-backend job once
            // GPU paths land.)
            let mut sorted = row.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = sorted[k - 1];
            for v in row.iter_mut() {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

// ----------------------------------------------------------------------
// TopPProcessor (nucleus sampling)
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TopPProcessor {
    pub p: f32,
}

impl TopPProcessor {
    pub const fn new(p: f32) -> Self {
        TopPProcessor { p }
    }
}

impl LogitProcessor for TopPProcessor {
    fn name(&self) -> &'static str {
        "top_p"
    }
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if logits.rank() != 2 {
            bail!(
                "TopPProcessor: logits must be rank-2 [batch, vocab], got shape {:?}",
                logits.shape()
            );
        }
        if !logits.is_contiguous() {
            bail!("TopPProcessor: logits must be contiguous");
        }
        if !matches!(logits.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "TopPProcessor: dtype must be F32/BF16/F16, got {}",
                logits.dtype()
            );
        }
        if !(0.0 < self.p && self.p <= 1.0) {
            bail!("TopPProcessor: p must be in (0, 1], got {}", self.p);
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];
        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        for row in rows.iter_mut() {
            // 1. Sort descending with index tracking.
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap_or(std::cmp::Ordering::Equal));
            // 2. Compute softmax over the sorted row.
            let max_v = idx.first().map(|&i| row[i]).unwrap_or(f32::NEG_INFINITY);
            let exps: Vec<f32> = idx.iter().map(|&i| (row[i] - max_v).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            // 3. Walk the sorted indices accumulating probability mass;
            //    keep entries until the cumulative sum reaches `p`.
            let mut keep = vec![false; vocab];
            let mut acc = 0.0_f32;
            for (rank, &i) in idx.iter().enumerate() {
                acc += exps[rank] / sum_exp;
                keep[i] = true;
                if acc >= self.p {
                    break;
                }
            }
            // 4. Mask out non-kept positions.
            for (i, k) in keep.iter().enumerate() {
                if !*k {
                    row[i] = f32::NEG_INFINITY;
                }
            }
        }
        store_rows(logits.dtype(), logits.shape(), &rows)
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn scale_all(logits: &Tensor, scale: f32) -> Result<Tensor> {
    let dtype = logits.dtype();
    let cpu = logits
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("logit_processor: storage must be CpuStorage on CPU"))?;
    let bytes = cpu.as_bytes();
    let n = logits.element_count();
    let per = dtype.size_in_bytes();
    let mut out = vec![0u8; n * per];
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let v =
                    f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) * scale;
                out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32()
                    * scale;
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let v = half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32()
                    * scale;
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(logits.shape().to_vec()), TensorId::next())
}

fn load_all_rows_f32(logits: &Tensor, batch: usize, vocab: usize) -> Result<Vec<Vec<f32>>> {
    let cpu = logits
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("logit_processor: storage must be CpuStorage"))?;
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
    match dtype {
        DType::F32 => {
            for (b, row) in rows.iter().enumerate() {
                for (v, &val) in row.iter().enumerate() {
                    out[(b * vocab + v) * 4..(b * vocab + v) * 4 + 4]
                        .copy_from_slice(&val.to_le_bytes());
                }
            }
        }
        DType::BF16 => {
            for (b, row) in rows.iter().enumerate() {
                for (v, &val) in row.iter().enumerate() {
                    out[(b * vocab + v) * 2..(b * vocab + v) * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(val).to_le_bytes());
                }
            }
        }
        DType::F16 => {
            for (b, row) in rows.iter().enumerate() {
                for (v, &val) in row.iter().enumerate() {
                    out[(b * vocab + v) * 2..(b * vocab + v) * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(val).to_le_bytes());
                }
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_rows(t: &Tensor, batch: usize, vocab: usize) -> Vec<Vec<f32>> {
        load_all_rows_f32(t, batch, vocab).unwrap()
    }

    #[test]
    fn temperature_scales_logits() {
        let logits = Tensor::from_slice(&[2.0f32, 4.0, 6.0], vec![1, 3]).unwrap();
        let p = TemperatureProcessor::new(2.0);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn temperature_zero_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let e = TemperatureProcessor::new(0.0).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("temperature=0"));
    }

    #[test]
    fn temperature_negative_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let e = TemperatureProcessor::new(-1.0).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("positive"));
    }

    #[test]
    fn top_k_masks_below_threshold() {
        // [1.0, 5.0, 3.0, 2.0] with k=2 → keep [5.0, 3.0]; mask [1.0, 2.0]
        let logits = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 2.0], vec![1, 4]).unwrap();
        let out = TopKProcessor::new(2).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert_eq!(rows[0][1], 5.0);
        assert_eq!(rows[0][2], 3.0);
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert!(rows[0][3].is_infinite() && rows[0][3] < 0.0);
    }

    #[test]
    fn top_k_larger_than_vocab_keeps_all() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let out = TopKProcessor::new(100).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn top_p_nucleus_keeps_top_mass() {
        // Logits chosen so softmax mass is concentrated: e^10 dominates.
        // [10.0, 0.0, 0.0, 0.0] → softmax ~ [1, 0, 0, 0]; with p=0.5
        // we keep only index 0; mask the others.
        let logits = Tensor::from_slice(&[10.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let out = TopPProcessor::new(0.5).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert_eq!(rows[0][0], 10.0);
        for i in 1..4 {
            assert!(
                rows[0][i].is_infinite() && rows[0][i] < 0.0,
                "expected mask at i={i}, got {}",
                rows[0][i]
            );
        }
    }

    #[test]
    fn top_p_at_one_keeps_everything() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let out = TopPProcessor::new(1.0).apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        for v in &rows[0] {
            assert!(v.is_finite(), "p=1.0 should keep every logit; got {v}");
        }
    }

    #[test]
    fn chain_applies_in_order() {
        // Temperature 2.0 halves logits; top-K keeps top 2.
        // Input: [2, 4, 6, 8] → after temp: [1, 2, 3, 4] → after top-K(2):
        // [-inf, -inf, 3, 4]
        let logits = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 8.0], vec![1, 4]).unwrap();
        let chain = LogitProcessorChain::new(vec![
            Box::new(TemperatureProcessor::new(2.0)),
            Box::new(TopKProcessor::new(2)),
        ]);
        let out = chain.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert!(rows[0][1].is_infinite() && rows[0][1] < 0.0);
        assert_eq!(rows[0][2], 3.0);
        assert_eq!(rows[0][3], 4.0);
    }

    #[test]
    fn empty_chain_is_identity() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let chain = LogitProcessorChain::empty();
        let out = chain.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        assert_eq!(rows[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn chain_names_lists_processors() {
        let chain = LogitProcessorChain::new(vec![
            Box::new(TemperatureProcessor::new(0.7)),
            Box::new(TopKProcessor::new(50)),
            Box::new(TopPProcessor::new(0.95)),
        ]);
        assert_eq!(chain.names(), vec!["temperature", "top_k", "top_p"]);
    }

    #[test]
    fn top_k_zero_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let e = TopKProcessor::new(0).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("k=0"));
    }

    #[test]
    fn top_p_invalid_p_errors() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let e = TopPProcessor::new(0.0).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("p must be"));
        let e = TopPProcessor::new(1.5).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("p must be"));
    }
}
