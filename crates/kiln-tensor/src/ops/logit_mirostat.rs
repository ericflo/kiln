//! Mirostat 2.0 logit processor (Basu et al. 2020,
//! "Mirostat: A Neural Text Decoding Algorithm that Directly Controls
//! Perplexity").
//!
//! # Algorithm
//!
//! Mirostat targets a fixed *surprise* (per-token cross-entropy) `tau`
//! by maintaining a running threshold `mu`:
//!
//! 1. Initialize `mu = 2 * tau`.
//! 2. For each step:
//!    a. Compute `surprise(tok) = -log p(tok)` for every token.
//!    b. Mask any token whose surprise > `mu` (i.e. keep only tokens
//!       softer than the budget).
//!    c. (Caller) samples one token from the masked distribution.
//!    d. (Caller) calls [`Mirostat2Processor::update`] with the
//!       observed surprise of the sampled token.
//!    e. Update `mu` via simple gradient descent on the surprise gap:
//!       `mu := mu - lr * (observed_surprise - tau)`.
//!
//! # Design notes
//!
//! - The `LogitProcessor` trait is `&self`, so `mu` lives behind a
//!   `Mutex<f32>`. After sampling, the caller must call
//!   [`Mirostat2Processor::update`] (a `&self` method with interior
//!   mutability) to advance the state. The chain only ever sees the
//!   masking step.
//!
//! - We deliberately skip Mirostat 1.0 (with the surprise-quantile
//!   estimator). Mirostat 2.0 is strictly simpler, doesn't need a
//!   power-law fit, and matches the formulation everyone in the
//!   open-source community has converged on.
//!
//! - The processor is `[batch, vocab]` aware: each batch row carries
//!   its own `mu` (held in a `Vec<f32>` behind the mutex), so
//!   independent streams in a batch evolve independently.

use std::sync::Mutex;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};
use std::sync::Arc;

use super::logit_processor::LogitProcessor;

// ----------------------------------------------------------------------
// Local helpers (mirrors of logit_misc; will be unified in Phase 2.x).
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
        .ok_or_else(|| Error::from_str("mirostat: storage must be CpuStorage"))?;
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
// Mirostat 2.0
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct Mirostat2Processor {
    /// Target surprise, in nats. Common values: 3.0 (less spicy)
    /// through 8.0 (more spicy).
    tau: f32,
    /// Learning rate for the threshold update. Per the paper, 0.1
    /// is a stable default.
    eta: f32,
    /// Per-batch-row threshold. Initialized lazily on first
    /// `apply` to match batch size; updated on `update`.
    mu: Mutex<Option<Vec<f32>>>,
}

impl Mirostat2Processor {
    pub fn new(tau: f32, eta: f32) -> Self {
        Mirostat2Processor {
            tau,
            eta,
            mu: Mutex::new(None),
        }
    }

    pub fn tau(&self) -> f32 {
        self.tau
    }

    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// Current threshold for a given batch row, if initialized.
    pub fn mu(&self, batch: usize) -> Option<f32> {
        self.mu.lock().unwrap().as_ref().and_then(|v| v.get(batch).copied())
    }

    /// Called by the sampler after observing a token. `observed`
    /// is the cross-entropy of the sampled token *before* the
    /// Mirostat mask was applied (i.e. `-log p(tok)` over the
    /// raw distribution).
    pub fn update(&self, batch: usize, observed_surprise: f32) -> Result<()> {
        let mut guard = self.mu.lock().unwrap();
        let mus = guard
            .as_mut()
            .ok_or_else(|| Error::from_str("Mirostat2Processor::update before apply"))?;
        if batch >= mus.len() {
            bail!(
                "Mirostat2Processor::update batch={batch} out of range {}",
                mus.len()
            );
        }
        let err = observed_surprise - self.tau;
        mus[batch] -= self.eta * err;
        Ok(())
    }
}

impl LogitProcessor for Mirostat2Processor {
    fn name(&self) -> &'static str {
        "mirostat2"
    }

    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        validate_logits(logits, "Mirostat2Processor")?;
        if self.tau <= 0.0 {
            bail!("Mirostat2Processor: tau must be positive, got {}", self.tau);
        }
        if self.eta <= 0.0 {
            bail!("Mirostat2Processor: eta must be positive, got {}", self.eta);
        }
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];

        // Lazily initialize mu = 2 * tau for each batch row, growing
        // if the batch size changes between calls.
        {
            let mut guard = self.mu.lock().unwrap();
            let mus = guard.get_or_insert_with(|| vec![2.0 * self.tau; batch]);
            if mus.len() < batch {
                mus.resize(batch, 2.0 * self.tau);
            }
        }

        let mut rows = load_all_rows_f32(logits, batch, vocab)?;
        let mus = self.mu.lock().unwrap().clone().unwrap();
        for (b, row) in rows.iter_mut().enumerate() {
            // Compute softmax + surprise = -log p.
            let max_v = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_v).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            // Mask any token whose surprise exceeds the current mu.
            let mu = mus[b];
            for (i, &e) in exps.iter().enumerate() {
                let p = e / sum_exp;
                // Guard against -log(0) = +inf; treat exact-zero as
                // already past the surprise budget.
                let surprise = if p > 0.0 { -p.ln() } else { f32::INFINITY };
                if surprise > mu {
                    row[i] = f32::NEG_INFINITY;
                }
            }
            // If we masked *everything* (mu too tight), keep the
            // argmax to preserve sampler safety — same fallback
            // every Mirostat impl uses.
            if row.iter().all(|v| v.is_infinite() && *v < 0.0) {
                let mut max_i = 0;
                let mut max_v = f32::NEG_INFINITY;
                for (i, &e) in exps.iter().enumerate() {
                    if e > max_v {
                        max_v = e;
                        max_i = i;
                    }
                }
                // Restore original logit by reading the input bytes
                // again. Easier: just put 0.0 there — since every
                // other value is -inf, the sampler picks this one
                // deterministically.
                row[max_i] = 0.0;
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
    fn mirostat_init_keeps_high_probability_tokens() {
        // Sharp distribution: index 0 is ~probability 1. Surprise(0) ≈ 0,
        // others ≈ +inf. mu starts at 2*tau, so tau=5 → mu=10. Index 0
        // survives, others are masked.
        let logits = Tensor::from_slice(&[10.0f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let p = Mirostat2Processor::new(5.0, 0.1);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 4);
        assert!(rows[0][0].is_finite());
        for i in 1..4 {
            assert!(rows[0][i].is_infinite() && rows[0][i] < 0.0);
        }
    }

    #[test]
    fn mirostat_apply_initializes_mu_to_2tau() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let p = Mirostat2Processor::new(3.0, 0.1);
        assert_eq!(p.mu(0), None);
        let _ = p.apply(&logits).unwrap();
        assert_eq!(p.mu(0), Some(6.0));
    }

    #[test]
    fn mirostat_update_decreases_mu_when_observed_above_tau() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let p = Mirostat2Processor::new(3.0, 0.1);
        p.apply(&logits).unwrap();
        // observed surprise 5.0 > tau 3.0 → mu should shrink by 0.1*2 = 0.2.
        p.update(0, 5.0).unwrap();
        let mu = p.mu(0).unwrap();
        assert!((mu - 5.8).abs() < 1e-6, "mu={mu}");
    }

    #[test]
    fn mirostat_update_increases_mu_when_observed_below_tau() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let p = Mirostat2Processor::new(3.0, 0.1);
        p.apply(&logits).unwrap();
        p.update(0, 1.0).unwrap();
        // 1.0 < 3.0 → mu grows by 0.1*2 = 0.2.
        let mu = p.mu(0).unwrap();
        assert!((mu - 6.2).abs() < 1e-6, "mu={mu}");
    }

    #[test]
    fn mirostat_per_batch_independent_state() {
        let logits =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let p = Mirostat2Processor::new(3.0, 0.1);
        p.apply(&logits).unwrap();
        p.update(0, 5.0).unwrap();
        p.update(1, 1.0).unwrap();
        assert!((p.mu(0).unwrap() - 5.8).abs() < 1e-6);
        assert!((p.mu(1).unwrap() - 6.2).abs() < 1e-6);
    }

    #[test]
    fn mirostat_update_before_apply_errors() {
        let p = Mirostat2Processor::new(3.0, 0.1);
        let e = p.update(0, 1.0).unwrap_err();
        assert!(e.to_string().contains("before apply"));
    }

    #[test]
    fn mirostat_tau_must_be_positive() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = Mirostat2Processor::new(0.0, 0.1).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("tau"));
    }

    #[test]
    fn mirostat_eta_must_be_positive() {
        let logits = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = Mirostat2Processor::new(3.0, 0.0).apply(&logits).unwrap_err();
        assert!(e.to_string().contains("eta"));
    }

    #[test]
    fn mirostat_argmax_fallback_when_everything_masked() {
        // mu very small (start = 2*tiny). Every surprise > mu. We keep
        // the argmax (index 1 here) with a finite value.
        let logits = Tensor::from_slice(&[1.0f32, 5.0, 2.0], vec![1, 3]).unwrap();
        let p = Mirostat2Processor::new(1e-6, 0.1);
        let out = p.apply(&logits).unwrap();
        let rows = read_rows(&out, 1, 3);
        // index 1 is the argmax of the input.
        assert!(rows[0][1].is_finite());
        // the other two must be -inf.
        assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
        assert!(rows[0][2].is_infinite() && rows[0][2] < 0.0);
    }

    #[test]
    fn mirostat_name() {
        assert_eq!(Mirostat2Processor::new(3.0, 0.1).name(), "mirostat2");
    }
}
