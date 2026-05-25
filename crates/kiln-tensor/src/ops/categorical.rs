//! `multinomial` — classical inverse-CDF categorical sampler.
//!
//! Provides an alternative to `GumbelSampler` for cases where the
//! caller has already computed probabilities (not logits) and wants
//! the standard CDF-based sampling path. Same output type — `[B]`
//! I64 token ids — but different algorithm.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Materialize `t` on CPU. CUDA inputs are D2H-copied via
/// `cuda_to_host_copy`; CPU inputs are cheap `Arc` bumps.
/// Multinomial is a per-row inverse-CDF sampler whose RNG
/// (`splitmix64`) lives on the CPU and is shared across batch
/// rows, so the sampling step is host-resident. The public API
/// must accept CUDA-resident probability tensors transparently —
/// upstream `softmax_last_dim` / sampler chain outputs already
/// live on the device they were produced on. See `#1082`.
fn to_cpu(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(t.device(), crate::Device::Cuda(_)) {
            return crate::cuda_to_host_copy(t);
        }
    }
    Ok(t.clone())
}

#[derive(Debug)]
pub struct Multinomial {
    rng: Mutex<u64>,
}

impl Multinomial {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xCAFEBABEDEADBEEFu64);
        Multinomial {
            rng: Mutex::new(seed.max(1)),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Multinomial {
            rng: Mutex::new(seed.max(1)),
        }
    }

    /// Sample one token id per row of `probs: [B, V]`. Probabilities
    /// must be non-negative; rows are renormalized internally so they
    /// don't need to sum to exactly 1.
    pub fn sample(&self, probs: &Tensor) -> Result<Tensor> {
        if probs.rank() != 2 {
            bail!(
                "Multinomial: probs must be rank-2 [batch, vocab], got {:?}",
                probs.shape()
            );
        }
        if !probs.is_contiguous() {
            bail!("Multinomial: probs must be contiguous");
        }
        if !matches!(probs.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!("Multinomial: dtype must be F32/BF16/F16, got {}", probs.dtype());
        }
        let batch = probs.shape()[0];
        let vocab = probs.shape()[1];
        let probs_host = to_cpu(probs)?;
        let cpu = probs_host
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("Multinomial: storage must be CpuStorage"))?;
        let bytes = cpu.as_bytes();
        let dtype = probs.dtype();
        let per = dtype.size_in_bytes();
        let mut rng = self.rng.lock().unwrap();
        let mut tokens = Vec::with_capacity(batch);
        for b in 0..batch {
            // Load row + accumulate cumulative mass.
            let mut row_sum = 0.0f32;
            let mut row = Vec::with_capacity(vocab);
            for v in 0..vocab {
                let idx = b * vocab + v;
                let p = match dtype {
                    DType::F32 => f32::from_le_bytes(bytes[idx * 4..idx * 4 + 4].try_into().unwrap()),
                    DType::BF16 => half::bf16::from_le_bytes(
                        bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                    )
                    .to_f32(),
                    DType::F16 => half::f16::from_le_bytes(
                        bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                    )
                    .to_f32(),
                    _ => unreachable!(),
                };
                if p < 0.0 {
                    bail!(
                        "Multinomial: probabilities must be non-negative, got {p} at row {b}, idx {v}"
                    );
                }
                row_sum += p;
                row.push(p);
            }
            if row_sum == 0.0 {
                bail!("Multinomial: row {b} has all-zero probabilities");
            }
            // Inverse-CDF sample.
            let r = next_u01(&mut rng) * row_sum;
            let mut acc = 0.0f32;
            let mut picked = (vocab - 1) as i64;
            for v in 0..vocab {
                acc += row[v];
                if r <= acc {
                    picked = v as i64;
                    break;
                }
            }
            tokens.push(picked);
        }
        let _ = per;
        let bytes_out: Vec<u8> = tokens.iter().flat_map(|&t| t.to_le_bytes()).collect();
        let cpu_out = CpuStorage::from_bytes(DType::I64, bytes_out)?;
        let storage: Storage = Arc::new(cpu_out);
        Tensor::from_parts(storage, Layout::contiguous(vec![batch]), TensorId::next())
    }
}

impl Default for Multinomial {
    fn default() -> Self {
        Self::new()
    }
}

fn next_u01(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15u64);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
    let r = (z ^ (z >> 31)) >> 11;
    let denom = (1u64 << 53) as f64;
    ((r as f64 + 0.5) / denom) as f32
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
    fn multinomial_deterministic_with_seed() {
        let probs = Tensor::from_slice(&[0.7f32, 0.2, 0.1], vec![1, 3]).unwrap();
        let m1 = Multinomial::with_seed(42);
        let m2 = Multinomial::with_seed(42);
        assert_eq!(read_i64(&m1.sample(&probs).unwrap()), read_i64(&m2.sample(&probs).unwrap()));
    }

    #[test]
    fn multinomial_only_one_class_always_returns_it() {
        let probs = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 0.0], vec![1, 4]).unwrap();
        let m = Multinomial::with_seed(1);
        for _ in 0..50 {
            let t = read_i64(&m.sample(&probs).unwrap());
            assert_eq!(t, vec![1]);
        }
    }

    #[test]
    fn multinomial_distribution_approximates() {
        // Simulate 10k draws from p=[0.5, 0.3, 0.2].
        let probs = Tensor::from_slice(&[0.5f32, 0.3, 0.2], vec![1, 3]).unwrap();
        let m = Multinomial::with_seed(7);
        let mut counts = [0u32; 3];
        for _ in 0..10_000 {
            let t = read_i64(&m.sample(&probs).unwrap())[0];
            counts[t as usize] += 1;
        }
        let total = 10_000.0_f32;
        let p0 = counts[0] as f32 / total;
        let p1 = counts[1] as f32 / total;
        let p2 = counts[2] as f32 / total;
        assert!((p0 - 0.5).abs() < 0.02, "p0={p0}");
        assert!((p1 - 0.3).abs() < 0.02, "p1={p1}");
        assert!((p2 - 0.2).abs() < 0.02, "p2={p2}");
    }

    #[test]
    fn multinomial_negative_prob_errors() {
        let probs = Tensor::from_slice(&[0.5f32, -0.1, 0.6], vec![1, 3]).unwrap();
        let m = Multinomial::with_seed(1);
        let e = m.sample(&probs).unwrap_err();
        assert!(e.to_string().contains("non-negative"));
    }

    #[test]
    fn multinomial_all_zero_row_errors() {
        let probs = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![1, 3]).unwrap();
        let m = Multinomial::with_seed(1);
        let e = m.sample(&probs).unwrap_err();
        assert!(e.to_string().contains("all-zero"));
    }

    /// CUDA parity: same RNG seed + same probability table produces
    /// byte-equal token ids when the prob tensor lives on CUDA vs
    /// CPU. The sampling step is CPU-resident (the splitmix64 RNG
    /// state lives in `self.rng`); the to_cpu helper just makes the
    /// probability bytes readable on host regardless of where they
    /// were produced.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_multinomial_parity() {
        let cdev = match candle_core::Device::cuda_if_available(0) {
            Ok(candle_core::Device::Cuda(c)) => c,
            _ => return,
        };
        let cdev = std::sync::Arc::new(cdev);

        let probs_cpu = Tensor::from_slice(
            &[0.5f32, 0.3, 0.2, 0.1, 0.6, 0.3],
            vec![2, 3],
        )
        .unwrap();
        let probs_cuda = probs_cpu
            .to_device(crate::Device::Cuda(0), Some(cdev.clone()))
            .unwrap();

        // Same RNG seed → same draws on either device.
        let m_cpu = Multinomial::with_seed(12345);
        let m_cuda = Multinomial::with_seed(12345);
        for _ in 0..50 {
            let t_cpu = read_i64(&m_cpu.sample(&probs_cpu).unwrap());
            let t_cuda = read_i64(&m_cuda.sample(&probs_cuda).unwrap());
            assert_eq!(t_cpu, t_cuda, "multinomial cuda parity drift");
        }
    }
}

