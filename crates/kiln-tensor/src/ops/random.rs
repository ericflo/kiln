//! `rand_uniform`, `rand_normal` — seedable random tensor constructors.
//!
//! splitmix64-seeded; F32 internal generation cast to BF16/F16 on
//! output. Used in:
//! - **Parameter initialization** (Xavier/Glorot/Kaiming via shape
//!   scaling)
//! - **Stochastic-rounding-policy test inputs**
//! - **MoE routing-noise injection** (Gumbel-like)
//! - **Bayesian / variational head sampling**
//!
//! Production paths should re-seed per training step from a higher-
//! level RNG; these are construction-time primitives only.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15u64);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
    z ^ (z >> 31)
}

fn next_u01(state: &mut u64) -> f32 {
    // 53-bit precision open-interval uniform — keeps log defined for
    // box-muller and Gumbel transforms.
    let r = splitmix64_step(state) >> 11;
    let denom = (1u64 << 53) as f64;
    ((r as f64 + 0.5) / denom) as f32
}

fn box_muller(state: &mut u64) -> f32 {
    // Standard Box-Muller transform: pair of uniform → pair of N(0,1).
    // Returns just one of the pair; the other is discarded (callers
    // can call again to get the second sample).
    let u1 = next_u01(state);
    let u2 = next_u01(state);
    (-2.0_f32 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Seedable uniform `[lo, hi)` tensor.
pub fn rand_uniform(shape: Vec<usize>, lo: f32, hi: f32, seed: u64, dtype: DType) -> Result<Tensor> {
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("rand_uniform: dtype must be F32/BF16/F16, got {dtype}");
    }
    if lo >= hi {
        bail!("rand_uniform: lo ({lo}) must be < hi ({hi})");
    }
    let n: usize = shape.iter().product();
    let mut state = seed.max(1);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        vals.push(lo + (hi - lo) * next_u01(&mut state));
    }
    finalize(dtype, shape, &vals)
}

/// Seedable standard normal — `N(mean, std²)` after the scale/shift.
pub fn rand_normal(shape: Vec<usize>, mean: f32, std: f32, seed: u64, dtype: DType) -> Result<Tensor> {
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("rand_normal: dtype must be F32/BF16/F16, got {dtype}");
    }
    if std < 0.0 {
        bail!("rand_normal: std must be ≥ 0, got {std}");
    }
    let n: usize = shape.iter().product();
    let mut state = seed.max(1);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        vals.push(mean + std * box_muller(&mut state));
    }
    finalize(dtype, shape, &vals)
}

fn finalize(dtype: DType, shape: Vec<usize>, vals: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; vals.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in vals.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in vals.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in vals.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn uniform_in_range() {
        let t = rand_uniform(vec![100], -1.0, 1.0, 42, DType::F32).unwrap();
        for v in read_f32(&t) {
            assert!(v >= -1.0 && v < 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn uniform_deterministic_with_same_seed() {
        let a = rand_uniform(vec![10], 0.0, 1.0, 7, DType::F32).unwrap();
        let b = rand_uniform(vec![10], 0.0, 1.0, 7, DType::F32).unwrap();
        assert_eq!(read_f32(&a), read_f32(&b));
    }

    #[test]
    fn uniform_lo_ge_hi_errors() {
        let e = rand_uniform(vec![1], 1.0, 1.0, 1, DType::F32).unwrap_err();
        assert!(e.to_string().contains("lo"));
    }

    #[test]
    fn normal_approximates_zero_mean() {
        let t = rand_normal(vec![10_000], 0.0, 1.0, 42, DType::F32).unwrap();
        let v = read_f32(&t);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        // Tolerance ~5σ/√N for N=10k → ~0.05.
        assert!(mean.abs() < 0.05, "mean {mean} too far from 0");
    }

    #[test]
    fn normal_approximates_unit_variance() {
        let t = rand_normal(vec![10_000], 0.0, 1.0, 42, DType::F32).unwrap();
        let v = read_f32(&t);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let var: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        // Empirical variance within ±0.1 of 1.0.
        assert!((var - 1.0).abs() < 0.1, "var {var} far from 1.0");
    }

    #[test]
    fn normal_zero_std_returns_constant() {
        let t = rand_normal(vec![10], 5.0, 0.0, 1, DType::F32).unwrap();
        for v in read_f32(&t) {
            assert!((v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn bf16_round_trips() {
        let t = rand_uniform(vec![4], 0.0, 1.0, 1, DType::BF16).unwrap();
        assert_eq!(t.dtype(), DType::BF16);
    }
}
