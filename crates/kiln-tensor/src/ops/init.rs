//! Parameter initializers: `xavier_uniform`, `xavier_normal`,
//! `kaiming_uniform`, `kaiming_normal`.
//!
//! Standard initialization schemes for matrix-shaped parameters
//! (rank-2 `[fan_in, fan_out]` weight tensors).
//!
//! - **Xavier (Glorot)** — designed for sigmoid/tanh activations:
//!   limit `a = √(6 / (fan_in + fan_out))` for uniform; `std =
//!   √(2 / (fan_in + fan_out))` for normal.
//! - **Kaiming (He)** — designed for ReLU activations: limit
//!   `a = √(6 / fan_in)` for uniform; `std = √(2 / fan_in)` for
//!   normal.
//!
//! Wired on top of `rand_uniform` / `rand_normal`.

use crate::{bail, DType, Result, Tensor};

use super::random::{rand_normal, rand_uniform};

fn rank2_or_error(shape: &[usize], name: &str) -> Result<(usize, usize)> {
    if shape.len() != 2 {
        bail!(
            "{name}: shape must be rank-2 [fan_in, fan_out], got {:?}",
            shape
        );
    }
    Ok((shape[0], shape[1]))
}

pub fn xavier_uniform(shape: Vec<usize>, seed: u64, dtype: DType) -> Result<Tensor> {
    let (fan_in, fan_out) = rank2_or_error(&shape, "xavier_uniform")?;
    let a = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    rand_uniform(shape, -a, a, seed, dtype)
}

pub fn xavier_normal(shape: Vec<usize>, seed: u64, dtype: DType) -> Result<Tensor> {
    let (fan_in, fan_out) = rank2_or_error(&shape, "xavier_normal")?;
    let std = (2.0_f32 / (fan_in + fan_out) as f32).sqrt();
    rand_normal(shape, 0.0, std, seed, dtype)
}

pub fn kaiming_uniform(shape: Vec<usize>, seed: u64, dtype: DType) -> Result<Tensor> {
    let (fan_in, _fan_out) = rank2_or_error(&shape, "kaiming_uniform")?;
    let a = (6.0_f32 / fan_in as f32).sqrt();
    rand_uniform(shape, -a, a, seed, dtype)
}

pub fn kaiming_normal(shape: Vec<usize>, seed: u64, dtype: DType) -> Result<Tensor> {
    let (fan_in, _fan_out) = rank2_or_error(&shape, "kaiming_normal")?;
    let std = (2.0_f32 / fan_in as f32).sqrt();
    rand_normal(shape, 0.0, std, seed, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuStorage;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn xavier_uniform_within_bound() {
        let fan_in = 100;
        let fan_out = 200;
        let a = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        let t = xavier_uniform(vec![fan_in, fan_out], 42, DType::F32).unwrap();
        for v in read_f32(&t) {
            assert!(v >= -a && v < a, "out of bound: {v} not in [-{a}, {a})");
        }
    }

    #[test]
    fn xavier_normal_mean_close_to_zero() {
        let t = xavier_normal(vec![100, 100], 42, DType::F32).unwrap();
        let v = read_f32(&t);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        // Expected std ≈ √(2/200) ≈ 0.1; tolerance ~0.02.
        assert!(mean.abs() < 0.02, "mean {mean} too far from 0");
    }

    #[test]
    fn kaiming_uniform_within_bound() {
        let fan_in = 100;
        let fan_out = 200;
        let a = (6.0_f32 / fan_in as f32).sqrt();
        let t = kaiming_uniform(vec![fan_in, fan_out], 42, DType::F32).unwrap();
        for v in read_f32(&t) {
            assert!(v >= -a && v < a, "out of bound: {v} not in [-{a}, {a})");
        }
    }

    #[test]
    fn kaiming_normal_has_expected_variance() {
        let fan_in = 256;
        let t = kaiming_normal(vec![fan_in, 64], 42, DType::F32).unwrap();
        let v = read_f32(&t);
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let var: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
        let expected = 2.0_f32 / fan_in as f32;
        // 30% tolerance — generous given the small sample.
        assert!(
            (var - expected).abs() / expected < 0.30,
            "var {var} not close to {expected}"
        );
    }

    #[test]
    fn rank_mismatch_errors() {
        let e = xavier_uniform(vec![10], 1, DType::F32).unwrap_err();
        assert!(e.to_string().contains("rank-2"));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let a = xavier_uniform(vec![3, 3], 7, DType::F32).unwrap();
        let b = xavier_uniform(vec![3, 3], 7, DType::F32).unwrap();
        assert_eq!(read_f32(&a), read_f32(&b));
    }
}
