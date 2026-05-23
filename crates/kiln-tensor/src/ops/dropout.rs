//! `dropout` — inverted dropout op for training.
//!
//! Forward (with drop probability `p`):
//!
//! ```text
//! mask_i = Bernoulli(1 - p)      # 1 with prob (1-p), else 0
//! y_i    = x_i * mask_i / (1 - p)
//! ```
//!
//! "Inverted" because the surviving elements are scaled by `1/(1-p)`
//! at training time so that **eval-time forward is a pure pass-through**
//! (no /(1-p) scale needed at inference). This is the convention every
//! modern framework (PyTorch, JAX, candle) uses.
//!
//! # API
//!
//! The forward returns `(y, mask)` — the caller stashes `mask` on the
//! `DropoutBackward` for the backward pass:
//!
//! ```ignore
//! let (y, mask) = dropout(&x, 0.1, /*seed=*/ 42)?;
//! tape.record(&y, &[&x], Box::new(DropoutBackward { mask, p: 0.1 }));
//! ```
//!
//! # Determinism
//!
//! `Constructive` given the seed. Tests use fixed seeds; production
//! callers reseed per training step from the global RNG.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Apply inverted dropout. Returns `(y, mask)` where `mask` is a U8
/// tensor of the same shape with `0` at dropped positions and `1` at
/// surviving positions.
pub fn dropout(x: &Tensor, p: f32, seed: u64) -> Result<(Tensor, Tensor)> {
    if !(0.0..1.0).contains(&p) {
        bail!("dropout: p must be in [0, 1), got {p}");
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("dropout: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("dropout: input must be contiguous");
    }
    let dtype = x.dtype();
    let n = x.element_count();
    let per = dtype.size_in_bytes();

    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("dropout: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();

    let mut state = seed.max(1);
    let mut mask = vec![0u8; n];
    let mut out_bytes = vec![0u8; n * per];
    let inv_keep = if p == 0.0 { 1.0 } else { 1.0 / (1.0 - p) };

    for i in 0..n {
        let r = next_u01(&mut state);
        let keep = r >= p;
        mask[i] = if keep { 1 } else { 0 };
        let x_v = match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        };
        let y_v = if keep { x_v * inv_keep } else { 0.0 };
        match dtype {
            DType::F32 => out_bytes[i * 4..i * 4 + 4].copy_from_slice(&y_v.to_le_bytes()),
            DType::BF16 => out_bytes[i * 2..i * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y_v).to_le_bytes()),
            DType::F16 => out_bytes[i * 2..i * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y_v).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let y_cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
    let y_storage: Storage = Arc::new(y_cpu);
    let y = Tensor::from_parts(y_storage, Layout::contiguous(x.shape().to_vec()), TensorId::next())?;

    let mask_cpu = CpuStorage::from_bytes(DType::U8, mask)?;
    let mask_storage: Storage = Arc::new(mask_cpu);
    let mask_t = Tensor::from_parts(
        mask_storage,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )?;
    Ok((y, mask_t))
}

fn next_u01(state: &mut u64) -> f32 {
    // splitmix64 step → top 24 bits → uniform [0, 1).
    *state = state.wrapping_add(0x9E3779B97F4A7C15u64);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9u64);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EBu64);
    let bits = ((z ^ (z >> 31)) >> 40) as u32;
    bits as f32 / (1u32 << 24) as f32
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

    fn read_u8(t: &Tensor) -> Vec<u8> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes().to_vec()
    }

    #[test]
    fn dropout_p_zero_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let (y, mask) = dropout(&x, 0.0, 1).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
        // p=0 → every position survives.
        assert_eq!(read_u8(&mask), vec![1u8, 1, 1, 1]);
    }

    #[test]
    fn dropout_p_near_one_drops_most() {
        // p = 0.99 → most positions zero.
        let n = 200;
        let x_data = vec![1.0f32; n];
        let x = Tensor::from_slice(&x_data, vec![n]).unwrap();
        let (y, mask) = dropout(&x, 0.99, 1).unwrap();
        let drop_count = read_u8(&mask).iter().filter(|&&v| v == 0).count();
        // 99% expected; allow significant variance.
        assert!(
            drop_count > n * 90 / 100,
            "expected most dropped, got {drop_count}/{n}"
        );
        // Surviving entries are scaled up by 1/(1-p) = 100.
        let y_vals = read_f32(&y);
        for (i, &m) in read_u8(&mask).iter().enumerate() {
            if m == 1 {
                assert!(
                    (y_vals[i] - 100.0).abs() < 1e-3,
                    "scale wrong at i={i}: y={}",
                    y_vals[i]
                );
            } else {
                assert_eq!(y_vals[i], 0.0);
            }
        }
    }

    #[test]
    fn dropout_deterministic_with_same_seed() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let (y1, m1) = dropout(&x, 0.5, 42).unwrap();
        let (y2, m2) = dropout(&x, 0.5, 42).unwrap();
        assert_eq!(read_f32(&y1), read_f32(&y2));
        assert_eq!(read_u8(&m1), read_u8(&m2));
    }

    #[test]
    fn dropout_different_seeds_diverge() {
        let n = 100;
        let x = Tensor::from_slice(&vec![1.0f32; n], vec![n]).unwrap();
        let (_, m1) = dropout(&x, 0.5, 1).unwrap();
        let (_, m2) = dropout(&x, 0.5, 100).unwrap();
        let same = read_u8(&m1) == read_u8(&m2);
        assert!(!same, "two different seeds produced identical masks");
    }

    #[test]
    fn dropout_mask_dtype_is_u8() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let (_, mask) = dropout(&x, 0.3, 1).unwrap();
        assert_eq!(mask.dtype(), DType::U8);
        assert_eq!(mask.shape(), &[2]);
    }

    #[test]
    fn dropout_bf16_round_trips() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![3]).unwrap();
        let (y, _) = dropout(&x, 0.0, 1).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn dropout_p_out_of_range_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = dropout(&x, 1.0, 1).unwrap_err();
        assert!(e.to_string().contains("p must be"));
        let e = dropout(&x, -0.1, 1).unwrap_err();
        assert!(e.to_string().contains("p must be"));
    }

    #[test]
    fn dropout_2d_shape_preserved() {
        let x = Tensor::from_slice(&[1.0f32; 6], vec![2, 3]).unwrap();
        let (y, m) = dropout(&x, 0.5, 1).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(m.shape(), &[2, 3]);
    }
}
