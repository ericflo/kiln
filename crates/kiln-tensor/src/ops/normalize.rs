//! `normalize` — L_p normalize along the trailing axis.
//!
//! ```text
//! out[..., :] = x[..., :] / max(‖x[..., :]‖_p, eps)
//! ```
//!
//! Different from `l2_norm` (which is hard-coded L2 + no eps clamp).
//! `normalize(x, p, eps)` is the general PyTorch-style operator.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn normalize(x: &Tensor, p: f32, eps: f32) -> Result<Tensor> {
    if p <= 0.0 {
        bail!("normalize: p must be > 0, got {p}");
    }
    if eps < 0.0 {
        bail!("normalize: eps must be ≥ 0, got {eps}");
    }
    if x.rank() == 0 {
        bail!("normalize: input must have rank ≥ 1");
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("normalize: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("normalize: input must be contiguous");
    }
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let last = *shape.last().unwrap();
    let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let per = dtype.size_in_bytes();
    let bytes = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("normalize: storage must be CpuStorage"))?
        .as_bytes();
    let mut out = vec![0u8; outer * last * per];
    for r in 0..outer {
        let mut row = Vec::with_capacity(last);
        for i in 0..last {
            let idx = r * last + i;
            row.push(match dtype {
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
            });
        }
        let pn: f32 = row.iter().map(|v| v.abs().powf(p)).sum::<f32>().powf(1.0 / p).max(eps);
        for i in 0..last {
            let y = row[i] / pn;
            let idx = r * last + i;
            match dtype {
                DType::F32 => out[idx * 4..idx * 4 + 4].copy_from_slice(&y.to_le_bytes()),
                DType::BF16 => out[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
                DType::F16 => out[idx * 2..idx * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
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
    fn l2_normalize_unit_norm() {
        // [3, 4] L2-norm = 5 → [0.6, 0.8].
        let x = Tensor::from_slice(&[3.0f32, 4.0], vec![1, 2]).unwrap();
        let y = normalize(&x, 2.0, 1e-8).unwrap();
        let v = read_f32(&y);
        assert!((v[0] - 0.6).abs() < 1e-5);
        assert!((v[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn l1_normalize_sums_to_one() {
        // [1, 2, 3] L1 sum = 6 → [1/6, 1/3, 1/2].
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = normalize(&x, 1.0, 1e-8).unwrap();
        let v = read_f32(&y);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalize_zero_vector_eps_clamp() {
        // All-zero vector with eps clamp returns zeros (not NaN).
        let x = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let y = normalize(&x, 2.0, 1e-8).unwrap();
        for v in read_f32(&y) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn normalize_p_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let e = normalize(&x, 0.0, 1e-8).unwrap_err();
        assert!(e.to_string().contains("p must be"));
    }
}
