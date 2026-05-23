//! `lerp` — linear interpolation between two tensors.
//!
//! `lerp(a, b, weight) = a + weight * (b - a)`
//!
//! Element-wise, both tensors must share shape + dtype. Weight is a
//! scalar `f32`. PyTorch parity with `torch.lerp(a, b, weight)`. Used
//! by EMA weight averaging, Lion/Muon momentum updates, and DPO
//! reference-policy mixing.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

pub fn lerp(a: &Tensor, b: &Tensor, weight: f32) -> Result<Tensor> {
    if a.shape() != b.shape() {
        bail!(
            "lerp: shape mismatch — a {:?} vs b {:?}",
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!(
            "lerp: dtype mismatch — a {} vs b {}",
            a.dtype(),
            b.dtype()
        );
    }
    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        bail!("lerp: dtype must be F32/BF16/F16, got {dtype}");
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("lerp: inputs must be contiguous");
    }
    let n = a.element_count();
    let a_cpu = a.storage();
    let a_cpu = a_cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("lerp: a must be CpuStorage"))?;
    let b_cpu = b.storage();
    let b_cpu = b_cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("lerp: b must be CpuStorage"))?;
    let ab = a_cpu.as_bytes();
    let bb = b_cpu.as_bytes();
    let mut out = vec![0u8; n * dtype.size_in_bytes()];
    match dtype {
        DType::F32 => {
            for i in 0..n {
                let va = f32::from_le_bytes(ab[i * 4..i * 4 + 4].try_into().unwrap());
                let vb = f32::from_le_bytes(bb[i * 4..i * 4 + 4].try_into().unwrap());
                let r = va + weight * (vb - va);
                out[i * 4..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..n {
                let va =
                    half::bf16::from_le_bytes(ab[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                let vb =
                    half::bf16::from_le_bytes(bb[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32();
                let r = va + weight * (vb - va);
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(r).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..n {
                let va = half::f16::from_le_bytes(ab[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32();
                let vb = half::f16::from_le_bytes(bb[i * 2..i * 2 + 2].try_into().unwrap())
                    .to_f32();
                let r = va + weight * (vb - va);
                out[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(r).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(a.shape().to_vec()), TensorId::next())
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
    fn lerp_weight_zero_returns_a() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let y = lerp(&a, &b, 0.0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lerp_weight_one_returns_b() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let y = lerp(&a, &b, 1.0).unwrap();
        assert_eq!(read_f32(&y), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn lerp_half_is_midpoint() {
        let a = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0, 6.0], vec![3]).unwrap();
        let y = lerp(&a, &b, 0.5).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lerp_extrapolation_negative_weight() {
        // weight = -1 → a - (b - a) = 2a - b
        let a = Tensor::from_slice(&[3.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[5.0f32], vec![1]).unwrap();
        let y = lerp(&a, &b, -1.0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0]);
    }

    #[test]
    fn lerp_extrapolation_above_one() {
        // weight = 2 → a + 2(b - a) = 2b - a
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[3.0f32], vec![1]).unwrap();
        let y = lerp(&a, &b, 2.0).unwrap();
        assert_eq!(read_f32(&y), vec![5.0]);
    }

    #[test]
    fn lerp_shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = lerp(&a, &b, 0.5).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn lerp_dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[half::bf16::from_f32(2.0)], vec![1]).unwrap();
        let e = lerp(&a, &b, 0.5).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }

    #[test]
    fn lerp_bf16() {
        let a = Tensor::from_slice(
            &[half::bf16::from_f32(0.0), half::bf16::from_f32(0.0)],
            vec![2],
        )
        .unwrap();
        let b = Tensor::from_slice(
            &[half::bf16::from_f32(4.0), half::bf16::from_f32(8.0)],
            vec![2],
        )
        .unwrap();
        let y = lerp(&a, &b, 0.5).unwrap();
        let cpu = y.storage();
        let cpu = cpu.as_any().downcast_ref::<CpuStorage>().unwrap();
        let vals: Vec<f32> = cpu
            .as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert!((vals[0] - 2.0).abs() < 1e-3);
        assert!((vals[1] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn lerp_2d() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let y = lerp(&a, &b, 0.25).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(read_f32(&y), vec![2.0, 3.0, 4.0, 5.0]);
    }
}
