//! Statistical reductions: `variance`, `std`, `mean_variance`.
//!
//! Returns full-tensor scalars. For per-axis variants compose with
//! `mean_axis` + elementwise sub + pow + reduce.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("stats: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    if !t.is_contiguous() {
        bail!("stats: input must be contiguous");
    }
    let bytes = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("stats: storage must be CpuStorage"))?
        .as_bytes();
    let n = t.element_count();
    let dtype = t.dtype();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match dtype {
            DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            DType::F16 => half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                .to_f32(),
            _ => unreachable!(),
        });
    }
    Ok(out)
}

fn scalar_tensor(dtype: DType, v: f32) -> Result<Tensor> {
    let bytes = match dtype {
        DType::F32 => v.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

/// Population variance: `Σ (x - μ)² / N`.
pub fn variance(t: &Tensor) -> Result<Tensor> {
    let v = load_f32(t)?;
    if v.is_empty() {
        bail!("variance: empty input");
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    scalar_tensor(t.dtype(), var)
}

/// Population standard deviation: `√variance`.
pub fn std_dev(t: &Tensor) -> Result<Tensor> {
    let v = load_f32(t)?;
    if v.is_empty() {
        bail!("std_dev: empty input");
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    scalar_tensor(t.dtype(), var.sqrt())
}

/// `(mean, variance)` of the tensor.
pub fn mean_variance(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let v = load_f32(t)?;
    if v.is_empty() {
        bail!("mean_variance: empty input");
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    Ok((scalar_tensor(t.dtype(), mean)?, scalar_tensor(t.dtype(), var)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn variance_known_values() {
        // [1, 2, 3, 4, 5]; mean=3; var = 2 (population variance).
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        assert!((scalar_f32(&variance(&t).unwrap()) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn std_is_sqrt_variance() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        assert!((scalar_f32(&std_dev(&t).unwrap()) - 2.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn variance_constant_is_zero() {
        let t = Tensor::from_slice(&[5.0f32; 10], vec![10]).unwrap();
        assert!(scalar_f32(&variance(&t).unwrap()).abs() < 1e-5);
    }

    #[test]
    fn mean_variance_pair_matches_separate() {
        let t = Tensor::from_slice(&[1.0f32, 4.0, 9.0, 16.0], vec![4]).unwrap();
        let (m, v) = mean_variance(&t).unwrap();
        let m_v = scalar_f32(&m);
        let v_v = scalar_f32(&v);
        assert!((m_v - 7.5).abs() < 1e-5);
        // var = mean((x - 7.5)²) = mean(42.25 + 12.25 + 2.25 + 72.25) = 32.25
        assert!((v_v - 32.25).abs() < 1e-3);
    }

    #[test]
    fn empty_input_errors() {
        let t = Tensor::from_slice::<f32>(&[], vec![0]).unwrap();
        let e = variance(&t).unwrap_err();
        assert!(e.to_string().contains("empty"));
    }
}
