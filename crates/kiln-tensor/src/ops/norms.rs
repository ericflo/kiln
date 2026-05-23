//! Norm utilities: `frobenius_norm`, `vector_norm`, `mean_squared`.
//!
//! Scalar reductions over a whole tensor. Used by gradient clipping,
//! diagnostics, regularizers.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn load_to_f32(t: &Tensor) -> Result<Vec<f32>> {
    let bytes = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("norms: storage must be CpuStorage"))?
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
            other => bail!("norms: unsupported dtype {other}"),
        });
    }
    Ok(out)
}

fn scalar(dtype: DType, v: f32) -> Result<Tensor> {
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

/// Frobenius norm: `‖t‖_F = √(Σ t_i²)`. Scalar output.
pub fn frobenius_norm(t: &Tensor) -> Result<Tensor> {
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("frobenius_norm: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    let v = load_to_f32(t)?;
    let sq_sum: f32 = v.iter().map(|&x| x * x).sum();
    scalar(t.dtype(), sq_sum.sqrt())
}

/// L_p vector norm: `‖t‖_p = (Σ |t_i|^p)^(1/p)`. `p` must be > 0.
pub fn vector_norm(t: &Tensor, p: f32) -> Result<Tensor> {
    if p <= 0.0 {
        bail!("vector_norm: p must be > 0, got {p}");
    }
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("vector_norm: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    let v = load_to_f32(t)?;
    let s: f32 = v.iter().map(|&x| x.abs().powf(p)).sum();
    scalar(t.dtype(), s.powf(1.0 / p))
}

/// Mean of squares: `Σ t_i² / N`. Scalar output.
pub fn mean_squared(t: &Tensor) -> Result<Tensor> {
    if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("mean_squared: dtype must be F32/BF16/F16, got {}", t.dtype());
    }
    let v = load_to_f32(t)?;
    let sq_sum: f32 = v.iter().map(|&x| x * x).sum();
    scalar(t.dtype(), sq_sum / v.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn frobenius_norm_known() {
        // ‖[3, 4]‖_F = √(9+16) = 5
        let t = Tensor::from_slice(&[3.0f32, 4.0], vec![2]).unwrap();
        assert!((scalar_f32(&frobenius_norm(&t).unwrap()) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn frobenius_norm_rank2() {
        // ‖[[1, 2], [2, 1]]‖_F = √(1+4+4+1) = √10
        let t = Tensor::from_slice(&[1.0f32, 2.0, 2.0, 1.0], vec![2, 2]).unwrap();
        let f = scalar_f32(&frobenius_norm(&t).unwrap());
        assert!((f - 10.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn vector_norm_l1() {
        // ‖[1, -2, 3]‖_1 = 6
        let t = Tensor::from_slice(&[1.0f32, -2.0, 3.0], vec![3]).unwrap();
        assert!((scalar_f32(&vector_norm(&t, 1.0).unwrap()) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn vector_norm_l2_matches_frobenius() {
        let t = Tensor::from_slice(&[3.0f32, 4.0], vec![2]).unwrap();
        let l2 = scalar_f32(&vector_norm(&t, 2.0).unwrap());
        let fro = scalar_f32(&frobenius_norm(&t).unwrap());
        assert!((l2 - fro).abs() < 1e-6);
    }

    #[test]
    fn vector_norm_p_zero_errors() {
        let t = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = vector_norm(&t, 0.0).unwrap_err();
        assert!(e.to_string().contains("p must be"));
    }

    #[test]
    fn mean_squared_known() {
        // mean(1+4+9) = 14/3
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let m = scalar_f32(&mean_squared(&t).unwrap());
        assert!((m - 14.0 / 3.0).abs() < 1e-6);
    }
}
