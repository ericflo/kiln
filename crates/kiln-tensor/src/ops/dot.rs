//! `dot` — 1D dot product (inner product) of two same-length vectors.
//!
//! ```text
//! out = Σᵢ a[i] * b[i]
//! ```
//!
//! Returns a rank-0 scalar. Composable from `mul + sum_all` but
//! provided as a primitive for ergonomics and downstream backend
//! fusion (the GPU port can fuse the elementwise mul with the
//! reduction).

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn dot(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.rank() != 1 || b.rank() != 1 {
        bail!(
            "dot: both inputs must be rank-1, got a={:?}, b={:?}",
            a.shape(),
            b.shape()
        );
    }
    if a.shape() != b.shape() {
        bail!(
            "dot: shape mismatch: a={:?}, b={:?}",
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!("dot: dtype mismatch");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("dot: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("dot: inputs must be contiguous");
    }
    let dtype = a.dtype();
    let n = a.element_count();
    let a_bytes = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("dot: a storage must be CpuStorage"))?
        .as_bytes();
    let b_bytes = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("dot: b storage must be CpuStorage"))?
        .as_bytes();
    let mut acc = 0.0f32;
    for i in 0..n {
        let av = match dtype {
            DType::F32 => f32::from_le_bytes(a_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(a_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(a_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        };
        let bv = match dtype {
            DType::F32 => f32::from_le_bytes(b_bytes[i * 4..i * 4 + 4].try_into().unwrap()),
            DType::BF16 => {
                half::bf16::from_le_bytes(b_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            DType::F16 => {
                half::f16::from_le_bytes(b_bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            }
            _ => unreachable!(),
        };
        acc += av * bv;
    }
    let bytes = match dtype {
        DType::F32 => acc.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(acc).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(acc).to_le_bytes().to_vec(),
        _ => unreachable!(),
    };
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_f32(t: &Tensor) -> f32 {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
    }

    #[test]
    fn dot_simple() {
        // [1, 2, 3] · [4, 5, 6] = 4 + 10 + 18 = 32.
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], vec![3]).unwrap();
        let y = dot(&a, &b).unwrap();
        assert_eq!(y.shape(), &[] as &[usize]);
        assert_eq!(scalar_f32(&y), 32.0);
    }

    #[test]
    fn dot_orthogonal_is_zero() {
        let a = Tensor::from_slice(&[1.0f32, 0.0, 0.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[0.0f32, 1.0, 0.0], vec![3]).unwrap();
        assert_eq!(scalar_f32(&dot(&a, &b).unwrap()), 0.0);
    }

    #[test]
    fn dot_squared_norm() {
        // dot(x, x) = |x|².
        let x = Tensor::from_slice(&[3.0f32, 4.0], vec![2]).unwrap();
        assert_eq!(scalar_f32(&dot(&x, &x).unwrap()), 25.0);
    }

    #[test]
    fn dot_shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = dot(&a, &b).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn dot_rank_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let e = dot(&a, &b).unwrap_err();
        assert!(e.to_string().contains("rank-1"));
    }

    #[test]
    fn dot_bf16_path() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let a = Tensor::from_slice(&bf, vec![2]).unwrap();
        let b = Tensor::from_slice(&bf, vec![2]).unwrap();
        let y = dot(&a, &b).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }
}
