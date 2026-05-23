//! `outer` — outer product of two 1D vectors.
//!
//! ```text
//! out[i, j] = a[i] * b[j]
//! ```
//!
//! Shape `[M, N]` from `a: [M]` and `b: [N]`. Used in LoRA rank-1
//! reconstructions, attention rank-1 perturbations, and various
//! decomposition test fixtures.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn outer(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.rank() != 1 || b.rank() != 1 {
        bail!(
            "outer: both inputs must be rank-1, got a={:?}, b={:?}",
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!("outer: dtype mismatch");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("outer: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("outer: inputs must be contiguous");
    }
    let dtype = a.dtype();
    let per = dtype.size_in_bytes();
    let m = a.element_count();
    let n = b.element_count();
    let a_bytes = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("outer: a storage must be CpuStorage"))?
        .as_bytes();
    let b_bytes = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("outer: b storage must be CpuStorage"))?
        .as_bytes();
    let mut out = vec![0u8; m * n * per];
    for i in 0..m {
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
        for j in 0..n {
            let bv = match dtype {
                DType::F32 => f32::from_le_bytes(b_bytes[j * 4..j * 4 + 4].try_into().unwrap()),
                DType::BF16 => {
                    half::bf16::from_le_bytes(b_bytes[j * 2..j * 2 + 2].try_into().unwrap())
                        .to_f32()
                }
                DType::F16 => {
                    half::f16::from_le_bytes(b_bytes[j * 2..j * 2 + 2].try_into().unwrap())
                        .to_f32()
                }
                _ => unreachable!(),
            };
            let y = av * bv;
            let off = (i * n + j) * per;
            match dtype {
                DType::F32 => out[off..off + 4].copy_from_slice(&y.to_le_bytes()),
                DType::BF16 => out[off..off + 2]
                    .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
                DType::F16 => out[off..off + 2]
                    .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(vec![m, n]), TensorId::next())
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
    fn outer_2x3() {
        // a = [1, 2]; b = [3, 4, 5]
        // out = [[3, 4, 5], [6, 8, 10]]
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0f32, 4.0, 5.0], vec![3]).unwrap();
        let y = outer(&a, &b).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(read_f32(&y), vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn outer_identity_when_one_vec_is_one() {
        // a = ones[3]; outer = b broadcast across rows.
        let a = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[7.0f32, 8.0], vec![2]).unwrap();
        let y = outer(&a, &b).unwrap();
        assert_eq!(read_f32(&y), vec![7.0, 8.0, 7.0, 8.0, 7.0, 8.0]);
    }

    #[test]
    fn outer_lora_shape() {
        // Typical LoRA test: U [4] outer V [4] = rank-1 [4, 4] matrix.
        let u = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let v = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], vec![4]).unwrap();
        let m = outer(&u, &v).unwrap();
        assert_eq!(m.shape(), &[4, 4]);
        // Row r = u[r] * 0.5 * ones, so first row = [0.5, 0.5, 0.5, 0.5].
        let v = read_f32(&m);
        assert_eq!(v[0], 0.5);
        assert_eq!(v[4], 1.0); // u[1] * 0.5 = 1.0
        assert_eq!(v[8], 1.5);
        assert_eq!(v[12], 2.0);
    }

    #[test]
    fn outer_rank_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = outer(&a, &b).unwrap_err();
        assert!(e.to_string().contains("rank-1"));
    }

    #[test]
    fn outer_dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bf = vec![half::bf16::from_f32(1.0)];
        let b = Tensor::from_slice(&bf, vec![1]).unwrap();
        let e = outer(&a, &b).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }
}
