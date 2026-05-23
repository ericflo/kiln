//! `minimum` / `maximum` — elementwise binary min/max of two
//! same-shape tensors. (PyTorch / numpy semantics.)
//!
//! ```text
//! minimum(a, b)[i] = min(a[i], b[i])
//! maximum(a, b)[i] = max(a[i], b[i])
//! ```
//!
//! Different from `max_axis`/`min_axis` which reduce along an axis.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

fn apply(f: impl Fn(f32, f32) -> f32, a: &Tensor, b: &Tensor, name: &str) -> Result<Tensor> {
    if a.shape() != b.shape() {
        bail!("{name}: shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    }
    if a.dtype() != b.dtype() {
        bail!("{name}: dtype mismatch");
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("{name}: dtype must be F32/BF16/F16, got {}", a.dtype());
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{name}: inputs must be contiguous");
    }
    let dtype = a.dtype();
    let per = dtype.size_in_bytes();
    let n = a.element_count();
    let a_bytes = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("binary_minmax: a storage must be CpuStorage"))?
        .as_bytes();
    let b_bytes = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("binary_minmax: b storage must be CpuStorage"))?
        .as_bytes();
    let mut out = vec![0u8; n * per];
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
        let y = f(av, bv);
        match dtype {
            DType::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&y.to_le_bytes()),
            DType::BF16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::bf16::from_f32(y).to_le_bytes()),
            DType::F16 => out[i * 2..i * 2 + 2]
                .copy_from_slice(&half::f16::from_f32(y).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(a.shape().to_vec()), TensorId::next())
}

pub fn minimum(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(|x, y| x.min(y), a, b, "minimum")
}

pub fn maximum(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(|x, y| x.max(y), a, b, "maximum")
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
    fn minimum_picks_smaller() {
        let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 7.0], vec![4]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 0.5], vec![4]).unwrap();
        let y = read_f32(&minimum(&a, &b).unwrap());
        assert_eq!(y, vec![1.0, 4.0, 3.0, 0.5]);
    }

    #[test]
    fn maximum_picks_larger() {
        let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 7.0], vec![4]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0, 6.0, 0.5], vec![4]).unwrap();
        let y = read_f32(&maximum(&a, &b).unwrap());
        assert_eq!(y, vec![2.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn equal_values_propagate_either() {
        // min(3, 3) == max(3, 3) == 3 — sanity.
        let a = Tensor::from_slice(&[3.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[3.0f32], vec![1]).unwrap();
        assert_eq!(read_f32(&minimum(&a, &b).unwrap()), vec![3.0]);
        assert_eq!(read_f32(&maximum(&a, &b).unwrap()), vec![3.0]);
    }

    #[test]
    fn shape_preserved() {
        let a = Tensor::from_slice(&[1.0f32; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(&[2.0f32; 6], vec![2, 3]).unwrap();
        assert_eq!(minimum(&a, &b).unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = minimum(&a, &b).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bf = vec![half::bf16::from_f32(1.0)];
        let b = Tensor::from_slice(&bf, vec![1]).unwrap();
        let e = maximum(&a, &b).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }
}
