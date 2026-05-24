//! Elementwise comparison ops: `eq`, `ne`, `lt`, `le`, `gt`, `ge`.
//!
//! Both inputs F32/BF16/F16, same shape. Output is a U8 mask where
//! `1` indicates the comparison held and `0` indicates it did not.
//! These compose naturally with [`where_select`] and `masked_fill`.
//!
//! Non-differentiable (boolean output).
//!
//! # Dispatch
//!
//! - CPU path: per-element F32-promoted comparison.
//! - CUDA path: `cuda_compare` (kernel `csrc/compare.cu`) when both
//!   inputs are CUDA-resident, contiguous, and F32/BF16/F16. (#1082)

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpKind {
    pub const fn name(self) -> &'static str {
        match self {
            CmpKind::Eq => "eq",
            CmpKind::Ne => "ne",
            CmpKind::Lt => "lt",
            CmpKind::Le => "le",
            CmpKind::Gt => "gt",
            CmpKind::Ge => "ge",
        }
    }
    /// Integer tag matching `csrc/compare.cu` `KIND_*` macros.
    pub const fn as_i32(self) -> i32 {
        match self {
            CmpKind::Eq => 0,
            CmpKind::Ne => 1,
            CmpKind::Lt => 2,
            CmpKind::Le => 3,
            CmpKind::Gt => 4,
            CmpKind::Ge => 5,
        }
    }
    fn apply_f32(self, a: f32, b: f32) -> bool {
        match self {
            CmpKind::Eq => a == b,
            CmpKind::Ne => a != b,
            CmpKind::Lt => a < b,
            CmpKind::Le => a <= b,
            CmpKind::Gt => a > b,
            CmpKind::Ge => a >= b,
        }
    }
}

fn apply(kind: CmpKind, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        bail!(
            "{}: shape mismatch: {:?} vs {:?}",
            kind.name(),
            a.shape(),
            b.shape()
        );
    }
    if a.dtype() != b.dtype() {
        bail!(
            "{}: dtype mismatch: {} vs {}",
            kind.name(),
            a.dtype(),
            b.dtype()
        );
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "{}: dtype must be F32/BF16/F16, got {}",
            kind.name(),
            a.dtype()
        );
    }

    // CUDA fast path: both inputs on the same CUDA device, contiguous,
    // F32/BF16/F16. Routes through the kernel in `csrc/compare.cu` and
    // returns a U8 mask. (#1082)
    #[cfg(feature = "cuda")]
    {
        if matches!(a.device(), crate::Device::Cuda(_))
            && matches!(b.device(), crate::Device::Cuda(_))
            && a.device() == b.device()
            && a.is_contiguous()
            && b.is_contiguous()
        {
            return crate::cuda_compare(a, b, kind.as_i32());
        }
    }

    if !a.is_contiguous() || !b.is_contiguous() {
        bail!("{}: inputs must be contiguous", kind.name());
    }
    let dtype = a.dtype();
    let per = dtype.size_in_bytes();
    let n = a.element_count();
    let a_cpu = a
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("compare: a storage must be CpuStorage"))?;
    let b_cpu = b
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("compare: b storage must be CpuStorage"))?;
    let a_bytes = a_cpu.as_bytes();
    let b_bytes = b_cpu.as_bytes();
    let mut out = vec![0u8; n];
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
        let _ = per;
        out[i] = if kind.apply_f32(av, bv) { 1 } else { 0 };
    }
    let cpu = CpuStorage::from_bytes(DType::U8, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(a.shape().to_vec()), TensorId::next())
}

pub fn eq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(CmpKind::Eq, a, b)
}
pub fn ne(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(CmpKind::Ne, a, b)
}
pub fn lt(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(CmpKind::Lt, a, b)
}
pub fn le(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(CmpKind::Le, a, b)
}
pub fn gt(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(CmpKind::Gt, a, b)
}
pub fn ge(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    apply(CmpKind::Ge, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u8(t: &Tensor) -> Vec<u8> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes().to_vec()
    }

    #[test]
    fn eq_and_ne() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 7.0], vec![4]).unwrap();
        assert_eq!(read_u8(&eq(&a, &b).unwrap()), vec![1, 0, 1, 0]);
        assert_eq!(read_u8(&ne(&a, &b).unwrap()), vec![0, 1, 0, 1]);
    }

    #[test]
    fn lt_le_gt_ge() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 2.0, 2.0], vec![3]).unwrap();
        assert_eq!(read_u8(&lt(&a, &b).unwrap()), vec![1, 0, 0]);
        assert_eq!(read_u8(&le(&a, &b).unwrap()), vec![1, 1, 0]);
        assert_eq!(read_u8(&gt(&a, &b).unwrap()), vec![0, 0, 1]);
        assert_eq!(read_u8(&ge(&a, &b).unwrap()), vec![0, 1, 1]);
    }

    #[test]
    fn output_dtype_is_u8() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let y = eq(&a, &b).unwrap();
        assert_eq!(y.dtype(), DType::U8);
    }

    #[test]
    fn shape_preserved() {
        let a = Tensor::from_slice(&[1.0f32; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32; 6], vec![2, 3]).unwrap();
        assert_eq!(eq(&a, &b).unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bf = vec![half::bf16::from_f32(1.0)];
        let b = Tensor::from_slice(&bf, vec![1]).unwrap();
        let e = eq(&a, &b).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = eq(&a, &b).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn kind_names() {
        assert_eq!(CmpKind::Eq.name(), "eq");
        assert_eq!(CmpKind::Ne.name(), "ne");
        assert_eq!(CmpKind::Lt.name(), "lt");
        assert_eq!(CmpKind::Le.name(), "le");
        assert_eq!(CmpKind::Gt.name(), "gt");
        assert_eq!(CmpKind::Ge.name(), "ge");
    }
}
