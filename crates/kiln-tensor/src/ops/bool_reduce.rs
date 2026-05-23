//! `all` / `any` — boolean reduction along an axis (or full reduction).
//!
//! Takes a U8 mask (typical output of `eq` / `lt` / etc.) and
//! reduces it along an axis (or all axes). Output is U8.
//!
//! - `all(mask, axis)` returns 1 iff every element along `axis` is
//!   nonzero.
//! - `any(mask, axis)` returns 1 iff at least one element is nonzero.
//!
//! Same shape rules as `sum_axis` (axis removed); full-reduction
//! variant produces a rank-0 scalar.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolReduce {
    All,
    Any,
}

impl BoolReduce {
    pub const fn name(self) -> &'static str {
        match self {
            BoolReduce::All => "all",
            BoolReduce::Any => "any",
        }
    }
}

fn apply_axis(kind: BoolReduce, mask: &Tensor, axis: usize) -> Result<Tensor> {
    if mask.dtype() != DType::U8 {
        bail!(
            "{}: mask dtype must be U8, got {}",
            kind.name(),
            mask.dtype()
        );
    }
    if axis >= mask.rank() {
        bail!(
            "{}: axis {axis} out of range for rank-{}",
            kind.name(),
            mask.rank()
        );
    }
    if !mask.is_contiguous() {
        bail!("{}: mask must be contiguous", kind.name());
    }
    let shape = mask.shape().to_vec();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let axis_dim = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let cpu = mask
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("bool_reduce: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let mut out = vec![0u8; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = match kind {
                BoolReduce::All => true,
                BoolReduce::Any => false,
            };
            for a in 0..axis_dim {
                let idx = (o * axis_dim + a) * inner + i;
                let b = bytes[idx] != 0;
                acc = match kind {
                    BoolReduce::All => acc && b,
                    BoolReduce::Any => acc || b,
                };
            }
            out[o * inner + i] = if acc { 1 } else { 0 };
        }
    }
    let mut out_shape = shape;
    out_shape.remove(axis);
    let cpu_out = CpuStorage::from_bytes(DType::U8, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
}

fn apply_all_axes(kind: BoolReduce, mask: &Tensor) -> Result<Tensor> {
    if mask.dtype() != DType::U8 {
        bail!(
            "{}: mask dtype must be U8, got {}",
            kind.name(),
            mask.dtype()
        );
    }
    if !mask.is_contiguous() {
        bail!("{}: mask must be contiguous", kind.name());
    }
    let cpu = mask
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("bool_reduce: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let result = match kind {
        BoolReduce::All => bytes.iter().all(|&b| b != 0),
        BoolReduce::Any => bytes.iter().any(|&b| b != 0),
    };
    let cpu_out = CpuStorage::from_bytes(DType::U8, vec![if result { 1u8 } else { 0u8 }])?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(Vec::<usize>::new()), TensorId::next())
}

pub fn all_axis(mask: &Tensor, axis: usize) -> Result<Tensor> {
    apply_axis(BoolReduce::All, mask, axis)
}
pub fn any_axis(mask: &Tensor, axis: usize) -> Result<Tensor> {
    apply_axis(BoolReduce::Any, mask, axis)
}
pub fn all_reduce(mask: &Tensor) -> Result<Tensor> {
    apply_all_axes(BoolReduce::All, mask)
}
pub fn any_reduce(mask: &Tensor) -> Result<Tensor> {
    apply_all_axes(BoolReduce::Any, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u8(t: &Tensor) -> Vec<u8> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes().to_vec()
    }

    #[test]
    fn all_axis_simple() {
        // [[1, 1, 1], [1, 0, 1]] all axis 1 → [1, 0]
        let m = Tensor::from_slice(&[1u8, 1, 1, 1, 0, 1], vec![2, 3]).unwrap();
        assert_eq!(read_u8(&all_axis(&m, 1).unwrap()), vec![1, 0]);
    }

    #[test]
    fn any_axis_simple() {
        // [[1, 0, 0], [0, 0, 0]] any axis 1 → [1, 0]
        let m = Tensor::from_slice(&[1u8, 0, 0, 0, 0, 0], vec![2, 3]).unwrap();
        assert_eq!(read_u8(&any_axis(&m, 1).unwrap()), vec![1, 0]);
    }

    #[test]
    fn all_axis_0() {
        // [[1, 1, 0], [1, 1, 1]] all axis 0 → [1, 1, 0]
        let m = Tensor::from_slice(&[1u8, 1, 0, 1, 1, 1], vec![2, 3]).unwrap();
        assert_eq!(read_u8(&all_axis(&m, 0).unwrap()), vec![1, 1, 0]);
    }

    #[test]
    fn all_reduce_scalar() {
        let m = Tensor::from_slice(&[1u8, 1, 1, 1], vec![4]).unwrap();
        let y = all_reduce(&m).unwrap();
        assert_eq!(y.shape(), &[] as &[usize]);
        assert_eq!(read_u8(&y), vec![1]);
        let m = Tensor::from_slice(&[1u8, 0, 1, 1], vec![4]).unwrap();
        assert_eq!(read_u8(&all_reduce(&m).unwrap()), vec![0]);
    }

    #[test]
    fn any_reduce_scalar() {
        let m = Tensor::from_slice(&[0u8; 4], vec![4]).unwrap();
        assert_eq!(read_u8(&any_reduce(&m).unwrap()), vec![0]);
        let m = Tensor::from_slice(&[0u8, 0, 1, 0], vec![4]).unwrap();
        assert_eq!(read_u8(&any_reduce(&m).unwrap()), vec![1]);
    }

    #[test]
    fn mask_dtype_errors() {
        let m = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = all_axis(&m, 0).unwrap_err();
        assert!(e.to_string().contains("U8"));
    }

    #[test]
    fn kind_names() {
        assert_eq!(BoolReduce::All.name(), "all");
        assert_eq!(BoolReduce::Any.name(), "any");
    }
}
