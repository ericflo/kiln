//! Backward for the four reduction ops:
//! `sum_all`, `mean_all`, `sum_axis`, `mean_axis`.
//!
//! Reduction backward "broadcasts" the upstream gradient back to the
//! input shape. kiln-tensor Phase 1.x has no broadcast op, so this
//! lives at byte level until Phase 2.x adds an `expand` op.
//!
//! # Math
//!
//! Let `D` = the reduced count (for `*_all`: `x.element_count()`;
//! for `*_axis`: `x.shape[axis]`).
//!
//! | Forward | Backward |
//! |---------|----------|
//! | `sum_all(x)` | `d_x[i] = grad_output` (scalar broadcast) |
//! | `mean_all(x)` | `d_x[i] = grad_output / D` |
//! | `sum_axis(x, a)` | `d_x[..., i, ...] = grad_output[...]` |
//! | `mean_axis(x, a)` | `d_x[..., i, ...] = grad_output[...] / D` |
//!
//! "Broadcast along axis" means: for each output element, replicate
//! its gradient `D` times along the reduced axis.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

/// Which kind of reduction the forward op did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
}

/// Scope of the reduction: full tensor → scalar, or one axis only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceScope {
    All,
    Axis(usize),
}

#[derive(Debug, Clone)]
pub struct ReduceBackward {
    /// Original input shape — backward broadcasts the gradient to
    /// this shape.
    pub input_shape: Vec<usize>,
    /// dtype of the forward input (and so of the output gradient).
    pub dtype: DType,
    pub kind: ReduceKind,
    pub scope: ReduceScope,
}

impl BackwardOp for ReduceBackward {
    fn name(&self) -> &'static str {
        match (self.kind, self.scope) {
            (ReduceKind::Sum, ReduceScope::All) => "sum_all_backward",
            (ReduceKind::Mean, ReduceScope::All) => "mean_all_backward",
            (ReduceKind::Sum, ReduceScope::Axis(_)) => "sum_axis_backward",
            (ReduceKind::Mean, ReduceScope::Axis(_)) => "mean_axis_backward",
        }
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if grad_output.dtype() != self.dtype {
            bail!(
                "ReduceBackward: grad_output dtype {} != saved input dtype {}",
                grad_output.dtype(),
                self.dtype
            );
        }
        match self.scope {
            ReduceScope::All => self.apply_all(grad_output),
            ReduceScope::Axis(a) => self.apply_axis(grad_output, a),
        }
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

impl ReduceBackward {
    fn apply_all(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if grad_output.rank() != 0 {
            bail!(
                "ReduceBackward (All): grad_output must be scalar (rank 0), got rank {}",
                grad_output.rank()
            );
        }
        let n_in: usize = self.input_shape.iter().product::<usize>().max(1);
        let g = read_scalar_f32(grad_output)?;
        let scaled = match self.kind {
            ReduceKind::Sum => g,
            ReduceKind::Mean => g / n_in as f32,
        };
        let dx_data = vec![scaled; n_in];
        let out = store_f32(self.dtype, &self.input_shape, &dx_data)?;
        Ok(vec![Some(out)])
    }

    fn apply_axis(&self, grad_output: &Tensor, axis: usize) -> Result<Vec<Option<Tensor>>> {
        if axis >= self.input_shape.len() {
            bail!(
                "ReduceBackward (Axis {axis}): out of range for input shape {:?}",
                self.input_shape
            );
        }
        // Expected grad_output shape: input_shape with axis removed.
        let expected: Vec<usize> = self
            .input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect();
        if grad_output.shape() != expected.as_slice() {
            bail!(
                "ReduceBackward (Axis {axis}): grad_output shape {:?} != expected {:?}",
                grad_output.shape(),
                expected
            );
        }

        let outer: usize = self.input_shape[..axis].iter().product::<usize>().max(1);
        let axis_dim = self.input_shape[axis];
        let inner: usize = self.input_shape[axis + 1..]
            .iter()
            .product::<usize>()
            .max(1);
        let g = load_f32(grad_output)?;
        let scale = match self.kind {
            ReduceKind::Sum => 1.0f32,
            ReduceKind::Mean => 1.0f32 / axis_dim as f32,
        };
        let mut dx = vec![0.0f32; outer * axis_dim * inner];
        for o in 0..outer {
            for a in 0..axis_dim {
                for i in 0..inner {
                    let g_idx = o * inner + i;
                    dx[(o * axis_dim + a) * inner + i] = g[g_idx] * scale;
                }
            }
        }
        let out = store_f32(self.dtype, &self.input_shape, &dx)?;
        Ok(vec![Some(out)])
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn read_scalar_f32(t: &Tensor) -> Result<f32> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("reduce_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    Ok(match t.dtype() {
        DType::F32 => f32::from_le_bytes(bytes[..4].try_into().unwrap()),
        DType::BF16 => half::bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32(),
        DType::F16 => half::f16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32(),
        d => return Err(Error::Msg(format!("reduce_backward: unsupported dtype {d}"))),
    })
}

fn load_f32(t: &Tensor) -> Result<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("reduce_backward: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
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
            d => return Err(Error::Msg(format!("reduce_backward: unsupported dtype {d}"))),
        });
    }
    Ok(out)
}

fn store_f32(dtype: DType, shape: &[usize], data: &[f32]) -> Result<Tensor> {
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; data.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in data.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
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
    fn sum_all_backward_broadcasts_scalar() {
        let bo = ReduceBackward {
            input_shape: vec![2, 3],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        };
        let g = Tensor::from_slice(&[2.5f32], vec![]).unwrap();
        let dx = bo.apply(&g).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(dx.shape(), &[2, 3]);
        assert_eq!(read_f32(&dx), vec![2.5; 6]);
    }

    #[test]
    fn mean_all_backward_scales_by_n() {
        let bo = ReduceBackward {
            input_shape: vec![2, 3],
            dtype: DType::F32,
            kind: ReduceKind::Mean,
            scope: ReduceScope::All,
        };
        // grad = 6.0; n = 6; → 1.0 in each position.
        let g = Tensor::from_slice(&[6.0f32], vec![]).unwrap();
        let dx = bo.apply(&g).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&dx), vec![1.0; 6]);
    }

    #[test]
    fn sum_axis_backward_broadcasts_along_axis() {
        // input [2, 3], reduce axis 1 → output [2].
        // grad = [10, 20]. d_x[r, c] = grad[r] for all c.
        // → [[10, 10, 10], [20, 20, 20]]
        let bo = ReduceBackward {
            input_shape: vec![2, 3],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::Axis(1),
        };
        let g = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let dx = bo.apply(&g).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(dx.shape(), &[2, 3]);
        assert_eq!(read_f32(&dx), vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    }

    #[test]
    fn mean_axis_backward_divides_by_axis_dim() {
        // input [2, 4], reduce axis 1 → output [2]. mean divides by 4.
        let bo = ReduceBackward {
            input_shape: vec![2, 4],
            dtype: DType::F32,
            kind: ReduceKind::Mean,
            scope: ReduceScope::Axis(1),
        };
        let g = Tensor::from_slice(&[8.0f32, 16.0], vec![2]).unwrap();
        let dx = bo.apply(&g).unwrap()[0].as_ref().unwrap().clone();
        // First row entries = 8/4 = 2.0, second row = 16/4 = 4.0.
        assert_eq!(
            read_f32(&dx),
            vec![2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]
        );
    }

    #[test]
    fn sum_axis_outer_axis() {
        // input [3, 2], reduce axis 0 → output [2]. grad = [10, 20].
        // d_x[r, c] = grad[c] for all r.
        // → [[10, 20], [10, 20], [10, 20]]
        let bo = ReduceBackward {
            input_shape: vec![3, 2],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::Axis(0),
        };
        let g = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let dx = bo.apply(&g).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&dx), vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn reduce_all_rejects_non_scalar_grad() {
        let bo = ReduceBackward {
            input_shape: vec![3],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        };
        let bad = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("rank 0"));
    }

    #[test]
    fn reduce_axis_rejects_wrong_grad_shape() {
        let bo = ReduceBackward {
            input_shape: vec![2, 3],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::Axis(1),
        };
        let bad = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("grad_output shape"));
    }

    #[test]
    fn op_metadata() {
        let bo = ReduceBackward {
            input_shape: vec![1],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        };
        assert_eq!(bo.name(), "sum_all_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));

        let bo = ReduceBackward {
            input_shape: vec![1],
            dtype: DType::F32,
            kind: ReduceKind::Mean,
            scope: ReduceScope::All,
        };
        assert_eq!(bo.name(), "mean_all_backward");

        let bo = ReduceBackward {
            input_shape: vec![1],
            dtype: DType::F32,
            kind: ReduceKind::Sum,
            scope: ReduceScope::Axis(0),
        };
        assert_eq!(bo.name(), "sum_axis_backward");

        let bo = ReduceBackward {
            input_shape: vec![1],
            dtype: DType::F32,
            kind: ReduceKind::Mean,
            scope: ReduceScope::Axis(0),
        };
        assert_eq!(bo.name(), "mean_axis_backward");
    }
}
