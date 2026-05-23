//! `ConcatBackward` — gradient of `concat(inputs, axis)`.
//!
//! Forward: `out[..., axis_offset_i .. axis_offset_i + size_i, ...] =
//! inputs[i]`. The gradient slices `d_out` along the concat axis back
//! into per-input contiguous gradients of the original shapes.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct ConcatBackward {
    /// Concat axis from the forward pass.
    pub axis: usize,
    /// `inputs[i].shape[axis]` from the forward pass — in order.
    /// Used to compute the slice offsets for each input's gradient.
    pub input_axis_sizes: Vec<usize>,
    /// Each input's shape (so we can reconstruct contiguous output
    /// gradient tensors).
    pub input_shapes: Vec<Vec<usize>>,
    /// dtype of the forward inputs (== dtype of the output gradient).
    pub dtype: DType,
}

impl BackwardOp for ConcatBackward {
    fn name(&self) -> &'static str {
        "concat_backward"
    }
    fn input_count(&self) -> usize {
        self.input_axis_sizes.len()
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let n = self.input_count();
        if n == 0 {
            bail!("ConcatBackward: zero inputs");
        }
        if self.input_shapes.len() != n {
            bail!(
                "ConcatBackward: input_shapes.len() {} != input_axis_sizes.len() {n}",
                self.input_shapes.len()
            );
        }
        let go_shape = grad_output.shape();
        let rank = go_shape.len();
        if self.axis >= rank {
            bail!(
                "ConcatBackward: axis {} out of range for grad shape {go_shape:?}",
                self.axis
            );
        }
        let dtype = grad_output.dtype();
        if dtype != self.dtype {
            bail!(
                "ConcatBackward: grad dtype {dtype} != saved dtype {}",
                self.dtype
            );
        }
        if dtype.is_packed() {
            bail!("ConcatBackward: packed dtype {dtype} not supported");
        }
        if !grad_output.is_contiguous() {
            bail!("ConcatBackward: grad must be contiguous");
        }
        let per = dtype.size_in_bytes();
        let axis_total: usize = self.input_axis_sizes.iter().sum();
        if go_shape[self.axis] != axis_total {
            bail!(
                "ConcatBackward: grad axis size {} != sum of input axis sizes {axis_total}",
                go_shape[self.axis]
            );
        }

        let outer: usize = go_shape[..self.axis].iter().product::<usize>().max(1);
        let inner: usize = go_shape[self.axis + 1..]
            .iter()
            .product::<usize>()
            .max(1);

        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("ConcatBackward: grad storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();

        let mut grads: Vec<Option<Tensor>> = Vec::with_capacity(n);
        let mut axis_offset = 0usize;
        for (i, &size_i) in self.input_axis_sizes.iter().enumerate() {
            let mut bytes = vec![0u8; outer * size_i * inner * per];
            for o in 0..outer {
                let src_start = (o * axis_total + axis_offset) * inner * per;
                let src_end = src_start + size_i * inner * per;
                let dst_start = o * size_i * inner * per;
                let dst_end = dst_start + size_i * inner * per;
                bytes[dst_start..dst_end].copy_from_slice(&go_bytes[src_start..src_end]);
            }
            let cpu = CpuStorage::from_bytes(dtype, bytes)?;
            let storage: Storage = Arc::new(cpu);
            let g = Tensor::from_parts(
                storage,
                Layout::contiguous(self.input_shapes[i].clone()),
                TensorId::next(),
            )?;
            grads.push(Some(g));
            axis_offset += size_i;
        }
        Ok(grads)
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
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
    fn concat_backward_rank1_two_inputs() {
        // Forward concat([a=[1,2,3], b=[4,5]]) along axis 0 → [1,2,3,4,5].
        // d_out = [10, 20, 30, 40, 50] → d_a = [10, 20, 30], d_b = [40, 50].
        let dy = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let bo = ConcatBackward {
            axis: 0,
            input_axis_sizes: vec![3, 2],
            input_shapes: vec![vec![3], vec![2]],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        assert_eq!(grads.len(), 2);
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![10.0, 20.0, 30.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![40.0, 50.0]);
    }

    #[test]
    fn concat_backward_rank2_axis_0() {
        // Forward concat along axis 0: shapes [2,2] + [1,2] = [3,2].
        // d_out [3, 2] = [[a, b], [c, d], [e, f]]
        // d_a = [[a, b], [c, d]];   d_b = [[e, f]]
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let bo = ConcatBackward {
            axis: 0,
            input_axis_sizes: vec![2, 1],
            input_shapes: vec![vec![2, 2], vec![1, 2]],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        let g0 = grads[0].as_ref().unwrap();
        let g1 = grads[1].as_ref().unwrap();
        assert_eq!(g0.shape(), &[2, 2]);
        assert_eq!(g1.shape(), &[1, 2]);
        assert_eq!(read_f32(g0), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(read_f32(g1), vec![5.0, 6.0]);
    }

    #[test]
    fn concat_backward_rank2_axis_1() {
        // Forward concat along axis 1: [[a,b],[c,d]] + [[e],[f]] = [[a,b,e],[c,d,f]]
        // d_out [2, 3] = [[1, 2, 3], [4, 5, 6]]
        // d_input0 [2,2] = [[1,2],[4,5]]; d_input1 [2,1] = [[3],[6]]
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let bo = ConcatBackward {
            axis: 1,
            input_axis_sizes: vec![2, 1],
            input_shapes: vec![vec![2, 2], vec![2, 1]],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        let g0 = grads[0].as_ref().unwrap();
        let g1 = grads[1].as_ref().unwrap();
        assert_eq!(read_f32(g0), vec![1.0, 2.0, 4.0, 5.0]);
        assert_eq!(read_f32(g1), vec![3.0, 6.0]);
    }

    #[test]
    fn concat_backward_three_inputs() {
        // sizes [1, 2, 1] along axis 0; total 4.
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let bo = ConcatBackward {
            axis: 0,
            input_axis_sizes: vec![1, 2, 1],
            input_shapes: vec![vec![1], vec![2], vec![1]],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        assert_eq!(grads.len(), 3);
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![1.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![2.0, 3.0]);
        assert_eq!(read_f32(grads[2].as_ref().unwrap()), vec![4.0]);
    }

    #[test]
    fn concat_backward_axis_total_mismatch_errors() {
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = ConcatBackward {
            axis: 0,
            input_axis_sizes: vec![2, 2], // sum 4 != grad size 3
            input_shapes: vec![vec![2], vec![2]],
            dtype: DType::F32,
        };
        let e = bo.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("sum of input axis sizes"));
    }

    #[test]
    fn op_metadata() {
        let bo = ConcatBackward {
            axis: 1,
            input_axis_sizes: vec![2, 3],
            input_shapes: vec![vec![1, 2], vec![1, 3]],
            dtype: DType::F32,
        };
        assert_eq!(bo.name(), "concat_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(!bo.requires_input(0));
    }
}
