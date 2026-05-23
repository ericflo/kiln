//! `StackBackward` — gradient of `stack(inputs, axis)`.
//!
//! Forward inserts a new axis of size N and writes input `i` into
//! the `axis=i` slice. Backward extracts each `axis=i` slice from
//! the upstream grad, dropping the new axis, and routes it to the
//! corresponding input.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct StackBackward {
    /// Stack axis from the forward pass.
    pub axis: usize,
    /// Number of inputs (= the size of the new axis in the output).
    pub n_inputs: usize,
    /// Original per-input shape (each input shared this shape).
    pub input_shape: Vec<usize>,
    /// dtype of the inputs / output.
    pub dtype: DType,
}

impl BackwardOp for StackBackward {
    fn name(&self) -> &'static str {
        "stack_backward"
    }
    fn input_count(&self) -> usize {
        self.n_inputs
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let go_shape = grad_output.shape();
        let expected_rank = self.input_shape.len() + 1;
        if go_shape.len() != expected_rank {
            bail!(
                "StackBackward: grad rank {} != expected {expected_rank}",
                go_shape.len()
            );
        }
        if self.axis > self.input_shape.len() {
            bail!(
                "StackBackward: axis {} > input rank {}",
                self.axis,
                self.input_shape.len()
            );
        }
        // Validate the grad output shape matches what stack would produce.
        let mut expected = self.input_shape.clone();
        expected.insert(self.axis, self.n_inputs);
        if go_shape != expected.as_slice() {
            bail!(
                "StackBackward: grad shape {:?} != expected {:?}",
                go_shape,
                expected
            );
        }
        if grad_output.dtype() != self.dtype {
            bail!("StackBackward: dtype mismatch");
        }
        if !grad_output.is_contiguous() {
            bail!("StackBackward: grad must be contiguous");
        }
        let per = self.dtype.size_in_bytes();
        let outer: usize = self.input_shape[..self.axis].iter().product::<usize>().max(1);
        let inner: usize = self.input_shape[self.axis..]
            .iter()
            .product::<usize>()
            .max(1);
        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("StackBackward: storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();

        let mut grads: Vec<Option<Tensor>> = Vec::with_capacity(self.n_inputs);
        for i in 0..self.n_inputs {
            let mut bytes = vec![0u8; outer * inner * per];
            for o in 0..outer {
                let src_start = (o * self.n_inputs + i) * inner * per;
                let src_end = src_start + inner * per;
                let dst_start = o * inner * per;
                let dst_end = dst_start + inner * per;
                bytes[dst_start..dst_end].copy_from_slice(&go_bytes[src_start..src_end]);
            }
            let cpu = CpuStorage::from_bytes(self.dtype, bytes)?;
            let storage: Storage = Arc::new(cpu);
            let g = Tensor::from_parts(
                storage,
                Layout::contiguous(self.input_shape.clone()),
                TensorId::next(),
            )?;
            grads.push(Some(g));
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
    fn stack_backward_axis_0() {
        // Forward stack [a [3], b [3]] at axis 0 → [2, 3].
        // d_out [2, 3]: row 0 → d_a, row 1 → d_b.
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let bo = StackBackward {
            axis: 0,
            n_inputs: 2,
            input_shape: vec![3],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[3]);
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![1.0, 2.0, 3.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_backward_axis_1() {
        // Forward stack [a [3], b [3]] at axis 1 → [3, 2].
        // d_out [3, 2] = [[1, 10], [2, 20], [3, 30]]
        // d_a = column 0 = [1, 2, 3]; d_b = column 1 = [10, 20, 30].
        let dy = Tensor::from_slice(&[1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0], vec![3, 2]).unwrap();
        let bo = StackBackward {
            axis: 1,
            n_inputs: 2,
            input_shape: vec![3],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![1.0, 2.0, 3.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn stack_backward_three_inputs() {
        // 3 inputs of shape [2] stacked at axis 0 → [3, 2].
        let dy = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        )
        .unwrap();
        let bo = StackBackward {
            axis: 0,
            n_inputs: 3,
            input_shape: vec![2],
            dtype: DType::F32,
        };
        let grads = bo.apply(&dy).unwrap();
        assert_eq!(grads.len(), 3);
        assert_eq!(read_f32(grads[0].as_ref().unwrap()), vec![1.0, 2.0]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![3.0, 4.0]);
        assert_eq!(read_f32(grads[2].as_ref().unwrap()), vec![5.0, 6.0]);
    }

    #[test]
    fn stack_backward_shape_mismatch_errors() {
        let dy = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = StackBackward {
            axis: 0,
            n_inputs: 2,
            input_shape: vec![3],
            dtype: DType::F32,
        };
        let e = bo.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn op_metadata() {
        let bo = StackBackward {
            axis: 0,
            n_inputs: 3,
            input_shape: vec![4],
            dtype: DType::F32,
        };
        assert_eq!(bo.name(), "stack_backward");
        assert_eq!(bo.input_count(), 3);
        assert!(!bo.requires_input(0));
    }
}
