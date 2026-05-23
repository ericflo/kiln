//! `RepeatBackward` — gradient of `repeat(x, axis, n)`.
//!
//! Forward tiles the input `n` times along `axis`. Backward
//! **sum-reduces** across the `n` copies — for each original
//! position, the gradient is the sum of the gradients at all
//! repeated positions.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct RepeatBackward {
    pub axis: usize,
    pub n: usize,
    pub input_shape: Vec<usize>,
    pub dtype: DType,
}

impl BackwardOp for RepeatBackward {
    fn name(&self) -> &'static str {
        "repeat_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let go_shape = grad_output.shape();
        if go_shape.len() != self.input_shape.len() {
            bail!(
                "RepeatBackward: grad rank {} != input rank {}",
                go_shape.len(),
                self.input_shape.len()
            );
        }
        if self.axis >= self.input_shape.len() {
            bail!("RepeatBackward: axis {} out of range", self.axis);
        }
        let mut expected = self.input_shape.clone();
        expected[self.axis] *= self.n;
        if go_shape != expected.as_slice() {
            bail!(
                "RepeatBackward: grad shape {:?} != expected {:?}",
                go_shape,
                expected
            );
        }
        if grad_output.dtype() != self.dtype {
            bail!("RepeatBackward: dtype mismatch");
        }
        if !grad_output.is_contiguous() {
            bail!("RepeatBackward: grad must be contiguous");
        }
        let dtype = self.dtype;
        let outer: usize = self.input_shape[..self.axis].iter().product::<usize>().max(1);
        let axis_in = self.input_shape[self.axis];
        let inner: usize = self.input_shape[self.axis + 1..]
            .iter()
            .product::<usize>()
            .max(1);

        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("RepeatBackward: grad storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();

        let axis_out = axis_in * self.n;
        let mut acc = vec![0.0f32; outer * axis_in * inner];

        for o in 0..outer {
            for rep in 0..self.n {
                for a in 0..axis_in {
                    for i in 0..inner {
                        let go_idx = (o * axis_out + rep * axis_in + a) * inner + i;
                        let v = match dtype {
                            DType::F32 => f32::from_le_bytes(
                                go_bytes[go_idx * 4..go_idx * 4 + 4].try_into().unwrap(),
                            ),
                            DType::BF16 => half::bf16::from_le_bytes(
                                go_bytes[go_idx * 2..go_idx * 2 + 2].try_into().unwrap(),
                            )
                            .to_f32(),
                            DType::F16 => half::f16::from_le_bytes(
                                go_bytes[go_idx * 2..go_idx * 2 + 2].try_into().unwrap(),
                            )
                            .to_f32(),
                            _ => unreachable!(),
                        };
                        acc[(o * axis_in + a) * inner + i] += v;
                    }
                }
            }
        }
        let per = dtype.size_in_bytes();
        let mut out_bytes = vec![0u8; acc.len() * per];
        match dtype {
            DType::F32 => {
                for (i, &v) in acc.iter().enumerate() {
                    out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::BF16 => {
                for (i, &v) in acc.iter().enumerate() {
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
            }
            DType::F16 => {
                for (i, &v) in acc.iter().enumerate() {
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let d_x = Tensor::from_parts(
            storage,
            Layout::contiguous(self.input_shape.clone()),
            TensorId::next(),
        )?;
        Ok(vec![Some(d_x)])
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
    fn repeat_backward_rank1_sums_repetitions() {
        // Forward: [1, 2, 3] repeated 3x at axis 0 → [9].
        // d_y = [10, 20, 30, 40, 50, 60, 70, 80, 90].
        // d_x[0] = 10+40+70 = 120; d_x[1] = 20+50+80 = 150; d_x[2] = 30+60+90 = 180.
        let dy = Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            vec![9],
        )
        .unwrap();
        let bo = RepeatBackward {
            axis: 0,
            n: 3,
            input_shape: vec![3],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![120.0, 150.0, 180.0]);
    }

    #[test]
    fn repeat_backward_rank2_axis_0() {
        // Forward [2, 2] repeated 2x at axis 0 → [4, 2].
        // d_y rows: [a, b], [c, d], [e, f], [g, h]
        // d_x[0] = [a+e, b+f]; d_x[1] = [c+g, d+h]
        let dy = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![4, 2],
        )
        .unwrap();
        let bo = RepeatBackward {
            axis: 0,
            n: 2,
            input_shape: vec![2, 2],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn repeat_backward_n_one_is_identity() {
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = RepeatBackward {
            axis: 0,
            n: 1,
            input_shape: vec![3],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_backward_axis_1() {
        // [[1,2],[3,4]] repeated 2x at axis 1 → [[1,2,1,2],[3,4,3,4]].
        // d_y = [[a,b,c,d],[e,f,g,h]] (8 values)
        // d_x[0,:] = [a+c, b+d]; d_x[1,:] = [e+g, f+h]
        let dy = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        )
        .unwrap();
        let bo = RepeatBackward {
            axis: 1,
            n: 2,
            input_shape: vec![2, 2],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![4.0, 6.0, 12.0, 14.0]);
    }

    #[test]
    fn op_metadata() {
        let bo = RepeatBackward {
            axis: 1,
            n: 2,
            input_shape: vec![3, 4],
            dtype: DType::F32,
        };
        assert_eq!(bo.name(), "repeat_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
