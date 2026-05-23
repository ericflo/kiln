//! `CumsumBackward` — gradient of `cumsum(x, axis)`.
//!
//! Forward: `y[i] = Σ_{j ≤ i} x[j]` along `axis`.
//!
//! Backward: `dx[i] = Σ_{j ≥ i} dy[j]` — the reverse cumulative
//! sum. (Every later position's gradient flows back to this position
//! because y_j depends on x_i for all j ≥ i.)

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct CumsumBackward {
    pub axis: usize,
    pub input_shape: Vec<usize>,
    pub dtype: DType,
}

impl BackwardOp for CumsumBackward {
    fn name(&self) -> &'static str {
        "cumsum_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if grad_output.shape() != self.input_shape.as_slice() {
            bail!(
                "CumsumBackward: grad shape {:?} != input shape {:?}",
                grad_output.shape(),
                self.input_shape
            );
        }
        if grad_output.dtype() != self.dtype {
            bail!("CumsumBackward: dtype mismatch");
        }
        if self.axis >= self.input_shape.len() {
            bail!("CumsumBackward: axis {} out of range", self.axis);
        }
        if !grad_output.is_contiguous() {
            bail!("CumsumBackward: grad must be contiguous");
        }
        let dtype = self.dtype;
        let outer: usize = self.input_shape[..self.axis].iter().product::<usize>().max(1);
        let axis_dim = self.input_shape[self.axis];
        let inner: usize = self.input_shape[self.axis + 1..]
            .iter()
            .product::<usize>()
            .max(1);
        let cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("CumsumBackward: storage must be CpuStorage"))?;
        let bytes = cpu.as_bytes();

        // Reverse cumsum: walk from the largest axis index back to 0,
        // accumulating dy values into dx.
        let mut dx = vec![0.0f32; outer * axis_dim * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0.0f32;
                for a in (0..axis_dim).rev() {
                    let idx = (o * axis_dim + a) * inner + i;
                    let v = match dtype {
                        DType::F32 => f32::from_le_bytes(
                            bytes[idx * 4..idx * 4 + 4].try_into().unwrap(),
                        ),
                        DType::BF16 => half::bf16::from_le_bytes(
                            bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                        )
                        .to_f32(),
                        DType::F16 => half::f16::from_le_bytes(
                            bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                        )
                        .to_f32(),
                        _ => unreachable!(),
                    };
                    acc += v;
                    dx[idx] = acc;
                }
            }
        }
        let per = dtype.size_in_bytes();
        let mut out_bytes = vec![0u8; dx.len() * per];
        match dtype {
            DType::F32 => {
                for (i, &v) in dx.iter().enumerate() {
                    out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            DType::BF16 => {
                for (i, &v) in dx.iter().enumerate() {
                    out_bytes[i * 2..i * 2 + 2]
                        .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
            }
            DType::F16 => {
                for (i, &v) in dx.iter().enumerate() {
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
    fn cumsum_backward_rank1() {
        // Forward cumsum [1,2,3,4] = [1,3,6,10].
        // For sum-of-output loss, d_y = ones.
        // d_x[i] = Σ_{j ≥ i} 1 = (n - i) = [4, 3, 2, 1].
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let bo = CumsumBackward {
            axis: 0,
            input_shape: vec![4],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn cumsum_backward_arbitrary_grad() {
        // dy = [10, 20, 30, 40]
        // dx[0] = 10+20+30+40 = 100
        // dx[1] = 20+30+40 = 90
        // dx[2] = 30+40 = 70
        // dx[3] = 40
        let dy = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let bo = CumsumBackward {
            axis: 0,
            input_shape: vec![4],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![100.0, 90.0, 70.0, 40.0]);
    }

    #[test]
    fn cumsum_backward_rank2_axis_1() {
        // dy [[1,2,3], [4,5,6]] axis 1
        // Row 0 reverse-cumsum: [6, 5, 3] (right-to-left running sum)
        //   dx[0,2]=3; dx[0,1]=2+3=5; dx[0,0]=1+2+3=6
        // Row 1: [15, 11, 6]
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let bo = CumsumBackward {
            axis: 1,
            input_shape: vec![2, 3],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]);
    }

    #[test]
    fn cumsum_backward_rank2_axis_0() {
        // dy [[1,2,3], [4,5,6]] axis 0
        // dx[:, c] = reverse cumsum along axis 0.
        // Col 0: dx[1,0]=4; dx[0,0]=1+4=5
        // Col 1: dx[1,1]=5; dx[0,1]=2+5=7
        // Col 2: dx[1,2]=6; dx[0,2]=3+6=9
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let bo = CumsumBackward {
            axis: 0,
            input_shape: vec![2, 3],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![5.0, 7.0, 9.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn cumsum_backward_finite_difference() {
        use kiln_tensor::ops::cumsum;
        let x_data = vec![1.5f32, -2.0, 3.3, 0.7];
        let x = Tensor::from_slice(&x_data, vec![4]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32; 4], vec![4]).unwrap();
        let bo = CumsumBackward {
            axis: 0,
            input_shape: vec![4],
            dtype: DType::F32,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        let loss = |xv: &[f32]| -> f32 {
            let xt = Tensor::from_slice(xv, vec![4]).unwrap();
            read_f32(&cumsum(&xt, 0).unwrap()).iter().sum()
        };
        let step = 1e-3;
        let mut fd = Vec::with_capacity(4);
        for i in 0..4 {
            let mut up = x_data.clone();
            up[i] += step;
            let mut dn = x_data.clone();
            dn[i] -= step;
            fd.push((loss(&up) - loss(&dn)) / (2.0 * step));
        }
        for (i, (g, f)) in dx.iter().zip(fd.iter()).enumerate() {
            assert!((g - f).abs() < 1e-3, "idx {i}: got {g}, fd {f}");
        }
    }

    #[test]
    fn op_metadata() {
        let bo = CumsumBackward {
            axis: 0,
            input_shape: vec![1],
            dtype: DType::F32,
        };
        assert_eq!(bo.name(), "cumsum_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
