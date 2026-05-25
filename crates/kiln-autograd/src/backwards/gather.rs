//! `GatherBackward` — gradient of `gather(x, axis, indices)`.
//!
//! Forward (PyTorch-style, `torch.gather`):
//! ```text
//! out[c_0, ..., c_{rank-1}] = x[c_0, ..., c_axis = indices[c_0, ..., c_{rank-1}], ..., c_{rank-1}]
//! ```
//!
//! `indices` shares rank with `x` and matches `x.shape` everywhere
//! except possibly along `axis`. Output shape == `indices.shape`.
//!
//! Backward: each gradient element `grad_output[c]` flows back to
//! `d_x[c_with_axis_replaced_by_indices[c]]`, accumulating on
//! collisions:
//! ```text
//! d_x[c_0, ..., c_axis = indices[c], ..., c_{rank-1}] += grad_output[c]
//! ```
//!
//! This is element-wise scatter-add, which is *different* from
//! `kiln_tensor::ops::scatter_add` (which uses index_select-style
//! slab semantics). We therefore implement the byte-level accumulation
//! directly, mirroring the gather forward's iteration.

use std::sync::Arc;

use kiln_tensor::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

use crate::BackwardOp;

#[derive(Debug)]
pub struct GatherBackward {
    /// `axis` from the forward call.
    pub axis: usize,
    /// Saved `x.shape` from the forward (target shape for `d_x`).
    pub source_shape: Vec<usize>,
    /// Saved indices from the forward call.
    pub indices: Tensor,
}

fn read_indices(t: &Tensor) -> Result<Vec<i64>> {
    if !t.is_contiguous() {
        bail!("GatherBackward: indices must be contiguous");
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("GatherBackward: indices must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::I64 => {
            for i in 0..n {
                out.push(i64::from_le_bytes(
                    bytes[i * 8..i * 8 + 8].try_into().unwrap(),
                ));
            }
        }
        DType::U32 => {
            for i in 0..n {
                out.push(u32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ) as i64);
            }
        }
        other => bail!(
            "GatherBackward: indices dtype must be I64 or U32, got {other}"
        ),
    }
    Ok(out)
}

impl BackwardOp for GatherBackward {
    fn name(&self) -> &'static str {
        "gather_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let rank = self.source_shape.len();
        if self.axis >= rank {
            bail!(
                "GatherBackward: axis {} out of bounds for rank {}",
                self.axis,
                rank
            );
        }
        if self.indices.rank() != rank {
            bail!(
                "GatherBackward: indices rank {} != source rank {}",
                self.indices.rank(),
                rank
            );
        }
        if grad_output.shape() != self.indices.shape() {
            bail!(
                "GatherBackward: grad shape {:?} != indices shape {:?}",
                grad_output.shape(),
                self.indices.shape()
            );
        }
        let dtype = grad_output.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "GatherBackward: grad dtype must be F32/BF16/F16, got {dtype}"
            );
        }
        if !grad_output.is_contiguous() {
            bail!("GatherBackward: grad must be contiguous");
        }
        // Validate non-axis shape agreement.
        let i_shape: Vec<usize> = self.indices.shape().to_vec();
        for d in 0..rank {
            if d == self.axis {
                continue;
            }
            if self.source_shape[d] != i_shape[d] {
                bail!(
                    "GatherBackward: shape mismatch at axis {d} — source {:?} vs indices {i_shape:?} (axis {})",
                    self.source_shape,
                    self.axis
                );
            }
        }
        let source_axis_len = self.source_shape[self.axis];
        let idx_flat = read_indices(&self.indices)?;
        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("GatherBackward: grad storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();

        // Strides for the indices (== grad_output) shape and source shape.
        let mut i_strides = vec![1usize; rank];
        let mut x_strides = vec![1usize; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            i_strides[d] = i_strides[d + 1] * i_shape[d + 1];
            x_strides[d] = x_strides[d + 1] * self.source_shape[d + 1];
        }

        let source_total: usize = self.source_shape.iter().product();
        let mut acc = vec![0.0f32; source_total];
        let n_grad: usize = i_shape.iter().product();

        let mut coord = vec![0usize; rank];
        for out_idx in 0..n_grad {
            let mut rem = out_idx;
            for d in 0..rank {
                coord[d] = rem / i_strides[d];
                rem %= i_strides[d];
            }
            let idx_val = idx_flat[out_idx];
            if idx_val < 0 || (idx_val as usize) >= source_axis_len {
                bail!(
                    "GatherBackward: index {idx_val} out of bounds for axis {} of length {source_axis_len}",
                    self.axis
                );
            }
            // Compute destination offset in d_x: same coord but axis
            // replaced with idx_val.
            let mut dst_off = 0usize;
            for d in 0..rank {
                let c = if d == self.axis {
                    idx_val as usize
                } else {
                    coord[d]
                };
                dst_off += c * x_strides[d];
            }
            let v = match dtype {
                DType::F32 => f32::from_le_bytes(
                    go_bytes[out_idx * 4..out_idx * 4 + 4].try_into().unwrap(),
                ),
                DType::BF16 => half::bf16::from_le_bytes(
                    go_bytes[out_idx * 2..out_idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    go_bytes[out_idx * 2..out_idx * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            acc[dst_off] += v;
        }

        // Cast back to dtype and build the output tensor.
        let per = dtype.size_in_bytes();
        let mut out_bytes = vec![0u8; source_total * per];
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
            Layout::contiguous(self.source_shape.clone()),
            TensorId::next(),
        )?;
        Ok(vec![Some(d_x), None /* indices non-differentiable */])
    }
    fn requires_input(&self, idx: usize) -> bool {
        match idx {
            0 => false, // source not needed (only shape, saved on struct)
            1 => true,  // indices saved
            _ => false,
        }
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
    fn gather_backward_1d_routes_grad() {
        // Forward: x=[a,b,c,d,e], idx=[4,0,2] -> [e,a,c]
        // Grad in: [1,2,3] -> d_x = [2, 0, 3, 0, 1].
        let indices = Tensor::from_slice(&[4i64, 0, 2], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = GatherBackward {
            axis: 0,
            source_shape: vec![5],
            indices,
        };
        let grads = bo.apply(&grad).unwrap();
        let d = grads[0].as_ref().unwrap();
        assert_eq!(d.shape(), &[5]);
        assert_eq!(read_f32(d), vec![2.0, 0.0, 3.0, 0.0, 1.0]);
        assert!(grads[1].is_none());
    }

    #[test]
    fn gather_backward_2d_axis_1() {
        // Forward: x=[[1,2,3],[4,5,6]], idx=[[2,0,1],[1,1,0]] -> y[2,3].
        // Grad y = [[10,20,30],[40,50,60]].
        // d_x[r=0, idx[0,k]] += grad[0,k]:
        //   d_x[0, 2] += 10, d_x[0, 0] += 20, d_x[0, 1] += 30 -> [20, 30, 10]
        // d_x[r=1, idx[1,k]] += grad[1,k]:
        //   d_x[1, 1] += 40, d_x[1, 1] += 50, d_x[1, 0] += 60 -> [60, 90, 0]
        let indices = Tensor::from_slice(&[2i64, 0, 1, 1, 1, 0], vec![2, 3]).unwrap();
        let grad = Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![2, 3],
        )
        .unwrap();
        let bo = GatherBackward {
            axis: 1,
            source_shape: vec![2, 3],
            indices,
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 3]);
        assert_eq!(
            read_f32(&d),
            vec![20.0, 30.0, 10.0, 60.0, 90.0, 0.0]
        );
    }

    #[test]
    fn gather_backward_collisions_accumulate() {
        // Forward: idx=[1,1,1] -> all gather row 1.
        // Backward accumulates all grads into row 1.
        let indices = Tensor::from_slice(&[1i64, 1, 1], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 10.0, 100.0], vec![3]).unwrap();
        let bo = GatherBackward {
            axis: 0,
            source_shape: vec![3],
            indices,
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&d), vec![0.0, 111.0, 0.0]);
    }

    #[test]
    fn gather_backward_flce_label_shape() {
        // FLCE-style: gather one column per row.
        // source [hits=2, cols=4]. idx [[3],[1]] axis=1 → out shape [2,1].
        // Grad [[g0],[g1]] → d_x[0,3]+=g0, d_x[1,1]+=g1.
        let indices = Tensor::from_slice(&[3i64, 1], vec![2, 1]).unwrap();
        let grad = Tensor::from_slice(&[7.0f32, 11.0], vec![2, 1]).unwrap();
        let bo = GatherBackward {
            axis: 1,
            source_shape: vec![2, 4],
            indices,
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 4]);
        assert_eq!(
            read_f32(&d),
            vec![0.0, 0.0, 0.0, 7.0, 0.0, 11.0, 0.0, 0.0]
        );
    }

    #[test]
    fn gather_backward_bf16_round_trips() {
        // Same numeric pattern as gather_backward_1d_routes_grad in BF16.
        let indices = Tensor::from_slice(&[4i64, 0, 2], vec![3]).unwrap();
        let grad_bf: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let grad = Tensor::from_slice(&grad_bf, vec![3]).unwrap();
        let bo = GatherBackward {
            axis: 0,
            source_shape: vec![5],
            indices,
        };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.dtype(), DType::BF16);
        let bytes = d
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes();
        let vals: Vec<f32> = (0..5)
            .map(|i| {
                half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            })
            .collect();
        assert_eq!(vals, vec![2.0, 0.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn gather_backward_oob_errors() {
        let indices = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = GatherBackward {
            axis: 0,
            source_shape: vec![3],
            indices,
        };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("out of bounds"));
    }

    #[test]
    fn gather_backward_shape_mismatch_errors() {
        let indices = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        // Grad shape mismatch with indices shape.
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = GatherBackward {
            axis: 0,
            source_shape: vec![3],
            indices,
        };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("grad shape"));
    }

    #[test]
    fn op_metadata() {
        let idx = Tensor::from_slice(&[0i64], vec![1]).unwrap();
        let bo = GatherBackward {
            axis: 0,
            source_shape: vec![1],
            indices: idx,
        };
        assert_eq!(bo.name(), "gather_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(!bo.requires_input(0));
        assert!(bo.requires_input(1));
    }
}
