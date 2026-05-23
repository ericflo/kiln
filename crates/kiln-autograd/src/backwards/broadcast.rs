//! `BroadcastToBackward` — gradient of `broadcast_to(x, target_shape)`.
//!
//! Forward replicates input axes whose size is 1 to match the target
//! shape. Backward sums the upstream gradient along those broadcast
//! axes back to the original input shape.
//!
//! ```text
//! d_x[indices_in] = Σ over all output indices that share `indices_in`
//!                   after the broadcast — i.e. sum-reduce along every
//!                   axis where the input had size 1 and the output had
//!                   size > 1.
//! ```

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct BroadcastToBackward {
    /// Original input shape from the forward pass.
    pub input_shape: Vec<usize>,
}

impl BackwardOp for BroadcastToBackward {
    fn name(&self) -> &'static str {
        "broadcast_to_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let target_shape = grad_output.shape();
        if self.input_shape.len() != target_shape.len() {
            bail!(
                "BroadcastToBackward: rank mismatch: input {:?} vs grad {:?}",
                self.input_shape,
                target_shape
            );
        }
        for (axis, (&in_d, &out_d)) in self.input_shape.iter().zip(target_shape).enumerate() {
            if in_d != out_d && in_d != 1 {
                bail!(
                    "BroadcastToBackward: axis {axis} input dim {in_d} != grad dim {out_d} (not a broadcast axis)"
                );
            }
        }
        let dtype = grad_output.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "BroadcastToBackward: dtype must be F32/BF16/F16, got {dtype}"
            );
        }
        if !grad_output.is_contiguous() {
            bail!("BroadcastToBackward: grad must be contiguous");
        }
        let go_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("BroadcastToBackward: storage must be CpuStorage"))?;
        let go_bytes = go_cpu.as_bytes();
        let rank = self.input_shape.len();
        let target_total: usize = target_shape.iter().product();
        let input_total: usize = self.input_shape.iter().product::<usize>().max(1);

        // out_strides for the grad shape (row-major).
        let mut out_strides = vec![1usize; rank];
        for k in (0..rank.saturating_sub(1)).rev() {
            out_strides[k] = out_strides[k + 1] * target_shape[k + 1];
        }
        // in_strides for the original input shape (row-major).
        let mut in_strides = vec![1usize; rank];
        for k in (0..rank.saturating_sub(1)).rev() {
            in_strides[k] = in_strides[k + 1] * self.input_shape[k + 1];
        }

        // Sum-reduce: walk every grad index; accumulate into the
        // corresponding input slot (after collapsing broadcast axes
        // to index 0).
        let mut acc = vec![0.0f32; input_total];
        for flat_out in 0..target_total {
            let mut rem = flat_out;
            let mut in_offset = 0usize;
            for k in 0..rank {
                let idx_out = rem / out_strides[k];
                rem %= out_strides[k];
                let idx_in = if self.input_shape[k] == 1 { 0 } else { idx_out };
                in_offset += idx_in * in_strides[k];
            }
            let v = match dtype {
                DType::F32 => f32::from_le_bytes(
                    go_bytes[flat_out * 4..flat_out * 4 + 4].try_into().unwrap(),
                ),
                DType::BF16 => half::bf16::from_le_bytes(
                    go_bytes[flat_out * 2..flat_out * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                DType::F16 => half::f16::from_le_bytes(
                    go_bytes[flat_out * 2..flat_out * 2 + 2].try_into().unwrap(),
                )
                .to_f32(),
                _ => unreachable!(),
            };
            acc[in_offset] += v;
        }
        // Cast back to dtype and build the output tensor.
        let per = dtype.size_in_bytes();
        let mut out_bytes = vec![0u8; input_total * per];
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
    fn broadcast_backward_rank1_sums_along_axis() {
        // Forward broadcast [5.0] → [5.0, 5.0, 5.0]. d_y = [1, 2, 3] →
        // d_x = sum = [6.0].
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = BroadcastToBackward {
            input_shape: vec![1],
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![6.0]);
    }

    #[test]
    fn broadcast_backward_rank2_sums_axis_0() {
        // Forward [1, 3] → [4, 3]. d_y [4, 3]:
        //   [[1, 2, 3],
        //    [4, 5, 6],
        //    [7, 8, 9],
        //    [10, 11, 12]]
        // d_x[0, c] = sum_r d_y[r, c] = column sums = [22, 26, 30]
        let dy = Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let bo = BroadcastToBackward {
            input_shape: vec![1, 3],
        };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(dx.shape(), &[1, 3]);
        assert_eq!(read_f32(&dx), vec![22.0, 26.0, 30.0]);
    }

    #[test]
    fn broadcast_backward_rank2_sums_axis_1() {
        // Forward [2, 1] → [2, 3]. d_y [2, 3]:
        //   [[1, 2, 3], [4, 5, 6]]
        // d_x[r, 0] = sum_c d_y[r, c] = [6, 15]
        let dy = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let bo = BroadcastToBackward {
            input_shape: vec![2, 1],
        };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&dx), vec![6.0, 15.0]);
    }

    #[test]
    fn broadcast_backward_both_axes() {
        // Forward [1, 1] → [2, 2]. d_y = [[1, 2], [3, 4]].
        // d_x = sum = [10].
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = BroadcastToBackward {
            input_shape: vec![1, 1],
        };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&dx), vec![10.0]);
    }

    #[test]
    fn broadcast_backward_identity_no_reduction() {
        // Same shape → identity copy.
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = BroadcastToBackward {
            input_shape: vec![3],
        };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&dx), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcast_backward_rank_mismatch_errors() {
        let dy = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = BroadcastToBackward {
            input_shape: vec![1, 1],
        };
        let e = bo.apply(&dy).unwrap_err();
        assert!(e.to_string().contains("rank mismatch"));
    }

    #[test]
    fn op_metadata() {
        let bo = BroadcastToBackward {
            input_shape: vec![1, 3],
        };
        assert_eq!(bo.name(), "broadcast_to_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
