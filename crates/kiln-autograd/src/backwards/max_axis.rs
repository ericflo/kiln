//! `MaxAxisBackward` — gradient of `max_axis(x, axis)`.
//!
//! Forward reduces `x` along `axis`, returning the per-slab max
//! (output shape == input shape with `axis` removed).
//!
//! Backward routes the gradient to the **first** position whose
//! value equals the slab maximum (PyTorch convention for
//! `torch.max(x, dim=k).values.backward()`):
//!
//! ```text
//! d_x[..., j, ...] = grad_output[...]  if j == argmax(x[..., :, ...])
//!                  = 0                  otherwise
//! ```
//!
//! Ties along `axis` are routed entirely to the first occurrence
//! (mirrors `argmax` tie-break in PyTorch / NumPy).
//!
//! The forward input `x` is saved on the struct; the saved output
//! max is *not* needed since we recompute argmax from `x` during the
//! backward pass.

use std::sync::Arc;

use kiln_tensor::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

use crate::BackwardOp;

#[derive(Debug)]
pub struct MaxAxisBackward {
    /// `axis` from the forward call.
    pub axis: usize,
    /// Saved forward input `x` (needed to locate argmax positions).
    pub input: Tensor,
}

fn read_one_f32(dtype: DType, bytes: &[u8], i: usize, per: usize) -> f32 {
    let s = i * per;
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[s..s + 4].try_into().unwrap()),
        DType::BF16 => half::bf16::from_le_bytes(bytes[s..s + 2].try_into().unwrap()).to_f32(),
        DType::F16 => half::f16::from_le_bytes(bytes[s..s + 2].try_into().unwrap()).to_f32(),
        _ => unreachable!(),
    }
}

impl BackwardOp for MaxAxisBackward {
    fn name(&self) -> &'static str {
        "max_axis_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let rank = self.input.rank();
        if self.axis >= rank {
            bail!(
                "MaxAxisBackward: axis {} out of bounds for rank {}",
                self.axis,
                rank
            );
        }
        let dtype = grad_output.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "MaxAxisBackward: grad dtype must be F32/BF16/F16, got {dtype}"
            );
        }
        if dtype != self.input.dtype() {
            bail!(
                "MaxAxisBackward: grad dtype {dtype} != saved input dtype {}",
                self.input.dtype()
            );
        }
        if !grad_output.is_contiguous() || !self.input.is_contiguous() {
            bail!("MaxAxisBackward: grad and input must be contiguous");
        }
        let x_shape = self.input.shape().to_vec();
        let mut expected_g_shape = x_shape.clone();
        expected_g_shape.remove(self.axis);
        if grad_output.shape() != expected_g_shape.as_slice() {
            bail!(
                "MaxAxisBackward: grad shape {:?} != expected {:?} (input {:?} reduced on axis {})",
                grad_output.shape(),
                expected_g_shape,
                x_shape,
                self.axis
            );
        }
        let per = dtype.size_in_bytes();
        let x_cpu = self
            .input
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaxAxisBackward: input must be CpuStorage"))?;
        let x_bytes = x_cpu.as_bytes();
        let g_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaxAxisBackward: grad must be CpuStorage"))?;
        let g_bytes = g_cpu.as_bytes();

        let axis_len = x_shape[self.axis];
        let outer: usize = x_shape[..self.axis].iter().product();
        let inner: usize = x_shape[self.axis + 1..].iter().product();

        let x_total: usize = x_shape.iter().product();
        let mut out_bytes = vec![0u8; x_total * per];

        // For each (outer, inner) slab, find the first argmax along
        // axis, and write the gradient to that slot.
        for o in 0..outer {
            for k in 0..inner {
                // Find the position of the maximum value along axis.
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for a in 0..axis_len {
                    let src_off = (o * axis_len + a) * inner + k;
                    let v = read_one_f32(dtype, x_bytes, src_off, per);
                    if v > best_val {
                        best_val = v;
                        best_idx = a;
                    }
                }
                let dst_off = (o * axis_len + best_idx) * inner + k;
                let grad_off = o * inner + k;
                let src_slice = &g_bytes[grad_off * per..grad_off * per + per];
                out_bytes[dst_off * per..dst_off * per + per].copy_from_slice(src_slice);
            }
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let d_x = Tensor::from_parts(
            storage,
            Layout::contiguous(x_shape),
            TensorId::next(),
        )?;
        Ok(vec![Some(d_x)])
    }
    fn requires_input(&self, idx: usize) -> bool {
        // Need the forward input to locate argmax.
        idx == 0
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
    fn max_axis_backward_rank2_axis_1() {
        // x = [[1, 5, 3], [9, 2, 7]], max axis=1 → [5, 9].
        // argmax along axis 1: row 0 → 1; row 1 → 0.
        // d_x: zero except where the max occurred. Grad in [10, 20]:
        //   d_x[0, 1] = 10; d_x[1, 0] = 20; rest 0.
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]).unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let bo = MaxAxisBackward { axis: 1, input: x };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 3]);
        assert_eq!(
            read_f32(&d),
            vec![
                0.0, 10.0, 0.0, // row 0: col 1
                20.0, 0.0, 0.0, // row 1: col 0
            ]
        );
    }

    #[test]
    fn max_axis_backward_rank2_axis_0() {
        // x = [[1, 5, 3], [9, 2, 7]], max axis=0 → [9, 5, 7].
        // argmax along axis 0: col 0 → 1; col 1 → 0; col 2 → 1.
        // Grad [a, b, c] → d_x[1,0]=a, d_x[0,1]=b, d_x[1,2]=c.
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = MaxAxisBackward { axis: 0, input: x };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 3]);
        assert_eq!(
            read_f32(&d),
            vec![
                0.0, 2.0, 0.0, // row 0
                1.0, 0.0, 3.0, // row 1
            ]
        );
    }

    #[test]
    fn max_axis_backward_first_tie_wins() {
        // Ties along axis → grad routes to the FIRST occurrence.
        // x = [5, 5, 5] → max=5; argmax=0; d_x = [grad, 0, 0].
        let x = Tensor::from_slice(&[5.0f32, 5.0, 5.0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[7.0f32], vec![]).unwrap();
        let bo = MaxAxisBackward { axis: 0, input: x };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&d), vec![7.0, 0.0, 0.0]);
    }

    #[test]
    fn max_axis_backward_rank1_reduce_to_scalar() {
        // x = [3, 1, 4, 1, 5, 9, 2], max axis 0 → scalar 9, argmax=5.
        let x = Tensor::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], vec![7]).unwrap();
        let grad = Tensor::from_slice(&[2.0f32], vec![]).unwrap();
        let bo = MaxAxisBackward { axis: 0, input: x };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[7]);
        assert_eq!(read_f32(&d), vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn max_axis_backward_3d_axis_1() {
        // x.shape = [2, 3, 2]. axis 1 → out shape [2, 2].
        // Hand-pick maxes to verify routing.
        // Slab (o=0, k=0) over rows: 1, 5, 3 → argmax = 1 → d_x[0,1,0] = g[0,0]
        // Slab (o=0, k=1) over rows: 2, 4, 6 → argmax = 2 → d_x[0,2,1] = g[0,1]
        // Slab (o=1, k=0) over rows: 9, 0, 9 → argmax = 0 (first tie) → d_x[1,0,0] = g[1,0]
        // Slab (o=1, k=1) over rows: 7, 8, 7 → argmax = 1 → d_x[1,1,1] = g[1,1]
        let x = Tensor::from_slice(
            &[
                1.0f32, 2.0, // [0, 0]
                5.0, 4.0, // [0, 1]
                3.0, 6.0, // [0, 2]
                9.0, 7.0, // [1, 0]
                0.0, 8.0, // [1, 1]
                9.0, 7.0, // [1, 2]
            ],
            vec![2, 3, 2],
        )
        .unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let bo = MaxAxisBackward { axis: 1, input: x };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 3, 2]);
        let expected = vec![
            0.0, 0.0, // [0, 0]
            10.0, 0.0, // [0, 1]
            0.0, 20.0, // [0, 2]
            30.0, 0.0, // [1, 0]
            0.0, 40.0, // [1, 1]
            0.0, 0.0, // [1, 2]
        ];
        assert_eq!(read_f32(&d), expected);
    }

    #[test]
    fn max_axis_backward_bf16_round_trips() {
        let x_bf: Vec<half::bf16> = [1.0f32, 5.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&x_bf, vec![3]).unwrap();
        let g_bf: Vec<half::bf16> = [4.0f32].iter().map(|&v| half::bf16::from_f32(v)).collect();
        let grad = Tensor::from_slice(&g_bf, vec![]).unwrap();
        let bo = MaxAxisBackward { axis: 0, input: x };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        let bytes = d
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes();
        let vals: Vec<f32> = (0..3)
            .map(|i| {
                half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            })
            .collect();
        assert_eq!(vals, vec![0.0, 4.0, 0.0]);
    }

    #[test]
    fn max_axis_backward_shape_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        // Grad has wrong rank.
        let grad = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = MaxAxisBackward { axis: 0, input: x };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("grad shape"));
    }

    #[test]
    fn op_metadata() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = MaxAxisBackward { axis: 0, input: x };
        assert_eq!(bo.name(), "max_axis_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(bo.requires_input(0));
    }
}
