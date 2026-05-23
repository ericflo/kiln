//! `MaskedFillBackward` — gradient of `masked_fill(x, mask, fill_value)`.
//!
//! Forward (from `kiln_tensor::ops::masked_fill`):
//!
//! ```text
//! out[i] = mask[i] != 0 ? fill_value : x[i]
//! ```
//!
//! Backward: positions where the mask was set get zero gradient (the
//! input `x[i]` had no effect on the output there). Positions where
//! the mask was clear pass the gradient through unchanged.
//!
//! ```text
//! d_x[i]   = mask[i] != 0 ? 0 : grad_output[i]
//! d_mask   = None (boolean — non-differentiable)
//! ```

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct MaskedFillBackward {
    /// Saved mask (U8) from the forward pass.
    pub mask: Tensor,
}

impl BackwardOp for MaskedFillBackward {
    fn name(&self) -> &'static str {
        "masked_fill_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if self.mask.dtype() != DType::U8 {
            bail!(
                "MaskedFillBackward: mask dtype must be U8, got {}",
                self.mask.dtype()
            );
        }
        if self.mask.shape() != grad_output.shape() {
            bail!(
                "MaskedFillBackward: mask shape {:?} != grad shape {:?}",
                self.mask.shape(),
                grad_output.shape()
            );
        }
        if !matches!(grad_output.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "MaskedFillBackward: grad dtype must be F32/BF16/F16, got {}",
                grad_output.dtype()
            );
        }
        if !self.mask.is_contiguous() || !grad_output.is_contiguous() {
            bail!("MaskedFillBackward: mask and grad must be contiguous");
        }

        let dtype = grad_output.dtype();
        let per = dtype.size_in_bytes();
        let n = grad_output.element_count();
        let g_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaskedFillBackward: grad storage must be CpuStorage"))?;
        let m_cpu = self
            .mask
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("MaskedFillBackward: mask storage must be CpuStorage"))?;
        let g_bytes = g_cpu.as_bytes();
        let m_bytes = m_cpu.as_bytes();
        let mut out = vec![0u8; n * per];

        match dtype {
            DType::F32 => {
                for i in 0..n {
                    if m_bytes[i] == 0 {
                        out[i * 4..i * 4 + 4]
                            .copy_from_slice(&g_bytes[i * 4..i * 4 + 4]);
                    }
                    // else: zero, already initialized.
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    if m_bytes[i] == 0 {
                        out[i * 2..i * 2 + 2]
                            .copy_from_slice(&g_bytes[i * 2..i * 2 + 2]);
                    }
                }
            }
            DType::F16 => {
                for i in 0..n {
                    if m_bytes[i] == 0 {
                        out[i * 2..i * 2 + 2]
                            .copy_from_slice(&g_bytes[i * 2..i * 2 + 2]);
                    }
                }
            }
            _ => unreachable!(),
        }

        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        let d_x = Tensor::from_parts(
            storage,
            Layout::contiguous(grad_output.shape().to_vec()),
            TensorId::next(),
        )?;
        Ok(vec![Some(d_x), None /* mask non-differentiable */])
    }
    fn requires_input(&self, idx: usize) -> bool {
        match idx {
            0 => false, // x not needed
            1 => true,  // mask saved
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
    fn masked_fill_backward_zeros_masked_positions() {
        // mask = [0, 1, 0, 1]. grad = [10, 20, 30, 40].
        // d_x = [10, 0, 30, 0].
        let mask = Tensor::from_slice(&[0u8, 1, 0, 1], vec![4]).unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let bo = MaskedFillBackward { mask };
        let grads = bo.apply(&grad).unwrap();
        let d_x = grads[0].as_ref().unwrap();
        assert_eq!(read_f32(d_x), vec![10.0, 0.0, 30.0, 0.0]);
        assert!(grads[1].is_none());
    }

    #[test]
    fn masked_fill_backward_all_masked_is_zero() {
        let mask = Tensor::from_slice(&[1u8, 1, 1], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = MaskedFillBackward { mask };
        let d_x = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&d_x), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn masked_fill_backward_no_mask_is_identity() {
        let mask = Tensor::from_slice(&[0u8, 0, 0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = MaskedFillBackward { mask };
        let d_x = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(read_f32(&d_x), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn masked_fill_backward_2d() {
        // mask [[0, 1], [1, 0]] grad [[1, 2], [3, 4]] → [[1, 0], [0, 4]]
        let mask = Tensor::from_slice(&[0u8, 1, 1, 0], vec![2, 2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = MaskedFillBackward { mask };
        let d_x = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d_x.shape(), &[2, 2]);
        assert_eq!(read_f32(&d_x), vec![1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn masked_fill_backward_dtype_mismatch_errors() {
        // mask wrong dtype.
        let mask = Tensor::from_slice(&[1.0f32, 0.0], vec![2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let e = MaskedFillBackward { mask }.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("U8"));
    }

    #[test]
    fn masked_fill_backward_shape_mismatch_errors() {
        let mask = Tensor::from_slice(&[0u8, 1, 0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let e = MaskedFillBackward { mask }.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn op_metadata() {
        let mask = Tensor::from_slice(&[0u8], vec![1]).unwrap();
        let bo = MaskedFillBackward { mask };
        assert_eq!(bo.name(), "masked_fill_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(!bo.requires_input(0));
        assert!(bo.requires_input(1));
    }
}
