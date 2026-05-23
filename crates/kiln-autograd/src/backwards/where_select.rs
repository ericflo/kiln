//! `WhereSelectBackward` — gradient of `where_select(mask, t, f)`.
//!
//! Forward: `out[i] = if mask[i] != 0 { t[i] } else { f[i] }`.
//!
//! Backward routes the gradient:
//! ```text
//! d_t[i] = if mask[i] != 0 { grad_output[i] } else { 0 }
//! d_f[i] = if mask[i] != 0 { 0 } else { grad_output[i] }
//! d_mask = None (boolean, non-differentiable)
//! ```

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct WhereSelectBackward {
    /// U8 mask from the forward pass.
    pub mask: Tensor,
}

impl BackwardOp for WhereSelectBackward {
    fn name(&self) -> &'static str {
        "where_select_backward"
    }
    fn input_count(&self) -> usize {
        3
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if self.mask.dtype() != DType::U8 {
            bail!(
                "WhereSelectBackward: mask dtype must be U8, got {}",
                self.mask.dtype()
            );
        }
        if self.mask.shape() != grad_output.shape() {
            bail!(
                "WhereSelectBackward: mask shape {:?} != grad shape {:?}",
                self.mask.shape(),
                grad_output.shape()
            );
        }
        if !matches!(grad_output.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "WhereSelectBackward: grad dtype must be F32/BF16/F16, got {}",
                grad_output.dtype()
            );
        }
        if !grad_output.is_contiguous() || !self.mask.is_contiguous() {
            bail!("WhereSelectBackward: inputs must be contiguous");
        }
        let dtype = grad_output.dtype();
        let per = dtype.size_in_bytes();
        let n = grad_output.element_count();
        let g_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("WhereSelectBackward: grad storage must be CpuStorage"))?;
        let m_cpu = self
            .mask
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("WhereSelectBackward: mask storage must be CpuStorage"))?;
        let g_bytes = g_cpu.as_bytes();
        let m_bytes = m_cpu.as_bytes();

        let mut dt = vec![0u8; n * per];
        let mut df = vec![0u8; n * per];
        for i in 0..n {
            let g_slice = &g_bytes[i * per..(i + 1) * per];
            if m_bytes[i] != 0 {
                dt[i * per..(i + 1) * per].copy_from_slice(g_slice);
            } else {
                df[i * per..(i + 1) * per].copy_from_slice(g_slice);
            }
        }
        let shape = grad_output.shape().to_vec();
        let t_cpu = CpuStorage::from_bytes(dtype, dt)?;
        let f_cpu = CpuStorage::from_bytes(dtype, df)?;
        let t_storage: Storage = Arc::new(t_cpu);
        let f_storage: Storage = Arc::new(f_cpu);
        let d_t = Tensor::from_parts(t_storage, Layout::contiguous(shape.clone()), TensorId::next())?;
        let d_f = Tensor::from_parts(f_storage, Layout::contiguous(shape), TensorId::next())?;
        Ok(vec![None, Some(d_t), Some(d_f)])
    }
    fn requires_input(&self, idx: usize) -> bool {
        // input 0 = mask (saved on struct); 1 = t (not needed); 2 = f (not needed).
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
    fn where_select_backward_routes_grad() {
        // mask = [1, 0, 1, 0]; grad = [10, 20, 30, 40].
        // d_t = [10, 0, 30, 0]; d_f = [0, 20, 0, 40]; d_mask = None.
        let mask = Tensor::from_slice(&[1u8, 0, 1, 0], vec![4]).unwrap();
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let bo = WhereSelectBackward { mask };
        let grads = bo.apply(&grad).unwrap();
        assert!(grads[0].is_none());
        assert_eq!(
            read_f32(grads[1].as_ref().unwrap()),
            vec![10.0, 0.0, 30.0, 0.0]
        );
        assert_eq!(
            read_f32(grads[2].as_ref().unwrap()),
            vec![0.0, 20.0, 0.0, 40.0]
        );
    }

    #[test]
    fn where_select_backward_2d_shape_preserved() {
        let mask = Tensor::from_slice(&[1u8, 0, 0, 1], vec![2, 2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = WhereSelectBackward { mask };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 2]);
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[2, 2]);
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![1.0, 0.0, 0.0, 4.0]);
        assert_eq!(read_f32(grads[2].as_ref().unwrap()), vec![0.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn where_select_backward_all_t_path() {
        let mask = Tensor::from_slice(&[1u8, 1, 1], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = WhereSelectBackward { mask };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![1.0, 2.0, 3.0]);
        assert_eq!(read_f32(grads[2].as_ref().unwrap()), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn where_select_backward_all_f_path() {
        let mask = Tensor::from_slice(&[0u8, 0, 0], vec![3]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let bo = WhereSelectBackward { mask };
        let grads = bo.apply(&grad).unwrap();
        assert_eq!(read_f32(grads[1].as_ref().unwrap()), vec![0.0, 0.0, 0.0]);
        assert_eq!(read_f32(grads[2].as_ref().unwrap()), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn where_select_backward_mask_dtype_errors() {
        let mask = Tensor::from_slice(&[1.0f32, 0.0], vec![2]).unwrap();
        let grad = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let bo = WhereSelectBackward { mask };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("mask dtype"));
    }

    #[test]
    fn op_metadata() {
        let mask = Tensor::from_slice(&[0u8], vec![1]).unwrap();
        let bo = WhereSelectBackward { mask };
        assert_eq!(bo.name(), "where_select_backward");
        assert_eq!(bo.input_count(), 3);
        assert!(bo.requires_input(0));
        assert!(!bo.requires_input(1));
        assert!(!bo.requires_input(2));
    }
}
