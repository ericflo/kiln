//! `DropoutBackward` — gradient of inverted dropout.
//!
//! Forward (from `kiln_tensor::ops::dropout`):
//! ```text
//! mask_i = Bernoulli(1 - p)
//! y_i    = x_i * mask_i / (1 - p)
//! ```
//!
//! Backward:
//! ```text
//! d_x_i = d_y_i * mask_i / (1 - p)
//! ```
//!
//! Same scaling as forward, masked at the same positions. The mask
//! is saved on the BackwardOp at forward time.

use std::sync::Arc;

use kiln_tensor::{
    bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId,
};

use crate::BackwardOp;

#[derive(Debug)]
pub struct DropoutBackward {
    /// U8 mask from the forward pass.
    pub mask: Tensor,
    /// Drop probability used in the forward.
    pub p: f32,
}

impl BackwardOp for DropoutBackward {
    fn name(&self) -> &'static str {
        "dropout_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if self.mask.dtype() != DType::U8 {
            bail!(
                "DropoutBackward: mask dtype must be U8, got {}",
                self.mask.dtype()
            );
        }
        if self.mask.shape() != grad_output.shape() {
            bail!(
                "DropoutBackward: mask shape {:?} != grad shape {:?}",
                self.mask.shape(),
                grad_output.shape()
            );
        }
        if !(0.0..1.0).contains(&self.p) {
            bail!("DropoutBackward: p must be in [0, 1), got {}", self.p);
        }
        if !matches!(grad_output.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "DropoutBackward: grad dtype must be F32/BF16/F16, got {}",
                grad_output.dtype()
            );
        }
        if !grad_output.is_contiguous() || !self.mask.is_contiguous() {
            bail!("DropoutBackward: inputs must be contiguous");
        }
        let dtype = grad_output.dtype();
        let per = dtype.size_in_bytes();
        let n = grad_output.element_count();
        let g_cpu = grad_output
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("DropoutBackward: grad storage must be CpuStorage"))?;
        let m_cpu = self
            .mask
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("DropoutBackward: mask storage must be CpuStorage"))?;
        let g_bytes = g_cpu.as_bytes();
        let m_bytes = m_cpu.as_bytes();
        let inv_keep = if self.p == 0.0 { 1.0 } else { 1.0 / (1.0 - self.p) };
        let mut out = vec![0u8; n * per];
        match dtype {
            DType::F32 => {
                for i in 0..n {
                    if m_bytes[i] != 0 {
                        let g = f32::from_le_bytes(g_bytes[i * 4..i * 4 + 4].try_into().unwrap());
                        let v = g * inv_keep;
                        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    if m_bytes[i] != 0 {
                        let g = half::bf16::from_le_bytes(g_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                            .to_f32();
                        let v = g * inv_keep;
                        out[i * 2..i * 2 + 2]
                            .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                    }
                }
            }
            DType::F16 => {
                for i in 0..n {
                    if m_bytes[i] != 0 {
                        let g = half::f16::from_le_bytes(g_bytes[i * 2..i * 2 + 2].try_into().unwrap())
                            .to_f32();
                        let v = g * inv_keep;
                        out[i * 2..i * 2 + 2]
                            .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
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
        Ok(vec![Some(d_x)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::ops::dropout;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn dropout_backward_matches_forward_scale() {
        // Forward p=0.0 → mask all 1s, scale 1.0. Backward: d_x = d_y.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let (_, mask) = dropout(&x, 0.0, 1).unwrap();
        let dy = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let bo = DropoutBackward { mask, p: 0.0 };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        assert_eq!(dx, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn dropout_backward_zeros_dropped_positions() {
        // Forward p=0.99 with seed=1 → most positions dropped. Backward
        // must zero d_x at those positions and scale the rest by 100.
        let n = 50;
        let x = Tensor::from_slice(&vec![1.0f32; n], vec![n]).unwrap();
        let (_, mask) = dropout(&x, 0.99, 1).unwrap();
        let dy = Tensor::from_slice(&vec![1.0f32; n], vec![n]).unwrap();
        let bo = DropoutBackward {
            mask: mask.clone(),
            p: 0.99,
        };
        let dx = read_f32(bo.apply(&dy).unwrap()[0].as_ref().unwrap());
        let mask_cpu = mask
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes()
            .to_vec();
        for i in 0..n {
            if mask_cpu[i] == 1 {
                assert!((dx[i] - 100.0).abs() < 1e-3, "i={i}: dx={}", dx[i]);
            } else {
                assert_eq!(dx[i], 0.0, "i={i}: dx={}", dx[i]);
            }
        }
    }

    #[test]
    fn dropout_backward_2d_shape_preserved() {
        let mask = Tensor::from_slice(&[1u8, 0, 1, 0], vec![2, 2]).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bo = DropoutBackward { mask, p: 0.5 };
        let dx = bo.apply(&dy).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(dx.shape(), &[2, 2]);
        // p=0.5 → inv_keep=2. d_x = dy * mask * 2 = [2, 0, 6, 0]
        assert_eq!(read_f32(&dx), vec![2.0, 0.0, 6.0, 0.0]);
    }

    #[test]
    fn dropout_backward_shape_mismatch_errors() {
        let mask = Tensor::from_slice(&[1u8, 0, 1], vec![3]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = DropoutBackward { mask, p: 0.3 };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn dropout_backward_invalid_p_errors() {
        let mask = Tensor::from_slice(&[1u8], vec![1]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = DropoutBackward { mask, p: 1.5 };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("p must be"));
    }

    #[test]
    fn op_metadata() {
        let mask = Tensor::from_slice(&[1u8], vec![1]).unwrap();
        let bo = DropoutBackward { mask, p: 0.5 };
        assert_eq!(bo.name(), "dropout_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
