//! Backwards for the four elementwise binary ops:
//! `add`, `sub`, `mul`, `div`.
//!
//! All four are broadcastable, but Phase 1.x's kiln-tensor ops require
//! exact-shape inputs (no implicit broadcast). The backward shapes
//! therefore match the forward input shapes exactly.
//!
//! # Gradients
//!
//! Given `c = a OP b` and `dc = ∂L/∂c`:
//!
//! | OP  | da              | db              |
//! |-----|-----------------|-----------------|
//! | add | dc              | dc              |
//! | sub | dc              | -dc             |
//! | mul | dc * b          | dc * a          |
//! | div | dc / b          | -dc * a / b^2   |

use std::sync::Arc;

use kiln_tensor::ops::{div, mul, sub};
use kiln_tensor::{CpuStorage, Layout, Result, Storage, Tensor, TensorId};

use crate::BackwardOp;

// ----------------------------------------------------------------------
// AddBackward — `c = a + b` → da = dc, db = dc.
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct AddBackward;

impl BackwardOp for AddBackward {
    fn name(&self) -> &'static str {
        "add_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.clone()), Some(grad_output.clone())])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // add's backward needs nothing about its inputs.
        false
    }
}

// ----------------------------------------------------------------------
// SubBackward — `c = a - b` → da = dc, db = -dc.
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct SubBackward;

impl BackwardOp for SubBackward {
    fn name(&self) -> &'static str {
        "sub_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // db = 0 - dc.
        let zero = zeros_like(grad_output)?;
        let neg = sub(&zero, grad_output)?;
        Ok(vec![Some(grad_output.clone()), Some(neg)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

// ----------------------------------------------------------------------
// MulBackward — `c = a * b` → da = dc * b, db = dc * a.
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct MulBackward {
    /// Saved `a` from forward.
    pub a: Tensor,
    /// Saved `b` from forward.
    pub b: Tensor,
}

impl BackwardOp for MulBackward {
    fn name(&self) -> &'static str {
        "mul_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let da = mul(grad_output, &self.b)?;
        let db = mul(grad_output, &self.a)?;
        Ok(vec![Some(da), Some(db)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// DivBackward — `c = a / b` → da = dc / b, db = -dc * a / b^2.
// ----------------------------------------------------------------------

#[derive(Debug)]
pub struct DivBackward {
    pub a: Tensor,
    pub b: Tensor,
}

impl BackwardOp for DivBackward {
    fn name(&self) -> &'static str {
        "div_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // da = dc / b.
        let da = div(grad_output, &self.b)?;
        // db = -dc * a / (b * b).
        let bb = mul(&self.b, &self.b)?;
        let dc_a = mul(grad_output, &self.a)?;
        let neg_dc_a = sub(&zeros_like(&dc_a)?, &dc_a)?;
        let db = div(&neg_dc_a, &bb)?;
        Ok(vec![Some(da), Some(db)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn zeros_like(t: &Tensor) -> Result<Tensor> {
    let n_bytes = t.element_count() * t.dtype().size_in_bytes();
    let bytes = vec![0u8; n_bytes];
    let cpu = CpuStorage::from_bytes(t.dtype(), bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(t.shape().to_vec()), TensorId::next())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::ops::add;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn add_backward_passes_through() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], vec![3]).unwrap();
        let _c = add(&a, &b).unwrap();
        let dc = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let grads = AddBackward.apply(&dc).unwrap();
        assert_eq!(grads.len(), 2);
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        assert_eq!(read_f32(da), vec![1.0, 1.0, 1.0]);
        assert_eq!(read_f32(db), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn sub_backward_negates_second() {
        let dc = Tensor::from_slice(&[2.0f32, -3.0, 0.5], vec![3]).unwrap();
        let grads = SubBackward.apply(&dc).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        assert_eq!(read_f32(da), vec![2.0, -3.0, 0.5]);
        assert_eq!(read_f32(db), vec![-2.0, 3.0, -0.5]);
    }

    #[test]
    fn mul_backward_product_rule() {
        let a = Tensor::from_slice(&[2.0f32, 3.0, 4.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0], vec![3]).unwrap();
        let dc = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let bo = MulBackward {
            a: a.clone(),
            b: b.clone(),
        };
        let grads = bo.apply(&dc).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        // da = dc * b = b
        assert_eq!(read_f32(da), vec![5.0, 6.0, 7.0]);
        // db = dc * a = a
        assert_eq!(read_f32(db), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn div_backward_quotient_rule() {
        let a = Tensor::from_slice(&[6.0f32, 12.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 4.0], vec![2]).unwrap();
        let dc = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let bo = DivBackward {
            a: a.clone(),
            b: b.clone(),
        };
        let grads = bo.apply(&dc).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        // da = 1/b = [0.5, 0.25]
        let da_v = read_f32(da);
        assert!((da_v[0] - 0.5).abs() < 1e-6);
        assert!((da_v[1] - 0.25).abs() < 1e-6);
        // db = -a/b^2 = [-6/4, -12/16] = [-1.5, -0.75]
        let db_v = read_f32(db);
        assert!((db_v[0] - (-1.5)).abs() < 1e-6);
        assert!((db_v[1] - (-0.75)).abs() < 1e-6);
    }

    #[test]
    fn op_metadata() {
        assert_eq!(AddBackward.name(), "add_backward");
        assert_eq!(AddBackward.input_count(), 2);
        assert!(!AddBackward.requires_input(0));

        assert_eq!(SubBackward.name(), "sub_backward");
        assert_eq!(SubBackward.input_count(), 2);

        let one = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let m = MulBackward { a: one.clone(), b: one.clone() };
        assert_eq!(m.name(), "mul_backward");
        assert_eq!(m.input_count(), 2);
        assert!(m.requires_input(0));

        let d = DivBackward { a: one.clone(), b: one };
        assert_eq!(d.name(), "div_backward");
        assert_eq!(d.input_count(), 2);
    }
}
