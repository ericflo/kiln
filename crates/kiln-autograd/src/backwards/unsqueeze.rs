//! `UnsqueezeBackward` — gradient of `unsqueeze(x, axis)`.
//!
//! Forward inserts a size-1 axis at position `axis`. The op is a
//! zero-copy reshape; nothing happens to the data, only the shape
//! grows by one dimension of size 1.
//!
//! Backward removes that size-1 axis (i.e. `squeeze(axis)`), which is
//! also a zero-copy reshape. The gradient values pass through
//! unchanged.
//!
//! ```text
//! d_x = grad_output.squeeze(axis)
//! ```
//!
//! No saved tensors are required — only the inserted-axis position.

use kiln_tensor::{bail, Result, Tensor};

use crate::BackwardOp;

#[derive(Debug)]
pub struct UnsqueezeBackward {
    /// Axis that was inserted by the forward `unsqueeze`.
    pub axis: usize,
}

impl BackwardOp for UnsqueezeBackward {
    fn name(&self) -> &'static str {
        "unsqueeze_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let shape = grad_output.shape();
        if self.axis >= shape.len() {
            bail!(
                "UnsqueezeBackward: axis {} out of range for grad rank {}",
                self.axis,
                shape.len()
            );
        }
        if shape[self.axis] != 1 {
            bail!(
                "UnsqueezeBackward: grad axis {} has size {}, expected 1",
                self.axis,
                shape[self.axis]
            );
        }
        let d_x = grad_output.squeeze(self.axis)?;
        Ok(vec![Some(d_x)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // We only need the inserted-axis position (saved on the struct).
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::CpuStorage;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn unsqueeze_backward_removes_axis_0() {
        // Forward: [3] → unsqueeze(0) → [1, 3]. Backward squeezes axis 0.
        let grad = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let bo = UnsqueezeBackward { axis: 0 };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(read_f32(&d), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unsqueeze_backward_removes_axis_1() {
        // Forward: [2, 3] → unsqueeze(1) → [2, 1, 3]. Backward squeezes axis 1.
        let grad = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 1, 3],
        )
        .unwrap();
        let bo = UnsqueezeBackward { axis: 1 };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[2, 3]);
        assert_eq!(read_f32(&d), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn unsqueeze_backward_trailing_axis() {
        // Forward: [4] → unsqueeze(1) → [4, 1]. Backward squeezes axis 1.
        let grad = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4, 1]).unwrap();
        let bo = UnsqueezeBackward { axis: 1 };
        let d = bo.apply(&grad).unwrap()[0].as_ref().unwrap().clone();
        assert_eq!(d.shape(), &[4]);
        assert_eq!(read_f32(&d), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn unsqueeze_backward_wrong_axis_size_errors() {
        let grad = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let bo = UnsqueezeBackward { axis: 0 };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("expected 1"));
    }

    #[test]
    fn unsqueeze_backward_axis_out_of_range_errors() {
        let grad = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bo = UnsqueezeBackward { axis: 5 };
        let e = bo.apply(&grad).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }

    #[test]
    fn op_metadata() {
        let bo = UnsqueezeBackward { axis: 1 };
        assert_eq!(bo.name(), "unsqueeze_backward");
        assert_eq!(bo.input_count(), 1);
        assert!(!bo.requires_input(0));
    }
}
