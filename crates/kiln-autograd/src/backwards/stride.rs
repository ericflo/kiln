//! Stride/layout backward ops — `reshape` / `transpose` / `permute` /
//! `contiguous`. (#1082 Phase 6b: "reshape / transpose / permute / narrow —
//! stride math, no kernel". `narrow` already has [`crate::NarrowBackward`];
//! this file covers the other three plus `contiguous`.)
//!
//! All four are **linear, value-preserving** layout ops: the forward only
//! reorders / re-views elements, never combines them. Their adjoints are
//! therefore the *inverse layout op* applied to the upstream gradient — no
//! arithmetic, no kernel, fully device-agnostic (they go through the kt
//! `Tensor` layout methods, which run on CPU / CUDA / Vulkan / Metal alike).
//!
//! ```text
//! reshape(x, s)        -> d_x = reshape(d_y, x.shape)
//! transpose(x, a, b)   -> d_x = transpose(d_y, a, b)     (involution)
//! permute(x, axes)     -> d_x = permute(d_y, inverse(axes))
//! contiguous(x)        -> d_x = d_y                       (value identity)
//! ```

use kiln_tensor::{Result, Tensor, bail};

use crate::BackwardOp;

/// Adjoint of `y = x.reshape(new_shape)`. Reshape is a bijection on the
/// flat element order, so the gradient is just reshaped back to `x`'s shape.
#[derive(Debug)]
pub struct ReshapeBackward {
    /// Saved `x.shape` from the forward (target shape for `d_x`).
    pub input_shape: Vec<usize>,
}

impl BackwardOp for ReshapeBackward {
    fn name(&self) -> &'static str {
        "reshape_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let want: usize = self.input_shape.iter().product();
        let got: usize = grad_output.shape().iter().product();
        if want != got {
            bail!(
                "ReshapeBackward: grad has {got} elems, input_shape {:?} has {want}",
                self.input_shape
            );
        }
        // reshape requires a contiguous source; the upstream grad may be a
        // strided view (e.g. from a transpose adjoint), so materialise first.
        let dx = grad_output
            .contiguous()?
            .reshape(self.input_shape.clone())?;
        Ok(vec![Some(dx)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

/// Adjoint of `y = x.transpose(axis_a, axis_b)`. Transposing the same pair of
/// axes is an involution, so the adjoint re-applies the identical transpose.
#[derive(Debug)]
pub struct TransposeBackward {
    pub axis_a: usize,
    pub axis_b: usize,
}

impl BackwardOp for TransposeBackward {
    fn name(&self) -> &'static str {
        "transpose_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let rank = grad_output.rank();
        if self.axis_a >= rank || self.axis_b >= rank {
            bail!(
                "TransposeBackward: axes ({}, {}) out of bounds for rank {rank}",
                self.axis_a,
                self.axis_b
            );
        }
        // Materialise contiguous: the transpose adjoint is a strided view,
        // but the tape bridge and kernel-dispatching consumers (e.g.
        // GdnRecurrentBackward, the trainer GradStore copy) require
        // contiguous storage. Value-preserving; mirrors ReshapeBackward.
        Ok(vec![Some(
            grad_output
                .transpose(self.axis_a, self.axis_b)?
                .contiguous()?,
        )])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

/// Adjoint of `y = x.permute(axes)`. The gradient is permuted by the inverse
/// permutation: `inv[axes[i]] = i`.
#[derive(Debug)]
pub struct PermuteBackward {
    /// The `axes` passed to the forward `permute`.
    pub axes: Vec<usize>,
}

impl BackwardOp for PermuteBackward {
    fn name(&self) -> &'static str {
        "permute_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let n = self.axes.len();
        if grad_output.rank() != n {
            bail!(
                "PermuteBackward: grad rank {} != axes len {n}",
                grad_output.rank()
            );
        }
        // Validate `axes` is a genuine permutation of 0..n and build inverse.
        let mut inv = vec![usize::MAX; n];
        for (i, &a) in self.axes.iter().enumerate() {
            if a >= n || inv[a] != usize::MAX {
                bail!(
                    "PermuteBackward: axes {:?} is not a permutation of 0..{n}",
                    self.axes
                );
            }
            inv[a] = i;
        }
        Ok(vec![Some(grad_output.permute(&inv)?)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

/// Adjoint of `y = x.contiguous()`. `contiguous` is a value-identity copy
/// (only the memory layout changes), so the gradient passes through unchanged.
#[derive(Debug)]
pub struct ContiguousBackward;

impl BackwardOp for ContiguousBackward {
    fn name(&self) -> &'static str {
        "contiguous_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.clone())])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{CpuStorage, DType, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let c = t.contiguous().unwrap();
        c.storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .unwrap()
            .as_bytes()
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn arange(n: usize, shape: Vec<usize>) -> Tensor {
        let v: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        Tensor::from_slice(&v, shape).unwrap()
    }

    // ---- reshape ----
    #[test]
    fn reshape_backward_reshapes_grad_to_input_shape() {
        let dy = arange(24, vec![4, 6]);
        let bo = ReshapeBackward {
            input_shape: vec![2, 3, 4],
        };
        let dx = bo.apply(&dy).unwrap()[0].clone().unwrap();
        assert_eq!(dx.shape(), &[2, 3, 4]);
        // reshape preserves flat order, so values are identical.
        assert_eq!(read_f32(&dx), read_f32(&dy));
    }

    #[test]
    fn reshape_backward_elem_count_mismatch_errors() {
        let dy = arange(6, vec![6]);
        let bo = ReshapeBackward {
            input_shape: vec![2, 4],
        };
        assert!(bo.apply(&dy).unwrap_err().to_string().contains("elems"));
    }

    // ---- transpose ----
    #[test]
    fn transpose_backward_is_involution() {
        // forward y = x.transpose(0,1); x:[2,3] -> y:[3,2]
        let x = arange(6, vec![2, 3]);
        let y = x.transpose(0, 1).unwrap();
        let dy = arange(6, vec![3, 2]); // grad has y's shape
        let bo = TransposeBackward {
            axis_a: 0,
            axis_b: 1,
        };
        let dx = bo.apply(&dy).unwrap()[0].clone().unwrap();
        assert_eq!(dx.shape(), x.shape());
        // Re-applying the forward transpose to d_x must recover d_y.
        assert_eq!(read_f32(&dx.transpose(0, 1).unwrap()), read_f32(&dy));
        let _ = y;
    }

    #[test]
    fn transpose_backward_oob_errors() {
        let dy = arange(6, vec![3, 2]);
        let bo = TransposeBackward {
            axis_a: 0,
            axis_b: 5,
        };
        assert!(
            bo.apply(&dy)
                .unwrap_err()
                .to_string()
                .contains("out of bounds")
        );
    }

    // ---- permute ----
    #[test]
    fn permute_backward_applies_inverse_permutation() {
        // forward y = x.permute([2,0,1]); x:[2,3,4] -> y:[4,2,3]
        let axes = vec![2usize, 0, 1];
        let x = arange(24, vec![2, 3, 4]);
        let y = x.permute(&axes).unwrap();
        assert_eq!(y.shape(), &[4, 2, 3]);
        let dy = arange(24, vec![4, 2, 3]);
        let bo = PermuteBackward { axes: axes.clone() };
        let dx = bo.apply(&dy).unwrap()[0].clone().unwrap();
        assert_eq!(dx.shape(), &[2, 3, 4]);
        // Re-applying the forward permute to d_x must recover d_y.
        assert_eq!(read_f32(&dx.permute(&axes).unwrap()), read_f32(&dy));
    }

    #[test]
    fn permute_backward_rejects_non_permutation() {
        let dy = arange(24, vec![4, 2, 3]);
        let bo = PermuteBackward {
            axes: vec![0, 0, 1],
        };
        assert!(
            bo.apply(&dy)
                .unwrap_err()
                .to_string()
                .contains("permutation")
        );
    }

    // ---- contiguous ----
    #[test]
    fn contiguous_backward_is_identity() {
        let dy = arange(6, vec![2, 3]);
        let bo = ContiguousBackward;
        let dx = bo.apply(&dy).unwrap()[0].clone().unwrap();
        assert_eq!(dx.shape(), dy.shape());
        assert_eq!(read_f32(&dx), read_f32(&dy));
    }

    #[test]
    fn contiguous_backward_passes_strided_grad_through() {
        // A strided (transposed) grad should pass through unchanged in value.
        let base = arange(6, vec![2, 3]);
        let strided = base.transpose(0, 1).unwrap(); // [3,2] view
        let dx = ContiguousBackward.apply(&strided).unwrap()[0]
            .clone()
            .unwrap();
        assert_eq!(dx.shape(), &[3, 2]);
        assert_eq!(read_f32(&dx), read_f32(&strided));
    }

    #[test]
    fn op_metadata() {
        assert_eq!(
            (ReshapeBackward {
                input_shape: vec![1]
            })
            .name(),
            "reshape_backward"
        );
        assert_eq!(
            (TransposeBackward {
                axis_a: 0,
                axis_b: 1
            })
            .name(),
            "transpose_backward"
        );
        assert_eq!(
            (PermuteBackward { axes: vec![0] }).name(),
            "permute_backward"
        );
        assert_eq!(ContiguousBackward.name(), "contiguous_backward");
        assert_eq!(ContiguousBackward.input_count(), 1);
        assert!(!ContiguousBackward.requires_input(0));
        let _ = DType::F32;
    }
}
