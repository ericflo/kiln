//! `MatmulBackward` — gradients for `c = a @ b`.
//!
//! # Math
//!
//! For `a` of shape `[..., M, K]` and `b` of shape `[..., K, N]`,
//! producing `c` of shape `[..., M, N]`, given upstream gradient
//! `dc` of shape `[..., M, N]`:
//!
//! - `da = dc @ b^T`   shape `[..., M, K]`
//! - `db = a^T @ dc`   shape `[..., K, N]`
//!
//! where `^T` is a transpose of the last two axes.
//!
//! # Implementation notes
//!
//! `da` and `db` both go through transposed-GEMM helpers, so backends with
//! resident support can consume the saved forward operands without allocating
//! physical trailing-axis transposes first.

use kiln_tensor::ops::{matmul_lhs_transposed, matmul_rhs_transposed};
use kiln_tensor::{Result, Tensor, bail};

use crate::BackwardOp;

#[derive(Debug)]
pub struct MatmulBackward {
    /// Saved `a` from the forward pass.
    pub a: Tensor,
    /// Saved `b` from the forward pass.
    pub b: Tensor,
}

impl BackwardOp for MatmulBackward {
    fn name(&self) -> &'static str {
        "matmul_backward"
    }
    fn input_count(&self) -> usize {
        2
    }
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let target_device = grad_output.device();
        let a = if self.a.device() == target_device {
            self.a.clone()
        } else {
            self.a.to_device(target_device)?
        };
        let b = if self.b.device() == target_device {
            self.b.clone()
        } else {
            self.b.to_device(target_device)?
        };

        let ar = a.rank();
        let br = b.rank();
        if ar < 2 || br < 2 {
            bail!("MatmulBackward: saved tensors must have rank ≥ 2");
        }
        if ar != br {
            bail!("MatmulBackward: rank mismatch between saved a ({ar}) and b ({br})");
        }
        if grad_output.rank() != ar {
            bail!(
                "MatmulBackward: grad_output rank {} != forward output rank {ar}",
                grad_output.rank()
            );
        }
        let da = matmul_rhs_transposed(grad_output, &b)?;
        let db = matmul_lhs_transposed(&a, grad_output)?;
        Ok(vec![Some(da), Some(db)])
    }
    fn requires_input(&self, _idx: usize) -> bool {
        true
    }
}

/// Input-only adjoint for `out = activation @ frozen_weight`.
///
/// LoRA training differentiates the activation but never the resident base
/// matrix. Recording that matrix as a tape input would run the much larger
/// `d_weight = activation^T @ grad_output` GEMM and retain a gradient the
/// optimizer cannot consume. This op saves the frozen right-hand side only as
/// backward data and emits `d_activation = grad_output @ frozen_weight^T`.
#[derive(Debug)]
pub struct FrozenRhsMatmulBackward {
    /// Saved frozen right-hand side from the forward pass, shape `[..., K, N]`.
    pub b: Tensor,
}

impl BackwardOp for FrozenRhsMatmulBackward {
    fn name(&self) -> &'static str {
        "frozen_rhs_matmul_backward"
    }

    fn input_count(&self) -> usize {
        1
    }

    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let target_device = grad_output.device();
        let b = if self.b.device() == target_device {
            self.b.clone()
        } else {
            self.b.to_device(target_device)?
        };
        if b.rank() < 2 {
            bail!("FrozenRhsMatmulBackward: saved weight must have rank >= 2");
        }
        if grad_output.rank() != b.rank() {
            bail!(
                "FrozenRhsMatmulBackward: grad_output rank {} != saved weight rank {}",
                grad_output.rank(),
                b.rank()
            );
        }
        Ok(vec![Some(matmul_rhs_transposed(grad_output, &b)?)])
    }

    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::CpuStorage;
    use kiln_tensor::ops::matmul;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn matmul_backward_2d() {
        // a [2, 3] @ b [3, 2] = c [2, 2]
        // a = [[1, 2, 3], [4, 5, 6]]
        // b = [[1, 2], [3, 4], [5, 6]]
        // c = [[22, 28], [49, 64]]
        // dc = [[1, 1], [1, 1]]
        // da = dc @ b^T = [[1,1],[1,1]] @ [[1,3,5],[2,4,6]]
        //    = [[3, 7, 11], [3, 7, 11]]
        // db = a^T @ dc = [[1,4],[2,5],[3,6]] @ [[1,1],[1,1]]
        //    = [[5, 5], [7, 7], [9, 9]]
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let dc = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();

        let bo = MatmulBackward { a, b };
        let grads = bo.apply(&dc).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        assert_eq!(da.shape(), &[2, 3]);
        assert_eq!(db.shape(), &[3, 2]);
        assert_eq!(read_f32(da), vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
        assert_eq!(read_f32(db), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn frozen_rhs_backward_matches_full_matmul_input_gradient_only() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let dc = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();

        let full = MatmulBackward { a, b: b.clone() }.apply(&dc).unwrap();
        let frozen = FrozenRhsMatmulBackward { b }.apply(&dc).unwrap();

        assert_eq!(frozen.len(), 1);
        assert_eq!(
            read_f32(frozen[0].as_ref().expect("activation gradient")),
            read_f32(full[0].as_ref().expect("full activation gradient"))
        );
    }

    #[test]
    fn matmul_backward_batched_3d() {
        // a [B=2, M=2, K=3], b [B=2, K=3, N=2], c [B=2, M=2, N=2]
        // Use the same per-batch values to keep math simple.
        let row = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_data: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let b_data: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        let dc_data: Vec<f32> = std::iter::repeat(1.0f32).take(8).collect();
        let a = Tensor::from_slice(&a_data, vec![2, 2, 3]).unwrap();
        let b = Tensor::from_slice(&b_data, vec![2, 3, 2]).unwrap();
        let dc = Tensor::from_slice(&dc_data, vec![2, 2, 2]).unwrap();

        let bo = MatmulBackward { a, b };
        let grads = bo.apply(&dc).unwrap();
        let da = grads[0].as_ref().unwrap();
        let db = grads[1].as_ref().unwrap();
        assert_eq!(da.shape(), &[2, 2, 3]);
        assert_eq!(db.shape(), &[2, 3, 2]);

        // Per-batch da should equal the 2D answer: [3, 7, 11, 3, 7, 11].
        let da_v = read_f32(da);
        let expected_da: Vec<f32> = std::iter::repeat([3.0f32, 7.0, 11.0, 3.0, 7.0, 11.0])
            .take(2)
            .flatten()
            .collect();
        assert_eq!(da_v, expected_da);

        // Per-batch db should equal: [5, 5, 7, 7, 9, 9].
        let db_v = read_f32(db);
        let expected_db: Vec<f32> = std::iter::repeat([5.0f32, 5.0, 7.0, 7.0, 9.0, 9.0])
            .take(2)
            .flatten()
            .collect();
        assert_eq!(db_v, expected_db);
    }

    #[test]
    fn matmul_backward_rejects_bad_grad_rank() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bad = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let bo = MatmulBackward { a, b };
        let e = bo.apply(&bad).unwrap_err();
        assert!(e.to_string().contains("grad_output rank"));
    }

    #[test]
    fn matmul_backward_finite_difference_parity() {
        // Numerical check: vary one entry of `a`, see if da matches the
        // partial derivative of the sum loss.
        let a = Tensor::from_slice(&[2.0f32, 3.0, 5.0, 7.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[11.0f32, 13.0, 17.0, 19.0], vec![2, 2]).unwrap();
        let c = matmul(&a, &b).unwrap();
        let dc = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let bo = MatmulBackward {
            a: a.clone(),
            b: b.clone(),
        };
        let grads = bo.apply(&dc).unwrap();
        let da = grads[0].as_ref().unwrap();
        let da_v = read_f32(da);

        // Compute finite difference at a[0,0]:
        // loss(a) = sum(a @ b). ∂loss/∂a[0,0] = sum_n b[0,n] = 11 + 13 = 24.
        // Generally ∂loss/∂a[i,k] = sum_n b[k, n].
        let _ = c; // suppress unused warning
        assert!((da_v[0] - 24.0).abs() < 1e-4);
        // ∂loss/∂a[0,1] = sum_n b[1,n] = 17 + 19 = 36
        assert!((da_v[1] - 36.0).abs() < 1e-4);
        // ∂loss/∂a[1,0] = sum_n b[0,n] = 24
        assert!((da_v[2] - 24.0).abs() < 1e-4);
        // ∂loss/∂a[1,1] = sum_n b[1,n] = 36
        assert!((da_v[3] - 36.0).abs() < 1e-4);
    }

    #[test]
    fn op_metadata() {
        let one = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let bo = MatmulBackward {
            a: one.clone(),
            b: one,
        };
        assert_eq!(bo.name(), "matmul_backward");
        assert_eq!(bo.input_count(), 2);
        assert!(bo.requires_input(0));
        assert!(bo.requires_input(1));

        let frozen = FrozenRhsMatmulBackward {
            b: Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![2, 2]).unwrap(),
        };
        assert_eq!(frozen.name(), "frozen_rhs_matmul_backward");
        assert_eq!(frozen.input_count(), 1);
        assert!(!frozen.requires_input(0));
    }
}
