//! `linear` — `y = x @ W + b` convenience op.
//!
//! Standard fully-connected layer. Supports optional bias.

use crate::ops::{add, broadcast_to, matmul};
use crate::{bail, Result, Tensor};

/// `y = x @ w + b` (b is optional).
///
/// `x: [..., in_dim]`, `w: [in_dim, out_dim]`, `b: [out_dim]`
/// (broadcast across the batch axes).
pub fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    if w.rank() != 2 {
        bail!("linear: w must be rank-2 [in, out], got {:?}", w.shape());
    }
    if x.rank() < 1 {
        bail!("linear: x must have rank ≥ 1");
    }
    let in_dim = w.shape()[0];
    let out_dim = w.shape()[1];
    let x_last = *x.shape().last().unwrap();
    if x_last != in_dim {
        bail!(
            "linear: x trailing axis {x_last} != w.in {in_dim}"
        );
    }
    // Reshape x to [N, in_dim] for matmul; reshape back to original
    // shape with last axis = out_dim.
    let mut out_shape = x.shape().to_vec();
    *out_shape.last_mut().unwrap() = out_dim;
    let leading: usize = x.shape()[..x.rank() - 1].iter().product::<usize>().max(1);
    let x_2d = x.reshape(vec![leading, in_dim])?;
    let y_2d = matmul(&x_2d, w)?;

    if let Some(bias) = b {
        if bias.rank() != 1 || bias.shape()[0] != out_dim {
            bail!(
                "linear: bias must be rank-1 [{out_dim}], got {:?}",
                bias.shape()
            );
        }
        // Broadcast bias [out_dim] → [leading, out_dim].
        let bias_2d = bias.reshape(vec![1, out_dim])?;
        let bias_b = broadcast_to(&bias_2d, &[leading, out_dim])?;
        let y_with_bias = add(&y_2d, &bias_b)?;
        return y_with_bias.reshape(out_shape);
    }
    y_2d.reshape(out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuStorage;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn linear_no_bias_matches_matmul() {
        // x = [[1, 2], [3, 4]]; w = [[1, 0], [0, 1]] (identity).
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn linear_with_bias() {
        // x @ w = [[1, 2], [3, 4]]; bias = [10, 20].
        // y = [[11, 22], [13, 24]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let y = linear(&x, &w, Some(&b)).unwrap();
        assert_eq!(read_f32(&y), vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn linear_3d_input() {
        // x: [B=2, S=2, in=2]; w: [2, 3] → y: [2, 2, 3].
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
        let w = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![2, 3],
        )
        .unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.shape(), &[2, 2, 3]);
    }

    #[test]
    fn linear_bias_shape_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = linear(&x, &w, Some(&b)).unwrap_err();
        assert!(e.to_string().contains("bias"));
    }

    #[test]
    fn linear_x_w_dim_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let e = linear(&x, &w, None).unwrap_err();
        assert!(e.to_string().contains("trailing axis"));
    }
}
