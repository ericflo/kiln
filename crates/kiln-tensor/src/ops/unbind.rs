//! `unbind` — split an axis into per-element views, dropping the axis.
//!
//! For shape `[..., D, ...]`, returns `D` tensors each of shape
//! `[..., ...]` (the axis is removed). Each view is a zero-copy
//! narrow + squeeze: stride math only, no kernel dispatch, no copy.
//!
//! Inverse of `stack` along the same axis. PyTorch parity:
//! `tensor.unbind(dim)`.

use crate::{bail, Result, Tensor};

/// Split `axis` into `axis_len` views, each with the axis removed.
pub fn unbind(t: &Tensor, axis: usize) -> Result<Vec<Tensor>> {
    if axis >= t.rank() {
        bail!(
            "unbind: axis {axis} out of bounds for rank {}",
            t.rank()
        );
    }
    let axis_len = t.shape()[axis];
    let mut out = Vec::with_capacity(axis_len);
    for i in 0..axis_len {
        let view = t.narrow(axis, i, 1)?;
        // narrow gives shape [..., 1, ...]; drop the size-1 axis.
        let mut shape: Vec<usize> = view.shape().to_vec();
        shape.remove(axis);
        out.push(view.reshape(shape)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn unbind_axis0_drops_outer() {
        // [3, 2] → three [2] views.
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let out = unbind(&t, 0).unwrap();
        assert_eq!(out.len(), 3);
        for v in &out {
            assert_eq!(v.shape(), &[2]);
        }
    }

    #[test]
    fn unbind_axis1_drops_inner() {
        // [2, 4] → four [2] views.
        let t = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        )
        .unwrap();
        let out = unbind(&t, 1).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert_eq!(v.shape(), &[2]);
        }
    }

    #[test]
    fn unbind_3d_middle_axis() {
        // [2, 3, 4] unbind dim=1 → three [2, 4] views.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&data, vec![2, 3, 4]).unwrap();
        let out = unbind(&t, 1).unwrap();
        assert_eq!(out.len(), 3);
        for v in &out {
            assert_eq!(v.shape(), &[2, 4]);
        }
    }

    #[test]
    fn unbind_singleton_axis_returns_one() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let out = unbind(&t, 0).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape(), &[3]);
    }

    #[test]
    fn unbind_axis_oob_errors() {
        let t = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = unbind(&t, 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn unbind_vector_yields_scalars() {
        // [4] unbind dim=0 → four scalars (shape []).
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let out = unbind(&t, 0).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert_eq!(v.shape() as &[usize], &[] as &[usize]);
        }
    }
}
