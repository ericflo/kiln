//! `expand_dims` — insert size-1 axes at multiple positions in one call.
//!
//! `Tensor::unsqueeze(axis)` handles a single position; `expand_dims`
//! handles a sorted list of positions in one batched call.

use crate::{bail, Result, Tensor};

pub fn expand_dims(t: &Tensor, axes: &[usize]) -> Result<Tensor> {
    let mut sorted: Vec<usize> = axes.to_vec();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != axes.len() {
        bail!("expand_dims: duplicate axes in {axes:?}");
    }
    let new_rank = t.rank() + sorted.len();
    for &a in &sorted {
        if a >= new_rank {
            bail!(
                "expand_dims: axis {a} >= new rank {new_rank} (input rank {}, n_new_axes {})",
                t.rank(),
                sorted.len()
            );
        }
    }
    let mut new_shape: Vec<usize> = t.shape().to_vec();
    for &a in &sorted {
        new_shape.insert(a, 1);
    }
    t.reshape(new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Tensor};

    #[test]
    fn expand_dims_single() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = expand_dims(&t, &[0]).unwrap();
        assert_eq!(e.shape(), &[1, 3]);
    }

    #[test]
    fn expand_dims_multiple() {
        // [3] → expand at 0 and 2 → [1, 3, 1]
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = expand_dims(&t, &[0, 2]).unwrap();
        assert_eq!(e.shape(), &[1, 3, 1]);
    }

    #[test]
    fn expand_dims_at_end() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = expand_dims(&t, &[1]).unwrap();
        assert_eq!(e.shape(), &[2, 1]);
    }

    #[test]
    fn expand_dims_duplicate_errors() {
        let t = Tensor::zeros_cpu(vec![3], DType::F32);
        let e = expand_dims(&t, &[0, 0]).unwrap_err();
        assert!(e.to_string().contains("duplicate"));
    }

    #[test]
    fn expand_dims_out_of_range_errors() {
        let t = Tensor::zeros_cpu(vec![3], DType::F32);
        let e = expand_dims(&t, &[5]).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }
}
