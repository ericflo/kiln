//! `chunk` and `split` — variadic axis-splitting.
//!
//! - `chunk(t, n_chunks, axis)`: split the axis into `n_chunks` pieces
//!   of as-even-as-possible size. The trailing chunk holds the
//!   remainder when the dim is not divisible by `n_chunks`.
//! - `split_with_sizes(t, &sizes, axis)`: split the axis into the
//!   exact section sizes given. Errors if the sizes do not sum to the
//!   axis length.
//!
//! Both return `Vec<Tensor>` of zero-copy `narrow` views over the
//! input — no kernel dispatch, no allocation, no copy.

use crate::{bail, Result, Tensor};

/// Split `axis` into `n_chunks` near-equal pieces, returning views.
///
/// If `axis_len % n_chunks != 0`, the first `axis_len % n_chunks`
/// chunks each get one extra element so the chunks sum to `axis_len`.
/// This matches PyTorch's `tensor.chunk(n, dim)` behavior on
/// non-divisible cases.
pub fn chunk(t: &Tensor, n_chunks: usize, axis: usize) -> Result<Vec<Tensor>> {
    if n_chunks == 0 {
        bail!("chunk: n_chunks must be > 0");
    }
    if axis >= t.rank() {
        bail!("chunk: axis {axis} out of bounds for rank {}", t.rank());
    }
    let axis_len = t.shape()[axis];
    if axis_len == 0 {
        bail!("chunk: cannot split a zero-length axis");
    }
    let base = axis_len / n_chunks;
    let extra = axis_len % n_chunks;
    let mut out = Vec::with_capacity(n_chunks);
    let mut off = 0usize;
    for i in 0..n_chunks {
        let len = base + if i < extra { 1 } else { 0 };
        if len == 0 {
            break;
        }
        out.push(t.narrow(axis, off, len)?);
        off += len;
    }
    Ok(out)
}

/// Split `axis` into the exact section lengths `sizes`. The sizes
/// must sum to `t.shape()[axis]`.
pub fn split_with_sizes(t: &Tensor, sizes: &[usize], axis: usize) -> Result<Vec<Tensor>> {
    if axis >= t.rank() {
        bail!(
            "split_with_sizes: axis {axis} out of bounds for rank {}",
            t.rank()
        );
    }
    let axis_len = t.shape()[axis];
    let sum: usize = sizes.iter().sum();
    if sum != axis_len {
        bail!(
            "split_with_sizes: sizes {sizes:?} sum to {sum} but axis {axis} has length {axis_len}"
        );
    }
    let mut out = Vec::with_capacity(sizes.len());
    let mut off = 0usize;
    for &len in sizes {
        out.push(t.narrow(axis, off, len)?);
        off += len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn chunk_even_divides() {
        // [6] split into 3 → three length-2 chunks.
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&data, vec![6]).unwrap();
        let out = chunk(&t, 3, 0).unwrap();
        assert_eq!(out.len(), 3);
        for c in &out {
            assert_eq!(c.shape(), &[2]);
        }
    }

    #[test]
    fn chunk_uneven_puts_extras_first() {
        // [7] split into 3 → 3, 2, 2 (extras go to leading chunks).
        let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&data, vec![7]).unwrap();
        let out = chunk(&t, 3, 0).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].shape(), &[3]);
        assert_eq!(out[1].shape(), &[2]);
        assert_eq!(out[2].shape(), &[2]);
    }

    #[test]
    fn chunk_more_than_axis_len_yields_axis_len_chunks() {
        // [2] split into 5 → only 2 chunks of length 1.
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let out = chunk(&t, 5, 0).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].shape(), &[1]);
        assert_eq!(out[1].shape(), &[1]);
    }

    #[test]
    fn chunk_along_inner_axis() {
        // [2, 4] split inner into 2 → two [2, 2] views.
        let t = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
        )
        .unwrap();
        let out = chunk(&t, 2, 1).unwrap();
        assert_eq!(out.len(), 2);
        for c in &out {
            assert_eq!(c.shape(), &[2, 2]);
        }
    }

    #[test]
    fn chunk_zero_n_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = chunk(&t, 0, 0).unwrap_err();
        assert!(e.to_string().contains("n_chunks"));
    }

    #[test]
    fn chunk_axis_oob_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = chunk(&t, 2, 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn split_with_sizes_exact() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&data, vec![10]).unwrap();
        let out = split_with_sizes(&t, &[3, 2, 5], 0).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].shape(), &[3]);
        assert_eq!(out[1].shape(), &[2]);
        assert_eq!(out[2].shape(), &[5]);
    }

    #[test]
    fn split_with_sizes_mismatch_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let e = split_with_sizes(&t, &[2, 3], 0).unwrap_err();
        assert!(e.to_string().contains("sum"));
    }

    #[test]
    fn split_with_sizes_axis_oob_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = split_with_sizes(&t, &[2], 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn split_qkv_shape_pattern() {
        // QKV projection pattern: [B*T, hidden_qkv] → q, k, v.
        // For Qwen3.5-4B Q: 2560, K: 256, V: 256 → hidden_qkv = 3072.
        let mut data = vec![0.0f32; 4 * 3072];
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f32;
        }
        let t = Tensor::from_slice(&data, vec![4, 3072]).unwrap();
        let parts = split_with_sizes(&t, &[2560, 256, 256], 1).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[4, 2560]);
        assert_eq!(parts[1].shape(), &[4, 256]);
        assert_eq!(parts[2].shape(), &[4, 256]);
    }
}
