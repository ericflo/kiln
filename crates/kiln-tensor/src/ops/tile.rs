//! `tile` — repeat a tensor along multiple axes in one call.
//!
//! `tile(x, reps)` replicates `x` `reps[i]` times along axis `i` for
//! each provided multiplier. PyTorch / NumPy parity with
//! `torch.tile(x, dims)` / `np.tile(x, reps)`.
//!
//! Semantics:
//! - If `reps.len() < x.rank()`, the leading reps default to 1
//!   (effectively prepended).
//! - If `reps.len() > x.rank()`, `x` is left-padded with size-1 axes
//!   so it has rank `reps.len()`. This matches NumPy.
//!
//! Equivalent to chaining `repeat` over each axis; this op fuses the
//! per-axis pass into one allocation, which matters for large reps
//! lists where each intermediate would otherwise materialize.

use std::sync::Arc;

use crate::{bail, CpuStorage, Layout, Result, Storage, Tensor, TensorId};

pub fn tile(x: &Tensor, reps: &[usize]) -> Result<Tensor> {
    if reps.is_empty() {
        bail!("tile: reps must not be empty");
    }
    if reps.iter().any(|&r| r == 0) {
        bail!("tile: each rep must be > 0");
    }
    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("tile: packed dtype {dtype} not supported");
    }
    if !x.is_contiguous() {
        bail!("tile: input must be contiguous");
    }

    // Effective input shape: left-pad with 1s to reps.len() if needed.
    let in_shape: Vec<usize> = x.shape().to_vec();
    let (eff_in_shape, eff_reps) = if reps.len() >= in_shape.len() {
        let mut s = vec![1usize; reps.len() - in_shape.len()];
        s.extend(in_shape.iter().copied());
        (s, reps.to_vec())
    } else {
        // Pad reps with leading 1s.
        let mut r = vec![1usize; in_shape.len() - reps.len()];
        r.extend(reps.iter().copied());
        (in_shape.clone(), r)
    };

    let rank = eff_in_shape.len();
    let out_shape: Vec<usize> = eff_in_shape
        .iter()
        .zip(eff_reps.iter())
        .map(|(d, r)| d * r)
        .collect();
    let per = dtype.size_in_bytes();
    let n_out: usize = out_shape.iter().product();
    let mut out = vec![0u8; n_out * per];

    let cpu = x.storage();
    let cpu = cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("tile: storage must be CpuStorage"))?;
    let src = cpu.as_bytes();

    // Compute strides (in elements) for in_shape and out_shape.
    let mut in_strides = vec![1usize; rank];
    let mut out_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * eff_in_shape[d + 1];
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    // Walk output coordinates; for each, mod-down to input coord.
    let mut coord = vec![0usize; rank];
    for out_idx in 0..n_out {
        // Decode out_idx to coord.
        let mut rem = out_idx;
        for d in 0..rank {
            coord[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        // Map to input coord by modding each axis.
        let mut in_off = 0usize;
        for d in 0..rank {
            in_off += (coord[d] % eff_in_shape[d]) * in_strides[d];
        }
        let src_byte = in_off * per;
        let dst_byte = out_idx * per;
        out[dst_byte..dst_byte + per].copy_from_slice(&src[src_byte..src_byte + per]);
    }

    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
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
    fn tile_rank1_simple() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = tile(&x, &[2]).unwrap();
        assert_eq!(y.shape(), &[6]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn tile_rank2_both_axes() {
        // [[1,2],[3,4]] tiled (2, 2) → [[1,2,1,2],[3,4,3,4],[1,2,1,2],[3,4,3,4]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = tile(&x, &[2, 2]).unwrap();
        assert_eq!(y.shape(), &[4, 4]);
        assert_eq!(
            read_f32(&y),
            vec![
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0,
            ]
        );
    }

    #[test]
    fn tile_reps_shorter_than_rank_pads_leading_ones() {
        // x rank=2; reps=[3] → effectively [1, 3], so tile inner only.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = tile(&x, &[3]).unwrap();
        assert_eq!(y.shape(), &[2, 6]);
        // Each row repeats 3x.
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn tile_reps_longer_than_rank_pads_input_with_leading_ones() {
        // x = [3] (rank=1); reps=[2, 1] → input becomes [1, 3]; tile to [2, 3].
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = tile(&x, &[2, 1]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn tile_ones_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = tile(&x, &[1, 1]).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tile_zero_rep_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = tile(&x, &[0]).unwrap_err();
        assert!(e.to_string().contains("rep"));
    }

    #[test]
    fn tile_empty_reps_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = tile(&x, &[]).unwrap_err();
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn tile_matches_repeat_for_single_axis() {
        // tile(x, [n]) on rank-1 == repeat(x, 0, n).
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let tiled = tile(&x, &[3]).unwrap();
        let repeated = crate::ops::repeat(&x, 0, 3).unwrap();
        assert_eq!(read_f32(&tiled), read_f32(&repeated));
    }
}
