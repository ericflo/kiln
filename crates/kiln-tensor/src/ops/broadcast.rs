//! `broadcast_to` — materialize broadcasting of size-1 axes.
//!
//! Given an input shape and a target shape of the same rank, each
//! axis must either match or be the expansion of a size-1 input
//! axis to the target axis size. The output is a fresh contiguous
//! Tensor with the target shape; the broadcast axes are replicated.
//!
//! # Why a forward op
//!
//! kiln-tensor's per-op surface (matmul, mul, add, etc.) requires
//! **contiguous** inputs. Layout-level broadcasting via zero-stride
//! cannot pass through those op signatures. A materializing
//! `broadcast_to` lets backward ops compose cleanly via the public
//! op surface instead of dropping to byte-level loops.
//!
//! # Determinism
//!
//! `Constructive`. Fixed iteration order; the only operation per
//! element is a byte copy from the source slot.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Broadcast `x` to `target_shape`. Each axis of `target_shape`
/// must either equal the corresponding `x` axis or be a multiple
/// of a size-1 input axis.
pub fn broadcast_to(x: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let in_shape = x.shape();
    if in_shape.len() != target_shape.len() {
        bail!(
            "broadcast_to: rank mismatch: input {:?} vs target {:?}",
            in_shape,
            target_shape
        );
    }
    let mut broadcast_factor = vec![1usize; in_shape.len()];
    for (axis, (&in_d, &out_d)) in in_shape.iter().zip(target_shape.iter()).enumerate() {
        if in_d == out_d {
            broadcast_factor[axis] = 1;
        } else if in_d == 1 {
            broadcast_factor[axis] = out_d;
        } else {
            bail!(
                "broadcast_to: cannot broadcast axis {axis} from {in_d} to {out_d} (input shape {:?}, target {:?})",
                in_shape,
                target_shape
            );
        }
    }
    if !x.is_contiguous() {
        bail!("broadcast_to: input must be contiguous");
    }
    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("broadcast_to: packed dtype {dtype} not supported");
    }
    let per = dtype.size_in_bytes();
    let in_cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("broadcast_to: storage must be CpuStorage"))?;
    let in_bytes = in_cpu.as_bytes();

    let target_total: usize = target_shape.iter().product();
    let mut out_bytes = vec![0u8; target_total * per];
    // Walk every output index and copy the corresponding input element.
    // For each output index (i0, i1, …, i_{R-1}) compute the source
    // index (i0/bf0, i1/bf1, …) where bf_k = broadcast_factor[k].
    // The strides for the input are derived from `in_shape`.
    let rank = in_shape.len();
    let mut in_strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        in_strides[k] = in_strides[k + 1] * in_shape[k + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        out_strides[k] = out_strides[k + 1] * target_shape[k + 1];
    }
    for flat_out in 0..target_total {
        // Decompose flat_out into per-axis indices.
        let mut rem = flat_out;
        let mut in_offset = 0usize;
        for k in 0..rank {
            let idx_out = rem / out_strides[k];
            rem %= out_strides[k];
            let idx_in = if in_shape[k] == 1 { 0 } else { idx_out };
            in_offset += idx_in * in_strides[k];
        }
        let src = in_offset * per;
        let dst = flat_out * per;
        out_bytes[dst..dst + per].copy_from_slice(&in_bytes[src..src + per]);
    }
    let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(target_shape.to_vec()), TensorId::next())
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
    fn broadcast_rank1_size1_to_3() {
        let x = Tensor::from_slice(&[5.0f32], vec![1]).unwrap();
        let y = broadcast_to(&x, &[3]).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(read_f32(&y), vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn broadcast_rank2_one_axis_only() {
        // [[1, 2, 3]] [1, 3] → [4, 3] = [[1,2,3], [1,2,3], [1,2,3], [1,2,3]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = broadcast_to(&x, &[4, 3]).unwrap();
        assert_eq!(y.shape(), &[4, 3]);
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn broadcast_rank2_other_axis() {
        // [[1], [2]] [2, 1] → [2, 3] = [[1,1,1], [2,2,2]]
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2, 1]).unwrap();
        let y = broadcast_to(&x, &[2, 3]).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn broadcast_both_axes() {
        // [[5]] [1, 1] → [3, 4]: every element 5.0
        let x = Tensor::from_slice(&[5.0f32], vec![1, 1]).unwrap();
        let y = broadcast_to(&x, &[3, 4]).unwrap();
        assert_eq!(read_f32(&y), vec![5.0; 12]);
    }

    #[test]
    fn broadcast_identity_when_shapes_match() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = broadcast_to(&x, &[3]).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn broadcast_rejects_rank_mismatch() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = broadcast_to(&x, &[1, 3]).unwrap_err();
        assert!(e.to_string().contains("rank mismatch"));
    }

    #[test]
    fn broadcast_rejects_incompatible_axis() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = broadcast_to(&x, &[3]).unwrap_err();
        assert!(e.to_string().contains("cannot broadcast"));
    }

    #[test]
    fn broadcast_bf16_path() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![1, 2]).unwrap();
        let y = broadcast_to(&x, &[3, 2]).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape(), &[3, 2]);
    }
}
