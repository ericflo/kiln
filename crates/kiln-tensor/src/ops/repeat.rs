//! `repeat` — tile a tensor `n` times along an axis.
//!
//! ```text
//! repeat(x, axis, n).shape[axis] = x.shape[axis] * n
//! ```
//!
//! Different from `broadcast_to` (which only expands size-1 axes).
//! `repeat` tiles **arbitrary-size** axes; the output is
//! conceptually `concat([x, x, …, x] /* n copies */, axis)`.
//!
//! Used for:
//! - **Sequence broadcasting** — replicate a `[1, S, D]` tensor to
//!   `[B, S, D]` (could also use broadcast_to for size-1 axes; this
//!   is the general case)
//! - **Attention bias spreading**
//! - **MoE expert input duplication** when the same hidden state
//!   feeds multiple experts

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn repeat(x: &Tensor, axis: usize, n: usize) -> Result<Tensor> {
    if x.rank() == 0 {
        bail!("repeat: input must have rank ≥ 1");
    }
    if axis >= x.rank() {
        bail!(
            "repeat: axis {axis} out of range for rank-{} input",
            x.rank()
        );
    }
    if n == 0 {
        bail!("repeat: n must be > 0");
    }
    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("repeat: packed dtype {dtype} not supported");
    }
    if !x.is_contiguous() {
        bail!("repeat: input must be contiguous");
    }
    let per = dtype.size_in_bytes();
    let shape = x.shape();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let axis_in = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("repeat: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let axis_out = axis_in * n;
    let mut out = vec![0u8; outer * axis_out * inner * per];

    // For each outer slab, copy the axis-block `n` times into the
    // output's expanded axis dimension.
    for o in 0..outer {
        for rep in 0..n {
            let src_start = o * axis_in * inner * per;
            let src_end = src_start + axis_in * inner * per;
            let dst_start = (o * axis_out + rep * axis_in) * inner * per;
            let dst_end = dst_start + axis_in * inner * per;
            out[dst_start..dst_end].copy_from_slice(&bytes[src_start..src_end]);
        }
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] = axis_out;
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
    fn repeat_rank1_simple() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = repeat(&x, 0, 3).unwrap();
        assert_eq!(y.shape(), &[9]);
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn repeat_rank2_axis_0() {
        // [[1,2],[3,4]] repeated 2x along axis 0 → 4 rows.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = repeat(&x, 0, 2).unwrap();
        assert_eq!(y.shape(), &[4, 2]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_rank2_axis_1() {
        // [[1,2],[3,4]] repeated 2x along axis 1 → 4 cols per row.
        // Layout: out[r, :] = [a, b, a, b]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = repeat(&x, 1, 2).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_n_one_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = repeat(&x, 0, 1).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_rank3_middle_axis() {
        // [B=2, H=1, D=2] repeated 3x at axis 1 → [B=2, H=3, D=2].
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]).unwrap();
        let y = repeat(&x, 1, 3).unwrap();
        assert_eq!(y.shape(), &[2, 3, 2]);
        // Batch 0: [1, 2] x 3 = [1,2,1,2,1,2]
        // Batch 1: [3, 4] x 3 = [3,4,3,4,3,4]
        assert_eq!(
            read_f32(&y),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }

    #[test]
    fn repeat_n_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = repeat(&x, 0, 0).unwrap_err();
        assert!(e.to_string().contains("n must be > 0"));
    }

    #[test]
    fn repeat_axis_out_of_range_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = repeat(&x, 5, 2).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn repeat_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![2]).unwrap();
        let y = repeat(&x, 0, 3).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        assert_eq!(y.shape(), &[6]);
    }
}
