//! `repeat_interleave` — interleaved repetition.
//!
//! Different from `repeat`:
//! - `repeat(x=[1,2,3], axis=0, n=3)` → `[1, 2, 3, 1, 2, 3, 1, 2, 3]`
//! - `repeat_interleave(x=[1,2,3], axis=0, n=3)` →
//!   `[1, 1, 1, 2, 2, 2, 3, 3, 3]`
//!
//! Used for GQA (grouped-query attention) head expansion and for
//! one-hot-like indexing tricks.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn repeat_interleave(x: &Tensor, axis: usize, n: usize) -> Result<Tensor> {
    if x.rank() == 0 {
        bail!("repeat_interleave: input must have rank ≥ 1");
    }
    if axis >= x.rank() {
        bail!(
            "repeat_interleave: axis {axis} out of range for rank-{}",
            x.rank()
        );
    }
    if n == 0 {
        bail!("repeat_interleave: n must be > 0");
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "repeat_interleave: dtype must be F32/BF16/F16, got {}",
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("repeat_interleave: input must be contiguous");
    }
    let dtype = x.dtype();
    let per = dtype.size_in_bytes();
    let shape = x.shape();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let axis_in = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let bytes = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("repeat_interleave: storage must be CpuStorage"))?
        .as_bytes();
    let axis_out = axis_in * n;
    let mut out = vec![0u8; outer * axis_out * inner * per];
    // For each outer slab, for each input axis index, copy `n` consecutive
    // copies of the inner block.
    for o in 0..outer {
        for a in 0..axis_in {
            let src_start = (o * axis_in + a) * inner * per;
            let src_end = src_start + inner * per;
            for k in 0..n {
                let dst_start = (o * axis_out + a * n + k) * inner * per;
                let dst_end = dst_start + inner * per;
                out[dst_start..dst_end].copy_from_slice(&bytes[src_start..src_end]);
            }
        }
    }
    let mut out_shape = shape.to_vec();
    out_shape[axis] = axis_out;
    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
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
    fn repeat_interleave_rank1() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = repeat_interleave(&x, 0, 3).unwrap();
        assert_eq!(y.shape(), &[9]);
        assert_eq!(
            read_f32(&y),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn repeat_interleave_differs_from_repeat() {
        use crate::ops::repeat;
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let r = repeat(&x, 0, 3).unwrap();
        let ri = repeat_interleave(&x, 0, 3).unwrap();
        assert_eq!(read_f32(&r), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(read_f32(&ri), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn repeat_interleave_rank2_axis_0() {
        // [[1, 2], [3, 4]] interleave 2x at axis 0 → [[1,2], [1,2], [3,4], [3,4]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = repeat_interleave(&x, 0, 2).unwrap();
        assert_eq!(y.shape(), &[4, 2]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn repeat_interleave_rank2_axis_1() {
        // [[1, 2], [3, 4]] interleave 2x at axis 1 → [[1, 1, 2, 2], [3, 3, 4, 4]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = repeat_interleave(&x, 1, 2).unwrap();
        assert_eq!(y.shape(), &[2, 4]);
        assert_eq!(read_f32(&y), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn repeat_interleave_n_one_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(read_f32(&repeat_interleave(&x, 0, 1).unwrap()), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_interleave_gqa_pattern() {
        // GQA: 2 KV heads, expand to 8 attention heads → factor of 4.
        // Shape [B=1, H_kv=2, S=4, D=2] → [B=1, H=8, S=4, D=2] via axis=1.
        let x = Tensor::from_slice(&[0.0f32; 16], vec![1, 2, 4, 2]).unwrap();
        let y = repeat_interleave(&x, 1, 4).unwrap();
        assert_eq!(y.shape(), &[1, 8, 4, 2]);
    }

    #[test]
    fn repeat_interleave_n_zero_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = repeat_interleave(&x, 0, 0).unwrap_err();
        assert!(e.to_string().contains("n must be > 0"));
    }
}
