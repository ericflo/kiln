//! `roll` — cyclic shift along an axis.
//!
//! `roll(x, shift, axis)` rotates elements along `axis` by `shift`
//! positions. Positive `shift` moves elements toward higher indices;
//! elements that fall off the end wrap to index 0. Negative `shift`
//! moves toward lower indices.
//!
//! PyTorch parity with `torch.roll(x, shifts, dims)`. For now
//! single-axis only; multi-axis composition is two `roll` calls.

use std::sync::Arc;

use crate::{bail, CpuStorage, Layout, Result, Storage, Tensor, TensorId};

pub fn roll(x: &Tensor, shift: i64, axis: usize) -> Result<Tensor> {
    if axis >= x.rank() {
        bail!("roll: axis {axis} out of bounds for rank {}", x.rank());
    }
    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("roll: packed dtype {dtype} not supported");
    }
    if !x.is_contiguous() {
        bail!("roll: input must be contiguous");
    }
    let shape: Vec<usize> = x.shape().to_vec();
    let axis_len = shape[axis];
    if axis_len == 0 {
        // Empty axis is identity.
        return x.reshape(shape);
    }
    // Normalize shift into [0, axis_len).
    let n = axis_len as i64;
    let s = ((shift % n) + n) % n;
    let s = s as usize;
    if s == 0 {
        // No-op; clone the storage view.
        return x.reshape(shape);
    }

    let per = dtype.size_in_bytes();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let cpu = x.storage();
    let cpu = cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("roll: storage must be CpuStorage"))?;
    let src = cpu.as_bytes();
    let mut out = vec![0u8; outer * axis_len * inner * per];

    // For each outer slab, for each axis-position i, write src[i] to
    // dst[(i + s) % axis_len].
    let row_bytes = inner * per;
    let slab_bytes = axis_len * row_bytes;
    for o in 0..outer {
        let slab_off = o * slab_bytes;
        for i in 0..axis_len {
            let j = (i + s) % axis_len;
            let src_off = slab_off + i * row_bytes;
            let dst_off = slab_off + j * row_bytes;
            out[dst_off..dst_off + row_bytes]
                .copy_from_slice(&src[src_off..src_off + row_bytes]);
        }
    }

    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(shape), TensorId::next())
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
    fn roll_positive_single_axis() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let y = roll(&x, 1, 0).unwrap();
        // [1,2,3,4,5] roll +1 → [5,1,2,3,4]
        assert_eq!(read_f32(&y), vec![5.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn roll_negative_shift() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let y = roll(&x, -1, 0).unwrap();
        // [1,2,3,4,5] roll -1 → [2,3,4,5,1]
        assert_eq!(read_f32(&y), vec![2.0, 3.0, 4.0, 5.0, 1.0]);
    }

    #[test]
    fn roll_full_axis_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = roll(&x, 4, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn roll_zero_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = roll(&x, 0, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn roll_wraps_large_shift() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        // shift=5 ≡ shift=1 (mod 4).
        let y1 = roll(&x, 5, 0).unwrap();
        let y2 = roll(&x, 1, 0).unwrap();
        assert_eq!(read_f32(&y1), read_f32(&y2));
    }

    #[test]
    fn roll_2d_axis_1() {
        // [[1,2,3],[4,5,6]] roll +1 axis=1 → [[3,1,2],[6,4,5]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = roll(&x, 1, 1).unwrap();
        assert_eq!(read_f32(&y), vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
    }

    #[test]
    fn roll_2d_axis_0() {
        // [[1,2],[3,4],[5,6]] roll +1 axis=0 → [[5,6],[1,2],[3,4]]
        let x =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = roll(&x, 1, 0).unwrap();
        assert_eq!(read_f32(&y), vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn roll_axis_oob_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = roll(&x, 0, 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }
}
