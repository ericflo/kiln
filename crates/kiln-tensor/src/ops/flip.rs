//! `flip` — reverse the order of elements along given axes.
//!
//! `flip(x, axes)` reverses `x` along each axis in `axes`. PyTorch /
//! NumPy parity with `torch.flip(x, dims)` / `np.flip(x, axes)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, Layout, Result, Storage, Tensor, TensorId};

pub fn flip(x: &Tensor, axes: &[usize]) -> Result<Tensor> {
    if axes.is_empty() {
        // No-op clone of the shape; still allocate a fresh tensor id
        // for a clean parity contract.
        return x.reshape(x.shape().to_vec());
    }
    let rank = x.rank();
    for &a in axes {
        if a >= rank {
            bail!("flip: axis {a} out of bounds for rank {rank}");
        }
    }
    let mut seen = vec![false; rank];
    for &a in axes {
        if seen[a] {
            bail!("flip: duplicate axis {a} in {axes:?}");
        }
        seen[a] = true;
    }

    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("flip: packed dtype {dtype} not supported");
    }
    if !x.is_contiguous() {
        bail!("flip: input must be contiguous");
    }

    let shape: Vec<usize> = x.shape().to_vec();
    let per = dtype.size_in_bytes();
    let n_elem: usize = shape.iter().product();
    let mut out = vec![0u8; n_elem * per];

    // Compute output strides (== input strides since shape unchanged).
    let mut strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    let cpu = x.storage();
    let cpu = cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("flip: storage must be CpuStorage"))?;
    let src = cpu.as_bytes();

    // For each linear input index, decode to coord, flip selected
    // axes, re-encode, copy one element.
    let mut coord = vec![0usize; rank];
    for in_idx in 0..n_elem {
        let mut rem = in_idx;
        for d in 0..rank {
            coord[d] = rem / strides[d];
            rem %= strides[d];
        }
        for &a in axes {
            coord[a] = shape[a] - 1 - coord[a];
        }
        let mut out_off = 0usize;
        for d in 0..rank {
            out_off += coord[d] * strides[d];
        }
        let src_byte = in_idx * per;
        let dst_byte = out_off * per;
        out[dst_byte..dst_byte + per].copy_from_slice(&src[src_byte..src_byte + per]);
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
    fn flip_rank1() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(read_f32(&y), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn flip_axis_0_2d() {
        // [[1,2],[3,4],[5,6]] flip axis=0 → [[5,6],[3,4],[1,2]]
        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        )
        .unwrap();
        let y = flip(&x, &[0]).unwrap();
        assert_eq!(read_f32(&y), vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn flip_axis_1_2d() {
        // [[1,2,3],[4,5,6]] flip axis=1 → [[3,2,1],[6,5,4]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = flip(&x, &[1]).unwrap();
        assert_eq!(read_f32(&y), vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn flip_both_axes_2d() {
        // [[1,2],[3,4]] flip both → [[4,3],[2,1]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = flip(&x, &[0, 1]).unwrap();
        assert_eq!(read_f32(&y), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn flip_empty_axes_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = flip(&x, &[]).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn flip_axis_oob_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = flip(&x, &[5]).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn flip_duplicate_axis_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = flip(&x, &[0, 0]).unwrap_err();
        assert!(e.to_string().contains("duplicate"));
    }

    #[test]
    fn flip_twice_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let once = flip(&x, &[0]).unwrap();
        let twice = flip(&once, &[0]).unwrap();
        assert_eq!(read_f32(&twice), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn flip_3d_middle_axis() {
        // shape [2, 3, 2], flip axis=1.
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x = Tensor::from_slice(&data, vec![2, 3, 2]).unwrap();
        let y = flip(&x, &[1]).unwrap();
        let expected: Vec<f32> = vec![
            4.0, 5.0, 2.0, 3.0, 0.0, 1.0, // first slab axis1 reversed
            10.0, 11.0, 8.0, 9.0, 6.0, 7.0, // second slab axis1 reversed
        ];
        assert_eq!(read_f32(&y), expected);
    }
}
