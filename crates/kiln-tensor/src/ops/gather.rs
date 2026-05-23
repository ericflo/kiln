//! `gather` — general per-element index along an axis.
//!
//! Different from [`crate::ops::index_select`] (which picks whole
//! slabs by index) — `gather` reads one element per output position,
//! with the index tensor matching the output shape exactly.
//!
//! Given:
//! - `x: Tensor` — source data
//! - `indices: Tensor` — same rank as `x`, same shape except along
//!   `axis` (where it can be any positive length)
//! - `axis: usize` — axis to gather along
//!
//! Returns: a tensor with `indices.shape`; each `out[..., i, ...] =
//! x[..., indices[..., i, ...], ...]`. PyTorch parity with
//! `torch.gather(x, dim, index)`.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn read_indices(t: &Tensor) -> Result<Vec<i64>> {
    if !t.is_contiguous() {
        bail!("gather: indices must be contiguous");
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("gather: indices must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::I64 => {
            for i in 0..n {
                out.push(i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()));
            }
        }
        DType::U32 => {
            for i in 0..n {
                out.push(u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as i64);
            }
        }
        other => bail!("gather: indices dtype must be I64 or U32, got {other}"),
    }
    Ok(out)
}

pub fn gather(x: &Tensor, axis: usize, indices: &Tensor) -> Result<Tensor> {
    if axis >= x.rank() {
        bail!(
            "gather: axis {axis} out of bounds for rank {}",
            x.rank()
        );
    }
    if x.rank() != indices.rank() {
        bail!(
            "gather: rank mismatch — x rank {} vs indices rank {}",
            x.rank(),
            indices.rank()
        );
    }
    let x_shape: Vec<usize> = x.shape().to_vec();
    let i_shape: Vec<usize> = indices.shape().to_vec();
    for d in 0..x.rank() {
        if d == axis {
            continue;
        }
        if x_shape[d] != i_shape[d] {
            bail!(
                "gather: shape mismatch at axis {d} — x {x_shape:?} vs indices {i_shape:?} (axis {axis})"
            );
        }
    }
    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("gather: packed dtype {dtype} not supported");
    }
    if !x.is_contiguous() {
        bail!("gather: x must be contiguous");
    }

    let per = dtype.size_in_bytes();
    let x_axis_len = x_shape[axis];
    let idx_flat = read_indices(indices)?;

    let n_out: usize = i_shape.iter().product();
    let mut out = vec![0u8; n_out * per];

    let cpu = x.storage();
    let cpu = cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("gather: x must be CpuStorage"))?;
    let src = cpu.as_bytes();

    // Compute element-strides for x and indices/out (same shape as
    // indices).
    let rank = x.rank();
    let mut x_strides = vec![1usize; rank];
    let mut i_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        x_strides[d] = x_strides[d + 1] * x_shape[d + 1];
        i_strides[d] = i_strides[d + 1] * i_shape[d + 1];
    }

    let mut coord = vec![0usize; rank];
    for out_idx in 0..n_out {
        // Decode out_idx to coord in indices' shape.
        let mut rem = out_idx;
        for d in 0..rank {
            coord[d] = rem / i_strides[d];
            rem %= i_strides[d];
        }
        // Look up the index value at this position.
        let idx_val = idx_flat[out_idx];
        if idx_val < 0 || (idx_val as usize) >= x_axis_len {
            bail!(
                "gather: index {idx_val} out of bounds for axis {axis} of length {x_axis_len}"
            );
        }
        // Compute source offset in x: same coord but axis replaced.
        let mut src_off = 0usize;
        for d in 0..rank {
            let c = if d == axis { idx_val as usize } else { coord[d] };
            src_off += c * x_strides[d];
        }
        let src_byte = src_off * per;
        let dst_byte = out_idx * per;
        out[dst_byte..dst_byte + per].copy_from_slice(&src[src_byte..src_byte + per]);
    }

    let cpu_out = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(i_shape), TensorId::next())
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
    fn gather_1d() {
        // x = [10, 20, 30, 40, 50], idx = [4, 0, 2] → [50, 10, 30]
        let x =
            Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let idx = Tensor::from_slice(&[4i64, 0, 2], vec![3]).unwrap();
        let y = gather(&x, 0, &idx).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(read_f32(&y), vec![50.0, 10.0, 30.0]);
    }

    #[test]
    fn gather_2d_axis_1() {
        // x = [[1,2,3],[4,5,6]], idx = [[2,0,1],[1,1,0]]
        // y = [[3,1,2],[5,5,4]]
        let x =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let idx = Tensor::from_slice(&[2i64, 0, 1, 1, 1, 0], vec![2, 3]).unwrap();
        let y = gather(&x, 1, &idx).unwrap();
        assert_eq!(read_f32(&y), vec![3.0, 1.0, 2.0, 5.0, 5.0, 4.0]);
    }

    #[test]
    fn gather_2d_axis_0() {
        // x = [[1,2],[3,4],[5,6]], idx along axis 0 = [[2,0],[1,1]] (shape [2,2])
        // y[0,0] = x[2,0] = 5; y[0,1] = x[0,1] = 2; y[1,0] = x[1,0] = 3; y[1,1] = x[1,1] = 4
        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        )
        .unwrap();
        let idx = Tensor::from_slice(&[2i64, 0, 1, 1], vec![2, 2]).unwrap();
        let y = gather(&x, 0, &idx).unwrap();
        assert_eq!(read_f32(&y), vec![5.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gather_u32_indices() {
        let x = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let idx = Tensor::from_slice(&[2u32, 0, 1], vec![3]).unwrap();
        let y = gather(&x, 0, &idx).unwrap();
        assert_eq!(read_f32(&y), vec![30.0, 10.0, 20.0]);
    }

    #[test]
    fn gather_oob_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let idx = Tensor::from_slice(&[5i64], vec![1]).unwrap();
        let e = gather(&x, 0, &idx).unwrap_err();
        assert!(e.to_string().contains("out of bounds"));
    }

    #[test]
    fn gather_rank_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let idx = Tensor::from_slice(&[0i64, 1], vec![2]).unwrap();
        let e = gather(&x, 0, &idx).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn gather_shape_mismatch_errors() {
        // x [2, 3] but indices [3, 3] (mismatched axis 0).
        let x =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let idx = Tensor::from_slice(
            &[0i64, 0, 0, 1, 1, 1, 0, 0, 0],
            vec![3, 3],
        )
        .unwrap();
        let e = gather(&x, 1, &idx).unwrap_err();
        assert!(e.to_string().contains("shape mismatch"));
    }
}
