//! `meshgrid` — build coordinate grids from 1-D axis tensors.
//!
//! Given `N` 1-D tensors `[a, b, c, …]` of lengths `[La, Lb, Lc, …]`,
//! returns `N` tensors each of shape `[La, Lb, Lc, …]`:
//!
//! - `out[0][i,j,k,…] = a[i]`
//! - `out[1][i,j,k,…] = b[j]`
//! - `out[2][i,j,k,…] = c[k]`
//! - …
//!
//! Indexing matches PyTorch's `indexing='ij'` default — pure axis
//! iteration without the NumPy `xy`-swap. Useful for RoPE table
//! construction, position-encoding maps, 2-D mask construction.

use std::sync::Arc;

use crate::{bail, CpuStorage, Layout, Result, Storage, Tensor, TensorId};

pub fn meshgrid(axes: &[&Tensor]) -> Result<Vec<Tensor>> {
    if axes.is_empty() {
        bail!("meshgrid: at least one axis required");
    }
    let dtype = axes[0].dtype();
    if dtype.is_packed() {
        bail!("meshgrid: packed dtype {dtype} not supported");
    }
    for a in axes {
        if a.rank() != 1 {
            bail!(
                "meshgrid: every axis must be rank-1, got rank {}",
                a.rank()
            );
        }
        if a.dtype() != dtype {
            bail!(
                "meshgrid: dtype mismatch — first axis {} vs {}",
                dtype,
                a.dtype()
            );
        }
        if !a.is_contiguous() {
            bail!("meshgrid: axes must be contiguous");
        }
    }
    let per = dtype.size_in_bytes();
    let lens: Vec<usize> = axes.iter().map(|a| a.element_count()).collect();
    let total: usize = lens.iter().product();
    let n_axes = axes.len();

    // Compute output strides (in elements).
    let mut strides = vec![1usize; n_axes];
    for d in (0..n_axes.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * lens[d + 1];
    }

    let mut outputs: Vec<Tensor> = Vec::with_capacity(n_axes);
    for (axis_idx, axis) in axes.iter().enumerate() {
        let src = axis
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| crate::Error::from_str("meshgrid: storage must be CpuStorage"))?
            .as_bytes()
            .to_vec();
        let mut out = vec![0u8; total * per];
        for out_idx in 0..total {
            // Compute coord[axis_idx] for this output position.
            let c = (out_idx / strides[axis_idx]) % lens[axis_idx];
            let src_byte = c * per;
            let dst_byte = out_idx * per;
            out[dst_byte..dst_byte + per].copy_from_slice(&src[src_byte..src_byte + per]);
        }
        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        let shape: Vec<usize> = lens.clone();
        outputs.push(Tensor::from_parts(
            storage,
            Layout::contiguous(shape),
            TensorId::next(),
        )?);
    }
    Ok(outputs)
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
    fn meshgrid_2d_ij() {
        // a = [0, 1, 2], b = [10, 20]
        // out[0] (3, 2) = [[0,0],[1,1],[2,2]]
        // out[1] (3, 2) = [[10,20],[10,20],[10,20]]
        let a = Tensor::from_slice(&[0.0f32, 1.0, 2.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let outs = meshgrid(&[&a, &b]).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape(), &[3, 2]);
        assert_eq!(outs[1].shape(), &[3, 2]);
        assert_eq!(read_f32(&outs[0]), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        assert_eq!(read_f32(&outs[1]), vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn meshgrid_1d_is_identity() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let outs = meshgrid(&[&a]).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].shape(), &[3]);
        assert_eq!(read_f32(&outs[0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn meshgrid_3d() {
        // 2 x 1 x 2 grid: a=[0,1], b=[5], c=[10,20]
        let a = Tensor::from_slice(&[0.0f32, 1.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32], vec![1]).unwrap();
        let c = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let outs = meshgrid(&[&a, &b, &c]).unwrap();
        assert_eq!(outs.len(), 3);
        for o in &outs {
            assert_eq!(o.shape(), &[2, 1, 2]);
        }
        assert_eq!(read_f32(&outs[0]), vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(read_f32(&outs[1]), vec![5.0, 5.0, 5.0, 5.0]);
        assert_eq!(read_f32(&outs[2]), vec![10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn meshgrid_empty_axes_errors() {
        let e = meshgrid(&[]).unwrap_err();
        assert!(e.to_string().contains("at least one axis"));
    }

    #[test]
    fn meshgrid_rank2_axis_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = meshgrid(&[&a, &b]).unwrap_err();
        assert!(e.to_string().contains("rank-1"));
    }

    #[test]
    fn meshgrid_dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let b = Tensor::from_slice(&[half::bf16::from_f32(2.0)], vec![1]).unwrap();
        let e = meshgrid(&[&a, &b]).unwrap_err();
        assert!(e.to_string().contains("dtype mismatch"));
    }
}
