//! `pad` — constant-value padding along chosen axes.
//!
//! Given a tensor of rank `R` and per-axis `(before, after)` pad
//! widths, returns a new tensor with each axis expanded by
//! `before[i] + after[i]`. Pad regions are filled with `pad_value`.
//!
//! PyTorch parity with `torch.nn.functional.pad(x, pad,
//! mode='constant', value=...)` — note that PyTorch's `pad` arg is
//! given in reverse axis order; here we take `(before, after)`
//! tuples in natural axis order for clarity.
//!
//! Used for: causal-attention chunk alignment, prefill block-size
//! rounding, convolution prep.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

fn fill_bytes(dtype: DType, value: f32) -> Vec<u8> {
    match dtype {
        DType::F32 => value.to_le_bytes().to_vec(),
        DType::BF16 => half::bf16::from_f32(value).to_le_bytes().to_vec(),
        DType::F16 => half::f16::from_f32(value).to_le_bytes().to_vec(),
        DType::U8 => vec![value as u8],
        DType::U32 => (value as u32).to_le_bytes().to_vec(),
        DType::I64 => (value as i64).to_le_bytes().to_vec(),
        _ => unreachable!(),
    }
}

pub fn pad(x: &Tensor, widths: &[(usize, usize)], pad_value: f32) -> Result<Tensor> {
    if widths.len() != x.rank() {
        bail!(
            "pad: widths length {} != input rank {}",
            widths.len(),
            x.rank()
        );
    }
    let dtype = x.dtype();
    if dtype.is_packed() {
        bail!("pad: packed dtype {dtype} not supported");
    }
    if !x.is_contiguous() {
        bail!("pad: input must be contiguous");
    }

    let in_shape: Vec<usize> = x.shape().to_vec();
    let out_shape: Vec<usize> = in_shape
        .iter()
        .zip(widths.iter())
        .map(|(d, (b, a))| d + b + a)
        .collect();
    let rank = in_shape.len();
    let per = dtype.size_in_bytes();
    let n_out: usize = out_shape.iter().product();

    // Initialize all output bytes to pad_value.
    let fill = fill_bytes(dtype, pad_value);
    let mut out = vec![0u8; n_out * per];
    for i in 0..n_out {
        let off = i * per;
        out[off..off + per].copy_from_slice(&fill);
    }

    // Compute strides.
    let mut in_strides = vec![1usize; rank];
    let mut out_strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    }

    let cpu = x.storage();
    let cpu = cpu
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("pad: storage must be CpuStorage"))?;
    let src = cpu.as_bytes();
    let n_in: usize = in_shape.iter().product();

    // Walk input coordinates; place each at the offset shifted by
    // the (before) pad widths.
    let mut coord = vec![0usize; rank];
    for in_idx in 0..n_in {
        let mut rem = in_idx;
        for d in 0..rank {
            coord[d] = rem / in_strides[d];
            rem %= in_strides[d];
        }
        let mut out_off = 0usize;
        for d in 0..rank {
            out_off += (coord[d] + widths[d].0) * out_strides[d];
        }
        let src_byte = in_idx * per;
        let dst_byte = out_off * per;
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
    fn pad_rank1_zeros() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = pad(&x, &[(1, 2)], 0.0).unwrap();
        assert_eq!(y.shape(), &[6]);
        assert_eq!(read_f32(&y), vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn pad_rank1_custom_value() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let y = pad(&x, &[(1, 1)], -9.0).unwrap();
        assert_eq!(read_f32(&y), vec![-9.0, 1.0, 2.0, -9.0]);
    }

    #[test]
    fn pad_rank2_both_axes() {
        // [[1, 2], [3, 4]] pad ((1, 0), (0, 1)) with 0
        // shape becomes [3, 3]:
        // [[0, 0, 0],
        //  [1, 2, 0],
        //  [3, 4, 0]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = pad(&x, &[(1, 0), (0, 1)], 0.0).unwrap();
        assert_eq!(y.shape(), &[3, 3]);
        assert_eq!(
            read_f32(&y),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
        );
    }

    #[test]
    fn pad_zero_width_is_identity() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let y = pad(&x, &[(0, 0), (0, 0)], 0.0).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pad_rank3() {
        // [[[1, 2]]] pad ((0, 1), (0, 0), (1, 1)) → shape [2, 1, 4]
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 1, 2]).unwrap();
        let y = pad(&x, &[(0, 1), (0, 0), (1, 1)], 7.0).unwrap();
        assert_eq!(y.shape(), &[2, 1, 4]);
        assert_eq!(
            read_f32(&y),
            vec![7.0, 1.0, 2.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        );
    }

    #[test]
    fn pad_widths_length_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = pad(&x, &[(1, 1), (2, 2)], 0.0).unwrap_err();
        assert!(e.to_string().contains("widths length"));
    }

    #[test]
    fn pad_bf16() {
        let x = Tensor::from_slice(
            &[half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)],
            vec![2],
        )
        .unwrap();
        let y = pad(&x, &[(1, 1)], 5.0).unwrap();
        let cpu = y.storage();
        let cpu = cpu.as_any().downcast_ref::<CpuStorage>().unwrap();
        let vals: Vec<f32> = cpu
            .as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 5.0).abs() < 1e-3);
        assert!((vals[1] - 1.0).abs() < 1e-3);
        assert!((vals[2] - 2.0).abs() < 1e-3);
        assert!((vals[3] - 5.0).abs() < 1e-3);
    }
}
