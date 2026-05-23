//! `cumprod` — cumulative product along an axis.
//!
//! ```text
//! y[..., i, ...] = Π_{j ≤ i} x[..., j, ...]
//! ```
//!
//! Same shape as input. PyTorch parity with `torch.cumprod(x,
//! dim)`. Used for normalizing-flow Jacobian accumulation, n-gram
//! probability composition, and gate-product attention masking.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn cumprod(x: &Tensor, axis: usize) -> Result<Tensor> {
    if axis >= x.rank() {
        bail!("cumprod: axis {axis} out of range for rank-{}", x.rank());
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("cumprod: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("cumprod: input must be contiguous");
    }
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let axis_dim = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
    let cpu = x
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::from_str("cumprod: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = outer * axis_dim * inner;
    let mut out = vec![0.0f32; n];

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 1.0f32;
            for a in 0..axis_dim {
                let idx = (o * axis_dim + a) * inner + i;
                let v = match dtype {
                    DType::F32 => f32::from_le_bytes(
                        bytes[idx * 4..idx * 4 + 4].try_into().unwrap(),
                    ),
                    DType::BF16 => half::bf16::from_le_bytes(
                        bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                    )
                    .to_f32(),
                    DType::F16 => half::f16::from_le_bytes(
                        bytes[idx * 2..idx * 2 + 2].try_into().unwrap(),
                    )
                    .to_f32(),
                    _ => unreachable!(),
                };
                acc *= v;
                out[idx] = acc;
            }
        }
    }
    let per = dtype.size_in_bytes();
    let mut out_bytes = vec![0u8; n * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in out.iter().enumerate() {
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in out.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in out.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let cpu_out = CpuStorage::from_bytes(dtype, out_bytes)?;
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
    fn cumprod_rank1() {
        let x = Tensor::from_slice(&[2.0f32, 3.0, 4.0], vec![3]).unwrap();
        let y = cumprod(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![2.0, 6.0, 24.0]);
    }

    #[test]
    fn cumprod_ones_is_ones() {
        let x = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let y = cumprod(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn cumprod_contains_zero_propagates() {
        let x = Tensor::from_slice(&[2.0f32, 0.0, 3.0, 4.0], vec![4]).unwrap();
        let y = cumprod(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn cumprod_2d_axis_0() {
        // [[2, 3], [4, 5]] axis=0 → [[2, 3], [8, 15]]
        let x = Tensor::from_slice(&[2.0f32, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let y = cumprod(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![2.0, 3.0, 8.0, 15.0]);
    }

    #[test]
    fn cumprod_2d_axis_1() {
        // [[2, 3, 4], [1, 2, 3]] axis=1 → [[2, 6, 24], [1, 2, 6]]
        let x = Tensor::from_slice(&[2.0f32, 3.0, 4.0, 1.0, 2.0, 3.0], vec![2, 3]).unwrap();
        let y = cumprod(&x, 1).unwrap();
        assert_eq!(read_f32(&y), vec![2.0, 6.0, 24.0, 1.0, 2.0, 6.0]);
    }

    #[test]
    fn cumprod_axis_oob_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = cumprod(&x, 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn cumprod_bf16() {
        let x = Tensor::from_slice(
            &[
                half::bf16::from_f32(2.0),
                half::bf16::from_f32(3.0),
                half::bf16::from_f32(4.0),
            ],
            vec![3],
        )
        .unwrap();
        let y = cumprod(&x, 0).unwrap();
        let cpu = y.storage();
        let cpu = cpu.as_any().downcast_ref::<CpuStorage>().unwrap();
        let vals: Vec<f32> = cpu
            .as_bytes()
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert!((vals[0] - 2.0).abs() < 1e-3);
        assert!((vals[1] - 6.0).abs() < 1e-3);
        assert!((vals[2] - 24.0).abs() < 1e-3);
    }
}
