//! `cumsum` — cumulative sum along an axis.
//!
//! ```text
//! y[..., i, ...] = Σ_{j ≤ i} x[..., j, ...]
//! ```
//!
//! Same shape as input. Used by:
//! - **Sampling probability cumulation** (cumulative softmax,
//!   inverse-CDF sampling)
//! - **Sequence position encoding** based on prefix sums
//! - **Prefix-sum-based attention masking**

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn cumsum(x: &Tensor, axis: usize) -> Result<Tensor> {
    if axis >= x.rank() {
        bail!("cumsum: axis {axis} out of range for rank-{}", x.rank());
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("cumsum: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if !x.is_contiguous() {
        bail!("cumsum: input must be contiguous");
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
        .ok_or_else(|| Error::from_str("cumsum: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = outer * axis_dim * inner;
    let mut out = vec![0.0f32; n];

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
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
                acc += v;
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
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in out.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
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
    fn cumsum_rank1() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let y = cumsum(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn cumsum_rank2_trailing() {
        // [[1, 2, 3], [4, 5, 6]] cumsum axis 1 → [[1,3,6], [4,9,15]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = cumsum(&x, 1).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn cumsum_rank2_outer() {
        // [[1, 2, 3], [4, 5, 6]] cumsum axis 0 → [[1,2,3], [5,7,9]]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let y = cumsum(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn cumsum_single_element_unchanged() {
        let x = Tensor::from_slice(&[5.0f32], vec![1]).unwrap();
        let y = cumsum(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![5.0]);
    }

    #[test]
    fn cumsum_negative_values() {
        let x = Tensor::from_slice(&[1.0f32, -1.0, 2.0, -2.0], vec![4]).unwrap();
        let y = cumsum(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn cumsum_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![3]).unwrap();
        let y = cumsum(&x, 0).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }

    #[test]
    fn cumsum_axis_out_of_range_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = cumsum(&x, 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }
}
