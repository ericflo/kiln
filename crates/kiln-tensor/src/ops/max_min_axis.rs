//! `max_axis` / `min_axis` — reduce by max/min along one axis.
//!
//! Returns **values** (not indices). For indices use `top_k(t, 1)`
//! or `argmax_last_dim` (the latter only along the trailing axis).
//!
//! Output shape: input shape with `axis` removed.
//!
//! Used by:
//! - **Max-pooling** along an axis
//! - **Per-row max** for softmax stabilization (we also have an
//!   inline path in softmax_last_dim; this is the general op)
//! - **Reward shaping** in RL (max-return per trajectory)

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinMaxKind {
    Max,
    Min,
}

impl MinMaxKind {
    pub const fn name(self) -> &'static str {
        match self {
            MinMaxKind::Max => "max_axis",
            MinMaxKind::Min => "min_axis",
        }
    }
}

fn apply(kind: MinMaxKind, x: &Tensor, axis: usize) -> Result<Tensor> {
    if axis >= x.rank() {
        bail!(
            "{}: axis {axis} out of range for rank-{}",
            kind.name(),
            x.rank()
        );
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "{}: dtype must be F32/BF16/F16, got {}",
            kind.name(),
            x.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("{}: input must be contiguous", kind.name());
    }
    // CUDA fast path: if the storage is on CUDA, route through the
    // dedicated minmax reduction kernel in
    // `csrc/reduce_arbitrary_axis.cu` (issue #1082). The CPU branch
    // below still handles `CpuStorage`-backed tensors with identical
    // numerics.
    #[cfg(feature = "cuda")]
    {
        if x.storage().as_any().is::<crate::CudaStorage>() {
            return match kind {
                MinMaxKind::Min => crate::cuda_min_axis(x, axis),
                MinMaxKind::Max => crate::cuda_max_axis(x, axis),
            };
        }
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
        .ok_or_else(|| Error::from_str("max_min_axis: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();

    let mut out_vals = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc: Option<f32> = None;
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
                acc = Some(match (acc, kind) {
                    (None, _) => v,
                    (Some(cur), MinMaxKind::Max) => cur.max(v),
                    (Some(cur), MinMaxKind::Min) => cur.min(v),
                });
            }
            out_vals[o * inner + i] = acc.unwrap_or(0.0);
        }
    }
    // Write back to dtype.
    let per = dtype.size_in_bytes();
    let mut out_bytes = vec![0u8; out_vals.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in out_vals.iter().enumerate() {
                out_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for (i, &v) in out_vals.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in out_vals.iter().enumerate() {
                out_bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    let mut out_shape = shape;
    out_shape.remove(axis);
    let cpu_out = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(storage, Layout::contiguous(out_shape), TensorId::next())
}

pub fn max_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    apply(MinMaxKind::Max, x, axis)
}

pub fn min_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    apply(MinMaxKind::Min, x, axis)
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
    fn max_axis_rank2_trailing() {
        // [[1, 5, 3], [9, 2, 7]] max axis 1 → [5, 9]
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]).unwrap();
        let y = max_axis(&x, 1).unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(read_f32(&y), vec![5.0, 9.0]);
    }

    #[test]
    fn min_axis_rank2_trailing() {
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]).unwrap();
        let y = min_axis(&x, 1).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0]);
    }

    #[test]
    fn max_axis_outer() {
        // [[1, 5, 3], [9, 2, 7]] max axis 0 → [9, 5, 7]
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]).unwrap();
        let y = max_axis(&x, 0).unwrap();
        assert_eq!(y.shape(), &[3]);
        assert_eq!(read_f32(&y), vec![9.0, 5.0, 7.0]);
    }

    #[test]
    fn max_axis_rank3_middle() {
        // [B=2, M=2, D=2] max axis 1 → [B=2, D=2]
        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        )
        .unwrap();
        let y = max_axis(&x, 1).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        // batch 0: max([[1,2],[3,4]], axis=0) = [3, 4]
        // batch 1: max([[5,6],[7,8]], axis=0) = [7, 8]
        assert_eq!(read_f32(&y), vec![3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn max_axis_single_axis_size() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = max_axis(&x, 0).unwrap();
        assert_eq!(read_f32(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn max_axis_axis_out_of_range_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = max_axis(&x, 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn max_axis_bf16_round_trip() {
        let bf: Vec<half::bf16> = [1.0f32, 5.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&bf, vec![1, 3]).unwrap();
        let y = max_axis(&x, 1).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
    }
}
