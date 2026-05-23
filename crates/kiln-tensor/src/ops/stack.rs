//! `stack` — join N same-shape tensors into a new axis.
//!
//! ```text
//! stack([t1, t2, …, tN], axis) — new axis at position `axis` of size N.
//! ```
//!
//! Different from `concat` (which joins along an existing axis) —
//! `stack` adds a new axis. All inputs must share **exact** shape
//! and dtype.
//!
//! # Use cases
//!
//! - **Batching** — stack per-sample tensors into a batch
//! - **Multi-head attention** — stack per-head Q/K/V projections
//! - **Beam search bookkeeping** — stack candidate logits per beam
//!
//! # Output shape
//!
//! `inputs[0].shape` with axis inserted at position `axis` of size
//! `N = inputs.len()`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise; bit-identical at the same input dtype.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

pub fn stack(inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
    if inputs.is_empty() {
        bail!("stack: at least one input required");
    }
    let rank = inputs[0].rank();
    if axis > rank {
        bail!(
            "stack: axis {axis} > input rank {rank} (axis can be 0..=rank inclusive)"
        );
    }
    let dtype = inputs[0].dtype();
    if dtype.is_packed() {
        bail!("stack: packed dtype {dtype} not supported");
    }
    let shape = inputs[0].shape().to_vec();
    for (i, t) in inputs.iter().enumerate() {
        if t.shape() != shape.as_slice() {
            bail!(
                "stack: input {i} shape {:?} != input 0 shape {:?}",
                t.shape(),
                shape
            );
        }
        if t.dtype() != dtype {
            bail!(
                "stack: input {i} dtype {} != input 0 dtype {dtype}",
                t.dtype()
            );
        }
        if !t.is_contiguous() {
            bail!("stack: input {i} must be contiguous");
        }
    }
    let n = inputs.len();
    let per = dtype.size_in_bytes();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis..].iter().product::<usize>().max(1);

    // Output shape: shape with N inserted at `axis`.
    let mut out_shape = shape.clone();
    out_shape.insert(axis, n);

    let mut out_bytes = vec![0u8; outer * n * inner * per];
    // For each outer slab, for each input (axis index), copy that
    // input's contiguous inner-block into the output at the right
    // offset.
    for o in 0..outer {
        for (i, t) in inputs.iter().enumerate() {
            let t_cpu = t
                .storage()
                .as_any()
                .downcast_ref::<CpuStorage>()
                .ok_or_else(|| Error::from_str("stack: storage must be CpuStorage"))?;
            let t_bytes = t_cpu.as_bytes();
            let src_start = o * inner * per;
            let src_end = src_start + inner * per;
            let dst_start = (o * n + i) * inner * per;
            let dst_end = dst_start + inner * per;
            out_bytes[dst_start..dst_end].copy_from_slice(&t_bytes[src_start..src_end]);
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
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
    fn stack_two_rank1_axis_0() {
        // a [3] + b [3] → [2, 3].
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], vec![3]).unwrap();
        let out = stack(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stack_two_rank1_axis_1() {
        // a [3] + b [3] stacked at axis 1 → [3, 2].
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], vec![3]).unwrap();
        let out = stack(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        // Layout: out[i, 0] = a[i], out[i, 1] = b[i]
        // → [1, 10, 2, 20, 3, 30]
        assert_eq!(read_f32(&out), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn stack_three_rank2_axis_0() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = Tensor::from_slice(&[9.0f32, 10.0, 11.0, 12.0], vec![2, 2]).unwrap();
        let out = stack(&[&a, &b, &c], 0).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);
        assert_eq!(
            read_f32(&out),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn stack_rank2_axis_middle() {
        // 2 inputs of shape [2, 3] stacked at axis 1 → [2, 2, 3].
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![2, 3],
        )
        .unwrap();
        let out = stack(&[&a, &b], 1).unwrap();
        assert_eq!(out.shape(), &[2, 2, 3]);
        // out[b, 0, :] = a[b, :]; out[b, 1, :] = b[b, :]
        // → [1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60]
        assert_eq!(
            read_f32(&out),
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0]
        );
    }

    #[test]
    fn stack_one_input_adds_singleton_axis() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let out = stack(&[&a], 0).unwrap();
        assert_eq!(out.shape(), &[1, 3]);
        assert_eq!(read_f32(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn stack_empty_inputs_errors() {
        let e = stack(&[], 0).unwrap_err();
        assert!(e.to_string().contains("at least one"));
    }

    #[test]
    fn stack_axis_out_of_range_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = stack(&[&a], 5).unwrap_err();
        assert!(e.to_string().contains("axis"));
    }

    #[test]
    fn stack_shape_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0], vec![2]).unwrap();
        let e = stack(&[&a, &b], 0).unwrap_err();
        assert!(e.to_string().contains("shape"));
    }

    #[test]
    fn stack_dtype_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let bf = vec![half::bf16::from_f32(1.0)];
        let b = Tensor::from_slice(&bf, vec![1]).unwrap();
        let e = stack(&[&a, &b], 0).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }

    #[test]
    fn stack_bf16_round_trips() {
        let bf1: Vec<half::bf16> = [1.0f32, 2.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let bf2: Vec<half::bf16> = [3.0f32, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let a = Tensor::from_slice(&bf1, vec![2]).unwrap();
        let b = Tensor::from_slice(&bf2, vec![2]).unwrap();
        let out = stack(&[&a, &b], 0).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.shape(), &[2, 2]);
    }
}
