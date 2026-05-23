//! `argmax_last_dim` — index of the maximum along the trailing axis.
//!
//! The greedy-sampler entry point. Replaces candle's
//! `Tensor::argmax_keepdim(-1)` and the on-device sampler kernel paths
//! at `crates/kiln-model/src/sampling.rs:5-110` (Phase 0.1's audit cites
//! these as the "preserve as-is" sampler surface).
//!
//! # Semantics
//!
//! For each row along the last axis of `x: [..., D]`, returns the
//! index `argmax_d x[r, d]` as `I64`. The output shape drops the
//! trailing axis: `[..., D] -> [...]`.
//!
//! Ties are broken by **lowest index** — same convention as
//! `slice.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1))` in
//! standard Rust and `candle_core::Tensor::argmax`.
//!
//! # Determinism
//!
//! `Constructive`. Single-pass max-scan with deterministic tie-break.

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// argmax along the trailing axis. Output dtype is fixed at `I64`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArgmaxLastDimOp;

impl DeviceOp1 for ArgmaxLastDimOp {
    fn name(&self) -> &'static str {
        "argmax_last_dim"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if x.rank() == 0 {
            bail!("ArgmaxLastDimOp: input must have rank ≥ 1");
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "ArgmaxLastDimOp: input dtype must be F32/BF16/F16, got {}",
                x.dtype()
            );
        }
        if !x.is_contiguous() {
            bail!("ArgmaxLastDimOp: input must be contiguous");
        }

        let shape = x.shape();
        let hidden = *shape.last().unwrap();
        let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let dtype = x.dtype();

        let x_cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("ArgmaxLastDimOp: storage must be CpuStorage"))?;
        let x_bytes = x_cpu.as_bytes();

        let mut out: Vec<i64> = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..hidden {
                let v = read_one_f32(dtype, x_bytes, r * hidden + i);
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
                // Tie: keep best_idx unchanged (lowest-index wins).
            }
            out.push(best_idx as i64);
        }

        // Output shape: drop the trailing axis.
        let out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        let bytes: Vec<u8> = out
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let cpu = CpuStorage::from_bytes(DType::I64, bytes)?;
        let storage: Storage = Arc::new(cpu);
        // For rank-1 input the output is a scalar (rank-0).
        let layout = Layout::contiguous(out_shape);
        let tensor = Tensor::from_parts(storage, layout, TensorId::next())?;
        Ok(Some(tensor))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        // argmax has no gradient.
        None
    }
}

/// Dispatch `ArgmaxLastDimOp`. Returns I64 indices, shape =
/// `x.shape[..-1]`.
pub fn argmax_last_dim(x: &Tensor) -> Result<Tensor> {
    dispatch1(&ArgmaxLastDimOp, x)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn read_one_f32(dtype: DType, bytes: &[u8], i: usize) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()),
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_i64(t: &Tensor) -> Vec<i64> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, i64>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn argmax_1d_basic() {
        let x = Tensor::from_slice(&[0.1f32, 0.2, 0.5, 0.05], vec![4]).unwrap();
        let y = argmax_last_dim(&x).unwrap();
        assert_eq!(y.dtype(), DType::I64);
        assert_eq!(y.shape(), &[] as &[usize]);
        assert_eq!(read_i64(&y), vec![2]);
    }

    #[test]
    fn argmax_2d_per_row() {
        // [[1, 5, 3], [9, 2, 7]] -> [1, 0]
        let x = Tensor::from_slice(&[1.0f32, 5.0, 3.0, 9.0, 2.0, 7.0], vec![2, 3]).unwrap();
        let y = argmax_last_dim(&x).unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(read_i64(&y), vec![1, 0]);
    }

    #[test]
    fn argmax_3d_drops_trailing_axis() {
        // shape [2, 1, 3] -> [2, 1]
        let x = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 1.0, 0.0, 0.0], vec![2, 1, 3]).unwrap();
        let y = argmax_last_dim(&x).unwrap();
        assert_eq!(y.shape(), &[2, 1]);
        assert_eq!(read_i64(&y), vec![1, 0]);
    }

    #[test]
    fn argmax_ties_break_to_lowest_index() {
        // Two equal max values at idx 0 and 2 -> argmax = 0.
        let x = Tensor::from_slice(&[5.0f32, 1.0, 5.0, 1.0], vec![4]).unwrap();
        let y = argmax_last_dim(&x).unwrap();
        assert_eq!(read_i64(&y), vec![0]);
    }

    #[test]
    fn argmax_bf16_path() {
        let xv: Vec<half::bf16> = [0.1f32, 0.9, 0.5]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![3]).unwrap();
        let y = argmax_last_dim(&x).unwrap();
        assert_eq!(read_i64(&y), vec![1]);
    }

    #[test]
    fn argmax_negative_values() {
        // All-negative input: argmax is the least-negative.
        let x = Tensor::from_slice(&[-5.0f32, -1.0, -10.0, -2.0], vec![4]).unwrap();
        let y = argmax_last_dim(&x).unwrap();
        assert_eq!(read_i64(&y), vec![1]);
    }

    #[test]
    fn argmax_rejects_rank0() {
        let x = Tensor::zeros_cpu(vec![], DType::F32);
        let e = argmax_last_dim(&x).unwrap_err();
        assert!(e.to_string().contains("rank ≥ 1"));
    }

    #[test]
    fn argmax_rejects_bad_dtype() {
        let x = Tensor::from_slice(&[1u32, 2, 3], vec![3]).unwrap();
        let e = argmax_last_dim(&x).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn op_metadata() {
        let op = ArgmaxLastDimOp;
        assert_eq!(op.name(), "argmax_last_dim");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }
}
