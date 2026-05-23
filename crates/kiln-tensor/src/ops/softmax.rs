//! `softmax_last_dim` — numerically stable softmax over the trailing axis.
//!
//! Replaces candle's `candle_nn::ops::softmax_last_dim` at every call
//! site (attention scores → probabilities, sampler input, etc.).
//!
//! # Semantics
//!
//! For each row along the last axis of `x: [..., D]`:
//!
//! ```text
//! m = max_d x[r, d]
//! e[d] = exp(x[r, d] - m)
//! out[r, d] = e[d] / sum_d e[d]
//! ```
//!
//! The max-subtraction is the standard numerical-stability transform —
//! without it, `exp(x)` overflows for `x > ~88` in F32 and saturates
//! BF16 even sooner.
//!
//! F32-promoted compute for BF16/F16 inputs; cast back on store.
//!
//! # Determinism
//!
//! `Constructive` — two fixed-tree reductions per row (max and sum)
//! over a single row's D elements. Bit-identical at the same dtype.

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Softmax over the trailing axis.
#[derive(Debug, Default, Clone, Copy)]
pub struct SoftmaxLastDimOp;

impl DeviceOp1 for SoftmaxLastDimOp {
    fn name(&self) -> &'static str {
        "softmax_last_dim"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if x.rank() == 0 {
            bail!("SoftmaxLastDimOp: input must have rank ≥ 1");
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "SoftmaxLastDimOp: dtype must be F32/BF16/F16, got {}",
                x.dtype()
            );
        }
        if !x.is_contiguous() {
            bail!("SoftmaxLastDimOp: input must be contiguous");
        }

        let dtype = x.dtype();
        let shape = x.shape();
        let hidden = *shape.last().unwrap();
        let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

        let x_cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("SoftmaxLastDimOp: storage must be CpuStorage"))?;
        let x_bytes = x_cpu.as_bytes();
        let per = dtype.size_in_bytes();
        let mut out_bytes = vec![0u8; n_rows * hidden * per];

        for r in 0..n_rows {
            let row = load_row_f32(dtype, x_bytes, r, hidden)?;
            // First pass: find the row max for numerical stability.
            let mut m = f32::NEG_INFINITY;
            for &v in &row {
                if v > m {
                    m = v;
                }
            }
            // Handle the all-`-inf` row safely: every output is NaN
            // candidate; we emit uniform 1/D instead since that's the
            // limit. (Matches candle's behaviour on attention masks that
            // mask everything out — practically irrelevant in normal
            // operation, but useful to keep tests deterministic.)
            if !m.is_finite() {
                let uniform = 1.0_f32 / hidden as f32;
                let uniforms = vec![uniform; hidden];
                store_row(dtype, &uniforms, &mut out_bytes, r, hidden)?;
                continue;
            }
            // Second pass: exp and sum.
            let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let inv = 1.0_f32 / sum;
            let probs: Vec<f32> = exps.iter().map(|&e| e * inv).collect();
            store_row(dtype, &probs, &mut out_bytes, r, hidden)?;
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let out = Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())?;
        Ok(Some(out))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Convenience: dispatch `SoftmaxLastDimOp`.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    dispatch1(&SoftmaxLastDimOp, x)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn load_row_f32(dtype: DType, bytes: &[u8], row: usize, hidden: usize) -> Result<Vec<f32>> {
    let per = dtype.size_in_bytes();
    let start = row * hidden * per;
    let end = start + hidden * per;
    if bytes.len() < end {
        bail!(
            "SoftmaxLastDimOp: buffer len {} < {} for row {row} hidden {hidden}",
            bytes.len(),
            end
        );
    }
    let raw = &bytes[start..end];
    let mut out = Vec::with_capacity(hidden);
    match dtype {
        DType::F32 => {
            for i in 0..hidden {
                let chunk: [u8; 4] = raw[i * 4..i * 4 + 4].try_into().unwrap();
                out.push(f32::from_le_bytes(chunk));
            }
        }
        DType::BF16 => {
            for i in 0..hidden {
                let chunk: [u8; 2] = raw[i * 2..i * 2 + 2].try_into().unwrap();
                out.push(half::bf16::from_le_bytes(chunk).to_f32());
            }
        }
        DType::F16 => {
            for i in 0..hidden {
                let chunk: [u8; 2] = raw[i * 2..i * 2 + 2].try_into().unwrap();
                out.push(half::f16::from_le_bytes(chunk).to_f32());
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn store_row(
    dtype: DType,
    values: &[f32],
    out: &mut [u8],
    row: usize,
    hidden: usize,
) -> Result<()> {
    let per = dtype.size_in_bytes();
    let start = row * hidden * per;
    let end = start + hidden * per;
    let raw = &mut out[start..end];
    match dtype {
        DType::F32 => {
            for i in 0..hidden {
                raw[i * 4..i * 4 + 4].copy_from_slice(&values[i].to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..hidden {
                raw[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(values[i]).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..hidden {
                raw[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(values[i]).to_le_bytes());
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    fn approx(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    #[test]
    fn softmax_uniform_input_uniform_output() {
        // softmax([0, 0, 0, 0]) = [0.25, 0.25, 0.25, 0.25]
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let got = read_f32(&y);
        for &v in &got {
            assert!(approx(v, 0.25, 1e-7));
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let got = read_f32(&y);
        let sum: f32 = got.iter().sum();
        assert!(approx(sum, 1.0, 1e-6));
    }

    #[test]
    fn softmax_handles_large_input_no_overflow() {
        // Inputs that would overflow exp() without the max-subtraction.
        let x = Tensor::from_slice(&[1000.0f32, 1001.0], vec![2]).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let got = read_f32(&y);
        for &v in &got {
            assert!(v.is_finite(), "got non-finite: {v}");
        }
        let sum: f32 = got.iter().sum();
        assert!(approx(sum, 1.0, 1e-6));
        // The larger logit should dominate.
        assert!(got[1] > got[0]);
    }

    #[test]
    fn softmax_multi_row_independent() {
        // [[1, 0], [0, 1]] — softmax along last axis.
        let x = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let got = read_f32(&y);
        // row 0: softmax([1, 0]) — same shape as row 1: softmax([0, 1]) reversed.
        let e1 = 1.0_f32.exp();
        let denom = e1 + 1.0;
        let p1 = e1 / denom;
        let p0 = 1.0 / denom;
        assert!(approx(got[0], p1, 1e-6));
        assert!(approx(got[1], p0, 1e-6));
        assert!(approx(got[2], p0, 1e-6));
        assert!(approx(got[3], p1, 1e-6));
    }

    #[test]
    fn softmax_all_neg_inf_returns_uniform() {
        // Mask-everything case: all-neg-inf row -> uniform output.
        let neg_inf = f32::NEG_INFINITY;
        let x = Tensor::from_slice(&[neg_inf, neg_inf, neg_inf], vec![3]).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        let got = read_f32(&y);
        let expected = 1.0 / 3.0;
        for &v in &got {
            assert!(approx(v, expected, 1e-7));
        }
    }

    #[test]
    fn softmax_bf16_within_tolerance() {
        let xv: Vec<half::bf16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![3]).unwrap();
        let y = softmax_last_dim(&x).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        // Sum should be ~1 within bf16 tolerance.
        let mut sum = 0.0_f32;
        for i in 0..3 {
            let v = half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32();
            sum += v;
        }
        assert!(approx(sum, 1.0, 1e-2));
    }

    #[test]
    fn softmax_rank0_errors() {
        let x = Tensor::zeros_cpu(vec![], DType::F32);
        let e = softmax_last_dim(&x).unwrap_err();
        assert!(e.to_string().contains("rank ≥ 1"));
    }

    #[test]
    fn op_metadata() {
        let op = SoftmaxLastDimOp;
        assert_eq!(op.name(), "softmax_last_dim");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
    }
}
