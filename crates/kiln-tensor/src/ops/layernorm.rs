//! `layer_norm` — LayerNorm with weight + bias, per trailing-axis row.
//!
//! ```text
//! mean[r] = (1/D) Σⱼ x[r, j]
//! var[r]  = (1/D) Σⱼ (x[r, j] - mean[r])^2
//! y[r, i] = ((x[r, i] - mean[r]) / sqrt(var[r] + eps)) * weight[i] + bias[i]
//! ```
//!
//! Different from RMSNorm (which subtracts no mean and has no bias).
//! Used by older transformer architectures (BERT, the original
//! attention-is-all-you-need stack); included for completeness +
//! migration parity with `candle_nn::LayerNorm`.
//!
//! # Determinism
//!
//! `Constructive`. Per-row mean + var reduction in F32 with fixed
//! iteration order; bit-identical at the same input dtype.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::DeviceOp3;
use crate::{bail, CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug, Clone, Copy)]
pub struct LayerNormOp {
    eps: f32,
}

impl LayerNormOp {
    pub const fn new(eps: f32) -> Self {
        LayerNormOp { eps }
    }
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

#[cfg(feature = "cuda")]
impl crate::DeviceOp3 for LayerNormOp {
    fn name(&self) -> &'static str {
        "layernorm"
    }

    fn determinism(&self) -> crate::Determinism {
        crate::Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Option<Tensor>> {
        Ok(Some(layer_norm_cpu(x, weight, bias, self.eps)?))
    }

    fn cuda_fwd(&self, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cpu_fwd. Returning Ok(None)
        // triggers CPU fallthrough in DeviceOp3 dispatch.
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if x.rank() == 0 || weight.rank() != 1 || bias.rank() != 1 {
            return Ok(None);
        }
        if x.dtype() != weight.dtype() || x.dtype() != bias.dtype() {
            return Ok(None);
        }
        if !x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() {
            return Ok(None);
        }
        let d = *x.shape().last().unwrap();
        if weight.shape()[0] != d || bias.shape()[0] != d {
            return Ok(None);
        }
        Ok(Some(crate::cuda_layernorm_last_axis(
            x, weight, bias, self.eps,
        )?))
    }
}

/// `y = ((x - mean) / sqrt(var + eps)) * weight + bias` per trailing-axis row.
///
/// `x: [..., D]`, `weight: [D]`, `bias: [D]`. All F32/BF16/F16; dtypes
/// must match across the three inputs.
#[cfg(feature = "cuda")]
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    crate::dispatch3(&LayerNormOp::new(eps), x, weight, bias)
}

/// CPU-only build: no DeviceOp3 dispatch needed; `layer_norm` lowers
/// directly to the CPU path.
#[cfg(not(feature = "cuda"))]
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    layer_norm_cpu(x, weight, bias, eps)
}

/// CPU LayerNorm implementation (the canonical numerical reference
/// for parity tests).
fn layer_norm_cpu(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    validate(x, weight, bias)?;
    let dtype = x.dtype();
    let shape = x.shape();
    let hidden = *shape.last().unwrap();
    let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
    let per = dtype.size_in_bytes();

    let x_cpu = downcast_cpu(x, "x")?;
    let w_cpu = downcast_cpu(weight, "weight")?;
    let b_cpu = downcast_cpu(bias, "bias")?;
    let x_bytes = x_cpu.as_bytes();
    let w_bytes = w_cpu.as_bytes();
    let b_bytes = b_cpu.as_bytes();

    let w_f32 = load_row_f32(dtype, w_bytes, hidden);
    let b_f32 = load_row_f32(dtype, b_bytes, hidden);

    let mut out = vec![0u8; n_rows * hidden * per];
    for r in 0..n_rows {
        let row = load_x_row_f32(dtype, x_bytes, r, hidden);
        let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
        let inv = 1.0_f32 / (var + eps).sqrt();
        for i in 0..hidden {
            let y_i = (row[i] - mean) * inv * w_f32[i] + b_f32[i];
            write_out_f32(dtype, &mut out, r * hidden + i, y_i);
        }
    }

    let cpu = CpuStorage::from_bytes(dtype, out)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<()> {
    if x.rank() == 0 {
        bail!("LayerNormOp: x must have rank ≥ 1");
    }
    if w.rank() != 1 || b.rank() != 1 {
        bail!(
            "LayerNormOp: weight/bias must be rank-1, got w={:?} b={:?}",
            w.shape(),
            b.shape()
        );
    }
    let d = *x.shape().last().unwrap();
    if w.shape()[0] != d || b.shape()[0] != d {
        bail!(
            "LayerNormOp: weight/bias len {} / {} != x trailing axis {d}",
            w.shape()[0],
            b.shape()[0]
        );
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!("LayerNormOp: dtype must be F32/BF16/F16, got {}", x.dtype());
    }
    if x.dtype() != w.dtype() || x.dtype() != b.dtype() {
        bail!(
            "LayerNormOp: dtype mismatch x={}, w={}, b={}",
            x.dtype(),
            w.dtype(),
            b.dtype()
        );
    }
    if !x.is_contiguous() || !w.is_contiguous() || !b.is_contiguous() {
        bail!("LayerNormOp: all inputs must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("LayerNormOp: {label} storage must be CpuStorage")))
}

fn load_row_f32(dtype: DType, bytes: &[u8], len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(read_one_f32(dtype, bytes, i));
    }
    out
}

fn load_x_row_f32(dtype: DType, bytes: &[u8], row: usize, hidden: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(hidden);
    for i in 0..hidden {
        out.push(read_one_f32(dtype, bytes, row * hidden + i));
    }
    out
}

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

fn write_out_f32(dtype: DType, out: &mut [u8], i: usize, v: f32) {
    match dtype {
        DType::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes()),
        DType::BF16 => out[i * 2..i * 2 + 2]
            .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
        DType::F16 => out[i * 2..i * 2 + 2]
            .copy_from_slice(&half::f16::from_f32(v).to_le_bytes()),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "len mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "idx {i}: got {x}, want {y} (tol {tol})"
            );
        }
    }

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn layernorm_zero_input_with_unit_weight_zero_bias() {
        // x = zeros → mean=0, var=0 → /sqrt(eps) division produces 0
        // for the normalized term; with weight=1, bias=0, y = bias = 0.
        let x = Tensor::from_slice(&[0.0f32; 4], vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&[1.0f32; 4], vec![4]).unwrap();
        let b = Tensor::from_slice(&[0.0f32; 4], vec![4]).unwrap();
        let y = layer_norm(&x, &w, &b, 1e-6).unwrap();
        for v in read_f32(&y) {
            assert!(v.abs() < 1e-3, "v={v}");
        }
    }

    #[test]
    fn layernorm_centered_unit_variance_input() {
        // x = [-1, 1] → mean=0, var=1. normalized = [-1, 1].
        // weight = [2, 3], bias = [5, 7] → y = [-1*2+5, 1*3+7] = [3, 10].
        let x = Tensor::from_slice(&[-1.0f32, 1.0], vec![1, 2]).unwrap();
        let w = Tensor::from_slice(&[2.0f32, 3.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 7.0], vec![2]).unwrap();
        let y = layer_norm(&x, &w, &b, 0.0).unwrap();
        approx(&read_f32(&y), &[3.0, 10.0], 1e-5);
    }

    #[test]
    fn layernorm_mean_subtraction() {
        // x = [1, 2, 3, 4] → mean = 2.5, var = 1.25, σ ≈ 1.1180.
        // normalized = (-1.5, -0.5, 0.5, 1.5) / 1.1180 ≈ (-1.34, -0.45, 0.45, 1.34)
        // With w = ones, b = zeros: y is just the normalized values.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let w = Tensor::from_slice(&[1.0f32; 4], vec![4]).unwrap();
        let b = Tensor::from_slice(&[0.0f32; 4], vec![4]).unwrap();
        let y = read_f32(&layer_norm(&x, &w, &b, 0.0).unwrap());
        let sigma = (1.25_f32).sqrt();
        approx(&y, &[-1.5 / sigma, -0.5 / sigma, 0.5 / sigma, 1.5 / sigma], 1e-5);
    }

    #[test]
    fn layernorm_batched_rows_independent() {
        // [2, 3] input — each row normalizes separately.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0], vec![2, 3]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let y = read_f32(&layer_norm(&x, &w, &b, 0.0).unwrap());

        // Row 0: mean = 2, var = 2/3, σ = √(2/3) ≈ 0.8165.
        let s0 = (2.0_f32 / 3.0).sqrt();
        approx(&y[..3], &[-1.0 / s0, 0.0, 1.0 / s0], 1e-5);
        // Row 1: mean = 20, var = 200/3, σ = √(200/3) ≈ 8.165 → same normalized.
        let s1 = (200.0_f32 / 3.0).sqrt();
        approx(&y[3..], &[-10.0 / s1, 0.0, 10.0 / s1], 1e-5);
    }

    #[test]
    fn layernorm_bias_offsets_output() {
        // x = zeros, weight = anything, bias = [1, 2, 3] → y = bias.
        let x = Tensor::from_slice(&[0.0f32; 3], vec![1, 3]).unwrap();
        let w = Tensor::from_slice(&[5.0f32, 7.0, 11.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let y = read_f32(&layer_norm(&x, &w, &b, 1e-6).unwrap());
        approx(&y, &[1.0, 2.0, 3.0], 1e-3);
    }

    #[test]
    fn layernorm_dtype_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let w = Tensor::from_slice(&[half::bf16::from_f32(1.0)], vec![1]).unwrap();
        let b = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let e = layer_norm(&x, &w, &b, 1e-6).unwrap_err();
        assert!(e.to_string().contains("dtype"));
    }

    #[test]
    fn layernorm_shape_mismatch_errors() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let e = layer_norm(&x, &w, &b, 1e-6).unwrap_err();
        assert!(e.to_string().contains("trailing axis"));
    }

    #[test]
    fn op_metadata() {
        let op = LayerNormOp::new(1e-6);
        assert_eq!(op.eps(), 1e-6);
    }
}
