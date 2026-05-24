//! `RmsNormOp` — root-mean-square norm + per-element scale.
//!
//! Replaces candle's `RmsNorm` layer + the 16+ NVTX call sites in
//! `forward.rs` (per Phase 0.7's preserve-list): `kiln/norm/pre_attn`,
//! `kiln/norm/pre_mlp`, `kiln/final_rmsnorm`, etc.
//!
//! # Semantics
//!
//! Given `x: [..., D]` and `weight: [D]` and a scalar `eps`:
//!
//! ```text
//! for each [...]-row r:
//!     rms = sqrt(mean_over_D(x[r, d]^2) + eps)
//!     out[r, d] = (x[r, d] / rms) * weight[d]
//! ```
//!
//! The reduction is **F32-promoted** regardless of input dtype:
//! BF16 / F16 inputs accumulate in F32, then cast back. This matches
//! candle's RmsNorm semantics and the existing
//! `crates/kiln-rmsnorm-kernel/src/lib.rs:4909` tolerance-band notes:
//! the BF16 forward is bit-stable in the reduction order (fixed-tree)
//! and the cast-back is the only source of per-element ULP variation.
//!
//! # Determinism
//!
//! `Determinism::Constructive`. Forward is fixed-tree reduction over a
//! single row's D elements — bit-identical across runs at the same
//! dtype. (The atomicAdd zone Phase 0.3 documents only applies to the
//! `grad_w` cross-row accumulation in the bwd, not the fwd here.)

use crate::{
    bail, dispatch2, BackwardOp, CpuStorage, DType, Determinism, DeviceOp2, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// RMSNorm op with a fixed epsilon.
///
/// Construction:
///
/// ```rust
/// use kiln_tensor::ops::RmsNormOp;
/// let op = RmsNormOp::new(1e-6);
/// // dispatch via the convenience `rms_norm` helper instead of the
/// // bare op when possible.
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RmsNormOp {
    eps: f32,
}

impl RmsNormOp {
    pub const fn new(eps: f32) -> Self {
        RmsNormOp { eps }
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Default for RmsNormOp {
    /// Default eps matches Qwen3.5-4B's config (`rms_norm_eps = 1e-6`).
    fn default() -> Self {
        RmsNormOp::new(1e-6)
    }
}

impl DeviceOp2 for RmsNormOp {
    fn name(&self) -> &'static str {
        "rmsnorm"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
        validate(x, weight)?;

        let dtype = x.dtype();
        let shape = x.shape();
        let hidden = *shape.last().unwrap();
        let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

        let x_cpu = downcast_cpu(x, "x")?;
        let w_cpu = downcast_cpu(weight, "weight")?;
        let x_bytes = x_cpu.as_bytes();
        let w_bytes = w_cpu.as_bytes();

        let mut out_bytes = vec![0u8; x_bytes.len()];

        // F32 reduction over each row.
        for r in 0..n_rows {
            // 1. Load row to f32.
            let row_f32 = load_row_f32(dtype, x_bytes, r, hidden)?;
            // 2. Compute mean of squares + eps, then 1/sqrt of it.
            let mean_sq: f32 = row_f32.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0_f32 / (mean_sq + self.eps).sqrt();
            // 3. Load weight to f32.
            let w_f32 = load_row_f32(weight.dtype(), w_bytes, 0, hidden)?;
            // 4. Apply scale + cast back to dtype.
            store_row_scaled(dtype, &row_f32, &w_f32, inv_rms, &mut out_bytes, r, hidden)?;
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let out = Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())?;
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor, weight: &Tensor) -> Result<Option<Tensor>> {
        // Gate on the same preconditions as cpu_fwd. Returning Ok(None)
        // triggers CPU fallthrough in DeviceOp2 dispatch.
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if x.rank() == 0 || weight.rank() != 1 {
            return Ok(None);
        }
        if x.dtype() != weight.dtype() {
            return Ok(None);
        }
        if !x.is_contiguous() || !weight.is_contiguous() {
            return Ok(None);
        }
        if *x.shape().last().unwrap() != weight.shape()[0] {
            return Ok(None);
        }
        Ok(Some(crate::cuda_rmsnorm_last_axis(x, weight, self.eps)?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        // Backward is in the atomic-bwd tolerance band (the cross-row
        // grad_w sum uses atomicAdd in F32). Lands under kiln-autograd
        // in a follow-up; today returns None per Phase 1.12 scaffold.
        None
    }
}

/// Convenience: dispatch `RmsNormOp` on `x` and `weight` with the op's eps.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    dispatch2(&RmsNormOp::new(eps), x, weight)
}

// ----------------------------------------------------------------------
// Validation + per-dtype helpers.
// ----------------------------------------------------------------------

fn validate(x: &Tensor, weight: &Tensor) -> Result<()> {
    if x.rank() == 0 {
        bail!("RmsNormOp: x must have rank ≥ 1");
    }
    if weight.rank() != 1 {
        bail!(
            "RmsNormOp: weight must be rank-1, got shape {:?}",
            weight.shape()
        );
    }
    let last_x = *x.shape().last().unwrap();
    let last_w = weight.shape()[0];
    if last_x != last_w {
        bail!(
            "RmsNormOp: x last-axis {last_x} != weight len {last_w}"
        );
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "RmsNormOp: x dtype must be F32/BF16/F16, got {}",
            x.dtype()
        );
    }
    if !matches!(weight.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "RmsNormOp: weight dtype must be F32/BF16/F16, got {}",
            weight.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("RmsNormOp: x must be contiguous");
    }
    if !weight.is_contiguous() {
        bail!("RmsNormOp: weight must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("RmsNormOp: {label} storage must be CpuStorage on CPU device")))
}

fn load_row_f32(dtype: DType, bytes: &[u8], row: usize, hidden: usize) -> Result<Vec<f32>> {
    let per = dtype.size_in_bytes();
    let start = row * hidden * per;
    let end = start + hidden * per;
    if bytes.len() < end {
        bail!(
            "RmsNormOp: load_row_f32: buffer len {} < {} for row {row} hidden {hidden}",
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
        _ => unreachable!("validate() already rejected this dtype"),
    }
    Ok(out)
}

fn store_row_scaled(
    dtype: DType,
    x_f32: &[f32],
    w_f32: &[f32],
    inv_rms: f32,
    out_bytes: &mut [u8],
    row: usize,
    hidden: usize,
) -> Result<()> {
    let per = dtype.size_in_bytes();
    let start = row * hidden * per;
    let end = start + hidden * per;
    let raw = &mut out_bytes[start..end];
    match dtype {
        DType::F32 => {
            for i in 0..hidden {
                let v = x_f32[i] * inv_rms * w_f32[i];
                raw[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => {
            for i in 0..hidden {
                let v = x_f32[i] * inv_rms * w_f32[i];
                raw[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for i in 0..hidden {
                let v = x_f32[i] * inv_rms * w_f32[i];
                raw[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        _ => unreachable!("validate() already rejected this dtype"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn rms_norm_f32_simple() {
        // Input [1, 2, 3, 4]; weight [1,1,1,1]; eps=0.
        // mean_sq = (1+4+9+16)/4 = 7.5; rms = sqrt(7.5) ≈ 2.7386;
        // out[i] = x[i]/rms.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let out = rms_norm(&x, &w, 0.0).unwrap();
        let got = read_f32(&out);
        let rms = (7.5_f32).sqrt();
        let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(approx_eq_f32(*g, *e, 1e-6), "got {g}, expected {e}");
        }
    }

    #[test]
    fn rms_norm_f32_with_weight() {
        // Input [2, 0, 0, 0]; weight [10, 20, 30, 40]; eps=0.
        // mean_sq = 4/4 = 1; rms = 1; out = x * weight = [20, 0, 0, 0].
        let x = Tensor::from_slice(&[2.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let w = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], vec![4]).unwrap();
        let out = rms_norm(&x, &w, 0.0).unwrap();
        let got = read_f32(&out);
        for (g, e) in got.iter().zip([20.0_f32, 0.0, 0.0, 0.0].iter()) {
            assert!(approx_eq_f32(*g, *e, 1e-6));
        }
    }

    #[test]
    fn rms_norm_f32_multi_row() {
        // Two rows, hidden=2.
        let x = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let out = rms_norm(&x, &w, 0.0).unwrap();
        let got = read_f32(&out);
        // Row 0: mean_sq = (9+16)/2 = 12.5; rms = sqrt(12.5).
        let rms0 = 12.5_f32.sqrt();
        // Row 1: mean_sq = (1+0)/2 = 0.5; rms = sqrt(0.5).
        let rms1 = 0.5_f32.sqrt();
        let expected = [3.0 / rms0, 4.0 / rms0, 1.0 / rms1, 0.0 / rms1];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(approx_eq_f32(*g, *e, 1e-6), "{g} != {e}");
        }
    }

    #[test]
    fn rms_norm_bf16_path_promotes_to_f32() {
        // BF16 path: build [3, 4] as bf16, weight ones, expect rms_norm
        // outputs match F32 within bf16 ULP.
        let x_bf16: Vec<half::bf16> = [3.0f32, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let w_bf16: Vec<half::bf16> = [1.0f32, 1.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&x_bf16, vec![2]).unwrap();
        let w = Tensor::from_slice(&w_bf16, vec![2]).unwrap();
        let out = rms_norm(&x, &w, 0.0).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        let cpu = out.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let v0 = half::bf16::from_le_bytes(bytes[0..2].try_into().unwrap()).to_f32();
        let v1 = half::bf16::from_le_bytes(bytes[2..4].try_into().unwrap()).to_f32();
        let rms = 12.5_f32.sqrt();
        // BF16 atol per bench-results/parity-tolerance.csv reduction band.
        assert!(approx_eq_f32(v0, 3.0 / rms, 1e-2), "{v0} vs {}", 3.0 / rms);
        assert!(approx_eq_f32(v1, 4.0 / rms, 1e-2), "{v1} vs {}", 4.0 / rms);
    }

    #[test]
    fn rms_norm_eps_avoids_div_by_zero() {
        // All-zeros input: without eps, this NaNs. With eps>0, out is zero.
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], vec![4]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let out = rms_norm(&x, &w, 1e-6).unwrap();
        let got = read_f32(&out);
        // 0 / sqrt(eps) = 0; no NaN.
        for v in got {
            assert!(v.is_finite(), "got non-finite value: {v}");
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn rms_norm_rejects_rank0() {
        let x = Tensor::zeros_cpu(vec![], DType::F32);
        let w = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = rms_norm(&x, &w, 1e-6).unwrap_err();
        assert!(e.to_string().contains("rank ≥ 1"));
    }

    #[test]
    fn rms_norm_rejects_dim_mismatch() {
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        let e = rms_norm(&x, &w, 1e-6).unwrap_err();
        assert!(e.to_string().contains("last-axis"));
    }

    #[test]
    fn rms_norm_rejects_bad_dtype() {
        let x = Tensor::from_slice(&[1u32, 2, 3, 4], vec![4]).unwrap();
        let w = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let e = rms_norm(&x, &w, 1e-6).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn rms_norm_op_metadata() {
        let op = RmsNormOp::default();
        assert_eq!(op.name(), "rmsnorm");
        assert!(op.determinism().is_constructive());
        assert_eq!(op.eps(), 1e-6);
        assert!(op.bwd().is_none());
    }
}
