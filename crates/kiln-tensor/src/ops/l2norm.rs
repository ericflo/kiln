//! `l2_norm_last_dim` — L2 normalization along the trailing axis.
//!
//! Replaces candle's `Tensor::l2_normalize(-1)` and is the migration
//! target for the QK-norm path inside attention (`kiln/gdn/qk_norm`,
//! `kiln/gdn/qk_norm_gqa`, etc. from Phase 0.7's preserve-list).
//!
//! # Semantics
//!
//! For each row along the last axis of `x: [..., D]`:
//!
//! ```text
//! norm = sqrt(sum_d x[r, d]^2 + eps)
//! out[r, d] = x[r, d] / norm
//! ```
//!
//! F32-promoted reduction; cast back on store. Eps default `1e-6`
//! prevents NaN on all-zero rows.
//!
//! # Determinism
//!
//! `Constructive`. Fixed-tree per-row reduction; bit-identical at
//! the same input dtype.

use crate::{
    bail, dispatch1, BackwardOp, CpuStorage, DType, Determinism, DeviceOp1, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// L2 normalization along the last axis.
#[derive(Debug, Clone, Copy)]
pub struct L2NormOp {
    eps: f32,
}

impl L2NormOp {
    pub const fn new(eps: f32) -> Self {
        L2NormOp { eps }
    }
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Default for L2NormOp {
    fn default() -> Self {
        L2NormOp::new(1e-6)
    }
}

impl DeviceOp1 for L2NormOp {
    fn name(&self) -> &'static str {
        "l2_norm"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if x.rank() == 0 {
            bail!("L2NormOp: input must have rank ≥ 1");
        }
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "L2NormOp: dtype must be F32/BF16/F16, got {}",
                x.dtype()
            );
        }
        if !x.is_contiguous() {
            bail!("L2NormOp: input must be contiguous");
        }

        let dtype = x.dtype();
        let shape = x.shape();
        let hidden = *shape.last().unwrap();
        let n_rows: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);
        let per = dtype.size_in_bytes();
        let x_cpu = x
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("L2NormOp: storage must be CpuStorage"))?;
        let x_bytes = x_cpu.as_bytes();
        let mut out = vec![0u8; n_rows * hidden * per];

        for r in 0..n_rows {
            let row = load_row_f32(dtype, x_bytes, r, hidden)?;
            let sq: f32 = row.iter().map(|&v| v * v).sum();
            let inv = 1.0_f32 / (sq + self.eps).sqrt();
            let scaled: Vec<f32> = row.iter().map(|&v| v * inv).collect();
            store_row(dtype, &scaled, &mut out, r, hidden)?;
        }

        let cpu = CpuStorage::from_bytes(dtype, out)?;
        let storage: Storage = Arc::new(cpu);
        let t = Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())?;
        Ok(Some(t))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return Ok(None);
        }
        if x.rank() == 0 {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        Ok(Some(crate::cuda_l2norm_last_axis(x, self.eps)?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Dispatch L2-norm with the given eps.
pub fn l2_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    dispatch1(&L2NormOp::new(eps), x)
}

fn load_row_f32(dtype: DType, bytes: &[u8], row: usize, hidden: usize) -> Result<Vec<f32>> {
    let per = dtype.size_in_bytes();
    let start = row * hidden * per;
    let end = start + hidden * per;
    if bytes.len() < end {
        bail!(
            "L2NormOp: buffer len {} < {} for row {row} hidden {hidden}",
            bytes.len(),
            end
        );
    }
    let raw = &bytes[start..end];
    let mut out = Vec::with_capacity(hidden);
    for i in 0..hidden {
        let chunk = &raw[i * per..(i + 1) * per];
        let v = match dtype {
            DType::F32 => f32::from_le_bytes(chunk.try_into().unwrap()),
            DType::BF16 => half::bf16::from_le_bytes(chunk.try_into().unwrap()).to_f32(),
            DType::F16 => half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32(),
            _ => unreachable!(),
        };
        out.push(v);
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
    let raw = &mut out[start..start + hidden * per];
    for i in 0..hidden {
        let chunk = &mut raw[i * per..(i + 1) * per];
        match dtype {
            DType::F32 => chunk.copy_from_slice(&values[i].to_le_bytes()),
            DType::BF16 => chunk.copy_from_slice(&half::bf16::from_f32(values[i]).to_le_bytes()),
            DType::F16 => chunk.copy_from_slice(&half::f16::from_f32(values[i]).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, atol: f32) -> bool {
        (a - b).abs() <= atol
    }

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
    }

    #[test]
    fn l2_norm_unit_vector_unchanged() {
        // [1, 0, 0] has L2-norm 1; eps=0 -> unchanged.
        let x = Tensor::from_slice(&[1.0f32, 0.0, 0.0], vec![3]).unwrap();
        let y = l2_norm(&x, 0.0).unwrap();
        let got = read_f32(&y);
        for (g, e) in got.iter().zip([1.0f32, 0.0, 0.0].iter()) {
            assert!(approx(*g, *e, 1e-6));
        }
    }

    #[test]
    fn l2_norm_normalizes_to_unit_length() {
        let x = Tensor::from_slice(&[3.0f32, 4.0], vec![2]).unwrap();
        let y = l2_norm(&x, 0.0).unwrap();
        let got = read_f32(&y);
        // 3/5, 4/5
        assert!(approx(got[0], 0.6, 1e-6));
        assert!(approx(got[1], 0.8, 1e-6));
        let l2: f32 = got.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(approx(l2, 1.0, 1e-6));
    }

    #[test]
    fn l2_norm_multi_row_independent() {
        // Row 0 = [3, 4]; row 1 = [1, 0].
        let x = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let y = l2_norm(&x, 0.0).unwrap();
        let got = read_f32(&y);
        assert!(approx(got[0], 0.6, 1e-6));
        assert!(approx(got[1], 0.8, 1e-6));
        assert!(approx(got[2], 1.0, 1e-6));
        assert!(approx(got[3], 0.0, 1e-6));
    }

    #[test]
    fn l2_norm_eps_avoids_nan_on_zeros() {
        let x = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
        let y = l2_norm(&x, 1e-6).unwrap();
        let got = read_f32(&y);
        for v in got {
            assert!(v.is_finite());
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn l2_norm_bf16_within_tolerance() {
        let xv: Vec<half::bf16> = [3.0f32, 4.0]
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let x = Tensor::from_slice(&xv, vec![2]).unwrap();
        let y = l2_norm(&x, 0.0).unwrap();
        assert_eq!(y.dtype(), DType::BF16);
        let cpu = y.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let bytes = cpu.as_bytes();
        let v0 = half::bf16::from_le_bytes(bytes[0..2].try_into().unwrap()).to_f32();
        let v1 = half::bf16::from_le_bytes(bytes[2..4].try_into().unwrap()).to_f32();
        assert!(approx(v0, 0.6, 1e-2));
        assert!(approx(v1, 0.8, 1e-2));
    }

    #[test]
    fn l2_norm_rank0_errors() {
        let x = Tensor::zeros_cpu(vec![], DType::F32);
        let e = l2_norm(&x, 1e-6).unwrap_err();
        assert!(e.to_string().contains("rank ≥ 1"));
    }

    #[test]
    fn l2_norm_bad_dtype_errors() {
        let x = Tensor::from_slice(&[1u32, 2, 3], vec![3]).unwrap();
        let e = l2_norm(&x, 1e-6).unwrap_err();
        assert!(e.to_string().contains("F32/BF16/F16"));
    }

    #[test]
    fn op_metadata() {
        let op = L2NormOp::default();
        assert_eq!(op.name(), "l2_norm");
        assert!(op.determinism().is_constructive());
        assert_eq!(op.eps(), 1e-6);
        assert!(op.bwd().is_none());
    }
}
