//! Rotary Position Embedding (RoPE) — CPU reference path.
//!
//! Replaces candle's `candle_nn::rotary_emb::rope` at the RoPE call
//! sites in `forward.rs` (per Phase 0.7's preserve-list: `kiln/proj/qkv`,
//! `kiln/attn/q_fa_transpose`, etc. include RoPE-applied tensors).
//!
//! # Semantics
//!
//! Given:
//!
//! - `x: [..., seq, head_dim]` — query or key tensor, F32/BF16/F16.
//! - `cos: [seq, rotary_dim/2]` — precomputed cosines.
//! - `sin: [seq, rotary_dim/2]` — precomputed sines.
//!
//! For each seq position `s`, for each pair `(2i, 2i+1)` in
//! `0..rotary_dim`:
//!
//! ```text
//! x_new[..., s, 2i]   = x[..., s, 2i]   * cos[s, i] - x[..., s, 2i+1] * sin[s, i]
//! x_new[..., s, 2i+1] = x[..., s, 2i]   * sin[s, i] + x[..., s, 2i+1] * cos[s, i]
//! ```
//!
//! Indices beyond `rotary_dim` are passed through unchanged. This
//! supports the **partial-rotary** case (Qwen3.5-4B uses
//! `partial_rotary_factor = 0.25`, so `rotary_dim = 64` of
//! `head_dim = 256`).
//!
//! `rotary_dim` is configured on the [`RopeOp`] struct rather than
//! inferred from the cos/sin shape. Pair count = `rotary_dim / 2`
//! must equal `cos.shape[-1]`.
//!
//! # Determinism
//!
//! `Constructive`. Pointwise per (position, pair); no reduction.

use crate::{
    bail, dispatch3, BackwardOp, CpuStorage, DType, Determinism, DeviceOp3, Error, Layout, Result,
    Storage, Tensor, TensorId,
};
use std::sync::Arc;

/// Rotary position embedding op.
///
/// `rotary_dim` controls how many leading dimensions of `head_dim` get
/// rotated; the remaining `head_dim - rotary_dim` are passed through.
/// Set `rotary_dim == head_dim` for full RoPE; `rotary_dim < head_dim`
/// for partial-rotary (Qwen3.5-4B: `64` of `256`).
#[derive(Debug, Clone, Copy)]
pub struct RopeOp {
    rotary_dim: usize,
}

impl RopeOp {
    pub const fn new(rotary_dim: usize) -> Self {
        RopeOp { rotary_dim }
    }
    pub const fn rotary_dim(self) -> usize {
        self.rotary_dim
    }
}

impl DeviceOp3 for RopeOp {
    fn name(&self) -> &'static str {
        "rope"
    }

    fn determinism(&self) -> Determinism {
        Determinism::Constructive
    }

    fn cpu_fwd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Option<Tensor>> {
        validate(x, cos, sin, self.rotary_dim)?;

        let dtype = x.dtype();
        let shape = x.shape();
        let head_dim = *shape.last().unwrap();
        let seq = shape[shape.len() - 2];
        // Leading rows = product of all-but-trailing-two dims, treating
        // each (...) entry as one row of (seq, head_dim).
        let leading: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
        let rotary_dim = self.rotary_dim;
        let pair_count = rotary_dim / 2;

        let x_cpu = downcast_cpu(x, "x")?;
        let cos_cpu = downcast_cpu(cos, "cos")?;
        let sin_cpu = downcast_cpu(sin, "sin")?;

        let mut out_bytes = x_cpu.as_bytes().to_vec();

        // For each leading row, for each seq position, rotate the first
        // rotary_dim of head_dim.
        let per = dtype.size_in_bytes();
        let row_bytes = seq * head_dim * per;
        let cos_row_bytes = pair_count * cos.dtype().size_in_bytes();
        let sin_row_bytes = pair_count * sin.dtype().size_in_bytes();

        for l in 0..leading {
            for s in 0..seq {
                let cos_row = read_f32_row(cos.dtype(), cos_cpu.as_bytes(), s * cos_row_bytes, pair_count);
                let sin_row = read_f32_row(sin.dtype(), sin_cpu.as_bytes(), s * sin_row_bytes, pair_count);
                for i in 0..pair_count {
                    let two_i = l * row_bytes + (s * head_dim + 2 * i) * per;
                    let two_ip1 = l * row_bytes + (s * head_dim + 2 * i + 1) * per;
                    let a = read_one_f32(dtype, &out_bytes, two_i, per);
                    let b = read_one_f32(dtype, &out_bytes, two_ip1, per);
                    let c = cos_row[i];
                    let s_ = sin_row[i];
                    let new_a = a * c - b * s_;
                    let new_b = a * s_ + b * c;
                    write_one_f32(dtype, &mut out_bytes, two_i, per, new_a);
                    write_one_f32(dtype, &mut out_bytes, two_ip1, per, new_b);
                }
                // The remaining head_dim - rotary_dim entries are
                // pass-through (already present from the initial
                // `to_vec()` copy of x's bytes).
            }
        }

        let cpu = CpuStorage::from_bytes(dtype, out_bytes)?;
        let storage: Storage = Arc::new(cpu);
        let out = Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())?;
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Option<Tensor>> {
        // Validate up front so error semantics match cpu_fwd.
        validate(x, cos, sin, self.rotary_dim)?;
        // CUDA path requires contiguous inputs of supported dtypes.
        // Anything off-path returns None so dispatch3 falls back to CPU.
        if !x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
            return Ok(None);
        }
        if !matches!(
            x.dtype(),
            crate::DType::F32 | crate::DType::BF16 | crate::DType::F16
        ) {
            return Ok(None);
        }
        if !matches!(
            cos.dtype(),
            crate::DType::F32 | crate::DType::BF16 | crate::DType::F16
        ) {
            return Ok(None);
        }
        Ok(Some(crate::cuda_rope(x, cos, sin, self.rotary_dim)?))
    }

    fn bwd(&self) -> Option<Box<dyn BackwardOp>> {
        None
    }
}

/// Dispatch RoPE with the given rotary_dim.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<Tensor> {
    dispatch3(&RopeOp::new(rotary_dim), x, cos, sin)
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn validate(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<()> {
    if x.rank() < 2 {
        bail!(
            "RopeOp: x must have rank ≥ 2 (..., seq, head_dim), got shape {:?}",
            x.shape()
        );
    }
    if cos.rank() != 2 || sin.rank() != 2 {
        bail!(
            "RopeOp: cos and sin must be rank-2 [seq, rotary_dim/2], got cos={:?}, sin={:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if cos.shape() != sin.shape() {
        bail!(
            "RopeOp: cos and sin shapes must match (got {:?} vs {:?})",
            cos.shape(),
            sin.shape()
        );
    }
    let seq = x.shape()[x.rank() - 2];
    let head_dim = x.shape()[x.rank() - 1];
    if cos.shape()[0] != seq {
        bail!(
            "RopeOp: cos.shape[0] ({}) != x seq ({})",
            cos.shape()[0],
            seq
        );
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        bail!(
            "RopeOp: rotary_dim must be positive and even, got {rotary_dim}"
        );
    }
    if rotary_dim > head_dim {
        bail!(
            "RopeOp: rotary_dim ({rotary_dim}) > head_dim ({head_dim})"
        );
    }
    if cos.shape()[1] * 2 != rotary_dim {
        bail!(
            "RopeOp: cos.shape[1] ({}) * 2 != rotary_dim ({rotary_dim})",
            cos.shape()[1]
        );
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "RopeOp: x dtype must be F32/BF16/F16, got {}",
            x.dtype()
        );
    }
    if !matches!(cos.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "RopeOp: cos dtype must be F32/BF16/F16, got {}",
            cos.dtype()
        );
    }
    if !matches!(sin.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        bail!(
            "RopeOp: sin dtype must be F32/BF16/F16, got {}",
            sin.dtype()
        );
    }
    if !x.is_contiguous() {
        bail!("RopeOp: x must be contiguous");
    }
    if !cos.is_contiguous() || !sin.is_contiguous() {
        bail!("RopeOp: cos and sin must be contiguous");
    }
    Ok(())
}

fn downcast_cpu<'a>(t: &'a Tensor, label: &str) -> Result<&'a CpuStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| Error::Msg(format!("RopeOp: {label} storage must be CpuStorage")))
}

fn read_one_f32(dtype: DType, bytes: &[u8], byte_off: usize, per: usize) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[byte_off..byte_off + per].try_into().unwrap()),
        DType::BF16 => {
            half::bf16::from_le_bytes(bytes[byte_off..byte_off + per].try_into().unwrap()).to_f32()
        }
        DType::F16 => {
            half::f16::from_le_bytes(bytes[byte_off..byte_off + per].try_into().unwrap()).to_f32()
        }
        _ => unreachable!(),
    }
}

fn write_one_f32(dtype: DType, bytes: &mut [u8], byte_off: usize, per: usize, value: f32) {
    match dtype {
        DType::F32 => bytes[byte_off..byte_off + per].copy_from_slice(&value.to_le_bytes()),
        DType::BF16 => bytes[byte_off..byte_off + per]
            .copy_from_slice(&half::bf16::from_f32(value).to_le_bytes()),
        DType::F16 => bytes[byte_off..byte_off + per]
            .copy_from_slice(&half::f16::from_f32(value).to_le_bytes()),
        _ => unreachable!(),
    }
}

fn read_f32_row(dtype: DType, bytes: &[u8], byte_off: usize, n: usize) -> Vec<f32> {
    let per = dtype.size_in_bytes();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(read_one_f32(dtype, bytes, byte_off + i * per, per));
    }
    out
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
    fn rope_identity_when_angle_zero() {
        // cos=1, sin=0 -> identity rotation.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let cos = Tensor::from_slice(&[1.0f32, 1.0], vec![1, 2]).unwrap();
        let sin = Tensor::from_slice(&[0.0f32, 0.0], vec![1, 2]).unwrap();
        let y = rope(&x, &cos, &sin, 4).unwrap();
        let got = read_f32(&y);
        for (g, e) in got.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!(approx(*g, *e, 1e-6));
        }
    }

    #[test]
    fn rope_quarter_turn_swaps_pair() {
        // cos=0, sin=1: (a, b) -> (a*0 - b*1, a*1 + b*0) = (-b, a)
        let x = Tensor::from_slice(&[3.0f32, 4.0], vec![1, 2]).unwrap();
        let cos = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let y = rope(&x, &cos, &sin, 2).unwrap();
        let got = read_f32(&y);
        assert!(approx(got[0], -4.0, 1e-6));
        assert!(approx(got[1], 3.0, 1e-6));
    }

    #[test]
    fn rope_partial_rotary_passes_through_tail() {
        // x[..., 0..4] rotates with rotary_dim=2 (1 pair); x[..., 2..4]
        // passes through.
        let x = Tensor::from_slice(&[3.0f32, 4.0, 5.0, 6.0], vec![1, 4]).unwrap();
        let cos = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let y = rope(&x, &cos, &sin, 2).unwrap();
        let got = read_f32(&y);
        // First pair (3, 4) quarter-turned to (-4, 3); tail (5, 6) unchanged.
        assert!(approx(got[0], -4.0, 1e-6));
        assert!(approx(got[1], 3.0, 1e-6));
        assert!(approx(got[2], 5.0, 1e-6));
        assert!(approx(got[3], 6.0, 1e-6));
    }

    #[test]
    fn rope_multi_position_independent() {
        // 2 positions, head_dim=2, rotary_dim=2.
        // Pos 0: (1, 0) with cos=1, sin=0 -> (1, 0)
        // Pos 1: (0, 1) with cos=0, sin=1 -> (-1, 0)
        let x = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let cos = Tensor::from_slice(&[1.0f32, 0.0], vec![2, 1]).unwrap();
        let sin = Tensor::from_slice(&[0.0f32, 1.0], vec![2, 1]).unwrap();
        let y = rope(&x, &cos, &sin, 2).unwrap();
        let got = read_f32(&y);
        assert!(approx(got[0], 1.0, 1e-6));
        assert!(approx(got[1], 0.0, 1e-6));
        assert!(approx(got[2], -1.0, 1e-6));
        assert!(approx(got[3], 0.0, 1e-6));
    }

    #[test]
    fn rope_3d_batched() {
        // Shape [batch=2, seq=1, head_dim=2]; quarter-turn applied to each.
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]).unwrap();
        let cos = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        let sin = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let y = rope(&x, &cos, &sin, 2).unwrap();
        let got = read_f32(&y);
        assert!(approx(got[0], -2.0, 1e-6));
        assert!(approx(got[1], 1.0, 1e-6));
        assert!(approx(got[2], -4.0, 1e-6));
        assert!(approx(got[3], 3.0, 1e-6));
    }

    #[test]
    fn rope_rejects_odd_rotary_dim() {
        let x = Tensor::zeros_cpu(vec![1, 4], DType::F32);
        let cos = Tensor::zeros_cpu(vec![1, 1], DType::F32);
        let sin = Tensor::zeros_cpu(vec![1, 1], DType::F32);
        let e = rope(&x, &cos, &sin, 3).unwrap_err();
        assert!(e.to_string().contains("must be positive and even"));
    }

    #[test]
    fn rope_rejects_rotary_dim_gt_head_dim() {
        let x = Tensor::zeros_cpu(vec![1, 4], DType::F32);
        let cos = Tensor::zeros_cpu(vec![1, 4], DType::F32);
        let sin = Tensor::zeros_cpu(vec![1, 4], DType::F32);
        let e = rope(&x, &cos, &sin, 8).unwrap_err();
        assert!(e.to_string().contains("> head_dim"));
    }

    #[test]
    fn rope_rejects_cos_sin_shape_mismatch() {
        let x = Tensor::zeros_cpu(vec![1, 2], DType::F32);
        let cos = Tensor::zeros_cpu(vec![1, 1], DType::F32);
        let sin = Tensor::zeros_cpu(vec![2, 1], DType::F32);
        let e = rope(&x, &cos, &sin, 2).unwrap_err();
        assert!(e.to_string().contains("must match"));
    }

    #[test]
    fn op_metadata() {
        let op = RopeOp::new(64);
        assert_eq!(op.name(), "rope");
        assert!(op.determinism().is_constructive());
        assert!(op.bwd().is_none());
        assert_eq!(op.rotary_dim(), 64);
    }
}
