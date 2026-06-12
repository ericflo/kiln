//! `rope_split_half` — split-half ("rotate-half" / GPT-NeoX-style) rotary
//! position embedding, the convention kiln's Qwen3.5-4B model uses.
//!
//! # Why this exists (vs [`crate::ops::rope`])
//!
//! [`crate::ops::rope`] implements the **interleaved** (GPT-J) convention,
//! pairing adjacent lanes `(2i, 2i+1)`. kiln's production `apply_rope`
//! (kiln-model `forward.rs`) instead uses the **split-half** (GPT-NeoX)
//! convention, splitting the rotary block into two contiguous halves and
//! pairing lane `i` with lane `i + rotary_dim/2`:
//!
//! ```text
//! x1 = x[..., 0            : rotary_dim/2]
//! x2 = x[..., rotary_dim/2 : rotary_dim ]
//! out[..., 0            : rotary_dim/2] = x1*cos - x2*sin
//! out[..., rotary_dim/2 : rotary_dim ] = x1*sin + x2*cos
//! out[..., rotary_dim   : head_dim    ] = x[...]   (pass-through tail)
//! ```
//!
//! The two conventions disagree for `rotary_dim >= 4`, so routing the
//! model's rotary embedding through the interleaved op would silently
//! corrupt outputs. This op fills that gap for the #1082 candle->kt
//! migration of `apply_rope`.
//!
//! # Layout
//!
//! Matches `apply_rope`'s contract exactly: `x` is rank-4
//! `[batch, seq, num_heads, head_dim]`; `cos`/`sin` are rank-2
//! `[seq, rotary_dim/2]`, broadcast over the batch (dim 0) and num_heads
//! (dim 2) axes and indexed by the seq (dim 1) axis.
//!
//! # Device-agnostic by construction
//!
//! This is a **composite** of existing device-agnostic core ops
//! (`cast`, `narrow`, `broadcast_to`, `mul`, `add`, `sub`, `concat`),
//! so it runs unmodified on CPU / CUDA / Vulkan / Metal — wherever those
//! primitives are implemented — with no per-backend kernel and no host
//! round-trip. Intermediate arithmetic is done in f32 (matching the
//! production composite's precision) and cast back to the input dtype.
//!
//! # Backward
//!
//! RoPE is a unitary rotation, so its adjoint is rotation by the negated
//! angle: `rope_split_half(dy, cos, -sin, rotary_dim)`. See
//! `kiln_autograd::backwards::rope_split_half::RopeSplitHalfBackward`.

use crate::ops::{add, broadcast_to, cast, concat, mul, sub};
use crate::{DType, Result, Tensor, bail};

/// Split-half (GPT-NeoX-style) rotary position embedding. See module docs.
///
/// * `x` — rank-4 `[batch, seq, num_heads, head_dim]`, contiguous,
///   dtype F32/BF16/F16.
/// * `cos`, `sin` — rank-2 `[seq, rotary_dim/2]`, contiguous, same shape,
///   dtype F32/BF16/F16.
/// * `rotary_dim` — even, `0 < rotary_dim <= head_dim`. The first
///   `rotary_dim` lanes are rotated; the remaining pass through.
pub fn rope_split_half(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<Tensor> {
    validate(x, cos, sin, rotary_dim)?;

    let shape = x.shape();
    let (batch, seq, heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let half = rotary_dim / 2;

    // Work in f32 for precision parity with the production composite.
    let dtype = x.dtype();
    let to_f32 = |t: &Tensor| -> Result<Tensor> {
        if t.dtype() == DType::F32 {
            Ok(t.clone())
        } else {
            cast(t, DType::F32)
        }
    };
    let xf = to_f32(x)?;
    let cosf = to_f32(cos)?;
    let sinf = to_f32(sin)?;

    // Broadcast cos/sin [seq, half] -> [batch, seq, heads, half].
    let target = [batch, seq, heads, half];
    let cos_b = broadcast_to(&cosf.unsqueeze(0)?.unsqueeze(2)?, &target)?.contiguous()?;
    let sin_b = broadcast_to(&sinf.unsqueeze(0)?.unsqueeze(2)?, &target)?.contiguous()?;

    // Split the rotary block into two contiguous halves.
    let x1 = xf.narrow(3, 0, half)?.contiguous()?;
    let x2 = xf.narrow(3, half, half)?.contiguous()?;

    // r1 = x1*cos - x2*sin ; r2 = x1*sin + x2*cos
    let r1 = sub(&mul(&x1, &cos_b)?, &mul(&x2, &sin_b)?)?;
    let r2 = add(&mul(&x1, &sin_b)?, &mul(&x2, &cos_b)?)?;

    let out_f32 = if rotary_dim < head_dim {
        let pass = xf
            .narrow(3, rotary_dim, head_dim - rotary_dim)?
            .contiguous()?;
        concat(&[&r1, &r2, &pass], 3)?
    } else {
        concat(&[&r1, &r2], 3)?
    };

    if dtype == DType::F32 {
        Ok(out_f32)
    } else {
        cast(&out_f32, dtype)
    }
}

fn validate(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<()> {
    if x.rank() != 4 {
        bail!(
            "rope_split_half: x must be rank-4 [batch, seq, num_heads, head_dim], got shape {:?}",
            x.shape()
        );
    }
    if cos.rank() != 2 || sin.rank() != 2 {
        bail!(
            "rope_split_half: cos and sin must be rank-2 [seq, rotary_dim/2], got cos={:?}, sin={:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if cos.shape() != sin.shape() {
        bail!(
            "rope_split_half: cos and sin shapes must match (got {:?} vs {:?})",
            cos.shape(),
            sin.shape()
        );
    }
    let seq = x.shape()[1];
    let head_dim = x.shape()[3];
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        bail!("rope_split_half: rotary_dim must be positive and even, got {rotary_dim}");
    }
    if rotary_dim > head_dim {
        bail!("rope_split_half: rotary_dim ({rotary_dim}) > head_dim ({head_dim})");
    }
    if cos.shape()[0] != seq {
        bail!(
            "rope_split_half: cos.shape[0] ({}) != x seq ({seq})",
            cos.shape()[0]
        );
    }
    if cos.shape()[1] * 2 != rotary_dim {
        bail!(
            "rope_split_half: cos.shape[1] ({}) * 2 != rotary_dim ({rotary_dim})",
            cos.shape()[1]
        );
    }
    for (name, t) in [("x", x), ("cos", cos), ("sin", sin)] {
        if !matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            bail!(
                "rope_split_half: {name} dtype must be F32/BF16/F16, got {}",
                t.dtype()
            );
        }
        if !t.is_contiguous() {
            bail!("rope_split_half: {name} must be contiguous");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuStorage, Tensor};

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Independent host f32 split-half forward reference.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut out = x.to_vec();
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..heads {
                    let row = (((b * seq) + s) * heads + h) * head_dim;
                    let sched = s * half;
                    for i in 0..half {
                        let c = cos[sched + i];
                        let sn = sin[sched + i];
                        let x1 = x[row + i];
                        let x2 = x[row + half + i];
                        out[row + i] = x1 * c - x2 * sn;
                        out[row + half + i] = x1 * sn + x2 * c;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn matches_reference_partial_rotary() {
        let (batch, seq, heads, head_dim, rotary_dim) = (2, 3, 2, 8, 4);
        let half = rotary_dim / 2;
        let n = batch * seq * heads * head_dim;
        let x: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for s in 0..seq {
            for i in 0..half {
                let theta = (s as f32) * 0.5 + (i as f32) * 0.3;
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        let xt = Tensor::from_slice(&x, vec![batch, seq, heads, head_dim]).unwrap();
        let ct = Tensor::from_slice(&cos, vec![seq, half]).unwrap();
        let st = Tensor::from_slice(&sin, vec![seq, half]).unwrap();
        let got = read_f32(&rope_split_half(&xt, &ct, &st, rotary_dim).unwrap());
        let want = reference(&x, &cos, &sin, batch, seq, heads, head_dim, rotary_dim);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "idx {i}: got {g}, want {w}");
        }
    }

    #[test]
    fn full_rotary_no_passthrough() {
        let (batch, seq, heads, head_dim, rotary_dim) = (1, 2, 1, 4, 4);
        let half = rotary_dim / 2;
        let n = batch * seq * heads * head_dim;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 0.3).collect();
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for s in 0..seq {
            for i in 0..half {
                let theta = 0.4 * (s as f32 + 1.0) + 0.2 * (i as f32);
                cos.push(theta.cos());
                sin.push(theta.sin());
            }
        }
        let xt = Tensor::from_slice(&x, vec![batch, seq, heads, head_dim]).unwrap();
        let ct = Tensor::from_slice(&cos, vec![seq, half]).unwrap();
        let st = Tensor::from_slice(&sin, vec![seq, half]).unwrap();
        let got = read_f32(&rope_split_half(&xt, &ct, &st, rotary_dim).unwrap());
        let want = reference(&x, &cos, &sin, batch, seq, heads, head_dim, rotary_dim);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5);
        }
    }

    #[test]
    fn split_half_not_interleaved() {
        // Guard against regressing to the (2i, 2i+1) interleaved pairing.
        let (batch, seq, heads, head_dim, rotary_dim) = (1, 1, 1, 4, 4);
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let cos = vec![1.0f32, 0.0];
        let sin = vec![0.0f32, 1.0];
        let xt = Tensor::from_slice(&x, vec![batch, seq, heads, head_dim]).unwrap();
        let ct = Tensor::from_slice(&cos, vec![seq, 2]).unwrap();
        let st = Tensor::from_slice(&sin, vec![seq, 2]).unwrap();
        let got = read_f32(&rope_split_half(&xt, &ct, &st, rotary_dim).unwrap());
        // split-half: x1=[1,2], x2=[3,4]
        //   out[0]= x1[0]*1 - x2[0]*0 = 1
        //   out[1]= x1[1]*0 - x2[1]*1 = -4
        //   out[2]= x1[0]*0 + x2[0]*1 = 3
        //   out[3]= x1[1]*1 + x2[1]*0 = 2
        let want = vec![1.0f32, -4.0, 3.0, 2.0];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn rejects_bad_rank() {
        let x = Tensor::from_slice(&[1.0f32, 2.0], vec![1, 2]).unwrap();
        let c = Tensor::from_slice(&[1.0f32], vec![1, 1]).unwrap();
        let s = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
        assert!(rope_split_half(&x, &c, &s, 2).is_err());
    }
}
