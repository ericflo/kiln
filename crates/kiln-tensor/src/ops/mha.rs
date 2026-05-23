//! `multi_head_attention` — multi-head attention with shape
//! reshape + per-head SDPA.
//!
//! Takes `Q`, `K`, `V` each `[B, M, n_heads * head_dim]` and:
//! 1. Reshapes to `[B, M, n_heads, head_dim]`
//! 2. Transposes to `[B, n_heads, M, head_dim]`
//! 3. Calls `causal_scaled_dot_product_attention` (or non-causal)
//! 4. Transposes + reshapes back to `[B, M, n_heads * head_dim]`
//!
//! Reference CPU path; Phase 6.x fuses into a single GPU kernel.

use crate::ops::{causal_scaled_dot_product_attention, scaled_dot_product_attention};
use crate::{bail, Result, Tensor};

pub fn multi_head_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    causal: bool,
) -> Result<Tensor> {
    if n_heads == 0 {
        bail!("multi_head_attention: n_heads must be > 0");
    }
    if q.rank() != 3 {
        bail!(
            "multi_head_attention: Q must be rank-3 [B, M, hidden], got {:?}",
            q.shape()
        );
    }
    let qs = q.shape();
    let hidden = qs[2];
    if !hidden.is_multiple_of(n_heads) {
        bail!(
            "multi_head_attention: hidden ({hidden}) must be a multiple of n_heads ({n_heads})"
        );
    }
    let head_dim = hidden / n_heads;
    let b = qs[0];
    let m = qs[1];
    if k.shape() != qs || v.shape() != qs {
        bail!(
            "multi_head_attention: Q/K/V shapes must match, got Q={:?}, K={:?}, V={:?}",
            qs,
            k.shape(),
            v.shape()
        );
    }
    // 1. Reshape to [B, M, H, D]
    let q4 = q.reshape(vec![b, m, n_heads, head_dim])?;
    let k4 = k.reshape(vec![b, m, n_heads, head_dim])?;
    let v4 = v.reshape(vec![b, m, n_heads, head_dim])?;
    // 2. Transpose to [B, H, M, D]
    let q4 = q4.transpose(1, 2)?.contiguous()?;
    let k4 = k4.transpose(1, 2)?.contiguous()?;
    let v4 = v4.transpose(1, 2)?.contiguous()?;
    // 3. SDPA along the trailing two axes [M, D]
    let attn = if causal {
        causal_scaled_dot_product_attention(&q4, &k4, &v4)?
    } else {
        scaled_dot_product_attention(&q4, &k4, &v4)?
    };
    // 4. Transpose back to [B, M, H, D] + reshape to [B, M, hidden]
    let attn = attn.transpose(1, 2)?.contiguous()?;
    attn.reshape(vec![b, m, hidden])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuStorage;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn mha_returns_correct_shape() {
        // B=1, M=2, hidden=8, n_heads=2 → head_dim=4.
        let q = Tensor::from_slice(&[0.0f32; 16], vec![1, 2, 8]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 16], vec![1, 2, 8]).unwrap();
        let v = Tensor::from_slice(&[1.0f32; 16], vec![1, 2, 8]).unwrap();
        let y = multi_head_attention(&q, &k, &v, 2, false).unwrap();
        assert_eq!(y.shape(), &[1, 2, 8]);
    }

    #[test]
    fn mha_uniform_attention_averages_v() {
        // Q=K=zeros → uniform attention → output = average of V across seq.
        // B=1, M=2, hidden=2, n_heads=1 (so head_dim=2).
        let q = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let y = multi_head_attention(&q, &k, &v, 1, false).unwrap();
        let yv = read_f32(&y);
        // Both rows average V → [2, 3].
        for row in 0..2 {
            assert!((yv[row * 2] - 2.0).abs() < 1e-5);
            assert!((yv[row * 2 + 1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn mha_n_heads_zero_errors() {
        let q = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let v = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let e = multi_head_attention(&q, &k, &v, 0, false).unwrap_err();
        assert!(e.to_string().contains("n_heads"));
    }

    #[test]
    fn mha_hidden_not_divisible_errors() {
        let q = Tensor::from_slice(&[0.0f32; 6], vec![1, 2, 3]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 6], vec![1, 2, 3]).unwrap();
        let v = Tensor::from_slice(&[0.0f32; 6], vec![1, 2, 3]).unwrap();
        let e = multi_head_attention(&q, &k, &v, 2, false).unwrap_err();
        assert!(e.to_string().contains("multiple"));
    }

    #[test]
    fn mha_causal_variant_runs() {
        let q = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let v = Tensor::from_slice(&[1.0f32; 8], vec![1, 2, 4]).unwrap();
        let y = multi_head_attention(&q, &k, &v, 2, true).unwrap();
        assert_eq!(y.shape(), &[1, 2, 4]);
        for val in read_f32(&y) {
            assert!(val.is_finite());
        }
    }
}
