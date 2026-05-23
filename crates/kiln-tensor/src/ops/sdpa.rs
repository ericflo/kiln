//! `scaled_dot_product_attention` — the standard attention primitive.
//!
//! ```text
//! attn(Q, K, V) = softmax(Q @ K^T / √d) @ V
//! ```
//!
//! `Q`, `K`, `V` all `[..., M, D]`. Output `[..., M, D]`. This is a
//! reference CPU implementation built from existing primitives;
//! Phase 6.x will fuse into a single GPU kernel (FlashAttention).

use crate::ops::{matmul, mul_scalar, softmax_last_dim};
use crate::{bail, Result, Tensor};

pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let qs = q.shape();
    let ks = k.shape();
    let vs = v.shape();
    if qs.len() != ks.len() || qs.len() != vs.len() {
        bail!(
            "sdpa: Q/K/V must share rank; got Q={qs:?}, K={ks:?}, V={vs:?}"
        );
    }
    if qs.len() < 2 {
        bail!("sdpa: rank must be ≥ 2");
    }
    let q_rank = qs.len();
    if qs[q_rank - 1] != ks[q_rank - 1] {
        bail!(
            "sdpa: Q.head_dim ({}) != K.head_dim ({})",
            qs[q_rank - 1],
            ks[q_rank - 1]
        );
    }
    if ks[q_rank - 2] != vs[q_rank - 2] {
        bail!(
            "sdpa: K.seq ({}) != V.seq ({})",
            ks[q_rank - 2],
            vs[q_rank - 2]
        );
    }
    let head_dim = qs[q_rank - 1];
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    // K^T: swap last two axes + contiguous.
    let k_t = k.transpose(q_rank - 2, q_rank - 1)?.contiguous()?;
    // scores = Q @ K^T
    let scores_raw = matmul(q, &k_t)?;
    // scale
    let scores_scaled = mul_scalar(&scores_raw, scale)?;
    // softmax over last axis
    let attn = softmax_last_dim(&scores_scaled)?;
    // attn @ V
    matmul(&attn, v)
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
    fn sdpa_returns_correct_shape() {
        // Q, K, V each [B=1, M=2, D=4] → output [1, 2, 4].
        let q = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 4]).unwrap();
        let y = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(y.shape(), &[1, 2, 4]);
    }

    #[test]
    fn sdpa_uniform_attention_averages_v() {
        // Q=K=zeros → scores all 0 → softmax uniform → output is V averaged.
        // V = [[1, 2], [3, 4]] (M=2, D=2). avg = [(1+3)/2, (2+4)/2] = [2, 3].
        let q = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let y = scaled_dot_product_attention(&q, &k, &v).unwrap();
        let yv = read_f32(&y);
        for row in 0..2 {
            assert!((yv[row * 2] - 2.0).abs() < 1e-5, "row {row} col 0");
            assert!((yv[row * 2 + 1] - 3.0).abs() < 1e-5, "row {row} col 1");
        }
    }

    #[test]
    fn sdpa_q_k_dim_mismatch_errors() {
        let q = Tensor::from_slice(&[0.0f32; 6], vec![1, 2, 3]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let v = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let e = scaled_dot_product_attention(&q, &k, &v).unwrap_err();
        assert!(e.to_string().contains("head_dim"));
    }

    #[test]
    fn sdpa_rank_mismatch_errors() {
        let q = Tensor::from_slice(&[0.0f32; 4], vec![2, 2]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let v = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let e = scaled_dot_product_attention(&q, &k, &v).unwrap_err();
        assert!(e.to_string().contains("rank"));
    }
}
