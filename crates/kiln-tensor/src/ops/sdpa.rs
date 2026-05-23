//! `scaled_dot_product_attention` — the standard attention primitive.
//!
//! ```text
//! attn(Q, K, V) = softmax(Q @ K^T / √d) @ V
//! ```
//!
//! `Q`, `K`, `V` all `[..., M, D]`. Output `[..., M, D]`. This is a
//! reference CPU implementation built from existing primitives;
//! Phase 6.x will fuse into a single GPU kernel (FlashAttention).

use crate::ops::{causal_mask, masked_fill, matmul, mul_scalar, softmax_last_dim};
use crate::{bail, Result, Tensor};

/// Causal variant of [`scaled_dot_product_attention`]. Applies a
/// causal mask to the scores so each position only attends to itself
/// and prior positions. Requires `Q.seq == K.seq` (self-attention).
pub fn causal_scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    let qs = q.shape();
    let q_rank = qs.len();
    if q_rank < 2 {
        bail!("causal_sdpa: rank must be ≥ 2");
    }
    let seq_q = qs[q_rank - 2];
    let seq_k = k.shape()[q_rank - 2];
    if seq_q != seq_k {
        bail!(
            "causal_sdpa: Q.seq ({seq_q}) must equal K.seq ({seq_k}) for self-attention"
        );
    }
    let head_dim = qs[q_rank - 1];
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let k_t = k.transpose(q_rank - 2, q_rank - 1)?.contiguous()?;
    let scores_raw = matmul(q, &k_t)?;
    let scores_scaled = mul_scalar(&scores_raw, scale)?;

    // Build causal mask matching the scores' trailing two axes,
    // then broadcast the mask over the batch axes via a reshape into
    // the expected layout.
    let mask = causal_mask(seq_q)?;
    // Mask shape is [seq, seq]; scores are [..., seq, seq]. We need a
    // same-shape mask. We replicate via broadcast_to-like manual
    // construction: build a fresh mask of the scores' shape with the
    // [seq, seq] pattern repeated across batch axes.
    let scores_shape = scores_scaled.shape();
    if scores_shape.len() != q_rank {
        bail!("causal_sdpa: unexpected scores rank");
    }
    let scores_outer: usize = scores_shape[..q_rank - 2].iter().product::<usize>().max(1);
    // Replicate the mask `scores_outer` times.
    let mask_bytes = mask
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("causal_sdpa: mask storage must be CpuStorage"))?
        .as_bytes()
        .to_vec();
    let mut full_mask = vec![0u8; scores_outer * seq_q * seq_q];
    for o in 0..scores_outer {
        let start = o * seq_q * seq_q;
        let end = start + seq_q * seq_q;
        full_mask[start..end].copy_from_slice(&mask_bytes);
    }
    let mut full_mask_shape = scores_shape.to_vec();
    let _ = full_mask_shape; // shape not directly used; rebuild below
    let mut new_shape = scores_shape.to_vec();
    // mask shape matches scores shape since both end in [seq_q, seq_k=seq_q].
    let cpu = crate::CpuStorage::from_bytes(crate::DType::U8, full_mask)?;
    let storage: crate::Storage = std::sync::Arc::new(cpu);
    let mask_t = Tensor::from_parts(
        storage,
        crate::Layout::contiguous(new_shape.clone()),
        crate::TensorId::next(),
    )?;
    new_shape.clear();
    let masked = masked_fill(&scores_scaled, &mask_t, f32::NEG_INFINITY)?;
    let attn = softmax_last_dim(&masked)?;
    matmul(&attn, v)
}

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

    #[test]
    fn causal_sdpa_first_position_attends_only_to_self() {
        // Single-head, seq=2, head_dim=2. Q=K=I; V = [[1,0],[0,1]].
        // Without causal mask attention is mixed; with causal:
        //   Row 0 attends only to position 0 → V[0] = [1, 0].
        //   Row 1 attends to both 0 and 1.
        let q = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![1, 2, 2]).unwrap();
        let k = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![1, 2, 2]).unwrap();
        let v = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![1, 2, 2]).unwrap();
        let y = causal_scaled_dot_product_attention(&q, &k, &v).unwrap();
        let yv = read_f32(&y);
        // Row 0 = V[0] = [1, 0] exactly.
        assert!((yv[0] - 1.0).abs() < 1e-4);
        assert!(yv[1].abs() < 1e-4);
    }

    #[test]
    fn causal_sdpa_returns_correct_shape() {
        let q = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 8], vec![1, 2, 4]).unwrap();
        let v = Tensor::from_slice(&[1.0f32; 8], vec![1, 2, 4]).unwrap();
        let y = causal_scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(y.shape(), &[1, 2, 4]);
    }

    #[test]
    fn causal_sdpa_seq_mismatch_errors() {
        let q = Tensor::from_slice(&[0.0f32; 6], vec![1, 3, 2]).unwrap();
        let k = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let v = Tensor::from_slice(&[0.0f32; 4], vec![1, 2, 2]).unwrap();
        let e = causal_scaled_dot_product_attention(&q, &k, &v).unwrap_err();
        assert!(e.to_string().contains("Q.seq"));
    }
}
