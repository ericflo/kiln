//! End-to-end transformer block integration test.
//!
//! Pipeline:
//! ```text
//! x        → rms_norm   → input
//!          → linear(q_proj) → q       (with RoPE)
//!          → linear(k_proj) → k       (with RoPE)
//!          → linear(v_proj) → v
//! q, k, v  → multi_head_attention(causal=true)
//! attn_out → linear(o_proj) → residual_add(x)
//! mid      → rms_norm
//!          → linear(up_gate_proj)
//!          → swiglu
//!          → linear(down_proj)
//!          → residual_add → out
//! ```
//!
//! No backward / training — just verifies the forward graph composes
//! cleanly across all the new ops.

use kiln_tensor::ops::{
    add, layer_norm, linear, multi_head_attention, precompute_rope_freqs, rms_norm, rope,
    swiglu, xavier_uniform,
};
use kiln_tensor::{CpuStorage, DType, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn transformer_block_full_forward_runs_e2e() {
    // Hyperparameters chosen small for fast CPU execution.
    const BATCH: usize = 1;
    const SEQ: usize = 4;
    const HIDDEN: usize = 16;
    const N_HEADS: usize = 2;
    const HEAD_DIM: usize = HIDDEN / N_HEADS;
    const INTERMEDIATE: usize = 32;

    let x = xavier_uniform(vec![BATCH * SEQ, HIDDEN], 42, DType::F32).unwrap();
    let x = x.reshape(vec![BATCH, SEQ, HIDDEN]).unwrap();

    // Init norm weights/biases.
    let attn_norm_w = Tensor::from_slice(&[1.0f32; HIDDEN], vec![HIDDEN]).unwrap();
    let ffn_norm_w = Tensor::from_slice(&[1.0f32; HIDDEN], vec![HIDDEN]).unwrap();
    let ffn_norm_b = Tensor::from_slice(&[0.0f32; HIDDEN], vec![HIDDEN]).unwrap();

    // Q/K/V/O projection weights.
    let w_q = xavier_uniform(vec![HIDDEN, HIDDEN], 1, DType::F32).unwrap();
    let w_k = xavier_uniform(vec![HIDDEN, HIDDEN], 2, DType::F32).unwrap();
    let w_v = xavier_uniform(vec![HIDDEN, HIDDEN], 3, DType::F32).unwrap();
    let w_o = xavier_uniform(vec![HIDDEN, HIDDEN], 4, DType::F32).unwrap();

    // FFN weights: up_gate has 2× intermediate (gate + value halves), down is intermediate→hidden.
    let w_up_gate = xavier_uniform(vec![HIDDEN, INTERMEDIATE * 2], 5, DType::F32).unwrap();
    let w_down = xavier_uniform(vec![INTERMEDIATE, HIDDEN], 6, DType::F32).unwrap();

    // RoPE freqs.
    let (cos, sin) = precompute_rope_freqs(SEQ, HEAD_DIM, 10000.0).unwrap();

    // ── Attention sub-block ──
    let normed = rms_norm(&x, &attn_norm_w, 1e-6).unwrap();
    let q = linear(&normed, &w_q, None).unwrap();
    let k = linear(&normed, &w_k, None).unwrap();
    let v = linear(&normed, &w_v, None).unwrap();

    // Apply RoPE per-head — reshape to [B*N_HEADS, S, head_dim] for rope, then back.
    // rope expects [..., seq, head_dim].
    let q = q.reshape(vec![BATCH, SEQ, N_HEADS, HEAD_DIM]).unwrap();
    let k = k.reshape(vec![BATCH, SEQ, N_HEADS, HEAD_DIM]).unwrap();
    // Permute to [B, n_heads, S, head_dim] then collapse leading.
    let q = q.transpose(1, 2).unwrap().contiguous().unwrap();
    let k = k.transpose(1, 2).unwrap().contiguous().unwrap();
    let q = rope(&q, &cos, &sin, HEAD_DIM).unwrap();
    let k = rope(&k, &cos, &sin, HEAD_DIM).unwrap();
    // Permute back to [B, S, n_heads, head_dim] → reshape to [B, S, hidden].
    let q = q.transpose(1, 2).unwrap().contiguous().unwrap().reshape(vec![BATCH, SEQ, HIDDEN]).unwrap();
    let k = k.transpose(1, 2).unwrap().contiguous().unwrap().reshape(vec![BATCH, SEQ, HIDDEN]).unwrap();

    let attn_out = multi_head_attention(&q, &k, &v, N_HEADS, /*causal=*/ true).unwrap();
    let attn_out_proj = linear(&attn_out, &w_o, None).unwrap();

    // Residual add.
    let mid = add(&x, &attn_out_proj).unwrap();
    assert_eq!(mid.shape(), &[BATCH, SEQ, HIDDEN]);

    // ── FFN sub-block ──
    let normed = layer_norm(&mid, &ffn_norm_w, &ffn_norm_b, 1e-6).unwrap();
    let up_gate = linear(&normed, &w_up_gate, None).unwrap();
    // swiglu expects [..., 2K] → [..., K]. Output: [B, S, INTERMEDIATE].
    let hidden_states = swiglu(&up_gate).unwrap();
    assert_eq!(hidden_states.shape(), &[BATCH, SEQ, INTERMEDIATE]);

    let down = linear(&hidden_states, &w_down, None).unwrap();
    let out = add(&mid, &down).unwrap();
    assert_eq!(out.shape(), &[BATCH, SEQ, HIDDEN]);

    // Output should have finite values (smoke check).
    for v in read_f32(&out) {
        assert!(v.is_finite(), "got non-finite: {v}");
    }
}
