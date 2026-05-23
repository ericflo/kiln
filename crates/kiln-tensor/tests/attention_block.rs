//! Causal attention block integration test.
//!
//! Wires `matmul`, `causal_mask`, `masked_fill`, `softmax_last_dim`,
//! and `matmul` into a real attention block (single-head, F32, no
//! RoPE). Verifies that the masking pattern correctly nullifies
//! future positions before softmax + that the full QKV → output
//! pipeline produces finite, sum-to-1 attention weights.
//!
//! This is the canonical CPU reference attention path; Phase 3's
//! per-backend attention impls (FlashAttention-2 on CUDA, MPS-SDPA
//! on Metal, kiln-vulkan-kernel's vk_ops::attention on Vulkan) all
//! parity-test against this.

use kiln_tensor as kt;
use kt::ops::{causal_mask, masked_fill, matmul, mean_all, softmax_last_dim};

/// Read a tensor's F32 contents (assumes contiguous F32 storage).
fn read_f32(t: &kt::Tensor) -> Vec<f32> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec()
}

fn approx(a: f32, b: f32, atol: f32) -> bool {
    (a - b).abs() <= atol
}

#[test]
fn single_head_causal_attention_cpu() {
    // Shapes:
    //   seq_len = 4, head_dim = 2
    //   Q, K, V: [seq_len=4, head_dim=2]
    let seq_len = 4;
    let head_dim = 2;

    let q = kt::Tensor::from_slice(
        &[1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![seq_len, head_dim],
    )
    .unwrap();
    let k = q.clone(); // self-attention setup; K == Q
    let v = kt::Tensor::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        vec![seq_len, head_dim],
    )
    .unwrap();

    // 1. Compute scores = Q @ K^T   ([seq, head_dim] @ [head_dim, seq] = [seq, seq])
    let kt_transposed = k.transpose(0, 1).unwrap();
    let kt_contig = kt_transposed.contiguous().unwrap();
    let scores = matmul(&q, &kt_contig).unwrap();
    assert_eq!(scores.shape(), &[seq_len, seq_len]);

    // 2. Apply causal mask (no scale here; toy values are small enough).
    let mask = causal_mask(seq_len).unwrap();
    let masked = masked_fill(&scores, &mask, f32::NEG_INFINITY).unwrap();

    // 3. Softmax along the last axis.
    let attn = softmax_last_dim(&masked).unwrap();
    assert_eq!(attn.shape(), &[seq_len, seq_len]);

    // 4. Verify each row sums to ~1 (sanity check the softmax path).
    let attn_f = read_f32(&attn);
    for row in 0..seq_len {
        let s: f32 = attn_f[row * seq_len..(row + 1) * seq_len].iter().sum();
        assert!(
            approx(s, 1.0, 1e-5),
            "row {row} sums to {s}, expected 1.0 (attn row={:?})",
            &attn_f[row * seq_len..(row + 1) * seq_len]
        );
    }

    // 5. Verify the causal mask zeroed future positions: attn[i, j] == 0 for j > i.
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            let v = attn_f[i * seq_len + j];
            assert!(
                approx(v, 0.0, 1e-7),
                "future-position attn[{i}, {j}] = {v}, expected 0 after causal mask"
            );
        }
    }

    // 6. Row 0 attends only to position 0 → softmax([x, -inf, -inf, -inf]) = [1, 0, 0, 0]
    assert!(approx(attn_f[0], 1.0, 1e-5));

    // 7. Output = attn @ V   ([seq, seq] @ [seq, head_dim] = [seq, head_dim])
    let out = matmul(&attn, &v).unwrap();
    assert_eq!(out.shape(), &[seq_len, head_dim]);

    // 8. Row 0 output must equal V[0] (since attn row 0 is [1, 0, 0, 0]).
    let out_f = read_f32(&out);
    assert!(approx(out_f[0], 10.0, 1e-5)); // V[0, 0]
    assert!(approx(out_f[1], 20.0, 1e-5)); // V[0, 1]

    // 9. Every output element must be finite.
    for (i, v) in out_f.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] = {v} non-finite");
    }
}

#[test]
fn attention_uniform_when_all_keys_equal() {
    // If Q is any vector and K is the all-ones key tensor, every
    // score is equal -> softmax distributes uniformly over the
    // unmasked positions on each row (1/(i+1) for the i-th row).
    let seq_len = 3;
    let head_dim = 2;
    let q = kt::Tensor::from_slice(&[1.0f32; 6], vec![seq_len, head_dim]).unwrap();
    let k = kt::Tensor::from_slice(&[1.0f32; 6], vec![seq_len, head_dim]).unwrap();

    let kt_t = k.transpose(0, 1).unwrap().contiguous().unwrap();
    let scores = matmul(&q, &kt_t).unwrap();
    let mask = causal_mask(seq_len).unwrap();
    let masked = masked_fill(&scores, &mask, f32::NEG_INFINITY).unwrap();
    let attn = softmax_last_dim(&masked).unwrap();
    let attn_f = read_f32(&attn);

    // Row 0: [1, 0, 0]
    assert!(approx(attn_f[0], 1.0, 1e-5));
    assert!(approx(attn_f[1], 0.0, 1e-7));
    assert!(approx(attn_f[2], 0.0, 1e-7));
    // Row 1: [0.5, 0.5, 0]
    assert!(approx(attn_f[3], 0.5, 1e-5));
    assert!(approx(attn_f[4], 0.5, 1e-5));
    assert!(approx(attn_f[5], 0.0, 1e-7));
    // Row 2: [1/3, 1/3, 1/3]
    assert!(approx(attn_f[6], 1.0 / 3.0, 1e-5));
    assert!(approx(attn_f[7], 1.0 / 3.0, 1e-5));
    assert!(approx(attn_f[8], 1.0 / 3.0, 1e-5));
}

#[test]
fn mean_all_summary_stat_after_attention() {
    // Confirm `mean_all` composes with the attention block — useful
    // smoke test that the new Phase 1.33 reduction op fits the rest
    // of the pipeline.
    let q = kt::Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let k = q.clone();
    let v = kt::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

    let kt_t = k.transpose(0, 1).unwrap().contiguous().unwrap();
    let scores = matmul(&q, &kt_t).unwrap();
    let mask = causal_mask(2).unwrap();
    let masked = masked_fill(&scores, &mask, f32::NEG_INFINITY).unwrap();
    let attn = softmax_last_dim(&masked).unwrap();
    let out = matmul(&attn, &v).unwrap();
    let mean = mean_all(&out).unwrap();
    assert_eq!(mean.rank(), 0);
    let cpu = mean.storage().as_any().downcast_ref::<kt::CpuStorage>().unwrap();
    let v = f32::from_le_bytes(cpu.as_bytes()[0..4].try_into().unwrap());
    assert!(v.is_finite());
}
