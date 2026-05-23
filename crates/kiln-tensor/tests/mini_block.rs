//! Mini transformer block integration test.
//!
//! Wires the Phase 1 op families into a tiny end-to-end "forward step":
//!
//! ```text
//! embed(token_ids)
//!   → rmsnorm
//!   → matmul(weight) (toy QK)
//!   → rope
//!   → softmax_last_dim
//!   → matmul(weight) (toy V)
//!   → silu_mul (toy gated)
//!   → add (residual)
//! ```
//!
//! Catches API drift between ops, demonstrates the `dispatch{1,2,3}`
//! patterns in a realistic context, and gives Phase 2+ contributors a
//! concrete usage example to follow when porting per-backend impls.
//!
//! Shapes are tiny (hidden=4, seq=2, vocab=5) so the test is fast and
//! readable. Numerical correctness is asserted via finiteness + sum-of-
//! probabilities checks — exact values would over-constrain the test
//! to the F32 rounding sequence and risk false failures under future
//! kernel micro-optimizations.

use kiln_tensor as kt;
use kt::ops::{add, argmax_last_dim, cast, embedding, l2_norm, matmul, mul_sigmoid_gate, rms_norm, rope, silu, softmax_last_dim};

#[test]
fn mini_block_runs_end_to_end_cpu() {
    // ── Setup ────────────────────────────────────────────────────────
    let vocab = 5;
    let hidden = 4;
    let seq = 2;

    // Embed weights: [vocab=5, hidden=4]
    let embed_w: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.1).collect();
    let embed_w = kt::Tensor::from_slice(&embed_w, vec![vocab, hidden]).unwrap();

    // Token ids: [seq=2]
    let token_ids = kt::Tensor::from_slice(&[2i64, 0], vec![seq]).unwrap();

    // RMSNorm weight + linear projection weights.
    let norm_w = kt::Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![hidden]).unwrap();
    let proj_w =
        kt::Tensor::from_slice(&(0..hidden * hidden).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(), vec![hidden, hidden])
            .unwrap();
    let value_w =
        kt::Tensor::from_slice(&(0..hidden * hidden).map(|i| (i as f32) * 0.05 - 0.1).collect::<Vec<_>>(), vec![hidden, hidden])
            .unwrap();

    // RoPE cos/sin tables: [seq=2, rotary_dim/2 = 1]
    let cos = kt::Tensor::from_slice(&[1.0f32, 0.5_f32.sqrt()], vec![seq, 1]).unwrap();
    let sin = kt::Tensor::from_slice(&[0.0f32, 0.5_f32.sqrt()], vec![seq, 1]).unwrap();

    // ── Forward ──────────────────────────────────────────────────────

    // Step 1: embed → [seq, hidden]
    let x = embedding(&embed_w, &token_ids).unwrap();
    assert_eq!(x.shape(), &[seq, hidden]);
    assert_eq!(x.dtype(), kt::DType::F32);

    // Step 2: RMSNorm (over hidden axis)
    let x = rms_norm(&x, &norm_w, 1e-6).unwrap();
    assert_eq!(x.shape(), &[seq, hidden]);

    // Step 3: linear projection → toy QK
    let qk = matmul(&x, &proj_w).unwrap();
    assert_eq!(qk.shape(), &[seq, hidden]);

    // Step 4: RoPE (partial-rotary 2 of 4)
    let qk = rope(&qk, &cos, &sin, 2).unwrap();
    assert_eq!(qk.shape(), &[seq, hidden]);

    // Step 5: softmax_last_dim — attention probabilities along hidden
    let attn = softmax_last_dim(&qk).unwrap();
    assert_eq!(attn.shape(), &[seq, hidden]);
    // Each row should sum to ~1.
    let attn_cpu = attn
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let attn_f: Vec<f32> = bytemuck::cast_slice::<u8, f32>(attn_cpu.as_bytes()).to_vec();
    for row in 0..seq {
        let s: f32 = attn_f[row * hidden..(row + 1) * hidden].iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row {row} sum {s} != 1");
    }

    // Step 6: value matmul
    let v = matmul(&attn, &value_w).unwrap();
    assert_eq!(v.shape(), &[seq, hidden]);

    // Step 7: silu_mul (gate=attn, up=v)
    let mlp = mul_sigmoid_gate(&attn, &v).unwrap();
    assert_eq!(mlp.shape(), &[seq, hidden]);

    // Step 8: residual add (x + mlp)
    let out = add(&x, &mlp).unwrap();
    assert_eq!(out.shape(), &[seq, hidden]);
    assert_eq!(out.dtype(), kt::DType::F32);

    // ── Final assertions ─────────────────────────────────────────────
    // Every output element must be finite (no NaN / Inf from any op).
    let out_cpu = out
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let out_f: Vec<f32> = bytemuck::cast_slice::<u8, f32>(out_cpu.as_bytes()).to_vec();
    for (i, v) in out_f.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} is not finite");
    }

    // argmax_last_dim sanity: output should be I64, shape = [seq], all
    // indices < hidden.
    let argmax = argmax_last_dim(&out).unwrap();
    assert_eq!(argmax.dtype(), kt::DType::I64);
    assert_eq!(argmax.shape(), &[seq]);
    let argmax_cpu = argmax
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let ids: Vec<i64> = bytemuck::cast_slice::<u8, i64>(argmax_cpu.as_bytes()).to_vec();
    for &id in &ids {
        assert!(
            (0..hidden as i64).contains(&id),
            "argmax id {id} out of range"
        );
    }
}

#[test]
fn cast_round_trip_through_bf16_preserves_finiteness() {
    // Exercise the cast op in a flow: F32 -> BF16 -> RMSNorm -> F32.
    let x = kt::Tensor::from_slice(
        &[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0],
        vec![2, 3],
    )
    .unwrap();
    let w = kt::Tensor::from_slice(&[1.0f32, 1.0, 1.0], vec![3]).unwrap();

    let x_bf = cast(&x, kt::DType::BF16).unwrap();
    let w_bf = cast(&w, kt::DType::BF16).unwrap();
    let normed_bf = rms_norm(&x_bf, &w_bf, 1e-6).unwrap();
    let normed_f = cast(&normed_bf, kt::DType::F32).unwrap();
    assert_eq!(normed_f.dtype(), kt::DType::F32);

    let cpu = normed_f
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
    for &v in &back {
        assert!(v.is_finite(), "got non-finite {v}");
    }
}

#[test]
fn l2_norm_then_silu_is_finite() {
    // Validates that the L2-normed output stays in silu's well-behaved range.
    let x = kt::Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0], vec![1, 4]).unwrap();
    let normed = l2_norm(&x, 0.0).unwrap();
    let after = silu(&normed).unwrap();
    let cpu = after
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
    for v in back {
        assert!(v.is_finite());
    }
}
