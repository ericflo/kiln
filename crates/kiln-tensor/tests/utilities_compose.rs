//! Integration tests for the Phase 1.81+ utility ops.
//!
//! Exercises top_k → one_hot → cross_entropy chains, init → norm
//! chains, and the random ctors compose cleanly with the main
//! op surface.

use kiln_tensor::ops::{
    arange, clip_grad_norm, cosine_similarity, cross_entropy, frobenius_norm, kaiming_normal,
    layer_norm, linspace, log_softmax_last_dim, mse_loss, nll_loss, one_hot, rand_normal,
    rand_uniform, repeat, sin, top_k, xavier_uniform,
};
use kiln_tensor::{CpuStorage, DType, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i64(t: &Tensor) -> Vec<i64> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn scalar_f32(t: &Tensor) -> f32 {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
}

#[test]
fn top_k_feeds_one_hot_for_routing() {
    // Toy MoE expert routing: pick top-1 expert from scores; convert
    // to one-hot for downstream masking.
    let scores = Tensor::from_slice(
        &[
            0.1f32, 0.2, 0.7, 0.0, // sample 0 → expert 2
            0.5, 0.4, 0.0, 0.1, // sample 1 → expert 0
            0.0, 0.6, 0.3, 0.1, // sample 2 → expert 1
        ],
        vec![3, 4],
    )
    .unwrap();
    let (_, indices) = top_k(&scores, 1).unwrap();
    assert_eq!(read_i64(&indices), vec![2, 0, 1]);
    // Squeeze the trailing-1 axis to feed into one_hot.
    let idx_1d = indices.reshape(vec![3]).unwrap();
    let one_hot_mask = one_hot(&idx_1d, /*depth=*/ 4, DType::F32).unwrap();
    assert_eq!(one_hot_mask.shape(), &[3, 4]);
    // Row 0 should be [0, 0, 1, 0]; row 1 [1, 0, 0, 0]; row 2 [0, 1, 0, 0].
    assert_eq!(
        read_f32(&one_hot_mask),
        vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn log_softmax_plus_nll_matches_cross_entropy_on_realistic_logits() {
    // Real-shape logits + targets — sanity check the
    // mathematical-equivalence chain holds beyond toy inputs.
    let logits = rand_normal(vec![8, 16], 0.0, 1.0, 7, DType::F32).unwrap();
    let targets = Tensor::from_slice(
        &[0i64, 5, 11, 2, 9, 14, 3, 7],
        vec![8],
    )
    .unwrap();
    let ce = scalar_f32(&cross_entropy(&logits, &targets).unwrap());
    let lp = log_softmax_last_dim(&logits).unwrap();
    let nll = scalar_f32(&nll_loss(&lp, &targets).unwrap());
    assert!((ce - nll).abs() < 1e-5, "ce={ce}, nll={nll}");
}

#[test]
fn xavier_init_norm_is_bounded() {
    // Xavier initialization for [256, 512] — output Frobenius norm
    // should be ≈ √(out_count * 2/(fan_in + fan_out)) elements
    // squared, but bounded.
    let w = xavier_uniform(vec![256, 512], 42, DType::F32).unwrap();
    let fro = scalar_f32(&frobenius_norm(&w).unwrap());
    // 256*512 = 131072 elements, uniform in [-a, a] with a ≈ √(6/768) ≈ 0.0884.
    // Expected mean square ≈ a²/3 ≈ 0.0026; sum ≈ 343; sqrt ≈ 18.5.
    assert!(fro > 10.0 && fro < 30.0, "fro={fro} outside expected range");
}

#[test]
fn clip_grad_norm_keeps_post_below_target() {
    let g1 = rand_uniform(vec![100], -5.0, 5.0, 1, DType::F32).unwrap();
    let g2 = rand_uniform(vec![50], -5.0, 5.0, 2, DType::F32).unwrap();
    let (norm, clipped) = clip_grad_norm(&[&g1, &g2], 0.5).unwrap();
    // Post-clip norm:
    let post_sq: f32 = read_f32(&clipped[0]).iter().map(|v| v * v).sum::<f32>()
        + read_f32(&clipped[1]).iter().map(|v| v * v).sum::<f32>();
    let post = post_sq.sqrt();
    assert!(
        post <= 0.51,
        "pre-clip norm {norm}, post-clip norm {post} > 0.5"
    );
}

#[test]
fn cosine_similarity_normalized_lookup() {
    // Normalize two batches; check cosine similarity matches
    // dot product of the normalized vectors.
    let q = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
    let k = Tensor::from_slice(&[4.0f32, 5.0, 6.0], vec![1, 3]).unwrap();
    let sim = scalar_f32(&cosine_similarity(&q, &k, 1e-8).unwrap());
    // Hand-computed: dot = 4+10+18 = 32; norms = √14 * √77 = √1078 ≈ 32.83;
    // sim ≈ 0.9746.
    assert!((sim - 0.9746).abs() < 1e-3, "sim={sim}");
}

#[test]
fn positional_encoding_pattern() {
    // Use arange + sin to build a sinusoidal positional encoding
    // 1-channel test slice.
    let positions = arange(0.0, 8.0, 1.0, DType::F32).unwrap();
    let encoded = sin(&positions).unwrap();
    let v = read_f32(&encoded);
    // sin(0) = 0; sin(7) ≈ 0.6570.
    assert!(v[0].abs() < 1e-6);
    assert!((v[7] - 7.0f32.sin()).abs() < 1e-5);
}

#[test]
fn linspace_layernorm_pipeline() {
    // linspace input through LayerNorm — verifies that the standard
    // construction primitives compose with the substrate ops.
    let x = linspace(-1.0, 1.0, 8, DType::F32).unwrap();
    let x2 = x.reshape(vec![1, 8]).unwrap();
    let w = Tensor::from_slice(&[1.0f32; 8], vec![8]).unwrap();
    let b = Tensor::from_slice(&[0.0f32; 8], vec![8]).unwrap();
    let y = layer_norm(&x2, &w, &b, 1e-6).unwrap();
    // Sum of LayerNorm output should be 0 (mean subtraction makes it
    // mean-0; weight=ones bias=zeros preserves that).
    let s: f32 = read_f32(&y).iter().sum();
    assert!(s.abs() < 1e-3);
}

#[test]
fn repeat_kaiming_init_for_replicated_experts() {
    // Init a single-expert weight, then repeat 4x along axis 0 to
    // build a 4-expert MoE.
    let w = kaiming_normal(vec![16, 32], 42, DType::F32).unwrap();
    let w2 = w.reshape(vec![1, 16, 32]).unwrap();
    let experts = repeat(&w2, /*axis=*/ 0, /*n=*/ 4).unwrap();
    assert_eq!(experts.shape(), &[4, 16, 32]);
    // Every expert should be identical to the original.
    let single = read_f32(&w);
    let all = read_f32(&experts);
    for e in 0..4 {
        let start = e * 16 * 32;
        let end = start + 16 * 32;
        assert_eq!(&all[start..end], &single[..], "expert {e} differs");
    }
}

#[test]
fn mse_train_step_via_substrate() {
    // Pred vs target: prove mse_loss > 0 and decreases when pred → target.
    let target = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
    let pred_bad = Tensor::from_slice(&[0.0f32, 0.0, 0.0], vec![3]).unwrap();
    let pred_good = Tensor::from_slice(&[0.9f32, 1.9, 2.9], vec![3]).unwrap();
    let l_bad = scalar_f32(&mse_loss(&pred_bad, &target).unwrap());
    let l_good = scalar_f32(&mse_loss(&pred_good, &target).unwrap());
    assert!(
        l_good < l_bad,
        "mse should decrease when pred → target: good={l_good} >= bad={l_bad}"
    );
}
