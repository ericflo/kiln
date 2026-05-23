//! Full sampler chain integration test.
//!
//! Composes the Phase 1.48 / 1.49 LogitProcessor chain on the
//! Qwen3.5-4B-style production sampler order:
//!
//! ```text
//! penalty_repetition → penalty_frequency → penalty_presence
//!                    → temperature → top_k → top_p
//!                    → softmax_last_dim
//!                    → argmax_last_dim   (greedy decode tap)
//! ```
//!
//! Each unit test focuses on one composition property; together they
//! validate the full Phase 4 sampler design from the #1082 issue.

use kiln_tensor as kt;
use kt::ops::{argmax_last_dim, softmax_last_dim};
use kt::ops::logit_penalties::{
    FrequencyPenaltyProcessor, PresencePenaltyProcessor, RepetitionPenaltyProcessor,
};
use kt::ops::logit_processor::{
    LogitProcessorChain, TemperatureProcessor, TopKProcessor, TopPProcessor,
};

fn read_rows(t: &kt::Tensor, batch: usize, vocab: usize) -> Vec<Vec<f32>> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let per = t.dtype().size_in_bytes();
    let mut rows = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = Vec::with_capacity(vocab);
        for v in 0..vocab {
            let off = (b * vocab + v) * per;
            row.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        }
        rows.push(row);
    }
    rows
}

fn read_i64(t: &kt::Tensor) -> Vec<i64> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<kt::CpuStorage>()
        .unwrap();
    bytemuck::cast_slice::<u8, i64>(cpu.as_bytes()).to_vec()
}

#[test]
fn greedy_chain_picks_unmasked_top_logit() {
    // [1.0, 5.0, 3.0, 5.5, 2.0] with top-K(2) keeps only [5.0, 5.5]
    // (indices 1 and 3). argmax picks index 3 (the larger).
    let logits = kt::Tensor::from_slice(&[1.0f32, 5.0, 3.0, 5.5, 2.0], vec![1, 5]).unwrap();
    let chain =
        LogitProcessorChain::new(vec![Box::new(TopKProcessor::new(2))]);
    let post = chain.apply(&logits).unwrap();
    let ids = read_i64(&argmax_last_dim(&post).unwrap());
    assert_eq!(ids, vec![3]);
}

#[test]
fn temperature_scaling_does_not_change_argmax() {
    // Temperature scaling preserves the ranking — argmax is invariant.
    let logits = kt::Tensor::from_slice(&[1.0f32, 5.0, 3.0], vec![1, 3]).unwrap();
    let chain = LogitProcessorChain::new(vec![Box::new(TemperatureProcessor::new(0.5))]);
    let post = chain.apply(&logits).unwrap();
    let ids = read_i64(&argmax_last_dim(&post).unwrap());
    assert_eq!(ids, vec![1]);
}

#[test]
fn full_chain_pipeline_produces_finite_softmax() {
    // Run the canonical Qwen3.5-4B chain on a toy batch. Verifies
    // every stage composes cleanly and softmax produces a valid prob
    // distribution.
    let vocab = 6;
    let batch = 2;
    let logits = kt::Tensor::from_slice(
        &[
            // batch 0: ascending
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0,
            // batch 1: peaked at index 2
            0.5, 0.5, 8.0, 0.5, 0.5, 0.5,
        ],
        vec![batch, vocab],
    )
    .unwrap();
    let history = vec![
        vec![5u32], // batch 0 has seen index 5
        vec![],     // batch 1 has no history
    ];
    let chain = LogitProcessorChain::new(vec![
        Box::new(RepetitionPenaltyProcessor::new(2.0, history.clone())),
        Box::new(FrequencyPenaltyProcessor::new(0.1, history.clone())),
        Box::new(PresencePenaltyProcessor::new(0.05, history)),
        Box::new(TemperatureProcessor::new(0.7)),
        Box::new(TopKProcessor::new(3)),
        Box::new(TopPProcessor::new(0.95)),
    ]);
    let post = chain.apply(&logits).unwrap();
    let probs = softmax_last_dim(&post).unwrap();
    let rows = read_rows(&probs, batch, vocab);
    for (b, row) in rows.iter().enumerate() {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "row {b} probabilities sum to {sum}, expected ~1"
        );
        for &v in row {
            assert!(
                v >= 0.0 && v.is_finite(),
                "row {b} contains negative or non-finite prob: {v}"
            );
        }
    }
}

#[test]
fn repetition_penalty_changes_argmax_when_strong_enough() {
    // logits [10.0, 11.0]; history [1] with penalty 2.0 →
    // logits[1] = 11.0 / 2.0 = 5.5; argmax now index 0 (10.0 > 5.5).
    let logits = kt::Tensor::from_slice(&[10.0f32, 11.0], vec![1, 2]).unwrap();
    let chain = LogitProcessorChain::new(vec![Box::new(
        RepetitionPenaltyProcessor::new(2.0, vec![vec![1]]),
    )]);
    let post = chain.apply(&logits).unwrap();
    let ids = read_i64(&argmax_last_dim(&post).unwrap());
    assert_eq!(ids, vec![0], "rep penalty should flip argmax 1 → 0");
}

#[test]
fn top_p_then_softmax_concentrates_probability() {
    // Without top_p: softmax([1, 2, 3, 4]) is a smooth distribution.
    // With top_p(0.5): only the top entries survive (mask others with
    // -inf), so softmax over the survivors is more concentrated.
    let logits = kt::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();

    let no_chain = LogitProcessorChain::empty();
    let baseline_probs = softmax_last_dim(&no_chain.apply(&logits).unwrap()).unwrap();
    let baseline = read_rows(&baseline_probs, 1, 4);

    let chain = LogitProcessorChain::new(vec![Box::new(TopPProcessor::new(0.5))]);
    let post = chain.apply(&logits).unwrap();
    let probs = softmax_last_dim(&post).unwrap();
    let rows = read_rows(&probs, 1, 4);

    // After top_p(0.5), the max probability should be strictly higher
    // (the mass that was on masked indices is redistributed onto the
    // remaining ones).
    let max_baseline = baseline[0]
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let max_filtered = rows[0].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    assert!(
        max_filtered > max_baseline,
        "top_p should concentrate probability (baseline max={max_baseline}, filtered max={max_filtered})"
    );
}

#[test]
fn presence_then_temperature_then_topk_composes() {
    // logits [1, 2, 3, 4]; presence 0.5 on history [3] subtracts 0.5
    // from logits[3] → [1, 2, 3, 3.5]; temp=1 (no-op); top_k=2 →
    // [-inf, -inf, 3, 3.5]; argmax picks 3.
    let logits = kt::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    let chain = LogitProcessorChain::new(vec![
        Box::new(PresencePenaltyProcessor::new(0.5, vec![vec![3]])),
        Box::new(TemperatureProcessor::new(1.0)),
        Box::new(TopKProcessor::new(2)),
    ]);
    let post = chain.apply(&logits).unwrap();
    let ids = read_i64(&argmax_last_dim(&post).unwrap());
    assert_eq!(ids, vec![3]);
    let rows = read_rows(&post, 1, 4);
    assert!(rows[0][0].is_infinite() && rows[0][0] < 0.0);
    assert!(rows[0][1].is_infinite() && rows[0][1] < 0.0);
    assert!((rows[0][2] - 3.0).abs() < 1e-6);
    assert!((rows[0][3] - 3.5).abs() < 1e-6);
}

#[test]
fn empty_chain_softmax_argmax_path() {
    // Smoke: empty chain → softmax → argmax produces the expected
    // index even with no transformations.
    let logits = kt::Tensor::from_slice(&[0.5f32, 1.5, 1.0], vec![1, 3]).unwrap();
    let chain = LogitProcessorChain::empty();
    let post = chain.apply(&logits).unwrap();
    let probs = softmax_last_dim(&post).unwrap();
    let rows = read_rows(&probs, 1, 3);
    let sum: f32 = rows[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    let ids = read_i64(&argmax_last_dim(&post).unwrap());
    assert_eq!(ids, vec![1]);
}

#[test]
fn chain_with_multi_batch_independent_decisions() {
    // Multi-batch: penalties only affect the row they're configured
    // for. Verifies independent histories produce independent argmax.
    let logits = kt::Tensor::from_slice(
        &[10.0f32, 11.0, 11.0, 10.0],
        vec![2, 2],
    )
    .unwrap();
    let chain = LogitProcessorChain::new(vec![Box::new(
        RepetitionPenaltyProcessor::new(2.0, vec![vec![1], vec![0]]),
    )]);
    let post = chain.apply(&logits).unwrap();
    let ids = read_i64(&argmax_last_dim(&post).unwrap());
    // Batch 0: history [1] → logits[0,1] /= 2 → [10.0, 5.5] → argmax 0
    // Batch 1: history [0] → logits[1,0] /= 2 → [5.5, 10.0] → argmax 1
    assert_eq!(ids, vec![0, 1]);
}
