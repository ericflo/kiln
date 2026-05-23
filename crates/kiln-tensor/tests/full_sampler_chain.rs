//! End-to-end Phase 4 sampler chain integration test.
//!
//! Exercises **every** LogitProcessor produced in Phases 1.48–1.57
//! plus the GumbelSampler from Phase 1.58 as a single pipeline.
//! The intent is to prove the contract that the full menu composes:
//!
//! 1. penalties (repetition + frequency + presence) — Phase 1.49
//! 2. DRY — Phase 1.56
//! 3. ngram-block — Phase 1.54
//! 4. logit-bias — Phase 1.54
//! 5. temperature — Phase 1.48
//! 6. top-K — Phase 1.48
//! 7. top-P — Phase 1.48
//! 8. min-P — Phase 1.53
//! 9. typical-P — Phase 1.53
//! 10. Mirostat 2 — Phase 1.55
//! 11. XTC — Phase 1.57
//! 12. GumbelSampler — Phase 1.58 (terminal)
//!
//! These tests are CPU-only and run against the canonical
//! `kiln_tensor::ops::*` surface.

use std::collections::{HashMap, HashSet};

use kiln_tensor::ops::logit_dry::DryProcessor;
use kiln_tensor::ops::logit_mirostat::Mirostat2Processor;
use kiln_tensor::ops::logit_misc::{LogitBiasProcessor, NgramBlockProcessor};
use kiln_tensor::ops::logit_modern::{MinPProcessor, TypicalPProcessor};
use kiln_tensor::ops::logit_penalties::{
    FrequencyPenaltyProcessor, PresencePenaltyProcessor, RepetitionPenaltyProcessor,
};
use kiln_tensor::ops::logit_processor::{
    LogitProcessor, LogitProcessorChain, TemperatureProcessor, TopKProcessor, TopPProcessor,
};
use kiln_tensor::ops::logit_xtc::XtcProcessor;
use kiln_tensor::ops::GumbelSampler;
use kiln_tensor::{CpuStorage, DType, Tensor};

fn read_i64(t: &Tensor) -> Vec<i64> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build a "kitchen sink" chain that exercises every Phase 4 step.
fn make_full_chain(history: Vec<u32>) -> LogitProcessorChain {
    let mut bias = HashMap::new();
    bias.insert(0u32, 0.1); // tiny nudge on token 0
    let mut breakers = HashSet::new();
    breakers.insert(2u32);

    LogitProcessorChain::new(vec![
        // ── penalties ──
        Box::new(RepetitionPenaltyProcessor::new(
            1.05,
            vec![history.clone()],
        )),
        Box::new(FrequencyPenaltyProcessor::new(0.1, vec![history.clone()])),
        Box::new(PresencePenaltyProcessor::new(0.1, vec![history.clone()])),
        // ── DRY ──
        Box::new(
            DryProcessor::new(0.5, 1.75, 3, vec![history.clone()])
                .with_sequence_breakers(breakers),
        ),
        // ── ngram-block ──
        Box::new(NgramBlockProcessor::new(2, vec![history])),
        // ── logit_bias ──
        Box::new(LogitBiasProcessor::new(bias)),
        // ── temperature ──
        Box::new(TemperatureProcessor::new(0.9)),
        // ── top-K ──
        Box::new(TopKProcessor::new(50)),
        // ── top-P ──
        Box::new(TopPProcessor::new(0.95)),
        // ── min-P ──
        Box::new(MinPProcessor::new(0.01)),
        // ── typical-P ──
        Box::new(TypicalPProcessor::new(0.95)),
        // ── Mirostat 2 ──
        Box::new(Mirostat2Processor::new(5.0, 0.1)),
        // ── XTC ──
        Box::new(XtcProcessor::with_seed(0.1, 0.0, 1)), // probability=0 → no-op
    ])
}

#[test]
fn full_chain_composes_and_samples_a_token() {
    // [B=1, V=32] random-ish logits.
    let mut raw = Vec::with_capacity(32);
    for i in 0..32 {
        raw.push((i as f32).sin() * 3.0);
    }
    let logits = Tensor::from_slice(&raw, vec![1, 32]).unwrap();
    let chain = make_full_chain(vec![5, 10, 15, 5, 10]);
    assert_eq!(chain.len(), 13);

    let masked = chain.apply(&logits).unwrap();
    assert_eq!(masked.shape(), &[1, 32]);
    assert_eq!(masked.dtype(), DType::F32);

    let sampler = GumbelSampler::with_seed(42);
    let ids = sampler.sample(&masked).unwrap();
    assert_eq!(ids.shape(), &[1]);
    let tok = read_i64(&ids)[0];
    assert!(
        (0..32).contains(&tok),
        "sampled token {tok} must be in [0, 32)"
    );
}

#[test]
fn full_chain_respects_logit_bias_neg_inf_ban() {
    // Push token 7 to -inf via bias. After the full chain, the
    // sampler must never pick it.
    let mut bias = HashMap::new();
    bias.insert(7u32, f32::NEG_INFINITY);

    let chain = LogitProcessorChain::new(vec![
        Box::new(LogitBiasProcessor::new(bias)),
        Box::new(TemperatureProcessor::new(1.0)),
        Box::new(TopKProcessor::new(20)),
    ]);

    let logits = Tensor::from_slice(&vec![3.0f32; 16], vec![1, 16]).unwrap();
    let masked = chain.apply(&logits).unwrap();

    let sampler = GumbelSampler::with_seed(11);
    for _ in 0..200 {
        let tok = read_i64(&sampler.sample(&masked).unwrap())[0];
        assert_ne!(tok, 7, "banned token slipped through");
    }
}

#[test]
fn full_chain_ngram_block_prevents_immediate_loop() {
    // History [1, 2]. n=2: pair (1)→2 seen. Tail = [2]. So if the
    // candidate is *any token* the pair (2, c) hasn't been seen
    // before, no block. But add candidate 2 again to seen by
    // building history that contains (2, 2): impossible with [1, 2].
    // Instead: history [1, 2, 1]. Pairs seen: (1, 2), (2, 1). Tail =
    // [1]. Candidate 2: (1, 2) seen → block 2.
    let chain = LogitProcessorChain::new(vec![Box::new(NgramBlockProcessor::new(
        2,
        vec![vec![1, 2, 1]],
    ))]);

    let logits = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![1, 4]).unwrap();
    let masked = chain.apply(&logits).unwrap();

    let sampler = GumbelSampler::with_seed(31);
    for _ in 0..200 {
        let tok = read_i64(&sampler.sample(&masked).unwrap())[0];
        assert_ne!(tok, 2, "ngram-blocked token 2 was sampled");
    }
}

#[test]
fn full_chain_multi_batch_independent_streams() {
    // 2 rows with different forced biases: row 0 must produce token 1,
    // row 1 must produce token 2.
    let mut bias_row0 = HashMap::new();
    bias_row0.insert(0u32, f32::NEG_INFINITY);
    bias_row0.insert(2u32, f32::NEG_INFINITY);
    bias_row0.insert(3u32, f32::NEG_INFINITY);
    // → token 1 is the only survivor
    let mut bias_row1 = HashMap::new();
    bias_row1.insert(0u32, f32::NEG_INFINITY);
    bias_row1.insert(1u32, f32::NEG_INFINITY);
    bias_row1.insert(3u32, f32::NEG_INFINITY);
    // → token 2 is the only survivor

    // LogitBiasProcessor is a single-bias-map type; for batched
    // forced sampling we run the chain per row.
    let logits =
        Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 4]).unwrap();

    // Apply per-row by slicing and reassembling.
    let bytes = logits
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .unwrap()
        .as_bytes()
        .to_vec();
    let row0 =
        Tensor::from_slice(bytemuck::cast_slice::<u8, f32>(&bytes[0..16]), vec![1, 4]).unwrap();
    let row1 =
        Tensor::from_slice(bytemuck::cast_slice::<u8, f32>(&bytes[16..32]), vec![1, 4]).unwrap();
    let c0 = LogitProcessorChain::new(vec![Box::new(LogitBiasProcessor::new(bias_row0))]);
    let c1 = LogitProcessorChain::new(vec![Box::new(LogitBiasProcessor::new(bias_row1))]);
    let m0 = c0.apply(&row0).unwrap();
    let m1 = c1.apply(&row1).unwrap();

    let sampler = GumbelSampler::with_seed(7);
    for _ in 0..50 {
        assert_eq!(read_i64(&sampler.sample(&m0).unwrap()), vec![1]);
        assert_eq!(read_i64(&sampler.sample(&m1).unwrap()), vec![2]);
    }
}

#[test]
fn full_chain_names_lists_all_processors_in_order() {
    let chain = make_full_chain(vec![1, 2, 3]);
    let names = chain.names();
    assert_eq!(
        names,
        vec![
            "penalty_repetition",
            "penalty_frequency",
            "penalty_presence",
            "dry",
            "ngram_block",
            "logit_bias",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "typical_p",
            "mirostat2",
            "xtc",
        ]
    );
}

#[test]
fn full_chain_bf16_propagates() {
    // BF16 logits should pass through every chain step and produce a
    // sample.
    let bf: Vec<half::bf16> = (0..16)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.1).sin()))
        .collect();
    let logits = Tensor::from_slice(&bf, vec![1, 16]).unwrap();

    let chain = LogitProcessorChain::new(vec![
        Box::new(TemperatureProcessor::new(0.7)),
        Box::new(TopKProcessor::new(8)),
    ]);
    let masked = chain.apply(&logits).unwrap();
    assert_eq!(masked.dtype(), DType::BF16);

    let sampler = GumbelSampler::with_seed(99);
    let tok = read_i64(&sampler.sample(&masked).unwrap())[0];
    assert!((0..16).contains(&tok));
}

#[test]
fn full_chain_empty_chain_then_sample_is_argmax_in_expectation() {
    // Empty chain → sampler reads raw logits.
    let chain = LogitProcessorChain::empty();
    let logits = Tensor::from_slice(&[1.0f32, 100.0, 1.0, 1.0], vec![1, 4]).unwrap();
    let masked = chain.apply(&logits).unwrap();
    let sampler = GumbelSampler::with_seed(123);
    let tok = read_i64(&sampler.sample(&masked).unwrap())[0];
    // With one logit of 100 vs the rest at 1, the Gumbel noise is
    // overwhelmed and token 1 is picked.
    assert_eq!(tok, 1);
}
