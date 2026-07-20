//! ROCm batched W8 LM-head sampling contract.
//!
//! Run with:
//! `cargo test -p kiln-tensor --features rocm --test rocm_w8_sample_batch`
#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{Device, Tensor};

fn no_rocm() -> bool {
    if kiln_tensor::rocm_is_available() {
        false
    } else {
        eprintln!("no ROCm device available; skipping batched W8 sampling test");
        true
    }
}

fn fixture() -> (Tensor, Tensor, Tensor) {
    // Dequantized rows are four basis projections, their positive mean, and
    // their negative mean. Every asserted winner has a wide score margin so
    // BF16 activation and W8A8 activation quantization cannot change it.
    let signed_weights: [[i8; 4]; 6] = [
        [10, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 10, 0],
        [0, 0, 0, 10],
        [3, 3, 3, 3],
        [-3, -3, -3, -3],
    ];
    let weights: Vec<u8> = signed_weights
        .into_iter()
        .flatten()
        .map(|value| value as u8)
        .collect();
    let activations = [
        [5.0f32, 1.0, 2.0, 0.0],
        [1.0, 4.0, 4.0, 0.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 2.0, 2.0, 2.0],
    ];
    let x = Tensor::from_vec_on(
        Device::Rocm(0),
        activations
            .into_iter()
            .flatten()
            .map(bf16::from_f32)
            .collect::<Vec<_>>(),
        vec![4, 1, 4],
    )
    .expect("upload activations");
    let w_q = Tensor::from_vec_on(Device::Rocm(0), weights, vec![6, 4]).expect("upload W8 weights");
    let scales =
        Tensor::from_vec_on(Device::Rocm(0), vec![0.1f32; 6], vec![6]).expect("upload W8 scales");
    (x, w_q, scales)
}

fn run(quantized_activations: bool) -> Vec<i64> {
    let (x, w_q, scales) = fixture();
    let call = if quantized_activations {
        kiln_tensor::rocm_w8a8_gemv_sample_batch_bf16
    } else {
        kiln_tensor::rocm_w8a16_gemv_sample_batch_bf16
    };
    let tokens = call(
        &x,
        &w_q,
        &scales,
        &[1],
        &[1],
        &[1],
        &[1.0, 2.0, 1.0, 1.0],
        &[0.0, 1.0, 0.0, 0.0],
        &[0.0; 4],
        &[0.0, 0.01, 0.1, 1.0],
        &[0, 4, 4, 5],
        &[1.0, 0.95, 1.0, 0.1],
        &[0.0, 0.0, 0.95, 0.0],
        &[11, 22, 33, 44],
    )
    .expect("batched W8 sample");
    tokens.to_vec1::<i64>().expect("read batch tokens")
}

#[derive(Clone, Copy)]
struct SamplingCase {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    seed: u64,
    repetition: f32,
    presence: f32,
    frequency: f32,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn uniform01(seed: u64) -> f32 {
    let random = splitmix64(seed);
    ((((random >> 40) & 0x00ff_ffff) as f32 + 0.5) * 5.960_464_5e-8).clamp(1.0e-7, 1.0 - 1.0e-7)
}

fn projected_logits(input: &[bf16], quantized_activations: bool) -> Vec<f32> {
    const WEIGHT: i32 = 100;
    const WEIGHT_SCALE: f32 = 0.01;
    if !quantized_activations {
        return input
            .iter()
            .map(|value| value.to_f32() * WEIGHT as f32 * WEIGHT_SCALE)
            .collect();
    }

    let max_abs = input
        .iter()
        .map(|value| value.to_f32().abs())
        .fold(0.0f32, f32::max);
    let activation_scale = if max_abs <= 1.0e-12 {
        1.0
    } else {
        max_abs / 127.0
    };
    input
        .iter()
        .map(|value| {
            let quantized = (value.to_f32() / activation_scale)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i32;
            (quantized * WEIGHT) as f32 * activation_scale * WEIGHT_SCALE
        })
        .collect()
}

fn cpu_sample(mut logits: Vec<f32>, history: &[(u32, u32)], case: SamplingCase) -> i64 {
    for &(token, count) in history {
        let logit = &mut logits[token as usize];
        if case.repetition != 1.0 {
            *logit = if *logit > 0.0 {
                *logit / case.repetition
            } else {
                *logit * case.repetition
            };
        }
        *logit -= case.presence;
        *logit -= case.frequency * count as f32;
    }

    let mut candidates = (0..logits.len()).collect::<Vec<_>>();
    candidates.sort_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    candidates.truncate(case.top_k as usize);

    let max_logit = logits[candidates[0]] / case.temperature;
    let mut probabilities = candidates
        .iter()
        .map(|&index| (logits[index] / case.temperature - max_logit).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>();
    for probability in &mut probabilities {
        *probability /= sum;
    }

    if case.min_p > 0.0 {
        let threshold = case.min_p * probabilities[0];
        for probability in &mut probabilities {
            if *probability < threshold {
                *probability = 0.0;
            }
        }
        let keep_sum = probabilities.iter().sum::<f32>();
        for probability in &mut probabilities {
            *probability /= keep_sum;
        }
    }

    if case.top_p > 0.0 && case.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut cutoff = probabilities.len();
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative >= case.top_p {
                cutoff = index + 1;
                break;
            }
        }
        probabilities[cutoff..].fill(0.0);
        let keep_sum = probabilities[..cutoff].iter().sum::<f32>();
        for probability in &mut probabilities[..cutoff] {
            *probability /= keep_sum;
        }
    }

    let random = uniform01(case.seed);
    let mut cumulative = 0.0f32;
    let mut sampled = candidates[0];
    for (&candidate, &probability) in candidates.iter().zip(&probabilities) {
        cumulative += probability;
        sampled = candidate;
        if random < cumulative {
            break;
        }
    }
    sampled as i64
}

fn oracle_fixture(rows: usize) -> (Tensor, Tensor, Tensor, Vec<bf16>) {
    const VOCAB: usize = 64;
    let base = (0..VOCAB)
        .map(|index| bf16::from_f32(3.875 - index as f32 * 0.125))
        .collect::<Vec<_>>();
    let activations = (0..rows)
        .flat_map(|_| base.iter().copied())
        .collect::<Vec<_>>();
    let x = Tensor::from_vec_on(Device::Rocm(0), activations, vec![rows, 1, VOCAB])
        .expect("upload oracle activations");
    let mut weights = vec![0u8; VOCAB * VOCAB];
    for index in 0..VOCAB {
        weights[index * VOCAB + index] = 100;
    }
    let w_q = Tensor::from_vec_on(Device::Rocm(0), weights, vec![VOCAB, VOCAB])
        .expect("upload oracle W8 weights");
    let scales = Tensor::from_vec_on(Device::Rocm(0), vec![0.01f32; VOCAB], vec![VOCAB])
        .expect("upload oracle W8 scales");
    (x, w_q, scales, base)
}

fn oracle_cases() -> ([SamplingCase; 6], [Vec<(u32, u32)>; 6]) {
    let cases = [
        SamplingCase {
            temperature: 0.80,
            top_k: 6,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0x0000_0001_1234_5678,
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
        },
        SamplingCase {
            temperature: 1.25,
            top_k: 8,
            top_p: 0.68,
            min_p: 0.0,
            seed: 0x0000_0102_89ab_cdef,
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
        },
        SamplingCase {
            temperature: 0.95,
            top_k: 9,
            top_p: 1.0,
            min_p: 0.24,
            seed: 0x1357_2468_dead_beef,
            repetition: 1.0,
            presence: 0.0,
            frequency: 0.0,
        },
        SamplingCase {
            temperature: 1.10,
            top_k: 20,
            top_p: 0.83,
            min_p: 0.08,
            seed: 0xfedc_ba98_7654_3210,
            repetition: 1.35,
            presence: 0.30,
            frequency: 0.17,
        },
        SamplingCase {
            temperature: 1.65,
            top_k: 64,
            top_p: 0.91,
            min_p: 0.03,
            seed: 0,
            repetition: 0.85,
            presence: -0.10,
            frequency: 0.07,
        },
        SamplingCase {
            temperature: 0.55,
            top_k: 5,
            top_p: 0.74,
            min_p: 0.12,
            seed: 0xaaaa_5555_0102_0304,
            repetition: 1.20,
            presence: 0.15,
            frequency: -0.04,
        },
    ];
    let histories = [
        vec![],
        vec![],
        vec![],
        vec![(0, 3), (2, 1), (7, 4)],
        vec![(1, 2), (8, 5)],
        vec![(0, 1), (3, 6), (11, 2)],
    ];
    (cases, histories)
}

fn run_oracle_batch(rows: usize, quantized_activations: bool) -> (Vec<i64>, Vec<i64>) {
    let (cases, histories) = oracle_cases();
    let (x, w_q, scales, base) = oracle_fixture(rows);
    let mut history_rows = Vec::new();
    let mut history_indices = Vec::new();
    let mut history_counts = Vec::new();
    for (row, history) in histories[..rows].iter().enumerate() {
        for &(token, count) in history {
            history_rows.push(row as u32);
            history_indices.push(token);
            history_counts.push(count);
        }
    }
    let call = if quantized_activations {
        kiln_tensor::rocm_w8a8_gemv_sample_batch_bf16
    } else {
        kiln_tensor::rocm_w8a16_gemv_sample_batch_bf16
    };
    let got = call(
        &x,
        &w_q,
        &scales,
        &history_rows,
        &history_indices,
        &history_counts,
        &cases[..rows]
            .iter()
            .map(|case| case.repetition)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.presence)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.frequency)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.temperature)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.top_k)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.top_p)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.min_p)
            .collect::<Vec<_>>(),
        &cases[..rows]
            .iter()
            .map(|case| case.seed)
            .collect::<Vec<_>>(),
    )
    .expect("batched W8 oracle sample")
    .to_vec1::<i64>()
    .expect("read oracle tokens");
    let projected = projected_logits(&base, quantized_activations);
    let expected = cases[..rows]
        .iter()
        .zip(&histories[..rows])
        .map(|(&case, history)| cpu_sample(projected.clone(), history, case))
        .collect();
    (got, expected)
}

#[test]
fn mixed_rows_apply_penalties_and_filters_with_one_batch_result() {
    if no_rocm() {
        return;
    }
    // Row 0: greedy basis winner.
    // Row 1: token 1 starts highest, then repetition+presence penalties make
    // token 2 the overwhelming low-temperature winner.
    // Row 2: min-p retains only token 3.
    // Row 3: top-p retains only the positive-mean token 4.
    let expected = vec![0, 2, 3, 4];
    assert_eq!(run(false), expected, "W8A16 mixed-row result");
    assert_eq!(run(true), expected, "W8A8 mixed-row result");
}

#[test]
fn duplicate_history_and_unbounded_top_k_fail_closed() {
    if no_rocm() {
        return;
    }
    let (x, w_q, scales) = fixture();
    let common = (
        &[1.0; 4][..],
        &[0.0; 4][..],
        &[0.0; 4][..],
        &[1.0; 4][..],
        &[4u32; 4][..],
        &[1.0; 4][..],
        &[0.0; 4][..],
        &[1u64; 4][..],
    );
    let duplicate = kiln_tensor::rocm_w8a16_gemv_sample_batch_bf16(
        &x,
        &w_q,
        &scales,
        &[0, 0],
        &[1, 1],
        &[1, 2],
        common.0,
        common.1,
        common.2,
        common.3,
        common.4,
        common.5,
        common.6,
        common.7,
    )
    .expect_err("duplicate history must fail");
    assert!(duplicate.to_string().contains("duplicate history entry"));

    let unbounded = kiln_tensor::rocm_w8a16_gemv_sample_batch_bf16(
        &x,
        &w_q,
        &scales,
        &[],
        &[],
        &[],
        common.0,
        common.1,
        common.2,
        common.3,
        &[65, 4, 4, 4],
        common.5,
        common.6,
        common.7,
    )
    .expect_err("top_k beyond the kernel envelope must fail");
    assert!(unbounded.to_string().contains("outside 1..=64"));
}

#[test]
fn stochastic_filters_match_cpu_oracle_and_replay_exactly() {
    if no_rocm() {
        return;
    }
    for quantized_activations in [false, true] {
        for rows in [1, 2, 4, 6] {
            let (got, expected) = run_oracle_batch(rows, quantized_activations);
            assert_eq!(
                got,
                expected,
                "seeded ROCm sampler diverged from CPU oracle (W8A{}, rows={rows})",
                if quantized_activations { 8 } else { 16 }
            );
        }
        let (first, expected) = run_oracle_batch(6, quantized_activations);
        let (second, _) = run_oracle_batch(6, quantized_activations);
        assert_eq!(first, expected);
        assert_eq!(second, expected, "same seeded batch did not replay exactly");
        assert!(
            first.windows(2).any(|pair| pair[0] != pair[1]),
            "oracle fixture did not exercise distinct categorical outcomes"
        );
    }
}
