//! Real-device parity and validation coverage for compact prompt-logprobs.
#![cfg(feature = "rocm")]

use kiln_tensor::{DType, Device, DevicePromptLogprobCandidate, Tensor};

fn rocm_available_or_skip() -> bool {
    if kiln_tensor::rocm_is_available() {
        return true;
    }
    if std::env::var("KILN_QUALIFICATION").ok().as_deref() == Some("1") {
        panic!("ROCm device unavailable while KILN_QUALIFICATION=1");
    }
    eprintln!("no ROCm device available; skipping prompt-logprob qualification");
    false
}

fn values_for(rows: usize, cols: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let hash = (row as u64)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add((col as u64).wrapping_mul(0x85eb_ca6b))
                .wrapping_add(17);
            let value = ((hash % 20_003) as f32 - 10_001.0) / 317.0;
            values.push(value);
        }
    }
    // Deliberate ties exercise ascending-token-ID tie ordering.
    if cols >= 7 {
        for row in 0..rows {
            values[row * cols + 1] = 42.0;
            values[row * cols + 3] = 42.0;
            values[row * cols + 6] = 42.0;
        }
    }
    values
}

fn cpu_row(
    values: &[f32],
    observed: u32,
    top_k: usize,
) -> (f32, f32, f32, usize, Vec<DevicePromptLogprobCandidate>) {
    assert!(values.iter().all(|value| value.is_finite()));
    let row_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted_sum = values
        .iter()
        .copied()
        .map(|value| (value - row_max).exp())
        .sum::<f32>();
    let log_sum = shifted_sum.ln();
    let observed_logit = values[observed as usize];
    let observed_rank = values
        .iter()
        .filter(|&&value| value >= observed_logit)
        .count();
    let mut pairs = values.iter().copied().enumerate().collect::<Vec<_>>();
    pairs.sort_unstable_by(|(left_id, left), (right_id, right)| {
        right
            .partial_cmp(left)
            .unwrap()
            .then_with(|| left_id.cmp(right_id))
    });
    let candidates = pairs
        .into_iter()
        .take(top_k)
        .map(|(token_id, logit)| DevicePromptLogprobCandidate {
            token_id: token_id as u32,
            logit,
            logprob: (logit - row_max) - log_sum,
        })
        .collect();
    (row_max, log_sum, observed_logit, observed_rank, candidates)
}

fn assert_close(got: f32, expected: f32, context: &str) {
    let tolerance = 2e-4 + 2e-5 * expected.abs();
    assert!(
        (got - expected).abs() <= tolerance,
        "{context}: got {got}, expected {expected}, tolerance {tolerance}"
    );
}

fn run_case(dtype: DType, rows: usize, cols: usize, top_k: usize) {
    let source = values_for(rows, cols);
    let logits = Tensor::from_vec_on(Device::Rocm(0), source, vec![rows, cols])
        .expect("upload prompt logits")
        .to_dtype(dtype)
        .expect("cast prompt logits");
    // The independent oracle consumes the dtype-rounded values actually seen
    // by the kernel rather than assuming host and device casts are identical.
    let rounded = logits
        .to_dtype(DType::F32)
        .expect("promote rounded logits")
        .to_device(Device::Cpu)
        .expect("read rounded logits")
        .to_vec2::<f32>()
        .expect("materialize rounded logits");
    let observed = (0..rows)
        .map(|row| ((row * 97 + cols / 2) % cols) as u32)
        .collect::<Vec<_>>();

    let got = kiln_tensor::rocm_prompt_logprobs(&logits, &observed, top_k)
        .expect("compact ROCm prompt-logprobs");
    assert_eq!(got.len(), rows);
    for row in 0..rows {
        let (row_max, log_sum, observed_logit, observed_rank, candidates) =
            cpu_row(&rounded[row], observed[row], top_k);
        assert_eq!(got[row].row_max, row_max, "row {row} maximum");
        assert_close(
            got[row].log_sum_exp_shifted,
            log_sum,
            &format!("row {row} log-sum"),
        );
        assert_eq!(got[row].observed_logit, observed_logit, "row {row}");
        assert_eq!(got[row].observed_full_rank, observed_rank, "row {row}");
        assert_close(
            got[row].observed_logprob,
            (observed_logit - row_max) - log_sum,
            &format!("row {row} observed logprob"),
        );
        assert_eq!(got[row].candidates.len(), top_k);
        for (rank, (actual, expected)) in got[row].candidates.iter().zip(&candidates).enumerate() {
            assert_eq!(actual.token_id, expected.token_id, "row {row} rank {rank}");
            assert_eq!(actual.logit, expected.logit, "row {row} rank {rank}");
            assert_close(
                actual.logprob,
                expected.logprob,
                &format!("row {row} rank {rank} logprob"),
            );
        }
    }
}

#[test]
fn prompt_logprobs_match_cpu_across_dtypes_waves_and_production_width() {
    if !rocm_available_or_skip() {
        return;
    }
    for dtype in [DType::F32, DType::BF16, DType::F16] {
        for &(rows, cols, top_k) in &[
            (3, 31, 0),
            (3, 33, 5),
            (3, 63, 5),
            (3, 65, 7),
            (2, 1_025, 16),
            (1, 513, 256),
            (2, 151_936, 5),
        ] {
            run_case(dtype, rows, cols, top_k);
        }
    }
}

#[test]
fn prompt_logprobs_report_first_non_finite_logit() {
    if !rocm_available_or_skip() {
        return;
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = values_for(2, 129);
        values[129 + 64] = value;
        values[129 + 96] = f32::NAN;
        let logits = Tensor::from_vec_on(Device::Rocm(0), values, vec![2, 129])
            .expect("upload non-finite logits");
        let error = kiln_tensor::rocm_prompt_logprobs(&logits, &[0, 1], 5).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("row 1"), "{message}");
        assert!(message.contains("non-finite logit"), "{message}");
        assert!(message.contains("token id 64"), "{message}");
    }
}

#[test]
fn prompt_logprobs_report_finite_logit_subtraction_overflow() {
    if !rocm_available_or_skip() {
        return;
    }
    let logits = Tensor::from_vec_on(Device::Rocm(0), vec![f32::MAX, -f32::MAX, 0.0], vec![1, 3])
        .expect("upload extreme finite logits");
    let error = kiln_tensor::rocm_prompt_logprobs(&logits, &[0], 1).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("non-finite log-probability"), "{message}");
    assert!(message.contains("token id 1"), "{message}");
}

#[test]
fn prompt_logprobs_reject_invalid_host_contracts_before_launch() {
    if !rocm_available_or_skip() {
        return;
    }
    let logits =
        Tensor::from_vec_on(Device::Rocm(0), vec![0.0f32; 8], vec![2, 4]).expect("upload logits");
    assert!(
        kiln_tensor::rocm_prompt_logprobs(&logits, &[0], 1)
            .unwrap_err()
            .to_string()
            .contains("observed token count")
    );
    assert!(
        kiln_tensor::rocm_prompt_logprobs(&logits, &[0, 4], 1)
            .unwrap_err()
            .to_string()
            .contains("outside vocabulary")
    );
    assert!(
        kiln_tensor::rocm_prompt_logprobs(&logits, &[0, 1], 5)
            .unwrap_err()
            .to_string()
            .contains("top_k")
    );
}
