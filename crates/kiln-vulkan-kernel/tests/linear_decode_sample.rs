use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::kernels;

mod support;

const SUITE: &str = "linear_decode_sample";

fn f32_bytes(values: &[f32]) -> &[u8] {
    bytemuck::cast_slice(values)
}

#[test]
fn linear_decode_sample_command_batch_returns_argmax_top1() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };

    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [1.0f32, -2.0, 0.5];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 2.0, 0.0, //
        0.0, 0.0, 3.0, 0.0, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let token = kernels::dispatch_linear_decode_sample_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        hidden,
        out_dim,
        &[],
        &[],
        1.0,
        0.0,
        0.0,
        1.0,
        1,
        1.0,
        0.0,
        1234,
    )?;
    assert_eq!(token, 1);
    Ok(())
}

#[test]
fn linear_decode_sample_command_batch_applies_penalties() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };

    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [1.0f32, -2.0, 0.5];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 3.0, 3.6, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let token = kernels::dispatch_linear_decode_sample_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        hidden,
        out_dim,
        &[1],
        &[1],
        1.0,
        10.0,
        0.0,
        1.0,
        1,
        1.0,
        0.0,
        1234,
    )?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn linear_decode_sample_batch_returns_argmax_top1() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };

    let batch = 2usize;
    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [
        1.0f32, -2.0, 0.5, //
        0.0, 1.0, 2.0,
    ];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 2.0, 0.0, //
        0.0, 0.0, 3.0, 0.0, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let tokens = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        batch,
        hidden,
        out_dim,
        &[],
        &[],
        &[],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[1, 1],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[1234, 5678],
    )?;
    assert_eq!(tokens, vec![1, 2]);
    Ok(())
}

#[test]
fn linear_decode_sample_batch_applies_row_penalties() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };

    let batch = 2usize;
    let hidden = 3usize;
    let out_dim = 5usize;
    let x = [
        1.0f32, -2.0, 0.5, //
        0.0, 1.0, 2.0,
    ];
    let weight_t = [
        1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 3.0, 0.0, 0.0, //
        0.0, 0.0, 3.0, 3.6, -1.0,
    ];
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let tokens = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        batch,
        hidden,
        out_dim,
        &[0],
        &[1],
        &[1],
        &[1.0, 1.0],
        &[10.0, 0.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[1, 1],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[1234, 5678],
    )?;
    assert_eq!(tokens, vec![3, 2]);
    Ok(())
}

#[test]
fn linear_decode_sample_batch_rows8_bf16_top1_matches_cpu() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };

    let batch = 65usize;
    let hidden = 11usize;
    let out_dim = 67usize;
    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| (((i * 17 + 13) % 101) as f32 - 50.0) * 0.013671875)
        .collect();
    let weight_t: Vec<bf16> = (0..hidden * out_dim)
        .map(|i| bf16::from_f32((((i * 31 + 7) % 127) as f32 - 63.0) * 0.0048828125))
        .collect();
    let weight_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &weight_t)?;

    let mut expected = vec![0u32; batch];
    for row in 0..batch {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0u32;
        for col in 0..out_dim {
            let mut score = 0.0f32;
            for h in 0..hidden {
                score += x[row * hidden + h] * weight_t[h * out_dim + col].to_f32();
            }
            if score > best_score || (score == best_score && (col as u32) < best_idx) {
                best_score = score;
                best_idx = col as u32;
            }
        }
        expected[row] = best_idx;
    }

    let tokens = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        true,
        batch,
        hidden,
        out_dim,
        &[],
        &[],
        &[],
        &vec![1.0; batch],
        &vec![0.0; batch],
        &vec![0.0; batch],
        &vec![1.0; batch],
        &vec![1; batch],
        &vec![1.0; batch],
        &vec![0.0; batch],
        &vec![1234; batch],
    )?;
    assert_eq!(tokens, expected);
    Ok(())
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

fn xorshift32(mut state: u32) -> u32 {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    state
}

fn cpu_sample(mut logits: Vec<f32>, history: &[(u32, u32)], case: SamplingCase) -> u32 {
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

    let low = case.seed as u32;
    let high = (case.seed >> 32) as u32;
    let mut state = low ^ high.wrapping_mul(2_654_435_761);
    if state == 0 {
        state = 0x9E37_79B9;
    }
    let random = (xorshift32(state) as f32) * (1.0f32 / 4_294_967_295.0f32);
    let mut cumulative = 0.0f32;
    let mut sampled = candidates[0];
    for (&candidate, &probability) in candidates.iter().zip(&probabilities) {
        cumulative += probability;
        sampled = candidate;
        if random < cumulative {
            break;
        }
    }
    sampled as u32
}

#[test]
fn linear_decode_sample_batch_stochastic_filters_match_cpu_oracle() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };

    const VOCAB: usize = 12;
    let base_logits = [
        3.25f32, 2.75, 2.10, 1.60, 1.15, 0.70, 0.20, -0.25, -0.80, -1.35, -2.0, -2.75,
    ];
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
            top_k: 9,
            top_p: 0.83,
            min_p: 0.08,
            seed: 0xfedc_ba98_7654_3210,
            repetition: 1.35,
            presence: 0.30,
            frequency: 0.17,
        },
        SamplingCase {
            temperature: 1.65,
            top_k: 12,
            top_p: 0.91,
            min_p: 0.03,
            seed: 0x0000_0000_0000_0000,
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
        vec![(0u32, 3u32), (2, 1), (7, 4)],
        vec![(1u32, 2u32), (8, 5)],
        vec![(0u32, 1u32), (3, 6), (11, 2)],
    ];
    let batch = cases.len();

    // Identity projection makes the desired logits exact inputs to the sampler,
    // while still exercising the production fused linear + penalty + sample path.
    let x = cases.iter().flat_map(|_| base_logits).collect::<Vec<_>>();
    let weight_t = (0..VOCAB)
        .flat_map(|row| (0..VOCAB).map(move |col| if row == col { 1.0f32 } else { 0.0 }))
        .collect::<Vec<_>>();
    let weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &weight_t)?;

    let mut history_rows = Vec::new();
    let mut history_indices = Vec::new();
    let mut history_counts = Vec::new();
    for (row, history) in histories.iter().enumerate() {
        for &(token, count) in history {
            history_rows.push(row as u32);
            history_indices.push(token);
            history_counts.push(count);
        }
    }

    let expected = cases
        .iter()
        .zip(&histories)
        .map(|(&case, history)| cpu_sample(base_logits.to_vec(), history, case))
        .collect::<Vec<_>>();
    let got = kernels::dispatch_linear_decode_sample_batch_bytes(
        &dev,
        f32_bytes(&x),
        &weight_buf,
        false,
        batch,
        VOCAB,
        VOCAB,
        &history_rows,
        &history_indices,
        &history_counts,
        &cases.iter().map(|case| case.repetition).collect::<Vec<_>>(),
        &cases.iter().map(|case| case.presence).collect::<Vec<_>>(),
        &cases.iter().map(|case| case.frequency).collect::<Vec<_>>(),
        &cases
            .iter()
            .map(|case| case.temperature)
            .collect::<Vec<_>>(),
        &cases.iter().map(|case| case.top_k).collect::<Vec<_>>(),
        &cases.iter().map(|case| case.top_p).collect::<Vec<_>>(),
        &cases.iter().map(|case| case.min_p).collect::<Vec<_>>(),
        &cases.iter().map(|case| case.seed).collect::<Vec<_>>(),
    )?;

    assert_eq!(
        got, expected,
        "seeded Vulkan sampler diverged from CPU oracle"
    );
    assert!(
        got.windows(2).any(|pair| pair[0] != pair[1]),
        "oracle fixture did not exercise distinct categorical outcomes"
    );
    eprintln!(
        "KILN_VULKAN_SAMPLING_CPU_ORACLE_PASS rows={} tokens={got:?}",
        cases.len()
    );
    Ok(())
}
