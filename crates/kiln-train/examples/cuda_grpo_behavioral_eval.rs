//! CUDA GRPO behavioral evaluation harness.
//!
//! Phase 3a: a downstream signal that is not "training-time loss". Loads the
//! base model + an optional LoRA adapter, generates N completions per held-out
//! prompt, and reports the diagnostics the entropy-collapse / length-drift /
//! mode-collapse line of work in DAPO, Magistral, Open-Reasoner-Zero, and the
//! Cui et al. KL-Cov paper all cite as the load-bearing signals when no
//! domain-specific verifier is available:
//!
//! - mean completion length in tokens
//! - p50 / p95 / max completion length
//! - truncation rate (fraction hitting `--max-tokens`)
//! - repetition rate (fraction of tokens that are an exact match to one of
//!   the previous 32 tokens — a cheap proxy for the "loop" failure mode)
//! - self-similarity (mean Jaccard overlap on bigrams across the N
//!   completions for the same prompt — drops when entropy collapses)
//!
//! Output is one JSON line per prompt, plus a final aggregate JSON summary,
//! so a downstream analyzer can diff adapters along these axes.

use anyhow::Result;

#[cfg(feature = "cuda")]
use std::collections::HashSet;
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use kiln_core::config::ModelConfig;
#[cfg(feature = "cuda")]
use kiln_core::sampling::SamplingParams;
#[cfg(feature = "cuda")]
use kiln_core::tokenizer::KilnTokenizer;
#[cfg(feature = "cuda")]
use kiln_model::forward::GpuWeights;
#[cfg(feature = "cuda")]
use kiln_model::generate::{FinishReason, ModelRunner};
#[cfg(feature = "cuda")]
use kiln_train::{ChatMessage, GrpoGroup};

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    data: PathBuf,
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    max_prompts: Option<usize>,
    num_samples: usize,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed_base: u64,
    label: String,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut data = None;
        let mut model_path = None;
        let mut adapter_path: Option<PathBuf> = None;
        let mut max_prompts: Option<usize> = None;
        let mut num_samples: usize = 4;
        let mut max_tokens: usize = 256;
        let mut temperature: f32 = 1.0;
        let mut top_p: f32 = 0.95;
        let mut seed_base: u64 = 0xC0DE_BEEF;
        let mut label = "eval".to_string();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--data" => data = Some(PathBuf::from(args.next().context("--data needs value")?)),
                "--model" => {
                    model_path = Some(PathBuf::from(args.next().context("--model needs value")?))
                }
                "--adapter" => {
                    adapter_path =
                        Some(PathBuf::from(args.next().context("--adapter needs value")?))
                }
                "--max-prompts" => {
                    max_prompts = Some(
                        args.next()
                            .context("--max-prompts needs value")?
                            .parse()
                            .context("--max-prompts must be a positive integer")?,
                    )
                }
                "--samples" => {
                    num_samples = args
                        .next()
                        .context("--samples needs value")?
                        .parse()
                        .context("--samples must be a positive integer")?
                }
                "--max-tokens" => {
                    max_tokens = args
                        .next()
                        .context("--max-tokens needs value")?
                        .parse()
                        .context("--max-tokens must be a positive integer")?
                }
                "--temperature" => {
                    temperature = args
                        .next()
                        .context("--temperature needs value")?
                        .parse()
                        .context("--temperature must be a float")?
                }
                "--top-p" => {
                    top_p = args
                        .next()
                        .context("--top-p needs value")?
                        .parse()
                        .context("--top-p must be a float")?
                }
                "--seed" => {
                    seed_base = args
                        .next()
                        .context("--seed needs value")?
                        .parse()
                        .context("--seed must be u64")?
                }
                "--label" => label = args.next().context("--label needs value")?,
                "--help" | "-h" => {
                    println!(
                        "cuda_grpo_behavioral_eval --data <jsonl> --model <dir> \
                         [--adapter <dir>] [--label NAME] [--max-prompts N] \
                         [--samples N] [--max-tokens N] [--temperature F] \
                         [--top-p F] [--seed N]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unexpected argument: {other}"),
            }
        }
        Ok(Args {
            data: data.context("--data is required")?,
            model_path: model_path.context("--model is required")?,
            adapter_path,
            max_prompts,
            num_samples,
            max_tokens,
            temperature,
            top_p,
            seed_base,
            label,
        })
    }
}

#[cfg(feature = "cuda")]
fn load_tokenizer(model_path: &PathBuf) -> Result<KilnTokenizer> {
    let tokenizer_path = model_path.join("tokenizer.json");
    let mut tokenizer = KilnTokenizer::from_file(
        tokenizer_path
            .to_str()
            .context("tokenizer path is not valid UTF-8")?,
    )
    .map_err(|err| anyhow::anyhow!("{err}"))?;
    let template_path = model_path.join("chat_template.jinja");
    if template_path.exists() {
        let template = std::fs::read_to_string(&template_path)
            .with_context(|| format!("reading {}", template_path.display()))?;
        tokenizer = tokenizer.with_chat_template(template);
    }
    Ok(tokenizer)
}

#[cfg(feature = "cuda")]
fn load_groups(path: &PathBuf, limit: Option<usize>) -> Result<Vec<GrpoGroup>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
        let g: GrpoGroup = serde_json::from_str(line).with_context(|| "parsing JSONL group")?;
        out.push(g);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn prompt_text(group: &GrpoGroup, tokenizer: &KilnTokenizer) -> Result<String> {
    let messages: Vec<kiln_core::tokenizer::ChatMessage> = group
        .messages
        .iter()
        .map(|m: &ChatMessage| kiln_core::tokenizer::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
            ..Default::default()
        })
        .collect();
    tokenizer
        .apply_chat_template(&messages)
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(feature = "cuda")]
fn repetition_rate(token_ids: &[u32], window: usize) -> f32 {
    if token_ids.len() < 2 {
        return 0.0;
    }
    let mut hits = 0usize;
    for i in 1..token_ids.len() {
        let start = i.saturating_sub(window);
        if token_ids[start..i].contains(&token_ids[i]) {
            hits += 1;
        }
    }
    hits as f32 / (token_ids.len() - 1) as f32
}

#[cfg(feature = "cuda")]
fn bigrams(token_ids: &[u32]) -> HashSet<(u32, u32)> {
    let mut s = HashSet::new();
    for w in token_ids.windows(2) {
        s.insert((w[0], w[1]));
    }
    s
}

#[cfg(feature = "cuda")]
fn jaccard(a: &HashSet<(u32, u32)>, b: &HashSet<(u32, u32)>) -> f32 {
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

#[cfg(feature = "cuda")]
fn mean_pairwise_jaccard(samples: &[Vec<u32>]) -> f32 {
    if samples.len() < 2 {
        return 1.0;
    }
    let grams: Vec<_> = samples.iter().map(|s| bigrams(s)).collect();
    let mut sum = 0.0_f32;
    let mut cnt = 0usize;
    for i in 0..grams.len() {
        for j in (i + 1)..grams.len() {
            sum += jaccard(&grams[i], &grams[j]);
            cnt += 1;
        }
    }
    if cnt == 0 { 0.0 } else { sum / cnt as f32 }
}

#[cfg(feature = "cuda")]
fn percentile(sorted: &[usize], q: f32) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (q * (sorted.len().saturating_sub(1)) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse()?;
    let started = Instant::now();
    eprintln!("label={} adapter={:?}", args.label, args.adapter_path);

    let tokenizer = load_tokenizer(&args.model_path)?;
    let groups = load_groups(&args.data, args.max_prompts)?;
    anyhow::ensure!(!groups.is_empty(), "no prompts loaded");

    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA not available"
    );
    let device = candle_core::Device::new_cuda(0).context("create CUDA device 0")?;
    let model_config = ModelConfig::qwen3_5_4b();
    let model_weights = kiln_model::load_model_with_options(
        &args.model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load model weights")?;
    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("transfer weights to CUDA")?;
    drop(model_weights);

    // ModelRunner takes ownership of weights + tokenizer + config.
    let mut runner = ModelRunner::new(gpu_weights, tokenizer.clone(), model_config.clone());
    if let Some(adapter) = &args.adapter_path {
        runner
            .load_adapter(adapter)
            .with_context(|| format!("load adapter {}", adapter.display()))?;
        eprintln!("adapter_loaded={}", adapter.display());
    } else {
        eprintln!("adapter_loaded=base");
    }

    let mut all_lengths: Vec<usize> = Vec::new();
    let mut all_rep: Vec<f32> = Vec::new();
    let mut truncated = 0usize;
    let mut per_prompt_diversity: Vec<f32> = Vec::new();

    for (prompt_idx, group) in groups.iter().enumerate() {
        let pt = prompt_text(group, &tokenizer)?;
        let mut samples_tokens: Vec<Vec<u32>> = Vec::with_capacity(args.num_samples);
        for s in 0..args.num_samples {
            let mut sp = SamplingParams::default();
            sp.temperature = args.temperature;
            sp.top_p = args.top_p;
            sp.max_tokens = args.max_tokens;
            sp.seed = Some(args.seed_base.wrapping_add(
                (prompt_idx as u64).wrapping_mul(1_000_003).wrapping_add(s as u64),
            ));
            let out = runner
                .generate(&pt, &sp)
                .with_context(|| format!("generate prompt={prompt_idx} sample={s}"))?;
            let len = out.token_ids.len();
            let rep = repetition_rate(&out.token_ids, 32);
            let truncated_here = matches!(out.finish_reason, FinishReason::MaxTokens);
            if truncated_here {
                truncated += 1;
            }
            all_lengths.push(len);
            all_rep.push(rep);
            samples_tokens.push(out.token_ids);
            println!(
                "{{\"label\":\"{}\",\"prompt\":{},\"sample\":{},\"len\":{},\"rep\":{:.6},\"truncated\":{}}}",
                args.label, prompt_idx, s, len, rep, truncated_here
            );
        }
        let div = mean_pairwise_jaccard(&samples_tokens);
        per_prompt_diversity.push(div);
        println!(
            "{{\"label\":\"{}\",\"prompt\":{},\"mean_bigram_jaccard\":{:.6}}}",
            args.label, prompt_idx, div
        );
    }

    // Aggregate.
    all_lengths.sort_unstable();
    let total_samples = all_lengths.len();
    let mean_len = if total_samples == 0 {
        0.0
    } else {
        all_lengths.iter().copied().sum::<usize>() as f32 / total_samples as f32
    };
    let p50 = percentile(&all_lengths, 0.5);
    let p95 = percentile(&all_lengths, 0.95);
    let p99 = percentile(&all_lengths, 0.99);
    let max_len = all_lengths.last().copied().unwrap_or(0);
    let mean_rep = if all_rep.is_empty() {
        0.0
    } else {
        all_rep.iter().copied().sum::<f32>() / all_rep.len() as f32
    };
    let mean_div = if per_prompt_diversity.is_empty() {
        0.0
    } else {
        per_prompt_diversity.iter().copied().sum::<f32>() / per_prompt_diversity.len() as f32
    };
    let trunc_rate = if total_samples == 0 {
        0.0
    } else {
        truncated as f32 / total_samples as f32
    };
    println!(
        "{{\"label\":\"{}\",\"summary\":true,\"prompts\":{},\"samples\":{},\"mean_len\":{:.2},\"p50_len\":{},\"p95_len\":{},\"p99_len\":{},\"max_len\":{},\"mean_rep_rate\":{:.6},\"mean_bigram_jaccard\":{:.6},\"truncation_rate\":{:.6},\"elapsed_secs\":{:.2}}}",
        args.label,
        groups.len(),
        total_samples,
        mean_len,
        p50,
        p95,
        p99,
        max_len,
        mean_rep,
        mean_div,
        trunc_rate,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("cuda_grpo_behavioral_eval requires --features cuda");
}
