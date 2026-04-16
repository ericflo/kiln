//! Kiln benchmark suite — measures inference throughput, latency, VRAM, and training speed.
//!
//! Run with: `cargo run --release --features cuda --bin kiln-bench -- --model-path /path/to/weights`
//!
//! Requires a GPU with the Qwen3.5-4B model weights downloaded.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_core::vram::detect_vram;
use kiln_model::forward::{model_forward, GpuWeights, LinearAttentionState};
use kiln_model::kv_cache::KvCache;
use kiln_model::sampling::greedy_sample;
use kiln_model::ModelRunner;

/// Results from the full benchmark suite.
#[derive(Debug, Serialize)]
struct BenchmarkResults {
    gpu_info: GpuInfo,
    model_load: ModelLoadResult,
    inference: Vec<InferenceBenchResult>,
    latency: LatencyResult,
    training: Option<TrainingResult>,
}

#[derive(Debug, Serialize)]
struct GpuInfo {
    name: String,
    total_vram_mb: u64,
    vram_source: String,
}

#[derive(Debug, Serialize)]
struct ModelLoadResult {
    load_time_secs: f64,
    model_vram_mb: u64,
}

#[derive(Debug, Serialize)]
struct InferenceBenchResult {
    batch_size: usize,
    prompt_tokens: usize,
    output_tokens: usize,
    total_time_secs: f64,
    tokens_per_sec: f64,
    peak_vram_mb: u64,
}

#[derive(Debug, Serialize)]
struct LatencyResult {
    prompt_tokens: usize,
    prefill_time_ms: f64,
    prefill_tokens_per_sec: f64,
    time_to_first_token_ms: f64,
    mean_inter_token_ms: f64,
    p50_inter_token_ms: f64,
    p99_inter_token_ms: f64,
    num_tokens_generated: usize,
    decode_tokens_per_sec: f64,
}

#[derive(Debug, Serialize)]
struct TrainingResult {
    num_steps: usize,
    total_time_secs: f64,
    secs_per_step: f64,
    peak_vram_mb: u64,
}

/// Parse command-line arguments.
struct BenchArgs {
    model_path: String,
    max_output_tokens: usize,
    prompt_tokens: usize,
    training_steps: usize,
    skip_training: bool,
}

fn parse_args() -> Result<BenchArgs> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut max_output_tokens = 128;
    let mut prompt_tokens = 512;
    let mut training_steps = 10;
    let mut skip_training = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                i += 1;
                model_path = args.get(i).cloned().unwrap_or_default();
            }
            "--max-output-tokens" => {
                i += 1;
                max_output_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(128);
            }
            "--prompt-tokens" => {
                i += 1;
                prompt_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(512);
            }
            "--training-steps" => {
                i += 1;
                training_steps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--skip-training" => {
                skip_training = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: kiln-bench --model-path <path> [options]");
                eprintln!("  --model-path <path>       Path to Qwen3.5-4B weights directory");
                eprintln!("  --max-output-tokens <n>   Max tokens to generate per request (default: 128)");
                eprintln!("  --prompt-tokens <n>       Approximate prompt length in tokens (default: 512)");
                eprintln!("  --training-steps <n>      Number of SFT training steps (default: 10)");
                eprintln!("  --skip-training           Skip training benchmarks");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        anyhow::bail!("--model-path is required. Run with --help for usage.");
    }

    Ok(BenchArgs {
        model_path,
        max_output_tokens,
        prompt_tokens,
        training_steps,
        skip_training,
    })
}

/// Get GPU name from nvidia-smi.
fn gpu_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().lines().next().unwrap_or("unknown").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get current VRAM usage in bytes via nvidia-smi.
fn current_vram_used_bytes() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next()?.trim().parse::<u64>().ok())
        .map(|mib| mib * 1024 * 1024)
        .unwrap_or(0)
}

/// Build a prompt string of approximately `target_tokens` tokens by repeating sentences.
fn build_prompt(tokenizer: &KilnTokenizer, target_tokens: usize) -> String {
    let base = "The quick brown fox jumps over the lazy dog near the river bank. \
                Scientists discovered a new species of deep-sea fish in the Pacific Ocean. \
                The quantum computer solved the optimization problem in record time. \
                She wrote a comprehensive analysis of market trends for the quarterly report. ";

    let mut prompt = String::new();
    loop {
        prompt.push_str(base);
        let tokens = tokenizer.encode(&prompt).unwrap_or_default();
        if tokens.len() >= target_tokens {
            // Trim back to approximately target length
            while tokenizer.encode(&prompt).unwrap_or_default().len() > target_tokens {
                if let Some(pos) = prompt.rfind(". ") {
                    prompt.truncate(pos + 1);
                } else {
                    break;
                }
            }
            return prompt;
        }
    }
}

/// Benchmark inference throughput.
///
/// Runs `num_runs` sequential generations and reports aggregate throughput.
/// This measures single-request performance (not continuous batching).
fn bench_inference(
    runner: &ModelRunner,
    tokenizer: &KilnTokenizer,
    num_runs: usize,
    prompt_tokens: usize,
    max_output_tokens: usize,
) -> Result<InferenceBenchResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens);
    let actual_prompt_tokens = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .len();

    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        max_tokens: max_output_tokens,
        repetition_penalty: 1.0,
        stop: vec![],
        seed: Some(42),
    };

    // Warmup
    eprintln!("  Warmup...");
    let warmup_params = SamplingParams {
        max_tokens: 4,
        ..params.clone()
    };
    let _ = runner.generate(&prompt, &warmup_params);

    // Timed runs
    eprintln!("  Running {num_runs} sequential generations...");
    let mut total_output_tokens = 0usize;
    let overall_start = Instant::now();

    for i in 0..num_runs {
        let run_start = Instant::now();
        let output = runner
            .generate(&prompt, &params)
            .context("generation failed")?;
        let run_time = run_start.elapsed();
        let gen_tokens = output.token_ids.len();
        total_output_tokens += gen_tokens;

        eprintln!(
            "    Run {}/{}: {} tokens in {:.1}ms ({:.1} tok/s)",
            i + 1,
            num_runs,
            gen_tokens,
            run_time.as_secs_f64() * 1000.0,
            gen_tokens as f64 / run_time.as_secs_f64()
        );
    }

    let total_time = overall_start.elapsed();
    let peak_vram = current_vram_used_bytes() / (1024 * 1024);

    Ok(InferenceBenchResult {
        batch_size: num_runs,
        prompt_tokens: actual_prompt_tokens,
        output_tokens: total_output_tokens,
        total_time_secs: total_time.as_secs_f64(),
        tokens_per_sec: total_output_tokens as f64 / total_time.as_secs_f64(),
        peak_vram_mb: peak_vram,
    })
}

/// Benchmark latency by directly timing prefill and each decode step.
///
/// Uses `model_forward` directly for precise per-step timing.
fn bench_latency(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
) -> Result<LatencyResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device = weights.embed_tokens.device();
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };

    let max_total = actual_prompt_tokens + max_output_tokens;
    let mut kv_cache = KvCache::new(
        config.num_full_attention_layers,
        config.num_kv_heads,
        config.head_dim,
        max_total,
        dtype,
        device,
    )?;
    let mut linear_state = LinearAttentionState::new(config, device)?;

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!("  Measuring latency ({actual_prompt_tokens} prompt tokens)...");

    // Prefill: forward pass on all prompt tokens
    let prefill_start = Instant::now();
    let logits = model_forward(
        &prompt_token_ids,
        weights,
        config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
    )
    .context("prefill forward pass failed")?;
    kv_cache.advance(actual_prompt_tokens);

    // Sample first token
    let mut next_token = greedy_sample(&logits)?;
    let prefill_time = prefill_start.elapsed();

    eprintln!(
        "    Prefill: {:.1}ms ({:.0} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        actual_prompt_tokens as f64 / prefill_time.as_secs_f64()
    );

    // Decode: time each individual step
    let mut inter_token_ms: Vec<f64> = Vec::new();
    let mut num_tokens = 1usize; // counting the first token from prefill

    for _step in 0..max_output_tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }

        let step_start = Instant::now();
        let logits = model_forward(
            &[next_token],
            weights,
            config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
        )
        .context("decode forward pass failed")?;
        kv_cache.advance(1);
        next_token = greedy_sample(&logits)?;
        let step_time = step_start.elapsed();

        inter_token_ms.push(step_time.as_secs_f64() * 1000.0);
        num_tokens += 1;
    }

    let mean_itl = if inter_token_ms.is_empty() {
        0.0
    } else {
        inter_token_ms.iter().sum::<f64>() / inter_token_ms.len() as f64
    };

    let mut sorted = inter_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };

    let p99 = if sorted.is_empty() {
        0.0
    } else {
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    };

    let decode_tok_per_sec = if inter_token_ms.is_empty() {
        0.0
    } else {
        let total_decode_ms: f64 = inter_token_ms.iter().sum();
        inter_token_ms.len() as f64 / (total_decode_ms / 1000.0)
    };

    eprintln!(
        "    Decode: {num_tokens} tokens, mean ITL {:.1}ms ({:.1} tok/s)",
        mean_itl, decode_tok_per_sec
    );

    Ok(LatencyResult {
        prompt_tokens: actual_prompt_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: actual_prompt_tokens as f64 / prefill_time.as_secs_f64(),
        time_to_first_token_ms: prefill_time.as_secs_f64() * 1000.0,
        mean_inter_token_ms: mean_itl,
        p50_inter_token_ms: p50,
        p99_inter_token_ms: p99,
        num_tokens_generated: num_tokens,
        decode_tokens_per_sec: decode_tok_per_sec,
    })
}

/// Benchmark SFT training speed.
fn bench_training(
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    num_steps: usize,
) -> Result<TrainingResult> {
    use kiln_train::{ChatMessage, SftConfig, SftExample};

    eprintln!("  Running {num_steps} SFT training steps...");

    // Create synthetic training examples
    let examples: Vec<SftExample> = (0..num_steps)
        .map(|i| SftExample {
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "What is the capital of country number {i}? Please explain in detail."
                    ),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: format!(
                        "The capital of country number {i} is City{i}. It is located in the \
                         central region and has a population of approximately {} million people. \
                         The city was founded in the {}th century and is known for its historic \
                         architecture and vibrant cultural scene.",
                        i * 3 + 1,
                        (i % 15) + 5
                    ),
                },
            ],
        })
        .collect();

    let config = SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some("bench-adapter".to_string()),
        auto_load: false,
        checkpoint_interval: None,
    };

    let adapter_dir = std::env::temp_dir().join("kiln-bench-adapters");
    std::fs::create_dir_all(&adapter_dir)?;

    let progress_cb =
        Some(
            Box::new(|progress: kiln_train::trainer::TrainingProgress| {
                eprintln!(
                    "    Step {}/{}: loss={:.6}",
                    progress.step, progress.total_steps, progress.loss
                );
            }) as kiln_train::trainer::ProgressCallback,
        );

    let start = Instant::now();
    let result = kiln_train::trainer::sft_train(
        &examples,
        &config,
        model_config,
        weights,
        tokenizer,
        &adapter_dir,
        "bench-adapter",
        progress_cb,
    );
    let elapsed = start.elapsed();

    let peak_vram = current_vram_used_bytes() / (1024 * 1024);

    // Clean up temp adapter
    let _ = std::fs::remove_dir_all(adapter_dir.join("bench-adapter"));

    match result {
        Ok(_path) => Ok(TrainingResult {
            num_steps,
            total_time_secs: elapsed.as_secs_f64(),
            secs_per_step: elapsed.as_secs_f64() / num_steps as f64,
            peak_vram_mb: peak_vram,
        }),
        Err(e) => {
            eprintln!("  Training failed: {e}");
            Err(e)
        }
    }
}

fn print_summary(results: &BenchmarkResults) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  KILN BENCHMARK RESULTS");
    eprintln!("{}\n", "=".repeat(60));

    eprintln!(
        "GPU: {} ({} MB VRAM, source: {})",
        results.gpu_info.name, results.gpu_info.total_vram_mb, results.gpu_info.vram_source
    );
    eprintln!(
        "Model load time: {:.2}s (VRAM: {} MB)\n",
        results.model_load.load_time_secs, results.model_load.model_vram_mb
    );

    eprintln!("--- Inference Throughput ---");
    eprintln!(
        "{:<8} {:>10} {:>10} {:>12} {:>10}",
        "Runs", "Prompt", "Output", "tok/s", "VRAM MB"
    );
    for r in &results.inference {
        eprintln!(
            "{:<8} {:>10} {:>10} {:>12.1} {:>10}",
            r.batch_size, r.prompt_tokens, r.output_tokens, r.tokens_per_sec, r.peak_vram_mb
        );
    }

    eprintln!("\n--- Latency (single request) ---");
    eprintln!(
        "Prompt tokens:         {}",
        results.latency.prompt_tokens
    );
    eprintln!(
        "Prefill time:          {:.1} ms ({:.0} tok/s)",
        results.latency.prefill_time_ms, results.latency.prefill_tokens_per_sec
    );
    eprintln!(
        "Time to first token:   {:.1} ms",
        results.latency.time_to_first_token_ms
    );
    eprintln!(
        "Mean inter-token:      {:.1} ms ({:.1} tok/s)",
        results.latency.mean_inter_token_ms, results.latency.decode_tokens_per_sec
    );
    eprintln!(
        "P50 inter-token:       {:.1} ms",
        results.latency.p50_inter_token_ms
    );
    eprintln!(
        "P99 inter-token:       {:.1} ms",
        results.latency.p99_inter_token_ms
    );
    eprintln!(
        "Tokens generated:      {}",
        results.latency.num_tokens_generated
    );

    if let Some(t) = &results.training {
        eprintln!("\n--- SFT Training ---");
        eprintln!("Steps:          {}", t.num_steps);
        eprintln!("Total time:     {:.2} s", t.total_time_secs);
        eprintln!("Time per step:  {:.2} s", t.secs_per_step);
        eprintln!("Peak VRAM:      {} MB", t.peak_vram_mb);
    }

    eprintln!();
}

fn main() -> Result<()> {
    // Initialize logging for training progress
    tracing_subscriber::fmt()
        .with_env_filter("kiln=info")
        .with_writer(std::io::stderr)
        .init();

    let args = parse_args()?;
    let model_path = Path::new(&args.model_path);

    eprintln!("=== Kiln Benchmark Suite ===\n");

    // GPU info
    let vram = detect_vram();
    let gpu_info = GpuInfo {
        name: gpu_name(),
        total_vram_mb: vram.total_bytes / (1024 * 1024),
        vram_source: vram.source.to_string(),
    };
    eprintln!("GPU: {} ({} MB)", gpu_info.name, gpu_info.total_vram_mb);

    // Load model
    let model_config = ModelConfig::qwen3_5_4b();
    eprintln!("Loading model from {}...", model_path.display());

    let vram_before = current_vram_used_bytes();
    let load_start = Instant::now();

    let model_weights = kiln_model::load_model(model_path, &model_config)
        .context("failed to load model weights")?;

    let device = if candle_core::utils::cuda_is_available() {
        candle_core::Device::new_cuda(0)?
    } else {
        anyhow::bail!("CUDA not available — benchmarks require a GPU");
    };

    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("failed to transfer weights to GPU")?;
    drop(model_weights); // Free CPU memory

    let load_time = load_start.elapsed();
    let vram_after = current_vram_used_bytes();
    let model_vram = (vram_after.saturating_sub(vram_before)) / (1024 * 1024);

    eprintln!(
        "Model loaded in {:.2}s (VRAM: {} MB)\n",
        load_time.as_secs_f64(),
        model_vram
    );

    let model_load = ModelLoadResult {
        load_time_secs: load_time.as_secs_f64(),
        model_vram_mb: model_vram,
    };

    // Load tokenizer from model directory
    let tokenizer = {
        let tok_file = model_path.join("tokenizer.json");
        if tok_file.exists() {
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            eprintln!("Loading tokenizer from HuggingFace Hub...");
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    // Latency benchmark (uses model_forward directly — must run before runner takes ownership)
    eprintln!("--- Latency Benchmark ---");
    let latency = bench_latency(
        &gpu_weights,
        &model_config,
        &tokenizer,
        args.prompt_tokens,
        args.max_output_tokens,
    )
    .context("latency benchmark failed")?;

    // Training benchmark (borrows gpu_weights — must run before runner takes ownership)
    let training = if args.skip_training {
        eprintln!("\n--- Training Benchmark (skipped) ---");
        None
    } else {
        eprintln!("\n--- Training Benchmark ---");
        match bench_training(&model_config, &gpu_weights, &tokenizer, args.training_steps) {
            Ok(result) => {
                eprintln!(
                    "  => {:.2}s/step, peak VRAM {} MB",
                    result.secs_per_step, result.peak_vram_mb
                );
                Some(result)
            }
            Err(e) => {
                eprintln!("  Training benchmark failed: {e}");
                None
            }
        }
    };

    // Load a second tokenizer for the runner (ModelRunner takes ownership)
    let runner_tokenizer = {
        let tok_file = model_path.join("tokenizer.json");
        if tok_file.exists() {
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    // Create runner for throughput benchmarks (takes ownership of weights)
    let runner = ModelRunner::new(gpu_weights, runner_tokenizer, model_config.clone());

    // Inference throughput at different run counts
    eprintln!("\n--- Inference Throughput Benchmarks ---");
    let run_counts = [1, 4, 8, 16];
    let mut inference_results = Vec::new();

    for &n in &run_counts {
        eprintln!("\n{n} sequential runs:");
        match bench_inference(&runner, &tokenizer, n, args.prompt_tokens, args.max_output_tokens) {
            Ok(result) => {
                eprintln!("  => {:.1} tok/s aggregate", result.tokens_per_sec);
                inference_results.push(result);
            }
            Err(e) => {
                eprintln!("  => FAILED: {e}");
            }
        }
    }

    let results = BenchmarkResults {
        gpu_info,
        model_load,
        inference: inference_results,
        latency,
        training,
    };

    // Print human-readable summary to stderr
    print_summary(&results);

    // Print JSON to stdout for machine parsing
    let json = serde_json::to_string_pretty(&results)?;
    println!("{json}");

    Ok(())
}
