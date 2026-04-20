//! Phase 10 — Fused Linear Cross-Entropy (FLCE) preflight bench.
//!
//! One-shot measurement of peak VRAM during a single SFT training step at
//! T ∈ {2048, 8192, 16384} on Qwen3.5-4B (V=248320), with gradient
//! checkpointing ON (default). Paired with the static Liger-audit math for
//! the three retained output-layer tensors:
//!
//!   (a) bf16 logits       [T, V]       T × 248320 × 2 bytes
//!   (b) F32 cast of logits [T_active, V] T × 248320 × 4 bytes
//!   (c) Retained grad-of-logits for backward [T, V] in bf16 ~ T × 248320 × 2 bytes
//!
//! The preflight gating signal is: at T=16384, is (a+b+c) ≥ 30% of peak VRAM?
//!
//! Run:
//!   cargo run --release --features cuda -p kiln-server \
//!     --example flce_preflight_bench -- --model-path /path/to/qwen3.5-4b
//!
//! Env toggles honored by the inner training loop:
//!   KILN_GRAD_CHECKPOINT_SEGMENTS  (default 4; audit math assumed ON)
//!   KILN_NO_GRAD_CHECKPOINT=1      (disables checkpointing)

use anyhow::{Context, Result};
use candle_core::Device;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::GpuWeights;
use kiln_train::trainer::{sft_train, SftConfig};
use kiln_train::{ChatMessage, SftExample};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

const VOCAB_SIZE: u64 = 248_320;
const HIDDEN: u64 = 2560;

/// One measurement row.
struct Row {
    target_t: usize,
    actual_t: usize,
    baseline_mib: u64,
    peak_mib: u64,
    delta_mib: u64,
    step_secs: f64,
}

fn current_vram_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next()?.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Build an SftExample whose assistant response is long enough to hit ~target_t
/// tokens total after the chat template is applied.
fn build_example(tokenizer: &KilnTokenizer, target_t: usize) -> Result<(SftExample, usize)> {
    let base = "The quick brown fox jumps over the lazy dog near the river bank. \
                Scientists discovered a new species of deep-sea fish in the Pacific Ocean. \
                The quantum computer solved the optimization problem in record time. \
                She wrote a comprehensive analysis of market trends for the quarterly report. \
                Engineers refined the turbine blade geometry to shave another half percent of drag. ";

    // Build assistant content by repeating `base` until tokenized chat-template
    // output reaches target_t. We leave a short user prompt fixed.
    let user = "Summarize.".to_string();
    let mut assistant = String::new();

    loop {
        assistant.push_str(base);
        let messages = vec![
            kiln_core::tokenizer::ChatMessage {
                role: "user".to_string(),
                content: user.clone(),
            },
            kiln_core::tokenizer::ChatMessage {
                role: "assistant".to_string(),
                content: assistant.clone(),
            },
        ];
        let text = tokenizer
            .apply_chat_template(&messages)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = tokenizer
            .encode(&text)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if ids.len() >= target_t {
            // Trim from the END (assistant side) by dropping sentences until we're
            // at or just below target.
            while tokenizer
                .encode(
                    &tokenizer
                        .apply_chat_template(&[
                            kiln_core::tokenizer::ChatMessage {
                                role: "user".to_string(),
                                content: user.clone(),
                            },
                            kiln_core::tokenizer::ChatMessage {
                                role: "assistant".to_string(),
                                content: assistant.clone(),
                            },
                        ])
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .len()
                > target_t
            {
                if let Some(pos) = assistant[..assistant.len().saturating_sub(1)].rfind(". ") {
                    assistant.truncate(pos + 2);
                } else {
                    break;
                }
            }
            break;
        }
    }

    // Final measurement.
    let final_messages = vec![
        kiln_core::tokenizer::ChatMessage {
            role: "user".to_string(),
            content: user.clone(),
        },
        kiln_core::tokenizer::ChatMessage {
            role: "assistant".to_string(),
            content: assistant.clone(),
        },
    ];
    let final_text = tokenizer
        .apply_chat_template(&final_messages)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let final_ids = tokenizer
        .encode(&final_text)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_t = final_ids.len();

    let example = SftExample {
        messages: vec![
            ChatMessage {
                role: "user".to_string(),
                content: user,
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: assistant,
            },
        ],
    };

    Ok((example, actual_t))
}

/// Run one SFT step at the given T with a peak-VRAM poller. Returns a Row.
fn run_one(
    target_t: usize,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    baseline_mib: u64,
) -> Result<(Row, bool)> {
    eprintln!("\n--- T target = {target_t} ---");
    let (example, actual_t) = build_example(tokenizer, target_t)?;
    eprintln!("  Tokenized length: {actual_t} tokens (target {target_t})");

    let config = SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some(format!("flce-preflight-T{target_t}")),
        auto_load: false,
        checkpoint_interval: None,
    };
    let adapter_dir = std::env::temp_dir().join("kiln-flce-preflight");
    let _ = std::fs::create_dir_all(&adapter_dir);

    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(baseline_mib));
    let samples = Arc::new(AtomicU64::new(0));
    let stop_c = stop.clone();
    let peak_c = peak.clone();
    let samples_c = samples.clone();

    let poller = thread::spawn(move || {
        while !stop_c.load(Ordering::Relaxed) {
            let v = current_vram_mib();
            let mut cur = peak_c.load(Ordering::Relaxed);
            while v > cur {
                match peak_c.compare_exchange(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(now) => cur = now,
                }
            }
            samples_c.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(50));
        }
    });

    let start = Instant::now();
    let result = sft_train(
        &[example],
        &config,
        model_config,
        weights,
        tokenizer,
        &adapter_dir,
        &format!("flce-preflight-T{target_t}"),
        None,
    );
    let step_secs = start.elapsed().as_secs_f64();

    // Capture one final sample post-step before stopping the poller.
    let post_mib = current_vram_mib();
    let mut cur = peak.load(Ordering::Relaxed);
    while post_mib > cur {
        match peak.compare_exchange(cur, post_mib, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(now) => cur = now,
        }
    }
    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();
    let n_samples = samples.load(Ordering::Relaxed);
    let peak_mib = peak.load(Ordering::Relaxed);
    let delta_mib = peak_mib.saturating_sub(baseline_mib);

    // Clean up adapter dir for this run
    let _ = std::fs::remove_dir_all(adapter_dir.join(format!("flce-preflight-T{target_t}")));

    let ok = match result {
        Ok(_) => true,
        Err(e) => {
            eprintln!("  [!] training returned error (possibly OOM): {e}");
            false
        }
    };

    eprintln!(
        "  baseline {} MiB  peak {} MiB  delta {} MiB  samples {}  wall {:.1}s  ok={}",
        baseline_mib, peak_mib, delta_mib, n_samples, step_secs, ok
    );

    Ok((
        Row {
            target_t,
            actual_t,
            baseline_mib,
            peak_mib,
            delta_mib,
            step_secs,
        },
        ok,
    ))
}

fn parse_model_path() -> Result<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model-path" && i + 1 < args.len() {
            return Ok(PathBuf::from(&args[i + 1]));
        }
        i += 1;
    }
    anyhow::bail!("--model-path <dir> is required");
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("kiln=info,warn")
        .with_writer(std::io::stderr)
        .init();

    let model_path = parse_model_path()?;

    // Model + weights
    let model_config = ModelConfig::qwen3_5_4b();
    eprintln!(
        "=== FLCE preflight — Qwen3.5-4B, V={}, hidden={} ===",
        VOCAB_SIZE, HIDDEN
    );
    eprintln!("Loading model from {}", model_path.display());
    let model_weights =
        kiln_model::load_model(&model_path, &model_config).context("load_model")?;
    let device = kiln_server::device::select_device()?;
    if matches!(device, Device::Cpu) {
        anyhow::bail!("CUDA device required — preflight measures real VRAM");
    }
    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("from_model_weights")?;
    drop(model_weights);

    // Tokenizer
    let tokenizer = {
        let tok_file = model_path.join("tokenizer.json");
        if tok_file.exists() {
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    // Measure baseline post-load.
    let baseline_mib = current_vram_mib();
    eprintln!("Baseline VRAM (post-load): {} MiB", baseline_mib);

    // Warmup — a tiny T=256 run to move allocator past its cold state.
    eprintln!("Warmup pass at T~256...");
    let _ = run_one(256, &tokenizer, &model_config, &gpu_weights, baseline_mib);
    let baseline_mib = current_vram_mib();
    eprintln!("Baseline VRAM (post-warmup): {} MiB", baseline_mib);

    // Three measurements.
    let mut rows = Vec::new();
    for &t in &[2048usize, 8192, 16384] {
        match run_one(t, &tokenizer, &model_config, &gpu_weights, baseline_mib) {
            Ok((row, _ok)) => rows.push(row),
            Err(e) => {
                eprintln!("[!!] run at T={t} failed entirely: {e}");
                rows.push(Row {
                    target_t: t,
                    actual_t: 0,
                    baseline_mib,
                    peak_mib: 0,
                    delta_mib: 0,
                    step_secs: 0.0,
                });
            }
        }
    }

    // Final structured table to stdout.
    println!("\n=== FLCE preflight results ===");
    println!(
        "{:<10} {:<10} {:<14} {:<12} {:<14} {:<14} {:<14} {:<12} {:<10} {:<10}",
        "target_T",
        "actual_T",
        "peak_VRAM_MiB",
        "delta_MiB",
        "logits_bf16_GB",
        "logits_f32_GB",
        "grad_bf16_GB",
        "abc_GB",
        "abc/peak",
        "step_s"
    );
    for r in &rows {
        let t = r.actual_t as u64;
        let logits_bf16_bytes = t * VOCAB_SIZE * 2;
        let logits_f32_bytes = t * VOCAB_SIZE * 4;
        let grad_bf16_bytes = t * VOCAB_SIZE * 2;
        let logits_bf16_gb = logits_bf16_bytes as f64 / 1024.0_f64.powi(3);
        let logits_f32_gb = logits_f32_bytes as f64 / 1024.0_f64.powi(3);
        let grad_bf16_gb = grad_bf16_bytes as f64 / 1024.0_f64.powi(3);
        let abc_gb = logits_bf16_gb + logits_f32_gb + grad_bf16_gb;
        let peak_gb = r.peak_mib as f64 / 1024.0;
        let ratio = if peak_gb > 0.0 { abc_gb / peak_gb } else { 0.0 };
        println!(
            "{:<10} {:<10} {:<14} {:<12} {:<14.3} {:<14.3} {:<14.3} {:<12.3} {:<10.3} {:<10.1}",
            r.target_t,
            r.actual_t,
            r.peak_mib,
            r.delta_mib,
            logits_bf16_gb,
            logits_f32_gb,
            grad_bf16_gb,
            abc_gb,
            ratio,
            r.step_secs
        );
    }

    // JSON for easy scrape
    let json = serde_json::json!({
        "vocab_size": VOCAB_SIZE,
        "hidden": HIDDEN,
        "baseline_mib": baseline_mib,
        "rows": rows.iter().map(|r| {
            let t = r.actual_t as u64;
            let logits_bf16 = t * VOCAB_SIZE * 2;
            let logits_f32 = t * VOCAB_SIZE * 4;
            let grad_bf16 = t * VOCAB_SIZE * 2;
            serde_json::json!({
                "target_t": r.target_t,
                "actual_t": r.actual_t,
                "baseline_mib": r.baseline_mib,
                "peak_mib": r.peak_mib,
                "delta_mib": r.delta_mib,
                "step_secs": r.step_secs,
                "logits_bf16_bytes": logits_bf16,
                "logits_f32_bytes": logits_f32,
                "grad_bf16_bytes": grad_bf16,
            })
        }).collect::<Vec<_>>(),
    });
    println!("\n=== JSON ===");
    println!("{}", serde_json::to_string_pretty(&json)?);

    Ok(())
}
