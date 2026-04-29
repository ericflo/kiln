//! Phase 10 §1 — Liger-style RMSNorm with custom backward: A6000 SFT
//! peak-VRAM and loss parity validation bench.
//!
//! Companion to `flce_phase_a_validation_bench.rs`. Compares the new
//! `RmsNormCustomOp` (default) path against the
//! `KILN_DISABLE_RMSNORM_BACKWARD=1` fallback (pre-Phase-10 candle-op
//! chain) at three SFT lengths on Qwen3.5-4B + KILN_W4A16=1:
//!
//!   1. T=2048, custom op ON  — baseline final loss + peak VRAM.
//!   2. T=2048, custom op OFF — final loss must match (1) within 1e-3
//!      (parity check) and peak VRAM should be HIGHER (the candle-op
//!      chain materializes per-layer F32 intermediates that the custom
//!      op skips).
//!   3. T=4096, custom op ON  — peak VRAM (training peak) target cell.
//!   4. T=4096, custom op OFF — peak VRAM; (4) − (3) ≥ 0.5 GiB
//!      indicates the saved-tensor reduction lands as expected.
//!   5. T=8192, custom op ON  — does it fit, or how close to ceiling?
//!   6. T=8192, custom op OFF — baseline OOM signal (matches PR #637).
//!
//! Run:
//!   cargo run --release --features cuda -p kiln-server \
//!     --example phase10_rmsnorm_bench -- --model-path /workspace/qwen3.5-4b
//!
//! Env vars set/unset by this bench:
//!   KILN_DISABLE_RMSNORM_BACKWARD  (1 = fallback path; unset = custom op)
//!   KILN_USE_FLCE                  (always 1 — training peak is dominated
//!                                   by the head materialization without
//!                                   FLCE, which would mask the per-layer
//!                                   RMSNorm saved-tensor delta)
//!
//! Each cell records peak VRAM (background `nvidia-smi` poller @ 50 ms),
//! step wall-time, status (Ok / OOM / other-error), and final loss.
//! Output is markdown-table format on stdout for direct paste into the
//! audit doc.

use anyhow::{Context, Result};
use candle_core::Device;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::GpuWeights;
use kiln_train::trainer::{ProgressCallback, TrainingProgress, sft_train};
use kiln_train::{ChatMessage, SftConfig, SftExample};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
enum Status {
    Ok,
    Oom,
    OtherError(String),
}

impl Status {
    fn label(&self) -> &str {
        match self {
            Status::Ok => "ok",
            Status::Oom => "OOM",
            Status::OtherError(_) => "err",
        }
    }
}

#[derive(Debug)]
struct Row {
    target_t: usize,
    actual_t: usize,
    custom_op: bool,
    baseline_mib: u64,
    peak_mib: u64,
    delta_mib: u64,
    step_secs: f64,
    final_loss: Option<f64>,
    status: Status,
}

fn current_vram_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next()?.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn build_example(tokenizer: &KilnTokenizer, target_t: usize) -> Result<(SftExample, usize)> {
    // Same builder as flce_phase_a_validation_bench. Repeats a paragraph
    // until tokenized chat-template length is approximately target_t.
    let base = "The quick brown fox jumps over the lazy dog near the river bank. \
                Scientists discovered a new species of deep-sea fish in the Pacific Ocean. \
                The quantum computer solved the optimization problem in record time. \
                She wrote a comprehensive analysis of market trends for the quarterly report. \
                Engineers refined the turbine blade geometry to shave another half percent of drag. ";

    let user = "Summarize.".to_string();
    let mut assistant = String::new();

    loop {
        assistant.push_str(base);
        let messages = vec![
            kiln_core::tokenizer::ChatMessage {
                role: "user".to_string(),
                content: user.clone(),
                ..Default::default()
            },
            kiln_core::tokenizer::ChatMessage {
                role: "assistant".to_string(),
                content: assistant.clone(),
                ..Default::default()
            },
        ];
        let text = tokenizer
            .apply_chat_template(&messages)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = tokenizer
            .encode(&text)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if ids.len() >= target_t {
            while tokenizer
                .encode(
                    &tokenizer
                        .apply_chat_template(&[
                            kiln_core::tokenizer::ChatMessage {
                                role: "user".to_string(),
                                content: user.clone(),
                                ..Default::default()
                            },
                            kiln_core::tokenizer::ChatMessage {
                                role: "assistant".to_string(),
                                content: assistant.clone(),
                                ..Default::default()
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

    let final_messages = vec![
        kiln_core::tokenizer::ChatMessage {
            role: "user".to_string(),
            content: user.clone(),
            ..Default::default()
        },
        kiln_core::tokenizer::ChatMessage {
            role: "assistant".to_string(),
            content: assistant.clone(),
            ..Default::default()
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

#[allow(clippy::too_many_arguments)]
fn run_one(
    target_t: usize,
    custom_op: bool,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    baseline_mib: u64,
) -> Result<Row> {
    eprintln!(
        "\n--- T target = {target_t}  RMSNORM_CUSTOM_OP = {} ---",
        if custom_op { "ON" } else { "OFF" }
    );

    // SAFETY: bench is single-threaded at the env-var level (one run_one
    // at a time, no concurrent training threads).
    if custom_op {
        unsafe { std::env::remove_var("KILN_DISABLE_RMSNORM_BACKWARD") };
    } else {
        unsafe { std::env::set_var("KILN_DISABLE_RMSNORM_BACKWARD", "1") };
    }
    // FLCE on so the dominant peak comes from the per-layer activations
    // (where the RMSNorm saved-tensor delta lives), not from logits.
    unsafe { std::env::set_var("KILN_USE_FLCE", "1") };

    let (example, actual_t) = build_example(tokenizer, target_t)?;
    eprintln!("  Tokenized length: {actual_t} tokens (target {target_t})");

    let tag = if custom_op {
        "rmsnorm-on"
    } else {
        "rmsnorm-off"
    };
    let config = SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some(format!("phase10-rmsnorm-T{target_t}-{tag}")),
        auto_load: false,
        checkpoint_interval: None,
    };
    let adapter_dir = std::env::temp_dir().join("kiln-phase10-rmsnorm-bench");
    let _ = std::fs::create_dir_all(&adapter_dir);

    // VRAM poller — read nvidia-smi every 50 ms while the step runs.
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_t = stop.clone();
    let peak_t = peak.clone();
    let poller = thread::spawn(move || {
        while !stop_t.load(Ordering::Relaxed) {
            let mib = current_vram_mib();
            let prev = peak_t.load(Ordering::Relaxed);
            if mib > prev {
                peak_t.store(mib, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(50));
        }
    });

    let final_loss: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    let final_loss_cb = final_loss.clone();
    let progress: ProgressCallback = Box::new(move |step: TrainingProgress| {
        if let Ok(mut g) = final_loss_cb.lock() {
            *g = Some(step.loss);
        }
    });

    let t0 = Instant::now();
    let adapter_name = format!("phase10-rmsnorm-T{target_t}-{tag}");
    let result = sft_train(
        &[example],
        &config,
        model_config,
        weights,
        tokenizer,
        &adapter_dir,
        &adapter_name,
        Some(progress),
    );
    let elapsed = t0.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();

    let peak_mib = peak.load(Ordering::Relaxed);
    let delta_mib = peak_mib.saturating_sub(baseline_mib);
    let final_loss = *final_loss.lock().unwrap();

    let status = match result {
        Ok(_) => Status::Ok,
        Err(e) => {
            let msg = format!("{e:?}");
            if msg.contains("out of memory") || msg.contains("OutOfMemory") || msg.contains("CUDA_ERROR_OUT_OF_MEMORY") {
                Status::Oom
            } else {
                eprintln!("  step error: {msg}");
                Status::OtherError(msg)
            }
        }
    };

    eprintln!(
        "  status={}  peak={} MiB  delta={} MiB  step={:.2}s  final_loss={:?}",
        status.label(),
        peak_mib,
        delta_mib,
        elapsed,
        final_loss
    );

    Ok(Row {
        target_t,
        actual_t,
        custom_op,
        baseline_mib,
        peak_mib,
        delta_mib,
        step_secs: elapsed,
        final_loss,
        status,
    })
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

fn print_table(rows: &[Row]) {
    println!();
    println!("| T target | T actual | RMSNorm op | status | peak (MiB) | delta (MiB) | step (s) | final loss |");
    println!("|---------:|---------:|:----------:|:------:|-----------:|------------:|---------:|-----------:|");
    for r in rows {
        let loss = match r.final_loss {
            Some(l) => format!("{:.4}", l),
            None => "-".to_string(),
        };
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.2} | {} |",
            r.target_t,
            r.actual_t,
            if r.custom_op { "on" } else { "off" },
            r.status.label(),
            r.peak_mib,
            r.delta_mib,
            r.step_secs,
            loss,
        );
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("kiln=info,warn")
        .with_writer(std::io::stderr)
        .init();

    unsafe {
        std::env::remove_var("KILN_DISABLE_RMSNORM_BACKWARD");
        std::env::remove_var("KILN_DISABLE_RMSNORM_KERNEL");
    }

    let model_path = parse_model_path()?;
    let model_config = ModelConfig::qwen3_5_4b();
    eprintln!("=== Phase 10 §1 — RMSNorm CustomOp2 SFT validation (Qwen3.5-4B) ===");
    eprintln!("Loading model from {}", model_path.display());

    let model_weights = kiln_model::load_model_with_options(
        &model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load_model")?;
    let device = kiln_server::device::select_device()?;
    if matches!(device, Device::Cpu) {
        anyhow::bail!("CUDA device required");
    }
    let gpu_weights =
        GpuWeights::from_model_weights(&model_weights, &model_config, &device).context("from_model_weights")?;
    drop(model_weights);

    let tokenizer = {
        let tok_file = model_path.join("tokenizer.json");
        if tok_file.exists() {
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    let baseline_mib = current_vram_mib();
    eprintln!("Baseline VRAM (post-load): {} MiB", baseline_mib);

    // NOTE: warmup skipped — the original T=256 warmup left ~16 GB resident in the
    // candle CUDA allocator, eating the headroom required for subsequent cells. Cell 1
    // (T=2048 ON) absorbs the cold-allocator variance instead. Per-cell measurements
    // are reported relative to post-load baseline.
    eprintln!("Warmup skipped (would leave 16 GB cached in allocator).");

    let mut rows: Vec<Row> = Vec::new();

    // Cells 1-2: T=2048 parity check (same loss expected).
    rows.push(run_one(
        2048, true, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);
    rows.push(run_one(
        2048, false, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);

    // Cells 3-4: T=4096 saved-tensor delta.
    rows.push(run_one(
        4096, true, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);
    rows.push(run_one(
        4096, false, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);

    // Cells 5-6: T=8192 OOM check.
    rows.push(run_one(
        8192, true, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);
    rows.push(run_one(
        8192, false, &tokenizer, &model_config, &gpu_weights, baseline_mib,
    )?);

    print_table(&rows);

    // Parity check: cells 1 and 2 must match within 1e-3 if both completed.
    if let (Some(a), Some(b)) = (rows[0].final_loss, rows[1].final_loss) {
        let diff = (a - b).abs();
        eprintln!(
            "\nT=2048 loss parity: custom_op={a:.6}  fallback={b:.6}  diff={diff:.2e} (tol=1e-3)"
        );
        if diff > 1e-3 {
            anyhow::bail!(
                "T=2048 loss parity FAILED: custom_op={a} fallback={b} diff={diff} > 1e-3",
            );
        }
    } else {
        eprintln!("\nT=2048 parity check skipped (one or both cells failed)");
    }

    // Saved-tensor delta at T=4096: ON peak should be ≤ OFF peak − 500 MiB.
    let on_4k = rows.iter().find(|r| r.target_t == 4096 && r.custom_op);
    let off_4k = rows.iter().find(|r| r.target_t == 4096 && !r.custom_op);
    if let (Some(on), Some(off)) = (on_4k, off_4k) {
        if matches!(on.status, Status::Ok) && matches!(off.status, Status::Ok) {
            let savings = off.peak_mib as i64 - on.peak_mib as i64;
            eprintln!(
                "\nT=4096 saved-tensor delta: off={} MiB - on={} MiB = {} MiB savings (target ≥512)",
                off.peak_mib, on.peak_mib, savings,
            );
        }
    }

    Ok(())
}
