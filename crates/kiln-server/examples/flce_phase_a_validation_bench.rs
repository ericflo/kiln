//! Phase 10 — Fused Linear Cross-Entropy (FLCE) Phase A validation bench.
//!
//! Phase 10's preflight (PR #235, `flce_preflight_bench.rs`) measured peak VRAM
//! during one SFT step on Qwen3.5-4B at T ∈ {2048, 8192, 16384} on an A6000
//! **without** FLCE. Result: T=8192 and T=16384 OOM'd. That gating signal is
//! what motivated Phase A (PR #241, `KILN_USE_FLCE=1`, chunked-vocab
//! cross-entropy in pure candle).
//!
//! Phase A has now shipped in releases since `kiln-v0.2.0`, but the empirical
//! check that it actually unblocks long-context SFT on the GPU we ship on
//! (A6000 48 GB) has not been done. This bench closes that gap.
//!
//! Cells measured:
//!   1. T=2048, FLCE OFF — baseline loss (parity reference)
//!   2. T=2048, FLCE ON  — loss must match (1) to within 1e-3 (parity check)
//!   3. T=8192, FLCE ON  — must complete without OOM (Phase A's headline claim)
//!   4. T=16384, FLCE ON — must complete without OOM (Phase A's headline claim)
//!
//! Each cell records peak VRAM (background `nvidia-smi` poller @ 50 ms),
//! step wall-time, and final loss (captured via the `sft_train` progress
//! callback).
//!
//! Run:
//!   cargo run --release --features cuda -p kiln-server \
//!     --example flce_phase_a_validation_bench -- --model-path /path/to/qwen3.5-4b
//!
//! Env toggles honored by the inner training loop:
//!   KILN_GRAD_CHECKPOINT_SEGMENTS  (default 4; matches the preflight)
//!   KILN_NO_GRAD_CHECKPOINT=1      (disables checkpointing — not recommended)
//!
//! The bench sets/unsets `KILN_USE_FLCE` itself, so do NOT export it before
//! invoking the binary.

use anyhow::{Context, Result};
use candle_core::Device;
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_model::forward::GpuWeights;
use kiln_train::trainer::{ProgressCallback, sft_train};
use kiln_train::{ChatMessage, SftConfig, SftExample};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

const VOCAB_SIZE: u64 = 248_320;
const HIDDEN: u64 = 2560;
const PARITY_TOLERANCE: f64 = 1.0e-3;

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
            Status::OtherError(_) => "other-error",
        }
    }
}

#[derive(Debug)]
struct Row {
    target_t: usize,
    actual_t: usize,
    use_flce: bool,
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

/// Build an SftExample whose tokenized chat-template output is approximately
/// `target_t` tokens long. Mirrors the helper used by `flce_preflight_bench`.
fn build_example(tokenizer: &KilnTokenizer, target_t: usize) -> Result<(SftExample, usize)> {
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

/// Run one SFT step at the given T with the given FLCE setting. Captures peak
/// VRAM, step wall-time, final loss (via progress callback), and status.
fn run_one(
    target_t: usize,
    use_flce: bool,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    baseline_mib: u64,
) -> Result<Row> {
    eprintln!(
        "\n--- T target = {target_t}  FLCE = {} ---",
        if use_flce { "ON" } else { "OFF" }
    );

    // Set the env var deterministically for this run. SAFETY: bench is single-
    // threaded at the env-var level (one `run_one` at a time, no concurrent
    // training threads). See std::env::set_var docs for the multi-threaded
    // caveat we explicitly avoid here.
    if use_flce {
        unsafe { std::env::set_var("KILN_USE_FLCE", "1") };
    } else {
        unsafe { std::env::remove_var("KILN_USE_FLCE") };
    }

    let (example, actual_t) = build_example(tokenizer, target_t)?;
    eprintln!("  Tokenized length: {actual_t} tokens (target {target_t})");

    let tag = if use_flce { "flce-on" } else { "flce-off" };
    let config = SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some(format!("flce-phaseA-T{target_t}-{tag}")),
        auto_load: false,
        checkpoint_interval: None,
    };
    let adapter_dir = std::env::temp_dir().join("kiln-flce-phaseA-validation");
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

    // Capture final per-step loss via the progress callback.
    let last_loss = Arc::new(Mutex::new(None::<f64>));
    let last_loss_c = last_loss.clone();
    let progress_cb: ProgressCallback = Box::new(move |p| {
        if let Ok(mut slot) = last_loss_c.lock() {
            *slot = Some(p.loss);
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
        &format!("flce-phaseA-T{target_t}-{tag}"),
        Some(progress_cb),
    );
    let step_secs = start.elapsed().as_secs_f64();

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

    let _ = std::fs::remove_dir_all(adapter_dir.join(format!("flce-phaseA-T{target_t}-{tag}")));

    let final_loss = last_loss.lock().ok().and_then(|g| *g);
    let status = match &result {
        Ok(_) => Status::Ok,
        Err(e) => {
            let msg = format!("{e:#}").to_lowercase();
            if msg.contains("out of memory")
                || msg.contains("oom")
                || msg.contains("cuda_error_out_of_memory")
                || msg.contains("cudaerrormemoryallocation")
            {
                Status::Oom
            } else {
                Status::OtherError(format!("{e:#}"))
            }
        }
    };

    eprintln!(
        "  baseline {} MiB  peak {} MiB  delta {} MiB  samples {}  wall {:.1}s  loss={:?}  status={}",
        baseline_mib,
        peak_mib,
        delta_mib,
        n_samples,
        step_secs,
        final_loss,
        status.label()
    );
    if let Status::OtherError(ref e) = status {
        eprintln!("  [!] non-OOM error: {e}");
    }

    Ok(Row {
        target_t,
        actual_t,
        use_flce,
        baseline_mib,
        peak_mib,
        delta_mib,
        step_secs,
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

fn print_row(r: &Row) {
    let logits_bf16 = (r.actual_t as f64) * (VOCAB_SIZE as f64) * 2.0 / 1024.0_f64.powi(3);
    let logits_f32 = (r.actual_t as f64) * (VOCAB_SIZE as f64) * 4.0 / 1024.0_f64.powi(3);
    let grad_bf16 = (r.actual_t as f64) * (VOCAB_SIZE as f64) * 2.0 / 1024.0_f64.powi(3);
    let abc = logits_bf16 + logits_f32 + grad_bf16;
    let peak_gb = r.peak_mib as f64 / 1024.0;
    let ratio = if peak_gb > 0.0 { abc / peak_gb } else { 0.0 };
    let loss_str = match r.final_loss {
        Some(v) => format!("{v:.6}"),
        None => "n/a".into(),
    };
    println!(
        "{:<10} {:<10} {:<6} {:<14} {:<12} {:<12} {:<10} {:<10} {:<12}",
        r.target_t,
        r.actual_t,
        if r.use_flce { "ON" } else { "OFF" },
        r.peak_mib,
        r.delta_mib,
        format!("{abc:.3}"),
        format!("{ratio:.3}"),
        format!("{:.1}", r.step_secs),
        loss_str,
    );
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("kiln=info,warn")
        .with_writer(std::io::stderr)
        .init();

    // Make sure we don't inherit a stale KILN_USE_FLCE from the shell.
    unsafe { std::env::remove_var("KILN_USE_FLCE") };

    let model_path = parse_model_path()?;

    let model_config = ModelConfig::qwen3_5_4b();
    eprintln!(
        "=== FLCE Phase A validation — Qwen3.5-4B, V={}, hidden={} ===",
        VOCAB_SIZE, HIDDEN
    );
    eprintln!("Loading model from {}", model_path.display());
    let model_weights = kiln_model::load_model_with_options(
        &model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load_model")?;
    let device = kiln_server::device::select_device()?;
    if matches!(device, Device::Cpu) {
        anyhow::bail!("CUDA device required — validation measures real VRAM");
    }
    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("from_model_weights")?;
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

    // Warmup at T~256 (FLCE off) to push allocator past cold state and
    // to surface any obvious wiring bug before the real cells.
    eprintln!("Warmup pass at T~256 (FLCE OFF)...");
    let _ = run_one(
        256,
        false,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    );
    let baseline_mib = current_vram_mib();
    eprintln!("Baseline VRAM (post-warmup): {} MiB", baseline_mib);

    // The four cells.
    let mut rows: Vec<Row> = Vec::new();

    // 1. T=2048, FLCE OFF — parity reference.
    let r = run_one(
        2048,
        false,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    let parity_off_loss = r.final_loss;
    rows.push(r);

    // 2. T=2048, FLCE ON — must match (1) within PARITY_TOLERANCE.
    let r = run_one(
        2048,
        true,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    let parity_on_loss = r.final_loss;
    rows.push(r);

    // 3. T=8192, FLCE ON.
    let r = run_one(
        8192,
        true,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // 4. T=16384, FLCE ON.
    let r = run_one(
        16384,
        true,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // Final structured table to stdout.
    println!("\n=== FLCE Phase A validation results ===");
    println!(
        "{:<10} {:<10} {:<6} {:<14} {:<12} {:<12} {:<10} {:<10} {:<12}",
        "target_T", "actual_T", "FLCE", "peak_VRAM_MiB", "delta_MiB", "abc_GB", "abc/peak", "step_s",
        "loss"
    );
    for r in &rows {
        print_row(r);
    }

    // Parity verdict.
    let parity_delta = match (parity_off_loss, parity_on_loss) {
        (Some(off), Some(on)) => Some((off - on).abs()),
        _ => None,
    };
    let parity_pass = parity_delta.map(|d| d <= PARITY_TOLERANCE);

    println!("\n=== Parity (T=2048 FLCE off vs on) ===");
    println!(
        "  off_loss = {:?}\n  on_loss  = {:?}\n  delta    = {:?}\n  tolerance= {:.1e}\n  result   = {}",
        parity_off_loss,
        parity_on_loss,
        parity_delta,
        PARITY_TOLERANCE,
        match parity_pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "n/a",
        },
    );

    // Per-cell verdicts.
    let cell = |t: usize, flce: bool| -> Option<&Row> {
        rows.iter().find(|r| r.target_t == t && r.use_flce == flce)
    };
    let t2048_off = cell(2048, false);
    let t2048_on = cell(2048, true);
    let t8192_on = cell(8192, true);
    let t16384_on = cell(16384, true);

    println!("\n=== Cell summary ===");
    for (label, r) in [
        ("T=2048   FLCE=OFF", t2048_off),
        ("T=2048   FLCE=ON ", t2048_on),
        ("T=8192   FLCE=ON ", t8192_on),
        ("T=16384  FLCE=ON ", t16384_on),
    ] {
        match r {
            Some(r) => println!(
                "  {label}  -> status={:<10}  peak={} MiB  step={:.1}s  loss={:?}",
                r.status.label(),
                r.peak_mib,
                r.step_secs,
                r.final_loss
            ),
            None => println!("  {label}  -> not run"),
        }
    }

    let phase_a_unblocked = matches!(t8192_on.map(|r| &r.status), Some(Status::Ok))
        && matches!(t16384_on.map(|r| &r.status), Some(Status::Ok));
    let parity_ok = parity_pass.unwrap_or(false);

    let verdict = if phase_a_unblocked && parity_ok {
        "GREEN — Phase A unblocks long-context SFT on A6000"
    } else if phase_a_unblocked && !parity_ok {
        "AMBER — Phase A unblocks long-context SFT but parity exceeds tolerance"
    } else {
        "RED — Phase A insufficient on A6000"
    };
    println!("\n=== Verdict: {verdict} ===");

    // Machine-readable JSON for the audit doc.
    let json = serde_json::json!({
        "vocab_size": VOCAB_SIZE,
        "hidden": HIDDEN,
        "baseline_mib": baseline_mib,
        "parity": {
            "off_loss": parity_off_loss,
            "on_loss": parity_on_loss,
            "delta": parity_delta,
            "tolerance": PARITY_TOLERANCE,
            "pass": parity_pass,
        },
        "verdict": verdict,
        "rows": rows.iter().map(|r| {
            serde_json::json!({
                "target_t": r.target_t,
                "actual_t": r.actual_t,
                "use_flce": r.use_flce,
                "baseline_mib": r.baseline_mib,
                "peak_mib": r.peak_mib,
                "delta_mib": r.delta_mib,
                "step_secs": r.step_secs,
                "final_loss": r.final_loss,
                "status": r.status.label(),
            })
        }).collect::<Vec<_>>(),
    });
    println!("\n=== JSON ===");
    println!("{}", serde_json::to_string_pretty(&json)?);

    Ok(())
}
