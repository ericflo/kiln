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
//! Cells measured (Phase A baseline + Phase 10 GDN-streaming-audit):
//!   1. T=2048, FLCE OFF — baseline loss (parity reference)
//!   2. T=2048, FLCE ON  — loss must match (1) to within 1e-3 (parity check)
//!   3. T=8192, FLCE ON  — must complete without OOM (Phase A's headline claim)
//!   4. T=16384, FLCE ON — must complete without OOM (Phase A's headline claim)
//!   5. T=2048, FLCE ON, STREAMING ON tile=default — control: peak VRAM
//!      should match cell (2) within polling noise if the streaming dispatch
//!      is unreachable from the SFT path.
//!   6. T=8192, FLCE ON, STREAMING ON tile=default — headline: does
//!      `KILN_STREAMING_PREFILL=1` unblock T=8192 SFT on A6000?
//!   7. T=8192, FLCE ON, STREAMING ON tile=4096
//!   8. T=8192, FLCE ON, STREAMING ON tile=2048
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
//! The bench sets/unsets `KILN_USE_FLCE`, `KILN_STREAMING_PREFILL`, and
//! `KILN_STREAMING_TILE_TOKENS` itself, so do NOT export them before invoking
//! the binary.

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
    /// Streaming GDN prefill toggle. `None` means the env var is unset (matches
    /// the historical Phase A cells); `Some(true)` / `Some(false)` set
    /// `KILN_STREAMING_PREFILL=1|0` for the cell.
    streaming: Option<bool>,
    /// Optional `KILN_STREAMING_TILE_TOKENS` override for the cell. `None`
    /// leaves the env unset (default tile = 8192 tokens).
    tile_tokens: Option<usize>,
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

/// Run one SFT step at the given T with the given FLCE setting. Captures peak
/// VRAM, step wall-time, final loss (via progress callback), and status.
///
/// `streaming`: if `Some(b)` sets `KILN_STREAMING_PREFILL=b`, otherwise unsets
/// it. `tile_tokens`: if `Some(n)` sets `KILN_STREAMING_TILE_TOKENS=n`,
/// otherwise unsets it. Both env reads happen inside `kiln-model::forward` on
/// every prefill, so per-cell mutation works the same way `KILN_USE_FLCE` does.
#[allow(clippy::too_many_arguments)]
fn run_one(
    target_t: usize,
    use_flce: bool,
    streaming: Option<bool>,
    tile_tokens: Option<usize>,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    baseline_mib: u64,
) -> Result<Row> {
    let stream_label = match streaming {
        Some(true) => "ON",
        Some(false) => "OFF",
        None => "unset",
    };
    let tile_label = tile_tokens
        .map(|n| n.to_string())
        .unwrap_or_else(|| "default".to_string());
    eprintln!(
        "\n--- T target = {target_t}  FLCE = {}  STREAMING = {stream_label}  TILE = {tile_label} ---",
        if use_flce { "ON" } else { "OFF" }
    );

    // Set the env vars deterministically for this run. SAFETY: bench is single-
    // threaded at the env-var level (one `run_one` at a time, no concurrent
    // training threads). See std::env::set_var docs for the multi-threaded
    // caveat we explicitly avoid here.
    if use_flce {
        unsafe { std::env::set_var("KILN_USE_FLCE", "1") };
    } else {
        unsafe { std::env::remove_var("KILN_USE_FLCE") };
    }
    match streaming {
        Some(true) => unsafe { std::env::set_var("KILN_STREAMING_PREFILL", "1") },
        Some(false) => unsafe { std::env::set_var("KILN_STREAMING_PREFILL", "0") },
        None => unsafe { std::env::remove_var("KILN_STREAMING_PREFILL") },
    }
    match tile_tokens {
        Some(n) => unsafe { std::env::set_var("KILN_STREAMING_TILE_TOKENS", n.to_string()) },
        None => unsafe { std::env::remove_var("KILN_STREAMING_TILE_TOKENS") },
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
        streaming,
        tile_tokens,
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
    let stream_label = match r.streaming {
        Some(true) => "ON",
        Some(false) => "OFF",
        None => "unset",
    };
    let tile_label = r
        .tile_tokens
        .map(|n| n.to_string())
        .unwrap_or_else(|| "default".to_string());
    println!(
        "{:<10} {:<10} {:<6} {:<7} {:<8} {:<14} {:<12} {:<12} {:<10} {:<10} {:<12}",
        r.target_t,
        r.actual_t,
        if r.use_flce { "ON" } else { "OFF" },
        stream_label,
        tile_label,
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

    // Make sure we don't inherit stale toggles from the shell.
    unsafe { std::env::remove_var("KILN_USE_FLCE") };
    unsafe { std::env::remove_var("KILN_STREAMING_PREFILL") };
    unsafe { std::env::remove_var("KILN_STREAMING_TILE_TOKENS") };

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
        None,
        None,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    );
    let baseline_mib = current_vram_mib();
    eprintln!("Baseline VRAM (post-warmup): {} MiB", baseline_mib);

    // Cells: original Phase A baseline (1-4) + streaming-prefill follow-up
    // (5-8) for the GDN training-streaming audit.
    let mut rows: Vec<Row> = Vec::new();

    // 1. T=2048, FLCE OFF — parity reference.
    let r = run_one(
        2048,
        false,
        None,
        None,
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
        None,
        None,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    let parity_on_loss = r.final_loss;
    rows.push(r);

    // 3. Control (BEFORE the OOM cells): T=2048 FLCE ON STREAMING ON
    //    (default tile). If streaming were reachable from the SFT path, peak
    //    VRAM would change here vs cell 2; if it's unreachable (the audit's
    //    hypothesis), peak should match cell 2 within nvidia-smi 50 ms
    //    polling noise. We run this BEFORE cells 4/5 so post-OOM CUDA
    //    allocator residue does not contaminate the peak measurement.
    let r = run_one(
        2048,
        true,
        Some(true),
        None,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    let stream_on_t2048_loss = r.final_loss;
    rows.push(r);

    // 4. T=8192, FLCE ON.
    let r = run_one(
        8192,
        true,
        None,
        None,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // 5. T=16384, FLCE ON.
    let r = run_one(
        16384,
        true,
        None,
        None,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // 6. Headline: T=8192 FLCE ON STREAMING ON (default tile = 8192). At the
    //    default tile size (which equals T=8192) no actual tiling happens
    //    even if streaming were reachable, so this primarily tests that the
    //    flag is parsed without crashing.
    let r = run_one(
        8192,
        true,
        Some(true),
        None,
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // 7. T=8192 FLCE ON STREAMING ON tile=4096 (would tile into 2 chunks).
    let r = run_one(
        8192,
        true,
        Some(true),
        Some(4096),
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // 8. T=8192 FLCE ON STREAMING ON tile=2048 (would tile into 4 chunks —
    //    if streaming were reachable, this is the most aggressive memory
    //    reduction available and should show the largest peak VRAM delta).
    let r = run_one(
        8192,
        true,
        Some(true),
        Some(2048),
        &tokenizer,
        &model_config,
        &gpu_weights,
        baseline_mib,
    )?;
    rows.push(r);

    // Final structured table to stdout.
    println!("\n=== FLCE Phase A + GDN-streaming-audit validation results ===");
    println!(
        "{:<10} {:<10} {:<6} {:<7} {:<8} {:<14} {:<12} {:<12} {:<10} {:<10} {:<12}",
        "target_T",
        "actual_T",
        "FLCE",
        "STREAM",
        "TILE",
        "peak_VRAM_MiB",
        "delta_MiB",
        "abc_GB",
        "abc/peak",
        "step_s",
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

    // Per-cell verdicts. The original Phase A cells use streaming=None;
    // the GDN-streaming audit cells use streaming=Some(true).
    let cell = |t: usize, flce: bool, streaming: Option<bool>, tile: Option<usize>| -> Option<&Row> {
        rows.iter().find(|r| {
            r.target_t == t
                && r.use_flce == flce
                && r.streaming == streaming
                && r.tile_tokens == tile
        })
    };
    let t2048_off = cell(2048, false, None, None);
    let t2048_on = cell(2048, true, None, None);
    let t8192_on = cell(8192, true, None, None);
    let t16384_on = cell(16384, true, None, None);
    let t2048_on_stream = cell(2048, true, Some(true), None);
    let t8192_on_stream = cell(8192, true, Some(true), None);
    let t8192_on_stream_4096 = cell(8192, true, Some(true), Some(4096));
    let t8192_on_stream_2048 = cell(8192, true, Some(true), Some(2048));

    println!("\n=== Cell summary ===");
    for (label, r) in [
        ("T=2048   FLCE=ON  STREAM=unset           ", t2048_on),
        ("T=2048   FLCE=OFF STREAM=unset           ", t2048_off),
        ("T=8192   FLCE=ON  STREAM=unset           ", t8192_on),
        ("T=16384  FLCE=ON  STREAM=unset           ", t16384_on),
        ("T=2048   FLCE=ON  STREAM=ON  tile=default", t2048_on_stream),
        ("T=8192   FLCE=ON  STREAM=ON  tile=default", t8192_on_stream),
        ("T=8192   FLCE=ON  STREAM=ON  tile=4096   ", t8192_on_stream_4096),
        ("T=8192   FLCE=ON  STREAM=ON  tile=2048   ", t8192_on_stream_2048),
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

    let phase_a_verdict = if phase_a_unblocked && parity_ok {
        "GREEN — Phase A unblocks long-context SFT on A6000"
    } else if phase_a_unblocked && !parity_ok {
        "AMBER — Phase A unblocks long-context SFT but parity exceeds tolerance"
    } else {
        "RED — Phase A insufficient on A6000"
    };
    println!("\n=== Phase A verdict: {phase_a_verdict} ===");

    // GDN-streaming-audit verdict. The hypothesis is that streaming GDN
    // prefill is unreachable from `sft_train` (training calls
    // `model_forward_segment`, which does not dispatch through
    // `model_forward_paged_streaming*`). Falsifying the hypothesis would mean
    // T=8192 + STREAMING ON completes without OOM, or T=2048 + STREAMING ON
    // shows a peak-VRAM delta vs T=2048 + STREAMING unset that is meaningfully
    // larger than the nvidia-smi 50 ms polling noise (~50-100 MiB).
    let stream_t8192_ok = matches!(t8192_on_stream.map(|r| &r.status), Some(Status::Ok));
    let stream_t8192_4096_ok =
        matches!(t8192_on_stream_4096.map(|r| &r.status), Some(Status::Ok));
    let stream_t8192_2048_ok =
        matches!(t8192_on_stream_2048.map(|r| &r.status), Some(Status::Ok));
    let any_8192_stream_ok = stream_t8192_ok || stream_t8192_4096_ok || stream_t8192_2048_ok;

    let t2048_peak_delta_mib = match (t2048_on, t2048_on_stream) {
        (Some(a), Some(b)) => Some(b.peak_mib as i64 - a.peak_mib as i64),
        _ => None,
    };

    let stream_audit_verdict = if any_8192_stream_ok {
        "GREEN — streaming GDN prefill at T=8192 SFT unblocks long-context training on A6000"
    } else {
        // T=8192 still OOM with streaming ON. Check the T=2048 control to
        // confirm streaming has no effect on training-time peak (not just
        // insufficient).
        match t2048_peak_delta_mib {
            Some(delta) if delta.abs() <= 200 => {
                "RED — KILN_STREAMING_PREFILL has no effect on SFT path; \
                 streaming dispatch is not wired into training-time forward (model_forward_segment)"
            }
            Some(delta) if delta < 0 => {
                "AMBER — T=8192 still OOMs with streaming ON, but T=2048 shows a peak-VRAM \
                 reduction; partial reachability suspected"
            }
            _ => {
                "RED — streaming GDN prefill at T=8192 SFT still OOMs on A6000"
            }
        }
    };
    println!("\n=== GDN-streaming-audit verdict: {stream_audit_verdict} ===");
    if let Some(delta) = t2048_peak_delta_mib {
        println!(
            "T=2048 peak delta (STREAMING ON vs unset): {} MiB  (control for streaming reachability)",
            delta
        );
    }

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
        "phase_a_verdict": phase_a_verdict,
        "stream_audit_verdict": stream_audit_verdict,
        "stream_audit_t2048_peak_delta_mib": t2048_peak_delta_mib,
        "stream_on_t2048_loss": stream_on_t2048_loss,
        "rows": rows.iter().map(|r| {
            serde_json::json!({
                "target_t": r.target_t,
                "actual_t": r.actual_t,
                "use_flce": r.use_flce,
                "streaming": r.streaming,
                "tile_tokens": r.tile_tokens,
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
