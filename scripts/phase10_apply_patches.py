#!/usr/bin/env python3
"""
Phase 10 §1 VRAM regression mechanism — Tier 1 + Tier 2 patcher.

Run after `git checkout <SHA>` in /workspace/kiln. Applies:
  - Tier 1: tracing::info!() at top of `rms_norm()` dispatch in
    crates/kiln-model/src/forward.rs (handles both c983ca7 and 1edecd7
    variants of the function body).
  - Tier 2: per-segment VRAM probes (eprintln "VRAM_PROBE <label> <MiB>")
    inside checkpointed_forward_backward() in crates/kiln-train/src/trainer.rs.
  - Slimmed bench: replaces crates/kiln-server/examples/phase10_rmsnorm_bench.rs
    with a single-cell version that runs only T=2048 + custom_op=true.
"""
import sys, re, pathlib

ROOT = pathlib.Path("/workspace/kiln")

# ---------------------------------------------------------------------------
# Tier 1: forward.rs rms_norm dispatch logging
# ---------------------------------------------------------------------------
FORWARD = ROOT / "crates/kiln-model/src/forward.rs"
src = FORWARD.read_text()

A_OLD = """pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let kernel_disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        let bwd_disabled = std::env::var("KILN_DISABLE_RMSNORM_BACKWARD").is_ok();
        if !kernel_disabled && !bwd_disabled && kiln_rmsnorm_kernel::supports(x, weight) {
            return kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd(x, weight, eps as f32)
                .context("fused_rmsnorm_with_autograd CustomOp2 failed");
        }
    }"""
A_NEW = """pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let kernel_disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        let bwd_disabled = std::env::var("KILN_DISABLE_RMSNORM_BACKWARD").is_ok();
        let supports = kiln_rmsnorm_kernel::supports(x, weight);
        let take_kernel = !kernel_disabled && !bwd_disabled && supports;
        tracing::info!(
            target: "kiln::rms_norm_dispatch",
            kernel_disabled,
            bwd_disabled,
            supports,
            take_kernel,
            shape = ?x.shape().dims(),
            "rms_norm dispatch"
        );
        if take_kernel {
            return kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd(x, weight, eps as f32)
                .context("fused_rmsnorm_with_autograd CustomOp2 failed");
        }
    }"""

B_OLD = """pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        if !disabled && kiln_rmsnorm_kernel::supports(x, weight) {
            return kiln_rmsnorm_kernel::fused_rmsnorm(x, weight, eps as f32)
                .context("fused_rmsnorm kernel failed");
        }
    }"""
B_NEW = """pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        let disabled = std::env::var("KILN_DISABLE_RMSNORM_KERNEL").is_ok();
        let supports = kiln_rmsnorm_kernel::supports(x, weight);
        let take_kernel = !disabled && supports;
        tracing::info!(
            target: "kiln::rms_norm_dispatch",
            kernel_disabled = disabled,
            supports,
            take_kernel,
            shape = ?x.shape().dims(),
            "rms_norm dispatch"
        );
        if take_kernel {
            return kiln_rmsnorm_kernel::fused_rmsnorm(x, weight, eps as f32)
                .context("fused_rmsnorm kernel failed");
        }
    }"""

if A_OLD in src:
    src = src.replace(A_OLD, A_NEW)
    print("forward.rs: Variant A (1edecd7) patched")
elif B_OLD in src:
    src = src.replace(B_OLD, B_NEW)
    print("forward.rs: Variant B (c983ca7) patched")
else:
    print("forward.rs: NEITHER variant matched! Aborting.", file=sys.stderr)
    sys.exit(1)

FORWARD.write_text(src)

# ---------------------------------------------------------------------------
# Tier 2: trainer.rs per-segment VRAM probes
# ---------------------------------------------------------------------------
TRAINER = ROOT / "crates/kiln-train/src/trainer.rs"
src = TRAINER.read_text()

HELPER = """
fn _phase10_vram_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next().map(|l| l.trim().to_string()))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

fn _phase10_vram_probe(label: &str) {
    if std::env::var("KILN_PHASE10_VRAM_PROBE").is_ok() {
        eprintln!("VRAM_PROBE\\t{}\\t{}", label, _phase10_vram_mib());
    }
}

"""

anchor = "pub fn sft_train("
if anchor not in src:
    print("trainer.rs: anchor 'pub fn sft_train(' not found", file=sys.stderr); sys.exit(1)
if "_phase10_vram_probe" not in src:
    src = src.replace(anchor, HELPER + anchor, 1)

# Probe injection points inside checkpointed_forward_backward
probe_anchors = [
    (
        "    let mut boundary_states: Vec<Tensor> = Vec::with_capacity(num_segments + 1);\n"
        "    boundary_states.push(embed_hidden.detach());\n"
        "\n"
        "    {\n"
        "        let mut current = boundary_states[0].clone();\n"
        "        for &(start, end) in segments.iter() {\n",

        "    let mut boundary_states: Vec<Tensor> = Vec::with_capacity(num_segments + 1);\n"
        "    boundary_states.push(embed_hidden.detach());\n"
        "    _phase10_vram_probe(\"after_embed\");\n"
        "\n"
        "    {\n"
        "        let mut current = boundary_states[0].clone();\n"
        "        for (boundary_idx, &(start, end)) in segments.iter().enumerate() {\n"
        "            _phase10_vram_probe(&format!(\"boundary_pre_{}\", boundary_idx));\n",
    ),
    (
        "            boundary_states.push(current.detach());\n"
        "            current = boundary_states.last().unwrap().clone();\n"
        "        }\n"
        "    }\n"
        "\n"
        "    // Step 2: For each segment, recompute with grad tracking and backprop.\n",

        "            boundary_states.push(current.detach());\n"
        "            current = boundary_states.last().unwrap().clone();\n"
        "            _phase10_vram_probe(&format!(\"boundary_post_{}\", boundary_idx));\n"
        "        }\n"
        "    }\n"
        "    _phase10_vram_probe(\"after_boundary_forward\");\n"
        "\n"
        "    // Step 2: For each segment, recompute with grad tracking and backprop.\n",
    ),
    (
        "    for seg_idx in 0..num_segments {\n"
        "        if let Some(tile) = tile_size {\n",

        "    for seg_idx in 0..num_segments {\n"
        "        _phase10_vram_probe(&format!(\"recompute_pre_{}\", seg_idx));\n"
        "        if let Some(tile) = tile_size {\n",
    ),
    # Tiled-path end-of-iteration probe
    (
        "            total_loss += seg_loss;\n"
        "            continue;\n"
        "        }\n",

        "            total_loss += seg_loss;\n"
        "            _phase10_vram_probe(&format!(\"recompute_post_tiled_{}\", seg_idx));\n"
        "            continue;\n"
        "        }\n",
    ),
    # Monolithic end-of-iteration probe (after accumulate_grads, before closing brace)
    (
        "        let grads = loss.backward().context(\"checkpointed backward pass\")?;\n"
        "        accumulate_grads(&mut accumulated_grads, &grads, &all_vars)?;\n"
        "    }\n",

        "        let grads = loss.backward().context(\"checkpointed backward pass\")?;\n"
        "        accumulate_grads(&mut accumulated_grads, &grads, &all_vars)?;\n"
        "        _phase10_vram_probe(&format!(\"recompute_post_mono_{}\", seg_idx));\n"
        "    }\n",
    ),
    # Final probe after the seg_idx loop completes (before avg_loss compute)
    (
        "    // Average loss across segments. In both monolithic and tiled paths the\n"
        "    // per-iteration `total_loss` accumulates the same segment-equivalent\n"
        "    // value, so dividing by `num_segments` recovers the mean cross-entropy\n"
        "    // over active positions.\n"
        "    let avg_loss = total_loss / num_segments as f64;\n",

        "    _phase10_vram_probe(\"after_recompute\");\n"
        "    // Average loss across segments. In both monolithic and tiled paths the\n"
        "    // per-iteration `total_loss` accumulates the same segment-equivalent\n"
        "    // value, so dividing by `num_segments` recovers the mean cross-entropy\n"
        "    // over active positions.\n"
        "    let avg_loss = total_loss / num_segments as f64;\n",
    ),
]

for old, new in probe_anchors:
    if new in src:
        continue
    if old not in src:
        print("trainer.rs: probe anchor not found:\n  ", repr(old[:120]), file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)

TRAINER.write_text(src)
print("trainer.rs: Tier 2 probes injected")

# ---------------------------------------------------------------------------
# Slim bench
# ---------------------------------------------------------------------------
BENCH = ROOT / "crates/kiln-server/examples/phase10_rmsnorm_bench.rs"
SLIM = '''//! Phase 10 §1 — VRAM regression MECHANISM bench (Tier 1 + Tier 2).
//!
//! Companion to docs/audits/PHASE10_VRAM_REGRESSION_MECHANISM.md. Runs a
//! single SFT cell at T=2048 with the kernel kill switch (set on the
//! command line) so the executed code path is byte-equal at c983ca7 and
//! 1edecd7. stderr emits dispatch logs (Tier 1) and per-segment VRAM
//! probes (Tier 2).

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

fn current_vram_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next().map(|l| l.trim().to_string()))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

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
        let text = tokenizer.apply_chat_template(&messages).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = tokenizer.encode(&text).map_err(|e| anyhow::anyhow!("{e}"))?;
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
        kiln_core::tokenizer::ChatMessage { role: "user".to_string(), content: user.clone(), ..Default::default() },
        kiln_core::tokenizer::ChatMessage { role: "assistant".to_string(), content: assistant.clone(), ..Default::default() },
    ];
    let final_text = tokenizer.apply_chat_template(&final_messages).map_err(|e| anyhow::anyhow!("{e}"))?;
    let final_ids = tokenizer.encode(&final_text).map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_t = final_ids.len();
    let example = SftExample {
        messages: vec![
            ChatMessage { role: "user".to_string(), content: user },
            ChatMessage { role: "assistant".to_string(), content: assistant },
        ],
    };
    Ok((example, actual_t))
}

fn run_one(
    target_t: usize,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    weights: &GpuWeights,
    label: &str,
) -> Result<()> {
    eprintln!("\\n--- T target = {target_t}  label = {label} ---");
    unsafe { std::env::set_var("KILN_USE_FLCE", "1") };
    let (example, actual_t) = build_example(tokenizer, target_t)?;
    eprintln!("  Tokenized length: {actual_t} tokens (target {target_t})");
    let config = SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some(format!("phase10-mech-{label}-T{target_t}")),
        auto_load: false,
        checkpoint_interval: None,
    };
    let adapter_dir = std::env::temp_dir().join("kiln-phase10-mech");
    let _ = std::fs::create_dir_all(&adapter_dir);
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_t = stop.clone();
    let peak_t = peak.clone();
    let poller = thread::spawn(move || {
        while !stop_t.load(Ordering::Relaxed) {
            let mib = current_vram_mib();
            let prev = peak_t.load(Ordering::Relaxed);
            if mib > prev { peak_t.store(mib, Ordering::Relaxed); }
            thread::sleep(Duration::from_millis(50));
        }
    });
    let final_loss: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    let final_loss_cb = final_loss.clone();
    let progress: ProgressCallback = Box::new(move |step: TrainingProgress| {
        if let Ok(mut g) = final_loss_cb.lock() { *g = Some(step.loss); }
    });
    let t0 = Instant::now();
    let adapter_name = format!("phase10-mech-{label}-T{target_t}");
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
    let final_loss = *final_loss.lock().unwrap();
    eprintln!(
        "  RESULT label={} T={} actual_t={} peak_mib={} step_secs={:.2} final_loss={:?} status={}",
        label,
        target_t,
        actual_t,
        peak_mib,
        elapsed,
        final_loss,
        if result.is_ok() { "ok" } else { "err" },
    );
    if let Err(e) = result {
        eprintln!("  step error: {e:?}");
    }
    Ok(())
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
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "kiln=info,warn".to_string()),
        )
        .with_writer(std::io::stderr)
        .init();
    unsafe {
        std::env::remove_var("KILN_DISABLE_RMSNORM_BACKWARD");
    }
    let model_path = parse_model_path()?;
    let model_config = ModelConfig::qwen3_5_4b();
    eprintln!("=== Phase 10 §1 — VRAM regression MECHANISM bench (Qwen3.5-4B) ===");
    eprintln!("Loading model from {}", model_path.display());
    let model_weights = kiln_model::load_model_with_options(
        &model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    ).context("load_model")?;
    let device = kiln_server::device::select_device()?;
    if matches!(device, Device::Cpu) {
        anyhow::bail!("CUDA device required");
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
    eprintln!("Warmup pass at T~256...");
    let _ = run_one(256, &tokenizer, &model_config, &gpu_weights, "warmup");
    let baseline_mib = current_vram_mib();
    eprintln!("Baseline VRAM (post-warmup): {} MiB", baseline_mib);
    run_one(2048, &tokenizer, &model_config, &gpu_weights, "main")?;
    Ok(())
}
'''
BENCH.write_text(SLIM)
print("phase10_rmsnorm_bench.rs: slim mechanism bench written")
