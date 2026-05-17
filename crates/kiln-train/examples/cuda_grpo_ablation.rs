//! CUDA GRPO ablation runner.
//!
//! Drives `grpo_train_jsonl` against a single dataset under one of a small set
//! of named configurations, so the same dataset / model / seed can be run
//! through each config in turn and the resulting progress logs compared.
//!
//! Output: `progress step=N/Total epoch=1/1 loss=... vram_mib=...` lines per
//! group, plus a final `adapter=...` and `elapsed_secs=...` line. A wrapping
//! shell loop captures one log per mode and the analysis step diffs the loss
//! curves across modes.
//!
//! See the Phase 1 / Phase 2 GRPO modernization design for the modes.

use anyhow::Result;

#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[cfg(feature = "cuda")]
use std::thread;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use kiln_core::config::ModelConfig;
#[cfg(feature = "cuda")]
use kiln_core::tokenizer::KilnTokenizer;
#[cfg(feature = "cuda")]
use kiln_model::forward::GpuWeights;
#[cfg(feature = "cuda")]
use kiln_train::trainer::grpo_train_jsonl;
#[cfg(feature = "cuda")]
use kiln_train::{
    AdvantageMode, GrpoConfig, IsLevel, KlEstimator, LossAggregation, Optimizer, ReferencePolicy,
};

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// Vanilla DeepSeekMath / R1 recipe — historical kiln defaults.
    Baseline,
    /// Phase 1 stack: DrGrpo + TokenLevel + Clip-Higher 0.20/0.28 +
    /// dynamic_sampling, KL k1 retained (Magistral-like). KL kept on so the
    /// reference-forward overhead is the same as Baseline.
    Phase1,
    /// Phase 1 + GSPO sequence-level IS.
    Phase1Gspo,
    /// Phase 1 + CISPO weight-clip IS.
    Phase1Cispo,
    /// Phase 1 + ReferencePolicy::None (skip reference forward, REINFORCE
    /// with group-relative advantages).
    Phase1Reinforce,
    /// Phase 1 + ReferencePolicy::Ema (refresh every 8 groups, decay 0.0).
    Phase3Ema,
    /// Phase 1 + ReferencePolicy::Ema(decay=0.9, refresh=8) (slow-moving).
    Phase3EmaSlow,
    /// Phase 1 + selective KL (entropy_aware_kl_quantile = 0.8).
    Phase3KlCov,
    /// Phase 1 + EMA(decay=0, refresh=8) + selective KL (q=0.8).
    Phase3EmaKlCov,
}

#[cfg(feature = "cuda")]
impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "phase1" => Ok(Self::Phase1),
            "phase1_gspo" | "gspo" => Ok(Self::Phase1Gspo),
            "phase1_cispo" | "cispo" => Ok(Self::Phase1Cispo),
            "phase1_reinforce" | "reinforce" => Ok(Self::Phase1Reinforce),
            "phase3_ema" | "ema" => Ok(Self::Phase3Ema),
            "phase3_ema_slow" | "ema_slow" => Ok(Self::Phase3EmaSlow),
            "phase3_kl_cov" | "kl_cov" => Ok(Self::Phase3KlCov),
            "phase3_ema_kl_cov" | "ema_kl_cov" => Ok(Self::Phase3EmaKlCov),
            other => anyhow::bail!(
                "unknown --mode {other}; expected one of: baseline, phase1, phase1_gspo, \
                 phase1_cispo, phase1_reinforce"
            ),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Phase1 => "phase1",
            Self::Phase1Gspo => "phase1_gspo",
            Self::Phase1Cispo => "phase1_cispo",
            Self::Phase1Reinforce => "phase1_reinforce",
            Self::Phase3Ema => "phase3_ema",
            Self::Phase3EmaSlow => "phase3_ema_slow",
            Self::Phase3KlCov => "phase3_kl_cov",
            Self::Phase3EmaKlCov => "phase3_ema_kl_cov",
        }
    }

    fn apply(self, base: GrpoConfig) -> GrpoConfig {
        match self {
            // Historical pre-#1045 DeepSeekMath/R1 recipe. With the defaults
            // flipped, `GrpoConfig::default()` is now Phase 1, so the
            // baseline mode must explicitly restore the old knobs.
            Self::Baseline => GrpoConfig {
                advantage_mode: AdvantageMode::Vanilla,
                loss_aggregation: LossAggregation::PerSample,
                clip_epsilon: 0.20,
                clip_eps_high: None,
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: false,
                is_level: IsLevel::Token,
                reference_policy: ReferencePolicy::BasePerStep,
                entropy_aware_kl_quantile: None,
                ..base
            },
            Self::Phase1 => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                ..base
            },
            Self::Phase1Gspo => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                is_level: IsLevel::Sequence,
                ..base
            },
            Self::Phase1Cispo => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                is_level: IsLevel::Cispo,
                ..base
            },
            Self::Phase1Reinforce => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                dynamic_sampling: true,
                reference_policy: ReferencePolicy::None,
                ..base
            },
            Self::Phase3Ema => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                reference_policy: ReferencePolicy::Ema {
                    decay: 0.0,
                    refresh_every: 8,
                },
                ..base
            },
            Self::Phase3EmaSlow => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                reference_policy: ReferencePolicy::Ema {
                    decay: 0.9,
                    refresh_every: 8,
                },
                ..base
            },
            Self::Phase3KlCov => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                entropy_aware_kl_quantile: Some(0.8),
                ..base
            },
            Self::Phase3EmaKlCov => GrpoConfig {
                advantage_mode: AdvantageMode::DrGrpo,
                loss_aggregation: LossAggregation::TokenLevel,
                clip_epsilon: 0.20,
                clip_eps_high: Some(0.28),
                kl_estimator: KlEstimator::K1,
                dynamic_sampling: true,
                reference_policy: ReferencePolicy::Ema {
                    decay: 0.0,
                    refresh_every: 8,
                },
                entropy_aware_kl_quantile: Some(0.8),
                ..base
            },
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    data: PathBuf,
    model_path: PathBuf,
    output_dir: PathBuf,
    adapter_name: String,
    mode: Mode,
    max_groups: Option<usize>,
    lora_rank: usize,
    lora_alpha: f32,
    learning_rate: f64,
    seed: u64,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut data = None;
        let mut model_path = None;
        let mut output_dir = std::env::temp_dir().join("kiln-cuda-grpo-ablation");
        let mut adapter_name = "grpo-ablation".to_string();
        let mut mode = Mode::Baseline;
        let mut max_groups: Option<usize> = None;
        let mut lora_rank = 8usize;
        let mut lora_alpha = 16.0f32;
        let mut learning_rate = 1e-5f64;
        let mut seed = 0x6752_504f_u64;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--data" => data = Some(PathBuf::from(args.next().context("--data needs a value")?)),
                "--model" => {
                    model_path = Some(PathBuf::from(
                        args.next().context("--model needs a value")?,
                    ))
                }
                "--output" => output_dir = PathBuf::from(args.next().context("--output needs a value")?),
                "--adapter" => adapter_name = args.next().context("--adapter needs a value")?,
                "--mode" => mode = Mode::parse(&args.next().context("--mode needs a value")?)?,
                "--max-groups" => {
                    max_groups = Some(
                        args.next()
                            .context("--max-groups needs a value")?
                            .parse()
                            .context("--max-groups must be a positive integer")?,
                    )
                }
                "--rank" => {
                    lora_rank = args
                        .next()
                        .context("--rank needs a value")?
                        .parse()
                        .context("--rank must be a positive integer")?
                }
                "--alpha" => {
                    lora_alpha = args
                        .next()
                        .context("--alpha needs a value")?
                        .parse()
                        .context("--alpha must be a float")?
                }
                "--lr" => {
                    learning_rate = args
                        .next()
                        .context("--lr needs a value")?
                        .parse()
                        .context("--lr must be a float")?
                }
                "--seed" => {
                    seed = args
                        .next()
                        .context("--seed needs a value")?
                        .parse()
                        .context("--seed must be a u64")?
                }
                "--help" | "-h" => {
                    println!(
                        "cuda_grpo_ablation --data <jsonl> --model <dir> [--output <dir>] \
                         [--adapter <name>] --mode <baseline|phase1|phase1_gspo|phase1_cispo|\
                         phase1_reinforce> [--max-groups N] [--rank N] [--alpha F] [--lr F] \
                         [--seed N]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unexpected argument: {other}"),
            }
        }

        Ok(Args {
            data: data.context("--data is required")?,
            model_path: model_path.context("--model is required")?,
            output_dir,
            adapter_name,
            mode,
            max_groups,
            lora_rank,
            lora_alpha,
            learning_rate,
            seed,
        })
    }
}

#[cfg(feature = "cuda")]
fn current_vram_mib() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next()?.trim().parse::<u64>().ok())
        .unwrap_or(0)
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
fn maybe_subset_dataset(input: &PathBuf, max_groups: Option<usize>) -> Result<PathBuf> {
    let Some(max) = max_groups else {
        return Ok(input.clone());
    };
    let raw = std::fs::read_to_string(input)
        .with_context(|| format!("reading {}", input.display()))?;
    let mut buf = String::new();
    let mut kept = 0usize;
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if kept >= max {
            break;
        }
        buf.push_str(line);
        buf.push('\n');
        kept += 1;
    }
    let out = input.with_extension(format!("max{max}.jsonl"));
    std::fs::write(&out, buf).with_context(|| format!("writing {}", out.display()))?;
    println!("subset_dataset={} kept_groups={kept}", out.display());
    Ok(out)
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse()?;
    let start = Instant::now();
    let baseline_mib = current_vram_mib();
    println!("mode={} baseline_vram_mib={}", args.mode.as_str(), baseline_mib);

    let tokenizer = load_tokenizer(&args.model_path)?;
    let dataset_path = maybe_subset_dataset(&args.data, args.max_groups)?;

    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA is not available in this build/runtime"
    );
    let device = candle_core::Device::new_cuda(0).context("create CUDA device 0")?;
    let model_config = ModelConfig::qwen3_5_4b();
    println!("loading_model={}", args.model_path.display());
    let model_weights = kiln_model::load_model_with_options(
        &args.model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load model weights")?;
    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("transfer weights to CUDA")?;
    drop(model_weights);
    println!("model_loaded_vram_mib={}", current_vram_mib());

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    // Base config: kiln Phase-1+2 defaults (which still match historical
    // behavior). Mode::apply layers the desired overrides on top.
    let base = GrpoConfig {
        learning_rate: args.learning_rate,
        lora_rank: args.lora_rank,
        lora_alpha: args.lora_alpha,
        output_name: Some(args.adapter_name.clone()),
        auto_load: false,
        seed: Some(args.seed),
        optimizer: Optimizer::default(),
        ..GrpoConfig::default()
    };
    let config = args.mode.apply(base);
    println!(
        "config mode={} advantage_mode={:?} loss_aggregation={:?} clip=({},{:?}) \
         kl_estimator={:?} dynamic_sampling={} is_level={:?} reference_policy={:?} \
         lr={} rank={} alpha={} seed={}",
        args.mode.as_str(),
        config.advantage_mode,
        config.loss_aggregation,
        config.clip_epsilon,
        config.clip_eps_high,
        config.kl_estimator,
        config.dynamic_sampling,
        config.is_level,
        config.reference_policy,
        config.learning_rate,
        config.lora_rank,
        config.lora_alpha,
        args.seed,
    );

    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(baseline_mib));
    let stop_c = stop.clone();
    let peak_c = peak.clone();
    let poller = thread::spawn(move || {
        while !stop_c.load(Ordering::Relaxed) {
            let vram = current_vram_mib();
            let mut current = peak_c.load(Ordering::Relaxed);
            while vram > current {
                match peak_c.compare_exchange(current, vram, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(next) => current = next,
                }
            }
            thread::sleep(Duration::from_millis(1_000));
        }
    });

    let progress = Some(Box::new(|progress: kiln_train::trainer::TrainingProgress| {
        println!(
            "progress step={}/{} loss={:.6} vram_mib={}",
            progress.step,
            progress.total_steps,
            progress.loss,
            current_vram_mib()
        );
    }) as kiln_train::trainer::ProgressCallback);

    let result = grpo_train_jsonl(
        &dataset_path,
        &config,
        &model_config,
        &gpu_weights,
        &tokenizer,
        &args.output_dir,
        &args.adapter_name,
        progress,
        None,
    );

    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();
    let peak_mib = peak.load(Ordering::Relaxed);
    let output_path = result?;
    println!("adapter={}", output_path.display());
    println!("peak_vram_mib={peak_mib}");
    println!("elapsed_secs={:.3}", start.elapsed().as_secs_f64());
    println!("mode_done={}", args.mode.as_str());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("cuda_grpo_ablation requires --features cuda");
}
