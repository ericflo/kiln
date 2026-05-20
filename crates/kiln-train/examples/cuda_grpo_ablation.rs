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
    /// ECHO lambda override. None = leave the LossConfig default (0.05).
    echo_lambda: Option<f64>,
    /// Disable ECHO entirely. Sets `loss.echo = None` so the auxiliary
    /// env-CE term is not added. Useful for ablation runs.
    no_echo: bool,
    /// Placeholder for OPD lambda override — accepted at the CLI so caps
    /// can pass `--opd-lambda` without the parser bailing, but not yet
    /// wired into the loss (OPD branch rebases on top of ECHO).
    opd_lambda: Option<f64>,
    /// Disable the GRPO policy-gradient term entirely. Only ECHO's
    /// env-CE drives gradients. Paper §5.5 verifier-free adaptation mode.
    /// Requires --echo-lambda not be zero (otherwise the loss is
    /// identically zero and no gradient flows).
    no_policy_loss: bool,
    /// Base adapter to start training from. Accepts either a full path
    /// (e.g. `/workspace/echo-iter3-out/on/echo-iter3-on`) or an
    /// adapter name resolved against the output dir. The trainer
    /// calls `TrainableLoraParams::load_from_safetensors` to copy the
    /// base adapter's LoRA weights into the freshly seeded Vars, so
    /// continued training starts from those values rather than from
    /// scratch. Used by Phase 3 verifier-free chaining: take a strong
    /// Phase 2 adapter, run `--no-policy-loss` from those weights.
    base_adapter: Option<String>,
    /// Reserved opt-in for future explicit base-adapter shape conversion.
    allow_adapter_shape_conversion: bool,
    /// Allow alpha/rank above the default safety limit for deliberate tests.
    allow_high_lora_scale: bool,
    /// Optional serve-ready adapter registry. When set, the completed adapter
    /// directory is installed here under --install-adapter-name or --adapter.
    install_adapter_dir: Option<PathBuf>,
    /// Optional install name used with --install-adapter-dir.
    install_adapter_name: Option<String>,
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
        let mut echo_lambda: Option<f64> = None;
        let mut no_echo = false;
        let mut opd_lambda: Option<f64> = None;
        let mut no_policy_loss = false;
        let mut base_adapter: Option<String> = None;
        let mut allow_adapter_shape_conversion = false;
        let mut allow_high_lora_scale = false;
        let mut install_adapter_dir: Option<PathBuf> = None;
        let mut install_adapter_name: Option<String> = None;

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
                "--echo-lambda" => {
                    echo_lambda = Some(
                        args.next()
                            .context("--echo-lambda needs a value")?
                            .parse()
                            .context("--echo-lambda must be a float (typical: 0.05)")?,
                    )
                }
                "--no-echo" => no_echo = true,
                "--opd-lambda" => {
                    opd_lambda = Some(
                        args.next()
                            .context("--opd-lambda needs a value")?
                            .parse()
                            .context("--opd-lambda must be a float")?,
                    )
                }
                "--no-policy-loss" => no_policy_loss = true,
                "--base-adapter" => {
                    base_adapter = Some(args.next().context("--base-adapter needs a path")?)
                }
                "--allow-adapter-shape-conversion" => allow_adapter_shape_conversion = true,
                "--allow-high-lora-scale" => allow_high_lora_scale = true,
                "--install-adapter-dir" => {
                    install_adapter_dir = Some(PathBuf::from(
                        args.next().context("--install-adapter-dir needs a value")?,
                    ))
                }
                "--install-adapter-name" => {
                    install_adapter_name =
                        Some(args.next().context("--install-adapter-name needs a value")?)
                }
                "--help" | "-h" => {
                    println!(
                        "cuda_grpo_ablation --data <jsonl> --model <dir> [--output <dir>] \
                         [--adapter <name>] --mode <baseline|phase1|phase1_gspo|phase1_cispo|\
                         phase1_reinforce|...> [--max-groups N] [--rank N] [--alpha F] \
                         [--lr F] [--seed N] [--echo-lambda F | --no-echo] [--opd-lambda F] \
                         [--base-adapter DIR] [--allow-adapter-shape-conversion] \
                         [--allow-high-lora-scale] \
                         [--install-adapter-dir DIR] [--install-adapter-name NAME]"
                    );
                    println!();
                    println!("ECHO flags (Phase 1, paper §3.3):");
                    println!(
                        "  --echo-lambda <f64>    Override the env-CE coefficient. Default \
                         from LossConfig::default() = 0.05."
                    );
                    println!(
                        "  --no-echo              Disable ECHO entirely (sets loss.echo = None)."
                    );
                    println!(
                        "  --opd-lambda <f64>     Reserved for OPD branch rebase; accepted \
                         here so cap scripts don't fail to parse. Currently ignored — \
                         OPD wiring lands in the OPD merge."
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unexpected argument: {other}"),
            }
        }

        if no_echo && echo_lambda.is_some() {
            anyhow::bail!("--no-echo and --echo-lambda are mutually exclusive");
        }
        if no_policy_loss && no_echo {
            anyhow::bail!(
                "--no-policy-loss requires ECHO to drive gradients; can't combine with --no-echo"
            );
        }
        if install_adapter_name.is_some() && install_adapter_dir.is_none() {
            anyhow::bail!("--install-adapter-name requires --install-adapter-dir");
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
            echo_lambda,
            no_echo,
            opd_lambda,
            no_policy_loss,
            base_adapter,
            allow_adapter_shape_conversion,
            allow_high_lora_scale,
            install_adapter_dir,
            install_adapter_name,
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
    // Init tracing so kiln-train's tracing::debug!/info!/warn! lines land
    // in stderr when the user sets RUST_LOG. Without this every tracing
    // call inside the trainer (including the ECHO env-CE active debug log
    // on both the uncheckpointed and checkpointed paths) gets silently
    // dropped. Default: warn. Override with e.g.
    // `RUST_LOG=info,kiln_train=debug` to see per-completion ECHO firing.
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();

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
        allow_adapter_shape_conversion: args.allow_adapter_shape_conversion,
        allow_high_lora_scale: args.allow_high_lora_scale,
        ..GrpoConfig::default()
    };
    let mut config = args.mode.apply(base);

    // ECHO knob overrides — applied AFTER the mode patch so cap scripts can
    // pin a specific lambda without depending on a mode that sets it.
    if args.no_echo {
        config.loss.echo = None;
    } else if let Some(lambda) = args.echo_lambda {
        config.loss.echo = Some(kiln_train::EchoConfig {
            lambda,
            ..kiln_train::EchoConfig::default()
        });
    }
    // KILN_ECHO_* env-var overrides take precedence over CLI flags so
    // operators can override caps from the shell without editing scripts.
    // (CLI is for inline knob-tweaking during development; env vars are
    // for ops/CI orchestration.)
    config.loss.apply_kiln_echo_env_overrides();

    if args.no_policy_loss {
        config.loss.no_policy_loss = true;
    }
    if let Some(ref base_adapter) = args.base_adapter {
        config.base_adapter = Some(base_adapter.clone());
    }
    if let Some(_opd_lambda) = args.opd_lambda {
        // Reserved for OPD branch rebase. Accept the flag but don't fire —
        // the loss path doesn't read config.loss.opd yet (see lib.rs
        // LossConfig design notes).
        eprintln!(
            "warning: --opd-lambda accepted but OPD loss not yet wired; \
             ignoring (lambda={_opd_lambda})"
        );
    }

    let echo_lambda_str = config
        .loss
        .echo
        .as_ref()
        .map(|c| format!("{}", c.lambda))
        .unwrap_or_else(|| "off".to_string());
    println!(
        "config mode={} advantage_mode={:?} loss_aggregation={:?} clip=({},{:?}) \
         kl_estimator={:?} dynamic_sampling={} is_level={:?} reference_policy={:?} \
         lr={} rank={} alpha={} seed={} echo_lambda={}",
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
        echo_lambda_str,
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
    let adapter_dir = output_path
        .canonicalize()
        .unwrap_or_else(|_| output_path.clone());
    let mut installed_adapter_dir: Option<PathBuf> = None;
    if let Some(install_dir) = args.install_adapter_dir.as_deref() {
        let install_name = args
            .install_adapter_name
            .as_deref()
            .unwrap_or(&args.adapter_name);
        let installed =
            kiln_train::install_adapter_symlink(&adapter_dir, install_dir, install_name)
                .with_context(|| {
                    format!(
                        "install adapter {} into {} as {}",
                        adapter_dir.display(),
                        install_dir.display(),
                        install_name
                    )
                })?;
        println!("INSTALLED_ADAPTER_DIR={}", installed.display());
        installed_adapter_dir = Some(installed);
    }
    kiln_train::write_adapter_output_receipt(
        &adapter_dir,
        &args.adapter_name,
        installed_adapter_dir.as_deref(),
    )?;
    println!("ADAPTER_DIR={}", adapter_dir.display());
    println!("adapter={}", adapter_dir.display());
    println!("peak_vram_mib={peak_mib}");
    println!("elapsed_secs={:.3}", start.elapsed().as_secs_f64());
    println!("mode_done={}", args.mode.as_str());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("cuda_grpo_ablation requires --features cuda");
}
