//! Synthetic long-context GRPO benchmark.
//!
//! Default mode is CPU/dry: it builds compaction-shaped synthetic rollouts and
//! reports tokenization plus mask-build timing as JSON. Pass `--cuda` with a
//! Qwen3.5-4B model directory to run one GRPO optimizer step per length.

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use kiln_core::config::ModelConfig;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::trainer::{GrpoBenchmarkReport, grpo_benchmark_tokenization};
use kiln_train::{
    ChatMessage, GrpoConfig, GrpoGroup, Optimizer, ScoredRollout, TurnKind, TurnSegment,
};
use serde::Serialize;

#[cfg(feature = "cuda")]
use kiln_train::trainer::{TrainableLoraParams, grpo_benchmark_training_step};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchMode {
    Dry,
    Cuda,
}

impl BenchMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Dry => "dry",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug)]
struct Args {
    model_path: Option<PathBuf>,
    output: Option<PathBuf>,
    lengths: Vec<usize>,
    completions: usize,
    mode: BenchMode,
    checkpoint_segments: usize,
    lora_rank: usize,
    lora_alpha: f32,
    learning_rate: f64,
    seed: u64,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model_path = None;
        let mut output = None;
        let mut lengths = vec![8_192, 16_384, 32_768, 65_536];
        let mut completions = 2usize;
        let mut mode = BenchMode::Dry;
        let mut checkpoint_segments = 4usize;
        let mut lora_rank = 8usize;
        let mut lora_alpha = 16.0f32;
        let mut learning_rate = 1e-5f64;
        let mut seed = 0x6c6f_6e67_6772_706f_u64;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model_path = Some(PathBuf::from(args.next().context("--model needs a value")?))
                }
                "--output" => {
                    output = Some(PathBuf::from(args.next().context("--output needs a value")?))
                }
                "--lengths" => {
                    lengths = parse_lengths(&args.next().context("--lengths needs a value")?)?
                }
                "--completions" => {
                    completions = args
                        .next()
                        .context("--completions needs a value")?
                        .parse()
                        .context("--completions must be a positive integer")?
                }
                "--dry-run" => mode = BenchMode::Dry,
                "--cuda" => mode = BenchMode::Cuda,
                "--segments" => {
                    checkpoint_segments = args
                        .next()
                        .context("--segments needs a value")?
                        .parse()
                        .context("--segments must be an integer; use 0 to disable checkpointing")?
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
                    print_help();
                    std::process::exit(0);
                }
                other => anyhow::bail!("unexpected argument: {other}"),
            }
        }

        anyhow::ensure!(!lengths.is_empty(), "--lengths must not be empty");
        anyhow::ensure!(
            completions >= 2,
            "--completions must be at least 2 so GRPO advantages are non-degenerate"
        );
        anyhow::ensure!(lora_rank > 0, "--rank must be at least 1");

        if mode == BenchMode::Cuda {
            anyhow::ensure!(model_path.is_some(), "--model is required with --cuda");
        }

        Ok(Self {
            model_path,
            output,
            lengths,
            completions,
            mode,
            checkpoint_segments,
            lora_rank,
            lora_alpha,
            learning_rate,
            seed,
        })
    }

    fn model_label(&self) -> String {
        self.model_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "synthetic-byte-tokenizer".to_string())
    }
}

#[derive(Debug, Serialize)]
struct BenchRecord {
    event: &'static str,
    mode: &'static str,
    requested_seq_len: usize,
    observed_seq_len: usize,
    completions: usize,
    checkpoint_segments: Option<usize>,
    model: String,
    peak_vram_mib: Option<u64>,
    kernel_launch_count: Option<u64>,
    report: GrpoBenchmarkReport,
}

struct VramPoller {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: thread::JoinHandle<()>,
}

impl VramPoller {
    fn start() -> Option<Self> {
        let baseline = current_vram_mib()?;
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(baseline));
        let stop_c = Arc::clone(&stop);
        let peak_c = Arc::clone(&peak);
        let handle = thread::spawn(move || {
            while !stop_c.load(Ordering::Relaxed) {
                if let Some(vram) = current_vram_mib() {
                    let mut current = peak_c.load(Ordering::Relaxed);
                    while vram > current {
                        match peak_c.compare_exchange(
                            current,
                            vram,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(next) => current = next,
                        }
                    }
                }
                thread::sleep(Duration::from_millis(250));
            }
        });
        Some(Self { stop, peak, handle })
    }

    fn finish(self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
        self.peak.load(Ordering::Relaxed)
    }
}

fn parse_lengths(value: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid length {part:?}"))
        })
        .collect()
}

fn print_help() {
    println!(
        "long_context_grpo_bench [--model <qwen3.5-4b-dir>] [--dry-run|--cuda] \
         [--lengths 8192,16384,32768,65536] [--output results.json] \
         [--completions N] [--segments N] [--rank N] [--alpha F] [--lr F] [--seed N]"
    );
    println!();
    println!(
        "Default mode is --dry-run, which requires no CUDA and uses a built-in byte tokenizer when --model is omitted."
    );
    println!("Use --segments 0 to disable checkpointing on CUDA runs.");
}

fn load_tokenizer(model_path: &Path) -> Result<KilnTokenizer> {
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

fn synthetic_tokenizer() -> Result<KilnTokenizer> {
    let mut vocab = String::from("{");
    for b in 0u32..256 {
        let ch = char::from_u32(b).context("invalid byte vocab char")?;
        let key = match ch {
            '"' => "\\\"".to_string(),
            '\\' => "\\\\".to_string(),
            '\n' => "\\n".to_string(),
            '\r' => "\\r".to_string(),
            '\t' => "\\t".to_string(),
            c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
            c => c.to_string(),
        };
        if b > 0 {
            vocab.push(',');
        }
        vocab.push_str(&format!("\"{}\":{}", key, b));
    }
    vocab.push('}');
    let json = format!(
        r#"{{"version": "1.0", "model": {{"type": "BPE", "vocab": {}, "merges": []}}}}"#,
        vocab
    );
    let template = "{% for message in messages -%}\
{% if message.role == 'tool' %}\
{% if loop.previtem is undefined or loop.previtem.role != 'tool' %}<|im_start|>user
{% endif %}<tool_response>
{{ message.content }}
</tool_response>\
{% if loop.last or loop.nextitem.role != 'tool' %}<|im_end|>
{% endif %}\
{% else %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endif %}\
{% endfor %}";
    Ok(KilnTokenizer::from_bytes(json.as_bytes())
        .map_err(|err| anyhow::anyhow!("{err}"))?
        .with_chat_template(template.to_string()))
}

fn current_vram_mib() -> Option<u64> {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|stdout| stdout.trim().lines().next()?.trim().parse::<u64>().ok())
}

fn prompt_messages() -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a concise agent compressing long terminal traces.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Inspect the synthetic trace and produce the final compact answer."
                .to_string(),
        },
    ]
}

fn synthetic_rollout(repeats: usize, completion_idx: usize, reward: f64) -> ScoredRollout {
    let observation_unit = format!(
        "trace_line={completion_idx} status=ok metric=pi_compaction payload=abcdefghijklmnopqrstuvwxyz0123456789\n"
    );
    let observation = observation_unit.repeat(repeats);
    ScoredRollout::from_trajectory(
        vec![
            TurnSegment {
                role: "assistant".to_string(),
                content: format!(
                    "<tool_call>\n{{\"cmd\":\"inspect_trace\",\"completion\":{completion_idx}}}\n</tool_call>"
                ),
                kind: TurnKind::Action,
                tool_call_id: Some(format!("inspect-{completion_idx}")),
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "tool".to_string(),
                content: observation,
                kind: TurnKind::Observation,
                tool_call_id: Some(format!("inspect-{completion_idx}")),
                warning_prefix_len: None,
            },
            TurnSegment {
                role: "assistant".to_string(),
                content: format!(
                    "Final compact answer for completion {completion_idx}: retain causal facts and discard repeated trace noise."
                ),
                kind: TurnKind::Action,
                tool_call_id: None,
                warning_prefix_len: None,
            },
        ],
        reward,
    )
}

fn rollout_seq_len(
    tokenizer: &KilnTokenizer,
    messages: &[ChatMessage],
    repeats: usize,
) -> Result<usize> {
    let rollout = synthetic_rollout(repeats, 0, 1.0);
    let masked = kiln_train::trajectory_mask::build_masks_from_trajectory(
        &rollout.trajectory,
        messages,
        tokenizer,
        &kiln_train::trajectory_mask::MaskConfig::default(),
    )?;
    Ok(masked.input_ids.len())
}

fn synthetic_group_for_length(
    tokenizer: &KilnTokenizer,
    target_len: usize,
    completions: usize,
) -> Result<GrpoGroup> {
    let messages = prompt_messages();
    let mut high = 1usize;
    while rollout_seq_len(tokenizer, &messages, high)? < target_len {
        high = high.saturating_mul(2);
        anyhow::ensure!(
            high <= target_len.saturating_mul(32).max(1_024),
            "could not synthesize a rollout near {target_len} tokens"
        );
    }

    let mut low = 0usize;
    while low < high {
        let mid = low + (high - low) / 2;
        if rollout_seq_len(tokenizer, &messages, mid)? < target_len {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    let completions = (0..completions)
        .map(|idx| {
            let reward = if idx % 2 == 0 { 1.0 } else { 0.0 };
            synthetic_rollout(low, idx, reward)
        })
        .collect();
    Ok(GrpoGroup {
        messages,
        completions,
    })
}

fn checkpoint_segments(num_layers: usize, requested: usize) -> Option<Vec<(usize, usize)>> {
    if requested == 0 {
        return None;
    }
    let segment_count = requested.min(num_layers).max(1);
    let mut segments = Vec::with_capacity(segment_count);
    for idx in 0..segment_count {
        let start = idx * num_layers / segment_count;
        let end = (idx + 1) * num_layers / segment_count;
        if start < end {
            segments.push((start, end));
        }
    }
    Some(segments)
}

fn bench_config(args: &Args) -> GrpoConfig {
    GrpoConfig {
        learning_rate: args.learning_rate,
        lora_rank: args.lora_rank,
        lora_alpha: args.lora_alpha,
        seed: Some(args.seed),
        auto_load: false,
        optimizer: Optimizer::default(),
        ..GrpoConfig::default()
    }
}

#[cfg(feature = "cuda")]
struct CudaState {
    device: candle_core::Device,
    backend: Arc<dyn kiln_model::backend::BackendRuntime>,
    weights: kiln_model::forward::GpuWeights,
}

#[cfg(feature = "cuda")]
fn load_cuda_state(model_path: &Path, model_config: &ModelConfig) -> Result<CudaState> {
    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA is not available in this build/runtime"
    );
    let device = candle_core::Device::new_cuda(0).context("create CUDA device 0")?;
    let model_weights = kiln_model::load_model_with_options(
        model_path,
        model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load model weights")?;
    let weights = kiln_model::forward::GpuWeights::from_model_weights(
        &model_weights,
        model_config,
        &device,
    )
    .context("transfer model weights to CUDA")?;
    drop(model_weights);
    let backend = kiln_model::backend::for_device(&device);
    Ok(CudaState {
        device,
        backend,
        weights,
    })
}

#[cfg(feature = "cuda")]
fn run_cuda_record(
    args: &Args,
    state: &CudaState,
    tokenizer: &KilnTokenizer,
    model_config: &ModelConfig,
    group: &GrpoGroup,
) -> Result<(GrpoBenchmarkReport, Option<u64>)> {
    let config = bench_config(args);
    let params = TrainableLoraParams::initialize_seeded(
        model_config,
        &state.weights,
        args.lora_rank,
        args.lora_alpha,
        &state.device,
        Some(args.seed),
    )?;
    params.register_with_backend(&*state.backend)?;
    let mut opt_state = match config.optimizer {
        Optimizer::AdamW { .. } => Some(params.allocate_adamw_state(&state.device)?),
        Optimizer::Sgd => None,
    };
    if let Some(state_opt) = opt_state.as_ref() {
        state_opt.register_with_backend(&*state.backend)?;
    }
    let segments = checkpoint_segments(model_config.num_layers, args.checkpoint_segments);
    let poller = VramPoller::start();
    let result = grpo_benchmark_training_step(
        &*state.backend,
        group,
        &state.weights,
        model_config,
        &params,
        &config,
        segments.as_deref(),
        &state.device,
        tokenizer,
        opt_state.as_mut(),
    );
    let peak_vram_mib = poller.map(VramPoller::finish);
    if let Some(state_opt) = opt_state.as_ref() {
        state_opt.evict_from_backend(&*state.backend);
    }
    params.evict_from_backend(&*state.backend);
    Ok((result?, peak_vram_mib))
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_record(
    _args: &Args,
    _tokenizer: &KilnTokenizer,
    _model_config: &ModelConfig,
    _group: &GrpoGroup,
) -> Result<(GrpoBenchmarkReport, Option<u64>)> {
    anyhow::bail!("--cuda requires building this example with --features cuda")
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let tokenizer = match args.model_path.as_deref() {
        Some(model_path) => load_tokenizer(model_path)?,
        None => synthetic_tokenizer()?,
    };
    let model_config = ModelConfig::qwen3_5_4b();

    #[cfg(feature = "cuda")]
    let cuda_state = if args.mode == BenchMode::Cuda {
        Some(load_cuda_state(
            args.model_path
                .as_deref()
                .context("--model is required with --cuda")?,
            &model_config,
        )?)
    } else {
        None
    };

    let mut records = Vec::with_capacity(args.lengths.len());
    for &target_len in &args.lengths {
        let group = synthetic_group_for_length(&tokenizer, target_len, args.completions)
            .with_context(|| format!("building synthetic group for {target_len} tokens"))?;

        let (report, peak_vram_mib) = match args.mode {
            BenchMode::Dry => (grpo_benchmark_tokenization(&group, &tokenizer)?, None),
            BenchMode::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    run_cuda_record(
                        &args,
                        cuda_state.as_ref().context("CUDA state not loaded")?,
                        &tokenizer,
                        &model_config,
                        &group,
                    )?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    run_cuda_record(&args, &tokenizer, &model_config, &group)?
                }
            }
        };

        let record = BenchRecord {
            event: "long_context_grpo_bench",
            mode: args.mode.as_str(),
            requested_seq_len: target_len,
            observed_seq_len: report.max_seq_len,
            completions: args.completions,
            checkpoint_segments: (args.mode == BenchMode::Cuda && args.checkpoint_segments > 0)
                .then_some(args.checkpoint_segments),
            model: args.model_label(),
            peak_vram_mib,
            kernel_launch_count: None,
            report,
        };
        println!("{}", serde_json::to_string(&record)?);
        records.push(serde_json::to_value(record)?);
    }

    if let Some(output) = args.output.as_deref() {
        std::fs::write(output, serde_json::to_string_pretty(&records)?)
            .with_context(|| format!("writing {}", output.display()))?;
    }

    Ok(())
}
