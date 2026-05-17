//! Run true on-policy OPD against a `RemoteTeacher` (vLLM/sglang).
//!
//! Each step:
//!   1. Sample a student rollout under the current LoRA.
//!   2. Query the teacher's top-K logprobs at the sampled positions
//!      via vLLM `/v1/completions` (`prompt_logprobs=K`).
//!   3. Compute reverse-KL, backprop into LoRA params, optimizer step.
//!
//! Usage:
//! ```bash
//! cuda_opd_remote \
//!   --data train.jsonl \
//!   --model-path /workspace/Qwen3.5-4B \
//!   --teacher-url http://localhost:8002 \
//!   --teacher-model qwen3.6-27b-fp8 \
//!   --output-dir /tmp/opd-out \
//!   --adapter-name opd-onpolicy-v1 \
//!   --epochs 1 --rank 8 --alpha 16 --lr 1e-4 \
//!   --top-k 8 --temperature 1.0 --top-p 0.9 --max-tokens 256
//! ```

use anyhow::Result;

#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use kiln_core::config::ModelConfig;
#[cfg(feature = "cuda")]
use kiln_core::tokenizer::KilnTokenizer;
#[cfg(feature = "cuda")]
use kiln_model::forward::GpuWeights;
#[cfg(feature = "cuda")]
use kiln_train::logit_source::LogitSource;
#[cfg(feature = "cuda")]
use kiln_train::opd::{OpdConfig, OpdLossGranularity, OpdPrompt, opd_train};
#[cfg(feature = "cuda")]
use kiln_train::{Optimizer, RemoteProvider, RemoteTeacher, RemoteTeacherConfig};
#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    data: PathBuf,
    model_path: PathBuf,
    output_dir: PathBuf,
    adapter_name: String,
    teacher_url: String,
    teacher_model: String,
    epochs: usize,
    max_examples: Option<usize>,
    lora_rank: usize,
    lora_alpha: f32,
    learning_rate: f64,
    top_k: usize,
    temperature: f64,
    top_p: f64,
    max_tokens: usize,
    samples_per_prompt: usize,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut data = None;
        let mut model_path = None;
        let mut output_dir = std::env::temp_dir().join("kiln-cuda-opd-remote");
        let mut adapter_name = "opd-onpolicy".to_string();
        let mut teacher_url = "http://localhost:8002".to_string();
        let mut teacher_model = "qwen3.6-27b-fp8".to_string();
        let mut epochs = 1usize;
        let mut max_examples = None;
        let mut lora_rank = 8usize;
        let mut lora_alpha = 16.0f32;
        let mut learning_rate = 1e-4f64;
        let mut top_k = 8usize;
        let mut temperature = 1.0f64;
        let mut top_p = 0.9f64;
        let mut max_tokens = 256usize;
        let mut samples_per_prompt = 1usize;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--data" => data = args.next().map(PathBuf::from),
                "--model-path" => model_path = args.next().map(PathBuf::from),
                "--output-dir" => {
                    output_dir = args.next().map(PathBuf::from).context("--output-dir value")?
                }
                "--adapter-name" => {
                    adapter_name = args.next().context("--adapter-name value")?
                }
                "--teacher-url" => teacher_url = args.next().context("--teacher-url value")?,
                "--teacher-model" => {
                    teacher_model = args.next().context("--teacher-model value")?
                }
                "--epochs" => epochs = args.next().context("--epochs value")?.parse()?,
                "--max-examples" => {
                    max_examples = Some(args.next().context("--max-examples value")?.parse()?)
                }
                "--rank" => lora_rank = args.next().context("--rank value")?.parse()?,
                "--alpha" => lora_alpha = args.next().context("--alpha value")?.parse()?,
                "--lr" => learning_rate = args.next().context("--lr value")?.parse()?,
                "--top-k" => top_k = args.next().context("--top-k value")?.parse()?,
                "--temperature" => {
                    temperature = args.next().context("--temperature value")?.parse()?
                }
                "--top-p" => top_p = args.next().context("--top-p value")?.parse()?,
                "--max-tokens" => max_tokens = args.next().context("--max-tokens value")?.parse()?,
                "--samples-per-prompt" => {
                    samples_per_prompt = args
                        .next()
                        .context("--samples-per-prompt value")?
                        .parse()?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cuda_opd_remote --data <jsonl> --model-path <dir> \
                         [--teacher-url URL] [--teacher-model NAME] \
                         [--output-dir <dir>] [--adapter-name <name>] \
                         [--epochs N] [--max-examples N] \
                         [--rank N] [--alpha F] [--lr F] \
                         [--top-k K] [--temperature F] [--top-p F] [--max-tokens N] \
                         [--samples-per-prompt N]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown arg {other}"),
            }
        }
        Ok(Self {
            data: data.context("--data required")?,
            model_path: model_path.context("--model-path required")?,
            output_dir,
            adapter_name,
            teacher_url,
            teacher_model,
            epochs,
            max_examples,
            lora_rank,
            lora_alpha,
            learning_rate,
            top_k,
            temperature,
            top_p,
            max_tokens,
            samples_per_prompt,
        })
    }
}

#[cfg(feature = "cuda")]
fn load_prompts(path: &PathBuf, max: Option<usize>) -> Result<Vec<OpdPrompt>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(limit) = max {
            if out.len() >= limit {
                break;
            }
        }
        let p: OpdPrompt = serde_json::from_str(line)
            .with_context(|| format!("parsing line {} of {}", i + 1, path.display()))?;
        out.push(p);
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn load_tokenizer(model_path: &PathBuf) -> Result<KilnTokenizer> {
    let tok_path = model_path.join("tokenizer.json");
    let mut tokenizer = KilnTokenizer::from_file(
        tok_path.to_str().context("tokenizer path utf-8")?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let tpl = model_path.join("chat_template.jinja");
    if tpl.exists() {
        let template = std::fs::read_to_string(&tpl)?;
        tokenizer = tokenizer.with_chat_template(template);
    }
    Ok(tokenizer)
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
fn main() -> Result<()> {
    let args = Args::parse()?;
    let start = Instant::now();
    println!("baseline_vram_mib={}", current_vram_mib());

    let prompts = load_prompts(&args.data, args.max_examples)?;
    anyhow::ensure!(!prompts.is_empty(), "no prompts loaded");
    println!("prompts={}", prompts.len());

    let tokenizer = load_tokenizer(&args.model_path)?;
    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA not available"
    );
    let device = candle_core::Device::new_cuda(0)?;
    let model_config = ModelConfig::qwen3_5_4b();

    println!("loading_student={}", args.model_path.display());
    let weights = kiln_model::load_model_with_options(
        &args.model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load student weights")?;
    let gpu_weights = GpuWeights::from_model_weights(&weights, &model_config, &device)?;
    drop(weights);
    println!("student_loaded_vram_mib={}", current_vram_mib());

    let teacher_cfg = RemoteTeacherConfig {
        provider: RemoteProvider::Vllm,
        model: args.teacher_model.clone(),
        url: args.teacher_url.clone(),
        api_key_env: None,
        teacher_id: format!("vllm/{}", args.teacher_model),
        tokenizer_hash: None,
        max_top_k: args.top_k,
        vocab_size: 0,
        max_cost_usd: None,
        timeout_ms: 120_000,
    };
    let teacher: Arc<dyn LogitSource> = Arc::new(RemoteTeacher::new(teacher_cfg));
    println!("teacher_caps={:?}", teacher.capabilities());

    std::fs::create_dir_all(&args.output_dir)?;

    let mut cfg = OpdConfig::default();
    cfg.epochs = args.epochs;
    cfg.learning_rate = args.learning_rate;
    cfg.lora_rank = args.lora_rank;
    cfg.lora_alpha = args.lora_alpha;
    cfg.top_k = args.top_k;
    cfg.temperature = args.temperature;
    cfg.top_p = args.top_p;
    cfg.max_tokens = args.max_tokens;
    cfg.samples_per_prompt = args.samples_per_prompt;
    cfg.loss = OpdLossGranularity::TeacherTopK;
    cfg.optimizer = Optimizer::default();
    cfg.seed = Some(0xC0DA_5EED);

    let progress = Some(Box::new(|p: kiln_train::trainer::TrainingProgress| {
        println!(
            "progress step={}/{} epoch={}/{} loss={:.6} vram_mib={}",
            p.step,
            p.total_steps,
            p.epoch,
            p.total_epochs,
            p.loss,
            current_vram_mib()
        );
    }) as kiln_train::trainer::ProgressCallback);

    let out = opd_train(
        &prompts,
        &cfg,
        &model_config,
        &gpu_weights,
        &tokenizer,
        teacher,
        &args.output_dir,
        &args.adapter_name,
        progress,
    )?;

    println!("adapter={}", out.display());
    println!("elapsed_secs={:.3}", start.elapsed().as_secs_f64());
    println!("peak_vram_mib={}", current_vram_mib());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("cuda_opd_remote requires --features cuda");
}
