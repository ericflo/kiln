//! `cuda_opd_from_fixture` — run kiln OPD training against a pre-computed
//! teacher fixture JSONL. The fixture stores the teacher's top-K
//! logprobs at each active position so a different-architecture teacher
//! (e.g. Qwen3.6-27B) can drive an OPD run on the kiln-served student
//! (Qwen3.5-4B) without holding both models in VRAM at the same time.
//!
//! Fixture line format:
//!
//! ```jsonl
//! {
//!   "tokens": [248044, 102, ...],   // u32 list, full sequence after chat template
//!   "active_positions": [12, 13, ...], // u32 list of assistant token positions
//!   "topk_indices": [[..K..], ..],  // T_active x K u32 (one row per active pos)
//!   "topk_logprobs": [[..K..], ..]  // T_active x K f32
//! }
//! ```
//!
//! Usage:
//!
//! ```bash
//! cuda_opd_from_fixture \
//!   --model-path /workspace/kiln/Qwen3.5-4B \
//!   --prompts datasets/train.opd.jsonl \
//!   --teacher-fixture datasets/teacher.fixture.jsonl \
//!   --output-dir /workspace/kiln/Qwen3.5-4B/adapters \
//!   --adapter-name opd-json-v1 \
//!   --top-k 32 --rank 32 --lr 1e-5
//! ```

use anyhow::Result;

#[cfg(feature = "cuda")]
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use kiln_core::config::ModelConfig;
#[cfg(feature = "cuda")]
use kiln_core::tokenizer::KilnTokenizer;
#[cfg(feature = "cuda")]
use kiln_model::forward::GpuWeights;
#[cfg(feature = "cuda")]
use kiln_train::logit_source::FixtureLogitSource;
#[cfg(feature = "cuda")]
use kiln_train::opd::{OpdConfig, OpdLossGranularity, OpdPrompt};
#[cfg(feature = "cuda")]
use kiln_train::{ChatMessage, LogitSource, Optimizer};

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    prompts: PathBuf,
    teacher_fixture: PathBuf,
    model_path: PathBuf,
    output_dir: PathBuf,
    adapter_name: String,
    top_k: usize,
    rank: usize,
    lr: f64,
    samples_per_prompt: usize,
    epochs: usize,
    seed: u64,
    max_prompts: Option<usize>,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut prompts = None;
        let mut teacher_fixture = None;
        let mut model_path = None;
        let mut output_dir = std::env::temp_dir().join("kiln-cuda-opd-fixture");
        let mut adapter_name = "opd-fixture".to_string();
        let mut top_k = 32usize;
        let mut rank = 32usize;
        let mut lr = 1e-5f64;
        let mut samples_per_prompt = 1usize;
        let mut epochs = 1usize;
        let mut seed = 0xC0DA_5EEDu64;
        let mut max_prompts: Option<usize> = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--prompts" => prompts = args.next().map(PathBuf::from),
                "--teacher-fixture" => teacher_fixture = args.next().map(PathBuf::from),
                "--model-path" => model_path = args.next().map(PathBuf::from),
                "--output-dir" => output_dir = args.next().map(PathBuf::from).unwrap_or(output_dir),
                "--adapter-name" => adapter_name = args.next().unwrap_or(adapter_name),
                "--top-k" => top_k = args.next().context("--top-k value")?.parse()?,
                "--rank" => rank = args.next().context("--rank value")?.parse()?,
                "--lr" => lr = args.next().context("--lr value")?.parse()?,
                "--samples-per-prompt" => {
                    samples_per_prompt =
                        args.next().context("--samples-per-prompt value")?.parse()?
                }
                "--epochs" => {
                    epochs = args.next().context("--epochs value")?.parse()?
                }
                "--seed" => seed = args.next().context("--seed value")?.parse()?,
                "--max-prompts" => {
                    max_prompts = Some(args.next().context("--max-prompts value")?.parse()?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cuda_opd_from_fixture --prompts <jsonl> --teacher-fixture <jsonl> \
                         --model-path <dir> [--output-dir <dir>] [--adapter-name <name>] \
                         [--top-k 32] [--rank 32] [--lr 1e-5] [--samples-per-prompt 1] \
                         [--seed N] [--max-prompts N]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }
        Ok(Self {
            prompts: prompts.context("--prompts required")?,
            teacher_fixture: teacher_fixture.context("--teacher-fixture required")?,
            model_path: model_path.context("--model-path required")?,
            output_dir,
            adapter_name,
            top_k,
            rank,
            lr,
            samples_per_prompt,
            epochs,
            seed,
            max_prompts,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct PromptRow {
    messages: Vec<MessageRow>,
}

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct MessageRow {
    role: String,
    content: String,
}

#[cfg(feature = "cuda")]
#[derive(serde::Deserialize)]
struct FixtureRow {
    tokens: Vec<u32>,
    active_positions: Vec<usize>,
    topk_indices: Vec<Vec<u32>>,
    topk_logprobs: Vec<Vec<f32>>,
}

#[cfg(feature = "cuda")]
fn load_prompts(path: &PathBuf, max: Option<usize>) -> Result<Vec<OpdPrompt>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut prompts = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(m) = max {
            if prompts.len() >= m {
                break;
            }
        }
        let row: PromptRow = serde_json::from_str(line)
            .with_context(|| format!("parse prompt line {}", i + 1))?;
        let messages = row
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        prompts.push(OpdPrompt { messages });
    }
    Ok(prompts)
}

#[cfg(feature = "cuda")]
fn load_teacher_fixture(
    path: &PathBuf,
    top_k: usize,
    vocab_size: usize,
) -> Result<FixtureLogitSource> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut fixture = FixtureLogitSource::uniform_topk("qwen3.6-27b@offline", vocab_size, top_k);
    let mut total_positions = 0usize;
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: FixtureRow = serde_json::from_str(line)
            .with_context(|| format!("parse fixture line {}", i + 1))?;
        anyhow::ensure!(
            row.active_positions.len() == row.topk_indices.len()
                && row.topk_indices.len() == row.topk_logprobs.len(),
            "fixture line {} active/indices/logprobs length mismatch",
            i + 1
        );
        let h = FixtureLogitSource::hash_tokens(&row.tokens);
        for (idx, &pos) in row.active_positions.iter().enumerate() {
            let ind = &row.topk_indices[idx];
            let lp = &row.topk_logprobs[idx];
            anyhow::ensure!(
                ind.len() == top_k && lp.len() == top_k,
                "fixture line {} pos {} expected top_k={} but got indices={} logprobs={}",
                i + 1, pos, top_k, ind.len(), lp.len()
            );
            fixture.insert(h, pos, ind.clone(), lp.clone());
            total_positions += 1;
        }
    }
    eprintln!(
        "fixture loaded: {} entries across {} positions (vocab_size={})",
        path.display(),
        total_positions,
        vocab_size,
    );
    Ok(fixture)
}

#[cfg(feature = "cuda")]
fn load_tokenizer(model_path: &PathBuf) -> Result<KilnTokenizer> {
    let tokenizer_path = model_path.join("tokenizer.json");
    let mut tokenizer = KilnTokenizer::from_file(
        tokenizer_path
            .to_str()
            .context("tokenizer path not UTF-8")?,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let tmpl = model_path.join("chat_template.jinja");
    if tmpl.exists() {
        tokenizer = tokenizer.with_chat_template(std::fs::read_to_string(&tmpl)?);
    }
    Ok(tokenizer)
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse()?;
    let start = std::time::Instant::now();
    eprintln!("cuda_opd_from_fixture starting");

    let prompts = load_prompts(&args.prompts, args.max_prompts)?;
    anyhow::ensure!(!prompts.is_empty(), "no prompts loaded");
    eprintln!("prompts_loaded={}", prompts.len());

    let tokenizer = load_tokenizer(&args.model_path)?;
    anyhow::ensure!(
        candle_core::utils::cuda_is_available(),
        "CUDA is not available in this build/runtime"
    );
    let device = candle_core::Device::new_cuda(0).context("create CUDA device 0")?;
    let model_config = ModelConfig::qwen3_5_4b();

    let fixture = load_teacher_fixture(
        &args.teacher_fixture,
        args.top_k,
        model_config.vocab_size,
    )?;

    eprintln!("loading_model={}", args.model_path.display());
    let model_weights = kiln_model::load_model_with_options(
        &args.model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("load model weights")?;
    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("transfer weights to CUDA")?;
    drop(model_weights);

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;

    let mut config = OpdConfig::default();
    config.loss = OpdLossGranularity::TeacherTopK;
    config.top_k = args.top_k;
    config.samples_per_prompt = args.samples_per_prompt;
    config.learning_rate = args.lr;
    config.lora_rank = args.rank;
    config.output_name = Some(args.adapter_name.clone());
    config.auto_load = false;
    config.seed = Some(args.seed);
    config.optimizer = Optimizer::default();
    config.epochs = args.epochs;

    let teacher: std::sync::Arc<dyn LogitSource> = std::sync::Arc::new(fixture);

    let progress = Some(Box::new(|p: kiln_train::trainer::TrainingProgress| {
        eprintln!(
            "progress step={}/{} epoch={}/{} loss={:.6}",
            p.step, p.total_steps, p.epoch, p.total_epochs, p.loss
        );
    }) as kiln_train::trainer::ProgressCallback);

    let output_path = kiln_train::opd::opd_train(
        &prompts,
        &config,
        &model_config,
        &gpu_weights,
        &tokenizer,
        teacher,
        &args.output_dir,
        &args.adapter_name,
        progress,
    )
    .context("opd_train")?;

    eprintln!(
        "OK adapter={} elapsed_secs={:.3}",
        output_path.display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("cuda_opd_from_fixture requires --features cuda");
}
