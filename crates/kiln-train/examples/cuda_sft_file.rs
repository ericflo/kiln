use anyhow::Result;

#[cfg(feature = "cuda")]
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
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
use kiln_train::{Optimizer, SftConfig, SftExample};

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct Args {
    data: PathBuf,
    model_path: PathBuf,
    output_dir: PathBuf,
    adapter_name: String,
    epochs: usize,
    max_examples: Option<usize>,
    skip_examples: usize,
}

#[cfg(feature = "cuda")]
impl Args {
    fn parse() -> Result<Self> {
        let mut data = None;
        let mut model_path = None;
        let mut output_dir = std::env::temp_dir().join("kiln-cuda-sft-file");
        let mut adapter_name = "cuda-sft-file".to_string();
        let mut epochs = 1usize;
        let mut max_examples = None;
        let mut skip_examples = 0usize;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--data" => data = args.next().map(PathBuf::from),
                "--model-path" => model_path = args.next().map(PathBuf::from),
                "--output-dir" => {
                    output_dir = args
                        .next()
                        .map(PathBuf::from)
                        .context("--output-dir requires a value")?
                }
                "--adapter-name" => {
                    adapter_name = args.next().context("--adapter-name requires a value")?
                }
                "--epochs" => {
                    epochs = args
                        .next()
                        .context("--epochs requires a value")?
                        .parse()
                        .context("--epochs must be a positive integer")?
                }
                "--max-examples" => {
                    max_examples = Some(
                        args.next()
                            .context("--max-examples requires a value")?
                            .parse()
                            .context("--max-examples must be a positive integer")?,
                    )
                }
                "--skip-examples" => {
                    skip_examples = args
                        .next()
                        .context("--skip-examples requires a value")?
                        .parse()
                        .context("--skip-examples must be a non-negative integer")?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cuda_sft_file --data <jsonl> --model-path <dir> \
                         [--output-dir <dir>] [--adapter-name <name>] \
                         [--epochs <n>] [--skip-examples <n>] [--max-examples <n>]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }

        let data = data.context("--data <jsonl> is required")?;
        let model_path = model_path.context("--model-path <dir> is required")?;
        anyhow::ensure!(epochs > 0, "--epochs must be positive");
        Ok(Self {
            data,
            model_path,
            output_dir,
            adapter_name,
            epochs,
            max_examples,
            skip_examples,
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
fn load_examples(path: &PathBuf, skip: usize, max: Option<usize>) -> Result<Vec<SftExample>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut examples = Vec::new();
    for (line_idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if examples.len() < skip {
            examples.push(
                serde_json::from_str::<SftExample>(line)
                    .with_context(|| format!("parsing JSONL line {}", line_idx + 1))?,
            );
            continue;
        }
        if max.is_some_and(|limit| examples.len().saturating_sub(skip) >= limit) {
            break;
        }
        examples.push(
            serde_json::from_str::<SftExample>(line)
                .with_context(|| format!("parsing JSONL line {}", line_idx + 1))?,
        );
    }
    Ok(examples.into_iter().skip(skip).collect())
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
fn token_len(example: &SftExample, tokenizer: &KilnTokenizer) -> Result<usize> {
    let messages = example
        .messages
        .iter()
        .map(|message| kiln_core::tokenizer::ChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let text = tokenizer
        .apply_chat_template(&messages)
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    tokenizer
        .encode(&text)
        .map(|ids| ids.len())
        .map_err(|err| anyhow::anyhow!("{err}"))
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let args = Args::parse()?;
    let start = Instant::now();
    let baseline_mib = current_vram_mib();
    println!("baseline_vram_mib={baseline_mib}");

    let examples = load_examples(&args.data, args.skip_examples, args.max_examples)?;
    anyhow::ensure!(!examples.is_empty(), "no examples selected");
    println!(
        "selected_examples={} skip={} max={:?}",
        examples.len(),
        args.skip_examples,
        args.max_examples
    );

    let tokenizer = load_tokenizer(&args.model_path)?;
    for (idx, example) in examples.iter().enumerate() {
        println!(
            "example={} tokens={}",
            idx + 1,
            token_len(example, &tokenizer)?
        );
    }

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
    let config = SftConfig {
        epochs: args.epochs,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some(args.adapter_name.clone()),
        auto_load: false,
        checkpoint_interval: None,
        seed: Some(0xC0DA_5EED),
        optimizer: Optimizer::default(),
    };

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
            thread::sleep(Duration::from_millis(50));
        }
    });

    let progress = Some(Box::new(|progress: kiln_train::trainer::TrainingProgress| {
        println!(
            "progress step={}/{} epoch={}/{} loss={:.6} progress={:.4} vram_mib={}",
            progress.step,
            progress.total_steps,
            progress.epoch,
            progress.total_epochs,
            progress.loss,
            progress.progress,
            current_vram_mib()
        );
    }) as kiln_train::trainer::ProgressCallback);

    let result = kiln_train::cuda_train::cuda_native_sft_train(
        &examples,
        &config,
        &model_config,
        &gpu_weights,
        &tokenizer,
        &args.output_dir,
        &args.adapter_name,
        progress,
    );

    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();
    let peak_mib = peak.load(Ordering::Relaxed);
    let output_path = result?;
    println!("adapter={}", output_path.display());
    println!("peak_vram_mib={peak_mib}");
    println!("elapsed_secs={:.3}", start.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    anyhow::bail!("cuda_sft_file requires --features cuda");
}
