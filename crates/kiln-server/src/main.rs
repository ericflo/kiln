use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use kiln_server::cli::{
    AdapterCommands, AdaptersArgs, BenchArgs, Cli, Commands, ConfigArgs, ServeArgs, TrainCommands,
    TrainArgs,
};
use kiln_server::config::KilnConfig;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config_path = cli.config_path.as_deref();

    match cli.command {
        Some(Commands::Serve(args)) => cmd_serve(args, config_path).await,
        Some(Commands::Train(args)) => cmd_train(args).await,
        Some(Commands::Adapters(args)) => cmd_adapters(args).await,
        Some(Commands::Bench(args)) => cmd_bench(args),
        Some(Commands::Config(args)) => cmd_config(args, config_path),
        // No subcommand defaults to serve (backwards-compatible)
        None => {
            cmd_serve(ServeArgs { model_path: None, host: None, port: None }, config_path).await
        }
    }
}

// ---------------------------------------------------------------------------
// Startup banner
// ---------------------------------------------------------------------------

fn print_banner(config: &KilnConfig, mode: &str) {
    let version = env!("CARGO_PKG_VERSION");
    eprintln!();
    eprintln!(
        "  {} v{}",
        style("⚡ kiln").bold().cyan(),
        style(version).dim()
    );
    eprintln!(
        "  {}",
        style("Single-model inference server with live online learning").dim()
    );
    eprintln!();
    eprintln!(
        "  {} {}",
        style("Mode:").bold(),
        style(mode).yellow()
    );
    eprintln!(
        "  {} {}:{}",
        style("Listen:").bold(),
        config.server.host,
        style(config.server.port).green()
    );
    eprintln!(
        "  {} {}",
        style("Model:").bold(),
        config.model.path.as_deref().unwrap_or(&config.model.model_id)
    );

    // Feature flags
    let mut features = Vec::new();
    if config.memory.kv_cache_fp8 {
        features.push("FP8 KV");
    }
    if config.memory.cuda_graphs {
        features.push("CUDA graphs");
    }
    if config.prefix_cache.enabled {
        features.push("prefix cache");
    }
    if config.speculative.enabled {
        features.push("speculative decoding");
    }
    if !features.is_empty() {
        eprintln!(
            "  {} {}",
            style("Features:").bold(),
            features.join(", ")
        );
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Progress bar helpers
// ---------------------------------------------------------------------------

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn finish_spinner(pb: &ProgressBar, msg: String) {
    pb.set_style(
        ProgressStyle::with_template("  {msg}")
            .unwrap(),
    );
    pb.finish_with_message(format!("{} {}", style("✓").green().bold(), msg));
}

// ---------------------------------------------------------------------------
// serve
// ---------------------------------------------------------------------------

async fn cmd_serve(args: ServeArgs, config_path: Option<&str>) -> Result<()> {
    use kiln_core::config::ModelConfig;
    use kiln_core::tokenizer::KilnTokenizer;
    use kiln_model::engine::MockEngine;
    use kiln_model::forward::GpuWeights;
    use kiln_model::ModelRunner;
    use kiln_scheduler::{Scheduler, SchedulerConfig};
    use kiln_server::state::AppState;

    let mut config = KilnConfig::load(config_path)?;

    // CLI args override config
    if let Some(ref mp) = args.model_path {
        config.model.path = Some(mp.clone());
    }
    if let Some(ref h) = args.host {
        config.server.host = h.clone();
    }
    if let Some(p) = args.port {
        config.server.port = p;
    }

    let model_path = config.model.path.as_deref();
    let mode = if model_path.is_some() { "inference" } else { "mock" };

    // Initialize logging before banner so tracing works
    kiln_server::logging::init(&config.logging.level, &config.logging.format)?;

    print_banner(&config, mode);

    let model_config = ModelConfig::qwen3_5_4b();
    let model_id = &config.model.model_id;
    let tokenizer_path = config.model.tokenizer_path.as_deref();

    // Load tokenizer with spinner
    let pb = spinner("Loading tokenizer...");
    let tokenizer = load_tokenizer(tokenizer_path, model_path, model_id)?;
    finish_spinner(&pb, format!(
        "Tokenizer loaded (vocab size: {})",
        style(tokenizer.vocab_size()).cyan()
    ));

    let mut state = if let Some(mp) = model_path {
        // Real inference mode
        let pb = spinner("Loading model weights...");
        let load_start = Instant::now();
        let model_weights = kiln_model::load_model(Path::new(mp), &model_config)?;
        let device = if candle_core::utils::cuda_is_available() {
            candle_core::Device::new_cuda(0)?
        } else {
            candle_core::Device::Cpu
        };
        let gpu_weights = GpuWeights::from_model_weights(&model_weights, &device)?;
        let load_secs = load_start.elapsed().as_secs_f64();
        finish_spinner(&pb, format!(
            "Model loaded in {:.1}s on {}",
            load_secs,
            if matches!(device, candle_core::Device::Cpu) {
                style("CPU").yellow()
            } else {
                style("CUDA").green()
            }
        ));

        let runner_tokenizer = load_tokenizer(tokenizer_path, Some(mp), model_id)?;
        let runner = ModelRunner::new_with_options(
            gpu_weights,
            runner_tokenizer,
            model_config.clone(),
            config.memory.cuda_graphs,
        );

        let adapter_dir = config
            .model
            .adapter_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(mp).join("adapters"));

        if !adapter_dir.exists() {
            std::fs::create_dir_all(&adapter_dir)?;
        }

        tracing::info!(adapter_dir = %adapter_dir.display(), "real inference mode");
        AppState::new_real(
            model_config,
            runner,
            tokenizer,
            device,
            adapter_dir,
            &config.memory,
            config.server.request_timeout_secs,
        )
    } else {
        // Mock mode
        let scheduler_config = SchedulerConfig {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: config.prefix_cache.enabled,
            prefix_cache_max_blocks: config.prefix_cache.max_blocks,
        };
        let num_blocks = 8192;
        let scheduler = Scheduler::new(scheduler_config, num_blocks);
        let engine = MockEngine::new(model_config.clone());
        AppState::new_mock(
            model_config,
            scheduler,
            Arc::new(engine),
            tokenizer,
            config.server.request_timeout_secs,
        )
    };

    state.checkpoint_interval = config.training.checkpoint_interval;

    let shutdown_flag = state.shutdown.clone();
    kiln_server::training_queue::spawn_training_worker(state.clone(), shutdown_flag.clone());

    let app = kiln_server::api::router(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    eprintln!(
        "  {} Listening on {}",
        style("→").green().bold(),
        style(&addr).underlined()
    );
    eprintln!();

    let shutdown_timeout_secs = config.server.shutdown_timeout_secs;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown_flag))
        .await?;

    eprintln!(
        "\n  {} Shutting down (draining requests, {}s timeout)...",
        style("⏳").yellow(),
        shutdown_timeout_secs
    );
    tokio::time::sleep(std::time::Duration::from_secs(shutdown_timeout_secs)).await;

    Ok(())
}

fn load_tokenizer(
    tokenizer_path: Option<&str>,
    model_path: Option<&str>,
    model_id: &str,
) -> Result<kiln_core::tokenizer::KilnTokenizer> {
    use kiln_core::tokenizer::KilnTokenizer;

    if let Some(path) = tokenizer_path {
        return KilnTokenizer::from_file(path).context("failed to load tokenizer from path");
    }
    if let Some(mp) = model_path {
        let tok_file = Path::new(mp).join("tokenizer.json");
        if tok_file.exists() {
            return KilnTokenizer::from_file(tok_file.to_str().unwrap())
                .context("failed to load tokenizer from model dir");
        }
    }
    KilnTokenizer::from_pretrained(model_id).context("failed to load tokenizer from HF Hub")
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

async fn cmd_train(args: TrainArgs) -> Result<()> {
    match args.command {
        TrainCommands::Sft(sft) => cmd_train_sft(sft).await,
        TrainCommands::Grpo(grpo) => cmd_train_grpo(grpo).await,
        TrainCommands::Status(status) => cmd_train_status(status).await,
    }
}

async fn cmd_train_sft(args: kiln_server::cli::TrainSftArgs) -> Result<()> {
    let data = read_file_or_stdin(&args.file)?;
    let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();

    eprintln!(
        "  {} Submitting {} SFT examples to {}",
        style("→").cyan().bold(),
        style(lines.len()).cyan(),
        style(&args.server).underlined()
    );

    let pb = ProgressBar::new(1);
    pb.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✓"]),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message("Sending training request...");

    // Parse examples
    let examples: Vec<serde_json::Value> = lines
        .iter()
        .map(|l| serde_json::from_str(l))
        .collect::<std::result::Result<_, _>>()
        .context("failed to parse JSONL — each line must be a valid JSON object")?;

    let mut body = serde_json::Map::new();
    body.insert("examples".into(), serde_json::Value::Array(examples));
    body.insert("learning_rate".into(), args.learning_rate.into());
    body.insert("lora_rank".into(), args.lora_rank.into());
    body.insert("epochs".into(), args.epochs.into());
    if let Some(name) = &args.adapter_name {
        body.insert("output_name".into(), name.clone().into());
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/train/sft", args.server))
        .json(&body)
        .send()
        .await
        .context("failed to connect to kiln server")?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    finish_spinner(&pb, if status.is_success() {
        format!("Training job submitted ({})", style(status).green())
    } else {
        format!("Request failed ({})", style(status).red())
    });

    if !status.is_success() {
        eprintln!(
            "\n  {} {}\n",
            style("Error:").red().bold(),
            text
        );
        eprintln!("  {} Is the kiln server running at {}?", style("Hint:").yellow(), args.server);
        std::process::exit(1);
    }

    // Pretty-print response
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
        eprintln!("\n{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n{text}");
    }

    Ok(())
}

async fn cmd_train_grpo(args: kiln_server::cli::TrainGrpoArgs) -> Result<()> {
    let data = read_file_or_stdin(&args.file)?;

    let pb = spinner("Sending GRPO training request...");

    let body: serde_json::Value =
        serde_json::from_str(&data).context("failed to parse GRPO JSON file")?;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/train/grpo", args.server))
        .json(&body)
        .send()
        .await
        .context("failed to connect to kiln server")?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    finish_spinner(&pb, if status.is_success() {
        format!("GRPO training submitted ({})", style(status).green())
    } else {
        format!("Request failed ({})", style(status).red())
    });

    if !status.is_success() {
        eprintln!("\n  {} {}", style("Error:").red().bold(), text);
        std::process::exit(1);
    }

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
        eprintln!("\n{}", serde_json::to_string_pretty(&json)?);
    } else {
        eprintln!("\n{text}");
    }

    Ok(())
}

async fn cmd_train_status(args: kiln_server::cli::TrainStatusArgs) -> Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/train/status", args.server))
        .send()
        .await
        .context("failed to connect to kiln server")?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        eprintln!("  {} {} {}", style("✗").red(), style("Error:").red().bold(), text);
        std::process::exit(1);
    }

    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
        print_training_status(&json);
    } else {
        println!("{text}");
    }

    Ok(())
}

fn print_training_status(json: &serde_json::Value) {
    let active = json.get("active").and_then(|v| v.as_bool()).unwrap_or(false);

    if active {
        let step = json.get("step").and_then(|v| v.as_u64()).unwrap_or(0);
        let total = json.get("total_steps").and_then(|v| v.as_u64()).unwrap_or(0);
        let loss = json.get("loss").and_then(|v| v.as_f64()).unwrap_or(0.0);

        eprintln!(
            "  {} Training in progress: step {}/{} (loss: {:.6})",
            style("⚡").yellow(),
            style(step).cyan(),
            total,
            loss
        );

        if total > 0 {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::with_template(
                    "  [{bar:30.cyan/dim}] {pos}/{len} steps"
                )
                .unwrap()
                .progress_chars("━╸─"),
            );
            pb.set_position(step);
            pb.abandon(); // show it without waiting
        }
    } else {
        eprintln!("  {} No training in progress", style("○").dim());
    }

    if let Some(queue) = json.get("queue_length").and_then(|v| v.as_u64()) {
        if queue > 0 {
            eprintln!("  {} {} job(s) queued", style("◎").yellow(), queue);
        }
    }
}

// ---------------------------------------------------------------------------
// adapters
// ---------------------------------------------------------------------------

async fn cmd_adapters(args: AdaptersArgs) -> Result<()> {
    match args.command {
        AdapterCommands::List(a) => cmd_adapter_list(a).await,
        AdapterCommands::Load(a) => cmd_adapter_action("load", &a.name, &a.server).await,
        AdapterCommands::Unload(a) => cmd_adapter_action("unload", &a.name, &a.server).await,
        AdapterCommands::Delete(a) => cmd_adapter_action("delete", &a.name, &a.server).await,
    }
}

async fn cmd_adapter_list(args: kiln_server::cli::AdapterServerArgs) -> Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/adapters", args.server))
        .send()
        .await
        .context("failed to connect to kiln server")?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if !status.is_success() {
        eprintln!("  {} {} {}", style("✗").red(), style("Error:").red().bold(), text);
        std::process::exit(1);
    }

    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    let adapters = json.get("adapters").and_then(|v| v.as_array());

    match adapters {
        Some(list) if list.is_empty() => {
            eprintln!("  {} No adapters loaded", style("○").dim());
        }
        Some(list) => {
            eprintln!(
                "  {} {} adapter(s):\n",
                style("●").green(),
                list.len()
            );
            for adapter in list {
                let name = adapter.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let active = adapter.get("active").and_then(|v| v.as_bool()).unwrap_or(false);
                let status_icon = if active {
                    style("●").green()
                } else {
                    style("○").dim()
                };
                let status_text = if active { "active" } else { "loaded" };
                eprintln!("    {} {} ({})", status_icon, style(name).bold(), status_text);
            }
        }
        None => {
            // Fallback: print raw
            println!("{text}");
        }
    }

    Ok(())
}

async fn cmd_adapter_action(action: &str, name: &str, server: &str) -> Result<()> {
    let pb = spinner(&format!(
        "{}ing adapter '{}'...",
        capitalize(action),
        name
    ));

    let client = reqwest::Client::new();
    let url = match action {
        "load" => format!("{server}/v1/adapters/{name}/load"),
        "unload" => format!("{server}/v1/adapters/{name}/unload"),
        "delete" => format!("{server}/v1/adapters/{name}"),
        _ => unreachable!(),
    };

    let resp = if action == "delete" {
        client.delete(&url).send().await
    } else {
        client.post(&url).send().await
    }
    .context("failed to connect to kiln server")?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if status.is_success() {
        finish_spinner(&pb, format!(
            "Adapter '{}' {}ed",
            style(name).bold(),
            action
        ));
    } else {
        pb.finish_and_clear();
        eprintln!(
            "  {} Failed to {} adapter '{}': {}",
            style("✗").red(),
            action,
            name,
            text
        );
        std::process::exit(1);
    }

    Ok(())
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

// ---------------------------------------------------------------------------
// bench (delegates to existing bench logic)
// ---------------------------------------------------------------------------

fn cmd_bench(args: BenchArgs) -> Result<()> {
    // The bench binary remains separate (kiln-bench) for now.
    // This subcommand provides a convenience wrapper that execs it.
    let exe = std::env::current_exe()?;
    let bench_bin = exe.with_file_name("kiln-bench");

    if !bench_bin.exists() {
        eprintln!(
            "  {} Benchmark binary not found at {}",
            style("✗").red(),
            bench_bin.display()
        );
        eprintln!(
            "  {} Build with: {}",
            style("Hint:").yellow(),
            style("cargo build --release --features cuda --bin kiln-bench").dim()
        );
        std::process::exit(1);
    }

    let mut cmd = std::process::Command::new(&bench_bin);
    cmd.arg("--model-path").arg(&args.model_path);
    cmd.arg("--max-output-tokens")
        .arg(args.max_output_tokens.to_string());
    cmd.arg("--prompt-tokens")
        .arg(args.prompt_tokens.to_string());
    cmd.arg("--training-steps")
        .arg(args.training_steps.to_string());
    if args.skip_training {
        cmd.arg("--skip-training");
    }

    let status = cmd.status().context("failed to run kiln-bench")?;
    std::process::exit(status.code().unwrap_or(1));
}

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

fn cmd_config(args: ConfigArgs, config_path: Option<&str>) -> Result<()> {
    let config = KilnConfig::load(config_path)?;

    if args.raw {
        // Show reconstructed TOML
        println!(
            "# Effective configuration (defaults + config file + env overrides)\n\
             # Config file: {}\n",
            config_path.unwrap_or("(none — using defaults)")
        );
        print_config_toml(&config);
    } else {
        print_config_summary(&config);
    }

    Ok(())
}

fn print_config_summary(config: &KilnConfig) {
    eprintln!("  {}", style("Kiln Configuration").bold().underlined());
    eprintln!();

    eprintln!("  {}", style("[server]").cyan().bold());
    eprintln!("    host:              {}", config.server.host);
    eprintln!("    port:              {}", style(config.server.port).green());
    eprintln!("    request_timeout:   {}s", config.server.request_timeout_secs);
    eprintln!("    shutdown_timeout:  {}s", config.server.shutdown_timeout_secs);
    eprintln!();

    eprintln!("  {}", style("[model]").cyan().bold());
    eprintln!(
        "    path:              {}",
        config.model.path.as_deref().unwrap_or("(none — mock mode)")
    );
    eprintln!("    model_id:          {}", config.model.model_id);
    eprintln!(
        "    tokenizer:         {}",
        config.model.tokenizer_path.as_deref().unwrap_or("(auto)")
    );
    eprintln!(
        "    adapter_dir:       {}",
        config.model.adapter_dir.as_deref().unwrap_or("(auto)")
    );
    eprintln!();

    eprintln!("  {}", style("[memory]").cyan().bold());
    eprintln!(
        "    num_blocks:        {}",
        config.memory.num_blocks.map_or("(auto)".to_string(), |n| n.to_string())
    );
    eprintln!(
        "    gpu_memory_gb:     {}",
        config.memory.gpu_memory_gb.map_or("(auto)".to_string(), |g| format!("{g:.1}"))
    );
    eprintln!("    inference_frac:    {:.0}%", config.memory.inference_memory_fraction * 100.0);
    eprintln!(
        "    kv_cache_fp8:      {}",
        if config.memory.kv_cache_fp8 { style("enabled").green() } else { style("disabled").dim() }
    );
    eprintln!(
        "    cuda_graphs:       {}",
        if config.memory.cuda_graphs { style("enabled").green() } else { style("disabled").dim() }
    );
    eprintln!();

    eprintln!("  {}", style("[training]").cyan().bold());
    eprintln!(
        "    grad_checkpoint:   {}",
        if config.training.no_grad_checkpoint {
            style("disabled").dim().to_string()
        } else {
            config.training.grad_checkpoint_segments
                .map_or("enabled (auto)".to_string(), |s| format!("enabled ({s} segments)"))
        }
    );
    eprintln!(
        "    checkpoint_interval: {}",
        config.training.checkpoint_interval.map_or("(end only)".to_string(), |n| format!("every {n} steps"))
    );
    eprintln!();

    eprintln!("  {}", style("[logging]").cyan().bold());
    eprintln!("    level:             {}", config.logging.level);
    eprintln!("    format:            {}", config.logging.format);
    eprintln!();

    eprintln!("  {}", style("[prefix_cache]").cyan().bold());
    eprintln!(
        "    enabled:           {}",
        if config.prefix_cache.enabled { style("yes").green() } else { style("no").dim() }
    );
    eprintln!(
        "    max_blocks:        {}",
        config.prefix_cache.max_blocks.map_or("(auto — 25% of total)".to_string(), |n| n.to_string())
    );
    eprintln!();

    eprintln!("  {}", style("[speculative]").cyan().bold());
    eprintln!(
        "    enabled:           {}",
        if config.speculative.enabled { style("yes").green() } else { style("no").dim() }
    );
    if config.speculative.enabled {
        eprintln!("    draft_layers:      {}", config.speculative.draft_layers);
        eprintln!("    spec_tokens:       {}", config.speculative.num_speculative_tokens);
    }
    eprintln!();
}

fn print_config_toml(config: &KilnConfig) {
    println!("[server]");
    println!("host = \"{}\"", config.server.host);
    println!("port = {}", config.server.port);
    println!("request_timeout_secs = {}", config.server.request_timeout_secs);
    println!("shutdown_timeout_secs = {}", config.server.shutdown_timeout_secs);
    println!();

    println!("[model]");
    if let Some(ref p) = config.model.path {
        println!("path = \"{p}\"");
    }
    println!("model_id = \"{}\"", config.model.model_id);
    if let Some(ref p) = config.model.tokenizer_path {
        println!("tokenizer_path = \"{p}\"");
    }
    if let Some(ref p) = config.model.adapter_dir {
        println!("adapter_dir = \"{p}\"");
    }
    println!();

    println!("[memory]");
    if let Some(n) = config.memory.num_blocks {
        println!("num_blocks = {n}");
    }
    if let Some(g) = config.memory.gpu_memory_gb {
        println!("gpu_memory_gb = {g}");
    }
    println!("inference_memory_fraction = {}", config.memory.inference_memory_fraction);
    println!("kv_cache_fp8 = {}", config.memory.kv_cache_fp8);
    println!("cuda_graphs = {}", config.memory.cuda_graphs);
    println!();

    println!("[training]");
    if let Some(s) = config.training.grad_checkpoint_segments {
        println!("grad_checkpoint_segments = {s}");
    }
    println!("no_grad_checkpoint = {}", config.training.no_grad_checkpoint);
    if let Some(n) = config.training.checkpoint_interval {
        println!("checkpoint_interval = {n}");
    }
    println!();

    println!("[logging]");
    println!("level = \"{}\"", config.logging.level);
    println!("format = \"{}\"", config.logging.format);
    println!();

    println!("[prefix_cache]");
    println!("enabled = {}", config.prefix_cache.enabled);
    if let Some(n) = config.prefix_cache.max_blocks {
        println!("max_blocks = {n}");
    }
    println!();

    println!("[speculative]");
    println!("enabled = {}", config.speculative.enabled);
    println!("num_speculative_tokens = {}", config.speculative.num_speculative_tokens);
    println!("draft_layers = {}", config.speculative.draft_layers);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn read_file_or_stdin(path: &str) -> Result<String> {
    if path == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("failed to read from stdin")?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read file: {path}"))
    }
}

/// Wait for SIGTERM or SIGINT, then signal shutdown.
async fn shutdown_signal(shutdown_flag: Arc<std::sync::atomic::AtomicBool>) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            eprintln!("\n  {} Received SIGINT", style("⚡").yellow());
        }
        _ = terminate => {
            eprintln!("\n  {} Received SIGTERM", style("⚡").yellow());
        }
    }

    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
}
