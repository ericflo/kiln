//! CLI interface for kiln — structured subcommands with clap.

use std::io::Write;

use clap::{Parser, Subcommand};
use console::style;

const TOP_LEVEL_OVERVIEW: &str = r#"Kiln serves Qwen3.5-4B from one Rust process and lets you adapt it live with LoRA training.

Running `kiln` with no subcommand starts the OpenAI-compatible server, just like `kiln serve`. Commands such as `kiln health`, `kiln train sft`, `kiln train grpo`, and `kiln adapters list` talk to a running server.

After `kiln serve`, open http://127.0.0.1:8420/ui for the embedded dashboard: status, adapters, training monitoring, and quick inference.

Common next steps:
  kiln serve          start the server explicitly
  kiln health         inspect a running server
  kiln train sft      train a LoRA adapter from corrections
  kiln train grpo     train a LoRA adapter from scored completions
  kiln adapters list  list saved adapters and show which one is active
"#;

const TOP_LEVEL_EXAMPLES: &str = r#"Examples:
  kiln serve
      Start the inference server explicitly. Running `kiln` with no subcommand also starts serving.

      Then open http://127.0.0.1:8420/ui for status, adapters, training monitoring, and quick inference.

  kiln health
      Check whether the local server is ready and show model, adapter, scheduler, and training status.

  kiln train sft --file examples.jsonl --adapter my-task
      Teach the model from corrected chat examples and hot-swap the trained LoRA adapter.

  kiln train grpo --file grpo-batch.json --adapter my-task
      Improve an adapter from scored completions using GRPO rewards.

  kiln adapters list
      Show saved adapters and which adapter is active on the running server.
"#;

const TRAIN_OVERVIEW: &str = r#"Submit SFT or GRPO training jobs to the running Kiln server at http://localhost:8420 by default.

SFT reads newline-delimited chat correction examples from JSONL. GRPO reads one JSON request/batch with scored completions.
"#;

const TRAIN_EXAMPLES: &str = r#"Examples:
  kiln train sft --file corrections.jsonl --adapter support-bot
      Train from JSONL chat correction examples and hot-swap the resulting LoRA adapter.

  kiln train grpo --file grpo-batch.json --adapter support-bot
      Train from one JSON GRPO request/batch containing prompts, completions, and rewards.

  kiln train status
      Show the training queue and recent jobs on the running server.

  kiln train status --job-id train_123
      Inspect one training job by ID.
"#;

const ADAPTERS_OVERVIEW: &str = r#"Inspect and manage LoRA adapters on the running Kiln server at http://localhost:8420 by default.

Use these commands after `kiln serve` is running; they call the adapter API rather than reading local files directly.
"#;

const ADAPTERS_EXAMPLES: &str = r#"Examples:
  kiln adapters list
      Show saved adapters and which adapter is active on the running server.

  kiln adapters load support-bot
      Load a saved adapter into the running server.

  kiln adapters unload
      Unload the active adapter and revert the running server to the base model.

  kiln adapters unload support-bot
      Backcompat form; the name is ignored because the server unloads the active adapter.

  kiln adapters delete support-bot
      Delete an adapter through the running server.
"#;

const CONFIG_OVERVIEW: &str = r#"Validate a Kiln TOML config file without starting the server.

Use this before `kiln serve` to catch invalid values, confirm resolved model settings, and preview feature toggles such as prefix cache, CUDA graphs, and speculative decoding.

By default, `kiln config` checks the built-in defaults plus environment overrides. Pass `--file` to validate a specific TOML file and see the effective settings that `kiln serve --config <file>` would use.
"#;

const CONFIG_EXAMPLES: &str = r#"Examples:
  kiln config
      Validate the default configuration and any KILN_* environment overrides.

  kiln config --file kiln.toml
      Validate kiln.toml before starting the server with `kiln serve --config kiln.toml`.

  kiln config --file ./config/production.toml
      Check a production config file and print the effective server, model, logging, and feature settings.
"#;

/// Render a structured server error response. Falls back to HTTP status if the body
/// is not the expected `{error: {code, message, hint}}` shape.
///
/// The server's `ApiError` returns errors in OpenAI's nested-object form (see
/// `crates/kiln-server/src/error.rs`). The CLI previously assumed `error` was a
/// bare string and silently dropped the helpful `hint` field; this helper plugs
/// the CLI back into that contract.
fn render_api_error(body: &serde_json::Value, status: reqwest::StatusCode) -> String {
    if let Some(err) = body.get("error").and_then(|e| e.as_object()) {
        let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("");
        let hint = err.get("hint").and_then(|h| h.as_str()).unwrap_or("");
        let code = err.get("code").and_then(|c| c.as_str()).unwrap_or("");
        let mut out = if msg.is_empty() {
            status.to_string()
        } else {
            msg.to_string()
        };
        if !code.is_empty() {
            out = format!("{out} ({code})");
        }
        if !hint.is_empty() {
            out = format!("{out}\n  {} {hint}", style("hint:").dim().cyan());
        }
        out
    } else if let Some(s) = body.get("error").and_then(|e| e.as_str()) {
        s.to_string()
    } else {
        status.to_string()
    }
}

/// Kiln — single-model inference server with live online learning
#[derive(Parser)]
#[command(
    name = "kiln",
    version,
    about = "Single-model inference server with live online learning",
    long_about = TOP_LEVEL_OVERVIEW,
    after_help = TOP_LEVEL_EXAMPLES,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path to TOML config file
    #[arg(long, short, global = true)]
    pub config: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the inference server explicitly; running `kiln` with no subcommand also serves
    Serve {
        /// Override the served model identifier exposed at /v1/models.
        /// Wins over KILN_SERVED_MODEL_ID env and TOML `model.served_model_id`.
        #[arg(long, value_name = "ID")]
        served_model_id: Option<String>,
    },

    /// Submit training data to a running server
    #[command(
        subcommand,
        long_about = TRAIN_OVERVIEW,
        after_help = TRAIN_EXAMPLES
    )]
    Train(TrainCommands),

    /// Manage LoRA adapters on a running server
    #[command(
        subcommand,
        long_about = ADAPTERS_OVERVIEW,
        after_help = ADAPTERS_EXAMPLES
    )]
    Adapters(AdapterCommands),

    /// Check health of a running server
    Health {
        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,

        /// Emit raw JSON instead of the pretty-printed tree
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Validate a config file without starting the server
    #[command(
        name = "config",
        long_about = CONFIG_OVERVIEW,
        after_help = CONFIG_EXAMPLES
    )]
    ConfigCheck {
        /// Path to config file to validate
        #[arg(long, short)]
        file: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum TrainCommands {
    /// Train a LoRA adapter from corrected SFT examples
    Sft {
        /// Path to JSONL chat correction examples, one example per line
        #[arg(long, short)]
        file: String,

        /// Adapter name to train (created if it doesn't exist)
        #[arg(long, default_value = "default")]
        adapter: String,

        /// Learning rate
        #[arg(long, default_value = "1e-4")]
        lr: f64,

        /// Number of epochs
        #[arg(long, default_value = "1")]
        epochs: u32,

        /// LoRA rank for the trained adapter
        #[arg(long)]
        lora_rank: Option<usize>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Train a LoRA adapter from scored GRPO completions
    Grpo {
        /// Path to one JSON GRPO request/batch with scored completions
        #[arg(long, short)]
        file: String,

        /// Adapter name to train
        #[arg(long, default_value = "default")]
        adapter: String,

        /// LoRA rank for the trained adapter
        #[arg(long)]
        lora_rank: Option<usize>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Show training queue / per-job status
    Status {
        /// Specific job ID to look up. If omitted, shows the full queue.
        #[arg(long)]
        job_id: Option<String>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
}

#[derive(Subcommand)]
pub enum AdapterCommands {
    /// List saved adapters and show which one is active
    List {
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Load an adapter from disk
    Load {
        /// Adapter name
        name: String,
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Unload the active adapter and revert to the base model
    Unload {
        /// Optional legacy adapter name; ignored because the server unloads the active adapter
        name: Option<String>,
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Delete an adapter
    Delete {
        /// Adapter name
        name: String,
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
}

/// Probe GPU 0 via `nvidia-smi` for device name and VRAM (total, free) in MiB.
///
/// Returns `None` if nvidia-smi is missing, exits non-zero, or output cannot be parsed.
/// Banner display is purely cosmetic, so any failure is silent.
fn probe_gpu_info() -> Option<(String, u64, u64)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = std::str::from_utf8(&output.stdout).ok()?;
    let line = stdout.lines().next()?.trim();
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next()?.to_string();
    let total_mib: u64 = parts.next()?.parse().ok()?;
    let free_mib: u64 = parts.next()?.parse().ok()?;
    if name.is_empty() {
        return None;
    }
    Some((name, total_mib, free_mib))
}

/// Print the startup banner with config details.
pub fn print_banner(host: &str, port: u16, model_path: Option<&str>, config_path: Option<&str>) {
    let mut stderr = std::io::stderr();

    let _ = writeln!(stderr);
    let _ = writeln!(
        stderr,
        "  {}",
        style("┌─────────────────────────────────────┐").cyan()
    );
    let _ = writeln!(
        stderr,
        "  {}",
        style("│           🔥 K I L N 🔥             │").cyan().bold()
    );
    let _ = writeln!(
        stderr,
        "  {}",
        style("│   inference · training · adapters   │").cyan()
    );
    let _ = writeln!(
        stderr,
        "  {}",
        style("└─────────────────────────────────────┘").cyan()
    );
    let _ = writeln!(stderr);

    let _ = writeln!(
        stderr,
        "  {} {}",
        style("Version:").dim(),
        style(env!("CARGO_PKG_VERSION")).white().bold()
    );

    if let Some(cp) = config_path {
        let _ = writeln!(
            stderr,
            "  {} {}",
            style("Config:").dim(),
            style(cp).white()
        );
    }

    let mode = if model_path.is_some() {
        style("GPU inference").green().bold()
    } else {
        style("mock (no model loaded)").yellow().bold()
    };
    let _ = writeln!(stderr, "  {} {}", style("Mode:").dim(), mode);

    if model_path.is_none() {
        let _ = writeln!(stderr);
        let _ = writeln!(
            stderr,
            "  {} set {} or {} in TOML for real inference/training.",
            style("Next:").dim(),
            style("KILN_MODEL_PATH=./Qwen3.5-4B").yellow().bold(),
            style("model.path").yellow().bold()
        );
        let _ = writeln!(
            stderr,
            "  {} training endpoints return 503 in mock mode.",
            style("Note:").dim()
        );
    }

    if let Some(mp) = model_path {
        let _ = writeln!(
            stderr,
            "  {} {}",
            style("Model:").dim(),
            style(mp).white()
        );
    }

    let cuda_status = if candle_core::utils::cuda_is_available() {
        style("available ✓").green()
    } else {
        style("not available").yellow()
    };
    let _ = writeln!(stderr, "  {} {}", style("CUDA:").dim(), cuda_status);

    if let Some((name, total_mib, free_mib)) = probe_gpu_info() {
        let _ = writeln!(
            stderr,
            "  {} {}",
            style("GPU:").dim(),
            style(name).white().bold()
        );
        let _ = writeln!(
            stderr,
            "  {} {} MiB total, {} MiB free",
            style("VRAM:").dim(),
            style(format!("{total_mib}")).cyan().bold(),
            style(format!("{free_mib}")).cyan()
        );
    }

    let _ = writeln!(
        stderr,
        "  {} {}",
        style("Listen:").dim(),
        style(format!("http://{host}:{port}")).cyan().bold()
    );

    let _ = writeln!(stderr);
    let _ = writeln!(
        stderr,
        "  {} /ui, /v1/chat/completions, /v1/train/sft, /health, /metrics",
        style("Endpoints:").dim()
    );
    let _ = writeln!(stderr);
}

/// Format a uptime duration in seconds as a compact human string ("1h 23m 4s",
/// "5m 30s", "12s"). Drops leading zero units. Used by the pretty health view.
fn format_uptime_secs(total: u64) -> String {
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// Render the /health response body as a pretty tree matching the `kiln config`
/// style. Pure function over a parsed JSON value so tests can pin the layout
/// without standing up a live server. Returns the body to print *after* the
/// "✓ Server is healthy" header, with no leading or trailing newline.
pub fn format_health_pretty(body: &serde_json::Value) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();

    let version = body.get("version").and_then(|v| v.as_str()).unwrap_or("?");
    let uptime_secs = body
        .get("uptime_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("?");
    let backend = body.get("backend").and_then(|v| v.as_str()).unwrap_or("?");
    let active_adapter = body
        .get("active_adapter")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| "(none)".to_string());
    let adapters_loaded = body
        .get("adapters_loaded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let _ = writeln!(out, "  {} {}", style("Version:").dim(), style(version).white().bold());
    let _ = writeln!(
        out,
        "  {}  {}",
        style("Uptime:").dim(),
        style(format_uptime_secs(uptime_secs)).white().bold()
    );
    let _ = writeln!(out, "  {}   {}", style("Model:").dim(), style(model).white().bold());
    let _ = writeln!(out, "  {} {}", style("Backend:").dim(), style(backend).white().bold());
    let _ = writeln!(
        out,
        "  {} {}",
        style("Adapter:").dim(),
        style(&active_adapter).white().bold()
    );
    let _ = writeln!(
        out,
        "  {} {} loaded",
        style("Adapters:").dim(),
        style(adapters_loaded).cyan().bold()
    );

    if let Some(sched) = body.get("scheduler").and_then(|v| v.as_object()) {
        let waiting = sched.get("waiting").and_then(|v| v.as_u64()).unwrap_or(0);
        let running = sched.get("running").and_then(|v| v.as_u64()).unwrap_or(0);
        let blocks_used = sched.get("blocks_used").and_then(|v| v.as_u64()).unwrap_or(0);
        let blocks_free = sched.get("blocks_free").and_then(|v| v.as_u64()).unwrap_or(0);
        let blocks_total = sched.get("blocks_total").and_then(|v| v.as_u64()).unwrap_or(0);
        let _ = writeln!(
            out,
            "  {} waiting={} running={}  blocks={}/{} ({} free)",
            style("Scheduler:").dim(),
            style(waiting).cyan(),
            style(running).cyan(),
            style(blocks_used).cyan(),
            style(blocks_total).cyan(),
            style(blocks_free).cyan()
        );
    }

    if let Some(gpu) = body.get("gpu_memory").and_then(|v| v.as_object()) {
        let total = gpu.get("total_vram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let model_gb = gpu.get("model_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let kv = gpu.get("kv_cache_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let train = gpu
            .get("training_budget_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let _ = writeln!(
            out,
            "  {} {} GB total  model={} GB  kv={} GB  train={} GB",
            style("GPU VRAM:").dim(),
            style(format!("{total:.1}")).cyan().bold(),
            style(format!("{model_gb:.1}")).cyan(),
            style(format!("{kv:.1}")).cyan(),
            style(format!("{train:.1}")).cyan()
        );
    }

    let training = body.get("training").and_then(|v| v.as_object());
    let active_job = training
        .and_then(|t| t.get("active_job"))
        .and_then(|v| v.as_object());
    let queued = training
        .and_then(|t| t.get("queued"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if let Some(job) = active_job {
        let job_id = job.get("job_id").and_then(|v| v.as_str()).unwrap_or("?");
        let progress = job.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let _ = writeln!(
            out,
            "  {} running job={} progress={}%",
            style("Training:").dim(),
            style(job_id).white().bold(),
            style(format!("{:.1}", progress * 100.0)).cyan().bold()
        );
    } else if queued > 0 {
        let _ = writeln!(
            out,
            "  {} idle (queued={})",
            style("Training:").dim(),
            style(queued).cyan()
        );
    } else {
        let _ = writeln!(out, "  {} idle", style("Training:").dim());
    }

    if let Some(checks) = body.get("checks").and_then(|v| v.as_array()) {
        if !checks.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "  {}", style("Checks:").dim());
            for c in checks {
                let name = c.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let pass = c.get("pass").and_then(|v| v.as_bool()).unwrap_or(false);
                if pass {
                    let _ = writeln!(out, "    {} {}", style("✓").green().bold(), name);
                } else {
                    let _ = writeln!(out, "    {} {}", style("✗").red().bold(), name);
                }
            }
        }
    }

    out
}

/// Run the `health` CLI subcommand: GET /health on the server.
///
/// `json=false` (default) renders a tree-style diagnostic that matches
/// `kiln config`. `json=true` preserves the older raw `serde_json::to_string_pretty`
/// behavior — useful when scripting or piping into `jq`. On non-success status,
/// the raw JSON error body is always printed regardless of `json` so failure
/// diagnostics are never lossy.
pub async fn run_health(url: &str, json: bool) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/health")).await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!(
            "{} Server is healthy",
            style("✓").green().bold()
        );
        if json {
            println!("{}", serde_json::to_string_pretty(&body)?);
        } else {
            println!();
            print!("{}", format_health_pretty(&body));
        }
    } else {
        eprintln!(
            "{} Server returned {}",
            style("✗").red().bold(),
            style(status).red()
        );
        eprintln!("{}", serde_json::to_string_pretty(&body)?);
        std::process::exit(1);
    }
    Ok(())
}

/// Run the `config check` CLI subcommand: validate config without starting.
pub fn run_config_check(file: Option<&str>) -> anyhow::Result<()> {
    use crate::config::KilnConfig;

    match KilnConfig::load(file) {
        Ok(config) => {
            println!(
                "{} Configuration is valid",
                style("✓").green().bold()
            );
            println!();
            println!(
                "  {} {}:{}",
                style("Server:").dim(),
                config.server.host,
                config.server.port
            );
            println!(
                "  {} {}",
                style("Model ID:").dim(),
                config.model.model_id
            );
            println!(
                "  {} {}",
                style("Served as:").dim(),
                config.model.effective_served_model_id()
            );
            if let Some(ref p) = config.model.path {
                println!("  {} {}", style("Model path:").dim(), p);
            }
            println!(
                "  {} {}",
                style("Log level:").dim(),
                config.logging.level
            );
            println!(
                "  {} {}",
                style("Log format:").dim(),
                config.logging.format
            );
            println!(
                "  {} {}",
                style("KV cache FP8:").dim(),
                config.memory.kv_cache_fp8
            );
            println!(
                "  {} {}",
                style("CUDA graphs:").dim(),
                config.memory.cuda_graphs
            );
            println!(
                "  {} {}",
                style("Prefix cache:").dim(),
                config.prefix_cache.enabled
            );
            println!(
                "  {} {}",
                style("Speculative:").dim(),
                config.speculative.enabled
            );
            Ok(())
        }
        Err(e) => {
            eprintln!(
                "{} Configuration error: {e}",
                style("✗").red().bold()
            );
            std::process::exit(1);
        }
    }
}

/// Run the `adapters list` CLI subcommand.
pub async fn run_adapters_list(url: &str) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/v1/adapters")).await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        print_adapters_list(&body)?;
    } else {
        eprintln!(
            "{} Server returned {}",
            style("✗").red().bold(),
            style(status).red()
        );
        eprintln!("{}", serde_json::to_string_pretty(&body)?);
        std::process::exit(1);
    }
    Ok(())
}

fn format_size_bytes(size: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    if size >= 1024 * 1024 * 1024 {
        format!("{:.1} GiB", size as f64 / GIB)
    } else if size >= 1024 * 1024 {
        format!("{:.1} MiB", size as f64 / MIB)
    } else if size >= 1024 {
        format!("{:.1} KiB", size as f64 / KIB)
    } else {
        format!("{size} B")
    }
}

fn format_adapters_list(body: &serde_json::Value) -> anyhow::Result<String> {
    use std::fmt::Write as _;

    let Some(available) = body.get("available").and_then(|a| a.as_array()) else {
        return Ok(serde_json::to_string_pretty(body)?);
    };

    let active = body.get("active").and_then(|a| a.as_str());
    if available.is_empty() {
        return Ok(style("No saved adapters are available").dim().to_string());
    }

    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} {} saved adapter(s):",
        style("✓").green().bold(),
        available.len()
    );
    for adapter in available {
        let name = adapter.get("name").and_then(|n| n.as_str()).unwrap_or("?");
        let status_str = if active == Some(name) {
            style("active").green()
        } else {
            style("available").dim()
        };
        let has_config = adapter
            .get("has_config")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let has_weights = adapter
            .get("has_weights")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let size = adapter
            .get("size_bytes")
            .and_then(|v| v.as_u64())
            .map(format_size_bytes)
            .unwrap_or_else(|| "unknown size".to_string());

        let _ = writeln!(
            out,
            "  {} [{}] config={} weights={} size={}",
            style(name).white().bold(),
            status_str,
            style(has_config).cyan(),
            style(has_weights).cyan(),
            style(size).cyan()
        );
    }

    Ok(out.trim_end().to_string())
}

fn print_adapters_list(body: &serde_json::Value) -> anyhow::Result<()> {
    println!("{}", format_adapters_list(body)?);
    Ok(())
}

fn adapter_load_url(url: &str) -> String {
    format!("{url}/v1/adapters/load")
}

fn adapter_unload_url(url: &str) -> String {
    format!("{url}/v1/adapters/unload")
}

fn build_adapter_load_payload(name: &str) -> serde_json::Value {
    serde_json::json!({ "name": name })
}

/// Run the `adapters load` CLI subcommand.
pub async fn run_adapters_load(url: &str, name: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .post(adapter_load_url(url))
        .json(&build_adapter_load_payload(name))
        .send()
        .await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!(
            "{} Adapter '{}' loaded",
            style("✓").green().bold(),
            style(name).white().bold()
        );
    } else {
        eprintln!(
            "{} Failed to load adapter '{}': {}",
            style("✗").red().bold(),
            name,
            render_api_error(&body, status)
        );
        std::process::exit(1);
    }
    Ok(())
}

/// Run the `adapters unload` CLI subcommand.
pub async fn run_adapters_unload(url: &str, name: Option<&str>) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .post(adapter_unload_url(url))
        .send()
        .await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        if let Some(name) = name {
            println!(
                "{} Active adapter unloaded; reverted to base model (ignored legacy name '{}')",
                style("✓").green().bold(),
                style(name).white().bold()
            );
        } else {
            println!(
                "{} Active adapter unloaded; reverted to base model",
                style("✓").green().bold()
            );
        }
    } else {
        eprintln!(
            "{} Failed to unload active adapter: {}",
            style("✗").red().bold(),
            render_api_error(&body, status)
        );
        std::process::exit(1);
    }
    Ok(())
}

/// Run the `adapters delete` CLI subcommand.
pub async fn run_adapters_delete(url: &str, name: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .delete(format!("{url}/v1/adapters/{name}"))
        .send()
        .await?;
    let status = resp.status();

    if status.is_success() {
        println!(
            "{} Adapter '{}' deleted",
            style("✓").green().bold(),
            style(name).white().bold()
        );
    } else {
        let body: serde_json::Value = resp.json().await?;
        eprintln!(
            "{} Failed to delete adapter '{}': {}",
            style("✗").red().bold(),
            name,
            render_api_error(&body, status)
        );
        std::process::exit(1);
    }
    Ok(())
}

/// Run the `train sft` CLI subcommand.
pub async fn run_train_sft(
    url: &str,
    file: &str,
    adapter: &str,
    lr: f64,
    epochs: u32,
    lora_rank: Option<usize>,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read {file}: {e}"))?;

    let mut examples = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let val: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("Invalid JSON on line {}: {e}", i + 1))?;
        examples.push(val);
    }

    println!(
        "{} Submitting {} example(s) for SFT training on adapter '{}'",
        style("→").cyan().bold(),
        style(examples.len()).white().bold(),
        style(adapter).white().bold()
    );

    let body = build_sft_training_payload(examples, adapter, lr, epochs, lora_rank);

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{url}/v1/train/sft"))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!(
            "{} Training job submitted",
            style("✓").green().bold()
        );
        let job_id = resp_body.get("job_id").and_then(|j| j.as_str());
        if let Some(id) = job_id {
            println!("  {} {}", style("Job ID:").dim(), id);
        }
        match job_id {
            Some(id) => println!(
                "  {} kiln train status --job-id {id} --url {url}",
                style("Check status:").dim()
            ),
            None => println!(
                "  {} kiln train status --url {url}",
                style("Check status:").dim()
            ),
        }
    } else {
        eprintln!(
            "{} Training submission failed: {}",
            style("✗").red().bold(),
            render_api_error(&resp_body, status)
        );
        std::process::exit(1);
    }
    Ok(())
}

/// Run the `train grpo` CLI subcommand.
pub async fn run_train_grpo(
    url: &str,
    file: &str,
    adapter: &str,
    lora_rank: Option<usize>,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read {file}: {e}"))?;

    let body: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in {file}: {e}"))?;

    let body = build_grpo_training_payload(body, adapter, lora_rank)?;

    println!(
        "{} Submitting GRPO training batch on adapter '{}'",
        style("→").cyan().bold(),
        style(adapter).white().bold()
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{url}/v1/train/grpo"))
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!(
            "{} GRPO training job submitted",
            style("✓").green().bold()
        );
        let job_id = resp_body.get("job_id").and_then(|j| j.as_str());
        if let Some(id) = job_id {
            println!("  {} {}", style("Job ID:").dim(), id);
        }
        match job_id {
            Some(id) => println!(
                "  {} kiln train status --job-id {id} --url {url}",
                style("Check status:").dim()
            ),
            None => println!(
                "  {} kiln train status --url {url}",
                style("Check status:").dim()
            ),
        }
    } else {
        eprintln!(
            "{} GRPO submission failed: {}",
            style("✗").red().bold(),
            render_api_error(&resp_body, status)
        );
        std::process::exit(1);
    }
    Ok(())
}

fn build_sft_training_payload(
    examples: Vec<serde_json::Value>,
    adapter: &str,
    lr: f64,
    epochs: u32,
    lora_rank: Option<usize>,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "output_name": adapter,
        "learning_rate": lr,
        "epochs": epochs,
    });
    if let Some(rank) = lora_rank {
        config["lora_rank"] = serde_json::json!(rank);
    }

    serde_json::json!({
        "examples": examples,
        "config": config,
    })
}

fn build_grpo_training_payload(
    mut body: serde_json::Value,
    adapter: &str,
    lora_rank: Option<usize>,
) -> anyhow::Result<serde_json::Value> {
    let obj = body
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("GRPO request must be a JSON object with groups and config"))?;
    obj.remove("adapter_name");

    let config = obj
        .entry("config")
        .or_insert_with(|| serde_json::json!({}));
    let config_obj = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("GRPO request config must be a JSON object"))?;
    config_obj.remove("epochs");
    config_obj.remove("num_epochs");
    config_obj.insert("output_name".into(), serde_json::json!(adapter));
    if let Some(rank) = lora_rank {
        config_obj.insert("lora_rank".into(), serde_json::json!(rank));
    }

    Ok(body)
}

/// Run the `train status` CLI subcommand.
///
/// With `job_id` set, GETs `/v1/train/status/{id}` and prints a one-job summary.
/// Without `job_id`, GETs `/v1/train/status` (overall list) and prints all jobs
/// grouped by state: running first, then queued, then completed/failed.
pub async fn run_train_status(url: &str, job_id: Option<&str>) -> anyhow::Result<()> {
    if let Some(id) = job_id {
        return print_single_job_status(url, id).await;
    }
    print_all_job_statuses(url).await
}

async fn print_single_job_status(url: &str, id: &str) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/v1/train/status/{id}")).await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        eprintln!(
            "{} Failed to get status for job '{}': {}",
            style("✗").red().bold(),
            id,
            render_api_error(&body, status)
        );
        std::process::exit(1);
    }

    print_job_summary(&body);
    Ok(())
}

async fn print_all_job_statuses(url: &str) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/v1/train/status")).await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if !status.is_success() {
        eprintln!(
            "{} Server returned {}",
            style("✗").red().bold(),
            style(status).red()
        );
        eprintln!("{}", serde_json::to_string_pretty(&body)?);
        std::process::exit(1);
    }

    // The server returns a bare JSON array; some older shapes wrap it as
    // {"jobs": [...]}. Accept either.
    let jobs = body
        .as_array()
        .cloned()
        .or_else(|| body.get("jobs").and_then(|j| j.as_array()).cloned())
        .unwrap_or_default();

    if jobs.is_empty() {
        println!("{}", style("No training jobs").dim());
        return Ok(());
    }

    // Group by state; order: running, queued, completed, failed
    let mut running = Vec::new();
    let mut queued = Vec::new();
    let mut terminal = Vec::new();
    for job in &jobs {
        match job.get("state").and_then(|s| s.as_str()).unwrap_or("") {
            "running" => running.push(job),
            "queued" => queued.push(job),
            _ => terminal.push(job),
        }
    }
    // Sort terminal by elapsed_secs ascending (most recent submissions last).
    terminal.sort_by(|a, b| {
        let ea = a.get("elapsed_secs").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let eb = b.get("elapsed_secs").and_then(|v| v.as_f64()).unwrap_or(0.0);
        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{} {} job(s):",
        style("✓").green().bold(),
        jobs.len()
    );
    for job in running.iter().chain(queued.iter()).chain(terminal.iter()) {
        print_job_line(job);
    }
    Ok(())
}

fn style_state(state: &str) -> console::StyledObject<String> {
    let s = state.to_string();
    match state {
        "queued" => style(s).dim(),
        "running" => style(s).cyan(),
        "completed" => style(s).green(),
        "failed" => style(s).red(),
        _ => style(s),
    }
}

fn print_job_summary(job: &serde_json::Value) {
    let id = job.get("job_id").and_then(|v| v.as_str()).unwrap_or("?");
    let state = job.get("state").and_then(|v| v.as_str()).unwrap_or("?");
    let adapter = job
        .get("adapter_name")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let progress_pct = (job
        .get("progress")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        * 100.0)
        .round() as i64;
    let elapsed = job
        .get("elapsed_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .round() as i64;

    println!("{} Job {}", style("✓").green().bold(), style(id).white().bold());
    println!("  {} {}", style("State:").dim(), style_state(state));
    println!("  {} {}", style("Adapter:").dim(), style(adapter).white());
    println!("  {} {}%", style("Progress:").dim(), progress_pct);
    if let Some(loss) = job.get("current_loss").and_then(|v| v.as_f64()) {
        println!("  {} {loss:.4}", style("Loss:").dim());
    }
    println!("  {} {}s", style("Elapsed:").dim(), elapsed);
}

fn print_job_line(job: &serde_json::Value) {
    let id = job.get("job_id").and_then(|v| v.as_str()).unwrap_or("?");
    let state = job.get("state").and_then(|v| v.as_str()).unwrap_or("?");
    let adapter = job
        .get("adapter_name")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let progress_pct = (job
        .get("progress")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        * 100.0)
        .round() as i64;
    let elapsed = job
        .get("elapsed_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .round() as i64;
    println!(
        "  {} [{}] adapter={} {}% ({}s)",
        style(id).white().bold(),
        style_state(state),
        adapter,
        progress_pct,
        elapsed
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use serde_json::json;

    #[test]
    fn build_sft_training_payload_uses_nested_config() {
        let body = build_sft_training_payload(
            vec![json!({
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Bonjour"},
                ]
            })],
            "sft-adapter",
            2e-4,
            3,
            Some(8),
        );

        assert_eq!(body["config"]["output_name"], "sft-adapter");
        assert_eq!(body["config"]["learning_rate"], 2e-4);
        assert_eq!(body["config"]["epochs"], 3);
        assert_eq!(body["config"]["lora_rank"], 8);
        assert!(body.get("adapter_name").is_none());
        assert!(body.get("num_epochs").is_none());
    }

    #[test]
    fn build_sft_training_payload_omits_unset_lora_rank() {
        let body = build_sft_training_payload(vec![], "sft-adapter", 1e-4, 1, None);

        assert_eq!(body["config"]["output_name"], "sft-adapter");
        assert!(body["config"].get("lora_rank").is_none());
    }

    #[test]
    fn build_grpo_training_payload_overrides_output_name_in_config() {
        let mut body = json!({
            "groups": [{
                "messages": [{"role": "user", "content": "Write a haiku"}],
                "completions": [{"text": "Moonlit pond", "reward": 1.0}],
            }],
            "config": {
                "output_name": "old-adapter",
                "learning_rate": 5e-5,
                "epochs": 3,
            },
        });
        body.as_object_mut()
            .unwrap()
            .insert("adapter_name".to_string(), json!("legacy-top-level"));

        let body = build_grpo_training_payload(body, "grpo-adapter", Some(16)).unwrap();

        assert_eq!(body["config"]["output_name"], "grpo-adapter");
        assert_eq!(body["config"]["learning_rate"], 5e-5);
        assert_eq!(body["config"]["lora_rank"], 16);
        assert!(body.get("adapter_name").is_none());
        assert!(body["config"].get("epochs").is_none());
        assert!(body["config"].get("num_epochs").is_none());
    }

    #[test]
    fn build_grpo_training_payload_creates_config() {
        let body = json!({
            "groups": [{
                "messages": [{"role": "user", "content": "Write a haiku"}],
                "completions": [{"text": "Moonlit pond", "reward": 1.0}],
            }],
        });

        let body = build_grpo_training_payload(body, "grpo-adapter", None).unwrap();

        assert_eq!(body["config"]["output_name"], "grpo-adapter");
        assert!(body["config"].get("lora_rank").is_none());
    }

    #[test]
    fn render_api_error_structured_with_hint() {
        let body = json!({
            "error": {
                "code": "adapter_not_found",
                "message": "Adapter 'foo' does not exist",
                "hint": "List available adapters with GET /v1/adapters",
            }
        });
        let out = render_api_error(&body, StatusCode::NOT_FOUND);
        assert!(
            out.contains("Adapter 'foo' does not exist"),
            "expected message in output, got: {out}"
        );
        assert!(
            out.contains("(adapter_not_found)"),
            "expected code annotation in output, got: {out}"
        );
        assert!(
            out.contains("List available adapters with GET /v1/adapters"),
            "expected hint in output, got: {out}"
        );
        // The "hint:" label is emitted (possibly with ANSI styling around it).
        assert!(out.contains("hint:"), "expected hint label, got: {out}");
    }

    #[test]
    fn render_api_error_structured_without_hint() {
        let body = json!({
            "error": {
                "code": "invalid_messages",
                "message": "Bad request",
            }
        });
        let out = render_api_error(&body, StatusCode::BAD_REQUEST);
        assert!(out.contains("Bad request"));
        assert!(out.contains("(invalid_messages)"));
        assert!(
            !out.contains("hint:"),
            "should not render a hint label when hint is missing, got: {out}"
        );
    }

    #[test]
    fn cli_format_adapters_list_uses_current_available_shape() {
        let body = json!({
            "active": "support-bot",
            "available": [
                {
                    "name": "support-bot",
                    "has_config": true,
                    "has_weights": true,
                    "size_bytes": 2048,
                    "modified_at": "2026-05-03T00:00:00Z",
                    "files": ["adapter_config.json", "adapter_model.safetensors"],
                },
                {
                    "name": "draft-bot",
                    "has_config": true,
                    "has_weights": false,
                    "size_bytes": 0,
                    "modified_at": null,
                    "files": ["adapter_config.json"],
                },
            ],
        });

        let out = format_adapters_list(&body).expect("format failed");

        assert!(out.contains("2 saved adapter(s)"), "got: {out}");
        assert!(out.contains("support-bot"), "got: {out}");
        assert!(out.contains("[active]"), "got: {out}");
        assert!(out.contains("draft-bot"), "got: {out}");
        assert!(out.contains("[available]"), "got: {out}");
        assert!(out.contains("config=true"), "got: {out}");
        assert!(out.contains("weights=false"), "got: {out}");
        assert!(out.contains("size=2.0 KiB"), "got: {out}");
    }

    #[test]
    fn cli_format_adapters_list_empty_saved_state() {
        let body = json!({
            "active": null,
            "available": [],
        });

        let out = format_adapters_list(&body).expect("format failed");

        assert!(
            out.contains("No saved adapters are available"),
            "got: {out}"
        );
        assert!(
            !out.contains("No adapters loaded"),
            "old loaded-adapter empty state should not appear, got: {out}"
        );
    }

    #[test]
    fn cli_adapter_load_and_unload_routes_match_current_api() {
        assert_eq!(
            adapter_load_url("http://localhost:8420"),
            "http://localhost:8420/v1/adapters/load"
        );
        assert_eq!(
            adapter_unload_url("http://localhost:8420"),
            "http://localhost:8420/v1/adapters/unload"
        );
        assert_eq!(build_adapter_load_payload("support-bot"), json!({ "name": "support-bot" }));
    }

    #[test]
    fn cli_parses_adapters_unload_without_name() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "adapters", "unload"]).expect("parse failed");
        match cli.command {
            Some(Commands::Adapters(AdapterCommands::Unload { name, url })) => {
                assert_eq!(name, None);
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected adapters unload, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn cli_parses_adapters_unload_legacy_name() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "adapters", "unload", "support-bot"])
            .expect("parse failed");
        match cli.command {
            Some(Commands::Adapters(AdapterCommands::Unload { name, url })) => {
                assert_eq!(name.as_deref(), Some("support-bot"));
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected adapters unload, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn render_api_error_legacy_string_shape() {
        // Older / non-ApiError handlers may still return error as a bare string.
        let body = json!({"error": "boom"});
        let out = render_api_error(&body, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(out, "boom");
    }

    #[test]
    fn render_api_error_missing_error_key_falls_back_to_status() {
        let body = json!({"unrelated": "field"});
        let out = render_api_error(&body, StatusCode::BAD_GATEWAY);
        assert!(
            out.contains("502"),
            "expected HTTP status fallback, got: {out}"
        );
    }

    #[test]
    fn parses_status_subcommand() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "train", "status", "--job-id", "abc"])
            .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Status { job_id, url })) => {
                assert_eq!(job_id.as_deref(), Some("abc"));
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected Train(Status), got {:?}", other.is_some()),
        }
    }

    #[test]
    fn parses_status_subcommand_no_job_id() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "train", "status"]).expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Status { job_id, url })) => {
                assert!(job_id.is_none(), "expected no job_id");
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected Train(Status), got {:?}", other.is_some()),
        }
    }

    #[test]
    fn parses_status_subcommand_custom_url() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "train",
            "status",
            "--url",
            "http://example.com:9000",
        ])
        .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Status { job_id, url })) => {
                assert!(job_id.is_none());
                assert_eq!(url, "http://example.com:9000");
            }
            other => panic!("expected Train(Status), got {:?}", other.is_some()),
        }
    }

    #[test]
    fn format_uptime_secs_renders_compact() {
        assert_eq!(format_uptime_secs(0), "0s");
        assert_eq!(format_uptime_secs(45), "45s");
        assert_eq!(format_uptime_secs(60), "1m 0s");
        assert_eq!(format_uptime_secs(330), "5m 30s");
        assert_eq!(format_uptime_secs(3600), "1h 0m 0s");
        assert_eq!(format_uptime_secs(4984), "1h 23m 4s");
    }

    #[test]
    fn format_health_pretty_full() {
        let body = json!({
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": 4984,
            "model": "qwen3.5-4b-kiln (32L, 16H, 4KV)",
            "backend": "model",
            "active_adapter": "my-adapter",
            "adapters_loaded": 3,
            "scheduler": {
                "waiting": 1,
                "running": 2,
                "blocks_used": 100,
                "blocks_free": 156,
                "blocks_total": 256,
            },
            "gpu_memory": {
                "total_vram_gb": 47.5,
                "model_gb": 8.2,
                "kv_cache_gb": 12.4,
                "training_budget_gb": 6.0,
                "inference_memory_fraction": 0.85,
            },
            "training": {
                "active_job": null,
                "queued": 0,
            },
            "checks": [
                {"name": "model_loaded", "pass": true},
                {"name": "scheduler_responsive", "pass": true},
            ],
        });
        let out = format_health_pretty(&body);
        assert!(out.contains("Version:"), "got: {out}");
        assert!(out.contains("0.1.0"), "got: {out}");
        assert!(out.contains("Uptime:"), "got: {out}");
        assert!(out.contains("1h 23m 4s"), "got: {out}");
        assert!(out.contains("Model:"), "got: {out}");
        assert!(out.contains("qwen3.5-4b-kiln"), "got: {out}");
        assert!(out.contains("Backend:"), "got: {out}");
        assert!(out.contains("model"), "got: {out}");
        assert!(out.contains("Adapter:"), "got: {out}");
        assert!(out.contains("my-adapter"), "got: {out}");
        assert!(out.contains("Adapters:"), "got: {out}");
        assert!(out.contains("3 loaded"), "got: {out}");
        assert!(out.contains("Scheduler:"), "got: {out}");
        assert!(out.contains("waiting=1"), "got: {out}");
        assert!(out.contains("running=2"), "got: {out}");
        assert!(out.contains("blocks=100/256"), "got: {out}");
        assert!(out.contains("(156 free)"), "got: {out}");
        assert!(out.contains("GPU VRAM:"), "got: {out}");
        assert!(out.contains("47.5 GB total"), "got: {out}");
        assert!(out.contains("model=8.2 GB"), "got: {out}");
        assert!(out.contains("kv=12.4 GB"), "got: {out}");
        assert!(out.contains("train=6.0 GB"), "got: {out}");
        assert!(out.contains("Training:"), "got: {out}");
        assert!(out.contains("idle"), "got: {out}");
        assert!(out.contains("Checks:"), "got: {out}");
        assert!(out.contains("model_loaded"), "got: {out}");
        assert!(out.contains("scheduler_responsive"), "got: {out}");
        // ✓ glyph appears at least twice (once per check, plus the runtime header is rendered separately)
        assert!(out.contains("✓"), "expected at least one ✓ glyph, got: {out}");
    }

    #[test]
    fn format_health_pretty_minimal() {
        // Mock backend without GPU memory budget, no checks, no scheduler stats.
        let body = json!({
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": 12,
            "model": "mock-model",
            "backend": "mock",
            "active_adapter": null,
            "adapters_loaded": 0,
            "scheduler": null,
            "gpu_memory": null,
            "training": {
                "active_job": null,
                "queued": 0,
            },
            "checks": [],
        });
        let out = format_health_pretty(&body);
        assert!(out.contains("Version:"), "got: {out}");
        assert!(out.contains("Uptime:"), "got: {out}");
        assert!(out.contains("12s"), "got: {out}");
        assert!(out.contains("Adapter:"), "got: {out}");
        assert!(out.contains("(none)"), "got: {out}");
        assert!(out.contains("0 loaded"), "got: {out}");
        assert!(out.contains("Training:"), "got: {out}");
        assert!(out.contains("idle"), "got: {out}");
        // Subgroups must be ABSENT when the corresponding fields are null/empty.
        assert!(
            !out.contains("Scheduler:"),
            "scheduler subgroup should not render when null, got: {out}"
        );
        assert!(
            !out.contains("GPU VRAM:"),
            "gpu_memory subgroup should not render when null, got: {out}"
        );
        assert!(
            !out.contains("Checks:"),
            "checks subgroup should not render when empty, got: {out}"
        );
    }

    #[test]
    fn format_health_pretty_active_job() {
        let body = json!({
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": 60,
            "model": "qwen3.5-4b-kiln",
            "backend": "model",
            "active_adapter": null,
            "adapters_loaded": 0,
            "scheduler": null,
            "gpu_memory": null,
            "training": {
                "active_job": {
                    "job_id": "sft-7f9c",
                    "progress": 0.4237,
                },
                "queued": 2,
            },
            "checks": [],
        });
        let out = format_health_pretty(&body);
        assert!(out.contains("Training:"), "got: {out}");
        assert!(out.contains("running"), "got: {out}");
        assert!(out.contains("job=sft-7f9c"), "got: {out}");
        assert!(out.contains("progress=42.4%"), "got: {out}");
        // Active job line should NOT also say "idle".
        assert!(
            !out.contains("idle"),
            "active job render should not include 'idle', got: {out}"
        );
    }

    #[test]
    fn format_health_pretty_queued_only() {
        // No active job but queue is non-empty: idle line must include the queue depth.
        let body = json!({
            "status": "ok",
            "version": "0.1.0",
            "uptime_seconds": 60,
            "model": "qwen3.5-4b-kiln",
            "backend": "model",
            "active_adapter": null,
            "adapters_loaded": 0,
            "scheduler": null,
            "gpu_memory": null,
            "training": {
                "active_job": null,
                "queued": 5,
            },
            "checks": [],
        });
        let out = format_health_pretty(&body);
        assert!(out.contains("Training:"), "got: {out}");
        assert!(out.contains("idle"), "got: {out}");
        assert!(out.contains("queued=5"), "got: {out}");
    }

    #[test]
    fn parses_health_with_json_flag() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "health", "--json"]).expect("parse failed");
        match cli.command {
            Some(Commands::Health { url, json }) => {
                assert_eq!(url, "http://localhost:8420");
                assert!(json, "--json should set json=true");
            }
            other => panic!("expected Health, got {:?}", other.is_some()),
        }
    }

    #[test]
    fn parses_health_default_is_pretty() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "health"]).expect("parse failed");
        match cli.command {
            Some(Commands::Health { url, json }) => {
                assert_eq!(url, "http://localhost:8420");
                assert!(!json, "default json flag should be false");
            }
            other => panic!("expected Health, got {:?}", other.is_some()),
        }
    }
}
