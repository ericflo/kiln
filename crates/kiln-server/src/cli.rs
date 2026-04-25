//! CLI interface for kiln — structured subcommands with clap.

use std::io::Write;

use clap::{Parser, Subcommand};
use console::style;

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
    long_about = None,
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
    /// Start the inference server (default when no subcommand given)
    Serve {
        /// Override the served model identifier exposed at /v1/models.
        /// Wins over KILN_SERVED_MODEL_ID env and TOML `model.served_model_id`.
        #[arg(long, value_name = "ID")]
        served_model_id: Option<String>,
    },

    /// Submit training data to a running server
    #[command(subcommand)]
    Train(TrainCommands),

    /// Manage LoRA adapters on a running server
    #[command(subcommand)]
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
    #[command(name = "config")]
    ConfigCheck {
        /// Path to config file to validate
        #[arg(long, short)]
        file: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum TrainCommands {
    /// Submit SFT training examples
    Sft {
        /// Path to JSONL file with training examples
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

        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Submit GRPO training batch
    Grpo {
        /// Path to JSONL file with scored completions
        #[arg(long, short)]
        file: String,

        /// Adapter name to train
        #[arg(long, default_value = "default")]
        adapter: String,

        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Show training queue / per-job status
    Status {
        /// Specific job ID to look up. If omitted, shows the full queue.
        #[arg(long)]
        job_id: Option<String>,

        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
}

#[derive(Subcommand)]
pub enum AdapterCommands {
    /// List all adapters
    List {
        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Load an adapter from disk
    Load {
        /// Adapter name
        name: String,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Unload an active adapter
    Unload {
        /// Adapter name
        name: String,
        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Delete an adapter
    Delete {
        /// Adapter name
        name: String,
        /// Server URL
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
        style("│   inference · training · adapters    │").cyan()
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
        "  {} /v1/chat/completions, /v1/train/sft, /health, /metrics",
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
        let adapters = body.get("adapters").and_then(|a| a.as_array());
        match adapters {
            Some(list) if list.is_empty() => {
                println!("{}", style("No adapters loaded").dim());
            }
            Some(list) => {
                println!(
                    "{} {} adapter(s):",
                    style("✓").green().bold(),
                    list.len()
                );
                for a in list {
                    let name = a.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                    let active = a
                        .get("active")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let status_str = if active {
                        style("active").green()
                    } else {
                        style("loaded").dim()
                    };
                    println!("  {} [{}]", style(name).white().bold(), status_str);
                }
            }
            None => {
                println!("{}", serde_json::to_string_pretty(&body)?);
            }
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

/// Run the `adapters load` CLI subcommand.
pub async fn run_adapters_load(url: &str, name: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{url}/v1/adapters/{name}/load"))
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
pub async fn run_adapters_unload(url: &str, name: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{url}/v1/adapters/{name}/unload"))
        .send()
        .await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!(
            "{} Adapter '{}' unloaded",
            style("✓").green().bold(),
            style(name).white().bold()
        );
    } else {
        eprintln!(
            "{} Failed to unload adapter '{}': {}",
            style("✗").red().bold(),
            name,
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

    let body = serde_json::json!({
        "adapter_name": adapter,
        "examples": examples,
        "learning_rate": lr,
        "num_epochs": epochs,
    });

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
pub async fn run_train_grpo(url: &str, file: &str, adapter: &str) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read {file}: {e}"))?;

    let body: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Invalid JSON in {file}: {e}"))?;

    // Wrap with adapter name if not already present
    let body = if body.get("adapter_name").is_some() {
        body
    } else {
        let mut obj = body;
        obj.as_object_mut()
            .map(|m| m.insert("adapter_name".into(), serde_json::json!(adapter)));
        obj
    };

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
