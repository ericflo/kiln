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

/// Run the `health` CLI subcommand: GET /health on the server.
pub async fn run_health(url: &str) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/health")).await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!(
            "{} Server is healthy",
            style("✓").green().bold()
        );
        println!("{}", serde_json::to_string_pretty(&body)?);
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
        if let Some(job_id) = resp_body.get("job_id").and_then(|j| j.as_str()) {
            println!("  {} {}", style("Job ID:").dim(), job_id);
        }
        println!(
            "  {} kiln health --url {url}",
            style("Check status:").dim()
        );
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
        if let Some(job_id) = resp_body.get("job_id").and_then(|j| j.as_str()) {
            println!("  {} {}", style("Job ID:").dim(), job_id);
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
}
