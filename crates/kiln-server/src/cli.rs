//! CLI definitions using clap.
//!
//! Provides subcommands for all major kiln operations:
//! - `serve` — start the inference server
//! - `train` — submit training data to a running server
//! - `adapters` — manage LoRA adapters on a running server
//! - `bench` — run the benchmark suite
//! - `config` — validate and display effective configuration

use clap::{Parser, Subcommand};

/// Kiln — single-model inference server with live online learning.
///
/// A pure-Rust inference server for Qwen3.5-4B with continuous LoRA
/// adaptation via SFT and GRPO training endpoints.
#[derive(Parser, Debug)]
#[command(
    name = "kiln",
    version,
    about = "Single-model inference server with live online learning",
    long_about = "Kiln is a pure-Rust inference server for Qwen3.5-4B with continuous LoRA\n\
                  adaptation. It serves OpenAI-compatible inference while accepting training\n\
                  data via HTTP — your model improves in seconds, not hours."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Path to kiln.toml configuration file
    #[arg(long = "config", short = 'c', global = true)]
    pub config_path: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the inference server
    #[command(alias = "s")]
    Serve(ServeArgs),

    /// Submit training data to a running kiln server
    #[command(alias = "t")]
    Train(TrainArgs),

    /// Manage LoRA adapters on a running server
    #[command(alias = "a")]
    Adapters(AdaptersArgs),

    /// Run the benchmark suite (requires GPU + model weights)
    #[command(alias = "b")]
    Bench(BenchArgs),

    /// Validate and display the effective configuration
    Config(ConfigArgs),
}

// --- Serve ---

#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Model weights directory (overrides config file)
    #[arg(long)]
    pub model_path: Option<String>,

    /// Listen host (default: from config or 0.0.0.0)
    #[arg(long)]
    pub host: Option<String>,

    /// Listen port (default: from config or 8420)
    #[arg(long, short = 'p')]
    pub port: Option<u16>,
}

// --- Train ---

#[derive(Parser, Debug)]
pub struct TrainArgs {
    #[command(subcommand)]
    pub command: TrainCommands,
}

#[derive(Subcommand, Debug)]
pub enum TrainCommands {
    /// Submit SFT training examples from a JSONL file
    Sft(TrainSftArgs),

    /// Submit GRPO training data from a JSONL file
    Grpo(TrainGrpoArgs),

    /// Check training status on the server
    Status(TrainStatusArgs),
}

#[derive(Parser, Debug)]
pub struct TrainSftArgs {
    /// Path to JSONL file with training examples
    #[arg(long, short = 'f')]
    pub file: String,

    /// Server URL (default: http://localhost:8420)
    #[arg(long, default_value = "http://localhost:8420")]
    pub server: String,

    /// Learning rate
    #[arg(long, default_value = "1e-4")]
    pub learning_rate: f64,

    /// LoRA rank
    #[arg(long, default_value = "8")]
    pub lora_rank: usize,

    /// Number of training epochs
    #[arg(long, default_value = "3")]
    pub epochs: usize,

    /// Name for the output adapter
    #[arg(long)]
    pub adapter_name: Option<String>,
}

#[derive(Parser, Debug)]
pub struct TrainGrpoArgs {
    /// Path to JSONL file with GRPO training data
    #[arg(long, short = 'f')]
    pub file: String,

    /// Server URL (default: http://localhost:8420)
    #[arg(long, default_value = "http://localhost:8420")]
    pub server: String,
}

#[derive(Parser, Debug)]
pub struct TrainStatusArgs {
    /// Server URL (default: http://localhost:8420)
    #[arg(long, default_value = "http://localhost:8420")]
    pub server: String,
}

// --- Adapters ---

#[derive(Parser, Debug)]
pub struct AdaptersArgs {
    #[command(subcommand)]
    pub command: AdapterCommands,
}

#[derive(Subcommand, Debug)]
pub enum AdapterCommands {
    /// List all adapters on the server
    #[command(alias = "ls")]
    List(AdapterServerArgs),

    /// Load an adapter by name
    Load(AdapterNameArgs),

    /// Unload an adapter by name
    Unload(AdapterNameArgs),

    /// Delete an adapter by name
    #[command(alias = "rm")]
    Delete(AdapterNameArgs),
}

#[derive(Parser, Debug)]
pub struct AdapterServerArgs {
    /// Server URL (default: http://localhost:8420)
    #[arg(long, default_value = "http://localhost:8420")]
    pub server: String,
}

#[derive(Parser, Debug)]
pub struct AdapterNameArgs {
    /// Adapter name
    pub name: String,

    /// Server URL (default: http://localhost:8420)
    #[arg(long, default_value = "http://localhost:8420")]
    pub server: String,
}

// --- Bench ---

#[derive(Parser, Debug)]
pub struct BenchArgs {
    /// Path to Qwen3.5-4B weights directory
    #[arg(long)]
    pub model_path: String,

    /// Max tokens to generate per request
    #[arg(long, default_value = "128")]
    pub max_output_tokens: usize,

    /// Approximate prompt length in tokens
    #[arg(long, default_value = "512")]
    pub prompt_tokens: usize,

    /// Number of SFT training steps to benchmark
    #[arg(long, default_value = "10")]
    pub training_steps: usize,

    /// Skip training benchmarks
    #[arg(long)]
    pub skip_training: bool,
}

// --- Config ---

#[derive(Parser, Debug)]
pub struct ConfigArgs {
    /// Show the raw TOML instead of a summary
    #[arg(long)]
    pub raw: bool,
}
