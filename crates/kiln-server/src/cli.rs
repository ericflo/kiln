//! CLI interface for kiln — structured subcommands with clap.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use console::style;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::trajectory_inspect::{
    RolloutInspection, SegmentInspection, TrajectoryInspectReport, inspect_trajectory_file,
};

use crate::adapter_verify::{
    AdapterVerifyOptions, AdapterVerifyServerReceipt, DEFAULT_VERIFY_PROMPT,
    DETERMINISTIC_GREEDY_TEXT_NOTE, finalize_status, push_check, verify_adapter_offline,
};

const TOP_LEVEL_OVERVIEW: &str = r#"Kiln serves Qwen3.5-4B from one Rust process and lets you adapt it with live LoRA training.

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

const SERVE_OVERVIEW: &str = r#"Start the OpenAI-compatible Kiln server for Qwen3.5-4B on http://127.0.0.1:8420 by default.

Before starting, point Kiln at model weights with KILN_MODEL_PATH or pass a TOML config with --config. After startup, open http://127.0.0.1:8420/ui for the embedded dashboard and use kiln health to check readiness.

If setup stalls, follow https://ericflo.github.io/kiln/quickstart.html first, then https://ericflo.github.io/kiln/troubleshooting.html for model path, CUDA, and config checks.
"#;

const SERVE_EXAMPLES: &str = r#"Examples:
  KILN_MODEL_PATH=/models/Qwen3.5-4B kiln serve
      Start the local server with model weights from KILN_MODEL_PATH.

  kiln serve --eval-mode
      Start with deterministic eval defaults, no-thinking chat-template defaults, adapter headers, and per-request transient cache cleanup.

  KILN_DEFAULT_THINKING_ENABLED=false kiln serve
      Default Qwen/DeepSeek-style chat templates to non-thinking mode unless a request overrides chat_template_kwargs.enable_thinking.

  KILN_FOLD_REASONING_INTO_CONTENT=true kiln serve
      Duplicate separated reasoning into assistant content for compatibility with clients that treat empty content as no response.

  kiln serve --config kiln.toml
      Start with a checked TOML config. Run `kiln config --file kiln.toml` first if you want to preview the effective settings.

  open http://127.0.0.1:8420/ui
      Open the dashboard for status, adapters, training monitoring, and quick inference.

  kiln health
      Confirm the server is ready and inspect model, adapter, scheduler, and training status.

  See https://ericflo.github.io/kiln/quickstart.html and https://ericflo.github.io/kiln/troubleshooting.html if the server cannot find weights, CUDA is unavailable, or config validation fails.
"#;

const HEALTH_OVERVIEW: &str = r#"Check readiness and setup diagnostics for a running Kiln server at http://localhost:8420 by default.

`kiln health` calls the server's /health endpoint and prints a terminal-friendly tree with model, adapter, scheduler, GPU memory, and training status. Use it after `kiln serve` starts, or point --url at a remote server when debugging another host.

If readiness fails, check the /health response first, then follow https://ericflo.github.io/kiln/quickstart.html and https://ericflo.github.io/kiln/troubleshooting.html for model path, CUDA, config, and server-start diagnostics.
"#;

const HEALTH_EXAMPLES: &str = r#"Examples:
  kiln health
      Check whether the local server at http://localhost:8420 is ready.

  kiln health --url http://localhost:8420
      Check a specific Kiln server URL when the default is not the target.

  kiln health --json
      Print the raw /health JSON response for scripts or bug reports.

  curl http://localhost:8420/health
      Call the same readiness endpoint directly if you are narrowing down CLI vs server setup.

  See Troubleshooting if /health is not ready, the model path is wrong, CUDA is unavailable, or the server is not reachable.
"#;

const TRAIN_OVERVIEW: &str = r#"Submit SFT or GRPO training jobs to the running Kiln server at http://localhost:8420 by default.

SFT reads JSONL: one chat correction example per line with a messages array. GRPO reads either one JSON request/batch with groups or JSONL with one group per line; each group has prompt messages plus candidate completions containing text and reward scores.

Add --adapter-smoke-test to record a small base-vs-adapter canary check in train_receipt.json after successful training.

Prefer http://127.0.0.1:8420/ui for guided submission and status. See docs/GRPO_GUIDE.md or docs/site/grpo.html for reward-loop examples.
"#;

const TRAIN_SFT_OVERVIEW: &str = r#"Train from SFT JSONL: one chat correction example per line with a messages array.

Use --adapter-smoke-test to compare base vs trained adapter logits and short greedy outputs before running a full eval.

Open http://127.0.0.1:8420/ui for guided submission and training status.
"#;

const TRAIN_GRPO_OVERVIEW: &str = r#"Train from GRPO data: either one JSON request/batch with groups, or JSONL with one group per line.

Use --adapter-smoke-test to compare base vs trained adapter logits and short greedy outputs before running a full eval.

Open http://127.0.0.1:8420/ui for guided submission and training status. See docs/GRPO_GUIDE.md or docs/site/grpo.html for reward-loop examples.
"#;

const TRAIN_EXAMPLES: &str = r#"Examples:
  kiln train sft --file corrections.jsonl --adapter support-bot
      Train from SFT JSONL: one chat correction example per line with a messages array.

  kiln train sft --file corrections.jsonl --adapter support-bot --adapter-smoke-test
      Train and record adapter-effect smoke metrics in train_receipt.json.

  kiln train grpo --file grpo-batch.json --adapter support-bot
      Train from one GRPO JSON request/batch with groups.

  kiln train grpo --file grpo-groups.jsonl --adapter support-bot
      Train from GRPO JSONL, streaming one group per line through the Vulkan-native path.

  kiln train status
      Show the training queue and recent jobs on the running server.

  kiln train status --job-id train_123
      Inspect one training job by ID.
"#;

const ADAPTERS_OVERVIEW: &str = r#"Inspect and manage LoRA adapters on the running Kiln server at http://localhost:8420 by default.

Most commands call the adapter API after `kiln serve` is running. `kiln adapter verify` can validate local adapter directories offline, and `kiln adapter restore` copies a manifest-described adapter into a local registry.

`kiln adapter verify <name-or-path>` also accepts the singular alias and can validate an adapter directory offline before optionally checking a running server with --url.
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

  kiln adapter verify ./runs/grpo/support-bot
      Validate adapter_config.json, adapter_model.safetensors, rank consistency, and nonzero LoRA effect offline. Prints a JSON receipt.

  kiln adapter verify support-bot --adapter-dir ./Qwen3.5-4B/adapters --url http://localhost:8420
      Validate the installed adapter, load it through the running server, confirm registry state, and compare a fixed base-vs-adapter prompt.

  kiln adapter restore ./runs/grpo/support-bot/adapter_manifest.json --adapter-dir ./Qwen3.5-4B/adapters
      Restore a manifest-described adapter into an adapter registry and verify copied file hashes.
"#;

const EVAL_ADAPTER_OVERVIEW: &str = r#"Run a paired base-vs-adapter eval through a running Kiln server.

The tasks file is JSONL, one task object per line. The request template is a
chat-completions JSON body; string placeholders like {{prompt}}, {{task.prompt}},
{{seed}}, and {{adapter_label}} are rendered for each task/seed pair. The CLI
then forces `adapter: null` for the base request and `adapter: NAME` for the
candidate request so both sides use the same task and seed.

The scorer executable receives one JSON object on stdin with the task, seed,
base response/content, and candidate response/content. It may print a JSON
object with `base_score` and `adapter_score`, a JSON object with `lift`, or a
single numeric lift. The summary records mean lift, stdev, zero-count,
wall-clock stats, config hashes, and adapter hashes from /v1/debug/model-state.
"#;

const EVAL_ADAPTER_EXAMPLES: &str = r#"Examples:
  kiln eval-adapter --adapter support-bot --tasks eval.tasks.jsonl --seeds 3 --request-template request.json --scorer ./score_one.py
      Run paired base/support-bot requests for every task and seed, score each pair, and write eval_summary.json.

  kiln eval-adapter --adapter support-bot --tasks eval.tasks.jsonl --seeds 5 --request-template request.json --scorer ./score_one.py --output support-bot.eval_summary.json --url http://127.0.0.1:8420
      Write the summary to a named file against a specific server.
"#;

const ROLLOUT_GENERATE_OVERVIEW: &str = r#"Generate scored single-turn GRPO rollouts through a running Kiln server.

The tasks file is JSONL, one task object per line. The request template is a
chat-completions JSON body with placeholders like {{prompt}}, {{task.prompt}},
{{seed}}, {{adapter_label}}, and {{thinking_enabled}}. For every task/seed, the
CLI forces an explicit `adapter`, deterministic `seed`, non-streaming response,
performance metadata, and `chat_template_kwargs.enable_thinking`.

The scorer executable receives one JSON object on stdin with the task, request,
response, content, seed, adapter, token usage, and latency. It may print a
single numeric reward or a JSON object with `reward`, `score`, or `value`.
Output JSONL contains one GRPO group per task with ScoredRollout-compatible
completions and metadata recording latency, token counts, seed, adapter, and
scorer output.
"#;

const ROLLOUT_GENERATE_EXAMPLES: &str = r#"Examples:
  kiln rollout-generate --adapter support-bot --thinking false --tasks tasks.jsonl --seeds 4 --request-template request.json --scorer ./score_one.py
      Generate four deterministic scored completions per task and write rollouts.scored.jsonl.

  kiln rollout-generate --adapter base --thinking false --tasks tasks.jsonl --request-template request.json --scorer ./score_one.py --output base.rollouts.jsonl --summary-output base.rollouts.summary.json --url http://127.0.0.1:8420
      Generate base-model rollouts by forcing `adapter: null` in every request.
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

const TRAJECTORY_OVERVIEW: &str = r#"Inspect Pi or Kiln agentic trajectory JSONL before training.

`kiln trajectory inspect` renders each rollout through Kiln's tokenizer and canonical trajectory mask builder, then reports action/env/context token counts, per-segment spans, warning-prefix filtering, decoded target-token previews, and schema warnings.

The command accepts Pi session JSONL events as well as Kiln ScoredRollout or AgenticGroup JSONL. It exits non-zero when no trainable action tokens are present.
"#;

const TRAJECTORY_EXAMPLES: &str = r#"Examples:
  kiln trajectory inspect session.jsonl --tokenizer /models/Qwen3.5-4B/tokenizer.json
      Inspect a Pi session capture with an explicit tokenizer.

  kiln --config kiln.toml trajectory inspect rollouts.jsonl --json
      Emit a machine-readable report using the tokenizer/model paths from kiln.toml.

  kiln trajectory inspect rollouts.jsonl --model-path /models/Qwen3.5-4B --preview-tokens 128
      Load tokenizer.json and chat_template.jinja from a local model directory.
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

/// True if a `reqwest::Error` indicates the Kiln server is unreachable
/// (connect refused, DNS failure, or transport timeout).
///
/// Walks the error chain so I/O errors wrapped under hyper/h2/rustls are still
/// classified — `reqwest::Error::is_connect()` alone misses some cases.
fn is_connection_error(err: &reqwest::Error) -> bool {
    use std::error::Error as _;
    if err.is_connect() || err.is_timeout() {
        return true;
    }
    let mut source: Option<&(dyn std::error::Error + 'static)> = err.source();
    while let Some(e) = source {
        if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            use std::io::ErrorKind;
            if matches!(
                io_err.kind(),
                ErrorKind::ConnectionRefused
                    | ErrorKind::ConnectionReset
                    | ErrorKind::ConnectionAborted
                    | ErrorKind::TimedOut
                    | ErrorKind::AddrNotAvailable
                    | ErrorKind::NotConnected
            ) {
                return true;
            }
        }
        source = e.source();
    }
    false
}

/// Map a `reqwest::Error` to a friendly Kiln CLI diagnostic.
///
/// On connect / DNS / timeout errors (the "is the server even running?" class),
/// print a tip pointing at `kiln serve` + QUICKSTART and `exit(1)` so the user
/// is not buried under raw `reqwest`/`hyper` chain text. For any other transport
/// error, return it wrapped in `anyhow::Error` so today's `?` propagation keeps
/// working unchanged. HTTP-error responses (`status.is_success() == false`)
/// still flow through `render_api_error` since they are reached only after the
/// response body has been read successfully.
fn handle_request_error(url: &str, err: reqwest::Error) -> anyhow::Error {
    if is_connection_error(&err) {
        eprintln!(
            "{} Could not reach Kiln server at {}.",
            style("✗").red().bold(),
            style(url).cyan()
        );
        eprintln!(
            "  Is the server running? Try {} first, or pass {} to point at another host.",
            style("kiln serve").white().bold(),
            style("--url <addr>").white().bold()
        );
        eprintln!("  See https://ericflo.github.io/kiln/quickstart.html for setup help.");
        std::process::exit(1);
    }
    anyhow::Error::new(err)
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

    /// Increase verbosity. `-v` shows debug logs (the legacy startup firehose),
    /// `-vv` adds trace-level kernel detail. Wins over `KILN_LOG_LEVEL` and the
    /// TOML `[logging] level`, but loses to `RUST_LOG` if set explicitly.
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode — only warnings and errors. Mutually exclusive with `--verbose`.
    #[arg(long, short, global = true, conflicts_with = "verbose")]
    pub quiet: bool,
}

impl Cli {
    /// Resolve the effective tracing-subscriber filter directive based on
    /// `--verbose`/`--quiet` flags and a fallback (typically the TOML
    /// `[logging] level`). `RUST_LOG` is *not* consulted here; it's checked
    /// inside `logging::init` and wins regardless of CLI flags.
    pub fn effective_log_level<'a>(&'a self, fallback: &'a str) -> &'a str {
        if self.quiet {
            "warn"
        } else {
            match self.verbose {
                0 => fallback,
                1 => "debug",
                _ => "trace",
            }
        }
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the inference server explicitly; running `kiln` with no subcommand also serves
    #[command(long_about = SERVE_OVERVIEW, after_help = SERVE_EXAMPLES)]
    Serve {
        /// Override the served model identifier exposed at /v1/models.
        /// Wins over KILN_SERVED_MODEL_ID env and TOML `model.served_model_id`.
        #[arg(long, value_name = "ID")]
        served_model_id: Option<String>,
        /// Enable deterministic eval-serving defaults, adapter headers, and
        /// transient cache cleanup between direct completions.
        #[arg(long)]
        eval_mode: bool,
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
        alias = "adapter",
        subcommand,
        long_about = ADAPTERS_OVERVIEW,
        after_help = ADAPTERS_EXAMPLES
    )]
    Adapters(AdapterCommands),

    /// Inspect agentic trajectory JSONL masks before training
    #[command(
        subcommand,
        long_about = TRAJECTORY_OVERVIEW,
        after_help = TRAJECTORY_EXAMPLES
    )]
    Trajectory(TrajectoryCommands),

    /// Check health of a running server
    #[command(long_about = HEALTH_OVERVIEW, after_help = HEALTH_EXAMPLES)]
    Health {
        /// Server URL
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,

        /// Emit raw JSON instead of the pretty-printed tree
        #[arg(long, default_value_t = false)]
        json: bool,
    },

    /// Run a paired base-vs-adapter eval with an external scorer
    #[command(name = "eval-adapter", long_about = EVAL_ADAPTER_OVERVIEW, after_help = EVAL_ADAPTER_EXAMPLES)]
    EvalAdapter {
        /// Adapter name to compare against the base model
        #[arg(long)]
        adapter: String,

        /// JSONL task file, one task object per line
        #[arg(long)]
        tasks: PathBuf,

        /// Number of paired seeds to run for each task
        #[arg(long, default_value_t = 1)]
        seeds: usize,

        /// Chat-completions request template JSON
        #[arg(long = "request-template")]
        request_template: PathBuf,

        /// Executable scorer. Receives pair JSON on stdin and prints JSON or a numeric lift.
        #[arg(long)]
        scorer: PathBuf,

        /// Output summary path
        #[arg(long, default_value = "eval_summary.json")]
        output: PathBuf,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },

    /// Generate scored single-turn rollouts for GRPO training
    #[command(
        name = "rollout-generate",
        long_about = ROLLOUT_GENERATE_OVERVIEW,
        after_help = ROLLOUT_GENERATE_EXAMPLES
    )]
    RolloutGenerate {
        /// Adapter name to use, or `base`/`none`/`null` to force the base model
        #[arg(long)]
        adapter: String,

        /// Explicit Qwen thinking mode to set on every request
        #[arg(
            long,
            action = clap::ArgAction::Set,
            value_parser = clap::value_parser!(bool),
            default_value_t = false
        )]
        thinking: bool,

        /// JSONL task file, one task object per line
        #[arg(long)]
        tasks: PathBuf,

        /// Number of deterministic completions to generate for each task
        #[arg(long, default_value_t = 1)]
        seeds: usize,

        /// First seed value; later seeds increment by one with wrapping arithmetic
        #[arg(long = "seed-start", default_value_t = 0)]
        seed_start: u64,

        /// Chat-completions request template JSON
        #[arg(long = "request-template")]
        request_template: PathBuf,

        /// Executable scorer. Receives completion JSON on stdin and prints a reward.
        #[arg(long)]
        scorer: PathBuf,

        /// Output GRPO JSONL path
        #[arg(long, default_value = "rollouts.scored.jsonl")]
        output: PathBuf,

        /// Output summary JSON path
        #[arg(long = "summary-output", default_value = "rollout_summary.json")]
        summary_output: PathBuf,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
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

    /// §10.14: merge kiln into pi's models/settings config.
    ///
    /// The canonical pi + kiln pipeline (§10.14 in the grand plan):
    ///
    ///   brew install kiln pi
    ///   kiln serve &
    ///   kiln pi-setup       # one-time
    ///   kiln judge distill  # one-time, §10.6.1
    ///   pi                  # use normally; sessions captured
    ///   kiln self-improve   # auto-runs Saturday
    #[command(name = "pi-setup", long_about = PI_SETUP_OVERVIEW)]
    PiSetup {
        /// Override the kiln server URL. `/v1` is appended when omitted.
        #[arg(long, alias = "kiln-url", default_value = "http://localhost:8420")]
        url: String,
        /// Output path for the models.json file. Default
        /// `$HOME/.pi/agent/models.json`.
        #[arg(long)]
        out: Option<String>,
    },

    /// §10.6 self-distillation engine — the "centerpiece" of the
    /// grand plan's agentic deployment.
    #[command(subcommand, name = "judge", long_about = JUDGE_OVERVIEW)]
    Judge(JudgeCommands),

    /// §10.6.2 + §10.14 — kick the weekly self-improve loop.
    #[command(name = "self-improve", long_about = SELF_IMPROVE_OVERVIEW)]
    SelfImprove {
        /// Server URL.
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
        /// Agent adapter to improve.
        #[arg(long, default_value = "pi-coder-current")]
        agent: String,
        /// Judge LoRA alias.
        #[arg(long, default_value = "judge-pi-v1")]
        judge: String,
        /// Disable the §10.6.4 CRISP terseness pass.
        #[arg(long, default_value_t = false)]
        no_crisp: bool,
    },
}

/// §10.6 subcommands.
#[derive(Subcommand)]
pub enum JudgeCommands {
    /// §10.6.1 — distil a turn-judge LoRA from the configured 27B
    /// teacher's multi-axis scoring of (turn, context) pairs.
    Distill {
        /// Server URL.
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
        /// Output adapter name.
        #[arg(long, default_value = "judge-pi-v1")]
        name: String,
        /// Teacher alias.
        #[arg(long, default_value = "qwen3.6-27b@local")]
        teacher: String,
    },
    /// §10.6.3 drift check — periodically re-score with the 27B
    /// teacher and refresh the judge if agreement < 80%.
    DriftCheck {
        /// Server URL.
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
        /// Judge LoRA alias.
        #[arg(long, default_value = "judge-pi-v1")]
        judge: String,
        /// 27B teacher alias.
        #[arg(long, default_value = "qwen3.6-27b@local")]
        teacher: String,
    },
}

const PI_SETUP_OVERVIEW: &str = "Merge kiln into ~/.pi/agent/models.json and settings.json.\n\
Backs up existing files first and preserves unrelated pi providers/settings.\n\
Part of the §10.14 canonical pi + kiln pipeline.";

const JUDGE_OVERVIEW: &str = "§10.6 self-distillation engine.\n\
Distill a turn-judge LoRA once (judge distill), then run the perpetual\n\
self-improve loop forever (kiln self-improve). Drift-check periodically.";

const SELF_IMPROVE_OVERVIEW: &str = "§10.6.2 + §10.14 — kick the weekly self-improve loop.\n\
Scores the week's rollouts with the local judge LoRA, runs GRPO with\n\
judge-derived advantages, optionally engages the §10.6.4 CRISP\n\
terseness pass on successful trajectories.";

#[derive(Subcommand)]
pub enum TrainCommands {
    /// Train a LoRA adapter from corrected SFT examples
    #[command(long_about = TRAIN_SFT_OVERVIEW)]
    Sft {
        /// Path to SFT JSONL: one chat correction example per line with a messages array
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

        /// Run an adapter-effect smoke test after successful training
        #[arg(long)]
        adapter_smoke_test: bool,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value = "http://localhost:8420")]
        url: String,
    },
    /// Train a LoRA adapter from scored GRPO completions
    #[command(long_about = TRAIN_GRPO_OVERVIEW)]
    Grpo {
        /// Path to one GRPO JSON request/batch, or JSONL with one group per line
        #[arg(long, short)]
        file: String,

        /// Adapter name to train
        #[arg(long, default_value = "default")]
        adapter: String,

        /// LoRA rank for the trained adapter
        #[arg(long)]
        lora_rank: Option<usize>,

        /// Run an adapter-effect smoke test after successful training
        #[arg(long)]
        adapter_smoke_test: bool,

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
    /// Verify an adapter directory and optionally prove it through a running server
    Verify {
        /// Adapter name under adapter_dir, or path to an adapter directory
        name_or_path: String,
        /// Adapter registry directory used to resolve bare adapter names offline
        #[arg(long)]
        adapter_dir: Option<PathBuf>,
        /// Optional server URL. When set, the verifier loads the named adapter,
        /// checks /v1/adapters, and compares a fixed base-vs-adapter prompt.
        #[arg(long)]
        url: Option<String>,
        /// Prompt used for the optional server behavior check
        #[arg(long)]
        prompt: Option<String>,
    },
    /// Restore a manifest-described adapter into an adapter registry
    Restore {
        /// Path to adapter_manifest.json
        manifest: PathBuf,
        /// Adapter registry directory. Defaults to config/model adapter_dir when omitted.
        #[arg(long)]
        adapter_dir: Option<PathBuf>,
        /// Override restored adapter name. Defaults to manifest.adapter_name.
        #[arg(long)]
        name: Option<String>,
        /// Replace an existing adapter directory or symlink with the same name
        #[arg(long)]
        overwrite: bool,
    },
}

#[derive(Subcommand)]
pub enum TrajectoryCommands {
    /// Inspect Pi session or Kiln ScoredRollout JSONL through the mask builder
    Inspect {
        /// Pi session JSONL, Kiln ScoredRollout JSONL, or AgenticGroup JSONL
        file: PathBuf,

        /// Emit the full report as pretty JSON
        #[arg(long, default_value_t = false)]
        json: bool,

        /// Treat system/user Pi messages as non-trainable trajectory context segments
        #[arg(long, default_value_t = false)]
        include_context: bool,

        /// Number of action/env target tokens to decode in previews
        #[arg(long, default_value_t = 64)]
        preview_tokens: usize,

        /// Explicit tokenizer.json path. Defaults to KILN_TOKENIZER_PATH,
        /// config model.tokenizer_path, <model-path>/tokenizer.json, or HF model_id.
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Explicit chat template file. Defaults to chat_template.jinja or
        /// tokenizer_config.json beside the tokenizer/model path when available.
        #[arg(long)]
        chat_template: Option<PathBuf>,

        /// Local model directory used to find tokenizer.json and chat template.
        /// Overrides config model.path for this inspection only.
        #[arg(long)]
        model_path: Option<PathBuf>,
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
        style("│           🔥 K I L N 🔥             │")
            .cyan()
            .bold()
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
        let _ = writeln!(stderr, "  {} {}", style("Config:").dim(), style(cp).white());
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
        let _ = writeln!(stderr, "  {} {}", style("Model:").dim(), style(mp).white());
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

/// Build a spinner-style indicatif bar for long-running startup phases (model
/// load, weight upload, KV cache allocation). Returns `None` when stderr isn't
/// attended (CI, systemd, docker) so the JSON log path stays clean.
///
/// The returned `ProgressBar` ticks every 80ms; callers should `set_message`
/// between phases and `finish_and_clear` once done so the line goes away
/// before [`print_ready_line`] writes the final status.
pub fn make_startup_spinner(
    initial_message: impl Into<std::borrow::Cow<'static, str>>,
) -> Option<indicatif::ProgressBar> {
    if !console::Term::stderr().features().is_attended() {
        return None;
    }
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::with_template("  {spinner:.cyan} {msg} {elapsed:.dim}")
            .expect("static spinner template is valid")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message(initial_message);
    Some(pb)
}

/// Emit the single human-readable "ready" line at INFO level. Called once after
/// the listener binds and after any model-load progress finishes. Stays terse on
/// purpose — the banner already covered version/GPU/model details, and we want
/// silence-until-requests after this line.
pub fn print_ready_line(host: &str, port: u16) {
    let mut stderr = std::io::stderr();
    let addr = format!("http://{host}:{port}");
    let _ = writeln!(
        stderr,
        "  {} {} {}",
        style("✓").green().bold(),
        style("Ready on").dim(),
        style(addr).cyan().bold()
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

    let _ = writeln!(
        out,
        "  {} {}",
        style("Version:").dim(),
        style(version).white().bold()
    );
    let _ = writeln!(
        out,
        "  {}  {}",
        style("Uptime:").dim(),
        style(format_uptime_secs(uptime_secs)).white().bold()
    );
    let _ = writeln!(
        out,
        "  {}   {}",
        style("Model:").dim(),
        style(model).white().bold()
    );
    let _ = writeln!(
        out,
        "  {} {}",
        style("Backend:").dim(),
        style(backend).white().bold()
    );
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
        let blocks_used = sched
            .get("blocks_used")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let blocks_free = sched
            .get("blocks_free")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let blocks_total = sched
            .get("blocks_total")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
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
        let total = gpu
            .get("total_vram_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let model_gb = gpu.get("model_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let kv = gpu
            .get("kv_cache_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
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
    let resp = reqwest::get(format!("{url}/health"))
        .await
        .map_err(|e| handle_request_error(url, e))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!("{} Server is healthy", style("✓").green().bold());
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
            println!("{} Configuration is valid", style("✓").green().bold());
            println!();
            println!(
                "  {} {}:{}",
                style("Server:").dim(),
                config.server.host,
                config.server.port
            );
            println!("  {} {}", style("Model ID:").dim(), config.model.model_id);
            println!(
                "  {} {}",
                style("Served as:").dim(),
                config.model.effective_served_model_id()
            );
            if let Some(ref p) = config.model.path {
                println!("  {} {}", style("Model path:").dim(), p);
            }
            println!("  {} {}", style("Log level:").dim(), config.logging.level);
            println!("  {} {}", style("Log format:").dim(), config.logging.format);
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
            eprintln!("{} Configuration error: {e}", style("✗").red().bold());
            std::process::exit(1);
        }
    }
}

/// Run `kiln trajectory inspect`: load a tokenizer, inspect the JSONL file
/// through kiln-train's canonical mask builder, and print the report.
pub fn run_trajectory_inspect(
    config_file: Option<&str>,
    file: &Path,
    tokenizer_path: Option<&Path>,
    chat_template_path: Option<&Path>,
    model_path: Option<&Path>,
    json: bool,
    include_context: bool,
    preview_tokens: usize,
) -> anyhow::Result<()> {
    let tokenizer = load_trajectory_inspect_tokenizer(
        config_file,
        tokenizer_path,
        chat_template_path,
        model_path,
    )?;
    let report = inspect_trajectory_file(file, &tokenizer, include_context, preview_tokens)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", format_trajectory_inspect_report(&report));
    }
    Ok(())
}

fn load_trajectory_inspect_tokenizer(
    config_file: Option<&str>,
    tokenizer_path: Option<&Path>,
    chat_template_path: Option<&Path>,
    model_path: Option<&Path>,
) -> anyhow::Result<KilnTokenizer> {
    use crate::config::KilnConfig;
    use anyhow::Context as _;

    let config = KilnConfig::load(config_file)?;
    let configured_tokenizer = config.model.tokenizer_path.as_deref().map(PathBuf::from);
    let configured_model_path = config.model.path.as_deref().map(PathBuf::from);
    let tokenizer_path = tokenizer_path.map(PathBuf::from).or(configured_tokenizer);
    let model_path = model_path.map(PathBuf::from).or(configured_model_path);

    let (mut tokenizer, template_dir) = if let Some(path) = tokenizer_path.as_deref() {
        let path_str = path
            .to_str()
            .with_context(|| format!("tokenizer path is not UTF-8: {}", path.display()))?;
        let tokenizer = KilnTokenizer::from_file(path_str)
            .with_context(|| format!("load tokenizer from {}", path.display()))?;
        (tokenizer, path.parent().map(Path::to_path_buf))
    } else if let Some(path) = model_path.as_deref() {
        let tok_file = path.join("tokenizer.json");
        if tok_file.exists() {
            let path_str = tok_file
                .to_str()
                .with_context(|| format!("tokenizer path is not UTF-8: {}", tok_file.display()))?;
            let tokenizer = KilnTokenizer::from_file(path_str)
                .with_context(|| format!("load tokenizer from {}", tok_file.display()))?;
            (tokenizer, Some(path.to_path_buf()))
        } else {
            let tokenizer = KilnTokenizer::from_pretrained(&config.model.model_id)
                .with_context(|| format!("load tokenizer for {}", config.model.model_id))?;
            (tokenizer, None)
        }
    } else {
        let tokenizer = KilnTokenizer::from_pretrained(&config.model.model_id)
            .with_context(|| format!("load tokenizer for {}", config.model.model_id))?;
        (tokenizer, None)
    };

    if let Some(path) = chat_template_path {
        let template = std::fs::read_to_string(path)
            .with_context(|| format!("read chat template {}", path.display()))?;
        tokenizer = tokenizer.with_chat_template(template);
    } else if let Some(dir) = template_dir.as_deref() {
        if let Some((_source, template)) = load_inspect_chat_template_from_model_dir(dir)
            .with_context(|| format!("load chat template from {}", dir.display()))?
        {
            tokenizer = tokenizer.with_chat_template(template);
        }
    }

    Ok(tokenizer)
}

fn load_inspect_chat_template_from_model_dir(
    dir: &Path,
) -> anyhow::Result<Option<(&'static str, String)>> {
    let standalone = dir.join("chat_template.jinja");
    if standalone.exists() {
        let template = std::fs::read_to_string(&standalone)?;
        return Ok(Some(("chat_template.jinja", template)));
    }
    let config_path = dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&config_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)?;
    Ok(parsed
        .get("chat_template")
        .and_then(|v| v.as_str())
        .map(|s| ("tokenizer_config.json", s.to_string())))
}

fn format_trajectory_inspect_report(report: &TrajectoryInspectReport) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(out, "Trajectory inspection: {}", report.path);
    let _ = writeln!(out, "  Source format: {}", report.source_format);
    let _ = writeln!(out, "  Rollouts: {}", report.rollouts.len());
    let _ = writeln!(out, "  Action tokens: {}", report.action_tokens);
    let _ = writeln!(out, "  Env tokens: {}", report.env_tokens);
    let _ = writeln!(out, "  Context tokens: {}", report.context_tokens);
    let _ = writeln!(
        out,
        "  Warning-prefix stripped bytes: {}",
        report.warning_prefix_stripped_bytes
    );

    if report.schema_warnings.is_empty() {
        let _ = writeln!(out, "  Schema warnings: none");
    } else {
        let _ = writeln!(out, "  Schema warnings:");
        for warning in &report.schema_warnings {
            let _ = writeln!(out, "    - {warning}");
        }
    }

    for rollout in &report.rollouts {
        append_rollout_inspection(&mut out, rollout);
    }

    out
}

fn append_rollout_inspection(out: &mut String, rollout: &RolloutInspection) {
    use std::fmt::Write as _;

    let _ = writeln!(out);
    let _ = writeln!(out, "Rollout {}:", rollout.index);
    let _ = writeln!(out, "  Prompt messages: {}", rollout.prompt_messages.len());
    let _ = writeln!(out, "  Action tokens: {}", rollout.action_tokens);
    let _ = writeln!(out, "  Env tokens: {}", rollout.env_tokens);
    let _ = writeln!(out, "  Context tokens: {}", rollout.context_tokens);
    let _ = writeln!(
        out,
        "  Warning-prefix stripped bytes: {}",
        rollout.warning_prefix_stripped_bytes
    );
    append_indented_block(out, "  Rendered messages:", &rollout.rendered_messages, 4);
    append_indented_block(out, "  Action preview:", &rollout.action_preview, 4);
    append_indented_block(out, "  Env preview:", &rollout.env_preview, 4);

    let _ = writeln!(out, "  Segments:");
    for segment in &rollout.segments {
        append_segment_inspection(out, segment);
    }
}

fn append_segment_inspection(out: &mut String, segment: &SegmentInspection) {
    use std::fmt::Write as _;

    let span = match (segment.token_start, segment.token_end) {
        (Some(start), Some(end)) => format!("{start}..{end}"),
        _ => "unspanned".to_string(),
    };
    let _ = writeln!(
        out,
        "    [{}] role={} kind={:?} tokens={} span={}",
        segment.index, segment.role, segment.kind, segment.token_count, span
    );
    if let Some(tool_call_id) = segment.tool_call_id.as_deref() {
        let _ = writeln!(out, "      tool_call_id: {tool_call_id}");
    }
    if let Some(len) = segment.warning_prefix_len {
        let _ = writeln!(out, "      warning_prefix_len: {len}");
    }
    if segment.warning_prefix_stripped_bytes > 0 {
        let _ = writeln!(
            out,
            "      warning-prefix stripped bytes: {}",
            segment.warning_prefix_stripped_bytes
        );
    }
    append_indented_block(out, "      content:", &segment.content, 8);
}

fn append_indented_block(out: &mut String, header: &str, body: &str, indent: usize) {
    use std::fmt::Write as _;

    let _ = writeln!(out, "{header}");
    let pad = " ".repeat(indent);
    if body.is_empty() {
        let _ = writeln!(out, "{pad}<empty>");
        return;
    }
    for line in body.lines() {
        let _ = writeln!(out, "{pad}{line}");
    }
    if body.ends_with('\n') {
        let _ = writeln!(out, "{pad}");
    }
}

/// Run the `adapters list` CLI subcommand.
pub async fn run_adapters_list(url: &str) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/v1/adapters"))
        .await
        .map_err(|e| handle_request_error(url, e))?;
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
        .await
        .map_err(|e| handle_request_error(url, e))?;
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
        .await
        .map_err(|e| handle_request_error(url, e))?;
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
        .await
        .map_err(|e| handle_request_error(url, e))?;
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

/// Run the `adapter verify` / `adapters verify` CLI subcommand.
pub async fn run_adapter_verify(
    config_file: Option<&str>,
    url: Option<&str>,
    adapter_dir: Option<&Path>,
    name_or_path: &str,
    prompt: Option<&str>,
) -> anyhow::Result<()> {
    let resolved_adapter_dir = adapter_dir
        .map(Path::to_path_buf)
        .or_else(|| adapter_dir_from_config(config_file));
    let mut receipt = verify_adapter_offline(AdapterVerifyOptions {
        input: name_or_path.to_string(),
        adapter_dir: resolved_adapter_dir,
    });

    if let Some(url) = url {
        let adapter_name = server_adapter_name(name_or_path, receipt.name.as_deref());
        let server_receipt = verify_adapter_with_server(
            url,
            &adapter_name,
            prompt.unwrap_or(DEFAULT_VERIFY_PROMPT),
            receipt.logit_delta_summary.measurable,
        )
        .await;
        receipt.server = Some(server_receipt);
    }

    finalize_status(&mut receipt);
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if receipt.status != "ok" {
        std::process::exit(1);
    }
    Ok(())
}

/// Run the `adapter restore` / `adapters restore` CLI subcommand.
pub fn run_adapter_restore(
    config_file: Option<&str>,
    manifest: &Path,
    adapter_dir: Option<&Path>,
    name: Option<&str>,
    overwrite: bool,
) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let target_adapter_dir = adapter_dir
        .map(Path::to_path_buf)
        .or_else(|| adapter_dir_from_config(config_file))
        .context(
            "could not determine adapter registry; pass --adapter-dir or use --config/KILN_MODEL_PATH",
        )?;
    let receipt = kiln_train::restore_adapter_from_manifest(kiln_train::AdapterRestoreOptions {
        manifest_path: manifest.to_path_buf(),
        adapter_dir: target_adapter_dir,
        adapter_name: name.map(str::to_string),
        overwrite,
    })?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn adapter_dir_from_config(config_file: Option<&str>) -> Option<PathBuf> {
    let config = crate::config::KilnConfig::load(config_file).ok()?;
    if let Some(adapter_dir) = config.model.adapter_dir {
        return Some(PathBuf::from(adapter_dir));
    }
    config
        .model
        .path
        .map(|model_path| PathBuf::from(model_path).join("adapters"))
}

fn server_adapter_name(name_or_path: &str, offline_name: Option<&str>) -> String {
    let path = Path::new(name_or_path);
    if path.exists()
        || path.is_absolute()
        || name_or_path.contains('/')
        || name_or_path.contains('\\')
    {
        offline_name.unwrap_or(name_or_path).to_string()
    } else {
        name_or_path.to_string()
    }
}

async fn verify_adapter_with_server(
    url: &str,
    adapter_name: &str,
    prompt: &str,
    offline_delta_measurable: bool,
) -> AdapterVerifyServerReceipt {
    let mut receipt = AdapterVerifyServerReceipt {
        url: url.to_string(),
        adapter_name: adapter_name.to_string(),
        prompt: prompt.to_string(),
        checks: Vec::new(),
        base_output: None,
        adapter_output: None,
        generated_text_different: None,
        behavior_diagnosis: None,
        behavior_note: None,
    };
    let client = reqwest::Client::new();

    match client
        .post(adapter_load_url(url))
        .json(&build_adapter_load_payload(adapter_name))
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let body = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|err| serde_json::json!({ "error": err.to_string() }));
            push_check(
                &mut receipt.checks,
                "server_load_adapter",
                status.is_success(),
                if status.is_success() {
                    format!("server loaded adapter `{adapter_name}`")
                } else {
                    render_api_error(&body, status)
                },
            );
        }
        Err(err) => {
            push_check(
                &mut receipt.checks,
                "server_load_adapter",
                false,
                format!("could not reach Kiln server at {url}: {err}"),
            );
            return receipt;
        }
    }

    match client.get(format!("{url}/v1/adapters")).send().await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp
                .json::<serde_json::Value>()
                .await
                .unwrap_or_else(|err| serde_json::json!({ "error": err.to_string() }));
            if status.is_success() {
                let loaded = registry_reports_loaded(&body, adapter_name);
                push_check(
                    &mut receipt.checks,
                    "server_registry_loaded",
                    loaded,
                    if loaded {
                        format!("/v1/adapters reports `{adapter_name}` as loaded")
                    } else {
                        format!("/v1/adapters does not report `{adapter_name}` as loaded")
                    },
                );
            } else {
                push_check(
                    &mut receipt.checks,
                    "server_registry_loaded",
                    false,
                    render_api_error(&body, status),
                );
            }
        }
        Err(err) => {
            push_check(
                &mut receipt.checks,
                "server_registry_loaded",
                false,
                format!("could not fetch /v1/adapters from {url}: {err}"),
            );
        }
    }

    let base_output = chat_verify_output(&client, url, prompt, serde_json::Value::Null).await;
    let adapter_output = chat_verify_output(
        &client,
        url,
        prompt,
        serde_json::Value::String(adapter_name.to_string()),
    )
    .await;
    match (base_output, adapter_output) {
        (Ok(base), Ok(adapter)) => {
            let changed = base != adapter;
            receipt.base_output = Some(base);
            receipt.adapter_output = Some(adapter);
            receipt.generated_text_different = Some(changed);
            if changed {
                receipt.behavior_diagnosis = Some("generated_text_changed".to_string());
            } else if offline_delta_measurable {
                receipt.behavior_diagnosis =
                    Some("measurable_adapter_delta_with_identical_greedy_text".to_string());
                receipt.behavior_note = Some(DETERMINISTIC_GREEDY_TEXT_NOTE.to_string());
            } else {
                receipt.behavior_diagnosis =
                    Some("no_generated_text_change_and_no_offline_delta".to_string());
            }
            push_check(
                &mut receipt.checks,
                "server_behavior_delta",
                changed || offline_delta_measurable,
                if changed {
                    "fixed prompt output differs between base and adapter".to_string()
                } else if offline_delta_measurable {
                    format!(
                        "fixed prompt output matched, but offline LoRA delta proxy is nonzero; {DETERMINISTIC_GREEDY_TEXT_NOTE}"
                    )
                } else {
                    "fixed prompt output did not differ between base and adapter, and offline LoRA delta proxy was not measurable".to_string()
                },
            );
        }
        (Err(err), _) => {
            push_check(
                &mut receipt.checks,
                "server_behavior_delta",
                false,
                format!("base prompt failed: {err}"),
            );
        }
        (_, Err(err)) => {
            push_check(
                &mut receipt.checks,
                "server_behavior_delta",
                false,
                format!("adapter prompt failed: {err}"),
            );
        }
    }

    receipt
}

fn registry_reports_loaded(body: &serde_json::Value, adapter_name: &str) -> bool {
    body.get("loaded_adapter").and_then(|v| v.as_str()) == Some(adapter_name)
        || body.get("active_adapter").and_then(|v| v.as_str()) == Some(adapter_name)
        || body
            .get("loaded_adapters")
            .and_then(|v| v.as_array())
            .is_some_and(|adapters| adapters.iter().any(|v| v.as_str() == Some(adapter_name)))
        || body
            .get("available_adapters")
            .and_then(|v| v.as_array())
            .is_some_and(|adapters| {
                adapters.iter().any(|entry| {
                    entry.get("name").and_then(|v| v.as_str()) == Some(adapter_name)
                        && entry.get("status").and_then(|v| v.as_str()) == Some("loaded")
                })
            })
}

async fn chat_verify_output(
    client: &reqwest::Client,
    url: &str,
    prompt: &str,
    adapter: serde_json::Value,
) -> anyhow::Result<String> {
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8,
        "seed": 7,
        "adapter": adapter,
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;
    if !status.is_success() {
        anyhow::bail!("{}", render_api_error(&body, status));
    }
    let content = body
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string();
    Ok(content)
}

/// Run the `train sft` CLI subcommand.
pub async fn run_train_sft(
    url: &str,
    file: &str,
    adapter: &str,
    lr: f64,
    epochs: u32,
    lora_rank: Option<usize>,
    adapter_smoke_test: bool,
) -> anyhow::Result<()> {
    let content =
        std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("Failed to read {file}: {e}"))?;

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

    let body =
        build_sft_training_payload(examples, adapter, lr, epochs, lora_rank, adapter_smoke_test);

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{url}/v1/train/sft"))
        .json(&body)
        .send()
        .await
        .map_err(|e| handle_request_error(url, e))?;

    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!("{} Training job submitted", style("✓").green().bold());
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
    adapter_smoke_test: bool,
) -> anyhow::Result<()> {
    let body = if is_grpo_jsonl_path(file) {
        build_grpo_jsonl_training_payload(file, adapter, lora_rank, adapter_smoke_test)?
    } else {
        let content = std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("Failed to read {file}: {e}"))?;

        let body: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Invalid JSON in {file}: {e}"))?;

        build_grpo_training_payload(body, adapter, lora_rank, adapter_smoke_test)?
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
        .await
        .map_err(|e| handle_request_error(url, e))?;

    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await?;

    if status.is_success() {
        println!("{} GRPO training job submitted", style("✓").green().bold());
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

fn is_grpo_jsonl_path(file: &str) -> bool {
    std::path::Path::new(file)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("jsonl") || ext.eq_ignore_ascii_case("ndjson"))
        .unwrap_or(false)
}

fn build_grpo_jsonl_training_payload(
    file: &str,
    adapter: &str,
    lora_rank: Option<usize>,
    adapter_smoke_test: bool,
) -> anyhow::Result<serde_json::Value> {
    let dataset_path = std::fs::canonicalize(file)
        .map_err(|e| anyhow::anyhow!("Failed to resolve GRPO JSONL file {file}: {e}"))?;
    let dataset_path = dataset_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("GRPO JSONL path is not valid UTF-8: {file}"))?
        .to_string();
    let mut config = serde_json::json!({
        "output_name": adapter,
    });
    if let Some(rank) = lora_rank {
        config["lora_rank"] = serde_json::json!(rank);
    }
    if adapter_smoke_test {
        config["adapter_smoke_test"] = serde_json::json!(true);
    }
    Ok(serde_json::json!({
        "dataset_path": dataset_path,
        "config": config,
    }))
}

fn build_sft_training_payload(
    examples: Vec<serde_json::Value>,
    adapter: &str,
    lr: f64,
    epochs: u32,
    lora_rank: Option<usize>,
    adapter_smoke_test: bool,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "output_name": adapter,
        "learning_rate": lr,
        "epochs": epochs,
    });
    if let Some(rank) = lora_rank {
        config["lora_rank"] = serde_json::json!(rank);
    }
    if adapter_smoke_test {
        config["adapter_smoke_test"] = serde_json::json!(true);
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
    adapter_smoke_test: bool,
) -> anyhow::Result<serde_json::Value> {
    let obj = body.as_object_mut().ok_or_else(|| {
        anyhow::anyhow!("GRPO request must be a JSON object with groups and config")
    })?;
    obj.remove("adapter_name");

    let config = obj.entry("config").or_insert_with(|| serde_json::json!({}));
    let config_obj = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("GRPO request config must be a JSON object"))?;
    config_obj.remove("epochs");
    config_obj.remove("num_epochs");
    config_obj.insert("output_name".into(), serde_json::json!(adapter));
    if let Some(rank) = lora_rank {
        config_obj.insert("lora_rank".into(), serde_json::json!(rank));
    }
    if adapter_smoke_test {
        config_obj.insert("adapter_smoke_test".into(), serde_json::json!(true));
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
    let resp = reqwest::get(format!("{url}/v1/train/status/{id}"))
        .await
        .map_err(|e| handle_request_error(url, e))?;
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
    let resp = reqwest::get(format!("{url}/v1/train/status"))
        .await
        .map_err(|e| handle_request_error(url, e))?;
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
        let ea = a
            .get("elapsed_secs")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let eb = b
            .get("elapsed_secs")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("{} {} job(s):", style("✓").green().bold(), jobs.len());
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
    let progress_pct =
        (job.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0).round() as i64;
    let elapsed = job
        .get("elapsed_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .round() as i64;

    println!(
        "{} Job {}",
        style("✓").green().bold(),
        style(id).white().bold()
    );
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
    let progress_pct =
        (job.get("progress").and_then(|v| v.as_f64()).unwrap_or(0.0) * 100.0).round() as i64;
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

// ===========================================================================
// §10.14 pi + kiln canonical pipeline runners
// ===========================================================================

const PI_PROVIDER_ID: &str = "kiln-local";
const PI_MODEL_ID: &str = "Qwen3.5-4B";

/// §10.14 `kiln pi-setup` — merge kiln into pi's models/settings config.
pub async fn run_pi_setup(url: &str, out: Option<&str>) -> anyhow::Result<()> {
    let default_out: PathBuf = match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h)
            .join(".pi")
            .join("agent")
            .join("models.json"),
        Err(_) => PathBuf::from("/tmp/pi-agent-models.json"),
    };
    let path = match out {
        Some(p) => PathBuf::from(p),
        None => default_out,
    };
    let settings_path = pi_settings_path_for_models_path(&path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = settings_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let models_backup = backup_existing_file(&path)?;
    let settings_backup = backup_existing_file(&settings_path)?;

    let models = merge_pi_models_config(read_json_file_if_exists(&path)?, url)?;
    let settings = merge_pi_settings_config(read_json_file_if_exists(&settings_path)?)?;
    write_json_pretty(&path, &models)?;
    write_json_pretty(&settings_path, &settings)?;

    println!(
        "{} Updated pi models.json → {}",
        style("✓").green().bold(),
        path.display()
    );
    println!(
        "{} Updated pi settings.json → {}",
        style("✓").green().bold(),
        settings_path.display()
    );
    if let Some(backup) = models_backup {
        println!("  backup: {}", backup.display());
    }
    if let Some(backup) = settings_backup {
        println!("  backup: {}", backup.display());
    }
    println!(
        "  pi now talks to {} as {}.",
        pi_openai_base_url(url),
        PI_MODEL_ID
    );
    println!(
        "  Next: {} once, then use pi normally; {} on Saturdays.",
        style("kiln judge distill").cyan(),
        style("kiln self-improve").cyan(),
    );
    Ok(())
}

fn pi_settings_path_for_models_path(models_path: &Path) -> PathBuf {
    models_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("settings.json")
}

fn read_json_file_if_exists(path: &Path) -> anyhow::Result<Option<serde_json::Value>> {
    match std::fs::read_to_string(path) {
        Ok(body) => serde_json::from_str(&body)
            .map(Some)
            .map_err(|err| anyhow::anyhow!("parse {}: {err}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err).map_err(|err| anyhow::anyhow!("read {}: {err}", path.display())),
    }
}

fn backup_existing_file(path: &Path) -> anyhow::Result<Option<PathBuf>> {
    if !path.exists() {
        return Ok(None);
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .ok_or_else(|| {
            anyhow::anyhow!("cannot back up path without file name: {}", path.display())
        })?;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| anyhow::anyhow!("system clock before unix epoch: {err}"))?
        .as_secs();

    for suffix in 0..1000 {
        let backup_name = if suffix == 0 {
            format!("{file_name}.bak-{timestamp}")
        } else {
            format!("{file_name}.bak-{timestamp}-{suffix}")
        };
        let backup = parent.join(backup_name);
        if !backup.exists() {
            std::fs::copy(path, &backup).map_err(|err| {
                anyhow::anyhow!("backup {} to {}: {err}", path.display(), backup.display())
            })?;
            return Ok(Some(backup));
        }
    }

    Err(anyhow::anyhow!(
        "could not find an unused backup path for {}",
        path.display()
    ))
}

fn write_json_pretty(path: &Path, value: &serde_json::Value) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes).map_err(|err| anyhow::anyhow!("write {}: {err}", path.display()))
}

fn merge_pi_models_config(
    existing: Option<serde_json::Value>,
    url: &str,
) -> anyhow::Result<serde_json::Value> {
    let mut root = match existing {
        Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
        Some(_) => {
            return Err(anyhow::anyhow!(
                "models.json root must be a JSON object; existing file was backed up"
            ));
        }
        None => serde_json::json!({}),
    };

    let root_obj = root
        .as_object_mut()
        .expect("root was constructed as a JSON object");
    let providers_value = root_obj
        .remove("providers")
        .unwrap_or_else(|| serde_json::json!({}));
    let mut providers = pi_providers_object_from_value(providers_value);
    providers.insert(PI_PROVIDER_ID.to_string(), kiln_pi_provider_config(url));
    root_obj.insert(
        "providers".to_string(),
        serde_json::Value::Object(providers),
    );
    Ok(root)
}

fn merge_pi_settings_config(
    existing: Option<serde_json::Value>,
) -> anyhow::Result<serde_json::Value> {
    let mut root = match existing {
        Some(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
        Some(_) => {
            return Err(anyhow::anyhow!(
                "settings.json root must be a JSON object; existing file was backed up"
            ));
        }
        None => serde_json::json!({}),
    };
    let root_obj = root
        .as_object_mut()
        .expect("root was constructed as a JSON object");
    root_obj.insert(
        "defaultProvider".to_string(),
        serde_json::json!(PI_PROVIDER_ID),
    );
    root_obj.insert("defaultModel".to_string(), serde_json::json!(PI_MODEL_ID));
    Ok(root)
}

fn pi_providers_object_from_value(
    value: serde_json::Value,
) -> serde_json::Map<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => map,
        serde_json::Value::Array(items) => {
            let mut map = serde_json::Map::new();
            for (index, item) in items.into_iter().enumerate() {
                let serde_json::Value::Object(provider) = item else {
                    continue;
                };
                let key = provider
                    .get("name")
                    .or_else(|| provider.get("id"))
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("provider-{index}"));
                map.insert(key, serde_json::Value::Object(provider));
            }
            map
        }
        _ => serde_json::Map::new(),
    }
}

fn kiln_pi_provider_config(url: &str) -> serde_json::Value {
    serde_json::json!({
        "baseUrl": pi_openai_base_url(url),
        "api": "openai-completions",
        "apiKey": "dummy",
        "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
        },
        "models": [{
            "id": PI_MODEL_ID,
            "name": "Qwen 3.5 4B via Kiln",
            "input": ["text"],
            "contextWindow": 262144,
            "maxTokens": 32768,
        }],
    })
}

fn pi_openai_base_url(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

/// §10.6 `kiln judge ...` dispatcher.
pub async fn run_judge(cmd: &JudgeCommands) -> anyhow::Result<()> {
    match cmd {
        JudgeCommands::Distill { url, name, teacher } => {
            let body = serde_json::json!({
                "name": name,
                "teacher": teacher
            });
            let resp = reqwest::Client::new()
                .post(format!("{url}/v1/agent/judge_distill"))
                .json(&body)
                .send()
                .await
                .map_err(|e| handle_request_error(url, e))?;
            print_simple_json_response("judge distill", resp).await
        }
        JudgeCommands::DriftCheck {
            url,
            judge,
            teacher,
        } => {
            let body = serde_json::json!({
                "judge": judge,
                "teacher": teacher
            });
            let resp = reqwest::Client::new()
                .post(format!("{url}/v1/agent/judge_drift_check"))
                .json(&body)
                .send()
                .await
                .map_err(|e| handle_request_error(url, e))?;
            print_simple_json_response("judge drift-check", resp).await
        }
    }
}

/// §10.6.2 / §10.14 `kiln self-improve` — POST the weekly self-
/// improvement loop request.
pub async fn run_self_improve(
    url: &str,
    agent: &str,
    judge: &str,
    crisp: bool,
) -> anyhow::Result<()> {
    let body = serde_json::json!({
        "agent": agent,
        "judge": judge,
        "crisp": crisp
    });
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/agent/self_improve"))
        .json(&body)
        .send()
        .await
        .map_err(|e| handle_request_error(url, e))?;
    print_simple_json_response("self-improve", resp).await
}

/// Run the `kiln eval-adapter` CLI subcommand.
pub async fn run_eval_adapter(
    url: &str,
    adapter: &str,
    tasks: &Path,
    seeds: usize,
    request_template: &Path,
    scorer: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    let summary = crate::eval_adapter_cli::run_eval_adapter(
        crate::eval_adapter_cli::EvalAdapterOptions {
            url: url.to_string(),
            adapter: adapter.to_string(),
            tasks: tasks.to_path_buf(),
            seeds,
            request_template: request_template.to_path_buf(),
            scorer: scorer.to_path_buf(),
            output: output.to_path_buf(),
        },
    )
    .await?;

    println!(
        "{} eval-adapter completed: {} pair(s), mean lift {:.6}, stdev {:.6}, zero_count {}, wrote {}",
        style("✓").green().bold(),
        summary.pair_count,
        summary.stats.mean_lift,
        summary.stats.stdev_lift,
        summary.stats.zero_count,
        output.display()
    );
    for warning in &summary.warnings {
        eprintln!("{} {warning}", style("warning:").yellow().bold());
    }
    Ok(())
}

/// Run the `kiln rollout-generate` CLI subcommand.
#[allow(clippy::too_many_arguments)]
pub async fn run_rollout_generate(
    url: &str,
    adapter: &str,
    thinking: bool,
    tasks: &Path,
    seeds: usize,
    seed_start: u64,
    request_template: &Path,
    scorer: &Path,
    output: &Path,
    summary_output: &Path,
) -> anyhow::Result<()> {
    let summary = crate::rollout_generate_cli::run_rollout_generate(
        crate::rollout_generate_cli::RolloutGenerateOptions {
            url: url.to_string(),
            adapter: adapter.to_string(),
            thinking,
            tasks: tasks.to_path_buf(),
            seeds,
            seed_start,
            request_template: request_template.to_path_buf(),
            scorer: scorer.to_path_buf(),
            output: output.to_path_buf(),
            summary_output: summary_output.to_path_buf(),
        },
    )
    .await?;

    println!(
        "{} rollout-generate completed: {} group(s), {} completion(s), mean reward {:.6}, total tokens {}, wrote {}",
        style("✓").green().bold(),
        summary.group_count,
        summary.completion_count,
        summary.stats.mean_reward,
        summary.stats.total_tokens,
        output.display()
    );
    println!("summary: {}", summary_output.display());
    for warning in &summary.warnings {
        eprintln!("{} {warning}", style("warning:").yellow().bold());
    }
    Ok(())
}

async fn print_simple_json_response(label: &str, resp: reqwest::Response) -> anyhow::Result<()> {
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;
    if status.is_success() {
        println!("{} {label}", style("✓").green().bold());
        println!("{}", serde_json::to_string_pretty(&body)?);
        Ok(())
    } else {
        eprintln!("{} {label} failed ({})", style("✗").red().bold(), status);
        eprintln!("{}", serde_json::to_string_pretty(&body)?);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;
    use serde_json::json;

    fn cli_with(verbose: u8, quiet: bool) -> Cli {
        Cli {
            command: None,
            config: None,
            verbose,
            quiet,
        }
    }

    #[test]
    fn effective_log_level_returns_fallback_with_no_flags() {
        assert_eq!(cli_with(0, false).effective_log_level("info"), "info");
        assert_eq!(
            cli_with(0, false).effective_log_level("kiln=info,tower_http=warn"),
            "kiln=info,tower_http=warn"
        );
    }

    #[test]
    fn effective_log_level_v_promotes_to_debug() {
        assert_eq!(cli_with(1, false).effective_log_level("info"), "debug");
    }

    #[test]
    fn effective_log_level_vv_promotes_to_trace() {
        assert_eq!(cli_with(2, false).effective_log_level("info"), "trace");
        assert_eq!(cli_with(3, false).effective_log_level("info"), "trace");
    }

    #[test]
    fn effective_log_level_quiet_overrides_verbose_fallback() {
        assert_eq!(cli_with(0, true).effective_log_level("info"), "warn");
        // quiet wins regardless of fallback severity
        assert_eq!(cli_with(0, true).effective_log_level("trace"), "warn");
    }

    #[test]
    fn parses_serve_eval_mode() {
        let cli = Cli::parse_from(["kiln", "serve", "--eval-mode", "--served-model-id", "eval"]);
        match cli.command {
            Some(Commands::Serve {
                served_model_id,
                eval_mode,
            }) => {
                assert_eq!(served_model_id.as_deref(), Some("eval"));
                assert!(eval_mode);
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn parses_eval_adapter_command() {
        let cli = Cli::parse_from([
            "kiln",
            "eval-adapter",
            "--adapter",
            "support-bot",
            "--tasks",
            "eval.tasks.jsonl",
            "--seeds",
            "3",
            "--request-template",
            "request.json",
            "--scorer",
            "./score_one.py",
        ]);

        let Some(Commands::EvalAdapter {
            adapter,
            tasks,
            seeds,
            request_template,
            scorer,
            output,
            url,
        }) = cli.command
        else {
            panic!("expected eval-adapter command");
        };

        assert_eq!(adapter, "support-bot");
        assert_eq!(tasks, PathBuf::from("eval.tasks.jsonl"));
        assert_eq!(seeds, 3);
        assert_eq!(request_template, PathBuf::from("request.json"));
        assert_eq!(scorer, PathBuf::from("./score_one.py"));
        assert_eq!(output, PathBuf::from("eval_summary.json"));
        assert_eq!(url, "http://localhost:8420");
    }

    #[test]
    fn parses_rollout_generate_command() {
        let cli = Cli::parse_from([
            "kiln",
            "rollout-generate",
            "--adapter",
            "support-bot",
            "--thinking",
            "false",
            "--tasks",
            "tasks.jsonl",
            "--seeds",
            "3",
            "--seed-start",
            "42",
            "--request-template",
            "request.json",
            "--scorer",
            "./score_one.py",
            "--output",
            "rollouts.jsonl",
            "--summary-output",
            "summary.json",
        ]);

        let Some(Commands::RolloutGenerate {
            adapter,
            thinking,
            tasks,
            seeds,
            seed_start,
            request_template,
            scorer,
            output,
            summary_output,
            url,
        }) = cli.command
        else {
            panic!("expected rollout-generate command");
        };

        assert_eq!(adapter, "support-bot");
        assert!(!thinking);
        assert_eq!(tasks, PathBuf::from("tasks.jsonl"));
        assert_eq!(seeds, 3);
        assert_eq!(seed_start, 42);
        assert_eq!(request_template, PathBuf::from("request.json"));
        assert_eq!(scorer, PathBuf::from("./score_one.py"));
        assert_eq!(output, PathBuf::from("rollouts.jsonl"));
        assert_eq!(summary_output, PathBuf::from("summary.json"));
        assert_eq!(url, "http://localhost:8420");
    }

    #[test]
    fn parses_trajectory_inspect_command() {
        let cli = Cli::parse_from([
            "kiln",
            "--config",
            "kiln.toml",
            "trajectory",
            "inspect",
            "session.jsonl",
            "--json",
            "--include-context",
            "--preview-tokens",
            "12",
            "--tokenizer",
            "tokenizer.json",
            "--chat-template",
            "chat_template.jinja",
            "--model-path",
            "/models/Qwen3.5-4B",
        ]);

        assert_eq!(cli.config.as_deref(), Some("kiln.toml"));
        let Some(Commands::Trajectory(TrajectoryCommands::Inspect {
            file,
            json,
            include_context,
            preview_tokens,
            tokenizer,
            chat_template,
            model_path,
        })) = cli.command
        else {
            panic!("expected trajectory inspect command");
        };

        assert_eq!(file, PathBuf::from("session.jsonl"));
        assert!(json);
        assert!(include_context);
        assert_eq!(preview_tokens, 12);
        assert_eq!(tokenizer.as_deref(), Some(Path::new("tokenizer.json")));
        assert_eq!(
            chat_template.as_deref(),
            Some(Path::new("chat_template.jinja"))
        );
        assert_eq!(model_path.as_deref(), Some(Path::new("/models/Qwen3.5-4B")));
    }

    #[test]
    fn trajectory_report_formatter_includes_mask_fields() {
        let report = TrajectoryInspectReport {
            path: "rollouts.jsonl".to_string(),
            source_format: "kiln_rollout_jsonl".to_string(),
            rollouts: vec![RolloutInspection {
                index: 0,
                prompt_messages: vec![kiln_train::ChatMessage {
                    role: "user".to_string(),
                    content: "run it".to_string(),
                }],
                rendered_messages: "<|im_start|>assistant\ncall".to_string(),
                segments: vec![SegmentInspection {
                    index: 0,
                    role: "assistant".to_string(),
                    kind: kiln_train::trajectory::TurnKind::Action,
                    content: "call".to_string(),
                    token_start: Some(2),
                    token_end: Some(6),
                    token_count: 4,
                    warning_prefix_len: None,
                    warning_prefix_stripped_bytes: 0,
                    tool_call_id: Some("tool-1".to_string()),
                }],
                action_tokens: 4,
                env_tokens: 2,
                context_tokens: 8,
                warning_prefix_stripped_bytes: 3,
                action_preview: "call".to_string(),
                env_preview: "ok".to_string(),
            }],
            action_tokens: 4,
            env_tokens: 2,
            context_tokens: 8,
            warning_prefix_stripped_bytes: 3,
            schema_warnings: vec![
                "legacy text-only ScoredRollout synthesized one Action segment".to_string(),
            ],
        };

        let rendered = format_trajectory_inspect_report(&report);

        assert!(rendered.contains("Rendered messages:"));
        assert!(rendered.contains("role=assistant kind=Action tokens=4 span=2..6"));
        assert!(rendered.contains("Action tokens: 4"));
        assert!(rendered.contains("Env tokens: 2"));
        assert!(rendered.contains("Warning-prefix stripped bytes: 3"));
        assert!(rendered.contains("Action preview:"));
        assert!(rendered.contains("Env preview:"));
        assert!(rendered.contains("Schema warnings:"));
    }

    fn read_json(path: &std::path::Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn backup_count(dir: &std::path::Path, prefix: &str) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(&format!("{prefix}.bak-"))
            })
            .count()
    }

    #[tokio::test]
    async fn pi_setup_merges_models_and_settings_with_backups() {
        let tmp = tempfile::tempdir().unwrap();
        let models_path = tmp.path().join("models.json");
        let settings_path = tmp.path().join("settings.json");
        std::fs::write(
            &models_path,
            serde_json::to_vec_pretty(&json!({
                "providers": {
                    "other-provider": {
                        "baseUrl": "https://example.invalid/v1",
                        "api": "openai"
                    }
                },
                "unrelatedTopLevel": true
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            &settings_path,
            serde_json::to_vec_pretty(&json!({
                "lastChangelogVersion": "0.75.0",
                "theme": "quiet"
            }))
            .unwrap(),
        )
        .unwrap();

        run_pi_setup("http://localhost:8420", Some(models_path.to_str().unwrap()))
            .await
            .unwrap();

        let models = read_json(&models_path);
        assert_eq!(models["unrelatedTopLevel"], true);
        assert_eq!(
            models["providers"]["other-provider"]["baseUrl"],
            "https://example.invalid/v1"
        );
        let kiln = &models["providers"][PI_PROVIDER_ID];
        assert_eq!(kiln["baseUrl"], "http://localhost:8420/v1");
        assert_eq!(kiln["api"], "openai-completions");
        assert_eq!(kiln["apiKey"], "dummy");
        assert_eq!(kiln["compat"]["supportsDeveloperRole"], false);
        assert_eq!(kiln["compat"]["supportsReasoningEffort"], false);
        assert_eq!(kiln["models"][0]["id"], PI_MODEL_ID);
        assert_eq!(kiln["models"][0]["name"], "Qwen 3.5 4B via Kiln");
        assert_eq!(kiln["models"][0]["input"], json!(["text"]));
        assert_eq!(kiln["models"][0]["contextWindow"], 262144);
        assert_eq!(kiln["models"][0]["maxTokens"], 32768);

        let settings = read_json(&settings_path);
        assert_eq!(settings["lastChangelogVersion"], "0.75.0");
        assert_eq!(settings["theme"], "quiet");
        assert_eq!(settings["defaultProvider"], PI_PROVIDER_ID);
        assert_eq!(settings["defaultModel"], PI_MODEL_ID);
        assert_eq!(backup_count(tmp.path(), "models.json"), 1);
        assert_eq!(backup_count(tmp.path(), "settings.json"), 1);
    }

    #[tokio::test]
    async fn pi_setup_repairs_legacy_provider_array_without_dropping_named_providers() {
        let tmp = tempfile::tempdir().unwrap();
        let models_path = tmp.path().join("models.json");
        let settings_path = tmp.path().join("settings.json");
        std::fs::write(
            &models_path,
            serde_json::to_vec_pretty(&json!({
                "providers": [
                    {
                        "name": "other-provider",
                        "baseUrl": "https://example.invalid/v1",
                        "api": "openai"
                    },
                    {
                        "name": "kiln-local",
                        "base_url": "http://bad.invalid",
                        "api_key": "wrong"
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();

        run_pi_setup(
            "http://office-kiln:8420/v1/",
            Some(models_path.to_str().unwrap()),
        )
        .await
        .unwrap();

        let models = read_json(&models_path);
        let providers = models["providers"].as_object().unwrap();
        assert!(providers.contains_key("other-provider"));
        assert_eq!(
            providers["other-provider"]["baseUrl"],
            "https://example.invalid/v1"
        );
        assert_eq!(
            providers[PI_PROVIDER_ID]["baseUrl"],
            "http://office-kiln:8420/v1"
        );
        assert!(providers[PI_PROVIDER_ID].get("base_url").is_none());
        assert!(providers[PI_PROVIDER_ID].get("api_key").is_none());

        let settings = read_json(&settings_path);
        assert_eq!(settings["defaultProvider"], PI_PROVIDER_ID);
        assert_eq!(settings["defaultModel"], PI_MODEL_ID);
        assert_eq!(backup_count(tmp.path(), "models.json"), 1);
        assert_eq!(backup_count(tmp.path(), "settings.json"), 0);
    }

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
            false,
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
        let body = build_sft_training_payload(vec![], "sft-adapter", 1e-4, 1, None, false);

        assert_eq!(body["config"]["output_name"], "sft-adapter");
        assert!(body["config"].get("lora_rank").is_none());
        assert!(body["config"].get("adapter_smoke_test").is_none());
    }

    #[test]
    fn build_sft_training_payload_sets_adapter_smoke_test_when_requested() {
        let body = build_sft_training_payload(vec![], "sft-adapter", 1e-4, 1, None, true);

        assert_eq!(body["config"]["adapter_smoke_test"], true);
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

        let body = build_grpo_training_payload(body, "grpo-adapter", Some(16), true).unwrap();

        assert_eq!(body["config"]["output_name"], "grpo-adapter");
        assert_eq!(body["config"]["learning_rate"], 5e-5);
        assert_eq!(body["config"]["lora_rank"], 16);
        assert_eq!(body["config"]["adapter_smoke_test"], true);
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

        let body = build_grpo_training_payload(body, "grpo-adapter", None, false).unwrap();

        assert_eq!(body["config"]["output_name"], "grpo-adapter");
        assert!(body["config"].get("lora_rank").is_none());
        assert!(body["config"].get("adapter_smoke_test").is_none());
    }

    #[test]
    fn build_grpo_jsonl_training_payload_uses_dataset_path() {
        let path =
            std::env::temp_dir().join(format!("kiln-cli-grpo-jsonl-{}.jsonl", std::process::id()));
        std::fs::write(
            &path,
            r#"{"messages":[{"role":"user","content":"hi"}],"completions":[{"text":"ok","reward":1.0}]}"#,
        )
        .unwrap();

        assert!(is_grpo_jsonl_path(path.to_str().unwrap()));
        let body =
            build_grpo_jsonl_training_payload(path.to_str().unwrap(), "grpo-jsonl", Some(12), true)
                .unwrap();
        assert!(body.get("groups").is_none());
        assert_eq!(body["config"]["output_name"], "grpo-jsonl");
        assert_eq!(body["config"]["lora_rank"], 12);
        assert_eq!(body["config"]["adapter_smoke_test"], true);
        assert!(body["dataset_path"].as_str().unwrap().ends_with(".jsonl"));
        let _ = std::fs::remove_file(path);
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
        assert_eq!(
            build_adapter_load_payload("support-bot"),
            json!({ "name": "support-bot" })
        );
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
    fn cli_parses_singular_adapter_verify() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "adapter",
            "verify",
            "support-bot",
            "--adapter-dir",
            "/tmp/adapters",
        ])
        .expect("parse failed");
        match cli.command {
            Some(Commands::Adapters(AdapterCommands::Verify {
                name_or_path,
                adapter_dir,
                url,
                prompt,
            })) => {
                assert_eq!(name_or_path, "support-bot");
                assert_eq!(
                    adapter_dir.unwrap(),
                    std::path::PathBuf::from("/tmp/adapters")
                );
                assert_eq!(url, None);
                assert_eq!(prompt, None);
            }
            other => panic!("expected adapter verify, got {:?}", other.is_some()),
        }
    }

    #[tokio::test]
    async fn handle_request_error_classifies_unreachable_server() {
        // Bind a TCP listener on an OS-assigned port, capture the port, then
        // drop the listener so that subsequent connects to the same port get
        // ECONNREFUSED. This is more reliable than picking a "probably closed"
        // high port.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
        let port = listener.local_addr().expect("local_addr").port();
        drop(listener);
        let url = format!("http://127.0.0.1:{port}");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("build reqwest client");
        let err = client
            .get(format!("{url}/health"))
            .send()
            .await
            .expect_err("expected connect error against closed loopback port");

        assert!(
            is_connection_error(&err),
            "expected is_connection_error=true (is_connect={}, is_timeout={}); err={err:?}",
            err.is_connect(),
            err.is_timeout(),
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
            "model": "Qwen3.5-4B (32L, 16H, 4KV)",
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
        assert!(out.contains("Qwen3.5-4B"), "got: {out}");
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
        assert!(
            out.contains("✓"),
            "expected at least one ✓ glyph, got: {out}"
        );
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
            "model": "Qwen3.5-4B",
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
            "model": "Qwen3.5-4B",
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
