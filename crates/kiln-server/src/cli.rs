//! CLI interface for kiln — structured subcommands with clap.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use console::style;
pub use kiln_core::thinking_budget::ExplicitThinkingBudget as ThinkingBudgetArg;
use kiln_core::tokenizer::KilnTokenizer;
use kiln_train::trajectory_inspect::{
    RolloutInspection, SegmentInspection, TrajectoryInspectReport, inspect_trajectory_file,
};

use crate::adapter_verify::{
    AdapterVerifyOptions, AdapterVerifyServerReceipt, DEFAULT_VERIFY_PROMPT,
    DETERMINISTIC_GREEDY_TEXT_NOTE, finalize_status, push_check, verify_adapter_offline,
};
use crate::config::default_server_url;

const TOP_LEVEL_OVERVIEW: &str = r#"Kiln serves Qwen3.5-4B from one Rust process and lets you adapt it with live LoRA training.

Running `kiln` with no subcommand starts the OpenAI-compatible server, just like `kiln serve`. Commands such as `kiln health`, `kiln train sft`, `kiln train grpo`, `kiln train opd`, and `kiln adapters list` talk to a running server.

After `kiln serve`, open http://127.0.0.1:8420/ui/ for the embedded dashboard: status, adapters, training monitoring, and quick inference.

Common next steps:
  kiln serve          start the server explicitly
  kiln health         inspect a running server
  kiln train sft      train a LoRA adapter from corrections
  kiln train grpo     train a LoRA adapter from scored completions
  kiln train opd      distill a LoRA adapter from a registered teacher
  kiln train hf       create or manage a verified HF/TRL handoff
  kiln adapters list  list saved adapters and show which one is active
"#;

const TOP_LEVEL_EXAMPLES: &str = r#"Examples:
  kiln serve
      Start the inference server explicitly. Running `kiln` with no subcommand also starts serving.

      Then open http://127.0.0.1:8420/ui/ for status, adapters, training monitoring, and quick inference.

  kiln health
      Check whether the local server is ready and show model, adapter, scheduler, and training status.

  kiln train sft --file examples.jsonl --adapter my-task
      Teach the model from corrected chat examples and hot-swap the trained LoRA adapter.

  kiln train grpo --file grpo-batch.json --adapter my-task
      Improve an adapter from scored completions using GRPO rewards.

  kiln train opd --file opd-request.json --adapter distilled-task --teacher teacher-v1
      Distill an adapter from a registered teacher with exact resume points.

  kiln train hf export-sft --file /data/examples.jsonl --name my-task-hf
      Create, download, and verify an immutable bundle for the pinned HF/TRL runner.

  kiln train hf export-grpo --file /data/recorded.jsonl --name my-task-grpo
      Create the same verified handoff from provenance-complete recorded GRPO JSONL.

  kiln train hf import-peft --bundle ./my-task-hf.kiln-hf --name my-task
      Verify a completed external-training bundle and stream its PEFT adapter into Kiln.

  kiln adapters list
      Show saved adapters and which adapter is active on the running server.
"#;

const SERVE_OVERVIEW: &str = r#"Start the OpenAI-compatible Kiln server for Qwen3.5-4B on http://127.0.0.1:8420 by default.

Before starting, point Kiln at model weights with KILN_MODEL_PATH or pass a TOML config with --config. After startup, open http://127.0.0.1:8420/ui/ for the embedded dashboard and use kiln health to check readiness.

If setup stalls, follow https://ericflo.github.io/kiln/quickstart.html first, then https://ericflo.github.io/kiln/troubleshooting.html for model path, CUDA, and config checks.
"#;

const SERVE_EXAMPLES: &str = r#"Examples:
  KILN_MODEL_PATH=/models/Qwen3.5-4B kiln serve
      Start the local server with model weights from KILN_MODEL_PATH.

  kiln serve --eval-mode
      Start with deterministic eval defaults, no-thinking chat-template defaults, adapter headers, and per-request transient cache cleanup.

  KILN_SERVER_DEFAULT_THINKING_ENABLED=false kiln serve
      Default Qwen/DeepSeek-style chat templates to non-thinking mode unless a request overrides chat_template_kwargs.enable_thinking.

  KILN_SERVER_FOLD_REASONING_INTO_CONTENT=true kiln serve
      Duplicate separated reasoning into assistant content for compatibility with clients that treat empty content as no response.

  kiln serve --config kiln.toml
      Start with a checked TOML config. Run `kiln config --file kiln.toml` first if you want to preview the effective settings.

  open http://127.0.0.1:8420/ui/
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

const TRAIN_OVERVIEW: &str = r#"Submit SFT, GRPO, or OPD training jobs to the running Kiln server at http://localhost:8420 by default.

SFT reads JSONL: one chat correction example per line with a messages array. GRPO reads either one JSON request/batch with groups or JSONL with one group per line; each group has prompt messages plus candidate completions containing text and reward scores. OPD reads one JSON API request or a JSON array of prompts and requires a registered teacher alias.

Add --adapter-smoke-test to record a small base-vs-adapter canary check in train_receipt.json after successful training.

Use `kiln train hf export-sft` or `kiln train hf export-grpo` to create, download, and verify an immutable handoff bundle for the pinned Hugging Face TRL/PEFT correctness runner. Export source paths are read by the server process, not uploaded by the CLI. After external training, use `kiln train hf import-peft` to verify the completed local bundle and stream only its model-identity and PEFT result envelope back to the server.

Prefer http://127.0.0.1:8420/ui/ for guided submission and status. See docs/GRPO_GUIDE.md or docs/site/grpo.html for reward-loop examples.
"#;

const TRAIN_SFT_OVERVIEW: &str = r#"Train from SFT JSONL: one chat correction example per line with a messages array.

Native SFT uses the fixed native_online_lora_v1 online-LoRA profile: one conversation and one optimizer update at a time, constant learning rate, and no accumulation, warmup, decay, or gradient clipping. Use `kiln train hf export-sft` for an immutable bundle containing the pinned HF/TRL correctness runner when general trainer configuration is required.

Rows fail closed by default. Use --invalid-row-policy skip only when you intend to review the stable kept/rejected row hashes in train_receipt.json.

Use --adapter-smoke-test to compare base vs trained adapter logits and short greedy outputs before running a full eval.

Use --checkpoint-interval N to emit exact resumable checkpoints every N optimizer steps. Resume with the same file and configuration plus --resume-checkpoint BASENAME.

Open http://127.0.0.1:8420/ui/ for guided submission and training status.
"#;

const TRAIN_GRPO_OVERVIEW: &str = r#"Train from GRPO data: either one JSON request/batch with groups, or JSONL with one group per line.

Use `kiln train hf export-grpo` for a provenance-complete recorded-rollout corpus that should run through the pinned external TRL/PEFT route instead of the bounded native trainer.

Use --adapter-smoke-test to compare base vs trained adapter logits and short greedy outputs before running a full eval.

Use --checkpoint-interval N to emit exact resumable checkpoints every N optimizer groups. Resume with the same file and configuration plus --resume-checkpoint BASENAME.

Open http://127.0.0.1:8420/ui/ for guided submission and training status. See docs/GRPO_GUIDE.md or docs/site/grpo.html for reward-loop examples.
"#;

const TRAIN_OPD_OVERVIEW: &str = r#"Train with on-policy or off-policy distillation from a registered teacher.

The input is either one /v1/train/opd JSON request object or a JSON array of prompts. Use --teacher to supply or override the request's teacher alias; --adapter always sets config.output_name.

OPD publishes exact resumable checkpoints every 25 committed optimizer steps by default. Use --checkpoint-interval N to change the cadence, and resume with the same input/effective configuration plus --resume-checkpoint BASENAME from `kiln train status --job-id ID`.

Open http://127.0.0.1:8420/ui/ for guided submission and training status.
"#;

const TRAIN_EXAMPLES: &str = r#"Examples:
  kiln train sft --file corrections.jsonl --adapter support-bot
      Train from SFT JSONL: one chat correction example per line with a messages array.

  kiln train sft --file corrections.jsonl --adapter support-bot --adapter-smoke-test
      Train and record adapter-effect smoke metrics in train_receipt.json.

  kiln train grpo --file grpo-batch.json --adapter support-bot
      Train from one GRPO JSON request/batch with groups.

  kiln train grpo --file grpo-groups.jsonl --adapter support-bot
      Train from GRPO JSONL with one group per line without retaining the full dataset in memory.

  kiln train grpo --file grpo-groups.jsonl --adapter support-bot --checkpoint-interval 25
      Publish an exact immutable resume point every 25 optimizer groups.

  kiln train opd --file opd-request.json --adapter distilled-bot --teacher teacher-v1
      Submit an OPD request while setting its output adapter and teacher alias.

  kiln train opd --file opd-request.json --adapter distilled-bot --teacher teacher-v1 --resume-checkpoint distilled-bot-checkpoint-step-00000025.kiln-checkpoint
      Resume from the exact immutable OPD checkpoint reported by job status.

  kiln train hf export-sft --file /data/corrections.jsonl --name support-hf-01
      Export a server-local SFT corpus, download and fully verify its immutable HF/TRL bundle, then remove the server copy.

  kiln train hf export-grpo --file /data/recorded-rollouts.jsonl --name support-grpo-01
      Export canonical provenance-complete GRPO JSONL through the same verified handoff lifecycle.

  kiln train hf list
      List immutable HF/TRL exports still retained by the running server.

  kiln train status
      Show the training queue and recent jobs on the running server.

  kiln train status --job-id train_123
      Inspect one training job by ID.

  kiln train cancel --job-id train_123
      Cancel a job: queued jobs leave the queue; running jobs stop at the
      next step boundary.
"#;

const TRAIN_HF_OVERVIEW: &str = r#"Create and manage immutable Hugging Face TRL/PEFT handoff bundles.

`export-sft` asks the running server to snapshot a canonical SFT corpus and optional input adapter, streams the resulting tar.gz into a sibling temporary file, verifies its exact archive shape and every manifest-bound byte, and only then publishes the requested local output without overwriting an existing path.

`export-grpo` applies the same transport and publication workflow to server-local canonical JSONL containing exact recorded rollout provenance. The server validates resident model, base-shard, tokenizer, template, optional adapter, behavior-policy, token, and mask identities before publication.

`import-peft` takes the completed extracted `.kiln-hf` directory after external training. It verifies the unchanged export plus the result manifest and PEFT files locally, derives the expected server receipt, streams a minimal bounded envelope, and requires exact response identity agreement. It never modifies or removes the completed source bundle.

The `--file` path is server-local: the running Kiln process must be able to read it. Use `--dataset corrections:active` to snapshot active corrections or another named server dataset. Successful local verification removes the server copy by default; pass `--keep-server-copy` when another client still needs it.
"#;

const TRAIN_HF_EXPORT_SFT_OVERVIEW: &str = r#"Create and download a verified immutable SFT handoff bundle.

Exactly one of `--file` or `--dataset` is required. `--file` names JSONL visible to the server process; it is not uploaded from the CLI machine. `--dataset` names a server-side eval dataset, including the special `corrections:active` snapshot.

The output defaults to `{name}.kiln-hf.tar.gz` in the current directory. Existing output paths are never replaced. The CLI refuses redirects, streams rather than buffering the archive, rejects links and unsafe tar paths, requires exactly one `{name}.kiln-hf` root, and verifies the pristine Kiln manifest before atomic publication.
"#;

const TRAIN_HF_EXPORT_GRPO_OVERVIEW: &str = r#"Create and download a verified immutable recorded-GRPO handoff bundle.

`--file` names canonical compact JSONL visible to the server process; it is not uploaded from the CLI machine. Every LF-terminated group must contain a uniform number of scored completions with exact recorded rollout provenance matching the resident model, base shards, tokenizer, inference template, and optional input adapter.

The output defaults to `{name}.kiln-hf.tar.gz` in the current directory. Existing outputs are never replaced. Creation, bounded streaming, ETag binding, strict archive verification, atomic local publication, and identity-conditional server cleanup use the same implementation as `export-sft`.
"#;

const TRAIN_HF_IMPORT_PEFT_OVERVIEW: &str = r#"Verify and import a completed external-training PEFT result.

`--bundle` is the extracted `.kiln-hf` directory after the embedded runner has completed successfully. The original export must remain byte-identical and exactly the executed script, adapter config, adapter weights, and self-verifying result manifest must have been added.

The CLI verifies the complete local directory before connecting, derives a ten-file corpus-free envelope in private staging, and streams a deterministic gzip tar through a bounded channel. The server must return HTTP 201 with the exact locally predicted import digest, content revision, installed size, file count, task, and strong ETag. The source bundle is retained on every outcome.
"#;

const ADAPTERS_OVERVIEW: &str = r#"Inspect and manage LoRA adapters on the running Kiln server at http://localhost:8420 by default.

Most commands call the adapter API after `kiln serve` is running. `kiln adapter verify` can validate local adapter directories offline, and `kiln adapters restore` copies a manifest-described adapter into a local registry.

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

  kiln adapters restore ./runs/grpo/support-bot/adapter_manifest.json --adapter-dir ./Qwen3.5-4B/adapters
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
one choice, exact rollout provenance, performance metadata, and
`chat_template_kwargs.enable_thinking`. The server must use real model weights
with the batching engine enabled; mock, streaming, and trace-free generation
paths cannot produce the behavior-policy record this command requires. Tool
definitions and tool-call messages are rejected until the scored training
schema can represent them exactly.

`--thinking-budget-tokens` and `--thinking-budget-ms` override matching fields
from the request template. Omit a flag to preserve the template or server
default, pass a non-negative integer for a limit (`0` closes thinking
immediately), or pass `unlimited` to send an explicit JSON null and bypass a
server default. Budget flags and top-level template budget fields require
`--thinking true`; the command rejects them while thinking is disabled instead
of silently accepting controls that cannot apply. The requested states are
recorded in the rollout summary.

The scorer executable receives one JSON object on stdin with the task, request,
response, content, seed, adapter, token usage, and latency. It may print a
single numeric reward or a JSON object with `reward`, `score`, or `value`.
Output JSONL contains one canonical GRPO group per task with only the strict
training fields and exact `kiln.rollout-provenance.v1` records. The separate
rollout summary records latency, token counts, seed, adapter, performance, and
scorer output. Before scoring, the CLI
validates the record's schema, seed, adapter, prompt hash, content hash, action
coverage, and usage counts. It publishes each output by atomic replacement only
after every completion succeeds, so an invalid late response cannot leave a
plausible partial dataset. Train this output with
`config.behavior_policy="recorded"`.
"#;

const ROLLOUT_GENERATE_EXAMPLES: &str = r#"Examples:
  kiln rollout-generate --adapter support-bot --thinking false --tasks tasks.jsonl --seeds 4 --request-template request.json --scorer ./score_one.py
      Generate four deterministic scored completions per task and write rollouts.scored.jsonl.

  kiln rollout-generate --adapter support-bot --thinking true --thinking-budget-tokens 96 --thinking-budget-ms 1500 --tasks tasks.jsonl --request-template request.json --scorer ./score_one.py
      Bound each thinking block by both tokens and decode time; the first limit reached closes it.

  kiln rollout-generate --adapter support-bot --thinking true --thinking-budget-tokens unlimited --tasks tasks.jsonl --request-template request.json --scorer ./score_one.py
      Explicitly disable the server's default token budget while preserving any template/server time budget.

  kiln rollout-generate --adapter base --thinking false --tasks tasks.jsonl --request-template request.json --scorer ./score_one.py --output base.rollouts.jsonl --summary-output base.rollouts.summary.json --url http://127.0.0.1:8420
      Generate base-model rollouts by forcing `adapter: null` in every request.
"#;

const CONFIG_OVERVIEW: &str = r#"Validate a Kiln TOML config file without starting the server.

Use this before `kiln serve` to catch invalid values, confirm resolved model settings, and preview process-lifetime accelerator, cache, and decoding policies.

By default, `kiln config` checks the built-in defaults plus environment overrides. Pass `--file` to validate a specific TOML file and see the effective settings that `kiln serve --config <file>` would use. Pass `--json` for the complete source-aware 118-field startup object; sensitive values are present as redacted entries rather than omitted.

Pass `--backend` to resolve and validate the target backend's scheduling policy without probing hardware or loading model weights.
"#;

const CONFIG_EXAMPLES: &str = r#"Examples:
  kiln config
      Validate the default configuration and any KILN_* environment overrides.

  kiln config --file kiln.toml
      Validate kiln.toml before starting the server with `kiln serve --config kiln.toml`.

  kiln config --file kiln.toml --backend rocm
      Resolve ROCm batching and streaming-prefill policy and reject an invalid actor-prefill contract without touching the accelerator.

  kiln config --file kiln.toml --backend rocm --json
      Emit the complete deterministic effective-configuration document and typed hardware-free ROCm policy preview.

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
    /// `-vv` adds trace-level kernel detail. Wins over `KILN_LOGGING_LEVEL` and the
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
        /// Wins over `KILN_MODEL_SERVED_MODEL_ID` and TOML
        /// `model.served_model_id`.
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
        #[arg(long, default_value_t = default_server_url())]
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
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },

    /// Generate scored, provenance-bound single-turn rollouts for GRPO training
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

        /// Thinking token budget (requires `--thinking true`): omit to preserve the template/server value, use 0 to close immediately, or `unlimited` for no limit
        #[arg(long, value_name = "TOKENS|unlimited")]
        thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,

        /// Thinking decode-time budget (requires `--thinking true`): omit to preserve the template/server value, use 0 to close immediately, or `unlimited` for no limit
        #[arg(long, value_name = "MILLISECONDS|unlimited")]
        thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,

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
        #[arg(long, default_value_t = default_server_url())]
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

        /// Resolve and validate scheduling policy for a target backend without
        /// probing hardware or loading model weights.
        #[arg(long, value_enum, value_name = "BACKEND")]
        backend: Option<ConfigCheckBackend>,

        /// Emit the complete source-aware effective configuration as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Configure pi to use this Kiln server as its model backend.
    ///
    /// The full pi + kiln loop:
    ///
    ///   kiln serve &
    ///   kiln pi-setup       # one-time
    ///   pi                  # use normally; sessions are captured
    ///   kiln self-improve   # retrain from the week's sessions
    // (Grand plan §10.14 — the canonical pi + kiln pipeline.)
    #[command(name = "pi-setup", long_about = PI_SETUP_OVERVIEW)]
    PiSetup {
        /// Override the kiln server URL. `/v1` is appended when omitted.
        #[arg(long, alias = "kiln-url", default_value_t = default_server_url())]
        url: String,
        /// Output path for the models.json file. Defaults to the operating-system
        /// account's `.pi/agent/models.json` path.
        #[arg(long)]
        out: Option<String>,
    },

    /// Train and maintain a local judge LoRA that scores agent turns.
    // (Grand plan §10.6 — the self-distillation engine.)
    #[command(subcommand, name = "judge", long_about = JUDGE_OVERVIEW)]
    Judge(JudgeCommands),

    /// Run the weekly self-improvement loop: judge-score recent agent
    /// sessions, then train on the results.
    // (Grand plan §10.6.2 + §10.14.)
    #[command(name = "self-improve", long_about = SELF_IMPROVE_OVERVIEW)]
    SelfImprove {
        /// Server URL.
        #[arg(long, default_value_t = default_server_url())]
        url: String,
        /// Agent adapter to improve.
        #[arg(long, default_value = "pi-coder-current")]
        agent: String,
        /// Judge LoRA alias.
        #[arg(long, default_value = "judge-pi-v1")]
        judge: String,
        /// Skip the CRISP terseness pass (training the model to reach the
        /// same outcomes with fewer tokens).
        #[arg(long, default_value_t = false)]
        no_crisp: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ConfigCheckBackend {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
}

impl ConfigCheckBackend {
    fn policy_identity(self) -> (&'static str, kiln_tensor::Device) {
        match self {
            Self::Cpu => ("cpu", kiln_tensor::Device::Cpu),
            Self::Cuda => ("cuda", kiln_tensor::Device::Cuda(0)),
            Self::Rocm => ("rocm", kiln_tensor::Device::Rocm(0)),
            Self::Metal => ("metal", kiln_tensor::Device::Metal(0)),
            Self::Vulkan => ("vulkan", kiln_tensor::Device::Vulkan(0)),
        }
    }
}

/// `kiln judge` subcommands. (Grand plan §10.6.)
#[derive(Subcommand)]
pub enum JudgeCommands {
    /// Distill a turn-judge LoRA from a teacher model's scoring of
    /// (turn, context) pairs. Requires a configured teacher alias.
    Distill {
        /// Server URL.
        #[arg(long, default_value_t = default_server_url())]
        url: String,
        /// Output adapter name.
        #[arg(long, default_value = "judge-pi-v1")]
        name: String,
        /// Teacher alias.
        #[arg(long, default_value = "qwen3.6-27b@local")]
        teacher: String,
    },
    /// Re-score a sample with the teacher and refresh the judge LoRA
    /// when agreement drops below 80%.
    DriftCheck {
        /// Server URL.
        #[arg(long, default_value_t = default_server_url())]
        url: String,
        /// Judge LoRA alias.
        #[arg(long, default_value = "judge-pi-v1")]
        judge: String,
        /// 27B teacher alias.
        #[arg(long, default_value = "qwen3.6-27b@local")]
        teacher: String,
    },
}

const PI_SETUP_OVERVIEW: &str = "Point pi at this Kiln server.\n\
Merges a kiln-local provider into ~/.pi/agent/models.json and settings.json,\n\
backing up existing files first and preserving unrelated providers/settings.";

const JUDGE_OVERVIEW: &str = "Train and maintain a local judge LoRA for scoring agent turns.\n\
Distill it once from a teacher model (kiln judge distill — requires a\n\
configured teacher alias, default qwen3.6-27b@local), then let\n\
`kiln self-improve` use it weekly. Re-check drift periodically with\n\
`kiln judge drift-check`.";

const SELF_IMPROVE_OVERVIEW: &str = "Run the weekly self-improvement loop.\n\
Scores recent agent sessions with the local judge LoRA, runs GRPO with\n\
judge-derived advantages, and (unless --no-crisp) trains a terseness\n\
pass on successful trajectories. Needs a judge LoRA from `kiln judge\n\
distill` first.";

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

        /// Learning rate. Omit to let the server pick the per-optimizer
        /// default (Muon 1e-3, AdamW/SGD 1e-4 for native SFT).
        #[arg(long)]
        lr: Option<f64>,

        /// Number of epochs
        #[arg(long, default_value = "1")]
        epochs: u32,

        /// LoRA rank for the trained adapter
        #[arg(long)]
        lora_rank: Option<usize>,

        /// Invalid-row behavior: fail the submission or skip and receipt rows
        #[arg(long, default_value = "fail", value_parser = ["fail", "skip"])]
        invalid_row_policy: String,

        /// Run an adapter-effect smoke test after successful training
        #[arg(long)]
        adapter_smoke_test: bool,

        /// JSON string array or single-text file containing smoke-test prompts
        #[arg(long, requires = "adapter_smoke_test")]
        adapter_smoke_prompts_file: Option<String>,

        /// Scan every backward gradient for NaN/Inf and fail at its producer
        #[arg(long)]
        detect_anomaly: bool,

        /// Emit an exact resumable checkpoint every N optimizer steps
        #[arg(long)]
        checkpoint_interval: Option<std::num::NonZeroUsize>,

        /// Immutable .kiln-checkpoint basename reported by job status
        #[arg(long)]
        resume_checkpoint: Option<String>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
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

        /// JSON string array or single-text file containing smoke-test prompts
        #[arg(long, requires = "adapter_smoke_test")]
        adapter_smoke_prompts_file: Option<String>,

        /// Use the per-completion reference fallback instead of shared prefix state
        #[arg(long)]
        no_shared_prefix_reference: bool,

        /// Scan every backward gradient for NaN/Inf and fail at its producer
        #[arg(long)]
        detect_anomaly: bool,

        /// Emit an exact resumable checkpoint every N optimizer groups
        #[arg(long)]
        checkpoint_interval: Option<std::num::NonZeroUsize>,

        /// Immutable .kiln-checkpoint basename reported by job status
        #[arg(long)]
        resume_checkpoint: Option<String>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Train a LoRA adapter by on-policy or off-policy distillation
    #[command(long_about = TRAIN_OPD_OVERVIEW)]
    Opd {
        /// Path to one OPD request object or a JSON array of prompts
        #[arg(long, short)]
        file: String,

        /// Adapter name to train
        #[arg(long, default_value = "default")]
        adapter: String,

        /// Registered teacher alias; overrides the request when present
        #[arg(long)]
        teacher: Option<String>,

        /// LoRA rank for the trained adapter
        #[arg(long)]
        lora_rank: Option<usize>,

        /// Scan every backward gradient for NaN/Inf and fail at its producer
        #[arg(long)]
        detect_anomaly: bool,

        /// Layer segments for the memory-bounded student sampler
        #[arg(long)]
        sampler_segments: Option<std::num::NonZeroUsize>,

        /// Student rollout-prefix construction algorithm
        #[arg(
            long,
            value_parser = ["legacy_action_boundary", "chat_template"]
        )]
        rollout_prompt_rendering: Option<String>,

        /// Emit an exact resumable checkpoint every N committed optimizer steps
        #[arg(long)]
        checkpoint_interval: Option<std::num::NonZeroUsize>,

        /// Immutable .kiln-checkpoint basename reported by job status
        #[arg(long)]
        resume_checkpoint: Option<String>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Create and manage immutable Hugging Face TRL/PEFT handoff bundles
    #[command(subcommand, long_about = TRAIN_HF_OVERVIEW)]
    Hf(HfTrainCommands),
    /// Show training queue / per-job status
    Status {
        /// Specific job ID to look up. If omitted, shows the full queue.
        #[arg(long)]
        job_id: Option<String>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Cancel a queued or running training job
    Cancel {
        /// Job ID to cancel (queued jobs leave the queue; running jobs
        /// stop at the next step boundary)
        #[arg(long)]
        job_id: String,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
}

#[derive(Subcommand)]
pub enum HfTrainCommands {
    /// Create, download, and verify an immutable SFT handoff bundle
    #[command(name = "export-sft", long_about = TRAIN_HF_EXPORT_SFT_OVERVIEW)]
    ExportSft {
        /// Server-local SFT JSONL path; the CLI does not upload this file
        #[arg(
            long,
            short,
            conflicts_with = "dataset",
            required_unless_present = "dataset"
        )]
        file: Option<String>,

        /// Named server dataset, including corrections:active
        #[arg(long, conflicts_with = "file", required_unless_present = "file")]
        dataset: Option<String>,

        /// Immutable server export name
        #[arg(long)]
        name: String,

        /// Local tar.gz destination; defaults to {name}.kiln-hf.tar.gz
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Invalid-row behavior: fail the export or skip and receipt rows
        #[arg(long, default_value = "fail", value_parser = ["fail", "skip"])]
        invalid_row_policy: String,

        /// Existing server adapter to snapshot and continue training
        #[arg(long)]
        input_adapter: Option<String>,

        /// Local JSON file containing an object-valued split manifest
        #[arg(long)]
        split_manifest: Option<PathBuf>,

        /// Retain the verified export in the server registry after download
        #[arg(long)]
        keep_server_copy: bool,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Create, download, and verify an immutable recorded-GRPO handoff bundle
    #[command(name = "export-grpo", long_about = TRAIN_HF_EXPORT_GRPO_OVERVIEW)]
    ExportGrpo {
        /// Server-local canonical GRPO JSONL path; the CLI does not upload it
        #[arg(long, short)]
        file: String,

        /// Immutable server export name
        #[arg(long)]
        name: String,

        /// Local tar.gz destination; defaults to {name}.kiln-hf.tar.gz
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Existing server adapter matching the recorded behavior policy
        #[arg(long)]
        input_adapter: Option<String>,

        /// Local JSON file containing an object-valued split manifest
        #[arg(long)]
        split_manifest: Option<PathBuf>,

        /// Retain the verified export in the server registry after download
        #[arg(long)]
        keep_server_copy: bool,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Verify and import a completed external-training PEFT result
    #[command(name = "import-peft", long_about = TRAIN_HF_IMPORT_PEFT_OVERVIEW)]
    ImportPeft {
        /// Completed extracted .kiln-hf directory containing result artifacts
        #[arg(long)]
        bundle: PathBuf,

        /// New adapter name on the receiving server
        #[arg(long)]
        name: String,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// List immutable HF/TRL exports retained by the server
    List {
        /// Emit the exact server response as JSON
        #[arg(long)]
        json: bool,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Delete one immutable HF/TRL export from the server
    Delete {
        /// Export name
        #[arg(long)]
        name: String,

        /// Delete only when the current export has this sha256:<64-hex> identity
        #[arg(long)]
        export_sha256: Option<String>,

        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
}

#[derive(Subcommand)]
pub enum AdapterCommands {
    /// List saved adapters and show which one is active
    List {
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Load an adapter from disk
    Load {
        /// Adapter name
        name: String,
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Unload the active adapter and revert to the base model
    Unload {
        /// Optional legacy adapter name; ignored because the server unloads the active adapter
        name: Option<String>,
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
        url: String,
    },
    /// Delete an adapter
    Delete {
        /// Adapter name
        name: String,
        /// Server URL; defaults to the local kiln serve instance
        #[arg(long, default_value_t = default_server_url())]
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

        /// Explicit tokenizer.json path. Defaults to KILN_MODEL_TOKENIZER_PATH,
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

    let cuda_status = if kiln_tensor::cuda_is_available() {
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
        "  {} /ui/, /v1/chat/completions, /v1/completions, /v1/train/sft, /health, /metrics",
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

/// Body of the `kiln serve` failed-bind diagnostic when the address is
/// already in use. Pure so tests can assert the wording.
pub(crate) fn bind_addr_in_use_message(host: &str, port: u16) -> String {
    format!(
        "{host}:{port} is already in use — another kiln (or another service) is likely listening there.\n  \
         Run `kiln health` to check for a live server, stop the other process,\n  \
         or pick a different port via KILN_SERVER_PORT or `[server] port` in kiln.toml."
    )
}

/// Print the AddrInUse diagnostic for a failed `kiln serve` listener bind.
pub fn print_bind_addr_in_use(host: &str, port: u16) {
    eprintln!(
        "{} {}",
        style("✗").red().bold(),
        bind_addr_in_use_message(host, port)
    );
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
    let thinking_tokens = body
        .get("default_thinking_budget_tokens")
        .and_then(|value| value.as_u64())
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unlimited".to_string());
    let thinking_time = body
        .get("default_thinking_budget_ms")
        .and_then(|value| value.as_u64())
        .map(|value| format!("{value} ms"))
        .unwrap_or_else(|| "unlimited".to_string());
    let _ = writeln!(
        out,
        "  {} tokens={} time={}",
        style("Thinking:").dim(),
        style(thinking_tokens).cyan(),
        style(thinking_time).cyan()
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

/// Warning text when the configured host listens beyond loopback, since the
/// HTTP surface (inference AND training) ships with no built-in auth. Pure so
/// tests can assert the wording.
pub(crate) fn non_loopback_host_warning(host: &str) -> Option<String> {
    let is_loopback = host == "localhost"
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false);
    (!is_loopback).then(|| {
        format!(
            "host \"{host}\" listens beyond loopback. Kiln has no built-in auth: anyone \
who can reach this port can run inference and submit training data. Front it with an \
authenticated reverse proxy or keep it on a private network (see README \"Security model\")."
        )
    })
}

fn format_checkpoint_boundary_config(
    policy: kiln_train::CheckpointBoundaryPolicy,
    diagnostics: crate::config::CheckpointBoundaryConfigDiagnostics,
) -> String {
    use std::fmt::Write as _;

    let stride = policy
        .anchor_stride()
        .map(|value| {
            format!(
                "{value} (explicit; source: {})",
                diagnostics.checkpoint_boundary_anchor_stride_source
            )
        })
        .unwrap_or_else(|| {
            format!(
                "auto (source: {})",
                diagnostics.checkpoint_boundary_anchor_stride_source
            )
        });
    let mut output = String::new();
    let _ = writeln!(
        output,
        "  {} {} (source: {})",
        style("Checkpoint boundary mode:").dim(),
        policy.recompute_mode(),
        diagnostics.recompute_checkpoint_boundaries_source
    );
    let _ = writeln!(
        output,
        "  {} {} tokens (source: {})",
        style("Boundary recompute threshold:").dim(),
        policy.recompute_threshold_tokens(),
        diagnostics.recompute_boundary_threshold_tokens_source
    );
    let _ = writeln!(
        output,
        "  {} {stride}",
        style("Boundary anchor stride:").dim()
    );
    let _ = writeln!(
        output,
        "  {} {} GiB / {} bytes (source: {})",
        style("Boundary cache target:").dim(),
        diagnostics.checkpoint_boundary_cache_gb,
        policy.cache_target_bytes(),
        diagnostics.checkpoint_boundary_cache_gb_source
    );
    let _ = writeln!(
        output,
        "  {} immutable after startup; restart required to change",
        style("Boundary policy lifecycle:").dim()
    );
    let _ = writeln!(
        output,
        "  {} includes mode, threshold, stride, and cache target; changes reject exact resume",
        style("Planning identity v3:").dim()
    );
    output
}

fn format_actor_prefill_config(config: &crate::config::KilnConfig) -> String {
    use std::fmt::Write as _;

    let configured = |value: Option<usize>, unit: &str| {
        value
            .map(|number| format!("{number}{unit}"))
            .unwrap_or_else(|| "auto".to_owned())
    };
    let mut output = String::new();
    for (label, value, source) in [
        (
            "Actor cycle token budget:",
            format!("{} tokens", config.server.max_batch_tokens.tokens()),
            config.server.max_batch_tokens.source(),
        ),
        (
            "Actor prefill token ceiling:",
            format!(
                "{} tokens",
                config.server.max_prefill_tokens_per_cycle.tokens()
            ),
            config.server.max_prefill_tokens_per_cycle.source(),
        ),
        (
            "Actor prefill layer ceiling:",
            format!(
                "{} layers",
                config.server.max_prefill_layers_per_cycle.layers()
            ),
            config.server.max_prefill_layers_per_cycle.source(),
        ),
        (
            "Decode width ceiling:",
            configured(config.server.max_decode_batch.limit(), ""),
            config.server.max_decode_batch.source(),
        ),
        (
            "Actor cycle idle:",
            format!("{} ms", config.batching.actor_cycle_idle_ms.millis()),
            config.batching.actor_cycle_idle_ms.source(),
        ),
        (
            "Streaming prefill mode:",
            config.streaming_prefill.mode.mode().to_string(),
            config.streaming_prefill.mode.source(),
        ),
        (
            "Streaming prefill threshold:",
            configured(
                config.streaming_prefill.threshold_tokens.configured(),
                " tokens",
            ),
            config.streaming_prefill.threshold_tokens.source(),
        ),
        (
            "Streaming base tile:",
            configured(config.streaming_prefill.tile_tokens.configured(), " tokens"),
            config.streaming_prefill.tile_tokens.source(),
        ),
        (
            "Streaming tape tile:",
            configured(
                config.streaming_prefill.tape_tile_tokens.configured(),
                " tokens",
            ),
            config.streaming_prefill.tape_tile_tokens.source(),
        ),
        (
            "Streaming detached full-attention tile:",
            configured(
                config
                    .streaming_prefill
                    .detached_full_attn_tile_tokens
                    .configured(),
                " tokens",
            ),
            config
                .streaming_prefill
                .detached_full_attn_tile_tokens
                .source(),
        ),
    ] {
        let _ = writeln!(
            output,
            "  {} {value} (source: {source})",
            style(label).dim()
        );
    }
    let _ = writeln!(
        output,
        "  {} configured values resolve once against the selected backend and fail before model-weight loading",
        style("Prefill policy lifecycle:").dim(),
    );
    output
}

fn format_streaming_prefill_dispatch(
    rule: crate::config::StreamingPrefillDispatchRuleDiagnostics,
) -> String {
    match rule.policy {
        crate::config::StreamingPrefillDispatchPolicy::Never => "never".to_owned(),
        crate::config::StreamingPrefillDispatchPolicy::AllNonEmpty => "all_non_empty".to_owned(),
        crate::config::StreamingPrefillDispatchPolicy::PromptTokensAtLeast => format!(
            "prompt_tokens_at_least {} tokens",
            rule.minimum_prompt_tokens
                .expect("prompt-token dispatch rule must carry its threshold")
        ),
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct ConfigCheckBackendPreview {
    target_backend: &'static str,
    hardware_probed: bool,
    decode_runtime: crate::config::DecodeRuntimeConfig,
    batching: crate::config::BatchingRuntimeConfig,
    streaming_prefill: crate::config::StreamingPrefillRuntimeConfig,
    actor_prefill_contract_valid: bool,
}

fn resolve_actor_prefill_backend_config(
    config: &crate::config::KilnConfig,
    backend: ConfigCheckBackend,
) -> anyhow::Result<ConfigCheckBackendPreview> {
    use anyhow::Context as _;

    let (backend_name, device) = backend.policy_identity();
    let decode_policy = kiln_model::DecodeExecutionPolicy::for_backend(backend_name, device);
    let decode_runtime = crate::batching_engine::resolve_decode_runtime_config(
        config.server.deterministic,
        config.server.max_decode_batch,
        Some(decode_policy),
        config.server.max_batch_tokens,
    );
    let batching = config.batching.resolve(
        crate::config::BatchingBackendPolicy::from_decode_execution_policy(decode_policy),
        decode_runtime.max_decode_batch.effective,
    );
    let streaming =
        config
            .streaming_prefill
            .resolve(kiln_model::StreamingPrefillBackendPolicy::for_backend(
                backend_name,
                device,
            ));
    crate::config::validate_actor_prefill_tile_contract(
        batching,
        streaming,
        config.server.max_batch_tokens,
        config.server.max_prefill_tokens_per_cycle,
        decode_runtime.max_decode_batch.effective,
    )
    .with_context(|| format!("invalid {backend_name} actor-prefill contract"))?;

    Ok(ConfigCheckBackendPreview {
        target_backend: backend_name,
        hardware_probed: false,
        decode_runtime,
        batching,
        streaming_prefill: streaming,
        actor_prefill_contract_valid: true,
    })
}

fn format_actor_prefill_backend_preview(preview: &ConfigCheckBackendPreview) -> String {
    use std::fmt::Write as _;

    let mut output = String::new();
    let _ = writeln!(
        output,
        "  {} {} (hardware-free policy preview)",
        style("Target backend:").dim(),
        preview.target_backend
    );
    let _ = writeln!(
        output,
        "  {} true (source: built_in)",
        style("Batching actor effective:").dim(),
    );
    let _ = writeln!(
        output,
        "  {} {} (source: backend_policy)",
        style("Actor prefill alignment required:").dim(),
        preview.batching.actor_prefill_tile_alignment_required
    );
    let _ = writeln!(
        output,
        "  {} {} rows (source: {})",
        style("Effective decode width:").dim(),
        preview.decode_runtime.max_decode_batch.effective,
        preview.decode_runtime.max_decode_batch.effective_source
    );
    let _ = writeln!(
        output,
        "  {} {} (source: {})",
        style("Effective streaming dispatch:").dim(),
        format_streaming_prefill_dispatch(preview.streaming_prefill.dispatch.effective),
        preview.streaming_prefill.dispatch.effective_source
    );
    for (label, diagnostics) in [
        (
            "Effective streaming base tile:",
            preview.streaming_prefill.tile_tokens,
        ),
        (
            "Effective streaming tape tile:",
            preview.streaming_prefill.tape_tile_tokens,
        ),
        (
            "Effective detached full-attention tile:",
            preview.streaming_prefill.detached_full_attn_tile_tokens,
        ),
    ] {
        let _ = writeln!(
            output,
            "  {} {} tokens (source: {})",
            style(label).dim(),
            diagnostics.effective,
            diagnostics.effective_source
        );
    }
    let _ = writeln!(
        output,
        "  {} valid (no hardware probe or model load)",
        style("Actor-prefill backend contract:").dim()
    );
    output
}

fn format_actor_prefill_backend_config(
    config: &crate::config::KilnConfig,
    backend: ConfigCheckBackend,
) -> anyhow::Result<String> {
    resolve_actor_prefill_backend_config(config, backend)
        .map(|preview| format_actor_prefill_backend_preview(&preview))
}

#[derive(serde::Serialize)]
struct ConfigCheckJsonDocument {
    schema_id: &'static str,
    schema_version: u32,
    hardware_probed: bool,
    effective_configuration: crate::config::EffectiveConfiguration,
    accelerator_runtime: crate::config::ResolvedAcceleratorRuntimePolicy,
    application_paths: crate::config::ResolvedApplicationPaths,
    checkpoint_boundary_policy: kiln_train::CheckpointBoundaryPolicy,
    backend_preview: Option<ConfigCheckBackendPreview>,
}

/// Run the `config check` CLI subcommand: validate config without starting.
pub fn run_config_check(
    file: Option<&str>,
    backend: Option<ConfigCheckBackend>,
    json: bool,
) -> anyhow::Result<()> {
    use crate::config::KilnConfig;

    match KilnConfig::load(file).and_then(|config| {
        config
            .speculative
            .validate_for_model(&kiln_core::config::ModelConfig::qwen3_5_4b())?;
        config.speculative.validate_for_serving()?;
        let checkpoint_boundary_policy = config.training.checkpoint_boundary_policy()?;
        let backend_preview = backend
            .map(|target| resolve_actor_prefill_backend_config(&config, target))
            .transpose()?;
        Ok((config, checkpoint_boundary_policy, backend_preview))
    }) {
        Ok((config, checkpoint_boundary_policy, backend_preview)) => {
            let accelerator_runtime = config
                .accelerator
                .resolved_policy(config.server.serving_profile);
            let application_paths = config.paths.resolve()?;
            if json {
                let document = ConfigCheckJsonDocument {
                    schema_id: "kiln.config-check.v1",
                    schema_version: 1,
                    hardware_probed: false,
                    effective_configuration: config.effective_configuration()?,
                    accelerator_runtime,
                    application_paths,
                    checkpoint_boundary_policy,
                    backend_preview,
                };
                println!("{}", serde_json::to_string_pretty(&document)?);
                return Ok(());
            }
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
            println!(
                "  {} {} (source: {})",
                style("Cache root:").dim(),
                application_paths.cache_root.display(),
                application_paths.cache_root_source
            );
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
                style("CUDA graph cache entries:").dim(),
                config.memory.cuda_graph_cache_entries
            );
            println!(
                "  {} {}",
                style("Accelerator policy schema:").dim(),
                accelerator_runtime.schema_id,
            );
            println!(
                "  {} {}",
                style("Vulkan kernel policy:").dim(),
                accelerator_runtime.vulkan_kernel_policy_schema_id,
            );
            println!(
                "  {} {}",
                style("Vulkan device policy:").dim(),
                accelerator_runtime.vulkan_device_policy_schema_id,
            );
            println!(
                "  {} {} (source: {})",
                style("Vulkan physical device:").dim(),
                accelerator_runtime
                    .vulkan_device_index
                    .effective
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "auto".to_owned()),
                accelerator_runtime.vulkan_device_index.source,
            );
            println!(
                "  {} {} (source: {})",
                style("Vulkan validation:").dim(),
                accelerator_runtime.vulkan_validation.effective,
                accelerator_runtime.vulkan_validation.source,
            );
            println!(
                "  {} {} (source: {})",
                style("Kiln-tensor API routes:").dim(),
                accelerator_runtime.kt_api_mode.effective,
                accelerator_runtime.kt_api_mode.source,
            );
            println!(
                "  {} {} MiB (source: {})",
                style("Full-attention score ceiling:").dim(),
                accelerator_runtime
                    .full_attention_score_budget_mib
                    .effective,
                accelerator_runtime.full_attention_score_budget_mib.source,
            );
            println!(
                "  {} {} (source: {})",
                style("CUDA kernel profile:").dim(),
                accelerator_runtime.cuda_kernel_profile.effective,
                accelerator_runtime.cuda_kernel_profile.source,
            );
            println!(
                "  {} {} (source: {})",
                style("CUDA Marlin profile:").dim(),
                accelerator_runtime.cuda_marlin_profile.effective,
                accelerator_runtime.cuda_marlin_profile.source,
            );
            println!(
                "  {} {} (source: {})",
                style("CUDA FlashAttention backward:").dim(),
                accelerator_runtime.cuda_flash_backward_mode.effective,
                accelerator_runtime.cuda_flash_backward_mode.source,
            );
            println!(
                "  {} {} (source: {})",
                style("Metal kernel profile:").dim(),
                accelerator_runtime.metal_kernel_profile.effective,
                accelerator_runtime.metal_kernel_profile.source,
            );
            println!(
                "  {} {} (source: {})",
                style("ROCm synchronization:").dim(),
                accelerator_runtime.rocm_synchronization_mode.effective,
                accelerator_runtime.rocm_synchronization_mode.source,
            );
            println!(
                "  {} {} (source: {})",
                style("ROCm strided batched matmul:").dim(),
                accelerator_runtime
                    .rocm_strided_batched_matmul_mode
                    .effective,
                accelerator_runtime.rocm_strided_batched_matmul_mode.source,
            );
            println!(
                "  {} {} (source: {})",
                style("ROCm BF16 matmul output:").dim(),
                accelerator_runtime.rocm_bf16_matmul_output_mode.effective,
                accelerator_runtime.rocm_bf16_matmul_output_mode.source,
            );
            println!(
                "  {} {} (source: {})",
                style("ROCm kernel profile:").dim(),
                accelerator_runtime.rocm_kernel_profile.effective,
                accelerator_runtime.rocm_kernel_profile.source,
            );
            println!(
                "  {} {} -> {} (source: {})",
                style("ROCm graph mode:").dim(),
                accelerator_runtime.rocm_graph_mode.configured,
                accelerator_runtime.rocm_graph_mode.effective,
                accelerator_runtime.rocm_graph_mode.source,
            );
            println!(
                "  {} {} (source: {})",
                style("ROCm graph cache:").dim(),
                accelerator_runtime.rocm_graph_cache_entries.effective,
                accelerator_runtime.rocm_graph_cache_entries.source,
            );
            println!(
                "  {} {} MiB ({} bytes; source: {})",
                style("ROCm graph byte budget:").dim(),
                accelerator_runtime.rocm_graph_cache_max_bytes.effective / (1024 * 1024),
                accelerator_runtime.rocm_graph_cache_max_bytes.effective,
                accelerator_runtime.rocm_graph_cache_max_bytes.source,
            );
            println!(
                "  {} {}",
                style("Prefix cache:").dim(),
                config.prefix_cache.enabled
            );
            println!("  {} always enabled", style("Batching actor:").dim(),);
            print!("{}", format_actor_prefill_config(&config));
            if let Some(preview) = backend_preview.as_ref() {
                print!("{}", format_actor_prefill_backend_preview(preview));
            }
            println!(
                "  {} {}",
                style("Rowwise decode:").dim(),
                config.batching.rowwise_decode.enabled()
            );
            println!(
                "  {} {}",
                style("Prefix-aware admission:").dim(),
                config.batching.prefix_aware_admission.enabled()
            );
            println!(
                "  {} {}",
                style("Prefill admission quantum:").dim(),
                config
                    .batching
                    .prefill_admission_quantum
                    .configured()
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "auto".to_string())
            );
            print!(
                "{}",
                format_checkpoint_boundary_config(
                    checkpoint_boundary_policy,
                    config.training.checkpoint_boundary_diagnostics(),
                )
            );
            println!(
                "  {} {}",
                style("Thinking tokens:").dim(),
                config
                    .server
                    .default_thinking_budget_tokens
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unlimited".to_string())
            );
            println!(
                "  {} {}",
                style("Thinking time:").dim(),
                config
                    .server
                    .default_thinking_budget_ms
                    .map(|value| format!("{value} ms"))
                    .unwrap_or_else(|| "unlimited".to_string())
            );
            println!(
                "  {} {}",
                style("Speculative serving:").dim(),
                "off (pending local accelerator qualification)"
            );
            if let Some(warning) = non_loopback_host_warning(&config.server.host) {
                println!();
                println!("  {} {}", style("⚠").yellow().bold(), warning);
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("{} Configuration error: {e:#}", style("✗").red().bold());
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
fn load_adapter_smoke_prompts(path: Option<&str>) -> anyhow::Result<Option<Vec<String>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let contents = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("Failed to read adapter smoke prompts {path}: {error}"))?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        anyhow::bail!("Adapter smoke prompts file {path} is empty");
    }
    let prompts = if trimmed.starts_with('[') {
        serde_json::from_str::<Vec<String>>(trimmed).map_err(|error| {
            anyhow::anyhow!("Invalid JSON string array in adapter smoke prompts {path}: {error}")
        })?
    } else {
        vec![contents]
    };
    if prompts.is_empty() {
        anyhow::bail!("Adapter smoke prompts file {path} contains an empty JSON array");
    }
    for (index, prompt) in prompts.iter().enumerate() {
        if prompt.trim().is_empty() {
            anyhow::bail!("Adapter smoke prompts file {path} has a blank prompt at index {index}");
        }
    }
    Ok(Some(prompts))
}

pub async fn run_train_sft(
    url: &str,
    file: &str,
    adapter: &str,
    lr: Option<f64>,
    epochs: u32,
    lora_rank: Option<usize>,
    invalid_row_policy: &str,
    adapter_smoke_test: bool,
    adapter_smoke_prompts_file: Option<&str>,
    detect_anomaly: bool,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
) -> anyhow::Result<()> {
    let adapter_smoke_prompts = load_adapter_smoke_prompts(adapter_smoke_prompts_file)?;
    // Skip must reach the server with raw JSONL intact so malformed rows can
    // receive stable rejection hashes. Fail-mode legacy non-JSONL inputs keep
    // the inline parsing behavior for compatibility.
    let body = if is_sft_jsonl_path(file) || invalid_row_policy == "skip" {
        println!(
            "{} Submitting SFT JSONL dataset_path for adapter '{}'",
            style("→").cyan().bold(),
            style(adapter).white().bold()
        );
        build_sft_jsonl_training_payload(
            file,
            adapter,
            lr,
            epochs,
            lora_rank,
            invalid_row_policy,
            adapter_smoke_test,
            adapter_smoke_prompts.as_deref(),
            detect_anomaly,
            checkpoint_interval,
            resume_checkpoint,
        )?
    } else {
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

        build_sft_training_payload(
            examples,
            adapter,
            lr,
            epochs,
            lora_rank,
            invalid_row_policy,
            adapter_smoke_test,
            adapter_smoke_prompts.as_deref(),
            detect_anomaly,
            checkpoint_interval,
            resume_checkpoint,
        )
    };

    submit_training_payload(url, "sft", "SFT", &body).await
}

/// Run the `train grpo` CLI subcommand.
pub async fn run_train_grpo(
    url: &str,
    file: &str,
    adapter: &str,
    lora_rank: Option<usize>,
    adapter_smoke_test: bool,
    adapter_smoke_prompts_file: Option<&str>,
    no_shared_prefix_reference: bool,
    detect_anomaly: bool,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
) -> anyhow::Result<()> {
    let adapter_smoke_prompts = load_adapter_smoke_prompts(adapter_smoke_prompts_file)?;
    let body = if is_grpo_jsonl_path(file) {
        build_grpo_jsonl_training_payload(
            file,
            adapter,
            lora_rank,
            adapter_smoke_test,
            adapter_smoke_prompts.as_deref(),
            no_shared_prefix_reference,
            detect_anomaly,
            checkpoint_interval,
            resume_checkpoint,
        )?
    } else {
        let content = std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("Failed to read {file}: {e}"))?;

        let body: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("Invalid JSON in {file}: {e}"))?;

        build_grpo_training_payload(
            body,
            adapter,
            lora_rank,
            adapter_smoke_test,
            adapter_smoke_prompts.as_deref(),
            no_shared_prefix_reference,
            detect_anomaly,
            checkpoint_interval,
            resume_checkpoint,
        )?
    };

    println!(
        "{} Submitting GRPO training batch on adapter '{}'",
        style("→").cyan().bold(),
        style(adapter).white().bold()
    );

    submit_training_payload(url, "grpo", "GRPO", &body).await
}

/// Run the `train opd` CLI subcommand.
pub async fn run_train_opd(
    url: &str,
    file: &str,
    adapter: &str,
    teacher: Option<&str>,
    lora_rank: Option<usize>,
    detect_anomaly: bool,
    sampler_segments: Option<usize>,
    rollout_prompt_rendering: Option<&str>,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(file)
        .map_err(|error| anyhow::anyhow!("Failed to read {file}: {error}"))?;
    let request: serde_json::Value = serde_json::from_str(&content)
        .map_err(|error| anyhow::anyhow!("Invalid JSON in {file}: {error}"))?;
    let body = build_opd_training_payload(
        request,
        adapter,
        teacher,
        lora_rank,
        detect_anomaly,
        sampler_segments,
        rollout_prompt_rendering,
        checkpoint_interval,
        resume_checkpoint,
    )?;
    let resolved_teacher = body["teacher"]
        .as_str()
        .expect("validated OPD request has a teacher");

    println!(
        "{} Submitting OPD training on adapter '{}' with teacher '{}'",
        style("→").cyan().bold(),
        style(adapter).white().bold(),
        style(resolved_teacher).white().bold()
    );
    submit_training_payload(url, "opd", "OPD", &body).await
}

async fn submit_training_payload(
    url: &str,
    route: &str,
    label: &str,
    body: &serde_json::Value,
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{url}/v1/train/{route}"))
        .json(body)
        .send()
        .await
        .map_err(|error| handle_request_error(url, error))?;
    let status = response.status();
    let response_body: serde_json::Value = response.json().await?;

    if !status.is_success() {
        eprintln!(
            "{} {label} submission failed: {}",
            style("✗").red().bold(),
            render_api_error(&response_body, status)
        );
        std::process::exit(1);
    }

    println!(
        "{} {label} training job submitted",
        style("✓").green().bold()
    );
    let job_id = response_body.get("job_id").and_then(|value| value.as_str());
    if let Some(id) = job_id {
        println!("  {} {}", style("Job ID:").dim(), id);
    }
    if let Some(seed) = response_body
        .get("effective_seed")
        .and_then(|value| value.as_str())
    {
        println!("  {} {}", style("Effective seed:").dim(), seed);
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
    Ok(())
}

fn is_grpo_jsonl_path(file: &str) -> bool {
    std::path::Path::new(file)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("jsonl") || ext.eq_ignore_ascii_case("ndjson"))
        .unwrap_or(false)
}

fn is_sft_jsonl_path(file: &str) -> bool {
    std::path::Path::new(file)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("jsonl") || ext.eq_ignore_ascii_case("ndjson"))
        .unwrap_or(false)
}

fn build_sft_jsonl_training_payload(
    file: &str,
    adapter: &str,
    lr: Option<f64>,
    epochs: u32,
    lora_rank: Option<usize>,
    invalid_row_policy: &str,
    adapter_smoke_test: bool,
    adapter_smoke_prompts: Option<&[String]>,
    detect_anomaly: bool,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
) -> anyhow::Result<serde_json::Value> {
    let dataset_path = std::fs::canonicalize(file)
        .map_err(|e| anyhow::anyhow!("Failed to resolve SFT JSONL file {file}: {e}"))?;
    let dataset_path = dataset_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("SFT JSONL path is not valid UTF-8: {file}"))?
        .to_string();
    let mut config = serde_json::json!({
        "training_profile": "native_online_lora_v1",
        "output_name": adapter,
        "epochs": epochs,
        "invalid_row_policy": invalid_row_policy,
    });
    if let Some(lr) = lr {
        config["learning_rate"] = serde_json::json!(lr);
    }
    if let Some(rank) = lora_rank {
        config["lora_rank"] = serde_json::json!(rank);
    }
    if adapter_smoke_test {
        config["adapter_smoke_test"] = serde_json::json!(true);
    }
    if let Some(prompts) = adapter_smoke_prompts {
        config["adapter_smoke_prompts"] = serde_json::json!(prompts);
    }
    if detect_anomaly {
        config["detect_anomaly"] = serde_json::json!(true);
    }
    if let Some(interval) = checkpoint_interval {
        config["checkpoint_interval"] = serde_json::json!(interval);
    }
    if let Some(checkpoint) = resume_checkpoint {
        config["resume_checkpoint"] = serde_json::json!(checkpoint);
    }
    Ok(serde_json::json!({
        "dataset_path": dataset_path,
        "config": config,
    }))
}

fn build_grpo_jsonl_training_payload(
    file: &str,
    adapter: &str,
    lora_rank: Option<usize>,
    adapter_smoke_test: bool,
    adapter_smoke_prompts: Option<&[String]>,
    no_shared_prefix_reference: bool,
    detect_anomaly: bool,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
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
    if let Some(prompts) = adapter_smoke_prompts {
        config["adapter_smoke_prompts"] = serde_json::json!(prompts);
    }
    if no_shared_prefix_reference {
        config["shared_prefix_reference"] = serde_json::json!(false);
    }
    if detect_anomaly {
        config["detect_anomaly"] = serde_json::json!(true);
    }
    if let Some(interval) = checkpoint_interval {
        config["checkpoint_interval"] = serde_json::json!(interval);
    }
    if let Some(checkpoint) = resume_checkpoint {
        config["resume_checkpoint"] = serde_json::json!(checkpoint);
    }
    Ok(serde_json::json!({
        "dataset_path": dataset_path,
        "config": config,
    }))
}

fn build_sft_training_payload(
    examples: Vec<serde_json::Value>,
    adapter: &str,
    lr: Option<f64>,
    epochs: u32,
    lora_rank: Option<usize>,
    invalid_row_policy: &str,
    adapter_smoke_test: bool,
    adapter_smoke_prompts: Option<&[String]>,
    detect_anomaly: bool,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "training_profile": "native_online_lora_v1",
        "output_name": adapter,
        "epochs": epochs,
        "invalid_row_policy": invalid_row_policy,
    });
    // Omitted --lr means "let the server resolve per optimizer".
    if let Some(lr) = lr {
        config["learning_rate"] = serde_json::json!(lr);
    }
    if let Some(rank) = lora_rank {
        config["lora_rank"] = serde_json::json!(rank);
    }
    if adapter_smoke_test {
        config["adapter_smoke_test"] = serde_json::json!(true);
    }
    if let Some(prompts) = adapter_smoke_prompts {
        config["adapter_smoke_prompts"] = serde_json::json!(prompts);
    }
    if detect_anomaly {
        config["detect_anomaly"] = serde_json::json!(true);
    }
    if let Some(interval) = checkpoint_interval {
        config["checkpoint_interval"] = serde_json::json!(interval);
    }
    if let Some(checkpoint) = resume_checkpoint {
        config["resume_checkpoint"] = serde_json::json!(checkpoint);
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
    adapter_smoke_prompts: Option<&[String]>,
    no_shared_prefix_reference: bool,
    detect_anomaly: bool,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
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
    if let Some(prompts) = adapter_smoke_prompts {
        config_obj.insert("adapter_smoke_prompts".into(), serde_json::json!(prompts));
    }
    if no_shared_prefix_reference {
        config_obj.insert("shared_prefix_reference".into(), serde_json::json!(false));
    }
    if detect_anomaly {
        config_obj.insert("detect_anomaly".into(), serde_json::json!(true));
    }
    if let Some(interval) = checkpoint_interval {
        config_obj.insert("checkpoint_interval".into(), serde_json::json!(interval));
    }
    if let Some(checkpoint) = resume_checkpoint {
        config_obj.insert("resume_checkpoint".into(), serde_json::json!(checkpoint));
    }

    Ok(body)
}

fn build_opd_training_payload(
    body: serde_json::Value,
    adapter: &str,
    teacher_override: Option<&str>,
    lora_rank: Option<usize>,
    detect_anomaly: bool,
    sampler_segments: Option<usize>,
    rollout_prompt_rendering: Option<&str>,
    checkpoint_interval: Option<usize>,
    resume_checkpoint: Option<&str>,
) -> anyhow::Result<serde_json::Value> {
    if adapter.trim().is_empty() {
        anyhow::bail!("OPD --adapter must not be empty");
    }
    let mut body = match body {
        serde_json::Value::Object(object) => serde_json::Value::Object(object),
        serde_json::Value::Array(prompts) => serde_json::json!({ "prompts": prompts }),
        _ => {
            anyhow::bail!(
                "OPD input must be a JSON request object or a JSON array of prompt objects"
            )
        }
    };
    let object = body
        .as_object_mut()
        .expect("OPD body was normalized to an object");
    object.remove("adapter_name");

    if let Some(teacher) = teacher_override {
        if teacher.trim().is_empty() {
            anyhow::bail!("OPD --teacher must not be empty");
        }
        object.insert("teacher".into(), serde_json::json!(teacher));
    }
    let teacher = object
        .get("teacher")
        .and_then(serde_json::Value::as_str)
        .filter(|teacher| !teacher.trim().is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "OPD input must contain a non-empty teacher alias or be submitted with --teacher"
            )
        })?;
    if teacher.trim() != teacher {
        anyhow::bail!("OPD teacher alias must not have leading or trailing whitespace");
    }

    let config = object
        .entry("config")
        .or_insert_with(|| serde_json::json!({}));
    let config = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("OPD request config must be a JSON object"))?;
    config.insert("output_name".into(), serde_json::json!(adapter));
    if let Some(rank) = lora_rank {
        config.insert("lora_rank".into(), serde_json::json!(rank));
    }
    if detect_anomaly {
        config.insert("detect_anomaly".into(), serde_json::json!(true));
    }
    if let Some(segments) = sampler_segments {
        config.insert("sampler_segments".into(), serde_json::json!(segments));
    }
    if let Some(rendering) = rollout_prompt_rendering {
        config.insert(
            "rollout_prompt_rendering".into(),
            serde_json::json!(rendering),
        );
    }
    if let Some(interval) = checkpoint_interval {
        config.insert("checkpoint_interval".into(), serde_json::json!(interval));
    }
    if let Some(checkpoint) = resume_checkpoint {
        config.insert("resume_checkpoint".into(), serde_json::json!(checkpoint));
    }

    let request: kiln_train::OpdRequest = serde_json::from_value(body.clone())
        .map_err(|error| anyhow::anyhow!("Invalid OPD request: {error}"))?;
    request
        .config
        .validate_runtime_contract()
        .map_err(|error| anyhow::anyhow!("Invalid OPD config: {error:#}"))?;
    if request
        .dataset_path
        .as_deref()
        .is_some_and(|path| path.trim().is_empty())
    {
        anyhow::bail!("OPD dataset_path must not be empty");
    }
    let has_prompts = !request.prompts.is_empty();
    let has_dataset = request.dataset_path.is_some();
    if has_prompts == has_dataset {
        anyhow::bail!(
            "OPD request must contain exactly one non-empty source: prompts or dataset_path"
        );
    }

    Ok(body)
}

/// Run the `train cancel` CLI subcommand.
pub async fn run_train_cancel(url: &str, job_id: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let resp = client
        .delete(format!("{url}/v1/train/queue/{job_id}"))
        .send()
        .await
        .map_err(|e| handle_request_error(url, e))?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().await?;
    if !status.is_success() {
        eprintln!(
            "{} Could not cancel job '{}': {}",
            style("✗").red().bold(),
            job_id,
            render_api_error(&body, status)
        );
        std::process::exit(1);
    }
    let outcome = body
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("cancelled");
    let message = body
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("job cancelled");
    println!(
        "{} Job {} {}: {}",
        style("✓").green().bold(),
        style(job_id).white().bold(),
        outcome,
        message
    );
    Ok(())
}

/// Run the `train status` CLI subcommand.
///
/// With `job_id` set, GETs `/v1/train/jobs/{id}` and prints a one-job summary.
/// Without `job_id`, GETs `/v1/train/status` (overall list) and prints all jobs
/// grouped by state: running first, then queued, then completed/failed.
pub async fn run_train_status(url: &str, job_id: Option<&str>) -> anyhow::Result<()> {
    if let Some(id) = job_id {
        return print_single_job_status(url, id).await;
    }
    print_all_job_statuses(url).await
}

async fn print_single_job_status(url: &str, id: &str) -> anyhow::Result<()> {
    let resp = reqwest::get(format!("{url}/v1/train/jobs/{id}"))
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
    if let Some(seed) = job.get("effective_seed").and_then(|v| v.as_str()) {
        println!("  {} {}", style("Effective seed:").dim(), seed);
    }
    println!("  {} {}%", style("Progress:").dim(), progress_pct);
    if let Some(loss) = job.get("current_loss").and_then(|v| v.as_f64()) {
        println!("  {} {loss:.4}", style("Loss:").dim());
    }
    println!("  {} {}s", style("Elapsed:").dim(), elapsed);
    if let Some(verdict) = job.get("post_eval_verdict").and_then(|v| v.as_str()) {
        // §8.7 promotion gate: color by outcome so a demotion can't be
        // missed in a wall of job summaries.
        let styled = if verdict.contains("promoted") && !verdict.contains("NOT") {
            style(verdict).green()
        } else if verdict.contains(".failed") || verdict.contains("demoted") {
            style(verdict).red()
        } else {
            style(verdict).yellow()
        };
        println!("  {} {}", style("Gate:").dim(), styled);
    }
    if state == "failed" {
        if let Some(err) = job.get("error").and_then(|v| v.as_str()) {
            println!("  {} {}", style("Error:").red().bold(), style(err).red());
        }
    }
    if let Some(checkpoint) = job.get("latest_checkpoint") {
        if let Some(name) = checkpoint.get("resume_checkpoint").and_then(|v| v.as_str()) {
            let step = checkpoint
                .get("global_step")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let total = checkpoint
                .get("total_steps")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!(
                "  {} {} (step {step}/{total})",
                style("Resume checkpoint:").dim(),
                style(name).white()
            );
        }
    }
    if let Some(error) = job.get("checkpoint_error").and_then(|v| v.as_str()) {
        println!("  {} {}", style("Checkpoint warning:").yellow(), error);
    }
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
    let seed = job
        .get("effective_seed")
        .and_then(|v| v.as_str())
        .map(|seed| format!(" seed={seed}"))
        .unwrap_or_default();
    println!(
        "  {} [{}] adapter={}{} {}% ({}s)",
        style(id).white().bold(),
        style_state(state),
        adapter,
        seed,
        progress_pct,
        elapsed
    );
    if state == "failed" {
        if let Some(err) = job.get("error").and_then(|v| v.as_str()) {
            println!(
                "      {} {}",
                style("error:").red().bold(),
                style(err).red()
            );
        }
    }
}

// ===========================================================================
// §10.14 pi + kiln canonical pipeline runners
// ===========================================================================

const PI_PROVIDER_ID: &str = "kiln-local";
const PI_MODEL_ID: &str = "Qwen3.5-4B";

/// Ask a running Kiln what model id it actually serves (`GET /v1/models`),
/// so the pi provider block matches the server's announcement instead of a
/// compiled-in guess. Short timeout — pi-setup must stay instant offline.
async fn probe_kiln_served_model(url: &str) -> anyhow::Result<String> {
    let base = pi_openai_base_url(url);
    let resp = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?
        .get(format!("{base}/models"))
        .send()
        .await?
        .error_for_status()?;
    let body: serde_json::Value = resp.json().await?;
    body.get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("id"))
        .and_then(|id| id.as_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("no models in /v1/models response"))
}

/// Serializes in-process pi-setup merges: embedded runs re-merge before
/// every spawn (up to `max_concurrent_runs` drivers at once) and the
/// dashboard terminal does too — unserialized read-modify-writes here
/// lost edits and let a spawning pi read a half-written config.
static PI_SETUP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Quiet variant of `run_pi_setup` for in-process callers (the embedded
/// dashboard terminal and the embedded run engine): performs the same
/// non-destructive merge into pi's default config location, but logs
/// instead of printing, and returns the models.json path. In-process
/// callers know the served model id — pass it so the provider block
/// matches the live server. No-op (no backup, no write) when the merged
/// config equals what's already on disk — the common case for every
/// embedded run after the first, which would otherwise litter
/// `~/.pi/agent` with a backup pair per run.
pub fn apply_pi_setup_quiet(url: &str, model_id: Option<&str>) -> anyhow::Result<PathBuf> {
    let _guard = PI_SETUP_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let model_id = model_id.unwrap_or(PI_MODEL_ID);
    let path: PathBuf = match kiln_resource::user_home_dir() {
        Some(home) => home.join(".pi").join("agent").join("models.json"),
        None => std::env::temp_dir().join("pi-agent-models.json"),
    };
    let settings_path = pi_settings_path_for_models_path(&path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = settings_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let existing_models = read_json_file_if_exists(&path)?;
    let models = merge_pi_models_config(existing_models.clone(), url, model_id)?;
    let existing_settings = read_json_file_if_exists(&settings_path)?;
    let settings = merge_pi_settings_config(existing_settings.clone(), model_id)?;
    let models_changed = existing_models.as_ref() != Some(&models);
    let settings_changed = existing_settings.as_ref() != Some(&settings);
    if models_changed {
        backup_existing_file(&path)?;
        write_json_pretty(&path, &models)?;
    }
    if settings_changed {
        backup_existing_file(&settings_path)?;
        write_json_pretty(&settings_path, &settings)?;
    }
    if models_changed || settings_changed {
        tracing::info!(models = %path.display(), settings = %settings_path.display(), url = %url, "pi config merged for embedded agent");
    }
    Ok(path)
}

/// §10.14 `kiln pi-setup` — merge kiln into pi's models/settings config.
pub async fn run_pi_setup(url: &str, out: Option<&str>) -> anyhow::Result<()> {
    let default_out: PathBuf = match kiln_resource::user_home_dir() {
        Some(home) => home.join(".pi").join("agent").join("models.json"),
        None => std::env::temp_dir().join("pi-agent-models.json"),
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

    // Handshake with the running server so the provider block carries the
    // model id Kiln actually announces. Silent success when the server is
    // down was the #1 way pi-setup "worked" and pi still couldn't connect.
    let probed_model = match probe_kiln_served_model(url).await {
        Ok(id) => {
            println!(
                "{} Kiln reachable at {} — serving {}",
                style("✓").green().bold(),
                pi_openai_base_url(url),
                style(&id).cyan()
            );
            Some(id)
        }
        Err(err) => {
            println!(
                "{} Kiln is not reachable at {} ({}).",
                style("⚠").yellow().bold(),
                pi_openai_base_url(url),
                format_args!("{err:#}")
            );
            println!(
                "  Writing the config anyway — start the server with {} and pi will connect.",
                style("kiln serve").cyan()
            );
            None
        }
    };
    let model_id = probed_model.as_deref().unwrap_or(PI_MODEL_ID);

    let models_backup = backup_existing_file(&path)?;
    let settings_backup = backup_existing_file(&settings_path)?;

    let models = merge_pi_models_config(read_json_file_if_exists(&path)?, url, model_id)?;
    let settings = merge_pi_settings_config(read_json_file_if_exists(&settings_path)?, model_id)?;
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
        model_id
    );
    println!("  Next: just use pi normally — your sessions become training data.");
    println!(
        "  Optional: {} (needs a teacher model) enables {} for weekly retraining.",
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

/// Write via temp-file + rename so a concurrently *starting* pi never
/// reads a truncated config — embedded runs re-merge before every
/// spawn, and pi children read these files at startup.
fn write_json_pretty(path: &Path, value: &serde_json::Value) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "pi-config".into()),
        std::process::id()
    ));
    std::fs::write(&tmp, bytes).map_err(|err| anyhow::anyhow!("write {}: {err}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|err| {
        let _ = std::fs::remove_file(&tmp);
        anyhow::anyhow!("rename {} -> {}: {err}", tmp.display(), path.display())
    })
}

fn merge_pi_models_config(
    existing: Option<serde_json::Value>,
    url: &str,
    model_id: &str,
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
    providers.insert(
        PI_PROVIDER_ID.to_string(),
        kiln_pi_provider_config(url, model_id),
    );
    root_obj.insert(
        "providers".to_string(),
        serde_json::Value::Object(providers),
    );
    Ok(root)
}

fn merge_pi_settings_config(
    existing: Option<serde_json::Value>,
    model_id: &str,
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
    root_obj.insert("defaultModel".to_string(), serde_json::json!(model_id));
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

fn kiln_pi_provider_config(url: &str, model_id: &str) -> serde_json::Value {
    let display_name = if model_id == PI_MODEL_ID {
        "Qwen 3.5 4B via Kiln".to_string()
    } else {
        format!("{model_id} via Kiln")
    };
    serde_json::json!({
        "baseUrl": pi_openai_base_url(url),
        "api": "openai-completions",
        "apiKey": "dummy",
        "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
        },
        "models": [{
            "id": model_id,
            "name": display_name,
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
    let summary =
        crate::eval_adapter_cli::run_eval_adapter(crate::eval_adapter_cli::EvalAdapterOptions {
            url: url.to_string(),
            adapter: adapter.to_string(),
            tasks: tasks.to_path_buf(),
            seeds,
            request_template: request_template.to_path_buf(),
            scorer: scorer.to_path_buf(),
            output: output.to_path_buf(),
        })
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
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
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
            thinking_budget_tokens,
            thinking_budget_ms,
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
    fn bind_addr_in_use_message_names_addr_and_remedies() {
        let msg = bind_addr_in_use_message("127.0.0.1", 8420);
        assert!(msg.contains("127.0.0.1:8420"));
        assert!(msg.contains("kiln health"));
        assert!(msg.contains("KILN_SERVER_PORT"));
        assert!(msg.contains("[server] port"));
    }

    #[test]
    fn non_loopback_host_warning_skips_loopback_hosts() {
        assert_eq!(non_loopback_host_warning("127.0.0.1"), None);
        assert_eq!(non_loopback_host_warning("localhost"), None);
        assert_eq!(non_loopback_host_warning("::1"), None);
        assert_eq!(non_loopback_host_warning("127.0.0.2"), None);
    }

    #[test]
    fn non_loopback_host_warning_names_host_and_security_model() {
        for host in ["0.0.0.0", "::", "192.168.1.10", "my-box.example.com"] {
            let warning = non_loopback_host_warning(host)
                .unwrap_or_else(|| panic!("expected warning for host {host}"));
            assert!(warning.contains(host));
            assert!(warning.contains("no built-in auth"));
            assert!(warning.contains("Security model"));
        }
    }

    #[test]
    fn checkpoint_boundary_config_output_reports_auto_defaults_and_resume_semantics() {
        let config = crate::config::KilnConfig::default();
        let policy = config.training.checkpoint_boundary_policy().unwrap();
        let output = format_checkpoint_boundary_config(
            policy,
            config.training.checkpoint_boundary_diagnostics(),
        );

        for expected in [
            "Checkpoint boundary mode:",
            "auto (source: default)",
            "8192 tokens (source: default)",
            "Boundary anchor stride:",
            "Boundary cache target:",
            "6 GiB / 6442450944 bytes (source: default)",
            "immutable after startup; restart required to change",
            "Planning identity v3:",
            "changes reject exact resume",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output:?}"
            );
        }
    }

    #[test]
    fn checkpoint_boundary_config_output_reports_explicit_values_and_sources() {
        let config: crate::config::KilnConfig = toml::from_str(
            r#"
[training]
recompute_checkpoint_boundaries = "enabled"
recompute_boundary_threshold_tokens = 4096
checkpoint_boundary_anchor_stride = 3
checkpoint_boundary_cache_gb = 2.5
"#,
        )
        .unwrap();
        let policy = config.training.checkpoint_boundary_policy().unwrap();
        let output = format_checkpoint_boundary_config(
            policy,
            config.training.checkpoint_boundary_diagnostics(),
        );

        for expected in [
            "enabled (source: config_file)",
            "4096 tokens (source: config_file)",
            "3 (explicit; source: config_file)",
            "2.5 GiB / 2684354560 bytes (source: config_file)",
            "includes mode, threshold, stride, and cache target",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output:?}"
            );
        }
    }

    #[test]
    fn actor_prefill_config_output_reports_complete_configured_contract() {
        let defaults = crate::config::KilnConfig::default();
        let default_output = format_actor_prefill_config(&defaults);
        for expected in [
            "Actor cycle token budget: 512 tokens (source: default)",
            "Actor prefill token ceiling: 256 tokens (source: default)",
            "Actor prefill layer ceiling: 4 layers (source: default)",
            "Decode width ceiling: auto (source: default)",
            "Actor cycle idle: 0 ms (source: default)",
            "Streaming prefill mode: auto (source: default)",
            "Streaming prefill threshold: auto (source: default)",
            "Streaming base tile: auto (source: default)",
            "Streaming tape tile: auto (source: default)",
            "Streaming detached full-attention tile: auto (source: default)",
            "fail before model-weight loading",
        ] {
            assert!(
                default_output.contains(expected),
                "missing {expected:?} in {default_output:?}"
            );
        }

        let explicit: crate::config::KilnConfig = toml::from_str(
            r#"
[server]
max_batch_tokens = 640
max_prefill_tokens_per_cycle = 256
max_prefill_layers_per_cycle = 8
max_decode_batch = 16

[batching]
actor_cycle_idle_ms = 75

[streaming_prefill]
mode = "enabled"
threshold_tokens = 256
tile_tokens = 256
tape_tile_tokens = 512
detached_full_attn_tile_tokens = 8192
"#,
        )
        .unwrap();
        let explicit_output = format_actor_prefill_config(&explicit);
        for expected in [
            "Actor cycle token budget: 640 tokens (source: config_file)",
            "Actor prefill token ceiling: 256 tokens (source: config_file)",
            "Actor prefill layer ceiling: 8 layers (source: config_file)",
            "Decode width ceiling: 16 (source: config_file)",
            "Actor cycle idle: 75 ms (source: config_file)",
            "Streaming prefill mode: enabled (source: config_file)",
            "Streaming prefill threshold: 256 tokens (source: config_file)",
            "Streaming base tile: 256 tokens (source: config_file)",
            "Streaming tape tile: 512 tokens (source: config_file)",
            "Streaming detached full-attention tile: 8192 tokens (source: config_file)",
        ] {
            assert!(
                explicit_output.contains(expected),
                "missing {expected:?} in {explicit_output:?}"
            );
        }
    }

    #[test]
    fn actor_prefill_backend_preflight_resolves_rocm_and_rejects_legacy_chunk() {
        let defaults = crate::config::KilnConfig::default();
        let output = format_actor_prefill_backend_config(&defaults, ConfigCheckBackend::Rocm)
            .expect("default ROCm actor-prefill contract should be valid");
        for expected in [
            "Target backend: rocm (hardware-free policy preview)",
            "Batching actor effective: true (source: built_in)",
            "Actor prefill alignment required: true (source: backend_policy)",
            "Effective decode width: 8 rows (source: backend_policy)",
            "Effective streaming dispatch: prompt_tokens_at_least 256 tokens (source: backend_policy)",
            "Effective streaming base tile: 256 tokens (source: backend_policy)",
            "Effective streaming tape tile: 256 tokens (source: backend_policy)",
            "Actor-prefill backend contract: valid (no hardware probe or model load)",
        ] {
            assert!(
                output.contains(expected),
                "missing {expected:?} in {output:?}"
            );
        }

        let legacy: crate::config::KilnConfig = toml::from_str(
            r#"
[server]
max_prefill_tokens_per_cycle = 64
"#,
        )
        .unwrap();
        let error = format_actor_prefill_backend_config(&legacy, ConfigCheckBackend::Rocm)
            .expect_err("legacy 64-token ROCm actor chunk must fail preflight");
        let error = format!("{error:#}");
        assert!(error.contains("invalid rocm actor-prefill contract"));
        assert!(error.contains(
            "server.max_prefill_tokens_per_cycle=64 must equal the backend's effective streaming_prefill.tile_tokens=256"
        ));

        let actor_disabled = toml::from_str::<crate::config::KilnConfig>(
            r#"
[server]
max_prefill_tokens_per_cycle = 64

[batching]
mode = "disabled"
"#,
        )
        .expect_err("batching.mode must remain removed");
        assert!(actor_disabled.to_string().contains("unknown field `mode`"));

        let streaming_disabled: crate::config::KilnConfig = toml::from_str(
            r#"
[streaming_prefill]
mode = "disabled"
"#,
        )
        .unwrap();
        let error =
            format_actor_prefill_backend_config(&streaming_disabled, ConfigCheckBackend::Rocm)
                .expect_err("ROCm must preserve tiled prefill under mandatory actor ownership");
        assert!(format!("{error:#}").contains("enable tiled streaming prefill"));
    }

    #[test]
    fn config_check_backend_cli_is_explicit_and_closed() {
        let cli =
            Cli::try_parse_from(["kiln", "config", "--file", "kiln.toml", "--backend", "rocm"])
                .expect("documented ROCm config preflight should parse");
        match cli.command {
            Some(Commands::ConfigCheck {
                file,
                backend,
                json,
            }) => {
                assert_eq!(file.as_deref(), Some("kiln.toml"));
                assert_eq!(backend, Some(ConfigCheckBackend::Rocm));
                assert!(!json);
            }
            _ => panic!("expected config command"),
        }

        let cli = Cli::try_parse_from(["kiln", "config", "--json", "--backend", "vulkan"])
            .expect("machine-readable Vulkan config preflight should parse");
        match cli.command {
            Some(Commands::ConfigCheck {
                file,
                backend,
                json,
            }) => {
                assert_eq!(file, None);
                assert_eq!(backend, Some(ConfigCheckBackend::Vulkan));
                assert!(json);
            }
            _ => panic!("expected config command"),
        }

        let error = match Cli::try_parse_from(["kiln", "config", "--backend", "auto"]) {
            Ok(_) => panic!("config preflight must require one concrete target backend"),
            Err(error) => error,
        };
        let rendered = error.to_string();
        for backend in ["cpu", "cuda", "rocm", "metal", "vulkan"] {
            assert!(
                rendered.contains(backend),
                "missing {backend:?} in {rendered:?}"
            );
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
    fn merge_pi_models_config_uses_probed_model_id() {
        let merged = merge_pi_models_config(None, "http://localhost:8420", "my-served-id").unwrap();
        let provider = &merged["providers"][PI_PROVIDER_ID];
        assert_eq!(provider["baseUrl"], "http://localhost:8420/v1");
        assert_eq!(provider["models"][0]["id"], "my-served-id");
        assert_eq!(provider["models"][0]["name"], "my-served-id via Kiln");
    }

    #[test]
    fn merge_pi_models_config_keeps_existing_providers_and_default_display_name() {
        let existing = json!({"providers": {"other": {"baseUrl": "http://elsewhere"}}});
        let merged =
            merge_pi_models_config(Some(existing), "http://localhost:8420/", "Qwen3.5-4B").unwrap();
        assert!(merged["providers"]["other"].is_object());
        let provider = &merged["providers"][PI_PROVIDER_ID];
        assert_eq!(provider["models"][0]["id"], "Qwen3.5-4B");
        assert_eq!(provider["models"][0]["name"], "Qwen 3.5 4B via Kiln");
    }

    #[test]
    fn merge_pi_settings_config_sets_default_model_to_served_id() {
        let merged = merge_pi_settings_config(None, "custom-id").unwrap();
        assert_eq!(merged["defaultProvider"], PI_PROVIDER_ID);
        assert_eq!(merged["defaultModel"], "custom-id");
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
            "true",
            "--thinking-budget-tokens",
            "96",
            "--thinking-budget-ms",
            "1500",
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
            thinking_budget_tokens,
            thinking_budget_ms,
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
        assert!(thinking);
        assert_eq!(thinking_budget_tokens, Some(ThinkingBudgetArg::Limited(96)));
        assert_eq!(thinking_budget_ms, Some(ThinkingBudgetArg::Limited(1500)));
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
    fn rollout_thinking_budget_flags_support_inherit_and_unlimited() {
        let inherited = Cli::parse_from([
            "kiln",
            "rollout-generate",
            "--adapter",
            "base",
            "--tasks",
            "tasks.jsonl",
            "--request-template",
            "request.json",
            "--scorer",
            "./score.py",
        ]);
        let Some(Commands::RolloutGenerate {
            thinking_budget_tokens,
            thinking_budget_ms,
            ..
        }) = inherited.command
        else {
            panic!("expected rollout-generate command");
        };
        assert_eq!(thinking_budget_tokens, None);
        assert_eq!(thinking_budget_ms, None);

        let unlimited = Cli::parse_from([
            "kiln",
            "rollout-generate",
            "--adapter",
            "base",
            "--thinking",
            "true",
            "--thinking-budget-tokens",
            "unlimited",
            "--thinking-budget-ms",
            "UNLIMITED",
            "--tasks",
            "tasks.jsonl",
            "--request-template",
            "request.json",
            "--scorer",
            "./score.py",
        ]);
        let Some(Commands::RolloutGenerate {
            thinking_budget_tokens,
            thinking_budget_ms,
            ..
        }) = unlimited.command
        else {
            panic!("expected rollout-generate command");
        };
        assert_eq!(thinking_budget_tokens, Some(ThinkingBudgetArg::Unlimited));
        assert_eq!(thinking_budget_ms, Some(ThinkingBudgetArg::Unlimited));
    }

    #[test]
    fn thinking_budget_arg_uses_api_shaped_json() {
        assert_eq!(
            "0".parse::<ThinkingBudgetArg<usize>>().unwrap(),
            ThinkingBudgetArg::Limited(0)
        );
        assert_eq!(
            "unlimited".parse::<ThinkingBudgetArg<usize>>().unwrap(),
            ThinkingBudgetArg::Unlimited
        );
        assert!("-1".parse::<ThinkingBudgetArg<usize>>().is_err());
        assert_eq!(
            serde_json::to_value(ThinkingBudgetArg::<usize>::Unlimited).unwrap(),
            serde_json::Value::Null
        );
        assert_eq!(
            serde_json::to_value(ThinkingBudgetArg::Limited(0usize)).unwrap(),
            serde_json::json!(0)
        );
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
                prompt_messages: vec![kiln_train::ChatMessage::new("user", "run it")],
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
            Some(2e-4),
            3,
            Some(8),
            "fail",
            false,
            None,
            false,
            None,
            None,
        );

        assert_eq!(body["config"]["output_name"], "sft-adapter");
        assert_eq!(body["config"]["training_profile"], "native_online_lora_v1");
        assert_eq!(body["config"]["learning_rate"], 2e-4);
        assert_eq!(body["config"]["epochs"], 3);
        assert_eq!(body["config"]["lora_rank"], 8);
        assert!(body.get("adapter_name").is_none());
        assert!(body.get("num_epochs").is_none());
    }

    #[test]
    fn build_sft_training_payload_omits_unset_lora_rank() {
        let body = build_sft_training_payload(
            vec![],
            "sft-adapter",
            Some(1e-4),
            1,
            None,
            "fail",
            false,
            None,
            false,
            None,
            None,
        );

        assert_eq!(body["config"]["output_name"], "sft-adapter");
        assert_eq!(body["config"]["training_profile"], "native_online_lora_v1");
        assert!(body["config"].get("lora_rank").is_none());
        assert!(body["config"].get("adapter_smoke_test").is_none());
    }

    #[test]
    fn build_sft_training_payload_omits_unset_learning_rate() {
        // No --lr → no learning_rate key, so the server resolves the
        // per-optimizer default instead of an AdamW-era pin.
        let body = build_sft_training_payload(
            vec![],
            "sft-adapter",
            None,
            1,
            None,
            "fail",
            false,
            None,
            false,
            None,
            None,
        );

        assert!(body["config"].get("learning_rate").is_none());
        assert_eq!(body["config"]["epochs"], 1);
    }

    #[test]
    fn build_sft_training_payload_sets_adapter_smoke_test_when_requested() {
        let body = build_sft_training_payload(
            vec![],
            "sft-adapter",
            Some(1e-4),
            1,
            None,
            "fail",
            true,
            None,
            true,
            None,
            None,
        );

        assert_eq!(body["config"]["adapter_smoke_test"], true);
        assert_eq!(body["config"]["detect_anomaly"], true);
    }

    #[test]
    fn build_sft_training_payload_sets_exact_resume_options() {
        let body = build_sft_training_payload(
            vec![],
            "sft-adapter",
            None,
            2,
            None,
            "skip",
            false,
            None,
            false,
            Some(25),
            Some("sft-adapter-checkpoint-step-00000025.kiln-checkpoint"),
        );

        assert_eq!(body["config"]["checkpoint_interval"], 25);
        assert_eq!(body["config"]["invalid_row_policy"], "skip");
        assert_eq!(
            body["config"]["resume_checkpoint"],
            "sft-adapter-checkpoint-step-00000025.kiln-checkpoint"
        );
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

        let smoke_prompts = vec!["Return one word.".to_string()];
        let body = build_grpo_training_payload(
            body,
            "grpo-adapter",
            Some(16),
            true,
            Some(&smoke_prompts),
            true,
            true,
            Some(25),
            Some("grpo-adapter-checkpoint-step-00000025.kiln-checkpoint"),
        )
        .unwrap();

        assert_eq!(body["config"]["output_name"], "grpo-adapter");
        assert_eq!(body["config"]["learning_rate"], 5e-5);
        assert_eq!(body["config"]["lora_rank"], 16);
        assert_eq!(body["config"]["adapter_smoke_test"], true);
        assert_eq!(
            body["config"]["adapter_smoke_prompts"],
            serde_json::json!(smoke_prompts)
        );
        assert_eq!(body["config"]["shared_prefix_reference"], false);
        assert_eq!(body["config"]["detect_anomaly"], true);
        assert_eq!(body["config"]["checkpoint_interval"], 25);
        assert_eq!(
            body["config"]["resume_checkpoint"],
            "grpo-adapter-checkpoint-step-00000025.kiln-checkpoint"
        );
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

        let body = build_grpo_training_payload(
            body,
            "grpo-adapter",
            None,
            false,
            None,
            false,
            false,
            None,
            None,
        )
        .unwrap();

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
        let body = build_grpo_jsonl_training_payload(
            path.to_str().unwrap(),
            "grpo-jsonl",
            Some(12),
            true,
            None,
            false,
            false,
            Some(2),
            Some("grpo-jsonl-checkpoint-step-00000002.kiln-checkpoint"),
        )
        .unwrap();
        assert!(body.get("groups").is_none());
        assert_eq!(body["config"]["output_name"], "grpo-jsonl");
        assert_eq!(body["config"]["lora_rank"], 12);
        assert_eq!(body["config"]["adapter_smoke_test"], true);
        assert_eq!(body["config"]["checkpoint_interval"], 2);
        assert_eq!(
            body["config"]["resume_checkpoint"],
            "grpo-jsonl-checkpoint-step-00000002.kiln-checkpoint"
        );
        assert!(body["dataset_path"].as_str().unwrap().ends_with(".jsonl"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn build_sft_jsonl_training_payload_uses_explicit_native_profile() {
        let path =
            std::env::temp_dir().join(format!("kiln-cli-sft-jsonl-{}.jsonl", std::process::id()));
        std::fs::write(
            &path,
            r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]}"#,
        )
        .unwrap();

        let body = build_sft_jsonl_training_payload(
            path.to_str().unwrap(),
            "sft-jsonl",
            None,
            2,
            Some(8),
            "fail",
            false,
            None,
            false,
            None,
            None,
        )
        .unwrap();
        assert_eq!(body["config"]["training_profile"], "native_online_lora_v1");
        assert_eq!(body["config"]["invalid_row_policy"], "fail");
        assert!(body["dataset_path"].as_str().unwrap().ends_with(".jsonl"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn build_opd_training_payload_applies_exact_resume_overrides() {
        let body = build_opd_training_payload(
            json!({
                "prompts": [{
                    "messages": [{"role": "user", "content": "Explain the result"}]
                }],
                "teacher": "stale-teacher",
                "config": {"top_k": 16, "seed": 73, "output_name": "stale-output"}
            }),
            "opd-adapter",
            Some("teacher-v2"),
            Some(24),
            true,
            Some(12),
            Some("chat_template"),
            Some(25),
            Some("opd-adapter-checkpoint-step-00000025.kiln-checkpoint"),
        )
        .unwrap();

        assert_eq!(body["teacher"], "teacher-v2");
        assert_eq!(body["config"]["output_name"], "opd-adapter");
        assert_eq!(body["config"]["lora_rank"], 24);
        assert_eq!(body["config"]["detect_anomaly"], true);
        assert_eq!(body["config"]["sampler_segments"], 12);
        assert_eq!(body["config"]["rollout_prompt_rendering"], "chat_template");
        assert_eq!(body["config"]["top_k"], 16);
        assert_eq!(body["config"]["seed"], 73);
        assert_eq!(body["config"]["checkpoint_interval"], 25);
        assert_eq!(
            body["config"]["resume_checkpoint"],
            "opd-adapter-checkpoint-step-00000025.kiln-checkpoint"
        );
        serde_json::from_value::<kiln_train::OpdRequest>(body).unwrap();
    }

    #[test]
    fn build_opd_training_payload_wraps_prompt_arrays_and_requires_one_source() {
        let prompt = json!({
            "messages": [{"role": "user", "content": "Summarize this"}]
        });
        let body = build_opd_training_payload(
            json!([prompt.clone()]),
            "opd-adapter",
            Some("teacher-v1"),
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(body["prompts"], json!([prompt]));
        assert_eq!(body["teacher"], "teacher-v1");
        assert_eq!(body["config"]["output_name"], "opd-adapter");

        let dataset = build_opd_training_payload(
            json!({
                "teacher": "teacher-v1",
                "dataset_path": "/data/opd.jsonl",
                "config": {"training_mode": "off_policy"}
            }),
            "opd-adapter",
            None,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(dataset["teacher"], "teacher-v1");
        assert_eq!(dataset["dataset_path"], "/data/opd.jsonl");
        assert_eq!(dataset["config"]["training_mode"], "off_policy");

        let neither = build_opd_training_payload(
            json!({"teacher": "teacher-v1"}),
            "opd-adapter",
            None,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(neither.to_string().contains("exactly one non-empty source"));

        let both = build_opd_training_payload(
            json!({
                "teacher": "teacher-v1",
                "prompts": [{"messages": [{"role": "user", "content": "hello"}]}],
                "dataset_path": "/data/opd.jsonl"
            }),
            "opd-adapter",
            None,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(both.to_string().contains("exactly one non-empty source"));
    }

    #[test]
    fn build_opd_training_payload_rejects_missing_teacher_and_invalid_config() {
        let prompts = json!([{
            "messages": [{"role": "user", "content": "hello"}]
        }]);
        let missing = build_opd_training_payload(
            prompts.clone(),
            "opd-adapter",
            None,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(missing.to_string().contains("--teacher"));

        let invalid = build_opd_training_payload(
            json!({
                "teacher": "teacher-v1",
                "prompts": prompts,
                "config": {"checkpoint_interval": 0}
            }),
            "opd-adapter",
            None,
            None,
            false,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(invalid.to_string().contains("checkpoint_interval"));
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
    fn parses_sft_invalid_row_policy_and_rejects_unknown_values() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "train",
            "sft",
            "--file",
            "rows.jsonl",
            "--invalid-row-policy",
            "skip",
            "--detect-anomaly",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Train(TrainCommands::Sft {
                invalid_row_policy,
                detect_anomaly,
                ..
            })) => {
                assert_eq!(invalid_row_policy, "skip");
                assert!(detect_anomaly);
            }
            other => panic!("expected Train(Sft), got {:?}", other.is_some()),
        }
        assert!(
            Cli::try_parse_from([
                "kiln",
                "train",
                "sft",
                "--file",
                "rows.jsonl",
                "--invalid-row-policy",
                "drop",
            ])
            .is_err()
        );
    }

    #[test]
    fn adapter_smoke_prompt_files_are_explicit_validated_inputs() {
        use clap::Parser;

        assert!(
            Cli::try_parse_from([
                "kiln",
                "train",
                "sft",
                "--file",
                "rows.jsonl",
                "--adapter-smoke-prompts-file",
                "prompts.json",
            ])
            .is_err()
        );
        Cli::try_parse_from([
            "kiln",
            "train",
            "sft",
            "--file",
            "rows.jsonl",
            "--adapter-smoke-test",
            "--adapter-smoke-prompts-file",
            "prompts.json",
        ])
        .expect("explicit smoke test should accept a prompt file");

        let path = std::env::temp_dir().join(format!(
            "kiln-cli-adapter-smoke-prompts-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, r#"["Return one word.","Say done."]"#).unwrap();
        assert_eq!(
            load_adapter_smoke_prompts(path.to_str()).unwrap(),
            Some(vec!["Return one word.".into(), "Say done.".into()])
        );

        std::fs::write(&path, "Return one word.\nKeep this second line.").unwrap();
        assert_eq!(
            load_adapter_smoke_prompts(path.to_str()).unwrap(),
            Some(vec!["Return one word.\nKeep this second line.".into()])
        );

        std::fs::write(&path, "[]").unwrap();
        let error = load_adapter_smoke_prompts(path.to_str()).unwrap_err();
        assert!(error.to_string().contains("empty JSON array"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn parses_hf_sft_export_and_requires_exactly_one_source() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "train",
            "hf",
            "export-sft",
            "--file",
            "/srv/data/rows.jsonl",
            "--name",
            "portable-run",
            "--output",
            "artifacts/portable.tar.gz",
            "--invalid-row-policy",
            "skip",
            "--input-adapter",
            "seed-adapter",
            "--split-manifest",
            "split.json",
            "--keep-server-copy",
        ])
        .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Hf(HfTrainCommands::ExportSft {
                file,
                dataset,
                name,
                output,
                invalid_row_policy,
                input_adapter,
                split_manifest,
                keep_server_copy,
                url,
            }))) => {
                assert_eq!(file.as_deref(), Some("/srv/data/rows.jsonl"));
                assert!(dataset.is_none());
                assert_eq!(name, "portable-run");
                assert_eq!(output, Some(PathBuf::from("artifacts/portable.tar.gz")));
                assert_eq!(invalid_row_policy, "skip");
                assert_eq!(input_adapter.as_deref(), Some("seed-adapter"));
                assert_eq!(split_manifest, Some(PathBuf::from("split.json")));
                assert!(keep_server_copy);
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected Train(Hf(ExportSft)), got {:?}", other.is_some()),
        }

        assert!(
            Cli::try_parse_from([
                "kiln",
                "train",
                "hf",
                "export-sft",
                "--name",
                "missing-source",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "kiln",
                "train",
                "hf",
                "export-sft",
                "--file",
                "rows.jsonl",
                "--dataset",
                "corrections:active",
                "--name",
                "two-sources",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_hf_grpo_export_and_requires_a_file() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "train",
            "hf",
            "export-grpo",
            "--file",
            "/srv/data/recorded.jsonl",
            "--name",
            "recorded-run",
            "--output",
            "artifacts/recorded.tar.gz",
            "--input-adapter",
            "behavior-adapter",
            "--split-manifest",
            "split.json",
            "--keep-server-copy",
        ])
        .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Hf(HfTrainCommands::ExportGrpo {
                file,
                name,
                output,
                input_adapter,
                split_manifest,
                keep_server_copy,
                url,
            }))) => {
                assert_eq!(file, "/srv/data/recorded.jsonl");
                assert_eq!(name, "recorded-run");
                assert_eq!(output, Some(PathBuf::from("artifacts/recorded.tar.gz")));
                assert_eq!(input_adapter.as_deref(), Some("behavior-adapter"));
                assert_eq!(split_manifest, Some(PathBuf::from("split.json")));
                assert!(keep_server_copy);
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected Train(Hf(ExportGrpo)), got {:?}", other.is_some()),
        }

        assert!(
            Cli::try_parse_from([
                "kiln",
                "train",
                "hf",
                "export-grpo",
                "--name",
                "missing-file",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_hf_dataset_import_list_and_delete_commands() {
        use clap::Parser;
        let export = Cli::try_parse_from([
            "kiln",
            "train",
            "hf",
            "export-sft",
            "--dataset",
            "corrections:active",
            "--name",
            "corrections-01",
        ])
        .unwrap();
        assert!(matches!(
            export.command,
            Some(Commands::Train(TrainCommands::Hf(
                HfTrainCommands::ExportSft {
                    file: None,
                    dataset: Some(_),
                    ..
                }
            )))
        ));

        let import = Cli::try_parse_from([
            "kiln",
            "train",
            "hf",
            "import-peft",
            "--bundle",
            "corrections-01.kiln-hf",
            "--name",
            "corrections.peft-01",
        ])
        .unwrap();
        assert!(matches!(
            import.command,
            Some(Commands::Train(TrainCommands::Hf(
                HfTrainCommands::ImportPeft {
                    bundle,
                    name,
                    url,
                }
            ))) if bundle == PathBuf::from("corrections-01.kiln-hf")
                && name == "corrections.peft-01"
                && url == "http://localhost:8420"
        ));

        let list = Cli::try_parse_from(["kiln", "train", "hf", "list", "--json"]).unwrap();
        assert!(matches!(
            list.command,
            Some(Commands::Train(TrainCommands::Hf(HfTrainCommands::List {
                json: true,
                ..
            })))
        ));

        let delete =
            Cli::try_parse_from(["kiln", "train", "hf", "delete", "--name", "corrections-01"])
                .unwrap();
        assert!(matches!(
            delete.command,
            Some(Commands::Train(TrainCommands::Hf(HfTrainCommands::Delete {
                name,
                export_sha256: None,
                ..
            }))) if name == "corrections-01"
        ));

        let conditional_delete = Cli::try_parse_from([
            "kiln",
            "train",
            "hf",
            "delete",
            "--name",
            "corrections-01",
            "--export-sha256",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ])
        .unwrap();
        assert!(matches!(
            conditional_delete.command,
            Some(Commands::Train(TrainCommands::Hf(HfTrainCommands::Delete {
                export_sha256: Some(identity),
                ..
            }))) if identity.starts_with("sha256:")
        ));
    }

    #[test]
    fn parses_grpo_exact_resume_options() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "train",
            "grpo",
            "--file",
            "groups.jsonl",
            "--adapter",
            "demo",
            "--checkpoint-interval",
            "25",
            "--no-shared-prefix-reference",
            "--detect-anomaly",
            "--resume-checkpoint",
            "demo-checkpoint-step-00000025.kiln-checkpoint",
        ])
        .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Grpo {
                checkpoint_interval,
                resume_checkpoint,
                detect_anomaly,
                no_shared_prefix_reference,
                ..
            })) => {
                assert_eq!(
                    checkpoint_interval.map(std::num::NonZeroUsize::get),
                    Some(25)
                );
                assert_eq!(
                    resume_checkpoint.as_deref(),
                    Some("demo-checkpoint-step-00000025.kiln-checkpoint")
                );
                assert!(detect_anomaly);
                assert!(no_shared_prefix_reference);
            }
            other => panic!("expected Train(Grpo), got {:?}", other.is_some()),
        }

        let err = Cli::try_parse_from([
            "kiln",
            "train",
            "grpo",
            "--file",
            "groups.jsonl",
            "--adapter",
            "demo",
            "--checkpoint-interval",
            "0",
        ])
        .err()
        .expect("zero checkpoint interval must be rejected");
        assert!(err.to_string().contains("invalid value '0'"));
    }

    #[test]
    fn parses_opd_exact_resume_options() {
        use clap::Parser;
        let cli = Cli::try_parse_from([
            "kiln",
            "train",
            "opd",
            "--file",
            "opd-request.json",
            "--adapter",
            "demo",
            "--teacher",
            "teacher-v1",
            "--lora-rank",
            "24",
            "--checkpoint-interval",
            "25",
            "--sampler-segments",
            "12",
            "--rollout-prompt-rendering",
            "chat_template",
            "--detect-anomaly",
            "--resume-checkpoint",
            "demo-checkpoint-step-00000025.kiln-checkpoint",
        ])
        .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Opd {
                file,
                adapter,
                teacher,
                lora_rank,
                detect_anomaly,
                sampler_segments,
                rollout_prompt_rendering,
                checkpoint_interval,
                resume_checkpoint,
                url,
            })) => {
                assert_eq!(file, "opd-request.json");
                assert_eq!(adapter, "demo");
                assert_eq!(teacher.as_deref(), Some("teacher-v1"));
                assert_eq!(lora_rank, Some(24));
                assert!(detect_anomaly);
                assert_eq!(sampler_segments.map(std::num::NonZeroUsize::get), Some(12));
                assert_eq!(rollout_prompt_rendering.as_deref(), Some("chat_template"));
                assert_eq!(
                    checkpoint_interval.map(std::num::NonZeroUsize::get),
                    Some(25)
                );
                assert_eq!(
                    resume_checkpoint.as_deref(),
                    Some("demo-checkpoint-step-00000025.kiln-checkpoint")
                );
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected Train(Opd), got {:?}", other.is_some()),
        }

        let err = Cli::try_parse_from([
            "kiln",
            "train",
            "opd",
            "--file",
            "opd-request.json",
            "--checkpoint-interval",
            "0",
        ])
        .err()
        .expect("zero checkpoint interval must be rejected");
        assert!(err.to_string().contains("invalid value '0'"));
    }

    #[test]
    fn parses_cancel_subcommand() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["kiln", "train", "cancel", "--job-id", "abc"])
            .expect("parse failed");
        match cli.command {
            Some(Commands::Train(TrainCommands::Cancel { job_id, url })) => {
                assert_eq!(job_id, "abc");
                assert_eq!(url, "http://localhost:8420");
            }
            other => panic!("expected Train(Cancel), got {:?}", other.is_some()),
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
