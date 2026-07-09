//! `kiln-eval` — command-line driver for the eval API.
//!
//! Talks to a running `kiln serve` instance over HTTP. Three sub-commands:
//!
//!   - `kiln-eval list [--server URL]` — list registered suites
//!   - `kiln-eval register --file PATH [--force] [--server URL]` —
//!       upload a suite JSON file to the server
//!   - `kiln-eval run [--suite NAME | --file PATH] [--adapter NAME]
//!                    [--include-baseline] [--max-tokens N] [--temperature F]
//!                    [--watch] [--json] [--server URL]` —
//!       submit an eval and (optionally) wait for results
//!   - `kiln-eval compare --suite NAME --adapter NAME [NAME ...] [--watch]`
//!       run a compare across multiple adapters and print a head-to-head
//!   - `kiln-eval trace-suite --input TRACE.jsonl --output SUITE.json`
//!       sample production tool-call turns from a generic JSONL export
//!   - `kiln-eval panel-suite --suite FULL.json --max-examples N`
//!       build a weighted stratified fast panel from a full eval suite
//!
//! All commands respect `KILN_SERVER_URL` and the `--server` flag (default
//! `http://localhost:8420`). Output is human-readable by default; pass
//! `--json` to emit the raw `EvalResult`.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use kiln_eval::qwen3::extract_first_tool_call;
use kiln_eval::{
    EvalBudgetOverride, EvalChatMessage, EvalCompareSpec, EvalExample, EvalGenerationParams,
    EvalSuite, ProductionTraceError, ProductionTraceFormat, ProductionTraceInputLine,
    ProductionTraceSuiteConfig, SuiteResult,
};
use kiln_server::cli::ThinkingBudgetArg;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const DEFAULT_SERVER_URL: &str = "http://localhost:8420";

#[derive(Parser, Debug)]
#[command(name = "kiln-eval", about = "Run kiln evals from the command line")]
struct Cli {
    /// Kiln server base URL.
    #[arg(long, env = "KILN_SERVER_URL", default_value = DEFAULT_SERVER_URL)]
    server: String,
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List registered suites.
    List,
    /// Upload a suite JSON file to the server registry.
    Register {
        /// Path to a JSON file matching `EvalSuite`.
        #[arg(long)]
        file: PathBuf,
        /// Overwrite an existing suite by the same name.
        #[arg(long)]
        force: bool,
    },
    /// Run an eval against an adapter.
    Run(RunArgs),
    /// Run a head-to-head comparison across multiple adapters.
    Compare(CompareArgs),
    /// Quick-eval helper: build a single-example suite from CLI flags
    /// (great for sanity checks during development).
    Probe(ProbeArgs),
    /// Build a tool-call eval suite from production trace JSONL.
    TraceSuite(TraceSuiteArgs),
    /// Build a weighted stratified fast panel from an existing EvalSuite.
    PanelSuite(PanelSuiteArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Registered suite name. Mutually exclusive with --file.
    #[arg(long)]
    suite: Option<String>,
    /// Path to a JSON file matching `EvalSuite`. Mutually exclusive with --suite.
    #[arg(long)]
    file: Option<PathBuf>,
    /// Adapter to evaluate. Pass `""` (empty string) to mean the base model.
    /// When omitted, the server uses the currently-active adapter.
    #[arg(long)]
    adapter: Option<String>,
    /// Override the suite-wide temperature.
    #[arg(long)]
    temperature: Option<f32>,
    /// Override the suite-wide max_tokens.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Thinking token budget: omit to inherit, use 0 to close immediately, or `unlimited` for no limit.
    #[arg(long, value_name = "TOKENS|unlimited")]
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    /// Thinking decode-time budget: omit to inherit, use 0 to close immediately, or `unlimited` for no limit.
    #[arg(long, value_name = "MILLISECONDS|unlimited")]
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
    /// Wait for the job to finish, polling every 2s.
    #[arg(long)]
    watch: bool,
    /// Emit raw JSON on completion instead of the human-readable summary.
    #[arg(long)]
    json: bool,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    /// Registered suite name.
    #[arg(long)]
    suite: String,
    /// Adapters to compare. Pass `""` for the base model. Repeat the flag.
    #[arg(long = "adapter", required = true)]
    adapters: Vec<String>,
    /// Wait for the job to finish.
    #[arg(long)]
    watch: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Parser, Debug)]
struct ProbeArgs {
    /// User prompt that becomes the only example's content.
    #[arg(long)]
    prompt: String,
    /// Expected target string for the scorer.
    #[arg(long)]
    target: String,
    /// Scorer kind: `exact_match` | `contains` | `numeric` | `regex`.
    #[arg(long, default_value = "exact_match")]
    scorer: String,
    /// Adapter to probe.
    #[arg(long)]
    adapter: Option<String>,
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Thinking token budget: omit to inherit, use 0 to close immediately, or `unlimited` for no limit.
    #[arg(long, value_name = "TOKENS|unlimited")]
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    /// Thinking decode-time budget: omit to inherit, use 0 to close immediately, or `unlimited` for no limit.
    #[arg(long, value_name = "MILLISECONDS|unlimited")]
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
    #[arg(long)]
    json: bool,
}

#[derive(Parser, Debug)]
struct TraceSuiteArgs {
    /// Production trace JSONL file. Repeat --input to sample one suite across
    /// multiple exports.
    #[arg(long, required = true)]
    input: Vec<PathBuf>,
    /// Output EvalSuite JSON file. Required unless --stdout is set.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optional JSON sidecar with sampling config, skip counts, tool
    /// histograms, and sampled example provenance.
    #[arg(long)]
    stats_output: Option<PathBuf>,
    /// Suite name to write into the EvalSuite.
    #[arg(long)]
    suite_name: String,
    /// Optional suite description.
    #[arg(long)]
    description: Option<String>,
    /// Input format: auto | prompt_chosen_jsonl | openai_jsonl |
    /// openai_trajectory_jsonl | anthropic_jsonl |
    /// anthropic_trajectory_jsonl. `auto` inspects each row: explicit
    /// per-row `format` labels win; OpenAI or Anthropic `messages` arrays
    /// that look like full trajectories (tool-role messages, tool_use
    /// blocks, or multiple tool-calling assistant turns) get per-turn
    /// materialization; other `messages` rows score the final assistant.
    #[arg(long, default_value = "auto", value_parser = parse_trace_format)]
    format: ProductionTraceFormat,
    /// Reservoir sample size. Omit to use the production-trace default.
    #[arg(long)]
    max_examples: Option<usize>,
    /// Deterministic sampling seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Prompt size guard in chars.
    #[arg(long)]
    max_prompt_chars: Option<usize>,
    /// Target size guard in chars.
    #[arg(long)]
    max_target_chars: Option<usize>,
    /// Dedupe exact prompt+target pairs. Off by default because repetition
    /// is workload-frequency signal.
    #[arg(long)]
    dedupe: bool,
    /// Require Qwen3.5 native XML output when scoring. Off by default so the
    /// eval measures semantic tool-call correctness, not format differences.
    #[arg(long)]
    require_qwen_xml: bool,
    /// Override suite generation max_tokens.
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Override suite generation temperature.
    #[arg(long)]
    temperature: Option<f32>,
    /// Thinking token budget: omit to inherit, use 0 to close immediately, or `unlimited` for no limit.
    #[arg(long, value_name = "TOKENS|unlimited")]
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    /// Thinking decode-time budget: omit to inherit, use 0 to close immediately, or `unlimited` for no limit.
    #[arg(long, value_name = "MILLISECONDS|unlimited")]
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
    /// Print the generated suite JSON to stdout instead of writing --output.
    #[arg(long)]
    stdout: bool,
}

#[derive(Parser, Debug)]
struct PanelSuiteArgs {
    /// Full EvalSuite JSON file to sample from.
    #[arg(long)]
    suite: PathBuf,
    /// Output panel EvalSuite JSON file. Required unless --stdout is set.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optional JSON sidecar with stratum populations, sample counts, and
    /// selected example IDs.
    #[arg(long)]
    stats_output: Option<PathBuf>,
    /// Suite name for the generated panel. Defaults to
    /// `<source-suite>-panel-<max-examples>`.
    #[arg(long)]
    suite_name: Option<String>,
    /// Maximum examples to keep in the fast panel.
    #[arg(long, default_value_t = 200)]
    max_examples: usize,
    /// Deterministic sample seed.
    #[arg(long, default_value_t = 17)]
    seed: u64,
    /// Print the generated panel JSON to stdout instead of writing --output.
    #[arg(long)]
    stdout: bool,
}

fn parse_trace_format(s: &str) -> std::result::Result<ProductionTraceFormat, String> {
    match s {
        "auto" => Ok(ProductionTraceFormat::Auto),
        "prompt_chosen_jsonl" | "prompt-chosen-jsonl" => {
            Ok(ProductionTraceFormat::PromptChosenJsonl)
        }
        "openai_jsonl" | "openai-jsonl" => Ok(ProductionTraceFormat::OpenAiJsonl),
        "openai_trajectory_jsonl" | "openai-trajectory-jsonl" | "openai_trajectory" => {
            Ok(ProductionTraceFormat::OpenAiTrajectoryJsonl)
        }
        "anthropic_jsonl" | "anthropic-jsonl" => Ok(ProductionTraceFormat::AnthropicJsonl),
        "anthropic_trajectory_jsonl" | "anthropic-trajectory-jsonl" | "anthropic_trajectory" => {
            Ok(ProductionTraceFormat::AnthropicTrajectoryJsonl)
        }
        // `jsonl` used to silently alias anthropic_jsonl, quietly skipping
        // every non-Anthropic row. Too ambiguous to guess.
        "jsonl" => Err(
            "ambiguous trace format `jsonl`; pass an explicit format (auto | prompt_chosen_jsonl \
             | openai_jsonl | openai_trajectory_jsonl | anthropic_jsonl | anthropic_trajectory_jsonl)"
                .to_string(),
        ),
        other => Err(format!(
            "unknown trace format `{other}` (try auto | prompt_chosen_jsonl | openai_jsonl | openai_trajectory_jsonl | anthropic_jsonl | anthropic_trajectory_jsonl)"
        )),
    }
}

#[derive(Debug, Deserialize)]
struct EvalRunResponse {
    job_id: String,
    state: String,
    #[allow(dead_code)]
    message: String,
}

#[derive(Debug, Deserialize, serde::Serialize)]
struct EvalResultPayload {
    job_id: String,
    state: String,
    runs: Vec<SuiteResult>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    progress: Option<kiln_eval::EvalProgress>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let server = cli.server.trim_end_matches('/').to_string();

    match cli.cmd {
        Command::List => cmd_list(&client, &server).await,
        Command::Register { file, force } => cmd_register(&client, &server, &file, force).await,
        Command::Run(args) => cmd_run(&client, &server, args).await,
        Command::Compare(args) => cmd_compare(&client, &server, args).await,
        Command::Probe(args) => cmd_probe(&client, &server, args).await,
        Command::TraceSuite(args) => cmd_trace_suite(args),
        Command::PanelSuite(args) => cmd_panel_suite(args),
    }
}

async fn cmd_list(client: &reqwest::Client, server: &str) -> Result<()> {
    let resp = client
        .get(format!("{server}/v1/eval/suites"))
        .send()
        .await
        .context("GET /v1/eval/suites")?;
    if !resp.status().is_success() {
        bail!("server returned {}: {}", resp.status(), resp.text().await?);
    }
    // `EvalSuiteSummary::default_scorer_kind` is `&'static str` which can't
    // be deserialized into an owned struct, so consume the response as
    // serde_json::Value and walk it manually.
    let body: serde_json::Value = resp.json().await?;
    let suites = body
        .get("suites")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if suites.is_empty() {
        println!(
            "(no registered suites — register one with `kiln-eval register --file SUITE.json`)"
        );
        return Ok(());
    }
    println!(
        "{:<32}  {:<10}  {:<18}  description",
        "name", "examples", "scorer"
    );
    for s in suites {
        println!(
            "{:<32}  {:<10}  {:<18}  {}",
            s.get("name").and_then(|v| v.as_str()).unwrap_or(""),
            s.get("num_examples").and_then(|v| v.as_u64()).unwrap_or(0),
            s.get("default_scorer_kind")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            s.get("description").and_then(|v| v.as_str()).unwrap_or(""),
        );
    }
    Ok(())
}

async fn cmd_register(
    client: &reqwest::Client,
    server: &str,
    file: &PathBuf,
    force: bool,
) -> Result<()> {
    let bytes = std::fs::read(file).with_context(|| format!("read {}", file.display()))?;
    let suite: EvalSuite = serde_json::from_slice(&bytes).context("parsing suite")?;
    let url = if force {
        format!("{server}/v1/eval/suites?force=true")
    } else {
        format!("{server}/v1/eval/suites")
    };
    let resp = client.post(&url).json(&suite).send().await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        bail!("register failed ({status}): {text}");
    }
    println!("{text}");
    Ok(())
}

fn cmd_trace_suite(args: TraceSuiteArgs) -> Result<()> {
    if !args.stdout && args.output.is_none() {
        bail!("--output is required unless --stdout is set");
    }
    let input_paths = args.input.clone();

    let mut cfg = ProductionTraceSuiteConfig::new(args.suite_name);
    cfg.description = args.description;
    cfg.input_format = args.format;
    cfg.require_xml_format = args.require_qwen_xml;
    if let Some(max_examples) = args.max_examples {
        cfg.sampling.max_examples = Some(max_examples);
    }
    if let Some(seed) = args.seed {
        cfg.sampling.seed = Some(seed);
    }
    if let Some(max_prompt_chars) = args.max_prompt_chars {
        cfg.sampling.max_prompt_chars = max_prompt_chars;
    }
    if let Some(max_target_chars) = args.max_target_chars {
        cfg.sampling.max_target_chars = max_target_chars;
    }
    cfg.sampling.dedupe = args.dedupe;
    if let Some(max_tokens) = args.max_tokens {
        cfg.generation.max_tokens = max_tokens;
    }
    if let Some(temperature) = args.temperature {
        cfg.generation.temperature = temperature;
    }
    cfg.generation.thinking_budget_tokens = eval_budget_override(args.thinking_budget_tokens);
    cfg.generation.thinking_budget_ms = eval_budget_override(args.thinking_budget_ms);

    let input_lines = TraceInputLines::new(input_paths.clone());
    let (suite, stats) = kiln_eval::synthesize_production_trace_suite_from_lines(input_lines, &cfg)
        .with_context(|| {
            format!(
                "building production trace suite from {}",
                display_input_paths(&input_paths)
            )
        })?;
    let suite_json = serde_json::to_string_pretty(&suite)?;
    if args.stdout {
        println!("{suite_json}");
    }
    if let Some(output) = args.output.as_ref() {
        write_string_file(output, &suite_json)?;
    }

    if let Some(stats_output) = args.stats_output.as_ref() {
        let report = serde_json::json!({
            "schema_version": 1,
            "kind": "production_trace_suite_report",
            "input": input_paths.first().map(|p| p.display().to_string()),
            "inputs": input_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "suite_output": args.output.as_ref().map(|p| p.display().to_string()),
            "suite_name": &suite.name,
            "config": &cfg,
            "stats": &stats,
        });
        let report_json = serde_json::to_string_pretty(&report)?;
        write_string_file(stats_output, &report_json)?;
    }

    eprintln!(
        "trace-suite: rows={} parsed={} eligible_tool_turns={} kept={} seed={}",
        stats.rows_seen,
        stats.rows_parsed,
        stats.eligible_tool_turns,
        stats.sample_kept,
        stats.effective_seed,
    );
    if !stats.target_tool_histogram.is_empty() {
        eprintln!("target tools:");
        for (tool, count) in &stats.target_tool_histogram {
            eprintln!("  {tool}: {count}");
        }
    }
    if stats.skipped_parse_error > 0
        || stats.skipped_no_tool_call > 0
        || stats.skipped_malformed_tool_call > 0
        || stats.skipped_prompt_too_long > 0
        || stats.skipped_target_too_long > 0
        || stats.skipped_duplicate > 0
    {
        eprintln!(
            "skipped: parse_error={} no_tool_call={} malformed_tool_call={} empty_prompt={} prompt_too_long={} target_too_long={} duplicate={}",
            stats.skipped_parse_error,
            stats.skipped_no_tool_call,
            stats.skipped_malformed_tool_call,
            stats.skipped_empty_prompt,
            stats.skipped_prompt_too_long,
            stats.skipped_target_too_long,
            stats.skipped_duplicate,
        );
    }
    if !stats.parse_error_examples.is_empty() {
        eprintln!("first parse errors:");
        for err in &stats.parse_error_examples {
            eprintln!("  {err}");
        }
    }
    Ok(())
}

fn cmd_panel_suite(args: PanelSuiteArgs) -> Result<()> {
    if !args.stdout && args.output.is_none() {
        bail!("--output is required unless --stdout is set");
    }
    if args.max_examples == 0 {
        bail!("--max-examples must be greater than zero");
    }

    let source = EvalSuite::load_json(&args.suite)
        .with_context(|| format!("loading suite {}", args.suite.display()))?;
    let (panel, stats) = build_panel_suite(&source, &args)?;
    let panel_json = serde_json::to_string_pretty(&panel)?;
    if args.stdout {
        println!("{panel_json}");
    }
    if let Some(output) = args.output.as_ref() {
        write_string_file(output, &panel_json)?;
    }

    if let Some(stats_output) = args.stats_output.as_ref() {
        let report = serde_json::json!({
            "schema_version": 1,
            "kind": "production_trace_panel_suite_report",
            "source_suite": args.suite.display().to_string(),
            "suite_output": args.output.as_ref().map(|p| p.display().to_string()),
            "stats": stats.clone(),
        });
        let report_json = serde_json::to_string_pretty(&report)?;
        write_string_file(stats_output, &report_json)?;
    }

    eprintln!(
        "panel-suite: source_examples={} kept={} strata={} dropped_strata={} seed={}",
        source.examples.len(),
        panel.examples.len(),
        stats
            .get("num_strata")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        stats
            .get("dropped_strata")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        args.seed,
    );
    Ok(())
}

#[derive(Debug)]
struct PanelStratum {
    key: String,
    tool: String,
    split: String,
    prompt_bucket: String,
    prompt_size_basis: &'static str,
    indices: Vec<usize>,
    weight_sum: f64,
    prompt_chars_sum: usize,
    prompt_tokens_sum: usize,
    prompt_tokens_count: usize,
}

fn build_panel_suite(
    source: &EvalSuite,
    args: &PanelSuiteArgs,
) -> Result<(EvalSuite, serde_json::Value)> {
    let total_examples = source.examples.len();
    if total_examples == 0 {
        bail!("source suite has no examples");
    }
    let target_examples = args.max_examples.min(total_examples);
    let suite_tools_chars = source
        .tools
        .as_ref()
        .and_then(|tools| serde_json::to_string(tools).ok())
        .map(|s| s.len())
        .unwrap_or(0);

    let mut strata_by_key: BTreeMap<String, PanelStratum> = BTreeMap::new();
    for (idx, example) in source.examples.iter().enumerate() {
        let prompt_chars = metadata_usize(
            example.metadata.as_ref(),
            &["prompt_chars", "prompt_char_count", "input_chars"],
        )
        .unwrap_or_else(|| {
            example_prompt_chars(example, source.tools.as_deref(), suite_tools_chars)
        });
        let prompt_tokens = metadata_usize(
            example.metadata.as_ref(),
            &[
                "prompt_tokens",
                "prompt_token_count",
                "input_tokens",
                "input_token_count",
            ],
        );
        let (basis, prompt_bucket) = prompt_size_bucket(prompt_tokens, prompt_chars);
        let tool = panel_tool_key(example);
        let split = panel_split_key(example);
        let key = format!("{tool}|{prompt_bucket}|{split}");
        let stratum = strata_by_key
            .entry(key.clone())
            .or_insert_with(|| PanelStratum {
                key,
                tool,
                split,
                prompt_bucket,
                prompt_size_basis: basis,
                indices: Vec::new(),
                weight_sum: 0.0,
                prompt_chars_sum: 0,
                prompt_tokens_sum: 0,
                prompt_tokens_count: 0,
            });
        stratum.indices.push(idx);
        stratum.weight_sum += example.weight as f64;
        stratum.prompt_chars_sum += prompt_chars;
        if let Some(tokens) = prompt_tokens {
            stratum.prompt_tokens_sum += tokens;
            stratum.prompt_tokens_count += 1;
        }
    }

    let strata = strata_by_key.into_values().collect::<Vec<_>>();
    let keep_counts = allocate_panel_counts(&strata, target_examples);
    let mut selected = Vec::new();
    let mut selected_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut selected_weight_sums: BTreeMap<String, f64> = BTreeMap::new();
    let mut selected_prompt_chars: BTreeMap<String, usize> = BTreeMap::new();
    let mut selected_prompt_tokens: BTreeMap<String, usize> = BTreeMap::new();
    let mut selected_ids_by_stratum: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (stratum_idx, stratum) in strata.iter().enumerate() {
        let keep = keep_counts[stratum_idx];
        if keep == 0 {
            continue;
        }
        let mut candidates = stratum
            .indices
            .iter()
            .map(|idx| {
                let example_id = source.examples[*idx].resolved_id();
                (
                    stable_sample_key(args.seed, &stratum.key, &example_id, *idx),
                    *idx,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let chosen = candidates
            .into_iter()
            .take(keep)
            .map(|(_, idx)| idx)
            .collect::<Vec<_>>();
        let selected_source_weight = chosen
            .iter()
            .map(|idx| source.examples[*idx].weight as f64)
            .sum::<f64>();
        let equal_weight = if keep > 0 {
            (stratum.weight_sum / keep as f64) as f32
        } else {
            0.0
        };
        for idx in chosen {
            let source_weight = source.examples[idx].weight as f64;
            let panel_weight = if stratum.weight_sum == 0.0 {
                0.0
            } else if selected_source_weight > 0.0 {
                (source_weight * stratum.weight_sum / selected_source_weight) as f32
            } else {
                equal_weight
            };
            selected.push((idx, panel_weight, stratum_idx));
        }
    }

    selected.sort_by_key(|(idx, _, _)| *idx);
    let mut panel_examples = Vec::with_capacity(selected.len());
    let mut selected_example_ids = Vec::with_capacity(selected.len());
    for (idx, panel_weight, stratum_idx) in &selected {
        let mut example = source.examples[*idx].clone();
        example.weight = *panel_weight;
        let stratum = &strata[*stratum_idx];
        let example_id = example.resolved_id();
        *selected_counts.entry(stratum.key.clone()).or_default() += 1;
        *selected_weight_sums.entry(stratum.key.clone()).or_default() += *panel_weight as f64;
        *selected_prompt_chars
            .entry(stratum.key.clone())
            .or_default() += metadata_usize(
            example.metadata.as_ref(),
            &["prompt_chars", "prompt_char_count", "input_chars"],
        )
        .unwrap_or_else(|| {
            example_prompt_chars(&example, source.tools.as_deref(), suite_tools_chars)
        });
        if let Some(tokens) = metadata_usize(
            example.metadata.as_ref(),
            &[
                "prompt_tokens",
                "prompt_token_count",
                "input_tokens",
                "input_token_count",
            ],
        ) {
            *selected_prompt_tokens
                .entry(stratum.key.clone())
                .or_default() += tokens;
        }
        selected_ids_by_stratum
            .entry(stratum.key.clone())
            .or_default()
            .push(example_id.clone());
        selected_example_ids.push(example_id);
        panel_examples.push(example);
    }

    let source_weight_sum = source
        .examples
        .iter()
        .map(|example| example.weight as f64)
        .sum::<f64>();
    let panel_weight_sum = panel_examples
        .iter()
        .map(|example| example.weight as f64)
        .sum::<f64>();
    let source_prompt_chars = strata.iter().map(|s| s.prompt_chars_sum).sum::<usize>();
    let source_prompt_tokens = strata.iter().map(|s| s.prompt_tokens_sum).sum::<usize>();
    let selected_prompt_chars_sum = selected_prompt_chars.values().copied().sum::<usize>();
    let selected_prompt_tokens_sum = selected_prompt_tokens.values().copied().sum::<usize>();
    let dropped_strata = keep_counts.iter().filter(|keep| **keep == 0).count();
    let strata_report = strata
        .iter()
        .enumerate()
        .map(|(idx, stratum)| {
            let selected_count = selected_counts.get(&stratum.key).copied().unwrap_or(0);
            serde_json::json!({
                "key": &stratum.key,
                "tool": &stratum.tool,
                "split": &stratum.split,
                "prompt_bucket": &stratum.prompt_bucket,
                "prompt_size_basis": stratum.prompt_size_basis,
                "population": stratum.indices.len(),
                "selected": selected_count,
                "source_weight_sum": stratum.weight_sum,
                "panel_weight_sum": selected_weight_sums.get(&stratum.key).copied().unwrap_or(0.0),
                "source_prompt_chars": stratum.prompt_chars_sum,
                "selected_prompt_chars": selected_prompt_chars.get(&stratum.key).copied().unwrap_or(0),
                "source_prompt_tokens": (stratum.prompt_tokens_count > 0).then_some(stratum.prompt_tokens_sum),
                "selected_prompt_tokens": selected_prompt_tokens.get(&stratum.key).copied(),
                "selected_example_ids": selected_ids_by_stratum.get(&stratum.key).cloned().unwrap_or_default(),
                "dropped": selected_count == 0,
                "allocation_rank": idx,
            })
        })
        .collect::<Vec<_>>();

    let suite_name = args
        .suite_name
        .clone()
        .unwrap_or_else(|| format!("{}-panel-{}", source.name, target_examples));
    let panel = EvalSuite {
        name: suite_name,
        description: Some(format!(
            "Weighted stratified fast panel sampled from `{}`: {}/{} examples. Prompts are preserved exactly; weights expand selected examples back to their source strata.",
            source.name,
            panel_examples.len(),
            source.examples.len(),
        )),
        default_scorer: source.default_scorer.clone(),
        generation: source.generation.clone(),
        system_prompt: source.system_prompt.clone(),
        examples: panel_examples,
        schema_version: source.schema_version,
        tools: source.tools.clone(),
    };
    let stats = serde_json::json!({
        "source_suite_name": &source.name,
        "panel_suite_name": &panel.name,
        "seed": args.seed,
        "requested_max_examples": args.max_examples,
        "source_examples": total_examples,
        "kept_examples": panel.examples.len(),
        "num_strata": strata.len(),
        "dropped_strata": dropped_strata,
        "source_weight_sum": source_weight_sum,
        "panel_weight_sum": panel_weight_sum,
        "source_prompt_chars": source_prompt_chars,
        "panel_prompt_chars": selected_prompt_chars_sum,
        "source_prompt_tokens": (source_prompt_tokens > 0).then_some(source_prompt_tokens),
        "panel_prompt_tokens": (selected_prompt_tokens_sum > 0).then_some(selected_prompt_tokens_sum),
        "selected_example_ids": selected_example_ids,
        "strata": strata_report,
    });
    Ok((panel, stats))
}

fn allocate_panel_counts(strata: &[PanelStratum], target_examples: usize) -> Vec<usize> {
    let total_examples = strata.iter().map(|s| s.indices.len()).sum::<usize>();
    if target_examples >= total_examples {
        return strata.iter().map(|s| s.indices.len()).collect();
    }

    let mut keep_counts = vec![0usize; strata.len()];
    if target_examples >= strata.len() {
        for keep in &mut keep_counts {
            *keep = 1;
        }
        let remaining = target_examples - strata.len();
        let remaining_capacity = strata
            .iter()
            .map(|s| s.indices.len().saturating_sub(1))
            .sum::<usize>();
        distribute_panel_remainder(strata, &mut keep_counts, remaining, remaining_capacity);
    } else {
        let mut order = (0..strata.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| {
            strata[*b]
                .indices
                .len()
                .cmp(&strata[*a].indices.len())
                .then_with(|| strata[*a].key.cmp(&strata[*b].key))
        });
        for idx in order.into_iter().take(target_examples) {
            keep_counts[idx] = 1;
        }
    }
    keep_counts
}

fn distribute_panel_remainder(
    strata: &[PanelStratum],
    keep_counts: &mut [usize],
    remaining: usize,
    remaining_capacity: usize,
) {
    if remaining == 0 || remaining_capacity == 0 {
        return;
    }
    let mut assigned = 0usize;
    let mut remainders = Vec::new();
    for (idx, stratum) in strata.iter().enumerate() {
        let capacity = stratum.indices.len().saturating_sub(keep_counts[idx]);
        if capacity == 0 {
            remainders.push((idx, 0.0f64));
            continue;
        }
        let ideal = remaining as f64 * capacity as f64 / remaining_capacity as f64;
        let whole = ideal.floor() as usize;
        let add = whole.min(capacity);
        keep_counts[idx] += add;
        assigned += add;
        remainders.push((idx, ideal - whole as f64));
    }
    while assigned < remaining {
        remainders.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| strata[b.0].indices.len().cmp(&strata[a.0].indices.len()))
                .then_with(|| strata[a.0].key.cmp(&strata[b.0].key))
        });
        let mut advanced = false;
        for (idx, _) in &remainders {
            if keep_counts[*idx] < strata[*idx].indices.len() {
                keep_counts[*idx] += 1;
                assigned += 1;
                advanced = true;
                break;
            }
        }
        if !advanced {
            break;
        }
    }
}

fn panel_tool_key(example: &EvalExample) -> String {
    if let Some(tag) = example.tags.iter().find(|tag| tag.starts_with("tool:")) {
        return tag.clone();
    }
    if let Some(target) = example.target.as_deref()
        && let Some(call) = extract_first_tool_call(target)
    {
        return format!("tool:{}", call.name);
    }
    "tool:<none>".to_string()
}

fn panel_split_key(example: &EvalExample) -> String {
    if let Some(tag) = example.tags.iter().find(|tag| tag.starts_with("split:")) {
        return tag.clone();
    }
    if let Some(split) = example
        .metadata
        .as_ref()
        .and_then(|m| m.get("split"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return format!("split:{split}");
    }
    "split:<none>".to_string()
}

fn prompt_size_bucket(prompt_tokens: Option<usize>, prompt_chars: usize) -> (&'static str, String) {
    if let Some(tokens) = prompt_tokens {
        let label = match tokens {
            0..=16_383 => "prompt_tokens:<16k",
            16_384..=32_767 => "prompt_tokens:16k-32k",
            32_768..=65_535 => "prompt_tokens:32k-65k",
            65_536..=131_071 => "prompt_tokens:65k-131k",
            _ => "prompt_tokens:>=131k",
        };
        return ("tokens", label.to_string());
    }
    let label = match prompt_chars {
        0..=65_535 => "prompt_chars:<64k",
        65_536..=131_071 => "prompt_chars:64k-128k",
        131_072..=262_143 => "prompt_chars:128k-256k",
        262_144..=524_287 => "prompt_chars:256k-512k",
        _ => "prompt_chars:>=512k",
    };
    ("chars", label.to_string())
}

fn example_prompt_chars(
    example: &EvalExample,
    suite_tools: Option<&[serde_json::Value]>,
    suite_tools_chars: usize,
) -> usize {
    let message_chars = serde_json::to_string(&example.messages)
        .map(|s| s.len())
        .unwrap_or_else(|_| {
            example
                .messages
                .iter()
                .map(|m| m.role.len() + m.content.len())
                .sum()
        });
    let tool_chars = if let Some(tools) = example.tools.as_ref() {
        serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0)
    } else if suite_tools.is_some() {
        suite_tools_chars
    } else {
        0
    };
    message_chars + tool_chars
}

fn metadata_usize(metadata: Option<&serde_json::Value>, keys: &[&str]) -> Option<usize> {
    let object = metadata?.as_object()?;
    for key in keys {
        if let Some(value) = object.get(*key)
            && let Some(parsed) = json_value_usize(value)
        {
            return Some(parsed);
        }
    }
    None
}

fn json_value_usize(value: &serde_json::Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .or_else(|| value.as_f64().filter(|v| *v >= 0.0).map(|v| v as usize))
        .or_else(|| value.as_str().and_then(|s| s.parse::<usize>().ok()))
}

fn stable_sample_key(seed: u64, stratum_key: &str, example_id: &str, ordinal: usize) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(stratum_key.as_bytes());
    hasher.update([0]);
    hasher.update(example_id.as_bytes());
    hasher.update([0]);
    hasher.update(ordinal.to_le_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

struct TraceInputLines {
    paths: Vec<PathBuf>,
    next_path: usize,
    current_path: Option<PathBuf>,
    current_reader: Option<std::io::BufReader<std::fs::File>>,
    current_line: usize,
}

impl TraceInputLines {
    fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            next_path: 0,
            current_path: None,
            current_reader: None,
            current_line: 0,
        }
    }
}

impl Iterator for TraceInputLines {
    type Item = std::result::Result<ProductionTraceInputLine, ProductionTraceError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(reader) = self.current_reader.as_mut() {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        self.current_reader = None;
                        self.current_path = None;
                        self.current_line = 0;
                        continue;
                    }
                    Ok(_) => {
                        self.current_line += 1;
                        return Some(Ok(ProductionTraceInputLine {
                            json: line,
                            source_path: self
                                .current_path
                                .as_ref()
                                .map(|p| p.display().to_string()),
                            source_line: Some(self.current_line),
                        }));
                    }
                    Err(e) => {
                        let path = self
                            .current_path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "<unknown>".to_string());
                        return Some(Err(ProductionTraceError::Io(format!("read {path}: {e}"))));
                    }
                }
            }

            if self.next_path >= self.paths.len() {
                return None;
            }

            let path = self.paths[self.next_path].clone();
            self.next_path += 1;
            match std::fs::File::open(&path) {
                Ok(file) => {
                    self.current_path = Some(path);
                    self.current_reader = Some(std::io::BufReader::new(file));
                    self.current_line = 0;
                }
                Err(e) => {
                    return Some(Err(ProductionTraceError::Io(format!(
                        "open {}: {e}",
                        path.display()
                    ))));
                }
            }
        }
    }
}

fn display_input_paths(paths: &[PathBuf]) -> String {
    match paths {
        [] => "<none>".to_string(),
        [one] => one.display().to_string(),
        many => format!(
            "{} inputs ({})",
            many.len(),
            many.iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn write_string_file(path: &PathBuf, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(path, contents).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

async fn cmd_run(client: &reqwest::Client, server: &str, args: RunArgs) -> Result<()> {
    if args.suite.is_some() == args.file.is_some() {
        bail!("exactly one of --suite or --file must be set");
    }
    let mut body = serde_json::Map::new();
    if let Some(name) = args.suite.as_deref() {
        body.insert("suite".into(), serde_json::json!(name));
    }
    if let Some(path) = args.file.as_deref() {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let inline: EvalSuite = serde_json::from_slice(&bytes).context("parsing suite")?;
        body.insert("inline_suite".into(), serde_json::to_value(&inline)?);
    }
    if let Some(adapter) = args.adapter {
        body.insert("adapter".into(), serde_json::json!(adapter));
    }
    if let Some(generation) = generation_override_json(
        args.temperature,
        args.max_tokens,
        args.thinking_budget_tokens,
        args.thinking_budget_ms,
    )? {
        body.insert("generation".into(), generation);
    }
    let resp = client
        .post(format!("{server}/v1/eval/run"))
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        bail!("run failed ({status}): {text}");
    }
    let parsed: EvalRunResponse = serde_json::from_str(&text)?;
    eprintln!("queued eval {} (state={})", parsed.job_id, parsed.state);
    if !args.watch {
        println!("{text}");
        return Ok(());
    }
    let result = poll_until_done(client, server, &parsed.job_id).await?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_human(&result);
    }
    Ok(())
}

fn generation_override_json(
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
) -> Result<Option<serde_json::Value>> {
    if temperature.is_none()
        && max_tokens.is_none()
        && thinking_budget_tokens.is_none()
        && thinking_budget_ms.is_none()
    {
        return Ok(None);
    }

    let mut generation = serde_json::Map::new();
    if let Some(value) = temperature {
        generation.insert("temperature".into(), serde_json::json!(value));
    }
    if let Some(value) = max_tokens {
        generation.insert("max_tokens".into(), serde_json::json!(value));
    }
    if let Some(value) = thinking_budget_tokens {
        generation.insert(
            "thinking_budget_tokens".into(),
            serde_json::to_value(eval_budget_override(Some(value)))?,
        );
    }
    if let Some(value) = thinking_budget_ms {
        generation.insert(
            "thinking_budget_ms".into(),
            serde_json::to_value(eval_budget_override(Some(value)))?,
        );
    }
    Ok(Some(serde_json::Value::Object(generation)))
}

fn eval_budget_override<T>(value: Option<ThinkingBudgetArg<T>>) -> EvalBudgetOverride<T> {
    match value {
        None => EvalBudgetOverride::Inherit,
        Some(ThinkingBudgetArg::Unlimited) => EvalBudgetOverride::Unlimited,
        Some(ThinkingBudgetArg::Limited(value)) => EvalBudgetOverride::Limited(value),
    }
}

async fn cmd_compare(client: &reqwest::Client, server: &str, args: CompareArgs) -> Result<()> {
    let body = EvalCompareSpec {
        suite: args.suite,
        adapters: args.adapters,
        generation: None,
    };
    let resp = client
        .post(format!("{server}/v1/eval/compare"))
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        bail!("compare failed ({status}): {text}");
    }
    let parsed: EvalRunResponse = serde_json::from_str(&text)?;
    eprintln!("queued compare {} (state={})", parsed.job_id, parsed.state);
    if !args.watch {
        println!("{text}");
        return Ok(());
    }
    let result = poll_until_done(client, server, &parsed.job_id).await?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_compare(&result);
    }
    Ok(())
}

async fn cmd_probe(client: &reqwest::Client, server: &str, args: ProbeArgs) -> Result<()> {
    let scorer = match args.scorer.as_str() {
        "exact_match" => kiln_eval::Scorer::ExactMatch {
            case_sensitive: false,
            strip_whitespace: true,
        },
        "contains" => kiln_eval::Scorer::Contains {
            phrases: vec![args.target.clone()],
            mode: Default::default(),
            case_sensitive: false,
        },
        "numeric" => kiln_eval::Scorer::NumericTolerance(kiln_eval::NumericTolerance::default()),
        "regex" => kiln_eval::Scorer::Regex {
            pattern: args.target.clone(),
            capture_group: None,
            case_sensitive: false,
        },
        other => bail!("unknown scorer `{other}` (try exact_match | contains | numeric | regex)"),
    };
    let suite = EvalSuite {
        name: format!("probe-{}", &uuid_v4()[..8]),
        description: Some("ad-hoc probe from kiln-eval CLI".into()),
        default_scorer: scorer,
        generation: EvalGenerationParams {
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            thinking_budget_tokens: eval_budget_override(args.thinking_budget_tokens),
            thinking_budget_ms: eval_budget_override(args.thinking_budget_ms),
            ..Default::default()
        },
        system_prompt: None,
        examples: vec![EvalExample {
            id: Some("probe".into()),
            messages: vec![EvalChatMessage::new("user", args.prompt)],
            target: Some(args.target),
            tags: vec!["probe".into()],
            ..Default::default()
        }],
        schema_version: 1,
        tools: None,
    };
    let body = serde_json::json!({
        "inline_suite": suite,
        "adapter": args.adapter,
    });
    let resp = client
        .post(format!("{server}/v1/eval/run"))
        .json(&body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        bail!("probe failed ({status}): {text}");
    }
    let parsed: EvalRunResponse = serde_json::from_str(&text)?;
    let result = poll_until_done(client, server, &parsed.job_id).await?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        print_human(&result);
    }
    Ok(())
}

fn uuid_v4() -> String {
    uuid::Uuid::new_v4().to_string()
}

async fn poll_until_done(
    client: &reqwest::Client,
    server: &str,
    job_id: &str,
) -> Result<EvalResultPayload> {
    let mut last_progress: Option<f32> = None;
    loop {
        let resp = client
            .get(format!("{server}/v1/eval/jobs/{job_id}"))
            .send()
            .await?;
        if !resp.status().is_success() {
            bail!(
                "status check failed ({}): {}",
                resp.status(),
                resp.text().await?
            );
        }
        let payload: EvalResultPayload = resp.json().await?;
        match payload.state.as_str() {
            "completed" | "failed" | "cancelled" => return Ok(payload),
            _ => {
                if let Some(p) = payload.progress.as_ref() {
                    if p.examples_total > 0 {
                        let frac = p.examples_completed as f32 / p.examples_total as f32;
                        if last_progress.map_or(true, |last| (frac - last).abs() > 0.01) {
                            eprintln!(
                                "  {}/{} ({:>5.1}% acc, mean={:.2})",
                                p.examples_completed,
                                p.examples_total,
                                p.running_accuracy * 100.0,
                                p.running_mean_score
                            );
                            last_progress = Some(frac);
                        }
                    }
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

fn print_human(result: &EvalResultPayload) {
    if result.state != "completed" {
        eprintln!("job ended in state `{}`", result.state);
        if let Some(e) = result.error.as_ref() {
            eprintln!("  error: {e}");
        }
    }
    for r in &result.runs {
        let adapter_label = r
            .adapter
            .as_deref()
            .map(|s| {
                if s.is_empty() {
                    "<base>".to_string()
                } else {
                    s.to_string()
                }
            })
            .unwrap_or_else(|| "<base>".to_string());
        println!();
        println!("Suite: {} | Adapter: {}", r.suite_name, adapter_label);
        println!("  job: {}  hash: {}", result.job_id, r.suite_hash);
        println!(
            "  accuracy: {:>5.1}% {}  |  mean: {:.3}  |  weighted: {:.3}",
            r.metrics.accuracy * 100.0,
            format_ci(&r.metrics.accuracy_confidence_interval),
            r.metrics.mean_score,
            r.metrics.weighted_mean_score
        );
        println!(
            "  pass:{}  fail:{}  invalid:{}  error:{}  (n={})",
            r.metrics.num_pass,
            r.metrics.num_fail,
            r.metrics.num_invalid,
            r.metrics.num_error,
            r.metrics.num_examples
        );
        if !r.metrics.latency.p50_ms.is_nan() && r.metrics.latency.mean_ms > 0.0 {
            println!(
                "  latency ms: p50={:.0}  p90={:.0}  p99={:.0}  mean={:.0}",
                r.metrics.latency.p50_ms,
                r.metrics.latency.p90_ms,
                r.metrics.latency.p99_ms,
                r.metrics.latency.mean_ms,
            );
        }
        if !r.metrics.by_scorer.is_empty() && r.metrics.by_scorer.len() > 1 {
            // Only worth showing when more than one scorer is in play —
            // single-scorer suites read this off the suite header.
            println!("  by scorer:");
            for br in &r.metrics.by_scorer {
                println!(
                    "    {:<24}  {:>5.1}%  ({} examples)",
                    br.scorer_kind,
                    br.pass_rate * 100.0,
                    br.num_examples
                );
            }
        }
        if !r.metrics.tag_breakdown.is_empty() {
            println!("  by tag:");
            for (tag, br) in &r.metrics.tag_breakdown {
                println!(
                    "    {:<24}  {:>5.1}% {}  ({}/{})",
                    tag,
                    br.pass_rate * 100.0,
                    format_ci(&br.confidence_interval),
                    br.num_pass,
                    br.num_examples
                );
            }
        } else if !r.metrics.pass_rate_by_tag.is_empty() {
            println!("  by tag:");
            for (tag, rate) in &r.metrics.pass_rate_by_tag {
                println!("    {:<24}  {:>5.1}%", tag, rate * 100.0);
            }
        }
        if !r.metrics.pass_rate_by_tool.is_empty() {
            println!("  by tool:");
            for (tool, br) in &r.metrics.pass_rate_by_tool {
                println!(
                    "    {:<24}  {:>5.1}% {}  ({}/{})",
                    tool,
                    br.pass_rate * 100.0,
                    format_ci(&br.confidence_interval),
                    br.num_pass,
                    br.num_examples
                );
            }
        }
        if !r.metrics.confusion_by_tool.is_empty() {
            // Only show entries where the model picked something *different*
            // from the target (or skipped) — the matrix is mostly diagonal
            // on a passing run, no need to print every cell.
            let mut printed_header = false;
            for (target, row) in &r.metrics.confusion_by_tool {
                for (predicted, count) in row {
                    if predicted == target {
                        continue;
                    }
                    if !printed_header {
                        println!("  confusion (target → predicted):");
                        printed_header = true;
                    }
                    println!("    {:<20} → {:<20}  ×{}", target, predicted, count);
                }
            }
        }
        if r.metrics.reasoning_length.num_with_thinking > 0 {
            println!(
                "  reasoning: n={}  mean={:.0} chars  p50={}  p90={}  max={}",
                r.metrics.reasoning_length.num_with_thinking,
                r.metrics.reasoning_length.mean_chars,
                r.metrics.reasoning_length.p50_chars,
                r.metrics.reasoning_length.p90_chars,
                r.metrics.reasoning_length.max_chars,
            );
        }
        if r.metrics.num_unclosed_thinking > 0 {
            println!(
                "  ⚠ {} completion(s) emitted <think> without </think> (max_tokens?)",
                r.metrics.num_unclosed_thinking
            );
        }
        if r.metrics.num_non_xml_tool_calls > 0 {
            println!(
                "  ⚠ {} completion(s) used non-XML tool-call format (Qwen3.5 native is XML)",
                r.metrics.num_non_xml_tool_calls
            );
        }
        if r.metrics.num_schema_missing_required > 0 || r.metrics.num_schema_extra_unknown > 0 {
            println!(
                "  schema: missing-required={}  extra-unknown={}",
                r.metrics.num_schema_missing_required, r.metrics.num_schema_extra_unknown,
            );
        }
    }
}

fn format_ci(ci: &kiln_eval::result::PassRateConfidenceInterval) -> String {
    if ci.confidence_level <= 0.0 {
        return String::new();
    }
    format!("(95% CI {:.1}-{:.1}%)", ci.lower * 100.0, ci.upper * 100.0)
}

fn print_compare(result: &EvalResultPayload) {
    if result.runs.is_empty() {
        eprintln!("no runs in compare result");
        return;
    }
    println!(
        "{:<24}  {:>10}  {:>10}  {:>10}",
        "adapter", "accuracy", "mean", "p50 ms"
    );
    for r in &result.runs {
        let adapter = r
            .adapter
            .as_deref()
            .map(|s| {
                if s.is_empty() {
                    "<base>".to_string()
                } else {
                    s.to_string()
                }
            })
            .unwrap_or_else(|| "<base>".to_string());
        println!(
            "{:<24}  {:>9.1}%  {:>10.3}  {:>10.0}",
            adapter,
            r.metrics.accuracy * 100.0,
            r.metrics.mean_score,
            r.metrics.latency.p50_ms
        );
    }
    // Pairwise delta: best - worst.
    let mut sorted = result.runs.clone();
    sorted.sort_by(|a, b| {
        b.metrics
            .accuracy
            .partial_cmp(&a.metrics.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if sorted.len() >= 2 {
        let best = &sorted[0];
        let worst = &sorted[sorted.len() - 1];
        println!(
            "\nbest: {}  ({:.1}%)\nworst: {}  ({:.1}%)\ndelta: {:+.1} pp",
            best.adapter.clone().unwrap_or_default(),
            best.metrics.accuracy * 100.0,
            worst.adapter.clone().unwrap_or_default(),
            worst.metrics.accuracy * 100.0,
            (best.metrics.accuracy - worst.metrics.accuracy) * 100.0
        );
    }
    // Flip diff between the first two runs (baseline → candidate).
    if let Some(diff) = compute_flip_diff(result) {
        println!("\nflips: {} → {}", diff.baseline, diff.candidate);
        println!(
            "  both_pass: {}  both_fail: {}",
            diff.both_pass, diff.both_fail
        );
        if !diff.improved.is_empty() {
            println!(
                "  ✓ improved ({}): {}",
                diff.improved.len(),
                diff.improved.join(", ")
            );
        }
        if !diff.regressed.is_empty() {
            println!(
                "  ✗ regressed ({}): {}",
                diff.regressed.len(),
                diff.regressed.join(", ")
            );
        }
        let test = diff.significance();
        println!(
            "  sign test: improved {} / regressed {} -> p={:.4} (two-sided exact)",
            test.improved, test.regressed, test.p_value
        );
        if test.improved + test.regressed == 0 {
            println!(
                "  verdict: no discordant examples — the adapters are indistinguishable on this suite"
            );
        } else if test.significant() {
            println!("  verdict: significant at p<0.05");
        } else {
            println!(
                "  verdict: NOT significant at p<0.05 — the delta could be noise; add examples or rerun"
            );
        }
    }
}

fn compute_flip_diff(result: &EvalResultPayload) -> Option<kiln_eval::FlipDiff> {
    // Reuse the canonical flip-diff logic by reshaping into the real
    // EvalResult type via a JSON round-trip. Cheap (the payload is
    // typically a few KB) and avoids keeping two implementations in sync.
    let synthetic = kiln_eval::EvalResult {
        job_id: result.job_id.clone(),
        state: kiln_eval::EvalJobState::Completed,
        runs: result.runs.clone(),
        progress: None,
        error: None,
    };
    synthetic.flip_diff()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_format_alias_is_rejected() {
        let err = parse_trace_format("jsonl").unwrap_err();
        assert!(err.contains("ambiguous"), "{err}");
        assert!(err.contains("anthropic_jsonl"), "{err}");
        assert!(parse_trace_format("anthropic_jsonl").is_ok());
        assert!(parse_trace_format("auto").is_ok());
    }

    #[test]
    fn thinking_budget_flags_parse_for_generation_commands() {
        let cli = Cli::parse_from([
            "kiln-eval",
            "run",
            "--suite",
            "smoke",
            "--thinking-budget-tokens",
            "0",
            "--thinking-budget-ms",
            "1250",
        ]);
        let Command::Run(args) = cli.cmd else {
            panic!("expected run command");
        };
        assert_eq!(
            args.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Limited(0))
        );
        assert_eq!(
            args.thinking_budget_ms,
            Some(ThinkingBudgetArg::Limited(1250))
        );

        let cli = Cli::parse_from([
            "kiln-eval",
            "probe",
            "--prompt",
            "2+2",
            "--target",
            "4",
            "--thinking-budget-tokens",
            "64",
            "--thinking-budget-ms",
            "500",
        ]);
        let Command::Probe(args) = cli.cmd else {
            panic!("expected probe command");
        };
        assert_eq!(
            args.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Limited(64))
        );
        assert_eq!(
            args.thinking_budget_ms,
            Some(ThinkingBudgetArg::Limited(500))
        );

        let cli = Cli::parse_from([
            "kiln-eval",
            "trace-suite",
            "--input",
            "trace.jsonl",
            "--suite-name",
            "trace",
            "--stdout",
            "--thinking-budget-tokens",
            "96",
            "--thinking-budget-ms",
            "1500",
        ]);
        let Command::TraceSuite(args) = cli.cmd else {
            panic!("expected trace-suite command");
        };
        assert_eq!(
            args.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Limited(96))
        );
        assert_eq!(
            args.thinking_budget_ms,
            Some(ThinkingBudgetArg::Limited(1500))
        );

        let cli = Cli::parse_from([
            "kiln-eval",
            "run",
            "--suite",
            "smoke",
            "--thinking-budget-tokens",
            "unlimited",
            "--thinking-budget-ms",
            "unlimited",
        ]);
        let Command::Run(args) = cli.cmd else {
            panic!("expected run command");
        };
        assert_eq!(
            args.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Unlimited)
        );
        assert_eq!(args.thinking_budget_ms, Some(ThinkingBudgetArg::Unlimited));
    }

    #[test]
    fn run_generation_override_serializes_limited_thinking_budgets() {
        assert!(
            generation_override_json(None, None, None, None)
                .unwrap()
                .is_none()
        );

        let generation = generation_override_json(
            None,
            None,
            Some(ThinkingBudgetArg::Limited(0)),
            Some(ThinkingBudgetArg::Limited(1250)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(generation["thinking_budget_tokens"], 0);
        assert_eq!(generation["thinking_budget_ms"], 1250);
        assert!(generation.get("temperature").is_none());
        assert!(generation.get("max_tokens").is_none());

        let unlimited = generation_override_json(
            None,
            None,
            Some(ThinkingBudgetArg::Unlimited),
            Some(ThinkingBudgetArg::Unlimited),
        )
        .unwrap()
        .unwrap();
        assert!(
            unlimited
                .get("thinking_budget_tokens")
                .is_some_and(serde_json::Value::is_null)
        );
        assert!(
            unlimited
                .get("thinking_budget_ms")
                .is_some_and(serde_json::Value::is_null)
        );
        assert_eq!(
            eval_budget_override::<usize>(None),
            EvalBudgetOverride::Inherit
        );
        assert_eq!(
            eval_budget_override(Some(ThinkingBudgetArg::<usize>::Unlimited)),
            EvalBudgetOverride::Unlimited
        );
    }

    #[test]
    fn trace_suite_writes_output_file_even_with_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("trace.jsonl");
        let row = serde_json::json!({
            "prompt_messages": [{"role":"user", "content":"run pwd"}],
            "chosen": {"role":"assistant", "tool_calls":[{
                "type":"function",
                "function":{"name":"Bash", "arguments":"{\"command\":\"pwd\"}"}
            }]}
        });
        std::fs::write(&input, format!("{row}\n")).unwrap();
        let output = dir.path().join("suite.json");
        let args = TraceSuiteArgs {
            input: vec![input],
            output: Some(output.clone()),
            stats_output: None,
            suite_name: "stdout-and-output".into(),
            description: None,
            format: ProductionTraceFormat::Auto,
            max_examples: None,
            seed: Some(1),
            max_prompt_chars: None,
            max_target_chars: None,
            dedupe: false,
            require_qwen_xml: false,
            max_tokens: None,
            temperature: None,
            thinking_budget_tokens: Some(ThinkingBudgetArg::Limited(0)),
            thinking_budget_ms: Some(ThinkingBudgetArg::Limited(1250)),
            stdout: true,
        };
        cmd_trace_suite(args).unwrap();
        let suite: EvalSuite = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(suite.examples.len(), 1);
        assert_eq!(
            suite.generation.thinking_budget_tokens,
            EvalBudgetOverride::Limited(0)
        );
        assert_eq!(
            suite.generation.thinking_budget_ms,
            EvalBudgetOverride::Limited(1250)
        );
    }
}
