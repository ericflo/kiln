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
//!
//! All commands respect `KILN_SERVER_URL` and the `--server` flag (default
//! `http://localhost:8420`). Output is human-readable by default; pass
//! `--json` to emit the raw `EvalResult`.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use kiln_eval::{
    EvalChatMessage, EvalCompareSpec, EvalExample, EvalGenerationParams, EvalSuite,
    ProductionTraceFormat, ProductionTraceSuiteConfig, SuiteResult,
};
use serde::Deserialize;

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
    #[arg(long)]
    json: bool,
}

#[derive(Parser, Debug)]
struct TraceSuiteArgs {
    /// Production trace JSONL. Supports prompt_chosen_jsonl, openai_jsonl,
    /// and anthropic_jsonl rows from any exporter.
    #[arg(long)]
    input: PathBuf,
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
    /// Input format: auto | prompt_chosen_jsonl | openai_jsonl | anthropic_jsonl.
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
    /// Print the generated suite JSON to stdout instead of writing --output.
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
        "anthropic_jsonl" | "anthropic-jsonl" | "jsonl" => {
            Ok(ProductionTraceFormat::AnthropicJsonl)
        }
        other => Err(format!(
            "unknown trace format `{other}` (try auto | prompt_chosen_jsonl | openai_jsonl | anthropic_jsonl)"
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
    let file = std::fs::File::open(&args.input)
        .with_context(|| format!("open {}", args.input.display()))?;
    let reader = std::io::BufReader::new(file);

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

    let (suite, stats) =
        kiln_eval::synthesize_production_trace_suite(reader, &cfg).with_context(|| {
            format!(
                "building production trace suite from {}",
                args.input.display()
            )
        })?;
    let suite_json = serde_json::to_string_pretty(&suite)?;
    if args.stdout {
        println!("{suite_json}");
    } else if let Some(output) = args.output.as_ref() {
        write_string_file(output, &suite_json)?;
    }

    if let Some(stats_output) = args.stats_output.as_ref() {
        let report = serde_json::json!({
            "schema_version": 1,
            "kind": "production_trace_suite_report",
            "input": args.input.display().to_string(),
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
        || stats.skipped_prompt_too_long > 0
        || stats.skipped_target_too_long > 0
        || stats.skipped_duplicate > 0
    {
        eprintln!(
            "skipped: parse_error={} no_tool_call={} empty_prompt={} prompt_too_long={} target_too_long={} duplicate={}",
            stats.skipped_parse_error,
            stats.skipped_no_tool_call,
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
    if args.temperature.is_some() || args.max_tokens.is_some() {
        let mut generation = serde_json::Map::new();
        if let Some(t) = args.temperature {
            generation.insert("temperature".into(), serde_json::json!(t));
        }
        if let Some(m) = args.max_tokens {
            generation.insert("max_tokens".into(), serde_json::json!(m));
        }
        body.insert("generation".into(), serde_json::Value::Object(generation));
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
            "  accuracy: {:>5.1}%  |  mean: {:.3}  |  weighted: {:.3}",
            r.metrics.accuracy * 100.0,
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
        if !r.metrics.pass_rate_by_tag.is_empty() {
            println!("  by tag:");
            for (tag, rate) in &r.metrics.pass_rate_by_tag {
                println!("    {:<24}  {:>5.1}%", tag, rate * 100.0);
            }
        }
        if !r.metrics.pass_rate_by_tool.is_empty() {
            println!("  by tool:");
            for (tool, br) in &r.metrics.pass_rate_by_tool {
                println!(
                    "    {:<24}  {:>5.1}%  ({}/{})",
                    tool,
                    br.pass_rate * 100.0,
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
