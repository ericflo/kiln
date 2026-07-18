//! Kiln benchmark suite — measures inference throughput, latency, VRAM, and training speed.
//!
//! Run with: `cargo run --release --features cuda --bin kiln-bench -- --model-path /path/to/weights`
//! or `cargo run --release --features vulkan --bin kiln-bench -- --model-path /path/to/weights`.
//!
//! Requires a GPU with the Qwen3.5-4B model weights downloaded.

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::{ChatMessage, KilnTokenizer};
use kiln_memory::vram::{VramProbeSelector, detect_used_vram_bytes_for, detect_vram_for};
use kiln_model::PagedKvCacheKt;
use kiln_model::backend::{self as runtime_backend, LinearBackend, ResidencyBackend};
use kiln_model::forward::{
    GpuWeights,
    LinearAttentionState,
    StreamingPrefillExecutionPolicy,
    lm_head_sample_backend_decode_if,
    model_forward_head_backend_decode_if,
    model_forward_kt_with_policy,
    model_forward_paged_batched_decode_hidden,
    model_forward_paged_last_token,
    model_forward_paged_last_token_greedy,
    model_forward_paged_last_token_hidden,
    model_forward_paged_last_token_with_last_hidden,
    model_forward_paged_next_token_greedy,
    model_forward_paged_streaming_last_token_hidden_with_policy,
    model_forward_paged_streaming_last_token_with_last_hidden_with_policy,
    model_forward_paged_streaming_with_policy,
    // Phase 7 #1082: kt twin entry point + allocator stub for the first
    // end-to-end PagedKvCacheKt production wiring (latency bench decode
    // loop). Both are CUDA-only and active only when
    // `accelerator.kt_api_mode = "all"`; behavior is identical otherwise.
};
use kiln_model::kv_cache::KvCache;
use kiln_model::sampling::{greedy_sample, sample_step};
use kiln_model::speculative::{
    SpeculativeConfig, speculative_decode_step, speculative_decode_step_paged_greedy,
    speculative_mtp_decode_step,
};
use kiln_model::{
    BackendCapabilityQueries, BackendIdentity, ModelRunner, ModelRunnerRuntimeOptions,
    ReplayBackend, ReplayNativePrimitive, ReplayRequest, ServerTrainingDispatchPolicy,
    SpeculativeDecodePolicy, Support,
};
use kiln_server::config::{KilnConfig, SpecMethod};

/// Block size used for the paged-path benchmark. Matches the real server default.
const PAGED_BLOCK_SIZE: usize = 64;

/// Results from the full benchmark suite.
#[derive(Debug, Serialize)]
struct BenchmarkResults {
    /// Which `BackendRuntime` ran the forward pass — one of
    /// `cuda` / `metal` / `vulkan` / `cpu`. Lets downstream comparison scripts split
    /// runs by hardware path without parsing GPU names.
    backend: String,
    gpu_info: GpuInfo,
    model_load: ModelLoadResult,
    inference: Vec<InferenceBenchResult>,
    latency: LatencyResult,
    training: Option<TrainingResult>,
}

#[derive(Debug, Serialize)]
struct GpuInfo {
    name: String,
    total_vram_mb: u64,
    vram_source: String,
}

#[derive(Debug, Serialize)]
struct ModelLoadResult {
    load_time_secs: f64,
    model_vram_mb: u64,
}

#[derive(Debug, Serialize)]
struct InferenceBenchResult {
    batch_size: usize,
    prompt_tokens: usize,
    output_tokens: usize,
    total_time_secs: f64,
    tokens_per_sec: f64,
    peak_vram_mb: u64,
}

#[derive(Debug, Serialize)]
struct LatencyResult {
    prompt_tokens: usize,
    prefill_time_ms: f64,
    prefill_tokens_per_sec: f64,
    time_to_first_token_ms: f64,
    mean_inter_token_ms: f64,
    p50_inter_token_ms: f64,
    p99_inter_token_ms: f64,
    num_tokens_generated: usize,
    decode_tokens_per_sec: f64,
    /// Which speculative decoding arm produced this result. Lowercase string
    /// — "off" / "skip_layer" / "mtp" — emitted on every run so downstream
    /// comparison scripts can split by arm without reparsing env state.
    spec_method: String,
    /// MTP draft acceptance rate α = `draft_accepted / total_draft_attempts`.
    /// Populated only by the MTP arm (`spec_method = "mtp"`); `None` for
    /// `off` and `skip_layer` since those have no comparable single-α metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    acceptance_rate: Option<f64>,
    /// Phase C39 domain isolation tag. `"all"` for every pre-C39 arm and for
    /// off/skip-layer (which still pull from the full pool). `"gsm8k"` /
    /// `"humaneval"` / `"c4"` only when the MTP arm ran with `--prompt-subset`
    /// set explicitly.
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_subset: Option<String>,
}

fn runtime_backend_for_bench(
    device: &kiln_tensor::Device,
    weights: &GpuWeights,
) -> Result<std::sync::Arc<dyn kiln_model::BackendRuntime>> {
    let backend = runtime_backend::for_device_kt(device);
    LinearBackend::runtime_prewarm_decode_weights(backend.as_ref(), weights)
        .context("backend decode weight prewarm failed")?;
    Ok(backend)
}

/// Map the configured model dtype to the kt compute dtype used by the
/// KV cache / paged cache allocators below. Consolidates five identical
/// inline match blocks that previously appeared throughout this file
/// (issue #1082, candle removal).
fn kiln_config_dtype_to_kt(dtype: kiln_core::config::DType) -> kiln_tensor::DType {
    match dtype {
        kiln_core::config::DType::BF16 => kiln_tensor::DType::BF16,
        kiln_core::config::DType::FP16 => kiln_tensor::DType::F16,
        kiln_core::config::DType::FP32 => kiln_tensor::DType::F32,
    }
}

/// Greedy-sample a single token from kt logits.
///
/// The paged forward entry points now return kt `Tensor`s, and the sampler is
/// kt-native across GPU backends. Keep this helper as the bench-local
/// contiguity shim, but do not gate it to CUDA: ROCm/Vulkan/Metal latency
/// benches need the same scalar argmax path when the backend does not expose a
/// fused LM-head argmax.
fn greedy_sample_kt(logits: &kiln_tensor::Tensor) -> Result<u32> {
    let contig;
    let logits = if logits.is_contiguous() {
        logits
    } else {
        contig = logits.contiguous()?;
        &contig
    };
    greedy_sample(logits)
}

fn native_replay_support_enabled(support: Support) -> bool {
    matches!(support, Support::Native | Support::NativeWithConstraints)
}

fn bench_paged_decode_replay_primitive_enabled(
    backend: &dyn kiln_model::BackendRuntime,
    config: &ModelConfig,
    primitive: ReplayNativePrimitive,
) -> bool {
    let request =
        ReplayRequest::paged_decode_graph_outputs(config.hidden_size, config.intermediate_size, 1)
            .with_dtype(kiln_config_dtype_to_kt(config.dtype));
    let support = ReplayBackend::runtime_supports_replay_request(backend, &request);
    let authority = ReplayBackend::runtime_replay_authority(backend);
    native_replay_support_enabled(support) && authority.native_primitive == primitive
}

// (#1082) Deleted `bench_kt_tensor_to_candle`: the MTP bench arm's decode step
// + host sampler are now kt-native, so the kt->candle copy-bridge it provided
// has zero callers. rustc-confirmed dead.

struct BenchGdnRecurrentResidentStateScope<'a> {
    backend: &'a dyn kiln_model::BackendRuntime,
    active: bool,
}

impl<'a> BenchGdnRecurrentResidentStateScope<'a> {
    fn new(backend: &'a dyn kiln_model::BackendRuntime) -> Self {
        let active = ResidencyBackend::runtime_enter_gdn_recurrent_resident_state_scope(backend);
        Self { backend, active }
    }
}

impl Drop for BenchGdnRecurrentResidentStateScope<'_> {
    fn drop(&mut self) {
        if self.active {
            ResidencyBackend::runtime_exit_gdn_recurrent_resident_state_scope(self.backend);
        }
    }
}

#[derive(Debug, Serialize)]
struct TrainingResult {
    num_steps: usize,
    total_time_secs: f64,
    secs_per_step: f64,
    peak_vram_mb: u64,
}

/// Which PROMPT_POOL subset the MTP bench draws from.
///
/// Phase C39 isolates per-domain α after C38's N=30 all-domain re-bench
/// showed strong heterogeneity (GSM8K 0.789, HumanEval 0.689, C4 0.716).
/// `All` preserves C38 behavior (full 30-prompt pool, seed % 30 indexing).
/// Single-domain variants index the 10-prompt contiguous subslice so seeds
/// 0..9 hit each prompt once and N=20 covers every prompt twice.
///
/// Only affects the MTP bench arm (`--spec-method mtp`); ignored by
/// off / skip-layer, throughput, and training benches, which keep the C38
/// full-pool indexing to avoid surprising existing numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptSubset {
    /// All 30 prompts (C38 anchor; `seed % 30`).
    All,
    /// GSM8K-style grade-school math word problems (prompts 0-9).
    Gsm8k,
    /// HumanEval-style Python function signatures + docstrings (prompts 10-19).
    HumanEval,
    /// C4-style natural English text fragments (prompts 20-29).
    C4,
}

impl PromptSubset {
    /// Contiguous indices into `PROMPT_POOL` that this subset covers.
    fn indices(self) -> &'static [usize] {
        // Indices deliberately hand-listed so a PROMPT_POOL re-ordering
        // breaks the compile rather than silently mixing domains.
        const ALL: &[usize] = &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29,
        ];
        const GSM8K: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        const HUMAN_EVAL: &[usize] = &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19];
        const C4: &[usize] = &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
        match self {
            PromptSubset::All => ALL,
            PromptSubset::Gsm8k => GSM8K,
            PromptSubset::HumanEval => HUMAN_EVAL,
            PromptSubset::C4 => C4,
        }
    }

    fn as_tag(self) -> &'static str {
        match self {
            PromptSubset::All => "all",
            PromptSubset::Gsm8k => "gsm8k",
            PromptSubset::HumanEval => "humaneval",
            PromptSubset::C4 => "c4",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "all" => Some(PromptSubset::All),
            "gsm8k" => Some(PromptSubset::Gsm8k),
            "humaneval" => Some(PromptSubset::HumanEval),
            "c4" => Some(PromptSubset::C4),
            _ => None,
        }
    }
}

/// Parse command-line arguments.
struct BenchArgs {
    config_path: Option<String>,
    model_path: String,
    max_output_tokens: usize,
    prompt_tokens: usize,
    training_steps: usize,
    skip_training: bool,
    /// When true, latency phase routes through PagedKvCache + model_forward_paged
    /// (the production HTTP/scheduler path). Default false keeps the original
    /// non-paged contiguous KvCache + model_forward path so prior numbers stay
    /// comparable.
    paged: bool,
    /// When true, stop after latency and emit JSON with empty throughput
    /// results and no training result. This keeps rapid decode-path iteration
    /// from paying unrelated benchmark costs.
    latency_only: bool,
    /// Number of throwaway latency runs to execute before the measured run.
    /// This keeps low-level kernel A/Bs from mixing first-use Metal/Candle
    /// compilation latency into prefill and decode timing.
    latency_warmup_runs: usize,
    /// RNG seed threaded through `SamplingParams` and `StdRng` sites so bench
    /// runs are fully reproducible. Phase B3 multi-prompt A/B relies on varying
    /// this across {0..=7} to get independent prompt/sampling trajectories.
    seed: u64,
    /// When true, wrap the MTP bench prompt in the tokenizer's chat template
    /// (Qwen ChatML framing) before encoding. Phase C35 H13 residual A/B —
    /// tests whether raw-prose prompts cause the α degradation vs the paper.
    /// Only affects the MTP bench arm (`--spec-method mtp`); ignored for
    /// skip-layer and off.
    chat_template: bool,
    /// Which subset of PROMPT_POOL the MTP bench draws from (default: all).
    /// Phase C39 domain isolation — see `PromptSubset` docs.
    prompt_subset: PromptSubset,
    /// Sampling temperature threaded through `SamplingParams` at every bench
    /// site. Default 0.0 preserves greedy decode (byte-identical to all prior
    /// bench numbers). Phase C40b — tests greedy-is-uniquely-harmful hypothesis
    /// on code MTP α (HumanEval).
    temperature: f32,
    /// Optional benchmark-only method override. When omitted, the typed
    /// `[speculative]` startup configuration is authoritative.
    spec_method: Option<SpecMethod>,
    /// Optional benchmark-only draft-window override.
    spec_num_tokens: Option<usize>,
    /// Optional benchmark-only draft-depth override.
    spec_draft_layers: Option<usize>,
    /// Bypass the benchmark shape router and exercise raw MTP.
    force_mtp: bool,
    /// Emit every token ID generated by the paged baseline arm.
    log_tokens: bool,
    /// Emit every inter-token latency measured by the paged baseline arm.
    log_itl: bool,
    /// Acknowledge that speculative benchmark paths are experimental and do
    /// not represent supported serving behavior.
    allow_experimental_speculative: bool,
    /// `-v` / `-vv`: bump tracing filter (info → debug → trace). Default keeps
    /// the bench output clean by suppressing per-site tracing chatter.
    verbose: u8,
    /// `--quiet` / `-q`: drop tracing to `warn` only. Wins over `--verbose`.
    quiet: bool,
}

fn parse_args_from(args: &[String]) -> Result<BenchArgs> {
    fn value<'a>(args: &'a [String], index: &mut usize, name: &str) -> Result<&'a str> {
        *index += 1;
        args.get(*index)
            .map(String::as_str)
            .with_context(|| format!("{name} requires a value"))
    }

    fn number<T>(args: &[String], index: &mut usize, name: &str) -> Result<T>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let raw = value(args, index, name)?;
        raw.parse::<T>()
            .map_err(|error| anyhow::anyhow!("{name} has invalid value {raw:?}: {error}"))
    }

    let mut config_path = None;
    let mut model_path = String::new();
    let mut max_output_tokens = 128;
    let mut prompt_tokens = 512;
    let mut training_steps = 10;
    let mut skip_training = false;
    let mut paged = false;
    let mut latency_only = false;
    let mut latency_warmup_runs = 0usize;
    let mut seed: u64 = 42;
    let mut chat_template = false;
    let mut prompt_subset = PromptSubset::All;
    let mut temperature: f32 = 0.0;
    let mut spec_method = None;
    let mut spec_num_tokens = None;
    let mut spec_draft_layers = None;
    let mut force_mtp = false;
    let mut log_tokens = false;
    let mut log_itl = false;
    let mut allow_experimental_speculative = false;
    let mut verbose: u8 = 0;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                config_path = Some(value(&args, &mut i, "--config")?.to_string());
            }
            "--model-path" => {
                model_path = value(&args, &mut i, "--model-path")?.to_string();
            }
            "--max-output-tokens" => {
                max_output_tokens = number(&args, &mut i, "--max-output-tokens")?;
            }
            "--prompt-tokens" => {
                prompt_tokens = number(&args, &mut i, "--prompt-tokens")?;
            }
            "--training-steps" => {
                training_steps = number(&args, &mut i, "--training-steps")?;
            }
            "--skip-training" => {
                skip_training = true;
            }
            "--paged" => {
                paged = true;
            }
            "--latency-only" => {
                latency_only = true;
            }
            "--latency-warmup-runs" => {
                latency_warmup_runs = number(&args, &mut i, "--latency-warmup-runs")?;
            }
            "--seed" => {
                seed = number(&args, &mut i, "--seed")?;
            }
            "--chat-template" => {
                chat_template = true;
            }
            "--prompt-subset" => {
                let s = value(&args, &mut i, "--prompt-subset")?;
                prompt_subset = PromptSubset::parse(&s).ok_or_else(|| {
                    anyhow::anyhow!(
                        "invalid --prompt-subset value '{s}' (expected all|gsm8k|humaneval|c4)"
                    )
                })?;
            }
            "--temperature" => {
                temperature = number(&args, &mut i, "--temperature")?;
            }
            "--spec-method" => {
                let raw = value(&args, &mut i, "--spec-method")?;
                spec_method = Some(SpecMethod::parse_env(raw).ok_or_else(|| {
                    anyhow::anyhow!(
                        "invalid --spec-method value {raw:?} (expected off|skip_layer|mtp)"
                    )
                })?);
            }
            "--spec-num-tokens" => {
                spec_num_tokens = Some(number(&args, &mut i, "--spec-num-tokens")?);
            }
            "--spec-draft-layers" => {
                spec_draft_layers = Some(number(&args, &mut i, "--spec-draft-layers")?);
            }
            "--force-mtp" => {
                force_mtp = true;
            }
            "--log-tokens" => {
                log_tokens = true;
            }
            "--log-itl" => {
                log_itl = true;
            }
            "--allow-experimental-speculative" => {
                allow_experimental_speculative = true;
            }
            "--verbose" | "-v" => {
                verbose = verbose.saturating_add(1);
            }
            "-vv" => {
                verbose = verbose.saturating_add(2);
            }
            "--quiet" | "-q" => {
                quiet = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: kiln-bench --model-path <path> [options]");
                eprintln!(
                    "  --config <path>           Path to kiln.toml (defaults to normal discovery)"
                );
                eprintln!("  --model-path <path>       Path to Qwen3.5-4B weights directory");
                eprintln!(
                    "  --max-output-tokens <n>   Max tokens to generate per request (default: 128)"
                );
                eprintln!(
                    "  --prompt-tokens <n>       Approximate prompt length in tokens (default: 512)"
                );
                eprintln!("  --training-steps <n>      Number of SFT training steps (default: 10)");
                eprintln!("  --skip-training           Skip training benchmarks");
                eprintln!(
                    "  --paged                   Route latency phase through PagedKvCache + model_forward_paged"
                );
                eprintln!(
                    "                            (matches the HTTP/scheduler production path)"
                );
                eprintln!(
                    "  --latency-only            Stop after latency and skip training/throughput"
                );
                eprintln!(
                    "  --latency-warmup-runs <n> Run n throwaway latency passes before measurement"
                );
                eprintln!(
                    "  --seed <u64>              RNG seed + prompt selector from 8-prompt pool (default: 42)"
                );
                eprintln!(
                    "  --chat-template           Wrap MTP bench prompt in Qwen ChatML framing before encoding"
                );
                eprintln!(
                    "                            (Phase C35 H13 A/B; MTP arm only; no-op for off/skip-layer)"
                );
                eprintln!(
                    "  --prompt-subset <name>    Filter PROMPT_POOL for MTP bench: all|gsm8k|humaneval|c4"
                );
                eprintln!(
                    "                            (default: all; Phase C39 domain isolation; MTP arm only)"
                );
                eprintln!(
                    "  --temperature <f32>       Sampling temperature threaded to all bench arms (default: 0.0 = greedy)"
                );
                eprintln!(
                    "                            (Phase C40b — tests greedy-is-uniquely-harmful hypothesis on code MTP α)"
                );
                eprintln!(
                    "  --spec-method <method>    Override typed [speculative] method: off|skip_layer|mtp"
                );
                eprintln!(
                    "  --spec-num-tokens <n>     Override typed speculative draft proposal count"
                );
                eprintln!("  --spec-draft-layers <n>   Override typed speculative draft depth");
                eprintln!(
                    "  --force-mtp               Bypass benchmark shape routing and exercise raw MTP"
                );
                eprintln!("  --log-tokens              Emit paged-baseline generated token IDs");
                eprintln!("  --log-itl                 Emit paged-baseline inter-token latencies");
                eprintln!("  --allow-experimental-speculative");
                eprintln!(
                    "                            Acknowledge unsupported speculative research behavior"
                );
                eprintln!(
                    "  -v, --verbose             Show per-site tracing logs (repeat for trace)"
                );
                eprintln!("  -q, --quiet               Drop tracing to warnings and errors only");
                std::process::exit(0);
            }
            unknown => anyhow::bail!("unknown argument {unknown:?}; run with --help for usage"),
        }
        i += 1;
    }

    if model_path.is_empty() {
        anyhow::bail!("--model-path is required. Run with --help for usage.");
    }
    anyhow::ensure!(
        max_output_tokens > 0,
        "--max-output-tokens must be greater than zero"
    );
    anyhow::ensure!(
        prompt_tokens > 0,
        "--prompt-tokens must be greater than zero"
    );
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "--temperature must be a finite non-negative number"
    );

    Ok(BenchArgs {
        config_path,
        model_path,
        max_output_tokens,
        prompt_tokens,
        training_steps,
        skip_training,
        paged,
        latency_only,
        latency_warmup_runs,
        seed,
        chat_template,
        prompt_subset,
        temperature,
        spec_method,
        spec_num_tokens,
        spec_draft_layers,
        force_mtp,
        log_tokens,
        log_itl,
        allow_experimental_speculative,
        verbose,
        quiet,
    })
}

fn parse_args() -> Result<BenchArgs> {
    let args: Vec<String> = std::env::args().collect();
    parse_args_from(&args)
}

/// Resolve the tracing filter directive from `-v` / `-q` flags. Default is
/// `kiln=warn` so the bench output stays clean — internal tracing chatter only
/// appears when the user opts in. `-v` lifts to `info`, `-vv` to `trace`,
/// `--quiet` clamps to `warn` regardless of `-v`. `RUST_LOG` (handled by
/// `EnvFilter::try_from_default_env`) still wins if set.
fn bench_filter(verbose: u8, quiet: bool) -> &'static str {
    if quiet {
        "kiln=warn,kiln_train=warn"
    } else {
        match verbose {
            0 => "kiln=warn,kiln_train=warn",
            1 => "kiln=info,kiln_train=info",
            _ => "kiln=trace,kiln_train=trace",
        }
    }
}

/// Cyan-bold `▌ heading` for major bench sections. Matches the kiln demo
/// aesthetic in `crates/kiln-server/src/cli.rs::print_banner`.
fn section_header(title: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr);
    let _ = writeln!(
        stderr,
        "  {} {}",
        style("▌").cyan().bold(),
        style(title).cyan().bold()
    );
}

/// Build a TTY-only progress bar over the N sequential runs in a throughput
/// arm. Returns `None` for non-attended stderr (CI, log pipelines) so the
/// JSON-on-stdout path stays clean. Caller is responsible for `inc(1)` per
/// completed run and `finish_and_clear` at the end.
fn make_run_progress(total: u64, label: &str) -> Option<ProgressBar> {
    if !console::Term::stderr().features().is_attended() {
        return None;
    }
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template("  {prefix:.dim} [{bar:24.cyan/blue}] {pos}/{len} {msg:.dim}")
            .expect("static progress template is valid")
            .progress_chars("=>-"),
    );
    pb.set_prefix(label.to_string());
    Some(pb)
}

/// Get the selected accelerator name for benchmark output.
fn gpu_name() -> String {
    if let Some(name) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().lines().next().unwrap_or("unknown").to_string())
        .filter(|s| !s.is_empty() && s != "unknown")
    {
        return name;
    }

    #[cfg(feature = "vulkan")]
    if let Some(name) = runtime_backend::vulkan::vulkan_device_name() {
        if !name.is_empty() {
            return name;
        }
    }

    "unknown".to_string()
}

fn vram_probe_selector_for_device(device: &kiln_tensor::Device) -> VramProbeSelector {
    device.memory_probe_selector()
}

/// Get current VRAM usage for the benchmark's selected accelerator.
fn current_vram_used_bytes(selector: VramProbeSelector) -> u64 {
    detect_used_vram_bytes_for(selector).unwrap_or(0)
}

/// 30 distinct prompt bases spanning three domains, indexed by `seed % 30`.
/// Replaces the 8-prompt prose-only pool used in Phases B2-C37 after Phase C37
/// (kiln PR #369) found that seed variance was dominated by prompt re-sampling
/// (seeds 8+ wrapped back to seed 0's prompt), so the N=10 α CI was a prompt-
/// variance CI rather than a true seed-variance CI. C38 expands the pool to 30
/// prompts so seeds 0-29 each hit a distinct base prompt, and adds domain
/// diversity across three structural token distributions:
///   - 0-9:   GSM8K-style grade-school math word problems (prose format)
///   - 10-19: HumanEval-style Python function signatures + docstrings
///   - 20-29: C4-style natural English text fragments
/// This exercises MTP acceptance across math prose, source code, and
/// general-domain English, producing a true domain-balanced variance band.
const PROMPT_POOL: [&str; 30] = [
    // === 0-9: GSM8K-style grade-school math word problems ===
    // 0: eggs-per-day revenue
    "Janet's ducks lay sixteen eggs per day. She eats three for breakfast every morning and bakes muffins for her friends with four more. She sells the remainder at the local farmers' market daily for two dollars per fresh duck egg. We want to know how much she makes every day at the farmers' market. First subtract eaten and baked eggs from the daily lay, then multiply the leftover count by the per-egg price. ",
    // 1: robe bolts
    "A robe takes two bolts of blue fiber and half that much white fiber. The bolts are purchased separately from two different mills, each with its own shipping schedule. We need the total number of bolts required to make a single robe for one customer at the shop. Half of two bolts is one bolt of white fiber, and that amount is added to the original two bolts of blue. ",
    // 2: house flip profit
    "Josh buys a run-down property for eighty thousand dollars and then spends fifty thousand more on repairs. The renovation increases the value of the house by one hundred and fifty percent over the original purchase price. We want the profit after selling at the appreciated market price. Compute the new value using the percentage increase, then subtract the purchase price and the repair cost to find the net profit. ",
    // 3: weekly sprint distance
    "James decides to run three sprints three times each week as part of his off-season training plan. He runs sixty meters during each individual sprint, without rest intervals counted in the distance. We want the total meters he runs each week across every sprint session combined. Three sprints multiplied by sixty meters gives a per-session distance, which is multiplied by the three weekly sessions. ",
    // 4: chicken feed cups
    "Every day, Wendi feeds each of her chickens three cups of mixed feed in three separate meals. Her flock has twenty chickens in total, all fed identically. In the morning she gives the flock fifteen cups, and in the afternoon she gives twenty-five cups. We want the number of cups in the final evening meal. Compute the full daily requirement, then subtract the cups already served to find the remainder. ",
    // 5: glass shelves discount
    "Kylar goes to the store to buy glasses for his new apartment. One glass costs five dollars, but every second glass costs only sixty percent of the regular price. Kylar wants to buy sixteen glasses in total, arranging them across two open shelves. We need the total cost after applying the alternating discount pattern. Count full-price and discounted positions separately, then sum the two partial totals. ",
    // 6: bakery dozens
    "Toula went to the bakery and bought three dozen donuts at sixty-eight dollars per dozen, two dozen mini cupcakes at eighty dollars per dozen, and six dozen mini cheesecakes at fifty-five dollars per dozen. We want the total cost of her whole pastry order as it appears on the receipt. Compute each line separately by multiplying quantity by unit price, then add the three line totals. ",
    // 7: lemon tree break-even
    "Carlos plants a lemon tree that costs ninety dollars to plant, including labor and the sapling. Each year the tree yields seven lemons, which he sells at the market for one dollar and fifty cents each. Watering and feeding the tree cost him three dollars per year during the growing season. We want the number of years before he starts earning net money on the tree overall. ",
    // 8: vacuum starting count
    "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners at a neighborhood on the east side of town during her morning route. She then sold two more to her cousin, who runs a small rental business. She is left with five vacuum cleaners in the boot of her car. We want the number she started with at the beginning of the day. ",
    // 9: mountain round trips
    "Stephen made ten round trips up and down a forty thousand foot tall mountain over the course of the last week. He reached three quarters of the mountain's height on each of his round trips before turning around. We want the total distance in feet he covered across every round trip combined. Compute the effective height per trip, double it for the round trip, then multiply by the number of trips. ",
    // === 10-19: HumanEval-style Python function signatures with docstrings ===
    // 10: has_close_elements
    "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check whether any two numbers in the input list are closer together than the given threshold. Return True if such a pair exists, otherwise return False. Both arguments are guaranteed to be valid, and the threshold is always positive. \"\"\"\n",
    // 11: separate_paren_groups
    "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Split the input string of balanced parenthesis groups into the individual groups. Ignore any whitespace in the input. Each returned group is balanced on its own and contains no outer whitespace characters. \"\"\"\n",
    // 12: truncate_number
    "def truncate_number(number: float) -> float:\n    \"\"\"Return the fractional part of a positive floating point number. For instance, calling truncate_number on three point five yields zero point five. The return value is always in the half open interval from zero to one. \"\"\"\n",
    // 13: below_zero
    "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\"Given a list of bank account deposit and withdrawal integers applied in order, return True if the running balance ever drops below zero. Otherwise return False when the balance stays non-negative throughout the whole sequence. \"\"\"\n",
    // 14: mean_absolute_deviation
    "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\"Compute the mean absolute deviation of a non-empty list of real numbers. The result is the average absolute difference between each element and the arithmetic mean of the entire list. The list is never empty. \"\"\"\n",
    // 15: intersperse
    "from typing import List\n\ndef intersperse(numbers: List[int], delimiter: int) -> List[int]:\n    \"\"\"Insert the delimiter integer between every pair of consecutive numbers from the input list. The delimiter is not added before the first element or after the last element. Empty input yields an empty list. \"\"\"\n",
    // 16: parse_nested_parens
    "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\"Given a space separated string of parenthesis groups, return the maximum nesting depth of each group as a list of integers. Each group is independently balanced. The returned list has one entry per space separated group. \"\"\"\n",
    // 17: filter_by_substring
    "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\"Return only the strings from the input that contain the given substring. Preserve the original order of occurrence. An empty input list yields an empty output list. \"\"\"\n",
    // 18: sum_product
    "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\"Return a tuple consisting of the sum and the product of all integers in the input list. An empty list must yield a sum of zero and a product of one, matching the standard neutral elements. \"\"\"\n",
    // 19: rolling_max
    "from typing import List\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\"Return a list of rolling maxima ending at each prefix of the input sequence. The i-th element of the output is the maximum of all input numbers from index zero through index i inclusive. \"\"\"\n",
    // === 20-29: C4-style natural English text fragments ===
    // 20: weather forecast
    "The forecast for next Tuesday calls for widespread thunderstorms across the central plains, with scattered severe cells developing through the late afternoon. Meteorologists have already raised the flood watch for several counties along the river corridor. Local emergency coordinators are urging residents near low-lying creeks to prepare sandbags and monitor updated guidance from the national weather service. ",
    // 21: bus rapid transit
    "The city council voted narrowly on Tuesday to approve a new bus rapid transit corridor that will connect the downtown district to the eastern suburbs over the next four years. Supporters described the plan as a critical step toward reducing highway congestion. Opponents raised concerns about impacts on small businesses along the proposed route and the source of matching federal dollars. ",
    // 22: sea turtle migration
    "Researchers at the coastal marine institute have documented an unusual migration pattern in juvenile sea turtles this season. Using satellite tags attached to the rear carapace, the team tracked individual animals crossing two major gyre systems earlier than in any previous recorded year. The preliminary data will be presented at an international conservation conference in the fall. ",
    // 23: theater renovation
    "A quiet renovation of the historic downtown theater has drawn praise from preservation advocates and a few complaints from nearby residents. The restoration preserves the original art deco ceiling mural and the marble foyer while adding a modern climate control system behind the existing plaster walls. The theater will reopen with a retrospective film festival the weekend after the holiday. ",
    // 24: healthcare cybersecurity
    "New federal guidance published this week outlines updated cybersecurity requirements for medium and large healthcare providers across the country. The rules focus on access logging, encryption of patient records at rest, and mandatory quarterly vulnerability assessments. Industry groups have requested a longer implementation window, arguing that smaller hospital networks will struggle to meet the initial compliance deadline. ",
    // 25: garden tour
    "The annual community garden tour drew a record turnout on Saturday as visitors walked through more than two dozen backyard plots across three neighborhoods. Organizers highlighted water conservation strategies, native pollinator beds, and the growing popularity of no till methods among first year gardeners. Proceeds from ticket sales will fund the community seed library for the following growing season. ",
    // 26: small model conference
    "A regional technology conference in the mountain states this month focused on the practical application of small language models to field service and logistics problems. Several vendors demonstrated on device assistants running on laptop class hardware. Conference organizers said registration exceeded last year's total by roughly one third, with attendance skewing toward smaller enterprise operators. ",
    // 27: stone fruit harvest
    "Local farmers are reporting mixed results for this year's stone fruit harvest after a colder than usual spring. Peaches and nectarines show strong volume in the northern orchards, while apricot yields in the southern valley are down by almost a quarter compared to the five year average. The state agricultural commission plans to release a full post harvest summary in late October. ",
    // 28: steam launch restoration
    "The maritime museum unveiled a fully restored steam launch on Friday morning in front of a small crowd gathered at the main pier. Volunteers spent more than four years documenting and rebuilding the hull, original boiler, and brass fittings to their original working condition. Short demonstration cruises are planned on the first Saturday of each month throughout the summer. ",
    // 29: remote work vacancy study
    "A new study from the university economics department examines the long term effect of remote work on mid size commercial real estate vacancy rates. Using lease data from twenty three cities, the researchers observed a clear divergence between coastal and interior markets beginning in the middle of the decade. The full paper is scheduled for peer reviewed publication later this year. ",
];

/// Build a prompt string of approximately `target_tokens` tokens by repeating sentences.
/// `seed` selects which base prompt to use from an 8-prompt pool (via `seed % 8`).
/// Seed 0 reproduces the original Phase B2 baseline; other seeds use distinct content
/// so a multi-prompt A/B actually varies the token distribution seen by the model.
fn build_prompt(tokenizer: &KilnTokenizer, target_tokens: usize, seed: u64) -> String {
    build_prompt_with_subset(tokenizer, target_tokens, seed, PromptSubset::All)
}

/// Like `build_prompt` but restricts selection to the subset's indices.
///
/// `seed` indexes `subset.indices()` modulo its length. With the 10-prompt
/// domain subsets, N=10 covers each prompt once and N=20 covers each twice,
/// which keeps per-seed variance a pure sampling effect rather than a prompt
/// re-hit artifact (the bug C37/C38 exposed in the old 8-prompt pool).
fn build_prompt_with_subset(
    tokenizer: &KilnTokenizer,
    target_tokens: usize,
    seed: u64,
    subset: PromptSubset,
) -> String {
    let idxs = subset.indices();
    let base = PROMPT_POOL[idxs[(seed % idxs.len() as u64) as usize]];

    let mut prompt = String::new();
    loop {
        prompt.push_str(base);
        let tokens = tokenizer.encode(&prompt).unwrap_or_default();
        if tokens.len() >= target_tokens {
            // Trim back to approximately target length
            while tokenizer.encode(&prompt).unwrap_or_default().len() > target_tokens {
                if let Some(pos) = prompt.rfind(". ") {
                    prompt.truncate(pos + 1);
                } else {
                    break;
                }
            }
            return prompt;
        }
    }
}

/// Benchmark inference throughput.
///
/// Runs `num_runs` sequential generations and reports aggregate throughput.
/// This measures single-request performance (not continuous batching).
fn bench_inference(
    runner: &ModelRunner,
    tokenizer: &KilnTokenizer,
    num_runs: usize,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    temperature: f32,
    vram_probe_selector: VramProbeSelector,
) -> Result<InferenceBenchResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens, seed);
    let actual_prompt_tokens = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .len();

    let params = SamplingParams {
        temperature,
        top_p: 1.0,
        top_k: 0,
        max_tokens: max_output_tokens,
        repetition_penalty: 1.0,
        stop: vec![],
        seed: Some(seed),
        ..SamplingParams::default()
    };

    // Warmup
    let warmup_params = SamplingParams {
        max_tokens: 4,
        ..params.clone()
    };
    let _ = runner.generate(&prompt, &warmup_params);

    // Timed runs — TTY users see a progress bar; CI gets one INFO log per run.
    let pb = make_run_progress(num_runs as u64, &format!("{num_runs} runs"));
    let mut total_output_tokens = 0usize;
    let overall_start = Instant::now();

    for i in 0..num_runs {
        let run_start = Instant::now();
        let output = runner
            .generate(&prompt, &params)
            .context("generation failed")?;
        let run_time = run_start.elapsed();
        let gen_tokens = output.token_ids.len();
        total_output_tokens += gen_tokens;

        let run_tps = gen_tokens as f64 / run_time.as_secs_f64();
        if let Some(pb) = pb.as_ref() {
            pb.set_message(format!("{} tok @ {:.0} tok/s", gen_tokens, run_tps));
            pb.inc(1);
        } else {
            tracing::info!(
                target: "kiln_bench",
                run = i + 1,
                total = num_runs,
                tokens = gen_tokens,
                ms = run_time.as_secs_f64() * 1000.0,
                tok_per_sec = run_tps,
                "throughput run"
            );
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    let total_time = overall_start.elapsed();
    let peak_vram = current_vram_used_bytes(vram_probe_selector) / (1024 * 1024);

    Ok(InferenceBenchResult {
        batch_size: num_runs,
        prompt_tokens: actual_prompt_tokens,
        output_tokens: total_output_tokens,
        total_time_secs: total_time.as_secs_f64(),
        tokens_per_sec: total_output_tokens as f64 / total_time.as_secs_f64(),
        peak_vram_mb: peak_vram,
    })
}

/// Benchmark latency by directly timing prefill and each decode step.
///
/// Uses `model_forward_kt` directly for precise per-step timing.
fn bench_latency(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<LatencyResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens, 0);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device_kt = weights.device_kt();
    let dtype = kiln_config_dtype_to_kt(config.dtype);

    let max_total = actual_prompt_tokens + max_output_tokens;
    let mut kv_cache = KvCache::new_kt(
        config.num_full_attention_layers,
        config.num_kv_heads,
        config.head_dim,
        max_total,
        dtype,
        &device_kt,
    )?;
    let backend = runtime_backend_for_bench(&device_kt, weights)?;
    // #1082 forward-flip: `LinearAttentionState::new_with_batch_for_inference_backend`
    // now takes a kt `&Device`, so pass `&device_kt` directly (no candle bridge).
    let mut linear_state = LinearAttentionState::new_with_batch_for_inference_backend(
        config,
        1,
        &device_kt,
        Some(BackendIdentity::runtime_name(backend.as_ref())),
    )?;

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!("  Measuring latency ({actual_prompt_tokens} prompt tokens)...");

    // Prefill: forward pass on all prompt tokens
    let prefill_start = Instant::now();
    let logits = model_forward_kt_with_policy(
        &*backend,
        &prompt_token_ids,
        weights,
        config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
        streaming_prefill,
    )
    .context("prefill forward pass failed")?;
    kv_cache.advance(actual_prompt_tokens);

    // Sample first token
    let mut next_token = greedy_sample(&logits)?;
    let prefill_time = prefill_start.elapsed();

    eprintln!(
        "    Prefill: {:.1}ms ({:.0} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        actual_prompt_tokens as f64 / prefill_time.as_secs_f64()
    );

    // Decode: time each individual step
    let mut inter_token_ms: Vec<f64> = Vec::new();
    let mut num_tokens = 1usize; // counting the first token from prefill

    for _step in 0..max_output_tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }

        let step_start = Instant::now();
        let logits = model_forward_kt_with_policy(
            &*backend,
            &[next_token],
            weights,
            config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
            streaming_prefill,
        )
        .context("decode forward pass failed")?;
        kv_cache.advance(1);
        next_token = greedy_sample(&logits)?;
        let step_time = step_start.elapsed();

        inter_token_ms.push(step_time.as_secs_f64() * 1000.0);
        num_tokens += 1;
    }

    let mean_itl = if inter_token_ms.is_empty() {
        0.0
    } else {
        inter_token_ms.iter().sum::<f64>() / inter_token_ms.len() as f64
    };

    let mut sorted = inter_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };

    let p99 = if sorted.is_empty() {
        0.0
    } else {
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    };

    let decode_tok_per_sec = if inter_token_ms.is_empty() {
        0.0
    } else {
        let total_decode_ms: f64 = inter_token_ms.iter().sum();
        inter_token_ms.len() as f64 / (total_decode_ms / 1000.0)
    };

    eprintln!(
        "    Decode: {num_tokens} tokens, mean ITL {:.1}ms ({:.1} tok/s)",
        mean_itl, decode_tok_per_sec
    );

    Ok(LatencyResult {
        prompt_tokens: actual_prompt_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: actual_prompt_tokens as f64 / prefill_time.as_secs_f64(),
        time_to_first_token_ms: prefill_time.as_secs_f64() * 1000.0,
        mean_inter_token_ms: mean_itl,
        p50_inter_token_ms: p50,
        p99_inter_token_ms: p99,
        num_tokens_generated: num_tokens,
        decode_tokens_per_sec: decode_tok_per_sec,
        spec_method: "off".to_string(),
        acceptance_rate: None,
        prompt_subset: None,
    })
}

/// Benchmark latency along the PAGED production path.
///
/// Mirrors `bench_latency` but uses `PagedKvCacheKt` + `BlockTable` +
/// `model_forward_paged` (the same code path the HTTP server / scheduler
/// drives). This is what production inference actually runs; the non-paged
/// `bench_latency` measures a code path that no real request takes.
///
/// Block size is fixed at `PAGED_BLOCK_SIZE` (matches kiln-core default).
/// A single sequence is allocated `ceil(max_total / block_size)` physical
/// blocks, mapped sequentially. CUDA graph capture is bypassed (we call
/// `model_forward_paged` directly) for apples-to-apples timing with the
/// non-paged latency phase.
fn bench_latency_paged(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    temperature: f32,
    streaming_prefill: StreamingPrefillExecutionPolicy,
    log_tokens: bool,
    log_itl: bool,
) -> Result<LatencyResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens, seed);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device_kt = weights.device_kt();
    let dtype = kiln_config_dtype_to_kt(config.dtype);

    let max_total = actual_prompt_tokens + max_output_tokens;
    let num_blocks = (max_total + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;

    // #1082 candle-drop: candle `PagedKvCache::new_kt(&device_kt, ...)` ->
    // kt `PagedKvCacheKt::new(..., device)`. Pools are allocated on the
    // model's runtime `Device` so the per-layer K/V writes match it.
    let paged_cache = PagedKvCacheKt::new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device_kt,
    )?;

    // Phase 7 #1082: first end-to-end PagedKvCacheKt production wiring.
    // When `accelerator.kt_api_mode = "all"` is on
    // AND we're on a CUDA device, allocate a kt twin alongside the
    // candle `paged_cache` and pass it to `model_forward_paged_with_kt`
    // below so every paged-KV write inside the GQA attention writer
    // mirrors into the kt cache. When the gate is off (the default) or
    // the device isn't CUDA, `try_kt_paged_kv_cache_new` returns `Ok(None)`
    // and the decode loop is bit-identical to the previous behavior.
    //
    // The candle `paged_cache` remains authoritative for reads; the kt
    // mirror only exercises the writer surface — that's enough to
    // validate constructor + writer end-to-end on a real production
    // workload (the latency bench).
    //
    // #1082 forward-flip: `try_kt_paged_kv_cache_new` now takes kt
    // `DType`/`Device` directly, so pass `dtype` and `&device_kt` without
    // bridging to candle.
    #[cfg(feature = "cuda")]
    let paged_cache_kt = kiln_model::forward::try_kt_paged_kv_cache_new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        &device_kt,
    )?;
    #[cfg(feature = "cuda")]
    if let Some(ref kt) = paged_cache_kt {
        eprintln!(
            "  Phase 7 #1082: PagedKvCacheKt twin allocated (layers={}, blocks={}, \
             block_size={}, fp8={}); decode loop will mirror writes",
            kt.num_layers(),
            kt.num_blocks(),
            kt.block_size(),
            kt.is_fp8(),
        );
    }

    let backend = runtime_backend_for_bench(&device_kt, weights)?;
    let backend_capabilities = BackendCapabilityQueries::backend_capabilities(backend.as_ref());
    let greedy_token_decode_enabled = backend_capabilities.decode.linear_argmax.is_native()
        || backend_capabilities.decode_batcher.use_greedy_token_decode;
    let mut rocm_graph = kiln_model::rocm_graph::RocmGraphRunner::new(
        &device_kt,
        kiln_model::RocmGraphExecutionPolicy::lazy_capture_replay(),
    );
    // This runner is scoped to exactly one benchmark generation, so a fixed
    // concrete owner is unique for its entire lifetime.
    let rocm_graph_row_id = 1_u64;
    let hip_graph_decode_enabled = bench_paged_decode_replay_primitive_enabled(
        backend.as_ref(),
        config,
        ReplayNativePrimitive::HipGraph,
    ) && rocm_graph.is_enabled();
    // #1082 forward-flip: `LinearAttentionState::new_with_batch_for_inference_backend`
    // now takes a kt `&Device`, so pass `&device_kt` directly (no candle bridge).
    let mut linear_state = LinearAttentionState::new_with_batch_for_inference_backend(
        config,
        1,
        &device_kt,
        Some(BackendIdentity::runtime_name(backend.as_ref())),
    )?;

    // Build a block table that maps logical block i -> physical block i (sequential).
    let mut block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        block_table.push(i);
    }

    let eos_token_ids = tokenizer.eos_token_ids();
    let mut sampling_params = SamplingParams::greedy();
    sampling_params.temperature = temperature;
    sampling_params.seed = Some(seed);
    sampling_params.max_tokens = max_output_tokens;
    let sampled_decode = temperature.is_finite() && temperature > 0.0;

    eprintln!(
        "  Measuring latency [PAGED, block_size={PAGED_BLOCK_SIZE}, blocks={num_blocks}] \
         ({actual_prompt_tokens} prompt tokens, temperature={temperature})..."
    );

    // Prefill: forward pass on all prompt tokens via the paged path and the
    // startup-resolved dispatch policy.
    let prefill_start = Instant::now();
    let use_streaming_prefill = streaming_prefill.enabled_for(actual_prompt_tokens);
    let mut next_token = if sampled_decode {
        let hidden = if use_streaming_prefill {
            model_forward_paged_streaming_last_token_hidden_with_policy(
                &*backend,
                &prompt_token_ids,
                weights,
                config,
                &paged_cache,
                &block_table,
                0,
                Some(&mut linear_state),
                None,
                streaming_prefill,
            )
            .context("paged sampled prefill hidden pass (streaming) failed")?
        } else {
            model_forward_paged_last_token_hidden(
                &*backend,
                &prompt_token_ids,
                weights,
                config,
                &paged_cache,
                &block_table,
                0,
                Some(&mut linear_state),
                None,
            )
            .context("paged sampled prefill hidden pass failed")?
        };
        if let Some(token) = lm_head_sample_backend_decode_if(
            Some(&*backend),
            &hidden,
            weights,
            config,
            &sampling_params,
            Some(seed),
            &[],
        )
        .context("paged sampled prefill fused lm-head sample failed")?
        {
            token
        } else {
            let logits =
                model_forward_head_backend_decode_if(Some(&*backend), &hidden, weights, config)
                    .context("paged sampled prefill lm-head fallback failed")?;
            sample_step(&logits, &sampling_params, Some(seed), &[])
                .context("paged sampled prefill host sample failed")?
        }
    } else if use_streaming_prefill {
        let logits = model_forward_paged_streaming_with_policy(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            streaming_prefill,
        )
        .context("paged prefill forward pass (streaming) failed")?;
        greedy_sample_kt(&logits)?
    } else if greedy_token_decode_enabled {
        model_forward_paged_last_token_greedy(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("paged greedy prefill forward pass failed")?
    } else {
        let logits = model_forward_paged_last_token(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("paged prefill forward pass failed")?;
        greedy_sample_kt(&logits)?
    };
    let prefill_time = prefill_start.elapsed();

    eprintln!(
        "    Prefill (paged): {:.1}ms ({:.0} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        actual_prompt_tokens as f64 / prefill_time.as_secs_f64()
    );

    // Decode: time each individual step.
    // The paged path tracks position via `start_pos` (no advance() like KvCache).
    let mut inter_token_ms: Vec<f64> = Vec::new();
    let mut num_tokens = 1usize; // counting the first token from prefill
    let mut current_pos = actual_prompt_tokens;
    let mut decoded_tokens: Vec<u32> = Vec::new();
    let mut generated_tokens: Vec<TokenId> = vec![next_token];
    if log_tokens {
        decoded_tokens.push(next_token);
    }

    let _resident_state_scope = BenchGdnRecurrentResidentStateScope::new(backend.as_ref());
    for step in 0..max_output_tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }

        let step_start = Instant::now();
        next_token = if sampled_decode {
            let hidden = if hip_graph_decode_enabled {
                rocm_graph
                    .decode_step_paged_hidden(
                        &*backend,
                        next_token,
                        weights,
                        config,
                        &paged_cache,
                        &block_table,
                        current_pos,
                        &mut linear_state,
                        None,
                        rocm_graph_row_id,
                    )
                    .context("paged sampled ROCm graph hidden pass failed")?
            } else {
                let sequence_lengths = [current_pos];
                let mut linear_states: [&mut LinearAttentionState; 1] = [&mut linear_state];
                model_forward_paged_batched_decode_hidden(
                    &*backend,
                    &[next_token],
                    weights,
                    config,
                    &paged_cache,
                    std::slice::from_ref(&block_table),
                    &sequence_lengths,
                    &mut linear_states,
                    None,
                )
                .context("paged sampled decode hidden pass failed")?
            };
            let step_seed = seed.wrapping_add(num_tokens as u64);
            if let Some(token) = lm_head_sample_backend_decode_if(
                Some(&*backend),
                &hidden,
                weights,
                config,
                &sampling_params,
                Some(step_seed),
                &generated_tokens,
            )
            .context("paged sampled decode fused lm-head sample failed")?
            {
                token
            } else {
                let logits =
                    model_forward_head_backend_decode_if(Some(&*backend), &hidden, weights, config)
                        .context("paged sampled decode lm-head fallback failed")?;
                sample_step(
                    &logits,
                    &sampling_params,
                    Some(step_seed),
                    &generated_tokens,
                )
                .context("paged sampled decode host sample failed")?
            }
        } else if hip_graph_decode_enabled {
            rocm_graph
                .decode_step_paged_greedy(
                    &*backend,
                    next_token,
                    weights,
                    config,
                    &paged_cache,
                    &block_table,
                    current_pos,
                    &mut linear_state,
                    None,
                    rocm_graph_row_id,
                )
                .context("paged ROCm graph greedy decode forward pass failed")?
        } else if greedy_token_decode_enabled {
            model_forward_paged_next_token_greedy(
                &*backend,
                next_token,
                weights,
                config,
                &paged_cache,
                &block_table,
                current_pos,
                Some(&mut linear_state),
                None,
                None,
            )
            .context("paged greedy decode forward pass failed")?
        } else {
            // Phase 7 #1082: route through the kt-aware entry point.
            // `paged_cache_kt.as_ref()` is `None` when the env gate is
            // off, making this a no-op vs. `model_forward_paged`. When
            // the gate is on, every paged-KV write inside this fn
            // mirrors into the kt cache.
            let logits = {
                #[cfg(feature = "cuda")]
                {
                    kiln_model::forward::model_forward_paged_with_kt(
                        &*backend,
                        &[next_token],
                        weights,
                        config,
                        &paged_cache,
                        &block_table,
                        current_pos,
                        Some(&mut linear_state),
                        None,
                        None,
                        paged_cache_kt.as_ref(),
                    )
                    .context("paged decode forward pass failed")?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    kiln_model::forward::model_forward_paged(
                        &*backend,
                        &[next_token],
                        weights,
                        config,
                        &paged_cache,
                        &block_table,
                        current_pos,
                        Some(&mut linear_state),
                        None,
                        None,
                    )
                    .context("paged decode forward pass failed")?
                }
            };
            greedy_sample_kt(&logits)?
        };
        current_pos += 1;
        generated_tokens.push(next_token);
        let step_time = step_start.elapsed();

        let step_ms = step_time.as_secs_f64() * 1000.0;
        inter_token_ms.push(step_ms);
        if log_itl {
            eprintln!(
                "    Paged decode step {}: {:.1}ms (pos {})",
                step + 1,
                step_ms,
                current_pos - 1
            );
        }
        #[cfg(feature = "vulkan")]
        kiln_model::vk_decode_resident::drain_resident_decode_timing();
        num_tokens += 1;
        if log_tokens {
            decoded_tokens.push(next_token);
        }
    }

    if log_tokens {
        let first_n: Vec<String> = decoded_tokens
            .iter()
            .take(32)
            .map(|t| t.to_string())
            .collect();
        eprintln!(
            "    Paged decode first 32 token ids: [{}]",
            first_n.join(",")
        );
    }

    let mean_itl = if inter_token_ms.is_empty() {
        0.0
    } else {
        inter_token_ms.iter().sum::<f64>() / inter_token_ms.len() as f64
    };

    let mut sorted = inter_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };

    let p99 = if sorted.is_empty() {
        0.0
    } else {
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    };

    let decode_tok_per_sec = if inter_token_ms.is_empty() {
        0.0
    } else {
        let total_decode_ms: f64 = inter_token_ms.iter().sum();
        inter_token_ms.len() as f64 / (total_decode_ms / 1000.0)
    };

    eprintln!(
        "    Decode (paged): {num_tokens} tokens, mean ITL {:.1}ms ({:.1} tok/s)",
        mean_itl, decode_tok_per_sec
    );

    Ok(LatencyResult {
        prompt_tokens: actual_prompt_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: actual_prompt_tokens as f64 / prefill_time.as_secs_f64(),
        time_to_first_token_ms: prefill_time.as_secs_f64() * 1000.0,
        mean_inter_token_ms: mean_itl,
        p50_inter_token_ms: p50,
        p99_inter_token_ms: p99,
        num_tokens_generated: num_tokens,
        decode_tokens_per_sec: decode_tok_per_sec,
        spec_method: "off".to_string(),
        acceptance_rate: None,
        prompt_subset: None,
    })
}

fn require_speculative_benchmark_opt_in(
    method: SpecMethod,
    allow_experimental: bool,
) -> Result<()> {
    if method == SpecMethod::Off {
        return Ok(());
    }
    anyhow::ensure!(
        allow_experimental,
        "speculative benchmark method {method:?} is experimental and unsupported for serving; pass --allow-experimental-speculative to acknowledge that this benchmark is research, not qualification evidence"
    );
    Ok(())
}

/// Resolve the benchmark-only comparison arm using the historical shape policy.
/// This supports offline accelerator qualification and does not describe server
/// request routing: serving currently rejects every enabled speculative method
/// at startup.
fn resolve_bench_spec_method(
    configured: SpecMethod,
    requested_prompt_tokens: usize,
    max_output_tokens: usize,
    temperature: f32,
    mtp_supported: bool,
    native_mtp_allowed: bool,
    speculative_policy: SpeculativeDecodePolicy,
    force_raw_mtp: bool,
) -> SpecMethod {
    resolve_bench_spec_method_with_force(
        configured,
        requested_prompt_tokens,
        max_output_tokens,
        temperature,
        mtp_supported,
        native_mtp_allowed,
        speculative_policy,
        force_raw_mtp,
    )
}

fn resolve_bench_spec_method_with_force(
    configured: SpecMethod,
    requested_prompt_tokens: usize,
    max_output_tokens: usize,
    temperature: f32,
    mtp_supported: bool,
    native_mtp_allowed: bool,
    speculative_policy: SpeculativeDecodePolicy,
    force_raw_mtp: bool,
) -> SpecMethod {
    match configured {
        SpecMethod::Off => SpecMethod::Off,
        SpecMethod::SkipLayer => SpecMethod::SkipLayer,
        SpecMethod::Mtp => {
            if force_raw_mtp {
                return SpecMethod::Mtp;
            }
            let greedy = temperature == 0.0;
            if mtp_supported
                && native_mtp_allowed
                && greedy
                && requested_prompt_tokens <= speculative_policy.mtp_max_prompt_tokens
            {
                SpecMethod::Mtp
            } else if greedy
                && requested_prompt_tokens
                    >= speculative_policy.long_prompt_skip_layer_min_prompt_tokens
                && max_output_tokens >= speculative_policy.long_prompt_skip_layer_min_output_tokens
            {
                SpecMethod::SkipLayer
            } else {
                SpecMethod::Off
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_memory::vram::LinuxDrmVendor;

    #[test]
    fn vram_probe_selector_tracks_selected_benchmark_device() {
        assert_eq!(
            vram_probe_selector_for_device(&kiln_tensor::Device::Cuda(2)),
            VramProbeSelector::Nvidia(2)
        );
        assert_eq!(
            vram_probe_selector_for_device(&kiln_tensor::Device::Rocm(1)),
            VramProbeSelector::LinuxDrm {
                index: 1,
                vendor: Some(LinuxDrmVendor::Amd),
            }
        );
        assert_eq!(
            vram_probe_selector_for_device(&kiln_tensor::Device::Vulkan(3)),
            VramProbeSelector::LinuxDrm {
                index: 3,
                vendor: None,
            }
        );
        assert_eq!(
            vram_probe_selector_for_device(&kiln_tensor::Device::Metal(4)),
            VramProbeSelector::AppleUnified
        );
        assert_eq!(
            vram_probe_selector_for_device(&kiln_tensor::Device::Cpu),
            VramProbeSelector::None
        );
    }

    fn default_speculative_policy_for_test() -> SpeculativeDecodePolicy {
        SpeculativeDecodePolicy::default()
    }

    fn metal_speculative_policy_for_test() -> SpeculativeDecodePolicy {
        SpeculativeDecodePolicy::for_backend("metal", kiln_tensor::Device::Metal(0))
    }

    #[test]
    fn bench_mtp_short_prompt_stays_mtp() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                128,
                64,
                0.0,
                true,
                true,
                default_speculative_policy_for_test(),
                false
            ),
            SpecMethod::Mtp
        );
    }

    #[test]
    fn bench_mtp_long_greedy_prompt_falls_back_to_skip_layer() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                SpeculativeDecodePolicy::LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
                64,
                0.0,
                true,
                true,
                default_speculative_policy_for_test(),
                false
            ),
            SpecMethod::SkipLayer
        );
    }

    #[test]
    fn bench_mtp_long_short_output_or_sampled_prompt_turns_off() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                SpeculativeDecodePolicy::LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
                31,
                0.0,
                true,
                true,
                default_speculative_policy_for_test(),
                false
            ),
            SpecMethod::Off
        );
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                SpeculativeDecodePolicy::LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_DEFAULT,
                64,
                0.7,
                true,
                true,
                default_speculative_policy_for_test(),
                false
            ),
            SpecMethod::Off
        );
    }

    #[test]
    fn bench_mtp_medium_prompt_stays_off_until_skip_layer_crossover() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                512,
                64,
                0.0,
                true,
                true,
                default_speculative_policy_for_test(),
                false
            ),
            SpecMethod::Off
        );
    }

    #[test]
    fn bench_mtp_short_prompt_stays_off_when_native_mtp_is_disallowed() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                64,
                64,
                0.0,
                true,
                false,
                default_speculative_policy_for_test(),
                false
            ),
            SpecMethod::Off
        );
    }

    #[test]
    fn bench_force_raw_mtp_bypasses_shape_routing() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                8192,
                64,
                0.0,
                true,
                false,
                default_speculative_policy_for_test(),
                true
            ),
            SpecMethod::Mtp
        );
    }

    #[test]
    fn bench_mtp_metal_medium_prompt_stays_off_until_4096() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                2048,
                64,
                0.0,
                true,
                false,
                metal_speculative_policy_for_test(),
                false
            ),
            SpecMethod::Off
        );
    }

    #[test]
    fn bench_mtp_metal_4096_prompt_falls_back_to_skip_layer() {
        assert_eq!(
            resolve_bench_spec_method_with_force(
                SpecMethod::Mtp,
                SpeculativeDecodePolicy::LONG_PROMPT_SKIP_LAYER_MIN_PROMPT_TOKENS_METAL,
                64,
                0.0,
                true,
                false,
                metal_speculative_policy_for_test(),
                false
            ),
            SpecMethod::SkipLayer
        );
    }

    #[test]
    fn bench_filter_default_is_warn() {
        assert_eq!(bench_filter(0, false), "kiln=warn,kiln_train=warn");
    }

    #[test]
    fn bench_filter_v_lifts_to_info() {
        assert_eq!(bench_filter(1, false), "kiln=info,kiln_train=info");
    }

    #[test]
    fn bench_filter_vv_lifts_to_trace() {
        assert_eq!(bench_filter(2, false), "kiln=trace,kiln_train=trace");
        assert_eq!(bench_filter(3, false), "kiln=trace,kiln_train=trace");
    }

    #[test]
    fn bench_filter_quiet_wins_over_verbose() {
        assert_eq!(bench_filter(2, true), "kiln=warn,kiln_train=warn");
    }

    #[test]
    fn benchmark_cli_parses_explicit_speculative_and_diagnostic_controls() {
        let args = [
            "kiln-bench",
            "--model-path",
            "/models/qwen",
            "--spec-method",
            "mtp",
            "--spec-num-tokens",
            "2",
            "--spec-draft-layers",
            "6",
            "--force-mtp",
            "--log-tokens",
            "--log-itl",
            "--allow-experimental-speculative",
        ]
        .map(str::to_owned);
        let parsed = parse_args_from(&args).expect("typed benchmark arguments");
        assert_eq!(parsed.spec_method, Some(SpecMethod::Mtp));
        assert_eq!(parsed.spec_num_tokens, Some(2));
        assert_eq!(parsed.spec_draft_layers, Some(6));
        assert!(parsed.force_mtp);
        assert!(parsed.log_tokens);
        assert!(parsed.log_itl);
        assert!(parsed.allow_experimental_speculative);
    }

    #[test]
    fn benchmark_cli_rejects_unknown_speculative_method() {
        let args = [
            "kiln-bench",
            "--model-path",
            "/models/qwen",
            "--spec-method",
            "maybe",
        ]
        .map(str::to_owned);
        let error = parse_args_from(&args)
            .err()
            .expect("invalid method must fail");
        assert!(error.to_string().contains("invalid --spec-method"));
    }

    #[test]
    fn speculative_benchmark_requires_explicit_experimental_opt_in() {
        let error = require_speculative_benchmark_opt_in(SpecMethod::Mtp, false)
            .expect_err("speculative research must require an explicit acknowledgment");
        assert!(
            error
                .to_string()
                .contains("--allow-experimental-speculative")
        );
    }

    #[test]
    fn non_speculative_benchmark_does_not_require_experimental_opt_in() {
        require_speculative_benchmark_opt_in(SpecMethod::Off, false)
            .expect("the supported baseline must remain directly runnable");
    }
}

/// Benchmark latency along the SKIP-LAYER speculative path.
///
/// Uses the same flat `KvCache` + `model_forward` path as the existing
/// `generate_from_tokens_speculative` in `kiln-model::generate`. Drives
/// `speculative_decode_step` directly per iteration so each step's wall time
/// is divided across the tokens emitted by that step.
///
fn bench_latency_skiplayer(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    temperature: f32,
    streaming_prefill: StreamingPrefillExecutionPolicy,
    speculative: &SpeculativeConfig,
) -> Result<LatencyResult> {
    use rand::SeedableRng;

    let num_speculative_tokens = speculative.num_speculative_tokens;
    let draft_layers = speculative.draft_layers;

    let prompt = build_prompt(tokenizer, prompt_tokens, seed);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device_kt = weights.device_kt();
    let dtype = kiln_config_dtype_to_kt(config.dtype);

    // Skip-layer verifies `[last_token, draft_0, ..., draft_k-1]` in one
    // forward pass. Near the end of generation those speculative KV writes can
    // extend past the committed token budget before stale slots are overwritten
    // or ignored, so reserve headroom for the verify window.
    let max_spec_window = num_speculative_tokens.min(max_output_tokens.max(1));
    let max_total = actual_prompt_tokens
        .checked_add(max_output_tokens)
        .and_then(|tokens| tokens.checked_add(max_spec_window))
        .and_then(|tokens| tokens.checked_add(1))
        .context("skip-layer benchmark cache size overflowed")?;
    let mut kv_cache = KvCache::new_kt(
        config.num_full_attention_layers,
        config.num_kv_heads,
        config.head_dim,
        max_total,
        dtype,
        &device_kt,
    )?;
    let backend = runtime_backend_for_bench(&device_kt, weights)?;
    // #1082 forward-flip: `LinearAttentionState::new_with_batch_for_inference_backend`
    // now takes a kt `&Device`, so pass `&device_kt` directly (no candle bridge).
    let mut linear_state = LinearAttentionState::new_with_batch_for_inference_backend(
        config,
        1,
        &device_kt,
        Some(BackendIdentity::runtime_name(backend.as_ref())),
    )?;

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!(
        "  Measuring latency [SKIP-LAYER, k={num_speculative_tokens}, draft_layers={draft_layers}] \
         ({actual_prompt_tokens} prompt tokens)..."
    );

    // Prefill: full forward pass, no speculative draft yet.
    let prefill_start = Instant::now();
    let logits = model_forward_kt_with_policy(
        &*backend,
        &prompt_token_ids,
        weights,
        config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
        streaming_prefill,
    )
    .context("skip-layer prefill forward pass failed")?;
    kv_cache.advance(actual_prompt_tokens);

    let draft_linear_layers = weights.linear_attention_layers_in_prefix(draft_layers);
    let mut draft_linear_state = linear_state
        .snapshot_for_decode_rollback_prefix(draft_linear_layers)
        .context("clone draft linear-attention prefix from skip-layer prefill")?;

    let mut last_token = greedy_sample(&logits)?;
    let prefill_time = prefill_start.elapsed();

    eprintln!(
        "    Prefill (skip-layer): {:.1}ms ({:.0} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        actual_prompt_tokens as f64 / prefill_time.as_secs_f64()
    );

    // Decode: each speculative step produces 1..=k+1 tokens; per-emitted-token
    // ITL is `step_time / accepted_count` so the resulting distribution is
    // comparable to the Off arm's per-token ITL.
    let mut inter_token_ms: Vec<f64> = Vec::new();
    let mut num_tokens = 1usize; // counting the first token from prefill
    let mut emitted: Vec<u32> = vec![last_token];
    let params = SamplingParams {
        temperature,
        top_p: 1.0,
        top_k: 0,
        max_tokens: max_output_tokens,
        repetition_penalty: 1.0,
        stop: vec![],
        seed: Some(seed),
        ..SamplingParams::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    while num_tokens < max_output_tokens {
        if eos_token_ids.contains(&last_token) {
            break;
        }

        let remaining = max_output_tokens - num_tokens;
        let effective_k = num_speculative_tokens.min(remaining.max(1));
        let effective_config = SpeculativeConfig {
            num_speculative_tokens: effective_k,
            draft_layers,
        };

        let step_start = Instant::now();
        let step = speculative_decode_step(
            &*backend,
            last_token,
            weights,
            config,
            &mut kv_cache,
            &mut linear_state,
            &mut draft_linear_state,
            &effective_config,
            &params,
            &eos_token_ids,
            &mut rng,
            None,
        )
        .context("skip-layer speculative_decode_step failed")?;
        let step_time = step_start.elapsed();

        if step.accepted_tokens.is_empty() {
            break;
        }

        let per_token_ms = (step_time.as_secs_f64() * 1000.0) / step.accepted_tokens.len() as f64;
        for &tok in &step.accepted_tokens {
            inter_token_ms.push(per_token_ms);
            emitted.push(tok);
            num_tokens += 1;
            if num_tokens >= max_output_tokens {
                break;
            }
        }

        last_token = *step.accepted_tokens.last().unwrap();
        if step.hit_eos {
            break;
        }
    }

    let mean_itl = if inter_token_ms.is_empty() {
        0.0
    } else {
        inter_token_ms.iter().sum::<f64>() / inter_token_ms.len() as f64
    };
    let mut sorted = inter_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let p99 = if sorted.is_empty() {
        0.0
    } else {
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    };
    let decode_tok_per_sec = if inter_token_ms.is_empty() {
        0.0
    } else {
        let total_decode_ms: f64 = inter_token_ms.iter().sum();
        inter_token_ms.len() as f64 / (total_decode_ms / 1000.0)
    };

    eprintln!(
        "    Decode (skip-layer): {num_tokens} tokens, mean ITL {:.1}ms ({:.1} tok/s)",
        mean_itl, decode_tok_per_sec
    );

    Ok(LatencyResult {
        prompt_tokens: actual_prompt_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: actual_prompt_tokens as f64 / prefill_time.as_secs_f64(),
        time_to_first_token_ms: prefill_time.as_secs_f64() * 1000.0,
        mean_inter_token_ms: mean_itl,
        p50_inter_token_ms: p50,
        p99_inter_token_ms: p99,
        num_tokens_generated: num_tokens,
        decode_tokens_per_sec: decode_tok_per_sec,
        spec_method: "skip_layer".to_string(),
        acceptance_rate: None,
        prompt_subset: None,
    })
}

/// Benchmark latency along the PAGED SKIP-LAYER speculative path.
///
/// This is benchmark-only scaffolding for the paged self-speculative
/// implementation. It keeps the paged prefill/cache/block-table setup in this
/// harness and delegates each decode iteration to
/// `speculative_decode_step_paged_greedy`, which is expected to verify against
/// `PagedKvCacheKt` at the caller-provided `base_pos`.
///
/// Enable with `--spec-method skip_layer --paged`. Without `--paged`, the
/// legacy flat-KV skip-layer benchmark remains available for comparisons.
fn bench_latency_paged_skiplayer(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    temperature: f32,
    streaming_prefill: StreamingPrefillExecutionPolicy,
    speculative: &SpeculativeConfig,
) -> Result<LatencyResult> {
    let num_speculative_tokens = speculative.num_speculative_tokens;
    let draft_layers = speculative.draft_layers;

    let prompt = build_prompt(tokenizer, prompt_tokens, seed);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device_kt = weights.device_kt();
    let dtype = kiln_config_dtype_to_kt(config.dtype);

    // Verification writes `[last_token, draft_0, ..., draft_k-1]` starting at
    // `base_pos`. Reserve headroom so late-generation verify windows do not
    // run off the allocated paged cache before stale speculative slots are
    // overwritten by the next committed step.
    let max_spec_window = num_speculative_tokens.min(max_output_tokens.max(1));
    let max_total = actual_prompt_tokens
        .checked_add(max_output_tokens)
        .and_then(|tokens| tokens.checked_add(max_spec_window))
        .and_then(|tokens| tokens.checked_add(1))
        .context("paged skip-layer benchmark cache size overflowed")?;
    let num_blocks = max_total.div_ceil(PAGED_BLOCK_SIZE);

    // #1082 candle-drop: `PagedKvCache::new_kt(&device_kt, ...)` ->
    // `PagedKvCacheKt::new(..., device)` — pools on the runtime `Device`.
    let paged_cache = PagedKvCacheKt::new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device_kt,
    )?;
    let backend = runtime_backend_for_bench(&device_kt, weights)?;
    // #1082 forward-flip: the linear state takes a kt `&Device` directly.
    let mut linear_state = LinearAttentionState::new_with_batch_for_inference_backend(
        config,
        1,
        &device_kt,
        Some(BackendIdentity::runtime_name(backend.as_ref())),
    )?;

    let mut block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        block_table.push(i);
    }

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!(
        "  Measuring latency [SKIP-LAYER, paged, k={num_speculative_tokens}, \
         draft_layers={draft_layers}, blocks={num_blocks}] ({actual_prompt_tokens} prompt tokens)..."
    );

    let prefill_start = Instant::now();
    let logits = if streaming_prefill.enabled_for(actual_prompt_tokens) {
        model_forward_paged_streaming_with_policy(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            streaming_prefill,
        )
        .context("paged skip-layer prefill forward pass (streaming) failed")?
    } else {
        model_forward_paged_last_token(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("paged skip-layer prefill forward pass failed")?
    };

    let draft_linear_layers = weights.linear_attention_layers_in_prefix(draft_layers);
    let mut draft_linear_state = linear_state
        .snapshot_for_decode_rollback_prefix(draft_linear_layers)
        .context("clone draft linear-attention prefix from paged skip-layer prefill")?;

    let mut last_token = greedy_sample_kt(&logits)?;
    let prefill_time = prefill_start.elapsed();

    eprintln!(
        "    Prefill (skip-layer paged): {:.1}ms ({:.0} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        actual_prompt_tokens as f64 / prefill_time.as_secs_f64()
    );

    let mut inter_token_ms: Vec<f64> = Vec::new();
    let mut num_tokens = 1usize; // counting the first token from prefill
    let mut base_pos = actual_prompt_tokens;
    let mut accepted_draft_tokens = 0usize;
    let mut attempted_draft_tokens = 0usize;
    let params = SamplingParams {
        temperature,
        top_p: 1.0,
        top_k: 0,
        max_tokens: max_output_tokens,
        repetition_penalty: 1.0,
        stop: vec![],
        seed: Some(seed),
        ..SamplingParams::default()
    };

    while num_tokens < max_output_tokens {
        if eos_token_ids.contains(&last_token) {
            break;
        }

        let remaining = max_output_tokens - num_tokens;
        let effective_config = SpeculativeConfig {
            num_speculative_tokens: num_speculative_tokens.min(remaining.max(1)),
            draft_layers,
        };

        let step_start = Instant::now();
        let step = speculative_decode_step_paged_greedy(
            &*backend,
            last_token,
            weights,
            config,
            &paged_cache,
            &block_table,
            base_pos,
            &mut linear_state,
            &mut draft_linear_state,
            &effective_config,
            &params,
            &eos_token_ids,
            None,
        )
        .context("skip-layer speculative_decode_step_paged_greedy failed")?;
        let step_time = step_start.elapsed();

        if step.accepted_tokens.is_empty() {
            break;
        }
        accepted_draft_tokens += step.accepted_draft_tokens;
        attempted_draft_tokens += step.attempted_draft_tokens;

        let accepted_len = step.accepted_tokens.len();
        let per_token_ms = (step_time.as_secs_f64() * 1000.0) / accepted_len as f64;
        for &tok in &step.accepted_tokens {
            inter_token_ms.push(per_token_ms);
            num_tokens += 1;
            if num_tokens >= max_output_tokens {
                break;
            }
            if eos_token_ids.contains(&tok) {
                break;
            }
        }

        last_token = *step.accepted_tokens.last().unwrap();
        base_pos += step.base_advance;

        if step.hit_eos {
            break;
        }
    }

    let mean_itl = if inter_token_ms.is_empty() {
        0.0
    } else {
        inter_token_ms.iter().sum::<f64>() / inter_token_ms.len() as f64
    };
    let mut sorted = inter_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let p99 = if sorted.is_empty() {
        0.0
    } else {
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    };
    let decode_tok_per_sec = if inter_token_ms.is_empty() {
        0.0
    } else {
        let total_decode_ms: f64 = inter_token_ms.iter().sum();
        inter_token_ms.len() as f64 / (total_decode_ms / 1000.0)
    };

    eprintln!(
        "    Decode (skip-layer paged): {num_tokens} tokens, mean ITL {:.1}ms ({:.1} tok/s)",
        mean_itl, decode_tok_per_sec
    );
    let acceptance_rate = if attempted_draft_tokens > 0 {
        Some(accepted_draft_tokens as f64 / attempted_draft_tokens as f64)
    } else {
        None
    };
    if let Some(rate) = acceptance_rate {
        eprintln!("    Acceptance (skip-layer paged): {:.3}", rate);
    }

    Ok(LatencyResult {
        prompt_tokens: actual_prompt_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: actual_prompt_tokens as f64 / prefill_time.as_secs_f64(),
        time_to_first_token_ms: prefill_time.as_secs_f64() * 1000.0,
        mean_inter_token_ms: mean_itl,
        p50_inter_token_ms: p50,
        p99_inter_token_ms: p99,
        num_tokens_generated: num_tokens,
        decode_tokens_per_sec: decode_tok_per_sec,
        spec_method: "skip_layer_paged".to_string(),
        acceptance_rate,
        prompt_subset: None,
    })
}

/// Benchmark latency along the NATIVE-MTP speculative path.
///
/// Uses two `PagedKvCacheKt` instances (base + 1-layer MTP), threads `h_prev`
/// across iterations, and drives `speculative_mtp_decode_step` per step.
/// Reports α = `draft_accepted / total_draft_attempts`.
///
/// Greedy-only (k=1 native MTP for Qwen3.5-4B).
fn bench_latency_paged_mtp(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    chat_template: bool,
    prompt_subset: PromptSubset,
    temperature: f32,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<LatencyResult> {
    use rand::SeedableRng;

    anyhow::ensure!(
        weights.mtp.is_some(),
        "--spec-method mtp requires the loaded checkpoint to ship MTP weights \
         (Qwen3.5-4B includes them)"
    );

    // Phase C35 H13 A/B — optional chat-template framing. The base prompt
    // comes from the 8-prompt pool via `build_prompt`; when `chat_template`
    // is set we re-wrap it as a single `user` turn via the tokenizer's
    // chat template (falls back to plain ChatML in tokenizer.rs when no
    // Jinja template is loaded). Prompt budget (`prompt_tokens`) is still
    // targeted on the raw prose; the framing adds ~10-20 tokens of overhead,
    // which is fine for α measurement.
    let raw_prompt = build_prompt_with_subset(tokenizer, prompt_tokens, seed, prompt_subset);
    let prompt = if chat_template {
        let messages = [ChatMessage {
            role: "user".to_string(),
            content: raw_prompt,
            ..Default::default()
        }];
        tokenizer
            .apply_chat_template(&messages)
            .map_err(|e| anyhow::anyhow!("chat template application failed: {e}"))?
    } else {
        raw_prompt
    };
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device_kt = weights.device_kt();
    let dtype = kiln_config_dtype_to_kt(config.dtype);

    // Reserve enough blocks to cover prompt + 2*max_output_tokens (each MTP
    // step writes up to 2 base-cache slots: [last_token, draft_token]).
    let max_total_base = max_output_tokens
        .checked_mul(2)
        .and_then(|output| actual_prompt_tokens.checked_add(output))
        .context("MTP benchmark cache size overflowed")?;
    let num_blocks = max_total_base.div_ceil(PAGED_BLOCK_SIZE);

    // #1082 candle-drop: `PagedKvCache::new_kt(&device_kt, ...)` ->
    // `PagedKvCacheKt::new(..., device)` for both base + MTP caches (pools
    // on the runtime `Device`).
    let base_cache = PagedKvCacheKt::new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device_kt,
    )?;
    let mtp_cache = PagedKvCacheKt::new(
        1,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device_kt,
    )?;
    let backend = runtime_backend_for_bench(&device_kt, weights)?;
    // #1082 forward-flip: the linear state takes a kt `&Device` directly.
    let mut linear_state = LinearAttentionState::new_with_batch_for_inference_backend(
        config,
        1,
        &device_kt,
        Some(BackendIdentity::runtime_name(backend.as_ref())),
    )?;

    let mut base_block_table = BlockTable::new();
    let mut mtp_block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        base_block_table.push(i);
        mtp_block_table.push(i);
    }

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!(
        "  Measuring latency [MTP, paged, blocks={num_blocks}] \
         ({actual_prompt_tokens} prompt tokens)..."
    );

    // Prefill: paged forward returning (logits, last-position hidden state)
    // so we can seed h_prev for the first MTP draft step.
    let prefill_start = Instant::now();
    let (prefill_logits, prefill_h_prev_kt) = if streaming_prefill.enabled_for(actual_prompt_tokens)
    {
        model_forward_paged_streaming_last_token_with_last_hidden_with_policy(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &base_cache,
            &base_block_table,
            0,
            Some(&mut linear_state),
            None,
            streaming_prefill,
        )
        .context("MTP prefill (streaming paged with last-hidden) failed")?
    } else {
        model_forward_paged_last_token_with_last_hidden(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &base_cache,
            &base_block_table,
            0,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("MTP prefill (paged with last-hidden) failed")?
    };

    // #1082 forward-flip: the paged-with-last-hidden entry now returns kt
    // tensors. The MTP step (`speculative_mtp_decode_step`) and the candle
    // host sampler both still consume candle tensors, so bridge the
    // last-position hidden state to candle once here and thread the candle
    // `h_prev` through the decode loop.
    // #1082: speculative_mtp_decode_step + greedy_sample are kt-native now —
    // keep h_prev / prefill_last as kt (no candle bridge).
    let mut h_prev = prefill_h_prev_kt;

    // prefill_logits is already [1, 1, V] (kt). Squeeze the time dim.
    let prefill_last = prefill_logits.squeeze(1)?;
    let mut last_token = greedy_sample(&prefill_last)?;
    let prefill_time = prefill_start.elapsed();

    eprintln!(
        "    Prefill (MTP): {:.1}ms ({:.0} tok/s)",
        prefill_time.as_secs_f64() * 1000.0,
        actual_prompt_tokens as f64 / prefill_time.as_secs_f64()
    );

    // Decode loop. Each MTP step emits 1 or 2 tokens; per-token ITL =
    // step_time / accepted_count (matching the skip-layer arm).
    let mut inter_token_ms: Vec<f64> = Vec::new();
    let mut num_tokens = 1usize; // counting the first token from prefill
    let mut base_pos = actual_prompt_tokens;
    let mut mtp_pos = 0usize;
    let mut generated_tokens: Vec<TokenId> = Vec::new();
    let mut draft_accepted_count = 0usize;
    let mut total_draft_attempts = 0usize;
    let params = SamplingParams {
        temperature,
        top_p: 1.0,
        top_k: 0,
        max_tokens: max_output_tokens,
        repetition_penalty: 1.0,
        stop: vec![],
        seed: Some(seed),
        ..SamplingParams::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    while num_tokens < max_output_tokens {
        if eos_token_ids.contains(&last_token) {
            break;
        }

        generated_tokens.push(last_token);
        let step_start = Instant::now();
        let step = speculative_mtp_decode_step(
            &*backend,
            last_token,
            &h_prev,
            weights,
            config,
            &base_cache,
            &base_block_table,
            base_pos,
            &mut linear_state,
            &mtp_cache,
            &mtp_block_table,
            mtp_pos,
            &params,
            &eos_token_ids,
            &mut rng,
            // The bench measures base-model MTP throughput — no adapter.
            None,
        );
        let step = step.context("speculative_mtp_decode_step failed")?;
        let step_time = step_start.elapsed();

        if step.accepted_tokens.is_empty() {
            break;
        }

        total_draft_attempts += 1;
        if step.draft_accepted {
            draft_accepted_count += 1;
        }

        let per_token_ms = (step_time.as_secs_f64() * 1000.0) / step.accepted_tokens.len() as f64;
        for &_tok in &step.accepted_tokens {
            inter_token_ms.push(per_token_ms);
            num_tokens += 1;
            if num_tokens >= max_output_tokens {
                break;
            }
        }

        for &tok in &step.accepted_tokens[..step.accepted_tokens.len().saturating_sub(1)] {
            generated_tokens.push(tok);
        }

        last_token = *step.accepted_tokens.last().unwrap();
        base_pos += step.base_advance;
        mtp_pos += step.mtp_advance;
        h_prev = step.new_h_prev;

        if step.hit_eos {
            break;
        }
    }

    let mean_itl = if inter_token_ms.is_empty() {
        0.0
    } else {
        inter_token_ms.iter().sum::<f64>() / inter_token_ms.len() as f64
    };
    let mut sorted = inter_token_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let p99 = if sorted.is_empty() {
        0.0
    } else {
        let idx = ((sorted.len() as f64 * 0.99) as usize).min(sorted.len() - 1);
        sorted[idx]
    };
    let decode_tok_per_sec = if inter_token_ms.is_empty() {
        0.0
    } else {
        let total_decode_ms: f64 = inter_token_ms.iter().sum();
        inter_token_ms.len() as f64 / (total_decode_ms / 1000.0)
    };
    let alpha = if total_draft_attempts == 0 {
        0.0
    } else {
        draft_accepted_count as f64 / total_draft_attempts as f64
    };

    eprintln!(
        "    Decode (MTP): {num_tokens} tokens, mean ITL {:.1}ms ({:.1} tok/s), \
         α = {:.3} ({}/{})",
        mean_itl, decode_tok_per_sec, alpha, draft_accepted_count, total_draft_attempts
    );

    Ok(LatencyResult {
        prompt_tokens: actual_prompt_tokens,
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens_per_sec: actual_prompt_tokens as f64 / prefill_time.as_secs_f64(),
        time_to_first_token_ms: prefill_time.as_secs_f64() * 1000.0,
        mean_inter_token_ms: mean_itl,
        p50_inter_token_ms: p50,
        p99_inter_token_ms: p99,
        num_tokens_generated: num_tokens,
        decode_tokens_per_sec: decode_tok_per_sec,
        spec_method: "mtp".to_string(),
        acceptance_rate: Some(alpha),
        prompt_subset: Some(prompt_subset.as_tag().to_string()),
    })
}

/// Benchmark SFT training speed.
fn bench_training(
    model_config: &ModelConfig,
    weights: &GpuWeights,
    tokenizer: &KilnTokenizer,
    num_steps: usize,
    server_training_dispatch: ServerTrainingDispatchPolicy,
    vram_probe_selector: VramProbeSelector,
    runtime: &kiln_train::TrainingRuntimeContext,
) -> Result<TrainingResult> {
    use kiln_train::{ChatMessage, SftConfig, SftExample};

    {
        let mut stderr = std::io::stderr();
        let _ = writeln!(
            stderr,
            "  {} running {num_steps} SFT training steps",
            style("→").cyan()
        );
    }

    // Create synthetic training examples
    let examples: Vec<SftExample> = (0..num_steps)
        .map(|i| SftExample {
            messages: vec![
                ChatMessage::new(
                    "user",
                    format!("What is the capital of country number {i}? Please explain in detail."),
                ),
                ChatMessage::new(
                    "assistant",
                    format!(
                        "The capital of country number {i} is City{i}. It is located in the \
                         central region and has a population of approximately {} million people. \
                         The city was founded in the {}th century and is known for its historic \
                         architecture and vibrant cultural scene.",
                        i * 3 + 1,
                        (i % 15) + 5
                    ),
                ),
            ],
        })
        .collect();

    let config = SftConfig {
        training_profile: kiln_train::SftTrainingProfile::NativeOnlineLoraV1,
        invalid_row_policy: kiln_train::SftInvalidRowPolicy::Fail,
        train_mtp: Some(false), // bench measures the main SFT step only
        epochs: 1,
        learning_rate: None,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        allow_adapter_shape_conversion: false,
        allow_high_lora_scale: false,
        output_name: Some("bench-adapter".to_string()),
        auto_load: false,
        checkpoint_interval: None,
        resume_checkpoint: None,
        grad_checkpoint_segments: None,
        detect_anomaly: false,
        seed: None,
        optimizer: kiln_train::Optimizer::default(),
        adapter_smoke_test: false,
        adapter_smoke_prompts: None,
    };

    let adapter_dir = std::env::temp_dir().join("kiln-bench-adapters");
    std::fs::create_dir_all(&adapter_dir)?;
    let prepared = kiln_train::sft_ingestion::prepare_sft_examples(
        examples,
        tokenizer,
        config.invalid_row_policy,
        "kiln-bench",
        None,
    )
    .context("admit benchmark SFT examples")?;

    let progress_cb = Some(Box::new(|progress: kiln_train::trainer::TrainingProgress| {
        eprintln!(
            "    Step {}/{}: loss={:.6}",
            progress.step, progress.total_steps, progress.loss
        );
        kiln_train::trainer::TrainControl::Continue
    }) as kiln_train::trainer::ProgressCallback);

    let start = Instant::now();
    let native_route_enabled = server_training_dispatch.native_route_enabled();

    #[cfg(feature = "cuda")]
    let result = if native_route_enabled {
        kiln_train::cuda_train::cuda_native_sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
            &prepared.examples,
            &prepared.ingestion,
            &config,
            model_config,
            weights,
            tokenizer,
            &adapter_dir,
            &adapter_dir,
            &adapter_dir,
            "bench-adapter",
            progress_cb,
            None,
            runtime,
        )
    } else {
        kiln_train::trainer::sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
            &prepared.examples,
            &prepared.ingestion,
            &config,
            model_config,
            weights,
            tokenizer,
            &adapter_dir,
            &adapter_dir,
            &adapter_dir,
            "bench-adapter",
            progress_cb,
            None,
            None,
            runtime,
        )
    };
    #[cfg(not(feature = "cuda"))]
    let result = {
        if native_route_enabled {
            let native_route_env = server_training_dispatch
                .native_training_env
                .unwrap_or("backend_native_training_policy");
            eprintln!(
                "    Native training route enabled via {native_route_env}, but kiln-bench was \
                 built without --features cuda; falling back to kt-tape SFT"
            );
        }
        kiln_train::trainer::sft_train_to_with_checkpoint_root_and_ingestion_with_runtime(
            &prepared.examples,
            &prepared.ingestion,
            &config,
            model_config,
            weights,
            tokenizer,
            &adapter_dir,
            &adapter_dir,
            &adapter_dir,
            "bench-adapter",
            progress_cb,
            None,
            None,
            runtime,
        )
    };
    let elapsed = start.elapsed();

    let peak_vram = current_vram_used_bytes(vram_probe_selector) / (1024 * 1024);

    // Clean up temp adapter
    let _ = std::fs::remove_dir_all(adapter_dir.join("bench-adapter"));

    match result {
        Ok(_path) => Ok(TrainingResult {
            num_steps,
            total_time_secs: elapsed.as_secs_f64(),
            secs_per_step: elapsed.as_secs_f64() / num_steps as f64,
            peak_vram_mb: peak_vram,
        }),
        Err(e) => {
            eprintln!("  Training failed: {e}");
            Err(e)
        }
    }
}

/// Render one `key  value [unit]` line at the indent used by the summary.
fn metric(label: &str, value: impl std::fmt::Display, unit: Option<&str>) {
    let mut stderr = std::io::stderr();
    let label_w = 22;
    match unit {
        Some(u) => {
            let _ = writeln!(
                stderr,
                "  {:label_w$} {} {}",
                style(label).dim(),
                style(value).white().bold(),
                style(u).dim(),
                label_w = label_w
            );
        }
        None => {
            let _ = writeln!(
                stderr,
                "  {:label_w$} {}",
                style(label).dim(),
                style(value).white().bold(),
                label_w = label_w
            );
        }
    }
}

fn print_summary(results: &BenchmarkResults) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr);
    let _ = writeln!(
        stderr,
        "  {} {}",
        style("▌").cyan().bold(),
        style("Benchmark results").cyan().bold()
    );

    metric(
        "GPU",
        format!(
            "{} ({} MB VRAM, source: {})",
            results.gpu_info.name, results.gpu_info.total_vram_mb, results.gpu_info.vram_source
        ),
        None,
    );
    metric(
        "Model load",
        format!(
            "{:.2}s ({} MB VRAM)",
            results.model_load.load_time_secs, results.model_load.model_vram_mb
        ),
        None,
    );

    section_header("Inference throughput");
    let _ = writeln!(
        stderr,
        "  {}",
        style(format!(
            "{:<8} {:>10} {:>10} {:>12} {:>10}",
            "Runs", "Prompt", "Output", "tok/s", "VRAM MB"
        ))
        .dim()
    );
    for r in &results.inference {
        let _ = writeln!(
            stderr,
            "  {:<8} {:>10} {:>10} {:>12} {:>10}",
            r.batch_size,
            r.prompt_tokens,
            r.output_tokens,
            style(format!("{:.1}", r.tokens_per_sec)).white().bold(),
            r.peak_vram_mb
        );
    }

    section_header("Latency (single request)");
    metric("Prompt tokens", results.latency.prompt_tokens, None);
    metric(
        "Prefill",
        format!(
            "{:.1} ms ({:.0} tok/s)",
            results.latency.prefill_time_ms, results.latency.prefill_tokens_per_sec
        ),
        None,
    );
    metric(
        "Time to first token",
        format!("{:.1}", results.latency.time_to_first_token_ms),
        Some("ms"),
    );
    metric(
        "Mean inter-token",
        format!(
            "{:.1} ms ({:.1} tok/s)",
            results.latency.mean_inter_token_ms, results.latency.decode_tokens_per_sec
        ),
        None,
    );
    metric(
        "P50 inter-token",
        format!("{:.1}", results.latency.p50_inter_token_ms),
        Some("ms"),
    );
    metric(
        "P99 inter-token",
        format!("{:.1}", results.latency.p99_inter_token_ms),
        Some("ms"),
    );
    metric(
        "Tokens generated",
        results.latency.num_tokens_generated,
        None,
    );
    metric("Spec method", &results.latency.spec_method, None);
    if let Some(alpha) = results.latency.acceptance_rate {
        metric("Draft acceptance α", format!("{alpha:.3}"), None);
    }

    if let Some(t) = &results.training {
        section_header("SFT training");
        metric("Steps", t.num_steps, None);
        metric("Total time", format!("{:.2}", t.total_time_secs), Some("s"));
        metric(
            "Time per step",
            format!("{:.2}", t.secs_per_step),
            Some("s"),
        );
        metric("Peak VRAM", t.peak_vram_mb, Some("MB"));
    }

    let _ = writeln!(stderr);
}

fn bench_selected_latency(
    spec_method: SpecMethod,
    speculative: &SpeculativeConfig,
    args: &BenchArgs,
    gpu_weights: &GpuWeights,
    model_config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    streaming_prefill: StreamingPrefillExecutionPolicy,
) -> Result<LatencyResult> {
    match spec_method {
        SpecMethod::Mtp => {
            section_header("Latency benchmark (MTP — native speculative, paged)");
            bench_latency_paged_mtp(
                gpu_weights,
                model_config,
                tokenizer,
                args.prompt_tokens,
                args.max_output_tokens,
                args.seed,
                args.chat_template,
                args.prompt_subset,
                args.temperature,
                streaming_prefill,
            )
            .context("MTP latency benchmark failed")
        }
        SpecMethod::SkipLayer => {
            if args.paged {
                section_header("Latency benchmark (SKIP-LAYER — self-speculative, paged)");
                bench_latency_paged_skiplayer(
                    gpu_weights,
                    model_config,
                    tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                    args.seed,
                    args.temperature,
                    streaming_prefill,
                    speculative,
                )
                .context("paged skip-layer latency benchmark failed")
            } else {
                section_header("Latency benchmark (SKIP-LAYER — self-speculative, flat KV)");
                bench_latency_skiplayer(
                    gpu_weights,
                    model_config,
                    tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                    args.seed,
                    args.temperature,
                    streaming_prefill,
                    speculative,
                )
                .context("skip-layer latency benchmark failed")
            }
        }
        SpecMethod::Off => {
            if args.paged {
                section_header("Latency benchmark (PAGED — production path)");
                bench_latency_paged(
                    gpu_weights,
                    model_config,
                    tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                    args.seed,
                    args.temperature,
                    streaming_prefill,
                    args.log_tokens,
                    args.log_itl,
                )
                .context("paged latency benchmark failed")
            } else {
                section_header("Latency benchmark");
                bench_latency(
                    gpu_weights,
                    model_config,
                    tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                    streaming_prefill,
                )
                .context("latency benchmark failed")
            }
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // Tracing filter is flag-driven so default bench output stays clean.
    // RUST_LOG still wins via try_from_default_env.
    let directive = bench_filter(args.verbose, args.quiet);
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(directive));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    let startup_config = KilnConfig::load(args.config_path.as_deref())
        .context("load benchmark startup configuration")?;
    let model_config = ModelConfig::qwen3_5_4b();
    let spec_method = args
        .spec_method
        .unwrap_or_else(|| startup_config.speculative.effective_method());
    let speculative = SpeculativeConfig {
        num_speculative_tokens: args
            .spec_num_tokens
            .unwrap_or(startup_config.speculative.num_speculative_tokens),
        draft_layers: args
            .spec_draft_layers
            .unwrap_or(startup_config.speculative.draft_layers),
    };
    speculative
        .validate(&model_config)
        .context("invalid benchmark speculative configuration")?;
    require_speculative_benchmark_opt_in(spec_method, args.allow_experimental_speculative)?;

    let accelerator_runtime_policy = startup_config
        .accelerator
        .resolved_policy(startup_config.server.serving_profile);
    kiln_server::accelerator_runtime::install_pre_device_startup_policy(accelerator_runtime_policy)
        .context("install benchmark accelerator policy before device selection")?;
    let gradient_checkpoint_policy = kiln_train::GradientCheckpointPolicy::from_parts(
        startup_config.training.grad_checkpoint_segments,
        startup_config.training.no_grad_checkpoint,
    )
    .context("resolve benchmark gradient-checkpoint policy")?;
    let checkpoint_boundary_policy = startup_config
        .training
        .checkpoint_boundary_policy()
        .context("resolve benchmark checkpoint-boundary policy")?;
    tracing::info!(
        recompute_mode = %checkpoint_boundary_policy.recompute_mode(),
        recompute_threshold_tokens = checkpoint_boundary_policy.recompute_threshold_tokens(),
        anchor_stride = ?checkpoint_boundary_policy.anchor_stride(),
        cache_target_bytes = checkpoint_boundary_policy.cache_target_bytes(),
        immutable_after_startup = true,
        "SFT checkpoint-boundary policy resolved"
    );
    let model_path = Path::new(&args.model_path);

    // Compact banner — the rich box+GPU panel lives in `kiln serve`. Bench is
    // a tool, not a daemon; one cyan tagline is enough.
    {
        let mut stderr = std::io::stderr();
        let _ = writeln!(stderr);
        let _ = writeln!(
            stderr,
            "  {} {}",
            style("kiln-bench").cyan().bold(),
            style(format!("v{}", env!("CARGO_PKG_VERSION"))).dim()
        );
    }

    // Resolve the benchmark device before any memory probe so mixed-accelerator
    // hosts never report or budget against whichever GPU an auto probe finds first.
    let device_kt = kiln_server::device::select_device_kt()?;
    kiln_server::accelerator_runtime::install_startup_policy(device_kt, accelerator_runtime_policy)
        .context("install benchmark accelerator startup policy")?;
    let vram_probe_selector = vram_probe_selector_for_device(&device_kt);
    let backend = runtime_backend::for_device_kt(&device_kt);
    let backend_name = BackendIdentity::runtime_name(backend.as_ref());
    if backend_name == "cpu" {
        anyhow::bail!(
            "No accelerated backend available — benchmarks require CUDA, Metal, or Vulkan"
        );
    }
    let backend_capabilities = BackendCapabilityQueries::backend_capabilities(backend.as_ref());
    let streaming_prefill_runtime = startup_config
        .streaming_prefill
        .resolve(backend_capabilities.streaming_prefill);
    let streaming_prefill = streaming_prefill_runtime.execution_policy();

    // GPU info
    let vram = detect_vram_for(vram_probe_selector);
    let bench_runtime = kiln_train::TrainingRuntimeContext::new_for_device(
        device_kt,
        vram,
        gradient_checkpoint_policy,
    )
    .with_checkpoint_boundary_policy(checkpoint_boundary_policy)
    .with_streaming_prefill_policy(streaming_prefill);
    kiln_train::ensure_memory_governor_for_runtime(device_kt, &bench_runtime)
        .context("failed to initialize benchmark memory governor")?;
    let gpu_info = GpuInfo {
        name: gpu_name(),
        total_vram_mb: vram.total_bytes / (1024 * 1024),
        vram_source: vram.source.to_string(),
    };
    metric(
        "GPU",
        format!("{} ({} MB)", gpu_info.name, gpu_info.total_vram_mb),
        None,
    );

    // Load model
    metric("Model path", model_path.display(), None);

    let vram_before = current_vram_used_bytes(vram_probe_selector);
    let load_start = Instant::now();

    let model_weights = kiln_model::load_model_with_options(
        model_path,
        &model_config,
        kiln_model::LoadModelOptions { load_mtp: false },
    )
    .context("failed to load model weights")?;

    let native_mtp_allowed = backend_capabilities
        .decode
        .mtp_speculative_generation
        .is_native();
    let speculative_policy = backend_capabilities.decode.speculative_policy;

    let gpu_weights = GpuWeights::from_model_weights_kt(&model_weights, &model_config, &device_kt)
        .context("failed to transfer weights to GPU")?;
    let post_load_memory = kiln_memory::MemoryGovernor::global().refresh();
    if post_load_memory.total_bytes == 0 {
        anyhow::bail!("selected-device memory probe failed after benchmark model load");
    }
    drop(model_weights); // Free CPU memory

    let load_time = load_start.elapsed();
    let vram_after = current_vram_used_bytes(vram_probe_selector);
    let model_vram = (vram_after.saturating_sub(vram_before)) / (1024 * 1024);

    metric(
        "Model loaded",
        format!(
            "{:.2}s (backend: {}, {} MB VRAM)",
            load_time.as_secs_f64(),
            backend_name,
            model_vram
        ),
        None,
    );

    let model_load = ModelLoadResult {
        load_time_secs: load_time.as_secs_f64(),
        model_vram_mb: model_vram,
    };

    // Load tokenizer from model directory
    let tokenizer = {
        let tok_file = model_path.join("tokenizer.json");
        if tok_file.exists() {
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            tracing::info!(
                target: "kiln_bench",
                "tokenizer.json missing locally — fetching from HuggingFace Hub"
            );
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    let requested_spec_method = spec_method;
    let spec_method = resolve_bench_spec_method(
        requested_spec_method,
        args.prompt_tokens,
        args.max_output_tokens,
        args.temperature,
        gpu_weights.mtp.is_some(),
        native_mtp_allowed,
        speculative_policy,
        args.force_mtp,
    );
    if spec_method != requested_spec_method {
        let mut stderr = std::io::stderr();
        let _ = writeln!(
            stderr,
            "  {} resolved requested spec method {:?} → {:?} for shape (prompt={}, max_output={}, temperature={})",
            style("→").yellow().bold(),
            requested_spec_method,
            spec_method,
            args.prompt_tokens,
            args.max_output_tokens,
            args.temperature
        );
    }

    // Latency benchmark (uses model_forward directly — must run before runner takes ownership).
    // This dispatch is benchmark-only; speculative serving fails closed and
    // does not use these shape heuristics. Benchmark dispatch order:
    //   * --spec-method mtp        → qualified short greedy shapes use MTP;
    //                                   qualifying long greedy shapes use
    //                                   skip-layer, otherwise speculative is off
    //   * --spec-method skip_layer → bench_latency_paged_skiplayer when
    //                                   --paged, else bench_latency_skiplayer
    //                                   (flat KV + skip-layer)
    //   * default / off               → bench_latency_paged (paged Off) when
    //                                   --paged, else bench_latency (flat Off).
    for warmup_idx in 0..args.latency_warmup_runs {
        section_header(&format!(
            "Latency warmup run {}/{} (not measured)",
            warmup_idx + 1,
            args.latency_warmup_runs
        ));
        let _ = bench_selected_latency(
            spec_method,
            &speculative,
            &args,
            &gpu_weights,
            &model_config,
            &tokenizer,
            streaming_prefill,
        )
        .with_context(|| format!("latency warmup run {} failed", warmup_idx + 1))?;
    }

    let latency = bench_selected_latency(
        spec_method,
        &speculative,
        &args,
        &gpu_weights,
        &model_config,
        &tokenizer,
        streaming_prefill,
    )
    .context("latency benchmark failed")?;

    if args.latency_only {
        let results = BenchmarkResults {
            backend: backend_name.to_string(),
            gpu_info,
            model_load,
            inference: Vec::new(),
            latency,
            training: None,
        };

        print_summary(&results);

        let json = serde_json::to_string_pretty(&results)?;
        println!("{json}");

        return Ok(());
    }

    // Training benchmark (borrows gpu_weights — must run before runner takes ownership)
    let training = if args.skip_training {
        section_header("Training benchmark (skipped)");
        None
    } else {
        section_header("Training benchmark");
        match bench_training(
            &model_config,
            &gpu_weights,
            &tokenizer,
            args.training_steps,
            backend_capabilities.training.server_dispatch,
            vram_probe_selector,
            &bench_runtime,
        ) {
            Ok(result) => {
                let mut stderr = std::io::stderr();
                let _ = writeln!(
                    stderr,
                    "  {} {:.2}s/step, peak VRAM {} MB",
                    style("✓").green().bold(),
                    result.secs_per_step,
                    result.peak_vram_mb
                );
                Some(result)
            }
            Err(e) => {
                let mut stderr = std::io::stderr();
                let _ = writeln!(
                    stderr,
                    "  {} training benchmark failed: {e}",
                    style("✗").red().bold()
                );
                None
            }
        }
    };

    // Load a second tokenizer for the runner (ModelRunner takes ownership)
    let runner_tokenizer = {
        let tok_file = model_path.join("tokenizer.json");
        if tok_file.exists() {
            KilnTokenizer::from_file(tok_file.to_str().unwrap())?
        } else {
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    // Create runner for throughput benchmarks (takes ownership of weights)
    let runner = ModelRunner::new_with_runtime_options(
        gpu_weights,
        runner_tokenizer,
        model_config.clone(),
        ModelRunnerRuntimeOptions {
            cuda_graph: kiln_model::CudaGraphExecutionPolicy::disabled(),
            rocm_graph: kiln_model::RocmGraphExecutionPolicy::lazy_capture_replay(),
            metal_graphs: true,
            max_decode_batch: None,
            streaming_prefill: Some(streaming_prefill),
        },
    );

    // Inference throughput at different run counts
    section_header("Inference throughput");
    let run_counts = [1, 4, 8, 16];
    let mut inference_results = Vec::new();

    for &n in &run_counts {
        let mut stderr = std::io::stderr();
        let _ = writeln!(
            stderr,
            "  {} {}",
            style("→").cyan(),
            style(format!(
                "{n} sequential run{}",
                if n == 1 { "" } else { "s" }
            ))
            .white()
        );
        match bench_inference(
            &runner,
            &tokenizer,
            n,
            args.prompt_tokens,
            args.max_output_tokens,
            args.seed,
            args.temperature,
            vram_probe_selector,
        ) {
            Ok(result) => {
                let _ = writeln!(
                    stderr,
                    "    {} {} tok/s aggregate",
                    style("✓").green().bold(),
                    style(format!("{:.1}", result.tokens_per_sec))
                        .white()
                        .bold()
                );
                inference_results.push(result);
            }
            Err(e) => {
                let _ = writeln!(stderr, "    {} {e}", style("✗ FAILED:").red().bold());
            }
        }
    }

    let results = BenchmarkResults {
        backend: backend_name.to_string(),
        gpu_info,
        model_load,
        inference: inference_results,
        latency,
        training,
    };

    // Print human-readable summary to stderr
    print_summary(&results);

    // Print JSON to stdout for machine parsing
    let json = serde_json::to_string_pretty(&results)?;
    println!("{json}");

    Ok(())
}
