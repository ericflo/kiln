//! Kiln benchmark suite — measures inference throughput, latency, VRAM, and training speed.
//!
//! Run with: `cargo run --release --features cuda --bin kiln-bench -- --model-path /path/to/weights`
//!
//! Requires a GPU with the Qwen3.5-4B model weights downloaded.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::tokenizer::{ChatMessage, KilnTokenizer};
use kiln_core::vram::detect_vram;
use kiln_model::ModelRunner;
use kiln_model::backend as runtime_backend;
use kiln_model::forward::{
    GpuWeights, LinearAttentionState, model_forward, model_forward_paged,
    model_forward_paged_last_token, model_forward_paged_last_token_with_last_hidden,
    model_forward_paged_streaming, model_forward_paged_streaming_last_token_with_last_hidden,
    streaming_prefill_enabled_for,
};
use kiln_model::kv_cache::KvCache;
use kiln_model::paged_kv_cache::PagedKvCache;
use kiln_model::sampling::greedy_sample;
use kiln_model::speculative::{
    SpeculativeConfig, speculative_decode_step, speculative_decode_step_paged_greedy,
    speculative_mtp_decode_step,
};
use kiln_server::config::SpecMethod;

/// Block size used for the paged-path benchmark. Matches the kiln-core default.
const PAGED_BLOCK_SIZE: usize = 16;

/// Results from the full benchmark suite.
#[derive(Debug, Serialize)]
struct BenchmarkResults {
    /// Which `BackendRuntime` ran the forward pass — one of
    /// `cuda` / `metal` / `cpu`. Lets downstream comparison scripts split
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
/// Only affects the MTP bench arm (`KILN_SPEC_METHOD=mtp`); ignored by
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
    /// RNG seed threaded through `SamplingParams` and `StdRng` sites so bench
    /// runs are fully reproducible. Phase B3 multi-prompt A/B relies on varying
    /// this across {0..=7} to get independent prompt/sampling trajectories.
    seed: u64,
    /// When true, wrap the MTP bench prompt in the tokenizer's chat template
    /// (Qwen ChatML framing) before encoding. Phase C35 H13 residual A/B —
    /// tests whether raw-prose prompts cause the α degradation vs the paper.
    /// Only affects the MTP bench arm (`KILN_SPEC_METHOD=mtp`); ignored for
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
}

fn parse_args() -> Result<BenchArgs> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut max_output_tokens = 128;
    let mut prompt_tokens = 512;
    let mut training_steps = 10;
    let mut skip_training = false;
    let mut paged = false;
    let mut latency_only = false;
    let mut seed: u64 = 42;
    let mut chat_template = false;
    let mut prompt_subset = PromptSubset::All;
    let mut temperature: f32 = 0.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                i += 1;
                model_path = args.get(i).cloned().unwrap_or_default();
            }
            "--max-output-tokens" => {
                i += 1;
                max_output_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(128);
            }
            "--prompt-tokens" => {
                i += 1;
                prompt_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(512);
            }
            "--training-steps" => {
                i += 1;
                training_steps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
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
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(42);
            }
            "--chat-template" => {
                chat_template = true;
            }
            "--prompt-subset" => {
                i += 1;
                let s = args.get(i).cloned().unwrap_or_default();
                prompt_subset = PromptSubset::parse(&s).ok_or_else(|| {
                    anyhow::anyhow!(
                        "invalid --prompt-subset value '{s}' (expected all|gsm8k|humaneval|c4)"
                    )
                })?;
            }
            "--temperature" => {
                i += 1;
                temperature = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--help" | "-h" => {
                eprintln!("Usage: kiln-bench --model-path <path> [options]");
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
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if model_path.is_empty() {
        anyhow::bail!("--model-path is required. Run with --help for usage.");
    }

    Ok(BenchArgs {
        model_path,
        max_output_tokens,
        prompt_tokens,
        training_steps,
        skip_training,
        paged,
        latency_only,
        seed,
        chat_template,
        prompt_subset,
        temperature,
    })
}

/// Get GPU name from nvidia-smi.
fn gpu_name() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().lines().next().unwrap_or("unknown").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get current VRAM usage in bytes via nvidia-smi.
fn current_vram_used_bytes() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().lines().next()?.trim().parse::<u64>().ok())
        .map(|mib| mib * 1024 * 1024)
        .unwrap_or(0)
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
    };

    // Warmup
    eprintln!("  Warmup...");
    let warmup_params = SamplingParams {
        max_tokens: 4,
        ..params.clone()
    };
    let _ = runner.generate(&prompt, &warmup_params);

    // Timed runs
    eprintln!("  Running {num_runs} sequential generations...");
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

        eprintln!(
            "    Run {}/{}: {} tokens in {:.1}ms ({:.1} tok/s)",
            i + 1,
            num_runs,
            gen_tokens,
            run_time.as_secs_f64() * 1000.0,
            gen_tokens as f64 / run_time.as_secs_f64()
        );
    }

    let total_time = overall_start.elapsed();
    let peak_vram = current_vram_used_bytes() / (1024 * 1024);

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
/// Uses `model_forward` directly for precise per-step timing.
fn bench_latency(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
) -> Result<LatencyResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens, 0);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device = weights.embed_tokens.device();
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };

    let max_total = actual_prompt_tokens + max_output_tokens;
    let mut kv_cache = KvCache::new(
        config.num_full_attention_layers,
        config.num_kv_heads,
        config.head_dim,
        max_total,
        dtype,
        device,
    )?;
    let mut linear_state = LinearAttentionState::new(config, device)?;
    let backend = runtime_backend::for_device(device);

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!("  Measuring latency ({actual_prompt_tokens} prompt tokens)...");

    // Prefill: forward pass on all prompt tokens
    let prefill_start = Instant::now();
    let logits = model_forward(
        &*backend,
        &prompt_token_ids,
        weights,
        config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
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
        let logits = model_forward(
            &*backend,
            &[next_token],
            weights,
            config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            None,
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
/// Mirrors `bench_latency` but uses `PagedKvCache` + `BlockTable` +
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
) -> Result<LatencyResult> {
    let prompt = build_prompt(tokenizer, prompt_tokens, 0);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device = weights.embed_tokens.device();
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };

    let max_total = actual_prompt_tokens + max_output_tokens;
    let num_blocks = (max_total + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;

    let mut paged_cache = PagedKvCache::new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device,
    )?;
    let mut linear_state = LinearAttentionState::new(config, device)?;
    let backend = runtime_backend::for_device(device);

    // Build a block table that maps logical block i -> physical block i (sequential).
    let mut block_table = BlockTable::new();
    for i in 0..num_blocks as u32 {
        block_table.push(i);
    }

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!(
        "  Measuring latency [PAGED, block_size={PAGED_BLOCK_SIZE}, blocks={num_blocks}] \
         ({actual_prompt_tokens} prompt tokens)..."
    );

    // Prefill: forward pass on all prompt tokens via paged path. Long Metal
    // prompts use tiled streaming prefill by default; env overrides can force
    // either path.
    let prefill_start = Instant::now();
    let logits = if streaming_prefill_enabled_for(device, actual_prompt_tokens) {
        model_forward_paged_streaming(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &mut paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
        )
        .context("paged prefill forward pass (streaming) failed")?
    } else {
        model_forward_paged_last_token(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &mut paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("paged prefill forward pass failed")?
    };

    // Sample first token
    let mut next_token = greedy_sample(&logits)?;
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
    let log_tokens = std::env::var("KILN_BENCH_LOG_TOKENS").is_ok();
    let mut decoded_tokens: Vec<u32> = Vec::new();
    if log_tokens {
        decoded_tokens.push(next_token);
    }

    for _step in 0..max_output_tokens {
        if eos_token_ids.contains(&next_token) {
            break;
        }

        let step_start = Instant::now();
        let logits = model_forward_paged(
            &*backend,
            &[next_token],
            weights,
            config,
            &mut paged_cache,
            &block_table,
            current_pos,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("paged decode forward pass failed")?;
        current_pos += 1;
        next_token = greedy_sample(&logits)?;
        let step_time = step_start.elapsed();

        inter_token_ms.push(step_time.as_secs_f64() * 1000.0);
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

/// Read `KILN_SPEC_METHOD` from the environment and resolve it to a
/// `SpecMethod`. Defaults to `Off` when unset; warns and falls back to `Off`
/// when set to an unrecognised string. `KILN_SPEC_ENABLED=0` (or `false`)
/// forces `Off` even when a method is set, mirroring `effective_method()`
/// in the runtime config so bench / serve agree on dispatch.
fn read_spec_method_from_env() -> SpecMethod {
    let enabled = std::env::var("KILN_SPEC_ENABLED")
        .ok()
        .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"))
        .unwrap_or(true);
    if !enabled {
        return SpecMethod::Off;
    }
    match std::env::var("KILN_SPEC_METHOD") {
        Ok(v) => match SpecMethod::parse_env(&v) {
            Some(m) => m,
            None => {
                eprintln!(
                    "  WARN: ignoring unknown KILN_SPEC_METHOD='{}' (expected off|skip_layer|mtp)",
                    v
                );
                SpecMethod::Off
            }
        },
        Err(_) => SpecMethod::Off,
    }
}

/// Benchmark latency along the SKIP-LAYER speculative path.
///
/// Uses the same flat `KvCache` + `model_forward` path as the existing
/// `generate_from_tokens_speculative` in `kiln-model::generate` (skip-layer was
/// never paged). Drives `speculative_decode_step` directly per iteration so
/// each step's wall time is divided across the tokens emitted that step,
/// giving a per-emitted-token ITL distribution comparable to the `Off` arm.
///
/// Reads `KILN_SPEC_NUM_TOKENS` (default 256) and `KILN_SPEC_DRAFT_LAYERS`
/// (default 8). `temperature` defaults to 0 (greedy) for deterministic
/// per-seed reproducibility; Phase C40b threads `--temperature` through to
/// support sampled-decode α probes.
fn bench_latency_skiplayer(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    temperature: f32,
) -> Result<LatencyResult> {
    use rand::SeedableRng;

    let num_speculative_tokens = std::env::var("KILN_SPEC_NUM_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);
    let draft_layers = std::env::var("KILN_SPEC_DRAFT_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);

    let prompt = build_prompt(tokenizer, prompt_tokens, seed);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device = weights.embed_tokens.device();
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };

    // Skip-layer verifies `[last_token, draft_0, ..., draft_k-1]` in one
    // forward pass. Near the end of generation those speculative KV writes can
    // extend past the committed token budget before stale slots are overwritten
    // or ignored, so reserve headroom for the verify window.
    let max_spec_window = num_speculative_tokens.min(max_output_tokens.max(1));
    let max_total = actual_prompt_tokens + max_output_tokens + max_spec_window + 1;
    let mut kv_cache = KvCache::new(
        config.num_full_attention_layers,
        config.num_kv_heads,
        config.head_dim,
        max_total,
        dtype,
        device,
    )?;
    let mut linear_state = LinearAttentionState::new(config, device)?;
    let backend = runtime_backend::for_device(device);

    let eos_token_ids = tokenizer.eos_token_ids();

    eprintln!(
        "  Measuring latency [SKIP-LAYER, k={num_speculative_tokens}, draft_layers={draft_layers}] \
         ({actual_prompt_tokens} prompt tokens)..."
    );

    // Prefill: full forward pass, no speculative draft yet.
    let prefill_start = Instant::now();
    let logits = model_forward(
        &*backend,
        &prompt_token_ids,
        weights,
        config,
        Some(&mut kv_cache),
        Some(&mut linear_state),
        None,
    )
    .context("skip-layer prefill forward pass failed")?;
    kv_cache.advance(actual_prompt_tokens);

    let mut draft_linear_state = linear_state
        .snapshot_for_decode_rollback()
        .context("clone draft state from skip-layer prefill")?;

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
/// This is benchmark-only scaffolding for the production-style paged
/// self-speculative implementation. It keeps the paged prefill/cache/block-table
/// setup in this harness and delegates each decode iteration to
/// `speculative_decode_step_paged_greedy`, which is expected to verify against
/// `PagedKvCache` at the caller-provided `base_pos`.
///
/// Enable with `KILN_SPEC_METHOD=skip_layer --paged`. Without `--paged`, the
/// legacy flat-KV skip-layer benchmark remains available for comparisons.
fn bench_latency_paged_skiplayer(
    weights: &GpuWeights,
    config: &ModelConfig,
    tokenizer: &KilnTokenizer,
    prompt_tokens: usize,
    max_output_tokens: usize,
    seed: u64,
    temperature: f32,
) -> Result<LatencyResult> {
    let num_speculative_tokens = std::env::var("KILN_SPEC_NUM_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);
    let draft_layers = std::env::var("KILN_SPEC_DRAFT_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);

    let prompt = build_prompt(tokenizer, prompt_tokens, seed);
    let prompt_token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let actual_prompt_tokens = prompt_token_ids.len();

    let device = weights.embed_tokens.device();
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };

    // Verification writes `[last_token, draft_0, ..., draft_k-1]` starting at
    // `base_pos`. Reserve headroom so late-generation verify windows do not
    // run off the allocated paged cache before stale speculative slots are
    // overwritten by the next committed step.
    let max_spec_window = num_speculative_tokens.min(max_output_tokens.max(1));
    let max_total = actual_prompt_tokens + max_output_tokens + max_spec_window + 1;
    let num_blocks = (max_total + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;

    let mut paged_cache = PagedKvCache::new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device,
    )?;
    let mut linear_state = LinearAttentionState::new(config, device)?;
    let backend = runtime_backend::for_device(device);

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
    let logits = if streaming_prefill_enabled_for(device, actual_prompt_tokens) {
        model_forward_paged_streaming(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &mut paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
        )
        .context("paged skip-layer prefill forward pass (streaming) failed")?
    } else {
        model_forward_paged_last_token(
            &*backend,
            &prompt_token_ids,
            weights,
            config,
            &mut paged_cache,
            &block_table,
            0,
            Some(&mut linear_state),
            None,
            None,
        )
        .context("paged skip-layer prefill forward pass failed")?
    };

    let mut draft_linear_state = linear_state
        .snapshot_for_decode_rollback()
        .context("clone draft state from paged skip-layer prefill")?;

    let mut last_token = greedy_sample(&logits)?;
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
            &mut paged_cache,
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

/// Phase C35 H13 A/B — read `KILN_MTP_ARGMAX_FP32=1` once per process, cached
/// via `OnceLock`. Matches the identically-named helper in kiln-model so both
/// the speculative decode path and the bench prefill seed agree on whether to
/// promote logits to FP32 before argmax.
fn mtp_argmax_fp32_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("KILN_MTP_ARGMAX_FP32")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Benchmark latency along the NATIVE-MTP speculative path.
///
/// Uses two `PagedKvCache` instances (base + 1-layer MTP), threads `h_prev`
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
) -> Result<LatencyResult> {
    use rand::SeedableRng;

    anyhow::ensure!(
        weights.mtp.is_some(),
        "KILN_SPEC_METHOD=mtp requires the loaded checkpoint to ship MTP weights \
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

    let device = weights.embed_tokens.device();
    let dtype = match config.dtype {
        kiln_core::config::DType::BF16 => candle_core::DType::BF16,
        kiln_core::config::DType::FP16 => candle_core::DType::F16,
        kiln_core::config::DType::FP32 => candle_core::DType::F32,
    };

    // Reserve enough blocks to cover prompt + 2*max_output_tokens (each MTP
    // step writes up to 2 base-cache slots: [last_token, draft_token]).
    let max_total_base = actual_prompt_tokens + 2 * max_output_tokens;
    let num_blocks = (max_total_base + PAGED_BLOCK_SIZE - 1) / PAGED_BLOCK_SIZE;

    let mut base_cache = PagedKvCache::new(
        config.num_full_attention_layers,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device,
    )?;
    let mut mtp_cache = PagedKvCache::new(
        1,
        num_blocks,
        PAGED_BLOCK_SIZE,
        config.num_kv_heads,
        config.head_dim,
        dtype,
        device,
    )?;
    let mut linear_state = LinearAttentionState::new(config, device)?;
    let backend = runtime_backend::for_device(device);

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
    let (prefill_logits, mut h_prev) =
        if streaming_prefill_enabled_for(device, actual_prompt_tokens) {
            model_forward_paged_streaming_last_token_with_last_hidden(
                &*backend,
                &prompt_token_ids,
                weights,
                config,
                &mut base_cache,
                &base_block_table,
                0,
                Some(&mut linear_state),
                None,
            )
            .context("MTP prefill (streaming paged with last-hidden) failed")?
        } else {
            model_forward_paged_last_token_with_last_hidden(
                &*backend,
                &prompt_token_ids,
                weights,
                config,
                &mut base_cache,
                &base_block_table,
                0,
                Some(&mut linear_state),
                None,
                None,
            )
            .context("MTP prefill (paged with last-hidden) failed")?
        };

    // prefill_logits is already [1, 1, V].
    let prefill_last = prefill_logits.squeeze(1)?;
    // Phase C35 H13 A/B — optionally cast logits to FP32 before argmax so the
    // bench prefill matches vLLM's sampler contract (rejection_sampler.py
    // casts `raw_target_logits` to float32 before greedy). BF16 argmax can
    // flip top-1 under ties when two candidates share the same BF16 bucket.
    let mut last_token = if mtp_argmax_fp32_enabled() {
        let prefill_last_fp32 = prefill_last.to_dtype(candle_core::DType::F32)?;
        greedy_sample(&prefill_last_fp32)?
    } else {
        greedy_sample(&prefill_last)?
    };
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
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Phase C1 — reset the attribution sink (and its step counter) at the
    // start of each bench run. The capture hook inside
    // `speculative_mtp_decode_step` is a no-op unless `KILN_C1_ATTR_PATH`
    // is set, so this clear is cheap either way.
    kiln_model::c1_attr::clear();

    while num_tokens < max_output_tokens {
        if eos_token_ids.contains(&last_token) {
            break;
        }

        let step_start = Instant::now();
        let step = speculative_mtp_decode_step(
            &*backend,
            last_token,
            &h_prev,
            weights,
            config,
            &mut base_cache,
            &base_block_table,
            base_pos,
            &mut linear_state,
            &mut mtp_cache,
            &mtp_block_table,
            mtp_pos,
            &params,
            &eos_token_ids,
            &mut rng,
        )
        .context("speculative_mtp_decode_step failed")?;
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

    // Phase C1 — drain the attribution sink to CSV when
    // `KILN_C1_ATTR_PATH` is set. The path may include `{seed}` which is
    // substituted with the current run's seed so a single env var covers
    // multi-seed runs cleanly.
    if let Ok(path_template) = std::env::var("KILN_C1_ATTR_PATH") {
        if !path_template.is_empty() {
            let path = path_template.replace("{seed}", &seed.to_string());
            match kiln_model::c1_attr::drain_to_csv(&path) {
                Ok(n) => eprintln!("    C1 attribution: wrote {n} rows → {path}"),
                Err(e) => eprintln!("    C1 attribution: WRITE FAILED ({path}): {e:#}"),
            }
        }
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
) -> Result<TrainingResult> {
    use kiln_train::{ChatMessage, SftConfig, SftExample};

    eprintln!("  Running {num_steps} SFT training steps...");

    // Create synthetic training examples
    let examples: Vec<SftExample> = (0..num_steps)
        .map(|i| SftExample {
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "What is the capital of country number {i}? Please explain in detail."
                    ),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: format!(
                        "The capital of country number {i} is City{i}. It is located in the \
                         central region and has a population of approximately {} million people. \
                         The city was founded in the {}th century and is known for its historic \
                         architecture and vibrant cultural scene.",
                        i * 3 + 1,
                        (i % 15) + 5
                    ),
                },
            ],
        })
        .collect();

    let config = SftConfig {
        epochs: 1,
        learning_rate: 1e-4,
        lora_rank: 8,
        lora_alpha: 16.0,
        base_adapter: None,
        output_name: Some("bench-adapter".to_string()),
        auto_load: false,
        checkpoint_interval: None,
    };

    let adapter_dir = std::env::temp_dir().join("kiln-bench-adapters");
    std::fs::create_dir_all(&adapter_dir)?;

    let progress_cb = Some(Box::new(|progress: kiln_train::trainer::TrainingProgress| {
        eprintln!(
            "    Step {}/{}: loss={:.6}",
            progress.step, progress.total_steps, progress.loss
        );
    }) as kiln_train::trainer::ProgressCallback);

    let start = Instant::now();
    let result = kiln_train::trainer::sft_train(
        &examples,
        &config,
        model_config,
        weights,
        tokenizer,
        &adapter_dir,
        "bench-adapter",
        progress_cb,
    );
    let elapsed = start.elapsed();

    let peak_vram = current_vram_used_bytes() / (1024 * 1024);

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

fn print_summary(results: &BenchmarkResults) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("  KILN BENCHMARK RESULTS");
    eprintln!("{}\n", "=".repeat(60));

    eprintln!(
        "GPU: {} ({} MB VRAM, source: {})",
        results.gpu_info.name, results.gpu_info.total_vram_mb, results.gpu_info.vram_source
    );
    eprintln!(
        "Model load time: {:.2}s (VRAM: {} MB)\n",
        results.model_load.load_time_secs, results.model_load.model_vram_mb
    );

    eprintln!("--- Inference Throughput ---");
    eprintln!(
        "{:<8} {:>10} {:>10} {:>12} {:>10}",
        "Runs", "Prompt", "Output", "tok/s", "VRAM MB"
    );
    for r in &results.inference {
        eprintln!(
            "{:<8} {:>10} {:>10} {:>12.1} {:>10}",
            r.batch_size, r.prompt_tokens, r.output_tokens, r.tokens_per_sec, r.peak_vram_mb
        );
    }

    eprintln!("\n--- Latency (single request) ---");
    eprintln!("Prompt tokens:         {}", results.latency.prompt_tokens);
    eprintln!(
        "Prefill time:          {:.1} ms ({:.0} tok/s)",
        results.latency.prefill_time_ms, results.latency.prefill_tokens_per_sec
    );
    eprintln!(
        "Time to first token:   {:.1} ms",
        results.latency.time_to_first_token_ms
    );
    eprintln!(
        "Mean inter-token:      {:.1} ms ({:.1} tok/s)",
        results.latency.mean_inter_token_ms, results.latency.decode_tokens_per_sec
    );
    eprintln!(
        "P50 inter-token:       {:.1} ms",
        results.latency.p50_inter_token_ms
    );
    eprintln!(
        "P99 inter-token:       {:.1} ms",
        results.latency.p99_inter_token_ms
    );
    eprintln!(
        "Tokens generated:      {}",
        results.latency.num_tokens_generated
    );
    eprintln!("Spec method:           {}", results.latency.spec_method);
    if let Some(alpha) = results.latency.acceptance_rate {
        eprintln!("Draft acceptance α:    {:.3}", alpha);
    }

    if let Some(t) = &results.training {
        eprintln!("\n--- SFT Training ---");
        eprintln!("Steps:          {}", t.num_steps);
        eprintln!("Total time:     {:.2} s", t.total_time_secs);
        eprintln!("Time per step:  {:.2} s", t.secs_per_step);
        eprintln!("Peak VRAM:      {} MB", t.peak_vram_mb);
    }

    eprintln!();
}

fn main() -> Result<()> {
    // Initialize logging for training progress
    tracing_subscriber::fmt()
        .with_env_filter("kiln=info")
        .with_writer(std::io::stderr)
        .init();

    let args = parse_args()?;
    let model_path = Path::new(&args.model_path);

    eprintln!("=== Kiln Benchmark Suite ===\n");

    // GPU info
    let vram = detect_vram();
    let gpu_info = GpuInfo {
        name: gpu_name(),
        total_vram_mb: vram.total_bytes / (1024 * 1024),
        vram_source: vram.source.to_string(),
    };
    eprintln!("GPU: {} ({} MB)", gpu_info.name, gpu_info.total_vram_mb);

    // Load model
    let model_config = ModelConfig::qwen3_5_4b();
    eprintln!("Loading model from {}...", model_path.display());

    let vram_before = current_vram_used_bytes();
    let load_start = Instant::now();

    let spec_method = read_spec_method_from_env();
    let model_weights = kiln_model::load_model_with_options(
        model_path,
        &model_config,
        kiln_model::LoadModelOptions {
            load_mtp: matches!(spec_method, SpecMethod::Mtp),
        },
    )
    .context("failed to load model weights")?;

    let device = kiln_server::device::select_device()?;
    if matches!(device, candle_core::Device::Cpu) {
        anyhow::bail!("No GPU available — benchmarks require CUDA or Metal");
    }
    let backend_name = runtime_backend::for_device(&device).name();

    let gpu_weights = GpuWeights::from_model_weights(&model_weights, &model_config, &device)
        .context("failed to transfer weights to GPU")?;
    drop(model_weights); // Free CPU memory

    let load_time = load_start.elapsed();
    let vram_after = current_vram_used_bytes();
    let model_vram = (vram_after.saturating_sub(vram_before)) / (1024 * 1024);

    eprintln!(
        "Model loaded in {:.2}s (backend: {}, VRAM: {} MB)\n",
        load_time.as_secs_f64(),
        backend_name,
        model_vram
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
            eprintln!("Loading tokenizer from HuggingFace Hub...");
            KilnTokenizer::from_pretrained("Qwen/Qwen3.5-4B")?
        }
    };

    // Latency benchmark (uses model_forward directly — must run before runner takes ownership).
    // Dispatch order:
    //   * KILN_SPEC_METHOD=mtp        → bench_latency_paged_mtp (paged + MTP)
    //   * KILN_SPEC_METHOD=skip_layer → bench_latency_paged_skiplayer when
    //                                   --paged, else bench_latency_skiplayer
    //                                   (flat KV + skip-layer)
    //   * default / off               → bench_latency_paged (paged Off) when
    //                                   --paged, else bench_latency (flat Off).
    let latency = match spec_method {
        SpecMethod::Mtp => {
            eprintln!("--- Latency Benchmark (MTP — native speculative, paged) ---");
            bench_latency_paged_mtp(
                &gpu_weights,
                &model_config,
                &tokenizer,
                args.prompt_tokens,
                args.max_output_tokens,
                args.seed,
                args.chat_template,
                args.prompt_subset,
                args.temperature,
            )
            .context("MTP latency benchmark failed")?
        }
        SpecMethod::SkipLayer => {
            if args.paged {
                eprintln!("--- Latency Benchmark (SKIP-LAYER — self-speculative, paged) ---");
                bench_latency_paged_skiplayer(
                    &gpu_weights,
                    &model_config,
                    &tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                    args.seed,
                    args.temperature,
                )
                .context("paged skip-layer latency benchmark failed")?
            } else {
                eprintln!("--- Latency Benchmark (SKIP-LAYER — self-speculative, flat KV) ---");
                bench_latency_skiplayer(
                    &gpu_weights,
                    &model_config,
                    &tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                    args.seed,
                    args.temperature,
                )
                .context("skip-layer latency benchmark failed")?
            }
        }
        SpecMethod::Off => {
            if args.paged {
                eprintln!("--- Latency Benchmark (PAGED — production path) ---");
                bench_latency_paged(
                    &gpu_weights,
                    &model_config,
                    &tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                )
                .context("paged latency benchmark failed")?
            } else {
                eprintln!("--- Latency Benchmark ---");
                bench_latency(
                    &gpu_weights,
                    &model_config,
                    &tokenizer,
                    args.prompt_tokens,
                    args.max_output_tokens,
                )
                .context("latency benchmark failed")?
            }
        }
    };

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
        eprintln!("\n--- Training Benchmark (skipped) ---");
        None
    } else {
        eprintln!("\n--- Training Benchmark ---");
        match bench_training(&model_config, &gpu_weights, &tokenizer, args.training_steps) {
            Ok(result) => {
                eprintln!(
                    "  => {:.2}s/step, peak VRAM {} MB",
                    result.secs_per_step, result.peak_vram_mb
                );
                Some(result)
            }
            Err(e) => {
                eprintln!("  Training benchmark failed: {e}");
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
    let runner = ModelRunner::new(gpu_weights, runner_tokenizer, model_config.clone());

    // Inference throughput at different run counts
    eprintln!("\n--- Inference Throughput Benchmarks ---");
    let run_counts = [1, 4, 8, 16];
    let mut inference_results = Vec::new();

    for &n in &run_counts {
        eprintln!("\n{n} sequential runs:");
        match bench_inference(
            &runner,
            &tokenizer,
            n,
            args.prompt_tokens,
            args.max_output_tokens,
            args.seed,
            args.temperature,
        ) {
            Ok(result) => {
                eprintln!("  => {:.1} tok/s aggregate", result.tokens_per_sec);
                inference_results.push(result);
            }
            Err(e) => {
                eprintln!("  => FAILED: {e}");
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
