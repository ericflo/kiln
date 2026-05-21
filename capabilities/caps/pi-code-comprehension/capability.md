# Capability: pi-code-comprehension

## Description

Given a target symbol (a function or class) in a small Python code
snapshot, the agent — using pi's tool surface (`read`, `bash`/`grep`,
`edit`, `write`) — produces a STRUCTURED JSON summary of the symbol's:

- **inputs** (name, type, source_line)
- **returns** (type, source_line)
- **mutates** (`filesystem:`, `global:`, `arg:`)
- **calls** (name, file, line)
- **called_by** (file, line — including cross-file callers found by grep)
- **invariants** (implicit + explicit preconditions)
- **side_effects** (raises, log output, I/O)

The answer is emitted as the FINAL pi assistant turn, wrapped in
`<answer>{...}</answer>` tags (or as a top-level JSON object; the rubric
accepts both).

This capability is the agentic complement to the OPD
`faithful-code-summarization` cap: that one trains generation when the
context is already provided, this one trains the model to *fetch* the
context across files before summarising.

## Why this capability matters

Two clouderic patterns this targets:

1. **Edit-without-comprehension.** Without structured comprehension of a
   target function the agent edits around invariants ("assumes the lock
   is held") and ships green CI that subtly breaks runtime behaviour.
2. **Drive-by edits to long files.** Even when an agent reads the
   target file, it skips the cross-file caller graph — and edits a
   public function as if it had a single caller. The
   `verify-architecture-claims-in-source` agent note (memory) is exactly
   this.

A model that reliably produces grounded, structured summaries on every
edit-class task is the lowest-bandwidth deployment story for a 4B
coding agent. **And this matters specifically because the 7-field
schema becomes the model's mental scaffolding for edit tasks.**

## Base model

Qwen3.5-4B served by `kiln serve` on port 8420. Adapter loaded via
`POST /v1/adapters/load`.

## Rollout source

Multi-turn pi session. The model reads + greps + emits a final
`<answer>{...}</answer>` block. Sampling defaults (temperature=0.8,
top_p=0.95) from kiln. N=4 generations per task per training step.

Pi configured via `kiln pi-setup` to use `Qwen3.5-4B`. Headless
`pi -p` with `--session-dir`, `--offline`, `--no-context-files`.

## Reward function (v1 — multi-component, adversarially audited)

See `rubric.py` for the implementation. Composite:

```
composite = outcome × (
    0.20 · grounding
  + 0.15 · cross_file_caller_recall
  + 0.10 · invariant_coverage
  + 0.05 · format_compliance
  + 0.50
)
```

Where `outcome` is the *mean F1 across the 7 structured fields* (with
type / identifier normalisation and a small "abstention beats lying"
bonus on `called_by`):

| Field | Score | Notes |
|-------|-------|-------|
| `inputs`        | F1 on (name, type) pairs (60%) + F1 on names (40%) |
| `returns`       | F1 on type set (normalized) |
| `mutates`       | F1 on `tag:target` strings; tag-only matches get half-credit |
| `calls`         | F1 on name (75%) + (name,file) (25%) |
| `called_by`     | F1 on file basenames; abstention bonus when gold non-empty |
| `invariants`    | semantic F1 against gold paraphrase lists (Jaccard >= 0.30, ≥ 2 shared content tokens) |
| `side_effects`  | semantic F1, looser threshold |

Headline sub-scores (the ones the model can move with training):

| Sub-score | Weight | What it rewards |
|-----------|--------|-----------------|
| `outcome` (multiplier) | hard floor | Mean field-F1 across all 7 fields |
| `grounding` | 0.20 | Line-number citations within ±2 of gold |
| `cross_file_caller_recall` | 0.15 | Fraction of gold cross-file callers recovered |
| `invariant_coverage` | 0.10 | Fraction of gold invariants the summary matches (semantically) |
| `format_compliance` | 0.05 | All 6 required fields present in the JSON |

A perfect rollout scores 1.0; a no-answer rollout scores 0.0. See
`rubric_sanity.py` for the calibration battery — `perfect=1.0`,
`good_paraphrase=0.95`, `no_grep=0.70`, `intra_file_only=0.69`,
`line_1_bluff=0.80`, `bluff_unread=0.01`, `empty_json=0.01`,
`no_answer=0.0`, `garbage_text=0.0`.

## Adversarial design (§0)

| Cheat | How the rubric blocks it |
|-------|--------------------------|
| Empty JSON | outcome=0 → composite=0 |
| Skip reads, bluff JSON | outcome F1 ≪ 1 because names/types wrong |
| Always-cite-line-1 | grounding ≪ 1 |
| Copy docstring as invariants | gold has *implicit* invariants the docstring doesn't state |
| Overstuff fields | F1 (not recall) caps the precision loss |
| Same-file fake callers | cross_file_caller_recall = 0 |
| Pure abstention (empty list) | small honesty bonus on called_by (0.10) but penalises overall outcome |

## Headroom

To be measured at iter 0 baseline. Working hypothesis based on the
`pi-doctest` baseline shape and the OPD `code-symbol-extraction`
results: composite ~0.30-0.50, dominated by `grounding=0` and
`cross_file_caller_recall=0` because the 4B base default doesn't grep
for callers and rarely cites lines without prompting.

## Hypotheses to try

The goal is to lift composite by ~+0.10 in 50 iters. Each hypothesis
gets one row in `capability.jsonl`. Initial sequence:

**Recipe-search hypotheses (1-10):**
- H1: Default Phase 1 recipe (ECHO 0.05, lr 1e-5, rank 16) on full corpus
- H2: Strong-signal filter (var > 0.02) on iter-1 pool
- H3: Higher ECHO lambda (0.075) — paper §3.3 says still productive
- H4: Lower lr (5e-6) for less overshoot
- H5: Higher rank (32) for more representation budget
- H6: Doubled num_generations (4 → 8) for sharper advantages
- H7: Filter to "hard" tasks only (baseline composite < 0.5)
- H8: 2-epoch on strong-signal pool (validates 1-epoch sweet spot)
- H9: Strict outcome floor (gate outcome ≥ 0.3 before scoring sub-scores)
- H10: Pre-prompt the model with field schema reminders in the user message

**Target-rebalance hypotheses (11-20):**
- H11: Up-weight `grounding` (0.20 → 0.30) — biggest movable mass
- H12: Up-weight `cross_file_caller_recall` (0.15 → 0.25)
- H13: Penalize overstuffed JSON harder (precision-weighted F1)
- H14: Reward shorter pi sessions (tool-call efficiency add-on)
- H15: Require ≥ 1 grep call before allowing `called_by` to score
- H16: Add `consistency` bonus when same line is cited across fields
- H17: Add `partial_credit` for paraphrased invariants the rubric missed
- H18: KL coeff sweep (0.05, 0.10, 0.20)
- H19: clip_epsilon sweep (0.10, 0.20, 0.30)
- H20: Reduce max_wall_clock_s (60s) — forces decisive behavior

**Corpus + curriculum hypotheses (21-35):**
- H21: Curriculum: easy tasks (functions ≤ 10 lines) only, iter 1
- H22: Curriculum: hard tasks (functions ≥ 30 lines) only
- H23: Mixed corpus: seed + kiln's own scripts/
- H24: Synthetic-caller heavy corpus (every task has ≥ 3 cross-file callers)
- H25: Real-only corpus (only tasks where called_by came from real callers)
- H26: Adversarial-eval corpus (some functions have NO callers in pool)
- H27: Heuristic-augmented gold (use astroid for richer call-graph)
- H28: Multi-file targets: include classes spread across modules
- H29: Reward annealing: outcome multiplier → linear blend over iters
- H30: Field-disable ablation: train without scoring `mutates`
- H31: Hard-only filter: pre-baseline-composite < 0.3
- H32: Replay best-of-prior on each iter (off-policy distillation)
- H33: Self-play: model bootstraps the gold (then we audit it)
- H34: Caller-blind tasks: hide caller files; force outside-context grep
- H35: Invariant-rich tasks: only functions with ≥ 3 invariant patterns

**Sampling + decoding hypotheses (36-45):**
- H36: Temperature sweep (0.6, 0.8, 1.0)
- H37: Top-p sweep (0.9, 0.95, 0.99)
- H38: Min-p sampling (replaces top-p)
- H39: System prompt anchor: explicit "first, read; then, grep; then, answer"
- H40: Few-shot system prompt with one calibration-good example
- H41: Stop-token after `</answer>` (prevents continuation)
- H42: max-turns lower (8 → 5) to force decisive tool use
- H43: max-turns higher (8 → 15) for deeper exploration
- H44: Token-level loss aggregation vs. sample-level (config ablation)
- H45: Importance-sampling level: token vs sequence

**Verification + robustness hypotheses (46-50):**
- H46: Best iter, 2nd seed for eval variance estimate
- H47: Best iter, 2nd seed for training reproducibility
- H48: Best iter, expanded eval set
- H49: Best iter, hidden-tests stress (eval on out-of-distribution corpus)
- H50: Final adapter merge / ensemble across top-3 iters

## Pi prompt template (verbatim)

See `task_scaffold.py::pi_prompt`. Each task scaffolds the workdir
with the target file + 2-6 sibling files (some may contain callers).

## Hypothesis log

See `capability.jsonl` — one row per iter, append-only.

## Kiln-polish prerequisites

Multi-turn assistant masking is closed by ECHO Phase 0 (per shared
README §1). Single-turn final-answer JSON keeps things simple; the
rubric pulls the final assistant turn and ignores the rest of the
trajectory for scoring (but the trajectory IS the signal for GRPO via
the action+observation token masks).


## Round 2 setup

This cap was normalized to the round-2 layout on 2026-05-21. The previous
iter log and writeups are preserved in [`archive/`](archive/). The
`capability.jsonl` starts empty for the new round.

### Kiln features the new round uses

- `kiln adapter verify` (#4) — adapter loadability + behavioral check.
- `cuda_*` trainer `--install-adapter-dir` / `--install-adapter-name` (#5) —
  atomic install into the registry; no more `output/adapter/` symlink bugs.
- `train_receipt.json` (#8) — the canonical per-run artifact with kiln SHA,
  data hashes, hyperparameters, LoRA delta norms, and ECHO metrics.
- `cuda_grpo_ablation --dry-run` (#9) — pre-GPU validation of data, masks,
  base-adapter shape, and saturated-reward warnings.
- `kiln trajectory inspect` (#10) — Rust-native mask + token-count
  diagnostic; replaces the Python `lib/pi_trajectory.py` for new code.
- ECHO observability in receipt (#12) — env-token CE, action-token count,
  warning-prefix masked-out byte count.
- `kiln serve --eval-mode` (#15) — deterministic, no thinking, no
  per-request adapter drift.
- `--adapter-smoke-test` (#19) — post-train base-vs-adapter logit-delta check.
- `--filter-var-min` (#22) — official strong-signal filtering.
- `kiln eval-adapter --seeds N` (#33) — multi-seed paired-eval driver wrapped
  by `capability.oracle.sh`.
- `adapter_manifest.json` + `kiln adapter restore` (#36) — replaces ad-hoc B2
  backup scripts.

### Workflow

```bash
./capability.oracle.sh                     # baseline (no adapter)
./run_iter.sh h1-default-recipe            # first training iter
./run_iter.sh h2-lower-lr                  # subsequent
```

See [`run_iter.sh`](run_iter.sh) for the full pipeline.

## Round 2 improvement plan
Round 1 result: **+12.93pp big win** (iter 4, ECHO λ=0.075). Outcome,
grounding, cross-file all moved. ECHO λ=0.075 vs paper-suggested 0.05
was the productive ceiling.

Highest-leverage improvements:

1. **Cross-file *generalization* eval set.** Round 1 saturated
   cross-file recall at 1.00 by training the model to always grep.
   That's a process win but not proof of generalization. Build a
   held-out eval with a *different* repo layout (deeper nesting,
   different file naming, monorepo vs single-package) and measure
   whether the always-grep behavior transfers.
2. **OPD from 27B for structured JSON output.** The format
   sub-score has clean teacher signal — the 7-field JSON shape is
   exactly what OPD distills well. Try `cuda_opd_remote` chained
   *after* the round-1 best GRPO adapter (kiln #6 validates the chain
   shape). Hypothesis: format polish from OPD, behavior from GRPO,
   composing.
3. **Anchor regression suite.** Does this adapter hurt
   `pi-code-search` or `pi-doctest`? Run them after each iter as a
   side-check; round 1 hinted that broad code-reading training may
   come with style drift.
4. **Investigate ECHO λ=0.075 vs paper's 0.01–0.05.** Round 1 found
   the productive ceiling at 0.075. Worth a dedicated ablation:
   λ ∈ {0.025, 0.05, 0.075, 0.10} at fixed everything else.
