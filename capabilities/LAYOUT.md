# Uniform Capability Layout

This document defines the canonical layout that every capability dir under
`capabilities/{agentic-grpo,opd,sft}/<cap>/` follows. It supersedes the
ad-hoc shapes the first experimental round produced.

Companion docs:

- `capabilities/README.md` — top-level intro and tree map.
- `capabilities/agentic-grpo/README.md` — agentic-GRPO ECHO defaults and
  the pi-rollout shared library.
- `capabilities/agentic-grpo/KILN_IMPROVEMENT_ISSUES.md` — the 40 kiln
  improvements this layout assumes are landed.

## Why a uniform layout

The first experimental round produced 23 capabilities with wildly different
file shapes. Some had `WRITEUP.md`, others `FINAL_RESULTS.md`, others
`closeout.md`. Some scripts called `cuda_grpo_ablation` with hand-computed
`output/adapter/` paths; one cap symlinked the parent and silently ran evals
on stale weights. Multi-seed eval was a free-text section of the spec in one
cap and a `for` loop in another.

A second round can do better. The kiln improvements in `KILN_IMPROVEMENT_ISSUES.md`
make a tighter contract possible:

- training writes `train_receipt.json` and `adapter_receipt.json` (#5, #8)
- `kiln adapter verify` proves an adapter is loadable, has measurable effect (#4)
- `kiln eval-adapter` is the standard multi-seed paired eval driver (#33)
- `kiln trajectory inspect` is the standard rollout-debugging tool (#10)
- `cuda_grpo_ablation --dry-run` catches data/mask bugs before GPU work (#9)
- `cuda_grpo_ablation --filter-var-min` is the official strong-signal filter (#22)
- `--install-adapter-dir / --install-adapter-name` removes the path-symlink class of bugs (#5)
- `--adapter-smoke-test` is the standard post-train sanity check (#19)
- `kiln serve --eval-mode` is the standard during eval (#15)
- `adapter_manifest.json` + `kiln adapter restore` replaces ad-hoc B2 backup scripts (#36)

The uniform layout is designed so each cap script is a thin orchestration of
these kiln primitives rather than a re-implementation of them.

## File tree

Every cap dir SHOULD look like this. Items marked **R** are required for a
cap to be runnable; items marked **O** are optional but conventional.

```
<paradigm>/<cap>/
├── capability.md                # R   The contract (goal, task shape, rubric, hypotheses)
├── capability.config.json       # R   Trainer + rollout defaults (versioned)
├── capability.jsonl             # R   Append-only iter log (one row per iter)
├── rubric.py                    # R   Composite reward function
├── rubric_sanity.py             # O   Calibration-fixture sanity check
├── build_corpus.py              # R   Task generator → datasets/{train,eval}.tasks.jsonl
├── rollout.py                   # R*  Agentic only: pi-runner → rollout JSONL
├── capability.oracle.sh         # R   Wraps `kiln eval-adapter` for blind eval
├── run_iter.sh                  # R   Full iter recipe (rollouts → train → eval)
├── README.md                    # O   Cap-specific quickstart (links capability.md)
├── archive/                     # O   Old experimental artifacts (read-only history)
│   ├── README.md                #     Index of archived material
│   ├── capability.jsonl         #     The prior iter log (preserved verbatim)
│   ├── writeups/                #     Old WRITEUP.md / FINAL_*.md / closeout.md
│   ├── datasets/                #     Old datasets that may inform fresh corpus
│   ├── scripts/                 #     Old drive_iters.sh / record_iter.py / etc.
│   └── kiln-polish.jsonl        #     Old kiln-polish issue list (now in KILN_IMPROVEMENT_ISSUES.md)
├── hypotheses/                  # O   Alternative experiments + verdicts
│   └── <slug>.md
├── calibration/                 # O   Rubric sanity fixtures
│   ├── good.jsonl               #     Known-high-quality rollouts
│   └── bad.jsonl                #     Known-low-quality rollouts (cheats included)
├── datasets/                    # R   Task data
│   ├── train.tasks.jsonl        # R   Committed
│   └── eval.tasks.jsonl         #     GITIGNORED — blind-eval firewall
├── manifest/                    # O   Per-iter reproducibility manifests
│   └── README.md
└── prompts/                     # O*  OPD/SFT only: training prompts (committed)
    └── train.jsonl
```

## Roles per file

### `capability.md` — The contract (committed)

Authoritative for the cap. Must include the sections below. The
`pi-doctest/capability.md` is the established reference for agentic-GRPO;
`opd/code-symbol-extraction/capability.md` for OPD; `sft/math-broad/capability.md`
for SFT.

Required sections (in this order):

1. `# Capability: <name>`
2. `## Description` — what the model has to do; concrete failure modes.
3. `## Base model` — Qwen3.5-4B served by kiln on :8420.
4. `## Rollout source` — pi/HTTP/teacher; how trajectories are gathered.
5. `## Rubric (v<N>)` — sub-score table with weights and cheat resistance.
6. `## Adversarial design (§0)` — at least 3 named cheats and their mitigations.
7. `## Baseline + Headroom` — measured baseline composite, headroom number, target sub-score.
8. `## Hypotheses` — list of H1, H2, … (design intent; verdicts go in `capability.jsonl`).
9. `## Standard workflow` — pointer to `run_iter.sh` and `kiln`-CLI usage.
10. `## Kiln features used` — explicit list (verify, eval-adapter, dry-run, etc.).

### `capability.config.json` — Trainer + rollout defaults (committed)

Single source of truth for hyperparameters. New round defaults:

```json
{
  "schema_version": 2,
  "base_model_path": "/workspace/qwen3.5-4b",
  "kiln_url": "http://localhost:8420",
  "pi_bin": "/usr/bin/pi",
  "pi_model_id": "qwen-3.5-4b-kiln",
  "adapter_dir": "/workspace/adapters",
  "sandbox_root": "/tmp/<cap>-rollouts",
  "rollout": {
    "num_generations_train": 4,
    "num_generations_eval": 1,
    "max_wall_clock_s": 120,
    "max_turns": 8,
    "max_tokens_per_turn": 1024,
    "temperature": 0.8,
    "top_p": 0.95,
    "parallel": 1
  },
  "training_phase1_defaults": {
    "mode": "phase1",
    "advantage_mode": "dr_grpo",
    "loss_aggregation": "token_level",
    "kl_estimator": "k1",
    "kl_coeff": 0.1,
    "clip_epsilon": 0.20,
    "dynamic_sampling": true,
    "is_level": "token",
    "reference_policy": "base_per_step",
    "lr": 1e-5,
    "rank": 16,
    "alpha": 32,
    "seed": 3141592653,
    "filter_var_min": null,
    "loss": {
      "echo": {
        "lambda": 0.05,
        "env_mask_mode": "env_only",
        "warning_filter": true
      },
      "opd": null,
      "no_policy_loss": false
    },
    "adapter_smoke_test": true
  },
  "eval": {
    "seeds": 3,
    "max_tasks": null,
    "thinking_mode": "off"
  }
}
```

OPD caps replace `training_phase1_defaults` with `training_defaults` keyed by
OPD knobs (`rank`, `alpha`, `lr`, `epochs`, `samples_per_prompt`, etc.).
SFT caps use a similar shape with `epochs` and `dataset_size_cap`.

### `capability.jsonl` — The iter log (committed, append-only)

One JSON object per line, one line per iter. Required fields:

```json
{
  "iter": 0,
  "slug": "baseline",
  "ts": "2026-05-21T00:00:00Z",
  "status": "kept" | "kept-with-caveat" | "ablation" | "infra-fail" | "negative",
  "family": "baseline" | "H1" | "H2" | ...,
  "hypothesis": "<one-sentence claim under test>",
  "rubric_version": "v0",
  "composite": 0.0,
  "composite_delta": 0.0,
  "sub_scores": { "<name>": 0.0 },
  "verdict": "positive" | "null" | "negative" | "inconclusive",
  "training": { "...": "..." },
  "rollout_stats": { "...": "..." },
  "kiln_commit": "<sha>",
  "adapter_manifest": "<path or null>",
  "train_receipt": "<path or null>",
  "notes": "..."
}
```

For a fresh round, start with a single header row (iter 0 = scaffold) or
leave empty. Do NOT carry the previous round's rows into the new log;
archive them.

### `rubric.py` — Composite reward (committed)

Pure function. Must expose:

- `score_one(rollout: dict) -> dict[str, float]` returning every sub-score and `composite`.
- A top-level `RUBRIC_VERSION` constant string.
- A `CHEAT_PROBES: list[Callable]` list (optional) for `rubric_sanity.py`.

`rubric.py` must be importable on a CPU-only dev box without network or a
running kiln server.

### `rubric_sanity.py` — Calibration (committed, optional)

Loads `calibration/good.jsonl` and `calibration/bad.jsonl`, scores both with
`rubric.score_one`, and asserts the good set scores cleanly above the bad
set. If your cap doesn't have one, start from `pi-doctest/calibration/sanity.py`.

### `build_corpus.py` — Task generator (committed)

Reads seed data + writes `datasets/train.tasks.jsonl` and
`datasets/eval.tasks.jsonl`. `eval.tasks.jsonl` is **gitignored** per
`.gitignore` — once it exists the agent must not read it. Use a deterministic
seed so the eval split is reproducible.

### `rollout.py` — Pi runner (agentic-grpo only, committed)

Reads a task JSONL, drives pi to produce session JSONLs, scores them via
`rubric.py`, writes `rollout.jsonl` (raw) and `grpo-train.jsonl`
(GRPO-trainer-shaped). For new caps, prefer importing from
`capabilities/agentic-grpo/lib/pi_trajectory.py` until `kiln rollout` (#34)
is exercised as the canonical path.

OPD and SFT caps do not need a `rollout.py`. The teacher response data is
the rollout for OPD; SFT trains directly from `prompts/`.

### `capability.oracle.sh` — Blind eval (committed)

A thin wrapper around `kiln eval-adapter` (#33). Signature:

```bash
./capability.oracle.sh <adapter-name-or-empty> [--seeds N] [--tasks PATH]
```

Reference implementation:

```bash
#!/usr/bin/env bash
set -euo pipefail
ADAPTER="${1:-}"
shift || true
TASKS="${TASKS:-datasets/eval.tasks.jsonl}"
KILN_URL="${KILN_URL:-http://localhost:8420}"
SEEDS="${SEEDS:-3}"
ADAPTER_DIR="${ADAPTER_DIR:-/workspace/adapters}"

if ! curl -sf "$KILN_URL/v1/health" > /dev/null 2>&1; then
  echo "ORACLE_ERROR: kiln-server not reachable at $KILN_URL" >&2
  exit 2
fi

# kiln eval-adapter handles base vs adapter, multi-seed, summary writing.
kiln eval-adapter \
  --url "$KILN_URL" \
  --adapter "$ADAPTER" \
  --adapter-dir "$ADAPTER_DIR" \
  --tasks "$TASKS" \
  --seeds "$SEEDS" \
  --scorer "./rubric.py" \
  --output "/tmp/<cap>-eval-${ADAPTER:-base}.json" \
  --thinking off \
  "$@"

# Standard contract: print SCORE=<composite> for the orchestrator.
python3 - <<PY
import json
d = json.load(open("/tmp/<cap>-eval-${ADAPTER:-base}.json"))
print(f"SCORE={d['mean_composite']:.4f}")
print(f"N={d['n_tasks']}")
for k, v in d['sub_scores_mean'].items():
    print(f"{k}={v:.4f}")
PY
```

### `run_iter.sh` — Iter recipe (committed)

Parameterized iter recipe. Argument: optional iter slug.

Reference shape for agentic-GRPO:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SLUG="${1:-h1-default-recipe}"
CFG="capability.config.json"
OUT_ROOT="/tmp/<cap>-iter-${SLUG}"
ROLLOUT_DIR="$OUT_ROOT/rollouts"
ADAPTER_NAME="<cap>-${SLUG}"
ADAPTER_REGISTRY="${ADAPTER_DIR:-/workspace/adapters}"
mkdir -p "$ROLLOUT_DIR"

# 1. Gather training rollouts via pi.
python3 rollout.py \
  --tasks datasets/train.tasks.jsonl \
  --out-dir "$ROLLOUT_DIR" \
  --config "$CFG" \
  --num-generations 4 \
  --mode train \
  --limit 30

# 2. Sanity-check the trajectories. Fails if no trainable action tokens.
kiln trajectory inspect "$ROLLOUT_DIR/grpo-train.jsonl" --json \
  > "$OUT_ROOT/trajectory_inspect.json"

# 3. Dry-run validation BEFORE any GPU work.
KILN_CUDA_ARCHS=86 /workspace/kiln/target/release/examples/cuda_grpo_ablation \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
  --model /workspace/qwen3.5-4b \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --mode phase1 \
  --rank 16 --alpha 32 --lr 1e-5 \
  --num-generations 4 \
  --seed 3141592653 \
  --filter-var-min 0.05 \
  --dry-run

# 4. Real training. Auto-installs into the registry.
KILN_CUDA_ARCHS=86 /workspace/kiln/target/release/examples/cuda_grpo_ablation \
  --data "$ROLLOUT_DIR/grpo-train.jsonl" \
  --model /workspace/qwen3.5-4b \
  --output "$OUT_ROOT/adapter" \
  --adapter "$ADAPTER_NAME" \
  --mode phase1 \
  --rank 16 --alpha 32 --lr 1e-5 \
  --num-generations 4 \
  --seed 3141592653 \
  --filter-var-min 0.05 \
  --adapter-smoke-test \
  --install-adapter-dir "$ADAPTER_REGISTRY" \
  --install-adapter-name "$ADAPTER_NAME"

# 5. Verify the installed adapter is loadable and behavioral.
kiln adapter verify "$ADAPTER_NAME" --adapter-dir "$ADAPTER_REGISTRY" --url http://localhost:8420

# 6. Blind eval.
SEEDS=3 ./capability.oracle.sh "$ADAPTER_NAME"

# 7. Append a row to capability.jsonl referencing the train_receipt.json.
python3 record_iter.py \
  --iter-slug "$SLUG" \
  --train-receipt "$OUT_ROOT/adapter/train_receipt.json" \
  --eval-summary "/tmp/<cap>-eval-${ADAPTER_NAME}.json" \
  >> capability.jsonl
```

OPD `run_iter.sh` is the same minus rollouts; SFT is the same minus rollouts
and ECHO.

### `archive/` — Read-only history (optional)

Holds whatever experimental data the prior round produced. The contract:

- The agent picking up a fresh round does **NOT** need to read anything in
  `archive/` to run the cap.
- Anything that would otherwise be at the cap root but predates this layout
  goes here.
- Includes a `README.md` summarising the archived material so a curious
  reader can find prior context without spelunking.

## Kiln CLIs the layout depends on

Every cap script depends on these being available on `$PATH`:

| Command | Purpose | Issue # |
| --- | --- | --- |
| `kiln serve --eval-mode` | Deterministic serving during eval | 15 |
| `kiln adapter verify <name>` | Prove adapter is loadable + behavioral | 4 |
| `kiln adapter restore <manifest>` | Re-materialize adapter from manifest | 36 |
| `kiln trajectory inspect <jsonl>` | Mask + token-count diagnostic | 10 |
| `kiln eval-adapter --adapter ... --seeds N` | Multi-seed paired eval | 33 |
| `kiln rollout --adapter ... --tasks ...` | Direct HTTP rollout (alternative to pi) | 34 |
| `cuda_grpo_ablation --dry-run` | Pre-GPU validation | 9 |
| `cuda_grpo_ablation --filter-var-min` | Strong-signal filter | 22 |
| `cuda_grpo_ablation --install-adapter-dir / --install-adapter-name` | Atomic install | 5 |
| `cuda_grpo_ablation --adapter-smoke-test` | Post-train sanity check | 19 |
| `cuda_opd_remote ...` | Off-policy distillation trainer | 37 |
| `cuda_sft_file ...` | SFT trainer | (pre-existing) |

Cap scripts SHOULD NOT re-implement any of the above. If a kiln CLI is
missing a feature your cap needs, file an item against `KILN_IMPROVEMENT_ISSUES.md`
and use a stop-gap in the cap until it lands.

## What changed vs. round 1

| Round 1 | Round 2 |
| --- | --- |
| Each cap re-implements eval driver | `kiln eval-adapter` + `capability.oracle.sh` shim |
| Adapter paths hand-computed (`output/adapter/`) | `--install-adapter-dir` + `kiln adapter verify` |
| Per-cap `backup_to_b2.py` | `adapter_manifest.json` + `kiln adapter restore` |
| Per-cap `record_iter.py` | Read `train_receipt.json` (canonical fields) |
| `kiln-polish.jsonl` per cap | Single `KILN_IMPROVEMENT_ISSUES.md` (now resolved) |
| `lib/pi_trajectory.py` (Python) | `kiln trajectory inspect` (Rust, canonical) |
| Custom strong-signal filtering | `--filter-var-min` flag |
| Wide variation in writeup filenames | One cap-level closeout in `capability.md` §Hypotheses + `capability.jsonl` |
| Ad-hoc dry-run logic | `cuda_grpo_ablation --dry-run` |
| ECHO observability via grep | `train_receipt.json::echo_metrics` |

## When in doubt

The reference caps are:

- `agentic-grpo/pi-doctest` — most mature multi-component rubric and a
  three-seed reproducibility result. Use it as the agentic template.
- `agentic-grpo/pi-terminal-bench-lite` — multi-turn paper-track recipe and
  the `--no-policy-loss` verifier-free shape.
- `opd/code-symbol-extraction` — OPD reference (clean closeout + concrete
  next-iter recipe).
- `sft/math-broad` — SFT reference with anchor suite for regression.

The reference docs are:

- `capabilities/agentic-grpo/README.md` — ECHO defaults and pi-rollout shape.
- `capabilities/agentic-grpo/KILN_IMPROVEMENT_ISSUES.md` — the kiln features
  this layout assumes are landed (all 40 are done in the next round).
