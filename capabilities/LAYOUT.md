# Uniform Capability Layout

This document defines the canonical layout that every capability dir under
`capabilities/caps/<cap>/` follows. Round 3 unifies the previously
methodology-keyed buckets (`agentic-grpo/`, `opd/`, `sft/`) into a single
flat `caps/` tree because **methodology is per-stage metadata, not per-cap
identity**.

Companion docs:

- [`README.md`](README.md) — top-level entry and tree map.
- [`METHODS.md`](METHODS.md) — when to choose SFT / OPD / GRPO / agentic-GRPO at any stage.
- [`PIPELINE.md`](PIPELINE.md) — multi-stage operating manual.
- [`DISTILLATION.md`](DISTILLATION.md) — cluster → new base flywheel.
- [`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) — the 40 kiln features this layout assumes are landed (all complete).
- [`CONSOLIDATED_REPORT.md`](CONSOLIDATED_REPORT.md) — round-1 lessons.
- [`lib/agentic-grpo-notes.md`](lib/agentic-grpo-notes.md) — ECHO defaults and the pi-rollout shape.

## Round-3 strategic updates

Round 3 carries the round-2 hardening (mandatory `rubric_sanity.py`,
multiplicative format gates, `integration/cross-cap-coherence`,
`hard_eval.tasks.jsonl`) and adds two new structural changes:

### 1. Flat capability tree (no paradigm split)

Round 2:
```
capabilities/
├── agentic-grpo/<cap>/
├── opd/<cap>/
├── sft/<cap>/
└── integration/
```

Round 3:
```
capabilities/
├── caps/<cap>/             ← every capability lives here
├── lib/                    ← shared helpers (was agentic-grpo/lib/)
├── integration/
└── rounds/                 ← round history + cluster manifests
```

A cap's directory no longer encodes which trainer it uses. The trainer
choice is per-stage metadata inside the cap.

### 2. Multi-stage pipelines

Round 2 caps were single-method (whichever bucket they lived in). Round 3
caps may be multi-method, in **stages**:

- `stages/stage-<N>-<slug>.json` records each kept stage.
- `pipeline.md` is the front-and-center recipe (the chain that won).
- `methods/<method>.config.json` holds per-method recipe defaults.
- `run_stage.sh <method> <slug>` runs a single stage.
- `run_pipeline.sh` re-runs the whole chain (e.g. after a base refresh).

`METHODS.md` decides which method to use at each stage; `PIPELINE.md`
governs stage transitions, validation gates, and base-adapter chaining.

Most round-2 caps migrate as single-stage pipelines. Multi-stage is the
*option* unlocked by round 3, not the default.

## Why a uniform layout

Round 1 produced 23 capabilities with wildly different file shapes. Round 2
normalized the file tree per-paradigm. Round 3 unifies across paradigms
because the same cap can use multiple paradigms across stages.

The kiln improvements in
[`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) make a single
tight contract possible:

- training writes `train_receipt.json` and `adapter_manifest.json` (#5, #8, #36)
- `kiln adapter verify` proves an adapter is loadable + behavioral (#4)
- `kiln eval-adapter` is the standard multi-seed paired eval driver (#33)
- `kiln trajectory inspect` is the standard rollout-debugging tool (#10)
- `cuda_grpo_ablation --dry-run` catches data/mask bugs before GPU work (#9)
- `--filter-var-min` is the official strong-signal filter (#22)
- `--install-adapter-dir`/`--install-adapter-name` removes path-symlink bugs (#5)
- `--adapter-smoke-test` is the standard post-train sanity check (#19)
- `kiln serve --eval-mode` is the standard during eval (#15)
- `cuda_opd_remote` is the OPD trainer (#37)

Each cap's `run_stage.sh` is a thin orchestration of those primitives,
plus the stage-validation gates from `PIPELINE.md` §4.

## File tree

Every cap dir SHOULD look like this. Items marked **R** are required for
a cap to be runnable; items marked **O** are optional but conventional;
**S** are stage-aware additions introduced in round 3.

```
caps/<cap>/
├── capability.md                # R   The contract (description, rubric, §0, baseline, headroom, hypotheses)
├── capability.config.json       # R   Per-method configs (schema_version=3, see §"capability.config.json")
├── capability.jsonl             # R   Append-only iter log (stage/method/base_adapter/output_adapter fields)
├── pipeline.md                  # S   The chosen pipeline: header + per-stage prose (see PIPELINE.md §3)
├── stages/                      # S   One file per kept stage (see PIPELINE.md §2.3)
│   ├── stage-1-<slug>.json
│   ├── stage-2-<slug>.json
│   └── ...
├── methods/                     # S   Per-method recipe defaults
│   ├── sft.config.json          #     ← present if SFT used in any stage
│   ├── opd.config.json          #     ← present if OPD used
│   ├── grpo.config.json
│   └── agentic-grpo.config.json
├── rubric.py                    # R   Composite reward function
├── rubric_sanity.py             # R   Calibration sanity check (MANDATORY)
├── build_corpus.py              # R   Task generator → datasets/{train,eval,hard_eval}.tasks.jsonl, sft/opd/grpo training data
├── rollout.py                   # R*  Agentic only: pi-runner → rollout JSONL
├── capability.oracle.sh         # R   Wraps `kiln eval-adapter` for blind eval
├── run_stage.sh                 # S   `./run_stage.sh <method> <slug> [--base-adapter <prev>]`
├── run_pipeline.sh              # S   Re-runs the chain from pipeline.md
├── capability.anchor.sh         # O   SFT-only: regression watch on a non-target domain
├── README.md                    # O   Cap-specific quickstart (links capability.md)
├── archive/                     # O   Old experimental artifacts (read-only history)
│   ├── README.md                #     Index of archived material
│   ├── round-1.jsonl            #     Prior iter logs by round
│   ├── round-2.jsonl
│   ├── writeups/                #     Old WRITEUP.md / FINAL_*.md / closeout.md
│   └── scripts/                 #     Old drive_iters.sh etc.
├── hypotheses/                  # O   Alternative experiments + verdicts
│   └── <slug>.md
├── calibration/                 # R   Rubric sanity fixtures (MANDATORY)
│   ├── README.md
│   ├── good.jsonl               #     ≥5 known-high-quality rollouts
│   └── bad.jsonl                #     ≥5 known-low-quality rollouts (one per §0 cheat)
├── datasets/                    # R   Task data + method-specific training data
│   ├── train.tasks.jsonl        # R   Source corpus, committed
│   ├── eval.tasks.jsonl         #     GITIGNORED — blind-eval firewall
│   ├── hard_eval.tasks.jsonl    #     GITIGNORED — round-failures-derived hard pool
│   ├── hard_eval.README.md      #     How to build the hard pool
│   ├── sft.train.jsonl          # S   Method-specific training data (built by build_corpus.py)
│   ├── opd.prompts.jsonl        # S
│   └── grpo.tasks.jsonl         # S
├── manifest/                    # O   Per-iter reproducibility manifests
│   └── README.md
└── prompts/                     # O   Legacy round-2 OPD/SFT location; new caps use datasets/*.{train,prompts}.jsonl instead
    └── train.jsonl
```

### Top-level (under `capabilities/`)

```
capabilities/
├── README.md
├── LAYOUT.md                ← this file
├── METHODS.md               ← when to choose which methodology
├── PIPELINE.md              ← multi-stage operating manual
├── DISTILLATION.md          ← cluster → new base flywheel (Phase G, deferred)
├── NEXT_ROUND.md            ← round-3 operating manual
├── CONSOLIDATED_REPORT.md   ← round-1 lessons (promoted from agentic-grpo/)
├── KILN_IMPROVEMENT_ISSUES.md  ← canonical kiln backlog (promoted from agentic-grpo/)
├── caps/                    ← every capability (flat — no paradigm split)
│   ├── pi-doctest/
│   ├── code-symbol-extraction/
│   ├── math-broad/
│   └── ...
├── lib/                     ← shared helpers (promoted from agentic-grpo/lib/)
│   ├── README.md
│   ├── pi_trajectory.py
│   ├── test_pi_trajectory.py
│   ├── stage_manifest.py    ← NEW: read/write/validate pipeline.md ↔ stages/
│   ├── method_router.py     ← NEW: apply METHODS.md decision tree
│   ├── headroom.py          ← NEW: per-sub-score headroom analysis
│   ├── cluster_summary.py   ← NEW: aggregate pipeline winners for distillation
│   └── agentic-grpo-notes.md  ← agentic-specific lore (ECHO defaults, pi shape)
├── integration/             ← cross-cap evaluation track
│   ├── README.md
│   ├── cross-cap-coherence/
│   ├── pi-tool-call-efficiency/  ← repurposed as transfer-eval-only
│   └── pi-source-mod-workflow/   ← repurposed as integration test
└── rounds/                  ← round history + cluster manifests
    ├── round-1/
    ├── round-2/
    └── round-3/             ← current
```

## Roles per file

### `capability.md` — The contract (committed)

Authoritative for the cap. Sections (in this order):

1. `# Capability: <name>`
2. `## Description` — what the model has to do; concrete failure modes.
3. `## Base model` — Qwen3.5-4B served by kiln on :8420.
4. `## Rollout source` — pi / direct HTTP / teacher; how data is gathered.
5. `## Rubric (v<N>)` — sub-score table with weights and cheat resistance.
6. `## Adversarial design (§0)` — at least 3 named cheats and mitigations.
7. `## Baseline + Headroom` — measured baseline composite, headroom number,
   target sub-score(s).
8. `## Hypotheses` — H1, H2, … (design intent; verdicts go in `capability.jsonl`).
9. `## Standard workflow` — pointer to `run_stage.sh`/`run_pipeline.sh` and
   `kiln`-CLI usage.
10. `## Kiln features used` — explicit list (verify, eval-adapter, dry-run, etc.).

`capability.md` describes the **goal**, not the methodology. Methodology
choice is in `pipeline.md` and `stages/`.

### `capability.config.json` — Per-method configs (committed, schema_version=3)

Single source of truth for hyperparameters. Schema:

```json
{
  "schema_version": 3,
  "shared": {
    "base_model_path": "/workspace/Qwen3.5-4B",
    "kiln_url": "http://localhost:8420",
    "adapter_dir": "/workspace/adapters",
    "sandbox_root": "/tmp/<cap>-stages",
    "eval": {
      "seeds": 3,
      "max_tasks": null,
      "thinking_mode": "off"
    }
  },
  "methods": {
    "sft": {
      "trainer": "cuda_sft_file",
      "data_file": "datasets/sft.train.jsonl",
      "defaults": {
        "rank": 4,
        "alpha": 8,
        "lr": 1e-4,
        "epochs": 1,
        "dataset_cap": 128,
        "seed": 3141592653,
        "adapter_smoke_test": true
      }
    },
    "opd": {
      "trainer": "cuda_opd_remote",
      "prompts_file": "datasets/opd.prompts.jsonl",
      "teacher_url": "http://localhost:8002",
      "teacher_name": "qwen3.6-27b-awq",
      "defaults": {
        "rank": 16,
        "alpha": 32,
        "lr": 1e-4,
        "epochs": 6,
        "samples_per_prompt": 2,
        "seed": 3141592653,
        "adapter_smoke_test": true
      }
    },
    "grpo": {
      "trainer": "cuda_grpo_ablation",
      "data_file": "datasets/grpo.tasks.jsonl",
      "defaults": {
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
        "filter_var_min": 0.05,
        "adapter_smoke_test": true
      }
    },
    "agentic-grpo": {
      "trainer": "cuda_grpo_ablation",
      "data_file": "datasets/grpo.tasks.jsonl",
      "rollout": {
        "pi_bin": "/usr/bin/pi",
        "pi_model_id": "Qwen3.5-4B",
        "num_generations_train": 4,
        "num_generations_eval": 1,
        "max_wall_clock_s": 120,
        "max_turns": 8,
        "max_tokens_per_turn": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "parallel": 1
      },
      "defaults": {
        "mode": "phase1",
        "advantage_mode": "dr_grpo",
        "loss_aggregation": "token_level",
        "kl_estimator": "k1",
        "kl_coeff": 0.1,
        "clip_epsilon": 0.20,
        "dynamic_sampling": true,
        "lr": 1e-5,
        "rank": 16,
        "alpha": 32,
        "seed": 3141592653,
        "filter_var_min": 0.05,
        "loss": {
          "echo": {
            "lambda": 0.05,
            "env_mask_mode": "env_only",
            "warning_filter": true
          },
          "no_policy_loss": false
        },
        "adapter_smoke_test": true
      }
    }
  },
  "pipeline": {
    "max_stages": 5,
    "between_stages": {
      "run_sibling_check": true,
      "stop_on_sibling_regression": true,
      "sibling_threshold": -0.02,
      "preserve_prior_stage_threshold": -0.02
    },
    "transitions": {
      "criterion_format_floor": 0.7,
      "criterion_process_headroom_min": 0.08,
      "criterion_reward_variance_min": 0.05
    }
  }
}
```

A method appears in `methods.<name>` ONLY if the cap's pipeline currently
uses that method (in any stage) or could plausibly use it. Most caps will
have 1-3 methods listed.

### `capability.jsonl` — The iter log (committed, append-only)

One JSON object per line, one line per iter. Round-3 row schema:

```json
{
  "iter": 7,
  "stage": 2,
  "method": "opd",
  "slug": "stage-2-opd-polish",
  "ts": "2026-06-08T...",
  "status": "kept",
  "family": "stage-transition",
  "hypothesis": "<one-sentence claim under test>",
  "rubric_version": "v1",

  "base_adapter": "<cap>-stage-1-sft-bootstrap",
  "output_adapter": "<cap>-stage-2-opd-polish",
  "stage_transition_rationale": "METHODS.md Rule E fired: format=0.78 stable, process headroom=0.32 of total, teacher available.",

  "composite": 0.852,
  "composite_delta": 0.104,
  "sub_scores": {...},
  "verdict": "positive",
  "sigma_warning": null,

  "method_specific": {
    "opd": {"effective_steps": 32, "teacher_calls_made": 64, "env_ce_delta": -0.45, "skip_rate": 0.21}
  },

  "sibling_regression_check": {"max_delta": -0.008, "sigma": 0.011, "passed": true},
  "prior_stage_preservation_check": {"prev_stage_composite_now": 0.752, "prev_stage_composite_orig": 0.748, "delta": 0.004, "passed": true},

  "kiln_commit": "<sha>",
  "train_receipt": "<path>",
  "adapter_manifest": "<path>",
  "notes": "..."
}
```

The 5 round-3-new fields are `stage`, `method`, `base_adapter`,
`output_adapter`, `stage_transition_rationale`. Plus `method_specific`,
`sibling_regression_check`, `prior_stage_preservation_check`.

Old round-2 rows without these fields are interpreted as
`stage: 1, method: <inferred-from-archive>`. Migration is non-destructive.

### `pipeline.md` — The chosen pipeline (committed, round-3-new)

See [`PIPELINE.md`](PIPELINE.md) §3 for the full schema. Header is YAML
front matter parseable by `lib/stage_manifest.py`; body is human-written
per-stage rationale and evidence.

A cap without `pipeline.md` is **not yet shipped**. Even single-stage caps
write a one-stage `pipeline.md`.

### `stages/stage-<N>-<slug>.json` — Per-stage record (round-3-new)

See [`PIPELINE.md`](PIPELINE.md) §2.3. Every kept iter that becomes a stage
gets a corresponding `stages/` file. The invariant is:

> Every file in `stages/` corresponds to exactly one kept iter in
> `capability.jsonl`. Their `output_adapter`, `composite`, and slug match.

`lib/stage_manifest.py --validate <cap>` checks this.

### `methods/<method>.config.json` — Per-method recipe defaults (round-3-new)

Optional override of `capability.config.json::methods.<method>.defaults`
for cap-specific tuning. Used when the same cap has different
hyperparameters per stage that don't fit cleanly in the
`capability.config.json` model.

### `rubric.py` — Composite reward (committed)

Pure function. Must expose:

- `score_one(rollout: dict) -> dict[str, float]` returning every sub-score
  and `composite`.
- A top-level `RUBRIC_VERSION` constant string.
- A `CHEAT_PROBES: list[Callable]` list (optional) for `rubric_sanity.py`.

Importable on a CPU-only dev box without network or a running kiln server.

### `rubric_sanity.py` — Calibration (committed, MANDATORY)

Loads `calibration/good.jsonl` and `calibration/bad.jsonl`, scores both
with `rubric.score_one`, asserts the good set scores cleanly above the bad
set with margin > 0.2. `run_stage.sh` runs this BEFORE any GPU work.

### `build_corpus.py` — Task generator (committed)

Reads seed data + writes:

- `datasets/train.tasks.jsonl` (the task corpus; committed)
- `datasets/eval.tasks.jsonl` (held-out eval; gitignored — blind firewall)
- `datasets/hard_eval.tasks.jsonl` (round-failures-derived; gitignored)
- `datasets/sft.train.jsonl` (built lazily when SFT is used)
- `datasets/opd.prompts.jsonl` (built lazily when OPD is used)
- `datasets/grpo.tasks.jsonl` (built lazily when GRPO is used; usually
  the same as `train.tasks.jsonl` with grpo-specific shape)

Method-specific data files MAY be built lazily by `run_stage.sh` when the
corresponding method is invoked, to avoid building data the cap doesn't
need yet.

### `rollout.py` — Pi runner (agentic-GRPO only, committed)

Reads a task JSONL, drives pi to produce session JSONLs, scores them via
`rubric.py`, writes `rollout.jsonl` (raw) and `grpo-train.jsonl`
(GRPO-trainer-shaped). For new caps, import from
[`lib/pi_trajectory.py`](lib/pi_trajectory.py) until `kiln rollout` (#34)
is exercised as the canonical path.

OPD and SFT methods don't need a `rollout.py`. The teacher response data
is the rollout for OPD; SFT trains directly from `datasets/sft.train.jsonl`.

### `capability.oracle.sh` — Blind eval (committed)

A thin wrapper around `kiln eval-adapter` (#33). Signature:

```bash
./capability.oracle.sh <adapter-name-or-empty> [--seeds N] [--tasks PATH]
```

Reference implementation lives in
`.agents/skills/capability-creator/templates/capability.oracle.sh`.

### `run_stage.sh` — Single-stage runner (round-3-new)

```bash
./run_stage.sh <method> <slug> [--base-adapter <name>] [--iter <N>]
```

See [`PIPELINE.md`](PIPELINE.md) §5 for the full contract. Reference
implementations per method in
`.agents/skills/capability-creator/templates/run_stage_<method>.sh`.

### `run_pipeline.sh` — Full pipeline runner (round-3-new)

```bash
./run_pipeline.sh [--from-stage N] [--validate-only]
```

Re-runs all stages in `pipeline.md` from `--from-stage` (default 1).
`--validate-only` runs eval + adapter verify on each stage without
re-training. See [`PIPELINE.md`](PIPELINE.md) §6.

### `archive/` — Read-only history (optional)

Holds whatever experimental data prior rounds produced. The contract:

- A fresh-round agent does **not** need to read anything in `archive/`
  to run the cap.
- Anything that would otherwise clutter the cap root but predates this
  round goes here.
- Include a `README.md` summarising the archived material so a curious
  reader can find prior context.

Round-3 migration adds `archive/round-2.jsonl` capturing the prior
`capability.jsonl` verbatim.

## Kiln CLIs the layout depends on

Every cap script depends on these being on `$PATH`:

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

Cap scripts SHOULD NOT re-implement any of the above. Missing features
go in `KILN_IMPROVEMENT_ISSUES.md` with a stop-gap in the cap until they
land.

## What changed vs. round 2

| Round 2 | Round 3 |
| --- | --- |
| Caps live under `capabilities/{agentic-grpo,opd,sft}/<cap>/` | All caps live under `capabilities/caps/<cap>/` |
| `lib/` lives at `capabilities/agentic-grpo/lib/` | `lib/` lives at `capabilities/lib/` |
| `KILN_IMPROVEMENT_ISSUES.md` and `CONSOLIDATED_REPORT.md` under `agentic-grpo/` | At `capabilities/` top level |
| `run_iter.sh` (method baked in by paradigm dir) | `run_stage.sh <method> <slug>` (method is explicit) |
| No notion of stages; `capability.jsonl` rows are flat | `stage`/`method`/`base_adapter`/`output_adapter` are first-class row fields |
| Pipeline shape implicit in archive narrative | `pipeline.md` + `stages/<N>.json` are committed first-class artifacts |
| Methodology choice baked into directory | METHODS.md decision tree routes per stage |
| Multi-stage = abandoned cap + new cap in different paradigm dir | Multi-stage = additional row in `capability.jsonl` + new `stages/<N>.json` |
| Four overlapping methodology skills (3152 lines) | One `.agents/skills/capability-creator/` skill (~700 lines + per-method resources) |
| Distillation is unscoped future work | `DISTILLATION.md` defines the flywheel contract (Phase G, deferred) |
| `rounds/` doesn't exist | `rounds/round-N/` snapshots cluster manifests + distillation recipes |

## When in doubt

The reference caps are (after round-3 migration):

- **`caps/pi-doctest`** — most mature multi-component rubric and a
  three-seed reproducibility result. Reference template for agentic stages.
- **`caps/pi-terminal-bench-lite`** — multi-turn paper-track recipe and
  the `--no-policy-loss` verifier-free shape.
- **`caps/code-symbol-extraction`** — OPD reference (clean closeout
  + concrete next-iter recipe).
- **`caps/math-broad`** — SFT reference with anchor suite for regression.
- **`caps/pi-faithful-completion`** — first round-3 multi-stage pilot
  (SFT bootstrap → OPD polish → agentic-GRPO final, planned).

The reference docs are:

- [`METHODS.md`](METHODS.md) — methodology decision tree.
- [`PIPELINE.md`](PIPELINE.md) — multi-stage operating manual.
- [`DISTILLATION.md`](DISTILLATION.md) — cluster → new base flywheel.
- [`lib/agentic-grpo-notes.md`](lib/agentic-grpo-notes.md) — agentic
  defaults + ECHO design.
- [`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) — the kiln
  features this layout assumes.
