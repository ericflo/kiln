# `capabilities/` — capability uplift workdir (round 3)

One subdirectory per capability the team is training. Round 3 unifies the
previously methodology-keyed buckets (`agentic-grpo/`, `opd/`, `sft/`)
into a single flat `caps/` tree because **methodology is per-stage metadata,
not per-cap identity**.

```
capabilities/
├── README.md                  ← you are here
├── LAYOUT.md                  ← uniform cap-dir layout (round 3, stage-aware)
├── METHODS.md                 ← when to choose SFT / OPD / GRPO / agentic-GRPO at any stage
├── PIPELINE.md                ← multi-stage operating manual
├── DISTILLATION.md            ← cluster → new base flywheel (Phase G, deferred)
├── NEXT_ROUND.md              ← round-3 operating manual (the practical guide)
├── CONSOLIDATED_REPORT.md     ← round-1 lessons (promoted from agentic-grpo/)
├── KILN_IMPROVEMENT_ISSUES.md ← canonical kiln backlog (all 40 complete)
│
├── caps/                      ← every capability (flat — no paradigm split)
│   ├── pi-doctest/
│   ├── pi-code-search/
│   ├── pi-code-comprehension/
│   ├── pi-faithful-completion/
│   ├── pi-failure-triage/
│   ├── pi-diff-patch-apply/
│   ├── pi-compaction/
│   ├── pi-terminal-bench-lite/
│   ├── pi-script-fixup/
│   ├── pi-precondition-check/
│   ├── pi-shell-hygiene/
│   ├── pi-test-interpretation/
│   ├── pi-error-recovery/
│   ├── pi-context-aware-edits/
│   ├── pi-incremental-progress/
│   ├── pi-search-then-read/
│   ├── code-fence-language-fidelity/
│   ├── code-symbol-extraction/
│   ├── diff-patch-fluency/
│   ├── faithful-code-summarization/
│   ├── tool-call-arg-fidelity/
│   ├── transcript-compaction/
│   ├── json-schema-adherence/
│   ├── math-broad/
│   └── python-algo/
│
├── lib/                       ← shared helpers
│   ├── pi_trajectory.py       (pi session → ScoredRollout normalizer)
│   ├── stage_manifest.py      (validate pipeline.md ↔ stages/ ↔ capability.jsonl)
│   ├── method_router.py       (apply METHODS.md decision tree)
│   ├── headroom.py            (per-sub-score headroom analysis)
│   ├── cluster_summary.py     (aggregate pipelines for distillation cluster)
│   └── agentic-grpo-notes.md  (ECHO defaults + pi-rollout shape)
│
├── integration/               ← cross-cap evaluation track (eval-only, no training)
│   ├── README.md
│   ├── cross-cap-coherence/   (composite of per-cap composites)
│   ├── pi-tool-call-efficiency/  (transfer eval — wraps other caps' adapters)
│   └── pi-source-mod-workflow/   (end-to-end clone→PR integration test)
│
└── rounds/                    ← round history + cluster manifests
    ├── round-1/               (snapshot: round-1 winners, no distillation)
    ├── round-2/               (snapshot: hardening + new caps, no distillation)
    └── round-3/               (current; distillation when ≥5 multi-stage caps ship)
```

## Round-3 status

Round 3 is **the unification round.** Two structural changes vs. round 2:

### 1. Capability tree is flat

Methodology no longer appears in the directory path. A cap that started
under `opd/` lives at `caps/<cap>/` now. Its choice of methodology per
stage is recorded in `pipeline.md` and per-iter rows in `capability.jsonl`.

### 2. Pipelines can be multi-stage

A capability's final adapter is now the result of an ordered chain of
training stages, each possibly using a different methodology. The chain
that won is documented in `pipeline.md`; each kept stage gets a
`stages/stage-<N>-<slug>.json` file. Most round-2 caps migrate as
single-stage pipelines initially; multi-stage is unlocked from round 3
onward by [`METHODS.md`](METHODS.md) routing decisions.

### Strategic carry-forward from round 2

- **`rubric_sanity.py` is MANDATORY.** Round 1 hit "rubric too lax" 3
  times. The gate runs BEFORE training in every stage and blocks the iter
  on calibration failure (margin > 0.2 required between good/bad fixtures).
- **Multiplicative format gate** is the default composite shape:
  `composite = outcome × format × (process + base)`.
- **`integration/cross-cap-coherence/`** runs eval against held-out slices
  of every member cap. Round 3 mandates it **between every stage**, not
  just at closeout, because stacked adapters clobber siblings more readily.
- **`hard_eval.tasks.jsonl` pattern** per cap: a failures-derived pool
  where base composite is < 0.5. Lift on hard_eval is the cleanest signal
  of capability uplift vs. lucky-tasks.

### What rounds 1 + 2 shipped (preserved)

Round 1 winners (single-method) all live under `caps/` now with their
`archive/` preserved:

| Cap | Round 1 result | Round 3 plan |
| --- | --- | --- |
| `pi-faithful-completion` | +8.28pp 3-seed | Multi-stage pilot: SFT → OPD → agentic-GRPO |
| `pi-code-comprehension` | +12.93pp | Cross-file generalization eval + OPD polish |
| `pi-doctest` | +4.2pp 3-seed | Hidden tests sub-score; possibly OPD addition |
| `pi-code-search` | +2.4pp | precision_of_read sub-score + harder corpus |

The 40 kiln improvements in
[`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) are **all
complete**. Round-3 cap scripts assume them.

## Reading order for a new agent

1. **`README.md`** (this file) — tree map, where things live.
2. **`LAYOUT.md`** — the uniform cap-dir layout, schema_version=3.
3. **`METHODS.md`** — the decision tree. **The most important new doc.**
4. **`PIPELINE.md`** — multi-stage operating manual.
5. **`NEXT_ROUND.md`** — practical operating guide: pre-flight, server
   hygiene, stage lifecycle, receipt schema, diagnostics ladder.
6. **The cap's `capability.md`** — the contract.
7. **The cap's `pipeline.md`** — the chain that won (if shipped).
8. **The cap's `README.md`** — quickstart.
9. **`DISTILLATION.md`** — read when ≥5 multi-stage caps ship and Phase G fires.

## Standard quickstart per cap (round 3)

```bash
cd capabilities/caps/<cap>/

# 0. Build corpus (one time).
python3 build_corpus.py

# 1. Populate calibration (mandatory).
$EDITOR calibration/good.jsonl calibration/bad.jsonl

# 2. Sanity-check the rubric.
python3 rubric_sanity.py

# 3. Baseline eval (no adapter).
./capability.oracle.sh

# 4. Get methodology recommendation from the decision tree.
python3 ../../lib/method_router.py --eval-summary <baseline-eval.json>

# 5. Run the first stage with the recommended method.
./run_stage.sh <method> stage-1-<slug>

# 6. Inspect the new row.
tail -1 capability.jsonl | python3 -m json.tool

# 7. Cross-cap regression check (mandatory between stages).
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh <new-stage-adapter>

# 8. Decide on the next stage.
python3 ../../lib/method_router.py --eval-summary <stage-1-eval.json>
# If the tree recommends a different method AND headroom > 0.05, run:
cd ../../caps/<cap>/
./run_stage.sh <next-method> stage-2-<slug> --base-adapter <cap>-stage-1-<slug>
```

The full pipeline (after stages are settled) re-runs with:

```bash
./run_pipeline.sh
```

## Reproducibility — no adapters in git

Adapters are **derived artefacts**. Given:

- the kiln commit SHA (in `train_receipt.json` and `adapter_manifest.json`)
- the base model identifier
- the teacher model identifier + version (OPD only)
- the training data path + sha256 (in `train_receipt.json`)
- the hyperparameters + seed (in `train_receipt.json` and
  `capability.config.json`)
- the rubric (`rubric.py`, committed)
- **the pipeline structure** (`pipeline.md` and `stages/`, committed)

…the adapter chain is reproducible by re-running `./run_pipeline.sh`.
The trainer-owned `train_receipt.json` (#8) and `adapter_manifest.json` (#36)
capture everything needed. `kiln adapter restore <manifest>` re-materializes
any individual stage adapter from an archived copy + verifies hashes.

## What's committed vs. ignored

See `.gitignore` for the full list. In short:

| Committed | Ignored |
| --- | --- |
| `capability.md` (contract) | `datasets/eval.tasks.jsonl` (blind-eval firewall) |
| `capability.config.json` | `datasets/hard_eval.tasks.jsonl` (blind-eval firewall) |
| `capability.jsonl` (iter log) | `adapters/` |
| `pipeline.md` (the chain) | `responses/`, `experiments/`, `.eval/` |
| `stages/*.json` (kept stages) | `*.log` |
| `methods/<m>.config.json` | Teacher fixtures (regenerable) |
| `rubric.py`, `rubric_sanity.py` | `__pycache__/` |
| `build_corpus.py`, `rollout.py` | |
| `capability.oracle.sh`, `run_stage.sh`, `run_pipeline.sh` | |
| `hypotheses/*.md` | |
| `prompts/train.jsonl` (committed) | |
| `datasets/{train,sft.train,opd.prompts,grpo.tasks}.jsonl` (committed) | |
| `manifest/<iter>.json` | |
| `calibration/{good,bad}.jsonl` | |
| `archive/` (round-1+2 history) | |

## When in doubt

| Question | Where to look |
| --- | --- |
| "How is a cap dir laid out?" | [`LAYOUT.md`](LAYOUT.md) |
| "Which methodology should I use at this stage?" | [`METHODS.md`](METHODS.md) |
| "How do stages chain and validate?" | [`PIPELINE.md`](PIPELINE.md) |
| "What's the round-3 operating manual?" | [`NEXT_ROUND.md`](NEXT_ROUND.md) |
| "What kiln features can I assume?" | [`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) |
| "What did round 1 learn?" | [`CONSOLIDATED_REPORT.md`](CONSOLIDATED_REPORT.md) and each cap's `archive/` |
| "What's the ECHO default for agentic stages?" | [`lib/agentic-grpo-notes.md`](lib/agentic-grpo-notes.md) |
| "How does pi-session normalization work?" | [`lib/README.md`](lib/README.md) + `kiln trajectory inspect --help` |
| "What's the reference cap shape?" | `caps/pi-doctest/` (agentic) / `caps/code-symbol-extraction/` (OPD) / `caps/math-broad/` (SFT) / `caps/pi-faithful-completion/` (multi-stage pilot) |
| "Does my adapter regress other caps?" | `integration/cross-cap-coherence/` |
| "When does distillation fire?" | [`DISTILLATION.md`](DISTILLATION.md) §1 (≥5 multi-stage shipped pipelines) |
| "What round are we in and what shipped?" | `rounds/round-<N>/README.md` |
