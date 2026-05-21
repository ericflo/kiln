# `capabilities/` — capability uplift workdir (round 2)

One subdirectory per capability the team is running an experiment session on,
grouped by training paradigm:

```
capabilities/
├── LAYOUT.md                                   # ← read first: uniform layout spec
├── NEXT_ROUND.md                               # ← operating manual for round 2
├── opd/             # On-policy distillation (kiln OPD trainer + remote teacher)
│   ├── code-fence-language-fidelity/
│   ├── code-symbol-extraction/
│   ├── diff-patch-fluency/
│   ├── faithful-code-summarization/
│   ├── tool-call-arg-fidelity/
│   └── transcript-compaction/
├── sft/             # Supervised fine-tuning (kiln SFT trainer)
│   ├── json-schema-adherence/
│   ├── math-broad/
│   └── python-algo/
├── agentic-grpo/    # Multi-turn agentic GRPO (kiln + pi, ECHO-on by default)
│   ├── CONSOLIDATED_REPORT.md
│   ├── KILN_IMPROVEMENT_ISSUES.md
│   ├── README.md
│   ├── lib/
│   ├── pi-compaction/                          # paper-track caps
│   ├── pi-doctest/
│   ├── pi-script-fixup/
│   ├── pi-terminal-bench-lite/
│   ├── pi-precondition-check/                  # round-1 scaffolds (refined)
│   ├── pi-code-search/                         # round-1 verified wins
│   ├── pi-code-comprehension/
│   ├── pi-diff-patch-apply/                    # round-1 saturated, reshaped
│   ├── pi-failure-triage/                      # round-1 saturated, reshaped
│   ├── pi-tool-call-efficiency/                # repurposed as transfer eval
│   ├── pi-shell-hygiene/
│   ├── pi-test-interpretation/
│   ├── pi-source-mod-workflow/                 # reframed as integration test
│   ├── pi-faithful-completion/
│   ├── pi-error-recovery/                      # NEW round 2
│   ├── pi-context-aware-edits/                 # NEW round 2
│   ├── pi-incremental-progress/                # NEW round 2
│   └── pi-search-then-read/                    # NEW round 2
└── integration/                                # NEW round 2 — eval-only track
    ├── README.md
    └── cross-cap-coherence/
```

## Round 2 status

Every cap dir has been normalized to the uniform layout in
[`LAYOUT.md`](LAYOUT.md). The round-1 experimental artifacts (writeups,
old iter logs, ad-hoc scripts) live under each cap's `archive/`.

### Strategic changes since round 1

1. **4 new caps added** targeting high-leverage process behaviors:
   `pi-error-recovery`, `pi-context-aware-edits`,
   `pi-incremental-progress`, `pi-search-then-read`.

2. **2 saturated caps reshaped** with multiplicative format gates:
   `pi-diff-patch-apply`, `pi-failure-triage`. Round-1 evidence showed
   format moved +12.5pp on these while composite barely moved; the
   new rubric makes that signal visible.

3. **Integration track added** (`integration/cross-cap-coherence/`) —
   eval-only suite that measures whether one cap's training regresses
   any other cap.

4. **`rubric_sanity.py` is now mandatory** — every cap has a
   calibration gate that runs BEFORE training. Round-1 saw "rubric too
   lax" 3 times; this gate catches it for free.

5. **`hard_eval.tasks.jsonl` pattern documented** per cap — a
   round-1-failures-derived eval pool gives a much cleaner signal than
   the standard eval set when the baseline is near saturation.

6. **`pi-tool-call-efficiency` repurposed as transfer eval** —
   not a training cap; it measures tool-call efficiency across other
   adapters.

7. **`pi-source-mod-workflow` reframed as integration test** —
   the full clone→PR sequence was too long for clean GRPO signal.

### Round-1 wins preserved + extended

| Cap | Round 1 result | Round 2 plan |
| --- | --- | --- |
| `pi-faithful-completion` | +8.28pp 3-seed | Chain training from iter-50 best |
| `pi-code-comprehension` | +12.93pp | Cross-file generalization eval + OPD format polish |
| `pi-doctest` | +4.2pp 3-seed | Hidden tests sub-score (deferred §0 A1 mitigation) |
| `pi-code-search` | +2.4pp | `precision_of_read` sub-score + harder corpus |

### Round-1 kiln backlog: all 40 done

The 40 kiln improvements in
[`agentic-grpo/KILN_IMPROVEMENT_ISSUES.md`](agentic-grpo/KILN_IMPROVEMENT_ISSUES.md)
are **all completed**. Round-2 cap scripts assume they're available:

- `kiln eval-adapter --seeds N` (#33) — multi-seed paired eval
- `kiln adapter verify` (#4) — loadability + behavioral check
- `kiln trajectory inspect` (#10) — Rust mask + token-count diagnostic
- `kiln adapter restore` (#36) — adapter restoration from manifest
- `cuda_grpo_ablation --dry-run` (#9) — pre-GPU validation
- `cuda_grpo_ablation --filter-var-min` (#22) — strong-signal filter
- `cuda_*` trainer `--install-adapter-dir / --install-adapter-name` (#5)
- `cuda_*` trainer `--adapter-smoke-test` (#19)
- Trainer-owned `train_receipt.json` / `adapter_receipt.json` / `adapter_manifest.json`
- `kiln serve --eval-mode` (#15)

## Reading order for a new agent

1. **`LAYOUT.md`** — the uniform structure every cap follows. ~5 min read.
2. **`NEXT_ROUND.md`** — operating manual: pre-flight, server hygiene,
   adapter lifecycle, receipt schema, diagnostics ladder.
3. **The cap's `capability.md`** — the contract: goal, task shape, rubric,
   adversarial design (§0), hypotheses, Round 2 improvement plan.
4. **The cap's `README.md`** — quickstart.
5. **The cap's `capability.config.json`** — hyperparameters.
6. **`agentic-grpo/README.md`** — only if you're picking up an agentic-GRPO
   cap. Defines ECHO defaults and the pi-rollout shape.

## Standard quickstart per cap

```bash
cd capabilities/<paradigm>/<cap>/

# 0. Build corpus (one time).
python3 build_corpus.py

# 1. Populate calibration (mandatory).
#    Edit calibration/good.jsonl and calibration/bad.jsonl with at least
#    5 good and 5 bad fixtures (one bad per §0 cheat).
$EDITOR calibration/good.jsonl calibration/bad.jsonl

# 2. Sanity-check the rubric.
python3 rubric_sanity.py

# 3. Baseline eval (no adapter).
./capability.oracle.sh

# 4. First training iter.
./run_iter.sh h1-default-recipe

# 5. Inspect the new row.
tail -1 capability.jsonl | python3 -m json.tool

# 6. Cross-cap regression check.
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh <adapter-name>
```

## Reproducibility — no adapters in git

Adapters are **derived artefacts**. Given:

- the kiln commit SHA (in `train_receipt.json` and `adapter_manifest.json`)
- the base model identifier
- the teacher model identifier + version (OPD only)
- the training data path + sha256 (in `train_receipt.json`)
- the hyperparameters + seed (in `train_receipt.json` and `capability.config.json`)
- the rubric (`rubric.py`, committed)

…the adapter is reproducible by re-running training. Round-2 trainers
produce a stable `train_receipt.json` (kiln #8) and `adapter_manifest.json`
(kiln #36) that capture everything needed. `kiln adapter restore <manifest>`
re-materializes the adapter from an archived copy + verifies hashes.

## What's committed vs. ignored

See `.gitignore` for the full list. In short:

| Committed | Ignored |
| --- | --- |
| `capability.md` (contract) | `datasets/eval.tasks.jsonl` (blind-eval firewall) |
| `capability.config.json` | `datasets/hard_eval.tasks.jsonl` (blind-eval firewall) |
| `capability.jsonl` (iter log) | `adapters/` |
| `rubric.py`, `rubric_sanity.py` | `responses/`, `experiments/`, `.eval/` |
| `build_corpus.py`, `rollout.py` | `*.log` |
| `capability.oracle.sh`, `run_iter.sh` | Teacher fixtures (regenerable) |
| `hypotheses/*.md` | `__pycache__/` |
| `prompts/train.jsonl` (committed) | |
| `datasets/train.tasks.jsonl` (committed) | |
| `manifest/<iter>.json` | |
| `calibration/{good,bad}.jsonl` | |
| `archive/` (round-1 history) | |

## When in doubt

| Question | Where to look |
| --- | --- |
| "How is a cap dir laid out?" | [`LAYOUT.md`](LAYOUT.md) |
| "What's the next-round operating manual?" | [`NEXT_ROUND.md`](NEXT_ROUND.md) |
| "What kiln features can I assume?" | [`agentic-grpo/KILN_IMPROVEMENT_ISSUES.md`](agentic-grpo/KILN_IMPROVEMENT_ISSUES.md) |
| "What did round 1 learn?" | `agentic-grpo/CONSOLIDATED_REPORT.md` and each `archive/` |
| "What's the ECHO default?" | [`agentic-grpo/README.md`](agentic-grpo/README.md) |
| "How does pi-session normalization work?" | `agentic-grpo/lib/README.md` + `kiln trajectory inspect --help` |
| "What's the reference cap shape?" | `agentic-grpo/pi-doctest` (agentic) / `opd/code-symbol-extraction` (OPD) / `sft/math-broad` (SFT) |
| "Does my adapter regress other caps?" | `integration/cross-cap-coherence/` |
