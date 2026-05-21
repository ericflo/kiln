# `capabilities/` — capability uplift workdir (round 2)

One subdirectory per capability the team is running an experiment session on,
grouped by training paradigm:

```
capabilities/
├── LAYOUT.md                                   # ← read first: uniform layout spec
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
└── agentic-grpo/    # Multi-turn agentic GRPO (kiln + pi, ECHO-on by default)
    ├── CONSOLIDATED_REPORT.md                  # round-1 lessons; archived context
    ├── KILN_IMPROVEMENT_ISSUES.md              # 40 round-2 kiln improvements (all done)
    ├── README.md                               # agentic-GRPO contract + ECHO defaults
    ├── lib/                                    # pi-trajectory python compat shim
    ├── pi-compaction/                          # paper-track caps
    ├── pi-doctest/
    ├── pi-script-fixup/
    ├── pi-terminal-bench-lite/
    ├── pi-precondition-check/                  # coding-agent caps (10-cap suite)
    ├── pi-code-search/
    ├── pi-code-comprehension/
    ├── pi-diff-patch-apply/
    ├── pi-tool-call-efficiency/
    ├── pi-shell-hygiene/
    ├── pi-test-interpretation/
    ├── pi-failure-triage/
    ├── pi-source-mod-workflow/
    └── pi-faithful-completion/
```

## Round 2 status

Every cap dir has been normalized to the uniform layout in
[`LAYOUT.md`](LAYOUT.md). The round-1 experimental artifacts (writeups,
old iter logs, ad-hoc scripts) live under each cap's `archive/`.

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
2. **The cap's `capability.md`** — the contract: goal, task shape, rubric,
   adversarial design (§0), hypotheses.
3. **The cap's `README.md`** — quickstart.
4. **The cap's `capability.config.json`** — hyperparameters.
5. **`agentic-grpo/README.md`** — only if you're picking up an agentic-GRPO
   cap. Defines ECHO defaults and the pi-rollout shape.

## Standard quickstart per cap

```bash
cd capabilities/<paradigm>/<cap>/

# 0. Build corpus (one time).
python3 build_corpus.py

# 1. Baseline eval (no adapter).
./capability.oracle.sh

# 2. First training iter.
./run_iter.sh h1-default-recipe

# 3. Inspect the new row.
tail -1 capability.jsonl | python3 -m json.tool
```

`run_iter.sh` orchestrates the full pipeline: rollouts → `kiln trajectory inspect`
→ `cuda_grpo_ablation --dry-run` → real training → `kiln adapter verify` →
`kiln eval-adapter` → append iter row.

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

### Teacher reproducibility note

The remote teacher (vLLM) is the only source of non-determinism that isn't
fully captured by the manifest. Two paths:

1. **Pin the teacher and live with it.** The manifest records vLLM version,
   model identifier, the quantisation, and launch flags. Same flags → same
   top-K logprobs to Q4 numerical noise. Good enough for adapter quality,
   not bit-identical reproduction.

2. **Save teacher logprobs as a fixture.** Dump
   `(rollout_tokens, active_positions, teacher_top_k_logprobs)` per step
   to a JSONL during training and replay offline with `FixtureLogitSource`.
   Tracked by individual OPD caps.

## What's committed vs. ignored

See `.gitignore` for the full list. In short:

| Committed | Ignored |
| --- | --- |
| `capability.md` (contract) | `datasets/eval.jsonl` (blind-eval firewall) |
| `capability.config.json` | `adapters/` |
| `capability.jsonl` (iter log) | `responses/`, `experiments/`, `.eval/` |
| `rubric.py`, `rubric_sanity.py` | `*.log` |
| `build_corpus.py`, `rollout.py` | Teacher fixtures (regenerable) |
| `capability.oracle.sh`, `run_iter.sh` | `__pycache__/` |
| `hypotheses/*.md` | |
| `prompts/train.jsonl` (committed) | |
| `datasets/train.tasks.jsonl` (committed) | |
| `manifest/<iter>.json` | |
| `archive/` (round-1 history) | |

The eval set itself is gitignored because the skill's information firewall
requires the agent not to read it. Treating `datasets/eval.jsonl` as a
write-only artefact prevents accidental inclusion in a diff review.

## When in doubt

| Question | Where to look |
| --- | --- |
| "How is a cap dir laid out?" | [`LAYOUT.md`](LAYOUT.md) |
| "What kiln features can I assume?" | [`agentic-grpo/KILN_IMPROVEMENT_ISSUES.md`](agentic-grpo/KILN_IMPROVEMENT_ISSUES.md) |
| "What did round 1 learn?" | `agentic-grpo/CONSOLIDATED_REPORT.md` and each `archive/` |
| "What's the ECHO default?" | [`agentic-grpo/README.md`](agentic-grpo/README.md) |
| "How does pi-session normalization work?" | `agentic-grpo/lib/README.md` + `kiln trajectory inspect --help` |
| "What's the reference cap shape?" | `agentic-grpo/pi-doctest` (agentic) / `opd/code-symbol-extraction` (OPD) / `sft/math-broad` (SFT) |
