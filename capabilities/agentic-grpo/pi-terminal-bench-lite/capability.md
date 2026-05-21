# pi-terminal-bench-lite — ECHO paper reproduction

**Status:** Scaffold, awaiting iter 0 baseline.

**Goal:** Reproduce the ECHO paper's headline §5.1 result on a kiln-shaped
agentic-GRPO loop. The paper claims **TerminalBench-2.0 pass@1 doubles**:

| Model | GRPO | ECHO | Factor |
| --- | --- | --- | --- |
| Qwen3-8B | 2.70% | 5.17% | 1.9× |
| Qwen3-14B | 5.17% | 10.79% | 2.1× |

We can't run TerminalBench-2.0 directly on kiln's Qwen3.5-4B production
backbone (different model family, different scale), but the **lite**
variant — `OpenThoughts-TBLite` from the paper Table 1 (100 small-model
calibrated tasks) — is exactly the right shape for our hardware.

This cap is the dedicated paper-reproduction receipt the integration
plan specified at Phase 2 (`docs/plans/echo-integration-plan.md` §5
Phase 2). It's deliberately separate from `pi-doctest` so the receipt is
clean: same rubric, same task budget, same recipe across runs; only
`--no-echo` vs default ECHO changes between paired runs.

## Pi configuration
- Pi binary: `/usr/bin/pi` on the kiln-runpod image.
- Model id served by kiln: `qwen-3.5-4b-kiln` (production target).
- Session JSONL location: `~/.pi/sessions/<uuid>.jsonl`.
- Turn budget: **16 turns per rollout** (matches paper §4).
- Wall-clock budget: **120 s per rollout** (kiln's standard ceiling).

## Reward function (designed with adversarial review applied — §0)

| Sub-score | Weight | What it measures | What it CANNOT be cheated by |
| --- | --- | --- | --- |
| `outcome` | 0.70 | Verifier exit code 0 (unit tests pass) | Empty workdir / no-op session |
| `tool_call_efficiency` | 0.15 | `1 - clip(n_calls / 12, 0, 1)` | Spamming `ls` |
| `format_compliance` | 0.10 | Fraction of assistant turns with parseable tool calls or final text | Mid-turn malformed `<tool_call>` |
| `no_loop` | 0.05 | Fraction of unique tool calls (deduped by name+args hash) | Identical bash repeat |

`composite = outcome × Σ(weights × sub_scores)`. Outcome is a hard
multiplicative floor — failing the verifier zeroes the composite even if
the model wrote pretty code along the way. Higher weights on outcome
than `pi-doctest` because TBLite tasks are calibrated harder: the
small-model pass rate is low (~5–10%), so per-rollout success is much
more informative than tool-call efficiency.

### Adversarial design (§0)
- **Q: Can the model pass by guessing without running tests?**
  Yes, technically — TBLite tasks have a deterministic test command but
  the model could write the answer file directly. Mitigation: hidden
  test cases in the verifier (not exposed in the user prompt).
- **Q: Can the model loop until something passes?**
  Yes — spam tool calls until reward triggers. Mitigation: 16-turn
  budget + `tool_call_efficiency`.
- **Q: Can the model pre-baked solutions from the task prompt?**
  Yes if the task prompt leaks the expected file content. Mitigation:
  task scaffold strips the canonical solution before pi sees it.

### Headroom + group-variance baseline (TO BE FILLED ITER 0)
- baseline composite: <0.xx>
- headroom: <1.0 - composite>
- baseline group variance: <0.xx>
- typical wall-clock per rollout: <S> seconds
- typical assistant tokens per rollout: <T> tokens

## Hypothesis families

**H_no_echo (ablation baseline)**
Run with `--no-echo`. Verifies the ECHO uplift is real by removing the
auxiliary loss term. Paper §5.1 expects this to fall back to the
GRPO-only baseline.

**H_echo_default (ECHO at λ=0.05)**
Run with default config (ECHO on, λ=0.05). Paper §5.1 expects ~2×
improvement over `--no-echo` on TBLite. Our kiln gate: ≥+0.10 composite
over `--no-echo` at 3-seed-verified variance (std<0.02).

**H_echo_lambda_sweep**
Sweep λ ∈ {0.01, 0.02, 0.05, 0.10}. Paper §3.3 productive range is
0.01–0.05; 0.1 degrades the policy. Confirms the same shape holds on
kiln's setup.

**H_echo_warning_filter_off**
Run with `KILN_ECHO_WARNING_FILTER=false`. Paper §3.2 shows warnings
memorize within ~60 steps; with the filter off we expect a worse
dynamics-holdout score even if composite is similar.

## Files

```
capabilities/agentic-grpo/pi-terminal-bench-lite/
├── capability.md                 — this file
├── capability.config.json        — kiln + pi paths, hyperparams
├── capability.jsonl              — iter log (one JSON line per iter)
├── capability.oracle.sh          — blind eval wrapper
├── rubric.py                     — score_rollout
├── task_scaffold.py              — init workdir for one task
├── rollout.py                    — spawns pi, scores, emits AgenticGroup JSONL
├── build_corpus.py               — TBLite → tasks.jsonl
├── calibration/sanity.py         — rubric sanity (3 good / 3 bad)
├── datasets/
│   ├── train.tasks.jsonl         — 70 TBLite tasks for training
│   └── eval.tasks.jsonl          — 30 TBLite tasks for held-out eval
├── hypotheses/                   — markdown per hypothesis
└── run_iter1.sh                  — per-iter training recipe
```

## Notes
- **Phase 2 validation gate:** dynamics-holdout cross-entropy on
  trajectories from a stronger model (Qwen3-32B if available, otherwise
  the strongest model the pod can run) drops ≥30% on the ECHO
  checkpoint vs the GRPO-only checkpoint AND pass-rate strictly
  improves. See `calibration/dynamics_holdout.py`.
- Receipt-grade evidence: each iter writes `receipt.json` capturing
  `echo.lambda`, `echo.env_ce_initial`, `echo.env_ce_final`,
  `echo.env_ce_drop_pct`, `echo.lambda_effective_final`.
- Once shipped: this becomes the structural template for any future
  agentic-GRPO cap that wants to validate ECHO contribution on its
  own task set.


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
