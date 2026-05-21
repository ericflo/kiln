# pi-script-fixup — verifier-free ECHO adaptation

**Status:** Scaffold. Phase 3 of `docs/plans/echo-integration-plan.md` —
the showcase cap for paper §5.5 *verifier-free env-only adaptation*.

**Goal:** demonstrate that a strong-but-stable agentic LoRA can keep
improving on out-of-distribution Python tasks *without* a verifier —
purely by learning to predict the environment outputs caused by its
own actions. The paper §5.5 numbers we're chasing on PyTerm-shaped
tasks:

| Eval set | Setting | Δ from baseline |
| --- | --- | --- |
| val100 (in-dist) | unfiltered ECHO-only | **+3.8 pp** |
| ITD (OOD) | parse/tool=1.0 filter, 100 steps | **+5.2 pp** |
| PyTerm (OOD) | parse/tool=1.0 filter, 100 steps | **+10.0 pp** |
| TBLite (OOD) | same recipe | -3.9 pp (negative result; recipe doesn't generalize uniformly) |

The cap takes the strongest Phase 2 ECHO checkpoint (i.e. from
`pi-terminal-bench-lite`), runs `cuda_grpo_ablation --no-policy-loss
--echo-lambda 0.05` for 100 steps on a filtered held-out task set, and
measures pass-rate before/after.

## Recipe (paper §5.5)

```bash
# 1. Take strongest Phase 2 ECHO checkpoint.
CHECKPOINT=/workspace/adapters/echo-tblite-iter5

# 2. Gather rollouts on PyTerm (held-out set) with the existing
#    checkpoint. Filter to "clean" trajectories: parse_compliance=1.0
#    AND no malformed tool calls.
python rollout.py --tasks datasets/pyterm.tasks.jsonl \
                  --adapter $CHECKPOINT --mode train \
                  --num-generations 4 \
                  --filter clean_tool_calls

# 3. Train 100 steps with --no-policy-loss (no GRPO term, only ECHO).
cuda_grpo_ablation \
    --data /tmp/pyterm-train.clean.jsonl \
    --model /workspace/qwen3.5-4b \
    --base-adapter $CHECKPOINT \
    --output /workspace/adapters/echo-verifier-free \
    --adapter echo-verifier-free \
    --mode phase1 \
    --no-policy-loss \
    --echo-lambda 0.05 \
    --max-groups 100

# 4. Eval pass-rate on the three OOD sets.
capability.oracle.sh echo-verifier-free  # PyTerm + ITD + val100
```

## Hypotheses

**H_unfiltered_rollouts (paper §5.5 control)**
Run the same 100 steps WITHOUT the parse/tool-quality filter. Paper
reports that on OOD task sets the policy enters bad interaction regimes
(malformed tool calls, parse errors, unproductive loops) so the env
signal becomes noisy. Filtering recovers ~+5–10 pp. Verifies the
recipe's robustness depends on rollout quality, not just the loss term.

**H_lambda_sweep**
Sweep λ_echo ∈ {0.01, 0.05, 0.10} during the verifier-free phase. Paper
§3.3 productive range is 0.01–0.05; we expect 0.10 to degrade.

**H_tblite_negative**
Reproduce paper §5.5's negative result: same recipe on TBLite-shaped
tasks (filesystem orchestration, less direct env feedback) degrades
performance. Confirms ECHO's verifier-free regime works when the
environment feedback is *informative* (Python tracebacks tightly
coupled to executed code) and breaks down when it isn't (shell state
visible only through commands).

## Files

```
capabilities/agentic-grpo/pi-script-fixup/
├── capability.md                — this file
├── capability.config.json       — config with --no-policy-loss default
├── capability.oracle.sh         — blind eval across val100/ITD/PyTerm/TBLite
├── capability.jsonl             — iter log
├── rubric.py                    — reuses pi-doctest's rubric
├── rollout.py                   — same as pi-doctest but with parse/tool filter
├── task_scaffold.py             — Python-script-shaped scaffolds
├── datasets/
│   ├── pyterm.tasks.jsonl       — paper §5.5 PyTerm (held-out)
│   ├── itd.tasks.jsonl          — internal-dev (held-out)
│   ├── val100.tasks.jsonl       — in-distribution
│   └── tblite.tasks.jsonl       — TBLite-shaped (negative control)
└── run_verifier_free.sh         — paper §5.5 recipe
```

## Notes
- This cap exists to demonstrate the technique, not to ship an adapter.
  The kept artifact is the cap itself (recipe + eval scaffolding) so
  future workflows can reproduce the verifier-free uplift on their
  own task distributions.
- The `--no-policy-loss` flag landed in Phase 3 of the integration
  plan; before that, training with the GRPO term masked required
  custom code. Now it's a single CLI flag.
- Receipt-grade evidence: each iter writes `receipt.json` capturing
  `loss.no_policy_loss=true`, `loss.echo.lambda`, and the pass-rate
  delta vs baseline on each held-out set.


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
Round 1 status: **scaffold for paper §5.5 verifier-free adaptation**.

Highest-leverage improvements:

1. **Base adapter dependency is mandatory.** This cap follows the
   paper's recipe of taking the strongest Phase 2 ECHO checkpoint and
   running 100 verifier-free steps. Without a strong base adapter the
   recipe produces noise. Wire `BASE_ADAPTER` resolution into
   `run_iter.sh` with a check that the named adapter exists in the
   registry; fail early if missing. Default `BASE_ADAPTER` should be
   the kept adapter from `pi-terminal-bench-lite` (or `pi-doctest`
   when terminal-bench-lite hasn't been trained yet).
2. **Run the paper's negative control on TBLite.** The paper expects
   TBLite to *regress* under §5.5 adaptation (-3.9pp). If we don't
   see the regression, the recipe didn't transfer; the experiment
   wasn't a faithful reproduction. Build TBLite into the eval suite
   alongside val100 / ITD / PyTerm.
3. **Verifier-free assertion.** `cuda_grpo_ablation --no-policy-loss`
   must zero the policy term; `train_receipt.echo_metrics` should
   show env CE dropping while `lora_delta_norm_summary` is nonzero.
   Round-2 receipts make this asserrtable in the iter row.
