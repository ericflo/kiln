# Round 3 — How To Run It

A practical guide for the agent picking up any capability dir in this tree.
Read [`LAYOUT.md`](LAYOUT.md), [`METHODS.md`](METHODS.md), and
[`PIPELINE.md`](PIPELINE.md) first; this doc is the *operating* manual.

## What changed strategically between round 2 and round 3

Quick orientation:

1. **Capabilities are flat** under `caps/`. Methodology is no longer in the
   directory path; it's per-stage metadata. `agentic-grpo/`, `opd/`, `sft/`
   directories are gone.
2. **One unified skill** — `.agents/skills/capability-creator/`. The four
   methodology-specific skills (`sft-`, `grpo-`, `opd-`, `agentic-grpo-
   capability-creator`) are deleted. Their irreducible lore lives in
   `resources/<method>-mode.md` inside the unified skill.
3. **`METHODS.md` is the brain.** It routes which methodology to use at any
   stage given the cap's measured state. Re-read it before every stage.
4. **Pipelines can be multi-stage.** Each cap's `pipeline.md` records the
   chain that won. `stages/<N>.json` files preserve per-stage records.
5. **`run_stage.sh <method> <slug>`** replaces the old method-baked
   `run_iter.sh`. Stage chaining via `--base-adapter <prev>`.
6. **Integration check is between every stage**, not just at closeout.
7. **`DISTILLATION.md` defines the flywheel.** Deferred to Phase G but
   the doc exists so the agent knows the destination.
8. **`rounds/round-<N>/`** is the round snapshot pattern.

Round-2 carry-overs (still mandatory):

- `rubric_sanity.py` before every iter
- Multiplicative format gate as the default composite shape
- `hard_eval.tasks.jsonl` pool for hardest signal
- 3-seed eval default (kiln #33)
- Strong-signal filtering (kiln #22)

## The 30-second version

```bash
# 0. Server
kiln serve --eval-mode --model-path /workspace/Qwen3.5-4B \
  --adapter-dir /workspace/adapters &

# 1. Pick a cap
cd capabilities/caps/pi-doctest

# 2. Build corpus + populate calibration + baseline
python3 build_corpus.py
$EDITOR calibration/good.jsonl calibration/bad.jsonl   # ≥5 good, ≥5 bad
python3 rubric_sanity.py                               # gate passes?
./capability.oracle.sh                                 # baseline (no adapter)

# 3. Ask the decision tree which method to use at stage 1
python3 ../../lib/method_router.py --eval-summary /tmp/eval-base.json

# 4. Run that stage
./run_stage.sh <method> stage-1-<slug>

# 5. Inspect the row + integration check
tail -1 capability.jsonl | python3 -m json.tool
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh <new-stage-adapter>

# 6. Decide on stage 2 (or stop)
cd ../../caps/pi-doctest/
python3 ../../lib/method_router.py --eval-summary /tmp/eval-stage-1.json
# If a different method is recommended AND headroom > 0.05:
./run_stage.sh <new-method> stage-2-<slug> --base-adapter pi-doctest-stage-1-<slug>
```

Everything from rubric-sanity → method choice → rollouts → trajectory
inspection → dry-run → train → install → adapter verify → multi-seed eval
→ iter-row append → sibling check is wrapped by `run_stage.sh`. The
full pipeline reruns with `./run_pipeline.sh`.

## Pre-flight before any GPU work

The single largest round-1 cost driver was bad data / bad config making it
all the way to GPU before failing. `cuda_*_ablation --dry-run` (kiln #9)
catches this. `run_stage.sh` runs the dry-run pass *before* real training
in every stage iter. What it catches before pod time:

- empty training JSONL or bad schema (kiln #24 `data_schema_error`)
- empty `action_mask` after tokenization (kiln #24 `zero_action_tokens`)
- ECHO enabled but `env_mask` is all-zero (kiln #24 `zero_env_tokens`)
- base-adapter rank mismatch / missing tensor (kiln #6, #24)
- alpha/rank > 2 without override (kiln #7, #24 `unsafe_lora_scale`)
- zero kept groups after `--filter-var-min` (kiln #22)
- saturated reward distribution (kiln #21, #38) → warns to use
  `--no-policy-loss` instead

If `cuda_*_ablation --dry-run` fails, fix it before touching the GPU.

## Round-3 pre-stage gates

In addition to the pre-flight above, every stage N ≥ 2 must pass the
[`PIPELINE.md`](PIPELINE.md) §4.1 pre-stage gates:

1. Previous stage adapter exists and verifies (loadable + behavioral).
2. `lib/method_router.py` recommendation matches this stage's chosen method
   OR `pipeline.md::stage_transition_rationale` documents an explicit override.
3. Stage hypothesis is falsifiable (names sub-score, magnitude, failure case).
4. `rubric_sanity.py` passes.
5. Integration check on previous stage adapter still passes (no silent
   regression introduced between stages by other caps' adapters or server drift).

## Server hygiene during eval

`kiln serve --eval-mode` (kiln #15) is the canonical eval-time config:

- thinking disabled by default for tool agents (kiln #17)
- adapter changes during eval are flagged as warnings (kiln #16 stress-test)
- deterministic decode where possible
- `/v1/health` exposes p50/p95/p99 latency + last adapter (kiln #13)

If `capability.oracle.sh` reports drift between paired-seed evals beyond
`stdev > delta`, that's a server-state issue; check `/v1/health` for
recent latency creep before re-running.

## Adapter lifecycle (the round-1 footgun, still relevant)

Round 1 had three separate adapter-related bugs:

1. **Output-path symlink mistake.** `cuda_grpo_ablation --output X/Y/`
   wrote to `X/Y/`; caps that symlinked `X/` instead loaded stale weights.
2. **Chat completion silently unloaded the active adapter** when the
   request omitted the `adapter` field.
3. **No fast way to prove an adapter actually changes behavior** —
   regressions looked like "the model got worse" but the adapter wasn't loaded.

Round-2 + 3 patterns:

- Trainer accepts `--install-adapter-dir <dir> --install-adapter-name <name>`
  (kiln #5). Atomic symlink into registry; no cap-side path arithmetic.
- After train, `kiln adapter verify <name>` (kiln #4) proves: layout OK,
  config + safetensors consistent, loads through `/v1/adapters/load`,
  registry shows it active, behavioral logit-delta nonzero on a fixed prompt.
- Chat-completion semantics (kiln #1): omitted `adapter` uses server default,
  explicit `null` or `""` uses base for that request, explicit name uses
  it only for that request.

In round 3, `run_stage.sh` always uses `--base-adapter <prev-stage-output>`
when N ≥ 2, never path arithmetic.

## Receipts replace log scraping

Every trainer (`cuda_grpo_ablation`, `cuda_sft_file`, `cuda_opd_remote`)
writes `train_receipt.json` next to the adapter (kiln #8). Schema in
`docs/TRAIN_RECEIPT_SCHEMA.md`. Key fields:

```json
{
  "schema_version": 1,
  "status": "ok" | "failed",
  "failure_reason": null | "<standard reason>",
  "kiln_commit": "<sha>",
  "kiln_dirty": false,
  "model_path": "/workspace/Qwen3.5-4B",
  "model_config_hash": "<sha256>",
  "tokenizer_config_hash": "<sha256>",
  "training_data_path": "...",
  "training_data_sha256": "<sha256>",
  "base_adapter_path": null | "...",
  "base_adapter_sha256": null | "<sha256>",
  "output_adapter_path": "...",
  "output_adapter_sha256": "<sha256>",
  "rank": 16,
  "alpha": 32,
  "alpha_over_rank": 2.0,
  "lr": 1e-5,
  "epochs": 1,
  "seed": 3141592653,
  "mode": "phase1",
  "kl_coeff": 0.1,
  "clip_epsilon": 0.20,
  "dynamic_sampling": true,
  "echo_enabled": true,
  "echo_lambda": 0.05,
  "echo_env_mask_mode": "env_only",
  "echo_warning_filter": true,
  "no_policy_loss": false,
  "groups_seen": 30,
  "groups_filtered": 5,
  "groups_trained": 25,
  "filter_var_min": 0.05,
  "reward_mean": 0.65,
  "reward_stdev": 0.22,
  "reward_min": 0.0,
  "reward_max": 1.0,
  "group_variance_histogram": [...],
  "action_token_count": 12000,
  "env_token_count": 35000,
  "context_token_count": 22000,
  "warning_filter_masked_bytes": 240,
  "echo_metrics": {
    "env_token_ce_initial": 2.14,
    "env_token_ce_final": 1.87,
    "env_ce_steps_observed": 25
  },
  "wall_clock_s": 312.4,
  "peak_vram_mib": 78539,
  "lora_delta_norm_summary": {...},
  "grad_norm_min_mean_max": [1.1e-4, 3.2e-3, 7.1e-2],
  "adapter_manifest_path": "/workspace/adapters/<name>/adapter_manifest.json"
}
```

Cap scripts read this directly; no log-grepping. In round 3, the iter-row
append also pulls `method_specific` diagnostics from this receipt.

## Eval the kiln way

`kiln eval-adapter` (kiln #33) is the canonical eval driver:

```bash
kiln eval-adapter \
  --url http://localhost:8420 \
  --adapter "<name>" \
  --adapter-dir /workspace/adapters \
  --tasks datasets/eval.tasks.jsonl \
  --seeds 3 \
  --scorer ./rubric.py \
  --output eval_summary.json \
  --thinking off
```

Paired mode across N seeds (default 3). Writes a stable `eval_summary.json`
with `mean_composite`, `composite_stdev`, `composite_delta`, sub-scores
mean/stdev, `sigma_warning`, `verdict`, `status`.

## Multi-seed eval is the rule, not the exception

Round 1 burned cycles on single-seed "wins" that didn't reproduce
(pi-doctest iter-2 +11pp single-seed → +4.2pp 3-seed). `kiln eval-adapter`
defaults to `--seeds 3`. Always run ≥3 seeds before declaring a positive iter.

## Strong-signal filtering is officialized

`--filter-var-min 0.05` (kiln #22) keeps only groups with reward variance
above the threshold. Round 1 found this is the single most reproducibly
useful knob beyond the defaults. Sidecar JSON records the exact kept/dropped
group ids so iter-to-iter reproduction is exact.

Empty-filter behavior is explicit:

- `--on-empty-filter fail` (default): trainer exits non-zero if zero
  groups remain after filtering.
- `--on-empty-filter train-all`: ignore the filter when it would empty
  the training set.
- `--on-empty-filter skip`: write a "skipped" receipt and exit zero.

## Stage transitions in practice

After every stage's promotion:

1. **Read METHODS.md §4 to identify candidate next stages.** The decision
   tree is the gating logic.
2. **Confirm headroom for the candidate.** If no sub-score has > 0.05
   residual headroom, stop the pipeline.
3. **State the next stage hypothesis up front** in `capability.md::Hypotheses`
   and `pipeline.md::stage_transition_rationale`.
4. **Run the integration check on the just-shipped stage adapter** before
   committing to a next stage. If sibling regression appears, fix it
   before extending the chain.

`run_pipeline.sh` enforces #4 mechanically. The agent is responsible for #1-3.

## Diagnostics ladder for "iter didn't move"

If `composite_delta` ≈ 0 after a real training step:

1. **Did rubric_sanity gate fire?** Check `$LOG_DIR/rubric_sanity.log`.
   If calibration didn't pass, the rubric itself is broken — fix that first.
2. **Was the adapter actually loaded?** Check `verify.behavioral == true`
   in `$LOG_DIR/verify.json`.
3. **Did ECHO fire** (agentic-GRPO only)? Check
   `train_receipt.json::echo_metrics::env_ce_steps_observed > 0`.
4. **Did weights move?** `lora_delta_norm_summary` not all near-zero.
5. **Is reward saturated?** Receipt `reward_saturation_warning` — switch to
   `--no-policy-loss` (ECHO-only) or use harder eval tasks.
6. **Is the rubric rewarding the wrong thing?** If sub-scores moved but
   composite didn't, check the weight matrix in `rubric.py` — consider
   whether multiplicative gates are needed.
7. **Cross-cap regression?** Run `integration/cross-cap-coherence/` against
   the adapter — sometimes the iter "didn't move *this* cap" but moved a
   sibling cap heavily, indicating skill-clobber.
8. **Stage transition wrong-method?** Re-run `lib/method_router.py`. If the
   recommended method has changed since this stage was started, you may
   be using the wrong tool now.

## Verifier-free / no-policy-loss recipes

For caps following paper §5.5 (e.g. `pi-script-fixup`):

```bash
ECHO_LAMBDA=0.05 BASE_ADAPTER=<best-stage-adapter> \
  ./run_stage.sh agentic-grpo h-vf-adaptation
# Inside run_stage_agentic_grpo.sh, --no-policy-loss is set when ECHO_LAMBDA > 0
# and config.methods.agentic-grpo.defaults.loss.no_policy_loss = true.
```

The trainer asserts the policy-loss path is zeroed but env-CE training
still runs. `train_receipt.echo_metrics` should show env CE dropping.

## Long-context caps (pi-compaction)

Round 1 hit the long-context training wall: 32K-64K sequences with
near-byte-identical adapters. Round 2 shipped:

- long-context bench suite at 8K/16K/32K/64K (kiln #25)
- per-phase progress logging (kiln #26)
- byte-identical-adapter diagnostic (kiln #27)
- testable warning-prefix masking (kiln #28)

`caps/pi-compaction/run_stage.sh` inherits these. Trust the receipts and
adapter-verify output rather than re-reading text.

## OPD method specifics

OPD (kiln #37) is first-class in `cuda_opd_remote`. Standard call (inside
`run_stage_opd.sh`):

```bash
cuda_opd_remote \
  --prompts datasets/opd.prompts.jsonl \
  --model /workspace/Qwen3.5-4B \
  --base-adapter $BASE_ADAPTER \
  --teacher-url http://localhost:8002 \
  --teacher-name qwen3.6-27b-awq \
  --output /tmp/.../adapter \
  --adapter <name> \
  --rank 16 --alpha 32 --lr 1e-4 --epochs 6 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name <name>
```

`train_receipt.json` carries `n_prompts`, `effective_steps` (true step count
after EOS-skip filtering), `teacher_calls_made`, and `skip_rate`. Round-1
`code-symbol-extraction` showed 97% EOS skip rate; the fix is visible in
the receipt.

## SFT method specifics

SFT caps use `cuda_sft_file`. Standard call:

```bash
cuda_sft_file \
  --data datasets/sft.train.jsonl \
  --model /workspace/Qwen3.5-4B \
  --base-adapter $BASE_ADAPTER \
  --output /tmp/.../adapter \
  --adapter <name> \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 \
  --dataset-cap 128 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name <name>
```

SFT caps also have `capability.anchor.sh` — a regression watch on a
non-target domain to catch stylistic clobber. `run_stage_sft.sh` runs it
after the target eval. Anchor regression > 0.02 fails the stage.

## Per-stage rubric design discipline

Round-1 + 2 hit "rubric too lax" three times. The discipline (also lives
in METHODS.md Rule B):

1. **Write the adversarial design (§0) section first**, before `rubric.py`.
   Name ≥3 cheats that would score 1.0 without doing the capability.
2. **Make outcome a hard floor**, not just a weighted component. Pattern:
   `composite = outcome × (w1·sub1 + w2·sub2 + ... + base)` — zero outcome
   → zero composite no matter how clean the process.
3. **Calibrate.** Write 5-10 known-good and 5-10 known-bad rollouts by hand.
   Run `rubric_sanity.py` and confirm clean separation.
4. **Check the baseline distribution.** If baseline composite > 0.95 over
   your eval set, you're in the "rubric too lax" zone. Round-2 trainer
   warns via `reward_saturation_warning`.

The rubric is not allowed to change without re-baselining. A rubric edit
mid-pipeline invalidates the stages above it.

## Iter / stage naming convention

**Within a stage:** `<family>-<descriptor>`. Examples:
- `h1-default-recipe`, `h2-lower-lr`, `h-vf-adaptation`

**Promoted stage:** `stage-<N>-<slug>`. Examples:
- `stage-1-sft-bootstrap`, `stage-2-opd-polish`, `stage-3-grpo-final`

**Adapter:** `<cap>-stage-<N>-<slug>`. Examples:
- `pi-faithful-completion-stage-1-sft-bootstrap`
- `pi-faithful-completion-stage-2-opd-polish`

Warm-best within a stage gets `-wbN`:
`pi-faithful-completion-stage-3-grpo-final-wb1`.

## When to stop iterating

Stop the pipeline when:

- 3-seed mean improvement on the cap's eval is within σ of the previous stage
  for 2 consecutive iters
- Sub-scores are saturated; `reward_saturation_warning` is firing
- METHODS.md decision tree's only remaining recommendation is one already tried
- Cross-cap regression cap is binding (max sibling delta near threshold)

A clean closeout is a `pipeline.md::status: shipped` plus a final iter row
with `status: "closeout"` in `capability.jsonl`. The cap's `capability.md`
gets a "## Closeout" section summarising winning recipe and remaining risks.

## Reproducibility checklist

Before declaring a pipeline shipped, the cap should pass:

- [ ] Final stage 3-seed mean Δ > 2σ above baseline
- [ ] Every kept stage's `train_receipt.json` exists and `status == "ok"`
- [ ] Every kept stage's `adapter_manifest.json` exists;
      `kiln adapter restore <manifest>` round-trips with matching SHA256
- [ ] `kiln adapter verify <final-adapter>` exits 0 (loadable + behavioral)
- [ ] `calibration/{good,bad}.jsonl` rubric sanity passes
- [ ] No `sigma_warning` in final stage's `eval_summary.json`
- [ ] `pipeline.md` header is valid YAML and matches `stages/`
- [ ] `lib/stage_manifest.py --validate <cap-dir>` exits 0
- [ ] `integration/cross-cap-coherence/capability.oracle.sh <final-adapter>`
      shows max sibling delta ≥ −0.02
- [ ] Iter rows in `capability.jsonl` carry `kiln_commit`,
      `adapter_manifest`, `train_receipt`, and stage/method/base_adapter fields

## Common questions

**Q: How do I add a new hypothesis?**
A: Add `H<N>: <one-sentence claim>` to the `## Hypotheses` section of
`capability.md`. Run `./run_stage.sh <method> h<N>-<descriptor>`. The
verdict auto-appears in `capability.jsonl`.

**Q: How do I chain training (use a previous adapter as base)?**
A: `./run_stage.sh <method> stage-<N>-<slug> --base-adapter <prev-stage-name>`.
Kiln #6 validates rank + target-module compatibility before optimizer setup.

**Q: How do I back up an adapter to B2?**
A: Don't write a custom backup script. `adapter_manifest.json` (kiln #36)
is the canonical archive shape. Copy the adapter dir to B2 and store the
manifest path. `kiln adapter restore <manifest>` round-trips it.

**Q: How do I add a new cap?**
A: Use `.agents/skills/capability-creator/templates/scaffold.sh` to create
`capabilities/caps/<slug>/` with the canonical files from
[`LAYOUT.md`](LAYOUT.md). Reference `caps/pi-doctest/` (agentic) /
`caps/code-symbol-extraction/` (OPD) / `caps/math-broad/` (SFT) /
`caps/pi-faithful-completion/` (multi-stage) for shape.

**Q: A round-1 cap is broken; should I read its archive?**
A: Only if the archive flags a behavior the round-3 rubric needs to handle.
Otherwise treat round-3 as a fresh start and trust the kiln receipts and
`capability.md` contract.

**Q: When do I run distillation?**
A: When ≥5 multi-stage round-3 caps ship with composite_delta > 0.05.
Until then, `DISTILLATION.md` is read-only context.

## Round 3 priority ranking

If you have to pick caps to run first (limited pod budget), this is the
recommended order:

### Tier 1 — Multi-stage pilot

1. **`caps/pi-faithful-completion`** — round-1 winner +8.3pp single-stage;
   round-3 pilot for multi-stage (SFT → OPD → agentic-GRPO). Validates the
   new stage chaining mechanics end-to-end.

### Tier 2 — High-leverage round-2 wins to extend

2. **`caps/pi-code-comprehension`** — round-1 +12.9pp; round-3 plan is
   cross-file generalization eval + OPD polish on top of agentic-GRPO base.
3. **`caps/pi-doctest`** — round-1 +4.2pp 3-seed; add hidden-tests
   sub-score; possibly OPD addition on the hard-tail.
4. **`caps/code-symbol-extraction`** — OPD reference; the round-1 EOS-skip
   bug is fixed; should jump 0.937 → 0.96+ with fresh recipe.

### Tier 3 — New caps with high estimated headroom (from round-2 NEXT_ROUND)

5. **`caps/pi-error-recovery`** — baseline ~0.40, headroom ~0.60.
6. **`caps/pi-incremental-progress`** — baseline ~0.50, headroom ~0.50.
7. **`caps/pi-context-aware-edits`** — baseline ~0.45, headroom ~0.55.
8. **`caps/pi-search-then-read`** — baseline ~0.40, headroom ~0.60.

### Tier 4 — Reshapes (re-test multiplicative gates)

9. **`caps/pi-diff-patch-apply`** — v2 multiplicative-gate rubric.
10. **`caps/pi-failure-triage`** — same.

### Tier 5 — Dependency-gated

11. **`caps/pi-compaction`** — first prove kiln #25 long-context bench
    moves weights at 32K; then switch to OPD chain.
12. **`caps/pi-script-fixup`** — needs strong base adapter from
    pi-terminal-bench-lite Phase 2.

### Tier 6 — Eval-only

13. **`integration/cross-cap-coherence/`** — run after every Tier 1-3
    stage to surface sibling regressions.
14. **`integration/pi-tool-call-efficiency/`** — run periodically against
    the suite of installed adapters.

### Tier 7 — Lower-priority but still in queue

15. **5 more OPD caps** (`caps/code-fence-language-fidelity`,
    `diff-patch-fluency`, `faithful-code-summarization`,
    `tool-call-arg-fidelity`, `transcript-compaction`) — run sequentially.
16. **SFT caps** (`caps/json-schema-adherence`, `caps/math-broad`,
    `caps/python-algo`) — standardize anchor suites; multi-seed eval;
    coordinate corpus with `pi-doctest` to avoid leak.
17. **Remaining agentic scaffolds** (`pi-shell-hygiene`,
    `pi-test-interpretation`, `pi-precondition-check`) — the §0 + corpus
    design work is non-trivial.

### When to start thinking about Phase G (distillation)

Trigger: ≥5 Tier 1-3 caps have `pipeline.md::status: shipped` with
composite_delta > 0.05 on 3-seed eval AND
`integration/cross-cap-coherence/` reports compatible clusters in
`rounds/round-3/sibling_matrix.json`. Until then, `DISTILLATION.md` is
read-only.
