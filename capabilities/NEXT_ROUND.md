# Round 2 — How To Run It

A practical guide for the agent picking up any capability dir in this tree.
Read [`LAYOUT.md`](LAYOUT.md) first; this doc is the *operating* manual.

## What changed strategically between round 1 and round 2

Quick orientation if you ran round 1 and are picking up round 2:

1. **`rubric_sanity.py` is mandatory.** It runs BEFORE training in
   every `run_iter.sh`. Failures block the iter. Populate
   `calibration/{good,bad}.jsonl` with at least 5 good and 5 bad
   fixtures (one bad per §0 cheat) before the first iter.
2. **Multiplicative format gate is the default composite shape** for
   new caps: `composite = outcome × format × (process + base)`. Round 1
   evidence showed additive composites trap +12.5pp of signal when
   outcome is saturated.
3. **`integration/cross-cap-coherence/`** runs eval against held-out
   slices of every member cap. Use it after every meaningful training
   iter to catch sibling-cap regressions.
4. **4 new caps** target high-leverage process behaviors:
   `pi-error-recovery`, `pi-context-aware-edits`,
   `pi-incremental-progress`, `pi-search-then-read`. These are the
   highest-priority round-2 caps in the new-work bucket.
5. **2 reshaped caps**: `pi-diff-patch-apply` and `pi-failure-triage`
   now use multiplicative format gates. Their round-1 v1 rubrics live
   under `archive/rubric_v1_additive.py`.
6. **`hard_eval.tasks.jsonl` pattern** per cap: a round-1-failures-derived
   pool where base composite is < 0.5. Lift on hard-eval is the cleanest
   signal of capability uplift vs. lucky-tasks.
7. **`pi-tool-call-efficiency` is eval-only** — wraps other caps'
   adapters and reports tool-call distributions; doesn't train.

## The 30-second version

```bash
# 0. Server
kiln serve --eval-mode --model-path /workspace/Qwen3.5-4B \
  --adapter-dir /workspace/adapters &

# 1. Pick a cap
cd capabilities/agentic-grpo/pi-doctest

# 2. Build corpus + populate calibration + baseline
python3 build_corpus.py
$EDITOR calibration/good.jsonl calibration/bad.jsonl   # >=5 good, >=5 bad
python3 rubric_sanity.py                               # gate passes?
./capability.oracle.sh                                 # baseline (no adapter)

# 3. First training iter
./run_iter.sh h1-default-recipe

# 4. Check the row
tail -1 capability.jsonl | python3 -m json.tool

# 5. Cross-cap regression check
cd ../../integration/cross-cap-coherence/
./capability.oracle.sh pi-doctest-h1-default-recipe
```

Everything from rubric-sanity → rollouts → trajectory inspection → dry-run
→ train → install → adapter verify → multi-seed eval → iter-row append is
in `run_iter.sh`. The cross-cap regression check is one-line.

## Pre-flight before any GPU work

The single largest round-1 cost driver was bad data / bad config making it all
the way to GPU before failing. Round 2 fixes this with `--dry-run` (kiln #9).
`run_iter.sh` runs the dry-run pass *before* real training in every iter.
What it catches before you spend pod time:

- empty training JSONL or bad schema (kiln #24 `data_schema_error`)
- empty `action_mask` after tokenization (kiln #24 `zero_action_tokens`)
- ECHO enabled but `env_mask` is all-zero (kiln #24 `zero_env_tokens`)
- base-adapter rank mismatch / missing tensor (kiln #6, #24)
- alpha/rank > 2 without override (kiln #7, #24 `unsafe_lora_scale`)
- zero kept groups after `--filter-var-min` (kiln #22)
- saturated reward distribution (kiln #21, #38) → warns to use
  `--no-policy-loss` instead

If `cuda_grpo_ablation --dry-run` fails, fix it before touching the GPU.

## Server hygiene during eval

`kiln serve --eval-mode` (kiln #15) is the canonical eval-time config:

- thinking disabled by default for tool agents (kiln #17)
- adapter changes during eval are flagged as warnings (kiln #16 stress-test)
- deterministic decode where possible
- `/v1/health` exposes p50/p95/p99 latency + last adapter (kiln #13)

If `capability.oracle.sh` reports drift between paired-seed evals beyond
`stdev > delta`, that's a server-state issue; check `/v1/health` for
recent latency creep before re-running.

## Adapter lifecycle (the round-1 footgun)

Round 1 had three separate adapter-related bugs:

1. **Output-path symlink mistake.** `cuda_grpo_ablation --output X/Y/`
   wrote to `X/Y/`. Caps that symlinked `X/` instead of `X/Y/` loaded
   the wrong dir into kiln, which kept the old adapter active.
2. **Chat completion silently unloaded the active adapter** when the
   request omitted the `adapter` field.
3. **No fast way to prove an adapter actually changes behavior** —
   regressions read as "the model got worse," but the adapter wasn't
   loaded at all.

Round 2 patterns:

- Trainer accepts `--install-adapter-dir <dir> --install-adapter-name <name>`
  (kiln #5). It validates the produced adapter, then atomically symlinks
  into the registry. No cap-side path arithmetic.
- Trainer prints `ADAPTER_DIR=<absolute path>` on its own line on
  success. Receipt confirms.
- After train, `kiln adapter verify <name>` (kiln #4) proves: layout OK,
  config + safetensors consistent, loads through `/v1/adapters/load`,
  registry shows it active, behavioral logit-delta nonzero on a fixed prompt.
- Chat-completion semantics: omitted `adapter` uses server default
  (kiln #1). Explicit `null` or `""` uses base for that request without
  changing default. Explicit name uses it only for that request.

## Receipts replace log scraping

Every trainer (`cuda_grpo_ablation`, `cuda_sft_file`, `cuda_opd_remote`)
writes `train_receipt.json` next to the adapter (kiln #8). Schema documented
in `docs/TRAIN_RECEIPT_SCHEMA.md`. Key fields:

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
  "base_adapter_path": null,
  "base_adapter_sha256": null,
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
  "lora_delta_norm_summary": {
    "q_proj.lora_A": 0.014,
    "q_proj.lora_B": 0.022,
    "...": "..."
  },
  "grad_norm_min_mean_max": [1.1e-4, 3.2e-3, 7.1e-2],
  "adapter_manifest_path": "/workspace/adapters/<name>/adapter_manifest.json"
}
```

Cap scripts read this directly; no more grep'ing `train.log`.

## Eval the kiln way

`kiln eval-adapter` (kiln #33) is the canonical eval driver. Standard call
from `capability.oracle.sh`:

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

What it does:

- Runs base and adapter in **paired** mode across N seeds (default 3).
- Calls the cap's `rubric.py::score_one` per response. Aggregates per-task and overall.
- Writes a stable `eval_summary.json`:

```json
{
  "kiln_commit": "<sha>",
  "adapter": "...",
  "adapter_manifest_sha256": "...",
  "tasks_sha256": "...",
  "seeds": [1, 2, 3],
  "n_tasks": 24,
  "rubric_version": "v1",
  "mean_composite": 0.87,
  "composite_stdev": 0.012,
  "composite_delta": 0.04,
  "sigma_warning": null,
  "sub_scores_mean": { "outcome": 0.91, "...": "..." },
  "sub_scores_stdev": { "outcome": 0.02, "...": "..." },
  "wall_clock_s": 184,
  "verdict": "positive" | "null" | "negative" | "inconclusive",
  "status": "kept" | "kept-with-caveat" | "ablation" | ...
}
```

- Records adapter and config hashes — if you ran on stale weights, the summary will say so.
- Emits a `sigma_warning` field when `composite_stdev` is comparable to lift.

## Multi-seed eval is the rule, not the exception

Round 1 burned cycles on single-seed "wins" that didn't reproduce
(pi-doctest iter-2 +11pp single-seed → +4.2pp 3-seed). `kiln eval-adapter`
defaults to `--seeds 3`. Always run ≥3 seeds before declaring a positive iter.

Round-1 evidence: pi-doctest +9.4pp single-seed → +4.2pp 3-seed mean.
That recipe was still real; the headline was just wrong. The reproducible
recipe needs σ-bounded reporting.

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

## Diagnostics ladder for "iter didn't move"

If `composite_delta` ≈ 0 after a real training step:

1. **Was the adapter actually loaded?** Check `verify.behavioral == true`
   in `$LOG_DIR/verify.json`. If false, `kiln adapter verify` would have
   exited non-zero — re-check `run_iter.sh` logs.
2. **Did ECHO fire?** Check `train_receipt.json::echo_metrics`. If
   `env_ce_steps_observed == 0`, ECHO didn't run; mask or data shape
   was wrong.
3. **Did weights move?** Check `lora_delta_norm_summary`. All near-zero
   means no gradient signal.
4. **Did logits change but text didn't?** That's a determinism issue
   (sampling, temperature, kv-cache state). The `--adapter-smoke-test`
   (kiln #19) reports `logit_delta_summary` separately from generated text.
5. **Is reward saturated?** Receipt warns via `reward_saturation_warning`
   when mean reward > threshold and variance is low. Try
   `--no-policy-loss` (ECHO-only) or harder eval tasks.
6. **Is the rubric rewarding the wrong thing?** If sub-scores moved
   but composite didn't, check the weight matrix in `rubric.py`.

## Verifier-free / no-policy-loss recipes

For caps following paper §5.5 (e.g. `pi-script-fixup`):

```bash
ECHO_LAMBDA=0.05 BASE_ADAPTER=<best-iter-N> \
  ./run_iter.sh h-vf-adaptation
# Inside run_iter.sh, --no-policy-loss is set when ECHO_LAMBDA > 0 and
# config.training_phase1_defaults.loss.no_policy_loss = true.
```

The trainer asserts the policy-loss path is zeroed but env-CE training
still runs. `train_receipt.echo_metrics` should show env CE dropping.

## Long-context caps (pi-compaction)

Round-1 hit the long-context training wall: 32K-64K sequences with
near-byte-identical adapters. Round 2 ships:

- long-context bench suite at 8K/16K/32K/64K (kiln #25)
- per-phase progress logging (kiln #26) — no more "is it hung?"
- byte-identical-adapter diagnostic (kiln #27)
- testable warning-prefix masking (kiln #28)

`pi-compaction/run_iter.sh` inherits these; trust the receipts and
adapter-verify output rather than re-reading text.

## OPD caps

OPD (#37) is now first-class in `cuda_opd_remote`. Caps under `opd/`:

- `code-fence-language-fidelity`
- `code-symbol-extraction`
- `diff-patch-fluency`
- `faithful-code-summarization`
- `tool-call-arg-fidelity`
- `transcript-compaction`

Standard call (in `run_iter.sh`):

```bash
cuda_opd_remote \
  --prompts prompts/train.jsonl \
  --model /workspace/Qwen3.5-4B \
  --teacher-url http://localhost:8002 \
  --teacher-name qwen3.6-27b-awq \
  --output /tmp/.../adapter \
  --adapter <name> \
  --rank 16 --alpha 32 --lr 1e-4 --epochs 6 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name <name>
```

OPD `train_receipt.json` carries `n_prompts`, `effective_steps` (the
true training step count after EOS-skip filtering), and `teacher_calls_made`.
The round-1 `code-symbol-extraction` finding (97% EOS skip rate) is now
visible in the receipt; you don't have to grep for it.

## SFT caps

SFT caps (`sft/json-schema-adherence`, `sft/math-broad`, `sft/python-algo`)
use `cuda_sft_file`. Standard call:

```bash
cuda_sft_file \
  --data datasets/train.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output /tmp/.../adapter \
  --adapter <name> \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 \
  --dataset-cap 128 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name <name>
```

SFT caps also have `capability.anchor.sh` — a regression watch on a
non-target domain to catch stylistic clobber. `run_iter.sh` runs it
after the target eval.

## Per-cap rubric design discipline

Round 1 hit "rubric too lax" three times. The discipline:

1. **Write the adversarial design (§0) section first**, before
   `rubric.py`. Name ≥3 cheats that would score 1.0 without doing the
   capability. Design each mitigation into the rubric.
2. **Make outcome a hard floor**, not just a weighted component. Patterns:
   `composite = outcome * (w1*sub1 + w2*sub2 + ... + base)` — zero
   outcome → zero composite no matter how clean the process.
3. **Calibrate.** Write 5-10 known-good and 5-10 known-bad rollouts by
   hand into `calibration/{good,bad}.jsonl`. Run `python rubric_sanity.py`
   and confirm clean separation. If they don't separate, the rubric is
   broken.
4. **Check the baseline distribution.** If baseline composite > 0.95
   over your eval set, you're in the "rubric too lax" zone; tighten
   sub-scores before training. Round-2 trainer warns on this directly
   via `reward_saturation_warning`.

## Iter naming convention

`<family>-<descriptor>`. Examples:

- `baseline` — iter 0, no adapter.
- `h1-default-recipe` — first training iter with default Phase 1 GRPO.
- `h2-lower-lr` — single-knob variant of H1.
- `h3-warm-best` — chain training from the previous best adapter.
- `h12-strong-signal-only` — strong-signal filter applied.
- `h-vf-adaptation` — verifier-free `--no-policy-loss` adaptation.

The `family` field in `capability.jsonl` rolls up sibling iters; the
`slug` is unique. Slug appears in adapter name (`<cap>-<slug>`) so
`kiln adapter verify` and `capability.oracle.sh` can address it by name.

## When to stop iterating

Round 1 stopped some caps too early and some too late. Stop when:

- 3-seed mean improvement is within σ of base; sub-scores are saturated
  or stable; `reward_saturation_warning` is firing → switch to OPD or
  harder eval set.
- Receipt shows ECHO fully saturated (env CE delta near zero) AND
  policy loss isn't moving → the cap's headroom is exhausted at this
  rank.
- Multiple consecutive iters within σ in both directions → declare
  closeout.

A clean closeout is a row in `capability.jsonl` with `status: "closeout"`
and a final hypothesis verdict in `notes:`. The cap's `capability.md`
gets a "## Closeout" section summarizing winning recipe and remaining
risks.

## Reproducibility checklist

Before declaring a closeout, the cap should pass:

- [ ] 3-seed eval mean Δ > 2σ of base
- [ ] `train_receipt.json` exists and `status == "ok"`
- [ ] `adapter_manifest.json` exists; `kiln adapter restore <manifest>`
      round-trips with matching SHA256
- [ ] `kiln adapter verify <name>` exits 0 (`loadable` + `behavioral`)
- [ ] `calibration/{good,bad}.jsonl` rubric sanity passes
- [ ] No `sigma_warning` in `eval_summary.json`
- [ ] Iter row in `capability.jsonl` carries `kiln_commit`,
      `adapter_manifest`, and `train_receipt` paths

## Common questions

**Q: How do I add a new hypothesis?**
A: Add `H<N>: <one-sentence claim>` to the `## Hypotheses` section of
`capability.md`. Run `./run_iter.sh h<N>-<descriptor>`. The verdict
auto-appears in `capability.jsonl`.

**Q: How do I chain training (use a previous adapter as base)?**
A: `BASE_ADAPTER=<prev-name> ./run_iter.sh h-chain-prev`. Kiln #6
validates rank + target-module compatibility before optimizer setup.

**Q: How do I back up an adapter to B2?**
A: Don't write a custom backup script. `adapter_manifest.json` (kiln #36)
is the canonical archive shape. Copy the adapter dir to B2 and store
the manifest path. `kiln adapter restore <manifest>` round-trips it.

**Q: How do I add a new cap?**
A: Create `capabilities/<paradigm>/<slug>/` with the canonical files
from [`LAYOUT.md`](LAYOUT.md). Reference `pi-doctest` (agentic) /
`opd/code-symbol-extraction` (OPD) / `sft/math-broad` (SFT) for shape.

**Q: A round-1 cap is broken; should I read its archive?**
A: Only if the round-1 writeup explicitly flags a behavior the round-2
rubric needs to handle. Otherwise treat round-2 as a fresh start and
trust the kiln receipts and `capability.md` contract.

## Round 2 priority ranking

If you have to pick caps to run first (limited pod budget), this is
the recommended order:

### Tier 1 — High-leverage, ready to run

1. **`opd/code-symbol-extraction`** — the OPD canary. Round-1 bug
   diagnosed and fixed; this cap should jump from 7 effective steps to
   30+ and composite from 0.937 → 0.96+. Cheap, fast, validates the
   OPD trainer fix unblocks the other 5 OPD caps.
2. **`agentic-grpo/pi-doctest`** with hidden-tests sub-score — round-1
   winner +4.2pp 3-seed; hidden-tests fix mitigates the §0 A1 cheat and
   should give another 2-3pp on a harder eval.
3. **`agentic-grpo/pi-terminal-bench-lite`** — round-1 validated the
   plumbing; round-2 produces the paper-scale headline number.

### Tier 2 — New caps with high estimated headroom

4. **`agentic-grpo/pi-error-recovery`** — baseline ~0.40, headroom
   ~0.60. Process behavior, strong ECHO signal.
5. **`agentic-grpo/pi-incremental-progress`** — baseline ~0.50,
   headroom ~0.50. Foundational habit that composes well across caps.
6. **`agentic-grpo/pi-context-aware-edits`** — baseline ~0.45,
   headroom ~0.55. Style-consistency win is highly visible.
7. **`agentic-grpo/pi-search-then-read`** — baseline ~0.40, headroom
   ~0.60. Reduces context burn; composes naturally with
   `pi-code-search`.

### Tier 3 — Existing wins to extend

8. **`agentic-grpo/pi-code-comprehension`** — round-1 big win; OPD
   format polish + cross-file generalization eval.
9. **`agentic-grpo/pi-faithful-completion`** — chain training from
   iter-50 best.
10. **`agentic-grpo/pi-precondition-check`** — round-1 rank-1; needs
    the strengthened evidence-based rubric.

### Tier 4 — Reshapes (re-test the new gates)

11. **`agentic-grpo/pi-diff-patch-apply`** — v2 multiplicative-gate
    rubric; will the +12.5pp format gain now show as composite uplift?
12. **`agentic-grpo/pi-failure-triage`** — same.

### Tier 5 — Dependency-gated

13. **`agentic-grpo/pi-compaction`** — first prove kiln #25 long-context
    bench moves weights at 32K; then switch this cap to OPD.
14. **`agentic-grpo/pi-script-fixup`** — needs strong base adapter from
    pi-terminal-bench-lite Phase 2.

### Tier 6 — Eval-only

15. **`integration/cross-cap-coherence/`** — run after every Tier 2-3
    iter to surface sibling regressions.
16. **`agentic-grpo/pi-tool-call-efficiency`** — run periodically
    against the suite of installed adapters.

### Tier 7 — Lower-priority but still in queue

17. **`opd/code-fence-language-fidelity`** + 4 more OPD caps —
    run sequentially after the canary lands.
18. **`sft/*`** — standardize anchor suites; multi-seed eval; coordinate
    corpus with `pi-doctest` to avoid leak.
19. **`agentic-grpo/pi-shell-hygiene`**, **`pi-test-interpretation`**,
    **`pi-source-mod-workflow`** — scaffolds; the §0 + corpus design
    work is non-trivial.

## Diagnostic checklist for "iter didn't move" (round-2 version)

If `composite_delta` ≈ 0 after a training step:

1. **Did rubric_sanity gate fire?** Check `$LOG_DIR/rubric_sanity.log`.
   If calibration didn't pass, the rubric itself is broken — fix that
   first.
2. **Was the adapter actually loaded?** `verify.json::behavioral == true`.
3. **Did ECHO fire?** `train_receipt.json::echo_metrics::env_ce_steps_observed > 0`.
4. **Did weights move?** `lora_delta_norm_summary` not all near-zero.
5. **Is reward saturated?** Receipt `reward_saturation_warning` —
   switch to `--no-policy-loss` (ECHO-only) or use a multiplicative
   format gate.
6. **Is the rubric rewarding the wrong thing?** If sub-scores moved
   but composite didn't, check the weight matrix in `rubric.py` —
   consider whether multiplicative gates are needed.
7. **Cross-cap regression?** Run `integration/cross-cap-coherence/`
   against the adapter — sometimes the iter "didn't move" *this cap*
   but moved a sibling cap heavily, indicating skill-clobber.
