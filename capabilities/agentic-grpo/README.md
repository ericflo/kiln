# `agentic-grpo/` — capability scaffolds for the Pi coding agent

This bucket holds capabilities where a **multi-turn agentic loop** is the
unit of work. Each cap trains a narrow coding-agent skill using
`cuda_grpo_ablation` with **ECHO on by default**, on rollouts gathered
through `pi` running against a kiln server (`kiln serve` on
`http://localhost:8420`).

For the uniform file layout every cap follows, read
[`../LAYOUT.md`](../LAYOUT.md) **first**.

## Why agentic-GRPO (and why ECHO is the default)

GRPO without ECHO learns from the policy-gradient on action tokens only.
That works on single-turn tasks where the reward is a clean function of
the final assistant message. On multi-turn tool-calling rollouts — which
is what coding agents actually do — the model has to predict its *next*
action conditioned on what the environment said in response to the
*last* action. Without ECHO, the auto-regressive loss on tool-result
("observation") tokens is silently masked out and the model never learns
to model its own environment.

ECHO adds a small env-token cross-entropy term (λ default `0.05`, paper
§3.3 productive range 0.01–0.05) computed on the same forward pass.
Empirically this:

- closes the train/test gap on multi-turn agentic tasks,
- enables verifier-free adaptation (paper §5.5, see `pi-script-fixup`),
- composes cleanly with the existing GRPO surrogate and KL terms.

ECHO is **on by default** in `LossConfig::default()` (see
`docs/plans/echo-integration-plan.md` §3.4). The CLI surface is
`--echo-lambda <f64>` to override and `--no-echo` to disable. Every cap
in this bucket should keep ECHO on unless the cap's `capability.md`
documents an explicit reason to turn it off.

## ECHO defaults you can rely on

From `kiln-train::LossConfig::default()` and
`crates/kiln-train/examples/cuda_grpo_ablation.rs`:

| Knob | Default | Where to override |
| --- | --- | --- |
| `echo.lambda` | `0.05` | `--echo-lambda` or `capability.config.json::training_phase1_defaults.loss.echo.lambda` |
| `echo.env_mask_mode` | `env_only` | `--echo-env-mask-mode` |
| `echo.warning_filter` | `true` | config only |
| `policy.kl_coeff` | `0.1` | `--kl-coeff` |
| `policy.clip_epsilon` | `0.20` | `--clip-epsilon` |
| `dynamic_sampling` | `true` | `--dynamic-sampling` |
| `lr` | `1e-5` | `--lr` |
| `rank` / `alpha` | `16` / `32` | `--rank` / `--alpha` |
| `seed` | `3141592653` | `--seed` |
| `filter_var_min` | (unset) | `--filter-var-min 0.05` for strong-signal filtering (kiln #22) |
| `adapter_smoke_test` | `true` | `--no-adapter-smoke-test` to disable (kiln #19) |

**Verifier-free mode.** For paper §5.5 adaptation recipes (no policy loss,
only ECHO on env tokens), set `training_phase1_defaults.loss.no_policy_loss = true`
in the config and pass `--no-policy-loss` to the trainer.
`pi-script-fixup` is the reference cap for this mode.

## Pi-side trajectory capture

All caps SHOULD now use **`kiln trajectory inspect`** (kiln #10) to
validate pi sessions and `kiln_train::pi_trajectory` (kiln #11) for the
canonical Rust normalizer. The Python parser at
[`lib/pi_trajectory.py`](lib/pi_trajectory.py) is preserved as a
backwards-compat shim and a round-trip validator. See
[`lib/README.md`](lib/README.md).

**Do not re-implement pi-session rendering in each cap's `rollout.py`.**
For new caps, either:

- shell out to `kiln trajectory inspect <session.jsonl> --json` for the
  segment list, or
- import `lib/pi_trajectory.py` if you need to do the rendering in-process.

## What the layout assumes

Every cap dir under `agentic-grpo/` follows the structure in
[`../LAYOUT.md`](../LAYOUT.md). Round-2-specific changes:

- `run_iter.sh` calls `cuda_grpo_ablation --dry-run` (kiln #9) BEFORE any
  GPU training step.
- `run_iter.sh` calls `cuda_grpo_ablation --adapter-smoke-test
  --install-adapter-dir --install-adapter-name` to atomically install the
  adapter into the registry (kiln #5).
- `run_iter.sh` calls `kiln adapter verify` after install (kiln #4).
- `capability.oracle.sh` is a thin wrapper around `kiln eval-adapter`
  (kiln #33).
- Strong-signal filtering uses `--filter-var-min` (kiln #22). Do not
  re-implement.
- Iter rows are derived from the trainer-owned `train_receipt.json` (kiln #8)
  and the eval-adapter `eval_summary.json`, not from log scraping.

## Standard workflow per cap (the "first iter" recipe)

1. **Read the cap's `capability.md`.** It tells you the task shape,
   rubric, headroom estimate, and adversarial cheats to design against.
2. **Build the corpus.** `python build_corpus.py` produces
   `datasets/train.tasks.jsonl` and `datasets/eval.tasks.jsonl`. The eval
   file is gitignored — once it exists, do not read it from inside the
   training loop.
3. **Calibrate the rubric.** Write 5–10 known-good and 5–10 known-bad
   trajectories by hand into `calibration/{good,bad}.jsonl`, then run
   `python rubric_sanity.py`. The good set should score above the bad
   set with a clean separation.
4. **Iter 0 baseline.** Run `./capability.oracle.sh` (no adapter) to get
   the base model composite. Compute headroom = `1 - composite`. Record
   the row with `slug=baseline-v0` in `capability.jsonl`.
5. **Iter 1 training.** `./run_iter.sh h1-default-recipe` runs the full
   pipeline. Re-score eval; if mean composite moved beyond
   group-variance noise, you have a positive iter. **Always run for at
   least 3 seeds before claiming a win** — kiln-eval-adapter does this
   by default (`SEEDS=3`).
6. **Inspect the row.** `tail -1 capability.jsonl | jq` shows the
   verdict, sub-scores, adapter manifest path, and verify status.

If iter 1 is null (within noise), check the receipt for
`echo_metrics.env_token_ce_holdout` — if that *didn't* drop, ECHO isn't
doing anything and the headroom is elsewhere. If it did drop but
composite didn't move, the rubric is rewarding the wrong thing.

## Adversarial design (§0) — the cheat-resistance discipline

Every cap's `capability.md` has an `## Adversarial design (§0)` section.
**Fill it before writing rubric.py.** Ask: "what's the cheapest path to
score 1.0 without doing the capability?" and design the rubric to make
each cheat score ≤ 0. The single-component `outcome` rubric has hit
the §0 "rubric too lax" zone three times in round 1; the v1
multi-component rubric (outcome × (efficiency · w1 + verify · w2 +
format · w3 + base)) is the established pattern. See pi-doctest §0 for
the worked example.

## Base model + infra

- **Model:** Qwen3.5-4B served by kiln on `http://localhost:8420`.
  Round-2 kiln has a first-class Qwen3.5-4B defaults profile (kiln #31).
- **Pi:** `/usr/bin/pi` with model id `qwen-3.5-4b-kiln`. Session
  JSONLs land at `~/.pi/agent/sessions/<workdir-encoded>/<uuid>.jsonl`.
- **GPU:** RunPod A6000 on-demand (never spot — see
  `kiln-skill/SKILL.md` "money-burning anti-patterns"). For pod
  lifecycle use `ce kiln-pod-acquire` / `ce kiln-pod-release`.

## Round 1 evidence preserved

[`CONSOLIDATED_REPORT.md`](CONSOLIDATED_REPORT.md) summarizes round-1
findings across all 14 caps. The 40 kiln improvements in
[`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) are the
backlog that round-1 surfaced; all are now completed and round-2 caps
depend on them. Per-cap round-1 artifacts are preserved under each
`pi-<cap>/archive/`.

## Cross-references

- [`../LAYOUT.md`](../LAYOUT.md) — uniform cap-dir layout
- [`KILN_IMPROVEMENT_ISSUES.md`](KILN_IMPROVEMENT_ISSUES.md) — round-2
  kiln features (all done)
- [`CONSOLIDATED_REPORT.md`](CONSOLIDATED_REPORT.md) — round-1 lessons
- `docs/plans/echo-integration-plan.md` — the ECHO design + masking layer
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md`
- `crates/kiln-train/src/echo.rs` — the loss term
- `crates/kiln-train/src/trajectory_mask.rs` — the masking layer
- `crates/kiln-train/src/pi_trajectory.rs` — kiln's pi-session normalizer
- [`lib/pi_trajectory.py`](lib/pi_trajectory.py) — Python compat shim
- `pi-doctest/capability.md` — the most mature reference cap; its v1
  multi-component rubric is the established pattern.
