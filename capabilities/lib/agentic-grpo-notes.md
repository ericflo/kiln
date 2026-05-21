# Agentic-GRPO notes — ECHO defaults and pi-rollout shape

This file holds the agentic-specific lore that used to live in
`capabilities/agentic-grpo/README.md`. Round 3 promoted methodology-agnostic
docs to `capabilities/` top-level; this remainder is the irreducible
agentic-GRPO content for stages that use the `agentic-grpo` method.

Read [`../METHODS.md`](../METHODS.md) §3.4 for when agentic-GRPO is the
recommended method. This doc tells you the *defaults* once it's chosen.

## Why agentic-GRPO (and why ECHO is the default)

GRPO without ECHO learns from the policy gradient on action tokens only.
That works on single-turn tasks where the reward is a clean function of
the final assistant message. On multi-turn tool-calling rollouts — which
is what coding agents actually do — the model has to predict its *next*
action conditioned on what the environment said in response to the *last*
action. Without ECHO, the auto-regressive loss on tool-result
("observation") tokens is silently masked out and the model never learns
to model its own environment.

ECHO adds a small env-token cross-entropy term (λ default `0.05`,
paper §3.3 productive range 0.01–0.05) computed on the same forward pass.
Empirically this:

- closes the train/test gap on multi-turn agentic tasks,
- enables verifier-free adaptation (paper §5.5, see `caps/pi-script-fixup`),
- composes cleanly with the existing GRPO surrogate and KL terms.

ECHO is **on by default** in `LossConfig::default()` (see
`docs/plans/echo-integration-plan.md` §3.4). The CLI surface is
`--echo-lambda <f64>` to override and `--no-echo` to disable. Every stage
using the `agentic-grpo` method should keep ECHO on unless the cap's
`pipeline.md::stage_transition_rationale` documents an explicit reason to
turn it off.

## ECHO defaults you can rely on

From `kiln-train::LossConfig::default()` and
`crates/kiln-train/examples/cuda_grpo_ablation.rs`:

| Knob | Default | Where to override |
| --- | --- | --- |
| `echo.lambda` | `0.05` | `--echo-lambda` or `capability.config.json::methods.agentic-grpo.defaults.loss.echo.lambda` |
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

### Verifier-free mode

For paper §5.5 adaptation recipes (no policy loss, only ECHO on env tokens),
set `methods.agentic-grpo.defaults.loss.no_policy_loss = true` in
`capability.config.json` and pass `--no-policy-loss` to the trainer.

`caps/pi-script-fixup/` is the reference cap for this mode. It usually
arrives as a *later stage* after a policy-gradient stage saturates and ECHO
env-CE still has headroom.

## Pi-side trajectory capture

All caps using agentic-GRPO SHOULD use **`kiln trajectory inspect`** (kiln #10)
to validate pi sessions and `kiln_train::pi_trajectory` (kiln #11) for the
canonical Rust normalizer. The Python parser at
[`pi_trajectory.py`](pi_trajectory.py) is a backwards-compat shim and a
round-trip validator. See [`README.md`](README.md) for the matrix of when
to use which.

**Do not re-implement pi-session rendering in each cap's `rollout.py`.**
For new caps, either:

- shell out to `kiln trajectory inspect <session.jsonl> --json` for the
  segment list, or
- import `pi_trajectory.py` if you need to do the rendering in-process.

### Pi-session schema quirks (hard-won)

- Pi tool-call arguments are under `input`, not `arguments`.
- Pi 0.75.1 emits role `tool`; Pi 0.75.3 can emit `toolResult`; kiln
  normalizes both to canonical `tool`.
- Pi session events use `message` singular in current observed JSONL,
  not the top-level shape older kiln parsers assumed.
- Warning prefixes (system banner / user notice) get `warning_prefix_len`
  metadata so the env_mask layer can skip them.

If a cap's `rollout.py` encounters new schema variation, add a fixture and
test in `lib/test_pi_trajectory.py` rather than working around it ad hoc.

## What the round-3 layout assumes per agentic-GRPO stage

Per [`../LAYOUT.md`](../LAYOUT.md), an agentic-GRPO stage's `run_stage.sh`
calls:

- `cuda_grpo_ablation --dry-run` (kiln #9) BEFORE any GPU training step.
- `cuda_grpo_ablation --adapter-smoke-test --install-adapter-dir
  --install-adapter-name` to atomically install the adapter into the
  registry (kiln #5).
- `kiln adapter verify` after install (kiln #4).
- `capability.oracle.sh` (thin wrapper around `kiln eval-adapter`, kiln #33).
- Strong-signal filtering uses `--filter-var-min` (kiln #22). Do not
  re-implement.
- The iter row is derived from the trainer-owned `train_receipt.json` (kiln #8)
  and the eval-adapter `eval_summary.json`, not from log scraping.

## Adversarial design (§0) — the cheat-resistance discipline

Every cap's `capability.md` has an `## Adversarial design (§0)` section.
**Fill it before writing rubric.py.** Ask: "what's the cheapest path to
score 1.0 without doing the capability?" and design the rubric to make
each cheat score ≤ 0. The single-component `outcome` rubric has hit the
§0 "rubric too lax" zone three times in round 1; the v1 multi-component
rubric (outcome × (efficiency · w1 + verify · w2 + format · w3 + base))
is the established pattern. See `caps/pi-doctest/capability.md` §0 for the
worked example.

## Base model + infra

- **Model:** Qwen3.5-4B served by kiln on `http://localhost:8420`.
  Round-2 kiln has a first-class Qwen3.5-4B defaults profile (kiln #31).
- **Pi:** `/usr/bin/pi` with model id `Qwen3.5-4B`. Session JSONLs land
  at `~/.pi/agent/sessions/<workdir-encoded>/<uuid>.jsonl`.
- **GPU:** RunPod A6000 on-demand (never spot — see
  `.agents/skills/capability-creator/SKILL.md` "money-burning anti-patterns").
  For pod lifecycle use `ce kiln-pod-acquire` / `ce kiln-pod-release`.

## Round-1 evidence preserved

[`../CONSOLIDATED_REPORT.md`](../CONSOLIDATED_REPORT.md) summarizes round-1
findings across all 14 caps. The 40 kiln improvements in
[`../KILN_IMPROVEMENT_ISSUES.md`](../KILN_IMPROVEMENT_ISSUES.md) are the
backlog that round-1 surfaced; all are now completed and round-3 caps
depend on them. Per-cap round-1 artifacts are preserved under each
`caps/<cap>/archive/`.

## Cross-references

- [`../LAYOUT.md`](../LAYOUT.md) — uniform cap-dir layout
- [`../METHODS.md`](../METHODS.md) — when to choose agentic-GRPO
- [`../PIPELINE.md`](../PIPELINE.md) — how agentic-GRPO chains with other methods
- [`../KILN_IMPROVEMENT_ISSUES.md`](../KILN_IMPROVEMENT_ISSUES.md) — kiln features
- [`../CONSOLIDATED_REPORT.md`](../CONSOLIDATED_REPORT.md) — round-1 lessons
- `docs/plans/echo-integration-plan.md` — the ECHO design + masking layer
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md`
- `crates/kiln-train/src/echo.rs` — the loss term
- `crates/kiln-train/src/trajectory_mask.rs` — the masking layer
- `crates/kiln-train/src/pi_trajectory.rs` — kiln's pi-session normalizer
- [`pi_trajectory.py`](pi_trajectory.py) — Python compat shim
- `caps/pi-doctest/capability.md` — most mature reference cap; v1
  multi-component rubric is the established pattern.
