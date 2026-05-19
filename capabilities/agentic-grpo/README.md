# `agentic-grpo/` — capability scaffolds for the Pi coding agent

This bucket holds capabilities where a **multi-turn agentic loop** is the
unit of work. Each cap trains a narrow coding-agent skill using
`cuda_grpo_ablation` (or the same loss math in `vk_train.rs`) with **ECHO
on by default**, on rollouts gathered through `pi` running against a kiln
server (`kiln serve` on `http://localhost:8420`).

If you're picking up one of the 10 scaffolds (`pi-precondition-check`
through `pi-faithful-completion`), read this file first — it explains
what every cap shares so the cap-specific `capability.md` can stay
tight.

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

## What each capability dir contains

Mandatory (committed):

```
pi-<name>/
├── capability.md             # the contract: goal, task shape, rubric, hypotheses
├── capability.config.json    # trainer + rollout defaults (ECHO-on)
├── capability.jsonl          # append-only iter log
├── rubric.py                 # composite reward function
├── task_scaffold.py          # task generator → datasets/*.tasks.jsonl
├── rollout.py                # pi-runner → rollout JSONL + scored grpo-train.jsonl
├── build_corpus.py           # rollouts → train/eval splits
├── capability.oracle.sh      # blind eval scoring for a given adapter
└── run_iter.sh               # full iter recipe (rollouts → train → eval)
```

Optional (committed):

```
hypotheses/<name>.md          # alternative experiments + verdicts
calibration/{good,bad}.jsonl  # rubric sanity fixtures (known-good vs known-bad)
manifest/<iter>.json          # reproducibility manifest per iter
```

Not committed (see `capabilities/.gitignore`):

```
datasets/eval.jsonl           # blind-eval firewall — the agent must not read this
adapters/                     # regenerable from the manifest
responses/                    # per-rollout intermediate artifacts
*.log                         # training/eval logs
```

The four scaffolded files (`capability.md`, `capability.config.json`,
`capability.jsonl`, and this README) are what each new cap in this PR
ships with. The agent picking up a cap creates the remaining files as
the first iter runs.

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

**Verifier-free mode.** If your cap is following the paper §5.5
adaptation recipe (no policy loss, only ECHO on env tokens), set
`training_phase1_defaults.loss.no_policy_loss = true` in the config and
pass `--no-policy-loss` to the trainer. `pi-script-fixup` is the
reference for this mode.

## Pi-side trajectory capture

All caps use `capabilities/agentic-grpo/lib/pi_trajectory.py` to parse
pi session JSONL into `(input_ids, action_mask, env_mask)` via the
canonical kiln Trajectory schema. **Do not re-implement the rendering
in each cap's `rollout.py`** — import from `lib/`. The schema lives in
`kiln-train::trajectory::ScoredRollout` and is documented in
`docs/plans/echo-integration-plan.md` §3.3.

## How hypotheses log to capability.jsonl

The `## Hypotheses` section in each cap's `capability.md` is **design
intent**: it lists the experiments worth running once the v0 rubric is
stable. The **canonical iter log** is `capability.jsonl` — one row per
iter, append-only, recording the hypothesis under test, the
hyperparameters, the adapter SHA (or null), the eval delta vs baseline,
and the verdict (`positive` / `null` / `negative` / `inconclusive`).

When you ship an iter that tested one of the `H_*` hypotheses, append a
row referencing the hypothesis slug. Do **not** edit the
`capability.md` hypothesis bullet retroactively — the log is the
source of truth. (See `capabilities/agentic-grpo/pi-doctest/capability.jsonl`
for the established row shape.)

## Standard workflow per cap (the "first iter" recipe)

1. **Read the cap's `capability.md`.** It tells you the task shape,
   rubric, headroom estimate, and adversarial cheats to design against.
2. **Build the corpus.** `python build_corpus.py` produces
   `datasets/train.tasks.jsonl` and `datasets/eval.jsonl`. The eval file
   is gitignored — once it exists, do not read it from inside the
   training loop.
3. **Calibrate the rubric.** Write 5–10 known-good and 5–10 known-bad
   trajectories by hand into `calibration/{good,bad}.jsonl`, then run
   `python rubric_sanity.py` (if your cap has one — start from
   `pi-compaction/rubric_sanity.py` as a template). The good set should
   score above the bad set with a clean separation; if it doesn't, the
   rubric is broken.
4. **Iter 0 baseline.** Run `./run_iter.sh 0` to gather rollouts with
   the base model and score them. Compute mean composite, group-variance
   stdev per sub-score, and wall-clock per rollout. Append a row to
   `capability.jsonl` with the headroom numbers.
5. **Iter 1 training.** `./run_iter.sh 1` runs `cuda_grpo_ablation` with
   ECHO on for `--max-groups <N>` steps and emits an adapter under
   `adapters/`. Re-score eval; if mean composite moved beyond
   group-variance noise, you have a positive iter.
6. **Append the verdict to `capability.jsonl`** with the adapter SHA, the
   manifest path, and the delta vs baseline.

If iter 1 is null (within noise), check the diagnostics receipt for
`env_token_ce_holdout` — if that *didn't* drop, ECHO isn't doing
anything and the headroom is elsewhere. If it did drop but composite
didn't move, the rubric is rewarding the wrong thing.

## Adversarial design (§0) — the cheat-resistance discipline

Every cap's `capability.md` has an `## Adversarial design (§0)` section.
**Fill it before writing rubric.py.** Ask: "what's the cheapest path to
score 1.0 without doing the capability?" and design the rubric to make
each cheat score ≤ 0. The single-component `outcome` rubric has hit
the §0 "rubric too lax" zone three times in this bucket; the v1
multi-component rubric (outcome × (efficiency · w1 + verify · w2 +
format · w3 + base)) is the established pattern. See pi-doctest §0 for
the worked example.

## Base model + infra

- **Model:** Qwen3.5-4B served by kiln on `http://localhost:8420`.
- **Pi:** `/usr/bin/pi` with model id `qwen-3.5-4b-kiln`. Session
  JSONLs land at `~/.pi/agent/sessions/<workdir-encoded>/<uuid>.jsonl`.
- **GPU:** RunPod A6000 on-demand (never spot — see
  `kiln-skill/SKILL.md` "money-burning anti-patterns"). For pod
  lifecycle use `ce kiln-pod-acquire` / `ce kiln-pod-release`.

## Cross-references

- `docs/plans/echo-integration-plan.md` — the ECHO design + masking layer
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md`
- `crates/kiln-train/src/echo.rs` — the loss term
- `crates/kiln-train/src/trajectory_mask.rs` — the masking layer
- `capabilities/agentic-grpo/lib/pi_trajectory.py` — pi → Trajectory
- `capabilities/agentic-grpo/pi-doctest/capability.md` — the most mature
  reference cap; its v1 multi-component rubric is the established
  pattern.
