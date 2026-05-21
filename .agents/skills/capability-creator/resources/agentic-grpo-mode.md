# Agentic-GRPO mode — irreducible lore

Read this when [`capabilities/METHODS.md`](../../../../capabilities/METHODS.md)
Rule A fires (multi-turn tool-calling task). Plain-GRPO lore in
`grpo-mode.md` mostly still applies; this file covers what's *only*
agentic-specific.

The authoritative agentic content lives in
[`capabilities/lib/agentic-grpo-notes.md`](../../../../capabilities/lib/agentic-grpo-notes.md).
This file is a per-skill pointer + the agentic-specific failure modes
that aren't in METHODS.md / PIPELINE.md / NEXT_ROUND.md.

## When agentic-GRPO is the right tool

- Task is **multi-turn tool-calling**: the model must use tools (read files,
  run commands, edit files, search) over multiple turns to reach a goal.
- This OVERRIDES every other rule. Even if SFT bootstrap or OPD polish are
  also needed, the *primary trainer* is agentic-GRPO with ECHO.

## Why ECHO is the default

Plain GRPO learns from policy gradient on **action tokens only**. Tool-result
("observation") tokens get masked out. On multi-turn rollouts the model has
to predict its next action *conditioned on what the environment said in
response to the last action* — without modeling environment tokens, this
signal is lost.

ECHO adds a small env-token cross-entropy term (λ default 0.05) so the model
learns to predict its environment, not just its own action text.

ECHO is on by default in `LossConfig::default()`. Override with
`--echo-lambda <f>`; disable with `--no-echo`. Every agentic-GRPO stage
should keep ECHO on unless `pipeline.md::stage_transition_rationale`
documents a reason.

## ECHO defaults (kiln-train::LossConfig::default)

| Knob | Default | Where to override |
|---|---|---|
| `echo.lambda` | 0.05 | `--echo-lambda` |
| `echo.env_mask_mode` | `env_only` | `--echo-env-mask-mode` |
| `echo.warning_filter` | `true` | config only |
| `policy.kl_coeff` | 0.1 | `--kl-coeff` |
| `policy.clip_epsilon` | 0.20 | `--clip-epsilon` |
| `dynamic_sampling` | `true` | `--dynamic-sampling` |
| `lr` | 1e-5 | `--lr` |
| `rank` / `alpha` | 16 / 32 | `--rank` / `--alpha` |
| `seed` | 3141592653 | `--seed` |
| `filter_var_min` | unset | `--filter-var-min 0.05` |
| `adapter_smoke_test` | `true` | `--no-adapter-smoke-test` |

ECHO λ productive range (paper §3.3): 0.01-0.05. Round-1 evidence:
- λ=0.05 default works on most caps
- λ=0.075 lifted `pi-code-comprehension` (+12.9pp)
- λ=0.10 often hurts; do not push above 0.075 without paired ECHO-off ablation

## Verifier-free mode (`--no-policy-loss`)

For paper §5.5 adaptation recipes (no policy loss, only ECHO on env
tokens):

```bash
cuda_grpo_ablation \
  --no-policy-loss \
  --echo-lambda 0.05 \
  --base-adapter <best-prior-stage-adapter> \
  --data ... --model ... --output ... --adapter ...
```

The trainer asserts the policy-loss path is zeroed but env-CE training
still runs. Receipt should show `echo_metrics.env_token_ce_*` dropping
while `groups_trained > 0` AND `policy_loss == 0`.

**When to use:**
- After a policy-gradient stage saturates and ECHO env-CE still has headroom
- METHODS.md Rule G (saturated reward, hard_eval headroom)
- `caps/pi-script-fixup/` is the reference cap

## Pi-side trajectory capture

All agentic stages SHOULD use:
- `kiln trajectory inspect <jsonl>` (kiln #10) to validate pi sessions
- `kiln_train::pi_trajectory` (kiln #11) for the canonical Rust normalizer
- `capabilities/lib/pi_trajectory.py` as a Python compat shim

**Do not re-implement pi-session rendering** in each cap's `rollout.py`.
Either shell out to `kiln trajectory inspect <session.jsonl> --json` for
the segment list, or import `lib/pi_trajectory.py` in-process.

## Pi-session schema quirks (hard-won, round-1)

- Pi tool-call arguments under `input`, not `arguments`
- Pi 0.75.1 emits role `tool`; Pi 0.75.3 can emit `toolResult`; both
  normalize to canonical `tool`
- Pi session events use `message` singular in current JSONL, not the
  top-level shape older parsers assumed
- Warning prefixes get `warning_prefix_len` so the env_mask layer skips
  them (kiln #28 made this testable)

New schema variations → add fixture + test in `lib/test_pi_trajectory.py`;
do not work around ad hoc.

## Pi-smoke (mandatory before first agentic iter)

Before iter 1 of any agentic cap, run:

```bash
bash $SKILL/templates/pi_smoke.sh
```

What it verifies (template includes):

1. `/usr/bin/pi` is on PATH
2. Kiln is serving the base model on `:8420`
3. Pi is configured against kiln (`~/.pi/config.json`)
4. Headless pi session produces JSONL
5. Tool-call session: pi can write a file and read it back
6. Two pi instances in parallel don't interfere
7. `kiln trajectory inspect` parses the session JSONL into segments
   with nonzero action_mask AND nonzero env_mask

If pi-smoke fails at any step, debug *before* burning GPU time on a cap.

## Agentic-specific failure modes

### Env_mask zero (ECHO doesn't fire)

`train_receipt.json::echo_metrics::env_ce_steps_observed == 0` means the
env_mask was empty. The env tokens didn't make it to the trajectory.

**Causes:**
- Pi session had no tool segments (single-turn or all-assistant)
- Warning prefix consumed all tokens before tool result starts
- Schema parser failed silently (check `kiln trajectory inspect --json`)

**Mitigation:**
- Verify trajectory shape with `kiln trajectory inspect <jsonl> --json`
- Confirm warning_prefix_len isn't bleeding into env content

### Warning prefix bleed

System/user warnings leak into env_mask, polluting ECHO with non-environment
text. Round-2 kiln #28 made this testable.

**Mitigation:** `--echo-warning-filter true` (default). Receipt shows
`warning_filter_masked_bytes` > 0 if it caught anything.

### Multi-turn rollout budget blowup

Pi runs with `--max-turns 8` and `--max-tokens-per-turn 1024` by default.
A pathological loop (model tries the same failed action forever) blows
the budget without making progress.

**Mitigation:**
- Set `rollout.max_wall_clock_s 120` in capability.config.json
- `pi-tool-call-efficiency` transfer eval surfaces this across caps
- Watch for zero-progress rollouts in iter receipts

### Stale pi sessions

Pi caches old session JSONLs in `~/.pi/agent/sessions/`. Iter-to-iter, the
parser may read a stale session if the path collides.

**Mitigation:** rollout.py writes to a per-iter sandbox; `run_stage.sh`
sets `--sandbox-root /tmp/<cap>-stages/iter-<N>/`.

### Pi version drift

Pi 0.75.1 vs 0.75.3 emit different role names. `capability.config.json::
methods.agentic-grpo.rollout.pi_bin` records the absolute path; pin to
`/usr/bin/pi` to avoid PATH surprises.

## Adversarial design (§0) for agentic caps

Round-1 hit "rubric too lax" three times on agentic caps with single-component
`outcome` rubrics. The v1 multi-component pattern is now the established
default:

```
composite = outcome × (efficiency · w1 + verify · w2 + format · w3 + base)
```

Round-3 caps add multiplicative format gates where format matters
independently:

```
composite = outcome × format × (process_sub_scores + base)
```

`caps/pi-doctest/capability.md` §0 is the worked example for agentic rubric design.

## Receipt fields specific to agentic-GRPO

In addition to GRPO fields (groups_*, reward_*, kl_*):

- `echo_enabled`, `echo_lambda`, `echo_env_mask_mode`, `echo_warning_filter`
- `no_policy_loss` (bool)
- `action_token_count`, `env_token_count`, `context_token_count`
- `warning_filter_masked_bytes`
- `echo_metrics.env_token_ce_initial`, `env_token_ce_final`,
  `env_ce_steps_observed`

## Stage transitions specific to agentic-GRPO

- **agentic-GRPO → agentic-GRPO with --no-policy-loss:** policy saturated,
  ECHO env-CE still has headroom (METHODS.md Rule G adaptation).
- **agentic-GRPO chained with SFT or OPD:** if format/process need bootstrap,
  the agentic-GRPO stage usually comes LAST after SFT/OPD have stabilized
  the inputs.

## References

- [`capabilities/lib/agentic-grpo-notes.md`](../../../../capabilities/lib/agentic-grpo-notes.md) — full ECHO + pi notes (canonical)
- `caps/pi-doctest/` — most mature agentic reference; v1 rubric pattern
- `caps/pi-terminal-bench-lite/` — multi-turn paper-track + verifier-free
- `caps/pi-script-fixup/` — `--no-policy-loss` reference
- `caps/pi-faithful-completion/` — multi-stage pilot for round 3
- `docs/plans/echo-integration-plan.md` — ECHO design + masking layer
- `crates/kiln-train/src/echo.rs` — the loss term
- `crates/kiln-train/src/trajectory_mask.rs` — the masking layer
