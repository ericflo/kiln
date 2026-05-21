# ECHO Guide

> *Make every multi-turn rollout teach two things: how to act, and how the environment responds.*

ECHO (**E**nvironment **C**ross-entropy **H**ybrid **O**bjective; Shrivastava, Awadallah, Papailiopoulos — MSR AI Frontiers, 2026) adds a length-normalized cross-entropy loss on environment-observation tokens to kiln's standard GRPO policy-gradient loss on action tokens. The math:

$$\mathcal{L}_{\text{ECHO}} = \mathcal{L}_{\text{GRPO}}(\text{actions}) + \lambda \cdot \mathcal{L}_{\text{Env}}(\text{observations})$$

The headline result on the paper's TerminalBench-2.0 benchmark: **pass@1 nearly doubles** at 8B (2.7% → 5.2%) and 14B (5.2% → 10.8%) — at **zero extra forward-pass cost** because the env tokens were already in the rollout context.

This guide is the operational companion to the integration plan
([`docs/plans/echo-integration-plan.md`](plans/echo-integration-plan.md)) and the paper archives ([`docs/papers/echo/echo_paper.md`](papers/echo/echo_paper.md)).

## Quick decisions

**Is ECHO on by default?** Yes. `GrpoConfig::default()` ships with `loss.echo = Some(EchoConfig { lambda: 0.05, ... })` (paper §3.3 recommended). To opt out: pass `--no-echo` to `cuda_grpo_ablation` or set `loss.echo = None` in JSON.

**Do my legacy single-turn GRPO rollouts get ECHO?** No, but they get no harm either. For rollouts without a `trajectory` field, `env_mask` is empty → ECHO contributes exactly zero. The loss math is bit-identical to the pre-ECHO commit.

**Do I need to change my capability code?** No, if you're on the new shared lib. The shared `capabilities/agentic-grpo/lib/pi_trajectory.py` already emits the canonical trajectory schema; cap authors call `pi_trajectory.build_scored_rollout(...)` instead of hand-concatenating turns.

**Does ECHO work on all backends?** Yes. CUDA / CPU / Metal share `crates/kiln-train/src/trainer.rs` (both the uncheckpointed loss path and the checkpointed analytic-tail variant — ECHO is folded into both). Vulkan-native gets the same loss term via `crates/kiln-train/src/vk_train.rs::vk_recompute_grpo_train_step_with_state` (paper §3.1 `|O|` normalization preserved).

## How to use ECHO

### From the CLI

```bash
# Default — ECHO on at λ=0.05.
cuda_grpo_ablation \
    --data train.jsonl --model /workspace/Qwen3.5-4B \
    --output adapter --mode phase1

# Override λ.
cuda_grpo_ablation ... --echo-lambda 0.02

# Ablation — disable ECHO entirely.
cuda_grpo_ablation ... --no-echo

# Verifier-free env-only adaptation (paper §5.5).
cuda_grpo_ablation ... --no-policy-loss --echo-lambda 0.05
```

`--no-policy-loss` masks out the GRPO surrogate so only the ECHO env-CE drives gradients. Used for keeping a strong agent improving on tasks where no programmatic verifier is available.

### From the kiln-server HTTP API

```bash
# Canonical: POST /v1/train/agentic (alias of /v1/train/grpo).
curl -X POST http://localhost:8420/v1/train/agentic \
  -H 'content-type: application/json' \
  -d '{
    "agentic_groups": [
      {
        "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
        "rollouts": [
          {
            "text": "<TURN_BREAK-joined action text>",
            "reward": 1.0,
            "trajectory": [
              {"role":"assistant","content":"...","kind":"action"},
              {"role":"tool","content":"...","kind":"observation"},
              {"role":"assistant","content":"...","kind":"action"}
            ]
          }
        ]
      }
    ],
    "config": {
      "lora_rank": 16,
      "lora_alpha": 32,
      "loss": {
        "echo": {"lambda": 0.05, "env_mask_mode": "env_only", "warning_filter": true},
        "opd": null,
        "no_policy_loss": false
      }
    }
  }'
```

The legacy `/v1/train/grpo` endpoint accepts the same payload shape; both routes serve the same handler. Legacy clients posting `{ groups: [{messages, completions: [{text, reward}, ...]}] }` keep working — `completions` is a serde alias for `rollouts`, and `text`-only rollouts deserialize as one-segment Action trajectories.

### From environment variables

For ops / CI orchestration without editing the request payload:

| Variable | Effect |
| --- | --- |
| `KILN_ECHO_ENABLED=0` (or `false`/`no`) | disable ECHO globally |
| `KILN_ECHO_ENABLED=1` | enable ECHO with default knobs |
| `KILN_ECHO_LAMBDA=0.02` | override λ (auto-enables if previously off) |
| `KILN_ECHO_ENV_MASK_MODE=env_only` (default) \| `full_obs` | which env positions to mask |
| `KILN_ECHO_WARNING_FILTER=false` | include harness warning prefix in env_mask |

Env vars take **precedence over CLI flags** in `cuda_grpo_ablation` — CLI is for inline dev tweaks; env vars are the right tool for shell-scripted orchestration.

## How to read the diagnostics

The trainer writes ECHO-specific fields to `<adapter_dir>/receipt.json` (when receipt-writing lands; the schema is in place today):

```json
{
  "diagnostic_summary": {
    "final_loss": 0.43,
    "echo": {
      "lambda": 0.05,
      "env_ce_initial": 4.21,
      "env_ce_final": 0.83,
      "env_ce_drop_pct": 80.3,
      "lambda_effective_final": 0.07,
      "env_tokens_supervised": 24576,
      "dynamics_holdout_ce_initial": 3.96,
      "dynamics_holdout_ce_final": 1.12
    }
  }
}
```

- `env_ce_initial` / `env_ce_final` — paper §5.2's headline diagnostic on the training rollouts. ECHO is expected to drop env-CE *sharply* (paper Figure 3 shows ~75% drops); GRPO alone barely moves it.
- `env_ce_drop_pct` — direct comparison to paper §5.2. ≥30% drop is the Phase 2 validation gate.
- `lambda_effective_final` — `λ · L_envCE / L_GRPO` at end of training. Paper §3.3: ECHO auto-anneals as the model learns terminal structure, so this should *shrink* over the course of training (or at least stay bounded). If it grows, λ is too aggressive.
- `env_tokens_supervised` — total count of env-position log-prob gradients. Useful for the "did ECHO actually fire" smoke check; legacy single-turn rollouts have this = 0.
- `dynamics_holdout_ce_initial` / `dynamics_holdout_ce_final` — paper §5.2 *dynamics test*: env-CE on a teacher-generated held-out trajectory set. ECHO is expected to drop this comparably to `env_ce`, proving the model **generalized** the terminal dynamics rather than memorized the training rollouts. `pi-terminal-bench-lite/calibration/dynamics_holdout.py` populates these when the cap provides a teacher.

## When to turn ECHO off

Three cases where opting out (`--no-echo`) is the right move:

1. **Single-turn rollouts with no observations.** ECHO is mathematically a no-op (env_mask is empty), so the cost is zero — but the diagnostic noise might be confusing. Cleaner to flip it off.
2. **λ tuning ablations.** Run a paired `--no-echo` baseline so the ECHO contribution is visible as a delta.
3. **Tokenizer / chat-template debugging.** If you suspect the env-mask is locating the wrong tokens (test it on a fixture with `cargo test build_masks_against_real_qwen`), turn ECHO off until the mask is right.

## When to use `--no-policy-loss`

Paper §5.5 calls this **verifier-free env-only adaptation**: once a base policy is strong enough to produce informative rollouts, the env-CE alone can keep improving it on OOD tasks — *without* a verifier.

- **Pre-conditions:**
  - You already have a strong-but-not-saturated agentic adapter (e.g. from a Phase 2 cap).
  - Your rollouts pass a "clean tool-call" filter (no malformed XML, no parse errors).
  - The environment feedback is *informative* — Python tracebacks tightly coupled to executed code, NOT generic shell state.
- **Pre-flight:** sanity-check that `--no-echo` reaches near-zero loss on the same dataset. If ECHO can't beat that baseline, `--no-policy-loss` won't either.
- **Paper §5.5 target deltas** (on the strongest 8B Qwen3 ECHO checkpoint, 100 steps):
  - val100: +3.8 pp
  - ITD: +5.2 pp (filtered)
  - PyTerm: +10.0 pp (filtered)
  - TBLite: −3.9 pp (the negative result — orchestration-heavy tasks don't benefit because env feedback is too indirect)

See `capabilities/agentic-grpo/pi-script-fixup/` for the canonical kiln reproduction of this experiment.

## Composition with OPD

When the OPD branch rebases on top of ECHO, both contribute to the same `LossConfig`:

```rust
LossConfig {
    echo: Some(EchoConfig { lambda: 0.05, .. }),    // env-CE on observations
    opd:  Some(OpdAuxConfig { lambda: 0.10, .. }),  // reverse-KL on actions
    no_policy_loss: false,
}
```

Total loss: `L = L_policy + λ_echo · L_envCE + λ_opd · L_revKL`. The three terms target three different position sets:

| Term | Loss type | Active at | Source of supervision |
| --- | --- | --- | --- |
| `L_policy` | GRPO surrogate | action positions | outcome reward |
| `L_envCE` | cross-entropy | observation positions | environment (raw env tokens) |
| `L_revKL` | reverse-KL | action positions | teacher (top-K logprobs) |

This composition is what the integration plan §1 framing calls *"completing the agentic loop"* — every position in the rollout carries gradient.

## Capability author checklist

When writing a new agentic-GRPO cap, you basically don't need to do anything ECHO-specific. The defaults handle it:

1. **`rollout.py`:** import `pi_trajectory` from `capabilities/agentic-grpo/lib/` and call `pi_trajectory.build_scored_rollout(session_path, reward=...)`. Done — your JSONL now carries the canonical trajectory schema.
2. **`capability.config.json`:** leave `loss.echo` at its default (which is `null` if you omit, but the trainer's `GrpoConfig::default()` will fill in `Some(EchoConfig::default())` at deserialize time). Explicit:
   ```json
   "loss": { "echo": { "lambda": 0.05, "env_mask_mode": "env_only", "warning_filter": true } }
   ```
3. **`rubric.py`:** unchanged. ECHO operates on the loss term, not the reward.
4. **Iter logs:** when the trainer starts populating `receipt.json::echo`, your `capability.jsonl` should pull through `env_ce_drop_pct` so the iter table shows ECHO contribution at a glance.

## Further reading

- **Paper PDF:** [`docs/papers/echo/echo_paper.md`](papers/echo/echo_paper.md) (Markdown conversion of `github.com/microsoft/echo-rl/blob/main/echo.pdf`)
- **Blog post:** [`docs/papers/echo/echo_blog_post.md`](papers/echo/echo_blog_post.md) (Markdown of the X/Twitter thread)
- **Integration plan:** [`docs/plans/echo-integration-plan.md`](plans/echo-integration-plan.md) (~1750 lines, full design + risk register + acceptance tests)
- **Long-form companion:** [`docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md`](plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md) (the OPD-plan-style aspirational write-up; Phase 3)
- **Skill update:** [`.agents/skills/agentic-grpo-capability-creator/SKILL.md`](../.agents/skills/agentic-grpo-capability-creator/SKILL.md) §0 (cap authors)

## Versioning notes

ECHO landed via PR #1054 on branch `use-breakthrough-echo-grpo-technique-throughout`. Receipt schema version stayed at 1 (the `echo` field is an additive optional). Old payloads continue to deserialize byte-identically (`#[serde(alias)]` everywhere).
