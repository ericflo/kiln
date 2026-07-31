# ECHO Guide

ECHO adds an auxiliary learning signal to multi-turn GRPO and OPD
trajectories. The policy-gradient term teaches the model which actions earned
better outcomes; ECHO teaches it to predict the environment observations that
followed those actions.

The generated [Training and Agent Control Plane API
Schema](../contracts/kiln-control-plane-v1.schema.json) owns request fields,
defaults, aliases, and constraints. This guide explains when the term is active,
how trajectories identify its tokens, and what evidence a completed run
records.

## The objective

For a trajectory with action tokens \(A\) and environment-observation tokens
\(O\), Kiln composes:

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{policy}}(A)
+ \lambda_{\mathrm{echo}}\mathcal{L}_{\mathrm{envCE}}(O)
$$

The environment term is length-normalized over the full observation span. The
default coefficient is `0.05`.

ECHO reuses the trajectory's model forward pass, but it is still real training
work: Kiln must identify observation rows, compute their cross-entropy, and add
their gradient contribution to the shared loss root. Do not describe it as
free, and do not infer effectiveness from configuration alone.

## When ECHO is active

`GrpoConfig::default()` enables `loss.echo`. Whether the term contributes to a
step depends on the data:

| Rollout shape | ECHO configuration | Result |
|---|---|---|
| Multi-turn trajectory with usable `observation` tokens | Enabled, nonzero lambda | Environment cross-entropy contributes |
| Legacy text-only completion | Enabled | No observation rows; ECHO contributes zero |
| Any rollout | `loss.echo: null` or lambda `0` | ECHO contributes zero |
| Observation-only adaptation | `no_policy_loss: true` and ECHO enabled | Policy-gradient coefficients are removed; environment cross-entropy drives the update |

Legacy single-turn GRPO remains valid. Its synthesized trajectory has one
`action` segment and no `observation` segment, so the ECHO term is zero.

ECHO does not make an unsupported training backend available. It follows the
backend and route admitted for GRPO or OPD. On Vulkan, an ECHO-active GRPO step
uses the composite loss root because the fused active-rows route does not carry
environment rows. This is a capability-derived fallback, not a machine-name or
device-ID exception.

## Trajectory contract

Each scored rollout can carry a canonical `trajectory`. Every segment has:

- `role`: chat-template metadata such as `assistant` or `tool`;
- `content`: the raw text before chat-template rendering;
- `kind`: the authoritative supervision route;
- optional `tool_call_id` correlation metadata;
- optional `warning_prefix_len` for a known harness-warning prefix.

The `kind` values are:

| Kind | Meaning | Gradient target |
|---|---|---|
| `context` | System, user, or other non-trainable scaffolding | None |
| `action` | Model-generated assistant output | GRPO policy objective |
| `observation` | Tool result or environment output | ECHO environment cross-entropy |

`role` does not choose the loss. `kind` does. This keeps an unusual chat role
from silently becoming trainable and lets a producer describe the supervision
boundary explicitly.

Kiln renders the prompt and trajectory as one conversation, tokenizes it, and
builds disjoint action and environment masks. A token cannot belong to both
masks.

## Submit an agentic group

`POST /v1/train/agentic` is the canonical agentic alias of
`POST /v1/train/grpo`. It accepts `agentic_groups`/`rollouts`; the legacy
`groups`/`completions` names remain accepted aliases.

```json
{
  "agentic_groups": [
    {
      "messages": [
        {"role": "system", "content": "Use the shell carefully."},
        {"role": "user", "content": "Print the current directory."}
      ],
      "rollouts": [
        {
          "text": "<tool_call>pwd</tool_call><TURN_BREAK>The current directory is /srv/project.",
          "reward": 1.0,
          "trajectory": [
            {
              "role": "assistant",
              "content": "<tool_call>pwd</tool_call>",
              "kind": "action",
              "tool_call_id": "call-1"
            },
            {
              "role": "tool",
              "content": "/srv/project",
              "kind": "observation",
              "tool_call_id": "call-1"
            },
            {
              "role": "assistant",
              "content": "The current directory is /srv/project.",
              "kind": "action"
            }
          ]
        },
        {
          "text": "<tool_call>cwd</tool_call><TURN_BREAK>I could not determine it.",
          "reward": 0.0,
          "trajectory": [
            {
              "role": "assistant",
              "content": "<tool_call>cwd</tool_call>",
              "kind": "action",
              "tool_call_id": "call-2"
            },
            {
              "role": "tool",
              "content": "cwd: command not found",
              "kind": "observation",
              "tool_call_id": "call-2"
            },
            {
              "role": "assistant",
              "content": "I could not determine it.",
              "kind": "action"
            }
          ]
        }
      ]
    }
  ],
  "config": {
    "output_name": "shell-agent-r001",
    "loss": {
      "echo": {
        "lambda": 0.05,
        "env_mask_mode": "env_only",
        "warning_filter": true
      },
      "opd": null,
      "no_policy_loss": false
    }
  }
}
```

`text` remains required for compatibility. For a multi-turn trajectory, its
conventional value is the `<TURN_BREAK>`-joined action text. Kiln uses the
structured trajectory for masks; it does not recover observation boundaries by
parsing the flattened text.

As with ordinary GRPO, a useful group needs reward contrast. ECHO can add an
observation-learning signal, but it does not turn an all-equal reward group
into evidence that one action was better than another.

## Configuration

ECHO is request-local. It has no process-global environment override.

| Field | Default | Effect |
|---|---:|---|
| `loss.echo` | enabled | Set to `null` to disable the term |
| `loss.echo.lambda` | `0.05` | Coefficient on environment cross-entropy |
| `loss.echo.env_mask_mode` | `env_only` | Use the filtered environment span |
| `loss.echo.warning_filter` | `true` | Trim the producer-declared warning prefix |
| `loss.no_policy_loss` | `false` | Remove the GRPO policy term and train from ECHO only |
| `loss.opd` | `null` | Reserved on the GRPO route; a non-null value is rejected |

The ordinary `kiln train grpo` command preserves `config.loss` from a JSON
request file. Its JSONL convenience form constructs a smaller request and does
not expose ECHO-specific flags. Use an explicit JSON request or the HTTP API
when changing the loss composition.

The internal `cuda_grpo_ablation` example executable has development-only
`--echo-lambda`, `--no-echo`, and `--no-policy-loss` flags. Those are not
`kiln serve` options and do not create global policy.

## Warning filtering

Some trajectory producers prepend predictable harness warnings to an
observation. A segment can report the exact byte length in
`warning_prefix_len`.

With the defaults:

- `warning_filter: true` starts the environment mask after that prefix;
- `env_mask_mode: env_only` respects the filtered start;
- length normalization still uses the full semantic observation span.

`env_mask_mode: full_obs` includes the full observation, including the warning
prefix. It is primarily a mask-debugging mode. `warning_prefix_len` is
producer-supplied metadata; Kiln does not guess a prefix by searching arbitrary
tool output.

Inspect the receipt token counts when validating a producer:

- `env_tokens_before_warning_filter`;
- `env_tokens_after_warning_filter`;
- `warning_tokens_filtered`;
- `env_tokens`.

Kiln emits a warning when filtering removes most reported observation tokens.

## Observation-only adaptation

`loss.no_policy_loss: true` is live. It removes the GRPO policy-gradient term
and leaves ECHO's environment cross-entropy as the gradient source.

Use it only when every retained rollout has usable observation rows. The
configuration is rejected when ECHO is disabled, and an individual completion
with no environment rows cannot produce an observation-only step.

This mode answers a different training question from GRPO: “Can the model
better predict the consequences represented in these trajectories?” It does
not optimize the scalar reward directly. Keep a held-out behavioral evaluation
and compare against the previous accepted adapter before promotion.

## OPD boundary

`config.loss.opd` on the GRPO route is a reserved shape and currently rejects.
Use `POST /v1/train/opd` for on-policy distillation.

The OPD route has its own optional ECHO configuration for off-policy agentic
trajectories. When active there, action-token distillation and
observation-token cross-entropy share a step but retain separate positions,
teacher evidence, and receipt fields. Do not enable the reserved GRPO
`loss.opd` field to imitate that route.

## Read the receipt

The completed `train_receipt.json` records the configured ECHO policy at
top-level `echo`:

```json
{
  "enabled": true,
  "lambda": 0.05,
  "env_mask_mode": "env_only",
  "warning_filter": true,
  "initial_env_ce": 4.21,
  "final_env_ce": 1.12
}
```

Interpret the fields together:

- `enabled` says the nonzero term was armed for the run;
- non-null `initial_env_ce` and `final_env_ce` prove that environment rows were
  measured;
- `token_counts.env_tokens` proves how many filtered environment positions
  participated;
- null cross-entropy fields with zero environment tokens mean the configured
  term had no applicable rows.

The receipt does not currently publish `env_ce_drop_pct`,
`lambda_effective_final`, or held-out dynamics cross-entropy. Compute any
derived percentage from the recorded values and store held-out diagnostics in
the evaluation artifact that produced them; do not present aspirational fields
as trainer output.

A lower training-set environment cross-entropy is evidence that the auxiliary
objective fitted those observations. It is not proof that reward, task success,
or held-out environment prediction improved. Promotion still requires a
held-out evaluation.

## When to disable ECHO

Disable ECHO for a job when:

- the data is intentionally action-only and you want an unambiguous GRPO
  baseline;
- you are running a paired loss ablation;
- trajectory masks or warning-prefix metadata are under investigation;
- the observation stream contains secrets or artifacts that should not become
  training targets.

Do not disable it merely because the current dataset has no observation
segments; leaving the default enabled still contributes zero. Choose the
setting that makes the run's intent clearest in its effective configuration and
receipt.

## Verification

The implementation has CPU contract tests for configuration, mask separation,
warning-prefix trimming, ECHO-only validation, and closed-form gradient
composition. Backend qualification remains separate from those shared-math
tests. A passing mask test does not claim that every backend supports or
performs the complete training workload.

For an operational check:

1. submit a tiny trajectory with at least one action and one observation;
2. confirm nonzero action and environment token counts;
3. confirm `echo.initial_env_ce` and `echo.final_env_ce` are non-null;
4. inspect the adapter smoke test;
5. run a held-out comparison before activation.

## Further reading

- [GRPO Training Guide](GRPO_GUIDE.md)
- [Evaluation Guide](EVAL_GUIDE.md)
- [ECHO paper archive](papers/echo/echo_paper.md)
- [ECHO integration plan](plans/echo-integration-plan.md)
