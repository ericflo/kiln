# H1: Strong-Signal Agentic-GRPO for Format and Outcome Reliability

## Hypothesis

The repaired baseline is bimodal: half the eval rollouts score nonzero, with
headroom dominated by `format_compliance` and `outcome`, while
`convention_consistency` is already near-saturated. Agentic-GRPO with ECHO
on high-variance rollout groups should increase the probability of completing
the edit and emitting the required final response without spending materially
more thinking per tool call.

## Recipe

- Base: Qwen3.5-4B, thinking enabled by server default.
- Pi model: `qwen-3.5-4b-kiln-pi1024`.
- Rollouts: train corpus, 4 generations/task, max wall clock 120s.
- Trainer: `cuda_grpo_ablation`, `rank=16`, `alpha=32`, `lr=1e-5`,
  `kl_coeff=0.1`, `dr_grpo`, token-level aggregation.
- ECHO: enabled, `lambda=0.05`, `env_only`, warning filter enabled.
- Strong-signal filter: `filter_var_min=0.05`.
- Target: `format_compliance` and `outcome`; preserve the already-high
  `convention_consistency`.

## Falsification

Reject or do not promote if the 3-seed blind eval shows any of:

- composite lift < +0.05 versus `baseline-0`;
- `format_compliance` or `outcome` fails to improve;
- `convention_consistency` drops below 0.90;
- mean thinking chars/tool call increases by more than 25% over the
  baseline value of 302.7;
- receipt shows zero ECHO env-token steps or adapter verification fails.

## Results

Status: rejected.

- Default 8-segment checkpointing OOMed on the full rank-16 arm, a
  max-groups=4 rank-8 arm, and a max-groups=1 rank-4 arm during
  checkpointed GRPO reverse backward.
- A two-completion rank-4 micro arm trained and verified but regressed
  blind eval composite from 0.4800 to 0.4528.
- Explicit `KILN_GRAD_CHECKPOINT_SEGMENTS=32` fit the max-groups=4
  rank-8 arm on the RTX 4090: 3 kept groups, 12 completions, 10,278
  action tokens, 4,997 env tokens, 948s wall clock, observed peak VRAM
  18,839 MiB.
- The 32-segment arm regressed blind eval composite to 0.1625
  (`delta=-0.3175`). It preserved `convention_consistency` at 0.9583
  and `read_before_edit` at 1.0000, but `format_compliance` fell to
  0.4861 and `outcome` to 0.4167. Thinking chars/tool improved
  (302.7 to 284.4), but mean tool calls increased (5.00 to 5.42) and
  nonzero rollouts collapsed (18 to 6).

Conclusion: gradient checkpointing solves the local memory bottleneck for
larger H1 slices, but the reward signal is too sparse/noisy for this
agentic-GRPO recipe. Scaling H1 data worsens the exact target sub-scores,
so the next iteration should switch methods rather than continue GRPO
hyperparameter sweeps on the same rollout distribution.

## Rationale

`pi-doctest` found `filter_var_min > 0.05` more robust than training on all
groups, and the repaired baseline has high reward variance
(`composite_stdev=0.4869`). This is a multi-turn tool-calling cap, so
METHODS Rule A routes to agentic-GRPO with ECHO.
