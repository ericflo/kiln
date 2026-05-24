# PB-H7: Full-Train Conservative GRPO/ECHO

## Hypothesis

PB-H2/H5/H6 all used the small clean PB-H2 train rollout set, and each update
found a different unstable tradeoff between outcome, format, and convention.
Collecting rollouts over the full train task pool should reduce overfitting to
the small 16-group sample. A conservative GRPO/ECHO update can then use the
larger reward-variance pool to preserve convention while recovering the H5
outcome and format gains.

## Recipe

- Source: fresh default-prompt train-only rollouts from all 32 train tasks.
- Rollout sampling: 4 generations per task, no active adapter.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`, `rank=8`, `alpha=16`,
  `lr=5e-6`, `echo_lambda=0.03`, `filter_var_min=0.05`.
- Keep `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Keep thinking enabled during rollout collection and eval through the server
  default.

## Falsification

Reject if any of:

- fewer than 12 reward-variant train groups remain after filtering;
- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Rejected. The full train rollout set produced enough reward variance for this
test: 20/32 groups survived `filter_var_min=0.05`, with 80/128 completions
trained. The CUDA 13.2 gradient-checkpointed GRPO run completed successfully on
the RTX 4090 with peak VRAM 19,710 MiB and elapsed time 7,133.393 seconds. It
also exercised the long-context checkpoint path: a 12k-token trajectory used
boundary spooling without OOM.

Adapter verification passed for
`pi-context-aware-edits-postbreach-pb-h7-fulltrain-conservative-grpo-echo-r8a16`
(`rank=8`, `alpha=16`, 400 nonzero tensors, LoRA update L2 upper bound
3.793617). Blind three-seed eval nevertheless regressed:

- composite: 0.2483 (`delta=-0.0513`, stdev 0.4302)
- `outcome`: 0.4375
- `format_compliance`: 0.5208
- `convention_consistency`: 0.9167
- `read_before_edit`: 0.9583
- `no_redundant_imports`: 1.0000
- `no_style_drift`: 1.0000
- efficiency: 5.88 tool calls/rollout, 2080.6 thinking chars, 347.0 thinking
  chars/tool call

This falsifies the idea that simply broadening the default-prompt reward pool
stabilizes the GRPO update. It worsened every non-saturated behavioral gate
relative to baseline.
