# PB-H14: Temperature-0.6 No-ECHO GRPO

## Hypothesis

PB-H10 is still the only postbreach adapter above baseline, but PB-H11 through
PB-H13 show that rank, scale, and reward-filter sweeps around the same PB-H7
rollout pool are exhausted. The strongest neighboring lesson from
`pi-faithful-completion` is that lowering rollout temperature from `0.8` to
`0.6` can clean up the advantage signal enough to move score without changing
the objective. Generate a fresh train-only rollout pool with the same
final-only prompt and output cap, but force the provider payload to
`temperature=0.6` and `top_p=0.95` via a Pi extension. Then train the H10
conservative no-ECHO recipe on that fresh pool.

If PB-H10's improvement came from useful successes buried in a noisy rollout
distribution, the lower-temperature pool should preserve the outcome/format
movement while reducing convention and read-before-edit drift.

## Recipe

- Config: `configs/pb-h14-temp06.config.json`.
- Pi extension: `extensions/provider-payload-temp06.ts`, adding
  `temperature=0.6` and `top_p=0.95` to each provider request.
- Rollouts: train split only, 32 tasks, 4 generations/task, thinking enabled
  through the CUDA server default, 1024 Pi output cap.
- Trainer: `cuda_grpo_ablation`, `mode=phase1`.
- Hyperparameters: `rank=4`, `alpha=8`, `lr=5e-6`, `filter_var_min=0.05`,
  `--no-echo`.
- Keep `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.

## Falsification

Reject if any of:

- the extension does not visibly apply the lower-temperature payload during
  smoke testing;
- fewer than 12 reward-variant train groups survive `filter_var_min=0.05`;
- composite lift < +0.05 versus postbreach baseline;
- composite is not above PB-H10's 0.3404 current-best score;
- `outcome < 0.5208`;
- `format_compliance < 0.6042`;
- `convention_consistency < 0.9542`;
- `read_before_edit < 0.9667`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Payload smoke passed before full rollout generation. With
`KILN_PI_PAYLOAD_TRACE=1`, a one-task train smoke wrote 7 provider-request
trace rows, each showing `temperature=0.6` and `top_p=0.95`. The smoke rollout
is not used for training. Full rollout generation and training are pending.
