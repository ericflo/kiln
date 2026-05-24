# H68 Repair-Continuation ECHO 0.075 Low Dose

## Hypothesis

H64 was the strongest recent near-miss: it passed smoke and one `LIMIT=8`
promotion draw, then failed confirmation with hard-tail zero rollouts. H68
tested whether H64's failure was a conditioning problem rather than a data
problem. The repair-continuation examples include real doctest observations,
so adding ECHO at the `pi-code-comprehension` productive ceiling (`0.075`)
could improve observation conditioning. The policy dose was lowered from
`5e-7` to `2e-7` to reduce H64's reliability drift.

The falsifiable prediction was that the same train-only repair-continuation
data would keep H64's short-run lift while reducing zero-rollout instability.

## Data

Source:
`/tmp/pi-doctest-h64-repair-continuation/grpo-train.repair-continuation.g2.jsonl`.

This is the same two-group H64 dataset:

- Preferred branch: direct successful read, edit, doctest pass, `DONE`.
- Lower-reward branch: real wrong first edit, failed doctest observation,
  concise repair continuation, passing doctest, `DONE`.
- Rewards: 1.0 vs 0.75.

Dry-run shape:

- Groups: 2.
- Completions: 4.
- Action tokens: 828.
- Env tokens: 938.
- Context tokens: 1168.
- Reward stdev: 0.125.

## Training

Adapter: `pi-doctest-h68-repair-echo075-lowdose-r4a4lr2e7`.

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4, lr
`2e-7`, seed `3141592653`, ECHO lambda `0.075`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 176.931s observed, with peak observed VRAM 16011
MiB. The receipt reported 828 action tokens, 938 env tokens, 1168 context
tokens, and reward stdev 0.125.

The local ECHO objective improved: env CE moved from 3.3952 to 2.4413.

Adapter verify passed through the running server with 400 nonzero LoRA tensors,
200 projection pairs, and LoRA update proxy 0.011593.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.953125, no zero rollouts, mean wall-clock 27.73s.
- H68: composite 0.75, one zero rollout, mean wall-clock 45.25s.
- Delta: -0.203125 composite, slower by 17.52s.

No wider promotion check was run.

## Verdict

Rejected at smoke. ECHO learned the training observations locally, but the
adapter became less reliable and slower on blind eval. This falsifies the
simple "H64 failed because no ECHO" explanation.

The repair-continuation distribution remains interesting because H64 produced
two positive gates before failing confirmation, but H68 shows that adding a
stronger env-token objective and lowering policy dose is not sufficient. Future
repair-tail work needs a broader reliability anchor or a different contrast
that does not compare direct success against repaired near-misses.

No eval task contents or per-example eval transcripts were inspected.
