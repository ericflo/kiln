# H64 Repair Continuation GRPO

## Hypothesis

H63 made real reward-variance GRPO locally trainable, but its rejected branch
ended with `DONE` immediately after a failed doctest. H64 keeps the same H44
train-only natural-variance source, but changes the near-miss branch into a
real failed edit followed by a concise repair continuation, passing doctest,
and `DONE`.

The falsifiable prediction was that a softer direct-success-versus-repair
contrast would preserve H63's trainability while avoiding the failed-doctest
terminal-stop reliability failure.

## Data

- Source: `/tmp/pi-doctest-h44-natural-compact-grpo/grpo-train.compact-pair.jsonl`.
- Output: `/tmp/pi-doctest-h64-repair-continuation/grpo-train.repair-continuation.g2.jsonl`.
- Groups: 2.
- Completions: 4.
- Rewards: 1.0 for direct success, 0.75 for real failed edit plus concise repair.
- Preferred trajectory: real successful read, edit, doctest pass, `DONE`.
- Repaired trajectory: real read, wrong first edit, real failed doctest
  observation, then the original compact repair action, passing doctest, and
  `DONE`.

A larger candidate was also built at
`/tmp/pi-doctest-h64-repair-continuation/grpo-train.repair-continuation-plus-postfail.g4.jsonl`.
It added post-failed-doctest repair-vs-DONE rescue groups, but dry-run context
grew to 4072 tokens, so it was not trained in this iteration.

Dry-run for the trained g2 set:

- action tokens: 828.
- env tokens: 938.
- context tokens: 1168.
- reward stdev: 0.125.

## Training

Adapter: `pi-doctest-h64-repair-continuation-g2-r4a4lr5e7`.

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4, lr
`5e-7`, seed `3141592653`, no ECHO, reward variance filter min `0.001`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 676.986s observed with peak observed VRAM 15998
MiB. The receipt reported 2 groups trained, 4 completions, 828 action tokens,
938 env tokens, 1168 context tokens, and reward stdev 0.125. Adapter verify
passed with 400 nonzero LoRA tensors and a LoRA update proxy of 0.029918.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.934375, no zero rollouts, mean wall-clock 39.44s.
- H64: composite 0.9625, no zero rollouts, mean wall-clock 44.66s.
- Delta: +0.028125 composite, slower by 5.22s.

First wider gate, `LIMIT=8 SEEDS=1`:

- Base: composite 0.73125, two zero rollouts, mean wall-clock 48.44s.
- H64: composite 0.8375, one zero rollout, mean wall-clock 57.44s.
- Delta: +0.10625 composite, slower by 8.99s.

Confirmation, `LIMIT=8 SEEDS=1`:

- Base: composite 0.722917, one zero rollout, mean wall-clock 71.21s.
- H64: composite 0.575, three zero rollouts, mean wall-clock 67.09s.
- Delta: -0.147917 composite, faster by 4.13s.

Across the two `LIMIT=8` pairs, base averaged 0.727083 composite with three
zero rollouts over 16 total rollouts. H64 averaged 0.70625 composite with four
zero rollouts over 16 total rollouts.

## Verdict

Rejected at confirmation. H64 is a better data direction than H63 in one
important sense: removing the failed-doctest terminal-stop branch allowed a
smoke win and one positive `LIMIT=8` draw. But the confirmation run showed the
same hard-tail instability, with more zero rollouts than paired base across
the two wider gates.

The lesson is not "repair continuations are bad"; it is that full-trajectory
direct-success-over-repair still penalizes the repair behavior along with the
bad first edit. The next real-variance attempt should either shorten and train
the post-failed-doctest rescue contrast directly, or create a lower-context
variant where the same failed state chooses repair over terminal `DONE`.

No eval task contents or per-example eval transcripts were inspected.
