# H78 Real Repair Moderate Anchor

## Hypothesis

H77 supported the user's point that the goal is not simply less thinking:
moderate task-specific thinking beat terse thinking on the early gates. It
failed confirmation because the data was still a small synthetic style-only
contrast.

H78 mixed two real repair-continuation groups from H64 with one mild H77
moderate-thinking anchor. The goal was to keep real reward variance as the main
signal while preserving enough successful, task-specific thinking to avoid the
terse-imitation failures from H76.

## Data

Dataset:
`/tmp/pi-doctest-h78-real-repair-moderate-anchor/grpo-train.real-repair-plus-moderate-anchor.g3.jsonl`

Sources:

- `/tmp/pi-doctest-h64-repair-continuation/grpo-train.repair-continuation.g2.jsonl`
- `/tmp/pi-doctest-h77-moderate-thinking-workflow/grpo-train.moderate-thinking-workflow.g3x3.jsonl`

The final data contained two real repair-continuation groups with rewards
`1.0` vs `0.75`, plus one mild moderate-thinking anchor group with rewards
`0.94`, `0.90`, and `0.90`.

Dry-run passed:

| metric | value |
| --- | ---: |
| groups | 3 |
| completions | 7 |
| action tokens | 1399 |
| env tokens | 1193 |
| context tokens | 2002 |
| reward stdev | 0.097164 |
| data hash | `sha256:0793dd228c0e48ccfcb8c507dfc134929c13521a41b78815179fda28a56f9a86` |

## Training

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4 / lr
`1e-7`, no ECHO, seed `3141592653`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed locally in 1661.690s observed, with peak observed VRAM 16,001
MiB. The receipt reported 1657.644s wall-clock, 3 groups trained, 7
completions, 1399 action tokens, 1193 env tokens, 2002 context tokens, and
adapter hash
`sha256:c90897e23ebf277d0746ded6a6093cffc3017eb0319b040515bc1d13bfa3418b`.
Adapter verify passed with 400 nonzero tensors, 200 LoRA projection pairs, and
LoRA update proxy 0.008076.

Adapter:
`pi-doctest-h78-real-repair-moderate-anchor-r4a4lr1e7`

## Blind Gates

Cheap smoke, `LIMIT=4 SEEDS=1`:

| metric | base | H78 |
| --- | ---: | ---: |
| composite | 0.750000 | 0.866667 |
| zero rollouts | 1 | 0 |
| mean wall-clock s | 20.75 | 44.89 |

Wider confirmation, `LIMIT=8 SEEDS=1`:

| metric | base | H78 |
| --- | ---: | ---: |
| composite | 0.809375 | 0.726562 |
| zero rollouts | 1 | 2 |
| mean wall-clock s | 45.31 | 63.30 |

## Verdict

Rejected at confirmation.

H78 reproduced the now-common pi-doctest pattern: a promising small smoke win
that does not survive the wider gate. Mixing real repair-continuation signal
with a mild thinking anchor was not enough; it reduced confirmed composite,
introduced another zero rollout, and added latency. The useful lesson is that
real reward variance helps early, but these repair-continuation traces are too
expensive and too narrow unless paired with a stronger base-distribution
regularizer or broader success anchor.
