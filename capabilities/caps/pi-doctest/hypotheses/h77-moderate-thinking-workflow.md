# H77 Moderate Thinking Workflow

## Hypothesis

H76 punished verbose thinking and strongly preferred terse successful
workflows. It trained quickly but failed blind smoke, which is consistent with
the broader observation that this model needs some thinking to score well.

H77 tested a three-way compact workflow contrast: the same correct
read/edit/doctest/DONE tool sequence and same solution code, but with a
moderate task-specific thought preferred over both a terse and a verbose
variant. The intent was not "think less"; it was "think just enough, then use
the tools."

## Data

Dataset:
`/tmp/pi-doctest-h77-moderate-thinking-workflow/grpo-train.moderate-thinking-workflow.g3x3.jsonl`

Train-only task IDs:

- `task_0024`
- `task_0026`
- `task_0057`

Each group had three completions:

- moderate task-specific thought, reward 1.0
- terse thought, reward 0.82
- verbose generic thought, reward 0.82

Dry-run passed:

| metric | value |
| --- | ---: |
| groups | 3 |
| completions | 9 |
| action tokens | 1523 |
| env tokens | 1083 |
| context tokens | 2502 |
| reward stdev | 0.084853 |
| data hash | `sha256:071acc422adc062103e7dc393f22be03b27d3330aa6b7d7e7f2344c679f6933c` |

## Training

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4 / lr
`1e-7`, no ECHO, seed `3141592653`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed locally in 465.773s observed, with peak observed VRAM 15,986 MiB.
The receipt reported 461.577s wall-clock, 3 groups trained, 9 completions,
1523 action tokens, 1083 env tokens, 2502 context tokens, and adapter hash
`sha256:a0bd759f15ca6ccca76fcd42e3d8bad34b04d3b9b33e8e3a146511bc1e40ee33`.
Adapter verify passed with 400 nonzero tensors, 200 LoRA projection pairs, and
LoRA update proxy 0.009156.

Adapter:
`pi-doctest-h77-moderate-thinking-workflow-r4a4lr1e7`

## Blind Gates

Cheap smoke, `LIMIT=4 SEEDS=1`:

| metric | base | H77 |
| --- | ---: | ---: |
| composite | 0.925000 | 0.971875 |
| zero rollouts | 0 | 0 |
| mean wall-clock s | 27.34 | 28.83 |

First wider gate, `LIMIT=8 SEEDS=1`:

| metric | base | H77 |
| --- | ---: | ---: |
| composite | 0.753906 | 0.842188 |
| zero rollouts | 1 | 1 |
| mean wall-clock s | 54.04 | 67.70 |

Confirmation wider gate, `LIMIT=8 SEEDS=1`:

| metric | base | H77 |
| --- | ---: | ---: |
| composite | 0.707813 | 0.593750 |
| zero rollouts | 2 | 3 |
| mean wall-clock s | 64.44 | 70.99 |

Across the two wider gates, base averaged 0.730859 composite with three zero
rollouts over sixteen, while H77 averaged 0.717969 with four zero rollouts
over sixteen. H77 was also slower on both wider draws.

## Verdict

Rejected at confirmation.

Moderate thinking is a better target than terse-over-verbose imitation: it won
the cheap smoke and the first wider gate. But it did not survive confirmation.
The direction still fails to stabilize hard-tail reliability and adds latency.
Future work should avoid small synthetic style-only workflow contrasts unless
they are paired with a real reward-variance signal or a base-distribution
regularizer.
