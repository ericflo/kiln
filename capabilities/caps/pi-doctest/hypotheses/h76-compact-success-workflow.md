# H76 Compact Success Workflow

## Hypothesis

H74 and H75 showed that suffixes sliced from long thinking-on transcripts are
too context-heavy. H76 switched to compact-from-start train-only data: verified
short solution trajectories built directly from train task stubs.

Each group used the same correct read/edit/doctest/DONE workflow on both
completions. The preferred completion used terse thinking, while the weaker
completion used verbose thinking. This tested whether full successful workflow
pairs could teach compactness without failed code, wrong edits, or premature
terminal negatives.

## Data

The first build attempted six train tasks, then excluded two task shapes during
local train-verifier construction:

- `task_0060` has doctest examples expecting `None`, which the doctest display
  hook treats as no output.
- `task_0063` has a median doctest inconsistent with the usual sorted median
  implementation.

The final trained-attempt subset used three verified short tasks:

- `task_0024`
- `task_0026`
- `task_0057`

Dataset:
`/tmp/pi-doctest-h76-compact-success-grpo/grpo-train.compact-success-workflow.g3.jsonl`

Dry-run passed:

| metric | value |
| --- | ---: |
| groups | 3 |
| completions | 6 |
| action tokens | 953 |
| env tokens | 722 |
| context tokens | 1668 |
| reward stdev | 0.075000 |
| data hash | `sha256:13b4ceb792b586b721c4530ca83e8834f2fad235a5cae0ebb4577b71baa518dc` |

## Training

Training used `cuda_grpo_ablation --mode phase1`, rank 4 / alpha 4 / lr
`2e-7`, no ECHO, seed `3141592653`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed locally in 107.683s observed, with peak observed VRAM 15,941 MiB.
The training receipt reported 103.961s wall-clock, 953 action tokens, 722 env
tokens, 1668 context tokens, 3 groups trained, and no filtered groups.
Adapter smoke passed with changed logits. Adapter verify passed with 400
nonzero tensors, 200 LoRA projection pairs, and LoRA update proxy 0.018225.

Adapter:
`pi-doctest-h76-compact-success-workflow-r4a4lr2e7`

## Blind Smoke

Paired `LIMIT=4 SEEDS=1` smoke:

| metric | base | H76 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.703125 |
| zero rollouts | 0 | 1 |
| nonzero rollouts | 4 | 3 |
| mean wall-clock s | 29.40 | 60.36 |

## Verdict

Rejected at smoke.

H76 answers the H74/H75 throughput question: compact-from-start full workflow
data trains quickly on the laptop. It also shows that success-only
concise-over-verbose full workflows still produce the same behavioral drift
seen in earlier SFT and suffix-ranking attempts: lower reliability and much
higher latency. The next successful route likely needs real reward variance or
a regularizer that preserves base distribution, not more positive full
workflow imitation at small scale.
