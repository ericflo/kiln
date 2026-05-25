# H82 Broad Post-Read Edit Choice

## Hypothesis

H81 showed that final-action suffix training does not reduce inference cost.
H82 moved the signal earlier in the workflow: after the model reads
`solution.py`, rank a correct semantic body edit above a plausible but wrong
body edit.

This differs from H71/H72 in two ways:

- it uses a broader set of train-only tasks instead of two natural wrong-edit
  groups;
- every rejected edit is hand-checked to fail the train doctests, giving a
  clean semantic contrast without failed-doctest repair tails or terminal-stop
  artifacts.

The hypothesis was that a broader local edit-choice signal could improve
outcome reliability without backpropagating through full multi-turn tool
trajectories.

## Prompt Diagnostic Side Probe

Before training, a no-adapter prompt probe tested whether extra runtime
instructions could preserve "think enough" while shortening decisions:

| metric | base | extra prompt |
| --- | ---: | ---: |
| composite | 0.971875 | 0.750000 |
| zero rollouts | 0 | 1 |
| mean wall-clock s | 21.70 | 18.65 |

The prompt was faster but damaged reliability, so it was not distilled.

## Dataset

Dataset:
`/tmp/pi-doctest-h82-broad-postread-edit-choice/grpo-train.broad-postread-edit-choice.g9.jsonl`

Source:
`capabilities/caps/pi-doctest/datasets/train.tasks.jsonl`

Task IDs:

- `task_0024`
- `task_0025`
- `task_0026`
- `task_0028`
- `task_0029`
- `task_0032`
- `task_0034`
- `task_0037`
- `task_0038`

Each group context contains the original system/user prompt, a read action,
and the train-only `solution.py` stub observation. The preferred completion is
a task-specific correct `edit` action. The rejected completion is a plausible
wrong `edit` action that was verified to fail the train doctests.

Dataset stats:

| metric | value |
| --- | ---: |
| groups | 9 |
| completions | 18 |
| reward mean | 0.575 |
| reward stdev | 0.425 |
| action tokens | 486 |
| env tokens | 0 |
| context tokens | 7386 |
| sha256 | `sha256:577b846a8d5f02d69ce958a77b8c74748d347f91e53228a9149a7428b8e84a60` |

## Training

Adapter:
`pi-doctest-h82-broad-postread-edit-choice-r4a4lr1e7`

Command shape:

- `cuda_grpo_ablation --mode phase1`
- rank 4 / alpha 4
- lr `1e-7`
- seed `3141592653`
- no ECHO
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

Training completed successfully in 425.397s observed. The receipt reported
421.236s wall-clock, 54.508s reference forward, 29.842s policy forward, and
334.470s backward. Peak observed VRAM was 15,973 MiB.

Adapter verify passed:

| metric | value |
| --- | ---: |
| rank | 4 |
| alpha | 4 |
| alpha / rank | 1.0 |
| tensor count | 400 |
| projection pairs | 200 |
| nonzero tensors | 400 |
| LoRA update proxy | 0.022547 |
| adapter hash | `sha256:657867e57e324423efbec8d594e5ac284a6b6ee90050414c13cda8a36514919e` |

## Blind Gates

Cheap smoke, `LIMIT=4 SEEDS=1`:

| metric | base | H82 |
| --- | ---: | ---: |
| composite | 0.900000 | 0.934375 |
| zero rollouts | 0 | 0 |
| mean wall-clock s | 43.16 | 49.25 |

First wider gate, `LIMIT=8 SEEDS=1`:

| metric | base | H82 |
| --- | ---: | ---: |
| composite | 0.601562 | 0.818750 |
| zero rollouts | 3 | 1 |
| mean wall-clock s | 63.77 | 62.02 |

Second wider gate, `LIMIT=8 SEEDS=1`:

| metric | base | H82 |
| --- | ---: | ---: |
| composite | 0.828125 | 0.556250 |
| zero rollouts | 1 | 3 |
| mean wall-clock s | 51.31 | 76.83 |

Aggregate over the two `LIMIT=8` gates:

| metric | base | H82 |
| --- | ---: | ---: |
| composite mean | 0.714844 | 0.687500 |
| total zero rollouts | 4 | 4 |
| mean wall-clock s | 57.54 | 69.43 |

## Verdict

Rejected at confirmation.

H82 was the strongest recent smoke result and converted one weak hard-tail
sample, but it failed the second wider confirmation. Across the two wider
gates, it slightly trailed base, had the same total zero count, and was much
slower. The broader post-read semantic-edit signal is real enough to move the
model, but still too unstable as a standalone adapter update.

Lesson: semantic edit-choice data is a better direction than terminal suffix
micro-contrasts, but it needs a stabilizer before promotion. A future attempt
could use this as one component in an oscillating chain with a confirmed
successful-workflow distribution, or collect teacher-quality edit choices
instead of hand-authored plausible wrong negatives.
