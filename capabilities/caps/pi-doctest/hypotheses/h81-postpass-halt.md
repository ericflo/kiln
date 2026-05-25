# H81 Post-Pass Halt

## Hypothesis

H80 closed the immediate adapter-composition branch around old fragile
adapters. H81 switched back to new train-only behavior data, but narrowed the
target to the final decision point: after the agent has already observed a
successful doctest result, prefer stopping immediately over spending another
tool call on a redundant second doctest.

The hypothesis was that a suffix-only post-pass contrast could reduce
wall-clock without changing read/edit/test solution behavior. This also tests
whether the efficiency problem can be fixed at the final-action policy level
instead of through broad workflow imitation.

## Dataset

Dataset:
`/tmp/pi-doctest-h81-postpass-halt/grpo-train.postpass-halt.g8.jsonl`

Source:
`capabilities/caps/pi-doctest/datasets/train.tasks.jsonl`

The dataset contains eight train-only post-pass contexts. Each context already
includes read, edit, and one successful doctest observation. The preferred
completion emits `DONE` immediately; the rejected completion runs a redundant
second doctest and then emits `DONE`.

Tasks:

- `task_0024`
- `task_0026`
- `task_0028`
- `task_0029`
- `task_0032`
- `task_0034`
- `task_0037`
- `task_0038`

Dataset stats:

| metric | value |
| --- | ---: |
| groups | 8 |
| completions | 16 |
| reward mean | 0.775 |
| reward stdev | 0.225 |
| action tokens | 926 |
| env tokens | 802 |
| context tokens | 7156 |
| sha256 | `sha256:ab3459edecf5e3589bfc7f635507922b1370471e78d2b1bb08c3f954068ff810` |

## Training

Adapter:
`pi-doctest-h81-postpass-halt-r4a4lr1e7`

Command shape:

- `cuda_grpo_ablation --mode phase1`
- rank 4 / alpha 4
- lr `1e-7`
- seed `3141592653`
- no ECHO
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

Training completed successfully in 970.108s observed. The receipt reported
965.569s wall-clock. Peak observed VRAM was 16,013 MiB. Phase timing was
229.032s reference forward, 170.546s policy forward, and 563.616s backward.

Adapter verify passed:

| metric | value |
| --- | ---: |
| rank | 4 |
| alpha | 4 |
| alpha / rank | 1.0 |
| tensor count | 400 |
| projection pairs | 200 |
| nonzero tensors | 400 |
| LoRA update proxy | 0.021651 |
| adapter hash | `sha256:aa949b81ae7423cdcd4ee3db8843eb4312130412190dfba6b72d480f5888bd98` |

## Blind Gate

Cheap smoke, `LIMIT=4 SEEDS=1`:

| metric | base | H81 |
| --- | ---: | ---: |
| composite | 0.953125 | 0.953125 |
| zero rollouts | 0 | 0 |
| mean wall-clock s | 33.36 | 40.60 |

## Verdict

Rejected at smoke.

H81 tied paired base on composite and zero rollouts, but was slower by about
7.24s mean wall-clock. The narrow post-pass halt contrast did not reduce
inference cost under thinking-on decoding, so no wider gate was run.

Lesson: final-action suffix training alone is not enough to make thinking
efficient. The model may still spend extra time before the post-pass state, or
the adapter update may add latency without changing the actual stop decision.
Gradient checkpointing made the local training run fit in VRAM, but it remains
a training-memory tool and does not control inference thinking tokens.
