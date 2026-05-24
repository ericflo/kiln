# H44: complete-behavior throughput probe

## Hypothesis

H41-H43 kept producing smoke-positive, promotion-negative adapters because the
training rows were isolated tool-choice micro-contrasts. H44 moved back to
complete task-solving behavior: first a broad SFT anchor from successful
train-only rollouts, then a natural same-task GRPO contrast from compact
successes versus compact near-misses.

The goal was not to turn thinking off. The goal was to preserve the read,
write, test, DONE workflow while making the thinking efficient.

## Data

Fresh train-only base rollouts were collected in:

- `/tmp/pi-doctest-h44-broad-success-sft-rollouts`
- `/tmp/pi-doctest-h44-broad-success-sft-rollouts-supplement`

The SFT corpus was written to
`/tmp/pi-doctest-h44-broad-success-sft/sft.train.jsonl`.

- 10 examples.
- Task IDs: `task_0052`, `task_0034`, `task_0028`, `task_0039`,
  `task_0029`, `task_0024`, `task_0025`, `task_0086`, `task_0048`,
  `task_0074`.
- Selection: reward 1.0, exit 0, outcome/tested/tool-efficiency all 1.0,
  source trajectory under 3000 action+observation chars.
- Assistant text was normalized to compact read, write, test, DONE steps while
  preserving the real initial file, final solution, and doctest output.
- Tokenized examples ranged from 565 to 925 tokens.

The natural GRPO collection was written to
`/tmp/pi-doctest-h44-natural-compact-grpo-rollouts`.

- 8 train tasks, 3 generations each.
- Mean train composite 0.990625.
- Two natural variance groups remained: `task_0028` and `task_0039`, each
  with two reward-1.0 successes and one 0.8875 near-miss.

## Training Attempts

SFT native, 10 examples, rank 4 / alpha 4 / lr `1e-6`, with
`KILN_CUDA_RECOMPUTE_SFT=1`, fit in memory but was too slow for iteration.
The first step completed after roughly twenty minutes, with VRAM around
15975 MiB. The run was stopped with no adapter saved.

SFT generic, 6 examples, rank 4 / alpha 4 / lr `2e-6`, failed with CUDA OOM:

`failure_reason=oom: training forward pass (FLCE): segment gated deltanet layer 26`

The failed receipt was copied to
`/tmp/pi-doctest-h44-broad-success-sft/failed-generic6/train_receipt.json`, and
the receipt-only adapter directory was removed from the registry.

Raw natural GRPO on the two variance groups dry-ran successfully but was too
large:

- 2 groups, 6 completions.
- 3278 action tokens.
- 2859 env tokens.
- 1780 context tokens.

Compact natural GRPO stripped verbose thinking while preserving tool-call
sequence and rewards. The pair version dry-ran as:

- 2 groups, 4 completions.
- 1116 action tokens.
- 1292 env tokens.
- 1224 context tokens.

Training this pair with rank 4 / alpha 4 / lr `1e-6`, no ECHO, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24` reached a mid-run progress line
(`step=9887/18076`, loss 0.03225, VRAM 15948 MiB) but hit the 900s guard
before saving an adapter.

The smallest natural policy test used only `task_0039`:

- 1 group, 2 completions.
- 458 action tokens.
- 798 env tokens.
- 612 context tokens.

It also hit the 900s guard before saving an adapter, with VRAM around
15965 MiB.

## Verdict

Reject H44 as a completed adapter route. No adapter was produced, so no blind
smoke or promotion eval was run.

The data direction is still more defensible than H41-H43: complete train-only
workflow traces avoid isolated terminal/tool micro-contrasts. The local
throughput limit is the blocker. Even compact natural near-misses with 7 tool
calls are too expensive for policy-on local training unless the negative
completion is reduced further.

Next attempts should enforce a stricter cap before training: likely sub-300
action tokens per negative completion and sub-800 total sequence tokens, or a
preconditioning route that avoids policy backward on full trajectories. Keep
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`; it reliably keeps GRPO under the local
VRAM ceiling, and lowering checkpointing is not the path to better throughput.

No eval task contents or per-example eval transcripts were inspected.
