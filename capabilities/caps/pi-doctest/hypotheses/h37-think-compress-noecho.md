# H37 Think-Compress No-ECHO

## Hypothesis

The user constraint is that thinking helps scores, but the thinking must be as
efficient as possible. H37 tests a safer data signal than hard negatives:
preserve successful workflows, but rank a compressed-thinking version above the
original verbose successful trace.

This should train shorter reasoning while keeping the same read/edit/doctest
tool skeleton and avoiding wrong/no-test terminal negatives.

## Data

- Source: `/tmp/pi-doctest-h17-action650-success-anchors/grpo-train.jsonl`.
- Output: `/tmp/pi-doctest-h37-think-compress/grpo-train.think-compress.jsonl`.
- Groups: 2 train-only successful traces converted into compressed-vs-original
  pairs.
- Completions: 4.
- Rewards: `[1.0, 0.0]` in each group, where `1.0` is the compressed-thinking
  version and `0.0` is the original successful trace.
- Base adapter: none.

Selection:

| source line | completion | original reward | original chars | compressed chars |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.987196 | 987 | 384 |
| 1 | 2 | 0.981482 | 1390 | 512 |

Dry-run token shape:

| metric | value |
| --- | ---: |
| groups | 2 |
| completions | 4 |
| action tokens | 815 |
| env tokens | 1298 |
| context tokens | 1140 |
| max seq length | 1010 |
| max action tokens/completion | 325 |

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, `--no-echo`. Gradient checkpointing
was explicitly enabled with `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

The full train timed out at 900s before writing an adapter. Group 1 completed,
but the original verbose counterpart dominated runtime: its backward pass took
393243 ms, including 93361 ms in checkpoint segment 7. Group 2 then entered a
longer original-trace shape; reference forward alone took 366307 ms for
`max_seq_len=1010`, `action_tokens=443`, `env_tokens=786`. The timeout fired
after group 2 reference forward and before any adapter artifact was saved.

Adapter: none.

## Verdict

Rejected for throughput. The signal is conceptually safer than wrong/no-test
hard negatives, but the current pair shape is not practical locally because the
original verbose counterpart can dominate wall-clock even with 24 checkpoint
segments.

Lessons:

- Thinking-compression rank data is still worth testing, but only with every
  completion under roughly 850 sequence tokens and under 300 action tokens.
- Reference forward can become the bottleneck before backward when grouped
  max sequence length reaches about 1000 with large env-token observations.
- For the next retry, build one group from the short `task_0025` traces only,
  or compress both sides enough that the low-reward original remains within the
  local throughput envelope.
