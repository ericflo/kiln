# H75 Compressed Success Suffix

## Hypothesis

H74 showed that natural success-only suffix ranking is semantically cleaner than
failure-negative suffixes, but the long thinking-on prefixes made local
training too slow. H75 kept the same success-only suffix signal and
canonicalized prior assistant prefix messages before training.

The prediction was that shorter prefix messages would preserve the useful
natural preference while moving the run off the slow long-prefix path.

## Data

Source:
`/tmp/pi-doctest-h74-natural-success-suffix/grpo-train.natural-success-suffix.g2x2-soft.jsonl`.

Output:
`/tmp/pi-doctest-h75-compressed-success-suffix/grpo-train.compressed-natural-success-suffix.g2x2-soft.jsonl`.

The transformation rewrote nine prior assistant prefix messages into short
canonical read/edit/doctest forms while leaving the preferred and rejected
completion actions unchanged. Prefix chars fell from 6414 to 4266.

Dry-run passed:

| metric | value |
| --- | ---: |
| groups | 4 |
| completions | 8 |
| action tokens | 184 |
| env tokens | 0 |
| context tokens | 6400 |
| reward stdev | 0.100000 |
| data hash | `sha256:f9ce8c82dc9f3750fd5e2449cc5a16188fc75ad766899b67e09f4ce7592d3e93` |

## Training Attempt

Training command family:

`cuda_grpo_ablation --mode phase1 --rank 4 --alpha 4 --lr 2e-7 --no-echo`

Environment:

- `KILN_CUDA_ARCHS=86`
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

The run memory-fit and produced two progress lines:

- `progress step=9648/24214 loss=-0.000853 vram_mib=15986`
- `progress step=13272/24214 loss=-0.001178 vram_mib=15968`

After roughly 23 minutes observed wall-clock it had passed only about 55% of
the progress counter, projecting worse than H74 overall. It was manually
terminated before adapter write. No adapter was installed or evaluated.

## Verdict

Rejected as `aborted-throughput`.

Prefix canonicalization reduced action tokens and total steps, but did not
change the practical runtime class. The failure is now clearer: sliced
thinking-on transcript suffixes remain too context-heavy even when assistant
prefixes are shortened. The next experiment should stop slicing long
transcripts and instead collect or synthesize short successful rollouts whose
entire prompt+prefix is already compact.
