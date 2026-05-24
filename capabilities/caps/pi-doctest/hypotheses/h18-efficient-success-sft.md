# H18: efficient success SFT

## Hypothesis

H17 showed that local server-native GRPO was still throughput-limited even
after action-token filtering. H18 switched to a gentle SFT anchor: train on
successful train rollouts, preserve thinking-on behavior, but render the
successful workflow more compactly.

The intended signal was not "think less by disabling thinking." It was:

- read `solution.py`;
- make one code-writing edit;
- run doctest;
- finish with `DONE`;
- keep the thinking blocks short and task-directed.

Training was submitted through the SFT API so rank 4 / alpha 8 could be used
with lr `1e-5` and one epoch.

## Attempts

### H18a: edit-style compacted successes

Dataset: `/tmp/pi-doctest-h18-efficient-success-sft/sft.train.jsonl`.

- 9 examples from 3 train tasks.
- Paths normalized to `solution.py`.
- Real tool arguments preserved.
- Char range: 2615 to 3950.

Job `77aa1f88-2db2-4ccd-95a1-196378406e63` reached 56% progress and loss
0.3587, then stayed there until 689s elapsed. It was stopped by restarting the
server.

The likely issue was supervised length: `edit` examples carry both old and new
code inside the assistant action.

### H18b: compact write-style successes

Dataset: `/tmp/pi-doctest-h18-compact-write-sft-cap3100/sft.train.jsonl`.

- 7 examples from the same train source.
- Converted the code-writing step to a single `write` call containing the
  final file.
- Char cap: 3100.

Job `9b9e0590-780a-4dd9-9fc9-d347a20d3bb4` reached 71% progress and loss
0.4627, then stayed there until 580s elapsed. It was stopped by restarting the
server.

## Result

Aborted for throughput, no H18 adapter promoted and no blind eval run.

H18 established that local SFT is viable only when the supervised examples are
kept very short. H19 should keep the write-style rendering but drop the slow
examples with char count above about 2900.

No eval task contents or per-example eval transcripts were inspected.
