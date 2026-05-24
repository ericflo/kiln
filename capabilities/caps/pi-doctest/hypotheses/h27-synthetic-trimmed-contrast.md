# H27 Synthetic Trimmed Contrast

## Hypothesis

H26 showed that local policy-on GRPO can train under a strict length cap, but
the mined same-task pair had tiny saturated reward variance and hurt outcome at
the larger gate. H27 tests a stronger contrast while keeping the same local
shape: one concise passing trace versus one still-passing but inefficient
repair trace per train task.

The intended signal is action-side efficiency without disabling thinking:
prefer read, edit, doctest, done over redundant repair workflows, while keeping
the negative examples outcome-positive so the adapter is not rewarded for
skipping verification.

## Data

- Source: train-only H17 rollouts, not eval tasks or eval transcripts.
- Initial synthetic data: `/tmp/pi-doctest-h27-synthetic-contrast/grpo-train.synthetic-contrast.jsonl`.
- Trimmed data: `/tmp/pi-doctest-h27-synthetic-contrast/grpo-train.trimmed-contrast.jsonl`.
- Groups: 2 train tasks, 4 completions.
- Rewards: high 1.0, low 0.8125 in each group.
- Trimmed low traces: one read, one bad edit, one failing doctest, one fix,
  one passing doctest, `DONE`.

Dry-run token shape:

| completion | seq tokens | action tokens | env tokens |
| --- | ---: | ---: | ---: |
| prime concise | 652 | 150 | 224 |
| prime repair | 706 | 239 | 161 |
| sort concise | 650 | 183 | 189 |
| sort repair | 762 | 266 | 190 |

Total dry-run counts: 838 action tokens, 764 env tokens, 1168 context tokens.

## Training

Command family: `cuda_grpo_ablation --mode baseline`, rank 4 / alpha 8,
learning rate 5e-6, policy loss enabled, ECHO lambda 0.05.

Gradient checkpointing was explicitly enabled:
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. The trainer confirmed 24 checkpoint
segments in the log. Training completed in 167s with observed peak VRAM about
15980 MiB. The receipt reports 154866 ms spent in backward, 5649 ms in
reference forward, and 3067 ms in policy forward.

Adapter: `pi-doctest-h27-trim-contrast-r4a8`.

`kiln adapter verify` passed with 400 nonzero LoRA tensors and delta proxy
0.332205.

## Blind Smoke

`LIMIT=4 SEEDS=1`, paired against `/tmp/pi-doctest-thinking-on-smoke.json`.

| metric | base | H27 |
| --- | ---: | ---: |
| composite | 0.934375 | 0.915625 |
| delta | | -0.018750 |
| outcome | 1.000000 | 1.000000 |
| tested_before_done | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.718750 |
| mean tool calls | 5.25 | 5.75 |
| mean thinking chars | 1966.5 | 2102.75 |
| mean wall-clock s | 35.82 | 36.48 |

## Verdict

Rejected at smoke. The stronger synthetic contrast trained efficiently and
preserved outcome, but the adapter moved tool-call efficiency in the wrong
direction on the blind aggregate. Do not run the larger gate.

Lesson: checkpointed baseline-mode policy GRPO is now viable locally when
completions stay below roughly 800 sequence tokens and 300 action tokens, but
synthetic repair negatives can still teach extra tool use. The next attempt
needs either naturally occurring concise-vs-wasteful train pairs with stronger
reward spread, or a trainer-side loss that directly penalizes unnecessary
action turns without showing verbose repair behavior as the contrast example.
