# H17: action650 success anchors

## Hypothesis

H16 showed that text length was the wrong throughput guard. H17 keeps the
outcome-gated anchor idea, but builds the GRPO batch from
`kiln trajectory inspect --json` token diagnostics and rejects any selected
completion with more than 650 action tokens.

Reward reshape is narrower than H16, because every retained completion is
already outcome-perfect and non-timeout:

```
reward = 0.75 + 0.125 * tool_call_efficiency
       + 0.125 * (1 - min(thinking_chars, 7000) / 7000)
```

The server stays in thinking-enabled mode. The point is not to disable
thinking, but to keep successful thinking traces short enough to train.

## Dataset

Built from the H15 partial32 train rollouts.

- Candidate successful completions: 12.
- Selected groups: 2.
- Selected completions: 7.
- Action-token cap: 650 per completion.
- Total action tokens: 2675.
- Total env tokens: 2341.
- Trajectory schema warnings: none.

Selected groups:

- `task_0025`: 4 completions, 1161 action tokens, reward range
  0.9854464286 to 0.9871964286.
- `task_0041`: 3 completions, 1514 action tokens, reward range
  0.953625 to 0.9814821429.

`task_0033` was excluded because only one successful completion survived the
650 action-token cap, which is not enough for an in-group preference signal.

## Result

Aborted for throughput, no adapter promoted and no blind eval run.

Training job `d3865165-d4d6-42c8-bb29-139fd9caad9c` used rank 16 with the
default alpha 32 to keep alpha/rank at 2. It reached 44% progress and loss
0.7438, then stayed on the second selected group until elapsed time reached
587s. GPU utilization was 100%, so this was active but too slow for a
low-signal two-group dataset. The server was restarted after the abort.

The cap helped, but not enough: the second group still contains completions at
585 and 590 action tokens. Another server-native GRPO attempt should either
cap individual completions closer to 350 action tokens or switch methods
instead of spending GPU time on tiny, slow preference groups.

No eval task contents or per-example eval transcripts were inspected.
