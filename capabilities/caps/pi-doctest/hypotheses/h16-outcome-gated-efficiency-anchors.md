# H16: outcome-gated efficiency anchors

## Hypothesis

H15 improved tool-call efficiency but harmed outcome because the lean
high-variance data still contained failure-heavy groups. H16 should preserve
outcome by training only on outcome-perfect, non-timeout train completions,
then ranking those successful completions by tool-call and thinking efficiency.

Reward reshape:

```
reward = 0.5 + 0.25 * tool_call_efficiency
       + 0.25 * (1 - min(thinking_chars, 7000) / 7000)
```

Completions with `outcome < 1.0` or nonzero timeout exit are excluded. The
server is kept in thinking-enabled mode; this is not a no-thinking recipe.

## Result

Aborted for throughput, no adapter promoted.

Attempts:

- `e96d1172-ae6c-4f20-a3ca-b74282fbd4b9`: rank 8 failed preflight because the
  train CLI defaulted alpha to 32, giving unsafe alpha/rank 4.
- `f1ac29f5-5667-4dba-87e6-4edec18dd24a`: rank 16 full anchor batch
  selected 3 groups / 11 completions / 6984 action tokens / 3927 env tokens.
  It hit a pathological 2795-token selected completion and was stopped by
  restarting the server.
- `b357b336-a106-4041-93af-7ce5e2b54a5a`: rank 16 short anchor batch applied a
  3000-character completion cap, leaving 3 groups / 10 completions / 4827
  action tokens / 3538 env tokens. It still spent too long on a
  1480-token / 830-action-token completion and was stopped by restarting the
  server.

The data direction remains plausible: it removes H15's short-failure reward
path and trains only among successful traces. The implementation constraint is
now sharper: before another server-native GRPO submission, build the dataset
from `kiln trajectory inspect --json` and filter by per-completion action
tokens, not by text length. A practical next cutoff is a few hundred action
tokens per completion.

No blind eval was run because no H16 adapter completed training.
