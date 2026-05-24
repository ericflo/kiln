# PB-H9: 768-Token Thinking Cap

## Hypothesis

Thinking is useful for this cap, but the 1024-token Pi alias may leave too much
room for verbose loops. A 768-token per-turn output cap might preserve the
final-only prompt guard's format benefit while reducing thinking characters and
tool-call wandering. This is a runtime/data-collection control before the next
adapter chain: if it improves or preserves blind score with lower thinking
cost, future train rollouts should use the 768 alias.

## Recipe

- Source: no new training data.
- Adapter: none.
- Config: `configs/pb-h9-pi768-final-prompt.config.json`.
- Pi model alias: `qwen-3.5-4b-kiln-pi768`, same local Kiln endpoint with
  `maxTokens=768`.
- Keep the default final-only appended prompt.
- Keep thinking enabled through the server default.

## Falsification

Reject if any of:

- composite < postbreach baseline;
- `outcome < 0.50`;
- `format_compliance < 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call >= baseline (`>=308.8`);
- server-side eval errors or timeouts occur.

## Result

Pending.
