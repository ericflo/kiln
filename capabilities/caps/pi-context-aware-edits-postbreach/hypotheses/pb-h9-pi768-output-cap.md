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

Rejected. The 768-token cap reduced thinking cost but also reduced capability:

- composite: 0.2475 (`delta=-0.0521`, stdev 0.4164)
- `outcome`: 0.4583
- `format_compliance`: 0.5417
- `convention_consistency`: 0.9750
- `read_before_edit`: 1.0000
- `no_redundant_imports`: 1.0000
- `no_style_drift`: 1.0000
- efficiency: 5.90 tool calls/rollout, 1874.9 thinking chars, 306.8 thinking
  chars/tool call

This confirms that a lower output cap can recover baseline-like thinking
efficiency, but it cuts off enough reasoning/finalization to miss the outcome
and format gates. Keep the 1024-token alias for score-seeking runs.
