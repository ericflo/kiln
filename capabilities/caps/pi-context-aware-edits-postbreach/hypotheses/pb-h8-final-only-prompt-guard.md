# PB-H8: Remove Final-Only Prompt Guard

## Hypothesis

The default postbreach config already appends a final-only prompt guard that
tells Pi to name the modified file and preserved convention. That may improve
`format_compliance`, but PB-H3 showed that even light workflow wording can hurt
`outcome`. A no-adapter control with no appended system prompt tests whether
the existing final-only guard is actually worth keeping as the baseline for
future adapter data.

## Recipe

- Source: no new training data.
- Adapter: none.
- Config: `configs/pb-h8-no-final-prompt.config.json`, identical to the
  default config except it removes `pi_append_system_prompt`.
- Keep thinking enabled through the server default.

## Falsification

Reject if any of:

- composite lift < +0.05 versus postbreach baseline;
- `outcome < 0.50`;
- `format_compliance < 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- server-side eval errors or timeouts occur.

## Result

Pending.
