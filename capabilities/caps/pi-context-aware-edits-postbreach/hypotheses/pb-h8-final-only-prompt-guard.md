# PB-H8: Final-Only Prompt Guard

## Hypothesis

All postbreach adapter attempts have either preserved conventions while hurting
outcome/format, or improved one terminal behavior while damaging the edit
contract. A zero-training prompt guard that only constrains the final response
may improve `format_compliance` without changing the read/edit/verify behavior
that drives `outcome`.

## Recipe

- Source: no new training data.
- Adapter: none.
- Config: default task prompt plus one final-only instruction: after the edit is
  complete, the final response must name the modified file and one preserved
  local convention.
- Do not add strict read/edit/verify sequencing language; PB-H3 showed that
  workflow prompting can damage outcome.
- Keep thinking enabled through the server default.

## Falsification

Reject if any of:

- composite lift < +0.05 versus postbreach baseline;
- `outcome < 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.95`;
- thinking chars/tool call > 25% above baseline (`>386`);
- server-side eval errors or timeouts occur.

## Result

Pending.
