# H3: Strict Workflow Prompt Diagnostic

## Hypothesis

H1 and H2 both damaged `outcome` while trying to move the dominant
`format_compliance`/`outcome` headroom through weights. Before another adapter
run, test whether a stricter runtime contract can improve those sub-scores
directly: read first, make a minimal edit, verify, use relative paths, and end
with a single sentence naming the modified file and preserved conventions.

If prompt-only base lifts materially, the next training attempt should distill
prompt-conditioned train rollouts or synthesize idealized traces. If it does
not lift, the next attempt should skip prompt distillation and build idealized
train-only SFT traces instead.

## Recipe

- No adapter; base Qwen3.5-4B only.
- Config: `configs/h3-strict-workflow.config.json`.
- Keep thinking enabled through the server default.
- First run a train-task aggregate smoke to verify the prompt does not break
  the workflow.
- Promotion-quality diagnostic: full 12-task x 3-seed blind aggregate eval
  through `capability.oracle.sh` with `CONFIG` pointing to the H3 config.

## Falsification

Reject prompt distillation if the full blind eval shows any of:

- composite lift < +0.05 versus `baseline-0`;
- `outcome` drops versus baseline;
- `format_compliance` fails to improve;
- mean thinking chars/tool call increases by more than 25% over baseline.

## Result

Status: rejected.

- Full 12-task x 3-seed blind aggregate eval scored 0.2778 versus the
  0.4800 baseline (`delta=-0.2022`, stdev 0.4479).
- The prompt reduced cost: mean tool calls fell from 5.00 to 4.03, mean
  thinking chars fell from 1488.5 to 1165.2, and thinking chars/tool fell
  from 302.7 to 284.2.
- The target sub-scores regressed. `format_compliance` fell from 0.6528 to
  0.4861 and `outcome` fell from 0.6944 to 0.6111. Nonzero rollouts fell
  from 18 to 10.

Conclusion: strict prompting is an efficiency lever, not a capability lever
for this cap. It narrows the workflow enough to reduce thinking and tool use,
but it also suppresses the final-response contract and does not improve edit
completion. Do not distill this prompt. Move to idealized train-only traces
that preserve completion while teaching compact workflow shape.
