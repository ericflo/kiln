# H45: post-read edit workflow

## Hypothesis

H44 showed that complete natural repair trajectories are too expensive for
local policy-on training. H45 keeps the complete workflow target but compresses
the action surface: after the real `read solution.py` context, prefer a compact
`edit` -> doctest -> `DONE` completion over two rejected variants:

- `edit` -> `DONE` without verification;
- `edit` -> doctest -> redundant doctest -> `DONE`.

This tests whether a short post-read complete-workflow contrast can preserve
task-solving behavior while teaching efficient verification and stopping.

## Data

Dataset:
`/tmp/pi-doctest-h45-postread-workflow-grpo/grpo-train.postread-workflow-edit.g3.jsonl`.

Source:
`/tmp/pi-doctest-h44-broad-success-sft/sft.train.jsonl`.

The data used the first three compact H44 success examples. Each group kept
the real train-only system/user prompt, the read action, and the real
`solution.py` read result in context. The generated completions converted the
H44 full-file `write` action into a minimal `edit` that replaces
`raise NotImplementedError`.

Dry-run shape:

- 3 groups.
- 9 completions.
- Rewards per group: 1.0, 0.0, 0.55.
- 1146 action tokens.
- 0 env tokens.
- 3165 context tokens.
- Reward stdev: 0.408928.
- All groups passed the variance filter.

## Training

Adapter:
`pi-doctest-h45-postread-edit-g3-noecho-r4a4lr1e6`.

Training command used rank 4 / alpha 4 / lr `1e-6`, Phase 1 GRPO, no ECHO,
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`, and `--adapter-smoke-test`.

Training completed successfully:

- elapsed: 356.717s observed;
- receipt wall clock: 352913 ms;
- peak observed VRAM: 15941 MiB;
- groups trained: 3;
- completions trained: 9;
- final logged loss: -0.037168.

Adapter verify passed. The verifier found 400 nonzero LoRA tensors and LoRA
update L2 upper-bound 0.079543.

## Smoke

Blind aggregate `LIMIT=4 SEEDS=1` smoke rejected H45.

Paired base:

- Composite: 0.9625.
- Mean wall-clock: 23.8416s.
- Zero rollouts: 0.

H45:

- Composite: 0.86875.
- Delta: -0.09375.
- Mean wall-clock: 65.4882s.
- Zero rollouts: 0.

No promotion check was run because the smoke gate was negative and slower than
base.

## Verdict

Reject H45.

Edit-form compression solved the local throughput problem, but the data signal
was still harmful. The adapter likely overfit the post-read edit workflow and
made blind eval rollouts slower. The failure is different from H44: H45 can
train locally, but this particular no-test/retest contrast is not a safe
adapter update.

Next attempts should keep the edit-form token discipline but change the signal:
use outcome-preserving positive pairs that differ in thinking/tool count only
after full verification, or alternate edit-form workflow rows with anchor rows
that explicitly preserve broad task-solving reliability. Do not expand back to
natural repair traces unless negative completions stay under the H45 token
budget.

No eval task contents or per-example eval transcripts were inspected.
