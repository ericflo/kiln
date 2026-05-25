# H80 H33/H77 Concat Balanced

## Hypothesis

H79 showed that a TIES blend of H64 and H77 did not rescue the repair-plus-
thinking direction. H80 tested a different composition axis: preserve H33's
hard-negative reliability update and H77's moderate-thinking update as separate
rank blocks using concat merge.

H33 had one larger-gate win before failing confirmation, so it is not a
shipped base. The narrow question was whether rank-concat could keep the useful
parts of H33's reliability behavior while adding H77's better thinking target,
without the tensor-level averaging/interference of a linear or TIES merge.

## Merge

Output adapter:
`pi-doctest-h80-h33-h77-concat-balanced`

Sources:

- `pi-doctest-h33-hardneg-g2-noecho-r4a8`, weight 0.5
- `pi-doctest-h77-moderate-thinking-workflow-r4a4lr1e7`, weight 0.25

The H77 source weight is 0.25 because H33 has `alpha/r = 2` and H77 has
`alpha/r = 1`; concat preserves the first source's scale in the output config,
so the lower H77 source weight keeps the intended effective update balance.

Merge command used `/v1/adapters/merge` with `mode="concat"`. The merge wrote
400 tensors and produced rank 8 / alpha 16.

## Verification

Adapter verify passed:

| metric | value |
| --- | ---: |
| rank | 8 |
| alpha | 16 |
| alpha / rank | 2.0 |
| tensor count | 400 |
| projection pairs | 200 |
| nonzero tensors | 400 |
| adapter size bytes | 61001600 |
| LoRA update proxy | 0.181218 |
| adapter hash | `sha256:6f7a00979a7b1bb0411f32246b64944e9c9de6e238065c2373be5d669308909e` |

## Blind Gate

Cheap smoke, `LIMIT=4 SEEDS=1`:

| metric | base | H80 |
| --- | ---: | ---: |
| composite | 0.925000 | 0.925000 |
| zero rollouts | 0 | 0 |
| mean wall-clock s | 44.33 | 49.07 |

## Verdict

Rejected at smoke.

H80 tied paired base on composite and zero count, but was slower. Preserving
H33 and H77 as separate rank blocks did not recover H33's one-off reliability
win and did not add measurable thinking efficiency. This closes the immediate
adapter-composition branch around H33/H64/H77. The next useful H81-style
experiment should build genuinely new data: broader train-only behavior
anchors, a stronger teacher/OPD signal, or a fresh real-reward source.
