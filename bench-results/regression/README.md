# Perf-regression baselines (#1077 Tier 2)

JSON baselines for the nightly A6000 perf-regression workflow
(`.github/workflows/perf-regression-nightly.yml`). Each file pins
`secs_per_step` and `peak_vram_mb` for one `(workload, trainer, gpu)`
cell of the regression matrix. The nightly run gates against these.

## Schema

```json
{
  "schema_version": 1,
  "workload": "sft_short",
  "trainer": "native",
  "gpu": "NVIDIA RTX A6000",
  "secs_per_step": null,
  "peak_vram_mb": null,
  "comment": "placeholder — first nightly run seeds this via --write-baseline-if-null",
  "pinned_at_commit": ""
}
```

`null` values mark "no baseline pinned yet" — the first nightly run on a
freshly-added workload row populates them via
`check_sft_train_regression.py --write-baseline-if-null`. Subsequent
runs compare observed numbers against the seeded values and fail the
workflow if the regression exceeds the configured tolerance.

## How to add a workload

1. Add a new placeholder JSON in this directory with the desired
   `workload` / `trainer` / `gpu` triple and `null` perf fields.
2. Add a matching row to the nightly workflow's job matrix.
3. The next nightly run seeds the baseline.
4. Subsequent runs gate.

## How to update a baseline after intentional perf changes

A planned perf change (e.g. a kernel fusion that legitimately moves
`secs_per_step` 20% down) must update the baseline alongside the code
change. The intended workflow:

1. Run the bench on your branch:
   ```bash
   cargo run --release --bin kiln-bench --features cuda -- \
     --model-path /path/to/qwen3.5-4b \
     --training-steps 5 \
     2>&1 | tee /tmp/kiln_bench.txt
   ```
2. Extract the JSON from stdout and copy the `training.secs_per_step` /
   `training.peak_vram_mb` numbers into the appropriate baseline JSON.
3. Add a `comment` field explaining the regression (e.g. `"+22%
   secs_per_step traded for -50% peak_vram on long-context — see PR
   #NNNN"`).
4. Commit the baseline JSON change alongside the code change.

## How to update a baseline after the nightly catches a real regression

1. Investigate why it regressed.
2. If the regression is intentional, follow the steps above to pin the
   new baseline.
3. If the regression is a bug, revert the code change. The nightly
   stops failing once `secs_per_step` returns within tolerance.

## Tolerances

| Field          | Default tolerance | Rationale                                            |
|----------------|-------------------|------------------------------------------------------|
| `secs_per_step`| ±10%              | A6000 + isolated pod is stable to ~3-5% run-to-run.  |
| `peak_vram_mb` | ±15%              | VRAM allocator churn is noisier than step time.      |

Per-workload tolerances can be overridden via workflow inputs.
