# Backend Latency Result Artifact Schema

Hardware latency fixtures remain `fixture_required` until every backend fixture
has a checked result artifact, locked numeric thresholds, and passing measured
metrics. The validator is `scripts/check_backend_latency_fixtures.py`.

## Pending Fixtures

Pending fixtures live in `docs/backend-latency-fixtures.json` with:

- `threshold_state`: `pending_fixture_result`
- metric `max`: `null`
- `status`: `fixture_required`

This state is valid for local and default-feature CI. It must not mark the
hardware latency conformance gate covered.

## Covered Fixtures

To cover the gate, each fixture must:

- set `threshold_state` to `locked_threshold`
- set every metric `max` to a numeric threshold
- write the referenced `result_artifact`
- set the manifest `status` to `covered`
- pass `python3 scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json --require-covered`

The result artifact is JSON:

```json
{
  "fixture_id": "metal_apple_silicon_matmul_qwen35_4b",
  "backend": "metal",
  "status": "passed",
  "metrics": {
    "decode_qkv_m1_2560x4096_ms": 1.23
  }
}
```

Required fields:

- `fixture_id`: exactly matches the fixture `id`
- `backend`: exactly matches the fixture `backend`
- `status`: `passed`
- `metrics`: object containing every metric named by the fixture

For each fixture metric, the observed value must be numeric and satisfy its
comparison against `max`. For example, `comparison: "<="` requires observed
latency to be less than or equal to `max`; `comparison: ">="` requires observed
throughput to be greater than or equal to `max`.

## Metric Log Lines

Fixture benchmarks emit machine-readable metric lines alongside their normal
human-readable output:

```text
KILN_LATENCY_METRIC <metric> <value> <unit>
```

The artifact writer extracts the metric names declared by the selected fixture
and ignores extra metric lines. Capture a hardware fixture run with `tee`, then
materialize the result artifact:

```sh
python3 scripts/write_backend_latency_result_artifact.py \
  docs/backend-latency-fixtures.json \
  metal_apple_silicon_matmul_qwen35_4b \
  /path/to/raw-benchmark.log
```

By default the script writes the fixture's `result_artifact`; use `--output` for
a scratch artifact.

## Locking Thresholds

After reviewing the hardware result artifacts, lock numeric thresholds in the
manifest with explicit headroom:

```sh
python3 scripts/lock_backend_latency_thresholds.py \
  docs/backend-latency-fixtures.json \
  --headroom 0.10
```

The threshold locker requires every fixture result artifact to exist, have
`status: "passed"`, match the fixture `id` and `backend`, and contain every
declared metric. It sets every fixture `threshold_state` to `locked_threshold`,
sets the manifest `status` to `covered`, and applies the headroom by comparison:
`<=` thresholds are raised above observed latency, while `>=` thresholds are
lowered below observed throughput. Use `--check` to validate without writing.

Then run the covered gate:

```sh
python3 scripts/check_backend_latency_fixtures.py \
  docs/backend-latency-fixtures.json \
  --require-covered
```

Run `python3 scripts/write_backend_latency_result_artifact.py --self-test` to
validate the log-line parser and artifact writer without hardware.
Run `python3 scripts/lock_backend_latency_thresholds.py --self-test` to validate
the threshold-locking logic without hardware.
Run `python3 scripts/check_backend_latency_fixtures.py --self-test` to validate
the artifact-checking logic without hardware.
