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

Run `python3 scripts/check_backend_latency_fixtures.py --self-test` to validate
the artifact-checking logic without hardware.
