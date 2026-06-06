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
  "artifact_schema_version": 1,
  "created_at_utc": "2026-06-06T12:00:00Z",
  "fixture_id": "metal_apple_silicon_matmul_qwen35_4b",
  "backend": "metal",
  "status": "passed",
  "manifest": "docs/backend-latency-fixtures.json",
  "manifest_schema_version": 1,
  "fixture_spec_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  "hardware": "Apple Silicon Metal fixture",
  "source": "crates/kiln-tensor/tests/metal_matmul_bench.rs",
  "command": "/home/ericflo/.cargo/bin/cargo test -p kiln-tensor --features metal --test metal_matmul_bench -- --ignored --nocapture",
  "raw_log": "bench-results/backend-latency/raw/metal-apple-silicon-matmul-qwen35-4b-20260606T120000Z.log",
  "raw_log_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "metrics": {
    "decode_qkv_m1_2560x4096_ms": 1.23
  }
}
```

Required fields:

- `artifact_schema_version`: result artifact schema version, currently `1`
- `created_at_utc`: ISO-8601 UTC timestamp ending in `Z` for when the artifact
  was materialized
- `fixture_id`: exactly matches the fixture `id`
- `backend`: exactly matches the fixture `backend`
- `status`: `passed`
- `manifest`: non-empty manifest path; checked against the validator input path
  when available
- `manifest_schema_version`: exactly matches the fixture manifest
  `schema_version`
- `fixture_spec_sha256`: lowercase SHA-256 hex digest of the stable fixture
  definition (`id`, `backend`, `hardware`, `source`, `command`, metric
  `name`/`unit`/`comparison`, and `selected_cases` when present)
- `hardware`: exactly matches the fixture `hardware`
- `source`: exactly matches the fixture `source`
- `command`: exactly matches the fixture `command`
- `raw_log`: non-empty path or identifier for the captured raw fixture log
- `raw_log_sha256`: lowercase SHA-256 hex digest of the raw fixture log
- `metrics`: object containing every metric named by the fixture, with finite
  numeric values

When the referenced `raw_log` file is present in the checkout, the validator
also checks that its SHA-256 digest matches `raw_log_sha256`.

The fixture digest deliberately excludes metric `max`, `threshold_state`, and
`result_artifact` so a reviewed artifact remains valid while thresholds are
locked from pending to covered. Changing the command, hardware label, source,
metric identities, units, comparisons, or selected cases requires a fresh
hardware artifact.

For each fixture metric, the observed value must be finite numeric and satisfy
its comparison against finite numeric `max`. For example, `comparison: "<="`
requires observed latency to be less than or equal to `max`; `comparison: ">="`
requires observed throughput to be greater than or equal to `max`.

## Metric Log Lines

Fixture benchmarks emit machine-readable metric lines alongside their normal
human-readable output:

```text
KILN_LATENCY_METRIC <metric> <value> <unit>
```

The artifact writer extracts the metric names declared by the selected fixture
and ignores extra metric lines. To run one manifest fixture, capture its raw
log, and materialize the result artifact in one step:

```sh
python3 scripts/run_backend_latency_fixture.py \
  docs/backend-latency-fixtures.json \
  metal_apple_silicon_matmul_qwen35_4b
```

The fixture runner executes the selected fixture `command`, writes a timestamped
raw log under `bench-results/backend-latency/raw`, and then invokes the same
artifact materialization contract used by the standalone writer. If the fixture
has already been run manually, materialize the result artifact from the raw log:

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
`status: "passed"`, match the result artifact schema version, include a valid
UTC creation timestamp, match the fixture `id`, `backend`, manifest schema
version, stable fixture digest, hardware/source/command provenance, and contain
every declared metric. It sets every fixture `threshold_state` to
`locked_threshold`, sets the manifest `status` to `covered`, and applies the
headroom by comparison: `<=` thresholds are raised above observed latency, while
`>=` thresholds are lowered below observed throughput. Use `--check` to validate
without writing.

Then run the covered gate:

```sh
python3 scripts/check_backend_latency_fixtures.py \
  docs/backend-latency-fixtures.json \
  --require-covered
```

Run `python3 scripts/run_backend_latency_fixture.py --self-test` to validate the
fixture-runner capture path without hardware.
Run `python3 scripts/write_backend_latency_result_artifact.py --self-test` to
validate the log-line parser and artifact writer without hardware.
Run `python3 scripts/lock_backend_latency_thresholds.py --self-test` to validate
the threshold-locking logic without hardware.
Run `python3 scripts/check_backend_latency_fixtures.py --self-test` to validate
the artifact-checking logic without hardware.
