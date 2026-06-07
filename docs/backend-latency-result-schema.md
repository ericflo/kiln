# Backend Latency Result Artifact Schema

Hardware latency fixtures remain `fixture_required` until every backend fixture
has a checked result artifact, locked numeric thresholds, and passing measured
metrics. The validator is `scripts/check_backend_latency_fixtures.py`.

## Pending Fixtures

Pending fixtures live in `docs/backend-latency-fixtures.json` with:

- `schema_version`: `1`
- `threshold_state`: `pending_fixture_result`
- metric `max`: `null`
- `status`: `fixture_required`
- `policy.covered_gate_requires`: the checked policy list explaining why the
  hardware latency gate cannot be covered by local/default-feature checks alone

This state is valid for local and default-feature CI. It must not mark the
hardware latency conformance gate covered.

Fixture `command` strings are part of the stable fixture digest used by
reviewed result artifacts. Keep them runner-portable, such as `cargo ...` with
environment variables or fixture arguments for local model paths, and do not
bake a developer-local Cargo installation path into the manifest.

## Covered Fixtures

To cover the gate, each fixture must:

- set `threshold_state` to `locked_threshold`
- set every metric `max` to a numeric threshold
- write the referenced repo-relative `result_artifact` under
  `bench-results/backend-latency` with a `.json` extension
- track the `result_artifact` and referenced `raw_log` in git
- set the manifest `status` to `covered`
- pass `python3 scripts/check_backend_latency_fixtures.py docs/backend-latency-fixtures.json --require-covered`

A manifest with `status: "covered"` is intentionally rejected unless the
checker is run with `--require-covered`, so default-feature local checks cannot
mark the hardware latency gate covered without exercising the strict artifact
contract.

The result artifact is JSON:

```json
{
  "artifact_schema_version": 3,
  "created_at_utc": "2026-06-06T12:00:00Z",
  "fixture_id": "metal_apple_silicon_matmul_qwen35_4b",
  "backend": "metal",
  "status": "passed",
  "manifest": "docs/backend-latency-fixtures.json",
  "manifest_schema_version": 1,
  "fixture_spec_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
  "hardware": "Apple Silicon Metal fixture",
  "source": "crates/kiln-tensor/tests/metal_matmul_bench.rs",
  "source_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  "command": "cargo test -p kiln-tensor --features metal --test metal_matmul_bench -- --ignored --nocapture",
  "raw_log": "bench-results/backend-latency/raw/metal-apple-silicon-matmul-qwen35-4b-20260606T120000Z.log",
  "raw_log_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "git_commit": "1111111111111111111111111111111111111111",
  "git_tracked_dirty": false,
  "metrics": {
    "decode_qkv_m1_2560x4096_ms": 1.23
  }
}
```

Required fields:

- `artifact_schema_version`: result artifact schema version, currently `3`
- `created_at_utc`: ISO-8601 UTC timestamp ending in `Z` for when the artifact
  was materialized
- `fixture_id`: exactly matches the fixture `id`
- `backend`: exactly matches the fixture `backend`
- `status`: `passed`
- `manifest`: non-empty repo-relative manifest path; checked against the
  validator input path when available
- `manifest_schema_version`: exactly matches the fixture manifest
  `schema_version`
- `fixture_spec_sha256`: lowercase SHA-256 hex digest of the stable fixture
  definition (`id`, `backend`, `hardware`, `source`, `command`, metric
  `name`/`unit`/`comparison`, and `selected_cases` when present)
- `hardware`: exactly matches the fixture `hardware`
- `source`: exactly matches the fixture `source`
- `source_sha256`: lowercase SHA-256 hex digest of the fixture source file
- `command`: exactly matches the fixture `command`
- `raw_log`: non-empty repo-relative path for the captured raw fixture log;
  covered fixtures require this file to live under
  `bench-results/backend-latency/raw`, use a `.log` extension, and exist when
  validated
- `raw_log_sha256`: lowercase SHA-256 hex digest of the raw fixture log
- `git_commit`: lowercase 40-character git commit object for the checkout that
  captured the artifact
- `git_tracked_dirty`: boolean reporting whether tracked files were dirty when
  the artifact was materialized; covered fixtures require this to be `false`
- `metrics`: object containing exactly every metric named by the fixture, with
  finite numeric values

Covered result artifacts must not contain additional top-level keys, and
`metrics` must not contain undeclared metric names. Additive schema changes
should bump `artifact_schema_version` and update the validator.

When `--require-covered` is set, the validator requires the fixture
`result_artifact`, fixture `source`, result `manifest`, and result `raw_log`
paths to be repo-relative. It also requires fixture `result_artifact` paths to
live under `bench-results/backend-latency` with a `.json` extension, result
`raw_log` paths to live under `bench-results/backend-latency/raw` with a `.log`
extension, the result artifact and raw log to be tracked by git, and the
referenced source and `raw_log` files to exist in the checkout and checks that
their SHA-256 digests match
`source_sha256` and `raw_log_sha256`. It requires `git_commit` to be a
lowercase 40-character commit that exists in the local repository, requires the
fixture source to exist at `git_commit`, requires `source_sha256` to match the
source bytes at that commit, and requires `git_tracked_dirty` to be `false` for
covered validation. It then re-parses the
raw log `KILN_LATENCY_METRIC` lines and requires every declared artifact metric
value and unit to match the raw log. It rejects unknown artifact keys and
undeclared artifact metrics.

The fixture digest deliberately excludes metric `max`, `threshold_state`, and
`result_artifact` so a reviewed artifact remains valid while thresholds are
locked from pending to covered. Changing the command, hardware label, source
path, source file content, metric identities, units, comparisons, or selected
cases requires a fresh hardware artifact.

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
and ignores extra metric lines. Covered validation later re-parses the same raw
log and rejects artifacts whose metric values or units do not match the captured
`KILN_LATENCY_METRIC` lines; artifact metrics must match the raw log. To run
one manifest fixture, capture its raw log, and materialize the result artifact
in one step:

```sh
python3 scripts/run_backend_latency_fixture.py \
  docs/backend-latency-fixtures.json \
  metal_apple_silicon_matmul_qwen35_4b
```

The fixture runner executes the selected fixture `command`, writes a timestamped
raw log under `bench-results/backend-latency/raw`, and then invokes the same
artifact materialization contract used by the standalone writer. If the fixture
has already been run manually, materialize the result artifact from the raw log.
Covered artifacts should use a raw log checked into the repository, such as one
captured by the fixture runner under `bench-results/backend-latency/raw`:

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

The threshold locker requires manifest `schema_version: 1`, a recognized
manifest `status`, and `required_backends` to be a non-empty array of valid
backend names. It refuses to lock until every required backend has at least one
fixture. It also requires every fixture result artifact path to be repo-relative,
live under `bench-results/backend-latency` with a `.json` extension, exist, have
`status: "passed"`, match the result artifact schema version, include a valid
UTC creation timestamp, match the fixture `id`, `backend`, manifest schema
version, stable fixture digest,
hardware/source/command provenance, source file digest, and contain every
declared metric with no unknown artifact keys or undeclared metrics. It also
requires the fixture `source`, result `manifest`, and result `raw_log` paths to
be repo-relative, result `raw_log` to live under
`bench-results/backend-latency/raw` with a `.log` extension, and the referenced
source and raw log files to exist and match `source_sha256`/`raw_log_sha256`.
It requires `git_commit` to be a lowercase 40-character commit that exists in
the local repository, requires the fixture source to exist at `git_commit`,
requires `source_sha256` to match the source bytes at that commit, and requires
`git_tracked_dirty` to be `false` before thresholds can lock. It re-parses the
raw log and requires each
artifact metric value and unit to match before deriving thresholds. It sets every
fixture `threshold_state` to
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
