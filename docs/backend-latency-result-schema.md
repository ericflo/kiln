# Legacy Backend Latency Evidence

> **Use this pipeline only to maintain or retire an existing fixture.** It
> predates structured local qualification and keeps a compact result tied to a
> separately retained raw log. New correctness, performance, capacity, and
> endurance evidence belongs in `scripts/qualification/`; see
> [Local hardware qualification](qualification.md).

A legacy latency artifact answers one narrow question: did a named
microbenchmark, on its named hardware and source, satisfy its own reviewed
threshold? It is not a qualification receipt, a cross-device benchmark, a
backend support gate, or a product default.

## Current Legacy Gate State

`docs/backend-latency-fixtures.json` currently reports
`status: "fixture_required"`.

| Backend | Fixture state | Meaning |
| --- | --- | --- |
| CUDA | One locked RTX 4090 matmul fixture | Its own reviewed metrics have numeric thresholds |
| ROCm | One locked gfx1151 matmul fixture | Its own reviewed metrics have numeric thresholds |
| Metal | Two Apple Silicon fixtures pending | Result files exist, but reviewed thresholds are not locked |
| Vulkan | One Strix Halo decode fixture pending | This machine-specific regression fixture is incomplete |

The Vulkan fixture’s device name and self-hosted runner label identify where
that old measurement must run. They do **not** restrict Kiln’s Vulkan runtime
to Strix Halo, select a Vulkan device in product code, or define support for
other Vulkan-capable devices.

Check the non-strict manifest state with:

```bash
python3 scripts/check_backend_latency_fixtures.py \
  docs/backend-latency-fixtures.json
```

The strict command fails until every required fixture has a reviewed artifact
and locked thresholds:

```bash
python3 scripts/check_backend_latency_fixtures.py \
  docs/backend-latency-fixtures.json \
  --require-covered
```

## State Machine

| Manifest state | Required fixture fields | What CI may claim |
| --- | --- | --- |
| `fixture_required` | At least one fixture still uses `pending_fixture_result`, with metric `max: null` | Default checks may validate the manifest, but may not claim the legacy gate is covered |
| `covered` | Every fixture uses `locked_threshold`, every metric has a finite numeric threshold, and every result passes strict validation | The legacy fixture set is internally covered |

For a pending fixture:

- `threshold_state` is `pending_fixture_result`;
- each metric `max` is `null`; and
- the manifest remains `fixture_required`.

For a covered fixture:

- `threshold_state` is `locked_threshold`;
- each metric has a reviewed finite threshold;
- `result_artifact` names a tracked JSON file below
  `bench-results/backend-latency`; and
- the ignored or externally retained raw log is bound by digest.

“Covered” means this legacy fixture contract is complete. It does not mean the
backend is fast on every device, or even that an end-to-end model server met an
SLO.

## Result Artifact

Artifact schema version 3 is a closed JSON object:

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

| Field | Meaning and validation |
| --- | --- |
| `artifact_schema_version` | Must be `3` |
| `created_at_utc` | UTC ISO-8601 creation time ending in `Z` |
| `fixture_id`, `backend`, `hardware`, `source`, `command` | Must exactly match the selected manifest fixture |
| `manifest` | Non-empty repository-relative manifest path; checked against the validator input when available |
| `manifest_schema_version` | Must match the manifest’s schema version |
| `fixture_spec_sha256` | SHA-256 of the stable fixture definition |
| `source_sha256` | SHA-256 of the fixture source bytes |
| `raw_log`, `raw_log_sha256` | Repository-relative retention reference and digest for the captured log |
| `git_commit` | Lowercase 40-hex commit that must exist locally during strict validation |
| `git_tracked_dirty` | Historical field name for a clean-checkout marker; `false` is required for coverage |
| `status` | Must be `passed` |
| `metrics` | Exactly the declared metric names, each with a finite numeric value |

The writer computes `git_tracked_dirty` from
`git status --porcelain --untracked-files=all`. Despite its name, tracked
changes and new untracked repository files both make the artifact dirty.

Unknown top-level keys and undeclared metrics are rejected. Additive artifact
changes require a schema-version bump and validator update.

## Fixture Identity

`fixture_spec_sha256` covers:

- fixture ID and backend;
- hardware label;
- source path;
- command;
- metric name, unit, and comparison; and
- selected cases, when present.

It deliberately excludes metric thresholds, `threshold_state`, and
`result_artifact`. This lets reviewers ingest a measurement first and lock
headroom afterward. Changing the command, hardware label, source, source
bytes, metric identity, unit, comparison, or selected cases requires a new
hardware artifact.

Fixture command strings are part of this digest. Keep them runner-portable:
use `cargo ...`, declared environment variables, or explicit fixture arguments.
Do not store a developer-local Cargo path or a placeholder model path.

## Metric Boundary

Benchmarks emit machine-readable lines alongside normal output:

```text
KILN_LATENCY_METRIC <metric> <value> <unit>
```

The writer extracts only metrics declared by the selected fixture. Import and
threshold locking reparse the raw log and require every retained value and
unit to match.

Each finite observed value is tested against its own fixture threshold:

- `comparison: "<="` means the value must be at most `max`; and
- `comparison: ">="` means the value must be at least `max`.

Valid conclusions stay within one fixture definition. Do not compare two
artifacts as interchangeable when their hardware, command, source, selected
cases, metric definition, or fixture digest differs. These microbenchmark
numbers are not end-to-end tokens per second or request latency.

## Trust Boundary

Strict validation checks that the manifest, compact artifact, source, commit,
raw-log digest, parsed metrics, and thresholds agree. It does not prove that:

- the named hardware was honestly reported;
- the host or artifact producer was authenticated;
- an external or missing raw log still exists;
- the artifact is signed;
- another device will reproduce the measurement; or
- a model server will achieve corresponding throughput or latency.

At import and threshold-lock time, the raw log must exist below
`bench-results/backend-latency/raw`, remain ignored, match its SHA-256 digest,
and reparse to the retained metrics. After locking, the raw log may be absent
from a clean checkout if it remains in external workflow storage. When a local
copy is present, strict validation checks it again.

The compact result must be tracked under `bench-results/backend-latency` with a
`.json` extension. The fixture source must match both the working tree and the
recorded commit. The receipt has no independent signature or chain of custody.

## Maintain One Existing Fixture

Run the selected command, capture its log, and write the artifact in one step:

```bash
python3 scripts/run_backend_latency_fixture.py \
  docs/backend-latency-fixtures.json \
  metal_apple_silicon_matmul_qwen35_4b
```

To materialize from an already captured log:

```bash
python3 scripts/write_backend_latency_result_artifact.py \
  docs/backend-latency-fixtures.json \
  metal_apple_silicon_matmul_qwen35_4b \
  /absolute/path/to/raw-benchmark.log
```

By default, the writer uses the fixture’s `result_artifact`. Use `--output` for
a scratch artifact.

If a manual GitHub Actions run produced a downloadable artifact, import its zip
or extracted directory:

```bash
python3 scripts/import_backend_latency_artifact.py \
  /absolute/path/to/downloaded-artifact.zip \
  --fixture-id metal_apple_silicon_matmul_qwen35_4b
```

The importer validates record shape, fixture and source identity, raw-log
digest, parsed values, commit identity, and the clean-checkout marker. It does
not require locked thresholds. If canonical files already contain different
bytes, use `--force` only after review has chosen the replacement run.

Commit the compact result and reviewed manifest change. Do not force-add the
ignored raw log.

## Manual Workflow Compatibility

The manual-only `Perf regression nightly` workflow is an unscheduled
compatibility handoff. Pull requests do not run it, and its result is not
structured qualification evidence.

List the remaining fixture work:

```bash
python3 scripts/plan_backend_latency_fixture_dispatch.py
```

The JSON plan reports whether site-local self-hosted runner labels are missing
and supplies dispatch, download, import, lock, and strict-check commands.
`runner_labels` is CI routing metadata; it is excluded from the fixture digest
and has no product-runtime meaning.

Use `--check-runners` to query GitHub runner availability through `gh api`.
Use `--shell` only when every selected fixture has labels:

```bash
python3 scripts/plan_backend_latency_fixture_dispatch.py \
  --fixture-id cuda_rtx4090_matmul_qwen35_4b \
  --shell
```

## Lock Reviewed Thresholds

After reviewing the artifact and raw log, add explicit headroom:

```bash
python3 scripts/lock_backend_latency_thresholds.py \
  docs/backend-latency-fixtures.json \
  --headroom 0.10 \
  --fixture-id metal_apple_silicon_matmul_qwen35_4b
```

The locker:

- validates the manifest, result, source, commit, clean-checkout marker, raw
  log, parsed metrics, and units;
- raises `<=` limits above the observed value;
- lowers `>=` floors below the observed value;
- leaves unselected fixtures pending; and
- changes the manifest to `covered` only when every fixture is locked.

Use `--check` to preview validation without writing. Then run the strict gate:

```bash
python3 scripts/check_backend_latency_fixtures.py \
  docs/backend-latency-fixtures.json \
  --require-covered
```

## Failure Triage

| Failure | Inspect |
| --- | --- |
| Missing metric | Required `KILN_LATENCY_METRIC` line, spelling, and unit |
| Fixture digest mismatch | Command, hardware label, source, metrics, or selected cases changed |
| Source digest mismatch | Working-tree source differs from the captured or committed source |
| Dirty marker rejected | Tracked changes or untracked repository files existed at capture |
| Raw-log mismatch | Wrong log, changed bytes, or retained metrics differ from reparsing |
| Commit rejected | Commit is malformed, unavailable locally, or lacks the fixture source |
| Threshold failure | Observed value violates its own comparison and locked limit |
| Strict gate remains incomplete | At least one required backend fixture is still pending |

Do not resolve a fixture failure by copying its machine identity into portable
backend code. Fix the implementation, recapture the same regression fixture,
or replace the legacy fixture through review.

## Tooling Self-Tests

These checks exercise parsing and workflow mechanics without hardware:

```bash
python3 scripts/run_backend_latency_fixture.py --self-test
python3 scripts/write_backend_latency_result_artifact.py --self-test
python3 scripts/import_backend_latency_artifact.py --self-test
python3 scripts/lock_backend_latency_thresholds.py --self-test
python3 scripts/check_backend_latency_fixtures.py --self-test
python3 scripts/plan_backend_latency_fixture_dispatch.py --self-test
```
