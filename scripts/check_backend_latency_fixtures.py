#!/usr/bin/env python3
"""Validate the backend hardware-latency fixture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from write_backend_latency_result_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    fixture_spec_sha256,
    repo_relative_path,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_BACKENDS = {"cuda", "rocm", "metal", "vulkan"}
VALID_STATUS = {"fixture_required", "covered"}
VALID_THRESHOLD_STATES = {"pending_fixture_result", "locked_threshold"}
VALID_COMPARISONS = {"<=", ">="}
VALID_RESULT_STATUSES = {"passed", "failed"}
CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_string(errors: list[str], obj: dict[str, Any], key: str, context: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{context}.{key} must be a non-empty string")
        return ""
    return value


def validate_metric(
    errors: list[str],
    metric: Any,
    context: str,
    threshold_state: str,
    require_covered: bool,
) -> None:
    if not isinstance(metric, dict):
        errors.append(f"{context} must be an object")
        return

    require_string(errors, metric, "name", context)
    require_string(errors, metric, "unit", context)
    comparison = require_string(errors, metric, "comparison", context)
    if comparison and comparison not in VALID_COMPARISONS:
        errors.append(
            f"{context}.comparison must be one of {sorted(VALID_COMPARISONS)}, got {comparison!r}"
        )

    max_value = metric.get("max")
    if max_value is not None and not is_number(max_value):
        errors.append(f"{context}.max must be null or finite numeric")
    if threshold_state == "locked_threshold" and not is_number(max_value):
        errors.append(f"{context}.max must be finite numeric for locked_threshold")
    if require_covered and not is_number(max_value):
        errors.append(f"{context}.max must be finite numeric when --require-covered is set")


def metric_threshold_passes(metric: dict[str, Any], observed: float) -> bool:
    comparison = metric.get("comparison")
    threshold = metric.get("max")
    if not is_number(threshold):
        return False
    if comparison == "<=":
        return observed <= threshold
    if comparison == ">=":
        return observed >= threshold
    return False


def valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(None)


def validate_result_provenance(
    errors: list[str],
    fixture: dict[str, Any],
    result: dict[str, Any],
    context: str,
    manifest_schema_version: Any,
    manifest_path: Path | None,
) -> None:
    if result.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append(
            f"{context}.result_artifact.artifact_schema_version must be {ARTIFACT_SCHEMA_VERSION!r}, got {result.get('artifact_schema_version')!r}"
        )

    if not valid_utc_timestamp(result.get("created_at_utc")):
        errors.append(
            f"{context}.result_artifact.created_at_utc must be an ISO-8601 UTC timestamp ending in Z"
        )

    manifest = result.get("manifest")
    if not isinstance(manifest, str) or not manifest:
        errors.append(f"{context}.result_artifact.manifest must be a non-empty string")
    elif manifest_path is not None and manifest != repo_relative_path(manifest_path):
        errors.append(
            f"{context}.result_artifact.manifest must be {repo_relative_path(manifest_path)!r}, got {manifest!r}"
        )

    if result.get("manifest_schema_version") != manifest_schema_version:
        errors.append(
            f"{context}.result_artifact.manifest_schema_version must be {manifest_schema_version!r}, got {result.get('manifest_schema_version')!r}"
        )

    expected_fixture_digest = fixture_spec_sha256(fixture)
    observed_fixture_digest = result.get("fixture_spec_sha256")
    if not isinstance(observed_fixture_digest, str) or not CHECKSUM_RE.match(
        observed_fixture_digest
    ):
        errors.append(
            f"{context}.result_artifact.fixture_spec_sha256 must be a lowercase sha256 hex digest"
        )
    elif observed_fixture_digest != expected_fixture_digest:
        errors.append(
            f"{context}.result_artifact.fixture_spec_sha256 must be {expected_fixture_digest!r}, got {observed_fixture_digest!r}"
        )

    for key in ["hardware", "source", "command"]:
        expected = fixture.get(key)
        observed = result.get(key)
        if observed != expected:
            errors.append(
                f"{context}.result_artifact.{key} must be {expected!r}, got {observed!r}"
            )

    raw_log = result.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        errors.append(f"{context}.result_artifact.raw_log must be a non-empty string")

    raw_log_sha256 = result.get("raw_log_sha256")
    if not isinstance(raw_log_sha256, str) or not CHECKSUM_RE.match(raw_log_sha256):
        errors.append(
            f"{context}.result_artifact.raw_log_sha256 must be a lowercase sha256 hex digest"
        )
        return

    if isinstance(raw_log, str) and raw_log:
        raw_log_path = Path(raw_log)
        if not raw_log_path.is_absolute():
            raw_log_path = ROOT / raw_log_path
        if raw_log_path.is_file():
            digest = hashlib.sha256()
            with raw_log_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != raw_log_sha256:
                errors.append(
                    f"{context}.result_artifact.raw_log_sha256 does not match raw_log"
                )


def validate_result_artifact(
    errors: list[str],
    fixture: dict[str, Any],
    result_path: Path,
    context: str,
    manifest_schema_version: Any,
    manifest_path: Path | None,
) -> None:
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{context}.result_artifact is not readable JSON: {exc}")
        return

    fixture_id = fixture.get("id")
    backend = fixture.get("backend")
    if result.get("fixture_id") != fixture_id:
        errors.append(
            f"{context}.result_artifact.fixture_id must be {fixture_id!r}, got {result.get('fixture_id')!r}"
        )
    if result.get("backend") != backend:
        errors.append(
            f"{context}.result_artifact.backend must be {backend!r}, got {result.get('backend')!r}"
        )
    status = result.get("status")
    if status not in VALID_RESULT_STATUSES:
        errors.append(
            f"{context}.result_artifact.status must be one of {sorted(VALID_RESULT_STATUSES)}"
        )
    elif status != "passed":
        errors.append(f"{context}.result_artifact.status must be passed")

    validate_result_provenance(
        errors,
        fixture,
        result,
        context,
        manifest_schema_version,
        manifest_path,
    )

    observed_metrics = result.get("metrics")
    if not isinstance(observed_metrics, dict):
        errors.append(f"{context}.result_artifact.metrics must be an object")
        observed_metrics = {}

    for metric_idx, metric in enumerate(fixture.get("metrics", [])):
        if not isinstance(metric, dict):
            continue
        metric_name = metric.get("name")
        metric_context = f"{context}.metrics[{metric_idx}]"
        observed = observed_metrics.get(metric_name)
        if not is_number(observed):
            errors.append(
                f"{context}.result_artifact.metrics.{metric_name} must be finite numeric"
            )
            continue
        if not metric_threshold_passes(metric, observed):
            errors.append(
                f"{metric_context} observed value {observed} does not satisfy "
                f"{metric.get('comparison')} {metric.get('max')}"
            )


def validate_manifest(
    manifest: dict[str, Any],
    require_covered: bool,
    manifest_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    status = manifest.get("status")
    if status not in VALID_STATUS:
        errors.append(f"status must be one of {sorted(VALID_STATUS)}")
    if require_covered and status != "covered":
        errors.append("status must be covered when --require-covered is set")

    required_backends = manifest.get("required_backends")
    if not isinstance(required_backends, list) or not required_backends:
        errors.append("required_backends must be a non-empty array")
        required_backend_set: set[str] = set()
    else:
        required_backend_set = set()
        for idx, backend in enumerate(required_backends):
            if backend not in VALID_BACKENDS:
                errors.append(f"required_backends[{idx}] is not a valid backend: {backend!r}")
                continue
            required_backend_set.add(backend)

    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        errors.append("fixtures must be a non-empty array")
        fixtures = []

    fixture_ids: set[str] = set()
    fixture_backends: set[str] = set()
    pending_thresholds = 0
    for idx, fixture in enumerate(fixtures):
        context = f"fixtures[{idx}]"
        if not isinstance(fixture, dict):
            errors.append(f"{context} must be an object")
            continue

        fixture_id = require_string(errors, fixture, "id", context)
        if fixture_id:
            if fixture_id in fixture_ids:
                errors.append(f"{context}.id is duplicated: {fixture_id}")
            fixture_ids.add(fixture_id)

        backend = require_string(errors, fixture, "backend", context)
        if backend:
            if backend not in VALID_BACKENDS:
                errors.append(f"{context}.backend is not valid: {backend!r}")
            else:
                fixture_backends.add(backend)

        require_string(errors, fixture, "hardware", context)
        source = require_string(errors, fixture, "source", context)
        if source and not (ROOT / source).is_file():
            errors.append(f"{context}.source does not exist: {source}")
        require_string(errors, fixture, "command", context)
        result_artifact = require_string(errors, fixture, "result_artifact", context)
        result_path = ROOT / result_artifact if result_artifact else None
        result_exists = result_path is not None and result_path.is_file()
        if require_covered and result_artifact and not result_exists:
            errors.append(f"{context}.result_artifact does not exist: {result_artifact}")

        threshold_state = require_string(errors, fixture, "threshold_state", context)
        if threshold_state:
            if threshold_state not in VALID_THRESHOLD_STATES:
                errors.append(
                    f"{context}.threshold_state must be one of {sorted(VALID_THRESHOLD_STATES)}"
                )
            elif threshold_state == "pending_fixture_result":
                pending_thresholds += 1
                if require_covered:
                    errors.append(
                        f"{context}.threshold_state must be locked_threshold when --require-covered is set"
                    )

        metrics = fixture.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"{context}.metrics must be a non-empty array")
        else:
            for metric_idx, metric in enumerate(metrics):
                validate_metric(
                    errors,
                    metric,
                    f"{context}.metrics[{metric_idx}]",
                    threshold_state,
                    require_covered,
                )

        if require_covered and result_exists:
            validate_result_artifact(
                errors,
                fixture,
                result_path,
                context,
                manifest.get("schema_version"),
                manifest_path,
            )

    missing_fixture_slots = manifest.get("missing_fixture_slots", [])
    if not isinstance(missing_fixture_slots, list):
        errors.append("missing_fixture_slots must be an array")
        missing_fixture_slots = []

    missing_slot_backends: set[str] = set()
    for idx, slot in enumerate(missing_fixture_slots):
        context = f"missing_fixture_slots[{idx}]"
        if not isinstance(slot, dict):
            errors.append(f"{context} must be an object")
            continue
        backend = require_string(errors, slot, "backend", context)
        if backend:
            if backend not in VALID_BACKENDS:
                errors.append(f"{context}.backend is not valid: {backend!r}")
            else:
                missing_slot_backends.add(backend)
        require_string(errors, slot, "required_source", context)
        require_string(errors, slot, "reason", context)

    unaccounted_backends = required_backend_set - fixture_backends - missing_slot_backends
    for backend in sorted(unaccounted_backends):
        errors.append(f"required backend has no fixture or missing slot: {backend}")

    if require_covered and missing_fixture_slots:
        errors.append("missing_fixture_slots must be empty when --require-covered is set")
    if status == "covered" and (missing_fixture_slots or pending_thresholds):
        errors.append("covered manifest cannot have missing slots or pending thresholds")

    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        source = tmp_root / "bench.rs"
        source.write_text("// fixture source\n")
        artifact = tmp_root / "result.json"
        raw_log = tmp_root / "raw.log"
        raw_log.write_text("KILN_LATENCY_METRIC latency_ms 9.5 ms\n")
        raw_log_sha256 = hashlib.sha256(raw_log.read_bytes()).hexdigest()
        created_at_utc = "2026-06-06T12:00:00Z"
        fixture = {
            "id": "cuda_fixture",
            "backend": "cuda",
            "hardware": "fixture",
            "source": str(source.relative_to(ROOT))
            if source.is_relative_to(ROOT)
            else str(source),
            "command": "cargo bench",
            "result_artifact": str(artifact.relative_to(ROOT))
            if artifact.is_relative_to(ROOT)
            else str(artifact),
            "threshold_state": "locked_threshold",
            "metrics": [
                {"name": "latency_ms", "unit": "ms", "comparison": "<=", "max": 10.0},
                {"name": "tokens_per_s", "unit": "tok/s", "comparison": ">=", "max": 100.0},
            ],
        }
        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors: list[str] = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if errors:
            print(
                json.dumps(
                    {"ok": False, "case": "passing artifact", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 12.0, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if not any("does not satisfy <= 10.0" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "threshold failure", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "fixture_id": "wrong",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if not any("fixture_id must be" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "fixture id mismatch", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "wrong",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if not any("hardware must be" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "provenance mismatch", "errors": errors},
                    indent=2,
                )
            )
            return 1

        stale_fixture = dict(fixture)
        stale_fixture["command"] = "cargo bench --different"
        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(stale_fixture),
                    "hardware": "fixture",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if not any("fixture_spec_sha256 must be" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "fixture spec mismatch", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": float("inf"), "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if not any("metrics.latency_ms must be finite numeric" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "non-finite metric", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": "2026-06-06T12:00:00-07:00",
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": str(source.relative_to(ROOT))
                    if source.is_relative_to(ROOT)
                    else str(source),
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(errors, fixture, artifact, "fixtures[0]", 1, None)
        if not any("created_at_utc must be" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "artifact timestamp", "errors": errors},
                    indent=2,
                )
            )
            return 1

    print(json.dumps({"ok": True, "self_test": "backend latency fixture validator"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="docs/backend-latency-fixtures.json",
        help="Path to backend-latency-fixtures.json",
    )
    parser.add_argument(
        "--require-covered",
        action="store_true",
        help="Require locked thresholds and checked result artifacts for every backend",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run validator self-tests instead of checking a manifest",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    manifest = json.loads(manifest_path.read_text())
    errors = validate_manifest(manifest, args.require_covered, manifest_path)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "status": manifest["status"],
                "fixtures": len(manifest["fixtures"]),
                "missing_fixture_slots": len(manifest.get("missing_fixture_slots", [])),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
