#!/usr/bin/env python3
"""Validate the backend hardware-latency fixture manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALID_BACKENDS = {"cuda", "rocm", "metal", "vulkan"}
VALID_STATUS = {"fixture_required", "covered"}
VALID_THRESHOLD_STATES = {"pending_fixture_result", "locked_threshold"}
VALID_COMPARISONS = {"<=", ">="}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
        errors.append(f"{context}.max must be null or numeric")
    if threshold_state == "locked_threshold" and not is_number(max_value):
        errors.append(f"{context}.max must be numeric for locked_threshold")
    if require_covered and not is_number(max_value):
        errors.append(f"{context}.max must be numeric when --require-covered is set")


def validate_manifest(manifest: dict[str, Any], require_covered: bool) -> list[str]:
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
        if require_covered and result_artifact and not (ROOT / result_artifact).is_file():
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
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    manifest = json.loads(manifest_path.read_text())
    errors = validate_manifest(manifest, args.require_covered)
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
