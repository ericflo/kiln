#!/usr/bin/env python3
"""Write a backend latency result artifact from a fixture benchmark log."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRIC_RE = re.compile(r"^\s*KILN_LATENCY_METRIC\s+(\S+)\s+([-+0-9.eE]+)\s+(\S+)\s*$")
VALID_RESULT_STATUSES = {"passed", "failed"}


class ArtifactError(Exception):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"manifest is not readable JSON: {exc}") from exc


def find_fixture(manifest: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise ArtifactError("manifest.fixtures must be an array")
    for fixture in fixtures:
        if isinstance(fixture, dict) and fixture.get("id") == fixture_id:
            return fixture
    raise ArtifactError(f"fixture id not found in manifest: {fixture_id}")


def parse_metric_log(log_path: Path) -> dict[str, tuple[float, str]]:
    observations: dict[str, tuple[float, str]] = {}
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        raise ArtifactError(f"log is not readable: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        match = METRIC_RE.match(line)
        if not match:
            continue
        metric_name, raw_value, unit = match.groups()
        if metric_name in observations:
            raise ArtifactError(
                f"duplicate KILN_LATENCY_METRIC line for {metric_name!r} at line {line_number}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ArtifactError(
                f"invalid numeric value for {metric_name!r} at line {line_number}: {raw_value!r}"
            ) from exc
        observations[metric_name] = (value, unit)

    if not observations:
        raise ArtifactError(f"no KILN_LATENCY_METRIC lines found in {log_path}")
    return observations


def fixture_metric_specs(fixture: dict[str, Any]) -> dict[str, str]:
    metrics = fixture.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ArtifactError("fixture.metrics must be a non-empty array")

    specs: dict[str, str] = {}
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            raise ArtifactError(f"fixture.metrics[{index}] must be an object")
        name = metric.get("name")
        unit = metric.get("unit")
        if not isinstance(name, str) or not name:
            raise ArtifactError(f"fixture.metrics[{index}].name must be a non-empty string")
        if not isinstance(unit, str) or not unit:
            raise ArtifactError(f"fixture.metrics[{index}].unit must be a non-empty string")
        specs[name] = unit
    return specs


def build_result_artifact(
    fixture: dict[str, Any],
    observations: dict[str, tuple[float, str]],
    status: str,
) -> dict[str, Any]:
    if status not in VALID_RESULT_STATUSES:
        raise ArtifactError(f"status must be one of {sorted(VALID_RESULT_STATUSES)}")

    specs = fixture_metric_specs(fixture)
    artifact_metrics: dict[str, float] = {}
    missing: list[str] = []
    unit_mismatches: list[str] = []
    for metric_name, expected_unit in specs.items():
        observed = observations.get(metric_name)
        if observed is None:
            missing.append(metric_name)
            continue
        value, observed_unit = observed
        if observed_unit != expected_unit:
            unit_mismatches.append(
                f"{metric_name}: expected unit {expected_unit!r}, got {observed_unit!r}"
            )
            continue
        artifact_metrics[metric_name] = value

    if missing:
        raise ArtifactError(f"missing required metrics: {', '.join(sorted(missing))}")
    if unit_mismatches:
        raise ArtifactError("metric unit mismatch: " + "; ".join(unit_mismatches))

    fixture_id = fixture.get("id")
    backend = fixture.get("backend")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise ArtifactError("fixture.id must be a non-empty string")
    if not isinstance(backend, str) or not backend:
        raise ArtifactError("fixture.backend must be a non-empty string")

    return {
        "fixture_id": fixture_id,
        "backend": backend,
        "status": status,
        "metrics": artifact_metrics,
    }


def default_output_path(fixture: dict[str, Any]) -> Path:
    result_artifact = fixture.get("result_artifact")
    if not isinstance(result_artifact, str) or not result_artifact:
        raise ArtifactError("fixture.result_artifact must be a non-empty string")
    path = Path(result_artifact)
    return path if path.is_absolute() else ROOT / path


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        manifest_path = tmp_root / "fixtures.json"
        log_path = tmp_root / "bench.log"
        output_path = tmp_root / "result.json"
        manifest = {
            "fixtures": [
                {
                    "id": "cuda_fixture",
                    "backend": "cuda",
                    "result_artifact": str(output_path),
                    "metrics": [
                        {"name": "latency_ms", "unit": "ms"},
                        {"name": "tokens_per_s", "unit": "tok/s"},
                    ],
                }
            ]
        }
        manifest_path.write_text(json.dumps(manifest))
        log_path.write_text(
            "\n".join(
                [
                    "human-readable bench output",
                    "KILN_LATENCY_METRIC latency_ms 9.5 ms",
                    "KILN_LATENCY_METRIC tokens_per_s 125.0 tok/s",
                ]
            )
            + "\n"
        )

        loaded = load_manifest(manifest_path)
        fixture = find_fixture(loaded, "cuda_fixture")
        observations = parse_metric_log(log_path)
        artifact = build_result_artifact(fixture, observations, "passed")
        write_artifact(output_path, artifact)
        written = json.loads(output_path.read_text())
        if written != {
            "fixture_id": "cuda_fixture",
            "backend": "cuda",
            "status": "passed",
            "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
        }:
            print(
                json.dumps(
                    {"ok": False, "case": "write artifact", "artifact": written},
                    indent=2,
                )
            )
            return 1

        try:
            build_result_artifact(
                fixture,
                {"latency_ms": (9.5, "ms")},
                "passed",
            )
        except ArtifactError as exc:
            if "missing required metrics" not in str(exc):
                print(json.dumps({"ok": False, "case": "missing metric", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "missing metric did not fail"}))
            return 1

        try:
            build_result_artifact(
                fixture,
                {"latency_ms": (9.5, "s"), "tokens_per_s": (125.0, "tok/s")},
                "passed",
            )
        except ArtifactError as exc:
            if "metric unit mismatch" not in str(exc):
                print(json.dumps({"ok": False, "case": "unit mismatch", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "unit mismatch did not fail"}))
            return 1

    print(json.dumps({"ok": True, "self_test": "backend latency artifact writer"}))
    return 0


def fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, indent=2), file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="Path to backend-latency-fixtures.json")
    parser.add_argument("fixture_id", nargs="?", help="Fixture id to materialize")
    parser.add_argument("log_path", nargs="?", help="Benchmark log with KILN_LATENCY_METRIC lines")
    parser.add_argument(
        "--output",
        help="Override output JSON path; defaults to the fixture result_artifact path",
    )
    parser.add_argument(
        "--status",
        choices=sorted(VALID_RESULT_STATUSES),
        default="passed",
        help="Result status to write",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run artifact-writer self-tests instead of writing a result artifact",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.manifest or not args.fixture_id or not args.log_path:
        return fail("manifest, fixture_id, and log_path are required unless --self-test is set")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path

    try:
        manifest = load_manifest(manifest_path)
        fixture = find_fixture(manifest, args.fixture_id)
        observations = parse_metric_log(log_path)
        artifact = build_result_artifact(fixture, observations, args.status)
        output_path = Path(args.output) if args.output else default_output_path(fixture)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        write_artifact(output_path, artifact)
    except ArtifactError as exc:
        return fail(str(exc))

    print(
        json.dumps(
            {
                "ok": True,
                "fixture_id": artifact["fixture_id"],
                "backend": artifact["backend"],
                "output": str(output_path),
                "metrics": sorted(artifact["metrics"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
