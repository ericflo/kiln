#!/usr/bin/env python3
"""Lock backend latency fixture thresholds from checked result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from write_backend_latency_result_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactError,
    GIT_COMMIT_RE,
    LATENCY_RAW_LOG_DIR,
    LATENCY_RESULT_ARTIFACT_DIR,
    RESULT_ARTIFACT_KEYS,
    fixture_spec_sha256,
    is_canonical_raw_log_path,
    is_canonical_result_artifact_path,
    is_repo_relative_path,
    parse_metric_log,
    repo_relative_path,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_COMPARISONS = {"<=", ">="}
CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")


class ThresholdLockError(Exception):
    pass


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ThresholdLockError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ThresholdLockError(f"{label} must be a JSON object")
    return value


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_from_observed(observed: float, comparison: str, headroom: float) -> float:
    if comparison == "<=":
        return observed * (1.0 + headroom)
    if comparison == ">=":
        return observed * (1.0 - headroom)
    raise ThresholdLockError(
        f"metric comparison must be one of {sorted(VALID_COMPARISONS)}, got {comparison!r}"
    )


def rounded_threshold(value: float) -> float:
    return round(value, 6)


def valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(None)


def validate_result_keys(result: dict[str, Any], fixture_id: str) -> None:
    observed_keys = set(result)
    missing_keys = sorted(RESULT_ARTIFACT_KEYS - observed_keys)
    extra_keys = sorted(observed_keys - RESULT_ARTIFACT_KEYS)
    if missing_keys:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact missing required keys: {missing_keys}"
        )
    if extra_keys:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact contains unknown keys: {extra_keys}"
        )


def expected_metric_names(fixture: dict[str, Any]) -> set[str]:
    return {
        metric["name"]
        for metric in fixture.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("name"), str)
    }


def lock_fixture_thresholds(
    fixture: dict[str, Any],
    headroom: float,
    manifest_schema_version: Any,
    manifest_path: Path | None,
) -> dict[str, Any]:
    fixture_id = fixture.get("id")
    backend = fixture.get("backend")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise ThresholdLockError("fixture.id must be a non-empty string")
    if not isinstance(backend, str) or not backend:
        raise ThresholdLockError(f"{fixture_id}.backend must be a non-empty string")

    result_artifact = fixture.get("result_artifact")
    if not isinstance(result_artifact, str) or not result_artifact:
        raise ThresholdLockError(f"{fixture_id}.result_artifact must be a non-empty string")
    if not is_repo_relative_path(result_artifact):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact must be repo-relative before thresholds can lock"
        )
    if not is_canonical_result_artifact_path(result_artifact):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact must live under {LATENCY_RESULT_ARTIFACT_DIR} "
            "with a .json extension before thresholds can lock"
        )
    result_path = resolve_repo_path(result_artifact)
    if not result_path.is_file():
        raise ThresholdLockError(f"{fixture_id}.result_artifact does not exist: {result_artifact}")

    result = load_json(result_path, f"{fixture_id}.result_artifact")
    validate_result_keys(result, fixture_id)
    if result.get("fixture_id") != fixture_id:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.fixture_id must be {fixture_id!r}, got {result.get('fixture_id')!r}"
        )
    if result.get("backend") != backend:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.backend must be {backend!r}, got {result.get('backend')!r}"
        )
    if result.get("status") != "passed":
        raise ThresholdLockError(f"{fixture_id}.result_artifact.status must be passed")
    if result.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.artifact_schema_version must be {ARTIFACT_SCHEMA_VERSION!r}, got {result.get('artifact_schema_version')!r}"
        )
    if not valid_utc_timestamp(result.get("created_at_utc")):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.created_at_utc must be an ISO-8601 UTC timestamp ending in Z"
        )
    git_commit = result.get("git_commit")
    if not isinstance(git_commit, str) or not GIT_COMMIT_RE.match(git_commit):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.git_commit must be a lowercase 40-character git commit"
        )
    git_tracked_dirty = result.get("git_tracked_dirty")
    if not isinstance(git_tracked_dirty, bool):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.git_tracked_dirty must be a boolean"
        )
    if git_tracked_dirty:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.git_tracked_dirty must be false before thresholds can lock"
        )
    manifest = result.get("manifest")
    if not isinstance(manifest, str) or not manifest:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.manifest must be a non-empty string"
        )
    if not is_repo_relative_path(manifest):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.manifest must be repo-relative before thresholds can lock"
        )
    if manifest_path is not None and manifest != repo_relative_path(manifest_path):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.manifest must be {repo_relative_path(manifest_path)!r}, got {manifest!r}"
        )
    if result.get("manifest_schema_version") != manifest_schema_version:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.manifest_schema_version must be {manifest_schema_version!r}, got {result.get('manifest_schema_version')!r}"
        )
    expected_fixture_digest = fixture_spec_sha256(fixture)
    observed_fixture_digest = result.get("fixture_spec_sha256")
    if not isinstance(observed_fixture_digest, str) or not CHECKSUM_RE.match(
        observed_fixture_digest
    ):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.fixture_spec_sha256 must be a lowercase sha256 hex digest"
        )
    if observed_fixture_digest != expected_fixture_digest:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.fixture_spec_sha256 must be {expected_fixture_digest!r}, got {observed_fixture_digest!r}"
        )
    for key in ["hardware", "source", "command"]:
        if result.get(key) != fixture.get(key):
            raise ThresholdLockError(
                f"{fixture_id}.result_artifact.{key} must be {fixture.get(key)!r}, got {result.get(key)!r}"
            )
    source = fixture.get("source")
    if not isinstance(source, str) or not source:
        raise ThresholdLockError(f"{fixture_id}.source must be a non-empty string")
    if not is_repo_relative_path(source):
        raise ThresholdLockError(
            f"{fixture_id}.source must be repo-relative before thresholds can lock"
        )
    source_path = resolve_repo_path(source)
    if not source_path.is_file():
        raise ThresholdLockError(
            f"{fixture_id}.source must exist before thresholds can lock: {source}"
        )
    source_sha256 = result.get("source_sha256")
    if not isinstance(source_sha256, str) or not CHECKSUM_RE.match(source_sha256):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.source_sha256 must be a lowercase sha256 hex digest"
        )
    if sha256_file(source_path) != source_sha256:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.source_sha256 does not match source"
        )
    raw_log = result.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log must be a non-empty string"
        )
    if not is_repo_relative_path(raw_log):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log must be repo-relative before thresholds can lock"
        )
    if not is_canonical_raw_log_path(raw_log):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log must live under {LATENCY_RAW_LOG_DIR} "
            "with a .log extension before thresholds can lock"
        )
    raw_log_path = resolve_repo_path(raw_log)
    if not raw_log_path.is_file():
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log must exist before thresholds can lock: {raw_log}"
        )
    raw_log_sha256 = result.get("raw_log_sha256")
    if not isinstance(raw_log_sha256, str) or not CHECKSUM_RE.match(raw_log_sha256):
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log_sha256 must be a lowercase sha256 hex digest"
        )
    if sha256_file(raw_log_path) != raw_log_sha256:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log_sha256 does not match raw_log"
        )
    try:
        raw_observations = parse_metric_log(raw_log_path)
    except ArtifactError as exc:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.raw_log metrics are invalid: {exc}"
        ) from exc

    observations = result.get("metrics")
    if not isinstance(observations, dict):
        raise ThresholdLockError(f"{fixture_id}.result_artifact.metrics must be an object")
    extra_metrics = sorted(set(observations) - expected_metric_names(fixture))
    if extra_metrics:
        raise ThresholdLockError(
            f"{fixture_id}.result_artifact.metrics contains undeclared metrics: {extra_metrics}"
        )

    metrics = fixture.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ThresholdLockError(f"{fixture_id}.metrics must be a non-empty array")

    locked = deepcopy(fixture)
    locked["threshold_state"] = "locked_threshold"
    locked_metrics = locked["metrics"]
    for index, metric in enumerate(locked_metrics):
        if not isinstance(metric, dict):
            raise ThresholdLockError(f"{fixture_id}.metrics[{index}] must be an object")
        metric_name = metric.get("name")
        if not isinstance(metric_name, str) or not metric_name:
            raise ThresholdLockError(f"{fixture_id}.metrics[{index}].name must be a non-empty string")
        observed = observations.get(metric_name)
        if not is_number(observed):
            raise ThresholdLockError(
                f"{fixture_id}.result_artifact.metrics.{metric_name} must be finite numeric"
            )
        raw_observed = raw_observations.get(metric_name)
        if raw_observed is None:
            raise ThresholdLockError(
                f"{fixture_id}.result_artifact.raw_log missing metric {metric_name}"
            )
        raw_value, raw_unit = raw_observed
        unit = metric.get("unit")
        if raw_unit != unit:
            raise ThresholdLockError(
                f"{fixture_id}.result_artifact.raw_log metric {metric_name} unit must be {unit!r}, got {raw_unit!r}"
            )
        if raw_value != observed:
            raise ThresholdLockError(
                f"{fixture_id}.result_artifact.metrics.{metric_name} must match raw_log value {raw_value}"
            )
        comparison = metric.get("comparison")
        if not isinstance(comparison, str):
            raise ThresholdLockError(f"{fixture_id}.metrics[{index}].comparison must be a string")
        metric["max"] = rounded_threshold(
            threshold_from_observed(float(observed), comparison, headroom)
        )

    return locked


def lock_manifest_thresholds(
    manifest: dict[str, Any],
    headroom: float,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not 0.0 <= headroom < 1.0:
        raise ThresholdLockError("--headroom must be >= 0 and < 1")

    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ThresholdLockError("manifest.fixtures must be a non-empty array")
    missing_slots = manifest.get("missing_fixture_slots", [])
    if missing_slots:
        raise ThresholdLockError("missing_fixture_slots must be empty before thresholds can lock")

    locked = deepcopy(manifest)
    locked["fixtures"] = [
        lock_fixture_thresholds(
            fixture,
            headroom,
            manifest.get("schema_version"),
            manifest_path,
        )
        for fixture in fixtures
    ]
    locked["status"] = "covered"
    return locked


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def self_test() -> int:
    temp_parent = ROOT / "target"
    temp_parent.mkdir(parents=True, exist_ok=True)
    result_parent = ROOT / LATENCY_RESULT_ARTIFACT_DIR
    raw_parent = ROOT / LATENCY_RAW_LOG_DIR
    result_parent.mkdir(parents=True, exist_ok=True)
    raw_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="backend-latency-locker-", dir=temp_parent
    ) as tmp, tempfile.TemporaryDirectory(
        prefix="backend-latency-locker-", dir=result_parent
    ) as result_tmp, tempfile.TemporaryDirectory(
        prefix="backend-latency-locker-", dir=raw_parent
    ) as raw_tmp:
        tmp_root = Path(tmp)
        result_path = Path(result_tmp) / "result.json"
        raw_log_path = Path(raw_tmp) / "bench.log"
        source_path = tmp_root / "bench.py"
        source_path.write_text("# latency fixture source\n")
        raw_log_path.write_text(
            "KILN_LATENCY_METRIC latency_ms 10.0 ms\n"
            "KILN_LATENCY_METRIC tokens_per_s 200.0 tok/s\n"
        )
        raw_log_sha256 = sha256_file(raw_log_path)
        source_sha256 = sha256_file(source_path)
        created_at_utc = "2026-06-06T12:00:00Z"
        git_commit = "a" * 40
        git_tracked_dirty = False
        result_artifact_path = repo_relative_path(result_path)
        raw_log_artifact_path = repo_relative_path(raw_log_path)
        source_artifact_path = repo_relative_path(source_path)
        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        manifest = {
            "schema_version": 1,
            "status": "fixture_required",
            "fixtures": [
                {
                    "id": "fixture",
                    "backend": "cuda",
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "command": "python bench.py",
                    "result_artifact": result_artifact_path,
                    "threshold_state": "pending_fixture_result",
                    "metrics": [
                        {"name": "latency_ms", "unit": "ms", "comparison": "<=", "max": None},
                        {
                            "name": "tokens_per_s",
                            "unit": "tok/s",
                            "comparison": ">=",
                            "max": None,
                        },
                    ],
                }
            ],
            "missing_fixture_slots": [],
        }
        result = json.loads(result_path.read_text())
        result["fixture_spec_sha256"] = fixture_spec_sha256(manifest["fixtures"][0])
        result_path.write_text(json.dumps(result))
        locked = lock_manifest_thresholds(manifest, 0.10)
        metrics = locked["fixtures"][0]["metrics"]
        if locked["status"] != "covered" or locked["fixtures"][0]["threshold_state"] != "locked_threshold":
            print(json.dumps({"ok": False, "case": "status lock", "manifest": locked}, indent=2))
            return 1
        if metrics[0]["max"] != 11.0 or metrics[1]["max"] != 180.0:
            print(json.dumps({"ok": False, "case": "headroom", "metrics": metrics}, indent=2))
            return 1

        result = json.loads(result_path.read_text())
        result["unexpected"] = True
        result_path.write_text(json.dumps(result))
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "contains unknown keys" not in str(exc):
                print(json.dumps({"ok": False, "case": "unknown artifact key", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "unknown artifact key did not fail"}))
            return 1

        result.pop("unexpected")
        result["metrics"]["undeclared_ms"] = 1.0
        result_path.write_text(json.dumps(result))
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "contains undeclared metrics" not in str(exc):
                print(json.dumps({"ok": False, "case": "undeclared metric", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "undeclared metric did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "tokens_per_s" not in str(exc):
                print(json.dumps({"ok": False, "case": "missing metric", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "missing metric did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": "1" * 64,
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "fixture_spec_sha256" not in str(exc):
                print(json.dumps({"ok": False, "case": "fixture digest", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "fixture digest did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": float("inf"), "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "finite numeric" not in str(exc):
                print(json.dumps({"ok": False, "case": "non-finite metric", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "non-finite metric did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": "2026-06-06T12:00:00-07:00",
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "created_at_utc" not in str(exc):
                print(json.dumps({"ok": False, "case": "artifact timestamp", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "artifact timestamp did not fail"}))
            return 1

        dirty_result = json.loads(result_path.read_text())
        dirty_result["created_at_utc"] = created_at_utc
        dirty_result["git_tracked_dirty"] = True
        result_path.write_text(json.dumps(dirty_result))
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "git_tracked_dirty must be false" not in str(exc):
                print(json.dumps({"ok": False, "case": "dirty git checkout", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "dirty git checkout did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": repo_relative_path(Path(raw_tmp) / "missing.log"),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "raw_log must exist" not in str(exc):
                print(json.dumps({"ok": False, "case": "missing raw log", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "missing raw log did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": "1" * 64,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "raw_log_sha256 does not match raw_log" not in str(exc):
                print(json.dumps({"ok": False, "case": "raw log checksum", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "raw log checksum did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "must match raw_log value 10.0" not in str(exc):
                print(json.dumps({"ok": False, "case": "raw log value", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "raw log value did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": "1" * 64,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "source_sha256 does not match source" not in str(exc):
                print(json.dumps({"ok": False, "case": "source checksum", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "source checksum did not fail"}))
            return 1

        noncanonical_result_path = tmp_root / "result.json"
        noncanonical_result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": raw_log_artifact_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        noncanonical_manifest = deepcopy(manifest)
        noncanonical_manifest["fixtures"][0]["result_artifact"] = repo_relative_path(
            noncanonical_result_path
        )
        try:
            lock_manifest_thresholds(noncanonical_manifest, 0.10)
        except ThresholdLockError as exc:
            if "result_artifact must live under" not in str(exc):
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "case": "noncanonical artifact",
                            "error": str(exc),
                        }
                    )
                )
                return 1
        else:
            print(
                json.dumps(
                    {"ok": False, "case": "noncanonical artifact did not fail"}
                )
            )
            return 1

        noncanonical_raw_log = tmp_root / "bench.log"
        noncanonical_raw_log.write_text(raw_log_path.read_text())
        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": repo_relative_path(noncanonical_raw_log),
                    "raw_log_sha256": sha256_file(noncanonical_raw_log),
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "raw_log must live under" not in str(exc):
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "case": "noncanonical raw log",
                            "error": str(exc),
                        }
                    )
                )
                return 1
        else:
            print(
                json.dumps(
                    {"ok": False, "case": "noncanonical raw log did not fail"}
                )
            )
            return 1

        absolute_manifest = deepcopy(manifest)
        absolute_manifest["fixtures"][0]["result_artifact"] = str(result_path)
        try:
            lock_manifest_thresholds(absolute_manifest, 0.10)
        except ThresholdLockError as exc:
            if "result_artifact must be repo-relative" not in str(exc):
                print(json.dumps({"ok": False, "case": "absolute artifact", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "absolute artifact did not fail"}))
            return 1

        result_path.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(manifest["fixtures"][0]),
                    "hardware": "fixture hardware",
                    "source": source_artifact_path,
                    "source_sha256": source_sha256,
                    "command": "python bench.py",
                    "raw_log": str(raw_log_path),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 10.0, "tokens_per_s": 200.0},
                }
            )
        )
        try:
            lock_manifest_thresholds(manifest, 0.10)
        except ThresholdLockError as exc:
            if "raw_log must be repo-relative" not in str(exc):
                print(json.dumps({"ok": False, "case": "absolute raw log", "error": str(exc)}))
                return 1
        else:
            print(json.dumps({"ok": False, "case": "absolute raw log did not fail"}))
            return 1

    print(json.dumps({"ok": True, "self_test": "backend latency threshold locker"}))
    return 0


def fail(message: str) -> int:
    print(json.dumps({"ok": False, "error": message}, indent=2), file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        default="docs/backend-latency-fixtures.json",
        help="Path to backend-latency-fixtures.json",
    )
    parser.add_argument(
        "--output",
        help="Output manifest path; defaults to overwriting the input manifest",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.10,
        help="Fractional threshold headroom applied to observed metrics; default 0.10",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that thresholds can lock without writing the manifest",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run threshold-locker self-tests instead of locking a manifest",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_path = Path(args.output) if args.output else manifest_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    try:
        manifest = load_json(manifest_path, "manifest")
        locked = lock_manifest_thresholds(manifest, args.headroom, manifest_path)
        if not args.check:
            write_manifest(output_path, locked)
    except ThresholdLockError as exc:
        return fail(str(exc))

    print(
        json.dumps(
            {
                "ok": True,
                "status": locked["status"],
                "fixtures": len(locked["fixtures"]),
                "headroom": args.headroom,
                "output": None if args.check else str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
