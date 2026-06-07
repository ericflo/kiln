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
    ArtifactError,
    GIT_COMMIT_RE,
    LATENCY_RAW_LOG_DIR,
    LATENCY_RESULT_ARTIFACT_DIR,
    RESULT_ARTIFACT_KEYS,
    current_git_commit,
    fixture_spec_sha256,
    git_commit_exists,
    git_file_sha256_at_commit,
    git_path_is_tracked,
    is_canonical_raw_log_path,
    is_canonical_result_artifact_path,
    is_repo_relative_path,
    parse_metric_log,
    repo_relative_path,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_BACKENDS = {"cuda", "rocm", "metal", "vulkan"}
VALID_STATUS = {"fixture_required", "covered"}
VALID_THRESHOLD_STATES = {"pending_fixture_result", "locked_threshold"}
VALID_COMPARISONS = {"<=", ">="}
VALID_RESULT_STATUSES = {"passed", "failed"}
CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_KEYS = {
    "fixtures",
    "missing_fixture_slots",
    "policy",
    "required_backends",
    "schema_version",
    "status",
}
REQUIRED_COVERED_GATE_POLICY = [
    "Every required backend has at least one known hardware fixture.",
    "Every fixture has locked numeric thresholds for its required measurements.",
    "Every locked threshold has a checked hardware-result artifact from the named fixture.",
    "Default-feature local tests must not mark the hardware latency gate covered.",
]


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


def resolve_artifact_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def raw_log_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result_provenance(
    errors: list[str],
    fixture: dict[str, Any],
    result: dict[str, Any],
    context: str,
    manifest_schema_version: Any,
    manifest_path: Path | None,
    require_raw_log_file: bool,
    require_tracked_files: bool,
) -> None:
    if result.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append(
            f"{context}.result_artifact.artifact_schema_version must be {ARTIFACT_SCHEMA_VERSION!r}, got {result.get('artifact_schema_version')!r}"
        )

    if not valid_utc_timestamp(result.get("created_at_utc")):
        errors.append(
            f"{context}.result_artifact.created_at_utc must be an ISO-8601 UTC timestamp ending in Z"
        )

    git_commit = result.get("git_commit")
    if not isinstance(git_commit, str) or not GIT_COMMIT_RE.match(git_commit):
        errors.append(
            f"{context}.result_artifact.git_commit must be a lowercase 40-character git commit"
        )
    elif require_raw_log_file and not git_commit_exists(git_commit):
        errors.append(
            f"{context}.result_artifact.git_commit must exist in the local repository when --require-covered is set"
        )

    git_tracked_dirty = result.get("git_tracked_dirty")
    if not isinstance(git_tracked_dirty, bool):
        errors.append(f"{context}.result_artifact.git_tracked_dirty must be a boolean")
    elif require_raw_log_file and git_tracked_dirty:
        errors.append(
            f"{context}.result_artifact.git_tracked_dirty must be false when --require-covered is set"
        )

    manifest = result.get("manifest")
    if not isinstance(manifest, str) or not manifest:
        errors.append(f"{context}.result_artifact.manifest must be a non-empty string")
    elif manifest_path is not None and manifest != repo_relative_path(manifest_path):
        errors.append(
            f"{context}.result_artifact.manifest must be {repo_relative_path(manifest_path)!r}, got {manifest!r}"
        )
    elif require_raw_log_file and not is_repo_relative_path(manifest):
        errors.append(
            f"{context}.result_artifact.manifest must be repo-relative when --require-covered is set"
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

    source = fixture.get("source")
    source_sha256 = result.get("source_sha256")
    if not isinstance(source_sha256, str) or not CHECKSUM_RE.match(source_sha256):
        errors.append(
            f"{context}.result_artifact.source_sha256 must be a lowercase sha256 hex digest"
        )
    elif isinstance(source, str) and source:
        source_path = resolve_artifact_path(source)
        if source_path.is_file():
            if raw_log_digest(source_path) != source_sha256:
                errors.append(
                    f"{context}.result_artifact.source_sha256 does not match source"
                )
            elif (
                require_raw_log_file
                and isinstance(git_commit, str)
                and GIT_COMMIT_RE.match(git_commit)
            ):
                commit_source_sha256 = git_file_sha256_at_commit(git_commit, source)
                if commit_source_sha256 is None:
                    errors.append(
                        f"{context}.result_artifact.source must exist at git_commit when --require-covered is set"
                    )
                elif commit_source_sha256 != source_sha256:
                    errors.append(
                        f"{context}.result_artifact.source_sha256 must match source at git_commit"
                    )
        elif require_raw_log_file:
            errors.append(
                f"{context}.source must exist when --require-covered is set: {source}"
            )

    raw_log = result.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        errors.append(f"{context}.result_artifact.raw_log must be a non-empty string")
    elif require_raw_log_file and not is_repo_relative_path(raw_log):
        errors.append(
            f"{context}.result_artifact.raw_log must be repo-relative when --require-covered is set"
        )
    elif require_raw_log_file and not is_canonical_raw_log_path(raw_log):
        errors.append(
            f"{context}.result_artifact.raw_log must live under {LATENCY_RAW_LOG_DIR} "
            "with a .log extension when --require-covered is set"
        )

    raw_log_sha256 = result.get("raw_log_sha256")
    if not isinstance(raw_log_sha256, str) or not CHECKSUM_RE.match(raw_log_sha256):
        errors.append(
            f"{context}.result_artifact.raw_log_sha256 must be a lowercase sha256 hex digest"
        )
        return

    if isinstance(raw_log, str) and raw_log:
        raw_log_path = resolve_artifact_path(raw_log)
        if raw_log_path.is_file():
            if raw_log_digest(raw_log_path) != raw_log_sha256:
                errors.append(
                    f"{context}.result_artifact.raw_log_sha256 does not match raw_log"
                )
            if require_tracked_files and not git_path_is_tracked(raw_log):
                errors.append(
                    f"{context}.result_artifact.raw_log must be tracked by git when --require-covered is set"
                )
        elif require_raw_log_file:
            errors.append(
                f"{context}.result_artifact.raw_log must exist when --require-covered is set: {raw_log}"
            )


def validate_result_keys(errors: list[str], result: dict[str, Any], context: str) -> None:
    observed_keys = set(result)
    missing_keys = sorted(RESULT_ARTIFACT_KEYS - observed_keys)
    extra_keys = sorted(observed_keys - RESULT_ARTIFACT_KEYS)
    if missing_keys:
        errors.append(f"{context}.result_artifact missing required keys: {missing_keys}")
    if extra_keys:
        errors.append(f"{context}.result_artifact contains unknown keys: {extra_keys}")


def expected_metric_names(fixture: dict[str, Any]) -> set[str]:
    return {
        metric["name"]
        for metric in fixture.get("metrics", [])
        if isinstance(metric, dict) and isinstance(metric.get("name"), str)
    }


def validate_result_metric_keys(
    errors: list[str],
    fixture: dict[str, Any],
    observed_metrics: dict[str, Any],
    context: str,
) -> None:
    expected = expected_metric_names(fixture)
    extra_metrics = sorted(set(observed_metrics) - expected)
    if extra_metrics:
        errors.append(
            f"{context}.result_artifact.metrics contains undeclared metrics: {extra_metrics}"
        )


def validate_raw_log_metrics(
    errors: list[str],
    fixture: dict[str, Any],
    result: dict[str, Any],
    observed_metrics: dict[str, Any],
    context: str,
    require_raw_log_file: bool,
) -> None:
    raw_log = result.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        return

    raw_log_path = resolve_artifact_path(raw_log)
    if not raw_log_path.is_file():
        return

    try:
        raw_observations = parse_metric_log(raw_log_path)
    except ArtifactError as exc:
        errors.append(f"{context}.result_artifact.raw_log metrics are invalid: {exc}")
        return

    for metric_idx, metric in enumerate(fixture.get("metrics", [])):
        if not isinstance(metric, dict):
            continue
        metric_name = metric.get("name")
        if not isinstance(metric_name, str) or not metric_name:
            continue
        raw_observed = raw_observations.get(metric_name)
        if raw_observed is None:
            if require_raw_log_file:
                errors.append(
                    f"{context}.result_artifact.raw_log missing metric {metric_name}"
                )
            continue
        raw_value, raw_unit = raw_observed
        if raw_unit != metric.get("unit"):
            errors.append(
                f"{context}.result_artifact.raw_log metric {metric_name} unit must be {metric.get('unit')!r}, got {raw_unit!r}"
            )
        artifact_value = observed_metrics.get(metric_name)
        if is_number(artifact_value) and raw_value != artifact_value:
            errors.append(
                f"{context}.result_artifact.metrics.{metric_name} must match raw_log value {raw_value}"
            )


def validate_result_artifact(
    errors: list[str],
    fixture: dict[str, Any],
    result_path: Path,
    context: str,
    manifest_schema_version: Any,
    manifest_path: Path | None,
    require_raw_log_file: bool = False,
    require_tracked_files: bool = False,
) -> None:
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{context}.result_artifact is not readable JSON: {exc}")
        return

    validate_result_keys(errors, result, context)

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
        require_raw_log_file,
        require_tracked_files,
    )

    observed_metrics = result.get("metrics")
    if not isinstance(observed_metrics, dict):
        errors.append(f"{context}.result_artifact.metrics must be an object")
        observed_metrics = {}
    else:
        validate_result_metric_keys(errors, fixture, observed_metrics, context)

    validate_raw_log_metrics(
        errors,
        fixture,
        result,
        observed_metrics,
        context,
        require_raw_log_file,
    )

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


def validate_manifest_keys(errors: list[str], manifest: dict[str, Any]) -> None:
    observed_keys = set(manifest)
    missing_keys = sorted(MANIFEST_KEYS - observed_keys)
    extra_keys = sorted(observed_keys - MANIFEST_KEYS)
    if missing_keys:
        errors.append(f"manifest missing required keys: {missing_keys}")
    if extra_keys:
        errors.append(f"manifest contains unknown keys: {extra_keys}")


def validate_manifest_policy(errors: list[str], manifest: dict[str, Any]) -> None:
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        errors.append("policy must be an object")
        return
    observed_keys = set(policy)
    if observed_keys != {"covered_gate_requires"}:
        missing = sorted({"covered_gate_requires"} - observed_keys)
        extra = sorted(observed_keys - {"covered_gate_requires"})
        if missing:
            errors.append(f"policy missing required keys: {missing}")
        if extra:
            errors.append(f"policy contains unknown keys: {extra}")
    covered_gate_requires = policy.get("covered_gate_requires")
    if covered_gate_requires != REQUIRED_COVERED_GATE_POLICY:
        errors.append("policy.covered_gate_requires must match the hardware latency gate policy")


def validate_manifest(
    manifest: dict[str, Any],
    require_covered: bool,
    manifest_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    validate_manifest_keys(errors, manifest)
    validate_manifest_policy(errors, manifest)

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    status = manifest.get("status")
    if status not in VALID_STATUS:
        errors.append(f"status must be one of {sorted(VALID_STATUS)}")
    if status == "covered" and not require_covered:
        errors.append("status covered requires --require-covered")
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
        if require_covered and source and not is_repo_relative_path(source):
            errors.append(f"{context}.source must be repo-relative when --require-covered is set")
        if source and not (ROOT / source).is_file():
            errors.append(f"{context}.source does not exist: {source}")
        require_string(errors, fixture, "command", context)
        result_artifact = require_string(errors, fixture, "result_artifact", context)
        if require_covered and result_artifact and not is_repo_relative_path(result_artifact):
            errors.append(
                f"{context}.result_artifact must be repo-relative when --require-covered is set"
            )
        elif (
            require_covered
            and result_artifact
            and not is_canonical_result_artifact_path(result_artifact)
        ):
            errors.append(
                f"{context}.result_artifact must live under {LATENCY_RESULT_ARTIFACT_DIR} "
                "with a .json extension when --require-covered is set"
            )
        result_path = ROOT / result_artifact if result_artifact else None
        result_exists = result_path is not None and result_path.is_file()
        if require_covered and result_artifact and not result_exists:
            errors.append(f"{context}.result_artifact does not exist: {result_artifact}")
        if (
            require_covered
            and result_artifact
            and result_exists
            and not git_path_is_tracked(result_artifact)
        ):
            errors.append(
                f"{context}.result_artifact must be tracked by git when --require-covered is set"
            )

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
                require_raw_log_file=require_covered,
                require_tracked_files=require_covered,
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
    temp_parent = ROOT / "target"
    temp_parent.mkdir(parents=True, exist_ok=True)
    result_parent = ROOT / LATENCY_RESULT_ARTIFACT_DIR
    raw_parent = ROOT / LATENCY_RAW_LOG_DIR
    result_parent.mkdir(parents=True, exist_ok=True)
    raw_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="backend-latency-validator-", dir=temp_parent
    ) as tmp, tempfile.TemporaryDirectory(
        prefix="backend-latency-validator-", dir=result_parent
    ) as result_tmp, tempfile.TemporaryDirectory(
        prefix="backend-latency-validator-", dir=raw_parent
    ) as raw_tmp:
        tmp_root = Path(tmp)
        result_root = Path(result_tmp)
        raw_root = Path(raw_tmp)
        source = ROOT / "crates/kiln-tensor/tests/rocm_latency_bench.rs"
        untracked_source = tmp_root / "bench.rs"
        untracked_source.write_text("// fixture source\n")
        artifact = result_root / "result.json"
        raw_log = raw_root / "raw.log"
        raw_log.write_text(
            "KILN_LATENCY_METRIC latency_ms 9.5 ms\n"
            "KILN_LATENCY_METRIC tokens_per_s 125.0 tok/s\n"
        )
        raw_log_sha256 = hashlib.sha256(raw_log.read_bytes()).hexdigest()
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        created_at_utc = "2026-06-06T12:00:00Z"
        git_commit = current_git_commit()
        git_tracked_dirty = False
        source_path = repo_relative_path(source)
        artifact_path = repo_relative_path(artifact)
        raw_log_path = repo_relative_path(raw_log)
        fixture = {
            "id": "cuda_fixture",
            "backend": "cuda",
            "hardware": "fixture",
            "source": source_path,
            "command": "cargo bench",
            "result_artifact": artifact_path,
            "threshold_state": "locked_threshold",
            "metrics": [
                {"name": "latency_ms", "unit": "ms", "comparison": "<=", "max": 10.0},
                {"name": "tokens_per_s", "unit": "tok/s", "comparison": ">=", "max": 100.0},
            ],
        }
        policy = {"covered_gate_requires": REQUIRED_COVERED_GATE_POLICY}
        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors: list[str] = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if errors:
            print(
                json.dumps(
                    {"ok": False, "case": "passing artifact", "errors": errors},
                    indent=2,
                )
            )
            return 1

        result = json.loads(artifact.read_text())
        result["unexpected"] = True
        artifact.write_text(json.dumps(result))
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("contains unknown keys" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "unknown artifact key", "errors": errors},
                    indent=2,
                )
            )
            return 1

        result.pop("unexpected")
        result["metrics"]["undeclared_ms"] = 1.0
        artifact.write_text(json.dumps(result))
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("contains undeclared metrics" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "undeclared metric", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": repo_relative_path(raw_root / "missing.log"),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("raw_log must exist" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "missing raw log", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
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
        if not any("must match raw_log value 9.5" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "raw log value mismatch", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "wrong",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
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
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "wrong",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
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
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(stale_fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
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
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
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
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
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

        dirty_result = json.loads(artifact.read_text())
        dirty_result["created_at_utc"] = created_at_utc
        dirty_result["git_tracked_dirty"] = True
        artifact.write_text(json.dumps(dirty_result))
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("git_tracked_dirty must be false" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "dirty git checkout", "errors": errors},
                    indent=2,
                )
            )
            return 1

        missing_commit_result = json.loads(artifact.read_text())
        missing_commit_result["git_commit"] = "0" * 40
        missing_commit_result["git_tracked_dirty"] = False
        artifact.write_text(json.dumps(missing_commit_result))
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("git_commit must exist in the local repository" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "missing git commit", "errors": errors},
                    indent=2,
                )
            )
            return 1

        missing_commit_source_path = repo_relative_path(untracked_source)
        missing_commit_source_fixture = dict(fixture)
        missing_commit_source_fixture["source"] = missing_commit_source_path
        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(
                        missing_commit_source_fixture
                    ),
                    "hardware": "fixture",
                    "source": missing_commit_source_path,
                    "source_sha256": hashlib.sha256(
                        untracked_source.read_bytes()
                    ).hexdigest(),
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(
            errors,
            missing_commit_source_fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("source must exist at git_commit" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "source missing from git commit",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": "1" * 64,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("source_sha256 does not match source" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "source checksum", "errors": errors},
                    indent=2,
                )
            )
            return 1

        errors = validate_manifest(
            {
                "schema_version": 1,
                "status": "covered",
                "policy": policy,
                "required_backends": ["cuda"],
                "fixtures": [fixture],
                "missing_fixture_slots": [],
            },
            require_covered=False,
        )
        if not any("status covered requires --require-covered" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "covered manifest without strict validation",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1

        errors = validate_manifest(
            {
                "schema_version": 1,
                "status": "fixture_required",
                "policy": policy,
                "required_backends": ["cuda"],
                "fixtures": [fixture],
                "missing_fixture_slots": [],
                "unexpected": True,
            },
            require_covered=False,
        )
        if not any("manifest contains unknown keys" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "unknown manifest key",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1

        errors = validate_manifest(
            {
                "schema_version": 1,
                "status": "fixture_required",
                "required_backends": ["cuda"],
                "fixtures": [fixture],
                "missing_fixture_slots": [],
            },
            require_covered=False,
        )
        if not any("policy must be an object" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "missing manifest policy",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1

        errors = validate_manifest(
            {
                "schema_version": 1,
                "status": "covered",
                "policy": policy,
                "required_backends": ["cuda"],
                "fixtures": [fixture],
                "missing_fixture_slots": [],
            },
            require_covered=True,
        )
        if not any("result_artifact must be tracked by git" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "untracked result artifact",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1
        if not any("raw_log must be tracked by git" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "untracked raw log",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1

        noncanonical_artifact = tmp_root / "result.json"
        noncanonical_artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": raw_log_path,
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        noncanonical_fixture = dict(fixture)
        noncanonical_fixture["result_artifact"] = repo_relative_path(noncanonical_artifact)
        errors = validate_manifest(
            {
                "schema_version": 1,
                "status": "covered",
                "policy": policy,
                "required_backends": ["cuda"],
                "fixtures": [noncanonical_fixture],
                "missing_fixture_slots": [],
            },
            require_covered=True,
        )
        if not any("result_artifact must live under" in error for error in errors):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "case": "noncanonical result artifact",
                        "errors": errors,
                    },
                    indent=2,
                )
            )
            return 1

        noncanonical_raw_log = tmp_root / "raw.log"
        noncanonical_raw_log.write_text(raw_log.read_text())
        noncanonical_raw_log_path = repo_relative_path(noncanonical_raw_log)
        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": noncanonical_raw_log_path,
                    "raw_log_sha256": hashlib.sha256(
                        noncanonical_raw_log.read_bytes()
                    ).hexdigest(),
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("raw_log must live under" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "noncanonical raw log", "errors": errors},
                    indent=2,
                )
            )
            return 1

        artifact.write_text(
            json.dumps(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "created_at_utc": created_at_utc,
                    "git_commit": git_commit,
                    "git_tracked_dirty": git_tracked_dirty,
                    "fixture_id": "cuda_fixture",
                    "backend": "cuda",
                    "status": "passed",
                    "manifest": "fixtures.json",
                    "manifest_schema_version": 1,
                    "fixture_spec_sha256": fixture_spec_sha256(fixture),
                    "hardware": "fixture",
                    "source": source_path,
                    "source_sha256": source_sha256,
                    "command": "cargo bench",
                    "raw_log": str(raw_log),
                    "raw_log_sha256": raw_log_sha256,
                    "metrics": {"latency_ms": 9.5, "tokens_per_s": 125.0},
                }
            )
        )
        errors = []
        validate_result_artifact(
            errors,
            fixture,
            artifact,
            "fixtures[0]",
            1,
            None,
            require_raw_log_file=True,
        )
        if not any("raw_log must be repo-relative" in error for error in errors):
            print(
                json.dumps(
                    {"ok": False, "case": "absolute raw log", "errors": errors},
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
