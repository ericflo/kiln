#!/usr/bin/env python3
"""Validate compact local hardware qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from source_tree_hash import HASH_FORMAT, SourceTreeHashError, source_tree_hash


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
RESULT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
METRIC_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
HOST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
CONFIG_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
JSON_INTEGER_MAX_DIGITS = 4096
MAX_RESULT_DETAIL_CHARACTERS = 2048
KINDS = {"environment", "correctness", "serving", "performance", "training", "eval", "soak"}
BACKENDS = {"cpu", "cuda", "rocm", "vulkan", "metal"}
VERDICTS = {"passed", "failed"}
RESULT_STATUSES = {"passed", "failed", "skipped"}
ARTIFACT_LOCATIONS = {"local_ignored", "external"}

TOP_LEVEL_KEYS = {
    "schema_version",
    "receipt_id",
    "created_at_utc",
    "source",
    "qualification",
    "environment",
    "model",
    "workload",
    "effective_config",
    "results",
    "metrics",
    "artifacts",
    "unsupported",
    "notes",
}
SOURCE_KEYS = {"tree_hash_format", "tree_hash", "git_commit", "git_worktree_clean"}
QUALIFICATION_KEYS = {
    "kind",
    "backend",
    "profile",
    "verdict",
    "started_at_utc",
    "finished_at_utc",
    "duration_seconds",
    "command",
}
ENVIRONMENT_KEYS = {"host_id", "os", "device", "runtime", "compiler"}
OS_KEYS = {"name", "version", "kernel", "architecture"}
DEVICE_KEYS = {"name", "architecture", "memory_bytes", "unified_memory", "driver"}
MODEL_KEYS = {"id", "path", "weight_files", "config_hash", "tokenizer_hash", "chat_template_hash"}
WEIGHT_KEYS = {"path", "sha256", "bytes"}
WORKLOAD_KEYS = {"id", "sha256", "seed", "parameters"}
RESULT_KEYS = {"id", "required", "status", "duration_seconds", "metrics", "details"}
METRIC_KEYS = {"name", "value", "unit", "aggregation", "lower_is_better"}
ARTIFACT_KEYS = {"kind", "location", "path", "sha256", "bytes"}


class ReceiptLoadError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise ReceiptLoadError(f"non-finite JSON number is not allowed: {value}")


def _parse_finite_float(value: str) -> float:
    try:
        exact = Decimal(value)
        parsed = float(value)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise ReceiptLoadError(f"invalid JSON number: {value}") from exc
    if not math.isfinite(parsed):
        raise ReceiptLoadError(f"JSON number overflows finite float range: {value}")
    if parsed == 0.0:
        if exact != 0:
            raise ReceiptLoadError(f"JSON number underflows finite float range: {value}")
        return 0.0
    if Decimal(str(parsed)) != exact:
        raise ReceiptLoadError(f"JSON number is not exactly representable: {value}")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > JSON_INTEGER_MAX_DIGITS:
        raise ReceiptLoadError(f"JSON integer exceeds {JSON_INTEGER_MAX_DIGITS} digits")
    try:
        return int(value)
    except ValueError as exc:
        raise ReceiptLoadError(f"invalid JSON integer: {value}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptLoadError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
    except (OSError, json.JSONDecodeError, ReceiptLoadError) as exc:
        raise ReceiptLoadError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptLoadError(f"{path}: receipt must be a JSON object")
    return value


def _is_number(value: Any) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    except OverflowError:
        return False


def _check_exact_keys(errors: list[str], value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return {}
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing:
        errors.append(f"{context} missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{context} has unknown keys: {', '.join(unknown)}")
    return value


def _check_string(errors: list[str], value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        errors.append(f"{context} must be a {'string' if allow_empty else 'non-empty string'}")
        return ""
    return value


def _check_bool(errors: list[str], value: Any, context: str) -> bool | None:
    if not isinstance(value, bool):
        errors.append(f"{context} must be a boolean")
        return None
    return value


def _check_nonnegative_number(errors: list[str], value: Any, context: str) -> float | None:
    if not _is_number(value) or value < 0:
        errors.append(f"{context} must be a finite non-negative number")
        return None
    return float(value)


def _check_positive_int(errors: list[str], value: Any, context: str) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{context} must be a positive integer")
        return None
    return value


def _check_sha256(errors: list[str], value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{context} must match sha256:<64 lowercase hex characters>")
        return ""
    return value


def _parse_timestamp(errors: list[str], value: Any, context: str) -> datetime | None:
    text = _check_string(errors, value, context)
    if not text or not text.endswith("Z"):
        if text:
            errors.append(f"{context} must be an ISO-8601 UTC timestamp ending in Z")
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        errors.append(f"{context} must be a valid ISO-8601 UTC timestamp")
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"{context} must use UTC")
        return None
    return parsed


def _check_string_map(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, dict) or not value:
        errors.append(f"{context} must be a non-empty object")
        return
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{context} keys must be non-empty strings")
        _check_string(errors, item, f"{context}.{key}")


def _validate_config(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return
    for key, item in value.items():
        if not isinstance(key, str) or not CONFIG_SEGMENT_RE.fullmatch(key):
            errors.append(f"{context} key {key!r} must be a dot-path-compatible segment")
            continue
        item_context = f"{context}.{key}"
        if isinstance(item, dict):
            _validate_config(errors, item, item_context)
        elif item is None or isinstance(item, (str, bool)) or _is_number(item):
            continue
        else:
            errors.append(f"{item_context} must be a finite JSON scalar or nested object")


def _validate_metric(errors: list[str], value: Any, context: str) -> str:
    metric = _check_exact_keys(errors, value, METRIC_KEYS, context)
    name = _check_string(errors, metric.get("name"), f"{context}.name")
    if name and not METRIC_NAME_RE.fullmatch(name):
        errors.append(f"{context}.name has invalid metric syntax")
    if not _is_number(metric.get("value")):
        errors.append(f"{context}.value must be finite numeric")
    _check_string(errors, metric.get("unit"), f"{context}.unit")
    _check_string(errors, metric.get("aggregation"), f"{context}.aggregation")
    _check_bool(errors, metric.get("lower_is_better"), f"{context}.lower_is_better")
    return name


def _validate_metrics(errors: list[str], value: Any, context: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{context} must be an array")
        return
    names: set[str] = set()
    for index, metric in enumerate(value):
        name = _validate_metric(errors, metric, f"{context}[{index}]")
        if name in names:
            errors.append(f"{context} contains duplicate metric name {name!r}")
        names.add(name)


def _validate_model(errors: list[str], value: Any, context: str) -> None:
    model = _check_exact_keys(errors, value, MODEL_KEYS, context)
    _check_string(errors, model.get("id"), f"{context}.id")
    _check_string(errors, model.get("path"), f"{context}.path")
    for key in ("config_hash", "tokenizer_hash"):
        _check_sha256(errors, model.get(key), f"{context}.{key}")
    template_hash = model.get("chat_template_hash")
    if template_hash is not None:
        _check_sha256(errors, template_hash, f"{context}.chat_template_hash")

    weights = model.get("weight_files")
    if not isinstance(weights, list) or not weights:
        errors.append(f"{context}.weight_files must be a non-empty array")
        return
    paths: set[str] = set()
    for index, raw_weight in enumerate(weights):
        weight_context = f"{context}.weight_files[{index}]"
        weight = _check_exact_keys(errors, raw_weight, WEIGHT_KEYS, weight_context)
        path = _check_string(errors, weight.get("path"), f"{weight_context}.path")
        if path in paths:
            errors.append(f"{context}.weight_files contains duplicate path {path!r}")
        paths.add(path)
        _check_sha256(errors, weight.get("sha256"), f"{weight_context}.sha256")
        _check_positive_int(errors, weight.get("bytes"), f"{weight_context}.bytes")


def _validate_workload(errors: list[str], value: Any, context: str) -> None:
    workload = _check_exact_keys(errors, value, WORKLOAD_KEYS, context)
    _check_string(errors, workload.get("id"), f"{context}.id")
    _check_sha256(errors, workload.get("sha256"), f"{context}.sha256")
    seed = workload.get("seed")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
    ):
        errors.append(f"{context}.seed must be null or a non-negative integer")
    if not isinstance(workload.get("parameters"), dict):
        errors.append(f"{context}.parameters must be an object")


def validate_receipt(
    receipt: dict[str, Any],
    *,
    root: Path = ROOT,
    require_current_source: bool = False,
    require_local_artifacts: bool = False,
) -> list[str]:
    errors: list[str] = []
    top = _check_exact_keys(errors, receipt, TOP_LEVEL_KEYS, "receipt")
    if top.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"receipt.schema_version must be {SCHEMA_VERSION}")
    receipt_id = _check_string(errors, top.get("receipt_id"), "receipt.receipt_id")
    if receipt_id and not ID_RE.fullmatch(receipt_id):
        errors.append("receipt.receipt_id has invalid syntax")
    created_at = _parse_timestamp(errors, top.get("created_at_utc"), "receipt.created_at_utc")

    source = _check_exact_keys(errors, top.get("source"), SOURCE_KEYS, "receipt.source")
    if source.get("tree_hash_format") != HASH_FORMAT:
        errors.append(f"receipt.source.tree_hash_format must be {HASH_FORMAT!r}")
    tree_hash = _check_sha256(errors, source.get("tree_hash"), "receipt.source.tree_hash")
    commit = source.get("git_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        errors.append("receipt.source.git_commit must be a lowercase 40-character commit")
    clean = _check_bool(
        errors, source.get("git_worktree_clean"), "receipt.source.git_worktree_clean"
    )

    qualification = _check_exact_keys(
        errors, top.get("qualification"), QUALIFICATION_KEYS, "receipt.qualification"
    )
    kind = qualification.get("kind")
    if kind not in KINDS:
        errors.append(f"receipt.qualification.kind must be one of {sorted(KINDS)}")
    backend = qualification.get("backend")
    if backend not in BACKENDS:
        errors.append(f"receipt.qualification.backend must be one of {sorted(BACKENDS)}")
    _check_string(errors, qualification.get("profile"), "receipt.qualification.profile")
    verdict = qualification.get("verdict")
    if verdict not in VERDICTS:
        errors.append(f"receipt.qualification.verdict must be one of {sorted(VERDICTS)}")
    started_at = _parse_timestamp(
        errors, qualification.get("started_at_utc"), "receipt.qualification.started_at_utc"
    )
    finished_at = _parse_timestamp(
        errors, qualification.get("finished_at_utc"), "receipt.qualification.finished_at_utc"
    )
    duration = _check_nonnegative_number(
        errors, qualification.get("duration_seconds"), "receipt.qualification.duration_seconds"
    )
    command = qualification.get("command")
    if not isinstance(command, list) or not command:
        errors.append("receipt.qualification.command must be a non-empty argv array")
    else:
        for index, item in enumerate(command):
            _check_string(errors, item, f"receipt.qualification.command[{index}]")

    if started_at is not None and finished_at is not None:
        elapsed = (finished_at - started_at).total_seconds()
        if elapsed < 0:
            errors.append("receipt.qualification.finished_at_utc precedes started_at_utc")
        elif duration is not None and abs(duration - elapsed) > 1.0:
            errors.append(
                "receipt.qualification.duration_seconds differs from timestamps by more than 1 second"
            )
        if created_at is not None and created_at < finished_at:
            errors.append("receipt.created_at_utc precedes qualification completion")

    environment = _check_exact_keys(
        errors, top.get("environment"), ENVIRONMENT_KEYS, "receipt.environment"
    )
    host_id = _check_string(errors, environment.get("host_id"), "receipt.environment.host_id")
    if host_id and not HOST_ID_RE.fullmatch(host_id):
        errors.append("receipt.environment.host_id has invalid syntax")
    os_value = _check_exact_keys(errors, environment.get("os"), OS_KEYS, "receipt.environment.os")
    for key in OS_KEYS:
        _check_string(errors, os_value.get(key), f"receipt.environment.os.{key}")
    device = _check_exact_keys(
        errors, environment.get("device"), DEVICE_KEYS, "receipt.environment.device"
    )
    for key in ("name", "architecture", "driver"):
        _check_string(errors, device.get(key), f"receipt.environment.device.{key}")
    memory_bytes = device.get("memory_bytes")
    if memory_bytes is not None:
        _check_positive_int(errors, memory_bytes, "receipt.environment.device.memory_bytes")
    _check_bool(
        errors, device.get("unified_memory"), "receipt.environment.device.unified_memory"
    )
    _check_string_map(errors, environment.get("runtime"), "receipt.environment.runtime")
    _check_string_map(errors, environment.get("compiler"), "receipt.environment.compiler")

    model = top.get("model")
    workload = top.get("workload")
    if model is not None:
        _validate_model(errors, model, "receipt.model")
    if workload is not None:
        _validate_workload(errors, workload, "receipt.workload")
    if kind in {"serving", "performance", "training", "eval", "soak"}:
        if model is None:
            errors.append(f"receipt.model is required for qualification kind {kind!r}")
    if kind in KINDS - {"environment"}:
        if workload is None:
            errors.append(f"receipt.workload is required for qualification kind {kind!r}")
    _validate_config(errors, top.get("effective_config"), "receipt.effective_config")

    results = top.get("results")
    required_failures = 0
    if not isinstance(results, list) or not results:
        errors.append("receipt.results must be a non-empty array")
    else:
        result_ids: set[str] = set()
        for index, raw_result in enumerate(results):
            context = f"receipt.results[{index}]"
            result = _check_exact_keys(errors, raw_result, RESULT_KEYS, context)
            result_id = _check_string(errors, result.get("id"), f"{context}.id")
            if result_id and not RESULT_ID_RE.fullmatch(result_id):
                errors.append(f"{context}.id has invalid syntax")
            if result_id in result_ids:
                errors.append(f"receipt.results contains duplicate id {result_id!r}")
            result_ids.add(result_id)
            required = _check_bool(errors, result.get("required"), f"{context}.required")
            status = result.get("status")
            if status not in RESULT_STATUSES:
                errors.append(f"{context}.status must be one of {sorted(RESULT_STATUSES)}")
            if required and status != "passed":
                required_failures += 1
            _check_nonnegative_number(errors, result.get("duration_seconds"), f"{context}.duration_seconds")
            _validate_metrics(errors, result.get("metrics"), f"{context}.metrics")
            details = result.get("details")
            if details is not None:
                checked_details = _check_string(
                    errors, details, f"{context}.details", allow_empty=True
                )
                if (
                    checked_details is not None
                    and len(checked_details) > MAX_RESULT_DETAIL_CHARACTERS
                ):
                    errors.append(
                        f"{context}.details must be at most "
                        f"{MAX_RESULT_DETAIL_CHARACTERS} characters"
                    )

    _validate_metrics(errors, top.get("metrics"), "receipt.metrics")

    artifacts = top.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("receipt.artifacts must be an array")
    else:
        artifact_keys: set[tuple[str, str]] = set()
        for index, raw_artifact in enumerate(artifacts):
            context = f"receipt.artifacts[{index}]"
            artifact = _check_exact_keys(errors, raw_artifact, ARTIFACT_KEYS, context)
            kind_value = _check_string(errors, artifact.get("kind"), f"{context}.kind")
            location = artifact.get("location")
            if location not in ARTIFACT_LOCATIONS:
                errors.append(f"{context}.location must be one of {sorted(ARTIFACT_LOCATIONS)}")
            path_value = _check_string(errors, artifact.get("path"), f"{context}.path")
            if (kind_value, path_value) in artifact_keys:
                errors.append(f"receipt.artifacts contains duplicate kind/path {kind_value!r}/{path_value!r}")
            artifact_keys.add((kind_value, path_value))
            expected_hash = _check_sha256(errors, artifact.get("sha256"), f"{context}.sha256")
            expected_bytes = artifact.get("bytes")
            if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
                errors.append(f"{context}.bytes must be a non-negative integer")
            if require_local_artifacts and location == "local_ignored" and path_value:
                artifact_path = Path(path_value)
                if not artifact_path.is_absolute():
                    artifact_path = root / artifact_path
                try:
                    resolved = artifact_path.resolve(strict=True)
                    ignored_root = (root / ".qualification").resolve()
                    resolved.relative_to(ignored_root)
                except (OSError, ValueError):
                    errors.append(f"{context}.path must exist under .qualification for local validation")
                else:
                    content = resolved.read_bytes()
                    observed_hash = f"sha256:{hashlib.sha256(content).hexdigest()}"
                    if expected_hash and observed_hash != expected_hash:
                        errors.append(f"{context}.sha256 does not match local artifact")
                    if isinstance(expected_bytes, int) and len(content) != expected_bytes:
                        errors.append(f"{context}.bytes does not match local artifact")

    for key in ("unsupported", "notes"):
        value = top.get(key)
        if not isinstance(value, list):
            errors.append(f"receipt.{key} must be an array")
        else:
            for index, item in enumerate(value):
                _check_string(errors, item, f"receipt.{key}[{index}]")

    if verdict == "passed":
        if clean is not True:
            errors.append("a passed receipt requires receipt.source.git_worktree_clean=true")
        if required_failures:
            errors.append("a passed receipt cannot contain failed or skipped required results")
    elif verdict == "failed" and required_failures == 0:
        errors.append("a failed receipt must contain at least one failed or skipped required result")

    if require_current_source and tree_hash:
        try:
            current_hash, _ = source_tree_hash(root)
        except SourceTreeHashError as exc:
            errors.append(f"cannot compute current source tree hash: {exc}")
        else:
            if current_hash != tree_hash:
                errors.append(
                    f"receipt.source.tree_hash is {tree_hash}, current source tree is {current_hash}"
                )
    return errors


def _git_commit_exists(root: Path, commit: str) -> bool:
    if not COMMIT_RE.fullmatch(commit):
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-current-source", action="store_true")
    parser.add_argument("--require-local-artifacts", action="store_true")
    parser.add_argument("--require-known-commit", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    report: list[dict[str, Any]] = []
    failed = False
    for receipt_path in args.receipts:
        path = receipt_path if receipt_path.is_absolute() else root / receipt_path
        try:
            receipt = load_receipt(path)
        except ReceiptLoadError as exc:
            errors = [str(exc)]
        else:
            errors = validate_receipt(
                receipt,
                root=root,
                require_current_source=args.require_current_source,
                require_local_artifacts=args.require_local_artifacts,
            )
            commit = receipt.get("source", {}).get("git_commit")
            if args.require_known_commit and isinstance(commit, str) and not _git_commit_exists(root, commit):
                errors.append(f"receipt.source.git_commit does not exist locally: {commit}")
        failed = failed or bool(errors)
        report.append({"path": str(receipt_path), "ok": not errors, "errors": errors})

    if args.json_output:
        print(json.dumps({"ok": not failed, "receipts": report}, indent=2, sort_keys=True))
    else:
        for item in report:
            if item["ok"]:
                print(f"OK {item['path']}")
            else:
                print(f"FAILED {item['path']}", file=sys.stderr)
                for error in item["errors"]:
                    print(f"  - {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
