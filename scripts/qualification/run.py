#!/usr/bin/env python3
"""Run one committed qualification workload variant and emit a strict receipt.

The runner never invokes a shell. It captures raw case output below
.qualification/runs, binds the receipt to the exact committed workload and
source tree, and refuses to replace either an existing run directory or receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import environment as environment_module
from model_fingerprint import ModelFingerprintError, fingerprint_model
from receipt import MAX_RESULT_DETAIL_CHARACTERS, validate_receipt
from result_details import compact_details, join_details
from source_tree_hash import HASH_FORMAT, SourceTreeHashError, source_tree_hash
from strict_json import JSON_INTEGER_MAX_DIGITS, loads as strict_json_loads
from workload import (
    MAX_CASE_EXECUTIONS,
    MAX_CASE_TIMEOUT_SECONDS,
    MAX_DECLARED_WALL_SECONDS,
    MAX_REPETITIONS,
    WorkloadLoadError,
    WorkloadValidationError,
    load_workload_document,
    runner_metric_definition,
    validate_workload,
)


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_DIRECTORY = Path("qualification/workloads")
RESULT_PATH_ENVIRONMENT_VARIABLE = "KILN_QUALIFICATION_CASE_RESULT"
VARIANT_ID_ENVIRONMENT_VARIABLE = "KILN_QUALIFICATION_VARIANT_ID"
CASE_ENVIRONMENT_POLICY = "closed-qualification-case-v1"
MACOS_NETWORK_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(allow network* (local ip \"localhost:*\"))
(allow network* (remote ip \"localhost:*\"))
"""
# Host plumbing needed to locate tools and enter the bounded user service. Any
# backend, compiler, device-selection, or product control belongs in the
# committed case environment instead.
CASE_BASE_ENVIRONMENT_NAMES = (
    "CARGO_HOME",
    "DBUS_SESSION_BUS_ADDRESS",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "RUSTUP_HOME",
    "SHELL",
    "TMPDIR",
    "USER",
    "XDG_RUNTIME_DIR",
)
MODEL_REQUIRED_KINDS = {"serving", "performance", "training", "eval", "soak"}
HOST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
RECEIPT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
METRIC_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
CASE_RESULT_KEYS = {
    "schema_version",
    "case_id",
    "status",
    "duration_seconds",
    "effective_config",
    "metrics",
    "tolerances",
    "details",
}
METRIC_KEYS = {"name", "value", "unit", "aggregation", "lower_is_better"}
TOLERANCE_KEYS = {"metric", "absolute_tolerance", "relative_tolerance"}
CASE_RESULT_LIMIT_BYTES = 16 * 1024 * 1024
CASE_OUTPUT_LIMIT_BYTES = 16 * 1024 * 1024
MAX_RUN_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_RUN_STRUCTURED_BYTES = 64 * 1024 * 1024
MAX_TERMINATION_GRACE_SECONDS = 75.0
DEFAULT_TERMINATION_GRACE_SECONDS = 65.0
SUCCESS_DESCENDANT_SETTLEMENT_SECONDS = 1.0


class QualificationRunError(RuntimeError):
    """A preflight or runner-integrity failure that prevents a receipt."""


class CaseResultError(RuntimeError):
    """A command-produced case result failed its closed contract."""


class CaseResultTooLargeError(CaseResultError):
    """A command-produced case result exceeded its evidence budget."""


@dataclass(frozen=True)
class EnvironmentCapture:
    environment: dict[str, Any]
    probe_results: list[dict[str, Any]]
    raw: dict[str, Any]


@dataclass(frozen=True)
class NetworkIsolation:
    mechanism: str
    argv_prefix: tuple[str, ...]


@dataclass(frozen=True)
class RunnerHooks:
    capture_environment: Callable[[str, str, Path], EnvironmentCapture]
    fingerprint_model: Callable[[Path, str | None], dict[str, Any]]
    network_isolation: Callable[[Path], NetworkIsolation]


@dataclass(frozen=True)
class Execution:
    returncode: int | None
    duration_seconds: float
    timed_out: bool
    error: str | None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    result_limit_exceeded: bool = False


@dataclass
class _StreamCapture:
    bytes_seen: int = 0
    bytes_written: int = 0
    truncated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class RunOutcome:
    receipt_path: Path
    receipt: dict[str, Any]
    exit_code: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _canonical_number(value: int | float) -> float:
    converted = float(value)
    return 0.0 if converted == 0.0 else converted


def _within_root(path: Path, root: Path, *, strict: bool = True) -> Path:
    try:
        resolved = path.resolve(strict=strict)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise QualificationRunError(f"path must resolve inside repository {root}: {path}") from exc
    return resolved


def _prospective_repo_path(
    root: Path,
    path: Path,
    *,
    allowed_roots: tuple[Path, ...],
    description: str,
) -> Path:
    candidate = path if path.is_absolute() else root / path
    normalized = Path(os.path.abspath(candidate))
    try:
        relative = normalized.relative_to(root)
    except ValueError as exc:
        raise QualificationRunError(
            f"{description} must stay inside repository {root}: {path}"
        ) from exc
    if not any(
        relative != allowed and relative.is_relative_to(allowed)
        for allowed in allowed_roots
    ):
        choices = " or ".join(str(item) for item in allowed_roots)
        raise QualificationRunError(f"{description} must be below {choices}")

    current = root
    for index, component in enumerate(relative.parts):
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise QualificationRunError(f"cannot inspect {description} path {current}: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise QualificationRunError(
                f"{description} path cannot contain symlinks: {current}"
            )
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise QualificationRunError(
                f"{description} parent is not a directory: {current}"
            )
    return normalized


def _evidence_output_path(root: Path, output: Path) -> Path:
    normalized = _prospective_repo_path(
        root,
        output,
        allowed_roots=(Path("qualification/receipts"), Path(".qualification/receipts")),
        description="receipt output",
    )
    if normalized.suffix != ".json":
        raise QualificationRunError("receipt output must use a .json filename")
    return normalized


def _run_directory_path(root: Path, receipt_id: str) -> Path:
    return _prospective_repo_path(
        root,
        Path(".qualification/runs") / receipt_id,
        allowed_roots=(Path(".qualification/runs"),),
        description="qualification run directory",
    )


def _git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode(errors="replace").strip()
        raise QualificationRunError(
            f"git {' '.join(args)} failed" + (f": {detail}" if detail else "")
        ) from exc
    return completed.stdout


def _git_commit(root: Path) -> str:
    commit = _git(root, "rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise QualificationRunError("Git HEAD is not a lowercase 40-character commit")
    return commit


def _git_clean(root: Path) -> bool:
    return not _git(root, "status", "--porcelain", "--untracked-files=all").strip()


def _committed_workload(root: Path, path: Path) -> tuple[dict[str, Any], bytes, str]:
    resolved = _within_root(path, root)
    try:
        relative = resolved.relative_to(root)
        relative.relative_to(WORKLOAD_DIRECTORY)
    except ValueError as exc:
        raise QualificationRunError(
            f"workload must resolve under {root / WORKLOAD_DIRECTORY}"
        ) from exc
    if resolved.suffix != ".json" or not resolved.is_file() or resolved.is_symlink():
        raise QualificationRunError("workload must be a regular, non-symlink JSON file")
    try:
        workload, raw = load_workload_document(resolved)
    except WorkloadLoadError as exc:
        raise QualificationRunError(str(exc)) from exc
    errors = validate_workload(workload)
    if errors:
        raise QualificationRunError("invalid workload:\n  - " + "\n  - ".join(errors))
    try:
        committed = _git(root, "show", f"HEAD:{relative.as_posix()}")
    except QualificationRunError as exc:
        raise QualificationRunError(
            f"workload must be committed at HEAD: {relative.as_posix()}"
        ) from exc
    if committed != raw:
        raise QualificationRunError(
            f"workload does not exactly match its committed HEAD bytes: {relative.as_posix()}"
        )
    return workload, raw, sha256_bytes(raw)


def _parse_variable_value(raw: str, definition: dict[str, Any]) -> Any:
    variable_type = definition["type"]
    name = definition["name"]
    try:
        if variable_type == "string":
            value: Any = raw
        elif variable_type == "integer":
            if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw) is None:
                raise ValueError("expected a base-10 integer")
            if len(raw.lstrip("-")) > JSON_INTEGER_MAX_DIGITS:
                raise ValueError(
                    f"integer exceeds {JSON_INTEGER_MAX_DIGITS} digits"
                )
            value = int(raw, 10)
        elif variable_type == "number":
            value = strict_json_loads(raw)
            if not _is_finite_number(value):
                raise ValueError("number must be finite")
            value = _canonical_number(value)
        elif variable_type == "boolean":
            if raw not in {"true", "false"}:
                raise ValueError("expected exactly 'true' or 'false'")
            value = raw == "true"
        else:
            raise ValueError(f"unsupported variable type {variable_type!r}")
    except ValueError as exc:
        raise QualificationRunError(f"invalid --var {name}: {exc}") from exc

    constraints = definition["constraints"]
    allowed = constraints["allowed_values"]
    if variable_type == "number":
        try:
            allowed_match = any(
                _is_finite_number(item) and float(item) == value for item in allowed
            )
        except OverflowError:
            allowed_match = False
    else:
        allowed_match = any(_canonical_hash(value) == _canonical_hash(item) for item in allowed)
    if allowed and not allowed_match:
        raise QualificationRunError(f"--var {name} is not one of its allowed values")
    minimum = constraints["minimum"]
    maximum = constraints["maximum"]
    if minimum is not None and value < minimum:
        raise QualificationRunError(f"--var {name} is below its minimum {minimum}")
    if maximum is not None and value > maximum:
        raise QualificationRunError(f"--var {name} is above its maximum {maximum}")
    pattern = constraints["pattern"]
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise QualificationRunError(f"--var {name} does not match its declared pattern")
    return value


def resolve_variables(
    definitions: list[dict[str, Any]],
    assignments: list[str],
    *,
    selected_references: set[str],
) -> dict[str, Any]:
    by_name = {definition["name"]: definition for definition in definitions}
    raw_values: dict[str, str] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise QualificationRunError("--var values must use NAME=VALUE syntax")
        name, raw = assignment.split("=", 1)
        if name not in by_name:
            raise QualificationRunError(f"unknown workload variable {name!r}")
        if name not in selected_references:
            raise QualificationRunError(
                f"workload variable {name!r} is not used by the selected variant"
            )
        if name in raw_values:
            raise QualificationRunError(f"workload variable {name!r} was supplied more than once")
        raw_values[name] = raw

    values: dict[str, Any] = {}
    for name in sorted(selected_references):
        definition = by_name[name]
        if name in raw_values:
            values[name] = _parse_variable_value(raw_values[name], definition)
        elif definition["default"] is not None:
            default = definition["default"]
            values[name] = (
                _canonical_number(default) if definition["type"] == "number" else default
            )
        else:
            raise QualificationRunError(
                f"selected variant references variable {name!r}, but no value or default was provided"
            )
    return values


def _selected_variable_references(variant: dict[str, Any]) -> set[str]:
    references: set[str] = set()
    placeholder = re.compile(r"^\$\{([a-z][a-z0-9_]{1,63}|model_path)\}$")
    for case in variant["cases"]:
        for value in [*case["command"], *case["environment"].values()]:
            match = placeholder.fullmatch(value)
            if match and match.group(1) not in {"seed", "model_path"}:
                references.add(match.group(1))
    return references


def _selected_model_reference(variant: dict[str, Any]) -> bool:
    return any(
        value == "${model_path}"
        for case in variant["cases"]
        for value in [*case["command"], *case["environment"].values()]
    )


def _validate_operational_bounds(
    workload: dict[str, Any], variant: dict[str, Any]
) -> None:
    repetitions = workload["determinism"]["repetitions"]
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or not 1 <= repetitions <= MAX_REPETITIONS
    ):
        raise QualificationRunError(
            f"workload repetitions must be from 1 through {MAX_REPETITIONS}"
        )
    timeout_total = 0
    for case in variant["cases"]:
        timeout = case["timeout_seconds"]
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 1 <= timeout <= MAX_CASE_TIMEOUT_SECONDS
        ):
            raise QualificationRunError(
                f"case {case['id']!r} timeout must be from 1 through "
                f"{MAX_CASE_TIMEOUT_SECONDS} seconds"
            )
        timeout_total += timeout
    execution_count = repetitions * len(variant["cases"])
    if execution_count > MAX_CASE_EXECUTIONS:
        raise QualificationRunError(
            f"selected variant declares more than {MAX_CASE_EXECUTIONS} case executions"
        )
    if repetitions * timeout_total > MAX_DECLARED_WALL_SECONDS:
        raise QualificationRunError(
            f"selected variant declared timeout budget exceeds "
            f"{MAX_DECLARED_WALL_SECONDS} seconds"
        )


def _argument_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    return str(value)


def _resolve_text(
    value: str,
    variables: dict[str, Any],
    seed: int | None,
    model_path: str | None,
) -> str:
    match = re.fullmatch(r"\$\{([a-z][a-z0-9_]{1,63}|seed|model_path)\}", value)
    if match is None:
        return value
    name = match.group(1)
    if name == "seed":
        if seed is None:
            raise QualificationRunError("workload references ${seed} but seed is null")
        return str(seed)
    if name == "model_path":
        if model_path is None:
            raise QualificationRunError(
                "selected variant references ${model_path}, but --model was not provided"
            )
        return model_path
    if name not in variables:
        raise QualificationRunError(f"workload variable {name!r} has no effective value")
    return _argument_text(variables[name])


def _redacted_environment(environment: dict[str, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in sorted(environment.items()):
        sensitive = environment_module.is_sensitive_environment_name(key)
        result[key] = {
            "redacted": sensitive,
            "value": sha256_bytes(value.encode("utf-8")) if sensitive else value,
        }
    return result


def _closed_case_base_environment(environment: dict[str, str]) -> dict[str, str]:
    return {
        name: environment[name]
        for name in CASE_BASE_ENVIRONMENT_NAMES
        if name in environment
    }


def _unavailable_environment(
    backend: str, host_id: str, detail: str
) -> EnvironmentCapture:
    return EnvironmentCapture(
        environment={
            "host_id": host_id,
            "os": environment_module.parse_os_release(),
            "device": {
                "name": "unavailable",
                "architecture": "unavailable",
                "memory_bytes": None,
                "unified_memory": False,
                "driver": "unavailable",
            },
            "runtime": {backend: "unavailable"},
            "compiler": {"python": platform.python_version()},
        },
        probe_results=[
            {
                "id": "backend-environment",
                "required": True,
                "status": "failed",
                "duration_seconds": 0.0,
                "metrics": [],
                "details": detail,
            }
        ],
        raw={"error": detail},
    )


def capture_backend_environment(backend: str, host_id: str, root: Path) -> EnvironmentCapture:
    """Capture current backend identity using the existing environment probes."""

    if root != ROOT:
        return _unavailable_environment(
            backend, host_id, "backend collector only supports the Kiln repository root"
        )
    if backend not in {"rocm", "vulkan", "cuda", "metal"}:
        return _unavailable_environment(
            backend, host_id, f"backend environment collector is not implemented for {backend}"
        )
    raw: dict[str, Any] = {
        "captured_environment": environment_module.captured_environment(),
        "backend": backend,
    }
    try:
        device, runtime, compiler, results = environment_module.collect_backend(backend, raw)
    except Exception as exc:  # The failure must become evidence, not erase the run.
        return _unavailable_environment(backend, host_id, f"environment capture failed: {exc}")
    return EnvironmentCapture(
        environment={
            "host_id": host_id,
            "os": environment_module.parse_os_release(),
            "device": device,
            "runtime": runtime,
            "compiler": compiler,
        },
        probe_results=results,
        raw=raw,
    )


def establish_network_isolation(root: Path) -> NetworkIsolation:
    """Probe a local no-network process wrapper before any run artifacts exist."""

    if sys.platform == "darwin":
        sandbox_exec = shutil.which("sandbox-exec")
        if sandbox_exec is None:
            raise QualificationRunError(
                "sandbox-exec is required to enforce workload network_access='forbidden' on macOS"
            )
        prefix = (sandbox_exec, "-p", MACOS_NETWORK_SANDBOX_PROFILE)
        probe = (
            "import errno,socket; "
            "listener=socket.socket(); "
            "listener.bind(('127.0.0.1',0)); listener.listen(); "
            "external=socket.socket(); "
            "result=external.connect_ex(('192.0.2.1',9)); "
            "assert result in {errno.EACCES,errno.EPERM}, result"
        )
        try:
            completed = subprocess.run(
                [*prefix, sys.executable, "-c", probe],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QualificationRunError(
                f"macOS sandbox network-isolation probe failed: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode(errors="replace").strip()
            raise QualificationRunError(
                "macOS sandbox did not preserve loopback while denying external networking"
                + (f": {detail}" if detail else "")
            )
        return NetworkIsolation("macos-sandbox-loopback-only-v1", prefix)
    if sys.platform != "linux":
        raise QualificationRunError(
            f"network isolation is not implemented on {sys.platform}"
        )
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        raise QualificationRunError(
            "bubblewrap is required to enforce workload network_access='forbidden'"
        )
    prefix = (
        bubblewrap,
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        "--bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--",
    )
    probe = (
        "import socket; "
        "names={name for _,name in socket.if_nameindex()}; "
        "assert names <= {'lo'} and 'lo' in names; "
        "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen()"
    )
    try:
        completed = subprocess.run(
            [*prefix, sys.executable, "-c", probe],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationRunError(f"bubblewrap network-isolation probe failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise QualificationRunError(
            "bubblewrap could not establish isolated loopback-only networking"
            + (f": {detail}" if detail else "")
        )
    return NetworkIsolation("bubblewrap-unshare-net-pid-v1", prefix)


DEFAULT_HOOKS = RunnerHooks(
    capture_environment=capture_backend_environment,
    fingerprint_model=fingerprint_model,
    network_isolation=establish_network_isolation,
)


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaseResultError(f"{context} must be an object")
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown keys: {', '.join(unknown)}")
        raise CaseResultError(f"{context} " + "; ".join(parts))
    return value


def _validate_config_object(value: Any, context: str = "effective_config") -> None:
    if not isinstance(value, dict):
        raise CaseResultError(f"{context} must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", key) is None:
            raise CaseResultError(f"{context} key {key!r} has invalid config syntax")
        if isinstance(item, dict):
            _validate_config_object(item, f"{context}.{key}")
        elif item is None or isinstance(item, (str, bool)) or _is_finite_number(item):
            continue
        else:
            raise CaseResultError(
                f"{context}.{key} must be a finite JSON scalar or nested object"
            )


def _json_equal(left: Any, right: Any) -> bool:
    options = {"allow_nan": False, "sort_keys": True, "separators": (",", ":")}
    return json.dumps(left, **options) == json.dumps(right, **options)


def validate_case_result(
    value: Any,
    *,
    expected_case_id: str,
    declared_metrics: set[str] | None,
    expected_effective_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _exact_keys(value, CASE_RESULT_KEYS, "case result")
    if result["schema_version"] != 1:
        raise CaseResultError("case result schema_version must be 1")
    if result["case_id"] != expected_case_id or not CASE_ID_RE.fullmatch(result["case_id"]):
        raise CaseResultError(f"case result case_id must be {expected_case_id!r}")
    if result["status"] not in {"passed", "failed", "skipped"}:
        raise CaseResultError("case result status must be passed, failed, or skipped")
    if not _is_finite_number(result["duration_seconds"]) or result["duration_seconds"] < 0:
        raise CaseResultError("case result duration_seconds must be finite and non-negative")
    _validate_config_object(result["effective_config"], "case result effective_config")
    if expected_effective_config is not None and not _json_equal(
        result["effective_config"], expected_effective_config
    ):
        raise CaseResultError(
            "case result effective_config does not exactly match the selected variant"
        )
    if result["details"] is not None and not isinstance(result["details"], str):
        raise CaseResultError("case result details must be a string or null")
    if (
        isinstance(result["details"], str)
        and len(result["details"]) > MAX_RESULT_DETAIL_CHARACTERS
    ):
        raise CaseResultError(
            "case result details exceeds "
            f"{MAX_RESULT_DETAIL_CHARACTERS} characters"
        )

    if not isinstance(result["metrics"], list):
        raise CaseResultError("case result metrics must be an array")
    metric_names: list[str] = []
    for index, raw_metric in enumerate(result["metrics"]):
        metric = _exact_keys(raw_metric, METRIC_KEYS, f"case result metrics[{index}]")
        name = metric["name"]
        if not isinstance(name, str) or not METRIC_RE.fullmatch(name):
            raise CaseResultError(f"case result metrics[{index}].name is invalid")
        if not _is_finite_number(metric["value"]):
            raise CaseResultError(f"case result metrics[{index}].value must be finite")
        for key in ("unit", "aggregation"):
            if not isinstance(metric[key], str) or not metric[key]:
                raise CaseResultError(f"case result metrics[{index}].{key} must be non-empty")
        if not isinstance(metric["lower_is_better"], bool):
            raise CaseResultError(
                f"case result metrics[{index}].lower_is_better must be boolean"
            )
        metric_names.append(name)
    if len(metric_names) != len(set(metric_names)):
        raise CaseResultError("case result metrics contain duplicate names")
    if metric_names != sorted(metric_names):
        raise CaseResultError("case result metrics must use ascending name order")
    if declared_metrics is not None and set(metric_names) != declared_metrics:
        missing = sorted(declared_metrics - set(metric_names))
        unknown = sorted(set(metric_names) - declared_metrics)
        parts = []
        if missing:
            parts.append(f"missing declared metrics: {', '.join(missing)}")
        if unknown:
            parts.append(f"undeclared metrics: {', '.join(unknown)}")
        raise CaseResultError("case result metric contract mismatch: " + "; ".join(parts))

    if not isinstance(result["tolerances"], list):
        raise CaseResultError("case result tolerances must be an array")
    tolerance_names: list[str] = []
    for index, raw_tolerance in enumerate(result["tolerances"]):
        tolerance = _exact_keys(
            raw_tolerance, TOLERANCE_KEYS, f"case result tolerances[{index}]"
        )
        name = tolerance["metric"]
        if name not in metric_names:
            raise CaseResultError(
                f"case result tolerances[{index}].metric must name a reported metric"
            )
        for key in ("absolute_tolerance", "relative_tolerance"):
            if not _is_finite_number(tolerance[key]) or tolerance[key] < 0:
                raise CaseResultError(
                    f"case result tolerances[{index}].{key} must be finite and non-negative"
                )
        tolerance_names.append(name)
    if len(tolerance_names) != len(set(tolerance_names)):
        raise CaseResultError("case result tolerances contain duplicate metrics")
    if tolerance_names != sorted(tolerance_names):
        raise CaseResultError("case result tolerances must use ascending metric order")
    return result


def load_case_result(
    path: Path,
    *,
    expected_case_id: str,
    declared_metrics: set[str],
    max_bytes: int = CASE_RESULT_LIMIT_BYTES,
) -> dict[str, Any]:
    if max_bytes <= 0:
        raise QualificationRunError("case result limit must be positive")
    descriptor: int | None = None
    try:
        original_metadata = path.lstat()
        if not stat.S_ISREG(original_metadata.st_mode) or stat.S_ISLNK(
            original_metadata.st_mode
        ):
            raise CaseResultError(
                "command case result must be a regular, non-symlink file"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError as exc:
        raise CaseResultError("command did not create its case result") from exc
    except CaseResultError:
        raise
    except OSError as exc:
        raise CaseResultError(f"cannot open command case result safely: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != original_metadata.st_dev
            or metadata.st_ino != original_metadata.st_ino
        ):
            raise CaseResultError("command case result changed while being opened")
        if metadata.st_size > max_bytes:
            raise CaseResultTooLargeError(
                f"command case result exceeds {max_bytes} byte limit"
            )
        chunks: list[bytes] = []
        bytes_read = 0
        while bytes_read <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - bytes_read))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
        if bytes_read > max_bytes:
            raise CaseResultTooLargeError(
                f"command case result exceeds {max_bytes} byte limit"
            )
        value = strict_json_loads(b"".join(chunks))
    except CaseResultTooLargeError:
        raise
    except (OSError, UnicodeError, ValueError, CaseResultError) as exc:
        raise CaseResultError(f"cannot load command case result: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return validate_case_result(
        value,
        expected_case_id=expected_case_id,
        declared_metrics=declared_metrics,
    )


def _process_group_members(process_group: int) -> tuple[tuple[int, str, int], ...]:
    if sys.platform != "linux":
        return ()
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return ()
    members: list[tuple[int, str, int]] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_line = (entry / "stat").read_text()
            fields = stat_line[stat_line.rfind(")") + 2 :].split()
            state = fields[0]
            parent_pid = int(fields[1])
            member_group = int(fields[2])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
        if member_group == process_group:
            members.append((int(entry.name), state, parent_pid))
    return tuple(members)


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform != "linux":
        return True
    members = _process_group_members(process_group)
    if any(state != "Z" for _pid, state, _parent_pid in members):
        return True
    if members:
        return False
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_for_process_group_exit(process_group: int, grace_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while _group_exists(process_group) and time.monotonic() < deadline:
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return not _group_exists(process_group)


def _signal_process_member(pid: int, process_group: int, signal_number: int) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.pidfd_open(pid)
        members = {
            member_pid: state
            for member_pid, state, _parent_pid in _process_group_members(process_group)
        }
        if pid not in members or members[pid] == "Z":
            return
        signal.pidfd_send_signal(descriptor, signal_number)
    except ProcessLookupError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    process_group = process.pid
    members = _process_group_members(process_group)
    parent_pids = {parent_pid for _pid, _state, parent_pid in members}
    # Interrupt leaf commands while sandbox supervisors remain alive so driver
    # finally blocks can stop separately grouped servers and delete snapshots.
    descendants = [
        pid
        for pid, state, _parent_pid in members
        if pid != process_group and state != "Z" and pid not in parent_pids
    ]
    if descendants:
        for pid in descendants:
            _signal_process_member(pid, process_group, signal.SIGINT)
    else:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and _group_exists(process_group):
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if _group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        _wait_for_process_group_exit(process_group, min(1.0, grace_seconds))
    if _group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(1.0, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _regular_file_exceeds_limit(path: Path, max_bytes: int) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CaseResultError(f"cannot inspect command case result: {exc}") from exc
    return stat.S_ISREG(metadata.st_mode) and metadata.st_size > max_bytes


def execute_argv(
    argv: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
    termination_grace_seconds: float,
    output_limit_bytes: int = CASE_OUTPUT_LIMIT_BYTES,
    result_path: Path | None = None,
    result_limit_bytes: int | None = None,
) -> Execution:
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    if output_limit_bytes <= 0:
        raise QualificationRunError("case output limit must be positive")
    if (result_path is None) != (result_limit_bytes is None):
        raise QualificationRunError(
            "case result path and limit must either both be set or both be omitted"
        )
    if result_limit_bytes is not None and result_limit_bytes <= 0:
        raise QualificationRunError("case result limit must be positive")
    limit_reached = threading.Event()
    stdout_capture = _StreamCapture()
    stderr_capture = _StreamCapture()

    def drain(
        pipe: Any,
        destination: Any,
        capture: _StreamCapture,
        stream_name: str,
    ) -> None:
        try:
            while True:
                chunk = os.read(pipe.fileno(), 64 * 1024)
                if not chunk:
                    break
                capture.bytes_seen += len(chunk)
                remaining = output_limit_bytes - capture.bytes_written
                if remaining > 0:
                    kept = chunk[:remaining]
                    destination.write(kept)
                    capture.bytes_written += len(kept)
                    destination.flush()
                if capture.bytes_seen > output_limit_bytes:
                    capture.truncated = True
                    limit_reached.set()
            destination.flush()
        except Exception as exc:
            capture.error = f"{stream_name} capture failed: {exc}"
            limit_reached.set()
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        threads: list[threading.Thread] = []
        timed_out = False
        result_limit_exceeded = False
        errors: list[str] = []
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
            )
            assert process.stdout is not None and process.stderr is not None
            threads = [
                threading.Thread(
                    target=drain,
                    args=(process.stdout, stdout, stdout_capture, "stdout"),
                    daemon=True,
                ),
                threading.Thread(
                    target=drain,
                    args=(process.stderr, stderr, stderr_capture, "stderr"),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()

            deadline = started + timeout_seconds
            while process.poll() is None:
                if limit_reached.is_set():
                    _terminate_process_group(process, termination_grace_seconds)
                    break
                if result_path is not None and result_limit_bytes is not None:
                    try:
                        result_limit_exceeded = _regular_file_exceeds_limit(
                            result_path, result_limit_bytes
                        )
                    except CaseResultError as exc:
                        errors.append(str(exc))
                        _terminate_process_group(process, termination_grace_seconds)
                        break
                    if result_limit_exceeded:
                        _terminate_process_group(process, termination_grace_seconds)
                        break
                if time.monotonic() >= deadline:
                    timed_out = True
                    _terminate_process_group(process, termination_grace_seconds)
                    break
                time.sleep(0.02)

            returncode = process.wait()
            if (
                not result_limit_exceeded
                and result_path is not None
                and result_limit_bytes is not None
            ):
                try:
                    result_limit_exceeded = _regular_file_exceeds_limit(
                        result_path, result_limit_bytes
                    )
                except CaseResultError as exc:
                    errors.append(str(exc))
            if not timed_out and not limit_reached.is_set() and _group_exists(process.pid):
                settled = _wait_for_process_group_exit(
                    process.pid, SUCCESS_DESCENDANT_SETTLEMENT_SECONDS
                )
                if not settled:
                    _terminate_process_group(process, termination_grace_seconds)
                    errors.append("command left descendant processes running")
            for thread in threads:
                thread.join(timeout=max(1.0, termination_grace_seconds))
            if any(thread.is_alive() for thread in threads):
                errors.append("command output pipes remained open after process cleanup")
            for capture in (stdout_capture, stderr_capture):
                if capture.error:
                    errors.append(capture.error)
            if stdout_capture.truncated:
                errors.append(f"stdout exceeded {output_limit_bytes} byte capture limit")
            if stderr_capture.truncated:
                errors.append(f"stderr exceeded {output_limit_bytes} byte capture limit")
            if timed_out:
                errors.append(f"timed out after {timeout_seconds} seconds")
            if result_limit_exceeded:
                errors.append(
                    f"command case result exceeded {result_limit_bytes} byte limit"
                )
            return Execution(
                returncode,
                time.monotonic() - started,
                timed_out,
                "; ".join(errors) if errors else None,
                stdout_capture.bytes_seen,
                stderr_capture.bytes_seen,
                stdout_capture.truncated,
                stderr_capture.truncated,
                result_limit_exceeded,
            )
        except OSError as exc:
            return Execution(None, time.monotonic() - started, False, str(exc))
        except BaseException:
            if process is not None:
                _terminate_process_group(process, termination_grace_seconds)
            for thread in threads:
                thread.join(timeout=max(1.0, termination_grace_seconds))
            raise


def _assert_output(case: dict[str, Any], stdout: str, stderr: str) -> list[str]:
    streams = {"stdout": stdout, "stderr": stderr, "combined": stdout + "\n" + stderr}
    failures: list[str] = []
    for assertion in case["output_assertions"]:
        matched = re.search(assertion["pattern"], streams[assertion["stream"]]) is not None
        if assertion["match"] == "required" and not matched:
            failures.append(
                f"required {assertion['stream']} pattern did not match: {assertion['pattern']}"
            )
        elif assertion["match"] == "forbidden" and matched:
            failures.append(
                f"forbidden {assertion['stream']} pattern matched: {assertion['pattern']}"
            )
    return failures


def _runner_metrics(
    declared: set[str], execution: Execution, assertion_failures: int, passed: bool
) -> list[dict[str, Any]]:
    metric_values = {
        "case_pass": 1 if passed else 0,
        "exit_code": execution.returncode if execution.returncode is not None else -1,
        "output_assertion_failures": assertion_failures,
    }
    values: dict[str, dict[str, Any]] = {}
    for name, value in metric_values.items():
        definition = runner_metric_definition(name, 1)
        if definition is None:
            raise QualificationRunError(f"runner metric {name!r} has no canonical definition")
        values[name] = {"name": name, "value": value, **definition}
    return [values[name] for name in sorted(declared)]


def _compact_details(value: str) -> str:
    result = compact_details(value, MAX_RESULT_DETAIL_CHARACTERS)
    assert result is not None
    return result


def _join_details(*values: str | None) -> str | None:
    return join_details(*values, max_characters=MAX_RESULT_DETAIL_CHARACTERS)


def _artifact(root: Path, path: Path, kind: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to((root / ".qualification").resolve(strict=True))
        metadata = path.lstat()
    except (OSError, ValueError) as exc:
        raise QualificationRunError(f"local artifact is not under .qualification: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise QualificationRunError(f"local artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    final_metadata = path.lstat()
    if (
        final_metadata.st_dev != metadata.st_dev
        or final_metadata.st_ino != metadata.st_ino
        or final_metadata.st_size != byte_count
    ):
        raise QualificationRunError(f"local artifact changed while hashing: {path}")
    return {
        "kind": kind,
        "location": "local_ignored",
        "path": resolved.relative_to(root).as_posix(),
        "sha256": f"sha256:{digest.hexdigest()}",
        "bytes": byte_count,
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write_json_new(path: Path, value: Any) -> None:
    """Atomically create a JSON document while refusing replacement races."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise QualificationRunError(f"refusing to overwrite existing file: {path}") from exc
        temporary.unlink()
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _normalize_probe_failures(capture: EnvironmentCapture) -> list[str]:
    failures: list[str] = []
    for raw in capture.probe_results:
        if not isinstance(raw, dict):
            failures.append("environment collector returned a malformed probe result")
            continue
        if raw.get("required") is True and raw.get("status") != "passed":
            detail = raw.get("details")
            failures.append(
                f"environment probe {raw.get('id', 'unknown')!r} failed"
                + (f": {detail}" if isinstance(detail, str) and detail else "")
            )
    return failures


def _device_failure(capture: EnvironmentCapture, required: bool) -> str | None:
    if not required:
        return None
    device = capture.environment.get("device")
    if not isinstance(device, dict):
        return "required backend device metadata is missing"
    name = device.get("name")
    architecture = device.get("architecture")
    unavailable = {None, "", "unknown", "unavailable"}
    if name in unavailable or architecture in unavailable:
        return "required backend device was not detected"
    return None


def _aggregate_metrics(
    repetitions: list[dict[str, Any]],
    *,
    producer: str,
) -> tuple[list[dict[str, Any]], str | None]:
    if len(repetitions) == 1:
        return repetitions[0]["metrics"], None
    if not repetitions:
        return [], None
    metric_maps = [
        {metric["name"]: metric for metric in repetition["metrics"]}
        for repetition in repetitions
    ]
    names = set(metric_maps[0])
    if any(set(metric_map) != names for metric_map in metric_maps[1:]):
        return [], "repetition metric names were inconsistent"
    aggregated: list[dict[str, Any]] = []
    for name in sorted(names):
        metrics = [metric_map[name] for metric_map in metric_maps]
        definitions = {
            (metric["unit"], metric["aggregation"], metric["lower_is_better"])
            for metric in metrics
        }
        if len(definitions) != 1:
            return [], f"repetition metric definition for {name!r} was inconsistent"
        unit, aggregation, lower_is_better = next(iter(definitions))
        values = [metric["value"] for metric in metrics]
        if producer == "runner":
            definition = runner_metric_definition(name, len(repetitions))
            if definition is None:
                return [], f"runner metric {name!r} has no canonical definition"
            if name == "case_pass":
                value: float | int = int(all(value == 1 for value in values))
            elif name == "exit_code":
                if any(value != values[0] for value in values[1:]):
                    return [], "exit_code differed across repetitions"
                value = values[0]
            elif name == "output_assertion_failures":
                value = sum(values)
                if not _is_finite_number(value):
                    return [], "output_assertion_failures overflowed across repetitions"
            else:
                return [], f"unsupported runner metric {name!r}"
            aggregated.append({"name": name, "value": value, **definition})
            continue
        try:
            if all(isinstance(item, int) and not isinstance(item, bool) for item in values):
                total = sum(values)
                quotient, remainder = divmod(total, len(values))
                value = quotient if remainder == 0 else total / len(values)
            else:
                total = math.fsum(float(item) for item in values)
                value = total / len(values)
        except (OverflowError, ValueError) as exc:
            return [], f"repetition metric {name!r} could not be aggregated: {exc}"
        if not _is_finite_number(value):
            return [], f"repetition metric {name!r} produced a non-finite mean"
        if value == 0 and total != 0:
            return [], f"repetition metric {name!r} mean underflowed to zero"
        aggregate_name = f"mean_of_{len(metrics)}_{aggregation}"
        aggregated.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "aggregation": aggregate_name,
                "lower_is_better": lower_is_better,
            }
        )
    return aggregated, None


def _metric_policy_failures(
    workload: dict[str, Any], results: list[dict[str, Any]]
) -> list[str]:
    policy = workload["comparison_policy"]
    if not isinstance(policy, dict):
        return []
    by_result = {result["id"]: result for result in results}
    failures: list[str] = []
    for rule in policy["metric_rules"]:
        result = by_result.get(rule["result_id"])
        if result is None:
            failures.append(
                f"policy metric {rule['metric']!r} names missing result {rule['result_id']!r}"
            )
            continue
        metric = next(
            (item for item in result["metrics"] if item["name"] == rule["metric"]),
            None,
        )
        if metric is None:
            if rule["required"]:
                failures.append(
                    f"required policy metric {rule['metric']!r} is missing from "
                    f"result {rule['result_id']!r}"
                )
            continue
        observed = {
            "unit": metric["unit"],
            "aggregation": metric["aggregation"],
            "lower_is_better": metric["lower_is_better"],
        }
        expected = {
            "unit": rule["unit"],
            "aggregation": rule["aggregation"],
            "lower_is_better": rule["lower_is_better"],
        }
        if not _json_equal(observed, expected):
            failures.append(
                f"metric {rule['metric']!r} definition in result "
                f"{rule['result_id']!r} does not match committed policy"
            )
    return failures


def _receipt_id(
    started_at: datetime,
    backend: str,
    host_id: str,
    workload_id: str,
    variant_id: str,
) -> str:
    timestamp = started_at.strftime("%Y%m%dT%H%M%S%fZ").lower()
    identity = hashlib.sha256(f"{workload_id}\0{variant_id}".encode()).hexdigest()[:10]
    value = f"{timestamp}-{backend}-{host_id[:32]}-{workload_id[:24]}-{identity}-v1"
    if not RECEIPT_ID_RE.fullmatch(value):
        raise QualificationRunError(f"generated receipt ID is invalid: {value}")
    return value


def _run_qualification_impl(
    workload_path: Path,
    *,
    variant_id: str,
    host_id: str,
    variable_assignments: list[str] | None = None,
    model_path: Path | None = None,
    model_id: str | None = None,
    output: Path | None = None,
    receipt_id: str | None = None,
    root: Path = ROOT,
    invocation: list[str] | None = None,
    hooks: RunnerHooks = DEFAULT_HOOKS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    active_run_directories: list[Path],
) -> RunOutcome:
    root = root.resolve(strict=True)
    if not HOST_ID_RE.fullmatch(host_id):
        raise QualificationRunError(
            "--host-id must match ^[a-z0-9][a-z0-9._-]{1,63}$"
        )
    if not math.isfinite(termination_grace_seconds) or termination_grace_seconds < 0:
        raise QualificationRunError("termination grace must be finite and non-negative")
    if termination_grace_seconds > MAX_TERMINATION_GRACE_SECONDS:
        raise QualificationRunError(
            f"termination grace must be at most {MAX_TERMINATION_GRACE_SECONDS:g} seconds"
        )
    path = workload_path if workload_path.is_absolute() else root / workload_path
    workload, _workload_raw, workload_hash = _committed_workload(root, path)
    variants = {variant["id"]: variant for variant in workload["variants"]}
    if variant_id not in variants:
        raise QualificationRunError(
            f"variant {variant_id!r} is not declared; choose one of {', '.join(sorted(variants))}"
        )
    variant = variants[variant_id]
    _validate_operational_bounds(workload, variant)
    variant_effective_config = variant["effective_config"]
    _validate_config_object(variant_effective_config, "selected variant effective_config")
    producers = {
        case["result_protocol"]["producer"] for case in variant["cases"]
    }
    if variant_effective_config and producers != {"command"}:
        raise QualificationRunError(
            "a non-empty variant effective_config requires every case to use "
            "a command-produced result"
        )
    selected_references = _selected_variable_references(variant)
    model_referenced = _selected_model_reference(variant)
    variables = resolve_variables(
        workload["variables"],
        variable_assignments or [],
        selected_references=selected_references,
    )

    preflight_receipt_path: Path | None = None
    if output is not None:
        preflight_receipt_path = _evidence_output_path(root, output)
        if preflight_receipt_path.exists():
            raise QualificationRunError(
                f"refusing to overwrite existing receipt: {preflight_receipt_path}"
            )
    if receipt_id is not None:
        if not RECEIPT_ID_RE.fullmatch(receipt_id):
            raise QualificationRunError("--receipt-id has invalid receipt identifier syntax")
        preflight_run_directory = _run_directory_path(root, receipt_id)
        if preflight_run_directory.exists():
            raise QualificationRunError(
                f"refusing to reuse existing run directory: {preflight_run_directory}"
            )
    else:
        _run_directory_path(root, "preflight-generated-receipt-v1")

    # No qualification can pass from a dirty source tree. Refuse before model
    # hashing, device probes, or creation of ignored/raw output.
    if not _git_clean(root):
        raise QualificationRunError(
            "Git worktree must be clean before qualification; no run was started"
        )
    commit = _git_commit(root)
    try:
        tree_hash, _ = source_tree_hash(root)
    except SourceTreeHashError as exc:
        raise QualificationRunError(f"source-tree hash failed: {exc}") from exc

    kind = workload["kind"]
    if model_id is not None and model_path is None:
        raise QualificationRunError("--model-id requires --model")
    if (kind in MODEL_REQUIRED_KINDS or model_referenced) and model_path is None:
        raise QualificationRunError(f"--model is required for {kind!r} workloads")
    if kind == "environment" and model_path is not None:
        raise QualificationRunError("environment workloads do not accept --model")
    model: dict[str, Any] | None = None
    if model_path is not None:
        try:
            model = hooks.fingerprint_model(model_path, model_id)
        except (ModelFingerprintError, OSError) as exc:
            raise QualificationRunError(f"model fingerprint failed: {exc}") from exc
    resolved_model_path = model["path"] if model is not None else None
    network_isolation = hooks.network_isolation(root)

    try:
        preflight_tree_hash, _ = source_tree_hash(root)
    except SourceTreeHashError as exc:
        raise QualificationRunError(f"preflight source-tree verification failed: {exc}") from exc
    if (
        not _git_clean(root)
        or _git_commit(root) != commit
        or preflight_tree_hash != tree_hash
    ):
        raise QualificationRunError(
            "Git HEAD or source tree changed during preflight; no run was started"
        )

    started_at = utc_now()
    started_monotonic = time.monotonic()
    clean_at_start = True
    case_base_environment = _closed_case_base_environment(dict(os.environ))
    execution_environment_hash = _canonical_hash(
        {
            "policy": CASE_ENVIRONMENT_POLICY,
            "environment": case_base_environment,
        }
    )

    resolved_receipt_id = receipt_id or _receipt_id(
        started_at,
        variant["backend"],
        host_id,
        workload["workload_id"],
        variant_id,
    )
    if not RECEIPT_ID_RE.fullmatch(resolved_receipt_id):
        raise QualificationRunError("--receipt-id has invalid receipt identifier syntax")
    receipt_path = preflight_receipt_path or _evidence_output_path(
        root,
        Path(
            f"qualification/receipts/{variant['backend']}/{host_id}/{resolved_receipt_id}.json"
        ),
    )
    if receipt_path.exists():
        raise QualificationRunError(f"refusing to overwrite existing receipt: {receipt_path}")
    run_directory = _run_directory_path(root, resolved_receipt_id)
    if run_directory.exists():
        raise QualificationRunError(
            f"refusing to reuse existing run directory: {run_directory}"
        )
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise QualificationRunError(
            f"refusing to reuse existing run directory: {run_directory}"
        ) from exc
    active_run_directories.append(run_directory)

    infrastructure_failures: list[str] = []
    try:
        capture = hooks.capture_environment(variant["backend"], host_id, root)
    except Exception as exc:  # Hook failures must still produce a failed receipt.
        capture = _unavailable_environment(
            variant["backend"], host_id, f"environment capture failed: {exc}"
        )
    infrastructure_failures.extend(_normalize_probe_failures(capture))
    runtime = capture.environment.get("runtime")
    if isinstance(runtime, dict):
        runtime["execution_environment_sha256"] = execution_environment_hash
    else:
        infrastructure_failures.append(
            "environment collector returned malformed runtime identity"
        )
    device_failure = _device_failure(capture, variant["device_requirement"] == "required")
    if device_failure is not None:
        infrastructure_failures.append(device_failure)

    environment_raw_path = run_directory / "environment.json"
    _atomic_write_json_new(
        environment_raw_path,
        {
            "environment": capture.environment,
            "probe_results": capture.probe_results,
            "raw": capture.raw,
        },
    )
    artifacts = [_artifact(root, environment_raw_path, "environment_probes")]
    run_config_cases: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    effective_config_verified = True
    seed = workload["determinism"]["seed"]
    repetitions = workload["determinism"]["repetitions"]
    execution_count = repetitions * len(variant["cases"])
    per_stream_output_limit = min(
        CASE_OUTPUT_LIMIT_BYTES,
        MAX_RUN_CAPTURE_BYTES // (2 * execution_count),
    )
    per_case_result_limit = min(
        CASE_RESULT_LIMIT_BYTES,
        MAX_RUN_STRUCTURED_BYTES // (2 * execution_count),
    )
    if per_case_result_limit <= 0:
        raise QualificationRunError(
            "structured evidence budget is too small for the selected workload"
        )

    for case_index, case in enumerate(variant["cases"], start=1):
        repetition_results: list[dict[str, Any]] = []
        for repetition in range(1, repetitions + 1):
            case_directory = run_directory / "cases" / (
                f"{case_index:03d}-{case['id']}-r{repetition:03d}"
            )
            case_directory.mkdir(parents=True, exist_ok=False)
            stdout_path = case_directory / "stdout.log"
            stderr_path = case_directory / "stderr.log"
            command_result_path = case_directory / "command-result.json"
            normalized_result_path = case_directory / "case-result.json"
            argv = [
                _resolve_text(item, variables, seed, resolved_model_path)
                for item in case["command"]
            ]
            executed_argv = [*network_isolation.argv_prefix, *argv]
            working_directory = _within_root(root / case["working_directory"], root)
            if not working_directory.is_dir():
                raise QualificationRunError(
                    f"case {case['id']!r} working directory is not a directory"
                )
            overrides = {
                key: _resolve_text(value, variables, seed, resolved_model_path)
                for key, value in case["environment"].items()
            }
            process_environment = dict(case_base_environment)
            process_environment.update(overrides)
            process_environment[RESULT_PATH_ENVIRONMENT_VARIABLE] = str(command_result_path)
            process_environment[VARIANT_ID_ENVIRONMENT_VARIABLE] = variant_id
            run_config_cases.append(
                {
                    "case_id": case["id"],
                    "repetition": repetition,
                    "argv": argv,
                    "executed_argv": executed_argv,
                    "working_directory": working_directory.relative_to(root).as_posix() or ".",
                    "environment_overrides": overrides,
                    "runner_environment": {
                        VARIANT_ID_ENVIRONMENT_VARIABLE: variant_id,
                    },
                    "case_result_path": command_result_path.relative_to(root).as_posix(),
                    "process_environment_sha256": _canonical_hash(process_environment),
                    "timeout_seconds": case["timeout_seconds"],
                    "expected_exit_codes": case["expected_exit_codes"],
                    "output_assertions": case["output_assertions"],
                    "result_protocol": case["result_protocol"],
                }
            )
            execution = execute_argv(
                executed_argv,
                cwd=working_directory,
                environment=process_environment,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_seconds=case["timeout_seconds"],
                termination_grace_seconds=termination_grace_seconds,
                output_limit_bytes=per_stream_output_limit,
                result_path=command_result_path,
                result_limit_bytes=per_case_result_limit,
            )
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
            assertion_failures = _assert_output(case, stdout_text, stderr_text)
            contract_failures: list[str] = []
            if execution.error:
                contract_failures.append(execution.error)
            if execution.timed_out:
                contract_failures.append(f"timed out after {case['timeout_seconds']} seconds")
            if execution.returncode not in case["expected_exit_codes"]:
                contract_failures.append(
                    f"exit code {execution.returncode!r} not in {case['expected_exit_codes']}"
                )
            contract_failures.extend(assertion_failures)
            declared_metrics = set(case["result_protocol"]["declared_metrics"])
            command_result: dict[str, Any] | None = None
            command_result_oversized = execution.result_limit_exceeded
            if case["result_protocol"]["producer"] == "command":
                if not command_result_oversized:
                    try:
                        command_result = load_case_result(
                            command_result_path,
                            expected_case_id=case["id"],
                            declared_metrics=declared_metrics,
                            max_bytes=per_case_result_limit,
                        )
                        if not _json_equal(
                            command_result["effective_config"],
                            variant_effective_config,
                        ):
                            contract_failures.append(
                                "case result effective_config does not exactly match "
                                "the selected variant"
                            )
                            effective_config_verified = False
                    except CaseResultTooLargeError as exc:
                        contract_failures.append(str(exc))
                        command_result_oversized = True
                        effective_config_verified = False
                    except CaseResultError as exc:
                        contract_failures.append(str(exc))
                        effective_config_verified = False
                else:
                    effective_config_verified = False
            elif command_result_path.exists():
                contract_failures.append(
                    "runner-produced case unexpectedly wrote the command-result path"
                )

            if command_result_oversized:
                try:
                    command_result_path.unlink(missing_ok=True)
                except OSError as exc:
                    contract_failures.append(
                        f"cannot discard oversized command case result: {exc}"
                    )

            if command_result is None:
                status = "passed" if not contract_failures else "failed"
                metrics = _runner_metrics(
                    declared_metrics,
                    execution,
                    len(assertion_failures),
                    status == "passed",
                ) if case["result_protocol"]["producer"] == "runner" else []
                details = _join_details(*contract_failures)
                tolerances: list[dict[str, Any]] = []
                observed_effective_config: dict[str, Any] = {}
            else:
                status = command_result["status"]
                if contract_failures:
                    status = "failed"
                metrics = command_result["metrics"]
                tolerances = command_result["tolerances"]
                observed_effective_config = command_result["effective_config"]
                details = _join_details(command_result["details"], *contract_failures)
            normalized = {
                "schema_version": 1,
                "case_id": case["id"],
                "status": status,
                "duration_seconds": execution.duration_seconds,
                "effective_config": observed_effective_config,
                "metrics": metrics,
                "tolerances": tolerances,
                "details": details,
            }
            normalized_bytes = len(_json_bytes(normalized))
            if normalized_bytes > per_case_result_limit:
                effective_config_verified = False
                normalized = {
                    "schema_version": 1,
                    "case_id": case["id"],
                    "status": "failed",
                    "duration_seconds": execution.duration_seconds,
                    "effective_config": {},
                    "metrics": [],
                    "tolerances": [],
                    "details": _join_details(
                        details,
                        "normalized case result exceeded "
                        f"{per_case_result_limit} byte limit",
                    ),
                }
                if len(_json_bytes(normalized)) > per_case_result_limit:
                    raise QualificationRunError(
                        "structured evidence budget cannot hold a minimal failed case result"
                    )
            validate_case_result(
                normalized,
                expected_case_id=case["id"],
                declared_metrics=None,
            )
            _atomic_write_json_new(normalized_result_path, normalized)
            artifacts.extend(
                (
                    _artifact(root, stdout_path, "case_stdout"),
                    _artifact(root, stderr_path, "case_stderr"),
                    _artifact(root, normalized_result_path, "case_result"),
                )
            )
            try:
                command_result_mode = command_result_path.lstat().st_mode
            except FileNotFoundError:
                command_result_mode = None
            if (
                not command_result_oversized
                and command_result_mode is not None
                and stat.S_ISREG(command_result_mode)
            ):
                artifacts.append(_artifact(root, command_result_path, "command_case_result"))
            repetition_results.append(normalized)

        statuses = [item["status"] for item in repetition_results]
        if "failed" in statuses:
            status = "failed"
        elif "skipped" in statuses:
            status = "skipped"
        else:
            status = "passed"
        details = _join_details(
            *(item["details"] for item in repetition_results),
            *(
                infrastructure_failures
                if case["required"] and infrastructure_failures
                else []
            ),
        )
        if case["required"] and infrastructure_failures:
            status = "failed"
        metrics, aggregation_error = _aggregate_metrics(
            repetition_results,
            producer=case["result_protocol"]["producer"],
        )
        if aggregation_error is not None:
            status = "failed"
            details = _join_details(details, aggregation_error)
        for metric in metrics:
            if metric["name"] == "case_pass" and case["result_protocol"]["producer"] == "runner":
                definition = runner_metric_definition("case_pass", repetitions)
                assert definition is not None
                metric.update({"value": 1 if status == "passed" else 0, **definition})
            elif case["result_protocol"]["producer"] == "runner":
                definition = runner_metric_definition(metric["name"], repetitions)
                if definition is not None:
                    metric.update(definition)
        result = {
            "id": case["id"],
            "required": case["required"],
            "status": status,
            "duration_seconds": sum(item["duration_seconds"] for item in repetition_results),
            "metrics": metrics,
            "details": details,
        }
        results.append(result)

    if variant["skip_policy"] == "fail" and any(
        result["status"] == "skipped" for result in results
    ):
        first_required = next(result for result in results if result["required"])
        first_required["status"] = "failed"
        first_required["details"] = _join_details(
            first_required["details"], "variant skip_policy forbids skipped cases"
        )

    infrastructure_failures.extend(_metric_policy_failures(workload, results))

    if model_path is not None:
        try:
            final_model = hooks.fingerprint_model(model_path, model_id)
        except Exception as exc:
            infrastructure_failures.append(f"final model fingerprint failed: {exc}")
        else:
            if not _json_equal(final_model, model):
                infrastructure_failures.append(
                    "model fingerprint changed during qualification"
                )

    try:
        final_commit = _git_commit(root)
        final_clean = _git_clean(root)
        final_tree_hash, _ = source_tree_hash(root)
    except (QualificationRunError, SourceTreeHashError) as exc:
        infrastructure_failures.append(f"final source verification failed: {exc}")
    else:
        if final_commit != commit:
            infrastructure_failures.append(
                f"Git HEAD changed during qualification: {commit} -> {final_commit}"
            )
        if final_tree_hash != tree_hash:
            infrastructure_failures.append(
                f"source tree changed during qualification: {tree_hash} -> {final_tree_hash}"
            )
        if clean_at_start and not final_clean:
            infrastructure_failures.append("Git worktree became dirty during qualification")

    if infrastructure_failures:
        for result in results:
            if not result["required"]:
                continue
            result["status"] = "failed"
            result["details"] = _join_details(
                result["details"], *infrastructure_failures
            )
            for metric in result["metrics"]:
                if metric["name"] == "case_pass":
                    metric.update(
                        {
                            "value": 0,
                            "unit": "bool",
                            "aggregation": "exact",
                            "lower_is_better": False,
                        }
                    )

    run_config_path = run_directory / "run-config.json"
    _atomic_write_json_new(
        run_config_path,
        {
            "schema_version": 1,
            "receipt_id": resolved_receipt_id,
            "workload_id": workload["workload_id"],
            "workload_sha256": workload_hash,
            "variant_id": variant_id,
            "backend": variant["backend"],
            "variables": variables,
            "determinism": workload["determinism"],
            "network_isolation": network_isolation.mechanism,
            "case_execution_count": execution_count,
            "per_stream_output_limit_bytes": per_stream_output_limit,
            "max_run_capture_bytes": MAX_RUN_CAPTURE_BYTES,
            "per_case_result_limit_bytes": per_case_result_limit,
            "max_run_structured_bytes": MAX_RUN_STRUCTURED_BYTES,
            "case_environment_policy": CASE_ENVIRONMENT_POLICY,
            "case_base_environment_sha256": execution_environment_hash,
            "case_base_environment": _redacted_environment(case_base_environment),
            "infrastructure_failures": infrastructure_failures,
            "cases": run_config_cases,
        },
    )
    artifacts.append(_artifact(root, run_config_path, "effective_run_config"))

    finished_at = utc_now()
    duration = time.monotonic() - started_monotonic
    passed = all(not result["required"] or result["status"] == "passed" for result in results)
    effective_invocation = invocation or [
        sys.executable,
        "scripts/qualification/run.py",
        str(path.relative_to(root)),
        "--variant",
        variant_id,
        "--host-id",
        host_id,
    ]
    receipt = {
        "schema_version": 1,
        "receipt_id": resolved_receipt_id,
        "created_at_utc": utc_text(finished_at),
        "source": {
            "tree_hash_format": HASH_FORMAT,
            "tree_hash": tree_hash,
            "git_commit": commit,
            "git_worktree_clean": clean_at_start,
        },
        "qualification": {
            "kind": kind,
            "backend": variant["backend"],
            "profile": workload["workload_id"][:128],
            "verdict": "passed" if passed else "failed",
            "started_at_utc": utc_text(started_at),
            "finished_at_utc": utc_text(finished_at),
            "duration_seconds": duration,
            "command": effective_invocation,
        },
        "environment": capture.environment,
        "model": model,
        "workload": {
            "id": workload["workload_id"],
            "sha256": workload_hash,
            "seed": seed,
            "parameters": {"variant_id": variant_id, **variables},
        },
        "effective_config": (
            variant_effective_config if effective_config_verified else {}
        ),
        "results": results,
        "metrics": [],
        "artifacts": artifacts,
        "unsupported": [],
        "notes": [],
    }
    strict_errors = validate_receipt(
        receipt,
        root=root,
        require_current_source=passed,
        require_local_artifacts=True,
    )
    if strict_errors:
        if passed:
            first_required = next(result for result in receipt["results"] if result["required"])
            first_required["status"] = "failed"
            first_required["details"] = _join_details(
                first_required["details"],
                "runner validation failed: " + " | ".join(strict_errors),
            )
            receipt["qualification"]["verdict"] = "failed"
            receipt["notes"].append(
                "Receipt was downgraded because current-source validation failed."
            )
            structural_errors = validate_receipt(
                receipt, root=root, require_local_artifacts=True
            )
            if structural_errors:
                raise QualificationRunError(
                    "generated failed receipt is invalid:\n  - "
                    + "\n  - ".join(structural_errors)
                )
        else:
            raise QualificationRunError(
                "generated failed receipt is invalid:\n  - " + "\n  - ".join(strict_errors)
            )

    _atomic_write_json_new(receipt_path, receipt)
    active_run_directories.remove(run_directory)
    return RunOutcome(
        receipt_path=receipt_path,
        receipt=receipt,
        exit_code=0 if receipt["qualification"]["verdict"] == "passed" else 1,
    )


def run_qualification(
    workload_path: Path,
    *,
    variant_id: str,
    host_id: str,
    variable_assignments: list[str] | None = None,
    model_path: Path | None = None,
    model_id: str | None = None,
    output: Path | None = None,
    receipt_id: str | None = None,
    root: Path = ROOT,
    invocation: list[str] | None = None,
    hooks: RunnerHooks = DEFAULT_HOOKS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> RunOutcome:
    active_run_directories: list[Path] = []
    try:
        return _run_qualification_impl(
            workload_path,
            variant_id=variant_id,
            host_id=host_id,
            variable_assignments=variable_assignments,
            model_path=model_path,
            model_id=model_id,
            output=output,
            receipt_id=receipt_id,
            root=root,
            invocation=invocation,
            hooks=hooks,
            termination_grace_seconds=termination_grace_seconds,
            active_run_directories=active_run_directories,
        )
    except BaseException as exc:
        cleanup_errors: list[str] = []
        for run_directory in reversed(active_run_directories):
            try:
                shutil.rmtree(run_directory)
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                cleanup_errors.append(f"{run_directory}: {cleanup_exc}")
        if cleanup_errors:
            raise QualificationRunError(
                "cannot clean interrupted qualification run directories: "
                + " | ".join(cleanup_errors)
            ) from exc
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "workload",
        type=Path,
        help="committed JSON manifest below qualification/workloads",
    )
    parser.add_argument("--variant", required=True, help="exact variant ID to execute")
    parser.add_argument(
        "--host-id",
        required=True,
        help="stable, non-secret physical host identifier",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="typed workload variable; repeat for multiple variables",
    )
    parser.add_argument("--model", type=Path, help="local model directory")
    parser.add_argument("--model-id", help="receipt model ID (requires --model)")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "receipt path below qualification/receipts or .qualification/receipts; "
            "defaults to the checked-in evidence tree"
        ),
    )
    parser.add_argument(
        "--receipt-id",
        help="explicit unique receipt ID (normally generated with a UTC timestamp)",
    )
    parser.add_argument(
        "--term-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
        help=(
            "time for driver-first cleanup before containment KILL after a timeout "
            f"(maximum {MAX_TERMINATION_GRACE_SECONDS:g})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)
    invocation = [sys.executable, "scripts/qualification/run.py", *raw_argv]
    try:
        outcome = run_qualification(
            args.workload,
            variant_id=args.variant,
            host_id=args.host_id,
            variable_assignments=args.var,
            model_path=args.model,
            model_id=args.model_id,
            output=args.output,
            receipt_id=args.receipt_id,
            invocation=invocation,
            termination_grace_seconds=args.term_grace_seconds,
        )
    except (QualificationRunError, WorkloadValidationError) as exc:
        print(f"qualification run failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "qualification run interrupted; owned processes and run data were cleaned",
            file=sys.stderr,
        )
        return 130
    try:
        display = outcome.receipt_path.relative_to(ROOT)
    except ValueError:
        display = outcome.receipt_path
    print(display)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
