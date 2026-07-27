#!/usr/bin/env python3
"""Build and run the Kiln ROCm first-divergence path attribution."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import hf_next_token_contract as contract
import hf_process_runner as process_runner
import rocm_hf_next_token_oracle as hf_oracle
from strict_json import loads as strict_json_loads


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = "rocm_hf_path_attribution"
BUILD_SCRIPT = ROOT / "scripts/cargo-bounded.sh"
GUARDED_EXEC = ROOT / "scripts/qualification/guarded_exec.py"
PROCESS_RUNNER = ROOT / "scripts/qualification/hf_process_runner.py"
SCHEMA = "kiln.rocm-hf-path-attribution-result.v2"
WORKER_SCHEMA = "kiln.rocm-hf-path-attribution.v2"
WORKER_PREFIX = "KILN_ROCM_HF_PATH_ATTRIBUTION "
RUNTIME_MAX_SECONDS = 900
BUILD_ENVIRONMENT_POLICY = "closed-source-build-v1"


class AttributionError(RuntimeError):
    """The ROCm path-attribution contract or execution is invalid."""


def _path_from_repository(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AttributionError(f"{field} must be a nonempty repository path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AttributionError(f"{field} must stay within the repository")
    resolved = (ROOT / path).resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise AttributionError(f"{field} escapes the repository") from exc
    return resolved


def _relative(path: Path, field: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise AttributionError(f"{field} must resolve inside the repository") from exc


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AttributionError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise AttributionError(f"refusing stale temporary output {temporary}")
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
    ) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    environment = {
        name: value for name, value in source.items() if not name.startswith("KILN_")
    }
    environment.update(
        {
            "CARGO_NET_OFFLINE": "true",
            "KILN_CARGO_ENVIRONMENT_POLICY": BUILD_ENVIRONMENT_POLICY,
            "KILN_CARGO_EXECUTION_MODE": "transient-service",
            "KILN_CARGO_MIN_AVAILABLE_GIB": "1",
            "KILN_CARGO_PRIVATE_NETWORK": "1",
            "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS": "1800",
        }
    )
    return environment


def _build_binary() -> tuple[Path, str, float]:
    environment = _build_environment()
    started = time.monotonic()
    completed = subprocess.run(
        [
            str(BUILD_SCRIPT),
            "build",
            "--release",
            "--locked",
            "--offline",
            "-p",
            "kiln-model",
            "--no-default-features",
            "--features",
            "rocm,hardware-qualification",
            "--example",
            EXAMPLE,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise AttributionError(
            f"bounded release build exited {completed.returncode}: "
            f"{completed.stderr[-3000:]}"
        )
    binary = ROOT / f"target/release/examples/{EXAMPLE}"
    binary = hf_oracle._validate_executable(binary)
    return binary, hf_oracle._file_sha256(binary), time.monotonic() - started


def _worker_spec(
    *,
    workspace: Path,
    binary: Path,
    binary_sha256: str,
    model: Path,
    request: Path,
    reference: Path,
) -> Path:
    home = workspace / "home"
    temporary = workspace / "tmp"
    home.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    spec = {
        "argv": [
            str(binary),
            "--model",
            str(model),
            "--request",
            str(request),
            "--hf-reference",
            str(reference),
        ],
        "cwd": str(ROOT),
        "environment": {
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "RUST_BACKTRACE": "1",
            "TMPDIR": str(temporary),
        },
        "executable": {"path": str(binary), "sha256": binary_sha256},
        "schema": "kiln.guarded-exec.v1",
    }
    path = workspace / "worker-spec.json"
    _write_new(path, spec)
    return path


def _service_command(
    *, python: Path, workspace: Path, spec: Path, unit: str
) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--wait",
        "--collect",
        "--pipe",
        "--quiet",
        "--same-dir",
        "--unit",
        unit,
        "-p",
        "Type=exec",
        "-p",
        "KillMode=control-group",
        "-p",
        "SendSIGKILL=yes",
        "-p",
        "TimeoutStopSec=15s",
        "-p",
        f"RuntimeMaxSec={RUNTIME_MAX_SECONDS}s",
        "-p",
        "PrivateNetwork=yes",
        "/usr/bin/env",
        "-i",
        f"HOME={workspace / 'home'}",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "PATH=/usr/bin:/bin",
        "PYTHONHASHSEED=20260715",
        f"TMPDIR={workspace / 'tmp'}",
        str(python),
        str(PROCESS_RUNNER),
        "--workspace",
        str(workspace),
        "--",
        str(python),
        str(GUARDED_EXEC),
        "--spec",
        str(spec),
    ]


def _parse_worker_marker(output: str) -> dict[str, Any]:
    records = [
        line[len(WORKER_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(records) != 1:
        raise AttributionError(
            f"expected one ROCm path-attribution marker, found {len(records)}"
        )
    try:
        value = strict_json_loads(records[0])
    except Exception as exc:
        raise AttributionError(f"ROCm path-attribution marker is invalid JSON: {exc}") from exc
    return validate_worker_marker(value)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttributionError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise AttributionError(f"{field} must be finite")
    return converted


def _validate_full_path(value: Any, name: str, hf_argmax: int) -> None:
    if not isinstance(value, dict) or set(value) != {"comparison", "observed_next_tokens"}:
        raise AttributionError(f"{name} fields are not closed")
    observed = value["observed_next_tokens"]
    if (
        not isinstance(observed, list)
        or len(observed) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < contract.VOCAB_SIZE
            for item in observed
        )
    ):
        raise AttributionError(f"{name}.observed_next_tokens is invalid")
    comparison = value["comparison"]
    fields = {
        "argmax",
        "argmax_equal",
        "candidate_tokens",
        "cosine_similarity",
        "logits_sha256",
        "max_abs_error",
        "mean_abs_error",
        "top10_overlap",
    }
    if not isinstance(comparison, dict) or set(comparison) != fields:
        raise AttributionError(f"{name}.comparison fields are not closed")
    if (
        isinstance(comparison["argmax"], bool)
        or not isinstance(comparison["argmax"], int)
        or not 0 <= comparison["argmax"] < contract.VOCAB_SIZE
        or not isinstance(comparison["argmax_equal"], bool)
        or comparison["argmax_equal"] is not (comparison["argmax"] == hf_argmax)
        or isinstance(comparison["top10_overlap"], bool)
        or not isinstance(comparison["top10_overlap"], int)
        or not 0 <= comparison["top10_overlap"] <= 10
        or re.fullmatch(r"sha256:[0-9a-f]{64}", comparison["logits_sha256"])
        is None
    ):
        raise AttributionError(f"{name}.comparison identity fields are invalid")
    cosine = _finite_number(
        comparison["cosine_similarity"], f"{name}.comparison.cosine_similarity"
    )
    maximum = _finite_number(
        comparison["max_abs_error"], f"{name}.comparison.max_abs_error"
    )
    mean = _finite_number(
        comparison["mean_abs_error"], f"{name}.comparison.mean_abs_error"
    )
    if not -1 <= cosine <= 1 or maximum < 0 or mean < 0 or mean > maximum:
        raise AttributionError(f"{name}.comparison numerical metrics are invalid")
    candidates = comparison["candidate_tokens"]
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise AttributionError(f"{name}.comparison.candidate_tokens is invalid")
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"logit", "rank", "token_id"}:
            raise AttributionError(f"{name} candidate fields are not closed")
        _finite_number(candidate["logit"], f"{name} candidate logit")
        if (
            isinstance(candidate["rank"], bool)
            or not isinstance(candidate["rank"], int)
            or not 1 <= candidate["rank"] <= contract.VOCAB_SIZE
            or isinstance(candidate["token_id"], bool)
            or not isinstance(candidate["token_id"], int)
            or not 0 <= candidate["token_id"] < contract.VOCAB_SIZE
        ):
            raise AttributionError(f"{name} candidate rank or token is invalid")


def _validate_greedy_path(value: Any, name: str, hf_argmax: int) -> None:
    if not isinstance(value, dict) or set(value) != {
        "final_token_matches_reference",
        "observed_next_tokens",
    }:
        raise AttributionError(f"{name} fields are not closed")
    observed = value["observed_next_tokens"]
    if (
        not isinstance(observed, list)
        or len(observed) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < contract.VOCAB_SIZE
            for item in observed
        )
        or not isinstance(value["final_token_matches_reference"], bool)
        or value["final_token_matches_reference"] is not (observed[-1] == hf_argmax)
    ):
        raise AttributionError(f"{name} token evidence is inconsistent")


def validate_worker_marker(value: Any) -> dict[str, Any]:
    fields = {
        "attribution",
        "eager_full",
        "eager_greedy",
        "graph",
        "graph_full",
        "graph_greedy",
        "hf_argmax",
        "input_token_count",
        "input_token_ids_sha256",
        "kernel_policy",
        "request_id",
        "schema",
    }
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != WORKER_SCHEMA:
        raise AttributionError("ROCm path-attribution marker fields or schema are invalid")
    hf_argmax = value["hf_argmax"]
    if (
        isinstance(hf_argmax, bool)
        or not isinstance(hf_argmax, int)
        or not 0 <= hf_argmax < contract.VOCAB_SIZE
    ):
        raise AttributionError("worker hf_argmax is invalid")
    _validate_full_path(value["eager_full"], "eager_full", hf_argmax)
    _validate_full_path(value["graph_full"], "graph_full", hf_argmax)
    _validate_greedy_path(value["eager_greedy"], "eager_greedy", hf_argmax)
    _validate_greedy_path(value["graph_greedy"], "graph_greedy", hf_argmax)
    expected_prefix = [1206, 5517, 264]
    for name in ("eager_full", "graph_full", "eager_greedy", "graph_greedy"):
        if value[name]["observed_next_tokens"][:3] != expected_prefix:
            raise AttributionError(f"{name} diverged before the declared candidate token")
    graph = value["graph"]
    graph_fields = {
        "cache_admission_successes",
        "capture_attempts",
        "capture_failures",
        "capture_successes",
        "enabled",
        "fallbacks",
        "replay_attempts",
        "replay_failures",
        "replay_successes",
    }
    if (
        not isinstance(graph, dict)
        or set(graph) != graph_fields
        or graph["enabled"] is not True
        or any(
            isinstance(graph[name], bool) or not isinstance(graph[name], int)
            for name in graph_fields - {"enabled"}
        )
        or graph["capture_successes"] < 1
        or graph["cache_admission_successes"] < 1
        or graph["replay_successes"] < 7
        or graph["capture_attempts"]
        < graph["capture_successes"] + graph["capture_failures"]
        or graph["replay_attempts"]
        != graph["replay_successes"] + graph["replay_failures"]
        or graph["capture_failures"] != 0
        or graph["replay_failures"] != 0
        or graph["fallbacks"] != 0
    ):
        raise AttributionError("worker graph evidence does not prove retained replay")
    recomputed = (
        "eager_full_logits"
        if not value["eager_full"]["comparison"]["argmax_equal"]
        else "hip_graph_full_logits"
        if not value["graph_full"]["comparison"]["argmax_equal"]
        else "eager_greedy_selection"
        if not value["eager_greedy"]["final_token_matches_reference"]
        else "hip_graph_greedy_selection"
        if not value["graph_greedy"]["final_token_matches_reference"]
        else "serving_only_or_not_reproduced"
    )
    if value["attribution"] != recomputed or value["kernel_policy"] != "qualified":
        raise AttributionError("worker attribution or kernel policy is inconsistent")
    if (
        value["input_token_count"] != 166
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["input_token_ids_sha256"])
        is None
        or not isinstance(value["request_id"], str)
        or not value["request_id"]
    ):
        raise AttributionError("worker request identity is invalid")
    return value


def execute(
    *,
    model_path: Path,
    request_path: Path,
    oracle_result_path: Path,
    reference_path: Path,
    python_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    if not result_path.is_absolute():
        raise AttributionError("--out must be absolute")
    source = hf_oracle._repository_identity()
    request_path = request_path.resolve(strict=True)
    request, request_sha256 = contract.load_request(request_path)
    contract.validate_source_receipts(request, ROOT)
    oracle_result_path = oracle_result_path.resolve(strict=True)
    oracle_result = hf_oracle.validate_result(oracle_result_path)
    if (
        oracle_result["request"]["id"] != request["id"]
        or oracle_result["request"]["sha256"] != request_sha256
        or oracle_result["oracle"]["input_token_ids_sha256"]
        != request["input_token_ids_sha256"]
    ):
        raise AttributionError("retained HF result does not bind the requested input")
    reference_path = reference_path.resolve(strict=True)
    reference_sha256 = hf_oracle._file_sha256(reference_path)
    if (
        reference_sha256 != oracle_result["reference_artifact"]["sha256"]
        or reference_path.stat().st_size != oracle_result["reference_artifact"]["bytes"]
    ):
        raise AttributionError("raw HF reference does not match the retained result")
    python = hf_oracle._validate_executable(python_path)
    model, model_identity, model_fingerprint = hf_oracle._validate_model(
        model_path,
        request["model_identity"],
        python=python,
    )
    if hf_oracle._repository_identity() != source:
        raise AttributionError("repository identity changed during model fingerprinting")
    available = hf_oracle._available_gib()
    if available < 1:
        raise AttributionError("host reports less than 1 GiB available memory")
    binary, binary_sha256, build_seconds = _build_binary()
    if hf_oracle._repository_identity() != source:
        raise AttributionError("repository identity changed during the bounded build")

    workspace = result_path.parent / f".{result_path.stem}.artifacts"
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    spec = _worker_spec(
        workspace=workspace,
        binary=binary,
        binary_sha256=binary_sha256,
        model=model,
        request=request_path,
        reference=reference_path,
    )
    unit = f"kiln-rocm-hf-path-{uuid.uuid4().hex[:12]}.service"
    command = _service_command(
        python=python,
        workspace=workspace,
        spec=spec,
        unit=unit,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=RUNTIME_MAX_SECONDS + 60,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    combined = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise AttributionError(
            f"guarded ROCm attribution service exited {completed.returncode}: {combined[-4000:]}"
        )
    worker = _parse_worker_marker(combined)
    try:
        process = process_runner.parse_pass_marker(combined)
    except process_runner.RunnerError as exc:
        raise AttributionError(str(exc)) from exc
    if (
        worker["request_id"] != request["id"]
        or worker["input_token_ids_sha256"] != request["input_token_ids_sha256"]
        or worker["hf_argmax"] != oracle_result["oracle"]["argmax"]
        or [
            item["token_id"]
            for item in worker["eager_full"]["comparison"]["candidate_tokens"]
        ]
        != [item["token_id"] for item in request["candidates"]]
    ):
        raise AttributionError("worker output does not match the source-bound request and oracle")
    if hf_oracle._repository_identity() != source:
        raise AttributionError("repository identity changed during ROCm attribution")
    result = {
        "binary": {
            "build_duration_seconds": build_seconds,
            "build_environment_policy": BUILD_ENVIRONMENT_POLICY,
            "path": binary.relative_to(ROOT).as_posix(),
            "sha256": binary_sha256,
        },
        "containment": {
            "host_available_before_gib": available,
            "network": "forbidden",
            "process": process,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_seconds": time.monotonic() - started,
        "implementation": {
            "guarded_exec_sha256": hf_oracle._file_sha256(GUARDED_EXEC),
            "process_runner_sha256": hf_oracle._file_sha256(PROCESS_RUNNER),
            "runner_sha256": hf_oracle._file_sha256(Path(__file__)),
        },
        "model_fingerprint": model_fingerprint,
        "model_identity": model_identity,
        "oracle_reference": {
            "bytes": reference_path.stat().st_size,
            "oracle_result_path": _relative(oracle_result_path, "--oracle-result"),
            "oracle_result_sha256": oracle_result["result_sha256"],
            "raw_location": "local_ignored",
            "raw_sha256": reference_sha256,
        },
        "request": {
            "path": _relative(request_path, "--request"),
            "sha256": request_sha256,
        },
        "schema": SCHEMA,
        "source": source,
        "worker": worker,
    }
    result["result_sha256"] = contract.canonical_sha256(result)
    _write_new(result_path, result)
    return result


def validate_result(path: Path, *, require_current_source: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AttributionError(f"result is not a non-symlink regular file: {path}")
    try:
        value = strict_json_loads(path.read_bytes())
    except Exception as exc:
        raise AttributionError(f"result is invalid JSON: {exc}") from exc
    fields = {
        "binary",
        "containment",
        "created_at_utc",
        "duration_seconds",
        "implementation",
        "model_fingerprint",
        "model_identity",
        "oracle_reference",
        "request",
        "result_sha256",
        "schema",
        "source",
        "worker",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value["schema"] != SCHEMA
    ):
        raise AttributionError("result fields or schema are invalid")
    unsigned = dict(value)
    unsigned.pop("result_sha256")
    expected_hash = contract.canonical_sha256(unsigned)
    if value["result_sha256"] != expected_hash:
        raise AttributionError("result_sha256 is inconsistent")
    try:
        hf_oracle.validate_model_fingerprint_evidence(
            value["model_fingerprint"],
        )
    except hf_oracle.OracleRunError as exc:
        raise AttributionError(str(exc)) from exc
    worker = validate_worker_marker(value["worker"])
    if not isinstance(value["containment"], dict):
        raise AttributionError("result containment must be an object")
    try:
        process = process_runner.validate_evidence(value["containment"].get("process"))
    except process_runner.RunnerError as exc:
        raise AttributionError(str(exc)) from exc
    if (
        set(value["containment"])
        != {"host_available_before_gib", "network", "process"}
        or value["containment"]["network"] != "forbidden"
        or value["containment"]["host_available_before_gib"] < 1
        or process != value["containment"]["process"]
    ):
        raise AttributionError("result containment is inconsistent")
    if not isinstance(value["request"], dict):
        raise AttributionError("result request must be an object")
    request_path = _path_from_repository(value["request"].get("path"), "request.path")
    request, request_sha256 = contract.load_request(request_path)
    contract.validate_source_receipts(request, ROOT)
    if (
        set(value["request"]) != {"path", "sha256"}
        or value["request"]["sha256"] != request_sha256
        or worker["request_id"] != request["id"]
        or worker["input_token_ids_sha256"] != request["input_token_ids_sha256"]
        or worker["input_token_count"] != len(request["input_token_ids"])
        or value["model_identity"] != request["model_identity"]
    ):
        raise AttributionError("result request or model binding is inconsistent")
    continuation = [item["token_id"] for item in request["continuation_prefix"]]
    candidates = [item["token_id"] for item in request["candidates"]]
    if any(
        worker[name]["observed_next_tokens"][:-1] != continuation
        for name in ("eager_full", "graph_full", "eager_greedy", "graph_greedy")
    ) or any(
        [item["token_id"] for item in worker[name]["comparison"]["candidate_tokens"]]
        != candidates
        for name in ("eager_full", "graph_full")
    ):
        raise AttributionError("worker token evidence does not bind the tracked request")
    reference = value["oracle_reference"]
    if not isinstance(reference, dict) or set(reference) != {
        "bytes",
        "oracle_result_path",
        "oracle_result_sha256",
        "raw_location",
        "raw_sha256",
    }:
        raise AttributionError("oracle_reference fields are not closed")
    oracle_path = _path_from_repository(
        reference["oracle_result_path"], "oracle_reference.oracle_result_path"
    )
    oracle = hf_oracle.validate_result(oracle_path)
    if (
        reference["oracle_result_sha256"] != oracle["result_sha256"]
        or reference["raw_sha256"] != oracle["reference_artifact"]["sha256"]
        or reference["bytes"] != oracle["reference_artifact"]["bytes"]
        or reference["raw_location"] != "local_ignored"
        or worker["hf_argmax"] != oracle["oracle"]["argmax"]
        or oracle["request"]["id"] != request["id"]
        or oracle["request"]["sha256"] != request_sha256
    ):
        raise AttributionError("result oracle reference is inconsistent")
    source = value["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"commit", "origin_main", "tree"}
        or source["commit"] != source["origin_main"]
        or any(re.fullmatch(r"[0-9a-f]{40}", source[name]) is None for name in source)
    ):
        raise AttributionError("result source identity is invalid")
    if require_current_source and source != hf_oracle._repository_identity():
        raise AttributionError("result source does not match the current clean pushed source")
    for record_name, expected_fields in (
        (
            "binary",
            {
                "build_duration_seconds",
                "build_environment_policy",
                "path",
                "sha256",
            },
        ),
        (
            "implementation",
            {"guarded_exec_sha256", "process_runner_sha256", "runner_sha256"},
        ),
    ):
        record = value[record_name]
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise AttributionError(f"{record_name} fields are not closed")
    if value["binary"]["path"] != f"target/release/examples/{EXAMPLE}":
        raise AttributionError("result binary path is not the qualified example")
    if value["binary"]["build_environment_policy"] != BUILD_ENVIRONMENT_POLICY:
        raise AttributionError("result binary build policy is inconsistent")
    for digest in [
        value["binary"]["sha256"],
        *value["implementation"].values(),
        reference["raw_sha256"],
    ]:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise AttributionError("result contains an invalid SHA-256 digest")
    if require_current_source:
        expected_implementation = {
            "guarded_exec_sha256": hf_oracle._file_sha256(GUARDED_EXEC),
            "process_runner_sha256": hf_oracle._file_sha256(PROCESS_RUNNER),
            "runner_sha256": hf_oracle._file_sha256(Path(__file__)),
        }
        if value["implementation"] != expected_implementation:
            raise AttributionError("result implementation does not match current source")
        if value["model_fingerprint"]["implementation_sha256"] != hf_oracle._file_sha256(
            hf_oracle.MODEL_FINGERPRINT_SCRIPT
        ):
            raise AttributionError("model fingerprint implementation does not match current source")
    if (
        _finite_number(value["binary"]["build_duration_seconds"], "binary build duration")
        <= 0
        or _finite_number(value["duration_seconds"], "duration_seconds") <= 0
        or isinstance(reference["bytes"], bool)
        or not isinstance(reference["bytes"], int)
        or reference["bytes"] <= 0
    ):
        raise AttributionError("result durations and reference size must be positive")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="build and execute the guarded attribution")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--oracle-result", required=True, type=Path)
    run.add_argument("--hf-reference", required=True, type=Path)
    run.add_argument("--python", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    check = commands.add_parser("check", help="strictly validate retained results")
    check.add_argument("result", nargs="+", type=Path)
    check.add_argument("--require-current-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "check":
        for path in args.result:
            try:
                value = validate_result(path, require_current_source=args.require_current_source)
            except BaseException as exc:
                print(f"ROCm HF path-attribution result is invalid: {path}: {exc}", file=sys.stderr)
                return 1
            print(f"OK {path} {value['result_sha256']}")
        return 0
    try:
        result = execute(
            model_path=args.model,
            request_path=args.request,
            oracle_result_path=args.oracle_result,
            reference_path=args.hf_reference,
            python_path=args.python,
            result_path=args.out,
        )
    except BaseException as exc:
        print(f"ROCm HF path attribution failed: {exc}", file=sys.stderr)
        return 1
    print(f"KILN_ROCM_HF_PATH_ATTRIBUTION_RESULT {result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
