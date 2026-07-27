#!/usr/bin/env python3
"""Run sequential guarded HF and Kiln ROCm layer-boundary attribution."""

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
from typing import Any

import hf_next_token_contract as contract
import hf_process_runner as process_runner
import qwen35_hf_logits as hf_worker
import rocm_hf_next_token_oracle as hf_oracle
import rocm_hf_path_attribution as path_attribution
from strict_json import loads as strict_json_loads


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "kiln.rocm-hf-layer-attribution-result.v2"
WORKER_SCHEMA = "kiln.rocm-hf-layer-attribution.v2"
WORKER_PREFIX = "KILN_ROCM_HF_LAYER_ATTRIBUTION "
HF_RUNTIME_MAX_SECONDS = 600
KERNEL_PROFILES = (
    "qualified",
    "portable_fallback",
    "model_fallback",
    "tensor_fallback",
    "fused_norm_mlp_fallback",
    "fused_norm_mlp_only",
    "fused_rmsnorm_fallback",
    "fused_mlp_silu_mul_fallback",
    "fused_mlp_gate_up_prefill_fallback",
    "gdn_fallback",
    "non_gdn_fallback",
    "split_q_gate_fallback",
    "split_q_gate_only",
)


class LayerAttributionError(RuntimeError):
    """The layer-attribution input, execution, or result is invalid."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerAttributionError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise LayerAttributionError(f"{field} must be finite")
    return converted


def _expected_boundary_names() -> list[str]:
    names = ["embedding"]
    names.extend(
        f"layer_{index:02}_{'full_attention' if (index + 1) % 4 == 0 else 'linear_attention'}"
        for index in range(32)
    )
    names.append("final_norm")
    return names


def _parse_hf_marker(output: str) -> dict[str, Any]:
    records = [
        line[len(hf_worker.LAYER_PASS_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(hf_worker.LAYER_PASS_PREFIX)
    ]
    if len(records) != 1:
        raise LayerAttributionError(
            f"expected one HF layer reference marker, found {len(records)}"
        )
    try:
        value = strict_json_loads(records[0])
    except Exception as exc:
        raise LayerAttributionError(f"HF layer marker is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LayerAttributionError("HF layer marker must be an object")
    extra_fields = {
        "boundary_count",
        "boundary_names",
        "hidden_size",
        "layer_last_rows_sha256",
    }
    if not extra_fields.issubset(value):
        raise LayerAttributionError("HF layer marker omits layer evidence")
    common = {name: item for name, item in value.items() if name not in extra_fields}
    try:
        contract.validate_evidence(common)
    except contract.ContractError as exc:
        raise LayerAttributionError(str(exc)) from exc
    names = value["boundary_names"]
    if (
        value["boundary_count"] != 34
        or value["hidden_size"] != 2560
        or names != _expected_boundary_names()
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["layer_last_rows_sha256"])
        is None
    ):
        raise LayerAttributionError("HF layer marker identity is inconsistent")
    return value


def _bounded_hf_command(
    *,
    unit: str,
    python: Path,
    model: Path,
    request: Path,
    output: Path,
    workspace: Path,
) -> list[str]:
    return [
        *hf_oracle._bounded_command(
            unit=unit,
            python=python,
            model=model,
            request=request,
            output=output,
            workspace=workspace,
        ),
        "--capture-layer-last-rows",
    ]


def _layer_worker_spec(
    *,
    workspace: Path,
    binary: Path,
    binary_sha256: str,
    model: Path,
    request: Path,
    reference: Path,
    kernel_profile: str,
) -> Path:
    if kernel_profile not in KERNEL_PROFILES:
        raise LayerAttributionError(f"unsupported ROCm kernel profile {kernel_profile}")
    home = workspace / "home"
    temporary = workspace / "tmp"
    home.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    spec = {
        "argv": [
            str(binary),
            "--layer-attribution",
            "--model",
            str(model),
            "--request",
            str(request),
            "--hf-reference",
            str(reference),
            "--kernel-profile",
            kernel_profile,
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
    path_attribution._write_new(path, spec)
    return path


def validate_worker_marker(value: Any) -> dict[str, Any]:
    fields = {
        "boundaries",
        "final_logits_sha256",
        "hf_layer_last_rows_sha256",
        "input_token_count",
        "input_token_ids_sha256",
        "kernel_policy",
        "largest_relative_error_growth",
        "observed_next_tokens",
        "request_id",
        "schema",
    }
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != WORKER_SCHEMA:
        raise LayerAttributionError("Kiln layer worker fields or schema are invalid")
    names = _expected_boundary_names()
    boundaries = value["boundaries"]
    metric_fields = {
        "cosine_similarity",
        "hf_sha256",
        "index",
        "kiln_sha256",
        "max_abs_error",
        "mean_abs_error",
        "name",
        "reference_root_mean_square",
        "relative_root_mean_square_error",
        "root_mean_square_error",
    }
    if not isinstance(boundaries, list) or len(boundaries) != len(names):
        raise LayerAttributionError("Kiln layer boundary count is invalid")
    relative_errors: list[float] = []
    for index, (boundary, name) in enumerate(zip(boundaries, names)):
        if (
            not isinstance(boundary, dict)
            or set(boundary) != metric_fields
            or boundary["index"] != index
            or boundary["name"] != name
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", boundary[field]) is None
                for field in ("hf_sha256", "kiln_sha256")
            )
        ):
            raise LayerAttributionError(f"layer boundary {index} identity is invalid")
        cosine = _finite(boundary["cosine_similarity"], f"boundary {index} cosine")
        maximum = _finite(boundary["max_abs_error"], f"boundary {index} max_abs")
        mean = _finite(boundary["mean_abs_error"], f"boundary {index} mean_abs")
        reference_rms = _finite(
            boundary["reference_root_mean_square"], f"boundary {index} reference_rms"
        )
        relative = _finite(
            boundary["relative_root_mean_square_error"],
            f"boundary {index} relative_rmse",
        )
        rmse = _finite(boundary["root_mean_square_error"], f"boundary {index} rmse")
        if (
            not -1 <= cosine <= 1
            or maximum < 0
            or mean < 0
            or mean > maximum
            or reference_rms <= 0
            or relative < 0
            or rmse < 0
            or not math.isclose(relative, rmse / reference_rms, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise LayerAttributionError(f"layer boundary {index} metrics are invalid")
        relative_errors.append(relative)
    growth = value["largest_relative_error_growth"]
    if not isinstance(growth, dict) or set(growth) != {
        "index",
        "name",
        "relative_root_mean_square_error_delta",
    }:
        raise LayerAttributionError("largest error-growth fields are invalid")
    previous = 0.0
    deltas = []
    for relative in relative_errors:
        deltas.append(relative - previous)
        previous = relative
    expected_index = max(range(len(deltas)), key=lambda index: (deltas[index], -index))
    if (
        growth["index"] != expected_index
        or growth["name"] != names[expected_index]
        or not math.isclose(
            _finite(
                growth["relative_root_mean_square_error_delta"],
                "largest relative RMSE delta",
            ),
            deltas[expected_index],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise LayerAttributionError("largest error-growth boundary is inconsistent")
    observed = value["observed_next_tokens"]
    if (
        not isinstance(observed, list)
        or len(observed) != 4
        or observed[:3] != [1206, 5517, 264]
        or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < contract.VOCAB_SIZE for item in observed)
        or value["input_token_count"] != 166
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["input_token_ids_sha256"])
        is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["final_logits_sha256"])
        is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["hf_layer_last_rows_sha256"])
        is None
        or value["kernel_policy"] not in KERNEL_PROFILES
        or not isinstance(value["request_id"], str)
        or not value["request_id"]
    ):
        raise LayerAttributionError("Kiln layer worker request or output evidence is invalid")
    return value


def _parse_worker_marker(output: str) -> dict[str, Any]:
    records = [
        line[len(WORKER_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(records) != 1:
        raise LayerAttributionError(
            f"expected one Kiln layer-attribution marker, found {len(records)}"
        )
    try:
        value = strict_json_loads(records[0])
    except Exception as exc:
        raise LayerAttributionError(f"Kiln layer marker is invalid JSON: {exc}") from exc
    return validate_worker_marker(value)


def execute(
    *,
    model_path: Path,
    request_path: Path,
    python_path: Path,
    result_path: Path,
    kernel_profile: str,
) -> dict[str, Any]:
    started = time.monotonic()
    for path, label in (
        (model_path, "model"),
        (request_path, "request"),
        (python_path, "python"),
        (result_path, "out"),
    ):
        if not path.is_absolute():
            raise LayerAttributionError(f"--{label} must be absolute")
    if result_path.exists() or result_path.is_symlink():
        raise LayerAttributionError(f"refusing to overwrite {result_path}")
    if kernel_profile not in KERNEL_PROFILES:
        raise LayerAttributionError(f"unsupported ROCm kernel profile {kernel_profile}")
    source = hf_oracle._repository_identity()
    request_path = request_path.resolve(strict=True)
    request, request_sha256 = contract.load_request(request_path)
    contract.validate_source_receipts(request, ROOT)
    python = hf_oracle._validate_executable(python_path)
    model, model_identity, model_fingerprint = hf_oracle._validate_model(
        model_path,
        request["model_identity"],
        python=python,
    )
    if hf_oracle._repository_identity() != source:
        raise LayerAttributionError("repository identity changed during model fingerprinting")
    available_before_hf = hf_oracle._available_gib()
    if available_before_hf < 1:
        raise LayerAttributionError("host reports less than 1 GiB available memory")
    workspace = result_path.parent / f".{result_path.stem}.artifacts"
    workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
    hf_workspace = workspace / "hf"
    hf_workspace.mkdir(mode=0o700)
    reference = workspace / "hf-layer-reference.safetensors"
    hf_command = _bounded_hf_command(
        unit=f"kiln-rocm-hf-layers-{uuid.uuid4().hex[:12]}.service",
        python=python,
        model=model,
        request=request_path,
        output=reference,
        workspace=hf_workspace,
    )
    hf_completed = subprocess.run(
        hf_command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=HF_RUNTIME_MAX_SECONDS + 60,
    )
    sys.stdout.write(hf_completed.stdout)
    sys.stderr.write(hf_completed.stderr)
    hf_combined = hf_completed.stdout + "\n" + hf_completed.stderr
    if hf_completed.returncode != 0:
        raise LayerAttributionError(
            f"guarded HF layer service exited {hf_completed.returncode}: {hf_combined[-4000:]}"
        )
    hf_evidence = _parse_hf_marker(hf_combined)
    try:
        hf_process = process_runner.parse_pass_marker(hf_combined)
    except process_runner.RunnerError as exc:
        raise LayerAttributionError(str(exc)) from exc
    if (
        hf_evidence["request_id"] != request["id"]
        or hf_evidence["request_sha256"] != request_sha256
        or hf_evidence["input_token_ids_sha256"] != request["input_token_ids_sha256"]
        or hf_evidence["input_token_count"] != len(request["input_token_ids"])
        or not reference.is_file()
        or reference.is_symlink()
        or hf_evidence["output_bytes"] != reference.stat().st_size
        or any(
            hf_evidence[name] != 0
            for name in (
                "memory_high_events",
                "memory_max_events",
                "memory_oom_events",
                "memory_oom_kill_events",
                "memory_swap_bytes",
            )
        )
    ):
        raise LayerAttributionError("HF layer evidence does not bind the request or containment")
    available_before_kiln = hf_oracle._available_gib()
    if available_before_kiln < 1:
        raise LayerAttributionError("host reports less than 1 GiB available memory")
    binary, binary_sha256, build_seconds = path_attribution._build_binary()
    if hf_oracle._repository_identity() != source:
        raise LayerAttributionError("repository identity changed during layer build")

    kiln_workspace = workspace / "kiln"
    kiln_workspace.mkdir(mode=0o700)
    spec = _layer_worker_spec(
        workspace=kiln_workspace,
        binary=binary,
        binary_sha256=binary_sha256,
        model=model,
        request=request_path,
        reference=reference,
        kernel_profile=kernel_profile,
    )
    kiln_command = path_attribution._service_command(
        python=python,
        workspace=kiln_workspace,
        spec=spec,
        unit=f"kiln-rocm-layer-attribution-{uuid.uuid4().hex[:12]}.service",
    )
    kiln_completed = subprocess.run(
        kiln_command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=path_attribution.RUNTIME_MAX_SECONDS + 60,
    )
    sys.stdout.write(kiln_completed.stdout)
    sys.stderr.write(kiln_completed.stderr)
    kiln_combined = kiln_completed.stdout + "\n" + kiln_completed.stderr
    if kiln_completed.returncode != 0:
        raise LayerAttributionError(
            f"guarded Kiln layer service exited {kiln_completed.returncode}: {kiln_combined[-4000:]}"
        )
    worker = _parse_worker_marker(kiln_combined)
    try:
        kiln_process = process_runner.parse_pass_marker(kiln_combined)
    except process_runner.RunnerError as exc:
        raise LayerAttributionError(str(exc)) from exc
    if (
        worker["request_id"] != request["id"]
        or worker["input_token_ids_sha256"] != request["input_token_ids_sha256"]
        or worker["input_token_count"] != len(request["input_token_ids"])
        or worker["hf_layer_last_rows_sha256"]
        != hf_evidence["layer_last_rows_sha256"]
        or worker["kernel_policy"] != kernel_profile
    ):
        raise LayerAttributionError("Kiln layer evidence does not bind the request")
    if hf_oracle._repository_identity() != source:
        raise LayerAttributionError("repository identity changed during layer attribution")
    result = {
        "binary": {
            "build_duration_seconds": build_seconds,
            "build_environment_policy": path_attribution.BUILD_ENVIRONMENT_POLICY,
            "path": binary.relative_to(ROOT).as_posix(),
            "sha256": binary_sha256,
        },
        "containment": {
            "hf": {
                "host_available_before_gib": available_before_hf,
                "network": "forbidden",
                "process": hf_process,
            },
            "kiln": {
                "host_available_before_gib": available_before_kiln,
                "network": "forbidden",
                "process": kiln_process,
            },
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_seconds": time.monotonic() - started,
        "implementation": {
            "guarded_exec_sha256": hf_oracle._file_sha256(path_attribution.GUARDED_EXEC),
            "hf_worker_sha256": hf_oracle._file_sha256(Path(hf_worker.__file__)),
            "process_runner_sha256": hf_oracle._file_sha256(
                path_attribution.PROCESS_RUNNER
            ),
            "python_sha256": hf_oracle._file_sha256(python),
            "runner_sha256": hf_oracle._file_sha256(Path(__file__)),
        },
        "model_fingerprint": model_fingerprint,
        "model_identity": model_identity,
        "reference": {
            "bytes": reference.stat().st_size,
            "evidence": hf_evidence,
            "location": "local_ignored",
            "sha256": hf_oracle._file_sha256(reference),
        },
        "request": {
            "path": path_attribution._relative(request_path, "--request"),
            "sha256": request_sha256,
        },
        "schema": SCHEMA,
        "source": source,
        "worker": worker,
    }
    result["result_sha256"] = contract.canonical_sha256(result)
    path_attribution._write_new(result_path, result)
    return result


def _repository_path(value: Any, field: str) -> Path:
    return path_attribution._path_from_repository(value, field)


def validate_result(path: Path, *, require_current_source: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LayerAttributionError(f"result is not a non-symlink regular file: {path}")
    try:
        value = strict_json_loads(path.read_bytes())
    except Exception as exc:
        raise LayerAttributionError(f"result is invalid JSON: {exc}") from exc
    fields = {
        "binary",
        "containment",
        "created_at_utc",
        "duration_seconds",
        "implementation",
        "model_fingerprint",
        "model_identity",
        "reference",
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
        raise LayerAttributionError("layer result fields or schema are invalid")
    unsigned = dict(value)
    unsigned.pop("result_sha256")
    if value["result_sha256"] != contract.canonical_sha256(unsigned):
        raise LayerAttributionError("result_sha256 is inconsistent")
    try:
        hf_oracle.validate_model_fingerprint_evidence(
            value["model_fingerprint"],
        )
    except hf_oracle.OracleRunError as exc:
        raise LayerAttributionError(str(exc)) from exc
    worker = validate_worker_marker(value["worker"])
    request_record = value["request"]
    if not isinstance(request_record, dict) or set(request_record) != {"path", "sha256"}:
        raise LayerAttributionError("result request fields are invalid")
    request_path = _repository_path(request_record["path"], "request.path")
    request, request_sha256 = contract.load_request(request_path)
    contract.validate_source_receipts(request, ROOT)
    if (
        request_record["sha256"] != request_sha256
        or value["model_identity"] != request["model_identity"]
        or worker["request_id"] != request["id"]
        or worker["input_token_ids_sha256"] != request["input_token_ids_sha256"]
    ):
        raise LayerAttributionError("layer result request binding is invalid")
    reference = value["reference"]
    if not isinstance(reference, dict) or set(reference) != {
        "bytes",
        "evidence",
        "location",
        "sha256",
    }:
        raise LayerAttributionError("layer result reference fields are invalid")
    evidence = _parse_hf_marker(
        hf_worker.LAYER_PASS_PREFIX
        + json.dumps(reference["evidence"], separators=(",", ":"), sort_keys=True)
    )
    if (
        reference["location"] != "local_ignored"
        or isinstance(reference["bytes"], bool)
        or not isinstance(reference["bytes"], int)
        or reference["bytes"] <= 0
        or evidence["output_bytes"] != reference["bytes"]
        or evidence["request_id"] != request["id"]
        or evidence["request_sha256"] != request_sha256
        or evidence["input_token_ids_sha256"] != request["input_token_ids_sha256"]
        or worker["hf_layer_last_rows_sha256"]
        != evidence["layer_last_rows_sha256"]
        or any(
            evidence[name] != 0
            for name in (
                "memory_high_events",
                "memory_max_events",
                "memory_oom_events",
                "memory_oom_kill_events",
                "memory_swap_bytes",
            )
        )
    ):
        raise LayerAttributionError("layer result HF reference binding is invalid")
    containment = value["containment"]
    if not isinstance(containment, dict) or set(containment) != {"hf", "kiln"}:
        raise LayerAttributionError("layer result containment fields are invalid")
    for name in ("hf", "kiln"):
        record = containment[name]
        if not isinstance(record, dict):
            raise LayerAttributionError(f"layer result {name} containment is invalid")
        try:
            process = process_runner.validate_evidence(record.get("process"))
        except process_runner.RunnerError as exc:
            raise LayerAttributionError(str(exc)) from exc
        if (
            set(record) != {"host_available_before_gib", "network", "process"}
            or record["network"] != "forbidden"
            or record["host_available_before_gib"] < 1
            or process != record["process"]
        ):
            raise LayerAttributionError(f"layer result {name} containment is invalid")
    model_fingerprint = value["model_fingerprint"]
    source = value["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"commit", "origin_main", "tree"}
        or source["commit"] != source["origin_main"]
        or any(re.fullmatch(r"[0-9a-f]{40}", source[name]) is None for name in source)
    ):
        raise LayerAttributionError("layer result source identity is invalid")
    if require_current_source and source != hf_oracle._repository_identity():
        raise LayerAttributionError("layer result source does not match current pushed source")
    binary = value["binary"]
    if (
        not isinstance(binary, dict)
        or set(binary)
        != {
            "build_duration_seconds",
            "build_environment_policy",
            "path",
            "sha256",
        }
        or binary["path"]
        != f"target/release/examples/{path_attribution.EXAMPLE}"
        or binary["build_environment_policy"]
        != path_attribution.BUILD_ENVIRONMENT_POLICY
        or _finite(binary["build_duration_seconds"], "binary build duration") <= 0
    ):
        raise LayerAttributionError("layer result binary identity is invalid")
    implementation = value["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != {
        "guarded_exec_sha256",
        "hf_worker_sha256",
        "process_runner_sha256",
        "python_sha256",
        "runner_sha256",
    }:
        raise LayerAttributionError("layer result implementation fields are invalid")
    if model_fingerprint["python_sha256"] != implementation["python_sha256"]:
        raise LayerAttributionError("model fingerprint and layer interpreter hashes differ")
    digests = [
        value["result_sha256"],
        reference["sha256"],
        binary["sha256"],
        *implementation.values(),
    ]
    if any(re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None for digest in digests):
        raise LayerAttributionError("layer result contains an invalid SHA-256 digest")
    if require_current_source:
        expected_implementation = {
            "guarded_exec_sha256": hf_oracle._file_sha256(path_attribution.GUARDED_EXEC),
            "hf_worker_sha256": hf_oracle._file_sha256(Path(hf_worker.__file__)),
            "process_runner_sha256": hf_oracle._file_sha256(
                path_attribution.PROCESS_RUNNER
            ),
            "python_sha256": implementation["python_sha256"],
            "runner_sha256": hf_oracle._file_sha256(Path(__file__)),
        }
        if implementation != expected_implementation:
            raise LayerAttributionError("layer implementation does not match current source")
        if value["model_fingerprint"]["implementation_sha256"] != hf_oracle._file_sha256(
            hf_oracle.MODEL_FINGERPRINT_SCRIPT
        ):
            raise LayerAttributionError(
                "model fingerprint implementation does not match current source"
            )
    if _finite(value["duration_seconds"], "duration_seconds") <= 0:
        raise LayerAttributionError("layer result duration must be positive")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute sequential guarded layer attribution")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--python", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument(
        "--kernel-profile",
        choices=KERNEL_PROFILES,
        default="qualified",
        help="immutable ROCm model/tensor profile for the Kiln arm",
    )
    check = commands.add_parser("check", help="strictly validate retained layer results")
    check.add_argument("result", nargs="+", type=Path)
    check.add_argument("--require-current-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "check":
        for path in args.result:
            try:
                result = validate_result(path, require_current_source=args.require_current_source)
            except BaseException as exc:
                print(f"ROCm HF layer-attribution result is invalid: {path}: {exc}", file=sys.stderr)
                return 1
            print(f"OK {path} {result['result_sha256']}")
        return 0
    try:
        result = execute(
            model_path=args.model,
            request_path=args.request,
            python_path=args.python,
            result_path=args.out,
            kernel_profile=args.kernel_profile,
        )
    except BaseException as exc:
        print(f"ROCm HF layer attribution failed: {exc}", file=sys.stderr)
        return 1
    print(f"KILN_ROCM_HF_LAYER_ATTRIBUTION_RESULT {result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
