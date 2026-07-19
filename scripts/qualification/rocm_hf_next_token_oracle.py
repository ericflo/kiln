#!/usr/bin/env python3
"""Run a source-bound, memory- and thermal-contained ROCm HF next-token oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hf_next_token_contract as contract
import hf_thermal_supervisor as supervisor
from strict_json import loads as strict_json_loads


ROOT = Path(__file__).resolve().parents[2]
HF_SCRIPT = ROOT / "scripts/qualification/qwen35_hf_logits.py"
SUPERVISOR_SCRIPT = ROOT / "scripts/qualification/hf_thermal_supervisor.py"
MODEL_FINGERPRINT_SCRIPT = ROOT / "scripts/qualification/model_fingerprint.py"
SCHEMA = "kiln.rocm-hf-next-token-oracle.v1"
PASS_PREFIX = "KILN_ROCM_HF_NEXT_TOKEN_ORACLE_PASS "
MEMORY_MAX_GIB = 16
HOST_RESERVE_GIB = 7
MIN_AVAILABLE_GIB = MEMORY_MAX_GIB + HOST_RESERVE_GIB
RUNTIME_MAX_SECONDS = 600
LEGACY_UNGUARDED_FINGERPRINT_RESULTS = {
    "sha256:f65f3a40c1ed2a41c675991a4a7345109efeb8ffb4ef976d2920a735e408751b",
}


class OracleRunError(RuntimeError):
    """The contained next-token oracle did not prove its declared result."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _model_content(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "chat_template_hash": value["chat_template_hash"],
        "config_hash": value["config_hash"],
        "id": value["id"],
        "tokenizer_hash": value["tokenizer_hash"],
        "weight_files": value["weight_files"],
    }


def _bind_model_identity(value: dict[str, Any]) -> dict[str, Any]:
    result = _model_content(value)
    result["content_sha256"] = contract.canonical_sha256(result)
    return result


def _available_gib(meminfo: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return int(fields[1]) // 1024 // 1024
    except (OSError, UnicodeError, ValueError) as exc:
        raise OracleRunError(f"cannot read host MemAvailable: {exc}") from exc
    raise OracleRunError("host MemAvailable is absent from /proc/meminfo")


def _repository_identity() -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise OracleRunError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    if status:
        raise OracleRunError("oracle execution requires a clean worktree")
    commit = git("rev-parse", "HEAD")
    pushed = git("rev-parse", "refs/remotes/origin/main")
    if commit != pushed:
        raise OracleRunError("oracle execution requires HEAD to equal origin/main")
    return {"commit": commit, "origin_main": pushed, "tree": git("rev-parse", "HEAD^{tree}")}


def _validate_executable(path: Path) -> Path:
    path = path.absolute()
    try:
        metadata = path.stat()
    except OSError as exc:
        raise OracleRunError(f"cannot inspect --trainer-python: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
        raise OracleRunError("--trainer-python must resolve to an executable file")
    return path


def _fingerprint_environment(workspace: Path) -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "20260715",
        "TMPDIR": str(workspace),
    }


def _parse_guarded_fingerprint_output(output: str) -> dict[str, Any]:
    try:
        identity = strict_json_loads(output)
    except Exception as exc:
        raise OracleRunError(f"guarded model fingerprint output is invalid JSON: {exc}") from exc
    if not isinstance(identity, dict):
        raise OracleRunError("guarded model fingerprint output must be an object")
    return identity


def _validate_model(
    model: Path,
    expected: dict[str, Any],
    *,
    policy: Path,
    python: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    model = model.absolute()
    if model.is_symlink():
        raise OracleRunError("--model must be a non-symlink directory")
    model = model.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="kiln-model-fingerprint-") as raw_workspace:
        workspace = Path(raw_workspace).resolve(strict=True)
        command = [
            str(python),
            str(MODEL_FINGERPRINT_SCRIPT),
            "--model-path",
            str(model),
            "--model-id",
            expected["id"],
            "--json",
        ]
        returncode, stdout, stderr, thermal = supervisor.supervise(
            policy_path=policy,
            workspace=workspace,
            worker_command=command,
            worker_environment=_fingerprint_environment(workspace),
            worker_phase="model-fingerprint",
        )
    sys.stderr.write(stderr)
    if returncode != 0:
        raise OracleRunError(
            f"thermally guarded model fingerprint exited {returncode}: {stderr[-3000:]}"
        )
    actual_raw = _parse_guarded_fingerprint_output(stdout)
    actual = _bind_model_identity(actual_raw)
    if actual != expected:
        raise OracleRunError("model fingerprint does not match the source-paired request")
    return model, actual, {
        "implementation_sha256": _file_sha256(MODEL_FINGERPRINT_SCRIPT),
        "python_sha256": _file_sha256(python),
        "thermal": thermal,
    }


def validate_model_fingerprint_evidence(
    value: Any,
    *,
    result_sha256: str,
    legacy_result_sha256s: set[str],
) -> dict[str, Any] | None:
    if value is None:
        if result_sha256 not in legacy_result_sha256s:
            raise OracleRunError("model fingerprint thermal evidence is required")
        return None
    if not isinstance(value, dict) or set(value) != {
        "implementation_sha256",
        "python_sha256",
        "thermal",
    }:
        raise OracleRunError("model fingerprint evidence fields are not closed")
    if any(
        not isinstance(value[name], str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value[name]) is None
        for name in ("implementation_sha256", "python_sha256")
    ):
        raise OracleRunError("model fingerprint implementation hashes are not canonical")
    try:
        thermal = supervisor.validate_evidence(value["thermal"])
    except supervisor.SupervisorError as exc:
        raise OracleRunError(str(exc)) from exc
    if thermal != value["thermal"]:
        raise OracleRunError("model fingerprint thermal evidence is inconsistent")
    return value


def _bounded_command(
    *,
    unit: str,
    python: Path,
    model: Path,
    request: Path,
    output: Path,
    policy: Path,
    workspace: Path,
) -> list[str]:
    temporary = workspace / "tmp"
    temporary.mkdir(mode=0o700)
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
        f"MemoryMax={MEMORY_MAX_GIB}G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=kill",
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
        f"HOME={os.environ.get('HOME', str(Path.home()))}",
        "HF_HUB_OFFLINE=1",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        f"PATH={os.environ.get('PATH', '/usr/bin:/bin')}",
        "PYTHONHASHSEED=20260715",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        f"TMPDIR={temporary}",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
        str(python),
        str(SUPERVISOR_SCRIPT),
        "--host-thermal-policy",
        str(policy),
        "--workspace",
        str(workspace),
        "--",
        str(python),
        str(HF_SCRIPT),
        "--model",
        str(model),
        "--output",
        str(output),
        "--request",
        str(request),
    ]


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode(
                    "ascii"
                )
            )
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _repository_relative(path: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(ROOT).as_posix()
    except (OSError, ValueError) as exc:
        raise OracleRunError(f"{label} must be a regular file inside the repository") from exc


def validate_result(path: Path, *, require_current_source: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OracleRunError(f"oracle result is not a regular file: {path}")
    try:
        value = strict_json_loads(path.read_bytes())
    except Exception as exc:
        raise OracleRunError(f"cannot parse oracle result {path}: {exc}") from exc
    fields = {
        "containment",
        "created_at_utc",
        "duration_seconds",
        "implementation",
        "model_identity",
        "oracle",
        "reference_artifact",
        "request",
        "result_sha256",
        "schema",
        "source",
        "verdict",
    }
    allowed_fields = (fields, fields | {"model_fingerprint"})
    if not isinstance(value, dict) or set(value) not in allowed_fields:
        raise OracleRunError("oracle result fields are not closed")
    recorded_hash = value["result_sha256"]
    unsigned = dict(value)
    unsigned.pop("result_sha256")
    if recorded_hash != contract.canonical_sha256(unsigned):
        raise OracleRunError("oracle result_sha256 does not match its content")
    if value["schema"] != SCHEMA:
        raise OracleRunError(f"oracle result schema must equal {SCHEMA}")
    validate_model_fingerprint_evidence(
        value.get("model_fingerprint"),
        result_sha256=recorded_hash,
        legacy_result_sha256s=LEGACY_UNGUARDED_FINGERPRINT_RESULTS,
    )
    request_ref = value["request"]
    if not isinstance(request_ref, dict) or set(request_ref) != {
        "contract_path",
        "id",
        "sha256",
        "source",
    }:
        raise OracleRunError("oracle result request fields are not closed")
    request, request_sha256 = contract.load_request(ROOT / request_ref["contract_path"])
    contract.validate_source_receipts(request, ROOT)
    if (
        request_ref["id"] != request["id"]
        or request_ref["sha256"] != request_sha256
        or request_ref["source"] != request["source"]
        or value["model_identity"] != request["model_identity"]
    ):
        raise OracleRunError("oracle result request/model binding is inconsistent")
    containment = value["containment"]
    if not isinstance(containment, dict) or set(containment) != {
        "host_available_before_gib",
        "memory_max_gib",
        "network",
        "service",
        "swap_max_bytes",
    }:
        raise OracleRunError("oracle result containment fields are not closed")
    try:
        oracle = contract.validate_evidence(value["oracle"])
        thermal = supervisor.validate_evidence(containment["service"])
    except (contract.ContractError, supervisor.SupervisorError) as exc:
        raise OracleRunError(str(exc)) from exc
    if (
        containment["memory_max_gib"] != MEMORY_MAX_GIB
        or containment["swap_max_bytes"] != 0
        or containment["network"] != "forbidden"
        or containment["host_available_before_gib"] < MIN_AVAILABLE_GIB
        or thermal["worker_exit_code"] != 0
    ):
        raise OracleRunError("oracle result containment evidence is inconsistent")
    model_fingerprint = value.get("model_fingerprint")
    if (
        model_fingerprint is not None
        and model_fingerprint["thermal"]["policy"] != thermal["policy"]
    ):
        raise OracleRunError("model fingerprint and oracle thermal policies differ")
    if oracle["request_id"] != request["id"] or oracle["request_sha256"] != request_sha256:
        raise OracleRunError("oracle result HF evidence does not bind its request")
    if oracle["input_token_ids_sha256"] != request["input_token_ids_sha256"]:
        raise OracleRunError("oracle result HF input hash does not bind its request")
    for actual, expected in zip(oracle["candidate_tokens"], request["candidates"]):
        if any(actual[name] != expected[name] for name in ("engine", "text", "token_id")):
            raise OracleRunError("oracle result HF candidate evidence changed")
    matching = [
        item["engine"]
        for item in oracle["candidate_tokens"]
        if item["token_id"] == oracle["argmax"]
    ]
    attribution = matching[0] if len(matching) == 1 else "neither"
    expected_verdict = {
        "argmax_candidate": attribution,
        "argmax_token_id": oracle["argmax"],
        "candidate_attribution_complete": attribution in {"kiln", "vllm"},
    }
    if value["verdict"] != expected_verdict:
        raise OracleRunError("oracle result verdict is inconsistent with HF evidence")
    artifact = value["reference_artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {"bytes", "location", "sha256"}:
        raise OracleRunError("oracle result reference artifact fields are not closed")
    if artifact["location"] != "local_ignored" or artifact["bytes"] != oracle["output_bytes"]:
        raise OracleRunError("oracle result reference artifact evidence is inconsistent")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["sha256"]) is None:
        raise OracleRunError("oracle result reference artifact hash is not canonical")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {"commit", "origin_main", "tree"}:
        raise OracleRunError("oracle result source fields are not closed")
    if source["commit"] != source["origin_main"] or any(
        re.fullmatch(r"[0-9a-f]{40}", source[name]) is None
        for name in ("commit", "origin_main", "tree")
    ):
        raise OracleRunError("oracle result source identity is not a clean pushed commit")
    implementation = value["implementation"]
    if not isinstance(implementation, dict) or set(implementation) != {
        "hf_worker_sha256",
        "python_sha256",
        "supervisor_sha256",
    } or any(
        not isinstance(item, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", item) is None
        for item in implementation.values()
    ):
        raise OracleRunError("oracle result implementation hashes are not closed and canonical")
    if (
        model_fingerprint is not None
        and model_fingerprint["python_sha256"] != implementation["python_sha256"]
    ):
        raise OracleRunError("model fingerprint and oracle interpreter hashes differ")
    if require_current_source:
        if source != _repository_identity():
            raise OracleRunError("oracle result source does not equal the current pushed source")
        expected_implementation = {
            "hf_worker_sha256": _file_sha256(HF_SCRIPT),
            "python_sha256": implementation.get("python_sha256"),
            "supervisor_sha256": _file_sha256(SUPERVISOR_SCRIPT),
        }
        if implementation != expected_implementation:
            raise OracleRunError("oracle implementation hashes do not match current source")
        if value["model_fingerprint"]["implementation_sha256"] != _file_sha256(
            MODEL_FINGERPRINT_SCRIPT
        ):
            raise OracleRunError("model fingerprint implementation does not match current source")
    return value


def execute(
    *,
    model_path: Path,
    python_path: Path,
    request_path: Path,
    policy_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    result_path = result_path.absolute()
    if result_path.exists() or result_path.is_symlink():
        raise OracleRunError(f"refusing to replace result {result_path}")
    for path, label in ((request_path, "request"), (policy_path, "policy")):
        if not path.is_absolute():
            raise OracleRunError(f"--{label} must be absolute")
    request, request_sha256 = contract.load_request(request_path)
    contract.validate_source_receipts(request, ROOT)
    source = _repository_identity()
    python = _validate_executable(python_path)
    policy = policy_path.resolve(strict=True)
    model, model_identity, model_fingerprint = _validate_model(
        model_path,
        request["model_identity"],
        policy=policy,
        python=python,
    )
    if _repository_identity() != source:
        raise OracleRunError("repository identity changed during model fingerprinting")
    available = _available_gib()
    if available < MIN_AVAILABLE_GIB:
        raise OracleRunError(
            f"refusing HF oracle with {available} GiB available; require at least "
            f"{MIN_AVAILABLE_GIB} GiB for {MEMORY_MAX_GIB} GiB service plus "
            f"{HOST_RESERVE_GIB} GiB host reserve"
        )
    workspace = result_path.parent / f".{result_path.stem}.artifacts"
    workspace.mkdir(mode=0o700, parents=True)
    reference = workspace / "hf-reference.safetensors"
    command = _bounded_command(
        unit=f"kiln-rocm-hf-next-token-{uuid.uuid4().hex}.service",
        python=python,
        model=model,
        request=request_path,
        output=reference,
        policy=policy,
        workspace=workspace,
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
    if completed.returncode != 0:
        raise OracleRunError(
            f"bounded HF oracle exited {completed.returncode}: {completed.stderr[-3000:]}"
        )
    try:
        oracle = contract.parse_pass_marker(completed.stdout)
        containment = supervisor.parse_pass_marker(completed.stdout)
    except (contract.ContractError, supervisor.SupervisorError) as exc:
        raise OracleRunError(str(exc)) from exc
    if oracle["request_id"] != request["id"] or oracle["request_sha256"] != request_sha256:
        raise OracleRunError("HF evidence does not bind the requested input")
    if oracle["input_token_ids_sha256"] != request["input_token_ids_sha256"]:
        raise OracleRunError("HF evidence input-token hash disagrees with the request")
    if oracle["input_token_count"] != len(request["input_token_ids"]):
        raise OracleRunError("HF evidence input-token count disagrees with the request")
    for actual, expected in zip(oracle["candidate_tokens"], request["candidates"]):
        if any(actual[name] != expected[name] for name in ("engine", "text", "token_id")):
            raise OracleRunError("HF evidence candidate tokens disagree with the request")
    for name in (
        "memory_high_events",
        "memory_max_events",
        "memory_oom_events",
        "memory_oom_kill_events",
        "memory_swap_bytes",
    ):
        if oracle[name] != 0:
            raise OracleRunError(f"bounded HF oracle reported nonzero {name}={oracle[name]}")
    if not reference.is_file() or reference.is_symlink():
        raise OracleRunError("bounded HF oracle did not retain a regular logits artifact")
    if oracle["output_bytes"] != reference.stat().st_size:
        raise OracleRunError("HF evidence output byte count disagrees with the artifact")
    if _repository_identity() != source:
        raise OracleRunError("repository identity changed during HF oracle execution")
    matching = [
        item["engine"]
        for item in oracle["candidate_tokens"]
        if item["token_id"] == oracle["argmax"]
    ]
    verdict = matching[0] if len(matching) == 1 else "neither"
    result = {
        "containment": {
            "host_available_before_gib": available,
            "memory_max_gib": MEMORY_MAX_GIB,
            "network": "forbidden",
            "service": containment,
            "swap_max_bytes": 0,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_seconds": time.monotonic() - started,
        "implementation": {
            "hf_worker_sha256": _file_sha256(HF_SCRIPT),
            "python_sha256": _file_sha256(python),
            "supervisor_sha256": _file_sha256(SUPERVISOR_SCRIPT),
        },
        "model_fingerprint": model_fingerprint,
        "model_identity": model_identity,
        "oracle": oracle,
        "reference_artifact": {
            "bytes": reference.stat().st_size,
            "location": "local_ignored",
            "sha256": _file_sha256(reference),
        },
        "request": {
            "contract_path": _repository_relative(request_path, "--request"),
            "id": request["id"],
            "sha256": request_sha256,
            "source": request["source"],
        },
        "schema": SCHEMA,
        "source": source,
        "verdict": {
            "argmax_candidate": verdict,
            "argmax_token_id": oracle["argmax"],
            "candidate_attribution_complete": verdict in {"kiln", "vllm"},
        },
    }
    result["result_sha256"] = contract.canonical_sha256(result)
    _write_json_new(result_path, result)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute the contained hardware oracle")
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--trainer-python", required=True, type=Path)
    run.add_argument("--request", required=True, type=Path)
    run.add_argument("--host-thermal-policy", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    check = commands.add_parser("check", help="strictly validate a retained result")
    check.add_argument("result", nargs="+", type=Path)
    check.add_argument("--require-current-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "check":
        for result_path in args.result:
            try:
                result = validate_result(
                    result_path,
                    require_current_source=args.require_current_source,
                )
            except BaseException as exc:
                print(
                    f"ROCm HF next-token oracle result is invalid: "
                    f"{result_path}: {exc}",
                    file=sys.stderr,
                )
                return 1
            print(f"OK {result_path} {result['result_sha256']}")
        return 0
    try:
        result = execute(
            model_path=args.model,
            python_path=args.trainer_python,
            request_path=args.request,
            policy_path=args.host_thermal_policy,
            result_path=args.out,
        )
    except BaseException as exc:
        print(f"ROCm HF next-token oracle failed: {exc}", file=sys.stderr)
        return 1
    print(
        PASS_PREFIX
        + json.dumps(
            {
                "argmax_candidate": result["verdict"]["argmax_candidate"],
                "argmax_token_id": result["verdict"]["argmax_token_id"],
                "request_sha256": result["request"]["sha256"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
