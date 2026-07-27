#!/usr/bin/env python3
"""Qualify model-resident CUDA pressure, release, and request recovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import serve_mixed_load as mixed


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "cuda-model-resident-low-memory"
VARIANT_ID = "cuda-rtx4090-laptop-16gb"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
NETWORK_ENV = "KILN_QUALIFICATION_NETWORK_ISOLATION"
SCOPE_BOUNDARY_ENV = "KILN_WSL2_SCOPE_BOUNDARY"
SCOPE_MEMORY_MAX_ENV = "KILN_WSL2_SCOPE_MEMORY_MAX_BYTES"
SCOPE_PIDS_MAX_ENV = "KILN_WSL2_SCOPE_PIDS_MAX"
SCOPE_CPU_QUOTA_ENV = "KILN_WSL2_SCOPE_CPU_QUOTA_PERCENT"
SCOPE_UNIT_ENV = "KILN_WSL2_SCOPE_UNIT"
SCOPE_HOST_UID_ENV = "KILN_WSL2_SCOPE_HOST_UID"
ALLOWED_RUNNER_KILN_ENV = frozenset(
    {
        RESULT_ENV,
        VARIANT_ENV,
        NETWORK_ENV,
        SCOPE_BOUNDARY_ENV,
        SCOPE_MEMORY_MAX_ENV,
        SCOPE_PIDS_MAX_ENV,
        SCOPE_CPU_QUOTA_ENV,
        SCOPE_UNIT_ENV,
        SCOPE_HOST_UID_ENV,
    }
)
MIB = 1024 * 1024
GIB = 1024 * MIB
OVERALL_TIMEOUT_SECONDS = 3300.0
BUILD_TIMEOUT_SECONDS = 1800.0
REQUEST_TIMEOUT_SECONDS = 300.0
PRESSURE_READY_TIMEOUT_SECONDS = 120.0
RECOVERY_TIMEOUT_SECONDS = 60.0
PEER_SHUTDOWN_SECONDS = 30.0
TARGET_FREE_MIB = 1024
MINIMUM_FREE_MIB = 768
PEER_CHUNK_MIB = 256
PEER_MAX_ALLOCATION_MIB = 1280
PEER_HOLD_SECONDS = 300
PEER_POLL_MILLISECONDS = 100
RECOVERY_TOLERANCE_MIB = 512
HEALTH_RECOVERY_TOLERANCE_MIB = 768
BUILD_MIN_AVAILABLE_GIB = 13
BUILD_HOST_RESERVE_GIB = 3
BUILD_MAX_MEMORY_GIB = 10
REQUEST_PROMPT_WORDS = 16
REQUEST_MAX_TOKENS = 32
KV_BLOCKS = 62
KV_CACHE_BYTES = 130_023_424
MODEL_ID = "Qwen3.5-4B"
MODEL_SOURCE_ID = "Qwen/Qwen3.5-4B"
CUDA_LIBRARY = Path("/usr/lib/wsl/lib/libcuda.so.1")
NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
PEER_SCRIPT = ROOT / "scripts/qualification/cuda_pressure_peer.py"
SERVER_CONFIG = (
    ROOT
    / "qualification/server-config/"
    "kiln-cuda-rtx4090-laptop-serving-bootstrap-v1.toml"
)
SERVER_LAUNCH = (
    ROOT
    / "qualification/server-launch/"
    "kiln-cuda-rtx4090-laptop-serving-bootstrap-v1.json"
)
ADMISSION_RECEIPT = (
    ROOT
    / "qualification/receipts/cuda/rtx4090-laptop/"
    "20260725t065312369587z-cuda-rtx4090-laptop-"
    "cuda-memory-lifecycle-v1-61a2e68c95-v1.json"
)
SERVER_CONFIG_SHA256 = (
    "sha256:455dee1f50c87e7d7eb2674fcf30823073500141e1f05a311b068633deb6f494"
)
SERVER_LAUNCH_SHA256 = (
    "sha256:62c237b2cc5209ff834d2aac655d196af128aa7990556cb45cbf287ad4f60889"
)
ADMISSION_RECEIPT_SHA256 = (
    "sha256:377583f15bc6365c4baf8a12f02c8f38e4f7b6a863ebd5958bdb321204956aeb"
)
ADMISSION_SOURCE_COMMIT = "9fadc2592f1814d7dd68b3c96ab008e8b886d665"
ADMISSION_WORKLOAD_SHA256 = (
    "sha256:ae9fc43d6284f1d5dd1f9cee2cde188883560536c67d10735fe504624bfdf95a"
)
SERVER_LOG_MAX_BYTES = 4 * 1024 * 1024
SERVER_FAULT_RE = re.compile(
    r"(?i)(CUDA_ERROR_OUT_OF_MEMORY|out of memory|device-side assert|"
    r"illegal memory access|generation_error|backend[^\r\n]{0,120}quarantin)"
)


def _load_benchmark_module() -> Any:
    path = ROOT / "scripts/bench-concurrent-batch.py"
    spec = importlib.util.spec_from_file_location(
        "kiln_bench_concurrent_batch", path
    )
    if spec is None or spec.loader is None:
        raise mixed.QualificationError(f"cannot load benchmark module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


EFFECTIVE_CONFIG: dict[str, Any] = {
    "admission_prerequisite": {
        "receipt_sha256": ADMISSION_RECEIPT_SHA256,
        "source_commit": ADMISSION_SOURCE_COMMIT,
        "workload_sha256": ADMISSION_WORKLOAD_SHA256,
    },
    "build": {
        "binary": "kiln",
        "cargo_environment_policy": "closed-qualification-test-v1",
        "cargo_execution_mode": "delegated-cgroup",
        "cargo_host_reserve_gib": BUILD_HOST_RESERVE_GIB,
        "cargo_jobs": 1,
        "cargo_max_memory_gib": BUILD_MAX_MEMORY_GIB,
        "cargo_memory_scope": "outer-wsl2-qualification-scope",
        "cargo_min_available_gib": BUILD_MIN_AVAILABLE_GIB,
        "cargo_private_network": True,
        "cargo_service_runtime_max_seconds": 1740,
        "cargo_wrapper": "scripts/cargo-bounded.sh",
        "features": "cuda",
        "locked": True,
        "no_default_features": True,
        "offline": True,
        "package": "kiln-server",
        "profile": "release",
        "timeout_seconds": int(BUILD_TIMEOUT_SECONDS),
    },
    "model": {
        "accelerator_weight_upload_mib_per_second": 256,
        "checkpoint_read_mib_per_second": 256,
        "model_id": MODEL_SOURCE_ID,
        "served_model_id": MODEL_ID,
    },
    "pressure": {
        "allocator_memory_source": "cuMemGetInfo_v2",
        "chunk_mib": PEER_CHUNK_MIB,
        "hold_seconds": PEER_HOLD_SECONDS,
        "maximum_allocation_mib": PEER_MAX_ALLOCATION_MIB,
        "memory_source": "nvidia-smi",
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "poll_milliseconds": PEER_POLL_MILLISECONDS,
        "target_free_mib": TARGET_FREE_MIB,
    },
    "recovery": {
        "health_tolerance_mib": HEALTH_RECOVERY_TOLERANCE_MIB,
        "peer_tolerance_mib": RECOVERY_TOLERANCE_MIB,
        "timeout_seconds": int(RECOVERY_TIMEOUT_SECONDS),
    },
    "request": {
        "max_tokens": REQUEST_MAX_TOKENS,
        "prompt_words": REQUEST_PROMPT_WORDS,
        "timeout_seconds": int(REQUEST_TIMEOUT_SECONDS),
    },
    "server": {
        "config_sha256": SERVER_CONFIG_SHA256,
        "floor_gib": 1.0,
        "gpu_capacity_source": "nvidia-smi",
        "gpu_memory_gib": None,
        "http_send_buffer_bytes": 212992,
        "inference_memory_fraction": 0.1,
        "kv_blocks": KV_BLOCKS,
        "kv_cache_bytes": KV_CACHE_BYTES,
        "launch_sha256": SERVER_LAUNCH_SHA256,
        "serving_profile": "stable",
    },
}

METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "admission_prerequisite_pass_count": ("count", "sum", False),
    "backend_quarantine_count": ("count", "sum", True),
    "baseline_free_bytes": ("bytes", "exact", False),
    "baseline_request_completion_tokens": ("tokens", "sum", False),
    "build_duration_ms": ("ms", "exact", True),
    "deterministic_recovery_match_count": ("count", "sum", False),
    "device_identity_pass_count": ("count", "sum", False),
    "model_identity_pass_count": ("count", "sum", False),
    "model_resident_bytes": ("bytes", "exact", False),
    "peer_allocated_bytes": ("bytes", "exact", False),
    "peer_allocation_count": ("count", "sum", False),
    "peer_exit_code": ("code", "exact", True),
    "peer_forced_shutdown_count": ("count", "sum", True),
    "peer_process_group_residue_count": ("count", "sum", True),
    "pressure_minimum_observed_free_bytes": ("bytes", "min", False),
    "pressure_ready_free_bytes": ("bytes", "exact", False),
    "pressure_request_completion_tokens": ("tokens", "sum", False),
    "pressure_request_success_count": ("count", "sum", False),
    "pressure_sample_count": ("count", "sum", False),
    "recovery_duration_ms": ("ms", "exact", True),
    "recovery_free_bytes": ("bytes", "exact", False),
    "recovery_request_completion_tokens": ("tokens", "sum", False),
    "request_failure_count": ("count", "sum", True),
    "server_exit_code": ("code", "exact", True),
    "server_fault_log_count": ("count", "sum", True),
    "server_forced_shutdown_count": ("count", "sum", True),
    "server_log_bytes": ("bytes", "exact", True),
    "server_process_group_residue_count": ("count", "sum", True),
    "snapshot_residue_count": ("count", "sum", True),
}


@dataclass(frozen=True)
class PeerShutdown:
    returncode: int
    forced: bool
    process_group_alive_end: bool
    output: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def require_artifact_hash(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise mixed.QualificationError(
            f"{label} is not a regular non-symlink file: {path}"
        )
    observed = sha256_file(path)
    if observed != expected:
        raise mixed.QualificationError(
            f"{label} hash drifted: {observed}, expected {expected}"
        )


def parse_closed_toml(source: str) -> dict[str, dict[str, Any]]:
    """Parse the scalar-only subset used by the immutable CUDA launch config."""
    parsed: dict[str, dict[str, Any]] = {}
    section: dict[str, Any] | None = None
    for line_number, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                or name in parsed
            ):
                raise mixed.QualificationError(
                    f"invalid or duplicate TOML section on line {line_number}"
                )
            section = {}
            parsed[name] = section
            continue
        if section is None or "=" not in line:
            raise mixed.QualificationError(
                f"invalid TOML scalar on line {line_number}"
            )
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", key)
            or key in section
        ):
            raise mixed.QualificationError(
                f"invalid or duplicate TOML key on line {line_number}"
            )
        try:
            value = json.loads(
                raw_value,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise mixed.QualificationError(
                f"unsupported TOML scalar on line {line_number}: {exc}"
            ) from exc
        if isinstance(value, (dict, list)) or value is None:
            raise mixed.QualificationError(
                f"non-scalar TOML value on line {line_number}"
            )
        section[key] = value
    return parsed


def validate_admission_prerequisite() -> None:
    require_artifact_hash(
        ADMISSION_RECEIPT,
        ADMISSION_RECEIPT_SHA256,
        "CUDA admission prerequisite receipt",
    )
    try:
        value = json.loads(ADMISSION_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise mixed.QualificationError(
            f"cannot parse CUDA admission prerequisite: {exc}"
        ) from exc
    qualification = value.get("qualification")
    source = value.get("source")
    workload = value.get("workload")
    if not isinstance(qualification, dict) or qualification.get("verdict") != "passed":
        raise mixed.QualificationError(
            "CUDA admission prerequisite is not a passed qualification"
        )
    if (
        not isinstance(source, dict)
        or source.get("git_commit") != ADMISSION_SOURCE_COMMIT
        or source.get("git_worktree_clean") is not True
    ):
        raise mixed.QualificationError(
            "CUDA admission prerequisite source identity drifted"
        )
    expected_workload = {
        "id": "cuda-memory-lifecycle-v1",
        "sha256": ADMISSION_WORKLOAD_SHA256,
    }
    if not isinstance(workload, dict) or any(
        workload.get(field) != expected
        for field, expected in expected_workload.items()
    ):
        raise mixed.QualificationError(
            "CUDA admission prerequisite workload identity drifted"
        )
    parameters = workload.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("variant_id") != VARIANT_ID:
        raise mixed.QualificationError(
            "CUDA admission prerequisite variant identity drifted"
        )
    results = value.get("results")
    if not isinstance(results, list):
        raise mixed.QualificationError(
            "CUDA admission prerequisite results are missing"
        )
    statuses = {
        row.get("id"): row.get("status")
        for row in results
        if isinstance(row, dict)
    }
    if statuses.get("server-admission-rejection") != "passed":
        raise mixed.QualificationError(
            "CUDA admission prerequisite did not pass server admission rejection"
        )


def validate_server_config_contract(config: dict[str, dict[str, Any]]) -> None:
    expected = {
        ("server", "serving_profile"): "stable",
        ("server", "host"): "127.0.0.1",
        ("server", "port"): 8420,
        ("server", "http_send_buffer_bytes"): EFFECTIVE_CONFIG["server"][
            "http_send_buffer_bytes"
        ],
        ("model", "model_id"): MODEL_SOURCE_ID,
        ("model", "served_model_id"): MODEL_ID,
        ("model", "checkpoint_read_mib_per_second"): 256,
        ("model", "accelerator_weight_upload_mib_per_second"): 256,
        ("memory", "gpu_memory_gb"): None,
        ("memory", "num_blocks"): EFFECTIVE_CONFIG["server"]["kv_blocks"],
        ("memory", "inference_memory_fraction"): EFFECTIVE_CONFIG["server"][
            "inference_memory_fraction"
        ],
        ("memory", "floor_gb"): EFFECTIVE_CONFIG["server"]["floor_gib"],
        ("memory", "reclaim_mode"): "off",
        ("memory", "kv_autoscale"): False,
        ("memory", "kv_cache_fp8"): False,
        ("memory", "cuda_graphs"): False,
    }
    for (section, field), expected_value in expected.items():
        observed = (config.get(section) or {}).get(field)
        if observed != expected_value:
            raise mixed.QualificationError(
                f"server config {section}.{field}={observed!r}, "
                f"expected {expected_value!r}"
            )


def validate_server_inputs(model_path: Path) -> None:
    require_artifact_hash(
        SERVER_CONFIG, SERVER_CONFIG_SHA256, "CUDA server configuration"
    )
    require_artifact_hash(
        SERVER_LAUNCH, SERVER_LAUNCH_SHA256, "CUDA server launch document"
    )
    try:
        config = parse_closed_toml(
            SERVER_CONFIG.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError) as exc:
        raise mixed.QualificationError(
            f"cannot parse CUDA server configuration: {exc}"
        ) from exc
    validate_server_config_contract(config)
    configured_model = Path(config["model"]["path"])
    if not configured_model.is_absolute():
        configured_model = ROOT / configured_model
    if configured_model.resolve(strict=True) != model_path:
        raise mixed.QualificationError(
            "server configuration model path does not match --model-path"
        )


def validate_device_identity() -> int:
    command = [
        "/usr/lib/wsl/lib/nvidia-smi",
        "-i",
        "0",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise mixed.QualificationError(
            f"cannot query CUDA device identity: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise mixed.QualificationError(
            "CUDA device identity query failed: " + completed.stderr[-1000:]
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise mixed.QualificationError(
            f"CUDA device identity returned {len(lines)} rows"
        )
    fields = [field.strip() for field in lines[0].rsplit(",", 1)]
    if len(fields) != 2 or fields[0] != "NVIDIA GeForce RTX 4090 Laptop GPU":
        raise mixed.QualificationError(
            f"CUDA logical device zero identity drifted: {lines[0]!r}"
        )
    try:
        total_mib = int(fields[1])
    except ValueError as exc:
        raise mixed.QualificationError(
            f"CUDA device memory is not an integer MiB value: {fields[1]!r}"
        ) from exc
    if not 16_000 <= total_mib <= 16_500:
        raise mixed.QualificationError(
            f"CUDA device memory {total_mib} MiB is outside the laptop envelope"
        )
    return total_mib * MIB


def validate_runner_environment(source: dict[str, str]) -> None:
    unexpected = sorted(
        key
        for key in source
        if key.startswith("KILN_") and key not in ALLOWED_RUNNER_KILN_ENV
    )
    if unexpected:
        raise mixed.QualificationError(
            "CUDA low-memory qualification rejects ambient Kiln controls: "
            + ", ".join(unexpected)
        )
    if source.get(NETWORK_ENV) != "util-linux-unshare-user-net-pid-landlock-v1":
        raise mixed.QualificationError(
            "CUDA low-memory qualification requires the WSL2 private-network "
            "and Landlock boundary"
        )
    expected_scope = {
        SCOPE_BOUNDARY_ENV: "systemd-user-scope-feedback-v1",
        SCOPE_MEMORY_MAX_ENV: str(10 * GIB),
        SCOPE_PIDS_MAX_ENV: "512",
        SCOPE_CPU_QUOTA_ENV: "0",
    }
    for field, expected in expected_scope.items():
        if source.get(field) != expected:
            raise mixed.QualificationError(
                f"CUDA low-memory qualification scope {field}="
                f"{source.get(field)!r}, expected {expected!r}"
            )
    if re.fullmatch(
        r"kiln-wsl-scope-[0-9a-f]{32}", source.get(SCOPE_UNIT_ENV, "")
    ) is None:
        raise mixed.QualificationError(
            "CUDA low-memory qualification scope unit is invalid"
        )
    if re.fullmatch(r"[1-9][0-9]*", source.get(SCOPE_HOST_UID_ENV, "")) is None:
        raise mixed.QualificationError(
            "CUDA low-memory qualification scope host UID is invalid"
        )


def child_environment(source: dict[str, str]) -> dict[str, str]:
    validate_runner_environment(source)
    return {
        key: value
        for key, value in source.items()
        if not key.startswith("KILN_") and key != "RUST_LOG"
    }


def build_environment(source: dict[str, str]) -> dict[str, str]:
    validate_runner_environment(source)
    environment = dict(source)
    environment.update(
        {
            "CARGO_NET_OFFLINE": "true",
            "CUDARC_CUDA_VERSION": "12080",
            "KILN_CARGO_ENVIRONMENT_POLICY": "closed-qualification-test-v1",
            "KILN_CARGO_EXECUTION_MODE": "delegated-cgroup",
            "KILN_CARGO_HOST_RESERVE_GIB": str(BUILD_HOST_RESERVE_GIB),
            "KILN_CARGO_JOBS": "1",
            "KILN_CARGO_MAX_MEMORY_GIB": str(BUILD_MAX_MEMORY_GIB),
            "KILN_CARGO_MIN_AVAILABLE_GIB": str(BUILD_MIN_AVAILABLE_GIB),
            "KILN_CARGO_PRIVATE_NETWORK": "1",
            "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS": "1740",
            "KILN_CUDA_ARCHS": "89",
            "KILN_QUALIFICATION": "1",
        }
    )
    return environment


def build_binary(absolute_deadline: float) -> tuple[Path, str, float]:
    started = time.monotonic()
    remaining = min(
        BUILD_TIMEOUT_SECONDS,
        max(0.001, absolute_deadline - time.monotonic()),
    )
    environment = build_environment(dict(os.environ))
    command = [
        str(ROOT / "scripts/cargo-bounded.sh"),
        "build",
        "--locked",
        "--offline",
        "--release",
        "-p",
        "kiln-server",
        "--bin",
        "kiln",
        "--no-default-features",
        "--features",
        "cuda",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise mixed.QualificationError(
            f"CUDA server build exceeded {remaining:.3f} seconds"
        ) from exc
    if completed.returncode != 0:
        raise mixed.QualificationError(
            f"CUDA server build returned {completed.returncode}"
        )
    binary = ROOT / "target/release/kiln"
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise mixed.QualificationError(
            "CUDA server build did not produce target/release/kiln"
        )
    return binary, sha256_file(binary), time.monotonic() - started


def launch_server(
    config: Any, environment: dict[str, str], run_id: str
) -> Any:
    log_path = bench.owned_server_log_path(config.log_directory, run_id)
    config.log_directory.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = log_path.open("xb", buffering=0)
    except OSError as exc:
        raise mixed.QualificationError(
            f"cannot create owned CUDA server log {log_path}: {exc}"
        ) from exc
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(config.command),
            cwd=config.working_directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        attach_deadline = time.monotonic() + 2.0
        while True:
            if process.poll() is not None:
                raise mixed.QualificationError(
                    "owned CUDA server exited before process identity binding: "
                    f"{process.returncode}"
                )
            try:
                identity = bench.AttachedProcessGroup.attach(process.pid)
                break
            except bench.BenchmarkError as exc:
                if time.monotonic() >= attach_deadline:
                    raise mixed.QualificationError(str(exc)) from exc
                time.sleep(0.005)
        return bench.OwnedServer(
            process=process,
            identity=identity,
            config=config,
            log_path=log_path,
            log_handle=log_handle,
        )
    except Exception:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10.0)
        log_handle.close()
        raise


def wait_server_ready(
    server: Any, port: int, absolute_deadline: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = min(
        time.monotonic() + server.config.startup_timeout_seconds,
        absolute_deadline,
    )
    last_error = "server has not accepted a readiness probe"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise mixed.QualificationError(
                "owned CUDA server exited during startup with status "
                f"{server.process.returncode}:\n"
                + bench.server_log_tail(server.log_path)
            )
        try:
            health = mixed.json_request(port, "GET", "/health")
            models = mixed.json_request(port, "GET", "/v1/models")
            if mixed.health_reports_ready_after_prewarm(health):
                return health, models
            last_error = "health did not report ready after prewarm"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(server.config.readiness_poll_interval_seconds)
    raise mixed.QualificationError(
        "owned CUDA server readiness timed out; last probe: "
        f"{last_error}\n{bench.server_log_tail(server.log_path)}"
    )


def live_free_bytes(health: dict[str, Any], label: str) -> int:
    gpu_memory = health.get("gpu_memory")
    live = gpu_memory.get("live") if isinstance(gpu_memory, dict) else None
    if not isinstance(live, dict):
        raise mixed.QualificationError(f"{label} GPU live-memory state is missing")
    expected = {
        "probe_failed": False,
        "sample_stale": False,
        "sampler_required": True,
        "sampler_running": True,
        "sampler_healthy": True,
        "source": "nvidia-smi",
        "unified": False,
    }
    for field, expected_value in expected.items():
        if live.get(field) != expected_value:
            raise mixed.QualificationError(
                f"{label} live-memory {field}={live.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    free_gb = live.get("free_gb")
    if (
        isinstance(free_gb, bool)
        or not isinstance(free_gb, (int, float))
        or not math.isfinite(float(free_gb))
        or free_gb < 0
    ):
        raise mixed.QualificationError(
            f"{label} live-memory free_gb is invalid: {free_gb!r}"
        )
    return int(round(float(free_gb) * 1e9))


def attest_model(
    health: dict[str, Any],
    models: dict[str, Any],
    debug: dict[str, Any],
    binary_sha256: str,
    expected_total_vram_bytes: int,
) -> int:
    failures: list[str] = []
    if health.get("status") != "ok":
        failures.append(f"health.status={health.get('status')!r}")
    if health.get("backend") != "model":
        failures.append(f"health.backend={health.get('backend')!r}")
    backend = health.get("backend_runtime")
    if not isinstance(backend, dict):
        failures.append("health.backend_runtime is missing")
    else:
        expected_backend = {
            "healthy": True,
            "quarantined": False,
            "reason": None,
            "restart_required": False,
        }
        for field, expected in expected_backend.items():
            if backend.get(field) != expected:
                failures.append(
                    f"backend_runtime.{field}={backend.get(field)!r}, "
                    f"expected {expected!r}"
                )
    rows = models.get("data") if isinstance(models, dict) else None
    model_ids = (
        [
            row.get("id")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        ]
        if isinstance(rows, list)
        else []
    )
    if model_ids != [MODEL_ID]:
        failures.append(f"/v1/models returned {model_ids!r}, expected [{MODEL_ID!r}]")
    base = health.get("base_weight_identity")
    if (
        not isinstance(base, dict)
        or base.get("manifest_type") != "kiln.base-weight-shards.v1"
        or base.get("aggregate_algorithm") != "kiln.base-model-content.v1"
        or base.get("shard_count") != 2
        or not isinstance(base.get("total_size_bytes"), int)
        or base.get("total_size_bytes", 0) < 8 * GIB
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(base.get("aggregate_sha256"))
        )
        is None
    ):
        failures.append("health base-weight identity is not the two-shard public model")
    gpu = health.get("gpu_memory")
    total_vram_bytes = (
        gpu.get("total_vram_bytes") if isinstance(gpu, dict) else None
    )
    if total_vram_bytes != expected_total_vram_bytes:
        failures.append(
            f"health total_vram_bytes={total_vram_bytes!r}, "
            f"expected physical device total {expected_total_vram_bytes}"
        )
    kv_cache_bytes = (
        gpu.get("kv_cache_bytes") if isinstance(gpu, dict) else None
    )
    if kv_cache_bytes != KV_CACHE_BYTES:
        failures.append(
            f"health kv_cache_bytes={kv_cache_bytes!r}, "
            f"expected {KV_CACHE_BYTES}"
        )
    model_resident_bytes = (
        gpu.get("post_load_used_bytes") if isinstance(gpu, dict) else None
    )
    if (
        isinstance(model_resident_bytes, bool)
        or not isinstance(model_resident_bytes, int)
        or model_resident_bytes < 4 * GIB
    ):
        failures.append(
            "health post-load CUDA residency is not at least four GiB"
        )
        model_resident_bytes = 0
    provenance = (
        ((debug.get("model") or {}).get("execution_provenance"))
        if isinstance(debug, dict)
        else None
    )
    if not isinstance(provenance, dict):
        failures.append("debug execution provenance is missing")
    else:
        backend_identity = provenance.get("backend")
        build_identity = provenance.get("build")
        if (
            not isinstance(backend_identity, dict)
            or str(backend_identity.get("name", "")).lower() != "cuda"
            or backend_identity.get("device") != "cuda:0"
        ):
            failures.append("debug execution provenance is not CUDA device zero")
        if (
            not isinstance(build_identity, dict)
            or build_identity.get("executable_sha256") != binary_sha256
            or build_identity.get("source_dirty") is not False
        ):
            failures.append(
                "debug execution provenance does not bind the clean source-built binary"
            )
        if provenance.get("provenance_type") != "kiln.execution-provenance.v1":
            failures.append("debug execution provenance type drifted")
    if failures:
        raise mixed.QualificationError(
            "CUDA model attestation failed: " + " | ".join(failures)
        )
    live_free_bytes(health, "model attestation")
    return model_resident_bytes


def load_json_no_extra(path: Path, expected_keys: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise mixed.QualificationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise mixed.QualificationError(f"{label} has an unexpected shape")
    return value


def validate_memory_snapshot(
    value: Any, label: str, minimum_free_bytes: int = 0
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"total_bytes", "free_bytes"}:
        raise mixed.QualificationError(f"{label} has an unexpected shape")
    for field in ("total_bytes", "free_bytes"):
        field_value = value[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value < 0
        ):
            raise mixed.QualificationError(f"{label}.{field} is invalid")
    if value["total_bytes"] <= 0 or value["free_bytes"] > value["total_bytes"]:
        raise mixed.QualificationError(f"{label} byte totals are invalid")
    if value["free_bytes"] < minimum_free_bytes:
        raise mixed.QualificationError(
            f"{label} crossed the {minimum_free_bytes}-byte floor"
        )
    return value


def load_peer_ready(path: Path, pid: int) -> dict[str, Any]:
    value = load_json_no_extra(
        path,
        {
            "schema_version",
            "pid",
            "device_ordinal",
            "memory_source",
            "allocator_memory_source",
            "allocated_bytes",
            "allocation_count",
            "configured_target_free_bytes",
            "effective_target_free_bytes",
            "minimum_free_bytes",
            "baseline",
            "ready",
            "allocator_baseline",
            "allocator_ready",
        },
        "CUDA pressure readiness",
    )
    expected = {
        "schema_version": 2,
        "pid": pid,
        "device_ordinal": 0,
        "memory_source": "nvidia-smi",
        "allocator_memory_source": "cuMemGetInfo_v2",
        "configured_target_free_bytes": TARGET_FREE_MIB * MIB,
        "minimum_free_bytes": MINIMUM_FREE_MIB * MIB,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise mixed.QualificationError(
                f"CUDA pressure readiness {field}={value.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    baseline = validate_memory_snapshot(
        value["baseline"], "CUDA pressure baseline", MINIMUM_FREE_MIB * MIB
    )
    ready = validate_memory_snapshot(
        value["ready"], "CUDA pressure ready", MINIMUM_FREE_MIB * MIB
    )
    allocator_baseline = validate_memory_snapshot(
        value["allocator_baseline"], "CUDA allocator baseline"
    )
    allocator_ready = validate_memory_snapshot(
        value["allocator_ready"], "CUDA allocator ready"
    )
    if baseline["total_bytes"] != ready["total_bytes"]:
        raise mixed.QualificationError("CUDA pressure total memory changed")
    if allocator_baseline["total_bytes"] != allocator_ready["total_bytes"]:
        raise mixed.QualificationError("CUDA allocator total memory changed")
    effective_target = value.get("effective_target_free_bytes")
    if (
        isinstance(effective_target, bool)
        or not isinstance(effective_target, int)
        or effective_target < MINIMUM_FREE_MIB * MIB
        or effective_target > TARGET_FREE_MIB * MIB
    ):
        raise mixed.QualificationError(
            "CUDA pressure effective target is invalid"
        )
    if ready["free_bytes"] > effective_target:
        raise mixed.QualificationError(
            "CUDA pressure peer declared readiness above its target"
        )
    if baseline["free_bytes"] - ready["free_bytes"] < 64 * MIB:
        raise mixed.QualificationError(
            "CUDA pressure global free-memory drop is below 64 MiB"
        )
    for field in ("allocated_bytes", "allocation_count"):
        observed = value.get(field)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
            raise mixed.QualificationError(
                f"CUDA pressure readiness {field} is not positive"
            )
    if value["allocated_bytes"] < 64 * MIB:
        raise mixed.QualificationError(
            "CUDA pressure peer allocated less than the 64 MiB evidence minimum"
        )
    if value["allocated_bytes"] > PEER_MAX_ALLOCATION_MIB * MIB:
        raise mixed.QualificationError(
            "CUDA pressure peer exceeded the declared allocation cap"
        )
    return value


def load_peer_release(path: Path, pid: int, ready: dict[str, Any]) -> dict[str, Any]:
    value = load_json_no_extra(
        path,
        {
            "schema_version",
            "pid",
            "device_ordinal",
            "memory_source",
            "allocator_memory_source",
            "ready_written",
            "completed",
            "allocated_bytes",
            "allocation_count",
            "minimum_observed_free_bytes",
            "sample_count",
            "elapsed_seconds",
            "release_failures",
            "final",
            "allocator_final",
        },
        "CUDA pressure release",
    )
    expected = {
        "schema_version": 2,
        "pid": pid,
        "device_ordinal": 0,
        "memory_source": "nvidia-smi",
        "allocator_memory_source": "cuMemGetInfo_v2",
        "ready_written": True,
        "completed": True,
        "allocated_bytes": ready["allocated_bytes"],
        "allocation_count": ready["allocation_count"],
        "release_failures": [],
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise mixed.QualificationError(
                f"CUDA pressure release {field}={value.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    minimum = value.get("minimum_observed_free_bytes")
    samples = value.get("sample_count")
    elapsed = value.get("elapsed_seconds")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < MINIMUM_FREE_MIB * MIB
    ):
        raise mixed.QualificationError(
            "CUDA pressure release reports a free-memory floor violation"
        )
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise mixed.QualificationError(
            "CUDA pressure release reports insufficient memory samples"
        )
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed <= 0
    ):
        raise mixed.QualificationError(
            "CUDA pressure release elapsed time is invalid"
        )
    final = validate_memory_snapshot(
        value["final"], "CUDA pressure final", MINIMUM_FREE_MIB * MIB
    )
    validate_memory_snapshot(
        value["allocator_final"], "CUDA allocator final"
    )
    recovery_floor = max(
        MINIMUM_FREE_MIB * MIB,
        ready["baseline"]["free_bytes"] - RECOVERY_TOLERANCE_MIB * MIB,
    )
    if final["free_bytes"] < recovery_floor:
        raise mixed.QualificationError(
            "CUDA memory did not recover after peer release: "
            f"{final['free_bytes']} < {recovery_floor} bytes"
        )
    return value


def start_pressure_peer(
    ready_path: Path, release_path: Path, environment: dict[str, str]
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(PEER_SCRIPT),
        "--ready-file",
        str(ready_path),
        "--release-file",
        str(release_path),
        "--device",
        "0",
        "--target-free-mib",
        str(TARGET_FREE_MIB),
        "--minimum-free-mib",
        str(MINIMUM_FREE_MIB),
        "--chunk-mib",
        str(PEER_CHUNK_MIB),
        "--max-allocation-mib",
        str(PEER_MAX_ALLOCATION_MIB),
        "--hold-seconds",
        str(PEER_HOLD_SECONDS),
        "--poll-milliseconds",
        str(PEER_POLL_MILLISECONDS),
        "--cuda-library",
        str(CUDA_LIBRARY),
        "--nvidia-smi",
        str(NVIDIA_SMI),
    ]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def wait_peer_ready(
    path: Path, peer: subprocess.Popen[str], absolute_deadline: float
) -> dict[str, Any]:
    deadline = min(
        time.monotonic() + PRESSURE_READY_TIMEOUT_SECONDS,
        absolute_deadline,
    )
    while time.monotonic() < deadline:
        if path.is_file():
            return load_peer_ready(path, peer.pid)
        if peer.poll() is not None:
            output, _ = peer.communicate(timeout=5.0)
            raise mixed.QualificationError(
                "CUDA pressure peer exited before readiness "
                f"({peer.returncode}): {output[-4000:]}"
            )
        time.sleep(0.1)
    raise mixed.QualificationError(
        "CUDA pressure peer did not reach its target before the deadline"
    )


def terminate_peer(process: subprocess.Popen[str]) -> PeerShutdown:
    forced = False
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        output, _ = process.communicate(timeout=PEER_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        forced = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate(timeout=10.0)
    return PeerShutdown(
        returncode=process.returncode,
        forced=forced,
        process_group_alive_end=bench.process_group_alive(process.pid),
        output=output[-16000:],
    )


def wait_health_pressure(
    port: int, absolute_deadline: float
) -> tuple[dict[str, Any], int]:
    deadline = min(time.monotonic() + 10.0, absolute_deadline)
    last_free = 0
    while time.monotonic() < deadline:
        health = mixed.json_request(port, "GET", "/health")
        last_free = live_free_bytes(health, "pressure")
        if (
            MINIMUM_FREE_MIB * MIB - 2 * MIB
            <= last_free
            <= (TARGET_FREE_MIB + 256) * MIB
        ):
            return health, last_free
        time.sleep(0.25)
    raise mixed.QualificationError(
        "server live-memory sampler did not corroborate bounded CUDA pressure; "
        f"last free={last_free} bytes"
    )


def wait_health_recovery(
    port: int,
    baseline_free_bytes: int,
    absolute_deadline: float,
) -> tuple[dict[str, Any], int, float]:
    started = time.monotonic()
    deadline = min(started + RECOVERY_TIMEOUT_SECONDS, absolute_deadline)
    threshold = max(
        MINIMUM_FREE_MIB * MIB,
        baseline_free_bytes - HEALTH_RECOVERY_TOLERANCE_MIB * MIB,
    )
    last_free = 0
    while time.monotonic() < deadline:
        health = mixed.json_request(port, "GET", "/health")
        last_free = live_free_bytes(health, "recovery")
        if last_free >= threshold:
            return health, last_free, (time.monotonic() - started) * 1000.0
        time.sleep(0.25)
    raise mixed.QualificationError(
        f"server live memory did not recover to {threshold} bytes; "
        f"last free={last_free}"
    )


def run_request(
    port: int,
    *,
    name: str,
    marker: str,
    seed: int,
    absolute_deadline: float,
) -> mixed.StreamResult:
    result = mixed.run_stream(
        port,
        name=name,
        marker=marker,
        prompt_words=REQUEST_PROMPT_WORDS,
        max_tokens=REQUEST_MAX_TOKENS,
        seed=seed,
        absolute_deadline=absolute_deadline,
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )
    if (
        not result.success
        or result.finish_reason != "length"
        or result.completion_tokens != REQUEST_MAX_TOKENS
        or len(result.token_ids) != REQUEST_MAX_TOKENS
    ):
        raise mixed.QualificationError(
            f"{name} failed: success={result.success}, error={result.error!r}, "
            f"finish={result.finish_reason!r}, "
            f"completion_tokens={result.completion_tokens}, "
            f"token_ids={len(result.token_ids)}"
        )
    return result


def metrics_from_values(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        raise mixed.QualificationError(
            "CUDA low-memory metric set mismatch: "
            f"missing={sorted(set(METRIC_DEFINITIONS) - set(values))}, "
            f"extra={sorted(set(values) - set(METRIC_DEFINITIONS))}"
        )
    metrics: list[dict[str, Any]] = []
    for name in sorted(values):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise mixed.QualificationError(
                f"metric {name} is not finite numeric evidence"
            )
        unit, aggregation, lower_is_better = METRIC_DEFINITIONS[name]
        metrics.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "aggregation": aggregation,
                "lower_is_better": lower_is_better,
            }
        )
    return metrics


def zero_metrics() -> list[dict[str, Any]]:
    values = {name: 0 for name in METRIC_DEFINITIONS}
    values["request_failure_count"] = 1
    values["server_exit_code"] = -1
    values["peer_exit_code"] = -1
    return metrics_from_values(values)


def emit_server_log(log_path: Path, log_artifact: dict[str, Any]) -> str:
    try:
        data = log_path.read_bytes()
    except OSError as exc:
        raise mixed.QualificationError(
            f"cannot read owned CUDA server log: {exc}"
        ) from exc
    if len(data) != log_artifact["bytes"]:
        raise mixed.QualificationError(
            "owned CUDA server log size changed after hashing"
        )
    if len(data) > SERVER_LOG_MAX_BYTES:
        raise mixed.QualificationError(
            f"owned CUDA server log exceeds {SERVER_LOG_MAX_BYTES} bytes"
        )
    text = data.decode("utf-8", errors="replace")
    print(
        json.dumps(
            {
                "event": "cuda_server_log_artifact",
                "bytes": log_artifact["bytes"],
                "sha256": log_artifact["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
    print("----- begin owned CUDA server log -----", file=sys.stderr)
    print(text, end="" if text.endswith("\n") or not text else "\n", file=sys.stderr)
    print("----- end owned CUDA server log -----", file=sys.stderr)
    return text


def execute(
    model_path: Path, seed: int
) -> tuple[list[dict[str, Any]], str | None]:
    absolute_deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS
    validate_admission_prerequisite()
    validate_server_inputs(model_path)
    device_total_bytes = validate_device_identity()
    original_environment = dict(os.environ)
    server_environment = child_environment(original_environment)
    peer_environment = child_environment(original_environment)
    binary, binary_sha256, build_seconds = build_binary(absolute_deadline)
    config = bench.load_server_launch_config(SERVER_LAUNCH)
    expected_command = (
        str(binary.resolve()),
        "serve",
        "--config",
        "qualification/server-config/"
        "kiln-cuda-rtx4090-laptop-serving-bootstrap-v1.toml",
    )
    if config.command != expected_command or config.record["id"] != (
        "kiln-cuda-rtx4090-laptop-serving-bootstrap-v1"
    ):
        raise mixed.QualificationError(
            "CUDA server launch command or identity drifted"
        )
    port = bench.require_owned_base_url_unbound("http://127.0.0.1:8420")
    snapshot_dir = ROOT / ".qualification/cuda-rtx4090-laptop/snapshots"
    initial_snapshot_residue = mixed.snapshot_payload_residue(snapshot_dir)
    if initial_snapshot_residue:
        raise mixed.QualificationError(
            "CUDA snapshot directory was not empty before launch: "
            + ", ".join(initial_snapshot_residue)
        )
    run_dir = mixed.create_serving_run_dir(
        "cuda-low-memory",
        parent=ROOT / ".qualification/serving-cuda-low-memory",
    )
    ready_path = run_dir / "pressure-ready.json"
    release_path = run_dir / "pressure-release.json"
    server = launch_server(config, server_environment, run_dir.name)
    peer: subprocess.Popen[str] | None = None
    peer_shutdown: PeerShutdown | None = None
    shutdown: dict[str, Any] | None = None
    log_artifact: dict[str, Any] | None = None
    server_log_text = ""
    result: tuple[list[dict[str, Any]], str | None] | None = None
    snapshot_residue: list[str] = []
    try:
        health_start, models = wait_server_ready(server, port, absolute_deadline)
        bench.verify_owned_listener(server, "http://127.0.0.1:8420")
        debug_start = mixed.json_request(port, "GET", "/v1/debug/model-state")
        model_resident_bytes = attest_model(
            health_start,
            models,
            debug_start,
            binary_sha256,
            device_total_bytes,
        )
        warmup = run_request(
            port,
            name="low-memory-warmup",
            marker=mixed.workload_marker(seed, "cuda-low-memory-warmup"),
            seed=seed,
            absolute_deadline=absolute_deadline,
        )
        replay_marker = mixed.workload_marker(seed, "cuda-low-memory-replay")
        baseline = run_request(
            port,
            name="low-memory-baseline",
            marker=replay_marker,
            seed=seed,
            absolute_deadline=absolute_deadline,
        )
        health_baseline = mixed.json_request(port, "GET", "/health")
        baseline_free_bytes = live_free_bytes(health_baseline, "baseline")

        peer = start_pressure_peer(
            ready_path, release_path, peer_environment
        )
        ready = wait_peer_ready(ready_path, peer, absolute_deadline)
        pressure = run_request(
            port,
            name="low-memory-pressure",
            marker=mixed.workload_marker(seed, "cuda-low-memory-pressure"),
            seed=seed + 1,
            absolute_deadline=absolute_deadline,
        )
        if peer.poll() is not None:
            output, _ = peer.communicate(timeout=5.0)
            raise mixed.QualificationError(
                "CUDA pressure peer exited during the pressure request "
                f"({peer.returncode}): {output[-4000:]}"
            )
        pressure_health, _pressure_health_free = wait_health_pressure(
            port, absolute_deadline
        )
        attest_model(
            pressure_health,
            mixed.json_request(port, "GET", "/v1/models"),
            mixed.json_request(port, "GET", "/v1/debug/model-state"),
            binary_sha256,
            device_total_bytes,
        )
        peer_shutdown = terminate_peer(peer)
        peer = None
        print(peer_shutdown.output, end="" if peer_shutdown.output.endswith("\n") else "\n")
        if (
            peer_shutdown.forced
            or peer_shutdown.returncode != 0
            or peer_shutdown.process_group_alive_end
        ):
            raise mixed.QualificationError(
                "CUDA pressure peer did not exit cleanly: "
                f"returncode={peer_shutdown.returncode}, "
                f"forced={peer_shutdown.forced}, "
                f"group_alive={peer_shutdown.process_group_alive_end}"
            )
        release = load_peer_release(release_path, ready["pid"], ready)
        health_recovery, recovery_health_free, recovery_ms = (
            wait_health_recovery(
                port, baseline_free_bytes, absolute_deadline
            )
        )
        recovery = run_request(
            port,
            name="low-memory-recovery",
            marker=replay_marker,
            seed=seed,
            absolute_deadline=absolute_deadline,
        )
        deterministic_match = int(
            recovery.token_ids == baseline.token_ids
            and recovery.finish_reason == baseline.finish_reason
            and recovery.completion_tokens == baseline.completion_tokens
        )
        if deterministic_match != 1:
            raise mixed.QualificationError(
                "post-pressure request did not exactly reproduce baseline token IDs"
            )
        final_health = mixed.json_request(port, "GET", "/health")
        attest_model(
            final_health,
            mixed.json_request(port, "GET", "/v1/models"),
            mixed.json_request(port, "GET", "/v1/debug/model-state"),
            binary_sha256,
            device_total_bytes,
        )
        requests = final_health.get("requests")
        request_errors = (
            requests.get("error") if isinstance(requests, dict) else None
        )
        if request_errors != 0:
            raise mixed.QualificationError(
                f"server recorded {request_errors!r} request errors"
            )
        backend = final_health.get("backend_runtime")
        quarantine_count = int(
            not isinstance(backend, dict)
            or backend.get("quarantined") is not False
        )
        values: dict[str, float | int] = {
            "admission_prerequisite_pass_count": 1,
            "backend_quarantine_count": quarantine_count,
            "baseline_free_bytes": baseline_free_bytes,
            "baseline_request_completion_tokens": baseline.completion_tokens,
            "build_duration_ms": build_seconds * 1000.0,
            "deterministic_recovery_match_count": deterministic_match,
            "device_identity_pass_count": 1,
            "model_identity_pass_count": 1,
            "model_resident_bytes": model_resident_bytes,
            "peer_allocated_bytes": ready["allocated_bytes"],
            "peer_allocation_count": ready["allocation_count"],
            "peer_exit_code": peer_shutdown.returncode,
            "peer_forced_shutdown_count": int(peer_shutdown.forced),
            "peer_process_group_residue_count": int(
                peer_shutdown.process_group_alive_end
            ),
            "pressure_minimum_observed_free_bytes": release[
                "minimum_observed_free_bytes"
            ],
            "pressure_ready_free_bytes": ready["ready"]["free_bytes"],
            "pressure_request_completion_tokens": pressure.completion_tokens,
            "pressure_request_success_count": int(pressure.success),
            "pressure_sample_count": release["sample_count"],
            "recovery_duration_ms": recovery_ms,
            "recovery_free_bytes": release["final"]["free_bytes"],
            "recovery_request_completion_tokens": recovery.completion_tokens,
            "request_failure_count": sum(
                not item.success
                for item in (warmup, baseline, pressure, recovery)
            ),
            "server_exit_code": 0,
            "server_fault_log_count": 0,
            "server_forced_shutdown_count": 0,
            "server_log_bytes": 0,
            "server_process_group_residue_count": 0,
            "snapshot_residue_count": 0,
        }
        if recovery_health_free < (
            baseline_free_bytes - HEALTH_RECOVERY_TOLERANCE_MIB * MIB
        ):
            raise mixed.QualificationError(
                "health recovery fell outside the declared tolerance"
            )
        result = metrics_from_values(values), None
    finally:
        if peer is not None:
            peer_shutdown = terminate_peer(peer)
            print(
                peer_shutdown.output,
                end="" if peer_shutdown.output.endswith("\n") else "\n",
            )
        try:
            shutdown = bench.shutdown_owned_server(server)
        finally:
            try:
                log_artifact = bench.close_owned_server_log(server)
                server_log_text = emit_server_log(
                    server.log_path, log_artifact
                )
            finally:
                server.log_path.unlink(missing_ok=True)
        snapshot_residue = mixed.snapshot_payload_residue(snapshot_dir)
        shutil.rmtree(run_dir, ignore_errors=True)

    if result is None or shutdown is None or log_artifact is None:
        raise AssertionError(
            "CUDA low-memory qualification completed without lifecycle evidence"
        )
    metrics, details = result
    values = {row["name"]: row["value"] for row in metrics}
    faults = len(SERVER_FAULT_RE.findall(server_log_text))
    values.update(
        {
            "server_exit_code": shutdown["returncode"],
            "server_fault_log_count": faults,
            "server_forced_shutdown_count": int(shutdown["forced"]),
            "server_log_bytes": log_artifact["bytes"],
            "server_process_group_residue_count": int(
                shutdown["process_group_alive_end"]
            ),
            "snapshot_residue_count": len(snapshot_residue),
        }
    )
    failures: list[str] = []
    if shutdown["forced"]:
        failures.append("CUDA server required forced termination")
    if shutdown["returncode"] not in shutdown["acceptable_exit_codes"]:
        failures.append(
            "CUDA server shutdown returned "
            f"{shutdown['returncode']}, expected "
            f"{shutdown['acceptable_exit_codes']}"
        )
    if shutdown["process_group_alive_end"]:
        failures.append("CUDA server process group remained alive after shutdown")
    if faults:
        failures.append(
            f"CUDA server log contained {faults} OOM/device/quarantine fault matches"
        )
    if snapshot_residue:
        failures.append(
            "CUDA server left model snapshot payload: "
            + ", ".join(snapshot_residue)
        )
    return (
        metrics_from_values(values),
        " | ".join(filter(None, [details, *failures])) or None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    result_path_value = os.environ.get(RESULT_ENV)
    variant = os.environ.get(VARIANT_ENV, "")
    if not result_path_value:
        print(f"{RESULT_ENV} is required", file=sys.stderr)
        return 2
    status = "failed"
    details: str | None = None
    metrics = zero_metrics()
    try:
        if variant != VARIANT_ID:
            raise mixed.QualificationError(
                f"{VARIANT_ENV} must be {VARIANT_ID!r}, got {variant!r}"
            )
        model_path = args.model_path.resolve(strict=True)
        if not model_path.is_dir():
            raise mixed.QualificationError("--model-path must be a directory")
        if args.seed < 0:
            raise mixed.QualificationError("--seed must be nonnegative")
        metrics, details = execute(model_path, args.seed)
        status = "passed" if details is None else "failed"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
        mixed.trace("qualification_error", details=details)
    result = {
        "schema_version": 1,
        "case_id": CASE_ID,
        "status": status,
        "duration_seconds": time.monotonic() - started,
        "effective_config": EFFECTIVE_CONFIG,
        "metrics": metrics,
        "tolerances": [],
        "details": mixed.bounded_details(details),
    }
    try:
        mixed.write_result(Path(result_path_value), result)
    except Exception as exc:
        print(f"cannot write qualification result: {exc}", file=sys.stderr)
        return 2
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
