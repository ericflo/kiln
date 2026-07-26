#!/usr/bin/env python3
"""Run a source-bound paired CUDA serving campaign on the 4090 Laptop GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import serve_mixed_load as mixed
import wsl_pacing_evidence as pacing
from strict_json import loads as strict_json_loads


ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "cuda-serving-performance-c1"
VARIANT_ID = "cuda-rtx4090-laptop-c1"
RESULT_ENV = mixed.RESULT_ENV
VARIANT_ENV = mixed.VARIANT_ENV
PROFILES = (
    "greedy-short",
    "api-default-sampled",
    "long-prefill",
    "prefix-hit",
    "mixed",
)
SIZES = "1"
SIZE_VALUES = (1,)
REPEATS = 1
MAX_TOKENS = 64
WARMUP_REQUESTS = 1
MODEL_FINGERPRINT_READ_MIB_PER_SECOND = 64
MEMORY_LIMIT_BYTES = 16_500_000_000
MEMORY_SAMPLE_MS = 100
REQUEST_TIMEOUT_SECONDS = 900
CASE_TIMEOUT_SECONDS = 43_200.0
BUILD_TIMEOUT_SECONDS = 1_800.0
BUILD_THERMAL_PAUSE_ALLOWANCE_SECONDS = 14_400.0
BUILD_WALL_TIMEOUT_SECONDS = (
    BUILD_TIMEOUT_SECONDS + BUILD_THERMAL_PAUSE_ALLOWANCE_SECONDS
)
BUILD_POLL_SECONDS = 1.0
GPU_UUID = "GPU-fff83066-80fa-ac5f-edbe-4ebd3ac9bbfd"
MODEL_ID = "Qwen3.5-4B"
PROMPT_SET_ID = "cuda-rtx4090-laptop-performance-v1"
THERMAL_POLICY = (
    ROOT
    / "qualification/host-policies/"
    "rtx4090-laptop-wsl2-cgroup-pacing-v2.json"
)
KILN_LAUNCH = (
    ROOT
    / "qualification/server-launch/"
    "kiln-cuda-rtx4090-laptop-serving-performance-v1.json"
)
VLLM_LAUNCH = (
    ROOT
    / "qualification/server-launch/"
    "vllm-cuda-rtx4090-laptop-serving-performance-v1.json"
)
KILN_CONFIG = (
    ROOT
    / "qualification/server-config/"
    "kiln-cuda-rtx4090-laptop-serving-performance-v1.toml"
)
VLLM_RUNTIME_MANIFEST = (
    ROOT / "qualification/runtime/vllm/cuda/rtx4090-laptop/performance-v1.json"
)
CAMPAIGN_RUNNER = ROOT / "scripts/run-serving-benchmark-campaign.py"
PROMPT_CONTEXT_CHECKER = (
    ROOT / "scripts/qualification/check_serving_prompt_context.py"
)
VLLM_PYTHON = ROOT / ".qualification/vllm-cuda-venv/bin/python-kiln"
KILN_CONTEXT_CEILING_TOKENS = 62 * 64
PROMPT_CONTEXT_MAX_PROMPT_TOKENS = 3_883
PROMPT_CONTEXT_MIN_HEADROOM_TOKENS = 21
TOKENIZER_SHA256 = (
    "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
CHAT_TEMPLATE_SHA256 = (
    "sha256:a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
)
KILN_LAUNCH_SHA256 = (
    "sha256:2d351c605bb0da71dde85521b6a0a4546fb9990a7f0c45f84f926dcb157f344d"
)
VLLM_LAUNCH_SHA256 = (
    "sha256:37b3ce1244241eb7e3cb5c8b30cbb853636a73bc16f270662dc947d32676a35e"
)
KILN_CONFIG_SHA256 = (
    "sha256:21c068c0ec39b532c364e6bf9495d235ed5006e78140bb47f8548ca3269c8943"
)
VLLM_RUNTIME_MANIFEST_SHA256 = (
    "sha256:50d46bd54df16f1ea9095dace7656708b7347db3591ad8ebc74d1238d284d125"
)
THERMAL_POLICY_CONTENT_SHA256 = (
    "sha256:d389a7f632baab0448bd41efc205349dee4ff3944152b48cf17e52866322e3e9"
)


def _load_benchmark_module() -> Any:
    path = ROOT / "scripts/bench-concurrent-batch.py"
    spec = importlib.util.spec_from_file_location(
        "kiln_cuda_performance_benchmark", path
    )
    if spec is None or spec.loader is None:
        raise mixed.QualificationError(f"cannot load benchmark module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()

EFFECTIVE_CONFIG: dict[str, Any] = {
    "build": {
        "binary": "kiln",
        "cargo_cpu_quota_percent": 50,
        "cargo_environment_policy": "closed-qualification-test-v1",
        "cargo_execution_mode": "delegated-cgroup",
        "cargo_host_reserve_gib": 3,
        "cargo_jobs": 1,
        "cargo_max_memory_gib": 10,
        "cargo_memory_scope": "outer-wsl2-qualification-scope",
        "cargo_min_available_gib": 13,
        "cargo_private_network": True,
        "cargo_runtime_authority": (
            "contained-parent-pause-aware-process-group-v1"
        ),
        "cargo_termination": "new-session-term-then-kill-process-group-v1",
        "cargo_wrapper": "scripts/cargo-bounded.sh",
        "duration_metric_accounting": (
            "wall_seconds_minus_verified_wsl2_thermal_pause_overlap"
        ),
        "features": "cuda",
        "locked": True,
        "no_default_features": True,
        "offline": True,
        "package": "kiln-server",
        "profile": "release",
        "thermal_pause_allowance_seconds": 14400,
        "timeout_accounting": (
            "wall_seconds_minus_verified_wsl2_thermal_pause_overlap"
        ),
        "timeout_seconds": 1800,
        "wall_timeout_seconds": 16200,
    },
    "campaign": {
        "continue_after_failure": True,
        "max_tokens": MAX_TOKENS,
        "memory_device_uuid": GPU_UUID,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "memory_sample_ms": MEMORY_SAMPLE_MS,
        "model_fingerprint_read_mib_per_second": (
            MODEL_FINGERPRINT_READ_MIB_PER_SECOND
        ),
        "output_evidence": "hashes",
        "prompt_context_ceiling_tokens": KILN_CONTEXT_CEILING_TOKENS,
        "prompt_context_max_prompt_tokens": PROMPT_CONTEXT_MAX_PROMPT_TOKENS,
        "prompt_context_min_headroom_tokens": PROMPT_CONTEXT_MIN_HEADROOM_TOKENS,
        "prompt_template_version": bench.PROMPT_TEMPLATE_VERSION,
        "prompt_tokenizer_sha256": TOKENIZER_SHA256,
        "profiles": {
            f"profile_{index}": profile
            for index, profile in enumerate(PROFILES)
        },
        "prompt_set_id": PROMPT_SET_ID,
        "repeats": REPEATS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "sizes": {
            f"concurrency_{index}": concurrency
            for index, concurrency in enumerate(SIZE_VALUES)
        },
        "warmup_requests": WARMUP_REQUESTS,
    },
    "kiln": {
        "base_url": "http://127.0.0.1:8420",
        "config_sha256": KILN_CONFIG_SHA256,
        "launch_sha256": KILN_LAUNCH_SHA256,
    },
    "model": {
        "model_id": "Qwen/Qwen3.5-4B",
        "served_model_id": MODEL_ID,
        "shared_closed_model_view": (
            ".qualification/cuda-rtx4090-laptop/performance-model-v1"
        ),
    },
    "thermal": {
        "outer_hard_limit_gpu_millicelsius": 85_000,
        "outer_hard_limit_host_millicelsius": 95_000,
        "pacing_start_gpu_millicelsius": 75_000,
        "pacing_start_host_millicelsius": 80_000,
        "policy_content_sha256": THERMAL_POLICY_CONTENT_SHA256,
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8421",
        "launch_sha256": VLLM_LAUNCH_SHA256,
        "runtime_manifest_sha256": VLLM_RUNTIME_MANIFEST_SHA256,
    },
}

METRIC_DEFINITIONS: dict[str, tuple[str, str, bool]] = {
    "build_duration_ms": ("ms", "exact", True),
    "exact_output_mismatch_count": ("count", "sum", True),
    "kiln_completion_token_count": ("tokens", "sum", False),
    "kiln_output_token_throughput_per_second": (
        "tokens_per_second",
        "exact",
        False,
    ),
    "kiln_peak_device_memory_bytes": ("bytes", "max", True),
    "kiln_profile_pass_count": ("count", "sum", False),
    "kiln_request_failure_count": ("count", "sum", True),
    "kiln_request_success_count": ("count", "sum", False),
    "paired_profile_pass_count": ("count", "sum", False),
    "vllm_completion_token_count": ("tokens", "sum", False),
    "vllm_output_token_throughput_per_second": (
        "tokens_per_second",
        "exact",
        False,
    ),
    "vllm_peak_device_memory_bytes": ("bytes", "max", True),
    "vllm_profile_pass_count": ("count", "sum", False),
    "vllm_request_failure_count": ("count", "sum", True),
    "vllm_request_success_count": ("count", "sum", False),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise mixed.QualificationError("cannot bind the performance run to HEAD")
    return commit


def prompt_context_check_command(model_path: Path) -> list[str]:
    return [
        str(VLLM_PYTHON),
        str(PROMPT_CONTEXT_CHECKER),
        "--model-path",
        str(model_path),
        "--prompt-set-id",
        PROMPT_SET_ID,
        "--profiles",
        ",".join(PROFILES),
        "--sizes",
        SIZES,
        "--repeats",
        str(REPEATS),
        "--warmup-requests",
        str(WARMUP_REQUESTS),
        "--max-tokens",
        str(MAX_TOKENS),
        "--context-ceiling",
        str(KILN_CONTEXT_CEILING_TOKENS),
        "--expected-max-prompt-tokens",
        str(PROMPT_CONTEXT_MAX_PROMPT_TOKENS),
    ]


def run_prompt_context_check(model_path: Path, deadline: float) -> dict[str, Any]:
    if (
        VLLM_PYTHON.is_symlink()
        or not VLLM_PYTHON.is_file()
        or not os.access(VLLM_PYTHON, os.X_OK)
    ):
        raise mixed.QualificationError(
            f"prompt context checker interpreter is unavailable: {VLLM_PYTHON}"
        )
    remaining = max(0.001, deadline - time.monotonic())
    try:
        completed = subprocess.run(
            prompt_context_check_command(model_path),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(120.0, remaining),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise mixed.QualificationError(
            "serving prompt context check exceeded its bounded startup deadline"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise mixed.QualificationError(
            f"serving prompt context check returned {completed.returncode}: {detail}"
        )
    try:
        record = strict_json_loads(completed.stdout)
    except Exception as exc:
        raise mixed.QualificationError(
            f"serving prompt context check emitted invalid JSON: {exc}"
        ) from exc
    expected_keys = {
        "chat_template_sha256",
        "checked_prompt_count",
        "context_ceiling_tokens",
        "driver_version",
        "max_prompt_tokens",
        "max_tokens",
        "max_total_tokens",
        "minimum_headroom_tokens",
        "profiles",
        "prompt_template_version",
        "schema",
        "sizes",
        "tokenizer_sha256",
        "transformers_version",
        "verdict",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise mixed.QualificationError("serving prompt context check schema drifted")
    expected = {
        "schema": "kiln.serving-prompt-context-check.v1",
        "verdict": "passed",
        "driver_version": bench.DRIVER_VERSION,
        "prompt_template_version": bench.PROMPT_TEMPLATE_VERSION,
        "context_ceiling_tokens": KILN_CONTEXT_CEILING_TOKENS,
        "max_tokens": MAX_TOKENS,
        "max_prompt_tokens": PROMPT_CONTEXT_MAX_PROMPT_TOKENS,
        "max_total_tokens": KILN_CONTEXT_CEILING_TOKENS
        - PROMPT_CONTEXT_MIN_HEADROOM_TOKENS,
        "minimum_headroom_tokens": PROMPT_CONTEXT_MIN_HEADROOM_TOKENS,
        "profiles": list(PROFILES),
        "sizes": list(SIZE_VALUES),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise mixed.QualificationError(
                f"serving prompt context check {name} drifted: "
                f"{record.get(name)!r}, expected {value!r}"
            )
    if (
        isinstance(record["checked_prompt_count"], bool)
        or not isinstance(record["checked_prompt_count"], int)
        or record["checked_prompt_count"] <= 0
        or not isinstance(record["transformers_version"], str)
        or not record["transformers_version"]
    ):
        raise mixed.QualificationError(
            "serving prompt context check returned malformed positive evidence"
        )
    return record


def build_environment(source: dict[str, str]) -> dict[str, str]:
    environment = dict(source)
    environment.pop("KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS", None)
    environment.update(
        {
            "CARGO_NET_OFFLINE": "true",
            "CUDARC_CUDA_VERSION": "12080",
            "KILN_CARGO_CPU_QUOTA_PERCENT": "50",
            "KILN_CARGO_ENVIRONMENT_POLICY": "closed-qualification-test-v1",
            "KILN_CARGO_EXECUTION_MODE": "delegated-cgroup",
            "KILN_CARGO_HOST_RESERVE_GIB": "3",
            "KILN_CARGO_JOBS": "1",
            "KILN_CARGO_MAX_MEMORY_GIB": "10",
            "KILN_CARGO_MIN_AVAILABLE_GIB": "13",
            "KILN_CARGO_PRIVATE_NETWORK": "1",
            "KILN_CUDA_ARCHS": "89",
            "KILN_QUALIFICATION": "1",
        }
    )
    return environment


def _terminate_build_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired as exc:
        raise mixed.QualificationError(
            "CUDA performance build process group did not terminate"
        ) from exc


def _build_elapsed_seconds(
    started: float,
    finished: float,
    source: dict[str, str],
) -> tuple[float, float, float]:
    snapshot = pacing.read_pacing_snapshot(
        source,
        expected_policy_sha256=THERMAL_POLICY_CONTENT_SHA256,
    )
    wall_seconds = finished - started
    if (
        not math.isfinite(wall_seconds)
        or wall_seconds < 0
        or wall_seconds > BUILD_WALL_TIMEOUT_SECONDS + BUILD_POLL_SECONDS
    ):
        raise mixed.QualificationError(
            "CUDA performance build wall-time accounting is invalid"
        )
    pause_seconds = snapshot.overlap_seconds(started, finished)
    active_seconds = wall_seconds - pause_seconds
    if (
        pause_seconds < 0
        or pause_seconds > wall_seconds
        or active_seconds < -1e-6
    ):
        raise mixed.QualificationError(
            "CUDA performance build thermal-pause accounting is invalid"
        )
    return wall_seconds, pause_seconds, max(0.0, active_seconds)


def build_binary(deadline: float) -> tuple[Path, str, float, float, float]:
    started = time.monotonic()
    wall_deadline = min(
        deadline,
        started + BUILD_WALL_TIMEOUT_SECONDS,
    )
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
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=build_environment(dict(os.environ)),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise mixed.QualificationError(
            f"cannot start CUDA performance build: {exc}"
        ) from exc
    try:
        while process.poll() is None:
            now = time.monotonic()
            wall_seconds, pause_seconds, active_seconds = (
                _build_elapsed_seconds(started, now, dict(os.environ))
            )
            if pause_seconds > BUILD_THERMAL_PAUSE_ALLOWANCE_SECONDS:
                raise mixed.QualificationError(
                    "CUDA performance build exceeded "
                    f"{BUILD_THERMAL_PAUSE_ALLOWANCE_SECONDS:.3f} seconds "
                    "of verified thermal pacing"
                )
            if active_seconds > BUILD_TIMEOUT_SECONDS:
                raise mixed.QualificationError(
                    "CUDA performance build exceeded "
                    f"{BUILD_TIMEOUT_SECONDS:.3f} active seconds"
                )
            if now >= wall_deadline:
                raise mixed.QualificationError(
                    "CUDA performance build exceeded its "
                    f"{wall_deadline - started:.3f}-second hard wall deadline"
                )
            try:
                process.wait(
                    timeout=min(
                        BUILD_POLL_SECONDS,
                        max(0.001, wall_deadline - now),
                    )
                )
            except subprocess.TimeoutExpired:
                pass
        finished = time.monotonic()
        wall_seconds, pause_seconds, active_seconds = _build_elapsed_seconds(
            started,
            finished,
            dict(os.environ),
        )
    except BaseException:
        _terminate_build_process(process)
        raise
    if process.returncode != 0:
        raise mixed.QualificationError(
            f"CUDA performance build returned {process.returncode}"
        )
    if pause_seconds > BUILD_THERMAL_PAUSE_ALLOWANCE_SECONDS:
        raise mixed.QualificationError(
            "CUDA performance build completed after exceeding "
            f"{BUILD_THERMAL_PAUSE_ALLOWANCE_SECONDS:.3f} seconds "
            "of verified thermal pacing"
        )
    if active_seconds > BUILD_TIMEOUT_SECONDS:
        raise mixed.QualificationError(
            "CUDA performance build completed after exceeding "
            f"{BUILD_TIMEOUT_SECONDS:.3f} active seconds"
        )
    binary = ROOT / "target/release/kiln"
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise mixed.QualificationError(
            "CUDA performance build did not produce target/release/kiln"
        )
    return (
        binary,
        sha256_file(binary),
        active_seconds,
        wall_seconds,
        pause_seconds,
    )


def campaign_command(
    *,
    engine: str,
    model_path: Path,
    commit: str,
    output_directory: Path,
    binary: Path,
    reference_directory: Path | None,
) -> list[str]:
    if engine not in {"kiln", "vllm"}:
        raise mixed.QualificationError(f"unsupported campaign engine {engine!r}")
    command = [
        sys.executable,
        str(CAMPAIGN_RUNNER),
        "--engine",
        engine,
        "--base-url",
        "http://127.0.0.1:8420" if engine == "kiln" else "http://127.0.0.1:8421",
        "--model",
        MODEL_ID,
        "--model-path",
        str(model_path),
        "--runtime-identity",
        (
            f"kiln-git:{commit}"
            if engine == "kiln"
            else f"vllm-manifest:{VLLM_RUNTIME_MANIFEST_SHA256.removeprefix('sha256:')}"
        ),
        "--runtime-artifact",
        str(binary if engine == "kiln" else VLLM_RUNTIME_MANIFEST),
        "--campaign-id",
        f"cuda4090l-c1-{commit[:12]}",
        "--prompt-set-id",
        PROMPT_SET_ID,
        "--out-dir",
        str(output_directory),
        "--sizes",
        SIZES,
        "--repeats",
        str(REPEATS),
        "--max-tokens",
        str(MAX_TOKENS),
        "--warmup-requests",
        str(WARMUP_REQUESTS),
        "--seed",
        "20260725",
        "--memory-source",
        "nvml",
        "--memory-device-uuid",
        GPU_UUID,
        "--memory-limit-bytes",
        str(MEMORY_LIMIT_BYTES),
        "--memory-sample-ms",
        str(MEMORY_SAMPLE_MS),
        "--model-fingerprint-read-mib-per-second",
        str(MODEL_FINGERPRINT_READ_MIB_PER_SECOND),
        "--external-wsl2-thermal-policy",
        str(THERMAL_POLICY),
        "--server-launch-config",
        str(KILN_LAUNCH if engine == "kiln" else VLLM_LAUNCH),
        "--timeout-secs",
        str(REQUEST_TIMEOUT_SECONDS),
        "--output-evidence",
        "hashes",
        "--continue-after-failure",
    ]
    if reference_directory is not None:
        command.extend(("--reference-dir", str(reference_directory)))
    return command


def run_campaign(command: list[str], deadline: float, label: str) -> None:
    remaining = max(0.001, deadline - time.monotonic())
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise mixed.QualificationError(
            f"{label} campaign exceeded the case deadline"
        ) from exc
    if completed.returncode != 0:
        raise mixed.QualificationError(
            f"{label} campaign returned {completed.returncode}"
        )


def _load_campaign_summary(path: Path, engine: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise mixed.QualificationError(f"{engine} campaign summary is missing")
    try:
        value = strict_json_loads(path.read_bytes())
    except Exception as exc:
        raise mixed.QualificationError(
            f"cannot parse {engine} campaign summary: {exc}"
        ) from exc
    expected_keys = {
        "schema",
        "created_at",
        "campaign_id",
        "prompt_set_id",
        "engine",
        "reference_role",
        "reference_dir",
        "output_evidence",
        "model_fingerprint_read_mib_per_second",
        "execution_policy",
        "memory_sampler",
        "thermal_policy",
        "server_owner",
        "profiles",
        "verdict",
        "summary_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise mixed.QualificationError(f"{engine} campaign summary schema drifted")
    recorded_hash = value["summary_sha256"]
    unhashed = dict(value)
    unhashed.pop("summary_sha256")
    if recorded_hash != canonical_sha256(unhashed):
        raise mixed.QualificationError(
            f"{engine} campaign summary hash does not match"
        )
    if (
        value["schema"] != "kiln.serving-benchmark-campaign.v9"
        or value["engine"] != engine
        or value["verdict"] != "passed"
        or value["execution_policy"] != "continue_after_failure"
        or value["model_fingerprint_read_mib_per_second"]
        != MODEL_FINGERPRINT_READ_MIB_PER_SECOND
    ):
        raise mixed.QualificationError(
            f"{engine} campaign summary did not pass its bound contract"
        )
    return value


def summarize_campaign(
    output_directory: Path,
    engine: str,
    *,
    receipt_loader: Callable[[Path], dict[str, Any]] = bench.validate_benchmark_receipt_path,
) -> tuple[dict[str, float | int], list[str]]:
    summary = _load_campaign_summary(
        output_directory / f"campaign.{engine}.json", engine
    )
    rows = summary["profiles"]
    if (
        not isinstance(rows, list)
        or [row.get("profile") for row in rows if isinstance(row, dict)]
        != list(PROFILES)
    ):
        raise mixed.QualificationError(f"{engine} campaign profile order drifted")
    completion_tokens = 0
    elapsed_seconds = 0.0
    success_count = 0
    failure_count = 0
    peak_memory = 0
    profile_pass_count = 0
    receipt_hashes: list[str] = []
    for profile, row in zip(PROFILES, rows, strict=True):
        receipt_path = output_directory / f"{profile}.{engine}.json"
        receipt = receipt_loader(receipt_path)
        observed_hash = sha256_file(receipt_path)
        if (
            row.get("status") != "completed"
            or row.get("exit_code") != 0
            or row.get("receipt_sha256") != observed_hash
            or receipt["engine"]["name"] != engine
            or receipt["verdict"] != "passed"
            or receipt["workload"]["profile"] != profile
            or receipt["workload"]["concurrency"] != list(SIZE_VALUES)
            or receipt["workload"]["repeats"] != REPEATS
            or receipt["workload"]["max_tokens"] != MAX_TOKENS
        ):
            raise mixed.QualificationError(
                f"{engine} {profile} receipt disagrees with the campaign"
            )
        profile_pass_count += 1
        receipt_hashes.append(observed_hash)
        for run in receipt["runs"]:
            completion_tokens += run["completion_tokens"]
            elapsed_seconds += run["elapsed_s"]
            success_count += run["success_count"]
            failure_count += run["error_count"]
            peak_memory = max(peak_memory, run["memory"]["peak_bytes"])
    if elapsed_seconds <= 0:
        raise mixed.QualificationError(
            f"{engine} campaign has no positive request-window duration"
        )
    return (
        {
            "completion_token_count": completion_tokens,
            "output_token_throughput_per_second": (
                completion_tokens / elapsed_seconds
            ),
            "peak_device_memory_bytes": peak_memory,
            "profile_pass_count": profile_pass_count,
            "request_failure_count": failure_count,
            "request_success_count": success_count,
        },
        receipt_hashes,
    )


def exact_output_mismatch_count(vllm_directory: Path) -> int:
    mismatch_count = 0
    for profile in PROFILES:
        receipt = bench.validate_benchmark_receipt_path(
            vllm_directory / f"{profile}.vllm.json"
        )
        comparison = receipt.get("comparison")
        if not isinstance(comparison, dict):
            raise mixed.QualificationError(
                f"vLLM {profile} receipt lacks reference comparison"
            )
        mismatches = comparison.get("mismatches")
        if not isinstance(mismatches, list):
            raise mixed.QualificationError(
                f"vLLM {profile} comparison mismatch evidence is invalid"
            )
        mismatch_count += len(mismatches)
    return mismatch_count


def metrics_from_values(values: dict[str, float | int]) -> list[dict[str, Any]]:
    if set(values) != set(METRIC_DEFINITIONS):
        raise mixed.QualificationError(
            "CUDA performance metric set does not match its closed contract"
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
    return metrics_from_values({name: 0 for name in METRIC_DEFINITIONS})


def execute(
    model_path: Path, seed: int
) -> tuple[list[dict[str, Any]], str]:
    if seed != 20260725:
        raise mixed.QualificationError("CUDA performance seed drifted")
    bench.load_external_wsl2_boundary(THERMAL_POLICY)
    require_artifact_hash(KILN_LAUNCH, KILN_LAUNCH_SHA256, "Kiln launch")
    require_artifact_hash(VLLM_LAUNCH, VLLM_LAUNCH_SHA256, "vLLM launch")
    require_artifact_hash(KILN_CONFIG, KILN_CONFIG_SHA256, "Kiln config")
    require_artifact_hash(
        VLLM_RUNTIME_MANIFEST,
        VLLM_RUNTIME_MANIFEST_SHA256,
        "vLLM runtime manifest",
    )
    commit = git_commit()
    output_root = (
        ROOT
        / ".qualification/serving/cuda-rtx4090-laptop-performance-c1-v1"
        / commit
    )
    if output_root.exists():
        raise mixed.QualificationError(
            f"refusing to reuse performance artifact directory {output_root}"
        )
    deadline = time.monotonic() + CASE_TIMEOUT_SECONDS
    prompt_context = run_prompt_context_check(model_path, deadline)
    mixed.trace("cuda_performance_prompt_context_checked", **prompt_context)
    kiln_directory = output_root / "kiln"
    vllm_directory = output_root / "vllm"
    (
        binary,
        binary_sha256,
        build_seconds,
        build_wall_seconds,
        build_thermal_pause_seconds,
    ) = build_binary(deadline)
    mixed.trace(
        "cuda_performance_binary_built",
        binary_sha256=binary_sha256,
        build_seconds=build_seconds,
        build_thermal_pause_seconds=build_thermal_pause_seconds,
        build_wall_seconds=build_wall_seconds,
    )
    run_campaign(
        campaign_command(
            engine="kiln",
            model_path=model_path,
            commit=commit,
            output_directory=kiln_directory,
            binary=binary,
            reference_directory=None,
        ),
        deadline,
        "Kiln",
    )
    kiln, kiln_hashes = summarize_campaign(kiln_directory, "kiln")
    run_campaign(
        campaign_command(
            engine="vllm",
            model_path=model_path,
            commit=commit,
            output_directory=vllm_directory,
            binary=binary,
            reference_directory=kiln_directory,
        ),
        deadline,
        "vLLM",
    )
    vllm, vllm_hashes = summarize_campaign(vllm_directory, "vllm")
    mismatch_count = exact_output_mismatch_count(vllm_directory)
    values: dict[str, float | int] = {
        "build_duration_ms": build_seconds * 1000.0,
        "exact_output_mismatch_count": mismatch_count,
        **{f"kiln_{name}": value for name, value in kiln.items()},
        "paired_profile_pass_count": min(
            int(kiln["profile_pass_count"]),
            int(vllm["profile_pass_count"]),
        ),
        **{f"vllm_{name}": value for name, value in vllm.items()},
    }
    if (
        mismatch_count != 0
        or values["paired_profile_pass_count"] != len(PROFILES)
        or values["kiln_request_failure_count"] != 0
        or values["vllm_request_failure_count"] != 0
    ):
        raise mixed.QualificationError(
            "paired CUDA campaign did not satisfy every correctness gate"
        )
    details = json.dumps(
        {
            "artifact_root": str(output_root.relative_to(ROOT)),
            "binary_sha256": binary_sha256,
            "kiln_receipt_sha256": kiln_hashes,
            "vllm_receipt_sha256": vllm_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return metrics_from_values(values), details


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
        metrics, details = execute(model_path, args.seed)
        status = "passed"
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
