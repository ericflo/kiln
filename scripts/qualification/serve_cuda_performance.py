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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import serve_mixed_load as mixed
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
    "sha256:998e65f84651ce28d36acfcadb3f1216a589883054a9f27853fe755f373e3745"
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
        "cargo_service_runtime_max_seconds": 1740,
        "cargo_wrapper": "scripts/cargo-bounded.sh",
        "features": "cuda",
        "locked": True,
        "no_default_features": True,
        "offline": True,
        "package": "kiln-server",
        "profile": "release",
        "timeout_seconds": 1800,
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


def build_environment(source: dict[str, str]) -> dict[str, str]:
    environment = dict(source)
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
            "KILN_CARGO_SERVICE_RUNTIME_MAX_SECONDS": "1740",
            "KILN_CUDA_ARCHS": "89",
            "KILN_QUALIFICATION": "1",
        }
    )
    return environment


def build_binary(deadline: float) -> tuple[Path, str, float]:
    started = time.monotonic()
    remaining = min(BUILD_TIMEOUT_SECONDS, max(0.001, deadline - started))
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
            env=build_environment(dict(os.environ)),
            stdin=subprocess.DEVNULL,
            timeout=remaining,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise mixed.QualificationError(
            f"CUDA performance build exceeded {remaining:.3f} seconds"
        ) from exc
    if completed.returncode != 0:
        raise mixed.QualificationError(
            f"CUDA performance build returned {completed.returncode}"
        )
    binary = ROOT / "target/release/kiln"
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise mixed.QualificationError(
            "CUDA performance build did not produce target/release/kiln"
        )
    return binary, sha256_file(binary), time.monotonic() - started


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
    kiln_directory = output_root / "kiln"
    vllm_directory = output_root / "vllm"
    deadline = time.monotonic() + CASE_TIMEOUT_SECONDS
    binary, binary_sha256, build_seconds = build_binary(deadline)
    mixed.trace(
        "cuda_performance_binary_built",
        binary_sha256=binary_sha256,
        build_seconds=build_seconds,
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
