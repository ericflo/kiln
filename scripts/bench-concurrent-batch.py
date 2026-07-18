#!/usr/bin/env python3
"""Fail-closed OpenAI-compatible serving concurrency benchmark.

Measured requests use only ``POST /v1/chat/completions`` with streaming usage,
so the same driver and request bodies can be used for Kiln and vLLM.  Kiln's
``/health`` diagnostics are optional side evidence and never change the timed
request path.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA = "kiln.serving-benchmark.v1"
WORKLOAD_SCHEMA = "kiln.serving-benchmark-workload.v1"
HOST_THERMAL_POLICY_SCHEMA = "kiln.host-thermal-policy.v1"
DRIVER_VERSION = "3"
SUPPORTED_DRIVER_VERSIONS = {"2", DRIVER_VERSION}
LEGACY_PROMPT_TEMPLATE_VERSION = "equal-token-multiset-v1"
PROMPT_TEMPLATE_VERSION = "fixed-serving-profiles-v1"
ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_DIR = ROOT / "scripts" / "qualification"
if str(QUALIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(QUALIFICATION_DIR))

import host_thermal_guard as thermal  # noqa: E402
from model_fingerprint import (  # noqa: E402
    ModelFingerprintError,
    fingerprint_model,
)
from strict_json import loads as strict_json_loads  # noqa: E402

PROFILE_CONTRACTS = {
    "greedy-short": {
        "prompt_profile": "short",
        "temperature": 0.0,
        "top_p": 1.0,
        "require_uniform_prompt_tokens": True,
        "comparison_mode": "exact_output",
    },
    "api-default-sampled": {
        "prompt_profile": "short",
        "temperature": 1.0,
        "top_p": 1.0,
        "require_uniform_prompt_tokens": True,
        "comparison_mode": "inputs_only",
    },
    "long-prefill": {
        "prompt_profile": "long-prefill",
        "temperature": 0.0,
        "top_p": 1.0,
        "require_uniform_prompt_tokens": True,
        "comparison_mode": "exact_output",
    },
    "prefix-hit": {
        "prompt_profile": "prefix-hit",
        "temperature": 0.0,
        "top_p": 1.0,
        "require_uniform_prompt_tokens": True,
        "comparison_mode": "exact_output",
    },
    "mixed": {
        "prompt_profile": "mixed",
        "temperature": 0.0,
        "top_p": 1.0,
        "require_uniform_prompt_tokens": False,
        "comparison_mode": "exact_output",
    },
}

LONG_PROMPT_BLOCK = (
    "A production inference system must preserve request identity, bounded resource "
    "ownership, deterministic accounting, explicit cancellation, and observable phase "
    "transitions. Measurements must distinguish admission, prefill, decode, streaming, "
    "and teardown while retaining errors and tail latency. Shared prefixes exercise cache "
    "reuse only when every byte before the unique suffix is identical. "
)
LONG_PROMPT_REPETITIONS = 64

PROMPT_MARKERS = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "frost",
    "granite",
    "harbor",
    "indigo",
    "juniper",
    "keystone",
    "linen",
    "maple",
    "nickel",
    "onyx",
    "prairie",
    "quartz",
    "raven",
    "silver",
    "timber",
    "umber",
    "violet",
    "willow",
    "zinc",
)

COUNTER_FIELDS = (
    "total_decode_forwards",
    "total_batched_decode_forwards",
    "total_decode_rows",
    "total_decode_tokens",
    "total_decode_forward_ms",
    "total_prefill_forwards",
    "total_prefill_tokens",
    "total_prefill_layers",
    "total_prefill_layer_yields",
    "total_prefill_forward_ms",
    "total_admission_calls",
    "total_admission_ms",
    "total_errors",
)

SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
RECEIPT_KEYS = {
    "schema",
    "driver_version",
    "created_at",
    "engine",
    "driver_environment",
    "workload",
    "workload_fingerprint",
    "memory_sampler",
    "diagnostics",
    "warmup",
    "runs",
    "verdict",
    "receipt_sha256",
}
COMPLETION_CHECK_NAMES = (
    "repository_unchanged",
    "model_identity_unchanged",
    "runtime_artifact_unchanged",
    "runtime_manifest_unchanged",
    "execution_identity_unchanged",
    "host_thermal_handoff",
)
COMPLETION_CHECK_STATUSES = {"passed", "failed", "not_applicable"}
COMPLETION_FAILURE_PHASES = {
    "host_thermal_startup",
    "warmup",
    "measurement",
    "memory_sampler_stop",
    "reference_comparison",
    *COMPLETION_CHECK_NAMES,
}
RUN_KEYS = {
    "concurrency",
    "repeat",
    "elapsed_s",
    "request_count",
    "success_count",
    "error_count",
    "completion_tokens",
    "client_visible_stream_event_count",
    "request_throughput_per_s",
    "output_token_throughput_per_s",
    "slo_good_request_count",
    "slo_goodput_requests_per_s",
    "slo_goodput_tokens_per_s",
    "dispatch_spread_ms",
    "ttft_ms_p50",
    "ttft_ms_p99",
    "ttft_ms_p999",
    "client_visible_itl_ms_p50",
    "client_visible_itl_ms_p99",
    "client_visible_itl_ms_p999",
    "e2e_ms_p50",
    "e2e_ms_p99",
    "e2e_ms_p999",
    "prompt_tokens_min",
    "prompt_tokens_max",
    "prompt_set_sha256",
    "output_set_sha256",
    "memory",
    "server",
    "errors",
    "gates",
    "verdict",
}
RUN_KEYS_V3 = RUN_KEYS | {"prompt_token_counts", "host_thermal"}
MODEL_IDENTITY_KEYS = {
    "id",
    "path",
    "weight_files",
    "config_hash",
    "tokenizer_hash",
    "chat_template_hash",
    "content_sha256",
}
RUNTIME_ARTIFACT_KEYS = {"path", "bytes", "sha256"}
VLLM_RUNTIME_MANIFEST_KEYS = {
    "identity",
    "canonical_json",
    "system_fingerprint",
    "runtime_content_sha256",
}


class BenchmarkError(RuntimeError):
    """A benchmark contract or preflight failure."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be an object")
    return value


def _exact_keys(
    value: dict[str, Any], required: set[str], label: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise BenchmarkError(f"{label} missing keys: {', '.join(missing)}")
    if unknown:
        raise BenchmarkError(f"{label} has unknown keys: {', '.join(unknown)}")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BenchmarkError(f"{label} must be sha256:<64 lowercase hex>")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{label} must be a finite non-negative number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise BenchmarkError(f"{label} must be a finite non-negative number")
    return converted


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{label} must be a non-negative integer")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@dataclasses.dataclass(frozen=True)
class AttachedProcessGroup:
    pid: int
    process_group_id: int
    start_time_ticks: int
    boot_id: str
    executable: str
    cmdline_sha256: str
    proc_root: Path = dataclasses.field(default=Path("/proc"), repr=False)

    @staticmethod
    def _read_stat(pid: int, proc_root: Path) -> tuple[str, int, int]:
        try:
            raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError as exc:
            raise BenchmarkError(
                f"cannot read process identity for PID {pid}: {exc}"
            ) from exc
        close = raw.rfind(")")
        if close < 0:
            raise BenchmarkError(f"/proc/{pid}/stat has no command terminator")
        fields = raw[close + 2 :].split()
        if len(fields) < 20:
            raise BenchmarkError(f"/proc/{pid}/stat is truncated")
        try:
            process_group_id = int(fields[2])
            start_time_ticks = int(fields[19])
        except ValueError as exc:
            raise BenchmarkError(
                f"/proc/{pid}/stat has invalid identity fields"
            ) from exc
        return fields[0], process_group_id, start_time_ticks

    @classmethod
    def attach(
        cls,
        pid: int,
        *,
        proc_root: Path = Path("/proc"),
    ) -> "AttachedProcessGroup":
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise BenchmarkError("server PID must be an integer greater than one")
        state, process_group_id, start_time_ticks = cls._read_stat(pid, proc_root)
        if state == "Z":
            raise BenchmarkError(f"server PID {pid} is a zombie")
        if process_group_id != pid:
            raise BenchmarkError(
                f"server PID {pid} must lead its process group; observed PGID "
                f"{process_group_id}"
            )
        try:
            boot_id = (proc_root / "sys/kernel/random/boot_id").read_text(
                encoding="utf-8"
            ).strip()
            executable = os.readlink(proc_root / str(pid) / "exe")
            cmdline = (proc_root / str(pid) / "cmdline").read_bytes()
        except OSError as exc:
            raise BenchmarkError(f"cannot bind server PID {pid}: {exc}") from exc
        if not boot_id or not executable or not cmdline:
            raise BenchmarkError(f"server PID {pid} has incomplete process identity")
        return cls(
            pid=pid,
            process_group_id=process_group_id,
            start_time_ticks=start_time_ticks,
            boot_id=boot_id,
            executable=executable,
            cmdline_sha256="sha256:" + hashlib.sha256(cmdline).hexdigest(),
            proc_root=proc_root,
        )

    def poll(self) -> int | None:
        try:
            state, process_group_id, start_time_ticks = self._read_stat(
                self.pid, self.proc_root
            )
        except BenchmarkError:
            return 0
        if (
            state == "Z"
            or process_group_id != self.process_group_id
            or start_time_ticks != self.start_time_ticks
        ):
            return 0
        return None

    def receipt_identity(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "start_time_ticks": self.start_time_ticks,
            "boot_id": self.boot_id,
            "executable": self.executable,
            "cmdline_sha256": self.cmdline_sha256,
        }


HOST_THERMAL_POLICY_KEYS = {
    "schema",
    "id",
    "sensor",
    "limit_millicelsius",
    "poll_interval_ms",
    "pacing",
    "safe_handoff",
    "phase_settlement_timeout_seconds",
}


def validate_host_thermal_policy_value(
    value: Any,
    label: str,
) -> tuple[dict[str, Any], thermal.HostThermalPolicy, float]:
    value = _object(value, label)
    has_content_hash = "content_sha256" in value
    _exact_keys(
        value,
        HOST_THERMAL_POLICY_KEYS
        | ({"content_sha256"} if has_content_hash else set()),
        label,
    )
    raw = dict(value)
    recorded_hash = raw.pop("content_sha256", None)
    if raw["schema"] != HOST_THERMAL_POLICY_SCHEMA:
        raise BenchmarkError(
            f"host thermal policy must use schema {HOST_THERMAL_POLICY_SCHEMA}"
        )
    if not isinstance(raw["id"], str) or re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,127}", raw["id"]
    ) is None:
        raise BenchmarkError("host thermal policy id must be a portable identifier")
    sensor = _object(raw["sensor"], f"{label}.sensor")
    _exact_keys(sensor, {"hwmon_name", "label"}, f"{label}.sensor")
    for name in ("hwmon_name", "label"):
        if not isinstance(sensor[name], str) or not sensor[name]:
            raise BenchmarkError(f"{label}.sensor.{name} must be non-empty")
    pacing = _object(raw["pacing"], f"{label}.pacing")
    legacy_hashed_pacing = (
        has_content_hash and "resume_stable_samples" not in pacing
    )
    _exact_keys(
        pacing,
        {"start_millicelsius", "resume_millicelsius"}
        | (set() if legacy_hashed_pacing else {"resume_stable_samples"}),
        f"{label}.pacing",
    )
    handoff = _object(raw["safe_handoff"], f"{label}.safe_handoff")
    _exact_keys(
        handoff,
        {"target_millicelsius", "stable_samples", "timeout_seconds"},
        f"{label}.safe_handoff",
    )
    settlement_timeout = _nonnegative_number(
        raw["phase_settlement_timeout_seconds"],
        f"{label}.phase_settlement_timeout_seconds",
    )
    if settlement_timeout <= 0:
        raise BenchmarkError(
            "host thermal policy phase settlement timeout must be positive"
        )
    policy = thermal.HostThermalPolicy(
        hwmon_name=sensor.get("hwmon_name"),
        label=sensor.get("label"),
        limit_millicelsius=raw["limit_millicelsius"],
        poll_interval_ms=raw["poll_interval_ms"],
        pacing_start_millicelsius=pacing["start_millicelsius"],
        pacing_resume_millicelsius=pacing["resume_millicelsius"],
        pacing_resume_stable_samples=pacing.get("resume_stable_samples", 1),
        cooldown_target_millicelsius=handoff["target_millicelsius"],
        cooldown_stable_samples=handoff["stable_samples"],
        cooldown_timeout_seconds=handoff["timeout_seconds"],
        cooldown_mode="live_process_safe_handoff",
        error_type=BenchmarkError,
    )
    content_hash = canonical_sha256(raw)
    if recorded_hash is not None and _sha256(
        recorded_hash, f"{label}.content_sha256"
    ) != content_hash:
        raise BenchmarkError(f"{label}.content_sha256 does not match policy content")
    normalized = dict(raw)
    normalized["content_sha256"] = content_hash
    return normalized, policy, settlement_timeout


def load_host_thermal_policy(
    path: Path,
) -> tuple[dict[str, Any], thermal.HostThermalPolicy, float]:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkError(f"host thermal policy is not a regular file: {path}")
    data = path.read_bytes()
    if len(data) > 64 * 1024:
        raise BenchmarkError("host thermal policy exceeds 64 KiB")
    try:
        raw = strict_json_loads(data)
    except Exception as exc:
        raise BenchmarkError(f"cannot load host thermal policy {path}: {exc}") from exc
    return validate_host_thermal_policy_value(raw, "host thermal policy")


def model_content(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": value["id"],
        "weight_files": value["weight_files"],
        "config_hash": value["config_hash"],
        "tokenizer_hash": value["tokenizer_hash"],
        "chat_template_hash": value["chat_template_hash"],
    }


def bind_model_identity(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = canonical_sha256(model_content(result))
    return result


def validate_model_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _object(value, label)
    _exact_keys(identity, MODEL_IDENTITY_KEYS, label)
    if not isinstance(identity["id"], str) or not identity["id"]:
        raise BenchmarkError(f"{label}.id must be a non-empty string")
    if not isinstance(identity["path"], str) or not Path(identity["path"]).is_absolute():
        raise BenchmarkError(f"{label}.path must be an absolute path")
    weights = identity["weight_files"]
    if not isinstance(weights, list) or not weights:
        raise BenchmarkError(f"{label}.weight_files must be a non-empty array")
    paths: list[str] = []
    for index, item_value in enumerate(weights):
        item = _object(item_value, f"{label}.weight_files[{index}]")
        _exact_keys(item, {"path", "sha256", "bytes"}, f"{label}.weight_files[{index}]")
        if not isinstance(item["path"], str) or not item["path"]:
            raise BenchmarkError(f"{label}.weight_files[{index}].path must be non-empty")
        paths.append(item["path"])
        _positive_int(item["bytes"], f"{label}.weight_files[{index}].bytes")
        _sha256(item["sha256"], f"{label}.weight_files[{index}].sha256")
    if paths != sorted(set(paths), key=lambda path: path.encode("utf-8")):
        raise BenchmarkError(f"{label}.weight_files must be unique and bytewise sorted")
    for name in ("config_hash", "tokenizer_hash"):
        _sha256(identity[name], f"{label}.{name}")
    if identity["chat_template_hash"] is not None:
        _sha256(identity["chat_template_hash"], f"{label}.chat_template_hash")
    expected = canonical_sha256(model_content(identity))
    if _sha256(identity["content_sha256"], f"{label}.content_sha256") != expected:
        raise BenchmarkError(f"{label}.content_sha256 does not match model content")
    return identity


def fingerprint_runtime_artifact(path: Path) -> dict[str, Any]:
    absolute = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise BenchmarkError(f"cannot open runtime artifact {absolute}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise BenchmarkError(f"runtime artifact is not a non-empty regular file: {absolute}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = absolute.stat(follow_symlinks=False)
        except OSError as exc:
            raise BenchmarkError(f"cannot recheck runtime artifact {absolute}: {exc}") from exc
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
        ):
            raise BenchmarkError(f"runtime artifact changed while hashing: {absolute}")
        return {
            "path": str(absolute),
            "bytes": before.st_size,
            "sha256": "sha256:" + digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def validate_runtime_artifact(value: Any, label: str) -> dict[str, Any]:
    artifact = _object(value, label)
    _exact_keys(artifact, RUNTIME_ARTIFACT_KEYS, label)
    if not isinstance(artifact["path"], str) or not Path(artifact["path"]).is_absolute():
        raise BenchmarkError(f"{label}.path must be an absolute path")
    _positive_int(artifact["bytes"], f"{label}.bytes")
    _sha256(artifact["sha256"], f"{label}.sha256")
    return artifact


def validate_vllm_runtime_manifest(value: Any, label: str) -> dict[str, Any]:
    manifest = _object(value, label)
    _exact_keys(manifest, VLLM_RUNTIME_MANIFEST_KEYS, label)
    identity = _object(manifest["identity"], f"{label}.identity")
    canonical_json = manifest["canonical_json"]
    if not isinstance(canonical_json, str) or not canonical_json:
        raise BenchmarkError(f"{label}.canonical_json must be non-empty")
    expected_json = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if canonical_json != expected_json:
        raise BenchmarkError(f"{label}.canonical_json does not match its identity")
    fingerprint = manifest["system_fingerprint"]
    if not isinstance(fingerprint, str):
        raise BenchmarkError(f"{label}.system_fingerprint must be a string")
    parts = fingerprint.split(".")
    if (
        len(parts) != 3
        or parts[0] != "kiln-teacher-v1"
        or re.fullmatch(r"[A-Za-z0-9_-]+", parts[1]) is None
        or re.fullmatch(r"[0-9a-f]{64}", parts[2]) is None
    ):
        raise BenchmarkError(f"{label}.system_fingerprint has an invalid shape")
    try:
        payload = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
    except (ValueError, binascii.Error) as exc:
        raise BenchmarkError(f"{label}.system_fingerprint has invalid base64url") from exc
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    if (
        encoded != parts[1]
        or payload != canonical_json.encode("utf-8")
        or hashlib.sha256(payload).hexdigest() != parts[2]
    ):
        raise BenchmarkError(f"{label}.system_fingerprint does not bind canonical_json")
    runtime_hash = manifest["runtime_content_sha256"]
    if not isinstance(runtime_hash, str) or re.fullmatch(r"[0-9a-f]{64}", runtime_hash) is None:
        raise BenchmarkError(f"{label}.runtime_content_sha256 must be 64 lowercase hex")
    if not isinstance(identity.get("implementation"), str) or not identity[
        "implementation"
    ].startswith("vllm:"):
        raise BenchmarkError(f"{label}.identity implementation must identify vLLM")
    if not isinstance(identity.get("served_model_id"), str) or not identity[
        "served_model_id"
    ]:
        raise BenchmarkError(f"{label}.identity served_model_id must be non-empty")
    return manifest


def load_vllm_runtime_manifest(path: Path) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.expanduser().absolute().read_bytes())
    except Exception as exc:
        raise BenchmarkError(f"cannot load vLLM runtime manifest {path}: {exc}") from exc
    return validate_vllm_runtime_manifest(value, "vLLM runtime manifest")


RUN_HOST_THERMAL_KEYS = {
    "phase",
    "phase_wall_seconds",
    "thermally_sustainable_output_token_throughput_per_s",
    "host_temperature_start_millicelsius",
    "host_temperature_end_millicelsius",
    "host_temperature_peak_millicelsius",
    "host_temperature_sample_count",
    "host_thermal_guard_trip_count",
    "host_thermal_pacing_event_count",
    "host_thermal_pacing_completed_event_count",
    "host_thermal_pacing_seconds",
}


def validate_run_host_thermal(
    value: Any,
    *,
    label: str,
    phase: str,
    completion_tokens: int,
) -> None:
    if value is None:
        return
    evidence = _object(value, label)
    _exact_keys(evidence, RUN_HOST_THERMAL_KEYS, label)
    if evidence["phase"] != phase:
        raise BenchmarkError(f"{label}.phase disagrees with its run")
    wall_seconds = _nonnegative_number(
        evidence["phase_wall_seconds"], f"{label}.phase_wall_seconds"
    )
    if wall_seconds <= 0:
        raise BenchmarkError(f"{label}.phase_wall_seconds must be positive")
    throughput = _nonnegative_number(
        evidence["thermally_sustainable_output_token_throughput_per_s"],
        f"{label}.thermally_sustainable_output_token_throughput_per_s",
    )
    if not math.isclose(
        throughput,
        completion_tokens / wall_seconds,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise BenchmarkError(f"{label} has inconsistent sustainable throughput")
    for name in (
        "host_temperature_start_millicelsius",
        "host_temperature_end_millicelsius",
        "host_temperature_peak_millicelsius",
    ):
        temperature = evidence[name]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, int)
            or not -100_000 <= temperature <= 250_000
        ):
            raise BenchmarkError(f"{label}.{name} is not a plausible temperature")
    sample_count = _positive_int(
        evidence["host_temperature_sample_count"],
        f"{label}.host_temperature_sample_count",
    )
    if sample_count < 2:
        raise BenchmarkError(f"{label} requires boundary temperature samples")
    for name in (
        "host_thermal_guard_trip_count",
        "host_thermal_pacing_event_count",
        "host_thermal_pacing_completed_event_count",
    ):
        _nonnegative_int(evidence[name], f"{label}.{name}")
    if evidence["host_thermal_guard_trip_count"] not in {0, 1}:
        raise BenchmarkError(f"{label}.host_thermal_guard_trip_count must be zero or one")
    if (
        evidence["host_thermal_pacing_completed_event_count"]
        > evidence["host_thermal_pacing_event_count"]
    ):
        raise BenchmarkError(f"{label} completed more pacing events than it started")
    _nonnegative_number(
        evidence["host_thermal_pacing_seconds"],
        f"{label}.host_thermal_pacing_seconds",
    )


def validate_benchmark_run(
    value: Any,
    *,
    label: str,
    concurrency: int,
    repeat: int,
    max_tokens: int,
    driver_version: str,
    memory_limit_bytes: int | None,
    workload_profile: str | None,
) -> None:
    row = _object(value, label)
    _exact_keys(row, RUN_KEYS_V3 if driver_version == "3" else RUN_KEYS, label)
    if row["concurrency"] != concurrency or row["repeat"] != repeat:
        raise BenchmarkError(f"{label} does not match its declared concurrency/repeat")
    if row["request_count"] != concurrency:
        raise BenchmarkError(f"{label}.request_count must equal concurrency")
    for name in (
        "success_count",
        "error_count",
        "completion_tokens",
        "client_visible_stream_event_count",
        "slo_good_request_count",
    ):
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkError(f"{label}.{name} must be a non-negative integer")
    for name in (
        "elapsed_s",
        "request_throughput_per_s",
        "output_token_throughput_per_s",
        "slo_goodput_requests_per_s",
        "slo_goodput_tokens_per_s",
        "dispatch_spread_ms",
    ):
        _nonnegative_number(row[name], f"{label}.{name}")
    if driver_version == "3":
        phase = (
            f"warmup-c{concurrency:03d}"
            if repeat == -1
            else f"measure-c{concurrency:03d}-r{repeat:03d}"
        )
        validate_run_host_thermal(
            row["host_thermal"],
            label=f"{label}.host_thermal",
            phase=phase,
            completion_tokens=row["completion_tokens"],
        )
    for name in (
        "ttft_ms_p50",
        "ttft_ms_p99",
        "ttft_ms_p999",
        "client_visible_itl_ms_p50",
        "client_visible_itl_ms_p99",
        "client_visible_itl_ms_p999",
        "e2e_ms_p50",
        "e2e_ms_p99",
        "e2e_ms_p999",
    ):
        if row[name] is not None:
            _nonnegative_number(row[name], f"{label}.{name}")
    _sha256(row["prompt_set_sha256"], f"{label}.prompt_set_sha256")
    _sha256(row["output_set_sha256"], f"{label}.output_set_sha256")
    if driver_version == "3":
        prompt_token_counts = row["prompt_token_counts"]
        if (
            not isinstance(prompt_token_counts, list)
            or len(prompt_token_counts) != concurrency
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in prompt_token_counts
            )
        ):
            raise BenchmarkError(
                f"{label}.prompt_token_counts must contain one non-negative integer per request"
            )
        if min(prompt_token_counts) != row["prompt_tokens_min"] or max(
            prompt_token_counts
        ) != row["prompt_tokens_max"]:
            raise BenchmarkError(f"{label}.prompt token summaries disagree")

    errors = row["errors"]
    if not isinstance(errors, list):
        raise BenchmarkError(f"{label}.errors must be an array")
    error_indices: set[int] = set()
    for index, error_value in enumerate(errors):
        error = _object(error_value, f"{label}.errors[{index}]")
        _exact_keys(error, {"index", "error"}, f"{label}.errors[{index}]")
        if (
            isinstance(error["index"], bool)
            or not isinstance(error["index"], int)
            or not 0 <= error["index"] < concurrency
            or error["index"] in error_indices
        ):
            raise BenchmarkError(f"{label}.errors has an invalid or duplicate index")
        if not isinstance(error["error"], str) or not error["error"]:
            raise BenchmarkError(f"{label}.errors[{index}].error must be non-empty")
        error_indices.add(error["index"])
    if len(errors) != row["error_count"]:
        raise BenchmarkError(f"{label}.error_count does not match errors")
    if row["success_count"] + row["error_count"] != concurrency:
        raise BenchmarkError(f"{label} success and error counts must cover every request")
    if driver_version == "3":
        for index, count in enumerate(row["prompt_token_counts"]):
            if (index in error_indices) != (count == 0):
                raise BenchmarkError(
                    f"{label}.prompt_token_counts does not align with request errors"
                )
    gates = row["gates"]
    if not isinstance(gates, list) or not gates:
        raise BenchmarkError(f"{label}.gates must be a non-empty array")
    gate_names: set[str] = set()
    for index, gate_value in enumerate(gates):
        gate = _object(gate_value, f"{label}.gates[{index}]")
        _exact_keys(gate, {"name", "detail", "passed"}, f"{label}.gates[{index}]")
        if not isinstance(gate["name"], str) or not gate["name"] or gate["name"] in gate_names:
            raise BenchmarkError(f"{label}.gates has an empty or duplicate name")
        if not isinstance(gate["detail"], str) or not isinstance(gate["passed"], bool):
            raise BenchmarkError(f"{label}.gates[{index}] has invalid field types")
        gate_names.add(gate["name"])

    if row["memory"] is not None:
        memory = _object(row["memory"], f"{label}.memory")
        _exact_keys(
            memory,
            {"baseline_bytes", "peak_bytes", "peak_delta_bytes", "samples"},
            f"{label}.memory",
        )
        for name, item in memory.items():
            if item is not None:
                _nonnegative_number(item, f"{label}.memory.{name}")
    if memory_limit_bytes is not None:
        memory_gate = next(
            (item for item in gates if item["name"] == "absolute_memory_limit"), None
        )
        expected_memory_pass = (
            row["memory"] is not None
            and row["memory"]["peak_bytes"] <= memory_limit_bytes
        )
        if memory_gate is None or memory_gate["passed"] != expected_memory_pass:
            raise BenchmarkError(f"{label} has an inconsistent absolute-memory gate")
        memory_measured_gate = next(
            (item for item in gates if item["name"] == "memory_measured"), None
        )
        expected_measured = row["memory"] is not None and row["memory"]["samples"] >= 2
        if memory_measured_gate is None or memory_measured_gate["passed"] != expected_measured:
            raise BenchmarkError(f"{label} has an inconsistent memory-measurement gate")
    if driver_version == "3" and workload_profile is not None:
        uniform = PROFILE_CONTRACTS[workload_profile]["require_uniform_prompt_tokens"]
        expected_name = (
            "mixed_prompt_tokens"
            if workload_profile == "mixed" and concurrency > 1
            else "uniform_prompt_tokens" if uniform else None
        )
        if expected_name is not None:
            token_gate = next(
                (item for item in gates if item["name"] == expected_name), None
            )
            expected_pass = (
                len(set(row["prompt_token_counts"])) > 1
                if expected_name == "mixed_prompt_tokens"
                else len(set(row["prompt_token_counts"])) == 1
            )
            if token_gate is None or token_gate["passed"] != expected_pass:
                raise BenchmarkError(f"{label} has an inconsistent prompt-shape gate")
    if row["server"] is not None:
        server = _object(row["server"], f"{label}.server")
        required_server = set(COUNTER_FIELDS) | {
            "total_batched_decode_forwards",
            "effective_max_decode_batch",
            "process_max_observed_batch",
            "mean_decode_rows_per_forward",
            "batched_decode_forward_fraction",
        }
        _exact_keys(server, required_server, f"{label}.server")
        for name, item in server.items():
            if item is not None:
                _nonnegative_number(item, f"{label}.server.{name}")

    passed = (
        row["success_count"] == concurrency
        and row["error_count"] == 0
        and row["completion_tokens"] == concurrency * max_tokens
        and all(gate["passed"] for gate in gates)
    )
    expected_verdict = "passed" if passed else "failed"
    if row["verdict"] != expected_verdict:
        raise BenchmarkError(f"{label}.verdict is inconsistent with its requests and gates")


HOST_THERMAL_EVIDENCE_KEYS = {
    "host_temperature_start_millicelsius",
    "host_temperature_end_millicelsius",
    "host_temperature_peak_millicelsius",
    "host_temperature_sample_count",
    "host_thermal_guard_trip_count",
    "host_thermal_pacing_active_end",
    "host_thermal_pacing_completed_event_count",
    "host_thermal_pacing_event_count",
    "host_thermal_pacing_max_seconds",
    "host_thermal_pacing_max_start_millicelsius",
    "host_thermal_pacing_seconds",
    "host_thermal_cooldown_active_end",
    "host_thermal_cooldown_completed_count",
    "host_thermal_cooldown_peak_millicelsius",
    "host_thermal_cooldown_sample_count",
    "host_thermal_cooldown_seconds",
    "host_thermal_cooldown_stable_sample_count",
    "host_thermal_cooldown_timeout_count",
    "sensor_path",
    "trip_reason",
    "errors",
    "process_alive_at_handoff",
}


def validate_host_thermal_receipt(value: Any) -> tuple[str, bool]:
    host_thermal = _object(value, "receipt.host_thermal")
    _exact_keys(
        host_thermal,
        {
            "mode",
            "unsafe_no_guard_acknowledged",
            "policy",
            "process_group",
            "evidence",
        },
        "receipt.host_thermal",
    )
    if not isinstance(host_thermal["unsafe_no_guard_acknowledged"], bool):
        raise BenchmarkError(
            "receipt.host_thermal.unsafe_no_guard_acknowledged must be boolean"
        )
    mode = host_thermal["mode"]
    if mode == "not_configured":
        if host_thermal["unsafe_no_guard_acknowledged"] is not True or any(
            host_thermal[name] is not None
            for name in ("policy", "process_group", "evidence")
        ):
            raise BenchmarkError(
                "unconfigured host thermal evidence requires an explicit unsafe acknowledgment"
            )
        return mode, False
    if mode != "attached_process_group":
        raise BenchmarkError("receipt.host_thermal.mode is unsupported")
    if host_thermal["unsafe_no_guard_acknowledged"] is not False:
        raise BenchmarkError("attached host thermal evidence cannot be marked unsafe")
    policy_record, _policy, _settlement_timeout = (
        validate_host_thermal_policy_value(
            host_thermal["policy"], "receipt.host_thermal.policy"
        )
    )
    process = _object(
        host_thermal["process_group"], "receipt.host_thermal.process_group"
    )
    _exact_keys(
        process,
        {
            "pid",
            "process_group_id",
            "start_time_ticks",
            "boot_id",
            "executable",
            "cmdline_sha256",
        },
        "receipt.host_thermal.process_group",
    )
    pid = _positive_int(process["pid"], "receipt.host_thermal.process_group.pid")
    if pid <= 1 or process["process_group_id"] != pid:
        raise BenchmarkError("receipt host thermal process must be its group leader")
    _positive_int(
        process["start_time_ticks"],
        "receipt.host_thermal.process_group.start_time_ticks",
    )
    for name in ("boot_id", "executable"):
        if not isinstance(process[name], str) or not process[name]:
            raise BenchmarkError(
                f"receipt.host_thermal.process_group.{name} must be non-empty"
            )
    _sha256(
        process["cmdline_sha256"],
        "receipt.host_thermal.process_group.cmdline_sha256",
    )
    evidence = _object(host_thermal["evidence"], "receipt.host_thermal.evidence")
    _exact_keys(
        evidence,
        HOST_THERMAL_EVIDENCE_KEYS,
        "receipt.host_thermal.evidence",
    )
    for name in (
        "host_temperature_start_millicelsius",
        "host_temperature_end_millicelsius",
        "host_temperature_peak_millicelsius",
        "host_thermal_pacing_max_start_millicelsius",
        "host_thermal_cooldown_peak_millicelsius",
    ):
        value = evidence[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -100_000 <= value <= 250_000
        ):
            raise BenchmarkError(f"receipt.host_thermal.evidence.{name} is invalid")
    for name in (
        "host_temperature_sample_count",
        "host_thermal_guard_trip_count",
        "host_thermal_pacing_active_end",
        "host_thermal_pacing_completed_event_count",
        "host_thermal_pacing_event_count",
        "host_thermal_cooldown_active_end",
        "host_thermal_cooldown_completed_count",
        "host_thermal_cooldown_sample_count",
        "host_thermal_cooldown_stable_sample_count",
        "host_thermal_cooldown_timeout_count",
    ):
        _nonnegative_int(evidence[name], f"receipt.host_thermal.evidence.{name}")
    if evidence["host_temperature_sample_count"] <= 0:
        raise BenchmarkError("receipt host thermal evidence has no samples")
    for name in (
        "host_thermal_guard_trip_count",
        "host_thermal_pacing_active_end",
        "host_thermal_cooldown_active_end",
        "host_thermal_cooldown_completed_count",
        "host_thermal_cooldown_timeout_count",
    ):
        if evidence[name] not in {0, 1}:
            raise BenchmarkError(f"receipt.host_thermal.evidence.{name} must be zero or one")
    for name in (
        "host_thermal_pacing_max_seconds",
        "host_thermal_pacing_seconds",
        "host_thermal_cooldown_seconds",
    ):
        _nonnegative_number(evidence[name], f"receipt.host_thermal.evidence.{name}")
    if (
        evidence["host_thermal_pacing_completed_event_count"]
        > evidence["host_thermal_pacing_event_count"]
    ):
        raise BenchmarkError("receipt completed more thermal pacing events than it began")
    if evidence["host_temperature_peak_millicelsius"] < max(
        evidence["host_temperature_start_millicelsius"],
        evidence["host_temperature_end_millicelsius"],
    ):
        raise BenchmarkError("receipt host thermal peak is below a boundary sample")
    if not isinstance(evidence["sensor_path"], str) or not evidence["sensor_path"]:
        raise BenchmarkError("receipt host thermal sensor path must be non-empty")
    if evidence["trip_reason"] is not None and (
        not isinstance(evidence["trip_reason"], str) or not evidence["trip_reason"]
    ):
        raise BenchmarkError("receipt host thermal trip reason must be null or non-empty")
    errors = evidence["errors"]
    if (
        not isinstance(errors, list)
        or len(errors) > 8
        or any(not isinstance(error, str) or not error for error in errors)
    ):
        raise BenchmarkError("receipt host thermal errors must be bounded strings")
    if not isinstance(evidence["process_alive_at_handoff"], bool):
        raise BenchmarkError("receipt host thermal process handoff state must be boolean")
    tripped = evidence["trip_reason"] is not None
    if evidence["host_thermal_guard_trip_count"] != int(tripped):
        raise BenchmarkError("receipt host thermal trip fields disagree")
    policy_limit = policy_record["limit_millicelsius"]
    if not tripped and evidence["host_temperature_peak_millicelsius"] >= policy_limit:
        raise BenchmarkError("receipt host thermal peak reached the limit without a trip")
    safe_handoff = policy_record["safe_handoff"]
    operationally_passed = (
        not tripped
        and not errors
        and evidence["process_alive_at_handoff"]
        and evidence["host_thermal_pacing_active_end"] == 0
        and evidence["host_thermal_pacing_completed_event_count"]
        == evidence["host_thermal_pacing_event_count"]
        and evidence["host_thermal_cooldown_active_end"] == 0
        and evidence["host_thermal_cooldown_completed_count"] == 1
        and evidence["host_thermal_cooldown_timeout_count"] == 0
        and evidence["host_thermal_cooldown_stable_sample_count"]
        >= safe_handoff["stable_samples"]
        and evidence["host_temperature_end_millicelsius"]
        <= safe_handoff["target_millicelsius"]
    )
    return mode, operationally_passed


def validate_benchmark_receipt(value: Any) -> dict[str, Any]:
    receipt = _object(value, "receipt")
    driver_version = receipt.get("driver_version")
    required_receipt_keys = set(RECEIPT_KEYS)
    if driver_version == "3":
        required_receipt_keys.update({"completion", "host_thermal"})
    _exact_keys(receipt, required_receipt_keys, "receipt", {"comparison"})
    if receipt["schema"] != SCHEMA or driver_version not in SUPPORTED_DRIVER_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_DRIVER_VERSIONS))
        raise BenchmarkError(f"receipt must use {SCHEMA} driver version in {{{supported}}}")
    try:
        created_at = dt.datetime.fromisoformat(receipt["created_at"])
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("receipt.created_at must be an ISO-8601 timestamp") from exc
    if created_at.tzinfo is None:
        raise BenchmarkError("receipt.created_at must include a timezone")
    recorded_hash = _sha256(receipt["receipt_sha256"], "receipt.receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256")
    if canonical_sha256(unhashed) != recorded_hash:
        raise BenchmarkError("receipt.receipt_sha256 does not match canonical content")

    engine = _object(receipt["engine"], "receipt.engine")
    engine_keys = {
        "name",
        "runtime_identity",
        "reported_version",
        "base_url",
        "model",
        "available_models",
        "authentication_configured",
    }
    engine_optional = {"authentication_source"}
    if driver_version == "3":
        engine_keys |= {
            "model_identity",
            "runtime_artifact",
            "runtime_execution_identity",
            "runtime_manifest",
        }
    _exact_keys(engine, engine_keys, "receipt.engine", engine_optional)
    if engine["name"] not in {"kiln", "vllm"}:
        raise BenchmarkError("receipt.engine.name must be kiln or vllm")
    for name in ("runtime_identity", "base_url", "model"):
        if not isinstance(engine[name], str) or not engine[name]:
            raise BenchmarkError(f"receipt.engine.{name} must be a non-empty string")
    if not isinstance(engine["authentication_configured"], bool):
        raise BenchmarkError("receipt.engine.authentication_configured must be boolean")
    if "authentication_source" in engine:
        if engine["authentication_source"] not in {"none", "argument", "environment"}:
            raise BenchmarkError("receipt.engine.authentication_source is invalid")
        if engine["authentication_configured"] != (engine["authentication_source"] != "none"):
            raise BenchmarkError("receipt.engine authentication fields disagree")
    if driver_version == "3":
        model_identity = validate_model_identity(
            engine["model_identity"], "receipt.engine.model_identity"
        )
        if model_identity["id"] != engine["model"]:
            raise BenchmarkError("receipt.engine model alias and model identity disagree")
        artifact = validate_runtime_artifact(
            engine["runtime_artifact"], "receipt.engine.runtime_artifact"
        )
        execution_identity = engine["runtime_execution_identity"]
        if engine["name"] == "kiln":
            execution_identity = _object(
                execution_identity, "receipt.engine.runtime_execution_identity"
            )
            if execution_identity.get("executable_sha256") != artifact["sha256"]:
                raise BenchmarkError(
                    "receipt Kiln execution identity does not bind the runtime artifact"
                )
            if (
                execution_identity.get("provenance_type")
                != "kiln.execution-provenance.v1"
            ):
                raise BenchmarkError("receipt Kiln execution provenance type is unsupported")
            for name in ("backend", "device", "inference_dtype", "training_policy"):
                if (
                    not isinstance(execution_identity.get(name), str)
                    or not execution_identity[name]
                ):
                    raise BenchmarkError(
                        f"receipt.engine.runtime_execution_identity.{name} must be non-empty"
                    )
            for name in (
                "provenance_sha256",
                "executable_sha256",
                "numerical_runtime_sha256",
                "kernel_contract_sha256",
                "effective_server_config_sha256",
                "effective_environment_sha256",
            ):
                _sha256(
                    execution_identity.get(name),
                    f"receipt.engine.runtime_execution_identity.{name}",
                )
        elif execution_identity is not None:
            raise BenchmarkError("receipt vLLM runtime execution identity must be null")
        runtime_manifest = engine["runtime_manifest"]
        if engine["name"] == "vllm":
            runtime_manifest = validate_vllm_runtime_manifest(
                runtime_manifest, "receipt.engine.runtime_manifest"
            )
            if runtime_manifest["identity"]["served_model_id"] != engine["model"]:
                raise BenchmarkError(
                    "receipt vLLM runtime manifest model disagrees with engine model"
                )
        elif runtime_manifest is not None:
            raise BenchmarkError("receipt Kiln runtime manifest must be null")

    driver_environment = _object(
        receipt["driver_environment"], "receipt.driver_environment"
    )
    _exact_keys(
        driver_environment,
        {"hostname", "platform", "machine", "python", "repository"},
        "receipt.driver_environment",
    )
    repository = _object(
        driver_environment["repository"], "receipt.driver_environment.repository"
    )
    _exact_keys(
        repository,
        {"commit", "dirty", "source_tree_sha256"},
        "receipt.driver_environment.repository",
    )
    if (
        not isinstance(repository["commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", repository["commit"]) is None
    ):
        raise BenchmarkError("receipt repository commit must be 40 lowercase hex characters")
    if not isinstance(repository["dirty"], bool):
        raise BenchmarkError("receipt repository dirty flag must be boolean")
    _sha256(repository["source_tree_sha256"], "receipt repository source_tree_sha256")

    workload = _object(receipt["workload"], "receipt.workload")
    workload_keys = {
        "schema",
        "prompt_template_version",
        "run_id",
        "model",
        "endpoint",
        "stream",
        "stream_include_usage",
        "concurrency",
        "repeats",
        "warmup_requests",
        "max_tokens",
        "sampling",
        "chat_template_kwargs",
        "arrival_pattern",
        "require_max_tokens",
        "require_uniform_prompt_tokens",
        "max_dispatch_spread_ms",
        "slo",
    }
    if driver_version == "3":
        workload_keys |= {"profile", "comparison_mode", "memory_limit_bytes"}
    _exact_keys(workload, workload_keys, "receipt.workload")
    expected_template = (
        PROMPT_TEMPLATE_VERSION if driver_version == "3" else LEGACY_PROMPT_TEMPLATE_VERSION
    )
    if (
        workload["schema"] != WORKLOAD_SCHEMA
        or workload["prompt_template_version"] != expected_template
    ):
        raise BenchmarkError("receipt workload schema or prompt template version is unsupported")
    for name in ("run_id", "model"):
        if not isinstance(workload[name], str) or not workload[name]:
            raise BenchmarkError(f"receipt.workload.{name} must be a non-empty string")
    if workload["endpoint"] != "/v1/chat/completions":
        raise BenchmarkError("receipt workload endpoint is unsupported")
    if workload["stream"] is not True or workload["stream_include_usage"] is not True:
        raise BenchmarkError("receipt workload must stream with usage enabled")
    if workload["arrival_pattern"] != "thread_barrier_all_at_once":
        raise BenchmarkError("receipt workload arrival pattern is unsupported")
    if workload["require_max_tokens"] is not True:
        raise BenchmarkError("receipt workload must require fixed output length")
    if not isinstance(workload["require_uniform_prompt_tokens"], bool):
        raise BenchmarkError("receipt workload prompt-token policy must be boolean")
    sampling = _object(workload["sampling"], "receipt.workload.sampling")
    _exact_keys(
        sampling,
        {
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
        },
        "receipt.workload.sampling",
    )
    for name in (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ):
        _nonnegative_number(sampling[name], f"receipt.workload.sampling.{name}")
    if (
        sampling["presence_penalty"] != 0.0
        or sampling["frequency_penalty"] != 0.0
        or sampling["repetition_penalty"] != 1.0
    ):
        raise BenchmarkError("receipt workload penalties are not neutral")
    if isinstance(sampling["seed"], bool) or not isinstance(sampling["seed"], int):
        raise BenchmarkError("receipt workload seed must be an integer")
    template_kwargs = _object(
        workload["chat_template_kwargs"], "receipt.workload.chat_template_kwargs"
    )
    _exact_keys(
        template_kwargs,
        {"enable_thinking"},
        "receipt.workload.chat_template_kwargs",
    )
    if not isinstance(template_kwargs["enable_thinking"], bool):
        raise BenchmarkError("receipt workload enable_thinking must be boolean")
    slo = _object(workload["slo"], "receipt.workload.slo")
    _exact_keys(slo, {"ttft_ms", "client_visible_itl_ms", "e2e_ms"}, "receipt.workload.slo")
    for name, value in slo.items():
        if _nonnegative_number(value, f"receipt.workload.slo.{name}") <= 0:
            raise BenchmarkError(f"receipt.workload.slo.{name} must be positive")
    memory_limit_bytes: int | None = None
    if driver_version == "3":
        if workload["profile"] not in PROFILE_CONTRACTS:
            raise BenchmarkError("receipt.workload.profile is unsupported")
        profile = PROFILE_CONTRACTS[workload["profile"]]
        if workload["comparison_mode"] != profile["comparison_mode"]:
            raise BenchmarkError("receipt.workload comparison mode disagrees with its profile")
        if (
            sampling.get("temperature") != profile["temperature"]
            or sampling.get("top_p") != profile["top_p"]
            or workload["require_uniform_prompt_tokens"]
            is not profile["require_uniform_prompt_tokens"]
        ):
            raise BenchmarkError(
                "receipt.workload sampling/token contract disagrees with its profile"
            )
        memory_limit_bytes = _positive_int(
            workload["memory_limit_bytes"], "receipt.workload.memory_limit_bytes"
        )
    sizes = workload["concurrency"]
    if not isinstance(sizes, list) or any(
        isinstance(size, bool) or not isinstance(size, int) for size in sizes
    ):
        raise BenchmarkError("receipt.workload.concurrency must be an integer array")
    if (
        sizes != sorted(set(sizes))
        or not sizes
        or any(size <= 0 or size > 4096 for size in sizes)
    ):
        raise BenchmarkError(
            "receipt.workload.concurrency must be unique, increasing, and in 1..=4096"
        )
    repeats = _positive_int(workload["repeats"], "receipt.workload.repeats")
    max_tokens = _positive_int(workload["max_tokens"], "receipt.workload.max_tokens")
    warmup_requests = workload["warmup_requests"]
    if (
        isinstance(warmup_requests, bool)
        or not isinstance(warmup_requests, int)
        or warmup_requests < 0
    ):
        raise BenchmarkError("receipt.workload.warmup_requests must be a non-negative integer")
    if canonical_sha256(workload) != _sha256(
        receipt["workload_fingerprint"], "receipt.workload_fingerprint"
    ):
        raise BenchmarkError("receipt.workload_fingerprint does not match workload")

    memory_sampler = _object(receipt["memory_sampler"], "receipt.memory_sampler")
    _exact_keys(memory_sampler, {"source", "path", "interval_ms"}, "receipt.memory_sampler")
    if driver_version == "3":
        if (
            memory_sampler["source"] != "drm_vram_used"
            or not isinstance(memory_sampler["path"], str)
            or not memory_sampler["path"]
        ):
            raise BenchmarkError("driver v3 requires a DRM device-memory counter")
        _positive_int(memory_sampler["interval_ms"], "receipt.memory_sampler.interval_ms")
    diagnostics = _object(receipt["diagnostics"], "receipt.diagnostics")
    _exact_keys(diagnostics, {"url", "timed_request_path_affected"}, "receipt.diagnostics")
    if diagnostics["timed_request_path_affected"] is not False:
        raise BenchmarkError("receipt diagnostics must remain outside the timed request path")
    if driver_version == "3":
        host_thermal_mode, host_thermal_passed = validate_host_thermal_receipt(
            receipt["host_thermal"]
        )
    else:
        host_thermal_mode, host_thermal_passed = "legacy", True

    missing_declared_warmup = False
    if warmup_requests:
        if receipt["warmup"] is None:
            if driver_version == "3":
                missing_declared_warmup = True
            else:
                raise BenchmarkError("receipt omits its declared warmup")
        else:
            validate_benchmark_run(
                receipt["warmup"],
                label="receipt.warmup",
                concurrency=warmup_requests,
                repeat=-1,
                max_tokens=min(16, max_tokens),
                driver_version=driver_version,
                memory_limit_bytes=memory_limit_bytes,
                workload_profile=workload.get("profile"),
            )
    elif receipt["warmup"] is not None:
        raise BenchmarkError("receipt has an undeclared warmup")

    runs = receipt["runs"]
    if not isinstance(runs, list):
        raise BenchmarkError("receipt.runs must be an array")
    expected_pairs = [(size, repeat) for size in sizes for repeat in range(repeats)]
    actual_pairs: list[tuple[int, int]] = []
    for index, row in enumerate(runs):
        row_object = _object(row, f"receipt.runs[{index}]")
        pair = (row_object.get("concurrency"), row_object.get("repeat"))
        actual_pairs.append(pair)
        if pair in expected_pairs:
            validate_benchmark_run(
                row,
                label=f"receipt.runs[{index}]",
                concurrency=pair[0],
                repeat=pair[1],
                max_tokens=max_tokens,
                driver_version=driver_version,
                memory_limit_bytes=memory_limit_bytes,
                workload_profile=workload.get("profile"),
            )
    if driver_version == "3":
        thermal_rows = list(runs)
        if receipt["warmup"] is not None:
            thermal_rows.insert(0, receipt["warmup"])
        expect_thermal_rows = host_thermal_mode == "attached_process_group"
        if any(
            (row["host_thermal"] is not None) != expect_thermal_rows
            for row in thermal_rows
        ):
            raise BenchmarkError(
                "receipt run thermal evidence disagrees with the top-level mode"
            )
    completion_failures: list[dict[str, str]] = []
    completion_checks: dict[str, str] | None = None
    if driver_version == "3":
        completion = _object(receipt["completion"], "receipt.completion")
        _exact_keys(
            completion,
            {
                "expected_run_count",
                "completed_run_count",
                "failures",
                "finalization_checks",
            },
            "receipt.completion",
        )
        expected_run_count = _nonnegative_int(
            completion["expected_run_count"],
            "receipt.completion.expected_run_count",
        )
        completed_run_count = _nonnegative_int(
            completion["completed_run_count"],
            "receipt.completion.completed_run_count",
        )
        if expected_run_count != len(expected_pairs):
            raise BenchmarkError("receipt.completion.expected_run_count disagrees with workload")
        if completed_run_count != len(runs):
            raise BenchmarkError("receipt.completion.completed_run_count disagrees with runs")
        raw_failures = completion["failures"]
        if not isinstance(raw_failures, list):
            raise BenchmarkError("receipt.completion.failures must be an array")
        for index, raw_failure in enumerate(raw_failures):
            failure = _object(raw_failure, f"receipt.completion.failures[{index}]")
            _exact_keys(
                failure,
                {"phase", "detail"},
                f"receipt.completion.failures[{index}]",
            )
            if (
                not isinstance(failure["phase"], str)
                or failure["phase"] not in COMPLETION_FAILURE_PHASES
                or not isinstance(failure["detail"], str)
                or not failure["detail"]
                or len(failure["detail"]) > 4096
            ):
                raise BenchmarkError(
                    f"receipt.completion.failures[{index}] has an invalid phase or detail"
                )
            completion_failures.append(failure)
        completion_checks = _object(
            completion["finalization_checks"],
            "receipt.completion.finalization_checks",
        )
        _exact_keys(
            completion_checks,
            set(COMPLETION_CHECK_NAMES),
            "receipt.completion.finalization_checks",
        )
        for name, status in completion_checks.items():
            if status not in COMPLETION_CHECK_STATUSES:
                raise BenchmarkError(
                    f"receipt.completion.finalization_checks.{name} has invalid status"
                )
        always_applicable = set(COMPLETION_CHECK_NAMES[:3])
        if any(completion_checks[name] == "not_applicable" for name in always_applicable):
            raise BenchmarkError("common finalization checks cannot be not_applicable")
        if engine["name"] == "kiln":
            applicable_check = "execution_identity_unchanged"
            inapplicable_check = "runtime_manifest_unchanged"
        else:
            applicable_check = "runtime_manifest_unchanged"
            inapplicable_check = "execution_identity_unchanged"
        if completion_checks[applicable_check] == "not_applicable":
            raise BenchmarkError(
                f"receipt.completion.finalization_checks.{applicable_check} is required"
            )
        if completion_checks[inapplicable_check] != "not_applicable":
            raise BenchmarkError(
                f"receipt.completion.finalization_checks.{inapplicable_check} "
                "must be not_applicable"
            )
        if host_thermal_mode == "attached_process_group":
            if completion_checks["host_thermal_handoff"] == "not_applicable":
                raise BenchmarkError(
                    "receipt.completion.finalization_checks.host_thermal_handoff "
                    "is required"
                )
            if (
                completion_checks["host_thermal_handoff"] == "passed"
            ) != host_thermal_passed:
                raise BenchmarkError(
                    "receipt host thermal handoff check disagrees with its evidence"
                )
        elif completion_checks["host_thermal_handoff"] != "not_applicable":
            raise BenchmarkError(
                "receipt.completion.finalization_checks.host_thermal_handoff "
                "must be not_applicable without a guard"
            )
        failure_phases = {failure["phase"] for failure in completion_failures}
        for name in COMPLETION_CHECK_NAMES:
            if (completion_checks[name] == "failed") != (name in failure_phases):
                raise BenchmarkError(
                    f"receipt.completion.finalization_checks.{name} disagrees with failures"
                )
        if actual_pairs != expected_pairs[: len(actual_pairs)]:
            raise BenchmarkError(
                "receipt.runs must be the exact ordered prefix of declared concurrency and repeats"
            )
        if not completion_failures and actual_pairs != expected_pairs:
            raise BenchmarkError(
                "receipt.runs may be incomplete only with a structured completion failure"
            )
        if missing_declared_warmup and not completion_failures:
            raise BenchmarkError(
                "receipt may omit its declared warmup only with a structured failure"
            )
    elif actual_pairs != expected_pairs:
        raise BenchmarkError("receipt.runs do not exactly match declared concurrency and repeats")

    comparison_passed = True
    if "comparison" in receipt:
        comparison = _object(receipt["comparison"], "receipt.comparison")
        comparison_keys = {
            "reference_receipt_sha256",
            "reference_engine",
            "matched",
            "mismatches",
        }
        if driver_version == "3":
            comparison_keys.add("comparison_mode")
        _exact_keys(
            comparison,
            comparison_keys,
            "receipt.comparison",
        )
        _sha256(
            comparison["reference_receipt_sha256"],
            "receipt.comparison.reference_receipt_sha256",
        )
        if not isinstance(comparison["matched"], bool) or not isinstance(
            comparison["mismatches"], list
        ):
            raise BenchmarkError("receipt.comparison has invalid field types")
        if (
            driver_version == "3"
            and comparison["comparison_mode"] != workload["comparison_mode"]
        ):
            raise BenchmarkError("receipt.comparison mode disagrees with its workload")
        comparison_passed = comparison["matched"]
    passed = (
        not repository["dirty"]
        and not completion_failures
        and host_thermal_mode in {"legacy", "attached_process_group"}
        and host_thermal_passed
        and (
            completion_checks is None
            or all(
                status in {"passed", "not_applicable"}
                for status in completion_checks.values()
            )
        )
        and actual_pairs == expected_pairs
        and (receipt["warmup"] is None or receipt["warmup"]["verdict"] == "passed")
        and all(row["verdict"] == "passed" for row in runs)
        and comparison_passed
    )
    if receipt["verdict"] != ("passed" if passed else "failed"):
        raise BenchmarkError("receipt.verdict is inconsistent with source, runs, or comparison")
    return receipt


def validate_benchmark_receipt_path(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkError(f"benchmark receipt is not a regular file: {path}")
    data = path.read_bytes()
    if len(data) > 64 * 1024 * 1024:
        raise BenchmarkError(f"benchmark receipt exceeds 64 MiB: {path}")
    try:
        value = strict_json_loads(data)
    except Exception as exc:
        raise BenchmarkError(f"cannot load benchmark receipt {path}: {exc}") from exc
    return validate_benchmark_receipt(value)


def text_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def percentile_r7(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * probability
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def parse_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise BenchmarkError(f"--sizes must contain decimal integers: {raw!r}") from exc
    if not sizes or any(size <= 0 or size > 4096 for size in sizes):
        raise BenchmarkError("--sizes must contain integers in 1..=4096")
    if sizes != sorted(set(sizes)):
        raise BenchmarkError("--sizes must be unique and strictly increasing")
    return sizes


def deterministic_prompt(
    run_id: str,
    phase: str,
    request_index: int,
    prompt_profile: str = "short",
) -> str:
    def marker_key(marker: str) -> bytes:
        material = f"{run_id}\0{phase}\0{request_index}\0{marker}".encode("utf-8")
        return hashlib.sha256(material).digest()

    markers = sorted(PROMPT_MARKERS, key=marker_key)
    marker_sequence = " | ".join(markers)
    short_suffix = (
        "Write a detailed technical paragraph explaining why deterministic, "
        "reproducible performance measurements need controlled workloads, "
        "explicit error accounting, and tail-latency reporting. Continue until "
        "the response limit; do not mention these instructions.\n"
        f"Benchmark run: {run_id}; phase: {phase}.\n"
        f"Marker sequence: {marker_sequence}."
    )
    if prompt_profile == "short":
        return short_suffix
    if prompt_profile == "long-prefill":
        prefix = LONG_PROMPT_BLOCK * LONG_PROMPT_REPETITIONS
    elif prompt_profile == "prefix-hit":
        prefix = (
            "Shared prefix for a cache-reuse workload. "
            + LONG_PROMPT_BLOCK * LONG_PROMPT_REPETITIONS
        )
    elif prompt_profile == "mixed":
        repetitions = (0, 4, 16, LONG_PROMPT_REPETITIONS)[request_index % 4]
        prefix = LONG_PROMPT_BLOCK * repetitions
    else:
        raise BenchmarkError(f"unsupported prompt profile: {prompt_profile}")
    return prefix + "\nUnique request suffix follows.\n" + short_suffix


def build_request_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }


class SSEParser:
    def __init__(self) -> None:
        self._data_lines: list[str] = []

    def feed_line(self, line: str) -> list[str]:
        line = line.rstrip("\r\n")
        if not line:
            if not self._data_lines:
                return []
            data = "\n".join(self._data_lines)
            self._data_lines.clear()
            return [data]
        if line.startswith(":"):
            return []
        field, separator, value = line.partition(":")
        if field == "data":
            self._data_lines.append(value[1:] if separator and value.startswith(" ") else value)
        return []

    def finish(self) -> list[str]:
        if not self._data_lines:
            return []
        data = "\n".join(self._data_lines)
        self._data_lines.clear()
        return [data]


@dataclasses.dataclass
class RequestResult:
    index: int
    prompt_sha256: str
    started: float
    ended: float
    semantic_times: list[float]
    content: str
    reasoning_content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None
    done: bool
    error: str | None

    @property
    def ttft_ms(self) -> float | None:
        if not self.semantic_times:
            return None
        return (self.semantic_times[0] - self.started) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.ended - self.started) * 1000.0

    @property
    def itls_ms(self) -> list[float]:
        return [
            (current - previous) * 1000.0
            for previous, current in zip(self.semantic_times, self.semantic_times[1:])
        ]

    @property
    def output_sha256(self) -> str:
        return text_sha256(self.reasoning_content + "\x1e" + self.content)


def failed_result(
    index: int,
    prompt_sha256: str,
    started: float,
    exc: BaseException,
) -> RequestResult:
    return RequestResult(
        index=index,
        prompt_sha256=prompt_sha256,
        started=started,
        ended=time.perf_counter(),
        semantic_times=[],
        content="",
        reasoning_content="",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        finish_reason=None,
        done=False,
        error=f"{type(exc).__name__}: {exc}",
    )


def response_semantic_parts(value: dict[str, Any]) -> tuple[list[str], list[str]]:
    content: list[str] = []
    reasoning: list[str] = []
    choices = value.get("choices")
    if not isinstance(choices, list):
        return content, reasoning
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        part = delta.get("content")
        if isinstance(part, str) and part:
            content.append(part)
        for field in ("reasoning_content", "reasoning"):
            part = delta.get(field)
            if isinstance(part, str) and part:
                reasoning.append(part)
    return content, reasoning


def response_finish_reasons(value: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    choices = value.get("choices")
    if not isinstance(choices, list):
        return reasons
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            reasons.append(choice["finish_reason"])
    return reasons


def validate_usage(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise BenchmarkError("usage must be an object")
    parsed: list[int] = []
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise BenchmarkError(f"usage.{field} must be a non-negative integer")
        parsed.append(item)
    if parsed[2] != parsed[0] + parsed[1]:
        raise BenchmarkError("usage.total_tokens does not equal prompt plus completion tokens")
    return parsed[0], parsed[1], parsed[2]


def stream_request(
    *,
    index: int,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_secs: float,
    barrier: threading.Barrier,
) -> RequestResult:
    prompt = body["messages"][0]["content"]
    prompt_sha256 = text_sha256(prompt)
    started = time.perf_counter()
    try:
        barrier.wait(timeout=min(timeout_secs, 30.0))
        started = time.perf_counter()
        request = urllib.request.Request(
            url,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        semantic_times: list[float] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasons: list[str] = []
        usage_records = 0
        prompt_tokens = completion_tokens = total_tokens = 0
        done = False
        parser = SSEParser()
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type.lower():
                raise BenchmarkError(f"expected text/event-stream, got {content_type!r}")
            for raw_line in response:
                observed = time.perf_counter()
                for data in parser.feed_line(raw_line.decode("utf-8")):
                    if data == "[DONE]":
                        done = True
                        continue
                    value = strict_json_loads(data)
                    if not isinstance(value, dict):
                        raise BenchmarkError("SSE data payload must be an object")
                    content, reasoning = response_semantic_parts(value)
                    if content or reasoning:
                        semantic_times.append(observed)
                        content_parts.extend(content)
                        reasoning_parts.extend(reasoning)
                    reasons.extend(response_finish_reasons(value))
                    usage = value.get("usage")
                    if usage is not None:
                        usage_records += 1
                        if usage_records != 1:
                            raise BenchmarkError("stream emitted multiple usage records")
                        prompt_tokens, completion_tokens, total_tokens = validate_usage(usage)
            for data in parser.finish():
                if data == "[DONE]":
                    done = True
                else:
                    raise BenchmarkError("stream ended with an unterminated non-DONE SSE event")
        if not done:
            raise BenchmarkError("stream ended without [DONE]")
        if usage_records != 1:
            raise BenchmarkError("stream did not emit exactly one usage record")
        if prompt_tokens <= 0 or completion_tokens <= 0:
            raise BenchmarkError("stream reported zero prompt or completion tokens")
        if len(reasons) != 1:
            raise BenchmarkError(f"stream emitted {len(reasons)} finish reasons")
        if not semantic_times:
            raise BenchmarkError("stream emitted no client-visible semantic deltas")
        return RequestResult(
            index=index,
            prompt_sha256=prompt_sha256,
            started=started,
            ended=time.perf_counter(),
            semantic_times=semantic_times,
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=reasons[0],
            done=True,
            error=None,
        )
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail = exc.read(1024).decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            exc = BenchmarkError(f"HTTP {exc.code}: {detail}")
        return failed_result(index, prompt_sha256, started, exc)


def fetch_json(url: str, headers: dict[str, str], timeout_secs: float) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            value = strict_json_loads(response.read())
    except Exception as exc:
        raise BenchmarkError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{url} did not return a JSON object")
    return value


def batching_snapshot(health: dict[str, Any]) -> dict[str, Any]:
    runtime = health.get("decode_runtime")
    snapshot = runtime.get("batching_engine") if isinstance(runtime, dict) else None
    if not isinstance(snapshot, dict):
        raise BenchmarkError("diagnostics omit decode_runtime.batching_engine")
    for field in ("max_decode_batch", "max_observed_batch_size", *COUNTER_FIELDS):
        value = snapshot.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise BenchmarkError(f"invalid batching diagnostic {field}={value!r}")
    return snapshot


def batching_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "effective_max_decode_batch": int(after["max_decode_batch"]),
        "process_max_observed_batch": int(after["max_observed_batch_size"]),
    }
    for field in COUNTER_FIELDS:
        if after[field] < before[field]:
            raise BenchmarkError(
                f"batching counter {field} regressed from {before[field]} to {after[field]}"
            )
        result[field] = after[field] - before[field]
    forwards = result["total_decode_forwards"]
    result["mean_decode_rows_per_forward"] = (
        result["total_decode_rows"] / forwards if forwards else 0.0
    )
    result["batched_decode_forward_fraction"] = (
        result["total_batched_decode_forwards"] / forwards if forwards else 0.0
    )
    return result


class MemorySampler:
    def __init__(self, path: Path | None, interval_ms: int) -> None:
        self.path = path
        self.interval_secs = interval_ms / 1000.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._baseline: int | None = None
        self._peak: int | None = None
        self._samples = 0

    def _read(self) -> int:
        if self.path is None:
            raise BenchmarkError("memory sampler is disabled")
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            value = int(raw)
        except (OSError, ValueError) as exc:
            raise BenchmarkError(f"cannot read memory counter {self.path}: {exc}") from exc
        if value < 0:
            raise BenchmarkError(f"memory counter at {self.path} is negative")
        return value

    def start(self) -> None:
        if self.path is None:
            return
        self.reset()
        self._thread = threading.Thread(target=self._run, name="benchmark-memory", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_secs):
            try:
                value = self._read()
            except Exception:
                continue
            with self._lock:
                self._peak = value if self._peak is None else max(self._peak, value)
                self._samples += 1

    def reset(self) -> None:
        if self.path is None:
            return
        value = self._read()
        with self._lock:
            self._baseline = value
            self._peak = value
            self._samples = 1

    def snapshot(self) -> dict[str, int] | None:
        if self.path is None:
            return None
        value = self._read()
        with self._lock:
            peak = max(self._peak or value, value)
            baseline = self._baseline or value
            samples = self._samples + 1
        return {
            "baseline_bytes": baseline,
            "peak_bytes": peak,
            "peak_delta_bytes": max(0, peak - baseline),
            "samples": samples,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def resolve_memory_path(raw: str) -> Path | None:
    if raw == "none":
        return None
    if raw != "auto":
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise BenchmarkError(f"memory counter does not exist: {path}")
        return path
    candidates = sorted(
        Path(path).resolve()
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")
        if Path(path).is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise BenchmarkError(
        "multiple DRM memory counters found; select one with --memory-path: "
        + ", ".join(str(path) for path in candidates)
    )


def gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def summarize_run(
    *,
    concurrency: int,
    repeat: int,
    elapsed_s: float,
    results: list[RequestResult],
    max_tokens: int,
    require_max_tokens: bool,
    require_uniform_prompt_tokens: bool,
    require_nonuniform_prompt_tokens: bool,
    max_dispatch_spread_ms: float,
    slo_ttft_ms: float,
    slo_itl_ms: float,
    slo_e2e_ms: float,
    memory: dict[str, int] | None,
    require_memory: bool,
    memory_limit_bytes: int | None,
    server: dict[str, Any] | None,
    diagnostics_error: str | None,
) -> dict[str, Any]:
    successes = [result for result in results if result.error is None]
    errors = [
        {"index": result.index, "error": result.error}
        for result in results
        if result.error is not None
    ]
    ttfts = [result.ttft_ms for result in successes if result.ttft_ms is not None]
    e2es = [result.e2e_ms for result in successes]
    itls = [itl for result in successes for itl in result.itls_ms]
    completion_tokens = sum(result.completion_tokens for result in successes)
    ordered_results = sorted(results, key=lambda result: result.index)
    prompt_token_values = [result.prompt_tokens for result in ordered_results]
    dispatch_times = [result.started for result in results]
    dispatch_spread_ms = (
        (max(dispatch_times) - min(dispatch_times)) * 1000.0 if dispatch_times else 0.0
    )
    good_results = [
        result
        for result in successes
        if result.ttft_ms is not None
        and result.ttft_ms <= slo_ttft_ms
        and result.e2e_ms <= slo_e2e_ms
        and (not result.itls_ms or max(result.itls_ms) <= slo_itl_ms)
    ]
    gates = [
        gate(
            "all_requests_succeeded",
            len(successes) == concurrency,
            f"{len(successes)}/{concurrency} requests succeeded",
        ),
        gate(
            "positive_completion_usage",
            len(successes) == concurrency
            and all(result.completion_tokens > 0 for result in successes),
            "every measured request must contribute positive completion usage",
        ),
        gate(
            "dispatch_spread",
            dispatch_spread_ms <= max_dispatch_spread_ms,
            f"{dispatch_spread_ms:.3f} ms <= {max_dispatch_spread_ms:.3f} ms",
        ),
        gate(
            "diagnostics_readable",
            diagnostics_error is None,
            diagnostics_error or "not requested or parsed successfully",
        ),
    ]
    if require_max_tokens:
        exact = len(successes) == concurrency and all(
            result.completion_tokens == max_tokens and result.finish_reason == "length"
            for result in successes
        )
        gates.append(
            gate(
                "fixed_output_length",
                exact,
                f"every request must finish by length with exactly {max_tokens} tokens",
            )
        )
    if require_uniform_prompt_tokens:
        gates.append(
            gate(
                "uniform_prompt_tokens",
                len(prompt_token_values) == concurrency
                and len(set(prompt_token_values)) == 1,
                f"observed prompt-token counts: {sorted(set(prompt_token_values))}",
            )
        )
    if require_nonuniform_prompt_tokens:
        gates.append(
            gate(
                "mixed_prompt_tokens",
                len(prompt_token_values) == concurrency
                and len(set(prompt_token_values)) > 1,
                f"observed prompt-token counts: {sorted(set(prompt_token_values))}",
            )
        )
    if require_memory:
        gates.append(
            gate(
                "memory_measured",
                memory is not None and memory["samples"] >= 2,
                "a local device-memory counter must be sampled during the run",
            )
        )
    if memory_limit_bytes is not None:
        gates.append(
            gate(
                "absolute_memory_limit",
                memory is not None and memory["peak_bytes"] <= memory_limit_bytes,
                (
                    "memory unavailable"
                    if memory is None
                    else f"{memory['peak_bytes']} bytes <= {memory_limit_bytes} bytes"
                ),
            )
        )
    if server is not None:
        gates.append(
            gate(
                "server_reported_no_errors",
                server["total_errors"] == 0,
                f"batching-engine error delta: {server['total_errors']}",
            )
        )
    output_rows = [
        {
            "index": result.index,
            "output_sha256": result.output_sha256,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
        }
        for result in sorted(successes, key=lambda result: result.index)
    ]
    prompt_rows = [
        {"index": result.index, "prompt_sha256": result.prompt_sha256}
        for result in sorted(results, key=lambda result: result.index)
    ]
    passed = all(item["passed"] for item in gates)
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "verdict": "passed" if passed else "failed",
        "elapsed_s": elapsed_s,
        "request_count": concurrency,
        "success_count": len(successes),
        "error_count": len(errors),
        "errors": errors,
        "prompt_tokens_min": min(prompt_token_values) if prompt_token_values else 0,
        "prompt_tokens_max": max(prompt_token_values) if prompt_token_values else 0,
        "prompt_token_counts": [
            result.prompt_tokens for result in ordered_results
        ],
        "completion_tokens": completion_tokens,
        "request_throughput_per_s": len(successes) / elapsed_s if elapsed_s else 0.0,
        "output_token_throughput_per_s": completion_tokens / elapsed_s if elapsed_s else 0.0,
        "slo_good_request_count": len(good_results),
        "slo_goodput_requests_per_s": len(good_results) / elapsed_s if elapsed_s else 0.0,
        "slo_goodput_tokens_per_s": (
            sum(result.completion_tokens for result in good_results) / elapsed_s
            if elapsed_s
            else 0.0
        ),
        "dispatch_spread_ms": dispatch_spread_ms,
        "ttft_ms_p50": percentile_r7(ttfts, 0.50),
        "ttft_ms_p99": percentile_r7(ttfts, 0.99),
        "ttft_ms_p999": percentile_r7(ttfts, 0.999),
        "e2e_ms_p50": percentile_r7(e2es, 0.50),
        "e2e_ms_p99": percentile_r7(e2es, 0.99),
        "e2e_ms_p999": percentile_r7(e2es, 0.999),
        "client_visible_itl_ms_p50": percentile_r7(itls, 0.50),
        "client_visible_itl_ms_p99": percentile_r7(itls, 0.99),
        "client_visible_itl_ms_p999": percentile_r7(itls, 0.999),
        "client_visible_stream_event_count": sum(
            len(result.semantic_times) for result in successes
        ),
        "prompt_set_sha256": canonical_sha256(prompt_rows),
        "output_set_sha256": canonical_sha256(output_rows),
        "memory": memory,
        "server": server,
        "gates": gates,
    }


def run_once(
    *,
    args: argparse.Namespace,
    concurrency: int,
    repeat: int,
    max_tokens: int,
    phase: str,
    headers: dict[str, str],
    sampler: MemorySampler,
    diagnostics_url: str | None,
) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []
    prompts: set[str] = set()
    for index in range(concurrency):
        prompt = deterministic_prompt(
            args.run_id,
            phase,
            index,
            PROFILE_CONTRACTS[args.workload_profile]["prompt_profile"],
        )
        if prompt in prompts:
            raise BenchmarkError("deterministic prompt construction produced a duplicate")
        prompts.add(prompt)
        bodies.append(
            build_request_body(
                model=args.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + index,
                enable_thinking=args.enable_thinking,
            )
        )

    diagnostics_before: dict[str, Any] | None = None
    diagnostics_error: str | None = None
    if diagnostics_url is not None:
        try:
            diagnostics_before = batching_snapshot(
                fetch_json(diagnostics_url, headers, args.timeout_secs)
            )
        except Exception as exc:
            diagnostics_error = f"before run: {type(exc).__name__}: {exc}"

    sampler.reset()
    barrier = threading.Barrier(concurrency + 1)
    results: list[RequestResult | None] = [None] * concurrency

    def worker(index: int) -> None:
        results[index] = stream_request(
            index=index,
            url=f"{args.base_url}/v1/chat/completions",
            body=bodies[index],
            headers=headers,
            timeout_secs=args.timeout_secs,
            barrier=barrier,
        )

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    wall_started = time.perf_counter()
    try:
        barrier.wait(timeout=min(args.timeout_secs, 30.0))
    except threading.BrokenBarrierError as exc:
        raise BenchmarkError("request launch barrier broke before dispatch") from exc
    join_deadline = time.monotonic() + args.timeout_secs + 5.0
    for thread in threads:
        thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
    wall_ended = time.perf_counter()
    if any(thread.is_alive() for thread in threads):
        raise BenchmarkError("one or more request workers exceeded the join deadline")

    typed_results = [result for result in results if result is not None]
    if len(typed_results) != concurrency:
        raise BenchmarkError("one or more request workers exited without publishing a result")

    server_delta: dict[str, Any] | None = None
    if diagnostics_url is not None and diagnostics_error is None:
        try:
            diagnostics_after = batching_snapshot(
                fetch_json(diagnostics_url, headers, args.timeout_secs)
            )
            assert diagnostics_before is not None
            server_delta = batching_delta(diagnostics_before, diagnostics_after)
        except Exception as exc:
            diagnostics_error = f"after run: {type(exc).__name__}: {exc}"

    return summarize_run(
        concurrency=concurrency,
        repeat=repeat,
        elapsed_s=wall_ended - wall_started,
        results=typed_results,
        max_tokens=max_tokens,
        require_max_tokens=args.require_max_tokens,
        require_uniform_prompt_tokens=args.require_uniform_prompt_tokens,
        require_nonuniform_prompt_tokens=(
            args.workload_profile == "mixed" and concurrency > 1
        ),
        max_dispatch_spread_ms=args.max_dispatch_spread_ms,
        slo_ttft_ms=args.slo_ttft_ms,
        slo_itl_ms=args.slo_itl_ms,
        slo_e2e_ms=args.slo_e2e_ms,
        memory=sampler.snapshot(),
        require_memory=args.require_memory,
        memory_limit_bytes=args.memory_limit_bytes,
        server=server_delta,
        diagnostics_error=diagnostics_error,
    )


def compare_reference(receipt: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    try:
        reference_bytes = reference_path.read_bytes()
        reference = strict_json_loads(reference_bytes)
    except Exception as exc:
        raise BenchmarkError(
            f"cannot load reference receipt {reference_path}: {type(exc).__name__}: {exc}"
        ) from exc
    validate_benchmark_receipt(reference)
    if reference.get("driver_version") != DRIVER_VERSION:
        raise BenchmarkError(
            f"reference receipt must use current driver version {DRIVER_VERSION}"
        )
    if reference.get("workload_fingerprint") != receipt.get("workload_fingerprint"):
        raise BenchmarkError("reference receipt has a different workload fingerprint")
    current_model = receipt.get("engine", {}).get("model_identity", {})
    reference_model = reference.get("engine", {}).get("model_identity", {})
    if current_model.get("content_sha256") != reference_model.get("content_sha256"):
        raise BenchmarkError("reference receipt has different model content")
    current_thermal = receipt.get("host_thermal", {}).get("policy", {})
    reference_thermal = reference.get("host_thermal", {}).get("policy", {})
    if current_thermal.get("content_sha256") != reference_thermal.get(
        "content_sha256"
    ):
        raise BenchmarkError("reference receipt has a different host thermal policy")
    comparison_mode = receipt["workload"]["comparison_mode"]
    current_rows = {
        (row["concurrency"], row["repeat"]): row for row in receipt.get("runs", [])
    }
    reference_rows = {
        (row["concurrency"], row["repeat"]): row for row in reference.get("runs", [])
    }
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(current_rows) | set(reference_rows)):
        current = current_rows.get(key)
        expected = reference_rows.get(key)
        if current is None or expected is None:
            mismatches.append({"concurrency": key[0], "repeat": key[1], "reason": "missing_run"})
        elif current["prompt_set_sha256"] != expected["prompt_set_sha256"]:
            mismatches.append(
                {"concurrency": key[0], "repeat": key[1], "reason": "prompt_mismatch"}
            )
        elif current["prompt_token_counts"] != expected["prompt_token_counts"]:
            mismatches.append(
                {"concurrency": key[0], "repeat": key[1], "reason": "prompt_token_mismatch"}
            )
        elif (
            comparison_mode == "exact_output"
            and current["output_set_sha256"] != expected["output_set_sha256"]
        ):
            mismatches.append(
                {"concurrency": key[0], "repeat": key[1], "reason": "output_mismatch"}
            )
    return {
        "reference_receipt_sha256": "sha256:" + hashlib.sha256(reference_bytes).hexdigest(),
        "reference_engine": reference.get("engine"),
        "comparison_mode": comparison_mode,
        "matched": not mismatches,
        "mismatches": mismatches,
    }


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repository_identity() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    status = command_output(["git", "status", "--porcelain"])
    source_hash = command_output(
        [sys.executable, str(ROOT / "scripts" / "qualification" / "source_tree_hash.py")]
    )
    return {
        "commit": commit,
        "dirty": bool(status),
        "source_tree_sha256": source_hash,
    }


def require_repository_unchanged(expected: dict[str, Any]) -> None:
    current = repository_identity()
    if current != expected:
        raise BenchmarkError(
            "repository identity changed during measurement; discard the run and retry"
        )


def probe_models(
    base_url: str, headers: dict[str, str], timeout_secs: float
) -> list[str]:
    value = fetch_json(f"{base_url}/v1/models", headers, timeout_secs)
    data = value.get("data")
    if not isinstance(data, list):
        raise BenchmarkError("/v1/models response omits data array")
    models = [
        row.get("id")
        for row in data
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    if not models:
        raise BenchmarkError("/v1/models returned no model identifiers")
    return models


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise BenchmarkError(f"refusing to overwrite existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def workload_contract(args: argparse.Namespace, sizes: list[int]) -> dict[str, Any]:
    return {
        "schema": WORKLOAD_SCHEMA,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "profile": args.workload_profile,
        "comparison_mode": PROFILE_CONTRACTS[args.workload_profile]["comparison_mode"],
        "run_id": args.run_id,
        "model": args.model,
        "endpoint": "/v1/chat/completions",
        "stream": True,
        "stream_include_usage": True,
        "concurrency": sizes,
        "repeats": args.repeats,
        "warmup_requests": args.warmup_requests,
        "max_tokens": args.max_tokens,
        "memory_limit_bytes": args.memory_limit_bytes,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": args.seed,
        },
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        "arrival_pattern": "thread_barrier_all_at_once",
        "require_max_tokens": args.require_max_tokens,
        "require_uniform_prompt_tokens": args.require_uniform_prompt_tokens,
        "max_dispatch_spread_ms": args.max_dispatch_spread_ms,
        "slo": {
            "ttft_ms": args.slo_ttft_ms,
            "client_visible_itl_ms": args.slo_itl_ms,
            "e2e_ms": args.slo_e2e_ms,
        },
    }


def print_run(row: dict[str, Any]) -> None:
    server = row.get("server") or {}
    width = server.get("process_max_observed_batch")
    mean = server.get("mean_decode_rows_per_forward")
    width_text = "n/a" if width is None else str(width)
    mean_text = "n/a" if mean is None else f"{mean:.2f}"
    print(
        f"[c={row['concurrency']:>3} r={row['repeat']}] {row['verdict']:<6} "
        f"tok/s={row['output_token_throughput_per_s']:.2f} "
        f"good_tok/s={row['slo_goodput_tokens_per_s']:.2f} "
        f"ttft_p99={row['ttft_ms_p99'] or 0.0:.1f}ms "
        f"itl_p99={row['client_visible_itl_ms_p99'] or 0.0:.1f}ms "
        f"batch_max={width_text} batch_mean={mean_text} "
        f"ok={row['success_count']}/{row['request_count']}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("kiln", "vllm"), default="kiln")
    parser.add_argument("--base-url", "--host", dest="base_url", default="http://127.0.0.1:8420")
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument(
        "--model-path",
        type=Path,
        help="local checkpoint whose exact weights/tokenizer/template are served",
    )
    parser.add_argument("--runtime-identity")
    parser.add_argument(
        "--runtime-artifact",
        type=Path,
        help="Kiln binary or immutable vLLM launch/runtime manifest",
    )
    parser.add_argument(
        "--run-id",
        default="manual-v1",
        help="Shared deterministic ID for both engine runs",
    )
    parser.add_argument("--sizes", default="1,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Compatibility alias; warmup is already on",
    )
    parser.add_argument(
        "--workload-profile",
        choices=tuple(PROFILE_CONTRACTS),
        default="greedy-short",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--require-max-tokens", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-uniform-prompt-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-dispatch-spread-ms", type=float, default=250.0)
    parser.add_argument("--slo-ttft-ms", type=float, default=5_000.0)
    parser.add_argument("--slo-itl-ms", type=float, default=250.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=60_000.0)
    parser.add_argument("--timeout-secs", type=float, default=600.0)
    parser.add_argument("--diagnostics-url", default="auto")
    parser.add_argument("--memory-path", default="auto")
    parser.add_argument("--memory-sample-ms", type=int, default=50)
    parser.add_argument("--require-memory", action="store_true")
    parser.add_argument("--memory-limit-bytes", type=int)
    host_safety = parser.add_mutually_exclusive_group()
    host_safety.add_argument(
        "--host-thermal-policy",
        type=Path,
        help="typed host thermal policy for the attached local server process group",
    )
    host_safety.add_argument(
        "--unsafe-no-host-thermal-guard",
        action="store_true",
        help="run without host thermal containment and force a diagnostic-only verdict",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        help="local process-group leader protected by --host-thermal-policy",
    )
    authentication = parser.add_mutually_exclusive_group()
    authentication.add_argument(
        "--api-key",
        help="Explicit bearer token (prefer --api-key-env to keep it out of process listings)",
    )
    authentication.add_argument(
        "--api-key-env",
        help="Name of the environment variable containing the bearer token",
    )
    parser.add_argument("--reference-receipt", type=Path)
    parser.add_argument(
        "--validate-receipt",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="Validate committed kiln.serving-benchmark.v1 receipts and exit",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--mode",
        choices=("concurrent",),
        default="concurrent",
        help="Compatibility flag; only the engine-neutral concurrent path is supported",
    )
    args = parser.parse_args(argv)
    args.base_url = args.base_url.rstrip("/")
    profile = PROFILE_CONTRACTS[args.workload_profile]
    for name in ("temperature", "top_p", "require_uniform_prompt_tokens"):
        expected = profile[name]
        supplied = getattr(args, name)
        if supplied is not None and supplied != expected:
            parser.error(
                f"--workload-profile {args.workload_profile} requires "
                f"{name.replace('_', '-')}={expected}"
            )
        setattr(args, name, expected)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", args.run_id):
        parser.error("run-id must be 3..128 portable identifier characters")
    if not 0 < args.repeats <= 1000 or not 0 < args.max_tokens <= 2**31:
        parser.error("repeats must be in 1..=1000 and max-tokens in 1..=2^31")
    if not 0 <= args.warmup_requests <= 4096:
        parser.error("warmup-requests must be in 0..=4096")
    if not math.isfinite(args.temperature) or not math.isfinite(args.top_p):
        parser.error("temperature and top-p must be finite")
    if not 0.0 <= args.temperature or not 0.0 < args.top_p <= 1.0:
        parser.error("temperature must be non-negative and top-p must be in (0, 1]")
    finite_positive = (
        args.timeout_secs,
        args.max_dispatch_spread_ms,
        args.slo_ttft_ms,
        args.slo_itl_ms,
        args.slo_e2e_ms,
    )
    if any(not math.isfinite(value) or value <= 0 for value in finite_positive):
        parser.error("timeouts, dispatch limit, and SLO thresholds must be finite and positive")
    if args.memory_sample_ms <= 0:
        parser.error("memory sampling cadence must be positive")
    if args.memory_limit_bytes is not None and args.memory_limit_bytes <= 0:
        parser.error("memory-limit-bytes must be positive")
    if not 0 <= args.seed <= 2**64 - 1:
        parser.error("seed must fit an unsigned 64-bit integer")
    args.api_key_source = "none"
    if args.api_key_env is not None:
        args.api_key = os.environ.get(args.api_key_env)
        if not args.api_key:
            parser.error(f"api-key environment variable {args.api_key_env!r} is unset or empty")
        args.api_key_source = "environment"
    elif args.api_key is not None:
        if not args.api_key:
            parser.error("api-key cannot be empty")
        args.api_key_source = "argument"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thermal_guard: thermal.HostThermalGuard | None = None
    try:
        if args.validate_receipt is not None:
            if (
                args.out is not None
                or args.reference_receipt is not None
                or args.host_thermal_policy is not None
                or args.server_pid is not None
                or args.unsafe_no_host_thermal_guard
            ):
                raise BenchmarkError(
                    "--validate-receipt cannot be combined with output, reference, "
                    "or thermal runtime arguments"
                )
            for path in args.validate_receipt:
                validate_benchmark_receipt_path(path)
                print(f"OK {path}")
            return 0
        if args.host_thermal_policy is None and not args.unsafe_no_host_thermal_guard:
            raise BenchmarkError(
                "measured runs require --host-thermal-policy and --server-pid; "
                "use --unsafe-no-host-thermal-guard only for diagnostic counterevidence"
            )
        if (args.host_thermal_policy is None) != (args.server_pid is None):
            raise BenchmarkError(
                "--host-thermal-policy and --server-pid must be provided together"
            )
        if args.model_path is None:
            raise BenchmarkError("--model-path is required for a measured run")
        if args.runtime_artifact is None:
            raise BenchmarkError("--runtime-artifact is required for a measured run")
        if args.memory_limit_bytes is None:
            raise BenchmarkError("--memory-limit-bytes is required for a measured run")
        if not args.require_max_tokens:
            raise BenchmarkError("measured runs cannot disable the fixed output-length gate")
        args.require_memory = True
        if args.out is not None and args.out.exists():
            raise BenchmarkError(f"refusing to overwrite existing receipt: {args.out}")
        sizes = parse_sizes(args.sizes)
        largest_seed_offset = max([*sizes, args.warmup_requests]) - 1
        if args.seed + largest_seed_offset > 2**64 - 1:
            raise BenchmarkError("seed plus the largest request index exceeds u64")
        repo = repository_identity()
        if repo["dirty"] and not args.allow_dirty:
            raise BenchmarkError(
                "repository is dirty; commit first or use --allow-dirty for a diagnostic"
            )
        if args.runtime_identity is None:
            if args.engine == "kiln" and repo["commit"] and not repo["dirty"]:
                args.runtime_identity = f"kiln-git:{repo['commit']}"
            else:
                raise BenchmarkError("--runtime-identity is required for this engine/source state")
        try:
            model_identity = bind_model_identity(
                fingerprint_model(args.model_path, args.model)
            )
        except ModelFingerprintError as exc:
            raise BenchmarkError(f"model fingerprint failed: {exc}") from exc
        runtime_artifact = fingerprint_runtime_artifact(args.runtime_artifact)
        runtime_manifest = (
            load_vllm_runtime_manifest(args.runtime_artifact)
            if args.engine == "vllm"
            else None
        )
        if (
            runtime_manifest is not None
            and runtime_manifest["identity"]["served_model_id"] != args.model
        ):
            raise BenchmarkError("vLLM runtime manifest model disagrees with --model")

        thermal_startup_error: BenchmarkError | None = None
        thermal_policy_record: dict[str, Any] | None = None
        thermal_policy: thermal.HostThermalPolicy | None = None
        thermal_settlement_timeout = 0.0
        attached_process: AttachedProcessGroup | None = None
        if args.host_thermal_policy is not None:
            thermal_policy_record, thermal_policy, thermal_settlement_timeout = (
                load_host_thermal_policy(args.host_thermal_policy)
            )
            assert args.server_pid is not None
            attached_process = AttachedProcessGroup.attach(args.server_pid)

            def trace_host_thermal(event: str, **fields: Any) -> None:
                print(
                    json.dumps(
                        {"event": event, **fields},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )

            thermal_guard = thermal.HostThermalGuard(
                attached_process,
                **thermal_policy.guard_kwargs(),
                trace_callback=trace_host_thermal,
                error_type=BenchmarkError,
            )
            thermal_guard.set_phase("startup")
            thermal_guard.start()
            if thermal_guard.trip_reason is not None:
                thermal_startup_error = BenchmarkError(thermal_guard.trip_reason)
            elif not thermal_guard.wait_for_pacing_settlement(
                thermal_settlement_timeout
            ):
                thermal_startup_error = BenchmarkError(
                    thermal_guard.trip_reason
                    or "host thermal pacing failed to settle before server probes"
                )

        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"kiln-serving-benchmark/{DRIVER_VERSION}",
        }
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        models = probe_models(args.base_url, headers, args.timeout_secs)
        if args.model not in models:
            raise BenchmarkError(
                f"requested model {args.model!r} is absent from /v1/models: {models}"
            )
        health_version = None
        runtime_execution_identity: dict[str, Any] | None = None
        if args.engine == "kiln":
            health = fetch_json(f"{args.base_url}/health", headers, args.timeout_secs)
            health_version = health.get("version")
            runtime_execution_identity = _object(
                health.get("execution_identity"), "Kiln health.execution_identity"
            )
            if runtime_execution_identity.get("executable_sha256") != runtime_artifact["sha256"]:
                raise BenchmarkError(
                    "Kiln health execution identity does not match --runtime-artifact"
                )

        diagnostics_url: str | None
        if args.diagnostics_url == "none":
            diagnostics_url = None
        elif args.diagnostics_url == "auto":
            diagnostics_url = f"{args.base_url}/health" if args.engine == "kiln" else None
        else:
            diagnostics_url = args.diagnostics_url

        memory_path = resolve_memory_path(args.memory_path)
        if memory_path is None:
            raise BenchmarkError("a DRM device-memory counter is required for a measured run")
        workload = workload_contract(args, sizes)
        sampler = MemorySampler(memory_path, args.memory_sample_ms)
        warmup: dict[str, Any] | None = None
        runs: list[dict[str, Any]] = []
        completion_failures: list[dict[str, str]] = []
        finalization_checks = {
            name: "not_run" for name in COMPLETION_CHECK_NAMES
        }

        def record_completion_failure(phase: str, exc: Exception | str) -> None:
            detail = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
            completion_failures.append(
                {"phase": phase, "detail": detail[:4096] or "unspecified failure"}
            )

        def run_finalization_check(name: str, operation: Callable[[], None]) -> None:
            try:
                operation()
            except Exception as exc:
                finalization_checks[name] = "failed"
                record_completion_failure(name, exc)
            else:
                finalization_checks[name] = "passed"

        def run_guarded(
            *,
            phase: str,
            **run_kwargs: Any,
        ) -> tuple[dict[str, Any], BenchmarkError | None]:
            if thermal_guard is None:
                row = run_once(phase=phase, **run_kwargs)
                row["host_thermal"] = None
                return row, None

            thermal_guard.set_phase(phase)
            phase_started = time.perf_counter()
            thermal_guard.sample_now()
            pre_run_settled = thermal_guard.wait_for_pacing_settlement(
                thermal_settlement_timeout
            )
            thermal_guard.sample_now()
            if not pre_run_settled:
                raise BenchmarkError(
                    thermal_guard.trip_reason
                    or f"host thermal pacing failed before {phase}"
                )
            row = run_once(phase=phase, **run_kwargs)
            thermal_guard.sample_now()
            post_run_settled = thermal_guard.wait_for_pacing_settlement(
                thermal_settlement_timeout
            )
            thermal_guard.sample_now()
            phase_wall_seconds = time.perf_counter() - phase_started
            phase_metrics = thermal_guard.phase_metric_values(phase_started)
            phase_metrics.update(
                {
                    "phase": phase,
                    "phase_wall_seconds": phase_wall_seconds,
                    "thermally_sustainable_output_token_throughput_per_s": (
                        row["completion_tokens"] / phase_wall_seconds
                    ),
                }
            )
            row["host_thermal"] = phase_metrics
            if not post_run_settled or thermal_guard.trip_reason is not None:
                return row, BenchmarkError(
                    thermal_guard.trip_reason
                    or f"host thermal pacing failed after {phase}"
                )
            return row, None

        if thermal_startup_error is not None:
            record_completion_failure("host_thermal_startup", thermal_startup_error)
        else:
            sampler.start()
            try:
                try:
                    if args.warmup_requests:
                        warmup, thermal_error = run_guarded(
                            args=args,
                            concurrency=args.warmup_requests,
                            repeat=-1,
                            max_tokens=min(16, args.max_tokens),
                            phase=f"warmup-c{args.warmup_requests:03d}",
                            headers=headers,
                            sampler=sampler,
                            diagnostics_url=diagnostics_url,
                        )
                        print(
                            f"[warmup] {warmup['verdict']} "
                            f"ok={warmup['success_count']}/{warmup['request_count']}"
                        )
                        if thermal_error is not None:
                            raise thermal_error

                    if warmup is None or warmup["verdict"] == "passed":
                        for concurrency in sizes:
                            for repeat in range(args.repeats):
                                row, thermal_error = run_guarded(
                                    args=args,
                                    concurrency=concurrency,
                                    repeat=repeat,
                                    max_tokens=args.max_tokens,
                                    phase=f"measure-c{concurrency:03d}-r{repeat:03d}",
                                    headers=headers,
                                    sampler=sampler,
                                    diagnostics_url=diagnostics_url,
                                )
                                runs.append(row)
                                print_run(row)
                                if thermal_error is not None:
                                    raise thermal_error
                    elif warmup is not None:
                        record_completion_failure("warmup", "warmup verdict failed")
                except Exception as exc:
                    record_completion_failure("measurement", exc)
            finally:
                try:
                    sampler.stop()
                except Exception as exc:
                    record_completion_failure("memory_sampler_stop", exc)

        run_finalization_check(
            "repository_unchanged",
            lambda: require_repository_unchanged(repo),
        )

        def verify_model_identity() -> None:
            try:
                model_after = bind_model_identity(
                    fingerprint_model(args.model_path, args.model)
                )
            except ModelFingerprintError as exc:
                raise BenchmarkError(f"model fingerprint recheck failed: {exc}") from exc
            if model_after != model_identity:
                raise BenchmarkError("model identity changed during measurement")

        run_finalization_check("model_identity_unchanged", verify_model_identity)

        def verify_runtime_artifact() -> None:
            if fingerprint_runtime_artifact(args.runtime_artifact) != runtime_artifact:
                raise BenchmarkError("runtime artifact changed during measurement")

        run_finalization_check("runtime_artifact_unchanged", verify_runtime_artifact)
        if args.engine == "vllm":
            def verify_runtime_manifest() -> None:
                if load_vllm_runtime_manifest(args.runtime_artifact) != runtime_manifest:
                    raise BenchmarkError("vLLM runtime manifest changed during measurement")

            run_finalization_check("runtime_manifest_unchanged", verify_runtime_manifest)
        else:
            finalization_checks["runtime_manifest_unchanged"] = "not_applicable"

        if args.engine == "kiln":
            def verify_execution_identity() -> None:
                health_after = fetch_json(
                    f"{args.base_url}/health", headers, args.timeout_secs
                )
                if health_after.get("execution_identity") != runtime_execution_identity:
                    raise BenchmarkError("Kiln execution identity changed during measurement")

            run_finalization_check("execution_identity_unchanged", verify_execution_identity)
        else:
            finalization_checks["execution_identity_unchanged"] = "not_applicable"

        if thermal_guard is not None:
            assert attached_process is not None
            assert thermal_policy_record is not None

            def verify_host_thermal_handoff() -> None:
                thermal_guard.set_phase("safe-handoff")
                thermal_guard.close()
                metrics = thermal_guard.metric_values()
                pacing = thermal_guard.pacing_metric_values()
                if thermal_guard.trip_reason is not None:
                    raise BenchmarkError(thermal_guard.trip_reason)
                if thermal_guard.errors:
                    raise BenchmarkError(
                        "host thermal guard errors: " + "; ".join(thermal_guard.errors)
                    )
                if attached_process.poll() is not None:
                    raise BenchmarkError(
                        "protected server process group exited before safe handoff"
                    )
                if metrics["host_thermal_cooldown_completed_count"] != 1:
                    raise BenchmarkError("host thermal safe handoff did not complete")
                if metrics["host_thermal_cooldown_timeout_count"] != 0:
                    raise BenchmarkError("host thermal safe handoff timed out")
                if pacing["host_thermal_pacing_active_end"] != 0:
                    raise BenchmarkError("host thermal pacing remained active at handoff")
                if (
                    pacing["host_thermal_pacing_completed_event_count"]
                    != pacing["host_thermal_pacing_event_count"]
                ):
                    raise BenchmarkError(
                        "host thermal pacing events did not all complete"
                    )

            run_finalization_check(
                "host_thermal_handoff", verify_host_thermal_handoff
            )
            host_thermal_record = {
                "mode": "attached_process_group",
                "unsafe_no_guard_acknowledged": False,
                "policy": thermal_policy_record,
                "process_group": attached_process.receipt_identity(),
                "evidence": {
                    **thermal_guard.metric_values(),
                    **thermal_guard.pacing_metric_values(),
                    "host_temperature_sample_count": len(thermal_guard.samples),
                    "sensor_path": (
                        str(thermal_guard.input_path)
                        if thermal_guard.input_path is not None
                        else None
                    ),
                    "trip_reason": thermal_guard.trip_reason,
                    "errors": list(thermal_guard.errors),
                    "process_alive_at_handoff": attached_process.poll() is None,
                },
            }
        else:
            finalization_checks["host_thermal_handoff"] = "not_applicable"
            host_thermal_record = {
                "mode": "not_configured",
                "unsafe_no_guard_acknowledged": True,
                "policy": None,
                "process_group": None,
                "evidence": None,
            }

        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "driver_version": DRIVER_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "engine": {
                "name": args.engine,
                "runtime_identity": args.runtime_identity,
                "reported_version": health_version,
                "base_url": args.base_url,
                "model": args.model,
                "available_models": models,
                "authentication_configured": bool(args.api_key),
                "authentication_source": args.api_key_source,
                "model_identity": model_identity,
                "runtime_artifact": runtime_artifact,
                "runtime_execution_identity": runtime_execution_identity,
                "runtime_manifest": runtime_manifest,
            },
            "driver_environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "repository": repo,
            },
            "workload": workload,
            "workload_fingerprint": canonical_sha256(workload),
            "memory_sampler": {
                "source": "drm_vram_used" if memory_path is not None else "unavailable",
                "path": str(memory_path) if memory_path is not None else None,
                "interval_ms": args.memory_sample_ms if memory_path is not None else None,
            },
            "diagnostics": {
                "url": diagnostics_url,
                "timed_request_path_affected": False,
            },
            "host_thermal": host_thermal_record,
            "warmup": warmup,
            "runs": runs,
            "completion": {
                "expected_run_count": len(sizes) * args.repeats,
                "completed_run_count": len(runs),
                "failures": completion_failures,
                "finalization_checks": finalization_checks,
            },
        }
        if (
            args.reference_receipt is not None
            and not completion_failures
            and host_thermal_record["mode"] == "attached_process_group"
            and len(runs) == len(sizes) * args.repeats
        ):
            try:
                receipt["comparison"] = compare_reference(
                    receipt, args.reference_receipt
                )
            except Exception as exc:
                record_completion_failure("reference_comparison", exc)
        passed = (
            not repo["dirty"]
            and not completion_failures
            and host_thermal_record["mode"] == "attached_process_group"
            and all(
                status in {"passed", "not_applicable"}
                for status in finalization_checks.values()
            )
            and (warmup is None or warmup["verdict"] == "passed")
            and len(runs) == len(sizes) * args.repeats
            and all(row["verdict"] == "passed" for row in runs)
            and receipt.get("comparison", {}).get("matched", True)
        )
        receipt["verdict"] = "passed" if passed else "failed"
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        if args.out is not None:
            atomic_write_json(args.out, receipt)
            print(f"wrote {args.out}")
        else:
            print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if passed else 2
    except BenchmarkError as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2
    finally:
        if thermal_guard is not None:
            thermal_guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
